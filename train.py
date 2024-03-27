import torch
from easydict import EasyDict
from datetime import datetime
from typing import Dict
import monai
from accelerate import Accelerator
from tqdm import tqdm

from src import utils
    
def train_detector_one_epoch(
    model: torch.nn.Module,
    config: EasyDict,
    loss_functions: Dict[str, torch.nn.modules.loss._Loss],
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    epoch: int,
    step: int,
):
    accelerator.print(f'train_start current GPU memory usage: {torch.cuda.memory_allocated()/1024/1024} Mib')
    start_time = datetime.now()
    progress_bar = tqdm(range(train_loader))
    model.train()
    
    for i, image_batch in enumerate(train_loader):
        img_input = image_batch["image"]
        label_input = image_batch["label"]
        
        logits = model(img_input)

        #Get Loss
        total_loss = 0
        for name in loss_functions:
            alpth = 1
            loss = loss_functions[name](logits, label_input)
            accelerator.log({"Train/" + name: float(loss)}, step=step)
            total_loss += alpth * loss
        
        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()
        
        #Get Matrix
        val_outputs = [post_trans(i) for i in logits]
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=label_input)

        accelerator.log(
            {
                "Train/Detector/Total Loss": float(total_loss),
            },
            step=step,
        )
        step += 1
        progress_bar.update(1)

    scheduler.step(epoch)
    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metric.update(
            {
                f"Train/mean {metric_name}": float(batch_acc.mean()),
            }
        )
    accelerator.print(
        f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Detector Training metric {metric}"
    )
    accelerator.log(metric, step=epoch)
    accelerator.print(f'train_end current GPU memory usage: {torch.cuda.memory_allocated()/1024/1024} Mib')
    accelerator.print(f'epoch runtime: {datetime.now()-start_time}')
    return val_outputs, step

def train_one_epoch_patch(
    model: torch.nn.Module,
    config: EasyDict,
    loss_functions: Dict[str, torch.nn.modules.loss._Loss],
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    epoch: int,
    step: int,
):
    
    accelerator.print(f'train_start current GPU memory usage: {torch.cuda.memory_allocated()/1024/1024} Mib')
    start_time = datetime.now()
    # train
    model.train()
    sum = 0.
    count = 0
    
        
    for i, image_batch in enumerate(train_loader):
        img_input = image_batch["image"]
        label_input = image_batch["label"]
        for p in range(img_input.shape[1]):
                
            #print("is batch cuda: ",image_batch["image"][0].is_cuda)
            seg_start = datetime.now()
            count = i
            # accelerator.print(str(i) + ' step started.')
            # accelerator.print(datetime.now())

            img_patch = img_input[:][p]
            lab_patch = label_input[:][p]
            
            #label_input = utils.restorePatch(image_batch["label"],(2,2,2))
            
            load_time = datetime.now()
            #accelerator.print(f'load time: {datetime.now() - seg_start}')
            sum += (load_time-start_time).total_seconds()
            
            logits = model(img_patch)
            
            #accelerator.print(f'model time: {datetime.now() - load_time}')
            total_loss = 0
            log = ""
            for name in loss_functions:
                alpth = 1
                loss = loss_functions[name](logits, lab_patch)
                accelerator.log({"Train/" + name: float(loss)}, step=step)
                total_loss += alpth * loss
                
            val_outputs = [post_trans(i) for i in logits]
            for metric_name in metrics:
                metrics[metric_name](y_pred=val_outputs, y=lab_patch)

            accelerator.backward(total_loss)
            optimizer.step()
            optimizer.zero_grad()
            accelerator.log(
                {
                    "Train/Total Loss": float(total_loss),
                },
                step=step,
            )
            accelerator.print(
                f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training [{(i)*(8)+p+1}/{len(train_loader)*8}] Loss: {total_loss:1.5f} {log}",
                flush=True,
            )
            step += 1
            # accelerator.print(str(i) + ' step finished.')
            # accelerator.print(datetime.now())

    scheduler.step(epoch)
    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metric.update(
            {
                f"Train/mean {metric_name}": float(batch_acc.mean()),
                f"Train/label1 {metric_name}": float(batch_acc[0]),
                #f"Train/label2 {metric_name}": float(batch_acc[1]),
            }
        )
        accelerator.print(f'seg runtime: {seg_start-start_time}')
    accelerator.print(
        f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training metric {metric}"
    )
    accelerator.log(metric, step=epoch)
    accelerator.print(f'train_end current GPU memory usage: {torch.cuda.memory_allocated()/1024/1024} Mib')
    accelerator.print(f'runtime: {datetime.now()-start_time}')
    accelerator.print(f'load time: {sum/count}')
    return step

@torch.no_grad()
def val_detector_one_epoch(
    model: torch.nn.Module,
    loss_functions: Dict[str, torch.nn.modules.loss._Loss],
    inference: monai.inferers.Inferer,
    val_loader: torch.utils.data.DataLoader,
    config: EasyDict,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    step: int,
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    epoch: int,
):
    accelerator.print(f'validation_start current GPU memory usage: {torch.cuda.memory_allocated()/1024/1024} Mib')
    start_time = datetime.now()
    progress_bar = tqdm(range(val_loader))
    model.eval()
    
    for i, image_batch in enumerate(val_loader):
        img_input = image_batch["image"]
        label_input = image_batch["label"]
        
        logits = inference(img_input,model)

        #Get Loss
        total_loss = 0
        for name in loss_functions:
            alpth = 1
            loss = loss_functions[name](logits, label_input)
            accelerator.log({"Train/" + name: float(loss)}, step=step)
            total_loss += alpth * loss
        
        #Get Matrix
        val_outputs = [post_trans(i) for i in logits]
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=label_input)

        accelerator.log(
            {
                "Train/Detector/Total Loss": float(total_loss),
            },
            step=step,
        )
        step += 1
        progress_bar.update(1)

    scheduler.step(epoch)
    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metric.update(
            {
                f"Train/mean {metric_name}": float(batch_acc.mean()),
            }
        )
    accelerator.print(
        f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Detector Training metric {metric}"
    )
    accelerator.log(metric, step=epoch)
    accelerator.print(f'train_end current GPU memory usage: {torch.cuda.memory_allocated()/1024/1024} Mib')
    accelerator.print(f'epoch runtime: {datetime.now()-start_time}')
    return val_outputs, step

@torch.no_grad()
def val_one_epoch_patch(
    model: torch.nn.Module,
    loss_functions: Dict[str, torch.nn.modules.loss._Loss],
    inference: monai.inferers.Inferer,
    val_loader: torch.utils.data.DataLoader,
    config: EasyDict,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    step: int,
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    epoch: int,
):
    accelerator.print(f'val_start current GPU memory usage: {torch.cuda.memory_allocated()/1024/1024} Mib')
    start_time = datetime.now()
    # val
    model.eval()
    for i, image_batch in enumerate(val_loader):
        seg = []
        for j in range(len(image_batch["image"][0])):
            seg.append(inference(image_batch["image"][:,j],model)) 

        logits = utils.restorePatch(torch.stack(seg,dim=0),(2,2,2))
        
        total_loss = 0
        log = ""
        for name in loss_functions:
            loss = loss_functions[name](logits, image_batch["label"])
            accelerator.log({"Val/" + name: float(loss)}, step=step)
            log += f" {name} {float(loss):1.5f} "
            total_loss += loss
        val_outputs = [post_trans(i) for i in logits]
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch["label"])
        accelerator.log(
            {
                "Val/Total Loss": float(total_loss),
            },
            step=step,
        )
        accelerator.print(
            f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Validation [{i + 1}/{len(val_loader)}] Loss: {total_loss:1.5f} {log}",
            flush=True,
        )
        step += 1

    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()
        if accelerator.num_processes > 1:
            batch_acc = (
                accelerator.reduce(batch_acc.to(accelerator.device))
                / accelerator.num_processes
            )
        metrics[metric_name].reset()
        metric.update(
            {
                f"Val/mean {metric_name}": float(batch_acc.mean()),
                f"Val/label1 {metric_name}": float(batch_acc[0]),
                #f"Val/label2 {metric_name}": float(batch_acc[1]),
            }
        )
    accelerator.print(
        f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Validation metric {metric}"
    )
    accelerator.log(metric, step=epoch)
    accelerator.print(f'val_end current GPU memory usage: {torch.cuda.memory_allocated()/1024/1024} Mib')
    accelerator.print(f'runtime: {datetime.now()-start_time}')
    return (
        torch.Tensor([metric["Val/mean dice_metric"]]).to(accelerator.device),
        batch_acc,
        step,
    )