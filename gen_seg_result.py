import os
import sys
from datetime import datetime

import monai
import torch
import yaml
import time
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
import numpy as np
import nibabel as nib
import SimpleITK as sitk
#import nrrd
from torch.profiler import profile, record_function, ProfilerActivity

from src import utils
from src.loader import get_dataloader_sap
from src.SlimUNETR.SlimUNETR import SlimUNETR
from src.utils import Logger, load_pretrain_model

from skimage.transform import resize

best_acc = 0
best_class = []

@torch.no_grad()
def gen_seg_result(
    model: torch.nn.Module,
    # config: EasyDict,
    inference: monai.inferers.Inferer,
    val_loader: torch.utils.data.DataLoader,
    # metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    step: int,
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    upsample: bool
):
    start = time.time()
    # inference
    model.eval()
    
    savePath = os.path.join(config.work_dir,'generation',config.finetune.checkpoint)
    os.makedirs(savePath,exist_ok=True)
    
    valid_extensions = ['.npy','.nii.gz', '.nrrd']
    #file_extensions = ['.nii.gz', '.nrrd'] # config로 변경
    file_extensions = ['.nrrd']
    transpose = False #config로 변경

    for i, image_batch in enumerate(val_loader):
        image_sample = image_batch["image"]
        
        #logits = inference(image_sample, model)
        seg = []
        for j in range(len(image_sample[0])):
            seg.append(inference(image_sample[:,j],model)) 

        logits = utils.restorePatch(torch.stack(seg,dim=0),(2,2,2))
        
        val_outputs = [post_trans(j) for j in logits]
        val_outputs_ups = []

        if upsample:
            for val_output in val_outputs:
                lab_res_stores = torch.zeros(val_output.shape[0], 256, 256, 256)

                for j in range(0, val_output.shape[0]):

                    vo_np = val_output[j, :, :, :].detach().cpu().numpy()
                    lab_reshape = resize(vo_np, (256, 256, 256),
                                           mode='edge',
                                           anti_aliasing=False,
                                           anti_aliasing_sigma=None,
                                           preserve_range=True,
                                           order=0)

                    lab_res_store = torch.zeros([256, 256, 256])
                    lab_res_store[lab_reshape > 1e-5] = 1
                    lab_res_stores[j] = lab_res_store

                lab_res_stores = lab_res_stores.cuda()
                val_outputs_ups.append(lab_res_stores)

        img = utils.restorePatch(image_sample.transpose(0,1),(2,2,2))[0, :, :, :].squeeze(dim=0).cpu().detach().numpy()
        #img = image_sample[0][0, :, :, :].cpu().detach().numpy()
        
        if upsample:
            img = resize(img, (256, 256, 256))
            lab_gen = val_outputs_ups[0].cpu().detach().numpy().astype(bool)
        else:
            lab_gen = val_outputs[0].cpu().detach().numpy().astype(bool)
            lab1_gen = val_outputs[0][0, :, :, :].cpu().detach().numpy().astype(bool)
            #lab2_gen = val_outputs[0][1, :, :, :].cpu().detach().numpy().astype(bool)
            

        lab_gt = image_batch["label"][0].cpu().detach().numpy().astype(bool)
        lab1_gt = image_batch["label"][0][0, :, :, :].cpu().detach().numpy().astype(bool)
        #lab2_gt = image_batch["label"][0][1, :, :, :].cpu().detach().numpy().astype(bool)
        
        #if transpose:
            #img = img.transpose()
        
        for file_extension in file_extensions:
            if file_extension not in valid_extensions:
                print(f'{file_extension}은 유효한 확장자가 아닙니다.')
                print(f'img shape: {img.shape}')
                print(f'lab_gen shape: {lab_gen.shape}')
                print(f'lab_gt shape: {lab_gen.shape}')
                
            elif file_extension == '.npy':
                # npy
                os.makedirs(os.path.join(savePath,'npy'),exist_ok=True)
                np.save(os.path.join(savePath,'npy',f'inf_img_{i:02}.npy'),img)
                np.save(os.path.join(savePath,'npy',f'inf_lab_gen_{i:02}.npy'),lab_gen)
                np.save(os.path.join(savePath,'npy',f'inf_lab_gt_{i:02}.npy'),lab_gt)
            elif file_extension == '.nii.gz':
                #nii.gz
                os.makedirs(os.path.join(savePath,'nii.gz'),exist_ok=True)
            elif file_extension == '.nrrd':
                #nrrd
                os.makedirs(os.path.join(savePath,'nrrd'),exist_ok=True)
                sitk.WriteImage(sitk.GetImageFromArray(img),os.path.join(savePath,'nrrd',f'inf_img_{i:02}.nrrd'))
                sitk.WriteImage(sitk.GetImageFromArray(lab1_gen.astype(int)),os.path.join(savePath,'nrrd',f'inf_lab1_gen_{i:02}.nrrd'))
                #sitk.WriteImage(sitk.GetImageFromArray(lab2_gen.astype(int)), os.path.join(savePath,'nrrd',f'inf_lab2_gen_{i:02}.nrrd'))
                sitk.WriteImage(sitk.GetImageFromArray(lab1_gt.astype(int)), os.path.join(savePath,'nrrd',f'inf_lab1_gt_{i:02}.nrrd'))
                #sitk.WriteImage(sitk.GetImageFromArray(lab2_gt.astype(int)), os.path.join(savePath,'nrrd',f'inf_lab2_gt_{i:02}.nrrd'))
                
                #sitk.WriteImage(sitk.GetImageFromArray(lab_gen.astype(int)),os.path.join(savePath,'nrrd',f'inf_lab_gen_{i:02}.nrrd'))
                #sitk.WriteImage(sitk.GetImageFromArray(lab_gt.astype(int)),os.path.join(savePath,'nrrd',f'inf_lab_gt_{i:02}.nrrd'))
                

        accelerator.print(f"[{i + 1}/{len(val_loader)}] Validation Loading", flush=True)
        step += 1

    accelerator.print(f"Generation Over!")
    done = time.time()
    elapsed = done - start
    accelerator.print(f'time: {elapsed}s')

if __name__ == "__main__":

    print(torch.cuda.is_available())

    start = time.time()

    device_num = 0
    torch.cuda.set_device(device_num)

    # load yml
    config = EasyDict(
        yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )
    
    config.trainer.batch_size = 1
    config.trainer.num_workers = 0

    utils.same_seeds(50)
    # logging_dir = os.getcwd() + "/logs/" + str(datetime.now())
    accelerator = Accelerator(cpu=False)
    # Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))

    accelerator.print("load model...")

    model = SlimUNETR(**config.finetune.slim_unetr)
    image_size = config.trainer.image_size

    accelerator.print("load dataset...")
    train_loader, val_loader = get_dataloader_sap(config)
    # val_loader = get_dataloader_val_only(config)

    inference = monai.inferers.SlidingWindowInferer(
        roi_size=ensure_tuple_rep(image_size//2, 3),
        #roi_size=ensure_tuple_rep(image_size, 3),
        overlap=0.5,
        sw_device=accelerator.device,
        device=accelerator.device,
    )
    # inference = monai.inferers.SimpleInferer()

    post_trans = monai.transforms.Compose(
        [
            monai.transforms.Activations(sigmoid=True),
            monai.transforms.AsDiscrete(threshold=0.5),
        ]
    )

    step = 0
    best_epoch = -1
    val_step = 0

    # load pre-train model
    model = load_pretrain_model(
        os.path.join(f'{config.work_dir}',f'{config.generation.pathseg}','model_store',f'{config.finetune.checkpoint}',f'{("epoch_"+f"{config.generation.epoch:05d}") if isinstance(config.generation.epoch, int) else "best"}','pytorch_model.bin'),
        model,
        accelerator,
    )

    model, val_loader = accelerator.prepare(
        model, val_loader
    )

    # start inference
    accelerator.print("Start Val！")
    accelerator.print(torch.cuda.memory_allocated() / 1024 / 1024,' MiB')
    done = time.time()
    elapsed = done - start
    accelerator.print("load: ",elapsed)
    #with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], record_shapes=True) as prof:
    gen_seg_result(
        model,
        # config,
        inference,
        val_loader,
        # metrics,
        val_step,
        post_trans,
        accelerator,
        False
    )
    #accelerator.print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))  # CPU 시간 기준으로 상위 연산 출력
    done = time.time()
    elapsed = done - start
    accelerator.print(elapsed)
    accelerator.print(torch.cuda.max_memory_allocated() / 1024 / 1024,' MiB')

    sys.exit(0)