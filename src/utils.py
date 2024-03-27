import os
import sys
from collections import OrderedDict

import numpy as np
import torch
from accelerate import Accelerator
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from torch import nn


def load_model_dict(download_path, save_path=None, check_hash=True) -> OrderedDict:
    if download_path.startswith("http"):
        state_dict = torch.hub.load_state_dict_from_url(
            download_path,
            model_dir=save_path,
            check_hash=check_hash,
            map_location=torch.device("cpu"),
        )
    else:
        state_dict = torch.load(download_path, map_location=torch.device("cpu"))
    return state_dict


def resume_train_state(
    model,
    path: str,
    train_loader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
):
    try:
        # Get the most recent checkpoint
        # base_path = os.getcwd() + "/" + "model_store" + "/" + path
        base_path = "/data/jionkim/Slim_UNETR/model_store/" + path
        dirs = [base_path + "/" + f.name for f in os.scandir(base_path) if f.is_dir()]
        dirs.sort(
            key=os.path.getctime
        )  # Sorts folders by date modified, most recent checkpoint is the last
        accelerator.print(f"try to load {dirs[-1]} train stage")
        model = load_pretrain_model(dirs[-1] + "/pytorch_model.bin", model, accelerator)
        training_difference = os.path.splitext(dirs[-1])[0]
        starting_epoch = int(training_difference.replace(f"{base_path}/epoch_", "")) + 1
        step = starting_epoch * len(train_loader)
        accelerator.print(
            f"Load state training success ！Start from {starting_epoch} epoch"
        )
        return model, starting_epoch, step, step
    except Exception as e:
        accelerator.print(e)
        accelerator.print(f"Load training status failed ！")
        # return model, 0, 0, 0
        return model, 820, 820 * len(train_loader), 820 * len(train_loader)


def load_pretrain_model(pretrain_path: str, model: nn.Module, accelerator: Accelerator):
    try:
        state_dict = load_model_dict(pretrain_path)
        model.load_state_dict(state_dict)
        accelerator.print(f"Successfully loaded the training model！")
        return model
    except Exception as e:
        accelerator.print(e)
        accelerator.print(f"Failed to load the training model！")
        return model


def same_seeds(seed):
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class Logger(object):
    def __init__(self, logdir: str):
        self.console = sys.stdout
        if logdir is not None:
            os.makedirs(logdir)
            self.log_file = open(logdir + "/log.txt", "w")
        else:
            self.log_file = None
        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.log_file is not None:
            self.log_file.write(msg)

    def flush(self):
        self.console.flush()
        if self.log_file is not None:
            self.log_file.flush()
            os.fsync(self.log_file.fileno())

    def close(self):
        self.console.close()
        if self.log_file is not None:
            self.log_file.close()

def restorePatch(patches:torch.TensorType,grid_size:tuple):
    p = 1
    for val in grid_size:
        p *= val
    assert len(patches) == p
    
    length = len(patches)
    for d in range(len(grid_size),0,-1):
        temp = []
        length //= 2
        for i in range(length):
            temp.append(torch.cat((patches[i*2],patches[i*2+1]),dim=d+1))
        patches = torch.stack(temp,dim=0)
        
    return patches[0]

def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """
        Computes Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1 (torch.Tensor): Predicted bounding boxes, shape (N, 4), where N is the number of boxes.
            box2 (torch.Tensor): Ground truth bounding boxes, shape (N, 4).

        Returns:
            torch.Tensor: IoU scores per bounding box, shape (N,).
        """
        # Calculate intersection
        #print(box1,box2)
        
        xA = torch.max(box1[:, 0], box2[:, 0])
        yA = torch.max(box1[:, 1], box2[:, 1])
        zA = torch.max(box1[:, 2], box2[:, 2])
        xB = torch.min(box1[:, 3], box2[:, 3])
        yB = torch.min(box1[:, 4], box2[:, 4])
        zB = torch.min(box1[:, 5], box2[:, 5])

        interArea = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0) * torch.clamp(zB - zA, min=0)

        # Calculate union
        box1Area = (box1[:, 3] - box1[:, 0]) * (box1[:, 4] - box1[:, 1]) * (box1[:, 5] - box1[:, 2])
        box2Area = (box2[:, 3] - box2[:, 0]) * (box2[:, 4] - box2[:, 1]) * (box2[:, 5] - box2[:, 2])
        unionArea = box1Area + box2Area - interArea

        # Compute IoU
        iou = interArea / unionArea
        
        return iou

def calculate_bounding_box(label_volumes):
    # label_volumes: 입력 3D 볼륨 레이블 텐서 (shape: [batch_size, channel, depth, height, width])
    #print(label_volumes.shape)
    
    batch_boxes = []

    for batch_idx in range(label_volumes.shape[0]):
        bounding_boxes = []

        for label_volume in label_volumes[batch_idx][:]:
            # 각 배치 내의 레이블에 대해 바운딩 박스 계산

            # 각 레이블에 대한 바운딩 박스를 저장할 리스트
            bounding_boxes_batch = []

            # 각 레이블에 대해 바운딩 박스 계산
            for label in torch.unique(label_volume):
                if label == 0:  # 배경 클래스는 무시
                    continue

                # 레이블에 해당하는 마스크 생성
                mask = (label_volume == label)

                # 마스크의 True 값을 가지는 인덱스 찾기
                indices = torch.nonzero(mask, as_tuple=True)

                # 바운딩 박스 계산
                
                bounding_box = torch.tensor([torch.min(indices[0]), torch.min(indices[1]), torch.min(indices[2]),torch.max(indices[0]), torch.max(indices[1]), torch.max(indices[2])],device='cuda') # shape:[6]

                bounding_boxes_batch.append(bounding_box)

            bounding_boxes.append(torch.stack(bounding_boxes_batch,dim=0).squeeze(0))

        batch_boxes.append(torch.stack(bounding_boxes,dim=0).squeeze())
        
    return torch.stack(batch_boxes,dim=0)