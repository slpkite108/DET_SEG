import json
import os
from typing import Dict, Hashable, Mapping, Tuple
from accelerate import Accelerator

import monai
import numpy as np
import torch
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from torch.profiler import profile, record_function, ProfilerActivity


class ConvertToMultiChannelClassesd(monai.transforms.MapTransform):
    """
    TC WT ET
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses`.
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = [monai.utils.TransformBackends.TORCH, monai.utils.TransformBackends.NUMPY]

    def __init__(
        self,
        keys: monai.config.KeysCollection,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)

    def converter(self, img: monai.config.NdarrayOrTensor):

        result = [img == 1, img == 2]
        #result = [img == 6]

        return (
            torch.stack(result, dim=0)
            if isinstance(img, torch.Tensor)
            else np.stack(result, axis=0)
        )

    def __call__(
        self, data: Mapping[Hashable, monai.config.NdarrayOrTensor]
    ) -> Dict[Hashable, monai.config.NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


def load_dataset_images(root):
    images_list = []

    for i in range (0, 266):
        img = root + "image/image_" + str(i).zfill(3) + ".nii.gz"
        seg_img = root + "label/label_" + str(i).zfill(3) + ".nii.gz"
        images_list.append(
            {"image": img, "label": seg_img}
        )
    
    # for i in range (0, 266):
    #     img = root + "image/image_static_" + str(i).zfill(3) + ".nii.gz"
    #     seg_img = root + "label/label_static_" + str(i).zfill(3) + ".nii.gz"
    #     images_list.append(
    #         {"image": img, "label": seg_img}
    #     )
    
    '''
    for i in range(0, 40):
        img = "J:\\Dataset\\TEE-Labeling\\_TTE_images\\image\\nii_gz_128\\image_" + str(i).zfill(3) + ".nii.gz"
        seg_img = "J:\\Dataset\\TEE-Labeling\\_TTE_images\\label\\nii_gz_128\\label_" + str(i).zfill(3) + ".nii.gz"
        images_list.append(
            {"image": img, "label": seg_img}
        )

    for i in range(0, 15):
        img = root + "image/1_label_" + str(i).zfill(3) + "_nm.nii.gz"
        seg_img = root + "label/1_label_" + str(i).zfill(3) + ".nii.gz"
        images_list.append(
            {"image": img, "label": seg_img}
        )
        '''
    return images_list

# Saparate train images to load > get_dataloader_sap()
def load_dataset_train_images_sap(root):
    images_list = []

    with open(os.path.join(root,'dataset.json')) as f:
        meta = json.load(f)
    
    for elem in meta["training"]:
        for key, value in elem.items():
            elem[key] = os.path.join(root,value)
        images_list.append(elem)
    
    # for i in range (0, 221):
    #     img = root + "train/image/image_" + str(i).zfill(3) + ".nii.gz"
    #     seg_img = root + "train/label/label_" + str(i).zfill(3) + ".nii.gz"
    #     images_list.append(
    #         {"image": img, "label": seg_img}
    #     )
    

    return images_list

def load_dataset_val_images_sap(root):
    images_list = []

    with open(os.path.join(root,'dataset.json')) as f:
        meta = json.load(f)
    
    for elem in meta["validation"]:
        for key, value in elem.items():
            elem[key] = os.path.join(root,value)
        images_list.append(elem)
    # for i in range (0, 221):
    #     img = root + "train/image/image_" + str(i).zfill(3) + ".nii.gz"
    #     seg_img = root + "train/label/label_" + str(i).zfill(3) + ".nii.gz"
    #     images_list.append(
    #         {"image": img, "label": seg_img}
    #     )
    

    return images_list

# Saparate test images to load > get_dataloader_sap()
def load_dataset_test_images_sap(root):
    images_list = []

    with open(os.path.join(root,'dataset.json')) as f:
            meta = json.load(f)
        
    for elem in meta["test"]:
        for key, value in elem.items():
            elem[key] = os.path.join(root,value)
        images_list.append(elem)
    # for i in range(0, 46):
    #     img = root + "test/image/image_" + str(i).zfill(3) + ".nii.gz"
    #     seg_img = root + "test/label/label_" + str(i).zfill(3) + ".nii.gz"
    #     images_list.append(
    #         {"image": img, "label": seg_img}
    #     )

    return images_list

def threshold_for_label(x):
    return x == 6

def get_transforms(
    config: EasyDict,
    accelerator: Accelerator=None,
) -> Tuple[monai.transforms.Compose, monai.transforms.Compose]:
    train_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"], dtype=torch.float32),
            monai.transforms.EnsureChannelFirstd(keys=["image", "label"]),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            #ConvertToMultiChannelClassesd(keys=["label"]),
            LabelCompressd(keys=["label"]),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            #Printd(keys=["image", "label"]),
            #Dynamics
            #monai.transforms.CropForegroundd(keys=["image","label"], source_key="label"),
            #RectPad(keys=["image","label"]),
            # Dynamincs
            # monai.transforms.CropForegroundd(keys=["image","label"], source_key="label"),
            # DynamicSquarePadd(keys=["image", "label"]),
            # monai.transforms.Resized(keys=["image", "label"], spatial_size=(config.trainer.image_size, config.trainer.image_size,config.trainer.image_size),mode=("bilinear", "nearest"),),
            #Printd(keys=["image", "label"]),
            # monai.transforms.SpatialPadD(
            #     keys=["image", "label"],
            #     spatial_size=(255, 255, 255),
            #     method="symmetric",
            #     mode="constant",
            # ),
            # monai.transforms.Spacingd(
            #     keys=["image", "label"],
            #     pixdim=(1.0, 1.0, 1.0),
            #     mode=("bilinear", "nearest"),
            # ),
            # monai.transforms.CenterSpatialCropD(
            #     keys=["image", "label"],
            #     roi_size=ensure_tuple_rep(config.trainer.image_size, 3),
            # ),
            
            #divide patch
            #monai.transforms.GridPatchd(keys=["image","label"],patch_size=(config.trainer.image_size//4,config.trainer.image_size//4,config.trainer.image_size//4),stride=(config.trainer.image_size//4,config.trainer.image_size//4,config.trainer.image_size//4)),
            
            ################
            monai.transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                num_samples=2,
                spatial_size=ensure_tuple_rep(config.trainer.image_size, 3),
                pos=1,
                neg=1,
                image_key="image",
                image_threshold=0,
            ),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=0
            ),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=1
            ),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=2
            ),
            monai.transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True
            ),
            monai.transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            monai.transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            #MakeGrid(keys=["image","label"], input_size=(config.trainer.image_size, config.trainer.image_size,config.trainer.image_size), grid_size=(2,2,2), accelerator = accelerator),
            monai.transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            #monai.transforms.ToDeviced(keys=["image", "label"], device='cuda'),
            monai.transforms.EnsureChannelFirstd(keys=["image", "label"]),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            #ConvertToMultiChannelClassesd(keys="label"),
            LabelCompressd(keys=["label"]),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            # monai.transforms.CropForegroundd(keys=["image","label"], source_key="label"),
            # DynamicSquarePadd(keys=["image", "label"]),
            #monai.transforms.Resized(keys=["image", "label"], spatial_size=(config.trainer.image_size, config.trainer.image_size,config.trainer.image_size),mode=("bilinear", "nearest"),),
            
            #monai.transforms.GridPatchd(keys=["image","label"],patch_size=(config.trainer.image_size//4)),
            #Dynamics
            #monai.transforms.CropForegroundd(keys=["image","label"], source_key="label"),
            #RectPad(keys=["image","label"]),
            #Dynamincs
            # monai.transforms.SpatialPadD(
            #     keys=["image", "label"],
            #     spatial_size=(255,255,config.trainer.image_size),#(138, 138, 138),
            #     method="symmetric",
            #     mode="constant",
            # ),
            # monai.transforms.Resized(keys=["image", "label"], spatial_size=(config.trainer.image_size, config.trainer.image_size,config.trainer.image_size),mode=("bilinear", "nearest"),),
            # monai.transforms.Spacingd(
            #     keys=["image", "label"],
            #     pixdim=(1.0, 1.0, 1.0),
            #     mode=("bilinear", "nearest"),
            # ),
            monai.transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True
            ),
            #MakeGrid(keys=["image"], input_size=(config.trainer.image_size, config.trainer.image_size,config.trainer.image_size), grid_size=(2,2,2),accelerator=accelerator),
            monai.transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    return train_transform, val_transform


def get_dataloader(
    config: EasyDict,
    accelerator:Accelerator = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    train_images = load_dataset_images(config.data_root)

    train_transform, val_transform = get_transforms(config,accelerator=accelerator)

    train_dataset = monai.data.Dataset(
        data=train_images[: int(len(train_images) * config.trainer.train_ratio)],
        transform=train_transform,
    )
    val_dataset = monai.data.Dataset(
        data=train_images[int(len(train_images) * config.trainer.train_ratio) :],
        transform=val_transform,
    )

    train_loader = monai.data.DataLoader(
        train_dataset,
        num_workers=config.trainer.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        pin_memory=True
    )

    batch_size = config.trainer.batch_size

    val_loader = monai.data.DataLoader(
        val_dataset,
        num_workers=config.trainer.num_workers,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, val_loader


def get_dataloader_val_only(
    config: EasyDict,
    accelerator:Accelerator = None,
) -> [torch.utils.data.DataLoader]:

    train_images = load_dataset_images(config.data_root)

    _, val_transform = get_transforms(config,accelerator=accelerator)

    val_dataset = monai.data.Dataset(
        data=train_images,
        transform=val_transform,
    )

    batch_size = config.trainer.batch_size

    val_loader = monai.data.DataLoader(
        val_dataset,
        num_workers=config.trainer.num_workers,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    return val_loader

# Saparate train images and test images in different directory
def get_dataloader_sap(
    config: EasyDict,
    accelerator:Accelerator = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    train_images = load_dataset_train_images_sap(config.data_root)
    val_images = load_dataset_val_images_sap(config.data_root)

    train_transform, val_transform = get_transforms(config,accelerator=accelerator)

    train_dataset = monai.data.Dataset(
        data=train_images,
        transform=train_transform,
    )
    val_dataset = monai.data.Dataset(
        data=val_images,
        transform=val_transform,
    )

    train_loader = monai.data.DataLoader(
        train_dataset,
        num_workers=config.trainer.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=True,
    )

    batch_size = config.trainer.batch_size

    val_loader = monai.data.DataLoader(
        val_dataset,
        num_workers=config.trainer.num_workers,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader

def get_dataloader_sap_val_only(
    config: EasyDict,
    accelerator:Accelerator = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    test_images = load_dataset_test_images_sap(config.data_root)

    _, val_transform = get_transforms(config,accelerator=accelerator)

    val_dataset = monai.data.Dataset(
        data=test_images,
        transform=val_transform,
    )

    batch_size = config.trainer.batch_size

    val_loader = monai.data.DataLoader(
        val_dataset,
        num_workers=config.trainer.num_workers,
        batch_size=batch_size,
        shuffle=False,
    )

    return val_loader


class RectPad:
    def __init__(self,keys):
        self.keys = keys
    
    def __call__(self, data):
        debug = False

        max_size = max([max(data[key].shape[1:]) for key in self.keys])
        if debug: print("max_size: ",max_size,"\nshape: ",data[key].shape)
        
        for key in self.key:
            padded_image = monai.transforms.SpatialPadd(spatial_size=(max_size, max_size),method="symmetric")(data[key])
            data[key] = padded_image
        
class DynamicSquarePadd:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            image = data[key]
            # 이미지의 현재 크기를 기반으로 최대 길이 계산
            max_size = max(image.shape[1:])  # 첫 번째 차원은 채널을 나타냄
            new_size = [max_size] * len(image.shape[1:])
            # MONAI의 SpatialPad를 사용하여 이미지를 동적으로 패딩
            pad_transform = monai.transforms.SpatialPad(spatial_size=new_size, method="symmetric", mode="constant")
            data[key] = pad_transform(image)
        return data
    
class Printd:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            image = data[key]
            # 이미지의 현재 크기를 기반으로 최대 길이 계산
            print(key," : ",image.shape)
            
class LabelCompressd:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            data[key][data[key]>0] = 1
            # 이미지의 현재 크기를 기반으로 최대 길이 계산
        return data
    
class MakeGrid:
    def __init__(self,keys, input_size, grid_size, accelerator):
        self.keys = keys
        self.input_size = input_size
        self.grid_size = grid_size
        self.accelerator = accelerator
    
    def __call__(self, data):
        for key in self.keys:
            data[key] = self.calGrid(data[key], input_size=self.input_size, grid_size=self.grid_size)
            if self.accelerator != None:
                self.accelerator.print(f'dataload current GPU memory usage: {torch.cuda.memory_allocated()/1024/1024} Mib')
        return data
    
    def calGrid(self,data,input_size:tuple,grid_size:tuple):
        assert tuple(a % b for a, b in zip(input_size, grid_size)) == (0,0,0)
    
        patch_size = tuple(a // b for a, b in zip(input_size, grid_size))
        
        base = []
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                for z in range(grid_size[2]):
                    start_pos = tuple(a * b for a, b in zip((x,y,z), patch_size))
                    end_pos = tuple(a + b for a, b in zip(start_pos, patch_size))
                    base.append(data[:,start_pos[0]:end_pos[0],start_pos[1]:end_pos[1],start_pos[2]:end_pos[2]])
                    
        return torch.stack(base,dim=0)