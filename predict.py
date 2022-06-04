import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"

import numpy as np
from glob import glob
import torch
from torch import nn
import argparse
import time

import shutil

import monai
from monai.transforms import (
    Compose,
    AddChanneld,
    AsDiscrete,
    CastToTyped,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    EnsureTyped,
    EnsureType,
)


# %% transforms: different for train, validation and inference
def get_transforms(mode="train", keys=("image", "label")):
    xforms = [
        LoadImaged(keys),
        AddChanneld(keys),
        Orientationd(keys, axcodes="LPI"),
        Spacingd(keys, pixdim=(4, 4, 10), mode=("bilinear", "nearest")[: len(keys)]),
        ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
    ]
    if mode == "train":
        xforms.extend(
            [
                SpatialPadd(keys, spatial_size=(128, 128, -1), mode="reflect"),  # ensure at least 192x192
                RandAffined(
                    keys,
                    prob=0.15,
                    rotate_range=(0.05, 0.05, None),  # 3 parameters control the transform on 3 dimensions
                    scale_range=(0.1, 0.1, None),
                    mode=("bilinear", "nearest"),
                    as_tensor_output=False,
                ),
                RandCropByPosNegLabeld(keys, label_key=keys[1], spatial_size=(128, 128, 16), num_samples=3),
                RandGaussianNoised(keys[0], prob=0.15, std=0.01),
                RandFlipd(keys, spatial_axis=0, prob=0.5),
                RandFlipd(keys, spatial_axis=1, prob=0.5),
                RandFlipd(keys, spatial_axis=2, prob=0.5),
            ]
        )
        dtype = (np.float32, np.uint8)
    if mode == "val":
        dtype = (np.float32, np.uint8)
    if mode == "infer":
        dtype = (np.float32,)
    xforms.extend([CastToTyped(keys, dtype=dtype), EnsureTyped(keys)])
    return Compose(xforms)


# %% model
st = time.time()
num_classes = 2
model_folder = "./model/"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = monai.networks.nets.BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        features=(16, 32, 64, 128, 128, 32),
        dropout=0.1,
    )

# model = nn.DataParallel(model)
model.to(device)

# %% inference
parser = argparse.ArgumentParser(description='Input/output folder')
parser.add_argument('-i', '--input_folder', required=True, type=str, help='input images folder')
parser.add_argument('-o', '--output_folder', required=True, type=str, help='output images folder')
args = parser.parse_args()
ingput_folder = args.input_folder
output_folder = args.output_folder

patch_size = (128, 128, 16)
sw_batch_size, overlap = 1, 0.5
keys = ("img",)
infer_transforms = get_transforms(mode="infer", keys=keys)

img_path = glob(os.path.join(ingput_folder, "*.nii.gz"))
infer_files = [{"img": img} for img in img_path]

infer_ds = monai.data.Dataset(data=infer_files, transform=infer_transforms)
infer_loader = monai.data.DataLoader(
    infer_ds,
    batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

inferer = monai.inferers.SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
        padding_mode="replicate",
    )
saver = monai.data.NiftiSaver(output_dir=output_folder, mode="nearest", separate_folder=False, output_dtype=np.uint8)

model.load_state_dict(torch.load(
    "./model/best_metric_model.pth"))
model.eval()

with torch.no_grad():
    for infer_data in infer_loader:
        try:
            print(f"segmenting {infer_data['img_meta_dict']['filename_or_obj']}")
            preds = inferer(infer_data[keys[0]].to(device), model)
            preds = (preds.argmax(dim=1, keepdims=True)).type(torch.uint8)
            saver.save_batch(preds, infer_data["img_meta_dict"])
            torch.cuda.empty_cache()
        except:
            shutil.copy(f"{infer_data['img_meta_dict']['filename_or_obj'][0]}", "./special_cases")

for i in os.listdir(output_folder):
    new_name = i[:12] + i[-7:]
    os.rename(os.path.join(output_folder, i), os.path.join(output_folder, new_name))

et = time.time()
print(f"total infer time: {et - st}")