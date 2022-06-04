# %% import dependencies
import glob
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"

import torch
import torch.nn as nn
from time import time

import monai
from monai.data import decollate_batch
from monai.transforms import (
    Compose,
    AsDiscrete,
    EnsureType,
)
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from data import get_transforms, InfiniteSampler
from ramps import get_current_consistency_weight, update_ema_variables


# %% dataset - labeled data and unlabeled data
labeled_data_folder = "./dataset/LabeledCase/" ############## need to change to your labeled cases folder ##############
labeled_images = sorted(glob.glob(os.path.join(labeled_data_folder, "images", "*.nii.gz")))
labeled_labels = sorted(glob.glob(os.path.join(labeled_data_folder, "labels", "*.nii.gz")))
print(f"training: total labeled image/label: ({len(labeled_images)}) from folder: {labeled_data_folder}")

unlabeled_data_folder = "./dataset/unlabeled/" ############## need to change to your unlabeled cases folder ##############
unlabeled_images = sorted(glob.glob(os.path.join(unlabeled_data_folder, "*.nii.gz")))[:501]
print(f"training: total unlabeled image: ({len(unlabeled_images)}) from folder: {unlabeled_data_folder}")
# %% dataloader - also different for labeled and unlabeled cases
keys = ("image", "label")
train_frac, val_frac = 0.9, 0.1
n_train = int(train_frac * len(labeled_images)) + 1
n_val = min(len(labeled_images) - n_train, int(val_frac * len(labeled_images)))
print(f"split: labeled train {n_train} val {n_val}, folder: {labeled_data_folder}")

train_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(labeled_images[:n_train], labeled_labels[:n_train])]
val_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(labeled_images[-n_val:], labeled_labels[-n_val:])]
unlabeled_train_files = [{keys[0]: img} for img in unlabeled_images]

# create a training data loader for labeled data
batch_size = 4
print(f"labeled batch size {batch_size}")
labeled_train_transforms = get_transforms("train", keys)
train_ds = monai.data.CacheDataset(data=train_files, transform=labeled_train_transforms)
labeled_sampler = InfiniteSampler(list_indices=list(range(len(train_files))), batch_size=batch_size)
train_loader = monai.data.DataLoader(
    train_ds,
    num_workers=2,
    pin_memory=False,#torch.cuda.is_available(),
    batch_sampler=labeled_sampler
)

# create a training data loader for unlabeled data
print(f"unlabeled batch size {batch_size}")
unlabeled_train_transforms = get_transforms("unlabeled", keys=("image",))
unlabeled_train_ds = monai.data.CacheDataset(data=unlabeled_train_files, transform=unlabeled_train_transforms)
unlabeled_train_loader = monai.data.DataLoader(
    unlabeled_train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=False#torch.cuda.is_available(),
)

# create a validation data loader
val_transforms = get_transforms("val", keys)
val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms)
val_loader = monai.data.DataLoader(
    val_ds,
    batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
    num_workers=2,
    pin_memory=False#torch.cuda.is_available(),
)

# %% model and loss
num_classes = 2 ############## need to change to num_classes ##############
model_folder = "./model/" ############## need to change to model save folder ##############
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# student model
model = monai.networks.nets.BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        features=(16, 32, 64, 128, 128, 32),
        dropout=0.1,
    )

# teacher model
ema_model = monai.networks.nets.BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        features=(16, 32, 64, 128, 128, 32),
        dropout=0.1,
    )
for param in ema_model.parameters():
    param.detach_()

# A test DP training strategy
model = nn.DataParallel(model) 
ema_model = nn.DataParallel(ema_model)
model.to(device)
ema_model.to(device)

lr= 1e-4
opt = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
dice_metric = DiceMetric(include_background=False, reduction="mean")

patch_size = (128, 128, 16)
sw_batch_size, overlap = 4, 0.5
# %% train
max_epochs = 1000
MeanTeacherEpoch = 50
val_interval = 1
best_metric = -1
best_metric_epoch = -1
iter_num = 0
epoch_loss_values = []
metric_values = []
post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=num_classes)])
post_label = Compose([EnsureType(), AsDiscrete(to_onehot=num_classes)])

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    start_time = time()
    model.train()
    epoch_loss = 0
    step = 0

    for labeled_batch, unlabeled_batch in zip(train_loader, unlabeled_train_loader):
        step += 1
        labeled_inputs, labels = (
            labeled_batch["image"].to(device),
            labeled_batch["label"].to(device),
        )
        unlabeled_inputs = unlabeled_batch["image"].to(device)

        opt.zero_grad()

        noise_labeled = torch.clamp(torch.randn_like(
                labeled_inputs) * 0.1, -0.2, 0.2)
        noise_unlabeled = torch.clamp(torch.randn_like(
                unlabeled_inputs) * 0.1, -0.2, 0.2)
        noise_labeled_inputs = labeled_inputs + noise_labeled
        noise_unlabeled_inputs = unlabeled_inputs + noise_unlabeled

        outputs = model(labeled_inputs)
        with torch.no_grad():
            soft_out = torch.softmax(outputs, dim=1)
            outputs_unlabeled = model(unlabeled_inputs)
            soft_unlabeled = torch.softmax(outputs_unlabeled, dim=1)
            outputs_aug = ema_model(noise_labeled_inputs)
            soft_aug = torch.softmax(outputs_aug, dim=1)
            outputs_unlabeled_aug = ema_model(noise_unlabeled_inputs)
            soft_unlabeled_aug = torch.softmax(outputs_unlabeled_aug, dim=1)

        supervised_loss = loss_function(outputs, labels)
        if epoch < MeanTeacherEpoch:
                consistency_loss = 0.0
        else:
            consistency_loss = torch.mean(
                (soft_out - soft_aug) ** 2) + \
                               torch.mean(
                (soft_unlabeled - soft_unlabeled_aug) ** 2)
        consistency_weight = get_current_consistency_weight(iter_num//150)
        iter_num += 1

        loss = supervised_loss + consistency_weight * consistency_loss
        loss.backward()
        opt.step()
        update_ema_variables(model, ema_model, 0.99, iter_num)

        epoch_loss += loss.item()
        print(
            f"{step}/{len(unlabeled_train_ds) // unlabeled_train_loader.batch_size}, "
            f"train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                val_outputs = sliding_window_inference(val_inputs, patch_size, sw_batch_size, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)

            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            print(f"val dice: {metric}")
            # reset the status for next validation round
            dice_metric.reset()

            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.module.state_dict(), os.path.join(
                    model_folder, "best_metric_model.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )
        print(f"epoch time = {time() - start_time}")
        


