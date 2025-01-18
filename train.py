from model import SeResNext50_Unet_Loc
import argparse
import rasterio
import torch
from adamw import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from utils import AverageMeter, TrainDataset, ValDataset, calculate_loss, calculate_metrics
import os
from torch.utils.data import DataLoader
import random
import numpy as np


def train_epoch(model, train_data_loader, optimizer, epoch, loss_function):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    losses = AverageMeter()
    scaler = GradScaler()

    iterator = tqdm(train_data_loader)
    model.train()

    for i, batch in enumerate(iterator):
        imgs = batch["img"].cuda(non_blocking=True)
        msks = batch["msk"].cuda(non_blocking=True)
        

        with autocast(device_type=device):
            out = model(imgs)
            loss_dict = calculate_loss(out, msks)
            loss = loss_dict[loss_function]

        losses.update(loss.item(), imgs.size(0))
        iterator.set_description(f"Epoch: {epoch+1}, Loss({loss_function}): {losses.avg:.4f}")

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

def val_epoch(model, val_data_loader, epoch, loss_function, all_loss_functions, all_metrics):
    model = model.eval()
    losses = {loss: AverageMeter() for loss in all_loss_functions}
    metrics = {metric: AverageMeter() for metric in all_metrics}

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_data_loader)):
            msks = batch["msk"].cuda(non_blocking=True)
            imgs = batch["img"].cuda(non_blocking=True)
            
            out = model(imgs)
            loss_dict = calculate_loss(out, msks)
            for loss in all_loss_functions:
                losses[loss].update(loss_dict[loss].item(), imgs.size(0))
            metric_dict = calculate_metrics(out, msks)
            for metric in all_metrics:
                metrics[metric].update(metric_dict[metric].item(), imgs.size(0))

    print(f"Val Losses: {', '.join([f'{loss}: {losses[loss].avg:.4f}' for loss in all_loss_functions])}")
    print(f"Val Metrics: {', '.join([f'{metric}: {metrics[metric].avg:.4f}' for metric in all_metrics])}")
    if best_loss is None:
        best_loss = losses[loss_function].avg
    else:
        if losses[loss_function].avg < best_loss:
            best_loss = losses[loss_function].avg
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
            }, os.path.join(models_folder, snapshot_name + '_best'))


def evaluate(model, test_data_loader, all_loss_functions, all_metrics):
    model = model.eval()
    losses = {loss: AverageMeter() for loss in all_loss_functions}
    metrics = {metric: AverageMeter() for metric in all_metrics}

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_data_loader)):
            msks = batch["msk"].cuda(non_blocking=True)
            imgs = batch["img"].cuda(non_blocking=True)
            
            out = model(imgs)
            loss_dict = calculate_loss(out, msks)
            for loss in all_loss_functions:
                losses[loss].update(loss_dict[loss].item(), imgs.size(0))
            metric_dict = calculate_metrics(out, msks)
            for metric in all_metrics:
                metrics[metric].update(metric_dict[metric].item(), imgs.size(0))

    print(f"Test Losses: {', '.join([f'{loss}: {losses[loss].avg:.4f}' for loss in all_loss_functions])}")
    print(f"Test Metrics: {', '.join([f'{metric}: {metrics[metric].avg:.4f}' for metric in all_metrics])}")


def main(args):
    seed = args.seed 
    batch_size = args.batch_size
    val_batch_size = args.val_batch_size
    epochs = args.epochs
    workers = args.num_workers
    damage_types = args.damage_types

    snapshot_name = f'SeResNext50_Unet_Loc_{seed}'
    models_folder = "/kaggle/working/models"

    np.random.seed(seed)
    random.seed(seed)

    data_train = TrainDataset(args.train_data_paths, damage_types)
    val_train = ValDataset(args.val_data_paths, damage_types)
    data_test = ValDataset(args.test_data_paths, damage_types)

    train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=workers,  pin_memory=False, drop_last=True)
    val_data_loader = DataLoader(val_train, batch_size=val_batch_size, num_workers=workers, pin_memory=False)
    test_data_loader = DataLoader(data_test, batch_size=val_batch_size, num_workers=workers, pin_memory=False)

    model = SeResNext50_Unet_Loc().cuda()
    params = model.parameters()
    optimizer = AdamW(params, lr=0.00015, weight_decay=1e-6)

    best_loss = None
    all_loss_functions = ["wce_loss", "focal_loss", "dice_loss", "combo_loss", "tversky_loss"]
    all_metrics = ["iou", "f1_score", "precision", "recall"]

    torch.cuda.empty_cache()
    for epoch in range(epochs):
        train_epoch(model, train_data_loader, optimizer, epoch, args.loss)
        val_epoch(model, val_data_loader, epoch, args.loss, all_loss_functions, all_metrics)
        torch.cuda.empty_cache()
    evaluate(model, test_data_loader, all_loss_functions, all_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Segemntation Model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument('--damage-types', nargs='+', required=True)
    parser.add_argument('--train-data-paths', nargs='+', required=True)
    parser.add_argument('--val-data-paths', nargs='+', required=True)
    parser.add_argument('--test-data-paths', nargs='+', required=True)
    parser.add_argument("--loss", type=str, default="combo_loss")
    args = parser.parse_args()
    main(args)