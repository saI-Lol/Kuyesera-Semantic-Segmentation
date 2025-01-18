from model import SeResNext50_Unet_Loc
import argparse
import rasterio
import torch
from adamw import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from utils import mae_loss, AverageMeter, TrainDataset, ValDataset
import os
from torch.utils.data import DataLoader

def train_epoch(model, train_data_loader, optimizer, epoch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    losses = AverageMeter()
    scaler = GradScaler()

    iterator = tqdm(train_data_loader)
    model.train()

    for i, batch in enumerate(iterator):
        imgs = batch["img"].cuda(non_blocking=True)
        msks = batch["msk"].cuda(non_blocking=True)
        print(imgs.shape, msks.shape)

        with autocast(device_type=device):
            out = model(imgs)
            loss_dict = mae_loss(out, msks)
            loss = loss_dict['all']

        losses.update(loss.item(), imgs.size(0))
        iterator.set_description(f"Epoch: {epoch+1}, Loss: {losses.avg:.4f}")

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

def val_epoch(model, val_data_loader, epoch):
    model = model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_data_loader)):
            msks = batch["msk"].cuda(non_blocking=True)
            imgs = batch["img"].cuda(non_blocking=True)
            
            out = model(imgs)
            loss_dict = mae_loss(out, msks)
            loss = loss_dict['all']
            losses.update(loss.item(), imgs.size(0))

    print(f"Epoch: {epoch+1}, Val Mae: {losses.avg:.4f}")
    if best_mae is None:
        best_mae = losses.avg
    else:
        if losses.avg < best_mae:
            best_mae = losses.avg
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_mae': best_mae,
            }, os.path.join(models_folder, snapshot_name + '_best'))


def evaluate(model, test_data_loader, damage_types):
    model = model.eval()
    losses = AverageMeter()
    damage_losses = {damage_type: AverageMeter() for damage_type in damage_types}

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_data_loader)):
            msks = batch["msk"].cuda(non_blocking=True)
            imgs = batch["img"].cuda(non_blocking=True)
            
            out = model(imgs)
            loss_dict = mae_loss(out, msks)
            loss = loss_dict['all']
            losses.update(loss.item(), imgs.size(0))
            for damage_type in damage_types:
                damage_losses[damage_type].update(loss_dict[damage_type].item(), imgs.size(0))

    print(f"Test Mae all: {losses.avg:.4f}")
    for damage_type in damage_types:
        print(f"Test Mae {damage_type}: {damage_losses[damage_type].avg:.4f}")


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

    model = SeResNext50_Unet_Loc(num_classes=len(damage_types)).cuda()
    params = model.parameters()
    optimizer = AdamW(params, lr=0.00015, weight_decay=1e-6)

    best_mae = None

    torch.cuda.empty_cache()
    for epoch in range(epochs):
        train_epoch(model, train_data_loader, optimizer, epoch)
        val_epoch(model, val_data_loader, epoch)
        torch.cuda.empty_cache()
    evaluate(model, test_data_loader, damage_types)


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
    args = parser.parse_args()
    main(args)