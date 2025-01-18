from pathlib import Path
import os
import rasterio
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from shapely import wkt
import json
import torch.nn.functional as F

def wce_loss(y_true, y_pred, gamma_1=0.8, gamma_0=0.2):
    """
    Compute Weighted Cross-Entropy (WCE) loss in PyTorch.
    
    Args:
        y_true: Tensor of ground truth values (shape: [batch_size, height, width]).
                Each value is either 0 (non-building) or 1 (building).
        y_pred: Tensor of predicted probabilities (shape: [batch_size, height, width]).
                Values should be in the range [0, 1], representing P(Y=1).
        gamma_1: Weight for positive class (building).
        gamma_0: Weight for negative class (non-building).

    Returns:
        Weighted Cross-Entropy loss as a scalar tensor.
    """
    # Clip predictions to avoid log(0) and ensure numerical stability
    epsilon = 1e-7
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    
    # Compute weighted cross-entropy loss
    wce_loss = - (gamma_1 * y_true * torch.log(y_pred) + 
                  gamma_0 * (1 - y_true) * torch.log(1 - y_pred))
    
    # Take the mean loss across all pixels in the batch
    return torch.mean(wce_loss)


def focal_loss(y_true, y_pred, alpha=0.5, gamma=2):
    """
    Compute Focal Loss in PyTorch.
    
    Args:
        y_true: Tensor of ground truth values (shape: [batch_size, height, width]).
                Each value is either 0 (non-building) or 1 (building).
        y_pred: Tensor of predicted probabilities (shape: [batch_size, height, width]).
                Values should be in the range [0, 1], representing P(Y=1).
        alpha: Weighting factor for the positive class (default: 0.5).
        gamma: Focusing parameter to reduce the loss contribution from easy examples (default: 2).

    Returns:
        Focal loss as a scalar tensor.
    """
    # Clip predictions to avoid log(0) and ensure numerical stability
    epsilon = 1e-7
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    
    # Calculate the focal loss
    focal_loss = - (
        alpha * (1 - y_pred)**gamma * y_true * torch.log(y_pred) +
        (1 - alpha) * y_pred**gamma * (1 - y_true) * torch.log(1 - y_pred)
    )
    
    # Take the mean loss across all pixels in the batch
    return torch.mean(focal_loss)


def dice_loss(y_true, y_pred, epsilon=1e-7):
    """
    Compute Dice Loss for binary segmentation.

    Args:
        y_true: Tensor of ground truth values (shape: [batch_size, height, width]).
                Each value is either 0 (non-building) or 1 (building).
        y_pred: Tensor of predicted probabilities (shape: [batch_size, height, width]).
                Values should be in the range [0, 1], representing P(Y=1).
        epsilon: Small constant to prevent division by zero (default: 1e-7).

    Returns:
        Dice loss as a scalar tensor.
    """
    # Flatten the tensors to compute the intersection and union over the entire batch
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)
    
    # Compute intersection and union
    intersection = torch.sum(y_true_flat * y_pred_flat)
    union = torch.sum(y_true_flat) + torch.sum(y_pred_flat)
    
    # Compute Dice coefficient and loss
    dice_coeff = (2.0 * intersection + epsilon) / (union + epsilon)
    dice_loss = 1.0 - dice_coeff
    
    return dice_loss


def tversky_loss(y_true, y_pred, beta=0.8, epsilon=0.05):
    """
    Compute Tversky Loss for binary segmentation.

    Args:
        y_true: Tensor of ground truth values (shape: [batch_size, height, width]).
                Each value is either 0 (non-building) or 1 (building).
        y_pred: Tensor of predicted probabilities (shape: [batch_size, height, width]).
                Values should be in the range [0, 1], representing P(Y=1).
        beta: Controls the penalty for false positives (default: 0.8).
        epsilon: Small constant to prevent division by zero (default: 0.05).

    Returns:
        Tversky loss as a scalar tensor.
    """
    # Flatten the true and predicted labels
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    
    # True positive, false positive, false negative
    tp = torch.sum(y_true_f * y_pred_f)
    fp = torch.sum((1 - y_true_f) * y_pred_f)
    fn = torch.sum(y_true_f * (1 - y_pred_f))
    
    # Calculate the Tversky loss
    tversky = 1 - (tp + epsilon) / (tp + beta * fp + (1 - beta) * fn + epsilon)
    
    return tversky


def combo_loss(y_true, y_pred, mu1=1, mu2=10):
    # Compute Dice loss
    dice = dice_loss(y_true, y_pred)
    
    # Compute Focal loss
    focal = focal_loss(y_true, y_pred)
    
    # Weighted combination of Dice and Focal loss
    combo = mu1 * dice + mu2 * focal
    
    return combo


def calculate_metrics(y_true, y_pred, epsilon=1e-7):
    # Flatten the ground truth and predicted masks
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)
    
    # Calculate True Positives, False Positives, False Negatives
    TP = torch.sum(y_true_flat * y_pred_flat)
    FP = torch.sum((1 - y_true_flat) * y_pred_flat)
    FN = torch.sum(y_true_flat * (1 - y_pred_flat))
    
    # Calculate Precision, Recall, F1 score
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    
    # Calculate Intersection over Union (IoU)
    iou = TP / (TP + FP + FN + epsilon)
    
    return {"precision": precision, "recall": recall, "f1_score": f1_score, "iou": iou}

def calculate_loss(y_pred, y_true):
    wce = wce_loss(y_true, y_pred)
    focal = focal_loss(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    tversky = tversky_loss(y_true, y_pred)
    combo = combo_loss(y_true, y_pred)
    
    return {"wce_loss": wce, "focal_loss": focal, "dice_loss": dice, "tversky_loss": tversky, "combo_loss": combo}




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mae_loss(output, target):
    pass


    

def mask_for_polygon(poly, im_size=(1024, 1024)):
    img_mask = np.zeros(im_size, np.uint8)
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords)]
    interiors = [int_coords(pi.coords) for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask

def get_damage_mask(damage_types, image_file_path):
    damage_dict = {damage_type:idx for idx, damage_type in enumerate(damage_types, start=1)}
    msk_damage = np.zeros((1024, 1024), dtype='uint8')
    label_file_path = str(image_file_path).replace("images", "labels").replace(".tif", ".json")
    with open(label_file_path, "r") as f:
        json_data = json.load(f)
    for item in json_data['features']['xy']:
        subtype = item['properties']['subtype'].replace('-', '_')
        poly = wkt.loads(item['wkt'])
        _msk = mask_for_polygon(poly)
        msk_damage[_msk > 0] = damage_dict.get(subtype, 0)
    msk_damage = msk_damage[np.newaxis, :, :]
    return msk_damage


class TrainDataset(Dataset):
    def __init__(self, train_data_paths, damage_types):
        super().__init__()
        image_ids = []
        for train_data_path in train_data_paths:
            train_data_path = Path(train_data_path)
            for filename in os.listdir(train_data_path / "images"):
                image_ids.append(train_data_path / "images" / f"{'_'.join(filename.split('_')[:-2])}_post_disaster.tif")
        image_ids = sorted(set(image_ids))
        self.image_ids = list(image_ids)
        self.damage_types = damage_types


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        with rasterio.open(image_id) as f:
            img = f.read()
        img = img / 255.0
        img_tensor = torch.from_numpy(img).float()
        msk = get_damage_mask(self.damage_types, image_id)
        item = {'img': img_tensor, 'msk': msk, 'fn': str(image_id)}
        return item


    
class ValDataset(Dataset):
    def __init__(self, val_data_paths, damage_types):
        super().__init__()
        image_ids = []
        for val_data_path in val_data_paths:
            val_data_path = Path(val_data_path)
            for filename in os.listdir(val_data_path / "images"):
                image_ids.append(val_data_path / "images" / f"{'_'.join(filename.split('_')[:-2])}_post_disaster.tif")
        image_ids = sorted(set(image_ids))
        self.image_ids = list(image_ids)
        self.damage_types = damage_types


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        with rasterio.open(image_id) as f:
            img = f.read()
        img_tensor = torch.from_numpy(img).float()
        msk = get_damage_mask(self.damage_types, image_id)
        item = {'img': img_tensor, 'msk': msk, 'fn': str(image_id)}
        return item