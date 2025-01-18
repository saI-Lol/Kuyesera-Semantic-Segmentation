from pathlib import Path
import os
import rasterio
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np


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

def mae_loss(input, target):
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
            for filename in os.listdir(str(val_data_path / "images")):
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