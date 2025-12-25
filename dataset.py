import os
import torch
import cv2
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class MaritimeDataset(Dataset):
    def __init__(self, root_dir, mode='UIEB', transform=None):
        """
        mode: 'UIEB' (增强训练) 或 'LIACI' (实用性验证/分割训练)
        """
        self.mode = mode
        self.transform = transform
        self.data = []
        self.target_size = (256, 256) # 统一尺寸

        if mode == 'UIEB':
            raw_dir = os.path.join(root_dir, 'UIEB/raw-890')
            ref_dir = os.path.join(root_dir, 'UIEB/reference-890')
            if not os.path.exists(raw_dir):
                print(f"警告: UIEB 路径不存在 {raw_dir}")
                return
            filenames = sorted(os.listdir(raw_dir))
            for f in filenames:
                self.data.append({
                    'raw': os.path.join(raw_dir, f),
                    'ref': os.path.join(ref_dir, f)
                })

        elif mode == 'LIACI':
            img_dir = os.path.join(root_dir, 'LIACI/images')
            mask_root = os.path.join(root_dir, 'LIACI/masks')
            split_csv = os.path.join(root_dir, 'LIACI/train_test_split.csv')
            
            # 定义类别映射
            self.categories = ['anode', 'bilge_keel', 'corrosion', 'defect']
            
            if not os.path.exists(split_csv):
                 print(f"警告: LIACI 分割文件不存在 {split_csv}")
                 return

            split_df = pd.read_csv(split_csv)
            train_files = split_df[split_df['split'] == 'train']['file_name'].tolist()

            for f in train_files:
                fname = os.path.basename(f)
                base_name = os.path.splitext(fname)[0]
                raw_path = os.path.join(img_dir, fname)
                
                found_mask = None
                cat_idx = 0
                
                # 寻找对应的掩码
                for i, cat in enumerate(self.categories):
                    potential_mask = os.path.join(mask_root, cat, base_name + '.bmp')
                    if os.path.exists(potential_mask):
                        found_mask = potential_mask
                        cat_idx = i + 1 # 类别从1开始
                        break
                
                if os.path.exists(raw_path) and found_mask:
                    self.data.append({
                        'raw': raw_path,
                        'mask': found_mask,
                        'label': cat_idx
                    })
            
            print(f"\nLIACI 成功加载样本: {len(self.data)} (包含类别: {self.categories})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 读取原图
        raw_img = cv2.imread(item['raw'])
        if raw_img is None:
            # 容错处理
            raw_img = np.zeros((256, 256, 3), dtype=np.uint8)
            
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        raw_img = cv2.resize(raw_img, self.target_size)
        
        # 归一化 [0,1]
        raw_tensor = torch.from_numpy(raw_img).permute(2, 0, 1).float() / 255.0

        if self.mode == 'UIEB':
            ref_img = cv2.imread(item['ref'])
            if ref_img is None: ref_img = np.zeros((256, 256, 3), dtype=np.uint8)
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            ref_img = cv2.resize(ref_img, self.target_size)
            ref_tensor = torch.from_numpy(ref_img).permute(2, 0, 1).float() / 255.0
            return raw_tensor, ref_tensor

        elif self.mode == 'LIACI':
            mask = cv2.imread(item['mask'], 0)
            if mask is None: mask = np.zeros((256, 256), dtype=np.uint8)

            # 强制数值转换: 255 -> 1 (或对应类别)
            mask[mask > 0] = item['label'] 
            # 如果想简化为二分类，用 mask[mask > 0] = 1

            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            mask_tensor = torch.from_numpy(mask).long()

            # 【BUG修复】这里必须用 'raw'，不能用 'img'
            # item['raw'] 已经在上面读取为 raw_tensor 了，直接复用即可
            return raw_tensor, mask_tensor
