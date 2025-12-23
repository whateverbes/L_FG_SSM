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
        self.target_size = (256, 256)  # 统一缩放尺寸，解决 Stack 报错

        if mode == 'UIEB':
            raw_dir = os.path.join(root_dir, 'UIEB/raw-890')
            ref_dir = os.path.join(root_dir, 'UIEB/reference-890')
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

            # 定义类别映射（实用性：区分不同工业特征）

            self.categories = ['anode', 'bilge_keel', 'corrosion', 'defect']

            split_df = pd.read_csv(split_csv)

            train_files = split_df[split_df['split'] == 'train']['file_name'].tolist()

            for f in train_files:

                fname = os.path.basename(f)

                base_name = os.path.splitext(fname)[0]

                raw_path = os.path.join(img_dir, fname)

                # 在四个子文件夹中寻找对应的 .bmp 掩码

                found_mask = None

                cat_idx = 0

                for i, cat in enumerate(self.categories):

                    # 拼接子文件夹路径，例如 LIACI/masks/corrosion/image_0001.bmp

                    potential_mask = os.path.join(mask_root, cat, base_name + '.bmp')

                    if os.path.exists(potential_mask):
                        found_mask = potential_mask

                        cat_idx = i + 1  # 类别从1开始，0留给背景

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
        raw_img = cv2.cvtColor(cv2.imread(item['raw']), cv2.COLOR_BGR2RGB)
        raw_img = cv2.resize(raw_img, self.target_size)

        if self.mode == 'UIEB':
            ref_img = cv2.imread(item['ref'])
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            ref_img = cv2.resize(ref_img, self.target_size)

            # 归一化并转为 Tensor [C, H, W]
            raw_tensor = torch.from_numpy(raw_img).permute(2, 0, 1).float() / 255.0
            ref_tensor = torch.from_numpy(ref_img).permute(2, 0, 1).float() / 255.0
            return raw_tensor, ref_tensor


        elif self.mode == 'LIACI':
            mask = cv2.imread(item['mask'], 0)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            # 将该掩码区域标记为对应的类别 ID (1, 2, 3, 或 4)
            # LIACI 的 bmp 掩码通常是 255 表示目标区域
            final_mask = np.zeros(self.target_size, dtype=np.int64)
            final_mask[mask > 0] = item['label']

            raw_tensor = torch.from_numpy(raw_img).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(final_mask).long()
            return raw_tensor, mask_tensor