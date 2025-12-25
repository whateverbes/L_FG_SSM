import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: [B, C, H, W] (Logits)
        # target: [B, H, W] (Indices)
        
        num_classes = pred.shape[1]
        pred_softmax = F.softmax(pred, dim=1)
        
        # One-hot encoding for target
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        
        # Calculate Dice for each class
        intersection = (pred_softmax * target_one_hot).sum(dim=(2, 3))
        union = pred_softmax.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Average Dice Loss across classes (1 - Dice)
        return 1.0 - dice_score.mean()

class JointLoss(nn.Module):
    def __init__(self, mode='UIEB'):
        super().__init__()
        self.mode = mode
        self.l1 = nn.L1Loss()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss() # 新增

    def forward(self, pred_img, target, pred_mask=None, raw_img=None):
        if self.mode == 'UIEB':
            return self.l1(pred_img, target)

        elif self.mode == 'LIACI':
            # 1. 结构一致性 Loss (L1)
            l_img = 0.0
            if raw_img is not None:
                l_img = self.l1(pred_img, raw_img)
            
            # 2. 分割任务 Loss (CE + Dice)
            l_ce = self.ce(pred_mask, target)
            l_dice = self.dice(pred_mask, target)
            
            # 混合 Loss：结构 + 分类 + 形状
            # 0.2*结构保真 + 1.0*像素分类 + 1.0*形状拟合
            return 0.2 * l_img + 1.0 * l_ce + 1.0 * l_dice
