import torch.nn as nn
import torch.nn.functional as F

class JointLoss(nn.Module):
    def __init__(self, mode='UIEB'):
        super().__init__()
        self.mode = mode
        self.l1 = nn.L1Loss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred_img, target, pred_mask=None):
        if self.mode == 'UIEB':
            # target 是 reference 图像
            return self.l1(pred_img, target)
        
        elif self.mode == 'LIACI':
            # target 是 mask 标签
            # 增强损失 (Self-Reference: 尽量保持原图结构)
            l_img = self.l1(pred_img, target) # 此处 target 若无 ref 则可设为输入图
            # 分割任务损失 (核心实用性指标)
            l_task = self.ce(pred_mask, target)
            return 0.5 * l_img + 1.0 * l_task