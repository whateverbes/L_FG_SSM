import torch.nn as nn
import torch.nn.functional as F


class JointLoss(nn.Module):
    def __init__(self, mode='UIEB'):
        super().__init__()
        self.mode = mode
        self.l1 = nn.L1Loss()
        self.ce = nn.CrossEntropyLoss()

    # 【关键修改】：添加 raw_img 参数，参数名统一为 pred_mask
    def forward(self, pred_img, target, pred_mask=None, raw_img=None):
        if self.mode == 'UIEB':
            # UIEB 阶段：target 是参考图 [B, 3, H, W]
            return self.l1(pred_img, target)

        elif self.mode == 'LIACI':
            # LIACI 阶段：target 是 Mask [B, H, W]

            # 1. 结构一致性 Loss：
            # 必须拿【增强图 pred_img】和【原图 raw_img】比
            # 如果 raw_img 没传（防呆），就只能和自己比（0 loss），但正常应该传进来
            if raw_img is not None:
                l_img = self.l1(pred_img, raw_img)
            else:
                l_img = 0.0

                # 2. 分割任务 Loss：
            # 拿【预测掩码 pred_mask】和【真实掩码 target】比
            l_task = self.ce(pred_mask, target)

            # 联合 Loss
            return 0.1 * l_img + 1.0 * l_task
