import torch
import os
import shutil
from torch.utils.data import DataLoader
from L_FG_SSM import L_FG_SSM
from dataset import MaritimeDataset
from losses import JointLoss
from tqdm import tqdm


def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 1. 路径与文件夹配置 ---
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)  # 自动创建 checkpoints 文件夹

    # 定义权重文件的完整路径
    uieb_best_path = os.path.join(checkpoint_dir, 'uieb_best.pth')
    liaci_best_path = os.path.join(checkpoint_dir, 'liaci_best.pth')

    # 初始化模型
    model = L_FG_SSM(base_dim=32, num_classes=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # --- 自动化流程：检查是否已完成 LIACI ---
    if os.path.exists(liaci_best_path):
        print(f"====> 检测到 LIACI 权重 {liaci_best_path}，训练已完成。")
        return  # 直接结束，或者可以在这里加推理逻辑

    # --- 第一阶段: UIEB 基础增强训练 ---
    # 检查是否有 UIEB 权重
    if os.path.exists(uieb_best_path):
        print(f"====> 发现 UIEB 预训练权重 {uieb_best_path}，正在加载并跳过第一阶段...")
        # strict=False 防止后续头不匹配报错
        model.load_state_dict(torch.load(uieb_best_path, map_location=device), strict=False)
    else:
        print("\n" + "=" * 30)
        print("开始 UIEB 基础增强预训练...")
        train_uieb = MaritimeDataset(root_dir='data', mode='UIEB')
        loader_uieb = DataLoader(train_uieb, batch_size=4, shuffle=True)
        criterion_uieb = JointLoss(mode='UIEB')

        best_uieb_loss = float('inf')  # 初始化最小 Loss

        for epoch in range(5):
            epoch_loss = 0
            model.train()  # 确保在训练模式
            with tqdm(loader_uieb, desc=f"UIEB [Epoch {epoch + 1}/5]", leave=True) as pbar:
                for raw, ref in loader_uieb:
                    raw, ref = raw.to(device), ref.to(device)
                    enhanced, _ = model(raw)
                    loss = criterion_uieb(enhanced, ref)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                    pbar.update(1)

            avg_loss = epoch_loss / len(loader_uieb)
            print(f"-> UIEB Epoch {epoch + 1} 结束，平均 Loss: {avg_loss:.4f}")

            # --- 核心：只保存最佳权重 ---
            if avg_loss < best_uieb_loss:
                best_uieb_loss = avg_loss
                torch.save(model.state_dict(), uieb_best_path)

        print(f"UIEB 阶段结束，权重已保存。")

    # --- 第二阶段: LIACI 实用性联合微调 ---
    print("\n" + "=" * 30)
    print("开始 LIACI 任务驱动微调...")
    train_liaci = MaritimeDataset(root_dir='data', mode='LIACI')
    loader_liaci = DataLoader(train_liaci, batch_size=4, shuffle=True)
    criterion_liaci = JointLoss(mode='LIACI')

    best_liaci_loss = float('inf')  # 初始化 LIACI 的最小 Loss

    for epoch in range(10):
        epoch_loss = 0
        model.train()
        with tqdm(loader_liaci, desc=f"LIACI [Epoch {epoch + 1}/10]", leave=True) as pbar:
            for img, mask in loader_liaci:
                img, mask = img.to(device), mask.to(device)

                # 前向传播
                enhanced, mask_pred = model(img)

                # 计算联合 Loss (确保参数名对应 losses.py 的定义)
                loss = criterion_liaci(enhanced, target=mask, pred_mask=mask_pred, raw_img=img)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                pbar.update(1)

        avg_loss = epoch_loss / len(loader_liaci)
        print(f"-> LIACI Epoch {epoch + 1} 结束, 平均 Loss: {avg_loss:.4f}")

        # --- 核心：只保存最佳权重 ---
        if avg_loss < best_liaci_loss:
            best_liaci_loss = avg_loss
            torch.save(model.state_dict(), liaci_best_path)

    print(f"\n全部训练完成！最佳模型位于: {liaci_best_path}")


if __name__ == "__main__":
    run_experiment()
