import torch
import os
import shutil
from torch.utils.data import DataLoader
from L_FG_SSM import L_FG_SSM
from dataset import MaritimeDataset
from losses import JointLoss
from tqdm import tqdm

# ==========================================
#        🚀 全局配置中心 (修改这里即可)
# ==========================================
CONFIG = {
    # --- 1. 路径设置 ---
    'root_dir': 'data',  # 数据集根目录
    'checkpoint_dir': 'checkpoints',  # 权重保存文件夹
    'uieb_filename': 'uieb_best.pth',  # UIEB 阶段保存的文件名
    'liaci_filename': 'liaci_best.pth',  # LIACI 阶段保存的文件名

    # --- 2. 硬件与系统 ---
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,  # 算力云 CPU 核心数，通常设 4 或 8 加速读取

    # --- 3. 模型参数 ---
    'base_dim': 32,  # 基础通道数
    'num_classes': 5,  # 类别数 (背景 + 4类缺陷)

    # --- 4. UIEB 训练配置 (第一阶段：增强) ---
    'uieb': {
        'epochs': 40,  # 推荐 40 轮，让 Mamba 充分收敛
        'batch_size': 8,  # 显存够大(24G)改成 16，显存小(8G)用 4 或 8
        'lr': 2e-4,  # 学习率
    },

    # --- 5. LIACI 训练配置 (第二阶段：任务微调) ---
    'liaci': {
        'epochs': 60,  # 推荐 60 轮，精细打磨分割边界
        'batch_size': 8,  # 微调建议 Batch 小一点，利于跳出局部最优
        'lr': 1e-4,  # 降低学习率，防止破坏预训练特征
    }
}


# ==========================================

def run_experiment():
    # 1. 初始化设置
    device = torch.device(CONFIG['device'])
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

    # 拼接完整路径
    uieb_path = os.path.join(CONFIG['checkpoint_dir'], CONFIG['uieb_filename'])
    liaci_path = os.path.join(CONFIG['checkpoint_dir'], CONFIG['liaci_filename'])


    # 2. 初始化模型
    model = L_FG_SSM(base_dim=CONFIG['base_dim'], num_classes=CONFIG['num_classes']).to(device)

    # 注意：优化器需要根据阶段不同可能需要重新初始化，这里为了简单，
    # 我们在每个阶段开始前单独定义优化器，以便使用不同的 LR。

    # ==========================================
    #      阶段一：UIEB 基础增强预训练
    # ==========================================

    # 检查是否已有 UIEB 权重 (实现断点续传/跳过)
    if os.path.exists(uieb_path):
        print(f"\n====> 检测到 UIEB 最佳权重 {uieb_path}，正在加载并跳过第一阶段...")
        # strict=False 允许加载部分权重 (防止分类头不匹配)
        model.load_state_dict(torch.load(uieb_path, map_location=device), strict=False)

    else:
        print("\n" + "=" * 40)
        print(f"开始 UIEB 阶段训练 (Epochs: {CONFIG['uieb']['epochs']}, LR: {CONFIG['uieb']['lr']})")
        print("=" * 40)

        # 定义 UIEB 专属的数据集和优化器
        dataset_uieb = MaritimeDataset(root_dir=CONFIG['root_dir'], mode='UIEB')
        loader_uieb = DataLoader(
            dataset_uieb,
            batch_size=CONFIG['uieb']['batch_size'],
            shuffle=True,
            num_workers=CONFIG['num_workers'],
            pin_memory=True  # 加速 GPU 传输
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['uieb']['lr'])
        criterion = JointLoss(mode='UIEB')

        best_loss = float('inf')

        for epoch in range(CONFIG['uieb']['epochs']):
            model.train()
            epoch_loss = 0

            with tqdm(loader_uieb, desc=f"UIEB [Epoch {epoch + 1}/{CONFIG['uieb']['epochs']}]", leave=False) as pbar:
                for raw, ref in loader_uieb:
                    raw, ref = raw.to(device), ref.to(device)

                    # 前向传播 (UIEB 只需要增强图，忽略分割输出)
                    enhanced, _ = model(raw)
                    loss = criterion(enhanced, ref)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = epoch_loss / len(loader_uieb)
            print(f"-> UIEB Epoch {epoch + 1} 结束，平均 Loss: {avg_loss:.4f}")

            # 只保存最佳权重
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), uieb_path)
                print(f"   ★ 刷新最佳记录！权重已保存至: {uieb_path}")

        print(f"\nUIEB 阶段训练完成！")

    # ==========================================
    #      阶段二：LIACI 任务驱动微调
    # ==========================================

    # 检查是否已有 LIACI 权重
    if os.path.exists(liaci_path):
        print(f"\n====> 检测到 LIACI 最佳权重 {liaci_path}，训练流程已全部完成！")
        return

    print("\n" + "=" * 40)
    print(f"开始 LIACI 阶段微调 (Epochs: {CONFIG['liaci']['epochs']}, LR: {CONFIG['liaci']['lr']})")
    print("=" * 40)

    # 定义 LIACI 专属的数据集和优化器 (LR 更低)
    dataset_liaci = MaritimeDataset(root_dir=CONFIG['root_dir'], mode='LIACI')
    loader_liaci = DataLoader(
        dataset_liaci,
        batch_size=CONFIG['liaci']['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    # 重新初始化优化器，使用微调的学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['liaci']['lr'])
    criterion = JointLoss(mode='LIACI')

    best_loss = float('inf')

    for epoch in range(CONFIG['liaci']['epochs']):
        model.train()
        epoch_loss = 0

        with tqdm(loader_liaci, desc=f"LIACI [Epoch {epoch + 1}/{CONFIG['liaci']['epochs']}]", leave=False) as pbar:
            for img, mask in loader_liaci:
                img, mask = img.to(device), mask.to(device)

                # 前向传播 (同时获取 增强图 和 分割预测)
                enhanced, mask_pred = model(img)

                # 计算联合损失 (必须传入 raw_img=img 以对齐维度)
                # target=mask 用于分割 Loss
                # raw_img=img 用于结构保持 Loss
                loss = criterion(enhanced, target=mask, pred_mask=mask_pred, raw_img=img)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(loader_liaci)
        print(f"-> LIACI Epoch {epoch + 1} 结束，平均 Loss: {avg_loss:.4f}")

        # 只保存最佳权重
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), liaci_path)
            print(f"权重已保存至: {liaci_path}")

    print(f"\n全部训练流程结束，模型位于: {liaci_path}")


if __name__ == "__main__":
    run_experiment()
