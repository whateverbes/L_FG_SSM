import torch
from torch.utils.data import DataLoader
from L_FG_SSM import L_FG_SSM
from dataset import MaritimeDataset
from losses import JointLoss
from tqdm import tqdm  # 导入进度条库


def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = L_FG_SSM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # --- 第一阶段: UIEB 基础增强训练 ---
    print("\n" + "=" * 20)
    print("开始 UIEB 基础增强预训练...")
    train_uieb = MaritimeDataset(root_dir='data', mode='UIEB')
    loader_uieb = DataLoader(train_uieb, batch_size=4, shuffle=True)
    criterion_uieb = JointLoss(mode='UIEB')

    for epoch in range(5):
        epoch_loss = 0
        with tqdm(total=len(loader_uieb), desc=f"UIEB [Epoch {epoch + 1}/5]", unit="batch", leave=True) as pbar:
            for raw, ref in loader_uieb:
                raw, ref = raw.to(device), ref.to(device)

                enhanced, _ = model(raw) # 使用下划线 _ 忽略掉目前不需要的分割输出
                loss = criterion_uieb(enhanced, ref)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                # 实时更新右侧信息
                pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
                pbar.update(1)

        avg_loss = epoch_loss / len(loader_uieb)
        print(f"\n-> UIEB Epoch {epoch + 1} 结束，平均 Loss: {avg_loss:.4f}")

    # --- 第二阶段: LIACI 实用性联合微调 ---
    print("\n" + "=" * 20)
    print("开始 LIACI 任务驱动微调 (验证实用性)...")
    train_liaci = MaritimeDataset(root_dir='data', mode='LIACI')
    loader_liaci = DataLoader(train_liaci, batch_size=4, shuffle=True)
    criterion_liaci = JointLoss(mode='LIACI')

    # --- LIACI 训练阶段 ---
    for epoch in range(10):
        epoch_loss = 0
        with tqdm(loader_liaci, desc=f"LIACI [Epoch {epoch + 1}/10]", leave=False) as pbar:
            for img, mask in loader_liaci:
                img, mask = img.to(device), mask.to(device)

                # 【核心修正】：一次调用 model，同时拿到增强图和分割预测
                # 此时 mask_pred 已经是 5 通道的了，不用再手动调 model.task_head
                enhanced, mask_pred = model(img)

                # 计算联合 Loss (在 losses.py 中 pred_mask 对应 mask_pred)
                loss = criterion_liaci(enhanced, mask, mask_pred)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
                pbar.update(1)
        print(f"\n-> LIACI Epoch {epoch + 1} 结束, 平均 Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'final_model.pth')
    print("\n训练完成！模型已保存为 final_model.pth")


if __name__ == "__main__":
    run_experiment()