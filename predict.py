import torch
import cv2
import os
import numpy as np
from L_FG_SSM import L_FG_SSM


def predict_folder(input_folder, output_folder, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_folder, exist_ok=True)

    # 1. 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        print("请检查 checkpoints 文件夹下是否有该权重文件。")
        return

    # 2. 加载模型
    print(f"正在加载模型: {model_path} ...")
    # 注意：必须保持和训练时一致的参数 (base_dim=32, num_classes=5)
    model = L_FG_SSM(base_dim=32, num_classes=5).to(device)

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 3. 准备图片列表
    if not os.path.exists(input_folder):
        print(f"错误: 找不到输入文件夹 {input_folder}")
        return

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print(f"在 {input_folder} 中找到 {len(image_files)} 张图片，开始推理...")

    # 4. 批量推理
    with torch.no_grad():
        for img_name in image_files:
            img_path = os.path.join(input_folder, img_name)

            # 读取原始图片
            original_img = cv2.imread(img_path)
            if original_img is None: continue

            # 为了输入模型，resize 到 256x256
            # (如果你的测试图片比例差异很大，这里可能需要保持比例缩放，但在验证阶段直接 resize 最稳)
            img = cv2.resize(original_img, (256, 256))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 归一化 + 转 Tensor [1, 3, 256, 256]
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            # --- 模型前向传播 ---
            enhanced, mask_pred = model(img_tensor)

            # --- 结果后处理 ---

            # A. 处理增强图
            # Tensor [1, 3, H, W] -> Numpy [H, W, 3] (BGR格式用于OpenCV保存)
            enhanced_np = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
            enhanced_np = np.clip(enhanced_np * 255.0, 0, 255).astype(np.uint8)
            enhanced_bgr = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)

            # B. 处理预测掩码
            # [1, 5, H, W] -> argmax 拿到类别索引 -> [H, W]
            mask_idx = torch.argmax(mask_pred, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # 给掩码上色以便观察 (背景黑，缺陷用不同颜色)
            colored_mask = np.zeros_like(img)
            # 定义颜色表: [背景, 类1, 类2, 类3, 类4]
            # 颜色格式 BGR: (Blue, Green, Red)
            colors = [
                (0, 0, 0),  # 0: Background (黑)
                (0, 0, 255),  # 1: 红色
                (0, 255, 0),  # 2: 绿色
                (255, 0, 0),  # 3: 蓝色
                (0, 255, 255)  # 4: 黄色
            ]

            for cls_id in range(1, 5):  # 遍历 1~4 类
                colored_mask[mask_idx == cls_id] = colors[cls_id]

            # --- 拼接对比图 ---
            # 左：原图 (Resize后)，中：增强图，右：预测掩码
            # 加个白色分隔线看起来更清楚
            sep = np.ones((256, 10, 3), dtype=np.uint8) * 255
            combined = np.hstack([img, sep, enhanced_bgr, sep, colored_mask])

            # 保存
            save_path = os.path.join(output_folder, "res_" + img_name)
            cv2.imwrite(save_path, combined)
            print(f"处理完成: {img_name} -> {save_path}")

    print("\n所有图片推理完成！")


if __name__ == "__main__":
    # --- 配置区域 ---

    # 1. 待测试图片的文件夹路径 (请根据你实际情况修改)
    test_images_dir = "data/LIACI/test"

    # 2. 结果保存位置
    #save_dir = "results/uieb"
    save_dir = "results"
    # 3. 模型权重路径 (已修改为指向 checkpoints 文件夹)
    # 如果你想测试 UIEB 阶段的效果，可以改为 'checkpoints/uieb_best.pth'
    model_path = "checkpoints/liaci_best.pth"
    #model_path = "checkpoints/uieb_best.pth"
    predict_folder(test_images_dir, save_dir, model_path)
