import cv2
import numpy as np
import os

# 把这里改成你 mask 文件夹的真实路径
mask_dir = "data/LIACI/masks" 

files = os.listdir(mask_dir)
if len(files) > 0:
    mask_path = os.path.join(mask_dir, files[0])
    mask = cv2.imread(mask_path, 0) # 单通道读取
    
    print(f"检查文件: {files[0]}")
    print(f"Mask 形状: {mask.shape}")
    print(f"Mask 里的唯一像素值有: {np.unique(mask)}")
    
    # 诊断
    values = np.unique(mask)
    if np.max(values) > 4:
        print("❌ 严重错误！Mask 像素值超过了 4 (类别数)！")
        print("模型把 255 当成了第 255 类，当然会崩！")
        print("需要把 255 转换成 1,2,3,4 或者 0。")
    else:
        print("✅ Mask 数据格式看起来正常 (0-4)。")
