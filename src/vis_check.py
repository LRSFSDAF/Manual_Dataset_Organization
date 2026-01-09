import inspect

# === 【保留这个补丁】解决 Python 3.10+ Chumpy 报错的问题 ===
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec
# ========================================================

import cv2
import numpy as np
import os
from pathlib import Path
from freihand_reader import FreiHandReader

# ================= 配置路径 =================

# 以当前脚本所在目录为基准
HERE = Path(__file__).resolve().parent
# 项目根目录
ROOT = HERE.parent

# 1. 数据集根目录 (指向 FreiHAND_pub_v2)
DATA_ROOT = str((ROOT / "FreiHAND" / "FreiHAND_pub_v2").resolve())

# 2. MANO 模型路径 (指向 models/mano_v1_2/models/MANO_RIGHT.pkl)
MANO_PATH = str((ROOT / "models" / "mano_v1_2" / "models" / "MANO_RIGHT.pkl").resolve())

# 3. 输出目录 (结果会保存在项目根目录下)
OUTPUT_DIR = str((ROOT / "output_vis_freihand").resolve())
# ==========================================================

def project_points(points_3d, K):
    points_2d = points_3d @ K.T
    points_2d = points_2d[:, :2] / points_2d[:, 2:]
    return points_2d

def render_mesh(img, verts, faces, K, color=(0, 255, 0)):
    verts_2d = project_points(verts, K)
    for face in faces:
        pts = verts_2d[face].astype(np.int32)
        cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)
    return img

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"初始化 Reader，路径: {DATA_ROOT}")
    try:
        reader = FreiHandReader(DATA_ROOT, MANO_PATH)
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    # 随机取 5 个样本
    indices = [0, 100, 200, 555, 1000] 
    
    print("开始生成可视化...")
    for i, idx in enumerate(indices):
        print(f"正在处理第 {idx} 张图片...")
        data = reader.get_frame_data(idx)
        
        img = data['image_bgr'].copy()
        K = data['K']
        
        # 1. 投影原始标注的 3D 关节 (红色实心点)
        gt_joints_3d = data['joints_3d']
        gt_joints_2d = project_points(gt_joints_3d, K)
        
        for pt in gt_joints_2d:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
            
        # 2. 从 MANO 参数生成 Mesh (绿色网格)
        try:
            # 传入 3D 关节真值，让函数自动对齐
            verts, _ = reader.get_mesh_verts(data['mano_pose'], data['mano_betas'], data['joints_3d'])
            faces = reader.mano_layer.faces
            img_mesh = render_mesh(img.copy(), verts, faces, K, color=(0, 255, 0))
            
            # 叠加显示
            alpha = 0.6
            vis_result = cv2.addWeighted(img, 1 - alpha, img_mesh, alpha, 0)
            
        except Exception as e:
            print(f"Mesh 生成失败: {e}")
            vis_result = img
            
        save_path = os.path.join(OUTPUT_DIR, f"freihand_vis_{idx:04d}.jpg")
        cv2.imwrite(save_path, vis_result)
        print(f"已保存: {save_path}")

    print("全部完成！请查看 output_vis_freihand 文件夹。")

if __name__ == "__main__":
    main()