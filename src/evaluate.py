import os
import sys
import numpy as np
from tqdm import tqdm

# 路径配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, "FreiHAND/FreiHAND_pub_v2")
MANO_PATH = os.path.join(PROJECT_ROOT, "models/mano_v1_2/models/MANO_RIGHT.pkl")

# 导入两个 Reader
from freihand_reader import FreiHandReader
from unified_reader import FreiHandUnifiedReader

def compute_mpjpe(pred, gt):
    """
    计算根节点对齐后的 MPJPE (单位: mm)
    pred, gt: (21, 3) 数组
    """
    # 1. 根节点对齐 (减去手腕坐标)
    pred_aligned = pred - pred[0:1]
    gt_aligned   = gt - gt[0:1]
    
    # 2. 计算欧氏距离
    diff = pred_aligned - gt_aligned
    dist = np.sqrt(np.sum(diff**2, axis=-1)) # (21,)
    
    return np.mean(dist) * 1000 # 转为 mm

def main():
    print("=== 开始评估 MPJPE ===")
    
    # 1. 加载真值 Reader
    gt_reader = FreiHandReader(DATA_ROOT, MANO_PATH)
    # 2. 加载预测 Reader
    pred_reader = FreiHandUnifiedReader(DATA_ROOT)
    
    if len(pred_reader) == 0:
        print("错误：没有找到预测文件，请先运行 run_wilor.py")
        return

    errors = []
    
    # 遍历所有预测结果
    print(f"正在对比 {len(pred_reader)} 个样本...")
    
    for idx in tqdm(range(len(pred_reader))):
        try:
            # 获取预测数据 (不读图，速度快)
            pred_data = pred_reader.fetch_frame_data(idx, get_img=False)
            
            # 检查是否有有效手部
            if pred_data['right'] is None or not pred_data['right']['hand_valid']:
                continue
                
            pred_joints = pred_data['right']['joint3d']
            
            # 获取对应的真值 (通过文件名里的 ID 对应)
            # 假设 pkl 是顺序生成的，且 FreiHandReader 也是顺序读的，ID 是一致的
            # 这里直接用 idx 索引 FreiHandReader
            gt_data = gt_reader.get_frame_data(idx) 
            gt_joints = gt_data['joints_3d']
            
            # 计算误差
            err = compute_mpjpe(pred_joints, gt_joints)
            errors.append(err)
            
        except Exception as e:
            continue
            
    if len(errors) > 0:
        mean_error = np.mean(errors)
        print(f"\n✅ 评估完成！")
        print(f"测试样本数: {len(errors)}")
        print(f"平均 MPJPE 误差: {mean_error:.4f} mm")
    else:
        print("没有计算出任何有效误差，请检查数据。")

if __name__ == "__main__":
    main()