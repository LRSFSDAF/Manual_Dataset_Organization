import os
import sys
import numpy as np
import pickle
import torch
import cv2
import traceback
from pathlib import Path
from tqdm import tqdm
from unittest.mock import MagicMock

# ================= 欺骗系统绕过 EGL =================
sys.modules["pyrender"] = MagicMock()
sys.modules["OpenGL"] = MagicMock()
sys.modules["OpenGL.EGL"] = MagicMock()
sys.modules["OpenGL.GL"] = MagicMock()
os.environ["PYOPENGL_PLATFORM"] = "" 
# ===================================================

# ================= 路径配置 =================
# 以当前脚本所在目录为基准
HERE = Path(__file__).resolve().parent
# 项目根目录
ROOT = HERE.parent

# 1. 数据集根目录 (指向 FreiHAND_pub_v2)
DATA_ROOT = str((ROOT / "FreiHAND" / "FreiHAND_pub_v2").resolve())

# 2. MANO 模型路径 (指向 models/mano_v1_2/models/MANO_RIGHT.pkl)
MANO_PATH = str((ROOT / "models" / "mano_v1_2" / "models" / "MANO_RIGHT.pkl").resolve())

# 3. WiLoR 模型根目录
WILOR_ROOT = str((ROOT / "models" / "WiLoR").resolve())

# 4. 输出目录 (结果会保存在项目根目录下)
SAVE_DIR = str((ROOT / "output_vis_freihand").resolve())

# ================= 环境植入 =================
sys.path.append(WILOR_ROOT)

from freihand_reader import FreiHandReader

try:
    from wilor.models.wilor import WiLoR
    from yacs.config import CfgNode as CN
    print("✅ WiLoR 模块导入成功！")
except ImportError as e:
    print(f"❌ WiLoR 导入失败: {e}")
    sys.exit(1)

# ================= 模型包装器 =================
class WiLoRWrapper:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"正在加载 WiLoR 模型 (Device: {self.device})...")
        
        cfg_path = os.path.join(WILOR_ROOT, "pretrained_models/model_config.yaml")
        detector_path = os.path.join(WILOR_ROOT, "pretrained_models/detector.pt")
        ckpt_path = os.path.join(WILOR_ROOT, "pretrained_models/wilor_final.ckpt")
        mano_data_dir = os.path.join(WILOR_ROOT, "mano_data")
        mano_mean_path = os.path.join(mano_data_dir, "mano_mean_params.npz")
        mano_pkl_path = os.path.join(mano_data_dir, "MANO_RIGHT.pkl")

        if not os.path.exists(mano_mean_path):
            print(f"⚠️ 警告: {mano_mean_path} 不存在")

        with open(cfg_path, 'r') as f:
            self.cfg = CN.load_cfg(f)
        
        self.cfg.defrost()
        self.cfg.MANO.DATA_DIR = mano_data_dir
        self.cfg.MANO.MEAN_PARAMS = mano_mean_path
        self.cfg.MANO.MODEL_PATH = mano_pkl_path
        self.cfg.MODEL.PRETRAINED_WEIGHTS = ckpt_path
        self.cfg.hw_detector_path = detector_path
        if hasattr(self.cfg.MODEL, 'BACKBONE'):
            self.cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS = ckpt_path
        self.cfg.freeze()

        # 兼容不同版本 WiLoR 模型的初始化参数：
        try:
            self.model = WiLoR(self.cfg, specs=None) 
        except TypeError:
             self.model = WiLoR(self.cfg)

        if os.path.exists(ckpt_path):
            print(f"加载 Checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict, strict=False)
        
        self.model.to(self.device)
        self.model.eval()
        
        # 智能解析输入尺寸
        if hasattr(self.cfg.MODEL, 'IMAGE_SIZE'):
            img_size = self.cfg.MODEL.IMAGE_SIZE
            # 如果是整数 (如 256)，转为 (256, 256)
            if isinstance(img_size, int):
                self.input_size = (img_size, img_size)
            # 如果是列表/元组 (如 [192, 256])，直接用
            else:
                self.input_size = tuple(img_size)
        else:
            self.input_size = (192, 256) # 默认值
            
        print(f"✅ 模型初始化完成！目标输入尺寸: {self.input_size}")

    def predict(self, img_bgr):
        """单帧推理"""
        orig_h, orig_w = img_bgr.shape[:2]
        
        # Resize 到模型指定尺寸
        img_resized = cv2.resize(img_bgr, self.input_size)
        
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().to(self.device)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # 构造字典输入
        input_batch = {'img': img_tensor}
        
        with torch.no_grad():
            outputs = self.model(input_batch)
            
        def get_val(keys, default_shape):
            for k in keys:
                if k in outputs:
                    return outputs[k][0].cpu().numpy()
            return np.zeros(default_shape, dtype=np.float32)

        pred_pose   = get_val(['pred_pose', 'pose', 'theta'], (16, 3))
        pred_shape  = get_val(['pred_shape', 'betas'], (10,))
        pred_joints = get_val(['pred_keypoints_3d', 'joints3d', 'pred_joints'], (21, 3))
        pred_cam    = get_val(['pred_cam', 'cam'], (3,))
        
        # BBox 还原
        bbox = [0, 0, orig_w, orig_h]
        if 'pred_boxes' in outputs and len(outputs['pred_boxes']) > 0:
            pred_box = outputs['pred_boxes'][0].cpu().numpy()
            scale_x = orig_w / self.input_size[0]
            scale_y = orig_h / self.input_size[1]
            bbox = [
                pred_box[0] * scale_x,
                pred_box[1] * scale_y,
                pred_box[2] * scale_x,
                pred_box[3] * scale_y
            ]

        if pred_pose.shape == (48,):
             pred_pose = pred_pose.reshape(16, 3)

        return {
            'mano_pose': pred_pose,
            'mano_shape': pred_shape,
            'joint3d': pred_joints,
            'bbox': bbox,
            'weak_cam': pred_cam,
            'valid': True
        }

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"初始化数据读取器...")
    reader = FreiHandReader(DATA_ROOT, MANO_PATH)
    
    try:
        wrapper = WiLoRWrapper()
    except Exception as e:
        print(f"\n❌ 模型初始化崩溃: {e}")
        # 打印详细错误，方便调试
        traceback.print_exc()
        return
    
    print(f"🚀 开始推理 {reader.num_samples} 张图片...")
    print(f"结果将保存至: {SAVE_DIR}")
    
    error_count = 0
    
    for idx in tqdm(range(reader.num_samples)):
        try:
            save_name = f"image_wilor_{idx:08d}.pkl"
            save_path = os.path.join(SAVE_DIR, save_name)
            
            if os.path.exists(save_path):
                continue
            
            frame_data = reader.get_frame_data(idx)
            img_bgr = frame_data['image_bgr']
            
            pred = wrapper.predict(img_bgr)
            
            save_data = {
                'right': {
                    'mano_pose': pred['mano_pose'],
                    'mano_shape': pred['mano_shape'],
                    'joint3d':   pred['joint3d'],
                    'joint2d':   None,
                    'bbox':      pred['bbox'],
                    'hand_valid': pred['valid'],
                    'weak_cam':  pred['weak_cam']
                },
                'left': None
            }
            save_pickle(save_data, save_path)
            error_count = 0 
            
        except KeyboardInterrupt:
            print("\n用户中断")
            break
        except Exception as e:
            error_count += 1
            if error_count <= 3:
                print(f"\n[Error ID {idx}] {e}")
            continue

    print("\n✅ 所有推理任务完成！")

if __name__ == "__main__":
    main()