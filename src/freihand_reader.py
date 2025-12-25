import os
import json
import cv2
import torch
import numpy as np
import smplx

class FreiHandReader:
    def __init__(self, base_path, mano_model_path):
        """
        初始化 FreiHand 数据读取器
        :param base_path: 指向 FreiHAND_pub_v2 文件夹的绝对路径
        :param mano_model_path: MANO_RIGHT.pkl 的路径
        """
        self.base_path = base_path
        # 根据你的截图5，图片路径在 base_path/training/rgb
        self.img_dir = os.path.join(base_path, 'training', 'rgb')
        
        # 检查图片目录是否存在
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"图片目录未找到: {self.img_dir}\n请检查 DATA_ROOT 设置是否正确。")

        # 加载 JSON 标注 (对应截图3中的文件)
        print(f"正在从 {base_path} 加载 FreiHand 标注文件...")
        
        try:
            with open(os.path.join(base_path, 'training_xyz.json'), 'r') as f:
                self.xyz_list = json.load(f) # 3D 关节点
            with open(os.path.join(base_path, 'training_K.json'), 'r') as f:
                self.K_list = json.load(f)   # 相机内参
            with open(os.path.join(base_path, 'training_mano.json'), 'r') as f:
                self.mano_list = json.load(f)# MANO 参数
        except FileNotFoundError as e:
            raise FileNotFoundError(f"缺少关键 JSON 文件 (training_xyz.json 等)。请确认 DATA_ROOT 指向了 FreiHAND_pub_v2 文件夹。\n错误信息: {e}")
            
        self.num_samples = len(self.xyz_list)
        print(f"标注加载完成，共 {self.num_samples} 个样本。")

        # 初始化 MANO Layer
        if not os.path.exists(mano_model_path):
             raise FileNotFoundError(f"MANO 模型文件未找到: {mano_model_path}\n请去官网下载 MANO_RIGHT.pkl 并修改路径。")

        # 初始化 MANO Layer (直接使用 MANO 类，避开 create 函数的目录检查)
        self.mano_layer = smplx.MANO(
            model_path=mano_model_path, # 直接传入你的 .pkl 文件完整路径
            is_rhand=True,
            use_pca=False,
            flat_hand_mean=False
        )

    def get_frame_data(self, idx):
        if idx >= self.num_samples:
            raise IndexError("Index out of bounds")

        # 1. 读取图像
        img_name = f"{idx:08d}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {img_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. 读取内参
        K = np.array(self.K_list[idx])

        # 3. 读取 MANO 参数 (【核心修复】)
        # 先转 numpy，然后强制拍平 (flatten)，确保变成一维数组
        raw_params = np.array(self.mano_list[idx], dtype=np.float32).flatten()
        
        # 拆解参数 (共 61 位)
        mano_pose = raw_params[:48]      # [0~48] 姿态
        mano_betas = raw_params[48:58]   # [48~58] 形状
        
        # # (调试打印，确认修复后可删除)
        # if idx == 0:
        #     print(f"\n✅ [修复确认] ID: {idx}")
        #     print(f"原始参数总长度: {len(raw_params)} (应为 61)")
        #     print(f"Pose 长度: {len(mano_pose)} (应为 48)")
        #     print(f"Betas 长度: {len(mano_betas)} (应为 10)")
        #     print(f"========================\n")

        # 4. 获取 3D 关节点
        joints_3d = np.array(self.xyz_list[idx])

        return {
            'image': img_rgb,
            'image_bgr': img,
            'K': K,
            'mano_pose': mano_pose,
            'mano_betas': mano_betas,
            'joints_3d': joints_3d,
            'id': idx
        }

    def get_mesh_verts(self, mano_pose, mano_betas, gt_joints_3d):
            """
            利用 GT 3D 关节来自动校准 Mesh 的位置
            """
            # 1. 准备 tensor
            torch_pose = torch.tensor(mano_pose, dtype=torch.float32).unsqueeze(0)
            torch_betas = torch.tensor(mano_betas, dtype=torch.float32).unsqueeze(0)
            
            # 2. 生成“零点”状态的 Mesh 和 Joints
            output = self.mano_layer(global_orient=torch_pose[:, :3],
                                    hand_pose=torch_pose[:, 3:],
                                    betas=torch_betas)
            
            mano_verts = output.vertices[0].detach().numpy() # (778, 3)
            mano_joints = output.joints[0].detach().numpy()  # (21, 3)
            
            # 3. 计算偏移量 (Translation)
            # 逻辑：目标中心(GT) - 当前中心(MANO) = 需要移动的向量
            # 注意：这里用手腕点 (第0个点) 对齐，通常比用重心对齐更准
            target_root = gt_joints_3d[0]
            current_root = mano_joints[0]
            translation = target_root - current_root
            
            # 4. 把 Mesh 搬运过去
            final_verts = mano_verts + translation
            final_joints = mano_joints + translation
            
            return final_verts, final_joints