import os
import os.path as osp
import pickle
import cv2

class FreiHandUnifiedReader:
    """
    统一数据接口：读取 FreiHand 图片 + WiLoR 预测标签
    符合任务文档要求的最终交付格式。
    """
    def __init__(self, data_dir, mode='train'):
        self.data_dir = data_dir
        self.img_dir = osp.join(data_dir, 'training', 'rgb')
        # 标签目录指向我们生成的 output_wilor_pkl
        self.label_dir = osp.join(data_dir, 'output_wilor_pkl')
        
        # 扫描有多少个 .pkl 文件
        if not osp.exists(self.label_dir):
            print(f"[Warning] 标签目录不存在: {self.label_dir}")
            self.file_list = []
        else:
            self.file_list = sorted([f for f in os.listdir(self.label_dir) if f.endswith('.pkl')])
            
        print(f"[UnifiedReader] 初始化完成，加载了 {len(self.file_list)} 个样本。")

    def __len__(self):
        return len(self.file_list)

    def fetch_frame_data(self, idx, get_img=True):
        """
        获取单帧数据
        """
        # 1. 解析文件名
        # pkl: image_wilor_00000123.pkl -> img: 00000123.jpg
        pkl_name = self.file_list[idx]
        file_id = pkl_name.split('_')[-1].replace('.pkl', '')
        
        img_path = osp.join(self.img_dir, f"{file_id}.jpg")
        lbl_path = osp.join(self.label_dir, pkl_name)

        # 2. 读取图像
        img = None
        if get_img:
            img = cv2.imread(img_path) # BGR
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 转 RGB
            else:
                raise FileNotFoundError(f"找不到图片: {img_path}")

        # 3. 读取标注 (PKL)
        with open(lbl_path, 'rb') as f:
            pred_data = pickle.load(f)

        # 4. 返回标准格式字典
        return {
            'image': img,
            'right': pred_data['right'],
            'left':  pred_data['left']
        }

if __name__ == "__main__":
    # 简单测试
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_ROOT = os.path.join(PROJECT_ROOT, "FreiHAND/FreiHAND_pub_v2")
    
    reader = FreiHandUnifiedReader(DATA_ROOT)
    if len(reader) > 0:
        data = reader.fetch_frame_data(0)
        print("测试成功！读取到的 Keys:", data['right'].keys())
    else:
        print("暂无数据，请先运行推理脚本。")