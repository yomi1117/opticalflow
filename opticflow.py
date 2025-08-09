import cv2
import numpy as np
import tarfile
import os
import tempfile
from tqdm import tqdm

class OpticalFlowProcessor:
    """光流处理器 - 简洁版本"""
    
    def __init__(self):
        ...
    

    def process_batch(self, video_tar_dir, output_path_dir):
        """
        处理tar包中的所有mp4视频，计算每个视频的平均光流强度，
        返回平均光流最大和最小的各5个样本的文件名和得分。
        """
        tar_path = video_tar_dir  # 例如: '/pfs/yangyuanming/code1/data_2/panda_30m_part0001.tar'
        results = []

        # 创建临时目录用于解压视频
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(tar_path, 'r') as tar:
                # 只提取mp4文件
                mp4_members = [m for m in tar.getmembers() if m.name.endswith('.mp4')]
                for member in tqdm(mp4_members):
                    tar.extract(member, path=tmpdir)
                    video_file_path = os.path.join(tmpdir, member.name)
                    # 输出文件名
                    output_file = os.path.join(output_path_dir, os.path.basename(member.name) + '_flow.mp4')
                    # 调用process处理
                    avg_motion = self.process(video_file_path, output_file)
                    print(f"平均光流强度: {avg_motion:.2f}")
                    results.append({
                        'filename': member.name,
                        'avg_motion': avg_motion
                    })

        # 按平均光流排序
        results_sorted = sorted(results, key=lambda x: x['avg_motion'])
        min5 = results_sorted[:5]
        max5 = results_sorted[-5:][::-1]  # 最大的5个，倒序

        print("平均光流最小的5个样本：")
        for item in min5:
            print(f"{item['filename']} : {item['avg_motion']:.4f}")

        print("平均光流最大的5个样本：")
        for item in max5:
            print(f"{item['filename']} : {item['avg_motion']:.4f}")

        # 返回结果
        return {
            'min5': min5,
            'max5': max5
        }


    def process(self, video_path, output_path='optical_flow_result.mp4'):
        """处理视频并生成光流结果"""
        self.motion_intensity = []
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # 获取视频信息
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"视频信息:")
        print(f"  总帧数: {total_frames}")
        print(f"  帧率: {fps:.2f} fps")
        print(f"  时长: {duration:.2f} 秒")
        
        # 读取第一帧
        ret, frame1 = self.cap.read()
        if not ret:
            print("无法读取视频")
            return
            
        # 设置输出
        height, width = frame1.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 初始化
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv_mask = np.zeros_like(frame1)
        hsv_mask[..., 1] = 255
        
        processed_frames = 0
        
        # 处理每一帧 - 使用8 fps采样
        target_fps = 8
        frame_interval = max(1, int(fps / target_fps))  # 计算跳帧间隔
        frame_count = 0
        # 计算帧间时间差
        if fps > 0 and frame_interval > 0:
            frame_time_delta = frame_interval / fps
        else:
            frame_time_delta = 1.0  # 防止除零
        
        print(f"  目标处理帧率: {target_fps} fps")
        print(f"  跳帧间隔: 每{frame_interval}帧处理1帧")
        print(f"  帧间时间差: {frame_time_delta:.4f} 秒")
        
        while True:
            ret, frame2 = self.cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # 只处理指定间隔的帧
            if frame_count % frame_interval != 0:
                continue
                
            next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # 计算光流
            flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # 转换为极坐标
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # 生成光流可视化
            hsv_mask[..., 0] = ang * 180 / np.pi / 2
            hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            flow_rgb = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
            
            # 记录运动强度
            self.motion_intensity.append(np.mean(mag))
            
            # 输出 - 使用原始fps保持视频时长一致
            out.write(flow_rgb)
            prvs = next_frame
            processed_frames += 1
            
        # 清理
        self.cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"光流处理完成:")
        print(f"  处理帧数: {processed_frames}")
        print(f"  处理时长: {processed_frames/fps:.2f} 秒")
        
        # 计算平均运动强度（除以帧间时间差）
        if len(self.motion_intensity) > 0:
            avg_motion_intensity = np.mean(self.motion_intensity) / frame_time_delta
        else:
            avg_motion_intensity = 0.0
        
        return avg_motion_intensity
    
    def get_motion_analysis(self):
        """获取运动分析结果"""
        # 需要考虑帧间时间差
        if hasattr(self, 'motion_intensity') and len(self.motion_intensity) > 0:
            # 重新计算帧间时间差
            if hasattr(self, 'cap') and hasattr(self, 'video_path'):
                cap_tmp = cv2.VideoCapture(self.video_path)
                fps = cap_tmp.get(cv2.CAP_PROP_FPS)
                target_fps = 8
                frame_interval = max(1, int(fps / target_fps))
                if fps > 0 and frame_interval > 0:
                    frame_time_delta = frame_interval / fps
                else:
                    frame_time_delta = 1.0
                cap_tmp.release()
            else:
                frame_time_delta = 1.0
            avg_intensity = np.mean(self.motion_intensity) / frame_time_delta
            max_intensity = np.max(self.motion_intensity) / frame_time_delta
            min_intensity = np.min(self.motion_intensity) / frame_time_delta
            std_intensity = np.std(self.motion_intensity) / frame_time_delta
        else:
            avg_intensity = 0.0
            max_intensity = 0.0
            min_intensity = 0.0
            std_intensity = 0.0
        return {
            'avg_intensity': avg_intensity,
            'max_intensity': max_intensity,
            'min_intensity': min_intensity,
            'std_intensity': std_intensity
        }

# 使用示例
if __name__ == "__main__":
    # 创建处理器
    processor = OpticalFlowProcessor()
    
    # # 处理视频
    # avg_motion = processor.process('/pfs/yangyuanming/code1/data_1/1pgBC7hZw5s-0:03:51.333-0:03:58.033.mp4', 'result_class.mp4')
    # print(f"平均运动强度: {avg_motion:.2f}")
    
    # # 获取详细分析
    # analysis = processor.get_motion_analysis()
    # print(f"运动分析: {analysis}")

    # 处理批量视频
    processor.process_batch('/pfs/yangyuanming/code1/data_2/panda_30m_part0001.tar', '/pfs/yangyuanming/code1/data_1/optical_flow_result')

