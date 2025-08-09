"""
光流处理器 - 简洁优雅版本
作者: 重构版本
功能: 批量处理视频文件，计算光流强度并生成可视化结果
"""

import cv2
import numpy as np
import tarfile
import os
import tempfile
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """视频信息数据类"""
    filename: str
    total_frames: int
    fps: float
    duration: float
    width: int
    height: int


@dataclass
class ProcessResult:
    """处理结果数据类"""
    filename: str
    avg_motion: float
    processed_frames: int
    processing_time: float


class OpticalFlowProcessor:
    """
    光流处理器 - 简洁版本
    
    功能:
    - 处理单个视频文件
    - 批量处理tar包中的视频
    - 计算光流强度并生成可视化
    - 支持多进程并行处理
    """
    
    def __init__(self, target_fps: int = 8, flow_params: Optional[Dict] = None):
        """
        初始化光流处理器
        
        Args:
            target_fps: 目标处理帧率
            flow_params: 光流计算参数
        """
        self.target_fps = target_fps
        self.flow_params = flow_params or {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }
        
    def _get_video_info(self, video_path: Union[str, Path]) -> Optional[VideoInfo]:
        """获取视频基本信息"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}")
            return None
            
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            return VideoInfo(
                filename=Path(video_path).name,
                total_frames=total_frames,
                fps=fps,
                duration=duration,
                width=width,
                height=height
            )
        finally:
            cap.release()
    
    def _calculate_optical_flow(self, prev_frame: np.ndarray, 
                              curr_frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        计算两帧之间的光流
        
        Returns:
            Tuple[光流可视化图像, 运动强度]
        """
        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, curr_frame, None, **self.flow_params
        )
        
        # 转换为极坐标
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # 创建HSV可视化
        hsv = np.zeros((prev_frame.shape[0], prev_frame.shape[1], 3), dtype=np.uint8)
        hsv[..., 1] = 255  # 饱和度设为最大
        hsv[..., 0] = angle * 180 / np.pi / 2  # 色调表示方向
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # 亮度表示强度
        
        # 转换为BGR用于输出
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 计算运动强度
        motion_intensity = np.mean(magnitude)
        
        return flow_bgr, motion_intensity
    
    def process_single_video(self, video_path: Union[str, Path], 
                           output_path: Optional[Union[str, Path]] = None) -> Optional[ProcessResult]:
        """
        处理单个视频文件
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径（可选）
            
        Returns:
            处理结果或None（如果失败）
        """
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"视频文件不存在: {video_path}")
            return None
            
        # 获取视频信息
        video_info = self._get_video_info(video_path)
        if not video_info:
            return None
            
        logger.info(f"处理视频: {video_info.filename}")
        logger.info(f"  分辨率: {video_info.width}x{video_info.height}")
        logger.info(f"  帧率: {video_info.fps:.2f} fps")
        logger.info(f"  时长: {video_info.duration:.2f} 秒")
        
        # 打开视频
        cap = cv2.VideoCapture(str(video_path))
        
        # 设置输出（如果指定）
        video_writer = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_path), fourcc, video_info.fps, 
                (video_info.width, video_info.height)
            )
        
        # 计算处理参数
        frame_interval = max(1, int(video_info.fps / self.target_fps))
        frame_time_delta = frame_interval / video_info.fps if video_info.fps > 0 else 1.0
        
        logger.info(f"  处理间隔: 每{frame_interval}帧处理1帧")
        
        # 读取第一帧
        ret, first_frame = cap.read()
        if not ret:
            logger.error("无法读取第一帧")
            cap.release()
            return None
            
        prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        motion_intensities = []
        processed_frames = 0
        frame_count = 0
        
        # 处理视频帧
        with tqdm(total=video_info.total_frames, desc="处理进度") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                pbar.update(1)
                
                # 跳帧处理
                if frame_count % frame_interval != 0:
                    continue
                    
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 计算光流
                flow_vis, motion_intensity = self._calculate_optical_flow(prev_gray, curr_gray)
                motion_intensities.append(motion_intensity)
                
                # 写入输出视频
                if video_writer:
                    video_writer.write(flow_vis)
                    
                prev_gray = curr_gray
                processed_frames += 1
        
        # 清理资源
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # 计算结果
        avg_motion = np.mean(motion_intensities) / frame_time_delta if motion_intensities else 0.0
        processing_time = processed_frames / video_info.fps
        
        logger.info(f"处理完成: {processed_frames}帧, 平均运动强度: {avg_motion:.4f}")
        
        return ProcessResult(
            filename=video_info.filename,
            avg_motion=avg_motion,
            processed_frames=processed_frames,
            processing_time=processing_time
        )
    
    def process_batch_from_tar(self, tar_path: Union[str, Path], 
                             output_dir: Union[str, Path], 
                             max_workers: int = 200) -> Dict[str, List[ProcessResult]]:
        """
        从tar包中批量处理视频文件
        
        Args:
            tar_path: tar包路径
            output_dir: 输出目录
            max_workers: 最大并行工作进程数
            
        Returns:
            包含最大和最小运动强度样本的字典
        """
        tar_path = Path(tar_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not tar_path.exists():
            logger.error(f"tar包不存在: {tar_path}")
            return {'min5': [], 'max5': []}
        
        logger.info(f"开始批量处理: {tar_path}")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"最大工作进程: {max_workers}")
        
        results = []
        
        # 临时目录处理
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 提取MP4文件
            with tarfile.open(tar_path, 'r') as tar:
                mp4_members = [m for m in tar.getmembers() if m.name.endswith('.mp4')]
                logger.info(f"发现 {len(mp4_members)} 个MP4文件")
                
                # 提取文件
                for member in tqdm(mp4_members, desc="提取文件"):
                    tar.extract(member, path=temp_path)
                
                # 多进程处理
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_to_video = {}
                    
                    # 提交任务
                    for member in mp4_members:
                        video_file = temp_path / member.name
                        output_file = output_dir / f"{Path(member.name).stem}_flow.mp4"
                        
                        future = executor.submit(
                            self.process_single_video, video_file, output_file
                        )
                        future_to_video[future] = member.name
                    
                    # 收集结果
                    for future in tqdm(as_completed(future_to_video), 
                                     total=len(future_to_video), desc="处理视频"):
                        try:
                            result = future.result()
                            if result:
                                results.append(result)
                                logger.info(f"完成: {result.filename}, 强度: {result.avg_motion:.4f}")
                        except Exception as e:
                            video_name = future_to_video[future]
                            logger.error(f"处理失败 {video_name}: {e}")
        
        # 排序和保存结果
        return self._save_and_analyze_results(results, output_dir)
    
    def _save_and_analyze_results(self, results: List[ProcessResult], 
                                output_dir: Path) -> Dict[str, List[ProcessResult]]:
        """保存并分析处理结果"""
        if not results:
            logger.warning("没有有效的处理结果")
            return {'min5': [], 'max5': []}
        
        # 按运动强度排序
        results_sorted = sorted(results, key=lambda x: x.avg_motion)
        
        # 保存完整结果
        result_file = output_dir / 'all_results_sorted.txt'
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("filename,avg_motion,processed_frames,processing_time\n")
            for result in results_sorted:
                f.write(f"{result.filename},{result.avg_motion:.6f},"
                       f"{result.processed_frames},{result.processing_time:.2f}\n")
        
        logger.info(f"完整结果已保存到: {result_file}")
        
        # 获取极值样本
        min5 = results_sorted[:5]
        max5 = results_sorted[-5:][::-1]  # 倒序，最大的在前
        
        # 打印结果摘要
        self._print_summary(min5, max5, len(results))
        
        return {'min5': min5, 'max5': max5}
    
    def _print_summary(self, min5: List[ProcessResult], 
                      max5: List[ProcessResult], total_count: int) -> None:
        """打印结果摘要"""
        logger.info(f"\n{'='*60}")
        logger.info(f"批量处理完成! 总计处理 {total_count} 个视频")
        logger.info(f"{'='*60}")
        
        print("\n🔽 运动强度最小的5个样本:")
        for i, result in enumerate(min5, 1):
            print(f"  {i}. {result.filename:<30} | 强度: {result.avg_motion:.4f}")
        
        print("\n🔼 运动强度最大的5个样本:")
        for i, result in enumerate(max5, 1):
            print(f"  {i}. {result.filename:<30} | 强度: {result.avg_motion:.4f}")
        
        print(f"\n{'='*60}")


def main():
    """主函数 - 使用示例"""
    # 创建处理器
    processor = OpticalFlowProcessor(target_fps=8)
    
    # 设置路径
    tar_path = '/pfs/yangyuanming/code1/data_2/panda_30m_part0001.tar'
    output_dir = '/pfs/yangyuanming/code1/data_1/optical_flow_result'
    
    try:
        # 批量处理
        results = processor.process_batch_from_tar(tar_path, output_dir, max_workers=200)
        
        # 可以进一步处理结果...
        logger.info("所有处理任务完成!")
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()