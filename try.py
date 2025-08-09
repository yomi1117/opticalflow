import cv2
import numpy as np

# 加载视频
video_path = '/pfs/yangyuanming/code1/data_1/1pgBC7hZw5s-0:03:51.333-0:03:58.033.mp4'
# video_path = '/pfs/yangyuanming/code1/data_1/2caGDBwNSj0-0:19:53.792-0:19:58.063.mp4'
cap = cv2.VideoCapture(video_path)

# 获取视频的第一帧
ret, frame1 = cap.read()
if not ret:
    print("无法读取视频第一帧")
    cap.release()
    exit()

# 获取帧宽高和帧率
frame_height, frame_width = frame1.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)

# 定义输出视频的编码器和输出文件名
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('optical_flow_result.mp4', fourcc, fps, (frame_width, frame_height))

# 将第一帧转换为灰度图像
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# 创建一个掩模用于显示光流
hsv_mask = np.zeros_like(frame1)
hsv_mask[..., 1] = 255

motion_intensity = []

while(cap.isOpened()):
    ret, frame2 = cap.read()
    if not ret:
        break

    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 计算光流 (Farneback 方法)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 计算运动的角度和大小
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 显示光流图
    hsv_mask[..., 0] = ang * 180 / np.pi / 2
    hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

    # 计算运动程度（可以通过光流的幅度均值来衡量）
    motion_intensity.append(np.mean(mag))

    # 写入到输出视频
    out.write(flow_rgb)

    # 更新上一帧
    prvs = next

cap.release()
out.release()
cv2.destroyAllWindows()

# 打印动态程度分析结果
print(f"视频平均运动强度: {np.mean(motion_intensity)}")
