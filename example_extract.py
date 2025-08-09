import tarfile
import os
import shutil

def extract_video_from_tar(tar_path, video_name, output_dir):
    """
    从tar包中提取指定视频文件到目标文件夹
    :param tar_path: tar包路径
    :param video_name: 要提取的视频文件名（如 'xxx.mp4' 或带子路径）
    :param output_dir: 输出文件夹
    :return: 提取后的视频文件路径
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with tarfile.open(tar_path, 'r') as tar:
        # 查找tar包内的目标视频
        members = [m for m in tar.getmembers() if os.path.basename(m.name) == video_name]
        if not members:
            print(f"未找到视频: {video_name}")
            return None
        member = members[0]
        tar.extract(member, path=output_dir)
        src_path = os.path.join(output_dir, member.name)
        dst_path = os.path.join(output_dir, video_name)
        # 如果有子目录，移动到output_dir根目录
        if src_path != dst_path:
            shutil.move(src_path, dst_path)
            # 清理空目录
            subdir = os.path.dirname(src_path)
            if subdir != output_dir and os.path.exists(subdir):
                try:
                    os.removedirs(subdir)
                except Exception:
                    pass
        print(f"已提取: {dst_path}")
        return dst_path

# 按照要求，将最小的5个和最大的5个分别提取到不同的文件夹
def extract_videos_by_group(tar_path, min5_names, max5_names, base_output_dir):
    """
    将最小5个和最大5个视频分别提取到不同的文件夹
    :param tar_path: tar包路径
    :param min5_names: 最小5个视频文件名列表
    :param max5_names: 最大5个视频文件名列表
    :param base_output_dir: 基础输出目录
    """
    min_dir = os.path.join(base_output_dir, 'min5')
    max_dir = os.path.join(base_output_dir, 'max5')
    if not os.path.exists(min_dir):
        os.makedirs(min_dir)
    if not os.path.exists(max_dir):
        os.makedirs(max_dir)
    print("正在提取平均光流最小的5个视频到:", min_dir)
    for video_name in min5_names:
        extract_video_from_tar(tar_path, video_name, min_dir)
    print("正在提取平均光流最大的5个视频到:", max_dir)
    for video_name in max5_names:
        extract_video_from_tar(tar_path, video_name, max_dir)


# 修改主函数代码，让视频分别存储在min5和max5两个文件夹

if __name__ == "__main__":
    tar_path = '/pfs/yangyuanming/code1/data_2/panda_30m_part0001.tar'
    base_output_dir = './extracted_videos'

    # 假设你已经有了min5和max5的文件名列表
    min5_names = [
        'TLZE3Z1cFxc-0:07:13.566-0:07:18.100.mp4',
        'BDDqHBwZ25Y-0:02:03.490-0:02:08.361.mp4',
        'M2HSbTUL_YA-0:05:14.522-0:05:17.191.mp4',
        'Pv8SKjRYF8c-0:02:09.229-0:02:11.464.mp4',
        'EiOtjmJQ49E-0:05:16.749-0:05:20.787.mp4'
    ]
    max5_names = [
        'OUanWcM_vj4-0:04:10.216-0:04:12.485.mp4',
        'C_zHSJaDL2M-0:00:36.669-0:00:42.042.mp4',
        '3JDPYkONYn8-0:09:21.227-0:09:29.235.mp4',
        'UegxT8rPHrY-0:07:03.523-0:07:12.832.mp4',
        'CG7FOthZ5DQ-0:02:14.050-0:02:18.388.mp4'
    ]

    extract_videos_by_group(tar_path, min5_names, max5_names, base_output_dir)
