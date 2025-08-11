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

# 支持任意数量的最小N和最大N视频提取到不同的文件夹
def extract_videos_by_group(tar_path, min_names, max_names, base_output_dir, min_label='min', max_label='max'):
    """
    将最小N个和最大N个视频分别提取到不同的文件夹
    :param tar_path: tar包路径
    :param min_names: 最小N个视频文件名列表
    :param max_names: 最大N个视频文件名列表
    :param base_output_dir: 基础输出目录
    :param min_label: 最小组文件夹名（默认'min'）
    :param max_label: 最大组文件夹名（默认'max'）
    """
    min_dir = os.path.join(base_output_dir, f'{min_label}{len(min_names)}')
    max_dir = os.path.join(base_output_dir, f'{max_label}{len(max_names)}')
    if not os.path.exists(min_dir):
        os.makedirs(min_dir)
    if not os.path.exists(max_dir):
        os.makedirs(max_dir)
    print(f"正在提取平均光流最小的{len(min_names)}个视频到:", min_dir)
    for video_name in min_names:
        extract_video_from_tar(tar_path, video_name, min_dir)
    print(f"正在提取平均光流最大的{len(max_names)}个视频到:", max_dir)
    for video_name in max_names:
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

    min10_names = [
        'TLZE3Z1cFxc-0:07:13.566-0:07:18.100.mp4',
        'BDDqHBwZ25Y-0:02:03.490-0:02:08.361.mp4',
        'M2HSbTUL_YA-0:05:14.522-0:05:17.191.mp4',
        'Pv8SKjRYF8c-0:02:09.229-0:02:11.464.mp4',
        'EiOtjmJQ49E-0:05:16.749-0:05:20.787.mp4',
        'IP4tmXKmqlk-0:00:06.520-0:00:19.000.mp4',
        'BK2sa7ZDSlo-0:15:03.040-0:15:11.480.mp4',
        '7RDj2FoGlBs-0:02:27.105-0:02:31.401.mp4',
        'MXI5HHS9mqs-0:00:47.814-0:00:52.519.mp4',
        'MmPUfR_Gfhw-0:08:34.766-0:08:38.433.mp4',
    ]

    max10_names = [
        'SsWQXcfKQrw-0:11:30.222-0:11:38.230.mp4',
        '2sRm6rBsjVI-0:04:19.892-0:04:46.119.mp4',
        'CNxCYhi9OgY-0:03:29.600-0:03:35.616.mp4',
        'KKx4cUZEyMI-0:03:43.633-0:03:47.233.mp4',
        'GG71p1r91cg-0:06:57.125-0:07:06.843.mp4',
        'LwF1Mj-fG-Y-0:03:22.200-0:03:25.080.mp4',
        'CG7FOthZ5DQ-0:02:14.050-0:02:18.388.mp4',
        'UegxT8rPHrY-0:07:03.523-0:07:12.832.mp4',
        '3JDPYkONYn8-0:09:21.227-0:09:29.235.mp4',
        'C_zHSJaDL2M-0:00:36.669-0:00:42.042.mp4',
        'OUanWcM_vj4-0:04:10.216-0:04:12.485.mp4',
    ]

    extract_videos_by_group(tar_path, min5_names, max5_names, base_output_dir)
    extract_videos_by_group(tar_path, min10_names, max10_names, base_output_dir)
