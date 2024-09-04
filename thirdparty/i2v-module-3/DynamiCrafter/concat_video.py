import os
import re
import subprocess

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# 定义路径
video_dir = 'results/dynamicrafter_kleber_1024_seed123/samples_separate/'
output_file = 'results/dynamicrafter_kleber_1024_seed123/full_video.mp4'

# 获取所有视频文件，按自然顺序排序
video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')], key=natural_sort_key)

# 创建一个临时文件列表，以供ffmpeg使用
with open('file_list.txt', 'w') as f:
    for video_file in video_files:
        f.write(f"file '{os.path.join(video_dir, video_file)}'\n")

# 使用ffmpeg将视频拼接成一个完整的视频
subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'file_list.txt', '-c', 'copy', output_file])

# 删除临时文件
os.remove('file_list.txt')

print(f"Full video saved as {output_file}")
