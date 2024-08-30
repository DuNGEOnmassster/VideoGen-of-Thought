import os
import shutil
import json

# 定义路径
frames_dir = 'data_v0/frames/'
caption_file = 'data_v0/Caption.json'
output_dir = 'prompts/Kleber/'

# 创建输出目录，如果不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取Caption.json文件
with open(caption_file, 'r') as f:
    captions = json.load(f)

# 初始化保存frames的列表
frames = []
for filename in sorted(os.listdir(frames_dir)):
    if filename.endswith('_start.png'):
        scene_id = filename.split('_')[0]
        frames.append((scene_id, filename))

# 提取对应的Caption并写入test_prompts.txt，同时复制图片文件
output_file = os.path.join(output_dir, 'test_prompts.txt')
with open(output_file, 'w') as f:
    for scene_id, frame in frames:
        # 找到与frame对应的Caption
        caption = next((item['Caption'] for item in captions if item['id'] == scene_id), None)
        if caption:
            f.write(caption + '\n')
        
        # 复制_start.png文件到output_dir
        src_file = os.path.join(frames_dir, frame)
        dest_file = os.path.join(output_dir, frame)
        shutil.copy(src_file, dest_file)

print(f"Successfully created {output_file} and copied images to {output_dir}")
