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

# 提取Caption并写入test_prompts.txt，同时复制图片文件
output_file = os.path.join(output_dir, 'test_prompts.txt')
with open(output_file, 'w') as f:
    for i, caption in enumerate(captions):
        # 写入Caption到test_prompts.txt
        f.write(caption['Caption'] + '\n')
        
        # 复制对应的_start.png文件到output_dir
        frame_file = f"scene_{i+1}_start.png"
        src_file = os.path.join(frames_dir, frame_file)
        dest_file = os.path.join(output_dir, frame_file)
        if os.path.exists(src_file):
            shutil.copy(src_file, dest_file)
        else:
            print(f"Warning: {src_file} not found and was skipped.")

print(f"Successfully created {output_file} and copied images to {output_dir}")
