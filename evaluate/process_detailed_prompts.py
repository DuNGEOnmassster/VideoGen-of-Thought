import os
import json

# 需要遍历的目录列表
base_dir = '/storage/home/mingzhe/code/VideoGen-of-Thought/evaluate/10-Keyframes-and-Prompt-Pairs'

# 遍历每个子目录
subdirs = [
    'artist', 'botanist', 'chef', 'conservationist', 'cyclist', 
    'fashion_designer', 'mary_life_2', 'pianist', 'scientist', 'technologist'
]

# 遍历每个子目录并提取prompt内容
for subdir in subdirs:
    subdir_path = os.path.join(base_dir, subdir)
    
    # json文件路径
    json_file = os.path.join(subdir_path, 'image_prompt_pairs.json')
    
    # 读取json文件内容
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 提取prompt内容
    prompts = [item['prompt'] for item in data]
    
    # 保存到detailed_prompts.txt
    output_file = os.path.join(subdir_path, 'detailed_prompts.txt')
    with open(output_file, 'w') as f:
        for prompt in prompts:
            f.write(f"{prompt}\n")

    print(f"Prompts saved to {output_file}")
