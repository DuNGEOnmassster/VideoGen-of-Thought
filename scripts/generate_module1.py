import os
import openai

# 从 configs/config.txt 中读取 API KEY
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.txt')

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    api_key = f.read().strip()

openai.api_key = api_key

# 要向 GPT-4 提交的请求
prompt = "Describe a set of one-sentence prompts, 30 shots, describe a story of a classic American woman Mary's life, from birth to death."

# 调用GPT-4接口
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role":"user","content": prompt}],
    temperature=0.7,  # 可根据需求调整
    max_tokens=1500,  # 根据需求调整，确保足够长以容纳30句
)

result_content = response['choices'][0]['message']['content']

# 将结果保存到 results/module1.txt
RESULT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'module1.txt')
with open(RESULT_PATH, 'w', encoding='utf-8') as result_file:
    result_file.write(result_content)

print("Result saved to results/module1.txt")
