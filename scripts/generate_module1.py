import os
import openai

# 从 configs/config.txt 中读取 API KEY
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.txt')

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    api_key = f.read().strip()

openai.api_key = api_key

# 调整后的system prompt
system_prompt = """You are a helpful assistant that will transform the user's single-sentence request into a set of 30 one-sentence "short shot descriptions." 
Each description should depict a moment in the life of a classic American woman named Mary, starting from her birth and following her through 
significant life moments until her death. Each shot should be one sentence only, and should vividly describe a scene, including details of setting, 
objects, and emotional tone, similar in style and level of detail to the following example segments:

'Birth: A soft-focus shot of Mary as an infant in a hospital bassinet, her tiny hand grasping her mother's finger.
Childhood Play: Young Mary, with pigtails and a bright smile, jumping rope in a sunlit backyard, her friends cheering her on.
First Day of School: Mary, wearing a crisp new dress and backpack, nervously walking into her first classroom, her mother waving goodbye from the doorway.
...
Goodbye: A final shot of Mary’s empty rocking chair on the porch, the wind gently rocking it, with a sunset in the background, symbolizing the end of her journey.'

These 30 shots should form a coherent narrative, show Mary's progression through life, maintain her as a stable protagonist, 
and display scenes that change over time in a natural, life-spanning progression. Include descriptive elements of lighting, emotion, 
and environment similar to the example, making each scene feel cinematic, with subtle but meaningful changes as Mary ages and her life evolves.

[!!!Try to have a coherent main plot, a stable protagonist, and similar but changing scenes.]
"""

# 用户请求
user_prompt = "Describe a set of one-sentence prompts, 30 shots, describe a story of a classic American woman Mary's life, from birth to death."

# 调用GPT-4接口
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.7,
    max_tokens=3000,  # 提高max_tokens以确保完整输出
)

result_content = response['choices'][0]['message']['content']

# 新的保存路径
RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'tmp')
os.makedirs(RESULT_DIR, exist_ok=True)
RESULT_PATH = os.path.join(RESULT_DIR, 'short_shot_description.txt')

with open(RESULT_PATH, 'w', encoding='utf-8') as result_file:
    result_file.write(result_content)

print("Result saved to results/tmp/short_shot_description.txt")
