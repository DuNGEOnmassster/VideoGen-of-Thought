import os
import openai

# 从 configs/config.txt 中读取 API KEY
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.txt')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    api_key = f.read().strip()

openai.api_key = api_key

# 第一个 GPT 的 system prompt
system_prompt_1 = """You are a helpful assistant that will transform the user's single-sentence request into a set of 30 one-sentence "short shot descriptions." 
Each description should depict a moment in the life of a classic American woman named Mary, starting from her birth and following her through 
significant life moments until her death. Each shot should be one sentence only, and should vividly describe a scene, including details of setting, 
objects, and emotional tone, similar in style and level of detail to the given example segments.

These 30 shots should form a coherent narrative, show Mary's progression through life, maintain her as a stable protagonist, 
and display scenes that change over time in a natural, life-spanning progression. Include descriptive elements of lighting, emotion, 
and environment, making each scene feel cinematic, with subtle but meaningful changes as Mary ages and her life evolves.

[!!!Try to have a coherent main plot, a stable protagonist, and similar but changing scenes.]"""

user_prompt_1 = "Describe a set of one-sentence prompts, 30 shots, describe a story of a classic American woman Mary's life, from birth to death."

# 调用第一个 GPT，生成 short_shot_description
response_1 = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt_1},
        {"role": "user", "content": user_prompt_1}
    ],
    temperature=0.7,
    max_tokens=3000
)

result_content_1 = response_1['choices'][0]['message']['content']

# 保存 short_shot_description.txt
RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'tmp')
os.makedirs(RESULT_DIR, exist_ok=True)
SHORT_DESC_PATH = os.path.join(RESULT_DIR, 'short_shot_description.txt')
with open(SHORT_DESC_PATH, 'w', encoding='utf-8') as result_file:
    result_file.write(result_content_1)

print("short_shot_description saved to results/tmp/short_shot_description.txt")


# 第二个 GPT 的 system prompt
system_prompt_2 = """You are a Hollywood Movie Level Director. You have been given a set of one-sentence short shot descriptions that describe moments in Mary’s life, from birth to death. For each shot description, you need to break it down into a structured prompt focusing on five key aspects of the scene:

- character: Describe the appearance, age, expression, and key attributes of Mary (and others if relevant).
- background: Describe the setting, location, and notable background elements.
- relation: Describe the emotional or narrative relationship between Mary and other elements in the scene.
- camera pose: Describe the camera angle, framing, and perspective to visualize the shot.
- HDR description: Describe the lighting, atmosphere, mood, and visual quality in high detail, as if shooting a cinematic movie scene.

I need you to be a Hollywood Movie Level Director. You need to write each prompt like a movie frame description. Use the form:

"character: xxx, background: xxx, relation: xxx, camera pose: xxx, HDR description: xxx."

You should reply to me with a JSON array of objects, where each object corresponds to one shot from the original list of short shot descriptions. For each object, include:

{
  "ip_img_path": "tmp",
  "prompt": "character:..., background:..., relation:..., camera pose:..., HDR description:..."
}

[!!!Be clear, be detailed, be loyal to content]"""


# 从 short_shot_description.txt 中读取生成的描述
with open(SHORT_DESC_PATH, 'r', encoding='utf-8') as f:
    short_descriptions = f.read().strip()

# 调用第二个 GPT，将 short_shot_description 转化为 JSON 格式
response_2 = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt_2},
        {"role": "user", "content": short_descriptions}
    ],
    temperature=0.7,
    max_tokens=7000
)

image_prompt_content = response_2['choices'][0]['message']['content']

# 保存 image_prompt_pairs.json
IMAGE_PROMPT_PATH = os.path.join(RESULT_DIR, 'image_prompt_pairs.json')
with open(IMAGE_PROMPT_PATH, 'w', encoding='utf-8') as json_file:
    json_file.write(image_prompt_content)

print("image_prompt_pairs saved to results/tmp/image_prompt_pairs.json")
