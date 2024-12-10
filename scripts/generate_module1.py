import os
import openai
import json

# Read API Key
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.txt')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    api_key = f.read().strip()

openai.api_key = api_key

# The original user prompt (renamed from user_prompt_1 to story_user_prompt)
story_user_prompt = "Describe a set of one-sentence prompts, 30 shots, describe a story of a classic American woman Mary's life, from birth to death."

# Extracting story_name from story_user_prompt:
# For simplicity, we know the protagonist is Mary from the prompt.
# In a more complex scenario, you could use NLP to parse the name,
# but here we hardcode that the story_name = "Mary".
story_name = "Mary"

########################################
# 1. First GPT: Generate short_shot_description.txt
########################################

system_prompt_1 = """You are a helpful assistant that will transform the user's single-sentence request into a set of 30 one-sentence "short shot descriptions." 
Each description should depict a moment in the life of a classic American woman named Mary, starting from her birth and following her through 
significant life moments until her death. Each shot should be one sentence only, and should vividly describe a scene, including details of setting, 
objects, and emotional tone.

These 30 shots should form a coherent narrative, show Mary's progression through life, maintain her as a stable protagonist, 
and display scenes that change over time in a natural, life-spanning progression. Include descriptive elements of lighting, emotion, 
and environment, making each scene feel cinematic, with subtle but meaningful changes as Mary ages and her life evolves.

[!!!Try to have a coherent main plot, a stable protagonist, and similar but changing scenes.]"""

response_1 = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt_1},
        {"role": "user", "content": story_user_prompt}
    ],
    temperature=0.7,
    max_tokens=3000
)

result_content_1 = response_1['choices'][0]['message']['content']

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'tmp')
os.makedirs(RESULT_DIR, exist_ok=True)
SHORT_DESC_PATH = os.path.join(RESULT_DIR, 'short_shot_description.txt')

with open(SHORT_DESC_PATH, 'w', encoding='utf-8') as result_file:
    result_file.write(result_content_1)

print("short_shot_description saved to results/tmp/short_shot_description.txt")

########################################
# 2. Second GPT: Generate avatar_prompt.json for the specified life stages
########################################

system_prompt_2 = f"""You are a master character concept artist and fashion designer. You have been given a narrative (30 short shot descriptions) about a protagonist's life. 
From the user prompt, the protagonist's name is {story_name} (story_name = "{story_name}"). 
You need to create avatar images for this protagonist at exactly five distinct life stages: Child, Teen, Mid, Mid-Elder, and Old.

Produce exactly 5 JSON objects, each representing {story_name} at one of these five life stages. For each object:
- "ip_image_path": use the format "data/{story_name}/avatar_{story_name}_<stage>.jpg" replacing <stage> with one of the specified stages.
- "prompt": A multi-line string describing the scene from five angles:
  - character: Appearance, age, clothing, facial expression, etc.
  - background: The setting and environment behind {story_name}.
  - relation: How {story_name} relates to her environment, emotions, or others at that stage.
  - camera pose: Cinematic angle and framing.
  - HDR description: Lighting, atmosphere, and visual qualities, ideally in high detail (e.g., 8K HDR).

[!!!Be consistent with the narrative, choose details that reflect each stage of {story_name}'s life, and ensure each prompt is detailed and visually evocative.]

Output a JSON array of these 5 objects.
"""

with open(SHORT_DESC_PATH, 'r', encoding='utf-8') as f:
    short_descriptions_for_avatar = f.read().strip()

AVATAR_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', story_name)
os.makedirs(AVATAR_DIR, exist_ok=True)
# We'll store avatar_prompt.json not inside avatar DIR (as per original instructions),
# but inside data/tmp to keep consistency. If needed, we can change location.
DATA_TMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'tmp')
os.makedirs(DATA_TMP_DIR, exist_ok=True)
AVATAR_PROMPT_PATH = os.path.join(DATA_TMP_DIR, 'avatar_prompt.json')

response_2 = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt_2},
        {"role": "user", "content": short_descriptions_for_avatar}
    ],
    temperature=0.7,
    max_tokens=3000
)

avatar_prompt_content = response_2['choices'][0]['message']['content']

with open(AVATAR_PROMPT_PATH, 'w', encoding='utf-8') as avatar_file:
    avatar_file.write(avatar_prompt_content)

print("avatar_prompt saved to data/tmp/avatar_prompt.json")

########################################
# 3. Third GPT: Generate image_prompt_pairs.json using avatar_prompt.json ip_image_paths
########################################

# Read avatar_prompt.json to extract ip_image_paths
with open(AVATAR_PROMPT_PATH, 'r', encoding='utf-8') as af:
    avatar_data = json.load(af)
avatar_paths = [item['ip_image_path'] for item in avatar_data]

system_prompt_3 = f"""You are now a cinematic director and image curator. You have 30 short shot descriptions depicting {story_name}'s life from birth to death, and a JSON array of 5 avatar image objects (from avatar_prompt.json) representing {story_name} at five distinct life stages: Child, Teen, Mid, Mid-Elder, and Old.

Your tasks:
1. Read the 30 short shot descriptions.
2. From avatar_prompt.json, you have these 5 avatar image paths:
{avatar_paths}

3. Assign these 5 avatar image paths to the 30 shots so that the distribution matches {story_name}'s aging process:
   - Earliest shots use the Child stage image.
   - As the narrative progresses into adolescence, use the Teen stage image.
   - Then use Mid, followed by Mid-Elder, and finally Old for the last portion of her life.
   
   Distribute these 5 avatar paths logically and evenly across the 30 shots, ensuring a chronological progression.

4. For each shot, create a JSON object:
   {{
     "ip_img_path": "<assigned_avatar_path_for_this_stage>",
     "prompt": "character:..., background:..., relation:..., camera pose:..., HDR description:..."
   }}

   The "prompt" should expand the original one-sentence shot description into a detailed, cinematic scene description.

[!!!Be clear, be detailed, faithful to the short shot descriptions, and ensure that the life stage distribution makes sense chronologically.]
"""

with open(SHORT_DESC_PATH, 'r', encoding='utf-8') as f:
    short_descriptions_for_image_pairs = f.read().strip()

IMAGE_PROMPT_PATH = os.path.join(RESULT_DIR, 'image_prompt_pairs.json')

response_3 = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt_3},
        {"role": "user", "content": short_descriptions_for_image_pairs}
    ],
    temperature=0.7,
    max_tokens=4000
)

image_prompt_pairs_content = response_3['choices'][0]['message']['content']

with open(IMAGE_PROMPT_PATH, 'w', encoding='utf-8') as json_file:
    json_file.write(image_prompt_pairs_content)

print("image_prompt_pairs saved to results/tmp/image_prompt_pairs.json")
