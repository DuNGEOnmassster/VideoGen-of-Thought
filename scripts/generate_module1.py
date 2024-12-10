import os
import openai

# Read API key
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.txt')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    api_key = f.read().strip()

openai.api_key = api_key

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

user_prompt_1 = "Describe a set of one-sentence prompts, 30 shots, describe a story of a classic American woman Mary's life, from birth to death."

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

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'tmp')
os.makedirs(RESULT_DIR, exist_ok=True)
SHORT_DESC_PATH = os.path.join(RESULT_DIR, 'short_shot_description.txt')

with open(SHORT_DESC_PATH, 'w', encoding='utf-8') as result_file:
    result_file.write(result_content_1)

print("short_shot_description saved to results/tmp/short_shot_description.txt")


########################################
# 2. Second GPT: Generate avatar_prompt.json
########################################

system_prompt_2 = """You are a master character concept artist and fashion designer, tasked with creating detailed character design prompts for a series of avatar images that represent Mary at various stages in her life, based on the previously generated short shot descriptions.

Generate 5 to 6 distinct JSON objects, each representing Mary at a different age or stage. Each object should contain:
- "ip_image_path": a file path (e.g., "/storage/home/mingzhe/code/VideoGen-of-Thought/data/fashion_designer/avatar_Mary_XXXX.jpg")
- "prompt": a multi-line string describing the scene from five angles:
  - character: Appearance, age, clothing, facial expression, etc.
  - background: The setting/environment behind Mary.
  - relation: How Mary relates to her environment, her emotions, or others at this stage.
  - camera pose: Cinematic angle and framing.
  - HDR description: Lighting, atmosphere, and visual qualities in high detail (e.g., 8K HDR).

[!!!Be consistent with the life progression and character style, and ensure each prompt is detailed and visually evocative.]

You should output a JSON array of these 5-6 objects."""

with open(SHORT_DESC_PATH, 'r', encoding='utf-8') as f:
    short_descriptions_for_avatar = f.read().strip()

AVATAR_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'tmp')
os.makedirs(AVATAR_DIR, exist_ok=True)
AVATAR_PROMPT_PATH = os.path.join(AVATAR_DIR, 'avatar_prompt.json')

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
# 3. Third GPT: Generate image_prompt_pairs.json with assigned ip_img_paths
########################################

# Here we define a list of 30 ip_img_paths for the 30 shots.
# For example, they could be something like this:
ip_img_paths = [
    f"/storage/home/mingzhe/code/VideoGen-of-Thought/data/tmp/img_{i+1}.jpg"
    for i in range(30)
]

# We'll provide these paths to the model in the system prompt instruction.
# The model should distribute these 30 paths over the 30 shots in order.

system_prompt_3 = f"""You are now a cinematic director and image curator. You have a list of 30 short shot descriptions detailing moments from Mary's life (from birth to death). 
You also have a predefined list of 30 image paths (one for each shot):

{ip_img_paths}

Your task:
1. Convert each of the 30 short shot descriptions into a JSON object with fields "ip_img_path" and "prompt".
2. The "prompt" should follow the format:
   "character:..., background:..., relation:..., camera pose:..., HDR description:..."
   
   - character: Describe Mary's appearance, age, expression, attire.
   - background: Describe the setting and environment.
   - relation: Emotional or narrative connection between Mary and the scene.
   - camera pose: How the camera frames and angles the shot.
   - HDR description: Lighting and atmospheric details in a cinematic 8K HDR style.

3. Assign each shot exactly one unique image path from the provided list, in order or a roughly even distribution. 
   For example, shot 1 gets the first path, shot 2 gets the second path, and so forth until all 30 shots are assigned.

4. Return a JSON array of 30 objects. Each object:
   {{
     "ip_img_path": "<one_of_the_provided_paths>",
     "prompt": "<the detailed cinematic prompt derived from the short shot description>"
   }}

[!!!Be clear, be detailed, and faithful to the original short shot description. Make sure each prompt is visually rich and consistent with Mary's life progression.]"""

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
