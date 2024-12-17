import os
import sys
sys.path.append("..")
sys.path.append(".")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai
import json
import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers.utils import load_image
from diffusers import AutoencoderKL, EulerDiscreteScheduler

from PIL import Image
from models.kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from models.kolors.models.modeling_chatglm import ChatGLMModel
from models.kolors.models.tokenization_chatglm import ChatGLMTokenizer
from models.kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256_ipadapter import StableDiffusionXLPipeline as StableDiffusionXLPipeline_ipadapter
from models.kolors.models.unet_2d_condition import UNet2DConditionModel
from models.lvdm.models.samplers.ddim import DDIMSampler
from models.lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond

import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange, repeat
from collections import OrderedDict
import datetime
import time
import random
from utils.utils import *

def get_system_prompts(story_name, avatar_paths):
    system_prompt_1 = """You are a helpful assistant that will transform the user's single-sentence request into a set of 30 one-sentence "short shot descriptions." 
    Each description should depict a moment in the life of a classic American woman named Mary, starting from her birth and following her through 
    significant life moments until her death. Each shot should be one sentence only, and should vividly describe a scene, including details of setting, 
    objects, and emotional tone.

    These 30 shots should form a coherent narrative, show Mary's progression through life, maintain her as a stable protagonist, 
    and display scenes that change over time in a natural, life-spanning progression. Include descriptive elements of lighting, emotion, 
    and environment, making each scene feel cinematic, with subtle but meaningful changes as Mary ages and her life evolves.

    [!!!Try to have a coherent main plot, a stable protagonist, and similar but changing scenes.]"""

    system_prompt_2 = f"""You are a master character concept artist and fashion designer. You have been given a narrative (30 short shot descriptions) about a protagonist's life. 
    From the user prompt, the protagonist's name is {story_name} (story_name = "{story_name}"). 
    You need to create avatar images for this protagonist at exactly six distinct life stages: Child, Teen, Early-30s, Late-40s, Mid-Elder(Late-50s), and Old.

    Produce exactly 6 JSON objects, each representing {story_name} at one of these five life stages. For each object:
    - "ip_image_path": use the format "data/{story_name}/avatar_{story_name}_<stage>.jpg" replacing <stage> with one of the specified stages.
    - "prompt": A multi-line string describing the scene from five angles:
    - Character: Appearance, age, clothing, facial expression, etc.
    - Background: The setting and environment behind {story_name}.
    - Relation: How {story_name} relates to her environment, emotions, or others at that stage.
    - Camera Pose: Cinematic angle and framing.
    - HDR Description: Lighting, atmosphere, and visual qualities, ideally in high detail (e.g., 8K HDR).

    [!!!Be consistent with the narrative, choose details that reflect each stage of {story_name}'s life, and ensure each prompt is detailed and visually evocative.]
    [!!!Be Creative, but also ensure the portrait to keep consistent despite the age changes.]

    Output a JSON array of these 6 objects.
    """

    system_prompt_3 = f"""You are now a cinematic director and image curator. You have 30 short shot descriptions depicting {story_name}'s life from birth to death, and a JSON array of 5 avatar image objects (from avatar_prompt.json) representing {story_name} at Six distinct life stages: Child, Teen, Early-30s, Late-40s, Mid-Elder, and Old.

    Your tasks:
    1. Read the 30 short shot descriptions.
    2. From avatar_prompt.json, you have these 6 avatar image paths:
    {avatar_paths}

    3. Assign these 6 avatar image paths to the 30 shots so that the distribution matches {story_name}'s aging process:
    - Earliest shots use the Child stage image.
    - As the narrative progresses into adolescence, use the Teen and Early-30s, Late-40s, and Mid-Elder stage image, as the majority of the story.
    - Then use Old for the last portion in final shots.
    
    Distribute these 6 avatar paths logically and evenly across the 30 shots, ensuring a chronological progression (each 5 shots represent a different stage).
    [!!! You can use any method to distribute the images, but it should be consistent and logical.]
    [!!! Child and Old only requires three shots, Teen and Late-40s requires seven shots, others require 5 shots each.]

    4. For each shot, create a JSON object:
    [
    {{
        "ip_img_path": "<assigned_avatar_path_for_this_stage>",
        "prompt": "Character:..., Background:..., Relation:..., Camera Pose:..., HDR Description:..."
    }}
    ]

    The "prompt" should expand the original one-sentence shot description into a detailed, cinematic scene description.

    [!!!Be clear, be detailed, faithful to the short shot descriptions, and ensure that the life stage distribution makes sense chronologically.]
    [!!!Only Reply the json contents]
    """
    return system_prompt_1, system_prompt_2, system_prompt_3
    

def script_generation(args):
    # Read API Key
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.txt')
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        api_key = f.read().strip()

    openai.api_key = api_key

    # The original user prompt
    # story_user_prompt = "Describe a set of one-sentence prompts, 30 shots, describe a story of a classic American woman Mary's life, from birth to death."

    # User prompt load from file
    with open('user_input.txt', 'r', encoding='utf-8') as f:
        story_user_prompt = f.read().strip()

    # Placing story_name for story_user_prompt, set by user in args
    story_name = args.story_name

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

    # RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', story_name)
    RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', story_name)
    os.makedirs(RESULT_DIR, exist_ok=True)
    SHORT_DESC_PATH = os.path.join(RESULT_DIR, 'short_shot_description.txt')

    with open(SHORT_DESC_PATH, 'w', encoding='utf-8') as result_file:
        result_file.write(result_content_1)

    print(f"short_shot_description saved to {SHORT_DESC_PATH}")

    ########################################
    # 2. Second GPT: Generate avatar_prompt.json for the specified life stages
    ########################################

    system_prompt_2 = f"""You are a master character concept artist and fashion designer. You have been given a narrative (30 short shot descriptions) about a protagonist's life. 
    From the user prompt, the protagonist's name is {story_name} (story_name = "{story_name}"). 
    You need to create avatar images for this protagonist at exactly six distinct life stages: Child, Teen, Early-30s, Late-40s, Mid-Elder(Late-50s), and Old.

    Produce exactly 6 JSON objects, each representing {story_name} at one of these five life stages. For each object:
    - "ip_image_path": use the format "data/{story_name}/avatar_{story_name}_<stage>.jpg" replacing <stage> with one of the specified stages.
    - "prompt": A multi-line string describing the scene from five angles:
    - Character: Appearance, age, clothing, facial expression, etc.
    - Background: The setting and environment behind {story_name}.
    - Relation: How {story_name} relates to her environment, emotions, or others at that stage.
    - Camera Pose: Cinematic angle and framing.
    - HDR Description: Lighting, atmosphere, and visual qualities, ideally in high detail (e.g., 8K HDR).

    [!!!Be consistent with the narrative, choose details that reflect each stage of {story_name}'s life, and ensure each prompt is detailed and visually evocative.]
    [!!!Be Creative, but also ensure the portrait to keep consistent despite the age changes.]

    Output a JSON array of these 6 objects.
    """

    with open(SHORT_DESC_PATH, 'r', encoding='utf-8') as f:
        short_descriptions_for_avatar = f.read().strip()

    AVATAR_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', story_name)
    os.makedirs(AVATAR_DIR, exist_ok=True)
    # We'll store avatar_prompt.json not inside avatar DIR (as per original instructions),
    # but inside data/tmp to keep consistency. If needed, we can change location.
    # DATA_TMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'tmp')
    DATA_TMP_DIR = AVATAR_DIR
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

    print(f"avatar_prompt saved to {AVATAR_PROMPT_PATH}")

    ########################################
    # 3. Third GPT: Generate image_prompt_pairs.json using avatar_prompt.json ip_image_paths
    ########################################

    # Read avatar_prompt.json to extract ip_image_paths
    with open(AVATAR_PROMPT_PATH, 'r', encoding='utf-8') as af:
        avatar_data = json.load(af)
    avatar_paths = [item['ip_image_path'] for item in avatar_data]

    system_prompt_3 = f"""You are now a cinematic director and image curator. You have 30 short shot descriptions depicting {story_name}'s life from birth to death, and a JSON array of 5 avatar image objects (from avatar_prompt.json) representing {story_name} at Six distinct life stages: Child, Teen, Early-30s, Late-40s, Mid-Elder, and Old.

    Your tasks:
    1. Read the 30 short shot descriptions.
    2. From avatar_prompt.json, you have these 6 avatar image paths:
    {avatar_paths}

    3. Assign these 6 avatar image paths to the 30 shots so that the distribution matches {story_name}'s aging process:
    - Earliest shots use the Child stage image.
    - As the narrative progresses into adolescence, use the Teen and Early-30s, Late-40s, and Mid-Elder stage image, as the majority of the story.
    - Then use Old for the last portion in final shots.
    
    Distribute these 6 avatar paths logically and evenly across the 30 shots, ensuring a chronological progression (each 5 shots represent a different stage).
    [!!! You can use any method to distribute the images, but it should be consistent and logical.]
    [!!! Child and Old only requires three shots, Teen and Late-40s requires seven shots, others require 5 shots each.]

    4. For each shot, create a JSON object:
    [
    {{
        "ip_img_path": "<assigned_avatar_path_for_this_stage>",
        "prompt": "Character:..., Background:..., Relation:..., Camera Pose:..., HDR Description:..."
    }}
    ]

    The "prompt" should expand the original one-sentence shot description into a detailed, cinematic scene description.

    [!!!Be clear, be detailed, faithful to the short shot descriptions, and ensure that the life stage distribution makes sense chronologically.]
    [!!!Only Reply the json contents]
    """

    with open(SHORT_DESC_PATH, 'r', encoding='utf-8') as f:
        short_descriptions_for_image_pairs = f.read().strip()

    IMAGE_PROMPT_PATH = os.path.join(DATA_TMP_DIR, 'image_prompt_pairs.json')

    response_3 = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt_3},
            {"role": "user", "content": short_descriptions_for_image_pairs}
        ],
        temperature=0.7,
        max_tokens=6000
    )

    image_prompt_pairs_content = response_3['choices'][0]['message']['content']

    with open(IMAGE_PROMPT_PATH, 'w', encoding='utf-8') as json_file:
        json_file.write(image_prompt_pairs_content)

    print(f"image_prompt_pairs saved to {IMAGE_PROMPT_PATH}")

    return result_content_1, avatar_prompt_content, image_prompt_pairs_content


# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Load image and prompt pairs from JSON file
def load_image_prompt_pairs(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return {idx: {'ip_image_path': item['ip_image_path'], 'prompt': item['prompt']} 
            for idx, item in enumerate(data)}

# Prepare model pipeline
def prepare_avatar_model():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = f'{root_dir}/weights/Kolors'
    text_encoder = ChatGLMModel.from_pretrained(
        f'{ckpt_dir}/text_encoder',
        torch_dtype=torch.float16).half()
    tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
    vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half()
    scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
    unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half()
    pipe = StableDiffusionXLPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            force_zeros_for_empty_prompt=False)
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    return pipe

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Load image and prompt pairs from JSON file
def load_avatar_image_prompt_pairs(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return {idx: {'image_path': item['ip_img_path'], 'prompt': item['prompt']} 
            for idx, item in enumerate(data)}

def avatar_generation(args):
    set_seed(args.seed)
    pipe = prepare_avatar_model()
    image_prompt_dict = load_avatar_image_prompt_pairs(args.avatar_json_path)
    # get output path from ip_image_path's root directory
    output_path = os.path.dirname(os.path.dirname(os.path.abspath(image_prompt_dict[0]['ip_image_path'])))
    os.makedirs(output_path, exist_ok=True)

    for idx, item in image_prompt_dict.items():
        ip_image_path = item['ip_image_path']
        prompt = item['prompt']
        
        image = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=50,
            guidance_scale=5.0,
            num_images_per_prompt=1,
            generator= torch.Generator(pipe.device).manual_seed(66)).images[0]
        image.save(ip_image_path)
        print(f"idx {idx}: Image saved to {ip_image_path}")

# Prepare model pipeline
def prepare_keyframe_model():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # root_dir = "."
    ckpt_dir = f'{root_dir}/weights/Kolors'

    text_encoder = ChatGLMModel.from_pretrained(f'{ckpt_dir}/text_encoder', torch_dtype=torch.float16).half()
    tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
    vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half()
    scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
    unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half()

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(f'{root_dir}/weights/Kolors-IP-Adapter-Plus/image_encoder', ignore_mismatched_sizes=True).to(torch.float16)
    clip_image_processor = CLIPImageProcessor(size=336, crop_size=336)

    pipe = StableDiffusionXLPipeline_ipadapter(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        image_encoder=image_encoder,
        feature_extractor=clip_image_processor,
        force_zeros_for_empty_prompt=False
    ).to("cuda")
    pipe.enable_model_cpu_offload()

    if hasattr(pipe.unet, 'encoder_hid_proj'):
        pipe.unet.text_encoder_hid_proj = pipe.unet.encoder_hid_proj

    pipe.load_ip_adapter(f'{root_dir}/weights/Kolors-IP-Adapter-Plus', subfolder="", weight_name=["ip_adapter_plus_general.bin"])

    # import pdb; pdb.set_trace()
    
    return pipe

# Main function to run the pipeline
def keyframe_generation(args):
    set_seed(args.seed)
    pipe = prepare_keyframe_model()
    image_prompt_dict = load_image_prompt_pairs(args.keyframe_json_path)
    os.makedirs(args.output_path, exist_ok=True)

    for idx, item in image_prompt_dict.items():
        ip_img_path = item['image_path']
        prompt = item['prompt']
        
        ip_img = Image.open(ip_img_path)
        pipe.set_ip_adapter_scale([0.5])  # Example scale, adjust as needed
        image = pipe(
            prompt=prompt,
            ip_adapter_image=[ip_img],
            negative_prompt="",
            height=1024,
            width=1024,
            num_inference_steps=50,
            guidance_scale=5.0,
            num_images_per_prompt=1,
            generator=torch.manual_seed(args.seed),
        ).images[0]
        image.save(f'{args.output_path}/sample_ip_{idx}.jpg')
        print(f"Image saved: {args.output_path}/sample_ip_{idx}.jpg")


def parse_args():
    parser = argparse.ArgumentParser(description="VGoT Full Methods")
    parser.add_argument('--story_name', type=str, default="Mary", help='Name of the story to generate.')
    parser.add_argument('--avatar_json_path', type=str, default="data/Mary/avatar_prompt.json", help='Path to the JSON file with image-path and prompt pairs.')
    parser.add_argument('--keyframe_json_path', type=str, default="data/Mary/image_prompt_pairs.json", help='Path to the JSON file with image-path and prompt pairs.')
    parser.add_argument('--output_path', type=str, default="KeyFrames/Mary", help='Directory to save the generated images.')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    script1, avatar_prompt, image_prompt_pairs = script_generation(args)
    avatar_generation(args)
    keyframe_generation(args)
