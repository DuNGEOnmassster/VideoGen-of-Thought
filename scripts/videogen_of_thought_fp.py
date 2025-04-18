import os
import sys
sys.path.append("..")
sys.path.append(".")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import openai
import json
import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers.utils import load_image
from diffusers import AutoencoderKL, EulerDiscreteScheduler
import traceback

from PIL import Image
# from models.kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from pipelines.pipeline_stable_diffusion_xl_chatglm_256_ipadapter import StableDiffusionXLPipeline as StableDiffusionXLPipeline_ipadapter
from models.kolors.models.modeling_chatglm import ChatGLMModel
from models.kolors.models.tokenization_chatglm import ChatGLMTokenizer
# from models.kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256_ipadapter import StableDiffusionXLPipeline as StableDiffusionXLPipeline_ipadapter
from models.kolors.models.unet_2d_condition import UNet2DConditionModel
# from models.lvdm.models.samplers.ddim import DDIMSampler # Removed DynamiCrafter import
# from models.lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond # Removed DynamiCrafter import
import math

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange, repeat
from collections import OrderedDict
import datetime
import time
import random
from utils.utils import *

# --- Add FramePack specific imports ---
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, resize_and_center_crop, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
from transformers import AutoTokenizer
# Ensure HF_HOME is set if needed for downloads, though paths are specified via args
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), '../hf_download')))
# --- End FramePack specific imports ---

# Global model cache to avoid reloading
MODEL_CACHE = {
    "text_embedding": None,
    "avatar_model": None,
    "keyframe_model": None
}

# Root directory for weight files
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def extract_text_features(text, tokenizer, model, device="cuda"):
    """Extract text features using BERT model"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the average of the last hidden state as features
    features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return features

def check_constraint_completeness(prompt, film_constraints):
    """Check if the prompt completely contains all five dimensions of constraints"""
    dimensions = ["Character:", "Background:", "Relation:", "Camera Pose:", "HDR Description:"]
    for dim in dimensions:
        if dim not in prompt:
            return False
    return True

def calculate_narrative_coherence(current_features, previous_features):
    """Calculate semantic similarity between current shot and previous shot"""
    if previous_features is None:
        return 1.0  # First shot has no previous shot, default to valid
    similarity = cosine_similarity(current_features, previous_features)[0][0]
    return similarity

def validate_prompt(current_prompt, previous_prompt, tokenizer, model, tau_c=0.85, tau_k=1):
    """Validate if generated prompt meets coherence and completeness criteria"""
    # Check constraint completeness
    k_valid = check_constraint_completeness(current_prompt, None)
    
    # If this is the first prompt or completeness check fails, only return completeness result
    if previous_prompt is None or not k_valid:
        return k_valid, 1.0, k_valid
    
    # Calculate semantic similarity
    current_features = extract_text_features(current_prompt, tokenizer, model)
    previous_features = extract_text_features(previous_prompt, tokenizer, model)
    coherence = calculate_narrative_coherence(current_features, previous_features)
    
    # Validation result
    c_valid = coherence > tau_c
    v_valid = c_valid and k_valid
    
    return v_valid, coherence, k_valid

def get_system_prompts(args, story_name, avatar_paths):
    """Get system prompts for different stages of story generation"""
    
    # Dynamic shot count based on args
    num_shot = args.num_shot

    system_prompt_1 = f"""You are a professional screenwriter and narrative expert who needs to transform the user's brief request into {num_shot} short "shot descriptions".
    
    Each description should be just one sentence, but should vividly describe a scene, including details of scene settings, objects, and emotional tone.
    
    These {num_shot} shots should form a coherent narrative, maintaining stable protagonists, and showcasing naturally changing scenes throughout the story.
    Include descriptive elements of lighting, emotion, and environment to make each scene cinematic.
    
    The user has requested a story about: "{args.user_input}"
    
    [!!!Try to build a coherent main plot with stable characters and scenes that evolve logically.]
    [!!!Ensure the narrative is logically sound, showing appropriate character and plot development.]"""

    # Number of life stages to be generated based on shot count
    num_stages = min(6, max(3, num_shot // 5))
    
    system_prompt_2 = f"""You are a professional character concept artist and fashion designer. You have received a narrative with {num_shot} short shot descriptions.
    
    Based on the user prompt: "{args.user_input}"
    
    You need to create avatar images for main characters in {num_stages} different stages or appearances throughout the story.
    
    Please create exactly {num_stages} JSON objects, each representing a different stage/appearance. For each object:
    - "ip_image_path": Use the format "{args.prompt_path}/avatar_{story_name}_<stage>.jpg", replacing <stage> with a descriptive word for the stage/appearance.
    - "prompt": A multi-line string describing the character from five angles:
    - Character: Appearance, age, clothing, facial expression, etc.
    - Background: Scene and environment surrounding the character.
    - Relation: Character's relationship with the environment, emotions, or others at that stage.
    - Camera Pose: Cinematic angle and composition.
    - HDR Description: Lighting, atmosphere, and visual quality, preferably high-detail (e.g., 8K HDR).
    
    [!!!Stay consistent with the narrative, choosing details that reflect character development, and ensure each prompt is detailed and visually compelling.]
    [!!!Be creative but also ensure visual consistency across related images.]
    
    Output a JSON array of these {num_stages} objects.
    """

    system_prompt_3 = f"""You are now a film director and image curator. You have {num_shot} short shot descriptions for a story based on: "{args.user_input}"
    
    You also have a JSON array containing {num_stages} avatar image objects (from avatar_prompt.json) representing different characters or stages, with these paths:
    {avatar_paths}
    
    Your task:
    1. Read the {num_shot} short shot descriptions.
    2. Assign these avatar image paths to the {num_shot} shots, distributing them appropriately based on the narrative.
    
    Distribute these avatar paths across the {num_shot} shots logically, ensuring narrative consistency.
    
    For each shot, create a JSON object:
    [
    {{
        "ip_image_path": "<assigned avatar path for this shot>",
        "prompt": "Character:..., Background:..., Relation:..., Camera Pose:..., HDR Description:..."
    }}
    ]
    
    The "prompt" should expand the original one-sentence shot description into a detailed, cinematic scene description.
    
    [!!!Descriptions should be clear, detailed, faithful to the short shot description, and maintain visual and narrative consistency.]
    [!!!Reply with JSON content only]
    """
    
    system_prompt_4 = """Evaluate the provided scene description. Check if it includes all five key dimensions:
    1. Character: Character's appearance, age, clothing, facial expression, etc.
    2. Background: Scene background and environment.
    3. Relation: Character's relationship with environment, emotions, or others.
    4. Camera Pose: Cinematic angle and composition.
    5. HDR Description: Lighting, atmosphere, and visual quality.
    
    If any dimension is missing, return "incomplete". Otherwise, evaluate narrative coherence, considering logical transitions between scenes and character development.
    
    Output format:
    {
        "completeness": true/false,
        "missing_dimensions": ["dimension1", "dimension2", ...],
        "coherence": float (0-1),
        "improvement_suggestions": "specific suggestions..."
    }
    """
    
    return system_prompt_1, system_prompt_2, system_prompt_3, system_prompt_4

def extract_story_name(user_input):
    """Extract a meaningful story name from user input if none is provided"""
    # Use OpenAI to generate a concise story name
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that creates concise story titles."},
            {"role": "user", "content": f"Create a short, concise title (1-3 words) for a story with this description, using only alphanumeric characters and underscores: '{user_input}'"}
        ],
        temperature=0.7,
        max_tokens=50
    )
    
    story_name = response['choices'][0]['message']['content'].strip()
    # Clean up the story name to ensure it's valid for file paths
    story_name = re.sub(r'[^\w]', '_', story_name)
    return story_name

def generate_short_shot_descriptions(api_key, system_prompt_1, story_user_prompt, short_desc_path, story_name, num_shot):
    """Generate short shot descriptions based on user input and save to file"""
    openai.api_key = api_key
    formatted_prompt = system_prompt_1
    response_1 = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": story_user_prompt}
        ],
        temperature=0.2,  # Lower temperature for consistency
        top_p=0.9,
        max_tokens=3000
    )

    result_content_1 = response_1['choices'][0]['message']['content']
    with open(short_desc_path, 'w', encoding='utf-8') as result_file:
        result_file.write(result_content_1)
    print(f"Generated {num_shot} short shot descriptions saved to {short_desc_path}")

    return result_content_1

def generate_avatar_prompt(api_key, system_prompt_2, short_desc_path, avatar_prompt_path, num_stages):
    """Generate avatar prompts for specific stages and save to JSON file"""
    openai.api_key = api_key
    with open(short_desc_path, 'r', encoding='utf-8') as f:
        short_descriptions_for_avatar = f.read().strip()

    response_2 = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt_2},
            {"role": "user", "content": short_descriptions_for_avatar}
        ],
        temperature=0.2,  # Lower temperature for consistency
        top_p=0.9,
        max_tokens=3000
    )

    avatar_prompt_content = response_2['choices'][0]['message']['content']
    with open(avatar_prompt_path, 'w', encoding='utf-8') as avatar_file:
        avatar_file.write(avatar_prompt_content)
    print(f"Generated {num_stages} avatar prompts saved to {avatar_prompt_path}")

    return avatar_prompt_content

def initialize_text_embedding_model(device="cuda"):
    """Initialize ChatGLM model for text feature extraction"""
    print("Loading text embedding model...")
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = f'{root_dir}/weights/Kolors'
    text_encoder = ChatGLMModel.from_pretrained(
        f'{ckpt_dir}/text_encoder',
        torch_dtype=torch.float16).half()
    tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    print("Text embedding model loaded successfully")
    return tokenizer, text_encoder

def generate_image_prompt_pairs_with_validation(api_key, system_prompts, short_desc_path, image_prompt_pairs_path, tokenizer, model, device="cuda"):
    """Generate image-prompt pairs with self-validation mechanism to ensure narrative coherence and constraint completeness"""
    openai.api_key = api_key
    system_prompt_3, system_prompt_4 = system_prompts[2], system_prompts[3]
    
    with open(short_desc_path, 'r', encoding='utf-8') as f:
        short_descriptions = f.read().strip()
    
    # Step 1: Generate initial image-prompt pairs
    response_3 = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt_3},
            {"role": "user", "content": short_descriptions}
        ],
        temperature=0.7,
        max_tokens=6000
    )
    
    image_prompt_pairs_content = response_3['choices'][0]['message']['content']
    try:
        # Try to parse JSON, if format is incorrect, do some cleaning
        if not image_prompt_pairs_content.strip().startswith('['):
            # Try to find the beginning of the JSON array
            start_idx = image_prompt_pairs_content.find('[')
            if start_idx != -1:
                image_prompt_pairs_content = image_prompt_pairs_content[start_idx:]
        
        image_prompt_pairs = json.loads(image_prompt_pairs_content)
    except json.JSONDecodeError:
        print("JSON parsing failed, attempting to clean content...")
        # Try to clean JSON string
        cleaned_content = re.sub(r'```json\s*|\s*```', '', image_prompt_pairs_content)
        start_idx = cleaned_content.find('[')
        end_idx = cleaned_content.rfind(']') + 1
        if start_idx != -1 and end_idx > start_idx:
            cleaned_content = cleaned_content[start_idx:end_idx]
            try:
                image_prompt_pairs = json.loads(cleaned_content)
            except:
                print("Still unable to parse JSON after cleaning, using original content...")
                with open(image_prompt_pairs_path, 'w', encoding='utf-8') as json_file:
                    json_file.write(image_prompt_pairs_content)
                return image_prompt_pairs_content
        else:
            print("Could not find valid JSON array, using original content...")
            with open(image_prompt_pairs_path, 'w', encoding='utf-8') as json_file:
                json_file.write(image_prompt_pairs_content)
            return image_prompt_pairs_content
    
    # Step 2: Self-validation and correction
    validated_pairs = []
    previous_prompt = None
    tau_c = 0.85  # Coherence threshold
    
    for i, pair in enumerate(image_prompt_pairs):
        prompt = pair["prompt"]
        valid, coherence, k_valid = validate_prompt(prompt, previous_prompt, tokenizer, model, tau_c)
        
        # If validation fails, attempt correction
        retry_count = 0
        while (not valid or not k_valid) and retry_count < 3:
            print(f"Shot {i+1} validation failed: Completeness={k_valid}, Coherence={coherence:.2f}")
            
            # Construct correction prompt
            correction_prompt = f"""Please modify the following scene description to ensure it includes all five key dimensions (Character, Background, Relation, Camera Pose, HDR Description) and maintains narrative coherence with the previous scene:

Original description:
{prompt}

Previous scene description:
{previous_prompt if previous_prompt else '(This is the first scene)'}

The modified description should be more detailed, maintain narrative coherence, and include all five dimensions."""

            response_correction = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a professional film screenwriter and director skilled at creating coherent and detailed scene descriptions."},
                    {"role": "user", "content": correction_prompt}
                ],
                temperature=0.5,
                max_tokens=1000
            )
            
            corrected_prompt = response_correction['choices'][0]['message']['content']
            prompt = corrected_prompt
            valid, coherence, k_valid = validate_prompt(prompt, previous_prompt, tokenizer, model, tau_c)
            retry_count += 1
        
        # Update validated pair
        pair["prompt"] = prompt
        validated_pairs.append(pair)
        previous_prompt = prompt
    
    # Save validated objects
    validated_content = json.dumps(validated_pairs, indent=2, ensure_ascii=False)
    with open(image_prompt_pairs_path, 'w', encoding='utf-8') as json_file:
        json_file.write(validated_content)
    
    print(f"Validated image-prompt pairs saved to {image_prompt_pairs_path}")
    return validated_content

def depart_script_generation(args):
    """Generate script with shots, avatars, and image-prompt pairs"""
    # Read API key
    CONFIG_PATH = os.path.join('configs', 'config.txt')
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        api_key = f.read().strip()

    # User input prompt
    if args.user_input is not None:
        story_user_prompt = args.user_input
    elif os.path.exists('user_input.txt'):
        with open('user_input.txt', 'r', encoding='utf-8') as f:
            story_user_prompt = f.read().strip()
    else:
        raise ValueError("No user prompt provided. Please provide a user prompt with --user_input or save it in user_input.txt.")

    openai.api_key = api_key
    
    # If story_name is not provided, extract one from the user input
    if not args.story_name or args.story_name == "":
        args.story_name = extract_story_name(story_user_prompt)
        print(f"Generated story name: {args.story_name}")

    story_name = args.story_name
    
    # Determine story type from arguments or try to infer
    if args.story_type:
        story_type = args.story_type
    else:
        # Try to infer story type
        story_type_prompt = f"Based on this story description: '{story_user_prompt}', classify it as one of: '1' (character life story), '2' (multi-character adventure), or '3' (flexible narrative like a short event). Just respond with the number 1, 2, or 3."
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful classification assistant."},
                {"role": "user", "content": story_type_prompt}
            ],
            temperature=0.3,
            max_tokens=10
        )
        story_type = response['choices'][0]['message']['content'].strip()
        # Default to type 3 if couldn't determine
        if not story_type.isdigit() or int(story_type) not in [1, 2, 3]:
            story_type = "3"
        print(f"Inferred story type: {story_type}")
    
    # Set up output directory for generated files based on story type
    # RESULT_DIR = os.path.join('asset', f'story_type{story_type}', story_name)
    RESULT_DIR = args.prompt_path
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # Update args with the result directory
    args.prompt_path = RESULT_DIR

    # Number of stages based on number of shots
    num_stages = min(6, max(3, args.num_shot // 5))

    # Get system prompts
    system_prompts = get_system_prompts(args, story_name, None)
    system_prompt_1, system_prompt_2 = system_prompts[0], system_prompts[1]
    
    # Generate short shot descriptions
    result_content_1 = generate_short_shot_descriptions(
        api_key, system_prompt_1, story_user_prompt, 
        os.path.join(RESULT_DIR, 'short_shot_description.txt'),
        story_name, args.num_shot
    )

    # Generate avatar prompts
    avatar_prompt_content = generate_avatar_prompt(
        api_key, system_prompt_2, 
        os.path.join(RESULT_DIR, 'short_shot_description.txt'), 
        os.path.join(RESULT_DIR, 'avatar_prompt.json'),
        num_stages
    )

    # Set avatar_json_path for further processing
    args.avatar_json_path = os.path.join(RESULT_DIR, 'avatar_prompt.json')

    # Load avatar paths
    with open(args.avatar_json_path, 'r', encoding='utf-8') as af:
        avatar_data = json.load(af)
    avatar_paths = [item['ip_image_path'] for item in avatar_data]
    
    # Update system prompts to include avatar paths
    system_prompts = get_system_prompts(args, story_name, avatar_paths)

    # Get tokenizer and text_encoder
    tokenizer, text_encoder = initialize_text_embedding_model()
    
    # Generate image-prompt pairs with validation
    args.keyframe_json_path = os.path.join(RESULT_DIR, 'image_prompt_pairs.json')
    image_prompt_pairs_content = generate_image_prompt_pairs_with_validation(
        api_key, system_prompts,
        os.path.join(RESULT_DIR, 'short_shot_description.txt'), 
        args.keyframe_json_path,
        tokenizer, text_encoder
    )

    return result_content_1, avatar_prompt_content, image_prompt_pairs_content, os.path.join(RESULT_DIR, 'short_shot_description.txt')

##################  Cross Shot Propagation  ##################
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
    
    return {idx: {'ip_image_path': item['ip_image_path'], 'prompt': item['prompt']} 
            for idx, item in enumerate(data)}


# Load image and prompt pairs from JSON file
def load_keyframe_image_prompt_pairs(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return {idx: {'ip_image_path': item['ip_image_path'], 'prompt': item['prompt']} 
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

        if not args.overwrite_visuals and os.path.exists(ip_image_path):
            print(f"Avatar {idx} exists: {ip_image_path}. Skipping generation (overwrite_visuals=False).")
            continue

        print(f"Generating Avatar {idx}: {ip_image_path}")
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
    image_prompt_dict = load_keyframe_image_prompt_pairs(args.keyframe_json_path)
    os.makedirs(args.keyframe_path, exist_ok=True)

    for idx, item in image_prompt_dict.items():
        ip_img_path = item['ip_image_path']
        prompt = item['prompt']
        output_keyframe_path = f'{args.keyframe_path}/sample_ip_{idx}.jpg'

        if not args.overwrite_visuals and os.path.exists(output_keyframe_path):
            print(f"Keyframe {idx} exists: {output_keyframe_path}. Skipping generation (overwrite_visuals=False).")
            continue
        
        # Ensure input avatar image exists
        if not os.path.exists(ip_img_path):
             print(f"Error: Input avatar image not found for keyframe {idx}: {ip_img_path}. Skipping.")
             continue

        print(f"Generating Keyframe {idx}: {output_keyframe_path}")
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
        image.save(f'{args.keyframe_path}/sample_ip_{idx}.jpg')
        print(f"Image saved: {args.keyframe_path}/sample_ip_{idx}.jpg")

# +++ Add FramePack Shot Generation Function +++
@torch.no_grad()
def generate_shots_framepack(args, keyframe_paths, prompts, output_dir):
    """
    Generates video shots using FramePack I2V for each keyframe and prompt.
    Loads models ONCE at the beginning.
    Optionally saves individual shots or a combined final video.
    """
    print("\n--- Initializing FramePack Models for Shot Generation (Loading ONCE) ---")

    # Check CUDA
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting shot generation.")
        return

    # --- Load FramePack Models (Load ONCE to GPU) ---
    text_encoder = None
    text_encoder_2 = None
    tokenizer = None
    tokenizer_2 = None
    vae = None
    feature_extractor = None
    image_encoder = None
    transformer = None

    try:
        # Clear cache before loading potentially large models
        print("Clearing CUDA cache before loading models...")
        torch.cuda.empty_cache()

        print("Loading FramePack models to CPU first...")
        # Load to CPU first to check availability before large GPU allocation
        text_encoder = LlamaModel.from_pretrained(args.fp_hf_model_dir, subfolder='text_encoder', torch_dtype=torch.float16).cpu()
        text_encoder_2 = CLIPTextModel.from_pretrained(args.fp_hf_model_dir, subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()

        print(f"Loading tokenizer from: {os.path.join(args.fp_hf_model_dir, 'tokenizer')}")
        tokenizer = AutoTokenizer.from_pretrained(args.fp_hf_model_dir, subfolder='tokenizer', use_fast=True)
        print(f"Loading tokenizer_2 from: {os.path.join(args.fp_hf_model_dir, 'tokenizer_2')}")
        tokenizer_2 = AutoTokenizer.from_pretrained(args.fp_hf_model_dir, subfolder='tokenizer_2', use_fast=True)

        vae = AutoencoderKLHunyuanVideo.from_pretrained(args.fp_hf_model_dir, subfolder='vae', torch_dtype=torch.float16).cpu()
        feature_extractor = SiglipImageProcessor.from_pretrained(args.fp_siglip_model_dir, subfolder='feature_extractor')
        image_encoder = SiglipVisionModel.from_pretrained(args.fp_siglip_model_dir, subfolder='image_encoder', torch_dtype=torch.float16).cpu()
        transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(args.fp_framepack_model_dir, torch_dtype=torch.bfloat16).cpu()

        print("Moving FramePack models to GPU...")
        # Now move all to GPU
        text_encoder.to(gpu)
        text_encoder_2.to(gpu)
        image_encoder.to(gpu)
        vae.to(gpu)
        transformer.to(gpu)

        print("Setting eval mode and configurations...")
        vae.eval()
        text_encoder.eval()
        text_encoder_2.eval()
        image_encoder.eval()
        transformer.eval()

        # Enable VAE slicing/tiling regardless of detected VRAM, as it's generally safer for stability
        print("Enabling VAE slicing and tiling.")
        vae.enable_slicing()
        vae.enable_tiling()

        transformer.high_quality_fp32_output_for_inference = True
        print('transformer.high_quality_fp32_output_for_inference = True')

        # Disable gradients (already done on CPU, but good practice)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        image_encoder.requires_grad_(False)
        transformer.requires_grad_(False)

        print("FramePack models loaded and moved to GPU successfully.")

    except Exception as e:
        print(f"Error loading or moving FramePack models: {e}")
        traceback.print_exc()
        # Attempt cleanup even if loading failed
        del text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, feature_extractor, image_encoder, transformer
        torch.cuda.empty_cache()
        return # Cannot proceed without models

    # --- Loop through keyframes and prompts ---
    num_shots = len(keyframe_paths)
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    all_shot_latents_list = [] # List to store latents of each shot (on CPU)

    for idx, (keyframe_path, prompt) in enumerate(tqdm(zip(keyframe_paths, prompts), total=num_shots, desc="Generating Shots (FramePack)")):
        shot_start_time = time.time()
        base_filename = os.path.splitext(os.path.basename(keyframe_path))[0]
        individual_output_filename = os.path.join(output_dir, f"{base_filename}.mp4") # Path for individual shot

        # --- Skip logic based on overwrite_shots ---
        should_skip = False
        if not args.overwrite_shots and os.path.exists(individual_output_filename):
            should_skip = True
            print(f"Shot {idx+1}/{num_shots} exists: {individual_output_filename}. Skipping generation (overwrite_shots=False).")

        if should_skip:
             if not args.save_individual:
                 # If only final video is needed, add placeholder even if skipping generation
                 all_shot_latents_list.append(None)
             continue # Skip the rest of the loop for this shot



        # If we reach here, we need to generate the shot (either overwrite=True or file doesn't exist)
        try:
            # --- Per-Shot Processing ---
            torch.cuda.empty_cache()

            # 1. Load and preprocess the keyframe image
            input_image = Image.open(keyframe_path).convert('RGB')
            input_image_np = np.array(input_image)
            H, W, C = input_image_np.shape
            height, width = find_nearest_bucket(H, W, resolution=args.fp_resolution)
            input_image_np_resized = resize_and_center_crop(input_image_np, target_width=width, target_height=height)
            input_image_pt = torch.from_numpy(input_image_np_resized).float() / 127.5 - 1.0
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None] # B C F H W

            # 2. Encode prompt (and negative prompt)
            # Models are on GPU, encode_prompt_conds should work
            llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
            if args.fp_cfg == 1.0:
                llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
            else:
                llama_vec_n, clip_l_pooler_n = encode_prompt_conds(args.fp_n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

            # 3. VAE encode image
            start_latent = vae_encode(input_image_pt.to(vae.device, dtype=vae.dtype), vae) # B C F(1) H W

            # 4. CLIP Vision encode image
            image_encoder_output = hf_clip_vision_encode(input_image_np_resized, feature_extractor, image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

            # 5. Dtype conversion (Inputs to transformer)
            # Make sure they are on the correct device (GPU) before casting dtype
            llama_vec = llama_vec.to(gpu, dtype=transformer.dtype)
            llama_vec_n = llama_vec_n.to(gpu, dtype=transformer.dtype)
            clip_l_pooler = clip_l_pooler.to(gpu, dtype=transformer.dtype)
            clip_l_pooler_n = clip_l_pooler_n.to(gpu, dtype=transformer.dtype)
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(gpu, dtype=transformer.dtype)
            # start_latent needs to be float32 for history buffer on CPU later
            start_latent = start_latent.to(cpu, dtype=torch.float32)

            # 6. Prepare for FramePack sampling loop
            current_seed = args.seed + idx
            rnd = torch.Generator(gpu).manual_seed(current_seed)
            total_latent_sections = int(max(round((args.fp_shot_duration_seconds * 30) / (args.fp_latent_window_size * 4)), 1))
            num_frames_per_step = args.fp_latent_window_size * 4 - 3

            history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
            history_pixels = None
            total_generated_latent_frames = 0

            latent_paddings = list(reversed(range(total_latent_sections)))
            if total_latent_sections > 4 and not args.fp_disable_padding_trick:
                latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

            # --- FramePack Inner Sampling Loop ---
            for section_idx, latent_padding in enumerate(latent_paddings):
                is_last_section = latent_padding == 0
                latent_padding_size = latent_padding * args.fp_latent_window_size

                # Prepare indices and conditioning latents (ensure devices are correct)
                indices = torch.arange(0, sum([1, latent_padding_size, args.fp_latent_window_size, 1, 2, 16])).unsqueeze(0).to(gpu) # Indices on GPU
                clean_latent_indices_pre, _, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = \
                    indices.split([1, latent_padding_size, args.fp_latent_window_size, 1, 2, 16], dim=1)
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

                # Prepare clean latents - move to GPU and cast dtype just before use
                clean_latents_pre = start_latent.to(gpu, dtype=torch.bfloat16) # Use start_latent (from CPU)
                hist_access_idx = min(history_latents.shape[2], 1 + 2 + 16)
                # History latents are on CPU, move parts to GPU for conditioning
                clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :hist_access_idx, :, :].split([1, 2, 16], dim=2)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post.to(gpu, dtype=torch.bfloat16)], dim=2)
                clean_latents_2x_gpu = clean_latents_2x.to(gpu, dtype=torch.bfloat16)
                clean_latents_4x_gpu = clean_latents_4x.to(gpu, dtype=torch.bfloat16)

                # Configure TeaCache
                if args.fp_use_teacache:
                    transformer.initialize_teacache(enable_teacache=True, num_steps=args.fp_steps)
                else:
                    transformer.initialize_teacache(enable_teacache=False)

                def callback(d): pass # Keep callback simple

                # Run Sampling
                generated_latents = sample_hunyuan(
                    transformer=transformer, # Already on GPU
                    sampler='unipc',
                    width=width,
                    height=height,
                    frames=num_frames_per_step,
                    real_guidance_scale=args.fp_cfg,
                    distilled_guidance_scale=args.fp_gs,
                    guidance_rescale=args.fp_rs,
                    num_inference_steps=args.fp_steps,
                    generator=rnd, # GPU generator
                    prompt_embeds=llama_vec, # Already on GPU/bfloat16
                    prompt_embeds_mask=llama_attention_mask.to(gpu), # Ensure mask is on GPU
                    prompt_poolers=clip_l_pooler, # Already on GPU/bfloat16
                    negative_prompt_embeds=llama_vec_n, # Already on GPU/bfloat16
                    negative_prompt_embeds_mask=llama_attention_mask_n.to(gpu), # Ensure mask is on GPU
                    negative_prompt_poolers=clip_l_pooler_n, # Already on GPU/bfloat16
                    device=gpu,
                    dtype=torch.bfloat16, # Transformer internal dtype
                    image_embeddings=image_encoder_last_hidden_state, # Already on GPU/bfloat16
                    latent_indices=latent_indices, # Already on GPU
                    clean_latents=clean_latents, # Already on GPU/bfloat16
                    clean_latent_indices=clean_latent_indices, # Already on GPU
                    clean_latents_2x=clean_latents_2x_gpu, # Prepared above
                    clean_latent_2x_indices=clean_latent_2x_indices.to(gpu), # Ensure on GPU
                    clean_latents_4x=clean_latents_4x_gpu, # Prepared above
                    clean_latent_4x_indices=clean_latent_4x_indices.to(gpu), # Ensure on GPU
                    callback=callback,
                )
                generated_latents = generated_latents.to(cpu, dtype=torch.float32) # Move results to CPU

                if is_last_section:
                    generated_latents = torch.cat([start_latent, generated_latents], dim=2)

                current_section_latent_frames = generated_latents.shape[2]
                total_generated_latent_frames += current_section_latent_frames
                history_latents = torch.cat([generated_latents, history_latents], dim=2)

                if is_last_section:
                    # Get the complete latents for THIS shot
                    current_shot_latents = history_latents[:, :, :total_generated_latent_frames, :, :].clone() # Clone to avoid issues later
                    all_shot_latents_list.append(current_shot_latents.cpu()) # Store on CPU

                    if args.save_individual:
                        # Save individual shot regardless of overwrite flag (since we decided not to skip)
                        print(f"    Decoding and saving individual shot {idx+1}...")
                        history_pixels = vae_decode(current_shot_latents.to(vae.device, dtype=vae.dtype), vae).cpu()
                        save_bcthw_as_mp4(history_pixels, individual_output_filename, fps=30)
                        print(f"    Saved individual shot: {individual_output_filename}")
                        del history_pixels
                        torch.cuda.empty_cache()
                    break # Exit inner sampling loop

        except Exception as e:
            print(f"\nError generating shot {idx+1} ({keyframe_path}): {e}")
            traceback.print_exc()
            if 'out of memory' in str(e).lower():
                print("Attempting to clear cache after OOM...")
                torch.cuda.empty_cache() # Try to recover for next shot

        # Clear potentially large tensors from this shot iteration
        del llama_vec, clip_l_pooler, llama_vec_n, clip_l_pooler_n, llama_attention_mask, llama_attention_mask_n
        del start_latent, image_encoder_last_hidden_state, history_latents # history_pixels might not exist if not saving individually
        if 'current_shot_latents' in locals(): del current_shot_latents # Delete if it was created
        if 'generated_latents' in locals(): del generated_latents
        if 'clean_latents_pre' in locals(): del clean_latents_pre, clean_latents_post, clean_latents_2x, clean_latents_4x, clean_latents
        if 'clean_latents_2x_gpu' in locals(): del clean_latents_2x_gpu, clean_latents_4x_gpu
        torch.cuda.empty_cache() # Clean cache after each shot

    # --- Final Processing: Combine and Decode if not saving individually ---
    if not args.save_individual:
        print("\n--- Combining and Decoding Final Video ---")
        valid_latents = [lat for lat in all_shot_latents_list if lat is not None]
        if not valid_latents:
            print("No valid latents generated or found. Cannot create final video.")
        else:
            print(f"Concatenating latents from {len(valid_latents)} shots...")
            try:
                # Concatenate along the time dimension (dim=2)
                final_concatenated_latents = torch.cat(valid_latents, dim=2)
                print(f"Final latent shape: {final_concatenated_latents.shape}")

                # Decode the full sequence
                # VAE should still be on GPU from the load-once strategy
                print("Decoding final concatenated latents...")
                final_pixels = vae_decode(final_concatenated_latents.to(vae.device, dtype=vae.dtype), vae).cpu()

                # Save the final video
                final_save_path = os.path.join(output_dir, f"{args.story_name}_final_combined.mp4") # Use story name for combined video
                print(f"Saving final combined video to {final_save_path}...")
                save_bcthw_as_mp4(final_pixels, final_save_path, fps=30)
                print("Final combined video saved.")
                del final_pixels # Clean up

            except Exception as e:
                print(f"Error during final video concatenation or decoding: {e}")
                traceback.print_exc()
            finally:
                 del final_concatenated_latents # Clean up concatenated latents
                 torch.cuda.empty_cache()


    # --- Final Cleanup ---
    print("\nUnloading FramePack models...")
    del text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, feature_extractor, image_encoder, transformer
    del all_shot_latents_list # Clear the list of latents
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"\n--- FramePack Shot Generation Finished ({num_shots} shots attempted) ---")
    print(f"Total time for shot generation: {end_time - start_time:.2f} seconds")
    print(f"Generated videos saved in: {output_dir}")
# +++ End FramePack Shot Generation Function +++


def parse_args():
    parser = argparse.ArgumentParser(description="VGoT Full Methods with FramePack")
    # --- Script Generation Args ---
    parser.add_argument('--user_input', type=str, default=None, help='User input prompt.')
    parser.add_argument('--story_name', type=str, default="", help='Name of the story to generate. If empty, will be extracted from user input.')
    parser.add_argument('--num_shot', type=int, default=30, help='Number of shots to generate for the script.')
    parser.add_argument('--story_type', type=str, default=None, help='Type of story: 1 (character life), 2 (multi-character), 3 (flexible)')
    parser.add_argument('--use_exist_prompt', type=str, default=None, help='Path to existing prompts dir. If provided, script generation will be skipped.')
    parser.add_argument("--base_path", type=str, default="asset", help="Base directory for saving generated assets (scripts, avatars, keyframes, shots)")
    parser.add_argument('--overwrite_script', action='store_true', help="Overwrite existing script files (short description, avatar prompts, keyframe prompts).")
    parser.add_argument('--overwrite_visuals', action='store_true', help="Overwrite existing visual files (avatar images, keyframe images).")
    parser.add_argument('--overwrite_shots', action='store_true', default=True, help="Overwrite existing shot video files (default: True).")

    # --- Avatar/Keyframe Generation Args (Kolors/IP-Adapter based) ---
    parser.add_argument('--avatar_json_path', type=str, default=None, help='Path to the JSON file for avatar prompts/paths (auto-derived if not use_exist_prompt).')
    parser.add_argument('--keyframe_json_path', type=str, default=None, help='Path to the JSON file for keyframe image-prompt pairs (auto-derived if not use_exist_prompt).')
    parser.add_argument('--keyframe_path', type=str, default=None, help='Directory to save the generated keyframe images (auto-derived if not use_exist_prompt).')
    parser.add_argument('--seed', type=int, default=3407, help='Base random seed')

    # --- Shot Generation Args (FramePack based) ---
    parser.add_argument("--shot_save_path", type=str, default=None, help="Directory to save the final generated video shots (auto-derived if not use_exist_prompt)")
    parser.add_argument('--fp_hf_model_dir', type=str, default='"./weights/HunyuanVideo"', help="FramePack: Directory for base HunyuanVideo models (tokenizers, text encoders, VAE).")
    parser.add_argument('--fp_framepack_model_dir', type=str, default='./weights/FramePackI2V_HY', help="FramePack: Directory for FramePack Transformer model.")
    parser.add_argument('--fp_siglip_model_dir', type=str, default='./weights/flux_redux_bfl', help="FramePack: Directory for Siglip models (image encoder).")
    parser.add_argument('--fp_shot_duration_seconds', type=float, default=4.0, help="FramePack: Target duration for each generated video shot in seconds.")
    parser.add_argument('--fp_steps', type=int, default=25, help="FramePack: Number of diffusion steps per shot.")
    parser.add_argument('--fp_gs', type=float, default=10.0, help="FramePack: Distilled CFG Scale.")
    parser.add_argument('--fp_n_prompt', type=str, default="low quality, worst quality, blurry, motion blur, text, watermark, signature, bad anatomy, bad hands", help="FramePack: Negative text prompt (used if fp_cfg != 1.0).")
    parser.add_argument('--fp_use_teacache', action='store_true', default=True, help="FramePack: Enable TeaCache.")
    parser.add_argument('--fp_no_teacache', action='store_false', dest='fp_use_teacache', help="FramePack: Disable TeaCache.")
    parser.add_argument('--fp_resolution', type=int, default=640, help="FramePack: Resolution bucket base size (e.g., 640 finds nearest bucket like 640x368, 512x512, etc.).")
    parser.add_argument('--fp_disable_padding_trick', action='store_true', help="FramePack: Disable the special padding sequence for videos longer than 4 sections.")
    parser.add_argument('--fp_latent_window_size', type=int, default=9, help=argparse.SUPPRESS) # Hide internal param
    parser.add_argument('--fp_cfg', type=float, default=1.0, help=argparse.SUPPRESS) # Hide internal param
    parser.add_argument('--fp_rs', type=float, default=0.0, help=argparse.SUPPRESS) # Hide internal param
    parser.add_argument("--save_individual", action='store_true', help="Save each generated shot video individually.")

    args = parser.parse_args()

    # --- Auto-derive paths (unchanged from previous version) ---
    if not args.use_exist_prompt:
        if not args.story_name:
             if not args.user_input:
                 raise ValueError("Cannot derive paths: Need --story_name or --user_input when not using --use_exist_prompt")
             # Need to generate story_name before path derivation, handled in main()

        # Construct base path dynamically based on other args
        # Ensure story_type is resolved if None initially
        resolved_story_type = args.story_type if args.story_type else "unknown"
        base_path = os.path.join(args.base_path, f"seed{args.seed}", f"story_type{resolved_story_type}", args.story_name)

        # Set derived paths if they weren't explicitly provided
        if args.avatar_json_path is None:
            args.avatar_json_path = os.path.join(base_path, 'avatar_prompt.json')
        if args.keyframe_json_path is None:
            args.keyframe_json_path = os.path.join(base_path, 'image_prompt_pairs.json')
        if args.keyframe_path is None:
            args.keyframe_path = os.path.join(base_path, 'KeyFrames')
        if args.shot_save_path is None:
            args.shot_save_path = os.path.join(base_path, 'Shot_Videos_FP')
        args.prompt_path = base_path
    else:
        base_path = args.use_exist_prompt
        if args.avatar_json_path is None:
            args.avatar_json_path = os.path.join(base_path, 'avatar_prompt.json')
        if args.keyframe_json_path is None:
            args.keyframe_json_path = os.path.join(base_path, 'image_prompt_pairs.json')
        if args.keyframe_path is None:
            args.keyframe_path = os.path.join(base_path, 'KeyFrames')
        if args.shot_save_path is None:
            args.shot_save_path = os.path.join(base_path, 'Shot_Videos_FP')
        args.prompt_path = base_path

    return args

def check_existing_prompts(path):
    """Check if all required prompt files exist"""
    required_files = [
        'avatar_prompt.json',
        'image_prompt_pairs.json',
        'short_shot_description.txt'
    ]
    return all(os.path.exists(os.path.join(path, f)) for f in required_files)


def main():
    args = parse_args() # Parse args first

    # --- Setup & Validation ---
    set_seed(args.seed)

    # Handle story name generation if needed (requires API key)
    if not args.use_exist_prompt and not args.story_name:
        if not args.user_input:
            raise ValueError("Need --user_input to generate story name if --story_name is not provided and not using --use_exist_prompt.")
        try:
            CONFIG_PATH = os.path.join('configs', 'config.txt')
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
            openai.api_key = api_key
            print("Generating story name from user input...")
            args.story_name = extract_story_name(args.user_input)
            print(f"Generated story name: {args.story_name}")
            # Re-run path derivation in parse_args might be cleaner, or update paths here:
            resolved_story_type = args.story_type if args.story_type else "unknown"
            base_path = os.path.join(args.base_path, f"seed{args.seed}", f"story_type{resolved_story_type}", args.story_name)
            args.prompt_path = base_path
            args.avatar_json_path = os.path.join(args.prompt_path, 'avatar_prompt.json')
            args.keyframe_json_path = os.path.join(args.prompt_path, 'image_prompt_pairs.json')
            args.keyframe_path = os.path.join(base_path, 'KeyFrames')
            args.shot_save_path = os.path.join(base_path, 'Shot_Videos_FP')
        except Exception as e:
             raise ValueError(f"Failed to generate story name. Ensure API key in 'configs/config.txt' is valid. Error: {e}")


    # Now paths should be finalized in args object
    print(f"Story Name: {args.story_name}")
    print(f"Prompt Path: {args.prompt_path}")
    print(f"Avatar JSON: {args.avatar_json_path}")
    print(f"Keyframe JSON: {args.keyframe_json_path}")
    print(f"Keyframe Path: {args.keyframe_path}")
    print(f"Shot Save Path: {args.shot_save_path}")

    # Create necessary directories based on final paths
    os.makedirs(args.prompt_path, exist_ok=True)
    os.makedirs(args.keyframe_path, exist_ok=True)
    os.makedirs(args.shot_save_path, exist_ok=True)


    # --- Pipeline Stages ---

    # 1. Script Generation
    script_files_exist = check_existing_prompts(args.prompt_path) # Check if all script JSONs/TXTs exist
    if args.use_exist_prompt:
        print("\n--- Stage 1: Skipping Script Generation (using existing prompts) ---")
        if not script_files_exist:
             print(f"Warning: Missing some expected files in {args.prompt_path}. Subsequent steps might fail.")
    elif script_files_exist and not args.overwrite_script:
        print(f"\n--- Stage 1: Skipping Script Generation (existing script files found in {args.prompt_path} and overwrite_script=False) ---")
    else:
        # Generate script only if not using existing, or if files don't exist, or if overwrite_script is True
        print("\n--- Stage 1: Generating Script ---")
        if script_files_exist and args.overwrite_script:
             print(f"  (overwrite_script=True, regenerating script files in {args.prompt_path})")
        try:
            script_content = depart_script_generation(args)
            print("Script generation completed successfully!")
            print(f"Generated {args.num_shot} shots script for story: {args.story_name}")
        except Exception as e:
             print(f"Error during script generation: {e}")
             traceback.print_exc()
             print("Skipping subsequent stages due to script generation error.")
             return # Exit if script generation fails

    # 2. Avatar Generation (using Kolors)
    print("\n--- Stage 2: Generating Avatars (using Kolors) ---")
    if not os.path.exists(args.avatar_json_path):
        print(f"Error: Avatar prompt file not found at {args.avatar_json_path}. Skipping avatar generation.")
    else:
        # Avatar generation logic will now check overwrite flag internally
        try:
            avatar_generation(args) # Pass args to the function
            print(f"Avatar generation process finished.")
        except Exception as e:
            print(f"Error during avatar generation: {e}")
            traceback.print_exc()

    # 3. Keyframe Generation (using Kolors IP-Adapter)
    print("\n--- Stage 3: Generating Keyframes (using Kolors IP-Adapter) ---")
    if not os.path.exists(args.keyframe_json_path):
         print(f"Error: Keyframe prompt file not found at {args.keyframe_json_path}. Skipping keyframe generation.")
    else:
        # Keyframe generation logic will now check overwrite flag internally
        try:
            keyframe_generation(args) # Pass args to the function
            print(f"Keyframe generation process finished.")
        except Exception as e:
            print(f"Error during keyframe generation: {e}")
            traceback.print_exc()


    # 4. Shot Generation (using FramePack I2V)
    print("\n--- Stage 4: Generating Shots (using FramePack I2V) ---")
    torch.cuda.empty_cache()
    if not os.path.exists(args.keyframe_json_path):
         print(f"Error: Keyframe prompt file not found at {args.keyframe_json_path}. Skipping shot generation.")
    elif not os.path.isdir(args.keyframe_path):
         print(f"Error: Keyframes directory not found at {args.keyframe_path}. Skipping shot generation.")
    else:
        try:
            with open(args.keyframe_json_path, 'r', encoding='utf-8') as f:
                keyframe_data = json.load(f)

            if not keyframe_data:
                 print("Error: Keyframe JSON data is empty. Cannot generate shots.")
            else:
                prompts_for_shots = [item['prompt'] for item in keyframe_data]
                keyframe_paths_for_shots = [os.path.join(args.keyframe_path, f"sample_ip_{idx}.jpg") for idx in range(len(prompts_for_shots))]

                missing_keyframes = [p for p in keyframe_paths_for_shots if not os.path.exists(p)]
                if missing_keyframes:
                    print(f"Error: Missing keyframe files needed for shot generation: {missing_keyframes}. Skipping shot generation.")
                else:
                     # Shot generation logic will now check overwrite flag internally
                     generate_shots_framepack(args, keyframe_paths_for_shots, prompts_for_shots, args.shot_save_path)

        except Exception as e:
            print(f"Error during preparation or execution of FramePack shot generation: {e}")
            traceback.print_exc()


    print("\n--- VGoT Pipeline Finished ---")


if __name__ == '__main__':
    main()
