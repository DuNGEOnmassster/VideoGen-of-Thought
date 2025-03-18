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

from PIL import Image
# from models.kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from pipelines.pipeline_stable_diffusion_xl_chatglm_256_ipadapter import StableDiffusionXLPipeline as StableDiffusionXLPipeline_ipadapter
from models.kolors.models.modeling_chatglm import ChatGLMModel
from models.kolors.models.tokenization_chatglm import ChatGLMTokenizer
# from models.kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256_ipadapter import StableDiffusionXLPipeline as StableDiffusionXLPipeline_ipadapter
from models.kolors.models.unet_2d_condition import UNet2DConditionModel
from models.lvdm.models.samplers.ddim import DDIMSampler
from models.lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

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
    RESULT_DIR = os.path.join('asset', f'story_type{story_type}', story_name)
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

def parse_args():
    parser = argparse.ArgumentParser(description="VGoT Full Methods")
    parser.add_argument('--user_input', type=str, default=None, help='User input prompt.')
    parser.add_argument('--story_name', type=str, default="", help='Name of the story to generate. If empty, will be extracted from user input.')
    parser.add_argument('--num_shot', type=int, default=30, help='Number of shots to generate.')
    parser.add_argument('--story_type', type=str, default=None, help='Type of story: 1 (character life), 2 (multi-character), 3 (flexible)')
    parser.add_argument('--avatar_json_path', type=str, default=None, help='Path to the JSON file with image-path and prompt pairs.')
    parser.add_argument('--keyframe_json_path', type=str, default=None, help='Path to the JSON file with image-path and prompt pairs.')
    parser.add_argument('--keyframe_path', type=str, default=None, help='Directory to save the generated images.')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed')
    parser.add_argument('--use_exist_prompt', type=str, default=None, help='Path to existing prompts. If provided, script generation will be skipped.')

    parser.add_argument("--shot_save_path", type=str, default=None, help="results saving path")
    parser.add_argument("--ckpt_path", type=str, default="weights/DynamiCrafter/model.ckpt", help="checkpoint path")
    parser.add_argument("--config", type=str, default="configs/inference_module3.yaml", help="config (yaml) path")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference, should be one")
    parser.add_argument("--height", type=int, default=576, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=1024, help="image width, in pixel space")
    parser.add_argument("--frame_stride", type=int, default=10, help="frame stride control for 256 model (larger->larger motion), FPS control for 512 or 1024 model (smaller->larger motion)")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=7.5, help="prompt classifier-free guidance")
    parser.add_argument("--video_length", type=int, default=16, help="inference video length")
    parser.add_argument("--negative_prompt", action='store_true', default=False, help="negative prompt")
    parser.add_argument("--text_input", action='store_true', default=False, help="input text to I2V model or not")
    parser.add_argument("--multiple_cond_cfg", action='store_true', default=False, help="use multi-condition cfg or not")
    parser.add_argument("--cfg_img", type=float, default=None, help="guidance scale for image conditioning")
    parser.add_argument("--timestep_spacing", type=str, default="uniform", help="The way the timesteps should be scaled.")
    parser.add_argument("--guidance_rescale", type=float, default=0.0, help="guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed]")
    parser.add_argument("--perframe_ae", action='store_true', default=False, help="if we use per-frame AE decoding, set it to True to save GPU memory")

    ## currently not support looping video and generative frame interpolation
    parser.add_argument("--loop", action='store_true', default=False, help="generate looping videos or not")
    parser.add_argument("--interp", action='store_true', default=False, help="generate generative frame interpolation or not")
    parser.add_argument("--save_individual", action='store_true', default=False, help="save each example individually or not")
    return parser.parse_args()


# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Global model initialization
def initialize_models(args):
    """Initialize all models and components once"""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = f'{root_dir}/weights/Kolors'
    
    # Common components initialization
    text_encoder = ChatGLMModel.from_pretrained(
        f'{ckpt_dir}/text_encoder',
        torch_dtype=torch.float16).half()
    tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
    vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half()
    scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
    unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half()
    
    # Avatar generation model
    avatar_pipe = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        force_zeros_for_empty_prompt=False
    ).to("cuda")
    
    # Keyframe generation model
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        f'{root_dir}/weights/Kolors-IP-Adapter-Plus/image_encoder',
        ignore_mismatched_sizes=True).to(torch.float16)
    clip_image_processor = CLIPImageProcessor(size=336, crop_size=336)
    
    keyframe_pipe = StableDiffusionXLPipeline_ipadapter(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        image_encoder=image_encoder,
        feature_extractor=clip_image_processor,
        force_zeros_for_empty_prompt=False
    ).to("cuda")

    if hasattr(keyframe_pipe.unet, 'encoder_hid_proj'):
        keyframe_pipe.unet.text_encoder_hid_proj = keyframe_pipe.unet.encoder_hid_proj

    keyframe_pipe.load_ip_adapter(f'{root_dir}/weights/Kolors-IP-Adapter-Plus', subfolder="", weight_name=["ip_adapter_plus_general.bin"])
    
    return {
        'avatar_pipe': avatar_pipe,
        'keyframe_pipe': keyframe_pipe,
        'text_encoder': text_encoder,
        'tokenizer': tokenizer
    }

def check_existing_prompts(path):
    """Check if all required prompt files exist"""
    required_files = [
        'avatar_prompt.json',
        'image_prompt_pairs.json',
        'short_shot_description.txt'
    ]
    return all(os.path.exists(os.path.join(path, f)) for f in required_files)

def main():
    args = parse_args()
    
    # If keyframe_path and shot_save_path are not specified, create based on story_name
    if args.story_name and not args.keyframe_path:
        args.keyframe_path = f"KeyFrames/{args.story_name}"
    if args.story_name and not args.shot_save_path:
        args.shot_save_path = f"Shot_Videos/{args.story_name}"
    
    # Create necessary directories
    if args.keyframe_path:
        os.makedirs(args.keyframe_path, exist_ok=True)
    if args.shot_save_path:
        os.makedirs(args.shot_save_path, exist_ok=True)
    
    set_seed(args.seed)
    
    # Process existing prompt path
    if args.use_exist_prompt:
        if not check_existing_prompts(args.use_exist_prompt):
            raise FileNotFoundError(f"Missing required files in {args.use_exist_prompt}")
        
        # Set path parameters
        args.avatar_json_path = os.path.join(args.use_exist_prompt, 'avatar_prompt.json')
        args.keyframe_json_path = os.path.join(args.use_exist_prompt, 'image_prompt_pairs.json')
        args.keyframe_path = os.path.join(args.use_exist_prompt, 'KeyFrames')
        args.shot_save_path = os.path.join(args.use_exist_prompt, 'Shot_Videos')
        short_desc_path = os.path.join(args.use_exist_prompt, 'short_shot_description.txt')
    else:
        # Process regular path
        if args.story_name:
            base_path = f"asset/story_type{args.story_type}/{args.story_name}"
            args.avatar_json_path = os.path.join(base_path, 'avatar_prompt.json')
            args.keyframe_json_path = os.path.join(base_path, 'image_prompt_pairs.json')
            args.keyframe_path = os.path.join(base_path, 'KeyFrames')
            args.shot_save_path = os.path.join(base_path, 'Shot_Videos')
            short_desc_path = os.path.join(base_path, 'short_shot_description.txt')
            os.makedirs(base_path, exist_ok=True)
    
    # Initialize all models
    # models = initialize_models(args)
    
    if not args.use_exist_prompt:
        # Generate script content
        script_content = depart_script_generation(args)
        print("Script generation completed successfully!")
        print(f"Generated {args.num_shot} shots for story: {args.story_name}")
    
    # Generate avatar and keyframe
    avatar_generation(args)
    keyframe_generation(args)
    print(f"Keyframe generation completed successfully!, saved to {args.keyframe_json_path}")
    print("Cross shot propagation completed successfully!")


if __name__ == '__main__':
    main()
