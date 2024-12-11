import torch
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers.utils import load_image
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from PIL import Image
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers import EulerDiscreteScheduler
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Argument parsing function
def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using paired image paths and prompts from JSON.")
    parser.add_argument('--json_path', type=str, default="olivia_avatar.json", help='Path to the JSON file with image-path and prompt pairs.')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed for reproducibility.')
    return parser.parse_args()

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
def prepare_model():
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

# Main function to run the pipeline
def run(args):
    set_seed(args.seed)
    pipe = prepare_model()
    image_prompt_dict = load_image_prompt_pairs(args.json_path)
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

if __name__ == '__main__':
    args = parse_args()
    run(args)
