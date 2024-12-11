import torch
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers.utils import load_image
import os
import json
from PIL import Image
from diffusers import AutoencoderKL, EulerDiscreteScheduler
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256_ipadapter import StableDiffusionXLPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from kolors.models.unet_2d_condition import UNet2DConditionModel
import argparse

# Argument parsing function
def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using paired image paths and prompts from JSON.")
    parser.add_argument('--json_path', type=str, default="data/data_ip/image_prompt_pairs.json", help='Path to the JSON file with image-path and prompt pairs.')
    parser.add_argument('--output_path', type=str, default="results/a1", help='Directory to save the generated images.')
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
    
    return {idx: {'image_path': item['ip_img_path'], 'prompt': item['prompt']} 
            for idx, item in enumerate(data)}

# Prepare model pipeline
def prepare_model():
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

    pipe = StableDiffusionXLPipeline(
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
def run(args):
    set_seed(args.seed)
    pipe = prepare_model()
    image_prompt_dict = load_image_prompt_pairs(args.json_path)
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

if __name__ == '__main__':
    args = parse_args()
    run(args)
