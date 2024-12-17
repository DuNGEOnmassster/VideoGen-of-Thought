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
    You need to create avatar images for this protagonist at exactly six distinct life stages: Child, Teen, Early-30s, Late-40s, Mid-Elder (Where Mid-Elder means Early-50s), and Old.

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
        "ip_image_path": "<assigned_avatar_path_for_this_stage>",
        "prompt": "Character:..., Background:..., Relation:..., Camera Pose:..., HDR Description:..."
    }}
    ]

    The "prompt" should expand the original one-sentence shot description into a detailed, cinematic scene description.

    [!!!Be clear, be detailed, faithful to the short shot descriptions, and ensure that the life stage distribution makes sense chronologically.]
    [!!!Only Reply the json contents]
    """
    return system_prompt_1, system_prompt_2, system_prompt_3
    
def generate_short_shot_descriptions(api_key, system_prompt_1, story_user_prompt, short_desc_path):
    """
    Generate 30 short shot descriptions and save them to short_shot_description.txt.
    """
    openai.api_key = api_key
    response_1 = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt_1},
            {"role": "user", "content": story_user_prompt}
        ],
        temperature=0.2,  # Lowered temperature for consistency
        top_p=0.9,
        max_tokens=3000
    )

    result_content_1 = response_1['choices'][0]['message']['content']
    with open(short_desc_path, 'w', encoding='utf-8') as result_file:
        result_file.write(result_content_1)
    print("short_shot_description saved to", short_desc_path)

    return result_content_1


def generate_avatar_prompt(api_key, system_prompt_2, short_desc_path, avatar_prompt_path):
    """
    Generate avatar prompts for specified life stages and save them to avatar_prompt.json.
    """
    openai.api_key = api_key
    with open(short_desc_path, 'r', encoding='utf-8') as f:
        short_descriptions_for_avatar = f.read().strip()

    response_2 = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt_2},
            {"role": "user", "content": short_descriptions_for_avatar}
        ],
        temperature=0.2,  # Lowered temperature for consistency
        top_p=0.9,
        max_tokens=3000
    )

    avatar_prompt_content = response_2['choices'][0]['message']['content']
    with open(avatar_prompt_path, 'w', encoding='utf-8') as avatar_file:
        avatar_file.write(avatar_prompt_content)
    print("avatar_prompt saved to", avatar_prompt_path)

    return avatar_prompt_content


def generate_image_prompt_pairs(api_key, system_prompt_3, short_desc_path, image_prompt_pairs_path):
    """
    Generate image prompt pairs for each shot and save them to image_prompt_pairs.json.
    """
    openai.api_key = api_key
    with open(short_desc_path, 'r', encoding='utf-8') as f:
        short_descriptions_for_image_pairs = f.read().strip()

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

    with open(image_prompt_pairs_path, 'w', encoding='utf-8') as json_file:
        json_file.write(image_prompt_pairs_content)

    print(f"image_prompt_pairs saved to {image_prompt_pairs_path}")

    return image_prompt_pairs_content


def depart_script_generation(args):
    # Read API Key
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
        raise ValueError("User prompt not provided. Please provide user prompt using --user_input or save it in user_input.txt.")

    openai.api_key = api_key
    story_name = args.story_name

    # Setting output_dir for saving generated files.
    RESULT_DIR = os.path.join('data', story_name)

    system_prompt_1, system_prompt_2, _ = get_system_prompts(story_name, None)
    # Generate short_shot_description.txt
    result_content_1 = generate_short_shot_descriptions(api_key, system_prompt_1, story_user_prompt, os.path.join(RESULT_DIR, 'short_shot_description.txt'))

    # Generate avatar_prompt.json for the specified life stages
    avatar_prompt_content = generate_avatar_prompt(api_key, system_prompt_2, os.path.join(RESULT_DIR,'short_shot_description.txt'), os.path.join(RESULT_DIR, 'avatar_prompt.json'))

    with open(args.avatar_json_path, 'r', encoding='utf-8') as af:
        avatar_data = json.load(af)
    avatar_paths = [item['ip_image_path'] for item in avatar_data]

    _, _, system_prompt_3 = get_system_prompts(story_name, avatar_paths)

    # Generate image_prompt_pairs.json using avatar_prompt.json ip_image_paths
    image_prompt_pairs_content = generate_image_prompt_pairs(api_key, system_prompt_3, os.path.join(RESULT_DIR,'short_shot_description.txt'), os.path.join(RESULT_DIR, 'image_prompt_pairs.json'))

    return result_content_1, avatar_prompt_content, image_prompt_pairs_content, os.path.join(RESULT_DIR, 'short_shot_description.txt')


# Prepare model pipeline
def prepare_avatar_model():
    ckpt_dir = 'weights/Kolors'
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
    ckpt_dir = 'weights/Kolors'

    text_encoder = ChatGLMModel.from_pretrained(f'{ckpt_dir}/text_encoder', torch_dtype=torch.float16).half()
    tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
    vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half()
    scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
    unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half()

    image_encoder = CLIPVisionModelWithProjection.from_pretrained('weights/Kolors-IP-Adapter-Plus/image_encoder', ignore_mismatched_sizes=True).to(torch.float16)
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

    pipe.load_ip_adapter('weights/Kolors-IP-Adapter-Plus', subfolder="", weight_name=["ip_adapter_plus_general.bin"])

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


# Referenced from DynamiCrafter, https://github.com/Doubiiu/DynamiCrafter/tree/main/scripts/evaluation/inference.py
def prepare_embeddings(model, prompts, videos, noise_shape, text_input=False, unconditional_guidance_scale=1.0, multiple_cond_cfg=False, cfg_img=None, loop=False, interp=False, **kwargs):
    batch_size = noise_shape[0]

    if not text_input:
        prompts = [""]*batch_size

    img = videos[:,:,0] #bchw
    img_emb = model.embedder(img) ## blc
    img_emb = model.image_proj_model(img_emb)

    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [torch.cat([cond_emb,img_emb], dim=1)]}
    if model.model.conditioning_key == 'hybrid':
        z = get_latent_z(model, videos) # b c t h w
        if loop or interp:
            img_cat_cond = torch.zeros_like(z)
            img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
            img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
        else:
            img_cat_cond = z[:,:,:1,:,:]
            img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
        cond["c_concat"] = [img_cat_cond] # b c 1 h w
    
    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            prompts = batch_size * [""]
            uc_emb = model.get_learned_conditioning(prompts)
        elif model.uncond_type == "zero_embed":
            uc_emb = torch.zeros_like(cond_emb)
        uc_img_emb = model.embedder(torch.zeros_like(img)) ## b l c
        uc_img_emb = model.image_proj_model(uc_img_emb)
        uc = {"c_crossattn": [torch.cat([uc_emb,uc_img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc["c_concat"] = [img_cat_cond]
    else:
        uc = None

    ## we need one more unconditioning image=yes, text=""
    if multiple_cond_cfg and cfg_img != 1.0:
        uc_2 = {"c_crossattn": [torch.cat([uc_emb,img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc_2["c_concat"] = [img_cat_cond]
        kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})
    
    return cond, uc


def image_guided_synthesis(model, prompts, videos, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., cond_z0=None, \
                        unconditional_guidance_scale=1.0, cfg_img=None, fs=None, text_input=False, multiple_cond_cfg=False, loop=False, interp=False, timestep_spacing='uniform', guidance_rescale=0.0, **kwargs):
    ddim_sampler = DDIMSampler(model) if not multiple_cond_cfg else DDIMSampler_multicond(model)
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    cond, uc = prepare_embeddings(model, prompts, videos, noise_shape, \
                                  text_input=text_input, unconditional_guidance_scale=unconditional_guidance_scale, \
                                    multiple_cond_cfg=multiple_cond_cfg, cfg_img=cfg_img, loop=loop, interp=interp, **kwargs)

    batch_variants = []
    for _ in range(n_samples):
        if ddim_sampler is not None:

            variants, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=batch_size,
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            cfg_img=cfg_img, 
                                            mask=None,
                                            x0=cond_z0,
                                            fs=fs,
                                            timestep_spacing=timestep_spacing,
                                            guidance_rescale=guidance_rescale,
                                            **kwargs
                                            )
            
        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(variants)
        batch_variants.append(batch_images)
    ## variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5), variants

def reconstruct_from_latent(model, latent_codes):
    # reconstruct from latent to pixel space
    batch_images = model.decode_first_stage(latent_codes)
    # ## batch, c, t, h, w
    return batch_images.permute(1, 0, 2, 3, 4)


def generate_shots(args, gpu_num, gpu_no, short_desc_path):
    ## step 1: model config
    ## -----------------------------------------------------------------
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model_config['params']['unet_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    model.perframe_ae = args.perframe_ae
    assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    ## sample shape
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.out_channels
    n_frames = args.video_length
    batch_size = args.bs
    print(f'Inference with {n_frames} frames')
    noise_shape = [args.bs, channels, n_frames, h, w]

    ## step 2: load data
    ## -----------------------------------------------------------------
    fakedir = os.path.join(args.shot_save_path, "samples")
    fakedir_separate = os.path.join(args.shot_save_path, "samples_separate")

    # os.makedirs(fakedir, exist_ok=True)
    os.makedirs(fakedir_separate, exist_ok=True)

    ## prompt file setting
    assert os.path.exists(short_desc_path), "Error: prompt file Not Found!"
    assert os.path.exists(args.keyframe_path), "Error: keyframe file Not Found!"
    filename_list, data_list, prompt_list = load_data_prompts(args.keyframe_path, short_desc_path, video_size=(args.height, args.width), video_frames=n_frames, interp=args.interp)
    num_samples = len(prompt_list)
    samples_split = num_samples // gpu_num
    # print('Prompts testing [rank:%d] %d/%d samples loaded.'%(gpu_no, samples_split, num_samples))
    #indices = random.choices(list(range(0, num_samples)), k=samples_per_device)
    indices = list(range(samples_split*gpu_no, samples_split*(gpu_no+1)))
    prompt_list_rank = [prompt_list[i] for i in indices]
    data_list_rank = [data_list[i] for i in indices]
    filename_list_rank = [filename_list[i] for i in indices]

    # fpss = torch.tensor([args.fps] * args.bs).to(model.device).long()

    # multiprompt_list = prompt_list  # Use all prompts for inference
    # text_embs_list = [model.get_learned_conditioning([mp]) for mp in multiprompt_list]
    # conds_list = [[{"c_crossattn": [text_emb], "fps": fps} for text_emb in text_embs] for text_embs, fps in zip(text_embs_list, fpss)]

    # import pdb; pdb.set_trace()

    start = time.time()
    batch_latents_list = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for idx, indice in tqdm(enumerate(range(0, len(prompt_list_rank), args.bs)), desc='Sample Batch'):
            cond_z0 = torch.randn(noise_shape)  # reset for each round of inference
            prompts = prompt_list_rank[indice:indice+args.bs]
            videos = data_list_rank[indice:indice+args.bs]
            filenames = filename_list_rank[indice:indice+args.bs]
            if isinstance(videos, list):
                videos = torch.stack(videos, dim=0).to("cuda")
            else:
                videos = videos.unsqueeze(0).to("cuda")

            batch_images, batch_latents = image_guided_synthesis(model, prompts, videos, noise_shape, args.n_samples, args.ddim_steps, args.ddim_eta, cond_z0, \
                                args.unconditional_guidance_scale, args.cfg_img, args.frame_stride, args.text_input, args.multiple_cond_cfg, args.loop, args.interp, args.timestep_spacing, args.guidance_rescale)

            ## save each example individually
            if args.save_individual:
                for ind_idx, samples in enumerate(batch_images):
                    ## samples : [n_samples,c,t,h,w]
                    prompt = prompts[ind_idx]
                    filename = filenames[ind_idx]
                    # save_results(prompt, samples, filename, fakedir, fps=8, loop=args.loop)
                    save_results_seperate(prompt, samples, filename, fakedir, fps=8, loop=args.loop)
        
            # import pdb; pdb.set_trace()
            batch_latents_list.append(batch_latents)

    shot_video_list = [model.decode_first_stage(shot) for shot in batch_latents_list]

    final_output = torch.stack(shot_video_list).flatten(0, 1)
    final_save_path = os.path.join(args.shot_save_path, "final_output.mp4")
    save_results_full(prompt, final_output, final_save_path, fps=8, loop=args.loop)
    print(f"Saved in {args.shot_save_path}. Time used: {(time.time() - start):.2f} seconds")


def parse_args():
    parser = argparse.ArgumentParser(description="VGoT Full Methods")
    parser.add_argument('--user_input', type=str, default="A set of one-sentence prompts, 30 shots, describe a story of a classic American woman Mary's life, from birth to death.", help='User input prompt.')
    parser.add_argument('--story_name', type=str, default="Mary", help='Name of the story to generate.')
    parser.add_argument('--avatar_json_path', type=str, default="data/Mary/avatar_prompt.json", help='Path to the JSON file with image-path and prompt pairs.')
    parser.add_argument('--keyframe_json_path', type=str, default="data/Mary/image_prompt_pairs.json", help='Path to the JSON file with image-path and prompt pairs.')
    parser.add_argument('--keyframe_path', type=str, default="KeyFrames/Mary", help='Directory to save the generated images.')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed')

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
    parser.add_argument("--timestep_spacing", type=str, default="uniform", help="The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.")
    parser.add_argument("--guidance_rescale", type=float, default=0.0, help="guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)")
    parser.add_argument("--perframe_ae", action='store_true', default=False, help="if we use per-frame AE decoding, set it to True to save GPU memory, especially for the model of 576x1024")

    ## currently not support looping video and generative frame interpolation
    parser.add_argument("--loop", action='store_true', default=False, help="generate looping videos or not")
    parser.add_argument("--interp", action='store_true', default=False, help="generate generative frame interpolation or not")
    parser.add_argument("--save_individual", action='store_true', default=False, help="save each example individually or not")
    return parser.parse_args()

def main():
    args = parse_args()
    args.avatar_json_path = f"data/{args.story_name}/avatar_prompt.json"
    args.keyframe_json_path = f"data/{args.story_name}/image_prompt_pairs.json"
    args.keyframe_path = f"KeyFrames/{args.story_name}"
    args.shot_save_path = f"Shot_Videos/{args.story_name}"
    os.makedirs(args.shot_save_path, exist_ok=True)
    
    set_seed(args.seed)
    # script1, avatar_prompt, image_prompt_pairs = script_generation(args)
    script1, avatar_prompt, image_prompt_pairs, short_desc_path = depart_script_generation(args)
    avatar_generation(args)
    keyframe_generation(args)

    rank, gpu_num = 0, 1
    generate_shots(args, gpu_num, rank, short_desc_path)

if __name__ == '__main__':
    main()