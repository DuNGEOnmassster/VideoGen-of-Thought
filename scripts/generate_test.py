import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange, repeat
from collections import OrderedDict
import datetime
import time
import random

import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from PIL import Image

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
sys.path.append("..")
sys.path.append(".")
from models.lvdm.models.samplers.ddim import DDIMSampler
from models.lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from utils.utils import *

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

def prepare_all_embeddings(model, prompts, videos, noise_shape, text_input=False, unconditional_guidance_scale=1.0, multiple_cond_cfg=False, cfg_img=None, loop=False, interp=False, **kwargs):
    """
    Prepare all embeddings at once and return lists of cond and uc for each prompt.
    """
    cond, uc = prepare_embeddings(model, prompts, videos, noise_shape, 
                                  text_input=text_input, 
                                  unconditional_guidance_scale=unconditional_guidance_scale,
                                  multiple_cond_cfg=multiple_cond_cfg, 
                                  cfg_img=cfg_img, 
                                  loop=loop, 
                                  interp=interp, 
                                  **kwargs)
    
    # Extract cond_list
    cond_list = []
    for c in cond["c_crossattn"]:
        cond_list.append(c.clone())
    
    # Extract uc_list
    if uc is not None and "c_crossattn" in uc:
        uc_list = []
        for u in uc["c_crossattn"]:
            uc_list.append(u.clone())
    else:
        uc_list = [None] * len(prompts)
    
    return cond_list, uc_list

def image_guided_synthesis(model, cond, uc, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., cond_z0=None, \
                           unconditional_guidance_scale=1.0, cfg_img=None, fs=None, **kwargs):
    """
    Generate images based on precomputed cond and uc.
    """
    # Initialize sampler
    if kwargs.get('multiple_cond_cfg', False):
        ddim_sampler = DDIMSampler_multicond(model)
    else:
        ddim_sampler = DDIMSampler(model)
    
    # Sample using DDIM
    variants, _ = ddim_sampler.sample(S=ddim_steps,
                                      conditioning=cond,
                                      batch_size=cond["c_crossattn"][0].shape[0],
                                      shape=noise_shape[1:],
                                      verbose=False,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=uc,
                                      eta=ddim_eta,
                                      cfg_img=cfg_img, 
                                      mask=None,
                                      x0=cond_z0,
                                      fs=fs,
                                      **kwargs
                                      )
    
    # Reconstruct from latent to pixel space
    batch_images = model.decode_first_stage(variants)
    batch_images = batch_images.permute(0, 1, 2, 3, 4, 5)  # Adjust dimensions if necessary
    
    return batch_images, variants

def reconstruct_from_latent(model, latent_codes):
    """
    Reconstruct images from latent codes.
    """
    batch_images = model.decode_first_stage(latent_codes)
    return batch_images.permute(1, 0, 2, 3, 4)

def run_inference(args, gpu_num, gpu_no):
    """
    Main inference function.
    """
    ## Step 1: Model Configuration
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model_config['params']['unet_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    model.perframe_ae = args.perframe_ae
    assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    ## Sample shape
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.out_channels
    n_frames = args.video_length
    batch_size = args.bs
    print(f'Inference with {n_frames} frames')
    noise_shape = [args.bs, channels, n_frames, h, w]

    ## Step 2: Load Data
    fakedir = os.path.join(args.savedir, "samples")
    fakedir_separate = os.path.join(args.savedir, "samples_separate")
    os.makedirs(fakedir_separate, exist_ok=True)

    ## Prompt file setting
    assert os.path.exists(args.prompt_dir), "Error: prompt file Not Found!"
    assert os.path.exists(args.keyframe_dir), "Error: keyframe file Not Found!"
    filename_list, data_list, prompt_list = load_data_prompts(args.keyframe_dir, args.prompt_dir, video_size=(args.height, args.width), video_frames=n_frames, interp=args.interp)
    num_samples = len(prompt_list)
    samples_split = num_samples // gpu_num
    print('Prompts testing [rank:%d] %d/%d samples loaded.'%(gpu_no, samples_split, num_samples))
    indices = list(range(samples_split*gpu_no, samples_split*(gpu_no+1)))
    prompt_list_rank = [prompt_list[i] for i in indices]
    data_list_rank = [data_list[i] for i in indices]
    filename_list_rank = [filename_list[i] for i in indices]

    start = time.time()
    batch_latents_list = []
    
    # Prepare all embeddings at once
    cond_list, uc_list = prepare_all_embeddings(model, prompt_list_rank, data_list_rank, noise_shape, 
                                                text_input=args.text_input, 
                                                unconditional_guidance_scale=args.unconditional_guidance_scale, 
                                                multiple_cond_cfg=args.multiple_cond_cfg, 
                                                cfg_img=args.cfg_img, 
                                                loop=args.loop, 
                                                interp=args.interp)

    with torch.no_grad(), torch.cuda.amp.autocast():
        for idx, indice in tqdm(enumerate(range(0, len(prompt_list_rank), args.bs)), desc='Sample Batch'):
            # Determine the current batch
            current_prompts = prompt_list_rank[indice:indice+args.bs]
            current_videos = data_list_rank[indice:indice+args.bs]
            current_filenames = filename_list_rank[indice:indice+args.bs]
            
            if isinstance(current_videos, list):
                current_videos = torch.stack(current_videos, dim=0).to("cuda")
            else:
                current_videos = current_videos.unsqueeze(0).to("cuda")
            
            # Get corresponding cond and uc for the current batch
            current_cond = cond_list[indice:indice+args.bs]
            current_uc = uc_list[indice:indice+args.bs]

            # Stack cond and uc for the batch
            cond = {"c_crossattn": [torch.cat(current_cond, dim=0)]}
            if args.unconditional_guidance_scale != 1.0:
                if current_uc[0] is not None:
                    uc = {"c_crossattn": [torch.cat(current_uc, dim=0)]}
                else:
                    uc = None
            else:
                uc = None

            # Prepare any additional kwargs if necessary
            additional_kwargs = {}
            if args.multiple_cond_cfg:
                # Handle multiple condition configurations if required
                additional_kwargs['multiple_cond_cfg'] = True

            # Generate images using precomputed cond and uc
            batch_images, batch_latents = image_guided_synthesis(model, cond, uc, noise_shape, 
                                                                 n_samples=args.n_samples, 
                                                                 ddim_steps=args.ddim_steps, 
                                                                 ddim_eta=args.ddim_eta, 
                                                                 cond_z0=None, 
                                                                 unconditional_guidance_scale=args.unconditional_guidance_scale, 
                                                                 cfg_img=args.cfg_img, 
                                                                 fs=None, 
                                                                 **additional_kwargs)

            ## Save each example individually
            if args.save_individual:
                for ind_idx, samples in enumerate(batch_images):
                    prompt = current_prompts[ind_idx]
                    filename = current_filenames[ind_idx]
                    save_results_seperate(prompt, samples, filename, fakedir, fps=8, loop=args.loop)
        
            batch_latents_list.append(batch_latents)

    # Reconstruct from latent to pixel space
    shot_video_list = [model.decode_first_stage(shot) for shot in batch_latents_list]
    final_output = torch.stack(shot_video_list).permute(1, 0, 2, 3, 4, 5)
    save_results_seperate(prompt, final_output, "final_output", fakedir, fps=8, loop=args.loop)
    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--prompt_dir", type=str, default=None, help="a data dir containing videos and prompts")
    parser.add_argument("--keyframe_dir", type=str, default=None, help="a data dir containing keyframes")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference, should be one")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frame_stride", type=int, default=3, help="frame stride control for 256 model (larger->larger motion), FPS control for 512 or 1024 model (smaller->larger motion)")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--seed", type=int, default=123, help="seed for seed_everything")
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
    return parser


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@DynamiCrafter cond-Inference: %s"%now)
    parser = get_parser()
    args = parser.parse_args()

    seed = args.seed
    if seed < 0:
        seed = random.randint(0, 2 ** 31)
    seed_everything(seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)