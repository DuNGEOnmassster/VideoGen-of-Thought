import argparse, os, sys, glob
import datetime, time
from omegaconf import OmegaConf
from tqdm import tqdm
from tqdm import trange
from einops import rearrange, repeat
from collections import OrderedDict

import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from PIL import Image
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from lvdm_tvg.models.samplers.ddim import DDIMSampler
from lvdm_tvg.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from utils.utils import instantiate_from_config
import random
import numpy as np
import json
from lpips import LPIPS
def load_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['data']

@torch.no_grad()
def slerp(latents0, latents1, fract_mixing):
    r""" Copied from lunarring/latentblending
    Helper function to correctly mix two random variables using spherical interpolation.
    The function will always cast up to float64 for sake of extra 4.
    Args:
        p0: 
            First tensor for interpolation
        p1: 
            Second tensor for interpolation
        fract_mixing: float 
            Mixing coefficient of interval [0, 1]. 
            0 will return in p0
            1 will return in p1
            0.x will return a mix between both preserving angular velocity.
    """
    if latents0 is None or latents1 is None:
        return latents0 if latents1 is None else latents1
    p0 = latents0
    p1 = latents1
    if p0.dtype == torch.float16:
        recast_to = 'fp16'
    else:
        recast_to = 'fp32'

    p0 = p0.double()
    p1 = p1.double()
        
    norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
    epsilon = 1e-7
    dot = torch.sum(p0 * p1) / norm
    dot = dot.clamp(-1+epsilon, 1-epsilon)

    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * fract_mixing
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    interp = p0*s0 + p1*s1

    if recast_to == 'fp16':
        interp = interp.half()
    elif recast_to == 'fp32':
        interp = interp.float()

    return interp


def get_filelist(data_dir, postfixes):
    if os.path.isfile(data_dir):
        return [data_dir]
    patterns = [os.path.join(data_dir, f"*.{postfix}") for postfix in postfixes]
    file_list = []
    for pattern in patterns:
        file_list.extend(glob.glob(pattern))
    file_list.sort()
    return file_list

def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            ## rename the keys for 256x256 model
            new_pl_sd = OrderedDict()
            for k,v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=True)
    else:
        # deepspeed
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]]=state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    print('>>> model checkpoint loaded.')
    return model



def load_data_prompts(json_list, video_size=(256,256), video_frames=16, interp=False):
    transform = transforms.Compose([
        transforms.Resize(min(video_size)),
        transforms.CenterCrop(video_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    ## load prompts
    # prompt_file = get_filelist(data_dir, ['txt'])
    # assert len(prompt_file) > 0, "Error: found NO prompt file!"
    ###### default prompt
    # default_idx = 0
    # default_idx = min(default_idx, len(prompt_file)-1)
    # if len(prompt_file) > 1:
    #     print(f"Warning: multiple prompt files exist. The one {os.path.split(prompt_file[default_idx])[1]} is used.")
    ## only use the first one (sorted by name) if multiple exist
    
    ## load video
    data_list = []
    prompt_list = []
    save_names = []
    for data in json_list:
        prompt = data.get('prompts')
        prompt_inverse = data.get('prompt_inverse')
        prompt_1 = data.get('prompt_1')
        prompt_2 = data.get('prompt_2')
        start_frame_1 = data.get('start_frame_1') if data.get('start_frame_1') else 0
        start_frame_2 = data.get('start_frame_2') if data.get('start_frame_2') else 0
        video_dir_1 = data['video_dir_1']
        video_dir_2 = data['video_dir_2']
        video_list_1 = get_filelist(video_dir_1, ['jpg', 'png', 'jpeg', 'JPEG', 'PNG'])
        video_list_2 = get_filelist(video_dir_2, ['jpg', 'png', 'jpeg', 'JPEG', 'PNG'])
        # assert len(file_list) == n_samples, "Error: data and prompts are NOT paired!"
        
        # prompt_list = load_prompts(prompt_file[default_idx])
        prompt_list.append([prompt,prompt_inverse, prompt_1,prompt_2])
        if interp:
            frame1_list=[]
            frame2_list=[]
            image1 = Image.open(video_list_1[start_frame_1]).convert('RGB')
            image_tensor1 = transform(image1).unsqueeze(1) # [c,1,h,w]
            frame1_list.append(image_tensor1)
            image2 = Image.open(video_list_2[start_frame_2]).convert('RGB')
            image_tensor2 = transform(image2).unsqueeze(1) # [c,1,h,w]
            frame2_list.append(image_tensor2)
            frame_tensor1 = torch.cat(frame1_list,dim=1)
            frame_tensor2 = torch.cat(frame2_list,dim=1)
            frame_tensor1 = repeat(frame_tensor1, 'c t h w -> c (repeat t) h w', repeat=video_frames//2)
            frame_tensor2 = repeat(frame_tensor2, 'c t h w -> c (repeat t) h w', repeat=video_frames//2)
            frame_tensor = torch.cat([frame_tensor1, frame_tensor2], dim=1)
            save_name = data.get('name')
            if save_name is None:
                save_name = os.path.basename(video_list_1[start_frame_1]).split('.')[0] + '_'+os.path.basename(video_list_2[start_frame_2]).split('.')[0]

            save_names.append(save_name)

        else:
            frame1_list=[]
            image1 = Image.open(video_list_1[start_frame_1]).convert('RGB')
            image_tensor1 = transform(image1).unsqueeze(1) # [c,1,h,w]
            frame1_list.append(image_tensor1)
            frame_tensor1 = torch.cat(frame1_list,dim=1)
            frame_tensor = repeat(frame_tensor1, 'c t h w -> c (repeat t) h w', repeat=video_frames)
            save_name = data.get('name')
            if save_name is None:
                
                save_name = os.path.basename(video_list_1[start_frame_1]).split('.')[0]

            save_names.append(save_name)

        data_list.append(frame_tensor)
        
    return data_list, prompt_list, save_names


def save_results(prompt, samples, filename, fakedir, fps=8, loop=False):
    filename = filename.split('.')[0]+'.mp4'
    prompt = prompt[0] if isinstance(prompt, list) else prompt

    ## save video
    videos = [samples]
    savedirs = [fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        # b,c,t,h,w
        video = video.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
        if loop:
            video = video[:-1,...]
        
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, h, n*w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        path = os.path.join(savedirs[idx], filename)
        torchvision.io.write_video(path, grid, fps=fps, video_codec='h264', options={'crf': '10'}) ## crf indicates the quality


def save_results_seperate(prompt, samples, fakedir, fps=10, loop=False,no=0):
    # prompt = prompt[0] if isinstance(prompt, list) else prompt
    prompt = '_'.join([elem.replace(' ', '_') for elem in prompt if elem is not None]) if isinstance(prompt, list) else prompt
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    ## save video
    videos = [samples]
    savedirs = [fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        # b,c,t,h,w
        video = video.detach().cpu()
        if loop: # remove the last frame
            video = video[:,:,:-1,...]
        video = torch.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        for i in range(n):
            grid = video[i,...]
            grid = (grid + 1.0) / 2.0
            grid = (grid * 255).to(torch.uint8).permute(1, 2, 3, 0) #thwc
            path = os.path.join(savedirs[idx].replace('samples', 'samples_separate'), f'{prompt}_{no}_sample{i}_{timestamp}.mp4')
            torchvision.io.write_video(path, grid, fps=fps, video_codec='h264', options={'crf': '10'})

def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


def shift_latents(latents):
    # shift latents
    latents[:,:,:-1] = latents[:,:,1:].clone()

    # add new noise to the last frame
    latents[:,:,-1] = torch.randn_like(latents[:,:,-1])

    return latents


def prepare_latents(latents_dir, sampler,num_inference_steps,video_length):
    latents_list = []

    video = torch.load(latents_dir+f"/{num_inference_steps}.pt")

    for i in range(video_length // 2):
        alpha = sampler.ddim_alphas[0]
        beta = 1 - alpha
        latents = alpha**(0.5) * video[:,:,[i]] + beta**(0.5) * torch.randn_like(video[:,:,[i]])
        latents_list.append(latents)

    for i in range(num_inference_steps):
        alpha = sampler.ddim_alphas[i] # image -> noise
        beta = 1 - alpha
        frame_idx = max(0, i-(num_inference_steps - video_length))
        latents = (alpha)**(0.5) * video[:,:,[frame_idx]] + (1-alpha)**(0.5) * torch.randn_like(video[:,:,[frame_idx]])
        latents_list.append(latents)

    latents = torch.cat(latents_list, dim=2)

    return latents

def image_guided_synthesis(model, prompts, videos, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., \
                        unconditional_guidance_scale=1.0, cfg_img=None, fs=None, text_input=False, multiple_cond_cfg=False, loop=False, interp=False, timestep_spacing='uniform', guidance_rescale=0.0,  **kwargs):
    # if fifo:
    #     ddim_steps = noise_shape[2]*num_partitions

    if multiple_cond_cfg:
        ddim_sampler = DDIMSampler_multicond(model)
    else:
        ddim_sampler = DDIMSampler(model)

    
    if loop or interp:
        batch_size = noise_shape[0]
    else:
        batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    if not text_input:
        prompts = [""]*batch_size

    if loop or interp:
        img_1 = videos[:,:,0] #bchw
        img_2 = videos[:,:,-1] #bchw
        
        img = slerp(img_1,img_2,0.5)
        
        img_emb = model.embedder(img) ## blc
        img_emb = model.image_proj_model(img_emb)
        img_emb = rearrange(img_emb, 'b (t l) c -> b l t c', t=videos.shape[2])

        # img_emb_1 = model.embedder(img_1) ## blc
        # img_emb_1 = model.image_proj_model(img_emb_1)
        # img_emb_2 = model.embedder(img_2) ## blc
        # img_emb_2 = model.image_proj_model(img_emb_2)
        # alpha_list = list(torch.linspace(0.1, 0.9, videos.shape[2]))
        # img_emb = []
        # for alpha in alpha_list:
        #     img_emb_ = slerp(img_emb_1, img_emb_2, alpha)
        #     img_emb.append(img_emb_.unsqueeze(2))
        # img_emb = torch.cat(img_emb,dim=2)
        # img_emb_reserve = torch.flip(img_emb,dims=[2])
        # img_emb = torch.cat([img_emb,img_emb_reserve], dim=0)
        # img_1 = videos[:,:,0] #bchw
        # img_2 = videos[:,:,-1] #bchw
        # img = torch.cat([img_1,img_2],dim=0)
        # img_emb = model.embedder(img) ## blc
        # img_emb = model.image_proj_model(img_emb)
        # img_emb = repeat(img_emb,'b l c')
     
    else:
        # raise ValueError
        img = videos[:,:,0] #bchw
        img_emb = model.embedder(img) ## blc
        img_emb = model.image_proj_model(img_emb)

    cond_emb =[]
    for prompt in prompts:
        prompt_full, prompt_inverse, prompt_1, prompt_2 = prompt
        cond_emb_full, _ , cond_emb_1,cond_emb_2=None, None, None, None
        if prompt_full is not None:
            cond_emb_full = model.get_learned_conditioning(prompt_full)
        if prompt_inverse is not None:
            cond_emb_inverse = model.get_learned_conditioning(prompt_inverse)
        if prompt_1 is not None:
            cond_emb_1 = model.get_learned_conditioning(prompt_1)
        if prompt_2 is not None:
            cond_emb_2 = model.get_learned_conditioning(prompt_2)

        if cond_emb_1 is not None and cond_emb_2 is not None:
            # 使用 slerp 方法进行插值
            alpha_list = list(torch.linspace(0.1, 0.9, videos.shape[2]))
            cond_emb_ = []
            for alpha in alpha_list:
                cond_emb_.append(slerp(cond_emb_1, cond_emb_2, alpha).unsqueeze(2))
            cond_emb_1_ = torch.cat(cond_emb_,dim=2)
            cond_emb_2_ = torch.flip(cond_emb_1_,dims=[2])

        # elif cond_emb_1 is not None:
        #     cond_emb_ = cond_emb_1
        # elif cond_emb_2 is not None:
        #     cond_emb_ = cond_emb_2
        # else:
        #     cond_emb_ = None
        
        if cond_emb_full is not None and cond_emb_ is not None:
            cond_emb_full = repeat(cond_emb_full.unsqueeze(2), 'b c t l -> b c (repeat t) l', repeat=cond_emb_1_.shape[2])
            cond_emb_full = 0.5*cond_emb_full+0.5*cond_emb_1_
        elif cond_emb_ is not None:
            cond_emb_full = cond_emb_1_
        cond_emb.append(cond_emb_full)

        # if cond_emb_inverse is not None and cond_emb_ is not None:
        #     cond_emb_inverse = repeat(cond_emb_inverse.unsqueeze(2), 'b c t l -> b c (repeat t) l', repeat=cond_emb_2_.shape[2])
        #     cond_emb_inverse = 0.5*cond_emb_inverse+0.5*cond_emb_2_
        # elif cond_emb_ is not None:
        #     cond_emb_inverse = cond_emb_2_
        # cond_emb.append(cond_emb_inverse)

    cond_emb = torch.cat(cond_emb, dim=0)
    cond = {"c_crossattn": [torch.cat([cond_emb,img_emb], dim=1)]}
    if model.model.conditioning_key == 'hybrid':
        # videos_reversed = torch.flip(videos, dims=[2])
        # videos = torch.cat([videos,videos_reversed],dim=0)
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
        if len(uc_emb.shape) == len(uc_img_emb.shape):
            uc = {"c_crossattn": [torch.cat([uc_emb,uc_img_emb],dim=1)]}
        else:
            uc_img_emb = repeat(uc_img_emb.unsqueeze(2), 'b c t h w -> b c (repeat t) h w', repeat=uc_emb.shape[2])
            uc = {"c_crossattn": [torch.cat([uc_emb,uc_img_emb],dim=1)]}

        
        if model.model.conditioning_key == 'hybrid':
            uc["c_concat"] = [img_cat_cond]
    else:
        uc = None

    ## we need one more unconditioning image=yes, text=""
    if multiple_cond_cfg and cfg_img != 1.0:
        if len(uc_emb.shape) == len(img_emb.shape):
            uc_2 = {"c_crossattn": [torch.cat([uc_emb,img_emb],dim=1)]}
        else:
            uc_emb = repeat(uc_emb.unsqueeze(2), 'b c t h w -> b c (repeat t) h w', repeat=img_emb.shape[2])
            uc_2 = {"c_crossattn": [torch.cat([uc_emb,img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc_2["c_concat"] = [img_cat_cond]
        kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})

    z0 = None
    cond_mask = None

    batch_variants = []
    for _ in range(n_samples):

        if z0 is not None:
            cond_z0 = z0.clone()
            kwargs.update({"clean_cond": True})
        else:
            cond_z0 = None
        if ddim_sampler is not None:
                samples, _ = ddim_sampler.sample(S=ddim_steps,
                                                conditioning=cond,
                                                batch_size=batch_size,
                                                shape=noise_shape[1:],
                                                verbose=True,
                                                unconditional_guidance_scale=unconditional_guidance_scale,
                                                unconditional_conditioning=uc,
                                                eta=ddim_eta,
                                                cfg_img=cfg_img, 
                                                mask=cond_mask,
                                                x0=cond_z0,
                                                fs=fs,
                                                timestep_spacing=timestep_spacing,
                                                guidance_rescale=guidance_rescale,
                                                **kwargs
                                                )


        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)
    ## variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants)
    batch_variants = batch_variants.permute(1, 0, 2, 3, 4, 5)

    return batch_variants

def run_inference(args, gpu_num, gpu_no):
    ## model config
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    
    ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
    model_config['params']['unet_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    model.perframe_ae = args.perframe_ae
    assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()
    
    perceptual_loss = LPIPS().cuda(gpu_no)
    
    eps = 1 / (args.video_length - 1)
    fact = 1 / (eps ** 2)
    
    ## run over data
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    assert args.bs == 1, "Current implementation only support [batch size = 1]!"

    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.out_channels
    n_frames = args.video_length
    print(f'Inference with {n_frames} frames')
    noise_shape = [args.bs, channels, n_frames, h, w]

    fakedir = os.path.join(args.savedir, "samples")
    fakedir_separate = os.path.join(args.savedir, "samples_separate")

    # os.makedirs(fakedir, exist_ok=True)
    os.makedirs(fakedir_separate, exist_ok=True)
    eval_list = load_json(args.json_file)
    # for data in data_list:
    #     args.prompt = data.get('prompts')
    #     args.prompt_inverse = data.get('prompt_inverse')
    #     args.prompt_1 = data.get('prompt_1')
    #     args.prompt_2 = data.get('prompt_2')
    #     args.video_dir_1 = data['video_dir_1']
    #     args.video_dir_2 = data['video_dir_2']

    data_list, prompt_list, save_names_list = load_data_prompts(eval_list,video_size=(args.height, args.width), video_frames=n_frames, interp=args.interp)
    num_samples = len(prompt_list)
    samples_split = num_samples // gpu_num
    print('Prompts testing [rank:%d] %d/%d samples loaded.'%(gpu_no, samples_split, num_samples))
    #indices = random.choices(list(range(0, num_samples)), k=samples_per_device)
    indices = list(range(samples_split*gpu_no, samples_split*(gpu_no+1)))
    prompt_list_rank = [prompt_list[i] for i in indices]
    data_list_rank = [data_list[i] for i in indices]
    save_names_rank = [save_names_list[i] for i in indices]

    start = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for idx, indice in tqdm(enumerate(range(0, len(prompt_list_rank), args.bs)), desc='Sample Batch'):
            prompts = prompt_list_rank[indice:indice+args.bs]
            videos = data_list_rank[indice:indice+args.bs]
            save_names = save_names_rank[indice:indice+args.bs]
            if isinstance(videos, list):
                videos = torch.stack(videos, dim=0).cuda(gpu_no)
            else:
                videos = videos.unsqueeze(0).cuda(gpu_no)

            batch_samples = image_guided_synthesis(model, prompts, videos, noise_shape, args.n_samples, args.ddim_steps, args.ddim_eta, \
                                args.unconditional_guidance_scale, args.cfg_img, args.frame_stride, args.text_input, args.multiple_cond_cfg, args.loop, args.interp, \
                                args.timestep_spacing, args.guidance_rescale)

            prompt = save_names[0]
            
            # first_frame = batch_samples[0,:,:,0]
            # last_frame = batch_samples[1,:,:,0]
            
            samples_1 = batch_samples[0]

                    
            save_results_seperate(prompt, samples_1, fakedir, fps=8, loop=args.loop, no=0)

    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")


def get_parser():
    parser = argparse.ArgumentParser()
    

    
    
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--json_file", type=str, required=True, help="path to the JSON file containing prompts and video directories")
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