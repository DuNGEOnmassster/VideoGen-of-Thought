import os, torch
# from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers import EulerDiscreteScheduler

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_prompt_list(prompt_file):
    with open(prompt_file, 'r') as f:
        prompt_list = f.readlines()
    prompt_list = [p.strip() for p in prompt_list]
    return prompt_list



def infer(prompt):
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

    prompt = """Character: Olivia, in her early teens, is a budding designer sketching her first dress. Her face is filled with concentration, her brows slightly furrowed as she works diligently on a piece of notebook paper. She wears casual clothing, and her hair is loosely tied back, reflecting her relaxed but focused nature.\nBackground: The scene takes place in Olivia's room, with fabrics and sketches scattered around, a small space that serves as her creative haven. There is a gentle warmth in the environment, symbolizing the nurturing of her early dreams.\nRelation: Olivia's relationship with her work is full of curiosity and passion. This is the beginning of her journey, where every line on the paper feels like a step toward something bigger.\nCamera Pose: The camera captures Olivia from the side, her hand moving steadily as she sketches. It’s a medium close-up, focusing on her expression and her sketching hand, capturing the essence of her concentration.\nHDR Description: The lighting is soft, warm, and natural, with 8K HDR capturing the details of her youthful features, the pencil strokes on the notebook, and the warmth of the space around her. Shadows are subtle, adding depth to her expression of focus."""
    
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=5.0,
        num_images_per_prompt=1,
        generator= torch.Generator(pipe.device).manual_seed(66)).images[0]
    image.save(f'/storage/home/mingzhe/code/VideoGen-of-Thought/data/data_fashion_designer/avatar_Olivia_young.jpg')


if __name__ == '__main__':
    import fire
    fire.Fire(infer)
