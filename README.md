# VideoGen-of-Thought (VGoT)

Official implementation of *VideoGen-of-Thought: Step-by-step generating multi-shot video with minimal manual intervention*, old version titled: *VideoGen-of-Thought: A Collaborative Framework for Multi-Shot Video Generation*

## 📣 News

* ⏳⏳⏳ Release the latest version of arxiv paper

* `[2025.03.19]`  We publish official code and detailed instruction & evaluation

* `[2024.12.03]`  🔥 We release the arXiv paper for VGoT, and you can click [here](https://arxiv.org/abs/2412.02259) to see more details.


## Abstract

Current video generation models excel at short clips but fail to produce cohesive multi-shot narratives due to disjointed visual dynamics and fractured storylines. 
Existing solutions either rely on extensive manual scripting/editing or prioritize single-shot fidelity over cross-scene continuity, limiting their practicality for movie-like content. 
We introduce VideoGen-of-Thought *(VGoT)*, a step-by-step framework that automates multi-shot video synthesis **from a single sentence** by systematically addressing three core challenges: 
**(1) Narrative Fragmentation**: Existing methods lack structured storytelling. We propose **dynamic storyline modeling**, which first converts the user prompt into concise shot descriptions, then elaborates them into detailed, cinematic specifications across five domains (character dynamics, background continuity, relationship evolution, camera movements, HDR lighting), ensuring logical narrative progression with self-validation.
**(2) Visual Inconsistency**: Existing approaches struggle with maintaining visual consistency across shots. Our **identity-aware cross-shot propagation** generates identity-preserving portrait (IPP) tokens that maintain character fidelity while allowing trait variations (expressions, aging) dictated by the storyline.
**(3) Transition Artifacts**: Abrupt shot changes disrupt immersion. Our **adjacent latent transition mechanisms** implement boundary-aware reset strategies that process adjacent shots' features at transition points, enabling seamless visual flow while preserving narrative continuity.
By integrating these innovations into a training-free pipeline, VGoT generates multi-shot videos that outperform state-of-the-art baselines by **20.4% in within-shot face consistency** and **17.4% in style consistency**, while achieving **over 100% better cross-shot consistency** and **10× fewer manual adjustments** than alternatives like MovieDreamer and DreamFactory. 
Our work redefines automated long-video generation, bridging the gap between raw visual synthesis and director-level storytelling.

<p align="center">
<img src="./assets/teaser/Illustration.png" width="800px"/>
<br>
<b>Illustration of <i>VideoGen-of-Thought (VGoT)</i>.</b>
</p>

## ⚙️ Preparations

We recommend the requirements as follows.

### Environment

```bash
# 0. Clone the repo
git clone --depth=1 https://github.com/DuNGEOnmassster/VideoGen-of-Thought.git
cd VideoGen-of-Thought

# 1. Create conda environment
conda create -n VideoGen-of-Thought python=3.10
conda activate VideoGen-of-Thought

# (Optional) Install PyTorch and other dependencies using conda, we test on cuda 11.8 and cuda 12.1
# CUDA 11.8
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia

# 2. Install pip dependencies, we have already provided paired PyTorch and xformers in our requirements.
pip install -r requirements.txt
```

### Download Pretrained Weights

We appriciate the amazing open-source work by [Kolor](https://github.com/Kwai-Kolors/Kolors.git) and [DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter.git), for the current version of *VGoT*, you can directly download the pretrained weights through:

```bash
sh download_weights.sh
```

Once ready, the weights will be organized in this format:

```
📦 weights/
├── 📂 DynamiCrafter/
├──── 📄 model.ckpt
├──── ...
├── 📂 Kolors/
├──── 📂 scheduler/
├──── 📂 text_encoder/
├──── 📂 tokenizer/
├──── 📂 unet/
├──── 📂 vae/
├──── 📄 model_index.json
├──── ...
├── 📂 Kolors-IP-Adapter-Plus/
├──── 📂 image_encoder/
├──── 📄 ip_adapter_plus_general.bin
├──── 📄 model_index.json
├──── 📄 config.json
├──── ...
├── 📂 ViCLIP-B-16-hf/
├──── ...
```

### Prepare GPT-4o token

Don't forget to create a file ``configs/config.txt`` to place your GPT-4o token, please refer to [this guide](https://github.com/DuNGEOnmassster/VideoGen-of-Thought/blob/main/configs/READMD.md)

## ✨ Inference

<p align="center">
<img src="./assets/teaser/Flowchart-v2.png" width="800px"/>
<br>
<b>The FlowChart of <i>VideoGen-of-Thought (VGoT)</i>.</b>
</p>

