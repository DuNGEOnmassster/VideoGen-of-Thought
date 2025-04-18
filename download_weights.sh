# For VGoT
huggingface-cli download --resume-download Doubiiu/DynamiCrafter_1024 --local-dir weights/DynamiCrafter
huggingface-cli download --resume-download Kwai-Kolors/Kolors --local-dir weights/Kolors
huggingface-cli download --resume-download Kwai-Kolors/Kolors-IP-Adapter-Plus --local-dir weights/Kolors-IP-Adapter-Plus
huggingface-cli download --resume-download OpenGVLab/ViCLIP-B-16-hf --local-dir weights/ViCLIP-B-16-hf

# For VGoT with FramePack
huggingface-cli download --resume-download hunyuanvideo-community/HunyuanVideo --local-dir weights/HunyuanVideo
huggingface-cli download --resume-download lllyasviel/FramePackI2V_HY --local-dir weights/FramePackI2V_HY
huggingface-cli download --resume-download lllyasviel/flux_redux_bfl --local-dir weights/flux_redux_bfl
