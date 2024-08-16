import transformers
import torch

model_id = "/shome/zhengmz/pretrained_models/pretrained/meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
  "text-generation",
  model=model_id,
  model_kwargs={"torch_dtype": torch.bfloat16},
  device="cuda",
)