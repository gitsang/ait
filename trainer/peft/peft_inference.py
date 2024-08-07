from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

model_name_or_path = "sang/mt0-large-lora"
tokenizer_name_or_path = "bigscience/mt0-large"

model = AutoPeftModelForCausalLM.from_pretrained(model_name_or_path).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

model.eval()
inputs = tokenizer("Preheat the oven to 350 degrees and place the cookie dough", return_tensors="pt")

outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
