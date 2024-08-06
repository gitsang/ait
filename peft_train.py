from peft import LoraConfig, TaskType
from peft import get_peft_model
from transformers import AutoModelForSeq2SeqLM

model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"
output_dir="sang/mt0-large-lora"
# model_name_or_path = "Qwen/Qwen2-0.5B-Instruct"
# tokenizer_name_or_path = "Qwen/Qwen2-0.5B-Instruct"
# output_dir = "sang/Qwen2-0.5B-Instruct-lora"

# https://huggingface.co/docs/peft/v0.12.0/en/package_reference/lora#peft.LoraConfig
peft_config = LoraConfig(
    # the task to train for
    # - SEQ_2_SEQ_LM: sequence-to-sequence language modeling
    task_type=TaskType.SEQ_2_SEQ_LM,
    # whether you’re using the model for inference or not
    inference_mode=True,
    # the dimension of the low-rank matrices
    r=8,
    # the scaling factor for the low-rank matrices
    lora_alpha=32,
    # the dropout probability of the LoRA layers
    lora_dropout=0.1
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
