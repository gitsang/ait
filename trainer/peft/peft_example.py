
!pip install torch
!pip install transformers
!pip install datasets
!pip install evaluate
!pip install accelerate
!pip install flake8
!pip install ffmpeg
!pip install wheel
!pip install ffmpeg
!pip install Pillow
!pip install sentencepiece
!pip install sacremoses
!pip install accelerate
!pip install bitsandbytes
!pip install ffmpeg
!pip install numpy
!pip install peft
!pip install datasets

# |%%--%%| <t3HEGjlmCF|ZqeOKFKTgK>

# configure
model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"
output_dir = "sang/mt0-large-lora"
dataset_name="yelp_review_full"

# |%%--%%| <ZqeOKFKTgK|Cx4flL0AAB>

# initialize
from peft import LoraConfig, TaskType
from peft import get_peft_model
from transformers import AutoModelForSeq2SeqLM

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

# |%%--%%| <Cx4flL0AAB|9ZPV8HI3lQ>

# datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq

dataset = load_dataset(dataset_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# |%%--%%| <9ZPV8HI3lQ|eUoY7PppA2>

# metrics
import numpy as np
import evaluate

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# |%%--%%| <eUoY7PppA2|d8CLv64bL3>

# trainer
from transformers import TrainingArguments, Trainer

# https://huggingface.co/docs/transformers/v4.43.4/en/main_classes/trainer#transformers.TrainingArguments
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# https://huggingface.co/docs/transformers/v4.43.4/en/main_classes/trainer#transformers.Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained(output_dir)


# |%%--%%| <d8CLv64bL3|nLqQNHUeAy>

model = AutoModelForCausalLM.from_pretrained(output_dir)
