from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments
from datasets import LocalJsonDataset

# configure
model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"
lora_model_dir = "lora_model"
trainer_output_dir = "test_trainer"
dataset_path = "train_data.json"
max_seq_length = 2048

# model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# datasets
custom_dataset = LocalJsonDataset(
    json_file=dataset_path,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length
)
dataset = custom_dataset.get_dataset()

# trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        output_dir=trainer_output_dir,
        eval_strategy="epoch"
    ),
)

# train
trainer.train()
model.save_pretrained(lora_model_dir)
tokenizer.save_pretrained(lora_model_dir)
