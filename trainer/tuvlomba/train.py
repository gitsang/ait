from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments
from json_datasets import LocalJsonDataset

# configure
model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"
lora_model_dir = "lora_model"
trainer_output_dir = "output_dir"
dataset_path = "train_data.json"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["input_text"], padding=True, truncation=True
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["label_text"], padding=True, truncation=True
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    # datasets
    custom_dataset = LocalJsonDataset(
        json_file=dataset_path,
        tokenizer=tokenizer,
    )
    dataset = custom_dataset.get_dataset()
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # split the dataset into train and eval
    train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    # trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=TrainingArguments(
            output_dir=trainer_output_dir,
            eval_strategy="epoch",
        ),
    )

    # train
    trainer.train()
    model.save_pretrained(lora_model_dir)
    tokenizer.save_pretrained(lora_model_dir)


if __name__ == "__main__":
    main()
