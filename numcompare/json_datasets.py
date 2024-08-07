import json
from datasets import Dataset

input_prompt = """下面列出了一个问题. 请写出问题的答案.
### 问题:
{}
### 答案:
{}"""

label_prompt = """{}"""


class LocalJsonDataset:
    def __init__(self, json_file, tokenizer):
        self.json_file = json_file
        self.tokenizer = tokenizer
        self.dataset = self.load_dataset()

    def load_dataset(self):
        with open(self.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        inputs = []
        labels = []
        for item in data:
            inputs.append(item['question'])
            labels.append(item['answer'])

        dataset = Dataset.from_dict({
            "input_text": inputs,
            "label_text": labels
        })
        return dataset

    def get_dataset(self):
        return self.dataset
