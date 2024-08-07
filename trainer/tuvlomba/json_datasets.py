import json
from datasets import Dataset

input_tmpl = """
In the extracted of `{}`, we found `{}` from `{}` file `{}:{}` by config `{}`.
Please tell me the risk.
"""

labels_prompt = """
There is the risk of `{}`, with confidence `{}`, because:
{}
"""


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
            input_text = input_tmpl.format(
                item['file']['name'],
                item['detect']['result']['text'],
                item['detect']['result']['type'],
                item['detect']['location']['path'],
                item['detect']['location']['line'],
                item['detect']['config']['name'],
            ) + self.tokenizer.eos_token
            inputs.append(input_text)

            label_text = labels_prompt.format(
                item['risk']['type'],
                item['risk']['confidence']['score'],
                item['risk']['confidence']['gists'],
            ) + self.tokenizer.eos_token
            labels.append(label_text)

        dataset = Dataset.from_dict({
            "input_text": inputs,
            "label_text": labels
        })
        return dataset

    def get_dataset(self):
        return self.dataset
