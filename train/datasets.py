import json
from datasets import Dataset

prompt = """
### File Info

- Name: {}
- Path: {}
- Type: {}

### Risk

- Type: {}
- Confidence: {}
- Description: {}

### Matched Result

{}

### Context

{}
"""

class LocalJsonDataset:
    def __init__(self, json_file, tokenizer, max_seq_length=2048):
        self.json_file = json_file
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.dataset = self.load_dataset()

    def load_dataset(self):
        with open(self.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        texts = []
        for item in data:
            text = prompt.format(
                item['file']['name'],
                item['file']['path'],
                item['file']['type'],
                item['risk']['type'],
                item['risk']['confidence'],
                item['risk']['description'],
                item['matched_result'],
                item['context']
            ) + self.tokenizer.eos_token
            texts.append(text)

        dataset_dict = {
            'text': texts
        }

        dataset = Dataset.from_dict(dataset_dict)
        return dataset

    def get_dataset(self):
        return self.dataset
