import json
from datasets import Dataset

prompt = """
# {}

## Detection

### Config: {}

{}

### Location

{}:{}

### 2.3 Result

#### Type

{}

#### Text

{}

#### Context

{}

### Details

```json
{}
```

### Risk

Type: {}

Ignore: {}

#### Confidence

score: {}

{}

#### Security

score: {}

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
                item['detect']['config']['name'],
                item['detect']['config']['regexs'],
                item['detect']['location']['path'],
                item['detect']['location']['line'],
                item['detect']['result']['type'],
                item['detect']['result']['text'],
                item['detect']['result']['context'],
                json.dumps(item['detect']['details']),
                item['risk']['type'],
                item['risk']['ignore'],
                item['risk']['confidence']['score'],
                item['risk']['confidence']['gists'],
                item['risk']['security']['score'],
                item['risk']['security']['gists'],
            ) + self.tokenizer.eos_token
            texts.append(text)

        dataset_dict = {
            'text': texts
        }

        dataset = Dataset.from_dict(dataset_dict)
        return dataset

    def get_dataset(self):
        return self.dataset
