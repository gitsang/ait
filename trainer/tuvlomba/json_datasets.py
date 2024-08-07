import json
from datasets import Dataset

input_prompt = """
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

labels_prompt = "{}"


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
            input_text = input_prompt.format(
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
            inputs.append(input_text)

            label_text = labels_prompt.format(
                item['risk']['confidence']['score']
            )
            labels.append(label_text)

        dataset = Dataset.from_dict({
            "input_text": inputs,
            "label_text": labels
        })
        return dataset

    def get_dataset(self):
        return self.dataset
