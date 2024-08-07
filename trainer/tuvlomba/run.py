from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

model_name_or_path = "lora_model"
tokenizer_name_or_path = "lora_model"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

input_tmpl = """
In the extracted of `{}`, we found `{}` from `{}` file `{}:{}` by config `{}`.
Please tell me the risk.
"""


def generate_answer(item):
    input_text = input_tmpl.format(
        item['file']['name'],
        item['detect']['result']['text'],
        item['detect']['result']['type'],
        item['detect']['location']['path'],
        item['detect']['location']['line'],
        item['detect']['config']['name'],
    )
    inputs = tokenizer(
        [input_text],
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    outputs = model.generate(**inputs, max_new_tokens=2048, use_cache=True)
    decoded_output = tokenizer.batch_decode(
        outputs, skip_special_tokens=True)[0]
    return decoded_output.split('<|im_end|>')[0].strip()


item = {
            "file": {
                "name": "dect-server-146.85.249.177.zip"
            },
            "detect": {
                "config": {
                    "name": "ip_detect",
                    "regexs": []
                },
                "location": {
                    "path": "/dectserver",
                    "line": 5913
                },
                "result": {
                    "type": "ELF 32-bit LSB executable",
                    "text": "1.1.1.1",
                    "context": "1.1.1.1[%d]speed test start"
                },
            }
        }
answer = generate_answer(item)
print("---")
print(answer)
