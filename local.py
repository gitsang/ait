from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import pipeline

model_path = "./test_trainer/checkpoint-375/"

config = AutoConfig.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, config=config)
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

input_text = "这是一个测试句子。"
# inputs = tokenizer(input_text, return_tensors="pt")
# outputs = model(**inputs)

pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=0)
output = pipe(input_text)
print(output)
