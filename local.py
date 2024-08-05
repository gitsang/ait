from transformers import AutoModel, AutoTokenizer, AutoConfig

model_path = "./test_trainer/checkpoint-375/"

config = AutoConfig.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_path)

input_text = "这是一个测试句子。"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model(**inputs)
generated_ids = model.generate(inputs['input_ids'])
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_text)
