from transformers import pipeline
import os

os.environ["HF_TOKEN"] = ''

pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B")

result = pipe("Hello, my name is", max_length=30, num_return_sequences=2)
print(result)
