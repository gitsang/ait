from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = model.tokenize(sequence)
print(tokens)
