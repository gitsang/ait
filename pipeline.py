from transformers import pipeline

classifier = pipeline(task="sentiment-analysis", device=-1)
preds = classifier(["I am really today!", "Great!"])
print(preds)
