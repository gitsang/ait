from transformers import pipeline

translation = pipeline(task="translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh", device=-1)
 
text = "I like to learn data science and AI."
translated_text = translation(text)
print(translated_text)
