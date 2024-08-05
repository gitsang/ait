from transformers import pipeline

vision_classifier = pipeline(model="google/vit-base-patch16-224")
translation = pipeline(task="translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh")

preds = vision_classifier(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)

for pred in preds:
    score = pred["score"]
    label = pred["label"]
    zh_label = translation(label)[0]["translation_text"]
    if len(zh_label) > 100:
        zh_label = label
    print(f"{score:.2}: {zh_label} ({label})")
