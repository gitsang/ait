from transformers import pipeline
import json

vision_classifier = pipeline(model="google/vit-base-patch16-224")
preds = vision_classifier(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)

print(json.dumps(preds, indent=2))
