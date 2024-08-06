from transformers import AutoTokenizer

def tokenize(model_name, sentences):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    print("Encoded Input: ", encoded_input)

def main():
    tokenize("google-bert/bert-base-uncased", "In a hole in the ground there lived a hobbit.")
    tokenize("google-bert/bert-base-cased", "In a hole in the ground there lived a hobbit.")
    tokenize("google-bert/bert-base-cased", [
            "But what about second breakfast?",
            "Don't think he knows about second breakfast, Pip.",
            "What about elevensies?",
        ]
    )

if __name__== "__main__" :
    main()
