from transformers import pipeline

# 1. Load a ready-made sentiment model from Hugging Face.
#    The first run downloads it (about 250 MB). Later runs are fast.
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

# 2. Some sentences to test
texts = [
    "I love this phone, the battery lasts all day!",
    "The delivery was late and the box was damaged.",
    "The movie was okay, nothing special.",
]

# 3. Predict and print
for text in texts:
    result = classifier(text)[0]
    print(f"{text}\n  -> {result['label']} ({result['score']:.1%} confident)\n")