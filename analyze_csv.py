import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

# 1. Load the reviews
df = pd.read_csv("reviews.csv")

# 2. Load the model
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

# 3. Analyze every review at once
results = classifier(df["text"].tolist(), truncation=True)

# 4. Add the results as new columns
df["sentiment"] = [r["label"] for r in results]
df["confidence"] = [round(r["score"], 3) for r in results]

print(df)

# 5. Save the results to a new file
df.to_csv("results.csv", index=False)

# 6. Draw a bar chart of positive vs negative counts
df["sentiment"].value_counts().plot(kind="bar", color=["green", "red"])
plt.title("Sentiment of reviews")
plt.ylabel("Number of reviews")
plt.xticks(rotation=0)
plt.show()