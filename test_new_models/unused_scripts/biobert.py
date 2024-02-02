# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline(model="biobert-base-cased-v1.1-squad",task="text-generation")
a = pipe("a")
print(a)
