from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-large-cased-v1.1-squad")
model = AutoModelForQuestionAnswering.from_pretrained("dmis-lab/biobert-large-cased-v1.1-squad")

# Example question and context
question = "Which process or pathway describes these genes best"
context = "UBE2I TRIP12 MARCHF7"

# Tokenize the input question and context
inputs = tokenizer(question, context, return_tensors="pt")

# Get model predictions
outputs = model(**inputs)
start_logits = outputs.start_logits
end_logits = outputs.end_logits

# Find the answer span with the highest probability
start_index = start_logits.argmax()
end_index = end_logits.argmax()
answer_span = inputs["input_ids"][0][start_index:end_index + 1]

# Decode the answer span back to text
answer = tokenizer.decode(answer_span, skip_special_tokens=True)

# Print the predicted answer
print("Predicted Answer:", answer)

