from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

model = "/exports/sascstudent/svanderwal2/programs/BioLinkBERT-base"

tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForQuestionAnswering.from_pretrained(model)

inputs = tokenizer("What is my dog's name?", "Hello, my dog is cute. He is named Fluffy.", return_tensors='pt')
outputs = model(**inputs)

start_logits = outputs.start_logits
end_logits = outputs.end_logits

# Find the tokens with the highest `start` and `end` scores
start_index = torch.argmax(start_logits)
end_index = torch.argmax(end_logits) + 1  # add 1 to include the end token

# Convert tokens to the answer string
answer_tokens = inputs.input_ids[0, start_index:end_index]
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(answer_tokens))

print(answer)

