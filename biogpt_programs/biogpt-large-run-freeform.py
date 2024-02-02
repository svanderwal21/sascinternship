import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("BioGPT-Large")
model = AutoModelForCausalLM.from_pretrained("BioGPT-Large")

# Input sequence
input_sequence = "Mitochondria are"
inputs = tokenizer.encode(input_sequence, return_tensors='pt')

# Initialize variables
max_length = 50  # You can adjust this as needed
eos_token_id = tokenizer.eos_token_id
attention_mask = tokenizer(input_sequence).attention_mask
attention_mask = torch.tensor([attention_mask])
past_key_values = None
generated_sequence = []

# Text generation loop
count=0
with torch.no_grad():
    while True:
        count+=1
        if count > 1:
            inputs=input_token
        model_out = model(input_ids=inputs, attention_mask=attention_mask, past_key_values=past_key_values)
        logits = model_out.logits[:, -1, :]
        past_key_values = model_out.past_key_values

        # Sample a token from the distribution
        probabilities = F.softmax(logits, dim=-1)
        input_token = torch.multinomial(probabilities, num_samples=1)

        # Append generated token and update attention mask
        generated_sequence.append(input_token.item())
        attention_mask = torch.cat((attention_mask, torch.ones((1, 1), dtype=torch.long)), dim=1)

        # Check for end of sequence or max length
        if input_token.item() == eos_token_id or len(generated_sequence) >= max_length:
            break

        inputs = input_token.unsqueeze(0)

# Decode generated tokens into text
decoded_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
print("Generated Text:", decoded_text)

