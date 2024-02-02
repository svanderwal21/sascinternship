import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

#device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("BioGPT-Large")
model = AutoModelForCausalLM.from_pretrained("BioGPT-Large")

input_sequence = "Mitochondria are"

print("CUDA available?"+torch.cuda.is_available())

inputs = torch.as_tensor(tokenizer.encode(input_sequence)).unsqueeze(0)
attention_mask = torch.as_tensor(tokenizer(input_sequence).attention_mask).unsqueeze(0)
past_key_values = None

count = 0
complete_token = []
with torch.no_grad():
    while count < 20:
        count += 1
        print("Iteration no.: " + str(count))
        if count > 1:
            inputs = input_token

        print(inputs)
        print(attention_mask)
        print(past_key_values[0][0].shape if past_key_values else None)

        model_out = model(input_ids=inputs, attention_mask=attention_mask, past_key_values=past_key_values)
        logits = model_out.logits[:, -1, :]
        past_key_values = model_out.past_key_values

        topk_values, topk_indices = torch.topk(logits, 5)

        log_probs = F.softmax(topk_values, dim=-1)
        inputs_in_topk = torch.multinomial(log_probs, num_samples=1, replacement=True)
        input_token = torch.gather(topk_indices, 1, inputs_in_topk)
        attention_mask = torch.concat((attention_mask, torch.tensor([[1]]).to(attention_mask)), dim=1)
        complete_token.append(input_token)
        complete_sequence = torch.cat(complete_token, dim=1).squeeze(0)
        decoded_sequence = tokenizer.decode(complete_sequence, skip_special_tokens=True)

        print("Generated Text:", decoded_sequence)
