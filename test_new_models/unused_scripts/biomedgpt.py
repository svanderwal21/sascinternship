import multi_testing
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForQuestionAnswering, AutoModel
import torch
#BioMedGPT-LM-7B
tokenizer = AutoTokenizer.from_pretrained("BioMedGPT-LM-7B")
model = AutoModelForCausalLM.from_pretrained("BioMedGPT-LM-7B")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(model.config._name_or_path)
prompt = """
fart
"""
model.to(device)
encoded_input = tokenizer(prompt, return_tensors='pt').input_ids
encoder_input = encoded_input.to(device)
output_sequences = model.generate(
        input_ids=encoded_input,
        max_length=150,
        temperature=0.1,
        num_return_sequences=1,
        do_sample=True)
generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print(generated_text)

