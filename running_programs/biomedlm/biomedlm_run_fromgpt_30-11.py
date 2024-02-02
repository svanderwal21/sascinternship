from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def generate_biological_processes(model_path, input_sequence, temperature):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Encode input text to tensor
    input_ids = tokenizer.encode(input_sequence, return_tensors='pt')

    # Generate a sequence of text
    output = model.generate(input_ids, max_length=500, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, temperature=temperature, do_sample=True)

    # Decode the output to text
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return output_text

# Define the input sequence
input_sequence = """
Propose a brief name for the 10 most prominent biological processes performed by this system of interacting proteins

Here are the interacting proteins: UBE2I TRIP12 MARCHF7

Process 1:
"""

# Define the path to your model
model_path = "/exports/sascstudent/svanderwal2/programs/BioMedLM"

# Generate output
output_text = generate_biological_processes(model_path, input_sequence, temperature=0.1)
print(output_text)

