from transformers import AutoTokenizer, AutoModelForCausalLM

name = "BioMedGPT-LM-7B"
model = AutoModelForCausalLM.from_pretrained(name)
tokenizer = AutoTokenizer.from_pretrained(name)
print(model.generation_config)

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

generate_ids = model.generate(inputs.input_ids, max_length=30)
z = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(z)
