from transformers import pipeline
def test_model(model_name, prompt):
    print(f"Testing model: {model_name}")
    generator = pipeline('text-generation', model=model_name)
    device = 0 if torch.cuda.is_available() and use_gpu else -1
    result = generator(prompt, max_length=500, num_return_sequences=1, device=device)
    print(result[0]['generated_text'])
    del generator

test_model("BioMedGPT-LM-7B", "fart")
