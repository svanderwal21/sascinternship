from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("BioGPT-Large")
tokenizer = AutoTokenizer.from_pretrained("BioGPT-Large")

# Encode some input text
inputs = tokenizer("Hello, I am", return_tensors="pt")

# Perform inference
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "My name is Wolfgang and I live in Berlin"

ner_results = nlp(example)
print(ner_results)
# Process the outputs as needed for your application

