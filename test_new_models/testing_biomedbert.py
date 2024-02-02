import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import csv

# Load pre-trained model tokenizer (vocabulary)
tokenizer = AutoTokenizer.from_pretrained("BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")

# Load pre-trained model (weights)
model = AutoModelForMaskedLM.from_pretrained("BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
model.eval()

# Encode text
prompt_og = "In cellular biology, the interaction between %s primarily contributes to [MASK]."

nested_list = [
        ["UBE2I", "TIP12", "MARCHF7"],
        ["MUC21", "MARCHF7", "HLA-DRB4"],
        ["TRIP12", "ZNF275", "TTC21B", "CELF1", "UBE2I", "BRPF1", "OVOL3"],
        ["TTC21B", "HPS4", "LOC100653049", "CCDC39", "HECW2", "UBE2I"],
        ["TTC21B", "ZNF275", "UBE2I", "BRPF1", "OVOL3"],
        ["VARS2", "POLR2J3", "SEM1", "TRIP12", "ZNF275", "TTC21B", "CELF1", "UBE2I", "BRPF1", "OVOL3", "MLYCD", "PRPF18"],
        ["TRIP12", "TTC21B", "MYLIP", "MARCHF7", "LOC100653049", "CELF1", "SRGAP2B", "CCDC39", "HECW2", "BRPF1", "OVOL3"]
    ]

#nested_list = [nested_list[0]]
results = []
for genes in nested_list:
    prompt = prompt_og % ', '.join(genes)

    #print(prompt)
    #text = "The cell structure of [MASK] is unique because of its mitochondria."
    inputs = tokenizer.encode_plus(prompt, return_tensors="pt")

    # Predict all tokens
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits

    # Find the predicted token (we replace the first mask, in case of multiple ones)
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = predictions[0, mask_token_index].topk(5).indices[0].tolist()

    # Decode the predicted token
    predicted_token = tokenizer.decode(predicted_token_id)

    # Replace mask token with the predicted token
    result = prompt.replace(tokenizer.mask_token, predicted_token)
    results.append(result)


to_write = [genes, [result]]
out="biomedbert_test.csv"
with open(out, "w", newline='') as file:
    writer = csv.writer(file, delimiter=";")
    for result_int in range(len(results)):
        genes = nested_list[result_int]
        result = results[result_int]
        to_write = [genes, [result]]
        writer.writerow(to_write)











