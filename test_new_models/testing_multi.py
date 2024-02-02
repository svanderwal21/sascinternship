from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import csv

def variables():
    # Define the models you want to test
    models = [
        "BioGPT-Large-PubMedQA",
        "biogpt",
        "BioMedLM",
        "BioGPT-Large",
        "BioMedGPT-LM-7B"
    ]

    # The prompt you want to use for all models
    prompt_og="""
    You are an efficient and insightful assistant to a molecular biologist

    Propose a brief name for the 10 most prominent biological processes performed by the system.

    One process can have multiple genes

    Be concise; do not use unnecessary words. Be specific; avoid overly general
    statements such as 'the proteins are involved in various cellular processes'

    Here are the interacting proteins: %s

    Formulate like this:

    1. 
    """

    nested_list = [
        ["UBE2I", "TRIP12", "MARCHF7"],
        ["MUC21", "MARCHF7", "HLA-DRB4"],
        ["TRIP12", "ZNF275", "TTC21B", "CELF1", "UBE2I", "BRPF1", "OVOL3"],
        ["TTC21B", "HPS4", "LOC100653049", "CCDC39", "HECW2", "UBE2I"],
        ["TTC21B", "ZNF275", "UBE2I", "BRPF1", "OVOL3"],
        ["VARS2", "POLR2J3", "SEM1", "TRIP12", "ZNF275", "TTC21B", "CELF1", "UBE2I", "BRPF1", "OVOL3", "MLYCD", "PRPF18"],
        ["TRIP12", "TTC21B", "MYLIP", "MARCHF7", "LOC100653049", "CELF1", "SRGAP2B", "CCDC39", "HECW2", "BRPF1", "OVOL3"]
    ]
    return models, prompt_og, nested_list


# Function to load a model, run the prompt, and then unload the model
def test_model(model_name, prompt):
    print(f"Testing model: {model_name}")

    #load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #load model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    #if cuda is avaiable device value is 0
    if torch.cuda.is_available():
        print("Cuda avail")
        model.to("cuda")

    #input encoding
    inputs = tokenizer(prompt, return_tensors="pt")

    #if GPU avaiable move inputs to GPU
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

    #generate text
    output_sequences = model.generate(**inputs, max_length=500, num_return_sequences=1)

    #decoding output sequences
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    return(generated_text)

def writer(big_list, out, models):
    models.insert(0, " ")
    #print(big_list)
    with open(out, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(models)
        for row in big_list:
            genes = row[0]
            empty=[]
            for result in row[1]:
                empty.append([result])
            empty.insert(0, genes)
            writer.writerow(empty)

    #print(big_list)
    file = open("out_for_semantic.txt", "w")
    file.write("@".join(models)+"\n")
    for row in big_list:
        genes = ' '.join(row[0])
        result_str = ""
        for result in row[1:]:
            result_str+="@".join(result)
        result_str = str([result_str])
        joined=genes+"@"+result_str
        file.write(joined+"\n")

    file.close()



def main():
    big_list=[]
    #get all variables for running
    models, prompt_og, nested_list = variables()
    # Run the test for each model
    for x in nested_list:
        list_per_genes=[]
        list_result=[]
        list_per_genes.append(x)
        #make full prompt by adding genes
        prompt = prompt_og % ', '.join(x)
        print("Prompt:\n"+prompt)

        for model in models:
            result = test_model(model, prompt)
            list_result.append(result)
        list_per_genes.append(list_result)
        big_list.append(list_per_genes)

    file_name_out="out.csv"
    writer(big_list, file_name_out, models)

main()

