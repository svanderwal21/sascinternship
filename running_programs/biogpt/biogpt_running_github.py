#from github of BioGPT, HuggingFace transformers library, for text generation
from transformers import pipeline, set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM
model = BioGptForCausalLM.from_pretrained("/exports/sascstudent/svanderwal2/programs/BioGPT-Large", cache_dir="/exports/sacstudent/svanderwal2/")
tokenizer = BioGptTokenizer.from_pretrained("/exports/sascstudent/svanderwal2/programs/BioGPT-Large", cache_dir="/exports/sascstudent/svanderwal2")

#Prompt is GUI1
input = """You are an efficient and insightful assistant to a molecular biologist

Propose a brief name for the 10 most prominent biological processes performed by the system.

One process can have multiple genes

Be concise; do not use unnecessary words. Be specific; avoid overly general
statements such as 'the proteins are involved in various cellular processes'

Formulate like this:

1. <process 1: genes present>
2. <process 2: genes present>

etc.

Here are the interacting proteins: UBE2I TRIP12 MARCHF7"""

input_engineered = """
You are an efficient and insightful assistant to a molecular biologist
Propose a brief name for the 10 most prominent biological processes performed by the system.
Here are the interacting proteins: UBE2I TRIP12 MARCHF7
Process 1: 
"""

input_eng_2 = """
You are an efficient and insightful assistant to a molecular biologist
Propose a brief name for the 10 most prominent biological processes performed by the system.
For every process give a reasoning and give which genes are present in the process
Here are the interacting proteins: UBE2I TRIP12 MARCHF7

"""
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
#set_seed(42)
gen = generator(input_engineered, max_length=800, num_return_sequences=1, do_sample=True)
for iter in gen:
        print(iter["generated_text"])
        print("------------------------------------------")

