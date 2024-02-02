#from github of BioGPT, HuggingFace transformers library, for text generation
from transformers import pipeline, set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM
model = BioGptForCausalLM.from_pretrained("/exports/sascstudent/svanderwal2/programs/BioGPT-Large", cache_dir="/exports/sacstudent/svanderwal2/")
tokenizer = BioGptTokenizer.from_pretrained("/exports/sascstudent/svanderwal2/programs/BioGPT-Large", cache_dir="/exports/sascstudent/svanderwal2")

#Prompt is GUI1
#Fine-tuned models for PubMedQA or other benchmarks can be downloaded
#In this script test performance of normal BioGPT with benchmarks
#Testing PubMedQA, added the last part
q1 = "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death? "
#a1 = yes

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
#generator = pipeline('question-answering', model=model, tokenizer=tokenizer)


#set_seed(42)
gen = generator(q1, max_length=150, num_return_sequences=1, do_sample=True)
for iter in gen:
        print(iter["generated_text"])
        print("------------------------------------------")

