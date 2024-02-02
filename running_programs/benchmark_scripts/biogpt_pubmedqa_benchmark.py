def run(q1):
	#from github of BioGPT, HuggingFace transformers library, for text generation
	from transformers import pipeline, set_seed
	from transformers import BioGptTokenizer, BioGptForCausalLM
	model = BioGptForCausalLM.from_pretrained("/exports/sascstudent/svanderwal2/programs/BioGPT-Large-PubMedQA", cache_dir="/exports/sacstudent/svanderwal2/")
	tokenizer = BioGptTokenizer.from_pretrained("/exports/sascstudent/svanderwal2/programs/BioGPT-Large-PubMedQA", cache_dir="/exports/sascstudent/svanderwal2")

	#model = BioGptForCausalLM.from_pretrained("/exports/sascstudent/svanderwal2/programs/BioGPT-Large", cache_dir="/exports/sacstudent/svanderwal2/")
	#tokenizer = BioGptTokenizer.from_pretrained("/exports/sascstudent/svanderwal2/programs/BioGPT-Large", cache_dir="/exports/sascstudent/svanderwal2")



	#Prompt is GUI1
	#Fine-tuned models for PubMedQA or other benchmarks can be downloaded
	#In this script test performance of normal BioGPT with benchmarks
	#bit before Q:
	bi = "Conclude with yes, maybe or no: "
	#Testing PubMedQA, added the last part
	#q1 = "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death? "
	#a1 = yes
	#q2 = "Landolt C and snellen e acuity: differences in strabismus amblyopia?"
	#a2 = no
	#q3 = "Syncope during bathing in infants, a pediatric form of water-induced urticaria?"
	#a3 = yes
	#q4 = "Are the long-term results of the transanal pull-through equal to those of the transabdominal pull-through?"
	#a4 = no

	generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
	#generator = pipeline('question-answering', model=model, tokenizer=tokenizer)


	#set_seed(42)
	gen = generator(bi+q1, max_length=250, num_return_sequences=1, do_sample=True)
	return gen


