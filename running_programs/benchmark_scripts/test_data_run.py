import json
import biogpt_pubmedqa_benchmark as pubmed
import biogpt_withtorch_for_qa as torch_qa
with open("ori_pqal.json", "r") as json_file:
	my_dict = json.load(json_file)
torch_txt = open("torch_results.txt", "w")
pipeline_txt = open("pipeline_results.txt", "w")
i=-1
keys = list(my_dict)
while i < 10:
	i+=1
	q = keys[i]
	question = my_dict[q]["QUESTION"]
	context = my_dict[q]["CONTEXTS"] #list
	decision = my_dict[q]["final_decision"]
	long_ans = my_dict[q]["LONG_ANSWER"]
	#print(question, context)
	torch_txt.write(torch_qa.run(question, context[0]))
	pipeline_txt.write(str(pubmed.run(question)))
torch_txt.close()
pipeline_txt.close()
