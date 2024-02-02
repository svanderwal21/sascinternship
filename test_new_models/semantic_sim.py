from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import numpy as np
import torch


def calculate_similarity(sentences, model):
    embedding_GPTv4 = model.encode(sentences[0], convert_to_tensor=True)
    embedding_model_local = model.encode(sentences[1], convert_to_tensor=True)

    score = util.pytorch_cos_sim(embedding_GPTv4, embedding_model_local)

    return score

def gpt4_reading():
    gpt4_list = []
    gpt4_file = open("gpt4_gui1.txt", "r").readlines()
    temp_list=[]
    first=True
    for line in gpt4_file:
        line = line.strip('"').strip("\n")
        #if there is 1 then begin a new list, every 1 is new gene set
        if line.split(".")[0] == "1" and first==False:
            gpt4_list.append('\n'.join(temp_list))
            temp_list=[]
        temp_list.append(line)
        first=False
    gpt4_list.append('\n'.join(temp_list))
    return gpt4_list

def genes_reading():
    genes_list = []
    genes_file = open("genes_test_set.txt", "r").readlines()
    for genes in genes_file:
        genes = genes.strip("\n")
        genes_list.append(genes)
    return genes_list

def gen_gpt4_dic():
    gpt4_list = gpt4_reading()
    genes_list = genes_reading()
    #make dict for GPT4 results with key the gene list and value the results
    gpt4_dict = {key: value for key, value in zip(genes_list, gpt4_list)}
    #print(gpt4_dict)
    return gpt4_dict

            
def local_models_reading(remove_prompt):
    local_file = open("out_for_semantic.txt", "r").readlines()

    #split with @, those are how the results are designed to be split.
    models = local_file[0].split("@")
    big_dict = {}


    #skip first one because that one is the model list
    for line in local_file[1:]:
        local_dict = {}
        temp_result = []
        #results are seperated by @
        line = line.split('@')

        genes = line[0]
        results = line[1:]

        for result, model in zip(results, models[1:]):
            #rint(result.strip("[").strip("]".strip('"')))
            result = result.strip("[").strip("]").strip('"').strip()
            #this bit makes sure only result is result, so remove prompt
            if remove_prompt == True:
                result = result.split("Formulate like this:")[1]
            local_dict[model] = result
        
        big_dict[genes] = local_dict

    
    return big_dict
        
def pre_semantic(gpt4_dict, big_dict, model):
    #dict maken, {model : score}
    results_dic = {}
    for genes_key in big_dict:
        result_gpt4 = gpt4_dict[genes_key]
        #in this dictionary store {gene set : score}
        for model_key in big_dict[genes_key]:
            #in this bit we can compare gpt4 result per gene set vs gpt4 results per gene set per model
            result_local = big_dict[genes_key][model_key]
            #do .item because it returns tensor type and this returns int
            score = calculate_similarity([result_gpt4, result_local], model).item()
            if results_dic.get(model_key) == None:
                results_dic[model_key] = [score]
            else:
                results_dic[model_key].append(score)

    #if you want the mean result and not multiple things
    #for key in results_dic:
        #results_dic[key] = sum(results_dic[key])/len(results_dic[key])
    return results_dic
            

def make_plots(result_dic):
    placeholder_data = result_dic
    unique_colors = plt.cm.viridis(np.linspace(0, 1, len(placeholder_data["biogpt"])))
    # Setting the same color for 'Score 1' across all models
    common_color = 'skyblue'
    colors = [common_color] + list(unique_colors[1:])

    # Creating the plots with unique colors for each bar
    max_score = max(max(scores) for scores in placeholder_data.values())

    # Now, we'll recreate the plots, ensuring that all have the same y-axis scale.
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 15), sharey=True)  # Share y-axis across all plots

    # Flattening the axes array for easy iteration and plotting
    axes = axes.flatten()

    # Plotting each model in its own subplot
    for idx, (model, scores) in enumerate(placeholder_data.items()):
        ax = axes[idx]
        ax.bar(range(len(scores)), scores, color=colors)
        ax.set_title(f'Scores for {model}')
        ax.set_xlabel('Score Index')
        ax.set_ylabel('Scores')
        ax.set_xticks(range(len(scores)))
        ax.set_xticklabels([f'Score {i+1}' for i in range(len(scores))], rotation=45)
        ax.set_ylim(0, max_score + 0.1)  # Use the same y-axis limit for all subplots

    # Hide the last subplot (bottom right) since we only have 5 models
    axes[-1].axis('off')

    # Hide the last subplot (bottom right) since we only have 5 models
    axes[-1].axis('off')

    plt.tight_layout()

    #plt.tight_layout()
    plt.savefig("bar_all.png", format="png")


def plots_line(results_dic):
    data = results_dic
    plt.figure(figsize=(10, 6))

    # Plotting each model in the same figure
    for model, scores in data.items():
        plt.plot(scores, label=model)

    plt.xlabel('Score Index')
    plt.ylabel('Scores')
    plt.title('Comparison of Scores for Models')
    plt.xticks(range(len(scores)), [f'Score {i+1}' for i in range(len(scores))])
    plt.legend()
    plt.savefig("line_all.png", format="png")



def main():
    name_m = "BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    name_m = "all-mpnet-base-v2"
    model = SentenceTransformer(name_m)


    gpt4_dict = gen_gpt4_dic()
    #remove_prompt = False to not remove the prompt that was given to the model, if True only get generated results.
    big_dict = local_models_reading(remove_prompt=True)
    result_dic = pre_semantic(gpt4_dict, big_dict, model)
    #print(result_dic)
    make_plots(result_dic)
    plots_line(result_dic)
    


main()

