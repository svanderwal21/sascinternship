import torch
from transformers import AutoModelForSequenceClassification
import joblib
from sklearn.preprocessing import LabelEncoder

def tokenize_input(text, gene_vocabulary, max_seq_length=512):
    # Split the text into gene names
    gene_names = text.split()
    # Convert gene names to IDs using the gene vocabulary
    tokenized_sequence = [gene_vocabulary.get(gene, 0) for gene in gene_names]
    # Truncate or pad the sequence to the max sequence length
    tokenized_sequence = tokenized_sequence[:max_seq_length] + [gene_vocabulary["[PAD]"]] * (max_seq_length - len(tokenized_sequence))
    attention_mask = [1 if token != gene_vocabulary["[PAD]"] else 0 for token in tokenized_sequence]
    return tokenized_sequence, attention_mask

def convert_to_tensors(tokenized_sequence, attention_mask):
    input_ids = torch.tensor([tokenized_sequence], dtype=torch.long)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long)
    return input_ids, attention_mask


def read(label, vocab):
    gene_vocab = {}
    with open(vocab, "r") as vocab_file:
        for line in vocab_file:
            line = line.split("@")
            gene_vocab[line[0]] = int(line[-1].strip("\n"))

    label_enc = {}
    with open(label, "r") as label_file:
        for line in label_file:
            line = line.split("\t")
            label_enc[int(line[0])] = line[1]

    return label_enc, gene_vocab

def load_model(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model

def predict(model, input_ids, attention_mask):
    model.eval()  # Put the model in evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
    return predictions

def interpret_prediction(prediction, label_encoder):
    label = label_encoder.inverse_transform(prediction.cpu().numpy())
    return label

#to do list
#Output label encoder

def main():
    #test text
    text = 'UBE2I TRIP12 MARCHF7'
    text = "MUC21 MARCHF7 HLA-DRB4"
    print("Input: ", text)

    #get information from the training part
    label_enc = "labels_enc.txt"
    gene_vocab="gene_vocab_biolinkbert.txt"
    label_encoder, gene_vocabulary = read(label_enc, gene_vocab)

    #tokenize input
    tokenized_sequence, attention_mask = tokenize_input(text, gene_vocabulary)
    
    #convert to tensors
    input_ids, attention_mask = convert_to_tensors(tokenized_sequence, attention_mask)

    #load model
    model_path = "trained_models_BERT/"
    model = load_model(model_path)

    #interfere
    file_write = open("BERT_classify_pathways_results_10.txt", "w")
    file_write.write(text+"\n")
    for tr in range(10):
        prediction = predict(model, input_ids, attention_mask)
        print(prediction.item())
        #get label
        label = label_encoder.get(prediction.item())

        file_write.write("Predicted label "+str(tr)+": "+label)


main()