from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

class CustomDataset(Dataset):
    def __init__(self, encodings, attention_masks, labels):
        self.encodings = encodings
        self.attention_masks = attention_masks
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: val[idx].clone().detach()
            for key, val in self.encodings.items()
        }
        item['attention_mask'] = torch.tensor(self.attention_masks[idx])
        item['labels'] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels)



def read_file_make_lists(file):
    data = []  # To hold processed records
    labels = []  # To hold pathway labels

    with open(file, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            pathway_name = parts[0].split('%')[0]  # Assuming the pathway name is the first part before '%'
            gene_ids = parts[2:]  # Assuming gene IDs are the last part of the line
            data.append(gene_ids)
            labels.append(pathway_name)

    # At this point, `data` contains lists of gene IDs and `labels` contains the pathway names.
    # Next steps would involve mapping labels to integers, tokenizing gene IDs, etc.

    return data, labels

def convert_ids(data, max_seq_length=512):
    gene_vocabulary = {gene: idx for idx, gene in enumerate(set([gene for sublist in data for gene in sublist]), 1)}
    gene_vocabulary["[PAD]"] = 0  # Adding a padding token

    file_gene_vocab = open("gene_vocab_biolinkbert.txt", "w")
    for x in gene_vocabulary:
        file_gene_vocab.write(str(x)+"@"+str(gene_vocabulary[x])+"\n")
    file_gene_vocab.close()

    tokenized_data = []
    attention_masks = []

    for gene_ids in data:
        truncated_ids = gene_ids[:max_seq_length]
        while len(truncated_ids) < max_seq_length:
            truncated_ids.append("[PAD]")  # Pad with [PAD] tokens

        tokenized_sequence = [gene_vocabulary.get(gene_id, 0) for gene_id in truncated_ids]
        tokenized_data.append(tokenized_sequence)

        # Create an attention mask
        attn_mask = [1] * len(tokenized_sequence) + [0] * (max_seq_length - len(tokenized_sequence))
        attention_masks.append(attn_mask)

    return gene_vocabulary, tokenized_data, attention_masks


def convert_labels(labels):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    write_l = open("labels_enc.txt", "w")
    for label in range(len(encoded_labels)):
        write_l.write(str(encoded_labels[label])+"\t"+str(labels[label])+"\n")
    write_l.close()
    return encoded_labels, label_encoder

def split_data(tokenized_data, encoded_labels):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(tokenized_data, encoded_labels, test_size=0.2, random_state=42)

    # Optionally, split the training set further into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
    split_list = [X_train, X_test, X_val, y_train, y_test, y_val]
    return split_list
    
def conv_to_tens(split_list, batch_size, attention_masks):
    # Convert inputs to dictionaries with 'input_ids' key
    train_encodings = {'input_ids': torch.tensor(split_list[0], dtype=torch.long)}
    validation_encodings = {'input_ids': torch.tensor(split_list[2], dtype=torch.long)}

    # Convert labels to tensors
    train_labels = torch.tensor(split_list[3], dtype=torch.long)
    validation_labels = torch.tensor(split_list[5], dtype=torch.long)

    # Create instances of CustomDataset
    train_data = CustomDataset(train_encodings, attention_masks, train_labels)
    validation_data = CustomDataset(validation_encodings, attention_masks, validation_labels)

    # Create DataLoaders
    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
    validation_dataloader = DataLoader(validation_data, sampler=SequentialSampler(validation_data), batch_size=batch_size)

    return train_data, validation_data


def train_func(model, batch_size, train_data, validation_data, encoded_labels):
    model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=len(set(encoded_labels)))


    training_args = TrainingArguments(
        output_dir='trained_models_BERT/',          # Output directory for model checkpoints
        num_train_epochs=5,              # Total number of training epochs
        per_device_train_batch_size=batch_size,  # Batch size per device during training
        per_device_eval_batch_size=batch_size,   # Batch size for evaluation
        warmup_steps=500,                # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # Strength of weight decay
        logging_dir='logs/',            # Directory for storing logs
        logging_steps=10,
        learning_rate=5e-5,
        lr_scheduler_type="linear",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        gradient_accumulation_steps = 4
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=validation_data,
)

    trainer.train()

    trainer.save_model("trained_models_BERT/")

    return trainer

def eval_test(split_list, batch_size, trainer, attention_masks):
    # Convert test data to the expected format
    test_encodings = {'input_ids': torch.tensor(split_list[1], dtype=torch.long)}
    test_attention_masks = attention_masks[len(split_list[0]) + len(split_list[2]):]  # Adjust index as needed

    # Convert test labels to tensors
    test_labels = torch.tensor(split_list[4], dtype=torch.long)

    # Create an instance of CustomDataset with the correct format
    test_data = CustomDataset(test_encodings, test_attention_masks, test_labels)

    # Evaluate the model
    eval_result = trainer.evaluate(test_data)
    for key, value in eval_result.items():
        print(f"{key}: {value}")


def main():
    #returns nested list with list of genes and list of all labels
    file = "custom_traintest.txt"
    data, labels = read_file_make_lists(file)
    print("Read files")

    #need to convert labels into numerical format
    encoded_labels, label_encoder = convert_labels(labels)
    print("Converted labels")

    #need to convert genes into text tokens, make own tokenizer?
    gene_vocab, tokenized_data, attention_masks = convert_ids(data)
    print("Tokenized data")

    #splitting data set
    split_list = split_data(tokenized_data, encoded_labels)
    print("Splitted data")

    #convert data into tensors
    batch_size = 12
    train_data, validation_data = conv_to_tens(split_list, batch_size, attention_masks)
    print("Data converted into tensors")

    #train
    model = "BioLinkBERT-large"
    trainer = train_func(model, batch_size, train_data, validation_data, encoded_labels)
    print("Model trained")

    #evaluate
    eval_test(split_list, batch_size, trainer, attention_masks)
    print("Model evaluated")

    #to interpret results need to reverse the gene vocab
    #if want to use model first convert genes with gene vocab, not reversed
    #tokenize > pad > truncate? > feed into model

main()
