import pandas as pd
from Bioconductor import *
entrez_to_hgnc = org.Hs.eg.db

from transformers import Trainer, TrainingArguments

# Read the .gmt file into a pandas DataFrame
df = pd.read_csv('wikipathways-20231210-gmt-Homo_sapiens.gmt', sep='\t')

# Extract the genes and their associated GO terms
genes = df['Gene'].tolist()
go_terms = df['Term'].tolist()

# Convert Entrez Gene IDs into HGNC Gene Symbols
entrez_to_hgnc = entrez_to_hgnc.asDF()
entrez_ids = genes
hgnc_symbols = entrez_to_hgnc[entrez_ids.isin(entrez_to_hgnc['ENTREZID'])]['SYMBOL']

# Replace Entrez Gene IDs with HGNC Gene Symbols in the dataframe
df['Gene'] = hgnc_symbols.tolist()

# Create a list of training instances
training_instances = []

# Iterate over the genes and their associated GO terms
for gene, go_term in zip(genes, go_terms):
    # Create a training instance
    training_instance = {'gene': gene, 'go_term': go_term}

    # Add the training instance to the list of training instances
    training_instances.append(training_instance)

# Convert the training dataset to a JSON file
import json
with open('pathways_with_hgnc_symbols.json', 'w') as f:
    json.dump(training_instances, f)

# Load the training dataset from the JSON file
with open('pathways_with_hgnc_symbols.json', 'r') as f:
    training_dataset = json.load(f)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./pathways_model',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=0.1,
    logging_dir='./pathways_logs',
    logging_steps=100
)

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=len(set(training_instances['go_term'])))

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_dataset,
    eval_dataset=training_dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

