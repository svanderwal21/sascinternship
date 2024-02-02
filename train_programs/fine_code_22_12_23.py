import pandas as pd
import json
import mygene
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

model_path = "/exports/sascstudent/svanderwal2/programs/BioMedLM"

# Initialize MyGeneInfo
mg = mygene.MyGeneInfo()

# Read the .gmt file into a pandas DataFrame
df = pd.read_csv('wikipathways-20231210-gmt-Homo_sapiens.gmt', sep='\t', header=None)
df.columns = ['Pathway', 'Description', 'Gene']

# Extract the genes
genes = df['Gene'].tolist()

# Query MyGeneInfo to convert Entrez Gene IDs into HGNC Gene Symbols
gene_info = mg.querymany(genes, scopes='entrezgene', fields='symbol', species='human')

# Create a mapping from Entrez IDs to HGNC symbols
entrez_to_hgnc = {item['query']: item['symbol'] for item in gene_info if 'symbol' in item}

# Replace Entrez Gene IDs with HGNC Gene Symbols in the dataframe
df['Gene'] = df['Gene'].map(entrez_to_hgnc)

# Create a list of training instances
training_instances = df.to_dict(orient='records')

# Split data into training and validation sets
train_instances, val_instances = train_test_split(training_instances, test_size=0.2)

# Function to compute metrics for evaluation
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Define the training arguments
training_args = TrainingArguments(
    output_dir=model_path,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    logging_dir='./pathways_logs',
    logging_steps=100
)

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(set(df['Description'])))

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_instances,
    eval_dataset=val_instances,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Evaluate the model
evaluation_results = trainer.evaluate()
print(evaluation_results)

