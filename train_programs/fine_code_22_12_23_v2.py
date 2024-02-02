import pandas as pd
import json
import mygene
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Model path
model_path = "/exports/sascstudent/svanderwal2/programs/BioMedLM"

# Read the .gmt file into a pandas DataFrame
columns = ['Pathway_Name', 'Source', 'Pathway_ID', 'Organism', 'Pathway_URL', 'Genes']
df = pd.read_csv('wikipathways-20231210-gmt-Homo_sapiens.gmt', sep='\t', names=columns, header=None)

# Convert the 'Genes' column into a list of genes for each pathway
df['Genes'] = df['Genes'].apply(lambda x: x.split())

# Initialize MyGeneInfo
mg = mygene.MyGeneInfo()

# Function to query MyGeneInfo and convert Entrez Gene IDs to HGNC Gene Symbols
def convert_gene_ids(genes):
    gene_info = mg.querymany(genes, scopes='entrezgene', fields='symbol', species='human')
    return [item['symbol'] if 'symbol' in item else None for item in gene_info]

# Apply conversion to each row in the dataframe
df['HGNC_Symbols'] = df['Genes'].apply(convert_gene_ids)

# Expand the dataframe so each gene has its own row
df_expanded = df.explode('HGNC_Symbols')
df_expanded.dropna(subset=['HGNC_Symbols'], inplace=True)

# Create training instances
training_instances = df_expanded[['Pathway_Name', 'HGNC_Symbols']].to_dict(orient='records')

# Split data into training and validation sets
train_instances, val_instances = train_test_split(training_instances, test_size=0.2)

# Define metrics function for evaluation
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

# Define training arguments
training_args = TrainingArguments(
    output_dir='./pathways_model',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    logging_dir='./pathways_logs',
    logging_steps=100
)

# Load tokenizer and model from the specified model path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(df['Pathway_Name'].unique()))

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_instances,
    eval_dataset=val_instances,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

# Train and evaluate the model
trainer.train()
evaluation_results = trainer.evaluate()
print(evaluation_results)

