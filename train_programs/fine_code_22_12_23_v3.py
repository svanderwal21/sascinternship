import pandas as pd
import mygene
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Model path
model_path = "/exports/sascstudent/svanderwal2/programs/BioMedLM"

# Function to read .gmt file
def read_gmt_file(file_path):
    pathways = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            pathway_name = parts[0]
            genes = parts[4:]  # Adjust if necessary based on your file's structure
            for gene in genes:
                pathways.append({'Pathway_Name': pathway_name, 'Gene': gene})
    return pd.DataFrame(pathways)

# Read the .gmt file
df = read_gmt_file('wikipathways-20231210-gmt-Homo_sapiens.gmt')

# Initialize MyGeneInfo
mg = mygene.MyGeneInfo()

# Convert Entrez Gene IDs to HGNC Gene Symbols
def convert_gene_ids(gene_id):
    result = mg.query(gene_id, scopes='entrezgene', fields='symbol', species='human')
    return result.get('symbol', gene_id)

df['HGNC_Symbol'] = df['Gene'].apply(convert_gene_ids)

# Prepare the training data
training_instances = df[['Pathway_Name', 'HGNC_Symbol']].dropna()

# Split the data into training and validation sets
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
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(set(df['Pathway_Name'])))

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_instances.to_dict('records'),
    eval_dataset=val_instances.to_dict('records'),
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

# Train and evaluate the model
trainer.train()
evaluation_results = trainer.evaluate()
print(evaluation_results)

