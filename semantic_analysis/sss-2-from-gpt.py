from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity

# Load the tokenizer and model
model = "/exports/sascstudent/svanderwal2/programs/BioLinkBERT-base"

tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModel.from_pretrained(model)

# Function to generate embeddings
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Aggregate token embeddings

# Texts to compare
text1 = "Mitochondria"
text2 = "Powerhouse of the cell"

# Generate embeddings
embeddings1 = get_embeddings(text1)
embeddings2 = get_embeddings(text2)

# Compute cosine similarity
similarity = cosine_similarity(embeddings1, embeddings2)
print(similarity)

