from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
sentences = ["Regulation of transcription by RNA polymerase II", "RNA polymerase II-mediated transcription"]



model = SentenceTransformer('/exports/sascstudent/svanderwal2/programs/all-MiniLM-L6-v2-pubmed-full')
#model = AutoModel.from_pretrained('/exports/sascstudent/svanderwal2/programs/SapBERT-from-PubMedBERT-fulltext')
#/exports/sascstudent/svanderwal2/programs/all-MiniLM-L6-v2-pubmed-full

embeddings = model.encode(sentences, convert_to_tensor=True)
embeds = util.pytorch_cos_sim(embeddings[0], embeddings[1])
print(embeds)
