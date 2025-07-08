from transformers import pipeline
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summarizer(chat_text, max_length=60, min_length=20)[0]['summary_text']
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(chat_text)
entities = [(ent.text, ent.label_) for ent in doc.ents]
from transformers import pipeline
sentiment = pipeline("sentiment-analysis")
sentiment_result = sentiment(chat_text)
from sentence_transformers import SentenceTransformer
import faiss
embedder = SentenceTransformer('all-MiniLM-L6-v2')

docs = [...]  # list of support articles
doc_embeddings = embedder.encode(docs)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

query_vec = embedder.encode([summary])
_, I = index.search(query_vec, k=1)
context_doc = docs[I[0][0]]
input_text = f"Customer issue: {summary}\nReference: {context_doc}\nResponse:"
response = summarizer(input_text, max_length=100)[0]['summary_text']
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=16, lora_dropout=0.1)
model = get_peft_model(model, lora_config)
