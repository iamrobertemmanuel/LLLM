from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils import load_config
import chromadb
import os
import shutil

config = load_config()

def get_embeddings():
    # Using a model with 768 dimensions
    return HuggingFaceEmbeddings(model_name="sentence-transformers/bert-base-nli-mean-tokens")

def reset_vectordb():
    db_path = config["chromadb"]["chromadb_path"]
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        print(f"Removed existing database at {db_path}")

def load_vectordb(embeddings=get_embeddings()):
    # Reset the database if it exists
    reset_vectordb()
    
    persistent_client = chromadb.PersistentClient(config["chromadb"]["chromadb_path"])

    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=config["chromadb"]["collection_name"],
        embedding_function=embeddings,
    )

    return langchain_chroma