from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils import load_config
import chromadb
import os

config = load_config()

def get_google_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def load_vectordb(embeddings=get_google_embeddings()):
    persistent_client = chromadb.PersistentClient(config["chromadb"]["chromadb_path"])

    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=config["chromadb"]["collection_name"],
        embedding_function=embeddings,
    )

    return langchain_chroma