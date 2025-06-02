# /workspaces/LLLM/vectordb_handler.py

# --- START OF SQLITE3 FIX FOR CHROMADB ---
# This block MUST be at the very top of the file, before any other imports
# that might indirectly or directly import 'sqlite3' (like chromadb).
import os
import sys
# Explicitly import pysqlite3 and replace the default sqlite3 in sys.modules
__import__('pysqlite3') 
sys.modules['sqlite3'] = sys.modules['pysqlite3']
# --- END OF SQLITE3 FIX ---

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils import load_config
import chromadb
import shutil # Already present in your code, keeping it.

# Load configuration globally
config = load_config()

def get_embeddings():
    """
    Returns a HuggingFaceEmbeddings instance for document embedding.
    Requires 'sentence-transformers' to be installed.
    """
    return HuggingFaceEmbeddings(model_name=config["gemini"]["embedding_model"]) 
    # NOTE: Your config.yaml still specifies "embedding-001" under gemini.
    # If you intend to use a HuggingFace model, you should put its name
    # in config.yaml under a specific HuggingFace section, or directly use
    # "sentence-transformers/bert-base-nli-mean-tokens" here.
    # For now, I'm assuming config["gemini"]["embedding_model"] will be
    # the HuggingFace model name if you're using this.
    # For example: HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def reset_vectordb():
    """Removes the existing ChromaDB persistent directory."""
    db_path = config["chromadb"]["chromadb_path"]
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        print(f"Removed existing database at {db_path}")
    else:
        print(f"No existing database found at {db_path} to remove.")

def load_vectordb(embeddings=None):
    """
    Loads or initializes the ChromaDB vector store.
    If 'embeddings' is not provided, it defaults to HuggingFaceEmbeddings.
    """
    # NOTE: Calling reset_vectordb() here will delete your database
    # every time load_vectordb() is called (e.g., on app startup).
    # If you want persistent data, you should remove this line
    # and call reset_vectordb() only when explicitly needed.
    reset_vectordb() 
    
    # Use provided embeddings or get default if not provided
    if embeddings is None:
        embeddings = get_embeddings()

    # Initialize ChromaDB client with persistent path from config
    # Ensure the directory exists before initializing client
    db_path = config["chromadb"]["chromadb_path"]
    os.makedirs(db_path, exist_ok=True)
    
    persistent_client = chromadb.PersistentClient(path=db_path)

    # Initialize LangChain's Chroma wrapper
    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=config["chromadb"]["collection_name"],
        embedding_function=embeddings,
    )

    print(f"ChromaDB loaded with collection: {config['chromadb']['collection_name']}")
    return langchain_chroma

# You can add a simple test/initialization call here if needed
# For Streamlit apps, load_vectordb will usually be called in app.py
# or chat_api_handler.py
