from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from vectordb_handler import load_vectordb
from utils import load_config, timeit
import pypdfium2 as pdfium
import streamlit as st

config = load_config()

def extract_text_from_pdf(pdf_bytes):
    try:
        pdf = pdfium.PdfDocument(pdf_bytes)
        text_pages = []
        for page in pdf:
            textpage = page.get_textpage()
            text = textpage.get_text_range()
            text_pages.append(text)
        return "\n".join(text_pages)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def get_pdf_texts(pdfs_bytes):
    texts = []
    for pdf_bytes in pdfs_bytes:
        text = extract_text_from_pdf(pdf_bytes)
        if text:
            texts.append(text)
    return texts

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=st.session_state.chunk_size, 
                                              chunk_overlap=st.session_state.chunk_overlap,
                                              separators=["\n", "\n\n"])
    return splitter.split_text(text)

def get_document_chunks(texts):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    documents = []
    for text in texts:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            documents.append(Document(page_content=chunk))
    return documents

@timeit
def add_documents_to_db(pdfs):
    try:
        pdfs_bytes = [pdf.read() for pdf in pdfs]
        texts = get_pdf_texts(pdfs_bytes)
        if not texts:
            st.error("No text could be extracted from the PDFs")
            return
        documents = get_document_chunks(texts)
        vector_db = load_vectordb()
        vector_db.add_documents(documents)
        st.success("Documents added to database successfully!")
    except Exception as e:
        st.error(f"Error adding documents to database: {str(e)}")
        # Reset the vector database in case of error
        from vectordb_handler import reset_vectordb
        reset_vectordb()