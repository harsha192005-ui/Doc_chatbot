import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

# We will use the FastAPI tutorial as our data source
# We can add more URLs here for a larger knowledge base
URLS = [
    # --- Framework: FastAPI ---
    "https://fastapi.tiangolo.com/tutorial/first-steps/",
    "https://fastapi.tiangolo.com/tutorial/path-params/",
    "https://fastapi.tiangolo.com/tutorial/query-params/",
    "https://fastapi.tiangolo.com/tutorial/body/",
    "https://fastapi.tiangolo.com/tutorial/response-model/",
    "https://fastapi.tiangolo.com/tutorial/handling-errors/",
    
    # --- Programming Language: Python ---
    "https://docs.python.org/3/tutorial/controlflow.html",
    "https://docs.python.org/3/tutorial/datastructures.html",
    "https://docs.python.org/3/tutorial/modules.html",
    "https://docs.python.org/3/tutorial/classes.html",
    "https://docs.python.org/3/tutorial/errors.html",
    "https://docs.python.org/3/tutorial/venv.html",
    
    # --- Software Library: Pandas ---
    "https://pandas.pydata.org/docs/user_guide/10min.html",
    "https://pandas.pydata.org/docs/getting_started/intro_tutorials/01_table_oriented.html",
    "https://pandas.pydata.org/docs/getting_started/intro_tutorials/02_read_write.html",
    "https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html",
    "https://pandas.pydata.org/docs/getting_started/intro_tutorials/04_plotting.html",
    "https://pandas.pydata.org/docs/getting_started/intro_tutorials/08_combine_dataframes.html"
]

def ingest_data():
    print("Loading documents...")
    loader = WebBaseLoader(URLS)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    splits = text_splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunks.")

    print("Creating vector store...")
    # Use local HuggingFace embeddings instead of Google API for vector store to avoid API issues
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create and persist ChromaDB vector store
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    
    print("Ingestion complete. Vector store created at ./chroma_db")

if __name__ == "__main__":
    ingest_data()
