import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

from app import get_vectorstore, init_chain

# Load env variables just in case
load_dotenv()

def run_tests():
    print("Loading vector store...")
    vectorstore = get_vectorstore()
    
    print("Initializing LLM chain...")
    chain, retriever = init_chain(vectorstore)
    
    print("\n" + "="*50)
    print("TEST 1: Relevant Question (Should answer using FastAPI docs)")
    print("="*50)
    q1 = "How do I create a path parameter?"
    print(f"Question: {q1}\n")
    
    print("Retrieved sources:")
    docs = retriever.invoke(q1)
    for i, doc in enumerate(docs, 1):
        print(f"  {i}. {doc.metadata.get('source')} (Snippet: {doc.page_content[:60]}...)")
    
    print("\nLLM Answer:")
    response1 = chain.invoke(q1)
    print(response1)
    
    
    print("\n" + "="*50)
    print("TEST 2: Irrelevant Question (Should decline to answer)")
    print("="*50)
    q2 = "What is the capital of France?"
    print(f"Question: {q2}\n")
    
    print("Retrieved sources:")
    docs2 = retriever.invoke(q2)
    for i, doc in enumerate(docs2, 1):
         print(f"  {i}. {doc.metadata.get('source')} (Snippet: {doc.page_content[:60]}...)")
         
    print("\nLLM Answer:")
    response2 = chain.invoke(q2)
    print(response2)
    print("="*50 + "\n")

if __name__ == "__main__":
    run_tests()
