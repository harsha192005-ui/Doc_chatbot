import os
from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    grading_details: List[dict]

def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown source")
        formatted_docs.append(f"Source: {source}\nContent: {doc.page_content}")
    return "\n\n".join(formatted_docs)

def get_crag_app(vectorstore, temperature=0.2):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    crag_api_key = os.environ.get("GOOGLE_API_KEY_CRAG")
    if crag_api_key:
        llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=temperature, google_api_key=crag_api_key)
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=temperature)
    
    # 1. Relevance Grader
    class DocumentGrade(BaseModel):
        document_id: int = Field(description="The index of the document (0, 1, 2, etc.)")
        binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

    class GradeDocuments(BaseModel):
        """Binary scores for relevance check on retrieved documents."""
        grades: List[DocumentGrade]
        
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    
    system = """You are a grader assessing relevance of multiple retrieved documents to a user question. \n 
    If a document contains keyword(s) or semantic meaning related to the question, grade it as relevant ('yes'). \n
    Otherwise, grade it as 'no'."""
    
    grade_prompt = PromptTemplate(
        template=system + "\n\nRetrieved documents: \n\n {documents} \n\n User question: {question}",
        input_variables=["documents", "question"],
    )
    
    retrieval_grader = grade_prompt | structured_llm_grader
    
    # 2. Generator
    prompt_template = """You are a highly skilled Technical Documentation Assistant.
Your sole purpose is to answer the user's question using ONLY the provided context from the documentation.
You must be precise, professional, and directly answer the question.
If the answer cannot be found in the context, you must state: 
"I'm sorry, but I can only answer questions based on the provided documentation, and I couldn't find the answer to that question in my knowledge base."
Do not use your general knowledge.

Context:
{context}

Question: {question}

Answer:"""
    from langchain_core.output_parsers import StrOutputParser
    generate_prompt = PromptTemplate.from_template(prompt_template)
    rag_chain = generate_prompt | llm | StrOutputParser()
    
    # 3. Define Nodes
    def retrieve_node(state):
        question = state["question"]
        # Use similarity_search_with_score to get vector distances
        docs_and_scores = vectorstore.similarity_search_with_score(question, k=4)
        
        documents = []
        for doc, score in docs_and_scores:
            # ChromaDB returns distance (lower is closer). We'll store it in metadata.
            # Normalizing it to a pseudo-similarity percentage for the UI:
            similarity = max(0.0, 1.0 - (score / 2.0)) * 100 
            doc.metadata['vector_similarity'] = similarity
            documents.append(doc)
            
        return {"documents": documents, "question": question}
        
    def grade_node(state):
        question = state["question"]
        documents = state["documents"]
        
        # Prepare batch string
        docs_str = ""
        for i, d in enumerate(documents):
            docs_str += f"--- Document [{i}] ---\n{d.page_content}\n\n"
            
        filtered_docs = []
        details = []
        try:
            res = retrieval_grader.invoke({"question": question, "documents": docs_str})
            # Map grades
            grades_map = {g.document_id: g.binary_score for g in res.grades}
            
            for i, d in enumerate(documents):
                grade = grades_map.get(i, "yes") # Default to yes if missing
                is_relevant = (grade.lower() == "yes")
                if is_relevant:
                    filtered_docs.append(d)
                
                details.append({
                    "source": d.metadata.get("source", "Unknown"),
                    "snippet": d.page_content.replace('\n', ' ')[:100],
                    "relevant": is_relevant,
                    "similarity": d.metadata.get('vector_similarity', 0.0)
                })
        except Exception:
            # Fallback if grading fails
            filtered_docs = documents
            details = [{"source": d.metadata.get("source", "Unknown"), "snippet": d.page_content.replace('\n', ' ')[:100], "relevant": True, "similarity": d.metadata.get('vector_similarity', 0.0)} for d in documents]
            
        return {"documents": filtered_docs, "question": question, "grading_details": details}
        
    def generate_node(state):
        question = state["question"]
        documents = state["documents"]
        if not documents:
            return {"generation": "I'm sorry, but I can only answer questions based on the provided documentation, and I couldn't find the answer to that question in my knowledge base."}
            
        context = format_docs(documents)
        generation = rag_chain.invoke({"context": context, "question": question})
        return {"generation": generation.content if hasattr(generation, 'content') else generation}
        
    # 4. Build Graph
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_node)
    workflow.add_node("generate", generate_node)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("grade_documents", "generate")
    workflow.add_edge("generate", END)
    
    app = workflow.compile()
    return app
