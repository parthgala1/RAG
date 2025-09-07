from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import pickle 
import faiss
import numpy as np
import os
from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Load environment variables from .env file
load_dotenv()

class QueryRequest(BaseModel):
    query: str

def load_pdf(file_path):
    """
    Load a PDF file and return its content as a list of documents.
    
    Args:
        file_path (str): The path to the PDF file.
        
    Returns:
        list: A list of documents loaded from the PDF.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def chunking(documents):
    """
    Split a list of documents into smaller chunks.
    
    Args:
        documents (list): A list of documents to be split.
        
    Returns:
        list: A list of split documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len, add_start_index=True)
    return text_splitter.split_documents(documents)

def get_save_embeddings():
    """
    Initialize the HuggingFaceEmbeddings model and save it to a file.
    
    Returns:
        HuggingFaceEmbeddings: The initialized embeddings model.
    """
    if os.path.exists("embeddings_model.pkl"):
        return pickle.load(open("embeddings_model.pkl", "rb"))
    
    else: 
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Save the embeddings model to a file
        with open("embeddings_model.pkl", "wb") as f:
            pickle.dump(embeddings, f)
    
    return embeddings

def load_to_faiss_index(embeddings, documents):
    """
    Load documents into a FAISS index using the provided embeddings.
    
    Args:
        embeddings (HuggingFaceEmbeddings): The embeddings model.
        documents (list): A list of documents to be indexed.
        
    Returns:
        faiss.Index: The FAISS index containing the document embeddings.
    """
    if os.path.exists("faiss_index.index"):
        return faiss.read_index("faiss_index.index")
    else: 
        # Get embeddings for the documents
        doc_embeddings = [embeddings.embed_query(doc.page_content) for doc in documents]
    
        # Create a FAISS index
        dimension = len(doc_embeddings[0])
        index = faiss.IndexFlatL2(dimension)
    
        # Add document embeddings to the index
        index.add(np.array(doc_embeddings).astype('float32'))
    
        return index
    
def top_k_texts(chunks, query, k=5):
    """
    Retrieve the top-k most similar texts from the FAISS index based on a query.
    
    Args:
        query (str): The query text to search for.
        k (int): The number of top results to return.
        
    Returns:
        list: A list of the top-k most similar texts.
    """
    embeddings = get_save_embeddings()
    index = load_to_faiss_index(embeddings, chunks)
    
    query_embedding = embeddings.embed_query(query)
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), k)
    
    return indices[0]

def system_prompt():
    """
    Print the system prompt for the application.
    """

    return (
        "You are an expert assistant that reads research papers and explains them clearly.\n"
        "You are given the top relevant context from a paper and a user query.\n"
        "Use your understanding of the context to answer the query in your own words.\n"
        "Do not copy sentences verbatim unless quoting is necessary.\n"
        "You may paraphrase, summarize, or combine insights to form a clear answer.\n"
        "If the context is unclear or insufficient, say 'The document does not contain enough information.'"
        "Avoid saying things like 'Based on the context' or 'According to the paper'.\n"
        "Just answer directly and confidently, as if you're knowledgeable about the topic."
    )

def build_rag(embeddings, query, k, chunks, faiss_index):
    """
    Build the RAG (Retrieval-Augmented Generation) system.
    
    This function initializes the system by loading the PDF, splitting documents,
    getting embeddings, and loading them into a FAISS index.
    """

    # Embed the query and get top-k indices
    query_vector = embeddings.embed_query(query)
    D, I = faiss_index.search(np.array([query_vector]).astype('float32'), k)
    top_chunks = [chunks[i].page_content for i in I[0]]

    # Build the final prompt
    system = system_prompt()
    context = "\n\n".join(top_chunks)
    prompt = f"""
    Answer the question as if you're explaining the idea to a student. 
    Don't mention 'context' or 'document' — just answer directly in your own words.
    {system}
    Context:
    {context}
    Question:
    {query}

Answer:"""

    model = os.getenv("MODEL", "llama-3.3-70b-versatile")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                    {"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()

def answer(embeddings, query, k, chunks, faiss_index):
    """
    Answer a query using the RAG system.
    
    Args:
        embeddings (HuggingFaceEmbeddings): The embeddings model.
        query (str): The query text to search for.
        k (int): The number of top results to return.
        chunks (list): A list of document chunks.
        faiss_index (faiss.Index): The FAISS index containing the document embeddings.
        
    Returns:
        str: The response from the RAG system.
    """
    top_indices = top_k_texts(chunks, query, k)
    top_chunks = [chunks[i].page_content for i in top_indices]
    
    system = system_prompt()
    context = "\n\n".join(top_chunks)
    
    prompt = f"""
You are writing a short academic-style explanation based on the following paper segment and your own knowledge.

Context from the paper:
{context}

User question:
{query}

Write a comprehensive answer using both the paper and your own general understanding. 
Cite the paper if applicable, and expand with relevant external knowledge.
"""
    
    model = os.getenv("MODEL", "llama-3.3-70b-versatile")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                    {"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()

app = FastAPI()

@app.post("/paper-rag")
async def rag_endpoint(query: QueryRequest):
    """
    Endpoint to handle RAG queries.
    
    Args:
        query (str): The query text to search for.
        
    Returns:
        str: The response from the RAG system.
    """
    pdf_path = "./IPD Research Paper.pdf"
    documents = load_pdf(pdf_path)

    chunks = chunking(documents)

    embeddings = get_save_embeddings()

    faiss_index = load_to_faiss_index(embeddings, chunks)
    # Save the FAISS index to a file
    faiss.write_index(faiss_index, "faiss_index.index")
    embeddings = get_save_embeddings()
    faiss_index = load_to_faiss_index(embeddings, chunks)
    
    response = build_rag(embeddings, query.query, k=5, chunks=chunks, faiss_index=faiss_index)
    
    return {"response": response}

@app.post("/get-answers")
async def get_answers(query: QueryRequest):
    """
    Endpoint to get answers from the RAG system.
    
    Args:
        query (QueryRequest): The query request containing the query text.
        
    Returns:
        dict: A dictionary containing the response from the RAG system.
    """
    pdf_path = "./IPD Research Paper.pdf"
    documents = load_pdf(pdf_path)

    chunks = chunking(documents)

    embeddings = get_save_embeddings()

    faiss_index = load_to_faiss_index(embeddings, chunks)
    
    response = answer(embeddings, query.query, k=5, chunks=chunks, faiss_index=faiss_index)
    
    return {"response": response}   

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost")
