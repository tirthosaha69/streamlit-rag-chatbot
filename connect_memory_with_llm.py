from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
import os

DB_FAISS_PATH = "vectorstore"
huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(api_key):
    return HuggingFaceEndpoint(model=huggingface_repo_id, huggingfacehub_api_token=api_key)

custom_prompt_template = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk, please.
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

def get_qa_chain(api_key):
    if not os.path.exists(DB_FAISS_PATH):
        raise ValueError(f"❌ FAISS database not found at {DB_FAISS_PATH}. Upload a PDF first.")

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)  


    return RetrievalQA.from_chain_type(
        llm=load_llm(api_key=api_key),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(custom_prompt_template)}
    )
