import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

VECTORSTORE_PATH = "vectorstore"

def process_pdfs_in_folder(folder_path):
    """Processes all PDFs in a folder, creates embeddings, and saves the FAISS index."""
    
    if not os.path.exists(folder_path):
        raise ValueError("❌ The specified folder does not exist.")
    
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    
    if not pdf_files:
        raise ValueError("❌ No PDF files found in the folder.")
    
    all_texts = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"📂 Processing: {pdf_path}")

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if not documents:
            print(f"⚠️ Skipping empty or unreadable PDF: {pdf_file}")
            continue

        all_texts.extend(text_splitter.split_documents(documents))

    if not all_texts:
        raise ValueError("❌ No valid text extracted from PDFs.")

    db = FAISS.from_documents(all_texts, embedding_model)
    db.save_local(VECTORSTORE_PATH)

    print("✅ FAISS index created successfully!")
