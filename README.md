# 🤖 RAG AI Chatbot with LangChain, FAISS, and Hugging Face

## 📌 Overview
This project implements a **Retrieval-Augmented Generation (RAG) AI chatbot** using:
- **LangChain** for integrating the LLM
- **FAISS** for vector search
- **Hugging Face models** for embeddings & inference
- **Streamlit** for an interactive UI

The chatbot allows users to **upload PDFs**, processes them into **embeddings**, and retrieves answers based on the document content.

## 🚀 Features
- **PDF Upload & Processing:** Converts PDFs into vector embeddings
- **Retrieval-Augmented Generation:** Finds the most relevant content before generating a response
- **FAISS-based Memory:** Efficient vector search for fast retrieval
- **Hugging Face LLM:** Uses `mistralai/Mistral-7B-Instruct-v0.3`
- **Streamlit UI:** Clean interface for interactions
- **Light & Dark Themes:** Customizable appearance

---

## 🔧 Installation
### **1️⃣ Clone the repository**
```sh
git clone https://github.com/your-repo/rag-chatbot.git
cd rag-chatbot
```

### **2️⃣ Create a virtual environment & activate it**
```sh
# Windows\python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### **3️⃣ Install dependencies**
```sh
pip install -r requirements.txt
```

---

## 🔑 API Key Setup
This project requires a **Hugging Face API Key** to access the `mistralai/Mistral-7B-Instruct-v0.3` model.

1. Get your API key from [Hugging Face](https://huggingface.co/settings/tokens).
2. Enter it in the Streamlit sidebar (`🔧 Settings`).

---

## 📂 File Processing Workflow
### **1️⃣ Upload PDFs**
- Users upload PDF files through the Streamlit sidebar.

### **2️⃣ Create FAISS Vector Store**
- `create_memory.py` extracts text, splits it, and stores embeddings in FAISS.

### **3️⃣ Retrieve & Generate Responses**
- `connect_memory.py` fetches relevant document chunks.
- The LLM (`Mistral-7B-Instruct`) generates responses.

---

## 🎯 Usage
### **1️⃣ Run the Chatbot**
```sh
streamlit run main.py
```

### **2️⃣ Upload a PDF**
- Drag & drop a PDF file in the sidebar.

### **3️⃣ Ask Questions**
- Type your question in the chat input.
- The bot will retrieve information from the uploaded PDFs.

---

## 📌 Folder Structure
```
📂 rag-chatbot/
│-- 📜 main.py              # Streamlit chatbot UI
│-- 📜 create_memory.py     # PDF processing & FAISS index creation
│-- 📜 connect_memory.py    # LLM & retrieval logic
│-- 📂 data/                # Stores uploaded PDFs
│-- 📂 vectorstore/         # Stores FAISS embeddings
│-- 📜 requirements.txt     # Dependencies
│-- 📜 README.md            # Documentation
```

---

## 📌 Troubleshooting
### **1️⃣ API Key Error**
🔹 Ensure you have entered a valid Hugging Face API key.

### **2️⃣ FAISS Not Found**
🔹 Upload & process a PDF before asking questions.

---

## 💡 Future Enhancements
- Multi-PDF support
- Streaming responses
- User authentication

---

## 💖 Contributions
Feel free to **fork, contribute, and improve** this project!

