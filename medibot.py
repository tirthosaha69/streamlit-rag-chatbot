import streamlit as st
import os
import glob
from connect_memory_with_llm import get_qa_chain
from create_memory_for_llm import process_pdfs_in_folder

# Set paths
DATA_FOLDER = "data"
VECTORSTORE_FOLDER = "vectorstore"

def clear_data_N_vectorstore_folder():
    # Clear PDF files in the data folder
    if os.path.exists(DATA_FOLDER):
        for file1 in glob.glob(os.path.join(DATA_FOLDER, "*.pdf")):
            os.remove(file1)
    if os.path.exists(VECTORSTORE_FOLDER):
        for file2 in glob.glob(os.path.join(VECTORSTORE_FOLDER, "*.faiss")):
            os.remove(file2)
    if os.path.exists(VECTORSTORE_FOLDER):
        for file3 in glob.glob(os.path.join(VECTORSTORE_FOLDER, "*.pkl")):
            os.remove(file3)

# Ensure directories exist
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)

# Run the clear function only once per session (i.e., when the app first loads)
if "initialized" not in st.session_state:
    clear_data_N_vectorstore_folder()  # Delete files only on page refresh
    st.session_state.initialized = True  # Mark as initialized

# Set page configuration
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

# Sidebar for API Key and PDF Upload
with st.sidebar:
    st.title("🔧 Settings")
    st.write("Customize your chatbot experience here.")

    # API Key Input
    api_key = st.text_input("🔑 Enter API Key:", type="password")

    # "Add PDF File" Button
    uploaded_file = st.file_uploader("📂 Upload a PDF", type=["pdf"])
    
    processed = False
    file_path = None  # Initialize file path

    if uploaded_file:
        file_path = os.path.join(DATA_FOLDER, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"✅ {uploaded_file.name} has been uploaded!")

    # "Process File to RAG" Button
    if uploaded_file and st.button("🚀 Process File to RAG"):
        if file_path:
            try:
                process_pdfs_in_folder(DATA_FOLDER)  # Generate FAISS index
                st.success("🔄 PDF processed and embeddings created successfully!")
                processed = True
            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")

    # Theme Selection (Default: Dark Mode)
    mode = st.radio("🌗 Select Theme:", ["🌙 Dark Mode", "🌞 Light Mode"], index=0)
    st.divider()
    st.write("ℹ️ **Tip:** The more detailed your question, the better the answer!")

# Dynamic Theme CSS
theme_styles = {
    "🌙 Dark Mode": {"bg_color": "#121212", "text_color": "#ffffff", "user_color": "#4caf50", "bot_color": "#303f9f"},
    "🌞 Light Mode": {"bg_color": "#f5f5f5", "text_color": "#000000", "user_color": "#1565c0", "bot_color": "#673ab7"},
}
theme = theme_styles[mode]

st.markdown(
    f"""
    <style>
        body {{ background-color: {theme["bg_color"]}; color: {theme["text_color"]}; }}
        .stApp {{ background-color: {theme["bg_color"]}; padding: 20px; border-radius: 10px; }}
        .message {{ padding: 12px; border-radius: 8px; margin-bottom: 10px; font-size: 16px; color: {theme["text_color"]}; }}
        .user {{ background-color: {theme["user_color"]}; color: white; }}
        .assistant {{ background-color: {theme["bot_color"]}; color: white; }}
        .header {{ font-size: 26px; font-weight: bold; text-align: center; padding: 10px; color: {theme["bot_color"]}; }}
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    st.markdown(f"<div class='header'>🤖 RAG AI Chatbot ({mode})</div>", unsafe_allow_html=True)
    st.write("👋 Welcome to your AI-powered chatbot! Upload PDFs, ask questions, and get responses with sources.")

    if not api_key:
        st.warning("⚠️ Please enter your API key in the sidebar to use the chatbot.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        role, content = msg["role"], msg["content"]
        css_class = "user" if role == "user" else "assistant"
        st.markdown(f"<div class='message {css_class}'>{content}</div>", unsafe_allow_html=True)

    prompt = st.chat_input("Type your message here...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        # Get response along with sources
        try:
            response_data = get_qa_chain(api_key).invoke({'query': prompt, 'api_key': api_key})
            response_text = response_data['result']
            source_documents = response_data.get('source_documents', [])

            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.chat_message("assistant").markdown(response_text)

            # ✅ Display source documents
            if source_documents:
                st.markdown("#### 📌 Sources:")
                for doc in source_documents:
                    metadata = doc.metadata
                    page_number = metadata.get("page", "Unknown") 
                    snippet = doc.page_content[:50] 
                    st.markdown(f"📄 **Page {page_number}:** {snippet}...")  # Show source details
        except Exception as e:
            st.error(f"❌ Error in chatbot response: {str(e)}")

if __name__ == "__main__":
    main()
