import sys
import subprocess
import pkg_resources
import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA

def main():
    # Verify document exists and is readable
    doc_path = "./docs/lilypadIssues.md"
    if not os.path.exists(doc_path):
        print(f"ERROR: Document not found at {doc_path}")
        return

    print("\nInitializing embedding model...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'},
            cache_folder="./model_cache"
        )
    except Exception as e:
        print(f"Error initializing embeddings: {e}")
        return

    print("\nLoading and splitting document...")
    try:
        loader = TextLoader(doc_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        print(f"Document split into {len(splits)} chunks")
    except Exception as e:
        print(f"Error processing document: {e}")
        return

    print("\nCreating vector store...")
    try:
        # Store the vectorstore in a variable we'll use later
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        print("Vector store created successfully!")
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return

    print("\nInitializing LLM...")
    try:
        model_path = "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        if not os.path.exists(model_path):
            print(f"ERROR: Model not found at {model_path}")
            return
        
        llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            temperature=0.7
        )
        print("LLM initialized successfully!")
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return

    print("\nCreating QA chain...")
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )
        print("QA chain created successfully!")
    except Exception as e:
        print(f"Error creating QA chain: {e}")
        return

    print("\nReady! You can now ask questions about the document.")
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        print("\nThinking...")
        try:
            response = qa_chain(question)
            print("\nAnswer:", response['result'])
        except Exception as e:
            print(f"Error processing question: {e}")

if __name__ == "__main__":
    main()