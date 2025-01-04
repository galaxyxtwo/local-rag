from langchain.chains import RetrievalQA
from mistral import get_llm
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def get_qa_chain():
    # Load the stored vectorstore
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda'}
    )
    
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    # Get the LLM
    llm = get_llm()
    
    # Create the chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    
    return qa_chain

if __name__ == "__main__":
    chain = get_qa_chain()
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        response = chain(question)
        print("\nAnswer:", response['result'])