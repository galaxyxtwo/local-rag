import os
from typing import Any, List, Optional
from langchain_core.runnables import Runnable
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Changed import
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings  # Changed import
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import LLM

class LilypadLLM(LLM, Runnable):
    """Custom LLM class that implements the Runnable interface"""
    
    module_version: str = "github.com/noryev/module-llama2:6d4fd8c07b5f64907bd22624603c2dd54165c215"
    target_address: str = "0xA7f9BD3837279C3776B17b952D97C619f3892BDE"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Execute the LLM call."""
        escaped_prompt = prompt.replace('"', '\\"')
        
        command = (
            f'lilypad run {self.module_version} '
            f'-i prompt="{escaped_prompt}" '
            f'--target {self.target_address} '
            f'--web3-private-key WEB3_PRIVATE_KEY'
        )
        
        return os.popen(command).read().strip()

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "lilypad"

def main():
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
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return

    # Initialize the custom LilypadLLM
    lilypad_llm = LilypadLLM()

    print("\nCreating QA chain...")
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=lilypad_llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template="Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\nContext: {context}\n\nQuestion: {question}\n\nHelpful Answer:",
                    input_variables=["context", "question"]
                )
            }
        )
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
            result = qa_chain.invoke({"query": question})
            if isinstance(result, dict):
                print("\nAnswer:", result.get('result', 'No answer found'))
            else:
                print("\nAnswer:", result)
        except Exception as e:
            print(f"Error processing question: {e}")

if __name__ == "__main__":
    main()