from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackManager

def get_llm():
    callback_manager = CallbackManager([StreamingStdOutCallbackManager()])
    
    return LlamaCpp(
        model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        callback_manager=callback_manager,
        n_gpu_layers=-1,
        n_ctx=4096,
        temperature=0.7
    )