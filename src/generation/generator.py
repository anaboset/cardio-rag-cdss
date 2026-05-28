from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import os

from src.utils.config_loader import CONFIG, load_prompt
from src.utils.logger import trace_task

@trace_task
def format_docs(docs):
    """
    Formats retrieved documents for the prompt context.
    """
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" 
        for doc in docs
    )

def get_rag_chain():
    """
    Builds the unified LCEL RAG chain (The logic structure).
    """
    llm_cfg = CONFIG["llm"]
    provider = llm_cfg.get("provider", "groq")

    # "auto" lets you run zero-paid-demo if GROQ_API_KEY isn't set.
    if provider == "auto":
        provider = "groq" if os.getenv("GROQ_API_KEY") else "ollama"

    temperature = llm_cfg.get("temperature", 0.1)
    model = llm_cfg["model"]

    if provider == "groq":
        from langchain_groq import ChatGroq

        llm = ChatGroq(model=model, temperature=temperature)
    elif provider == "ollama":
        # Use Ollama's OpenAI-compatible API.
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model, temperature=temperature)
    else:
        raise ValueError(f"Unsupported llm provider: {provider}")

    system_prompt = load_prompt("system_prompt") 
    user_template = load_prompt("recommendation_prompt") 

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_template) 
    ])

    # The chain expects 'context', 'query', and 'patient_summary'
    return (
        RunnableParallel({
            "context": itemgetter("context"),
            "query": itemgetter("query"),
            "patient_summary": itemgetter("patient_summary")
        })
        | prompt
        | llm
        | StrOutputParser()
    )