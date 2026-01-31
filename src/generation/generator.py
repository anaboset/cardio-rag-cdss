from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_groq import ChatGroq
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
    llm = ChatGroq(
        model=CONFIG["llm"]["model"],
        temperature=CONFIG["llm"]["temperature"],
    )

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