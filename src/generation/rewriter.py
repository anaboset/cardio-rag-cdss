from src.utils.config_loader import CONFIG
from src.utils.logger import trace_task, logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

@trace_task
def generate_query_variants(query: str) -> list[str]:
    """LLM-based query rewriting and expansion to improve recall."""
    if not CONFIG["query_expansion"]["enabled"]:
        return [query]
    
    model = ChatGroq(
        model=CONFIG["llm"]["model"],
        max_tokens=4096,
        temperature=CONFIG["query_expansion"]["temperature"]
    )
    
    num_variants = CONFIG["query_expansion"].get("num_variants", 3)
    system_prompt_text = CONFIG["query_expansion"].get("system_prompt", "Generate {num} variations.")
    
    prompt = ChatPromptTemplate.from_template(
        system_prompt_text.replace("{num}", str(num_variants)) + 
        "\n\nOriginal Query: {query}\n\nProvide variants separated by new lines."
    )
    
    chain = prompt | model | StrOutputParser()
    
    try:
        raw_output = chain.invoke({"query": query})
        variants = [line.strip("- ").strip() for line in raw_output.split("\n") if line.strip()]
        return [query] + variants[:num_variants]
    except Exception as e:
        logger.error(f"❌ Query expansion failed: {str(e)}", exc_info=True)
        return [query]