from src.utils.config_loader import CONFIG
from src.utils.logger import trace_task, logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

@trace_task
def generate_query_variants(query: str) -> list[str]:
    """LLM-based query rewriting and expansion to improve recall."""
    if not CONFIG["query_expansion"]["enabled"]:
        return [query]
    
    llm_cfg = CONFIG["llm"]
    provider = llm_cfg.get("provider", "groq")

    # "auto" lets you run zero-paid-demo if GROQ_API_KEY isn't set.
    if provider == "auto":
        provider = "groq" if os.getenv("GROQ_API_KEY") else "ollama"

    model_name = llm_cfg["model"]
    temperature = CONFIG["query_expansion"]["temperature"]

    if provider == "groq":
        from langchain_groq import ChatGroq

        model = ChatGroq(
            model=model_name,
            max_tokens=1024,
            temperature=temperature,
        )
    elif provider == "ollama":
        # Ollama OpenAI-compatible endpoint.
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model=model_name, temperature=temperature)
    else:
        logger.warning(f"Unknown llm provider '{provider}'. Disabling query expansion.")
        return [query]
    
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