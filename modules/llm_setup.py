"""LLM and embedding model setup (OpenAI / Azure OpenAI)."""

import os

from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAIEmbeddings,
)
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper


def setup_llm_and_embeddings(model: str = "gpt-4o-mini", provider: str = "auto"):
    """
    Set up the LLM and embedding models for RAGAS.

    Returns:
        Tuple of (generator_llm, generator_embeddings)
    """
    if provider == "auto":
        if os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"):
            provider = "azure"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            raise ValueError(
                "No API credentials found. Please set either:\n"
                "  - OPENAI_API_KEY for OpenAI, or\n"
                "  - AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT for Azure OpenAI"
            )

    if provider == "azure":
        return _setup_azure_openai(model)
    return _setup_openai(model)


def _setup_openai(model: str):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it with: export OPENAI_API_KEY='your-key'"
        )

    print(f"  LLM provider: OpenAI (model: {model})")
    llm = ChatOpenAI(model=model, temperature=0.3)
    generator_llm = LangchainLLMWrapper(llm)

    embeddings = OpenAIEmbeddings()
    generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    return generator_llm, generator_embeddings


def _setup_azure_openai(model: str):
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    chat_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", model)
    embedding_deployment = os.environ.get(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002"
    )

    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set.")
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is not set.")

    print(f"  LLM provider: Azure OpenAI")
    print(f"  Endpoint: {endpoint}")
    print(f"  Chat deployment: {chat_deployment}")
    print(f"  Embedding deployment: {embedding_deployment}")

    llm = AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment=chat_deployment,
        temperature=1,
    )
    generator_llm = LangchainLLMWrapper(llm)

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment=embedding_deployment,
    )
    generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    return generator_llm, generator_embeddings
