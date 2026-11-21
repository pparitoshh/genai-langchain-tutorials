"""Read the project settings from config.yaml in the project root."""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config() -> dict:
    """Return the content of config.yaml as a dictionary."""
    with open(CONFIG_PATH, encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def llm_model() -> str:
    """Return the name of the Groq model."""
    return load_config()["llm"]["model"]


def get_llm(**kwargs):
    """Return a Groq chat model. Pass options such as temperature or max_tokens."""
    from dotenv import load_dotenv
    from langchain_groq import ChatGroq

    load_dotenv()
    return ChatGroq(model=llm_model(), **kwargs)


def get_embeddings():
    """Return a local embedding model. Groq has no embedding API."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=load_config()["embeddings"]["model"])
