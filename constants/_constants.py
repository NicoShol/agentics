from enum import Enum


class ConstantsLLM(str, Enum):
    LIGHT = "gemma3:1b"  # no tools
    ADVANCED = "gpt-oss:20b"  # with tools


class ConstantsEmbedding(str, Enum):
    NOMIC = "nomic-embed-text"
