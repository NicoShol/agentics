from enum import Enum


class ConstantsLLM(Enum):
    LIGHT = "gemma3:1b"
    GPTOSS = "gpt-oss:20b"


class ConstantsEmbedding(Enum):
    NOMIC = "nomic-embed-text"
