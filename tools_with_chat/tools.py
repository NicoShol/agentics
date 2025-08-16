from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """
    Add two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.
    """
    return a + b


@tool
def matching(s: str) -> str:
    """
    Match a string against a pattern.

    Args:
        s (str): The input string.

    Returns:
        str: The matched string or an error message.
    """
    if s == "hello":
        return "submarine"
    elif s == "world":
        return "clerical"
    elif s == "foo":
        return "coucou"
    elif len(s) > 10:
        return "longer than 10 characters"
    else:
        return "no match found"


TOOLS: dict[str, callable] = {
    "add": add,
    "matching": matching,
}
