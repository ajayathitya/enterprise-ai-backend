from tiktoken import get_encoding

def calculate_tokens(string: str) -> int:
    """
    Calculates the number of tokens in the provided string using the 'cl100k_base' encoding.

    Parameters:
        string (str): The input string for which tokens need to be calculated.

    Returns:
        int: The number of tokens in the input string after encoding.

    Example:
        >>> calculate_tokens("This is a sample string.")
        5
    """
    encoding = get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(string))
    return num_tokens