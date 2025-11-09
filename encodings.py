import tiktoken as tk

prompt = "I have a white dog named Champ."

def get_tokens(sentence: str, encoding_name: str) -> list[int]:
    # get the encoding
    encoding = tk.get_encoding(encoding_name)

    # encode the sentence
    return encoding.encode(sentence)

def get_string(tokens: list[int], encoding_name: str) -> str:
    # get the encoding
    encoding = tk.get_encoding(encoding_name)

    # decode the tokens
    return encoding.decode(tokens)

if __name__ == "__main__":
    print("cl100k_base Tokens:", get_tokens(prompt, "cl100k_base"))
    print("  p50k_base Tokens:", get_tokens(prompt, "p50k_base"))
    print("  r50k_base Tokens:", get_tokens(prompt, "r50k_base"))

    print("Original String:", get_string([40, 617, 264, 4251, 5679, 7086, 56690, 13], "cl100k_base"))
