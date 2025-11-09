import tiktoken as tk
import click

# define the input string
prompt = "I have a white dog named Champ."

@click.command()
@click.option("--sentence", type=str, default=prompt, help="The sentence to tokenize")
@click.option("--encoding-name", type=click.Choice([
    "cl100k_base",
    "p50k_base",
    "r50k_base"])
)
def count_tokens(sentence: str, encoding_name: str):
    # get the encoding
    encoding = tk.get_encoding(encoding_name)

    # encode the string
    encoded_string = encoding.encode(sentence)

    # count the number of tokens
    num_tokens = len(encoded_string)
    print(f"Number of tokens: {num_tokens}")

if __name__ == "__main__":
    count_tokens()
