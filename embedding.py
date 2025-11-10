import os
from utils import openai_client

def get_embedding(text):
    response = openai_client.embeddings.create(
         model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

if __name__ == "__main__":
    embeddings = get_embedding("I have a white dog named Champ.")
    print("Embedding Length:", len(embeddings))
    print("Embedding:", embeddings[:5])
