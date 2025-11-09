import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    raise ValueError("No API key found. Please check ypur .env file.")


openai_client = OpenAI(
    api_key=api_key,
    base_url='https://vip.apiyi.com/v1'
)

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
