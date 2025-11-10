from utils import openai_client

response = openai_client.completions.create(
    model="text-davinci-003",
    prompt="I have a white dog named Champ.",
    temperature=1,
    max_tokens=256,
    top_p=0.3,
    frequency_penalty=0,
    presence_penalty=0
)
