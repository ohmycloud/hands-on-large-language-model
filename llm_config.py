from utils import openai_client

response = openai_client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt="I have a white dog named Champ.",
    temperature=1,
    max_tokens=16,
    top_p=0.3,
    frequency_penalty=0,
    presence_penalty=0
)

print(response.choices[0].text.strip())
