from utils import openai_client

prompt="""Suggest three names for a new pet salon business.
          The generated name ideas should evoke positive emotions and the
          following key features: Professional, friendly, Personalized Service."""

response = openai_client.chat.completions.create(
    model="gpt-4.1",
    temperature=0.7,
    max_tokens=100,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{prompt}"}
    ]
)
content = response.choices[0].message.content
print(content)
