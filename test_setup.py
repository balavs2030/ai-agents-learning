from dotenv import load_dotenv
import os
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=100,
    messages=[{"role": "user", "content": "Say hello in one sentence."}]
)

print(message.content[0].text)