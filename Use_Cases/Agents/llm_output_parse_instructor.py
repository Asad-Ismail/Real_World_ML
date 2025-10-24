from pydantic import BaseModel
from openai import OpenAI
import json
import os
from dotenv import load_dotenv
import instructor

load_dotenv()

token = os.getenv("MGA_API_KEY")
client = OpenAI(base_url="https://chat.int.bayer.com/api/v2", api_key=token)
model = 'o4-mini'

# 1. Define your model
class UserDetails(BaseModel):
    name: str
    age: int


client = instructor.patch(client)

# 3. Call the client and specify your 'response_model'
user_details = client.chat.completions.create(
    model=model,
    #response_model=UserDetails,  
    messages=[{"role": "user", "content": "Extract: Jason is 25 years old."}]
)

print(user_details)

assert isinstance(user_details, UserDetails)

print(user_details)

