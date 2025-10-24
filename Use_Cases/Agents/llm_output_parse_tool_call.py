from pydantic import BaseModel
from openai import OpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("MGA_API_KEY")
client = OpenAI(base_url="https://chat.int.bayer.com/api/v2", api_key=token)
model = 'o4-mini'

class UserDetails(BaseModel):
    name: str
    age: int



pydantic_schema = UserDetails.model_json_schema()


response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "Extract: Jason is 25 years old."}
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "extract_user_info",
                "description": "Extracts name and age from text.",
                "parameters": pydantic_schema
            }
        }
    ],
    #tool_choice="none"
    tool_choice={
        "type": "function",
        "function": {"name": "extract_user_info"}
    }
)

print(response.choices[0])
tool_call = response.choices[0].message.tool_calls[0]
json_string_arguments = tool_call.function.arguments
print(json_string_arguments)

print(f"LLM returned this string: {json_string_arguments}")

user_details = UserDetails.model_validate_json(json_string_arguments)

assert isinstance(user_details, UserDetails)