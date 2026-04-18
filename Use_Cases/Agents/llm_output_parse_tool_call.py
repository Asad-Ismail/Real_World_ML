import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


MODEL_NAME = "o4-mini"


class UserDetails(BaseModel):
    name: str
    age: int


def build_client() -> OpenAI:
    load_dotenv()
    token = os.getenv("MGA_API_KEY")
    if not token:
        raise EnvironmentError("Set MGA_API_KEY before running this example.")
    return OpenAI(base_url="https://chat.int.bayer.com/api/v2", api_key=token)


def extract_with_tool_call(text: str) -> UserDetails:
    client = build_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": f"Extract user information from: {text}"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "extract_user_info",
                    "description": "Extracts name and age from text.",
                    "parameters": UserDetails.model_json_schema(),
                },
            }
        ],
        tool_choice={
            "type": "function",
            "function": {"name": "extract_user_info"},
        },
    )
    tool_call = response.choices[0].message.tool_calls[0]
    return UserDetails.model_validate_json(tool_call.function.arguments)


def main() -> None:
    user_details = extract_with_tool_call("Jason is 25 years old.")
    print(user_details.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
