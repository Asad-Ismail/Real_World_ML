import os

import instructor
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
    client = OpenAI(base_url="https://chat.int.bayer.com/api/v2", api_key=token)
    return instructor.patch(client)


def extract_user_details(text: str) -> UserDetails:
    client = build_client()
    return client.chat.completions.create(
        model=MODEL_NAME,
        response_model=UserDetails,
        messages=[{"role": "user", "content": f"Extract the structured user information from: {text}"}],
    )


def main() -> None:
    user_details = extract_user_details("Jason is 25 years old.")
    print(user_details.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
