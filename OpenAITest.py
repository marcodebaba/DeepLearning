import os
from dotenv import load_dotenv
from openai import OpenAI

def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    model = os.getenv("OPENAI_MODEL", "gpt-5.2")
    client = OpenAI(api_key=api_key)

    try:
        resp = client.responses.create(
            model=model,
            input="请用一句话介绍 Python"
        )
        print(resp.output_text)
    except Exception as e:
        raise RuntimeError(f"OpenAI request failed: {e}") from e


if __name__ == "__main__":
    main()
