import os
from dotenv import load_dotenv
import anthropic

def main() -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "1024"))
    client = anthropic.Anthropic(api_key=api_key)

    try:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": "用一句话解释什么是机器学习"}
            ]
        )
        text = message.content[0].text if message.content else ""
        if not text:
            raise RuntimeError("Empty response from Anthropic API")
        print(text)
    except Exception as e:
        raise RuntimeError(f"Anthropic request failed: {e}") from e


if __name__ == "__main__":
    main()

