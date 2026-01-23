import asyncio
from ollama import AsyncClient
import json

# Configuration
BASE_URL = "http://localhost:11434"
MODEL = "qwen3:32b"
client = AsyncClient(host=BASE_URL)


async def call_llm(prompt: str) -> str:
    """An individual async worker to call the LLM."""
    response = await client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Return ONLY JSON matching the schema."},
            {"role": "user", "content": prompt},
        ],
        format="json",  # Using built-in JSON mode for 2026 reliability
        stream=False,
        options={"temperature": 0.4, "top_p": 0.9, "top_k": 40}
    )
    # Parse the specific key from your schema
    content = json.loads(response['message']['content'])
    return content.get("text", "No text found")


async def process_multiple_prompts(prompts):
    # 1. Create a list of tasks
    tasks = [call_llm(p) for p in prompts]

    print(f"Starting {len(prompts)} concurrent calls...")

    # 2. Use as_completed to process them as they finish
    for task in asyncio.as_completed(tasks):
        result = await task
        print(f"--- Received Response ---\n{result}\n")


# Run the script
prompts_list = [
    "Explain quantum physics in one sentence.",
    "What is the capital of France?",
    "Tell me a short joke about AI."
]

asyncio.run(process_multiple_prompts(prompts_list))
