import aiohttp
import os
import asyncio

async def async_together_chat(messages, model="deepseek-ai/DeepSeek-V3"):
    """
    Send messages to Together API asynchronously and return response.
    
    Args:
        messages (list of dict): Chat history in {"role":..., "content":...} format
        model (str): Model name for API
    """
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            result = await resp.json()
            return result["choices"][0]["message"]["content"]

def run_async_chat(messages, model="deepseek-ai/DeepSeek-V3"):
    """Helper to run async chat synchronously from Streamlit."""
    return asyncio.run(async_together_chat(messages, model=model))
