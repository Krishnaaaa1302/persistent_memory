import os
from huggingface_hub import InferenceClient

client = InferenceClient(model = 'mistralai/Mistral-7B-Instruct-v0.3',token=os.getenv('HF_TOKEN'))

def build_system_prompt(memories:list[str])->str:
    if not memories:
        return "You are a helpful assistant with persistent memory."
    memory_block = "\n".join(f"- {m}" for m in memories)
    return f"""You are a helpful assistant with persistent memory.
    Relevant things you remember about this user:
    {memory_block}
    Use this context naturally. Do not announce that you're using memory."""

def chat(message:str, memories:list[str],history:list[dict]):
    messages = [{'role':'system','content':build_system_prompt(memories)}] + history + [{'role':'user','content':message}]

    response = client.chat.completions.create(
        model='mistral',
        messages=messages,
        temperature=0.7,
        max_tokens=512
    )

    return response.choices[0].message.content