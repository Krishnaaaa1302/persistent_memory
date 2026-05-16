from fastapi import FastAPI, HTTPException
from models import ChatRequest, ChatResponse
from memory import store_memory, retrieve_memories
from llm import chat
import time


app = FastAPI(title='Memory Chatbot API',version='1.0')

session_histories : dict[str,list[dict]] = {}

@app.post('/chat',response_model = ChatResponse)
async def chat_endpoint(req : ChatRequest):
    t0 = time.time()

    memories = retrieve_memories(req.user_id, req.message, top_k = 5)

    history = session_histories.get(req.session_id or req.user_id, [])[-6:]

    try:
        reply = chat(req.message, memories, history)
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))
    
    store_memory(req.user_id, f"User said: {req.message}")
    store_memory(req.user_id, f"Assistant Replied: {reply}", role='assistant')

    sid = req.session_id or req.user_id
    session_histories.setdefault(sid, [])
    session_histories[sid].extend([
        {'role':"user",'content':req.message},
        {'role':'assistant','content':reply}
    ])

    latency = (time.time() - t0) * 1000
    safe_reply = reply or ""
    return ChatResponse(reply = safe_reply , memories_used = len(memories), latency_ms = round(latency))

@app.get('/memories/{user_id}')
async def list_memories(user_id: str,limit: int = 20):
    from memory import get_collection
    col = get_collection(user_id)
    results = col.get(limit = limit,include = ['documents','metadatas'])
    return {'count':col.count(), 'memories': results['documents']}

@app.delete('/memories/{user_id}')
async def clear_memories(user_id: str):
    from memory import client
    if user_id:
        client.delete_collection(f"user_{user_id}")
        return {'status':'cleared'}
    else:
        return {'status':'No user found'}
    
@app.get('/health')
async def health():
    return {'status':'ok'}