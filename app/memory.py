import chromadb
from app.embedder import embed
from datetime import datetime,UTC
import uuid

client = chromadb.PersistentClient(path = './chroma_store')

def get_collection(user_id : str):
    return client.get_or_create_collection(
        name = f'user_{user_id}',
        metadata={'hnsw:space':'cosine'}
    )

def store_memory(user_id : str,turn : str,role : str='user'):
    col = get_collection(user_id)
    embedding = embed(turn)
    col.add(
        documents = [turn],
        embeddings = [embedding],
        metadatas = [{'role' : role,'timestamp': datetime.now(UTC).isoformat()}],
        ids = [str(uuid.uuid4())]
    )

def retrieve_memories(user_id : str,query : str,top_k : int = 5) -> list[str]:
    col = get_collection(user_id)
    if col.count() == 0:
        return []
    results = col.query(
        query_embeddings = [embed(query)],
        n_results = min(top_k, col.count()),
        include = ['documents','metadatas']
    )
    if results and results['documents']:
        return results['documents'][0]
    
    return []