from fastapi import FastAPI
from models import ChatRequest, ChatResponse
from memory import store_memory, retrieve_memories
from llm import chat
import time

app = FastAPI(title='Memory Chatbot API',version='1.0')