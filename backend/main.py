from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.jarvis import Jarvis


app = FastAPI(
    title="JARVIS",
    description="Human-like Multimodal AI Companion",
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "null",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


jarvis = Jarvis(
    session_id="default"
)


# =========================================================
# Request models
# =========================================================

class ConversationRequest(
    BaseModel
):
    message: str


class ContextRequest(
    BaseModel
):
    key: str
    value: Any


# =========================================================
# Basic endpoints
# =========================================================

@app.get("/")
async def root():

    return {
        "name": "JARVIS",
        "status": "online",
        "version": "0.1.0",
    }


@app.get("/health")
async def health():

    return {
        "success": True,
        "status": "healthy",
    }


@app.get("/status")
async def status():

    return jarvis.status()


# =========================================================
# Conversation
# =========================================================

@app.post("/conversation")
async def conversation(
    request: ConversationRequest,
):

    return jarvis.receive(
        request.message
    )


@app.post("/conversation/reset")
async def reset_conversation():

    jarvis.reset_conversation()

    return {
        "success": True,
        "message": "Conversation reset.",
    }


# =========================================================
# Context
# =========================================================

@app.post("/context/update")
async def update_context(
    request: ContextRequest,
):

    return jarvis.update_context(
        key=request.key,
        value=request.value,
    )


# =========================================================
# Proactive reasoning
# =========================================================

@app.get("/proactive/check")
async def proactive_check():

    return (
        jarvis.evaluate_proactive()
    )


@app.post("/proactive/cycle")
async def proactive_cycle():

    return (
        jarvis.proactive_cycle()
    )


@app.get("/proactive/status")
async def proactive_status():

    return (
        jarvis.proactive.status()
    )