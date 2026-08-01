import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import health, routes
from app.core import obs
from app.rag import retrieval

logging.basicConfig(level=logging.INFO)
obs.setup()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loading the ONNX embedder and embedding the 5 reference chunks costs ~2.4s. Lazily,
    # that lands inside the first candidate's answer; here it lands before the port opens.
    try:
        retrieval.warmup()
        obs.log_event("index.ready", "-")
    except Exception as e:  # a cold-start failure must not stop the server booting —
        obs.log_event("index.warmup_failed", "-", exc=type(e).__name__)  # retrieve() retries lazily
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(health.router)
app.include_router(routes.router)
