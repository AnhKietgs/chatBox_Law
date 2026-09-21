from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .database import Base, engine
from .api import admin, chat


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="Law RAG", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8080", "http://localhost:5173"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
app.include_router(chat.router)
app.include_router(admin.router)


@app.get("/healthz")
def healthcheck():
    return {"status": "ok"}
