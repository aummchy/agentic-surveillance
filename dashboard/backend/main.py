import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dashboard.backend.routes import faces, events, live, reports, chat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CAPTURES_DIR = Path(__file__).resolve().parent.parent.parent / "captures"
CAPTURES_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Surveillance Dashboard API",
    description="API for managing and monitoring surveillance system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(faces.router, prefix="/api/faces", tags=["faces"])
app.include_router(events.router, prefix="/api/events", tags=["events"])
app.include_router(reports.router, prefix="/api", tags=["reports"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(live.router, prefix="/ws", tags=["websocket"])


app.mount("/captures", StaticFiles(directory=str(CAPTURES_DIR)), name="captures")


@app.get("/")
async def root():
    return {"message": "Surveillance Dashboard API", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
