from __future__ import annotations
import io, json, os, re, secrets, time, uuid
from collections import Counter
from pathlib import Path
from typing import Any

import fitz
import pytesseract
from PIL import Image
from docx import Document
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from xml.sax.saxutils import escape

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024
TTL_SECONDS = int(os.getenv("SESSION_TTL_MINUTES", "120")) * 60
ENABLE_OCR = os.getenv("ENABLE_OCR", "1").lower() not in {"0", "false", "no"}
SESSIONS: dict[str, dict[str, Any]] = {}
app = FastAPI(title="LexBrief AI"
