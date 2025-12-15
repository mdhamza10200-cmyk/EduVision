from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import uuid
import os

from .config import BASE_UPLOAD_DIR, IMAGE_OUTPUT_DIR, ORGAN_IMAGE_DIR
from .pdf_utils import save_upload, extract_text, extract_images
from .ai_utils import (
    summarize_text,
    translate_summary,
    generate_detailed_text,
    generate_references,
    identify_organ,
    get_static_organ_image,
    identify_organ_with_static_image,
)

app = FastAPI(title="Medical PDF Assistant")

# Allow all origins for development; restrict in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve original extracted images
app.mount("/files", StaticFiles(directory=BASE_UPLOAD_DIR), name="files")

# Serve generated diagram images (legacy, if ever used)
app.mount("/diagrams", StaticFiles(directory=IMAGE_OUTPUT_DIR), name="diagrams")

# Serve static detailed organ images
app.mount("/organs", StaticFiles(directory=ORGAN_IMAGE_DIR), name="organs")

# Simple in-memory store for demo only.
SESSION_DATA: dict[str, dict] = {}


def to_original_url(path: str) -> str:
    rel = os.path.relpath(path, BASE_UPLOAD_DIR).replace("\\", "/")
    return f"/files/{rel}"


def to_diagram_url(path: str | None) -> str | None:
    if not path:
        return None
    rel = os.path.relpath(path, IMAGE_OUTPUT_DIR).replace("\\", "/")
    return f"/diagrams/{rel}"


def to_organ_url(path: str | None) -> str | None:
    if not path:
        return None
    rel = os.path.relpath(path, ORGAN_IMAGE_DIR).replace("\\", "/")
    return f"/organs/{rel}"


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, extract text and images, and return a summary."""
    contents = await file.read()
    session_id = str(uuid.uuid4())

    pdf_path = save_upload(file.filename, contents)
    text = extract_text(pdf_path)
    summary = summarize_text(text)  # uses Azure OpenAI

    image_paths = extract_images(pdf_path, session_id)

    # ✅ Keep only images > 1 KB (1024 bytes) and delete tiny ones
    large_image_paths: list[str] = []
    for path in image_paths:
        try:
            size = os.path.getsize(path)
            if size > 1024:
                large_image_paths.append(path)
            else:
                # delete small image file, don't keep it
                os.remove(path)
        except OSError:
            # if file is missing or unreadable, just skip it
            continue

    SESSION_DATA[session_id] = {
        "pdf_path": pdf_path,
        "text": text,
        "summary": summary,
        "images": large_image_paths,   # ✅ only big images stored
        "translations": {},
        "details": None,
        "references": None,
        "labeled": [],
    }

    return {
        "session_id": session_id,
        "summary": summary,
        "image_count": len(large_image_paths),  # ✅ only >1KB counted
    }


@app.get("/summary/{session_id}")
async def get_summary(session_id: str):
    data = SESSION_DATA.get(session_id)
    if not data:
        return JSONResponse(status_code=404, content={"error": "Invalid session_id"})
    return {"summary": data["summary"]}


@app.get("/translate/{session_id}")
async def get_translation(session_id: str, language: str):
    data = SESSION_DATA.get(session_id)
    if not data:
        return JSONResponse(status_code=404, content={"error": "Invalid session_id"})

    if language not in data["translations"]:
        data["translations"][language] = translate_summary(data["summary"], language)
    return {"language": language, "summary": data["translations"][language]}


@app.get("/details/{session_id}")
async def get_details(session_id: str):
    data = SESSION_DATA.get(session_id)
    if not data:
        return JSONResponse(status_code=404, content={"error": "Invalid session_id"})

    if not data["details"]:
        data["details"] = generate_detailed_text(data["summary"], data["text"])
    return {"details": data["details"]}


@app.get("/details/translate/{session_id}")
async def translate_details(session_id: str, language: str):
    data = SESSION_DATA.get(session_id)
    if not data:
        return JSONResponse(status_code=404, content={"error": "Invalid session_id"})

    # Ensure details exist
    if not data["details"]:
        data["details"] = generate_detailed_text(data["summary"], data["text"])

    # Reuse existing translate_summary logic
    translated_details = translate_summary(data["details"], language)

    return {
        "language": language,
        "details": translated_details
    }


@app.get("/references/{session_id}")
async def get_refs(session_id: str):
    data = SESSION_DATA.get(session_id)
    if not data:
        return JSONResponse(status_code=404, content={"error": "Invalid session_id"})

    if not data["references"]:
        data["references"] = generate_references(data["summary"])
    return {"references": data["references"]}


@app.get("/images/{session_id}")
async def get_images(session_id: str):
    data = SESSION_DATA.get(session_id)
    if not data:
        return JSONResponse(status_code=404, content={"error": "Invalid session_id"})
    return {"images": data["images"], "labeled": data["labeled"]}


@app.post("/images/label/{session_id}")
async def label_images(session_id: str):
    data = SESSION_DATA.get(session_id)
    if not data:
        return JSONResponse(status_code=404, content={"error": "Invalid session_id"})

    labeled_outputs = []

    for img_path in data["images"]:
        # 1) Identify organ from the extracted image
        organ_info = identify_organ(img_path)
        organ = organ_info.get("organ", "unknown")
        labels = organ_info.get("labels", [])

        # 2) Use static organ PNG from /static/organs
        static_organ_path = get_static_organ_image(organ)

        labeled_outputs.append(
            {
                "original": to_original_url(img_path),
                "organ": organ,
                "labels": labels,
                "labeled_image": static_organ_path,
                "labeled_image_url": to_organ_url(static_organ_path),
                "image_generation_status": (
                    "ok" if static_organ_path else "not_found"
                ),
            }
        )

    data["labeled"] = labeled_outputs
    SESSION_DATA[session_id] = data

    return {"results": labeled_outputs}

@app.post("/identify-organ-image")
async def identify_organ_image(file: UploadFile = File(...)):
    """
    Accept a single image upload, identify the organ using the vision model,
    and return the corresponding detailed anatomical organ image
    from /static/organs (if available).
    """
    # 1) Save uploaded image under BASE_UPLOAD_DIR so it can be served via /files
    contents = await file.read()
    ext = os.path.splitext(file.filename)[1] or ".png"

    single_image_dir = os.path.join(BASE_UPLOAD_DIR, "single_images")
    os.makedirs(single_image_dir, exist_ok=True)

    image_path = os.path.join(single_image_dir, f"{uuid.uuid4()}{ext}")
    with open(image_path, "wb") as f:
        f.write(contents)

    # 2) Use helper to identify organ + get static anatomical image
    organ_info = identify_organ_with_static_image(image_path)
    organ = organ_info.get("organ", "unknown")
    labels = organ_info.get("labels", [])
    static_image_path = organ_info.get("static_image_path")

    # 3) Build URLs for frontend
    return {
        "organ": organ,
        "labels": labels,
        "original_image": to_original_url(image_path),
        "detailed_image": static_image_path,
        "detailed_image_url": to_organ_url(static_image_path),
        "image_generation_status": "ok" if static_image_path else "not_found",
    }

