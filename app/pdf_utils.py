
import os
import fitz  # pymupdf
from pypdf import PdfReader
from .config import BASE_UPLOAD_DIR

def save_upload(filename: str, contents: bytes) -> str:
    """Save uploaded PDF bytes to disk and return the path."""
    os.makedirs(BASE_UPLOAD_DIR, exist_ok=True)
    path = os.path.join(BASE_UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(contents)
    return path

def extract_text(pdf_path: str) -> str:
    """Extract text from all pages of the PDF."""
    reader = PdfReader(pdf_path)
    text_parts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        text_parts.append(t)
    return "\n".join(text_parts)

def extract_images(pdf_path: str, session_id: str) -> list[str]:
    """Extract images from the PDF and save them in a session-specific folder."""
    doc = fitz.open(pdf_path)
    session_dir = os.path.join(BASE_UPLOAD_DIR, session_id, "images")
    os.makedirs(session_dir, exist_ok=True)

    image_paths: list[str] = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n > 4:  # CMYK or other
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_path = os.path.join(
                session_dir,
                f"page{page_index+1}_img{img_index+1}.png"
            )
            pix.save(img_path)
            image_paths.append(img_path)
            pix = None
    doc.close()
    return image_paths
