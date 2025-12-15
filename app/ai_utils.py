import base64
import json
import os
from typing import List, Dict

from openai import AzureOpenAI, APIConnectionError
from .config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_DEPLOYMENT,
    AZURE_OPENAI_VISION_DEPLOYMENT,
    ORGAN_IMAGE_DIR,
)

# -------------------------------------------------------------------
# Azure OpenAI client
# -------------------------------------------------------------------
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# -------------------------------------------------------------------
# 1) Summarize PDF text
# -------------------------------------------------------------------
def summarize_text(text: str) -> str:
    # Truncate very long text just to be safe with context limits
    if len(text) > 12000:
        text = text[:12000]

    prompt = (
        "You are a medical assistant for doctors. "
        "Summarize the following medical/anatomy content as bullet points. "
        "Focus on: key anatomical structures, function, and clinical notes.\n\n"
        f"{text}"
    )

    try:
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_CHAT_DEPLOYMENT,  # deployment name in Azure
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except APIConnectionError as e:
        print("Azure OpenAI connection error in summarize_text:", e)
        return "Error: unable to contact Azure OpenAI for summary. Please check endpoint / network."
    except Exception as e:
        print("Unexpected error in summarize_text:", e)
        return "Error: summarization failed."


# -------------------------------------------------------------------
# 2) Translate summary to another language
# -------------------------------------------------------------------
def translate_text(text: str, language: str, context: str = "medical text") -> str:
    prompt = (
        f"Translate the following {context} into {language}. "
        "Keep medical terminology accurate and use a professional tone.\n\n"
        f"{text}"
    )
    try:
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except APIConnectionError as e:
        print("Azure OpenAI connection error in translate_text:", e)
        return f"Error: unable to contact Azure OpenAI for translation to {language}."
    except Exception as e:
        print("Unexpected error in translate_text:", e)
        return "Error: translation failed."

def translate_summary(summary: str, language: str) -> str:
    return translate_text(summary, language, context="medical summary")




# -------------------------------------------------------------------
# 3) Extra detailed explanation
# -------------------------------------------------------------------
def generate_detailed_text(summary: str, full_text: str) -> str:
    if len(full_text) > 12000:
        full_text = full_text[:12000]

    prompt = (
        "You are an expert medical educator. Using the PDF content and its summary, "
        "write a detailed explanation for doctors (around 1000 words). "
        "Include sections: Anatomy, Physiology, Common Pathologies, Diagnostics, "
        "and Clinical Notes.\n\n"
        f"SUMMARY:\n{summary}\n\nFULL TEXT:\n{full_text}"
    )
    try:
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except APIConnectionError as e:
        print("Azure OpenAI connection error in generate_detailed_text:", e)
        return "Error: unable to contact Azure OpenAI for detailed explanation."
    except Exception as e:
        print("Unexpected error in generate_detailed_text:", e)
        return "Error: details generation failed."

def generate_detailed_text_translated(
    summary: str,
    full_text: str,
    language: str
) -> str:
    # Step 1: generate English details
    details = generate_detailed_text(summary, full_text)

    # If generation failed, don’t translate garbage
    if details.startswith("Error:"):
        return details

    # Step 2: translate details
    return translate_text(details, language, context="detailed medical explanation")


# -------------------------------------------------------------------
# 4) Suggest reference links
# -------------------------------------------------------------------
def generate_references(summary: str) -> List[str]:
    prompt = (
        "Based on the following summary of a human organ/body part, "
        "list 5 to 8 reputable reference links (guidelines, textbooks, "
        "or review articles). Prefer major medical sites and journals. "
        "Return them as a simple numbered list with plain URLs.\n\n"
        f"{summary}"
    )
    try:
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content.strip()
        return [line for line in text.splitlines() if line.strip()]
    except APIConnectionError as e:
        print("Azure OpenAI connection error in generate_references:", e)
        return ["Error: unable to contact Azure OpenAI for references."]
    except Exception as e:
        print("Unexpected error in generate_references:", e)
        return ["Error: reference generation failed."]


# -------------------------------------------------------------------
# 5) Identify organ from image using vision (chat with image input)
# -------------------------------------------------------------------
def identify_organ(image_path: str) -> Dict:
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print("Error reading image in identify_organ:", e)
        return {"organ": "unknown", "labels": []}

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "This is a medical image from a PDF. "
                        "Identify the main human organ or body part. "
                        "Also list key anatomical structures visible (max 10). "
                        "Respond only in JSON with keys 'organ' and 'labels'."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
            ],
        }
    ]

    try:
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_VISION_DEPLOYMENT,  # e.g. gpt-4o
            messages=messages,
        )
        content = resp.choices[0].message.content

        # content can be string or list, normalize to string
        if isinstance(content, list):
            content_text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        else:
            content_text = content

        data = json.loads(content_text)
        if "organ" not in data:
            data["organ"] = "unknown"
        if "labels" not in data:
            data["labels"] = []
        return data
    except APIConnectionError as e:
        print("Azure OpenAI connection error in identify_organ:", e)
        return {"organ": "unknown", "labels": []}
    except Exception as e:
        print("Unexpected error in identify_organ:", e)
        return {"organ": "unknown", "labels": []}


# -------------------------------------------------------------------
# 6) Use static organ images from /static/organs
# -------------------------------------------------------------------
def get_static_organ_image(organ: str) -> str | None:
    """
    Given an organ name like 'heart' or 'left ventricle',
    return the file path to the corresponding static image
    in ORGAN_IMAGE_DIR, or None if not found.
    """
    if not organ:
        return None

    name = organ.lower().strip()

    # basic synonym mapping
    mapping = {
        "heart": "heart.jpg",
        "left ventricle": "heart.jpg",
        "right ventricle": "heart.jpg",
        "right atrium": "heart.jpg",
        "left atrium": "heart.jpg",

        "lung": "lungs.jpg",
        "lungs": "lungs.jpg",

        "brain": "brain.jpg",

        "liver": "liver.jpg",

        "kidney": "kidney.jpg",
        "kidneys": "kidney.jpg",
            "stomach": "stomach.jpg",

    "small intestine": "intestine.jpg",
    "large intestine": "intestine.jpg",
    "intestines": "intestine.jpg",

    "pancreas": "pancreas.jpg",

    "spleen": "spleen.jpg",

    "esophagus": "esophagus.jpg",

    "trachea": "trachea.jpg",

    "gallbladder": "gallbladder.jpg",

    "urinary bladder": "urinary_bladder.jpg",
    "bladder": "urinary_bladder.jpg",

    "thyroid": "thyroid.jpg",
    "thyroid gland": "thyroid.jpg",

    "adrenal gland": "adrenal.jpg",
    "adrenal glands": "adrenal.jpg",

    "skin": "skin.jpg",

    "eye": "eye.jpg",
    "eyes": "eye.jpg",

    "ear": "ear.jpg",
    "ears": "ear.jpg",

    "ovary": "ovary.jpg",
    "ovaries": "ovary.jpg",

    "testis": "testis.jpg",
    "testes": "testis.jpg",
        # add more as needed...
    }

    filename = None

    # 1) exact match
    if name in mapping:
        filename = mapping[name]
    else:
        # 2) substring match - e.g. "human heart" -> heart.jpg
        for key, value in mapping.items():
            if key in name:
                filename = value
                break

    if not filename:
        return None

    path = os.path.join(ORGAN_IMAGE_DIR, filename)
    return path if os.path.exists(path) else None

# -------------------------------------------------------------------
# 7) Convenience: identify organ AND get static detailed image
# -------------------------------------------------------------------
def identify_organ_with_static_image(image_path: str) -> Dict:
    """
    Given an image file path, use the vision model to identify the organ
    and then map it to a static detailed anatomical image in ORGAN_IMAGE_DIR.

    Returns a dict:
    {
        "organ": "<organ name or 'unknown'>",
        "labels": [ ... ],              # structures from vision model
        "static_image_path": "<path or None>",
    }
    """
    organ_info = identify_organ(image_path)
    organ = organ_info.get("organ", "unknown")
    labels = organ_info.get("labels", [])

    static_image_path = get_static_organ_image(organ)

    return {
        "organ": organ,
        "labels": labels,
        "static_image_path": static_image_path,
    }


# -------------------------------------------------------------------
# Helper: safely extract JSON object from model output
# -------------------------------------------------------------------
def _extract_json_object(text: str) -> str:
    """
    Try to pull out the first {...} JSON object from a string.
    This lets us handle responses like ```json { ... } ``` or
    explanations around the JSON.
    """
    if not text:
        return text

    text = text.strip()

    # Remove leading/trailing markdown code fences
    if text.startswith("```"):
        # take the part after the first fence
        parts = text.split("```", 2)
        if len(parts) >= 2:
            text = parts[1].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    # Find first '{' and last '}'
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text


# -------------------------------------------------------------------
# 8) Identify organ from image using vision (chat with image input)
# -------------------------------------------------------------------
def identify_organ(image_path: str) -> Dict:
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print("Error reading image in identify_organ:", e)
        return {"organ": "unknown", "labels": []}

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "This is a medical image from a PDF. "
                        "Identify the main human organ or body part. "
                        "Also list key anatomical structures visible (max 10). "
                        "Respond ONLY as pure JSON (no markdown, no code block) "
                        "exactly in this format:\n"
                        "{\"organ\": \"heart\", \"labels\": [\"left ventricle\", \"right ventricle\"]}"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
            ],
        }
    ]

    try:
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_VISION_DEPLOYMENT,  # e.g. gpt-4o
            messages=messages,
        )

        content = resp.choices[0].message.content

        # content can be string or list, normalize to string
        if isinstance(content, list):
            content_text = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and "text" in part
            )
        else:
            content_text = content or ""

        # Optional: debug the raw model response
        # print("RAW VISION RESPONSE:", repr(content_text))

        # Clean up markdown / extra text and extract JSON object
        content_text = _extract_json_object(content_text)

        try:
            data = json.loads(content_text)
        except Exception as e:
            print("JSON parse error in identify_organ, content was:", repr(content_text))
            print("Error:", e)
            return {"organ": "unknown", "labels": []}

        if "organ" not in data:
            data["organ"] = "unknown"
        if "labels" not in data:
            data["labels"] = []

        return data

    except APIConnectionError as e:
        print("Azure OpenAI connection error in identify_organ:", e)
        return {"organ": "unknown", "labels": []}
    except Exception as e:
        print("Unexpected error in identify_organ:", e)
        return {"organ": "unknown", "labels": []}