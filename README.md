# EduVision – AI-Powered Educational Content Assistant

## Problem Statement
Students struggle to understand complex textbook content due to language barriers
and lack of visual explanations.

## Solution
EduVision uses Azure OpenAI to summarize, translate, and visually explain
educational content from uploaded PDFs.

## Features
- PDF upload and parsing
- AI-based summarization
- Translation to regional languages
- Visual explanation support
- Azure-hosted backend

## Tech Stack
- Azure OpenAI
- Azure Blob Storage
- Azure Functions / FastAPI
- Python
- HTML/CSS

## Architecture
<img width="1186" height="956" alt="image" src="https://github.com/user-attachments/assets/26bbd154-27af-4473-a663-40aa9f35cb2b" />

## How to Run
running
---------
.\.venv\Scripts\activate
uvicorn app.main:app --reload

## Future Improvements
- Voice support
- Quiz generation
- Multi-agent reasoning
