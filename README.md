# IMDb Movie Sentiment Analyzer 🎬

A FastAPI web app that lets you search for IMDb movies using the OMDb API and run sentiment analysis on sample reviews using a pre-trained Hugging Face model.

## Features
- Movie search via OMDb API
- Sentiment analysis using BERT model
- Web UI built with Jinja2 templates
- Ready for Render (Free Tier) deployment

## Deployment
This app is configured for [Render](https://render.com) via `render.yaml`.

To run locally:
```bash
pip install -r requirements.txt
uvicorn main:app --reload
