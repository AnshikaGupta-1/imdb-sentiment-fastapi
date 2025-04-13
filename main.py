from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import os
import requests
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load sentiment analysis model
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

# TMDb API setup
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.head("/")
async def health_check():
    return

@app.post("/search", response_class=HTMLResponse)
async def search_movie(request: Request, movie: str = Form(...)):
    response = requests.get(f"{TMDB_BASE_URL}/search/movie", params={
        "api_key": TMDB_API_KEY,
        "query": movie
    })

    results = response.json().get("results", [])[:3]
    movies = [{
        "Title": r.get("title"),
        "imdbID": r.get("id"),
        "Year": r.get("release_date", "N/A")[:4]
    } for r in results]

    return templates.TemplateResponse("index.html", {"request": request, "movies": movies})

@app.get("/movie/{movie_id}", response_class=HTMLResponse)
async def movie_detail(request: Request, movie_id: int, page: int = 1):
    try:
        movie_response = requests.get(f"{TMDB_BASE_URL}/movie/{movie_id}", params={
            "api_key": TMDB_API_KEY
        })
        movie = movie_response.json()
        title = movie.get("title", "Unknown Title")

        review_response = requests.get(f"{TMDB_BASE_URL}/movie/{movie_id}/reviews", params={
            "api_key": TMDB_API_KEY,
            "page": page
        })
        review_data = review_response.json()
        reviews_raw = review_data.get("results", [])
        total_pages = review_data.get("total_pages", 1)

        # Shorten long reviews to first 500 characters (for model input)
        texts_to_analyze = [r.get("content", "")[:500] for r in reviews_raw if r.get("content")]
        results = sentiment_model(texts_to_analyze, batch_size=8)

        analyzed_reviews = []
        for i, result in enumerate(results):
            sentiment = result['label']
            label = "positive" if sentiment.startswith("4") or sentiment.startswith("5") else "negative"
            analyzed_reviews.append({
                "text": reviews_raw[i].get("content", ""),
                "sentiment": sentiment,
                "label": label
            })

    except Exception as e:
        print(f"TMDb API error: {e}")
        title = movie_id
        analyzed_reviews = [{"text": "Error fetching reviews.", "sentiment": "N/A", "label": "neutral"}]
        total_pages = 1
        page = 1

    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": title,
        "reviews": analyzed_reviews,
        "movie_id": movie_id,
        "current_page": page,
        "total_pages": total_pages
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=port)
