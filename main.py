from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import requests
import uvicorn
import os

# Load sentiment analysis model
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# TMDb API Setup (v3 API key method)
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.head("/")
async def health_check():
    return

@app.post("/search", response_class=HTMLResponse)
async def search_movie(request: Request, movie: str = Form(...)):
    url = f"{TMDB_BASE_URL}/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": movie,
        "language": "en-US",
    }
    response = requests.get(url, params=params)
    data = response.json()

    movies = []
    for result in data.get("results", [])[:3]:  # Limit to top 3 matches
        movies.append({
            "Title": result.get("title"),
            "imdbID": result.get("id"),  # TMDb ID
            "Year": result.get("release_date", "N/A")[:4],
        })

    return templates.TemplateResponse("index.html", {"request": request, "movies": movies})

@app.get("/movie/{movie_id}", response_class=HTMLResponse)
async def movie_detail(request: Request, movie_id: str, page: int = 1):
    try:
        # Fetch movie details
        movie_url = f"{TMDB_BASE_URL}/movie/{movie_id}"
        params = {"api_key": TMDB_API_KEY, "language": "en-US"}
        movie_res = requests.get(movie_url, params=params).json()
        title = movie_res.get("title", movie_id)

        # Fetch reviews with pagination
        review_url = f"{TMDB_BASE_URL}/movie/{movie_id}/reviews"
        params_reviews = {"api_key": TMDB_API_KEY, "language": "en-US", "page": page}
        reviews_res = requests.get(review_url, params=params_reviews).json()

        # Determine pagination limit: if there are more than 100 reviews, limit to 100 total.
        total_results = reviews_res.get("total_results", 0)
        total_pages = reviews_res.get("total_pages", 1)
        # Assume roughly 20 reviews per page; hence, limit to 5 pages (i.e., 100 reviews)
        safe_total_pages = total_pages
        if total_results > 100:
            safe_total_pages = min(total_pages, 5)

        reviews_raw = reviews_res.get("results", [])

        analyzed_reviews = []
        for r in reviews_raw:
            text = r.get("content", "")
            if text:
                sentiment_result = sentiment_model(text)[0]
                # Determine sentiment class (adjust thresholds as needed)
                label = "positive" if sentiment_result["label"].startswith("4") or sentiment_result["label"].startswith("5") else "negative"
                analyzed_reviews.append({
                    "text": text,
                    "sentiment": sentiment_result["label"],
                    "label": label
                })

        if not analyzed_reviews:
            analyzed_reviews.append({
                "text": "No reviews found.",
                "sentiment": "N/A",
                "label": "neutral"
            })

    except Exception as e:
        print(f"TMDb API error: {e}")
        title = movie_id
        analyzed_reviews = [{
            "text": "Error fetching reviews.",
            "sentiment": "N/A",
            "label": "neutral"
        }]
        safe_total_pages = 1
        page = 1

    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": title,
        "reviews": analyzed_reviews,
        "current_page": page,
        "total_pages": safe_total_pages,
        "movie_id": movie_id
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
