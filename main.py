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

# TMDb API Setup
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
async def movie_detail(request: Request, movie_id: str):
    try:
        # Fetch movie details
        movie_url = f"{TMDB_BASE_URL}/movie/{movie_id}"
        review_url = f"{TMDB_BASE_URL}/movie/{movie_id}/reviews"
        params = {"api_key": TMDB_API_KEY, "language": "en-US"}

        movie_res = requests.get(movie_url, params=params).json()
        reviews_res = requests.get(review_url, params=params).json()

        title = movie_res.get("title", movie_id)
        reviews = reviews_res.get("results", [])

        if not reviews:
            review = "No reviews found."
            sentiment = "N/A"
            label = "neutral"
        else:
            review = reviews[0]["content"]
            sentiment_result = sentiment_model(review)[0]
            sentiment = sentiment_result["label"]
            label = "positive" if sentiment.startswith("4") or sentiment.startswith("5") else "negative"

    except Exception as e:
        print(f"TMDb API error: {e}")
        title = movie_id
        review = "Error fetching reviews."
        sentiment = "N/A"
        label = "neutral"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": title,
        "review": review,
        "result": sentiment,
        "label": label
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
