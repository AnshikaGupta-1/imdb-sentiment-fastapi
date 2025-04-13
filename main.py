from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from imdb import IMDb
import uvicorn
import os

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

# IMDbPY instance
ia = IMDb()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.head("/")
async def health_check():
    return

@app.post("/search", response_class=HTMLResponse)
async def search_movie(request: Request, movie: str = Form(...)):
    search_results = ia.search_movie(movie)
    movies = []

    for result in search_results[:5]:
        movies.append({
            "Title": result.get('title'),
            "imdbID": result.movieID,
            "Year": result.get('year', 'N/A')
        })

    return templates.TemplateResponse("index.html", {"request": request, "movies": movies})

@app.get("/movie/{imdb_id}", response_class=HTMLResponse)
async def movie_detail(request: Request, imdb_id: str):
    try:
        movie = ia.get_movie(imdb_id)
        title = movie.get('title', imdb_id)

        ia.update(movie, 'reviews')
        reviews = movie.get('reviews', [])

        if not reviews:
            review = "No reviews found."
            sentiment = "N/A"
            label = "neutral"
        else:
            review = reviews[0]['content']
            sentiment = sentiment_model(review)[0]['label']
            label = "positive" if sentiment.startswith("4") or sentiment.startswith("5") else "negative"

    except Exception as e:
        print(f"IMDbPY error: {e}")
        title = imdb_id
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
