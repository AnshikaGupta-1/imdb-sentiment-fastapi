from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from transformers import pipeline
import requests
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
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

# Load model
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# OMDb API Key
API_KEY = os.getenv("OMDB_API_KEY")

# Jinja2 templates for rendering HTML
templates = Jinja2Templates(directory="templates")

# Dummy reviews
sample_reviews = {
    "tt1375666": "Inception is a masterpiece with brilliant direction and acting.",
    "tt0133093": "The Matrix changed the sci-fi genre forever. Amazing film.",
    "tt3896198": "Guardians of the Galaxy Vol. 2 has a great soundtrack but weaker story."
}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
    
@app.head("/")
async def health_check():
    return

@app.post("/search", response_class=HTMLResponse)
async def search_movie(request: Request, movie: str = Form(...)):
    response = requests.get(f"http://www.omdbapi.com/?s={movie}&apikey={API_KEY}")
    data = response.json()
    movies = data.get("Search", [])
    return templates.TemplateResponse("index.html", {"request": request, "movies": movies})

@app.get("/movie/{imdb_id}", response_class=HTMLResponse)
async def movie_detail(request: Request, imdb_id: str):
    details = requests.get(f"http://www.omdbapi.com/?i={imdb_id}&apikey={API_KEY}").json()
    title = details.get("Title")
    review = sample_reviews.get(imdb_id, "Great movie with decent performances and plot.")
    sentiment = sentiment_model(review)[0]['label']
    label = "positive" if sentiment.startswith("4") or sentiment.startswith("5") else "negative"
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": title,
        "review": review,
        "result": sentiment,
        "label": label
    })
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # fallback if PORT isn't set
    uvicorn.run("main:app", host="0.0.0.0", port=port)
