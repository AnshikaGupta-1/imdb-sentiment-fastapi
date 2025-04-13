from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from transformers import pipeline
import requests
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import time
from bs4 import BeautifulSoup

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

# Jinja2 templates for rendering HTML
templates = Jinja2Templates(directory="templates")

# Scrape IMDb user reviews
def get_imdb_reviews(imdb_id, max_reviews=1):
    url = f"https://www.imdb.com/title/{imdb_id}/reviews"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive"
    }

    try:
        time.sleep(1)
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        review_divs = soup.find_all("div", class_="text show-more__control")
        reviews = [div.get_text(strip=True) for div in review_divs[:max_reviews]]
        return reviews if reviews else ["No user reviews found."]
    except Exception as e:
        print(f"Error scraping IMDb: {e}")
        return ["Could not fetch reviews."]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.head("/")
async def health_check():
    return

@app.post("/search", response_class=HTMLResponse)
async def search_movie(request: Request, movie: str = Form(...)):
    search_url = f"https://www.imdb.com/find?q={movie}&s=tt"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    results = soup.select("td.result_text a")
    movies = [
        {
            "Title": item.text,
            "imdbID": item["href"].split("/")[2],
            "Year": item.parent.text.strip().replace(item.text, "").strip(" ()")
        }
        for item in results[:5]
    ]
    return templates.TemplateResponse("index.html", {"request": request, "movies": movies})

@app.get("/movie/{imdb_id}", response_class=HTMLResponse)
async def movie_detail(request: Request, imdb_id: str):
    title_url = f"https://www.imdb.com/title/{imdb_id}/"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(title_url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.text.replace(" - IMDb", "") if title_tag else imdb_id

    reviews = get_imdb_reviews(imdb_id)
    review = reviews[0]
    sentiment = sentiment_model(review)[0]['label']
    label = "positive" if sentiment.startswith("4") or sentiment.startswith("5") else "negative"
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
