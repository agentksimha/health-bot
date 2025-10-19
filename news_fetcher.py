import aiohttp
import asyncio
import feedparser
import datetime

RSS_URL = "https://news.google.com/rss/search?q=health+disease+awareness&hl=en-IN&gl=IN&ceid=IN:en"

async def fetch_rss_url(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()

def fetch_news():
    """Fetch latest 5 news articles."""
    raw_xml = asyncio.run(fetch_rss_url(RSS_URL))
    feed = feedparser.parse(raw_xml)
    articles = []
    for entry in feed.entries[:5]:
        articles.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.published
        })
    return articles

def update_news_hourly(session_state):
    """Update news in Streamlit session state every hour."""
    now = datetime.datetime.now()
    if "last_news_update" not in session_state or (now - session_state.last_news_update).seconds > 3600:
        session_state.last_news_update = now
        session_state.news_articles = fetch_news()
