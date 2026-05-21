---
name: data-collection
description: Data collection and web scraping agent. Use this skill when the user asks to scrape websites, extract data from web pages, build data ingestion pipelines, or interact with scraping scripts.
---

# Data Collection Agent

## Overview

This skill equips the agent with best practices and instructions for web scraping and data collection within the user's project. The primary tools used are Python, `requests`, and `BeautifulSoup4`.

## Workflow Decision Tree

1. **Analyze Target**: Check the target website structure and identify the elements containing the desired data.
2. **Fetch HTML**: Use the `requests` library to fetch the page content. Always include appropriate headers (e.g., `User-Agent`) to mimic a browser and avoid being blocked.
3. **Parse and Extract**: Use `BeautifulSoup` to parse the HTML and extract the necessary fields using CSS selectors or find methods.
4. **Data Persistence**: Store the extracted data in the SQLite database using SQLAlchemy models defined in `models.py` and the `SessionLocal` from `database.py`.

## Quick Start

### Basic BeautifulSoup Scraping Template

When asked to scrape a new source, follow this pattern:

```python
import requests
from bs4 import BeautifulSoup
from database import SessionLocal
import models

HEADERS = {
    'User-Agent': 'MLBBPredictiveAnalysis/1.0 (Contact info)',
    'Accept-Encoding': 'gzip'
}

def fetch_and_scrape(url):
    db = SessionLocal()
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Extract logic here...
            # Example: items = soup.select('.item-class')
            
            # Save to DB
            # new_record = models.MyModel(field=value)
            # db.add(new_record)
            # db.commit()
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        db.rollback()
    finally:
        db.close()
```

## Existing Context

- Existing scraping implementations are primarily located in `scraper.py`. Add new scraping functions there or create specialized modules as needed.
- The project's database schemas are defined in `models.py`. Ensure you understand the target schema before inserting data.
- Always use `SessionLocal()` from `database.py` for database operations and ensure the session is properly closed.
