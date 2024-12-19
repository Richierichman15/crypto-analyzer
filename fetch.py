import requests

def fetch_news():
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': 'crypto',
        'from': '2024-12-10',
        'sortBy': 'popularity',
        'apiKey': '274e832ccba44abfa54914e1ea6915f3'
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad responses
        news_data = response.json()
        
        # Print the number of articles found
        print(f"Found {len(news_data.get('articles', []))} articles.")
        
        # Print the titles of the first few articles
        for article in news_data.get('articles', [])[:5]:
            print(f"Title: {article.get('title')}")
            print(f"Published At: {article.get('publishedAt')}")
            print(f"Source: {article.get('source', {}).get('name')}")
            print("-" * 30)
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")

if __name__ == "__main__":
    fetch_news()
