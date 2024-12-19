import requests

class NewsFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"

    def fetch_news(self, query='crypto', from_date='2024-12-10', sort_by='popularity'):
        """Fetch news articles using News API"""
        params = {
            'q': query,
            'from': from_date,
            'sortBy': sort_by,
            'apiKey': self.api_key
        }

        try:
            response = requests.get(self.base_url, params=params)
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
    # Example usage
    api_key = 'your_news_api_key_here'
    news_fetcher = NewsFetcher(api_key)
    news_fetcher.fetch_news(query='crypto')
