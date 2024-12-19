from selenium.webdriver import Remote, ChromeOptions
from selenium.webdriver.chromium.remote_connection import ChromiumRemoteConnection
from bs4 import BeautifulSoup
import json
import os
import time
from datetime import datetime, timedelta
import hashlib
import requests

class WebScraper:
    def __init__(self, cache_dir='cache', cache_duration=3600):  # 1 hour cache by default
        self.AUTH = 'brd-customer-hl_93996b2b-zone-ai_scraper:pesj2tw37cgo'
        self.SBR_WEBDRIVER = f'https://{self.AUTH}@brd.superproxy.io:9515'
        self.cache_dir = cache_dir
        self.cache_duration = cache_duration
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _get_cache_key(self, url):
        """Generate a unique cache key for a URL"""
        return hashlib.md5(url.encode()).hexdigest()

    def _get_cached_data(self, url):
        """Try to get data from cache"""
        cache_key = self._get_cache_key(url)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                
            # Check if cache is still valid
            cached_time = datetime.fromisoformat(cached_data['timestamp'])
            if datetime.now() - cached_time < timedelta(seconds=self.cache_duration):
                print(f"Cache hit for {url}")
                return cached_data['content']
                
        return None

    def _save_to_cache(self, url, content):
        """Save scraped data to cache"""
        cache_key = self._get_cache_key(url)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        cache_data = {
            'url': url,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)

    def solve_captcha(self, driver, timeout=10000):
        """Solve captcha using Scraping Browser's built-in mechanism"""
        try:
            solve_res = driver.execute(
                "executeCdpCommand",
                {
                    "cmd": "Captcha.waitForSolve",
                    "params": {"detectTimeout": timeout},
                },
            )
            
            print("Captcha solve status:", solve_res["value"]["status"])
            return solve_res
            
        except Exception as e:
            print(f"Captcha solving error: {e}")
            return None

    def scrape_website(self, website):
        """Scrape website content using Selenium with proxy and caching"""
        try:
            options = ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            
            sbr_connection = ChromiumRemoteConnection(self.SBR_WEBDRIVER, "goog", "chrome")
            
            with Remote(sbr_connection, options=options) as driver:
                driver.set_page_load_timeout(20)
                print(f"\nFetching news from: {website}")
                driver.get(website)
                
                # Wait for content to load
                time.sleep(5)
                
                content = driver.page_source
                
                # Validate content
                if self._validate_content(content, website):
                    self._save_to_cache(website, content)
                    return content
                else:
                    return None
                    
        except Exception as e:
            print(f"Error scraping {website}: {str(e)}")
            return None

    def _validate_content(self, content, website):
        """Validate scraped content"""
        if not content or len(content) < 100:
            print("Warning: Empty or too short content")
            return False

        soup = BeautifulSoup(content, 'html.parser')
        
        # Check for error pages
        error_indicators = ['404', 'not found', 'error', 'page not found']
        page_text = soup.get_text().lower()
        if any(indicator in page_text for indicator in error_indicators):
            print("Warning: Error page detected")
            return False

        # Site-specific validation
        if 'cryptoslate.com' in website:
            return self._validate_cryptoslate(soup)
        elif 'beincrypto.com' in website:
            return self._validate_beincrypto(soup)
        
        return True

    def _validate_cryptoslate(self, soup):
        """Validate CryptoSlate specific content"""
        try:
            # Check for main content containers
            news_items = soup.find_all('div', class_='news-item')
            if not news_items:
                print("Warning: No news items found on CryptoSlate")
                return False

            # Validate content structure
            for item in news_items[:1]:  # Check at least one item
                if not (item.find('h2') and item.find('time')):
                    print("Warning: Invalid news item structure on CryptoSlate")
                    return False

            print(f"Found {len(news_items)} news items on CryptoSlate")
            return True

        except Exception as e:
            print(f"Error validating CryptoSlate content: {e}")
            return False

    def _validate_beincrypto(self, soup):
        """Validate BeInCrypto specific content"""
        try:
            # Check for main content containers
            news_items = soup.find_all('article', class_='post')
            if not news_items:
                print("Warning: No news items found on BeInCrypto")
                return False

            # Validate content structure
            for item in news_items[:1]:  # Check at least one item
                if not (item.find('h2') and item.find('time')):
                    print("Warning: Invalid news item structure on BeInCrypto")
                    return False

            print(f"Found {len(news_items)} news items on BeInCrypto")
            return True

        except Exception as e:
            print(f"Error validating BeInCrypto content: {e}")
            return False

    def clear_cache(self, older_than=None):
        """Clear cache files. If older_than is specified (in seconds), only clear older files"""
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            if older_than:
                # Only clear files older than specified duration
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if datetime.now() - file_time > timedelta(seconds=older_than):
                    os.remove(file_path)
            else:
                # Clear all cache files
                os.remove(file_path)

    def scrape_crypto_sentiment(self, coin_symbol):
        """Scrape crypto news and sentiment from CryptoSlate and BeInCrypto"""
        urls = [
            f"https://cryptoslate.com/coins/{coin_symbol.lower()}",
            f"https://beincrypto.com/coin/{coin_symbol.lower()}"
        ]
        
        news_data = []
        
        for url in urls:
            try:
                print(f"\nScraping news from: {url}")
                html = self.scrape_website(url)
                
                if html:
                    if "cryptoslate.com" in url:
                        data = self._extract_cryptoslate_data(html, coin_symbol)
                    elif "beincrypto.com" in url:
                        data = self._extract_beincrypto_data(html, coin_symbol)
                    
                    if data:
                        news_data.append(data)
                        
            except Exception as e:
                print(f"Error scraping {url}: {str(e)}")
                continue
        
        return news_data

    def _extract_cryptoslate_data(self, html, coin_symbol):
        """Extract specific data from CryptoSlate"""
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            news_data = {
                'source': 'CryptoSlate',
                'articles': [],
                'timestamp': datetime.now().isoformat()
            }
            
            news_items = soup.find_all('div', class_='news-item')
            print(f"Processing {len(news_items)} CryptoSlate articles")
            
            for item in news_items:
                try:
                    title = item.find('h2')
                    date = item.find('time')
                    summary = item.find('p', class_='summary')
                    
                    if title and date:
                        article = {
                            'title': title.text.strip(),
                            'date': date.text.strip(),
                            'summary': summary.text.strip() if summary else "No summary available"
                        }
                        news_data['articles'].append(article)
                        print(f"Added article: {article['title'][:50]}...")
                except Exception as e:
                    print(f"Error extracting article: {str(e)}")
                    continue
                    
            return news_data
            
        except Exception as e:
            print(f"Error parsing CryptoSlate data: {str(e)}")
            return None

    def _extract_beincrypto_data(self, html, coin_symbol):
        """Extract specific data from BeInCrypto"""
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            news_data = {
                'source': 'BeInCrypto',
                'articles': [],
                'timestamp': datetime.now().isoformat()
            }
            
            news_items = soup.find_all('article', class_='post')
            print(f"Processing {len(news_items)} BeInCrypto articles")
            
            for item in news_items:
                try:
                    title = item.find('h2')
                    date = item.find('time')
                    summary = item.find('div', class_='excerpt')
                    
                    if title and date:
                        article = {
                            'title': title.text.strip(),
                            'date': date.text.strip(),
                            'summary': summary.text.strip() if summary else "No summary available"
                        }
                        news_data['articles'].append(article)
                        print(f"Added article: {article['title'][:50]}...")
                except Exception as e:
                    print(f"Error extracting article: {str(e)}")
                    continue
                    
            return news_data
            
        except Exception as e:
            print(f"Error parsing BeInCrypto data: {str(e)}")
            return None

    def print_news_summary(self, news_data):
        """Print formatted news summary"""
        if not news_data:
            print("No news data available")
            return
            
        for source_data in news_data:
            print(f"\n📰 News from {source_data['source']}:")
            print("=" * 50)
            
            for article in source_data['articles']:
                print(f"\nTitle: {article['title']}")
                print(f"Date: {article['date']}")
                print(f"Summary: {article['summary'][:200]}...")
                print("-" * 30)

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
    api_key = '274e832ccba44abfa54914e1ea6915f3'
    news_fetcher = NewsFetcher(api_key)
    news_fetcher.fetch_news()
