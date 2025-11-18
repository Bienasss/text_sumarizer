import requests
from bs4 import BeautifulSoup
import json
import time
import os
from typing import List, Dict, Set
import re
import xml.etree.ElementTree as ET

class NewsCollector:
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.collected_urls: Set[str] = set()
    
    def collect_bbc_news(self, num_articles: int = 1500) -> List[Dict]:
        articles = []
        all_links = set()
        
        rss_feeds = [
            "http://feeds.bbci.co.uk/news/rss.xml",
            "http://feeds.bbci.co.uk/news/uk/rss.xml",
            "http://feeds.bbci.co.uk/news/world/rss.xml",
            "http://feeds.bbci.co.uk/news/business/rss.xml",
            "http://feeds.bbci.co.uk/news/technology/rss.xml",
            "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
            "http://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
            "http://feeds.bbci.co.uk/news/health/rss.xml",
            "http://feeds.bbci.co.uk/news/education/rss.xml"
        ]
        
        print("Collecting links from BBC RSS feeds...")
        for rss_url in rss_feeds:
            try:
                links = self._parse_rss_feed(rss_url)
                all_links.update(links)
                print(f"Found {len(links)} links from {rss_url}")
                time.sleep(0.5)
            except Exception as e:
                print(f"Error fetching RSS feed {rss_url}: {e}")
                continue
        
        base_urls = [
            "https://www.bbc.com/news",
            "https://www.bbc.com/news/uk",
            "https://www.bbc.com/news/world",
            "https://www.bbc.com/news/business",
            "https://www.bbc.com/news/technology"
        ]
        
        print("Collecting links from BBC category pages...")
        for base_url in base_urls:
            if len(all_links) >= num_articles * 3:
                break
            try:
                response = self.session.get(base_url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                for link in soup.find_all('a', href=True):
                    href = link.get('href')
                    if href and '/news/' in href and href.startswith('/') and len(href) > 10:
                        full_url = f"https://www.bbc.com{href}"
                        all_links.add(full_url)
                
                time.sleep(1)
            except Exception as e:
                print(f"Error fetching {base_url}: {e}")
                continue
        
        print(f"Found {len(all_links)} total potential article links")
        
        links = list(all_links)
        for i, url in enumerate(links):
            if len(articles) >= num_articles:
                break
            
            if url in self.collected_urls:
                continue
            
            try:
                article = self._fetch_bbc_article(url)
                if article and article.get('text') and len(article['text']) > 200:
                    articles.append(article)
                    self.collected_urls.add(url)
                    if len(articles) % 25 == 0:
                        print(f"Collected {len(articles)}/{num_articles} BBC articles")
                
                time.sleep(0.3)
            except Exception as e:
                continue
        
        return articles
    
    def _parse_rss_feed(self, rss_url: str) -> List[str]:
        links = []
        try:
            response = self.session.get(rss_url, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            
            for item in root.findall('.//item'):
                link_elem = item.find('link')
                if link_elem is not None and link_elem.text:
                    links.append(link_elem.text)
        except Exception as e:
            pass
        
        return links
    
    def _fetch_bbc_article(self, url: str) -> Dict:
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title_elem = soup.find('h1')
            title = title_elem.get_text(strip=True) if title_elem else ""
            
            article_body = soup.find('article') or soup.find('div', {'data-component': 'text-block'})
            if not article_body:
                article_body = soup.find('div', class_=re.compile('story-body|article-body'))
            
            paragraphs = []
            if article_body:
                for p in article_body.find_all(['p', 'div'], class_=re.compile('paragraph|text')):
                    text = p.get_text(strip=True)
                    if text and len(text) > 20:
                        paragraphs.append(text)
            
            if not paragraphs:
                for p in soup.find_all('p'):
                    text = p.get_text(strip=True)
                    if len(text) > 50:
                        paragraphs.append(text)
            
            text = ' '.join(paragraphs)
            
            if len(text) < 200:
                return None
            
            return {
                'title': title,
                'text': text,
                'url': url,
                'source': 'BBC'
            }
        except Exception as e:
            return None
    
    def collect_guardian_news(self, num_articles: int = 500) -> List[Dict]:
        articles = []
        all_links = set()
        
        rss_feeds = [
            "https://www.theguardian.com/world/rss",
            "https://www.theguardian.com/uk/rss",
            "https://www.theguardian.com/business/rss",
            "https://www.theguardian.com/technology/rss",
            "https://www.theguardian.com/science/rss",
            "https://www.theguardian.com/culture/rss",
            "https://www.theguardian.com/politics/rss",
            "https://www.theguardian.com/sport/rss"
        ]
        
        print("Collecting links from Guardian RSS feeds...")
        for rss_url in rss_feeds:
            try:
                links = self._parse_rss_feed(rss_url)
                all_links.update(links)
                print(f"Found {len(links)} links from {rss_url}")
                time.sleep(0.5)
            except Exception as e:
                print(f"Error fetching RSS feed {rss_url}: {e}")
                continue
        
        base_urls = [
            "https://www.theguardian.com/international",
            "https://www.theguardian.com/uk",
            "https://www.theguardian.com/world",
            "https://www.theguardian.com/business",
            "https://www.theguardian.com/technology"
        ]
        
        print("Collecting links from Guardian category pages...")
        for base_url in base_urls:
            if len(all_links) >= num_articles * 3:
                break
            try:
                response = self.session.get(base_url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                for link in soup.find_all('a', href=True):
                    href = link.get('href')
                    if href and '/202' in href and href.startswith('/') and len(href) > 10:
                        full_url = f"https://www.theguardian.com{href}"
                        all_links.add(full_url)
                
                time.sleep(1)
            except Exception as e:
                print(f"Error fetching {base_url}: {e}")
                continue
        
        print(f"Found {len(all_links)} total potential Guardian article links")
        
        links = list(all_links)
        for url in links:
            if len(articles) >= num_articles:
                break
            
            if url in self.collected_urls:
                continue
            
            try:
                article = self._fetch_guardian_article(url)
                if article and article.get('text') and len(article['text']) > 200:
                    articles.append(article)
                    self.collected_urls.add(url)
                    if len(articles) % 25 == 0:
                        print(f"Collected {len(articles)}/{num_articles} Guardian articles")
                
                time.sleep(0.3)
            except Exception as e:
                continue
        
        return articles
    
    def _fetch_guardian_article(self, url: str) -> Dict:
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title_elem = soup.find('h1')
            title = title_elem.get_text(strip=True) if title_elem else ""
            
            article_body = soup.find('div', {'data-gu-name': 'body'}) or soup.find('div', class_=re.compile('article-body'))
            
            paragraphs = []
            if article_body:
                for p in article_body.find_all('p'):
                    text = p.get_text(strip=True)
                    if text and len(text) > 20:
                        paragraphs.append(text)
            
            if not paragraphs:
                for p in soup.find_all('p', class_=re.compile('paragraph')):
                    text = p.get_text(strip=True)
                    if len(text) > 50:
                        paragraphs.append(text)
            
            text = ' '.join(paragraphs)
            
            if len(text) < 200:
                return None
            
            return {
                'title': title,
                'text': text,
                'url': url,
                'source': 'Guardian'
            }
        except Exception as e:
            return None
    
    def collect_fox_news(self, num_articles: int = 300) -> List[Dict]:
        articles = []
        all_links = set()
        
        rss_feeds = [
            "https://feeds.foxnews.com/foxnews/latest",
            "https://feeds.foxnews.com/foxnews/politics",
            "https://feeds.foxnews.com/foxnews/world",
            "https://feeds.foxnews.com/foxnews/business",
            "https://feeds.foxnews.com/foxnews/tech",
            "https://feeds.foxnews.com/foxnews/science",
            "https://feeds.foxnews.com/foxnews/health",
            "https://feeds.foxnews.com/foxnews/entertainment"
        ]
        
        print("Collecting links from Fox News RSS feeds...")
        for rss_url in rss_feeds:
            try:
                links = self._parse_rss_feed(rss_url)
                all_links.update(links)
                print(f"Found {len(links)} links from {rss_url}")
                time.sleep(0.5)
            except Exception as e:
                print(f"Error fetching RSS feed {rss_url}: {e}")
                continue
        
        base_urls = [
            "https://www.foxnews.com",
            "https://www.foxnews.com/politics",
            "https://www.foxnews.com/world",
            "https://www.foxnews.com/business",
            "https://www.foxnews.com/tech"
        ]
        
        print("Collecting links from Fox News category pages...")
        for base_url in base_urls:
            if len(all_links) >= num_articles * 3:
                break
            try:
                response = self.session.get(base_url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                for link in soup.find_all('a', href=True):
                    href = link.get('href')
                    if href and '/202' in href and href.startswith('/'):
                        if href.startswith('http'):
                            full_url = href
                        else:
                            full_url = f"https://www.foxnews.com{href}"
                        if 'foxnews.com' in full_url:
                            all_links.add(full_url)
                
                time.sleep(1)
            except Exception as e:
                print(f"Error fetching {base_url}: {e}")
                continue
        
        print(f"Found {len(all_links)} total potential Fox News article links")
        
        links = list(all_links)
        for url in links:
            if len(articles) >= num_articles:
                break
            
            if url in self.collected_urls:
                continue
            
            try:
                article = self._fetch_fox_article(url)
                if article and article.get('text') and len(article['text']) > 200:
                    articles.append(article)
                    self.collected_urls.add(url)
                    if len(articles) % 25 == 0:
                        print(f"Collected {len(articles)}/{num_articles} Fox News articles")
                
                time.sleep(0.3)
            except Exception as e:
                continue
        
        return articles
    
    def _fetch_fox_article(self, url: str) -> Dict:
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title_elem = soup.find('h1') or soup.find('h2', class_=re.compile('headline'))
            title = title_elem.get_text(strip=True) if title_elem else ""
            
            article_body = soup.find('div', class_=re.compile('article-body|entry-content|article-text'))
            
            paragraphs = []
            if article_body:
                for p in article_body.find_all('p'):
                    text = p.get_text(strip=True)
                    if text and len(text) > 20:
                        paragraphs.append(text)
            
            if not paragraphs:
                for p in soup.find_all('p', class_=re.compile('speakable|paragraph')):
                    text = p.get_text(strip=True)
                    if len(text) > 50:
                        paragraphs.append(text)
            
            if not paragraphs:
                for p in soup.find_all('p'):
                    text = p.get_text(strip=True)
                    if len(text) > 100:
                        paragraphs.append(text)
            
            text = ' '.join(paragraphs)
            
            if len(text) < 200:
                return None
            
            return {
                'title': title,
                'text': text,
                'url': url,
                'source': 'Fox News'
            }
        except Exception as e:
            return None
    
    def save_articles(self, articles: List[Dict], filename: str = "articles.json"):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(articles)} articles to {filepath}")
    
    def load_articles(self, filename: str = "articles.json") -> List[Dict]:
        filepath = os.path.join(self.output_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

if __name__ == "__main__":
    collector = NewsCollector()
    
    print("Collecting BBC news articles...")
    bbc_articles = collector.collect_bbc_news(num_articles=500)
    
    print("Collecting Guardian news articles...")
    guardian_articles = collector.collect_guardian_news(num_articles=400)
    
    print("Collecting Fox News articles...")
    fox_articles = collector.collect_fox_news(num_articles=300)
    
    all_articles = bbc_articles + guardian_articles + fox_articles
    print(f"Total articles collected: {len(all_articles)}")
    
    collector.save_articles(all_articles, "articles.json")

