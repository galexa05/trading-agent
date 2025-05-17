import os
import requests
import pickle
from newspaper import Article
from typing import List, Dict, Any, Optional

class NewsDataCollector:
    def __init__(self, api_key: Optional[str] = None, data_dir: str = '../data'):
        """
        Initialize the NewsDataCollector with an API key and a directory to store data.

        Args:
            api_key (Optional[str]): API key for NewsData API. If None, fetches from environment variable.
            data_dir (str): Directory path to save output pickle files.
        """
        self.api_key = api_key or os.getenv('NEWS_DATA_API')
        self.data_dir = data_dir

    def fetch_articles(
        self,
        query: str = "AAPL",
        language: str = "en",
        category: str = "business",
        max_iterations: int = 20,
        remove_duplicate: int = 1,
        save_pickle: bool = True,
        pickle_name: str = "article_dict.pkl",
    ) -> Dict[str, Any]:
        """
        Fetch articles from NewsData API and optionally save them as a pickle file.

        Args:
            query (str): Search query term.
            language (str): Language of the articles.
            category (str): News category to filter.
            max_iterations (int): Maximum number of paginated API requests.
            remove_duplicate (int): Whether to remove duplicates (1 = True).
            save_pickle (bool): Whether to save the result as a pickle.
            pickle_name (str): Name for the saved pickle file.

        Returns:
            Dict[str, Any]: Dictionary mapping article IDs to article metadata.
        """
        main_url = (
            "https://newsdata.io/api/1/latest"
            f"?apikey={self.api_key}"
            f"&q={query}"
            f"&language={language}"
            f"&category={category}"
            f"&removeduplicate={remove_duplicate}"
        )
        n_iterations = 0
        result_list = []
        url = main_url
        while n_iterations < max_iterations:
            response = requests.get(url)
            data = response.json()
            if 'results' in data:
                result_list += data['results']
            if 'nextPage' not in data:
                break
            nextPageId = data['nextPage']
            url = f"{main_url}&page={nextPageId}"
            n_iterations += 1
            # print(f'Number of iterations: {n_iterations}')

        article_dict = {item['article_id']: item for item in result_list if 'article_id' in item}
        if save_pickle:
            os.makedirs(self.data_dir, exist_ok=True)
            with open(os.path.join(self.data_dir, pickle_name), 'wb') as f:
                pickle.dump(article_dict, f)
        return article_dict


    def fetch_portfolio_articles(
        self,
        tickers: List[str],
        language: str = "en",
        category: str = "business",
        max_iterations_per_ticker: int = 10,
        remove_duplicate: int = 1,
        save_pickle: bool = True,
        pickle_name: str = "portfolio_articles.pkl",
    ) -> Dict[str, Any]:
        """
        Fetch articles for multiple stock tickers in a portfolio from NewsData API.

        Args:
            tickers (List[str]): List of stock ticker symbols (e.g., ["AAPL", "TSLA", "MSFT"]).
            language (str): Language of the articles.
            category (str): News category to filter.
            max_iterations_per_ticker (int): Maximum number of API requests per ticker.
            remove_duplicate (int): Whether to remove duplicates (1 = True).
            save_pickle (bool): Whether to save the result as a pickle.
            pickle_name (str): Name for the saved pickle file.

        Returns:
            Dict[str, Any]: Combined dictionary of articles for all tickers with ticker information added.
        """
        combined_articles = {}
        
        for ticker in tickers:
            print(f"Fetching articles for {ticker}...")
            
            # Get articles for this ticker
            ticker_articles = self.fetch_articles(
                query=ticker,
                language=language,
                category=category,
                max_iterations=max_iterations_per_ticker,
                remove_duplicate=remove_duplicate,
                save_pickle=False  # Don't save individual ticker results
            )
            
            # Add ticker information to each article and add to combined results
            for article_id, article_data in ticker_articles.items():
                # Skip if this article was already found for another ticker
                if article_id in combined_articles:
                    # If already in combined results, just add this ticker to the tickers list
                    if 'tickers' in combined_articles[article_id]:
                        if ticker not in combined_articles[article_id]['tickers']:
                            combined_articles[article_id]['tickers'].append(ticker)
                    else:
                        combined_articles[article_id]['tickers'] = [ticker]
                    continue
                
                # Add this article to combined results with ticker information
                article_data['tickers'] = [ticker]
                combined_articles[article_id] = article_data
        
        print(f"Found {len(combined_articles)} unique articles for portfolio of {len(tickers)} stocks")
        
        # Save combined results if requested
        if save_pickle and combined_articles:
            os.makedirs(self.data_dir, exist_ok=True)
            with open(os.path.join(self.data_dir, pickle_name), 'wb') as f:
                pickle.dump(combined_articles, f)
            print(f"Saved portfolio articles to {os.path.join(self.data_dir, pickle_name)}")
        
        return combined_articles


    def enrich_articles_context(
        self,
        article_dict: Dict[str, Any],
        context_pickle_name: str = "article_id_full_context.pkl",
        save_pickle: bool = True,
    ) -> Dict[str, Any]:
        """
        Enrich each article with full context (title, summary, keywords, text, etc.) using newspaper3k.

        Args:
            article_dict (Dict[str, Any]): Dictionary of articles with their metadata.
            context_pickle_name (str): Name for the output pickle file with full context.
            save_pickle (bool): Whether to save the enriched context as a pickle.

        Returns:
            Dict[str, Any]: Dictionary mapping article IDs to their enriched context.
        """
        article_full_context = {}
        for article_id, article_data in article_dict.items():
            if article_id in article_full_context:
                # print(f"{article_id} - Already collected")
                continue
            article_link = article_data.get('link')
            if not article_link:
                # print(f"{article_id} - No link found")
                continue
            article = Article(article_link)
            try:
                article.download()
                article.parse()
                article.nlp()
                context_info = {
                    'title': article.title,
                    'authors': article.authors,
                    'publish_date': article.publish_date,
                    'summary': article.summary,
                    'keywords': article.keywords,
                    'text': article.text,
                    'top_image': article.top_image,
                    'movies': article.movies,
                    'url': article_link
                }
                article_full_context[article_id] = context_info
                # print(f"{article_id} - Done")
            except Exception as e:
                pass
                # print(f"{article_id} - Error: {e}")
        if save_pickle:
            os.makedirs(self.data_dir, exist_ok=True)
            with open(os.path.join(self.data_dir, context_pickle_name), 'wb') as f:
                pickle.dump(article_full_context, f)
        return article_full_context

# Example usage:
# collector = NewsDataCollector()
# article_dict = collector.fetch_articles(query="AAPL", max_iterations=10)
# article_full_context = collector.enrich_articles_context(article_dict)