import pandas as pd
from typing import Dict, Any

def extract_article_fields(article_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract specified fields from a dictionary of articles and return as a DataFrame.

    Args:
        article_dict (Dict[str, Any]): Dictionary where key is article_id and value is a dict of article metadata.

    Returns:
        pd.DataFrame: DataFrame with one row per article and columns for selected fields.
    """
    fields = [
        'article_id', 'pubDate', 'pubDateTZ', 'title', 'link', 'creator',
        'description', 'source_id', 'source_name', 'source_url', 'source_icon'
    ]
    data = []
    for article_id, item in article_dict.items():
        row = []
        for field in fields:
            if field == 'article_id':
                row.append(article_id)
            elif field == 'creator':
                creators = item.get(field, [])
                if isinstance(creators, list):
                    row.append(' | '.join(str(c) for c in creators))
                else:
                    row.append(str(creators) if creators is not None else '')
            else:
                row.append(item.get(field, ''))
        data.append(row)
    df = pd.DataFrame(data, columns=fields)
    return df


def articles_to_dataframe(article_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert a dictionary of articles into a pandas DataFrame.

    Args:
        article_dict (Dict[str, Any]): Dictionary mapping article IDs to their information.

    Returns:
        pd.DataFrame: DataFrame with columns: article_id, title, summary, text.
    """
    data = []
    for article_id, info in article_dict.items():
        data.append({
            'article_id': article_id,
            # 'title': info.get('title', ''),
            'summary': info.get('summary', ''),
            'text': info.get('text', '')
        })
    return pd.DataFrame(data)