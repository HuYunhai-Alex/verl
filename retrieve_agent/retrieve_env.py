import requests

def search_mail(query: str, top_k: int = 5, filters: dict = None) -> dict:
    """
    调用 FastAPI 的 /search/search_mail 接口并返回 JSON 响应。

    Args:
        query: 要搜索的字符串。
        top_k: 返回结果的数量，默认为 5。
        filters: 额外的过滤条件，默认为空字典。

    Returns:
        接口返回的 JSON 数据，类型为 dict。

    Raises:
        HTTPError: 如果请求返回了 4xx/5xx 状态码，则抛出异常。
    """
    url = "http://127.0.0.1:8023/api/v1/search/search_mail"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {
        "search_query": {
            "query": query,
            "top_k": top_k,
            "filters": filters or {}
        }
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        # 如果状态码不是 2xx，会抛出 HTTPError
        response.raise_for_status()    
    except Exception as e:
        print(f"未知错误: {e}")
        return None
    return response.json()

def execute_sql(query: str, top_k: int = 5, filters: dict = None) -> dict:
    """
    调用 FastAPI 的 /search/search_mail 接口并返回 JSON 响应。

    Args:
        query: 要搜索的字符串。
        top_k: 返回结果的数量，默认为 5。
        filters: 额外的过滤条件，默认为空字典。

    Returns:
        接口返回的 JSON 数据，类型为 dict。

    Raises:
        HTTPError: 如果请求返回了 4xx/5xx 状态码，则抛出异常。
    """
    url = "http://127.0.0.1:8023/api/v1/search/execute_sql"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {
        "search_query": {
            "query": query,
            "top_k": top_k,
            "filters": filters or {}
        }
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        # 如果状态码不是 2xx，会抛出 HTTPError
        response.raise_for_status()    
    except Exception as e:
        print(f"未知错误: {e}")
        return None
    return response.json()

if __name__ == "__main__":
    # 示例调用
    try:
        result = search_mail(query="{'sql': 'SELECT message_id FROM email WHERE message_id IN <vector_0>', 'vector_query_list': [{'subject': '预算'}]}", top_k=5)
        print("Search Results:", result)
    except requests.HTTPError as e:
        print(f"请求失败: {e}  响应内容: {e.response.text}")
    except Exception as e:
        print(f"未知错误: {e}")