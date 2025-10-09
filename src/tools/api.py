import datetime
import io
import os
import pandas as pd
import requests
import time
from requests import exceptions as requests_exceptions

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyFactsResponse,
)

# Global cache instance
_cache = get_cache()


def _parse_timeout(env_var: str, default: float) -> float:
    value = os.environ.get(env_var)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_positive_int(env_var: str, default: int) -> int:
    value = os.environ.get(env_var)
    if value is None:
        return default
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except ValueError:
        return default


COMPANY_NEWS_TIMEOUT_SECONDS = _parse_timeout("COMPANY_NEWS_TIMEOUT_SECONDS", 10.0)
COMPANY_NEWS_FETCH_TIMEOUT_SECONDS = _parse_timeout("COMPANY_NEWS_FETCH_TIMEOUT_SECONDS", 60.0)
COMPANY_NEWS_MAX_PAGES = _parse_positive_int("COMPANY_NEWS_MAX_PAGES", 8)
DEFAULT_API_REQUEST_TIMEOUT_SECONDS = _parse_timeout("DEFAULT_API_REQUEST_TIMEOUT_SECONDS", 30.0)


def _make_api_request(
    url: str,
    headers: dict,
    method: str = "GET",
    json_data: dict = None,
    max_retries: int = 3,
    request_timeout: float | None = None,
) -> requests.Response:
    """
    Make an API request with rate limiting handling and moderate backoff.
    
    Args:
        url: The URL to request
        headers: Headers to include in the request
        method: HTTP method (GET or POST)
        json_data: JSON data for POST requests
        max_retries: Maximum number of retries (default: 3)
        request_timeout: Optional timeout applied to the request in seconds
    
    Returns:
        requests.Response: The response object
    
    Raises:
        Exception: If the request fails with a non-429 error
    """
    effective_timeout = (
        DEFAULT_API_REQUEST_TIMEOUT_SECONDS if request_timeout is None else request_timeout
    )

    for attempt in range(max_retries + 1):  # +1 for initial attempt
        request_kwargs = {"headers": headers}
        if method.upper() == "POST":
            request_kwargs["json"] = json_data
        if effective_timeout is not None:
            request_kwargs["timeout"] = effective_timeout

        try:
            if method.upper() == "POST":
                response = requests.post(url, **request_kwargs)
            else:
                response = requests.get(url, **request_kwargs)
        except requests_exceptions.Timeout as exc:
            raise TimeoutError(f"Request to {url} timed out") from exc
        except requests_exceptions.RequestException as exc:
            raise Exception(f"Error during request to {url}: {exc}") from exc
        
        if response.status_code == 429 and attempt < max_retries:
            # Linear backoff: 60s, 90s, 120s, 150s...
            delay = 60 + (30 * attempt)
            print(f"Rate limited (429). Attempt {attempt + 1}/{max_retries + 1}. Waiting {delay}s before retrying...")
            time.sleep(delay)
            continue
        
        # Return the response (whether success, other errors, or final 429)
        return response

STOOQ_SYMBOL_MAP = {
    "^GSPC": "^spx",
    "^SPX": "^spx",
    "SPX": "^spx",
    "SP500": "^spx",
    "S&P500": "^spx",
    "SPY": "spy.us",
}


def _fetch_stooq_prices(
    ticker: str,
    start_date: str,
    end_date: str,
    cache_key: str,
) -> list[Price] | None:
    """Fetch daily prices from Stooq as an unauthenticated fallback."""

    symbol = STOOQ_SYMBOL_MAP.get(ticker.upper()) or STOOQ_SYMBOL_MAP.get(ticker)
    if not symbol:
        return None

    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    try:
        response = requests.get(url, timeout=DEFAULT_API_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests_exceptions.RequestException:
        return []

    csv_buffer = io.StringIO(response.text)
    try:
        df = pd.read_csv(csv_buffer)
    except pd.errors.EmptyDataError:
        return []

    if df.empty or "Date" not in df:
        return []

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    df = df[(df["Date"] >= start) & (df["Date"] <= end)].sort_values("Date")

    prices: list[Price] = []
    for _, row in df.iterrows():
        try:
            price = Price(
                open=float(row["Open"]),
                close=float(row["Close"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                volume=int(row.get("Volume", 0) or 0),
                time=f"{row['Date'].strftime('%Y-%m-%d')}T00:00:00Z",
            )
            prices.append(price)
        except (TypeError, ValueError):
            continue

    if prices:
        _cache.set_prices(cache_key, [p.model_dump() for p in prices])

    return prices


def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
    """Fetch price data from cache or API."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{start_date}_{end_date}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]

    # If not in cache, fetch from API
    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    url = f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"
    try:
        response = _make_api_request(
            url,
            headers,
            request_timeout=DEFAULT_API_REQUEST_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        print(
            f"Skipping price fetch for {ticker} because the request timed out after {DEFAULT_API_REQUEST_TIMEOUT_SECONDS} seconds."
        )
        return []
    if response.status_code != 200:
        fallback_prices = _fetch_stooq_prices(ticker, start_date, end_date, cache_key)
        if fallback_prices is not None:
            return fallback_prices
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

    # Parse response with Pydantic model
    price_response = PriceResponse(**response.json())
    prices = price_response.prices

    if not prices:
        fallback_prices = _fetch_stooq_prices(ticker, start_date, end_date, cache_key)
        if fallback_prices is not None:
            return fallback_prices
        return []

    # Cache the results using the comprehensive cache key
    _cache.set_prices(cache_key, [p.model_dump() for p in prices])
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or API."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**metric) for metric in cached_data]

    # If not in cache, fetch from API
    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    url = f"https://api.financialdatasets.ai/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"
    try:
        response = _make_api_request(
            url,
            headers,
            request_timeout=DEFAULT_API_REQUEST_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        print(
            f"Skipping financial metrics for {ticker} because the request timed out after {DEFAULT_API_REQUEST_TIMEOUT_SECONDS} seconds."
        )
        return []
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

    # Parse response with Pydantic model
    metrics_response = FinancialMetricsResponse(**response.json())
    financial_metrics = metrics_response.financial_metrics

    if not financial_metrics:
        return []

    # Cache the results as dicts using the comprehensive cache key
    _cache.set_financial_metrics(cache_key, [m.model_dump() for m in financial_metrics])
    return financial_metrics


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    """Fetch line items from API."""
    cache_key = f"{ticker}_{period}_{end_date}_{limit}_{'-'.join(sorted(line_items))}"

    if cached_data := _cache.get_line_items(cache_key):
        return [LineItem(**item) for item in cached_data][:limit]

    # If not in cache or insufficient data, fetch from API
    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    url = "https://api.financialdatasets.ai/financials/search/line-items"

    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": end_date,
        "period": period,
        "limit": limit,
    }
    try:
        response = _make_api_request(
            url,
            headers,
            method="POST",
            json_data=body,
            request_timeout=DEFAULT_API_REQUEST_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        print(
            f"Skipping line item search for {ticker} because the request timed out after {DEFAULT_API_REQUEST_TIMEOUT_SECONDS} seconds."
        )
        return []
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
    data = response.json()
    response_model = LineItemResponse(**data)
    search_results = response_model.search_results
    if not search_results:
        return []

    _cache.set_line_items(cache_key, [item.model_dump() for item in search_results])
    return search_results[:limit]


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or API."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**trade) for trade in cached_data]

    # If not in cache, fetch from API
    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    all_trades = []
    current_end_date = end_date

    while True:
        url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
        if start_date:
            url += f"&filing_date_gte={start_date}"
        url += f"&limit={limit}"

        try:
            response = _make_api_request(
                url,
                headers,
                request_timeout=DEFAULT_API_REQUEST_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            print(
                f"Stopping insider trades fetch for {ticker} because the request timed out after {DEFAULT_API_REQUEST_TIMEOUT_SECONDS} seconds."
            )
            break
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

        data = response.json()
        response_model = InsiderTradeResponse(**data)
        insider_trades = response_model.insider_trades

        if not insider_trades:
            break

        all_trades.extend(insider_trades)

        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(insider_trades) < limit:
            break

        # Update end_date to the oldest filing date from current batch for next iteration
        current_end_date = min(trade.filing_date for trade in insider_trades).split("T")[0]

        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_trades:
        return []

    # Cache the results using the comprehensive cache key
    _cache.set_insider_trades(cache_key, [trade.model_dump() for trade in all_trades])
    return all_trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[CompanyNews]:
    """Fetch company news from cache or API."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_company_news(cache_key):
        return [CompanyNews(**news) for news in cached_data]

    # If not in cache, fetch from API
    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    all_news = []
    current_end_date = end_date

    fetch_timeout = COMPANY_NEWS_FETCH_TIMEOUT_SECONDS if COMPANY_NEWS_FETCH_TIMEOUT_SECONDS > 0 else None
    fetch_deadline = (time.time() + fetch_timeout) if fetch_timeout else None
    max_pages = COMPANY_NEWS_MAX_PAGES if COMPANY_NEWS_MAX_PAGES > 0 else None
    page_count = 0

    while True:
        if fetch_deadline and time.time() >= fetch_deadline:
            print(
                f"Stopping company news fetch for {ticker} after {fetch_timeout} seconds (partial results returned)."
            )
            break
        if max_pages is not None and page_count >= max_pages:
            print(f"Stopping company news fetch for {ticker} after reaching {max_pages} pages.")
            break

        url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={limit}"

        try:
            response = _make_api_request(
                url,
                headers,
                request_timeout=COMPANY_NEWS_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            print(
                f"Skipping company news for {ticker} because the request timed out after {COMPANY_NEWS_TIMEOUT_SECONDS} seconds."
            )
            return []
        if response.status_code == 404:
            print(
                f"Company news unavailable for {ticker} between {start_date or 'beginning'} and {current_end_date}; proceeding without news coverage."
            )
            _cache.set_company_news(cache_key, [])
            return []
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

        data = response.json()
        response_model = CompanyNewsResponse(**data)
        company_news = response_model.news

        if not company_news:
            break

        all_news.extend(company_news)
        page_count += 1

        if fetch_deadline and time.time() >= fetch_deadline:
            print(
                f"Stopping company news fetch for {ticker} after {fetch_timeout} seconds (partial results returned)."
            )
            break

        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(company_news) < limit:
            break

        # Update end_date to the oldest date from current batch for next iteration
        current_end_date = min(news.date for news in company_news).split("T")[0]

        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_news:
        return []

    # Cache the results using the comprehensive cache key
    _cache.set_company_news(cache_key, [news.model_dump() for news in all_news])
    return all_news


def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> float | None:
    """Fetch market cap from the API."""
    # Check if end_date is today
    if end_date == datetime.datetime.now().strftime("%Y-%m-%d"):
        # Get the market cap from company facts API
        headers = {}
        financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
        if financial_api_key:
            headers["X-API-KEY"] = financial_api_key

        url = f"https://api.financialdatasets.ai/company/facts/?ticker={ticker}"
        try:
            response = _make_api_request(
                url,
                headers,
                request_timeout=DEFAULT_API_REQUEST_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            print(
                f"Skipping company facts for {ticker} because the request timed out after {DEFAULT_API_REQUEST_TIMEOUT_SECONDS} seconds."
            )
            return None
        if response.status_code != 200:
            print(f"Error fetching company facts: {ticker} - {response.status_code}")
            return None

        data = response.json()
        response_model = CompanyFactsResponse(**data)
        return response_model.company_facts.market_cap

    financial_metrics = get_financial_metrics(ticker, end_date, api_key=api_key)
    if not financial_metrics:
        return None

    market_cap = financial_metrics[0].market_cap

    if not market_cap:
        return None

    return market_cap


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date, api_key=api_key)
    return prices_to_df(prices)
