import os
from datetime import datetime
from unittest.mock import ANY, call, Mock, patch

import pytest

from src.backtesting.metrics import PerformanceMetricsCalculator
from src.tools.api import _make_api_request, get_prices


class TestRateLimiting:
    """Test suite for API rate limiting functionality."""

    @patch("src.tools.api.time.sleep")
    @patch("src.tools.api.requests.get")
    def test_handles_single_rate_limit(self, mock_get, mock_sleep):
        """Test that API retries once after a 429 and succeeds."""
        # Setup mock responses: first 429, then 200
        mock_429_response = Mock()
        mock_429_response.status_code = 429

        mock_200_response = Mock()
        mock_200_response.status_code = 200
        mock_200_response.text = "Success"

        mock_get.side_effect = [mock_429_response, mock_200_response]

        # Call the function
        headers = {"X-API-KEY": "test-key"}
        url = "https://api.financialdatasets.ai/test"

        result = _make_api_request(url, headers)

        # Verify behavior
        assert result.status_code == 200
        assert result.text == "Success"

        # Verify requests.get was called twice
        assert mock_get.call_count == 2
        mock_get.assert_has_calls([call(url, headers=headers, timeout=ANY), call(url, headers=headers, timeout=ANY)])

        # Verify sleep was called once with 60 seconds (first retry)
        mock_sleep.assert_called_once_with(60)

    @patch("src.tools.api.time.sleep")
    @patch("src.tools.api.requests.get")
    def test_handles_multiple_rate_limits(self, mock_get, mock_sleep):
        """Test that API retries multiple times after 429s."""
        # Setup mock responses: three 429s, then 200
        mock_429_response = Mock()
        mock_429_response.status_code = 429

        mock_200_response = Mock()
        mock_200_response.status_code = 200
        mock_200_response.text = "Success"

        mock_get.side_effect = [mock_429_response, mock_429_response, mock_429_response, mock_200_response]

        # Call the function
        headers = {"X-API-KEY": "test-key"}
        url = "https://api.financialdatasets.ai/test"

        result = _make_api_request(url, headers)

        # Verify behavior
        assert result.status_code == 200
        assert result.text == "Success"

        # Verify requests.get was called 4 times
        assert mock_get.call_count == 4

        # Verify sleep was called 3 times with linear backoff: 60s, 90s, 120s
        assert mock_sleep.call_count == 3
        expected_calls = [call(60), call(90), call(120)]
        mock_sleep.assert_has_calls(expected_calls)

    @patch("src.tools.api.time.sleep")
    @patch("src.tools.api.requests.post")
    def test_handles_post_rate_limiting(self, mock_post, mock_sleep):
        """Test that POST requests handle rate limiting."""
        # Setup mock responses: first 429, then 200
        mock_429_response = Mock()
        mock_429_response.status_code = 429

        mock_200_response = Mock()
        mock_200_response.status_code = 200
        mock_200_response.text = "Success"

        mock_post.side_effect = [mock_429_response, mock_200_response]

        # Call the function with POST method
        headers = {"X-API-KEY": "test-key"}
        url = "https://api.financialdatasets.ai/test"
        json_data = {"test": "data"}

        result = _make_api_request(url, headers, method="POST", json_data=json_data)

        # Verify behavior
        assert result.status_code == 200
        assert result.text == "Success"

        # Verify requests.post was called twice
        assert mock_post.call_count == 2
        mock_post.assert_has_calls([call(url, headers=headers, json=json_data, timeout=ANY), call(url, headers=headers, json=json_data, timeout=ANY)])

        # Verify sleep was called once with 60 seconds (first retry)
        mock_sleep.assert_called_once_with(60)

    @patch("src.tools.api.time.sleep")
    @patch("src.tools.api.requests.get")
    def test_ignores_other_errors(self, mock_get, mock_sleep):
        """Test that non-429 errors are returned without retrying."""
        # Setup mock response: 500 error
        mock_500_response = Mock()
        mock_500_response.status_code = 500
        mock_500_response.text = "Internal Server Error"

        mock_get.return_value = mock_500_response

        # Call the function
        headers = {"X-API-KEY": "test-key"}
        url = "https://api.financialdatasets.ai/test"

        result = _make_api_request(url, headers)

        # Verify behavior
        assert result.status_code == 500
        assert result.text == "Internal Server Error"

        # Verify requests.get was called only once
        assert mock_get.call_count == 1

        # Verify sleep was never called
        mock_sleep.assert_not_called()

    @patch("src.tools.api.time.sleep")
    @patch("src.tools.api.requests.get")
    def test_normal_success_requests(self, mock_get, mock_sleep):
        """Test that successful requests return immediately without retry."""
        # Setup mock response: 200 success
        mock_200_response = Mock()
        mock_200_response.status_code = 200
        mock_200_response.text = "Success"

        mock_get.return_value = mock_200_response

        # Call the function
        headers = {"X-API-KEY": "test-key"}
        url = "https://api.financialdatasets.ai/test"

        result = _make_api_request(url, headers)

        # Verify behavior
        assert result.status_code == 200
        assert result.text == "Success"

        # Verify requests.get was called only once
        assert mock_get.call_count == 1

        # Verify sleep was never called
        mock_sleep.assert_not_called()

    @patch("src.tools.api._cache")
    @patch("src.tools.api.time.sleep")
    @patch("src.tools.api.requests.get")
    def test_full_integration(self, mock_get, mock_sleep, mock_cache):
        """Test that get_prices function properly handles rate limiting."""
        # Mock cache to return None (cache miss)
        mock_cache.get_prices.return_value = None

        # Setup mock responses: first 429, then 200 with valid data
        mock_429_response = Mock()
        mock_429_response.status_code = 429

        mock_200_response = Mock()
        mock_200_response.status_code = 200
        mock_200_response.json.return_value = {"ticker": "AAPL", "prices": [{"time": "2024-01-01T00:00:00Z", "open": 100.0, "close": 101.0, "high": 102.0, "low": 99.0, "volume": 1000}]}

        mock_get.side_effect = [mock_429_response, mock_200_response]

        # Set environment variable for API key
        with patch.dict(os.environ, {"FINANCIAL_DATASETS_API_KEY": "test-key"}):
            # Call get_prices
            result = get_prices("AAPL", "2024-01-01", "2024-01-02")

        # Verify the function succeeded and returned data
        assert len(result) == 1
        assert result[0].open == 100.0
        assert result[0].close == 101.0

        # Verify rate limiting behavior
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once_with(60)

        # Verify cache operations
        mock_cache.get_prices.assert_called_once()
        mock_cache.set_prices.assert_called_once()

    @patch("src.tools.api._cache")
    @patch("src.tools.api.requests.get")
    @patch("src.tools.api._make_api_request")
    def test_stooq_fallback_for_index(self, mock_api_request, mock_requests_get, mock_cache):
        """Ensure Stooq fallback provides index data when vendor API rejects the ticker."""

        mock_cache.get_prices.return_value = None
        mock_cache.set_prices.return_value = None

        api_failure = Mock()
        api_failure.status_code = 404
        api_failure.text = "No data"
        mock_api_request.return_value = api_failure

        stooq_response = Mock()
        stooq_response.status_code = 200
        stooq_response.text = "Date,Open,High,Low,Close,Volume\n" "2024-07-01,5471.08,5479.55,5446.53,5475.09,2433991467\n" "2024-07-02,5469.45,5488.52,5452.68,5481.11,2250709371\n"
        mock_requests_get.return_value = stooq_response

        result = get_prices("^GSPC", "2024-07-01", "2024-07-05")

        assert len(result) == 2
        assert result[0].close == 5475.09
        assert result[1].close == 5481.11

        mock_cache.get_prices.assert_called_once()
        mock_cache.set_prices.assert_called_once()
        mock_requests_get.assert_called_once()

    @patch("src.tools.api._cache")
    @patch("src.tools.api.requests.get")
    @patch("src.tools.api._make_api_request")
    def test_stooq_fallback_partial_coverage_handles_sparse_benchmark(
        self,
        mock_api_request,
        mock_requests_get,
        mock_cache,
    ):
        """Fallback data with gaps should still support downstream metric calculations."""

        mock_cache.get_prices.return_value = None
        mock_cache.set_prices.return_value = None

        api_failure = Mock()
        api_failure.status_code = 404
        api_failure.text = "Not Found"
        mock_api_request.return_value = api_failure

        stooq_response = Mock()
        stooq_response.status_code = 200
        stooq_response.text = "Date,Open,High,Low,Close,Volume\n" "2024-07-02,5471.08,5479.55,5446.53,5475.09,2433991467\n" "2024-07-03,5480.00,5495.10,5468.44,5488.90,2132210441\n"
        mock_requests_get.return_value = stooq_response

        prices = get_prices("^GSPC", "2024-07-01", "2024-07-05")

        assert len(prices) == 2

        calc = PerformanceMetricsCalculator(annual_trading_days=252, annual_rf_rate=0.0)
        portfolio_vals = [
            {
                "Date": datetime(2024, 7, 1),
                "Portfolio Value": 100_000.0,
                "Long Exposure": 0.0,
                "Short Exposure": 0.0,
                "Gross Exposure": 0.0,
                "Net Exposure": 0.0,
                "Long/Short Ratio": float("inf"),
            },
            {
                "Date": datetime(2024, 7, 2),
                "Portfolio Value": 101_500.0,
                "Long Exposure": 0.0,
                "Short Exposure": 0.0,
                "Gross Exposure": 0.0,
                "Net Exposure": 0.0,
                "Long/Short Ratio": float("inf"),
            },
            {
                "Date": datetime(2024, 7, 3),
                "Portfolio Value": 100_800.0,
                "Long Exposure": 0.0,
                "Short Exposure": 0.0,
                "Gross Exposure": 0.0,
                "Net Exposure": 0.0,
                "Long/Short Ratio": float("inf"),
            },
        ]

        closes_by_date: dict[datetime, float] = {}
        for price in prices:
            timestamp = datetime.strptime(price.time, "%Y-%m-%dT%H:%M:%SZ")
            closes_by_date[timestamp.replace(tzinfo=None)] = price.close

        benchmark_values = []
        last_close: float | None = None
        for day in range(1, 4):
            current = datetime(2024, 7, day)
            close = closes_by_date.get(current)
            if close is not None:
                last_close = close
            benchmark_values.append({"Date": current, "Benchmark Value": last_close})

        metrics = calc.compute_metrics(portfolio_vals, benchmark_values)

        assert metrics["sharpe_ratio"] is not None
        assert metrics["information_ratio"] is None

        mock_requests_get.assert_called_once()
        mock_cache.get_prices.assert_called_once()
        mock_cache.set_prices.assert_called_once()

    @patch("src.tools.api.time.sleep")
    @patch("src.tools.api.requests.get")
    def test_max_retries_exceeded(self, mock_get, mock_sleep):
        """Test that function stops retrying after max_retries and returns final 429."""
        # Setup mock responses: all 429s (exceeds max retries)
        mock_429_response = Mock()
        mock_429_response.status_code = 429
        mock_429_response.text = "Too Many Requests"

        mock_get.return_value = mock_429_response

        # Call the function with max_retries=2
        headers = {"X-API-KEY": "test-key"}
        url = "https://api.financialdatasets.ai/test"

        result = _make_api_request(url, headers, max_retries=2)

        # Verify final 429 is returned
        assert result.status_code == 429
        assert result.text == "Too Many Requests"

        # Verify requests.get was called 3 times (1 initial + 2 retries)
        assert mock_get.call_count == 3

        # Verify sleep was called 2 times with linear backoff: 60s, 90s
        assert mock_sleep.call_count == 2
        expected_calls = [call(60), call(90)]
        mock_sleep.assert_has_calls(expected_calls)


if __name__ == "__main__":
    pytest.main([__file__])
