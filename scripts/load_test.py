"""
Load and stress testing script for the Product Recommendation System

This script uses Apache Bench (ab) or Python requests to simulate load.

Usage:
    python scripts/load_test.py --host http://localhost:8000 --users 100 --duration 60
"""

import argparse
import json
import logging
import time
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_test_with_requests(
    base_url: str,
    num_users: int = 10,
    requests_per_user: int = 10,
    concurrent: bool = False,
):
    """
    TODO: ====Implement load testing with requests library====

    Parameters:
    - base_url: Base URL of the API
    - num_users: Number of simulated users
    - requests_per_user: Requests per user
    - concurrent: Use concurrent requests

    Measures:
    - Response times
    - Success/failure rates
    - Throughput (requests/sec)
    - Error rates
    """

    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        import requests
    except ImportError:
        logger.error("Please install: pip install requests")
        return

    logger.info(
        f"Starting load test: {num_users} users, {requests_per_user} requests each"
    )
    logger.info(f"Target: {base_url}/recommendations")

    results: Dict[str, List] = {
        "response_times": [],
        "success_count": 0,
        "error_count": 0,
        "errors": [],
    }

    def make_request(user_id: int):
        """Make a single recommendation request"""
        try:
            payload = {
                "user_id": f"user_{user_id}",
                "num_recommendations": 5,
                "recommendation_type": "collaborative",
            }

            start_time = time.time()
            response = requests.post(
                f"{base_url}/recommendations", json=payload, timeout=10
            )
            response_time = time.time() - start_time

            results["response_times"].append(response_time)

            if response.status_code == 200:
                results["success_count"] += 1
                logger.debug(f"User {user_id}: OK ({response_time:.3f}s)")
            else:
                results["error_count"] += 1
                results["errors"].append(f"Status {response.status_code}")
                logger.warning(f"User {user_id}: Status {response.status_code}")

        except Exception as e:
            results["error_count"] += 1
            results["errors"].append(str(e))
            logger.error(f"User {user_id}: {str(e)}")

    # Run load test
    start_time = time.time()

    if concurrent:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for user_id in range(num_users):
                for _ in range(requests_per_user):
                    futures.append(executor.submit(make_request, user_id))

            for future in as_completed(futures):
                future.result()
    else:
        for user_id in range(num_users):
            for _ in range(requests_per_user):
                make_request(user_id)

    total_time = time.time() - start_time

    # Print results
    print_load_test_results(results, total_time)


def load_test_with_locust():
    """
    TODO: ====Implement load testing with Locust====

    Locust provides:
    - Web UI for test monitoring
    - Spawning rate control
    - Real-time statistics
    - Response time distribution
    """

    logger.info("For Locust load testing:")
    logger.info("1. Install: pip install locust")
    logger.info("2. Create locustfile.py")
    logger.info("3. Run: locust -f locustfile.py --host http://localhost:8000")


def print_load_test_results(results: Dict, total_time: float):
    """Print load test results summary"""

    print("\n" + "=" * 70)
    print("LOAD TEST RESULTS")
    print("=" * 70)

    total_requests = results["success_count"] + results["error_count"]

    print(f"Total Requests:        {total_requests}")
    print(f"Successful:            {results['success_count']}")
    print(f"Failed:                {results['error_count']}")
    print(
        f"Success Rate:          {(results['success_count']/total_requests*100):.2f}%"
    )
    print(f"Total Time:            {total_time:.2f}s")
    print(f"Throughput:            {(total_requests/total_time):.2f} req/s")

    if results["response_times"]:
        response_times = results["response_times"]
        min_time = min(response_times)
        max_time = max(response_times)
        avg_time = sum(response_times) / len(response_times)

        print(f"\nResponse Times:")
        print(f"  Min:                 {min_time:.3f}s")
        print(f"  Max:                 {max_time:.3f}s")
        print(f"  Average:             {avg_time:.3f}s")

        # Calculate percentiles
        sorted_times = sorted(response_times)
        p50_idx = int(len(sorted_times) * 0.50)
        p95_idx = int(len(sorted_times) * 0.95)
        p99_idx = int(len(sorted_times) * 0.99)

        print(f"  P50 (Median):        {sorted_times[p50_idx]:.3f}s")
        print(f"  P95:                 {sorted_times[p95_idx]:.3f}s")
        print(f"  P99:                 {sorted_times[p99_idx]:.3f}s")

    if results["errors"]:
        print(f"\nErrors: {len(results['errors'])}")
        for error in results["errors"][:5]:  # Show first 5
            print(f"  - {error}")

    print("=" * 70 + "\n")


# TODO: ====Add Apache Bench integration====
def load_test_with_ab(base_url: str, num_requests: int = 100, concurrency: int = 10):
    """
    TODO: ====Use Apache Bench for load testing====

    Apache Bench provides:
    - C-level performance (compiled)
    - Simple concurrent testing
    - Detailed statistics

    Installation:
    - Ubuntu/Debian: sudo apt-get install apache2-utils
    - macOS: brew install httpd
    - Windows: Download from Apache official site
    """
    pass


# TODO: ====Add JMeter integration====
def load_test_with_jmeter():
    """
    TODO: ====Use Apache JMeter for advanced load testing====

    JMeter provides:
    - GUI for test planning
    - Complex load patterns
    - Multiple protocols support
    - Advanced assertions
    - Detailed HTML report generation
    """
    pass


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Load test for Product Recommendation System"
    )

    parser.add_argument(
        "--host",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--users", type=int, default=10, help="Number of simulated users (default: 10)"
    )
    parser.add_argument(
        "--requests", type=int, default=10, help="Requests per user (default: 10)"
    )
    parser.add_argument(
        "--concurrent", action="store_true", help="Use concurrent requests"
    )
    parser.add_argument(
        "--tool",
        choices=["requests", "locust", "ab", "jmeter"],
        default="requests",
        help="Load testing tool to use (default: requests)",
    )

    args = parser.parse_args()

    if args.tool == "requests":
        load_test_with_requests(
            args.host,
            num_users=args.users,
            requests_per_user=args.requests,
            concurrent=args.concurrent,
        )
    elif args.tool == "locust":
        load_test_with_locust()
    elif args.tool == "ab":
        load_test_with_ab(args.host, num_requests=args.users * args.requests)
    elif args.tool == "jmeter":
        load_test_with_jmeter()


if __name__ == "__main__":
    main()
