#!/usr/bin/env python3
"""
Test script for HR Assistant API
Tests all major endpoints and validates response format
"""

import json
import asyncio
import sys
from datetime import datetime
from typing import Dict, Any, List
import aiohttp
from colorama import init, Fore, Style

# Initialize colorama for colored output
init()

# Test configuration
BASE_URL = "http://localhost:8000"
API_KEY = "test-api-key-1"
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Test queries
TEST_QUERIES = [
    {
        "name": "Simple leave query",
        "request": {
            "query": "How many days of paid leave do employees get?",
            "n_results": 5
        },
        "validate_confidence": True
    },
    {
        "name": "Japanese query",
        "request": {
            "query": "有給休暇について教えてください",
            "n_results": 3
        },
        "validate_translation": True
    },
    {
        "name": "Benefits query",
        "request": {
            "query": "What health insurance benefits are available?",
            "collection_name": "benefits",
            "n_results": 5
        },
        "validate_sources": True
    },
    {
        "name": "No results query",
        "request": {
            "query": "What is the policy on bringing unicorns to work?",
            "n_results": 5
        },
        "expect_no_results": True
    }
]

class TestResult:
    """Store test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        
    def add_pass(self):
        self.passed += 1
        print(f"{Fore.GREEN}✓{Style.RESET_ALL}", end="")
        
    def add_fail(self, error: str):
        self.failed += 1
        self.errors.append(error)
        print(f"{Fore.RED}✗{Style.RESET_ALL}", end="")
    
    def print_summary(self):
        total = self.passed + self.failed
        print(f"\n\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Test Summary:{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Passed: {self.passed}/{total}{Style.RESET_ALL}")
        print(f"{Fore.RED}Failed: {self.failed}/{total}{Style.RESET_ALL}")
        
        if self.errors:
            print(f"\n{Fore.RED}Errors:{Style.RESET_ALL}")
            for error in self.errors:
                print(f"  - {error}")
        
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        return self.failed == 0

async def test_health_check(session: aiohttp.ClientSession, results: TestResult):
    """Test health check endpoint"""
    print(f"\n{Fore.YELLOW}Testing health check...{Style.RESET_ALL}")
    
    try:
        async with session.get(f"{BASE_URL}/hr/v1/health") as response:
            if response.status == 200:
                data = await response.json()
                if "status" in data:
                    results.add_pass()
                    print(f" Health status: {data['status']}")
                else:
                    results.add_fail("Health check missing status field")
            else:
                results.add_fail(f"Health check returned status {response.status}")
    except Exception as e:
        results.add_fail(f"Health check error: {str(e)}")

async def validate_response_format(response_data: Dict[str, Any]) -> List[str]:
    """Validate HR Assistant response format"""
    errors = []
    
    # Check for required fields
    if "answer_lines" not in response_data:
        errors.append("Missing 'answer_lines' field")
    elif len(response_data["answer_lines"]) != 4:
        errors.append(f"Expected 4 answer lines, got {len(response_data['answer_lines'])}")
    else:
        # Check each line is non-empty
        for i, line in enumerate(response_data["answer_lines"]):
            if not line or len(line.strip()) < 2:
                errors.append(f"Answer line {i+1} is empty or too short")
    
    if "metadata" not in response_data:
        errors.append("Missing 'metadata' field")
    else:
        metadata = response_data["metadata"]
        if "sources" not in metadata:
            errors.append("Missing 'sources' in metadata")
        if "recommendations" not in metadata:
            errors.append("Missing 'recommendations' in metadata")
        elif len(metadata["recommendations"]) > 3:
            errors.append(f"Too many recommendations: {len(metadata['recommendations'])}")
        if "confidence" not in metadata:
            errors.append("Missing 'confidence' in metadata")
        elif not (0.0 <= metadata["confidence"] <= 1.0):
            errors.append(f"Invalid confidence: {metadata['confidence']}")
    
    if "formatted_response" not in response_data:
        errors.append("Missing 'formatted_response' field")
    else:
        # Validate formatted response structure
        lines = response_data["formatted_response"].split('\n')
        if len(lines) < 5:
            errors.append("Formatted response must have at least 5 lines (4 answer + JSON)")
        else:
            # Try to parse JSON part
            try:
                json_part = '\n'.join(lines[4:])
                json.loads(json_part)
            except json.JSONDecodeError:
                errors.append("Invalid JSON in formatted response")
    
    return errors

async def test_query_endpoint(session: aiohttp.ClientSession, results: TestResult):
    """Test main query endpoint"""
    print(f"\n{Fore.YELLOW}Testing query endpoint...{Style.RESET_ALL}")
    
    for test_case in TEST_QUERIES:
        print(f"\n  {test_case['name']}: ", end="")
        
        try:
            async with session.post(
                f"{BASE_URL}/hr/v1/query",
                headers=HEADERS,
                json=test_case["request"]
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Validate format
                    format_errors = await validate_response_format(data)
                    if format_errors:
                        results.add_fail(f"Format errors: {', '.join(format_errors)}")
                        continue
                    
                    # Additional validations
                    metadata = data["metadata"]
                    
                    if test_case.get("validate_confidence"):
                        if metadata["confidence"] < 0.1:
                            results.add_fail("Confidence too low")
                            continue
                    
                    if test_case.get("expect_no_results"):
                        if len(metadata["sources"]) > 0:
                            results.add_fail("Expected no sources")
                            continue
                        if metadata["confidence"] > 0.1:
                            results.add_fail("Expected low confidence for no results")
                            continue
                    
                    if test_case.get("validate_sources"):
                        if len(metadata["sources"]) == 0 and not test_case.get("expect_no_results"):
                            results.add_fail("No sources returned")
                            continue
                        
                        # Validate source structure
                        for source in metadata["sources"]:
                            required = ["doc_id", "title", "page", "original_snippet", 
                                      "translated_snippet", "score"]
                            for field in required:
                                if field not in source:
                                    results.add_fail(f"Source missing {field}")
                                    break
                    
                    results.add_pass()
                    print(f" ✓ (Confidence: {metadata['confidence']:.2f}, Sources: {len(metadata['sources'])})")
                    
                    # Print sample answer
                    print(f"    Answer preview: {data['answer_lines'][0][:60]}...")
                    
                else:
                    error_text = await response.text()
                    results.add_fail(f"Status {response.status}: {error_text[:100]}")
                    
        except Exception as e:
            results.add_fail(f"Query error: {str(e)}")

async def test_collections_endpoint(session: aiohttp.ClientSession, results: TestResult):
    """Test collections listing endpoint"""
    print(f"\n{Fore.YELLOW}Testing collections endpoint...{Style.RESET_ALL}")
    
    try:
        async with session.get(
            f"{BASE_URL}/hr/v1/collections",
            headers=HEADERS
        ) as response:
            if response.status == 200:
                data = await response.json()
                if "collections" in data and "total_documents" in data:
                    results.add_pass()
                    print(f" Found {len(data['collections'])} collections")
                else:
                    results.add_fail("Invalid collections response format")
            else:
                results.add_fail(f"Collections endpoint returned status {response.status}")
    except Exception as e:
        results.add_fail(f"Collections error: {str(e)}")

async def test_rate_limiting(session: aiohttp.ClientSession, results: TestResult):
    """Test rate limiting"""
    print(f"\n{Fore.YELLOW}Testing rate limiting...{Style.RESET_ALL}")
    
    # Make rapid requests to trigger rate limit
    request_count = 0
    rate_limited = False
    
    try:
        for i in range(150):  # Exceeds per-minute limit
            async with session.post(
                f"{BASE_URL}/hr/v1/query",
                headers=HEADERS,
                json={"query": f"Test query {i}", "n_results": 1}
            ) as response:
                request_count += 1
                if response.status == 429:
                    rate_limited = True
                    retry_after = response.headers.get("Retry-After", "unknown")
                    print(f" Rate limited after {request_count} requests (Retry-After: {retry_after}s)")
                    results.add_pass()
                    break
                elif response.status != 200:
                    results.add_fail(f"Unexpected status: {response.status}")
                    break
        
        if not rate_limited:
            results.add_fail(f"Rate limit not triggered after {request_count} requests")
            
    except Exception as e:
        results.add_fail(f"Rate limit test error: {str(e)}")

async def test_mock_data_response():
    """Create a mock response for testing when services aren't running"""
    return {
        "answer_lines": [
            "Employees receive 15 days of paid annual leave after probation.",
            "Leave accrual starts from day one at 1.25 days per month.",
            "Unused leave can be carried forward up to 5 days.",
            "Part-time employees receive prorated leave based on hours."
        ],
        "metadata": {
            "sources": [
                {
                    "doc_id": "HR-POL-001",
                    "title": "Leave Policy 2024",
                    "page": 3,
                    "original_snippet": "Annual leave is 15 days",
                    "translated_snippet": "Annual leave is 15 days",
                    "score": 0.92
                }
            ],
            "recommendations": [
                {"title": "Check leave balance", "reason": "View your current accruals"},
                {"title": "Request time off", "reason": "Submit leave request"}
            ],
            "confidence": 0.85
        },
        "formatted_response": "Employees receive 15 days of paid annual leave after probation.\nLeave accrual starts from day one at 1.25 days per month.\nUnused leave can be carried forward up to 5 days.\nPart-time employees receive prorated leave based on hours.\n{\"sources\":[{\"doc_id\":\"HR-POL-001\",\"title\":\"Leave Policy 2024\",\"page\":3,\"original_snippet\":\"Annual leave is 15 days\",\"translated_snippet\":\"Annual leave is 15 days\",\"score\":0.92}],\"recommendations\":[{\"title\":\"Check leave balance\",\"reason\":\"View your current accruals\"},{\"title\":\"Request time off\",\"reason\":\"Submit leave request\"}],\"confidence\":0.85}"
    }

async def test_mock_mode():
    """Test response validation with mock data"""
    print(f"\n{Fore.YELLOW}Testing with mock data...{Style.RESET_ALL}")
    
    results = TestResult()
    mock_response = await test_mock_data_response()
    
    # Validate mock response format
    errors = await validate_response_format(mock_response)
    if errors:
        print(f"{Fore.RED}Mock validation failed:{Style.RESET_ALL}")
        for error in errors:
            print(f"  - {error}")
        results.add_fail("Mock validation failed")
    else:
        print(f"{Fore.GREEN}Mock response format is valid!{Style.RESET_ALL}")
        results.add_pass()
        
        # Display formatted response
        print(f"\n{Fore.CYAN}Formatted Response:{Style.RESET_ALL}")
        print(mock_response["formatted_response"])
    
    return results.print_summary()

async def main():
    """Main test runner"""
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}HR Assistant API Test Suite{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    # Check if we should run in mock mode
    if len(sys.argv) > 1 and sys.argv[1] == "--mock":
        return await test_mock_mode()
    
    # Create session
    async with aiohttp.ClientSession() as session:
        results = TestResult()
        
        # Run tests
        await test_health_check(session, results)
        await test_query_endpoint(session, results)
        await test_collections_endpoint(session, results)
        
        # Optional: Test rate limiting (comment out if not needed)
        # await test_rate_limiting(session, results)
        
        # Print summary
        return results.print_summary()

if __name__ == "__main__":
    # Run tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
