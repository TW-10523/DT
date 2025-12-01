"""
Example usage of the HR Assistant API
Demonstrates the strict format compliance with the system prompt
"""

import json
import requests
from typing import Dict, Any

class HRAssistantClient:
    """Client for interacting with HR Assistant API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
    
    def query_hr(self, question: str, collection: str = None) -> Dict[str, Any]:
        """
        Submit a query to the HR assistant
        
        Args:
            question: HR-related question
            collection: Optional collection to search
            
        Returns:
            Parsed response with answer and metadata
        """
        endpoint = f"{self.base_url}/hr/query"
        payload = {
            "query": question,
            "collection_name": collection,
            "n_results": 5
        }
        
        response = requests.post(endpoint, json=payload, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    def parse_formatted_response(self, formatted_response: str) -> Dict[str, Any]:
        """
        Parse the formatted response according to system specifications
        
        Args:
            formatted_response: Raw formatted response string
            
        Returns:
            Dictionary with answer lines and metadata
        """
        lines = formatted_response.split('\n')
        
        # First 4 lines are the answer
        answer_lines = lines[:4]
        
        # JSON metadata starts after the 4th newline
        json_str = '\n'.join(lines[4:]) if len(lines) > 4 else '{}'
        
        try:
            metadata = json.loads(json_str)
        except json.JSONDecodeError:
            metadata = {"error": "Failed to parse metadata"}
        
        return {
            "answer": answer_lines,
            "metadata": metadata
        }


def example_queries():
    """Demonstrate various HR queries and their expected formats"""
    
    # Example queries that would trigger the HR Assistant
    example_questions = [
        {
            "query": "How many days of paid leave do I get per year?",
            "expected_format": """
Employees receive 15 days of paid annual leave after probation period.
Leave accrual starts from the first day of employment at 1.25 days per month.
Unused leave can be carried forward up to 5 days to the next year.
Part-time employees receive prorated leave based on their working hours.
{"sources":[{"doc_id":"HR-POL-001","title":"Leave Policy 2024","page":3,"original_snippet":"Annual leave entitlement is 15 days","translated_snippet":"Annual leave entitlement is 15 days","score":0.92}],"recommendations":[{"title":"Check holiday calendar","reason":"View company holidays and blackout dates"},{"title":"Review leave balance","reason":"See your current leave accrual and usage"}],"confidence":0.85}"""
        },
        {
            "query": "産休について教えてください", # Japanese: Tell me about maternity leave
            "expected_format": """
Maternity leave is available for 14 weeks with full pay for eligible employees.
Coverage begins 6 weeks before expected delivery and extends 8 weeks after.
Additional unpaid leave up to 12 months is available upon request.
Employees must notify HR at least 3 months before the expected leave date.
{"sources":[{"doc_id":"HR-BEN-005","title":"Maternity Benefits","page":1,"original_snippet":"産休は出産予定日の6週間前から開始","translated_snippet":"[Translated from ja] Maternity leave starts 6 weeks before expected delivery","score":0.88}],"recommendations":[{"title":"View parental benefits","reason":"Complete overview of family-related benefits"},{"title":"Contact HR Team","reason":"Discuss your specific situation and requirements"}],"confidence":0.90}"""
        },
        {
            "query": "What is the work from home policy?",
            "expected_format": """
Employees can work from home up to 2 days per week after probation.
Remote work requires manager approval and must be scheduled in advance.
Core hours of 10 AM to 3 PM local time must be maintained for collaboration.
Company provides a monthly stipend of $50 for home office expenses.
{"sources":[{"doc_id":"HR-POL-012","title":"Remote Work Policy","page":2,"original_snippet":"Hybrid work model allows 2 days WFH per week","translated_snippet":"Hybrid work model allows 2 days WFH per week","score":0.95}],"recommendations":[{"title":"Request WFH approval","reason":"Submit formal work from home request"},{"title":"View IT policies","reason":"Check remote access and security requirements"}],"confidence":0.93}"""
        },
        {
            "query": "Retirement benefits eligibility",
            "expected_format": """
Employees are eligible for retirement benefits after 5 years of service.
Company matches 401(k) contributions up to 6% of base salary.
Vesting schedule is 20% per year, fully vested after 5 years.
Early retirement options available at age 55 with reduced benefits.
{"sources":[{"doc_id":"HR-BEN-008","title":"Retirement Plan Guide","page":5,"original_snippet":"401(k) matching up to 6% after 5 years","translated_snippet":"401(k) matching up to 6% after 5 years","score":0.87}],"recommendations":[{"title":"Calculate retirement savings","reason":"Use retirement planning calculator"},{"title":"Schedule benefits consultation","reason":"Discuss your retirement planning options"}],"confidence":0.82}"""
        }
    ]
    
    return example_questions


def demonstrate_strict_format_compliance():
    """
    Demonstrate that the system strictly follows the format rules:
    - Exactly 4 lines of English text
    - Followed immediately by JSON metadata
    - No markdown, no extra commentary
    - Temperature 0.0 for deterministic responses
    """
    
    print("HR ASSISTANT FORMAT COMPLIANCE DEMONSTRATION")
    print("=" * 50)
    
    # Show format structure
    format_template = """
Line 1: Main answer to the question with key information.
Line 2: Supporting details or conditions that apply.
Line 3: Additional context or exceptions to be aware of.
Line 4: Final relevant information or next steps.
{"sources":[...],"recommendations":[...],"confidence":0.00}
"""
    
    print("REQUIRED OUTPUT FORMAT:")
    print(format_template)
    print()
    
    # Demonstrate validation
    print("VALIDATION RULES:")
    print("✓ Must have exactly 4 lines of meaningful text")
    print("✓ Each line must contain actual content (not single words)")
    print("✓ Must be followed by valid JSON on line 5+")
    print("✓ JSON must contain: sources, recommendations, confidence")
    print("✓ Sources include original_snippet and translated_snippet")
    print("✓ Maximum 3 recommendations")
    print("✓ Confidence between 0.0 and 1.0")
    print()
    
    # Show conflict handling
    print("CONFLICT HANDLING:")
    conflict_example = """
Multiple documents show different leave entitlements ranging from 10-15 days.
The most recent HR policy (2024) indicates 15 days for full-time employees.
Older documents may reflect previous policies no longer in effect.
Please verify with HR for your specific employment contract terms.
{"sources":[{"doc_id":"HR-001","title":"Leave Policy 2024","page":3,"original_snippet":"15 days annual leave","translated_snippet":"15 days annual leave","score":0.92},{"doc_id":"HR-002","title":"Employee Handbook 2023","page":8,"original_snippet":"10 days annual leave","translated_snippet":"10 days annual leave","score":0.75}],"recommendations":[{"title":"Contact HR","reason":"Clarify which policy applies to you"}],"confidence":0.65}"""
    
    print("When sources conflict, confidence is lowered (0.4-0.7):")
    print(conflict_example)
    print()
    
    # Show no results handling
    print("NO RESULTS HANDLING:")
    no_results_example = """
No authoritative answer found in the docs.
The query did not match any relevant HR documentation.
Please try rephrasing your question or contact HR directly.
You may also browse the HR portal for general information.
{"sources":[],"recommendations":[{"title":"Browse HR Portal","reason":"Access all HR resources"},{"title":"Contact HR Team","reason":"Get direct assistance"}],"confidence":0.0}"""
    
    print("When no relevant documents found:")
    print(no_results_example)


def validate_response_format(response: str) -> bool:
    """
    Validate that a response meets the strict format requirements
    
    Args:
        response: The formatted response string
        
    Returns:
        True if valid, False otherwise
    """
    lines = response.strip().split('\n')
    
    # Check for exactly 4 text lines + JSON
    if len(lines) < 5:
        print(f"❌ Format error: Expected at least 5 lines, got {len(lines)}")
        return False
    
    # Check that first 4 lines are not empty
    for i in range(4):
        if not lines[i] or len(lines[i].strip()) < 2:
            print(f"❌ Format error: Line {i+1} is empty or too short")
            return False
    
    # Check JSON validity
    json_part = '\n'.join(lines[4:])
    try:
        metadata = json.loads(json_part)
        
        # Validate required fields
        required_fields = ['sources', 'recommendations', 'confidence']
        for field in required_fields:
            if field not in metadata:
                print(f"❌ JSON error: Missing required field '{field}'")
                return False
        
        # Validate confidence range
        if not (0.0 <= metadata['confidence'] <= 1.0):
            print(f"❌ Validation error: Confidence {metadata['confidence']} out of range")
            return False
        
        # Validate max 3 recommendations
        if len(metadata['recommendations']) > 3:
            print(f"❌ Validation error: Too many recommendations ({len(metadata['recommendations'])})")
            return False
        
        # Validate source structure
        for source in metadata['sources']:
            required_source_fields = ['doc_id', 'title', 'page', 'original_snippet', 
                                     'translated_snippet', 'score']
            for field in required_source_fields:
                if field not in source:
                    print(f"❌ Source error: Missing field '{field}'")
                    return False
        
        print("✅ Response format is valid!")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        return False


if __name__ == "__main__":
    # Run demonstration
    demonstrate_strict_format_compliance()
    
    print("\n" + "=" * 50)
    print("EXAMPLE QUERIES AND RESPONSES:")
    print("=" * 50)
    
    for example in example_queries():
        print(f"\nQUERY: {example['query']}")
        print("-" * 30)
        print("EXPECTED RESPONSE:")
        print(example['expected_format'])
        print("\nVALIDATION:")
        validate_response_format(example['expected_format'])
        print()
