#!/bin/bash
# Manual curl tests for HR Assistant API

# Configuration
BASE_URL="http://localhost:8000"
API_KEY="test-api-key-1"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}=====================================${NC}"
echo -e "${CYAN}HR Assistant API - Curl Test Commands${NC}"
echo -e "${CYAN}=====================================${NC}"
echo ""

echo -e "${YELLOW}1. Health Check:${NC}"
echo "curl -X GET $BASE_URL/hr/v1/health"
curl -X GET $BASE_URL/hr/v1/health 2>/dev/null | python3 -m json.tool
echo ""

echo -e "${YELLOW}2. Simple Query:${NC}"
echo "curl -X POST $BASE_URL/hr/v1/query \\"
echo "  -H 'X-API-Key: $API_KEY' \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"query\": \"How many days of paid leave?\", \"n_results\": 5}'"
echo ""
echo -e "${GREEN}Response:${NC}"
curl -X POST $BASE_URL/hr/v1/query \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "How many days of paid leave?", "n_results": 5}' \
  2>/dev/null | python3 -m json.tool
echo ""

echo -e "${YELLOW}3. Japanese Query:${NC}"
echo "curl -X POST $BASE_URL/hr/v1/query \\"
echo "  -H 'X-API-Key: $API_KEY' \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"query\": \"有給休暇について\", \"n_results\": 3}'"
echo ""
echo -e "${GREEN}Response:${NC}"
curl -X POST $BASE_URL/hr/v1/query \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "有給休暇について", "n_results": 3}' \
  2>/dev/null | python3 -m json.tool
echo ""

echo -e "${YELLOW}4. List Collections:${NC}"
echo "curl -X GET $BASE_URL/hr/v1/collections \\"
echo "  -H 'X-API-Key: $API_KEY'"
echo ""
echo -e "${GREEN}Response:${NC}"
curl -X GET $BASE_URL/hr/v1/collections \
  -H "X-API-Key: $API_KEY" \
  2>/dev/null | python3 -m json.tool
echo ""

echo -e "${YELLOW}5. Submit Feedback:${NC}"
echo "curl -X POST $BASE_URL/hr/v1/feedback \\"
echo "  -H 'X-API-Key: $API_KEY' \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"query\": \"Test query\", \"rating\": 5, \"was_helpful\": true}'"
echo ""
echo -e "${GREEN}Response:${NC}"
curl -X POST $BASE_URL/hr/v1/feedback \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "Test query", "rating": 5, "was_helpful": true, "feedback_text": "Very helpful!"}' \
  2>/dev/null | python3 -m json.tool
echo ""

echo -e "${YELLOW}6. Test Rate Limiting:${NC}"
echo "for i in {1..10}; do"
echo "  curl -X POST $BASE_URL/hr/v1/query \\"
echo "    -H 'X-API-Key: $API_KEY' \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"query\": \"Test query\"}' \\"
echo "    -w '\nStatus: %{http_code}\n'"
echo "done"
echo ""

echo -e "${CYAN}=====================================${NC}"
echo -e "${CYAN}Formatted Response Example:${NC}"
echo -e "${CYAN}=====================================${NC}"
echo ""
echo "The API returns responses in this exact format:"
echo ""
echo "Line 1: Main answer to the question"
echo "Line 2: Supporting details or conditions"  
echo "Line 3: Additional context or exceptions"
echo "Line 4: Final information or next steps"
echo '{"sources":[...],"recommendations":[...],"confidence":0.85}'
echo ""

echo -e "${GREEN}Test complete!${NC}"
