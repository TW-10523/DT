#!/bin/bash
# Quick test script for HR Assistant

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}================================${NC}"
echo -e "${CYAN}HR Assistant Test Runner${NC}"
echo -e "${CYAN}================================${NC}"

# Parse arguments
MODE="full"
if [ "$1" == "--mock" ]; then
    MODE="mock"
elif [ "$1" == "--quick" ]; then
    MODE="quick"
elif [ "$1" == "--help" ]; then
    echo "Usage: ./test.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --mock    Test with mock data only (no services required)"
    echo "  --quick   Start minimal services and run basic tests"
    echo "  --full    Start all services and run full test suite (default)"
    echo "  --clean   Stop all services and clean up"
    echo ""
    exit 0
elif [ "$1" == "--clean" ]; then
    echo -e "${YELLOW}Cleaning up test environment...${NC}"
    docker-compose -f docker-compose.test.yml down -v
    echo -e "${GREEN}Cleanup complete!${NC}"
    exit 0
fi

# Function to wait for service
wait_for_service() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=1
    
    echo -n "Waiting for $service..."
    while ! nc -z localhost $port 2>/dev/null; do
        if [ $attempt -eq $max_attempts ]; then
            echo -e "${RED}Failed (timeout)${NC}"
            return 1
        fi
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done
    echo -e "${GREEN}Ready${NC}"
    return 0
}

# Mock mode - no services needed
if [ "$MODE" == "mock" ]; then
    echo -e "${YELLOW}Running in mock mode (no services required)${NC}"
    echo ""
    
    # Install dependencies if needed
    if ! python3 -c "import aiohttp" 2>/dev/null; then
        echo -e "${YELLOW}Installing test dependencies...${NC}"
        pip install aiohttp colorama
    fi
    
    # Run mock tests
    python3 backend/test_hr_assistant.py --mock
    exit $?
fi

# Quick mode - minimal services
if [ "$MODE" == "quick" ]; then
    echo -e "${YELLOW}Starting minimal test environment...${NC}"
    echo ""
    
    # Start test services
    docker-compose -f docker-compose.test.yml up -d postgres-test redis-test mock-llm
    
    # Wait for services
    wait_for_service "PostgreSQL" 5432
    wait_for_service "Redis" 6379
    wait_for_service "Mock LLM" 8080
    
    echo ""
    echo -e "${GREEN}Test environment ready!${NC}"
    echo ""
    
    # Create test database schema
    echo -e "${YELLOW}Setting up database...${NC}"
    docker exec hr_postgres_test psql -U hruser -d hrdb_test -c "
        CREATE EXTENSION IF NOT EXISTS vector;
        
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            doc_id VARCHAR(255) UNIQUE NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            page INTEGER DEFAULT 0,
            collection VARCHAR(100),
            embedding vector(384),
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_embedding 
        ON documents USING ivfflat (embedding vector_cosine_ops) 
        WITH (lists = 10);
        
        -- Insert sample data
        INSERT INTO documents (doc_id, title, content, page, collection) 
        VALUES 
        ('HR-POL-001', 'Leave Policy 2024', 'Employees receive 15 days of paid annual leave after probation period. Leave accrual starts from the first day at 1.25 days per month.', 3, 'hr_policies'),
        ('HR-BEN-001', 'Health Benefits', 'Comprehensive health insurance coverage including medical, dental, and vision benefits for all full-time employees.', 1, 'benefits')
        ON CONFLICT (doc_id) DO NOTHING;
    " 2>/dev/null || true
    
    echo -e "${GREEN}Database ready!${NC}"
    echo ""
fi

# Start the API server
echo -e "${YELLOW}Starting HR Assistant API...${NC}"

# Check if virtual environment exists
if [ -d "backend/venv" ]; then
    source backend/venv/bin/activate
fi

# Install dependencies if needed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}Installing API dependencies...${NC}"
    pip install fastapi uvicorn sqlalchemy asyncpg redis aiohttp colorama pydantic pydantic-settings python-jose bleach tenacity circuitbreaker prometheus-client
fi

# Start API in background
cd backend
export PYTHONPATH=$PWD:$PWD/app
ENV_FILE=".env.test"
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}Error: $ENV_FILE not found${NC}"
    exit 1
fi

echo -e "${YELLOW}Starting API server...${NC}"
python3 -c "
import os
os.environ['ENVIRONMENT'] = 'development'
from dotenv import load_dotenv
load_dotenv('.env.test')
import uvicorn
from app.main import app
uvicorn.run(app, host='0.0.0.0', port=8000, log_level='error')
" &
API_PID=$!
cd ..

# Wait for API to start
wait_for_service "API" 8000

echo ""
echo -e "${GREEN}API server started (PID: $API_PID)${NC}"
echo ""

# Run tests
echo -e "${CYAN}Running test suite...${NC}"
echo ""

python3 backend/test_hr_assistant.py
TEST_RESULT=$?

# Cleanup
echo ""
echo -e "${YELLOW}Stopping API server...${NC}"
kill $API_PID 2>/dev/null || true

if [ "$MODE" == "quick" ]; then
    echo -e "${YELLOW}Stopping test services...${NC}"
    docker-compose -f docker-compose.test.yml down
fi

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
else
    echo -e "${RED}❌ Some tests failed. Check the output above.${NC}"
fi

exit $TEST_RESULT




TRY THIS 
https://stackblitz.com/edit/nuxt-starter-zqs6pfb4?file=README.md
