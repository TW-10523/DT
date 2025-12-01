-- Initialize HR Assistant Database
-- This script creates the necessary tables and sample data

-- Create pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create document collections table
CREATE TABLE IF NOT EXISTS document_collections (
    id SERIAL PRIMARY KEY,
    collection_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create documents table with vector support
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(255) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    page INTEGER DEFAULT 0,
    collection VARCHAR(100) REFERENCES document_collections(collection_name) ON DELETE CASCADE,
    language VARCHAR(10) DEFAULT 'en',
    embedding vector(384),  -- Using smaller dimension for testing
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS idx_documents_embedding 
ON documents USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 10);

-- Create feedback table
CREATE TABLE IF NOT EXISTS hr_feedback (
    feedback_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255),
    query TEXT NOT NULL,
    response_id VARCHAR(255),
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    feedback_text TEXT,
    was_helpful BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Insert sample collections
INSERT INTO document_collections (collection_name, description) VALUES
    ('hr_policies', 'Company HR policies and procedures'),
    ('benefits', 'Employee benefits documentation'),
    ('leave_policies', 'Leave and time-off policies')
ON CONFLICT (collection_name) DO NOTHING;

-- Insert sample documents
INSERT INTO documents (doc_id, title, content, page, collection, language) VALUES
    ('HR-POL-001', 
     'Leave Policy 2024', 
     'Employees receive 15 days of paid annual leave after probation period. Leave accrual starts from the first day of employment at 1.25 days per month. Unused leave can be carried forward up to 5 days to the next year. Part-time employees receive prorated leave based on their working hours.',
     3, 
     'leave_policies',
     'en'),
    
    ('HR-BEN-001', 
     'Health Benefits Guide', 
     'Comprehensive health insurance coverage including medical, dental, and vision benefits for all full-time employees. Coverage begins on the first day of employment. Family members can be added with additional premium contributions. Annual wellness benefits include gym membership reimbursement up to $500.',
     1, 
     'benefits',
     'en'),
    
    ('HR-POL-002', 
     'Remote Work Policy', 
     'Employees can work from home up to 2 days per week with manager approval. Remote work requires completion of home office setup checklist and signing of remote work agreement. Core hours of 10 AM to 3 PM must be maintained for team collaboration.',
     2, 
     'hr_policies',
     'en'),
    
    ('HR-POL-003', 
     'Code of Conduct', 
     'All employees must maintain professional behavior and adhere to company values. Discrimination, harassment, and retaliation are strictly prohibited. Violations will result in disciplinary action up to and including termination.',
     1, 
     'hr_policies',
     'en'),
     
    ('HR-BEN-002',
     '退職金制度', -- Retirement Benefits in Japanese
     '勤続5年以上の従業員は退職金制度の対象となります。会社は基本給の6%まで401(k)拠出金をマッチングします。権利確定スケジュールは年間20%で、5年後に完全に権利確定されます。',
     5,
     'benefits',
     'ja')
ON CONFLICT (doc_id) DO NOTHING;

-- Create a simple function to generate random vectors for testing
CREATE OR REPLACE FUNCTION random_vector(dim INTEGER)
RETURNS vector AS $$
DECLARE
    vec float4[];
BEGIN
    -- Generate random values
    FOR i IN 1..dim LOOP
        vec[i] := random() * 2 - 1;  -- Random between -1 and 1
    END LOOP;
    RETURN vec::vector;
END;
$$ LANGUAGE plpgsql;

-- Add random embeddings to documents for testing
UPDATE documents 
SET embedding = random_vector(384)
WHERE embedding IS NULL;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO hruser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO hruser;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO hruser;
