"""
HR Assistant Service - Retrieval Augmented Generation for HR queries
Provides structured, precise answers based on document retrieval with multilingual support
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from sqlalchemy.orm import Session
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class Source:
    """Represents a source document with metadata"""
    doc_id: str
    title: str
    page: int
    original_snippet: str
    translated_snippet: str
    score: float
    
    def to_dict(self):
        return asdict(self)

@dataclass
class Recommendation:
    """Represents a recommendation for related queries"""
    title: str
    reason: str
    
    def to_dict(self):
        return {"title": self.title, "reason": self.reason}

class HRAssistantService:
    """
    Expert HR Assistant with retrieval-augmented generation capabilities
    Follows strict formatting rules and provides structured responses
    """
    
    def __init__(self, retriever_service, translator_service, llm_service):
        """
        Initialize the HR Assistant
        
        Args:
            retriever_service: Service for vector database retrieval
            translator_service: Service for language translation
            llm_service: Service for LLM inference
        """
        self.retriever = retriever_service
        self.translator = translator_service
        self.llm = llm_service
        
    def process_query(self, 
                     query: str, 
                     collection_name: Optional[str] = None,
                     n_results: int = 5,
                     db: Optional[Session] = None) -> Dict[str, Any]:
        """
        Process an HR query through the full RAG pipeline
        
        Args:
            query: User's question
            collection_name: Optional collection to search in
            n_results: Number of passages to retrieve
            db: Database session
            
        Returns:
            Formatted response with 4-line answer and JSON metadata
        """
        try:
            # Step 1: Retrieve relevant passages from vector DB
            retrieved_passages = self._retrieve_passages(
                query, collection_name, n_results, db
            )
            
            if not retrieved_passages:
                return self._create_no_results_response()
            
            # Step 2: Process and translate passages if needed
            processed_passages = self._process_passages(retrieved_passages)
            
            # Step 3: Generate structured answer using LLM
            answer_lines, sources, recommendations, confidence = self._generate_answer(
                query, processed_passages
            )
            
            # Step 4: Format final response
            return self._format_response(
                answer_lines, sources, recommendations, confidence
            )
            
        except Exception as e:
            logger.error(f"Error processing HR query: {str(e)}")
            return self._create_error_response(str(e))
    
    def _retrieve_passages(self, 
                          query: str, 
                          collection_name: Optional[str],
                          n_results: int,
                          db: Optional[Session]) -> List[Dict[str, Any]]:
        """
        Retrieve relevant passages from vector database
        
        Args:
            query: Search query
            collection_name: Collection to search
            n_results: Number of results
            db: Database session
            
        Returns:
            List of retrieved passages with metadata
        """
        try:
            results = self.retriever.search(
                query=query,
                collection_name=collection_name,
                n_results=n_results,
                db=db
            )
            return results
        except Exception as e:
            logger.error(f"Retrieval error: {str(e)}")
            return []
    
    def _process_passages(self, passages: List[Dict[str, Any]]) -> List[Source]:
        """
        Process retrieved passages, detect language and translate if needed
        
        Args:
            passages: Raw retrieved passages
            
        Returns:
            List of processed Source objects
        """
        processed_sources = []
        
        for passage in passages:
            # Detect language of the passage
            lang = self._detect_language(passage.get('text', ''))
            
            # Translate if not English
            if lang != 'en':
                translated_text = self.translator.translate(
                    passage.get('text', ''),
                    source_lang=lang,
                    target_lang='en'
                )
            else:
                translated_text = passage.get('text', '')
            
            source = Source(
                doc_id=str(passage.get('doc_id', '')),
                title=passage.get('title', 'Untitled'),
                page=passage.get('page', 0),
                original_snippet=passage.get('text', ''),
                translated_snippet=translated_text,
                score=float(passage.get('score', 0.0))
            )
            processed_sources.append(source)
        
        return processed_sources
    
    def _detect_language(self, text: str) -> str:
        """
        Detect the language of a text snippet
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (e.g., 'en', 'ja')
        """
        # Simple heuristic - check for Japanese characters
        if any('\u3040' <= char <= '\u309F' or  # Hiragana
               '\u30A0' <= char <= '\u30FF' or  # Katakana
               '\u4E00' <= char <= '\u9FFF'     # Kanji
               for char in text):
            return 'ja'
        
        # Default to English
        return 'en'
    
    def _generate_answer(self, 
                        query: str, 
                        sources: List[Source]) -> Tuple[List[str], List[Source], List[Recommendation], float]:
        """
        Generate structured answer using LLM with retrieved context
        
        Args:
            query: User question
            sources: Processed source documents
            
        Returns:
            Tuple of (answer_lines, sources, recommendations, confidence)
        """
        # Prepare context for LLM
        context = self._prepare_context(sources)
        
        # Create prompt for LLM
        prompt = self._create_prompt(query, context)
        
        # Get LLM response (temperature=0 for deterministic)
        llm_response = self.llm.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=500
        )
        
        # Parse LLM response into structured components
        return self._parse_llm_response(llm_response, sources)
    
    def _prepare_context(self, sources: List[Source]) -> str:
        """
        Prepare context from sources for LLM
        
        Args:
            sources: List of source documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, source in enumerate(sources, 1):
            context_parts.append(
                f"[Source {i}] (Doc: {source.doc_id}, Page: {source.page}, Score: {source.score:.2f})\n"
                f"{source.translated_snippet}\n"
            )
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create prompt for LLM following the system instructions
        
        Args:
            query: User question
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        return f"""You are an expert HR assistant. Answer based ONLY on the provided sources.

RETRIEVED SOURCES:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Use ONLY information from the sources above. Do not hallucinate.
2. If sources conflict, mention this and be conservative.
3. Provide exactly 4 lines of meaningful English text as the answer.
4. Include specific evidence (counts, durations, conditions) when available.
5. If no source supports the answer, say "No authoritative answer found in the docs."

ANSWER (exactly 4 lines):"""
    
    def _parse_llm_response(self, 
                           llm_text: str, 
                           sources: List[Source]) -> Tuple[List[str], List[Source], List[Recommendation], float]:
        """
        Parse LLM response into structured components
        
        Args:
            llm_text: Raw LLM response
            sources: Original sources for reference
            
        Returns:
            Tuple of parsed components
        """
        # Split response into lines
        lines = llm_text.strip().split('\n')
        
        # Extract 4-line answer
        answer_lines = lines[:4] if len(lines) >= 4 else lines + [''] * (4 - len(lines))
        answer_lines = answer_lines[:4]  # Ensure exactly 4
        
        # Calculate confidence based on source scores and answer quality
        confidence = self._calculate_confidence(sources, answer_lines)
        
        # Generate recommendations based on query and sources
        recommendations = self._generate_recommendations(answer_lines[0] if answer_lines else "", sources)
        
        # Filter sources to only those actually used
        used_sources = self._filter_used_sources(sources, answer_lines)
        
        return answer_lines, used_sources, recommendations, confidence
    
    def _calculate_confidence(self, sources: List[Source], answer_lines: List[str]) -> float:
        """
        Calculate confidence score based on evidence quality
        
        Args:
            sources: Available sources
            answer_lines: Generated answer
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not sources:
            return 0.0
        
        # Base confidence on average source score
        avg_score = sum(s.score for s in sources) / len(sources)
        
        # Check for conflicting information indicator
        answer_text = ' '.join(answer_lines).lower()
        if 'conflict' in answer_text or 'disagree' in answer_text:
            avg_score *= 0.7  # Reduce confidence for conflicts
        
        # Check if answer indicates no information found
        if 'no authoritative answer' in answer_text.lower():
            return 0.1
        
        return min(max(avg_score, 0.0), 1.0)
    
    def _generate_recommendations(self, query: str, sources: List[Source]) -> List[Recommendation]:
        """
        Generate related query recommendations
        
        Args:
            query: Original query
            sources: Retrieved sources
            
        Returns:
            List of up to 3 recommendations
        """
        recommendations = []
        
        # Analyze query keywords
        query_lower = query.lower()
        
        # Generate contextual recommendations
        if 'leave' in query_lower or 'vacation' in query_lower:
            recommendations.append(
                Recommendation(
                    title="Check holiday calendar",
                    reason="View company holidays and blackout dates"
                )
            )
            recommendations.append(
                Recommendation(
                    title="Review leave balance",
                    reason="See your current leave accrual and usage"
                )
            )
        
        if 'benefit' in query_lower or 'insurance' in query_lower:
            recommendations.append(
                Recommendation(
                    title="Compare benefit plans",
                    reason="View detailed plan comparisons and costs"
                )
            )
        
        if 'policy' in query_lower:
            recommendations.append(
                Recommendation(
                    title="View all HR policies",
                    reason="Browse complete policy documentation"
                )
            )
        
        # Limit to 3 recommendations
        return recommendations[:3]
    
    def _filter_used_sources(self, sources: List[Source], answer_lines: List[str]) -> List[Source]:
        """
        Filter sources to only those referenced in the answer
        
        Args:
            sources: All retrieved sources
            answer_lines: Generated answer lines
            
        Returns:
            List of sources actually used
        """
        # For now, return top 3 sources by score
        # In production, would analyze answer to determine actual usage
        return sorted(sources, key=lambda s: s.score, reverse=True)[:3]
    
    def _format_response(self, 
                        answer_lines: List[str],
                        sources: List[Source],
                        recommendations: List[Recommendation],
                        confidence: float) -> Dict[str, Any]:
        """
        Format the final response according to specifications
        
        Args:
            answer_lines: 4-line answer
            sources: Source documents
            recommendations: Related recommendations
            confidence: Confidence score
            
        Returns:
            Formatted response dictionary
        """
        # Create the text response (4 lines)
        text_response = '\n'.join(answer_lines)
        
        # Create JSON metadata
        metadata = {
            "sources": [s.to_dict() for s in sources],
            "recommendations": [r.to_dict() for r in recommendations],
            "confidence": round(confidence, 2)
        }
        
        # Combine as specified in format rules
        formatted_output = f"{text_response}\n{json.dumps(metadata, separators=(',', ':'))}"
        
        return {
            "formatted_response": formatted_output,
            "answer_lines": answer_lines,
            "metadata": metadata
        }
    
    def _create_no_results_response(self) -> Dict[str, Any]:
        """
        Create response when no results are found
        
        Returns:
            Formatted no-results response
        """
        answer_lines = [
            "No authoritative answer found in the docs.",
            "The query did not match any relevant HR documentation.",
            "Please try rephrasing your question or contact HR directly.",
            "You may also browse the HR portal for general information."
        ]
        
        metadata = {
            "sources": [],
            "recommendations": [
                {"title": "Browse HR Portal", "reason": "Access all HR resources and documentation"},
                {"title": "Contact HR Team", "reason": "Get direct assistance from HR representatives"}
            ],
            "confidence": 0.0
        }
        
        formatted_output = f"{chr(10).join(answer_lines)}\n{json.dumps(metadata, separators=(',', ':'))}"
        
        return {
            "formatted_response": formatted_output,
            "answer_lines": answer_lines,
            "metadata": metadata
        }
    
    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        """
        Create response for error conditions
        
        Args:
            error_msg: Error message
            
        Returns:
            Formatted error response
        """
        answer_lines = [
            "An error occurred processing your request.",
            f"Error details: {error_msg[:100]}",
            "Please try again or contact support if the issue persists.",
            "Your query has been logged for review."
        ]
        
        metadata = {
            "sources": [],
            "recommendations": [
                {"title": "Retry Query", "reason": "Try submitting your question again"},
                {"title": "Contact Support", "reason": "Get help with technical issues"}
            ],
            "confidence": 0.0
        }
        
        formatted_output = f"{chr(10).join(answer_lines)}\n{json.dumps(metadata, separators=(',', ':'))}"
        
        return {
            "formatted_response": formatted_output,
            "answer_lines": answer_lines,
            "metadata": metadata,
            "error": error_msg
        }
