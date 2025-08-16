"""
OpenAI service for document analysis using o3 model.
"""

import base64
import logging
from openai import AsyncOpenAI

from ..config import settings
from ..utils import load_system_prompt

logger = logging.getLogger(__name__)


class OpenAIService:
    """Service for interacting with OpenAI API."""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.system_prompt = load_system_prompt('document_analysis_system_prompt.md')
        logger.info("OpenAI service initialized")
    
    async def analyze_document(
        self, 
        document_id: str, 
        image_data: bytes, 
        doc_metadata: dict,
        analysis_type: str, 
        case_context: str, 
        workflow_id: str = None
    ) -> dict:
        """Analyze document with OpenAI o3"""
        
        # Encode image as base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Create analysis prompt
        analysis_prompt = f"""Analyze this financial document image for family law discovery purposes.

Document Details:
- Document ID: {document_id}
- Filename: {doc_metadata.get('filename', 'Unknown')}
- File Size: {doc_metadata.get('file_size', 0)} bytes

Case Context: {case_context}
Analysis Type: {analysis_type}

Please extract and analyze:
1. All financial figures, amounts, dates, and account information
2. Any patterns, trends, or anomalies in the financial data
3. Potential red flags or concerning information
4. Overall confidence in the document authenticity and completeness

Provide a comprehensive analysis summary in text format."""

        try:
            logger.info(f"🧠 Sending document to o3 for analysis: {document_id}")
            
            response = await self.client.chat.completions.create(
                model="o3",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": analysis_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_completion_tokens=8000
            )
            
            # Extract response data
            analysis_content = response.choices[0].message.content
            usage = response.usage
            
            # Extract token count - o3 returns detailed token usage
            tokens_used = None
            if usage:
                if hasattr(usage, 'total_tokens'):
                    tokens_used = usage.total_tokens
                elif hasattr(usage, 'model_dump'):
                    # Fallback to model_dump if direct attribute access fails
                    usage_dict = usage.model_dump()
                    tokens_used = usage_dict.get('total_tokens')
            
            logger.info(f"📊 Token usage details: completion={getattr(usage, 'completion_tokens', 'N/A')}, prompt={getattr(usage, 'prompt_tokens', 'N/A')}, total={tokens_used}")
            logger.info(f"🧠 o3 analysis completed for {document_id} - tokens used: {tokens_used}")
            
            return {
                "document_id": document_id,
                "case_id": doc_metadata.get("case_id"),
                "workflow_id": workflow_id,
                "analysis_content": analysis_content,
                "model_used": "o3",
                "tokens_used": tokens_used,
                "analysis_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"o3 analysis failed for {document_id}: {e}")
            raise ValueError(f"o3 API analysis failed: {e}")


# Global OpenAI service instance
openai_service = OpenAIService()