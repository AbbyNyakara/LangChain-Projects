import sys
from pathlib import Path
import logging
from typing import Dict, List

src_dir = Path(__name__).parent.parent
sys.path.insert(0, str(src_dir))


from src.etl_pipeline.pipeline import ETLPipeline
from src.etl_pipeline.embedder import EmbeddingPipeline, EmbeddingConfig, PineconeConfig
from src.generation.prompts import MEDICAL_ASSISTANT_PROMPT, MEDICAL_DIAGNOSIS_PROMPT
from src.generation.llm import GenerateService
from src.etl_pipeline.chunker import ChunkingConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class MedicalRAGPipeline:
    """
    Complete Medical RAG Pipeline
    Orchestrates: ETL (indexing) → Retrieval (search) → Generation (LLM)
    """
    def __init__(
            self, 
            s3_bucket: str,
            embedding_config: EmbeddingConfig = None,
            pinecone_config: PineconeConfig = None,
            chunking_config: ChunkingConfig = None,
            llm_config: Dict = None,
            region: str = "us-east-1"
            ):
        """
        Initialize RAG pipeline with all components
        
        Args:
            s3_bucket: S3 bucket for documents
            embedding_config: OpenAI embedding config
            pinecone_config: Pinecone vector DB config
            chunking_config: Document chunking config
            llm_config: LLM generation config
            region: AWS region
        """
        self.s3_bucket = s3_bucket
        self.region = region
        
        logger.info("Initializing Medical RAG Pipeline...")
        # ==== ETL-For indexing
        self.etl_pipeline = ETLPipeline(
            s3_bucket=s3_bucket,
            embedding_config=embedding_config or EmbeddingConfig(),
            pinecone_config=pinecone_config or PineconeConfig(),
            chunking_config=chunking_config or ChunkingConfig(),
            region=region
        )

         # ============ LLM COMPONENT (for generation) ============
        llm_config = llm_config or {}
        self.generator = GenerateService(
            model=llm_config.get('model', 'gpt-4-turbo'),
            temperature=llm_config.get('temperature', 0.2),
            max_tokens=llm_config.get('max_tokens', 500),
            timeout=llm_config.get('timeout', 30),
            retry_attempts=llm_config.get('retry_attempts', 3)
        )

        self.prompt = MEDICAL_ASSISTANT_PROMPT
        logger.info("Medical RAG Pipeline initialized")


    # TODO 1- INDDEX DOCUMENTS

    def index_document(self, file_path: str) -> Dict:
        try:
            result = self.etl_pipeline.process_document(file_path)
            return result # if succesful
        except Exception as e:
            logger.error("Error loading document %s", e)
            raise

    # TODO 2 - RETRIEVE CONTEXT

    def _format_context(self, search_results: List[Dict]) -> str:
        """
        Format search results into context string for LLM 
        Args:
            search_results: List of search result dicts
        Returns:
            Formatted context string
        """
        if not search_results:
            return "No context available."
        
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            text = result.get('text', '')
            
            metadata = result.get('metadata', {})
            source = metadata.get('original_filename', 'Unknown source')
            score = result.get('score', 0)
   
            context_parts.append(
                f"[{i}] {source} (relevance: {score:.2f}):\n{text}"
            )
        return "\n\n---\n\n".join(context_parts)

    def retrieve_context(self, query: str, top_k: int = 5) -> Dict:
        """Search with reranking"""
        logger.info("Retrieving context")
        
        try:
            # Use search_and_rerank instead of simple search
            search_result = self.etl_pipeline.embedder.search_and_rerank(
                query=query,
                top_k=10  # Get 10, rerank to 5
            )
            
            if not search_result['success'] or not search_result['results']:
                return {
                    'success': True,
                    'query': query,
                    'results': [],
                    'context': "No relevant documents found.",
                    'num_results': 0
                }   
            # Format context
            context = self._format_context(search_result['results'])
            
            return {
                'success': True,
                'query': query,
                'results': search_result['results'],
                'context': context,
                'num_results': len(search_result['results'])
            }
        except Exception as e:
            logger.error("Retrieval failed")
            
            return {
                'success': False,
                'error': str(e),
                'context': "Error retrieving documents.",
                'num_results': 0
            }
        
    # TODO 3- GENERATE ANSWER
    def generate_answer(self, query:str, context: str,use_chain: bool = True):
        """Generate answer from query and context using LLM"""
        try:
            if use_chain:
                answer = self.generator.generate_with_chain(self.prompt,context,query)
            else:
                answer = self.generator.generate_with_llm(self.prompt, context, query)
            return answer
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error generating answer: {str(e)}"
        
    ###### ========COMPLETE FLOW ===========

    def answer_question(self, query: str, top_k: int = 5) -> Dict:
        """
        Complete RAG flow: Retrieve → Generate
        This is the main entry point for user queries
        """
        try:
            result = self.retrieve_context(query, top_k)
            if not result['success']:
                logger.error("Error retrieving")

            answer = self.generate_answer(query, context= result['context'])
            return {
            'success': True,
            'query': query,
            'answer': answer,
            'sources': result.get('results', []),
            'num_sources': result['num_results']
            }
    
        except Exception as e:
            logger.error("RAG pipeline exception %s", e)
            raise
    

if __name__ == "__main__":
    file_to_upload = "/Users/abigaelmogusu/projects/LangChain-Projects/medical-rag-service/data/fake-aps.pdf"
    question = "What is the patient's name?"

    # Initialize pipeline
    pipeline = MedicalRAGPipeline(s3_bucket="medical-rag-docs-abigael-2026")
    
    # Index document
    print("\n" + "="*60)
    print("INDEXING DOCUMENT")

    index_result = pipeline.index_document(file_to_upload)
    print(f"Success: {index_result['success']}")
    print(f"Chunks: {index_result.get('total_chunks', 'N/A')}")
    
    # Answer question
    print("\n" + "="*60)
    print("ANSWERING QUESTION")
    print("="*60)
    result = pipeline.answer_question(query=question)
    
    print(f"\nQuestion: {result['query']}")
    print(f"Answer: {result['answer']}")
    print(f"Sources used: {result['num_sources']}")
    print("="*60)



    
    