# ================================================================
# Advanced Features and Extensions for LangChain + Groq Integration
# ================================================================

import io
import re
import hashlib
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import sqlite3
import pickle
from datetime import datetime, timedelta
import asyncio
from pathlib import Path

# Additional imports for advanced features
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================================================================
# Enhanced Configuration with Advanced Options
# ================================================================

@dataclass
class AdvancedGroqConfig(GroqAgentConfig):
    """Extended configuration with advanced features"""
    
    # Caching options
    enable_cache: bool = True
    cache_dir: str = "cache/"
    cache_ttl_hours: int = 24
    
    # Vector search options
    enable_vector_search: bool = False
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store_path: str = "vector_store/"
    
    # Quality control
    min_confidence_threshold: float = 0.6
    enable_quality_scoring: bool = True
    
    # Advanced processing
    enable_table_detection: bool = True
    enable_image_description: bool = True
    enable_multilingual: bool = False
    
    # Output options
    generate_summary: bool = True
    extract_entities: bool = True
    generate_keywords: bool = True
    
    # Monitoring
    enable_metrics: bool = True
    metrics_file: str = "processing_metrics.json"

# ================================================================
# Caching System for API Calls
# ================================================================

class DocumentCache:
    """Intelligent caching system for processed documents"""
    
    def __init__(self, cache_dir: str, ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self.db_path = self.cache_dir / "cache.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite cache database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_cache (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT,
                processed_date TIMESTAMP,
                result_data BLOB,
                metadata TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash for file content"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def get_cached_result(self, file_path: str) -> Optional[Any]:
        """Retrieve cached processing result"""
        try:
            file_hash = self._get_file_hash(file_path)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT result_data, processed_date FROM document_cache WHERE file_hash = ?",
                (file_hash,)
            )
            row = cursor.fetchone()
            conn.close()
            
            if row:
                result_data, processed_date = row
                processed_time = datetime.fromisoformat(processed_date)
                
                # Check if cache is still valid
                if datetime.now() - processed_time < self.ttl:
                    return pickle.loads(result_data)
                else:
                    # Clean expired cache
                    self.remove_cached_result(file_hash)
            
            return None
        except Exception as e:
            logging.error(f"Cache retrieval error: {e}")
            return None
    
    def cache_result(self, file_path: str, result: Any):
        """Cache processing result"""
        try:
            file_hash = self._get_file_hash(file_path)
            result_data = pickle.dumps(result)
            
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT OR REPLACE INTO document_cache VALUES (?, ?, ?, ?, ?)",
                (file_hash, file_path, datetime.now().isoformat(), 
                 result_data, json.dumps(result.metadata))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Cache storage error: {e}")
    
    def remove_cached_result(self, file_hash: str):
        """Remove expired cache entry"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM document_cache WHERE file_hash = ?", (file_hash,))
        conn.commit()
        conn.close()

# ================================================================
# Advanced Document Analysis Agent
# ================================================================

class AdvancedGroqAgent(GroqDocumentAgent):
    """Enhanced agent with advanced document analysis capabilities"""
    
    def __init__(self, config: AdvancedGroqConfig):
        super().__init__(config)
        self.config = config
        self.cache = DocumentCache(config.cache_dir, config.cache_ttl_hours) if config.enable_cache else None
        self.setup_advanced_features()
    
    def setup_advanced_features(self):
        """Initialize advanced features"""
        
        # Setup vector store for semantic search
        if self.config.enable_vector_search:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model
            )
            self.vector_store_path = Path(self.config.vector_store_path)
            self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Enhanced prompts for advanced analysis
        self.advanced_prompts = {
            'entity_extraction': """
            Extract key entities from this document content:
            - PERSONS: Names of people
            - ORGANIZATIONS: Company names, institutions
            - LOCATIONS: Places, addresses
            - DATES: Important dates and time periods
            - MONEY: Financial amounts, costs
            - PRODUCTS: Product names, services
            
            Return as JSON: {"entities": {"PERSON": [...], "ORG": [...], ...}}
            """,
            
            'summary_generation': """
            Generate a comprehensive summary of this document:
            1. Main purpose/topic (1-2 sentences)
            2. Key findings or conclusions (bullet points)
            3. Important details or data points
            4. Action items or recommendations (if any)
            
            Format as structured JSON with clear sections.
            """,
            
            'table_analysis': """
            Analyze any tables in this content:
            1. Identify table structure (rows, columns, headers)
            2. Extract key data relationships
            3. Summarize main insights from the data
            4. Convert to structured format (JSON or CSV-like)
            
            Focus on numerical data, trends, and comparisons.
            """,
            
            'quality_assessment': """
            Assess the quality and completeness of this extracted content:
            1. Completeness score (0-1): How much of the original content is captured?
            2. Accuracy confidence (0-1): How confident are you in the extraction?
            3. Structure quality (0-1): How well is the document structure preserved?
            4. Missing elements: What might be missing or unclear?
            
            Return as JSON with scores and explanations.
            """
        }
    
    async def extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract named entities from content"""
        try:
            prompt = f"{self.advanced_prompts['entity_extraction']}\n\nContent:\n{content}"
            response = await asyncio.to_thread(
                self.llm.invoke,
                [HumanMessage(content=prompt)]
            )
            
            # Parse JSON response
            entities = json.loads(response.content)
            return entities.get('entities', {})
        except Exception as e:
            logging.error(f"Entity extraction failed: {e}")
            return {}
    
    async def generate_summary(self, content: str) -> Dict[str, Any]:
        """Generate document summary"""
        try:
            prompt = f"{self.advanced_prompts['summary_generation']}\n\nContent:\n{content}"
            response = await asyncio.to_thread(
                self.llm.invoke,
                [HumanMessage(content=prompt)]
            )
            
            summary = json.loads(response.content)
            return summary
        except Exception as e:
            logging.error(f"Summary generation failed: {e}")
            return {"summary": "Summary generation failed", "error": str(e)}
    
    async def analyze_tables(self, content: str) -> List[Dict[str, Any]]:
        """Analyze and extract table data"""
        try:
            prompt = f"{self.advanced_prompts['table_analysis']}\n\nContent:\n{content}"
            response = await asyncio.to_thread(
                self.llm.invoke,
                [HumanMessage(content=prompt)]
            )
            
            tables = json.loads(response.content)
            return tables.get('tables', [])
        except Exception as e:
            logging.error(f"Table analysis failed: {e}")
            return []
    
    async def assess_quality(self, original_content: str, extracted_content: str) -> Dict[str, Any]:
        """Assess extraction quality"""
        try:
            comparison_prompt = f"""
            {self.advanced_prompts['quality_assessment']}
            
            Original content length: {len(original_content)} characters
            Extracted content length: {len(extracted_content)} characters
            
            Extracted content:
            {extracted_content[:1000]}...
            """
            
            response = await asyncio.to_thread(
                self.llm.invoke,
                [HumanMessage(content=comparison_prompt)]
            )
            
            quality_assessment = json.loads(response.content)
            return quality_assessment
        except Exception as e:
            logging.error(f"Quality assessment failed: {e}")
            return {"completeness": 0.5, "accuracy": 0.5, "structure": 0.5}

# ================================================================
# Semantic Search and Vector Storage
# ================================================================

class DocumentVectorStore:
    """Vector store for semantic document search"""
    
    def __init__(self, embeddings_model: str, store_path: str):
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.vector_store = None
        self.load_or_create_store()
    
    def load_or_create_store(self):
        """Load existing vector store or create new one"""
        faiss_path = self.store_path / "faiss_index"
        if faiss_path.exists():
            try:
                self.vector_store = FAISS.load_local(
                    str(faiss_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                logging.warning(f"Failed to load vector store: {e}")
                self.vector_store = None
    
    def add_documents(self, documents: List[DocumentChunk], file_path: str):
        """Add document chunks to vector store"""
        try:
            # Convert chunks to LangChain documents
            docs = []
            for i, chunk in enumerate(documents):
                doc = Document(
                    page_content=chunk.content,
                    metadata={
                        "source": file_path,
                        "chunk_id": i,
                        "chunk_type": chunk.chunk_type,
                        "page_number": chunk.page_number,
                        "confidence": chunk.confidence
                    }
                )
                docs.append(doc)
            
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(docs, self.embeddings)
            else:
                self.vector_store.add_documents(docs)
            
            # Save updated store
            faiss_path = self.store_path / "faiss_index"
            self.vector_store.save_local(str(faiss_path))
            
        except Exception as e:
            logging.error(f"Failed to add documents to vector store: {e}")
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar document chunks"""
        if self.vector_store is None:
            return []
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            similar_chunks = []
            for doc, score in results:
                similar_chunks.append({
                    "content": doc.page_content,
                    "similarity_score": float(score),
                    "metadata": doc.metadata
                })
            
            return similar_chunks
        except Exception as e:
            logging.error(f"Vector search failed: {e}")
            return []

# ================================================================
# Enhanced Document Processor with Advanced Features
# ================================================================

class AdvancedDocumentProcessor(DocumentProcessor):
    """Enhanced processor with advanced analysis capabilities"""
    
    def __init__(self, config: AdvancedGroqConfig):
        super().__init__(config)
        self.config = config
        self.agent = AdvancedGroqAgent(config)
        
        # Initialize advanced features
        if config.enable_vector_search:
            self.vector_store = DocumentVectorStore(
                config.embedding_model,
                config.vector_store_path
            )
        else:
            self.vector_store = None
        
        # Metrics tracking
        self.metrics = {
            "documents_processed": 0,
            "total_chunks": 0,
            "processing_times": [],
            "errors": [],
            "quality_scores": []
        }
    
    async def process_document_advanced(self, file_path: str) -> ParsedDocument:
        """Enhanced document processing with advanced features"""
        start_time = datetime.now()
        
        try:
            # Check cache first
            if self.config.enable_cache and self.agent.cache:
                cached_result = self.agent.cache.get_cached_result(file_path)
                if cached_result:
                    logging.info(f"Using cached result for {file_path}")
                    return cached_result
            
            # Standard processing
            result = await super().process_document_async(file_path)
            
            # Enhanced analysis
            if result.chunks:
                await self._apply_advanced_analysis(result, file_path)
            
            # Cache result
            if self.config.enable_cache and self.agent.cache:
                self.agent.cache.cache_result(file_path, result)
            
            # Add to vector store
            if self.config.enable_vector_search and self.vector_store:
                self.vector_store.add_documents(result.chunks, file_path)
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(result, processing_time)
            
            return result
            
        except Exception as e:
            logging.error(f"Advanced processing failed for {file_path}: {e}")
            self.metrics["errors"].append({
                "file": file_path,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
            # Return basic result with error
            return ParsedDocument(
                chunks=[],
                markdown="",
                metadata={"error": str(e), "file_path": file_path},
                errors=[{"error": str(e), "error_type": "advanced_processing_error"}]
            )
    
    async def _apply_advanced_analysis(self, result: ParsedDocument, file_path: str):
        """Apply advanced analysis features"""
        full_content = "\n".join([chunk.content for chunk in result.chunks])
        
        # Entity extraction
        if self.config.extract_entities:
            try:
                entities = await self.agent.extract_entities(full_content)
                result.metadata["entities"] = entities
            except Exception as e:
                logging.error(f"Entity extraction failed: {e}")
        
        # Summary generation
        if self.config.generate_summary:
            try:
                summary = await self.agent.generate_summary(full_content)
                result.metadata["summary"] = summary
            except Exception as e:
                logging.error(f"Summary generation failed: {e}")
        
        # Table analysis
        if self.config.enable_table_detection:
            try:
                tables = await self.agent.analyze_tables(full_content)
                result.metadata["tables"] = tables
            except Exception as e:
                logging.error(f"Table analysis failed: {e}")
        
        # Quality assessment
        if self.config.enable_quality_scoring:
            try:
                quality = await self.agent.assess_quality(full_content, result.markdown)
                result.metadata["quality_assessment"] = quality
                
                # Filter low-quality chunks
                if quality.get("completeness", 1.0) < self.config.min_confidence_threshold:
                    logging.warning(f"Low quality extraction for {file_path}")
                
            except Exception as e:
                logging.error(f"Quality assessment failed: {e}")
    
    def _update_metrics(self, result: ParsedDocument, processing_time: float):
        """Update processing metrics"""
        self.metrics["documents_processed"] += 1
        self.metrics["total_chunks"] += len(result.chunks)
        self.metrics["processing_times"].append(processing_time)
        
        if "quality_assessment" in result.metadata:
            self.metrics["quality_scores"].append(result.metadata["quality_assessment"])
    
    def get_processing_report(self) -> Dict[str, Any]:
        """Generate comprehensive processing report"""
        report = {
            "summary": {
                "documents_processed": self.metrics["documents_processed"],
                "total_chunks": self.metrics["total_chunks"],
                "total_errors": len(self.metrics["errors"]),
                "average_processing_time": np.mean(self.metrics["processing_times"]) if self.metrics["processing_times"] else 0
            },
            "performance": {
                "processing_times": self.metrics["processing_times"],
                "min_time": min(self.metrics["processing_times"]) if self.metrics["processing_times"] else 0,
                "max_time": max(self.metrics["processing_times"]) if self.metrics["processing_times"] else 0
            },
            "quality": {
                "average_scores": {},
                "quality_distribution": {}
            },
            "errors": self.metrics["errors"]
        }
        
        # Calculate quality metrics
        if self.metrics["quality_scores"]:
            scores = self.metrics["quality_scores"]
            for metric in ["completeness", "accuracy", "structure"]:
                values = [score.get(metric, 0) for score in scores]
                if values:
                    report["quality"]["average_scores"][metric] = np.mean(values)
        
        return report
    
    def save_metrics(self):
        """Save metrics to file"""
        if self.config.enable_metrics:
            report = self.get_processing_report()
            with open(self.config.metrics_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)

# ================================================================
# Semantic Search Interface
# ================================================================

def search_documents(query: str, 
                    vector_store_path: str = "vector_store/",
                    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                    k: int = 5) -> List[Dict[str, Any]]:
    """Search processed documents semantically"""
    
    vector_store = DocumentVectorStore(embedding_model, vector_store_path)
    results = vector_store.search_similar(query, k)
    
    return results

# ================================================================
# Batch Processing with Advanced Features
# ================================================================

async def process_documents_advanced(file_paths: List[str],
                                   config: Optional[AdvancedGroqConfig] = None,
                                   **kwargs) -> List[ParsedDocument]:
    """Advanced batch processing with enhanced features"""
    
    if config is None:
        groq_api_key = kwargs.get('groq_api_key') or os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("Groq API key required")
        
        config = AdvancedGroqConfig(
            groq_api_key=groq_api_key,
            **{k: v for k, v in kwargs.items() if hasattr(AdvancedGroqConfig, k)}
        )
    
    processor = AdvancedDocumentProcessor(config)
    
    # Process documents
    tasks = []
    for file_path in file_paths:
        task = processor.process_document_advanced(file_path)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions and compile results
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logging.error(f"Failed to process {file_paths[i]}: {result}")
            processed_results.append(ParsedDocument(
                chunks=[],
                markdown="",
                metadata={"error": str(result), "file_path": file_paths[i]},
                errors=[{"error": str(result), "file": file_paths[i]}]
            ))
        else:
            processed_results.append(result)
    
    # Save metrics
    processor.save_metrics()
    
    # Generate final report
    report = processor.get_processing_report()
    logging.info(f"Processing complete: {report['summary']}")
    
    return processed_results