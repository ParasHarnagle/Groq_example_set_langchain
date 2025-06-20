#!/usr/bin/env python3
"""
Complete Example: Advanced LangChain + Groq Document Processing
This example demonstrates all the advanced features of the enhanced integration.
"""

import os
import json
import asyncio
from pathlib import Path
from groq_agent_integration import parse_documents, parse_and_save_documents
from advanced_features import (
    AdvancedGroqConfig, 
    process_documents_advanced, 
    search_documents
)

# ================================================================
# Configuration and Setup
# ================================================================

def setup_environment():
    """Setup environment and configuration"""
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Verify Groq API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("⚠️  Please set your GROQ_API_KEY environment variable")
        print("   Get one at: https://console.groq.com/")
        return None
    
    # Create advanced configuration
    config = AdvancedGroqConfig(
        groq_api_key=groq_api_key,
        
        # Model settings
        model_name="llama-3.1-70b-versatile",  # Use the most capable model
        temperature=0.1,
        max_tokens=4000,
        
        # Processing settings
        batch_size=3,
        max_workers=4,
        chunk_size=3000,
        chunk_overlap=300,
        
        # Advanced features
        enable_cache=True,
        cache_dir="document_cache/",
        cache_ttl_hours=24,
        
        enable_vector_search=True,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        vector_store_path="vector_db/",
        
        # Quality control
        min_confidence_threshold=0.7,
        enable_quality_scoring=True,
        
        # Analysis features
        enable_table_detection=True,
        enable_image_description=True,
        generate_summary=True,
        extract_entities=True,
        generate_keywords=True,
        
        # Monitoring
        enable_metrics=True,
        metrics_file="processing_metrics.json"
    )
    
    return config

# ================================================================
# Example 1: Basic Document Processing
# ================================================================

def example_basic_processing():
    """Example of basic document processing"""
    print("\n🔄 Example 1: Basic Document Processing")
    print("=" * 50)
    
    # Process a single document
    file_paths = ["sample_documents/report.pdf"]
    
    try:
        results = parse_documents(file_paths, groq_api_key=os.getenv("GROQ_API_KEY"))
        
        if results:
            doc = results[0]
            print(f"✅ Processed: {file_paths[0]}")
            print(f"📄 Extracted {len(doc.chunks)} chunks")
            
            # Show markdown output
            print("\n📝 Markdown Output (first 500 chars):")
            print("-" * 40)
            print(doc.markdown[:500] + "..." if len(doc.markdown) > 500 else doc.markdown)
            
            # Show structured chunks
            print(f"\n🔍 Document Structure:")
            print("-" * 40)
            chunk_types = {}
            for chunk in doc.chunks:
                chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
            
            for chunk_type, count in chunk_types.items():
                print(f"  {chunk_type}: {count} chunks")
    
    except Exception as e:
        print(f"❌ Error: {e}")

# ================================================================
# Example 2: Advanced Processing with All Features
# ================================================================

async def example_advanced_processing():
    """Example of advanced processing with all features enabled"""
    print("\n🚀 Example 2: Advanced Processing with All Features")
    print("=" * 60)
    
    config = setup_environment()
    if not config:
        return
    
    # Sample documents
    file_paths = [
        "sample_documents/financial_report.pdf",
        "sample_documents/research_paper.pdf",
        "sample_documents/meeting_notes.pdf"
    ]
    
    # Filter existing files
    existing_files = [f for f in file_paths if Path(f).exists()]
    if not existing_files:
        print("⚠️  No sample documents found. Creating dummy files...")
        # Create sample directory and dummy files for demonstration
        Path("sample_documents").mkdir(exist_ok=True)
        dummy_content = "This is a sample document for testing purposes."
        for file_path in file_paths:
            if not Path(file_path).exists():
                with open(file_path.replace('.pdf', '.txt'), 'w') as f:
                    f.write(dummy_content)
        existing_files = [f.replace('.pdf', '.txt') for f in file_paths]
    
    try:
        print(f"🔄 Processing {len(existing_files)} documents with advanced features...")
        
        # Process with advanced features
        results = await process_documents_advanced(existing_files, config)
        
        print(f"✅ Processing complete!")
        
        # Analyze results
        for i, (file_path, result) in enumerate(zip(existing_files, results)):
            print(f"\n📄 Document {i+1}: {Path(file_path).name}")
            print("-" * 40)
            
            if result.chunks:
                print(f"   Chunks extracted: {len(result.chunks)}")
                
                # Show entities if extracted
                if "entities" in result.metadata:
                    entities = result.metadata["entities"]
                    total_entities = sum(len(v) for v in entities.values())
                    print(f"   Entities found: {total_entities}")
                    
                    for entity_type, entity_list in entities.items():
                        if entity_list:
                            print(f"     {entity_type}: {', '.join(entity_list[:3])}{'...' if len(entity_list) > 3 else ''}")
                
                # Show summary if generated
                if "summary" in result.metadata:
                    summary = result.metadata["summary"]
                    if isinstance(summary, dict) and "main_points" in summary:
                        print(f"   Summary: {summary.get('main_points', ['No summary available'])[0]}")
                
                # Show quality scores
                if "quality_assessment" in result.metadata:
                    quality = result.metadata["quality_assessment"]
                    print(f"   Quality scores:")
                    for metric, score in quality.items():
                        if isinstance(score, (int, float)):
                            print(f"     {metric}: {score:.2f}")
            
            if result.errors:
                print(f"   ⚠️  {len(result.errors)} errors occurred")
    
    except Exception as e:
        print(f"❌ Advanced processing error: {e}")

# ================================================================
# Example 3: Semantic Search
# ================================================================

async def example_semantic_search():
    """Example of semantic search functionality"""
    print("\n🔍 Example 3: Semantic Search")
    print("=" * 40)
    
    config = setup_environment()
    if not config:
        return
    
    # First, make sure we have documents in the vector store
    file_paths = ["sample_documents/report.txt"]  # Using text file for simplicity
    
    if not Path(file_paths[0]).exists():
        Path("sample_documents").mkdir(exist_ok=True)
        sample_content = """
        This is a comprehensive financial report for Q3 2024.
        
        Key Financial Metrics:
        - Revenue: $2.5M (up 15% from Q2)
        - Expenses: $1.8M
        - Net Profit: $700K
        - Customer Acquisition Cost: $150
        
        Market Analysis:
        The technology sector showed strong growth this quarter.
        Our main competitors include TechCorp and InnovateLabs.
        
        Future Outlook:
        We expect continued growth in Q4 with new product launches.
        Investment in R&D will increase by 20%.
        """
        
        with open(file_paths[0], 'w') as f:
            f.write(sample_content)
    
    try:
        # Process documents to populate vector store
        print("🔄 Processing documents for vector store...")
        results = await process_documents_advanced(file_paths, config)
        
        if results and results[0].chunks:
            print("