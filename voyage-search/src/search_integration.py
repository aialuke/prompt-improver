#!/usr/bin/env python3
"""
Enhanced Claude Code CLI Integration with Optimized Voyage AI Search
Provides semantic code search capabilities directly within Claude Code CLI.
"""

import os
import sys
import json
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

def load_config() -> Dict[str, Any]:
    """Load configuration from config.json."""
    config_path = os.path.join(os.path.dirname(__file__), "../config.json")
    with open(config_path, "r") as f:
        return json.load(f)

# Load configuration
CONFIG = load_config()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", CONFIG["project_root"]))
EMBEDDINGS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", CONFIG["embeddings_file"]))

# Add project src to path for imports
project_src_path = os.path.join(PROJECT_ROOT, 'src')
if project_src_path not in sys.path:
    sys.path.insert(0, project_src_path)

# Now import from the local src directory
from .search_code import SearchResult, load_enhanced_embeddings, HybridCodeSearch, HYBRID_SEARCH_CONFIG


def enhanced_search_code_with_path(embeddings_path: str, query: str, top_k: int = 5,
                                  min_similarity: float = 0.2, search_method: str = "hybrid") -> List[SearchResult]:
    """
    Enhanced search for code chunks using custom embeddings path.
    """
    try:
        # Load embeddings and chunks from custom path
        embeddings, chunks, metadata, binary_embeddings = load_enhanced_embeddings(embeddings_path)

        print(f"🔍 Enhanced search for: '{query}'")
        # Handle both dict and EmbeddingMetadata formats
        if hasattr(metadata, 'total_files_processed'):
            files_count = metadata.total_files_processed
        elif isinstance(metadata, dict):
            files_count = metadata.get('total_files_processed', 'unknown')
        else:
            files_count = 'unknown'
        print(f"📊 Loaded {len(chunks)} code chunks from {files_count} files")
        print(f"🔧 Search method: {search_method}")

        # Initialize hybrid search system
        hybrid_search = HybridCodeSearch(embeddings, chunks, metadata, binary_embeddings)

        # Perform search based on method
        if search_method == "semantic":
            # Use only semantic search
            old_config = HYBRID_SEARCH_CONFIG.copy()
            HYBRID_SEARCH_CONFIG["semantic_weight"] = 1.0
            HYBRID_SEARCH_CONFIG["lexical_weight"] = 0.0
            HYBRID_SEARCH_CONFIG["enable_binary_rescoring"] = False

            results = hybrid_search.hybrid_search(query, top_k, min_similarity)

            # Restore config
            HYBRID_SEARCH_CONFIG.update(old_config)
        else:
            # Use hybrid search (default)
            results = hybrid_search.hybrid_search(query, top_k, min_similarity)

        return results

    except Exception as e:
        print(f"❌ Search error: {e}")
        return []


@dataclass
class ClaudeCodeSearchRequest:
    """Request for Claude Code semantic search."""
    query: str
    max_results: int = 5
    min_similarity: float = 0.3
    search_method: str = "hybrid"  # "semantic", "lexical", "hybrid"
    include_context: bool = True
    analysis_type: Optional[str] = None  # "explain", "optimize", "review", "refactor"


@dataclass
class ClaudeCodeSearchResult:
    """Enhanced search result for Claude Code CLI."""
    query: str
    results: List[SearchResult]
    total_found: int
    search_time_ms: float
    search_method: str
    claude_prompt: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ClaudeCodeSearchCLI:
    """
    Enhanced Claude Code CLI integration with optimized Voyage AI search.
    
    Provides semantic code search capabilities that can be used directly
    within Claude Code CLI for intelligent code analysis and development.
    """
    
    def __init__(self, voyage_api_key: Optional[str] = None):
        """Initialize the Claude Code search integration."""
        # FIXED: Use hardcoded API key for reliability, fallback to env var
        self.voyage_api_key = voyage_api_key or os.getenv("VOYAGE_API_KEY") or 'pa-fhFUHl_VQ2SjvA7TNQ0gF8v021efH3R82qbigC3pnnB'

        # Set environment variable for search module compatibility
        os.environ["VOYAGE_API_KEY"] = self.voyage_api_key
        
        print("🔍 Claude Code Search Integration initialized")
        print(f"   Using optimized Voyage AI embeddings with 2025 best practices")
    
    def search(self, request: ClaudeCodeSearchRequest) -> ClaudeCodeSearchResult:
        """
        Perform semantic code search optimized for Claude Code CLI.
        
        Args:
            request: Search request with query and parameters
            
        Returns:
            Enhanced search result with Claude Code integration
        """
        start_time = time.time()
        
        print(f"🔍 Claude Code Search: '{request.query}'")
        print(f"   Method: {request.search_method}")
        print(f"   Max results: {request.max_results}")
        print(f"   Min similarity: {request.min_similarity}")
        
        # Perform enhanced search using our optimized system
        search_results = enhanced_search_code_with_path(
            embeddings_path=EMBEDDINGS_FILE,
            query=request.query,
            top_k=request.max_results,
            min_similarity=request.min_similarity,
            search_method=request.search_method
        )
        
        search_time = (time.time() - start_time) * 1000
        
        # Generate Claude Code prompt if analysis requested
        claude_prompt = None
        if request.analysis_type and search_results:
            claude_prompt = self._generate_claude_code_prompt(
                request.query, 
                request.analysis_type, 
                search_results
            )
        
        result = ClaudeCodeSearchResult(
            query=request.query,
            results=search_results,
            total_found=len(search_results),
            search_time_ms=search_time,
            search_method=request.search_method,
            claude_prompt=claude_prompt,
            metadata={
                "voyage_ai_model": "voyage-context-3",  # FIXED: Use voyage-context-3 for contextualized embeddings
                "contextualized_embeddings": True,
                "binary_rescoring": True,
                "reranking_model": "rerank-2.5-lite",
                "timestamp": time.time()
            }
        )
        
        print(f"✅ Found {len(search_results)} results in {search_time:.1f}ms")
        return result
    
    def search_and_format_for_claude(
        self, 
        query: str, 
        analysis_type: str = "explain",
        max_results: int = 3,
        min_similarity: float = 0.3
    ) -> str:
        """
        Search and format results specifically for Claude Code CLI.
        
        Returns a formatted string ready for Claude Code analysis.
        """
        request = ClaudeCodeSearchRequest(
            query=query,
            max_results=max_results,
            min_similarity=min_similarity,
            analysis_type=analysis_type,
            include_context=True
        )
        
        result = self.search(request)
        
        if not result.results:
            return f"No relevant code found for query: '{query}'"
        
        # Format for Claude Code CLI
        formatted_output = []
        formatted_output.append(f"# Code Search Results for: '{query}'")
        formatted_output.append(f"Found {result.total_found} relevant code chunks in {result.search_time_ms:.1f}ms")
        formatted_output.append("")
        
        for i, search_result in enumerate(result.results, 1):
            formatted_output.append(f"## Result {i}: {search_result.chunk_name}")
            formatted_output.append(f"**File:** `{search_result.file_path}`")
            formatted_output.append(f"**Lines:** {search_result.start_line}-{search_result.end_line}")
            formatted_output.append(f"**Relevance:** {search_result.similarity_score:.3f}")
            formatted_output.append(f"**Method:** {search_result.search_method}")
            formatted_output.append("")
            formatted_output.append("```python")
            formatted_output.append(search_result.content)
            formatted_output.append("```")
            formatted_output.append("")
        
        # Add Claude Code prompt if generated
        if result.claude_prompt:
            formatted_output.append("---")
            formatted_output.append("## Claude Code Analysis Prompt")
            formatted_output.append("")
            formatted_output.append(result.claude_prompt)
        
        return "\n".join(formatted_output)
    
    def _generate_claude_code_prompt(
        self, 
        query: str, 
        analysis_type: str, 
        search_results: List[SearchResult]
    ) -> str:
        """Generate optimized prompt for Claude Code CLI analysis."""
        
        # Prepare code context
        code_context = []
        for result in search_results:
            code_context.append(f"// File: {result.file_path} (Lines {result.start_line}-{result.end_line})")
            code_context.append(f"// Relevance: {result.similarity_score:.3f}")
            code_context.append(result.content)
            code_context.append("")
        
        context_text = "\n".join(code_context)
        
        # Analysis-specific prompts
        prompts = {
            "explain": f"""Please explain the following code found for query '{query}':

{context_text}

Focus on:
1. What this code does and its purpose
2. Key algorithms or patterns used
3. How the components work together
4. Any notable design decisions""",
            
            "optimize": f"""Please analyze and suggest optimizations for the following code found for query '{query}':

{context_text}

Focus on:
1. Performance improvements
2. Code quality enhancements
3. Best practices alignment
4. Potential refactoring opportunities""",
            
            "review": f"""Please perform a code review of the following code found for query '{query}':

{context_text}

Focus on:
1. Code quality and maintainability
2. Potential bugs or issues
3. Security considerations
4. Adherence to best practices""",
            
            "refactor": f"""Please suggest refactoring improvements for the following code found for query '{query}':

{context_text}

Focus on:
1. Structural improvements
2. Design pattern applications
3. Code organization
4. Maintainability enhancements"""
        }
        
        return prompts.get(analysis_type, prompts["explain"])
    
    def export_results_json(self, result: ClaudeCodeSearchResult, output_path: str) -> None:
        """Export search results to JSON for Claude Code CLI integration."""
        
        # Convert SearchResult objects to dictionaries
        results_data = []
        for search_result in result.results:
            results_data.append({
                "chunk_name": search_result.chunk_name,
                "file_path": search_result.file_path,
                "start_line": search_result.start_line,
                "end_line": search_result.end_line,
                "content": search_result.content,
                "similarity_score": search_result.similarity_score,
                "search_method": search_result.search_method,
                "chunk_type": search_result.chunk_type.value if hasattr(search_result.chunk_type, 'value') else str(search_result.chunk_type)
            })
        
        export_data = {
            "query": result.query,
            "total_found": result.total_found,
            "search_time_ms": result.search_time_ms,
            "search_method": result.search_method,
            "results": results_data,
            "claude_prompt": result.claude_prompt,
            "metadata": result.metadata
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📄 Results exported to: {output_path}")


def main() -> None:
    """CLI interface for Claude Code search integration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Claude Code Search Integration")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--analysis", choices=["explain", "optimize", "review", "refactor"], 
                       help="Analysis type for Claude Code")
    parser.add_argument("--max-results", type=int, default=3, help="Maximum results")
    parser.add_argument("--min-similarity", type=float, default=0.3, help="Minimum similarity")
    parser.add_argument("--method", choices=["semantic", "lexical", "hybrid"], default="hybrid",
                       help="Search method")
    parser.add_argument("--export", help="Export results to JSON file")
    parser.add_argument("--format-claude", action="store_true", 
                       help="Format output for Claude Code CLI")
    
    args = parser.parse_args()
    
    try:
        # Initialize search integration
        search_cli = ClaudeCodeSearchCLI()
        
        if args.format_claude:
            # Format for Claude Code CLI
            output = search_cli.search_and_format_for_claude(
                query=args.query,
                analysis_type=args.analysis or "explain",
                max_results=args.max_results,
                min_similarity=args.min_similarity
            )
            print(output)
        else:
            # Standard search
            request = ClaudeCodeSearchRequest(
                query=args.query,
                max_results=args.max_results,
                min_similarity=args.min_similarity,
                search_method=args.method,
                analysis_type=args.analysis
            )
            
            result = search_cli.search(request)
            
            # Display results
            for i, search_result in enumerate(result.results, 1):
                print(f"\n{i}. {search_result.chunk_name} ({search_result.similarity_score:.3f})")
                print(f"   File: {search_result.file_path}")
                print(f"   Lines: {search_result.start_line}-{search_result.end_line}")
                print(f"   Method: {search_result.search_method}")
        
        # Export if requested
        if args.export:
            search_cli.export_results_json(result, args.export)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
