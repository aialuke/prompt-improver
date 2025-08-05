#!/usr/bin/env python3
"""
Query Analysis for BM25 Optimization Enhancement

Analyzes the structure and coverage of assessment queries vs optimization queries
to support the expansion from 6 to 20 queries for parameter optimization.

This analysis supports the BM25 optimization enhancement project by providing
detailed insights into query types, coverage, and conversion requirements.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from collections import Counter


@dataclass
class QueryAnalysis:
    """Analysis results for query sets."""
    total_queries: int
    query_types: Dict[str, int]
    coverage_analysis: Dict[str, Any]
    conversion_mapping: List[Dict[str, Any]]


class QuerySetAnalyzer:
    """Analyzes query sets for optimization enhancement."""
    
    def __init__(self):
        """Initialize analyzer with current query sets."""
        self.optimization_queries = self._get_optimization_queries()
        self.assessment_queries = self._get_assessment_queries()
    
    def _get_optimization_queries(self) -> List[Dict[str, Any]]:
        """Current 6 queries from optimize_bm25_parameters.py"""
        return [
            {
                "query": "calculate similarity",
                "expected_chunks": ["calculate_similarity"],
                "type": "function_search"
            },
            {
                "query": "BM25Tokenizer",
                "expected_chunks": ["BM25Tokenizer"],
                "type": "class_search"
            },
            {
                "query": "tokenizeForBM25",
                "expected_chunks": ["tokenizeForBM25"],
                "type": "camelcase_search"
            },
            {
                "query": "remove stopwords",
                "expected_chunks": ["removeStopwords"],
                "type": "underscore_search"
            },
            {
                "query": "import numpy",
                "expected_chunks": ["imports"],
                "type": "import_search"
            },
            {
                "query": "tokenization manager",
                "expected_chunks": ["TokenizationManager"],
                "type": "compound_search"
            }
        ]
    
    def _get_assessment_queries(self) -> List[Dict[str, Any]]:
        """All 20 queries from assess_bm25_quality.py"""
        return [
            # Function name queries
            {"query": "calculate similarity", "type": "function_name", "expected_terms": ["calculate", "similarity"]},
            {"query": "tokenize text", "type": "function_name", "expected_terms": ["tokenize", "text"]},
            {"query": "generate embeddings", "type": "function_name", "expected_terms": ["generate", "embedding"]},
            
            # Class name queries
            {"query": "BM25Tokenizer", "type": "class_name", "expected_terms": ["bm25", "tokenizer"]},
            {"query": "HybridCodeSearch", "type": "class_name", "expected_terms": ["hybrid", "search"]},
            {"query": "TokenizationManager", "type": "class_name", "expected_terms": ["tokenization", "manager"]},
            
            # Method/API queries
            {"query": "cosine similarity", "type": "method_call", "expected_terms": ["cosine", "similarity"]},
            {"query": "stem words", "type": "method_call", "expected_terms": ["stem", "word"]},
            {"query": "remove stopwords", "type": "method_call", "expected_terms": ["remove", "stopword"]},
            
            # Code pattern queries
            {"query": "for loop iteration", "type": "code_pattern", "expected_terms": ["for", "loop"]},
            {"query": "try except error", "type": "code_pattern", "expected_terms": ["try", "except"]},
            {"query": "import numpy", "type": "code_pattern", "expected_terms": ["import", "numpy"]},
            
            # Camel case variations
            {"query": "calculateSimilarity", "type": "camel_case", "expected_terms": ["calculate", "similarity"]},
            {"query": "tokenizeText", "type": "camel_case", "expected_terms": ["tokenize", "text"]},
            
            # Partial matching (stemming test)
            {"query": "running", "type": "stemming_test", "expected_terms": ["run", "running"]},
            {"query": "tokenization", "type": "stemming_test", "expected_terms": ["token", "tokenize"]},
            {"query": "similarities", "type": "stemming_test", "expected_terms": ["similar", "similarity"]},
            
            # Stopword filtering test
            {"query": "the best algorithm for search", "type": "stopword_test", "expected_terms": ["algorithm", "search"]},
            {"query": "a function to calculate", "type": "stopword_test", "expected_terms": ["function", "calculate"]},
        ]
    
    def analyze_coverage(self) -> QueryAnalysis:
        """Analyze coverage differences between optimization and assessment queries."""
        
        # Count query types in each set
        opt_types = Counter(q["type"] for q in self.optimization_queries)
        assess_types = Counter(q["type"] for q in self.assessment_queries)
        
        # Find overlapping queries
        opt_queries_set = {q["query"] for q in self.optimization_queries}
        assess_queries_set = {q["query"] for q in self.assessment_queries}
        overlapping = opt_queries_set.intersection(assess_queries_set)
        
        # Create conversion mapping
        conversion_mapping = []
        for assess_query in self.assessment_queries:
            # Find if this query exists in optimization set
            opt_match = None
            for opt_query in self.optimization_queries:
                if opt_query["query"] == assess_query["query"]:
                    opt_match = opt_query
                    break
            
            conversion_mapping.append({
                "assessment_query": assess_query,
                "optimization_match": opt_match,
                "needs_conversion": opt_match is None,
                "conversion_required": self._determine_conversion_requirements(assess_query, opt_match)
            })
        
        coverage_analysis = {
            "optimization_types": dict(opt_types),
            "assessment_types": dict(assess_types),
            "overlapping_queries": list(overlapping),
            "new_queries_count": len(assess_queries_set - opt_queries_set),
            "coverage_expansion": {
                "function_name": 3,  # vs 1 in optimization
                "class_name": 3,     # vs 1 in optimization  
                "method_call": 3,    # vs 0 in optimization
                "code_pattern": 3,   # vs 1 in optimization
                "camel_case": 2,     # vs 1 in optimization
                "stemming_test": 3,  # vs 0 in optimization
                "stopword_test": 2   # vs 0 in optimization
            }
        }
        
        return QueryAnalysis(
            total_queries=len(self.assessment_queries),
            query_types=dict(assess_types),
            coverage_analysis=coverage_analysis,
            conversion_mapping=conversion_mapping
        )
    
    def _determine_conversion_requirements(self, assess_query: Dict[str, Any], 
                                         opt_match: Dict[str, Any] = None) -> Dict[str, Any]:
        """Determine what conversion is needed for assessment query format."""
        if opt_match:
            return {"type": "format_alignment", "changes": "minimal"}
        
        # New query needs full conversion
        return {
            "type": "new_query_conversion",
            "changes": "convert_expected_terms_to_expected_chunks",
            "assessment_format": "expected_terms",
            "optimization_format": "expected_chunks",
            "example_conversion": {
                "from": assess_query.get("expected_terms", []),
                "to": "needs_chunk_mapping_analysis"
            }
        }
    
    def generate_conversion_plan(self) -> Dict[str, Any]:
        """Generate detailed plan for converting assessment queries to optimization format."""
        analysis = self.analyze_coverage()
        
        conversion_plan = {
            "summary": {
                "total_queries_to_add": analysis.coverage_analysis["new_queries_count"],
                "format_conversions_needed": sum(1 for m in analysis.conversion_mapping if m["needs_conversion"]),
                "overlapping_queries": len(analysis.coverage_analysis["overlapping_queries"])
            },
            "conversion_steps": [],
            "format_requirements": {
                "optimization_format": {
                    "required_fields": ["query", "expected_chunks", "type"],
                    "expected_chunks_format": "List[str] - chunk names that should be found",
                    "type_format": "descriptive string for query category"
                },
                "assessment_format": {
                    "current_fields": ["query", "type", "expected_terms"],
                    "expected_terms_format": "List[str] - terms that should be tokenized",
                    "conversion_needed": "expected_terms -> expected_chunks mapping"
                }
            }
        }
        
        # Generate specific conversion steps
        for mapping in analysis.conversion_mapping:
            if mapping["needs_conversion"]:
                assess_query = mapping["assessment_query"]
                conversion_plan["conversion_steps"].append({
                    "query": assess_query["query"],
                    "type": assess_query["type"],
                    "action": "convert_to_optimization_format",
                    "challenge": "map expected_terms to actual chunk names in codebase",
                    "solution": "analyze codebase to find relevant chunks for each query"
                })
        
        return conversion_plan


def main():
    """Analyze query sets and generate conversion plan."""
    print("🔍 Query Set Analysis for BM25 Optimization Enhancement")
    print("=" * 60)
    
    analyzer = QuerySetAnalyzer()
    analysis = analyzer.analyze_coverage()
    conversion_plan = analyzer.generate_conversion_plan()
    
    print(f"\n📊 Coverage Analysis:")
    print(f"   Current optimization queries: {len(analyzer.optimization_queries)}")
    print(f"   Assessment queries available: {analysis.total_queries}")
    print(f"   New queries to add: {analysis.coverage_analysis['new_queries_count']}")
    print(f"   Overlapping queries: {len(analysis.coverage_analysis['overlapping_queries'])}")
    
    print(f"\n🎯 Query Type Distribution:")
    for query_type, count in analysis.query_types.items():
        print(f"   {query_type}: {count} queries")
    
    print(f"\n📈 Coverage Expansion Benefits:")
    for category, count in analysis.coverage_analysis["coverage_expansion"].items():
        print(f"   {category}: {count} queries (enhanced coverage)")
    
    print(f"\n🔧 Conversion Requirements:")
    print(f"   Queries needing conversion: {conversion_plan['summary']['format_conversions_needed']}")
    print(f"   Main challenge: expected_terms -> expected_chunks mapping")
    print(f"   Solution: Codebase analysis to identify relevant chunks")
    
    print(f"\n✅ Analysis Complete - Ready for implementation")
    
    return analysis, conversion_plan


if __name__ == "__main__":
    main()
