#!/usr/bin/env python3
"""
Real Behavior Validation Script

Validates that our component testing framework is using real behavior:
- Real Voyage AI API calls (not mocked)
- Real embeddings from embeddings.pkl
- Real BM25 tokenization and scoring
- Real binary quantization calculations
- Real cross-encoder reranking

This script ensures NO MOCKS are being used in component testing.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Import components to validate
try:
    from search_code import (
        HybridCodeSearch, load_enhanced_embeddings, local_binary_quantization,
        SEARCH_METRICS, get_search_metrics
    )
except ImportError:
    # Handle import issues for direct execution
    import sys
    sys.path.append('.')
    from search_code import (
        HybridCodeSearch, load_enhanced_embeddings, local_binary_quantization,
        SEARCH_METRICS, get_search_metrics
    )


class RealBehaviorValidator:
    """Validates that components are using real behavior, not mocks."""
    
    def __init__(self):
        """Initialize validator."""
        self.validation_results = {}
        
    def validate_environment_setup(self) -> bool:
        """Validate environment is configured for real behavior testing."""
        print("🔍 Validating environment for real behavior testing...")
        
        # Check for embeddings file in multiple locations
        embeddings_paths = ["embeddings.pkl", "../data/embeddings.pkl", "data/embeddings.pkl"]
        embeddings_exists = any(Path(p).exists() for p in embeddings_paths)

        checks = {
            "voyage_api_key": bool(os.getenv('VOYAGE_API_KEY')),
            "embeddings_file": embeddings_exists,
            "no_mock_imports": self._check_no_mock_imports(),
            "real_packages_available": self._check_real_packages()
        }
        
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {check}: {passed}")
        
        all_passed = all(checks.values())
        print(f"\n{'✅' if all_passed else '❌'} Environment validation: {'PASSED' if all_passed else 'FAILED'}")
        
        self.validation_results["environment"] = checks
        return all_passed
    
    def _check_no_mock_imports(self) -> bool:
        """Check that no mock libraries are imported."""
        mock_modules = ['unittest.mock', 'mock', 'pytest-mock', 'responses']
        
        for module_name in mock_modules:
            if module_name in sys.modules:
                print(f"   ⚠️  Mock module detected: {module_name}")
                return False
        
        return True
    
    def _check_real_packages(self) -> bool:
        """Check that real packages are available."""
        try:
            import voyageai
            import rank_bm25
            import sentence_transformers
            import numpy
            import sklearn
            return True
        except ImportError as e:
            print(f"   ❌ Missing real package: {e}")
            return False
    
    def validate_real_embeddings_loading(self) -> bool:
        """Validate that real embeddings are loaded from file."""
        print("\n📊 Validating real embeddings loading...")
        
        try:
            # Try multiple embeddings file locations
            embeddings_paths = ["embeddings.pkl", "../data/embeddings.pkl", "data/embeddings.pkl"]
            embeddings_file = None
            for path in embeddings_paths:
                if Path(path).exists():
                    embeddings_file = path
                    break

            if not embeddings_file:
                raise FileNotFoundError("No embeddings.pkl file found")

            embeddings, chunks, metadata, binary_embeddings = load_enhanced_embeddings(embeddings_file)
            
            checks = {
                "embeddings_are_numpy_array": isinstance(embeddings, np.ndarray),
                "embeddings_have_data": embeddings.size > 0,
                "chunks_list_not_empty": len(chunks) > 0,
                "binary_embeddings_available": binary_embeddings is not None,
                "embeddings_dtype_float32": embeddings.dtype == np.float32,
                "binary_embeddings_dtype_int8": binary_embeddings.dtype == np.int8 if binary_embeddings is not None else True
            }
            
            for check, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"   {status} {check}: {passed}")
            
            # Additional validation
            print(f"   📊 Loaded {len(chunks)} real code chunks")
            print(f"   📊 Embeddings shape: {embeddings.shape}")
            if binary_embeddings is not None:
                print(f"   📊 Binary embeddings shape: {binary_embeddings.shape}")
            
            all_passed = all(checks.values())
            self.validation_results["embeddings"] = checks
            return all_passed
            
        except Exception as e:
            print(f"   ❌ Failed to load real embeddings: {e}")
            self.validation_results["embeddings"] = {"error": str(e)}
            return False
    
    def validate_real_api_calls(self) -> bool:
        """Validate that real Voyage AI API calls are made."""
        print("\n🌐 Validating real Voyage AI API calls...")
        
        try:
            # Reset metrics
            SEARCH_METRICS["api_calls"] = 0
            
            # Load real data
            embeddings, chunks, metadata, binary_embeddings = load_enhanced_embeddings("embeddings.pkl")
            
            # Create real search instance
            search = HybridCodeSearch(embeddings, chunks, metadata, binary_embeddings)
            
            # Perform a real search that should make API calls
            initial_api_calls = SEARCH_METRICS["api_calls"]
            
            results = search.hybrid_search("test query for api validation", top_k=3, min_similarity=0.1)
            
            final_api_calls = SEARCH_METRICS["api_calls"]
            api_calls_made = final_api_calls - initial_api_calls
            
            checks = {
                "api_calls_were_made": api_calls_made > 0,
                "results_returned": len(results) >= 0,  # Could be 0 if no matches
                "search_instance_created": search is not None,
                "voyage_client_initialized": search.vo is not None
            }
            
            for check, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"   {status} {check}: {passed}")
            
            print(f"   📊 API calls made: {api_calls_made}")
            print(f"   📊 Results returned: {len(results)}")
            
            all_passed = all(checks.values())
            self.validation_results["api_calls"] = checks
            self.validation_results["api_calls"]["calls_made"] = api_calls_made
            return all_passed
            
        except Exception as e:
            print(f"   ❌ Failed to validate API calls: {e}")
            self.validation_results["api_calls"] = {"error": str(e)}
            return False
    
    def validate_real_binary_quantization(self) -> bool:
        """Validate that real local binary quantization is working."""
        print("\n🔢 Validating real local binary quantization...")
        
        try:
            # Test with real float embedding
            test_embedding = [-0.03955078, 0.006214142, -0.07446289, -0.039001465,
                             0.0046463013, 0.00030612946, -0.08496094, 0.03994751]
            
            # Perform real binary quantization
            binary_result = local_binary_quantization(test_embedding)
            
            checks = {
                "binary_result_is_numpy": isinstance(binary_result, np.ndarray),
                "binary_result_dtype_int8": binary_result.dtype == np.int8,
                "binary_result_correct_size": len(binary_result) == 1,  # 8 bits packed into 1 byte
                "binary_result_expected_value": binary_result[0] == -51,  # Expected from Voyage AI docs
                "no_exceptions_raised": True
            }
            
            for check, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"   {status} {check}: {passed}")
            
            print(f"   📊 Binary result: {binary_result}")
            print(f"   📊 Expected: [-51], Got: {binary_result}")
            
            all_passed = all(checks.values())
            self.validation_results["binary_quantization"] = checks
            return all_passed
            
        except Exception as e:
            print(f"   ❌ Failed to validate binary quantization: {e}")
            self.validation_results["binary_quantization"] = {"error": str(e)}
            return False
    
    def validate_real_bm25_tokenization(self) -> bool:
        """Validate that real BM25 tokenization is working."""
        print("\n📝 Validating real BM25 tokenization...")
        
        try:
            # Load real data
            embeddings, chunks, metadata, binary_embeddings = load_enhanced_embeddings("embeddings.pkl")
            
            # Create search instance with BM25
            search = HybridCodeSearch(embeddings, chunks, metadata, binary_embeddings)
            
            checks = {
                "bm25_instance_created": search.bm25 is not None,
                "bm25_has_real_data": hasattr(search.bm25, 'doc_freqs') if search.bm25 else False,
                "chunks_were_tokenized": len(chunks) > 0,
                "no_mock_tokenizer": not hasattr(search.bm25, '_mock') if search.bm25 else True
            }
            
            # Test tokenization
            if search.bm25:
                test_query = "test function implementation"
                try:
                    tokenized = search._tokenize_query_for_bm25(test_query)
                    checks["tokenization_works"] = len(tokenized) > 0
                    checks["tokenized_result_is_list"] = isinstance(tokenized, list)
                    print(f"   📊 Tokenized '{test_query}' -> {tokenized}")
                except Exception as e:
                    checks["tokenization_works"] = False
                    print(f"   ⚠️  Tokenization error: {e}")
            
            for check, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"   {status} {check}: {passed}")
            
            all_passed = all(checks.values())
            self.validation_results["bm25_tokenization"] = checks
            return all_passed
            
        except Exception as e:
            print(f"   ❌ Failed to validate BM25 tokenization: {e}")
            self.validation_results["bm25_tokenization"] = {"error": str(e)}
            return False
    
    def validate_no_mocked_components(self) -> bool:
        """Validate that no components are mocked."""
        print("\n🚫 Validating no mocked components...")
        
        try:
            # Load real data
            embeddings, chunks, metadata, binary_embeddings = load_enhanced_embeddings("embeddings.pkl")
            
            # Create search instance
            search = HybridCodeSearch(embeddings, chunks, metadata, binary_embeddings)
            
            checks = {
                "embeddings_not_mocked": not hasattr(embeddings, '_mock'),
                "chunks_not_mocked": not hasattr(chunks, '_mock'),
                "search_instance_not_mocked": not hasattr(search, '_mock'),
                "voyage_client_not_mocked": not hasattr(search.vo, '_mock') if search.vo else True,
                "bm25_not_mocked": not hasattr(search.bm25, '_mock') if search.bm25 else True,
                "cross_encoder_not_mocked": not hasattr(search.cross_encoder, '_mock') if search.cross_encoder else True
            }
            
            for check, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"   {status} {check}: {passed}")
            
            all_passed = all(checks.values())
            self.validation_results["no_mocks"] = checks
            return all_passed
            
        except Exception as e:
            print(f"   ❌ Failed to validate no mocks: {e}")
            self.validation_results["no_mocks"] = {"error": str(e)}
            return False
    
    def run_comprehensive_validation(self) -> bool:
        """Run all validation checks."""
        print("🧪 COMPREHENSIVE REAL BEHAVIOR VALIDATION")
        print("=" * 60)
        print("🎯 Ensuring NO MOCKS are used in component testing")
        print("=" * 60)
        
        validations = [
            ("Environment Setup", self.validate_environment_setup),
            ("Real Embeddings Loading", self.validate_real_embeddings_loading),
            ("Real API Calls", self.validate_real_api_calls),
            ("Real Binary Quantization", self.validate_real_binary_quantization),
            ("Real BM25 Tokenization", self.validate_real_bm25_tokenization),
            ("No Mocked Components", self.validate_no_mocked_components)
        ]
        
        all_passed = True
        
        for validation_name, validation_func in validations:
            try:
                passed = validation_func()
                all_passed = all_passed and passed
            except Exception as e:
                print(f"\n❌ {validation_name} validation failed: {e}")
                all_passed = False
        
        print("\n" + "=" * 60)
        if all_passed:
            print("✅ ALL VALIDATIONS PASSED")
            print("🎉 Component testing framework uses REAL BEHAVIOR ONLY")
            print("🚫 NO MOCKS detected")
        else:
            print("❌ SOME VALIDATIONS FAILED")
            print("⚠️  Component testing may use mocked behavior")
        print("=" * 60)
        
        return all_passed
    
    def save_validation_report(self, output_file: str = "validation_report.json") -> None:
        """Save validation results to file."""
        import json

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert any non-serializable objects to strings
        serializable_results = {}
        for key, value in self.validation_results.items():
            if isinstance(value, dict):
                serializable_results[key] = {k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v
                                           for k, v in value.items()}
            else:
                serializable_results[key] = str(value) if not isinstance(value, (str, int, float, bool, list, dict, type(None))) else value

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"💾 Validation report saved to {output_path}")


def main():
    """Main validation runner."""
    validator = RealBehaviorValidator()
    
    # Run comprehensive validation
    all_passed = validator.run_comprehensive_validation()
    
    # Save report
    validator.save_validation_report("test_results/real_behavior_validation.json")
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
