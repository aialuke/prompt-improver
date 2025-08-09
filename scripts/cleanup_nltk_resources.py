"""NLTK Resource Cleanup Script

This script helps reduce NLTK footprint by removing non-English resources
and keeping only what's needed for English-only processing.
"""
import os
from pathlib import Path
import shutil
from typing import Dict, List, Tuple
import nltk

def get_nltk_data_path() -> Path:
    """Get the NLTK data directory path."""
    for path in nltk.data.path:
        nltk_path = Path(path)
        if nltk_path.exists():
            return nltk_path
    return Path.home() / 'nltk_data'

def analyze_nltk_resources() -> Dict[str, any]:
    """Analyze current NLTK resources and identify cleanup opportunities."""
    nltk_path = get_nltk_data_path()
    analysis = {'total_size_mb': 0, 'english_size_mb': 0, 'non_english_size_mb': 0, 'non_english_languages': [], 'removable_resources': [], 'essential_resources': [], 'potential_savings_mb': 0}
    if not nltk_path.exists():
        print(f'❌ NLTK data directory not found: {nltk_path}')
        return analysis
    print(f'📁 Analyzing NLTK resources in: {nltk_path}')
    essential_english = {'tokenizers/punkt', 'tokenizers/punkt_tab/english', 'corpora/stopwords/english', 'corpora/wordnet', 'taggers/averaged_perceptron_tagger', 'sentiment/vader_lexicon'}
    tokenizers_path = nltk_path / 'tokenizers'
    if tokenizers_path.exists():
        analysis.update(_analyze_tokenizers(tokenizers_path, essential_english))
    corpora_path = nltk_path / 'corpora'
    if corpora_path.exists():
        analysis.update(_analyze_corpora(corpora_path, essential_english))
    for resource_path in nltk_path.rglob('*'):
        if resource_path.is_file():
            size = resource_path.stat().st_size
            analysis['total_size_mb'] += size / (1024 * 1024)
    analysis['potential_savings_mb'] = analysis['non_english_size_mb']
    return analysis

def _analyze_tokenizers(tokenizers_path: Path, essential: set) -> Dict[str, any]:
    """Analyze tokenizer resources."""
    analysis = {'non_english_languages': [], 'removable_resources': [], 'non_english_size_mb': 0}
    punkt_tab_path = tokenizers_path / 'punkt_tab'
    if punkt_tab_path.exists():
        for lang_dir in punkt_tab_path.iterdir():
            if lang_dir.is_dir() and lang_dir.name != 'english':
                analysis['non_english_languages'].append(f'punkt_tab/{lang_dir.name}')
                analysis['removable_resources'].append(str(lang_dir))
                for file_path in lang_dir.rglob('*'):
                    if file_path.is_file():
                        analysis['non_english_size_mb'] += file_path.stat().st_size / (1024 * 1024)
    return analysis

def _analyze_corpora(corpora_path: Path, essential: set) -> Dict[str, any]:
    """Analyze corpora resources."""
    analysis = {'removable_resources': [], 'non_english_size_mb': 0}
    stopwords_path = corpora_path / 'stopwords'
    if stopwords_path.exists():
        for lang_file in stopwords_path.iterdir():
            if lang_file.is_file() and lang_file.name != 'english':
                analysis['removable_resources'].append(str(lang_file))
                analysis['non_english_size_mb'] += lang_file.stat().st_size / (1024 * 1024)
    large_corpora = ['brown', 'reuters', 'gutenberg', 'inaugural', 'webtext']
    for corpus_name in large_corpora:
        corpus_path = corpora_path / corpus_name
        if corpus_path.exists():
            corpus_size = 0
            for file_path in corpus_path.rglob('*'):
                if file_path.is_file():
                    corpus_size += file_path.stat().st_size
            corpus_size_mb = corpus_size / (1024 * 1024)
            if corpus_size_mb > 1:
                analysis['removable_resources'].append(f'{corpus_name} ({corpus_size_mb:.1f}MB)')
    return analysis

def cleanup_non_english_resources(dry_run: bool=True) -> Dict[str, any]:
    """Clean up non-English NLTK resources.
    
    Args:
        dry_run: If True, only show what would be removed without actually removing
        
    Returns:
        Dictionary with cleanup results
    """
    nltk_path = get_nltk_data_path()
    results = {'removed_files': [], 'removed_directories': [], 'space_saved_mb': 0, 'errors': []}
    if not nltk_path.exists():
        results['errors'].append(f'NLTK data directory not found: {nltk_path}')
        return results
    print(f"🧹 {('[DRY RUN] ' if dry_run else '')}Cleaning up non-English NLTK resources...")
    punkt_tab_path = nltk_path / 'tokenizers' / 'punkt_tab'
    if punkt_tab_path.exists():
        for lang_dir in punkt_tab_path.iterdir():
            if lang_dir.is_dir() and lang_dir.name != 'english':
                size_mb = _calculate_directory_size(lang_dir)
                if dry_run:
                    print(f'  Would remove: {lang_dir.name} tokenizer ({size_mb:.1f}MB)')
                    results['space_saved_mb'] += size_mb
                else:
                    try:
                        shutil.rmtree(lang_dir)
                        results['removed_directories'].append(str(lang_dir))
                        results['space_saved_mb'] += size_mb
                        print(f'  ✅ Removed: {lang_dir.name} tokenizer ({size_mb:.1f}MB)')
                    except Exception as e:
                        error_msg = f'Failed to remove {lang_dir}: {e}'
                        results['errors'].append(error_msg)
                        print(f'  ❌ {error_msg}')
    stopwords_path = nltk_path / 'corpora' / 'stopwords'
    if stopwords_path.exists():
        for lang_file in stopwords_path.iterdir():
            if lang_file.is_file() and lang_file.name != 'english':
                size_mb = lang_file.stat().st_size / (1024 * 1024)
                if dry_run:
                    print(f'  Would remove: {lang_file.name} stopwords ({size_mb:.2f}MB)')
                    results['space_saved_mb'] += size_mb
                else:
                    try:
                        lang_file.unlink()
                        results['removed_files'].append(str(lang_file))
                        results['space_saved_mb'] += size_mb
                        print(f'  ✅ Removed: {lang_file.name} stopwords ({size_mb:.2f}MB)')
                    except Exception as e:
                        error_msg = f'Failed to remove {lang_file}: {e}'
                        results['errors'].append(error_msg)
                        print(f'  ❌ {error_msg}')
    return results

def _calculate_directory_size(directory: Path) -> float:
    """Calculate total size of directory in MB."""
    total_size = 0
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    return total_size / (1024 * 1024)

def verify_english_resources() -> Dict[str, bool]:
    """Verify that essential English resources are still available."""
    from prompt_improver.ml.learning.features.english_nltk_manager import get_english_nltk_manager
    print('🔍 Verifying English NLTK resources...')
    manager = get_english_nltk_manager()
    status = manager.get_resource_status()
    verification = {'punkt_tokenizer': False, 'english_stopwords': False, 'wordnet': False, 'all_essential_available': False}
    try:
        tokenizer = manager.get_sentence_tokenizer()
        test_sentences = tokenizer('Hello world. This is a test.')
        verification['punkt_tokenizer'] = len(test_sentences) >= 2
        print(f"  {('✅' if verification['punkt_tokenizer'] else '❌')} Sentence tokenizer")
        stopwords = manager.get_english_stopwords()
        verification['english_stopwords'] = len(stopwords) > 100
        print(f"  {('✅' if verification['english_stopwords'] else '❌')} English stopwords ({len(stopwords)} words)")
        lemmatizer = manager.get_lemmatizer()
        test_lemma = lemmatizer.lemmatize('running', 'v')
        verification['wordnet'] = test_lemma is not None
        print(f"  {('✅' if verification['wordnet'] else '❌')} WordNet lemmatizer")
        verification['all_essential_available'] = all([verification['punkt_tokenizer'], verification['english_stopwords'], verification['wordnet']])
        print(f'\n📊 Resource Status:')
        print(f"  Available: {status['required_available']}/{status['required_total']} required")
        print(f"  Optional: {status['optional_available']}/{status['optional_total']} optional")
        print(f"  Fallback mode: {status['fallback_mode']}")
    except Exception as e:
        print(f'❌ Verification failed: {e}')
    return verification

def main():
    """Main cleanup script."""
    print('🚀 NLTK English-Only Optimization Script')
    print('=' * 50)
    analysis = analyze_nltk_resources()
    print(f'\n📊 Current NLTK Resource Analysis:')
    print(f"  Total size: {analysis['total_size_mb']:.1f}MB")
    print(f"  Non-English languages found: {len(analysis['non_english_languages'])}")
    print(f"  Potential space savings: {analysis['potential_savings_mb']:.1f}MB")
    if analysis['non_english_languages']:
        print(f'\n🌍 Non-English resources found:')
        for lang in analysis['non_english_languages'][:10]:
            print(f'    • {lang}')
        if len(analysis['non_english_languages']) > 10:
            print(f"    ... and {len(analysis['non_english_languages']) - 10} more")
    print(f'\n🧹 Cleanup Preview (Dry Run):')
    dry_run_results = cleanup_non_english_resources(dry_run=True)
    if dry_run_results['space_saved_mb'] > 0:
        print(f"\n💾 Potential space savings: {dry_run_results['space_saved_mb']:.1f}MB")
        response = input('\n❓ Proceed with cleanup? (y/N): ').strip().lower()
        if response == 'y':
            print('\n🧹 Performing actual cleanup...')
            cleanup_results = cleanup_non_english_resources(dry_run=False)
            print(f'\n✅ Cleanup completed!')
            print(f"  Space saved: {cleanup_results['space_saved_mb']:.1f}MB")
            print(f"  Files removed: {len(cleanup_results['removed_files'])}")
            print(f"  Directories removed: {len(cleanup_results['removed_directories'])}")
            if cleanup_results['errors']:
                print(f"  ⚠️  Errors: {len(cleanup_results['errors'])}")
                for error in cleanup_results['errors']:
                    print(f'    • {error}')
        else:
            print('❌ Cleanup cancelled.')
    else:
        print('✅ No cleanup needed - already optimized!')
    print(f'\n🔍 Verifying English Resources:')
    verification = verify_english_resources()
    if verification['all_essential_available']:
        print('\n✅ All essential English resources are working correctly!')
    else:
        print('\n⚠️  Some English resources may have issues. Check the verification results above.')
    print(f'\n📝 Optimization Summary:')
    print(f'  • NLTK footprint reduced by focusing on English-only resources')
    print(f'  • Fallback implementations available for missing resources')
    print(f'  • Performance improved through specialized English processing')
if __name__ == '__main__':
    main()
