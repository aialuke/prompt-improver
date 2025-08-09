"""English-Only NLTK Resource Manager

Optimized NLTK resource management for English-only processing.
Reduces footprint and improves performance by only loading English resources.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
logger = logging.getLogger(__name__)

class EnglishNLTKManager:
    """Lightweight NLTK manager for English-only processing.
    
    This manager:
    - Only loads English language resources
    - Provides fallback implementations when resources are unavailable
    - Optimizes for local machine deployment
    - Handles SSL certificate issues gracefully
    """

    def __init__(self):
        """Initialize English-only NLTK manager."""
        self.logger = logging.getLogger(__name__)
        self._resources_checked = False
        self._available_resources: Set[str] = set()
        self.english_resources = {'punkt', 'stopwords', 'wordnet'}
        self.optional_resources = {'averaged_perceptron_tagger', 'vader_lexicon'}
        self._tokenizer = None
        self._stopwords = None
        self._lemmatizer = None
        self._check_resources()

    def _check_resources(self) -> None:
        """Check which English resources are available."""
        if self._resources_checked:
            return
        self.logger.info('Checking English NLTK resources...')
        for resource in self.english_resources:
            if self._is_resource_available(resource):
                self._available_resources.add(resource)
                self.logger.debug('✅ %s available', resource)
            else:
                self.logger.warning('❌ %s not available', resource)
        for resource in self.optional_resources:
            if self._is_resource_available(resource):
                self._available_resources.add(resource)
                self.logger.debug('✅ %s (optional) available', resource)
        self._resources_checked = True
        available_count = len(self._available_resources)
        total_count = len(self.english_resources) + len(self.optional_resources)
        self.logger.info('NLTK resources: {available_count}/%s available', total_count)

    def _is_resource_available(self, resource: str) -> bool:
        """Check if a specific English resource is available."""
        try:
            if resource == 'punkt':
                try:
                    nltk.data.find('tokenizers/punkt')
                    return True
                except LookupError:
                    try:
                        nltk.data.find('tokenizers/punkt_tab/english')
                        return True
                    except LookupError:
                        return False
            elif resource == 'stopwords':
                try:
                    nltk.data.find('corpora/stopwords')
                    from nltk.corpus import stopwords
                    stopwords.words('english')
                    return True
                except (LookupError, OSError):
                    return False
            elif resource == 'wordnet':
                try:
                    nltk.data.find('corpora/wordnet')
                    return True
                except LookupError:
                    return False
            elif resource == 'averaged_perceptron_tagger':
                try:
                    nltk.data.find('taggers/averaged_perceptron_tagger')
                    return True
                except LookupError:
                    return False
            elif resource == 'vader_lexicon':
                try:
                    nltk.data.find('vader_lexicon')
                    return True
                except LookupError:
                    try:
                        nltk.data.find('sentiment/vader_lexicon')
                        return True
                    except LookupError:
                        return False
            return False
        except Exception as e:
            self.logger.debug('Error checking {resource}: %s', e)
            return False

    def get_sentence_tokenizer(self):
        """Get English sentence tokenizer with fallback."""
        if self._tokenizer is not None:
            return self._tokenizer
        if 'punkt' in self._available_resources:
            try:
                self._tokenizer = lambda text: sent_tokenize(text, language='english')
                self.logger.debug('Using NLTK punkt tokenizer')
                return self._tokenizer
            except Exception as e:
                self.logger.warning('NLTK punkt tokenizer failed: %s', e)
        import re

        def simple_sentence_tokenizer(text: str) -> List[str]:
            """Simple regex-based sentence tokenizer."""
            sentences = re.split('[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            return sentences
        self._tokenizer = simple_sentence_tokenizer
        self.logger.debug('Using fallback regex sentence tokenizer')
        return self._tokenizer

    def get_word_tokenizer(self):
        """Get English word tokenizer with fallback."""
        if 'punkt' in self._available_resources:
            try:
                return lambda text: word_tokenize(text, language='english')
            except Exception as e:
                self.logger.warning('NLTK word tokenizer failed: %s', e)
        import re

        def simple_word_tokenizer(text: str) -> List[str]:
            """Simple regex-based word tokenizer."""
            words = re.findall('\\b\\w+\\b', text.lower())
            return words
        self.logger.debug('Using fallback regex word tokenizer')
        return simple_word_tokenizer

    def get_english_stopwords(self) -> Set[str]:
        """Get English stopwords with fallback."""
        if self._stopwords is not None:
            return self._stopwords
        if 'stopwords' in self._available_resources:
            try:
                from nltk.corpus import stopwords
                self._stopwords = set(stopwords.words('english'))
                self.logger.debug('Using NLTK English stopwords')
                return self._stopwords
            except Exception as e:
                self.logger.warning('NLTK stopwords failed: %s', e)
        fallback_stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'}
        self._stopwords = fallback_stopwords
        self.logger.debug('Using fallback English stopwords')
        return self._stopwords

    def get_lemmatizer(self):
        """Get English lemmatizer with fallback."""
        if self._lemmatizer is not None:
            return self._lemmatizer
        if 'wordnet' in self._available_resources:
            try:
                self._lemmatizer = WordNetLemmatizer()
                self.logger.debug('Using NLTK WordNet lemmatizer')
                return self._lemmatizer
            except Exception as e:
                self.logger.warning('NLTK lemmatizer failed: %s', e)

        def simple_lemmatizer(word: str, pos: str='n') -> str:
            """Simple rule-based lemmatizer."""
            word = word.lower()
            if pos == 'n' and word.endswith('s') and (len(word) > 3):
                if word.endswith('ies'):
                    return word[:-3] + 'y'
                elif word.endswith('es'):
                    return word[:-2]
                else:
                    return word[:-1]
            if pos == 'v':
                if word.endswith('ed') and len(word) > 4:
                    return word[:-2]
                elif word.endswith('ing') and len(word) > 5:
                    return word[:-3]
            return word
        self._lemmatizer = type('SimpleLemmatizer', (), {'lemmatize': lambda self, word, pos='n': simple_lemmatizer(word, pos)})()
        self.logger.debug('Using fallback simple lemmatizer')
        return self._lemmatizer

    def get_resource_status(self) -> Dict[str, any]:
        """Get status of English NLTK resources."""
        return {'available_resources': list(self._available_resources), 'required_available': len(self._available_resources & self.english_resources), 'required_total': len(self.english_resources), 'optional_available': len(self._available_resources & self.optional_resources), 'optional_total': len(self.optional_resources), 'fallback_mode': len(self._available_resources & self.english_resources) < len(self.english_resources)}

    def is_fully_available(self) -> bool:
        """Check if all required English resources are available."""
        return self.english_resources.issubset(self._available_resources)

    def cleanup_unused_resources(self) -> Dict[str, int]:
        """Identify non-English resources that could be removed to save space."""
        cleanup_info = {'non_english_languages': 0, 'unused_corpora': 0, 'potential_savings_mb': 0}
        try:
            punkt_path = Path(nltk.data.find('tokenizers/punkt_tab'))
            if punkt_path.exists():
                non_english_dirs = [d for d in punkt_path.iterdir() if d.is_dir() and d.name != 'english']
                cleanup_info['non_english_languages'] = len(non_english_dirs)
                if non_english_dirs:
                    total_size = sum((sum((f.stat().st_size for f in d.rglob('*') if f.is_file())) for d in non_english_dirs))
                    cleanup_info['potential_savings_mb'] = total_size / (1024 * 1024)
        except Exception as e:
            self.logger.debug('Error calculating cleanup info: %s', e)
        return cleanup_info
_english_nltk_manager = None

def get_english_nltk_manager() -> EnglishNLTKManager:
    """Get singleton English NLTK manager."""
    global _english_nltk_manager
    if _english_nltk_manager is None:
        _english_nltk_manager = EnglishNLTKManager()
    return _english_nltk_manager
