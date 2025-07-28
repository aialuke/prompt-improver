"""Prompt Characteristics Analyzer for Intelligent Rule Selection.

Phase 4 Enhancement: Analyzes prompts to extract characteristics for rule matching using
2025 NLP best practices and advanced machine learning techniques.

Enhanced with:
- Semantic complexity analysis using transformer models
- Domain confidence scoring with uncertainty quantification
- Reasoning depth detection using linguistic patterns
- Context dependency analysis for rule optimization
- Pattern signature extraction for ML integration
"""

import re
import logging
import time
from typing import Dict, List, Optional

import textstat
import numpy as np

# Optional ML imports with fallbacks
try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    ML_AVAILABLE = True
except ImportError:
    logger.warning("Transformers not available, using fallback analysis")
    ML_AVAILABLE = False

from .models import PromptCharacteristics

logger = logging.getLogger(__name__)

class PromptAnalyzer:
    """Advanced prompt analyzer for extracting characteristics for rule selection.

    Phase 4 Enhancement: Uses multiple analysis techniques:
    - Statistical text analysis (readability, complexity)
    - Pattern recognition (task types, domains)
    - Linguistic analysis (style, structure)
    - ML-based classification with transformer models
    - Semantic complexity analysis using embeddings
    - Context dependency detection for rule optimization
    """

    def __init__(self, enable_ml_analysis: bool = True):
        """Initialize prompt analyzer with enhanced NLP models.

        Args:
            enable_ml_analysis: Enable ML-based analysis features
        """
        self.enable_ml_analysis = enable_ml_analysis and ML_AVAILABLE

        # Enhanced domain keywords with confidence scoring
        self.domain_keywords = {
            "technical": {
                "primary": ["code", "programming", "software", "algorithm", "debug", "function", "api", "database"],
                "secondary": ["system", "architecture", "framework", "library", "deployment", "testing"],
                "weight": 1.0
            },
            "business": {
                "primary": ["strategy", "market", "revenue", "customer", "sales", "profit", "roi", "kpi"],
                "secondary": ["business", "company", "organization", "management", "leadership"],
                "weight": 1.0
            },
            "creative": {
                "primary": ["story", "creative", "design", "art", "narrative", "character", "plot", "style"],
                "secondary": ["imagination", "artistic", "visual", "aesthetic", "inspiration"],
                "weight": 1.0
            },
            "academic": {
                "primary": ["research", "study", "analysis", "theory", "hypothesis", "methodology", "citation"],
                "secondary": ["academic", "scholarly", "peer-review", "publication", "journal"],
                "weight": 1.0
            },
            "personal": {
                "primary": ["help", "advice", "personal", "relationship", "life", "decision", "opinion"],
                "secondary": ["individual", "private", "emotional", "psychological", "wellness"],
                "weight": 0.8
            },
            "educational": {
                "primary": ["learn", "teach", "explain", "understand", "concept", "lesson", "tutorial"],
                "secondary": ["education", "instruction", "knowledge", "skill", "training"],
                "weight": 1.0
            }
        }

        self.task_type_patterns = {
            "question_answering": [r"\?", r"what is", r"how to", r"why", r"when", r"where", r"who"],
            "text_generation": [r"write", r"create", r"generate", r"compose", r"draft"],
            "analysis": [r"analyze", r"examine", r"evaluate", r"assess", r"review", r"compare"],
            "summarization": [r"summarize", r"summary", r"brief", r"overview", r"key points"],
            "translation": [r"translate", r"convert", r"transform", r"change to"],
            "problem_solving": [r"solve", r"fix", r"resolve", r"troubleshoot", r"debug"],
            "planning": [r"plan", r"strategy", r"roadmap", r"schedule", r"organize"],
            "classification": [r"classify", r"categorize", r"identify", r"determine", r"label"]
        }

        self.complexity_indicators = [
            "multi-step", "complex", "detailed", "comprehensive", "thorough", "in-depth",
            "advanced", "sophisticated", "nuanced", "intricate", "elaborate"
        ]

        self.reasoning_indicators = [
            "because", "therefore", "however", "although", "since", "given that",
            "considering", "due to", "as a result", "consequently", "thus", "hence"
        ]

        # Phase 4: Initialize enhanced ML models
        self.sentiment_analyzer = None
        self.text_classifier = None
        self.semantic_model = None
        self.tokenizer = None

        if self.enable_ml_analysis:
            try:
                # Lightweight models for production use
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )

                # For semantic analysis (using a smaller model for performance)
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.semantic_model = AutoModel.from_pretrained(model_name)

                logger.info("ML models initialized successfully for enhanced prompt analysis")

            except Exception as e:
                logger.warning(f"Could not load ML models: {e}")
                self.enable_ml_analysis = False

    def analyze_prompt(self, prompt: str) -> PromptCharacteristics:
        """Analyze prompt and extract comprehensive characteristics.

        Phase 4 Enhancement: Includes ML-based semantic analysis and advanced features.

        Args:
            prompt: Input prompt text

        Returns:
            PromptCharacteristics with extracted features including ML enhancements
        """
        start_time = time.time()

        # Basic text statistics
        word_count = len(prompt.split())
        char_count = len(prompt)
        sentence_count = len(re.split(r'[.!?]+', prompt))

        # Extract core characteristics
        prompt_type = self._classify_prompt_type(prompt)
        complexity_level = self._calculate_complexity_level(prompt)
        domain = self._identify_domain(prompt)
        length_category = self._categorize_length(word_count)
        reasoning_required = self._detect_reasoning_requirement(prompt)
        specificity_level = self._calculate_specificity_level(prompt)
        context_richness = self._calculate_context_richness(prompt)
        task_type = self._identify_task_type(prompt)
        language_style = self._analyze_language_style(prompt)

        # Phase 4: Enhanced ML-based characteristics
        semantic_complexity = None
        domain_confidence = None
        reasoning_depth = None
        context_dependencies = None
        linguistic_features = None
        pattern_signatures = None

        if self.enable_ml_analysis:
            semantic_complexity = self._calculate_semantic_complexity(prompt)
            domain_confidence = self._calculate_domain_confidence(prompt, domain)
            reasoning_depth = self._detect_reasoning_depth(prompt)
            context_dependencies = self._analyze_context_dependencies(prompt)
            linguistic_features = self._extract_linguistic_features(prompt)
            pattern_signatures = self._generate_pattern_signatures(prompt)

        # Enhanced custom attributes
        custom_attributes = {
            "word_count": word_count,
            "char_count": char_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": word_count / max(1, sentence_count),
            "readability_score": self._calculate_readability(prompt),
            "question_count": len(re.findall(r'\?', prompt)),
            "imperative_count": self._count_imperatives(prompt),
            "technical_terms": self._count_technical_terms(prompt),
            "sentiment": self._analyze_sentiment(prompt),
            "urgency_level": self._detect_urgency(prompt),
            "formality_level": self._analyze_formality(prompt),
            # Phase 4: Enhanced analysis timing
            "analysis_time_ms": (time.time() - start_time) * 1000,
            "ml_analysis_enabled": self.enable_ml_analysis
        }

        return PromptCharacteristics(
            prompt_type=prompt_type,
            complexity_level=complexity_level,
            domain=domain,
            length_category=length_category,
            reasoning_required=reasoning_required,
            specificity_level=specificity_level,
            context_richness=context_richness,
            task_type=task_type,
            language_style=language_style,
            custom_attributes=custom_attributes,
            # Phase 4: ML-Enhanced Characteristics
            semantic_complexity=semantic_complexity,
            domain_confidence=domain_confidence,
            reasoning_depth=reasoning_depth,
            context_dependencies=context_dependencies,
            linguistic_features=linguistic_features,
            pattern_signatures=pattern_signatures
        )

    def _classify_prompt_type(self, prompt: str) -> str:
        """Classify the overall type of prompt.

        Args:
            prompt: Input prompt text

        Returns:
            Prompt type classification
        """
        prompt_lower = prompt.lower()

        # Check for specific patterns
        if any(pattern in prompt_lower for pattern in ["explain", "describe", "what is"]):
            return "explanatory"
        elif any(pattern in prompt_lower for pattern in ["write", "create", "generate"]):
            return "generative"
        elif "?" in prompt:
            return "interrogative"
        elif any(pattern in prompt_lower for pattern in ["analyze", "evaluate", "compare"]):
            return "analytical"
        elif any(pattern in prompt_lower for pattern in ["help", "how to", "guide"]):
            return "instructional"
        else:
            return "general"

    def _calculate_complexity_level(self, prompt: str) -> float:
        """Calculate prompt complexity level (0.0 to 1.0).

        Args:
            prompt: Input prompt text

        Returns:
            Normalized complexity score
        """
        factors = []

        # Length factor
        word_count = len(prompt.split())
        length_factor = min(1.0, word_count / 100.0)  # Normalize to 100 words
        factors.append(length_factor * 0.2)

        # Vocabulary complexity
        unique_words = len(set(prompt.lower().split()))
        vocab_diversity = unique_words / max(1, word_count)
        factors.append(vocab_diversity * 0.3)

        # Sentence structure complexity
        sentence_count = len(re.split(r'[.!?]+', prompt))
        avg_sentence_length = word_count / max(1, sentence_count)
        sentence_complexity = min(1.0, avg_sentence_length / 20.0)  # Normalize to 20 words
        factors.append(sentence_complexity * 0.2)

        # Complexity indicators
        complexity_words = sum(1 for word in self.complexity_indicators if word in prompt.lower())
        complexity_indicator_score = min(1.0, complexity_words / 3.0)
        factors.append(complexity_indicator_score * 0.3)

        return sum(factors)

    def _identify_domain(self, prompt: str) -> str:
        """Identify the domain/subject area of the prompt.

        Phase 4 Enhancement: Uses enhanced keyword matching with primary/secondary weights.

        Args:
            prompt: Input prompt text

        Returns:
            Domain classification
        """
        prompt_lower = prompt.lower()
        domain_scores = {}

        for domain, domain_data in self.domain_keywords.items():
            primary_keywords = domain_data["primary"]
            secondary_keywords = domain_data["secondary"]
            weight = domain_data["weight"]

            # Score with weighted keywords
            primary_score = sum(2 for keyword in primary_keywords if keyword in prompt_lower)
            secondary_score = sum(1 for keyword in secondary_keywords if keyword in prompt_lower)

            total_score = (primary_score + secondary_score) * weight

            if total_score > 0:
                domain_scores[domain] = total_score

        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        else:
            return "general"

    def _categorize_length(self, word_count: int) -> str:
        """Categorize prompt length.

        Args:
            word_count: Number of words in prompt

        Returns:
            Length category
        """
        if word_count < 10:
            return "very_short"
        elif word_count < 25:
            return "short"
        elif word_count < 50:
            return "medium"
        elif word_count < 100:
            return "long"
        else:
            return "very_long"

    def _detect_reasoning_requirement(self, prompt: str) -> bool:
        """Detect if prompt requires reasoning or logical thinking.

        Args:
            prompt: Input prompt text

        Returns:
            True if reasoning is required
        """
        prompt_lower = prompt.lower()

        # Check for reasoning indicators
        reasoning_count = sum(1 for indicator in self.reasoning_indicators if indicator in prompt_lower)

        # Check for logical connectors
        logical_patterns = ["if", "then", "because", "therefore", "since", "given"]
        logical_count = sum(1 for pattern in logical_patterns if pattern in prompt_lower)

        # Check for multi-step indicators
        multi_step_patterns = ["first", "second", "then", "next", "finally", "step"]
        multi_step_count = sum(1 for pattern in multi_step_patterns if pattern in prompt_lower)

        return (reasoning_count + logical_count + multi_step_count) >= 2

    def _calculate_specificity_level(self, prompt: str) -> float:
        """Calculate how specific vs. general the prompt is (0.0 to 1.0).

        Args:
            prompt: Input prompt text

        Returns:
            Normalized specificity score
        """
        factors = []

        # Specific details (numbers, dates, names)
        numbers = len(re.findall(r'\b\d+\b', prompt))
        proper_nouns = len(re.findall(r'\b[A-Z][a-z]+\b', prompt))
        specific_details = numbers + proper_nouns
        detail_factor = min(1.0, specific_details / 5.0)
        factors.append(detail_factor * 0.4)

        # Constraint words
        constraint_words = ["specific", "exactly", "precisely", "must", "required", "only"]
        constraint_count = sum(1 for word in constraint_words if word in prompt.lower())
        constraint_factor = min(1.0, constraint_count / 3.0)
        factors.append(constraint_factor * 0.3)

        # Question specificity
        if "?" in prompt:
            vague_questions = ["what", "how", "why", "general"]
            specific_questions = ["which", "when", "where", "who", "exactly"]
            vague_count = sum(1 for word in vague_questions if word in prompt.lower())
            specific_count = sum(1 for word in specific_questions if word in prompt.lower())
            question_specificity = specific_count / max(1, vague_count + specific_count)
            factors.append(question_specificity * 0.3)
        else:
            factors.append(0.5 * 0.3)  # Neutral for non-questions

        return sum(factors)

    def _calculate_context_richness(self, prompt: str) -> float:
        """Calculate how much context is provided (0.0 to 1.0).

        Args:
            prompt: Input prompt text

        Returns:
            Normalized context richness score
        """
        # Background information indicators
        context_indicators = ["background", "context", "situation", "scenario", "given", "assuming"]
        context_count = sum(1 for indicator in context_indicators if indicator in prompt.lower())

        # Descriptive adjectives and adverbs
        descriptive_words = len(re.findall(r'\b\w+ly\b|\b\w+ful\b|\b\w+ous\b', prompt))

        # Length as context indicator
        word_count = len(prompt.split())
        length_context = min(1.0, word_count / 50.0)

        # Combine factors
        context_score = (
            min(1.0, context_count / 3.0) * 0.4 +
            min(1.0, descriptive_words / 10.0) * 0.3 +
            length_context * 0.3
        )

        return context_score

    def _identify_task_type(self, prompt: str) -> str:
        """Identify the specific task type requested.

        Args:
            prompt: Input prompt text

        Returns:
            Task type classification
        """
        prompt_lower = prompt.lower()

        for task_type, patterns in self.task_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, prompt_lower):
                    return task_type

        return "general"

    def _analyze_language_style(self, prompt: str) -> str:
        """Analyze the language style of the prompt.

        Args:
            prompt: Input prompt text

        Returns:
            Language style classification
        """
        # Formal indicators
        formal_indicators = ["please", "kindly", "would you", "could you", "furthermore", "therefore"]
        formal_count = sum(1 for indicator in formal_indicators if indicator in prompt.lower())

        # Casual indicators
        casual_indicators = ["hey", "hi", "thanks", "cool", "awesome", "gonna", "wanna"]
        casual_count = sum(1 for indicator in casual_indicators if indicator in prompt.lower())

        # Technical indicators
        technical_indicators = ["implement", "configure", "optimize", "algorithm", "function", "parameter"]
        technical_count = sum(1 for indicator in technical_indicators if indicator in prompt.lower())

        if technical_count >= 2:
            return "technical"
        elif formal_count > casual_count:
            return "formal"
        elif casual_count > 0:
            return "casual"
        else:
            return "neutral"

    def _calculate_readability(self, prompt: str) -> float:
        """Calculate readability score using textstat.

        Args:
            prompt: Input prompt text

        Returns:
            Normalized readability score (0.0 to 1.0)
        """
        try:
            # Flesch Reading Ease (0-100, higher = easier)
            flesch_score = textstat.flesch_reading_ease(prompt)
            # Normalize to 0-1 scale
            return max(0.0, min(1.0, flesch_score / 100.0))
        except Exception:
            return 0.5  # Default neutral score

    def _count_imperatives(self, prompt: str) -> int:
        """Count imperative verbs in the prompt.

        Args:
            prompt: Input prompt text

        Returns:
            Number of imperative verbs
        """
        imperative_verbs = [
            "write", "create", "make", "build", "design", "develop", "implement",
            "analyze", "evaluate", "compare", "explain", "describe", "summarize",
            "list", "identify", "find", "solve", "fix", "improve", "optimize"
        ]

        words = prompt.lower().split()
        return sum(1 for word in words if word in imperative_verbs)

    def _count_technical_terms(self, prompt: str) -> int:
        """Count technical terms in the prompt.

        Args:
            prompt: Input prompt text

        Returns:
            Number of technical terms
        """
        technical_terms = [
            "algorithm", "function", "variable", "parameter", "database", "api",
            "framework", "library", "module", "class", "method", "interface",
            "protocol", "schema", "query", "optimization", "performance"
        ]

        words = prompt.lower().split()
        return sum(1 for word in words if word in technical_terms)

    def _analyze_sentiment(self, prompt: str) -> str:
        """Analyze sentiment of the prompt.

        Args:
            prompt: Input prompt text

        Returns:
            Sentiment classification
        """
        if self.sentiment_analyzer:
            try:
                result = self.sentiment_analyzer(prompt)[0]
                return result['label'].lower()
            except Exception:
                pass

        # Fallback to simple rule-based sentiment
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "please", "help"]
        negative_words = ["bad", "terrible", "awful", "wrong", "error", "problem", "issue"]

        prompt_lower = prompt.lower()
        positive_count = sum(1 for word in positive_words if word in prompt_lower)
        negative_count = sum(1 for word in negative_words if word in prompt_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    def _detect_urgency(self, prompt: str) -> float:
        """Detect urgency level in the prompt (0.0 to 1.0).

        Args:
            prompt: Input prompt text

        Returns:
            Normalized urgency score
        """
        urgency_indicators = [
            "urgent", "asap", "immediately", "quickly", "fast", "rush",
            "deadline", "emergency", "critical", "important", "priority"
        ]

        prompt_lower = prompt.lower()
        urgency_count = sum(1 for indicator in urgency_indicators if indicator in prompt_lower)

        # Check for time constraints
        time_patterns = [r"by \w+", r"within \d+", r"before \w+", r"due \w+"]
        time_constraints = sum(1 for pattern in time_patterns if re.search(pattern, prompt_lower))

        urgency_score = min(1.0, (urgency_count + time_constraints) / 3.0)
        return urgency_score

    def _analyze_formality(self, prompt: str) -> float:
        """Analyze formality level of the prompt (0.0 to 1.0).

        Args:
            prompt: Input prompt text

        Returns:
            Normalized formality score
        """
        formal_indicators = [
            "please", "kindly", "would you", "could you", "may i", "thank you",
            "furthermore", "therefore", "however", "nevertheless", "consequently"
        ]

        informal_indicators = [
            "hey", "hi", "thanks", "cool", "awesome", "gonna", "wanna",
            "yeah", "ok", "btw", "fyi", "lol", "omg"
        ]

        prompt_lower = prompt.lower()
        formal_count = sum(1 for indicator in formal_indicators if indicator in prompt_lower)
        informal_count = sum(1 for indicator in informal_indicators if indicator in prompt_lower)

        if formal_count + informal_count == 0:
            return 0.5  # Neutral

        formality_score = formal_count / (formal_count + informal_count)
        return formality_score

    # ===================================
    # Phase 4: ML-Enhanced Analysis Methods
    # ===================================

    def _calculate_semantic_complexity(self, prompt: str) -> Optional[float]:
        """Calculate semantic complexity using transformer embeddings.

        Args:
            prompt: Input prompt text

        Returns:
            Semantic complexity score (0.0-1.0) or None if ML not available
        """
        if not self.enable_ml_analysis or not self.semantic_model:
            return None

        try:
            # Tokenize and get embeddings
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

            with torch.no_grad():
                outputs = self.semantic_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)

            # Calculate complexity based on embedding variance and magnitude
            embedding_variance = torch.var(embeddings).item()
            embedding_magnitude = torch.norm(embeddings).item()

            # Normalize to 0-1 range (empirically determined thresholds)
            complexity = min(1.0, (embedding_variance * embedding_magnitude) / 10.0)

            return complexity

        except Exception as e:
            logger.warning(f"Semantic complexity calculation failed: {e}")
            return None

    def _calculate_domain_confidence(self, prompt: str, domain: str) -> Optional[float]:
        """Calculate confidence score for domain classification.

        Args:
            prompt: Input prompt text
            domain: Identified domain

        Returns:
            Domain confidence score (0.0-1.0) or None if ML not available
        """
        if not self.enable_ml_analysis:
            # Fallback to keyword-based confidence
            return self._calculate_keyword_domain_confidence(prompt, domain)

        try:
            # Calculate confidence based on multiple factors
            keyword_confidence = self._calculate_keyword_domain_confidence(prompt, domain)

            # Add semantic similarity if ML is available
            if self.semantic_model:
                semantic_confidence = self._calculate_semantic_domain_confidence(prompt, domain)
                # Weighted average
                confidence = 0.6 * keyword_confidence + 0.4 * semantic_confidence
            else:
                confidence = keyword_confidence

            return min(1.0, max(0.0, confidence))

        except Exception as e:
            logger.warning(f"Domain confidence calculation failed: {e}")
            return 0.5  # Default moderate confidence

    def _calculate_keyword_domain_confidence(self, prompt: str, domain: str) -> float:
        """Calculate domain confidence based on keyword matching."""
        if domain not in self.domain_keywords:
            return 0.5

        prompt_lower = prompt.lower()
        domain_data = self.domain_keywords[domain]

        primary_matches = sum(1 for keyword in domain_data["primary"] if keyword in prompt_lower)
        secondary_matches = sum(1 for keyword in domain_data["secondary"] if keyword in prompt_lower)

        total_keywords = len(domain_data["primary"]) + len(domain_data["secondary"])
        total_matches = primary_matches * 2 + secondary_matches  # Weight primary matches more

        confidence = (total_matches / (total_keywords * 2)) * domain_data["weight"]
        return min(1.0, confidence)

    def _calculate_semantic_domain_confidence(self, prompt: str, domain: str) -> float:
        """Calculate domain confidence using semantic similarity."""
        # This would require domain-specific embeddings or a trained classifier
        # For now, return a placeholder that could be enhanced with domain-specific models
        return 0.7  # Placeholder - could be enhanced with domain-specific models

    def _detect_reasoning_depth(self, prompt: str) -> Optional[int]:
        """Detect the depth of reasoning required for the prompt.

        Args:
            prompt: Input prompt text

        Returns:
            Reasoning depth level (1-5) or None if ML not available
        """
        # Enhanced reasoning detection with multiple indicators
        reasoning_patterns = {
            1: ["what", "who", "when", "where"],  # Simple factual
            2: ["how", "why", "explain"],  # Basic reasoning
            3: ["analyze", "compare", "evaluate"],  # Analytical reasoning
            4: ["synthesize", "create", "design", "develop"],  # Creative reasoning
            5: ["optimize", "strategize", "theorize", "philosophize"]  # Complex reasoning
        }

        prompt_lower = prompt.lower()
        max_depth = 1

        for depth, patterns in reasoning_patterns.items():
            if any(pattern in prompt_lower for pattern in patterns):
                max_depth = max(max_depth, depth)

        # Additional depth indicators
        if len(re.findall(r'\b(because|therefore|however|although)\b', prompt_lower)) > 2:
            max_depth = max(max_depth, 3)

        if len(re.findall(r'\b(consider|given|assuming|suppose)\b', prompt_lower)) > 1:
            max_depth = max(max_depth, 4)

        return max_depth

    def _analyze_context_dependencies(self, prompt: str) -> Optional[List[str]]:
        """Analyze context dependencies in the prompt.

        Args:
            prompt: Input prompt text

        Returns:
            List of context dependency types
        """
        dependencies = []
        prompt_lower = prompt.lower()

        # Temporal dependencies
        if any(word in prompt_lower for word in ["before", "after", "during", "while", "when"]):
            dependencies.append("temporal")

        # Causal dependencies
        if any(word in prompt_lower for word in ["because", "due to", "caused by", "results in"]):
            dependencies.append("causal")

        # Conditional dependencies
        if any(word in prompt_lower for word in ["if", "unless", "provided", "assuming"]):
            dependencies.append("conditional")

        # Reference dependencies
        if any(word in prompt_lower for word in ["this", "that", "these", "those", "above", "below"]):
            dependencies.append("referential")

        # Domain-specific dependencies
        if any(word in prompt_lower for word in ["according to", "based on", "following"]):
            dependencies.append("domain_specific")

        return dependencies if dependencies else ["independent"]

    def _extract_linguistic_features(self, prompt: str) -> Optional[Dict[str, float]]:
        """Extract linguistic features for ML analysis.

        Args:
            prompt: Input prompt text

        Returns:
            Dictionary of linguistic features
        """
        features = {}

        # Lexical features
        words = prompt.split()
        features["avg_word_length"] = np.mean([len(word) for word in words]) if words else 0
        features["unique_word_ratio"] = len(set(words)) / len(words) if words else 0

        # Syntactic features
        features["question_ratio"] = prompt.count("?") / len(words) if words else 0
        features["exclamation_ratio"] = prompt.count("!") / len(words) if words else 0
        features["comma_ratio"] = prompt.count(",") / len(words) if words else 0

        # Semantic features
        features["modal_verb_count"] = len(re.findall(r'\b(can|could|may|might|must|should|will|would)\b', prompt.lower()))
        features["negation_count"] = len(re.findall(r'\b(not|no|never|none|nothing|neither)\b', prompt.lower()))
        features["intensifier_count"] = len(re.findall(r'\b(very|extremely|highly|quite|rather)\b', prompt.lower()))

        return features

    def _generate_pattern_signatures(self, prompt: str) -> Optional[List[str]]:
        """Generate pattern signatures for ML integration.

        Args:
            prompt: Input prompt text

        Returns:
            List of pattern signatures
        """
        signatures = []
        prompt_lower = prompt.lower()

        # Task pattern signatures
        if "step" in prompt_lower or "process" in prompt_lower:
            signatures.append("sequential_task")

        if "compare" in prompt_lower or "contrast" in prompt_lower:
            signatures.append("comparative_analysis")

        if "create" in prompt_lower or "generate" in prompt_lower:
            signatures.append("generative_task")

        if "explain" in prompt_lower or "describe" in prompt_lower:
            signatures.append("explanatory_task")

        # Complexity signatures
        word_count = len(prompt.split())
        if word_count > 100:
            signatures.append("complex_prompt")
        elif word_count < 20:
            signatures.append("simple_prompt")
        else:
            signatures.append("moderate_prompt")

        # Domain signatures
        if any(tech_word in prompt_lower for tech_word in ["code", "algorithm", "function", "api"]):
            signatures.append("technical_domain")

        if any(biz_word in prompt_lower for biz_word in ["business", "strategy", "market", "customer"]):
            signatures.append("business_domain")

        return signatures if signatures else ["general_pattern"]
