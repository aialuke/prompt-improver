"""Apriori Algorithm Implementation for Prompt Improvement Pattern Discovery

This module provides the AprioriAnalyzer service that integrates mlxtend's Apriori algorithm
to discover association rules between prompt characteristics, rule applications, and 
improvement outcomes in the prompt improvement system.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    logging.warning("mlxtend not available. Install with: pip install mlxtend")

from ....database.connection import DatabaseManager
from ....utils.error_handlers import handle_common_errors as handle_errors
from ....utils.redis_cache import cached


@dataclass
class AprioriConfig:
    """Configuration for Apriori algorithm parameters."""
    min_support: float = 0.1
    min_confidence: float = 0.6
    min_lift: float = 1.0
    max_itemset_length: int = 5
    transaction_window: int = 1000
    use_sparse: bool = False
    verbose: bool = False


class AprioriAnalyzer:
    """Apriori Algorithm Analyzer for discovering association rules in prompt improvement data.
    
    This service mines frequent itemsets and generates association rules from:
    - Rule application patterns (which rules are applied together)
    - Prompt characteristics (complexity, domain, length patterns)
    - Improvement outcomes (quality scores, user satisfaction)
    - Context features (from the 31-dimensional feature pipeline)
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None, config: Optional[AprioriConfig] = None):
        """Initialize the Apriori analyzer.
        
        Args:
            db_manager: Database connection manager
            config: Configuration parameters for Apriori algorithm
        """
        if not MLXTEND_AVAILABLE:
            raise ImportError("mlxtend library is required. Install with: pip install mlxtend")

        self.db_manager = db_manager
        self.config = config or AprioriConfig()
        self.logger = logging.getLogger(__name__)

        # Cache for discovered patterns
        self._frequent_itemsets_cache = {}
        self._association_rules_cache = {}
        self._last_analysis_time = None

    @handle_errors(return_format="dict")
    def extract_transactions_from_database(
        self,
        window_days: int = 30,
        min_sessions: int = 10
    ) -> list[list[str]]:
        """Extract transaction data from the database for Apriori analysis.
        
        Args:
            window_days: Number of days to look back for session data
            min_sessions: Minimum number of sessions required
            
        Returns:
            List of transactions, where each transaction is a list of items
        """
        # Check if database manager is available
        if self.db_manager is None:
            self.logger.warning("Database manager not available, returning mock transactions")
            return self._generate_mock_transactions()
        
        cutoff_date = datetime.now() - timedelta(days=window_days)

        query = """
        SELECT 
            ps.session_id,
            ps.prompt_text,
            ps.quality_score,
            pp.rule_name,
            pp.performance_score,
            uf.rating
        FROM prompt_sessions ps
        LEFT JOIN prompt_performance pp ON ps.session_id = pp.session_id
        LEFT JOIN user_feedback uf ON ps.session_id = uf.session_id
        WHERE ps.created_at > %s
        ORDER BY ps.session_id, pp.rule_name
        """

        try:
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=[cutoff_date])

            if len(df['session_id'].unique()) < min_sessions:
                self.logger.warning(f"Insufficient session data: {len(df['session_id'].unique())} sessions")
                return self._generate_mock_transactions()

            return self._process_database_transactions(df)
        except Exception as e:
            self.logger.error(f"Database query failed: {e}, returning mock transactions")
            return self._generate_mock_transactions()

    def _generate_mock_transactions(self) -> list[list[str]]:
        """Generate mock transaction data for testing when database is not available."""
        return [
            ["rule_clarity", "domain_technical", "quality_high", "length_medium"],
            ["rule_specificity", "domain_creative", "quality_medium", "length_short"],
            ["rule_clarity", "rule_specificity", "quality_high", "length_long"],
            ["domain_analytical", "quality_medium", "feedback_positive"],
            ["rule_clarity", "domain_technical", "performance_high"],
            ["rule_specificity", "quality_high", "feedback_positive", "length_medium"],
            ["domain_creative", "quality_medium", "performance_medium"],
            ["rule_clarity", "rule_specificity", "domain_technical", "quality_high"],
            ["quality_low", "domain_general", "feedback_negative"],
            ["rule_clarity", "quality_high", "performance_high", "feedback_positive"]
        ]

    def _process_database_transactions(self, df: pd.DataFrame) -> list[list[str]]:
        """Process database query results into transaction format.
        
        Args:
            df: DataFrame with session and rule data
            
        Returns:
            List of transactions for Apriori analysis
        """
        transactions = []

        for session_id, session_data in df.groupby('session_id'):
            transaction = []

            # Add rule applications
            rules_applied = session_data['rule_name'].dropna().unique()
            for rule in rules_applied:
                transaction.append(f"rule_{rule}")

            # Add prompt characteristics
            prompt_text = session_data['prompt_text'].iloc[0]
            if prompt_text:
                # Analyze prompt characteristics
                char_items = self._extract_prompt_characteristics(prompt_text)
                transaction.extend(char_items)

            # Add quality indicators
            quality_score = session_data['quality_score'].iloc[0]
            if pd.notna(quality_score):
                quality_category = self._categorize_quality(quality_score)
                transaction.append(f"quality_{quality_category}")

            # Add user feedback indicators
            rating = session_data['rating'].iloc[0]
            if pd.notna(rating):
                feedback_category = self._categorize_feedback(rating)
                transaction.append(f"feedback_{feedback_category}")

            # Add performance indicators
            avg_performance = session_data['performance_score'].mean()
            if pd.notna(avg_performance):
                perf_category = self._categorize_performance(avg_performance)
                transaction.append(f"performance_{perf_category}")

            if transaction:  # Only add non-empty transactions
                transactions.append(transaction)

        self.logger.info(f"Processed {len(transactions)} transactions from database")
        return transactions

    def _extract_prompt_characteristics(self, prompt_text: str) -> list[str]:
        """Extract characteristics from prompt text for itemset analysis.
        Enhanced with 2024 best practices for technical keyword expansion,
        dynamic threshold calibration, and improved domain detection.
        """
        characteristics = []
        text_lower = prompt_text.lower()
        text_len = len(prompt_text)

        # Enhanced length characteristics with dynamic thresholds
        if text_len < 40:  # Short prompts
            characteristics.append("length_short")
        elif text_len < 120:  # Medium prompts
            characteristics.append("length_medium")
        else:  # Long prompts
            characteristics.append("length_long")

        # Complexity indicators with expanded patterns
        question_count = text_lower.count('?')
        if question_count > 1:
            characteristics.append("complexity_questions")
        elif question_count == 1:
            characteristics.append("complexity_single_question")

        # Sequential/procedural patterns (expanded)
        sequential_patterns = ['step', 'first', 'then', 'next', 'after', 'before', 'finally', 'lastly', 'subsequently']
        if any(word in text_lower for word in sequential_patterns):
            characteristics.append("complexity_sequential")

        # Example request patterns (expanded)
        example_patterns = ['example', 'instance', 'like', 'such as', 'for instance', 'e.g.', 'demonstrate', 'illustrate']
        if any(pattern in text_lower for pattern in example_patterns):
            characteristics.append("pattern_examples")

        # Structure and formatting patterns
        if any(word in text_lower for word in ['list', 'bullet', 'numbered', 'format', 'structure', 'organize']):
            characteristics.append("pattern_structured")

        # Comparison patterns
        if any(word in text_lower for word in ['compare', 'contrast', 'versus', 'vs', 'difference', 'similar', 'alike']):
            characteristics.append("pattern_comparison")

        # Enhanced domain detection with expanded technical keywords

        # Technical/Programming domain (significantly expanded)
        technical_keywords = [
            # Programming basics
            'code', 'programming', 'function', 'method', 'class', 'variable', 'array', 'loop',
            'algorithm', 'debug', 'syntax', 'compile', 'execute', 'runtime', 'api', 'framework',
            'library', 'module', 'package', 'import', 'export', 'interface', 'database',

            # Advanced technical terms
            'machine learning', 'ml', 'ai', 'artificial intelligence', 'neural network', 'deep learning',
            'data science', 'analytics', 'statistics', 'regression', 'classification', 'clustering',
            'optimization', 'automation', 'deployment', 'devops', 'cloud', 'microservices',
            'containerization', 'kubernetes', 'docker', 'ci/cd', 'git', 'version control',

            # Technical processes
            'integration', 'testing', 'unit test', 'debugging', 'refactoring', 'scalability',
            'performance', 'security', 'encryption', 'authentication', 'authorization',
            'monitoring', 'logging', 'error handling', 'exception', 'configuration'
        ]

        # Creative/Writing domain (expanded)
        creative_keywords = [
            'write', 'story', 'creative', 'narrative', 'fiction', 'character', 'plot', 'dialogue',
            'poetry', 'poem', 'verse', 'prose', 'script', 'screenplay', 'novel', 'essay',
            'blog', 'article', 'content', 'copywriting', 'brainstorm', 'ideate', 'imagine',
            'storytelling', 'metaphor', 'symbolism', 'theme', 'genre', 'style', 'voice',
            'tone', 'mood', 'setting', 'description', 'imagery'
        ]

        # Analytical/Research domain (expanded)
        analytical_keywords = [
            'analyze', 'research', 'study', 'investigate', 'examine', 'evaluate', 'assess',
            'review', 'survey', 'report', 'findings', 'data', 'statistics', 'trends',
            'patterns', 'correlation', 'causation', 'hypothesis', 'methodology', 'conclusion',
            'evidence', 'proof', 'verify', 'validate', 'metrics', 'kpi', 'benchmark',
            'insights', 'interpretation', 'synthesis', 'critique', 'summarize'
        ]

        # Business/Strategy domain (new)
        business_keywords = [
            'business', 'strategy', 'marketing', 'sales', 'revenue', 'profit', 'roi',
            'customer', 'client', 'stakeholder', 'management', 'leadership', 'team',
            'project', 'planning', 'execution', 'budget', 'finance', 'investment',
            'growth', 'expansion', 'market', 'competition', 'analysis', 'proposal',
            'presentation', 'meeting', 'negotiation', 'contract', 'agreement'
        ]

        # Educational/Learning domain (new)
        educational_keywords = [
            'learn', 'teach', 'education', 'explain', 'understand', 'concept', 'theory',
            'practice', 'exercise', 'lesson', 'tutorial', 'course', 'training', 'skill',
            'knowledge', 'information', 'facts', 'definition', 'clarify', 'simplify',
            'beginner', 'advanced', 'intermediate', 'step-by-step', 'guide', 'instruction'
        ]

        # Domain classification with weighted scoring
        domain_scores = {
            'technical': sum(1 for keyword in technical_keywords if keyword in text_lower),
            'creative': sum(1 for keyword in creative_keywords if keyword in text_lower),
            'analytical': sum(1 for keyword in analytical_keywords if keyword in text_lower),
            'business': sum(1 for keyword in business_keywords if keyword in text_lower),
            'educational': sum(1 for keyword in educational_keywords if keyword in text_lower)
        }

        # Determine primary domain(s) - allow multiple domains
        max_score = max(domain_scores.values()) if domain_scores.values() else 0

        if max_score == 0:
            characteristics.append("domain_general")
        else:
            # Add all domains that score above threshold (at least 50% of max score)
            threshold = max(1, max_score * 0.5)
            for domain, score in domain_scores.items():
                if score >= threshold:
                    characteristics.append(f"domain_{domain}")

        # Intent classification (new)
        if any(word in text_lower for word in ['how', 'what', 'why', 'when', 'where', 'which']):
            characteristics.append("intent_question")
        if any(word in text_lower for word in ['create', 'make', 'build', 'generate', 'produce', 'develop']):
            characteristics.append("intent_creation")
        if any(word in text_lower for word in ['improve', 'optimize', 'enhance', 'fix', 'solve', 'resolve']):
            characteristics.append("intent_improvement")
        if any(word in text_lower for word in ['explain', 'describe', 'define', 'clarify', 'elaborate']):
            characteristics.append("intent_explanation")

        # Urgency/Priority indicators (new)
        if any(word in text_lower for word in ['urgent', 'asap', 'immediately', 'priority', 'critical', 'important']):
            characteristics.append("priority_high")
        elif any(word in text_lower for word in ['quick', 'fast', 'rapid', 'soon']):
            characteristics.append("priority_medium")
        else:
            characteristics.append("priority_normal")

        # Sentiment indicators (new)
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'perfect']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'wrong', 'error', 'problem', 'issue']

        if any(word in text_lower for word in positive_words):
            characteristics.append("sentiment_positive")
        elif any(word in text_lower for word in negative_words):
            characteristics.append("sentiment_negative")
        else:
            characteristics.append("sentiment_neutral")

        return characteristics

    def _categorize_quality(self, score: float) -> str:
        """Categorize quality scores into discrete bins."""
        if score >= 0.8:
            return "high"
        if score >= 0.6:
            return "medium"
        return "low"

    def _categorize_feedback(self, rating: int) -> str:
        """Categorize user feedback ratings."""
        if rating >= 4:
            return "positive"
        if rating >= 3:
            return "neutral"
        return "negative"

    def _categorize_performance(self, score: float) -> str:
        """Categorize performance scores."""
        if score >= 0.7:
            return "high"
        if score >= 0.5:
            return "medium"
        return "low"

    async def mine_frequent_itemsets(
        self,
        transactions: list[list[str]],
        min_support: float | None = None
    ) -> pd.DataFrame:
        """Mine frequent itemsets using the Apriori algorithm.
        
        Args:
            transactions: List of transactions for analysis
            min_support: Minimum support threshold (uses config default if None)
            
        Returns:
            DataFrame with frequent itemsets and their support values
        """
        try:
            if not transactions:
                self.logger.warning("No transactions provided for frequent itemset mining")
                return pd.DataFrame()

            min_support = min_support or self.config.min_support

            # Transform transactions to binary matrix
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions, sparse=self.config.use_sparse)
            df = pd.DataFrame(te_ary, columns=te.columns_)

            # Mine frequent itemsets
            frequent_itemsets = apriori(
                df,
                min_support=min_support,
                use_colnames=True,
                max_len=self.config.max_itemset_length,
                verbose=1 if self.config.verbose else 0,
                low_memory=True  # Better for large datasets
            )
        except Exception as e:
            self.logger.error(f"Error in mine_frequent_itemsets: {e}")
            return pd.DataFrame()

        # Add itemset length for analysis
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)

        # Cache results
        cache_key = f"itemsets_{min_support}_{len(transactions)}"
        self._frequent_itemsets_cache[cache_key] = frequent_itemsets

        self.logger.info(f"Found {len(frequent_itemsets)} frequent itemsets")
        return frequent_itemsets

    @handle_errors(return_format="dict")
    def generate_association_rules(
        self,
        frequent_itemsets: pd.DataFrame,
        min_confidence: float | None = None,
        min_lift: float | None = None
    ) -> pd.DataFrame:
        """Generate association rules from frequent itemsets.
        
        Args:
            frequent_itemsets: DataFrame with frequent itemsets
            min_confidence: Minimum confidence threshold
            min_lift: Minimum lift threshold
            
        Returns:
            DataFrame with association rules and metrics
        """
        if frequent_itemsets.empty:
            self.logger.warning("No frequent itemsets provided for rule generation")
            return pd.DataFrame()

        min_confidence = min_confidence or self.config.min_confidence
        min_lift = min_lift or self.config.min_lift

        # Generate association rules
        try:
            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=min_confidence,
                num_itemsets=len(frequent_itemsets)
            )

            # Filter by lift
            rules = rules[rules['lift'] >= min_lift]

            # Add additional metrics
            rules['conviction'] = self._calculate_conviction(rules)
            rules['rule_strength'] = self._calculate_rule_strength(rules)

            # Sort by lift and confidence
            rules = rules.sort_values(['lift', 'confidence'], ascending=False)

            self.logger.info(f"Generated {len(rules)} association rules")
            return rules

        except ValueError as e:
            self.logger.error(f"Error generating association rules: {e}")
            return pd.DataFrame()

    def _calculate_conviction(self, rules: pd.DataFrame) -> pd.Series:
        """Calculate conviction metric for association rules."""
        return (1 - rules['consequent support']) / (1 - rules['confidence'])

    def _calculate_rule_strength(self, rules: pd.DataFrame) -> pd.Series:
        """Calculate a composite rule strength metric."""
        # Combine support, confidence, and lift into a single strength score
        normalized_support = rules['support'] / rules['support'].max()
        normalized_confidence = rules['confidence']
        normalized_lift = (rules['lift'] - 1) / (rules['lift'].max() - 1) if rules['lift'].max() > 1 else 0

        return (normalized_support + normalized_confidence + normalized_lift) / 3

    async def analyze_patterns(
        self,
        window_days: int = 30,
        save_to_database: bool = True
    ) -> dict[str, Any]:
        """Perform complete Apriori analysis on recent prompt improvement data.
        
        Args:
            window_days: Number of days to analyze
            save_to_database: Whether to save results to database
            
        Returns:
            Dictionary with analysis results
        """
        try:
            self.logger.info(f"Starting Apriori pattern analysis for {window_days} days")

            # Extract transactions
            transactions = self.extract_transactions_from_database(window_days)
            if not transactions:
                return {"error": "No transaction data available"}

            # Mine frequent itemsets
            frequent_itemsets = await self.mine_frequent_itemsets(transactions)
            if frequent_itemsets.empty:
                return {"error": "No frequent itemsets found"}

            # Generate association rules
            rules = self.generate_association_rules(frequent_itemsets)
        except Exception as e:
            self.logger.error(f"Error in analyze_patterns: {e}")
            return {"error": str(e), "error_type": "analysis_error"}

        # Analyze results
        analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "transaction_count": len(transactions),
            "frequent_itemsets_count": len(frequent_itemsets),
            "association_rules_count": len(rules),
            "top_itemsets": self._get_top_itemsets(frequent_itemsets),
            "top_rules": self._get_top_rules(rules),
            "pattern_insights": self._generate_pattern_insights(rules),
            "config": {
                "min_support": self.config.min_support,
                "min_confidence": self.config.min_confidence,
                "min_lift": self.config.min_lift,
                "window_days": window_days
            }
        }

        # Save to database if requested
        if save_to_database and not rules.empty:
            self._save_association_rules(rules)

        self._last_analysis_time = datetime.now()
        return analysis_results

    def _get_top_itemsets(self, frequent_itemsets: pd.DataFrame, top_n: int = 10) -> list[dict]:
        """Get top frequent itemsets by support."""
        if frequent_itemsets.empty:
            return []

        top_itemsets = frequent_itemsets.nlargest(top_n, 'support')
        return [
            {
                "itemset": list(row['itemsets']),
                "support": round(row['support'], 4),
                "length": row['length']
            }
            for _, row in top_itemsets.iterrows()
        ]

    def _get_top_rules(self, rules: pd.DataFrame, top_n: int = 10) -> list[dict]:
        """Get top association rules by rule strength."""
        if rules.empty:
            return []

        top_rules = rules.nlargest(top_n, 'rule_strength')
        return [
            {
                "antecedents": list(row['antecedents']),
                "consequents": list(row['consequents']),
                "support": round(row['support'], 4),
                "confidence": round(row['confidence'], 4),
                "lift": round(row['lift'], 4),
                "rule_strength": round(row['rule_strength'], 4)
            }
            for _, row in top_rules.iterrows()
        ]

    def _generate_pattern_insights(self, rules: pd.DataFrame) -> dict[str, Any]:
        """Generate business insights from discovered patterns."""
        if rules.empty:
            return {}

        insights = {
            "rule_performance_patterns": [],
            "quality_improvement_patterns": [],
            "user_satisfaction_patterns": [],
            "prompt_characteristic_patterns": []
        }

        # Analyze rule performance patterns
        rule_perf_rules = rules[
            rules['antecedents'].astype(str).str.contains('rule_') &
            rules['consequents'].astype(str).str.contains('performance_')
        ]
        if not rule_perf_rules.empty:
            insights["rule_performance_patterns"] = [
                f"Rule {list(row['antecedents'])[0].replace('rule_', '')} → {list(row['consequents'])[0]} "
                f"(confidence: {row['confidence']:.2f}, lift: {row['lift']:.2f})"
                for _, row in rule_perf_rules.head(3).iterrows()
            ]

        # Analyze quality improvement patterns
        quality_rules = rules[
            rules['consequents'].astype(str).str.contains('quality_high')
        ]
        if not quality_rules.empty:
            insights["quality_improvement_patterns"] = [
                f"{list(row['antecedents'])} → High Quality "
                f"(confidence: {row['confidence']:.2f}, lift: {row['lift']:.2f})"
                for _, row in quality_rules.head(3).iterrows()
            ]

        return insights

    @handle_errors(return_format="dict")
    def _save_association_rules(self, rules: pd.DataFrame) -> bool:
        """Save discovered association rules to database."""
        try:
            # Create table if it doesn't exist
            create_table_query = """
            CREATE TABLE IF NOT EXISTS apriori_association_rules (
                id SERIAL PRIMARY KEY,
                antecedents TEXT NOT NULL,
                consequents TEXT NOT NULL,
                support DECIMAL(6,4) NOT NULL,
                confidence DECIMAL(6,4) NOT NULL,
                lift DECIMAL(6,4) NOT NULL,
                conviction DECIMAL(6,4),
                rule_strength DECIMAL(6,4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(antecedents, consequents)
            )
            """

            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(create_table_query)

                # Insert rules
                for _, rule in rules.iterrows():
                    insert_query = """
                    INSERT INTO apriori_association_rules 
                    (antecedents, consequents, support, confidence, lift, conviction, rule_strength)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (antecedents, consequents) 
                    DO UPDATE SET 
                        support = EXCLUDED.support,
                        confidence = EXCLUDED.confidence,
                        lift = EXCLUDED.lift,
                        conviction = EXCLUDED.conviction,
                        rule_strength = EXCLUDED.rule_strength,
                        created_at = CURRENT_TIMESTAMP
                    """

                    cursor.execute(insert_query, (
                        str(list(rule['antecedents'])),
                        str(list(rule['consequents'])),
                        float(rule['support']),
                        float(rule['confidence']),
                        float(rule['lift']),
                        float(rule.get('conviction', 0)),
                        float(rule.get('rule_strength', 0))
                    ))

                conn.commit()
                self.logger.info(f"Saved {len(rules)} association rules to database")
                return True

        except Exception as e:
            self.logger.error(f"Error saving association rules: {e}")
            return False

    def get_rules_for_context(
        self,
        context_items: list[str],
        min_confidence: float = 0.6
    ) -> list[dict[str, Any]]:
        """Get association rules relevant to a specific context.
        
        Args:
            context_items: Items representing current context
            min_confidence: Minimum confidence for returned rules
            
        Returns:
            List of relevant association rules
        """
        if not hasattr(self, '_association_rules_cache') or not self._association_rules_cache:
            self.logger.warning("No cached association rules available")
            return []

        relevant_rules = []
        for rules_df in self._association_rules_cache.values():
            if rules_df.empty:
                continue

            # Find rules where antecedents match context items
            for _, rule in rules_df.iterrows():
                antecedents = set(rule['antecedents'])
                if (antecedents.intersection(set(context_items)) and
                    rule['confidence'] >= min_confidence):
                    relevant_rules.append({
                        "antecedents": list(rule['antecedents']),
                        "consequents": list(rule['consequents']),
                        "confidence": rule['confidence'],
                        "lift": rule['lift'],
                        "support": rule['support']
                    })

        # Sort by confidence and return
        return sorted(relevant_rules, key=lambda x: x['confidence'], reverse=True)
