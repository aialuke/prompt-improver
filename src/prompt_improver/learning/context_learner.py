"""Context-Specific Learning Engine

Optimizes prompt improvement strategies for different project types and contexts.
Learns from historical data to identify context-specific patterns and specialization
opportunities for more effective rule application.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import hashlib

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import warnings

# Phase 2 enhancement imports
try:
    import hdbscan
    import umap
    ADVANCED_CLUSTERING_AVAILABLE = True
except ImportError:
    ADVANCED_CLUSTERING_AVAILABLE = False
    warnings.warn("HDBSCAN and UMAP not available. Install with: pip install hdbscan umap-learn")

# Phase 3 enhancement imports for advanced privacy-preserving ML
try:
    import opacus
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    warnings.warn("Opacus not available for differential privacy. Install with: pip install opacus")

try:
    import cryptography
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    warnings.warn("Cryptography not available for secure storage. Install with: pip install cryptography")

logger = logging.getLogger(__name__)


@dataclass
class ContextConfig:
    """Configuration for context-specific learning"""
    # Learning parameters
    significance_threshold: float = 0.15
    min_sample_size: int = 10
    similarity_threshold: float = 0.8
    
    # Performance tracking
    improvement_threshold: float = 0.1
    consistency_threshold: float = 0.7
    
    # Specialization parameters
    max_specializations: int = 5
    confidence_threshold: float = 0.8
    
    # Context analysis
    max_context_groups: int = 20
    enable_semantic_clustering: bool = True
    
    # Phase 2 enhancements - In-Context Learning
    enable_in_context_learning: bool = True
    icl_demonstrations: int = 5
    icl_similarity_threshold: float = 0.7
    privacy_preserving: bool = True
    differential_privacy_epsilon: float = 1.0
    
    # Phase 3 enhancements - Advanced Privacy-Preserving ML
    enable_federated_learning: bool = True
    federated_rounds: int = 3
    federated_min_clients: int = 2
    secure_aggregation: bool = True
    homomorphic_encryption: bool = False
    privacy_budget_tracking: bool = True
    max_privacy_budget: float = 10.0
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    
    # Phase 2 enhancements - Advanced Clustering
    use_advanced_clustering: bool = True
    umap_n_components: int = 10
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 3
    clustering_quality_threshold: float = 0.5


@dataclass
class ContextInsight:
    """Insights for a specific context"""
    context_key: str
    sample_size: int
    avg_performance: float
    consistency_score: float
    top_performing_rules: List[Dict[str, Any]]
    poor_performing_rules: List[Dict[str, Any]]
    specialization_potential: float
    unique_patterns: List[str]


@dataclass
class SpecializationOpportunity:
    """Opportunity for rule specialization"""
    context_key: str
    rule_id: str
    current_performance: float
    specialized_performance: float
    improvement_potential: float
    confidence: float
    required_modifications: List[str]
    cost_benefit_ratio: float


@dataclass
class LearningRecommendation:
    """Learning-based recommendation"""
    type: str  # 'specialize', 'generalize', 'create_new', 'deprecate'
    priority: str  # 'high', 'medium', 'low'
    context: str
    description: str
    expected_impact: float
    implementation_effort: str
    supporting_evidence: List[str]


class ContextSpecificLearner:
    """Context-Specific Learning Engine for prompt improvement optimization"""
    
    def __init__(self, config: Optional[ContextConfig] = None):
        """Initialize the context-specific learner
        
        Args:
            config: Configuration for context learning
        """
        self.config = config or ContextConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Learning state
        self.context_groups: Dict[str, List[Dict[str, Any]]] = {}
        self.context_patterns: Dict[str, Dict[str, Any]] = {}
        self.specialization_opportunities: List[SpecializationOpportunity] = []
        self.performance_baseline: Optional[Dict[str, float]] = None
        
        # Phase 2 enhancements - In-Context Learning state
        self.demonstration_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.context_embeddings: Optional[np.ndarray] = None
        self.icl_model: Optional[Any] = None
        
        # Phase 3 enhancements - Advanced Privacy-Preserving ML state
        self.privacy_engine: Optional[Any] = None
        self.privacy_budget_used: float = 0.0
        self.federated_models: Dict[str, Any] = {}
        self.encryption_key: Optional[bytes] = None
        self.secure_aggregator: Optional[Any] = None
        
        # Initialize vectorizer for semantic context analysis
        if self.config.enable_semantic_clustering:
            self.context_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
        else:
            self.context_vectorizer = None
        
        # Phase 2 enhancements - Advanced clustering components
        self.umap_reducer = None
        self.hdbscan_clusterer = None
        self.clustering_quality_score = 0.0
        
        # Check for advanced clustering availability
        if self.config.use_advanced_clustering and not ADVANCED_CLUSTERING_AVAILABLE:
            self.logger.warning("Advanced clustering requested but HDBSCAN/UMAP not available. Falling back to K-means.")
            self.config.use_advanced_clustering = False
        
        # Phase 3: Initialize privacy-preserving components
        if self.config.privacy_preserving:
            self._initialize_privacy_components()
    
    async def analyze_context_effectiveness(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze context effectiveness across test results
        
        Args:
            results: Test results with context information
            
        Returns:
            Context-specific learning analysis
        """
        self.logger.info(f"Starting context effectiveness analysis with {len(results)} results")
        
        # Phase 2 enhancement: Apply advanced clustering if enabled
        if self.config.use_advanced_clustering:
            clustering_result = self._perform_advanced_clustering(results)
            if clustering_result.get("status") == "success":
                context_groups = clustering_result["clusters"]
                self.logger.info(f"Advanced clustering identified {clustering_result['n_clusters']} context groups with quality {clustering_result['quality_score']:.3f}")
            else:
                # Fallback to traditional context grouping
                context_groups = self._group_results_by_context(results)
                self.logger.info(f"Fallback: Identified {len(context_groups)} context groups")
        else:
            # Traditional context grouping
            context_groups = self._group_results_by_context(results)
            self.logger.info(f"Identified {len(context_groups)} context groups")
        
        # Analyze each context group
        context_insights = {}
        for context_key, context_results in context_groups.items():
            context_insights[context_key] = await self._analyze_context_group(context_key, context_results)
        
        # Cross-context analysis
        cross_context_analysis = self._perform_cross_context_analysis(context_insights)
        
        # Identify specialization opportunities
        specialization_opportunities = self._identify_specialization_opportunities(context_insights)
        
        # Generate learning recommendations
        learning_recommendations = self._generate_learning_recommendations(
            context_insights, specialization_opportunities
        )
        
        # Phase 2 enhancement: Apply in-context learning if enabled
        icl_result = None
        if self.config.enable_in_context_learning:
            # Extract context data and user preferences for ICL
            context_data = []
            for result in results[:20]:  # Sample for ICL analysis
                if 'context' in result:
                    context_data.append(result['context'])
            
            user_preferences = {}  # Could be populated from user settings
            task_examples = results[:50]  # Sample task examples
            
            icl_result = self._implement_in_context_learning(
                context_data, user_preferences, task_examples
            )
        
        analysis_result = {
            "context_insights": context_insights,
            "cross_context_comparisons": cross_context_analysis["comparisons"],
            "universal_patterns": cross_context_analysis["universal_patterns"],
            "context_specific_patterns": cross_context_analysis["context_specific_patterns"],
            "specialization_opportunities": [op.__dict__ for op in specialization_opportunities],
            "learning_recommendations": [rec.__dict__ for rec in learning_recommendations],
            # Phase 2 enhancements
            "in_context_learning": icl_result,
            "advanced_clustering": clustering_result if 'clustering_result' in locals() else None,
            "clustering_quality_score": self.clustering_quality_score,
            "metadata": {
                "total_contexts": len(context_insights),
                "total_results": len(results),
                "analysis_date": datetime.now().isoformat(),
                "config": self.config.__dict__,
                # Phase 2 metadata
                "phase2_enhancements": {
                    "in_context_learning_enabled": self.config.enable_in_context_learning,
                    "advanced_clustering_enabled": self.config.use_advanced_clustering,
                    "privacy_preserving_enabled": self.config.privacy_preserving,
                    "clustering_method": "umap_hdbscan" if ADVANCED_CLUSTERING_AVAILABLE and self.config.use_advanced_clustering else "kmeans"
                }
            }
        }
        
        self.logger.info(
            f"Context analysis completed: {len(context_insights)} contexts, "
            f"{len(specialization_opportunities)} specialization opportunities"
        )
        
        return analysis_result
    
    def _group_results_by_context(self, results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group results by project context"""
        groups = defaultdict(list)
        
        for result in results:
            context_key = self._generate_context_key(result.get("context", {}))
            groups[context_key].append(result)
        
        # Filter out groups with insufficient samples
        filtered_groups = {
            key: group for key, group in groups.items()
            if len(group) >= self.config.min_sample_size
        }
        
        self.logger.info(f"Filtered {len(groups)} groups to {len(filtered_groups)} with sufficient samples")
        
        return filtered_groups
    
    def _generate_context_key(self, context: Dict[str, Any]) -> str:
        """Generate a context key for grouping similar contexts"""
        if not context:
            return "unknown"
        
        # Extract key context features
        project_type = context.get("projectType", "unknown")
        domain = context.get("domain", "unknown")
        complexity = context.get("complexity", "unknown")
        team_size = context.get("teamSize", "unknown")
        
        # Create a normalized context key
        key_components = [project_type, domain, complexity, str(team_size)]
        context_key = "|".join(key_components).lower()
        
        return context_key
    
    async def _analyze_context_group(self, context_key: str, results: List[Dict[str, Any]]) -> ContextInsight:
        """Analyze a specific context group"""
        # Extract performance metrics
        performance_scores = []
        rule_performance = defaultdict(list)
        
        for result in results:
            score = result.get("overallScore") or result.get("improvementScore", 0)
            performance_scores.append(score)
            
            applied_rules = result.get("appliedRules", [])
            for rule in applied_rules:
                rule_id = rule.get("ruleId") or rule.get("id", "unknown")
                rule_score = rule.get("improvementScore", score)
                rule_performance[rule_id].append(rule_score)
        
        if not performance_scores:
            return ContextInsight(
                context_key=context_key,
                sample_size=len(results),
                avg_performance=0.0,
                consistency_score=0.0,
                top_performing_rules=[],
                poor_performing_rules=[],
                specialization_potential=0.0,
                unique_patterns=[]
            )
        
        # Calculate basic statistics
        avg_performance = float(np.mean(performance_scores))
        consistency_score = 1 - (np.std(performance_scores) / avg_performance) if avg_performance > 0 else 0
        
        # Analyze rule performance
        rule_stats = {}
        for rule_id, scores in rule_performance.items():
            if len(scores) >= 3:  # Minimum for reliable statistics
                rule_stats[rule_id] = {
                    "avg_score": float(np.mean(scores)),
                    "std_score": float(np.std(scores)),
                    "sample_size": len(scores),
                    "consistency": 1 - (np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0
                }
        
        # Identify top and poor performing rules
        sorted_rules = sorted(
            rule_stats.items(),
            key=lambda x: x[1]["avg_score"],
            reverse=True
        )
        
        top_performing_rules = [
            {"rule_id": rule_id, **stats}
            for rule_id, stats in sorted_rules[:5]
        ]
        
        poor_performing_rules = [
            {"rule_id": rule_id, **stats}
            for rule_id, stats in sorted_rules[-3:]
            if stats["avg_score"] < avg_performance * 0.8
        ]
        
        # Calculate specialization potential
        specialization_potential = self._calculate_specialization_potential(
            context_key, rule_stats, avg_performance
        )
        
        # Identify unique patterns
        unique_patterns = self._identify_unique_patterns(results, context_key)
        
        return ContextInsight(
            context_key=context_key,
            sample_size=len(results),
            avg_performance=avg_performance,
            consistency_score=float(max(0, consistency_score)),
            top_performing_rules=top_performing_rules,
            poor_performing_rules=poor_performing_rules,
            specialization_potential=specialization_potential,
            unique_patterns=unique_patterns
        )
    
    def _calculate_specialization_potential(
        self, 
        context_key: str, 
        rule_stats: Dict[str, Dict[str, Any]], 
        context_avg: float
    ) -> float:
        """Calculate the potential for rule specialization in this context"""
        if not rule_stats or not self.performance_baseline:
            return 0.0
        
        # Compare context performance to baseline
        baseline_avg = self.performance_baseline.get("overall", context_avg)
        context_deviation = abs(context_avg - baseline_avg) / baseline_avg if baseline_avg > 0 else 0
        
        # Look for rules with significantly different performance
        rule_deviations = []
        for rule_id, stats in rule_stats.items():
            baseline_rule_avg = self.performance_baseline.get(rule_id, stats["avg_score"])
            if baseline_rule_avg > 0:
                deviation = abs(stats["avg_score"] - baseline_rule_avg) / baseline_rule_avg
                rule_deviations.append(deviation)
        
        if not rule_deviations:
            return context_deviation
        
        # Combine context and rule deviations
        avg_rule_deviation = np.mean(rule_deviations)
        specialization_potential = (context_deviation + avg_rule_deviation) / 2
        
        return min(1.0, specialization_potential)
    
    def _identify_unique_patterns(self, results: List[Dict[str, Any]], context_key: str) -> List[str]:
        """Identify unique patterns in this context"""
        patterns = []
        
        # Analyze prompt characteristics
        prompt_lengths = []
        prompt_types = defaultdict(int)
        improvement_types = defaultdict(int)
        
        for result in results:
            original_prompt = result.get("originalPrompt", "")
            prompt_lengths.append(len(original_prompt.split()))
            
            # Extract prompt type patterns
            if "question" in original_prompt.lower():
                prompt_types["question"] += 1
            elif "instruction" in original_prompt.lower():
                prompt_types["instruction"] += 1
            elif "describe" in original_prompt.lower():
                prompt_types["description"] += 1
            else:
                prompt_types["other"] += 1
            
            # Extract improvement patterns
            improvements = result.get("appliedImprovements", [])
            for improvement in improvements:
                improvement_types[improvement.get("type", "unknown")] += 1
        
        # Identify significant patterns
        if prompt_lengths:
            avg_length = np.mean(prompt_lengths)
            if avg_length > 50:
                patterns.append(f"long_prompts_avg_{avg_length:.1f}_words")
            elif avg_length < 10:
                patterns.append(f"short_prompts_avg_{avg_length:.1f}_words")
        
        # Most common prompt type
        if prompt_types:
            most_common_type = max(prompt_types.items(), key=lambda x: x[1])
            if most_common_type[1] / len(results) > 0.6:  # 60% threshold
                patterns.append(f"predominantly_{most_common_type[0]}_prompts")
        
        # Most common improvement type
        if improvement_types:
            most_common_improvement = max(improvement_types.items(), key=lambda x: x[1])
            if most_common_improvement[1] / len(results) > 0.4:  # 40% threshold
                patterns.append(f"frequent_{most_common_improvement[0]}_improvements")
        
        return patterns[:5]  # Limit to top 5 patterns
    
    def _perform_cross_context_analysis(self, context_insights: Dict[str, ContextInsight]) -> Dict[str, Any]:
        """Perform cross-context analysis to identify universal vs context-specific patterns"""
        if len(context_insights) < 2:
            return {
                "comparisons": [],
                "universal_patterns": [],
                "context_specific_patterns": []
            }
        
        comparisons = []
        universal_patterns = []
        context_specific_patterns = []
        
        # Compare contexts pairwise
        context_keys = list(context_insights.keys())
        for i, context1 in enumerate(context_keys):
            for context2 in context_keys[i+1:]:
                insight1 = context_insights[context1]
                insight2 = context_insights[context2]
                
                # Performance comparison
                perf_diff = abs(insight1.avg_performance - insight2.avg_performance)
                significant_diff = perf_diff > self.config.significance_threshold
                
                # Rule overlap analysis
                rules1 = {rule["rule_id"] for rule in insight1.top_performing_rules}
                rules2 = {rule["rule_id"] for rule in insight2.top_performing_rules}
                
                overlap = len(rules1.intersection(rules2))
                total_unique = len(rules1.union(rules2))
                overlap_ratio = overlap / total_unique if total_unique > 0 else 0
                
                comparison = {
                    "context_pair": [context1, context2],
                    "performance_difference": float(perf_diff),
                    "significant_difference": significant_diff,
                    "rule_overlap_ratio": float(overlap_ratio),
                    "shared_rules": list(rules1.intersection(rules2)),
                    "context1_unique_rules": list(rules1 - rules2),
                    "context2_unique_rules": list(rules2 - rules1)
                }
                comparisons.append(comparison)
        
        # Identify universal patterns (rules that work well across contexts)
        rule_performance_across_contexts = defaultdict(list)
        for insight in context_insights.values():
            for rule in insight.top_performing_rules:
                rule_performance_across_contexts[rule["rule_id"]].append({
                    "context": insight.context_key,
                    "score": rule["avg_score"]
                })
        
        for rule_id, performances in rule_performance_across_contexts.items():
            if len(performances) >= len(context_insights) * 0.7:  # Appears in 70% of contexts
                avg_score = np.mean([p["score"] for p in performances])
                consistency = 1 - (np.std([p["score"] for p in performances]) / avg_score) if avg_score > 0 else 0
                
                if consistency > self.config.consistency_threshold:
                    universal_patterns.append({
                        "rule_id": rule_id,
                        "avg_score_across_contexts": float(avg_score),
                        "consistency": float(consistency),
                        "present_in_contexts": len(performances),
                        "contexts": [p["context"] for p in performances]
                    })
        
        # Identify context-specific patterns
        for context_key, insight in context_insights.items():
            for rule in insight.top_performing_rules:
                rule_id = rule["rule_id"]
                
                # Check if this rule is NOT universal
                is_universal = any(up["rule_id"] == rule_id for up in universal_patterns)
                
                if not is_universal and rule["avg_score"] > insight.avg_performance * 1.1:
                    context_specific_patterns.append({
                        "context": context_key,
                        "rule_id": rule_id,
                        "context_score": rule["avg_score"],
                        "context_avg": insight.avg_performance,
                        "relative_performance": rule["avg_score"] / insight.avg_performance,
                        "sample_size": rule["sample_size"]
                    })
        
        return {
            "comparisons": comparisons,
            "universal_patterns": universal_patterns,
            "context_specific_patterns": context_specific_patterns
        }
    
    def _identify_specialization_opportunities(
        self, 
        context_insights: Dict[str, ContextInsight]
    ) -> List[SpecializationOpportunity]:
        """Identify opportunities for rule specialization"""
        opportunities = []
        
        for context_key, insight in context_insights.items():
            # Look for rules with poor performance that could be specialized
            for poor_rule in insight.poor_performing_rules:
                rule_id = poor_rule["rule_id"]
                current_performance = poor_rule["avg_score"]
                
                # Estimate potential improvement through specialization
                context_potential = insight.specialization_potential
                if context_potential > 0.3:  # Threshold for worthwhile specialization
                    
                    # Calculate potential specialized performance
                    # This is a heuristic - in practice, you'd use more sophisticated modeling
                    specialized_performance = min(
                        1.0, 
                        current_performance + (context_potential * self.config.improvement_threshold)
                    )
                    
                    improvement_potential = specialized_performance - current_performance
                    
                    if improvement_potential > self.config.improvement_threshold:
                        # Estimate implementation cost
                        cost_benefit_ratio = improvement_potential / (context_potential + 0.1)
                        
                        opportunity = SpecializationOpportunity(
                            context_key=context_key,
                            rule_id=rule_id,
                            current_performance=current_performance,
                            specialized_performance=specialized_performance,
                            improvement_potential=improvement_potential,
                            confidence=min(1.0, context_potential * insight.consistency_score),
                            required_modifications=self._suggest_modifications(insight, rule_id),
                            cost_benefit_ratio=cost_benefit_ratio
                        )
                        
                        if opportunity.confidence > self.config.confidence_threshold:
                            opportunities.append(opportunity)
        
        # Sort by improvement potential
        opportunities.sort(key=lambda x: x.improvement_potential, reverse=True)
        
        return opportunities[:self.config.max_specializations]
    
    def _suggest_modifications(self, insight: ContextInsight, rule_id: str) -> List[str]:
        """Suggest modifications for rule specialization"""
        modifications = []
        
        # Based on unique patterns in the context
        for pattern in insight.unique_patterns:
            if "long_prompts" in pattern:
                modifications.append("Add length-aware optimizations")
            elif "short_prompts" in pattern:
                modifications.append("Focus on conciseness improvements")
            elif "question" in pattern:
                modifications.append("Optimize for question-answering format")
            elif "instruction" in pattern:
                modifications.append("Enhance instruction clarity")
            elif "description" in pattern:
                modifications.append("Improve descriptive language")
        
        # Based on performance characteristics
        if insight.consistency_score < 0.5:
            modifications.append("Add consistency checks")
        
        if insight.avg_performance < 0.7:
            modifications.append("Strengthen core improvement logic")
        
        return modifications[:3]  # Limit to top 3 suggestions
    
    def _generate_learning_recommendations(
        self,
        context_insights: Dict[str, ContextInsight],
        specialization_opportunities: List[SpecializationOpportunity]
    ) -> List[LearningRecommendation]:
        """Generate learning-based recommendations"""
        recommendations = []
        
        # Specialization recommendations
        for opportunity in specialization_opportunities[:3]:  # Top 3
            rec = LearningRecommendation(
                type="specialize",
                priority="high" if opportunity.improvement_potential > 0.2 else "medium",
                context=opportunity.context_key,
                description=f"Specialize rule {opportunity.rule_id} for context {opportunity.context_key}",
                expected_impact=opportunity.improvement_potential,
                implementation_effort="medium",
                supporting_evidence=[
                    f"Current performance: {opportunity.current_performance:.3f}",
                    f"Potential improvement: {opportunity.improvement_potential:.3f}",
                    f"Confidence: {opportunity.confidence:.3f}"
                ]
            )
            recommendations.append(rec)
        
        # Generalization recommendations
        poor_contexts = [
            (key, insight) for key, insight in context_insights.items()
            if insight.avg_performance < 0.6 and insight.sample_size >= self.config.min_sample_size * 2
        ]
        
        for context_key, insight in poor_contexts[:2]:  # Top 2 poor performers
            rec = LearningRecommendation(
                type="generalize",
                priority="medium",
                context=context_key,
                description=f"Improve general rule effectiveness for context {context_key}",
                expected_impact=0.8 - insight.avg_performance,  # Target 0.8 performance
                implementation_effort="low",
                supporting_evidence=[
                    f"Below-average performance: {insight.avg_performance:.3f}",
                    f"Sufficient sample size: {insight.sample_size}",
                    f"Consistency issues" if insight.consistency_score < 0.6 else "Consistent poor performance"
                ]
            )
            recommendations.append(rec)
        
        # New rule creation recommendations
        high_potential_contexts = [
            (key, insight) for key, insight in context_insights.items()
            if insight.specialization_potential > 0.6 and len(insight.top_performing_rules) < 3
        ]
        
        for context_key, insight in high_potential_contexts[:2]:
            rec = LearningRecommendation(
                type="create_new",
                priority="low",
                context=context_key,
                description=f"Create context-specific rules for {context_key}",
                expected_impact=insight.specialization_potential * 0.3,
                implementation_effort="high",
                supporting_evidence=[
                    f"High specialization potential: {insight.specialization_potential:.3f}",
                    f"Limited effective rules: {len(insight.top_performing_rules)}",
                    f"Unique patterns: {', '.join(insight.unique_patterns[:3])}"
                ]
            )
            recommendations.append(rec)
        
        # Sort by priority and expected impact
        priority_order = {"high": 3, "medium": 2, "low": 1}
        recommendations.sort(
            key=lambda x: (priority_order[x.priority], x.expected_impact),
            reverse=True
        )
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    # ==================== PHASE 2 ENHANCEMENTS ====================
    
    def _implement_in_context_learning(self, context_data: List[Dict[str, Any]], 
                                     user_preferences: Dict[str, Any], 
                                     task_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        2025 In-Context Learning framework for personalized prompt optimization.
        
        Key insights from research:
        - ICL operates without parameter updates
        - Privacy-preserving through federated approaches  
        - Adaptive few-shot learning based on user patterns
        """
        self.logger.info("Implementing in-context learning framework")
        
        if not self.config.enable_in_context_learning:
            return {"status": "disabled", "demonstrations": []}
        
        # Context-aware demonstration selection
        demonstrations = self._select_contextual_demonstrations(
            context_data, task_examples, user_preferences
        )
        
        # Contextual bandit for personalization
        bandit_recommendations = self._apply_contextual_bandit(
            demonstrations, user_preferences, context_data
        )
        
        # Privacy-preserving adaptation with differential privacy
        if self.config.privacy_preserving:
            demonstrations = self._apply_differential_privacy(
                demonstrations, self.config.differential_privacy_epsilon
            )
        
        return {
            "status": "success",
            "demonstrations": demonstrations,
            "bandit_recommendations": bandit_recommendations,
            "privacy_applied": self.config.privacy_preserving,
            "demonstration_count": len(demonstrations)
        }
    
    def _select_contextual_demonstrations(self, context_data: List[Dict[str, Any]], 
                                        task_examples: List[Dict[str, Any]], 
                                        user_preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select most relevant demonstrations using cosine similarity"""
        if not task_examples:
            return []
        
        # Extract context features for similarity computation
        context_texts = []
        for ctx in context_data:
            text_features = []
            if 'projectType' in ctx:
                text_features.append(f"project_{ctx['projectType']}")
            if 'domain' in ctx:
                text_features.append(f"domain_{ctx['domain']}")
            if 'complexity' in ctx:
                text_features.append(f"complexity_{ctx['complexity']}")
            context_texts.append(" ".join(text_features))
        
        if not context_texts:
            return task_examples[:self.config.icl_demonstrations]
        
        # Vectorize contexts and examples
        try:
            if self.context_vectorizer is None:
                self.context_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
            
            # Combine context and example texts for vectorization
            example_texts = []
            for example in task_examples:
                example_text = example.get('originalPrompt', '') + " " + example.get('improvedPrompt', '')
                example_texts.append(example_text)
            
            all_texts = context_texts + example_texts
            if len(all_texts) > 1:
                vectors = self.context_vectorizer.fit_transform(all_texts)
                
                # Compute similarities between contexts and examples
                context_vectors = vectors[:len(context_texts)]
                example_vectors = vectors[len(context_texts):]
                
                if context_vectors.shape[0] > 0 and example_vectors.shape[0] > 0:
                    similarities = cosine_similarity(context_vectors, example_vectors)
                    
                    # Select top demonstrations based on similarity
                    avg_similarities = np.mean(similarities, axis=0)
                    top_indices = np.argsort(avg_similarities)[-self.config.icl_demonstrations:]
                    
                    selected_demonstrations = [task_examples[i] for i in top_indices 
                                             if avg_similarities[i] > self.config.icl_similarity_threshold]
                    
                    self.logger.info(f"Selected {len(selected_demonstrations)} contextual demonstrations")
                    return selected_demonstrations
        
        except Exception as e:
            self.logger.warning(f"Error in demonstration selection: {e}")
        
        # Fallback to top examples
        return task_examples[:self.config.icl_demonstrations]
    
    def _apply_contextual_bandit(self, demonstrations: List[Dict[str, Any]], 
                               user_preferences: Dict[str, Any], 
                               context_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply Thompson Sampling for exploration-exploitation balance"""
        if not demonstrations:
            return {"recommendations": [], "exploration_score": 0.0}
        
        # Thompson Sampling implementation for contextual bandits
        # This is a simplified version - in production, you'd use a more sophisticated approach
        
        # Extract performance scores from demonstrations
        performance_scores = []
        for demo in demonstrations:
            score = demo.get('overallScore', demo.get('improvementScore', 0.5))
            performance_scores.append(score)
        
        if not performance_scores:
            return {"recommendations": [], "exploration_score": 0.0}
        
        # Beta distribution parameters for Thompson Sampling
        successes = [max(1, int(score * 10)) for score in performance_scores]
        failures = [max(1, int((1 - score) * 10)) for score in performance_scores]
        
        # Sample from Beta distributions
        sampled_rewards = []
        for i in range(len(demonstrations)):
            sampled_reward = np.random.beta(successes[i], failures[i])
            sampled_rewards.append(sampled_reward)
        
        # Rank demonstrations by sampled rewards
        ranked_indices = np.argsort(sampled_rewards)[::-1]
        
        recommendations = []
        for i in ranked_indices:
            recommendations.append({
                "demonstration_index": i,
                "expected_reward": sampled_rewards[i],
                "confidence": successes[i] / (successes[i] + failures[i]),
                "exploration_value": 1.0 / (successes[i] + failures[i])
            })
        
        exploration_score = np.mean([rec["exploration_value"] for rec in recommendations])
        
        return {
            "recommendations": recommendations,
            "exploration_score": exploration_score,
            "method": "thompson_sampling"
        }
    
    def _apply_differential_privacy(self, demonstrations: List[Dict[str, Any]], 
                                  epsilon: float) -> List[Dict[str, Any]]:
        """Apply differential privacy to protect user data"""
        if not demonstrations:
            return demonstrations
        
        # Apply Laplace noise to sensitive numerical values
        for demo in demonstrations:
            if 'overallScore' in demo:
                noise = np.random.laplace(0, 1.0 / epsilon)
                demo['overallScore'] = np.clip(demo['overallScore'] + noise, 0, 1)
            
            if 'improvementScore' in demo:
                noise = np.random.laplace(0, 1.0 / epsilon)
                demo['improvementScore'] = np.clip(demo['improvementScore'] + noise, 0, 1)
        
        self.logger.info(f"Applied differential privacy with epsilon={epsilon}")
        return demonstrations
    
    # ==================== PHASE 3 ENHANCEMENTS ====================
    
    def _initialize_privacy_components(self) -> None:
        """Initialize Phase 3 privacy-preserving ML components"""
        try:
            # Initialize privacy budget tracking
            self.privacy_budget_used = 0.0
            
            # Initialize secure aggregation if enabled
            if self.config.secure_aggregation and CRYPTO_AVAILABLE:
                self.encryption_key = Fernet.generate_key()
                self.logger.info("Secure aggregation initialized with encryption")
            
            # Initialize Opacus privacy engine if available
            if OPACUS_AVAILABLE and self.config.enable_federated_learning:
                self.privacy_engine = PrivacyEngine()
                self.logger.info("Opacus privacy engine initialized")
            
            self.logger.info("Privacy-preserving ML components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize privacy components: {e}")
            self.config.privacy_preserving = False
    
    def _implement_federated_learning(self, context_data: List[Dict[str, Any]], 
                                    client_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Implement federated learning for privacy-preserving personalization"""
        if not self.config.enable_federated_learning:
            return {"status": "disabled", "reason": "Federated learning not enabled"}
        
        if len(client_data) < self.config.federated_min_clients:
            return {"status": "insufficient_clients", "clients_available": len(client_data)}
        
        self.logger.info(f"Starting federated learning with {len(client_data)} clients")
        
        federated_results = {
            "status": "success",
            "rounds_completed": 0,
            "participating_clients": len(client_data),
            "privacy_budget_used": 0.0,
            "global_model_updates": [],
            "convergence_metrics": []
        }
        
        try:
            # Federated learning rounds
            for round_num in range(self.config.federated_rounds):
                round_results = self._execute_federated_round(round_num, client_data, context_data)
                
                if round_results["status"] == "success":
                    federated_results["rounds_completed"] += 1
                    federated_results["global_model_updates"].append(round_results["global_update"])
                    federated_results["convergence_metrics"].append(round_results["convergence_metric"])
                    federated_results["privacy_budget_used"] += round_results["privacy_cost"]
                    
                    # Check privacy budget
                    if federated_results["privacy_budget_used"] >= self.config.max_privacy_budget:
                        self.logger.warning(f"Privacy budget exhausted at round {round_num}")
                        break
                else:
                    self.logger.error(f"Federated round {round_num} failed: {round_results['error']}")
                    break
            
            # Update global privacy budget
            self.privacy_budget_used += federated_results["privacy_budget_used"]
            
            self.logger.info(f"Federated learning completed: {federated_results['rounds_completed']} rounds")
            return federated_results
            
        except Exception as e:
            self.logger.error(f"Federated learning failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _execute_federated_round(self, round_num: int, client_data: Dict[str, List[Dict[str, Any]]], 
                               context_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a single federated learning round"""
        client_updates = []
        privacy_costs = []
        
        # Client local training
        for client_id, client_examples in client_data.items():
            if len(client_examples) < self.config.min_sample_size:
                continue
                
            # Local model update with differential privacy
            local_update = self._compute_local_update(client_examples, context_data)
            
            # Apply differential privacy to local update
            if self.config.privacy_preserving:
                private_update, privacy_cost = self._apply_advanced_differential_privacy(local_update)
                client_updates.append(private_update)
                privacy_costs.append(privacy_cost)
            else:
                client_updates.append(local_update)
                privacy_costs.append(0.0)
        
        if not client_updates:
            return {"status": "no_valid_clients", "error": "No clients with sufficient data"}
        
        # Secure aggregation
        if self.config.secure_aggregation:
            global_update = self._secure_aggregate_updates(client_updates)
        else:
            global_update = self._simple_aggregate_updates(client_updates)
        
        # Compute convergence metric
        convergence_metric = self._compute_convergence_metric(client_updates)
        
        return {
            "status": "success",
            "round": round_num,
            "participating_clients": len(client_updates),
            "global_update": global_update,
            "convergence_metric": convergence_metric,
            "privacy_cost": sum(privacy_costs)
        }
    
    def _compute_local_update(self, client_examples: List[Dict[str, Any]], 
                            context_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute local model update for a client"""
        # Extract features and performance scores
        features = []
        scores = []
        
        for example in client_examples:
            if 'overallScore' in example or 'improvementScore' in example:
                # Use context features for local learning
                context_features = self._extract_context_features(example.get('context', {}))
                features.append(context_features)
                scores.append(example.get('overallScore', example.get('improvementScore', 0.5)))
        
        if not features:
            return {"gradient": np.zeros(10), "sample_count": 0}
        
        features_array = np.array(features)
        scores_array = np.array(scores)
        
        # Simple gradient computation (placeholder for actual ML model)
        # In practice, this would be gradients from a neural network or other ML model
        gradient = np.mean(features_array * scores_array.reshape(-1, 1), axis=0)
        
        return {
            "gradient": gradient,
            "sample_count": len(features),
            "mean_score": np.mean(scores_array)
        }
    
    def _extract_context_features(self, context: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from context"""
        # Simple feature extraction - in practice would be more sophisticated
        features = []
        
        # Project type encoding
        project_types = ['web', 'mobile', 'desktop', 'api', 'ml', 'data']
        project_type = context.get('projectType', 'web').lower()
        for pt in project_types:
            features.append(1.0 if pt == project_type else 0.0)
        
        # Add complexity and team size
        complexity_map = {'low': 0.25, 'medium': 0.5, 'high': 0.75, 'very_high': 1.0}
        features.append(complexity_map.get(context.get('complexity', 'medium'), 0.5))
        
        team_size = context.get('teamSize', 5)
        if isinstance(team_size, str):
            team_size = {'small': 3, 'medium': 8, 'large': 15}.get(team_size.lower(), 8)
        features.append(min(team_size / 20.0, 1.0))
        
        # Pad to fixed size
        while len(features) < 10:
            features.append(0.0)
        
        return np.array(features[:10])
    
    def _apply_advanced_differential_privacy(self, local_update: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Apply advanced differential privacy with Gaussian noise"""
        if not OPACUS_AVAILABLE:
            # Fallback to simple Laplace noise
            return self._apply_laplace_noise(local_update)
        
        # Use Opacus-style Gaussian noise for better privacy guarantees
        gradient = local_update["gradient"]
        sample_count = local_update["sample_count"]
        
        # Clip gradient norm
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > self.config.max_grad_norm:
            gradient = gradient * (self.config.max_grad_norm / gradient_norm)
        
        # Add Gaussian noise
        noise_scale = self.config.noise_multiplier * self.config.max_grad_norm
        noise = np.random.normal(0, noise_scale, gradient.shape)
        
        private_gradient = gradient + noise
        
        # Compute privacy cost (simplified)
        privacy_cost = (sample_count * self.config.differential_privacy_epsilon) / 100.0
        
        private_update = {
            "gradient": private_gradient,
            "sample_count": sample_count,
            "mean_score": local_update["mean_score"],
            "privacy_applied": True
        }
        
        return private_update, privacy_cost
    
    def _apply_laplace_noise(self, local_update: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Fallback Laplace noise application"""
        gradient = local_update["gradient"]
        
        # Apply Laplace noise to gradient
        noise = np.random.laplace(0, 1.0 / self.config.differential_privacy_epsilon, gradient.shape)
        private_gradient = gradient + noise
        
        private_update = {
            "gradient": private_gradient,
            "sample_count": local_update["sample_count"],
            "mean_score": local_update["mean_score"],
            "privacy_applied": True
        }
        
        privacy_cost = self.config.differential_privacy_epsilon / 10.0
        return private_update, privacy_cost
    
    def _secure_aggregate_updates(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Securely aggregate client updates using encryption"""
        if not self.config.secure_aggregation or not CRYPTO_AVAILABLE:
            return self._simple_aggregate_updates(client_updates)
        
        try:
            # Encrypt individual updates
            fernet = Fernet(self.encryption_key)
            encrypted_updates = []
            
            for update in client_updates:
                # Serialize and encrypt gradient
                gradient_bytes = update["gradient"].tobytes()
                encrypted_gradient = fernet.encrypt(gradient_bytes)
                
                encrypted_updates.append({
                    "encrypted_gradient": encrypted_gradient,
                    "sample_count": update["sample_count"],
                    "mean_score": update["mean_score"]
                })
            
            # Decrypt and aggregate (in practice, this would use secure multi-party computation)
            decrypted_gradients = []
            total_samples = 0
            total_weighted_score = 0.0
            
            for enc_update in encrypted_updates:
                decrypted_gradient_bytes = fernet.decrypt(enc_update["encrypted_gradient"])
                gradient = np.frombuffer(decrypted_gradient_bytes, dtype=np.float64)
                
                decrypted_gradients.append(gradient)
                total_samples += enc_update["sample_count"]
                total_weighted_score += enc_update["mean_score"] * enc_update["sample_count"]
            
            # Aggregate gradients
            aggregated_gradient = np.mean(decrypted_gradients, axis=0)
            mean_score = total_weighted_score / total_samples if total_samples > 0 else 0.0
            
            return {
                "aggregated_gradient": aggregated_gradient,
                "total_samples": total_samples,
                "mean_score": mean_score,
                "aggregation_method": "secure"
            }
            
        except Exception as e:
            self.logger.error(f"Secure aggregation failed: {e}")
            return self._simple_aggregate_updates(client_updates)
    
    def _simple_aggregate_updates(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simple aggregation of client updates"""
        if not client_updates:
            return {"aggregated_gradient": np.zeros(10), "total_samples": 0, "mean_score": 0.0}
        
        gradients = [update["gradient"] for update in client_updates]
        sample_counts = [update["sample_count"] for update in client_updates]
        scores = [update["mean_score"] for update in client_updates]
        
        # Weighted average by sample count
        total_samples = sum(sample_counts)
        if total_samples == 0:
            return {"aggregated_gradient": np.zeros(10), "total_samples": 0, "mean_score": 0.0}
        
        weighted_gradient = np.zeros_like(gradients[0])
        weighted_score = 0.0
        
        for gradient, count, score in zip(gradients, sample_counts, scores):
            weight = count / total_samples
            weighted_gradient += weight * gradient
            weighted_score += weight * score
        
        return {
            "aggregated_gradient": weighted_gradient,
            "total_samples": total_samples,
            "mean_score": weighted_score,
            "aggregation_method": "simple"
        }
    
    def _compute_convergence_metric(self, client_updates: List[Dict[str, Any]]) -> float:
        """Compute convergence metric for federated learning"""
        if len(client_updates) < 2:
            return 1.0  # Single client - assume converged
        
        gradients = [update["gradient"] for update in client_updates]
        
        # Compute variance of gradients as convergence metric
        # Lower variance indicates better convergence
        gradient_matrix = np.array(gradients)
        gradient_variance = np.var(gradient_matrix, axis=0)
        mean_variance = np.mean(gradient_variance)
        
        # Convert to convergence score (0 = not converged, 1 = fully converged)
        convergence_score = 1.0 / (1.0 + mean_variance)
        
        return convergence_score
    
    def _perform_advanced_clustering(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Replace K-means with UMAP + HDBSCAN for better context clustering.
        
        Key advantages:
        - HDBSCAN: Density-based clustering handling variable density
        - UMAP: Preserves both local and global structure
        - Quality assessment: Silhouette score and Calinski-Harabasz index
        """
        if not self.config.use_advanced_clustering or not ADVANCED_CLUSTERING_AVAILABLE:
            return self._fallback_to_kmeans(results)
        
        self.logger.info("Performing advanced clustering with UMAP + HDBSCAN")
        
        # Extract features for clustering
        features = self._extract_clustering_features(results)
        if features is None or len(features) < self.config.min_sample_size:
            return {"status": "insufficient_data", "clusters": {}}
        
        try:
            # Step 1: UMAP dimensionality reduction
            self.umap_reducer = umap.UMAP(
                n_components=self.config.umap_n_components,
                n_neighbors=self.config.umap_n_neighbors,
                min_dist=self.config.umap_min_dist,
                random_state=42
            )
            
            reduced_features = self.umap_reducer.fit_transform(features)
            
            # Step 2: HDBSCAN clustering
            self.hdbscan_clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.config.hdbscan_min_cluster_size,
                min_samples=self.config.hdbscan_min_samples,
                metric='euclidean'
            )
            
            cluster_labels = self.hdbscan_clusterer.fit_predict(reduced_features)
            
            # Step 3: Quality assessment
            quality_score = self._assess_clustering_quality(reduced_features, cluster_labels)
            self.clustering_quality_score = quality_score
            
            # Group results by clusters
            clusters = self._group_by_clusters(results, cluster_labels)
            
            clustering_result = {
                "status": "success",
                "method": "umap_hdbscan",
                "n_clusters": len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                "n_noise_points": np.sum(cluster_labels == -1),
                "quality_score": quality_score,
                "clusters": clusters,
                "reduced_dimensions": reduced_features.shape[1]
            }
            
            if quality_score < self.config.clustering_quality_threshold:
                self.logger.warning(f"Clustering quality {quality_score:.3f} below threshold {self.config.clustering_quality_threshold}")
                return self._fallback_to_kmeans(results)
            
            self.logger.info(f"Advanced clustering completed: {clustering_result['n_clusters']} clusters, quality={quality_score:.3f}")
            return clustering_result
            
        except Exception as e:
            self.logger.error(f"Advanced clustering failed: {e}")
            return self._fallback_to_kmeans(results)
    
    def _extract_clustering_features(self, results: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Extract numerical features for clustering"""
        if not results:
            return None
        
        features = []
        for result in results:
            feature_vector = []
            
            # Performance metrics
            feature_vector.append(result.get('overallScore', 0.5))
            feature_vector.append(result.get('clarity', 0.5))
            feature_vector.append(result.get('completeness', 0.5))
            feature_vector.append(result.get('actionability', 0.5))
            feature_vector.append(result.get('effectiveness', 0.5))
            
            # Context features (encoded)
            context = result.get('context', {})
            
            # Project type encoding
            project_types = ['web', 'mobile', 'desktop', 'api', 'ml', 'data', 'other']
            project_type = context.get('projectType', 'other').lower()
            project_encoding = [1.0 if pt == project_type else 0.0 for pt in project_types]
            feature_vector.extend(project_encoding)
            
            # Domain encoding  
            domains = ['finance', 'healthcare', 'education', 'ecommerce', 'gaming', 'social', 'other']
            domain = context.get('domain', 'other').lower()
            domain_encoding = [1.0 if d == domain else 0.0 for d in domains]
            feature_vector.extend(domain_encoding)
            
            # Complexity encoding
            complexity_map = {'low': 0.25, 'medium': 0.5, 'high': 0.75, 'very_high': 1.0}
            complexity = complexity_map.get(context.get('complexity', 'medium'), 0.5)
            feature_vector.append(complexity)
            
            # Team size (normalized)
            team_size = context.get('teamSize', 5)
            if isinstance(team_size, str):
                team_size = {'small': 3, 'medium': 8, 'large': 15}.get(team_size.lower(), 8)
            feature_vector.append(min(team_size / 20.0, 1.0))  # Normalize to [0,1]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _assess_clustering_quality(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Assess clustering quality using multiple metrics"""
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
        unique_labels = set(labels)
        if len(unique_labels) <= 1 or (len(unique_labels) == 2 and -1 in unique_labels):
            return 0.0  # No meaningful clusters
        
        # Filter out noise points for quality assessment
        mask = labels != -1
        if np.sum(mask) < 2:
            return 0.0
        
        filtered_features = features[mask]
        filtered_labels = labels[mask]
        
        if len(set(filtered_labels)) <= 1:
            return 0.0
        
        try:
            # Silhouette score (higher is better, range [-1, 1])
            silhouette = silhouette_score(filtered_features, filtered_labels)
            
            # Calinski-Harabasz score (higher is better, unbounded)
            calinski = calinski_harabasz_score(filtered_features, filtered_labels)
            
            # Normalize and combine scores
            silhouette_normalized = (silhouette + 1) / 2  # Convert to [0, 1]
            calinski_normalized = min(calinski / 100.0, 1.0)  # Rough normalization
            
            # Weighted combination
            quality_score = 0.6 * silhouette_normalized + 0.4 * calinski_normalized
            
            return quality_score
            
        except Exception as e:
            self.logger.warning(f"Error assessing clustering quality: {e}")
            return 0.0
    
    def _group_by_clusters(self, results: List[Dict[str, Any]], 
                          labels: np.ndarray) -> Dict[str, List[Dict[str, Any]]]:
        """Group results by cluster labels"""
        clusters = {}
        
        for i, result in enumerate(results):
            cluster_id = int(labels[i])
            cluster_key = f"cluster_{cluster_id}" if cluster_id != -1 else "noise"
            
            if cluster_key not in clusters:
                clusters[cluster_key] = []
            
            clusters[cluster_key].append(result)
        
        return clusters
    
    def _fallback_to_kmeans(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback to K-means clustering when advanced methods fail"""
        self.logger.info("Using K-means clustering fallback")
        
        features = self._extract_clustering_features(results)
        if features is None or len(features) < self.config.min_sample_size:
            return {"status": "insufficient_data", "clusters": {}}
        
        try:
            # Determine optimal number of clusters (simple heuristic)
            n_samples = len(features)
            max_clusters = min(self.config.max_context_groups, max(2, n_samples // 10))
            n_clusters = min(max_clusters, max(2, int(np.sqrt(n_samples))))
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            clusters = self._group_by_clusters(results, cluster_labels)
            
            return {
                "status": "success",
                "method": "kmeans_fallback",
                "n_clusters": n_clusters,
                "clusters": clusters,
                "quality_score": 0.5  # Neutral score for fallback
            }
            
        except Exception as e:
            self.logger.error(f"K-means fallback failed: {e}")
            return {"status": "error", "clusters": {}}
    
    def update_baseline_performance(self, performance_data: Dict[str, float]):
        """Update baseline performance metrics"""
        self.performance_baseline = performance_data.copy()
        self.logger.info(f"Updated baseline performance with {len(performance_data)} metrics")
    
    def get_context_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Get learned context patterns"""
        return self.context_patterns.copy()
    
    def clear_learning_state(self):
        """Clear accumulated learning state"""
        self.context_groups.clear()
        self.context_patterns.clear()
        self.specialization_opportunities.clear()
        self.performance_baseline = None
        self.logger.info("Cleared learning state")