"""Training Data Loader for ML Pipeline

Automatically loads and combines real and synthetic training data
following 2025 best practices for ML training pipelines.
"""
from datetime import datetime, timedelta
import logging
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import and_, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np
from ...database.models import RuleMetadata, RulePerformance, TrainingPrompt
from ...utils.datetime_utils import aware_utc_now
logger = logging.getLogger(__name__)

class TrainingDataLoader:
    """Unified training data loader that automatically combines real and synthetic data"""

    def __init__(self, real_data_priority: bool=True, min_samples: int=20, lookback_days: int=30, synthetic_ratio: float=0.3):
        """Initialize the training data loader
        
        Args:
            real_data_priority: Whether to prioritize real data over synthetic
            min_samples: Minimum samples required for training
            lookback_days: Days to look back for training data
            synthetic_ratio: Maximum ratio of synthetic data to include
        """
        self.real_data_priority = real_data_priority
        self.min_samples = min_samples
        self.lookback_days = lookback_days
        self.synthetic_ratio = synthetic_ratio

    async def load_training_data(self, db_session: AsyncSession, rule_ids: Optional[List[str]]=None) -> Dict[str, Any]:
        """Load training data from both real usage and synthetic sources
        
        Returns a unified dataset ready for ML training with automatic
        combination of real and synthetic data.
        """
        logger.info('Loading training data with automatic real+synthetic combination')
        real_data = await self._load_real_performance_data(db_session, rule_ids)
        real_count = len(real_data.get('features', []))
        logger.info('Loaded %s real training samples', real_count)
        synthetic_needed = max(0, self.min_samples - real_count)
        max_synthetic = int(real_count * self.synthetic_ratio) if real_count > 0 else self.min_samples
        synthetic_data = await self._load_synthetic_training_data(db_session, limit=max(synthetic_needed, max_synthetic))
        synthetic_count = len(synthetic_data.get('features', []))
        logger.info('Loaded %s synthetic training samples', synthetic_count)
        combined_data = self._combine_training_data(real_data, synthetic_data)
        combined_data['metadata'] = {'real_samples': real_count, 'synthetic_samples': synthetic_count, 'total_samples': real_count + synthetic_count, 'synthetic_ratio': synthetic_count / (real_count + synthetic_count) if real_count + synthetic_count > 0 else 0, 'real_data_priority': self.real_data_priority, 'lookback_days': self.lookback_days, 'timestamp': datetime.utcnow().isoformat()}
        validation = self._validate_training_data(combined_data)
        combined_data['validation'] = validation
        logger.info('Training data ready: %s samples (%s real, %s synthetic)', combined_data['metadata']['total_samples'], real_count, synthetic_count)
        return combined_data

    async def _load_real_performance_data(self, db_session: AsyncSession, rule_ids: Optional[List[str]]=None) -> Dict[str, Any]:
        """Load real training data from rule_performance table"""
        recent_date = aware_utc_now() - timedelta(days=self.lookback_days)
        stmt = select(RulePerformance, RuleMetadata.rule_name, RuleMetadata.default_parameters, RuleMetadata.priority).join(RuleMetadata, RulePerformance.rule_id == RuleMetadata.rule_id).where(and_(RuleMetadata.enabled == True, RulePerformance.created_at >= recent_date, RulePerformance.improvement_score > 0))
        if rule_ids:
            stmt = stmt.where(RulePerformance.rule_id.in_(rule_ids))
        result = await db_session.execute(stmt)
        performance_data = result.fetchall()
        features = []
        labels = []
        metadata = []
        for perf, rule_name, default_params, priority in performance_data:
            feature_vector = [perf.improvement_score or 0.5, perf.confidence_level or 0.5, (perf.execution_time_ms or 100) / 1000.0, priority / 100.0]
            if perf.rule_parameters:
                for key, value in sorted(perf.rule_parameters.items()):
                    if isinstance(value, (int, float)):
                        feature_vector.append(float(value))
            features.append(feature_vector)
            labels.append(perf.improvement_score or 0.5)
            metadata.append({'rule_id': perf.rule_id, 'rule_name': rule_name, 'prompt_id': perf.prompt_id, 'source': 'real'})
        return {'features': features, 'labels': labels, 'metadata': metadata}

    async def _load_synthetic_training_data(self, db_session: AsyncSession, limit: int=100) -> Dict[str, Any]:
        """Load synthetic training data from training_prompts table"""
        stmt = select(TrainingPrompt).where(and_(TrainingPrompt.data_source == 'synthetic', TrainingPrompt.is_active == True)).order_by(TrainingPrompt.training_priority.desc()).limit(limit)
        result = await db_session.execute(stmt)
        training_prompts = result.scalars().all()
        features = []
        labels = []
        metadata = []
        for prompt in training_prompts:
            if prompt.enhancement_result:
                feature_vector = prompt.enhancement_result.get('feature_vector', [])
                effectiveness = prompt.enhancement_result.get('effectiveness_score', 0.5)
                if feature_vector:
                    features.append(feature_vector)
                    labels.append(effectiveness)
                    metadata.append({'prompt_id': prompt.id, 'domain': prompt.enhancement_result.get('metadata', {}).get('domain', 'unknown'), 'source': 'synthetic', 'session_id': prompt.session_id})
        return {'features': features, 'labels': labels, 'metadata': metadata}

    def _combine_training_data(self, real_data: Dict[str, Any], synthetic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligently combine real and synthetic training data"""
        if self.real_data_priority:
            combined_features = real_data['features'] + synthetic_data['features']
            combined_labels = real_data['labels'] + synthetic_data['labels']
            combined_metadata = real_data['metadata'] + synthetic_data['metadata']
        else:
            combined_features = []
            combined_labels = []
            combined_metadata = []
            real_idx = 0
            synth_idx = 0
            while real_idx < len(real_data['features']) or synth_idx < len(synthetic_data['features']):
                if real_idx < len(real_data['features']):
                    combined_features.append(real_data['features'][real_idx])
                    combined_labels.append(real_data['labels'][real_idx])
                    combined_metadata.append(real_data['metadata'][real_idx])
                    real_idx += 1
                if synth_idx < len(synthetic_data['features']) and real_idx % 3 == 0:
                    combined_features.append(synthetic_data['features'][synth_idx])
                    combined_labels.append(synthetic_data['labels'][synth_idx])
                    combined_metadata.append(synthetic_data['metadata'][synth_idx])
                    synth_idx += 1
        return {'features': combined_features, 'labels': combined_labels, 'metadata': combined_metadata}

    def _validate_training_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate training data quality"""
        features = np.array(data['features']) if data['features'] else np.array([])
        labels = np.array(data['labels']) if data['labels'] else np.array([])
        validation = {'is_valid': True, 'total_samples': len(features), 'has_minimum_samples': len(features) >= self.min_samples, 'feature_dimensions': features.shape[1] if len(features) > 0 else 0, 'label_distribution': {}, 'warnings': []}
        if len(features) < self.min_samples:
            validation['warnings'].append(f'Insufficient samples: {len(features)} < {self.min_samples}')
            validation['is_valid'] = False
        if len(features) > 0:
            unique_labels, counts = np.unique(labels, return_counts=True)
            validation['label_distribution'] = {f'class_{i}': int(count) for i, count in enumerate(counts)}
            if len(unique_labels) > 1:
                imbalance_ratio = counts.max() / counts.min()
                if imbalance_ratio > 10:
                    validation['warnings'].append(f'Severe class imbalance detected: {imbalance_ratio:.1f}:1')
        return validation

async def get_training_data_stats(db_session: AsyncSession) -> Dict[str, Any]:
    """Get comprehensive training data statistics"""
    real_stats = await db_session.execute(text('\n            SELECT \n                COUNT(*) as total_real,\n                COUNT(DISTINCT rule_id) as unique_rules,\n                AVG(improvement_score) as avg_improvement,\n                MIN(created_at) as oldest_record,\n                MAX(created_at) as newest_record\n            FROM rule_performance\n            WHERE improvement_score > 0\n        '))
    real_row = real_stats.fetchone()
    synthetic_stats = await db_session.execute(text("\n            SELECT \n                COUNT(*) as total_synthetic,\n                COUNT(DISTINCT \n                    CASE \n                        WHEN enhancement_result->>'metadata' IS NOT NULL \n                        THEN enhancement_result->'metadata'->>'domain' \n                    END\n                ) as unique_domains,\n                AVG(training_priority) as avg_priority,\n                MIN(created_at) as oldest_record,\n                MAX(created_at) as newest_record\n            FROM training_prompts\n            WHERE data_source = 'synthetic' AND is_active = true\n        "))
    synthetic_row = synthetic_stats.fetchone()
    return {'real_data': {'total_samples': real_row[0] or 0, 'unique_rules': real_row[1] or 0, 'avg_improvement': float(real_row[2] or 0), 'date_range': {'oldest': real_row[3].isoformat() if real_row[3] else None, 'newest': real_row[4].isoformat() if real_row[4] else None}}, 'synthetic_data': {'total_samples': synthetic_row[0] or 0, 'unique_domains': synthetic_row[1] or 0, 'avg_priority': float(synthetic_row[2] or 0), 'date_range': {'oldest': synthetic_row[3].isoformat() if synthetic_row[3] else None, 'newest': synthetic_row[4].isoformat() if synthetic_row[4] else None}}, 'combined': {'total_samples': (real_row[0] or 0) + (synthetic_row[0] or 0), 'real_ratio': (real_row[0] or 0) / ((real_row[0] or 0) + (synthetic_row[0] or 0)) if (real_row[0] or 0) + (synthetic_row[0] or 0) > 0 else 0}}
