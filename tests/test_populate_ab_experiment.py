"""
Test file to validate the populate_ab_experiment fixture.
This ensures the fixture creates proper ABExperiment and RulePerformance records.
"""
import asyncio
import pytest
from sqlalchemy import func, select
from prompt_improver.database.models import ABExperiment, PromptSession, RulePerformance

@pytest.mark.asyncio
async def test_populate_ab_experiment_basic(populate_ab_experiment):
    """Test that the fixture creates experiments and performance records."""
    experiments, rule_performance_records = populate_ab_experiment
    assert len(experiments) == 3
    exp_names = [exp.experiment_name for exp in experiments]
    assert 'Clarity Enhancement A/B Test' in exp_names
    assert 'Multi-Rule Performance Test' in exp_names
    assert 'Completed Experiment' in exp_names
    assert len(rule_performance_records) >= 700
    running_experiments = [exp for exp in experiments if exp.status == 'running']
    completed_experiments = [exp for exp in experiments if exp.status == 'completed']
    assert len(running_experiments) == 2
    assert len(completed_experiments) == 1
    completed_exp = completed_experiments[0]
    assert completed_exp.results is not None
    assert 'p_value' in completed_exp.results
    assert 'statistical_significance' in completed_exp.results

@pytest.mark.asyncio
async def test_populate_ab_experiment_database_queries(real_db_session, populate_ab_experiment):
    """Test that RealTimeAnalyticsService and ExperimentOrchestrator can query the data."""
    experiments, rule_performance_records = populate_ab_experiment
    stmt = select(ABExperiment).where(ABExperiment.status == 'running')
    result = await real_db_session.execute(stmt)
    running_experiments = result.scalars().all()
    assert len(running_experiments) == 2
    stmt = select(RulePerformance).where(RulePerformance.rule_id == 'clarity_rule')
    result = await real_db_session.execute(stmt)
    clarity_records = result.scalars().all()
    assert len(clarity_records) > 0
    stmt = select(func.avg(RulePerformance.improvement_score), func.count(RulePerformance.id)).where(RulePerformance.rule_id == 'clarity_rule')
    result = await real_db_session.execute(stmt)
    avg_score, count = result.first()
    assert avg_score is not None
    assert count > 0
    assert 0.0 <= avg_score <= 1.0

@pytest.mark.asyncio
async def test_populate_ab_experiment_referential_integrity(real_db_session, populate_ab_experiment):
    """Test that foreign key relationships are properly maintained."""
    experiments, rule_performance_records = populate_ab_experiment
    stmt = select(RulePerformance.session_id).distinct()
    result = await real_db_session.execute(stmt)
    session_ids = {row[0] for row in result.fetchall()}
    stmt = select(PromptSession.session_id).where(PromptSession.session_id.in_(session_ids))
    result = await real_db_session.execute(stmt)
    existing_session_ids = {row[0] for row in result.fetchall()}
    assert len(session_ids) == len(existing_session_ids)
    assert session_ids == existing_session_ids
if __name__ == '__main__':
    asyncio.run(pytest.main([__file__, '-v']))
