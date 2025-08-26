"""Batch Capacity Calculator for A/B Testing Framework.".

This script helps estimate:
1. Batch data processing throughput
2. Sample size requirements for A/B testing
3. Test duration planning based on capacity
4. Resource optimization recommendations

Based on your AdvancedABTestingFramework configuration and ML infrastructure.
"""

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy.stats import norm


@dataclass
class BatchConfig:
    """Configuration for batch processing capacity."""

    batch_size: int = 10
    max_concurrent_tests: int = 5
    timeout_seconds: int = 3600
    optimization_trials: int = 100
    processing_time_per_prompt_ms: float = 500


@dataclass
class StatisticalConfig:
    """Configuration for statistical power calculations."""

    significance_level: float = 0.05
    power_threshold: float = 0.8
    minimum_sample_size: int = 30
    balance_tolerance_percent: float = 5
    min_effect_size: float = 0.1


@dataclass
class CapacityEstimate:
    """Results of capacity estimation."""

    hourly_throughput: int
    daily_throughput: int
    weekly_throughput: int
    bottleneck_factor: str
    efficiency_score: float
    recommendations: list[str]


@dataclass
class SampleSizeEstimate:
    """Results of sample size calculation."""

    required_sample_size: int
    test_duration_days: float
    confidence_interval: tuple[float, float]
    power_analysis: dict[str, float]
    group_sequential_params: dict[str, float]


class BatchCapacityCalculator:
    """Calculator for batch processing capacity and A/B testing requirements."""

    def __init__(
        self,
        batch_config: BatchConfig | None = None,
        stats_config: StatisticalConfig | None = None,
    ) -> None:
        self.batch_config = batch_config or BatchConfig()
        self.stats_config = stats_config or StatisticalConfig()
        self.logger = logging.getLogger(__name__)

    def calculate_throughput_capacity(self) -> CapacityEstimate:
        """Calculate theoretical and practical throughput capacity."""
        prompts_per_batch = self.batch_config.batch_size
        concurrent_tests = self.batch_config.max_concurrent_tests
        processing_time_ms = self.batch_config.processing_time_per_prompt_ms
        batch_processing_time_ms = (
            processing_time_ms * prompts_per_batch / concurrent_tests
        )
        overhead_factor = 1.3
        actual_batch_time_ms = batch_processing_time_ms * overhead_factor
        hour_ms = 3600 * 1000
        batches_per_hour = hour_ms / actual_batch_time_ms
        hourly_prompts = batches_per_hour * prompts_per_batch
        daily_prompts = hourly_prompts * 24
        weekly_prompts = daily_prompts * 7
        bottleneck_factors = {
            "ML_INFERENCE": actual_batch_time_ms
            / (processing_time_ms * prompts_per_batch),
            "CONCURRENCY": concurrent_tests / 10,
            "BATCH_SIZE": prompts_per_batch / 20,
        }
        primary_bottleneck = min(
            bottleneck_factors.keys(), key=lambda k: bottleneck_factors[k]
        )
        efficiency = min(1.0, hour_ms / (actual_batch_time_ms * batches_per_hour))
        recommendations = self._generate_capacity_recommendations(
            hourly_prompts, bottleneck_factors, efficiency
        )
        return CapacityEstimate(
            hourly_throughput=int(hourly_prompts),
            daily_throughput=int(daily_prompts),
            weekly_throughput=int(weekly_prompts),
            bottleneck_factor=primary_bottleneck,
            efficiency_score=efficiency,
            recommendations=recommendations,
        )

    def calculate_sample_size_requirements(
        self, effect_size: float | None = None
    ) -> SampleSizeEstimate:
        """Calculate required sample size for A/B testing with group sequential design."""
        effect_size = effect_size or self.stats_config.min_effect_size
        alpha = self.stats_config.significance_level
        power = self.stats_config.power_threshold
        z_alpha_2 = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
        n_per_group = 2 * (z_alpha_2 + z_beta) ** 2 / effect_size**2
        total_sample_size = int(np.ceil(n_per_group * 2))
        sequential_inflation = 1.15
        sequential_sample_size = int(total_sample_size * sequential_inflation)
        final_sample_size = max(
            sequential_sample_size, self.stats_config.minimum_sample_size * 2
        )
        margin_error = z_alpha_2 * np.sqrt(2 / (final_sample_size / 2))
        confidence_interval = (effect_size - margin_error, effect_size + margin_error)
        power_analysis = {
            "achieved_power": power,
            "effect_size": effect_size,
            "alpha": alpha,
            "beta": 1 - power,
            "critical_value": z_alpha_2,
            "sample_size_per_group": final_sample_size // 2,
        }
        group_sequential_params = {
            "inflation_factor": sequential_inflation,
            "groups_planned": 5,
            "interim_alpha": alpha / 5,
            "monitoring_frequency": "daily",
        }
        return SampleSizeEstimate(
            required_sample_size=final_sample_size,
            test_duration_days=0,
            confidence_interval=confidence_interval,
            power_analysis=power_analysis,
            group_sequential_params=group_sequential_params,
        )

    def estimate_test_duration(
        self, capacity: CapacityEstimate, sample_size: SampleSizeEstimate
    ) -> float:
        """Estimate test duration based on capacity and sample size."""
        required_prompts = sample_size.required_sample_size
        daily_capacity = capacity.daily_throughput
        utilization_factor = 0.7
        effective_daily_capacity = daily_capacity * utilization_factor
        duration_days = required_prompts / effective_daily_capacity
        buffer_days = 2
        return max(duration_days + buffer_days, 3)

    def _generate_capacity_recommendations(
        self, hourly_throughput: float, bottlenecks: dict[str, float], efficiency: float
    ) -> list[str]:
        """Generate optimization recommendations."""
        recommendations = []
        if efficiency < 0.6:
            recommendations.append(
                "Low efficiency detected. Consider optimizing ML inference pipeline."
            )
        if bottlenecks["CONCURRENCY"] < 0.5:
            recommendations.append(
                "Increase max_concurrent_tests from 5 to 8-10 for better parallelization."
            )
        if bottlenecks["BATCH_SIZE"] < 0.5:
            recommendations.append(
                "Consider increasing batch_size from 10 to 15-20 for better throughput."
            )
        if hourly_throughput < 100:
            recommendations.append(
                "Low throughput. Consider ML model optimization or hardware scaling."
            )
        if bottlenecks["ML_INFERENCE"] > 1.5:
            recommendations.append(
                "ML inference is bottleneck. Cache models or use faster inference."
            )
        recommendations.append(f"Current efficiency: {efficiency:.1%}. Target: >80%")
        return recommendations

    def generate_capacity_report(self) -> dict:
        """Generate comprehensive capacity and sample size report."""
        print("🔄 Calculating batch processing capacity...")
        capacity = self.calculate_throughput_capacity()
        print("📊 Calculating sample size requirements...")
        effect_sizes = [0.05, 0.1, 0.15, 0.2, 0.3]
        sample_estimates = {}
        for effect_size in effect_sizes:
            sample_est = self.calculate_sample_size_requirements(effect_size)
            duration = self.estimate_test_duration(capacity, sample_est)
            sample_est.test_duration_days = duration
            sample_estimates[effect_size] = sample_est
        report = {
            "timestamp": datetime.now().isoformat(),
            "capacity_analysis": {
                "hourly_throughput": capacity.hourly_throughput,
                "daily_throughput": capacity.daily_throughput,
                "weekly_throughput": capacity.weekly_throughput,
                "bottleneck": capacity.bottleneck_factor,
                "efficiency": f"{capacity.efficiency_score:.1%}",
                "recommendations": capacity.recommendations,
            },
            "sample_size_analysis": {},
            "configuration": {
                "batch_size": self.batch_config.batch_size,
                "max_concurrent_tests": self.batch_config.max_concurrent_tests,
                "timeout_hours": self.batch_config.timeout_seconds / 3600,
                "significance_level": self.stats_config.significance_level,
                "power_threshold": self.stats_config.power_threshold,
            },
        }
        for effect_size, estimate in sample_estimates.items():
            report["sample_size_analysis"][f"effect_size_{effect_size}"] = {
                "required_sample_size": estimate.required_sample_size,
                "test_duration_days": round(estimate.test_duration_days, 1),
                "achieved_power": estimate.power_analysis["achieved_power"],
                "confidence_interval": estimate.confidence_interval,
                "group_sequential_inflation": estimate.group_sequential_params[
                    "inflation_factor"
                ],
            }
        return report

    def print_summary(self, report: dict):
        """Print formatted summary of the analysis."""
        print("\n" + "=" * 60)
        print("🚀 BATCH CAPACITY & A/B TESTING ANALYSIS")
        print("=" * 60)
        capacity = report["capacity_analysis"]
        print("\n📈 THROUGHPUT CAPACITY:")
        print(f"  • Hourly:  {capacity['hourly_throughput']:,} prompts")
        print(f"  • Daily:   {capacity['daily_throughput']:,} prompts")
        print(f"  • Weekly:  {capacity['weekly_throughput']:,} prompts")
        print(f"  • Efficiency: {capacity['efficiency']}")
        print(f"  • Bottleneck: {capacity['bottleneck']}")
        print("\n🎯 SAMPLE SIZE REQUIREMENTS:")
        for effect_name, analysis in report["sample_size_analysis"].items():
            effect_size = float(effect_name.split("_")[-1])
            print(f"  Effect Size {effect_size:0.2f}:")
            print(f"    - Sample Size: {analysis['required_sample_size']:,} prompts")
            print(f"    - Test Duration: {analysis['test_duration_days']} days")
            print(f"    - Power: {analysis['achieved_power']:.1%}")
        print("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(capacity["recommendations"], 1):
            print(f"  {i}. {rec}")
        config = report["configuration"]
        print("\n⚙️  CURRENT CONFIGURATION:")
        print(f"  • Batch Size: {config['batch_size']}")
        print(f"  • Max Concurrent: {config['max_concurrent_tests']}")
        print(f"  • Timeout: {config['timeout_hours']} hours")
        print(f"  • Significance Level: {config['significance_level']}")
        print(f"  • Power Threshold: {config['power_threshold']:.1%}")
        print("\n" + "=" * 60)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Batch Capacity Calculator for A/B Testing"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing (default: 10)",
    )
    parser.add_argument(
        "--concurrent-tests",
        type=int,
        default=5,
        help="Max concurrent tests (default: 5)",
    )
    parser.add_argument(
        "--processing-time",
        type=float,
        default=500,
        help="Processing time per prompt in ms (default: 500)",
    )
    parser.add_argument(
        "--effect-size",
        type=float,
        default=0.1,
        help="Minimum effect size to detect (default: 0.1)",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=0.8,
        help="Statistical power threshold (default: 0.8)",
    )
    parser.add_argument("--export", type=str, help="Export report to JSON file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    batch_config = BatchConfig(
        batch_size=args.batch_size,
        max_concurrent_tests=args.concurrent_tests,
        processing_time_per_prompt_ms=args.processing_time,
    )
    stats_config = StatisticalConfig(
        power_threshold=args.power, min_effect_size=args.effect_size
    )
    calculator = BatchCapacityCalculator(batch_config, stats_config)
    report = calculator.generate_capacity_report()
    calculator.print_summary(report)
    if args.export:
        import json

        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Report exported to: {args.export}")


if __name__ == "__main__":
    main()
