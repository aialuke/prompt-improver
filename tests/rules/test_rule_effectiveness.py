import json
import os
import time

import pytest

from prompt_improver.rule_engine.base import BasePromptRule
from prompt_improver.services.prompt.facade import (
    PromptServiceFacade as PromptImprovementService,
)

fixture_file = os.path.join(os.path.dirname(__file__), "../fixtures/prompts.json")
with open(fixture_file, encoding="utf-8") as f:
    fixtures = json.load(f)
prompt_service = PromptImprovementService()
rule_classes = BasePromptRule.__subclasses__()


@pytest.mark.parametrize("test_prompt", fixtures["test_prompts"])
def test_rule_effectiveness(test_prompt):
    original_prompt = test_prompt["original"]
    expected_issues = test_prompt["expected_issues"]
    improvements = []
    total_time = 0
    for RuleClass in rule_classes:
        rule_instance = RuleClass()
        start_time = time.time()
        check_result = rule_instance.check(original_prompt)
        if check_result.applies:
            apply_result = rule_instance.apply(original_prompt)
            improvements.append(apply_result.improved_prompt)
            total_time += time.time() - start_time
    initial_metrics = prompt_service._calculate_metrics(original_prompt)
    final_metrics = prompt_service._calculate_metrics(
        improvements[-1] if improvements else original_prompt
    )
    delta_clarity = final_metrics["clarity"] - initial_metrics["clarity"]
    delta_specificity = final_metrics["specificity"] - initial_metrics["specificity"]
    assert total_time <= 0.1, f"Rule processing time exceeded: {total_time:.3f}s"
    has_degradation = delta_clarity < 0 or delta_specificity < 0
    assert not has_degradation, "Prompt degraded in clarity or specificity"
    log_results(test_prompt, delta_clarity, delta_specificity, total_time)


def log_results(test_prompt, delta_clarity, delta_specificity, total_time):
    report_file = f"reports/rule_effectiveness_{time.strftime('%Y%m%d')}.html"
    with open(report_file, "a", encoding="utf-8") as report:
        report.write("<tr>\n")
        report.write(f"<td>{test_prompt['id']}</td>\n")
        report.write(f"<td>{test_prompt['category']}</td>\n")
        report.write(f"<td>{test_prompt['description']}</td>\n")
        report.write(f"<td>{delta_clarity:.2f}</td>\n")
        report.write(f"<td>{delta_specificity:.2f}</td>\n")
        report.write(f"<td>{total_time:.3f}s</td>\n")
        report.write("</tr>\n")


def pytest_configure(config):
    return open(
    f"reports/rule_effectiveness_{time.strftime('%Y%m%d')}.html", "w", encoding="utf-8"
).write(
    "<table><th>ID</th><th>Category</th><th>Description</th><th>Δ Clarity</th><th>Δ Specificity</th><th>Time</th>"
)


def pytest_unconfigure(config):
    return open(
    f"reports/rule_effectiveness_{time.strftime('%Y%m%d')}.html", "a", encoding="utf-8"
).write("</table>")
