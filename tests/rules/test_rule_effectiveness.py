import os
import json
import pytest
import time
from prompt_improver.rule_engine.base import BasePromptRule
from prompt_improver.services.prompt_improvement import PromptImprovementService

# Load prompts from fixtures
fixture_file = os.path.join(os.path.dirname(__file__), '../fixtures/prompts.json')
with open(fixture_file, 'r') as f:
    fixtures = json.load(f)

# Initialize the prompt improvement service
prompt_service = PromptImprovementService()

# Auto-discover rule classes
rule_classes = BasePromptRule.__subclasses__()

@pytest.mark.parametrize("test_prompt", fixtures['test_prompts'])
def test_rule_effectiveness(test_prompt):
    original_prompt = test_prompt['original']
    expected_issues = test_prompt['expected_issues']
    improvements = []
    total_time = 0

    for RuleClass in rule_classes:
        rule_instance = RuleClass()
        # Timing rule application
        start_time = time.time()
        check_result = rule_instance.check(original_prompt)
        if check_result.applies:
            apply_result = rule_instance.apply(original_prompt)
            improvements.append(apply_result.improved_prompt)
            # Calculate rule application time
            total_time += (time.time() - start_time)

    # Calculate initial and final metrics
    initial_metrics = prompt_service._calculate_metrics(original_prompt)
    final_metrics = prompt_service._calculate_metrics(improvements[-1] if improvements else original_prompt)
    delta_clarity = final_metrics['clarity'] - initial_metrics['clarity']
    delta_specificity = final_metrics['specificity'] - initial_metrics['specificity']

    # Check performance time threshold
    assert total_time <= 0.1, f"Rule processing time exceeded: {total_time:.3f}s"

    # Check for prompt degradation
    has_degradation = delta_clarity < 0 or delta_specificity < 0
    assert not has_degradation, "Prompt degraded in clarity or specificity"

    # Log results for HTML reporting
    log_results(test_prompt, delta_clarity, delta_specificity, total_time)


def log_results(test_prompt, delta_clarity, delta_specificity, total_time):
    # Log results to HTML report structure
    report_file = f"reports/rule_effectiveness_{time.strftime('%Y%m%d')}.html"
    with open(report_file, 'a') as report:
        report.write('<tr>\n')
        report.write(f"<td>{test_prompt['id']}</td>\n")
        report.write(f"<td>{test_prompt['category']}</td>\n")
        report.write(f"<td>{test_prompt['description']}</td>\n")
        report.write(f"<td>{delta_clarity:.2f}</td>\n")
        report.write(f"<td>{delta_specificity:.2f}</td>\n")
        report.write(f"<td>{total_time:.3f}s</td>\n")
        report.write('</tr>\n')


# Initialize the HTML report at the start
pytest_configure = lambda config: open(f"reports/rule_effectiveness_{time.strftime('%Y%m%d')}.html", 'w').write('<table><th>ID</th><th>Category</th><th>Description</th><th>Δ Clarity</th><th>Δ Specificity</th><th>Time</th>')

# Close the HTML report at the end
pytest_unconfigure = lambda config: open(f"reports/rule_effectiveness_{time.strftime('%Y%m%d')}.html", 'a').write('</table>')

