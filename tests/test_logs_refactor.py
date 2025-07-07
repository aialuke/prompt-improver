import pytest
from typer.testing import CliRunner

from prompt_improver.cli import app

runner = CliRunner()


def test_logs_command():
    result = runner.invoke(app, ["logs", "--level", "INFO", "--lines", "10"])
    assert result.exit_code == 0
    assert "Viewing logs:" in result.output
