from typer.testing import CliRunner

from reglens.cli import app

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("ingest", "ask", "eval", "serve"):
        assert command in result.output


def test_unimplemented_command_exits_nonzero() -> None:
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 1
