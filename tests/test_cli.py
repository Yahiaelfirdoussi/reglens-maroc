from typer.testing import CliRunner

from reglens.cli import app

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("ingest", "ask", "chat", "eval", "serve"):
        assert command in result.output


def test_unknown_command_exits_nonzero() -> None:
    result = runner.invoke(app, ["no-such-command"])
    assert result.exit_code != 0
