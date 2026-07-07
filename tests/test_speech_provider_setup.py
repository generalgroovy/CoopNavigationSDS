import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import setup_speech_providers as setup


def test_resolve_coqui_python_accepts_python_3_11_candidate(tmp_path):
    candidate = tmp_path / "python3.11"
    candidate.write_text("", encoding="utf-8")

    with patch.object(setup, "_candidate_python_paths", return_value=[candidate]), patch.object(
        setup,
        "interpreter_details",
        return_value={"executable": str(candidate), "version": [3, 11]},
    ):
        resolved, rejected = setup.resolve_coqui_python()

    assert resolved == candidate
    assert rejected == []


def test_resolve_coqui_python_rejects_project_python_3_13(tmp_path):
    candidate = tmp_path / "python3.13"
    candidate.write_text("", encoding="utf-8")

    with patch.object(setup, "_candidate_python_paths", return_value=[candidate]), patch.object(
        setup,
        "interpreter_details",
        return_value={"executable": str(candidate), "version": [3, 13]},
    ):
        resolved, rejected = setup.resolve_coqui_python()

    assert resolved is None
    assert "not supported" in rejected[0]["reason"]


def test_prepare_coqui_required_fails_with_actionable_message(tmp_path):
    with patch.object(setup, "resolve_coqui_python", return_value=(None, [{"candidate": "python3.13", "reason": "not supported"}])):
        with pytest.raises(RuntimeError, match="Coqui requires an isolated Python 3.10 or 3.11"):
            setup.prepare_coqui_provider(tmp_path, required=True)


def test_prepare_coqui_registers_isolated_provider(tmp_path):
    host_python = tmp_path / "host-python"
    host_python.write_text("", encoding="utf-8")

    with patch.object(setup, "resolve_coqui_python", return_value=(host_python, [])), patch.object(setup, "run") as run:
        assert setup.prepare_coqui_provider(tmp_path, required=True)

    provider_python = tmp_path / "coqui" / ("Scripts/python.exe" if setup.sys.platform == "win32" else "bin/python")
    assert run.call_args_list[0].args[0] == (host_python, "-m", "venv", tmp_path / "coqui")
    assert (tmp_path / "providers.json").is_file()
    manifest = json.loads((tmp_path / "providers.json").read_text(encoding="utf-8"))
    assert manifest["providers"]["coqui"]["python"] == str(provider_python.resolve())
