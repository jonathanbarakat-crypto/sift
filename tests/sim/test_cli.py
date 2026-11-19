"""CLI entrypoint tests (SIMULATED — infrastructure validation, not
user-behavior evidence): SIFT_ENV=test runs, anything else refuses, and the
emitted report file carries the banner."""

import json

from sift.sim.header import SIM_HEADER
from sift.sim.run import main


def test_cli_refuses_without_test_env(monkeypatch, capsys):
    monkeypatch.delenv("SIFT_ENV", raising=False)
    rc = main(["--sim", "all", "--n-users", "2"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "SIFT_ENV=test" in err


def test_cli_runs_all_sims_and_writes_bannered_report(sim_env, capsys, tmp_path):
    out = tmp_path / "report.json"
    rc = main(["--sim", "all", "--n-users", "6", "--runs-per-user", "1",
               "--workers", "4", "--out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["sim"] is True
    assert payload["header"] == SIM_HEADER
    assert set(payload["reports"]) == {"load_soak", "antiabuse_attack_sims",
                                       "card_lifecycle_sim"}
    for name, rep in payload["reports"].items():
        assert rep["header"] == SIM_HEADER, name
        assert rep["sim"] is True, name
    # stdout also leads with the banner
    stdout = capsys.readouterr().out
    assert stdout.splitlines()[0] == SIM_HEADER
