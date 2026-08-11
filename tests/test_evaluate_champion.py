import sys

import pytest

from neat_flappy.genome import (
    BASE_NODE_IDS,
    ConnectionGene,
    Genome,
    InnovationStore,
    save_genome,
)
from tools.evaluate_champion import _summarize, evaluate_champion, main


def save_no_action_genome(tmp_path, *, generation=12):
    store = InnovationStore.base()
    connections = {
        innovation: ConnectionGene(innovation, src, dst, 0.0, True)
        for innovation, (src, dst) in store.connection_endpoints.items()
    }
    genome = Genome(
        7,
        BASE_NODE_IDS,
        connections,
        rms_cache={innovation: 0.0 for innovation in connections},
    )
    path = tmp_path / "no_action.json"
    save_genome(path, genome, store, generation)
    return path


def test_evaluate_champion_enumerates_inclusive_ordered_seed_range(tmp_path):
    path = save_no_action_genome(tmp_path)
    report = evaluate_champion(path, -1, 1, 1)

    assert report["checkpoint"] == str(path)
    assert report["checkpoint_generation"] == 12
    assert report["engine"] == "jax_aabb"
    assert report["seed_start"] == -1
    assert report["seed_stop"] == 1
    assert report["seed_count"] == 3
    assert report["max_frames"] == 1
    assert [record["seed"] for record in report["results"]] == [-1, 0, 1]


def test_no_action_checkpoint_reports_death_and_zero_pipes(tmp_path):
    path = save_no_action_genome(tmp_path)
    report = evaluate_champion(path, 0, 0, 300)
    result = report["results"][0]

    assert result["pipes_cleared"] == 0
    assert type(result["death_frame"]) is int
    assert result["survived_cap"] is False


def test_short_cap_reports_survival_and_one_seed_percentiles(tmp_path):
    path = save_no_action_genome(tmp_path)
    report = evaluate_champion(path, 0, 0, 1)
    result = report["results"][0]

    assert result == {
        "seed": 0,
        "pipes_cleared": 0,
        "death_frame": None,
        "survived_cap": True,
    }
    assert report["summary"] == {
        "mean_pipes": 0.0,
        "median_pipes": 0.0,
        "p10_pipes": 0.0,
        "p90_pipes": 0.0,
        "minimum_pipes": 0,
        "cap_survival_rate": 1.0,
    }


def test_summarize_uses_declared_metrics():
    assert _summarize(
        [0, 10, 20, 30, 40], [False, False, False, True, True]
    ) == {
        "mean_pipes": 20.0,
        "median_pipes": 20.0,
        "p10_pipes": 4.0,
        "p90_pipes": 36.0,
        "minimum_pipes": 0,
        "cap_survival_rate": 0.4,
    }


def test_evaluate_champion_rejects_nonpositive_horizon(tmp_path):
    path = tmp_path / "unused.json"
    with pytest.raises(ValueError, match="^max_frames must be positive$"):
        evaluate_champion(path, 0, 0, 0)


def test_evaluate_champion_rejects_reversed_seed_range(tmp_path):
    path = tmp_path / "unused.json"
    with pytest.raises(
        ValueError, match="^seed_stop must be greater than or equal to seed_start$"
    ):
        evaluate_champion(path, 1, 0, 1)


def test_cli_rejects_reversed_seed_range(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_champion.py",
            "--genome",
            str(tmp_path / "unused.json"),
            "--seed-start",
            "1",
            "--seed-stop",
            "0",
            "--max-frames",
            "1",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2
    assert "seed_stop must be greater than or equal to seed_start" in capsys.readouterr().err
