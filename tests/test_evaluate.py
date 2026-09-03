import numpy as np

from src import evaluate as ev


def test_boundary_errors_and_iou():
    r = ev.boundary_errors((5.0, 15.0), (4.5, 15.5))
    assert abs(r["start_err"] - 0.5) < 1e-9
    assert abs(r["end_err"] - 0.5) < 1e-9
    assert abs(r["iou"] - 10.0 / 11.0) < 1e-6


def test_iou_is_zero_for_disjoint_intervals():
    assert ev.boundary_errors((1.0, 2.0), (10.0, 12.0))["iou"] == 0.0


def test_iou_is_one_for_identical_intervals():
    assert abs(ev.boundary_errors((3.0, 9.0), (3.0, 9.0))["iou"] - 1.0) < 1e-9


def test_summarise_reports_hit_rates():
    res = [{"start_err": 0.2, "end_err": 0.3, "iou": 0.9},
           {"start_err": 1.4, "end_err": 0.4, "iou": 0.7},
           {"start_err": 0.1, "end_err": 0.1, "iou": 0.95}]
    s = ev.summarise(res)
    assert abs(s["start_mae"] - (0.2 + 1.4 + 0.1) / 3) < 1e-9
    assert abs(s["start_within_0.5"] - 2 / 3) < 1e-9
    assert abs(s["median_iou"] - 0.9) < 1e-9


def test_summarise_handles_empty_input():
    assert ev.summarise([])["n"] == 0


def test_report_names_the_winner_honestly():
    strong = ev.summarise([{"start_err": 0.1, "end_err": 0.1, "iou": 0.95}] * 5)
    weak = ev.summarise([{"start_err": 3.0, "end_err": 3.0, "iou": 0.3}] * 5)
    assert "модель лучше" in ev.report(strong, weak)
    # если базовый детектор не хуже, отчёт обязан это сказать
    assert "классический детектор не хуже" in ev.report(weak, strong)


def test_report_states_when_there_is_nothing_to_evaluate():
    assert "нет данных" in ev.report(ev.summarise([]), ev.summarise([]))
