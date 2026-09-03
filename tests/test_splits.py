import json

from src import splits


def test_block_split_keeps_ids_contiguous():
    ids = list(range(1, 101))
    d = splits.block_split(ids, (0.7, 0.15, 0.15))
    assert d["train"] == list(range(1, 71))
    assert d["val"] == list(range(71, 86))
    assert d["test"] == list(range(86, 101))


def test_block_split_never_interleaves():
    """Случайное разбиение дало бы утечку: соседние номера — один испытуемый."""
    ids = list(range(1, 51))
    d = splits.block_split(ids)
    assert max(d["train"]) < min(d["val"])
    assert max(d["val"]) < min(d["test"])


def test_block_split_covers_every_id_exactly_once():
    ids = [3, 9, 14, 27, 31, 55, 78, 90]
    d = splits.block_split(ids)
    assert sorted(d["train"] + d["val"] + d["test"]) == sorted(ids)


def test_block_split_handles_tiny_input():
    d = splits.block_split([1, 2, 3])
    assert sorted(d["train"] + d["val"] + d["test"]) == [1, 2, 3]


def test_split_verified_is_disjoint_and_ordered():
    ids = list(range(1, 201))
    calib, test = splits.split_verified(ids, calib_frac=0.3)
    assert set(calib).isdisjoint(test)
    assert len(calib) + len(test) == 200
    assert 55 <= len(calib) <= 65
    assert max(calib) < min(test)


def test_read_verified_keeps_only_accepted_rows(tmp_path):
    f = tmp_path / "v.csv"
    f.write_text("id,start,end,verified\n7,1.5,9.5,1\n8,2.0,8.0,0\n9,3.0,7.0,1\n")
    v = splits.read_verified(f)
    assert set(v) == {7, 9}
    assert v[7] == (1.5, 9.5)


def test_save_and_load_roundtrip(tmp_path):
    f = tmp_path / "s.json"
    d = {"train": [1, 2], "val": [3], "test": [4], "verified_calib": [], "verified_test": []}
    splits.save(d, f)
    assert splits.load(f) == d
    assert json.loads(f.read_text())["train"] == [1, 2]


def test_excluded_ids_never_reach_any_split():
    from src import config

    ids = sorted(set(range(1, 60)) | config.EXCLUDED)
    d = splits.block_split(splits.usable_ids(ids))
    everything = set(d["train"] + d["val"] + d["test"])
    assert everything.isdisjoint(config.EXCLUDED)
