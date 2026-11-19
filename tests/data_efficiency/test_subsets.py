from src.data_efficiency.subsets import nested_subsets

IDS = [f"img_{i:04d}" for i in range(1369)]


def test_subsets_nested_and_sized():
    s = nested_subsets(IDS, seed=0)
    assert [len(s[f]) for f in (0.5, 0.25, 0.1)] == [684, 342, 137]
    assert set(s[0.1]) <= set(s[0.25]) <= set(s[0.5]) <= set(IDS)


def test_subsets_deterministic_and_seed_sensitive():
    assert nested_subsets(IDS, seed=0) == nested_subsets(IDS, seed=0)
    assert nested_subsets(IDS, seed=0)[0.1] != nested_subsets(IDS, seed=1)[0.1]
