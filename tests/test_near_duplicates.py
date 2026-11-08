import random

from ml.find_near_duplicates import BKTree, perceptual_hash
from PIL import Image


def test_bktree_matches_exhaustive_search_including_equal_hashes():
    rng = random.Random(42)
    values = [rng.getrandbits(63) for _ in range(50)] + [12, 12, 13, 15]
    tree = BKTree()
    for index, value in enumerate(values):
        tree.add(value, str(index))
    for query in (12, 13, values[0]):
        for radius in (0, 2, 6, 63):
            expected = sorted((str(i), (query ^ value).bit_count()) for i, value in enumerate(values)
                              if (query ^ value).bit_count() <= radius)
            assert sorted(tree.neighbors(query, radius)) == expected


def test_hash_is_deterministic_for_same_image(tmp_path):
    image = tmp_path / "one.png"
    Image.new("RGB", (100, 50), (100, 20, 40)).save(image)
    assert perceptual_hash(image) == perceptual_hash(image)
    assert 0 <= perceptual_hash(image) < 2**63
