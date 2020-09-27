#!/usr/bin/env python3
import os.path
from struct import unpack
from typing import List, Tuple, Iterator

TRAIN_FILES = (
    os.path.join("data", "train-labels.idx1-ubyte"),
    os.path.join("data", "train-images.idx3-ubyte"),
)
TEST_FILES = (
    os.path.join("data", "t10k-labels.idx1-ubyte"),
    os.path.join("data", "t10k-images.idx3-ubyte"),
)


def read_labels(fn: str, count: int) -> List[int]:
    with open(fn, "rb") as f:
        assert unpack(">i", f.read(4))[0] == 0x00000801
        assert unpack(">i", f.read(4))[0] == count
        labels: List[int] = [
            ord(f.read(1))
            for _ in range(count)
        ]
    return labels


def read_images(fn: str, count: int) -> List[List[List[List[float]]]]:
    with open(fn, "rb") as f:
        assert unpack(">i", f.read(4))[0] == 0x00000803
        assert unpack(">i", f.read(4))[0] == count
        assert unpack(">i", f.read(4))[0] == 28
        assert unpack(">i", f.read(4))[0] == 28
        images: List[List[List[List[int]]]] = [
            [
                [
                    [
                        ord(f.read(1)) / 255
                        for _ in range(28)
                    ]
                    for _ in range(28)
                ]
            ]
            for _ in range(count)
        ]
    return images


def get_data(files: Tuple[str, str], count: int) -> Tuple[List[int], List[List[List[List[float]]]]]:
    labels = read_labels(files[0], count)
    images = read_images(files[1], count)
    return labels, images


def train_data() -> Iterator[Tuple[int, List[List[List[float]]]]]:
    return zip(*get_data(TRAIN_FILES, 60000))


def test_data() -> Iterator[Tuple[int, List[List[List[float]]]]]:
    return zip(*get_data(TEST_FILES, 10000))
