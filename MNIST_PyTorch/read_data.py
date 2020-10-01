#!/usr/bin/env python3
"""Functions to read data from `.idx?-ubyte` files"""

import os.path
from struct import unpack
from typing import List, Tuple, Iterator

# `.idx?-ubyte` file paths
TRAIN_FILES = (
    os.path.join(os.path.dirname(__file__), "data", "train-labels.idx1-ubyte"),
    os.path.join(os.path.dirname(__file__), "data", "train-images.idx3-ubyte"),
)
TEST_FILES = (
    os.path.join(os.path.dirname(__file__), "data", "t10k-labels.idx1-ubyte"),
    os.path.join(os.path.dirname(__file__), "data", "t10k-images.idx3-ubyte"),
)

# Type hints
# Label is an integer
Label = int
# SampleLabels is a list of Labels (all labels in the sample)
SampleLabels = List[Label]
# Pixel is a float (0.0-1.0, step 1/255)
Pixel = float
# Image is a list of list of float (28×28)
Image = List[List[Pixel]]
# ChanneledImage is a list of Images (actually only 1 Image)
# This extra layer is required as PyTorch expects a "channel"
ChanneledImage = List[Image]
# SampleImages is a list of channels (all images in the sample)
SampleImages = List[ChanneledImage]


def read_labels(fn: str, count: int) -> SampleLabels:
    """Read label files (`-labels`)

    Args:
        fn    {str}: File path
        count {int}: Expected number of labels

    Returns:
        {SampleLabels}: List of integer labels
    """
    with open(fn, "rb") as f:
        assert unpack(">i", f.read(4))[0] == 0x00000801
        assert unpack(">i", f.read(4))[0] == count
        labels: List[int] = [
            ord(f.read(1))
            for _ in range(count)
        ]
    return labels


def read_images(fn: str, count: int) -> SampleImages:
    """Read image files (`-images`)

    Args:
        fn    {str}: File path
        count {int}: Expected number of images

    Returns:
        {SampleImages}:
            List of lists, each with one image.
            This extra layer is required as PyTorch expects a "channel".
            Each image itself is a list of list of float (28×28).
            Each float is a pixel's brightness (0.0-1.0)
    """
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


def get_data(files: Tuple[str, str], count: int) -> Tuple[SampleLabels, SampleImages]:
    """Get data for each scenario

    Args:
        files {Tuple[str, str]}: Label and image data file paths
        count {int}            : Expected number of tests

    Returns:
        {SampleLabels}: All labels
        {SampleImages}: All image data
    """
    labels = read_labels(files[0], count)
    images = read_images(files[1], count)
    return labels, images


def train_data() -> Iterator[Tuple[Label, ChanneledImage]]:
    """Get data for training

    Returns:
        {Iterator[Tuple[Label, ChanneledImage]]}: Iterator for each label-image pair
    """
    return zip(*get_data(TRAIN_FILES, 60000))


def test_data() -> Iterator[Tuple[Label, ChanneledImage]]:
    """Get data for testing

    Returns:
        {Iterator[Tuple[Label, ChanneledImage]]}: Iterator for each label-image pair
    """
    return zip(*get_data(TEST_FILES, 10000))
