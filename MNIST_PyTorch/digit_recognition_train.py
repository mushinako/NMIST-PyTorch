#!/usr/bin/env python3
import os.path
import random
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
from functools import reduce
from operator import mul
from typing import Generator, List, Tuple, Iterator

from read_data import train_data, Label, SampleLabels, ChanneledImage, SampleImages

# Path to save the trained model
SAVE_PATH = os.path.join(os.path.curdir, "recog_digits.pth")
# Path to save loss data
LOSS_JSON_PATH = os.path.join(os.path.curdir, "loss.json")
# Back propagation configurations
LEARNING_RATE = 0.01
MOMENTUM = 0.9
LOSS_FUNC_FACTORY = nn.MSELoss
# How many iterations of training
EPOCH_LIMIT = 1000

torch.set_default_dtype(torch.double)


class DigitNeural(nn.Module):
    """Neural net class"""

    def __init__(self) -> None:
        super(DigitNeural, self).__init__()
        # 1×28×28 => 6×24×24 (5×5 square convolution)
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 6×24×24 => 6×12×12 (MaxPool 2×2)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        # 6×12×12 => 16×10×10 (3×3 square convolution)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 16×10×10 => 16×5×5 (MaxPool 2×2)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        # 16×5×5 => 400 (flatten)
        self.flatten = lambda x: x.view(-1, 400)
        # 400 => 160 (linear)
        self.fc1 = nn.Linear(400, 160)
        # 160 => 100 (linear)
        self.fc2 = nn.Linear(160, 100)
        # 100 => 10 (linear)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutions and subsampling
        x = f.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = f.relu(self.conv2(x))
        x = self.maxpool2(x)
        # Transform to linear
        x = self.flatten(x)
        # Linear
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class Gen_Tests:
    """Generator of samples to train the neural net"""

    def __init__(self, data: Iterator[Tuple[Label, ChanneledImage]]) -> None:
        self._data = list(data)

    def gen_test(self) -> Generator[Tuple[SampleLabels, SampleImages], None, None]:
        """Generate samples of size 100

        Yields:
            {SampleLabels}: The 100 labels
            {SampleImages}: The 100 images
        """
        random.shuffle(self._data)
        for i in range(0, 60000, 100):
            data_sample = self._data[i:i+100]
            zipped = zip(*data_sample)
            labels: SampleLabels = next(zipped)
            images: SampleImages = next(zipped)
            yield labels, images


def train_cuda(digit_neural: DigitNeural, test_gen: Gen_Tests, loss_func: nn.modules.loss._Loss, optimizer: optim.Optimizer) -> None:
    """Train with CUDA, separated to avoid speculative execution when moving to CUDA"""
    cuda_device = torch.device("cuda")
    digit_neural.to(cuda_device)
    print("Model moved to CUDA!")
    all_loss: List[List[float]] = []
    for epoch in range(1, EPOCH_LIMIT+1):
        epoch_loss: List[float] = []
        loss_sum = 0.0
        print(" Epoch", epoch)
        for i, (labels_sample, images_sample) in enumerate(test_gen.gen_test(), 1):
            if not i % 100:
                print(f"  Loop {i}:", loss_sum)
                epoch_loss.append(loss_sum)
                loss_sum = 0.0
            # Preparations
            images_tensor = torch.tensor(images_sample)
            images_tensor = images_tensor.to(cuda_device)
            optimizer.zero_grad()
            # Run net (forward)
            output = digit_neural(images_tensor)
            # Calculate target
            target = [[0.]*i + [1.] + [0.]*(9-i) for i in labels_sample]
            target_tensor = torch.tensor(target)
            target_tensor = target_tensor.to(cuda_device)
            # Compute loss
            loss: torch.Tensor = loss_func(output, target_tensor)
            loss_sum += loss.item()
            # Backward propagation
            loss.backward()
            # Midify weights
            optimizer.step()
        all_loss.append(epoch_loss)
    print("Finished Training!")
    with open(LOSS_JSON_PATH, "w") as file_obj:
        json.dump(all_loss, file_obj)
    print("Loss data written!")
    torch.save(digit_neural.state_dict(), SAVE_PATH)


def train_cpu(digit_neural: DigitNeural, test_gen: Gen_Tests, loss_func: nn.modules.loss._Loss, optimizer: optim.Optimizer) -> None:
    """Train with CPU"""
    all_loss: List[List[float]] = []
    for epoch in range(1, EPOCH_LIMIT+1):
        epoch_loss: List[float] = []
        loss_sum = 0.0
        print(" Epoch", epoch)
        for i, (labels_sample, images_sample) in enumerate(test_gen.gen_test(), 1):
            if not i % 100:
                print(f"  Loop {i}:", loss_sum)
                epoch_loss.append(loss_sum)
                loss_sum = 0.0
            # Preparations
            images_tensor = torch.tensor(images_sample)
            optimizer.zero_grad()
            # Run net (forward)
            output = digit_neural(images_tensor)
            # Calculate target
            target = [[0.]*i + [1.] + [0.]*(9-i) for i in labels_sample]
            target_tensor = torch.tensor(target)
            # Compute loss
            loss: torch.Tensor = loss_func(output, target_tensor)
            loss_sum += loss.item()
            # Backward propagation
            loss.backward()
            # Midify weights
            optimizer.step()
        all_loss.append(epoch_loss)
    print("Finished Training!")
    with open(LOSS_JSON_PATH, "w") as file_obj:
        json.dump(all_loss, file_obj)
    print("Loss data written!")
    torch.save(digit_neural.state_dict(), SAVE_PATH)


def main() -> None:
    # Get data
    test_gen = Gen_Tests(train_data())
    # Create and configure neural net
    digit_neural = DigitNeural()
    loss_func = LOSS_FUNC_FACTORY()
    optimizer = optim.SGD(
        digit_neural.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM
    )
    if torch.cuda.is_available():
        print("Training with CUDA")
        train_cuda(digit_neural, test_gen, loss_func, optimizer)
    else:
        print("Training with CPU")
        train_cpu(digit_neural, test_gen, loss_func, optimizer)


if __name__ == "__main__":
    main()
