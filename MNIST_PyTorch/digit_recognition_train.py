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

from read_data import train_data

SAVE_PATH = os.path.join(os.path.curdir, "recog_digits.pth")
LOSS_JSON_PATH = os.path.join(os.path.curdir, "loss.json")
LEARNING_RATE = 0.01
MOMENTUM = 0.9
LOSS_FUNC_FACTORY = nn.MSELoss
EPOCH_LIMIT = 1000

torch.set_default_dtype(torch.double)


class DigitNeural(nn.Module):
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

    @staticmethod
    def num_flat_features(x: torch.Tensor) -> int:
        extra_dims: torch.Size = x.size()[1:]
        num_features: int = reduce(mul, extra_dims, 1)
        return num_features


class Gen_Tests:
    def __init__(self, data: Iterator[Tuple[int, List[List[List[float]]]]]) -> None:
        self._data = list(data)

    def gen_test(self) -> Generator[Tuple[List[int], List[List[List[List[float]]]]], None, None]:
        random.shuffle(self._data)
        for i in range(0, 60000, 100):
            data_sample = self._data[i:i+100]
            zipped = zip(*data_sample)
            labels: Tuple[int, ...] = next(zipped)
            images: Tuple[List[List[List[float]]], ...] = next(zipped)
            yield labels, images


def train_cuda(digit_neural: DigitNeural, test_gen: Gen_Tests, loss_func: nn.modules.loss._Loss, optimizer: optim.Optimizer) -> None:
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
            images_tensor = torch.tensor(images_sample)
            # print("    Images tensor got!")
            images_tensor = images_tensor.to(cuda_device)
            # print("    Images tensor moved to CUDA!")
            optimizer.zero_grad()
            # print("    Gradient zeroed!")
            output = digit_neural(images_tensor)
            # print("    Neural net run!")
            target = [[0.]*i + [1.] + [0.]*(9-i) for i in labels_sample]
            target_tensor = torch.tensor(target)
            # print("    Target tensor got!")
            target_tensor = target_tensor.to(cuda_device)
            # print("    Target tensor moved to CUDA!")
            loss: torch.Tensor = loss_func(output, target_tensor)
            loss_sum += loss.item()
            # print("    Loss got!", loss.item())
            loss.backward()
            # print("    Backward propagated!")
            optimizer.step()
            # print("    Weights modified!")
        all_loss.append(epoch_loss)
    print("Finished Training!")
    with open(LOSS_JSON_PATH, "w") as file_obj:
        json.dump(all_loss, file_obj)
    print("Loss data written!")
    torch.save(digit_neural.state_dict(), SAVE_PATH)


def train_cpu(digit_neural: DigitNeural, test_gen: Gen_Tests, loss_func: nn.modules.loss._Loss, optimizer: optim.Optimizer) -> None:
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
            images_tensor = torch.tensor(images_sample)
            # print("    Images tensor got!")
            optimizer.zero_grad()
            # print("    Gradient zeroed!")
            output = digit_neural(images_tensor)
            # print("    Neural net run!")
            target = [[0.]*i + [1.] + [0.]*(9-i) for i in labels_sample]
            target_tensor = torch.tensor(target)
            # print("    Target tensor got!")
            loss: torch.Tensor = loss_func(output, target_tensor)
            loss_sum += loss.item()
            # print("    Loss got!", loss.item())
            loss.backward()
            # print("    Backward propagated!")
            optimizer.step()
            # print("    Weights modified!")
        all_loss.append(epoch_loss)
    print("Finished Training!")
    with open(LOSS_JSON_PATH, "w") as file_obj:
        json.dump(all_loss, file_obj)
    print("Loss data written!")
    torch.save(digit_neural.state_dict(), SAVE_PATH)


def main() -> None:
    digit_neural = DigitNeural()
    print("Neural net initialized!")
    test_gen = Gen_Tests(train_data())
    print("Training data obtained!")
    loss_func = LOSS_FUNC_FACTORY()
    print("Loss function obtained!")
    optimizer = optim.SGD(digit_neural.parameters(),
                          lr=LEARNING_RATE, momentum=MOMENTUM)
    print("Optimizer set!")
    if torch.cuda.is_available():
        train_cuda(digit_neural, test_gen, loss_func, optimizer)
    else:
        train_cpu(digit_neural, test_gen, loss_func, optimizer)


if __name__ == "__main__":
    main()
