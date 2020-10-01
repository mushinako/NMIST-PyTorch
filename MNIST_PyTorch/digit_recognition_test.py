#!/usr/bin/env python3
import os
import torch
from PIL import Image as Img
from typing import List, Tuple, Iterator

from read_data import test_data, Label, Image, ChanneledImage
from digit_recognition_train import DigitNeural, SAVE_PATH

LOAD_PATH = SAVE_PATH
ALL_LOG_PATH = os.path.join(os.path.curdir, "test.log")
ERR_LOG_PATH = os.path.join(os.path.curdir, "test_err.log")
ERR_DIR = os.path.join(os.path.curdir, "test_error")
MAG_RATIO = 4


def mse(output: List[float], label: Label) -> float:
    """Mean squared error

    Args:
        output {List[float]}: Final output of the neural network (length 10)
        label  {Label}      : The correct number

    Returns:
        {float}: The MSE of the output
    """
    output_copy = output[:]
    expected = output_copy.pop(label)
    sum_nonexpected = sum(n * n for n in output_copy)
    return (sum_nonexpected + (1.0 - expected) ** 2)/len(output)


def fl2s(li: List[float]) -> str:
    """Convert list of float to string, with 3 digits after decimal point

    Args:
        li {List[float]}: List of float to be converted

    Returns:
        {str}: The converted string
    """
    return "[" + ", ".join(f"{n:.3f}" for n in li) + "]"


def generate_img(pixel_data: Image, name: str, mag: int = 1) -> None:
    """Generate actual image file

    Args:
        pixel_data {Image}: The data for all pixels
        name       {str}  : Name to save the pic as
        mag        {int}  : Magnification
    """
    # Convert 0-1 to 0-255 scale
    pixel_data_int = [[round(n*255) for n in row] for row in pixel_data]
    img = Img.new("L", (28*mag, 28*mag))
    for x in range(28*mag):
        for y in range(28*mag):
            img.putpixel((y, x), pixel_data_int[x//mag][y//mag])
    img.save(name)
    print(f"    Saved {name}!")


def test_cuda(digit_neural: DigitNeural, tests: Iterator[Tuple[Label, ChanneledImage]]) -> None:
    """Test with CUDA, separated to avoid speculative execution when moving to CUDA"""
    cuda_device = torch.device("cuda")
    digit_neural.to(cuda_device)
    print("Model moved to CUDA!")
    correct = wrong = 0
    tot_error = 0.0
    errs: List[Tuple[int, int, int, int, str, str]] = []
    with torch.no_grad(), open(ALL_LOG_PATH, "w") as file_obj_all:
        for i, (label, image) in enumerate(tests, 1):
            if not i % 1000:
                print(f"  Test {i}")
            # Preparations
            image_tensor: torch.Tensor = torch.tensor([image])
            image_tensor = image_tensor.to(cuda_device)
            # Run net (forward)
            output: List[float] = digit_neural(image_tensor)[0].tolist()
            # Compute loss
            error = mse(output, label)
            tot_error += error
            # Check success
            success = max(output) == output[label]
            if success:
                correct += 1
            else:
                wrong += 1
                thought = output.index(max(output))
                path = os.path.join(ERR_DIR, f"{label}-{thought}-{i}.png")
                generate_img(image[0], path, MAG_RATIO)
                errs.append((label, thought, i, int(success),
                             f"{error:.6f}", fl2s(output)))
            print(
                label,
                int(success),
                f"{error:.6f}",
                fl2s(output),
                file=file_obj_all
            )
    with open(ERR_LOG_PATH, "w") as file_obj_err:
        for err in sorted(errs):
            print(*err, file=file_obj_err)
    print("Errors written!")
    total = correct + wrong
    print(f"Correct: {correct}/{total} ({correct/total*100:.2f}%)")
    print(f"Wrong  : {wrong}/{total} ({wrong/total*100:.2f}%)")
    print(f"Tot err: {tot_error}")
    print(f"Avg err: {tot_error/total}")


def test_cpu(digit_neural: DigitNeural, tests: Iterator[Tuple[Label, ChanneledImage]]) -> None:
    """Test with CPU"""
    correct = wrong = 0
    tot_error = 0.0
    errs: List[Tuple[int, int, int, int, str, str]] = []
    with torch.no_grad(), open(ALL_LOG_PATH, "w") as file_obj_all:
        for i, (label, image) in enumerate(tests, 1):
            if not i % 1000:
                print(f"  Test {i}")
            # Preparations
            image_tensor: torch.Tensor = torch.tensor([image])
            # Run net (forward)
            output: List[float] = digit_neural(image_tensor)[0].tolist()
            # Compute loss
            error = mse(output, label)
            tot_error += error
            # Check success
            success = max(output) == output[label]
            if success:
                correct += 1
            else:
                wrong += 1
                thought = output.index(max(output))
                path = os.path.join(ERR_DIR, f"{label}-{thought}-{i}.png")
                generate_img(image[0], path, MAG_RATIO)
                errs.append((label, thought, i, int(success),
                             f"{error:.6f}", fl2s(output)))
            print(
                label,
                int(success),
                f"{error:.6f}",
                fl2s(output),
                file=file_obj_all
            )
    with open(ERR_LOG_PATH, "w") as file_obj_err:
        for err in sorted(errs):
            print(*err, file=file_obj_err)
    print("Errors written!")
    total = correct + wrong
    print(f"Correct: {correct}/{total} ({correct/total*100:.2f}%)")
    print(f"Wrong  : {wrong}/{total} ({wrong/total*100:.2f}%)")
    print(f"Tot err: {tot_error}")
    print(f"Avg err: {tot_error/total}")


def main() -> None:
    # Get data
    tests = test_data()
    # Create and configure neural net
    digit_neural = DigitNeural()
    digit_neural_dict = torch.load(LOAD_PATH)
    digit_neural.load_state_dict(digit_neural_dict)
    # Error logging preparation
    if not os.path.exists(ERR_DIR):
        os.mkdir(ERR_DIR)
    else:
        for file in os.listdir(ERR_DIR):
            if file.endswith(".png"):
                path = os.path.join(ERR_DIR, file)
                os.remove(path)
    if torch.cuda.is_available():
        print("Testing with CUDA")
        test_cuda(digit_neural, tests)
    else:
        print("Testing with CPU")
        test_cpu(digit_neural, tests)


if __name__ == "__main__":
    main()
