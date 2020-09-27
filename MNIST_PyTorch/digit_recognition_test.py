#!/usr/bin/env python3
import os
import torch
from PIL import Image
from typing import List, Tuple, Iterator

from read_data import test_data
from digit_recognition_train import DigitNeural, SAVE_PATH

LOAD_PATH = SAVE_PATH
ALL_LOG_PATH = os.path.join(os.path.curdir, "test.log")
ERR_LOG_PATH = os.path.join(os.path.curdir, "test_err.log")
ERR_DIR = os.path.join(os.path.curdir, "test_error")
MAG_RATIO = 4


def mse(output: List[float], label: int) -> float:
    output_copy = output[:]
    expected = output_copy.pop(label)
    sum_nonexpected = sum(n * n for n in output_copy)
    return sum_nonexpected + (1.0 - expected) ** 2


def fl2s(li: List[float]) -> str:
    return "[" + ", ".join(f"{n:.3f}" for n in li) + "]"


def generate_img(pixel_data: List[List[float]], name: str, mag: int = 1) -> None:
    pixel_data_int = [[round(n*255) for n in row] for row in pixel_data]
    img = Image.new("L", (28*mag, 28*mag))
    for x in range(28*mag):
        for y in range(28*mag):
            img.putpixel((y, x), pixel_data_int[x//mag][y//mag])
    img.save(name)
    print(f"    Saved {name}!")


def test_cuda(digit_neural: DigitNeural, tests: Iterator[Tuple[int, List[List[List[float]]]]]) -> None:
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
            image_tensor: torch.Tensor = torch.tensor([image])
            # print("    Images tensor got!")
            image_tensor = image_tensor.to(cuda_device)
            # print("    Images tensor moved to CUDA!")
            output: List[float] = digit_neural(image_tensor)[0].tolist()
            # print("    Neural net run!")
            error = mse(output, label)
            tot_error += error
            # print("    Error totaled!")
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
            # print("    Counting done!")
            print(
                label,
                int(success),
                f"{error:.6f}",
                fl2s(output),
                file=file_obj_all
            )
            # print("    Logged!")
    with open(ERR_LOG_PATH, "w") as file_obj_err:
        for err in sorted(errs):
            print(*err, file=file_obj_err)
    print("Errors written!")
    total = correct + wrong
    print(f"Correct: {correct}/{total} ({correct/total*100:.2f}%)")
    print(f"Wrong  : {wrong}/{total} ({wrong/total*100:.2f}%)")
    print(f"Tot err: {tot_error}")
    print(f"Avg err: {tot_error/total}")


def test_cpu(digit_neural: DigitNeural, tests: Iterator[Tuple[int, List[List[List[float]]]]]) -> None:
    correct = wrong = 0
    tot_error = 0.0
    errs: List[Tuple[int, int, int, int, str, str]] = []
    with torch.no_grad(), open(ALL_LOG_PATH, "w") as file_obj_all:
        for i, (label, image) in enumerate(tests, 1):
            if not i % 1000:
                print(f"  Test {i}")
            image_tensor: torch.Tensor = torch.tensor([image])
            # print("    Images tensor got!")
            output: List[float] = digit_neural(image_tensor)[0].tolist()
            # print("    Neural net run!")
            error = mse(output, label)
            tot_error += error
            # print("    Error totaled!")
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
            # print("    Counting done!")
            print(
                label,
                int(success),
                f"{error:.6f}",
                fl2s(output),
                file=file_obj_all
            )
            # print("    Logged!")
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
    digit_neural = DigitNeural()
    digit_neural_dict = torch.load(LOAD_PATH)
    digit_neural.load_state_dict(digit_neural_dict)
    print("Neural net loaded!")
    tests = test_data()
    print("Testing data obtained!")
    if not os.path.exists(ERR_DIR):
        os.mkdir(ERR_DIR)
    else:
        for file in os.listdir(ERR_DIR):
            if file.endswith(".png"):
                path = os.path.join(ERR_DIR, file)
                os.remove(path)
    print("Error directory initiated!")
    if torch.cuda.is_available():
        test_cuda(digit_neural, tests)
    else:
        test_cpu(digit_neural, tests)


if __name__ == "__main__":
    main()
