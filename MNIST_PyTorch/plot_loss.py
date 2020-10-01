#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
from typing import List, Dict

from digit_recognition_train import LOSS_JSON_PATH


def plot() -> None:
    """Plot the loss and see how it progresses"""
    with open(LOSS_JSON_PATH, "r") as file_obj:
        data: List[List[float]] = json.load(file_obj)
    # Flatten data
    data_dict: Dict[str, float] = {
        f"{epoch_id+1}.{(group_id+1)*100}": group
        for epoch_id, epoch in enumerate(data)
        for group_id, group in enumerate(epoch)
    }
    keys = data_dict.keys()
    values = data_dict.values()
    keys_ints = range(len(keys))
    # Plotting
    _, axs = plt.subplots(2, 1)
    ax_linear, ax_semilog = axs
    ax_linear.grid(True)
    ax_linear.plot(keys_ints, values)
    ax_semilog.grid(True)
    ax_semilog.semilogy(keys_ints, values)
    plt.show()


if __name__ == "__main__":
    plot()
