from collections import defaultdict
import os

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from torch.utils.tensorboard import SummaryWriter
import psutil


def ema(new_value, old_values, alpha=0.99, steps=1):
    """
    compute exponential moving average estimate for current time step
    given new_value (float) and old_values (list)
    optionally specify number of steps to perform the EMA update over."
    """
    value = old_values[-1] if len(old_values) > 0 else new_value
    for _ in range(steps):
        value = alpha * value + (1 - alpha) * new_value
    return value

class Metrics:
    """
    helper class used to log metrics to tensorboard and create matplotlib metrics plots.
    This is necessary because tensorboard is not supported on Mac M1 chip.
    The two methods add_scalar and add_scalars follow the same syntax as tensorboard.
    """
    def __init__(self, tensorboard: bool = False, log_dir: str = None):
        self.tensorboard = tensorboard
        self.writer = None
        if tensorboard:
            self.writer = SummaryWriter(log_dir=log_dir)

        self.ema_alpha = 0.99  # alpha value for matplotlib metrics.
        # metrics holds metric name -> {step, value, ema_value, label}
        # the same metric can have multiple labels. they will be plotted
        # on the same graph and the labels will be shown in the legend.
        self.metrics = defaultdict(list)

        if not tensorboard:
            plt.ion()  # Turn on interactive mode

    # util function to log CPU and RAM usage.
    def log_system_usage(self, step: int):
        # CPU & RAM
        process = psutil.Process(os.getpid())
        cpu_usage = psutil.cpu_percent()
        ram_usage = process.memory_info().rss / 1024 ** 2  # in MB
        self.add_scalar("ram_utilization", ram_usage, step)
        self.add_scalar("cpu_utilization", cpu_usage, step)

    def add_scalar(self, name, value, step):
        if self.tensorboard:
            # new_style=True is required for the label argument.
            self.writer.add_scalar(name, value, step)
        else:
            old_values = np.array([l["value"] for l in self.metrics[name]])
            old_steps = [l["step"] for l in self.metrics[name]]
            label_mask = [i for i, l in enumerate(self.metrics[name]) if l["label"] == "default"]
            last_step = old_steps[label_mask[-1]] if (old_steps and label_mask) else 0
            ema_value = ema(new_value=value, old_values=old_values[label_mask], alpha=self.ema_alpha, steps=step - last_step)
            self.metrics[name].append({
                "step": step,
                "value": value,
                "ema_value": ema_value,
                "label": "default",
            })

    def add_scalars(self, name, values, step):
        if self.tensorboard:
            self.writer.add_scalars(main_tag=name, tag_scalar_dict=values, global_step=step)
        else:
            # same code as `add_scalar` for each label, value pair
            for label, value in values.items():
                old_values = np.array([l["value"] for l in self.metrics[name]])
                old_steps = [l["step"] for l in self.metrics[name]]
                label_mask = [i for i, l in enumerate(self.metrics[name]) if l["label"] == label]
                last_step = old_steps[label_mask[-1]] if (old_steps and label_mask) else 0
                ema_value = ema(new_value=value, old_values=old_values[label_mask], alpha=self.ema_alpha, steps=step - last_step)
                self.metrics[name].append({
                    "step": step,
                    "value": value,
                    "ema_value": ema_value,
                    "label": label,
                })

    def plot(self):
        if self.tensorboard:
            print("can't call plot when tensorboard=True.")
        
        fig, axs = plt.subplots(nrows=len(self.metrics), figsize=(10, 20), sharex=True)
        i = 0
        for metric_name, metric_values in self.metrics.items():
            steps = np.array([l["step"] for l in metric_values])
            values = np.array([l["value"] for l in metric_values])
            ema_values = np.array([l["ema_value"] for l in metric_values])
            labels = [l["label"] for l in metric_values]

            # group by label
            for label in set(labels):
                label_mask = [i for i, l in enumerate(labels) if l == label]
                axs[i].plot(
                    steps[label_mask], values[label_mask],
                    marker="+",
                    color="blue",
                    alpha=0.2,
                    label=label if label != "default" else None
                )
                ema_label = f"{ema_values[-1]}" if len(ema_values) > 0 else ""
                ema_label = label + " " + ema_label if label != "default" else ema_label
                axs[i].plot(
                    steps[label_mask], ema_values[label_mask],
                    color="orange",
                    label=ema_label,
                )
                axs[i].set_title(metric_name)
                axs[i].grid(True)
            i += 1

        # add a marker with the epoch at the corresponding step
        # this removes the marker... TODO: add marker at every epoch
        # if step % len(train_set) == 1:
        #     for ax in axs:
        #         ax.axvline(x=step, color="red", linestyle="--", alpha=0.2, label=f"epoch {epoch}")
        clear_output(wait=True)
        plt.tight_layout()
        plt.show()

    def close(self):
        if self.tensorboard:
            self.writer.close()
        else:
            plt.ioff()