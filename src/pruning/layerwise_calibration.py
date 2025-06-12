import copy
import time
import gc

import numpy as np
import torch

from src.examples.loading_pretrained_models import load_pretrained_CleanUMamba
from src.pruning.importance import get_prune_channels
from src.pruning.pruninggroup import CleanUMambaPrunableChannels
from src.pruning.util import run_forward
from src.util.util import loss_fn

def normalize_scales(scales):
    # Get largest scale from the dictionary
    max_scale = max(scales.values())
    # Scale everything with that
    for key in scales:
        scales[key] /= max_scale

    return scales, max_scale

class calibrator:
    def __init__(self, ema_factor=1, min_scale=0.0000001):
        self.scales = {}
        self.ema_factor = ema_factor
        self.min_scale = min_scale

    def gather(self, model, prune_groups, loss_fn, importanec_metric, root, batch_size, loss_samples, random_seed):
        """Gathers the calibration scales by sampling the loss from layer wise pruning, will zero the model gradients and expects the gradients to be zero before hand"""
        scales, offsets, layer_wise_results = get_calibration(model, prune_groups, loss_fn, f"n_parameters*{importanec_metric}", root, False, batch_size, loss_samples, random_seed)

        for group, scale in scales.items():
            if group in self.scales:
                self.scales[group] = max(self.scales[group] * (1 - self.ema_factor) + scale * self.ema_factor, self.min_scale)
            else:
                self.scales[group] = max(scales[group], self.min_scale)

        # Reset gradients
        model.zero_grad()

    def scale(self, importances, group):
        scale = self.scales.get(group.name, 36)
        return importances * scale

    def log(self, log_file):
        normalized_scales, max_scale = normalize_scales(self.scales)

        log_file[f"Prune/calibration_scales/max_scale"] = max_scale

        for group, scale in normalized_scales.items():
            log_file[f"Prune/calibration_scales/{group}"] = scale


        return log_file


def get_calibration(model, prune_groups, loss_fn, importanec_metric, root, two_point=False, batch_size=2, loss_samples=16, random_seed=42):
    prune_group_percentages = [0.1, 0.4] if two_point else [0.2]
    layer_wise_results = calibrate_prune_groups(model, prune_groups, prune_group_percentages, loss_fn, importanec_metric, root, loss_samples,
                                                random_seed, batch_size)
    low_point = layer_wise_results[0]
    scales = {}
    offsets = {}

    for r in layer_wise_results:
        if two_point:
            if r["prune_percentage"] < 0.15:
                low_point = r
                continue

            if r["prune_percentage"] < 0.35 or r["prune_percentage"] > 0.45:
                continue

        if two_point:
            dx = r["total_importance"] - low_point["total_importance"]
            dy = r["loss_change"] - low_point["loss_change"]

            offsets[r["group"]] = low_point["total_importance"] - low_point["loss_change"] * dx / dy
            scales[r["group"]] = r["loss_change"] / (r["total_importance"] - offsets[r["group"]])
        else:
            offsets[r["group"]] = 0
            scales[r["group"]] = r["loss_change"] / r["total_importance"]

    return scales, offsets, layer_wise_results


def calibrate_prune_groups(model, prune_groups, prune_group_percentages, loss_fn, importanec_metric, root, loss_samples=16, random_seed=42, batch_size=2):
    old_random_state = np.random.get_state()

    np.random.seed(random_seed)  # Set the numpy seed to ensure the same random crop in the dataset
    print("running layerwise pruning")
    baseline_losses = run_forward(model, loss_fn, batch_size=batch_size, n_samples=loss_samples, backward=True)
    baseline_loss = sum(baseline_losses) / len(baseline_losses)
    print(f"Baseline loss: {baseline_loss}")

    results = []
    for i, group in enumerate(prune_groups):
        print(f"Pruning group {group.name}")

        importances = group.channel_importances()

        for prune_group_percentage in prune_group_percentages:
            gc.collect()  # Python thing
            torch.cuda.empty_cache()  # PyTorch thing

            # Taylor importance calculation:
            # Per module:  sum |(G * W)|   for the n parameters in the module/filter
            # Per group:  mean(module importance)  for the m modules/filters in the group
            # Final calibration: group * n_filters / n_parameters in order to get the importance per parameter.

            prune_channels, params_pruned, importance_min_dict = get_prune_channels([group],
                                                                                    importanec_metric, None,
                                                                                    prune_group_percentage, 8)

            # print(prune_channels)
            importances = [prune["importance"].item() for prune in prune_channels]
            if len(importances) == 0:
                continue

            mean_importance = sum(importances) / len(importances)
            total_importance = sum(importances)
            remove_index = [prune["index"].item() for prune in prune_channels]
            # print(remove_index)

            # Copy the model and remove forward hooks or the hooks from prune_groups will be copied and trigger
            model_copy = remove_forward_hooks(copy.deepcopy(model))
            copy_prune_channels = CleanUMambaPrunableChannels(model_copy)
            copy_prune_channels[i].prune(remove_index)

            np.random.seed(random_seed)  # Set the numpy seed to ensure the same random crop in the dataset
            losses = run_forward(model_copy, loss_fn, root, batch_size=batch_size, n_samples=loss_samples, backward=False)
            loss = sum(losses) / len(losses)
            print(
                f"n prune: {len(importances)} - prune %: {(100 * (group.n_channels - copy_prune_channels[i].n_channels) / group.n_channels):.2f} - importance: {total_importance:.3f} - loss-change: {(loss - baseline_loss):.3f}")

            result = {
                "group": group.name,
                "prune_percentage": (group.n_channels - copy_prune_channels[i].n_channels) / group.n_channels,
                "prune_parameters": params_pruned,
                "prune_groups": len(importances),
                "mean_importance": mean_importance,
                "total_importance": total_importance,
                "loss_change": (loss - baseline_loss)
            }
            results.append(result)

    # Restore the random seed
    np.random.set_state(old_random_state)

    return results

def remove_forward_hooks(model):
    from collections import OrderedDict

    for module in model.modules():
        module._forward_pre_hooks = OrderedDict()
        module._forward_hooks = OrderedDict()
    return model

def test_importance_per_layer(loss_samples = 512, batch_size = 2, sample_size = 6, n_remove = 4):
    # Disable the causal conv1d functions to enable the callback hook for mamba's conv1d
    import mamba_ssm.modules.mamba_simple as mamba_ssm
    mamba_ssm.causal_conv1d_fn, mamba_ssm.causal_conv1d_update = None, None

    model = load_pretrained_CleanUMamba().eval()

    prune_groups = CleanUMambaPrunableChannels(model, statistics=True)

    np.random.seed(42)  # Set the numpy seed to ensure the same random crop in the dataset for repeatability
    baseline_losses = run_forward(model, loss_fn, batch_size=batch_size, n_samples=loss_samples, backward=True)
    baseline_loss = sum(baseline_losses) / len(baseline_losses)
    print(f"Baseline loss: {baseline_loss}")

    results = []
    for i, group in enumerate(prune_groups):
        print(f"Pruning group {group.name}")

        for j in range(sample_size):
            gc.collect()  # Python thing
            torch.cuda.empty_cache()  # PyTorch thing

            # Prune a random channel index
            # remove_index = torch.randint(0, group.n_channels, (n_remove,))
            remove_index = torch.randperm(group.n_channels)[:n_remove]

            print(f"pruning {remove_index}")

            # Copy the model and remove forward hooks or the hooks from prune_groups will be copied and trigger
            model_copy = remove_forward_hooks(copy.deepcopy(model))

            copy_prune_channels = CleanUMambaPrunableChannels(model_copy)
            copy_prune_channels[i].prune(remove_index)

            np.random.seed(42)  # Set the numpy seed to ensure the same random crop in the dataset
            losses = run_forward(model_copy, loss_fn, batch_size=batch_size, n_samples=loss_samples, backward=False)
            loss = sum(losses) / len(losses)
            print(f"loss: {loss}")

            importances = group.channel_importances()

            result = {
                "group": group.name,
                "remove_index": remove_index.cpu().numpy().tolist(),
                "n_channels": group.n_channels,
                "weight_imp": importances["weight"][remove_index].mean().detach().cpu().item(),
                "taylor_ind_imp": importances["taylor_individual"][remove_index].mean().detach().cpu().item(),
                "taylor_gro_imp": importances["taylor_group"][remove_index].mean().detach().cpu().item(),
                "grad_imp": importances["grad"][remove_index].mean().detach().cpu().item(),
                "act_var": importances["act_var"][remove_index].mean().detach().cpu().item() if importances["act_var"] is not None else None,
                "param_per_channel": importances["n_parameters"],
                # "stoi_change": (baseline_stoi - pruned_stoi)/baseline_stoi,
                "loss_change": (loss - baseline_loss)/baseline_loss
            }
            results.append(result)

    torch.save({"results": results, "loss_samples": loss_samples, "sample_size": sample_size, "n_remove": n_remove},
               f"taylor_importance_loss_change_data_{int(time.time())}.pkl")

    print(results)
    return results

def scatter_importance_per_layer(results = None):
    from matplotlib import pyplot as plt
    import pandas as pd

    if results is None:
        # load data that could have been saved in test_importance_per_layer
        saved_data = torch.load("taylor_importance_loss_change_data_1748309612.pkl")
        results = saved_data["results"]
        del saved_data

    df = pd.DataFrame(results)
    print(df)

    groups = df["group"].unique()

    collumns = ["weight_imp", "taylor_ind_imp", "taylor_gro_imp", "grad_imp", "act_var", "param_per_channel", "loss_change"]

    def scatter_loss(metric = "weight_imp", mult_metric=None, div_metric=None):
        fig = plt.figure(figsize=(12, 6))
        plt.grid()
        for i, group in enumerate(df["group"].unique()):
            importances = np.array([r[metric] for r in results if r["group"] == group])
            if mult_metric:
                importances *= np.array([r[mult_metric] for r in results if r["group"] == group])
            elif div_metric:
                importances /= np.array([r[div_metric] for r in results if r["group"] == group])

            loss_changes = np.array([r["loss_change"] for r in results if r["group"] == group])
            plt.scatter(importances, loss_changes, label=group)

        if mult_metric:
            plt.title(f'{metric} * {mult_metric} vs Loss Change')
            plt.xlabel(f'{metric} * {mult_metric}')
        elif div_metric:
            plt.title(f'{metric} / {div_metric} vs Loss Change')
            plt.xlabel(f'{metric} / {div_metric}')
        else:
            plt.title(f'{metric} vs Loss Change')
            plt.xlabel(f'{metric}')

        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('Loss Change')
        plt.legend(loc='upper right', bbox_to_anchor=(0.8, 0.98))
        plt.show()

    # scatter_loss()
    scatter_loss("taylor_ind_imp")
    scatter_loss("taylor_gro_imp")


if __name__ == "__main__":
    # results = test_importance_per_layer()

    scatter_importance_per_layer()