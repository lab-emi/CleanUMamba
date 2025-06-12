# from src.Experiments.Pruning.tp_CleanUMamba import get_example_input
import logging

from src.pruning.importance import get_prune_channels
from src.pruning.pruninggroup import CleanUMambaPrunableChannels

import numpy as np
import torch

from src.examples.loading_pretrained_models import load_pretrained_CleanUMamba
from src.util.dataset import load_CleanNoisyPairDataset
from src.util.util import loss_fn

def get_example_input(batch_size = 1):
    logging.warning("[using_pruning_groups.py:get_example_input()] replace the root with your own DNS-Challenge dataset path")
    root = "/media/sjoerd/storage/boeken/jaar7/thesis/DNS-Challenge"

    dataloader = load_CleanNoisyPairDataset(root, batch_size=batch_size, num_workers=0)
    for i, (clean, noisy, min_max, fileid) in enumerate(dataloader):
        return (clean.cuda(), noisy.cuda())

def test_CleanUMamba_pruningV2():
    import matplotlib.pyplot as plt

    # Disable the causal conv1d functions to enable the callback hook for mamba's conv1d
    import mamba_ssm.modules.mamba_simple as mamba_ssm
    mamba_ssm.causal_conv1d_fn, mamba_ssm.causal_conv1d_update = None, None

    # Load the pretrained model
    model = load_pretrained_CleanUMamba().eval()

    # init the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # init the pruning groups
    prune_groups = CleanUMambaPrunableChannels(model, statistics=True)

    # Example of pruning the model and optimizer twice and running optimization
    for _ in range(2):

        # Accumulate gradients
        model.zero_grad()
        for _ in range(1):
            loss, output_dic = loss_fn(model, get_example_input(batch_size=1))
            loss.backward()

        # Set a step with the optimizer
        optimizer.step()

        _ = get_prune_channels(prune_groups, "taylor_group", 1, None, min_channels_per_group=8)

        # Go over all pruning groups and prune a difference amount of channels in each group
        print("Pruning groups:")
        for i, group in enumerate(prune_groups):
            print(group.name, group.n_channels, len(group.modules), group.modules[0].channel_telemetry()["count"], group.data_len)
            group.prune([n for n in range((i % 20) + 1)], optimizer=optimizer)
            print(group.name, group.n_channels, len(group.modules), group.modules[0].channel_telemetry()["count"], group.data_len)
            for module in group.modules:
                print(module.module)

        del _
        del loss

        # Test running forward with the now pruned model
        out = model(get_example_input(batch_size=1)[1])
        print(out.shape)


    # Print and plot the telemetry on activations that each group has seen
    for group in prune_groups:
        print(group.name, group.n_channels)
        for module in group.modules:
            print(module)

            if module.channel_telemetry()["count"] > 0:

                # side by side mean and variance histograms

                mean_data = module.channel_telemetry()["mean"].cpu().numpy()
                mean_counts, mean_bins = np.histogram(mean_data, bins=50)

                var_data = module.channel_telemetry()["var"].cpu().numpy()
                var_counts, var_bins = np.histogram(var_data, bins=50)

                fig, axs = plt.subplots(1, 2)
                axs[0].stairs(mean_counts, mean_bins)
                axs[0].set_title("Mean")
                axs[1].stairs(var_counts, var_bins)
                axs[1].set_title("Variance")
                plt.show()


if __name__ == "__main__":
    test_CleanUMamba_pruningV2()