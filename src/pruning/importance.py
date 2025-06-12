import torch


def calc_importance(importances, importance_metric):
    """Calculate the importance metric from the string. Supports * and / operations like a calculator."""
    if "+" in importance_metric:
        return sum([calc_importance(importances, im) for im in importance_metric.split("+")])
    if "-" in importance_metric:
        ims = importance_metric.split("-")
        result = calc_importance(importances, ims[0])
        for im in ims[1:]:
            result -= calc_importance(importances, im)
        return result
    if "/" in importance_metric:
        ims = importance_metric.split("/")
        result = calc_importance(importances, ims[0])
        for im in ims[1:]:
            result /= calc_importance(importances, im)
        return result
    if "**" in importance_metric:  # Exponentiation is more important than multiplication because ** and * look the same
        ims = importance_metric.split("**")
        assert len(ims) == 2, f"** should have 2 elements, got {len(ims)}" + str(ims)
        result = calc_importance(importances, ims[0])
        result **= calc_importance(importances, ims[1])
        return result
    if "*" in importance_metric:
        ims = importance_metric.split("*")
        result = calc_importance(importances, ims[0])
        for im in ims[1:]:
            result *= calc_importance(importances, im)
        return result

    try:
        value = float(importance_metric)
        return value
    except ValueError:
        return importances[importance_metric]

def get_prune_channels(prune_groups, importance_metric, n_prune_channels, perc_prune_channels_per_iter, min_channels_per_group, max_prune_importance_per_iter=None, calibrator_container=None, min_prune_channels=4):
    prunable = []
    prunable_params = 0

    if n_prune_channels is None:
        n_prune_channels = max(4, int(sum([group.n_channels for group in prune_groups]) * perc_prune_channels_per_iter))

    importance_min_dict = {}

    for group in prune_groups:

        importances = group.channel_importances()

        importances = calc_importance(importances, importance_metric)

        if calibrator_container:
            importances = calibrator_container.scale(importances, group)

        importance_min_dict[group.name] = min(importances)

        n_parameters = group.channel_importances()["n_parameters"]
        max_cutoff = min(n_prune_channels, group.n_channels - min_channels_per_group)
        if max_cutoff < 1:
            continue

        # Insert into prunable list sorted by importance, getting rid of the least important channels above parameters_per_iter
        insert_index = 0
        s = torch.sort(importances, descending=False)
        for j, (importance, index) in enumerate(zip(s.values, s.indices)):
            if j >= max_cutoff:
                break

            if len(prunable) < max_cutoff:
                prunable.append({
                    "group": group,
                    "index": index,
                    "importance": importance,
                    "n_parameters": n_parameters,
                })
                prunable_params += n_parameters
            else:
                for k in range(insert_index, len(prunable)):
                    if prunable[k]["importance"] > importance:
                        prunable.insert(k, {
                            "group": group,
                            "index": index,
                            "importance": importance,
                            "n_parameters": n_parameters,
                        })
                        prunable_params += n_parameters

                        break
                    insert_index += 1

    # remove last elements if we have too many parameters leaving some margin for the d_inner channels
    while len(prunable) > n_prune_channels + 8 * 3 and len(prunable) > min_prune_channels + 8 * 3:
        prunable_params -= prunable.pop()["n_parameters"]

    # remove elements if we exceed the max_prune_importance_per_iter
    if max_prune_importance_per_iter is not None:
        total_importance = sum([prune["importance"] for prune in prunable])
        print(f"total pruned importance inital: {total_importance:.2E}")
        while total_importance > max_prune_importance_per_iter and len(prunable) > min_prune_channels + 8 * 3:
            removed_chanel = prunable.pop()
            total_importance -= removed_chanel["importance"]
            prunable_params -= removed_chanel["n_parameters"]


    # Ensure that we have a multiple of 8 from the d_inner channels
    d_inner_counts = {}
    for i, prune in enumerate(prunable):
        if prune["group"].name.startswith("d_inner"):
            d_inner_group_name = prune["group"].name
            d_inner_counts[d_inner_group_name] = d_inner_counts.get(d_inner_group_name, 0) + 1
    for name, count in d_inner_counts.items():
        if count % 8 != 0:
            for i in reversed(range(len(prunable))):
                if prunable[i]["group"].name == name:
                    prunable_params -= prunable.pop(i)["n_parameters"]
                    d_inner_counts[name] -= 1
                    if d_inner_counts[name] % 8 == 0:
                        break

    # remove last elements if we have too many parameters excluding d_inner channels
    total_importance = sum([prune["importance"] for prune in prunable])
    skips = 0
    while (len(prunable) > n_prune_channels or (max_prune_importance_per_iter is not None and total_importance > max_prune_importance_per_iter)) and skips < len(prunable) - 1 and len(prunable) > min_prune_channels:
        if "d_inner" in prunable[-1 - skips]["group"].name:
            skips += 1
            continue
        removed_chanel = prunable.pop(-1 - skips)
        total_importance -= removed_chanel["importance"]
        prunable_params -= removed_chanel["n_parameters"]

    print(f"total pruned importance final: {total_importance:.2E}")

    return prunable, prunable_params, importance_min_dict
