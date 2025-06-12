import torch
from src.network.CleanUMamba import CleanUMamba
from src.network.network import Net
import logging


def load_pretrained_CleanUMamba(path="checkpoints/models/CleanUMamba-3N-E8-high.pkl"):
    checkpoint = torch.load(path)
    config = checkpoint["network_config"]
    config["device"] = "cuda"
    model = Net("CleanUMamba", config)
    if hasattr(model, 'load_pruned_state_dict') and callable(model.load_pruned_state_dict):
        model.load_pruned_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.cuda()
    model.float()

    return model


if __name__ == "__main__":
    from src.util.denoise_eval import validate

    model = load_pretrained_CleanUMamba("checkpoints/pruned/CleanUMamba-3N-E8_pruned-5M.pkl")
    # model = load_pretrained_CleanUMamba("../../checkpoints/models/CleanUMamba-3N-E8-high.pkl")

    print("DNS")
    logging.warning("Replace the testset_path with your own DNS-Challenge dataset path")
    results = validate(model, testset_path=
        "/media/sjoerd/storage/boeken/jaar7/thesis/DNS-Challenge/datasets/test_set/synthetic/no_reverb")

    for key in results.keys():
        if key != 'Test/count':
            print(f"Model {key[5:]}={results[key] / results['Test/count']:.4f}")