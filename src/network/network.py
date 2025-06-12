import torch
from src.network.CleanUMamba import CleanUMamba


def Net(network, net_config):
    if network == "CleanUMamba":
        return CleanUMamba(**net_config)
    elif network == "OtherNetwork":
        raise NotImplementedError
    else:
        raise NotImplementedError



if __name__ == '__main__':
    import json
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/exp/models/DNS-CleanUMamba-3N-E8.json',
                        help='JSON file for configuration')
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    network_config = config["network_config"]

    model = CleanUMamba(**network_config).cuda()
    from src.util import print_size

    print_size(model, keyword="tsfm")

    input_data = torch.ones([4, 1, int(4.5 * 16000)]).cuda()
    output = model(input_data)
    print(output.shape)

    y = torch.rand([4, 1, int(4.5 * 16000)]).cuda()
    loss = torch.nn.MSELoss()(y, output)
    loss.backward()
    print(loss.item())
