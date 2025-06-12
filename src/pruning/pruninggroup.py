import typing
import torch
import torch.nn as nn

from src.pruning.util import prune_parameter_and_grad

class ParameterContainer:
    def __init__(self, name, module):
        self.name = name
        self.module = module
        self.shape = getattr(module, name).shape

    def __repr__(self):
        return f"{self.name} {self.module}"

    def set(self, value):
        setattr(self.module, self.name, value)
        self.shape = value.shape

    def get(self):
        return getattr(self.module, self.name)

    def register_forward_hook(self, hook):
        pass

    def register_forward_pre_hook(self, hook):
        pass

PC = ParameterContainer

class PruningModule:
    """Pruning module for a single layer in a model. keeps track of channel telemetry."""

    def __init__(self, module: typing.Union[nn.Module, ParameterContainer], out=True, dim=0, n_heads=1, channel_offset=0, next_module_to_offset=None, statistics=True):
        supported_modules = [ParameterContainer, nn.Conv1d, nn.ConvTranspose1d, torch.nn.Linear, nn.LayerNorm]
        assert any([isinstance(module, m) for m in supported_modules]), f"Unsupported module type {module.__class__}"

        self.module = module
        self.dim = dim
        self.n_heads = n_heads
        self.channel_offset = channel_offset

        assert next_module_to_offset is None or self.module is next_module_to_offset.module, "next module to offset is ment to move the offset of a diffrent module acting on the same matrix"
        self.next_module_to_offset = next_module_to_offset

        self.channel_input_telemetry = {
            "mean": None,
            "var": None,
            "min": None,
            "max": None,
            "count": 0,
        }
        self.channel_output_telemetry = {
            "mean": None,
            "var": None,
            "min": None,
            "max": None,
            "count": 0,
        }

        if statistics:
            module.register_forward_hook(self._forward_hook)
            module.register_forward_pre_hook(self._pre_hook)

        self.out = out
        self.group = None

    def channel_telemetry(self):
        return self.channel_input_telemetry if not self.out else self.channel_output_telemetry

    def check(self, channels):
        module_channels = self.module.weight.shape[self.dim] \
            if issubclass(self.module.__class__, nn.Module) \
            else self.module.shape[self.dim]

        assert module_channels % self.n_heads == 0, f"Module channels {module_channels} % {self.n_heads} != 0"
        assert module_channels / self.n_heads == channels, f"Module channels {module_channels} != {channels}"

    def _forward_hook(self, module, args, output):
        assert module == self.module, f"Module mismatch {module} != {self.module}"
        self.update_channel_telemetry(output, True)

    def _pre_hook(self, module, args):
        assert module == self.module, f"Module mismatch {module} != {self.module}"
        input = args[0]
        self.update_channel_telemetry(input, False)

    @torch.no_grad()
    def update_channel_telemetry(self, data, output=False):
        """Update channel telemetry with data samples."""
        # n = 1
        if output != self.out:
            return

        transpose_output_modules = [nn.Conv1d, nn.ConvTranspose1d]
        transpose_input_modules = [nn.Conv1d, nn.ConvTranspose1d]

        assert len(data.shape) == 2 or len(data.shape) == 3, f"Unsupported data shape {data.shape}"
        if len(data.shape) == 3:
            # if data.shape[1] // self.n_heads == self.group.n_channels:
            if ((any([isinstance(self.module, m) for m in transpose_output_modules]) and output)
                    or (any([isinstance(self.module, m) for m in transpose_input_modules]) and not output)):
                data = data.transpose(1, 2)
            if (data.shape[2] - self.channel_offset) // self.n_heads != self.group.n_channels:
                print(f"model instance {self.module}")
                print(f"data.shape {data.shape}")
                print(f"model weight.shape {self.module.weight.shape}")
                print(f"n_heads {self.n_heads}")
                print(f"n_channels {self.group.n_channels}")
                assert False, "Output dim does not match group channels"
            data = data.reshape(-1, data.shape[2] + self.channel_offset)
        else:  # (data.shape) == 2:
            if (data.shape[0] - self.channel_offset)  // self.n_heads == self.group.n_channels:
                data = data.transpose(0, 1)
            if (data.shape[1] - self.channel_offset)  // self.n_heads != self.group.n_channels:
                print(f"data.shape {data.shape}")
                print(f"n_heads {self.n_heads}")
                print(f"n_channels {self.group.n_channels}")
                assert False, "Output dim does not match group channels"

        data = data[..., self.channel_offset:]

        # first dimension is now the data samples and the second dimension is the channels (samples, channels)

        data_m = torch.mean(data, dim=0)
        data_v = torch.var(data, dim=0)
        data_min = torch.min(data, dim=0)[0]
        data_max = torch.max(data, dim=0)[0]

        if output:
            if(self.channel_output_telemetry["count"] == 0):
                self.channel_output_telemetry["mean"] = data_m
                self.channel_output_telemetry["var"] = data_v
                self.channel_output_telemetry["min"] = data_min
                self.channel_output_telemetry["max"] = data_max
                self.channel_output_telemetry["count"] += data.shape[0]
            else:
                self.channel_output_telemetry["mean"] = (self.channel_output_telemetry["mean"] * self.channel_output_telemetry["count"] + data_m * data.shape[0]) / (self.channel_output_telemetry["count"] + data.shape[0])
                self.channel_output_telemetry["var"] = (self.channel_output_telemetry["var"] * self.channel_output_telemetry["count"] + data_v * data.shape[0]) / (self.channel_output_telemetry["count"] + data.shape[0])
                self.channel_output_telemetry["min"] = torch.minimum(self.channel_output_telemetry["min"], data_min)
                self.channel_output_telemetry["max"] = torch.maximum(self.channel_output_telemetry["max"], data_max)
                self.channel_output_telemetry["count"] += data.shape[0]
        else:
            if(self.channel_input_telemetry["count"] == 0):
                self.channel_input_telemetry["mean"] = data_m
                self.channel_input_telemetry["var"] = data_v
                self.channel_input_telemetry["min"] = data_min
                self.channel_input_telemetry["max"] = data_max
                self.channel_input_telemetry["count"] += data.shape[0]
            else:
                self.channel_input_telemetry["mean"] = (self.channel_input_telemetry["mean"] * self.channel_input_telemetry["count"] + data_m * data.shape[0]) / (self.channel_input_telemetry["count"] + data.shape[0])
                self.channel_input_telemetry["var"] = (self.channel_input_telemetry["var"] * self.channel_input_telemetry["count"] + data_v * data.shape[0]) / (self.channel_input_telemetry["count"] + data.shape[0])
                self.channel_input_telemetry["min"] = torch.minimum(self.channel_input_telemetry["min"], data_min)
                self.channel_input_telemetry["max"] = torch.maximum(self.channel_input_telemetry["max"], data_max)
                self.channel_input_telemetry["count"] += data.shape[0]


        # print(f"Channel telemetry {self.channel_telemetry}")

    def channel_importances(self):
        importances = {
            "weight": None,
            "grad": None,
            "taylor_individual": None,
            "taylor_squared_individual": None,
            "taylor_group": None,
            "act_var": None,
            "n_parameters": None,
        }

        weight = self.module.weight.data if hasattr(self.module, 'weight') else self.module.get().data
        grad_available = self.module.weight.grad is not None if hasattr(self.module,
                                                                   'weight') else self.module.get().grad is not None
        if grad_available:
            grad = self.module.weight.grad.data if hasattr(self.module, 'weight') else self.module.get().grad.data
        if self.dim == 1:
            weight = weight.transpose(1, 0)
            if grad_available:
                grad = grad.transpose(1, 0)

        next_start = 0 if self.next_module_to_offset is None else weight.shape[0] - self.next_module_to_offset.channel_offset
        assert (weight.shape[0] - self.channel_offset - next_start) // self.n_heads == self.group.n_channels

        # the weight is now in shape (channels * N_heads), (channels * N_heads, parameters) or (channels * N_heads, 1, parameters)

        # Flatten if we have more than 2 dimensions (e.g. conv1d or linear)
        if len(weight.shape) > 2:
            weight = weight.flatten(1)
            if grad_available:
                grad = grad.flatten(1)
        elif len(weight.shape) == 1:
            weight = weight.unsqueeze(1)
            if grad_available:
                grad = grad.unsqueeze(1)

        # the weight is now in shape (offset + channels * N_heads, parameters)
        next_size = weight.shape[0] - self.channel_offset - self.group.n_channels * self.n_heads
        _, weight, _ = weight.split([self.channel_offset, self.group.n_channels * self.n_heads, next_size], dim=0)
        weight = weight.reshape(self.group.n_channels, -1)
        if grad_available:
            _, grad, _ = grad.split([self.channel_offset, self.group.n_channels * self.n_heads, next_size], dim=0)
            grad = grad.reshape(self.group.n_channels, -1)

        # the weight is now in shape (channels, parameters * N_heads)

        assert len(weight.shape) == 2, f"{weight.shape} was not in 2 dimensions"
        if grad_available:
            assert len(grad.shape) == 2, f"{grad.shape} was not in 2 dimensions"

        importances['weight'] = weight.abs().pow(2).sum(1)

        if grad_available:
            importances['grad'] = grad.abs().pow(2).sum(1)

            importances['taylor_individual'] = (weight * grad).abs().sum(1)
            importances['taylor_squared_individual'] = (weight * grad).pow(2).sum(1)
            importances['taylor_group'] = (weight * grad).sum(1).abs()

        importances['n_parameters'] = weight.shape[1]

        if self.channel_telemetry()["count"] > 0 and self.n_heads > 1:
            importances['act_var'] = self.channel_telemetry()["var"].reshape(self.group.n_channels, self.n_heads).mean(1)
        else:
            importances['act_var'] = self.channel_telemetry()["var"]

        return importances

    def change_offset(self, change):
        self.channel_offset += change

        # Propagate the change to next modules that operate on the same matrix
        if self.next_module_to_offset:
            self.next_module_to_offset.change_offset(change)

    def prune(self, idxs, head=False, optimizer=None):
        """Prune channels from the module."""

        if isinstance(idxs, torch.Tensor):
            idxs = idxs.cpu().numpy().tolist()

        # branch for pruning heads
        if self.n_heads > 1 and not head:
            for i in range(self.n_heads - 1, -1, -1):
                self.prune([i*self.group.n_channels + idx for idx in idxs], head=True, optimizer=optimizer)
            return


        # Prune indexes
        weight = getattr(self.module, 'weight') if hasattr(self.module, 'weight') else self.module.get()
        # weight = self.module.weight if hasattr(self.module, 'weight') else self.module.get()
        keep_idxs = list(set(range(weight.shape[self.dim])) - set([i + self.channel_offset for i in idxs]))
        keep_idxs.sort()
        assert len(keep_idxs) > 0, f"{len(keep_idxs)}"

        # Reset telemetry
        self.channel_output_telemetry["count"] = 0
        for key in self.channel_output_telemetry:
            if key != "count" and self.channel_output_telemetry[key] is not None and len(self.channel_output_telemetry[key].shape) > 0:
                self.channel_output_telemetry[key] = None
                # self.channel_output_telemetry[key] = self.channel_output_telemetry[key][keep_idxs]

        self.channel_input_telemetry["count"] = 0
        for key in self.channel_input_telemetry:
            if key != "count" and self.channel_input_telemetry[key] is not None and len(self.channel_input_telemetry[key].shape) > 0:
                self.channel_input_telemetry[key] = None
                # self.channel_input_telemetry[key] = self.channel_input_telemetry[key][keep_idxs]

        # Prune weights
        prune_parameter_and_grad(weight, keep_idxs, self.dim, optimizer)

        # Prune bias
        if (hasattr(self.module, 'bias') and self.module.bias is not None and
                (self.dim == (1 if isinstance(self.module, nn.ConvTranspose1d) else 0)) and self.module.bias.shape[0] > 1):
            keep_idxs = list(set(range(self.module.bias.shape[0])) - set([i + self.channel_offset for i in idxs]))
            keep_idxs.sort()
            prune_parameter_and_grad(self.module.bias, keep_idxs, 0, optimizer)

        # Update module variables
        if isinstance(self.module, nn.LayerNorm):
            self.module.normalized_shape = weight.shape
        if isinstance(self.module, nn.Conv1d):
            # self.module.groups = pruned_weights.shape[0]
            self.module.in_channels = weight.shape[1]
            self.module.out_channels = weight.shape[0]
            if self.module.groups > 1:
                self.module.groups = weight.shape[0]
        if isinstance(self.module, nn.ConvTranspose1d):
            self.module.in_channels = weight.shape[0]
            self.module.out_channels = weight.shape[1]
        if isinstance(self.module, nn.Linear):
            self.module.in_features = weight.shape[1]
            self.module.out_features = weight.shape[0]

        # If there is a next module acting on the same matrix update it's offset
        if self.next_module_to_offset is not None:
            self.next_module_to_offset.change_offset(- len(idxs) * self.n_heads)

    def __repr__(self):
        return f"{self.__class__.__name__} <{str(self)}>"

    def __str__(self):
        if self.channel_telemetry()["count"] == 0:
            return f"{self.module.__class__.__name__} (dim={self.dim}, n_heads={self.n_heads}, channel_offset={self.channel_offset}, telemetry=None)"

        telemetry = "{mean: " + str(torch.mean(self.channel_telemetry()["mean"]).item()) + ", var: " + str(torch.mean(self.channel_telemetry()["var"]).item()) + ", min: " + str(torch.min(self.channel_telemetry()["min"]).item()) + ", max: " + str(torch.max(self.channel_telemetry()["max"]).item()) + ", count: " + str(self.channel_telemetry()["count"]) + "}"
        return f"{self.module.__class__.__name__} (dim={self.dim}, n_heads={self.n_heads}, channel_offset={self.channel_offset}, out={self.out}, telemetry=" + telemetry

class PruningGroup:
    """Group of pruning modules that are pruned together."""

    def __init__(self, name, n_channels, main_module=None, data_len=0):
        self.name = name
        self.n_channels = n_channels
        self.data_len = data_len
        self.main_module = main_module
        self.modules = []

    def add_module(self, module):
        assert isinstance(module, PruningModule)
        self.modules.append(module)
        module.group = self

    def prune(self, idxs, optimizer=None):
        if isinstance(idxs, torch.Tensor):
            idxs = idxs.cpu().numpy().tolist()

        assert isinstance(idxs, list), f"idxs is not a list {idxs}"

        if len(idxs) == 0:
            return

        if isinstance(idxs[0], torch.Tensor):
            idxs = [idx.item() for idx in idxs]

        for module in self.modules:
            module.prune(idxs, optimizer=optimizer)

        self.n_channels -= len(idxs)

        if "d_state" in self.name:
            self.modules[1].module.module.d_state = self.n_channels

        if "dt_rank" in self.name:
            self.main_module.dt_rank = self.n_channels

        if "d_inner" in self.name:
            self.main_module.d_inner = self.n_channels
            self.main_module.expand = self.n_channels / self.main_module.d_model

        if "d_model" in self.name:
            for module in self.main_module:
                module.d_model = self.n_channels


    def check(self):
        for module in self.modules:
            # assert module.dim == 0, "Only dim=0 is supported for now"
            # assert module.n_heads == 1, "Only n_heads=1 is supported for now"
            module_channels = module.module.weight.shape[module.dim] \
                if issubclass(module.module.__class__, nn.Module) else module.module.shape[module.dim]

            next_start = 0 if module.next_module_to_offset is None else module_channels - module.next_module_to_offset.channel_offset
            assert (module_channels - module.channel_offset - next_start) // module.n_heads == self.n_channels, f"{module.module}: ({module_channels} - {module.channel_offset} - {next_start}) // {module.n_heads} == {self.n_channels}"

    def channel_importances(self):
        importances = {
            "weight": None,     # Weight importance per channel
            "grad": None,       # Grad magnitude per channel
            "taylor_individual": None,     # Taylor importance per channel
            "taylor_squared_individual": None,  # Taylor squared importance per channel
            "taylor_group": None,  # Taylor importance per channel
            "act_var": None,    # Activation variance per channel
            "n_parameters": 0,
            "n_filters": 0,
        }

        metrics = ["weight", "grad", "taylor_individual", "taylor_squared_individual", "taylor_group", "act_var"]
        counts = {m: 0 for m in metrics}

        for i, module in enumerate(self.modules):
            module_importances = module.channel_importances()
            for m in metrics:
                if module_importances[m] is not None:
                    assert len(module_importances[m]) == self.n_channels, f"{len(module_importances[m])} != {self.n_channels} for metric {m} in module {module}"

                    importances[m] = module_importances[m] if importances[m] is None else (
                            ((importances[m] * counts[m]) + module_importances[m]) / (counts[m] + 1))

                    counts[m] += 1

            importances["n_parameters"] += module_importances["n_parameters"]
            importances["n_filters"] += 1

        return importances



    def __repr__(self):
        return f"{self.__class__.__name__} <{str(self)}>"

    def __str__(self):
        return f"{self.name} (n_channels={self.n_channels}, modules={[str(m) for m in self.modules]})"


def CleanUMambaPrunableChannels(model, statistics=False):
    """
    Create a list of prunable groups for the CleanUMamba model, with the following groups:
    - Encoder down sample channels
    - Decoder channel mixing
    - Skip connection blocks
    - Mamba dmodel channels
    - Mamba d_inner channels
    # - Mamba d_state channels (not input yet)
    """

    prune_groups = []


    # Add Encoder and decoder groups
    for i in range(len(model.encoder)):
        # Encoder down sample channels
        channels = model.encoder[i][0].weight.shape[0]
        prune_group_element = PruningGroup(f"encode_down_{i}", n_channels=channels, data_len=80126//(2**i))
        prune_group_element.add_module(PruningModule(model.encoder[i][0], True, dim=0, statistics=statistics))
        prune_group_element.add_module(PruningModule(model.encoder[i][2], False, dim=1, statistics=statistics))
        prune_group_element.check()
        prune_groups.append(prune_group_element)

        # Decoder channel mixing
        decoder_i = len(model.decoder) - i - 1
        channels = model.decoder[decoder_i][0].weight.shape[0] // 2
        prune_group_element = PruningGroup(f"decode_mix_{i}", n_channels=channels, data_len=80126//(2**i))
        prune_group_element.add_module(PruningModule(model.decoder[decoder_i][0], True, n_heads=2, dim=0, statistics=statistics))
        prune_group_element.add_module(PruningModule(model.decoder[decoder_i][2], False, dim=0, statistics=statistics))
        prune_group_element.check()
        prune_groups.append(prune_group_element)

        # Skip connection blocks
        channels = model.encoder[i][2].weight.shape[0] // 2
        prune_group_element = PruningGroup(f"skip_conn_{i}", n_channels=channels, data_len=80126//(2**i))
        prune_group_element.add_module(PruningModule(model.encoder[i][2], True, n_heads=2, dim=0, statistics=statistics))
        prune_group_element.add_module(PruningModule(model.decoder[decoder_i][0], False, dim=1, statistics=statistics))
        if i + 1 == len(model.encoder):
            prune_group_element.add_module(PruningModule(model.tsfm_conv1, False, dim=1, statistics=statistics))
            prune_group_element.add_module(PruningModule(model.tsfm_conv2, True, dim=0, statistics=statistics))
        else:
            prune_group_element.add_module(PruningModule(model.encoder[i + 1][0], False, dim=1, statistics=statistics))
            prune_group_element.add_module(PruningModule(model.decoder[decoder_i - 1][2], True, dim=1, statistics=statistics))
        prune_group_element.check()
        prune_groups.append(prune_group_element)

    # Add Mamba dmodel channels
    channels = model.tsfm_conv1.weight.shape[0]
    prune_group_element = PruningGroup(f"d_model", n_channels=channels, data_len=624, main_module=model.tsfm_Mamba_layers)
    prune_group_element.add_module(PruningModule(model.tsfm_conv1, True, dim=0, statistics=statistics))
    prune_group_element.add_module(PruningModule(model.tsfm_conv2, False, dim=1, statistics=statistics))
    prune_group_element.add_module(PruningModule(model.norm_f, False, dim=0, statistics=statistics))
    for i, block in enumerate(model.tsfm_Mamba_layers):
        prune_group_element.add_module(PruningModule(block.norm, False, dim=0, statistics=statistics))
        prune_group_element.add_module(PruningModule(block.mixer.in_proj, False, dim=1, statistics=statistics))
        prune_group_element.add_module(PruningModule(block.mixer.out_proj, True, dim=0, statistics=statistics))
    prune_group_element.check()
    prune_groups.append(prune_group_element)

    # Add mamba d_inner and d_state channels
    for i, block in enumerate(model.tsfm_Mamba_layers):
        # d_inner channels
        channels = block.mixer.in_proj.weight.shape[0] // 2
        prune_group_element = PruningGroup(f"d_inner{i}", n_channels=channels, data_len=624, main_module=block.mixer)
        prune_group_element.add_module(PruningModule(block.mixer.in_proj, True, n_heads=2, dim=0, statistics=statistics))
        prune_group_element.add_module(PruningModule(block.mixer.out_proj, False, dim=1, statistics=statistics))
        prune_group_element.add_module(PruningModule(block.mixer.conv1d, False, dim=0, statistics=statistics))
        prune_group_element.add_module(PruningModule(block.mixer.x_proj, False, dim=1, statistics=statistics))
        prune_group_element.add_module(PruningModule(block.mixer.dt_proj, True, dim=0, statistics=statistics))
        prune_group_element.add_module(PruningModule(ParameterContainer("A_log", block.mixer), False, dim=0, statistics=statistics))
        prune_group_element.add_module(PruningModule(ParameterContainer("D", block.mixer), True, dim=0, statistics=statistics))
        prune_group_element.check()
        prune_groups.append(prune_group_element)

        # d_state channels
        dt_rank = block.mixer.dt_proj.weight.shape[1]
        channels = block.mixer.A_log.shape[1]
        prune_group_element = PruningGroup(f"d_state{i}", n_channels=channels, data_len=624)
        x_proj_module = PruningModule(block.mixer.x_proj, True, dim=0, n_heads=2,
                                                     channel_offset=dt_rank,
                                                     statistics=statistics)
        prune_group_element.add_module(x_proj_module)
        prune_group_element.add_module(
            PruningModule(ParameterContainer("A_log", block.mixer), False, dim=1, statistics=statistics))
        prune_group_element.check()
        prune_groups.append(prune_group_element)

        # dt_rank size
        prune_group_element = PruningGroup(f"dt_rank{i}", n_channels=dt_rank, main_module=block.mixer)
        prune_group_element.add_module(PruningModule(block.mixer.x_proj, True, dim=0, next_module_to_offset=x_proj_module, statistics=False))
        prune_group_element.add_module(PruningModule(block.mixer.dt_proj, True, dim=1, statistics=statistics))
        prune_group_element.check()
        prune_groups.append(prune_group_element)


    return prune_groups
