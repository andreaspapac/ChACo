from hier_layers import *

class Hier_CwC_WAN(torch.nn.Module):

    def __init__(
        self,
        out_channels_list,
        batch_size,
        CFSE=False,
        sf_pred=False,
        dataset='MNIST',
        ILT='Acc',
        loss_='CwC',
        N_Classes=10,
        flow='RCB',
        alpha=2,
        n_epochs=200,
        skip_mode="cat",
        # ---- new: supergroup defaults ----
        num_supergroup_layers=1,
        use_supergroups=True,
        num_supergroups=None,         # if None -> rule based on N_Classes
        beta_start=1,
        beta_end=1,
        gate_warmup_steps=10,
        gate_update_every=10,
        gate_lr=0.03,
        gate_decay=0.005,
        balance_strength=0.35,
        tau0=1.0,
        tau_min=0.3,
        tau_decay=0.999,
        hard_forward=False,
        downsample_method="stride",
    ):
        super(Hier_CwC_WAN, self).__init__()

        self.iter = 1
        self.batch_size = batch_size
        self.show_iters = 800
        self.sf_pred = sf_pred
        self.nn_layers = []
        self.conv_layers = nn.ModuleList()
        self.num_supergroups_layers = num_supergroup_layers
        self.skip_mode = skip_mode

        self.gate_lr = gate_lr
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.flow = flow
        self.downsample_method = downsample_method

        # start_epochs = [0, 0, 0, 0, 0, 30, 50, 80]
        # start_epochs = [ 0,  1,  2,  8, 13, 19, 22, 27, 29, 32]
        if dataset == 'CIFARXX':
            start_epochs = [0, 2, 4, 6, 8, 10, 13]
            end_epochs = [25, 30, 40, 45, 50, 60, 75]
        else:
            start_epochs = [0, 1, 2, 3, 5, 8, 13]
            end_epochs = [25, 30, 40, 45, 50, 60, 75]

        # end_epochs =   [25, 26, 27, 34, 40, 47, 51, 57, 60, 70]

        # end_epochs = [40, 42, 44, 46, 50, 56, 66, 82, 100]

        self.start_end = [[start_epochs[i], end_epochs[i]] for i in range(len(out_channels_list))]
        if dataset == 'MNIST':
            CNN_l1_dims = [1, 28, 28]
            epoch_dur = 1
        elif dataset == 'FMNIST':
            CNN_l1_dims = [1, 28, 28]
            epoch_dur = 1
        elif dataset == 'STL10':
            CNN_l1_dims = [3, 96, 96]
            epoch_dur = 2
        elif dataset in ["TINYIMAGENET", "TINYIMAGENET200", "tiny-imagenet-200", "imgnet200"]:
            CNN_l1_dims = [3, 64, 64]
            epoch_dur = 3
        else:
            CNN_l1_dims = [3, 32, 32]
            if dataset == 'CIFAR10':
                epoch_dur = 2
            else:
                epoch_dur = 3
        self.start_end = [[start_epochs[i] * epoch_dur, end_epochs[i] * epoch_dur] for i in range(len(out_channels_list))]

        print(self.start_end)

        self.layer_out = []
        self.n_classes = N_Classes
        dims = [CNN_l1_dims]

        self.power = alpha
        self.final_channels = out_channels_list[-1]
        self.out_channels_ = out_channels_list
        self.maxpool = [False for _ in range(len(out_channels_list))]
        self.dropout_rates = []

        self.skip_from = []  # [1, 3, 5, 7]
        self.skip_to = []  # [3, 5, 7, 9]
        self.downsample = []  # [1, 5, 9]

        self.downsample = [2, 4, 6]
        print(self.skip_from)
        print(self.skip_to)
        print(self.downsample)

        # ---- supergroup K default rule (based on J=N_Classes) ----
        J = self.n_classes
        if num_supergroups is None:
            if J <= 20:
                K_default = 2
            else:
                K_default = 20 # min(20, max(4, J // 5))
        else:
            K_default = int(num_supergroups)

        K_default = max(2, min(K_default, J))
        print(K_default)
        self.num_supergroups = K_default
        self.use_supergroups = bool(use_supergroups)

        in_channels_list = []
        L = len(out_channels_list)

        K_perlayer = [20, 40, N_Classes, N_Classes, N_Classes, N_Classes]
        if dataset == 'CIFAR10':
            self.dropout_rates = [0,
            0,
            0,
            0.15,
            0.25,
            0.25,
            0.35,
            0.35
        ]
        else:
            self.dropout_rates = [0, 0, 0.1, 0.2, 0.3, 0.35, 0.35, 0.35]

        for i, out_channels in enumerate(out_channels_list):
            droprate = self.dropout_rates[i]  #0  # min(0.5, abs(out_channels - min(out_channels_list)) / (2 * max(out_channels_list))) if i > self.num_supergroups_layers else 0
            # self.dropout_rates.append(droprate)

            if out_channels >= N_Classes * 2:
                K_layer = min(int(out_channels/10), N_Classes)
            else:
                K_layer = self.num_supergroups

            if dataset == 'CIFAR10':
                K_layer = K_perlayer[i]

            in_channels = dims[-1][0]
            if i in self.skip_to and (self.skip_mode == "cat"):
                if i - 2 in self.skip_from:
                    in_channels = out_channels_list[i - 1] + in_channels_list[i - 2]
                else:
                    print(f'Skip connection invalid from Layer {i-2} to Layer {i}')
                    exit()
            in_channels_list.append(in_channels)

            use_downsample = i in self.downsample
            stride = 2 if (use_downsample and self.downsample_method == "stride") else 1
            pool_mode = self.downsample_method if use_downsample and self.downsample_method in {"avgpool", "maxpool"} else "none"
            # safer than your original (handles empty downsample)
            kernel_size = 3 #2 if (len(self.downsample) > 0 and i >= self.downsample[-1]) else 3

            # ---- layer-wise beta schedule ----
            if L > 1:
                beta_i = beta_start - (beta_start - beta_end) * (i / (L - 1))
            else:
                beta_i = beta_start

            if self.use_supergroups and i < self.num_supergroups_layers:

                print(f"CwC_SG Layer: {i} - Channels: {out_channels} - {K_layer} Supergroups - beta_start: {beta_start} - beta_end: {beta_end}")
                layer = CWC_SG(
                    dims[-1],
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1,
                    maxpool=False,
                    pool_mode=pool_mode,
                    droprate=droprate,
                    layer_n=i,
                    num_class=self.n_classes,          # IMPORTANT: keep consistent
                    num_supergroups=K_layer,
                    beta_coarse=beta_i,
                    gate_lr=gate_lr,
                    gate_decay=gate_decay,
                    gate_update_every=gate_update_every,
                    gate_warmup_steps=gate_warmup_steps,
                    balance_strength=balance_strength,
                    tau0=tau0,
                    tau_min=tau_min,
                    tau_decay=tau_decay,
                ).cuda()
            else:
                print(f"CwC Layer: {i} - Channels: {out_channels}")
                # fallback to your original layer if you ever want
                layer = CWC_Layer(
                    dims[-1],
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1,
                    maxpool=False,
                    pool_mode=pool_mode,
                    droprate=droprate,
                    layer_n=i,
                    num_class=self.n_classes,          # you should also pass this in original CWConv
                ).cuda()

            layer.apply(self.initialize_weights)
            layer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                layer.opt, T_max=n_epochs, eta_min=2e-4
            )

            self.conv_layers.append(layer)
            dims.append(layer.next_dims)

    def initialize_weights(self, m):
        initializer = 'He'
        nonlinearity = 'leaky_relu'
        if isinstance(m, nn.Conv2d):
            if initializer == 'He':
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity=nonlinearity)
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def match_spatial(self, skip, h):
        while skip.shape[2] > h.shape[2]:
            skip = self._downsample_tensor(skip)
        return skip

    def _downsample_tensor(self, x):
        if self.downsample_method == "avgpool":
            return F.avg_pool2d(x, 2, 2)
        if self.downsample_method == "maxpool":
            return F.max_pool2d(x, 2, 2)
        if self.downsample_method == "stride":
            return x[:, :, ::2, ::2]
        raise ValueError(f"Unknown downsample_method: {self.downsample_method}")

    def _match_channels_for_add(self, t, target_c):
        # deterministic channel match with no new params:
        # crop if too many channels, pad zeros if too few
        c = t.shape[1]
        if c == target_c:
            return t
        if c > target_c:
            return t[:, :target_c, :, :]
        pad = target_c - c
        return F.pad(t, (0, 0, 0, 0, 0, pad), mode="constant", value=0.0)

    def _apply_skip(self, h, skip_tensor):

        if skip_tensor.shape[2:] != h.shape[2:]:
            skip_tensor = self.match_spatial(skip_tensor, h)

        if self.skip_mode == "none":
            return h

        if self.skip_mode == "cat":
                # h = torch.cat((h, sk), dim=1)
            return torch.cat((h, skip_tensor), dim=1)

        if self.skip_mode == "add":
            skip_m = self._match_channels_for_add(skip_tensor, h.shape[1])
            if h.shape[1] != skip_tensor.shape[1]:
                skip_m = self._match_channels_for_add(skip_tensor, h.shape[1])
                # raise ValueError(f"skip_mode='add' requires same channels: h={h.shape}, skip={skip_tensor.shape}")
            return h + skip_m

        if self.skip_mode == "add_pad":
            skip_m = self._match_channels_for_add(skip_tensor, h.shape[1])
            return h + skip_m

        raise ValueError(f"Unknown skip_mode: {self.skip_mode}")

    def predict(self, x, gt, epoch):
        h = x
        layer_pred = []
        skip = {}

        for i, layer in enumerate(self.conv_layers):
            s, e = self.start_end[i]
            if epoch < s:
                layer_pred.append(1.1)
                continue

            if i in self.skip_to:
                h = self._apply_skip(h, skip[f"skip_{i-2}"])

            if i in self.skip_from:
                skip[f"skip_{i}"] = h

            # prediction requires gf
            h, g = layer.forward(h, eval=True, compute_gf=True)
            pred_err = layer.eval_pred(g, gt)
            layer_pred.append(pred_err)

        return layer_pred


class Hier_CwC_ResNet(torch.nn.Module):

    def __init__(
        self,
        out_channels_list,
        batch_size,
        CFSE=False,
        sf_pred=False,
        dataset='MNIST',
        ILT='Acc',
        loss_='CwC',
        N_Classes=10,
        flow='RCB',
        alpha=2,
        n_epochs=200,
        skip_mode="cat",
        # ---- new: supergroup defaults ----
        num_supergroup_layers=2,
        use_supergroups=True,
        num_supergroups=None,         # if None -> rule based on N_Classes
        beta_start=1,
        beta_end=1,
        gate_warmup_steps=10,
        gate_update_every=10,
        gate_lr=0.03,
        gate_decay=0.005,
        balance_strength=0.35,
        tau0=1.0,
        tau_min=0.3,
        tau_decay=0.999,
        hard_forward=False,
        downsample_method="stride",
    ):
        super(Hier_CwC_ResNet, self).__init__()

        self.iter = 1
        self.batch_size = batch_size
        self.show_iters = 800
        self.sf_pred = sf_pred
        self.nn_layers = []
        self.conv_layers = nn.ModuleList()
        self.num_supergroups_layers = num_supergroup_layers
        self.skip_mode = skip_mode

        self.gate_lr = gate_lr
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.flow = flow
        self.downsample_method = downsample_method

        # start_epochs = [0, 0, 0, 0, 0, 30, 50, 80]
        start_epochs = [ 0,  1,  2,  8, 13, 19, 22, 27, 29, 32]
        end_epochs =   [25, 26, 27, 34, 40, 47, 51, 57, 60, 70]
        # end_epochs = [40, 42, 44, 46, 50, 56, 66, 82, 100]

        self.start_end = [[start_epochs[i], end_epochs[i]] for i in range(len(out_channels_list))]
        if dataset == 'MNIST':
            CNN_l1_dims = [1, 28, 28]
        elif dataset == 'FMNIST':
            CNN_l1_dims = [1, 28, 28]
        elif dataset == 'STL10':
            CNN_l1_dims = [3, 96, 96]
        else:
            CNN_l1_dims = [3, 32, 32]
        self.start_end = [[start_epochs[i] * 2, end_epochs[i] * 2] for i in range(len(out_channels_list))]

        print(self.start_end)

        self.layer_out = []
        self.n_classes = N_Classes
        dims = [CNN_l1_dims]

        self.power = alpha
        self.final_channels = out_channels_list[-1]
        self.out_channels_ = out_channels_list
        self.maxpool = [False for _ in range(len(out_channels_list))]
        self.dropout_rates = []

        self.skip_from = [1, 3, 5, 7]
        self.skip_to = [3, 5, 7, 9]
        self.downsample = []  # [1, 5, 9]
        # for i in range(len(out_channels_list)):
        #     if ((i + 1) % 2) == 0:
        #         self.skip_from.append(i)
        #         self.skip_to.append(i + 2)
        #         if out_channels_list[i] != out_channels_list[i - 2]:
        #             self.downsample.append(i + 2)

        self.downsample = [3, 7]
        print(self.skip_from)
        print(self.skip_to)
        print(self.downsample)

        # ---- supergroup K default rule (based on J=N_Classes) ----
        J = self.n_classes
        if num_supergroups is None:
            if J <= 20:
                K_default = 2
            else:
                K_default = 20 # min(20, max(4, J // 5))
        else:
            K_default = int(num_supergroups)
        K_default = max(2, min(K_default, J))
        print(K_default)
        self.num_supergroups = K_default
        self.use_supergroups = bool(use_supergroups)

        in_channels_list = []
        L = len(out_channels_list)

        self.dropout_rates = [0, 0, 0, 0, 0, 0, 0.2, 0.2, 0.35, 0.35]
        for i, out_channels in enumerate(out_channels_list):
            droprate = self.dropout_rates[i]  #0  # min(0.5, abs(out_channels - min(out_channels_list)) / (2 * max(out_channels_list))) if i > self.num_supergroups_layers else 0
            # self.dropout_rates.append(droprate)

            in_channels = dims[-1][0]
            if i in self.skip_to and (self.skip_mode == "cat"):
                if i - 2 in self.skip_from:
                    in_channels = out_channels_list[i - 1] + in_channels_list[i - 2]
                else:
                    print(f'Skip connection invalid from Layer {i-2} to Layer {i}')
                    exit()
            in_channels_list.append(in_channels)

            use_downsample = i in self.downsample
            stride = 2 if (use_downsample and self.downsample_method == "stride") else 1
            pool_mode = self.downsample_method if use_downsample and self.downsample_method in {"avgpool", "maxpool"} else "none"
            # safer than your original (handles empty downsample)
            kernel_size = 3 #2 if (len(self.downsample) > 0 and i >= self.downsample[-1]) else 3

            # ---- layer-wise beta schedule ----
            if L > 1:
                beta_i = beta_start - (beta_start - beta_end) * (i / (L - 1))
            else:
                beta_i = beta_start

            if self.use_supergroups and i < self.num_supergroups_layers:
                layer = CWC_SG(
                    dims[-1],
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1,
                    maxpool=False,
                    pool_mode=pool_mode,
                    droprate=droprate,
                    layer_n=i,
                    num_class=self.n_classes,  # IMPORTANT: keep consistent
                    num_supergroups=self.num_supergroups,
                    beta_coarse=beta_i,
                    gate_lr=gate_lr,
                    gate_decay=gate_decay,
                    gate_update_every=gate_update_every,
                    gate_warmup_steps=gate_warmup_steps,
                    balance_strength=balance_strength,
                    tau0=tau0,
                    tau_min=tau_min,
                    tau_decay=tau_decay,
                ).cuda()
            else:
                # fallback to your original layer if you ever want
                layer = CWC_Layer(
                    dims[-1],
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1,
                    maxpool=False,
                    pool_mode=pool_mode,
                    droprate=droprate,
                    layer_n=i,
                    num_class=self.n_classes,          # you should also pass this in original CWConv
                ).cuda()

            layer.apply(self.initialize_weights)
            layer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                layer.opt, T_max=n_epochs, eta_min=2e-4
            )

            self.conv_layers.append(layer)
            dims.append(layer.next_dims)

    def initialize_weights(self, m):
        initializer = 'He'
        nonlinearity = 'leaky_relu'
        if isinstance(m, nn.Conv2d):
            if initializer == 'He':
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity=nonlinearity)
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def match_spatial(self, skip, h):
        while skip.shape[2] > h.shape[2]:
            skip = self._downsample_tensor(skip)
        return skip

    def _downsample_tensor(self, x):
        if self.downsample_method == "avgpool":
            return F.avg_pool2d(x, 2, 2)
        if self.downsample_method == "maxpool":
            return F.max_pool2d(x, 2, 2)
        if self.downsample_method == "stride":
            return x[:, :, ::2, ::2]
        raise ValueError(f"Unknown downsample_method: {self.downsample_method}")

    def _match_channels_for_add(self, t, target_c):
        # deterministic channel match with no new params:
        # crop if too many channels, pad zeros if too few
        c = t.shape[1]
        if c == target_c:
            return t
        if c > target_c:
            return t[:, :target_c, :, :]
        pad = target_c - c
        return F.pad(t, (0, 0, 0, 0, 0, pad), mode="constant", value=0.0)

    def _apply_skip(self, h, skip_tensor):

        if skip_tensor.shape[2:] != h.shape[2:]:
            skip_tensor = self.match_spatial(skip_tensor, h)

        if self.skip_mode == "none":
            return h

        if self.skip_mode == "cat":
                # h = torch.cat((h, sk), dim=1)
            return torch.cat((h, skip_tensor), dim=1)

        if self.skip_mode == "add":
            skip_m = self._match_channels_for_add(skip_tensor, h.shape[1])
            if h.shape[1] != skip_tensor.shape[1]:
                skip_m = self._match_channels_for_add(skip_tensor, h.shape[1])
                # raise ValueError(f"skip_mode='add' requires same channels: h={h.shape}, skip={skip_tensor.shape}")
            return h + skip_m

        if self.skip_mode == "add_pad":
            skip_m = self._match_channels_for_add(skip_tensor, h.shape[1])
            return h + skip_m

        raise ValueError(f"Unknown skip_mode: {self.skip_mode}")

    def predict(self, x, gt, epoch):
        h = x
        layer_pred = []
        skip = {}

        for i, layer in enumerate(self.conv_layers):
            s, e = self.start_end[i]
            if epoch < s:
                layer_pred.append(1.1)
                continue

            if i in self.skip_to:
                h = self._apply_skip(h, skip[f"skip_{i-2}"])

            if i in self.skip_from:
                skip[f"skip_{i}"] = h

            # prediction requires gf
            h, g = layer.forward(h, eval=True, compute_gf=True)
            pred_err = layer.eval_pred(g, gt)
            layer_pred.append(pred_err)

        return layer_pred


class Hier_CwC_ResNet17(torch.nn.Module):

    def __init__(
        self,
        out_channels_list,
        batch_size,
        CFSE=False,
        sf_pred=False,
        dataset='MNIST',
        ILT='Acc',
        loss_='CwC',
        N_Classes=10,
        flow='RCB',
        alpha=2,
        n_epochs=200,
        skip_mode="cat",
        num_supergroup_layers=2,
        use_supergroups=True,
        num_supergroups=None,
        beta_start=1,
        beta_end=1,
        gate_warmup_steps=10,
        gate_update_every=10,
        gate_lr=0.03,
        gate_decay=0.005,
        balance_strength=0.35,
        tau0=1.0,
        tau_min=0.3,
        tau_decay=0.999,
        hard_forward=False,
        downsample_method="avgpool",
    ):
        super(Hier_CwC_ResNet17, self).__init__()

        self.iter = 1
        self.batch_size = batch_size
        self.show_iters = 800
        self.sf_pred = sf_pred
        self.nn_layers = []
        self.conv_layers = nn.ModuleList()
        self.num_supergroups_layers = num_supergroup_layers
        self.skip_mode = skip_mode
        self.gate_lr = gate_lr
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.flow = flow
        self.downsample_method = downsample_method
        self.stage_channels = list(out_channels_list) if out_channels_list else [100, 200, 400, 800]
        if len(self.stage_channels) != 4:
            raise ValueError(
                f"Hier_CwC_ResNet17 expects 4 stage widths, got {len(self.stage_channels)}: {self.stage_channels}"
            )

        if dataset == 'MNIST':
            CNN_l1_dims = [1, 28, 28]
        elif dataset == 'FMNIST':
            CNN_l1_dims = [1, 28, 28]
        elif dataset == 'STL10':
            CNN_l1_dims = [3, 96, 96]
        elif dataset in ["TINYIMAGENET", "TINYIMAGENET200", "tiny-imagenet-200", "imgnet200"]:
            CNN_l1_dims = [3, 64, 64]
        else:
            CNN_l1_dims = [3, 32, 32]

        self.layer_out = []
        self.n_classes = N_Classes
        self.power = alpha
        self.out_channels_ = list(self.stage_channels)
        self.layer_channels_ = [self.stage_channels[0]]
        for stage_channels in self.stage_channels:
            self.layer_channels_.extend([stage_channels] * 4)
        self.final_channels = self.stage_channels[-1]
        self.maxpool = [False for _ in range(len(self.layer_channels_))]
        self.dropout_rates = [0, 0, 0, 0, 0, 0.05, 0.05, 0.05, 0.1, 0.15, 0.15, 0.2, 0.2, 0.25, 0.25, 0.3, 0.3]

        start_epochs = [0, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 17, 21, 22, 23, 24, 28]
        end_epochs =   [20, 22, 24, 26, 30, 32, 34, 36, 40, 44, 46, 48, 52, 54, 56, 58, 62]
        self.start_end = [[start_epochs[i] * 3, end_epochs[i] * 3] for i in range(len(self.layer_channels_))]

        J = self.n_classes
        if num_supergroups is None:
            if J <= 20:
                K_default = 2
            else:
                K_default = 20
        else:
            K_default = int(num_supergroups)
        self.num_supergroups = max(2, min(K_default, J))
        self.use_supergroups = bool(use_supergroups)

        self.block_specs = []
        for block_idx, stage_channels in enumerate(self.stage_channels):
            start_idx = 1 + 4 * block_idx
            self.block_specs.append(
                {
                    "name": f"block{block_idx + 1}",
                    "stride": 1 if block_idx == 0 else 2,
                    "target_out_channels": stage_channels,
                    "final_merge": "add",
                    "layer_indices": list(range(start_idx, start_idx + 4)),
                }
            )

        dims = [CNN_l1_dims]
        all_layer_specs = [{"out_channels": self.stage_channels[0], "stride": 1}]
        for block_idx, stage_channels in enumerate(self.stage_channels):
            for layer_pos in range(4):
                all_layer_specs.append(
                    {
                        "out_channels": stage_channels,
                        "stride": 2 if (block_idx > 0 and layer_pos == 0) else 1,
                    }
                )

        for i, spec in enumerate(all_layer_specs):
            in_channels = dims[-1][0]
            out_channels = spec["out_channels"]
            stride = spec["stride"]
            droprate = self.dropout_rates[i]

            if len(all_layer_specs) > 1:
                beta_i = beta_start - (beta_start - beta_end) * (i / (len(all_layer_specs) - 1))
            else:
                beta_i = beta_start

            if self.use_supergroups and i < self.num_supergroups_layers:
                layer = CWC_SG(
                    dims[-1],
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    maxpool=False,
                    droprate=droprate,
                    layer_n=i,
                    num_class=self.n_classes,
                    num_supergroups=self.num_supergroups,
                    beta_coarse=beta_i,
                    gate_lr=gate_lr,
                    gate_decay=gate_decay,
                    gate_update_every=gate_update_every,
                    gate_warmup_steps=gate_warmup_steps,
                    balance_strength=balance_strength,
                    tau0=tau0,
                    tau_min=tau_min,
                    tau_decay=tau_decay,
                ).cuda()
            else:
                layer = CWC_Layer(
                    dims[-1],
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    maxpool=False,
                    droprate=droprate,
                    layer_n=i,
                    num_class=self.n_classes,
                ).cuda()

            layer.apply(self.initialize_weights)
            layer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                layer.opt, T_max=n_epochs, eta_min=2e-4
            )
            self.conv_layers.append(layer)

            dims.append(layer.next_dims)

    def initialize_weights(self, m):
        initializer = 'He'
        nonlinearity = 'leaky_relu'
        if isinstance(m, nn.Conv2d):
            if initializer == 'He':
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity=nonlinearity)
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _match_channels_for_add(self, t, target_c):
        c = t.shape[1]
        if c == target_c:
            return t
        if c > target_c:
            return t[:, :target_c, :, :]
        pad = target_c - c
        return F.pad(t, (0, 0, 0, 0, 0, pad), mode="constant", value=0.0)

    def _downsample_shortcut(self, shortcut, target_hw):
        while shortcut.shape[2] > target_hw[0] or shortcut.shape[3] > target_hw[1]:
            if self.downsample_method == "avgpool":
                shortcut = F.avg_pool2d(shortcut, 2, 2)
            elif self.downsample_method == "maxpool":
                shortcut = F.max_pool2d(shortcut, 2, 2)
            elif self.downsample_method == "stride":
                shortcut = shortcut[:, :, ::2, ::2]
            else:
                raise ValueError(f"Unknown downsample_method: {self.downsample_method}")
        return shortcut

    def _train_or_forward_layer(self, layer, h, gt, epoch, idx, training):
        s, e = self.start_end[idx]
        if epoch < s:
            return h, 1.1, False

        if training and (s <= epoch < e):
            h_next, g = layer.learn(h, gt, False)
            err = layer.eval_pred(g, gt, eval=False)
        elif training:
            h_next, _ = layer.forward(h, eval=True, compute_gf=False)
            err = layer.last_tr_pred
        else:
            h_next, g = layer.forward(h, eval=True, compute_gf=True)
            err = layer.eval_pred(g, gt)

        return h_next, err, True

    def _process_block(self, h, gt, epoch, block, training):
        errs = []
        shortcut = h
        layer_indices = block["layer_indices"]

        h1, err1, started1 = self._train_or_forward_layer(self.conv_layers[layer_indices[0]], h, gt, epoch, layer_indices[0], training)
        errs.append(err1)
        if not started1:
            errs.extend([1.1, 1.1, 1.1])
            return h, errs

        h2, err2, started2 = self._train_or_forward_layer(self.conv_layers[layer_indices[1]], h1, gt, epoch, layer_indices[1], training)
        errs.append(err2)
        if not started2:
            errs.extend([1.1, 1.1])
            return h1, errs

        shortcut_ds = self._downsample_shortcut(shortcut, h2.shape[2:])
        z = h2 + self._match_channels_for_add(shortcut_ds, h2.shape[1])

        h3, err3, started3 = self._train_or_forward_layer(self.conv_layers[layer_indices[2]], z, gt, epoch, layer_indices[2], training)
        errs.append(err3)
        if not started3:
            errs.append(1.1)
            return z, errs

        h4, err4, started4 = self._train_or_forward_layer(self.conv_layers[layer_indices[3]], h3, gt, epoch, layer_indices[3], training)
        errs.append(err4)
        if not started4:
            return h3, errs

        if block["final_merge"] == "add":
            h_out = h4 + self._match_channels_for_add(z, h4.shape[1])
        elif block["final_merge"] == "cat":
            h_out = torch.cat((h4, z), dim=1)
        else:
            raise ValueError(f"Unknown final merge: {block['final_merge']}")

        return h_out, errs

    def train_batch(self, x, gt, epoch):
        layer_errs = []

        h, err0, _ = self._train_or_forward_layer(self.conv_layers[0], x, gt, epoch, 0, training=True)
        layer_errs.append(err0)

        for block in self.block_specs:
            h, block_errs = self._process_block(h, gt, epoch, block, training=True)
            layer_errs.extend(block_errs)

        return h, layer_errs

    def predict_batch(self, x, gt, epoch):
        layer_errs = []

        h, err0, started0 = self._train_or_forward_layer(self.conv_layers[0], x, gt, epoch, 0, training=False)
        layer_errs.append(err0 if started0 else 1.1)

        for block in self.block_specs:
            h, block_errs = self._process_block(h, gt, epoch, block, training=False)
            layer_errs.extend(block_errs)

        return layer_errs

    def predict(self, x, gt, epoch):
        return self.predict_batch(x, gt, epoch)
