import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class CWConv_HebbSuper(nn.Module):
    """
    CwC layer with online learned class->supergroup assignments.

    Keeps the same external API as CWConv:
      - forward(x, ...) -> (y, g) where g is [B, num_class] "adjusted" goodness (fine-shaped)
      - learn(x, gt, show) -> (y_detached, g_detached)

    Internals:
      gf_fine: [B, J]
      A:       [K, J]   (soft assignment columns sum to 1)
      z:       [B, K]   (coarse logits)
      g_adj:   [B, J]   (fine logits adjusted by supergroup structure)

    Loss:
      L = (1-beta_coarse)*CE(g_adj, gt) + beta_coarse*SoftCE(z, A[:,gt])
    Gate update (slow, no-grad):
      G <- (1-lam)*G + eta*(E[p(k|x)*r] - balance_strength*(usage - 1/K))
      A <- softmax(G / tau, dim=0)
    """

    def __init__(
        self,
        in_dims,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        bias=False,
        maxpool=False,
        num_class=10,
        droprate=0,
        first=False,
        layer_n=0,
        # --- supergroup params ---
        num_supergroups=None,          # K; if None -> heuristic K = max(2, num_class//5)
        beta_coarse=0.7,               # mix between coarse objective and fine adjusted CE
        hard_forward=False,            # use hard A in forward (still keep soft A for updates)
        # --- gate update params ---
        gate_lr=0.05,                  # eta_g
        gate_decay=0.01,               # lambda_g (EMA decay on gates)
        gate_update_every=10,          # update A every N learn() calls
        gate_warmup_steps=10,           # delay updates if you want
        balance_strength=0.5,          # prevent collapse (group usage regularizer)
        tau0=1.0,                      # initial assignment temperature
        tau_min=0.3,
        tau_decay=0.999,               # per gate-update (not per step)
        init_gate_noise=0.01,          # break symmetry
    ):
        super().__init__()

        # assert out_channels % num_class == 0 or out_channels % num_supergroups == , "out_channels must be divisible by num_class (fine blocks)"
        self.outc = out_channels
        self.num_class = num_class
        self.layer_n = layer_n
        self.FIRST = first
        self.maxpool = maxpool

        # --- conv trunk (same as CWConv) ---
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-4)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(droprate)

        self.dims = in_dims
        self.kernel_size = kernel_size
        self.outsize = int(((self.dims[1] - self.kernel_size + (2 * padding)) / stride) + 1)

        torch.nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        self.lr = 0.01
        self.opt = Adam(self.parameters(), lr=self.lr)
        self.scheduler = None

        self.criterion = nn.CrossEntropyLoss()
        self.last_pred = 1.0
        self.last_tr_pred = 1.0
        self.ep_losses = []

        self.N_neurons_out = self.outc * self.outsize ** 2
        self.next_dims = [self.outc, self.outsize, self.outsize]
        print(self.dims, self.N_neurons_out)

        # --- supergroup setup ---
        J = num_class
        if num_supergroups is None:
            K = max(2, min(J, J // 5))  # CIFAR10-10 -> 2, CIFAR10-100 -> 20
        else:
            K = int(num_supergroups)
        K = max(2, min(K, J))
        self.K = K
        self.J = J

        self.beta_coarse = float(beta_coarse)
        self.hard_forward = bool(hard_forward)

        # Gate matrix G (K x J), updated manually (NOT a parameter)
        G0 = init_gate_noise * torch.randn(K, J)
        self.register_buffer("G_gate", G0)

        # current A (K x J), stored as buffer for inspection/saving
        A0 = F.softmax(self.G_gate / tau0, dim=0)  # columns sum to 1
        self.register_buffer("A_assign", A0)

        # gate update hyperparams / counters
        self.gate_lr = float(gate_lr)
        self.gate_decay = float(gate_decay)
        self.gate_update_every = int(gate_update_every)
        self.gate_warmup_steps = int(gate_warmup_steps)
        self.balance_strength = float(balance_strength)

        self.tau0 = float(tau0)
        self.tau_min = float(tau_min)
        self.tau_decay = float(tau_decay)

        self._learn_steps = 0
        self._gate_updates = 0

        # caches (for debugging)
        self.gf_fine = None
        self.g_coarse = None
        self.g_adj = None

    # ---------------- core ops ----------------

    def goodness_factor(self, y):
        # same as your CWConv: class-block energy
        B, C, H, W = y.shape
        S = C // self.num_class
        gf = y.abs().reshape(B, self.num_class, S * H * W).square().mean(-1)
        return gf

    def _current_tau(self):
        tau = self.tau0 * (self.tau_decay ** self._gate_updates)
        return max(self.tau_min, float(tau))

    def _get_A(self):
        # soft A always exists; hard version is optional for forward use
        A_soft = self.A_assign
        if not self.hard_forward:
            return A_soft
        # hard per class (one-hot column)
        idx = torch.argmax(A_soft, dim=0)              # [J]
        A_hard = torch.zeros_like(A_soft)             # [K,J]
        A_hard[idx, torch.arange(self.J, device=idx.device)] = 1.0
        return A_hard

    def _soft_ce_coarse(self, z, gt, A_soft):
        """
        z:     [B,K]
        gt:    [B] fine class labels in [0..J-1]
        A_soft:[K,J]
        Loss = - sum_k A[k,gt] log softmax(z)[k]
        """
        logp = F.log_softmax(z, dim=1)                # [B,K]
        target = A_soft[:, gt].T                      # [B,K]
        loss = -(target * logp).sum(dim=1).mean()
        return loss

    def _maybe_update_gates(self, z_det, gt_det, A_soft_det):
        """
        HebbGate-style slow update:
          - compute p(k|x) from z
          - compute per-sample soft CE vs target A[:,gt]
          - r = exp(-loss_i) in (0,1]
          - accumulate mean(p*r) per class
          - apply balance penalty based on group usage
        """
        self._learn_steps += 1
        if self._learn_steps < self.gate_warmup_steps:
            return
        if (self._learn_steps % self.gate_update_every) != 0:
            return

        with torch.no_grad():
            B = z_det.shape[0]
            K, J = self.K, self.J

            p = F.softmax(z_det, dim=1)               # [B,K]
            logp = torch.log(p.clamp_min(1e-12))      # stable

            target = A_soft_det[:, gt_det].T          # [B,K]
            loss_i = -(target * logp).sum(dim=1)      # [B]
            r = torch.exp(-loss_i).clamp(0.0, 1.0)    # [B]

            # Vectorized per-class accumulation:
            onehot = F.one_hot(gt_det, num_classes=J).float()  # [B,J]
            pr = p * r.unsqueeze(1)                            # [B,K]
            sum_pr = pr.T @ onehot                             # [K,J]
            counts = onehot.sum(dim=0).clamp_min(1.0)          # [J]
            mean_pr = sum_pr / counts.unsqueeze(0)             # [K,J]

            # group usage (average assignment mass)
            usage = A_soft_det.mean(dim=1, keepdim=True)       # [K,1]
            imbalance = usage - (1.0 / K)                      # [K,1]

            # Update only for classes present in this batch:
            present = (onehot.sum(dim=0) > 0)                  # [J] bool
            if present.any():
                # EMA decay on gates + Hebbian increment
                # (centered by balance penalty to avoid collapse)
                delta = mean_pr - self.balance_strength * imbalance  # [K,J]
                self.G_gate[:, present] = (1.0 - self.gate_decay) * self.G_gate[:, present] + self.gate_lr * delta[:, present]

            # Refresh A using current temperature
            tau = self._current_tau()
            self.A_assign = F.softmax(self.G_gate / tau, dim=0)
            self._gate_updates += 1

    # ---------------- public API ----------------

    def forward(self, x, no_norm=False, eval=True, compute_gf=True):
        x = x.detach()

        y = self.conv(x)
        if self.maxpool:
            y = F.max_pool2d(y, 2, 2)
        y = self.bn(y)
        y = self.relu(y)
        if not eval:
            y = self.dropout(y)

        g = None
        if compute_gf:
            gf_fine = self.goodness_factor(y)         # [B,J]
            A_use = self._get_A()                     # [K,J] (soft or hard for forward)
            z = gf_fine @ A_use.T                     # [B,K]
            g_adj = z @ A_use                         # [B,J]

            # cache
            self.gf_fine = gf_fine
            self.g_coarse = z
            self.g_adj = g_adj

            # return fine-shaped scores so your predict() and layer_predict() keep working
            g = g_adj

        if no_norm:
            return y, g
        y = F.group_norm(y, self.num_class)
        return y, g

    def learn(self, x, gt, show):
        # forward (training mode)
        y, g_adj = self.forward(x, eval=False, compute_gf=True)

        # losses
        A_soft = self.A_assign                          # always use soft A for training + gate update
        z = self.g_coarse                               # [B,K] from forward cache
        loss_coarse = self._soft_ce_coarse(z, gt, A_soft)

        # optional fine loss on adjusted scores (stabilizes fine discrimination)
        loss_fine = self.criterion(g_adj, gt)

        beta = self.beta_coarse
        loss = (1.0 - beta) * loss_fine + beta * loss_coarse

        self.loss = loss
        self.ep_losses.append(loss.detach().item())

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # slow gate update (no grad)
        self._maybe_update_gates(z.detach(), gt.detach(), A_soft.detach())

        return y.detach(), g_adj.detach()

    def epoch_loss(self):
        epl_mean = torch.tensor(self.ep_losses).mean().item() if len(self.ep_losses) > 0 else 0.0
        self.ep_losses = []
        return epl_mean

    def eval_pred(self, g, gt, eval=True):
        _, predicted = torch.max(g, dim=1)
        pred_err = 1.0 - predicted.eq(gt).float().mean().item()
        if eval:
            self.last_pred = pred_err
        else:
            self.last_tr_pred = pred_err
        return pred_err

    def get_assignments(self):
        """
        Returns:
          A_soft: [K,J]  (columns sum to 1)
          hard_idx: [J]  (argmax group per class)
        """
        A = self.A_assign.detach()
        hard_idx = torch.argmax(A, dim=0)
        return A, hard_idx


class CWC_SG(nn.Module):
    """
    Coarse-only Hierarchical CwC layer with HebbGate-style assignment learning (SOFT only).

    Separation:
      - J = num_class        (fine dataset classes)
      - K = num_supergroups  (supergroups / channel-block count)
    Requirement:
      - out_channels % K == 0

    Forward:
      z     = goodness_factor_K(y)     in R^{B x K}
      g_adj = z @ A                   in R^{B x J}   (class-shaped scores for compatibility)

    Training (coarse-only):
      L = SoftCE(z, A[:, gt])

    Assignment update:
      - Update A until f = freeze_A_after_updates (counted in gate updates), then freeze.
      - No hardening during training.
    """

    def __init__(
        self,
        in_dims,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        bias=False,
        maxpool=False,
        pool_mode="none",
        num_class=10,                  # J (fine classes)
        droprate=0,
        first=False,
        layer_n=0,
        # --- supergroup params ---
        num_supergroups=None,          # K
        reward_mode='none',
        # --- gate update params ---
        gate_lr=0.05,
        gate_decay=0.01,
        gate_update_every=10,
        gate_warmup_steps=10,
        balance_strength=0.5,
        beta_coarse=1.0,
        tau0=1.0,
        tau_min=0.3,
        tau_decay=0.999,
        init_gate_noise=0.01,
        # --- freeze control (f) ---
        freeze_A_after_updates=80,     # stop updating A after this many gate updates
    ):
        super().__init__()

        self.reward_mode = reward_mode
        self.outc = out_channels
        self.num_class = int(num_class)   # J
        self.layer_n = layer_n
        self.FIRST = first
        self.pool_mode = "maxpool" if maxpool and pool_mode == "none" else pool_mode
        self.maxpool = self.pool_mode == "maxpool"

        # --- conv trunk ---
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-4)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(droprate)

        self.dims = in_dims
        self.kernel_size = kernel_size
        self.outsize = int(((self.dims[1] - self.kernel_size + (2 * padding)) / stride) + 1)
        if self.pool_mode in {"maxpool", "avgpool"}:
            self.outsize = int((self.outsize - 2) / 2 + 1)

        torch.nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        self.ce_loss = nn.CrossEntropyLoss()

        self.lr = 0.01
        self.opt = Adam(self.parameters(), lr=self.lr)
        self.scheduler = None

        self.beta_coarse = float(beta_coarse)

        self.last_pred = 1.0
        self.last_tr_pred = 1.0
        self.ep_losses = []

        self.N_neurons_out = self.outc * self.outsize ** 2
        self.next_dims = [self.outc, self.outsize, self.outsize]
        print(self.dims, self.N_neurons_out)

        # --- supergroup setup ---
        J = self.num_class
        if num_supergroups is None:
            K = max(2, min(J, J // 5))   # CIFAR-10 -> 2, CIFAR-100 -> 20
        else:
            K = int(num_supergroups)
        K = max(2, min(K, J))
        self.K = K
        self.J = J

        assert out_channels % self.K == 0, f"out_channels={out_channels} must be divisible by K={self.K}"

        # gate / assignment buffers
        G0 = init_gate_noise * torch.randn(self.K, self.J)
        self.register_buffer("G_gate", G0)

        A0 = F.softmax(self.G_gate / tau0, dim=0)  # softmax over K for each class column
        self.register_buffer("A_assign", A0)

        # gate hyperparams / counters
        self.gate_lr = float(gate_lr)
        self.gate_decay = float(gate_decay)
        self.gate_update_every = int(gate_update_every)
        self.gate_warmup_steps = int(gate_warmup_steps)
        self.balance_strength = float(balance_strength)

        self.tau0 = float(tau0)
        self.tau_min = float(tau_min)
        self.tau_decay = float(tau_decay)

        self._learn_steps = 0
        self._gate_updates = 0

        # freeze controls (f)
        self.freeze_A_after_updates = int(freeze_A_after_updates) if freeze_A_after_updates is not None else None
        self.A_frozen = False

        # caches
        self.g_coarse = None   # [B,K]
        self.g_adj = None      # [B,J]

    # ---------------- core ops ----------------

    def goodness_factor(self, y):
        """
        Compute goodness over K blocks (supergroups), not J classes.
        """
        B, C, H, W = y.shape
        S = C // self.K
        gf = y.abs().reshape(B, self.K, S * H * W).square().mean(-1)  # [B,K]
        return gf

    def _current_tau(self):
        tau = self.tau0 * (self.tau_decay ** self._gate_updates)
        return max(self.tau_min, float(tau))

    def _soft_ce_coarse(self, z, gt, A_soft):
        """
        z:     [B,K]
        gt:    [B] in [0..J-1]
        A_soft:[K,J]
        """
        logp = F.log_softmax(z, dim=1)          # [B,K]
        target = A_soft[:, gt].T                # [B,K]
        return -(target * logp).sum(dim=1).mean()

    def _freeze_A(self):
        self.A_frozen = True

    def _maybe_update_gates(self, z_det, gt_det):
        """
        HebbGate-style update, but stops permanently after freeze_A_after_updates.
        """
        if self.A_frozen:
            return

        if self.freeze_A_after_updates is not None and self._gate_updates >= self.freeze_A_after_updates:
            self._freeze_A()
            return

        self._learn_steps += 1
        if self._learn_steps < self.gate_warmup_steps:
            return
        if (self._learn_steps % self.gate_update_every) != 0:
            return

        with torch.no_grad():
            K, J = self.K, self.J

            p = F.softmax(z_det, dim=1)  # [B,K]

            if self.reward_mode == "none":
                r = torch.ones(p.shape[0], device=p.device, dtype=p.dtype)

            elif self.reward_mode == "entropy":
                # r in [0,1], higher when confident
                Hp = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1)  # [B]
                r = 1.0 - Hp / math.log(self.K)

            elif self.reward_mode == "margin":
                top2 = torch.topk(p, k=2, dim=1).values  # [B,2]
                r = (top2[:, 0] - top2[:, 1]).clamp(0.0, 1.0)

            elif self.reward_mode == "softce":
                logp = torch.log(p.clamp_min(1e-12))
                target = self.A_assign[:, gt_det].T
                loss_i = -(target * logp).sum(dim=1)
                r = torch.exp(-loss_i).clamp(0.0, 1.0)

            else:
                raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

            # class-conditioned accumulation
            onehot = F.one_hot(gt_det, num_classes=J).float()  # [B,J]
            pr = p * r.unsqueeze(1)                              # [B,K]
            sum_pr = pr.T @ onehot                               # [K,J]
            counts = onehot.sum(dim=0).clamp_min(1.0)            # [J]
            mean_pr = sum_pr / counts.unsqueeze(0)               # [K,J]

            # balance penalty (anti-collapse)
            usage = self.A_assign.mean(dim=1, keepdim=True)      # [K,1]
            imbalance = usage - (1.0 / K)                        # [K,1]
            delta = mean_pr - self.balance_strength * imbalance  # [K,J]

            present = (onehot.sum(dim=0) > 0)                    # [J]
            if present.any():
                self.G_gate[:, present] = (1.0 - self.gate_decay) * self.G_gate[:, present] + self.gate_lr * delta[:, present]

            # refresh A
            tau = self._current_tau()
            self.A_assign.copy_(F.softmax(self.G_gate / tau, dim=0))
            self._gate_updates += 1

            # freeze check after increment
            if self.freeze_A_after_updates is not None and self._gate_updates >= self.freeze_A_after_updates:
                self._freeze_A()

    # ---------------- public API ----------------

    def forward(self, x, no_norm=False, eval=True, compute_gf=True):
        x = x.detach()

        y = self.conv(x)
        if self.pool_mode == "maxpool":
            y = F.max_pool2d(y, 2, 2)
        elif self.pool_mode == "avgpool":
            y = F.avg_pool2d(y, 2, 2)
        y = self.bn(y)
        y = self.relu(y)
        if not eval:
            y = self.dropout(y)

        g = None
        if compute_gf:
            z = self.goodness_factor(y)          # [B,K]
            g_adj = z @ self.A_assign            # [B,J] (soft mapping)

            self.g_coarse = z
            self.g_adj = g_adj
            g = g_adj

        if no_norm:
            return y, g

        # group_norm groups must divide channels -> use K
        y = F.group_norm(y, self.K)
        return y, g

    def learn(self, x, gt, show):
        y, g_adj = self.forward(x, eval=False, compute_gf=True)

        # coarse-only loss
        loss_coarse = self._soft_ce_coarse(self.g_coarse, gt, self.A_assign)
        loss_fine = self.ce_loss(g_adj, gt)

        beta = self.beta_coarse
        loss = (1.0 - beta) * loss_fine + beta * loss_coarse

        self.loss = loss
        self.ep_losses.append(loss.detach().item())

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # update A until freeze
        self._maybe_update_gates(self.g_coarse.detach(), gt.detach())

        return y.detach(), g_adj.detach()

    def epoch_loss(self):
        if len(self.ep_losses) == 0:
            return 0.0
        epl_mean = torch.tensor(self.ep_losses).mean().item()
        self.ep_losses = []
        return epl_mean

    def eval_pred(self, g, gt, eval=True):
        _, predicted = torch.max(g, dim=1)
        pred_err = 1.0 - predicted.eq(gt).float().mean().item()
        if eval:
            self.last_pred = pred_err
        else:
            self.last_tr_pred = pred_err
        return pred_err

    def get_assignments(self):
        """
        Returns:
          A_soft: [K,J]
          hard_idx: [J]  argmax supergroup per class (for reporting only)
        """
        A = self.A_assign.detach()
        hard_idx = torch.argmax(A, dim=0)
        return A, hard_idx


class CWC_Layer(nn.Module):
    def __init__(self, in_dims, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False, maxpool=False, pool_mode="none", num_class=10, droprate=0, first=False, layer_n=0):
        super(CWC_Layer, self).__init__()
        assert out_channels % num_class == 0
        self.outc = out_channels
        self.num_class = num_class
        self.layer_n = layer_n
        self.FIRST = first
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-4)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.relu = nn.LeakyReLU(inplace=True) #nn.ReLU(inplace=True)
        self.dims = in_dims

        self.kernel_size = kernel_size
        self.outsize = int(((self.dims[1] - self.kernel_size + (2 * padding)) / stride) + 1)

        self.pool_mode = "maxpool" if maxpool and pool_mode == "none" else pool_mode
        self.maxpool = self.pool_mode == "maxpool"
        if self.pool_mode in {"maxpool", "avgpool"}:
            self.outsize = int((self.outsize - 2) / 2 + 1)
        torch.nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.dropout = nn.Dropout(droprate)

        self.lr = 0.01

        self.opt = Adam(self.parameters(), lr=self.lr)  # , weight_decay=1e-4
        # self.opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        self.last_pred = 1.0
        self.last_tr_pred = 1.0

        self.ep_losses = []
        self.num_classes = num_class
        self.gf = None
        self.g_adj = None

        self.N_neurons_out = self.outc * self.outsize ** 2  # total number of output neurons
        self.next_dims = [self.outc, self.outsize, self.outsize]

        print(self.dims, self.N_neurons_out)


    def forward(self, x, no_norm=False, eval=True, compute_gf=True):
        x = x.detach()

        y = self.conv(x)
        if self.pool_mode == "maxpool":
            y = F.max_pool2d(y, 2, 2)
        elif self.pool_mode == "avgpool":
            y = F.avg_pool2d(y, 2, 2)
        y = self.bn(y)
        y = self.relu(y)
        if not eval:
            y = self.dropout(y)
        # classified score
        g = None
        if compute_gf:
            g = self.goodness_factor(y)
        self.g_adj = g

        if no_norm:
            return y, g
        y = F.group_norm(y, self.num_class)
        return y, g

    def epoch_loss(self):
        epl_mean = torch.tensor(self.ep_losses).mean().item()
        # if abs(epl_mean - self.ep_losses[-1]) < 0.00001:
        #     self.lr_decay()
        #     print('lr decay, new lr = ', self.lr)
        self.ep_losses = []
        # print(self.ep_losses)
        # print('ep losses', self.ep_losses)
        return epl_mean

    def goodness_factor(self, y):
        B, C, H, W = y.shape
        S = C // self.num_classes
        gf = y.abs().reshape(B, self.num_classes, S * H * W).square().mean(-1)

        return gf

    def learn(self, x, gt, show):

        y, self.gf = self.forward(x, eval=False)

        loss = self.criterion(self.gf, gt)
        self.loss = loss
        self.ep_losses.append(loss.detach().item())


        self.opt.zero_grad()
        # this backward just compute the derivative and hence
        # is not considered backpropagation.
        loss.backward()
        self.opt.step()

        return y.detach(), self.gf.detach()

    def setdropout(self, drop_rate):
        self.dropout = torch.nn.Dropout(p=drop_rate)
        self.droprate = drop_rate


    def forward_cp(self, x, no_norm=False):
        x = x.detach()

        y = self.conv(x)
        y = self.relu(y)
        y = self.dropout(y)

        # classified score
        g = y.view(y.size(0), self.num_class, -1)
        g = g.mean(dim=2)
        if no_norm:
            return y, g

        # normalize feature y
        y = F.group_norm(y, self.num_class)
        return y, g

    def forward_rcb(self, x, no_norm=False, eval=True, compute_gf=True):
        x = x.detach()

        if not self.FIRST:
            x = self.relu(x)
        y = self.conv(x)
        if self.pool_mode == "maxpool":
            y = F.max_pool2d(y, 2, 2)
        elif self.pool_mode == "avgpool":
            y = F.avg_pool2d(y, 2, 2)
        y = self.bn(y)
        # if self.FIRST:
        #     y = self.relu(y)
        if not eval:
            y = self.dropout(y)
        # classified score
        g = None
        if compute_gf:
            g = self.goodness_factor(y)
        self.g_adj = g

        if no_norm:
            return y, g
        y = F.group_norm(y, self.num_class)
        return y, g

    def eval_pred(self, g, gt, eval=True):
        _, predicted = torch.max(g, dim=1)
        pred_err = 1.0 - predicted.eq(gt).float().mean().item()
        if eval:
            self.last_pred = pred_err
        else:
            self.last_tr_pred = pred_err
        return pred_err
