import random
import torch
import torch.nn as nn
import numpy as np
import csv
import os
import json
import socket
import subprocess
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    batch_range = range(x.shape[0])
    x_[batch_range, y] = x.max()

    return x_


def overlay_y_on_x3d(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """

    B, C, H, W = x.shape
    unflatten = nn.Unflatten(1, torch.Size([C, H, W]))

    x_ = x.clone()
    x_ = x_.reshape(x_.size(0), -1)

    x_[:, :10] *= 0.0
    x_[range(x_.shape[0]), y] = x_.max()

    x_ = unflatten(x_)


    return x_


def overlay_y_on_x4d(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """

    B, C, H, W = x.shape

    unflatten = nn.Unflatten(1, torch.Size([H, W]))

    x_ = x.clone()
    for ch in range(C):
        channel = x_[:, ch]
        # print(channel.shape)
        channel = channel.reshape(channel.size(0), -1)
        channel[:, :10] *= 0.0
        channel[range(channel.shape[0]), y] = channel.max()
        channel = unflatten(channel)
        # print(channel.shape)
        x_[:, ch, :, :] = channel

    return x_


def channel_shuffle(y, J):
    # Split activations by Number_of_classes, J, sets`
    B, C, H, W = y.shape
    groups = int(C / J)
    y_sets = torch.split(y, groups, dim=1)

    setrange = list(range(0, len(y_sets)))
    # print(setrange)

    for i in range(len(y_sets)):
        rand = random.randint(0, len(setrange)-1)
        yset = y_sets[setrange[rand]]
        setrange.pop(rand)

        if i == 0:
            group_y = yset
        else:
            group_y = torch.cat((group_y, yset), 1)

        # print(group_y.shape)

    return group_y


def save_model(model, model_id, dataset, epoch):
    b1_statedict = []
    b2_statedict = []
    nn_statedict = []
    b1_opt_statedict = []
    b2_opt_statedict = []
    nn_opt_statedict = []

    for i, layer in enumerate(model.convb1_layers):
        b1_statedict.append(layer.state_dict())
        b1_opt_statedict.append(layer.opt.state_dict())

    for i, layer in enumerate(model.convb2_layers):
        b2_statedict.append(layer.state_dict())
        b2_opt_statedict.append(layer.opt.state_dict())

    for i, layer in enumerate(model.nn_layers):
        nn_statedict.append(layer.state_dict())
        nn_opt_statedict.append(layer.opt.state_dict())

    checkpoint = {'model': model,
                  'SF_state_dict': model.classifier_b1.state_dict(),
                  'SF_optimizer': model.classifier_b1.opt.state_dict(),
                  'b1_state_dict': b1_statedict,
                  'b1_opt_state_dict': b1_opt_statedict,
                  'b2_state_dict': b2_statedict,
                  'b2_opt_state_dict': b2_opt_statedict,
                  'nn_state_dict': nn_statedict,
                  'nn_opt_state_dict': nn_opt_statedict
                  }

    torch.save(checkpoint, './weights/' + dataset + '/' + model_id + str(epoch) + '.pth')


def save_state(model, model_id, dataset, epoch, output_path=None):
    statedict = []
    nn_statedict = []
    opt_statedict = []
    nn_opt_statedict = []

    for i, layer in enumerate(model.conv_layers):
        statedict.append(layer.state_dict())
        opt_statedict.append(layer.opt.state_dict())

    for i, layer in enumerate(model.nn_layers):
        nn_statedict.append(layer.state_dict())
        nn_opt_statedict.append(layer.opt.state_dict())

    try:
        checkpoint = {'model': model,
                  'SF_state_dict': model.classifier_b1.state_dict(),
                  'SF_optimizer': model.classifier_b1.opt.state_dict(),
                  'state_dict': statedict,
                  'opt_state_dict': opt_statedict,
                  'nn_state_dict': nn_statedict,
                  'nn_opt_state_dict': nn_opt_statedict
                  }
    except:
        try:
            # Iterate through the layers to save activations
            checkpoint = {'model': model,
                          'state_dict': statedict,
                          'opt_state_dict': opt_statedict,
                          'activations': []
                          }
            for l, layer in enumerate(model.conv_layers):
                layer_activations = []
                for a, act in enumerate(layer.acts):
                    layer_activations.append({
                        'positive_slope': act.positive_slope.item(),
                        'negative_slope': act.negative_slope.item()
                    })
                checkpoint['activations'].append(layer_activations)
        except:
            checkpoint = {'model': model,
                          'state_dict': statedict,
                          'opt_state_dict': opt_statedict,
                          'activations': []
                          }

    out_path = output_path or ('./weights/' + dataset + '/' + model_id + str(epoch) + '.pth')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(checkpoint, out_path)
    return out_path


def save_FFrep(model, model_id, dataset, epoch):

    nn_statedict = []
    nn_opt_statedict = []
    for i, layer in enumerate(model.layers):
        nn_statedict.append(layer.state_dict())
        nn_opt_statedict.append(layer.opt.state_dict())

    checkpoint = {'model': model,
                  'nn_state_dict': nn_statedict,
                  'nn_opt_state_dict': nn_opt_statedict
                  }

    torch.save(checkpoint, './weights/' + dataset + '/' + model_id + str(epoch) + '.pth')


def load_model(model, model_id, dataset, epoch, param=False):
    checkpoint = torch.load('./weights/' + dataset + '/' + model_id + '.pth')

    for i, layer in enumerate(model.conv_layers):
        layer.load_state_dict(checkpoint['state_dict'][i])
        layer.opt.load_state_dict(checkpoint['opt_state_dict'][i])

    for i, layer in enumerate(model.nn_layers):
        layer.load_state_dict(checkpoint['nn_state_dict'][i])
        layer.opt.load_state_dict(checkpoint['nn_opt_state_dict'][i])

    try:
        model.classifier_b1.load_state_dict(checkpoint['SF_state_dict'])
        model.classifier_b1.opt.load_state_dict(checkpoint['SF_optimizer'])
    except:
        print('No SF predictor found.')

    try:
        for l, layer_activations in enumerate(checkpoint['activations']):
            for a, act_params in enumerate(layer_activations):
                model.conv_layers[l].acts[a].positive_slope.data = torch.tensor(act_params['positive_slope'])
                model.conv_layers[l].acts[a].negative_slope.data = torch.tensor(act_params['negative_slope'])
    except:
        print('No trainable Activations found')

    for parameter in model.parameters():
        parameter.requires_grad = param

    return model


def load_modelX(model, model_id, dataset, epoch, param=False):
    checkpoint = torch.load('./weights/' + dataset + '/' + model_id)

    for i, layer in enumerate(model.conv_layers):
        layer.load_state_dict(checkpoint['state_dict'][i])
        layer.opt.load_state_dict(checkpoint['opt_state_dict'][i])

    for i, layer in enumerate(model.nn_layers):
        layer.load_state_dict(checkpoint['nn_state_dict'][i])
        layer.opt.load_state_dict(checkpoint['nn_opt_state_dict'][i])

    try:
        model.classifier_b1.load_state_dict(checkpoint['SF_state_dict'])
        model.classifier_b1.opt.load_state_dict(checkpoint['SF_optimizer'])
    except:
        print('No SF predictor found.')

    for parameter in model.parameters():
        parameter.requires_grad = param

    return model


def save_traininglog(loss_log,filename, layer_losses=True):

    # Specify the output CSV file path
    csv_file_path = './TrRes/' + filename + '_log.csv'
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    # Open the CSV file in write mode
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the column headers
        if not layer_losses:
            loss_log = np.transpose(loss_log)
        else:
            arr = np.asarray(loss_log)
            num_layers = arr.shape[1] if arr.ndim > 1 else 1
            writer.writerow([f'L{i+1}' for i in range(num_layers)])

        # Write the data rows
        for i in range(np.shape(loss_log)[0]):
            epoch_losses = np.asarray(loss_log)[i, :]
            writer.writerow(epoch_losses)

    return csv_file_path


def save_traininglog_to_path(loss_log, output_path, layer_losses=True):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        if not layer_losses:
            loss_log = np.transpose(loss_log)
        else:
            arr = np.asarray(loss_log)
            num_layers = arr.shape[1] if arr.ndim > 1 else 1
            writer.writerow([f'L{i+1}' for i in range(num_layers)])

        for i in range(np.shape(loss_log)[0]):
            epoch_losses = np.asarray(loss_log)[i, :]
            writer.writerow(epoch_losses)

    return output_path


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.initial_seed(),
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def make_serializable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [make_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): make_serializable(v) for k, v in value.items()}
    return str(value)


def _safe_git_value(args):
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def collect_layer_logging_details(layer, layer_idx):
    info = {
        "layer_index": layer_idx,
        "layer_type": layer.__class__.__name__,
        "optimizer": layer.opt.__class__.__name__ if hasattr(layer, "opt") else None,
        "learning_rate": layer.opt.param_groups[0]["lr"] if hasattr(layer, "opt") else None,
        "out_channels": getattr(layer, "outc", None),
        "num_classes": getattr(layer, "num_class", getattr(layer, "num_classes", None)),
        "next_dims": getattr(layer, "next_dims", None),
        "maxpool": getattr(layer, "maxpool", None),
    }

    for attr in [
        "K",
        "J",
        "beta_coarse",
        "gate_lr",
        "gate_decay",
        "gate_update_every",
        "gate_warmup_steps",
        "balance_strength",
        "tau0",
        "tau_min",
        "tau_decay",
        "freeze_A_after_updates",
        "reward_mode",
    ]:
        if hasattr(layer, attr):
            info[attr] = getattr(layer, attr)

    return make_serializable(info)


class ExperimentLogger:
    def __init__(
        self,
        log_root,
        stem,
        dataset,
        run_name,
        seed,
        args_dict,
        model=None,
        online_logger="tensorboard",
        command=None,
        extra_path_parts=None,
    ):
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_run_name = run_name.replace(os.sep, "_")
        self.run_name = safe_run_name
        self.run_id = f"{timestamp}_{safe_run_name}_seed{seed}"
        self.extra_path_parts = [str(part) for part in (extra_path_parts or []) if str(part)]
        self.run_dir = os.path.join(log_root, stem, *self.extra_path_parts, dataset, self.run_id)
        self.metrics_dir = os.path.join(self.run_dir, "metrics")
        self.checkpoints_dir = os.path.join(self.run_dir, "checkpoints")
        self.artifacts_dir = os.path.join(self.run_dir, "artifacts")
        self.tb_dir = os.path.join(self.run_dir, "tensorboard")

        for path in [self.run_dir, self.metrics_dir, self.checkpoints_dir, self.artifacts_dir]:
            os.makedirs(path, exist_ok=True)

        self.online_logger = online_logger
        self.writer = None
        if online_logger != "none" and SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=self.tb_dir)

        self.epoch_metrics_path = os.path.join(self.metrics_dir, "epoch_metrics.jsonl")
        self.step_metrics_path = os.path.join(self.metrics_dir, "step_metrics.jsonl")
        self.epoch_metrics_csv_path = os.path.join(self.metrics_dir, "epoch_metrics.csv")
        self.step_metrics_csv_path = os.path.join(self.metrics_dir, "step_metrics.csv")
        self.summary_path = os.path.join(self.run_dir, "run_summary.json")
        self.legacy_log_path = os.path.join(self.run_dir, f"{safe_run_name}_log.txt")
        self.config_json_path = os.path.join(self.run_dir, "config.json")
        self._epoch_records = []
        self._step_records = []

        self.args_dict = make_serializable(args_dict)
        self.run_summary = {
            "run_id": self.run_id,
            "run_name": safe_run_name,
            "stem": stem,
            "dataset": dataset,
            "seed": seed,
            "created_utc": timestamp,
            "command": command,
            "hostname": socket.gethostname(),
            "git_commit": _safe_git_value(["git", "rev-parse", "HEAD"]),
            "git_branch": _safe_git_value(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "paths": {
                "run_dir": self.run_dir,
                "metrics_dir": self.metrics_dir,
                "checkpoints_dir": self.checkpoints_dir,
                "artifacts_dir": self.artifacts_dir,
            },
            "extra_path_parts": self.extra_path_parts,
            "artifacts": {},
            "best_checkpoints": [],
        }

        model_info = self._collect_model_info(model) if model is not None else None
        payload = {
            "run": self.run_summary,
            "args": self.args_dict,
            "model": model_info,
        }
        with open(self.config_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self._write_legacy_log(payload)
        self._flush_summary()

    def _collect_model_info(self, model):
        if model is None:
            return None

        info = {
            "model_class": model.__class__.__name__,
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "out_channels": getattr(model, "out_channels_", None),
            "start_end": getattr(model, "start_end", None),
            "dropout_rates": getattr(model, "dropout_rates", None),
            "skip_mode": getattr(model, "skip_mode", None),
            "skip_from": getattr(model, "skip_from", None),
            "skip_to": getattr(model, "skip_to", None),
            "downsample": getattr(model, "downsample", None),
            "downsample_method": getattr(model, "downsample_method", None),
            "num_supergroups_layers": getattr(model, "num_supergroups_layers", None),
            "num_supergroups": getattr(model, "num_supergroups", None),
            "use_supergroups": getattr(model, "use_supergroups", None),
            "beta_start": getattr(model, "beta_start", None),
            "beta_end": getattr(model, "beta_end", None),
            "flow": getattr(model, "flow", None),
            "stage_channels": getattr(model, "stage_channels", None),
            "layer_channels": getattr(model, "layer_channels_", None),
            "block_specs": getattr(model, "block_specs", None),
            "layer_details": [
                collect_layer_logging_details(layer, idx)
                for idx, layer in enumerate(getattr(model, "conv_layers", []))
            ],
        }
        return make_serializable(info)

    def _write_legacy_log(self, payload):
        lines = []
        lines.append(f"Experiment conducted on: {self.run_summary['created_utc']}")
        lines.append("Configuration:")
        lines.append(f"  run_id: {self.run_summary['run_id']}")
        lines.append(f"  run_dir: {self.run_dir}")
        lines.append(f"  git_commit: {self.run_summary['git_commit']}")
        lines.append(f"  git_branch: {self.run_summary['git_branch']}")
        for k, v in self.args_dict.items():
            lines.append(f"  {k}: {v}")
        model_info = payload.get("model") or {}
        for key in [
            "out_channels",
            "start_end",
            "dropout_rates",
            "skip_mode",
            "skip_from",
            "skip_to",
            "downsample",
            "downsample_method",
            "num_supergroups_layers",
            "num_supergroups",
            "use_supergroups",
            "beta_start",
            "beta_end",
            "flow",
            "stage_channels",
            "layer_channels",
            "block_specs",
            "parameter_count",
            "trainable_parameter_count",
        ]:
            if key in model_info:
                lines.append(f"  {key}: {model_info[key]}")
        lines.append("-" * 50)
        lines.append("Layer details:")
        for layer_info in model_info.get("layer_details", []):
            lines.append(f"  - {layer_info}")

        with open(self.legacy_log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def log_step(self, epoch, step, split, metrics):
        record = {
            "epoch": epoch,
            "step": step,
            "split": split,
            "metrics": make_serializable(metrics),
        }
        self._step_records.append(record)
        with open(self.step_metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self._write_records_csv(self.step_metrics_csv_path, self._step_records)

        if self.writer is not None:
            global_step = record["epoch"] * 1_000_000 + record["step"]
            for key, value in record["metrics"].items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{split}/{key}", value, global_step)

    def log_epoch(self, epoch, metrics):
        record = {"epoch": epoch, **make_serializable(metrics)}
        self._epoch_records.append(record)
        with open(self.epoch_metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self._write_records_csv(self.epoch_metrics_csv_path, self._epoch_records)

        if self.writer is not None:
            for key, value in record.items():
                if key == "epoch":
                    continue
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"epoch/{key}", value, epoch)
                elif isinstance(value, list):
                    for idx, sub_value in enumerate(value):
                        if isinstance(sub_value, (int, float)):
                            self.writer.add_scalar(f"epoch/{key}/layer_{idx}", sub_value, epoch)

    def save_array(self, loss_log, name, layer_losses=True):
        path = os.path.join(self.metrics_dir, f"{self.run_name}_{name}.csv")
        save_traininglog_to_path(loss_log, path, layer_losses=layer_losses)
        self.run_summary["artifacts"][name] = path
        self._flush_summary()
        return path

    def save_checkpoint(self, model, dataset, tag, score=None, max_to_keep=2):
        path = os.path.join(self.checkpoints_dir, f"{tag}.pth")
        save_state(model, "", dataset, "", output_path=path)
        checkpoint_info = {"tag": tag, "path": path, "score": score}
        current = self.run_summary.get("best_checkpoints", [])
        current.append(checkpoint_info)
        current = sorted(
            current,
            key=lambda item: float("inf") if item.get("score") is None else float(item["score"])
        )
        dropped = current[max_to_keep:]
        current = current[:max_to_keep]

        for item in dropped:
            stale_path = item.get("path")
            if stale_path and os.path.exists(stale_path):
                os.remove(stale_path)

        self.run_summary["best_checkpoints"] = current
        self.run_summary["artifacts"]["best_checkpoints"] = [item["path"] for item in current]
        self._flush_summary()
        return path

    def register_artifact(self, key, path):
        self.run_summary["artifacts"][key] = path
        self._flush_summary()

    def update_summary(self, **kwargs):
        self.run_summary.update(make_serializable(kwargs))
        self._flush_summary()

    def _flush_summary(self):
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(self.run_summary, f, indent=2)

    def _flatten_record(self, record):
        flat = {}
        for key, value in record.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat[f"{key}.{sub_key}"] = json.dumps(sub_value) if isinstance(sub_value, (list, dict)) else sub_value
            elif isinstance(value, list):
                flat[key] = json.dumps(value)
            else:
                flat[key] = value
        return flat

    def _write_records_csv(self, output_path, records):
        rows = [self._flatten_record(record) for record in records]
        if not rows:
            return
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def close(self):
        self._flush_summary()
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


def make_serializable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [make_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): make_serializable(v) for k, v in value.items()}
    return str(value)


def _safe_git_value(args):
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def collect_layer_logging_details(layer, layer_idx):
    info = {
        "layer_index": layer_idx,
        "layer_type": layer.__class__.__name__,
        "optimizer": layer.opt.__class__.__name__ if hasattr(layer, "opt") else None,
        "learning_rate": layer.opt.param_groups[0]["lr"] if hasattr(layer, "opt") else None,
        "out_channels": getattr(layer, "outc", None),
        "num_classes": getattr(layer, "num_class", getattr(layer, "num_classes", None)),
        "next_dims": getattr(layer, "next_dims", None),
        "maxpool": getattr(layer, "maxpool", None),
    }

    for attr in [
        "K",
        "J",
        "beta_coarse",
        "gate_lr",
        "gate_decay",
        "gate_update_every",
        "gate_warmup_steps",
        "balance_strength",
        "tau0",
        "tau_min",
        "tau_decay",
        "freeze_A_after_updates",
        "reward_mode",
    ]:
        if hasattr(layer, attr):
            info[attr] = getattr(layer, attr)

    return make_serializable(info)


class ExperimentLogger:
    def __init__(
        self,
        log_root,
        stem,
        dataset,
        run_name,
        seed,
        args_dict,
        model=None,
        online_logger="tensorboard",
        command=None,
        extra_path_parts=None,
    ):
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_run_name = run_name.replace(os.sep, "_")
        self.run_name = safe_run_name
        self.run_id = f"{timestamp}_{safe_run_name}_seed{seed}"
        self.extra_path_parts = [str(part) for part in (extra_path_parts or []) if str(part)]
        self.run_dir = os.path.join(log_root, stem, *self.extra_path_parts, dataset, self.run_id)
        self.metrics_dir = os.path.join(self.run_dir, "metrics")
        self.checkpoints_dir = os.path.join(self.run_dir, "checkpoints")
        self.artifacts_dir = os.path.join(self.run_dir, "artifacts")
        self.tb_dir = os.path.join(self.run_dir, "tensorboard")

        for path in [self.run_dir, self.metrics_dir, self.checkpoints_dir, self.artifacts_dir]:
            os.makedirs(path, exist_ok=True)

        self.online_logger = online_logger
        self.writer = None
        if online_logger != "none" and SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=self.tb_dir)

        self.epoch_metrics_path = os.path.join(self.metrics_dir, "epoch_metrics.jsonl")
        self.step_metrics_path = os.path.join(self.metrics_dir, "step_metrics.jsonl")
        self.epoch_metrics_csv_path = os.path.join(self.metrics_dir, "epoch_metrics.csv")
        self.step_metrics_csv_path = os.path.join(self.metrics_dir, "step_metrics.csv")
        self.summary_path = os.path.join(self.run_dir, "run_summary.json")
        self.legacy_log_path = os.path.join(self.run_dir, f"{safe_run_name}_log.txt")
        self.config_json_path = os.path.join(self.run_dir, "config.json")
        self._epoch_records = []
        self._step_records = []

        self.args_dict = make_serializable(args_dict)
        self.run_summary = {
            "run_id": self.run_id,
            "run_name": safe_run_name,
            "stem": stem,
            "dataset": dataset,
            "seed": seed,
            "created_utc": timestamp,
            "command": command,
            "hostname": socket.gethostname(),
            "git_commit": _safe_git_value(["git", "rev-parse", "HEAD"]),
            "git_branch": _safe_git_value(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "paths": {
                "run_dir": self.run_dir,
                "metrics_dir": self.metrics_dir,
                "checkpoints_dir": self.checkpoints_dir,
                "artifacts_dir": self.artifacts_dir,
            },
            "extra_path_parts": self.extra_path_parts,
            "artifacts": {},
            "best_checkpoints": [],
        }

        model_info = self._collect_model_info(model) if model is not None else None
        payload = {
            "run": self.run_summary,
            "args": self.args_dict,
            "model": model_info,
        }
        with open(self.config_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self._write_legacy_log(payload)
        self._flush_summary()

    def _collect_model_info(self, model):
        if model is None:
            return None

        info = {
            "model_class": model.__class__.__name__,
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "out_channels": getattr(model, "out_channels_", None),
            "start_end": getattr(model, "start_end", None),
            "dropout_rates": getattr(model, "dropout_rates", None),
            "skip_mode": getattr(model, "skip_mode", None),
            "skip_from": getattr(model, "skip_from", None),
            "skip_to": getattr(model, "skip_to", None),
            "downsample": getattr(model, "downsample", None),
            "downsample_method": getattr(model, "downsample_method", None),
            "num_supergroups_layers": getattr(model, "num_supergroups_layers", None),
            "num_supergroups": getattr(model, "num_supergroups", None),
            "use_supergroups": getattr(model, "use_supergroups", None),
            "beta_start": getattr(model, "beta_start", None),
            "beta_end": getattr(model, "beta_end", None),
            "flow": getattr(model, "flow", None),
            "stage_channels": getattr(model, "stage_channels", None),
            "layer_channels": getattr(model, "layer_channels_", None),
            "block_specs": getattr(model, "block_specs", None),
            "layer_details": [
                collect_layer_logging_details(layer, idx)
                for idx, layer in enumerate(getattr(model, "conv_layers", []))
            ],
        }
        return make_serializable(info)

    def _write_legacy_log(self, payload):
        lines = []
        lines.append(f"Experiment conducted on: {self.run_summary['created_utc']}")
        lines.append("Configuration:")
        lines.append(f"  run_id: {self.run_summary['run_id']}")
        lines.append(f"  run_dir: {self.run_dir}")
        lines.append(f"  git_commit: {self.run_summary['git_commit']}")
        lines.append(f"  git_branch: {self.run_summary['git_branch']}")
        for k, v in self.args_dict.items():
            lines.append(f"  {k}: {v}")
        model_info = payload.get("model") or {}
        for key in [
            "out_channels",
            "start_end",
            "dropout_rates",
            "skip_mode",
            "skip_from",
            "skip_to",
            "downsample",
            "downsample_method",
            "num_supergroups_layers",
            "num_supergroups",
            "use_supergroups",
            "beta_start",
            "beta_end",
            "flow",
            "stage_channels",
            "layer_channels",
            "block_specs",
            "parameter_count",
            "trainable_parameter_count",
        ]:
            if key in model_info:
                lines.append(f"  {key}: {model_info[key]}")
        lines.append("-" * 50)
        lines.append("Layer details:")
        for layer_info in model_info.get("layer_details", []):
            lines.append(f"  - {layer_info}")

        with open(self.legacy_log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def log_step(self, epoch, step, split, metrics):
        record = {
            "epoch": epoch,
            "step": step,
            "split": split,
            "metrics": make_serializable(metrics),
        }
        self._step_records.append(record)
        with open(self.step_metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self._write_records_csv(self.step_metrics_csv_path, self._step_records)

        if self.writer is not None:
            global_step = record["epoch"] * 1_000_000 + record["step"]
            for key, value in record["metrics"].items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{split}/{key}", value, global_step)

    def log_epoch(self, epoch, metrics):
        record = {"epoch": epoch, **make_serializable(metrics)}
        self._epoch_records.append(record)
        with open(self.epoch_metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self._write_records_csv(self.epoch_metrics_csv_path, self._epoch_records)

        if self.writer is not None:
            for key, value in record.items():
                if key == "epoch":
                    continue
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"epoch/{key}", value, epoch)
                elif isinstance(value, list):
                    for idx, sub_value in enumerate(value):
                        if isinstance(sub_value, (int, float)):
                            self.writer.add_scalar(f"epoch/{key}/layer_{idx}", sub_value, epoch)

    def save_array(self, loss_log, name, layer_losses=True):
        path = os.path.join(self.metrics_dir, f"{self.run_name}_{name}.csv")
        save_traininglog_to_path(loss_log, path, layer_losses=layer_losses)
        self.run_summary["artifacts"][name] = path
        self._flush_summary()
        return path

    def save_checkpoint(self, model, dataset, tag, score=None, max_to_keep=2):
        path = os.path.join(self.checkpoints_dir, f"{tag}.pth")
        save_state(model, "", dataset, "", output_path=path)
        checkpoint_info = {"tag": tag, "path": path, "score": score}
        current = self.run_summary.get("best_checkpoints", [])
        current.append(checkpoint_info)
        current = sorted(
            current,
            key=lambda item: float("inf") if item.get("score") is None else float(item["score"])
        )
        dropped = current[max_to_keep:]
        current = current[:max_to_keep]

        for item in dropped:
            stale_path = item.get("path")
            if stale_path and os.path.exists(stale_path):
                os.remove(stale_path)

        self.run_summary["best_checkpoints"] = current
        self.run_summary["artifacts"]["best_checkpoints"] = [item["path"] for item in current]
        self._flush_summary()
        return path

    def register_artifact(self, key, path):
        self.run_summary["artifacts"][key] = path
        self._flush_summary()

    def update_summary(self, **kwargs):
        self.run_summary.update(make_serializable(kwargs))
        self._flush_summary()

    def _flush_summary(self):
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(self.run_summary, f, indent=2)

    def _flatten_record(self, record):
        flat = {}
        for key, value in record.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat[f"{key}.{sub_key}"] = json.dumps(sub_value) if isinstance(sub_value, (list, dict)) else sub_value
            elif isinstance(value, list):
                flat[key] = json.dumps(value)
            else:
                flat[key] = value
        return flat

    def _write_records_csv(self, output_path, records):
        rows = [self._flatten_record(record) for record in records]
        if not rows:
            return
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def close(self):
        self._flush_summary()
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

def make_serializable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [make_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): make_serializable(v) for k, v in value.items()}
    return str(value)


def _safe_git_value(args):
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def collect_layer_logging_details(layer, layer_idx):
    info = {
        "layer_index": layer_idx,
        "layer_type": layer.__class__.__name__,
        "optimizer": layer.opt.__class__.__name__ if hasattr(layer, "opt") else None,
        "learning_rate": layer.opt.param_groups[0]["lr"] if hasattr(layer, "opt") else None,
        "out_channels": getattr(layer, "outc", None),
        "num_classes": getattr(layer, "num_class", getattr(layer, "num_classes", None)),
        "next_dims": getattr(layer, "next_dims", None),
        "maxpool": getattr(layer, "maxpool", None),
    }

    for attr in [
        "K",
        "J",
        "beta_coarse",
        "gate_lr",
        "gate_decay",
        "gate_update_every",
        "gate_warmup_steps",
        "balance_strength",
        "tau0",
        "tau_min",
        "tau_decay",
        "freeze_A_after_updates",
        "reward_mode",
    ]:
        if hasattr(layer, attr):
            info[attr] = getattr(layer, attr)

    return make_serializable(info)


class ExperimentLogger:
    def __init__(
        self,
        log_root,
        stem,
        dataset,
        run_name,
        seed,
        args_dict,
        model=None,
        online_logger="tensorboard",
        command=None,
        extra_path_parts=None,
    ):
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_run_name = run_name.replace(os.sep, "_")
        self.run_name = safe_run_name
        self.run_id = f"{timestamp}_{safe_run_name}_seed{seed}"
        self.extra_path_parts = [str(part) for part in (extra_path_parts or []) if str(part)]
        self.run_dir = os.path.join(log_root, stem, *self.extra_path_parts, dataset, self.run_id)
        self.metrics_dir = os.path.join(self.run_dir, "metrics")
        self.checkpoints_dir = os.path.join(self.run_dir, "checkpoints")
        self.artifacts_dir = os.path.join(self.run_dir, "artifacts")
        self.tb_dir = os.path.join(self.run_dir, "tensorboard")

        for path in [self.run_dir, self.metrics_dir, self.checkpoints_dir, self.artifacts_dir]:
            os.makedirs(path, exist_ok=True)

        self.online_logger = online_logger
        self.writer = None
        if online_logger != "none" and SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=self.tb_dir)

        self.epoch_metrics_path = os.path.join(self.metrics_dir, "epoch_metrics.jsonl")
        self.step_metrics_path = os.path.join(self.metrics_dir, "step_metrics.jsonl")
        self.epoch_metrics_csv_path = os.path.join(self.metrics_dir, "epoch_metrics.csv")
        self.step_metrics_csv_path = os.path.join(self.metrics_dir, "step_metrics.csv")
        self.summary_path = os.path.join(self.run_dir, "run_summary.json")
        self.legacy_log_path = os.path.join(self.run_dir, f"{safe_run_name}_log.txt")
        self.config_json_path = os.path.join(self.run_dir, "config.json")
        self._epoch_records = []
        self._step_records = []

        self.args_dict = make_serializable(args_dict)
        self.run_summary = {
            "run_id": self.run_id,
            "run_name": safe_run_name,
            "stem": stem,
            "dataset": dataset,
            "seed": seed,
            "created_utc": timestamp,
            "command": command,
            "hostname": socket.gethostname(),
            "git_commit": _safe_git_value(["git", "rev-parse", "HEAD"]),
            "git_branch": _safe_git_value(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "paths": {
                "run_dir": self.run_dir,
                "metrics_dir": self.metrics_dir,
                "checkpoints_dir": self.checkpoints_dir,
                "artifacts_dir": self.artifacts_dir,
            },
            "extra_path_parts": self.extra_path_parts,
            "artifacts": {},
            "best_checkpoints": [],
        }

        model_info = self._collect_model_info(model) if model is not None else None
        payload = {
            "run": self.run_summary,
            "args": self.args_dict,
            "model": model_info,
        }
        with open(self.config_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self._write_legacy_log(payload)
        self._flush_summary()

    def _collect_model_info(self, model):
        if model is None:
            return None

        info = {
            "model_class": model.__class__.__name__,
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "out_channels": getattr(model, "out_channels_", None),
            "start_end": getattr(model, "start_end", None),
            "dropout_rates": getattr(model, "dropout_rates", None),
            "skip_mode": getattr(model, "skip_mode", None),
            "skip_from": getattr(model, "skip_from", None),
            "skip_to": getattr(model, "skip_to", None),
            "downsample": getattr(model, "downsample", None),
            "downsample_method": getattr(model, "downsample_method", None),
            "num_supergroups_layers": getattr(model, "num_supergroups_layers", None),
            "num_supergroups": getattr(model, "num_supergroups", None),
            "use_supergroups": getattr(model, "use_supergroups", None),
            "beta_start": getattr(model, "beta_start", None),
            "beta_end": getattr(model, "beta_end", None),
            "flow": getattr(model, "flow", None),
            "stage_channels": getattr(model, "stage_channels", None),
            "layer_channels": getattr(model, "layer_channels_", None),
            "block_specs": getattr(model, "block_specs", None),
            "layer_details": [
                collect_layer_logging_details(layer, idx)
                for idx, layer in enumerate(getattr(model, "conv_layers", []))
            ],
        }
        return make_serializable(info)

    def _write_legacy_log(self, payload):
        lines = []
        lines.append(f"Experiment conducted on: {self.run_summary['created_utc']}")
        lines.append("Configuration:")
        lines.append(f"  run_id: {self.run_summary['run_id']}")
        lines.append(f"  run_dir: {self.run_dir}")
        lines.append(f"  git_commit: {self.run_summary['git_commit']}")
        lines.append(f"  git_branch: {self.run_summary['git_branch']}")
        for k, v in self.args_dict.items():
            lines.append(f"  {k}: {v}")
        model_info = payload.get("model") or {}
        for key in [
            "out_channels",
            "start_end",
            "dropout_rates",
            "skip_mode",
            "skip_from",
            "skip_to",
            "downsample",
            "downsample_method",
            "num_supergroups_layers",
            "num_supergroups",
            "use_supergroups",
            "beta_start",
            "beta_end",
            "flow",
            "stage_channels",
            "layer_channels",
            "block_specs",
            "parameter_count",
            "trainable_parameter_count",
        ]:
            if key in model_info:
                lines.append(f"  {key}: {model_info[key]}")
        lines.append("-" * 50)
        lines.append("Layer details:")
        for layer_info in model_info.get("layer_details", []):
            lines.append(f"  - {layer_info}")

        with open(self.legacy_log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def log_step(self, epoch, step, split, metrics):
        record = {
            "epoch": epoch,
            "step": step,
            "split": split,
            "metrics": make_serializable(metrics),
        }
        self._step_records.append(record)
        with open(self.step_metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self._write_records_csv(self.step_metrics_csv_path, self._step_records)

        if self.writer is not None:
            global_step = record["epoch"] * 1_000_000 + record["step"]
            for key, value in record["metrics"].items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{split}/{key}", value, global_step)

    def log_epoch(self, epoch, metrics):
        record = {"epoch": epoch, **make_serializable(metrics)}
        self._epoch_records.append(record)
        with open(self.epoch_metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self._write_records_csv(self.epoch_metrics_csv_path, self._epoch_records)

        if self.writer is not None:
            for key, value in record.items():
                if key == "epoch":
                    continue
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"epoch/{key}", value, epoch)
                elif isinstance(value, list):
                    for idx, sub_value in enumerate(value):
                        if isinstance(sub_value, (int, float)):
                            self.writer.add_scalar(f"epoch/{key}/layer_{idx}", sub_value, epoch)

    def save_array(self, loss_log, name, layer_losses=True):
        path = os.path.join(self.metrics_dir, f"{self.run_name}_{name}.csv")
        # save_traininglog_to_path(loss_log, path, layer_losses=layer_losses)
        self.run_summary["artifacts"][name] = path
        self._flush_summary()
        return path

    def save_checkpoint(self, model, dataset, tag, score=None, max_to_keep=2):
        path = os.path.join(self.checkpoints_dir, f"{tag}.pth")
        save_state(model, "", dataset, "", output_path=path)
        checkpoint_info = {"tag": tag, "path": path, "score": score}
        current = self.run_summary.get("best_checkpoints", [])
        current.append(checkpoint_info)
        current = sorted(
            current,
            key=lambda item: float("inf") if item.get("score") is None else float(item["score"])
        )
        dropped = current[max_to_keep:]
        current = current[:max_to_keep]

        for item in dropped:
            stale_path = item.get("path")
            if stale_path and os.path.exists(stale_path):
                os.remove(stale_path)

        self.run_summary["best_checkpoints"] = current
        self.run_summary["artifacts"]["best_checkpoints"] = [item["path"] for item in current]
        self._flush_summary()
        return path

    def register_artifact(self, key, path):
        self.run_summary["artifacts"][key] = path
        self._flush_summary()

    def update_summary(self, **kwargs):
        self.run_summary.update(make_serializable(kwargs))
        self._flush_summary()

    def _flush_summary(self):
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(self.run_summary, f, indent=2)

    def _flatten_record(self, record):
        flat = {}
        for key, value in record.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat[f"{key}.{sub_key}"] = json.dumps(sub_value) if isinstance(sub_value, (list, dict)) else sub_value
            elif isinstance(value, list):
                flat[key] = json.dumps(value)
            else:
                flat[key] = value
        return flat

    def _write_records_csv(self, output_path, records):
        rows = [self._flatten_record(record) for record in records]
        if not rows:
            return
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def close(self):
        self._flush_summary()
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()