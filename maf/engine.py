"""Portable training/evaluation entry points for the final MAF protocol.

Training only reads train/validation records. Test evaluation is a separate
explicit command and never changes checkpoint selection. CPU evaluation is
supported, but canonical reported predictions use CUDA BF16 at batch size16.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import CLIPConfig, CLIPTokenizerFast
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from tqdm import tqdm
from .core import (MAF, CORE_ABLATIONS, DGM4Dataset, build_optimizer, compute_loss,
                   first_epoch_sampler_manifest, load_json_or_jsonl,
                   loader_generator, module_state_sha256, move_batch, set_seed)


def binary_metrics(labels, probabilities):
    """ACC/AUC/EER as fractions, fixed threshold0.5 and interpolated ROC EER."""
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Nonempty equally sized vectors are required")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError("Invalid labels or probabilities")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("Expected probabilities in [0,1]")
    result = {"acc": float(accuracy_score(y, p >= 0.5)), "auc": None, "eer": None}
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, p, pos_label=1, drop_intermediate=False)
        delta = fpr - (1 - tpr)
        hi = int(np.flatnonzero(delta >= 0)[0])
        lo = hi if delta[hi] == 0 else hi - 1
        weight = 0 if hi == lo else -delta[lo] / (delta[hi] - delta[lo])
        result["auc"] = float(roc_auc_score(y, p))
        result["eer"] = float(fpr[lo] + weight * (fpr[hi] - fpr[lo]))
    return result

@torch.no_grad()
def evaluate_batches(model, loader, device, amp_dtype=torch.float16, prediction_path=None):
    model.eval()
    ys, base_ps, final_ps, states = ([], [], [], [])
    base_logits, final_logits = ([], [])
    diagnostics = defaultdict(list)
    for batch in tqdm(loader, desc='eval', leave=False):
        batch = move_batch(batch, device)
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
            out = model(batch)
        y = batch['label']
        # Preserve dtype before sigmoid: CUDA BF16 probability ties affect ROC metrics.
        base_p = torch.sigmoid(out['base_logit'])
        final_p = torch.sigmoid(out['final_logit'])
        ys.extend(y.cpu().numpy().tolist())
        base_ps.extend(base_p.float().cpu().numpy().tolist())
        final_ps.extend(final_p.float().cpu().numpy().tolist())
        states.extend(batch['state'].cpu().numpy().tolist())
        if prediction_path is not None:
            base_logits.extend(out['base_logit'].float().cpu().tolist())
            final_logits.extend(out['final_logit'].float().cpu().tolist())
            # Preserve internal predictions for the HAF analysis without recomputing fusion.
            for key in ("state_prob", "p_text", "p_image", "cross_head_agreement",
                        "fusion_gate", "state_coeff", "evidence_logit"):
                value = out.get(key)
                if value is None and key in ("p_text", "p_image"):
                    logit_key = "text_presence_logit" if key == "p_text" else "image_presence_logit"
                    if logit_key in out:
                        value = torch.sigmoid(out[logit_key])
                if value is not None:
                    diagnostics[key].extend(value.float().cpu().tolist())
    y = np.asarray(ys).astype(int)
    base_p = np.asarray(base_ps)
    final_p = np.asarray(final_ps)
    st = np.asarray(states).astype(int)
    base_pred = (base_p >= 0.5).astype(int)
    final_pred = (final_p >= 0.5).astype(int)

    def safe_auc(y_true, p):
        if np.unique(y_true).size < 2:
            return None
        try:
            return roc_auc_score(y_true, p)
        except Exception:
            return None
    metrics = {'base_acc': accuracy_score(y, base_pred), 'base_macro_f1': f1_score(y, base_pred, average='macro'), 'base_auc': safe_auc(y, base_p), 'final_acc': accuracy_score(y, final_pred), 'final_macro_f1': f1_score(y, final_pred, average='macro'), 'final_auc': safe_auc(y, final_p)}
    names = {0: 'real', 1: 'text', 2: 'image', 3: 'joint'}
    for k, name in names.items():
        mask = st == k
        if mask.any():
            metrics[f'acc_{name}'] = accuracy_score(y[mask], final_pred[mask])
            metrics[f'n_{name}'] = int(mask.sum())
    base_ok = base_pred == y
    final_ok = final_pred == y
    fixed = int((~base_ok & final_ok).sum())
    harmed = int((base_ok & ~final_ok).sum())
    metrics['fixed'] = fixed
    metrics['harmed'] = harmed
    metrics['net_correction'] = fixed - harmed
    metrics['both_correct'] = int((base_ok & final_ok).sum())
    metrics['both_wrong'] = int((~base_ok & ~final_ok).sum())
    if prediction_path is not None:
        destination = Path(prediction_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert len(loader.dataset.items) == len(y)
        temporary = destination.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            for i, item in enumerate(loader.dataset.items):
                record = dict(row_index=i, id=str(item['id']), image=item['image'], label=int(y[i]), state=int(st[i]), base_logit=base_logits[i], final_logit=final_logits[i], base_probability=float(base_p[i]), final_probability=float(final_p[i]), prediction=int(final_pred[i]))
                if hasattr(loader.dataset, "raw_indices"):
                    raw_index = int(loader.dataset.raw_indices[i])
                    split_name = getattr(loader.dataset, "split", "unknown")
                    record["metadata_row_index"] = raw_index
                    record["sample_id"] = f"{split_name}:{raw_index:08d}"
                for key, values in diagnostics.items():
                    if len(values) != len(y):
                        raise ValueError(f"Incomplete prediction diagnostics: {key}")
                    record[key] = values[i]
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        temporary.replace(destination)
    metrics['sample_count'] = int(len(y))
    if np.unique(y).size == 2:
        metrics['final_eer'] = binary_metrics(y, final_p)['eer']
    else:
        metrics['final_eer'] = None
    return metrics

DEFAULTS = {
    "data_root": "data/raw/DGM4", "source_metadata": "data/prepared/source_by_id.jsonl",
    "source": "bbc", "clip_model": "pretrained/clip-vit-base-patch16",
    "ablation": "full", "hidden_dim": 256, "epochs": 20, "seed": 42,
    "batch_size": 8, "grad_accum": 4, "workers": 0,
    "lr": 1e-4, "clip_lr": 5e-7, "weight_decay": 1e-4, "grad_clip": 1.0,
    "text_pos_weight": 5.0, "image_pos_weight": 8.0,
    "image_size": 224, "max_length": 77, "bf16": True,
    "disable_cudnn": True, "gradient_checkpointing": True,
    "limit_train": 0, "limit_eval": 0,
}
SOURCES = ("bbc", "guardian", "usa_today", "washington_post")


def configuration(values):
    """Fill explicit final-protocol defaults; reject misspelled or obsolete options."""
    allowed = set(DEFAULTS) | {"train_json", "val_json", "test_json"}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    c = dict(DEFAULTS)
    c.update(values)
    if c["ablation"] not in CORE_ABLATIONS or c["source"] not in SOURCES:
        raise ValueError("Invalid ablation or source")
    for key in ("epochs", "batch_size", "grad_accum", "max_length", "image_size", "hidden_dim"):
        if int(c[key]) < 1:
            raise ValueError(f"{key} must be positive")
    for key in ("workers", "limit_train", "limit_eval"):
        if int(c[key]) < 0:
            raise ValueError(f"{key} must be nonnegative")
    return c


def load_config(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text())
    else:
        import yaml
        value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("The configuration must be a flat mapping")
    return configuration(value)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def runtime(c, device):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    if c["disable_cudnn"]:
        torch.backends.cudnn.enabled = False
    if device.type == "cuda" and c["bf16"] and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The requested canonical BF16 evaluation requires a BF16-capable GPU")
    dtype = torch.bfloat16 if c["bf16"] else torch.float16
    return device, dtype


def build_loader(c, split, tokenizer, clip_config, device, *, source=None, limit=None, batch_size=None):
    """Filter the original split by VisualNews source without resplitting/deduplication."""
    mapping = {str(row["id"]): row["source"] for row in load_json_or_jsonl(c["source_metadata"])}
    if c["image_size"] != clip_config.vision_config.image_size:
        raise ValueError("Image resolution must match the CLIP configuration")
    if not 2 <= c["max_length"] <= clip_config.text_config.max_position_embeddings:
        raise ValueError("Invalid CLIP maximum text length")
    training = split == "train"
    data_root = Path(c["data_root"])
    annotation = c.get(f"{split}_json") or data_root / "metadata" / f"{split}.json"
    if limit is None:
        limit = c["limit_train"] if training else c["limit_eval"]
    ds = DGM4Dataset(annotation, data_root, tokenizer,
                     max_length=c["max_length"], image_size=c["image_size"],
                     patch_size=int(clip_config.vision_config.patch_size),
                     source=source or c["source"], source_by_id=mapping, limit=limit)
    ds.split = split
    if batch_size is None:
        batch_size = c["batch_size"] if training else c["batch_size"] * 2
    return DataLoader(ds, batch_size=batch_size, shuffle=training,
                      num_workers=c["workers"], pin_memory=device.type == "cuda",
                      persistent_workers=c["workers"] > 0,
                      generator=loader_generator(c["seed"], split))


def create_model(c, *, training=False):
    model = MAF(clip_name=c["clip_model"], hidden_dim=c["hidden_dim"], ablation=c["ablation"])
    model.set_clip_trainable(True)
    if training and c["gradient_checkpointing"]:
        model.clip.gradient_checkpointing_enable()
    return model


def load_weights(model, checkpoint, c=None):
    """Load exact tensors with strict keys; accepts original or portable checkpoints."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(saved, dict):
        raise ValueError("Expected a dictionary checkpoint")
    recorded = dict(saved.get("config", saved.get("args", {})))
    for key in ("ablation", "source", "seed"):
        if key in saved and key not in recorded:
            recorded[key] = saved[key]
    if c is not None:
        for key in ("ablation", "hidden_dim", "image_size", "max_length"):
            if key in recorded and recorded[key] != c[key]:
                raise ValueError(f"Checkpoint/config mismatch for {key}")
    model.load_state_dict(saved.get("model", saved), strict=True)
    return {"epoch": saved.get("epoch"), "metrics": saved.get("metrics"),
            "source": recorded.get("source"), "seed": recorded.get("seed")}


def train_one_epoch(model, loader, optimizer, scaler, device, args, epoch, amp_dtype):
    """Keep original microbatch loss, short-final-window accumulation and update order."""
    model.train()
    running = defaultdict(float)
    steps = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"train {epoch}")
    for step, batch in enumerate(progress):
        batch = move_batch(batch, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            out = model(batch)
            loss, parts = compute_loss(out, batch, args)
            accum_steps = min(args.grad_accum, len(loader) - (step // args.grad_accum) * args.grad_accum)
            loss = loss / accum_steps
        scaler.scale(loss).backward()
        if (step + 1) % args.grad_accum == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        running["loss"] += float(loss.detach()) * accum_steps
        for key, value in parts.items():
            running[key] += value
        steps += 1
        progress.set_postfix(loss=f"{running['loss'] / steps:.4f}")
    return {key: value / max(1, steps) for key, value in running.items()}


def save_checkpoint(path, model, optimizer, epoch, metrics, c):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "metrics": metrics, "config": c}, tmp)
    tmp.replace(path)


def train(config, output_dir, device="cpu"):
    """Train a fresh model using train/validation only. Existing runs are never overwritten."""
    c = configuration(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("best.pt", "last.pt", "metrics.jsonl")):
        raise FileExistsError(f"Existing training artifacts in {output}; choose a fresh directory")
    set_seed(c["seed"])
    device, dtype = runtime(c, device)
    tokenizer = CLIPTokenizerFast.from_pretrained(c["clip_model"])
    clip_config = CLIPConfig.from_pretrained(c["clip_model"])
    train_loader = build_loader(c, "train", tokenizer, clip_config, device)
    val_loader = build_loader(c, "val", tokenizer, clip_config, device)
    write_json(output / "config.json", c)
    write_json(output / "data_summary.json", {
        "source": c["source"], "preserve_duplicates": True,
        "train_count": len(train_loader.dataset), "val_count": len(val_loader.dataset),
        "test_access": False,
    })
    model = create_model(c, training=True)
    write_json(output / "model_initialization.json", {
        "seed": c["seed"], "source": "fresh_pretrained_clip_and_seeded_heads",
        "shared_module_state_sha256": {
            name: module_state_sha256(getattr(model, name))
            for name in ("clip", "base_fusion", "base_head")
        },
    })
    write_json(output / "sampler_first_epoch.json", first_epoch_sampler_manifest(train_loader))
    model.to(device)
    args = SimpleNamespace(**c)
    optimizer = build_optimizer(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
    best_acc = -1.0
    best_epoch = None
    for epoch in range(1, c["epochs"] + 1):
        start = time.time()
        stats = train_one_epoch(model, train_loader, optimizer, scaler, device, args, epoch, dtype)
        metrics = evaluate_batches(model, val_loader, device, amp_dtype=dtype)
        row = {"epoch": epoch, "train": stats, "val": metrics, "epoch_seconds": time.time() - start}
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        save_checkpoint(output / "last.pt", model, optimizer, epoch, metrics, c)
        if metrics["final_acc"] > best_acc:
            best_acc = metrics["final_acc"]
            best_epoch = epoch
            save_checkpoint(output / "best.pt", model, optimizer, epoch, metrics, c)
        print(json.dumps({"epoch": epoch, "val_acc": metrics["final_acc"], "best_epoch": best_epoch}), flush=True)
    selected = load_weights(model, output / "best.pt", c)
    checked = evaluate_batches(model, val_loader, device, amp_dtype=dtype,
                               prediction_path=output / "val_predictions.jsonl")
    for metric in ("final_acc", "final_auc", "final_macro_f1"):
        if checked[metric] is not None and not math.isclose(checked[metric], selected["metrics"][metric], rel_tol=0, abs_tol=1e-12):
            raise AssertionError(f"Reloaded validation metric mismatch: {metric}")
    completion = {"status": "COMPLETED", "epochs_completed": c["epochs"],
                  "selected_epoch": selected["epoch"], "seed": c["seed"],
                  "selection": "first strict maximum validation ACC", "threshold": 0.5,
                  "checkpoint_sha256": file_sha256(output / "best.pt"),
                  "validation_metrics": checked, "test_access": False,
                  "completed_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "completion.json", completion)
    return completion


def evaluate_checkpoint(config, checkpoint, output_dir, *, split="test", source=None,
                        device="cpu", limit=0, batch_size=None):
    """Evaluate a fixed checkpoint. An explicit target source enables cross-source evaluation."""
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    c = configuration(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{split}_metrics.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    if source is not None and source not in SOURCES:
        raise ValueError("Unknown target source")
    set_seed(c["seed"])
    device, dtype = runtime(c, device)
    model = create_model(c)
    selected = load_weights(model, checkpoint, c)
    if selected["source"] is not None and selected["source"] != c["source"]:
        raise ValueError("Config source must identify checkpoint training source; use target source separately")
    tokenizer = CLIPTokenizerFast.from_pretrained(c["clip_model"])
    clip_config = CLIPConfig.from_pretrained(c["clip_model"])
    loader = build_loader(c, split, tokenizer, clip_config, device,
                          source=source, limit=limit, batch_size=batch_size)
    digest = file_sha256(checkpoint)
    record = {"checkpoint_sha256": digest, "configuration": c,
              "source": source or c["source"], "training_source": c["source"],
              "split": split, "threshold": 0.5, "selected_epoch": selected["epoch"],
              "device": str(device), "precision": str(dtype) if device.type == "cuda" else "float32",
              "batch_size": loader.batch_size, "sample_limit": limit,
              "sample_count": len(loader.dataset), "full_split_count": loader.dataset.full_count,
              "scope": "partial" if limit else "full",
              "selection": "fixed supplied checkpoint; no test-driven selection"}
    write_json(output / "FREEZE.json", record)
    model.to(device)
    metrics = evaluate_batches(model, loader, device, amp_dtype=dtype,
                               prediction_path=output / f"{split}_predictions.jsonl")
    metrics.update({key: record[key] for key in ("checkpoint_sha256", "source", "training_source",
                   "split", "threshold", "selected_epoch", "precision", "scope", "batch_size")})
    metrics["seed"] = c["seed"]
    write_json(destination, metrics)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--data-root")
        p.add_argument("--source-metadata")
        p.add_argument("--clip-model")
        p.add_argument("--source", choices=SOURCES, help="Training source of the checkpoint")
        p.add_argument("--output-dir", required=True)
        p.add_argument("--device", default="cpu", help="Explicit cpu or cuda[:index]; never auto-selects a GPU")
        p.add_argument("--threads", type=int, default=2)
        if name == "train":
            p.add_argument("--seed", type=int)
            p.add_argument("--epochs", type=int)
            p.add_argument("--ablation", choices=CORE_ABLATIONS)
        else:
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--split", choices=["val", "test"], default="test")
            p.add_argument("--target-source", choices=SOURCES)
            p.add_argument("--limit", type=int, default=0, help="First N records for a clearly labeled partial evaluation")
            p.add_argument("--batch-size", type=int)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    c = load_config(args.config)
    for key in ("data_root", "source_metadata", "clip_model", "source", "seed", "epochs", "ablation"):
        if getattr(args, key, None) is not None:
            c[key] = getattr(args, key)
    if args.command == "train":
        result = train(c, args.output_dir, args.device)
    else:
        if args.limit < 0 or args.batch_size is not None and args.batch_size < 1:
            parser.error("Invalid sample limit or batch size")
        result = evaluate_checkpoint(c, args.checkpoint, args.output_dir, split=args.split,
                                     source=args.target_source, device=args.device,
                                     limit=args.limit, batch_size=args.batch_size)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
