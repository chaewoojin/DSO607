"""HW6: prepare -> train (resumable by repetition) -> finalize once.

Test outcomes are loaded only by prepare (to create a stratified split) and
finalize. Training sees development outcomes and unlabeled test predictors.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import torch
from torch import nn

METHODS = ["Lasso", "KNN", "RF", "DNN"]
SOURCE = "https://www.cs.toronto.edu/~kriz/cifar.html"


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def environment():
    versions = {}
    for p in ["numpy", "pandas", "scipy", "scikit-learn", "torch", "torchvision", "matplotlib", "joblib"]:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = "not installed"
    return dict(python=sys.version, platform=platform.platform(), packages=versions,
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                cpus=os.environ.get("SLURM_CPUS_PER_TASK", str(os.cpu_count())))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def counts(y):
    return {str(k): int((y == k).sum()) for k in [0, 1]}


def prepare(args):
    root = args.run
    if root.exists():
        raise FileExistsError(f"Use a new run directory: {root}")
    cfg = json.loads(args.config.read_text())
    if cfg["R"] < 2 or cfg["EPOCHS"] < 1 or cfg["patience"] < 1:
        raise ValueError("Require R >= 2, EPOCHS >= 1, patience >= 1")
    if cfg["image_size"] != 16 or any(len(cfg["candidates"][m]) < 2 for m in METHODS):
        raise ValueError("This architecture expects size 16 and >=2 candidates per method")
    if args.synthetic:
        # Integration testing only. Never used as homework results.
        rng = np.random.default_rng(607)
        images = rng.integers(0, 256, (200, 32, 32, 3), dtype=np.uint8)
        y = np.tile([0, 1], 100)
        original = np.arange(200)
    else:
        from torchvision.datasets import CIFAR10
        # Only the official training pool is used. The official test batch is excluded.
        ds = CIFAR10(str(args.data), train=True, download=True)
        labels = np.asarray(ds.targets)
        original = np.flatnonzero(np.isin(labels, [3, 5]))
        images = ds.data[original]
        y = (labels[original] == 5).astype(np.int64)  # cat=0, dog=1
    # Collapse byte-identical images; exclude any conflicting-label hash group.
    groups = {}
    for i, im in enumerate(images):
        groups.setdefault(hashlib.sha256(im.tobytes()).hexdigest(), []).append(i)
    keep = np.array(sorted(v[0] for v in groups.values() if len(set(y[v])) == 1))
    excluded_ids = original[np.setdiff1d(np.arange(len(y)), keep)].tolist()
    images, y, original = images[keep], y[keep], original[keep]
    # A fixed, nonlearned 2x2 average pools 32x32 RGB to 16x16 for EVERY method.
    x = images.astype(np.float32).reshape(-1, 16, 2, 16, 2, 3).mean((2, 4)) / 255.0
    x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))
    dev, test = train_test_split(np.arange(len(y)), test_size=0.2,
                                 stratify=y, random_state=cfg["split_seed"])
    split_arrays = {}
    rows = []
    for r in range(1, cfg["R"] + 1):
        split_seed = cfg["development_seed_base"] + r
        tr, va = train_test_split(np.arange(len(dev)), test_size=0.2,
                                  stratify=y[dev], random_state=split_seed)
        global_sets = [set(dev[tr]), set(dev[va]), set(test)]
        assert all(not global_sets[i] & global_sets[j] for i, j in [(0, 1), (0, 2), (1, 2)])
        assert set.union(*global_sets) == set(range(len(y)))
        for name, idx in [("train", dev[tr]), ("validation", dev[va]), ("test", test)]:
            assert len(np.unique(y[idx])) == 2
            rows.append(dict(repetition=r, subset=name, n=len(idx),
                             count_0=int((y[idx] == 0).sum()), count_1=int((y[idx] == 1).sum()),
                             split_seed=split_seed))
        split_arrays[f"train_{r}"] = tr
        split_arrays[f"validation_{r}"] = va
    if max(c["n_neighbors"] for c in cfg["candidates"]["KNN"]) > len(split_arrays["train_1"]):
        raise ValueError("K exceeds training sample size")
    root.mkdir(parents=True)
    dump(root / "config.json", cfg)
    np.savez_compressed(root / "development.npz", x=x[dev], y=y[dev])
    np.save(root / "test_x.npy", x[test])
    np.save(root / "SEALED_test_y.npy", y[test])
    np.savez(root / "indices.npz", development=original[dev], test=original[test],
             retained_source_ids=original, **split_arrays)
    pd.DataFrame(rows).to_csv(root / "split_counts.csv", index=False)
    dump(root / "data_description.json", dict(source=SOURCE, synthetic=args.synthetic,
         source_pool="CIFAR-10 official training batch only; original test batch excluded",
         label_map={"0": "cat (original label 3)", "1": "dog (original label 5)"},
         n=len(y), class_counts=counts(y), development_counts=counts(y[dev]),
         fixed_test_counts=counts(y[test]), excluded_source_ids=excluded_ids,
         deduplication="Keep first byte-identical image if labels agree; exclude conflicting groups",
         dependence_limit="No entity IDs; exact duplicates removed, near duplicates not ruled out"))
    tracked = ["config.json", "development.npz", "test_x.npy", "SEALED_test_y.npy", "indices.npz"]
    dump(root / "manifest.json", dict(files={f: digest(root / f) for f in tracked},
                                       code_sha256=digest(__file__), environment=environment()))
    print(f"Prepared {'SYNTHETIC TEST' if args.synthetic else 'CIFAR'}: {len(dev)} development, {len(test)} fixed test.")
    print("Test scores remain uncomputed. Split counts:")
    print(pd.DataFrame(rows).head(3).to_string(index=False))


def load_frozen(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["code_sha256"] != digest(__file__):
        raise ValueError("Code changed after preparation. Use a new run directory.")
    for name, expected in manifest["files"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    return json.loads((root / "config.json").read_text())


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.network(x).squeeze(1)


def cnn_fit(xtr, ytr, xva, yva, cfg, candidate, seed, device):
    seed_all(seed)
    model = CNN().to(device)  # fresh weights and optimizer for each candidate
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], **candidate)
    loss_fn = nn.BCEWithLogitsLoss()
    # Small enough to keep these arrays on a single GPU. No multiprocessing loader.
    xt = torch.from_numpy(xtr).to(device)
    yt = torch.from_numpy(ytr.astype(np.float32)).to(device)
    xv = torch.from_numpy(xva).to(device)
    yv = torch.from_numpy(yva.astype(np.float32)).to(device)
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_epoch, stale, history = float("inf"), 0, 0, []
    best = None
    batch = cfg["batch_size"]
    for epoch in range(1, cfg["EPOCHS"] + 1):
        model.train()
        order = torch.randperm(len(yt), generator=generator).to(device)
        total = 0.0
        for ids in order.split(batch):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[ids]), yt[ids])
            loss.backward()
            opt.step()
            total += loss.item() * len(ids)
        model.eval()
        with torch.no_grad():
            val_loss = sum(loss_fn(model(a), b).item() * len(b)
                           for a, b in zip(xv.split(batch), yv.split(batch))) / len(yv)
        if not np.isfinite(val_loss) or not np.isfinite(total):
            raise FloatingPointError("Nonfinite CNN loss")
        history.append(dict(epoch=epoch, training_loss=total / len(yt), validation_loss=val_loss))
        if val_loss < best_loss:  # strict improvement; min_delta=0, earliest equal minimum retained
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    model.load_state_dict(best)
    return model, dict(stopped_epoch=epoch, retained_epoch=best_epoch, history=history,
                      best_validation_loss=best_loss)


def cnn_probability(model, x, device, batch):
    model.eval()
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(model(t.to(device))).cpu().numpy()
                               for t in torch.from_numpy(x).split(batch)])


def train(args):
    root = args.run
    cfg = load_frozen(root)
    if (root / "FINALIZED.json").exists():
        raise ValueError("Finalized run: no further training allowed")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run through Slurm on a GPU node; --device cpu is for local testing.")
    torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))))
    data = np.load(root / "development.npz")
    x, y = data["x"], data["y"]
    test_x = np.load(root / "test_x.npy")  # no test labels loaded in this function
    indices = np.load(root / "indices.npz")
    (root / "repetitions").mkdir(exist_ok=True)
    env = environment()
    for r in range(1, cfg["R"] + 1):
        dest = root / "repetitions" / f"{r:03d}"
        if dest.exists():
            if json.loads((dest / "complete.json").read_text())["environment"]["packages"] != env["packages"]:
                raise ValueError("Package versions changed across resume")
            print(f"Repetition {r}: already complete", flush=True)
            continue
        temp = root / "repetitions" / f".{r:03d}.partial"
        if temp.exists():
            shutil.rmtree(temp)
        temp.mkdir()
        try:
            started = time.perf_counter()
            tr, va = indices[f"train_{r}"], indices[f"validation_{r}"]
            # Same training-fitted channel normalization for every method.
            mean = x[tr].mean(axis=(0, 2, 3), keepdims=True)
            std = np.maximum(x[tr].std(axis=(0, 2, 3), keepdims=True), 1e-6)
            np.savez(temp / "normalization.npz", mean=mean, std=std)
            xt, xv, xe = [np.ascontiguousarray((a - mean) / std) for a in [x[tr], x[va], test_x]]
            flat_tr, flat_va, flat_test = [a.reshape(len(a), -1) for a in [xt, xv, xe]]
            records, predictions = [], {}
            seed = cfg["model_seed_base"] + r
            for method in METHODS:
                best_key, selected, selected_extra, selected_record = None, None, None, None
                for j, candidate in enumerate(cfg["candidates"][method]):
                    t0 = time.perf_counter()
                    extra = {}
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        if method == "DNN":
                            model, extra = cnn_fit(xt, y[tr], xv, y[va], cfg, candidate, seed, device)
                            p = cnn_probability(model, xv, device, cfg["batch_size"])
                        else:
                            if method == "Lasso":
                                model = LogisticRegression(penalty="l1", solver="saga", fit_intercept=True,
                                    tol=cfg["lasso_tol"], max_iter=cfg["lasso_max_iter"], random_state=seed, **candidate)
                            elif method == "KNN":
                                model = KNeighborsClassifier(metric="euclidean", weights="uniform",
                                                             algorithm="brute", n_jobs=torch.get_num_threads(), **candidate)
                            else:
                                model = RandomForestClassifier(n_estimators=100, max_features="sqrt",
                                    min_samples_leaf=2, class_weight=None, random_state=seed,
                                    n_jobs=torch.get_num_threads(), **candidate)
                            model.fit(flat_tr, y[tr])
                            p = model.predict_proba(flat_va)[:, list(model.classes_).index(1)]
                    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
                        raise FloatingPointError(f"Invalid validation probabilities: {method}, candidate {j}")
                    rec = dict(repetition=r, method=method, candidate_index=j, settings=candidate,
                               model_seed=seed, split_seed=cfg["development_seed_base"] + r,
                               validation_error=float(np.mean((p >= 0.5) != y[va])),
                               fit_validation_seconds=time.perf_counter() - t0,
                               warnings=[f"{w.category.__name__}: {w.message}" for w in caught], **extra)
                    records.append(rec)
                    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                        dump(temp / "candidates.json", records)
                        raise RuntimeError("Lasso did not converge. See candidate warnings; resolve on development data in a new run.")
                    # Smaller candidate index wins an exact error tie.
                    key = (rec["validation_error"], j)
                    if best_key is None or key < best_key:
                        best_key, selected, selected_extra, selected_record = key, model, extra, rec
                for rec in records:
                    if rec["method"] == method:
                        rec["selected"] = rec is selected_record
                with warnings.catch_warnings(record=True) as prediction_warnings:
                    warnings.simplefilter("always")
                    if method == "DNN":
                        predictions[method] = cnn_probability(selected, xe, device, cfg["batch_size"])
                        torch.save({k: v.cpu() for k, v in selected.state_dict().items()}, temp / "DNN.pt")
                        pd.DataFrame(selected_extra["history"]).to_csv(temp / "DNN_curve.csv", index=False)
                    else:
                        predictions[method] = selected.predict_proba(flat_test)[:, list(selected.classes_).index(1)]
                        joblib.dump(selected, temp / f"{method}.joblib", compress=3)
                selected_record["test_prediction_warnings"] = [f"{w.category.__name__}: {w.message}" for w in prediction_warnings]
                if not np.isfinite(predictions[method]).all():
                    raise FloatingPointError(f"Nonfinite test probabilities: {method}")
                print(f"r={r} {method}: validation error={best_key[0]:.4f}, candidate={best_key[1]}", flush=True)
            # Probability = empirical training class frequency; >=0.5 resolves a tie to class 1.
            predictions["Majority"] = np.full(len(test_x), y[tr].mean(), dtype=np.float32)
            np.savez_compressed(temp / "test_probabilities.npz", **predictions)
            dump(temp / "candidates.json", records)
            dump(temp / "complete.json", dict(repetition=r, environment=env,
                                               total_seconds=time.perf_counter() - started))
            temp.rename(dest)  # only fully finished repetitions count for resume
        except Exception:
            with open(root / "failures.log", "a") as f:
                f.write(f"Repetition {r}, {time.ctime()}\n{traceback.format_exc()}\n")
            raise
    print("All repetitions finished. Test scores remain uncomputed. Run finalize when settings are frozen.")


def finalize(args):
    root = args.run
    cfg = load_frozen(root)
    description = json.loads((root / "data_description.json").read_text())
    if not description["synthetic"] and cfg["R"] != 100:
        raise ValueError("Real-data final evaluation requires R=100. Do not score the pilot test set.")
    dirs = [root / "repetitions" / f"{r:03d}" for r in range(1, cfg["R"] + 1)]
    if not all((d / "complete.json").exists() for d in dirs):
        raise RuntimeError("All configured repetitions must complete before test evaluation")
    # Validate every prediction file BEFORE reading outcomes.
    ntest = len(np.load(root / "test_x.npy", mmap_mode="r"))
    all_probs = []
    for d in dirs:
        with np.load(d / "test_probabilities.npz") as z:
            probs = {m: z[m].copy() for m in METHODS + ["Majority"]}
        if any(p.shape != (ntest,) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
               for p in probs.values()):
            raise ValueError(f"Invalid predictions: {d}")
        all_probs.append(probs)
    dump(root / "FINALIZED.json", dict(time=time.ctime(), R=cfg["R"], note="No further tuning of this experiment"))
    y = np.load(root / "SEALED_test_y.npy")  # first test-outcome use since split creation
    rows, candidate_rows, run_rows = [], [], []
    for r, (d, probs) in enumerate(zip(dirs, all_probs), start=1):
        candidate_records = json.loads((d / "candidates.json").read_text())
        candidate_rows.extend(candidate_records)
        run_rows.append(json.loads((d / "complete.json").read_text()))
        for method, p in probs.items():
            chosen = next((c for c in candidate_records if c["method"] == method and c["selected"]), {})
            rows.append(dict(repetition=r, method=method, test_error=float(np.mean((p >= 0.5) != y)),
                             validation_error=chosen.get("validation_error"),
                             selected_settings=json.dumps(chosen.get("settings", {})),
                             split_seed=cfg["development_seed_base"] + r,
                             model_seed=cfg["model_seed_base"] + r))
    table = pd.DataFrame(rows)
    table.to_csv(root / "errors_with_baseline.csv", index=False)
    table[table.method.isin(METHODS)].to_csv(root / "repetition_results.csv", index=False)
    table.groupby("method")["test_error"].agg(["mean", "std"]).reindex(METHODS + ["Majority"]).to_csv(root / "summary.csv")
    wide = table.pivot(index="repetition", columns="method", values="test_error")
    paired = wide[METHODS[1:]].sub(wide["Lasso"], axis=0)
    paired.to_csv(root / "paired_differences.csv")
    paired.agg(["mean", "std"]).T.to_csv(root / "paired_summary.csv")
    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(root / "all_candidates.csv", index=False)
    candidates.groupby("method")["fit_validation_seconds"].agg(["sum", "mean"]).to_csv(root / "timing_summary.csv")
    pd.DataFrame([dict(repetition=r["repetition"], total_seconds=r["total_seconds"])
                  for r in run_rows]).to_csv(root / "repetition_timings.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([wide[m] for m in METHODS])
    ax.set_xticks(range(1, len(METHODS) + 1))
    ax.set_xticklabels(METHODS)
    ax.set(xlabel="Classifier", ylabel="Fixed-test classification error (proportion)",
           title=f"CIFAR cat vs dog: {cfg['R']} development splits")
    fig.tight_layout()
    fig.savefig(root / "test_error_boxplot.png", dpi=180)
    plt.close(fig)
    curve = pd.read_csv(dirs[0] / "DNN_curve.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve.epoch, curve.training_loss, label="Training (online epoch average)")
    ax.plot(curve.epoch, curve.validation_loss, label="Validation")
    selected = next(c for c in candidate_rows if c["repetition"] == 1 and c["method"] == "DNN" and c["selected"])
    ax.axvline(selected["retained_epoch"], color="gray", linestyle="--", label="Retained epoch")
    ax.set(xlabel="Epoch", ylabel="Binary cross-entropy", title="Selected CNN, repetition 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "DNN_loss_rep1.png", dpi=180)
    plt.close(fig)
    print(pd.read_csv(root / "summary.csv").to_string(index=False))
    print("Errors are proportions; std uses ddof=1. These are NOT independent test samples.")


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "train", "finalize"])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--synthetic", action="store_true", help="Prepare synthetic integration-test data only")
    args = p.parse_args()
    {"prepare": prepare, "train": train, "finalize": finalize}[args.command](args)


if __name__ == "__main__":
    main()
