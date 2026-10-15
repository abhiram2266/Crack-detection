"""Asset-type classifier: which kind of asset fills the frame (building, solar panel, …).

YOLO-World finds whole objects in a scene, but a close-up (cracked concrete, a panel
surface) has no object outline, so it detects nothing. This classifier lets the "auto"
detector (ml/detector.py) label such close-ups without the operator choosing a type.

It is a multinomial logistic regression on the frozen backbone's global embedding (the
same embedding the identity cascade already computes), trained on the real images in
dataset/processed (train split) plus `augment_views` degraded copies of each (webcam
augmentation + screen capture: glare wash-out, low contrast, blur, resampling), and
evaluated on the held-out test split, both clean and degraded.

  python training/train_asset_type_classifier.py            → models/asset_type_classifier.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

from ml.augment import WebcamAugment  # noqa: E402
from ml.config import cfg, load_config, resolve_path  # noqa: E402
from ml.embedding_model import AssetEmbeddingModel  # noqa: E402


def sample(manifest: Path, split: str, per_type: int, seed: int):
    by_type = defaultdict(list)
    with open(manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                by_type[row["asset_type"]].append(row["path"])
    rng = random.Random(seed)
    out = []
    for t, paths in sorted(by_type.items()):
        rng.shuffle(paths)
        out += [(p, t) for p in paths[:per_type]]
    return out


def screen_capture(img: Image.Image, rng: random.Random) -> Image.Image:
    """A photo shown on a screen and filmed by a webcam: glare, low contrast, blur, resampling."""
    w, h = img.size
    small = img.resize((max(16, int(w * rng.uniform(0.25, 0.6))), max(16, int(h * rng.uniform(0.25, 0.6)))))
    img = small.resize((w, h), Image.BILINEAR)
    img = Image.blend(img, Image.new("RGB", img.size, (255, 255, 255)), rng.uniform(0.15, 0.5))
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.0)))


def degrade(webcam: WebcamAugment):
    def f(img: Image.Image, rng: random.Random) -> Image.Image:
        img = webcam(img, rng)
        return screen_capture(img, rng) if rng.random() < 0.7 else img
    return f


def embed(embedder: AssetEmbeddingModel, items, batch: int = 32, transform=None, seed: int = 0) -> np.ndarray:
    rng = random.Random(seed)
    feats = []
    for i in range(0, len(items), batch):
        imgs = [Image.open(ROOT / p).convert("RGB") for p, _ in items[i : i + batch]]
        if transform is not None:
            imgs = [transform(im, rng) for im in imgs]
        feats.append(embedder.encode_batch(imgs))
        print(f"  embedded {min(i + batch, len(items))}/{len(items)}", end="\r", flush=True)
    print()
    return np.concatenate(feats).astype(np.float32)


def main() -> int:
    config = load_config()
    c = cfg(config, "asset_type_classifier")
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-type", type=int, default=int(c["train_per_type"]))
    ap.add_argument("--test-per-type", type=int, default=int(c["test_per_type"]))
    args = ap.parse_args()

    manifest = resolve_path(cfg(config, "datasets", "prepare", "output_dir")) / "manifest.csv"
    embedder = AssetEmbeddingModel.from_config(config)
    t0 = time.time()
    train = sample(manifest, "train", args.train_per_type, 1)
    test = sample(manifest, "test", args.test_per_type, 2)
    print(f"train {len(train)} images, test {len(test)} images, backbone {embedder.active_model_name}")
    aug = degrade(WebcamAugment(cfg(config, "condition_classifier", "augment")))
    views = int(c["augment_views"])
    x_train = np.concatenate([embed(embedder, train)] + [embed(embedder, train, transform=aug, seed=10 + v) for v in range(views)])
    y_train = [t for _, t in train] * (1 + views)
    x_test = embed(embedder, test)
    x_test_degraded = embed(embedder, test, transform=aug, seed=99)
    y_test = [t for _, t in test]

    clf = LogisticRegression(C=float(c["regularization_c"]), max_iter=2000, class_weight="balanced")
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prob = clf.predict_proba(x_test).max(axis=1)
    acc = float((pred == np.array(y_test)).mean())
    report = classification_report(y_test, pred, output_dict=True)
    acc_degraded = float((clf.predict(x_test_degraded) == np.array(y_test)).mean())
    print(classification_report(y_test, pred, digits=3))
    print(f"degraded (webcam/screen) test accuracy {acc_degraded:.3f}")
    print("confusion (rows = true):", list(clf.classes_))
    print(confusion_matrix(y_test, pred, labels=clf.classes_))

    out = resolve_path(c["checkpoint"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        coef=clf.coef_.astype(np.float32),
        intercept=clf.intercept_.astype(np.float32),
        classes=np.array(clf.classes_),
        backbone=np.array(embedder.active_model_name),
    )
    metrics = {
        "backbone": embedder.active_model_name,
        "train_images": len(train),
        "test_images": len(test),
        "test_accuracy": acc,
        "test_accuracy_degraded": acc_degraded,
        "augment_views": views,
        "test_confidence_p05": float(np.percentile(prob, 5)),
        "per_class": {k: v for k, v in report.items() if k in clf.classes_},
        "seconds": round(time.time() - t0, 1),
    }
    mpath = resolve_path(c["metrics_path"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, indent=2))
    print(f"test accuracy {acc:.3f} -> {out} ({metrics['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
