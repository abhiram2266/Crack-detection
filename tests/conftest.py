"""Shared fixtures: isolated config, weight-free embedder, synthetic assets."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.config import load_config  # noqa: E402


def make_panel(seed: int, size=(320, 240)) -> Image.Image:
    """Textured synthetic asset with enough keypoints for geometric verification."""
    r = np.random.default_rng(seed)
    w, h = size
    im = Image.fromarray((r.random((h, w, 3)) * 60 + 80).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for _ in range(14):
        x, y = int(r.integers(0, w - 20)), int(r.integers(0, h - 20))
        d.rectangle(
            [x, y, x + int(r.integers(10, 40)), y + int(r.integers(10, 40))],
            outline=tuple(int(v) for v in r.integers(0, 255, 3)),
            width=3,
        )
    return im


def add_crack(im: Image.Image, width: int = 6) -> Image.Image:
    out = im.copy()
    w, h = out.size
    ImageDraw.Draw(out).line([(int(w * 0.1), int(h * 0.1)), (int(w * 0.9), int(h * 0.9))], fill=(10, 10, 10), width=width)
    return out


def add_dust(im: Image.Image, seed: int = 0) -> Image.Image:
    r = np.random.default_rng(seed)
    arr = np.asarray(im).astype(np.float32)
    arr = arr * 0.85 + 30 + r.normal(0, 6, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


@pytest.fixture
def config(tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg["model"]["name"] = "projection"
    cfg["storage"]["db_url"] = f"sqlite:///{tmp_path}/local.db"
    cfg["storage"]["images_dir"] = str(tmp_path / "images")
    cfg["change_detection"]["heatmap_dir"] = str(tmp_path / "heat")
    cfg["auth"]["secret_file"] = str(tmp_path / "jwt_secret")
    cfg["auth"]["credentials_file"] = str(tmp_path / "credentials.env")
    cfg["cloud"]["server"]["db_url"] = f"sqlite:///{tmp_path}/cloud.db"
    cfg["cloud"]["server"]["secret_file"] = str(tmp_path / "cloud_secret")
    cfg["detection"]["mode"] = "full_frame"
    cfg["context_risk"]["auto_refresh"] = False   # tests never call the real APIs
    return cfg


@pytest.fixture(scope="session")
def embedder():
    from ml.embedding_model import AssetEmbeddingModel

    return AssetEmbeddingModel.from_config(load_config(), "projection")


@pytest.fixture(scope="session")
def classifier(embedder):
    """Nearest-neighbour baseline over synthetic normal / cracked panels."""
    from ml.condition_classifier import PrototypeConditionClassifier

    normals = [make_panel(1000 + i) for i in range(6)]
    cracked = [add_crack(make_panel(2000 + i)) for i in range(6)]
    emb = embedder.encode_batch(normals + cracked)
    return PrototypeConditionClassifier(
        {"solar_panel": emb}, {"solar_panel": ["normal"] * 6 + ["crack"] * 6},
        float(load_config()["condition_classifier"]["prototype_temperature"]),
    )
