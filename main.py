"""Minimal TIPSv2 JAX/Flax vision encoder load + image inference."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import os
from pathlib import Path
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import tyro

from tips.scenic.configs import tips_model_config
from tips.scenic.models import tips
from tips.scenic.utils import checkpoint


PACKAGE_DIR = Path(__file__).resolve().parent


class Variant(StrEnum):
  tips_oss_g14_highres = "tips_oss_g14_highres"
  tips_oss_g14_lowres = "tips_oss_g14_lowres"
  tips_oss_so400m14_highres_largetext_distilled = (
      "tips_oss_so400m14_highres_largetext_distilled"
  )
  tips_oss_l14_highres_distilled = "tips_oss_l14_highres_distilled"
  tips_oss_b14_highres_distilled = "tips_oss_b14_highres_distilled"
  tips_oss_s14_highres_distilled = "tips_oss_s14_highres_distilled"
  tips_v2_g14 = "tips_v2_g14"
  tips_v2_so14 = "tips_v2_so14"
  tips_v2_l14 = "tips_v2_l14"
  tips_v2_b14 = "tips_v2_b14"


@dataclass(frozen=True)
class Args:
  variant: Variant = Variant.tips_v2_b14
  checkpoint: Path | None = None
  image: Path = PACKAGE_DIR / "scenic/images/example_image.jpg"
  image_size: int = 448


def variant_name(variant: Variant | str) -> str:
  return variant.value if isinstance(variant, Variant) else variant


def cache_dir() -> Path:
  base = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
  return base / "tips" / "checkpoints"


def checkpoint_url(variant: Variant | str) -> str:
  name = variant_name(variant)
  version = "v2_0" if name.startswith("tips_v2_") else "v1_0"
  return (
      f"https://storage.googleapis.com/tips_data/{version}/checkpoints/scenic/"
      f"{name}_vision.npz"
  )


def default_checkpoint_path(variant: Variant | str) -> Path:
  return cache_dir() / f"{variant_name(variant)}_vision.npz"


def resolve_checkpoint_path(
    variant: Variant | str,
    checkpoint_path: str | Path | None,
) -> Path:
  if checkpoint_path is not None:
    return Path(checkpoint_path).expanduser()

  name = variant_name(variant)
  path = default_checkpoint_path(name)
  if path.exists():
    return path

  path.parent.mkdir(parents=True, exist_ok=True)
  url = checkpoint_url(name)
  print(f"Downloading {name} JAX checkpoint to {path}")
  urllib.request.urlretrieve(url, path)
  return path


def load_tips_vision(
    variant: Variant | str,
    checkpoint_path: str | Path,
    image_size: int,
):
  config = tips_model_config.get_config(variant)
  model = tips.VisionEncoder(
      variant=config.variant,
      pooling=config.pooling,
      num_cls_tokens=config.num_cls_tokens,
      posembs=tuple(config.positional_embedding.shape),
  )

  dummy = jnp.ones((1, image_size, image_size, 3), dtype=jnp.float32)
  init_vars = model.init(jax.random.PRNGKey(0), dummy, train=False)
  params = checkpoint.load_checkpoint(
      Path(checkpoint_path).expanduser(), init_vars["params"]
  )
  return model, params


def load_image(image_path: str | Path, image_size: int):
  image = Image.open(Path(image_path).expanduser()).convert("RGB")
  image = np.asarray(image, dtype=np.float32) / 255.0
  return jax.image.resize(image, (image_size, image_size, 3), method="bilinear")


def l2_normalize(x, eps: float = 1e-6):
  return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


def main():
  args = tyro.cli(Args)
  checkpoint_path = resolve_checkpoint_path(args.variant, args.checkpoint)

  model, params = load_tips_vision(args.variant, checkpoint_path, args.image_size)
  image = load_image(args.image, args.image_size)

  spatial_features, cls_embeddings = model.apply(
      {"params": params},
      image[None],
      train=False,
  )
  first_cls = l2_normalize(cls_embeddings[:, 0, :])

  print("spatial_features:", spatial_features.shape)
  print("cls_embeddings:", cls_embeddings.shape)
  print("first_cls_first_5:", np.asarray(first_cls[0, :5]))


if __name__ == "__main__":
  main()
