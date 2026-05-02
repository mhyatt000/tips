"""Minimal TIPSv2 JAX/Flax vision encoder load + image inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from tips.scenic.configs import tips_model_config
from tips.scenic.models import tips
from tips.scenic.utils import checkpoint


PACKAGE_DIR = Path(__file__).resolve().parent


def load_tips_vision(
    variant: str,
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
  params = checkpoint.load_checkpoint(checkpoint_path, init_vars["params"])
  return model, params


def load_image(image_path: str | Path, image_size: int):
  image = Image.open(Path(image_path).expanduser()).convert("RGB")
  image = np.asarray(image, dtype=np.float32) / 255.0
  return jax.image.resize(image, (image_size, image_size, 3), method="bilinear")


def l2_normalize(x, eps: float = 1e-6):
  return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--variant", default="tips_v2_b14")
  parser.add_argument(
      "--checkpoint",
      default=PACKAGE_DIR / "scenic/checkpoints/tips_v2_b14_vision.npz",
  )
  parser.add_argument("--image", default=PACKAGE_DIR / "scenic/images/example_image.jpg")
  parser.add_argument("--image-size", type=int, default=448)
  args = parser.parse_args()

  model, params = load_tips_vision(args.variant, args.checkpoint, args.image_size)
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
