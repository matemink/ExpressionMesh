from __future__ import annotations

import argparse
import random
from pathlib import Path

EMOTION_PROMPTS = {
    "happy": "genuine smile",
    "sad": "sad facial expression",
    "surprised": "surprised expression, open mouth, raised eyebrows",
}
PRESENTATIONS = ("adult woman", "adult man")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a reproducible synthetic face dataset")
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-class", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
    )
    args = parser.parse_args()

    try:
        import torch
        from diffusers import DiffusionPipeline
    except ImportError as error:
        raise RuntimeError(
            'Generation dependencies are missing. Install with: pip install -e ".[generation]"'
        ) from error

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipeline = DiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipeline.to(device)
    randomizer = random.Random(args.seed)

    for emotion, expression in EMOTION_PROMPTS.items():
        class_directory = args.output / emotion
        class_directory.mkdir(parents=True, exist_ok=True)
        for image_index in range(args.per_class):
            presentation = randomizer.choice(PRESENTATIONS)
            seed = args.seed + image_index + 10_000 * list(EMOTION_PROMPTS).index(emotion)
            generator = torch.Generator(device=device).manual_seed(seed)
            prompt = (
                f"Front-facing color portrait photograph of an {presentation}, {expression}, "
                "looking at camera, realistic skin texture, even light, sharp focus"
            )
            image = pipeline(
                prompt,
                negative_prompt="cartoon, illustration, deformed face, blur, watermark, text",
                generator=generator,
            ).images[0]
            image.save(class_directory / f"{image_index:04d}.png")


if __name__ == "__main__":
    main()
