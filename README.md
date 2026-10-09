# Research Visualizations

Educational Manim animations for deep learning, computer vision, and robotics.

## Environment

- Ubuntu 24.04 host, Docker Engine and Docker Compose
- Ubuntu 24.04 container with Python 3.12 and uv
- Manim Community Edition, default render quality **720p30**
- Python environment and uv cache isolated in Docker named volumes

This is an independent project. [manim-kor](https://github.com/CodingVillainKor/manim-kor) is a reference, not a dependency.

## Setup

```bash
git clone https://github.com/nayana224/research-visualizations.git
cd research-visualizations
docker compose build
docker compose run --rm manim uv sync
```

## Render ViT scenes

```bash
# 01: (3, 4, 4) image -> (4, 12) patches -> (4, 6) embeddings
docker compose run --rm manim uv run manim scenes/vit/patch_embedding.py ViTPatchEmbedding

# 02: (4, 6) patch tokens -> CLS -> (5, 6) -> positional embeddings
docker compose run --rm manim uv run manim scenes/vit/cls_position.py ViTCLSPosition
```

Default output paths:
- `media/videos/patch_embedding/720p30/ViTPatchEmbedding.mp4`
- `media/videos/cls_position/720p30/ViTCLSPosition.mp4`

For a faster preview, add `-ql` after `manim` (480p15).
For 1080p, add `-qh` (1080p60).

## Full ViT video (Korean explanations)

A single 720p30 video combines the validated Patch Embedding choreography with CLS Token,
Positional Embedding, Multi-Head Self-Attention, Encoder, and Classification.
Architecture keywords remain in English while explanatory subtitles use Korean.

```bash
docker compose run --rm manim uv run manim scenes/vit/vit_full.py ViTFullPipeline
```

Output: `media/videos/vit_full/720p30/ViTFullPipeline.mp4`

This visual walkthrough is schematic, not a numerical implementation of trained ViT weights.
Verify Korean font rendering and key frames locally before publication.

## First Manim exercise

```bash
docker compose run --rm manim uv run manim scenes/basics/hello_manim.py HelloManim
```

The `media/` directory is bind-mounted to the host and ignored by Git. The scenes use illustrative colors rather than numerical trained weights. Rendering of the new CLS scene still needs to be confirmed on the workstation.

## Structure

```text
docker/Dockerfile
compose.yaml
manim.cfg                    720p30 default
pyproject.toml
scenes/basics/hello_manim.py
scenes/vit/patch_embedding.py
scenes/vit/cls_position.py
AGENTS.md
```

## Notes

- No `raenim`, PyTorch or CUDA is needed for these scenes.
- `.venv` and uv cache are Docker named volumes; `docker compose down -v` deletes those volumes.
- Docker can modify the bind-mounted source directory.
- Additional fonts or TeX packages may be required for future scenes.
