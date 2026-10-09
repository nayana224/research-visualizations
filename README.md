# Research Visualizations

Educational Manim visualizations for AI and robotics.

## Setup

```bash
git clone https://github.com/nayana224/research-visualizations.git
cd research-visualizations
docker compose build
docker compose run --rm manim uv sync
```

## ViT Full Pipeline

The repository maintains **one** self-contained scene: `ViTFullPipeline`.
The 720p30 video covers channel-first RGB patches, flattening, weight-matrix
multiplication for Linear Projection, CLS Token, Positional Embedding,
Self-Attention, Encoder, and Classification.

Technical keywords stay in English; short supporting explanations use Korean.
Colors illustrate tensor shapes; projected values are not numerically computed.

```bash
docker compose run --rm manim uv run manim scenes/vit/vit_full.py ViTFullPipeline
```

Output: `media/videos/vit_full/720p30/ViTFullPipeline.mp4`.
Use `-ql` for a quick preview. Actual rendering must be verified locally.

## Notes

- Python 3.12 via uv in an Ubuntu Docker container.
- `manim.cfg` defaults to 720p30.
- `media/` outputs are bind-mounted and ignored by Git.
- `docker compose down -v` deletes named cache/venv volumes.
