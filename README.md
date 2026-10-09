# Research Visualizations

Python animations for learning and explaining deep learning, computer vision, and robotics.

## Environment

- Host: Ubuntu 24.04 LTS with Docker Engine and Docker Compose plugin
- Container: Ubuntu 24.04
- Python: 3.12 (managed by `uv`)
- Renderer: Manim Community Edition (CPU-based first)
- Isolation: Python virtual environment and uv cache reside in Docker named volumes.

This is an independent project. [manim-kor](https://github.com/CodingVillainKor/manim-kor) is a learning reference, **not a dependency**.

## Quick start

```bash
git clone https://github.com/nayana224/research-visualizations.git
cd research-visualizations
docker compose build
docker compose run --rm manim uv sync
docker compose run --rm manim uv run manim -ql scenes/basics/hello_manim.py HelloManim
```

Open the generated MP4 on the host, under `media/videos/hello_manim/` (the exact subdirectory can depend on Manim version).

To use an interactive shell:

```bash
docker compose run --rm manim
# inside container:
uv sync
uv run manim -ql scenes/basics/hello_manim.py HelloManim
```

The default workflow writes MP4 files to the host-mounted `media/` directory. It does not open a video player inside Docker.

## ViT patch embedding visualization

The first research scene illustrates a shape-consistent toy example:
`(4, 4, 3)` RGB image -> four `(2, 2, 3)` patches -> four 12-value
flattened vectors -> learned linear projection -> four 6-dimensional tokens.

```bash
docker compose run --rm manim uv run manim -ql scenes/vit/patch_embedding.py ViTPatchEmbedding
```

Open `media/videos/patch_embedding/480p15/ViTPatchEmbedding.mp4` on the host.
The embedding bars illustrate the output shape; they are *not* values
calculated using actual trained projection weights.

## First hands-on exercise

Run the sample scene, then change the square's color, transform time, or add a `Text` object. Render again and compare the videos.

## Structure

```text
docker/Dockerfile          System dependencies and uv
compose.yaml               Container and named volume configuration
pyproject.toml             Minimal Python dependencies
scenes/basics/             Small learning exercises
scenes/transformer/        Future attention and positional encoding scenes
scenes/vit/                Future patch embedding scenes
scenes/dino/               Future teacher/student scenes
scenes/robotics/           Future robot geometry scenes
media/                     Generated videos (ignored by git)
```

The future scene folders are created as you add files; Git does not track empty directories.

## Notes

- The first setup uses standard Manim CE. `raenim`, PyTorch, and CUDA are deliberately omitted.
- The `.venv` environment and uv cache live in Docker volumes; `docker compose down -v` removes these volumes and forces reinstallation. Do not run this command unless you intend to delete them.
- `docker compose down` alone does not remove named volumes.
- Source files and output videos are bind-mounted and can be written from the container.
- The initial HelloManim render has been confirmed on the user's Ubuntu 24.04 workstation.\n- Font/LaTeX requirements may vary between scenes. The base image includes basic TeX packages; install extra TeX packages only if a scene needs them.
- Docker rendering has **not yet been verified on your workstation**. Verify the first MP4 before considering the environment validated.
