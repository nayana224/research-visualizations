# Agent Guidelines

## Purpose
Create one coherent educational animation with Manim Community Edition 0.20.1:
`scenes/vit/vit_full.py` → `ViTFullPipeline` (720p30 by default).

## Visual language
- Use English for technical terms (Patch, Token, Projection, Softmax, Residual, etc.) and short plain-Korean subtitles to explain *what happens*.
- One learning objective per beat. Do not overcrowd the screen with every pipeline component.
- Preserve a moving object between adjacent steps whenever it is the same tensor. In particular, the first row of stage 03 must be the exact object moved into stage 04.
- Reveal input → operation (W, Softmax, MSA, MLP) → output. Avoid mere simultaneous FadeIns for a mathematical process.
- Highlight the specific operands/column/row before the corresponding output appears; dim or clear unrelated geometry.
- Respect stage transitions: clear only objects that must disappear; never leave hidden accumulated labels/arrows behind.
- Keep math correct: individual images are (C,H,W); x(1,12) @ W(12,6) = y(1,6); (4,6) + CLS(1,6) → (5,6); positional vectors are added element-wise; two attention heads each have dimension 3; ViT uses pre-LayerNorm and residual additions.
- Clearly distinguish abstract colored bars or example attention weights from numerically computed results.

## Reference and compatibility
- Study the animation *principles* in [Imcommit GQA](https://github.com/CodingVillainKor/manimgl-imcommit/blob/master/src/gqa/main.py), [MLA](https://github.com/CodingVillainKor/manimgl-imcommit/blob/master/src/MLA/main.py), and [preLN](https://github.com/CodingVillainKor/manimgl-imcommit/blob/master/src/preln/main.py): target transforms, elementwise emphasis, staged focus, and persistent objects.
- Do not copy source code or import ManimGL, `manimlib`, or `raenimgl`; project uses Manim CE and must remain independently runnable.

## Engineering and validation
- Keep the single Scene and small, cohesive helpers; minimal comments except for non-obvious continuity and tensor-shape assumptions.
- Check that all Manim symbols are imported; avoid calling animations on empty or unrendered groups.
- Render the complete scene and inspect opening, stage transitions, multiplication, attention, residual paths, and closing for overlaps/clipping.
- Do not claim rendering or visual verification unless actually done. If not available, explicitly mark it unverified.
- Preserve unrelated work. Document accurate launch commands in README.
