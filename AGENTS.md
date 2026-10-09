# Agent Guidelines

This repository creates short, accurate educational animations with Manim.

- Prioritize conceptual correctness and readable motion over visual complexity.
- Use PyTorch-style tensor shapes (C, H, W) for individual images; state batch dimensions explicitly.
- Never hide a shape change. Show where each dimension comes from, especially channels and flattening order.
- Focus on one transformation at a time. Fade out or simplify previous objects before introducing new ones.
- Avoid overlapping objects, labels, arrows, and annotations. Keep labels close to their referents and within frame bounds.
- Prefer a small number of well-spaced objects; remove redundant arrows, borders, and text.
- Distinguish illustrative values/colors from numerically calculated outputs.
- Keep scenes independent, with minimal dependencies and comments.
- Before calling a scene complete, render it and inspect key frames (opening, each transition, final frame) for clipping and overlap.
- Do not claim render success without running the render. Report unverified results clearly.
- Preserve existing work and avoid overwriting uncommitted local changes. Update README commands when adding a scene.

- Keep architecture keywords in English, but use concise Korean explanations where they improve understanding. Use a Korean-capable font and verify glyph rendering.
- Full-pipeline videos must reuse validated stage choreography rather than copying unrelated scene files into one layout.

- For adjacent stages, preserve and move the same input mobject when it represents the next operation; avoid resetting it just to explain a new step.
- Animate each calculation as input -> operator/weights -> output before showing summary shapes; narration stays concise Korean, keywords English.
