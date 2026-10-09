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

## Publication-quality figure language
- Use the original ViT paper's Figure 1 as a conceptual reference for Patch Embeddings, prepend CLS, Positional Embedding, repeated Transformer Encoder and MLP classification head. The animation must not suggest that toy numbers are trained activations.
- Add paper-like component labels, dimensional annotations and operational links, but introduce one new relationship per beat (progressive disclosure).
- Attention: distinguish QK^T score matrix (N,N), row-wise softmax and attention-value multiplication. Highlight one query row before the whole matrix is revealed.
- Draw residuals as an actual bypass with an addition node, rather than a label attached to the output. Pre-LN must show LayerNorm before the sublayer.
- Keep detail readable at 720p: avoid tiny type, uncontrolled z-order, and lines crossing other blocks; allow a dedicated close-up instead of cramming a full figure onto the screen.
- Visual values in a colored attention map are illustrative unless actually computed; captions must disclose that.

## Mathematical narration and global structure
- Begin each conceptual section with a compact, original architecture map derived from ViT Figure 1, highlighting the component being explained. Remove the map before detailed calculation to protect readability.
- Use precise language: Query–Key score is a dot product, followed by scale by sqrt(d_k); Softmax normalizes a row of scores; the normalized row weights Values. Never describe a dot product solely as a vague "comparison".
- A projection maps x via a *learned* matrix. Demonstrate matching scalar products and their sum before showing output vectors.
- A Head is a parallel attention computation with its own projected Q_i, K_i and V_i. Do not imply that each head is only a decorative rectangle.
- Concat joins head features along the last dimension: for two heads, (5,3) and (5,3) become (5,6), not addition or stacking along token dimension; show the physical joining.
- Residual addition is an elementwise addition of the bypassed input to the sublayer output, after Pre-LN and the operation.
- Paper figures are conceptual references. Do not copy copyrighted imagery or duplicate its exact layout; reproduce technical relationships in original Manim graphics.
- Explicitly label synthetic values/shades and avoid implying random shapes constitute measured attention maps.

## Numeric demonstrations and classification (2026-10-09)
- Show an original whole-model Figure 1-inspired overview at the start, then show the active component before local explanations. Preserve object identity across stage 03 → 04.
- Explain dot products with concrete coordinatewise multiplication, summation, scalar output; distinguish unscaled from scaled scores.
- Show **mathematically correct** toy examples. Softmax weights must sum to 1; residual example must add coordinatewise; class probabilities must sum to 1.
- Clearly identify all toy values as illustrative, not measured model inference.
- Close with a CLS → MLP Head → class probability bars → highlighted class prediction animation. Class names and numbers are examples, not claims about a real image.
- Be accurate about the published ViT diagram while creating original vectors and layout; do not embed paper photo assets unless their reuse rights are confirmed.

## Paper-grounded ViT fidelity (source PDFs in this repository)
- Primary references: `references/papers/05_VisionTransformer_ViT_2020.pdf`, Figure 1 and Section 3.1 (equations 1–4); `references/papers/04_AttentionIsAllYouNeed_2017.pdf`, Section 3.2.1–3.2.2 (scaled dot-product, multi-head).
- Before changing an operation, check its defining equation in these PDFs, then annotate its real shape separately from reduced toy dimensions. Cite page/section in code comments only where it clarifies a non-obvious choice.
- In Figure 1, the trainable CLS embedding is prepended to projected patches, then 1-D learned positional embeddings are added element-wise.
- ViT encoder equations: z'_l = MSA(LN(z_(l-1))) + z_(l-1); z_l = MLP(LN(z'_l)) + z'_l. LN precedes both sublayers; each has its own residual skip. The MLP has two layers with a GELU nonlinearity.
- Scaled dot-product attention: A = softmax(QK^T / sqrt(d_k)); output = A V. Show the scaling step explicitly in narration before Softmax. A single dot product alone is a score, not the full attention operation.
- Classification uses the final CLS hidden state (not averaging all patch tokens). Original ViT pretrained classification uses a one-hidden-layer MLP; fine-tuning uses a single linear layer. Do not misrepresent illustrative class probabilities as output from a pretrained model.
- Keep the original Figure 1 as a conceptual source for the architecture schematic. If an exact PDF figure image is embedded, verify appropriate redistribution rights; otherwise draw original Manim geometry.
- Numerical values, matrix heatmaps and visual class outputs are explicitly illustrative unless calculations are executed. Ensure examples remain mathematically consistent.
