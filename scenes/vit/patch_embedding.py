"""A small, shape-consistent walkthrough of ViT patch embedding.

Toy example: 4x4 RGB pixels, 2x2 patches, 4 patches, 12 values per patch,
and an illustrative learned linear projection from 12 to 6 dimensions.
"""

from manim import (
    BLUE_B, BLUE_D, DOWN, FadeIn, FadeOut, GOLD_B, GREEN_B, GREY_B,
    LEFT, LaggedStart, Line, ORIGIN, RIGHT, Scene, Square, SurroundingRectangle,
    Text, TransformFromCopy, UP, VGroup, WHITE, YELLOW, Create, Rectangle,
)


class ViTPatchEmbedding(Scene):
    def construct(self):
        title = Text("Vision Transformer | Patch Embedding", font_size=34)
        title.to_edge(UP, buff=0.35)
        subtitle = Text(
            "Toy example: 4 x 4 RGB image, 2 x 2 patches",
            font_size=21, color=GREY_B,
        ).next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(title, shift=DOWN), FadeIn(subtitle, shift=DOWN))

        # A 4 x 4 RGB image, grouped into four 2 x 2 patches.
        palette = [
            [BLUE_D, BLUE_B, GREEN_B, GREEN_B],
            [BLUE_B, BLUE_D, GREEN_B, BLUE_B],
            [GOLD_B, GOLD_B, BLUE_B, BLUE_D],
            [GOLD_B, GREEN_B, BLUE_D, BLUE_B],
        ]
        pixel_size = 0.62
        pixels = VGroup()
        for row in range(4):
            for col in range(4):
                cell = Square(side_length=pixel_size, stroke_width=1.5)
                cell.set_fill(palette[row][col], opacity=0.92)
                cell.move_to(
                    LEFT * 3.85
                    + RIGHT * (col - 1.5) * pixel_size
                    + UP * (1.5 - row) * pixel_size
                )
                pixels.add(cell)

        image_label = Text("Input image  (4 x 4 x 3)", font_size=21)
        image_label.next_to(pixels, DOWN, buff=0.35)

        patch_boxes = VGroup()
        for row in range(2):
            for col in range(2):
                # Each 2x2 group occupies a 1.24x1.24 region.
                center = (
                    LEFT * 3.85
                    + RIGHT * (col - 0.5) * 2 * pixel_size
                    + UP * (0.5 - row) * 2 * pixel_size
                )
                box = Square(side_length=2 * pixel_size)
                box.move_to(center).set_stroke(WHITE, width=4)
                patch_boxes.add(box)

        self.play(FadeIn(pixels), FadeIn(image_label))
        self.play(LaggedStart(*[Create(b) for b in patch_boxes], lag_ratio=0.18))
        self.wait(0.4)

        # Extract the four patches to the right. Each patch is a distinct
        # copy of the original pixels, preserving their original colors.
        patch_groups = VGroup()
        for row in range(2):
            for col in range(2):
                patch_pixels = VGroup()
                for inner_row in range(2):
                    for inner_col in range(2):
                        src_idx = (row * 2 + inner_row) * 4 + col * 2 + inner_col
                        patch_pixels.add(pixels[src_idx].copy())
                patch_groups.add(patch_pixels)

        targets = VGroup()
        for i in range(4):
            target = patch_groups[i].copy()
            target.scale(0.72)
            target.move_to(
                RIGHT * 0.4
                + RIGHT * (i % 2) * 2.0
                + UP * (0.5 - i // 2) * 1.4
            )
            targets.add(target)

        patch_label = Text("4 patches  (2 x 2 x 3 each)", font_size=20)
        patch_label.next_to(targets, DOWN, buff=0.45)
        self.play(
            LaggedStart(
                *[TransformFromCopy(patch_groups[i], targets[i]) for i in range(4)],
                lag_ratio=0.2,
            ),
            run_time=2.2,
        )
        self.play(FadeIn(patch_label))
        self.wait(0.6)

        # Replace spatial patches with flattened RGB vectors, then project.
        self.play(
            FadeOut(pixels), FadeOut(patch_boxes), FadeOut(image_label),
            FadeOut(patch_groups), FadeOut(patch_label),
            FadeOut(targets),
        )

        flatten_label = Text("Flatten: 2 x 2 x 3 = 12", font_size=26)
        flatten_label.move_to(UP * 2.0)
        self.play(FadeIn(flatten_label))

        vectors = VGroup()
        for row in range(4):
            cells = VGroup()
            for col in range(12):
                cell = Rectangle(width=0.27, height=0.38)
                cell.set_stroke(WHITE, width=1)
                cell.set_fill([BLUE_D, GREEN_B, GOLD_B][col % 3], opacity=0.85)
                cells.add(cell)
            cells.arrange(RIGHT, buff=0.025)
            vectors.add(cells)
        vectors.arrange(DOWN, buff=0.22).move_to(LEFT * 3.25 + DOWN * 0.2)
        self.play(LaggedStart(*[FadeIn(v, shift=RIGHT * 0.3) for v in vectors], lag_ratio=0.17))

        projection = VGroup(
            Rectangle(width=1.65, height=1.5, color=YELLOW),
            Text("Linear\n12 -> 6", font_size=23, color=YELLOW),
        )
        projection[1].move_to(projection[0])
        projection.move_to(ORIGIN + DOWN * 0.2)
        self.play(FadeIn(projection))
        for row in range(4):
            self.play(
                Create(Line(vectors[row].get_right(), projection.get_left(), color=GREY_B)),
                run_time=0.13,
            )
        self.wait(0.2)
        # The output bars represent a learned embedding, not computed
        # numerical outputs. The diagram communicates shapes, not weights.
        token_rows = VGroup()
        for row in range(4):
            cells = VGroup()
            for col in range(6):
                cell = Rectangle(width=0.32, height=0.38)
                cell.set_stroke(WHITE, width=1)
                cell.set_fill([BLUE_B, GREEN_B, GOLD_B, BLUE_D][(row + col) % 4], opacity=0.85)
                cells.add(cell)
            cells.arrange(RIGHT, buff=0.035)
            token_rows.add(cells)
        token_rows.arrange(DOWN, buff=0.22).move_to(RIGHT * 3.2 + DOWN * 0.2)

        self.play(LaggedStart(*[FadeIn(t, shift=RIGHT * 0.4) for t in token_rows], lag_ratio=0.22))
        tokens_caption = Text("4 tokens x 6 features", font_size=22)
        tokens_caption.next_to(token_rows, DOWN, buff=0.4)
        self.play(FadeIn(tokens_caption))
        highlight = SurroundingRectangle(token_rows, color=YELLOW, buff=0.14)
        self.play(Create(highlight))
        self.wait(1)
