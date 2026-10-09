"""ViT patch embedding: channel-first shapes with uncluttered staged animation.

Illustrative RGB channel values; the linear projection output is schematic.
"""

from manim import (
    BLUE_B, BLUE_D, Create, DOWN, FadeIn, FadeOut, GREEN_B, GREY_B,
    LEFT, ORANGE, RIGHT, Scene, Square, Rectangle, Text, TransformFromCopy,
    UP, VGroup, WHITE, YELLOW, LaggedStart, Arrow,
)

COLORS = (ORANGE, GREEN_B, BLUE_B)  # R, G, B
CHANNEL_NAMES = ("R", "G", "B")


class ViTPatchEmbedding(Scene):
    def construct(self):
        title = Text("Vision Transformer | Patch Embedding", font_size=32)
        title.to_edge(UP, buff=0.3)
        self.add(title)

        # Each phase has its own composition; nothing accumulates under the next.
        self.show_partition()
        self.show_one_flatten()
        self.show_patch_matrix()
        self.show_projection()

    def heading(self, label, detail):
        main = Text(label, font_size=26)
        main.move_to(UP * 2.45)
        sub = Text(detail, font_size=19, color=GREY_B)
        sub.next_to(main, DOWN, buff=0.18)
        return VGroup(main, sub)

    def channel_grid(self, color, dimension, size):
        cells = VGroup()
        for row in range(dimension):
            for col in range(dimension):
                cell = Square(side_length=size, stroke_color=WHITE, stroke_width=1)
                cell.set_fill(color, opacity=0.82)
                cell.move_to(
                    RIGHT * (col - (dimension - 1) / 2) * size
                    + DOWN * (row - (dimension - 1) / 2) * size
                )
                cells.add(cell)
        return cells

    def channel_stack(self, dimension, size, offset=0.20):
        planes = VGroup()
        # Distinct planes, positioned diagonally like a C x H x W tensor.
        for i, color in enumerate(COLORS):
            plane = self.channel_grid(color, dimension, size)
            plane.shift(RIGHT * (2 - i) * offset + UP * (2 - i) * offset)
            planes.add(plane)
        return planes

    def patch_stack(self, size=0.32):
        return self.channel_stack(2, size, offset=0.17)

    def vector(self, width, height=0.35):
        pieces = VGroup()
        for color in COLORS:
            for _ in range(width // 3):
                cell = Rectangle(width=0.205, height=height, stroke_width=1,
                                 stroke_color=WHITE)
                cell.set_fill(color, opacity=0.86)
                pieces.add(cell)
        pieces.arrange(RIGHT, buff=0.025)
        return pieces

    def show_partition(self):
        header = self.heading("01  Patch partition", "x: (3, 4, 4)  |  patch size: 2")
        image = self.channel_stack(4, 0.46)
        image.move_to(LEFT * 3.0 + DOWN * 0.25)
        image_caption = Text("RGB channels", font_size=20)
        image_caption.next_to(image, DOWN, buff=0.35)

        patches = VGroup()
        for i in range(4):
            patch = self.patch_stack(size=0.29)
            patch.move_to(RIGHT * (1.8 + (i % 2) * 2.0)
                          + UP * (0.55 - (i // 2) * 1.65))
            patches.add(patch)
        patch_caption = Text("4 patches  |  each (3, 2, 2)", font_size=20)
        patch_caption.next_to(patches, DOWN, buff=0.34)

        self.play(FadeIn(header), FadeIn(image), FadeIn(image_caption))
        self.play(LaggedStart(*[FadeIn(p, scale=0.7) for p in patches],
                              lag_ratio=0.22), run_time=1.8)
        self.play(FadeIn(patch_caption))
        self.wait(1.0)
        self.play(FadeOut(VGroup(header, image, image_caption, patches, patch_caption)))

    def show_one_flatten(self):
        header = self.heading("02  Flatten one patch", "(3, 2, 2) -> 12 values")
        patch = self.patch_stack(size=0.62)
        patch.move_to(LEFT * 3.1 + DOWN * 0.1)
        patch_caption = Text("one RGB patch", font_size=20)
        patch_caption.next_to(patch, DOWN, buff=0.38)

        # A channel occupies four neighboring entries, in C-H-W order.
        row = self.vector(12, height=0.46)
        row.move_to(RIGHT * 2.7 + DOWN * 0.2)
        names = VGroup()
        for i, name in enumerate(CHANNEL_NAMES):
            label = Text(name, font_size=21, color=COLORS[i])
            label.next_to(VGroup(*row[i*4:(i+1)*4]), UP, buff=0.22)
            names.add(label)
        explanation = Text("4 R  +  4 G  +  4 B  =  12", font_size=20, color=GREY_B)
        explanation.next_to(row, DOWN, buff=0.6)

        self.play(FadeIn(header), FadeIn(patch), FadeIn(patch_caption))
        # One channel at a time: 2x2 cells visibly turn into four entries.
        for channel in range(3):
            self.play(
                TransformFromCopy(
                    patch[channel], VGroup(*row[channel*4:(channel+1)*4])
                ),
                FadeIn(names[channel]),
                run_time=0.9,
            )
        self.play(FadeIn(explanation))
        self.wait(1.1)
        self.play(FadeOut(VGroup(header, patch, patch_caption, row, names, explanation)))

    def show_patch_matrix(self):
        header = self.heading("03  Apply to all patches", "4 patches x 12 values = (4, 12)")
        patches = VGroup()
        rows = VGroup()
        for i in range(4):
            patch = self.patch_stack(size=0.19)
            patch.move_to(LEFT * 3.4 + UP * (1.5 - i) * 0.92 + DOWN * 0.15)
            patches.add(patch)
            line = self.vector(12)
            line.move_to(RIGHT * 1.5 + UP * (1.5 - i) * 0.92 + DOWN * 0.15)
            rows.add(line)

        patch_caption = Text("4 patches", font_size=19)
        patch_caption.next_to(patches, DOWN, buff=0.35)
        row_caption = Text("Patch matrix  (4, 12)", font_size=19)
        row_caption.next_to(rows, DOWN, buff=0.35)
        self.play(FadeIn(header), FadeIn(patches), FadeIn(patch_caption))
        self.play(LaggedStart(
            *[TransformFromCopy(patches[i], rows[i]) for i in range(4)],
            lag_ratio=0.22,
        ), run_time=2.6)
        self.play(FadeIn(row_caption))
        self.wait(0.9)
        self.play(FadeOut(VGroup(header, patches, rows, patch_caption, row_caption)))

    def show_projection(self):
        header = self.heading("04  Linear projection", "(4, 12)  ->  (4, 6)")
        inputs = VGroup()
        outputs = VGroup()
        for i in range(4):
            v = self.vector(12)
            v.move_to(LEFT * 3.8 + UP * (1.5 - i) * 0.62 + DOWN * 0.1)
            inputs.add(v)
            out = VGroup()
            for j in range(6):
                block = Rectangle(width=0.25, height=0.36,
                                  stroke_color=WHITE, stroke_width=1)
                block.set_fill((BLUE_D, GREEN_B, ORANGE)[(i+j) % 3], opacity=0.9)
                out.add(block)
            out.arrange(RIGHT, buff=0.03)
            out.move_to(RIGHT * 3.7 + UP * (1.5 - i) * 0.62 + DOWN * 0.1)
            outputs.add(out)

        linear = VGroup(
            Rectangle(width=2.0, height=1.3, color=YELLOW),
            Text("Linear\n12 -> 6", font_size=24, color=YELLOW),
        )
        linear[1].move_to(linear[0])
        linear.move_to(DOWN * 0.1)
        left_label = Text("(4, 12)", font_size=22)
        left_label.next_to(inputs, DOWN, buff=0.34)
        right_label = Text("(4, 6)", font_size=22)
        right_label.next_to(outputs, DOWN, buff=0.34)

        left_arrow = Arrow(inputs.get_right() + RIGHT*0.2,
                           linear.get_left() + LEFT*0.15,
                           buff=0, color=GREY_B, stroke_width=3)
        right_arrow = Arrow(linear.get_right() + RIGHT*0.15,
                            outputs.get_left() + LEFT*0.2,
                            buff=0, color=GREY_B, stroke_width=3)
        self.play(FadeIn(header), FadeIn(inputs), FadeIn(left_label))
        self.play(FadeIn(linear), Create(left_arrow))
        self.play(Create(right_arrow), FadeIn(outputs), FadeIn(right_label))
        note = Text("Illustrative embeddings (not computed weights)",
                    font_size=17, color=GREY_B)
        note.to_edge(DOWN, buff=0.48)
        self.play(FadeIn(note))
        self.wait(1.8)
