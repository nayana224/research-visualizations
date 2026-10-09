"""ViT class token and positional embedding, continuing the (4, 6) toy example.

Colors represent illustrative vectors, not calculated model parameters.
"""

from manim import (
    BLUE_B, BLUE_D, DOWN, FadeIn, FadeOut, GREEN_B, GREY_B,
    LEFT, ORANGE, RIGHT, Scene, Rectangle, Text, TransformFromCopy,
    UP, VGroup, WHITE, YELLOW, LaggedStart, Create, SurroundingRectangle,
)

PATCH_COLORS = (BLUE_D, GREEN_B, ORANGE)
POSITION_COLORS = (BLUE_B, GREEN_B)


class ViTCLSPosition(Scene):
    def construct(self):
        title = Text("Vision Transformer | CLS + Position", font_size=33)
        title.to_edge(UP, buff=0.32)
        self.add(title)

        self.show_patch_tokens()
        self.show_prepend_cls()
        self.show_position_addition()
        self.show_transformer_input()

    def heading(self, name, detail):
        head = Text(name, font_size=27).move_to(UP * 2.45)
        subtitle = Text(detail, font_size=20, color=GREY_B)
        subtitle.next_to(head, DOWN, buff=0.18)
        return VGroup(head, subtitle)

    def token_row(self, palette, index=0, cls=False):
        cells = VGroup()
        for col in range(6):
            cell = Rectangle(
                width=0.36, height=0.39, stroke_color=WHITE, stroke_width=1
            )
            color = YELLOW if cls else palette[(index + col) % len(palette)]
            cell.set_fill(color, opacity=0.87)
            cells.add(cell)
        cells.arrange(RIGHT, buff=0.035)
        return cells

    def token_matrix(self, n, mode="patch", include_cls=False):
        palette = POSITION_COLORS if mode == "position" else PATCH_COLORS
        rows = VGroup()
        for i in range(n):
            is_cls = include_cls and i == 0
            rows.add(self.token_row(palette, max(i - 1, 0) if include_cls else i,
                                    cls=is_cls))
        rows.arrange(DOWN, buff=0.16)
        return rows

    def show_patch_tokens(self):
        header = self.heading("01  Patch embeddings", "4 image patches -> 4 token vectors")
        matrix = self.token_matrix(4)
        matrix.move_to(DOWN * 0.08)
        label = Text("Patch embeddings  (4, 6)", font_size=22)
        label.next_to(matrix, DOWN, buff=0.38)
        self.play(FadeIn(header))
        self.play(LaggedStart(*[FadeIn(row, shift=RIGHT * 0.25) for row in matrix],
                              lag_ratio=0.2))
        self.play(FadeIn(label))
        self.wait(0.9)
        self.play(FadeOut(VGroup(header, matrix, label)))

    def show_prepend_cls(self):
        header = self.heading("02  Prepend CLS token", "Sequence length increases: 4 -> 5")
        patches = self.token_matrix(4)
        patches.move_to(LEFT * 3.0 + DOWN * 0.08)
        cls = self.token_row(PATCH_COLORS, cls=True)
        cls.next_to(patches, UP, buff=0.35)
        cls_tag = Text("CLS", color=YELLOW, font_size=20)
        cls_tag.next_to(cls, LEFT, buff=0.2)

        combined = self.token_matrix(5, include_cls=True)
        combined.move_to(RIGHT * 3.0 + DOWN * 0.06)
        label_before = Text("(4, 6) + CLS (1, 6)", font_size=20)
        label_before.next_to(patches, DOWN, buff=0.4)
        label_after = Text("Token sequence  (5, 6)", font_size=20)
        label_after.next_to(combined, DOWN, buff=0.35)

        self.play(FadeIn(header), FadeIn(patches), FadeIn(label_before))
        self.play(FadeIn(cls), FadeIn(cls_tag))
        self.play(
            LaggedStart(
                TransformFromCopy(cls, combined[0]),
                *[TransformFromCopy(patches[i], combined[i + 1]) for i in range(4)],
                lag_ratio=0.18,
            ),
            run_time=2.4,
        )
        self.play(FadeIn(label_after))
        self.wait(0.9)
        self.play(FadeOut(VGroup(header, patches, cls, cls_tag, combined,
                                 label_before, label_after)))

    def show_position_addition(self):
        header = self.heading("03  Positional embeddings", "Add a learnable (5, 6) matrix element-wise")
        tokens = self.token_matrix(5, include_cls=True)
        tokens.move_to(LEFT * 3.15 + DOWN * 0.03)
        positions = self.token_matrix(5, mode="position")
        positions.move_to(RIGHT * 3.15 + DOWN * 0.03)
        plus = Text("+", font_size=38)
        plus.move_to(DOWN * 0.03)

        token_label = Text("Tokens  (5, 6)", font_size=20)
        token_label.next_to(tokens, DOWN, buff=0.35)
        pos_label = Text("Positions  (5, 6)", font_size=20)
        pos_label.next_to(positions, DOWN, buff=0.35)

        self.play(FadeIn(header), FadeIn(tokens), FadeIn(token_label))
        self.play(FadeIn(plus), FadeIn(positions), FadeIn(pos_label))
        self.wait(1.1)
        self.play(FadeOut(VGroup(header, tokens, positions, plus,
                                 token_label, pos_label)))

    def show_transformer_input(self):
        header = self.heading("04  Transformer input", "Tokens + positions = (5, 6)")
        result = self.token_matrix(5, include_cls=True)
        result.move_to(DOWN * 0.05)
        cls_outline = SurroundingRectangle(result[0], color=YELLOW, buff=0.08)
        result_label = Text("Final token sequence  (5, 6)", font_size=22)
        result_label.next_to(result, DOWN, buff=0.38)
        info = Text("First row is the CLS token", font_size=19, color=GREY_B)
        info.next_to(result_label, DOWN, buff=0.18)
        note = Text("Colors are schematic; no numerical addition is shown",
                    font_size=17, color=GREY_B).to_edge(DOWN, buff=0.34)
        self.play(FadeIn(header))
        self.play(LaggedStart(*[FadeIn(row, shift=UP * 0.12) for row in result],
                              lag_ratio=0.15))
        self.play(Create(cls_outline), FadeIn(result_label), FadeIn(info))
        self.play(FadeIn(note))
        self.wait(1.7)
