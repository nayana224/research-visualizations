"""One continuous ViT explanation, with English keywords and Korean narration.

Run from the repository root: manim scenes/vit/vit_full.py ViTFullPipeline
Visual tokens are schematic. Attention equations describe actual operations.
"""
from manim import (
    BLUE_B, BLUE_D, DOWN, FadeIn, FadeOut, GREEN_B, GREY_B,
    LEFT, ORANGE, RIGHT, Scene, Square, Rectangle, Text, UP, VGroup,
    WHITE, YELLOW, Create, SurroundingRectangle, Arrow, LaggedStart, TransformFromCopy,
    MathTex,
)
COLORS = (ORANGE, GREEN_B, BLUE_B)
CHANNEL_NAMES = ("R", "G", "B")
KFONT = "Noto Sans CJK KR"


class ViTFullPipeline(Scene):
    """Single self-contained animation scene."""

    def construct(self):
        self.camera.background_color = "#10131C"
        title = Text("Vision Transformer | ViT", font_size=34)
        title.to_edge(UP, buff=0.28)
        self.add(title)
        stages = (
            self.show_partition,
            self.show_one_flatten,
            self.show_patch_matrix,
            self.show_projection,
            self.cls_and_position,
            self.attention,
            self.encoder,
            self.classification,
        )
        for index, stage in enumerate(stages):
            stage()
            # Some individual scenes intentionally leave their final frame on screen.
            # Remove everything except the persistent title before the next stage.
            if index < len(stages) - 1:
                leftovers = [item for item in self.mobjects if item is not title]
                if leftovers:
                    self.play(*[FadeOut(item) for item in leftovers], run_time=0.45)

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

    def ko(self, content, size=22, color=GREY_B):
        return Text(content, font=KFONT, font_size=size, color=color)

    def stage(self, number, english, korean):
        name = Text(f"{number}  {english}", font_size=28)
        name.move_to(UP * 2.40)
        desc = self.ko(korean, 21)
        desc.next_to(name, DOWN, buff=0.18)
        return VGroup(name, desc)

    def matrix(self, nrows, ncols=6, highlight_first=False, shades=None):
        shades = COLORS if shades is None else shades
        rows = VGroup()
        for r in range(nrows):
            cells = VGroup()
            for c in range(ncols):
                cell = Rectangle(width=0.33, height=0.38,
                                 stroke_color=WHITE, stroke_width=0.8)
                color = YELLOW if highlight_first and r == 0 else shades[(r+c) % len(shades)]
                cell.set_fill(color, opacity=0.88)
                cells.add(cell)
            cells.arrange(RIGHT, buff=0.025)
            rows.add(cells)
        rows.arrange(DOWN, buff=0.13)
        return rows

    def label(self, msg, target, buff=0.35):
        tag = Text(msg, font_size=21)
        tag.next_to(target, DOWN, buff=buff)
        return tag

    def cls_and_position(self):
        head = self.stage("05", "CLS Token", "Patch Embedding 앞에 CLS Token을 추가합니다.")
        patches = self.matrix(4).move_to(LEFT * 3.0 + DOWN * 0.2)
        destination = self.matrix(5, highlight_first=True).move_to(RIGHT * 2.7 + DOWN * 0.1)
        cls = self.matrix(1, highlight_first=True).move_to(LEFT * 3.0 + UP * 1.5)
        left_label = self.label("Patch Tokens  (4, 6)", patches)
        right_label = self.label("Sequence  (5, 6)", destination)
        cls_name = Text("CLS", color=YELLOW, font_size=22).next_to(cls, LEFT, buff=0.22)
        self.play(FadeIn(head), FadeIn(patches), FadeIn(left_label))
        self.play(FadeIn(cls, shift=DOWN * 0.2), FadeIn(cls_name))
        self.play(
            TransformFromCopy(cls, destination[0]),
            *[TransformFromCopy(patches[i], destination[i + 1]) for i in range(4)],
            run_time=2.0,
        )
        self.play(FadeIn(right_label))
        self.wait(0.7)
        self.play(FadeOut(VGroup(head, patches, destination, cls, cls_name,
                                 left_label, right_label)))

        head = self.stage("06", "Positional Embedding", "Token마다 대응하는 위치 벡터를 더합니다.")
        tokens = self.matrix(5, highlight_first=True).move_to(LEFT * 3.8 + DOWN * 0.05)
        positions = self.matrix(5, shades=(BLUE_B, GREEN_B)).move_to(DOWN * 0.05)
        result = self.matrix(5, shades=(ORANGE, BLUE_B, GREEN_B)).move_to(
            RIGHT * 3.8 + DOWN * 0.05
        )
        signs = VGroup(
            Text("+", font_size=35).move_to(LEFT * 1.9 + DOWN * 0.05),
            Text("=", font_size=35).move_to(RIGHT * 1.9 + DOWN * 0.05),
        )
        captions = VGroup(
            self.label("Tokens  (5, 6)", tokens),
            self.label("Position  (5, 6)", positions),
            self.label("Z0  (5, 6)", result),
        )
        self.play(FadeIn(head), FadeIn(tokens), FadeIn(captions[0]))
        self.play(FadeIn(positions), FadeIn(captions[1]), FadeIn(signs))
        for i in range(5):
            self.play(
                TransformFromCopy(tokens[i], result[i]),
                run_time=0.26,
            )
        self.play(FadeIn(captions[2]))
        self.wait(0.8)
        self.play(FadeOut(VGroup(head, tokens, positions, result, signs, captions)))

        head = self.stage("07", "Transformer Input", "CLS를 포함한 5개의 Token이 Encoder에 입력됩니다.")
        tokens = self.matrix(5, highlight_first=True).move_to(DOWN * 0.12)
        outline = SurroundingRectangle(tokens[0], color=YELLOW, buff=0.08)
        caption = self.label("Z0  (5, 6)", tokens)
        cls_caption = Text("CLS", color=YELLOW, font_size=20)
        cls_caption.next_to(tokens[0], LEFT, buff=0.28)
        self.play(FadeIn(head), FadeIn(tokens))
        self.play(Create(outline), FadeIn(cls_caption), FadeIn(caption))
        self.wait(0.8)
        self.play(FadeOut(VGroup(head, tokens, outline, cls_caption, caption)))

    def attention(self):
        head = self.stage("08", "Q, K, V Projection", "같은 입력에서 서로 다른 세 표현을 생성합니다.")
        x = self.matrix(5, highlight_first=True).move_to(LEFT * 4.7 + DOWN * 0.1)
        outputs = VGroup()
        for idx, name in enumerate(("Q", "K", "V")):
            matrix = self.matrix(5, shades=(COLORS[idx],))
            matrix.scale(0.58)
            box = VGroup(matrix, Text(name, font_size=23, color=COLORS[idx]))
            box[1].next_to(matrix, UP, buff=0.12)
            box.move_to(RIGHT * (idx * 2.4 - 0.4) + DOWN * 0.1)
            outputs.add(box)
        self.play(FadeIn(head), FadeIn(x))
        self.play(LaggedStart(
            *[TransformFromCopy(x, outputs[i][0]) for i in range(3)],
            lag_ratio=0.26,
        ), run_time=2.2)
        self.play(*[FadeIn(box[1]) for box in outputs])
        self.wait(0.7)
        self.play(FadeOut(VGroup(head, x, outputs)))

        head = self.stage("09", "Scaled Dot-Product Attention", "Q와 K로 관련도를 구하고 V를 가중합합니다.")
        q = Text("Q", font_size=41, color=BLUE_D)
        k = Text("K", font_size=41, color=GREEN_B)
        v = Text("V", font_size=41, color=ORANGE)
        boxes = VGroup()
        for txt in (q, k, v):
            frame = Rectangle(width=1.25, height=0.93, color=txt.get_color())
            txt.move_to(frame)
            boxes.add(VGroup(frame, txt))
        boxes.arrange(RIGHT, buff=1.10).move_to(UP * 0.85)
        formula = MathTex(
            r"\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V"
        ).scale(0.95).move_to(DOWN * 0.6)
        caption = self.ko("Attention Weight로 중요한 Token에 더 집중합니다.", 21)
        caption.next_to(formula, DOWN, buff=0.55)
        self.play(FadeIn(head), LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.2))
        self.play(FadeIn(formula, shift=UP * 0.2))
        self.play(FadeIn(caption))
        self.wait(1.2)
        self.play(FadeOut(VGroup(head, boxes, formula, caption)))

        head = self.stage("10", "Multi-Head Attention", "두 Head의 출력을 결합하고 다시 투영합니다.")
        source = self.matrix(5, highlight_first=True).move_to(LEFT * 4.65 + DOWN * 0.1)
        heads = VGroup()
        for i, color in enumerate((BLUE_B, GREEN_B)):
            rect = Rectangle(width=2.3, height=0.76, color=color)
            name = Text(f"Head {i+1}  (5, 3)", font_size=19).move_to(rect)
            heads.add(VGroup(rect, name))
        heads.arrange(DOWN, buff=0.45).move_to(LEFT * 0.25 + DOWN * 0.1)
        output = self.matrix(5).move_to(RIGHT * 4.3 + DOWN * 0.1)
        label = self.label("Concat + Linear  (5, 6)", output)
        self.play(FadeIn(head), FadeIn(source))
        self.play(
            *[TransformFromCopy(source, heads[i]) for i in range(2)],
            run_time=1.5,
        )
        self.play(TransformFromCopy(heads, output), run_time=1.3)
        self.play(FadeIn(label))
        self.wait(0.8)
        self.play(FadeOut(VGroup(head, source, heads, output, label)))

    def encoder(self):
        head = self.stage("11", "Transformer Encoder", "LayerNorm, MSA, MLP와 Residual을 순서대로 적용합니다.")
        stages = ("Input", "LayerNorm", "MSA", "+ Residual",
                  "LayerNorm", "MLP", "+ Residual")
        blocks = VGroup()
        for name in stages:
            rect = Rectangle(width=1.72, height=0.77,
                             color=YELLOW if "Residual" in name else BLUE_B)
            txt = Text(name, font_size=18).move_to(rect)
            blocks.add(VGroup(rect, txt))
        blocks.arrange(RIGHT, buff=0.16).move_to(DOWN * 0.13)
        note = self.ko("Residual: 입력을 연산 결과에 다시 더합니다.", 21)
        note.next_to(blocks, DOWN, buff=0.55)
        self.play(FadeIn(head))
        for i, block in enumerate(blocks):
            self.play(FadeIn(block, shift=RIGHT * 0.15), run_time=0.4)
            if i in (3, 6):
                marker = SurroundingRectangle(block, color=YELLOW, buff=0.09)
                self.play(Create(marker), run_time=0.23)
                self.play(FadeOut(marker), run_time=0.23)
        self.play(FadeIn(note))
        self.wait(0.9)
        self.play(FadeOut(VGroup(head, blocks, note)))

    def classification(self):
        head = self.stage("12", "Classification Head", "마지막 CLS Token을 읽어 클래스를 예측합니다.")
        sequence = self.matrix(5, highlight_first=True).move_to(LEFT * 3.7 + DOWN * 0.1)
        cls = SurroundingRectangle(sequence[0], color=YELLOW, buff=0.09)
        mlp_frame = Rectangle(width=2.15, height=0.9, color=YELLOW)
        mlp_text = Text("MLP Head", font_size=24).move_to(mlp_frame)
        mlp = VGroup(mlp_frame, mlp_text).move_to(RIGHT * 0.1 + DOWN * 0.1)
        output = self.ko("예측 클래스", 25, WHITE).move_to(RIGHT * 4.25 + DOWN * 0.1)
        arrow1 = Arrow(sequence[0].get_right() + RIGHT*0.14,
                       mlp.get_left() + LEFT*0.14, buff=0, color=GREY_B)
        arrow2 = Arrow(mlp.get_right() + RIGHT*0.14,
                       output.get_left() + LEFT*0.14, buff=0, color=GREY_B)
        self.play(FadeIn(head), FadeIn(sequence))
        self.play(Create(cls))
        self.play(Create(arrow1), FadeIn(mlp))
        self.play(Create(arrow2), FadeIn(output))
        self.wait(1.7)
