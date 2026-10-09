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
                # Keep the first patch vector between 03 and 04, otherwise clear.
                keep = self.focus_row if index == 2 else None
                leftovers = [item for item in self.mobjects if item is not title and item is not keep]
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
        header = self.stage("01", "Patch Partition", "RGB 이미지를 채널별로 작은 Patch로 나눕니다.")
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
        header = self.stage("02", "Flatten", "한 Patch의 R, G, B 채널을 한 줄의 벡터로 펼칩니다.")
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
        header = self.stage("03", "Patch Matrix", "네 개 Patch를 각각 12차원 벡터로 펼칩니다.")
        patches = VGroup()
        self.patch_rows = VGroup()
        for i in range(4):
            patch = self.patch_stack(size=0.19)
            patch.move_to(LEFT * 3.4 + UP * (1.5 - i) * 0.92 + DOWN * 0.15)
            patches.add(patch)
            row = self.vector(12)
            row.move_to(RIGHT * 1.5 + UP * (1.5 - i) * 0.92 + DOWN * 0.15)
            self.patch_rows.add(row)
        caption = Text("Patch Matrix  (4, 12)", font_size=21)
        caption.next_to(self.patch_rows, DOWN, buff=0.30)
        self.play(FadeIn(header), FadeIn(patches))
        for i in range(4):
            self.play(TransformFromCopy(patches[i], self.patch_rows[i]), run_time=0.60)
        self.play(FadeIn(caption))
        self.wait(0.40)

        # Keep the actual first row in the scene so 04 can move that same object.
        self.focus_row = self.patch_rows[0]
        focus = SurroundingRectangle(self.focus_row, color=YELLOW, buff=0.07)
        note = self.ko("첫 번째 Patch 벡터 x를 다음 단계에서 살펴봅니다.", 19)
        note.next_to(caption, DOWN, buff=0.15)
        self.play(Create(focus), FadeIn(note))
        self.wait(0.60)
        self.play(FadeOut(VGroup(header, patches, caption, note, focus)))
        remaining = [m for m in self.patch_rows[1:] if m in self.mobjects]
        if remaining:
            self.play(*[FadeOut(m) for m in remaining])
        # Do not fade out self.focus_row.

    def show_projection(self):
        header = self.stage("04", "Linear Projection", "입력 벡터 x와 학습 가능한 Weight Matrix W를 곱합니다.")
        x = self.focus_row

        weights = VGroup()
        for r in range(12):
            row = VGroup()
            for c in range(6):
                cell = Rectangle(width=0.21, height=0.17,
                                 stroke_color=WHITE, stroke_width=0.55)
                cell.set_fill((BLUE_B, GREEN_B, ORANGE)[(r+c)%3], opacity=0.85)
                row.add(cell)
            row.arrange(RIGHT, buff=0.018)
            weights.add(row)
        weights.arrange(DOWN, buff=0.016).move_to(LEFT * 0.10 + DOWN * 0.10)

        y = VGroup()
        for c in range(6):
            cell = Rectangle(width=0.27, height=0.40,
                             stroke_color=WHITE, stroke_width=1)
            cell.set_fill((BLUE_B, GREEN_B, ORANGE)[c%3], opacity=0.86)
            y.add(cell)
        y.arrange(RIGHT, buff=0.03).move_to(RIGHT * 4.05 + DOWN * 0.12)
        x_label = Text("x  (1, 12)", font_size=20)
        w_label = Text("W  (12, 6)", font_size=20).next_to(weights, DOWN, buff=0.30)
        y_label = Text("y  (1, 6)", font_size=20).next_to(y, DOWN, buff=0.34)
        matmul = Text("@", font_size=36).move_to(LEFT * 2.18)
        equal = Text("=", font_size=36).move_to(RIGHT * 2.22)
        hint = self.ko("W의 열 하나가 출력 벡터의 원소 하나를 만듭니다.", 19)
        hint.to_edge(DOWN, buff=0.43)

        self.play(FadeIn(header), x.animate.move_to(LEFT * 4.1 + DOWN * 0.12),
                  run_time=1.0)
        x_label.next_to(x, DOWN, buff=0.34)
        self.play(FadeIn(x_label), FadeIn(weights), FadeIn(w_label), FadeIn(matmul))
        self.play(FadeIn(equal), FadeIn(hint))
        for c in range(6):
            column = VGroup(*[weights[r][c] for r in range(12)])
            highlight = SurroundingRectangle(column, color=YELLOW, buff=0.05)
            self.play(Create(highlight), run_time=0.16)
            self.play(TransformFromCopy(VGroup(x, column), y[c]), run_time=0.35)
            self.play(FadeOut(highlight), run_time=0.12)
        self.play(FadeIn(y_label))
        self.wait(0.7)
        self.play(FadeOut(VGroup(header, x, weights, y, x_label, w_label,
                                 y_label, matmul, equal, hint)))

        header = self.stage("04", "Shared Linear Projection",
                            "같은 Weight Matrix를 네 개의 Patch에 모두 적용합니다.")
        inputs, outputs = VGroup(), VGroup()
        for i in range(4):
            v = self.vector(12)
            v.move_to(LEFT * 3.8 + UP * (1.5-i)*0.64 + DOWN*0.10)
            inputs.add(v)
            out = VGroup()
            for j in range(6):
                unit = Rectangle(width=0.26, height=0.37,
                                 stroke_color=WHITE, stroke_width=1)
                unit.set_fill((BLUE_B, GREEN_B, ORANGE)[(i+j)%3], opacity=0.86)
                out.add(unit)
            out.arrange(RIGHT, buff=0.03)
            out.move_to(RIGHT*3.8 + UP*(1.5-i)*0.64 + DOWN*0.10)
            outputs.add(out)
        frame = Rectangle(width=2.0, height=1.3, color=YELLOW)
        name = Text("Linear\nW (12, 6)", font_size=22, color=YELLOW)
        name.move_to(frame)
        linear = VGroup(frame, name).move_to(DOWN*0.10)
        left_label = Text("(4, 12)", font_size=21).next_to(inputs, DOWN, buff=0.35)
        right_label = Text("(4, 6)", font_size=21).next_to(outputs, DOWN, buff=0.35)
        self.play(FadeIn(header), FadeIn(inputs), FadeIn(left_label), FadeIn(linear))
        for i in range(4):
            self.play(TransformFromCopy(inputs[i], outputs[i]), run_time=0.52)
        self.play(FadeIn(right_label))
        self.wait(0.8)
        self.play(FadeOut(VGroup(header, inputs, outputs, left_label,
                                 right_label, linear)))

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
