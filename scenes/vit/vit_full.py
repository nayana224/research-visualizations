"""One continuous ViT explanation, with English keywords and Korean narration.

Run from the repository root: manim scenes/vit/vit_full.py ViTFullPipeline
Visual tokens are schematic. Attention equations describe actual operations.
"""
from manim import (
    BLUE_B, BLUE_D, DOWN, FadeIn, FadeOut, GREEN_B, GREY_B,
    LEFT, ORANGE, RIGHT, Scene, Square, Rectangle, Text, UP, VGroup,
    WHITE, YELLOW, Create, SurroundingRectangle, Arrow, LaggedStart, TransformFromCopy,
    MathTex, Circle, Line,
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
            if index != 3:  # The patch vector must remain onscreen during 03 -> 04.
                self.show_architecture_location(index)
            stage()
            # Some individual scenes intentionally leave their final frame on screen.
            # Remove everything except the persistent title before the next stage.
            if index < len(stages) - 1:
                # Keep the first patch vector between 03 and 04, otherwise clear.
                keep = self.focus_row if index == 2 else None
                leftovers = [item for item in self.mobjects if item is not title and item is not keep]
                if leftovers:
                    self.play(*[FadeOut(item) for item in leftovers], run_time=0.45)

    def show_architecture_location(self, stage_index):
        # Short paper-figure map before each chapter; avoid obstructing local math.
        names = ("Image", "Patch Embedding", "CLS + Position",
                 "Transformer Encoder x L", "MLP Head")
        active = (0, 1, 1, 1, 2, 3, 3, 4)[stage_index]
        cards = VGroup()
        for i, name in enumerate(names):
            frame = Rectangle(width=2.16, height=0.64,
                              color=YELLOW if i == active else GREY_B,
                              stroke_width=2 if i == active else 1)
            label = Text(name, font_size=17,
                         color=YELLOW if i == active else WHITE).move_to(frame)
            cards.add(VGroup(frame, label))
        cards.arrange(RIGHT, buff=0.24).move_to(DOWN * 0.15)
        arrows = VGroup()
        for i in range(4):
            arrows.add(Arrow(cards[i].get_right(), cards[i+1].get_left(),
                             buff=0.035, color=GREY_B, stroke_width=2))
        notice = self.ko("전체 ViT 구조에서 현재 설명하는 위치", 22)
        notice.next_to(cards, UP, buff=0.55)
        self.play(FadeIn(cards), FadeIn(notice), run_time=0.48)
        self.play(*[Create(arrow) for arrow in arrows], run_time=0.45)
        self.wait(0.45)
        self.play(FadeOut(VGroup(cards, arrows, notice)), run_time=0.38)

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
        # A token is projected through three distinct learned matrices.
        header = self.stage("08", "Q / K / V Projection",
                            "같은 Token을 세 종류의 Weight Matrix로 변환합니다.")
        source = self.matrix(5, highlight_first=True).move_to(LEFT * 4.3 + DOWN * 0.12)
        source_tag = self.label("Z0  (5, 6)", source)
        destinations = VGroup()
        projections = VGroup()
        for i, (symbol, color) in enumerate((("Q", BLUE_D), ("K", GREEN_B), ("V", ORANGE))):
            weight = VGroup(
                Rectangle(width=1.35, height=0.52, color=color),
                Text("W" + symbol, font_size=20, color=color),
            )
            weight[1].move_to(weight[0])
            dest = self.matrix(5, shades=(color,)).scale(0.52)
            group = VGroup(dest, Text(symbol + "  (5, 6)", font_size=19, color=color))
            group[1].next_to(dest, DOWN, buff=0.15)
            group.move_to(RIGHT * (0.2 + 2.4 * i) + DOWN * 0.45)
            weight.move_to(group.get_center() + UP * 1.55)
            projections.add(weight)
            destinations.add(group)
        self.play(FadeIn(header), FadeIn(source), FadeIn(source_tag))
        for i in range(3):
            self.play(FadeIn(projections[i]), run_time=0.4)
            self.play(
                TransformFromCopy(source, destinations[i][0]), run_time=0.75
            )
            self.play(FadeIn(destinations[i][1]), run_time=0.22)
        self.wait(0.75)
        self.play(FadeOut(VGroup(header, source, source_tag, projections, destinations)))

        # Explain a single query's five similarities before displaying softmax.
        header = self.stage("09", "Attention Scores",
                            "한 Query를 모든 Key와 비교하여 관련도를 구합니다.")
        query = self.matrix(1, shades=(BLUE_D,)).move_to(LEFT * 3.65 + DOWN * 0.1)
        keys = self.matrix(5, shades=(GREEN_B,)).move_to(DOWN * 0.1)
        scores = VGroup()
        for i in range(5):
            bar = Rectangle(width=0.30 + 0.18 * (i % 3), height=0.33,
                            stroke_color=ORANGE, stroke_width=1)
            bar.set_fill(ORANGE, opacity=0.25 + 0.13 * (i % 3))
            scores.add(bar)
        scores.arrange(DOWN, buff=0.22).move_to(RIGHT * 3.85 + DOWN * 0.1)
        query_tag = self.label("Query  (1, 6)", query)
        keys_tag = self.label("Keys  (5, 6)", keys)
        scores_tag = self.label("QKᵀ  (1, 5)", scores)
        self.play(FadeIn(header), FadeIn(query), FadeIn(keys),
                  FadeIn(query_tag), FadeIn(keys_tag))
        for i in range(5):
            q_outline = SurroundingRectangle(query, color=YELLOW, buff=0.06)
            k_outline = SurroundingRectangle(keys[i], color=YELLOW, buff=0.055)
            self.play(Create(q_outline), Create(k_outline), run_time=0.25)
            self.play(TransformFromCopy(keys[i], scores[i]), run_time=0.42)
            self.play(FadeOut(q_outline), FadeOut(k_outline), run_time=0.15)
        self.play(FadeIn(scores_tag))
        self.wait(0.5)
        self.play(FadeOut(VGroup(header, query, keys, scores, query_tag, keys_tag, scores_tag)))

        # Dot product close-up: coordinatewise multiply and summation -> scalar.
        head = self.stage("09", "Dot Product", "Query와 Key의 대응 원소를 곱한 뒤 모두 더합니다.")
        qvals = VGroup(*[Text(f"q{i}", font_size=24, color=BLUE_B) for i in range(1, 4)])
        kvals = VGroup(*[Text(f"k{i}", font_size=24, color=GREEN_B) for i in range(1, 4)])
        qvals.arrange(RIGHT, buff=0.48).move_to(LEFT * 2.6 + UP * 0.5)
        kvals.arrange(RIGHT, buff=0.48).move_to(LEFT * 2.6 + DOWN * 0.55)
        products = VGroup(*[Text(f"q{i}k{i}", font_size=24, color=YELLOW)
                            for i in range(1, 4)])
        products.arrange(RIGHT, buff=0.18).move_to(RIGHT * 2.85 + UP * 0.18)
        plus = Text("+", font_size=29).move_to(RIGHT * 2.85 + DOWN * 0.48)
        scalar = Text("score = sum(qi ki)", font_size=23)
        scalar.move_to(DOWN * 1.55)
        self.play(FadeIn(head), FadeIn(qvals), FadeIn(kvals))
        for i in range(3):
            self.play(TransformFromCopy(VGroup(qvals[i], kvals[i]), products[i]),
                      run_time=0.62)
        self.play(FadeIn(plus), FadeIn(scalar))
        note = self.ko("3차원 예시입니다. 실제 차원은 Head Dimension에 따라 달라집니다.", 17)
        note.to_edge(DOWN, buff=0.38)
        self.play(FadeIn(note))
        self.wait(0.75)
        self.play(FadeOut(VGroup(head, qvals, kvals, products, plus, scalar, note)))

        # Full paper-style attention map after the detailed single-query example.
        header = self.stage("09", "Attention Matrix", "모든 Query와 Key의 비교 결과는 5×5 행렬입니다.")
        qnames = VGroup(*[Text(f"Q{i}", font_size=17, color=BLUE_B) for i in range(5)])
        knames = VGroup(*[Text(f"K{i}", font_size=17, color=GREEN_B) for i in range(5)])
        cells = VGroup()
        for r in range(5):
            row = VGroup()
            for c in range(5):
                square = Square(side_length=0.53, stroke_color=WHITE, stroke_width=0.8)
                square.set_fill(BLUE_D, opacity=0.18 + 0.12 * ((r * 3 + c * 2) % 5))
                row.add(square)
            row.arrange(RIGHT, buff=0.05)
            cells.add(row)
        cells.arrange(DOWN, buff=0.05).move_to(DOWN * 0.08)
        for r in range(5):
            qnames[r].next_to(cells[r], LEFT, buff=0.22)
            knames[r].next_to(cells[0][r], UP, buff=0.25)
        shape = Text("QK^T : (5, 5)", font_size=23)
        shape.next_to(cells, DOWN, buff=0.35)
        note = self.ko("각 행은 Query 하나의 비교 결과입니다. 색상은 예시입니다.", 19)
        note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(header), FadeIn(qnames), FadeIn(knames))
        for r in range(5):
            self.play(
                LaggedStart(*[FadeIn(cells[r][c]) for c in range(5)],
                            lag_ratio=0.09),
                run_time=0.55,
            )
        self.play(FadeIn(shape), FadeIn(note))
        first = SurroundingRectangle(cells[0], color=YELLOW, buff=0.075)
        self.play(Create(first))
        self.wait(0.8)
        self.play(FadeOut(VGroup(header, cells, qnames, knames, shape, note, first)))


        header = self.stage("10", "Softmax & Weighted Sum",
                            "관련도를 정규화해 Value의 가중합을 만듭니다.")
        scores = VGroup(*[
            Rectangle(width=0.28 + 0.16 * (i % 3), height=0.30,
                      stroke_color=BLUE_B, stroke_width=1).set_fill(BLUE_B, opacity=0.65)
            for i in range(5)
        ]).arrange(DOWN, buff=0.19).move_to(LEFT * 4.4 + DOWN * 0.12)
        softmax = VGroup(Rectangle(width=2.0, height=0.85, color=YELLOW),
                         Text("Softmax", font_size=24))
        softmax[1].move_to(softmax[0])
        softmax.move_to(LEFT * 1.75 + DOWN * 0.12)
        weights = VGroup(*[
            Rectangle(width=w, height=0.28, stroke_width=0,
                      fill_color=GREEN_B, fill_opacity=0.85)
            for w in (0.8, 0.35, 0.6, 0.23, 0.5)
        ]).arrange(DOWN, buff=0.2, aligned_edge=LEFT).move_to(RIGHT * 1.05 + DOWN * 0.12)
        result = self.matrix(1, shades=(ORANGE, GREEN_B, BLUE_B)).move_to(
            RIGHT * 4.35 + DOWN * 0.12
        )
        labels = VGroup(self.label("Scores", scores),
                        self.label("Weights", weights),
                        self.label("Output (1, 6)", result))
        arrow1 = Arrow(scores.get_right() + RIGHT * 0.1, softmax.get_left() + LEFT * 0.12,
                       buff=0.05, stroke_width=2, color=GREY_B)
        arrow2 = Arrow(softmax.get_right() + RIGHT * 0.1, weights.get_left() + LEFT * 0.12,
                       buff=0.05, stroke_width=2, color=GREY_B)
        self.play(FadeIn(header), FadeIn(scores), FadeIn(labels[0]))
        self.play(FadeIn(softmax), Create(arrow1))
        self.play(Create(arrow2), LaggedStart(*[FadeIn(w) for w in weights], lag_ratio=0.12))
        self.play(FadeIn(labels[1]))
        self.play(TransformFromCopy(weights, result), FadeIn(labels[2]), run_time=1.2)
        note = self.ko("가중치는 설명용 예시이며 실제 계산값이 아닙니다.", 17)
        note.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(note))
        self.wait(0.7)
        self.play(FadeOut(VGroup(header, scores, weights, softmax, labels, result,
                                 arrow1, arrow2, note)))

        header = self.stage("11", "Multi-Head Attention",
                            "서로 다른 Head의 출력을 Concat하고 Linear로 결합합니다.")
        input_matrix = self.matrix(5, highlight_first=True).move_to(LEFT * 4.85 + DOWN * 0.12)
        heads = VGroup()
        for i, color in enumerate((BLUE_B, GREEN_B)):
            block = VGroup(Rectangle(width=2.0, height=0.65, color=color),
                           Text(f"Head {i+1}  (5, 3)", font_size=18))
            block[1].move_to(block[0])
            heads.add(block)
        heads.arrange(DOWN, buff=0.48).move_to(LEFT * 1.2 + DOWN * 0.12)
        concat = VGroup(Rectangle(width=1.6, height=0.85, color=YELLOW),
                        Text("Concat", font_size=22))
        concat[1].move_to(concat[0])
        concat.move_to(RIGHT * 1.55 + DOWN * 0.12)
        output = self.matrix(5).scale(0.82).move_to(RIGHT * 4.5 + DOWN * 0.12)
        cap = self.label("Linear (5, 6)", output)
        self.play(FadeIn(header), FadeIn(input_matrix))
        for h in heads:
            self.play(TransformFromCopy(input_matrix, h), run_time=0.85)
        # Materialize feature-wise concatenation: (5,3) + (5,3) -> (5,6).
        left_features = self.matrix(5, ncols=3, shades=(BLUE_B,)).scale(0.65)
        right_features = self.matrix(5, ncols=3, shades=(GREEN_B,)).scale(0.65)
        joined = VGroup(left_features, right_features).arrange(RIGHT, buff=0.04)
        joined.move_to(concat.get_center() + UP * 1.30)
        joined_tag = Text("axis=-1: (5,3) + (5,3) = (5,6)", font_size=17)
        joined_tag.next_to(joined, UP, buff=0.12)
        self.play(TransformFromCopy(heads[0], left_features),
                  TransformFromCopy(heads[1], right_features), run_time=1.1)
        self.play(FadeIn(joined_tag))
        self.play(TransformFromCopy(joined, concat), run_time=0.8)
        self.play(TransformFromCopy(concat, output), FadeIn(cap), run_time=1.0)
        self.play(FadeOut(joined), FadeOut(joined_tag))
        self.wait(0.85)
        self.play(FadeOut(VGroup(header, input_matrix, heads, concat, output, cap)))

    def encoder(self):
        # A detail view of the ViT pre-LN encoder, with two residual paths.
        # Show a physical bypass path and a plus node for each sublayer.
        for number, title, description, operation, color in (
            ("12", "Encoder: Attention Block",
             "입력을 LayerNorm과 MSA에 통과시키고 원래 입력을 더합니다.",
             "MSA", BLUE_B),
            ("13", "Encoder: Feed Forward",
             "다시 LayerNorm과 MLP를 통과한 결과에 입력을 더합니다.",
             "MLP", GREEN_B),
        ):
            header = self.stage(number, title, description)
            source = self.matrix(5, highlight_first=True).scale(0.74)
            source.move_to(LEFT * 5.10 + DOWN * 0.58)
            norm = VGroup(Rectangle(width=1.48, height=0.78, color=color),
                          Text("LayerNorm", font_size=18))
            norm[1].move_to(norm[0])
            norm.move_to(LEFT * 2.25 + DOWN * 0.58)
            block = VGroup(Rectangle(width=1.32, height=0.78, color=color),
                           Text(operation, font_size=22))
            block[1].move_to(block[0])
            block.move_to(RIGHT * 0.10 + DOWN * 0.58)
            add = VGroup(Circle(radius=0.30, color=YELLOW),
                         Text("+", font_size=27, color=YELLOW))
            add[1].move_to(add[0])
            add.move_to(RIGHT * 2.22 + DOWN * 0.58)
            result = self.matrix(5, highlight_first=True).scale(0.74)
            result.move_to(RIGHT * 5.18 + DOWN * 0.58)
            connections = VGroup(
                Arrow(source.get_right(), norm.get_left(), buff=0.11, color=GREY_B),
                Arrow(norm.get_right(), block.get_left(), buff=0.11, color=GREY_B),
                Arrow(block.get_right(), add.get_left(), buff=0.11, color=GREY_B),
                Arrow(add.get_right(), result.get_left(), buff=0.11, color=GREY_B),
            )
            # Route above the blocks but below the title and Korean subtitle.
            route = VGroup(
                Line(source.get_top(), source.get_top() + UP * 1.12, color=YELLOW),
                Line(source.get_top() + UP * 1.12,
                     add.get_top() + UP * 0.60, color=YELLOW),
                Arrow(add.get_top() + UP * 0.60, add.get_top(),
                      buff=0.04, color=YELLOW, stroke_width=2),
            )
            inlabel = self.label("Input (5, 6)", source, buff=0.18)
            outlabel = self.label("Output (5, 6)", result, buff=0.18)
            bypass_text = Text("Skip / Residual", font_size=18, color=YELLOW)
            bypass_text.move_to(UP * 1.10)
            self.play(FadeIn(header), FadeIn(source), FadeIn(inlabel))
            self.play(Create(connections[0]), FadeIn(norm))
            self.play(Create(connections[1]), FadeIn(block))
            self.play(Create(connections[2]), FadeIn(add))
            self.play(Create(route), FadeIn(bypass_text), run_time=1.35)
            self.play(Create(connections[3]),
                      TransformFromCopy(source, result),
                      FadeIn(outlabel), run_time=1.1)
            self.wait(0.85)
            self.play(FadeOut(VGroup(
                header, source, norm, block, add, result, connections, route,
                inlabel, outlabel, bypass_text,
            )))

    def classification(self):
        head = self.stage("14", "Classification Head", "마지막 CLS Token을 읽어 클래스를 예측합니다.")
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
