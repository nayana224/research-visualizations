"""One continuous ViT explanation, with English keywords and Korean narration.

Run from the repository root: manim scenes/vit/vit_full.py ViTFullPipeline
Visual tokens are schematic. Attention equations describe actual operations.
"""
from manim import (
    BLUE_B, BLUE_D, DOWN, FadeIn, FadeOut, GREEN_B, GREY_B,
    LEFT, ORANGE, RIGHT, Scene, Square, Rectangle, Text, UP, VGroup,
    WHITE, YELLOW, Create, SurroundingRectangle, Arrow, LaggedStart,
    MathTex,
)
# Manim loads scene files by path, so the repository root may not be on sys.path.
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_patch_path = Path(__file__).with_name("patch_embedding.py")
_spec = spec_from_file_location("_vit_patch_embedding", _patch_path)
_patch_module = module_from_spec(_spec)
_spec.loader.exec_module(_patch_module)
ViTPatchEmbedding = _patch_module.ViTPatchEmbedding

COLORS = (BLUE_D, GREEN_B, ORANGE)
KFONT = "Noto Sans CJK KR"


class ViTFullPipeline(ViTPatchEmbedding):
    """Reuse the existing 01 Patch Embedding choreography, then continue."""

    def construct(self):
        self.camera.background_color = "#10131C"
        title = Text("Vision Transformer | ViT", font_size=34)
        title.to_edge(UP, buff=0.28)
        self.add(title)
        self.show_partition()
        self.show_one_flatten()
        self.show_patch_matrix()
        self.show_projection()
        self.cls_and_position()
        self.attention()
        self.encoder()
        self.classification()

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
        head = self.stage("05", "CLS Token", "분류 정보를 모을 CLS Token을 맨 앞에 추가합니다.")
        patches = self.matrix(4).move_to(LEFT * 3.0 + DOWN * 0.15)
        cls = self.matrix(1, highlight_first=True).move_to(LEFT * 3.0 + UP * 1.42)
        combined = self.matrix(5, highlight_first=True).move_to(RIGHT * 3.0 + DOWN * 0.1)
        tags = VGroup(
            self.label("(4, 6) + CLS (1, 6)", patches),
            self.label("(5, 6)", combined),
        )
        self.play(FadeIn(head), FadeIn(patches), FadeIn(tags[0]))
        self.play(FadeIn(cls))
        self.play(LaggedStart(*[FadeIn(row, shift=RIGHT * 0.25) for row in combined],
                              lag_ratio=0.18), run_time=1.8)
        self.play(FadeIn(tags[1]))
        self.wait(0.6)
        self.play(FadeOut(VGroup(head, patches, cls, combined, tags)))

        head = self.stage("06", "Positional Embedding", "각 Token에 위치 정보를 원소별로 더합니다.")
        tokens = self.matrix(5, highlight_first=True).move_to(LEFT * 3.1 + DOWN * 0.1)
        positions = self.matrix(5, shades=(BLUE_B, GREEN_B)).move_to(RIGHT * 3.1 + DOWN * 0.1)
        plus = Text("+", font_size=42)
        plus.move_to(DOWN * 0.1)
        tags = VGroup(
            self.label("Tokens  (5, 6)", tokens),
            self.label("Position  (5, 6)", positions),
        )
        self.play(FadeIn(head), FadeIn(tokens), FadeIn(tags[0]))
        self.play(FadeIn(plus), FadeIn(positions), FadeIn(tags[1]))
        self.wait(1.2)
        self.play(FadeOut(VGroup(head, tokens, positions, plus, tags)))

        head = self.stage("07", "Transformer Input", "Token과 위치 정보를 합쳐도 Shape은 유지됩니다.")
        tokens = self.matrix(5, highlight_first=True)
        tokens.move_to(DOWN * 0.13)
        tag = self.label("Z0  (5, 6)", tokens)
        outline = SurroundingRectangle(tokens[0], color=YELLOW, buff=0.08)
        self.play(FadeIn(head), FadeIn(tokens), FadeIn(tag), Create(outline))
        self.wait(0.8)
        self.play(FadeOut(VGroup(head, tokens, tag, outline)))

    def attention(self):
        head = self.stage("08", "Multi-Head Self-Attention", "같은 Token에서 Q, K, V를 만들고 관계를 계산합니다.")
        x = self.matrix(5, highlight_first=True)
        x.move_to(LEFT * 4.5 + DOWN * 0.15)
        qs = VGroup()
        for idx, name in enumerate(("Q", "K", "V")):
            box = Rectangle(width=1.3, height=0.62, color=COLORS[idx])
            text = Text(name, font_size=29).move_to(box)
            qs.add(VGroup(box, text))
        qs.arrange(DOWN, buff=0.28).move_to(LEFT * 0.7 + DOWN * 0.1)
        arrow = Arrow(x.get_right()+RIGHT*0.13, qs.get_left()+LEFT*0.2, buff=0,
                      color=GREY_B, stroke_width=3)
        formula = MathTex(r"\\mathrm{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V")
        formula.scale(0.76).move_to(RIGHT * 3.35 + DOWN * 0.15)
        self.play(FadeIn(head), FadeIn(x), Create(arrow))
        self.play(LaggedStart(*[FadeIn(box) for box in qs], lag_ratio=0.25))
        self.play(FadeIn(formula))
        self.wait(1.2)
        self.play(FadeOut(VGroup(head, x, qs, arrow, formula)))

        head = self.stage("09", "Attention Heads", "두 Head가 서로 다른 관계를 학습하고 결과를 합칩니다.")
        left = VGroup(
            Rectangle(width=2.7, height=1.25, color=BLUE_B),
            Text("Head 1  (5, 3)", font_size=24),
        )
        left[1].move_to(left[0])
        right = VGroup(
            Rectangle(width=2.7, height=1.25, color=GREEN_B),
            Text("Head 2  (5, 3)", font_size=24),
        )
        right[1].move_to(right[0])
        left.move_to(LEFT * 3.0 + DOWN * 0.12)
        right.move_to(RIGHT * 0.2 + DOWN * 0.12)
        out = self.matrix(5).move_to(RIGHT * 4.45 + DOWN * 0.1)
        out_tag = self.label("Concat (5, 6)", out)
        self.play(FadeIn(head), FadeIn(left), FadeIn(right))
        self.play(FadeIn(out), FadeIn(out_tag))
        self.wait(1.0)
        self.play(FadeOut(VGroup(head, left, right, out, out_tag)))

    def encoder(self):
        head = self.stage("10", "Transformer Encoder", "LayerNorm, MSA, MLP와 Residual Connection을 사용합니다.")
        blocks = VGroup()
        names = ["Input Z", "LayerNorm", "MSA", "+ Residual", "LayerNorm", "MLP", "+ Residual"]
        for name in names:
            box = Rectangle(width=2.3, height=0.65, color=BLUE_B if name == "MSA" else GREY_B)
            label = Text(name, font_size=21).move_to(box)
            blocks.add(VGroup(box, label))
        blocks.arrange(RIGHT, buff=0.16)
        blocks.scale_to_fit_width(12.6)
        blocks.move_to(DOWN * 0.12)
        self.play(FadeIn(head))
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.15) for b in blocks],
                              lag_ratio=0.13), run_time=2.8)
        note = self.ko("Residual Connection은 입력을 연산 결과에 더합니다.", 22)
        note.next_to(blocks, DOWN, buff=0.6)
        self.play(FadeIn(note))
        self.wait(1.2)
        self.play(FadeOut(VGroup(head, blocks, note)))

    def classification(self):
        head = self.stage("11", "Classification Head", "최종 CLS Token으로 이미지의 클래스를 예측합니다.")
        seq = self.matrix(5, highlight_first=True).move_to(LEFT * 3.2 + DOWN * 0.08)
        cls_box = SurroundingRectangle(seq[0], color=YELLOW, buff=0.08)
        head_box = VGroup(
            Rectangle(width=2.25, height=0.95, color=YELLOW),
            Text("MLP Head", font_size=25),
        )
        head_box[1].move_to(head_box[0])
        head_box.move_to(RIGHT * 0.35 + DOWN * 0.1)
        prediction = self.ko("예측 클래스", 24, WHITE)
        prediction.move_to(RIGHT * 4.3 + DOWN * 0.1)
        arrow1 = Arrow(seq[0].get_right()+RIGHT*0.15, head_box.get_left()+LEFT*0.15,
                       buff=0, stroke_width=3, color=GREY_B)
        arrow2 = Arrow(head_box.get_right()+RIGHT*0.15, prediction.get_left()+LEFT*0.15,
                       buff=0, stroke_width=3, color=GREY_B)
        self.play(FadeIn(head), FadeIn(seq), Create(cls_box))
        self.play(Create(arrow1), FadeIn(head_box))
        self.play(Create(arrow2), FadeIn(prediction))
        footer = self.ko("Image -> Patch -> Token -> Encoder -> Classification", 20)
        footer.to_edge(DOWN, buff=0.42)
        self.play(FadeIn(footer))
        self.wait(2)
