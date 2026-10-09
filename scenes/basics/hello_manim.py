from manim import BLUE, YELLOW, Circle, Create, FadeOut, Scene, Square, Transform


class HelloManim(Scene):
    def construct(self):
        square = Square(color=BLUE)
        circle = Circle(color=YELLOW)

        self.play(Create(square))
        self.play(Transform(square, circle))
        self.wait(0.5)
        self.play(FadeOut(square))
