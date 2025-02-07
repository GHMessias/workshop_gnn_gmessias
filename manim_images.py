from manim import *

class image1(Scene):
    def construct(self):
        self.camera.background_color = "#00000000"
        G1 = Graph([0,1,2], [(0,1), (1,2)], layout='tree', root_vertex=1, vertex_config={'radius':0.6}).scale(0.5).to_corner(LEFT)
        v0 = MathTex('v_0', color = BLACK).move_to(G1.vertices[0].get_center()).scale(0.9)
        v1 = MathTex('v_1', color = BLACK).move_to(G1.vertices[1].get_center()).scale(0.9)
        v2 = MathTex('v_2', color = BLACK).move_to(G1.vertices[2].get_center()).scale(0.9)

        x0 = MathTex('x_0', color = WHITE, font_size = 60).next_to(G1.vertices[0], DOWN).scale(0.9)
        x1 = MathTex('x_1', color = WHITE, font_size = 60).next_to(G1.vertices[1], UP).scale(0.9)
        x2 = MathTex('x_2', color = WHITE, font_size = 60).next_to(G1.vertices[2], DOWN).scale(0.9)

        title = Text('Agregação').to_corner(UP)

        h0_calc = MathTex('\overline{\mathbf{h}}_0 = x_0 + x_1 + x_2').move_to(UP)
        h1_calc = MathTex('\overline{\mathbf{h}}_1 = x_1 + x_0' ).next_to(h0_calc, DOWN)
        h2_calc = MathTex('\\overline{\mathbf{h}}_2 = x_2 + x_2').next_to(h1_calc, DOWN)

        br = Brace(h0_calc, direction=DOWN).move_to(DOWN)

        H_line = MathTex('\overline{\mathbf{H}}^{(0)}').next_to(br, DOWN, buff = 0.4)

        text1 = Text('A fase de agregação envolve computar as características parciais de cada vértice', font_size=24).to_corner(DOWN)

        gg1 = Group()
        gg2 = Group()
        gg1.add(G1, v0, v1, v2, x0, x1, x2, title, )
        self.add(title)
        gg2.add(h0_calc, h1_calc, h2_calc, br, H_line, text1)
        self.add(gg1, gg2)

        # # Atualização
        # self.remove(title)
        # self.remove(gg2)

        # H = MathTex(r'\mathbf{H} = \varphi \left( \overline{\mathbf{H}} \cdot W \right)')

        # G2 = Graph([0,1,2], [(0,1), (1,2)], layout='tree', root_vertex=1, vertex_config={'radius':0.6}).scale(0.5).to_corner(LEFT)

        # self.add(H)




        