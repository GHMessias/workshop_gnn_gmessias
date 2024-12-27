# Workshop em Graph Neural Networks

Esse repositório é parte do material criado para o workshop em GNN destinado aos pesquisadores do MIDAS UFSCar.

## Sumário
1. Introdução
2. Modelos de Node Embedding
3. *Graph Neural Networks*
3.1 Node classification
3.2 Clustering
3.3 Link prediction
3.4 Domínios Heterogêneos e PPI/DPI

## 1. Introdução

### 1.1 Conceitos Fundamentais

Durante esse *workshop* nos preocuparemos em tarefas de aprendizado em grafos com atributos.

**Definição 1**: Um grafo com atributo é uma tripla ordenada $G = (V,E,X)$ formada por dois conjuntos $V$ e $E$ e uma matriz $X$. O conjunto $V$ é conhecido como conjunto dos vértices, enquanto $E$ é conhecido como o conjunto das arestas, formado por pares ordenados tais que $E = \{(u,v) | u,v \in V\}$. A matriz $X \in \mathbb{R_{m \times n}}$.

O conjunto de vértices de um grafo pode ser representado por uma matriz $A$ conhecida como matriz de adjacência, definida da seguinte forma:

$A_{i,j} = \begin{cases} 1 \text{, se } (i,j) \in E \\ 0 \text{, caso contrário}\end{cases}$

### 1.2 Transferência de Mensagem

O objetivo principal da transferência de mensagem é criar uma nova representação de cada vértice com base em sua vizinhança. Essa nova representação é utilizada para treinar as redes neurais. Tal mecanismo permite que vértices em um grafo compartilhem informações uns com os outros ao longo das suas arestas, criando novas representações locais.

A transferência de mensagem é descrita em três passos:

1. **Agregação**: 

* Cada vértice coleta informações de seus vizinhos diretos; 
* As mensagens coletadas podem incluir atributos dos vizinhos, informações dos vértices ou embeddings de etapas anteriores; e
* Um operador abeliano de agregação como soma, multiplicação, média, max, pooling, etc é utilizado para combinar essas mensagens.

2. **Atualização**

* Após a agregação, o estado do vértice é atualizado de acordo com as mensagens recebidas; e
* Uma função de atualização (geralmente um módulo linear) é utilizada para calcular o novo estado. Isso permite que o modelo mude as representações aprendidas ao longo da tarefa de aprendizado de máquina.

3. **Propagação**

* O processo de agregação e atualização é repetido por várias iterações, permitindo que a informação seja agregada em uma vizinhança de raio cada vez maior.

Assim, cada etapa pode ser descrita matematicamente da seguinte forma:

**Agregação**
$$\mathbf{m}_v^{(t)} = \mathrm{AGG}^{(t)}\left( \{ \mathbf{h}^ {(t - 1)}_u, e_{uv} | u \in \mathcal{N}(v)\} \right)$$

* $\mathbf{h}_u^{(t-1)}$ representa as características de $v$ na iteração anterior
* $e_{uv}$ é a informação do vértice que liga $u$ e $v$
* $\mathcal{N}(v)$ são os vizinhos de $v$ 

**Atualização**

$$\mathbf{h}_v^{(t)} = \mathrm{ATT}\left( \mathbf{m}_v^{(t)}, \mathbf{h}_v^{(t-1)}\right)$$

* $\mathbf{\hat{h}}_v^{(t)}$ representa as novas características de $v$ frente aos seus vizinhos e a si mesmo.

**Propagação**

$$\mathbf{h}_v^{(t)} = \mathrm{UPT}\left( \mathbf{\hat{h}}_v^{(t)} \right)$$

O *framework* completo pode ser escrito como

$$ \mathbf{h}_v^{(t)} = \mathrm{UPT} \left( \mathrm{ATT}\left(\mathrm{AGG}^{(t)}\left( \{ \mathbf{h}^ {(t - 1)}_u, e_{uv} | u \in \mathcal{N}(v)\} \right), \mathbf{h}^ {(t - 1)}_v  \right) \right)$$

### 1.3 Transferência de Mensagem - Exemplo

Vamos realizar um exemplo de transferência de mensagem com o agregador soma.

**Exemplo 1** - Sejam a matriz de adjacência $A$, a matriz de características $X$ e o operador linear $W_{3 \times 3} = [1]_{3 \times 3}$


$$X = \begin{bmatrix} 0.1 & 0.2 & 0.3 \\ 1.1 & 1.2 & 1.3 \\ 2.1 & 2.2 & 2.3 \end{bmatrix}, \quad A = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$


Vamos começar calculando a vizinhança de cada vértice:

$\mathcal{N}(0) = \{1\},\quad  \mathcal{N}(1) = \{0,2\}, \quad \mathcal{N}(2) = \{1\}$

utilizando o agregador soma, vamos calcular a nova representação do vértice 1. Somando os valores da vizinhança de 1 temos:

$m_1^{(1)} = x_0 + x_2  \rightarrow m_1^{(1)} = \begin{bmatrix}0.1 &  0.2 & 0.3 \end{bmatrix} + \begin{bmatrix}2.1 &  2.2 & 2.3 \end{bmatrix} = \begin{bmatrix}2.2 &  2.4 & 2.6 \end{bmatrix}$.

Assim, a nova representação do vértice 1 pode ser calculada a partir de $m_1^{(1)}$. Passando para a fase de atualização, vamos agregar os valores de $m_1^{(1)}$ com as características do vértice que estamos computando a transferência de mensagem, no caso, $v_1$. Dessa forma:

$$ \mathbf{\hat{h}_1^{(1)}} = m_1^{(1)} + x_1 \rightarrow \mathbf{\hat{h}_1^{(1)}} = \begin{bmatrix}3.3 &  3.6 & 3.9 \end{bmatrix}$$

Agora, basta que passemos para a fase de propagação, que consiste em propagar $\mathbf{\hat{h}_1^{(1)}}$ por uma camada linear de rede neural, cujos pesos são definidos por $W_{3 \times 3}$. Dessa forma:

$$\mathbf{h}_1^{(1)} = \mathbf{\hat{h}_1^{(1)}} \cdot W = \begin{bmatrix}10.8 & 10.8 & 10.8 \end{bmatrix}$$

Realizando as mesmas multiplicações para todos os vértices, temos:

$$\mathbf{h}_2^{(1)} = \begin{bmatrix}10.2 & 10.2 & 10.2 \end{bmatrix}$$

$$\mathbf{h}_3^{(1)} = \begin{bmatrix}4.2 & 4.2 & 4.2 \end{bmatrix}$$

Resumindo, o que fizemos foi calcular a soma de todos os vizinhos de cada vértice e passá-los por uma transformação linear. A matriz $W$ serve unicamente para que possamos atualizar esses pesos para as tarefas de aprendizado de máquina desejadas.

Podemos fazer as operações de Agregação, Atualização e Propagação unicamente por multiplicação de matrizes. Desse modo, conseguimos trazer todo *background* de operações de álgebra linear para as GNNs.

**Exemplo 2** - Considere as matrizes como no Exemplo 1.

Vamos aplicar a operação de transferência de mensagem a partir das multiplicações das matrizes $A$, $X$ e $W$ da seguinte forma: Seja $\hat{A} = A + I$

$$ H = \hat{A} XW$$

Substituindo os valores:

$$ H = \begin{bmatrix}  4.2 & 4.2 & 4.2 \\ 10.8 & 10.8 & 10.8 \\ 10.2 & 10.2 & 10.2\end{bmatrix}$$

As fases de agregação e atualização são dadas pela multiplicação de $A$ com $X$, soma de $A$ com $I$ (matriz identidade, que representa self loops) e multiplicação pelo módulo linear $W$. A fase de propagação é feita quando iteramos a agregação e a atualização. Em todos os modelos de GNNs uma função de ativação não linear $\varphi$ é utilizada entre cada propagação, porém, existem modelos onde isso não acontece.

Os principais modelos de GNNs são baseados em transferência de mensagem. Vejamos quais as diferenças entre cada modelo.

## 2. Modelos de GNN

O modelo *Graph Convolutional Networks* (GCN) proposto por Thomas Kipf e Max Welling [[1]](https://arxiv.org/abs/1609.02907) foi o principal para o começo do estudos em GNNs. A agregação utilizada pela GCN tem justificativas teóricas da área de convoluções em grafos, sendo que aqui não entraremos em detalhes. 

A função de agregação consiste em alterar o peso das arestas de acordo com o grau de cada vértice, por meio da matriz de grau $D$, a propagação é feita da seguinte forma:

$$ H^{(l+1)} = \varphi \left( \tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l+1)} \right)$$

$$H^{(0)} = X$$

