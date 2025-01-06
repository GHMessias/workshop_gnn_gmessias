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

### 2.1 Graph Convolutional Network

O modelo *Graph Convolutional Networks* (GCN) proposto por Thomas Kipf e Max Welling [[1]](https://arxiv.org/abs/1609.02907) foi o principal para o começo do estudos em GNNs. A agregação utilizada pela GCN tem justificativas teóricas da área de convoluções em grafos, sendo que aqui não entraremos em detalhes. 

A função de agregação consiste em alterar o peso das arestas de acordo com o grau de cada vértice, por meio da matriz de grau $D$, a propagação é feita da seguinte forma:

$$ H^{(l+1)} = \varphi \left( \tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l+1)} \right),$$

$$H^{(0)} = X,$$

em que $\tilde{D}$ é a matriz de grau calculada em cima de $A + I$.

Essa forma de propagação pondera a quantidade de vizinhos em um vértice, fazendo com que vértices mais importantes da rede propaguem mais informação. Podemos representar a propagação em cada vértice $v$ como:

$$h_v^{(t)} = \sigma\left(\sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{d_v d_u}} h_u^{(t-1)} W^{(t)}\right).$$

### 2.3 Graph Attention Network

O modelo Graph Attention Network (GAT) faz a agregação e a atualização usando redes neurais tradicionais. Cada interação entre vizinhos é calculada por um módulo linear em comum, com uma única saída. A saída (conhecida como coeficiente de atenção) é usada para refletir a influência desse par de vértices em sua vizinhança direta. Sejam $W$ uma matriz de pesos compartilhada, $||$ a operação de concatenação entre vetores e $a$ a matriz de peso do módulo linear, o coeficiente de atenção $e_{uv}$ entre os vetores $u$ e $v$ é dada por

$$ e_{uv} = \textrm{LeakyReLU}(\underbrace{a^T \cdot [Wh_u, || Wh_v])}_{\text{Linear Module}}, $$

em que a normalização de cada valor $e_{uv}$ é feita pela função $\textrm{softmax}$

$$ \alpha_{uv} = \frac{\exp{(e_{uv})}}{\sum_{k \in \mathcal{N(v)}} \exp{(e_{vk})}}.$$

Dessa forma, a propagação do modelo GAT é dado por

$$h_v^{(t)} = \sigma \left( \sum_{u \in \mathcal{N}(v) \cup \{v\}} \alpha_{uv} \cdot W^{(t)}h_u^{(t-1)} \right)$$

### 2.4 Graph Autoencoder e Variational Graph Autoencoder

Graph Autoencoders (GAEs) e Variational Graph Autoencoders (VGAEs) são modelos de aprendizado profundo usados para representar grafos em espaços latentes (de menor dimensão). Eles combinam técnicas de redes neurais e aprendizado em grafos, permitindo que tarefas como predição de arestas, aprendizado de representações de nós e geração de grafos sejam realizadas de maneira eficiente

Os modelos GAE são compostos por um *Encoder* e um *Decoder*, definidos da seguinte forma:

$$ \overline{A} = \textrm{Decoder} \left( \textrm{Encoder} (X,A) \right)$$

Tradicionalmente, usa-se uma GNN para fazer o mapeamento do grafo em um espaço vetorial. A reconstrução da entrada é feita a partir do produto do embedding $Z = \textrm{Encoder}(X,A)$ por sua transposta. Assim:

$$ Z = \textrm{Encoder}(X,A)$$

$$\hat{A} = Z\cdot Z^T$$

Os modelos VGAE estendem os GAE utilizando conceitos probabilísticos. A principal diferença é que os modelos VGAE aprendem distribuições probabilísticas no espaço latente, em vez de pontos determinísticos.

O *Encoder* aprende uma distribuição de probabilidade $\mathsf{N}(\mu, \sigma^2)$ para cada vértice, assim:

$$ q(Z|X,A) = \prod_{i=1}^{|V|} \mathsf{N}(z_i| \mu_i, \text{diag}(\sigma^2)), $$

em que $\mu$ e $\sigma^2$ representam a média e o desvio padrão. Dessa forma, o modelo aprende os valores de $\mu$ e $\sigma$.

Contudo, não é possível aplicar o aprendizado via *backpropagation* em valores de distribuição, para isso, cada exemplo é construído com o truque de reparametrização:

$$\mathbf{z}_i = \mu_i + \sigma_i \odot \epsilon, \quad \epsilon \sim \mathtt{N}(0, I)$$

Dessa forma, o decoder pode ser feito de maneira similar ao GAE. 

A função de perda em um VGAE leva em consideração 

$$ \mathcal{L} = \mathbb{E}_{q(\mathbf{Z} \mid \mathbf{A}, \mathbf{X})} \left[ \log p(\mathbf{A} \mid \mathbf{Z}) \right] - \text{KL}\left(q(\mathbf{Z} \mid \mathbf{A}, \mathbf{X}) \parallel p(\mathbf{Z})\right) $$

#### 2.4.1 Modelos Multidecoder

Algumas modelos GAE utilizam a *loss* calculada com a matriz de adjacência e com a matriz de características simultaneamente. Hao e Zhu (2022) [[2]](https://link.springer.com/article/10.1007/s10489-022-03381-y) propuseram um modelo profundo que calcula a loss das características reconstruídas por um AutoEncoder padrão juntamente com a loss de reconstrução da matriz de adjacência. O objetivo dessa abordagem é utilizar as features, fazendo a representação latente mais representativa das características.

$$ L_{re} =  \frac{1}{N} || X - \tilde{X} ||_2^2 $$

$$ L_{g\_re} = \textrm{Cross\_Entropy\_loss}(A, \tilde{A}) $$

$$ L = \alpha L_{re} + \beta L_{g\_re} $$

### 2.5 Modelos Lineares

Modelos lineares de GNN foram propostos em [[3]](http://proceedings.mlr.press/v97/wu19e/wu19e.pdf). Os modelos lineares consistem em não utilizar uma função não linear a cada layer, ou seja, considerando uma GCN onde o output é dado por $H^{(l+1)} = \varphi(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)})$, os modelos lineares ignoram a função $\varphi$, fazendo com que o output seja simplesmente uma potência da matriz de adjacência em um único módulo linear. Seja $S = \tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}$:

$$ Z = S^k X W$$

# 3. Aplicações de GNNs

## 3.1 DPI/PPI

## 3.2 Graph Clustering

## 3.3 Link prediction

## 3.4 Graph Drawing

## 3.5 OCC / Fake News Detection