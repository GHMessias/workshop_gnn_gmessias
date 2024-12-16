# Workshop em Graph Neural Networks

Esse repositório é parte do material criado para o workshop em GNN destinado aos alunos do MIDAS UFSCar.

## Sumário
1. Introdução
2. Modelos de Node Embedding
3. *Graph Neural Networks*
3.1 Node classification
3.2 Clustering
3.3 Link prediction
3.4 Domínios Heterogêneos e PPI/DPI

## 1. Introdução

Durante esse *workshop* nos preocuparemos em tarefas de aprendizado em grafos.

**Definição 1**: Um Grafo é uma tripla ordenada $G = (V,E,X)$ formada por dois conjuntos $V$ e $E$ e uma matriz $X$. O conjunto $V$ é conhecido como conjunto dos vértices, enquanto $E$ é conhecido como o conjunto das arestas, formado por pares ordenados tais que $E = \{(u,v) | u,v \in V\}$. A matriz $X \in \mathbb{R_{m \times n}}$  