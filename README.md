# Árvore de Decisão com Classes 

---

Integrante: João Gabriel Machado  
Matéria: Engenharia de Software  

---

## Diagrama de Classes:

![Minha imagem](https://github.com/jgabrielsg/software-engineering-decision-tree/blob/main/class_diagram.jpg)
 

## Introdução

No projeto, foi-nos pedido criar uma modelagem para uma árvore de decisão usando certos padrões de projeto. Conforme as restrições, não há implementação de algoritmos matemáticos reais (como Gini Impurity ou Entropy) e todo o comportamento é simulado utilizando a biblioteca padrão `random` do Python e `prints` para demonstrar o fluxo de execução e a troca de mensagens entre os objetos.

## Ideias -> Padrões de Projeto

A solução foi estruturada para atender aos quatro padrões obrigatórios solicitados:

### Composite Pattern
Utilizado para representar a estrutura hierárquica da árvore, permitindo tratar nós individuais e composições de nós de maneira uniforme.
* `Node` (Abstrato): Define a interface comum.
* `DecisionNode`: Nó composto que contém referências para filhos (`left` e `right`).
* `LeafNode`: Nó folha que contém o valor final (predição).

### State Pattern
Utilizado para controlar o ciclo de vida da construção da árvore dentro da classe `TreeBuilder`. O comportamento do builder muda dependendo do estado em que ele se encontra.
* `TreeBuilder` (Contexto): Mantém a referência para o estado atual.
* `TreeState`: (Abstrato): Define a interface comum.
* `SplittingState`: Estado responsável por buscar pelo melhor corte e criar novos nós de decisão (simulado).
* `StoppingState`: Estado responsável por verificar critérios de parada (profundidade máxima, pureza, etc.) e decidir se cria uma folha ou continua dividindo. 

Os estados são trocados entre sí até o fim da construção da árvore através da classe TreeBuilder, que possui uma função para troca de estado e para execução das funções de `execute` das classes `TreeState`.

### Visitor Pattern
Utilizado para separar algoritmos da estrutura de dados dos nós. Isso permite adicionar novas operações à árvore sem modificar as classes dos nós (além de uma função simples de `accept`).
* `NodeVisitor` (Abstrato): Define a interface comum.
* `GetDepthVisitor`: Percorre a árvore para calcular a profundidade máxima.
* `CountLeavesVisitor`: Percorre a árvore para contar quantas folhas existem.

### Iterator Pattern
Utilizado para navegar pela estrutura da árvore, permitindo percorrer os nós sequencialmente, sem ter que lidar com a complexidade da iteração dentro dela.
* `NodeVisitor` (Abstrato): Define a interface comum.
* `InOrderIterator`: Percorre a árvore na ordem: Esquerda -> Nó Atual -> Direita.
* `PostOrderIterator`: Percorre a árvore na ordem: Esquerda -> Direita -> Nó Atual.

---

## Arquivos

1.  `tree_design.py`: Contém todas as classes, interfaces e a lógica estrutural dos Design Patterns.
2.  `tree_demo.py`: Script de execução que instancia o `TreeBuilder`, treina a árvore com dados mockados, executa os Visitors, os Iterators e simula uma Poda. 

## Sobre a Poda
Embora a poda pudesse ser um estado, preferi por implementar uma classe `TreePruner` separada que atua sobre a árvore já construída. Ela simula a verificação de erro em um conjunto de validação e substitui `DecisionNodes` por `LeafNodes` caso o erro diminua (tudo simulado). Usando a classe `TreeState` como pai, não conseguia manter uma linguagem boa comum entre a classe de poda e as classes de divisão e parada, então deixei-a separada.
