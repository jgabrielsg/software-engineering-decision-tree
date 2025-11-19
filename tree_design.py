from abc import ABC, abstractmethod
from typing import List
import numpy as np

# =================================
# Classes para o Composite
class Node:
    def execute(self) -> None:
        raise NotImplementedError()

class LeafNode(Node):
    def __init__(self, value: int):
        # Valor salvo da folha
        self.value = value
    
    def execute(self):
        return self.value

class DecisionNode(Node):
    def __init__(self, left_node: Node, right_node: Node):
        self.left = left_node
        self.right = right_node
    
    def execute(self):
        # Retorna a média dos valores dos nós
        left_result = self.left.execute()
        right_result = self.right.execute()
        return (left_result + right_result) / 2
# ===================================

# ===================================
# Classes para o State da árvore (crescendo, parada ou podando)
class TreeState:
    @abstractmethod
    def process(self, builder: 'TreeBuilder', X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """
        Método abstrato que vai receber o X e y de treino e retornar
        ou um nó folha ou um nó de decisão

        Args:
            builder (TreeBuilder): O construtor da árvore, onde alteramos os estados
            X (np.ndarray): Dados de treino
            y (np.ndarray): Labels de treino
            depth (int): Qual profundidade da árvore estamos

        Returns:
            Node: Um nó folha ou de decisão, dependendo se tivemos Stop
        """
        raise NotImplementedError()
    
class SplittingState(TreeState):
    """
    É utilizado toda vez que o modelo faz um novo split, encontrando o
    melhor split possível para os dados e chamando recursivamente para 
    ambos os lados 
    """
    def process(self, builder: 'TreeBuilder', X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """
        1. Se não houver bom split, retorna uma folha
        2. Se tiver, acha o melhor split;
        3. Divide os dados;
        4. Chama o builder recursivamente para os filhos;
        5. Retorna um DecisionNode.
        """
        feature, threshold = self._find_best_split(builder, X, y)
        
        # Se não foi possível dividir, retorna uma folha
        if feature is None:
            return LeafNode(value=np.mean(y))

        X_left, y_left, X_right, y_right = self._do_split(X, y, feature, threshold)
        
        # Recursão para direita e esquerda
        left_node = builder.fit(X_left, y_left, depth + 1)
        right_node = builder.fit(X_right, y_right, depth + 1)
        
        return DecisionNode(left_node, right_node)

    def _find_best_split(self, builder, X, y):
        ...
        
    def _do_split(self, X, y, feature, threshold):
        ...

class StoppingState(TreeState):
    """
    É utilizado após o modelo fazer um novo split, depois checando, para
    os dois lados, se atingimos certos critérios de parada.
    """
    def process(self, builder: 'TreeBuilder', X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """
        Verifica critérios de parada (max_depth, min_samples, pureza)
        - Se deve parar: Retorna LeafNode;
        - Se não: Muda o estado do builder para o SplittingState.
        """
        if self._should_stop(builder, X, y, depth):
            return LeafNode(value=self._calculate_leaf_value(y))
        
        # Não parou, muda de estado e continua fazendo a árvore
        builder.change_state(SplittingState())
        return builder.processing_state(X, y, depth)

    def _should_stop(self, builder, X, y, depth) -> bool:
        ...
    
    def _calculate_leaf_value(self, y) -> float:
        ...

class PruningState(TreeState):
    """
    Irá fazer a poda da árvore depois que ela estiver pronta
    # TODO
    """
    def process(self, builder: 'TreeBuilder', X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        pass

    def _prune(self, node: Node, X_val: np.ndarray, y_val: np.ndarray) -> Node:
        pass
# ===================================

# ===================================
# Classe do TreeBuilder
class TreeBuilder:
    def __init__(self, max_depth: int = 5, min_samples: int = 5):
        self.max_depth = max_depth
        self.min_samples = min_samples
        
        # Estado inicial como Stopping
        self._state: TreeState = StoppingState()

    def change_state(self, state: TreeState):
        print(f"Transição de estados: {type(self._state).__name__} -> {type(state).__name__}")
        self._state = state

    def processing_state(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        print(f"Processando estado: {type(self._state).__name__}")
        return self._state.process(self, X, y, depth)

    def fit(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        self.change_state(StoppingState())
        return self.processing_state(X, y, depth)
# ====================================