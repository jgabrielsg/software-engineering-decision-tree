from abc import ABC, abstractmethod
from typing import List
import numpy as np

# =================================
# Classes para o Composite
class Node:
    @abstractmethod
    def execute(self, X) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    def accept(self, visitor: 'NodeVisitor'):
        raise NotImplementedError()

class LeafNode(Node):
    def __init__(self, value: int):
        # Valor salvo da folha
        self.value = value
    
    def execute(self, X):
        return self.value
    
    def accept(self, visitor: 'NodeVisitor'):
        return visitor.visit_leaf(self)

class DecisionNode(Node):
    def __init__(self, feature_index: int, threshold: float, left_node: Node, right_node: Node):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left_node
        self.right = right_node
    
    def execute(self, X):
        # Pega o valor do X na feature que foi feito o split
        feature_value = X[self.feature_index]

        # Passa pra direita ou esquerda dependendo do threshold
        if feature_value <= self.threshold:
            return self.left.execute(X)
        else:
            return self.right.execute(X)
    
    def accept(self, visitor: 'NodeVisitor'):
        return visitor.visit_decision(self)
# ===================================

# ===================================
# Classes para o State da árvore (crescendo, parada)
class TreeState:
    @abstractmethod
    def process(self, builder: 'TreeBuilder', X, y, depth: int) -> Node:
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
    def process(self, builder: 'TreeBuilder', X, y, depth: int) -> Node:
        """
        1. Se não houver bom split, retorna uma folha
        2. Se tiver, acha o melhor split;
        3. Divide os dados;
        4. Chama o builder recursivamente para os filhos;
        5. Retorna um DecisionNode.
        """
        feature, threshold = self._find_best_split(builder, X, y, depth)
        
        # Se não foi possível dividir, retorna uma folha
        if feature is None:
            return LeafNode(value=np.mean(y))

        X_left, y_left, X_right, y_right = self._do_split(X, y, feature, threshold)
        
        # Recursão para direita e esquerda
        left_node = builder.fit(X_left, y_left, depth + 1)
        right_node = builder.fit(X_right, y_right, depth + 1)
        
        print(left_node, right_node)
        
        return DecisionNode(left_node, right_node)

    # RANDOM POR ENQUANTO
    # TODO
    def _find_best_split(self, builder, X, y, depth):
        if np.random.random() > 0.1:
            return 0, np.random.random()
        return None, 0
        
    # RANDOM POR ENQUANTO
    # TODO
    def _do_split(self, X, y, feature, threshold):
        mid = len(y) // 2 
        return X[:mid], y[:mid], X[mid:], y[mid:]

class StoppingState(TreeState):
    """
    É utilizado após o modelo fazer um novo split, depois checando, para
    os dois lados, se atingimos certos critérios de parada.
    """
    def process(self, builder: 'TreeBuilder', X, y, depth: int) -> Node:
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

    # RANDOM POR ENQUANTO
    # TODO
    def _should_stop(self, builder, X, y, depth) -> bool:
        rand = np.random.random()
        if rand > 0.8:
            return True
        return False
    
    # RANDOM POR ENQUANTO
    # TODO
    def _calculate_leaf_value(self, y) -> float:
        return np.random.random()
# ===================================
# Classe para poda separada (não cabia bem no TreeState)
class TreePruner:
    """
    Irá fazer a poda da árvore depois que ela estiver pronta
    """
    def prune(self, node: Node, X_val, y_val) -> Node:
        """
        1. Se já for folha, retorna ela;
        2. Se for nó de decisão, pega a feature e threshold da decisão;
        3. Divide os dados para direita ou esquerda;
        4. Chama o builder recursivamente para os filhos;
        5. Se o erro for menor quando tiver podado, mantém a poda
        """
        # Se é folha, sem poda
        if isinstance(node, LeafNode):
            return node

        if isinstance(node, DecisionNode):
            # Mapeia os X e y de validação para direita ou esquerda dependendo
            # da feature que fizemos o split e do threshold dela
            mask = X_val[:, node.feature_index] <= node.threshold
            X_val_left, y_val_left = X_val[mask], y_val[mask]
            X_val_right, y_val_right = X_val[~mask], y_val[~mask]

            # Poda os filhos primeiro 
            node.left = self.prune(node.left, X_val_left, y_val_left)
            node.right = self.prune(node.right, X_val_right, y_val_right)

            # Erro mantendo a divisão
            current_error = self._calculate_error(node, X_val, y_val)
            
            # Erro se virar folha (poda)
            leaf_value = np.mean(y_val) if len(y_val) > 0 else 0
            pruned_error = np.mean((y_val - leaf_value) ** 2) # MSE
            
            if pruned_error <= current_error:
                return LeafNode(value=leaf_value)
    
            return node

    # RANDOM POR ENQUANTO
    # TODO
    def _calculate_error(self, node, X, y):
        return np.random.random()
# ===================================

# ===================================
# Classe do TreeBuilder
class TreeBuilder:
    def __init__(self, max_depth: int = 10, min_samples: int = 10):
        self.max_depth = max_depth
        self.min_samples = min_samples
        
        # Estado inicial como Stopping
        self._state: TreeState = StoppingState()
        self._node: Node = None

    def change_state(self, state: TreeState):
        #print(f"Transição de estados: {type(self._state).__name__} -> {type(state).__name__}")
        self._state = state

    def processing_state(self, X, y, depth: int) -> Node:
        #print(f"Processando estado: {type(self._state).__name__}")
        return self._state.process(self, X, y, depth)

    def fit(self, X, y, depth: int = 0) -> Node:
        self.change_state(StoppingState())
        return self.processing_state(X, y, depth)
    
    def prune(self, X_val, y_val):
        pruner = TreePruner()
        print(f"Fazendo Poda da Árvore: {type(pruner).__name__}")
        self.root = pruner.prune(self.root, X_val, y_val)
        return self.root
# ====================================

# ====================================
# Class do Visitor
class NodeVisitor(ABC):
    @abstractmethod
    def visit_leaf(self, node: LeafNode) -> int: ...
    
    @abstractmethod
    def visit_decision(self, node: DecisionNode) -> int: ...

class GetDepthVisitor(NodeVisitor):
    """
    Visita a árvore para calcular a profundidade máxima.
    """
    def visit_leaf(self, node: LeafNode) -> int:
        return 1 # folha tem profundidade 1

    def visit_decision(self, node: DecisionNode) -> int:
        left_depth = node.left.accept(self)
        right_depth = node.right.accept(self)
        return 1 + max(left_depth, right_depth)

class CountLeavesVisitor(NodeVisitor):
    """
    Conta a quantidade de folhas na árvore
    """
    def visit_leaf(self, node: LeafNode) -> int:
        return 1 # achamos 1 folha!

    def visit_decision(self, node: DecisionNode) -> int:
        left_leaves = node.left.accept(self)
        right_leaves = node.right.accept(self)
        return left_leaves + right_leaves
# ====================================

# ====================================
# Classe do Iterator
class Iterator(ABC):
    @abstractmethod
    def __iter__(self):
        return self
    
    @abstractmethod
    def __next__(self):
        raise StopIteration

class InOrderIterator(Iterator):
    def __init__(self, root: Node):
        self.traversal = self._traverse(root)

    def __iter__(self):
        return self
    
    def __next__(self):
        return next(self.traversal)

    def _traverse(self, node: Node):
        if node:
            # Desce na esquerda
            if isinstance(node, DecisionNode):
                yield from self._traverse(node.left)
            
            # Retorna o nó
            yield node
            
            # Desce na direita por último
            if isinstance(node, DecisionNode):
                yield from self._traverse(node.right)

class PostOrderIterator(Iterator):
    def __init__(self, root: Node):
        self.traversal = self._traverse(root)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.traversal)

    def _traverse(self, node: Node):
        if node:
            # Desce na esquerda
            if isinstance(node, DecisionNode):
                yield from self._traverse(node.left)
                
            # Desce na direita
            if isinstance(node, DecisionNode):
                yield from self._traverse(node.right)
                
            # Retorna o nó por último
            yield node
# ====================================