from abc import ABC, abstractmethod
import random

# =================================
# Classes para o Composite
# =================================
class Node(ABC):
    @abstractmethod
    def execute(self, x) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    def accept(self, visitor: 'NodeVisitor'):
        raise NotImplementedError()

class LeafNode(Node):
    def __init__(self, value: float):
        # Valor salvo da folha
        self.value = value
    
    def execute(self, x):
        print(f"  [LeafNode] Retornando valor fixo: {self.value}")
        return self.value
    
    def accept(self, visitor: 'NodeVisitor'):
        return visitor.visit_leaf(self)

class DecisionNode(Node):
    def __init__(self, feature_index: int, threshold: float, left_node: Node, right_node: Node):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left_node
        self.right = right_node
    
    def execute(self, x):
        # Apenas finge que verificou e só "desce" para a esquerda sempre
        print(f"  [DecisionNode] Verificando se feature {self.feature_index} <= {self.threshold}")
        print("  [DecisionNode] Decisão: indo para a Esquerda.") 
        return self.left.execute(x)
    
    def accept(self, visitor: 'NodeVisitor'):
        return visitor.visit_decision(self)
# ===================================

# ===================================
# Classes para o State da árvore (crescendo, parada)
class TreeState(ABC):
    @abstractmethod
    def process(self, builder: 'TreeBuilder', X, y, depth: int) -> Node:
        """
        Método abstrato que vai receber o X e y de treino e retornar
        ou um nó folha ou um nó de decisão

        Args:
            builder (TreeBuilder): O construtor da árvore, onde alteramos os estados
            X (list): Dados de treino
            y (list): Labels de treino 
            depth (int): Qual profundidade da árvore estamos

        Returns:
            Node: Um nó folha ou de decisão, dependendo se tivermos Stopping
        """
        raise NotImplementedError()
    
class SplittingState(TreeState):
    """
    É utilizado toda vez que o modelo faz um novo split, encontrando o
    melhor split possível para os dados e chamando recursivamente para 
    ambos os lados.
    """
    def process(self, builder: 'TreeBuilder', X, y, depth: int) -> Node:
        """
        1. Se não houver bom split, retorna uma folha
        2. Se tiver, acha o melhor split;
        3. Divide os dados;
        4. Chama o builder recursivamente para os filhos;
        5. Retorna um DecisionNode.
        """
        print(f" ====== [SplittingState] Processando split na profundidade {depth}...")
        feature, threshold = self._find_best_split(builder, X, y, depth)
        
        # Se não foi possível dividir, retorna uma folha
        if feature is None:
            print(" ====== [SplittingState] Nenhum split encontrado. Retornando LeafNode.")
            return LeafNode(value=random.random()) 

        X_left, y_left, X_right, y_right = self._do_split(X, y, feature, threshold)
        
        # Recursão para direita e esquerda
        print(f" ====== [SplittingState] Criando filhos recursivos para feature {feature} <= {threshold}")
        left_node = builder.fit(X_left, y_left, depth + 1)
        right_node = builder.fit(X_right, y_right, depth + 1)
        #print(left_node, right_node)
        
        return DecisionNode(feature, threshold, left_node, right_node)

    def _find_best_split(self, builder, X, y, depth):
        # =======================================================
        # Como é mock, deixei a chance de não achar aleatória
        # =======================================================
        if random.random() < 0.1:
            return None, None

        # =======================================================
        # Como é mock, deixei a feature e threshold aleatórios
        # =======================================================
        print(" ====== [SplittingState] -> Buscando melhor split...")
        feature_idx = random.randint(0, 4)
        threshold = random.random()
        
        return feature_idx, threshold
        
    def _do_split(self, X, y, feature_idx, threshold):
        # Cria o X_left e X_right quaisquer
        print(f" ====== [SplittingState] -> Dividindo dados (Split) com threshold {threshold}")
        X_left, y_left = ["mock_X_left"], ["mock_y_left"]
        X_right, y_right = ["mock_X_right"], ["mock_y_right"]
        
        return X_left, y_left, X_right, y_right

# ===================================
class StoppingState(TreeState):
    """
    É utilizado após o modelo fazer um novo split, depois checando, para
    os dois lados, se atingimos certos critérios de parada.
    """
    def process(self, builder: 'TreeBuilder', X, y, depth: int) -> Node:
        """
        1. Recebe o dado do TreeBuilder
        2. Se bater algum critério de parada, cria uma folha
        3. Caso não, muda o estado para Splitting de novo e continua o crescimento
        """
        print(f" XXXXX [StoppingState] Verificando critérios de parada na profundidade {depth}...")

        if self._should_stop(builder, X, y, depth):
            val = self._calculate_leaf_value(y)
            print(f" XXXXX [StoppingState] Critério atingido. Criando Folha com valor {val}.")
            return LeafNode(value=val)
        
        # Não parou, muda de estado e continua fazendo a árvore
        print(" XXXXX [StoppingState] Critérios não atingidos. Mudando para SplittingState.")
        builder.change_state(SplittingState())
        return builder.processing_state(X, y, depth)

    def _should_stop(self, builder, X, y, depth) -> bool:
        # trata lista vazia para evitar erro
        if len(y) == 0:
            print(" XXXXX [StoppingState] -> Lista vazia.")
            return True

        # profundidade máxima (pra não ficar gigante também)
        if depth >= builder.max_depth:
            print(" XXXXX [StoppingState] -> Profundidade máxima atingida.")
            return True
        
        # 20% de chance de parar
        if random.random() > 0.8 and depth >= 3:
            print(" XXXXX [StoppingState] -> Há piora!")
            return True
        
        return False
    
    def _calculate_leaf_value(self, y) -> float:
        print(" XXXXX [StoppingState] -> Calculando média dos valores...")
        if len(y) == 0:
            return 0.0
        return random.randint(0, 10)
# ===================================

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
        print("[TreePruner] Visitando nó...")

        # Se é folha, sem poda
        if isinstance(node, LeafNode):
            print("  -> [F] Nó é Folha! Mantendo.")
            return node

        if isinstance(node, DecisionNode):
            print(f"  -> [D] Nó de Decisão (Feat {node.feature_index}). Dividindo dados de validação.")
            
            # Dados mock
            X_val_left, y_val_left = ["mock_val_left"], ["mock_y_left"]
            X_val_right, y_val_right = ["mock_val_right"], ["mock_y_right"]

            # Poda os filhos
            node.left = self.prune(node.left, X_val_left, y_val_left)
            node.right = self.prune(node.right, X_val_right, y_val_right)

            # Erro mantendo a divisão
            current_error = self._calculate_error(node, X_val, y_val)
            
            # Erro folha e podado
            leaf_value = random.random()
            pruned_error = random.random()
            
            print(f"  -> Comparando Erros: Atual={current_error} vs Podado={pruned_error}")
            
            # Poda aleatóriamente 40% das vezes
            if random.random() < 0.4:
                print("  -> [PODA] Erro diminuiu. Transformando em Folha!")
                return LeafNode(value=leaf_value)
    
            print("  -> [XXXX] Erro aumentaria. Mantendo Decisão!")
            return node

    def _calculate_error(self, node, X, y):
        # lista vazia retorna 0
        if len(y) == 0:
            return 0.0
        
        # Retorna um erro aleatório
        return random.random()
# ===================================

# ===================================
# Classe do TreeBuilder
class TreeBuilder:
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        
        # Estado inicial como Stopping
        self._state: TreeState = StoppingState()
        self.root: Node = None

    def change_state(self, state: TreeState):
        # print(f"--- Transição de estados: {type(self._state).__name__} -> {type(state).__name__} ---")
        self._state = state

    def processing_state(self, X, y, depth: int) -> Node:
        # print(f"Processando estado: {type(self._state).__name__}")
        return self._state.process(self, X, y, depth)

    def fit(self, X, y, depth: int = 0) -> Node:
        # Reinicia o estado para cada recursão
        self.change_state(StoppingState())
        
        node = self.processing_state(X, y, depth)
        if depth == 0:
            self.root = node
            
        return node
    
    def prune(self, X_val, y_val):
        pruner = TreePruner()
        print(f"\n=== Iniciando Poda: {type(pruner).__name__} ===")
        
        if self.root:
            self.root = pruner.prune(self.root, X_val, y_val)
        else:
            print("Árvore não treinada ainda.")
            
        return self.root
# ===================================

# ====================================
# Class do Visitor
# ====================================
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
        print("    [GetDepthVisitor] Chegou em uma folha. Profundidade base = 1.")
        return 0 # folha tem profundidade 0

    def visit_decision(self, node: DecisionNode) -> int:
        print(f"    [GetDepthVisitor] Visitando nó de decisão (Feat {node.feature_index}). Descendo...")
        left_depth = node.left.accept(self)
        right_depth = node.right.accept(self)
        
        depth = 1 + max(left_depth, right_depth)
        print(f"    [GetDepthVisitor] Retornando profundidade calculada: {depth}")
        return depth

class CountLeavesVisitor(NodeVisitor):
    """
    Conta a quantidade de folhas na árvore
    """
    def visit_leaf(self, node: LeafNode) -> int:
        print("    [CountLeavesVisitor] +1 Folha encontrada.")
        return 1 # achamos 1 folha!

    def visit_decision(self, node: DecisionNode) -> int:
        print("    [CountLeavesVisitor] Passando por nó de decisão. Somando filhos...")
        left_leaves = node.left.accept(self)
        right_leaves = node.right.accept(self)
        return left_leaves + right_leaves
# ====================================


# ====================================
# Classe do Iterator
# ====================================
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
                print("  [InOrder Iterator] -> Indo para Esquerda...")
                yield from self._traverse(node.left)
            
            # Retorna o nó (Visit)
            if isinstance(node, LeafNode):
                print(f"  [InOrder Iterator] Folha (Val: {node.value})")
            else:
                print(f"  [InOrder Iterator] Decisão (Feat: {node.feature_index})")
            yield node
            
            # Desce na direita por último
            if isinstance(node, DecisionNode):
                print("  [InOrder Iterator] -> Indo para Direita...")
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
                print("  [PostOrder Iterator] -> Indo para Esquerda...")
                yield from self._traverse(node.left)
                
            # Desce na direita
            if isinstance(node, DecisionNode):
                print("  [PostOrder Iterator] -> Indo para Direita...")
                yield from self._traverse(node.right)
                
            # Retorna o nó por último
            if isinstance(node, LeafNode):
                print(f"  [PostOrder Iterator] Folha (Val: {node.value})")
            else:
                print(f"  [PostOrder Iterator] Decisão (Feat: {node.feature_index})")
            yield node
# ====================================