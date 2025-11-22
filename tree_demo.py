from tree_design import (
    TreeBuilder, TreePruner, Node,
    GetDepthVisitor, CountLeavesVisitor,
    InOrderIterator, PostOrderIterator,
    LeafNode, DecisionNode
)
import random

if __name__ == "__main__":
    print("==========================================")
    print("Iniciando o Teste")
    print("==========================================\n")
    random.seed(10)

    X_mock = ["dado1", "dado2"]
    y_mock = [1.0, 0.0]
    
    print("Chamando o construtor da árvore")
    builder = TreeBuilder(max_depth=5)
    
    # Chamamos o fit, que alterna entre o estados e cria a árvore (Stopping -> Splitting -> Stopping...)
    root_node = builder.fit(X_mock, y_mock)
    
    if root_node:
        print("\n==========================================")
        print("Visitors")
        print("==========================================\n")
        depth_visitor = GetDepthVisitor()
        max_depth = root_node.accept(depth_visitor)
        print(f"\n ------> [Visitor] Profundidade Máxima da Árvore: {max_depth}\n")

        leaves_visitor = CountLeavesVisitor()
        num_leaves = root_node.accept(leaves_visitor)
        print(f"\n ------> [Visitor] Número de folhas: {num_leaves}\n")

        print("\n==========================================")
        print("Iterators")
        print("==========================================\n")
        # Percorre Esquerda -> Nó -> Direita
        iteratorI = InOrderIterator(root_node)
        for i, node in enumerate(iteratorI):
            prefix = f"   [{i}] "
            
            if isinstance(node, DecisionNode):
                print(f"{prefix} DecisionNode | Feature {node.feature_index} <= {node.threshold:.2f}")
            elif isinstance(node, LeafNode):
                print(f"{prefix} LeafNode     | Valor Predito: {node.value:.2f}")
        
        print("\n==========================================")
        print("==========================================\n")
        
        iteratorP = PostOrderIterator(root_node)
        for i, node in enumerate(iteratorP):
            prefix = f"   [{i}] "
            
            if isinstance(node, DecisionNode):
                print(f"{prefix} DecisionNode | Feature {node.feature_index} <= {node.threshold:.2f}")
            elif isinstance(node, LeafNode):
                print(f"{prefix} LeafNode     | Valor Predito: {node.value:.2f}")

        print("\n==========================================")
        print("Poda")
        print("==========================================\n")
        X_val = ["val_1", "val_2"]
        y_val = [1.0, 0.0]
        
        pruner = TreePruner()
        root_node = pruner.prune(root_node, X_val, y_val)
        
        print("\n==========================================")
        print("Pós Poda")
        print("==========================================\n")
        new_depth = root_node.accept(depth_visitor)
        print(f"\n ------> [Visitor] Profundidade pós poda: {new_depth} (antigo: {max_depth})\n")
        
        new_leaves = root_node.accept(leaves_visitor)
        print(f"\n ------> [Visitor] Número de folhas pós poda: {new_leaves} (antigo: {num_leaves})\n")