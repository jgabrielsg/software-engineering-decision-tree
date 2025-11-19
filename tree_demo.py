import numpy as np
from tree_design import (
    TreeBuilder, TreePruner,
    GetDepthVisitor,
    InOrderIterator,
    LeafNode, DecisionNode  # Apenas para type checking ou testes manuais
)

if __name__ == "__main__":
    X_mock = np.random.random((1000, 10))
    y_mock = np.random.random(1000)
    
    print("--- CONSTRUINDO A ÁRVORE ---")
    builder = TreeBuilder(max_depth=5, min_samples=2)
    root_node = builder.fit(X_mock, y_mock)
    print("Árvore construída com sucesso!")

    print("\n--- VISITOR ---")
    depth_visitor = GetDepthVisitor()
    max_depth = root_node.accept(depth_visitor)
    print(f"Profundidade Máxima da Árvore: {max_depth}")

    print("\n--- ITERADOR ---")
    iterator = InOrderIterator(root_node)
    
    for node in iterator:
        if isinstance(node, DecisionNode):
            print(f"[Decision] Feature {node.feature_index} <= {node.threshold:.2f}")
        elif isinstance(node, LeafNode):
            print(f"  -> [Leaf] Valor: {node.value:.2f}")

    print("\n--- PODADOR ---")
    X_val = np.random.random((1000, 10))
    y_val = np.random.random(1000)
    
    pruner = TreePruner()
    root_node = pruner.prune(root_node, X_val, y_val)