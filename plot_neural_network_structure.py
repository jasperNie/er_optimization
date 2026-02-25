#!/usr/bin/env python3
"""
Visualize the structure of the neural network used in the optimizer.
"""
import matplotlib.pyplot as plt
import networkx as nx


# Architecture from FairNeuralEvolutionOptimizer
input_size = 6
hidden_size = 6
output_size = 1

layer_sizes = [input_size, hidden_size, output_size]
layer_labels = ['Input', 'Hidden', 'Output']


G = nx.DiGraph()
positions = {}
labels = {}
layer_x_gap = 2
layer_y_gap = 1.2

# Input feature labels from optimizer
input_features = [
    'Severity (normalized)',
    'Deterioration risk',
    'Wait time (normalized)',
    'Time of day (cycle)',
    'Queue length (normalized)',
    'Nurse availability'
]

# Build nodes and positions
for l, (size, label) in enumerate(zip(layer_sizes, layer_labels)):
    x = l * layer_x_gap
    y_offset = -(size-1) * layer_y_gap / 2
    for n in range(size):
        node_name = f'{label[0]}{n+1}_L{l}'
        G.add_node(node_name, layer=l)
        positions[node_name] = (x, y_offset + n * layer_y_gap)
        if l == 0:
            labels[node_name] = input_features[n]
        elif l == len(layer_sizes)-1:
            labels[node_name] = 'Triage Score'
        else:
            labels[node_name] = ''

# Add edges between layers
for l in range(len(layer_sizes)-1):
    src_size = layer_sizes[l]
    tgt_size = layer_sizes[l+1]
    for src in range(src_size):
        for tgt in range(tgt_size):
            src_node = f'{layer_labels[l][0]}{src+1}_L{l}'
            tgt_node = f'{layer_labels[l+1][0]}{tgt+1}_L{l+1}'
            G.add_edge(src_node, tgt_node)

plt.figure(figsize=(11, 6))
nx.draw_networkx_nodes(G, positions, node_size=1400, node_color='#A7C7E7', edgecolors='k')
nx.draw_networkx_edges(G, positions, arrows=False, alpha=0.5)

# Draw input/output labels to the left/right of nodes for better visibility
for node, (x, y) in positions.items():
    if node.startswith('I'):
        plt.text(x-0.5, y, labels[node], fontsize=11, ha='right', va='center', wrap=True)
    elif node.startswith('O'):
        plt.text(x+0.5, y, labels[node], fontsize=11, ha='left', va='center', wrap=True)


# Label hidden layer at the bottom
hidden_x = positions['H1_L1'][0]
hidden_y_min = min(positions[f'H{i+1}_L1'][1] for i in range(6))
plt.text(hidden_x, hidden_y_min-1.5, 'Hidden Layer (6 neurons)', fontsize=12, ha='center', va='top', fontweight='bold')

# Label input layer at the bottom
input_x = positions['I1_L0'][0]
input_y_min = min(positions[f'I{i+1}_L0'][1] for i in range(6))
plt.text(input_x, input_y_min-1.5, 'Input Layer (6 features)', fontsize=12, ha='center', va='top', fontweight='bold')

plt.title('Neural Network Structure', fontsize=16)
plt.axis('off')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('neural_network_structure.png', dpi=300, bbox_inches='tight')
plt.show()
print('Neural network structure graphic saved as neural_network_structure.png')
