import numpy as np
from itertools import combinations, permutations
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np



def find_k4_quadruples(adjacency_matrix):
    quadruples = []
    num_nodes = len(adjacency_matrix)

    for nodes in combinations(range(num_nodes), 4):
        # Check if all 6 pairs are connected ie it should be a K4 graph
        is_k4 = True
        for i in range(4):
            for j in range(i + 1, 4):
                if adjacency_matrix[nodes[i]][nodes[j]] != 1:
                    is_k4 = False
                    break
            if not is_k4:
                break

        if is_k4:
            quadruples.append(list(nodes))

    return quadruples

#use above code to find k4 quadruples embedded within a larger graph given by the adjacncy matrix
def find_k4_subgroups(adjacency_matrix, num_quadruples):
    k4_quadruples = find_k4_quadruples(adjacency_matrix)
    print(f"Total K4 quadruples: {len(k4_quadruples)}")

    num_nodes = len(adjacency_matrix)
    all_subgroups = []

    def is_valid(curr_quads):
        nodes = [node for quad in curr_quads for node in quad]
        return len(set(nodes)) == num_nodes

    def backtrack(curr_quads, remaining_quads):
        if len(curr_quads) == num_quadruples:
            if is_valid(curr_quads):
                print(f"Found valid partition: {curr_quads}")
                all_subgroups.append(sorted(curr_quads))
            return

        for quad in remaining_quads:
            if all(node not in [n for q in curr_quads for n in q] for node in quad):
                curr_quads.append(quad)
                next_quads = [q for q in remaining_quads if q != quad]
                backtrack(curr_quads, next_quads)
                curr_quads.pop()

    backtrack([], k4_quadruples)
    return all_subgroups

adj_matrix = np.array([[0, 1, 1, 1, 1, 1, 1, 1],
     [1, 0, 1, 1, 1, 1, 1, 1],
     [1, 1, 0, 1, 1, 1, 1, 1],
     [1, 1, 1, 0, 1, 1, 1, 1],
     [1, 1, 1, 1, 0, 1, 1, 1],
     [1, 1, 1, 1, 1, 0, 1, 1],
     [1, 1, 1, 1, 1, 1, 0, 1],
     [1, 1, 1, 1, 1, 1, 1, 0]])

num_quadruples=2

#codes to plot the quadruples

#THis plots the adjacency matrix and highlights the chosen quadruples
#Note it only plots a particular set of quadruples, not all of them
def plot_chosen_quadruples(adjacency_matrix, chosen_quadruples):
    plt.figure(figsize=(8, 8))
    plt.imshow(adjacency_matrix, cmap='binary', interpolation='none')
    plt.title('Adjacency Matrix with Chosen Quadruples')
    plt.xlabel('Node Index')
    plt.ylabel('Node Index')
    # Plot chosen quadruples with different colors
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
    for i, quad in enumerate(chosen_quadruples):
        color = colors[i % len(colors)]
        for j in range(len(quad)):
            for k in range(j + 1, len(quad)):
                plt.plot([quad[j], quad[k]], [quad[j], quad[k]], 'o', color=color, markersize=8, alpha=0.7)

    plt.colorbar()
    plt.show()

# This function plots the network graph of the adjacency matrix
def plot_network_graph_quadruples(adjacency_matrix):
    G = nx.Graph()
    G = nx.from_numpy_array(adjacency_matrix)
    pos = nx.circular_layout(G)

    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000, edge_color='gray', width=2)
    plt.title('Network Graph')
    plt.show()

# This function plots the network graph and highlights the chosen quadruples with different colors
#Note it only plots a particular set of quadruples, not all of them
def plot_network_graph_with_quadruple_highlight(adjacency_matrix, chosen_quadruples):
    G = nx.Graph()
    num_nodes = len(adjacency_matrix)

    # Add edges based on the adjacency matrix
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i][j] == 1:
                G.add_edge(i, j)

    # Add nodes
    for i in range(num_nodes):
        G.add_node(i)

    pos = nx.circular_layout(G)
    plt.figure(figsize=(8, 8))

    # Draw base graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000, font_size=12)

    # Highlight chosen quadruples with different colors
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
    for i, quad in enumerate(chosen_quadruples):
        color = colors[i % len(colors)]

        # Highlight edges within the quadruple
        quad_edges = [(quad[j], quad[k]) for j in range(len(quad)) for k in range(j + 1, len(quad))
                      if adjacency_matrix[quad[j]][quad[k]] == 1]

        nx.draw_networkx_edges(G, pos, edgelist=quad_edges, width=3, edge_color=color, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=quad, node_color=color, node_size=1200, alpha=0.6)

    plt.title('Network Graph with Highlighted Quadruples')
    plt.show()

# This function visualizes a set of quadruples as a network graph
def visualize_quadruple_network(quadruples,ax, title):
    G = nx.Graph()

    # Add edges within each quadruple (for visualization)
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
    edge_colors = []

    for i, quad in enumerate(quadruples):
        color = colors[i % len(colors)]
        for j in range(len(quad)):
            for k in range(j + 1, len(quad)):
                G.add_edge(quad[j], quad[k])
                edge_colors.append(color)

    # Get all nodes
    all_nodes = set()
    for quad in quadruples:
        all_nodes.update(quad)

    # Calculate positions for each node in a circle
    node_list = sorted(list(all_nodes))
    angle = 2 * np.pi / len(node_list)
    pos = {node: (np.cos(i * angle), np.sin(i * angle)) for i, node in enumerate(node_list)}

    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=500, edge_color=edge_colors, ax=ax, width=2)
    ax.set_title(title, fontsize=12)

# This function plots multiple sets of quadruple networks in a grid layout
def plot_multiple_quadruple_networks(list_of_quadruple_sets, rows=10, cols=10):
    fig, axes = plt.subplots(rows, cols, figsize=(30, 30))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    # Plot each network in the grid
    for idx, ax in enumerate(axes):
        if idx < len(list_of_quadruple_sets):
            visualize_quadruple_network(list_of_quadruple_sets[idx], ax, f"Quadruple Set {idx + 1}")
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


# Test it
k4_results = find_k4_subgroups(adj_matrix, 2)
#plot_chosen_quadruples(adj_matrix, k4_results[0:2])
plot_network_graph_quadruples(adj_matrix)
plot_network_graph_with_quadruple_highlight(adj_matrix,k4_results[2])
plot_multiple_quadruple_networks(k4_results)