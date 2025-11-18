import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_markov_chain(p1, p2, p3, p4):
    """
    Draw a simple Markov chain diagram
    p1: prob of j given (a,a)
    p2: prob of j given (b,b) 
    p3: prob of j given (b,a)
    p4: prob of j given (a,b)
    """

    fib, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Define positions for the 4 states
    positions = {
        '(a,a)': (2, 6),
        '(a,b)': (6, 6),
        '(b,a)': (2, 2),
        '(b,b)': (6, 2)
    }

    # Draw state circles
    for state, (x, y) in positions.items():
        circle = patches.Circle((x, y), 0.4, facecolor='powderblue',
                                edgecolor='gray', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, state, ha='center', va='center', fontsize=15)

    # Helper function to draw arrows with labels
    def draw_arrow(start_pos, end_pos, label, offset=0.3, color='palevioletred', curve=False):
        x1, y1 = start_pos
        x2, y2 = end_pos

        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx ** 2 + dy ** 2)

        if length > 0:
            norm_dx, norm_dy = dx / length, dy / length
            perp_dx, perp_dy = -norm_dy * offset, norm_dx * offset

            start_x = x1 + 0.8 * norm_dx
            start_y = y1 + 0.8 * norm_dy
            end_x = x2 - 0.8 * norm_dx
            end_y = y2 - 0.8 * norm_dy

            mid_x = (start_x + end_x) / 2 + perp_dx
            mid_y = (start_y + end_y) / 2 + perp_dy

            arrowprops = dict(arrowstyle='->', color=color, lw=3)
            if curve:
                arrowprops['connectionstyle'] = 'arc3,rad=0.4'

            ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                        arrowprops=arrowprops)

            ax.text(mid_x, mid_y, label, ha='center', va='center', fontsize=15)

    # Draw transitions based on the logic:
    # From (a,b), generate outcome c, go to (b,c)

    # From (a,a): p1 → j → (a,a), (1-p1) → g → (a,b)
    draw_arrow(positions['(a,a)'], positions['(a,a)'], f'p1={p1}', offset=0.5, color='darkseagreen',curve=False)
    draw_arrow(positions['(a,a)'], positions['(a,b)'], f'1-p1={1 - p1:.1f}', offset=0.2, color='palevioletred',curve=False)

    # From (a,b): p4 → j → (b,a), (1-p4) → g → (b,b)  
    draw_arrow(positions['(a,b)'], positions['(b,a)'], f'p4={p4}', offset=1.3, color='darkseagreen',curve=True)
    draw_arrow(positions['(a,b)'], positions['(b,b)'], f'1-p4={1 - p4:.1f}', offset=0.6, color='palevioletred',curve=False)

    # From (b,a): p3 → j → (a,a), (1-p3) → g → (a,b)
    draw_arrow(positions['(b,a)'], positions['(a,a)'], f'p3={p3}', offset=0.5, color='darkseagreen',curve=False)
    draw_arrow(positions['(b,a)'], positions['(a,b)'], f'1-p3={1 - p3:.1f}', offset=1.3, color='palevioletred',curve=True)

    # From (b,b): p2 → j → (b,a), (1-p2) → g → (b,b)
    draw_arrow(positions['(b,b)'], positions['(b,a)'], f'p2={p2}', offset=0.2, color='darkseagreen',curve=False)
    draw_arrow(positions['(b,b)'], positions['(b,b)'], f'1-p2={1 - p2:.1f}', offset=-0.5, color='palevioletred',curve=False)

    # Set up the plot
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Add title and legend
    plt.title('Markov Chain State Diagram\n(Blue = outcome a, palevioletred = outcome b)',
              fontsize=16, pad=10)
    plt.savefig("markov_chain_diagram.png", dpi=300)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    # Set your probabilities
    p1, p2, p3, p4 = 0.7, 0.3, 0.6, 0.4

    print("Drawing Markov Chain...")
    print(f"Probabilities: p1={p1}, p2={p2}, p3={p3}, p4={p4}")

    # Draw the visual diagram
    draw_markov_chain(p1, p2, p3, p4)

