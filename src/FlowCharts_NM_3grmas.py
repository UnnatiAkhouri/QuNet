import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_markov_chain(p1, p2, p3, p4):
    """
    Draw a simple Markov chain diagram
    p1: prob of j given (j,j)
    p2: prob of j given (g,g) 
    p3: prob of j given (g,j)
    p4: prob of j given (j,g)
    """

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Define positions for the 4 states
    positions = {
        '(j,j)': (2, 6),
        '(j,g)': (6, 6),
        '(g,j)': (2, 2),
        '(g,g)': (6, 2)
    }

    # Draw state circles
    for state, (x, y) in positions.items():
        circle = patches.Circle((x, y), 0.4, facecolor='lightblue',
                                edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, state, ha='center', va='center', fontsize=12)

    # Helper function to draw arrows with labels
    def draw_arrow(start_pos, end_pos, label, offset=0.2, color='red'):
        x1, y1 = start_pos
        x2, y2 = end_pos

        # Calculate direction and add offset for curved arrows
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx ** 2 + dy ** 2)

        if length > 0:
            # Normalize and add perpendicular offset
            norm_dx, norm_dy = dx / length, dy / length
            perp_dx, perp_dy = -norm_dy * offset, norm_dx * offset

            # Adjust start and end points to circle edges
            start_x = x1 + 0.8 * norm_dx
            start_y = y1 + 0.8 * norm_dy
            end_x = x2 - 0.8 * norm_dx
            end_y = y2 - 0.8 * norm_dy

            # Add curve
            mid_x = (start_x + end_x) / 2 + perp_dx
            mid_y = (start_y + end_y) / 2 + perp_dy

            ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2
                                        ))

            # Add label at midpoint
            ax.text(mid_x, mid_y, label, ha='center', va='center',
                    fontsize=10)

    # Draw transitions based on the logic:
    # From (a,b), generate outcome c, go to (b,c)

    # From (j,j): p1 → j → (j,j), (1-p1) → g → (j,g)
    draw_arrow(positions['(j,j)'], positions['(j,j)'], f'p1={p1}', offset=0.5, color='blue')
    draw_arrow(positions['(j,j)'], positions['(j,g)'], f'1-p1={1 - p1:.1f}', offset=0.2, color='red')

    # From (j,g): p4 → j → (g,j), (1-p4) → g → (g,g)  
    draw_arrow(positions['(j,g)'], positions['(g,j)'], f'p4={p4}', offset=0.2, color='blue')
    draw_arrow(positions['(j,g)'], positions['(g,g)'], f'1-p4={1 - p4:.1f}', offset=0.2, color='red')

    # From (g,j): p3 → j → (j,j), (1-p3) → g → (j,g)
    draw_arrow(positions['(g,j)'], positions['(j,j)'], f'p3={p3}', offset=0.2, color='blue')
    draw_arrow(positions['(g,j)'], positions['(j,g)'], f'1-p3={1 - p3:.1f}', offset=0.2, color='red')

    # From (g,g): p2 → j → (g,j), (1-p2) → g → (g,g)
    draw_arrow(positions['(g,g)'], positions['(g,j)'], f'p2={p2}', offset=0.2, color='blue')
    draw_arrow(positions['(g,g)'], positions['(g,g)'], f'1-p2={1 - p2:.1f}', offset=-0.5, color='red')

    # Set up the plot
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Add title and legend
    plt.title('Markov Chain State Diagram\n(Blue = outcome j, Red = outcome g)',
              fontsize=16, pad=10)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    # Set your probabilities
    p1, p2, p3, p4 = 0.7, 0.3, 0.6, 0.4

    print("Drawing Markov Chain...")
    print(f"Probabilities: p1={p1}, p2={p2}, p3={p3}, p4={p4}")

    # Draw the visual diagram
    draw_markov_chain(p1, p2, p3, p4)

