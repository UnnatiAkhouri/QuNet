import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import Arc, FancyBboxPatch
import matplotlib.cm as cm

# Set style for better aesthetics
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11


def draw_beautiful_markov_chain(p1, p2, p3, p4):
    """
    Draw a beautiful, minimalist Markov chain diagram
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.patch.set_facecolor('white')

    # Elegant color palette
    colors = {
        'node_fill': '#E8F4F8',  # Soft blue
        'node_edge': '#2C3E50',  # Dark blue-gray
        'green_low': '#A8D5BA',  # Soft sage green
        'green_high': '#2D6A4F',  # Deep forest green
        'red_low': '#F4ACB7',  # Soft rose
        'red_high': '#9D2235',  # Deep burgundy
        'text': '#2C3E50'  # Dark blue-gray
    }

    # Define positions
    positions = {
        '(A,A)': (2, 6),
        '(A,B)': (6, 6),
        '(B,A)': (2, 2),
        '(B,B)': (6, 2)
    }

    # State labels - cleaner format
    state_labels = {
        '(A,A)': 'AA',
        '(A,B)': 'AB',
        '(B,A)': 'BA',
        '(B,B)': 'BB'
    }

    def get_color(prob, outcome):
        """Get color based on probability and outcome"""
        if outcome == 'A':
            # Interpolate between soft green and deep green
            r = colors['green_low'] if prob < 0.5 else colors['green_high']
            return r
        else:
            # Interpolate between soft red and deep red
            r = colors['red_low'] if prob < 0.5 else colors['red_high']
            return r

    def draw_self_loop(pos, prob, outcome='A'):
        """Draw elegant self-loop"""
        x, y = pos
        linewidth = 1.5 + 4 * prob
        color = get_color(prob, outcome)

        if pos == positions['(A,A)']:
            # Arc above
            arc = Arc((x, y + 0.45), 1.0, 1.0, angle=0, theta1=25, theta2=155,
                      color=color, linewidth=linewidth, zorder=3,
                      linestyle='-', alpha=0.9)
            ax.add_patch(arc)
            # Arrowhead
            ax.annotate('', xy=(x - 0.32, y + 0.82), xytext=(x - 0.26, y + 0.87),
                        arrowprops=dict(arrowstyle='->', color=color, lw=linewidth,
                                        shrinkA=0, shrinkB=0),
                        zorder=4)
            # Probability label - no box, bigger font
            ax.text(x, y + 1.25, f'{prob:.2f}', ha='center', va='center',
                    fontsize=25, color=color, zorder=7)
        else:  # (B,B)
            # Arc below
            arc = Arc((x, y - 0.45), 1.0, 1.0, angle=0, theta1=205, theta2=335,
                      color=color, linewidth=linewidth, zorder=3,
                      linestyle='-', alpha=0.9)
            ax.add_patch(arc)
            # Arrowhead
            ax.annotate('', xy=(x + 0.32, y - 0.82), xytext=(x + 0.26, y - 0.87),
                        arrowprops=dict(arrowstyle='->', color=color, lw=linewidth,
                                        shrinkA=0, shrinkB=0),
                        zorder=4)
            # Probability label - no box, bigger font
            ax.text(x, y - 1.25, f'{prob:.2f}', ha='center', va='center',
                    fontsize=25, color=color, fontweight='bold',
                    zorder=7)

    def draw_arrow(start_pos, end_pos, prob, offset=0.3,proboff=0.1, outcome='A', curve=False):
        """Draw elegant arrow"""
        x1, y1 = start_pos
        x2, y2 = end_pos

        linewidth = 1.5 + 4 * prob
        color = get_color(prob, outcome)

        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx ** 2 + dy ** 2)

        norm_dx, norm_dy = dx / length, dy / length
        perp_dx, perp_dy = -norm_dy * offset, norm_dx * offset

        start_x = x1 + 0.48 * norm_dx
        start_y = y1 + 0.48 * norm_dy
        end_x = x2 - 0.48 * norm_dx
        end_y = y2 - 0.48 * norm_dy

        mid_x = (start_x + end_x) / 2 + perp_dx
        mid_y = (start_y + end_y) / 2 + perp_dy

        arrowprops = dict(arrowstyle='->', color=color, lw=linewidth,
                          zorder=3, alpha=0.9)
        if curve:
            arrowprops['connectionstyle'] = 'arc3,rad=0.25'

        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                    arrowprops=arrowprops)

        # Probability label - no box, bigger font
        ax.text(mid_x, mid_y-proboff, f'{prob:.2f}', ha='center', va='center',
                fontsize=25, color=color, fontweight='bold',
                zorder=7)

    # Draw all transitions
    draw_self_loop(positions['(A,A)'], p1, outcome='A')
    draw_arrow(positions['(A,A)'], positions['(A,B)'], 1 - p1,
               offset=0.15,proboff=-0.05, outcome='B', curve=False)

    draw_arrow(positions['(A,B)'], positions['(B,A)'], p4,
               offset=-1.2,proboff=0, outcome='A', curve=True)
    draw_arrow(positions['(A,B)'], positions['(B,B)'], 1 - p4,
               offset=0.45,proboff=0, outcome='B', curve=False)

    draw_arrow(positions['(B,A)'], positions['(A,A)'], p3,
               offset=0.45,proboff=0, outcome='A', curve=False)
    draw_arrow(positions['(B,A)'], positions['(A,B)'], 1 - p3,
               offset=-1.2, proboff=0, outcome='B', curve=True)

    draw_arrow(positions['(B,B)'], positions['(B,A)'], p2,
               offset=0.15,proboff=0.1, outcome='A', curve=False)
    draw_self_loop(positions['(B,B)'], 1 - p2, outcome='B')

    # Draw state nodes LAST (on top)
    for state, (x, y) in positions.items():
        # Outer circle with shadow effect
        shadow = patches.Circle((x + 0.03, y - 0.03), 0.45, facecolor='#CCCCCC',
                                edgecolor='none', alpha=0.3, zorder=8)
        ax.add_patch(shadow)

        # Main circle
        circle = patches.Circle((x, y), 0.45, facecolor=colors['node_fill'],
                                edgecolor=colors['node_edge'], linewidth=2.5, zorder=9)
        ax.add_patch(circle)

        # State label
        ax.text(x, y, state_labels[state], ha='center', va='center',
                fontsize=18, fontweight='bold', color=colors['text'],
                zorder=10, family='monospace')

    # Minimal title
    ax.text(4, 8.2, 'Markov Chain', ha='center', va='center',
            fontsize=25, fontweight='300', color=colors['text'])

    # Clean axes
    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(0.2, 8.5)
    ax.set_aspect('equal')
    ax.axis('off')

    return fig, ax

#0300 (0.05,0.95,0.05,0.05)
#3032 (0.95,0.05,0.95,0.5)
#2000 (0.5,0.05,0.05,0.05)
#1302 (0.1,0.95,0.05,0.5)
#3311 (0.95,0.95,0.1,0.1)
#0330 (0.05,0.95,0.95,0.05)
#rand(0.5,0.5,0.5,0.5)
# Create the beautiful version
print("Creating aesthetic Markov chain diagram...")
fig, ax = draw_beautiful_markov_chain(0.95,0.05,0.95,0.5)
plt.tight_layout()
plt.savefig("Markov_chain_rule_3032",dpi=600,bbox_inches='tight')
plt.show()





