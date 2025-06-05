import os
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)

def save_plot(figure, modelname):
    #Save a matplotlib figure to the 'figures' directory.

    filepath = os.path.join('figures', f'{modelname}.png')
    figure.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(figure)
    print(f"Figure saved as {filepath}")