import numpy as np

import matplotlib.pyplot as plt


def umatrix(som_model, umatrix_cmap = 'RdYlBu_r', **kwargs):
    """Plot Self-organizing map U-Matrix
    
    Args:
        som_model (minisom.MiniSom): MiniSom Model
        
        umatrix_cmap (str): matplotlib colormap to use in U-Matrix grid
        
        kwargs (dict): Parameters to matplotlib.pyplot.figure function
    
    Returns:
        matplotlib.figure.Figure: grid figure
    """    
    
    fig = plt.figure(**kwargs)
    plt.pcolormesh(som_model.distance_map().T, 
                   cmap = umatrix_cmap,
                   color = 'w', 
                   linestyle='-', 
                   linewidth=0.5)
    plt.colorbar()
        
    fig.tight_layout()
    return fig


def hitmap(som_model, data, hitmap_cmap = 'RdYlBu_r', **kwargs):
    """Plot Self-organizing map hitmap
    
    Args:
        som_model (minisom.MiniSom): MiniSom Model
        
        data (np.ndarray): n-dimensional data to estimulate neurons and generate
        activation map
        
        umatrix_cmap (str): matplotlib colormap to use in U-Matrix grid
        
        kwargs (dict): Parameters to matplotlib.pyplot.figure function

    Returns:
        matplotlib.figure.Figure: grid figure
    """
    
    fig = plt.figure(**kwargs)
    
    frequencies = som_model.activation_response(data).astype(int)
    
    plt.imshow(frequencies.T, cmap = hitmap_cmap)
    plt.colorbar()
    
    # labeling pixel-by-pixel
    for (i, j), value in np.ndenumerate(frequencies.T):
        plt.text(j, i, value, verticalalignment='center', 
                              horizontalalignment='center')
    
    fig.tight_layout()
    return fig


def heatmap(som_model, hitmap_cmap = 'RdYlBu_r', **kwargs):
    """Plot Self-organizing map heatmap
    
    Args:
        som_model (minisom.MiniSom): MiniSom Model
        
        umatrix_cmap (str): matplotlib colormap to use in U-Matrix grid
        
        kwargs (dict): Parameters to matplotlib.pyplot.figure function

    Returns:
        matplotlib.figure.Figure: grid figure
    """
    
    fig = plt.figure(**kwargs)
    
    distances = som_model.distance_map()
    
    plt.imshow(distances.T, cmap = hitmap_cmap)
    plt.colorbar()
    
    # labeling pixel-by-pixel
    for (i, j), value in np.ndenumerate(distances.T):
        plt.text(j, i, f'{value:.2f}', verticalalignment='center', 
                              horizontalalignment='center')
    
    fig.tight_layout()
    return fig
