import numpy as np

import sklearn.datasets

import matplotlib.pyplot as plt


def umatrix(som_model, use_colorbar=True, **kwargs):
    """Plot Self-organizing map U-Matrix
    
    Args:
        som_model (minisom.MiniSom): MiniSom Model
        
        use_colorbar (bool): Flag to enable colorbar on figure plot
        
        kwargs (dict): Parameters to matplotlib.pyplot.imshow function
    Returns:
        matplotlib.figure.Figure: grid figure
    """    
        
    im = plt.imshow(som_model.distance_map(), **kwargs)
    
    if use_colorbar: plt.colorbar(im)
              

def hitmap(som_model, data, use_colorbar=True, **kwargs):
    """Plot Self-organizing map hitmap
    
    Args:
        som_model (minisom.MiniSom): MiniSom Model
        
        data (np.ndarray): n-dimensional data to estimulate neurons and generate
        activation map
        
        use_colorbar (bool): Flag to enable colorbar on figure plot
        
        kwargs (dict): Parameters to matplotlib.pyplot.imshow function

    Returns:
        None
    """
     
    frequencies = som_model.activation_response(data).astype(int)

    im = plt.imshow(frequencies, **kwargs)
    if use_colorbar: plt.colorbar(im)
    
    for (i, j), value in np.ndenumerate(frequencies):
        plt.text(j, i, value, verticalalignment='center', 
                              horizontalalignment='center')


def heatmap(som_model, feature_names, grid_spec, use_colorbar=True, **kwargs):
    """Plot Self-organizing map heatmap
    
    Args:
        som_model (minisom.MiniSom): MiniSom Model
        
        feature_names (list): list of feature names
        
        grid_spec (tuple): tuple with grid plot dimensions
        
        use_colorbar (bool): Flag to enable colorbar on figure plot
  
        kwargs (dict): Parameters to matplotlib.pyplot.imshow function

    Returns:
        None
    """

    weights = som_model.get_weights()
    
    for i, fname in enumerate(feature_names):
        plt.subplot(*grid_spec, i + 1)
        plt.title(fname)
        im = plt.imshow(weights[:, :, i], **kwargs)
        
        if use_colorbar: plt.colorbar(im)
