import numpy as np

import sklearn.datasets

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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


def umatrix_labeled(som_model, 
                    data, 
                    labels, 
                    colors, 
                    markers, 
                    use_colorbar=True,
                    plot_lbl_args=None,
                    **kwargs):
    """Plot a U-Matrix with labels in each pixel
    
    Args:
        som_model (minisom.MiniSom): MiniSom Model
        
        data (np.ndarray): n-dimensional data to estimulate neurons and generate
        activation map
        
        labels (np.ndarray): 1-dimensional data with label of data. Must be a
        discrete values (e. g. 0, 1, 2, ...)
        
        colors (list): List of color to use in each class
        
        markers (list): List of markers to use in each class
        
        use_colorbar (bool): Flag to enable colorbar on figure plot
        
        plot_lbl_args (dict): Parameters to matplotlib.pyplot.plot function used
        on plot labels
        
        kwargs (dict): Parameters to matplotlib.pyplot.imshow function
    Returns:
        None
    """
    
    if not plot_lbl_args:
        plot_lbl_args = {
            'markerfacecolor': 'None',
            'markersize': 12,
            'markeredgewidth': 2
        }
    
    im = plt.imshow(som_model.distance_map(), **kwargs)
    if use_colorbar: plt.colorbar(im)
    
    for idx, de in enumerate(data):
        label_idx = labels[idx] - 1
        
        winner = som_model.winner(de)
        plt.plot(winner[1], winner[0], 
                 markers[label_idx], 
                 markeredgecolor=colors[label_idx],
                 **plot_lbl_args)


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


def grid_pie_labeled(som_model, data, labels_name) -> tuple:
    """
    
    Args:
        som_model (minisom.MiniSom): MiniSom Model
        
        data (np.ndarray): n-dimensional data to estimulate neurons and generate
        activation map
        
        labels_name (np.ndarray): 1-dimensional data with name of data label. Must be a
        string values (e. g. 'classe1', 'classe2', ...)
    
    Returns:
        tuple: Tuple with patches and texts used in each plotted neuron
    """
    
    patches, texts = None, None
    labels_map = som_model.labels_map(data, labels_name)
    n_neurons, m_neurons = som_model.get_weights().shape[0:2]


    grid_spec = gridspec.GridSpec(n_neurons, m_neurons, plt.gcf())
    for position in labels_map.keys():
        label_fracs = [labels_map[position][l] for l in labels_name]
        plt.subplot(grid_spec[n_neurons-1-position[1],
                             position[0]], aspect=1)
        patches, texts = plt.pie(label_fracs)

    return (patches, texts)
