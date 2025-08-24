import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

CMAPS = ['Pastel1','Accent','Paired', 'spring','viridis', 'plasma', 'magma', 'cividis']

def get_variable_dim(x):
    if isinstance(x, np.ndarray):
        return x.ndim
    return 0

def plot_4D_tensor(tensor, figsize=(12, 8)):
    """
    Visualize a 4D tensor as a grid of 2D slices.

    Args:
        tensor (np.ndarray): The 4D tensor to be visualized.
        figsize (tuple, optional): The size of the figure (width, height).

    Note:
        The dimensions of the tensor should be in the order (batch_size, height, width, num_channels).

    Example:
        input_data = np.random.randint(0, 10, size=(1, 2, 3, 1))
        plot_4D_tensor(input_data)
    """
    # Choose the dimensions to visualize as rows and columns
    batch_dim = 0  # Dimension index for the batch size
    channel_dim = -1  # Dimension index for the channel count (last dimension)

    # Get the number of rows and columns
    batch_size = tensor.shape[batch_dim]
    num_channels = tensor.shape[channel_dim]

    # Create a figure to plot each grid (2D slice)
    fig, axes = plt.subplots(num_channels, batch_size, figsize=figsize)
    
    # determine the dimension of axes whether it is 2D, 1D or scalar
    axes_dim = get_variable_dim(axes)
    
    
    # Loop through rows (channels) and columns (batch)
    for batch in range(batch_size):
        for channel in range(num_channels):
            # Extract the 2D slice from the 4D tensor
            grid_data = tensor[batch, :, :, channel]
            
            # Determine the appropriate subplot for the grid
            if axes_dim == 0:
                ax = axes
            elif axes_dim == 1:
                ax = axes[max(batch, channel)]
            elif axes_dim == 2:
                ax = axes[channel, batch]
            
            # Plot the 2D slice
            ax.imshow(grid_data, cmap=CMAPS[channel%len(CMAPS)])
            for (i, j), z in np.ndenumerate(grid_data):
                ax.text(j, i, '{}'.format(int(z)), ha='center', va='center')
            ax.set_title(f'Batch: {batch}, channel: {channel}')
            ax.axis('off')  # Turn off axis labels

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
