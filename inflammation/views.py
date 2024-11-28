"""Module containing code for plotting inflammation data."""

from matplotlib import pyplot as plt


def visualize(data_dict):
    """Display plots of basic statistical properties of the inflammation data.

    :param data_dict: Dictionary of name -> data to plot
    """
    # TODO(lesson-design) Extend to allow saving figure to file

    num_plots = len(data_dict)
    fig = plt.figure(figsize=((3 * num_plots) + 1, 3.0))

    for i, (name, data) in enumerate(data_dict.items()):
        axes = fig.add_subplot(1, num_plots, i + 1)

        axes.set_ylabel(name)
        axes.plot(data)

        # Add grid lines
        axes.grid(True)

        # Add title
        axes.set_title(name)

        # Add x and y axis labels
        axes.set_xlabel('X')
        axes.set_ylabel('Y')

        # Customize tick labels
        axes.tick_params(axis='x', rotation=0)
        axes.tick_params(axis='y', rotation=0)

    fig.tight_layout()

    plt.show()


def plot_std_data(daily_standard_deviation):
    """Plots the standard deviation by day between datasets."""
    graph_data = {
        'standard deviation by day': daily_standard_deviation,
    }
    visualize(graph_data)
