import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from numpy.random import default_rng
from scipy.optimize import curve_fit, minimize


# Lets make some functions to reduce boiler plate code
random_gen = default_rng(868686)
dist_list = [x for x in dir(random_gen) if not x.startswith('_')]
dist_cache = {}

def random_dist(dist_name, dist_args=[], use_cached=False, **kwargs):
    """
    Dynamically creates distribution objects from the generator
    object, reduces the amount of boiler plate code


    :param dist_name: Input function name for distribution
    :param dist_args: Passed to distribution constructor
    :param use_cached: Every time a distribution is created, it is put into a cache. This tells the funciton to return the cached function
    :param kwargs: Passed to distribution constructor

    :return dist_obj: Output Distribution all built
    """
    if use_cached and dist_name in dist_cache:
        return dist_cache[dist_name]
    if dist_name in dist_list:
        dist_obj = getattr(random_gen, dist_name)(*dist_args, **kwargs)
        dist_cache[dist_name] = dist_obj
        return dist_obj
    raise AttributeError(f"Unknown Distribution {dist_name}, please select from possible list: {dist_list}")

axes_keywords = [
    "xlabel", "xlim", "xmargin", "xscale",
    "xticklabels", "xticks", "ybound",
    "ylabel", "ylim", "ymargin", "yscale",
    "yticklabels", "yticks", "zorder", "title"
]

axes_override = ['xlabel', 'ylabel', 'title']


def plot_make(plot_args, plot_type, figure=None, **kwargs):
    """
    Builds Plots easily

    :param in_data: Input data for future plot
    :param plot_type: Which plot type is this function building
    :param kwargs: Passed to plot constructor

    :return plot_obj: Plot constructor return
    """

    # Parse kwargs
    axes_kwargs = {}
    plot_kwargs = {}

    for keyword, value in kwargs.items():
        if keyword in axes_keywords:
            axes_kwargs[keyword] = value
        else:
            plot_kwargs[keyword] = value

    # Check for inputted figure
    if figure is None:
        figure = plt.figure()

    # Utilize set_* functions
    axes_over_args = dict([(override, axes_kwargs.pop(override)) for override in axes_override if override in axes_kwargs])

    # Build Axes Object
    axis_obj = plt.axes(**axes_kwargs)

    # Parse override arguments
    for key, value in axes_over_args.items():
        set_args = []
        set_kwargs = {}
        if isinstance(value, list):
            for arg in value:
                if isinstance(arg, dict):
                    set_kwargs.update(arg)
                else:
                    set_args.append(arg)
        else:
            set_args.append(value)
        getattr(axis_obj, "set_" + key)(*set_args, **set_kwargs)

    # Put it all together to make a plot
    figure.add_axes(axis_obj)
    plt_return_values = getattr(axis_obj, plot_type)(*plot_args, **plot_kwargs)

    # Return all the relevant objects
    return (figure, axis_obj, plt_return_values)


# Helper Function to build the distribution plots

def make_info(dist_name, dist_print_name, dist_args=[]):
    dist_size_list = (2000 / (np.array(range(1, 100, 2)))).astype(int)
    dist_mean_list = np.zeros(len(dist_size_list))

    for index, size in enumerate(dist_size_list):
        test_sample = random_dist(dist_name, dist_args, size=size)
        dist_mean_list[index] = np.mean(test_sample)

    fig = plt.figure(figsize=(10, 10))
    fig, ax, rv = plot_make(
        [dist_mean_list],
        'hist',
        bins=6,
        figure=fig,
        xlabel=['Sample Mean', {'fontsize': 18}],
        ylabel=['Number of Means in Bin', {'fontsize': 18}],
        title=[f'Histogram of Sample Means for \nthe {dist_print_name} Distribution with Variable Sample Sizes', {'fontsize': 18}]
    )

    ax.tick_params(labelsize=15)

    return (fig, ax, rv)

dist_rv_cache = []


rv = make_info("normal", "Normal", [100])
dist_rv_cache.append(list(rv))

rv = make_info("poisson", "Poisson", [100])
dist_rv_cache.append(list(rv))

rv = make_info("binomial", "Binomial", [100, 0.5])
dist_rv_cache.append(list(rv))

rv = make_info("hypergeometric", "Hypergeometric", [100, 50, 25])
dist_rv_cache.append(list(rv))

def convert_to_x(in_bin_arr):
    return [in_bin_arr[i] + abs(in_bin_arr[i] - in_bin_arr[i+1])/2 for i in range(len(in_bin_arr) - 1)]

def gaus_fn(x, height, std_dev, mean):
    x = np.array(x)
    return height * 0.3989422804 * 1/std_dev * np.exp(np.square((x - mean)/std_dev) / -2)


fig = plt.figure(figsize=(10, 10))

plt_colors = ['#4392F1', '#DC493A', '#373F51', '#FFEEDB']

ax_index = 1
for old_fig, old_ax, dist_rv in dist_rv_cache:
    ys = list(dist_rv[0])
    xs = convert_to_x(dist_rv[1])

    inital_guess = (100.0, 1000.0, 1000.0)

    print(xs)
    print(ys)
    params, covar = curve_fit(gaus_fn, xs, ys, method='dogbox', p0=inital_guess)
    print(params)

    in_xs = np.linspace(min(xs) - 1.0, max(xs) + 1.0, 100)
    fitted_ys = gaus_fn(in_xs, *params)

    new_ax = fig.add_subplot(2, 2, ax_index)
    new_ax.plot(in_xs, fitted_ys, color=plt_colors[ax_index - 1])
    for bar in dist_rv[2]:

        cen = bar.get_xy()
        wid = bar.get_width()
        hei = bar.get_height()

        new_ax.add_artist(mpl.patches.Rectangle(cen, wid, hei))

    ax_index += 1

plt.show()
