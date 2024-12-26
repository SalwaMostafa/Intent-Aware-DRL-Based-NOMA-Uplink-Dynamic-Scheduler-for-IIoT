import matplotlib as mpl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.lines as mlines
import numpy as np
from colorsys import rgb_to_hls

def draw_boxplot(data, color, saturation=.75, **kws):
    colors = sns.color_palette([color], desat=saturation)
    rgb_colors = sns.color_palette(colors)
    # Determine the gray color to use for the lines framing the plot
    light_vals = [rgb_to_hls(*c)[1] for c in rgb_colors]
    lum = min(light_vals) * .6
    gray = mpl.colors.rgb2hex((lum, lum, lum))

    color = rgb_colors[0]
    props = {}
    for obj in ["box", "whisker", "cap", "median", "flier"]:
        props[obj] = kws.pop(obj + "props", {})        
    artist_dict = plt.boxplot(data,patch_artist=True,**kws)
    
    for box in artist_dict["boxes"]:
        box.update(dict(facecolor=color,zorder=.9,edgecolor=gray))
        box.update(props["box"])
    for whisk in artist_dict["whiskers"]:
        whisk.update(dict(color=gray,linestyle="-"))
        whisk.update(props["whisker"])
    for cap in artist_dict["caps"]:
        cap.update(dict(color=gray))
        cap.update(props["cap"])
    for med in artist_dict["medians"]:
        med.update(dict(color=gray))
        med.update(props["median"])
    for fly in artist_dict["fliers"]:
        fly.update(dict(markerfacecolor=gray,marker="d",markeredgecolor=gray))
        fly.update(props["flier"])    
    y_min, y_max = plt.ylim()
    
    return y_max

def draw_brace(ax, xspan, yy, text):
    """Draws an annotated brace on the axes."""
    xmin, xmax = xspan
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan/xax_span*100)*2+1 # guaranteed uneven
    beta = 300./xax_span # the higher this is, the smaller the radius
    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:int(resolution/2)+1]
    y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                    + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = yy + (.05*y - .01)*yspan # adjust vertical position
    ax.autoscale(False)
    ax.plot(x, y, color='black', clip_on=False, alpha=0.5, linestyle='-')
    ax.text((xmax+xmin)/2., yy+.07*yspan, text, ha='center', va='bottom')

def set_style():
    nord_palette = ["#5e81ac", "#b48ead", "#a3be8c", "#bf616a", "#ebcb8b", "#d08770", "#81a1c1", "#8fbcbb"]
    colors = sns.color_palette(nord_palette)
    sns.reset_orig()
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.axisbelow"] = True
    #plt.style.use(['science','ieee'])
    #plt.style.use(['science','no-latex'])
    plt.rcParams["figure.figsize"] = (6,4)
    plt.rcParams["figure.dpi"] = 500
    # sns.set_style("whitegrid")
    matplotlib.rc('font',**{'family':'serif','serif':['Times New Roman','Bitstream Vera Serif','Computer Modern Roman',
                            'New Century Schoolbook','Century Schoolbook L','Utopia','ITC Bookman','Bookman',
                            'Nimbus Roman No9 L','Times','Palatino','Charter','DejaVu Serif','serif']})
    return colors
    
def set_fonts():
    
    SMALL_SIZE = 14
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 17
    plt.rc('font', size=SMALL_SIZE)            # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)      # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)      # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)     # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)     # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)               # legend fontsize
    #plt.rc('title', fontsize=BIGGER_SIZE+1)    # legend fontsize