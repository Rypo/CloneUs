import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md

import seaborn as sns

def plot_activity(df_chat, year=None):
    #sns.set_style("whitegrid")
    dfc = df_chat.copy()
    if year is not None:
        dfc = dfc[dfc.Date.between(str(year), str(year+1))]
    
    fig,ax = plt.subplots(figsize=(12,6))
    ax = sns.histplot(dfc, x=dfc.timexbin, bins=dfc.timexbin.dt.time.nunique(), hue='Author', hue_order=df_chat.Author.unique(), element="step", kde=True, ax=ax)
    ax.xaxis.set_major_formatter(md.DateFormatter('%I:%M %p'))
    ax.xaxis.set_major_locator(md.HourLocator(interval=1))
    
    ax.set_xlim(datetime.datetime(1970,1,1,7,0),datetime.datetime(1970,1,2,7,0))
    #ax.set_xticklabels(ax.get_xticklabels(), rotation=90);
    ax.grid(alpha=0.3)
    ax.set(title=f'GSChat Activity {year}')
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    return fig,ax


def plot_yearly(df_chat):
    # https://seaborn.pydata.org/examples/kde_ridgeplot.html
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    # Create the data
    dfc = df_chat.assign(year=df_chat.timebin.dt.year)[['Author','timexbin','year']]
    
    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(dfc, col="year", col_wrap=2, hue="Author", aspect=3, height=4, sharey=False)
    
    # Draw the densities in a few steps
    #g.map(sns.histplot, "timexbin", clip_on=False, fill=True, alpha=1, linewidth=1.5)#, bw_adjust=.5,)
    #g.map(sns.histplot, "timexbin", clip_on=False, color="w", lw=2)#, bw_adjust=.5)
    g.map(sns.histplot, 'timexbin', bins=dfc.timexbin.dt.time.nunique(), element="step", kde=True, alpha=0.2)
    # passing color=None to refline() uses the hue mapping
    #g.refline(y=0, linewidth=2, linestyle="-",  clip_on=False) # color=None,
    
    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        #print(color, label)
        ax = plt.gca()
        ax.grid(alpha=0.3)
        #ax.text(0.2, 0.2, '{row_name}', fontweight="bold")
        # ax.text(0, .2, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)
    
    g.map(label, "timexbin")
    g.add_legend()
    # Set the subplots to overlap
    #g.figure.subplots_adjust(hspace=-.25)
    
    plt.gca().xaxis.set_major_formatter(md.DateFormatter('%I:%M %p'))
    plt.gca().xaxis.set_major_locator(md.HourLocator(interval=1))
    plt.gca().set_xlim(datetime.datetime(1970,1,1,7,0),datetime.datetime(1970,1,2,6,0))
    # Remove axes details that don't play well with overlap
    g.set_titles(col_template='{col_name}')
    g.set(ylabel="") # yticks=[],
    g.despine(bottom=True)#, left=True)
    g.fig.autofmt_xdate(rotation=45)
    g.tight_layout()
    return g