import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
from numba import cuda
from typing import List
from pathlib import Path
from os.path import exists
from kneed import KneeLocator
from collections import Counter
from itertools import cycle, chain
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, TextBox
from matplotlib.animation import FuncAnimation, PillowWriter
from dataclasses import dataclass, field
from rich import print as pprint
from rich.tree import Tree
from rich.text import Text
from rich.table import Table
from rich.theme import Theme
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn
)
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, stft, welch, convolve, butter, filtfilt, savgol_filter
from scipy.stats import wasserstein_distance, pearsonr, probplot, boxcox, yeojohnson, norm, linregress

########################### Custom imports ###############################
from utils import segment_ECG
from setup_globals import walk_directory
from support import logger, console, log_time, NumpyArrayEncoder

########################### Sklearn imports ###############################
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import accuracy_score as ACC_SC
from sklearn.metrics import log_loss as LOG_LOSS
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as RSQUARED
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import classification_report

########################### Sklearn model imports #########################
from sklearn.model_selection import KFold, StratifiedKFold, LeavePOut, LeaveOneOut, ShuffleSplit, StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC as SVM
from sklearn.ensemble import IsolationForest as IsoForest
from xgboost import XGBClassifier, DMatrix

#CLASS EDA
class EDA(object):
    def __init__(
            self,
            pig_data:dict,
            channels:list,
            fs:float,
            gpu_devices:list,
            all_data:np.array,
            ecg_lead:int,
            lad_lead:int, 
            car_lead:int,
        ):
        self.dataset = pig_data
        self.feature_names = channels
        self.gpu_devices = gpu_devices
        self.fs = fs
        self.ecg_lead = ecg_lead
        self.lad_lead = lad_lead
        self.car_lead = car_lead
        self.task = "classification"
        self.all_data = all_data
        # self.target = pd.Series(target, name="ShockClass")
        self.target_names = ["baseline", "class_1", "class_2", "class_3", "class_4"]
        self.rev_target_dict = {
            0:"baseline",
            1:"class_1",
            2:"class_2",
            3:"class_3",
            4:"class_4"
        }

    #FUNCTION clean_data
    def clean_data(self):
        #Calculate necessary segment averages
        cols = ["Avg_QRS", "Avg_QT", "Avg_PR", "Avg_ST"]
        add_cols = ["qrs_comp", "pr_intr", "qt_intr", "st_seg"]
        #If you're looking at old data.
        if not np.all(np.isin(cols, self.data.columns.tolist())):
            for col in cols:
                self.data[col] = np.zeros(shape=(self.data.shape[0]))
            for idx in self.data.index:
                valid = self.data.iloc[idx, 3] == 1.0
                if valid:
                    star = self.data.iloc[idx, 1]
                    fini = self.data.iloc[idx, 2]
                    inners = self.interior_peaks[(self.interior_peaks["r_peak"] > star) & (self.interior_peaks["r_peak"] < fini)]
                    for cidx, col in enumerate(add_cols):
                        avg = np.nonzero(inners[col])[0]
                        if avg.shape[0] > 0:
                            avg_vec = inners.iloc[avg, self.names_interior.index(col)]
                            self.data.loc[idx, cols[cidx]] = round(np.mean(avg_vec), 2)
            
        #Drop the target column.
        self.data = self.data.drop("valid", axis=1)

    #FUNCTION Imputation
    def imputate(self, imptype:str, col:str):
        """Function for imputing missing data.  
        Will be adding others in the future. 

        Args:
            imptype (str): Type of imputation you want
            col (str): column you want to perform it on
        """		
        if imptype == "mean":
            self.data[col].fillna(self.data[col].mean(), inplace=True)

        elif imptype == "median":
            self.data[col].fillna(self.data[col].median(), inplace=True)

        elif imptype == "mode":
            self.data[col].fillna(self.data[col].mode(), inplace=True)

    #FUNCTION drop_nulls
    def drop_nulls(self):
        """
        Null dropping routine

        """		
        #!Caution
        #Use at your own risk.  this will drop all rows with a null. (switch axis=1 if you want by all columns)
        logger.info(f'Shape before drop {self.data.shape}')
        self.data = self.data.dropna(axis=0, subset=self.data, how='any')
        logger.info(f'Shape after drop {self.data.shape}')
    
    #FUNCTION print_nulls
    def print_nulls(self, plotg=False):
        """Checks for nan's.  Color codes output to verify if over 30% of the
        data is missing.  Prints results to a Rich Table

        Args:
            plotg (bool, optional): Whether or not to plot the missing values
            Defaults to False.
        """		
        #column | nulls | % of Total | Over 30 % Nulls
        #str    | int   | float      | Boolean

        #Setting theme levels
        custom_theme = Theme(
            {"kindagood":"white on blue",
             "danger":"red on white"
            }
        )

        #Adding in a Rich Table for printing.
        table = Table(title="Null Report")
        table.add_column("Column", justify="right", no_wrap=True)
        table.add_column("Null Count", justify="center")
        table.add_column("Null %", justify="center")
        table.add_column("Over 30%", justify="center")

        for x in range(0, len(self.data.columns), 20):
            subslice = self.data.iloc[:, x:x+20]
            cols = list(subslice.columns)
            nulls = subslice.isnull().sum()
            perc = round(nulls / subslice.shape[0], 2)
            over30 = [True if x > .30 else False for x in perc]
            end_sect = False
            for ss in range(len(cols)):
                if ss == len(cols)-1:
                    end_sect = True
                if not over30[ss]:
                    table.add_row(
                        cols[ss], 
                        f"{nulls.iloc[ss]:.0f}", 
                        f"{perc.iloc[ss]:.2%}", 
                        f"{str(over30[ss])}", 
                        style="kindagood", 
                        end_section=end_sect
                    )
                else:
                    table.add_row(
                        cols[ss], 
                        f"{nulls.iloc[ss]:.0f}", 
                        f"{perc.iloc[ss]:.2%}", 
                        f"{str(over30[ss])}", 
                        style="danger", 
                        end_section=end_sect
                    )
        console.log("Printing Null Table")
        console.print(table)

        if plotg:
            console.log("plotting null visualization")
            #Print of a chart of the NA values.  Solid black square = no nulls.
            plt.figure(figsize=(10, 8))
            plt.imshow(self.data.isna(), aspect="auto", interpolation="nearest", cmap="gray")
            plt.xlabel("Column Number")
            plt.ylabel("Sample Number")
            plt.title("Null Visualization")
            plt.show()
            plt.close()

    #FUNCTION sum_stats
    def sum_stats(self, stat_list:list, title=str):
        """Accepts a list of features you want to be summarized. 
        Manipulate the .agg function below to return your desired format.

        Args:
            stat_list (list): List of feature names
            title (str): What you want to call the plot
        """		
        #Add a rich table for results. 
        table = Table(title=title)
        table.add_column("Measure Name", style="green", justify="right")
        table.add_column("mean", style="sky_blue3", justify="center")
        table.add_column("std", style="turquoise2", justify="center")
        table.add_column("max", style="yellow", justify="center")
        table.add_column("min", style="gold3", justify="center")
        table.add_column("count", style="cyan", justify="center")

        for col in stat_list:
            _mean, _stddev, _max, _min, _count = self.data.loc[self.data[col] != 0, col].agg(["mean", "std", "max", "min", "count"]).T
            table.add_row(
                col,
                f"{_mean:.2f}",
                f"{_stddev:.2f}",
                f"{_max:.2f}",
                f"{_min:.2f}",
                f"{_count:.0f}",
            )
        console.log(f"printing table for features :\n{stat_list}")
        console.print(table)

    #FUNCTION num_features
    def num_features(self, plotg:bool=False, print_stats:bool=True):
        """ isolates numeric features and does a quick plot of them. 		

        Args:
            plotg (bool, optional): Whether to plot. Defaults to False.
            print_stats (bool, optional): Whether to show table stats. Defaults to False.
        """		
        self.num_df = self.data.select_dtypes(include='number')

        #Add a rich table for results. 
        table = Table(title="Num Feature Report")
        table.add_column("Measure Name", style="green", justify="right")
        table.add_column("mean", style="sky_blue3", justify="center")
        table.add_column("std", style="turquoise2", justify="center")
        table.add_column("max", style="yellow", justify="center")
        table.add_column("min", style="gold3", justify="center")
        table.add_column("count", style="cyan", justify="center")
        table.add_column("nulls\nfound?", justify="center", no_wrap=False)
        # self.num_df.iloc[:, idx:idx+20].agg(["count", np.mean, np.std, max, min]).T

        if print_stats:
            for idx in range(0, self.num_df.shape[1], 40):
                subslice = self.num_df.iloc[:, idx:idx+40]
                cols = list(subslice.columns)
                for ss in range(len(cols)):
                    if ss == len(cols)-1:
                        end_sect = True
                    else:
                        end_sect = False
                    _count, _mean, _stddev, _max, _min = subslice.iloc[:, ss].agg(["count", np.nanmean, np.nanstd, max, min]).T
                    if _count == subslice.shape[0]:
                        nulls = "[bold green]No"
                    else:
                        nulls = "[bold red]Yes"
                    table.add_row(
                        cols[ss],
                        f"{_mean:.2f}",
                        f"{_stddev:.2f}",
                        f"{_max:.2f}",
                        f"{_min:.2f}",
                        f"{_count:.0f}",
                        nulls,
                        end_section=end_sect
                    )
                    if end_sect:
                        colnames = [table.columns[x].header for x in range(len(table.columns))]
                        table.add_row(
                            colnames[0],
                            colnames[1],
                            colnames[2],
                            colnames[3],
                            colnames[4],
                            colnames[5],
                            colnames[6],
                            style="white",
                            end_section=end_sect
                        )
                    
            # logger.info(self.num_df.iloc[:, idx:idx+20].agg(["count", np.mean, np.std, max, min]).T)
            console.log("Printing Num Feature Table")
            console.print(table)
        if plotg:
            for idx in range(0, len(self.num_df.columns), 40):
                self.num_df.iloc[:, idx:idx+40].plot(
                    lw=0, 
                    marker=".", 
                    subplots=True, 
                    layout=(-1, 3),
                    figsize=(12, 12), 
                    markersize=8
                )
                plt.tight_layout()
                plt.show()

    # FUNCTION cat_features
    # def cat_features(self, plotg:bool=False, print_stats:bool=True):
    # 	""" isolates categorical features and does a quick plot of them

    # 	Args:
    # 		plotg (bool, optional): Whether to plot. Defaults to False.
    # 		print_stats (bool, optional): Whether to show table stats. Defaults to False.
    # 	"""
    # 	self.cat_df = self.data.select_dtypes(exclude=['number', 'datetime'])

    # 	# if plotg:
    # 	# 	for x in range(0, len(self.data.columns), 20):
    # 	# 		self.cat_df.iloc[:, idx:idx+40].plot(
    # 	# 								lw=0, marker=".", 
    # 	# 								subplots=True, layout=(-1, 3),
    # 	# 								figsize=(12, 12), markersize=3
    # 	# 		)
    # 	# 		plt.show()
    # 	# 		print('\n\n')
    # 	#IDEA - Cat plotting

    # 	if print_stats:
    # 		#Add a rich table for results. 
    # 		table = Table(title="Cat Feature Report", expand=True)
    # 		table.add_column("Col Name", style="green", justify="right")
    # 		table.add_column("count", style="cyan", justify="center")
    # 		table.add_column("unique", style="sky_blue3", justify="center")
    # 		table.add_column("mf val", style="yellow", justify="center", no_wrap=False)
    # 		table.add_column("mf count", style="gold3", justify="center")
            
    # 		for idx in range(0, self.cat_df.shape[1], 40):
    # 			subslice = self.cat_df.iloc[:, idx:idx+40]
    # 			cols = list(subslice.columns)
    # 			for ss in range(len(cols)):
    # 				if ss == len(cols) - 1:
    # 					end_sect = True
    # 				else:
    # 					end_sect = False
    # 				_count, _unique, _tval, _tfreq = subslice.iloc[:, ss].describe().T
    # 				table.add_row(
    # 					f'{cols[ss]}',
    # 					f'{_count}',
    # 					f'{_unique}', 
    # 					f'{_tval}',
    # 					f'{_tfreq}',
    # 					end_section = end_sect
    # 				)
    # 				if end_sect:
    # 					colnames = [table.columns[x].header for x in range(len(table.columns))]
    # 					table.add_row(
    # 						colnames[0],
    # 						colnames[1],
    # 						colnames[2],
    # 						colnames[3],
    # 						colnames[4],
    # 						style="white",
    # 						end_section=end_sect)

    # 		console = Console()
    # 		console.log("Printing Cat Feature Table")
    # 		console.print(table)

    # FUNCTION heatmap
    def corr_heatmap(self, sel_cols:list):
        """Generates correlation heatmap of numeric variables.

        Args:
            sel_cols (list): columns you want to submit for a heatmap
        """		
        #! Up to the user to submit the correct columns for heatmaps (ie - numeric)
        #if you didn't select any columns, it will select all the numeric
        #columns for you. 

        if not sel_cols:
            sel_cols = self.data.select_dtypes(include='number')

        self.num_corr = self.data[sel_cols].corr(method="spearman") 

        #!Caution
        #If you want to automate null dropping  you can do so below		
        #Find the corr cols that are null
        # more_drop_cols = list(self.num_corr.isnull().columns)

        #Drop em!
        # self.num_df.drop(more_drop_cols, axis=1, inplace=True)

        #Make correlation chart
        plt.figure(figsize=(16, 12))
        mask = np.triu(np.ones_like(self.num_corr, dtype=bool))
        heatmap = sns.heatmap(self.num_corr,
            mask=mask,
            vmin=-1, 
            vmax=1, 
            annot=True, 
            annot_kws={
                'fontsize':10,
            },
            fmt='.1f',
            cmap='RdYlGn')
        heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':16}, pad=12)
        plt.show()
        plt.close()

    #FUNCTION eda_plot
    def eda_plot(self, 
        plot_type:str="histogram",
        feat_1:str=False, 
        feat_2:str=False, 
        group:str=False,
        ):
        """Basic plotting method for EDA class

        Plot types:
        - scatterplot
        - histogram
        - jointplot
        - pairplot

        To use the plotting method, try something like this.

        `explore.eda_plot("scatterplot", feat_1, feat_2, group)`

        The last variable, group, is what controls groupings.  Submit a
        categorical column, it will group the chart accordingly. Submit
        False, and it will just give you the standard chart.

        Args:
            plot_type(str): type of plot you want graphed. 
            feat_1 (str or bool): feature of interest (col name)
            feat_2 (str or bool, optional): feature of interest (col name)
            group (str or bool, optional): Column you want to group on. Usually
                a categorical. Defaults to False.
        """		

        #quick correlation
        if not isinstance(feat_1, bool) and not isinstance(feat_2, bool):
            self.corr = self.data[feat_1].corr(self.data[feat_2])
            logger.warning(f'correlation of {feat_1} and {feat_2}: {self.corr:.2f}')

        #Generates repeatable colordict for all values in the group
        if not isinstance(group, bool):
            #Get the size of the target responses. (how many are there)
            num_groups = np.unique(group).size
                #NOTE, best to use MPL's sequential colormaps as the ligthness is
                #monotonically increasing in intensity of the color
                #Avail cmaps here.  
                #https://matplotlib.org/stable/tutorials/colors/colormaps.html
            #Get the colormap
            color_cmap = mpl.colormaps["Paired"]
            #Generate a hex code for the color.  
            color_str = [mpl.colors.rgb2hex(color_cmap(i)) for i in range(num_groups)]
            #Now make a dictionary of the activities and their hex color.
            colcyc = color_str[:num_groups]
            cycol = cycle(colcyc)
            group_color_dict = {x:next(cycol) for x in np.unique(group)}

            ###### 
            #if the target isn't in the data, this add's it to a temp dataframe
            #for hue/group mapping
            #BUG - This code smells here.  come back and rewrite
                #Solutions:
                #1. change the dataset source (for all plots mind you)
                    #to the self.dataset.
                        #Wouldn't give you updated nulls if you dropped any)
                        #I'd like to keep that original dataset in tact if possible. 
                #2. Use the weird logic i came up with below.  It works ok. 
                    #ideaflow: iF the grouped target var isn't in the datset (ie
                    #the target), it adds it back in so i can assign color hue's
                    #(usually for classification)
                #3. Maybe brainstorm with Tom tomorrow about solutions here. 
                    #nothing coming to mind right away because I want to keep the original 
                    #dataset untouched.  

            hue_col = group.name

            if hue_col not in self.data.columns.tolist():
                _comb_df = pd.concat([self.data, group], axis=1)
            else:
                _comb_df = self.data
        else:
            _comb_df = self.data
        
        cur_col_idx = _comb_df.columns.tolist()
        if plot_type == "scatter":
            logger.info(f'plotting scatterplot\nfor {feat_1} and {feat_2}')
            fig = plt.figure(figsize=(8, 8))
            if isinstance(group, bool):
                assert _comb_df[feat_1].shape[0] == _comb_df[feat_2].shape[0], "Shape of feat_1 and feat_2 dont match"
                plt.scatter(
                    _comb_df[feat_1], 
                    _comb_df[feat_2],
                    )
                plt.title(f'Scatter of {feat_1} by {feat_2}')
            else:
                for grp in group_color_dict.keys():
                    #indexes for the group of eval. use that for indexing the feat_2 array
                    idxmask = np.where(_comb_df[hue_col]==grp)[0]
                    if hue_col == self.target.name:
                        plt.scatter(
                            _comb_df.iloc[idxmask, cur_col_idx.index(feat_1)], 
                            _comb_df.iloc[idxmask, cur_col_idx.index(feat_2)], 
                            c=group_color_dict[grp],
                            label=self.rev_target_dict[grp]
                        )
                    else:
                        plt.scatter(
                            _comb_df.iloc[idxmask, cur_col_idx.index(feat_1)], 
                            _comb_df.iloc[idxmask, cur_col_idx.index(feat_2)], 
                            c=group_color_dict[grp],
                            label=grp
                        )
                plt.title(f'Scatterplot {feat_1} by {feat_2} grouped by {hue_col}')
                plt.legend(loc='upper right')

            plt.xlabel(f'{feat_1}')
            plt.ylabel(f'{feat_2}')
            plt.show()
            plt.close()

        #If its a histogram
        if plot_type == "histogram":
            fig, (ax_hist, ax_box) = plt.subplots(
                2, 
                sharex=True, 
                figsize=(10, 8), 
                gridspec_kw={"height_ratios": (.85, .15)}
            )
            if not isinstance(group, bool):
                if hue_col == self.target.name:
                    group_color_dict = {self.rev_target_dict[k]:v for k, v in group_color_dict.items()}
                    sns.histplot(
                        data = _comb_df, 
                        x=feat_1, 
                        ax=ax_hist, 
                        hue=group.map(self.rev_target_dict),
                        palette=group_color_dict,
                        multiple='stack'
                        )
                else:
                    sns.histplot(
                        data = _comb_df, 
                        x=feat_1, 
                        ax=ax_hist, 
                        hue=group,
                        palette=group_color_dict,
                        multiple='stack'
                        )
                logger.info(f'plotting histogram for\n{feat_1} grouped by {hue_col}')
            else:
                sns.histplot(
                    data = _comb_df, 
                    x=feat_1, 
                    ax=ax_hist, 
                    )
                logger.info(f'plotting histogram for\n{feat_1}')

            sns.boxplot(data = _comb_df, x = _comb_df[feat_1], ax=ax_box)
            ax_hist.set_title(f'Histogram/Boxplot of {feat_1}')
            ax_hist.set_xlabel(f'Distribution of {feat_1}')
            ax_hist.set_ylabel('Count')
            ax_box.set_xlabel('')
            plt.show()
            plt.close()

        #If its a pairplot
        if plot_type == "pairplot":
            #select all columns in groups of 6 (visually any more and the plot becomes too crowded)
                #Problem is that it screws with the logic of the pairplot
            for colnum in range(0, _comb_df.shape[1]-1, 6): 
                cols = _comb_df.iloc[:, colnum:colnum+6].columns.tolist()
                #BUG Code smells here .  must fix
                if (hue_col in cols) and (hue_col == self.target.name):
                    cols.pop(cols.index(hue_col))
                if not isinstance(group, bool):
                    label_list = sorted(Counter(group).keys())
                    pg = sns.PairGrid(
                        data = _comb_df, 
                        vars = cols,
                        hue = hue_col,
                        hue_order = label_list,
                        palette = group_color_dict,
                        diag_sharey = False, 
                    
                        #old code from pairplot
                        # hue_order=np.unique(group),
                        # kind='reg',
                        # diag_kind='kde',
                        # plot_kws={
                        # 	'color':group_color_dict.values(),
                        # 	'line_kws':{'color':'red'}
                        # 	},
                        # diag_kws={
                        # 	'color':group_color_dict
                        # }
                        # height=10,
                        # aspect=5
                    )
                    pg.map_diag(sns.histplot, multiple="stack", element="step")
                    pg.map_offdiag(sns.scatterplot)
                    #Had to make a custom legend because sns legend being annoying
                    legend_elements = [
                        Line2D([0], [0], 
                         marker = 'o', 
                        color = 'w', 
                        label = val[0],
                        markerfacecolor = val[1], 
                        markersize = 10) for val in group_color_dict.items()
                    ]
                    pg.fig.get_axes()[-1].legend(
                        handles=legend_elements,
                        loc='upper right', 
                        # bbox_to_anchor = (0.98, 0.15),
                        fancybox=True,
                        shadow=True)

                    logger.info(f'plotting pairplot for\n{cols}\ngrouped by {hue_col}')
                else:
                    sns.pairplot(
                        data = _comb_df.iloc[:, colnum:colnum+6], 
                        kind='reg',
                        diag_kind='kde',
                        # diag_kws={'color':'dodgerblue'}
                        plot_kws={'color':'blue','line_kws':{'color':'red'}},
                        # height=10,
                        # aspect=5
                        )
                    logger.info(f'plotting pairplot for\n{cols}')
                plt.show()
                plt.close()

            # ax_hist.set_xlabel(f'Distribution of {feat_1}')
            # ax_hist.set_ylabel('Count')

        #If its a jointplot
        if plot_type == "jointplot":
            if not isinstance(group, bool):
                # mapped_df = _comb_df.copy()
                if hue_col == self.target.name:
                    group_color_dict = {self.rev_target_dict[k]:v for k, v in group_color_dict.items()}
                    # inv_t_dict = {v: k for k, v in self.target_dict.items()}
                    # mapped_df[hue_col] = mapped_df[hue_col].map(inv_t_dict)
                    # group_color_dict = {inv_t_dict[int(k)]:v for k, v in group_color_dict.items()}
                    label_list = self.target_names
                    hue_target = _comb_df[hue_col].map(self.rev_target_dict)
                else:
                    label_list = sorted(Counter(group).keys())
                    hue_target = _comb_df[hue_col]

                logger.info(f'plotting jointplot for\n{feat_1} and {feat_2}\ngrouped by {hue_col}')
                sns.jointplot(
                    data = _comb_df,
                    x = feat_1, 
                    y = feat_2,
                    hue = hue_target,
                    kind = 'scatter',
                    hue_order = label_list,
                    palette=group_color_dict,
                    s = 50
                    )
                for label, color in group_color_dict.items():
                    sns.regplot(
                        data = _comb_df.iloc[np.where(hue_target==label)[0], :],
                        x = feat_1, 
                        y = feat_2,
                        color=color,
                        label=label
                    )

            else:
                logger.info(f'plotting jointplot for\n{feat_1} and {feat_2}')
                sns.jointplot(
                    data = _comb_df,
                    x = feat_1, 
                    y = feat_2,
                    kind = 'reg',
                    space = 0,
                    )
            plt.show()
            plt.close()

#CLASS Feature Engineering
class FeatureEngineering(EDA):
    def __init__(self, ecg_data:dict, wave:np.array):
        #Inherit from the EDA class
        super().__init__(ecg_data, wave)
        EDA.clean_data(self)
        # EDA.drop_nulls(self)
        
        """		
        Inputs:
            data = pd dataframe of the feature columns
            target = pd series of the target column
            target_names = 	1D np.array of targets
            feature_names = 1D np.array of column names. 
            data_description = str description of the dataset.(Print with sep="\n")
            file_name = filename for when you want to export results. 
        Args:
            task (str): machine learning task you want to implement
            dataset (dict): sklearns dictionary import of varibles/names/filenames
        """	

    #FUNCTION cos_sim
    def get_cos_sim(
        self, 
        vec1:list, 
        vec2:list, 
        impute:str="None",
        dtype:str="float"
                ):
        """Calculates cosine similarity of two vectors. 
            -Will dilineate between a text cos sim and 
            numerical cos sim by the dtype argument

        Args:
            vec1 (list): array of first vector. 
            vec2 (list): array of second vector. 
            impute (str): Whether or not to impute the values with mean
            dtype (str): What type of cos sim you want to run

        Returns:
            float: Return the cosine similarity of both vectors
        """
        if dtype == 'str':
            from sklearn.feature_extraction.text import TfidfVectorizer

        else:
            if impute:
                v1mean = np.nanmean(vec1)
                v2mean = np.nanmean(vec2)
                vec1 = np.where(np.isnan(vec1), v1mean, vec1)
                vec2 = np.where(np.isnan(vec2), v2mean, vec2)

            dp = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return dp / (norm1 * norm2)


    # #FUNCTION categorical_encoding
    #def categorical_encoding(self, enc:str, feat:str, order:list=None):
    #     """Note, can only drop single columns at a time. I'll 
    #     Eventually make it do multiple

    #     Args:
    #         enc (str): _description_
    #         feat (str): _description_
    #         order (list, optional): _description_. Defaults to None.

    #     Raises:
    #         ValueError: _description_
    #     """		
    #     if isinstance(enc, str):
    #         enc_dict = {
    #             "onehot" :OneHotEncoder(
    #                 categories="auto",
    #                 drop=None,
    #                 sparse_output=False,
    #                 dtype=int,
    #                 handle_unknown="error",
    #                 min_frequency=None,
    #                 max_categories=None)
    #             ,
    #             "ordinal":OrdinalEncoder(
    #                 categories=[order],
    #                 dtype=np.float64,
    #                 # handle_unknown="error",
    #                 min_frequency=None,
    #                 max_categories=None)
    #         }
    #         encoder = enc_dict.get(enc)
    #         if not encoder:
    #             raise ValueError(f"Encoder not loaded, check before continuing")

    #         #Fit and transform the column (Needs to reshaped to 2d for transform)
    #         arr = encoder.fit_transform(self.data[feat].to_numpy().reshape(-1, 1))
    #         if enc == "onehot":
    #             #grab columns 
    #             ndf_cols = encoder.categories_[0].tolist()
    #             ndf_cols = ["oh_" + x for x in ndf_cols]
    #             #Add to dataset
    #             new_df = pd.DataFrame(arr, index = self.data.index, columns=ndf_cols)
    #             #Add colnames to feature_names
    #             self.feature_names.extend(ndf_cols)
            
    #         elif enc == "ordinal":
    #             #Make trans col name
    #             nn = "ord_" + feat
    #             new_df = pd.DataFrame(arr, index = self.data.index, columns=[nn])
    #             self.feature_names.append(nn)

    #         #Add the new col/cols
    #         self.data = pd.concat([self.data, new_df], axis=1)

    #         #Drop said feature of transform from dataset
    #         self.data.drop([feat], axis=1, inplace=True)
    #         self.feature_names.pop(self.feature_names.index(feat))
    #         logger.info(f"Feature: {feat} has been encoded with {encoder.__class__()} ")

    #FUNCTION engineer
    def engineer(self, features:list, transform:bool, display:bool, trans:str):
        """Feature Engineering function.  This function allows you to explore 
        individual column transformations.  

        Args:
            features (list): list of str (or one str) features you want to transform
            transform (bool): Whether you want to transform the column and drop the original column
            display (bool): boolean of if logger should show the transform
            trans (str):  What type of transformation you want

        """
        def transform_col(self, feature:str, trans:str):
            if trans == "log":
                tran_col = np.log(self.data[feature])

            elif trans == "recip":
                tran_col = np.reciprocal(self.data[feature])

            elif trans == "sqrt":
                tran_col = np.sqrt(self.data[feature])

            #FIXME
            #removing for short term.  Getting wierd errors. 
            # elif trans == "exp":
            # 	tran_col = np.exp(self.data[feature])

            elif trans == "BoxC":
                tran_col = boxcox(self.data[feature])
                tran_col = pd.Series(tran_col[0])

            elif trans == "YeoJ":
                tran_col = yeojohnson(self.data[feature])
                tran_col = pd.Series(tran_col[0])
        
            return tran_col
        
        def probability_plot(self, col_name:str, trans_name:str):
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize = (10, 8))
            probplot(
                self.data[col_name], 
                dist=norm,
                plot=ax1
            )
            probplot(
                self.data[trans_name], 
                dist=norm,
                plot=ax2
            )
            ax1.set_title(f'Probability plot\n{col_name}')
            ax2.set_title(f'Probability plot\n{trans_name}')
            plt.show()

        target = False

        #If its a single feature, it will come through as a string
        if isinstance(features, str):
            #If single column.  Repeat without the list comp
            #FIXME - Find a cleaner way to do this.  Really don't like 
            #repeating code more than once.  
            feature = features
            trans_col = transform_col(self, feature, trans)
            trans_name = trans + "_" + feature 
            self.data[trans_name] = trans_col.values

            if not transform and display:
                #we don't want to store the data, but we'd like to look at
                # it. Calls EDA histogram plot from EDA class and plots
                #then drops the column out
                logger.info(f'Distribution before {trans} transform\nfor {feature}')
                super().eda_plot("histogram", feature, False, target)
                logger.info(f'Distribution after {trans} transform\nfor {feature}')
                super().eda_plot("histogram", trans_name, False, target)
                logger.info(f'Probability plot for\n{feature}\nbefore-after\n{trans} transformation')
                probability_plot(self, feature, trans_name)
                
                self.data.drop(columns=trans_name, inplace=True)
                
            elif transform:
                #If we want charts, show them.  If we're running a ton of
                #models and don't want to see it everytime, set to false
                if display:
                    logger.info(f'Distribution before transform for {feature}')
                    super().eda_plot("histogram", feature, False, target)
                    logger.info(f'Distribution after transform for {feature}')
                    super().eda_plot("histogram", trans_name, False, target)
                    logger.info(f'Probability plot for\n{feature}\nbefore-after\n{trans} transformation')
                    probability_plot(self, feature, trans_name)

                self.data.drop(columns=feature, inplace=True)
                self.feature_names.pop(self.feature_names.index(feature))
                self.feature_names.append(trans_name)

        # if its a list, then it will transform each feature
        elif isinstance(features, list):
            for feature in features:
                trans_col = transform_col(self, feature, trans)
                trans_name = trans + "_" + feature 
                self.data[trans_name] = trans_col.values

                if not transform and display:
                    #we don't want to store the data, but we'd like to look at
                    # it. Calls EDA histogram plot from EDA class and plots
                    #then drops the column out
                    logger.info(f'Distribution before transform for {feature}')
                    super().eda_plot("histogram", feature, False, target)
                    logger.info(f'Distribution after transform for {feature}')
                    super().eda_plot("histogram", trans_name, False, target)
                    logger.info(f'Probability plot for\n{feature}\nbefore-after\n{trans} transformation')
                    probability_plot(self, feature, trans_name)

                    self.data.drop(columns=trans_name, inplace=True)
                    
                elif transform:
                    #If we want charts, show them.  If we're running a ton of
                    #models and don't want to see it everytime, set to false
                    if display:
                        logger.info(f'Distribution before transform for {feature}')
                        super().eda_plot("histogram", feature, False, target)
                        logger.info(f'Distribution after transform for {feature}')
                        super().eda_plot("histogram", trans_name, False, target)
                        logger.info(f'Probability plot for\n{feature}\nbefore-after\n{trans} transformation')
                        probability_plot(self, feature, trans_name)

                    self.data.drop(columns=feature, inplace=True)
                    self.feature_names.pop(self.feature_names.index(feature))
                    self.feature_names.append(trans_name)

        else:
            logger.warning(f'{trans} not found in transformers. Check and try again')

#CLASS DataPrep
class DataPrep(object):
    def __init__(self, features:list, scaler:str=False, cross_val:str=False, engin:object=False):  
        """This is the initalization in between feature engineering and modeltraining.
        
        Logic:
            1. Loads empty dict's for _performance, _predictions, and _models
            2. if the feature engineering object (engin) is passed into the intialization
               constructer, It will pull the features you've generated from there. 
            3. if it is submitted without an object, the data will inherit all data / dataset
               description data form the EDA class.  Allowing flexibility in modeling with
               and without engineered features. 

        Args:
            features (list):List of features you want to model
            scaler (str, optional):Type of scaler you want used. Defaults to False
            cross_val (str, optional):Cross validation scheme you want.  Defaults to False
            engin (object, optional):Feature Engineering object. Defaults to False.
        """		
        self._performance = {}
        self._predictions = {}
        self._models = {}
        self._traind = {}
        self.scaled = False
        self.scaler = scaler
        self.cross_val = cross_val
        
        if engin:
            self.data = engin.data[features]
            self.feature_names = features
            self.target = engin.target
            self.target_names = engin.target_names
            self.task = engin.task
        else:
            EDA.__init__(self) #BUG - I need a way to feed the data in here
            EDA.clean_data(self)
            EDA.drop_nulls(self)
            self.data = self.data[features]
            self.feature_names = features

        logger.info(f"Modeling task: {self.task}")
        logger.info(f'Dataset Shape:{self.data.shape}')
        logger.info(f'Dataset features:{self.feature_names}')
        logger.info(f'Dataset target:\t{self.target.name}')

    #FUNCTION dataprep
    def data_prep(
            self, 
               model_name:str, 
            split:float,
            model_category:str=None, 
            category_value:str=None
        ):
        """Prepares the DataPrep object to accept model parameters and categories
        Logic:
            1. Sets the split
            2. Sets the empty dictionaries for the _models, _predictions, _performance
            3. Sets the X and y for features and target respectively
            4. Scales the data if the algorithm calls for it.
            5. Performs the split of test and train datasets

        Args:
            model_name (str): abbreviated name of the model
            split (float): What test train split % that you want. (input = decimal %)
            scale (str): What scaler to use. 
            cross_val (str): Cross validation scheme
            model_category (str, optional): Grouping (categorical) of model target you'd like. Defaults to None.
            category_value (any, optional): What value you're targeting. Defaults to None.
        """		
          
        self.split = split
        self.model_category = {}
        self._performance[model_name] = {}
        self._predictions[model_name] = {}
        self._models[model_name] = {}
        self._traind[model_name] = {}

        if model_category != None and category_value != None:	
            self.model_category[model_name] = model_category
            self.category_value[model_name] = category_value

            #If the category's doesn't exist in the model results.  Add them. 
            if category_value not in self._predictions: #BUG <---.keys() maybe? 
                self._predictions[model_name][category_value]= {}
            if category_value not in self._performance:
                self._performance[model_name][category_value] = {}
            if category_value not in self._models:
                self._models[model_name][category_value] = {}

            self.data_cat = self.data[self.data[model_category] == category_value]
            self._traind[model_name]["X"] = self.data_cat[self.feature_names]
            self._traind[model_name]["y"] = self.data_cat[self.target]
            
        else:
            self.category_value = None
            self._traind[model_name]["X"] = self.data[self.feature_names]
            self._traind[model_name]["y"] = self.target

        #FUNCTION scalers
        #Models that don't need scaling
            #Tree-based algo's
            #Lda, NB 

        #Tips on choosing Scalers below
        #StandardScaler(with_mean=True)
            #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
            #Assumes normal distribution.  If not, needs a transformation to
            #a normal dist then, standardscaler.
            #sensitive to outliers. 
        
        #MinMaxScaler(feature_range=(0, 1))
            # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
            # sensitive to outliers. 
            
        #RobustScaler(quantile_range=(0.25, 0.75), with_scaling=True)
            # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler
            # Use this one if you've got outliers that you can't remove. (or alot of them)
            # Scales by quantile ranges

        #QuantileTransformer()
            # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#
            # Scales by transforming to a normal or uniform distribution
            # Nonlinear transformation, so it could distort correlations. 
            # Also known as a rankscaler. 

        #PowerTransformer()
            #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html#sklearn.preprocessing.PowerTransformer
            #parametric, monotonic transformation to fit a Gaussian Distribution
            #finds optimal scaling, for stabalizing variance and skewness. 
            #Supports box-cox (strictly positive) and yeo-johnson transforms (pos and neg values)

        if isinstance(self.scaler, str):
            scalernm = self.scaler
            scaler_dict = {
                "r_scale":RobustScaler(quantile_range=(0.25, 0.75), with_scaling=True),
            }
            scaler = scaler_dict.get(scalernm)
            if not scaler:
                raise ValueError(f"Scaler not loaded, check before continuing")
            
            self._traind[model_name]["X"] = scaler.fit_transform(self._traind[model_name]["X"], self._traind[model_name]["y"])
            self.scaled = True
            logger.info(f"{model_name}'s data has been scaled with {scaler.__class__()} ")
        

        #MEAS Test train split
        X_train, X_test, y_train, y_test = train_test_split(self._traind[model_name]["X"], self._traind[model_name]["y"], random_state=42, test_size=split)
        self._traind[model_name]["X_test"] = X_test
        self._traind[model_name]["y_test"] = y_test  
        self._traind[model_name]["X_train"] = X_train 
        self._traind[model_name]["y_train"] = y_train 

#CLASS Model Training
class ModelTraining(object):
    def __init__(self, dataprep):
        """This is the initalization for modeltraining.  It will inherit objects from the 
        
        Logic:
            1. Loads empty dict's for _performance, _predictions, and _models
            2. if the feature engineering object (engin) is passed into the intialization
               constructer, It will pull the features you've generated from there. 
            3. if it is submitted without an object, the data will inherit all
               data from the EDA class.  Allowing flexibility in modeling with
               and without engineered features. 

            Note:  This doesn't inherit automatically from the DataPrep class because you might
            create the DataPrep class with / or without engineered data.  If the DataPrep class is 
            initialized when empty, it will resort to the EDA class for its data
            sourcing, and thereby forefeit any engineered features from being
            input into the model. 
            
        Args:
            dataprep (object): dataprep class of how you want the data set up for modeling. 
        """
        
        self._models = dataprep._models
        self._predictions = dataprep._predictions
        self._performance = dataprep._performance
        self._traind = dataprep._traind
        self.category_value = dataprep.category_value
        self.feature_names = dataprep.feature_names
        self.split = dataprep.split
        self.target_names = dataprep.target_names
        self.task = dataprep.task
        self.cross_val = dataprep.cross_val
        self.CV_func = None
        self._model_params = {
            #MEAS Model params
            "isoforest":{
                "model_name":"isoforest",
                "model_type":"classification",
                "scoring_metric":"accuracy",
                #link to params
                #https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#
                "base_params":{
                    "n_estimators":100,				#int
                    "max_samples":"auto",			#int|float	
                    "contamination":"auto",         #auto|float
                    "max_features":1.0,    		    #int|float
                    "bootstrap":False,     		    #bool
                    "n_jobs":None,                  #int
                    "random_state":42,              #int
                    "warm_start":False              #bool
                },
                "init_params":{
                    "n_estimators":100,				#int
                    "max_samples":"auto",			#int|float	
                    "contamination":"auto",         #auto|float
                    "max_features":1.0,    		    #int|float
                    "bootstrap":False,     		    #bool
                    "n_jobs":None,                  #int
                    "random_state":42,              #int
                    "warm_start":False              #bool
                },
                "grid_srch_params":{
                    "criterion":["entropy", "gini"],
                    # "splitter":["best", "random"],
                    "max_depth":range(1, 100),
                    "min_samples_split":range(2, 25),
                    "min_samples_leaf":range(1, 20),
                }
            },
            "pca":{
                "model_name":"pca",
                "model_type":"classification",
                "scoring_metric":"accuracy",
                #link to params
                #https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
                #NOTE - Be sure to check docs here.  Different solvers work with
                #different penalties. 
                "base_params":{
                    "n_components":None,                #int | float			
                    "copy":True,                        #bool
                    "whiten":False,                     #bool	
                    "svd_solver":"auto",                #str | 'auto'			
                    "tol":0.0, 				            #float	
                    "iterated_power":"auto",            #int | 'auto'
                    "n_oversamples":10,                 #int
                    "power_iteration_normalizer":"auto",#str | 'auto'
                    "random_state":42                   #int | None
                },
                "init_params":{
                    "n_components":None,                #int | float			
                    "copy":True,                        #bool
                    "whiten":False,                     #bool	
                    "svd_solver":"auto",                #str | 'auto'			
                    "tol":0.0, 				            #float	
                    "iterated_power":"auto",            #int | 'auto'
                    "n_oversamples":10,                 #int
                    "power_iteration_normalizer":"auto",#str | 'auto'
                    "random_state":42                   #int | None
                },
                "grid_srch_params":{
                    # "n_neighbors":range(3, 20),
                    # "algorithm":["auto", "ball_tree", "kd_tree", "brute"],
                    # "leaf_size":range(1, 60),
                    # "metric":["cosine", "euclidean", "manhattan", "minkowski"]
                }
            },
            "svm":{
                #Notes. 
                    #
                "model_name":"svm",
                "model_type":"classification",
                "scoring_metric":"accuracy",
                #link to params
                #https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
                "base_params":{
                    "penalty":"l2",					#str
                    "loss":"squared_hinge",         #str
                    "dual":True,                    #
                    "C":1.0,						
                    "multi_class":"ovr",			
                    "fit_intercept":True,
                    "random_state":42,
                    "max_iter":1000
                },
                "init_params":{
                    "penalty":"l2",					#!MUSTCHANGEME
                    "loss":"squared_hinge",
                    "dual":False,
                    "C":0.1,						#!MUSTCHANGEME
                    "multi_class":"ovr",			#!MUSTCHANGEME
                    "fit_intercept":True,
                    "random_state":42,
                    "max_iter":1000
                },
                "grid_srch_params":{
                    "penalty":["l1","l2"],
                    "loss":["hinge","squared_hinge"],
                    "C":np.arange(0, 1.1, 0.1),
                    "max_iter":np.arange(1000, 10000, 500)
                }
            },
            "xgboost":{
                #Notes. 
                    #this model workflow to the others
                "model_name":"XGBClassifier()",
                "model_type":"classification",
                "scoring_metric":"accuracy",
                #link to params
                #https://xgboost.readthedocs.io/en/stable/parameter.html
                "base_params":{
                    "booster":"gbtree",
                    "gamma":0,
                    "max_depth":6,
                    "learning_rate": 0.3,
                    "nthread":4, 
                    "subsample":1,
                    "objective":"multi:softmax",
                    "n_estimators":1000,
                    "reg_alpha":0.3,
                    "num_class":2
                },
                "init_params":{
                    "booster":"gbtree",
                    "eval_metric":"error",
                    "max_depth":10,
                    "gamma":0,
                    "lambda":1,
                    "alpha":0,
                    "nthread":4,
                    "learning_rate": 0.1,
                    # "subsample":0.5,
                    "objective":"binary:hinge",
                    "n_estimators":100,
                    "reg_alpha":0.3,
                },
                "grid_srch_params":{
                    # "cv":5,
                    # "lambda":np.arange(0, 1.1, 0.1),
                    # "alpha":np.arange(0, 1.1, 0.1),
                    "learning_rate":np.arange(0, 1.1, 0.1),
                    # "max_depth":range(0, 26, 2),
                    # "subsample":np.arange(0.5, 1.1, 0.1)
                    # "loss":["hinge","squared_hinge"],
                    # "gamma":range(0, 100),
                    # "n_estimators":np.arange(0, 1000, 100)
                }
            }
        }

    #FUNCTION get_data
    def get_data(self, model_name:str):
        """Unpacks training and test data

        Args:
            model_name (str): Name of model
        """		
        self.X_train = self._traind[model_name]["X_train"]
        self.X_test = self._traind[model_name]["X_test"]
        self.y_train = self._traind[model_name]["y_train"]
        self.y_test = self._traind[model_name]["y_test"]
        self.X = self._traind[model_name]["X"]
        self.y = self._traind[model_name]["y"]

    #FUNCTION Load Model
    def load_model(self, model_name:str):
        """_summary_

        Args:
            model_name (str): _description_

        Returns:
            _type_: _description_
        """			
        params = self._model_params[model_name]['init_params']
        ####################  classification Models ##################### 
        match model_name:
            case 'pca':
                return PCA(**params)
            case 'svm':
                return SVM(**params)
            case 'isoforest':
                return IsoForest(**params)
            case 'xgboost':
                return XGBClassifier(**params)

    #FUNCTION models fit
    @log_time
    def fit(self, model_name:str):
        """This module handles the fit functions for each of the sklearn models. 
        Logic:\n
            1. Extracts model parameters from _model_params dictionary.\n
            2. Unpacks said dictionary, into the model being run.\n

        Args:
            model_name (str): abbreviated name of the model to run
            
        """

        #MEAS Model training \ Param loading
        ####################  Model Load  ##############################		
        self.model = ModelTraining.load_model(self, model_name)

        ####################  Fitting  ##############################
        logger.info(f'{model_name}: fitting model')
        
        #For super fun spinner action in your terminal.
            #Doesn't work in a notebook without ipywidgets, and even then it
            #doesn't look very good. 
        # progress = Progress(
        #     SpinnerColumn(
        #         spinner_name="pong",
        #         speed = 1.2, 
        #         finished_text="fit complete in",
        #     ),
        #     "time elapsed:",
        #     TimeElapsedColumn(),
        # )

        # with progress:
        #     task = progress.add_task("Fitting Model", total=1)
        #     self.model.fit(self.X_train, self.y_train)
        #     progress.update(task, advance=1)

        self.model.fit(self.X_train, self.y_train)
        
        if self.category_value != None:
            self._models[model_name][self.category_value] = self.model
        else:
            self._models[model_name] = self.model
        logger.info(f"fit complete for {model_name}")

    #FUNCTION predict
    def predict(self, model_name):
        """Fits the model in question
        Note:
            Will add an additional key for category if that is desired. 

        Args:
            model_name (str): abbreviated name of the model
        """		
        if self.category_value != None:
            self._predictions[model_name][self.category_value] = self._models[model_name][self.category_value].predict(self.X_test)
        else:
            self._predictions[model_name] = self._models[model_name].predict(self.X_test)
        
        logger.info(f'{model_name}: making predictions')
    
    #FUNCTION validate
    def validate(self, model_name):
        """This module handles the model metrics and which to run.   It
        summarizes model metrics and outputs them into a rich table as those
        look better than plain ol print statements.

        Logic:
            1. Pull out the parameters of the model it just ran. 
            2. Create a rich table for results storage. 
            3. Identify which task route to take for which metric to run.
                if Regressor
                    - Calculate desired metric
                    - Provide Model summary

                if Classification
                    - Generate a classification report, and confusion matrix.
                    - Format and print model results as a rich table.

        Args:
            model_name (str): Name of model
        """		

        #FUNCTION custom_confusion_matrix
        def custom_confusion_matrix(y_true, y_pred, display_labels=None):
            from sklearn.metrics import confusion_matrix
            """
            A function to plot a custom confusion matrix with
            positive class as the first row and the first column.
            """
            
            # Create a flipped matrix
            cm = np.flip(confusion_matrix(y_true, y_pred))

            #Lets make some variables.
            cats = ["True -", "False +", "False -", "True +"]
            counts = [f"{x:0.0f}" for x in cm.flatten()]
            perc = [f"{x / np.sum(cm):0.2%}" for x in cm.flatten()]
            labs = [f"{uno}\n{dos}\n{tres}" for uno, dos, tres in zip(cats, counts, perc)]

            labs = np.asarray(labs).reshape(2, 2)

            # Create the plot
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(
                cm, 
                ax=ax, 
                annot=labs, 
                fmt="", 
                annot_kws={
                    'fontsize':18,
                },
                cmap='Blues'
            )
            plt.title(f"{model_name.upper()} confusion matrix", fontsize=22)
            plt.xlabel("Predicted Label", fontsize=14)
            plt.ylabel("True Label", fontsize=14)
            plt.xticks(ax.get_xticks(), labels = display_labels, rotation=-30, fontsize=14)
            plt.yticks(ax.get_yticks(), labels = display_labels, rotation=-30, fontsize=14)
            plt.show()
            plt.close()	

        #FUNCTION classification_report
        def make_cls_report(y_true, y_pred, display_labels=None):
            report = classification_report(
                y_true, 
                y_pred, 
                labels = np.unique(y_pred), #BUG Still wary of using this. Look at this again to figure out a better way
                target_names=display_labels,
                zero_division=False
            ) 
            #BUG. Unrepresented classes throw a div by zero error.  Look into this later. 

            body = report.split("\n\n")
            header = body[0]
            rows = [body[x].split("\n") for x in range(1, len(body))]
            rows_flat = list(chain(*rows))
            table = Table(title = f'Classification report', header_style="Blue on white")
            table.add_column(header, justify='center', style='white on blue')
            for row in rows_flat:
                table.add_row(row)
            console.print(table)

        #FUNCTION ROC_AUC
        def roc_auc_curves(self, model:str):
            #Get the size of the target responses. (how many are there)
            num_groups = np.unique(self._traind[model]["y"]).size
            #Get the colormap
            color_cmap = plt.cm.get_cmap('Paired', num_groups) 
            #Generate a hex code for the color.  
            color_str = [mpl.colors.rgb2hex(color_cmap(i)) for i in range(color_cmap.N)]
            #Now make a dictionary of the activities and their hex color.
            colcyc = color_str[:num_groups]
            cycol = cycle(colcyc)
            group_color_dict = {x:next(cycol) for x in self.target_names}

            #Using one vs rest scheme for Aucroc
            test_prob = self._models[model].predict_proba(self._traind[model]["X_test"])
            fpr, tpr, auc_s = {}, {}, {}
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
            for cls in range(len(self.target_names)):  #BUG - Possible hardcoding here to a numbered target class. 
                fpr[cls], tpr[cls], _ = roc_curve(self._traind[model]["y_test"], test_prob[:, cls], pos_label=cls)
                auc_s[cls] = auc(fpr[cls], tpr[cls])
                # roc_auc_s[cls] = roc_auc_score(
                # 	y_true = self._traind[model]["y_test"],
                # 	y_score = test_prob[:, cls],
                # 	average = "macro",
                # 	multi_class = "ovr",
                # 	labels = cls
                # )
                plt.plot(
                    fpr[cls], 
                    tpr[cls], 
                    linestyle="--",
                    color = group_color_dict[self.target_names[cls]],
                    label = f"ROC curve for {self.target_names[cls]} vs rest AUC:{auc_s[cls]:.2f}" 
                )

            plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
            plt.title(f'{model.upper()} ROC Curve')
            plt.legend(loc="lower right")
            plt.show()
            plt.close()

        def cv_roc_auc_curves():
            pass

        #FUNCTION classification summary
        def classification_summary(model_name:str, y_pred:np.array, cv_class:str=False):
        #######################Confusion Matrix and classification report##########################
            labels = self.target_names
            no_proba = ["svc", ""]
            #Call confusion matrix
            logger.info(f'{model_name} confusion matrix')
            custom_confusion_matrix(self.y_test, y_pred, display_labels=labels)
            #Call classification report
            logger.info(f'{model_name} classification report')
            make_cls_report(self.y_test, y_pred, display_labels=labels)
            
            #Generate ROC curves for non CV runs. 
            if not cv_class:
                if model_name not in no_proba:
                    roc_auc_curves(self, model_name)
            else:
                # cv_roc_auc_curves() #Not finished
                logger.warning(f"ROC Curves not yet functional for CV")
            
        #FUNCTION No Crossval
        def no_cv_scoring(y_pred:np.array, cat_bool:bool, table)->float:
            #I'm not sure why i'm keeping no cross validation as an option, but
            #here we are. 
            scoring_dict = {
                #regression
                # "rsme"    : MSE(self.y_test, y_pred, squared=False),
                # "mse"     : MSE(self.y_test, y_pred),
                # "mae"     : MAE(self.y_test, y_pred),
                # "rsquared": RSQUARED(self.y_test, y_pred),
                #classification
                "accuracy": ACC_SC(self.y_test, y_pred),
                "logloss" : LOG_LOSS(self.y_test, y_pred)
                #clustering
            }
            if self.task == "regression":
                logger.info(f'{model_name}: Calculating {metric} for {self.task}')
                if cat_bool:
                    self._performance[self.category_value][model_name][metric.upper()] = scoring_dict[metric]
                    scores = self._performance[self.category_value][model_name][metric.upper()]
                    table.add_column(f'{scores:^.2f}', justify="center", style="white on blue")
                    
                else:
                    self._performance[model_name][metric.upper()] = scoring_dict[metric]
                    scores = self._performance[model_name][metric.upper()]
                    table.add_column(f'{scores:^.2f}', justify="center", style="white on blue")

            if self.task == "classification":
                logger.info(f'{model_name}: Calculating {metric} for {self.task}')
                if cat_bool:
                    self._performance[self.category_value][model_name][metric.upper()] = scoring_dict[metric]
                    scores = self._performance[self.category_value][model_name][metric.upper()]
                    table.add_column(f'{scores:^.2%}', justify="center", style="white on blue")
                    
                else:	
                    self._performance[model_name][metric.upper()] = scoring_dict[metric]
                    scores = self._performance[model_name][metric.upper()]
                    table.add_column(f'{scores:^.2%}', justify="center", style="white on blue")
                    
                classification_summary(model_name, y_pred)
                return scores
            
            else:

                return None

        #FUNCTION With Cross Validation
        def cv_scoring(y_pred:np.array, cat_bool:bool, model:str, table)->float:
            """_summary_

            Args:
                y_pred (np.array): _description_
                cat_bool (bool): _description_
                model (str): _description_
                table (_type_): _description_

            Returns:
                float: _description_
            """
            def load_cross_val(cv_name:str):
                cv_validators = {
                    "kfold"       :KFold(n_splits=10, shuffle=True, random_state=42),
                    "stratkfold"  :StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    "leaveoneout" :LeaveOneOut(),
                    # "leavepout"   :LeaveOneOut(p=2),
                    "shuffle"     :ShuffleSplit(n_splits=10, test_size=0.25, train_size=0.5, random_state=42),
                    "stratshuffle":StratifiedShuffleSplit(n_splits=10, test_size=0.25, train_size=0.5, random_state=42)
                }
                return cv_validators[cv_name]
            
            def generate_cv_predictions(freshmodel, CV_func, X_data:np.array, y_data:np.array):#-> Tuple[list, list]
                actual_t = np.array([])
                predicted_t = np.array([])

                for train_ix, test_ix in CV_func.split(X = X_data, y = y_data):
                    train_X, train_y, test_X, test_y = X_data[train_ix], y_data[train_ix], X_data[test_ix], y_data[test_ix]
                    freshmodel.fit(train_X, train_y)
                    predicted_labels = freshmodel.predict(test_X)
                    predicted_t = np.append(predicted_t, predicted_labels)
                    actual_t = np.append(actual_t, test_y)

                return predicted_t, actual_t

            #Load a fresh untrained model and score it.
            freshmodel = ModelTraining.load_model(self, model_name)

            #Load Cross Validation
            CV_func = load_cross_val(self.cross_val)

            #Validate
            scores = cross_validate(freshmodel, self.X_train, self.y_train.to_numpy(), cv=CV_func)["test_score"]
                #BUG - Don't i need to eval on the test set? for above?

            #reload model untrained model for cross_validation predictions
            freshmodel = ModelTraining.load_model(self, model_name)

            #Load Cross Validation
            CV_func = load_cross_val(self.cross_val)

            #Generate new predictions based on cross validated data.
            y_pred, y_target = generate_cv_predictions(freshmodel, CV_func, self.X_train, self.y_train.to_numpy())
            
            #Store them in the modeltraining object
            self._predictions[model_name] = y_pred
            self.y_test = y_target
        
            #IDEA
                #Do we want a permutation test at the end of cross validation to
                #see if the distributions changed? aka did the model find any
                #real relation to the inputs

                #? Two fold inner and outer CV? Make a custom scorer??

            if cat_bool:
                self._performance[self.category_value][model_name][metric.upper()] = scores
            else:
                self._performance[model_name][metric.upper()] = (scores.mean(), scores.std())

            #Add them to the table. 
            table.add_column(f'{scores.mean():^.2f}', justify="center", style="white on blue")
            
            if self.task == "classification":
                logger.info(f'{model_name}: Calculating model summary')
                classification_summary(model_name, y_pred, True)
                
            return scores.mean()


        ###################### METRIC CENTRAL ##################################################
        metric = self._model_params[model_name]["scoring_metric"]
        #Grab the model parameters used.
        params = {k:v for k, v in self.model.get_params().items()}
        #Make a results table
        table = Table(title = str(self.model.__class__).split(" ")[1].split(".")[-1].rstrip(">'"), header_style="white on blue")
        table.add_column(metric.upper(), justify="right", style="white on blue")

        #Grab predictions
        cat_bool = self.category_value != None
        if cat_bool:
            y_pred = self._predictions[self.category_value][model_name]
        else:
            y_pred = self._predictions[model_name]
        
        if self.cross_val:
            #Call cross validation function
            scores = cv_scoring(y_pred, cat_bool, model_name, table)
            table.add_row("CV:", f"{self.cross_val}", end_section=True)

        else:
            #Call regular holdout scoring function
            scores = no_cv_scoring(y_pred, cat_bool, table)
            table.add_row("Test holdout", f"{self.split:.0%}", end_section=True)
        

        #Add the model parameters to the table
        table.add_row("Params:", "", end_section=True)
        [table.add_row(k, str(v)) for k, v in params.items()]

        #Print them to the console
        logger.info(f'{model_name}: results \U00002193 \U0001f389')
        console.print(table)

        #TODO.  Add in prec/recall chart as found here. 
        #https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_display_object_visualization.html#sphx-glr-auto-examples-miscellaneous-plot-display-object-visualization-py

    #FUNCTION show_results
    def show_results(self, modellist:list, sort_des:bool=False):
        #Make a results table
        #Structure will be:
        #modelname | metric | score

        table = Table(title = "Model Results Table", header_style="white on blue")
        table.add_column("model name", justify="right", style="white on blue")
        table.add_column("metric", justify="center", style="white on blue")
        table.add_column("score", justify="center", style="white on blue")

        cat_bool = self.category_value != None
        #TODO - cat_bool + Metric units
            #Code in for category selection here too. 
            #Also need to reformat below to not be hardecoded to task. 
                #temp fix for now, but ultimately i'd it to format the metric
                #as it is supposed to be reported. 
        _templist = []
        for model in modellist:
            model_name = str(self._model_params[model]['model_name'])[:-2]
            metric = self._model_params[model]['scoring_metric']
            score = self._performance[model][metric.upper()]
            if self.task == "regression":
                score = f'{score :.2f}'
            elif self.task == "classification":
                if self.cross_val:
                    score, std = score[0], score[1]
                    score = f'Mean: {score:.2%} +/-:{std:.2%}'
                else:
                    score = f'{score:.2%}'
                    
            _templist.append((model_name, metric.upper(), score))
            
        if sort_des:
            _templist = sorted(_templist, key=lambda x: x[2], reverse=True)
        else:
            _templist = sorted(_templist, key=lambda x: x[2], reverse=False)

        [table.add_row(model_name, metric, score) for (model_name, metric, score) in _templist]
        logger.info(f'Model results \U00002193 \U0001f389')
        console.print(table)

    #FUNCTION importance plot
    def plot_feats(self, model:str, features:list, imps:list):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (10, 8))
        feat_imp = sorted(zip(features, imps), key=lambda x: -x[1])[:20]
        dfeats = pd.DataFrame(data = feat_imp, columns=["Name", "Imp"])
        plt.barh(
            y=dfeats["Name"], 
            height=0.8,
            width=dfeats["Imp"],
        )
        ax.invert_yaxis()
        plt.title(f"{model} Top 20 feature importance", fontsize=14)
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature Name")
        plt.show()

    #FUNCTION _grid_search
    @log_time
    def _grid_search(self, model_name:str, folds:int):
        from sklearn.model_selection import GridSearchCV
        logger.info(f'{model_name} grid search initiated')
        clf = self._models[model_name]
        params = self._model_params[model_name]["grid_srch_params"]
        metric = self._model_params[model_name]["scoring_metric"]
        grid = GridSearchCV(clf, param_grid=params, cv = folds, scoring=metric)
        
        # For super fun spinner action in your terminal.
        # progress = Progress(
        #         SpinnerColumn(
        #             spinner_name="shark",
        #             speed = 1.2, 
        #             finished_text="searching parameters",
        #         ),
        #         "time elapsed:",
        #         TimeElapsedColumn(),

        #     )
        # with progress:
        #     task = progress.add_task("Fitting Model", total=1)
        #     grid.fit(self.X_train, self.y_train)
        #     progress.update(task, advance=1)

        grid.fit(self.X_train, self.y_train)
        logger.info(f"{model_name} best params\n{grid.best_params_}")
        logger.info(f"{model_name} best {metric}: {grid.best_score_:.2%}")
        fp = "./data/datasets/JT/gridresults.txt"
        t = time.localtime()
        current_time = time.strftime("%m-%d-%Y %H:%M:%S", t)
        #Check to see if the file can be opened.
        if exists(fp):
            #If it exists, append to it.
            with open(fp, "a") as savef:
                savef.write(f"\n\nGridsearch ran on {current_time}\n")
                savef.write(f"Model {model_name} using {grid.cv} folds\n")
                savef.write(f"score:\n{grid.best_score_}\n")
                savef.write(f"parameters:\n{grid.best_params_}")
        else:
            #If it doesn't, make a new file
            with open(fp, "w") as savef:
                savef.write(f'Gridsearch ran on {current_time}\n')
                savef.write(f"Model {model_name} using {grid.cv} folds\n")
                savef.write(f"score:\n{grid.best_score_}\n")
                savef.write(f"parameters:\n{grid.best_params_}")

        return grid

# --- Wavelet / Phase Calculation Logic  ---
class CardiacPhaseTools:
    """Helper class for Phase Variance calculations."""
    def __init__(self, fs=1000, bandwidth_parameter=8.0):
        self.fs = fs
        self.c = bandwidth_parameter

    def complex_morlet_cwt(self, data, center_freq):
        """Performs CWT and returns envelope and phase.

        Args:
            data (np.array): view of the signal
            center_freq (int): Main Frequency to focus on

        Returns:
            envelope, phase: _description_
        """

        w_desired = 2 * np.pi * center_freq
        s = self.c * self.fs / w_desired
        M = int(2 * 4 * s) + 1
        t = np.arange(-M//2 + 1, M//2 + 1)
        norm = 1 / np.sqrt(s)
        wavelet = norm * np.exp(1j * self.c * t / s) * np.exp(-0.5 * (t / s)**2)
        cwt_complex = convolve(data, wavelet, mode='same')
        return np.abs(cwt_complex), np.angle(cwt_complex)

    def compute_continuous_phase_metric(self, signal, window_beats=10) -> np.array:
        """Generates a continuous time-series metric representing phase stability.
        1. Finds Peaks. (scipy)
        2. Segments Signal.
        3. Calculates Phase Variance across a rolling window of beats.

        Args:
            signal (np.array): Signal you want to look at
            window_beats (int, optional): default number of beats. Defaults to 10.

        Returns:
            metric_curve (np.array): Chunked array of the phase variance over time
        """

        # 1. Find Peaks
        #TODO - peak params
            # This could be improved upon - ie parameter adjustments

        peaks, _ = find_peaks(signal, distance=int(self.fs * 0.4), height=np.mean(signal))
        if len(peaks) < window_beats:
            return np.zeros_like(signal)

        metric_curve = np.zeros_like(signal, dtype=float)
        
        # We will use a rolling window of beats to calculate stability
        # Beat window size (fixed for alignment)
        beat_win = int(0.6 * self.fs) # 600ms
        pre_peak = int(0.2 * self.fs)
        
        # Center frequency for analysis (High freq usually shows jitter best)
        target_freq = 20 #30 
        
        # Pre-allocate beat segments
        beats = []
        valid_peaks = []
        
        for p in peaks:
            start = p - pre_peak
            end = start + beat_win
            if start >= 0 and end < len(signal):
                beats.append(signal[start:end])
                valid_peaks.append(p)
        
        beats = np.array(beats)
        n_beats = len(beats)
        
        if n_beats < window_beats:
            return metric_curve

        # Iterate through beats with rolling window
        half_w = window_beats // 2
        
        # Progress bar since this can be slow
        with Progress(
            SpinnerColumn(), 
            BarColumn(), 
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("Calculating Phase Variance...", total=n_beats)
            for i in range(n_beats):
                # Define rolling window indices
                start_b = max(0, i - half_w)
                end_b = min(n_beats, i + half_w)
                batch = beats[start_b:end_b]
                
                if len(batch) < 3: 
                    continue
                
                # --- Phase Variance Calculation ---
                # 1. CWT on batch
                phases = []
                envelopes = []
                for b in batch:
                    env, phi = self.complex_morlet_cwt(b, target_freq)
                    phases.append(phi)
                    envelopes.append(env)
                
                phases = np.array(phases)
                
                # 2. Mean Phase
                mean_phase = np.mean(phases, axis=0)
                
                # 3. Variance of deviations
                phase_dev = phases - mean_phase
                
                # Wrap phase differences to [-pi, pi] for correct variance
                phase_dev = (phase_dev + np.pi) % (2 * np.pi) - np.pi
                
                # Variance across the beat (time axis)
                phase_var_curve = np.var(phase_dev, axis=0)
                
                # 4. Collapse to scalar (Mean Variance during QRS complex)
                # We focus on the center 100ms where QRS is
                center_idx = len(phase_var_curve) // 2
                qrs_region = phase_var_curve[center_idx - 50 : center_idx + 50]
                scalar_score = np.mean(qrs_region)
                
                # Fill the metric curve for the duration of this R-R interval
                # (From current peak to next peak)
                current_p = valid_peaks[i]
                next_p = valid_peaks[i+1] if i < n_beats - 1 else len(signal)
                
                metric_curve[current_p : next_p] = scalar_score
                progress.advance(task)

        return metric_curve

# --- Data Loader ---
class SignalDataLoader:
    """Handles loading and structuring the NPZ data."""
    def __init__(self, file_path):
        self.file_path = str(file_path)
        if self.file_path.endswith("npz"):
            self.container = np.load(self.file_path)
            self.files = self.container.files
            self.channels = self._identify_and_sort_channels()
            self.full_data = self._stitch_blocks()
        elif self.file_path.endswith("pkl"):
            self.container = np.load(self.file_path, allow_pickle=True)
            self.full_data = self.container.to_dict(orient="series")
            self.channels = self.container.columns.to_list()
            if "ShockClass" in self.channels:
                self.outcomes = self.full_data.pop("ShockClass")
                self.channels.pop(self.channels.index("ShockClass"))
            else:
                self.outcomes = None
        
    def _identify_and_sort_channels(self):
        """
        Identifies unique channel names from NPZ keys and returns them 
        in a deterministic (alphabetical) order.
        """
        raw_names = set()
        
        for k in self.files:
            # Extract channel name from keys like 'ECG_block_1', 'HR_block_0'
            if '_block_' in k:
                name = k.split('_block_')[0]
                raw_names.add(name)
            else:
                # Catch-all for keys that don't follow the block naming convention
                raw_names.add(k)
        
        # Sort alphabetically to ensure the plot labels consistently map to the data indices
        return sorted(list(raw_names))
    
    def _stitch_blocks(self):
        full_data = {}
        for ch in self.channels:
            # Filter keys for this channel and sort by block index
            ch_blocks = sorted(
                [k for k in self.files if k.startswith(f"{ch}_block_")], 
                key=lambda x: int(x.split('_block_')[-1])
            )
            
            if ch_blocks:
                full_data[ch] = np.concatenate([self.container[b] for b in ch_blocks])
            else:
                # Fallback: if no blocks found, maybe it's a single file entry
                if ch in self.files:
                    full_data[ch] = self.container[ch]
        return full_data

# --- Advanced Viewer ---
class RegimeViewer:
    """
    Interactive viewer for signal, Semantic Segmentation via minimum CAC (Corrected Arc Curve), and Phase Variance.
    Includes frequency analysis and custom navigation.
    """
    def __init__(
        self, 
        signal_data  :np.array, 
        cac_data     :np.array, 
        regime_locs  :np.array, 
        m            :int, 
        sampling_rate:float=1000.0, 
        lead         :str='Carotid (TS420)'
        ):
        """
        Args:
            signal_data (np.array): np.array of the signal data
            cac_data (np.array): np.array of CAC curve data
            regime_locs (np.array): np.array of regime change locations
            m (int): stumpy search window width
            sampling_rate (float): in Hz. Defaults to 1000.0
            lead (str): Signal being analyzed
        """        
        # 1. Data Setup
        self.signal = signal_data
        self.cac = cac_data
        self.regime_locs = regime_locs
        self.m = m
        self.fs = sampling_rate
        self.lead = lead        

        # Calculate Phase Variance Stream
        console.print("[cyan]Pre-computing Phase Variance Stream...[/]")
        self.ptools = CardiacPhaseTools(fs=self.fs)
        self.phase_var_stream = self.ptools.compute_continuous_phase_metric(self.signal)
        
        # 2. State Settings
        self.window_size = 10_000
        self.current_pos = 0
        self.step_size = 20
        self.paused = False
        
        # Frequency State: 0=Off, 1=Stem, 2=Specgram
        self.freq_mode = 0 
        
        # 3. Setup Figure
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click_jump)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.setup_layout()
        self._init_axes_pool()
        
        # 4. Start Animation
        self.ani = FuncAnimation(
            self.fig, self.update_frame, interval=30, blit=False, cache_frame_data=False
        )
        plt.show()

    def setup_layout(self):
        """Define GridSpec layout."""
        self.gs_main = gridspec.GridSpec(1, 2, width_ratios=[10, 1.5], figure=self.fig)
        
        # Main plot area: signal, CAC, Phase Var, Navigator
        self.gs_plots = gridspec.GridSpecFromSubplotSpec(
            4, 1, 
            subplot_spec=self.gs_main[0], 
            height_ratios=[3, 1.5, 1.5, 0.5],
            hspace=0.15
        )
        
        # Side controls
        self.gs_side = gridspec.GridSpecFromSubplotSpec(
            9, 1, subplot_spec=self.gs_main[1], hspace=0.5
        )
        self.setup_controls()

    def setup_controls(self):
        """This will set all the objects you need into their respective axes
        """        
        self.btn_pause = Button(self.fig.add_subplot(self.gs_side[0]), 'Pause/Play')
        self.btn_freq = Button(self.fig.add_subplot(self.gs_side[1]), 'Freq: OFF')
        self.btn_reset = Button(self.fig.add_subplot(self.gs_side[2]), 'Reset Scale')
        self.btn_gif = Button(self.fig.add_subplot(self.gs_side[3]), 'Export GIF')

        ax_speed = self.fig.add_subplot(self.gs_side[4])
        self.txt_speed = TextBox(ax_speed, 'Speed: ', initial=str(self.step_size))
        ax_window = self.fig.add_subplot(self.gs_side[5])
        self.txt_window = TextBox(ax_window, 'Window: ', initial=str(self.window_size))
        
        self.btn_pause.on_clicked(self.toggle_pause)
        self.btn_freq.on_clicked(self.toggle_frequency)
        self.btn_reset.on_clicked(self.manual_rescale)
        self.btn_gif.on_clicked(self.export_gif)
        self.txt_speed.on_submit(self.update_speed)
        self.txt_window.on_submit(self.update_window_size)

    def _init_axes_pool(self):
        """to initlizae the axis.  Plot empty figures for faster filling at animation time
        """        
        # --- Row 1: signal + Frequency ---
        if self.freq_mode == 0:
            self.ax_sig = self.fig.add_subplot(self.gs_plots[0])
            self.ax_freq = None
        else:
            gs_row = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=self.gs_plots[0], wspace=0.1, width_ratios=[0.8, 0.8])
            self.ax_sig = self.fig.add_subplot(gs_row[0])
            self.ax_freq = self.fig.add_subplot(gs_row[1])

        self.line_sig, = self.ax_sig.plot([], [], color='black', lw=1, label=f"lead {self.lead}")
        self.ax_sig.set_ylabel(f"{self.lead}")
        self.ax_sig.legend(loc="upper right")
        
        # --- Row 2: CAC ---
        self.ax_cac = self.fig.add_subplot(self.gs_plots[1], sharex=self.ax_sig)
        self.line_cac, = self.ax_cac.plot([], [], color='dodgerblue', lw=1.5, label="FLUSS CAC")
        self.ax_cac.fill_between([], [], color='dodgerblue', alpha=0.1)
        self.ax_cac.set_ylabel("Arc Curve (0-1)")
        self.ax_cac.set_ylim(0, 1.05)
        self.ax_cac.legend(loc="upper right")

        # --- Row 3: Phase Variance ---
        self.ax_phase = self.fig.add_subplot(self.gs_plots[2], sharex=self.ax_sig)
        self.line_phase, = self.ax_phase.plot([], [], color='purple', lw=1.5, label="Phase Instability")
        self.ax_phase.set_ylabel("Phase Var (rad²)")
        self.ax_phase.legend(loc="lower right")

        # --- Row 4: Navigator ---
        self.ax_nav = self.fig.add_subplot(self.gs_plots[3])
        # Downsample for nav
        ds = max(1, len(self.signal) // 5000)
        self.ax_nav.plot(np.arange(0, len(self.signal), ds), self.signal[::ds], color='gray', alpha=0.5)
        self.nav_cursor = self.ax_nav.axvline(0, color='red', lw=2)
        
        # Mark Regimes
        for loc in self.regime_locs:
            self.ax_nav.axvline(loc, color='blue', alpha=0.3, ymax=0.5)
            # Add markers to main plots if handled in update, but typically static lines need careful management in blit

        self.ax_nav.set_yticks([])
        self.ax_nav.set_xlabel("Timeline (Click to Jump) | Press SPACE to Pause")
        
        # Hide x labels for shared axes
        plt.setp(self.ax_sig.get_xticklabels(), visible=False)
        plt.setp(self.ax_cac.get_xticklabels(), visible=False)

    def rebuild_layout(self):
        self.ani.event_source.stop()
        
        self.ax_sig.remove()
        if self.ax_freq: 
            self.ax_freq.remove()
        self.ax_cac.remove()
        self.ax_phase.remove()
        self.ax_nav.remove()
        self._init_axes_pool()
        self.fig.canvas.draw_idle()
        self.ani.event_source.start()

    def toggle_frequency(self, event):
        self.freq_mode = (self.freq_mode + 1) % 3
        labels = {0: "Freq: OFF", 1: "Freq: STEM", 2: "Freq: SPEC"}
        self.btn_freq.label.set_text(labels[self.freq_mode])
        self.rebuild_layout()

    def update_frame(self, frame):
        if not self.paused:
            self.current_pos += self.step_size
            if self.current_pos + self.window_size > len(self.signal):
                self.current_pos = 0 # Loop

        # Data Slicing
        s = self.current_pos
        e = s + self.window_size
        x_data = np.arange(s, e)

        # 1. Update signal
        view_sig = self.signal[s:e]
        self.line_sig.set_data(x_data, view_sig)
        
        # Auto-scale signal y-axis roughly
        if len(view_sig) > 0:
            # self._apply_scale(ax=self.ax_sig, view_data=view_sig)
            mn, mx = np.min(view_sig), np.max(view_sig)
            self.ax_sig.set_xlim(s, e)
            self.ax_sig.set_ylim(mn - 0.2, mx + 0.2)

        # 2. Update CAC (Corrected Arc Curve)
        view_cac = self.cac[s : min(e, len(self.cac))]
        # Pad if short
        if len(view_cac) < (e-s):
            view_cac = np.pad(view_cac, (0, (e-s)-len(view_cac)), constant_values=1.0)
            
        self.line_cac.set_data(x_data, view_cac)
        # Iterate and remove instead of clearing the ArtistList directly
        for c in list(self.ax_cac.collections):
            c.remove()
            
        self.ax_cac.fill_between(x_data, view_cac, color='dodgerblue', alpha=0.1)
        
        # 3. Update Phase Variance
        view_phase = self.phase_var_stream[s:e]
        self.line_phase.set_data(x_data, view_phase)
        if len(view_phase) > 0:
             self.ax_phase.set_ylim(0, max(np.max(view_phase)*1.1, 0.1))

        # 4. Regime Lines (Vertical Markers)
        # Clear previous vertical lines
        for line in self.ax_sig.lines[1:]: 
            line.remove() # Keep index 0 (signal)
        for line in self.ax_cac.lines[1:]:
            line.remove()
        
        local_regimes = [r for r in self.regime_locs if s <= r < e]
        for r in local_regimes:
            self.ax_sig.axvline(r, color='red', linestyle='--', alpha=0.8)
            self.ax_cac.axvline(r, color='red', linestyle='--', alpha=0.8)

        # 5. Frequency Plot
        if self.freq_mode > 0 and self.ax_freq:
            self.ax_freq.cla()
            # STEM
            if self.freq_mode == 1: 
                yf = np.abs(rfft(view_sig))                 #fft sample
                xf = rfftfreq(len(view_sig), 1 / self.fs)   #frequency list
                half_point = int(len(view_sig)/2)           #Find nyquist freq
                freqs = yf[:half_point]
                freq_l = xf[:half_point]
                self.ax_freq.plot(freq_l, freqs, color='purple', lw=1, label=f"FFT_{self.lead}")
                self.ax_freq.fill_between(freq_l, freqs, color='purple', alpha=0.3)
                self.ax_freq.set_xlim(0, 50)                # Zoom on relevant signal bands
                self.ax_freq.set_title(f"FFT {self.lead}")
            # SPECGRAM  
            elif self.freq_mode == 2: 
                try:
                    self.ax_freq.specgram(view_sig, NFFT=128, Fs=self.fs, noverlap=64, cmap='inferno')
                    self.ax_freq.set_yticks([])
                except Exception as e:
                    logger.error(f"{e}")

        # 6. Nav Cursor
        self.nav_cursor.set_xdata([s])
        
        return []

    def on_click_jump(self, event):
        if event.inaxes == self.ax_nav:
            self.current_pos = int(event.xdata)
            self.current_pos = max(0, min(self.current_pos, len(self.signal) - self.window_size))
            if self.paused:
                self.update_frame(0)
                self.fig.canvas.draw_idle()

    def toggle_pause(self, event=None):
        self.paused = not self.paused

    def _apply_scale(self, ax, view_data):
        if view_data.size > 1:
            v_min, v_max = np.min(view_data), np.max(view_data)
            pad = (v_max - v_min) * 0.1 if v_max != v_min else 0.1
            ax.set_ylim(v_min - pad, v_max + pad)

    def manual_rescale(self, event):
        s = self.current_pos
        e = s + self.window_size
        view = self.signal[s:e]
        if len(view) > 0:
            self._apply_scale(ax=self.ax_sig, view_data=view)
            self.fig.canvas.draw_idle()

    def update_speed(self, text):
        try: 
            self.step_size = int(text)
        except ValueError as v: 
            logger.error(f"{v}")

    def update_window_size(self, text):
        try: 
            self.window_size = int(text)
        except ValueError as v: 
            logger.error(f"{v}")

    def on_key_press(self, event):
        if event.key == ' ': 
            self.toggle_pause()

    def _on_close(self, event):
        self.ani.event_source.stop()

    def export_gif(self, event):
        was_paused = self.paused
        self.paused = True
        f_path = f"export_pos{self.current_pos}.gif"
        logger.info(f"Exporting GIF to {f_path}...")
        writer = PillowWriter(fps=15)
        with writer.saving(self.fig, f_path, dpi=80):
            for _ in range(60):
                self.current_pos += self.step_size
                self.update_frame(0)
                self.fig.canvas.draw()
                writer.grab_frame()
        logger.info("Gif saved :tada:")
        self.paused = was_paused

@dataclass
class BP_Feat():
    id       :str = None     #record index
    onset    :int = None     #Left trough of Systolic peak
    sbp_id   :int = None     #Sytolic Index
    dbp_id   :int = None     #Diastolic Index
    notch_id :int   = None   #Dicrotic notch
    SBP      :float = None   #Systolic peak val
    DBP      :float = None   #Diastolic trough val
    notch    :float = None   #Notch val
    true_MAP :float = None   #MAP via integral
    ap_MAP   :float = None   #MAP via formula
    shock_gap:float = None   #Diff of trueMAP and apMAP
    dni      :float = None   #dicrotic Notch Index
    sys_sl   :float = None   #systolic slope
    dia_sl   :float = None   #diastolic slope
    ri       :float = None   #resistive index
    pul_wid  :float = None   #pulse width 
    p1       :float = None   #TODO Percussion Wave (P1)
    p2       :float = None   #TODO Tidal Wave (P2)
    p3       :float = None   #TODO Dicrotic Wave (P3)
    p1_p2    :float = None   #TODO Ratio of P1 to P2
    p1_p3    :float = None   #TODO Ratio of P1 to P3,
    aix      :float = None   #TODO Augmentation Index (AIx)

class PigRAD:
    def __init__(self, npz_path):
        # 1. load data / params
        self.npz_path       :Path = npz_path
        self.fp_save        :Path = Path(npz_path).parent / (Path(npz_path).stem + "_feat.npz")    # For saving features
        # self.fp_dos         :Path = Path(npz_path).parent / (Path(npz_path).stem + "_cac.npz")     # For saving Corrected Arc Curve
        self.loader         :SignalDataLoader = SignalDataLoader(str(self.npz_path))
        self.full_data      :dict = self.loader.full_data
        self.channels       :list = self.loader.channels
        self.outcomes       :list = self.loader.outcomes
        self.fs             :float = 1000.0                       #Hz
        self.windowsize     :int = 8                              #size of section window 
        self.ecg_lead       :str = 2 # self.pick_lead("ECG")      #pick the ECG lead
        self.lad_lead       :str = 1 # self.pick_lead("Lad")      #pick the Lad lead
        self.car_lead       :str = 6 #self.pick_lead("Cartoid")   #pick the Carotid lead
        self.ss1_lead       :str = 4 # self.pick_lead("SS1")      #pick the SS1 lead
        self.sections       :np.array = segment_ECG(self.full_data[self.channels[self.ecg_lead]], self.fs, self.windowsize)
        self.all_dtypes = [
            ('start'    , 'i4'),  #start index
            ('end'      , 'i4'),  #end index
            ('valid'    , 'i4'),  #valid Section
            ('HR'       , 'i4'),  #Heart Rate
            ('true_MAP' , 'f4'),  #Mean Arterial Pressure (AUC)
            ('ap_MAP'   , 'f4'),  #approximate Mean Arterial pressure (Formula)
            ('shock_gap', 'f4'),  #difference between true and approximate MAP
            ('dni'      , 'f4'),  #dichrotic Notch Index
            ('sys_sl'   , 'f4'),  #systolic slope
            ('dia_sl'   , 'f4'),  #diastolic slope
            ('ri'       , 'f4'),  #resistive index
            ('pul_wid'  , 'f4'),  #pulse width 
            ('p1'       , 'f4'),  #TODO Percussion Wave (P1)
            ('p2'       , 'f4'),  #TODO Tidal Wave (P2)
            ('p3'       , 'f4'),  #TODO Dicrotic Wave (P3)
            ('p1_p2'    , 'f4'),  #TODO Ratio of P1 to P2
            ('p1_p3'    , 'f4'),  #TODO Ratio of P1 to P3,
            ('aix'      , 'f4'),  #TODO Augmentation Index (AIx)
        ]
        self.all_data        :np.array = np.zeros(self.sections.shape[0], dtype=self.all_dtypes)
        self.bp_data        :List[BP_Feat] = []
        self.gpu_devices    :list = [device.id for device in cuda.list_devices()]
        self.view_eda       :bool = False
        self.all_data["start"] = self.sections[:, 0]
        self.all_data["end"] = self.sections[:, 1]
        self.all_data["valid"] = self.sections[:, 2]
        del self.sections
        
    def pick_lead(self, col:str) -> str:
        """Picks the lead you'd like to analyze

        Args:
            col (str): Lead you want to pick

        Raises:
            ValueError: Gotta pick an integer

        Returns:
            lead (str): the lead you picked!
        """
        tree = Tree(f":select channel:", guide_style="bold bright_blue")
        for idx, channel in enumerate(self.channels):
            tree.add(Text(f'{idx}:', 'blue') + Text(f'{channel} ', 'red'))
        pprint(tree)
        question = f"Please select the {col} channel\n"
        file_choice = console.input(f"{question}")
        if file_choice.isnumeric():
            pprint(f"lead {col} loaded")
            return int(file_choice)
        else:
            raise ValueError("Invalid selection")
        
    def save_results(self):
        """Saves the Corrected Arc Curve Results
        """
        out_name = self.fp_save
        out_path = self.npz_path.parent / out_name
        output_path = Path(out_path).with_suffix('.npz')
        np.savez_compressed(output_path, self.all_data)
        
        # Log the size
        mb_size = os.path.getsize(output_path) / (1024 * 1024)
        logger.warning(f"Saved {output_path.name} ({mb_size:.2f} MB)")

    def _derivative(self, signal):
        """Calculates smoothed, 1st, and 2nd derivatives using scipy's Savitzky-Golay filter.

        Args:
            signal (np.array): waveform

        Returns:
            d1, d2 (pd.Series): 1st and 2nd derivative
        """        
        # Window length must be odd; approx 20-30ms is usually good for smoothing derivatives
        window = int(0.03 * self.fs) 
        if window % 2 == 0: 
            window += 1
        smoothed = savgol_filter(signal, window_length=window, polyorder=3)
        d1 = savgol_filter(signal, window_length=window, polyorder=3, deriv=1)
        d2 = savgol_filter(signal, window_length=window, polyorder=3, deriv=2)
        return smoothed, d1, d2
    
    def _integrate(self, signal:np.array)->float:
        """Apply integration of signal. Calculates area under curve

        Args:
            signal (np.array): waveform 

        Returns:
            float: area under the curve
        """        
        return np.trapezoid(signal, dx=1.0/self.fs).item()
    
    def _bandpass_filt(self, data:np.array, lowcut:float=0.1, highcut:float=40.0, fs=1000.0, order:int=4)->np.array:
        """Apply Band Pass Filter

        Args:
            data (np.array): Signal to filter
            lowcut (float, optional): lowcut frequency. Defaults to 0.1.
            highcut (float, optional): highcut frequency. Defaults to 40.0.
            fs (float, optional): sampling rate. Defaults to 1000.0.
            order (int, optional): Order of the filter. Defaults to 4.

        Returns:
            np.array: Filtered signal
        """            
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def _low_pass_filt(self, data:np.array, lowcut:float=5, fs=1000.0, order:int=4)->np.array:
        """Apply Low pass filter

        Args:
            data (np.array): Signal to filter
            lowcut (float, optional): lowcut frequency. Defaults to 5.
            fs (float, optional): sampling rate. Defaults to 1000.0.
            order (int, optional): Order of the filter. Defaults to 4.

        Returns:
            np.array: Filtered signal
        """        
        nyq = 0.5 * fs
        cutoff = lowcut / nyq
        b, a = butter(order, cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)

    def _high_pass_filt(self, data:np.array, highcut:float=20, fs=1000.0, order:int=4)->np.array:
        """Apply high pass filter

        Args:
            data (np.array): Signal to filter
            highcut (float, optional): highcut frequency. Defaults to 20.
            fs (float, optional): sampling rate. Defaults to 1000.0.
            order (int, optional): Order of the filter. Defaults to 4.

        Returns:
            np.array: Filtered signal
        """        
        nyq = 0.5 * fs
        cutoff = highcut / nyq
        b, a = butter(order, cutoff, btype='high', analog=False)
        return filtfilt(b, a, data)
    
    def band_pass(self):
        """Function to run the bandpass over LAD and Carotid leads
        """        
        #Bandpass the flow streams
        for lead in [self.lad_lead, self.car_lead]:
            self.full_data[self.channels[lead]] = self._bandpass_filt(data=self.full_data[self.channels[lead]])

    def calc_RI(self, psv:float, edv:float) -> float:
        """
        Calculates the Resistive Index (RI) from Peak Systolic Velocity (PSV) 
        and End-Diastolic Velocity (EDV).
        
        Args:
            psv (float): Peak systolic velocity.
            edv (float): End-diastolic velocity.
            
        Returns:
            float: The Resistive Index (RI).
        """
        if psv == 0:
            return None  # Avoid division by zero
        ri = (psv - edv) / psv
        return ri.item()
    
    def section_extract(self):
        """This is the main section for signal processing and feature creation. Updates the self.all_data object
        """        
        # Progress bar for section iteration
        precision = 4
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            SpinnerColumn(), 
            BarColumn(), 
            transient=True
        ) as progress:
            task = progress.add_task("Calculating Features...", total=self.all_data.shape[0])
            for idx, section in enumerate(self.all_data):
                #Find R peaks from ECG lead
                start = section[0].item()
                end = section[1].item()
                ecgwave = self.full_data[self.channels[self.ecg_lead]][start:end]
                
                #NOTE - could put STFT here for clean signal check
                #or phase variance.  Need something.  Running blind now
                #IDEA - What about phase variance?  Wavelets are also quick!

                e_peaks, e_heights = find_peaks(
                    x = ecgwave,
                    prominence = np.percentile(ecgwave, 95),  #99 -> stock
                    height = np.percentile(ecgwave, 90),      #95 -> stock
                    distance = round(self.fs*(0.200))           
                )

                if len(e_peaks) < 3:
                    logger.info(f"sect {idx} not enough peaks")
                else:                
                    #Calc HR
                    RR_diffs = np.diff(e_peaks)
                    HR = np.round((60 / (RR_diffs / self.fs)), 2)
                    self.all_data["HR"][idx] = int(np.nanmean(HR)) 

                #Debug plot
                # plt.plot(range(ecgwave.shape[0]), ecgwave.to_numpy())
                # plt.scatter(e_peaks, e_heights["peak_heights"], color="red")

                #Calculate DNI from ss1 lead
                ss1wave = self.full_data[self.channels[self.ss1_lead]][start:end].to_numpy()
                # first find systolic peaks
                s_peaks, s_heights = find_peaks(
                    x = ss1wave,
                    prominence = (np.max(ss1wave) - np.min(ss1wave)) * 0.20,
                    height = np.percentile(ss1wave, 60),
                    distance = int(self.fs*(0.4)),
                    wlen = int(self.fs*1.5)      
                )
                #BUG - left base
                    # the left bases aren't getting calculated correctly.  I tried to adjust wlen as a 
                    # width parameter, buuuut it didn't quite work

                #Debug plot
                # plt.plot(range(ss1wave.shape[0]), ss1wave.to_numpy())
                # plt.scatter(s_peaks, s_heights["peak_heights"], color="red")
                # plt.show()

                if len(s_peaks) < 3:
                    logger.info(f"sect {idx} no peaks in SS1")
                
                else:
                    for id, peak in enumerate(s_peaks[:-1]):
                        p1_vec, p2_vec, p3_vec = None, None, None # Containers for morphological features
                        
                        #Load a dataclass  of features to attach to pigrad bp_data
                        bpf = BP_Feat()
                        bpf.id = str(idx) + "_" + str(id)                  #Encode section_peak as dual index
                        bpf.sbp_id = peak.item()                           #Systolic
                        bpf.dbp_id = s_heights["right_bases"][id].item()   #Diastolic
                        bpf.SBP = ss1wave[bpf.sbp_id].item()
                        bpf.DBP = ss1wave[bpf.dbp_id].item()

                        if id == 0:
                            bpf.onset = s_heights["left_bases"][id].item()  
                        else:
                            #Left base of the peak (previous systolic)
                            bpf.onset = s_heights["right_bases"][id - 1].item()

                        #Get the width of the pulse (ms)
                        if bpf.dbp_id > bpf.onset:
                            pulse_dur = (bpf.dbp_id - bpf.onset) / self.fs
                            bpf.pul_wid = pulse_dur * 1000

                        #two waves selected for each slope of the wave. 
                        sub_sys = ss1wave[bpf.onset:bpf.sbp_id]
                        sub_dias = ss1wave[bpf.sbp_id:bpf.dbp_id]
                        sub_full = ss1wave[bpf.onset:bpf.dbp_id]

                        #Debug plot
                        # plt.plot(range(ss1wave.shape[0]), ss1wave)
                        # plt.scatter(syst, s_heights["peak_heights"][id].item(), color="red")
                        # plt.scatter(s_heights["right_bases"][id].item(), ss1wave.iloc[s_heights["right_bases"][id].item()], color="green")
                        # if id == 0:
                        #     plt.scatter(onset, ss1wave.iloc[s_heights["left_bases"][id].item()], color="yellow")
                        # else:
                        #     plt.scatter(onset, ss1wave.iloc[s_heights["right_bases"][id - 1].item()], color="yellow")
                        # plt.show()
                        # plt.close()

                        #Calc derivatives (returns smoothed, 1st and second deriv with sav_gol filter)
                        try:
                            # Get systolic deriv
                            _, d1_sys, _ = self._derivative(sub_sys)
                            # Get 1st, 2nd diastolic derivatives
                            _, d1_dias, d2_dias = self._derivative(sub_dias)

                            #Get Dichrotic notch index from max of 2nd deriv
                            bpf.notch_id = np.argmax(d2_dias).item()
                            bpf.notch = sub_dias[bpf.notch_id].item()
                            if bpf.notch:
                                bpf.dni = (bpf.notch - bpf.SBP) / (bpf.SBP - bpf.DBP)

                        except Exception as e:
                            logger.warning(f"{e}")

                        #Calc resistive index
                        psv = np.max(d1_sys)
                        edv = np.min(d1_dias)
                        bpf.ri = self.calc_RI(psv, edv)

                        #Calc MAP
                        if sub_full.shape[0] > 0:
                            bpf.true_MAP = self._integrate(sub_full) / (sub_full.shape[0] / self.fs)
                            bpf.ap_MAP = bpf.DBP + (1/3) * (bpf.SBP - bpf.DBP)
                            bpf.shock_gap = bpf.true_MAP - bpf.ap_MAP


                        #Get systolic slope
                        sys_rise = bpf.SBP - ss1wave[bpf.onset]
                        sys_run = (bpf.sbp_id - bpf.onset) / self.fs
                        if sys_run > 0:
                            sys_slope = sys_rise / sys_run 
                        else:
                            sys_slope = None
                        if sys_slope:
                            bpf.sys_sl = sys_slope.item()
                        
                        #Get diastolic slope via exponential decay (regression)
                        if bpf.notch:
                            pe, pe_heights = find_peaks(
                                ss1wave[bpf.notch_id:bpf.dbp_id],
                                height=np.mean(ss1wave[bpf.notch_id:bpf.dbp_id])
                            )
                            if pe.size > 0:
                                y_dia = ss1wave[bpf.notch_id + pe[0].item():bpf.dbp_id]
                                x_dia = np.arange(y_dia.shape[0]) / self.fs
                                slope_dia = linregress(y_dia, x_dia)
                                if slope_dia:
                                    bpf.dia_sl = slope_dia.slope.item()

                        #Add P1 amp add to vec
                        p1_vec = bpf.SBP
                            #Percussion Wave (P1)
                            #Tidal Wave (P2)
                            #Dicrotic Wave (P3) (Diastolic Slope)
                            # A_P1 = ss1wave.iloc[syst]
                            # A_P2 = KneeLocator(
                            #     range(peak, notch),
                            #     ss1wave.iloc[peak:notch],
                            #     curve="concave",
                            #     direction="decreasing"
                            # )
                            #TODO - tomorrows problem
                            # feat_pressure_p1_p3_ratio: $Amplitude(P1) / Amplitude(P3)
                            # feat_pressure_p1_p2_ratio: $Amplitude(P1) / Amplitude(P2)
                            # $feat_pressure_augmented_index: $(Amplitude(P2) - P_{dia}) / (Amplitude(P1) - P_{dia})$
                        self.bp_data.append(bpf)

                    self.all_data["dni"][idx]      = np.round(np.nanmean([rec.dni for rec in self.bp_data if rec.dni != None]), precision)
                    self.all_data["true_MAP"][idx]  = np.round(np.nanmean([rec.true_MAP for rec in self.bp_data if rec.true_MAP != None]), precision)
                    self.all_data["ap_MAP"][idx]  = np.round(np.nanmean([rec.ap_MAP for rec in self.bp_data if rec.ap_MAP != None]), precision)
                    self.all_data["shock_gap"][idx] = np.round(np.nanmean([rec.shock_gap for rec in self.bp_data if rec.shock_gap != None]), precision)
                    self.all_data["sys_sl"][idx]   = np.round(np.nanmean([rec.sys_sl for rec in self.bp_data if rec.sys_sl != None]), precision)
                    self.all_data["dia_sl"][idx]   = np.round(np.nanmean([rec.dia_sl for rec in self.bp_data if rec.dia_sl != None]), precision)
                    self.all_data["ri"][idx]       = np.round(np.nanmean([rec.ri for rec in self.bp_data if rec.ri != None]), precision)
                    # self.all_data["p1"][idx]     = np.round(np.nanmean(p1_vec), precision)
                    # self.all_data["p2"][idx]     = np.round(np.nanmean(p2_vec), precision)
                    # self.all_data["p3"][idx]     = np.round(np.nanmean(p3_vec), precision)
                    # self.all_data["p1_p2"][idx]  = np.round(np.nanmean(p1_p2_rat), precision)
                    # self.all_data["p1_p3"][idx]  = np.round(np.nanmean(p1_p3_rat), precision)
                    # self.all_data["aix"][idx]    = np.round(np.nanmean(aix_vec), precision)

                #Move the progbar
                progress.advance(task)

    def create_features(self):
        self.band_pass()
        self.section_extract()
        console.print("[bold green]Features created...[/]")

    def run_pipeline(self):
        """Checks for existing save files. If found, loads them to save computation time.
        If not found, runs the feature creation and modeling pipeline
        """
        # 1. Check if files exist
        if self.fp_save.exists():
            console.print(f"[green]Found saved files for {self.lead}. Loading...[/]")
            
            # Load NPZ
            container = np.load(self.fp_save)
            cac = container['arr_0']
            console.print("[bold green]Data loaded. Launching Viewer...[/]")
            
            if self.view_gui:
                pass
                # Launch Viewer
                #TODO - Regime Viewer update
                    #Will need to update this. 

                # RegimeViewer(
                #     signal_data=self.full_data[self.lead].astype(np.float64),
                #     cac_data=cac,
                #     regime_locs=regime_locs,
                #     m=m,
                #     sampling_rate=self.fs,
                #     lead=self.lead
                # )
        else:
            console.print("[yellow]No saved data found. Running pipeline...[/]")
            console.print("[green]creating features...[/]")
            self.create_features()
            #Load up the EDA class
            eda = EDA(
                self.full_data, 
                self.channels, 
                self.fs, 
                self.gpu_devices, 
                self.all_data,
                self.ecg_lead,
                self.lad_lead,
                self.car_lead
            )
            eda.clean_data()
            console.print("[green]prepping EDA...[/]")
            if self.view_eda:
                pass
                # ml.corr_heatmap()
                # ml.eda_plot()
            console.print("[green]Enginnering features...[/]")    
            fe = FeatureEngineering(eda)
            #filterout time col 0
            ofinterest = [fe.data.columns[x] for x in range(1, fe.data.shape[1])]
            
            #Engineer your features here. available transforms below
            #log:  Log Transform
            #recip:Reciprocal
            #sqrt: square root
            #exp:  exponential - Good for right skew #!Broken
            #BoxC: Box Cox - Good for only pos val
            #YeoJ: Yeo-Johnson - Good for pos and neg val
            # Ex:    
            # engin.engineer("NN50",    True, False, "YeoJ")
            # engin.engineer("Avg_QT",  True, False, "log")
            #Scale your variables to the same scale.  Necessary for most machine learning applications. 
            #available sklearn scalers
            #s_scale : StandardScaler
            #m_scale : MinMaxScaler
            #r_scale : RobustScaler
            #q_scale : QuantileTransformer
            #p_scale : PowerTransformer
            scaler = "s_scale"

            #Next choose your cross validation scheme. Input `None` for no cross validation
            #kfold       : KFold Validation
            #stratkfold  : StratifiedKFold
            #leavepout   : Leave p out 
            #leaveoneout : Leave one out
            #shuffle     : ShuffleSplit
            #stratshuffle: StratifiedShuffleSplit
            cross_val = "kfold"
            
            #Classifiers
            #'svm':LinearSVC
            #'isoforest':IsolationForest
            #'xgboost':XGBoostClassfier
            modellist = ['pca', 'svm', 'isoforest', 'xgboost']
            console.print("[green]Prepping Data...[/]")
            dp = DataPrep(ofinterest, scaler, cross_val, fe)
            modellist = ['svm', 'isoforest', 'xgboost']
            console.print("[green]training algorithms...[/]")
            #split the training data #splits: test 25%, train 75% 
            [dp.data_prep(model, 0.25) for model in modellist]
            
            #Load the ModelTraining Class
            modeltraining = ModelTraining(dp)
            for model in modellist:
                modeltraining.get_data(model)
                modeltraining.fit(model)
                modeltraining.predict(model)
                modeltraining.validate(model)
                time.sleep(1)
            
            modeltraining.show_results(modellist, sort_des=False) 
            forest = ['isoforest', 'xgboost']
            #Looking at feature importances
            for tree in forest: #Lol
                feats = modeltraining._models[tree].feature_importances_
                modeltraining.plot_feats(tree, ofinterest, feats)

# --- Entry Point ---
def load_choices(fp:str):
    """Loads whatever file you pick

    Args:
        fp (str): file path

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """    
    try:
        tree = Tree(f":open_file_folder: [link file://{fp}]{fp}", guide_style="bold bright_blue")
        walk_directory(Path(fp), tree)
        pprint(tree)
    except Exception as e:
        logger.warning(f"{e}")        

    question = "What file would you like to load?\n"
    file_choice = console.input(f"{question}")
    if file_choice.isnumeric():
        files = sorted(f for f in Path(str(fp)).iterdir() if f.is_file())
        return files[int(file_choice)]
    else:
        raise ValueError("Invalid choice")

def main():
    fp = Path.cwd() / "src/rad_ecg/data/datasets/JT"
    selected = load_choices(fp)
    rad = PigRAD(selected)
    rad.run_pipeline()

if __name__ == "__main__":
    main()

#Problem statement.  
# We're looking to classify the 4 stages of hemorhagic shock. 
# We'll look for 5 regime changes and hope for the best!  

#Workflow
#File choice and signal loading
#Pigrad initialization kicks off stumpy matrix profile calculation
#Runs FLUSS algorithm to look for semantic shifts in the morphology of the signal
#Loads stumpy CAC () curve results to RegimeViewer matplotlib GUI
#During that load, we calculate the phase variance over time with each beat
#To do that we need to isolate the beats and then set a standard window before and after
#Align all of them and then look for the variance in the aligned beat to run the wavelet over.
#Gives the phase variance a kind of a step curve.

#Good results
#sep-3-24

#NOTES 2-10-26
#FLUSS not performing as well as I'd like. 
#Possible problems. 
    #an m that changes throughout the signal because the signal I want to look at is too long.  
        # Could section the ecg according to the phase labels in BSOS data.  
    
    #It also could be that the  periodicity of the carotid flow is so strong it completely obliterates
    #any smaller signal change.  Which we did see in the freqwuency spectrum as the power for that signal
    #was really large.  

    #Additionally, euclidean distance's might break down in this instance because the change isn't immediate.  
    #It's gradual over time which FLUSS won't be able to see.

    #Mortlet Wavelet might not be suitable (meant for ecg's not flow traces)
    #debauchies 4/ symlet 5 and gaussian may be more appropriate

#IDEA - New Modeling path
#What about shooting for a change point detection algorithm.  BOCPD (Bayesian optimized change point detection) might work here.  
#Proposed outline
#1. Downsample if necessary
#2. apply a zero-phase butterworth bandpass filter (0.5 - 30 Hz) on the carotid and LAD traces in order to remove wander and artifacts. 
#3. dicrotic notch index (DNI)
    # find the r peak. Use scipy find_peaks
    # find the systolic peak(SBP) and diastolic trough (DBP)
    # use the second deriv to get the local maxima (aka the dichrotic notch)
    # dni =  (Pnotch - DBP) / (SBP - DBP)
    # supposedly falls off quickly from hem stages 2 and up.
#4. Pulse wave reflection ratios
    #p1 - percussion wave - initial upstroke by lv ejection
    #p2 - tidal wave - reflection from the upper body and renal
    #p3 - dicrotic wave - reflection from the lower body. 
#5. Systolic + Diastolic Slopes 
    #max slope - max value of the first derivative during the upstroke. 
        #gets greater in class 1.  decreases in following
    #Decay time constant
        #for an exponential decay func p(t) = P0e^-t/T to the diastolic portion - notch to end diastole
#6. Use AUC for calculating MAP
#7  Calculate shannon energy maybe?
#8. Calculate diastolic retrograde fraction
    # Don't really understand this one, need to come back. 
#9.  Maybe use a clustering approach for labeling sections
#10. Throw it all at an XGBOOST and look at feature importance. 
#11. Couldn't hurt to verify feature importance with some SHAP values


#Lets run 3 models.  
    #SVM, Xgboost annnnd isoforest

#New Data containers
#### self.sections ####
# |col idx | col name | data type |
# |:--- |:---|:---:|
# |0 | idx of wave section            | int32 |
# |1 | start_point of wave section    | int32 |
# |2 | end_point of wave section      | int32 |
# |3 | Avg HR                         | float32 |
# |4 | Avg peak to dia                | float32 |
# |4 | Avg dia_slope                  | float32 |
# |5 | Avg sys_slope                  | float32 |
# |6 | Avg PE ratio                   | float32 |
# |7 | Avg DNI                        | float32 |


#### self.interior ####
# |col idx | col name | val type | data type |
# |:--- |:---|:---:|:---:|
# | 0 | P peak        | idx  | int32 |
# | 1 | Q peak        | idx  | int32 |
# | 2 | R peak        | idx  | int32 |
# | 3 | S peak        | idx  | int32 |
# | 4 | T peak        | idx  | int32 |
# | 5 | PR Interval   | ms   | int32 |