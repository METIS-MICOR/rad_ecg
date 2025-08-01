########################### Main imports ###############################
import numpy as np
import pandas as pd
from rich.table import Table
from rich.theme import Theme
import setup_globals
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from support import log_time, console, logger
from scipy.stats import pearsonr, probplot, boxcox, yeojohnson, norm

########################### Sklearn imports ###############################
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, StratifiedKFold, LeavePOut, LeaveOneOut
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import accuracy_score as ACC_SC
from sklearn.metrics import log_loss as LOG_LOSS
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as RSQUARED
from sklearn.metrics import roc_auc_score, roc_curve, auc

#CLASS EDA
class EDA(object):
    def __init__(self, ecg_data:dict, wave:np.array):
        self.dataset = ecg_data
        self.wave = wave
        self.task = "classification"
        self.data = pd.DataFrame(self.dataset["interior_peaks"]) 
        self.target = pd.Series(ecg_data["section_info"][:, 3])
        self.target_names = ["anomaly", "stable"]
        self.rev_target_dict = {
            0:"anomaly",
            1:"stable"
        }
    
    #FUNCTION clean_data
    def clean_data(self):
        pass

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
                        f"{nulls[ss]:.0f}", 
                        f"{perc[ss]:.2%}", 
                        f"{str(over30[ss])}", 
                        style="kindagood", 
                        end_section=end_sect
                    )
                else:
                    table.add_row(
                        cols[ss], 
                        f"{nulls[ss]:.0f}", 
                        f"{perc[ss]:.2%}", 
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
            _mean, _stddev, _max, _min, _count = self.data[col].agg([np.mean, np.std, max, min, "count"]).T
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

    # #FUNCTION cat_features
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
    # 		# Possibly could plot the counts of each column vs the total shape. 

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

    #FUNCTION heatmap
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
            logging.info(f'correlation of {feat_1} and {feat_2}:\n{self.corr:.2f}')

        #Generates repeatable colordict for all values in the group
        if not isinstance(group, bool):
            #Get the size of the target responses. (how many are there)
            num_groups = np.unique(group).size
                #NOTE, best to use MPL's sequential colormaps as the ligthness is
                #monotonically increasing in intensity of the color
                #Avail cmaps here.  
                #https://matplotlib.org/stable/tutorials/colors/colormaps.html
            #Get the colormap
            color_cmap = plt.cm.get_cmap('Paired', num_groups) 
            #Generate a hex code for the color.  
            color_str = [mpl.colors.rgb2hex(color_cmap(i)) for i in range(color_cmap.N)]
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
    def __init__(self):
        #Inherit from the EDA class
        super().__init__()
        # EDA.clean_data(self)
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


    #FUNCTION categorical_encoding
    def categorical_encoding(self, enc:str, feat:str, order:list=None):
        """Note, can only drop single columns at a time. I'll 
        Eventually make it do multiple

        Args:
            enc (str): _description_
            feat (str): _description_
            order (list, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """		
        if isinstance(enc, str):
            enc_dict = {
                "onehot" :OneHotEncoder(
                    categories="auto",
                    drop=None,
                    sparse_output=False,
                    dtype=int,
                    handle_unknown="error",
                    min_frequency=None,
                    max_categories=None)
                ,
                "ordinal":OrdinalEncoder(
                    categories=[order],
                    dtype=np.float64,
                    # handle_unknown="error",
                    min_frequency=None,
                    max_categories=None)
            }
            encoder = enc_dict.get(enc)
            if not encoder:
                raise ValueError(f"Encoder not loaded, check before continuing")

            #Fit and transform the column (Needs to reshaped to 2d for transform)
            arr = encoder.fit_transform(self.data[feat].to_numpy().reshape(-1, 1))
            if enc == "onehot":
                #grab columns 
                ndf_cols = encoder.categories_[0].tolist()
                ndf_cols = ["oh_" + x for x in ndf_cols]
                #Add to dataset
                new_df = pd.DataFrame(arr, index = self.data.index, columns=ndf_cols)
                #Add colnames to feature_names
                self.feature_names.extend(ndf_cols)
            
            elif enc == "ordinal":
                #Make trans col name
                nn = "ord_" + feat
                new_df = pd.DataFrame(arr, index = self.data.index, columns=[nn])
                self.feature_names.append(nn)

            #Add the new col/cols
            self.data = pd.concat([self.data, new_df], axis=1)

            #Drop said feature of transform from dataset
            self.data.drop([feat], axis=1, inplace=True)
            self.feature_names.pop(self.feature_names.index(feat))
            logger.info(f"Feature: {feat} has been encoded with {encoder.__class__()} ")

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
                    logging.info(f'Probability plot for\n{feature}\nbefore-after\n{trans} transformation')
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
                    logging.info(f'Probability plot for\n{feature}\nbefore-after\n{trans} transformation')
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
                        logging.info(f'Probability plot for\n{feature}\nbefore-after\n{trans} transformation')
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
            self.data_description = engin.data_description
            self.feature_names = features
            self.target = engin.target
            self.target_names = engin.target_names
            self.task = engin.task
        else:
            EDA.__init__(self)
            EDA.clean_data(self)
            # EDA.drop_nulls(self)
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

def run_models(dataset:dict):
    #Load test/train data
    X = dataset
    y = dataset
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0)
    d_train = xgboost.DMatrix(X_train, label=y_train)
    d_test = xgboost.DMatrix(X_test, label=y_test)
    params = {
        "eta": 0.01,
        "objective": "binary:logistic",
        "subsample": 0.5,
        "base_score": np.mean(y_train),
        "eval_metric": "logloss",
    }
def run_eda(ecg_data:dict, wave:np.array):
    EDA.init() 

################################# Start Program ####################################
@log_time
def main():
    global configs
    configs = setup_globals.load_config()
    configs["slider"], configs["run_anomalyd"] = True, True
    datafile = setup_globals.launch_tui(configs)
    wave, fs, outputf = setup_globals.load_chart_data(configs, datafile, logger)
    wave_sect_dtype = [
        ('wave_section', 'i4'),
        ('start_point', 'i4'),
        ('end_point', 'i4'),
        ('valid', 'i4'),
        ('fail_reason', str, 16),
        ('Avg_HR', 'f4'), 
        ('SDNN', 'f4'),
        ('min_HR_diff', 'f4'), 
        ('max_HR_diff', 'f4'), 
        ('RMSSD', 'f4'),
        ('NN50', 'f4'),
        ('PNN50', 'f4')
    ]

    for fname in outputf:
        if fname.endswith("_section_info.csv"):
            fpath = datafile._str + "\\" + fname.split("_section_info")[0]
            break
    
    global ecg_data
    ecg_data = {
        "peaks": np.genfromtxt(fpath+"_peaks.csv", delimiter=",", dtype=np.int32, usecols=(0, 1)),
        "section_info": np.genfromtxt(fpath+"_section_info.csv", delimiter=",", dtype=wave_sect_dtype),
        "interior_peaks": np.genfromtxt(fpath+"_interior_peaks.csv", delimiter=",", dtype=np.int32, usecols=(range(16)), filling_values=0)
    }
    #TODO - Eventually throw some EDA in to get a feeling of summary stats
    run_eda(ecg_data, wave)
    run_models(ecg_data, wave)
    
if __name__ == "__main__":
    main()
    