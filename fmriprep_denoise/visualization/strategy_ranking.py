import pandas as pd
from fmriprep_denoise.visualization import degrees_of_freedom_loss, motion_metrics
from fmriprep_denoise.visualization import utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection


strategy_order = list(utils.GRID_LOCATION.values())
fmriprep_versions = ['fmriprep-20.2.1lts', 'fmriprep-20.2.5lts']
datasets = ['ds000228', 'ds000030']

def load_data(path_root, datasets, criteria_name='stringent'):
    mean_total_ranking = []
    for v in fmriprep_versions:
        dof = degrees_of_freedom_loss.load_data(
            path_root, datasets, criteria_name, v)
        for d in dof:
            df_data = pd.DataFrame()
            current_ranking = {}
            for s in strategy_order:
                m = dof[d].loc[:, (s, 'total')].mean()
                current_ranking[s] = [m]
            order = pd.DataFrame(current_ranking).T.sort_values(0)
            order.index = order.index.set_names(['strategy'])
            order = order.reset_index()
            order['version'] = v
            order['dataset'] = d
            order['loss_df'] = list(range(1, order.shape[0] + 1))
            order = order.drop(0, axis=1)
            df_data = pd.concat([df_data, order], axis=0)
            df_data = df_data.set_index(['strategy', 'version', 'dataset'])

            metrics = pd.DataFrame()
            for m in ['p_values', 'median', 'distance', 'modularity']:
                data, measure = motion_metrics.load_data(
                    path_root, datasets, criteria_name, v, measure_name=m)
                ascending = m != 'modularity'
                r = data[d].groupby('strategy')[measure['label']].describe()['mean'].sort_values(ascending=ascending)
                rk = pd.DataFrame(list(range(1, 12)), index=r.index, columns=[m])
                rk = rk.reset_index()
                rk['version'] = v
                rk['dataset'] = d
                rk = rk.set_index(['strategy', 'version', 'dataset'])
                metrics = pd.concat([metrics, rk], axis=1)
            df_data = pd.concat([df_data, metrics], axis=1)
            mean_total_ranking.append(df_data)
    mean_total_ranking = pd.concat(mean_total_ranking).reset_index()
    mean_total_ranking = mean_total_ranking.melt(
        id_vars=['strategy', 'version', 'dataset'],
        var_name='metrics',
        value_name='ranking')
    return pd.pivot_table(mean_total_ranking, columns='strategy', index=['dataset', 'version', 'metrics'], fill_value='ranking')


def plot_ranking(data):
    """Plot the ranking of 4 selected metrics as bubble heatmaps."""
    fig, axs = plt.subplots(2, 2, figsize=(11, 4.8), sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle(
        "Ranking of all strategies per dataset per fMRIPrep version",
        weight="heavy",
        fontsize="x-large",
    )
    for i, d in enumerate(datasets):
        for j, v in enumerate(fmriprep_versions):
            mat = data.xs(d, level='dataset', drop_level=True)
            mat = mat.xs(v, level='version', drop_level=True)
            mat = mat.droplevel(None, axis=1)
            mat = mat.loc[['modularity', 'distance', 'median', 'p_values'], strategy_order]

            x, y = np.meshgrid(np.arange(mat.shape[1]), np.arange(mat.shape[0]))

            R = (12 - mat) / 12 / 2
            circles = [plt.Circle((j,i), radius=r) for r, j, i in zip(R.values.flat, x.flat, y.flat)]
            col = PatchCollection(circles, array=mat.values.flatten(), cmap="rocket_r")
            axs[i, j].add_collection(col)

            axs[i, j].set(xticks=np.arange(mat.shape[1]), yticks=np.arange(mat.shape[0]),
                xticklabels=mat.columns,
                yticklabels=['Average network modularity', 'DM-FC', 'QC-FC: median', 'QC-FC: significant']
                )
            axs[i, j].set_xticklabels(mat.columns, rotation=45, ha='right')
            axs[i, j].set_xticks(np.arange(mat.shape[1] + 1)-0.5, minor=True)
            axs[i, j].set_yticks(np.arange(mat.shape[0] + 1)-0.5, minor=True)
            axs[i, j].grid(which='minor')
            axs[i, j].set_title(f"{d}: {v}")

    fig.colorbar(col)
    return fig