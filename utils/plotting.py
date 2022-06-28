"""
Functions for plotting results and descriptive analysis of data.
"""

#%%

import time
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from collections import defaultdict

ROOT_DIR = Path(__file__).parents[1]
RESULTS_DIR = ROOT_DIR / "results"

METRIC_FULL_NAME = {
    "Top1_Acc": "Accuracy",
    "BalAcc": "Balanced Accuracy",
    "Loss": "Loss",
}

STRATEGY_CATEGORY = {
    "Naive": "Baseline",
    "Cumulative": "Baseline",
    "EWC": "Regularization",
    "OnlineEWC": "Regularization",
    "SI": "Regularization",
    "LwF": "Regularization",
    "Replay": "Rehearsal",
    "GEM": "Rehearsal",
    "AGEM": "Rehearsal",
    "GDumb": "Rehearsal",
}

STRATEGY_COLOURS = {
    "Naive": "dodgerblue",
    "Cumulative": "deepskyblue",
    "EWC": "orange",
    "OnlineEWC": "gold",
    "SI": "tomato",
    "LwF": "peru",
    "Replay": "forestgreen",
    "GEM": "limegreen",
    "AGEM": "yellowgreen",
    "GDumb": "palegreen",
}


def get_timestamp():
    """
    Returns current timestamp as string.
    """
    ts = time.time()
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d-%H-%M-%S")


###################################
# Plot figs (metrics over epoch)
###################################


def stack_results(results, metric, mode, type="experience"):
    """
    Stacks results for multiple experiments along same axis in df.

    Either stacks:
    - multiple experiences' metric for same model/strategy, or
    - multiple strategies' [avg/stream] metrics for same model
    """

    results_dfs = []

    # Get metrics for each training "experience"'s test set
    n_repeats = len(results)

    for i in range(n_repeats):
        metric_dict = defaultdict(list)
        for k, v in results[i].items():
            if f"{metric}_Exp/eval_phase/{mode}_stream" in k:
                new_k = (
                    k.split("/")[-1].replace("Exp00", "Task ").replace("Exp0", "Task ")
                )
                metric_dict[new_k] = v[1]

        df = pd.DataFrame.from_dict(metric_dict)
        df.index.rename("Epoch", inplace=True)
        stacked = df.stack().reset_index()
        stacked.rename(
            columns={"level_1": "Task", 0: METRIC_FULL_NAME[metric]}, inplace=True
        )

        results_dfs.append(stacked)

    stacked = pd.concat(results_dfs, sort=False)

    return stacked


def stack_avg_results(results_strats, metric, mode):
    """
    Stack avg results for multiple strategies across epoch.
    """
    results_dfs = []

    # Get metrics for each training "experience"'s test set
    n_repeats = len(list(results_strats.values())[0])
    for i in range(n_repeats):
        metric_dict = defaultdict(list)

        # Get avg (stream) metrics for each strategy
        for strat, metrics in results_strats.items():
            for k, v in metrics[i].items():
                # if train stream in keys "BalancedAccuracy_On_Trained_Experiences"
                if (
                    f'{METRIC_FULL_NAME[metric].replace(" ","")}_On_Trained_Experiences/eval_phase/{mode}_stream'
                    in k
                ):
                    # JA: early stopping means uneven length arrays. Must subsample at n_tasks
                    metric_dict[strat] = v[1]
                    break
                elif f"{metric}_Stream/eval_phase/{mode}_stream" in k:
                    metric_dict[strat] = v[1]

        df = pd.DataFrame.from_dict(metric_dict)
        df.index.rename("Epoch", inplace=True)
        stacked = df.stack().reset_index()
        stacked.rename(
            columns={"level_1": "Strategy", 0: METRIC_FULL_NAME[metric]}, inplace=True
        )

        results_dfs.append(stacked)

    stacked = pd.concat(results_dfs, sort=False)

    return stacked


def plot_metric(method, model, results, mode, metric, ax=None):
    """
    Plots given metric from dict.
    Stacks multiple plots (i.e. different per-task metrics) over training time.

    `mode`: ['train','test'] (which stream to plot)
    """
    ax = ax or plt.gca()

    stacked = stack_results(results, metric, mode)

    # Only plot task accuracies after examples have been encountered
    # JA: this len() etc will screw up when plotting CI's
    tasks = stacked["Task"].str.split(" ", expand=True)[1].astype(int)
    n_epochs_per_task = (stacked["Epoch"].max() + 1) // stacked["Task"].nunique()
    stacked = stacked[tasks * n_epochs_per_task <= stacked["Epoch"].astype(int)]

    sns.lineplot(data=stacked, x="Epoch", y=METRIC_FULL_NAME[metric], hue="Task", ax=ax)
    ax.set_title(method, size=10)
    ax.set_ylabel(model)
    ax.set_xlabel("")


def plot_avg_metric(model, results, mode, metric, ax=None):
    """
    Plots given metric from dict.
    Stacks multiple plots (i.e. different strategies' metrics) over training time.

    `mode`: ['train','test'] (which stream to plot)
    """
    ax = ax or plt.gca()

    stacked = stack_avg_results(results, metric, mode)

    sns.lineplot(
        data=stacked,
        x="Epoch",
        y=METRIC_FULL_NAME[metric],
        hue="Strategy",
        ax=ax,
        palette=STRATEGY_COLOURS,
    )
    ax.set_title("Average performance over all tasks", size=10)
    ax.set_ylabel(model)
    ax.set_xlabel("")


def barplot_avg_metric(model, results, mode, metric, ax=None):
    ax = ax or plt.gca()

    stacked = stack_avg_results(results, metric, mode)
    stacked = stacked[stacked["Epoch"] == stacked["Epoch"].max()]

    sns.barplot(
        data=stacked,
        x="Strategy",
        y=METRIC_FULL_NAME[metric],
        ax=ax,
        palette=STRATEGY_COLOURS,
    )
    ax.set_title("Final average performance over all tasks", size=10)
    ax.set_xlabel("")


###################################
# Clean up plots
###################################


def clean_subplot(i, j, axes, metric):
    """Removes top and rights spines, titles, legend. Fixes y limits."""
    ax = axes[i, j]

    ax.spines[["top", "right"]].set_visible(False)

    if i > 0:
        ax.set_title("")
    if i > 0 or j > 0:
        try:
            ax.get_legend().remove()
        except AttributeError:
            pass

    if metric == "Loss":
        ylim = (0, 4)
    elif metric == "BalAcc":
        ylim = (0.5, 1)
        plt.setp(axes, ylim=ylim)
    else:
        ylim = (0.5, 1)

    # plt.setp(axes, ylim=ylim)


def clean_plot(fig, axes, metric):
    """Cleans all subpots. Removes duplicate legends."""
    for i in range(len(axes)):
        for j in range(len(axes[0])):
            clean_subplot(i, j, axes, metric)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].get_legend().remove()
    fig.legend(handles, labels, loc="center right", title="Task")


def annotate_plot(fig, domain, outcome, metric):
    """Adds x/y labels and suptitles."""
    fig.supxlabel("Epoch")
    fig.supylabel(METRIC_FULL_NAME[metric], x=0)

    fig.suptitle(
        f"Continual Learning model comparison \n"
        f"Outcome: {outcome} | Domain Increment: {domain}",
        y=1.1,
    )


###################################
# Decorating functions for plotting everything
###################################


def plot_all_model_strats(data, domain, outcome, mode, metric, timestamp, savefig=True):
    """Pairplot of all models vs strategies."""

    # Load results
    with open(
        RESULTS_DIR / f"results_{data}_{outcome}_{domain}.json", encoding="utf-8"
    ) as handle:
        res = json.load(handle)

    models = res.keys()
    strategies = next(iter(res.values())).keys()

    n_rows = len(models)
    n_cols = len(strategies)

    # Experience plots
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        sharex=True,
        sharey=True,
        figsize=(2 * 20 * 4 / n_cols, 20 * n_rows / n_cols),
        squeeze=False,
        dpi=250,
    )

    for i, model in enumerate(models):
        for j, strategy in enumerate(strategies):
            plot_metric(strategy, model, res[model][strategy], mode, metric, axes[i, j])

    clean_plot(fig, axes, metric)
    annotate_plot(fig, domain, outcome, metric)

    if savefig:
        file_loc = RESULTS_DIR / "figs" / data / outcome / domain / timestamp / mode
        file_loc.mkdir(parents=True, exist_ok=True)
        plt.savefig(file_loc / f"Exp_{metric}.png")

    # Stream plots
    fig, axes = plt.subplots(
        n_rows,
        2,
        sharex=False,
        sharey=True,
        figsize=(20, 20 * n_rows / n_cols),
        squeeze=False,
        dpi=250,
    )

    for i, model in enumerate(models):
        plot_avg_metric(model, res[model], mode, metric, axes[i, 0])
        barplot_avg_metric(model, res[model], mode, metric, axes[i, 1])

    clean_plot(fig, axes, metric)
    annotate_plot(fig, domain, outcome, metric)

    if savefig:
        file_loc = RESULTS_DIR / "figs" / data / outcome / domain / timestamp / mode
        file_loc.mkdir(parents=True, exist_ok=True)
        plt.savefig(file_loc / f"Stream_{metric}.png")


def results_to_latex():
    """Returns results in LaTeX format for paper tables."""
    raise NotImplementedError


def plot_all_figs(data, domain, outcome):
    """Plots all results figs for paper."""
    timestamp = get_timestamp()

    for mode in ["train", "test"]:
        for metric in ["Loss", "Top1_Acc", "BalAcc"]:
            plot_all_model_strats(data, domain, outcome, mode, metric, timestamp)


#####################
# DESCRIPTIVE PLOTS
#####################


def plot_demographics():
    """
    Plots demographic information of eICU dataset.
    """

    df = pd.DataFrame()  # data_processing.load_eicu(drop_dupes=True)
    _, axes = plt.subplots(3, 2, sharey=True, figsize=(18, 18), squeeze=False)

    df["gender"].value_counts().plot.bar(ax=axes[0, 0], rot=0, title="Gender")
    df["ethnicity"].value_counts().plot.bar(ax=axes[1, 0], rot=0, title="Ethnicity")
    df["ethnicity_coarse"].value_counts().plot.bar(
        ax=axes[1, 1], rot=0, title="Ethnicity (coarse)"
    )
    df["age"].plot.hist(bins=20, label="age", ax=axes[0, 1], title="Age")
    df["region"].value_counts().plot.bar(
        ax=axes[2, 0], rot=0, title="Region (North America)"
    )
    df["hospitaldischargestatus"].value_counts().plot.bar(
        ax=axes[2, 1], rot=0, title="Outcome"
    )
    plt.show()
    plt.close()


########################
# LATEX TABLES
########################


def ci_bound(std, count, ci=0.95):
    """Return Confidence Interval radius."""
    return (1 + ci) * std / np.sqrt(count)


def results_to_table(data, domain, outcome, mode, metric, verbose=False, n="max"):
    """Pairplot of all models vs strategies."""

    # Load results
    with open(
        RESULTS_DIR / f"results_{data}_{outcome}_{domain}.json", encoding="utf-8"
    ) as handle:
        res = json.load(handle)

    models = [k for k in res.keys() if k in ["MLP", "CNN", "LSTM", "Transformer"]]
    dfs = []

    for model in models:
        df = stack_avg_results(res[model], metric, mode)
        df["Model"] = model
        dfs.append(df)

    df = pd.concat(dfs)

    # Get final performance val
    if n == "max":
        df = df[df["Epoch"] == df["Epoch"].max()]
        domain_col = domain
    else:
        df = df[df["Epoch"] == n]
        domain_col = f"{domain} ({n})"

    stats = df.groupby(["Model", "Strategy"])[METRIC_FULL_NAME[metric]].agg(
        ["mean", "count", "std"]
    )

    stats["ci95"] = ci_bound(stats["std"], stats["count"])

    if verbose:
        stats["ci95_lo"] = stats["mean"] + stats["ci95"]
        stats["ci95_hi"] = stats["mean"] - stats["ci95"]
        stats[domain_col] = stats.apply(
            lambda x: f'{x["mean"]:.3f} ({x.ci95_lo:.3f}, {x.ci95_hi:.3f})', axis=1
        )
    else:
        stats[domain_col] = stats.apply(
            lambda x: f'{100*x["mean"]:.1f}$_{{\pm{100*x.ci95:.1f}}}$', axis=1
        )

    stats = pd.DataFrame(stats[domain_col])
    stats.reset_index(inplace=True)

    stats["Category"] = stats["Strategy"].apply(lambda x: STRATEGY_CATEGORY[x])
    stats = stats.pivot(["Category", "Strategy"], "Model")

    return stats


def generate_table_results(
    data="mimic3", outcome="mortality_48h", mode="test", metric="BalAcc", latex=False
):
    """
    Latex table of main results
    """
    domains = ["age", "ethnicity_coarse", "ward", "time_season"]
    dfs = []

    for domain in domains:
        try:
            dfs.append(results_to_table(data, domain, outcome, mode, metric))
        except:
            pass

    df = pd.concat(dfs, axis=1)

    if latex:
        idx = pd.IndexSlice
        sub_idx = idx["Regularization":"Rehearsal", :]
        df = df.style.highlight_max(
            axis=0, props="bfseries: ;", subset=sub_idx,
        ).to_latex()
        return df
    else:
        return df


def generate_hp_table_super(outcome="mortality_48h"):
    """
    Combines all tables into a nice latex format.
    """

    prefix = r"""
\begin{table}[h]
\centering

"""

    box_prefix = r"""
\begin{adjustbox}{max width=\columnwidth}

"""
    old = r"""\begin{tabular}{lllllll}"""
    repl = r"""\begin{tabular}{lllllll}
\multicolumn{7}{c}{\textsc{Age}} \\

"""
    box_suffix = r"""
\end{adjustbox}

"""
    suffix = fr"""
\caption{{Tuned hyperparameters for main experiments (outcome of {outcome}).}}
\label{{tab:hyperparameters}}
\end{{table}}

"""

    latex = (
        prefix
        + box_prefix
        + generate_hp_table(outcome=outcome, domain="age").to_latex().replace(old, repl)
        + generate_hp_table(outcome=outcome, domain="ethnicity_coarse")
        .to_latex()
        .replace(old, repl.replace("Age", "Ethnicity (broad)"))
        + box_suffix
        + box_prefix
        + generate_hp_table(outcome=outcome, domain="time_season")
        .to_latex()
        .replace(old, repl.replace("Age", "Time (season)"))
        + generate_hp_table(outcome=outcome, domain="ward")
        .to_latex()
        .replace(old, repl.replace("Age", "ICU Ward"))
        + box_suffix
        + suffix
    )

    return latex


def generate_table_hospitals(
    outcome="ARF_4h",
    mode="test",
    metric="BalAcc",
    hospitals=[6, 12, 18, 24, 30, 36],
    latex=False,
):
    """
    Latex table of main results
    """

    dfs = [
        results_to_table("eicu", "hospital", outcome, mode, metric, n=n)
        for n in hospitals
    ]

    df = pd.concat(dfs, axis=1)

    if latex:
        idx = pd.IndexSlice
        sub_idx = idx["Regularization":"Rehearsal", :]
        df = df.style.highlight_max(
            axis=0, props="bfseries: ;", subset=sub_idx,
        ).to_latex()
        return df
    else:
        return df


def generate_hp_table(data="mimic3", outcome="mortality_48h", domain="age"):

    models = ["MLP", "CNN", "LSTM", "Transformer"]
    strategies = ["EWC", "OnlineEWC", "LwF", "SI", "Replay", "AGEM", "GEM"]
    dfs = []
    col_rename_map = {
        "ewc_lambda": "lambda",
        "alpha": "lambda",
        "si_lambda": "lambda",
        "memory_strength": "temperature",
        "mem_size": "sample_size",
    }

    for model in models:
        for strategy in strategies:
            try:
                with open(
                    ROOT_DIR
                    / "config"
                    / data
                    / outcome
                    / domain
                    / f"config_{model}_{strategy}.json",
                    encoding="utf-8",
                ) as handle:
                    res = json.load(handle)["strategy"]

                df = pd.DataFrame([res]).rename(columns=col_rename_map)
                df["Model"] = model
                df["Strategy"] = strategy

                dfs.append(df)
            except:
                pass
    df = pd.concat(dfs)
    df = df.set_index(["Model", "Strategy"])
    df = df.replace(np.NaN, "")
    df = df.drop("mode", axis=1)

    return df


# %%
