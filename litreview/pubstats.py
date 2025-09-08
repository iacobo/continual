import uuid
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pymed import PubMed


def fetch_articles(pubmed, search_query, keys, max_results=5000):
    results = pubmed.query(search_query, max_results=max_results)
    return [{key: article.toDict()[key] for key in keys} for article in results]


def process_articles(articles):
    df = pd.DataFrame.from_dict(articles)
    df["pubmed_id"] = df["pubmed_id"].str.split("\n").str[0]
    df["year"] = pd.to_datetime(df["publication_date"]).dt.year
    return df


def count_keyword_occurrences(df, search_column, keyword):
    counts = df.groupby("year")[search_column].apply(
        lambda x: x.str.contains(keyword, case=False, na=False).sum()
    )
    return counts.rename(keyword)


def plot_dataframes(df, colors, ax):
    df.plot.bar(ax=ax, color=colors)
    ax.spines[["right", "top"]].set_visible(False)


def main(search_terms, keywords, outdir):
    id = str(uuid.uuid4())[:4]
    email = f"myemail{id}@ccc.com"
    pubmed = PubMed(tool="PubMedSearcher", email=email)

    keys = [
        "pubmed_id",
        "title",
        "keywords",
        "abstract",
        "conclusions",
        "methods",
        "results",
        "publication_date",
    ]

    for search, search_term in search_terms.items():
        articles = fetch_articles(pubmed, search_term, keys)
        df = process_articles(articles)
        df.to_csv(outdir / f"pubmed_{search}.csv", index=None, header=True)
        print(f"{search}: {len(df)}")

        outputs = {}
        for key, keyword_list in keywords.items():
            series_list = [
                count_keyword_occurrences(df, "abstract", keyword)
                for keyword in keyword_list
            ]
            outputs[key] = pd.concat(series_list, axis=1)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=True,
        figsize=(10, 7),
        dpi=250,
        gridspec_kw={"height_ratios": [310, 210, 150]},
    )

    color_schemes = [
        mpl.colormaps["tab10"].colors,
        mpl.colormaps["tab10"].colors[6:] + (mpl.colormaps["tab20"].colors[-5],),
        mpl.colormaps["tab20"].colors[0::5],
    ]

    for i, (key, output_df) in enumerate(outputs.items()):
        plot_dataframes(output_df, color_schemes[i], axes[i])
        if i == 1:
            axes[i].set_ylabel("Number of publications")

    fig.savefig(outdir / "publication_stats.png")


if __name__ == "__main__":
    root = Path.cwd()
    output_dir = root / "results"
    output_dir.mkdir(exist_ok=True)

    with open("assets/keywords.json") as kwf:
        keywords = json.load(kwf)

    with open("assets/searchterms.json") as stf:
        search_terms = json.load(stf)

    main(search_terms, keywords, output_dir)
