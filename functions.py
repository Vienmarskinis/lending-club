import polars as pl
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
)
import sklearn
import constants

figure_colors_cmap = sns.color_palette("coolwarm", as_cmap=True)
figure_colors_qualitative = sns.color_palette("colorblind", 10).as_hex()


def drop_duplicates_ignore_first_col(df: pl.DataFrame) -> pl.DataFrame:
    """Drops duplicates from Dataframe, ignoring the first column when making the decision to drop"""
    df_return = df.unique(subset=df.columns[1:])
    return df_return


def remove_text_cast_col(df: pl.DataFrame, col: str, to=pl.Int32) -> pl.DataFrame:
    """Removes text from the specified column and casts it"""
    df_no_text = df.filter(~pl.col(col).str.contains("\D"))
    df_casted = df_no_text.with_columns(pl.col(col).cast(to).alias(col))
    return df_casted


def parse_to_datetime(
    df: pl.DataFrame, col: str, fmt: str = "%Y-%m-%d"
) -> pl.DataFrame:
    """Parses the given column in the dataframe to datetime"""
    df_parsed = df.with_columns(pl.col(col).str.to_datetime(fmt).alias(f"{col}_parsed"))
    return df_parsed


def fraction_split_by_date(
    df: pl.LazyFrame,
    col_name: str,
    col_name_parsed: str,
    train_fraction: float = 0.8,
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """Splits given dataframe into train/test on the col_name_parsed. Tries to achieve train_fraction

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame to be split.
    col_name : str
        name of the column that has the date, unparsed. Used only for logging.
    col_name : str
        name of the column that has the date, parsed. Used for splitting.
    train_fraction : float
        train split fraction when compared to the whole dataset.
    """
    df_rowed = df.with_row_count()
    df_split = df_rowed.filter(
        pl.col("row_nr") >= pl.col("row_nr").max() * train_fraction
    )

    all_dates = df_rowed.select(col_name).collect().to_numpy()
    start_date = all_dates[0][0]
    end_date = all_dates[-1][0]
    split_date = df_split.select(col_name).collect().to_numpy()[0][0]
    split_date_parsed = df_split.select(col_name_parsed).collect().to_numpy()[0][0]

    df_train, df_test = train_test_split_by_date(
        df_rowed, col_name_parsed, split_date_parsed
    )
    df_train = df_train.drop(["row_nr", col_name_parsed])
    df_test = df_test.drop(["row_nr", col_name_parsed])

    real_split_ratio = df_train.collect().height / df_rowed.collect().height
    print(f"The data starts with {start_date}, ends with {end_date}")
    print(
        f"Splitting test/train at {split_date}, achieving {real_split_ratio:1.3f} train data fraction."
    )
    return df_train, df_test, split_date_parsed


def train_test_split_by_date(
    df: pl.LazyFrame, col_name: str, split_date: np.datetime64
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Splits given dataset into train/test by the given date"""
    df_train = df.filter(pl.col(col_name) <= split_date)
    df_test = df.filter(pl.col(col_name) > split_date)
    return df_train, df_test


def lazy_train_test_split(
    df: pl.LazyFrame, train_fraction: float
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Lazily samples approximately train_fraction rows into train and the rest - to test dataframes.

    heavily inspired by:
    https://github.com/pola-rs/polars/issues/3933
    """

    # need to wrap, as apply supplies the value of the column whilst random.random does not take one
    def random2(x):
        return random.random()

    random.seed(42)

    # this adds an additional column with +/- train_fraction% True values
    df_random_col = df.with_columns(
        (pl.first().map_elements(random2) < train_fraction).alias("_sample")
    )

    df_train = df_random_col.filter(pl.col("_sample")).drop("_sample")
    df_test = df_random_col.filter(~pl.col("_sample")).drop("_sample")
    return df_train, df_test


def clean_text_in_cols(df: pl.DataFrame, col: list[str]) -> pl.DataFrame:
    """Change strings to lowercase, remove spaces and underscores"""
    df_clean = df.with_columns(
        pl.col(col).str.to_lowercase().str.replace_all("[ _-]", "")
    )
    return df_clean


def add_labels(ax=None, *args, **kwargs) -> None:
    """Adds labels to the matplotlib bar plot figure."""
    if ax is None:
        ax = plt.gca()
    for container in ax.containers:
        ax.bar_label(container, *args, **kwargs)


def n_unique_in_each_col(df: pl.DataFrame) -> pl.DataFrame:
    """calculates number of unique values in each column of the dataframe.
    Based on https://github.com/pola-rs/polars/issues/10270.
    """
    n_unique_dict = {
        k: v[0] for k, v in df.select(pl.all().n_unique()).to_dict().items()
    }
    df_return = pl.DataFrame(n_unique_dict)
    df_return = df.with_columns(describe=pl.lit("n_unique"))
    df_return = df.select(pl.col("describe"), pl.exclude("describe"))
    return df_return


def normalised_count_by(df: pl.DataFrame, by: str, hue: str) -> pl.DataFrame:
    """Groups by 'by' and 'hue' and calculates the proportional count in ever 'by' group.

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame containing by and hue columns.
    by : str
        The name of the column used to split the dataframe into several groups.
    hue : str
        The name of the column used to split groups into several categories.
    """
    count_grouped_by = df.group_by([by, hue]).agg(pl.count())

    count_per_by_col = f"count_per_{by}"
    count_grouped_by_and_totals = count_grouped_by.with_columns(
        pl.col("count").sum().over(by).alias(count_per_by_col)
    )

    count_norm_col = f"count_norm_per_{by}"
    counts_normalised = count_grouped_by_and_totals.with_columns(
        (pl.col("count") / pl.col(count_per_by_col)).alias(count_norm_col)
    )
    return counts_normalised.drop(["count", count_per_by_col])


def aggregations_by_group(df: pl.DataFrame, by: str, agg_for: str) -> pl.DataFrame:
    """Do a group by for 'by' column and calculate aggregations for 'agg_for' column."""
    df_agg = (
        df.group_by(by)
        .agg(
            pl.min(agg_for).alias("min"),
            pl.quantile(agg_for, 0.25).alias("25%"),
            pl.median(agg_for).alias("median"),
            pl.quantile(agg_for, 0.75).alias("75%"),
            pl.max(agg_for).alias("max"),
            pl.count(agg_for).alias("total_cnt"),
            pl.col(agg_for).is_null().sum().alias("null_cnt"),
        )
        .sort(by)
        .with_columns(
            (pl.col("null_cnt") / pl.col("total_cnt")).alias("null_proportion")
        )
    )
    return df_agg


def norm_plot(
    df: pl.DataFrame,
    x: str,
    hue: str,
    ax: list[plt.Axes],
    for_step2: bool = False,
    *args,
    **kwargs,
):
    """Create a figure of two plots. First plot shows normalised counts and the second is just a countplot."""
    normalised_df = normalised_count_by(df, by=x, hue=hue).fill_null("None")

    sns.barplot(
        y=normalised_df.select(x).to_series(),
        x=normalised_df.select(f"count_norm_per_{x}").to_series(),
        hue=normalised_df.select(hue).to_series(),
        ax=ax[0],
        orient="h",
        *args,
        **kwargs,
    )
    if for_step2:
        mean_positive_proportion = (
            normalised_df.filter(pl.col(hue) == 1)
            .select(f"count_norm_per_{x}")
            .mean()
            .item()
        )
        ax[0].axvline(
            mean_positive_proportion,
            color="red",
            alpha=0.5,
            ls="--",
            label="mean of 1",
        )
        ax[0].legend()

    sns.countplot(
        y=df.select(x).to_series().fill_null("None"),
        stat="percent",
        ax=ax[1],
        *args,
        **kwargs,
    )
    sns.despine(ax=ax[0])
    sns.despine(ax=ax[1])
    add_labels(ax=ax[1], fmt="%1.1f%%")
    ax[0].set_xlabel("Proportion in Specific Group")
    ax[0].set_ylabel("")
    ax[1].set_xlabel("Percent of the Whole Dataset")


def add_year_month(
    df: pl.DataFrame, date_col: str, year: bool = True, month: bool = True
) -> pl.DataFrame:
    """Takes parsed datetime column, extracts and adds year and month"""
    df_augmented = df
    if year:
        df_augmented = df_augmented.with_columns(
            pl.col(date_col).dt.year().suffix("_year"),
        )
    if month:
        df_augmented = df_augmented.with_columns(
            pl.col(date_col).dt.month().suffix("_month"),
        )
    return df_augmented


def replace_value_in_col(
    df: pl.DataFrame, col: str, to_replace=-1, by=None
) -> pl.DataFrame:
    """Replace 'to_replace' with 'by' in the col column."""
    df_replaced = df.with_columns(
        pl.when(pl.col(col) == to_replace).then(by).otherwise(pl.col(col)).keep_name()
    )
    return df_replaced


def correlation_bar(
    df: pd.DataFrame,
    feature: str,
    title: str,
    imp_filter: float | None = None,
    *args,
    **kwargs,
) -> None:
    """Creates a correlation bar for the specified feature.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame containing the feature.
    feature : str
        The name of the feature for which the correlations will be calculated.
    title : str
        The name of the figure.
    kind : str
        The kind of correlation, check pandas .corr() method for available kinds.
    imp_filter : float
        importance filter. If float, then only features above the threshold will be shown.
    """
    corr = df.corr(*args, **kwargs)[[feature]]
    if imp_filter:
        corr = corr[(corr > imp_filter) | (corr < imp_filter * -1)].dropna()
    heatmap = sns.heatmap(
        corr.sort_values(by=feature, ascending=False).drop(feature),
        vmin=-1,
        vmax=1,
        annot=True,
        cmap=figure_colors_cmap,
    )
    heatmap.set_title(title)


def earlier_than_issue_date(
    df: pl.DataFrame, feat_col: str, issue_col: str = "issue_d_parsed"
) -> None:
    """Checks if there are any dates in feat_col earlier than in issue_col"""
    earlier_than_issue = (df.filter(pl.col(feat_col) < pl.col(issue_col))).height
    print(
        f"There're {earlier_than_issue} instances where {feat_col} is before {issue_col}"
    )


def get_excel_drop_features(path: str) -> np.ndarray:
    """Get list of features from excel that should be dropped"""
    features_described = pl.read_excel(path)
    to_drop = (
        features_described.filter(pl.col("keep/drop") == "drop")
        .select("Feature Name")
        .to_numpy()
        .flatten()
    )
    return to_drop


def cast_wrong_str(df: pl.DataFrame, cols: list[str], to=pl.Float64) -> pl.DataFrame:
    """Cast mis-typed str columns"""
    cols_dict = {feature: to for feature in cols}
    return df.cast(cols_dict)


def encode_binary_cols_for_EDA(df: pl.DataFrame, binary_cols: list) -> pl.DataFrame:
    """Encode binary columns in the dataframe as preparation step for EDA"""
    one_hot_encoder = OneHotEncoder(drop="if_binary")
    binary_encoded = one_hot_encoder.fit_transform(df.select(binary_cols).to_pandas())
    binary_df = pl.DataFrame(
        schema=one_hot_encoder.get_feature_names_out().tolist(),
        data=binary_encoded.toarray(),
    )
    df_encoded = df.with_columns(binary_df).drop(binary_cols)
    return df_encoded


def encode_ordinal_col_polars(df: pl.DataFrame, col: str, order: list) -> pl.DataFrame:
    """Encode a single ordinal column in the dataframe as preparation step for EDA"""
    encoder = OrdinalEncoder(
        categories=[order], handle_unknown="use_encoded_value", unknown_value=np.nan
    )
    encoded = encoder.fit_transform(df.select(col).to_numpy())
    df_encoded = df.with_columns(
        pl.Series(name=f"{col}_encoded", values=encoded.flatten())
    ).drop(col)
    return df_encoded


def get_high_corr_pairs(df: pd.DataFrame, greater_t: float = 0.9) -> pd.Series:
    """Calculate and return those pairs of features that have high correlation"""
    corr = df.corr(method="spearman", numeric_only=True)
    # mask for lower triangle
    mask = np.triu(np.ones(corr.shape)).astype(bool)
    corr = corr.where(~mask)
    corr_ser = corr.unstack().sort_values(ascending=False)
    corr_ser = corr_ser[corr_ser > greater_t]
    return corr_ser


# NOT USED
def plot_grade_rocs(
    df: pd.DataFrame, proba_pred: np.ndarray, y_true: pd.Series
) -> None:
    """Plots ROC for loan grade in subplots figure"""
    nrow = 4
    ncol = 2
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol, sharey=True, sharex=True)
    axs = axs.reshape(-1)

    for i, grade in enumerate(constants.grade_order):
        df_aux = df.copy()
        df_aux["class"] = [1 if y == i else 0 for y in y_true]
        df_aux["prob"] = proba_pred[:, i]

        ax = axs[i]

        fpr, tpr, thresholds = roc_curve(df_aux["class"], df_aux["prob"], pos_label=1)
        roc_df = pd.DataFrame(
            {"recall": tpr, "specificity": 1 - fpr, "thresh": thresholds}
        )
        roc_df.plot(x="specificity", y="recall", figsize=(4, 4), legend=False, ax=ax)
        roc_score = roc_auc_score(df_aux["class"], df_aux["prob"])
        ax.text(0.4, 0.2, f"ROC AUC: {roc_score:1.2f}")
        ax.set_ylim(0, 1)
        ax.set_xlim(1, 0)
        ax.plot((1, 0), (0, 1))
        ax.set_xlabel("specificity")
        ax.set_ylabel("recall")
        ax.set_title(f"{grade} Grade Loans")
        sns.despine()
    axs[7].set_xlabel("specificity")
    fig.set_figheight(8)
    fig.set_figwidth(7)
    plt.subplots_adjust(hspace=0.5)


def extract_year(df: pd.DataFrame) -> pd.DataFrame:
    """Extract year from a dataframe columns in %b-%Y format"""
    columns = df.columns
    for col in columns:
        df[f"{col}_month"] = pd.to_datetime(df[col], format="%b-%Y").dt.year
    return df.drop(columns=columns)


def year_pipe_names_out(self, cols):
    """quick workaround for feature_names_out"""
    return cols


def print_grade_metrics(
    conf_disp: ConfusionMatrixDisplay, preds: np.ndarray, truths: np.ndarray, title: str
) -> None:
    """Plots Loan Grade model's confusion matrix, classification report, RMSE and MAE"""
    conf_disp.plot(cmap=figure_colors_cmap)
    plt.title(title)
    plt.show()
    print(classification_report(truths, preds, zero_division=0))
    rmse = mean_squared_error(truths, preds, squared=False)
    mae = mean_absolute_error(truths, preds)
    print(f"RMSE: {rmse:2.3f}")
    print(f"MAE: {mae:2.3f}")


def print_subgrade_metrics(
    matrix: np.ndarray, preds: np.ndarray, truths: np.ndarray, title: str
) -> None:
    """Plots Loan Subgrade model's confusion matrix, RMSE and MAE"""
    plot_big_confusion_matrix(matrix, constants.subgrade_order)
    plt.title(title)
    plt.show()
    rmse = mean_squared_error(truths, preds, squared=False)
    mae = mean_absolute_error(truths, preds)
    print(f"RMSE: {rmse:2.3f}")
    print(f"MAE: {mae:2.3f}")


def plot_big_confusion_matrix(matrix: np.ndarray, labels: list) -> None:
    """Plot Confusion matrix without annotations, big figure, highlight diagonal"""
    sns.heatmap(
        matrix,
        annot=False,
        xticklabels=labels,
        yticklabels=labels,
        cmap=figure_colors_cmap,
    )
    plt.gcf().set_figheight(8)
    plt.gcf().set_figwidth(10)
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")

    # highlight diagonal with outline
    for i in range(len(matrix)):
        plt.gca().add_patch(
            plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="black", lw=0.4)
        )


def get_test_dfs(
    df_test: pl.DataFrame, preprocessor
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Get testing dataframes: X, grade, subgrade, interest rate."""
    X_test = df_test.drop(constants.target_cols).to_pandas()
    grade_test_pl = df_test.select("grade")
    grade_test = encode_ordinal_col_polars(
        grade_test_pl, "grade", constants.grade_order
    ).to_pandas()
    subgrade_test_pl = df_test.select("sub_grade")
    subgrade_test = encode_ordinal_col_polars(
        subgrade_test_pl, "sub_grade", constants.subgrade_order
    ).to_pandas()
    int_rate_test = df_test.select("int_rate").to_pandas()

    X_test_tf = preprocessor.transform(X_test)
    return X_test_tf, grade_test, subgrade_test, int_rate_test
