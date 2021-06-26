import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from re import sub
import numpy as np


def plot_categorical(df, col, target="TARGET_FLAG"):
    crosstab = pd.crosstab(df[col], df[target])
    crosstab_prop = 100 * crosstab.div(crosstab.sum(axis=1), axis=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(col)
    crosstab.plot(kind="bar", stacked=True, ax=ax1)
    crosstab_prop.plot(kind="bar", stacked=True, ax=ax2)
    ax1.title.set_text('Countplot')
    ax2.title.set_text('Target Proportions per Category')
    plt.show()


def plot_continuous(df, col, target="TARGET_FLAG"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(col)
    sns.boxplot(x="TARGET_FLAG", y=col, data=df, ax=ax1)
    df.groupby("TARGET_FLAG")[col].plot(kind='density', legend=True, ax=ax2)
    ax2.set_xlabel(col)
    ax1.title.set_text(f'Box Plot: {col} vs. {target}')
    ax2.title.set_text(f'Variable Density per {target} category')
    plt.show()


def missing_values_nb(df_train, df_test, col):
    miss_train = df_train[col].isna().sum()
    miss_test = df_test[col].isna().sum()

    return (f'{miss_train}, ({round(100 * miss_train / len(df_train), 2)}%)',
            f'{miss_test}, ({round(100 * miss_test / len(df_test), 2)}%)')


def get_chi2_pval(df, col, target):
    """
    Perform Chi-Squared test to assess dependency between two categorical features
    :param df: pandas DataFrame
    :param col: tested variable name (categorical)
    :param target: target column name (categorical)
    :return: p-value of the Chi2 test
    """
    le = LabelEncoder()
    test_res = chi2(le.fit_transform(df[col]).reshape(-1, 1), df[target])
    return test_res[1][0]


def clean_currency(df, cols):
    """
    transform columns with currency string values to floats
    :param df: pandas DataFrame
    :param cols: list of column names that need to be cleaned
    :return: df with cleaned cols
    """
    for col in cols:
        df[col] = df[col].apply(lambda x: float(sub(r'[^\d.]', '', x) if isinstance(x, str) else np.nan))
    return df