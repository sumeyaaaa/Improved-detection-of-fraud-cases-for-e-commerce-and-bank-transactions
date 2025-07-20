import matplotlib.pyplot as plt
import seaborn as sns

def plot_distributions(df, numeric_columns=None, categorical_columns=None, bins=30):
    """
    Plot distribution graphs for numeric and categorical columns.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - numeric_columns (list): List of numeric columns to plot as histograms.
    - categorical_columns (list): List of categorical columns to plot as count plots.
    - bins (int): Number of bins for histogram (default: 30).
    """
    total_plots = (len(numeric_columns or []) + len(categorical_columns or []))
    if total_plots == 0:
        print("No columns data specified for plotting.")
        return

    plt.figure(figsize=(6 * 2, 4 * ((total_plots + 1) // 2)))  # auto layout

    plot_idx = 1

    if numeric_columns:
        for col in numeric_columns:
            plt.subplot((total_plots + 1) // 2, 2, plot_idx)
            sns.histplot(df[col], kde=True, bins=bins)
            plt.title(f'Distribution of {col}')
            plot_idx += 1

    if categorical_columns:
        for col in categorical_columns:
            plt.subplot((total_plots + 1) // 2, 2, plot_idx)
            sns.countplot(x=col, data=df)
            plt.title(f'Distribution of {col}')
            plot_idx += 1

    plt.tight_layout()
    plt.show()
def plot_boxplots_by_class(df, numeric_columns, target='class'):
    """
    Plot boxplots for numeric features grouped by target class.

    Parameters:
    - df (pd.DataFrame): Dataframe containing the data.
    - numeric_columns (list): List of numerical column names.
    - target (str): Target column name (default: 'class').
    """
    for col in numeric_columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=target, y=col, data=df)
        plt.title(f'{col} vs {target}')
        plt.show()


def plot_countplots_by_class(df, categorical_columns, target='class'):
    """
    Plot countplots for categorical features grouped by target class.

    Parameters:
    - df (pd.DataFrame): Dataframe containing the data.
    - categorical_columns (list): List of categorical column names.
    - target (str): Target column name (default: 'class').
    """
    for col in categorical_columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=col, hue=target, data=df)
        plt.title(f'{col} vs {target}')
        plt.xticks(rotation=45)
        plt.show()