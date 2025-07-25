import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(df, save_path=None):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()