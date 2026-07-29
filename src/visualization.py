
import matplotlib.pyplot as plt


def plot_train_curve(df_evaluate, save_path):
    """
    Plot training and validation metric curves and save them to a file.

    Args:
        df_evaluate: DataFrame containing the evaluation history.
        save_path: Output path for the figure.
    """

    
    train_metrics = [col[len("train_"):] for col in df_evaluate.columns if col.startswith("train_")]
    val_metrics = {col[len("val_"):] for col in df_evaluate.columns if col.startswith("val_")}
    test_metrics = {col[len("test_"):] for col in df_evaluate.columns if col.startswith("test_")}
    metrics = [metric for metric in train_metrics if metric in val_metrics and metric in test_metrics]

    
    num_metrics = len(metrics)
    
    fig, axes = plt.subplots(num_metrics, 1, figsize=(6, num_metrics * 4))

    for i, metric in enumerate(metrics):
        train_col = f"train_{metric}"
        val_col = f"val_{metric}"
        test_col = f"test_{metric}"  

        ax = axes[i] if num_metrics > 1 else axes
        ax.plot(df_evaluate["epoch"], df_evaluate[train_col], label=f"Train {metric}")
        ax.plot(df_evaluate["epoch"], df_evaluate[val_col], label=f"Val {metric}")
        ax.plot(df_evaluate["epoch"], df_evaluate[test_col], label=f"Test {metric}")  
        ax.set_title(f"{metric} over epochs")
        ax.set_ylabel(metric)
        ax.set_xlabel("Epochs")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

    
    plt.savefig(save_path)
    plt.cla()
