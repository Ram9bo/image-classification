"""Computing metrics from total prediction/label file"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score

import util


def save_metrics(data, group, setting):
    # Colums: Run # | Prediction | True Label

    # ======================
    # Accuracy
    accuracy_per_run = data.groupby('Run #').apply(
        lambda group: (group['Prediction'] == group['True Label']).mean()
    )

    # Calculate mean accuracy and standard deviation
    mean_accuracy = accuracy_per_run.mean()
    std_accuracy = accuracy_per_run.std()

    # Display results
    print("\nMean accuracy:", mean_accuracy)
    print("Standard deviation of accuracy:", std_accuracy)

    # ======================
    # Off-by-one accuracy
    obo_per_run = data.groupby('Run #').apply(
        lambda group: (abs(group['Prediction'] - group['True Label']) <= 1).mean()
    )

    mean_obo = obo_per_run.mean()
    std_obo = obo_per_run.std()

    print("\nMean Off-By-One Accuracy", mean_obo)
    print("Standard deviation of off-by-one-accuracy", std_obo)

    # ======================
    # F1 Score
    # Function to calculate F1 score for a group
    def calculate_f1(group):
        return f1_score(group['True Label'], group['Prediction'], average='weighted')

    f1_per_run = data.groupby('Run #').apply(calculate_f1)

    # Calculate mean F1 score and standard deviation
    mean_f1 = f1_per_run.mean()
    std_f1 = f1_per_run.std()

    # Display results
    print("\nMean F1 score:", mean_f1)
    print("Standard deviation of F1 score:", std_f1)

    # Prepare data for plotting
    metrics_df = pd.DataFrame({
        'Run #': accuracy_per_run.index,
        'Accuracy': accuracy_per_run.values,
        'Off-By-One Accuracy': obo_per_run.values,
        'F1 Score': f1_per_run.values
    })

    metrics_df.to_csv(util.results_path(group, setting, f"metrics.csv"), index=False)

    # Melt the data for seaborn compatibility
    metrics_melted = metrics_df.melt(id_vars='Run #', var_name='Metric', value_name='Value')

    # Create a box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=metrics_melted, x='Metric', y='Value', whis=100)

    # Set y-axis scale and labels
    plt.ylim(0, 1.05)  # Set y-axis range
    plt.yticks([i / 10 for i in range(11)],
               [f'{i / 10:.1f}' for i in range(11)])  # Labels from 0 to 1 at intervals of 0.1

    plt.title('Box Plot of Metrics per Run')
    plt.ylabel('Value')
    plt.xlabel('Metric')
    plt.savefig(util.results_path(group, setting, f"metrics_boxplot-{setting}.png"))  # Save the plot to file
    plt.close()


def compute_metrics(data, setting):
    # Calculate metrics for each run
    accuracy_per_run = data.groupby('Run #').apply(
        lambda group: (group['Prediction'] == group['True Label']).mean()
    )

    obo_per_run = data.groupby('Run #').apply(
        lambda group: (abs(group['Prediction'] - group['True Label']) <= 1).mean()
    )

    def calculate_f1(group):
        return f1_score(group['True Label'], group['Prediction'], average='weighted')

    f1_per_run = data.groupby('Run #').apply(calculate_f1)

    # Combine metrics into a DataFrame
    metrics_df = pd.DataFrame({
        'Run #': accuracy_per_run.index,
        'Accuracy': accuracy_per_run.values,
        'Off-By-One Accuracy': obo_per_run.values,
        'F1 Score': f1_per_run.values
    })

    # Add the setting column
    metrics_df['Setting'] = setting

    return metrics_df


if __name__ == '__main__':
    # Read datasets
    datasets = [
        (
            'Standard',
            'fulldata-Standard.csv'),
        (
            'Compressed Start',
            'data/generated/box-plots-alternate-settings/Compressed Start/fulldata-Compressed Start.csv'),
        (
            'Compressed End',
            'data/generated/box-plots-alternate-settings/Compressed End/fulldata-Compressed End.csv'),
        (
            'Compressed Both',
            'data/generated/box-plots-alternate-settings/Compressed Both/fulldata-Compressed Both.csv')
    ]

    # Compute metrics for all settings
    all_metrics = []
    for setting, file_path in datasets:
        data = pd.read_csv(file_path)
        metrics_df = compute_metrics(data, setting)
        all_metrics.append(metrics_df)

    # Combine all metrics into one DataFrame
    combined_metrics = pd.concat(all_metrics, ignore_index=True)

    # Melt the data for seaborn compatibility
    metrics_melted = combined_metrics.melt(id_vars=['Run #', 'Setting'], var_name='Metric', value_name='Value')

    # Create a box plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=metrics_melted, x='Metric', y='Value', hue='Setting', whis=100, dodge=True, gap=0.2)

    # Set y-axis scale and labels
    plt.ylim(0, 1.05)  # Set y-axis range
    plt.yticks([i / 10 for i in range(11)],
               [f'{i / 10:.1f}' for i in range(11)])  # Labels from 0 to 1 at intervals of 0.1

    plt.title('Combined Box Plot of Metrics across Settings')
    plt.ylabel('Value')
    plt.xlabel('Metric')
    plt.legend(title='Setting', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('boxplot-metrics-combined.png')  # Save the plot to file

    # Compute and plot metrics for each dataset
    colors = {
        'Accuracy': '#4f81bd',  # Light blue
        'Off-By-One Accuracy': '#2e75b6',  # Medium blue
        'F1 Score': '#1f4e79',  # Dark blue
    }

    for setting, file_path in datasets:
        data = pd.read_csv(file_path)
        metrics_df = compute_metrics(data, setting)

        # Melt the data for seaborn compatibility
        metrics_melted = metrics_df.melt(id_vars=['Run #', 'Setting'], var_name='Metric', value_name='Value')

        # Create a box plot with specific colors
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=metrics_melted,
            hue='Metric',
            legend=False,
            x='Metric',
            y='Value',
            palette=[colors[metric] for metric in metrics_melted['Metric'].unique()],
            whis=100
        )

        # Set y-axis scale and labels
        plt.ylim(0, 1.05)  # Set y-axis range
        plt.yticks([i / 10 for i in range(11)],
                   [f'{i / 10:.1f}' for i in range(11)])  # Labels from 0 to 1 at intervals of 0.1

        plt.title(f'Box Plot of Metrics for {setting}')
        plt.ylabel('Value')
        plt.xlabel('Metric')
        plt.tight_layout()
        plt.savefig(f'boxplot-{setting}_metrics_boxplot.png')  # Save the plot to file
