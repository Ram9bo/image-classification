import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import util

# Example data (replace with your actual DataFrame)
# merged_all = pd.concat([t1, t2, t3, t4], ignore_index=True)

# Set seaborn style
sns.set_style("whitegrid")

name = "resize"

data = pd.read_csv(util.data_path(f"{name}.csv"))

# Define the settings you want to exclude
exclude_settings = ["Grayscale - Custom", "RGB - Custom"]

exclude_metrics = []

# Filter the DataFrame to exclude the specified settings
filtered_data = data[~data['Setting'].isin(exclude_settings)]
filtered_data = filtered_data[~filtered_data['Metric'].isin(exclude_metrics)]

# Create the line plot
plt.figure(figsize=(12, 6))

sns.lineplot(data=filtered_data, x='Epochs', y='Value', hue='Setting', style='Metric')

# Customize the plot labels and title
plt.xlabel('Epochs')
plt.ylabel('Metric')
plt.title('Comparing Simplified Classification Schemes')

# Show the legend outside the plot to the right
plt.legend(title='', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust the layout to provide extra space to the right for the legend
plt.subplots_adjust(right=0.75)  # Adjust the right value as needed

# Show the plot
# plt.show()

plt.savefig(util.data_path(f"{name}.png"))
