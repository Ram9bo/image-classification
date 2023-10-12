import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Example data (replace with your actual DataFrame)
# merged_all = pd.concat([t1, t2, t3, t4], ignore_index=True)

# Set seaborn style
sns.set_style("whitegrid")

data = pd.read_csv("class_mode.csv")

# Define the settings you want to exclude
exclude_settings = ["Halved Classes", "Compressed Outer Classes"]

exclude_metrics = ["Validation Off-By-Half Accuracy", "Validation Off-By-Tenth Accuracy"]

# Filter the DataFrame to exclude the specified settings
filtered_data = data[~data['Setting'].isin(exclude_settings)]
filtered_data = filtered_data[~filtered_data['Metric'].isin(exclude_metrics)]

# Create the line plot
plt.figure(figsize=(12, 6))

sns.lineplot(data=filtered_data, x='Epochs', y='Value', hue='Setting', style='Metric')

# Customize the plot labels and title
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy Over Epochs by Setting')

# Show the legend outside the plot to the right
plt.legend(title='Setting', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust the layout to provide extra space to the right for the legend
plt.subplots_adjust(right=0.75)  # Adjust the right value as needed

# Show the plot
# plt.show()

plt.savefig("accs.png")
