import pandas as pd
import matplotlib.pyplot as plt
# Load the data from the CSV file
file_path = 'CamreadyFFREP_Plots.csv'
file_path2 = 'CIFARAcclayerwise_losses_PredGASFGD.csv'

data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
data.head()

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

linewidth= 2
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Dropping the 'Unnamed: 4' column as it is irrelevant for the plots
data = data.drop(columns=['Unnamed: 4'], errors='ignore').iloc[:50]

# Extracting Convolution columns and Error columns
conv_columns = [col for col in data.columns if 'FF' in col]
error_columns = [col for col in data.columns if 'Error' in col]

# Create subplots with 2 rows and 1 column
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10), sharex=True)

# Plot Convolution values
for conv_col in conv_columns:
    axes[0].plot(data.index + 1, data[conv_col], label=conv_col, marker="|", linewidth=linewidth)
axes[0].set_yscale('log')
axes[0].set_ylabel('Layer-wise Training Loss Value')
axes[0].grid(True, which='both', axis='x', linestyle='--', linewidth=0.75)
axes[0].set_xticks(range(0, len(data), 5))
axes[0].legend()

# Set x-axis labels every 5 epochs for the Convolution plot
epochs = data.index
axes[0].set_xticks(epochs[::5])
axes[0].set_xlim([1, epochs.max() + 1])
# axes[1].set_ylim([0.2, 0.95])

# Plot Error values
for error_col in error_columns:
    axes[1].plot(data.index + 1, data[error_col], label=error_col, marker="|", linewidth=linewidth)
axes[1].set_ylabel('Error Rate (%)')
axes[1].set_xlabel('# Training Epoch')
axes[1].grid(True, which='both', axis='x', linestyle='--', linewidth=0.75)
axes[1].legend()

# Adjust the layout
plt.tight_layout()

# # Save the figure
# plot_path = '/mnt/data/conv_error_plots.png'
# plt.savefig(plot_path)
#
# # Show the figure
plt.show()
#
# plot_path

data = pd.read_csv(file_path2)

# Display the first few rows of the dataframe to understand its structure
data.head()

# Dropping the 'Unnamed: 4' column as it is irrelevant for the plots
data = data.drop(columns=['Unnamed: 4'], errors='ignore').iloc[:25]

# Extracting Convolution columns and Error columns
conv_columns = [col for col in data.columns if 'Conv' in col]
error_columns = [col for col in data.columns if 'Pred' in col]

# Create subplots with 2 rows and 1 column
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10), sharex=True)

# Plot Convolution values
for conv_col in conv_columns:
    axes[0].plot(data.index + 1, data[conv_col], label=conv_col, marker="|", linewidth=linewidth)
axes[0].set_yscale('log')
axes[0].set_ylabel('Layer-wise Training Loss Value')
axes[0].grid(True, which='both', axis='x', linestyle='--', linewidth=0.75)
axes[0].set_xticks(range(0, len(data), 5))
axes[0].legend()

# Set x-axis labels every 5 epochs for the Convolution plot
epochs = data.index
axes[0].set_xticks(epochs[::5])
axes[0].set_xlim([1, epochs.max()])
# axes[1].set_ylim([0.2, 0.95])
# Plot Error values
for error_col in error_columns:
    axes[1].plot(data.index + 1, data[error_col], label=error_col, marker="|", linewidth=linewidth)
axes[1].set_ylabel('Error Rate (%)')
axes[1].set_xlabel('# Training Epoch')
axes[1].grid(True, which='both', axis='x', linestyle='--', linewidth=0.75)
axes[1].legend()

# Adjust the layout
plt.tight_layout()

# # Save the figure
# plot_path = '/mnt/data/conv_error_plots.png'
# plt.savefig(plot_path)
#
# # Show the figure
plt.show()
#
# plot_path
