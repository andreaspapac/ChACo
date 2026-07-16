import matplotlib.pyplot as plt
import numpy as np

# Create a dummy figure and axis
fig, ax = plt.subplots(figsize=(8, 1))

# Create a colormap (jet in this case)
cmap = plt.get_cmap('jet')

# Create a colorbar using a ScalarMappable with the colormap
norm = plt.Normalize(vmin=0, vmax=1)
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')

# Remove axis labels and ticks for clarity
ax.set_axis_off()

# Display the colorbar
plt.show()
