import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import fastf1 as ff1

# define the variables to plot
year = 2024
grand_prix = "Monaco"
session_type = "R" 
driver_code = "LEC"  # Driver code for Charles Leclerc
colormap = mpl.cm.plasma

# load session data
session = ff1.get_session(year, grand_prix, session_type)
weekend = session.event
session.load()
laps = session.laps.pick_driver(driver_code)

# load telemetry data for the selected driver
x = laps.telemetry['X']              # values for x-axis
y = laps.telemetry['Y']              # values for y-axis
color = laps.telemetry['Speed']      # value to base color gradient on

# 
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# We create a plot with title and adjust some setting to make it look good.
fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(12, 6.75))
fig.suptitle(f'{grand_prix} {year} - {driver_code}\'s Speed', size=24, y=0.97)

# Adjust margins and turn of axis
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.12)
ax.axis('off')


# After this, we plot the data itself.
# Create background track line
ax.plot(laps.telemetry['X'], laps.telemetry['Y'],
        color='black', linestyle='-', linewidth=14, zorder=0)

# Create a continuous norm to map from data points to colors
norm = plt.Normalize(color.min(), color.max())
lc = LineCollection(segments, cmap=colormap, norm=norm,
                    linestyle='-', linewidth=5)

# Set the values used for colormapping
lc.set_array(color)

# Merge all line segments together
line = ax.add_collection(lc)


# Finally, we create a color bar as a legend.
cbaxes = fig.add_axes([0.25, 0.05, 0.5, 0.05])
normlegend = mpl.colors.Normalize(vmin=color.min(), vmax=color.max())
legend = mpl.colorbar.ColorbarBase(cbaxes, norm=normlegend, cmap=colormap,
                                   orientation="horizontal")


# Show the plot
plt.show()
