import pandas as pd
import plotly.express as px
from plotly.io import show

from fastf1.ergast import Ergast

# load race results for the 2024 season
ergast = Ergast()
races_2024 = ergast.get_race_schedule(season=2024)
results = pd.DataFrame()  # Initialize as empty DataFrame instead of list

# for each race in the season
for rnd, race in races_2024['raceName'].items():
    # Get race results (grand prix format)
    # round number +1 because rounds start at 1 not 0
    temp = ergast.get_race_results(season=2024, round=rnd + 1)
    temp = temp.content[0]  # extract dataframe from response
    
    # for weekends with a sprint race
    sprint_results = ergast.get_sprint_results(season=2024, round=rnd + 1)
    if sprint_results.content:  # Check if sprint results exist
        temp = pd.merge(temp, sprint_results.content[0], on='driverCode', how='left')
        # Add GP results and sprint results to main dataframe
        temp['points'] = temp['points_x'] + temp['points_y'].fillna(0)
        temp.drop(columns=['points_x', 'points_y'], inplace=True)
    
    # Add round number and grand prix name to dataframe
    temp['round'] = rnd + 1
    temp['race'] = race.removesuffix(' Grand Prix')
    
    # Append temp to results dataframe
    results = pd.concat([results, temp], ignore_index=True)

races = results['race'].drop_duplicates()
# Print column names to see what's available
print("Available columns:", results.columns.tolist())

results = results.pivot(index='driverCode', columns='round', values='points')
print(results)

# 24 races in 2024 season, 24 drivers over the season - only 20 per race

# Rank the drivers by results to make it pretty
results['total_points'] = results.sum(axis=1)
results = results.sort_values(by='total_points', ascending=False)
results.drop(columns='total_points', inplace=True)

# Use race name, instead of round number, as column names
results.columns = races

# Plot the heatmap 
fig = px.imshow(
    results,
    text_auto=True,
    aspect='auto',  # Automatically adjust the aspect ratio
    color_continuous_scale=[[0,    'rgb(198, 219, 239)'],  # Blue scale
                            [0.25, 'rgb(107, 174, 214)'],
                            [0.5,  'rgb(33,  113, 181)'],
                            [0.75, 'rgb(8,   81,  156)'],
                            [1,    'rgb(8,   48,  107)']],
    labels={'x': 'Race',
            'y': 'Driver',
            'color': 'Points'}       # Change hover texts
)
fig.update_xaxes(title_text='')      # Remove axis titles
fig.update_yaxes(title_text='')
fig.update_yaxes(tickmode='linear')  # Show all ticks, i.e. driver names
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey',
                 showline=False,
                 tickson='boundaries')              # Show horizontal grid only
fig.update_xaxes(showgrid=False, showline=False)    # And remove vertical grid
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')     # White background
fig.update_layout(coloraxis_showscale=False)        # Remove legend
fig.update_layout(xaxis=dict(side='top'))           # x-axis on top
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))  # Remove border margins
fig
show(fig)