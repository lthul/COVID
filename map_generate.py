from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
                   dtype={"fips": str})

colorscale = ["#f7fbff", "#ebf3fb", "#deebf7", "#d2e3f3", "#c6dbef", "#b3d2e9", "#9ecae1",
    "#85bcdb", "#6baed6", "#57a0ce", "#4292c6", "#3082be", "#2171b5", "#1361a9",
    "#08519c", "#0b4083", "#08306b"
]

import plotly.express as px
import plotly.graph_objs as go
from plotly.graph_objs import *
import chart_studio.plotly as py
import plotly.offline as offline
import data_loader as dl
import numpy as np


POP = dl.load_data('fips_populations.obj')
FLOW = dl.load_data('FLOW2.obj')
FLOW+=1
countyf = dl.load_data('fips_county.obj')
idx_dict = dl.load_data('idx_dict2.obj')
N = np.array(list(POP.values()))
T=10
S = np.zeros([N.shape[0],T+1])
S[:,0] = N.__copy__()

I = np.zeros(S.shape)
R = np.zeros(S.shape)
x = np.zeros(S.shape)
y = np.zeros(S.shape)
ii = np.arange(0,len(S))
np.random.shuffle(ii)

# CREATE RANDOM INFECTION SEEDS

# I[ii[0:50], 0] = np.random.randint(0,10,50)
n_seeds = 50
tot = np.sum(N)
cdf = np.cumsum(N/tot)
U = np.random.rand(n_seeds)
ii = [np.argmin(np.abs(U[i] - cdf)) for i in range(len(U))]
pct = 0.0001*np.random.rand(n_seeds)
infected = np.int32(N[ii] * pct)
I[ii,0] = infected

# gamma=np.linspace(0.8, 1.0, T)
# beta =np.linspace(1.2, 0.8, T)
# alpha = np.linspace(0.9, 0.2, T)
#
gamma = 0.8 * np.ones(T)
beta  = 0.9 * np.ones(T)
alpha = 0.8 * np.ones(T)

for t in range(T):
    x[:,t] = I[:,t]/N
    y[:,t] = S[:,t]/N
    S[:,t+1] = S[:,t] - (beta[t]*S[:,t]*I[:,t])/N - (alpha[t] * S[:,t] * np.dot(FLOW, beta[t] * x[:,t]))/(N + np.sum(FLOW, axis=0))
    # S[:,t+1] = np.max([np.zeros(S[:,t+1].shape), S[:,t+1]])
    I[:,t+1] = I[:,t] + (beta[t]*S[:,t]*I[:,t])/N + (alpha[t] * S[:,t] * np.dot(FLOW, beta[t] * x[:,t]))/(N + np.sum(FLOW, axis=0)) - gamma[t]*I[:,t]
    # I[:, t + 1] = np.max([np.zeros(I[:, t + 1].shape), I[:, t + 1]])
    R[:,t+1] = R[:,t] + gamma[t]*I[:,t]
    # R[:, t + 1] = np.max([np.zeros(R[:, t + 1].shape), R[:, t + 1]])
    print(sum(N))


data_slider = []

colorbar=dict(tickvals = [-1,0,1,2,3,4,5],
                  ticktext = ['0', '1', '10', '100', '1000', '10k','100k'])
for t in range(T):

    data_each_yr = dict(
        type='choropleth',
        geojson=counties,
        locations=list(countyf.keys()),
        z=np.max([np.log10(I[:, t] + 1e-1), -1 * np.ones(I[:, t].shape)], axis=0),
        colorbar=colorbar,
        colorscale='Jet',
        locationmode='geojson-id'
    )

    data_slider.append(data_each_yr)

steps = []
for i in range(len(data_slider)):
    step = dict(method='restyle',
                args=['visible', [False] * len(data_slider)],
                label='Day {}'.format(i))
    step['args'][1][i] = True
    steps.append(step)

sliders = [dict(active=0, pad={"t": 1}, steps=steps)]

layout = dict(title ='Virus cases', geo=dict(scope='usa',
                       projection={'type': 'albers usa'}),
              sliders=sliders)

fig = dict(data=data_slider, layout=layout)
# fig = dict(data=data_slider)
offline.plot(fig)
#
# fig = px.choropleth(df, geojson=counties, locations='fips', color='unemp',
#                            color_continuous_scale="Viridis",
#                            range_color=(0, 12),
#                            scope="usa",
#                            labels={'unemp':'unemployment rate'}
#                           )
# fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# fig.show()