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
FLOW = dl.load_data('FLOW.obj')
FLOW+=1
countyf = dl.load_data('fips_county.obj')
idx_dict = dl.load_data('idx_dict2.obj')
N = np.array(list(POP.values()))
T = 25
S = np.zeros([N.shape[0],T+1])
S[:,0] = N.__copy__()

I = np.zeros(S.shape)
R = np.zeros(S.shape)
x = np.zeros(S.shape)
y = np.zeros(S.shape)
ii = np.arange(0,len(S))
np.random.shuffle(ii)
I[ii[0:50], 0] = np.random.randint(0,10,50)

# gamma=np.linspace(0.6, 1.2, T)
# beta =np.linspace(1.0, 0.9, T)
# alpha = np.linspace(0.9, 0.2, T)

gamma = 0.8 * np.ones(T)
beta  = 1.0 * np.ones(T)
alpha = 0.8 * np.ones(T)

# for t in range(T):
#     x[:,t] = I[:,t]/N
#     y[:,t] = S[:,t]/N
#     S[:,t+1] = S[:,t] - (beta[t]*S[:,t]*I[:,t])/N - (alpha[t] * S[:,t] * np.dot(FLOW, beta[t] * x[:,t]))/(N + np.sum(FLOW, axis=0))
#     I[:,t+1] = I[:,t] + (beta[t]*S[:,t]*I[:,t])/N + (alpha[t] * S[:,t] * np.dot(FLOW, beta[t] * x[:,t]))/(N + np.sum(FLOW, axis=0)) - gamma[t]*I[:,t]
#     R[:,t+1] = R[:,t] + gamma[t]*I[:,t]
case_dict = dl.load_data('case_county_dict.obj')
# n = np.max([len(ll) for ll in list(case_dict.values())])
# nn = 13
# I = {}
# for fips, cases in case_dict.items():
#     j = np.arange(0, n, nn)
#     I[fips] = np.array(cases)[j]
#
# I = np.array(list(I.values()))
n = np.max([len(ll) for ll in list(case_dict.values())])
nn = 7
j = np.arange(0, n, nn)
I = np.zeros([len(idx_dict), len(j)])
for fips, ii in idx_dict.items():
    if case_dict.get(fips) is None:
        case_dict[fips] = [0]*n
    case = case_dict[fips]
    I[ii,:] = np.array(case)[j]

data_slider = []
T = len(j)
colorbar=dict(tickvals = [-1,0,1,2,3,4,5],
                  ticktext = ['0', '1', '10', '100', '1000', '10k','100k'])
for t in range(T):

    data_each_yr = dict(
                        type='choropleth',
                        geojson = counties,
                        locations = list(countyf.keys()),
                        z= np.max([np.log10(I[:,t] + 1e-1), -1*np.ones(I[:,t].shape)], axis=0),
                        colorbar = colorbar,
                        colorscale = 'Jet',
                        locationmode = 'geojson-id'
    )

    data_slider.append(data_each_yr)

steps = []
for i in range(len(data_slider)):
    step = dict(method='restyle',
                args=['visible', [False] * len(data_slider)],
                label='Week {}'.format(i))
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