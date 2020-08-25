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
import matplotlib.pyplot as plt


POP = dl.load_data('fips_populations.obj')
FLOW = dl.load_data('FLOW2.obj')
P = dl.load_data('Pmat.obj')

countyf = dl.load_data('fips_county.obj')
idx_dict = dl.load_data('idx_dict2.obj')
N = np.array(list(POP.values()))
nc = len(N)
T=25
S = np.zeros([N.shape[0],T+1])
S[:,0] = N.__copy__()

I = np.zeros(S.shape)
R = np.zeros(S.shape)



FLOW+=1
np.fill_diagonal(FLOW,0)
Nvals = N.repeat(nc).reshape(nc,nc)
FLOW_mat = np.divide(FLOW,Nvals)
FLOW_mat = np.min([FLOW_mat, 0.01*np.ones(FLOW_mat.shape)],axis=0)

# CREATE RANDOM INFECTION SEEDS

ii = np.arange(0,len(S))
np.random.shuffle(ii)
# I[ii[0:50], 0] = np.random.randint(0,10,50)
n_seeds = 100
tot = np.sum(N)
cdf = np.cumsum(N/tot)
U = np.random.rand(n_seeds)
ii = [np.argmin(np.abs(U[i] - cdf)) for i in range(len(U))]
pct = 0.01*np.random.rand(n_seeds)
infected = np.int32(N[ii] * pct)
I[ii,0] = infected
R[:,0] = np.int32(.001*N)
S[:,0] = N - I[:,0] - R[:,0]


# gamma=np.linspace(0.8, 1.0, T)
# beta =np.linspace(1.2, 0.8, T)
# alpha = np.linspace(0.9, 0.2, T)
#
gamma = 0.2 * np.ones(T)
beta  = 0.9 * np.ones(T)
alpha = 0.8 * np.ones(T)


print(np.sum(N))

for t in range(T):
    # USE binomial distribution to randomly generate people coming and going from county
    # p values generated from the FLOW matrices gathered from census data
    Nvals = np.int32(N.repeat(nc).reshape(nc, nc)).T
    FLOWn = np.random.binomial(np.int32(0.2*Nvals), FLOW_mat)
    # leaving = np.sum(FLOWn, axis=0)
    # entering = np.sum(FLOWn, axis=1)



    # net = entering - leaving
    # bools = np.abs(net) > 0.1 * N
    # signs = np.sign(net)
    # leaving[bools] = 0.1 * N[bools]
    # entering[bools] = 0.1 * N[bools]

    # compute percentage of each population
    # pS = S[:, t] / N
    # pI = I[:, t] / N
    # pR = 1 - pS - pI
    #
    # # generate random movements of
    # SIRleaving = np.array([np.random.multinomial(leaving[c], [pS[c], pI[c], pR[c]]) for c in range(nc)])
    # SIRentering = np.array([np.random.multinomial(entering[c], [pS[c], pI[c], pR[c]]) for c in range(nc)])
    #
    # SIRnet_flow = SIRentering - SIRleaving
    # N = N + (entering - leaving)

    # print(np.sum(N))
    #
    # print(SIRnet_flow[:,1])
    # Snew = S[:, t] + SIRnet_flow[:, 0]
    # Inew = I[:, t] + SIRnet_flow[:, 1]
    # Rnew = R[:, t] + SIRnet_flow[:, 2]

    # S[:, t + 1] = np.max([np.zeros(nc), Snew - (beta[t] * Inew * Snew) / N], axis=0)
    S[:, t + 1] = np.max([np.zeros(nc), S[:,t] - (beta[t] * S[:,t] * I[:,t]) / N - (alpha[t] * S[:,t] * np.dot(FLOWn, beta[t]*I[:,t] /N))/(N+np.sum(FLOWn,axis=1)) ], axis=0)
    jjj = ii[8]
    # print((alpha[t] * S[jjj,t] * np.dot(FLOWn, beta[t]*I[:,t] /N)[jjj])/(N[jjj]+np.sum(FLOWn,axis=1)[jjj]))
    S[:, t + 1] = np.min([S[:, t + 1], N],axis=0)
    I[:, t + 1] = np.max([np.zeros(nc), (1 - gamma[t]) * I[:,t] + (beta[t] * S[:,t] * I[:,t]) / N + (alpha[t] * S[:,t] * np.dot(FLOWn, beta[t]*I[:,t] /N))/(N+np.sum(FLOWn,axis=1))], axis=0)
    I[:, t + 1] = np.min([I[:, t + 1], N], axis=0)
    R[:, t + 1] = np.max([np.zeros(nc), R[:,t] + gamma[t] * I[:,t]], axis=0)
    R[:, t + 1] = np.min([R[:, t + 1], N], axis=0)
    Np = np.sum(S[:,t+1]) + np.sum(I[:,t+1]) + np.sum(R[:,t+1])
    print(Np)
c = ii[8]
plotgraph = True
if plotgraph:
    plt.plot(np.sum(S,axis=0),'b')
    plt.plot(np.sum(I,axis=0),'k')
    plt.plot(np.sum(R,axis=0),'g')
    plt.show()
plotmap = False
if plotmap:
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