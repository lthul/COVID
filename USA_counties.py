from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
import plotly.figure_factory as ff
import pandas as pd
import data_loader as dl
import numpy as np
import class_file as cf
import plotly.express as px
# df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
#                    dtype={"fips": str})


colorscale = ["#f7fbff", "#ebf3fb", "#deebf7", "#d2e3f3", "#c6dbef", "#b3d2e9", "#9ecae1",
    "#85bcdb", "#6baed6", "#57a0ce", "#4292c6", "#3082be", "#2171b5", "#1361a9",
    "#08519c", "#0b4083", "#08306b"
]
counties = cf.DATA_LOAD()



# endpts = list(np.linspace(1, 1000000, len(colorscale) - 1))
# fig = ff.create_choropleth(
#     fips=FIPS, values=CASES, scope=['usa'],
#     binning_endpoints=endpts, colorscale=colorscale,
#     show_state_data=False,
#     show_hover=True,
# )
# fig.layout.template = None
# fig.show()

tot_case_dict = cf.getFIPSList(counties, key='tot_cases')

endpts = list(np.logspace(0, 6, len(colorscale)-1))
fig = ff.create_choropleth(
    fips=list(tot_case_dict.keys()), values=list(tot_case_dict.values()), scope=['usa'],
    binning_endpoints=endpts, colorscale=colorscale,
    show_state_data=False,
    show_hover=True,
)
fig.layout.template = None
fig.show()