import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
#leer CSV
kps_file=input("ingresa la ruta del CSV de kps: ")
kf_file=input("ingresa la ruta del csv de kf: ")
ka_file=input("ingresa la ruta del csv de ka: ")
df_kps=pd.read_csv(kps_file)
df_kf=pd.read_csv(kf_file)
df_ka=pd.read_csv(ka_file)
#interpolación
kf_func=interp1d(df_kf["Ligando_M"],df_kf["Kf"],fill_value="extrapolate")
ka_func=interp1d(df_ka["H_total_M"],df_ka["Ka"],fill_value="extrapolate")
kps_func=interp1d(df_kps["Ca2+_M"],df_kps["Kps"],fill_value="extrapolate")
#parámetros y rango
CaCO3_list=np.linspace(0.005,0.05,30)
H2SO4_list=np.linspace(0.005,0.05,30)
#dash app
app=dash.Dash(__name__)
app.layout=html.Div([
    html.H1("estudio de formación de precipitados CaSO4.2H2O"),
    html.Div([
        html.Label("Concentración de ligando L (M)"),
        dcc.Slider(id='L_slider',min=0.001,max=0.02,step=0.001,value=0.01,marks={i/1000:f'{i/1000:.3f}'for i in range(1,21,2)}),
        html.Label("punto inicial del proceso (CaCO3 + H2SO4)"),
        html.Div([
            html.Label("CaCO3 inicial (M)"),
            dcc.Input(id='Ca_input',type='number',value=0.025,step=0.001,min=0.005,max=0.05)
        ]),
        html.Div([
            html.Label("H2SO4 inicial (M)"),
            dcc.Input(id='H_input',type='number',value=0.025,step=0.001,min=0.005,max=0.05)
        ])
    ],style={'witdh':'48%','display':'inline-black','padding':'20px'}),
    html.Div([
        dcc.Graph(id='heatmap_precip',config={'editable':True}),
        dcc.Graph(id='curvas'),
        html.Div(id='punto_optimo')
    ],style={'width':'48%','display':'inline-block','padding':'20px'})
])

 