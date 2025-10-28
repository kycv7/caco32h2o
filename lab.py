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

