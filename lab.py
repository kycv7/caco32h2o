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
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Laboratorio Virtual Extremo Interactivo de CaSO₄·2H₂O"),
    html.Div([
        html.Label("Concentración de ligando L (M)"),
        dcc.Slider(id='L_slider', min=0.001, max=0.02, step=0.001, value=0.01,
                   marks={i/1000: f'{i/1000:.3f}' for i in range(1, 21, 2)}),
        html.Label("Punto inicial de proceso (CaCO₃ y H₂SO₄)"),
        html.Div([
            html.Label("CaCO₃ inicial (M)"),
            dcc.Input(id='Ca_input', type='number', value=0.025, step=0.001, min=0.005, max=0.05)
        ]),
        html.Div([
            html.Label("H₂SO₄ inicial (M)"),
            dcc.Input(id='H_input', type='number', value=0.025, step=0.001, min=0.005, max=0.05)
        ])
    ], style={'width': '48%', 'display': 'inline-block', 'padding': '20px'}),
    html.Div([
        dcc.Graph(id='heatmap_precip', config={'editable': True}),
        dcc.Graph(id='curvas'),
        html.Div(id='punto_optimo')
    ], style={'width': '48%', 'display': 'inline-block', 'padding': '20px'})
])

@app.callback(
    [Output('heatmap_precip', 'figure'),
     Output('curvas', 'figure'),
     Output('punto_optimo', 'children')],
    [Input('L_slider', 'value'),
     Input('Ca_input', 'value'),
     Input('H_input', 'value')]
)
def laboratorio_extremo(L_conc_slider, Ca0_slider, H0_slider):
    precip_matrix = np.zeros((len(CaCO3_list), len(H2SO4_list)))

    # Calcular matriz de precipitación
    for i, Ca0 in enumerate(CaCO3_list):
        for j, H0 in enumerate(H2SO4_list):
            Ka_eff = Ka_func(H0)
            alpha_SO4 = Ka_eff / (Ka_eff + H0)
            SO4_eff = alpha_SO4 * H0

            H_needed = Ca0
            Ca2_plus_total = Ca0 if H0 >= H_needed else H0

            Kf_eff = Kf_func(L_conc_slider)
            Ca2_plus_eff = Ca2_plus_total / (1 + Kf_eff * L_conc_slider)

            precip_matrix[i, j] = max(min(Ca2_plus_eff, SO4_eff) - np.sqrt(np.mean(df_Kps["Kps"])), 0)

    # Punto óptimo
    max_idx = np.unravel_index(np.argmax(precip_matrix), precip_matrix.shape)
    optimal_Ca = CaCO3_list[max_idx[0]]
    optimal_H = H2SO4_list[max_idx[1]]
    optimal_precip = precip_matrix[max_idx]

    # Heatmap interactivo con punto arrastrable
    fig_heat = go.Figure(data=go.Heatmap(
        z=precip_matrix,
        x=H2SO4_list,
        y=CaCO3_list,
        colorscale='Viridis',
        colorbar=dict(title='Precipitado (M)')
    ))
    fig_heat.add_trace(go.Scatter(
        x=[H0_slider], y=[Ca0_slider], mode='markers',
        marker=dict(size=15, color='red'), name='Punto del proceso'
    ))

    # Curvas con punto actual
    fig_curvas = go.Figure()
    fig_curvas.add_trace(go.Scatter(x=df_Kf["Ligando_M"], y=df_Kf["Kf"], mode='lines+markers', name='Kf'))
    fig_curvas.add_trace(go.Scatter(x=df_Ka["H_total_M"], y=df_Ka["Ka"], mode='lines+markers', name='Ka'))
    fig_curvas.add_trace(go.Scatter(x=df_Kps["Ca2+_M"], y=df_Kps["Kps"], mode='lines+markers', name='Kps'))
    fig_curvas.add_trace(go.Scatter(x=[L_conc_slider], y=[Kf_func(L_conc_slider)], mode='markers',
                                    marker=dict(size=12, color='red'), name='Punto L'))
    fig_curvas.add_trace(go.Scatter(x=[H0_slider], y=[Ka_func(H0_slider)], mode='markers',
                                    marker=dict(size=12, color='orange'), name='Punto H₂SO₄'))
    fig_curvas.add_trace(go.Scatter(x=[Ca0_slider], y=[Kps_func(Ca0_slider)], mode='markers',
                                    marker=dict(size=12, color='green'), name='Punto CaCO₃'))

    texto_optimo = f"Punto óptimo: CaCO₃ = {optimal_Ca:.4f} M, H₂SO₄ = {optimal_H:.4f} M, Precipitado = {optimal_precip:.4f} M\n"
    texto_optimo += f"Punto actual: CaCO₃ = {Ca0_slider:.4f} M, H₂SO₄ = {H0_slider:.4f} M, Precipitado = {precip_matrix[(np.abs(CaCO3_list - Ca0_slider)).argmin(), (np.abs(H2SO4_list - H0_slider)).argmin()]:.4f} M"

    return fig_heat, fig_curvas, texto_optimo

if __name__ == '__main__':
    app.run_server(debug=True)
