import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import threading
import time

print("🧪 estudio análitico precipitación de CaSO4.2H2O")

# Cargar y corregir datos
kps_file = input("📂 Ruta del archivo curva_kps.csv: ")
kf_file = input("📂 Ruta del archivo curva_kf.csv: ")
ka_file = input("📂 Ruta del archivo curva_ka.csv: ")

df_kps = pd.read_csv(kps_file)
df_kf = pd.read_csv(kf_file)
df_ka = pd.read_csv(ka_file)

# Correcciones
df_kf['Kf'] = df_kf['Kf'] * 1000
df_ka['Ka'] = df_ka['Ka'] * 1000

print(f"✅ Kf: {df_kf['Kf'].min():.0f} - {df_kf['Kf'].max():.0f} M⁻¹")
print(f"✅ Ka: {df_ka['Ka'].min():.3f} - {df_ka['Ka'].max():.3f} M")

# Interpolaciones
kf_func = interp1d(df_kf["Ligando_M"], df_kf["Kf"], fill_value="extrapolate")
ka_func = interp1d(df_ka["H_total_M"], df_ka["Ka"], fill_value="extrapolate")
kps_func = interp1d(df_kps["Ca2+_M"], df_kps["Kps"], fill_value="extrapolate")

# Parámetros
CaCO3_range = np.linspace(0.005, 0.05, 25)
H2SO4_range = np.linspace(0.005, 0.05, 25)

def calcular_precipitacion(Ca0, H0, L0):
    try:
        Kf = float(kf_func(L0))
        Ka = float(ka_func(H0))
        Kps = float(kps_func(min(Ca0, H0)))
        
        Ca_total = min(Ca0, H0)
        H_plus = 2 * H0
        alpha_SO4 = Ka / (Ka + H_plus) if (Ka + H_plus) > 0 else 0.01
        SO4_free = alpha_SO4 * H0
        Ca_free = Ca_total / (1 + Kf * L0)
        Q = Ca_free * SO4_free
        
        if Q > Kps:
            precipitado = (Q - Kps) / (Ca_free + SO4_free) if (Ca_free + SO4_free) > 0 else 0
            return min(max(0, precipitado), Ca_total)
        return 0.0
    except:
        return 0.0

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("🧪 estudio análitico precipitación de CaSO4.2H2O", style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    # PANEL DE CONTROL
    html.Div([
        html.Div([
            html.H3("🎛️ CONTROLES"),
            html.Label("Ligando L (M):", style={'fontWeight': 'bold'}),
            dcc.Slider(id='L_slider', min=0.001, max=0.02, step=0.001, value=0.001,
                       marks={0.001: '0.001', 0.005: '0.005', 0.01: '0.01', 0.015: '0.015', 0.02: '0.02'},
                       tooltip={"placement": 'bottom', "always_visible": True}),
            
            html.Label("CaCO₃ (M):", style={'fontWeight': 'bold', 'marginTop': '20px'}),
            dcc.Input(id='Ca_input', type='number', value=0.01, step=0.001),
            
            html.Label("H₂SO₄ (M):", style={'fontWeight': 'bold', 'marginTop': '10px'}),
            dcc.Input(id='H_input', type='number', value=0.05, step=0.001),
            
            html.Button('🚀 CALCULAR', id='calc_button', n_clicks=0,
                       style={'width': '100%', 'padding': '12px', 'backgroundColor': '#28a745', 
                              'color': 'white', 'border': 'none', 'marginTop': '20px'})
            
        ], style={'width': '25%', 'display': 'inline-block', 'padding': '20px', 'verticalAlign': 'top'}),
        
        # RESULTADOS
        html.Div([
            dcc.Graph(id='heatmap'),
            html.Div(id='resultados', style={
                'marginTop': '15px', 'padding': '15px', 'backgroundColor': '#e7f3ff', 
                'borderRadius': '8px', 'border': '2px solid #007bff'
            })
        ], style={'width': '70%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'})
    ]),
    
    # CURVAS
    html.Div([
        html.H3("📊 ANÁLISIS DE CONSTANTES DE EQUILIBRIO", 
                style={'borderTop': '3px solid #dee2e6', 'paddingTop': '30px', 'marginTop': '30px'}),
        
        html.Div([
            html.Div([
                html.H4("Kf - Constante de Formación del Complejo", style={'color': '#007bff'}),
                dcc.Graph(id='curva_kf')
            ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px'}),
            
            html.Div([
                html.H4("Ka - Constante de Acidez", style={'color': '#28a745'}),
                dcc.Graph(id='curva_ka')
            ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px'}),
            
            html.Div([
                html.H4("Kps - Producto de Solubilidad", style={'color': '#dc3545'}),
                dcc.Graph(id='curva_kps')
            ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px'})
        ]),
        
        # GRÁFICO DE EFECTO DEL LIGANDO
        html.Div([
            html.H4("📈 Efecto del Ligando en la Precipitación", style={'marginTop': '30px', 'color': '#6f42c1'}),
            dcc.Graph(id='efecto_ligando')
        ])
    ], style={'padding': '20px'})
])

# Callback para heatmap y resultados
@app.callback(
    [Output('heatmap', 'figure'),
     Output('resultados', 'children')],
    [Input('calc_button', 'n_clicks')],
    [State('L_slider', 'value'),
     State('Ca_input', 'value'),
     State('H_input', 'value')]
)
def update_simulation(n_clicks, L_conc, Ca0, H0):
    if n_clicks == 0:
        return go.Figure(), "💡 **CONSEJO:** Usa ligando <0.005 M para máxima precipitación"
    
    start_time = time.time()
    
    # Calcular matriz
    precip_matrix = np.zeros((len(CaCO3_range), len(H2SO4_range)))
    for i, Ca in enumerate(CaCO3_range):
        for j, H in enumerate(H2SO4_range):
            precip_matrix[i,j] = calcular_precipitacion(Ca, H, L_conc) * 1e6  # μM
    
    calc_time = time.time() - start_time
    
    # Encontrar óptimo
    if np.max(precip_matrix) > 0:
        i_opt, j_opt = np.unravel_index(np.argmax(precip_matrix), precip_matrix.shape)
        opt_Ca, opt_H = CaCO3_range[i_opt], H2SO4_range[j_opt]
        opt_precip = precip_matrix[i_opt, j_opt] / 1e6
    else:
        opt_Ca, opt_H, opt_precip = 0, 0, 0
    
    # Precipitado actual
    i_curr = np.argmin(np.abs(CaCO3_range - Ca0))
    j_curr = np.argmin(np.abs(H2SO4_range - H0))
    curr_precip = precip_matrix[i_curr, j_curr] / 1e6
    
    # Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=precip_matrix, x=H2SO4_range, y=CaCO3_range, colorscale='Viridis',
        hovertemplate='CaCO₃: %{y:.3f} M<br>H₂SO₄: %{x:.3f} M<br>Precip: %{z:.1f} μM<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=[H0], y=[Ca0], mode='markers', name='Actual',
        marker=dict(size=12, color='red', line=dict(width=2, color='darkred'))
    ))
    
    if np.max(precip_matrix) > 0:
        fig.add_trace(go.Scatter(
            x=[opt_H], y=[opt_Ca], mode='markers', name='Óptimo',
            marker=dict(size=12, color='gold', line=dict(width=2, color='orange'))
        ))
    
    fig.update_layout(
        title=f"Precipitación de CaSO₄·2H₂O (Ligando: {L_conc} M)",
        xaxis_title="H₂SO₄ (M)", yaxis_title="CaCO₃ (M)", height=450
    )
    
    # Resultados
    Kf_act = kf_func(L_conc)
    Ka_act = ka_func(H0)
    Kps_act = kps_func(min(Ca0, H0))
    
    resultados = f"""
🎯 **PUNTO ÓPTIMO TEÓRICO:**
• CaCO₃ = {opt_Ca:.4f} M | H₂SO₄ = {opt_H:.4f} M  
• Precipitado máximo = {opt_precip:.4f} M ({opt_precip*1000:.1f} mM)

📊 **CONDICIÓN ACTUAL:**
• CaCO₃ = {Ca0:.4f} M | H₂SO₄ = {H0:.4f} M
• Precipitado estimado = {curr_precip:.4f} M ({curr_precip*1000:.1f} mM)

🧪 **CONSTANTES ACTUALES:**
• Kf = {Kf_act:.0f} M⁻¹ | Ka = {Ka_act:.3f} M | Kps = {Kps_act:.2e} M²

⚡ **Simulación completada en {calc_time:.2f} segundos**

💡 **INTERPRETACIÓN:**
• Ligando > 0.01 M reduce drásticamente la precipitación
• Concentraciones balanceadas mejoran el rendimiento
• H₂SO₄ alto compensa el efecto del ligando
"""
    
    return fig, resultados

# Callbacks separados para cada curva
@app.callback(Output('curva_kf', 'figure'), [Input('L_slider', 'value')])
def update_curva_kf(L_conc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_kf["Ligando_M"], y=df_kf["Kf"], 
                            mode='lines+markers', name='Kf', line=dict(color='#007bff', width=3)))
    fig.add_trace(go.Scatter(x=[L_conc], y=[kf_func(L_conc)], 
                            mode='markers', name='Actual', marker=dict(size=12, color='red')))
    fig.update_layout(title="Kf vs [Ligando]", xaxis_title="[Ligando] (M)", yaxis_title="Kf (M⁻¹)", height=300)
    return fig

@app.callback(Output('curva_ka', 'figure'), [Input('H_input', 'value')])
def update_curva_ka(H0):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_ka["H_total_M"], y=df_ka["Ka"], 
                            mode='lines+markers', name='Ka', line=dict(color='#28a745', width=3)))
    fig.add_trace(go.Scatter(x=[H0], y=[ka_func(H0)], 
                            mode='markers', name='Actual', marker=dict(size=12, color='red')))
    fig.update_layout(title="Ka vs [H⁺]", xaxis_title="[H⁺] (M)", yaxis_title="Ka (M)", height=300)
    return fig

@app.callback(Output('curva_kps', 'figure'), [Input('Ca_input', 'value')])
def update_curva_kps(Ca0):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_kps["Ca2+_M"], y=df_kps["Kps"], 
                            mode='lines+markers', name='Kps', line=dict(color='#dc3545', width=3)))
    fig.add_trace(go.Scatter(x=[Ca0], y=[kps_func(Ca0)], 
                            mode='markers', name='Actual', marker=dict(size=12, color='red')))
    fig.update_layout(title="Kps vs [Ca²⁺]", xaxis_title="[Ca²⁺] (M)", yaxis_title="Kps (M²)", height=300)
    return fig

@app.callback(Output('efecto_ligando', 'figure'), [Input('L_slider', 'value')])
def update_efecto_ligando(L_conc):
    # Calcular efecto del ligando en precipitación fija
    ligando_range = np.linspace(0.001, 0.02, 50)
    precipitacion = [calcular_precipitacion(0.03, 0.03, L) * 1e6 for L in ligando_range]  # μM
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ligando_range, y=precipitacion, 
                            mode='lines', name='Precipitación', line=dict(color='#6f42c1', width=3)))
    fig.add_trace(go.Scatter(x=[L_conc], y=[calcular_precipitacion(0.03, 0.03, L_conc) * 1e6], 
                            mode='markers', name='Actual', marker=dict(size=10, color='red')))
    
    fig.update_layout(
        title="Efecto del Ligando en la Precipitación (CaCO₃ = H₂SO₄ = 0.03 M)",
        xaxis_title="Concentración de Ligando (M)", 
        yaxis_title="Precipitación (μM)",
        height=350
    )
    
    # Añadir zona crítica
    fig.add_vrect(x0=0.01, x1=0.02, fillcolor="red", opacity=0.1, line_width=0, 
                  annotation_text="Zona de baja precipitación", annotation_position="top left")
    fig.add_vrect(x0=0.001, x1=0.005, fillcolor="green", opacity=0.1, line_width=0,
                  annotation_text="Zona óptima", annotation_position="top right")
    
    return fig

def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")

if __name__ == '__main__':
    print("🚀 EJECUTANDO...")
    threading.Timer(2, open_browser).start()
    app.run(debug=False, port=8050)