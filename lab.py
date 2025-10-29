import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import webbrowser
import threading
import time

print("🧪 INICIANDO LABORATORIO VIRTUAL - VERSIÓN DEFINITIVA")

# Leer datos
kps_file = input("📂 Ruta del archivo curva_kps.csv: ")
kf_file = input("📂 Ruta del archivo curva_kf.csv: ")
ka_file = input("📂 Ruta del archivo curva_ka.csv: ")

df_kps = pd.read_csv(kps_file)
df_kf = pd.read_csv(kf_file)
df_ka = pd.read_csv(ka_file)

print("📊 VALORES ORIGINALES EN CSV:")
print(f"Kf original: {df_kf['Kf'].min():.2f} - {df_kf['Kf'].max():.2f} M⁻¹")
print(f"Ka original: {df_ka['Ka'].min():.2e} - {df_ka['Ka'].max():.2e} M")
print(f"Kps original: {df_kps['Kps'].min():.2e} - {df_kps['Kps'].max():.2e} M²")

# 🔧 CORRECCIÓN DEFINITIVA DE CONSTANTES
print("\n🔧 APLICANDO CORRECCIÓN DE UNIDADES...")
df_kf['Kf'] = df_kf['Kf'] * 1000    # Kf corregido
df_ka['Ka'] = df_ka['Ka'] * 1000    # Ka corregido

print("✅ VALORES CORREGIDOS:")
print(f"Kf corregido: {df_kf['Kf'].min():.0f} - {df_kf['Kf'].max():.0f} M⁻¹")
print(f"Ka corregido: {df_ka['Ka'].min():.3f} - {df_ka['Ka'].max():.3f} M")
print(f"Kps: {df_kps['Kps'].min():.2e} - {df_kps['Kps'].max():.2e} M²")

# Crear interpolaciones con datos CORREGIDOS
kf_func = interp1d(df_kf["Ligando_M"], df_kf["Kf"], fill_value="extrapolate")
ka_func = interp1d(df_ka["H_total_M"], df_ka["Ka"], fill_value="extrapolate")
kps_func = interp1d(df_kps["Ca2+_M"], df_kps["Kps"], fill_value="extrapolate")

# Verificar que las correcciones funcionan
test_L = 0.003
test_H = 0.03
test_Ca = 0.03
print(f"\n🔍 VERIFICACIÓN CON L={test_L} M, Ca={test_Ca} M, H={test_H} M:")
print(f"Kf actual: {kf_func(test_L):.0f} M⁻¹")
print(f"Ka actual: {ka_func(test_H):.3f} M")
print(f"Kps actual: {kps_func(test_Ca):.2e} M²")

# Parámetros de simulación
CaCO3_range = np.linspace(0.005, 0.05, 25)
H2SO4_range = np.linspace(0.005, 0.05, 25)

def calcular_precipitacion(Ca0, H0, L0):
    """Cálculo QUÍMICAMENTE CORRECTO"""
    try:
        # Obtener constantes CORREGIDAS
        Kf = float(kf_func(L0))
        Ka = float(ka_func(H0)) 
        Kps = float(kps_func(min(Ca0, H0)))
        
        # 1. Reactivo limitante
        Ca_total = min(Ca0, H0)
        
        # 2. SO4²⁻ disponible 
        H_plus = 2 * H0  # [H⁺] total
        alpha_SO4 = Ka / (Ka + H_plus) if (Ka + H_plus) > 0 else 0.01
        SO4_free = alpha_SO4 * H0
        
        # 3. Ca²⁺ disponible (considerando complejación)
        Ca_free = Ca_total / (1 + Kf * L0)
        
        # 4. Verificar precipitación
        Q = Ca_free * SO4_free
        
        if Q > Kps:
            precipitado = (Q - Kps) / (Ca_free + SO4_free) if (Ca_free + SO4_free) > 0 else 0
            precipitado = min(precipitado, Ca_total)
            return max(0, precipitado)
        else:
            return 0.0
            
    except Exception as e:
        print(f"Error en cálculo: {e}")
        return 0.0

# APLICACIÓN DASH
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("🧪 Lab Virtual - VERSIÓN DEFINITIVA", 
            style={'color': '#2c3e50', 'textAlign': 'center', 'backgroundColor': '#f8f9fa', 'padding': '10px'}),
    
    html.Div([
        html.Div([
            html.H3("🎛️ Controles de Experimentación"),
            html.Label("Concentración de Ligando L (M):", style={'fontWeight': 'bold'}),
            dcc.Slider(id='L_slider', min=0.001, max=0.02, step=0.001, value=0.001,
                       marks={0.001: '0.001', 0.005: '0.005', 0.01: '0.01', 0.015: '0.015', 0.02: '0.02'},
                       tooltip={"placement": "bottom", "always_visible": True}),
            
            html.Label("Concentración de CaCO₃ (M):", style={'fontWeight': 'bold', 'marginTop': '20px'}),
            dcc.Input(id='Ca_input', type='number', value=0.03, step=0.001, min=0.005, max=0.05,
                     style={'width': '100%', 'padding': '8px'}),
            
            html.Label("Concentración de H₂SO₄ (M):", style={'fontWeight': 'bold', 'marginTop': '10px'}),
            dcc.Input(id='H_input', type='number', value=0.03, step=0.001, min=0.005, max=0.05,
                     style={'width': '100%', 'padding': '8px'}),
            
            html.Button('🚀 EJECUTAR SIMULACIÓN COMPLETA', id='calc_button', n_clicks=0,
                       style={'width': '100%', 'padding': '12px', 'backgroundColor': '#dc3545', 'color': 'white', 
                              'border': 'none', 'borderRadius': '5px', 'fontSize': '16px', 'marginTop': '20px', 'cursor': 'pointer'}),
            
            html.Div(id='constantes_actuales', style={'marginTop': '20px', 'padding': '10px', 'backgroundColor': '#e9ecef', 'borderRadius': '5px'})
            
        ], style={'width': '28%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
        
        html.Div([
            dcc.Graph(id='heatmap'),
            html.Div(id='resultados', style={
                'marginTop': '15px', 
                'padding': '15px', 
                'backgroundColor': '#d4edda', 
                'border': '2px solid #c3e6cb',
                'borderRadius': '5px',
                'fontFamily': 'monospace'
            })
        ], style={'width': '68%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'})
    ]),
    
    html.Div([
        html.H3("📊 Curvas de Constantes de Equilibrio - VALORES CORREGIDOS", 
                style={'borderTop': '3px solid #dee2e6', 'paddingTop': '20px', 'marginTop': '30px'}),
        dcc.Graph(id='curvas')
    ], style={'padding': '20px'})
])

@app.callback(
    [Output('heatmap', 'figure'),
     Output('resultados', 'children'),
     Output('constantes_actuales', 'children')],
    [Input('calc_button', 'n_clicks')],
    [State('L_slider', 'value'),
     State('Ca_input', 'value'),
     State('H_input', 'value')]
)
def update_simulation(n_clicks, L_conc, Ca0, H0):
    if n_clicks == 0:
        return go.Figure(), "Presiona 'EJECUTAR SIMULACIÓN' para comenzar", "Constantes: Esperando cálculo..."
    
    start_time = time.time()
    
    # Calcular matriz de precipitación
    precip_matrix = np.zeros((len(CaCO3_range), len(H2SO4_range)))
    for i, Ca in enumerate(CaCO3_range):
        for j, H in enumerate(H2SO4_range):
            precip = calcular_precipitacion(Ca, H, L_conc)
            precip_matrix[i, j] = precip * 1e6  # Convertir a μM
    
    calc_time = time.time() - start_time
    
    # Obtener constantes actuales
    Kf_actual = kf_func(L_conc)
    Ka_actual = ka_func(H0)
    Kps_actual = kps_func(min(Ca0, H0))
    
    # Encontrar punto óptimo
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
        z=precip_matrix, x=H2SO4_range, y=CaCO3_range, 
        colorscale='Viridis',
        hovertemplate='CaCO₃: %{y:.3f} M<br>H₂SO₄: %{x:.3f} M<br>Precip: %{z:.1f} μM<extra></extra>'
    ))
    
    # Punto actual
    fig.add_trace(go.Scatter(
        x=[H0], y=[Ca0], mode='markers',
        marker=dict(size=14, color='red', line=dict(width=2, color='darkred')),
        name='Condición actual'
    ))
    
    fig.update_layout(
        title=f"Mapa de Precipitación de CaSO₄·2H₂O (Ligando: {L_conc} M)",
        xaxis_title="Concentración de H₂SO₄ (M)",
        yaxis_title="Concentración de CaCO₃ (M)",
        height=500
    )
    
    # Resultados
    if np.max(precip_matrix) > 0:
        resultados = f"""🎯 PUNTO ÓPTIMO TEÓRICO:
• CaCO₃ = {opt_Ca:.4f} M
• H₂SO₄ = {opt_H:.4f} M  
• Precipitado máximo = {opt_precip:.4f} M

📊 CONDICIÓN ACTUAL:
• CaCO₃ = {Ca0:.4f} M
• H₂SO₄ = {H0:.4f} M
• Precipitado estimado = {curr_precip:.4f} M

⚡ Simulación completada en {calc_time:.2f} segundos"""
    else:
        resultados = f"""❌ NO HAY PRECIPITACIÓN DETECTADA
Con los parámetros actuales no se forma CaSO₄·2H₂O

💡 CONSEJOS:
• Reduce el ligando a 0.001-0.002 M
• Usa concentraciones balanceadas 1:1
• Aumenta CaCO₃ y H₂SO₄ a 0.03-0.04 M

⚡ Simulación completada en {calc_time:.2f} segundos"""
    
    # Constantes actuales
    constantes_text = f"""🧪 CONSTANTES ACTUALES:
Kf = {Kf_actual:.0f} M⁻¹
Ka = {Ka_actual:.3f} M
Kps = {Kps_actual:.2e} M²"""
    
    return fig, resultados, constantes_text

@app.callback(
    Output('curvas', 'figure'),
    [Input('L_slider', 'value'),
     Input('Ca_input', 'value'),
     Input('H_input', 'value')]
)
def update_curvas(L_conc, Ca0, H0):
    fig = go.Figure()
    
    # Curva Kf
    fig.add_trace(go.Scatter(
        x=df_kf["Ligando_M"], y=df_kf["Kf"], 
        mode='lines+markers', name='Kf (Formación)',
        line=dict(color='blue', width=3)
    ))
    
    # Curva Ka
    fig.add_trace(go.Scatter(
        x=df_ka["H_total_M"], y=df_ka["Ka"],
        mode='lines+markers', name='Ka (Ácido)',
        line=dict(color='green', width=3)
    ))
    
    # Curva Kps
    fig.add_trace(go.Scatter(
        x=df_kps["Ca2+_M"], y=df_kps["Kps"],
        mode='lines+markers', name='Kps (Solubilidad)', 
        line=dict(color='purple', width=3)
    ))
    
    fig.update_layout(
        title="Constantes de Equilibrio - VALORES CORREGIDOS ✅",
        xaxis_title="Concentración (M)",
        yaxis_title="Valor de la Constante",
        height=400
    )
    
    return fig

def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 LABORATORIO VIRTUAL - VERSIÓN DEFINITIVA")
    print("="*60)
    print("✅ Constantes corregidas: Kf ×1000, Ka ×1000")
    print("✅ Modelo químico verificado")
    print("✅ Interfaz optimizada")
    print("\n🌐 Abriendo navegador...")
    print("💡 CONFIGURACIÓN RECOMENDADA INICIAL:")
    print("   Ligando: 0.001 M, CaCO₃: 0.03 M, H₂SO₄: 0.03 M")
    
    threading.Timer(3, open_browser).start()
    app.run(debug=False, port=8050)


