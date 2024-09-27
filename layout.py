from dash import html, dcc

layout = html.Div([
    html.Div([
        html.Div([
            # Section Fichiers
            html.H2("Fichiers", style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Dropdown(id='file-dropdown1', options=[], placeholder='Select Ref', style={'marginBottom': '20px'}),
            dcc.Dropdown(id='file-dropdown2', options=[], placeholder='Select Pred'),
            
            # Section Dates
            html.H2("Dates", style={'textAlign': 'center', 'padding': '10px'}),
            html.Div([
                html.H4("Start Time", style={'textAlign': 'center'}),
                html.Div([
                    dcc.Dropdown(id='start-year-dropdown', options=[], placeholder='Year', style={'width': '80px', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='start-month-dropdown', 
                        options=[{'label': str(i), 'value': i} for i in range(1, 13)], 
                        placeholder='Month', 
                        style={'width': '80px', 'marginRight': '10px'}
                    ),
                    dcc.Dropdown(
                        id='start-day-dropdown', 
                        options=[{'label': str(i), 'value': i} for i in range(1, 32)], 
                        placeholder='Day', 
                        style={'width': '80px', 'marginRight': '10px'}
                    ),
                    dcc.Dropdown(id='start-hour-dropdown', options=[], placeholder='Hour', style={'width': '80px'}),
                ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '10px'}),
            ], style={'textAlign': 'center'}),
            html.Div([
                html.H4("End Time", style={'textAlign': 'center', 'marginTop': '20px'}),
                html.Div([
                    dcc.Dropdown(id='end-year-dropdown', options=[], placeholder='Year', style={'width': '80px', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='end-month-dropdown', 
                        options=[{'label': str(i), 'value': i} for i in range(1, 13)], 
                        placeholder='Month', 
                        style={'width': '80px', 'marginRight': '10px'}
                    ),
                    dcc.Dropdown(
                        id='end-day-dropdown', 
                        options=[{'label': str(i), 'value': i} for i in range(1, 32)], 
                        placeholder='Day', 
                        style={'width': '80px', 'marginRight': '10px'}
                    ),
                    dcc.Dropdown(id='end-hour-dropdown', options=[], placeholder='Hour', style={'width': '80px'}),
                ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '10px'}),
            ], style={'textAlign': 'center'}),
            html.Div(id='time-period1', style={'textAlign': 'center', 'marginTop': '10px'}),
            html.Div(id='time-period2', style={'textAlign': 'center', 'marginTop': '10px'}),
            html.Div(id='time-period11', style={'textAlign': 'center', 'marginTop': '10px'}),

            # Section Variables
            html.H2("Variables", style={'textAlign': 'center', 'padding': '10px'}),
            dcc.Dropdown(id='variable-dropdown1', options=[], placeholder='Select Variable Ref', style={'marginBottom': '20px'}),
            html.Div(id='forecast-number-container1', children=[
                dcc.Input(id='forecast-number-input1', type='number', min=0, placeholder='Forecast Number Ref', style={'width': '80%', 'padding': '10px', 'fontSize': '16px', 'textAlign': 'center'})
            ], style={'textAlign': 'center', 'marginBottom': '20px', 'display': 'none'}),
            dcc.Dropdown(id='variable-dropdown2', options=[], placeholder='Select Variable Pred', style={'marginBottom': '20px'}),
            html.Div(id='forecast-number-container2', children=[
                dcc.Input(id='forecast-number-input2', type='number', min=0, placeholder='Forecast Number Pred', style={'width': '80%', 'padding': '10px', 'fontSize': '16px', 'textAlign': 'center'})
            ], style={'textAlign': 'center', 'marginBottom': '20px', 'display': 'none'}),
            dcc.Dropdown(id='metrics-variables-options2', options=[], placeholder='Select Pred Metric', style={'marginBottom': '20px'}),

            # Section Range
            html.H2("Range", style={'textAlign': 'center', 'padding': '10px'}),
            html.Div([
                html.H4("Coordonnées", style={'textAlign': 'center'}),
                dcc.Input(id='range-x', type='number', min=0, placeholder='X', style={'width': '80px', 'textAlign': 'center', 'marginRight': '10px'}),
                dcc.Input(id='range-y', type='number', min=0, placeholder='Y', style={'width': '80px', 'textAlign': 'center'}),
            ], style={'textAlign': 'center', 'marginBottom': '10px'}),
            html.Div([
                html.H4("Rayon", style={'textAlign': 'center', 'marginTop': '20px'}),
                dcc.Input(id='range-rx', type='number', min=0, placeholder='Rx', style={'width': '80px', 'textAlign': 'center', 'marginRight': '10px'}),
                dcc.Input(id='range-ry', type='number', min=0, placeholder='Ry', style={'width': '80px', 'textAlign': 'center'}),
            ], style={'textAlign': 'center', 'marginBottom': '10px'}),
            html.Div([
                html.H4("Level", style={'textAlign': 'center', 'marginTop': '20px'}),
                dcc.Dropdown(id='range-z1-dropdown', options=[], placeholder='Select Level Ref', style={'width': '120px', 'display': 'inline-block', 'marginRight': '10px'}),
                dcc.Dropdown(id='range-z2-dropdown', options=[], placeholder='Select Level Pred', style={'width': '120px', 'display': 'inline-block'}),
            ], style={'textAlign': 'center','marginBottom': '20px'}),

            html.Button('Generate Graphics', id='generate-button1', n_clicks=0, style={'width': '100%', 'padding': '15px', 'fontSize': '18px'}),

            # Boutons pour défiler les images
            html.H2("Images", style={'textAlign': 'center', 'padding': '10px'}),
            html.Div([
                html.Button('Previous', id='prev-button1', n_clicks=0, style={'width': '33%', 'padding': '15px', 'fontSize': '18px'}),
                html.Button('Next', id='next-button1', n_clicks=0, style={'width': '33%', 'padding': '15px', 'fontSize': '18px'}),
                html.Button('Play', id='play-button', n_clicks=0, style={'width': '33%', 'padding': '15px', 'fontSize': '18px'}),
            ], style={'display': 'flex', 'justifyContent': 'space-between'}),
            dcc.Interval(id='interval', interval=1000, n_intervals=0, disabled=True),  # Intervalle pour la lecture automatique

            # Boutons pour gérer les métriques
            html.H2("Metrics", style={'textAlign': 'center', 'padding': '10px'}),
            html.Button('Generate Metrics', id='generate-metrics-button', n_clicks=0, style={'width': '100%', 'padding': '15px', 'fontSize': '18px','marginBottom': '40px'}),
            html.Div([
                html.Button('Previous', id='prev-button2', n_clicks=0, style={'width': '50%', 'padding': '15px', 'fontSize': '18px'}),
                html.Button('Next', id='next-button2', n_clicks=0, style={'width': '50%', 'padding': '15px', 'fontSize': '18px'}),
            ], style={'display': 'flex', 'justifyContent': 'space-between'}),
            
        ], style={"border": "1px solid black", "width": "20%", "height": "100vh", "display": "inline-block", "verticalAlign": "top", "boxSizing": "border-box", "padding": "10px", "overflowY": "auto"}),
        
        html.Div([
            dcc.Store(id='stored-images1'),
            html.Img(id='graph1-plot', style={'width': '100%', 'height': 'auto'}),
        ], style={"border": "1px solid black", "width": "40%", "height": "70vh", "display": "inline-block", "verticalAlign": "top", "boxSizing": "border-box"}),
        
        html.Div([
            dcc.Store(id='stored-images2'),
            html.Img(id='graph2-plot', style={'width': '100%', 'height': 'auto'}),
        ], style={"border": "1px solid black", "width": "40%", "height": "70vh", "display": "inline-block", "verticalAlign": "top", "boxSizing": "border-box"}),
    ], style={"display": "flex", "flexDirection": "row", "height": "70vh", "boxSizing": "border-box"}),

    # Section pour le graphique 3
    html.Div([
        dcc.Store(id='stored-images3'), 
        html.Img(id='graph3-plot', style={
            'width': '100%',  
            'height': 'auto',  
            'objectFit': 'contain',
            'margin': 'auto',  
            'display': 'block'  
        }),
    ], style={
        "width": "80%", 
        "height": "30vh", 
        "display": "flex", 
        "justifyContent": "center",  
        "alignItems": "center",  
        "border": "1px solid black", 
        "marginLeft": "auto", 
        "boxSizing": "border-box"
    })
], style={"height": "100vh", "display": "flex", "flexDirection": "column", "boxSizing": "border-box"})
