import os
import base64
import io
import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
import nltk
nltk.download('punkt')
from textblob import TextBlob
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from textblob import TextBlob
from database import init_db, add_comment, get_comments, edit_comment

# Initialize the database
init_db()

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "CSV Data Visualizer"

# Define styles for themes
theme_styles = {
    'standard': {
        'backgroundColor': '#000000',
        'color': '#ffffff',
        'fontFamily': 'Courier New, monospace',
        'padding': '20px',
        'minHeight': '100vh'  # Ensure the app takes full height
    },
    'naked_snake': {
        'backgroundColor': '#0a0a0a',
        'backgroundImage': 'url("/assets/naked-snake.jpg")',
        'backgroundSize': 'cover',
        'color': '#00ff00',  # Greenish color
        'fontFamily': 'Courier New, monospace',
        'padding': '20px',
        'minHeight': '100vh',
        'backgroundRepeat': 'no-repeat',
        'backgroundAttachment': 'fixed',
    },
    'solid_snake': {
        'backgroundColor': '#1a1a1a',
        'backgroundImage': 'url("/assets/solid-snake.jpg")',
        'backgroundSize': 'cover',
        'color': '#c0c0c0',  # Grayish color
        'fontFamily': 'Courier New, monospace',
        'padding': '20px',
        'minHeight': '100vh',
        'backgroundRepeat': 'no-repeat',
        'backgroundAttachment': 'fixed',
    }
}

# Define the layout of the app
app.layout = html.Div([
    dbc.Container([
        # Header
        dbc.Row([
            dbc.Col(html.H1("CSV Data Visualizer", id='app-title', style={
                'fontFamily': 'Courier New, monospace',
                'color': 'inherit',  # Use inherit to get color from parent
                'textShadow': '0 0 5px #ffffff'
            }), className="mb-4")
        ], id='header'),

        # Introduction
        dbc.Row([
            dbc.Col(html.Div([
                html.H3("Welcome to the CSV Data Visualizer!", style={'color': 'inherit'}),
                html.P("This application allows you to upload a CSV file, visualize its data, perform basic statistical analysis, and run machine learning models.", style={'color': 'inherit'}),
                html.P("How to use:", style={'color': 'inherit'}),
                html.Ul([
                    html.Li("Upload a CSV file using the 'Drag and Drop or Select a CSV File' button.", style={'color': 'inherit'}),
                    html.Li("View the data in the table below.", style={'color': 'inherit'}),
                    html.Li("Select a plot type from the dropdown to visualize the data.", style={'color': 'inherit'}),
                    html.Li("Use the column selector to view statistics for a specific column.", style={'color': 'inherit'}),
                    html.Li("Run machine learning models and view evaluation metrics.", style={'color': 'inherit'}),
                    html.Li("Download the edited CSV file using the 'Download Edited CSV' button.", style={'color': 'inherit'})
                ])
            ]))
        ]),

        # Theme Toggle
        dbc.Row([
            dbc.Col([
                html.Label("Select Theme:", style={'color': 'inherit'}),
                dcc.RadioItems(
                    id='theme-toggle',
                    options=[
                        {'label': 'Standard Mode', 'value': 'standard'},
                        {'label': 'Naked Snake Mode', 'value': 'naked_snake'},
                        {'label': 'Solid Snake Mode', 'value': 'solid_snake'}
                    ],
                    value='standard',
                    labelStyle={'display': 'inline-block', 'margin-right': '10px', 'color': 'inherit'},
                    inputStyle={"margin-right": "5px"}
                )
            ], width="auto")
        ], style={'margin-bottom': '20px'}),

        # Reference Text (MGS references)
        html.Div(id='reference-text'),

        # File Upload
        dbc.Row([
            dbc.Col([
                dcc.Upload(
                    id='upload-data',
                    children=html.Div(['Drag and Drop or ', html.A('Select a CSV File')], style={
                        'color': '#000000',
                        'fontFamily': 'Courier New, monospace'
                    }),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px',
                        'backgroundColor': 'rgba(255, 255, 255, 0.8)',  # Transparent white
                        'color': '#000000'
                    },
                    multiple=False
                ),
                html.Div(id='output-data-upload'),
            ])
        ]),
        # Data Table
        dbc.Row([
            dbc.Col([
                dash_table.DataTable(
                    id='data-table',
                    columns=[],
                    data=[],
                    editable=True,
                    row_deletable=True,
                    page_action="native",
                    page_current=0,
                    page_size=10,
                    style_table={'backgroundColor': 'rgba(255, 255, 255, 0.8)', 'color': '#000000'},
                    style_header={'backgroundColor': 'rgba(0, 66, 0, 0.8)', 'color': '#ffffff'},
                    style_cell={'backgroundColor': 'rgba(255, 255, 255, 0.8)', 'color': '#000000'}
                )
            ], style={'overflowX': 'auto'})  # Allow horizontal scrolling on small screens
        ]),
        # Plot Type Selector
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='plot-type',
                    options=[
                        {'label': 'Scatter Plot', 'value': 'scatter'},
                        {'label': 'Line Plot', 'value': 'line'},
                        {'label': 'Bar Chart', 'value': 'bar'},
                        {'label': 'Histogram', 'value': 'histogram'},
                        {'label': 'Box Plot', 'value': 'box'},
                        {'label': 'Violin Plot', 'value': 'violin'},
                        {'label': 'Density Plot', 'value': 'density'},
                        {'label': 'Correlation Heatmap', 'value': 'heatmap'},
                        {'label': 'Pair Plot', 'value': 'pairplot'},
                        {'label': '3D Scatter Plot', 'value': '3d_scatter'}
                    ],
                    value='scatter',
                    style={'backgroundColor': 'rgba(255, 255, 255, 0.8)', 'color': '#000000'}
                )
            ])
        ]),
        # Graph
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='data-graph')
            ])
        ]),
        # Download Button
        dbc.Row([
            dbc.Col([
                html.Button("Download Edited CSV", id="btn-download", style={
                    'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                    'color': '#000000',
                    'border': 'none',
                    'padding': '10px 20px',
                    'fontFamily': 'Courier New, monospace',
                    'textShadow': '0 0 5px #ffffff'
                }),
                dcc.Download(id="download-data")
            ])
        ]),
        # Statistical Summary
        dbc.Row([
            dbc.Col([
                html.H3("Statistical Summary", style={'color': 'inherit'}),
                html.Div(id='stat-summary')
            ])
        ]),
        # Column Statistics
        dbc.Row([
            dbc.Col([
                html.H3("Column Statistics", style={'color': 'inherit'}),
                dcc.Dropdown(
                    id='column-selector',
                    options=[],
                    placeholder="Select a column",
                    style={'backgroundColor': 'rgba(255, 255, 255, 0.8)', 'color': '#000000'}
                ),
                html.Div(id='column-stats')
            ])
        ]),

        # Machine Learning Section
        dbc.Row([
            dbc.Col([
                html.H3("Machine Learning Tools", style={'color': 'inherit'}),
                dcc.Dropdown(
                    id='ml-task',
                    options=[
                        {'label': 'Classification', 'value': 'classification'},
                        {'label': 'Regression', 'value': 'regression'},
                        {'label': 'Clustering', 'value': 'clustering'}
                    ],
                    placeholder="Select an ML task",
                    style={'backgroundColor': 'rgba(255, 255, 255, 0.8)', 'color': '#000000'}
                ),
                dcc.Dropdown(
                    id='ml-algorithm',
                    options=[],
                    placeholder="Select an algorithm",
                    style={'backgroundColor': 'rgba(255, 255, 255, 0.8)', 'color': '#000000'}
                ),
                html.Button("Run Model", id='run-model', style={
                    'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                    'color': '#000000',
                    'border': 'none',
                    'padding': '10px 20px',
                    'fontFamily': 'Courier New, monospace',
                    'textShadow': '0 0 5px #ffffff'
                }),
                html.Div(id='ml-results')
            ])
        ]),
        # Comments Section
        dbc.Row([
            dbc.Col([
                html.H3("Leave a Review", style={'color': 'inherit'}),
                dbc.Input(id='comment-name', placeholder="Your Name", style={'backgroundColor': 'rgba(255, 255, 255, 0.8)', 'color': '#000000'}),
                dbc.Textarea(id='comment-text', placeholder="Your Comment", style={'backgroundColor': 'rgba(255, 255, 255, 0.8)', 'color': '#000000'}),
                html.Button("Submit", id='submit-comment', style={
                    'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                    'color': '#000000',
                    'border': 'none',
                    'padding': '10px 20px',
                    'fontFamily': 'Courier New, monospace',
                    'textShadow': '0 0 5px #ffffff'
                }),
                html.Div(id='comments-display')
            ])
        ]),

        # Edit Comment Section
        dbc.Row([
            dbc.Col([
                html.H3("Edit Comment", style={'color': 'inherit'}),
                dbc.Input(id='edit-comment-id', placeholder="Comment ID", style={'backgroundColor': 'rgba(255, 255, 255, 0.8)', 'color': '#000000'}),
                dbc.Textarea(id='edit-comment-text', placeholder="New Comment", style={'backgroundColor': 'rgba(255, 255, 255, 0.8)', 'color': '#000000'}),
                html.Button("Edit", id='edit-comment-button', style={
                    'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                    'color': '#000000',
                    'border': 'none',
                    'padding': '10px 20px',
                    'fontFamily': 'Courier New, monospace',
                    'textShadow': '0 0 5px #ffffff'
                }),
                html.Div(id='edit-comment-status')
            ])
        ]),
        # Footer
        dbc.Row([
            dbc.Col([
                html.Div("Made by Krish Gaur❤️", id='footer-text', style={
                    'textAlign': 'center',
                    'color': 'inherit',
                    'fontFamily': 'Courier New, monospace',
                    'marginTop': '20px',
                    'marginBottom': '10px',
                    'fontSize': '14px',
                    'opacity': '0.7'
                })
            ])
        ], id='footer')
    ], fluid=True, id='container')
], id='main-div', style=theme_styles['standard'])

# Callback to update theme
@app.callback(
    Output('main-div', 'style'),
    Output('app-title', 'style'),
    Output('reference-text', 'children'),
    Input('theme-toggle', 'value')
)
def update_theme(theme_value):
    # Get the styles for the selected theme
    selected_style = theme_styles.get(theme_value, theme_styles['standard'])
    
    # Update app title style
    app_title_style = {
        'fontFamily': 'Courier New, monospace',
        'color': selected_style.get('color', '#ffffff'),
        'textShadow': '0 0 5px #ffffff'
    }
    
    # Update the reference text based on theme
    if theme_value == 'naked_snake':
        reference_text = html.Div([
            html.P("The mission comes first. This CSV data won't analyze itself.", style={'color': selected_style.get('color', '#ffffff')}),
        ])
    elif theme_value == 'solid_snake':
        reference_text = html.Div([
            html.P("Kept you waiting, huh? Time to get started with data analysis.", style={'color': selected_style.get('color', '#ffffff')}),
        ])
    else:
        reference_text = None

    return selected_style, app_title_style, reference_text

# Include the rest of your callbacks here (update_output, download_data, etc.)

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 6050))  # Use Render's PORT environment variable
    app.run_server(host='0.0.0.0', port=port, debug=False)