# main.py
import os
import base64
import io
import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from scipy import stats
from database import init_db, add_comment, get_comments

# Initialize the database
init_db()

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "CSV Data Visualizer - Unix/Linux Edition"

# Custom Unix/Linux terminal style
terminal_style = {
    'backgroundColor': '#000000',
    'color': '#00ff00',
    'fontFamily': 'Courier New, monospace',
    'padding': '20px'
}

# Define the layout of the app
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(html.H1("CSV Data Visualizer - Unix/Linux Edition", style={
            'fontFamily': 'Courier New, monospace',
            'color': '#00ff00',
            'textShadow': '0 0 5px #00ff00'
        }), className="mb-4")
    ]),
    # Introduction
    dbc.Row([
        dbc.Col(html.Div([
            html.H3("Welcome to the CSV Data Visualizer!", style={'color': '#00ff00'}),
            html.P("This application allows you to upload a CSV file, visualize its data, and perform basic statistical analysis.", style={'color': '#00ff00'}),
            html.P("How to use:", style={'color': '#00ff00'}),
            html.Ul([
                html.Li("Upload a CSV file using the 'Drag and Drop or Select a CSV File' button.", style={'color': '#00ff00'}),
                html.Li("View the data in the table below.", style={'color': '#00ff00'}),
                html.Li("Select a plot type from the dropdown to visualize the data.", style={'color': '#00ff00'}),
                html.Li("Use the column selector to view statistics for a specific column.", style={'color': '#00ff00'}),
                html.Li("Download the edited CSV file using the 'Download Edited CSV' button.", style={'color': '#00ff00'})
            ])
        ]))
    ]),
    # File Upload
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Drag and Drop or ', html.A('Select a CSV File')], style={
                    'color': '#00ff00',
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
                    'backgroundColor': '#002200',
                    'color': '#00ff00'
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
                style_table={'backgroundColor': '#002200', 'color': '#00ff00'},
                style_header={'backgroundColor': '#004400', 'color': '#00ff00'},
                style_cell={'backgroundColor': '#002200', 'color': '#00ff00'}
            )
        ])
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
                style={'backgroundColor': '#002200', 'color': '#00ff00'}
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
                'backgroundColor': '#004400',
                'color': '#00ff00',
                'border': 'none',
                'padding': '10px 20px',
                'fontFamily': 'Courier New, monospace',
                'textShadow': '0 0 5px #00ff00'
            }),
            dcc.Download(id="download-data")
        ])
    ]),
    # Statistical Summary
    dbc.Row([
        dbc.Col([
            html.H3("Statistical Summary", style={'color': '#00ff00'}),
            html.Div(id='stat-summary')
        ])
    ]),
    # Column Statistics
    dbc.Row([
        dbc.Col([
            html.H3("Column Statistics", style={'color': '#00ff00'}),
            dcc.Dropdown(
                id='column-selector',
                options=[],
                placeholder="Select a column",
                style={'backgroundColor': '#002200', 'color': '#00ff00'}
            ),
            html.Div(id='column-stats')
        ])
    ]),
    # Comments Section
    dbc.Row([
        dbc.Col([
            html.H3("Leave a Review", style={'color': '#00ff00'}),
            dbc.Input(id='comment-name', placeholder="Your Name", style={'backgroundColor': '#002200', 'color': '#00ff00'}),
            dbc.Textarea(id='comment-text', placeholder="Your Comment", style={'backgroundColor': '#002200', 'color': '#00ff00'}),
            html.Button("Submit", id='submit-comment', style={
                'backgroundColor': '#004400',
                'color': '#00ff00',
                'border': 'none',
                'padding': '10px 20px',
                'fontFamily': 'Courier New, monospace',
                'textShadow': '0 0 5px #00ff00'
            }),
            html.Div(id='comments-display')
        ])
    ])
], style=terminal_style)

# Callback to handle file upload, display data, and update table/graph
@app.callback(
    Output('output-data-upload', 'children'),
    Output('data-graph', 'figure'),
    Output('data-table', 'columns'),
    Output('data-table', 'data'),
    Output('column-selector', 'options'),
    Input('upload-data', 'contents'),
    Input('plot-type', 'value'),
    State('upload-data', 'filename')
)
def update_output(contents, plot_type, filename):
    if contents is None:
        return "No file uploaded yet.", {}, [], [], []

    # Parse the CSV file
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        return f"Error parsing CSV file: {e}", {}, [], [], []

    # Display the first few rows of the data
    data_preview = html.Div([
        html.H5(f"File: {filename}", style={'color': '#00ff00'}),
        html.H6("First 5 rows:", style={'color': '#00ff00'}),
        dbc.Table.from_dataframe(df.head(), striped=True, bordered=True, hover=True, style={
            'backgroundColor': '#002200',
            'color': '#00ff00'
        })
    ])

    # Create the selected plot
    if plot_type == 'scatter':
        if len(df.columns) >= 2:
            fig = px.scatter(df, x=df.columns[0], y=df.columns[1], title=f"{filename} - Scatter Plot")
        else:
            fig = {}
    elif plot_type == 'line':
        if len(df.columns) >= 2:
            fig = px.line(df, x=df.columns[0], y=df.columns[1], title=f"{filename} - Line Plot")
        else:
            fig = {}
    elif plot_type == 'bar':
        if len(df.columns) >= 2:
            fig = px.bar(df, x=df.columns[0], y=df.columns[1], title=f"{filename} - Bar Chart")
        else:
            fig = {}
    elif plot_type == 'histogram':
        if len(df.columns) >= 1:
            fig = px.histogram(df, x=df.columns[0], title=f"{filename} - Histogram")
        else:
            fig = {}
    elif plot_type == 'box':
        if len(df.columns) >= 1:
            fig = px.box(df, y=df.columns[0], title=f"{filename} - Box Plot")
        else:
            fig = {}
    elif plot_type == 'violin':
        if len(df.columns) >= 1:
            fig = px.violin(df, y=df.columns[0], title=f"{filename} - Violin Plot")
        else:
            fig = {}
    elif plot_type == 'density':
        if len(df.columns) >= 1:
            fig = px.density_contour(df, x=df.columns[0], title=f"{filename} - Density Plot")
        else:
            fig = {}
    elif plot_type == 'heatmap':
        if len(df.columns) >= 2:
            fig = px.imshow(df.corr(), title=f"{filename} - Correlation Heatmap")
        else:
            fig = {}
    elif plot_type == 'pairplot':
        if len(df.columns) >= 2:
            fig = px.scatter_matrix(df, title=f"{filename} - Pair Plot")
        else:
            fig = {}
    elif plot_type == '3d_scatter':
        if len(df.columns) >= 3:
            fig = px.scatter_3d(df, x=df.columns[0], y=df.columns[1], z=df.columns[2], title=f"{filename} - 3D Scatter Plot")
        else:
            fig = {}
    else:
        fig = {}

    # Prepare data for the table
    columns = [{"name": i, "id": i} for i in df.columns]
    data = df.to_dict('records')

    # Update column selector options
    column_options = [{'label': col, 'value': col} for col in df.columns]

    return data_preview, fig, columns, data, column_options

# Callback to handle downloading the edited data
@app.callback(
    Output("download-data", "data"),
    Input("btn-download", "n_clicks"),
    State("data-table", "data"),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def download_data(n_clicks, data, filename):
    df = pd.DataFrame(data)
    return dcc.send_data_frame(df.to_csv, filename or "edited_data.csv")

# Callback to display statistical summary
@app.callback(
    Output('stat-summary', 'children'),
    Input('data-table', 'data')
)
def display_stat_summary(data):
    if not data:
        return "No data available."
    df = pd.DataFrame(data)
    summary = df.describe().reset_index()
    return dbc.Table.from_dataframe(summary, striped=True, bordered=True, hover=True, style={
        'backgroundColor': '#002200',
        'color': '#00ff00'
    })

# Callback to display column statistics
@app.callback(
    Output('column-stats', 'children'),
    Input('column-selector', 'value'),
    State('data-table', 'data')
)
def display_column_stats(selected_column, data):
    if not selected_column or not data:
        return "No column selected."
    df = pd.DataFrame(data)
    if selected_column not in df.columns:
        return "Invalid column selected."
    column_data = df[selected_column]
    stats = {
        'Mean': column_data.mean(),
        'Median': column_data.median(),
        'Mode': stats.mode(column_data)[0][0],
        'Standard Deviation': column_data.std(),
        'Min': column_data.min(),
        'Max': column_data.max()
    }
    stats_df = pd.DataFrame(list(stats.items()), columns=['Statistic', 'Value'])
    return dbc.Table.from_dataframe(stats_df, striped=True, bordered=True, hover=True, style={
        'backgroundColor': '#002200',
        'color': '#00ff00'
    })

# Callback to handle comments
@app.callback(
    Output('comments-display', 'children'),
    Input('submit-comment', 'n_clicks'),
    State('comment-name', 'value'),
    State('comment-text', 'value'),
    prevent_initial_call=True
)
def add_and_display_comments(n_clicks, name, comment):
    if not name or not comment:
        return "Please enter both your name and comment."
    add_comment(name, comment)
    comments = get_comments()
    comments_list = [
        dbc.Card([
            dbc.CardHeader(comment[0], style={'color': '#00ff00'}),
            dbc.CardBody(comment[1], style={'color': '#00ff00'}),
            dbc.CardFooter(comment[2], style={'color': '#00ff00'})
        ], style={'backgroundColor': '#002200', 'marginBottom': '10px'})
        for comment in comments
    ]
    return comments_list

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))  # Use Render's PORT environment variable
    app.run_server(host='0.0.0.0', port=port, debug=False)