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
from database import init_db, add_comment, get_comments

# Initialize the database
init_db()

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "CSV Data Visualizer"

# Custom Unix/Linux terminal style
# Custom Unix/Linux terminal style with transparency
terminal_style = {
    'backgroundColor': '#000000',
    'color': '#ffffff',
    'fontFamily': 'Courier New, monospace',
    'padding': '20px',
    'minHeight': '100vh'  # Ensure the app takes full height
}

# Define the layout of the app
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(html.H1("CSV Data Visualizer", style={
            'fontFamily': 'Courier New, monospace',
            'color': '#ffffff',
            'textShadow': '0 0 5px #ffffff'
        }), className="mb-4")
    ]),
    # Introduction
    dbc.Row([
        dbc.Col(html.Div([
            html.H3("Welcome to the CSV Data Visualizer!", style={'color': '#ffffff'}),
            html.P("This application allows you to upload a CSV file, visualize its data, perform basic statistical analysis, and run machine learning models.", style={'color': '#ffffff'}),
            html.P("How to use:", style={'color': '#ffffff'}),
            html.Ul([
                html.Li("Upload a CSV file using the 'Drag and Drop or Select a CSV File' button.", style={'color': '#ffffff'}),
                html.Li("View the data in the table below.", style={'color': '#ffffff'}),
                html.Li("Select a plot type from the dropdown to visualize the data.", style={'color': '#ffffff'}),
                html.Li("Use the column selector to view statistics for a specific column.", style={'color': '#ffffff'}),
                html.Li("Run machine learning models and view evaluation metrics.", style={'color': '#ffffff'}),
                html.Li("Download the edited CSV file using the 'Download Edited CSV' button.", style={'color': '#ffffff'})
            ])
        ]))
    ]),
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
            html.H3("Statistical Summary", style={'color': '#ffffff'}),
            html.Div(id='stat-summary')
        ])
    ]),
    # Column Statistics
    dbc.Row([
        dbc.Col([
            html.H3("Column Statistics", style={'color': '#ffffff'}),
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
            html.H3("Machine Learning Tools", style={'color': '#ffffff'}),
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
            html.H3("Leave a Review", style={'color': '#ffffff'}),
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
        # Footer
    dbc.Row([
        dbc.Col([
            html.Div("Made by Krish Gaur", style={
                'textAlign': 'center',
                'color': '#ffffff',
                'fontFamily': 'Courier New, monospace',
                'marginTop': '20px',
                'marginBottom': '10px',
                'fontSize': '14px',
                'opacity': '0.7'
            })
        ])
    ])
], fluid=True, style=terminal_style)

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
        html.H5(f"File: {filename}", style={'color': '#ffffff'}),
        html.H6("First 5 rows:", style={'color': '#ffffff'}),
        dbc.Table.from_dataframe(df.head(), striped=True, bordered=True, hover=True, style={
            'backgroundColor': '#ffffff',
            'color': '#ffffff'
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
        'backgroundColor': '#ffffff',
        'color': '#ffffff'
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
        'backgroundColor': '#ffffff',
        'color': '#ffffff'
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
            dbc.CardHeader(comment[0], style={'color': '#ffffff'}),
            dbc.CardBody(comment[1], style={'color': '#ffffff'}),
            dbc.CardFooter(comment[2], style={'color': '#ffffff'})
        ], style={'backgroundColor': '#ffffff', 'marginBottom': '10px'})
        for comment in comments
    ]
    return comments_list

# Callback to update ML algorithm options based on the selected task
@app.callback(
    Output('ml-algorithm', 'options'),
    Input('ml-task', 'value')
)
def update_ml_algorithms(selected_task):
    if selected_task == 'classification':
        return [
            {'label': 'Logistic Regression', 'value': 'logistic_regression'},
            {'label': 'Random Forest Classifier', 'value': 'random_forest_classifier'}
        ]
    elif selected_task == 'regression':
        return [
            {'label': 'Linear Regression', 'value': 'linear_regression'},
            {'label': 'Random Forest Regressor', 'value': 'random_forest_regressor'}
        ]
    elif selected_task == 'clustering':
        return [
            {'label': 'K-Means', 'value': 'kmeans'}
        ]
    return []

# Callback to run the selected ML model
@app.callback(
    Output('ml-results', 'children'),
    Input('run-model', 'n_clicks'),
    State('ml-task', 'value'),
    State('ml-algorithm', 'value'),
    State('data-table', 'data'),
    prevent_initial_call=True
)
def run_ml_model(n_clicks, task, algorithm, data):
    if not task or not algorithm or not data:
        return "Please select a task and algorithm."

    df = pd.DataFrame(data)

    if task == 'classification':
        X = df.iloc[:, :-1]  # Features (all columns except the last)
        y = df.iloc[:, -1]   # Target (last column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if algorithm == 'logistic_regression':
            model = LogisticRegression()
        elif algorithm == 'random_forest_classifier':
            model = RandomForestClassifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return f"Accuracy: {accuracy:.2f}"

    elif task == 'regression':
        X = df.iloc[:, :-1]  # Features (all columns except the last)
        y = df.iloc[:, -1]   # Target (last column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if algorithm == 'linear_regression':
            model = LinearRegression()
        elif algorithm == 'random_forest_regressor':
            model = RandomForestRegressor()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return f"Mean Squared Error: {mse:.2f}"

    elif task == 'clustering':
        if algorithm == 'kmeans':
            X = df.iloc[:, :-1]  # Features (all columns except the last)
            model = KMeans(n_clusters=3)
            model.fit(X)
            silhouette = silhouette_score(X, model.labels_)
            return f"Silhouette Score: {silhouette:.2f}"

    return "Invalid task or algorithm."

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 6050))  # Use Render's PORT environment variable
    app.run_server(host='0.0.0.0', port=port, debug=False)