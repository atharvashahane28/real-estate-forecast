import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ml_models import execute
from models.neural_network import execute_nn

# print("Hello world")

data = pd.read_csv("datasets\california_housing\housing.csv")
app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children="California Housing Prices"),
    # html.Div(children="Dash"),
    # dcc.Graph(
    #     id="example-graph",
    #     figure=px.scatter(data, x="median_income", y="median_house_value")
    # )
    dcc.Dropdown(
        id="model_dropdown",
        options=[
            {"label": "Linear Regression", "value": "LR"},
            {"label": "Support Vector Regression", "value": "SVR"},
            {"label": "Decision Tree Regressor", "value": "DTR"},
            {"label": "Random Forest Regressor", "value": "RFR"},
            {"label": "Gradient Boosting Regressor", "value": "GBR"},
            {"label": "Simple Neural Network", "value": "SNN"},
            {"label": "Advanced Neural Network", "value": "ANN"},
        ],
        value="LR"
    ),
    dcc.Graph(
        id="prediction_graph"
    ),
    dcc.Graph(
        id="error_graph"    
    )
])

@app.callback(
    [Output("prediction_graph", "figure"), Output("error_graph", "figure")],
    [Input("model_dropdown", "value")]
)
def update_graphs(model):
    # y_test = data["median_house_value"]
    # y_pred = y_test * 0.9
    if model == "SNN" or model == "ANN":
        predictions, loss, mae, rmse, y_test = execute_nn(model)
    else:
        predictions, mse, mae, mape, y_test = execute(model)
    
    prediction_figure = {
        "data": [
            go.Scatter(x=list(range(len(y_test))), y=y_test, mode="lines", name="Actual"),
            go.Scatter(x=list(range(len(y_test))), y=predictions, mode="lines", name="Predicted")
        ],
        "layout": go.Layout(title="Actual vs Predicted", xaxis={"title": "Index"}, yaxis={"title": "Median House Value"})
    }
    error_figure = {
        "data": [
            go.Bar(x=list(range(len(y_test))), y=abs(y_test - predictions), name="Error")
        ],
        "layout": go.Layout(title="Error", xaxis={"title": "Index"}, yaxis={"title": "Error"})
    }
    return prediction_figure, error_figure

if __name__ == "__main__":
    app.run_server(debug=True)
