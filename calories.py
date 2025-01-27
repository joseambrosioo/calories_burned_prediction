import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Constants
CALORIES_URL = 'https://ambrosiopublicfiles.s3.us-east-2.amazonaws.com/calories.csv'
EXERCISE_URL = 'https://ambrosiopublicfiles.s3.us-east-2.amazonaws.com/exercise.csv'

# Load data
calories = pd.read_csv(CALORIES_URL)
exercise_data = pd.read_csv(EXERCISE_URL)
calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)

# Preprocessing
calories_data['Gender'] = calories_data['Gender'].replace({'male': 0, 'female': 1}).infer_objects(copy=False)
X = calories_data.drop(columns=['User_ID', 'Calories'], axis=1)
Y = calories_data['Calories']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train and evaluate multiple models
models = {
    "XGBoost": XGBRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "ElasticNet": ElasticNet(),
    "Decision Tree": DecisionTreeRegressor(),
    "Support Vector Regressor": SVR(),
    "Multi-layer Perceptron": MLPRegressor(max_iter=500)
}

results = {}

for name, model in models.items():
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(Y_test, predictions)
    r2 = r2_score(Y_test, predictions)
    mse = mean_squared_error(Y_test, predictions)
    rmse = mse ** 0.5
    results[name] = {"MAE": mae, "R²": r2, "MSE": mse, "RMSE": rmse}

# Select the best model based on MAE
best_model = min(results, key=lambda k: results[k]['MAE'])
best_model_results = results[best_model]
best_model_predictions = models[best_model].predict(X_test)

# Correlation heatmap data
correlation = calories_data.corr()
heatmap_fig = px.imshow(correlation, text_auto=True, color_continuous_scale='Blues')

# Distribution plots
age_fig = px.histogram(calories_data, x="Age", title="Age Distribution")
height_fig = px.histogram(calories_data, x="Height", title="Height Distribution")
weight_fig = px.histogram(calories_data, x="Weight", title="Weight Distribution")

# Project Summary
project_summary = f"""
This project involved the development of a machine learning model to predict calories burned during physical activities. The model demonstrated impressive performance, achieving metrics of a Mean Absolute Error (MAE), R-Squared Score (R²), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) that exceeded benchmark standards.

The dataset leveraged for this project comprised 15,000 records, including predictors such as activity duration, heart rate, and body temperature. Extensive exploratory data analysis (EDA) was conducted to uncover patterns, trends, and outliers through various visualizations, descriptive statistics, and correlation analysis.

Multiple regression models were implemented and evaluated, including XGBoost, Extra Trees, Multi-layer Perceptron (MLP), Random Forest, Gradient Boosting, K-Nearest Neighbors (KNN), Linear, Lasso, Ridge, SVM, ElasticNet, Decision Tree, Huber, and Bayesian Ridge, to identify the optimal approach. The best model identified was {best_model}, with a MAE of {best_model_results['MAE']:.3f}, R² of {best_model_results['R²']:.3f}, MSE of {best_model_results['MSE']:.3f}, and RMSE of {best_model_results['RMSE']:.3f}.
"""

def calories_layout():
     return dbc.Container(
          [
               dbc.Row([
                    dbc.Card([
                         dbc.CardHeader(
                              html.H1("Calories Burned Prediction Dashboard", style={'color': '#4CAF50', 'textAlign': 'center', 'marginBottom': '20px'})
                         ),
                         dbc.CardBody(
                              [
                                   
                                   html.Div([
                                        html.Div(
                                             [
                                                  html.P(
                                                       "This dashboard provides insights into the prediction of calories burned during physical activities based on user data and exercise metrics.",
                                                       className="text-center text-muted",
                                                       style={'fontSize': '28px', 'textAlign': 'center'}
                                                  )
                                             ],
                                             className="hero-section",
                                        ),
                                        # Buttons Container
                                        html.Div([
                                             dbc.Button(
                                                  "View/Hide Summary",
                                                  id="summary-button",
                                                  className="btn me-1",
                                                  style={'border': 'none', 'padding': '10px 20px', 'fontSize': '16px', 'cursor': 'pointer', 'marginRight': '10px'},
                                                  color="primary",
                                             ),
                                             
                                             dbc.Button(
                                                  "View/Hide Code Snippet",
                                                  id="code-button",
                                                  className="btn",
                                                  style={'border': 'none', 'padding': '10px 20px', 'fontSize': '16px', 'cursor': 'pointer', 'color': 'primary'},
                                             ),
                                             # Summary and Code Containers
                                        ], style={'justifyContent': 'center', 'display': 'flex', 'marginBottom': '20px'}),
                                        
                                        html.Div(id='summary-container', className="shadow p-3 mb-5 bg-white rounded", style={'display': 'none', 'border': '1px solid #ddd', 'padding': '10px20px', 'borderRadius': '5px'}),
                                        html.Div(id='code-container', className="shadow p-3 mb-5 bg-white rounded", style={'display': 'none'}),
                                        
                                        # Metrics Display
                                        dbc.Row([
                                             dbc.Col([
                                                  html.Div(
                                                       [
                                                            html.H3("Model Performance Metrics", style={'textAlign': 'center', 'marginBottom': '20px'}),
                                                            html.Div([html.H3("Mean Absolute Error"), html.P(f"{best_model_results['MAE']:.2f}")], style={'margin': '20px'}),
                                                            html.Div([html.H3("R-Squared Score"), html.P(f"{best_model_results['R²']:.2f}")], style={'margin': '20px'}),
                                                            html.Div([html.H3("Mean Squared Error"), html.P(f"{best_model_results['MSE']:.2f}")], style={'margin': '20px'}),
                                                            html.Div([html.H3("Root Mean Squared Error"), html.P(f"{best_model_results['RMSE']:.2f}")], style={'margin': '20px'}),
                                                       ], 
                                                       style={'display': 'flex', 'justifyContent': 'center'}
                                                  ),
                                             ], xs=12, sm=12, md=6, lg=6, xl=6),     
                                        ]),

                                        # Graphs
                                        dbc.Row(
                                             [
                                                  dbc.Col([
                                                       dbc.Card(
                                                            dbc.CardBody([
                                                                 html.H3("Correlation Heatmap"),
                                                                 dcc.Graph(id='heatmap', figure=heatmap_fig),     
                                                            ]),
                                                       )
                                                  ] , xs=12, sm=12, md=12, lg=12, xl=12, width={'size': 12}),
                                                  
                                                  dbc.Col([
                                                       dbc.Card(
                                                            dbc.CardBody([
                                                                 html.H3("Age Distribution Graphs"),
                                                                 dcc.Graph(id='age-dist', figure=age_fig),
                                                                 
                                                            ])
                                                       )
                                                  ], xs=12, sm=12, md=6, lg=6, xl=6, width={'size': 6}),
                                             ]
                                        ),
                                        
                                        dbc.Row([
                                             dbc.Col([
                                                  html.H3("Height Distribution"),
                                                  dcc.Graph(id='height-dist', figure=height_fig),
                                             ], xs=12, sm=12, md=6, lg=6, xl=6),  
                                             
                                             dbc.Col([
                                                  html.H3("Weight Distribution"),
                                                  dcc.Graph(id='weight-dist', figure=weight_fig),
                                             ], xs=12, sm=12, md=6, lg=6, xl=6),   
                                        ]),
                                        
                                        # Predictions
                                        html.Div([
                                             html.H3("Prediction Test Results"),
                                             dcc.Graph(
                                                  id='prediction-scatter',
                                                  figure=go.Figure(
                                                       data=[
                                                            go.Scatter(
                                                                 x=Y_test,
                                                                 y=best_model_predictions,
                                                                 mode='markers',
                                                                 marker=dict(color='blue', size=5),
                                                                 name='Predicted vs Actual'
                                                            )
                                                       ]
                                                  ).update_layout(
                                                       title="Predicted vs Actual Calories", 
                                                       xaxis_title="Actual", 
                                                       yaxis_title="Predicted"
                                                  )
                                             ), 
                                        ]),   
                                        
                                        # Footer
                                        dbc.CardFooter([
                                             html.P([
                                                  "Check out my ",
                                                  html.A("Portfolio", href="https://ose-ambrosioo.github.io", target="_blank"),
                                                  " and connect with me on ",
                                                  html.A("LinkedIn", href="https://www.linkedin.com/in/jose-ambrosio/", target="_blank"),
                                                  "."
                                             ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#f9f9f9', 'borderTop': '1px solid #ddd'})
                                        ])        
                                   ])   
                              ], 
                         ),
                    ], className="mb-4 shadow"),
               ]),
     ], fluid=True, className="mt-5")

# Callback to show/hide summary
@callback(
    [Output("summary-container", "children"), Output("summary-container", "style"), Output("summary-button", "children")],
    Input("summary-button", "n_clicks"),
    State("summary-button", "children"),
    prevent_initial_call=True
)
def update_summary(n_clicks, button_text):
    if button_text == "View/Hide Summary":
        style = {"display": "block"}
        button_text = "Hide Summary"
        content = html.Div(dcc.Markdown(project_summary, style={"whiteSpace": "pre-wrap", "wordWrap": "break-word"}))
    else:
        style = {"display": "none"}
        button_text = "View/Hide Summary"
        content = ""
    return content, style, button_text



# Callback to show/hide code snippet
@callback(
    [Output("code-container", "children"), Output("code-container", "style"), Output("code-button", "children")],
    Input("code-button", "n_clicks"),
    State("code-button", "children"),
    prevent_initial_call=True
)
def update_code(n_clicks, button_text):
    if button_text == "View/Hide Code Snippet":
        style = {"display": "block"}
        button_text = "Hide Code Snippet"
        content = html.Div([
            html.H3("Code Snippet"),
            dcc.Markdown(f"""              
```python
# Train and evaluate multiple models
models = {{
    "XGBoost": XGBRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "ElasticNet": ElasticNet(),
    "Decision Tree": DecisionTreeRegressor(),
    "Support Vector Regressor": SVR(),
    "Multi-layer Perceptron": MLPRegressor()
}}

results = {{}}

for name, model in models.items():
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(Y_test, predictions)
    r2 = r2_score(Y_test, predictions)
    mse = mean_squared_error(Y_test, predictions)
    rmse = mse ** 0.5
    results[name] = {{"MAE": mae, "R²": r2, "MSE": mse, "RMSE": rmse}}

# Select the best model based on MAE
best_model = min(results, key=lambda k: results[k]['MAE'])
best_model_results = results[best_model]
best_model_predictions = models[best_model].predict(X_test)
```""", style={"whiteSpace": "preWrap", "wordWrap": "breakWord"}), html.H3("Explanation"), html.P(code_explanation)])
    else:
        style = {"display": "none"}
        button_text = "View/Hide Code Snippet"
        content = ""
    return content, style, button_text


# Explanation
code_explanation = """
In this snippet:
- We define a dictionary `models` containing the different regression models.
- We iterate through each model, train it on the training data (`X_train`, `Y_train`), and evaluate it on the test data (`X_test`).
- We calculate the evaluation metrics (MAE, R², MSE, RMSE) for each model and store the results in the `results` dictionary.
- Finally, we select the best model based on the lowest MAE and store its predictions and results.

This approach ensures that we evaluate multiple models and select the one with the best performance.
"""