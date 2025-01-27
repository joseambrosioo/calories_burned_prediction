import dash

from calories import calories_layout

# Initialize Dash App
app = dash.Dash(__name__)
server = app.server # Required for Heroku deployment

# Layout
app.layout = calories_layout()

# Run App
if __name__ == '__main__':
    app.run_server(debug=True)
