import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import base64
from io import BytesIO

df = pd.read_csv("C:\Users\start\Downloads\preprocessed_tourism_reviews.csv")
df['date'] = pd.to_datetime(df['date'])

# generate word cloud image for a given sentiment
def generate_wordcloud(sentiment):
    text = ' '.join(df.loc[df['sentiment'] == sentiment, 'processed_review'].dropna())
    wc = WordCloud(
        width=800,
        height=400,
        background_color='rgba(0,0,0,0)', 
        colormap='plasma',
        max_words=120
    ).generate(text)
    buf = BytesIO()
    wc.to_image().save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

# Compute mean word count per sentiment
def compute_mean_words(filtered_df):
    return filtered_df.groupby('sentiment')['word_count'].mean().reset_index(name='mean_word_count')

# Initialize Dash app with dark theme
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "TripAdvisor Sentiment Dashboard"

app.layout = html.Div(style={'backgroundColor': '#111', 'color': '#eee', 'font-family': 'Arial, sans-serif'}, children=[
    html.Div(style={
        'background': 'linear-gradient(135deg, #1f1c2c 0%, #928dab 100%)',
        'padding': '40px',
        'textAlign': 'center',
        'border-radius': '0 0 50px 50px',
        'box-shadow': '0 4px 20px rgba(0,0,0,0.5)'
    }, children=[
        html.H1("🔮 Common Words by Sentiment", style={'font-size': '3rem', 'margin-bottom': '10px'}),
        dcc.Dropdown(
            id='wordcloud-dropdown',
            options=[{'label': s, 'value': s} for s in df['sentiment'].unique()],
            value=df['sentiment'].unique()[0],
            clearable=False,
            style={'width': '300px', 'margin': 'auto', 'color': '#000', 'border-radius': '20px'}
        ),
        html.Br(),
        html.Img(id='wordcloud-img', style={'width': '90%', 'max-width': '800px', 'border-radius': '20px', 'margin-top': '20px', 'box-shadow': '0 4px 15px rgba(0,0,0,0.6)'})
    ]),

    # Main Title
    html.H1("🧳 TripAdvisor Sentiment Dashboard", style={'textAlign': 'center', 'marginTop': '40px'}),

    # Filters Section
    html.Div(style={'display': 'flex', 'justify-content': 'center', 'gap': '20px', 'padding': '20px'}, children=[
        html.Div(children=[
            html.Label("Select Attraction", style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='attraction-dropdown',
                options=[{'label': a, 'value': a} for a in df['attraction'].unique()],
                multi=True,
                placeholder="All Attractions",
                style={'width': '250px'}
            )
        ]),
        html.Div(children=[
            html.Label("Select Sentiment", style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='sentiment-dropdown',
                options=[{'label': s, 'value': s} for s in df['sentiment'].unique()],
                multi=True,
                placeholder="All Sentiments",
                style={'width': '250px'}
            )
        ])
    ]),

    # Graphs Grid
    html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(400px, 1fr))', 'gap': '30px', 'padding': '20px'}, children=[
        dcc.Graph(id='sentiment-distribution', config={'displayModeBar': False}, style={'backgroundColor': '#222'}),
        dcc.Graph(id='rating-by-sentiment', config={'displayModeBar': False}, style={'backgroundColor': '#222'}),
        dcc.Graph(id='reviews-over-time', config={'displayModeBar': False}, style={'backgroundColor': '#222'}),
        dcc.Graph(id='avg-rating-attraction', config={'displayModeBar': False}, style={'backgroundColor': '#222'}),
        dcc.Graph(id='sentiment-attraction-stacked', config={'displayModeBar': False}, style={'backgroundColor': '#222'}),
        dcc.Graph(id='mean-word-count', config={'displayModeBar': False}, style={'backgroundColor': '#222'})
    ])
])

# Callbacks
@app.callback(
    Output('wordcloud-img', 'src'),
    Input('wordcloud-dropdown', 'value')
)
def update_wordcloud(selected_sentiment):
    return generate_wordcloud(selected_sentiment)

@app.callback(
    [Output('sentiment-distribution', 'figure'),
     Output('rating-by-sentiment', 'figure'),
     Output('reviews-over-time', 'figure'),
     Output('avg-rating-attraction', 'figure'),
     Output('sentiment-attraction-stacked', 'figure'),
     Output('mean-word-count', 'figure')],
    [Input('attraction-dropdown', 'value'), Input('sentiment-dropdown', 'value')]
)
def update_graphs(selected_attractions, selected_sentiments):
    flt = df.copy()
    if selected_attractions:
        flt = flt[flt['attraction'].isin(selected_attractions)]
    if selected_sentiments:
        flt = flt[flt['sentiment'].isin(selected_sentiments)]

    fig1 = px.histogram(flt, x='sentiment', color='sentiment', title='Sentiment Distribution', template='plotly_dark')
    fig2 = px.box(flt, x='sentiment', y='rate', color='sentiment', title='Rating by Sentiment', template='plotly_dark')

    tmp = flt.groupby(['date', 'sentiment']).size().reset_index(name='count')
    fig3 = px.line(tmp, x='date', y='count', color='sentiment', title='Reviews Over Time', template='plotly_dark')

    avg_rt = flt.groupby('attraction')['rate'].mean().reset_index()
    fig4 = px.bar(avg_rt, x='attraction', y='rate', title='Average Rating per Attraction', template='plotly_dark')

    sc = flt.groupby(['attraction', 'sentiment']).size().reset_index(name='count')
    fig5 = px.bar(sc, x='attraction', y='count', color='sentiment', barmode='stack', title='Sentiment per Attraction', template='plotly_dark')

    mw = compute_mean_words(flt)
    fig6 = px.bar(mw, x='sentiment', y='mean_word_count', color='sentiment', title='Mean Word Count per Sentiment', labels={'mean_word_count':'Avg Words'}, template='plotly_dark')

    # Style adjustments
    for fig in [fig1, fig2, fig3, fig4, fig5, fig6]:
        fig.update_layout(paper_bgcolor='#222', plot_bgcolor='#222', font_color='#eee')

    return fig1, fig2, fig3, fig4, fig5, fig6

# Run server
if __name__ == '__main__':
    app.run(debug=True, port=8050)