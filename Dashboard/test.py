import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import base64
from io import BytesIO
import numpy as np
from datetime import datetime

# Load and preprocess data
df = pd.read_csv(r"C:\Users\awali\Downloads\labeled_tourism_reviews_For_EDA.csv")

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Convert day_of_week numbers to names
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['day_name'] = df['day_of_week'].map(dict(enumerate(day_names)))

# Convert month numbers to names
month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
df['month_name'] = df['month'].map(dict(enumerate(month_names, 1)))

# Color scheme for light theme
colors = {
    'positive': '#2ecc71',  # Green
    'non_positive': '#e74c3c',  # Red
    'background': '#ffffff',
    'text': '#2c3e50',
    'accent': '#3498db',
    'grid': '#ecf0f1',
    'card': '#f8f9fa'
}

# Word cloud generator
def generate_wordcloud(sentiment):
    text = ' '.join(df[df['sentiment_ensemble'] == sentiment]['review'].dropna())
    if not text:
        return None
    wc = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=100,
        min_font_size=10,
        max_font_size=100
    )
    img = wc.generate(text)
    buf = BytesIO()
    img.to_image().save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Tourism Reviews Analysis Dashboard"

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🏛️ Tourism Reviews Analysis Dashboard", 
                style={'color': colors['text'], 'textAlign': 'center', 'padding': '20px', 'margin': '0'}),
        html.P("Analyzing customer reviews and sentiment patterns across tourist attractions",
               style={'color': colors['text'], 'textAlign': 'center', 'fontSize': '1.2rem', 'marginBottom': '20px'})
    ], style={'backgroundColor': colors['card'], 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),

    # Filters Section
    html.Div([
        html.Div([
            html.Label("Select Attractions", style={'color': colors['text'], 'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='attraction-dropdown',
                options=[{'label': x, 'value': x} for x in sorted(df['attraction'].unique())],
                multi=True,
                placeholder="Select attractions...",
                style={'backgroundColor': colors['background']}
            )
        ], style={'width': '45%', 'display': 'inline-block', 'margin': '10px'}),
        
        html.Div([
            html.Label("Date Range", style={'color': colors['text'], 'fontWeight': 'bold'}),
            dcc.DatePickerRange(
                id='date-range',
                start_date=df['date'].min(),
                end_date=df['date'].max(),
                style={'backgroundColor': colors['background']}
            )
        ], style={'width': '45%', 'display': 'inline-block', 'margin': '10px'})
    ], style={'backgroundColor': colors['card'], 'padding': '20px', 'margin': '20px', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),

    # Main content
    html.Div([
        # Row 1: Overview Section
        html.Div([
            # Sentiment Distribution
            html.Div([
                html.H3("Sentiment Distribution", style={'color': colors['text'], 'textAlign': 'center'}),
                dcc.Graph(id='sentiment-pie', style={'height': '400px'})
            ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': colors['card'], 'padding': '20px', 'margin': '1%', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'verticalAlign': 'top'}),
            
            # Word Cloud
            html.Div([
                html.H3("Review Word Clouds", style={'color': colors['text'], 'textAlign': 'center'}),
                html.Label("Select Sentiment", style={'color': colors['text'], 'textAlign': 'center', 'display': 'block'}),
                dcc.Dropdown(
                    id='wordcloud-sentiment',
                    options=[
                        {'label': 'Positive Reviews', 'value': 'positive'},
                        {'label': 'Non-Positive Reviews', 'value': 'non_positive'}
                    ],
                    value='positive',
                    style={'width': '50%', 'margin': '0 auto'}
                ),
                html.Img(id='wordcloud-img', style={'width': '100%', 'marginTop': '20px', 'height': '300px', 'objectFit': 'contain'})
            ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': colors['card'], 'padding': '20px', 'margin': '1%', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'verticalAlign': 'top'})
        ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}),

        # Row 2: Temporal Analysis
        html.Div([
            # Reviews Over Time
            html.Div([
                html.H3("Reviews Over Time", style={'color': colors['text'], 'textAlign': 'center'}),
                dcc.Graph(id='reviews-time-series', style={'height': '400px'})
            ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': colors['card'], 'padding': '20px', 'margin': '1%', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'verticalAlign': 'top'}),
            
            # Monthly Distribution
            html.Div([
                html.H3("Monthly Review Distribution", style={'color': colors['text'], 'textAlign': 'center'}),
                dcc.Graph(id='monthly-distribution', style={'height': '400px'})
            ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': colors['card'], 'padding': '20px', 'margin': '1%', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'verticalAlign': 'top'})
        ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}),

        # Row 3: Review Analysis
        html.Div([
            # Review Length Analysis
            html.Div([
                html.H3("Review Length by Sentiment", style={'color': colors['text'], 'textAlign': 'center'}),
                dcc.Graph(id='review-length-box', style={'height': '400px'})
            ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': colors['card'], 'padding': '20px', 'margin': '1%', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'verticalAlign': 'top'}),
            
            # Word Count Analysis
            html.Div([
                html.H3("Word Count by Sentiment", style={'color': colors['text'], 'textAlign': 'center'}),
                dcc.Graph(id='word-count-box', style={'height': '400px'})
            ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': colors['card'], 'padding': '20px', 'margin': '1%', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'verticalAlign': 'top'})
        ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}),

        # Row 4: Attraction Analysis
        html.Div([
            # Top Attractions
            html.Div([
                html.H3("Top 10 Most Reviewed Attractions", style={'color': colors['text'], 'textAlign': 'center'}),
                dcc.Graph(id='top-attractions', style={'height': '400px'})
            ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': colors['card'], 'padding': '20px', 'margin': '1%', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'verticalAlign': 'top'}),
            
            # Sentiment by Attraction
            html.Div([
                html.H3("Sentiment Distribution by Attraction", style={'color': colors['text'], 'textAlign': 'center'}),
                dcc.Graph(id='attraction-sentiment', style={'height': '400px'})
            ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': colors['card'], 'padding': '20px', 'margin': '1%', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'verticalAlign': 'top'})
        ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'})
    ], style={'backgroundColor': colors['background'], 'padding': '20px'})
], style={'backgroundColor': colors['background'], 'fontFamily': 'Arial, sans-serif'})

# Callbacks
@app.callback(
    Output('wordcloud-img', 'src'),
    Input('wordcloud-sentiment', 'value')
)
def update_wordcloud(sentiment):
    return generate_wordcloud(sentiment)

@app.callback(
    [Output('sentiment-pie', 'figure'),
     Output('reviews-time-series', 'figure'),
     Output('monthly-distribution', 'figure'),
     Output('review-length-box', 'figure'),
     Output('word-count-box', 'figure'),
     Output('top-attractions', 'figure'),
     Output('attraction-sentiment', 'figure')],
    [Input('attraction-dropdown', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_graphs(selected_attractions, start_date, end_date):
    # Filter data
    filtered_df = df.copy()
    if selected_attractions:
        filtered_df = filtered_df[filtered_df['attraction'].isin(selected_attractions)]
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]

    # 1. Sentiment Pie Chart with animation
    sentiment_counts = filtered_df['sentiment_ensemble'].value_counts()
    fig1 = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title='Sentiment Distribution',
        color=sentiment_counts.index,
        color_discrete_map={'positive': colors['positive'], 'non_positive': colors['non_positive']},
        hole=0.4  # Make it a donut chart
    )
    fig1.update_traces(textposition='inside', textinfo='percent+label')

    # 2. Reviews Time Series with animation
    time_series = filtered_df.groupby(['date', 'sentiment_ensemble']).size().reset_index(name='count')
    fig2 = px.line(
        time_series,
        x='date',
        y='count',
        color='sentiment_ensemble',
        title='Reviews Over Time',
        color_discrete_map={'positive': colors['positive'], 'non_positive': colors['non_positive']}
    )
    fig2.update_traces(mode='lines+markers', marker=dict(size=8))

    # 3. Monthly Distribution with animation
    monthly_data = filtered_df.groupby(['month_name', 'sentiment_ensemble']).size().reset_index(name='count')
    fig3 = px.bar(
        monthly_data,
        x='month_name',
        y='count',
        color='sentiment_ensemble',
        title='Monthly Review Distribution',
        color_discrete_map={'positive': colors['positive'], 'non_positive': colors['non_positive']}
    )
    fig3.update_traces(marker=dict(line=dict(width=1, color='white')))

    # 4. Review Length Box Plot with animation
    fig4 = px.box(
        filtered_df,
        x='sentiment_ensemble',
        y='review_length',
        color='sentiment_ensemble',
        title='Review Length Distribution',
        color_discrete_map={'positive': colors['positive'], 'non_positive': colors['non_positive']}
    )
    fig4.update_traces(boxmean=True)

    # 5. Word Count Box Plot with animation
    fig5 = px.box(
        filtered_df,
        x='sentiment_ensemble',
        y='word_count',
        color='sentiment_ensemble',
        title='Word Count Distribution',
        color_discrete_map={'positive': colors['positive'], 'non_positive': colors['non_positive']}
    )
    fig5.update_traces(boxmean=True)

    # 6. Top Attractions with animation
    top_attractions = filtered_df['attraction'].value_counts().nlargest(10).reset_index()
    top_attractions.columns = ['attraction', 'count']
    fig6 = px.bar(
        top_attractions,
        x='attraction',
        y='count',
        title='Top 10 Most Reviewed Attractions'
    )
    fig6.update_traces(marker=dict(line=dict(width=1, color='white')))

    # 7. Sentiment by Attraction with fixed overlapping
    attraction_sentiment = filtered_df.groupby(['attraction', 'sentiment_ensemble']).size().reset_index(name='count')
    fig7 = px.bar(
        attraction_sentiment,
        x='attraction',
        y='count',
        color='sentiment_ensemble',
        title='Sentiment Distribution by Attraction',
        color_discrete_map={'positive': colors['positive'], 'non_positive': colors['non_positive']}
    )

    # Update layout for all figures with enhanced styling and animations
    for fig in [fig1, fig2, fig3, fig4, fig5, fig6, fig7]:
        fig.update_layout(
            template='plotly_white',
            paper_bgcolor=colors['card'],
            plot_bgcolor=colors['card'],
            font_color=colors['text'],
            margin=dict(t=50, l=25, r=25, b=25),
            xaxis=dict(
                gridcolor=colors['grid'],
                showgrid=True,
                zeroline=True,
                zerolinecolor=colors['grid']
            ),
            yaxis=dict(
                gridcolor=colors['grid'],
                showgrid=True,
                zeroline=True,
                zerolinecolor=colors['grid']
            ),
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            # Add animation settings
            transition={
                'duration': 500,
                'easing': 'cubic-in-out'
            },
            # Add hover effects
            hovermode='closest',
            hoverlabel=dict(
                bgcolor=colors['card'],
                font_size=12,
                font_family="Arial"
            )
        )

    # Special handling for Sentiment by Attraction plot to fix overlapping
    fig7.update_layout(
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=10),
            tickmode='array',
            ticktext=[text[:20] + '...' if len(text) > 20 else text for text in attraction_sentiment['attraction'].unique()],
            tickvals=attraction_sentiment['attraction'].unique()
        ),
        margin=dict(b=100),  # Add bottom margin for rotated labels
        barmode='group',  # Group bars for better comparison
        bargap=0.15,  # Add gap between bar groups
        bargroupgap=0.1  # Add gap between bars in the same group
    )

    # Add animation to traces
    for fig in [fig1, fig2, fig3, fig4, fig5, fig6, fig7]:
        for trace in fig.data:
            trace.update(
                hoverlabel=dict(
                    bgcolor=colors['card'],
                    font_size=12,
                    font_family="Arial"
                )
            )

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7

if __name__ == '__main__':
    app.run(debug=True, port=8050)
