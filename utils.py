import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import json

def create_gauge_chart(value, title, max_value=1.0):
    """
    Create a gauge chart for confidence scores
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgreen"},
                {'range': [0.3, 0.7], 'color': "yellow"},
                {'range': [0.7, 1.0], 'color': "salmon"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.5
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_feature_importance_chart(features):
    """
    Create a bar chart for feature importance
    """
    df = pd.DataFrame(features)
    
    fig = px.bar(
        df.head(15),
        x='importance',
        y='name',
        orientation='h',
        title='Top 15 Most Important Features',
        color='importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='Importance Score',
        yaxis_title='Feature'
    )
    
    return fig

def create_sentiment_radar_chart(sentiment_scores):
    """
    Create a radar chart for sentiment analysis
    """
    categories = ['Negative', 'Neutral', 'Positive', 'Compound']
    
    fig = go.Figure(data=go.Scatterpolar(
        r=[
            sentiment_scores.get('vader_neg', 0),
            sentiment_scores.get('vader_neu', 0),
            sentiment_scores.get('vader_pos', 0),
            sentiment_scores.get('vader_compound', 0) + 0.5  # Shift to 0-1 scale
        ],
        theta=categories,
        fill='toself',
        marker=dict(color='rgba(102, 126, 234, 0.8)')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        height=400,
        title='Sentiment Analysis Radar'
    )
    
    return fig

def create_word_cloud(text):
    """
    Create a word cloud image
    """
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate(text)
    
    # Convert to base64 for display
    img = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(img, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    img.seek(0)
    
    return base64.b64encode(img.getvalue()).decode()

def format_analysis_results(result):
    """
    Format analysis results for display
    """
    formatted = {
        'classification': '🚫 FAKE NEWS' if result['is_fake'] else '✅ LEGITIMATE NEWS',
        'confidence': f"{result['confidence']*100:.1f}%",
        'fake_prob': f"{result['fake_probability']*100:.1f}%",
        'real_prob': f"{result['real_probability']*100:.1f}%",
        'fact_check': {
            'credibility': f"{result['fact_check']['credibility_score']*100:.1f}%",
            'sources_checked': result['fact_check']['sources_checked'],
            'matching_claims': result['fact_check']['matching_claims'],
            'last_checked': result['fact_check']['last_checked']
        }
    }
    
    return formatted

def get_writing_style_indicators(features):
    """
    Extract writing style indicators
    """
    indicators = []
    
    # Check for sensationalism
    if features.get('exclamation_count', 0) > 3:
        indicators.append({
            'type': 'warning',
            'message': 'Excessive use of exclamation marks (sensationalism)'
        })
    
    if features.get('caps_ratio', 0) > 0.3:
        indicators.append({
            'type': 'warning',
            'message': 'High proportion of uppercase text (potential shouting)'
        })
    
    if features.get('vader_compound', 0) > 0.5:
        indicators.append({
            'type': 'info',
            'message': 'Highly positive emotional language detected'
        })
    elif features.get('vader_compound', 0) < -0.5:
        indicators.append({
            'type': 'info',
            'message': 'Highly negative emotional language detected'
        })
    
    # Check for clickbait patterns
    if features.get('question_count', 0) > 2:
        indicators.append({
            'type': 'warning',
            'message': 'Multiple questions (potential clickbait)'
        })
    
    # Check readability
    readability = features.get('flesch_reading_ease', 50)
    if readability < 30:
        indicators.append({
            'type': 'info',
            'message': 'Very complex text (unusual for news articles)'
        })
    elif readability > 70:
        indicators.append({
            'type': 'success',
            'message': 'Good readability score'
        })
    
    return indicators

def export_report(result, text):
    """
    Export analysis results as JSON
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'analyzed_text': text[:500] + '...' if len(text) > 500 else text,
        'classification': {
            'is_fake': result['is_fake'],
            'confidence': result['confidence'],
            'fake_probability': result['fake_probability'],
            'real_probability': result['real_probability']
        },
        'features': result['features'],
        'fact_check': result['fact_check']
    }
    
    return json.dumps(report, indent=2)