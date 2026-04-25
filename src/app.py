import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# Set page layout and title
st.set_page_config(page_title="PRISM Persona Dashboard", layout="wide")
st.title("PRISM: Dynamic Persona Visualization")
st.markdown("This dashboard calculates persona scores directly from your clustering centroids.")

# Define the relative path to your CSV file
# Using raw string to handle Windows/Unix path separators correctly
CSV_PATH = r"./cluster_result/5centroid/2000user_cluster_centroids.csv"

@st.cache_data
def load_and_normalize_centroids(path):
    """
    Reads the centroid CSV and performs Min-Max normalization 
    to map raw statistics to a 0-100 scale for radar charts.
    """
    if not os.path.exists(path):
        return None, None
    
    df = pd.read_csv(path)
    
    # Mapping raw columns to descriptive Radar axes
    feature_mapping = {
        'post_frequency_per_day': 'Activity',
        'reply_count': 'Engagement',
        'original_post_to_repost_ratio': 'Originality',
        'avg_subjectivity': 'Subjectivity',
        'flesch_reading_ease': 'Readability',
        'avg_polarity': 'Polarity'
    }
    
    # Extract only the columns needed for the 6-axis chart
    cols_to_use = list(feature_mapping.keys())
    subset = df[['cluster'] + cols_to_use].copy()
    
    # Dynamic Normalization: (Value - Min) / (Max - Min) * 100
    # This prevents hardcoding and adapts to new data patterns
    norm_df = subset.copy()
    for col in cols_to_use:
        val_min = subset[col].min()
        val_max = subset[col].max()
        if val_max == val_min:
            norm_df[col] = 0
        else:
            norm_df[col] = (subset[col] - val_min) / (val_max - val_min) * 100
            
    # Rename columns for the chart UI
    norm_df = norm_df.rename(columns=feature_mapping)
    return norm_df, feature_mapping

# Attempt to load data
data, mapping = load_and_normalize_centroids(CSV_PATH)

if data is None:
    st.error(f"❌ Could not find CSV at: {os.path.abspath(CSV_PATH)}")
    st.info("Check if the folder structure matches: ../cluster_result/5centroid/")
    st.stop()

# Define the qualitative Persona labels based on your analysis
persona_labels = {
    0: "The Everyday Creator",
    1: "The Heavy Debater",
    2: "The Silent Lurker",
    3: "The Pure Curator",
    4: "The Hyper-Spreader"
}

# Sidebar: Selection of Cluster
st.sidebar.header("Select Persona Cluster")
selected_id = st.sidebar.selectbox(
    "Choose which persona to visualize:",
    options=list(persona_labels.keys()),
    format_func=lambda x: f"Cluster {x}: {persona_labels[x]}"
)

# Extract values for the selected cluster
categories = list(mapping.values())
current_values = data[data['cluster'] == selected_id][categories].values.flatten().tolist()

# Plotting the Radar Chart using Plotly
fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=current_values,
    theta=categories,
    fill='toself',
    name=persona_labels[selected_id],
    line_color='#1E88E5',
    fillcolor='rgba(30, 136, 229, 0.4)'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 100]) # Lock range to 0-100
    ),
    showlegend=False,
    height=600
)

# Render the interactive plot
st.plotly_chart(fig, use_container_width=True)

# Debugging section: Show raw processed data
with st.expander("View Normalized Score Table"):
    st.write("Calculated scores (0-100 scale) for all clusters:")
    st.dataframe(data.set_index('cluster').round(2))