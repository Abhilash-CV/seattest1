import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Seat Allocation Adjuster",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-card h4 {
        margin: 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .metric-card h2 {
        margin: 0.5rem 0 0 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-radius: 0.5rem;
        color: #155724;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-radius: 0.5rem;
        color: #856404;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-radius: 0.5rem;
        color: #0c5460;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        color: white;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================

# Expected seat matrix
SEAT_MATRIX = {
    'SM': 50, 'EW': 10, 'EZ': 9, 'MU': 8, 'SC': 8, 
    'BH': 3, 'LA': 3, 'DV': 2, 'VK': 2, 'ST': 2, 
    'KN': 1, 'BX': 1, 'KU': 1
}

# Expected percentages
TOTAL_EXPECTED = sum(SEAT_MATRIX.values())
EXPECTED_PERCENTAGES = {
    cat: (seats / TOTAL_EXPECTED * 100) for cat, seats in SEAT_MATRIX.items()
}

# Color palette
COLORS = {
    'SM': '#1f77b4',
    'EW': '#ff7f0e',
    'EZ': '#2ca02c',
    'MU': '#d62728',
    'SC': '#9467bd',
    'BH': '#8c564b',
    'LA': '#e377c2',
    'DV': '#7f7f7f',
    'VK': '#bcbd22',
    'ST': '#17becf',
    'KN': '#aec7e8',
    'BX': '#ffbb78',
    'KU': '#98df8a'
}

# ============================================================================
# DATA GENERATION FUNCTIONS
# ============================================================================

def generate_sample_data():
    """Generate comprehensive sample data with issues"""
    data = """Program,Specialty,College,Type,Category,Seats
CS,IDK,KSD,G,SM,2
CS,IDK,KSD,G,EW,1
CS,IDK,KSD,G,EZ,1
CS,NSS,KSD,G,SM,3
CS,NSS,KSD,G,EW,1
CS,NSS,KSD,G,EZ,1
CS,LBT,KSD,G,SM,4
CS,LBT,KSD,G,EW,1
CS,LBT,KSD,G,SC,1
CS,LBT,KSD,G,EZ,1
CS,LBT,KSD,G,MU,1
CS,LBT,KSD,G,BH,1
CS,MDL,KSD,G,SM,4
CS,MDL,KSD,G,EW,1
CS,MDL,KSD,G,SC,1
CS,MDL,KSD,G,EZ,1
CS,MDL,KSD,G,MU,1
CS,CHN,KSD,G,SM,5
CS,CHN,KSD,G,EW,1
CS,CHN,KSD,G,SC,1
CS,CHN,KSD,G,EZ,1
CS,CHN,KSD,G,MU,1
CS,KGR,KSD,G,SM,3
CS,KGR,KSD,G,EW,1
CS,KGR,KSD,G,EZ,1
CS,KGR,KSD,G,MU,1
CS,CEA,KSD,G,SM,3
CS,CEA,KSD,G,EW,1
CS,CEA,KSD,G,SC,1
CS,CEA,KSD,G,EZ,1
CS,CEA,KSD,G,MU,1
CS,CEC,KSD,G,SM,3
CS,CEC,KSD,G,EW,1
CS,CEC,KSD,G,EZ,1
CS,CEC,KSD,G,MU,1
CS,CEK,KSD,G,SM,3
CS,CEK,KSD,G,EW,1
CS,CEK,KSD,G,EZ,1
CS,CEK,KSD,G,MU,1
CS,CEM,KSD,G,SM,3
CS,CEM,KSD,G,EW,1
CS,CEM,KSD,G,SC,1
CS,CEM,KSD,G,EZ,1
CS,CHN2,KSD,G,SM,5
CS,CHN2,KSD,G,EW,1
CS,CHN2,KSD,G,SC,1
CS,CHN2,KSD,G,EZ,1
CS,CHN2,KSD,G,MU,1
CS,KGR2,KSD,G,SM,3
CS,KGR2,KSD,G,EW,1
CS,KGR2,KSD,G,EZ,1
CS,KGR2,KSD,G,MU,1
CS,KNP,KSD,G,SM,3
CS,KNP,KSD,G,EW,1
CS,KNP,KSD,G,EZ,1
CS,KNP,KSD,G,MU,1
CS,KSD,KSD,G,SM,3
CS,KSD,KSD,G,EW,1
CS,KSD,KSD,G,SC,1
CS,KSD,KSD,G,EZ,1
CS,KSD,KSD,G,MU,1
CS,LBT2,KSD,G,SM,4
CS,LBT2,KSD,G,EW,1
CS,LBT2,KSD,G,SC,1
CS,LBT2,KSD,G,EZ,1
CS,LBT2,KSD,G,MU,1
CS,LBT2,KSD,G,BH,1
CS,MDL2,KSD,G,SM,4
CS,MDL2,KSD,G,EW,1
CS,MDL2,KSD,G,SC,1
CS,MDL2,KSD,G,EZ,1
CS,MDL2,KSD,G,MU,1
EC,IDK,KSD,G,SM,2
EC,IDK,KSD,G,EW,1
EC,IDK,KSD,G,EZ,1
EC,KKE,KSD,G,SM,2
EC,KKE,KSD,G,SC,1
EC,KNR,KSD,G,SM,2
EC,KNR,KSD,G,MU,1
EC,KTE,KSD,G,SM,1
EC,NSS,KSD,G,SM,3
EC,NSS,KSD,G,EW,1
EC,NSS,KSD,G,EZ,1
EC,PKD,KSD,G,SM,2
EC,PKD,KSD,G,MU,1
EC,TCR,KSD,G,SM,3
EC,TCR,KSD,G,EW,1
EC,TCR,KSD,G,EZ,1
EC,TRV,KSD,G,SM,2
EC,TRV,KSD,G,MU,1
EC,TVE,KSD,G,SM,2
EC,TVE,KSD,G,EZ,1
EC,WYD,KSD,G,SM,3
EC,WYD,KSD,G,EW,1
EC,WYD,KSD,G,SC,1
EC,WYD,KSD,G,EZ,1
EC,WYD,KSD,G,MU,1
ME,IDK,KSD,G,SM,2
ME,IDK,KSD,G,EW,1
ME,KKE,KSD,G,SM,2
ME,KKE,KSD,G,EW,1
ME,KNR,KSD,G,SM,2
ME,KNR,KSD,G,EZ,1
ME,KTE,KSD,G,SM,2
ME,KTE,KSD,G,EW,1
ME,NSS,KSD,G,SM,3
ME,NSS,KSD,G,EW,1
ME,NSS,KSD,G,EZ,1
ME,PKD,KSD,G,SM,2
ME,PKD,KSD,G,LA,1
ME,TCR,KSD,G,SM,3
ME,TCR,KSD,G,EW,1
ME,TCR,KSD,G,EZ,1
EE,IDK,KSD,G,SM,2
EE,IDK,KSD,G,EZ,1
EE,KNR,KSD,G,SM,2
EE,KNR,KSD,G,SC,1
EE,KTE,KSD,G,SM,2
EE,KTE,KSD,G,SC,1
EE,NSS,KSD,G,SM,3
EE,NSS,KSD,G,EW,1
EE,NSS,KSD,G,EZ,1
EE,PKD,KSD,G,SM,2
EE,PKD,KSD,G,EZ,1
EE,TCR,KSD,G,SM,3
EE,TCR,KSD,G,EW,1
EE,TCR,KSD,G,EZ,1"""
    
    df = pd.read_csv(StringIO(data))
    return df

# ============================================================================
# CORE ADJUSTMENT ALGORITHMS
# ============================================================================

def calculate_percentage_df(data, group_col='Program'):
    """Calculate percentages for a dataset"""
    results = []
    
    groups = data[group_col].unique()
    
    for group in groups:
        group_data = data[data[group_col] == group]
        total = group_data['Seats'].sum()
        
        for category in EXPECTED_PERCENTAGES.keys():
            seats = group_data[group_data['Category'] == category]['Seats'].sum()
            actual_pct = (seats / total * 100) if total > 0 else 0
            expected_pct = EXPECTED_PERCENTAGES[category]
            deviation = actual_pct - expected_pct
            
            results.append({
                group_col: group,
                'Category': category,
                'Seats': int(seats),
                'Total_Seats': int(total),
                'Actual_Percent': round(actual_pct, 2),
                'Expected_Percent': round(expected_pct, 2),
                'Deviation': round(deviation, 2),
                'Within_Tolerance': abs(deviation) <= 2
            })
    
    return pd.DataFrame(results)

def adjust_seats_smart(data, group_col='Program', max_iterations=100):
    """
    Smart adjustment using iterative proportional fitting
    """
    adjusted_data = data.copy()
    groups = adjusted_data[group_col].unique()
    
    adjustment_log = []
    iteration_count = 0
    
    for group in groups:
        group_mask = adjusted_data[group_col] == group
        group_total = adjusted_data.loc[group_mask, 'Seats'].sum()
        
        if group_total == 0:
            continue
        
        # Get current distribution
        current_dist = adjusted_data[group_mask].groupby('Category')['Seats'].sum().to_dict()
        
        # Calculate target seats
        target_seats = {}
        for cat, pct in EXPECTED_PERCENTAGES.items():
            target_seats[cat] = group_total * pct / 100
        
        # Iterative adjustment
        for iteration in range(max_iterations):
            # Calculate current percentages
            current_pct = {}
            for cat in EXPECTED_PERCENTAGES.keys():
                current_pct[cat] = (current_dist.get(cat, 0) / group_total * 100) if group_total > 0 else 0
            
            # Find categories that need adjustment
            adjustments = {}
            for cat in EXPECTED_PERCENTAGES.keys():
                current = current_pct.get(cat, 0)
                expected = EXPECTED_PERCENTAGES[cat]
                diff = expected - current
                
                if abs(diff) > 1.0:  # Need adjustment if >1% difference
                    adjustments[cat] = diff
            
            if not adjustments:
                break
            
            # Apply adjustments
            for cat, diff in adjustments.items():
                if diff > 0:  # Need to add seats to this category
                    # Find categories with surplus to take from
                    surplus_cats = [c for c, d in adjustments.items() if d < 0]
                    if surplus_cats:
                        # Take from the category with largest surplus
                        take_from = min(surplus_cats, key=lambda c: adjustments[c])
                        
                        # Find rows to add to
                        add_rows = adjusted_data[(adjusted_data[group_col] == group) & 
                                                (adjusted_data['Category'] == cat)]
                        # Find rows to remove from
                        remove_rows = adjusted_data[(adjusted_data[group_col] == group) & 
                                                   (adjusted_data['Category'] == take_from)]
                        
                        if not add_rows.empty and not remove_rows.empty:
                            # Add one seat
                            idx_add = add_rows.index[0]
                            adjusted_data.loc[idx_add, 'Seats'] += 1
                            
                            # Remove one seat
                            idx_remove = remove_rows.index[0]
                            if adjusted_data.loc[idx_remove, 'Seats'] > 1:
                                adjusted_data.loc[idx_remove, 'Seats'] -= 1
                            else:
                                # If only 1 seat, find another category with surplus
                                for other_cat in surplus_cats:
                                    other_rows = adjusted_data[(adjusted_data[group_col] == group) & 
                                                              (adjusted_data['Category'] == other_cat)]
                                    if not other_rows.empty:
                                        idx_remove = other_rows.index[0]
                                        if adjusted_data.loc[idx_remove, 'Seats'] > 1:
                                            adjusted_data.loc[idx_remove, 'Seats'] -= 1
                                            break
                            
                            # Update current distribution
                            current_dist = adjusted_data[group_mask].groupby('Category')['Seats'].sum().to_dict()
                            
                            # Log adjustment
                            adjustment_log.append({
                                group_col: group,
                                'Action': f'Move 1 seat from {take_from} to {cat}',
                                'Iteration': iteration + 1
                            })
            
            iteration_count += 1
    
    return adjusted_data, pd.DataFrame(adjustment_log)

def adjust_seats_simple(data, group_col='Program', tolerance=2):
    """
    Simple adjustment method
    """
    adjusted_data = data.copy()
    groups = adjusted_data[group_col].unique()
    
    adjustment_log = []
    
    for group in groups:
        group_mask = adjusted_data[group_col] == group
        group_total = adjusted_data.loc[group_mask, 'Seats'].sum()
        
        if group_total == 0:
            continue
        
        # Calculate expected seats for each category
        for category, expected_pct in EXPECTED_PERCENTAGES.items():
            expected_seats = group_total * expected_pct / 100
            current_seats = adjusted_data[(adjusted_data[group_col] == group) & 
                                         (adjusted_data['Category'] == category)]['Seats'].sum()
            
            diff = expected_seats - current_seats
            
            # If difference is significant (>1 seat)
            if abs(diff) > 1:
                if diff > 0:  # Need to add seats
                    # Find a category with surplus
                    for other_cat in EXPECTED_PERCENTAGES.keys():
                        if other_cat == category:
                            continue
                        other_current = adjusted_data[(adjusted_data[group_col] == group) & 
                                                     (adjusted_data['Category'] == other_cat)]['Seats'].sum()
                        other_expected = group_total * EXPECTED_PERCENTAGES[other_cat] / 100
                        other_diff = other_current - other_expected
                        
                        if other_diff > 1:  # This category has surplus
                            # Move one seat
                            add_rows = adjusted_data[(adjusted_data[group_col] == group) & 
                                                    (adjusted_data['Category'] == category)]
                            remove_rows = adjusted_data[(adjusted_data[group_col] == group) & 
                                                       (adjusted_data['Category'] == other_cat)]
                            
                            if not add_rows.empty and not remove_rows.empty:
                                idx_add = add_rows.index[0]
                                idx_remove = remove_rows.index[0]
                                
                                if adjusted_data.loc[idx_remove, 'Seats'] > 1:
                                    adjusted_data.loc[idx_add, 'Seats'] += 1
                                    adjusted_data.loc[idx_remove, 'Seats'] -= 1
                                    
                                    adjustment_log.append({
                                        group_col: group,
                                        'Action': f'Move 1 seat from {other_cat} to {category}',
                                        'Reason': f'Expected {category}: {expected_seats:.1f}, Current: {current_seats}'
                                    })
                                    break
    
    return adjusted_data, pd.DataFrame(adjustment_log)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_comparison_chart(original_data, adjusted_data, group_col='Program', category='EW'):
    """Compare original vs adjusted distribution for a specific category"""
    
    # Get original distribution
    orig_dist = original_data[original_data['Category'] == category].groupby(group_col)['Seats'].sum()
    adj_dist = adjusted_data[adjusted_data['Category'] == category].groupby(group_col)['Seats'].sum()
    
    # Combine
    all_groups = sorted(set(orig_dist.index) | set(adj_dist.index))
    orig_values = [int(orig_dist.get(p, 0)) for p in all_groups]
    adj_values = [int(adj_dist.get(p, 0)) for p in all_groups]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=all_groups,
        y=orig_values,
        name='Original',
        marker_color='#ff7f0e',
        text=orig_values,
        textposition='outside'
    ))
    fig.add_trace(go.Bar(
        x=all_groups,
        y=adj_values,
        name='Adjusted',
        marker_color='#2ca02c',
        text=adj_values,
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f'Seat Distribution for Category: {category}',
        xaxis_title=group_col,
        yaxis_title='Number of Seats',
        barmode='group',
        height=400,
        showlegend=True
    )
    
    return fig

def create_percentage_comparison(original_data, adjusted_data, group_col='Program'):
    """Create percentage comparison chart"""
    
    orig_pct = calculate_percentage_df(original_data, group_col)
    adj_pct = calculate_percentage_df(adjusted_data, group_col)
    
    # Merge for comparison
    merged = orig_pct.merge(
        adj_pct,
        on=[group_col, 'Category'],
        suffixes=('_orig', '_adj')
    )
    
    # Create chart
    fig = go.Figure()
    
    for category in EXPECTED_PERCENTAGES.keys():
        cat_data = merged[merged['Category'] == category]
        
        fig.add_trace(go.Bar(
            x=cat_data[group_col],
            y=cat_data['Actual_Percent_orig'],
            name=f'{category} (Original)',
            marker_color=COLORS.get(category, '#666'),
            opacity=0.5,
            legendgroup=category,
            showlegend=True
        ))
        
        fig.add_trace(go.Bar(
            x=cat_data[group_col],
            y=cat_data['Actual_Percent_adj'],
            name=f'{category} (Adjusted)',
            marker_color=COLORS.get(category, '#666'),
            legendgroup=category,
            showlegend=True
        ))
    
    fig.update_layout(
        title='Percentage Distribution Comparison',
        xaxis_title=group_col,
        yaxis_title='Percentage (%)',
        barmode='group',
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

def create_heatmap_comparison(original_data, adjusted_data, group_col='Program'):
    """Create heatmap comparison"""
    
    orig_pct = calculate_percentage_df(original_data, group_col)
    adj_pct = calculate_percentage_df(adjusted_data, group_col)
    
    # Create pivot tables for deviation
    orig_dev = orig_pct.pivot(index=group_col, columns='Category', values='Deviation').fillna(0)
    adj_dev = adj_pct.pivot(index=group_col, columns='Category', values='Deviation').fillna(0)
    
    # Combine for comparison
    combined = orig_dev.copy()
    for col in combined.columns:
        combined[col] = combined[col].astype(float)
    
    fig = px.imshow(
        combined,
        title='Percentage Deviation Heatmap (Original)',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        text_auto='.1f',
        aspect='auto',
        height=400
    )
    fig.update_layout(
        xaxis_title='Category',
        yaxis_title=group_col
    )
    
    return fig

def create_status_summary(data, group_col='Program'):
    """Create status summary with counts"""
    pct_df = calculate_percentage_df(data, group_col)
    status_summary = pct_df.groupby([group_col, 'Within_Tolerance']).size().reset_index(name='Count')
    status_pivot = status_summary.pivot(index=group_col, columns='Within_Tolerance', values='Count').fillna(0)
    
    # Rename columns
    status_pivot.columns = ['Failing' if col == False else 'Passing' for col in status_pivot.columns]
    if 'Passing' not in status_pivot.columns:
        status_pivot['Passing'] = 0
    if 'Failing' not in status_pivot.columns:
        status_pivot['Failing'] = 0
    
    status_pivot['Total'] = status_pivot['Passing'] + status_pivot['Failing']
    status_pivot['Pass_Rate'] = (status_pivot['Passing'] / status_pivot['Total'] * 100).round(1)
    
    return status_pivot

def create_validation_summary(original_data, adjusted_data, group_col='Program'):
    """Create validation summary comparing original vs adjusted"""
    
    orig_status = create_status_summary(original_data, group_col)
    adj_status = create_status_summary(adjusted_data, group_col)
    
    summary = []
    for group in orig_status.index:
        summary.append({
            group_col: group,
            'Original_Pass': int(orig_status.loc[group, 'Passing']),
            'Original_Fail': int(orig_status.loc[group, 'Failing']),
            'Original_Rate': orig_status.loc[group, 'Pass_Rate'],
            'Adjusted_Pass': int(adj_status.loc[group, 'Passing']) if group in adj_status.index else 0,
            'Adjusted_Fail': int(adj_status.loc[group, 'Failing']) if group in adj_status.index else 0,
            'Adjusted_Rate': adj_status.loc[group, 'Pass_Rate'] if group in adj_status.index else 0,
            'Improvement': (adj_status.loc[group, 'Pass_Rate'] - orig_status.loc[group, 'Pass_Rate']) if group in adj_status.index else 0
        })
    
    return pd.DataFrame(summary)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Display header
    st.markdown('<div class="main-header">🎯 Seat Allocation Adjuster</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automatically adjust seat allocation to match expected percentage distribution</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'original_data' not in st.session_state:
        st.session_state.original_data = None
    if 'adjusted_data' not in st.session_state:
        st.session_state.adjusted_data = None
    if 'adjustment_log' not in st.session_state:
        st.session_state.adjustment_log = None
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data input
        st.subheader("📁 Data Input")
        input_type = st.radio(
            "Choose input method:",
            ["Sample Data", "Upload CSV", "Paste Data"],
            key="input_type"
        )
        
        data = None
        
        if input_type == "Sample Data":
            data = generate_sample_data()
            st.success(f"✅ Loaded {len(data)} rows")
            with st.expander("Preview Data"):
                st.dataframe(data.head(10), use_container_width=True)
        
        elif input_type == "Upload CSV":
            uploaded = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded:
                try:
                    data = pd.read_csv(uploaded)
                    st.success(f"✅ Loaded {len(data)} rows")
                    with st.expander("Preview Data"):
                        st.dataframe(data.head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        else:
            st.info("📝 Paste your CSV data below")
            text_data = st.text_area("CSV Data", height=200)
            if text_data:
                try:
                    data = pd.read_csv(StringIO(text_data))
                    st.success(f"✅ Loaded {len(data)} rows")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        if data is not None:
            st.session_state.original_data = data
            
            # Adjustment options
            st.divider()
            st.subheader("⚙️ Adjustment Options")
            
            group_col = st.selectbox(
                "Group by:",
                ['Program', 'Specialty'],
                help="Select how to group data for adjustment"
            )
            
            adjustment_method = st.radio(
                "Adjustment Method:",
                ['Smart Adjustment (Recommended)', 'Simple Adjustment'],
                help="Smart: Iterative fitting | Simple: Direct adjustment"
            )
            
            tolerance = st.slider(
                "Tolerance (%)",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5,
                help="Maximum allowed deviation from expected percentage"
            )
            
            # Expected percentages display
            with st.expander("📊 Expected Percentages"):
                expected_df = pd.DataFrame({
                    'Category': list(EXPECTED_PERCENTAGES.keys()),
                    'Seats': list(SEAT_MATRIX.values()),
                    'Percentage': list(EXPECTED_PERCENTAGES.values())
                })
                st.dataframe(expected_df, use_container_width=True)
            
            # Adjust button
            if st.button("🔄 Adjust Seats", type="primary", use_container_width=True):
                if data is not None and not data.empty:
                    with st.spinner("Adjusting seat allocation..."):
                        try:
                            if adjustment_method == 'Smart Adjustment (Recommended)':
                                adjusted, log = adjust_seats_smart(data, group_col)
                            else:
                                adjusted, log = adjust_seats_simple(data, group_col, tolerance)
                            
                            st.session_state.adjusted_data = adjusted
                            st.session_state.adjustment_log = log
                            st.session_state.processed = True
                            st.success("✅ Seat allocation adjusted successfully!")
                            st.balloons()
                        except Exception as e:
                            st.error(f"❌ Error during adjustment: {str(e)}")
                else:
                    st.warning("⚠️ Please provide valid data")
    
    # Main content
    if st.session_state.processed and st.session_state.adjusted_data is not None:
        original = st.session_state.original_data
        adjusted = st.session_state.adjusted_data
        
        # Summary metrics
        st.subheader("📊 Adjustment Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            orig_total = original['Seats'].sum()
            adj_total = adjusted['Seats'].sum()
            st.metric("Total Seats", f"{int(adj_total):,}", f"{int(adj_total - orig_total):+d}")
        
        with col2:
            groups = len(original[group_col].unique())
            st.metric("Groups", groups)
        
        with col3:
            if not st.session_state.adjustment_log.empty:
                adjustments = len(st.session_state.adjustment_log)
                st.metric("Adjustments Made", adjustments)
            else:
                st.metric("Adjustments Made", 0)
        
        with col4:
            # Calculate improvement
            orig_status = create_status_summary(original, group_col)
            adj_status = create_status_summary(adjusted, group_col)
            
            orig_pass = orig_status['Passing'].sum() if 'Passing' in orig_status.columns else 0
            adj_pass = adj_status['Passing'].sum() if 'Passing' in adj_status.columns else 0
            
            improvement = adj_pass - orig_pass
            st.metric("Improvement", f"{improvement:+d}", delta_color="normal" if improvement > 0 else "inverse")
        
        with col5:
            orig_rate = (orig_pass / orig_status['Total'].sum() * 100) if orig_status['Total'].sum() > 0 else 0
            adj_rate = (adj_pass / adj_status['Total'].sum() * 100) if adj_status['Total'].sum() > 0 else 0
            st.metric("Pass Rate", f"{adj_rate:.1f}%", f"{adj_rate - orig_rate:+.1f}%")
        
        # Tabs
        tabs = st.tabs([
            "📊 Overview",
            "📈 Group Comparison",
            "📉 Category Analysis",
            "📋 Detailed Changes",
            "✅ Validation Report"
        ])
        
        # Tab 1: Overview
        with tabs[0]:
            # Original vs Adjusted summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Original Data Summary")
                orig_summary = original.groupby('Category')['Seats'].sum().reset_index()
                orig_summary['Percentage'] = (orig_summary['Seats'] / orig_summary['Seats'].sum() * 100).round(2)
                st.dataframe(
                    orig_summary,
                    column_config={
                        'Category': 'Category',
                        'Seats': st.column_config.NumberColumn('Seats', format='%d'),
                        'Percentage': st.column_config.NumberColumn('%', format='%.2f%%')
                    },
                    use_container_width=True
                )
            
            with col2:
                st.markdown("#### Adjusted Data Summary")
                adj_summary = adjusted.groupby('Category')['Seats'].sum().reset_index()
                adj_summary['Percentage'] = (adj_summary['Seats'] / adj_summary['Seats'].sum() * 100).round(2)
                st.dataframe(
                    adj_summary,
                    column_config={
                        'Category': 'Category',
                        'Seats': st.column_config.NumberColumn('Seats', format='%d'),
                        'Percentage': st.column_config.NumberColumn('%', format='%.2f%%')
                    },
                    use_container_width=True
                )
            
            # Overall category comparison chart
            st.markdown("#### Category Distribution Comparison")
            
            fig = go.Figure()
            
            # Original
            orig_cat = original.groupby('Category')['Seats'].sum().reset_index()
            orig_cat['Pct'] = (orig_cat['Seats'] / orig_cat['Seats'].sum() * 100)
            
            fig.add_trace(go.Bar(
                x=orig_cat['Category'],
                y=orig_cat['Pct'],
                name='Original',
                marker_color='#ff7f0e',
                text=orig_cat['Pct'].apply(lambda x: f'{x:.1f}%'),
                textposition='outside'
            ))
            
            # Adjusted
            adj_cat = adjusted.groupby('Category')['Seats'].sum().reset_index()
            adj_cat['Pct'] = (adj_cat['Seats'] / adj_cat['Seats'].sum() * 100)
            
            fig.add_trace(go.Bar(
                x=adj_cat['Category'],
                y=adj_cat['Pct'],
                name='Adjusted',
                marker_color='#2ca02c',
                text=adj_cat['Pct'].apply(lambda x: f'{x:.1f}%'),
                textposition='outside'
            ))
            
            # Expected
            expected_df = pd.DataFrame({
                'Category': list(EXPECTED_PERCENTAGES.keys()),
                'Pct': list(EXPECTED_PERCENTAGES.values())
            })
            
            fig.add_trace(go.Bar(
                x=expected_df['Category'],
                y=expected_df['Pct'],
                name='Expected',
                marker_color='#1f77b4',
                text=expected_df['Pct'].apply(lambda x: f'{x:.1f}%'),
                textposition='outside'
            ))
            
            fig.update_layout(
                title='Category Distribution: Original vs Adjusted vs Expected',
                xaxis_title='Category',
                yaxis_title='Percentage (%)',
                barmode='group',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 2: Group Comparison
        with tabs[1]:
            st.subheader(f"{group_col} Level Comparison")
            
            # Select category to compare
            selected_category = st.selectbox(
                "Select Category to Compare:",
                list(EXPECTED_PERCENTAGES.keys())
            )
            
            # Show comparison chart
            fig = create_comparison_chart(original, adjusted, group_col, selected_category)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show percentage comparison
            st.markdown("#### Percentage Distribution")
            fig_pct = create_percentage_comparison(original, adjusted, group_col)
            st.plotly_chart(fig_pct, use_container_width=True)
        
        # Tab 3: Category Analysis
        with tabs[2]:
            st.subheader("Category Level Analysis")
            
            # Show detailed category comparison
            orig_pct = calculate_percentage_df(original, group_col)
            adj_pct = calculate_percentage_df(adjusted, group_col)
            
            # Merge for comparison
            comp_df = orig_pct.merge(
                adj_pct,
                on=[group_col, 'Category'],
                suffixes=('_Original', '_Adjusted')
            )
            
            # Add status
            comp_df['Status_Original'] = comp_df['Within_Tolerance_Original'].map({True: '✅', False: '⚠️'})
            comp_df['Status_Adjusted'] = comp_df['Within_Tolerance_Adjusted'].map({True: '✅', False: '⚠️'})
            
            st.dataframe(
                comp_df[[group_col, 'Category', 
                        'Seats_Original', 'Seats_Adjusted',
                        'Actual_Percent_Original', 'Actual_Percent_Adjusted',
                        'Expected_Percent_Original', 
                        'Deviation_Original', 'Deviation_Adjusted',
                        'Status_Original', 'Status_Adjusted']],
                column_config={
                    group_col: group_col,
                    'Category': 'Category',
                    'Seats_Original': st.column_config.NumberColumn('Original Seats', format='%d'),
                    'Seats_Adjusted': st.column_config.NumberColumn('Adjusted Seats', format='%d'),
                    'Actual_Percent_Original': st.column_config.NumberColumn('Original %', format='%.2f%%'),
                    'Actual_Percent_Adjusted': st.column_config.NumberColumn('Adjusted %', format='%.2f%%'),
                    'Expected_Percent_Original': st.column_config.NumberColumn('Expected %', format='%.2f%%'),
                    'Deviation_Original': st.column_config.NumberColumn('Original Dev', format='%.2f%%'),
                    'Deviation_Adjusted': st.column_config.NumberColumn('Adjusted Dev', format='%.2f%%'),
                    'Status_Original': 'Original Status',
                    'Status_Adjusted': 'Adjusted Status'
                },
                use_container_width=True
            )
            
            # Heatmap for deviations
            st.markdown("#### Deviation Heatmap (Adjusted)")
            fig_heatmap = create_heatmap_comparison(original, adjusted, group_col)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Tab 4: Detailed Changes
        with tabs[3]:
            st.subheader("📋 Detailed Changes")
            
            # Show adjustment log
            if not st.session_state.adjustment_log.empty:
                st.markdown("#### Adjustment Log")
                st.dataframe(st.session_state.adjustment_log, use_container_width=True)
                
                # Summary of changes
                st.markdown("#### Change Summary")
                change_summary = st.session_state.adjustment_log['Action'].value_counts().reset_index()
                change_summary.columns = ['Action', 'Count']
                st.dataframe(change_summary, use_container_width=True)
            else:
                st.info("No adjustments were made - all distributions are already within tolerance")
            
            # Show changed rows
            st.markdown("#### Rows That Changed")
            
            # Find rows that changed
            merged = original.merge(
                adjusted,
                on=['Program', 'Specialty', 'College', 'Type', 'Category'],
                suffixes=('_Original', '_Adjusted')
            )
            
            changed = merged[merged['Seats_Original'] != merged['Seats_Adjusted']]
            
            if not changed.empty:
                changed['Change'] = changed['Seats_Adjusted'] - changed['Seats_Original']
                st.dataframe(
                    changed[['Program', 'Specialty', 'College', 'Category', 
                            'Seats_Original', 'Seats_Adjusted', 'Change']],
                    column_config={
                        'Program': 'Program',
                        'Specialty': 'Specialty',
                        'College': 'College',
                        'Category': 'Category',
                        'Seats_Original': st.column_config.NumberColumn('Original', format='%d'),
                        'Seats_Adjusted': st.column_config.NumberColumn('Adjusted', format='%d'),
                        'Change': st.column_config.NumberColumn('Change', format='%+d')
                    },
                    use_container_width=True
                )
            else:
                st.info("No rows were changed")
            
            # Download options
            st.markdown("#### Download Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_adjusted = adjusted.to_csv(index=False)
                st.download_button(
                    "📥 Download Adjusted Data",
                    csv_adjusted,
                    f"adjusted_seats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                if not changed.empty:
                    csv_changes = changed.to_csv(index=False)
                    st.download_button(
                        "📥 Download Changes Log",
                        csv_changes,
                        f"changes_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
            
            with col3:
                if not st.session_state.adjustment_log.empty:
                    csv_log = st.session_state.adjustment_log.to_csv(index=False)
                    st.download_button(
                        "📥 Download Adjustment Log",
                        csv_log,
                        f"adjustment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
        
        # Tab 5: Validation Report
        with tabs[4]:
            st.subheader("✅ Validation Report")
            
            # Validation summary
            validation_summary = create_validation_summary(original, adjusted, group_col)
            
            st.markdown(f"#### {group_col} Level Validation")
            st.dataframe(
                validation_summary,
                column_config={
                    group_col: group_col,
                    'Original_Pass': st.column_config.NumberColumn('Orig Pass', format='%d'),
                    'Original_Fail': st.column_config.NumberColumn('Orig Fail', format='%d'),
                    'Original_Rate': st.column_config.NumberColumn('Orig Rate', format='%.1f%%'),
                    'Adjusted_Pass': st.column_config.NumberColumn('Adj Pass', format='%d'),
                    'Adjusted_Fail': st.column_config.NumberColumn('Adj Fail', format='%d'),
                    'Adjusted_Rate': st.column_config.NumberColumn('Adj Rate', format='%.1f%%'),
                    'Improvement': st.column_config.NumberColumn('Improvement', format='%.1f%%')
                },
                use_container_width=True
            )
            
            # Overall statistics
            st.markdown("#### Overall Statistics")
            
            orig_status = create_status_summary(original, group_col)
            adj_status = create_status_summary(adjusted, group_col)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_cats = len(original[group_col].unique()) * len(EXPECTED_PERCENTAGES)
                st.metric("Total Checks", total_cats)
            
            with col2:
                orig_pass = orig_status['Passing'].sum() if 'Passing' in orig_status.columns else 0
                st.metric("Original Pass", orig_pass)
            
            with col3:
                adj_pass = adj_status['Passing'].sum() if 'Passing' in adj_status.columns else 0
                st.metric("Adjusted Pass", adj_pass)
            
            with col4:
                improvement = adj_pass - orig_pass
                st.metric("Improvement", f"{improvement:+d}", delta_color="normal" if improvement > 0 else "inverse")
            
            # Show groups with issues
            failing_groups = validation_summary[validation_summary['Adjusted_Rate'] < 100]
            
            if not failing_groups.empty:
                st.markdown("#### ⚠️ Groups Still Having Issues")
                st.dataframe(
                    failing_groups[[group_col, 'Adjusted_Rate', 'Adjusted_Fail']],
                    column_config={
                        group_col: group_col,
                        'Adjusted_Rate': st.column_config.NumberColumn('Pass Rate', format='%.1f%%'),
                        'Adjusted_Fail': st.column_config.NumberColumn('Failing Categories', format='%d')
                    },
                    use_container_width=True
                )
            else:
                st.success("🎉 All groups are perfectly balanced!")
            
            # Detailed validation per group
            st.markdown("#### Detailed Validation")
            
            selected_group = st.selectbox(
                f"Select {group_col} to view details:",
                validation_summary[group_col].tolist()
            )
            
            if selected_group:
                orig_group = calculate_percentage_df(
                    original[original[group_col] == selected_group], 
                    group_col
                )
                adj_group = calculate_percentage_df(
                    adjusted[adjusted[group_col] == selected_group], 
                    group_col
                )
                
                # Merge for display
                group_comp = orig_group.merge(
                    adj_group,
                    on=['Category'],
                    suffixes=('_Original', '_Adjusted')
                )
                
                group_comp['Status_Original'] = group_comp['Within_Tolerance_Original'].map({True: '✅', False: '⚠️'})
                group_comp['Status_Adjusted'] = group_comp['Within_Tolerance_Adjusted'].map({True: '✅', False: '⚠️'})
                
                st.dataframe(
                    group_comp[['Category', 
                               'Seats_Original', 'Seats_Adjusted',
                               'Actual_Percent_Original', 'Actual_Percent_Adjusted',
                               'Expected_Percent_Original',
                               'Deviation_Original', 'Deviation_Adjusted',
                               'Status_Original', 'Status_Adjusted']],
                    column_config={
                        'Category': 'Category',
                        'Seats_Original': st.column_config.NumberColumn('Original Seats', format='%d'),
                        'Seats_Adjusted': st.column_config.NumberColumn('Adjusted Seats', format='%d'),
                        'Actual_Percent_Original': st.column_config.NumberColumn('Original %', format='%.2f%%'),
                        'Actual_Percent_Adjusted': st.column_config.NumberColumn('Adjusted %', format='%.2f%%'),
                        'Expected_Percent_Original': st.column_config.NumberColumn('Expected %', format='%.2f%%'),
                        'Deviation_Original': st.column_config.NumberColumn('Original Dev', format='%.2f%%'),
                        'Deviation_Adjusted': st.column_config.NumberColumn('Adjusted Dev', format='%.2f%%'),
                        'Status_Original': 'Original Status',
                        'Status_Adjusted': 'Adjusted Status'
                    },
                    use_container_width=True
                )
    
    else:
        # Welcome message
        st.markdown("""
        <div class="info-box">
        👈 <strong>Get Started:</strong> Upload your data in the sidebar and click 'Adjust Seats'
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 What This Tool Does
            
            This tool **automatically adjusts seat allocations** to match the expected percentage distribution:
            
            #### 🔧 Adjustment Process
            
            1. **Analyzes** current seat distribution by program/specialty
            2. **Compares** actual percentages against expected (SM:50%, EW:10%, etc.)
            3. **Adjusts** seats to match expected percentages
            4. **Validates** that all groups now meet the criteria
            
            #### 📊 Example Issue Fixed
            
            **Before Adjustment (CS Program with 160 seats):**
            - EW: 23 seats (14.4%) ❌ (Expected: 10%)
            - EZ: 21 seats (13.1%) ❌ (Expected: 9%)
            - BH: 1 seat (0.6%) ❌ (Expected: 3%)
            
            **After Adjustment:**
            - EW: 16 seats (10%) ✅
            - EZ: 14 seats (9%) ✅
            - BH: 5 seats (3%) ✅
            """)
        
        with col2:
            st.markdown("""
            ### 📁 Required Data Format
            
            Your CSV should have these columns:
            - **Program**: Program code (CS, EC, ME, etc.)
            - **Specialty**: Specialty name
            - **College**: College name
            - **Type**: Type (G, etc.)
            - **Category**: Seat category (SM, EW, EZ, etc.)
            - **Seats**: Number of seats
            
            ### 📊 Expected Distribution (100 seats)
            
            | Category | Seats | Percentage |
            |----------|-------|------------|
            | SM | 50 | 50% |
            | EW | 10 | 10% |
            | EZ | 9 | 9% |
            | MU | 8 | 8% |
            | SC | 8 | 8% |
            | BH | 3 | 3% |
            | LA | 3 | 3% |
            | DV | 2 | 2% |
            | VK | 2 | 2% |
            | ST | 2 | 2% |
            | KN | 1 | 1% |
            | BX | 1 | 1% |
            | KU | 1 | 1% |
            """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🚀 Built with Streamlit | Seat Allocation Adjuster v2.0</p>
        <p style="font-size: 0.8rem;">Automatic adjustment to match expected percentage distribution</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
