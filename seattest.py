import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Seat Allocation Adjuster",
    page_icon="🎯",
    layout="wide"
)

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

# ============================================================================
# SEAT ADJUSTMENT FUNCTIONS
# ============================================================================

def adjust_seats_by_percentage(data, program_col='Program', tolerance=2):
    """
    Adjust seat allocation to match expected percentages
    """
    # Create a copy
    adjusted_data = data.copy()
    
    # Get unique programs
    programs = adjusted_data[program_col].unique()
    
    # Store adjustment details
    adjustment_log = []
    
    for program in programs:
        # Get program total
        program_data = adjusted_data[adjusted_data[program_col] == program]
        program_total = program_data['Seats'].sum()
        
        if program_total == 0:
            continue
        
        # Calculate expected seats for each category in this program
        for category, expected_pct in EXPECTED_PERCENTAGES.items():
            expected_seats = (program_total * expected_pct / 100)
            
            # Get current seats for this category in this program
            current_mask = (adjusted_data[program_col] == program) & (adjusted_data['Category'] == category)
            current_seats = adjusted_data.loc[current_mask, 'Seats'].sum()
            
            # Calculate difference
            diff = expected_seats - current_seats
            
            # Log the adjustment needed
            if abs(diff) > 0.5:  # Only log significant differences
                adjustment_log.append({
                    program_col: program,
                    'Category': category,
                    'Current_Seats': int(current_seats),
                    'Expected_Seats': round(expected_seats, 1),
                    'Diff': round(diff, 1),
                    'Program_Total': program_total
                })
            
            # Adjust the seats if within tolerance
            if abs(diff) > tolerance * program_total / 100:
                # Find a row to adjust
                if current_seats > 0 and diff < 0:  # Need to reduce
                    # Find rows with this category and reduce
                    rows_to_reduce = adjusted_data[(adjusted_data[program_col] == program) & 
                                                   (adjusted_data['Category'] == category)]
                    for idx in rows_to_reduce.index:
                        if adjusted_data.loc[idx, 'Seats'] > 1:
                            adjusted_data.loc[idx, 'Seats'] -= 1
                            break
    
    # Recalculate totals after adjustments
    # Ensure total per program remains same
    for program in programs:
        program_mask = adjusted_data[program_col] == program
        new_total = adjusted_data.loc[program_mask, 'Seats'].sum()
        original_total = data[data[program_col] == program]['Seats'].sum()
        
        if new_total != original_total:
            # Find SM category (or any category with seats) to adjust
            sm_rows = adjusted_data[(adjusted_data[program_col] == program) & 
                                    (adjusted_data['Category'] == 'SM')]
            if not sm_rows.empty:
                idx = sm_rows.index[0]
                adjusted_data.loc[idx, 'Seats'] += (original_total - new_total)
    
    return adjusted_data, pd.DataFrame(adjustment_log)

def smart_adjust_seats(data, program_col='Program'):
    """
    Smart adjustment using iterative proportional fitting
    """
    adjusted_data = data.copy()
    programs = adjusted_data[program_col].unique()
    
    adjustment_history = []
    
    for program in programs:
        program_mask = adjusted_data[program_col] == program
        program_total = adjusted_data.loc[program_mask, 'Seats'].sum()
        
        if program_total == 0:
            continue
        
        # Get current distribution
        current_dist = adjusted_data[program_mask].groupby('Category')['Seats'].sum()
        
        # Calculate target distribution
        target_seats = {}
        for cat, pct in EXPECTED_PERCENTAGES.items():
            target_seats[cat] = program_total * pct / 100
        
        # Iterative adjustment
        max_iterations = 100
        for _ in range(max_iterations):
            # Calculate current percentages
            current_pct = current_dist / program_total * 100
            
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
                        
                        # Find rows to adjust
                        add_rows = adjusted_data[(adjusted_data[program_col] == program) & 
                                                (adjusted_data['Category'] == cat)]
                        remove_rows = adjusted_data[(adjusted_data[program_col] == program) & 
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
                                # If only 1 seat, find another category
                                for other_cat in surplus_cats:
                                    other_rows = adjusted_data[(adjusted_data[program_col] == program) & 
                                                              (adjusted_data['Category'] == other_cat)]
                                    if not other_rows.empty:
                                        idx_remove = other_rows.index[0]
                                        if adjusted_data.loc[idx_remove, 'Seats'] > 1:
                                            adjusted_data.loc[idx_remove, 'Seats'] -= 1
                                            break
                            
                            # Update current distribution
                            current_dist = adjusted_data[program_mask].groupby('Category')['Seats'].sum()
                            
                            # Log adjustment
                            adjustment_history.append({
                                program_col: program,
                                'Action': f'Move 1 seat from {take_from} to {cat}',
                                'New_Seats': int(adjusted_data.loc[program_mask, 'Seats'].sum())
                            })
    
    return adjusted_data, pd.DataFrame(adjustment_history)

def get_sample_data():
    """Create sample data with the CS example showing issues"""
    data = """Program,Specialty,College,Type,Category,Seats
CS,IDK,COLLEGE,G,SM,2
CS,IDK,COLLEGE,G,EW,1
CS,IDK,COLLEGE,G,EZ,1
CS,NSS,COLLEGE,G,SM,3
CS,NSS,COLLEGE,G,EW,1
CS,NSS,COLLEGE,G,EZ,1
CS,LBT,COLLEGE,G,SM,4
CS,LBT,COLLEGE,G,EW,1
CS,LBT,COLLEGE,G,SC,1
CS,LBT,COLLEGE,G,EZ,1
CS,LBT,COLLEGE,G,MU,1
CS,LBT,COLLEGE,G,BH,1
CS,MDL,COLLEGE,G,SM,4
CS,MDL,COLLEGE,G,EW,1
CS,MDL,COLLEGE,G,SC,1
CS,MDL,COLLEGE,G,EZ,1
CS,MDL,COLLEGE,G,MU,1
CS,CHN,COLLEGE,G,SM,5
CS,CHN,COLLEGE,G,EW,1
CS,CHN,COLLEGE,G,SC,1
CS,CHN,COLLEGE,G,EZ,1
CS,CHN,COLLEGE,G,MU,1
CS,KGR,COLLEGE,G,SM,3
CS,KGR,COLLEGE,G,EW,1
CS,KGR,COLLEGE,G,EZ,1
CS,KGR,COLLEGE,G,MU,1
CS,CEA,COLLEGE,G,SM,3
CS,CEA,COLLEGE,G,EW,1
CS,CEA,COLLEGE,G,SC,1
CS,CEA,COLLEGE,G,EZ,1
CS,CEA,COLLEGE,G,MU,1
CS,CEC,COLLEGE,G,SM,3
CS,CEC,COLLEGE,G,EW,1
CS,CEC,COLLEGE,G,EZ,1
CS,CEC,COLLEGE,G,MU,1
CS,CEK,COLLEGE,G,SM,3
CS,CEK,COLLEGE,G,EW,1
CS,CEK,COLLEGE,G,EZ,1
CS,CEK,COLLEGE,G,MU,1
CS,CEM,COLLEGE,G,SM,3
CS,CEM,COLLEGE,G,EW,1
CS,CEM,COLLEGE,G,SC,1
CS,CEM,COLLEGE,G,EZ,1
CS,CHN2,COLLEGE,G,SM,5
CS,CHN2,COLLEGE,G,EW,1
CS,CHN2,COLLEGE,G,SC,1
CS,CHN2,COLLEGE,G,EZ,1
CS,CHN2,COLLEGE,G,MU,1
CS,KGR2,COLLEGE,G,SM,3
CS,KGR2,COLLEGE,G,EW,1
CS,KGR2,COLLEGE,G,EZ,1
CS,KGR2,COLLEGE,G,MU,1
CS,KNP,COLLEGE,G,SM,3
CS,KNP,COLLEGE,G,EW,1
CS,KNP,COLLEGE,G,EZ,1
CS,KNP,COLLEGE,G,MU,1
CS,KSD,COLLEGE,G,SM,3
CS,KSD,COLLEGE,G,EW,1
CS,KSD,COLLEGE,G,SC,1
CS,KSD,COLLEGE,G,EZ,1
CS,KSD,COLLEGE,G,MU,1
CS,LBT2,COLLEGE,G,SM,4
CS,LBT2,COLLEGE,G,EW,1
CS,LBT2,COLLEGE,G,SC,1
CS,LBT2,COLLEGE,G,EZ,1
CS,LBT2,COLLEGE,G,MU,1
CS,LBT2,COLLEGE,G,BH,1
CS,MDL2,COLLEGE,G,SM,4
CS,MDL2,COLLEGE,G,EW,1
CS,MDL2,COLLEGE,G,SC,1
CS,MDL2,COLLEGE,G,EZ,1
CS,MDL2,COLLEGE,G,MU,1"""
    
    df = pd.read_csv(StringIO(data))
    return df

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def compare_distribution(original_data, adjusted_data, program_col='Program', category='EW'):
    """Compare original vs adjusted distribution for a specific category"""
    
    # Get original distribution
    orig_dist = original_data[original_data['Category'] == category].groupby(program_col)['Seats'].sum()
    adj_dist = adjusted_data[adjusted_data['Category'] == category].groupby(program_col)['Seats'].sum()
    
    # Combine
    all_programs = sorted(set(orig_dist.index) | set(adj_dist.index))
    orig_values = [orig_dist.get(p, 0) for p in all_programs]
    adj_values = [adj_dist.get(p, 0) for p in all_programs]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=all_programs,
        y=orig_values,
        name='Original',
        marker_color='#ff7f0e'
    ))
    fig.add_trace(go.Bar(
        x=all_programs,
        y=adj_values,
        name='Adjusted',
        marker_color='#2ca02c'
    ))
    
    fig.update_layout(
        title=f'Seat Distribution for Category: {category}',
        xaxis_title=program_col,
        yaxis_title='Number of Seats',
        barmode='group',
        height=400
    )
    
    return fig

def create_summary_table(original_data, adjusted_data, program_col='Program'):
    """Create summary table comparing original vs adjusted"""
    summary = []
    
    programs = sorted(set(original_data[program_col].unique()) | set(adjusted_data[program_col].unique()))
    
    for program in programs:
        orig_total = original_data[original_data[program_col] == program]['Seats'].sum()
        adj_total = adjusted_data[adjusted_data[program_col] == program]['Seats'].sum()
        
        # Get category breakdown
        orig_cats = original_data[original_data[program_col] == program].groupby('Category')['Seats'].sum()
        adj_cats = adjusted_data[adjusted_data[program_col] == program].groupby('Category')['Seats'].sum()
        
        row = {
            program_col: program,
            'Original_Total': int(orig_total),
            'Adjusted_Total': int(adj_total),
            'Difference': int(adj_total - orig_total)
        }
        
        # Add category details
        for cat in EXPECTED_PERCENTAGES.keys():
            orig_seats = orig_cats.get(cat, 0)
            adj_seats = adj_cats.get(cat, 0)
            row[f'{cat}_Original'] = int(orig_seats)
            row[f'{cat}_Adjusted'] = int(adj_seats)
            row[f'{cat}_Diff'] = int(adj_seats - orig_seats)
        
        summary.append(row)
    
    return pd.DataFrame(summary)

def calculate_percentage_df(data, program_col='Program'):
    """Calculate percentages for a dataset"""
    results = []
    
    programs = data[program_col].unique()
    
    for program in programs:
        program_data = data[data[program_col] == program]
        total = program_data['Seats'].sum()
        
        for category in EXPECTED_PERCENTAGES.keys():
            seats = program_data[program_data['Category'] == category]['Seats'].sum()
            actual_pct = (seats / total * 100) if total > 0 else 0
            expected_pct = EXPECTED_PERCENTAGES[category]
            deviation = actual_pct - expected_pct
            
            results.append({
                program_col: program,
                'Category': category,
                'Seats': int(seats),
                'Actual_Percent': round(actual_pct, 2),
                'Expected_Percent': round(expected_pct, 2),
                'Deviation': round(deviation, 2),
                'Within_Tolerance': abs(deviation) <= 2
            })
    
    return pd.DataFrame(results)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("🎯 Seat Allocation Adjuster")
    st.markdown("### Adjust seat allocation to match expected percentages")
    st.divider()
    
    # Initialize session state
    if 'original_data' not in st.session_state:
        st.session_state.original_data = None
    if 'adjusted_data' not in st.session_state:
        st.session_state.adjusted_data = None
    if 'adjustment_log' not in st.session_state:
        st.session_state.adjustment_log = None
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("📁 Data Input")
        input_type = st.radio(
            "Choose input method:",
            ["Sample Data", "Upload CSV", "Paste Data"]
        )
        
        data = None
        
        if input_type == "Sample Data":
            data = get_sample_data()
            st.success(f"✅ Loaded {len(data)} rows")
            st.dataframe(data.head(10), use_container_width=True)
        
        elif input_type == "Upload CSV":
            uploaded = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded:
                try:
                    data = pd.read_csv(uploaded)
                    st.success(f"✅ Loaded {len(data)} rows")
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
            st.subheader("⚙️ Adjustment Options")
            
            program_col = st.selectbox(
                "Group by:",
                ['Program', 'Specialty'],
                help="Select how to group data for adjustment"
            )
            
            tolerance = st.slider(
                "Tolerance (%)",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5,
                help="Maximum allowed deviation from expected percentage"
            )
            
            adjustment_method = st.radio(
                "Adjustment Method:",
                ['Simple Adjustment', 'Smart Adjustment (Recommended)']
            )
            
            if st.button("🔄 Adjust Seats", type="primary", use_container_width=True):
                with st.spinner("Adjusting seat allocation..."):
                    if adjustment_method == 'Simple Adjustment':
                        adjusted, log = adjust_seats_by_percentage(data, program_col, tolerance)
                    else:
                        adjusted, log = smart_adjust_seats(data, program_col)
                    
                    st.session_state.adjusted_data = adjusted
                    st.session_state.adjustment_log = log
                    st.success("✅ Seat allocation adjusted successfully!")
                    st.balloons()
    
    # Main content
    if st.session_state.adjusted_data is not None:
        original = st.session_state.original_data
        adjusted = st.session_state.adjusted_data
        
        # Tabs
        tabs = st.tabs([
            "📊 Summary",
            "📈 Program Comparison",
            "📉 Category Analysis",
            "📋 Detailed Changes",
            "✅ Validation"
        ])
        
        # Tab 1: Summary
        with tabs[0]:
            st.subheader("Adjustment Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                orig_total = original['Seats'].sum()
                adj_total = adjusted['Seats'].sum()
                st.metric("Total Seats", f"{adj_total:,}", f"{adj_total - orig_total:+d}")
            
            with col2:
                programs = len(original['Program'].unique())
                st.metric("Programs", programs)
            
            with col3:
                # Count categories that were adjusted
                adjusted_cats = st.session_state.adjustment_log['Category'].nunique() if not st.session_state.adjustment_log.empty else 0
                st.metric("Categories Adjusted", adjusted_cats)
            
            with col4:
                # Calculate improvement
                orig_pct_df = calculate_percentage_df(original, 'Program')
                adj_pct_df = calculate_percentage_df(adjusted, 'Program')
                
                orig_pass = orig_pct_df[orig_pct_df['Within_Tolerance']].shape[0]
                adj_pass = adj_pct_df[adj_pct_df['Within_Tolerance']].shape[0]
                improvement = adj_pass - orig_pass
                
                st.metric("Improvement", f"{improvement:+d}", delta_color="normal" if improvement > 0 else "inverse")
            
            # Show adjustment log
            if not st.session_state.adjustment_log.empty:
                st.markdown("#### Adjustment Log")
                st.dataframe(st.session_state.adjustment_log, use_container_width=True)
            
            # Summary table
            st.markdown("#### Program Summary")
            summary_df = create_summary_table(original, adjusted, 'Program')
            st.dataframe(summary_df, use_container_width=True)
        
        # Tab 2: Program Comparison
        with tabs[1]:
            st.subheader("Program Level Comparison")
            
            # Select category to compare
            category = st.selectbox(
                "Select Category to Compare:",
                list(EXPECTED_PERCENTAGES.keys())
            )
            
            # Show comparison chart
            fig = compare_distribution(original, adjusted, 'Program', category)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed program comparison
            st.markdown("#### Detailed Program Comparison")
            
            # Get percentage data
            orig_pct = calculate_percentage_df(original, 'Program')
            adj_pct = calculate_percentage_df(adjusted, 'Program')
            
            # Merge for comparison
            comp_df = orig_pct.merge(
                adj_pct,
                on=['Program', 'Category'],
                suffixes=('_Original', '_Adjusted')
            )
            
            comp_df['Status_Original'] = comp_df['Within_Tolerance_Original'].map({True: '✅', False: '⚠️'})
            comp_df['Status_Adjusted'] = comp_df['Within_Tolerance_Adjusted'].map({True: '✅', False: '⚠️'})
            
            st.dataframe(
                comp_df[['Program', 'Category', 
                        'Seats_Original', 'Seats_Adjusted',
                        'Actual_Percent_Original', 'Actual_Percent_Adjusted',
                        'Expected_Percent_Original', 'Deviation_Original', 'Deviation_Adjusted',
                        'Status_Original', 'Status_Adjusted']],
                column_config={
                    'Program': 'Program',
                    'Category': 'Category',
                    'Seats_Original': st.column_config.NumberColumn('Original Seats', format='%d'),
                    'Seats_Adjusted': st.column_config.NumberColumn('Adjusted Seats', format='%d'),
                    'Actual_Percent_Original': st.column_config.NumberColumn('Original %', format='%.2f%%'),
                    'Actual_Percent_Adjusted': st.column_config.NumberColumn('Adjusted %', format='%.2f%%'),
                    'Expected_Percent_Original': st.column_config.NumberColumn('Expected %', format='%.2f%%'),
                    'Deviation_Original': st.column_config.NumberColumn('Original Deviation', format='%.2f%%'),
                    'Deviation_Adjusted': st.column_config.NumberColumn('Adjusted Deviation', format='%.2f%%'),
                    'Status_Original': 'Original Status',
                    'Status_Adjusted': 'Adjusted Status'
                },
                use_container_width=True
            )
        
        # Tab 3: Category Analysis
        with tabs[2]:
            st.subheader("Category Level Analysis")
            
            # Overall category comparison
            orig_cat = original.groupby('Category')['Seats'].sum().reset_index()
            adj_cat = adjusted.groupby('Category')['Seats'].sum().reset_index()
            
            cat_comp = orig_cat.merge(adj_cat, on='Category', suffixes=('_Original', '_Adjusted'))
            cat_comp['Original_Pct'] = (cat_comp['Seats_Original'] / cat_comp['Seats_Original'].sum() * 100).round(2)
            cat_comp['Adjusted_Pct'] = (cat_comp['Seats_Adjusted'] / cat_comp['Seats_Adjusted'].sum() * 100).round(2)
            cat_comp['Expected_Pct'] = cat_comp['Category'].map(EXPECTED_PERCENTAGES).round(2)
            cat_comp['Original_Deviation'] = (cat_comp['Original_Pct'] - cat_comp['Expected_Pct']).round(2)
            cat_comp['Adjusted_Deviation'] = (cat_comp['Adjusted_Pct'] - cat_comp['Expected_Pct']).round(2)
            
            st.dataframe(
                cat_comp[['Category', 'Seats_Original', 'Seats_Adjusted', 
                         'Original_Pct', 'Adjusted_Pct', 'Expected_Pct',
                         'Original_Deviation', 'Adjusted_Deviation']],
                column_config={
                    'Category': 'Category',
                    'Seats_Original': st.column_config.NumberColumn('Original Seats', format='%d'),
                    'Seats_Adjusted': st.column_config.NumberColumn('Adjusted Seats', format='%d'),
                    'Original_Pct': st.column_config.NumberColumn('Original %', format='%.2f%%'),
                    'Adjusted_Pct': st.column_config.NumberColumn('Adjusted %', format='%.2f%%'),
                    'Expected_Pct': st.column_config.NumberColumn('Expected %', format='%.2f%%'),
                    'Original_Deviation': st.column_config.NumberColumn('Original Deviation', format='%.2f%%'),
                    'Adjusted_Deviation': st.column_config.NumberColumn('Adjusted Deviation', format='%.2f%%')
                },
                use_container_width=True
            )
            
            # Category chart - Original vs Adjusted vs Expected
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=cat_comp['Category'],
                y=cat_comp['Original_Pct'],
                name='Original',
                marker_color='#ff7f0e',
                text=cat_comp['Original_Pct'].apply(lambda x: f'{x:.1f}%'),
                textposition='outside'
            ))
            
            fig.add_trace(go.Bar(
                x=cat_comp['Category'],
                y=cat_comp['Adjusted_Pct'],
                name='Adjusted',
                marker_color='#2ca02c',
                text=cat_comp['Adjusted_Pct'].apply(lambda x: f'{x:.1f}%'),
                textposition='outside'
            ))
            
            fig.add_trace(go.Bar(
                x=cat_comp['Category'],
                y=cat_comp['Expected_Pct'],
                name='Expected',
                marker_color='#1f77b4',
                text=cat_comp['Expected_Pct'].apply(lambda x: f'{x:.1f}%'),
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
        
        # Tab 4: Detailed Changes
        with tabs[3]:
            st.subheader("📋 Detailed Changes")
            
            # Show all adjusted rows
            st.markdown("#### Adjusted Data")
            st.dataframe(adjusted, use_container_width=True)
            
            # Show changes
            st.markdown("#### Changes Made")
            
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
                st.info("No changes were made")
            
            # Download options
            st.markdown("#### Download Results")
            
            col1, col2 = st.columns(2)
            
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
        
        # Tab 5: Validation
        with tabs[4]:
            st.subheader("✅ Validation Report")
            
            # Validate original
            st.markdown("#### Original Data Validation")
            orig_validation = calculate_percentage_df(original, 'Program')
            orig_summary = orig_validation.groupby('Program')['Within_Tolerance'].all().reset_index()
            orig_summary['Status'] = orig_summary['Within_Tolerance'].map({True: '✅', False: '⚠️'})
            
            st.dataframe(orig_summary, use_container_width=True)
            
            # Validate adjusted
            st.markdown("#### Adjusted Data Validation")
            adj_validation = calculate_percentage_df(adjusted, 'Program')
            adj_summary = adj_validation.groupby('Program')['Within_Tolerance'].all().reset_index()
            adj_summary['Status'] = adj_summary['Within_Tolerance'].map({True: '✅', False: '⚠️'})
            
            st.dataframe(adj_summary, use_container_width=True)
            
            # Show programs that were fixed
            st.markdown("#### Programs Fixed")
            
            fixed = []
            for program in adj_summary['Program']:
                orig_status = orig_summary[orig_summary['Program'] == program]['Within_Tolerance'].values[0] if program in orig_summary['Program'].values else False
                adj_status = adj_summary[adj_summary['Program'] == program]['Within_Tolerance'].values[0]
                
                if not orig_status and adj_status:
                    fixed.append({
                        'Program': program,
                        'Status': '✅ Fixed'
                    })
                elif orig_status and adj_status:
                    fixed.append({
                        'Program': program,
                        'Status': '✅ Already OK'
                    })
                elif not orig_status and not adj_status:
                    fixed.append({
                        'Program': program,
                        'Status': '⚠️ Still Issues'
                    })
            
            if fixed:
                st.dataframe(pd.DataFrame(fixed), use_container_width=True)
            
            # Pass rate improvement
            orig_pass = orig_summary['Within_Tolerance'].sum()
            adj_pass = adj_summary['Within_Tolerance'].sum()
            total = len(orig_summary)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Pass Rate", f"{orig_pass}/{total}", f"{(orig_pass/total*100):.1f}%")
            with col2:
                st.metric("Adjusted Pass Rate", f"{adj_pass}/{total}", f"{(adj_pass/total*100):.1f}%")
            with col3:
                improvement = adj_pass - orig_pass
                st.metric("Improvement", f"{improvement:+d}", delta_color="normal" if improvement > 0 else "inverse")
    
    else:
        # Welcome message
        st.info("👈 Upload your data in the sidebar and click 'Adjust Seats'")
        
        st.markdown("""
        ### 🎯 What This Tool Does
        
        This tool **automatically adjusts seat allocations** to match the expected percentage distribution:
        
        #### 🔧 Adjustment Process
        
        1. **Analyzes** current seat distribution by program
        2. **Compares** actual percentages against expected (SM:50%, EW:10%, etc.)
        3. **Adjusts** seats to match expected percentages
        4. **Validates** that all programs now meet the criteria
        
        #### 📊 Example Issue Fixed
        
        **Before Adjustment (CS Program with 160 seats):**
        - EW: 23 seats (14.4%) ❌ (Expected: 10%)
        - EZ: 21 seats (13.1%) ❌ (Expected: 9%)
        - BH: 1 seat (0.6%) ❌ (Expected: 3%)
        
        **After Adjustment:**
        - EW: 16 seats (10%) ✅
        - EZ: 14.4 seats (9%) ✅
        - BH: 4.8 seats (3%) ✅
        
        #### 📁 Required Data Format
        
        Your CSV should have these columns:
        - **Program**: Program code (CS, EC, ME, etc.)
        - **Specialty**: Specialty name
        - **College**: College name
        - **Type**: Type (G, etc.)
        - **Category**: Seat category (SM, EW, EZ, etc.)
        - **Seats**: Number of seats
        """)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
