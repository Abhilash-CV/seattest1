import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Seat Allocation Percentage Analysis",
    page_icon="📊",
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
# DATA PROCESSING
# ============================================================================

def analyze_percentage_breakup(data):
    """
    Analyze percentage breakup for each category within each program/specialty
    """
    total_seats = data['Seats'].sum()
    
    # Group by Program and Category to get distribution
    program_category = data.groupby(['Program', 'Category'])['Seats'].sum().reset_index()
    
    # Get total per program
    program_totals = data.groupby('Program')['Seats'].sum().reset_index()
    program_totals.columns = ['Program', 'Program_Total']
    
    # Merge to get percentages
    program_category = program_category.merge(program_totals, on='Program')
    program_category['Actual_Percent'] = (program_category['Seats'] / program_category['Program_Total'] * 100).round(2)
    
    # Add expected percentage
    program_category['Expected_Percent'] = program_category['Category'].map(EXPECTED_PERCENTAGES).round(2)
    program_category['Percent_Difference'] = (program_category['Actual_Percent'] - program_category['Expected_Percent']).round(2)
    
    # Add expected seats
    program_category['Expected_Seats'] = (program_category['Program_Total'] * program_category['Expected_Percent'] / 100).round(0).astype(int)
    program_category['Seats_Difference'] = program_category['Seats'] - program_category['Expected_Seats']
    
    # Flag if percentage is within tolerance (±2%)
    program_category['Within_Tolerance'] = abs(program_category['Percent_Difference']) <= 2
    program_category['Status'] = program_category['Within_Tolerance'].map({True: '✅', False: '⚠️'})
    
    # Similar analysis by Specialty
    specialty_category = data.groupby(['Specialty', 'Category'])['Seats'].sum().reset_index()
    specialty_totals = data.groupby('Specialty')['Seats'].sum().reset_index()
    specialty_totals.columns = ['Specialty', 'Specialty_Total']
    specialty_category = specialty_category.merge(specialty_totals, on='Specialty')
    specialty_category['Actual_Percent'] = (specialty_category['Seats'] / specialty_category['Specialty_Total'] * 100).round(2)
    specialty_category['Expected_Percent'] = specialty_category['Category'].map(EXPECTED_PERCENTAGES).round(2)
    specialty_category['Percent_Difference'] = (specialty_category['Actual_Percent'] - specialty_category['Expected_Percent']).round(2)
    specialty_category['Expected_Seats'] = (specialty_category['Specialty_Total'] * specialty_category['Expected_Percent'] / 100).round(0).astype(int)
    specialty_category['Seats_Difference'] = specialty_category['Seats'] - specialty_category['Expected_Seats']
    specialty_category['Within_Tolerance'] = abs(specialty_category['Percent_Difference']) <= 2
    specialty_category['Status'] = specialty_category['Within_Tolerance'].map({True: '✅', False: '⚠️'})
    
    # Overall category summary
    overall_category = data.groupby('Category')['Seats'].sum().reset_index()
    overall_category['Expected'] = overall_category['Category'].map(SEAT_MATRIX)
    overall_category['Difference'] = overall_category['Seats'] - overall_category['Expected']
    overall_category['Actual_Percent'] = (overall_category['Seats'] / total_seats * 100).round(2)
    overall_category['Expected_Percent'] = overall_category['Category'].map(EXPECTED_PERCENTAGES).round(2)
    overall_category['Percent_Difference'] = (overall_category['Actual_Percent'] - overall_category['Expected_Percent']).round(2)
    overall_category['Status'] = overall_category.apply(
        lambda row: '✅' if abs(row['Percent_Difference']) <= 2 else '⚠️', 
        axis=1
    )
    
    return {
        'program_category': program_category,
        'specialty_category': specialty_category,
        'overall_category': overall_category,
        'program_totals': program_totals,
        'specialty_totals': specialty_totals,
        'total_seats': total_seats
    }

def get_sample_data():
    """Create sample data with your structure"""
    data = """Program,Specialty,College,Type,Category,Seats
CS,IDK,some_college,G,SM,2
CS,IDK,some_college,G,EW,1
CS,IDK,some_college,G,EZ,1
CS,NSS,some_college,G,SM,3
CS,NSS,some_college,G,EW,1
CS,NSS,some_college,G,EZ,1
CS,LBT,some_college,G,SM,4
CS,LBT,some_college,G,EW,1
CS,LBT,some_college,G,SC,1
CS,LBT,some_college,G,EZ,1
CS,LBT,some_college,G,MU,1
CS,LBT,some_college,G,BH,1
CS,MDL,some_college,G,SM,4
CS,MDL,some_college,G,EW,1
CS,MDL,some_college,G,SC,1
CS,MDL,some_college,G,EZ,1
CS,MDL,some_college,G,MU,1
CS,CHN,some_college,G,SM,5
CS,CHN,some_college,G,EW,1
CS,CHN,some_college,G,SC,1
CS,CHN,some_college,G,EZ,1
CS,CHN,some_college,G,MU,1"""
    
    df = pd.read_csv(StringIO(data))
    return df

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_percentage_comparison_chart(data, group_col):
    """Create bar chart comparing actual vs expected percentages"""
    fig = go.Figure()
    
    # Get unique categories
    categories = data['Category'].unique()
    
    for cat in categories:
        cat_data = data[data['Category'] == cat]
        
        fig.add_trace(go.Bar(
            name=f'{cat} (Expected)',
            x=cat_data[group_col],
            y=cat_data['Expected_Percent'],
            marker_color='lightgray',
            opacity=0.5,
            legendgroup=cat,
            showlegend=True
        ))
        
        fig.add_trace(go.Bar(
            name=f'{cat} (Actual)',
            x=cat_data[group_col],
            y=cat_data['Actual_Percent'],
            marker_color=px.colors.qualitative.Set3[list(categories).index(cat) % len(px.colors.qualitative.Set3)],
            legendgroup=cat,
            showlegend=True
        ))
    
    fig.update_layout(
        title='Actual vs Expected Percentage Distribution',
        xaxis_title=group_col,
        yaxis_title='Percentage (%)',
        barmode='group',
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

def create_percentage_deviation_chart(data, group_col):
    """Create chart showing percentage deviations"""
    # Pivot for heatmap
    pivot_data = data.pivot_table(
        index=group_col,
        columns='Category',
        values='Percent_Difference',
        fill_value=0
    )
    
    fig = px.imshow(
        pivot_data,
        title=f'Percentage Deviation from Expected ({group_col} level)',
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

def create_status_summary(data, group_col):
    """Create status summary with counts"""
    status_summary = data.groupby([group_col, 'Status']).size().reset_index(name='Count')
    status_pivot = status_summary.pivot(index=group_col, columns='Status', values='Count').fillna(0)
    
    # Add total
    status_pivot['Total'] = status_pivot.sum(axis=1)
    status_pivot['Pass_Rate'] = (status_pivot.get('✅', 0) / status_pivot['Total'] * 100).round(1)
    
    return status_pivot

def create_issue_report(data, group_col, threshold=2):
    """Create report of issues where percentage difference exceeds threshold"""
    issues = data[abs(data['Percent_Difference']) > threshold].copy()
    issues = issues.sort_values('Percent_Difference', ascending=False)
    
    if not issues.empty:
        issues['Issue_Type'] = issues['Percent_Difference'].apply(
            lambda x: 'Over-allocated' if x > 0 else 'Under-allocated'
        )
        issues['Severity'] = issues['Percent_Difference'].apply(
            lambda x: 'High' if abs(x) > 5 else 'Medium' if abs(x) > 3 else 'Low'
        )
    
    return issues

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("📊 Seat Allocation Percentage Analysis")
    st.markdown("### Validate percentage distribution against expected seat matrix")
    st.divider()
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    
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
            st.session_state.data = data
            
            if st.button("🔍 Analyze Percentages", type="primary", use_container_width=True):
                with st.spinner("Analyzing percentage distribution..."):
                    results = analyze_percentage_breakup(data)
                    st.session_state.analysis_results = results
                    st.success("✅ Analysis complete!")
                    st.balloons()
        
        # Show expected percentages
        with st.expander("📊 Expected Percentages"):
            expected_df = pd.DataFrame({
                'Category': list(EXPECTED_PERCENTAGES.keys()),
                'Expected_Seats': list(SEAT_MATRIX.values()),
                'Expected_Percent': list(EXPECTED_PERCENTAGES.values())
            })
            st.dataframe(expected_df, use_container_width=True)
    
    # Main content
    if st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Seats", results['total_seats'])
        with col2:
            total_programs = len(results['program_totals'])
            st.metric("Programs", total_programs)
        with col3:
            total_specialties = len(results['specialty_totals'])
            st.metric("Specialties", total_specialties)
        with col4:
            # Calculate overall pass rate
            program_status = results['program_category'].groupby('Program')['Within_Tolerance'].all()
            pass_rate = (program_status.sum() / len(program_status) * 100) if len(program_status) > 0 else 0
            st.metric("Program Pass Rate", f"{pass_rate:.1f}%")
        
        # Tabs
        tabs = st.tabs([
            "📊 Overall Analysis",
            "📈 Program Level",
            "🏛️ Specialty Level",
            "⚠️ Issues Report",
            "📋 Detailed Data"
        ])
        
        # Tab 1: Overall Analysis
        with tabs[0]:
            st.subheader("Overall Category Distribution")
            
            # Overall category summary
            overall = results['overall_category']
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                total_expected = sum(SEAT_MATRIX.values())
                diff = results['total_seats'] - total_expected
                st.metric("Total vs Expected", f"{diff:+d}", delta_color="inverse")
            with col2:
                passing = overall[overall['Status'] == '✅'].shape[0]
                st.metric("Categories Passing", f"{passing}/{len(overall)}")
            with col3:
                avg_deviation = overall['Percent_Difference'].abs().mean()
                st.metric("Avg Deviation", f"{avg_deviation:.2f}%")
            
            # Display overall summary
            st.dataframe(
                overall,
                column_config={
                    'Category': 'Category',
                    'Seats': st.column_config.NumberColumn('Actual Seats', format='%d'),
                    'Expected': st.column_config.NumberColumn('Expected Seats', format='%d'),
                    'Difference': st.column_config.NumberColumn('Seats Diff', format='%d'),
                    'Actual_Percent': st.column_config.NumberColumn('Actual %', format='%.2f%%'),
                    'Expected_Percent': st.column_config.NumberColumn('Expected %', format='%.2f%%'),
                    'Percent_Difference': st.column_config.NumberColumn('Deviation %', format='%.2f%%'),
                    'Status': 'Status'
                },
                use_container_width=True
            )
            
            # Category chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=overall['Category'],
                y=overall['Actual_Percent'],
                name='Actual',
                marker_color='#2ca02c',
                text=overall['Actual_Percent'].apply(lambda x: f'{x:.1f}%'),
                textposition='outside'
            ))
            
            fig.add_trace(go.Bar(
                x=overall['Category'],
                y=overall['Expected_Percent'],
                name='Expected',
                marker_color='#1f77b4',
                text=overall['Expected_Percent'].apply(lambda x: f'{x:.1f}%'),
                textposition='outside'
            ))
            
            fig.update_layout(
                title='Overall Category Distribution - Actual vs Expected',
                xaxis_title='Category',
                yaxis_title='Percentage (%)',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 2: Program Level
        with tabs[1]:
            st.subheader("Program Level Percentage Analysis")
            
            # Program selector
            programs = results['program_totals']['Program'].tolist()
            selected_program = st.selectbox("Select Program", programs)
            
            # Filter data for selected program
            program_data = results['program_category'][results['program_category']['Program'] == selected_program]
            
            if not program_data.empty:
                # Show program summary
                prog_total = results['program_totals'][results['program_totals']['Program'] == selected_program]['Program_Total'].values[0]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Seats", prog_total)
                with col2:
                    passing = program_data[program_data['Within_Tolerance']].shape[0]
                    st.metric("Categories Passing", f"{passing}/{len(program_data)}")
                with col3:
                    avg_dev = program_data['Percent_Difference'].abs().mean()
                    st.metric("Avg Deviation", f"{avg_dev:.2f}%")
                
                # Display program data
                st.dataframe(
                    program_data[['Category', 'Seats', 'Expected_Seats', 'Seats_Difference', 
                                  'Actual_Percent', 'Expected_Percent', 'Percent_Difference', 'Status']],
                    column_config={
                        'Category': 'Category',
                        'Seats': st.column_config.NumberColumn('Actual Seats', format='%d'),
                        'Expected_Seats': st.column_config.NumberColumn('Expected Seats', format='%d'),
                        'Seats_Difference': st.column_config.NumberColumn('Seats Diff', format='%d'),
                        'Actual_Percent': st.column_config.NumberColumn('Actual %', format='%.2f%%'),
                        'Expected_Percent': st.column_config.NumberColumn('Expected %', format='%.2f%%'),
                        'Percent_Difference': st.column_config.NumberColumn('Deviation %', format='%.2f%%'),
                        'Status': 'Status'
                    },
                    use_container_width=True
                )
                
                # Chart for selected program
                fig = create_percentage_comparison_chart(program_data, 'Category')
                st.plotly_chart(fig, use_container_width=True)
            
            # All programs summary
            st.markdown("#### All Programs Summary")
            program_status_summary = create_status_summary(results['program_category'], 'Program')
            st.dataframe(program_status_summary, use_container_width=True)
            
            # Program heatmap
            fig = create_percentage_deviation_chart(results['program_category'], 'Program')
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 3: Specialty Level
        with tabs[2]:
            st.subheader("Specialty Level Percentage Analysis")
            
            # Specialty selector
            specialties = results['specialty_totals']['Specialty'].tolist()
            selected_specialty = st.selectbox("Select Specialty", specialties)
            
            # Filter data for selected specialty
            specialty_data = results['specialty_category'][results['specialty_category']['Specialty'] == selected_specialty]
            
            if not specialty_data.empty:
                spec_total = results['specialty_totals'][results['specialty_totals']['Specialty'] == selected_specialty]['Specialty_Total'].values[0]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Seats", spec_total)
                with col2:
                    passing = specialty_data[specialty_data['Within_Tolerance']].shape[0]
                    st.metric("Categories Passing", f"{passing}/{len(specialty_data)}")
                with col3:
                    avg_dev = specialty_data['Percent_Difference'].abs().mean()
                    st.metric("Avg Deviation", f"{avg_dev:.2f}%")
                
                st.dataframe(
                    specialty_data[['Category', 'Seats', 'Expected_Seats', 'Seats_Difference',
                                  'Actual_Percent', 'Expected_Percent', 'Percent_Difference', 'Status']],
                    column_config={
                        'Category': 'Category',
                        'Seats': st.column_config.NumberColumn('Actual Seats', format='%d'),
                        'Expected_Seats': st.column_config.NumberColumn('Expected Seats', format='%d'),
                        'Seats_Difference': st.column_config.NumberColumn('Seats Diff', format='%d'),
                        'Actual_Percent': st.column_config.NumberColumn('Actual %', format='%.2f%%'),
                        'Expected_Percent': st.column_config.NumberColumn('Expected %', format='%.2f%%'),
                        'Percent_Difference': st.column_config.NumberColumn('Deviation %', format='%.2f%%'),
                        'Status': 'Status'
                    },
                    use_container_width=True
                )
                
                fig = create_percentage_comparison_chart(specialty_data, 'Category')
                st.plotly_chart(fig, use_container_width=True)
            
            # All specialties summary
            st.markdown("#### All Specialties Summary")
            specialty_status_summary = create_status_summary(results['specialty_category'], 'Specialty')
            st.dataframe(specialty_status_summary, use_container_width=True)
            
            # Specialty heatmap
            fig = create_percentage_deviation_chart(results['specialty_category'], 'Specialty')
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 4: Issues Report
        with tabs[3]:
            st.subheader("⚠️ Issues Report")
            st.markdown("Categories where percentage deviation exceeds 2%")
            
            # Program level issues
            st.markdown("#### Program Level Issues")
            program_issues = create_issue_report(results['program_category'], 'Program')
            
            if not program_issues.empty:
                st.dataframe(
                    program_issues[['Program', 'Category', 'Seats', 'Expected_Seats', 
                                   'Actual_Percent', 'Expected_Percent', 'Percent_Difference', 
                                   'Issue_Type', 'Severity']],
                    column_config={
                        'Program': 'Program',
                        'Category': 'Category',
                        'Seats': st.column_config.NumberColumn('Actual', format='%d'),
                        'Expected_Seats': st.column_config.NumberColumn('Expected', format='%d'),
                        'Actual_Percent': st.column_config.NumberColumn('Actual %', format='%.2f%%'),
                        'Expected_Percent': st.column_config.NumberColumn('Expected %', format='%.2f%%'),
                        'Percent_Difference': st.column_config.NumberColumn('Deviation', format='%.2f%%'),
                        'Issue_Type': 'Issue',
                        'Severity': 'Severity'
                    },
                    use_container_width=True
                )
                
                # Color-coded issues by severity
                severity_colors = {'High': '#ff4444', 'Medium': '#ffaa00', 'Low': '#ffdd00'}
                
                fig = go.Figure()
                for severity, color in severity_colors.items():
                    severity_data = program_issues[program_issues['Severity'] == severity]
                    if not severity_data.empty:
                        fig.add_trace(go.Bar(
                            x=severity_data['Category'],
                            y=severity_data['Percent_Difference'],
                            name=severity,
                            marker_color=color,
                            text=severity_data['Program'] + ' (' + severity_data['Percent_Difference'].apply(lambda x: f'{x:.1f}%') + ')',
                            textposition='outside'
                        ))
                
                fig.update_layout(
                    title='Program Level Issues by Severity',
                    xaxis_title='Category',
                    yaxis_title='Percentage Deviation (%)',
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No issues found at program level!")
            
            # Specialty level issues
            st.markdown("#### Specialty Level Issues")
            specialty_issues = create_issue_report(results['specialty_category'], 'Specialty')
            
            if not specialty_issues.empty:
                st.dataframe(
                    specialty_issues[['Specialty', 'Category', 'Seats', 'Expected_Seats',
                                    'Actual_Percent', 'Expected_Percent', 'Percent_Difference',
                                    'Issue_Type', 'Severity']],
                    column_config={
                        'Specialty': 'Specialty',
                        'Category': 'Category',
                        'Seats': st.column_config.NumberColumn('Actual', format='%d'),
                        'Expected_Seats': st.column_config.NumberColumn('Expected', format='%d'),
                        'Actual_Percent': st.column_config.NumberColumn('Actual %', format='%.2f%%'),
                        'Expected_Percent': st.column_config.NumberColumn('Expected %', format='%.2f%%'),
                        'Percent_Difference': st.column_config.NumberColumn('Deviation', format='%.2f%%'),
                        'Issue_Type': 'Issue',
                        'Severity': 'Severity'
                    },
                    use_container_width=True
                )
            else:
                st.success("✅ No issues found at specialty level!")
        
        # Tab 5: Detailed Data
        with tabs[4]:
            st.subheader("📋 Detailed Data")
            
            # Full data view
            st.markdown("#### All Data")
            st.dataframe(st.session_state.data, use_container_width=True)
            
            # Download options
            st.markdown("#### Download Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Program level analysis download
                csv_program = results['program_category'].to_csv(index=False)
                st.download_button(
                    "📥 Download Program Level Analysis",
                    csv_program,
                    f"program_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Specialty level analysis download
                csv_specialty = results['specialty_category'].to_csv(index=False)
                st.download_button(
                    "📥 Download Specialty Level Analysis",
                    csv_specialty,
                    f"specialty_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
    
    else:
        # Welcome message
        st.info("👈 Upload your data in the sidebar and click 'Analyze Percentages'")
        
        st.markdown("""
        ### 📊 What This Tool Does
        
        This tool analyzes seat allocation percentages to ensure they match the expected distribution:
        
        #### 🎯 Expected Distribution (Total 100 seats)
        - **SM**: 50% (50 seats)
        - **EW**: 10% (10 seats)
        - **EZ**: 9% (9 seats)
        - **MU**: 8% (8 seats)
        - **SC**: 8% (8 seats)
        - **BH**: 3% (3 seats)
        - **LA**: 3% (3 seats)
        - **DV**: 2% (2 seats)
        - **VK**: 2% (2 seats)
        - **ST**: 2% (2 seats)
        - **KN**: 1% (1 seat)
        - **BX**: 1% (1 seat)
        - **KU**: 1% (1 seat)
        
        #### 📋 What It Checks
        
        1. **Program Level**: For each program (CS, EC, EE, etc.), checks if the percentage distribution matches the expected percentages
        2. **Specialty Level**: For each specialty, checks if the percentage distribution matches the expected percentages
        3. **Identifies Issues**: Flags categories where percentage deviation exceeds 2%
        
        #### 📁 Required Data Format
        
        Your CSV should have these columns:
        - **Program**: Program code (CS, EC, ME, etc.)
        - **Specialty**: Specialty name
        - **College**: College name
        - **Type**: Type (G, etc.)
        - **Category**: Seat category (SM, EW, EZ, etc.)
        - **Seats**: Number of seats
        """)
        
        # Show example of what the analysis reveals
        st.markdown("### 🔍 Example Issues Detected")
        
        example_data = pd.DataFrame({
            'Program': ['CS', 'CS', 'CS', 'CS', 'CS', 'CS', 'CS'],
            'Category': ['SM', 'EW', 'EZ', 'MU', 'SC', 'BH', 'ST'],
            'Actual_Seats': [80, 23, 21, 15, 14, 1, 2],
            'Expected_Seats': [80, 16, 14.4, 12.8, 12.8, 4.8, 3.2],
            'Actual_Percent': [50.0, 14.4, 13.1, 9.4, 8.8, 0.6, 1.3],
            'Expected_Percent': [50.0, 10.0, 9.0, 8.0, 8.0, 3.0, 2.0],
            'Deviation': [0.0, 4.4, 4.1, 1.4, 0.8, -2.4, -0.7],
            'Status': ['✅', '⚠️', '⚠️', '✅', '✅', '⚠️', '✅']
        })
        
        st.dataframe(
            example_data,
            column_config={
                'Program': 'Program',
                'Category': 'Category',
                'Actual_Seats': st.column_config.NumberColumn('Actual Seats', format='%d'),
                'Expected_Seats': st.column_config.NumberColumn('Expected Seats', format='%.1f'),
                'Actual_Percent': st.column_config.NumberColumn('Actual %', format='%.1f%%'),
                'Expected_Percent': st.column_config.NumberColumn('Expected %', format='%.1f%%'),
                'Deviation': st.column_config.NumberColumn('Deviation', format='%.1f%%'),
                'Status': 'Status'
            },
            use_container_width=True
        )
        
        st.warning("""
        ⚠️ **Issues Found:**
        - **EW**: 23 seats (14.4%) vs expected 10% (+4.4% deviation) - **Over-allocated**
        - **EZ**: 21 seats (13.1%) vs expected 9% (+4.1% deviation) - **Over-allocated**
        - **BH**: 1 seat (0.6%) vs expected 3% (-2.4% deviation) - **Under-allocated**
        """)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
