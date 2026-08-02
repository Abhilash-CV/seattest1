import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Seat Allocation System",
    page_icon="🎓",
    layout="wide"
)

# ============================================================================
# SEAT ALLOCATOR CLASS
# ============================================================================

class SeatAllocator:
    def __init__(self, college_data, seat_matrix=None):
        """
        Initialize with college data and optional seat matrix
        
        Args:
            college_data: DataFrame with columns ['Program', 'Specialty', 'College', 'Seats', 'Type']
            seat_matrix: Dictionary of category to seat allocation (optional)
        """
        self.college_data = college_data.copy()
        
        # If seat_matrix not provided, create from data
        if seat_matrix is None:
            # Get all unique categories from data
            categories = self.college_data['Program'].unique()
            # Calculate seats per category based on total seats
            category_counts = self.college_data.groupby('Program')['Seats'].sum().to_dict()
            self.seat_matrix = category_counts
        else:
            self.seat_matrix = seat_matrix
        
        self.categories = list(self.seat_matrix.keys())
        self.total_seats = sum(self.seat_matrix.values())
        
        # Store total seats from input for validation
        self.input_total_seats = self.college_data['Seats'].sum()
    
    def hamilton_rounding(self, proportions, total_seats):
        """Hamilton (largest remainder) method"""
        proportions = np.array(proportions)
        if len(proportions) == 0 or total_seats <= 0:
            return np.zeros(len(proportions), dtype=int)
        
        # Calculate initial seats
        initial_seats = np.floor(proportions * total_seats).astype(int)
        remainder = proportions * total_seats - initial_seats
        
        # Distribute remaining seats
        remaining = total_seats - initial_seats.sum()
        if remaining > 0:
            idx = np.argsort(remainder)[::-1][:int(remaining)]
            initial_seats[idx] += 1
        
        return initial_seats
    
    def biproportional_allocation(self, row_margins, col_margins, max_iter=100):
        """Biproportional allocation"""
        n_rows, n_cols = len(row_margins), len(col_margins)
        if n_rows == 0 or n_cols == 0:
            return np.zeros((n_rows, n_cols), dtype=int)
        
        # Initialize matrix
        matrix = np.ones((n_rows, n_cols))
        
        # Iterative proportional fitting
        for _ in range(max_iter):
            # Scale rows
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            matrix = matrix * (row_margins.reshape(-1, 1) / row_sums)
            
            # Scale columns
            col_sums = matrix.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1
            matrix = matrix * (col_margins.reshape(1, -1) / col_sums)
        
        # Round to integers
        rounded = np.zeros((n_rows, n_cols), dtype=int)
        for i in range(n_rows):
            if matrix[i, :].sum() > 0:
                rounded[i, :] = self.hamilton_rounding(
                    matrix[i, :] / matrix[i, :].sum(),
                    int(row_margins[i])
                )
        
        return rounded
    
    def calculate_allocations(self):
        """Main allocation function"""
        colleges = self.college_data['College'].unique()
        specialties = self.college_data['Specialty'].unique()
        
        # Create mapping
        seat_map = {}
        for _, row in self.college_data.iterrows():
            key = (row['College'], row['Specialty'])
            seat_map[key] = row['Seats']
        
        n_colleges = len(colleges)
        n_specialties = len(specialties)
        
        # Calculate margins
        college_seats = np.array([
            sum(seat_map.get((c, s), 0) for s in specialties)
            for c in colleges
        ])
        
        specialty_seats = np.array([
            sum(seat_map.get((c, s), 0) for c in colleges)
            for s in specialties
        ])
        
        # Total seats from data
        total_data_seats = sum(college_seats)
        
        # Calculate category shares based on seat matrix
        category_shares = np.array(list(self.seat_matrix.values())) / self.total_seats
        # Scale to match total data seats
        category_shares = category_shares * total_data_seats / category_shares.sum()
        
        results = {'hamilton': {}, 'biproportional': {}}
        
        for idx, category in enumerate(self.categories):
            # Calculate category total based on proportion of total
            cat_total = int(self.seat_matrix[category])
            
            # Get proportions for Hamilton
            props = []
            for c in colleges:
                for s in specialties:
                    props.append(seat_map.get((c, s), 0))
            props = np.array(props)
            
            if props.sum() > 0:
                ham_alloc = self.hamilton_rounding(props / props.sum(), cat_total)
            else:
                ham_alloc = np.zeros(len(props))
            
            ham_matrix = ham_alloc.reshape(n_colleges, n_specialties)
            
            # Biproportional
            if n_colleges > 0 and n_specialties > 0:
                row_marg = college_seats * (cat_total / total_data_seats)
                col_marg = specialty_seats * (cat_total / total_data_seats)
                
                # Ensure margins sum correctly
                row_marg = np.round(row_marg).astype(int)
                col_marg = np.round(col_marg).astype(int)
                
                # Adjust margins to match category total
                while row_marg.sum() != cat_total and cat_total > 0:
                    if row_marg.sum() < cat_total:
                        row_marg[np.argmax(college_seats)] += 1
                    else:
                        row_marg[np.argmin(college_seats)] -= 1
                
                while col_marg.sum() != cat_total and cat_total > 0:
                    if col_marg.sum() < cat_total:
                        col_marg[np.argmax(specialty_seats)] += 1
                    else:
                        col_marg[np.argmin(specialty_seats)] -= 1
                
                bipro_matrix = self.biproportional_allocation(row_marg, col_marg)
            else:
                bipro_matrix = np.zeros((n_colleges, n_specialties), dtype=int)
            
            results['hamilton'][category] = ham_matrix
            results['biproportional'][category] = bipro_matrix
        
        return results, colleges, specialties

# ============================================================================
# UI FUNCTIONS
# ============================================================================

def get_sample_data():
    """Return sample data"""
    return pd.DataFrame({
        'Program': ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E'],
        'Specialty': ['DS', 'DS', 'DS', 'AI', 'AI', 'CS', 'CS', 'CS', 'ML', 'ML', 'DS', 'AI'],
        'College': ['CDI', 'CDP', 'CDT', 'CDI', 'CDP', 'CDT', 'CDI', 'CDP', 'CDT', 'CDI', 'CDP', 'CDT'],
        'Seats': [1, 1, 4, 2, 3, 1, 2, 1, 2, 1, 1, 2],
        'Type': ['G'] * 12
    })

def detect_seat_matrix_from_data(data):
    """Auto-detect seat matrix from data"""
    # Group by Program to get total seats per category
    category_seats = data.groupby('Program')['Seats'].sum().to_dict()
    return category_seats

def display_results(results, colleges, specialties, seat_matrix):
    """Display allocation results"""
    
    # Summary cards
    total_seats = sum(seat_matrix.values())
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Seats", total_seats)
    with col2:
        st.metric("Categories", len(seat_matrix))
    with col3:
        st.metric("Colleges", len(colleges))
    with col4:
        st.metric("Specialties", len(specialties))
    
    # Tabs
    tabs = st.tabs(["📊 Hamilton Method", "📈 Biproportional Method", "📉 Comparison", "📋 Download"])
    
    # Tab 1: Hamilton
    with tabs[0]:
        st.subheader("Hamilton Rounding Method")
        st.info("Allocates seats using the largest remainder method")
        
        # Summary
        summary = []
        for cat, matrix in results['hamilton'].items():
            total = matrix.sum()
            expected = seat_matrix.get(cat, 0)
            summary.append({
                'Category': cat,
                'Allocated': int(total),
                'Expected': int(expected),
                'Diff': int(total - expected),
                'Accuracy': f"{(total/expected*100):.1f}%" if expected > 0 else "N/A"
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True)
        
        # Detailed matrices
        with st.expander("📋 View Detailed Matrices"):
            for cat, matrix in results['hamilton'].items():
                st.write(f"**{cat}**")
                df = pd.DataFrame(matrix, index=colleges, columns=specialties)
                st.dataframe(df, use_container_width=True)
                st.markdown("---")
    
    # Tab 2: Biproportional
    with tabs[1]:
        st.subheader("Biproportional Method")
        st.info("Balances allocation across both colleges and specialties")
        
        # Summary
        summary = []
        for cat, matrix in results['biproportional'].items():
            total = matrix.sum()
            expected = seat_matrix.get(cat, 0)
            summary.append({
                'Category': cat,
                'Allocated': int(total),
                'Expected': int(expected),
                'Diff': int(total - expected),
                'Accuracy': f"{(total/expected*100):.1f}%" if expected > 0 else "N/A"
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True)
        
        # Detailed matrices
        with st.expander("📋 View Detailed Matrices"):
            for cat, matrix in results['biproportional'].items():
                st.write(f"**{cat}**")
                df = pd.DataFrame(matrix, index=colleges, columns=specialties)
                st.dataframe(df, use_container_width=True)
                st.markdown("---")
    
    # Tab 3: Comparison
    with tabs[2]:
        st.subheader("Method Comparison")
        
        # Comparison chart
        comp_data = []
        for cat in seat_matrix.keys():
            comp_data.append({
                'Category': cat,
                'Expected': int(seat_matrix[cat]),
                'Hamilton': int(results['hamilton'][cat].sum()),
                'Biproportional': int(results['biproportional'][cat].sum())
            })
        
        df_comp = pd.DataFrame(comp_data)
        st.dataframe(df_comp, use_container_width=True)
        
        # Bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_comp['Category'],
            y=df_comp['Expected'],
            name='Expected',
            marker_color='#1f77b4'
        ))
        fig.add_trace(go.Bar(
            x=df_comp['Category'],
            y=df_comp['Hamilton'],
            name='Hamilton',
            marker_color='#2ca02c'
        ))
        fig.add_trace(go.Bar(
            x=df_comp['Category'],
            y=df_comp['Biproportional'],
            name='Biproportional',
            marker_color='#ff7f0e'
        ))
        fig.update_layout(
            title='Seat Allocation Comparison',
            barmode='group',
            height=500,
            xaxis_title='Category',
            yaxis_title='Number of Seats'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Pie charts showing distribution
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(
                df_comp,
                values='Hamilton',
                names='Category',
                title='Hamilton Distribution'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_pie = px.pie(
                df_comp,
                values='Biproportional',
                names='Category',
                title='Biproportional Distribution'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # Tab 4: Download
    with tabs[3]:
        st.subheader("📥 Download Results")
        
        # Prepare download data
        download_data = []
        for method in ['hamilton', 'biproportional']:
            for cat in seat_matrix.keys():
                matrix = results[method][cat]
                for i, c in enumerate(colleges):
                    for j, s in enumerate(specialties):
                        if matrix[i, j] > 0:
                            download_data.append({
                                'Method': method.capitalize(),
                                'Category': cat,
                                'College': c,
                                'Specialty': s,
                                'Seats': int(matrix[i, j])
                            })
        
        df_download = pd.DataFrame(download_data)
        
        col1, col2 = st.columns(2)
        with col1:
            csv = df_download.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                f"allocations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        with col2:
            json_str = json.dumps({
                'timestamp': datetime.now().isoformat(),
                'seat_matrix': seat_matrix,
                'total_seats': sum(seat_matrix.values()),
                'colleges': list(colleges),
                'specialties': list(specialties),
                'results': {
                    m: {c: results[m][c].tolist() for c in results[m].keys()}
                    for m in ['hamilton', 'biproportional']
                }
            }, indent=2)
            st.download_button(
                "📥 Download JSON",
                json_str,
                f"allocations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                use_container_width=True
            )
        
        # Show total summary
        st.subheader("Summary Statistics")
        total_summary = []
        for method in ['hamilton', 'biproportional']:
            total = sum(results[method][cat].sum() for cat in seat_matrix.keys())
            total_summary.append({
                'Method': method.capitalize(),
                'Total Allocated': int(total),
                'Total Expected': sum(seat_matrix.values()),
                'Difference': int(total - sum(seat_matrix.values()))
            })
        st.dataframe(pd.DataFrame(total_summary), use_container_width=True)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Initialize session state
    if 'calculated' not in st.session_state:
        st.session_state.calculated = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'colleges' not in st.session_state:
        st.session_state.colleges = None
    if 'specialties' not in st.session_state:
        st.session_state.specialties = None
    if 'seat_matrix' not in st.session_state:
        st.session_state.seat_matrix = None
    
    # Header
    st.title("🎓 Seat Allocation System")
    st.markdown("### Hamilton Rounding vs Biproportional Method")
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data input
        st.subheader("📁 Data Input")
        input_type = st.radio(
            "Choose input method:",
            ["Sample Data", "Upload CSV", "Manual Entry"]
        )
        
        data = None
        
        if input_type == "Sample Data":
            data = get_sample_data()
            st.success("✅ Using sample data")
            st.dataframe(data, use_container_width=True)
        
        elif input_type == "Upload CSV":
            uploaded = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded:
                try:
                    data = pd.read_csv(uploaded)
                    st.success(f"✅ Loaded {len(data)} rows")
                    st.dataframe(data, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        else:  # Manual Entry
            st.info("✏️ Enter data manually")
            n = st.number_input("Number of rows", min_value=1, max_value=50, value=3)
            
            rows = []
            for i in range(n):
                cols = st.columns(5)
                with cols[0]:
                    prog = st.text_input(f"Program {i+1}", f"P{i+1}", key=f"p{i}")
                with cols[1]:
                    spec = st.text_input(f"Specialty {i+1}", f"S{i+1}", key=f"s{i}")
                with cols[2]:
                    col = st.text_input(f"College {i+1}", f"C{i+1}", key=f"c{i}")
                with cols[3]:
                    seat = st.number_input(f"Seats {i+1}", 1, 1000, 1, key=f"st{i}")
                with cols[4]:
                    typ = st.text_input(f"Type {i+1}", "G", key=f"t{i}")
                rows.append({'Program': prog, 'Specialty': spec, 'College': col, 'Seats': seat, 'Type': typ})
                st.markdown("---")
            
            if rows:
                data = pd.DataFrame(rows)
        
        # Seat Matrix Configuration
        if data is not None and not data.empty:
            st.subheader("🎯 Seat Matrix")
            
            # Auto-detect or manual
            matrix_option = st.radio(
                "Seat matrix configuration:",
                ["Auto-detect from data", "Manual configuration"]
            )
            
            if matrix_option == "Auto-detect from data":
                seat_matrix = detect_seat_matrix_from_data(data)
                st.success(f"Detected {len(seat_matrix)} categories")
                st.dataframe(
                    pd.DataFrame(list(seat_matrix.items()), columns=['Category', 'Seats']),
                    use_container_width=True
                )
            else:
                # Manual configuration
                categories = data['Program'].unique()
                seat_matrix = {}
                st.info("Enter seats per category")
                cols = st.columns(3)
                for i, cat in enumerate(categories):
                    with cols[i % 3]:
                        seats = st.number_input(
                            f"{cat} seats",
                            min_value=1,
                            value=int(data[data['Program'] == cat]['Seats'].sum()),
                            key=f"matrix_{cat}"
                        )
                        seat_matrix[cat] = seats
                
                st.dataframe(
                    pd.DataFrame(list(seat_matrix.items()), columns=['Category', 'Seats']),
                    use_container_width=True
                )
            
            # Calculate button
            if st.button("🚀 Calculate Allocations", type="primary", use_container_width=True):
                if data is not None and not data.empty and seat_matrix is not None:
                    try:
                        with st.spinner("🔄 Calculating allocations..."):
                            allocator = SeatAllocator(data, seat_matrix)
                            results, colleges, specialties = allocator.calculate_allocations()
                            
                            st.session_state.results = results
                            st.session_state.colleges = colleges
                            st.session_state.specialties = specialties
                            st.session_state.seat_matrix = seat_matrix
                            st.session_state.calculated = True
                            
                        st.success("✅ Allocations calculated successfully!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        st.exception(e)
                else:
                    st.warning("⚠️ Please provide valid data")
    
    # Main content
    if st.session_state.calculated and st.session_state.results is not None:
        display_results(
            st.session_state.results,
            st.session_state.colleges,
            st.session_state.specialties,
            st.session_state.seat_matrix
        )
    else:
        st.info("👈 Configure your data in the sidebar and click 'Calculate Allocations'")
        
        # Show how it works
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📊 Hamilton Method
            - **Largest remainder method**
            - Simple and transparent
            - Good for proportional allocation
            - Easy to understand and explain
            """)
        
        with col2:
            st.markdown("""
            ### ⚖️ Biproportional Method
            - **Iterative proportional fitting**
            - Balances row and column constraints
            - More precise for complex scenarios
            - Handles multiple constraints
            """)
        
        # Feature highlights
        st.subheader("✨ Features")
        features = st.columns(4)
        with features[0]:
            st.markdown("📁 **Multiple Inputs**\nCSV, Manual, Sample")
        with features[1]:
            st.markdown("📊 **Visualizations**\nCharts, Heatmaps, Tables")
        with features[2]:
            st.markdown("📥 **Export**\nCSV and JSON downloads")
        with features[3]:
            st.markdown("🎯 **Auto-detect**\nSmart category detection")

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
