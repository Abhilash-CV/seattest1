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
# CONSTANTS
# ============================================================================

SEAT_MATRIX = {
    'SM': 50, 'EW': 10, 'EZ': 9, 'MU': 8, 'SC': 8, 
    'BH': 3, 'LA': 3, 'DV': 2, 'VK': 2, 'ST': 2, 
    'KN': 1, 'BX': 1, 'KU': 1
}

# ============================================================================
# SEAT ALLOCATOR CLASS
# ============================================================================

class SeatAllocator:
    def __init__(self, college_data):
        self.college_data = college_data.copy()
        self.categories = list(SEAT_MATRIX.keys())
        self.total_seats = sum(SEAT_MATRIX.values())
    
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
        
        # Category shares
        category_shares = np.array(list(SEAT_MATRIX.values())) / self.total_seats
        
        results = {'hamilton': {}, 'biproportional': {}}
        
        for idx, category in enumerate(self.categories):
            cat_total = SEAT_MATRIX[category]
            
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
                row_marg = college_seats * category_shares[idx]
                col_marg = specialty_seats * category_shares[idx]
                
                if row_marg.sum() > 0:
                    row_marg = row_marg / row_marg.sum() * cat_total
                if col_marg.sum() > 0:
                    col_marg = col_marg / col_marg.sum() * cat_total
                
                row_marg = np.round(row_marg).astype(int)
                col_marg = np.round(col_marg).astype(int)
                
                # Adjust margins
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

def display_results(results, colleges, specialties):
    """Display allocation results"""
    
    # Summary cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Seats", sum(SEAT_MATRIX.values()))
    with col2:
        st.metric("Categories", len(SEAT_MATRIX))
    with col3:
        st.metric("Colleges", len(colleges))
    
    # Tabs
    tabs = st.tabs(["Hamilton Method", "Biproportional Method", "Comparison", "Download"])
    
    # Tab 1: Hamilton
    with tabs[0]:
        st.subheader("Hamilton Rounding Method")
        st.info("Allocates seats using the largest remainder method")
        
        # Summary
        summary = []
        for cat, matrix in results['hamilton'].items():
            total = matrix.sum()
            summary.append({
                'Category': cat,
                'Allocated': int(total),
                'Expected': SEAT_MATRIX[cat],
                'Diff': int(total - SEAT_MATRIX[cat])
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True)
        
        # Detailed matrices
        with st.expander("View Detailed Matrices"):
            for cat, matrix in results['hamilton'].items():
                st.write(f"**{cat}**")
                st.dataframe(
                    pd.DataFrame(matrix, index=colleges, columns=specialties),
                    use_container_width=True
                )
    
    # Tab 2: Biproportional
    with tabs[1]:
        st.subheader("Biproportional Method")
        st.info("Balances allocation across both colleges and specialties")
        
        # Summary
        summary = []
        for cat, matrix in results['biproportional'].items():
            total = matrix.sum()
            summary.append({
                'Category': cat,
                'Allocated': int(total),
                'Expected': SEAT_MATRIX[cat],
                'Diff': int(total - SEAT_MATRIX[cat])
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True)
        
        # Detailed matrices
        with st.expander("View Detailed Matrices"):
            for cat, matrix in results['biproportional'].items():
                st.write(f"**{cat}**")
                st.dataframe(
                    pd.DataFrame(matrix, index=colleges, columns=specialties),
                    use_container_width=True
                )
    
    # Tab 3: Comparison
    with tabs[2]:
        st.subheader("Method Comparison")
        
        # Comparison chart
        comp_data = []
        for cat in SEAT_MATRIX.keys():
            comp_data.append({
                'Category': cat,
                'Expected': SEAT_MATRIX[cat],
                'Hamilton': int(results['hamilton'][cat].sum()),
                'Biproportional': int(results['biproportional'][cat].sum())
            })
        
        df_comp = pd.DataFrame(comp_data)
        st.dataframe(df_comp, use_container_width=True)
        
        # Bar chart
        fig = go.Figure()
        for method, color in [('Expected', '#1f77b4'), ('Hamilton', '#2ca02c'), ('Biproportional', '#ff7f0e')]:
            fig.add_trace(go.Bar(
                x=df_comp['Category'],
                y=df_comp[method],
                name=method,
                marker_color=color
            ))
        fig.update_layout(
            title='Seat Allocation Comparison',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Download
    with tabs[3]:
        st.subheader("Download Results")
        
        # Prepare download data
        download_data = []
        for method in ['hamilton', 'biproportional']:
            for cat in SEAT_MATRIX.keys():
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
                "text/csv"
            )
        with col2:
            json_str = json.dumps({
                'timestamp': datetime.now().isoformat(),
                'results': {
                    m: {c: results[m][c].tolist() for c in results[m].keys()}
                    for m in ['hamilton', 'biproportional']
                }
            }, indent=2)
            st.download_button(
                "📥 Download JSON",
                json_str,
                f"allocations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.title("🎓 Seat Allocation System")
    st.markdown("### Hamilton Rounding vs Biproportional Method")
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Show seat matrix
        with st.expander("Seat Matrix", expanded=True):
            st.dataframe(
                pd.DataFrame(list(SEAT_MATRIX.items()), columns=['Category', 'Seats']),
                use_container_width=True
            )
        
        # Data input
        st.subheader("Data Input")
        input_type = st.radio(
            "Choose input method:",
            ["Sample Data", "Upload CSV", "Manual Entry"]
        )
        
        data = None
        
        if input_type == "Sample Data":
            data = get_sample_data()
            st.success("Using sample data")
            st.dataframe(data, use_container_width=True)
        
        elif input_type == "Upload CSV":
            uploaded = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded:
                try:
                    data = pd.read_csv(uploaded)
                    st.success("Data loaded successfully")
                    st.dataframe(data, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
        
        else:  # Manual Entry
            st.info("Enter data manually")
            n = st.number_input("Number of rows", min_value=1, max_value=20, value=3)
            
            rows = []
            for i in range(n):
                cols = st.columns(5)
                with cols[0]:
                    prog = st.text_input(f"Program {i+1}", "E", key=f"p{i}")
                with cols[1]:
                    spec = st.text_input(f"Specialty {i+1}", "DS", key=f"s{i}")
                with cols[2]:
                    col = st.text_input(f"College {i+1}", f"CD{i+1}", key=f"c{i}")
                with cols[3]:
                    seat = st.number_input(f"Seats {i+1}", 1, 20, 1, key=f"st{i}")
                with cols[4]:
                    typ = st.text_input(f"Type {i+1}", "G", key=f"t{i}")
                rows.append({'Program': prog, 'Specialty': spec, 'College': col, 'Seats': seat, 'Type': typ})
            
            if rows:
                data = pd.DataFrame(rows)
        
        # Calculate button
        if st.button("🚀 Calculate Allocations", type="primary", use_container_width=True):
            if data is not None and not data.empty:
                try:
                    with st.spinner("Calculating..."):
                        allocator = SeatAllocator(data)
                        results, colleges, specialties = allocator.calculate_allocations()
                        
                        st.session_state['results'] = results
                        st.session_state['colleges'] = colleges
                        st.session_state['specialties'] = specialties
                        st.session_state['calculated'] = True
                        
                    st.success("✅ Allocations calculated!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please provide valid data")
    
    # Main content
    if st.session_state.get('calculated', False) and st.session_state.get('results'):
        display_results(
            st.session_state['results'],
            st.session_state['colleges'],
            st.session_state['specialties']
        )
    else:
        st.info("👈 Configure your data in the sidebar and click 'Calculate Allocations'")
        
        # Show preview
        st.subheader("How it works")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Hamilton Method**
            - Largest remainder method
            - Simple and transparent
            - Good for proportional allocation
            """)
        
        with col2:
            st.markdown("""
            **Biproportional Method**
            - Iterative proportional fitting
            - Balances row and column constraints
            - More complex but precise
            """)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
