import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
from io import StringIO

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

# Original seat matrix from the problem
SEAT_MATRIX = {
    'SM': 50, 'EW': 10, 'EZ': 9, 'MU': 8, 'SC': 8, 
    'BH': 3, 'LA': 3, 'DV': 2, 'VK': 2, 'ST': 2, 
    'KN': 1, 'BX': 1, 'KU': 1
}

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def process_allocated_data(data):
    """
    Process the allocated data to show summary statistics
    """
    # Summary by Category
    category_summary = data.groupby('Category')['Seats'].sum().reset_index()
    category_summary['Expected'] = category_summary['Category'].map(SEAT_MATRIX)
    category_summary['Difference'] = category_summary['Seats'] - category_summary['Expected']
    category_summary['Accuracy'] = (category_summary['Seats'] / category_summary['Expected'] * 100).round(1)
    
    # Summary by College
    college_summary = data.groupby('College')['Seats'].sum().reset_index()
    college_summary = college_summary.sort_values('Seats', ascending=False)
    
    # Summary by Specialty
    specialty_summary = data.groupby('Specialty')['Seats'].sum().reset_index()
    specialty_summary = specialty_summary.sort_values('Seats', ascending=False)
    
    # Summary by Program
    program_summary = data.groupby('Program')['Seats'].sum().reset_index()
    
    # College-Specialty-Category breakdown
    pivot_csc = data.pivot_table(
        index=['College', 'Specialty'],
        columns='Category',
        values='Seats',
        fill_value=0
    ).reset_index()
    
    return {
        'category_summary': category_summary,
        'college_summary': college_summary,
        'specialty_summary': specialty_summary,
        'program_summary': program_summary,
        'pivot_csc': pivot_csc,
        'total_seats': data['Seats'].sum(),
        'total_categories': len(data['Category'].unique()),
        'total_colleges': len(data['College'].unique()),
        'total_specialties': len(data['Specialty'].unique())
    }

def validate_allocations(data):
    """
    Validate that allocations match the seat matrix
    """
    validation_results = []
    
    # Check each category
    for category, expected in SEAT_MATRIX.items():
        actual = data[data['Category'] == category]['Seats'].sum()
        status = '✅' if actual == expected else '⚠️'
        validation_results.append({
            'Category': category,
            'Expected': expected,
            'Actual': int(actual),
            'Difference': int(actual - expected),
            'Status': status
        })
    
    # Check total
    total_actual = data['Seats'].sum()
    total_expected = sum(SEAT_MATRIX.values())
    status = '✅' if total_actual == total_expected else '⚠️'
    validation_results.append({
        'Category': 'TOTAL',
        'Expected': total_expected,
        'Actual': int(total_actual),
        'Difference': int(total_actual - total_expected),
        'Status': status
    })
    
    return pd.DataFrame(validation_results)

# ============================================================================
# SAMPLE DATA
# ============================================================================

def get_sample_data():
    """Create sample data with the format you provided"""
    data = """Program,Specialty,College,Type,Category,Seats
E,CU,KSD,G,SM,2
E,CU,KSD,G,BH,1
E,CU,MDL,G,SM,2
E,CU,MDL,G,BH,1
E,CU,PRP,G,SM,2
E,CU,PRP,G,BH,1
E,ID,KSD,G,SM,1
E,ID,KSD,G,ST,1
E,ID,KSD,G,BH,1
E,ES,LBT,G,SM,1
E,ES,LBT,G,BH,1
E,ES,LBT,G,VK,1
E,ES,PRN,G,SM,1
E,ES,PRN,G,EZ,1
E,ES,PRN,G,BH,1
E,ES,TLY,G,SM,1
E,ES,TLY,G,BH,1
E,ES,TLY,G,LA,1
E,AD,CEC,G,SM,1
E,AD,CEC,G,BH,1
E,AD,CEC,G,VK,1
E,AD,KNP,G,SM,1
E,AD,KNP,G,BH,1
E,AD,KNP,G,DV,1
E,AD,LBT,G,SM,1
E,AD,LBT,G,ST,1
E,AD,LBT,G,BH,1
E,AD,PRN,G,SM,1
E,AD,PRN,G,BH,1
E,AD,PRN,G,VK,1
E,AD,UCE,G,SM,1
E,AD,UCE,G,BH,1
E,AD,UCE,G,LA,1
E,CL,CEA,G,SM,2
E,CL,CEA,G,BH,1
E,CL,CEK,G,SM,2
E,CL,CEK,G,LA,1
E,CL,CHN,G,SM,2
E,CL,CHN,G,LA,1
E,CL,SCT,G,SM,1
E,CL,SCT,G,BX,1
E,CV,WYD,G,SM,1
E,CV,WYD,G,KN,1
E,CO,ADR,G,SM,2
E,CO,ADR,G,BH,1
E,CO,TKR,G,SM,1
E,EB,MDL,G,SM,2
E,EB,MDL,G,VK,1
E,EV,MDL,G,SM,2
E,EV,MDL,G,DV,1
E,EP,UCC,G,SM,1
E,PT,UCC,G,SM,1
E,PO,UCE,G,SM,2
E,PO,UCE,G,BH,1
E,CT,TKR,G,SM,1
E,MA,SCT,G,SM,1
E,MA,SCT,G,ST,1
E,MA,SCT,G,LA,1
E,CY,PTA,G,SM,1
E,CY,PTA,G,ST,1
E,CY,PTA,G,LA,1
E,CY,UCE,G,SM,1
E,CY,UCE,G,ST,1
E,CY,UCE,G,LA,1
E,BB,SCT,G,SM,1
E,BB,SCT,G,DV,1
E,BB,SCT,G,VK,1
E,IE,TVE,G,SM,1
E,IE,TVE,G,EZ,1
E,IE,TVE,G,VK,1
E,PE,TCR,G,SM,2
E,PE,TCR,G,EW,1
E,PE,TCR,G,EZ,1
E,IT,IDK,G,SM,2
E,IT,IDK,G,EW,1
E,IT,IDK,G,EZ,1
E,IT,PKD,G,SM,2
E,IT,PKD,G,EW,1
E,IT,PKD,G,EZ,1
E,IT,PKD,G,MU,1
E,IT,TRV,G,SM,1
E,IT,TRV,G,KN,1
E,IT,KSD,G,SM,1
E,IT,KSD,G,DV,1
E,IT,KSD,G,VK,1
E,IT,LBT,G,SM,1
E,IT,LBT,G,ST,1
E,IT,LBT,G,LA,1
E,IT,TLY,G,SM,2
E,IT,TLY,G,LA,1
E,IT,UCK,G,SM,2
E,IT,UCK,G,EW,1
E,IT,VDA,G,SM,2
E,IT,VDA,G,LA,1
E,ME,IDK,G,SM,2
E,ME,IDK,G,EW,1
E,ME,KKE,G,SM,2
E,ME,KKE,G,EW,1
E,ME,KNR,G,SM,2
E,ME,KNR,G,EZ,1
E,ME,KTE,G,SM,2
E,ME,KTE,G,EW,1
E,ME,NSS,G,SM,3
E,ME,NSS,G,EW,1
E,ME,NSS,G,EZ,1
E,ME,PKD,G,SM,2
E,ME,PKD,G,LA,1
E,ME,TCR,G,SM,3
E,ME,TCR,G,EW,1
E,ME,TCR,G,EZ,1
E,ME,TRV,G,SM,1
E,ME,TRV,G,DV,1
E,ME,TRV,G,VK,1
E,ME,TVE,G,SM,2
E,ME,TVE,G,EW,1
E,ME,TVE,G,EZ,1
E,ME,WYD,G,SM,1
E,ME,WYD,G,DV,1
E,ME,WYD,G,VK,1
E,ME,ADR,G,SM,1
E,ME,ADR,G,ST,1
E,ME,ADR,G,DV,1
E,ME,CEM,G,SM,1
E,ME,CEM,G,EW,1
E,ME,CEM,G,LA,1
E,ME,KNP,G,KN,1
E,ME,KSD,G,SM,1
E,ME,KSD,G,EW,1
E,ME,KSD,G,VK,1
E,ME,MDL,G,SM,2
E,ME,MDL,G,LA,1
E,ME,MNR,G,SM,1
E,ME,MNR,G,SC,1
E,ME,MNR,G,MU,1
E,ME,PEC,G,SM,2
E,ME,PEC,G,SC,1
E,ME,PRN,G,SM,1
E,ME,PRN,G,EW,1
E,ME,PRN,G,MU,1
E,ME,PRP,G,SM,1
E,ME,PRP,G,EW,1
E,ME,PRP,G,SC,1
E,ME,SCT,G,SM,3
E,ME,SCT,G,EW,1
E,ME,SCT,G,SC,1
E,ME,SCT,G,EZ,1
E,ME,TLY,G,SM,1
E,ME,TLY,G,SC,1
E,ME,TLY,G,MU,1
E,ME,UCC,G,SM,1
E,ME,UCC,G,SC,1
E,ME,UCC,G,DV,1
E,RA,IDK,G,SM,1
E,RA,IDK,G,KN,1
E,RA,KTE,G,SM,1
E,RA,KTE,G,BX,1
E,EE,IDK,G,SM,2
E,EE,IDK,G,EZ,1
E,EE,KNR,G,SM,2
E,EE,KNR,G,SC,1
E,EE,KTE,G,SM,2
E,EE,KTE,G,SC,1
E,EE,NSS,G,SM,3
E,EE,NSS,G,EW,1
E,EE,NSS,G,EZ,1
E,EE,PKD,G,SM,2
E,EE,PKD,G,EZ,1
E,EE,TCR,G,SM,3
E,EE,TCR,G,EW,1
E,EE,TCR,G,EZ,1
E,EE,TRV,G,SM,2
E,EE,TRV,G,EW,1
E,EE,TVE,G,SM,3
E,EE,TVE,G,EW,1
E,EE,TVE,G,EZ,1
E,EE,WYD,G,SM,2
E,EE,WYD,G,EW,1
E,EE,ADR,G,SM,2
E,EE,ADR,G,EW,1
E,EE,CEA,G,SM,2
E,EE,CEA,G,MU,1
E,EE,CEC,G,KU,1
E,EE,CEM,G,SM,1
E,EE,CEM,G,SC,1
E,EE,CEM,G,MU,1
E,EE,CHN,G,SM,1
E,EE,CHN,G,SC,1
E,EE,CHN,G,MU,1
E,EE,KGR,G,KU,1
E,EE,KSD,G,SM,1
E,EE,KSD,G,SC,1
E,EE,KSD,G,MU,1
E,EE,MDL,G,SM,1
E,EE,MDL,G,EW,1
E,EE,MDL,G,SC,1
E,EE,MNR,G,SM,1
E,EE,MNR,G,SC,1
E,EE,MNR,G,MU,1
E,EE,PEC,G,SM,1
E,EE,PEC,G,SC,1
E,EE,PEC,G,MU,1
E,EE,PRN,G,SM,1
E,EE,PRN,G,SC,1
E,EE,PRN,G,EZ,1
E,EE,PRP,G,KU,1
E,EE,PTA,G,KU,1
E,EE,TKR,G,SM,1
E,EE,TKR,G,SC,1
E,EE,TKR,G,MU,1
E,EE,TLY,G,SM,1
E,EE,TLY,G,SC,1
E,EE,TLY,G,MU,1
E,EE,UCC,G,SM,1
E,EE,UCC,G,SC,1
E,EE,UCC,G,MU,1
E,EE,UCE,G,SM,1
E,EE,UCE,G,SC,1
E,EE,UCE,G,MU,1
E,EE,VDA,G,SM,2
E,EE,VDA,G,SC,1
E,EC,IDK,G,SM,2
E,EC,IDK,G,EW,1
E,EC,IDK,G,EZ,1
E,EC,KKE,G,SM,2
E,EC,KKE,G,SC,1
E,EC,KNR,G,SM,2
E,EC,KNR,G,MU,1
E,EC,KTE,G,SM,1
E,EC,NSS,G,SM,3
E,EC,NSS,G,EW,1
E,EC,NSS,G,EZ,1
E,EC,PKD,G,SM,2
E,EC,PKD,G,MU,1
E,EC,TCR,G,SM,3
E,EC,TCR,G,EW,1
E,EC,TCR,G,EZ,1
E,EC,TRV,G,SM,2
E,EC,TRV,G,MU,1
E,EC,TVE,G,SM,2
E,EC,TVE,G,EZ,1
E,EC,WYD,G,SM,3
E,EC,WYD,G,EW,1
E,EC,WYD,G,SC,1
E,EC,WYD,G,EZ,1
E,EC,WYD,G,MU,1
E,EC,ADR,G,SM,2
E,EC,ADR,G,SC,1
E,EC,AEC,G,SM,2
E,EC,AEC,G,MU,1
E,EC,CEA,G,SM,2
E,EC,CEA,G,MU,1
E,EC,CEC,G,SM,2
E,EC,CEC,G,EZ,1
E,EC,CEM,G,SM,2
E,EC,CEM,G,MU,1
E,EC,CHN,G,SM,3
E,EC,CHN,G,EW,1
E,EC,CHN,G,SC,1
E,EC,CHN,G,EZ,1
E,EC,KGR,G,SM,3
E,EC,KGR,G,EW,1
E,EC,KGR,G,EZ,1
E,EC,KGR,G,MU,1
E,EC,KNP,G,SM,1
E,EC,KNP,G,SC,1
E,EC,KNP,G,MU,1
E,EC,KSD,G,SM,1
E,EC,KSD,G,SC,1
E,EC,KSD,G,MU,1
E,EC,LBT,G,SM,2
E,EC,LBT,G,MU,1
E,EC,MDL,G,SM,3
E,EC,MDL,G,EW,1
E,EC,MDL,G,SC,1
E,EC,MDL,G,EZ,1
E,EC,MNR,G,SM,1
E,EC,MNR,G,SC,1
E,EC,MNR,G,MU,1
E,EC,PEC,G,SM,1
E,EC,PEC,G,SC,1
E,EC,PEC,G,MU,1
E,EC,PJR,G,KN,1
E,EC,PRN,G,SM,3
E,EC,PRN,G,EW,1
E,EC,PRN,G,EZ,1
E,EC,PRN,G,MU,1
E,EC,PRP,G,SM,2
E,EC,PRP,G,SC,1
E,EC,PTA,G,SM,2
E,EC,PTA,G,MU,1
E,EC,SCT,G,SM,3
E,EC,SCT,G,EW,1
E,EC,SCT,G,SC,1
E,EC,SCT,G,EZ,1
E,EC,TKR,G,SM,2
E,EC,TKR,G,EW,1
E,EC,TKR,G,EZ,1
E,EC,TLY,G,SM,3
E,EC,TLY,G,EW,1
E,EC,TLY,G,SC,1
E,EC,TLY,G,EZ,1
E,EC,UCC,G,SM,1
E,EC,UCC,G,BX,1
E,EC,UCE,G,SM,1
E,EC,UCE,G,MU,1
E,EC,UCE,G,BH,1
E,EC,UCK,G,SM,2
E,EC,UCK,G,ST,1
E,EC,VDA,G,SM,2
E,EC,VDA,G,MU,1
E,FT,COU,G,SM,3
E,FT,COU,G,EW,1
E,FT,COU,G,SC,1
E,FT,COU,G,EZ,1
E,FT,COU,G,MU,1
E,FT,KCT,G,SM,1
E,FT,KCT,G,BX,1
E,CS,IDK,G,SM,2
E,CS,IDK,G,EW,1
E,CS,IDK,G,EZ,1
E,CS,KNR,G,KU,1
E,CS,KTE,G,SM,2
E,CS,KTE,G,SC,1
E,CS,NSS,G,SM,3
E,CS,NSS,G,EW,1
E,CS,NSS,G,EZ,1
E,CS,PKD,G,SM,1
E,CS,PKD,G,ST,1
E,CS,PKD,G,DV,1
E,CS,TCR,G,BX,1
E,CS,WYD,G,SM,2
E,CS,WYD,G,ST,1
E,CS,ADR,G,SM,3
E,CS,ADR,G,EW,1
E,CS,ADR,G,EZ,1
E,CS,ADR,G,MU,1
E,CS,AEC,G,SM,2
E,CS,AEC,G,EW,1
E,CS,CEA,G,SM,3
E,CS,CEA,G,EW,1
E,CS,CEA,G,SC,1
E,CS,CEA,G,EZ,1
E,CS,CEA,G,MU,1
E,CS,CEC,G,SM,3
E,CS,CEC,G,EW,1
E,CS,CEC,G,EZ,1
E,CS,CEC,G,MU,1
E,CS,CEK,G,SM,3
E,CS,CEK,G,EW,1
E,CS,CEK,G,EZ,1
E,CS,CEK,G,MU,1
E,CS,CEM,G,SM,3
E,CS,CEM,G,EW,1
E,CS,CEM,G,SC,1
E,CS,CEM,G,EZ,1
E,CS,CHN,G,SM,5
E,CS,CHN,G,EW,1
E,CS,CHN,G,SC,1
E,CS,CHN,G,EZ,1
E,CS,CHN,G,MU,1
E,CS,KGR,G,SM,3
E,CS,KGR,G,EW,1
E,CS,KGR,G,EZ,1
E,CS,KGR,G,MU,1
E,CS,KNP,G,SM,3
E,CS,KNP,G,EW,1
E,CS,KNP,G,EZ,1
E,CS,KNP,G,MU,1
E,CS,KSD,G,SM,3
E,CS,KSD,G,EW,1
E,CS,KSD,G,SC,1
E,CS,KSD,G,EZ,1
E,CS,KSD,G,MU,1
E,CS,LBT,G,SM,4
E,CS,LBT,G,EW,1
E,CS,LBT,G,SC,1
E,CS,LBT,G,EZ,1
E,CS,LBT,G,MU,1
E,CS,LBT,G,BH,1
E,CS,MDL,G,SM,4
E,CS,MDL,G,EW,1
E,CS,MDL,G,SC,1
E,CS,MDL,G,EZ,1
E,CS,MDL,G,MU,1
E,CS,MNR,G,SM,1
E,CS,MNR,G,EW,1
E,CS,MNR,G,LA,1
E,CS,PEC,G,SM,1
E,CS,PEC,G,SC,1
E,CS,PEC,G,MU,1
E,CS,PJR,G,SM,2
E,CS,PJR,G,EW,1
E,CS,PJR,G,EZ,1
E,CS,PRN,G,SM,3
E,CS,PRN,G,EW,1
E,CS,PRN,G,SC,1
E,CS,PRN,G,EZ,1
E,CS,PRP,G,SM,3
E,CS,PRP,G,EW,1
E,CS,PRP,G,SC,1
E,CS,PRP,G,EZ,1
E,CS,PTA,G,SM,3
E,CS,PTA,G,EW,1
E,CS,PTA,G,EZ,1
E,CS,PTA,G,MU,1
E,CS,SCT,G,SM,3
E,CS,SCT,G,EW,1
E,CS,SCT,G,SC,1
E,CS,SCT,G,EZ,1
E,CS,TKR,G,SM,3
E,CS,TKR,G,EW,1
E,CS,TKR,G,SC,1
E,CS,TKR,G,EZ,1
E,CS,TLY,G,SM,3
E,CS,TLY,G,EW,1
E,CS,TLY,G,SC,1
E,CS,TLY,G,EZ,1
E,CS,UCC,G,SM,2
E,CS,UCC,G,SC,1
E,CS,UCE,G,SM,2
E,CS,UCE,G,MU,1
E,CS,UCK,G,SM,3
E,CS,UCK,G,EW,1
E,CS,UCK,G,EZ,1
E,CS,UCK,G,MU,1
E,CS,VDA,G,SM,2
E,CS,VDA,G,MU,1
E,AG,KCT,G,SM,2
E,AG,KCT,G,EW,1
E,AG,KCT,G,EZ,1
E,AJ,KCT,G,SM,1
E,AJ,KCT,G,KN,1
E,EL,KKE,G,SM,1
E,EL,KKE,G,DV,1
E,EL,TVE,G,SM,1
E,EL,TVE,G,KN,1
E,EL,KGR,G,KN,1
E,EL,PRP,G,BX,1
E,IC,NSS,G,SM,1
E,IC,NSS,G,EZ,1
E,IC,NSS,G,MU,1
E,CB,TCR,G,SM,1
E,CB,TCR,G,BX,1
E,CB,PTA,G,KU,1
E,CH,KKE,G,SM,1
E,CH,KKE,G,SC,1
E,CH,KKE,G,MU,1
E,CH,TCR,G,SM,3
E,CH,TCR,G,EW,1
E,CH,TCR,G,SC,1
E,CH,TCR,G,EZ,1
E,CG,KKE,G,SM,2
E,CG,KKE,G,LA,1
E,AE,KKE,G,SM,2
E,AE,KKE,G,EW,1
E,AE,KKE,G,EZ,1
E,AE,TVE,G,SM,1
E,AE,TVE,G,SC,1
E,AE,TVE,G,MU,1
E,CE,KKE,G,SM,1
E,CE,KKE,G,SC,1
E,CE,KKE,G,MU,1
E,CE,KNR,G,SM,1
E,CE,KNR,G,SC,1
E,CE,KNR,G,MU,1
E,CE,KTE,G,SM,3
E,CE,KTE,G,EW,1
E,CE,KTE,G,EZ,1
E,CE,NSS,G,SM,2
E,CE,NSS,G,EW,1
E,CE,NSS,G,EZ,1
E,CE,NSS,G,MU,1
E,CE,PKD,G,SM,1
E,CE,PKD,G,ST,1
E,CE,PKD,G,LA,1
E,CE,TCR,G,SM,3
E,CE,TCR,G,EW,1
E,CE,TCR,G,SC,1
E,CE,TCR,G,EZ,1
E,CE,TRV,G,SM,2
E,CE,TRV,G,EW,1
E,CE,TVE,G,SM,1
E,CE,TVE,G,ST,1
E,CE,TVE,G,VK,1
E,CE,AEC,G,SM,1
E,CE,AEC,G,EW,1
E,CE,AEC,G,ST,1
E,CE,CEM,G,SM,1
E,CE,CEM,G,LA,1
E,CE,CEM,G,DV,1
E,CE,KGR,G,SM,1
E,CE,KGR,G,BH,1
E,CE,KGR,G,LA,1
E,CE,KSD,G,SM,2
E,CE,KSD,G,BH,1
E,CE,LBT,G,SM,2
E,CE,LBT,G,VK,1
E,CE,PEC,G,SM,2
E,CE,PEC,G,DV,1
E,CE,PRP,G,SM,2
E,CE,PRP,G,DV,1
E,CE,TKR,G,SM,2
E,CE,TKR,G,VK,1
E,CE,TLY,G,SM,2
E,CE,TLY,G,LA,1
E,CE,VDA,G,SM,2
E,CE,VDA,G,BH,1
E,DS,CDI,G,KU,1
E,DS,CDP,G,BX,1
E,DS,CDT,G,SM,2
E,DS,CDT,G,EW,1
E,DS,CDT,G,EZ,1
E,DS,CDV,G,SM,1"""
    
    df = pd.read_csv(StringIO(data))
    return df

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_category_chart(summary_data):
    """Create category distribution chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=summary_data['Category'],
        y=summary_data['Seats'],
        name='Allocated',
        marker_color='#2ca02c',
        text=summary_data['Seats'],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        x=summary_data['Category'],
        y=summary_data['Expected'],
        name='Expected',
        marker_color='#1f77b4',
        text=summary_data['Expected'],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Category-wise Seat Allocation',
        xaxis_title='Category',
        yaxis_title='Number of Seats',
        barmode='group',
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    
    return fig

def create_college_chart(college_summary, top_n=20):
    """Create college distribution chart"""
    top_colleges = college_summary.head(top_n)
    
    fig = px.bar(
        top_colleges,
        x='College',
        y='Seats',
        title=f'Top {top_n} Colleges by Seat Allocation',
        color='Seats',
        color_continuous_scale='Viridis',
        text='Seats'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(height=500)
    
    return fig

def create_specialty_chart(specialty_summary, top_n=15):
    """Create specialty distribution chart"""
    top_specialties = specialty_summary.head(top_n)
    
    fig = px.bar(
        top_specialties,
        x='Specialty',
        y='Seats',
        title=f'Top {top_n} Specialties by Seat Allocation',
        color='Seats',
        color_continuous_scale='Plasma',
        text='Seats'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(height=500)
    
    return fig

def create_heatmap(data):
    """Create heatmap of allocations"""
    pivot = data.pivot_table(
        index='College',
        columns='Category',
        values='Seats',
        fill_value=0,
        aggfunc='sum'
    )
    
    # Filter to top colleges and categories for readability
    top_colleges = data.groupby('College')['Seats'].sum().nlargest(20).index
    top_categories = data.groupby('Category')['Seats'].sum().nlargest(10).index
    
    pivot = pivot.loc[top_colleges, top_categories]
    
    fig = px.imshow(
        pivot,
        title='Seat Allocation Heatmap (Top 20 Colleges x Top 10 Categories)',
        color_continuous_scale='Viridis',
        text_auto=True,
        aspect='auto',
        height=600
    )
    fig.update_layout(
        xaxis_title='Category',
        yaxis_title='College'
    )
    
    return fig

def create_sunburst_chart(data):
    """Create sunburst chart"""
    fig = px.sunburst(
        data,
        path=['Program', 'Specialty', 'College', 'Category'],
        values='Seats',
        title='Seat Allocation Hierarchy',
        color='Category',
        height=600
    )
    
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed' not in st.session_state:
        st.session_state.processed = None
    
    st.title("🎓 Seat Allocation System")
    st.markdown("### View and Analyze Seat Allocation Data")
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data input
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
        
        else:  # Paste Data
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
            
            # Process data
            if st.button("📊 Process Data", type="primary", use_container_width=True):
                with st.spinner("Processing data..."):
                    processed = process_allocated_data(data)
                    st.session_state.processed = processed
                    st.success("✅ Data processed successfully!")
                    st.balloons()
    
    # Main content
    if st.session_state.processed is not None:
        processed = st.session_state.processed
        data = st.session_state.data
        
        # Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Seats", processed['total_seats'])
        with col2:
            st.metric("Categories", processed['total_categories'])
        with col3:
            st.metric("Colleges", processed['total_colleges'])
        with col4:
            st.metric("Specialties", processed['total_specialties'])
        with col5:
            expected_total = sum(SEAT_MATRIX.values())
            diff = processed['total_seats'] - expected_total
            st.metric(
                "vs Expected", 
                f"{processed['total_seats'] - expected_total:+d}",
                delta_color="inverse"
            )
        
        # Validation
        validation_df = validate_allocations(data)
        
        # Tabs
        tabs = st.tabs([
            "📊 Summary",
            "📈 Category Analysis",
            "🏛️ College Analysis",
            "📚 Specialty Analysis",
            "🔍 Detailed View",
            "📋 Validation",
            "📥 Download"
        ])
        
        # Tab 1: Summary
        with tabs[0]:
            st.subheader("Summary Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Category Summary")
                st.dataframe(processed['category_summary'], use_container_width=True)
                
                st.markdown("#### Program Summary")
                st.dataframe(processed['program_summary'], use_container_width=True)
            
            with col2:
                st.markdown("#### Top 10 Colleges")
                st.dataframe(processed['college_summary'].head(10), use_container_width=True)
                
                st.markdown("#### Top 10 Specialties")
                st.dataframe(processed['specialty_summary'].head(10), use_container_width=True)
        
        # Tab 2: Category Analysis
        with tabs[1]:
            st.subheader("Category-wise Analysis")
            
            # Category chart
            fig = create_category_chart(processed['category_summary'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Category breakdown table
            st.markdown("#### Detailed Category Breakdown")
            
            # Pivot table for categories
            cat_pivot = data.pivot_table(
                index='Category',
                columns='Specialty',
                values='Seats',
                fill_value=0,
                aggfunc='sum'
            )
            st.dataframe(cat_pivot, use_container_width=True)
        
        # Tab 3: College Analysis
        with tabs[2]:
            st.subheader("College-wise Analysis")
            
            # College chart
            fig = create_college_chart(processed['college_summary'])
            st.plotly_chart(fig, use_container_width=True)
            
            # College-Category breakdown
            st.markdown("#### College-Category Breakdown")
            college_cat = data.pivot_table(
                index='College',
                columns='Category',
                values='Seats',
                fill_value=0,
                aggfunc='sum'
            )
            st.dataframe(college_cat, use_container_width=True)
            
            # College-Specialty breakdown
            st.markdown("#### College-Specialty Breakdown")
            college_spec = data.pivot_table(
                index='College',
                columns='Specialty',
                values='Seats',
                fill_value=0,
                aggfunc='sum'
            )
            st.dataframe(college_spec, use_container_width=True)
        
        # Tab 4: Specialty Analysis
        with tabs[3]:
            st.subheader("Specialty-wise Analysis")
            
            # Specialty chart
            fig = create_specialty_chart(processed['specialty_summary'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Specialty-Category breakdown
            st.markdown("#### Specialty-Category Breakdown")
            spec_cat = data.pivot_table(
                index='Specialty',
                columns='Category',
                values='Seats',
                fill_value=0,
                aggfunc='sum'
            )
            st.dataframe(spec_cat, use_container_width=True)
        
        # Tab 5: Detailed View
        with tabs[4]:
            st.subheader("Detailed Data View")
            
            # Heatmap
            st.markdown("#### Allocation Heatmap")
            fig = create_heatmap(data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Sunburst
            st.markdown("#### Hierarchical View")
            fig = create_sunburst_chart(data)
            st.plotly_chart(fig, use_container_width=True)
            
            # All data
            st.markdown("#### All Data")
            st.dataframe(data, use_container_width=True)
        
        # Tab 6: Validation - FIXED VERSION
        with tabs[5]:
            st.subheader("Data Validation")
            
            st.markdown("""
            This section validates the allocation against the expected seat matrix:
            - **SM**: 50 seats
            - **EW**: 10 seats
            - **EZ**: 9 seats
            - **MU**: 8 seats
            - **SC**: 8 seats
            - **BH**: 3 seats
            - **LA**: 3 seats
            - **DV**: 2 seats
            - **VK**: 2 seats
            - **ST**: 2 seats
            - **KN**: 1 seat
            - **BX**: 1 seat
            - **KU**: 1 seat
            """)
            
            # Create a styled dataframe manually
            validation_display = validation_df.copy()
            
            # Add a status column with emojis and color using HTML
            def format_status(row):
                if row['Status'] == '✅':
                    return '✅'
                else:
                    return '⚠️'
            
            validation_display['Status_Display'] = validation_display.apply(format_status, axis=1)
            
            # Display the validation results
            st.dataframe(
                validation_display[['Category', 'Expected', 'Actual', 'Difference', 'Status_Display']],
                column_config={
                    'Category': 'Category',
                    'Expected': 'Expected',
                    'Actual': 'Actual',
                    'Difference': 'Difference',
                    'Status_Display': 'Status'
                },
                use_container_width=True
            )
            
            # Show validation summary
            all_match = (validation_df['Difference'] == 0).all()
            
            if all_match:
                st.success("✅ All allocations match the expected seat matrix exactly!")
            else:
                st.warning("⚠️ Some allocations differ from the expected seat matrix")
                
                # Show mismatches
                mismatches = validation_df[validation_df['Difference'] != 0]
                if not mismatches.empty:
                    st.markdown("#### Mismatches found:")
                    st.dataframe(
                        mismatches[['Category', 'Expected', 'Actual', 'Difference']],
                        use_container_width=True
                    )
        
        # Tab 7: Download
        with tabs[6]:
            st.subheader("📥 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download CSV
                csv = data.to_csv(index=False)
                st.download_button(
                    "📥 Download Full Data as CSV",
                    csv,
                    f"seat_allocation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Download JSON
                json_data = {
                    'timestamp': datetime.now().isoformat(),
                    'seat_matrix': SEAT_MATRIX,
                    'total_seats': processed['total_seats'],
                    'summary': {
                        'category_summary': processed['category_summary'].to_dict('records'),
                        'college_summary': processed['college_summary'].to_dict('records'),
                        'specialty_summary': processed['specialty_summary'].to_dict('records')
                    },
                    'data': data.to_dict('records')
                }
                json_str = json.dumps(json_data, indent=2)
                st.download_button(
                    "📥 Download as JSON",
                    json_str,
                    f"seat_allocation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json",
                    use_container_width=True
                )
            
            # Download validation report
            st.markdown("#### Download Validation Report")
            validation_csv = validation_df.to_csv(index=False)
            st.download_button(
                "📥 Download Validation Report",
                validation_csv,
                f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
    
    else:
        # Welcome message
        st.info("👈 Load your data in the sidebar and click 'Process Data'")
        
        st.markdown("""
        ### 📋 How to Use This Application
        
        1. **Load Data**: Upload a CSV file or use sample data
        2. **Process**: Click 'Process Data' to analyze
        3. **Explore**: View results across multiple tabs
        4. **Export**: Download results in CSV or JSON format
        
        ### 📁 Required Data Format
        
        Your CSV should have these columns:
        - **Program**: Program code (e.g., E)
        - **Specialty**: Specialty name
        - **College**: College name
        - **Type**: Type (e.g., G)
        - **Category**: Seat category (SM, EW, EZ, etc.)
        - **Seats**: Number of seats
        
        ### 🎯 Expected Seat Matrix
        
        - **SM**: 50 seats
        - **EW**: 10 seats
        - **EZ**: 9 seats
        - **MU**: 8 seats
        - **SC**: 8 seats
        - **BH**: 3 seats
        - **LA**: 3 seats
        - **DV**: 2 seats
        - **VK**: 2 seats
        - **ST**: 2 seats
        - **KN**: 1 seat
        - **BX**: 1 seat
        - **KU**: 1 seat
        """)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
