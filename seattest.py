import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
from io import StringIO, BytesIO

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

# Original seat matrix from the problem (values are also percentage points, sum = 100)
SEAT_MATRIX = {
    'SM': 50, 'EW': 10, 'EZ': 9, 'MU': 8, 'SC': 8,
    'BH': 3, 'LA': 3, 'DV': 2, 'VK': 2, 'ST': 2,
    'KN': 1, 'BX': 1, 'KU': 1
}

# ============================================================================
# APPORTIONMENT (LARGEST REMAINDER / HAMILTON METHOD)
# ============================================================================

def apportion_largest_remainder(total, weights):
    """
    Split an integer `total` across the keys of `weights` (a dict of relative
    weights, e.g. SEAT_MATRIX) so that:
      - the result is all integers
      - the integers sum EXACTLY to `total`
      - each key's share is as close as possible to its ideal weighted fraction

    This is the Largest Remainder (Hamilton) apportionment method: give every
    key its floor share, then hand out the leftover seats one at a time to
    whichever keys have the largest fractional remainder.
    """
    total = int(round(total))
    weight_sum = sum(weights.values())
    if total <= 0 or weight_sum <= 0:
        return {k: 0 for k in weights}

    ideal = {k: total * w / weight_sum for k, w in weights.items()}
    floors = {k: int(np.floor(v)) for k, v in ideal.items()}
    leftover = total - sum(floors.values())
    remainders = {k: ideal[k] - floors[k] for k in weights}

    # Give leftover seats to the categories with the largest fractional remainder
    order = sorted(remainders.keys(), key=lambda k: remainders[k], reverse=True)
    result = dict(floors)
    for i in range(leftover):
        result[order[i % len(order)]] += 1

    return result


def get_overall_targets(total_seats):
    """
    Proportional overall target counts per category, scaled from the
    SEAT_MATRIX percentages to the actual dataset size (so they sum exactly
    to total_seats -- e.g. for 664 total seats, SM's target is ~332, not
    the literal matrix value of 50).
    """
    return apportion_largest_remainder(total_seats, SEAT_MATRIX)

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
    category_summary = category_summary.sort_values('Seats', ascending=False)

    # Summary by Specialty
    specialty_summary = data.groupby('Specialty')['Seats'].sum().reset_index()
    specialty_summary = specialty_summary.sort_values('Seats', ascending=False)

    # Summary by College
    college_summary = data.groupby('College')['Seats'].sum().reset_index()
    college_summary = college_summary.sort_values('Seats', ascending=False)

    # Summary by Program
    program_summary = data.groupby('Program')['Seats'].sum().reset_index()

    # Specialty-Category breakdown
    specialty_category = data.groupby(['Specialty', 'Category'])['Seats'].sum().reset_index()
    specialty_category = specialty_category.sort_values(['Specialty', 'Seats'], ascending=[True, False])

    # Pivot: Specialty x Category
    specialty_category_pivot = data.pivot_table(
        index='Specialty',
        columns='Category',
        values='Seats',
        fill_value=0,
        aggfunc='sum'
    )

    # Pivot: College x Category
    college_category_pivot = data.pivot_table(
        index='College',
        columns='Category',
        values='Seats',
        fill_value=0,
        aggfunc='sum'
    )

    # Pivot: College x Specialty
    college_specialty_pivot = data.pivot_table(
        index='College',
        columns='Specialty',
        values='Seats',
        fill_value=0,
        aggfunc='sum'
    )

    # Total seats by category
    total_by_category = data.groupby('Category')['Seats'].sum().to_dict()

    # Total seats by specialty
    total_by_specialty = data.groupby('Specialty')['Seats'].sum().to_dict()

    return {
        'category_summary': category_summary,
        'specialty_summary': specialty_summary,
        'college_summary': college_summary,
        'program_summary': program_summary,
        'specialty_category': specialty_category,
        'specialty_category_pivot': specialty_category_pivot,
        'college_category_pivot': college_category_pivot,
        'college_specialty_pivot': college_specialty_pivot,
        'total_seats': int(data['Seats'].sum()),
        'total_categories': int(len(data['Category'].unique())),
        'total_colleges': int(len(data['College'].unique())),
        'total_specialties': int(len(data['Specialty'].unique())),
        'total_by_category': total_by_category,
        'total_by_specialty': total_by_specialty
    }

def validate_allocations(data):
    """
    Validate that OVERALL (grand total) allocations match the seat matrix
    PERCENTAGES, scaled proportionally to the actual total seat count.

    Note: SEAT_MATRIX values (SM=50, EW=10, ...) sum to 100 -- they are
    percentages, not literal seat counts. Comparing raw dataset totals
    directly against those numbers only makes sense if the dataset has
    exactly 100 seats. For any other total, "Expected" is computed as
    total_seats * matrix_pct / 100 (apportioned so it sums exactly back
    to total_seats). This is what makes the overall check and the
    course-wise check mathematically compatible with each other.
    """
    validation_results = []

    total_actual = int(data['Seats'].sum())
    actual_totals = data.groupby('Category')['Seats'].sum().to_dict()
    expected_totals = get_overall_targets(total_actual)

    # Check each category
    for category, expected in expected_totals.items():
        actual = actual_totals.get(category, 0)
        status = '✅' if actual == expected else '⚠️'
        validation_results.append({
            'Category': category,
            'Expected %': SEAT_MATRIX[category],
            'Expected': int(expected),
            'Actual': int(actual),
            'Difference': int(actual - expected),
            'Status': status
        })

    # Check total (will always match by construction, since expected_totals
    # is apportioned to sum exactly to total_actual)
    total_expected = sum(expected_totals.values())
    status = '✅' if total_actual == total_expected else '⚠️'
    validation_results.append({
        'Category': 'TOTAL',
        'Expected %': 100,
        'Expected': int(total_expected),
        'Actual': int(total_actual),
        'Difference': int(total_actual - total_expected),
        'Status': status
    })

    return pd.DataFrame(validation_results)


def calculate_coursewise_compliance(data, tolerance_pts=2.0):
    """
    Check whether EACH Specialty (course) individually satisfies the SEAT_MATRIX
    percentage distribution -- not just the grand total.

    For every course, expected_seats(category) = course_total * matrix_pct / 100.
    A category is flagged as:
      - 'Missing'  -> it deserved at least ~0.5 of a seat but got literally zero
      - '⚠️'        -> actual % deviates from expected % by more than `tolerance_pts`
      - '✅'        -> within tolerance
    """
    results = []
    total_pct = sum(SEAT_MATRIX.values())  # should be 100

    for specialty, group in data.groupby('Specialty'):
        course_total = group['Seats'].sum()
        actual_by_cat = group.groupby('Category')['Seats'].sum().to_dict()

        for category, pct in SEAT_MATRIX.items():
            expected_seats = course_total * pct / total_pct
            actual_seats = actual_by_cat.get(category, 0)
            actual_pct = (actual_seats / course_total * 100) if course_total > 0 else 0
            deviation_pts = actual_pct - pct

            # Flag as "Missing" if the category deserved at least ~half a seat
            # (i.e. would round to >=1 under any sane rounding rule) but got zero.
            missing_mandatory = (expected_seats >= 0.5) and (actual_seats == 0)
            exceeds_tolerance = abs(deviation_pts) > tolerance_pts

            if missing_mandatory:
                status = 'Missing'
            elif exceeds_tolerance:
                status = '⚠️'
            else:
                status = '✅'

            results.append({
                'Specialty': specialty,
                'Category': category,
                'Course Total': int(course_total),
                'Expected %': round(pct, 2),
                'Actual %': round(actual_pct, 2),
                'Deviation (pts)': round(deviation_pts, 2),
                'Expected Seats': round(expected_seats, 2),
                'Actual Seats': int(actual_seats),
                'Status': status
            })

    return pd.DataFrame(results)


def summarize_coursewise_compliance(coursewise_df):
    """Roll the category-level compliance table up to one row per course."""
    summary = coursewise_df.groupby('Specialty').agg(
        Course_Total=('Course Total', 'first'),
        Category_Violations=('Status', lambda s: (s != '✅').sum()),
        Missing_Mandatory_Categories=('Status', lambda s: (s == 'Missing').sum())
    ).reset_index()
    summary = summary.sort_values(
        ['Missing_Mandatory_Categories', 'Category_Violations'],
        ascending=False
    )

    fully_compliant = int((summary['Category_Violations'] == 0).sum())
    total_courses = len(summary)

    return summary, fully_compliant, total_courses


def calculate_adjusted_allocation(data):
    """
    Produce a corrected (adjusted) category-seat count for every Specialty
    (course), using the Largest Remainder / Hamilton apportionment method.

    For each course: adjusted_seats sum EXACTLY to that course's original
    total seats, and each category's adjusted share is the closest possible
    integer to its ideal SEAT_MATRIX percentage of that course's total.
    Since this is applied consistently to every course, the resulting grand
    totals across all courses also land very close to (usually exactly on)
    the proportional overall targets -- satisfying BOTH the course-wise and
    the overall check simultaneously.

    This only rebalances CATEGORY TOTALS within each course. It does not
    reassign which specific college receives a seat -- that would require
    college-level capacity/eligibility rules not present in this dataset.
    """
    rows = []
    for specialty, group in data.groupby('Specialty'):
        course_total = int(group['Seats'].sum())
        actual_by_cat = group.groupby('Category')['Seats'].sum().to_dict()
        adjusted = apportion_largest_remainder(course_total, SEAT_MATRIX)

        for category in SEAT_MATRIX:
            original = int(actual_by_cat.get(category, 0))
            new_val = int(adjusted[category])
            rows.append({
                'Specialty': specialty,
                'Category': category,
                'Course Total': course_total,
                'Original Seats': original,
                'Adjusted Seats': new_val,
                'Change': new_val - original
            })

    return pd.DataFrame(rows)


def rebalance_global_allocation(adjusted_df, total_seats, max_iterations=5000):
    """
    Stage 2 of the adjustment: independent per-course rounding (Stage 1) can
    drift the OVERALL totals away from target, especially when many courses
    are too small to reflect all 13 categories individually (a 1-seat course
    can only give its single seat to ONE category).

    This pass fixes that by swapping single seats between categories WITHIN
    the same course (so course totals never change) -- moving a seat from
    whichever category is most over its overall target to whichever is most
    under, always picking the course where that swap does the LEAST damage
    (or the most good) to that course's own percentage fit.
    """
    pivot = adjusted_df.pivot(index='Specialty', columns='Category', values='Adjusted Seats').fillna(0).astype(int)
    for cat in SEAT_MATRIX:
        if cat not in pivot.columns:
            pivot[cat] = 0
    course_totals = pivot.sum(axis=1)

    # Per-course ideal shares, used only to pick the LEAST-DAMAGING course for each swap
    course_ideal = {
        specialty: apportion_largest_remainder(int(course_totals[specialty]), SEAT_MATRIX)
        for specialty in pivot.index
    }

    overall_target = get_overall_targets(total_seats)

    for _ in range(max_iterations):
        totals = pivot.sum(axis=0).to_dict()
        diffs = {cat: totals.get(cat, 0) - overall_target[cat] for cat in SEAT_MATRIX}

        surplus_cat = max(diffs, key=lambda c: diffs[c])
        deficit_cat = min(diffs, key=lambda c: diffs[c])

        # Balanced (allowing for the fact sum(diffs) == 0 always)
        if diffs[surplus_cat] <= 0 and diffs[deficit_cat] >= 0:
            break

        # Find the best course to move one seat: surplus_cat -> deficit_cat
        best_course, best_score = None, -1e18
        for specialty in pivot.index:
            if pivot.loc[specialty, surplus_cat] <= 0:
                continue
            ideal = course_ideal[specialty]
            over_amount = pivot.loc[specialty, surplus_cat] - ideal.get(surplus_cat, 0)
            under_amount = ideal.get(deficit_cat, 0) - pivot.loc[specialty, deficit_cat]
            score = over_amount + under_amount  # both positive => "double win" swap
            if score > best_score:
                best_score, best_course = score, specialty

        if best_course is None:
            break  # no seat available to move (shouldn't normally happen)

        pivot.loc[best_course, surplus_cat] -= 1
        pivot.loc[best_course, deficit_cat] += 1

    result = pivot.reset_index().melt(id_vars='Specialty', var_name='Category', value_name='Rebalanced Seats')
    merged = adjusted_df.merge(result, on=['Specialty', 'Category'], how='left')
    merged['Adjusted Seats'] = merged['Rebalanced Seats'].astype(int)
    merged['Change'] = merged['Adjusted Seats'] - merged['Original Seats']
    merged = merged.drop(columns=['Rebalanced Seats'])
    return merged


def validate_adjusted_overall(adjusted_df, total_seats):
    """Overall check applied to the ADJUSTED allocation (should match/near-match)."""
    adjusted_totals = adjusted_df.groupby('Category')['Adjusted Seats'].sum().to_dict()
    expected_totals = get_overall_targets(total_seats)

    rows = []
    for category, expected in expected_totals.items():
        actual = adjusted_totals.get(category, 0)
        status = '✅' if actual == expected else '⚠️'
        rows.append({
            'Category': category,
            'Expected %': SEAT_MATRIX[category],
            'Expected': int(expected),
            'Adjusted Actual': int(actual),
            'Difference': int(actual - expected),
            'Status': status
        })

    total_adjusted = sum(adjusted_totals.values())
    total_expected = sum(expected_totals.values())
    rows.append({
        'Category': 'TOTAL',
        'Expected %': 100,
        'Expected': int(total_expected),
        'Adjusted Actual': int(total_adjusted),
        'Difference': int(total_adjusted - total_expected),
        'Status': '✅' if total_adjusted == total_expected else '⚠️'
    })

    return pd.DataFrame(rows)


def validate_adjusted_coursewise(adjusted_df, tolerance_pts=2.0):
    """Course-wise check applied to the ADJUSTED allocation (should be clean)."""
    synthetic = adjusted_df.rename(columns={'Adjusted Seats': 'Seats'})[['Specialty', 'Category', 'Seats']]
    return calculate_coursewise_compliance(synthetic, tolerance_pts=tolerance_pts)

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

def create_specialty_chart(specialty_summary, top_n=20):
    """Create specialty distribution chart"""
    top_specialties = specialty_summary.head(top_n)

    fig = px.bar(
        top_specialties,
        x='Specialty',
        y='Seats',
        title=f'Top {top_n} Specialties by Seat Allocation',
        color='Seats',
        color_continuous_scale='Viridis',
        text='Seats'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(height=500)

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

def create_specialty_category_heatmap(data):
    """Create heatmap of Specialty x Category distribution"""
    pivot = data.pivot_table(
        index='Specialty',
        columns='Category',
        values='Seats',
        fill_value=0,
        aggfunc='sum'
    )

    # Sort by total seats per specialty
    specialty_totals = data.groupby('Specialty')['Seats'].sum().sort_values(ascending=False)
    pivot = pivot.loc[specialty_totals.index]

    fig = px.imshow(
        pivot,
        title='Specialty vs Category Heatmap',
        color_continuous_scale='Viridis',
        text_auto=True,
        aspect='auto',
        height=max(400, len(pivot.index) * 25)
    )
    fig.update_layout(
        xaxis_title='Category',
        yaxis_title='Specialty'
    )

    return fig

# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================

def convert_df_to_excel(df):
    """Convert dataframe to Excel bytes"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    return output.getvalue()

def create_download_data(data, processed, validation_df, coursewise_df=None, coursewise_summary=None,
                          adjusted_df=None, adjusted_overall_df=None, adjusted_coursewise_df=None):
    """Create comprehensive download data"""
    download_data = {}

    # Original data
    download_data['Original_Data'] = data

    # Category summary
    download_data['Category_Summary'] = processed['category_summary']

    # Specialty summary
    download_data['Specialty_Summary'] = processed['specialty_summary']

    # College summary
    download_data['College_Summary'] = processed['college_summary']

    # Program summary
    download_data['Program_Summary'] = processed['program_summary']

    # Specialty-Category breakdown
    download_data['Specialty_Category'] = processed['specialty_category']

    # Overall Validation results (BEFORE adjustment)
    download_data['Overall_Validation'] = validation_df

    # Course-wise (specialty-wise) percentage validation (BEFORE adjustment)
    if coursewise_df is not None:
        download_data['Coursewise_Validation'] = coursewise_df
    if coursewise_summary is not None:
        download_data['Coursewise_Summary'] = coursewise_summary

    # ADJUSTED (corrected) allocation -- satisfies both overall & course-wise
    if adjusted_df is not None:
        download_data['Adjusted_Allocation'] = adjusted_df
    if adjusted_overall_df is not None:
        download_data['Adjusted_Overall_Check'] = adjusted_overall_df
    if adjusted_coursewise_df is not None:
        download_data['Adjusted_Coursewise_Check'] = adjusted_coursewise_df

    # Pivot tables
    download_data['Specialty_Category_Pivot'] = processed['specialty_category_pivot'].reset_index()
    download_data['College_Category_Pivot'] = processed['college_category_pivot'].reset_index()
    download_data['College_Specialty_Pivot'] = processed['college_specialty_pivot'].reset_index()

    return download_data

def create_excel_download(download_data):
    """Create Excel file with multiple sheets"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in download_data.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return output.getvalue()

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

        st.divider()
        st.subheader("🎯 Course-wise Tolerance")
        tolerance = st.slider(
            "Allowed deviation (percentage points)",
            min_value=0.5, max_value=10.0, value=2.0, step=0.5,
            help="A category within a course is flagged if its actual % share deviates "
                 "from the seat-matrix % by more than this many points. Categories that "
                 "deserved at least ~half a seat but got zero are always flagged as 'Missing', "
                 "regardless of tolerance."
        )

    # Main content
    if st.session_state.processed is not None:
        processed = st.session_state.processed
        data = st.session_state.data

        # Overall (grand total) validation
        validation_df = validate_allocations(data)

        # Course-wise (specialty-wise) validation
        coursewise_df = calculate_coursewise_compliance(data, tolerance_pts=tolerance)
        coursewise_summary, fully_compliant, total_courses = summarize_coursewise_compliance(coursewise_df)

        overall_ok = (validation_df['Difference'] == 0).all()
        coursewise_ok = fully_compliant == total_courses

        # Adjusted (corrected) allocation -- fixes both overall & course-wise together
        # Stage 1: apportion each course independently (largest remainder method)
        adjusted_df = calculate_adjusted_allocation(data)
        # Stage 2: rebalance across courses so the OVERALL totals also hit target
        adjusted_df = rebalance_global_allocation(adjusted_df, processed['total_seats'])
        adjusted_overall_df = validate_adjusted_overall(adjusted_df, processed['total_seats'])
        adjusted_coursewise_df = validate_adjusted_coursewise(adjusted_df, tolerance_pts=tolerance)
        adj_coursewise_summary, adj_fully_compliant, adj_total_courses = summarize_coursewise_compliance(adjusted_coursewise_df)
        adjusted_overall_ok = (adjusted_overall_df['Difference'] == 0).all()
        adjusted_coursewise_ok = adj_fully_compliant == adj_total_courses

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
            st.metric(
                "vs Expected",
                f"{processed['total_seats'] - expected_total:+d}",
                delta_color="inverse"
            )

        # Top-level compliance banner (overall AND course-wise)
        if overall_ok and coursewise_ok:
            st.success("✅ Overall totals match the seat matrix **and** every course individually satisfies the percentage criteria.")
        else:
            msgs = []
            if not overall_ok:
                msgs.append("overall grand-total category counts do not match the (proportional) seat matrix target")
            if not coursewise_ok:
                msgs.append(f"{total_courses - fully_compliant} of {total_courses} courses do not satisfy the course-wise percentage criteria")
            adj_note = "✅ satisfies both" if (adjusted_overall_ok and adjusted_coursewise_ok) else "still has minor rounding gaps"
            st.error(
                "⚠️ Compliance issue in the **original data**: " + " and ".join(msgs) + ". "
                f"See the **Validation** and **Course-wise %** tabs for details, or open the "
                f"**🔧 Adjusted Allocation** tab for a corrected version ({adj_note})."
            )

        # Tabs
        tabs = st.tabs([
            "📊 Summary",
            "📈 Category Analysis",
            "🏛️ Specialty Analysis",
            "📚 College Analysis",
            "🔍 Specialty-Category View",
            "📋 Validation (Overall)",
            "🎯 Course-wise % (Per Course)",
            "🔧 Adjusted Allocation",
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
                st.markdown("#### Top 10 Specialties")
                st.dataframe(processed['specialty_summary'].head(10), use_container_width=True)

                st.markdown("#### Top 10 Colleges")
                st.dataframe(processed['college_summary'].head(10), use_container_width=True)

        # Tab 2: Category Analysis
        with tabs[1]:
            st.subheader("Category-wise Analysis")

            # Category chart
            fig = create_category_chart(processed['category_summary'])
            st.plotly_chart(fig, use_container_width=True)

            # Category breakdown table
            st.markdown("#### Detailed Category Breakdown by Specialty")

            # Pivot table for categories
            cat_pivot = data.pivot_table(
                index='Category',
                columns='Specialty',
                values='Seats',
                fill_value=0,
                aggfunc='sum'
            )
            st.dataframe(cat_pivot, use_container_width=True)

        # Tab 3: Specialty Analysis
        with tabs[2]:
            st.subheader("Specialty-wise Analysis")

            # Specialty chart
            fig = create_specialty_chart(processed['specialty_summary'])
            st.plotly_chart(fig, use_container_width=True)

            # CS Specialty details
            st.markdown("#### CS Specialty Details")
            cs_data = data[data['Specialty'] == 'CS']
            if not cs_data.empty:
                cs_total = cs_data['Seats'].sum()
                cs_by_category = cs_data.groupby('Category')['Seats'].sum().reset_index()

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("CS Total Seats", int(cs_total))
                with col2:
                    st.metric("CS Categories", len(cs_by_category))

                st.dataframe(cs_by_category, use_container_width=True)

                # CS distribution pie chart
                fig_cs = px.pie(
                    cs_by_category,
                    values='Seats',
                    names='Category',
                    title='CS Category Distribution'
                )
                st.plotly_chart(fig_cs, use_container_width=True)
            else:
                st.warning("No data found for CS specialty")

            # All specialty-category breakdown
            st.markdown("#### All Specialty-Category Breakdown")
            st.dataframe(processed['specialty_category'], use_container_width=True)

        # Tab 4: College Analysis
        with tabs[3]:
            st.subheader("College-wise Analysis")

            # College chart
            fig = create_college_chart(processed['college_summary'])
            st.plotly_chart(fig, use_container_width=True)

            # College-Category breakdown
            st.markdown("#### College-Category Breakdown")
            st.dataframe(processed['college_category_pivot'], use_container_width=True)

            # College-Specialty breakdown
            st.markdown("#### College-Specialty Breakdown")
            st.dataframe(processed['college_specialty_pivot'], use_container_width=True)

        # Tab 5: Specialty-Category View
        with tabs[4]:
            st.subheader("Specialty vs Category Analysis")

            # Heatmap
            fig = create_specialty_category_heatmap(data)
            st.plotly_chart(fig, use_container_width=True)

            # Detailed pivot table
            st.markdown("#### Specialty-Category Pivot Table")
            st.dataframe(processed['specialty_category_pivot'], use_container_width=True)

        # Tab 6: Validation (Overall)
        with tabs[5]:
            st.subheader("Overall (Grand-Total) Validation")

            st.markdown("""
            This checks the **grand total** allocation against the expected seat matrix:
            - **SM**: 50 seats &nbsp;•&nbsp; **EW**: 10 seats &nbsp;•&nbsp; **EZ**: 9 seats &nbsp;•&nbsp;
              **MU**: 8 seats &nbsp;•&nbsp; **SC**: 8 seats &nbsp;•&nbsp; **BH**: 3 seats &nbsp;•&nbsp;
              **LA**: 3 seats &nbsp;•&nbsp; **DV**: 2 seats &nbsp;•&nbsp; **VK**: 2 seats &nbsp;•&nbsp;
              **ST**: 2 seats &nbsp;•&nbsp; **KN**: 1 seat &nbsp;•&nbsp; **BX**: 1 seat &nbsp;•&nbsp; **KU**: 1 seat

            ⚠️ Note: matching the overall total does **not** guarantee that each individual
            course/specialty respects these percentages — see the **Course-wise %** tab for that check.
            """)

            # Display validation results
            st.dataframe(
                validation_df,
                column_config={
                    'Category': 'Category',
                    'Expected': st.column_config.NumberColumn('Expected', format='%d'),
                    'Actual': st.column_config.NumberColumn('Actual', format='%d'),
                    'Difference': st.column_config.NumberColumn('Difference', format='%d'),
                    'Status': 'Status'
                },
                use_container_width=True
            )

            # Show validation summary
            all_match = (validation_df['Difference'] == 0).all()

            if all_match:
                st.success("✅ All allocations match the expected seat matrix exactly (overall)!")
            else:
                st.warning("⚠️ Some allocations differ from the expected seat matrix (overall)")

                # Show mismatches
                mismatches = validation_df[validation_df['Difference'] != 0]
                if not mismatches.empty:
                    st.markdown("#### Mismatches found:")
                    st.dataframe(
                        mismatches[['Category', 'Expected', 'Actual', 'Difference']],
                        use_container_width=True
                    )

        # Tab 7: Course-wise Percentage Validation
        with tabs[6]:
            st.subheader("Course-wise (Per-Specialty) Percentage Validation")
            st.markdown("""
            The seat-matrix percentages must be satisfied **within each course/specialty individually**,
            not just in the grand total. For every specialty, expected seats per category =
            `course_total × matrix_% / 100`. A category is flagged **Missing** if it deserved at
            least ~half a seat but got zero, or **⚠️** if its actual share deviates from the
            expected share by more than the tolerance set in the sidebar (currently **{:.1f} pts**).
            """.format(tolerance))

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Courses", total_courses)
            with col2:
                st.metric("Fully Compliant", fully_compliant)
            with col3:
                st.metric("Courses w/ Violations", total_courses - fully_compliant)
            with col4:
                total_missing = int((coursewise_df['Status'] == 'Missing').sum())
                st.metric("Missing-Category Instances", total_missing)

            st.markdown("#### Compliance by Course")
            st.dataframe(
                coursewise_summary.rename(columns={
                    'Course_Total': 'Total Seats',
                    'Category_Violations': 'Category Violations',
                    'Missing_Mandatory_Categories': 'Missing Mandatory Categories'
                }),
                use_container_width=True
            )

            st.markdown("#### Detailed Category-Level Breakdown")
            filter_option = st.radio(
                "Show:",
                ["All", "Only Violations (⚠️ + Missing)", "Only Missing Mandatory"],
                horizontal=True
            )

            display_df = coursewise_df.copy()
            if filter_option == "Only Violations (⚠️ + Missing)":
                display_df = display_df[display_df['Status'] != '✅']
            elif filter_option == "Only Missing Mandatory":
                display_df = display_df[display_df['Status'] == 'Missing']

            st.dataframe(display_df, use_container_width=True, height=400)

            # Heatmap of deviation
            st.markdown("#### Deviation Heatmap (Actual % − Expected %)")
            deviation_pivot = coursewise_df.pivot_table(
                index='Specialty', columns='Category', values='Deviation (pts)', fill_value=0
            )
            specialty_order = data.groupby('Specialty')['Seats'].sum().sort_values(ascending=False).index
            deviation_pivot = deviation_pivot.loc[[s for s in specialty_order if s in deviation_pivot.index]]

            fig_dev = px.imshow(
                deviation_pivot,
                title='Percentage-Point Deviation by Course & Category',
                color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0,
                text_auto='.1f',
                aspect='auto',
                height=max(400, len(deviation_pivot.index) * 25)
            )
            fig_dev.update_layout(xaxis_title='Category', yaxis_title='Specialty')
            st.plotly_chart(fig_dev, use_container_width=True)

            if fully_compliant < total_courses:
                st.warning(
                    f"⚠️ {total_courses - fully_compliant} out of {total_courses} courses do not satisfy "
                    f"the course-wise percentage criteria within the {tolerance} pt tolerance."
                )
            else:
                st.success("✅ All courses satisfy the course-wise percentage criteria!")

        # Tab 8: Adjusted Allocation (corrected to satisfy BOTH overall & course-wise)
        with tabs[7]:
            st.subheader("🔧 Adjusted (Corrected) Allocation")
            st.markdown("""
            This shows a **corrected category allocation**, built in two stages:

            1. **Per-course apportionment** (Largest Remainder / Hamilton method) — each course's
               seats are re-split across categories to match its own percentage targets as closely
               as integers allow. Course totals never change.
            2. **Global rebalancing pass** — small courses (1-7 seats) can't individually reflect
               all 13 categories, which biases the grand totals even after step 1. This pass swaps
               single seats between categories *within the same course*, prioritizing swaps that
               also improve that course's own fit, until the overall grand totals hit their
               proportional target too.

            The result satisfies **both** the overall check and the course-wise check together,
            as closely as the integer seat counts allow.

            ⚠️ **Scope**: this only rebalances *category totals within each course*. It does **not**
            decide which specific college receives which seat — that requires college-level
            capacity/eligibility rules that aren't present in this dataset.
            """)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Check (Adjusted)", "✅ Pass" if adjusted_overall_ok else "⚠️ Off")
            with col2:
                st.metric("Courses Fully Compliant", f"{adj_fully_compliant} / {adj_total_courses}")
            with col3:
                total_seats_moved = int(adjusted_df[adjusted_df['Change'] != 0]['Change'].abs().sum()) // 2
                st.metric("Seats Re-categorized", total_seats_moved)
            with col4:
                courses_changed = int(adjusted_df[adjusted_df['Change'] != 0]['Specialty'].nunique())
                st.metric("Courses Affected", courses_changed)

            st.markdown("#### Overall Check: Original vs Adjusted")
            colA, colB = st.columns(2)
            with colA:
                st.markdown("**Before (Original Data)**")
                st.dataframe(validation_df, use_container_width=True)
            with colB:
                st.markdown("**After (Adjusted Data)**")
                st.dataframe(adjusted_overall_df, use_container_width=True)

            st.markdown("#### Course-wise Check on Adjusted Data")
            if adj_fully_compliant == adj_total_courses:
                st.success(f"✅ All {adj_total_courses} courses now satisfy the percentage criteria within {tolerance} pts.")
            else:
                st.info(
                    f"{adj_total_courses - adj_fully_compliant} of {adj_total_courses} courses still show a minor "
                    f"deviation beyond {tolerance} pts (this is the smallest possible gap given integer seat counts "
                    f"— e.g. a course with only 1-2 seats can't precisely reflect a 1% category)."
                )
            st.dataframe(adj_coursewise_summary.rename(columns={
                'Course_Total': 'Total Seats',
                'Category_Violations': 'Category Violations',
                'Missing_Mandatory_Categories': 'Missing Mandatory Categories'
            }), use_container_width=True)

            st.markdown("#### Detailed Adjusted Allocation (Original → Adjusted)")
            filter_changed = st.checkbox("Show only categories that changed", value=True)
            display_adj = adjusted_df.copy()
            if filter_changed:
                display_adj = display_adj[display_adj['Change'] != 0]
            st.dataframe(display_adj, use_container_width=True, height=400)

            # Chart: net change by category, summed across all courses
            st.markdown("#### Net Seat Change by Category (summed across all courses)")
            change_by_cat = adjusted_df.groupby('Category')['Change'].sum().reset_index().sort_values('Change')
            fig_change = px.bar(
                change_by_cat, x='Change', y='Category', orientation='h',
                color='Change', color_continuous_scale='RdYlGn',
                title='Categories gaining (green) vs losing (red) seats after adjustment'
            )
            fig_change.update_layout(height=400)
            st.plotly_chart(fig_change, use_container_width=True)

            st.markdown("#### Download Adjusted Allocation")
            csv_adj = adjusted_df.to_csv(index=False)
            st.download_button(
                "📥 Download Adjusted Allocation (CSV)",
                csv_adj,
                f"adjusted_allocation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )

        # Tab 9: Download
        with tabs[8]:
            st.subheader("📥 Download Results")

            st.markdown("Download the data in various formats:")

            # Create download data (includes overall, course-wise, AND adjusted allocation)
            download_data = create_download_data(
                data, processed, validation_df,
                coursewise_df=coursewise_df,
                coursewise_summary=coursewise_summary,
                adjusted_df=adjusted_df,
                adjusted_overall_df=adjusted_overall_df,
                adjusted_coursewise_df=adjusted_coursewise_df
            )

            # Excel download
            st.markdown("#### Download as Excel")
            excel_bytes = create_excel_download(download_data)
            st.download_button(
                "📥 Download Full Report (Excel, incl. Adjusted Allocation)",
                excel_bytes,
                f"seat_allocation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Download Original Data")
                csv = data.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    f"seat_allocation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )

            with col2:
                st.markdown("#### Download as JSON")
                json_data = {
                    'timestamp': datetime.now().isoformat(),
                    'seat_matrix': SEAT_MATRIX,
                    'total_seats': int(processed['total_seats']),
                    'coursewise_tolerance_pts': tolerance,
                    'coursewise_fully_compliant_courses': fully_compliant,
                    'coursewise_total_courses': total_courses,
                    'summary': {
                        'category_summary': processed['category_summary'].to_dict('records'),
                        'specialty_summary': processed['specialty_summary'].to_dict('records'),
                        'college_summary': processed['college_summary'].to_dict('records')
                    },
                    'overall_validation': validation_df.to_dict('records'),
                    'coursewise_validation': coursewise_df.to_dict('records'),
                    'adjusted_allocation': adjusted_df.to_dict('records'),
                    'adjusted_overall_check': adjusted_overall_df.to_dict('records'),
                    'data': data.to_dict('records')
                }

                # Convert numpy types to Python native types
                def convert_to_serializable(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_to_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_to_serializable(item) for item in obj]
                    else:
                        return obj

                json_data = convert_to_serializable(json_data)
                json_str = json.dumps(json_data, indent=2)

                st.download_button(
                    "📥 Download as JSON",
                    json_str,
                    f"seat_allocation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json",
                    use_container_width=True
                )

            # Download individual sheets
            st.markdown("#### Download Individual Reports")
            cols = st.columns(3)

            with cols[0]:
                # Category Summary
                csv_cat = processed['category_summary'].to_csv(index=False)
                st.download_button(
                    "📥 Category Summary",
                    csv_cat,
                    f"category_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                    use_container_width=True
                )

            with cols[1]:
                # Specialty Summary
                csv_spec = processed['specialty_summary'].to_csv(index=False)
                st.download_button(
                    "📥 Specialty Summary",
                    csv_spec,
                    f"specialty_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                    use_container_width=True
                )

            with cols[2]:
                # College Summary
                csv_col = processed['college_summary'].to_csv(index=False)
                st.download_button(
                    "📥 College Summary",
                    csv_col,
                    f"college_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                    use_container_width=True
                )

            cols2 = st.columns(3)

            with cols2[0]:
                # Overall Validation Report
                csv_val = validation_df.to_csv(index=False)
                st.download_button(
                    "📥 Overall Validation Report",
                    csv_val,
                    f"overall_validation_report_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                    use_container_width=True
                )

            with cols2[1]:
                # Course-wise Validation Report
                csv_cw = coursewise_df.to_csv(index=False)
                st.download_button(
                    "📥 Course-wise Validation Report",
                    csv_cw,
                    f"coursewise_validation_report_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                    use_container_width=True
                )

            with cols2[2]:
                # Specialty-Category Pivot
                csv_sc = processed['specialty_category_pivot'].reset_index().to_csv(index=False)
                st.download_button(
                    "📥 Specialty-Category Pivot",
                    csv_sc,
                    f"specialty_category_pivot_{datetime.now().strftime('%Y%m%d')}.csv",
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
        4. **Export**: Download results in CSV, Excel, or JSON format

        ### 📁 Required Data Format

        Your CSV should have these columns:
        - **Program**: Program code (e.g., E)
        - **Specialty**: Specialty name (e.g., CS, ME, EC)
        - **College**: College name
        - **Type**: Type (e.g., G)
        - **Category**: Seat category (SM, EW, EZ, etc.)
        - **Seats**: Number of seats

        ### 🎯 Expected Seat Matrix (also the target % per category)

        - **SM**: 50 seats (50%)
        - **EW**: 10 seats (10%)
        - **EZ**: 9 seats (9%)
        - **MU**: 8 seats (8%)
        - **SC**: 8 seats (8%)
        - **BH**: 3 seats (3%)
        - **LA**: 3 seats (3%)
        - **DV**: 2 seats (2%)
        - **VK**: 2 seats (2%)
        - **ST**: 2 seats (2%)
        - **KN**: 1 seat (1%)
        - **BX**: 1 seat (1%)
        - **KU**: 1 seat (1%)

        ### ✅ Two Levels of Validation

        This app checks compliance at **two** levels:
        1. **Overall** — do the grand totals across all courses match the seat matrix counts? (Validation tab)
        2. **Course-wise** — does *each individual course/specialty* also respect these percentages,
           within an adjustable tolerance? (Course-wise % tab) — a course can pass #1 while badly
           failing #2, e.g. one course over-allocating EW/EZ while giving zero seats to VK/KN/DV.
        """)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
