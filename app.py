"""
app.py
------
EduPath AI: Intelligent Student Performance & Personalized Learning System
Interactive Streamlit Web Dashboard.

DATANYX'26 Hackathon Prototype
------------------------------
Features:
- Student Assessment & Growth-Oriented Personalized Action Plans (No Negative Labels)
- Early-Warning Mode Toggle (Pre-Semester vs. Mid-Semester with G1/G2)
- Transparent Explainable AI (Global & Student-Specific Feature Importance)
- Model Performance Benchmarking (Random Forest vs. Decision Tree vs. Linear Regression)
- Interactive Exploratory Data Analysis (EDA)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Import ML pipeline functions
from model import (
    load_student_data,
    train_and_evaluate_pipeline,
    predict_student_performance,
    get_top_feature_importances,
    explain_student_prediction,
    generate_personalized_recommendations,
    EduPathModelBundle
)

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & STYLING
# ==============================================================================

st.set_page_config(
    page_title="EduPath AI | Student Performance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern card designs, badges, and clean typography
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: #FFFFFF !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .main-header p {
        color: #E0E7FF;
        font-size: 1.05rem;
        margin-bottom: 0px;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-title {
        color: #6B7280;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #111827;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    
    /* Performance status badge */
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 8px;
    }
    
    /* Recommendation Card */
    .rec-card {
        background-color: #F9FAFB;
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        margin-bottom: 12px;
        border-radius: 4px;
    }
    
    /* Hackathon badge */
    .badge-tag {
        background-color: rgba(255, 255, 255, 0.2);
        color: #FFFFFF;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# DATA & MODEL CACHING
# ==============================================================================

@st.cache_data
def get_cached_data() -> pd.DataFrame:
    """Load and cache the student performance dataset."""
    return load_student_data("data/student-mat.csv")


@st.cache_resource
def get_cached_model_bundle(include_grades: bool) -> EduPathModelBundle:
    """Train and cache the ML pipeline bundle for the chosen grade mode."""
    df = get_cached_data()
    return train_and_evaluate_pipeline(df, include_grades=include_grades)


# ==============================================================================
# SIDEBAR CONTROLS & CONFIGURATION
# ==============================================================================

st.sidebar.image("https://images.unsplash.com/photo-1503676260728-1c00da094a0b?w=400&auto=format&fit=crop&q=60", use_container_width=True)
st.sidebar.markdown("## ⚙️ System Settings")

# 1. Early-Warning Mode Toggle
grade_mode = st.sidebar.radio(
    "Prediction Horizon / Mode:",
    options=[
        "🛡️ Early-Warning Mode (Pre-Semester)",
        "📊 Mid-Semester Mode (With G1 & G2)"
    ],
    index=0,
    help=(
        "Early-Warning Mode forecasts outcomes before interim exams take place, "
        "allowing proactive counseling. Mid-Semester Mode includes G1 and G2 for maximum precision."
    )
)
include_grades = "With G1 & G2" in grade_mode

# 2. Model Selection
model_choice = st.sidebar.selectbox(
    "Active ML Algorithm:",
    options=[
        "Random Forest Regressor",
        "Decision Tree Regressor",
        "Linear Regression"
    ],
    index=0,
    help="Select the underlying machine learning algorithm used for predictions."
)

# 3. Quick-Fill Preset Student Profiles (Crowd-pleaser for Hackathon Demos!)
st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 Demo Student Profiles")
profile_selection = st.sidebar.selectbox(
    "Quick-fill profile:",
    options=[
        "Custom (Manual Input)",
        "🌟 Maya - High Achiever",
        "📈 Alex - Solid Average",
        "💡 Jordan - High Support Need",
        "🚀 Sam - Rising Comeback"
    ],
    index=0
)

# Load data and model bundle
raw_df = get_cached_data()
bundle = get_cached_model_bundle(include_grades=include_grades)

# Sidebar Quick Dataset Info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Telemetry")
st.sidebar.write(f"• **Records:** {raw_df.shape[0]} students")
st.sidebar.write(f"• **Attributes:** {raw_df.shape[1]} features")
st.sidebar.write(f"• **Target Scale:** 0 – 20 points (G3)")
st.sidebar.caption("DATANYX'26 Hackathon Prototype | v1.0")


# ==============================================================================
# HEADER HERO BANNER
# ==============================================================================

st.markdown("""
<div class="main-header">
    <div>
        <span class="badge-tag">DATANYX'26 HACKATHON</span>
        <span class="badge-tag">EXPLAINABLE AI</span>
        <span class="badge-tag">GROWTH MINDSET</span>
    </div>
    <h1>🎓 EduPath AI</h1>
    <p>Intelligent Student Performance Prediction & Personalized Learning System</p>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# PRESET PROFILE VALUES
# ==============================================================================

# Default values
default_vals = {
    'studytime': 2,
    'absences': 4,
    'failures': 0,
    'schoolsup': 'no',
    'famsup': 'yes',
    'paid': 'no',
    'higher': 'yes',
    'internet': 'yes',
    'freetime': 3,
    'goout': 3,
    'health': 4,
    'Medu': 2,
    'Fedu': 2,
    'G1': 11,
    'G2': 11
}

if profile_selection == "🌟 Maya - High Achiever":
    default_vals.update({
        'studytime': 4, 'absences': 1, 'failures': 0,
        'schoolsup': 'no', 'famsup': 'yes', 'paid': 'yes', 'higher': 'yes',
        'internet': 'yes', 'freetime': 2, 'goout': 2, 'health': 5,
        'Medu': 4, 'Fedu': 4, 'G1': 16, 'G2': 17
    })
elif profile_selection == "📈 Alex - Solid Average":
    default_vals.update({
        'studytime': 2, 'absences': 6, 'failures': 0,
        'schoolsup': 'no', 'famsup': 'yes', 'paid': 'no', 'higher': 'yes',
        'internet': 'yes', 'freetime': 3, 'goout': 3, 'health': 4,
        'Medu': 2, 'Fedu': 2, 'G1': 12, 'G2': 12
    })
elif profile_selection == "💡 Jordan - High Support Need":
    default_vals.update({
        'studytime': 1, 'absences': 16, 'failures': 2,
        'schoolsup': 'yes', 'famsup': 'no', 'paid': 'no', 'higher': 'yes',
        'internet': 'no', 'freetime': 4, 'goout': 4, 'health': 2,
        'Medu': 1, 'Fedu': 1, 'G1': 7, 'G2': 8
    })
elif profile_selection == "🚀 Sam - Rising Comeback":
    default_vals.update({
        'studytime': 3, 'absences': 5, 'failures': 1,
        'schoolsup': 'yes', 'famsup': 'yes', 'paid': 'yes', 'higher': 'yes',
        'internet': 'yes', 'freetime': 2, 'goout': 2, 'health': 4,
        'Medu': 3, 'Fedu': 2, 'G1': 8, 'G2': 14
    })


# ==============================================================================
# MAIN TABS ARCHITECTURE
# ==============================================================================

tab_predict, tab_explain, tab_eda = st.tabs([
    "🎯 Student Assessment & Learning Plan",
    "🔍 Explainability & AI Insights",
    "📊 Model Performance & EDA"
])


# ------------------------------------------------------------------------------
# TAB 1: STUDENT INPUT & PERSONALIZED LEARNING RECOMMENDATION
# ------------------------------------------------------------------------------
with tab_predict:
    st.markdown("### 📝 Student Academic Profile Input")
    st.info(
        f"**Active Mode:** {'🛡️ **Early-Warning Mode** (Forecasts final outcome before interim exams G1/G2)' if not include_grades else '📊 **Mid-Semester Mode** (Leverages G1 and G2 interim exam grades)'} | "
        f"**Algorithm:** `{model_choice}`"
    )
    
    with st.form("student_assessment_form"):
        col_habits, col_support, col_personal = st.columns(3)
        
        # Column 1: Academic Habits
        with col_habits:
            st.markdown("##### 📚 Study Habits & Attendance")
            studytime_choice = st.selectbox(
                "Weekly Study Time:",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "1. Minimal (< 2 hours/week)",
                    2: "2. Moderate (2 to 5 hours/week)",
                    3: "3. Diligent (5 to 10 hours/week)",
                    4: "4. Intensive (> 10 hours/week)"
                }[x],
                index=[1, 2, 3, 4].index(default_vals['studytime']),
                help="Typical weekly study hours dedicated outside regular school classes."
            )
            
            absences_input = st.number_input(
                "Total School Absences:",
                min_value=0,
                max_value=75,
                value=int(default_vals['absences']),
                step=1,
                help="Number of days the student was absent from school during the semester."
            )
            
            failures_input = st.selectbox(
                "Past Course Challenges / Failures:",
                options=[0, 1, 2, 3],
                format_func=lambda x: f"{x} prior failure(s)" if x > 0 else "0 (No prior setbacks)",
                index=default_vals['failures'],
                help="Number of previous class failures or challenges."
            )

        # Column 2: Academic Support & Ambitions
        with col_support:
            st.markdown("##### 🤝 Support Systems & Ambition")
            schoolsup_val = st.selectbox(
                "School Extra Support Program:",
                options=['no', 'yes'],
                index=0 if default_vals['schoolsup'] == 'no' else 1,
                format_func=lambda x: "Yes (Enrolled in school coaching)" if x == 'yes' else "No"
            )
            
            famsup_val = st.selectbox(
                "Family Educational Support:",
                options=['no', 'yes'],
                index=0 if default_vals['famsup'] == 'no' else 1,
                format_func=lambda x: "Yes (Family assists with studies)" if x == 'yes' else "No"
            )
            
            paid_val = st.selectbox(
                "Extra Paid Tutoring:",
                options=['no', 'yes'],
                index=0 if default_vals['paid'] == 'no' else 1,
                format_func=lambda x: "Yes (Attending private tutoring)" if x == 'yes' else "No"
            )
            
            higher_val = st.selectbox(
                "Aspiration for Higher Education:",
                options=['yes', 'no'],
                index=0 if default_vals['higher'] == 'yes' else 1,
                format_func=lambda x: "Yes (Plans university/college)" if x == 'yes' else "No"
            )
            
            internet_val = st.selectbox(
                "Internet Access at Home:",
                options=['yes', 'no'],
                index=0 if default_vals['internet'] == 'yes' else 1,
                format_func=lambda x: "Yes (Broadband / Wi-Fi)" if x == 'yes' else "No (Limited/None)"
            )

        # Column 3: Personal & Well-Being
        with col_personal:
            st.markdown("##### 🧘 Well-Being & Environment")
            health_val = st.slider(
                "Current Health Status (1-5):",
                min_value=1,
                max_value=5,
                value=int(default_vals['health']),
                help="1 = Frequent illness, 5 = Excellent physical well-being."
            )
            
            freetime_val = st.slider(
                "Free Time After School (1-5):",
                min_value=1,
                max_value=5,
                value=int(default_vals['freetime']),
                help="1 = Very low, 5 = Very high."
            )
            
            goout_val = st.slider(
                "Going Out with Friends (1-5):",
                min_value=1,
                max_value=5,
                value=int(default_vals['goout']),
                help="1 = Rarely, 5 = Very frequently."
            )
            
            medu_val = st.selectbox(
                "Mother's Education Level:",
                options=[0, 1, 2, 3, 4],
                format_func=lambda x: {
                    0: "None",
                    1: "Primary (4th grade)",
                    2: "5th to 9th grade",
                    3: "Secondary school",
                    4: "Higher education"
                }[x],
                index=default_vals['Medu']
            )

        # Conditional Interim Grades (Only shown in Mid-Semester Mode)
        if include_grades:
            st.markdown("---")
            st.markdown("##### 📋 Interim Exam Scores (Scale: 0 to 20)")
            g_col1, g_col2 = st.columns(2)
            with g_col1:
                g1_val = st.slider(
                    "First Period Grade (G1):",
                    min_value=0, max_value=20,
                    value=int(default_vals['G1']),
                    help="First trimester examination score out of 20."
                )
            with g_col2:
                g2_val = st.slider(
                    "Second Period Grade (G2):",
                    min_value=0, max_value=20,
                    value=int(default_vals['G2']),
                    help="Second trimester examination score out of 20."
                )
        else:
            g1_val, g2_val = 10, 10
            
        submitted = st.form_submit_button("🚀 Analyze Student & Generate Learning Path", type="primary", use_container_width=True)

    # Prepare student dictionary
    current_student = {
        'school': 'GP',
        'sex': 'F',
        'age': 16,
        'address': 'U',
        'famsize': 'GT3',
        'Pstatus': 'T',
        'Medu': medu_val,
        'Fedu': default_vals['Fedu'],
        'Mjob': 'services',
        'Fjob': 'services',
        'reason': 'course',
        'guardian': 'mother',
        'traveltime': 1,
        'studytime': studytime_choice,
        'failures': failures_input,
        'schoolsup': schoolsup_val,
        'famsup': famsup_val,
        'paid': paid_val,
        'activities': 'yes',
        'nursery': 'yes',
        'higher': higher_val,
        'internet': internet_val,
        'romantic': 'no',
        'famrel': 4,
        'freetime': freetime_val,
        'goout': goout_val,
        'Dalc': 1,
        'Walc': 1,
        'health': health_val,
        'absences': absences_input,
        'G1': g1_val,
        'G2': g2_val
    }

    # Run Prediction
    pred_grade, status_label, badge_emoji, badge_color = predict_student_performance(
        bundle=bundle,
        student_dict=current_student,
        algorithm_name=model_choice
    )

    # Convert to standard percentage and letter grade
    percent_score = round((pred_grade / 20.0) * 100, 1)
    if pred_grade >= 16.0:
        letter_grade = "A (Excellent)"
    elif pred_grade >= 14.0:
        letter_grade = "B (Good)"
    elif pred_grade >= 12.0:
        letter_grade = "C (Satisfactory)"
    elif pred_grade >= 10.0:
        letter_grade = "D (Passing)"
    else:
        letter_grade = "Growth Opportunity (< 10)"

    st.markdown("---")
    st.markdown("### 🏆 Academic Assessment Results")

    # Metrics Row
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Predicted Final Grade (G3)</div>
            <div class="metric-value" style="color: {badge_color};">{pred_grade} <span style="font-size: 1.1rem; color: #6B7280;">/ 20</span></div>
            <div style="font-size: 0.9rem; color: #4B5563; margin-top: 4px;">Equivalent: <strong>{percent_score}%</strong> ({letter_grade})</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Performance Trajectory</div>
            <div class="metric-value">{badge_emoji}</div>
            <div class="status-badge" style="background-color: {badge_color}20; color: {badge_color}; border: 1px solid {badge_color};">
                {status_label}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col3:
        mae_metric = bundle.metrics.get(model_choice, {}).get("MAE", 1.5)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Model Confidence Window</div>
            <div class="metric-value">±{mae_metric} <span style="font-size: 1.1rem; color: #6B7280;">pts</span></div>
            <div style="font-size: 0.85rem; color: #6B7280; margin-top: 4px;">Expected Range: <strong>{max(0.0, round(pred_grade - mae_metric, 1))} – {min(20.0, round(pred_grade + mae_metric, 1))}</strong></div>
        </div>
        """, unsafe_allow_html=True)

    # Supportive, growth-mindset narrative
    explanation_meta = explain_student_prediction(current_student, pred_grade, bundle)
    st.markdown(f"""
    <div style="background-color: #EFF6FF; border-left: 4px solid #3B82F6; padding: 14px 18px; border-radius: 6px; margin-top: 16px;">
        <strong style="color: #1E40AF;">💡 AI Counselor Insight:</strong> {explanation_meta['summary']}
    </div>
    """, unsafe_allow_html=True)

    # Personalized Learning Recommendations Table
    st.markdown("#### 🌟 Personalized Learning Action Plan")
    st.caption("Actionable, supportive developmental strategies tailored to this student's unique academic habits.")

    recommendations = generate_personalized_recommendations(current_student, pred_grade, include_grades=include_grades)
    rec_df = pd.DataFrame(recommendations)

    # Style table with priority coloring
    def color_priority(val):
        if 'High' in str(val):
            return 'background-color: #FEE2E2; color: #991B1B; font-weight: bold;'
        elif 'Medium' in str(val):
            return 'background-color: #FEF3C7; color: #92400E;'
        else:
            return 'background-color: #D1FAE5; color: #065F46;'

    styled_rec = rec_df.style.map(color_priority, subset=['Priority'])
    st.dataframe(styled_rec, use_container_width=True, hide_index=True)

    # Downloadable Action Plan
    report_text = f"""EduPath AI: Personalized Learning Action Plan
=============================================
Student Profile: {profile_selection}
Assessment Horizon: {'Early-Warning Mode (Pre-Semester)' if not include_grades else 'Mid-Semester Mode'}
Predicted Final Grade: {pred_grade}/20 ({percent_score}%)
Classification: {status_label}
Model Used: {model_choice} (Expected margin: ±{mae_metric} pts)

Actionable Learning Recommendations:
---------------------------------------------
"""
    for idx, r in enumerate(recommendations, 1):
        report_text += f"{idx}. [{r['Priority']} Priority] {r['Focus Area']}\n"
        report_text += f"   Indicator: {r['Current Indicator']}\n"
        report_text += f"   Action: {r['Actionable Strategy']}\n\n"

    st.download_button(
        label="📥 Download Personalized Student Action Plan (.txt)",
        data=report_text,
        file_name=f"EduPath_Learning_Plan_{profile_selection.replace(' ', '_')}.txt",
        mime="text/plain"
    )


# ------------------------------------------------------------------------------
# TAB 2: EXPLAINABILITY & AI INSIGHTS
# ------------------------------------------------------------------------------
with tab_explain:
    st.markdown("### 🔍 Model Explainability & Attribution")
    st.markdown(
        "Explainable AI (XAI) ensures that predictions are **transparent, fair, and actionable**. "
        "Educators and students can inspect what variables drove the prediction."
    )
    
    col_feat_chart, col_feat_breakdown = st.columns([1.1, 0.9])
    
    with col_feat_chart:
        st.markdown("##### 🌐 Global Feature Importance (Random Forest)")
        st.caption("Which factors exert the strongest overall weight across the student population?")
        
        top_importances = get_top_feature_importances(bundle, top_n=10)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=top_importances,
            x='Importance',
            y='Feature',
            palette="Blues_r",
            ax=ax
        )
        ax.set_title("Top 10 Influential Features for Final Grade", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Relative Importance Score", fontsize=10)
        ax.set_ylabel("")
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(
                f'{width:.2f}',
                (width, p.get_y() + p.get_height() / 2.),
                ha='left', va='center',
                xytext=(5, 0),
                textcoords='offset points',
                fontsize=9
            )
        plt.tight_layout()
        st.pyplot(fig)
        
    with col_feat_breakdown:
        st.markdown("##### 🎯 Student-Specific Attribution")
        st.caption(f"Deconstructing the forecast for: **{profile_selection}**")
        
        expl = explain_student_prediction(current_student, pred_grade, bundle)
        
        st.markdown("**✅ Positive Academic Catalysts (Strengths):**")
        for s in expl['strengths']:
            st.markdown(f"- 🟢 {s}")
            
        st.markdown("**🌱 High-Leverage Growth Opportunities:**")
        for g in expl['growth_areas']:
            st.markdown(f"- 🟠 {g}")
            
        st.markdown("---")
        st.markdown("##### 📐 Cohort Benchmark Comparison")
        c1, c2 = st.columns(2)
        with c1:
            cohort_study = round(raw_df['studytime'].mean(), 1)
            delta_study = current_student['studytime'] - cohort_study
            st.metric(
                "Study Time Tier",
                f"Level {current_student['studytime']}",
                f"{delta_study:+.1f} vs avg ({cohort_study})"
            )
        with c2:
            cohort_absences = round(raw_df['absences'].mean(), 1)
            delta_abs = current_student['absences'] - cohort_absences
            st.metric(
                "Absences",
                f"{current_student['absences']} days",
                f"{delta_abs:+.1f} vs avg ({cohort_absences})",
                delta_color="inverse"
            )


# ------------------------------------------------------------------------------
# TAB 3: MODEL PERFORMANCE & EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------------------------
with tab_eda:
    st.markdown("### 📊 Model Performance & Statistical EDA")
    
    # 1. Model Evaluation Metrics
    st.markdown("##### 🧪 80/20 Train-Test Evaluation")
    metrics_summary = []
    for model_name, m in bundle.metrics.items():
        metrics_summary.append({
            "Algorithm": model_name,
            "Mean Absolute Error (MAE)": m['MAE'],
            "Root Mean Squared Error (RMSE)": m['RMSE'],
            "R² Score (Variance Explained)": m['R2']
        })
    metrics_df = pd.DataFrame(metrics_summary)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    st.caption(
        f"**Note on Grade Mode:** {'In Early-Warning Mode, models explain underlying behavioral trends without relying on exam memorization.' if not include_grades else 'In Mid-Semester Mode, interim exam correlations (G1, G2) achieve high R² scores.'}"
    )

    st.markdown("---")
    st.markdown("##### 📈 Exploratory Data Analysis (EDA)")

    # 4 Visualizations in a 2x2 grid
    grid_col1, grid_col2 = st.columns(2)
    
    with grid_col1:
        # Plot 1: Target Grade Distribution
        fig1, ax1 = plt.subplots(figsize=(6, 3.8))
        sns.histplot(raw_df['G3'], bins=15, kde=True, color="#3B82F6", ax=ax1)
        ax1.axvline(raw_df['G3'].mean(), color='red', linestyle='--', label=f"Mean: {raw_df['G3'].mean():.1f}")
        ax1.set_title("Distribution of Final Grades (G3)", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Final Grade (0 to 20)")
        ax1.set_ylabel("Student Count")
        ax1.legend()
        plt.tight_layout()
        st.pyplot(fig1)
        
        # Plot 2: Study Time vs Final Grade Boxplot
        fig2, ax2 = plt.subplots(figsize=(6, 3.8))
        sns.boxplot(data=raw_df, x='studytime', y='G3', palette="Blues", ax=ax2)
        ax2.set_title("Weekly Study Time Tier vs Final Grade", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Study Time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)")
        ax2.set_ylabel("Final Grade (G3)")
        plt.tight_layout()
        st.pyplot(fig2)

    with grid_col2:
        # Plot 3: Absences vs Final Grade Scatter
        fig3, ax3 = plt.subplots(figsize=(6, 3.8))
        sns.regplot(
            data=raw_df, x='absences', y='G3',
            scatter_kws={'alpha': 0.4, 'color': '#2563EB'},
            line_kws={'color': '#DC2626'},
            ax=ax3
        )
        ax3.set_title("Absences vs. Final Grade Trendline", fontsize=11, fontweight='bold')
        ax3.set_xlabel("Number of School Absences")
        ax3.set_ylabel("Final Grade (G3)")
        plt.tight_layout()
        st.pyplot(fig3)
        
        # Plot 4: Correlation Heatmap of Key Indicators
        fig4, ax4 = plt.subplots(figsize=(6, 3.8))
        corr_cols = ['studytime', 'failures', 'absences', 'Medu', 'famrel', 'health', 'G1', 'G2', 'G3']
        corr_matrix = raw_df[corr_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=False, ax=ax4, annot_kws={"size": 7.5})
        ax4.set_title("Feature Correlation Matrix", fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig4)

    # Dataset Explorer & Summary Stats
    st.markdown("---")
    st.markdown("##### 🗄️ Dataset Browser & Summary Statistics")
    
    with st.expander("🔍 Click to inspect raw student dataset & summary statistics"):
        row_limit = st.slider("Rows to preview:", min_value=5, max_value=50, value=10)
        st.dataframe(raw_df.head(row_limit), use_container_width=True)
        st.markdown("**Descriptive Statistics:**")
        st.dataframe(raw_df.describe().round(2), use_container_width=True)


# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #9CA3AF; font-size: 0.85rem; padding-bottom: 20px;'>"
    "EduPath AI Prototype • DATANYX'26 Hackathon • Built with Python, Scikit-Learn, Streamlit & Seaborn<br>"
    "Empowering educators with explainable AI and encouraging growth-mindset student interventions."
    "</div>",
    unsafe_allow_html=True
)
