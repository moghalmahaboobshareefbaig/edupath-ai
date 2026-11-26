"""
model.py
--------
EduPath AI: Intelligent Student Performance & Personalized Learning System
Machine Learning Pipeline & Recommendation Engine.

Designed for Hackathon (DATANYX'26) - Beginner Friendly, Modular, and Robust.

Key Capabilities:
1. Data Ingestion: Loads UCI student performance dataset (student-mat.csv) or auto-generates a realistic fallback.
2. Cleaning & Preprocessing: Handles missing values, duplicates, and encodes categorical features using ColumnTransformer.
3. Early-Warning Mode: Train with or without interim grades (G1, G2) for timely academic interventions.
4. Model Training & Comparison: Random Forest Regressor (primary), Decision Tree, and Linear Regression.
5. Evaluation: Computes MAE, RMSE, and R² scores on an 80/20 train-test split.
6. Explainability: Extracts global feature importances and provides student-specific attribution breakdown.
7. Growth-Mindset Recommendation Engine: Actionable, empathetic learning strategies (no negative labels).
"""

import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
try:
    from sklearn.model_selection import train_test_split
except Exception:
    from sklearn.model_selection._split import train_test_split

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

try:
    from sklearn.metrics._regression import mean_absolute_error, mean_squared_error, r2_score
except Exception:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import fallback generator from data directory
try:
    from data.generate_data import ensure_dataset_exists
except ImportError:
    # If run from within data or alternate directory structure
    from generate_data import ensure_dataset_exists


# ==============================================================================
# 1. DATA LOADING & VALIDATION
# ==============================================================================

def load_student_data(file_path: str = "data/student-mat.csv") -> pd.DataFrame:
    """
    Load the student performance dataset from disk. If the CSV file is not
    present, automatically generates a statistically realistic dataset.
    
    Supports both semicolon (';') separated UCI format and standard comma (',') format.
    
    Parameters:
        file_path (str): Path to the student dataset CSV.
        
    Returns:
        pd.DataFrame: Cleaned Pandas DataFrame.
    """
    # Auto-generate or locate dataset
    df = ensure_dataset_exists(data_path=file_path)
    
    # Make a copy to avoid mutating cache
    df = df.copy()
    
    # Basic data hygiene: Remove potential exact duplicates
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    if len(df) < initial_rows:
        print(f"[EduPath AI] Removed {initial_rows - len(df)} duplicate records.")
        
    return df


# ==============================================================================
# 2. FEATURE ENGINEERING & PREPROCESSING PIPELINE
# ==============================================================================

def get_feature_lists(include_grades: bool = False) -> Tuple[List[str], List[str]]:
    """
    Define categorical and numerical feature lists.
    
    Parameters:
        include_grades (bool): If True, includes interim grades G1 and G2 in numerical features.
                               If False (Early-Warning Mode), G1 and G2 are excluded.
                               
    Returns:
        Tuple[List[str], List[str]]: (categorical_cols, numerical_cols)
    """
    categorical_cols = [
        'school', 'sex', 'address', 'famsize', 'Pstatus',
        'Mjob', 'Fjob', 'reason', 'guardian',
        'schoolsup', 'famsup', 'paid', 'activities',
        'nursery', 'higher', 'internet', 'romantic'
    ]
    
    numerical_cols = [
        'age', 'Medu', 'Fedu', 'traveltime', 'studytime',
        'failures', 'famrel', 'freetime', 'goout',
        'Dalc', 'Walc', 'health', 'absences'
    ]
    
    if include_grades:
        numerical_cols.extend(['G1', 'G2'])
        
    return categorical_cols, numerical_cols


def build_preprocessor(categorical_cols: List[str], numerical_cols: List[str]) -> ColumnTransformer:
    """
    Construct a Scikit-Learn ColumnTransformer for robust preprocessing.
    - Numerical: Median imputation + Standard scaling
    - Categorical: Most-frequent imputation + One-Hot Encoding (ignoring unknown values)
    
    Parameters:
        categorical_cols (List[str]): List of categorical column names.
        numerical_cols (List[str]): List of numerical column names.
        
    Returns:
        ColumnTransformer: Preprocessing pipeline.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor


# ==============================================================================
# 3. MODEL TRAINING & PIPELINE ORCHESTRATION
# ==============================================================================

class EduPathModelBundle:
    """
    Container bundle holding trained models, preprocessor, feature names,
    evaluation metrics, and training metadata.
    """
    def __init__(
        self,
        models: Dict[str, Any],
        preprocessor: ColumnTransformer,
        categorical_cols: List[str],
        numerical_cols: List[str],
        encoded_feature_names: List[str],
        metrics: Dict[str, Dict[str, float]],
        include_grades: bool,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ):
        self.models = models
        self.preprocessor = preprocessor
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.encoded_feature_names = encoded_feature_names
        self.metrics = metrics
        self.include_grades = include_grades
        self.X_test = X_test
        self.y_test = y_test
        self.primary_model_name = "Random Forest Regressor"

    @property
    def primary_model(self):
        return self.models.get(self.primary_model_name)


def train_and_evaluate_pipeline(
    df: pd.DataFrame,
    include_grades: bool = False,
    test_size: float = 0.20,
    random_state: int = 42
) -> EduPathModelBundle:
    """
    Full End-to-End ML Training Pipeline:
    1. Splits dataset into features (X) and target (y = G3).
    2. Builds preprocessor depending on whether early grades (G1, G2) are included.
    3. Performs 80/20 Train-Test split.
    4. Fits preprocessor on X_train and transforms both train and test.
    5. Trains Random Forest, Decision Tree, and Linear Regression models.
    6. Evaluates test MAE, RMSE, and R² for each algorithm.
    7. Extracts transformed feature names for explainability.
    
    Parameters:
        df (pd.DataFrame): Input student dataset.
        include_grades (bool): Early-warning toggle (False = no G1/G2; True = with G1/G2).
        test_size (float): Proportion for testing (default: 0.20).
        random_state (int): Random seed for reproducibility.
        
    Returns:
        EduPathModelBundle: Encapsulated trained pipeline.
    """
    # Target variable is final grade G3 (scale: 0 - 20)
    y = df['G3'].copy()
    
    # Feature columns
    cat_cols, num_cols = get_feature_lists(include_grades=include_grades)
    all_feature_cols = cat_cols + num_cols
    X = df[all_feature_cols].copy()
    
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Construct and fit preprocessor
    preprocessor = build_preprocessor(cat_cols, num_cols)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Extract encoded feature names for interpretability
    num_names = num_cols
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
    encoded_feature_names = num_names + cat_names
    
    # Define candidate regressors
    models = {
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=4,
            random_state=random_state
        ),
        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=4,
            random_state=random_state
        ),
        "Linear Regression": LinearRegression()
    }
    
    metrics = {}
    
    # Train and evaluate each algorithm
    for name, model in models.items():
        model.fit(X_train_proc, y_train)
        preds = model.predict(X_test_proc)
        
        mae = float(mean_absolute_error(y_test, preds))
        mse = float(mean_squared_error(y_test, preds))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, preds))
        
        metrics[name] = {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 3),
            "predictions": preds
        }
        
    bundle = EduPathModelBundle(
        models=models,
        preprocessor=preprocessor,
        categorical_cols=cat_cols,
        numerical_cols=num_cols,
        encoded_feature_names=encoded_feature_names,
        metrics=metrics,
        include_grades=include_grades,
        X_test=X_test,
        y_test=y_test
    )
    
    return bundle


# ==============================================================================
# 4. EXPLAINABILITY & FEATURE IMPORTANCE
# ==============================================================================

def get_top_feature_importances(
    bundle: EduPathModelBundle,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Extract and aggregate feature importances from the primary Random Forest model.
    Maps one-hot dummy columns back to base feature names for clean interpretation.
    
    Parameters:
        bundle (EduPathModelBundle): Trained bundle.
        top_n (int): Number of top features to return.
        
    Returns:
        pd.DataFrame: DataFrame with columns ['Feature', 'Importance', 'Percentage'].
    """
    rf_model = bundle.primary_model
    if not hasattr(rf_model, 'feature_importances_'):
        return pd.DataFrame(columns=['Feature', 'Importance', 'Percentage'])
    
    importances = rf_model.feature_importances_
    names = bundle.encoded_feature_names
    
    # Map back to readable names
    importance_dict: Dict[str, float] = {}
    for name, imp in zip(names, importances):
        # Clean up column prefixes for display
        clean_name = name
        for cat in bundle.categorical_cols:
            if clean_name.startswith(f"{cat}_"):
                clean_name = f"{cat.capitalize()} ({clean_name.split('_', 1)[1]})"
                break
        
        # User-friendly renaming for common terms
        rename_map = {
            'studytime': 'Weekly Study Hours',
            'failures': 'Past Academic Setbacks',
            'absences': 'School Absences',
            'health': 'Self-Reported Health',
            'freetime': 'Free Time After School',
            'goout': 'Going Out with Friends',
            'famrel': 'Family Relationship Quality',
            'Medu': "Mother's Education Level",
            'Fedu': "Father's Education Level",
            'G1': 'Interim Exam 1 (G1)',
            'G2': 'Interim Exam 2 (G2)',
            'age': 'Student Age',
            'traveltime': 'Travel Time to School',
            'schoolsup': 'School Extra Support',
            'famsup': 'Family Educational Support',
            'higher': 'Higher Education Ambition',
            'internet': 'Internet Access at Home'
        }
        display_name = rename_map.get(clean_name, clean_name)
        importance_dict[display_name] = importance_dict.get(display_name, 0.0) + imp
        
    imp_df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])
    imp_df = imp_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    imp_df['Percentage'] = (imp_df['Importance'] * 100).round(1)
    
    return imp_df.head(top_n)


def explain_student_prediction(
    student_record: Dict[str, Any],
    predicted_grade: float,
    bundle: EduPathModelBundle
) -> Dict[str, List[str]]:
    """
    Generate an empathetic, plain-English breakdown of why the model predicted
    the student's grade, identifying both key strengths and growth opportunities.
    
    Parameters:
        student_record (Dict[str, Any]): Inputs for the individual student.
        predicted_grade (float): Model predicted final score.
        bundle (EduPathModelBundle): Model bundle.
        
    Returns:
        Dict[str, List[str]]: {'strengths': [...], 'growth_areas': [...], 'summary': str}
    """
    strengths = []
    growth_areas = []
    
    # 1. Study time analysis
    st = student_record.get('studytime', 2)
    if st >= 3:
        strengths.append(f"Dedicated Study Time: Investing {'>10 hours' if st == 4 else '5-10 hours'} weekly provides strong academic momentum.")
    elif st == 1:
        growth_areas.append("Study Schedule: Currently under 2 hours per week. Increasing focused study sessions can significantly boost retention.")
        
    # 2. Absences analysis
    absences = student_record.get('absences', 0)
    if absences <= 4:
        strengths.append(f"Consistent Attendance: With only {absences} absences, the student benefits from high in-class engagement.")
    elif absences >= 12:
        growth_areas.append(f"Attendance Continuity: {absences} absences detected. In-class discussions and direct teacher explanations are hard to substitute.")
        
    # 3. Past Failures analysis
    failures = student_record.get('failures', 0)
    if failures == 0:
        strengths.append("Continuous Progression: No past subject failures indicates steady foundational learning.")
    else:
        growth_areas.append(f"Foundational Gaps: {failures} past course challenge(s) suggest addressing core prerequisites will unlock higher mastery.")
        
    # 4. Learning Support & Ambition
    if student_record.get('higher') == 'yes':
        strengths.append("High Ambition: Strong intrinsic motivation to pursue higher education correlates with academic resilience.")
    if student_record.get('schoolsup') == 'yes':
        strengths.append("Proactive Support: Actively utilizing institutional school support provides structured academic safety nets.")
    if student_record.get('internet') == 'yes':
        strengths.append("Digital Access: Home internet facilitates research, interactive tutorials, and peer collaboration.")
    else:
        growth_areas.append("Resource Access: Limited home internet access makes school library and offline resources essential.")
        
    # 5. Interim Grades (if included)
    if bundle.include_grades:
        g1 = student_record.get('G1', 10)
        g2 = student_record.get('G2', 10)
        if g2 >= 14:
            strengths.append(f"Strong Exam Trajectory: High midterm performance ({g2}/20 in G2) serves as the strongest predictor of final exam success.")
        elif g2 < 10:
            growth_areas.append(f"Midterm Recovery Needed: G2 score ({g2}/20) signals that targeted revision on challenging topics is urgently needed before finals.")
            
    # Default fallbacks if empty
    if not strengths:
        strengths.append("Growth Potential: The student possesses an open foundation ready to benefit from tailored study routines.")
    if not growth_areas:
        growth_areas.append("Mastery Refinement: Focus on deep problem-solving, teaching peers, and exploring advanced enrichment topics.")
        
    # Overall summary narrative
    if predicted_grade >= 15.0:
        tone = "The student shows exemplary indicators and is on track for honors-level mastery."
    elif predicted_grade >= 11.5:
        tone = "The student maintains a solid foundation with several bright spots, and targeted tweaks will elevate their performance."
    else:
        tone = "The student will benefit greatly from immediate structured guidance, attendance recovery, and dedicated study blocks."
        
    return {
        "strengths": strengths,
        "growth_areas": growth_areas,
        "summary": tone
    }


# ==============================================================================
# 5. PREDICTION & PERSONALIZED RECOMMENDATIONS ENGINE
# ==============================================================================

def classify_performance(grade: float) -> Tuple[str, str, str]:
    """
    Classify student performance using supportive, constructive tiers.
    Never uses words like 'Failure' or 'Failing'.
    
    Parameters:
        grade (float): Predicted grade (0 to 20).
        
    Returns:
        Tuple[str, str, str]: (Status Label, Badge Emoji, Color Hex)
    """
    if grade >= 16.0:
        return "Excelling (Mastery Track)", "🌟", "#10B981"  # Emerald Green
    elif grade >= 12.0:
        return "On Track (Growth Track)", "📈", "#3B82F6"     # Royal Blue
    else:
        return "Targeted Support Recommended", "💡", "#F59E0B" # Amber / Orange


def predict_student_performance(
    bundle: EduPathModelBundle,
    student_dict: Dict[str, Any],
    algorithm_name: Optional[str] = None
) -> Tuple[float, str, str, str]:
    """
    Predict final grade (G3) for a single student dictionary using the trained pipeline.
    
    Parameters:
        bundle (EduPathModelBundle): The trained model bundle.
        student_dict (Dict[str, Any]): Dictionary of student attributes.
        algorithm_name (str, optional): Chosen regressor name. Defaults to primary Random Forest.
        
    Returns:
        Tuple[float, str, str, str]: (predicted_grade, status_label, badge_emoji, color_hex)
    """
    model_name = algorithm_name or bundle.primary_model_name
    model = bundle.models.get(model_name, bundle.primary_model)
    
    # Construct single-row DataFrame
    cols = bundle.categorical_cols + bundle.numerical_cols
    # Fill in any missing default values safely
    safe_row = {}
    for c in cols:
        safe_row[c] = student_dict.get(c, 0 if c in bundle.numerical_cols else 'no')
        
    single_df = pd.DataFrame([safe_row])
    
    # Transform using fitted preprocessor
    single_proc = bundle.preprocessor.transform(single_df)
    
    # Predict and clip within realistic grade range (0 to 20)
    raw_pred = float(model.predict(single_proc)[0])
    clipped_pred = round(float(np.clip(raw_pred, 0.0, 20.0)), 1)
    
    status, emoji, color = classify_performance(clipped_pred)
    return clipped_pred, status, emoji, color


def generate_personalized_recommendations(
    student_dict: Dict[str, Any],
    predicted_grade: float,
    include_grades: bool = False
) -> List[Dict[str, str]]:
    """
    Generate an actionable, supportive, growth-mindset personalized learning plan.
    Categorized into clear developmental domains with explicit priority levels.
    
    Parameters:
        student_dict (Dict[str, Any]): Student input details.
        predicted_grade (float): Model predicted grade.
        include_grades (bool): Whether interim grades are in play.
        
    Returns:
        List[Dict[str, str]]: Structured recommendations table records.
    """
    plan = []
    
    # 1. Study Routine & Habit Dimension
    st = student_dict.get('studytime', 2)
    if st == 1:
        plan.append({
            "Focus Area": "Study Schedule & Cadence",
            "Current Indicator": "Under 2 hours/week",
            "Actionable Strategy": "Implement the 'Pomodoro Technique' (25 min study + 5 min break). Aim to increase study time to 4-5 hours weekly using consistent daily study blocks.",
            "Priority": "High"
        })
    elif st == 2:
        plan.append({
            "Focus Area": "Study Schedule & Cadence",
            "Current Indicator": "2-5 hours/week",
            "Actionable Strategy": "Solid baseline. Shift from passive reading to active recall (flashcards, practice problems) to maximize retention per hour.",
            "Priority": "Medium"
        })
    else:
        plan.append({
            "Focus Area": "Study Schedule & Cadence",
            "Current Indicator": "5+ hours/week (High)",
            "Actionable Strategy": "Outstanding dedication! Focus on spaced repetition and teaching complex concepts to peers (Feynman technique) for deep mastery.",
            "Priority": "Low (Enrichment)"
        })
        
    # 2. Attendance & Participation Dimension
    absences = student_dict.get('absences', 0)
    if absences >= 10:
        plan.append({
            "Focus Area": "Attendance & In-Class Engagement",
            "Current Indicator": f"{absences} school absences",
            "Actionable Strategy": "Partner with class study buddies to fill missed lecture notes. Schedule a 15-minute weekly office check-in with your teacher to clarify doubts.",
            "Priority": "High"
        })
    elif absences >= 5:
        plan.append({
            "Focus Area": "Attendance & In-Class Engagement",
            "Current Indicator": f"{absences} school absences",
            "Actionable Strategy": "Maintain current attendance and actively participate by asking at least one clarifying question per class session.",
            "Priority": "Medium"
        })
    else:
        plan.append({
            "Focus Area": "Attendance & In-Class Engagement",
            "Current Indicator": "High attendance record",
            "Actionable Strategy": "Excellent continuity. Continue engaging in live classroom discussions and leading small group problem-solving exercises.",
            "Priority": "Low (Maintain)"
        })
        
    # 3. Foundational Reinforcement & Mentorship
    failures = student_dict.get('failures', 0)
    schoolsup = student_dict.get('schoolsup', 'no')
    if failures > 0 or schoolsup == 'yes' or predicted_grade < 12.0:
        plan.append({
            "Focus Area": "Foundational Skill Reinforcement",
            "Current Indicator": f"{failures} past challenge(s)" if failures > 0 else "Intervention Recommended",
            "Actionable Strategy": "Enroll in after-school peer tutoring or departmental workshops. Break down past exam questions into step-by-step diagnostic solution guides.",
            "Priority": "High"
        })
    else:
        plan.append({
            "Focus Area": "Advanced Enrichment",
            "Current Indicator": "Consistent foundation",
            "Actionable Strategy": "Tackle challenging challenge problems or participate in math/science competitions to push boundaries and build portfolio depth.",
            "Priority": "Low (Enrichment)"
        })
        
    # 4. Learning Resources & Digital Tools
    internet = student_dict.get('internet', 'yes')
    if internet == 'no':
        plan.append({
            "Focus Area": "Resource Accessibility",
            "Current Indicator": "No home internet access",
            "Actionable Strategy": "Utilize school computer labs, download offline educational materials (e.g., Khan Academy offline, OpenStax textbooks), and borrow library reserve sets.",
            "Priority": "High"
        })
    else:
        plan.append({
            "Focus Area": "Digital Learning Integration",
            "Current Indicator": "Internet accessible",
            "Actionable Strategy": "Leverage interactive platforms (Desmos, GeoGebra, Coursera, YouTube Edu) for visual proofs and multi-angle explanations.",
            "Priority": "Medium"
        })
        
    # 5. Well-Being & Lifestyle Balance
    health = student_dict.get('health', 3)
    freetime = student_dict.get('freetime', 3)
    goout = student_dict.get('goout', 3)
    if health <= 2 or goout >= 4 or freetime >= 4:
        plan.append({
            "Focus Area": "Lifestyle & Well-Being Balance",
            "Current Indicator": "High social/free time or low health rating",
            "Actionable Strategy": "Protect cognitive stamina: ensure 7-8 hours of sleep nightly. Establish a clear boundary between recreational hours and study blocks.",
            "Priority": "Medium"
        })
    else:
        plan.append({
            "Focus Area": "Sustainable Well-Being",
            "Current Indicator": "Balanced routine",
            "Actionable Strategy": "Continue maintaining a healthy balance of physical exercise, social connection, and academic dedication.",
            "Priority": "Low (Maintain)"
        })
        
    # 6. Exam Readiness (Mid-Semester Mode)
    if include_grades:
        g2 = student_dict.get('G2', 10)
        if g2 < 11:
            plan.append({
                "Focus Area": "Exam Strategy & Mock Testing",
                "Current Indicator": f"G2 score: {g2}/20",
                "Actionable Strategy": "Complete two timed past papers under real exam conditions. Review incorrect answers with an error-log to prevent repeat mistakes.",
                "Priority": "High"
            })
            
    return plan
