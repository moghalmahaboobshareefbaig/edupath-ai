# 🎓 EduPath AI: Intelligent Student Performance & Personalized Learning System
> **DATANYX'26 Hackathon Prototype** | Explainable AI & Growth-Mindset Learning Interventions

EduPath AI is an educational data science application built with **Python, Scikit-learn, Pandas, Matplotlib/Seaborn, and Streamlit**. It analyzes student academic, behavioral, and demographic features, forecasts final performance (`G3`), provides transparent AI explainability, and generates actionable, growth-oriented learning recommendations without ever labeling students as failures.

---

## 🌟 Key Features

1. **Dual Horizon Prediction Modes:**
   - 🛡️ **Early-Warning Mode (Pre-Semester):** Forecasts final grades *before* midterm exams (`G1`, `G2`) take place. Enables proactive counselor interventions based on attendance, study habits, and support systems.
   - 📊 **Mid-Semester Mode (Comprehensive):** Incorporates interim grades (`G1`, `G2`) for maximum predictive accuracy ($R^2 > 0.80$).

2. **Growth-Mindset Recommendation Engine:**
   - Never uses punitive labels like "Failing" or "Failure".
   - Categorizes performance into supportive tiers:
     - 🌟 **Excelling (Mastery Track)**: $\ge 16.0 / 20$
     - 📈 **On Track (Growth Track)**: $12.0 - 15.9 / 20$
     - 💡 **Targeted Support Recommended**: $< 12.0 / 20$
   - Generates concrete, prioritized action plans across Study Cadence, In-Class Attendance, Foundational Reinforcement, Digital Access, and Well-Being.

3. **Explainable AI (XAI):**
   - **Global Feature Importance:** Bar charts of the top factors driving the Random Forest model (e.g., study time, prior setbacks, attendance).
   - **Local Student Breakdown:** Plain-English attribution highlighting positive catalysts vs. high-leverage growth areas.
   - **Cohort Benchmarking:** Student study hours and absences compared against the peer cohort median.

4. **Model Performance & Exploratory Data Analysis (EDA):**
   - Side-by-side benchmarking: **Random Forest Regressor**, **Decision Tree Regressor**, and **Linear Regression**.
   - Evaluation metrics: **MAE**, **RMSE**, and **$R^2$ Score** on an 80/20 train/test split.
   - Interactive statistical charts: Target Grade distribution, Study Time vs. Grade boxplots, Absences trendline, and Feature Correlation Heatmap.

5. **Built-in UCI Fallback Data Generator:**
   - Automatically loads or generates a 395-row dataset modeled after the **UCI Machine Learning Repository Student Performance Dataset** (`student-mat.csv`), ensuring the app runs immediately out-of-the-box.

---

## 📁 Project Structure

```text
├── app.py                 # Streamlit interactive frontend dashboard (3 main tabs)
├── model.py               # ML pipeline: data loading, preprocessing, training, evaluation, XAI
├── requirements.txt       # Dependencies (pandas, scikit-learn, streamlit, etc.)
├── README.md              # Project documentation and hackathon guide
└── data/
    ├── generate_data.py   # Fallback generator modeled after UCI student-mat.csv
    └── student-mat.csv    # Semicolon-separated student performance dataset
```

---

## 🚀 Quickstart & Execution Instructions

### Prerequisites
- Python 3.9+ (or `uv`)
- Recommended: A clean virtual environment

### Step 1: Clone or Open the Project Directory
```bash
cd "c:/Users/SHAREEF/OneDrive/Desktop/my html"
```

### Step 2: Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

*(Alternatively, if using `uv`:)*
```bash
uv venv
.\.venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the EduPath AI Streamlit App
```bash
streamlit run app.py
```
*The app will automatically open in your default browser at `http://localhost:8501`.*

---

## 🎯 Demo Walkthrough Guide (For Hackathon Judges)

1. **Test the Early-Warning Mode vs. Mid-Semester Mode:**
   - Toggle the radio button in the sidebar to see how the model adapts with and without interim exam grades ($G1$, $G2$).
2. **Explore Preset Profiles:**
   - Select **"🌟 Maya - High Achiever"** to view mastery recommendations.
   - Select **"💡 Jordan - High Support Need"** to view empathetic, high-priority intervention strategies.
   - Select **"🚀 Sam - Rising Comeback"** to observe how exam trajectory impacts the forecast.
3. **Inspect Explainability (Tab 2):**
   - View the **Top 10 Feature Importances** chart.
   - Inspect the **AI Counselor Insight** and cohort benchmark comparisons.
4. **Review Model Performance (Tab 3):**
   - Inspect the $80/20$ test metrics table comparing Random Forest, Decision Tree, and Linear Regression.
   - Examine the interactive correlation heatmap and distribution plots.
5. **Download Student Action Plan:**
   - Click **"📥 Download Personalized Student Action Plan (.txt)"** to export a printable guidance summary.

---

## ⚖️ Ethical AI & Pedagogical Principles
EduPath AI adheres to strict educational ethics:
- **No Deficit Framing:** Predictions are framed as trajectories and opportunities for growth rather than fixed labels.
- **Actionability Over Prediction:** Every assessment is paired with actionable pedagogical strategies.
- **Transparency:** Educators and students can inspect the exact factors influencing any prediction.
