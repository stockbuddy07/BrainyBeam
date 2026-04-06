import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Campaign Sense",
    page_icon="🏦",
    layout="wide",
)

# ── Simple Styling ───────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    [data-testid="stMetric"] {
        background: #1e1e2f;
        padding: 16px;
        border-radius: 12px;
        border-left: 4px solid #667eea;
    }
    [data-testid="stMetricLabel"] {
        color: #aaa !important;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    .stTabs [aria-selected="true"] {
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ── Data Loading ─────────────────────────────────────────────
def load_csv(source):
    """Try multiple separators to read the bank dataset."""
    for sep in [';', ',', '\t']:
        try:
            if hasattr(source, 'seek'):
                source.seek(0)
            df = pd.read_csv(source, sep=sep)
            df.columns = [c.strip().lower() for c in df.columns]
            if 'deposit' in df.columns and 'y' not in df.columns:
                df.rename(columns={'deposit': 'y'}, inplace=True)
            if 'y' in df.columns and len(df.columns) > 5:
                return df
        except:
            continue
    return None

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("🏦 Bank Campaign Sense")
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["xls", "csv"])

df = None
if uploaded_file:
    df = load_csv(uploaded_file)
else:
    for path in ['bank.xls', '../bank.xls', 'bank.csv']:
        if os.path.exists(path):
            df = load_csv(path)
            if df is not None and 'y' in df.columns:
                break
            df = None

if df is None or 'y' not in df.columns:
    st.title("🏦 Bank Campaign Sense")
    st.info("👈 Please upload the `bank.xls` dataset using the sidebar to get started.")
    if df is not None:
        st.warning(f"File loaded but target column `y` not found. Columns: {df.columns.tolist()}")
    st.stop()

# Clean data
df = df.drop_duplicates()

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Dataset Info")
st.sidebar.write(f"**Rows:** {len(df):,}")
st.sidebar.write(f"**Columns:** {df.shape[1]}")
st.sidebar.write(f"**Target:** `y` (yes/no)")
st.sidebar.markdown("---")
st.sidebar.caption("Built by Ayush Gajjar")

# ── Header ───────────────────────────────────────────────────
st.title("🏦 Bank Campaign Sense")
st.caption("Predicting Term Deposit Subscriptions using Machine Learning")
st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Model Comparison", "🎯 Predict"])

# ══════════════════════════════════════════════════════════════
# TAB 1: DASHBOARD (EDA)
# ══════════════════════════════════════════════════════════════
with tab1:

    # KPI Metrics
    conv_rate = (df['y'] == 'yes').mean() * 100
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Conversion Rate", f"{conv_rate:.1f}%")
    col3.metric("Avg Age", f"{df['age'].mean():.0f} yrs")
    col4.metric("Avg Balance", f"€{df['balance'].mean():,.0f}")

    st.markdown("---")

    # Row 1: Target distribution
    st.subheader("📈 Campaign Response")
    c1, c2 = st.columns(2)

    with c1:
        counts = df['y'].value_counts().reset_index()
        counts.columns = ['Response', 'Count']
        fig = px.bar(counts, x='Response', y='Count', color='Response',
                     color_discrete_map={'yes': '#66c2a5', 'no': '#fc8d62'},
                     text='Count', title='Response Count')
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.pie(df, names='y', title='Response Proportion',
                     color='y', color_discrete_map={'yes': '#66c2a5', 'no': '#fc8d62'},
                     hole=0.4)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: Age & Balance
    st.subheader("👤 Customer Demographics")
    c3, c4 = st.columns(2)

    with c3:
        fig = px.histogram(df, x='age', color='y', nbins=30, barmode='overlay',
                          color_discrete_map={'yes': '#66c2a5', 'no': '#fc8d62'},
                          title='Age Distribution', opacity=0.7)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.box(df, x='y', y='balance', color='y',
                    color_discrete_map={'yes': '#66c2a5', 'no': '#fc8d62'},
                    title='Balance by Response')
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Job & Education
    st.subheader("💼 Category Analysis")
    c5, c6 = st.columns(2)

    with c5:
        job_data = df.groupby(['job', 'y']).size().reset_index(name='count')
        fig = px.bar(job_data, x='job', y='count', color='y', barmode='group',
                    color_discrete_map={'yes': '#66c2a5', 'no': '#fc8d62'},
                    title='Job Type vs Response')
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        edu_data = df.groupby(['education', 'y']).size().reset_index(name='count')
        fig = px.bar(edu_data, x='education', y='count', color='y', barmode='group',
                    color_discrete_map={'yes': '#66c2a5', 'no': '#fc8d62'},
                    title='Education vs Response')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Row 4: Correlation & Duration
    st.subheader("🔗 Correlation & Call Duration")
    c7, c8 = st.columns(2)

    with c7:
        num_df = df.select_dtypes(include=np.number)
        corr = num_df.corr()
        fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                       title='Correlation Heatmap', zmin=-1, zmax=1, aspect='auto')
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with c8:
        fig = px.box(df, x='y', y='duration', color='y',
                    color_discrete_map={'yes': '#66c2a5', 'no': '#fc8d62'},
                    title='Call Duration vs Response',
                    labels={'duration': 'Duration (sec)', 'y': 'Subscribed'})
        fig.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Key Insights
    st.subheader("💡 Key Insights")
    yes_df = df[df['y'] == 'yes']
    no_df = df[df['y'] == 'no']

    i1, i2, i3 = st.columns(3)
    with i1:
        st.info(f"📞 **Call Duration**: Subscribers had avg call of **{yes_df['duration'].mean():.0f}s** "
                f"vs **{no_df['duration'].mean():.0f}s** for non-subscribers.")
    with i2:
        st.info(f"💰 **Balance**: Subscribers hold **€{yes_df['balance'].mean():,.0f}** avg balance "
                f"vs **€{no_df['balance'].mean():,.0f}**.")
    with i3:
        top_job = yes_df['job'].value_counts().index[0] if len(yes_df) > 0 else 'N/A'
        st.info(f"👔 **Top Job**: Most conversions from **{top_job}** "
                f"({yes_df['job'].value_counts().iloc[0] if len(yes_df)>0 else 0} subscribers).")

# ══════════════════════════════════════════════════════════════
# TAB 2: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🤖 Training & Comparing Classification Models")

    # Preprocessing
    @st.cache_data
    def preprocess(_df):
        d = _df.copy()
        d['y'] = d['y'].map({'yes': 1, 'no': 0})
        y = d['y']
        X = d.drop('y', axis=1)
        cats = X.select_dtypes(include='object').columns.tolist()
        X_enc = pd.get_dummies(X, columns=cats, drop_first=True)
        sc = StandardScaler()
        X_sc = pd.DataFrame(sc.fit_transform(X_enc), columns=X_enc.columns)
        Xtr, Xte, ytr, yte = train_test_split(X_sc, y, test_size=0.2, random_state=42, stratify=y)
        return Xtr, Xte, ytr, yte, X_enc, sc

    X_train, X_test, y_train, y_test, X_encoded, scaler = preprocess(df)

    st.write(f"**Train set:** {X_train.shape[0]:,} rows · **Test set:** {X_test.shape[0]:,} rows · **Features:** {X_train.shape[1]}")

    @st.cache_resource
    def train_models(_Xtr, _ytr):
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=6),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=7),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
        }
        trained = {}
        for name, model in models.items():
            model.fit(_Xtr, _ytr)
            trained[name] = model
        return trained

    with st.spinner("Training models..."):
        trained_models = train_models(X_train, y_train)

    # Evaluate
    results = []
    for name, model in trained_models.items():
        yp = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy (%)": round(accuracy_score(y_test, yp) * 100, 2),
            "Precision (%)": round(precision_score(y_test, yp, zero_division=0) * 100, 2),
            "Recall (%)": round(recall_score(y_test, yp, zero_division=0) * 100, 2),
            "F1-Score (%)": round(f1_score(y_test, yp, zero_division=0) * 100, 2),
        })

    res_df = pd.DataFrame(results).sort_values('F1-Score (%)', ascending=False).reset_index(drop=True)
    res_df.index += 1

    best_name = res_df.iloc[0]['Model']
    best_f1 = res_df.iloc[0]['F1-Score (%)']
    best_acc = res_df.iloc[0]['Accuracy (%)']

    st.success(f"🥇 **Best Model: {best_name}** — F1-Score: {best_f1}% · Accuracy: {best_acc}%")

    # Leaderboard Table
    st.markdown("#### 🏆 Model Leaderboard")
    st.dataframe(res_df, use_container_width=True, height=300)

    # Bar chart comparison
    st.markdown("#### 📊 Performance Comparison")
    fig = px.bar(res_df.melt(id_vars='Model', value_vars=['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)']),
                 x='value', y='Model', color='variable', barmode='group', orientation='h',
                 labels={'value': 'Score (%)', 'variable': 'Metric'},
                 color_discrete_sequence=['#667eea', '#fc8d62', '#66c2a5', '#a78bfa'])
    fig.update_layout(height=450, legend=dict(orientation='h', y=1.12))
    st.plotly_chart(fig, use_container_width=True)

    # Confusion Matrix & Feature Importance
    st.markdown("#### 🔍 Detailed Analysis")
    selected_model = st.selectbox("Select a model:", res_df['Model'].tolist())

    m1, m2 = st.columns(2)

    with m1:
        yp = trained_models[selected_model].predict(X_test)
        cm = confusion_matrix(y_test, yp)

        fig = px.imshow(cm, text_auto=True,
                       x=['Predicted No', 'Predicted Yes'],
                       y=['Actual No', 'Actual Yes'],
                       color_continuous_scale='Blues',
                       title=f'Confusion Matrix — {selected_model}')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with m2:
        if hasattr(trained_models[selected_model], 'feature_importances_'):
            imp = pd.Series(trained_models[selected_model].feature_importances_, index=X_encoded.columns)
            top10 = imp.sort_values(ascending=False).head(10).reset_index()
            top10.columns = ['Feature', 'Importance']

            fig = px.bar(top10, x='Importance', y='Feature', orientation='h',
                        title=f'Top 10 Features — {selected_model}',
                        color='Importance', color_continuous_scale='Viridis')
            fig.update_layout(height=400, yaxis=dict(autorange='reversed'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Feature importance not available for {selected_model}.")

# ══════════════════════════════════════════════════════════════
# TAB 3: PREDICTION
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🎯 Predict Customer Subscription")
    st.write("Enter customer details to predict if they will subscribe to a term deposit.")

    with st.form("predict_form"):
        st.markdown("**Customer Information**")
        r1, r2, r3 = st.columns(3)
        with r1:
            p_age = st.number_input("Age", 18, 100, 35)
            p_job = st.selectbox("Job", sorted(df['job'].unique()))
        with r2:
            p_marital = st.selectbox("Marital Status", sorted(df['marital'].unique()))
            p_education = st.selectbox("Education", sorted(df['education'].unique()))
        with r3:
            p_balance = st.number_input("Balance (€)", -10000, 100000, 1500)
            p_duration = st.number_input("Call Duration (sec)", 0, 5000, 250)

        st.markdown("**Loan Information**")
        l1, l2, l3 = st.columns(3)
        with l1:
            p_housing = st.selectbox("Housing Loan?", ["no", "yes"])
        with l2:
            p_loan = st.selectbox("Personal Loan?", ["no", "yes"])
        with l3:
            p_contact = st.selectbox("Contact Type", sorted(df['contact'].unique()) if 'contact' in df.columns else ["cellular", "telephone"])

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        X_train, X_test, y_train, y_test, X_encoded, scaler = preprocess(df)
        trained_models = train_models(X_train, y_train)

        # Build input
        input_dict = {
            'age': p_age, 'balance': p_balance, 'duration': p_duration,
            'job': p_job, 'marital': p_marital, 'education': p_education,
            'housing': p_housing, 'loan': p_loan,
            'day': 15, 'month': 'may', 'campaign': 1,
            'pdays': -1, 'previous': 0, 'default': 'no', 'contact': p_contact,
        }
        if 'poutcome' in df.columns:
            input_dict['poutcome'] = 'unknown'

        inp = pd.DataFrame([input_dict])
        inp_enc = pd.get_dummies(inp)
        for col in X_encoded.columns:
            if col not in inp_enc.columns:
                inp_enc[col] = 0
        inp_enc = inp_enc[X_encoded.columns]
        inp_sc = scaler.transform(inp_enc)

        # Predict with best model
        prob = float(trained_models[best_name].predict_proba(inp_sc)[0][1])
        label = "YES" if prob > 0.5 else "NO"

        st.markdown("---")
        st.markdown(f"**Model used:** {best_name}")

        r1, r2 = st.columns([2, 1])
        with r1:
            if label == "YES":
                st.success(f"### ✅ Likely to Subscribe! (Confidence: {prob:.1%})")
                st.balloons()
            else:
                st.error(f"### ❌ Unlikely to Subscribe (Confidence: {1-prob:.1%})")

        with r2:
            st.metric("Subscription Probability", f"{prob:.1%}")
            st.progress(float(prob))

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.caption("Bank Campaign Sense • Built by Ayush Gajjar • Brainy Beam Info-Tech PVT LTD")
