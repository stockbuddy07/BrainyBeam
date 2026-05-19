# Bank Campaign Sense - Streamlit Deployment Fix
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import time

# Page config
st.set_page_config(page_title="Bank Campaign Sense", page_icon="🏦", layout="wide")

# Custom CSS for dark theme adjustments
st.markdown("""
<style>
    .stApp {
        background-color: #0a0e1a;
        color: #f1f5f9;
    }
    .css-1d391kg {
        background-color: #1e293b;
    }
    h1, h2, h3 {
        color: #f8fafc;
    }
    .st-emotion-cache-16txtl3 {
        padding-top: 2rem;
    }
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        color: #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    with open('gb_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model_pipeline = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error("Error loading the model. Make sure you have run `train_model.py` to generate `gb_model.pkl`.")

# Load sample data for overview (cache to avoid reloading)
@st.cache_data
def load_data():
    try:
        return pd.read_csv('bank.xls')
    except:
        return pd.DataFrame() # Return empty if not found

df = load_data()

# ── Sidebar Navigation ──
st.sidebar.title("🏦 Bank Campaign Sense")
st.sidebar.markdown("AI-Powered Term Deposit Prediction")
page = st.sidebar.radio("Navigation", ["Overview", "Bulk Scanner", "Predict Subscription"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Created by:** Ayush Gajjar")
st.sidebar.markdown("**Tools:** Python, Streamlit, Scikit-Learn")

# ── PAGE 1: OVERVIEW ──
if page == "Overview":
    st.title("Dataset Overview")
    
    if df.empty:
        st.warning("Could not load `bank.xls`. Please ensure the file is in the same directory.")
    else:
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        total_records = len(df)
        subscribed = len(df[df['deposit'] == 'yes'])
        not_subscribed = total_records - subscribed
        rate = (subscribed / total_records) * 100
        
        col1.metric("Total Records", f"{total_records:,}")
        col2.metric("Subscription Rate", f"{rate:.1f}%")
        col3.metric("Subscribed (Yes)", f"{subscribed:,}")
        col4.metric("Not Subscribed (No)", f"{not_subscribed:,}")
        
        st.markdown("---")
        
        # Charts row 1
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Target Distribution")
            fig_donut = px.pie(df, names='deposit', color='deposit',
                              color_discrete_map={'yes': '#34d399', 'no': '#3b82f6'},
                              hole=0.6)
            fig_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#94a3b8")
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with c2:
            st.subheader("Parameter Statistics")
            with st.expander("View Detailed Parameter Statistics"):
                num_cols = df.select_dtypes(include=['number']).columns
                stats = df[num_cols].describe().T[['mean', 'std', 'min', 'max']]
                try:
                    st.dataframe(stats.style.format("{:.2f}").background_gradient(cmap='Blues'), use_container_width=True)
                except Exception:
                    st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)
            st.write("The statistics above summarise the numerical distribution of age, balance, duration, and campaign history across the 11,162 customer records.")
            
        # Charts row 2
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Age Distribution")
            fig_age = px.histogram(df, x='age', nbins=20, color_discrete_sequence=['#3b82f6'])
            fig_age.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#94a3b8")
            st.plotly_chart(fig_age, use_container_width=True)
            
        with c4:
            st.subheader("Job Type vs Campaign Response")
            job_response = pd.crosstab(df['job'], df['deposit']).reset_index()
            fig_job = px.bar(job_response, x='job', y=['no', 'yes'], barmode='group',
                            color_discrete_sequence=['#f87171', '#34d399'])
            fig_job.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#94a3b8")
            st.plotly_chart(fig_job, use_container_width=True)

# ── PAGE 2: BULK SCANNER ──
elif page == "Bulk Scanner":
    st.title("Bulk Campaign Scanner")
    st.write("Process multiple customer records at once — upload a CSV or Excel file and get batch subscription predictions.")
    
    if not model_loaded:
        st.stop()
        
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                upload_df = pd.read_csv(uploaded_file)
            else:
                try:
                    upload_df = pd.read_excel(uploaded_file)
                except ValueError:
                    # Fallback for CSV files disguised with .xls extension
                    uploaded_file.seek(0)
                    upload_df = pd.read_csv(uploaded_file)
                
            st.success(f"File '{uploaded_file.name}' uploaded successfully. ({len(upload_df)} records)")
            
            with st.spinner("Processing records..."):
                time.sleep(0.5) # Quick delay for visual feedback
                
                # Ensure all columns are present
                required_cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
                
                # Find missing columns
                missing = [c for c in required_cols if c not in upload_df.columns]
                if missing:
                    st.error(f"Missing required columns in uploaded file: {', '.join(missing)}")
                else:
                    # Make predictions using the ML model pipeline
                    predictions = model_pipeline.predict(upload_df[required_cols])
                    # Get probabilities for confidence score
                    probs = model_pipeline.predict_proba(upload_df[required_cols])
                    
                    upload_df['Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
                    upload_df['ConfidenceScore'] = [max(prob)*100 for prob in probs]
                    upload_df['Confidence'] = [f"{score:.1f}%" for score in upload_df['ConfidenceScore']]
                    
                    # Show summary
                    yes_count = sum(predictions)
                    st.subheader("Prediction Results")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Processed", len(upload_df))
                    c2.metric("Will Subscribe", yes_count)
                    c3.metric("Won't Subscribe", len(upload_df) - yes_count)
                    
                    st.markdown("---")
                    
                    # Visualisations Row 1
                    st.subheader("📊 Bulk Prediction Visualisations")
                    v1, v2 = st.columns(2)
                    with v1:
                        fig_pred_donut = px.pie(upload_df, names='Prediction', title='Predicted Subscription Split',
                                                color='Prediction', color_discrete_map={'Yes': '#34d399', 'No': '#f87171'},
                                                hole=0.6)
                        fig_pred_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#94a3b8")
                        st.plotly_chart(fig_pred_donut, use_container_width=True)
                    
                    with v2:
                        fig_conf_dist = px.histogram(upload_df, x='ConfidenceScore', title='Prediction Confidence Distribution',
                                                     color='Prediction', color_discrete_map={'Yes': '#34d399', 'No': '#f87171'},
                                                     labels={'ConfidenceScore': 'Confidence Score (%)'}, nbins=20)
                        fig_conf_dist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#94a3b8")
                        st.plotly_chart(fig_conf_dist, use_container_width=True)
                        
                    # Visualisations Row 2
                    v3, v4 = st.columns(2)
                    with v3:
                        job_pred = pd.crosstab(upload_df['job'], upload_df['Prediction']).reset_index()
                        for col in ['No', 'Yes']:
                            if col not in job_pred.columns:
                                job_pred[col] = 0
                        fig_job_pred = px.bar(job_pred, x='job', y=['No', 'Yes'], barmode='group',
                                              title='Predictions by Job Type',
                                              color_discrete_sequence=['#f87171', '#34d399'])
                        fig_job_pred.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#94a3b8")
                        st.plotly_chart(fig_job_pred, use_container_width=True)
                        
                    with v4:
                        fig_bal_box = px.box(upload_df, x='Prediction', y='balance', title='Account Balance by Prediction',
                                             color='Prediction', color_discrete_map={'Yes': '#34d399', 'No': '#f87171'},
                                             points="outliers")
                        fig_bal_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#94a3b8")
                        st.plotly_chart(fig_bal_box, use_container_width=True)
                        
                    st.markdown("---")
                    st.subheader("📋 Detailed Prediction Records")
                    
                    # Show data
                    display_cols = ['age', 'job', 'balance', 'duration', 'Prediction', 'Confidence']
                    st.dataframe(upload_df[display_cols].head(100), use_container_width=True)
                    if len(upload_df) > 100:
                        st.caption(f"Showing first 100 rows of {len(upload_df)}.")
                        
                    # Download button
                    csv = upload_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇ Download Complete Results CSV",
                        data=csv,
                        file_name='bulk_predictions.csv',
                        mime='text/csv',
                    )
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# ── PAGE 3: PREDICT SUBSCRIPTION ──
elif page == "Predict Subscription":
    st.title("Predict Term Deposit Subscription")
    st.write("Enter customer parameters to predict their likelihood of subscribing using the trained Gradient Boosting model.")
    
    if not model_loaded:
        st.stop()
        
    with st.form("prediction_form"):
        st.subheader("📝 Customer Profile & Campaign Details")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### 👤 Personal Info")
            age = st.number_input("Age", min_value=18, max_value=100, value=38)
            job = st.selectbox("Job", ["management", "technician", "entrepreneur", "blue-collar", "unknown", "retired", "admin.", "services", "self-employed", "unemployed", "housemaid", "student"])
            marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
            education = st.selectbox("Education", ["tertiary", "secondary", "unknown", "primary"])
            
            st.markdown("#### 💰 Financials")
            balance = st.number_input("Balance (₹)", value=2000, step=100)
            housing = st.selectbox("Housing Loan", ["no", "yes"])
            loan = st.selectbox("Personal Loan", ["no", "yes"])
            
        with c2:
            st.markdown("#### 📞 Call Details")
            duration = st.number_input("Call Duration (seconds)", value=250, step=10)
            campaign = st.number_input("Contacts in current campaign", min_value=1, value=1)
            
            # Collapsible Advanced Settings for all other inputs so the UI remains clean and simplified by default!
            st.markdown("#### ⚙️ Campaign Settings")
            with st.expander("🛠️ Advanced Parameters (Optional)", expanded=False):
                default = st.selectbox("Credit Default", ["no", "yes"])
                contact = st.selectbox("Contact Type", ["cellular", "unknown", "telephone"])
                day = st.number_input("Day of Month", min_value=1, max_value=31, value=15)
                month = st.selectbox("Month", ["may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "jan", "feb", "mar", "apr"])
                pdays = st.number_input("Days since last contact", value=-1)
                previous = st.number_input("Previous Contacts", min_value=0, value=0)
                poutcome = st.selectbox("Previous Outcome", ["unknown", "other", "failure", "success"])
                
        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("🔮 Run ML Prediction", type="primary", use_container_width=True)
        
    if submit:
        # Create input dataframe
        input_data = pd.DataFrame({
            'age': [age], 'job': [job], 'marital': [marital], 'education': [education],
            'default': [default], 'balance': [balance], 'housing': [housing], 'loan': [loan],
            'contact': [contact], 'day': [day], 'month': [month], 'duration': [duration],
            'campaign': [campaign], 'pdays': [pdays], 'previous': [previous], 'poutcome': [poutcome]
        })
        
        # Predict
        pred = model_pipeline.predict(input_data)[0]
        prob = model_pipeline.predict_proba(input_data)[0]
        confidence = max(prob) * 100
        
        st.markdown("---")
        st.subheader("Prediction Outcome")
        
        res_col1, res_col2 = st.columns(2)
        
        if pred == 1:
            res_col1.metric("Predicted Action", "✅ Will Subscribe")
            res_col2.metric("Confidence Score", f"{confidence:.1f}%")
            st.success("**Recommendation:** High-priority lead — schedule call promptly.")
            st.progress(int(confidence) / 100, text="Confidence Level")
        else:
            res_col1.metric("Predicted Action", "❌ Will Not Subscribe")
            res_col2.metric("Confidence Score", f"{confidence:.1f}%")
            st.error("**Recommendation:** Low-priority — consider skipping or scheduling later.")
            st.progress(int(confidence) / 100, text="Confidence Level")
            
        # Feature insight for the specific prediction
        factors = []
        
        # Duration insights
        if duration > 300:
            factors.append("• **Call Duration (Long):** Convincing conversation time (> 5 mins) is historically a strong positive indicator.")
        elif duration < 100:
            factors.append("• **Call Duration (Short):** Very brief conversation (< 1.6 mins) usually points to low subscriber interest.")
        else:
            factors.append(f"• **Call Duration (Moderate):** Conversation length of {duration} seconds provides a standard active window for building customer interest.")
            
        # Previous outcome
        if poutcome == 'success':
            factors.append("• **Previous Campaign Success:** Customer previously subscribed, indicating an exceptionally high propensity to buy again.")
        elif poutcome == 'failure':
            factors.append("• **Previous Campaign Failure:** Customer previously rejected a campaign, which requires a highly personalized follow-up strategy.")
        else:
            factors.append("• **New Contact Segment:** No previous campaign outcome recorded. Prediction depends highly on current call duration and financials.")
            
        # Balance insights
        if balance > 2000:
            factors.append(f"• **Account Balance (High):** Customer has a strong balance of ₹{balance:,}, making a term deposit purchase highly feasible.")
        elif balance < 0:
            factors.append(f"• **Account Balance (Negative):** Customer has a negative balance of ₹{balance:,}, decreasing subscription likelihood.")
        else:
            factors.append(f"• **Account Balance (Stable):** Customer has a stable balance of ₹{balance:,}.")

        # Loans
        if housing == 'no' and loan == 'no':
            factors.append("• **No Debt Load:** No housing or personal loans are active, giving the customer higher disposable cash for savings deposits.")
        elif housing == 'yes':
            factors.append("• **Housing Loan Active:** Existing mortgage commitment may reduce potential monthly deposit capacities.")
            
        # Job
        if job in ['retired', 'student']:
            factors.append(f"• **Demographics:** Customer's job category ({job}) belongs to a cohort that historically values fixed interest investments.")

        st.info("### 💡 Key Influencing Factors\n" + "\n".join(factors))
