import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import mlflow
import time
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# MUST be the first command
st.set_page_config(
    page_title="✨ Churn Prediction Wizard",
    layout="centered",
    page_icon="🔮",
    initial_sidebar_state="expanded"
)

# ============== Model Monitoring ==============
class ModelMonitor:
    def __init__(self):
        self.performance_history = []
        
    def log_performance(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        self.performance_history.append({
            'timestamp': datetime.now(),
            'accuracy': accuracy,
            'f1_score': f1
        })
        
        if len(self.performance_history) > 5 and np.mean([x['f1_score'] for x in self.performance_history[-5:]]) < 0.7:
            st.sidebar.error("🚨 Alert: Model performance degradation detected!")

if "monitor" not in st.session_state:
    st.session_state["monitor"] = ModelMonitor()

monitor = st.session_state["monitor"]

# ============== Retraining Strategy ==============
# ============== Retraining Strategy ==============
def retrain_model():
    with st.sidebar.expander("🔧 Model Retraining"):
        if st.button("Trigger Retraining", key="retrain_btn"):
            with st.spinner("Retraining model..."):
                # تحميل البيانات
                df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

                # تحويل الأعمدة النصية إلى قيم رقمية (Label Encoding)
                categorical_cols = ['Partner', 'Dependents', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                                    'Contract', 'PaperlessBilling', 'PaymentMethod']
                
                le = LabelEncoder()
                for col in categorical_cols:
                    df[col] = le.fit_transform(df[col])

                # تقسيم البيانات إلى ميزات (X) وهدف (y)
                X = df.drop('Churn', axis=1)  # حذف عمود Churn واستخدام بقية الأعمدة
                y = df['Churn']  # الهدف هو عمود Churn

                # تقسيم البيانات إلى تدريب واختبار
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # تدريب النموذج
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                # حفظ النموذج بعد التدريب
                with open("final_stacked_model.pkl", "wb") as f:
                    pickle.dump({
                        "model": model,
                        "threshold": 0.5  # يمكن تخصيص العتبة هنا
                    }, f)

                # عرض رسالة نجاح
                st.success("Model retrained and saved successfully!")
                st.balloons()


# Display GIF in the center
st.markdown("""
    <div style="display: flex; justify-content: center; margin-bottom: 2rem;">
        <img src="https://raw.githubusercontent.com/mahmoudewies/churn-prediction-app/main/Pay%20Per%20Click%20Digital%20Marketing%20(1).gif" alt="GIF" width="600">
    </div>
""", unsafe_allow_html=True)

# Load model and threshold
with open("final_stacked_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
threshold = model_data["threshold"]

# Get expected features from the model (if available)
try:
    expected_features = model.feature_names_in_
except AttributeError:
    expected_features = None

# ============== Custom CSS ==============
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
        
        body, .stApp {
            background-color: #121212;
            color: #f0f0f0;
            font-family: 'Poppins', sans-serif;
        }
        
        .title-text {
            font-size: 2.7rem !important;
            font-weight: 600;
            color: #bb86fc;
            text-align: center;
            margin-bottom: 0.3rem;
            text-shadow: 1px 1px 4px rgba(0,0,0,0.5);
        }
        
        .subtitle-text {
            font-size: 1.1rem !important;
            color: #d1c4e9;
            text-align: center;
            margin-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ============== App Header ==============
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown('<h1 class="title-text">✨ Churn Prediction Wizard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">Predict customer churn with machine learning precision</p>', unsafe_allow_html=True)

# ============== Input Form ==============
def get_user_input():
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            SeniorCitizen = st.selectbox("Is the customer a senior citizen?", [0, 1], key="senior")
            Partner = st.selectbox("Has a partner?", ["Yes", "No"], key="partner")
            Dependents = st.selectbox("Has dependents?", ["Yes", "No"], key="dependents")
            tenure = st.slider("Tenure (months)", 0, 72, 12, key="tenure")
            InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet")
            OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"], key="security")
            OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key="backup")
            DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key="device")
            
        with col2:
            TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], key="tech")
            StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="stream_tv")
            StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key="stream_movies")
            Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="contract")
            PaperlessBilling = st.selectbox("Paperless Billing?", ["Yes", "No"], key="paperless")
            PaymentMethod = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ], key="payment")
            MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, format="%.2f", key="monthly")
            TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, format="%.2f", key="total")
            TotalServices = st.slider("Total Services Used", 0, 10, 5, key="services")
        
        st.markdown('</div>', unsafe_allow_html=True)

    data = pd.DataFrame({
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges],
        'TotalServices': [TotalServices]
    })
    
    # Ensure columns match model expectations
    if expected_features is not None:
        missing_cols = set(expected_features) - set(data.columns)
        if missing_cols:
            for col in missing_cols:
                data[col] = 0  # Add missing columns with default value
        data = data[expected_features]  # Reorder columns
    
    return data

# ============== Prediction Logic ==============
def make_prediction(input_df):
    # Encode categorical data
    le = LabelEncoder()
    categorical_cols = input_df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        input_df[col] = le.fit_transform(input_df[col])
    
    # Ensure numeric types
    input_df = input_df.astype(float)
    
    # Predict
    try:
        prediction_proba = model.predict_proba(input_df)[0][1]
        prediction = 1 if prediction_proba >= threshold else 0
        return prediction_proba, prediction
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.error(f"Input data columns: {input_df.columns.tolist()}")
        if hasattr(model, 'feature_names_in_'):
            st.error(f"Model expects columns: {model.feature_names_in_.tolist()}")
        return None, None

# ============== Main App Logic ==============
def main():
    retrain_model()
    
    # Model Monitoring Dashboard
    if st.sidebar.checkbox("Show Model Monitoring", key="monitoring"):
        st.subheader("Model Performance Monitoring")
        
        if len(monitor.performance_history) == 0:
            st.info("No performance data yet. Make some predictions first.")
        else:
            perf_df = pd.DataFrame(monitor.performance_history)
            st.line_chart(perf_df.set_index('timestamp'))
            
            latest = perf_df.iloc[-1]
            col1, col2 = st.columns(2)
            col1.metric("Latest Accuracy", f"{latest['accuracy']:.2%}")
            col2.metric("Latest F1 Score", f"{latest['f1_score']:.2%}")

    # Get user input
    input_df = get_user_input()

    # Prediction button
    if st.button("✨ Predict Churn Probability", key="predict_btn"):
        with st.spinner('Analyzing customer data...'):
            time.sleep(1.5)
            
            prediction_proba, prediction = make_prediction(input_df.copy())

            if prediction is not None:
                # Display results
                if prediction == 1:
                    st.markdown(f"""
                        <div class="danger-box">
                            <h2 style='text-align:center;margin-bottom:0.5rem'>🚨 High Churn Risk</h2>
                            <p style='text-align:center;font-size:1.25rem;margin-bottom:0'>
                                Probability: {prediction_proba:.2%}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f"""
                        <div class="success-box">
                            <h2 style='text-align:center;margin-bottom:0.5rem'>✅ Loyal Customer</h2>
                            <p style='text-align:center;font-size:1.25rem;margin-bottom:0'>
                                Retention Probability: {(1-prediction_proba):.2%}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.snow()

                # Visualization
                fig = go.Figure(data=[go.Pie(
                    labels=['Will Stay', 'Will Churn'],
                    values=[1-prediction_proba, prediction_proba],
                    marker_colors=['#00b09b', '#ff416c'],
                    hole=0.5,
                    textinfo='percent+label'
                )])
                
                fig.update_layout(
                    showlegend=False,
                    margin=dict(t=30, b=0),
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)

                # Log performance (simulating ground truth)
                # Ask user for actual (ground truth) label
                ground_truth = st.radio("🔍 Confirm Customer Churn (Ground Truth)", [None, 1, 0], key="truth")
                if ground_truth is not None:
                    monitor.log_performance([ground_truth], [prediction])
                    st.success("Performance logged successfully!")

if __name__ == "__main__":
    main()
