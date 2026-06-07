"""
dashboard/app.py
────────────────
Streamlit dashboard for the Telco Customer Churn Predictor.

Two pages:
    1. Single Prediction  — score one customer via a form
    2. Batch Predictions  — upload CSV, score all customers

Calls the live FastAPI backend at API_URL.
"""

import json
import urllib.request
import urllib.error

import pandas as pd
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────

API_URL = 'https://churn-predictor-api-gonj.onrender.com'

st.set_page_config(
    page_title = 'Churn Predictor',
    page_icon  = '📊',
    layout     = 'wide',
)


# ── Helper functions ──────────────────────────────────────────────────────────

def api_post(endpoint: str, data: dict):
    """POST to the API and return the response dict."""
    url  = f'{API_URL}{endpoint}'
    body = json.dumps(data).encode()
    req  = urllib.request.Request(
        url, data=body,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        return None, f"API error {e.code}: {e.read().decode()}"
    except Exception as e:
        return None, str(e)


def api_get(endpoint: str):
    """GET from the API and return the response dict."""
    url = f'{API_URL}{endpoint}'
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read()), None
    except Exception as e:
        return None, str(e)


def risk_colour(tier: str) -> str:
    return {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}.get(tier, '⚪')


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title('📊 Churn Predictor')
st.sidebar.markdown('---')
page = st.sidebar.radio(
    'Navigate',
    ['Single Prediction', 'Batch Predictions', 'Model Info'],
)

st.sidebar.markdown('---')
st.sidebar.markdown(f'**API:** [{API_URL}]({API_URL}/docs)')
st.sidebar.markdown('Built with LightGBM + FastAPI + Streamlit')


# ══════════════════════════════════════════════════════════════════════════════
# Page 1 — Single Prediction
# ══════════════════════════════════════════════════════════════════════════════

if page == 'Single Prediction':

    st.title('Customer Churn Risk Predictor')
    st.markdown('Fill in the customer details below to get a churn risk score.')
    st.markdown('---')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader('Demographics')
        gender          = st.selectbox('Gender', ['Male', 'Female'])
        senior_citizen  = st.selectbox('Senior Citizen', [0, 1],
                                       format_func=lambda x: 'Yes' if x else 'No')
        partner         = st.selectbox('Partner', ['Yes', 'No'])
        dependents      = st.selectbox('Dependents', ['Yes', 'No'])

    with col2:
        st.subheader('Account')
        tenure          = st.slider('Tenure (months)', 0, 72, 12)
        contract        = st.selectbox('Contract',
                                       ['Month-to-month', 'One year', 'Two year'])
        paperless       = st.selectbox('Paperless Billing', ['Yes', 'No'])
        payment         = st.selectbox('Payment Method', [
                            'Electronic check', 'Mailed check',
                            'Bank transfer (automatic)',
                            'Credit card (automatic)'])
        monthly_charges = st.number_input('Monthly Charges ($)',
                                          min_value=0.01, max_value=200.0,
                                          value=65.0, step=0.01)
        total_charges   = st.number_input('Total Charges ($)',
                                          min_value=0.0, max_value=10000.0,
                                          value=float(tenure * monthly_charges),
                                          step=0.01)

    with col3:
        st.subheader('Services')
        phone_service   = st.selectbox('Phone Service', ['Yes', 'No'])
        multiple_lines  = st.selectbox('Multiple Lines',
                                       ['Yes', 'No', 'No phone service'])
        internet        = st.selectbox('Internet Service',
                                       ['Fiber optic', 'DSL', 'No'])

        if internet != 'No':
            online_sec  = st.selectbox('Online Security', ['Yes', 'No'])
            online_bk   = st.selectbox('Online Backup', ['Yes', 'No'])
            device_prot = st.selectbox('Device Protection', ['Yes', 'No'])
            tech_sup    = st.selectbox('Tech Support', ['Yes', 'No'])
            streaming_tv= st.selectbox('Streaming TV', ['Yes', 'No'])
            streaming_mv= st.selectbox('Streaming Movies', ['Yes', 'No'])
        else:
            online_sec = online_bk = device_prot = tech_sup = 'No internet service'
            streaming_tv = streaming_mv = 'No internet service'

    st.markdown('---')

    if st.button('🔍 Predict Churn Risk', type='primary', use_container_width=True):

        customer = {
            'gender'          : gender,
            'SeniorCitizen'   : senior_citizen,
            'Partner'         : partner,
            'Dependents'      : dependents,
            'tenure'          : tenure,
            'Contract'        : contract,
            'PaperlessBilling': paperless,
            'PaymentMethod'   : payment,
            'MonthlyCharges'  : monthly_charges,
            'TotalCharges'    : total_charges,
            'PhoneService'    : phone_service,
            'MultipleLines'   : multiple_lines,
            'InternetService' : internet,
            'OnlineSecurity'  : online_sec,
            'OnlineBackup'    : online_bk,
            'DeviceProtection': device_prot,
            'TechSupport'     : tech_sup,
            'StreamingTV'     : streaming_tv,
            'StreamingMovies' : streaming_mv,
        }

        with st.spinner('Getting prediction...'):
            result, error = api_post('/predict', customer)

        if error:
            st.error(f'API Error: {error}')
        else:
            prob  = result['churn_probability']
            tier  = result['risk_tier']
            pred  = result['prediction']
            reason= result['reason']
            icon  = risk_colour(tier)

            c1, c2, c3 = st.columns(3)
            c1.metric('Churn Probability', f'{prob:.1%}')
            c2.metric('Prediction', 'Will Churn' if pred == 1 else 'Will Stay')
            c3.metric('Risk Tier', f'{icon} {tier}')

            if tier == 'High':
                st.error(f'⚠️ {reason}')
            elif tier == 'Medium':
                st.warning(f'⚡ {reason}')
            else:
                st.success(f'✅ {reason}')

            st.progress(prob, text=f'Churn probability: {prob:.1%}')


# ══════════════════════════════════════════════════════════════════════════════
# Page 2 — Batch Predictions
# ══════════════════════════════════════════════════════════════════════════════

elif page == 'Batch Predictions':

    st.title('Batch Churn Scoring')
    st.markdown('Upload a CSV file of customers to score all of them at once.')

    st.info('''
    **CSV must have these columns:**
    gender, SeniorCitizen, Partner, Dependents, tenure, Contract,
    PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges,
    PhoneService, MultipleLines, InternetService, OnlineSecurity,
    OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
    ''')

    uploaded = st.file_uploader('Upload customer CSV', type=['csv'])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f'Loaded **{len(df)}** customers')
        st.dataframe(df.head(5), use_container_width=True)

        if st.button('🚀 Score All Customers', type='primary'):

            customers = df.to_dict(orient='records')
            payload   = {'customers': customers}

            with st.spinner(f'Scoring {len(customers)} customers...'):
                result, error = api_post('/predict/batch', payload)

            if error:
                st.error(f'API Error: {error}')
            else:
                preds = result['predictions']
                df['churn_probability'] = [p['churn_probability'] for p in preds]
                df['prediction']        = [p['prediction']        for p in preds]
                df['risk_tier']         = [p['risk_tier']         for p in preds]
                df['reason']            = [p['reason']            for p in preds]

                st.markdown('---')
                c1, c2, c3, c4 = st.columns(4)
                c1.metric('Total Customers', result['total'])
                c2.metric('🔴 High Risk',    result['high_risk'])
                c3.metric('🟡 Medium Risk',  result['medium_risk'])
                c4.metric('🟢 Low Risk',     result['low_risk'])

                st.markdown('### Results')
                st.dataframe(
                    df[['churn_probability', 'risk_tier', 'reason']
                       + [c for c in df.columns
                          if c not in ['churn_probability',
                                       'risk_tier', 'reason',
                                       'prediction']]],
                    use_container_width=True,
                )

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label     = '⬇️ Download Results CSV',
                    data      = csv,
                    file_name = 'churn_predictions.csv',
                    mime      = 'text/csv',
                    use_container_width=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# Page 3 — Model Info
# ══════════════════════════════════════════════════════════════════════════════

elif page == 'Model Info':

    st.title('Model Information')

    with st.spinner('Loading model info...'):
        info, error = api_get('/model-info')

    if error:
        st.error(f'Could not load model info: {error}')
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Performance Metrics')
            metrics = {
                'ROC-AUC'  : info['test_roc_auc'],
                'PR-AUC'   : info['test_pr_auc'],
                'F1 Score' : info['test_f1'],
                'Precision': info['test_precision'],
                'Recall'   : info['test_recall'],
            }
            for name, val in metrics.items():
                st.metric(name, f'{val:.4f}')

        with col2:
            st.subheader('Configuration')
            st.metric('Decision Threshold', info['optimal_threshold'])
            st.metric('Calibration',        info['calibration'])
            st.metric('Model Version',      info['model_version'])
            st.metric('Brier Score',        f"{info['brier_score']:.4f}")

        st.markdown('---')
        st.subheader('What these metrics mean')
        st.markdown(f'''
        - **ROC-AUC {info["test_roc_auc"]:.3f}** — the model correctly ranks
          a random churner above a random non-churner {info["test_roc_auc"]:.0%} of the time
        - **PR-AUC {info["test_pr_auc"]:.3f}** — precision-recall performance
          on the positive (churn) class; more informative than ROC-AUC for imbalanced data
        - **Threshold {info["optimal_threshold"]}** — probability above this value
          is classified as churn; tuned to maximise F1 on the validation set
        - **Brier Score {info["brier_score"]:.3f}** — calibration quality;
          lower is better (0 = perfect calibration)
        ''')