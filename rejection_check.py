import streamlit as st
import matplotlib.pyplot as plt
import xgboost
import shap
import pickle
import pandas as pd

def shap_waterfall_plot(model, input_df):
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], max_display=20, show=False)   
    ax.set_title("SHAP Waterfall Plot")
    plt.tight_layout()
    return fig

def predict_score(model, input_df):
    pre = model.predict_proba(input_df)[:, 1]
    return pre[0] 

def main():
    st.title("Mitra Score - SHAP Waterfall Plot")

    # Membuat input untuk memasukkan data
    st.subheader("Masukkan Data")
    df = pd.DataFrame([{'cstrip_pod' : 0.22,'risk_profile_area' : 0.15, 'max_dpd' : 2, 'ppob_amount' : 150000,
       'saving_max_amount' : 250000, 'saving_frequency' : 2, 'earn_frequency' : 1,
       'earn_amount' : 100000, 'is_cashless_repayment' : 1, 'is_cashless_disbursed' : 0,
       'total_dependence' : 3, 'umur_mitra' : 52, 'pendidikan_terakhir' : 3},
       {'cstrip_pod' : 0.12,'risk_profile_area' : 0.25, 'max_dpd' : 9, 'ppob_amount' : 0,
       'saving_max_amount' : 0, 'saving_frequency' : 0, 'earn_frequency' : 1,
       'earn_amount' : 100000, 'is_cashless_repayment' : 0, 'is_cashless_disbursed' : 0,
       'total_dependence' : 5, 'umur_mitra' : 59, 'pendidikan_terakhir' : 2}])
    
    loan_cycle = st.radio("Loan Cycle :",["First Loan", "Subsequent Loan"],index=None)

    # Load model XGBoost
    @st.cache_data
    def load_model(option):
        return pickle.load(open(option, 'rb'))    

    if loan_cycle == "First Loan":
        model = load_model('xgb_first.pkl')
        threshold = 0.131
    else:
        model = load_model('xgb_subseq.pkl')
        threshold = 0.183
        
    input_df = st.data_editor(df, num_rows="dynamic")

    data = input_df.to_dict(orient='records')

    st.write(f"Threshold Rejection = {threshold}")

    for ind,row in enumerate(data):
        score = predict_score(model, pd.DataFrame([row]))
        st.title(f"Loan **#{ind+1}**")
        st.write(f"Probability of Default (Mitra Score) : {score}")
        # Menampilkan Probability of Default
        if loan_cycle == "First Loan" and score > threshold:
            st.write(f"Loan Status: **Reject** ")
        elif loan_cycle == "First Loan" and score <= threshold:
            st.write(f"Loan Status: **Approve** ")
        elif loan_cycle == "Subsequent Loan" and score > threshold:
            st.write(f"Loan Status: **Reject** ")
        elif loan_cycle == "Subsequent Loan" and score <= threshold:
            st.write(f"Loan Status: **Approve** ")

        # Menampilkan plot waterfall SHAP
        st.subheader("SHAP Waterfall Plot")
        d = pd.DataFrame([row])
        waterfall_plot = shap_waterfall_plot(model, d)
        st.pyplot(waterfall_plot)

if __name__ == "__main__":
    main()
