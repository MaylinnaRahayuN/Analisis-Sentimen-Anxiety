import importlib.util
import streamlit as st
import pandas as pd

def load_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        st.success(f"Module {module_name} berhasil dimuat.")
        return module
    except Exception as e:
        st.error(f"Gagal memuat module {module_name}: {str(e)}")
        return None

def check_home():
    try:
        home_module = load_module("home", "home.py")
        if home_module:
            home_module.home()
    except Exception as e:
        st.error(f"Error pada home.py: {str(e)}")

def check_upload_dataset():
    try:
        upload_module = load_module("upload_dataset", "upload_dataset.py")
        if upload_module:
            upload_module.upload_and_view_dataset()
    except Exception as e:
        st.error(f"Error pada upload_dataset.py: {str(e)}")

def check_pre_processing():
    try:
        pre_process_module = load_module("pre_processing", "pre_processing.py")
        if pre_process_module:
            sample_df = pd.DataFrame({"text": ["I am anxious", "Feeling stressed", "All good here"]})
            pre_process_module.preprocess_data(sample_df)
    except Exception as e:
        st.error(f"Error pada pre_processing.py: {str(e)}")

def check_tfidf():
    try:
        tfidf_module = load_module("TF_IDF", "TF_IDF.py")
        if tfidf_module:
            sample_df = pd.DataFrame({"text_tokens": ["anxiety stress", "calm relaxed"]})
            tfidf_module.compute_tfidf(sample_df)
    except Exception as e:
        st.error(f"Error pada TF_IDF.py: {str(e)}")

def check_modelling():
    try:
        modelling_module = load_module("modelling", "modelling.py")
        if modelling_module:
            # Here you might need to create a valid tfidf_matrix and df for actual testing
            modelling_module.train_model(None, None)  # Use actual data
    except Exception as e:
        st.error(f"Error pada modelling.py: {str(e)}")

def check_all():
    check_home()
    check_upload_dataset()
    check_pre_processing()
    check_tfidf()
    check_modelling()

if __name__ == "__main__":
    check_all()
