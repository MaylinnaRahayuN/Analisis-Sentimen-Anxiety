import h2o
from h2o.estimators import H2OGradientBoostingEstimator
import pandas as pd
import streamlit as st
import logging
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def train_model(tfidf_matrix, df):
    if tfidf_matrix is None or df is None or df.empty:
        st.error("Input data is missing or empty.")
        return
    
    logging.debug("Input TF-IDF Matrix shape: %s", tfidf_matrix.shape)
    logging.debug("Input DataFrame shape: %s", df.shape)

    # Inisialisasi H2O
    h2o.init(max_mem_size='12G', nthreads=4)

    # Menggabungkan TF-IDF matrix dengan kolom 'label' dari df
    try:
        df_hex = h2o.H2OFrame(pd.concat([pd.DataFrame(tfidf_matrix.toarray()), df[['label']]], axis=1))
        df_hex['label'] = df_hex['label'].asfactor()
    except ValueError as e:
        st.error(f"Error converting DataFrame to H2OFrame: {e}")
        return

    # Train/test split (70/30)
    train_hex, test_hex = df_hex.split_frame(ratios=[0.7], seed=123)
    
    # Define model
    fit_1 = H2OGradientBoostingEstimator(ntrees=25,
                                         max_depth=6,
                                         min_rows=15,
                                         learn_rate=0.005,
                                         sample_rate=1,
                                         col_sample_rate=0.5,
                                         nfolds=5,  # Cross-validation
                                         score_each_iteration=True,
                                         stopping_metric='AUC',
                                         stopping_rounds=5,
                                         seed=123)

    # Train the model
    t1 = time.time()
    fit_1.train(x=list(df_hex.columns[:-1]), y='label', training_frame=train_hex)
    t2 = time.time()
    st.write('Elapsed time [s]: ', np.round(t2 - t1, 2))

    # Visualisasi AUC dari cross-validation
    for i in range(5):
        cv_model_temp = fit_1.cross_validation_models()[i]
        df_cv_score_history = cv_model_temp.score_history()
        plt.figure()
        plt.scatter(df_cv_score_history.number_of_trees, df_cv_score_history.training_auc, c='blue', label='training')
        plt.scatter(df_cv_score_history.number_of_trees, df_cv_score_history.validation_auc, c='darkorange', label='validation')
        plt.title(f'CV {1+i} - Scoring History [AUC]')
        plt.xlabel('Number of Trees')
        plt.ylabel('AUC')
        plt.ylim(0.9, 1)
        plt.legend()
        plt.grid()
        st.pyplot(plt)
        plt.close()

    # Cross-validation metrics
    cv_metrics = fit_1.cross_validation_metrics_summary()
    st.write("Cross-Validation Metrics Summary:")
    st.dataframe(cv_metrics.as_data_frame())

    # ===================== Evaluasi Model ===================== #
    
    # Evaluasi model pada test set
    performance = fit_1.model_performance(test_hex)
    
    # Mengambil berbagai metrik performa
    metrics = {
        "Accuracy": performance.accuracy()[0][1],
        "Error Rate": 1 - performance.accuracy()[0][1],
        "Precision": performance.precision()[0][1],
        "Recall": performance.recall()[0][1],
        "F1-Score": performance.F1()[0][1],
        "MSE": performance.mse(),
        "RMSE": performance.rmse(),
        "AUC": performance.auc()
    }

    # Tampilkan metrik performa
    st.write("Model Performance Metrics on Test Data:")
    for metric, value in metrics.items():
        st.write(f"{metric}: {value:.4f}")

    # ===================== Testing dan Confusion Matrix ===================== #
    
    # Prediksi pada data uji
    predictions_test = fit_1.predict(test_hex)
    predictions_test_df = predictions_test.as_data_frame()

    # Mengambil label sebenarnya dari data uji
    y_true_test = test_hex['label'].as_data_frame()

    # Hapus baris dengan NaN pada y_true_test dan prediksi
    valid_idx = ~y_true_test['label'].isna() & ~predictions_test_df['predict'].isna()
    y_true_test = y_true_test[valid_idx]
    predictions_test_df = predictions_test_df[valid_idx]

    # Menghitung confusion matrix untuk data uji
    conf_matrix_test = confusion_matrix(y_true_test, predictions_test_df['predict'])

    # Menampilkan confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix - Test Data')
    st.pyplot(plt)
    
    # Menghitung nilai specificity
    TN = conf_matrix_test[0][0]  # True Negative
    FP = conf_matrix_test[0][1]  # False Positive
    specificity = TN / (TN + FP)
    st.write(f'Specificity: {specificity:.4f}')
    
    # Menghitung dan menampilkan classification report
    st.write("Classification Report:")
    st.text(classification_report(y_true_test, predictions_test_df['predict']))
    
    return fit_1


