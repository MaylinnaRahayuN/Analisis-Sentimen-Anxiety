import streamlit as st

def home():
    st.header("ANALISIS SENTIMEN DI MEDIA SOSIAL MENGGUNAKAN H2O GRADIENT BOOSTING DAN TF-IDF VECTORIZATION UNTUK DETEKSI KECEMASAN (ANXIETY) SISWA")
    st.write("Desain Penelitian: Exploratory Data Analysis (EDA), Preprocessing Data, TF-IDF Vectorization, H2O Gradient Boosting, Evaluasi Model")

    # Atur layout dengan rasio kolom
    col1, col2 = st.columns([2, 1])  # Rasio kolom diubah menjadi 2:1

    with col1:
        st.write("""
        Penggunaan media sosial meningkat sebagai sarana mengungkapkan pendapat dan emosi.
        Korelasi kuat antara ekspresi emosi di media sosial dengan gangguan mental (Zhu et al., 2023).
        Anxiety menjadi garis depan dari berbagai gangguan psikologis (Wang et al., 2023) yang memicu dampak seperti stress hingga bunuh diri.
        Analisis sentimen di media sosial dapat mengungkap pola emosional yang membantu deteksi dini kecemasan.
        """)

    with col2:
        st.image('C:/Users/Maylinna Rahayu N/Downloads/skripsi/ilustrasi_anxiety.jpg', 
                 caption='Illustration of Anxiety', use_column_width=True)

    # Tambahkan sedikit spasi untuk memperbaiki visual
    st.write("\n\n")

def main():
    home()

if __name__ == "__main__":
    main()
