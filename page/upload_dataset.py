import streamlit as st
import pandas as pd

def upload_and_view_dataset():
    st.title("Upload Dataset")
    uploaded_file = st.file_uploader("Pilih file", type=["xlsx", "csv"])

    if uploaded_file is not None:
        try:
            # Mencoba membaca dataset
            if uploaded_file.name.endswith("xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                # Inisialisasi flag untuk melacak keberhasilan parsing
                success = False

                # Mencoba membaca dengan koma sebagai pemisah default
                try:
                    df = pd.read_csv(uploaded_file)
                    success = True
                except pd.errors.ParserError:
                    pass

                # Mencoba membaca dengan titik koma jika koma gagal
                if not success:
                    try:
                        st.warning("Error dengan pemisah default. Mencoba pemisah titik koma...")
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, delimiter=';')
                        success = True
                    except pd.errors.ParserError:
                        pass

                # Mencoba membaca dengan pemisah tab jika titik koma gagal
                if not success:
                    try:
                        st.warning("Error dengan pemisah titik koma. Mencoba pemisah tab...")
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, delimiter='\t')
                        success = True
                    except pd.errors.ParserError:
                        pass

                if not success:
                    st.error("Gagal mem-parsing file dengan pemisah yang tersedia.")
                    return None

            # Menampilkan preview dataframe
            st.write("Preview Dataset:")
            st.write(df.head())

            return df

        except Exception as e:
            st.error(f"Kesalahan saat membaca dataset: {str(e)}")
            return None

    st.info("Silakan unggah file untuk melanjutkan.")
    return None
