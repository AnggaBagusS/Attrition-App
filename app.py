"""
Entry point utama untuk deployment ke Streamlit Community Cloud.
Mendukung:
- streamlit run app.py
- streamlit run streamlit_app.py
- python app.py (otomatis memicu streamlit run)

Untuk menjalankan versi Flask legacy:
- python flask_app.py
"""
import sys
import os

if __name__ == "__main__" and "streamlit" not in sys.modules and not any("streamlit" in arg for arg in sys.argv):
    print("🚀 Memulai server Streamlit...")
    os.system("streamlit run streamlit_app.py")
else:
    # Saat dijalankan oleh Streamlit runner (Streamlit Cloud / Local)
    import streamlit_app
