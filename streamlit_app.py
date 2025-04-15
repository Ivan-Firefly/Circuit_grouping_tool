# app.py
import streamlit as st
from calc import main as calc_main
import tempfile
import os


TEMPLATE_PATH = "input.xlsx"  # Make sure this path is correct

st.set_page_config(page_title="CGT - Circui grouping tool", layout="centered")
st.title("CGT - Circuit Grouping Tool")

# =========================
# 🔹 Section: Download Template
# =========================
st.subheader("📥 Input Template")

if os.path.exists(TEMPLATE_PATH):
    with open(TEMPLATE_PATH, "rb") as f:
        template_data = f.read()

    st.download_button(
        label="📄 Download XLSX",
        data=template_data,
        file_name="template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.warning("⚠️ Template file not found in the project directory.")


st.divider()

# =========================
# 🔹 Section: File Upload and Processing
# =========================

# Function to reset session state
def reset_session_state():
    for key in ["uploaded_filename", "xlsx_bytes", "dxf_bytes"]:
        st.session_state.pop(key, None)

st.subheader("🖇 Upload your Excel file")
uploaded_file = st.file_uploader(" ", type="xlsx")

# Detect new file and reset state
if uploaded_file is not None:
    if (
        "uploaded_filename" not in st.session_state
        or uploaded_file.name != st.session_state["uploaded_filename"]
    ):
        reset_session_state()
        st.session_state["uploaded_filename"] = uploaded_file.name

# Process the file
if uploaded_file and st.button("Start"):
    with st.spinner("Groping..."):
        # Save input file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as input_tmp:
            input_tmp.write(uploaded_file.read())
            input_file_path = input_tmp.name

        # Define output paths
        output_xlsx_path = os.path.join(tempfile.gettempdir(), "output_result.xlsx")
        output_dxf_path = os.path.join(tempfile.gettempdir(), "output_result.dxf")

        # Run your main function
        calc_main(
            input_file_path,
            output_xlsx_path,
            output_dxf_path,
            settings={key: st.session_state[key] for key in DEFAULT_SETTINGS}
        )


        # Save outputs in session state
        with open(output_xlsx_path, "rb") as f1:
            st.session_state["xlsx_bytes"] = f1.read()
        with open(output_dxf_path, "rb") as f2:
            st.session_state["dxf_bytes"] = f2.read()

# =========================
# 🔹 Section: Download Results
# =========================
if "xlsx_bytes" in st.session_state and "dxf_bytes" in st.session_state:
    st.success("✅ Processing complete. Download your files below:")

    st.download_button(
        label="📄 Download XLSX",
        data=st.session_state["xlsx_bytes"],
        file_name="result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.download_button(
        label="✏ Download DXF",
        data=st.session_state["dxf_bytes"],
        file_name="result.dxf",
        mime="application/dxf"
    )
