import streamlit as st
from calc import main as calc_main
import tempfile
import os
import settings

# Default settings (could be imported from settings.py too)
DEFAULT_SETTINGS = {
    "SECTIONS_PER_PANEL": 3,
    "OUTGOINGS_PER_SECTION": 10,
    "SPARE_OUTGOINGS_PER_SECTION": 1,
    "MAX_CIRCUIT_CB_CURRENT": 20.0,
    "MAX_DISTANCE_TO_PANEL": 150,
    "MAX_DISTANCE_BETWEEN_CIRCUITS": 50,
    "PANEL_BLOCK_SIZE": 10000,
    "CIRCUIT_BLOCK_SIZE": 5000,
    "BLOCK_TEXT_SIZE": 2000
}

# Initialize settings in session state
for key, value in st.session_state.items():
    if hasattr(settings, key):
        setattr(settings, key, value)

st.set_page_config(page_title="CGT - Circuit Grouping Tool", layout="centered")
st.title("CGT - Circuit Grouping Tool")

# =========================
# 🔹 Tabs for UI Sections
# =========================
tab1, tab2 = st.tabs(["🧾 Main", "⚙️ Settings"])

# =========================
# ⚙️ Settings Tab
# =========================
with tab2:
    st.subheader("⚙️ Configuration Settings")

    st.session_state["SECTIONS_PER_PANEL"] = st.number_input("Sections per Panel", min_value=1, value=st.session_state["SECTIONS_PER_PANEL"])
    st.session_state["OUTGOINGS_PER_SECTION"] = st.number_input("Outgoings per Section", min_value=1, value=st.session_state["OUTGOINGS_PER_SECTION"])
    st.session_state["SPARE_OUTGOINGS_PER_SECTION"] = st.number_input("Spare Outgoings per Section", min_value=0, value=st.session_state["SPARE_OUTGOINGS_PER_SECTION"])
    st.session_state["MAX_CIRCUIT_CB_CURRENT"] = st.number_input("Max Circuit CB Current (A)", min_value=1.0, value=st.session_state["MAX_CIRCUIT_CB_CURRENT"])
    st.session_state["MAX_DISTANCE_TO_PANEL"] = st.number_input("Max Distance to Panel (m)", min_value=1, value=st.session_state["MAX_DISTANCE_TO_PANEL"])
    st.session_state["MAX_DISTANCE_BETWEEN_CIRCUITS"] = st.number_input("Max Distance Between Circuits (m)", min_value=1, value=st.session_state["MAX_DISTANCE_BETWEEN_CIRCUITS"])
    st.session_state["PANEL_BLOCK_SIZE"] = st.number_input("Panel Block Size", min_value=1000, value=st.session_state["PANEL_BLOCK_SIZE"])
    st.session_state["CIRCUIT_BLOCK_SIZE"] = st.number_input("Circuit Block Size", min_value=1000, value=st.session_state["CIRCUIT_BLOCK_SIZE"])
    st.session_state["BLOCK_TEXT_SIZE"] = st.number_input("Block Text Size", min_value=500, value=st.session_state["BLOCK_TEXT_SIZE"])

# =========================
# 📥 Template Download
# =========================
with tab1:
    st.subheader("📥 Input Template")

    TEMPLATE_PATH = "input.xlsx"

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

    # File Upload Section
    def reset_session_state():
        for key in ["uploaded_filename", "xlsx_bytes", "dxf_bytes"]:
            st.session_state.pop(key, None)

    st.subheader("🖇 Upload your Excel file")
    uploaded_file = st.file_uploader(" ", type="xlsx")

    if uploaded_file is not None:
        if (
            "uploaded_filename" not in st.session_state
            or uploaded_file.name != st.session_state["uploaded_filename"]
        ):
            reset_session_state()
            st.session_state["uploaded_filename"] = uploaded_file.name

    if uploaded_file and st.button("Start"):
        with st.spinner("Grouping..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as input_tmp:
                input_tmp.write(uploaded_file.read())
                input_file_path = input_tmp.name

            output_xlsx_path = os.path.join(tempfile.gettempdir(), "output_result.xlsx")
            output_dxf_path = os.path.join(tempfile.gettempdir(), "output_result.dxf")

            # Pass settings to calc_main
            calc_main(
                input_file_path,
                output_xlsx_path,
                output_dxf_path,
                settings={
                    key: st.session_state[key] for key in DEFAULT_SETTINGS
                }
            )

            with open(output_xlsx_path, "rb") as f1:
                st.session_state["xlsx_bytes"] = f1.read()
            with open(output_dxf_path, "rb") as f2:
                st.session_state["dxf_bytes"] = f2.read()

    # Download Results
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
