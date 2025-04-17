import streamlit as st
from f1 import execute_full_process
import tempfile
import os

# Default values for all settings
DEFAULT_VALUES = {
    "CHAIN_TYPE": "daisy",
    "PANEL_TYPE": "locked",
    "SECTIONS_PER_PANEL": 3,
    "OUTGOINGS_PER_SECTION": 10,
    "SPARE_OUTGOINGS_PER_SECTION": 1,
    "MAX_CB_CURRENT": 20.0,
    "MAX_DISTANCE_TO_PANEL": 150000,
    "PANEL_BLOCK_SIZE": 1000,
    "CIRCUIT_BLOCK_SIZE": 300,
    "GROUP_BLOCK_SIZE":400,
    "BLOCK_TEXT_SIZE": 250,
    "INIT_RADIUS": 5000,
    "RADIUS_STEP": 5000,
    "MAX_ITERATIONS": 3,
    "MAX_CIRCUIT_PER_GROUP": 4,
    "DXF_GROUPING": True
}


# Initialize session_state with default values
for key, value in DEFAULT_VALUES.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Initialize processing status
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

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
    st.header("⚙️ Configuration Settings")

    st.subheader("Panel configuration")
    st.session_state["SECTIONS_PER_PANEL"] = st.number_input("Sections per Panel", min_value=1,
                                                             value=st.session_state["SECTIONS_PER_PANEL"])
    st.session_state["OUTGOINGS_PER_SECTION"] = st.number_input("Outgoings per Section", min_value=1,
                                                                value=st.session_state["OUTGOINGS_PER_SECTION"])
    st.session_state["SPARE_OUTGOINGS_PER_SECTION"] = st.number_input("Spare Outgoings per Section", min_value=0,
                                                                      value=st.session_state[
                                                                          "SPARE_OUTGOINGS_PER_SECTION"])

    st.session_state["MAX_CB_CURRENT"] = st.number_input("Max Circuit CB Current (A)", min_value=1.0,
                                                                 value=st.session_state["MAX_CB_CURRENT"])

    st.session_state["MAX_CIRCUIT_PER_GROUP"] = st.number_input("Max Circuits Per Group", min_value=1,
                                                                value=st.session_state["MAX_CIRCUIT_PER_GROUP"])
    st.subheader("Algorithm Configuration")
    st.session_state["CHAIN_TYPE"] = st.selectbox("Chain Type",
                                                  options=["daisy", "MB"],
                                                  index=0 if st.session_state["CHAIN_TYPE"] == "daisy" else 1)

    st.session_state["PANEL_TYPE"] = st.selectbox("Panel Type",
                                                  options=["locked", "free"],
                                                  index=0 if st.session_state["PANEL_TYPE"] == "locked" else 1)

    st.session_state["MAX_ITERATIONS"] = st.number_input("Max Iterations",
                                                         min_value=1,
                                                         value=st.session_state["MAX_ITERATIONS"])

    st.session_state["INIT_RADIUS"] = st.number_input("Initial Radius",
                                                      min_value=1000,
                                                      value=st.session_state["INIT_RADIUS"])

    st.session_state["RADIUS_STEP"] = st.number_input("Radius Step",
                                                      min_value=1000,
                                                      value=st.session_state["RADIUS_STEP"])



    st.subheader("Limitations")
    st.session_state["MAX_DISTANCE_TO_PANEL"] = st.number_input("Max Distance to Panel", min_value=1,
                                                                value=st.session_state["MAX_DISTANCE_TO_PANEL"])

    st.subheader("Dxf settings")
    st.session_state["PANEL_BLOCK_SIZE"] = st.number_input("Panel Block Size", min_value=1,
                                                           value=st.session_state["PANEL_BLOCK_SIZE"])
    st.session_state["CIRCUIT_BLOCK_SIZE"] = st.number_input("Circuit Block Size", min_value=1,
                                                             value=st.session_state["CIRCUIT_BLOCK_SIZE"])
    st.session_state["GROUP_BLOCK_SIZE"] = st.number_input("Group Block Size", min_value=1,
                                                             value=st.session_state["GROUP_BLOCK_SIZE"])

    st.session_state["BLOCK_TEXT_SIZE"] = st.number_input("Block Text Size", min_value=1,
                                                          value=st.session_state["BLOCK_TEXT_SIZE"])
    st.session_state["DXF_GROUPING"] = st.checkbox("Group blocks?", value=st.session_state["DXF_GROUPING"])

# =========================
# 📥 Template Download
# =========================
with tab1:
    st.header("📥 Input Template")

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
        st.session_state.processing_complete = False


    # Only show upload section if processing is not complete
    if not st.session_state.processing_complete:
        st.subheader("🖇 Upload your Excel file")
        uploaded_file = st.file_uploader(" ", type="xlsx")

        if uploaded_file is not None:
            if (
                    "uploaded_filename" not in st.session_state
                    or uploaded_file.name != st.session_state["uploaded_filename"]
            ):
                reset_session_state()
                st.session_state["uploaded_filename"] = uploaded_file.name

        # Only show the Start button if file is uploaded
        if uploaded_file:
            if st.button("Start"):
                with st.spinner("Grouping..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as input_tmp:
                        input_tmp.write(uploaded_file.read())
                        input_file_path = input_tmp.name

                    output_xlsx_path = os.path.join(tempfile.gettempdir(), "output_result.xlsx")
                    output_dxf_path = os.path.join(tempfile.gettempdir(), "output_result.dxf")

                    # Use the execute_full_process function with all parameters
                    execute_full_process(
                        input_file_path,
                        output_xlsx_path,
                        chain_type=st.session_state["CHAIN_TYPE"],
                        init_radius=st.session_state["INIT_RADIUS"],
                        radius_step=st.session_state["RADIUS_STEP"],
                        max_iterations=st.session_state["MAX_ITERATIONS"],
                        max_current_per_group=st.session_state["MAX_CB_CURRENT"],
                        max_circuit_per_group=st.session_state["MAX_CIRCUIT_PER_GROUP"],
                        panel_type=st.session_state["PANEL_TYPE"],
                        sections_per_panel=st.session_state["SECTIONS_PER_PANEL"],
                        max_outgoings_per_section=st.session_state["OUTGOINGS_PER_SECTION"]-st.session_state["SPARE_OUTGOINGS_PER_SECTION"],
                        max_distance_limit=st.session_state["MAX_DISTANCE_TO_PANEL"],
                        dxf_outputfile=output_dxf_path,
                        panel_side=st.session_state["PANEL_BLOCK_SIZE"],
                        circuit_radius=st.session_state["CIRCUIT_BLOCK_SIZE"],
                        square_size=st.session_state["GROUP_BLOCK_SIZE"],
                        text_height=st.session_state["BLOCK_TEXT_SIZE"],
                        grouping=st.session_state["DXF_GROUPING"]
                    )

                    with open(output_xlsx_path, "rb") as f1:
                        st.session_state["xlsx_bytes"] = f1.read()
                    with open(output_dxf_path, "rb") as f2:
                        st.session_state["dxf_bytes"] = f2.read()

                    # Set processing complete flag to hide the button and upload section
                    st.session_state.processing_complete = True
                    st.rerun()

    # Download Results
    if "xlsx_bytes" in st.session_state and "dxf_bytes" in st.session_state:
        st.success("✅ Processing complete. Download your files below:")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.download_button(
                label="📄 Download XLSX",
                data=st.session_state["xlsx_bytes"],
                file_name="Grouping.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.download_button(
                label="✏ Download DXF",
                data=st.session_state["dxf_bytes"],
                file_name="Grouping.dxf",
                mime="application/dxf"
            )

        # Add a button to start over if needed - in the right column
        with col2:
            # Add some vertical space to align with the download buttons
            if st.button("♲ Process Another File"):
                reset_session_state()
                st.rerun()
