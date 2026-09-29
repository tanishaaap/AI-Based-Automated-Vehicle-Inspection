import os
import sys
import streamlit as st
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
sys.path.append(BASE_DIR)
from main import run_full_pipeline  # noqa: E402
from auth import create_user, verify_user, save_inspection, get_history  # noqa: E402
from pdf_report import generate_pdf_bytes  # noqa: E402

st.set_page_config(
    page_title="AI Vehicle Inspection Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%; background-color: #0d6efd; color: white; font-weight: bold;
        border-radius: 8px; padding: 0.5rem 1rem; border: none;
    }
    .stButton>button:hover { background-color: #0b5ed7; }
    .card {
        background: white !important; color: #212529 !important; padding: 20px; border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px;
    }
    .card * { color: #212529 !important; }
    .metric-badge {
        background-color: #e9ecef; padding: 6px 12px; border-radius: 6px;
        font-weight: 600; color: #495057; margin-right: 6px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def warm_up_models():
    """Importing these modules triggers their module-level model loading
    (SentenceTransformer, YOLO). Caching this means Streamlit only does that
    expensive work once per server process, not on every button click."""
    import retriever          # noqa: F401
    import predict_dent       # noqa: F401
    import predict_scratch    # noqa: F401
    return True


warm_up_models()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""


def render_auth_page():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## 🚗 Vehicle Inspection Portal")
        st.write("Please sign in or create an account to run inspections.")

        auth_mode = st.radio("Choose action", ["Login", "Sign Up"], horizontal=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if auth_mode == "Sign Up":
            confirm_pass = st.text_input("Confirm Password", type="password")
            if st.button("Create Account"):
                if password != confirm_pass:
                    st.error("Passwords do not match.")
                else:
                    ok, message = create_user(username, password)
                    if ok:
                        st.success(message)
                    else:
                        st.error(message)
        else:
            if st.button("Login"):
                if verify_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")
        st.markdown("</div>", unsafe_allow_html=True)


def render_inspection_app():
    st.sidebar.markdown(f"### Welcome, {st.session_state.username} 👋")
    if st.sidebar.button("Log Out"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("### 📋 Vehicle Metadata")
    make = st.sidebar.text_input("Make", "Honda")
    model_name = st.sidebar.text_input("Model", "City")
    year = st.sidebar.number_input("Year", min_value=2000, max_value=2026, value=2018)
    color = st.sidebar.text_input("Color", "White")
    panel_name = st.sidebar.selectbox("Panel Inspected", [
        "rear_left_door", "front_left_door", "front_right_door", "rear_right_door",
        "hood", "trunk", "bumper"
    ])

    st.sidebar.divider()
    st.sidebar.markdown("### 🎚️ Detection Sensitivity")
    dent_conf = st.sidebar.slider("Dent confidence threshold", 0.05, 0.90, 0.15, 0.05,
                                    help="Lower catches more dents but with more false positives.")
    scratch_conf = st.sidebar.slider("Scratch confidence threshold", 0.05, 0.90, 0.25, 0.05)

    with st.sidebar.expander("📜 Past Inspections"):
        history = get_history(st.session_state.username, limit=10)
        if not history:
            st.caption("No inspections yet.")
        for h in history:
            st.caption(f"{h['created_at'][:16]} -- {h['image_name']}")

    st.title("🔍 AI Vehicle Inspection Dashboard")
    st.write("Upload a vehicle photo with the ArUco calibration marker visible for grounded damage detection and dimension estimation.")

    uploaded_file = st.file_uploader("Upload vehicle image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            st.markdown("#### Uploaded Image")
            st.image(image, use_container_width=True)
        with col_up2:
            st.markdown("#### Inspection Controls")
            run_btn = st.button("🚀 Run Comprehensive AI Inspection")

        if run_btn:
            temp_path = os.path.join(BASE_DIR, "outputs", f"_temp_{uploaded_file.name}")
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            vehicle_info = {"make": make, "model": model_name, "year": year, "color": color}

            with st.spinner("Analyzing dents, scratches, and panel gaps..."):
                try:
                    inspection_json, summary_text, composite_path = run_full_pipeline(
                        temp_path, panel_name, vehicle_info,
                        dent_conf=dent_conf, scratch_conf=scratch_conf,
                    )
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")
                    inspection_json, summary_text, composite_path = None, None, None

            os.remove(temp_path)

            if inspection_json:
                save_inspection(st.session_state.username, uploaded_file.name, vehicle_info, inspection_json, summary_text)
                st.success("Inspection analysis complete!")

                st.divider()
                st.subheader("🛡️ Annotated Inspection")
                st.image(composite_path, use_container_width=True,
                          caption="Panel Gap | Dents | Scratches -- real detections, each panel shows only that defect type")

                st.divider()
                st.subheader("Detailed Findings")
                col_gaps, col_dents, col_scratches = st.columns(3)

                with col_gaps:
                    st.markdown("### 🟢 Panel Gaps")
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    if not inspection_json["panel_gaps"]:
                        st.warning("No gap measured -- marker or seam not found in this photo.")
                    for gap in inspection_json["panel_gaps"]:
                        cls = gap.get("classification", {})
                        st.markdown(f"**Panel:** {gap['location'].replace('_', ' ').title()}")
                        st.markdown(
                            f"<span class='metric-badge'>{gap['measurement_mm']}mm</span>"
                            f"<span class='metric-badge'>{cls.get('status', 'unknown').replace('_',' ').title()}</span>",
                            unsafe_allow_html=True,
                        )
                    st.markdown("</div>", unsafe_allow_html=True)

                with col_dents:
                    st.markdown("### 🟠 Dents")
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    if not inspection_json["dents"]:
                        st.caption("No dents detected.")
                    for dent in inspection_json["dents"]:
                        cls = dent.get("classification", {})
                        cal_note = "" if dent.get("calibration_valid") else " (uncalibrated est.)"
                        st.markdown(
                            f"<span class='metric-badge'>Conf: {dent['confidence']:.0%}</span>"
                            f"<span class='metric-badge'>Size: {dent['diameter_mm']}mm{cal_note}</span>"
                            f"<span class='metric-badge'>{cls.get('status','unknown').title()}</span>",
                            unsafe_allow_html=True,
                        )
                    st.markdown("</div>", unsafe_allow_html=True)

                with col_scratches:
                    st.markdown("### 🔵 Scratches")
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    if not inspection_json["scratches"]:
                        st.caption("No scratches detected.")
                    for scratch in inspection_json["scratches"]:
                        area = scratch.get("physical_area_mm2")
                        area_text = f"{area}mm²" if area is not None else f"{scratch['pixel_area_px2']}px² (uncal.)"
                        st.markdown(
                            f"<span class='metric-badge'>Conf: {scratch['confidence']:.0%}</span>"
                            f"<span class='metric-badge'>Area: {area_text}</span>",
                            unsafe_allow_html=True,
                        )
                    st.markdown("</div>", unsafe_allow_html=True)

                st.divider()
                st.subheader("📝 Inspection Summary")
                st.markdown(f"""
                <div style="background-color: #ffffff; padding: 20px; border-left: 5px solid #0d6efd; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <p style="font-size: 16px; color: #333333 !important; line-height: 1.6;">{summary_text}</p>
                </div>
                """, unsafe_allow_html=True)

                pdf_bytes = generate_pdf_bytes(inspection_json, summary_text, composite_path, vehicle_info)
                st.download_button(
                    label="📥 Download Full PDF Inspection Report",
                    data=pdf_bytes,
                    file_name=f"Inspection_Report_{make}_{model_name}.pdf",
                    mime="application/pdf",
                )

                with st.expander("Raw inspection JSON"):
                    st.json(inspection_json)


if not st.session_state.logged_in:
    render_auth_page()
else:
    render_inspection_app()