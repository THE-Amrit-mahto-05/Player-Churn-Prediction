import streamlit as st
import pandas as pd
import joblib
import time

from src.data_validator import validate_and_prepare_player_data, dict_to_ml_dataframe
from src.agents.workflow import run_churn_analysis_workflow
from src.pdf_generator import create_pdf_report

st.set_page_config(
    page_title="Player Retention AI",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_ml_model():
    return joblib.load("churn_model.pkl")

model = load_ml_model()

st.markdown("""
<style>
    /* Global App Background */
    .stApp {
        background: linear-gradient(135deg, #09090b 0%, #171720 100%);
        color: #e2e8f0;
        font-family: 'Inter', 'Roboto', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Glassmorphism Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Animations */
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(220, 38, 38, 0); }
        100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
    }
    @keyframes pulse-orange {
        0% { box-shadow: 0 0 0 0 rgba(249, 115, 22, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(249, 115, 22, 0); }
        100% { box-shadow: 0 0 0 0 rgba(249, 115, 22, 0); }
    }
    
    .neon-logo {
        text-align: center;
        font-size: 60px;
        color: #38bdf8;
        text-shadow: 0 0 10px #38bdf8, 0 0 20px #818cf8, 0 0 40px #c084fc;
        margin-bottom: -15px;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    .gradient-text {
        background: linear-gradient(to right, #00f2fe, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        letter-spacing: -1px;
    }
    
    /* Glowing Risk Badges */
    .badge-critical { 
        background: rgba(220, 38, 38, 0.15); color: #fca5a5; 
        border: 1px solid rgba(220, 38, 38, 0.5); 
        animation: pulse-red 2s infinite;
        padding: 6px 12px; border-radius: 8px; font-weight: bold; 
    }
    .badge-high { 
        background: rgba(249, 115, 22, 0.15); color: #fdba74; 
        border: 1px solid rgba(249, 115, 22, 0.5); 
        animation: pulse-orange 2s infinite;
        padding: 6px 12px; border-radius: 8px; font-weight: bold; 
    }
    .badge-healthy { 
        background: rgba(34, 197, 94, 0.15); color: #86efac; 
        border: 1px solid rgba(34, 197, 94, 0.5); 
        padding: 6px 12px; border-radius: 8px; font-weight: bold; 
    }

    .prio-high { background: #ef4444; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75em; font-weight: bold;}
    .prio-medium { background: #f59e0b; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75em; font-weight: bold;}
    .prio-low { background: #10b981; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75em; font-weight: bold;}

    .rag-box {
        background: rgba(14, 165, 233, 0.05);
        border-left: 4px solid #0ea5e9;
        padding: 15px 20px;
        border-radius: 0 8px 8px 0;
        font-size: 0.9em;
        line-height: 1.6;
        color: #e0f2fe;
    }
    
    .progress-bar-container {
        width: 100%;
        background-color: rgba(255,255,255,0.1);
        border-radius: 10px;
        margin-top: 5px;
    }
    .progress-bar {
        height: 8px;
        border-radius: 10px;
        background: linear-gradient(90deg, #ef4444, #f97316);
    }
</style>
""", unsafe_allow_html=True)

raw_data = None
display_data = None
results_df = pd.DataFrame()

with st.sidebar:
    st.markdown("<div class='neon-logo'>🎮</div>", unsafe_allow_html=True)
    st.markdown("<h2 class='gradient-text' style='text-align:center;'>Player Retention AI</h2>", unsafe_allow_html=True)
    st.markdown("""<div style='color: #a1a1aa; font-size: 0.9em; text-align:center; margin-bottom: 20px;'>Enterprise Churn Prediction & automated RAG intervention system.</div>""", unsafe_allow_html=True)
    
    st.markdown("### DATA INGESTION")
    uploaded_file = st.file_uploader("Upload Player CSV", type="csv")
    
    if uploaded_file is not None:
        with st.spinner("Ingesting telemetry and running Machine Learning risk models..."):
            raw_data = pd.read_csv(uploaded_file)
            
            if "PlayerID" in raw_data.columns:
                display_data = raw_data.drop("PlayerID", axis=1)
            else:
                display_data = raw_data.copy()

            if "EngagementLevel" in display_data.columns:
                display_data = display_data.drop("EngagementLevel", axis=1)

            processed_dicts = [validate_and_prepare_player_data(row.to_dict()) for _, row in display_data.iterrows()]
            ml_df = pd.concat([dict_to_ml_dataframe(pd) for pd in processed_dicts], ignore_index=True)
            
            probabilities = model.predict_proba(ml_df)[:, 1] * 100

            results_df = display_data.copy()
            results_df['Churn Prob (%)'] = probabilities.round(1)
            
            def get_risk_badge(prob):
                if prob >= 75: return "🔴 CRITICAL"
                elif prob >= 40: return "🟠 HIGH"
                else: return "🟢 HEALTHY"
                
            results_df['Risk Status'] = results_df['Churn Prob (%)'].apply(get_risk_badge)
            time.sleep(0.5) 

        st.markdown("<br>### GLOBAL ANALYTICS", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='glass-card' style='padding:15px; margin-bottom:10px;'>
            <div style='color:#a1a1aa; font-size:11px; font-weight:bold;'>TOTAL PLAYERS ANALYZED</div>
            <div style='font-size:20px; font-weight:900;'>{len(results_df)}</div>
        </div>
        <div class='glass-card' style='padding:15px; margin-bottom:10px;'>
            <div style='color:#a1a1aa; font-size:11px; font-weight:bold;'>CRITICAL CHURN RISK</div>
            <div style='font-size:20px; font-weight:900; color:#ef4444;'>{len(results_df[results_df['Churn Prob (%)'] >= 75])}</div>
        </div>
        <div class='glass-card' style='padding:15px; margin-bottom:10px;'>
            <div style='color:#a1a1aa; font-size:11px; font-weight:bold;'>AVG RISK GAUGE</div>
            <div style='font-size:20px; font-weight:900; color:#f59e0b;'>{round(results_df['Churn Prob (%)'].mean(), 1)}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.caption("Powered by LangGraph & ChromaDB")

if uploaded_file is None:
    st.markdown("<h1 class='gradient-text' style='font-size: 3.5rem; text-align: center; margin-top: 10vh;'>Player Retention AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #a1a1aa; font-size: 1.2em; text-align: center;'>Please upload your player telemetry CSV in the sidebar to begin.</p>", unsafe_allow_html=True)
else:
    st.markdown("<h1 class='gradient-text'>Risk Control Center</h1>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["At-Risk Intervention", "Live Telemetry Feed"])
    
    with tab2:
        st.dataframe(results_df, use_container_width=True)
        
    with tab1:
        at_risk_df = results_df[results_df['Churn Prob (%)'] >= 40].reset_index(drop=True)
        
        if at_risk_df.empty:
            st.success("All systems optimal. No players are currently at risk.")
        else:
            col_list, col_action = st.columns([2, 1])
            with col_list:
                st.dataframe(at_risk_df[['Risk Status', 'Churn Prob (%)', 'PlayTimeHours', 'SessionsPerWeek', 'GameGenre']], height=250, use_container_width=True)
            
            with col_action:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("### AI Agent")
                selected_idx = st.selectbox(
                    "Target Player ID (Index):", 
                    options=at_risk_df.index,
                    format_func=lambda x: f"Idx {x} - {at_risk_df.loc[x, 'Risk Status'].split()[1]} Risk"
                )
                generate_pressed = st.button("Generate AI Strategy", type="primary", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            if generate_pressed:
                st.markdown("---")
                selected_player_row = at_risk_df.loc[selected_idx]
                
                # --- Execution UI ---
                import time as time_lib
                start_time = time_lib.time()
                
                with st.status("Agentic System Working... (Estimated Time: ~4s)", expanded=True) as status:
                    st.write("Validating formatting and applying safe tensors...")
                    time_lib.sleep(0.6)
                    player_dict = selected_player_row.drop(['Churn Prob (%)', 'Risk Status']).to_dict()
                    cleaned_player = validate_and_prepare_player_data(player_dict)
                    
                    st.write("Analyzer Agent: Decoding structural behavior patterns...")
                    time_lib.sleep(0.8)
                    st.write("RAG Retriever: Searching Knowledge Base (retention_strategies.md)...")
                    time_lib.sleep(0.8)
                    st.write("Strategy & Report Agents: Structuring executive JSON output...")
                    
                    result = run_churn_analysis_workflow(
                        cleaned_player, 
                        churn_prob=selected_player_row['Churn Prob (%)'],
                        genre=cleaned_player.get('GameGenre', 'Unknown')
                    )
                    
                    end_time = time_lib.time()
                    elapsed = round(end_time - start_time, 2)
                    
                    if result.get('status') == 'complete':
                        status.update(label=f"Final Report Structured locally in {elapsed}s", state="complete", expanded=False)
                    else:
                        status.update(label="Agent Workflow Failed", state="error", expanded=False)
                        st.error(f"Workflow error: {result.get('status')}")
                        st.stop()
                
                # --- Premium Report Render ---
                st.markdown("<h2 class='gradient-text'>Executive Interaction Plan</h2>", unsafe_allow_html=True)
                
                report = result.get('report', {})
                rag_context = result.get('rag_context', '')
                prob_val = selected_player_row['Churn Prob (%)']
                
                if prob_val >= 75: 
                    badge_css = "badge-critical"
                    risk_txt = "CRITICAL"
                elif prob_val >= 40: 
                    badge_css = "badge-high"
                    risk_txt = "HIGH"
                else: 
                    badge_css = "badge-healthy"
                    risk_txt = "HEALTHY"
                    
                trend = report.get('risk_analysis', {}).get('trend', '---')
                
                st.markdown(f"""
                <div class='glass-card'>
                    <div style='display:flex; justify-content:space-around; align-items:center;'>
                        <div style='text-align:center; width:33%;'>
                            <div style='color:#a1a1aa; font-size:12px; margin-bottom:8px;'>CHURN PROBABILITY</div>
                            <span class='{badge_css}' style='font-size:24px;'>{prob_val}%</span>
                            <div class='progress-bar-container'><div class='progress-bar' style='width: {prob_val}%;'></div></div>
                        </div>
                        <div style='text-align:center; width:33%;'>
                            <div style='color:#a1a1aa; font-size:12px; margin-bottom:8px;'>SYSTEM CLASSIFICATION</div>
                            <span class='{badge_css}' style='font-size:24px;'>{risk_txt} RISK</span>
                        </div>
                        <div style='text-align:center; width:33%;'>
                            <div style='color:#a1a1aa; font-size:12px; margin-bottom:8px;'>ENGAGEMENT TREND</div>
                            <span style='font-size:24px; font-weight:bold; color:#e2e8f0;'>{trend}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                content_col, rag_col = st.columns([1.7, 1.3])
                
                with content_col:
                    st.markdown("### Player Profile Summary")
                    def stream_text(text):
                        for word in text.split(" "):
                            yield word + " "
                            time.sleep(0.04)
                    
                    summary_text = report.get('player_profile', {}).get('summary', '')
                    st.write_stream(stream_text(summary_text))
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    st.markdown("### Clean Action Cards (Recommendations)")
                    for rec in report.get('recommendations', []):
                        time.sleep(0.6)
                        prio = rec.get('priority', 'Medium')
                        prio_class = f"prio-{prio.lower()}"
                        
                        st.markdown(f"""
                        <div class='glass-card' style='padding: 20px; border-left: 4px solid {"#ef4444" if prio=="High" else "#f59e0b" if prio=="Medium" else "#10b981"};'>
                            <div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px;'>
                                <b style='font-size: 1.1em;'>{rec.get('action')}</b>
                                <span class='{prio_class}'>{prio} Priority</span>
                            </div>
                            <div style='color: #a1a1aa; font-size: 0.95em; line-height: 1.5;'>
                                <b>Rationale:</b> {rec.get('rationale')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                with rag_col:
                    st.markdown("### RETENTION KNOWLEDGE BASE REFERENCES")
                    time_lib.sleep(0.5)
                    st.caption("Document context retrieved locally via ChromaDB vectors mapping to the player's specific churn signals.")
                    st.markdown(f"<div class='rag-box'><b>Source:</b> <code>retention_strategies.md</code><br><br>{rag_context}</div>", unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.caption(f"**Ethical Disclaimer:** {report.get('disclaimers', 'This AI decision-support system provides recommendations; human review is required before executing interventions.')}")

                # --- PDF Export ---
                st.markdown("---")
                pdf_bytes = create_pdf_report(
                    player_id=str(selected_idx),
                    prob_val=prob_val,
                    report_data=report
                )
                
                col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
                with col_btn2:
                    st.download_button(
                        label="📄 Download Executive Strategy as PDF",
                        data=pdf_bytes,
                        file_name=f"Retention_Plan_{selected_idx}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )