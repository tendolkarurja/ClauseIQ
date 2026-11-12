"""
ClauseIQ: AI-Powered Legal Contract Analysis & Due Diligence Workspace (Streamlit)
Tailored specifically for Indian Buy, Sell, and Lease Contracts.
Governed under:
- Transfer of Property Act, 1882 (TPA)
- Indian Contract Act, 1872 (ICA)
- Model Tenancy Act, 2021 (MTA)
- Registration Act, 1908 & Indian Stamp Act, 1899

Split-Screen Workspace:
- Left Column: Document-Wide Due Diligence Memorandum (Scorecard, Contradictions, Ambiguities, Statutory Gaps, Full Covenants Schedule, Export)
- Right Column: Context-Aware Diagnostic Legal Chatbot with Quick Prompts & 100% Zero-Remediation Enforcement
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'
import sys
import json
import tempfile
import streamlit as st

# Ensure project root is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from clauseiq_engine import (
    perform_full_document_audit,
    answer_chatbot_query,
    extract_contract
)

# Page configuration
st.set_page_config(
    page_title="ClauseIQ — Indian Legal Due Diligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished Legal Tech interface
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 20px 24px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 20px;
    }
    .main-title {
        color: #38bdf8;
        font-size: 26px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-subtitle {
        color: #94a3b8;
        font-size: 13px;
        margin-top: 6px;
        margin-bottom: 0;
    }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .metric-val {
        font-size: 24px;
        font-weight: bold;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 11px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-high {
        background: #7f1d1d;
        color: #fca5a5;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 11px;
    }
    .badge-tension {
        background: #78350f;
        color: #fde68a;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 11px;
    }
    .badge-low {
        background: #14532d;
        color: #86efac;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 11px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Sample documents directory
TRAINING_DOCS_DIR = os.path.join(BASE_DIR, "training_docs")
SAMPLE_DOCS = {
    "Lease Deed (With Contradiction)": os.path.join(TRAINING_DOCS_DIR, "lease_doc(contradict).pdf"),
    "Lease Deed (High Risk & Contradiction)": os.path.join(TRAINING_DOCS_DIR, "lease_doc(risky_contradict).pdf"),
    "Sale Deed / Conveyance Deed": os.path.join(TRAINING_DOCS_DIR, "sell_doc.pdf"),
    "Standard Lease Deed (Baseline)": os.path.join(TRAINING_DOCS_DIR, "lease_doc.pdf")
}

# Session State Initialization
if "audit_results" not in st.session_state:
    st.session_state["audit_results"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "current_file_label" not in st.session_state:
    st.session_state["current_file_label"] = ""
if "pending_quick_query" not in st.session_state:
    st.session_state["pending_quick_query"] = None

# Header Banner
st.markdown("""
<div class="main-header">
    <h1 class="main-title">⚖️ ClauseIQ — Indian Legal Due Diligence Workspace</h1>
    <p class="main-subtitle">
        Autonomous Diagnostic Issue-Spotter for Indian Buy, Sell, and Lease Contracts.<br>
        Grounded in statutory compliance under <b>Transfer of Property Act 1882</b>, <b>Model Tenancy Act 2021</b>, <b>Indian Contract Act 1872</b> & <b>Registration Act 1908</b>.
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.subheader("📁 Contract Selection")
    
    doc_choice = st.selectbox(
        "Choose Pre-loaded Deed",
        options=list(SAMPLE_DOCS.keys()),
        index=0
    )
    
    uploaded_file = st.file_uploader(
        "Or Upload Custom Deed (PDF)",
        type=["pdf"],
        help="Upload an authentic Indian conveyance or tenancy agreement."
    )
    
    st.markdown("---")
    
    audit_clicked = st.button("🚀 Run Full Due Diligence Audit", type="primary", use_container_width=True)
    
    if st.button("🔄 Clear Chat & Reset Session", use_container_width=True):
        st.session_state["chat_history"] = []
        st.session_state["pending_quick_query"] = None
        st.rerun()

    st.markdown("---")
    st.markdown("### 🏛️ Statutory Framework")
    st.caption("""
    - **Transfer of Property Act, 1882** (§ 55, § 108)
    - **Model Tenancy Act, 2021** (§ 13, § 15, Sched. II)
    - **Indian Contract Act, 1872** (§ 56, § 73)
    - **Indian Stamp Act, 1899 & Reg. Act, 1908**
    """)
    
    st.markdown("---")
    st.markdown("""
    > **Zero-Remediation Notice:**  
    > ClauseIQ is an automated diagnostic issue-spotting tool. It identifies legal friction without drafting amendments or dispensing legal advice. All remedies remain under consulting advocate discretion.
    """)

# Execute Audit on Button Click or if not loaded yet
if audit_clicked or (st.session_state["audit_results"] is None and not uploaded_file):
    with st.spinner("Executing full macro-scan and calibrated NLI audit..."):
        if uploaded_file is not None:
            # Save uploaded document to system temp directory (never pollutes project root)
            temp_path = os.path.join(tempfile.gettempdir(), f"clauseiq_{uploaded_file.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            target_path = temp_path
            label = uploaded_file.name
        else:
            target_path = SAMPLE_DOCS[doc_choice]
            label = doc_choice

        audit_res = perform_full_document_audit(target_path)
        st.session_state["audit_results"] = audit_res
        st.session_state["current_file_label"] = label
        
        # Reset chat history for new document with introductory message
        doc_data = audit_res["doc_data"]
        sc = audit_res["scorecard"]
        welcome_text = (
            f"👋 Welcome to **ClauseIQ**! I have completed the automated Due Diligence Audit of **{doc_data['filename']}**.\n\n"
            f"- **Health Index:** `{sc['overall_health_score']}/100`\n"
            f"- **High Risk Covenants:** `{sc['high_risk_count']}`\n"
            f"- **Calibrated Contradictions:** `{sc['high_certainty_contradictions']} High Certainty, {sc['potential_tensions']} Tension`\n"
            f"- **Ambiguities:** `{sc['ambiguity_clauses_count']}`\n"
            f"- **Statutory Gaps:** `{sc['statutory_omissions_count']}`\n\n"
            "Ask me anything about these covenants, or click any of the quick prompt buttons below!"
        )
        st.session_state["chat_history"] = [{"role": "assistant", "content": welcome_text}]
        st.rerun()

# Check if audit is loaded
audit = st.session_state["audit_results"]

if audit and audit.get("status") == "success":
    sc = audit["scorecard"]
    doc_data = audit["doc_data"]
    contradictions = audit["contradictions"]
    ambiguities = audit["ambiguities"]
    omissions = audit["omissions"]
    clause_risk = audit["clause_risk_profile"]

    # ========================================================
    # SPLIT SCREEN LAYOUT: Left = Memorandum, Right = Chatbot
    # ========================================================
    col_memo, col_chat = st.columns([1.1, 0.9], gap="medium")

    # --------------------------------------------------------
    # LEFT COLUMN: DUE DILIGENCE LEGAL MEMORANDUM
    # --------------------------------------------------------
    with col_memo:
        st.subheader("📄 Due Diligence Legal Audit Memorandum")

        # Health Banner & Metrics
        health = sc["overall_health_score"]
        health_color = "#22c55e" if health >= 80 else ("#eab308" if health >= 60 else "#ef4444")
        health_status = "STABLE / STANDARD" if health >= 80 else ("MODERATE TENSION" if health >= 60 else "CRITICAL ATTENTION REQUIRED")

        # Top summary cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {health_color};">
                <div class="metric-label">Health Index</div>
                <div class="metric-val" style="color: {health_color};">{health}<span style="font-size: 14px; color: #94a3b8;">/100</span></div>
                <div style="font-size: 10px; color: {health_color}; font-weight: 600;">{health_status}</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #ef4444;">
                <div class="metric-label">High Risk</div>
                <div class="metric-val" style="color: #ef4444;">{sc['high_risk_count']}</div>
                <div style="font-size: 10px; color: #94a3b8;">Mutual clashes</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #f97316;">
                <div class="metric-label">Calibrated Clashes</div>
                <div class="metric-val" style="color: #f97316;">{sc['high_certainty_contradictions'] + sc['potential_tensions']}</div>
                <div style="font-size: 10px; color: #94a3b8;">θ ≥ 0.85 verified</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #a855f7;">
                <div class="metric-label">Statutory Gaps</div>
                <div class="metric-val" style="color: #a855f7;">{sc['statutory_omissions_count']}</div>
                <div style="font-size: 10px; color: #94a3b8;">MTA / TPA omissions</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # Tabs for Memorandum Breakdown
        tab_scorecard, tab_contra, tab_amb, tab_gaps, tab_covenants, tab_export = st.tabs([
            "📊 Scorecard",
            "⚖️ Contradictions",
            "🔍 Ambiguities",
            "📋 Statutory Gaps",
            "📜 All Covenants",
            "💾 Export Memo"
        ])

        # TAB 1: EXECUTIVE SCORECARD
        with tab_scorecard:
            st.markdown(f"#### Instrument: `{doc_data['title']}`")
            st.caption(f"**Filename:** `{doc_data['filename']}` | **Covenants Extracted:** `{sc['total_clauses_scanned']}`")
            if doc_data.get("parties"):
                st.markdown("##### 👥 Parties & Property Identification")
                for p in doc_data["parties"]:
                    st.markdown(f"- **{p}**")
                st.write("")

            st.markdown("##### Executive Risk Audit")
            scorecard_table = [
                {"Audit Dimension": "Total Covenants Scanned", "Finding": f"{sc['total_clauses_scanned']}", "Audit Status": "Complete"},
                {"Audit Dimension": "High Risk Covenants", "Finding": f"{sc['high_risk_count']}", "Audit Status": "Review Required" if sc['high_risk_count'] > 0 else "Clear"},
                {"Audit Dimension": "Calibrated Contradictions (θ ≥ 0.95)", "Finding": f"{sc['high_certainty_contradictions']} Direct Clashes", "Audit Status": "Critical"},
                {"Audit Dimension": "Potential Operational Tensions", "Finding": f"{sc['potential_tensions']} Inconsistencies", "Audit Status": "Advisory"},
                {"Audit Dimension": "Contextual Ambiguity Triggers", "Finding": f"{sc['ambiguity_clauses_count']} clauses flagged", "Audit Status": "Caution"},
                {"Audit Dimension": "Statutory Omission Gaps", "Finding": f"{sc['statutory_omissions_count']} missing covenants", "Audit Status": "Non-compliant"}
            ]
            st.table(scorecard_table)

            st.warning("""
            **Mandatory Advocate Discretion Notice:**  
            This audit scorecard identifies diagnostic friction points under Indian law. It does NOT prescribe legal amendments or provide legal advice. Revisions remain within the professional discretion of consulting counsel.
            """)

        # TAB 2: CONTRADICTIONS REGISTER
        with tab_contra:
            st.markdown("#### ⚖️ Calibrated Contradictions Register")
            if not contradictions:
                st.success("✅ **No contradictory covenant pairs detected** meeting the calibrated threshold (θ ≥ 0.85).")
            else:
                st.caption(f"Identified **{len(contradictions)} conflicting covenant pair(s)** via calibrated DeBERTa-base-MNLI inference:")
                for idx, item in enumerate(contradictions, 1):
                    is_high = item["confidence_tier"] == "HIGH_CERTAINTY"
                    badge_class = "badge-high" if is_high else "badge-tension"
                    tier_label = "HIGH CERTAINTY (θ ≥ 0.95)" if is_high else "POTENTIAL TENSION (0.85 ≤ θ < 0.95)"
                    
                    with st.expander(f"Conflict #{idx}: {item['clause_a_id']} ⟷ {item['clause_b_id']} ({item['confidence_tier']})", expanded=is_high):
                        st.markdown(f"<span class='{badge_class}'>{tier_label}</span> &nbsp; **Contradiction Score:** `{item['contradiction_score']}` &nbsp; | &nbsp; **Flow:** `{item['winning_direction']}`", unsafe_allow_html=True)
                        st.markdown(f"**Diagnostic Finding:** {item['diagnostic_note']}")
                        st.markdown("---")
                        
                        col_ca, col_cb = st.columns(2)
                        with col_ca:
                            st.markdown(f"**`{item['clause_a_id']}` Text:**")
                            st.info(f"\"{item['clause_a_text']}\"")
                        with col_cb:
                            st.markdown(f"**`{item['clause_b_id']}` Text:**")
                            st.info(f"\"{item['clause_b_text']}\"")

        # TAB 3: AMBIGUITY CATALOG
        with tab_amb:
            st.markdown("#### 🔍 Contextual Ambiguity & Vagueness Catalog")
            if not ambiguities:
                st.success("✅ **No subjective or unbounded triggers flagged.** All clauses evaluated contain bounded standards.")
            else:
                st.caption("Covenants containing subjective language, undefined cadences, or uncalibrated economic escalation under Indian law:")
                for amb_item in ambiguities:
                    cid = amb_item["clause_id"]
                    with st.expander(f"Ambiguity in `{cid}`", expanded=True):
                        st.write(f"**Covenant Text:** *\"{amb_item['clause_text']}\"*")
                        for a in amb_item["ambiguities"]:
                            st.markdown(f"- **Vague Trigger:** <span class='badge-tension'>{a['vague_term']}</span>", unsafe_allow_html=True)
                            st.markdown(f"  - **Statutory Flaw:** {a['flaw_description']}")
                            st.markdown(f"  - **Governing Statute:** *{a['statutory_reference']}*")

        # TAB 4: STATUTORY GAPS
        with tab_gaps:
            st.markdown("#### 📋 Statutory Compliance & Omission Gap Analysis")
            if not omissions:
                st.success("✅ **No standard statutory omissions detected.** Document incorporates all baseline statutory covenants.")
            else:
                st.caption("Standard protective covenants and statutory mandates absent from this deed:")
                for om in omissions:
                    prio_badge = "badge-high" if om["severity"] == "HIGH" else "badge-tension"
                    st.markdown(f"##### ⚠️ {om['covenant_name']} &nbsp; <span class='{prio_badge}'>{om['severity']} PRIORITY</span>", unsafe_allow_html=True)
                    st.markdown(f"- **Statutory Mandate:** *{om['statutory_mandate']}*")
                    st.markdown(f"- **Diagnostic Impact:** {om['diagnostic_observation']}")
                    st.markdown("---")

        # TAB 5: ALL COVENANTS
        with tab_covenants:
            st.markdown("#### 📜 Complete Covenants Schedule")
            st.caption("All extracted numbered covenants with individual diagnostic risk classifications:")
            for cid, text in doc_data["clauses"].items():
                r = clause_risk.get(cid, "LOW")
                badge_class = "badge-high" if r == "HIGH" else ("badge-tension" if r == "MEDIUM" else "badge-low")
                st.markdown(f"**`{cid}`** &nbsp; <span class='{badge_class}'>{r} RISK</span>", unsafe_allow_html=True)
                st.markdown(f"> {text}")
                st.write("")

        # TAB 6: EXPORT MEMORANDUM
        with tab_export:
            st.markdown("#### 💾 Export Due Diligence Memorandum")
            st.caption("Download the complete legal audit memorandum in Markdown format for client reporting:")
            st.download_button(
                label="📥 Download Memorandum (.md)",
                data=audit["memo_markdown"],
                file_name=f"Due_Diligence_Memo_{doc_data['filename']}.md",
                mime="text/markdown",
                type="primary",
                use_container_width=True
            )
            st.markdown("---")
            st.text_area("Memorandum Raw Text Preview", value=audit["memo_markdown"], height=350)

    # --------------------------------------------------------
    # RIGHT COLUMN: INTERACTIVE DIAGNOSTIC CHATBOT
    # --------------------------------------------------------
    with col_chat:
        st.subheader("💬 Diagnostic Legal Assistant")
        st.caption("Contextually grounded in active contract covenants & Due Diligence findings.")

        # Quick Prompt Chips
        st.markdown("**Quick Diagnostic Inquiries:**")
        qp_row1_c1, qp_row1_c2 = st.columns(2)
        with qp_row1_c1:
            if st.button("🔍 Why is Clause 17 flagged?", use_container_width=True):
                st.session_state["pending_quick_query"] = "Why was clause 17 flagged?"
        with qp_row1_c2:
            if st.button("⚖️ Show Contradictions", use_container_width=True):
                st.session_state["pending_quick_query"] = "Show all calibrated contradictions in this document"

        qp_row2_c1, qp_row2_c2 = st.columns(2)
        with qp_row2_c1:
            if st.button("⚠️ List Ambiguities", use_container_width=True):
                st.session_state["pending_quick_query"] = "List all ambiguous terms under Model Tenancy Act"
        with qp_row2_c2:
            if st.button("📋 Missing Covenants", use_container_width=True):
                st.session_state["pending_quick_query"] = "What statutory covenants are missing?"

        qp_row3_c1, qp_row3_c2 = st.columns(2)
        with qp_row3_c1:
            if st.button("✍️ Can you rewrite Clause 17?", use_container_width=True):
                st.session_state["pending_quick_query"] = "Rewrite clause 17 to fix the contradiction with clause 15"
        with qp_row3_c2:
            if st.button("📊 Health Overview", use_container_width=True):
                st.session_state["pending_quick_query"] = "Give me the document health scorecard overview"

        st.markdown("---")

        # Chat Message Container
        chat_container = st.container(height=480)
        with chat_container:
            for message in st.session_state["chat_history"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Process user query (either from chat_input or quick prompt buttons)
        user_query = st.chat_input("Ask about covenants, conflicts, or statutory compliance...")
        active_query = st.session_state["pending_quick_query"] or user_query

        if active_query:
            # Clear pending quick query
            st.session_state["pending_quick_query"] = None
            
            # Append user message
            st.session_state["chat_history"].append({"role": "user", "content": active_query})
            
            # Generate assistant response
            with st.spinner("Analyzing covenant context..."):
                response = answer_chatbot_query(active_query, audit, st.session_state["chat_history"])
            
            # Append assistant message
            st.session_state["chat_history"].append({"role": "assistant", "content": response})
            st.rerun()

else:
    st.info("👈 Please select a sample deed or upload a PDF from the sidebar, then click **'Run Full Due Diligence Audit'** to begin.")
