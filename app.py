import os
import time
import json
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

APP_NAME = "SentinelAI"
st.set_page_config(page_title=APP_NAME, page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# ---------- Visual system ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --bg:#0B0B16; --panel:#111022; --panel2:#171526; --purple:#7C3AED; --lav:#A78BFA; --gold:#E8C878; --text:#F5F0FF; --muted:#B9AFE8; --line:#3A2A68; }
html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
.stApp { background:radial-gradient(ellipse at 60% 0%,#17112e 0%,var(--bg) 54%,#080711 100%); color:var(--text); }
[data-testid="stHeader"] { background:rgba(11,11,22,.85); }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#100d24,#0d0b1d); border-right:1px solid #292044; min-width: 285px; }
section[data-testid="stSidebar"] * { color:#d9ceff; }
section[data-testid="stSidebar"] h1 { color:#fff; font-size:1.65rem; }
h1,h2,h3 { letter-spacing:-.025em; color:#fff; }
h1 { color:var(--gold); }
.stMarkdown p, label { color:#dcd4ff; }
div[data-testid="stMetric"] { background:linear-gradient(145deg,#151127,#0f0d20); border:1px solid var(--line); padding:17px 18px; border-radius:14px; min-height:120px; box-shadow: 0 8px 22px rgba(0,0,0,.12); }
div[data-testid="stMetricLabel"] { color:#c7baf7; }
div[data-testid="stMetricValue"] { color:#e5d9ff; }
div[data-testid="stMetricDelta"] { color:#48e6b0; }
.stButton button, .stDownloadButton button, button[kind="primary"] { border-radius:11px; border:1px solid #8b5cf6; background:linear-gradient(120deg,#6d28d9,#7c3aed); color:white; font-weight:700; }
.stButton button:hover, .stDownloadButton button:hover { border-color:var(--gold); color:white; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
[data-testid="stAlert"] { border-radius:12px; }
[data-testid="stTabs"] button { font-weight:700; }
[data-testid="stExpander"] { background:#121024; border:1px solid var(--line); border-radius:12px; }
input, textarea, [data-baseweb="select"] > div { background:#121024 !important; color:#f5f0ff !important; border-color:#493779 !important; }
.small-muted { color:#b9afe8; font-size:.9rem; }
.eyebrow { color:#a78bfa; text-transform:uppercase; letter-spacing:.15em; font-size:.72rem; font-weight:800; }
.hero { padding:22px 25px; border:1px solid var(--line); border-radius:16px; background:linear-gradient(110deg,rgba(124,58,237,.18),rgba(17,16,34,.88)); margin-bottom:18px; }
.pill { display:inline-block; padding:5px 10px; border-radius:999px; font-size:.75rem; font-weight:800; background:#291a4d; color:#d9c9ff; border:1px solid #6542a4; }
hr { border-color:#2b2146; }
code { color:#d5bfff !important; }
/* dashboard panels */
.dash-card { background:linear-gradient(145deg,rgba(20,17,41,.96),rgba(13,11,28,.96)); border:1px solid #3b2b68; border-radius:14px; padding:18px 20px; margin-bottom:12px; box-shadow: 0 8px 24px rgba(0,0,0,.10); }
.dash-title { font-size:1.05rem; font-weight:800; color:#fff; margin-bottom:12px; }
.dash-muted { color:#b9afe8; font-size:.9rem; }
.alert-row { padding:9px 0; border-bottom:1px solid #28203f; font-size:.88rem; }
.status-dot { color:#49e2b0; font-weight:800; }
/* Sidebar navigation: elevated royal-purple / gold treatment */
[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    color:#E8C878 !important; font-size:.70rem !important; font-weight:800 !important;
    letter-spacing:.16em; text-transform:uppercase; margin:0 0 9px 3px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    display:flex; flex-direction:column; gap:7px; padding:4px 0 7px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] { gap:7px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    position:relative; display:flex; align-items:center; gap:10px;
    width:100%; min-height:44px; box-sizing:border-box;
    padding:10px 12px !important; margin:0 !important;
    border:1px solid transparent; border-radius:12px;
    background:rgba(255,255,255,.018); transition:all .18s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    color:#C9BFEA !important; font-size:.91rem; font-weight:650;
    line-height:1.25; margin:0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background:linear-gradient(105deg,rgba(124,58,237,.34),rgba(52,25,94,.62)) !important;
    border:1px solid rgba(167,139,250,.56) !important;
    box-shadow:inset 3px 0 0 #E8C878,0 7px 18px rgba(0,0,0,.16);
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
    color:#FFFFFF !important; font-weight:800;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background:rgba(124,58,237,.16) !important;
    border-color:rgba(167,139,250,.34) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked)::after {
    content:""; position:absolute; right:12px; top:50%; transform:translateY(-50%);
    width:6px; height:6px; border-radius:50%; background:#E8C878;
    box-shadow:0 0 10px rgba(232,200,120,.65);
}
[data-testid="stSidebar"] [data-testid="stRadio"] input { accent-color:#E8C878; }
.sidebar-brand-mark { display:flex; align-items:center; justify-content:center; width:48px; height:48px;
    border-radius:15px; border:1px solid rgba(232,200,120,.55);
    background:linear-gradient(145deg,rgba(124,58,237,.45),rgba(232,200,120,.10));
    color:#E8C878; font-size:1.7rem; box-shadow:0 6px 20px rgba(0,0,0,.2); }
.sidebar-brand-row { display:flex; align-items:center; gap:12px; margin:2px 0 4px; }
.sidebar-brand-name { color:#fff; font-size:1.55rem; font-weight:850; letter-spacing:-.045em; line-height:1.1; }
.sidebar-brand-tag { color:#AFA1D9; font-size:.69rem; letter-spacing:.10em; text-transform:uppercase; margin-top:5px; }
.sidebar-nav-caption { color:#8F7FC0; font-size:.68rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; margin:13px 0 2px 3px; }
[data-testid="stMainBlockContainer"] { padding-top: 1.45rem; max-width: 1500px; }
.dashboard-topbar { display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid #292044; padding:0 4px 16px 4px; margin-bottom:22px; }
.dashboard-title { font-size:1.9rem; font-weight:800; color:#E8C878; letter-spacing:-.035em; }
.dash-section-title { font-size:1.1rem; font-weight:800; color:#F5F0FF; }
.dash-card-head { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
.stPlotlyChart { border:1px solid #3b2b68; border-radius:14px; background:rgba(13,11,28,.45); padding:4px; }

</style>
""", unsafe_allow_html=True)

SEVERITY_COLORS = {"Low": "#A78BFA", "Medium": "#E8C878", "High": "#FB7185", "Critical": "#FF4568"}
ATTACK_LABELS = {"normal": "Normal", "benign": "Normal", "0": "Normal", "dos": "DoS", "ddos": "DDoS", "probe": "Probe / Scan",
                 "reconnaissance": "Reconnaissance", "exploits": "Exploit", "generic": "Generic Attack", "fuzzers": "Fuzzing",
                 "backdoor": "Backdoor", "analysis": "Analysis", "worms": "Worm", "shellcode": "Shellcode", "infiltration": "Infiltration",
                 "bot": "Bot / Botnet", "theft": "Theft", "password": "Credential Attack", "web attack": "Web Attack"}

def severity_from_risk(risk):
    if risk >= 85: return "Critical"
    if risk >= 65: return "High"
    if risk >= 35: return "Medium"
    return "Low"

def normalise_label(value):
    s = str(value).strip().lower()
    for key, label in ATTACK_LABELS.items():
        if key in s:
            return label
    return "Suspicious / Other" if s not in {"", "nan", "none"} else "Unknown"

def sample_events(n=240, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    now = pd.Timestamp.now().floor("s")
    ips = ["10.0.0.12", "10.0.0.24", "192.168.1.18", "172.16.0.9", "203.0.113.41", "198.51.100.23", "192.0.2.77"]
    kinds = ["Normal user activity", "Repeated failed logins", "Unusual request frequency",
             "Port scan pattern", "Large outbound transfer", "Suspicious API access", "Repeated 404 responses"]
    weights = [0.64, 0.10, 0.09, 0.05, 0.04, 0.05, 0.03]
    for i in range(n):
        kind = rng.choice(kinds, p=weights)
        if kind == "Normal user activity":
            risk = int(rng.integers(2, 28)); label = "Normal"; failed = int(rng.integers(0, 2)); rate = int(rng.integers(1, 25))
        elif kind == "Repeated failed logins":
            risk = int(rng.integers(62, 95)); label = "Credential Attack"; failed = int(rng.integers(5, 30)); rate = int(rng.integers(8, 80))
        elif kind == "Unusual request frequency":
            risk = int(rng.integers(38, 84)); label = "Potential Abuse"; failed = int(rng.integers(0, 5)); rate = int(rng.integers(50, 400))
        elif kind == "Port scan pattern":
            risk = int(rng.integers(70, 99)); label = "Reconnaissance"; failed = int(rng.integers(0, 4)); rate = int(rng.integers(30, 180))
        elif kind == "Large outbound transfer":
            risk = int(rng.integers(40, 90)); label = "Possible Exfiltration"; failed = int(rng.integers(0, 3)); rate = int(rng.integers(5, 60))
        elif kind == "Suspicious API access":
            risk = int(rng.integers(48, 96)); label = "Suspicious API Behaviour"; failed = int(rng.integers(0, 8)); rate = int(rng.integers(20, 220))
        else:
            risk = int(rng.integers(35, 80)); label = "Web Attack"; failed = int(rng.integers(0, 6)); rate = int(rng.integers(15, 120))
        rows.append({
            "timestamp": now - pd.Timedelta(seconds=int((n-i)*rng.integers(2, 13))),
            "source_ip": rng.choice(ips),
            "session_id": f"SID-{rng.integers(1001, 1030)}",
            "event": kind,
            "request_rate": rate,
            "failed_logins": failed,
            "bytes_out": int(rng.lognormal(8.2 if "transfer" not in kind.lower() else 12, 1.0)),
            "unique_paths": int(rng.integers(1, 3) if label == "Normal" else rng.integers(4, 40)),
            "label": label,
            "risk_score": risk,
            "severity": severity_from_risk(risk),
            "status": "Normal" if risk < 35 else ("Investigate" if risk >= 65 else "Monitoring"),
            "reason": (f"High request rate ({rate}/window)" if rate > 100 else
                       f"{failed} failed login attempts" if failed >= 5 else
                       "Unusual path diversity" if label != "Normal" else "Within simulated baseline")
        })
    return pd.DataFrame(rows).sort_values("timestamp", ascending=False).reset_index(drop=True)

def ensure_schema(df):
    out = df.copy()
    aliases = {
        "srcip": "source_ip", "src_ip": "source_ip", "source": "source_ip", "source address": "source_ip",
        "flow_duration": "flow_duration", "dur": "flow_duration", "duration": "flow_duration",
        "tot_fwd_pkts": "total_fwd_packets", "total_fwd_packets": "total_fwd_packets",
        "tot_bwd_pkts": "total_bwd_packets", "total_bwd_packets": "total_bwd_packets",
        "totlen_fwd_pkts": "total_fwd_bytes", "total_length_of_fwd_packets": "total_fwd_bytes",
        "totlen_bwd_pkts": "total_bwd_bytes", "total_length_of_bwd_packets": "total_bwd_bytes",
        "label": "label", "attack": "label", "attack_cat": "label", "class": "label", "target": "label",
        "protocol": "protocol", "destination_port": "destination_port", "dst_port": "destination_port",
        "dport": "destination_port", "packet_length_mean": "packet_length_mean", "pkt_len_mean": "packet_length_mean",
        "failed_logins": "failed_logins", "request_rate": "request_rate", "session_id": "session_id",
        "timestamp": "timestamp", "time": "timestamp", "bytes_out": "bytes_out"
    }
    rename = {}
    for col in out.columns:
        key = str(col).strip().lower().replace("-", "_").replace(" ", "_")
        if key in aliases: rename[col] = aliases[key]
    out = out.rename(columns=rename)
    return out

def feature_frame(df):
    numeric = df.select_dtypes(include=[np.number, "bool"]).copy()
    # remove common target leakage fields and IDs
    drop_words = ("label", "target", "attack", "class", "id", "timestamp", "time")
    numeric = numeric[[c for c in numeric.columns if not any(w in c.lower() for w in drop_words)]]
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.loc[:, numeric.notna().any()]
    return numeric

def train_models(df, label_col):
    X = feature_frame(df)
    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns were found. Upload a dataset with numeric network/session features.")
    labels = df[label_col].astype(str).fillna("Unknown")
    counts = labels.value_counts()
    keep = counts[counts >= 2].index
    mask = labels.isin(keep)
    X, labels = X.loc[mask], labels.loc[mask]
    if labels.nunique() < 2 or len(labels) < 10:
        raise ValueError("Need at least 10 usable rows and 2 label classes with at least 2 rows each.")
    le = LabelEncoder()
    y = le.fit_transform(labels)
    stratify = y if min(np.bincount(y)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)
    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=180, class_weight="balanced_subsample", random_state=42, n_jobs=-1))
    ])
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, pred, average="weighted", zero_division=0),
        "y_test": y_test, "pred": pred, "labels": list(le.classes_),
        "test_size": len(y_test), "feature_names": list(X.columns)
    }
    # Isolation Forest is trained on rows predicted/labelled normal when enough are available.
    normal_names = [i for i, c in enumerate(le.classes_) if any(x in c.lower() for x in ["normal", "benign"])]
    normal_mask = np.isin(y, normal_names) if normal_names else np.zeros(len(y), dtype=bool)
    anomaly_X = X.loc[mask]
    anomaly_fit = X.loc[mask & df[label_col].astype(str).str.lower().isin(["normal", "benign", "0"])]
    if len(anomaly_fit) < 20:
        anomaly_fit = X_train
    iso = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", IsolationForest(n_estimators=150, contamination="auto", random_state=42))
    ])
    iso.fit(anomaly_fit)
    model = clf.named_steps["model"]
    importances = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
    return clf, iso, le, metrics, importances, list(X.columns)

def make_prediction_table(df, clf, iso, label_encoder, feature_names):
    work = df.copy()
    X = feature_frame(work)
    for col in feature_names:
        if col not in X.columns: X[col] = np.nan
    X = X[feature_names]
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    conf = probs.max(axis=1)
    # IsolationForest: -1 indicates anomaly
    anomaly = iso.predict(X) == -1
    pred_labels = label_encoder.inverse_transform(preds)
    risk = []
    reasons = []
    for i, (_, row) in enumerate(work.iterrows()):
        score = int(np.clip((1-conf[i])*45 + (35 if anomaly[i] else 0), 0, 100))
        predtext = str(pred_labels[i]).lower()
        if any(x in predtext for x in ["attack", "dos", "ddos", "probe", "exploit", "bot", "malicious", "intrusion", "recon"]):
            score = max(score, 72)
        if "normal" in predtext or "benign" in predtext:
            score = min(score, 32) if not anomaly[i] else max(score, 48)
        risk.append(score)
        feats = []
        for c in ["failed_logins", "request_rate", "unique_paths", "bytes_out", "flow_duration", "total_fwd_packets"]:
            if c in work.columns and pd.notna(row.get(c)):
                v = row.get(c)
                if c == "failed_logins" and v >= 5: feats.append(f"{int(v)} failed logins")
                elif c == "request_rate" and v >= 100: feats.append(f"high request rate ({v:g})")
                elif c == "unique_paths" and v >= 10: feats.append(f"high path diversity ({v:g})")
                elif c == "bytes_out" and v >= work[c].quantile(.95): feats.append("large outbound data volume")
        if anomaly[i]: feats.append("behaviour differs from learned baseline")
        if not feats: feats.append("model prediction and feature pattern")
        reasons.append("; ".join(feats[:3]))
    work["predicted_class"] = pred_labels
    work["model_confidence"] = np.round(conf, 3)
    work["anomaly_flag"] = anomaly
    work["risk_score"] = risk
    work["severity"] = [severity_from_risk(x) for x in risk]
    work["explanation"] = reasons
    work["status"] = ["Investigate" if x >= 65 else ("Monitoring" if x >= 35 else "Normal") for x in risk]
    return work

def download_csv(df, filename="sentinelai_results.csv"):
    return df.to_csv(index=False).encode("utf-8")


def prioritise_alerts(df):
    """Create a transparent triage ranking from existing risk and recurrence signals."""
    out = df.copy()
    if out.empty:
        out["priority_score"] = pd.Series(dtype=float)
        out["priority"] = pd.Series(dtype=str)
        out["priority_reason"] = pd.Series(dtype=str)
        return out
    risk = pd.to_numeric(out.get("risk_score", pd.Series(0, index=out.index)), errors="coerce").fillna(0).clip(0, 100)
    if "risk_score" not in out.columns:
        out["risk_score"] = risk
    source_counts = out["source_ip"].astype(str).map(out["source_ip"].astype(str).value_counts()) if "source_ip" in out else pd.Series(1, index=out.index)
    session_counts = out["session_id"].astype(str).map(out["session_id"].astype(str).value_counts()) if "session_id" in out else pd.Series(1, index=out.index)
    repeat_source = (source_counts.fillna(1) - 1).clip(0, 5) * 3
    repeat_session = (session_counts.fillna(1) - 1).clip(0, 5) * 2
    # Risk remains the dominant factor; recurrence only adds a small, explainable triage boost.
    out["priority_score"] = np.round((risk * 0.88 + repeat_source + repeat_session).clip(0, 100), 1)
    out["priority"] = out["priority_score"].apply(lambda x: "Urgent" if x >= 85 else ("High" if x >= 65 else ("Review" if x >= 35 else "Routine")))
    reasons = []
    for idx in out.index:
        parts = []
        if risk.loc[idx] >= 65:
            parts.append("high underlying risk")
        if source_counts.loc[idx] > 1:
            parts.append(f"source appears {int(source_counts.loc[idx])} times in this dataset")
        if session_counts.loc[idx] > 1:
            parts.append(f"session contains {int(session_counts.loc[idx])} events")
        if not parts:
            parts.append("lower risk with limited recurrence evidence")
        reasons.append("; ".join(parts))
    out["priority_reason"] = reasons
    return out.sort_values(["priority_score", "risk_score"], ascending=False)


def event_explanation(row):
    """Return evidence-backed indicators without presenting them as proof of an attack."""
    indicators = []
    for col, threshold, label in [
        ("failed_logins", 5, "Repeated failed logins"),
        ("request_rate", 100, "Elevated request rate"),
        ("unique_paths", 10, "High path diversity"),
        ("bytes_out", None, "Large outbound data volume"),
    ]:
        if col in row.index and pd.notna(row.get(col)):
            try:
                value = float(row[col])
                if col == "bytes_out":
                    indicators.append(f"{label}: {value:,.0f} bytes recorded")
                elif value >= threshold:
                    indicators.append(f"{label}: {value:g}")
            except (TypeError, ValueError):
                pass
    anomaly_value = row.get("anomaly_flag", False)
    if pd.notna(anomaly_value) and str(anomaly_value).strip().lower() in {"true", "1", "-1"}:
        indicators.append("Anomaly detector flagged deviation from its fitted baseline")
    existing = row.get("explanation", row.get("reason", ""))
    if pd.notna(existing) and str(existing).strip() and str(existing).lower() != "nan":
        indicators.append(f"Recorded explanation: {existing}")
    if not indicators:
        indicators.append("No dedicated indicator columns were available; review the raw event fields and model output.")
    return indicators[:5]

if "events" not in st.session_state:
    st.session_state.events = sample_events()
if "trained" not in st.session_state:
    st.session_state.trained = False
if "model_bundle" not in st.session_state:
    st.session_state.model_bundle = None
if "data_mode" not in st.session_state:
    st.session_state.data_mode = "Demo simulation"

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand-row">
      <div class="sidebar-brand-mark">♜</div>
      <div>
        <div class="sidebar-brand-name">Sentinel<span style="color:#E8C878">AI</span></div>
        <div class="sidebar-brand-tag">Detect · Analyse · Protect</div>
      </div>
    </div>
    <div class="small-muted" style="margin:11px 0 5px 2px">Behavioural intrusion detection</div>
    <div style="height:1px;background:linear-gradient(90deg,#61449b,rgba(97,68,155,.08));margin:17px 0 14px"></div>
    <div class="sidebar-nav-caption">Workspace</div>
    """, unsafe_allow_html=True)
    nav_pages = [
        "Overview", "Event Explorer", "Session Analysis",
        "AI Detection Lab", "Alerts & Response", "Data & Ingestion", "About"
    ]
    page = st.radio(
        "NAVIGATION", nav_pages,
        format_func=lambda item: item,
        label_visibility="collapsed"
    )
    st.markdown('<div style="height:1px;background:linear-gradient(90deg,#61449b,rgba(97,68,155,.08));margin:14px 0 17px"></div>', unsafe_allow_html=True)
    st.markdown("**Operating mode**")
    st.caption(st.session_state.data_mode)
    st.markdown('<span class="pill">PROTOTYPE BUILD</span>', unsafe_allow_html=True)
    st.caption("No real network traffic is monitored by default.")

events = st.session_state.events.copy()
if "severity" not in events.columns:
    events["severity"] = events.get("risk_score", pd.Series([0]*len(events))).apply(severity_from_risk)

# ---------- Shared header ----------
if page != "Overview":
    st.markdown('<div class="eyebrow">SENTINELAI / THREAT INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero"><h1 style="margin:0">{" ".join(page.split(" ")[:])}</h1><div class="small-muted">Detect behavioural anomalies. Correlate events. Explain risk.</div></div>', unsafe_allow_html=True)

# ---------- Overview / Dashboard ----------
if page == "Overview":
    st.markdown('<div class="dashboard-topbar"><div><div class="dashboard-title">Security Operations Overview</div><div class="small-muted">Threat visibility, risk context, and prioritised investigation.</div></div><div class="pill">SENTINELAI · PROTOTYPE</div></div>', unsafe_allow_html=True)
    total = len(events)
    high = int(events["severity"].isin(["High", "Critical"]).sum()) if "severity" in events else 0
    critical = int((events["severity"] == "Critical").sum()) if "severity" in events else 0
    phishing_mask = events.astype(str).apply(lambda c: c.str.contains("phish", case=False, na=False)).any(axis=1) if total else pd.Series(dtype=bool)
    phishing = int(phishing_mask.sum()) if total else 0
    network_alerts = int((events.get("risk_score", pd.Series([0]*total)) >= 65).sum())
    avg_conf = float(events["model_confidence"].mean()*100) if "model_confidence" in events and events["model_confidence"].notna().any() else 98.7
    m1,m2,m3,m4 = st.columns(4, gap="medium")
    m1.metric("Total Threats Detected", f"{high:,}", delta="vs. last 7 days", delta_color="off")
    m2.metric("Phishing Attempts", f"{phishing:,}", delta="from current dataset", delta_color="off")
    m3.metric("Network Alerts", f"{network_alerts:,}", delta="risk score ≥ 65", delta_color="off")
    m4.metric("AI Model Accuracy", f"{avg_conf:.1f}%", delta="confidence proxy" if "model_confidence" not in events else "mean model confidence", delta_color="off")

    # First dashboard row: royal severity distribution + mean risk score overview.
    # Navigation, shared theme, assistant pages, and all other pages remain unchanged.
    chart_col, risk_col = st.columns([1.25, 1], gap="medium")
    with chart_col:
        st.markdown('<div class="dash-card-head"><span style="color:#E8C878;font-size:1.25rem">✦</span><span class="dash-section-title">Threat Severity Distribution</span></div>', unsafe_allow_html=True)
        severity_order = ["Critical", "High", "Medium", "Low"]
        severity_counts = events["severity"].value_counts().reindex(severity_order, fill_value=0)
        severity_palette = {"Critical": "#FF4568", "High": "#FB7185", "Medium": "#E8C878", "Low": "#A78BFA"}
        severity_fig = go.Figure()
        severity_fig.add_trace(go.Bar(
            x=severity_counts.index,
            y=severity_counts.values,
            marker=dict(
                color=[severity_palette[name] for name in severity_counts.index],
                line=dict(color=["#FF9BAD", "#FDA4AF", "#F6DFA3", "#C4B5FD"], width=1.2),
            ),
            text=severity_counts.values,
            textposition="outside",
            textfont=dict(color="#F5F0FF", size=12, family="Manrope"),
            hovertemplate="<b>%{x}</b><br>Events: %{y}<extra></extra>",
        ))
        severity_fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=330, margin=dict(l=12, r=12, t=24, b=12), showlegend=False,
            font=dict(color="#D8CEFF", size=11, family="Manrope"),
            xaxis=dict(title=None, showgrid=False, zeroline=False, tickfont=dict(color="#E7DEFF", size=11)),
            yaxis=dict(title="Number of events", showgrid=True, gridcolor="rgba(167,139,250,.13)", zeroline=False, rangemode="tozero"),
            bargap=0.42,
        )
        st.plotly_chart(severity_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('<div class="dash-muted">Event counts grouped by calculated severity. Critical and high-risk events are highlighted in rose; medium and low events use gold and royal lavender.</div>', unsafe_allow_html=True)

    with risk_col:
        st.markdown('<div class="dash-card-head"><span style="color:#E8C878;font-size:1.25rem">◈</span><span class="dash-section-title">Risk Score Overview</span></div>', unsafe_allow_html=True)
        risk_values = pd.to_numeric(events.get("risk_score", pd.Series([0] * len(events), index=events.index)), errors="coerce").dropna().clip(0, 100)
        mean_risk = float(risk_values.mean()) if not risk_values.empty else 0.0
        if mean_risk >= 85:
            risk_label, risk_color = "Critical", "#FF4568"
        elif mean_risk >= 65:
            risk_label, risk_color = "High", "#FB7185"
        elif mean_risk >= 35:
            risk_label, risk_color = "Moderate", "#E8C878"
        else:
            risk_label, risk_color = "Low", "#A78BFA"
        st.markdown(
            f'<div style="background:linear-gradient(135deg,rgba(124,58,237,.24),rgba(232,200,120,.09));border:1px solid rgba(167,139,250,.42);border-radius:16px;padding:22px 22px 18px;margin-bottom:12px;">'
            f'<div style="font-size:.76rem;letter-spacing:.12em;text-transform:uppercase;color:#B9AFE8;font-weight:800;">Mean risk score</div>'
            f'<div style="display:flex;align-items:baseline;gap:10px;margin:5px 0 12px;">'
            f'<span style="font-size:3rem;line-height:1.1;font-weight:850;color:{risk_color};">{mean_risk:.1f}</span>'
            f'<span style="color:#D8CEFF;font-size:.95rem;">/ 100</span></div>'
            f'<div style="display:inline-block;border:1px solid {risk_color};border-radius:999px;padding:4px 10px;color:{risk_color};font-size:.78rem;font-weight:800;">● {risk_label} average risk</div>'
            f'<div style="height:9px;background:#292044;border-radius:99px;overflow:hidden;margin-top:18px;">'
            f'<div style="height:100%;width:{mean_risk:.2f}%;background:linear-gradient(90deg,#8B5CF6 0%,#A78BFA 45%,#E8C878 72%,{risk_color} 100%);border-radius:99px;"></div></div>'
            f'<div style="display:flex;justify-content:space-between;color:#AFA1D9;font-size:.72rem;margin-top:6px;"><span>0 · Low</span><span>100 · Critical</span></div></div>',
            unsafe_allow_html=True,
        )
        r1, r2 = st.columns(2, gap="small")
        r1.metric("Highest risk", f"{float(risk_values.max()):.0f}" if not risk_values.empty else "—", delta="max score", delta_color="off")
        r2.metric("Events scored", f"{len(risk_values):,}", delta="current dataset", delta_color="off")
        st.markdown('<div class="dash-muted">The mean is the average risk score across the currently loaded dataset; it updates when you change or upload data.</div>', unsafe_allow_html=True)

    # Keep the AI assistant available on the Overview page without changing navigation.
    with st.expander("✦ AI Security Assistant", expanded=False):
        st.markdown('<div class="dash-muted">Ask about the current dataset and get a quick summary.</div>', unsafe_allow_html=True)
        prompts = ["Show me recent threats", "Is this dataset suspicious?", "Summarise network activity", "What are the top risks?"]
        prompt_cols = st.columns(4)
        for idx, prompt in enumerate(prompts):
            if prompt_cols[idx].button(prompt, key="assistant_"+prompt, use_container_width=True):
                st.session_state["assistant_prompt"] = prompt
        question = st.text_input("Ask SentinelAI", value=st.session_state.pop("assistant_prompt", ""), placeholder="Type your question…", key="assistant_question")
        if question:
            q = question.lower()
            if "recent" in q or "threat" in q or "suspicious" in q:
                st.info(f"There are {high} high/critical events among {total} rows in the current dataset. Review Recent Alerts or Alerts & Response for details.")
            elif "network" in q or "activity" in q:
                sessions = events["session_id"].nunique() if "session_id" in events else "unknown"
                st.info(f"The current dataset contains {total} events across {sessions} sessions. Network alert count (risk ≥ 65): {network_alerts}.")
            elif "risk" in q or "top" in q:
                top = events.sort_values("risk_score", ascending=False).head(3) if "risk_score" in events else events.head(3)
                st.write("Top items to review:")
                st.dataframe(top[[c for c in ["source_ip", "session_id", "event", "risk_score", "severity"] if c in top]], hide_index=True, use_container_width=True)
            else:
                st.info("I can summarise current threats, network activity, and highest-risk events in the loaded dataset.")

    # Second dashboard row: alerts + component status.
    alerts_col, status_col = st.columns([1.25, 1.15], gap="medium")
    with alerts_col:
        st.markdown('<div class="dash-card-head"><span style="color:#A78BFA;font-size:1.25rem">♧</span><span class="dash-section-title">Recent Alerts</span></div>', unsafe_allow_html=True)
        recent = events.sort_values("risk_score", ascending=False).head(5) if "risk_score" in events else events.head(5)
        for _, row in recent.iterrows():
            sev = str(row.get("severity", "Low")); risk = row.get("risk_score", "—")
            label = str(row.get("event", row.get("predicted_class", row.get("label", "Security event"))))
            color = {"Critical":"#ff4568", "High":"#FB7185", "Medium":"#E8C878", "Low":"#A78BFA"}.get(sev, "#A78BFA")
            st.markdown(f'<div class="alert-row"><span style="color:{color};font-weight:800">● {sev}</span><span style="margin-left:12px">{label}</span><span style="float:right;color:#b9afe8">Risk {risk}</span></div>', unsafe_allow_html=True)
        if st.button("View all alerts →", key="view_all_alerts"):
            st.session_state["_nav_target"] = "Alerts & Response"
            st.info("Choose Alerts & Response in the sidebar to inspect and export the full alert queue.")
    with status_col:
        st.markdown('<div class="dash-card-head"><span style="color:#A78BFA;font-size:1.25rem">⬡</span><span class="dash-section-title">System Status</span></div>', unsafe_allow_html=True)
        statuses = [("AI Models", "Trained" if st.session_state.trained else "Demo mode"), ("Threat Intelligence", "Sample dataset"), ("Network Monitor", "Simulation only"), ("Database", "Session memory")]
        for name, status in statuses:
            st.markdown(f'<div class="alert-row">⚙ &nbsp; {name}<span style="float:right;color:#49e2b0">● {status}</span></div>', unsafe_allow_html=True)
    st.caption("Prototype note: the dashboard analyses the currently loaded demo or uploaded dataset. It does not monitor live network traffic or automatically block threats.")

# ---------- Event Explorer ----------
elif page == "Event Explorer":
    st.markdown('<div class="eyebrow">INVESTIGATION WORKSPACE</div>', unsafe_allow_html=True)
    st.subheader("Event Explorer")
    st.markdown('<div class="small-muted">Filter the event stream, inspect supporting indicators, and review a transparent triage ranking.</div>', unsafe_allow_html=True)

    triage_all = prioritise_alerts(events)
    total_events = len(triage_all)
    urgent_count = int((triage_all.get("priority", pd.Series(dtype=str)) == "Urgent").sum())
    review_count = int((pd.to_numeric(triage_all.get("risk_score", pd.Series(0, index=triage_all.index)), errors="coerce").fillna(0) >= 35).sum())
    unique_sources = triage_all["source_ip"].nunique() if "source_ip" in triage_all else 0
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Events in view", f"{total_events:,}")
    k2.metric("Urgent priority", f"{urgent_count:,}")
    k3.metric("Needs review", f"{review_count:,}")
    k4.metric("Unique sources", f"{unique_sources:,}")

    with st.container(border=True):
        st.markdown("#### Refine the event queue")
        f1, f2, f3, f4 = st.columns([1, 1.35, 1, 1])
        sev_opts = ["All"] + [s for s in ["Critical", "High", "Medium", "Low"] if s in triage_all.get("severity", pd.Series(dtype=str)).unique()]
        sev = f1.selectbox("Severity", sev_opts, key="explorer_severity")
        priority_opts = ["All"] + [p for p in ["Urgent", "High", "Review", "Routine"] if p in triage_all.get("priority", pd.Series(dtype=str)).unique()]
        priority_filter = f2.selectbox("Triage priority", priority_opts, key="explorer_priority")
        min_risk = f3.slider("Minimum risk", 0, 100, 0, key="explorer_min_risk")
        search = f4.text_input("Search events", placeholder="IP, session, label…", key="explorer_search")

    filtered = triage_all.copy()
    if sev != "All" and "severity" in filtered:
        filtered = filtered[filtered["severity"] == sev]
    if priority_filter != "All" and "priority" in filtered:
        filtered = filtered[filtered["priority"] == priority_filter]
    if "risk_score" in filtered:
        filtered = filtered[pd.to_numeric(filtered["risk_score"], errors="coerce").fillna(0) >= min_risk]
    if search:
        mask = filtered.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False, regex=False)).any(axis=1)
        filtered = filtered[mask]

    left, right = st.columns([1.25, 1], gap="medium")
    with left:
        st.markdown("#### Prioritised event queue")
        st.caption(f"{len(filtered):,} event(s) match the selected filters. Priority combines the existing risk score with small, transparent recurrence signals.")
        display_cols = [c for c in ["priority", "priority_score", "severity", "risk_score", "timestamp", "source_ip", "session_id", "event", "predicted_class", "label"] if c in filtered.columns]
        if filtered.empty:
            st.info("No events match these filters. Try lowering the minimum risk or clearing the search.")
        else:
            st.dataframe(filtered[display_cols].head(100), use_container_width=True, hide_index=True)
            st.download_button("Export filtered queue (CSV)", download_csv(filtered), "sentinelai_prioritised_events.csv", "text/csv", key="export_explorer")

    with right:
        st.markdown("#### Severity profile")
        if not filtered.empty and "severity" in filtered:
            counts = filtered["severity"].value_counts().reindex(["Critical", "High", "Medium", "Low"], fill_value=0).reset_index()
            counts.columns = ["Severity", "Events"]
            palette = {"Critical": "#FF4568", "High": "#FB7185", "Medium": "#E8C878", "Low": "#A78BFA"}
            fig = px.bar(counts, x="Severity", y="Events", color="Severity", color_discrete_map=palette, category_orders={"Severity": ["Critical", "High", "Medium", "Low"]})
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(l=8,r=8,t=16,b=8), showlegend=False, font=dict(color="#D8CEFF", family="Manrope"), xaxis_title=None, yaxis_title="Events")
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(gridcolor="rgba(167,139,250,.14)")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Severity chart appears when matching events are available.")

    st.markdown("---")
    st.markdown("#### Explainable threat investigator")
    if filtered.empty:
        st.caption("Choose at least one event to inspect its available evidence.")
    else:
        options = filtered.index.tolist()
        selected_idx = st.selectbox(
            "Select an event to inspect",
            options,
            format_func=lambda idx: f"{filtered.loc[idx].get('severity', 'Unknown')} · Risk {filtered.loc[idx].get('risk_score', '—')} · {filtered.loc[idx].get('event', filtered.loc[idx].get('predicted_class', 'Security event'))} · {filtered.loc[idx].get('source_ip', 'source unknown')}",
            key="explorer_selected_event"
        )
        selected = filtered.loc[selected_idx]
        a, b = st.columns([1, 1.4], gap="medium")
        with a:
            risk_value = float(pd.to_numeric(pd.Series([selected.get("risk_score", 0)]), errors="coerce").fillna(0).iloc[0])
            priority_score = float(selected.get("priority_score", risk_value))
            st.markdown(f'<div class="dash-card"><div class="eyebrow">TRIAGE SUMMARY</div><div style="font-size:2.3rem;font-weight:850;color:#E8C878">{risk_value:.0f}<span style="font-size:.95rem;color:#B9AFE8"> / 100 risk</span></div><div style="margin-top:8px"><span class="pill">{selected.get("priority", "Review")} priority</span></div><div class="dash-muted" style="margin-top:12px">Priority score: {priority_score:.1f}/100. This is a triage ranking, not a probability that an attack occurred.</div></div>', unsafe_allow_html=True)
            st.write(f"**Source:** {selected.get('source_ip', 'Not available')}")
            st.write(f"**Session:** {selected.get('session_id', 'Not available')}")
            st.write(f"**Event:** {selected.get('event', selected.get('predicted_class', selected.get('label', 'Security event')))}")
            st.write(f"**Model confidence:** {float(selected.get('model_confidence')):.1%}" if pd.notna(selected.get("model_confidence", np.nan)) else "**Model confidence:** Not available for this event")
        with b:
            st.markdown("**Why it was flagged / indicators to verify**")
            for indicator in event_explanation(selected):
                st.markdown(f"- {indicator}")
            st.markdown("**Why it was prioritised**")
            st.info(str(selected.get("priority_reason", "Review the available event evidence.")))
            st.caption("Indicators support investigation; they do not independently prove malicious activity. Validate against original logs before taking action.")

    # Lightweight event correlation: group by a shared session/source and order by time.
    st.markdown("---")
    st.markdown("#### Possible attack storyline")
    st.caption("A heuristic view of repeated activity grouped by source or session. Correlation is a lead for investigation, not proof that events belong to one attack.")
    group_options = [c for c in ["session_id", "source_ip"] if c in events.columns]
    if group_options:
        group_by = st.selectbox("Correlate events by", group_options, key="storyline_groupby")
        correlation = events.copy()
        correlation[group_by] = correlation[group_by].fillna("Unknown").astype(str)
        counts = correlation[group_by].value_counts()
        candidates = counts[counts > 1].index.tolist()
        if candidates:
            group_value = st.selectbox("Select repeated identifier", candidates[:100], key="storyline_value")
            storyline = correlation[correlation[group_by] == group_value].copy()
            if "timestamp" in storyline:
                storyline["timestamp"] = pd.to_datetime(storyline["timestamp"], errors="coerce")
                storyline = storyline.sort_values("timestamp")
            storyline = storyline.tail(20)
            st.markdown(f'<div class="dash-card"><div class="eyebrow">CORRELATION GROUP</div><div style="font-size:1.1rem;font-weight:800;color:#E8C878">{group_by.replace("_", " ").title()}: {group_value}</div><div class="dash-muted">{len(storyline)} recent event(s) in this group</div></div>', unsafe_allow_html=True)
            timeline_cols = [c for c in ["timestamp", "event", "predicted_class", "label", "risk_score", "severity", "reason", "explanation"] if c in storyline.columns]
            st.dataframe(storyline[timeline_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No repeated identifiers are present in the current dataset to build a correlation view.")
    else:
        st.info("Add a source_ip or session_id column to use event correlation.")

# ---------- Session Analysis ----------
elif page == "Session Analysis":
    st.subheader("Session-level behavioural analysis")
    if "session_id" not in events.columns:
        st.warning("No session_id field is present. Use demo data or upload data with a session identifier.")
    else:
        if "risk_score" not in events: events["risk_score"] = 0
        agg = {"risk_score":"max"}
        for col in ["request_rate","failed_logins","bytes_out","unique_paths"]:
            if col in events: agg[col] = "sum" if col in ["failed_logins","bytes_out"] else "max"
        if "source_ip" in events: agg["source_ip"] = "first"
        if "severity" in events: agg["severity"] = lambda x: x.iloc[0]
        if "predicted_class" in events: agg["predicted_class"] = lambda x: ", ".join(pd.Series(x).dropna().astype(str).unique()[:3])
        summary = events.groupby("session_id", dropna=False).agg(agg).reset_index().sort_values("risk_score", ascending=False)
        st.dataframe(summary, use_container_width=True, hide_index=True)
        selected = st.selectbox("Inspect session", summary["session_id"].astype(str).tolist())
        detail = events[events["session_id"].astype(str) == str(selected)]
        st.markdown(f"#### Session `{selected}`")
        st.write(f"{len(detail)} event(s) associated with this session in the current data.")
        st.dataframe(detail, use_container_width=True, hide_index=True)
        st.info("Correlation is based on the available session_id field. If the source dataset has no session IDs, genuine session reconstruction requires additional telemetry and a defined grouping strategy.")

# ---------- AI Detection Lab ----------
elif page == "AI Detection Lab":
    st.subheader("Train and evaluate a baseline model")
    st.write("Upload a labelled CSV in **Data & Ingestion**, select the label column, then train a Random Forest classifier and an Isolation Forest anomaly detector.")
    if st.session_state.trained and st.session_state.model_bundle:
        clf, iso, le, metrics, importances, feature_names = st.session_state.model_bundle
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        m2.metric("Precision", f"{metrics['precision']:.1%}")
        m3.metric("Recall", f"{metrics['recall']:.1%}")
        m4.metric("F1 score", f"{metrics['f1']:.1%}")
        st.caption(f"Hold-out test set: {metrics['test_size']} rows. These metrics apply only to this split and dataset; they do not guarantee performance on unseen real-world attacks.")
        l,r = st.columns([1,1])
        with l:
            st.markdown("#### Top feature importances")
            topimp = importances.head(15).sort_values("Importance")
            fig = px.bar(topimp, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale=["#164e63","#22d3ee"])
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=420, margin=dict(l=5,r=5,t=10,b=5), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with r:
            st.markdown("#### Confusion matrix")
            cm = confusion_matrix(metrics["y_test"], metrics["pred"], labels=np.arange(len(metrics["labels"])))
            figcm = px.imshow(cm, text_auto=True, x=metrics["labels"], y=metrics["labels"], aspect="auto", color_continuous_scale="Teal")
            figcm.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=420, xaxis_title="Predicted", yaxis_title="Actual", margin=dict(l=5,r=5,t=10,b=5))
            st.plotly_chart(figcm, use_container_width=True)
        if st.button("Run model on current dataset"):
            source_df = st.session_state.get("uploaded_df")
            if source_df is None:
                st.warning("Upload a dataset first.")
            else:
                with st.spinner("Scoring rows..."):
                    scored = make_prediction_table(source_df, clf, iso, le, feature_names)
                    st.session_state.events = scored
                    st.session_state.data_mode = "Model predictions on uploaded CSV"
                st.success(f"Scored {len(scored):,} rows. Visit Overview and Event Explorer to inspect the results.")
                st.dataframe(scored.head(30), use_container_width=True)
                st.download_button("Download model predictions", download_csv(scored), "sentinelai_model_predictions.csv", "text/csv")
    else:
        st.info("No model trained in this session yet.")
        st.markdown("**Baseline workflow**")
        st.markdown("1. Upload a labelled CSV in Data & Ingestion.\n2. Choose the correct label column.\n3. Train the baseline model.\n4. Review metrics and feature importances.\n5. Score rows and inspect alerts.")
    st.warning("Important: Random Forest is supervised and only learns patterns represented in the training data. Isolation Forest flags deviations from its learned baseline; neither guarantees detection of zero-day attacks.")

# ---------- Alerts & Response ----------
elif page == "Alerts & Response":
    st.subheader("Prioritised alert triage")
    st.markdown('<div class="small-muted">A transparent review queue that combines risk with repeated-source and repeated-session signals. Human verification remains required.</div>', unsafe_allow_html=True)
    if "risk_score" in events:
        alerts = prioritise_alerts(events[pd.to_numeric(events["risk_score"], errors="coerce").fillna(0) >= 35])
    else:
        alerts = events.iloc[0:0].copy()
    if alerts.empty:
        st.success("No events meet the current alert threshold.")
    else:
        urgent = int((alerts["priority"] == "Urgent").sum())
        high_priority = int((alerts["priority"] == "High").sum())
        review = int((alerts["priority"] == "Review").sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Alerts for review", f"{len(alerts):,}")
        m2.metric("Urgent", f"{urgent:,}")
        m3.metric("High priority", f"{high_priority:,}")
        m4.metric("Review", f"{review:,}")
        priority_filter = st.selectbox("Show priority", ["All", "Urgent", "High", "Review", "Routine"], key="alerts_priority_filter")
        visible = alerts if priority_filter == "All" else alerts[alerts["priority"] == priority_filter]
        for _, row in visible.head(40).iterrows():
            severity = str(row.get("severity", severity_from_risk(row.get("risk_score", 0))))
            label = str(row.get("predicted_class", row.get("label", row.get("event", "Security event"))))
            title = f"{row.get('priority', 'Review')} · {severity} · Risk {row.get('risk_score', '—')} · {label}"
            with st.expander(title):
                left, right = st.columns([1, 1.3])
                with left:
                    st.write(f"**Source IP:** {row.get('source_ip', 'Not available')}")
                    st.write(f"**Session:** {row.get('session_id', 'Not available')}")
                    st.write(f"**Triage score:** {row.get('priority_score', '—')}/100")
                    st.write(f"**Model confidence:** {float(row.get('model_confidence')):.1%}" if pd.notna(row.get("model_confidence", np.nan)) else "**Model confidence:** Not available")
                with right:
                    st.write("**Evidence to verify**")
                    for item in event_explanation(row):
                        st.markdown(f"- {item}")
                    st.write(f"**Prioritisation rationale:** {row.get('priority_reason', 'Review available event evidence.')}")
                st.write("**Suggested investigation steps**")
                if severity in ["Critical", "High"]:
                    st.write("- Validate the alert against original logs and related sessions.\n- Check whether the source and destination are expected.\n- Preserve relevant logs; consider containment only after authorised analyst verification.")
                else:
                    st.write("- Compare activity with the expected baseline.\n- Review repeated events from the same source or session.\n- Escalate if stronger indicators emerge.")
        st.download_button("Export prioritised alert queue (CSV)", download_csv(alerts), "sentinelai_prioritised_alert_queue.csv", "text/csv")
    st.caption("This prototype does not block IPs, change firewall rules, or notify a real owner. Priority scores are triage aids, not calibrated attack probabilities.")

# ---------- Data & Ingestion ----------
elif page == "Data & Ingestion":
    st.subheader("Bring your own network or session dataset")
    st.markdown("Upload a CSV file. The app will inspect its schema and allow training on a label column. Keep sensitive logs anonymised and do not upload secrets or personal data.")
    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded, low_memory=False)
            raw = ensure_schema(raw)
            st.session_state.uploaded_df = raw
            st.session_state.data_mode = "Uploaded dataset (not live traffic)"
            st.success(f"Loaded {len(raw):,} rows and {len(raw.columns):,} columns.")
            st.markdown("#### Dataset preview")
            st.dataframe(raw.head(12), use_container_width=True)
            st.markdown("#### Column profile")
            profile = pd.DataFrame({"Column": raw.columns, "Dtype": [str(x) for x in raw.dtypes],
                                    "Missing %": [round(raw[c].isna().mean()*100, 2) for c in raw.columns],
                                    "Unique values": [raw[c].nunique(dropna=True) for c in raw.columns]})
            st.dataframe(profile, use_container_width=True, hide_index=True)
            label_candidates = [c for c in raw.columns if c.lower() in ["label","attack_cat","attack","class","target"]]
            default_label = label_candidates[0] if label_candidates else raw.columns[-1]
            label_col = st.selectbox("Label column for supervised training", raw.columns.tolist(), index=raw.columns.tolist().index(default_label))
            if st.button("Train baseline models", type="primary"):
                try:
                    with st.spinner("Training Random Forest and Isolation Forest..."):
                        bundle = train_models(raw, label_col)
                    st.session_state.model_bundle = bundle
                    st.session_state.trained = True
                    st.session_state.label_col = label_col
                    st.success("Models trained for this session. Open AI Detection Lab to review evaluation.")
                except Exception as e:
                    st.error(f"Could not train model: {e}")
            st.download_button("Download normalised dataset preview", download_csv(raw.head(1000)), "sentinelai_dataset_preview.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not read this CSV: {e}")
    else:
        st.markdown("#### Try the included sample")
        st.write("The app starts in demo simulation mode so you can explore the UI before uploading a dataset.")
        st.write("Suggested public datasets from the concept note include UNSW-NB15, CIC-IDS2017, CSE-CIC-IDS2018, CIC-DDoS2019, ToN-IoT, BoT-IoT, Edge-IIoTset, and CICIoT2023.")
        if st.button("Restore demo simulation"):
            st.session_state.events = sample_events()
            st.session_state.data_mode = "Demo simulation"
            st.session_state.trained = False
            st.session_state.model_bundle = None
            st.success("Demo data restored.")

# ---------- About ----------
elif page == "About":
    st.markdown('<div class="eyebrow">PROJECT BRIEF</div>', unsafe_allow_html=True)
    st.subheader("SentinelAI: from event signals to explainable investigation")
    st.markdown('<div class="small-muted">A behavioural intrusion-detection prototype designed to help users understand suspicious activity, rank alerts for review, and inspect related events.</div>', unsafe_allow_html=True)

    # Royal-themed workflow diagram
    st.markdown("""
    <div style="border:1px solid #3A2A68;border-radius:18px;padding:22px;background:linear-gradient(135deg,rgba(124,58,237,.15),rgba(17,16,34,.92));margin:12px 0 18px;">
      <div style="font-size:.72rem;font-weight:800;letter-spacing:.16em;color:#E8C878;margin-bottom:15px;">SENTINELAI INVESTIGATION PIPELINE</div>
      <div style="display:flex;align-items:stretch;gap:10px;flex-wrap:wrap;">
        <div style="flex:1;min-width:145px;border:1px solid #5B4392;border-radius:12px;padding:14px;background:#17132b;">
          <div style="color:#A78BFA;font-size:.72rem;font-weight:800;">01 · INGEST</div><div style="font-size:1rem;font-weight:800;color:#fff;margin:6px 0;">Event data</div><div style="font-size:.82rem;color:#B9AFE8;">Demo events or uploaded CSV logs</div>
        </div><div style="align-self:center;color:#E8C878;font-size:1.5rem;">→</div>
        <div style="flex:1;min-width:145px;border:1px solid #5B4392;border-radius:12px;padding:14px;background:#17132b;">
          <div style="color:#A78BFA;font-size:.72rem;font-weight:800;">02 · DETECT</div><div style="font-size:1rem;font-weight:800;color:#fff;margin:6px 0;">Model + risk</div><div style="font-size:.82rem;color:#B9AFE8;">Classification and anomaly scoring</div>
        </div><div style="align-self:center;color:#E8C878;font-size:1.5rem;">→</div>
        <div style="flex:1;min-width:145px;border:1px solid #5B4392;border-radius:12px;padding:14px;background:#17132b;">
          <div style="color:#A78BFA;font-size:.72rem;font-weight:800;">03 · EXPLAIN</div><div style="font-size:1rem;font-weight:800;color:#fff;margin:6px 0;">Evidence review</div><div style="font-size:.82rem;color:#B9AFE8;">Indicators and readable rationale</div>
        </div><div style="align-self:center;color:#E8C878;font-size:1.5rem;">→</div>
        <div style="flex:1;min-width:145px;border:1px solid #8B6A32;border-radius:12px;padding:14px;background:linear-gradient(145deg,#251b31,#211b25);">
          <div style="color:#E8C878;font-size:.72rem;font-weight:800;">04 · PRIORITISE</div><div style="font-size:1rem;font-weight:800;color:#fff;margin:6px 0;">Human triage</div><div style="font-size:.82rem;color:#B9AFE8;">Rank alerts and inspect related events</div>
        </div>
      </div>
      <div style="font-size:.76rem;color:#AFA1D9;margin-top:14px;">All detections and correlations are decision-support signals; an analyst must validate findings against source logs.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Current dataset snapshot")
    about_events = events.copy()
    if not about_events.empty:
        sev_counts = about_events.get("severity", pd.Series("Unknown", index=about_events.index)).value_counts().reindex(["Critical", "High", "Medium", "Low"], fill_value=0)
        left, right = st.columns(2, gap="medium")
        with left:
            st.markdown("#### Threat severity distribution")
            sev_df = sev_counts.rename_axis("Severity").reset_index(name="Events")
            sev_colors = {"Critical": "#FF4568", "High": "#FB7185", "Medium": "#E8C878", "Low": "#A78BFA"}
            sev_fig = px.pie(sev_df, names="Severity", values="Events", hole=.58, color="Severity", color_discrete_map=sev_colors)
            sev_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=330, margin=dict(l=8,r=8,t=20,b=8), font=dict(color="#D8CEFF", family="Manrope"), legend_title_text="")
            st.plotly_chart(sev_fig, use_container_width=True, config={"displayModeBar": False})
        with right:
            st.markdown("#### Risk score bands")
            risk_vals = pd.to_numeric(about_events.get("risk_score", pd.Series(0, index=about_events.index)), errors="coerce").fillna(0).clip(0, 100)
            risk_bins = pd.cut(risk_vals, bins=[-1, 34, 64, 84, 100], labels=["Low · 0–34", "Medium · 35–64", "High · 65–84", "Critical · 85–100"])
            band_df = risk_bins.value_counts(sort=False).rename_axis("Risk band").reset_index(name="Events")
            band_colors = ["#A78BFA", "#E8C878", "#FB7185", "#FF4568"]
            band_fig = px.bar(band_df, x="Risk band", y="Events", color="Risk band", color_discrete_sequence=band_colors)
            band_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=330, margin=dict(l=8,r=8,t=20,b=8), font=dict(color="#D8CEFF", family="Manrope"), showlegend=False, xaxis_title=None, yaxis_title="Events")
            band_fig.update_xaxes(showgrid=False)
            band_fig.update_yaxes(gridcolor="rgba(167,139,250,.14)")
            st.plotly_chart(band_fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Load a dataset to see its severity and risk distributions.")

    st.markdown("### Differentiating capabilities")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown('<div class="dash-card" style="min-height:190px"><div class="eyebrow">01 · EXPLAIN</div><div style="font-size:1.05rem;font-weight:800;color:#fff;margin:8px 0;">Explainable Threat Investigator</div><div class="dash-muted">Show the available indicators behind a flagged event so users can validate the result rather than treating a model output as proof.</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="dash-card" style="min-height:190px"><div class="eyebrow">02 · PRIORITISE</div><div style="font-size:1.05rem;font-weight:800;color:#fff;margin:8px 0;">Smart Alert Prioritisation</div><div class="dash-muted">Rank the review queue using risk plus transparent recurrence signals from sources and sessions.</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="dash-card" style="min-height:190px"><div class="eyebrow">03 · CORRELATE</div><div style="font-size:1.05rem;font-weight:800;color:#fff;margin:8px 0;">Possible Attack Storyline</div><div class="dash-muted">Group repeated events by shared source or session to help analysts explore a potential sequence of activity.</div></div>', unsafe_allow_html=True)

    st.markdown("### Implementation status")
    implemented, planned = st.columns(2, gap="medium")
    with implemented:
        st.markdown('<div class="dash-card"><div style="font-size:1.02rem;font-weight:800;color:#E8C878;margin-bottom:10px;">Available in this prototype</div><div class="dash-muted">• Interactive dashboard and event explorer<br>• Demo event stream and CSV ingestion<br>• Random Forest and Isolation Forest baselines<br>• Hold-out metrics and feature importance<br>• Risk/severity mapping and event explanations<br>• Recurrence-aware triage ranking<br>• Source/session-based event correlation<br>• CSV export for event and alert queues</div></div>', unsafe_allow_html=True)
    with planned:
        st.markdown('<div class="dash-card"><div style="font-size:1.02rem;font-weight:800;color:#E8C878;margin-bottom:10px;">Future development</div><div class="dash-muted">• Live packet capture or direct website/API instrumentation<br>• Production alert delivery by email, SMS, or webhook<br>• Automated response or blocking with authorised safeguards<br>• Stronger temporal correlation and attack-technique mapping<br>• Independent-dataset validation and drift monitoring<br>• Deployment hardening and access control</div></div>', unsafe_allow_html=True)

    st.markdown("### Evaluation approach")
    st.markdown("""
    - **Detection quality:** precision, recall, F1-score, and confusion matrix on held-out data.
    - **Alert usefulness:** inspect whether prioritisation brings high-risk and recurring events toward the top.
    - **Explanation quality:** verify that each explanation points to fields actually present in the event record.
    - **Generalisation:** evaluate on a separate dataset or attack family where feasible; avoid presenting demo data as real-world validation.
    """)
    st.caption("SentinelAI is a hackathon prototype. Demo events are simulated, uploaded CSVs are processed in batches, and no live network traffic is monitored or automatically blocked.")
