# ============================================================
# PredictCars V15.7 MAX → V16 Premium
# Parabólica — Curvatura do Erro (Governança Pré-C4)
# Arquivo completo | NÃO altera Camada 4 | Leitura observacional
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime

ARQUIVO_ATIVO = "app_v15_7_MAX_com_orbita_P2_PARABOLA_PRE_C4_FIX.py"

st.set_page_config(page_title="PredictCars — Parabólica Pré-C4", layout="wide")
st.markdown(f"### 🧾 Rodando arquivo: `{ARQUIVO_ATIVO}`")

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def _state_from_curvature(c):
    if c < -0.05:
        return "DESCENDO"
    if c > 0.05:
        return "SUBINDO"
    return "PLANA"

snapshots = st.session_state.get("snapshots_p0", [])

st.markdown("## 📐 Parabólica — Curvatura do Erro (Governança Pré-C4)")

if not snapshots or len(snapshots) < 3:
    st.warning("É necessário ao menos 3 Snapshots P0 registrados para calcular a curvatura.")
    st.stop()

snapshots = sorted(snapshots, key=lambda s: s.get("k", 0))

E = []
Ks = []
for s in snapshots:
    v = s.get("fora_longe")
    if v is None:
        v = s.get("fora_total", 0)
    E.append(_safe_float(v))
    Ks.append(s.get("k"))

dE = np.diff(E)
ddE = np.diff(dE)

curvatura_atual = float(ddE[-1]) if len(ddE) else 0.0
estado = _state_from_curvature(curvatura_atual)

st.markdown("### 🧮 Métricas")
st.json({
    "Ks": Ks[-5:],
    "E_fora_longe": E[-5:],
    "dE": dE[-4:].tolist() if len(dE) else [],
    "curvatura": curvatura_atual,
    "estado_parabolica": estado
})

st.markdown("### 🔒 Governo Estrutural")
st.json({
    "P1": "permitido" if estado in ("DESCENDO", "PLANA") else "perde eficiência",
    "P2": "pode acordar" if estado == "SUBINDO" else "vetado"
})

st.caption("Leitura pré-C4. Não gera listas. Não altera Camada 4.")
