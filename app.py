import itertools
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
#  PREDICT CARS V13.8 — APP STREAMLIT (VERSÃO ORGANIZADA)
# ---------------------------------------------------------
#  Estrutura geral:
#  - Entrada de dados
#  - Estado atual + Barômetro
#  - IDX Avançado (similaridade + recência)
#  - Núcleo Resiliente (IPF + IPO + Anti-SelfBias)
#  - Séries Puras
#  - Séries Avaliadas (ICA + HLA + Faróis + Confiabilidade)
#  - Gerador Extra (por confiabilidade alvo)
#  - S6 + Ensamble Final
# ---------------------------------------------------------


# =========================
#  FUNÇÕES UTILITÁRIAS
# =========================
def parse_series_text(text: str) -> List[int]:
    if not text:
        return []
    separators = [",", ";", " "]
    tmp = text
    for sep in separators[1:]:
        tmp = tmp.replace(sep, separators[0])
    parts = [p.strip() for p in tmp.split(separators[0]) if p.strip()]
    return [int(p) for p in parts]


def series_to_str(series: List[int]) -> str:
    return " ".join(str(x) for x in series)


def ensure_session_state():
    defaults = {
        "data": None,
        "current_index": None,
        "current_series": [],
        "nucleo_resiliente": [],
        "idx_info": None,
        "series_puras": [],
        "series_avaliadas": pd.DataFrame(),
        "series_extras": [],
        "s6_ensamble": pd.DataFrame(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# =========================
#  CAMADA DE DADOS
# =========================
def ca
