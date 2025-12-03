# NOVO ARQUIVO COMPLETO — Predict Cars V14-FLEX REPLAY + Painel de Séries Alternativas Inteligentes
# (substitui totalmente o arquivo anterior)

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

# ============================================================
# CONFIGURAÇÃO BÁSICA
# ============================================================

st.set_page_config(
    page_title="Predict Cars V14-FLEX REPLAY",
    layout="wide",
)

st.markdown("""
# Predict Cars V14-FLEX REPLAY
Versão FLEX: número variável de passageiros + modo replay automático + validação empírica.
""")

# ============================================================
# CONSTANTES
# ============================================================

NUM_MIN = 1
NUM_MAX = 60

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def _coerce_int(x: Any) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return 0

# --- Parser FLEX ---

def preparar_historico_V14(df_raw: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for i, row in enumerate(df_raw.itertuples(index=False)):  # type: ignore
        valores = list(row)
        if not valores:
            continue

        s0 = str(valores[0]).strip()
        if s0 and not s0.isdigit():
            id_serie = s0
            resto = valores[1:]
        else:
            id_serie = f"C{i+1}"
            resto = valores

        if len(resto) < 2:
            continue

        k_val = _coerce_int(resto[-1])
        passageiros = [_coerce_int(x) for x in resto[:-1] if _coerce_int(x) > 0]
        if not passageiros:
            continue

        linhas.append({
            "id": id_serie,
            "passageiros": passageiros,
            "k": k_val,
            "n_passageiros": len(passageiros),
        })

    df = pd.DataFrame(linhas)
    if not df.empty:
        df["idx_numeric"] = range(1, len(df)+1)
    return df


def preparar_historico_de_texto(texto: str) -> pd.DataFrame:
    linhas = []
    for i, raw_line in enumerate(texto.splitlines()):
        linha = raw_line.strip()
        if not linha:
            continue

        if ";" in linha:
            partes = [p.strip() for p in linha.split(";")]
        elif "," in linha:
            partes = [p.strip() for p in linha.split(",")]
        else:
            partes = [p.strip() for p in linha.split()]

        if not partes:
            continue

        s0 = partes[0]
        if s0 and not s0.isdigit():
            id_serie = s0
            resto = partes[1:]
        else:
            id_serie = f"C{i+1}"
            resto = partes

        if len(resto) < 2:
            continue

        k_val = _coerce_int(resto[-1])
        passageiros = [_coerce_int(x) for x in resto[:-1] if _coerce_int(x) > 0]
        if not passageiros:
            continue

        linhas.append({
            "id": id_serie,
            "passageiros": passageiros,
            "k": k_val,
            "n_passageiros": len(passageiros),
        })

    df = pd.DataFrame(linhas)
    if not df.empty:
        df["idx_numeric"] = range(1, len(df)+1)
    return df

# ============================================================
# MÓDULO DE RISCO — k & k*
# ============================================================

def avaliar_risco_k(df: pd.DataFrame) -> Tuple[str, str]:
    if df.empty or "k" not in df.columns:
        return (
            "⚠️ k histórico da série alvo
Dados insuficientes.",
            "⚡ k* (sentinela preditivo)
Dados insuficientes.",
        )

    k_ultimo = int(df.iloc[-1]["k"])
    if k_ultimo <= 0:
        desc_k = "⚠️ k histórico da série alvo
🟢 Ambiente estável — regime normal."
    elif k_ultimo == 1:
        desc_k = "⚠️ k histórico da série alvo
🟡 Atenção — pré-ruptura local."
    else:
        desc_k = "⚠️ k histórico da série alvo
🔴 Ambiente crítico — turbulência elevada."

    janela = min(50, len(df))
    sub = df.tail(janela)
    prop = float((sub["k"] > 0).mean())
    risco = int(round(prop * 100))

    if risco <= 15:
        desc_k_star = f"⚡ k* (sentinela preditivo)
🟢 Tendência estável (risco ≈ {risco}%)."
    elif risco <= 40:
        desc_k_star = f"⚡ k* (sentinela preditivo)
🟡 Ruído moderado (risco ≈ {risco}%)."
    else:
        desc_k_star = f"⚡ k* (sentinela preditivo)
🔴 Tendência turbulenta (risco ≈ {risco}%)."

    return desc_k, desc_k_star

# ============================================================
# PIPELINE V14-FLEX — IPF, IPO, S6, Modo E
# ============================================================

def extrair_contexto(df: pd.DataFrame, idx_alvo: int, janela: int = 30) -> pd.DataFrame:
    pos = max(0, min(len(df)-1, idx_alvo-1))
    inicio = max(0, pos-janela)
    return df.iloc[inicio:pos].copy()


def gerar_leque_original(contexto: pd.DataFrame) -> list:
    cont = {}
    for passageiros in contexto.get("passageiros", []):
        for p in passageiros:
            cont[p] = cont.get(p, 0) + 1
    ordenado = sorted(cont.items(), key=lambda kv: (-kv[1], kv[0]))
    return sorted({n for n, _ in ordenado[:25]})


def gerar_leque_corrigido(contexto: pd.DataFrame, leque_original: list) -> list:
    if contexto.empty or not leque_original:
        return leque_original
    rec = contexto.tail(min(10, len(contexto)))
    cont = {}
    for passageiros in rec["passageiros"]:
        for p in passageiros:
            cont[p] = cont.get(p, 0) + 1
    filtrado = [n for n in leque_original if cont.get(n, 0) >= 1]
    return sorted(set(filtrado or leque_original))


def gerar_leque_misto(lo: list, lc: list) -> list:
    return sorted(set(lo) | set(lc))

# --- SELEÇÃO FINAL (Modo E) ---

def selecionar_serie_final_modo_E(leque_final: list[int]) -> list[int]:
    if not leque_final:
        return []
    nums = sorted(set(leque_final))
    n = len(nums)
    if n <= 6:
        return nums
    # recorte central
    if n > 10:
        corte = max(1, int(0.2*n))
        centro = nums[corte:-corte] or nums
    else:
        centro = nums
    indices = [0.12, 0.30, 0.48, 0.62, 0.78, 0.90]
    esc = []
    usados = set()
    m = len(centro)
    for rel in indices:
        idx = int(round(rel*(m-1)))
        idx = max(0, min(m-1, idx))
        v = centro[idx]
        if v not in usados:
            esc.append(v)
            usados.add(v)
    # completar
    if len(esc) < 6:
        for v in centro:
            if v not in usados:
                esc.append(v)
                usados.add(v)
                if len(esc) == 6:
                    break
    return sorted(esc)

# --- EXECUÇÃO DO PIPELINE COMPLETO ---

def executar_pipeline_v14_flex(df: pd.DataFrame, idx_alvo: int, janela: int = 30) -> Dict[str, Any]:
    if df.empty:
        return {}
    pos = max(0, min(len(df)-1, idx_alvo-1))
    alvo = df.iloc[pos]
    contexto = extrair_contexto(df, idx_alvo, janela)
    lo = gerar_leque_original(contexto)
    lc = gerar_leque_corrigido(contexto, lo)
    lm = gerar_leque_misto(lo, lc)
    leque_final = lm
    serie_final = selecionar_serie_final_modo_E(leque_final)
    desc_k, desc_k_star = avaliar_risco_k(df.iloc[:pos+1])
    return {
        "id_alvo": alvo["id"],
        "passageiros_alvo": alvo["passageiros"],
        "k_alvo": int(alvo["k"]),
        "leque_original": lo,
        "leque_corrigido": lc,
        "leque_misto": lm,
        "leque_final": leque_final,
        "serie_final": serie_final,
        "desc_k": desc_k,
        "desc_k_star": desc_k_star,
    }

# ============================================================
# MÓDULO — SÉRIES ALTERNATIVAS INTELIGENTES V14-FLEX
# ============================================================

def estimar_confiabilidade_heuristica(tipo: str, tamanho_leque: int) -> dict:
    base = {
        "principal": 0.75,
        "conservadora": 0.80,
        "intermediaria": 0.70,
        "agressiva": 
