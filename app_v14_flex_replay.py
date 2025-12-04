# ============================================================
# Predict Cars V14-FLEX ULTRA REAL (TURBO++)
# streamlit_app.py — NOVO ARQUIVO COMPLETO (PARTE 1/4)
#
# Versão: V14-FLEX ULTRA REAL (TURBO++)
# Características principais:
#   - Entrada FLEX (n variável de passageiros)
#   - Barômetro ULTRA REAL
#   - k* ULTRA REAL
#   - IDX ULTRA
#   - IPF / IPO refinados
#   - S6 Profundo (camadas)
#   - Micro-Leque ULTRA
#   - Monte Carlo Profundo ULTRA
#   - Pipeline de Previsão ULTRA REAL
#   - QDS REAL + Backtest REAL
#   - Replay LIGHT
#   - Replay ULTRA com horizonte ajustável
#   - Modo TURBO++ Adaptativo
#
# Arquivo dividido em 4 partes para facilitar a colagem.
# Esta é a PARTE 1/4.
# ============================================================

import io
import math
import textwrap
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# CONFIGURAÇÃO BÁSICA DO APP
# ------------------------------------------------------------

st.set_page_config(
    page_title="Predict Cars V14-FLEX ULTRA REAL (TURBO++)",
    layout="wide",
)

# ------------------------------------------------------------
# FUNÇÕES GERAIS DE SESSÃO E UTILITÁRIOS
# ------------------------------------------------------------

def inicializar_sessao() -> None:
    """
    Garante que todas as chaves importantes existam em st.session_state.
    Não perde nada que já esteja setado.
    """
    defaults = {
        "df": None,                  # histórico preparado
        "raw_df": None,              # histórico bruto
        "meta_cols": None,           # metadados sobre colunas
        "regime_state": "estavel",   # estado inicial do barômetro
        "k_star_estado": "estavel",  # estado inicial do k*
        "ultimo_idx_alvo": None,     # último índice alvo selecionado
        "log_eventos": [],           # lista de eventos/importantes do app
        "qds_cache": {},             # cache para QDS / Backtest
        "replay_cache": {},          # cache para modos de replay
        "turbo_config": {},          # configuração atual do modo TURBO++
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def registrar_evento(msg: str) -> None:
    """
    Registra um evento no log interno do app.
    """
    if "log_eventos" not in st.session_state:
        st.session_state["log_eventos"] = []
    st.session_state["log_eventos"].append(msg)


def mostrar_log_eventos(max_linhas: int = 15) -> None:
    """
    Mostra as últimas linhas do log em um expander discreto.
    """
    log = st.session_state.get("log_eventos", [])
    if not log:
        return
    with st.expander("📜 Log interno (últimos eventos)", expanded=False):
        for linha in log[-max_linhas:]:
            st.markdown(f"- {linha}")


def detectar_separador(conteudo: str) -> str:
    """
    Tenta detectar o separador principal de um bloco de texto.
    """
    # Conta ocorrências por linha
    linhas = [l for l in conteudo.splitlines() if l.strip()]
    if not linhas:
        return ";"

    candidatos = [";", ",", "\t", " "]
    contagens = {sep: 0 for sep in candidatos}
    for ln in linhas[:20]:
        for sep in candidatos:
            contagens[sep] += ln.count(sep)

    # Escolhe o separador com maior contagem
    sep_escolhido = max(contagens, key=contagens.get)
    if contagens[sep_escolhido] == 0:
        # fallback
        return ";"
    return sep_escolhido


def carregar_df_de_texto(conteudo: str) -> pd.DataFrame:
    """
    Lê o conteúdo colado pelo usuário e converte em DataFrame.
    Tenta detectar o separador.
    """
    sep = detectar_separador(conteudo)
    buffer = io.StringIO(conteudo.strip())
    try:
        df_raw = pd.read_csv(buffer, sep=sep, header=None)
    except Exception:
        # fallback bruto
        buffer.seek(0)
        df_raw = pd.read_csv(buffer, sep=";", header=None)
    return df_raw


def preparar_historico_flex(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Prepara o histórico no formato FLEX, assumindo:
        - 1ª coluna: ID da série (ex: "C1", "C2"... ou número)
        - colunas intermediárias: passageiros (n variável)
        - última coluna: k (rótulo)

    Retorna:
        df (DataFrame) com colunas:
            ["id", "p_1", ..., "p_n", "k"]
        meta (dict) com detalhes da estrutura.
    """
    if df_raw is None or df_raw.empty:
        raise ValueError("DataFrame bruto vazio.")

    df_work = df_raw.copy()

    # Remove colunas totalmente vazias
    df_work = df_work.dropna(axis=1, how="all")

    if df_work.shape[1] < 3:
        raise ValueError(
            "O histórico precisa de pelo menos 3 colunas: ID, passageiros, k."
        )

    n_cols = df_work.shape[1]
    idx_col = 0
    k_col = n_cols - 1
    n_passageiros = n_cols - 2  # tudo entre ID e k

    df = pd.DataFrame()
    df["id"] = df_work.iloc[:, idx_col]

    # Passageiros
    for i in range(n_passageiros):
        df[f"p_{i+1}"] = pd.to_numeric(
            df_work.iloc[:, 1 + i], errors="coerce"
        )

    # k
    df["k"] = pd.to_numeric(df_work.iloc[:, k_col], errors="coerce")

    # Limpa linhas totalmente vazias de passageiros
    passageiros_cols = [c for c in df.columns if c.startswith("p_")]
    df = df.dropna(subset=passageiros_cols, how="all")

    # Reseta índice
    df = df.reset_index(drop=True)

    # Cria coluna de índice numérico interno (1,2,3,...)
    df["idx"] = np.arange(1, len(df) + 1)

    # Metadados
    meta = {
        "n_series": len(df),
        "n_passageiros": n_passageiros,
        "cols_passageiros": passageiros_cols,
        "col_k": "k",
        "col_id": "id",
    }

    return df, meta


def extrair_passageiros_linha(df: pd.DataFrame, idx: int) -> List[int]:
    """
    Dado um DataFrame preparado e um índice interno (1-based, coluna 'idx'),
    retorna a lista de passageiros daquela série.
    """
    if df is None or df.empty:
        return []

    row = df.loc[df["idx"] == idx]
    if row.empty:
        return []

    passageiros_cols = [c for c in df.columns if c.startswith("p_")]
    valores = row[passageiros_cols].iloc[0].tolist()
    return [int(x) for x in valores if not pd.isna(x)]


def resumo_rapido_serie(df: pd.DataFrame, idx: int) -> str:
    """
    Gera um resumo rápido da série (ID, passageiros, k) para exibição.
    """
    if df is None or df.empty:
        return "Série inválida."

    linha = df.loc[df["idx"] == idx]
    if linha.empty:
        return "Série não encontrada."

    row = linha.iloc[0]
    passageiros_cols = [c for c in df.columns if c.startswith("p_")]
    passageiros = [int(x) for x in row[passageiros_cols].tolist() if not pd.isna(x)]
    k_val = row["k"]
    return f"{row['id']} — Passageiros: {passageiros} — k: {int(k_val) if not pd.isna(k_val) else 'NA'}"


# ------------------------------------------------------------
# PLACEHOLDERS DE MÓDULOS ULTRA
# (os detalhes internos serão desenvolvidos nas partes 2/4 e 3/4)
# ------------------------------------------------------------

def calcular_barometro_ultra(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Barômetro ULTRA REAL — placeholder de alto nível.
    Implementação detalhada virá na PARTE 2/4.
    """
    if df is None or df.empty:
        return {
            "estado": "desconhecido",
            "indice_turbulencia": None,
            "texto": "Histórico não carregado.",
        }

    # Placeholder simplificado (será refinado na parte 2/4)
    n_series = len(df)
    k_vals = df["k"].fillna(0).values
    vol_k = float(np.std(k_vals)) if len(k_vals) > 1 else 0.0

    if vol_k < 0.5:
        estado = "estavel"
        texto = "Ambiente historicamente estável."
    elif vol_k < 1.0:
        estado = "atencao"
        texto = "Flutuações moderadas — atenção."
    else:
        estado = "critico"
        texto = "Turbulência intensa na estrada."

    return {
        "estado": estado,
        "indice_turbulencia": vol_k,
        "texto": texto,
        "n_series": n_series,
    }


def calcular_k_star_ultra(df: pd.DataFrame) -> Dict[str, Any]:
    """
    k* ULTRA REAL — placeholder de alto nível.
    Implementação detalhada virá na PARTE 2/4.
    """
    if df is None or df.empty:
        return {
            "estado": "desconhecido",
            "k_star": None,
            "texto": "Histórico não carregado.",
        }

    # Placeholder simples com base na entropia do k
    k_vals = df["k"].dropna().values
    if len(k_vals) == 0:
        return {
            "estado": "desconhecido",
            "k_star": None,
            "texto": "k insuficiente para calcular k*.",
        }

    unique, counts = np.unique(k_vals, return_counts=True)
    probs = counts / counts.sum()
    entropia = -float(np.sum(probs * np.log2(probs + 1e-9)))

    # Normaliza entropia em 0-1 (placeholder)
    ent_norm = entropia / (math.log2(len(unique) + 1e-9))

    if ent_norm < 0.33:
        estado = "estavel"
        texto = "k*: ambiente concentrado — baixa turbulência estrutural."
    elif ent_norm < 0.66:
        estado = "atencao"
        texto = "k*: ambiente misto — pré-ruptura residual possível."
    else:
        estado = "critico"
        texto = "k*: ambiente crítico — alta dispersão estrutural."

    return {
        "estado": estado,
        "k_star": ent_norm,
        "texto": texto,
    }


# ============================================================
# INÍCIO DO APP — NAVEGAÇÃO GERAL
# ============================================================

inicializar_sessao()

st.title("🚗 Predict Cars V14-FLEX ULTRA REAL (TURBO++)")

st.markdown(
    """
Sistema ULTRA completo com:
**Barômetro**, **k\***, **IDX**, **IPF / IPO**, **S6 Profundo**, **Micro-Leque**,  
**Monte Carlo Profundo**, **QDS + Backtest**, **Replay LIGHT / ULTRA** e **Modo TURBO++ Adaptativo**.
"""
)

# ------------------------------------------------------------
# SIDEBAR — NAVEGAÇÃO PRINCIPAL
# ------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🧭 Navegação")

    painel = st.radio(
        "Escolha o painel:",
        [
            "📥 Histórico — Entrada (FLEX)",
            "🔍 Pipeline V14-FLEX ULTRA",
            "🚨 Monitor de Risco (Barômetro + k*)",
            "📊 IDX / IPF / IPO ULTRA",
            "🧬 S6 Profundo & Micro-Leque ULTRA",
            "🎲 Monte Carlo Profundo ULTRA",
            "🧪 QDS REAL & Backtest REAL",
            "📅 Replay LIGHT",
            "📅 Replay ULTRA (Horizonte Ajustável)",
            "🚀 Modo TURBO++ ULTRA",
        ],
    )

    st.markdown("---")
    mostrar_log = st.checkbox("Mostrar log interno", value=False)
    if mostrar_log:
        mostrar_log_eventos()

# ============================================================
# PAINEL 1 — HISTÓRICO — ENTRADA (FLEX)
# ============================================================

if painel == "📥 Histórico — Entrada (FLEX)":
    st.markdown("## 📥 Histórico — Entrada (FLEX)")

    st.markdown(
        """
Este painel permite carregar o **histórico FLEX**, onde o número de passageiros pode variar:

- 1ª coluna: **ID da série** (ex.: `C1`, `C2`, ...).
- Colunas intermediárias: **passageiros** (p_1, p_2, ..., p_n).
- Última coluna: **k** (rótulo / classe).
"""
    )

    opc = st.radio(
        "Como deseja carregar o histórico?",
        ["Enviar arquivo CSV", "Copiar e colar o histórico"],
        horizontal=True,
    )

    df = None
    meta = None

    # ---------- OPÇÃO 1 — UPLOAD DE ARQUIVO ----------
    if opc == "Enviar arquivo CSV":
        file = st.file_uploader("Selecione o arquivo CSV:", type=["csv"])
        if file is not None:
            try:
                df_raw = pd.read_csv(file, header=None)
                df_prep, meta = preparar_historico_flex(df_raw)

                st.session_state["raw_df"] = df_raw
                st.session_state["df"] = df_prep
                st.session_state["meta_cols"] = meta

                registrar_evento(
                    f"Histórico carregado via CSV: {meta['n_series']} séries, "
                    f"{meta['n_passageiros']} passageiros."
                )

                st.success("✅ Histórico carregado e preparado com sucesso!")
                st.write("**Prévia do histórico preparado:**")
                st.dataframe(df_prep.head(20))

                st.info(
                    f"Séries: `{meta['n_series']}` | Passageiros por série: `{meta['n_passageiros']}`"
                )
            except Exception as e:
                st.error(f"Erro ao carregar CSV: {e}")
                registrar_evento(f"Erro ao carregar CSV: {e}")

    # ---------- OPÇÃO 2 — COLAR TEXTO ----------
    if opc == "Copiar e colar o histórico":
        exemplo = textwrap.dedent(
            """
            C1;41;5;4;52;30;33;0
            C2;9;39;37;49;43;41;1
            C3;36;30;10;11;29;47;2
            C4;6;59;42;27;1;5;0
            C5;1;19;46;6;16;2;0
            """
        ).strip()

        texto = st.text_area(
            "Cole abaixo o histórico (ID;passageiros;k):",
            value=exemplo,
            height=200,
        )

        if st.button("📥 Processar histórico colado"):
            if not texto.strip():
                st.warning("Cole algum conteúdo antes de processar.")
            else:
                try:
                    df_raw = carregar_df_de_texto(texto)
                    df_prep, meta = preparar_historico_flex(df_raw)

                    st.session_state["raw_df"] = df_raw
                    st.session_state["df"] = df_prep
                    st.session_state["meta_cols"] = meta

                    registrar_evento(
                        f"Histórico carregado via texto colado: {meta['n_series']} séries, "
                        f"{meta['n_passageiros']} passageiros."
                    )

                    st.success("✅ Histórico colado e preparado com sucesso!")
                    st.write("**Prévia do histórico preparado:**")
                    st.dataframe(df_prep.head(20))

                    st.info(
                        f"Séries: `{meta['n_series']}` | Passageiros por série: `{meta['n_passageiros']}`"
                    )
                except Exception as e:
                    st.error(f"Erro ao processar histórico colado: {e}")
                    registrar_evento(f"Erro ao processar histórico colado: {e}")

    st.markdown("---")

    df = st.session_state.get("df", None)
    meta = st.session_state.get("meta_cols", None)

    if df is not None and meta is not None:
        st.markdown("### 🔎 Resumo do histórico atual")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de séries", meta["n_series"])
        with col2:
            st.metric("Passageiros por série", meta["n_passageiros"])
        with col3:
            st.metric("Coluna k", meta["col_k"])

        # Escolher uma série para inspecionar rapidamente
        idx_max = int(df["idx"].max())
        idx_sel = st.number_input(
            "Selecione um índice interno para inspecionar (1 = primeira série carregada):",
            min_value=1,
            max_value=idx_max,
            value=idx_max,
            step=1,
        )

        st.markdown("#### 📌 Série selecionada")
        st.code(resumo_rapido_serie(df, idx_sel), language="text")

        st.markdown("#### 📊 Visualização tabular (completa)")
        st.dataframe(df)

    else:
        st.info("Carregue o histórico para habilitar os demais painéis.")

# ============================================================
# FIM DA PARTE 1/4
# As próximas partes (2/4, 3/4, 4/4) continuarão a partir daqui.
# ============================================================
# ============================================================
# PARTE 2/4
# Barômetro ULTRA REAL, k* ULTRA REAL, IDX / IPF / IPO ULTRA,
# Pipeline de Previsão V14-FLEX ULTRA
# ============================================================

# ------------------------------------------------------------
# (Re)definição detalhada do Barômetro ULTRA REAL
# ------------------------------------------------------------

def calcular_barometro_ultra(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Barômetro ULTRA REAL:
    Mede a turbulência global da estrada a partir da série de k.

    Ideia ULTRA:
        - Usa desvio-padrão de k.
        - Usa média do módulo da variação de k (|Δk|).
        - Combina em um índice de turbulência normalizado (0–1).
        - Mapeia para estados: estável / atenção / crítico.
    """
    if df is None or df.empty:
        resultado = {
            "estado": "desconhecido",
            "indice_turbulencia": None,
            "texto": "Histórico não carregado.",
            "vol_k": None,
            "media_delta_k": None,
            "n_series": 0,
        }
        st.session_state["regime_state"] = "desconhecido"
        return resultado

    k_vals = df["k"].dropna().astype(float).values
    n_series = len(k_vals)

    if n_series < 3:
        resultado = {
            "estado": "estavel",
            "indice_turbulencia": 0.0,
            "texto": "Histórico curto — assumindo ambiente estável por falta de dados.",
            "vol_k": 0.0,
            "media_delta_k": 0.0,
            "n_series": n_series,
        }
        st.session_state["regime_state"] = "estavel"
        return resultado

    # Desvio-padrão de k
    vol_k = float(np.std(k_vals))

    # Média do módulo da variação de k
    deltas = np.diff(k_vals)
    media_delta_k = float(np.mean(np.abs(deltas))) if len(deltas) > 0 else 0.0

    # Normalização simples (ULTRA, mas controlada)
    # Escala "esperada" para k: assumimos k em [0, 5] como típico;
    # se for diferente, a normalização continua funcionando.
    escala_vol = max(np.max(np.abs(k_vals)), 1.0)
    escala_delta = max(np.max(np.abs(deltas)), 1.0) if len(deltas) > 0 else 1.0

    vol_norm = min(vol_k / (escala_vol + 1e-9), 1.0)
    delta_norm = min(media_delta_k / (escala_delta + 1e-9), 1.0)

    indice_turbulencia = float(0.6 * vol_norm + 0.4 * delta_norm)

    if indice_turbulencia < 0.3:
        estado = "estavel"
        texto = "Barômetro: estrada historicamente estável."
    elif indice_turbulencia < 0.6:
        estado = "atencao"
        texto = "Barômetro: turbulência moderada — atenção elevada."
    else:
        estado = "critico"
        texto = "Barômetro: turbulência pesada — ambiente crítico."

    resultado = {
        "estado": estado,
        "indice_turbulencia": indice_turbulencia,
        "texto": texto,
        "vol_k": vol_k,
        "media_delta_k": media_delta_k,
        "n_series": n_series,
    }

    st.session_state["regime_state"] = estado
    return resultado


# ------------------------------------------------------------
# (Re)definição detalhada do k* ULTRA REAL
# ------------------------------------------------------------

def calcular_k_star_ultra(df: pd.DataFrame) -> Dict[str, Any]:
    """
    k* ULTRA REAL:
    Mede o nível estrutural de desordem de k (entropia) e a assimetria
    entre estados "calmos" e "tensos".

    Ideia ULTRA:
        - Entropia da distribuição de k.
        - Foco maior nas últimas janelas (regime atual).
        - Normaliza em 0–100 como "sensibilidade" do sentinela k*.
    """
    if df is None or df.empty:
        resultado = {
            "estado": "desconhecido",
            "k_star": None,
            "texto": "Histórico não carregado.",
            "entropia": None,
            "entropia_norm": None,
            "n_k": 0,
        }
        st.session_state["k_star_estado"] = "desconhecido"
        return resultado

    k_vals = df["k"].dropna().astype(float).values
    n_k = len(k_vals)
    if n_k == 0:
        resultado = {
            "estado": "desconhecido",
            "k_star": None,
            "texto": "k insuficiente para calcular k*.",
            "entropia": None,
            "entropia_norm": None,
            "n_k": 0,
        }
        st.session_state["k_star_estado"] = "desconhecido"
        return resultado

    # Foco nas últimas 80 séries (ou tudo, se menor)
    janela = min(80, n_k)
    k_focus = k_vals[-janela:]

    unique, counts = np.unique(k_focus, return_counts=True)
    probs = counts / counts.sum()

    entropia = -float(np.sum(probs * np.log2(probs + 1e-12)))
    entropia_max = math.log2(len(unique) + 1e-12)
    ent_norm = float(entropia / (entropia_max + 1e-12)) if entropia_max > 0 else 0.0

    # k* em escala 0–100 (ULTRA)
    k_star_val = float(ent_norm * 100.0)

    if ent_norm < 0.33:
        estado = "estavel"
        texto = "k*: ambiente concentrado — estrutura previsível, baixa dispersão."
    elif ent_norm < 0.66:
        estado = "atencao"
        texto = "k*: ambiente misto — sinais de pré-ruptura residual."
    else:
        estado = "critico"
        texto = "k*: ambiente crítico — alta dispersão estrutural."

    resultado = {
        "estado": estado,
        "k_star": k_star_val,
        "texto": texto,
        "entropia": entropia,
        "entropia_norm": ent_norm,
        "n_k": n_k,
    }

    st.session_state["k_star_estado"] = estado
    return resultado


# ------------------------------------------------------------
# IDX / IPF / IPO ULTRA — Núcleos de referência
# ------------------------------------------------------------

def calcular_nucleos_idx_ipf_ipo_ultra(
    df: pd.DataFrame,
    idx_alvo: int,
    meta: Dict[str, Any],
    janela_max: int = 40,
) -> Dict[str, Any]:
    """
    Calcula os núcleos IDX / IPF / IPO ULTRA em torno de um índice alvo.

    Estratégia ULTRA (mas totalmente determinística):
        - Janela de histórico [idx_alvo - janela, idx_alvo - 1].
        - IDX ULTRA: média ponderada dos passageiros (peso maior nas séries mais recentes).
        - IPF ULTRA: mediana por passageiro (robusto a outliers).
        - IPO ULTRA original: média simples.
        - IPO ULTRA refinado: reorganização leve da IPO original para reduzir auto-sesgo.
    """
    if df is None or df.empty or meta is None:
        return {
            "ok": False,
            "msg": "Histórico ou metadados indisponíveis.",
        }

    if idx_alvo <= 1:
        return {
            "ok": False,
            "msg": "Índice alvo precisa ser maior que 1 para ter histórico anterior.",
        }

    idx_min = int(df["idx"].min())
    idx_max = int(df["idx"].max())
    idx_alvo = int(idx_alvo)

    if not (idx_min <= idx_alvo <= idx_max):
        return {
            "ok": False,
            "msg": f"Índice alvo fora do intervalo [{idx_min}, {idx_max}].",
        }

    # Janela de histórico ULTRA
    janela = min(janela_max, idx_alvo - idx_min)
    inicio = max(idx_min, idx_alvo - janela)
    fim = idx_alvo - 1

    df_janela = df[(df["idx"] >= inicio) & (df["idx"] <= fim)].copy()
    if df_janela.empty:
        return {
            "ok": False,
            "msg": "Janela de histórico vazia.",
        }

    cols_pass = meta["cols_passageiros"]
    n_pass = meta["n_passageiros"]

    # Matriz de passageiros (linhas = séries, colunas = passageiros)
    M = df_janela[cols_pass].astype(float).values

    # ---------- IPO ULTRA: média simples + reorganização anti-sesgo ----------
    ipo_original = np.nanmean(M, axis=0)

    # Anti-selfbias: dá mais peso às extremidades da janela
    pesos_border = np.linspace(1.0, 1.5, len(df_janela))
    pesos_border[: len(df_janela) // 4] *= 1.2
    pesos_border[-len(df_janela) // 4 :] *= 1.2
    pesos_border = pesos_border / (pesos_border.sum() + 1e-9)

    ipo_ultra = np.average(M, axis=0, weights=pesos_border)

    # Ordem refinada: tenta aproximar dos passageiros da série alvo
    serie_alvo = df.loc[df["idx"] == idx_alvo, cols_pass].astype(float).iloc[0].values
    diffs = np.abs(ipo_ultra - serie_alvo)
    ordem = np.argsort(diffs)  # ordena por proximidade à série alvo
    ipo_refinado = ipo_ultra[ordem]

    # ---------- IPF ULTRA: mediana (robusta) ----------
    ipf_ultra = np.nanmedian(M, axis=0)

    # ---------- IDX ULTRA: média ponderada, pesos crescentes ----------
    n_linhas = M.shape[0]
    pesos = np.linspace(1.0, 2.0, n_linhas)
    pesos = pesos / (pesos.sum() + 1e-9)
    idx_ultra = np.average(M, axis=0, weights=pesos)

    def arr_to_int_list(a: np.ndarray) -> List[int]:
        return [int(round(x)) for x in a.tolist()]

    resultado = {
        "ok": True,
        "msg": "",
        "idx_ultra": arr_to_int_list(idx_ultra),
        "ipf_ultra": arr_to_int_list(ipf_ultra),
        "ipo_original": arr_to_int_list(ipo_original),
        "ipo_ultra": arr_to_int_list(ipo_refinado),
        "janela_inicio": int(inicio),
        "janela_fim": int(fim),
        "janela_tamanho": int(len(df_janela)),
        "idx_alvo": int(idx_alvo),
        "serie_alvo": arr_to_int_list(serie_alvo),
    }

    return resultado


# ------------------------------------------------------------
# Painel: 🚨 Monitor de Risco (Barômetro + k*)
# ------------------------------------------------------------

if painel == "🚨 Monitor de Risco (Barômetro + k*)":
    st.markdown("## 🚨 Monitor de Risco (Barômetro + k*)")

    df = st.session_state.get("df", None)

    if df is None or df.empty:
        st.warning("Carregue o histórico no painel '📥 Histórico — Entrada (FLEX)' antes.")
    else:
        meta = st.session_state.get("meta_cols", None)

        col_top1, col_top2 = st.columns(2)

        with col_top1:
            st.markdown("### 🌡️ Barômetro ULTRA REAL")
            bar = calcular_barometro_ultra(df)

            if bar["estado"] == "estavel":
                emoji = "🟢"
            elif bar["estado"] == "atencao":
                emoji = "🟡"
            elif bar["estado"] == "critico":
                emoji = "🔴"
            else:
                emoji = "⚪"

            st.markdown(f"{emoji} **Estado do barômetro:** `{bar['estado']}`")
            st.write(bar["texto"])

            st.metric("Índice de turbulência", f"{bar['indice_turbulencia']:.3f}" if bar["indice_turbulencia"] is not None else "NA")
            st.metric("Desvio-padrão de k", f"{bar['vol_k']:.3f}" if bar["vol_k"] is not None else "NA")
            st.metric("Média de |Δk|", f"{bar['media_delta_k']:.3f}" if bar["media_delta_k"] is not None else "NA")

        with col_top2:
            st.markdown("### 🛰️ k* ULTRA REAL (Sentinela)")
            kinfo = calcular_k_star_ultra(df)

            if kinfo["estado"] == "estavel":
                emoji = "🟢"
            elif kinfo["estado"] == "atencao":
                emoji = "🟡"
            elif kinfo["estado"] == "critico":
                emoji = "🔴"
            else:
                emoji = "⚪"

            k_star_val = kinfo["k_star"]
            st.markdown(f"{emoji} **Estado do k\\*:** `{kinfo['estado']}`")
            st.write(kinfo["texto"])
            st.metric("k* (0–100)", f"{k_star_val:.1f}" if k_star_val is not None else "NA")

            if kinfo["entropia"] is not None:
                st.metric("Entropia de k", f"{kinfo['entropia']:.3f}")
                st.metric("Entropia normalizada", f"{kinfo['entropia_norm']:.3f}")

        st.markdown("---")

        # Combinação Barômetro + k*
        regime = st.session_state.get("regime_state", "desconhecido")
        k_estado = st.session_state.get("k_star_estado", "desconhecido")

        # Regras simples de combinação
        if regime == "critico" or k_estado == "critico":
            risco_global = "crítico"
            emoji_global = "🔴"
            texto_global = "Ambiente global crítico — usar qualquer previsão com máxima cautela."
        elif regime == "atencao" or k_estado == "atencao":
            risco_global = "atenção"
            emoji_global = "🟡"
            texto_global = "Ambiente global em atenção — turbulência moderada; valide bem cada passo."
        elif regime == "estavel" and k_estado == "estavel":
            risco_global = "estável"
            emoji_global = "🟢"
            texto_global = "Ambiente global estável — regime favorável para previsões ULTRA."
        else:
            risco_global = "indefinido"
            emoji_global = "⚪"
            texto_global = "Ambiente global pouco definido — dados insuficientes ou regime híbrido."

        st.markdown("### 🌐 Síntese Global de Risco")
        st.markdown(f"{emoji_global} **Nível global de risco:** `{risco_global}`")
        st.write(texto_global)

        registrar_evento(
            f"Monitor de risco consultado — barômetro={regime}, k*={k_estado}, risco_global={risco_global}."
        )


# ------------------------------------------------------------
# Painel: 📊 IDX / IPF / IPO ULTRA
# ------------------------------------------------------------

if painel == "📊 IDX / IPF / IPO ULTRA":
    st.markdown("## 📊 IDX / IPF / IPO ULTRA")

    df = st.session_state.get("df", None)
    meta = st.session_state.get("meta_cols", None)

    if df is None or df.empty or meta is None:
        st.warning("Carregue o histórico no painel '📥 Histórico — Entrada (FLEX)' antes.")
    else:
        idx_max = int(df["idx"].max())
        idx_default = st.session_state.get("ultimo_idx_alvo", idx_max)

        col_sel1, col_sel2 = st.columns([2, 1])
        with col_sel1:
            idx_alvo = st.number_input(
                "Selecione o índice alvo para calcular os núcleos (1 = primeira série):",
                min_value=1,
                max_value=idx_max,
                value=int(idx_default),
                step=1,
            )
        with col_sel2:
            st.write("")
            st.write("")
            if st.button("Usar última série como alvo"):
                idx_alvo = idx_max

        resultado = calcular_nucleos_idx_ipf_ipo_ultra(df, int(idx_alvo), meta)
        if not resultado["ok"]:
            st.error(resultado["msg"])
        else:
            st.session_state["ultimo_idx_alvo"] = int(idx_alvo)

            st.markdown("### 🎯 Série alvo (contexto imediato)")
            st.code(
                f"ID {df.loc[df['idx'] == resultado['idx_alvo'], 'id'].iloc[0]} | "
                f"Passageiros: {resultado['serie_alvo']}",
                language="text",
            )

            st.markdown("### 📦 Janela de histórico usada")
            col_j1, col_j2, col_j3 = st.columns(3)
            with col_j1:
                st.metric("Início da janela", resultado["janela_inicio"])
            with col_j2:
                st.metric("Fim da janela", resultado["janela_fim"])
            with col_j3:
                st.metric("Tamanho da janela", resultado["janela_tamanho"])

            st.markdown("### 🧠 Núcleos IDX / IPF / IPO ULTRA")
            col_n1, col_n2 = st.columns(2)

            with col_n1:
                st.markdown("#### IDX ULTRA (média ponderada dinâmica)")
                st.code(" ".join(str(x) for x in resultado["idx_ultra"]), language="text")

                st.markdown("#### IPF ULTRA (mediana robusta)")
                st.code(" ".join(str(x) for x in resultado["ipf_ultra"]), language="text")

            with col_n2:
                st.markdown("#### IPO ORIGINAL (média simples)")
                st.code(" ".join(str(x) for x in resultado["ipo_original"]), language="text")

                st.markdown("#### IPO ULTRA (refinada anti-sesgo)")
                st.code(" ".join(str(x) for x in resultado["ipo_ultra"]), language="text")

            registrar_evento(
                f"IDX/IPF/IPO ULTRA calculado para idx_alvo={int(idx_alvo)} "
                f"com janela [{resultado['janela_inicio']}, {resultado['janela_fim']}]."
            )


# ------------------------------------------------------------
# Painel: 🔍 Pipeline V14-FLEX ULTRA
# (camada de orquestração de alto nível — sem S6/Monte Carlo ainda)
# ------------------------------------------------------------

if painel == "🔍 Pipeline V14-FLEX ULTRA":
    st.markdown("## 🔍 Pipeline V14-FLEX ULTRA — Execução Base")

    df = st.session_state.get("df", None)
    meta = st.session_state.get("meta_cols", None)

    if df is None or df.empty or meta is None:
        st.warning("Carregue o histórico no painel '📥 Histórico — Entrada (FLEX)' antes.")
        st.stop()

    idx_max = int(df["idx"].max())
    idx_default = st.session_state.get("ultimo_idx_alvo", idx_max)

    st.markdown("### 🎯 Seleção da série alvo")

    col_selA, col_selB = st.columns([2, 1])
    with col_selA:
        modo_idx = st.radio(
            "Como deseja escolher o índice alvo?",
            ["Usar última série do histórico", "Escolher manualmente"],
            horizontal=True,
        )

    if modo_idx == "Usar última série do histórico":
        idx_alvo = idx_max
    else:
        with col_selB:
            idx_alvo = st.number_input(
                "Índice alvo:",
                min_value=1,
                max_value=idx_max,
                value=int(idx_default),
                step=1,
            )

    idx_alvo = int(idx_alvo)
    st.session_state["ultimo_idx_alvo"] = idx_alvo

    st.markdown("#### 📌 Série alvo selecionada")
    st.code(resumo_rapido_serie(df, idx_alvo), language="text")

    st.markdown("---")

    # ---------- Camada 1: Diagnóstico de risco (Barômetro + k*) ----------
    st.markdown("### 1️⃣ Diagnóstico de risco — Barômetro + k*")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        bar = calcular_barometro_ultra(df)
        if bar["estado"] == "estavel":
            emoji = "🟢"
        elif bar["estado"] == "atencao":
            emoji = "🟡"
        elif bar["estado"] == "critico":
            emoji = "🔴"
        else:
            emoji = "⚪"
        st.markdown(f"{emoji} **Barômetro:** `{bar['estado']}`")
        st.write(bar["texto"])

    with col_r2:
        kinfo = calcular_k_star_ultra(df)
        if kinfo["estado"] == "estavel":
            emoji = "🟢"
        elif kinfo["estado"] == "atencao":
            emoji = "🟡"
        elif kinfo["estado"] == "critico":
            emoji = "🔴"
        else:
            emoji = "⚪"
        st.markdown(f"{emoji} **k\\*:** `{kinfo['estado']}`")
        st.write(kinfo["texto"])

    st.markdown("---")

    # ---------- Camada 2: Núcleos IDX / IPF / IPO ----------
    st.markdown("### 2️⃣ Núcleos IDX / IPF / IPO ULTRA (base para previsão)")

    nucleos = calcular_nucleos_idx_ipf_ipo_ultra(df, idx_alvo, meta)
    if not nucleos["ok"]:
        st.error(nucleos["msg"])
        st.stop()

    col_nA, col_nB = st.columns(2)
    with col_nA:
        st.markdown("#### IDX ULTRA")
        st.code(" ".join(str(x) for x in nucleos["idx_ultra"]), language="text")

        st.markdown("#### IPF ULTRA")
        st.code(" ".join(str(x) for x in nucleos["ipf_ultra"]), language="text")

    with col_nB:
        st.markdown("#### IPO ORIGINAL")
        st.code(" ".join(str(x) for x in nucleos["ipo_original"]), language="text")

        st.markdown("#### IPO ULTRA (refinada)")
        st.code(" ".join(str(x) for x in nucleos["ipo_ultra"]), language="text")

    st.info(
        f"Janela usada: índices de `{nucleos['janela_inicio']}` até `{nucleos['janela_fim']}` "
        f"(tamanho `{nucleos['janela_tamanho']}` séries)."
    )

    # ---------- Camada 3: Pré-síntese para previsão ULTRA ----------
    st.markdown("---")
    st.markdown("### 3️⃣ Pré-síntese da base de previsão ULTRA")

    st.write(
        """
Nesta camada, o app consolida os núcleos IDX / IPF / IPO como
**ponto de partida** para o motor ULTRA (S6 Profundo, Micro-Leque,
Monte Carlo, QDS / Backtest), que será aplicado nas próximas camadas.
"""
    )

    # Para manter traço explícito no estado, armazenamos núcleos atuais
    st.session_state["nucleos_ultra_base"] = nucleos

    st.success(
        "Base ULTRA para previsão consolidada. As próximas camadas (S6, Micro-Leque, "
        "Monte Carlo, QDS / Backtest e Modo TURBO++) usarão estes núcleos."
    )

    registrar_evento(
        f"Pipeline V14-FLEX ULTRA executado para idx_alvo={idx_alvo} "
        f"com janela [{nucleos['janela_inicio']}, {nucleos['janela_fim']}]."
    )

# ============================================================
# FIM DA PARTE 2/4
# Próxima parte: S6 Profundo, Micro-Leque ULTRA, Monte Carlo
# Profundo, QDS REAL & Backtest REAL.
# ============================================================
# ============================================================
# PARTE 3/4
# S6 Profundo, Micro-Leque ULTRA, Monte Carlo Profundo ULTRA,
# QDS REAL & Backtest REAL
# ============================================================

# ------------------------------------------------------------
# Micro-Leque ULTRA — geração de séries candidatas
# ------------------------------------------------------------

def gerar_micro_leque_ultra(
    df: pd.DataFrame,
    idx_alvo: int,
    meta: Dict[str, Any],
    nucleos: Dict[str, Any],
    n_por_nucleo: int = 40,
) -> pd.DataFrame:
    """
    Gera o Micro-Leque ULTRA a partir dos núcleos:
        - IDX ULTRA
        - IPF ULTRA
        - IPO ULTRA

    Estratégia:
        - Usa cada núcleo como base.
        - Cria variações leves (perturbações) dentro do intervalo global da estrada.
        - Garante diversidade mantendo séries distintas.
    """
    if df is None or df.empty or meta is None or not nucleos.get("ok", False):
        return pd.DataFrame(columns=["source", "series"])

    cols_pass = meta["cols_passageiros"]
    n_pass = meta["n_passageiros"]

    # Faixa global da estrada (mín e máx por passageiro)
    M = df[cols_pass].astype(float).values
    global_min = int(np.nanmin(M))
    global_max = int(np.nanmax(M))

    def limitar(x: int) -> int:
        return int(max(global_min, min(global_max, x)))

    bases = [
        ("IDX_ULTRA", nucleos["idx_ultra"]),
        ("IPF_ULTRA", nucleos["ipf_ultra"]),
        ("IPO_ULTRA", nucleos["ipo_ultra"]),
    ]

    candidatos = []
    conjunto_series = set()

    rng = np.random.default_rng(seed=123456 + idx_alvo)

    for nome, base in bases:
        base_arr = np.array(base, dtype=int)
        for _ in range(n_por_nucleo):
            # Número de posições a perturbar (0–2)
            n_pert = rng.integers(0, min(3, n_pass + 1))
            serie = base_arr.copy()

            if n_pert > 0:
                posicoes = rng.choice(n_pass, size=n_pert, replace=False)
                for pos in posicoes:
                    delta = int(rng.integers(-3, 4))  # [-3, 3]
                    serie[pos] = limitar(serie[pos] + delta)

            chave = tuple(int(x) for x in serie)
            if chave not in conjunto_series:
                conjunto_series.add(chave)
                candidatos.append({"source": nome, "series": list(chave)})

        # Inclui a própria base explicitamente
        chave_base = tuple(int(x) for x in base_arr)
        if chave_base not in conjunto_series:
            conjunto_series.add(chave_base)
            candidatos.append({"source": f"{nome}_BASE", "series": list(chave_base)})

    df_micro = pd.DataFrame(candidatos)
    return df_micro


# ------------------------------------------------------------
# S6 Profundo ULTRA — ranqueamento estrutural
# ------------------------------------------------------------

def s6_profundo_ultra(
    df: pd.DataFrame,
    idx_alvo: int,
    meta: Dict[str, Any],
    nucleos: Dict[str, Any],
    n_por_nucleo: int = 40,
) -> pd.DataFrame:
    """
    S6 Profundo ULTRA:
        - Usa o Micro-Leque ULTRA.
        - Calcula score estrutural por série candidata.
        - Ordena do mais forte ao mais fraco.

    Score:
        - Similaridade com IPO ULTRA (estrutura).
        - Afinidade com o k local (consistência de regime).
    """
    if df is None or df.empty or meta is None or not nucleos.get("ok", False):
        return pd.DataFrame(columns=["rank", "source", "series", "score_total"])

    cols_pass = meta["cols_passageiros"]

    df_micro = gerar_micro_leque_ultra(df, idx_alvo, meta, nucleos, n_por_nucleo)
    if df_micro.empty:
        return pd.DataFrame(columns=["rank", "source", "series", "score_total"])

    ipo_ultra = np.array(nucleos["ipo_ultra"], dtype=float)

    # k atual e k da janela
    k_vals = df["k"].astype(float).values
    idx_min = int(df["idx"].min())
    pos_alvo = idx_alvo - idx_min
    if 0 <= pos_alvo < len(k_vals):
        k_atual = k_vals[pos_alvo]
    else:
        k_atual = float(np.nan)

    if pos_alvo > 0:
        k_janela = k_vals[max(0, pos_alvo - 40):pos_alvo]
        k_med = float(np.nanmean(k_janela)) if len(k_janela) > 0 else float(np.nan)
    else:
        k_med = float(np.nan)

    # Escala de normalização para afinidade de k
    escala_k = max(1.0, np.nanmax(np.abs(k_vals)) if len(k_vals) > 0 else 1.0)

    scores = []
    for _, row in df_micro.iterrows():
        serie = np.array(row["series"], dtype=float)

        # Similaridade estrutural com IPO ULTRA (distância euclidiana inversa)
        dist = float(np.linalg.norm(serie - ipo_ultra))
        sim_estrutura = 1.0 / (1.0 + dist)

        # Afinidade de k (se possível)
        if not (np.isnan(k_atual) or np.isnan(k_med)):
            delta_k = abs(k_atual - k_med)
            afin_k = 1.0 - min(delta_k / (escala_k + 1e-9), 1.0)
        else:
            afin_k = 0.5  # neutro

        # Combinação (peso maior na estrutura)
        score_total = 0.7 * sim_estrutura + 0.3 * afin_k

        scores.append(
            {
                "source": row["source"],
                "series": row["series"],
                "score_estrutura": sim_estrutura,
                "score_k": afin_k,
                "score_total": score_total,
            }
        )

    df_scores = pd.DataFrame(scores)
    df_scores = df_scores.sort_values(by="score_total", ascending=False).reset_index(drop=True)
    df_scores["rank"] = df_scores.index + 1

    # Guarda no estado
    st.session_state["s6_ultra_resultado"] = {
        "idx_alvo": idx_alvo,
        "nucleos": nucleos,
        "tabela": df_scores,
    }

    return df_scores


# ------------------------------------------------------------
# Monte Carlo Profundo ULTRA — robustez das séries
# ------------------------------------------------------------

def monte_carlo_profundo_ultra(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    s6_df: pd.DataFrame,
    n_simulacoes: int = 500,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Monte Carlo Profundo ULTRA:
        - Avalia a robustez das top séries de S6 Profundo.
        - Simula estradas alternativas com base no histórico real.
        - Mede a quantidade esperada de acertos (interseção de passageiros).

    Saída:
        DataFrame com colunas:
            - rank_s6, series, media_acertos, prob_ge4, prob_ge5, prob_ge6
    """
    if df is None or df.empty or meta is None or s6_df is None or s6_df.empty:
        return pd.DataFrame(
            columns=[
                "rank_s6",
                "series",
                "media_acertos",
                "prob_ge4",
                "prob_ge5",
                "prob_ge6",
            ]
        )

    cols_pass = meta["cols_passageiros"]
    M = df[cols_pass].astype(int).values
    n_series_hist = M.shape[0]

    # Top N do S6
    s6_top = s6_df.sort_values(by="score_total", ascending=False).head(top_n).copy()

    rng = np.random.default_rng(seed=987654)

    resultados = []
    for _, row in s6_top.iterrows():
        serie = [int(x) for x in row["series"]]
        conj_cand = set(serie)

        acertos_list = []

        for _ in range(n_simulacoes):
            idx_aleatorio = int(rng.integers(0, n_series_hist))
            real = M[idx_aleatorio, :]
            conj_real = set(int(x) for x in real)
            acertos = len(conj_cand.intersection(conj_real))
            acertos_list.append(acertos)

        acertos_arr = np.array(acertos_list)
        media_acertos = float(acertos_arr.mean()) if len(acertos_arr) > 0 else 0.0
        prob_ge4 = float(np.mean(acertos_arr >= 4)) if len(acertos_arr) > 0 else 0.0
        prob_ge5 = float(np.mean(acertos_arr >= 5)) if len(acertos_arr) > 0 else 0.0
        prob_ge6 = float(np.mean(acertos_arr >= 6)) if len(acertos_arr) > 0 else 0.0

        resultados.append(
            {
                "rank_s6": int(row["rank"]),
                "series": serie,
                "media_acertos": media_acertos,
                "prob_ge4": prob_ge4,
                "prob_ge5": prob_ge5,
                "prob_ge6": prob_ge6,
            }
        )

    mc_df = pd.DataFrame(resultados)
    mc_df = mc_df.sort_values(by=["prob_ge6", "prob_ge5", "prob_ge4", "media_acertos"], ascending=False).reset_index(drop=True)

    # Guarda no estado
    st.session_state["monte_carlo_ultra"] = mc_df

    return mc_df


# ------------------------------------------------------------
# QDS REAL + Backtest REAL
# ------------------------------------------------------------

def calcular_qds_real_e_backtest(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    n_testes_max: int = 60,
    janela_max: int = 40,
) -> Dict[str, Any]:
    """
    QDS REAL + Backtest REAL ULTRA (simplificado):

    Para cada índice alvo i, testa:
        - Usa histórico até i-1.
        - Calcula núcleos IDX/IPF/IPO ULTRA.
        - Usa IPO ULTRA como previsão da próxima série (i+1).
        - Compara com a série real (i+1) e mede acertos de passageiros.

    Gera:
        - Distribuição dos acertos.
        - Médias, percentuais com >=4, >=5, =6 acertos.
        - QDS em % como proporção média de acertos / n_passageiros.
    """
    if df is None or df.empty or meta is None:
        return {
            "ok": False,
            "msg": "Histórico ou metadados indisponíveis.",
        }

    idx_min = int(df["idx"].min())
    idx_max = int(df["idx"].max())
    cols_pass = meta["cols_passageiros"]
    n_pass = meta["n_passageiros"]

    resultados_acertos = []
    idxs_testados = []

    # Limita quantos testes para não estourar performance
    # Vamos de trás para frente (regime mais recente)
    candidatos = list(range(idx_max - 1, idx_min + janela_max, -1))  # até idx_max-1
    candidatos = candidatos[:n_testes_max]
    candidatos = sorted(candidatos)

    for idx_alvo in candidatos:
        nucleos = calcular_nucleos_idx_ipf_ipo_ultra(df, idx_alvo, meta, janela_max=janela_max)
        if not nucleos.get("ok", False):
            continue

        idx_next = idx_alvo + 1
        if idx_next > idx_max:
            continue

        # Previsão simplificada: IPO ULTRA como "previsão de passageiros"
        prev = set(int(x) for x in nucleos["ipo_ultra"])

        real_row = df.loc[df["idx"] == idx_next, cols_pass]
        if real_row.empty:
            continue
        real = set(int(x) for x in real_row.astype(int).iloc[0].tolist())

        acertos = len(prev.intersection(real))
        resultados_acertos.append(acertos)
        idxs_testados.append(idx_alvo)

    if not resultados_acertos:
        return {
            "ok": False,
            "msg": "Não foi possível executar o Backtest REAL com os parâmetros atuais.",
        }

    arr = np.array(resultados_acertos)
    media_acertos = float(arr.mean())
    mediana_acertos = float(np.median(arr))
    melhor_acerto = int(arr.max())
    p_ge4 = float(np.mean(arr >= 4))
    p_ge5 = float(np.mean(arr >= 5))
    p_ge6 = float(np.mean(arr >= 6))

    # QDS como proporção média de acertos sobre n_passageiros
    qds_pct = float(100.0 * media_acertos / n_pass)

    resumo = {
        "ok": True,
        "msg": "",
        "n_testes": len(resultados_acertos),
        "indices_testados": idxs_testados,
        "acertos": resultados_acertos,
        "media_acertos": media_acertos,
        "mediana_acertos": mediana_acertos,
        "melhor_acerto": melhor_acerto,
        "p_ge4": p_ge4,
        "p_ge5": p_ge5,
        "p_ge6": p_ge6,
        "qds_pct": qds_pct,
        "n_passageiros": n_pass,
    }

    # Guarda em cache
    st.session_state["qds_cache"]["default"] = resumo

    return resumo


# ------------------------------------------------------------
# Painel: 🧬 S6 Profundo & Micro-Leque ULTRA
# ------------------------------------------------------------

if painel == "🧬 S6 Profundo & Micro-Leque ULTRA":
    st.markdown("## 🧬 S6 Profundo & Micro-Leque ULTRA")

    df = st.session_state.get("df", None)
    meta = st.session_state.get("meta_cols", None)

    if df is None or df.empty or meta is None:
        st.warning("Carregue o histórico no painel '📥 Histórico — Entrada (FLEX)' antes.")
    else:
        idx_max = int(df["idx"].max())
        idx_default = st.session_state.get("ultimo_idx_alvo", idx_max)

        st.markdown("### 🎯 Seleção da série alvo")

        col_sel1, col_sel2 = st.columns([2, 1])
        with col_sel1:
            idx_alvo = st.number_input(
                "Índice alvo para S6 Profundo:",
                min_value=1,
                max_value=idx_max,
                value=int(idx_default),
                step=1,
            )
        with col_sel2:
            n_por_nucleo = st.slider(
                "Séries por núcleo no Micro-Leque:",
                min_value=10,
                max_value=80,
                value=40,
                step=5,
            )

        idx_alvo = int(idx_alvo)
        st.session_state["ultimo_idx_alvo"] = idx_alvo

        st.markdown("#### 📌 Série alvo")
        st.code(resumo_rapido_serie(df, idx_alvo), language="text")

        nucleos = calcular_nucleos_idx_ipf_ipo_ultra(df, idx_alvo, meta)
        if not nucleos.get("ok", False):
            st.error(nucleos["msg"])
        else:
            st.markdown("### 🧪 Execução do S6 Profundo ULTRA")

            if st.button("Rodar S6 Profundo ULTRA agora"):
                tabela_s6 = s6_profundo_ultra(
                    df,
                    idx_alvo,
                    meta,
                    nucleos,
                    n_por_nucleo=n_por_nucleo,
                )

                if tabela_s6.empty:
                    st.error("S6 Profundo não gerou séries. Verifique o histórico.")
                else:
                    st.success(
                        f"S6 Profundo ULTRA gerou {len(tabela_s6)} séries candidatas."
                    )

                    st.markdown("### 🏆 Top 15 séries do S6 Profundo ULTRA")
                    top15 = tabela_s6.head(15)
                    linhas = []
                    for _, row in top15.iterrows():
                        linhas.append(
                            f"#{int(row['rank']):02d} | "
                            f"{' '.join(str(x) for x in row['series'])} | "
                            f"score={row['score_total']:.4f}"
                        )
                    st.code("\n".join(linhas), language="text")

                    st.markdown("#### 🔬 Tabela completa S6 Profundo ULTRA")
                    st.dataframe(tabela_s6)

                    registrar_evento(
                        f"S6 Profundo ULTRA executado para idx_alvo={idx_alvo} "
                        f"com {len(tabela_s6)} séries candidatas."
                    )
            else:
                st.info("Configure os parâmetros e clique em **Rodar S6 Profundo ULTRA agora**.")


# ------------------------------------------------------------
# Painel: 🎲 Monte Carlo Profundo ULTRA
# ------------------------------------------------------------

if painel == "🎲 Monte Carlo Profundo ULTRA":
    st.markdown("## 🎲 Monte Carlo Profundo ULTRA")

    df = st.session_state.get("df", None)
    meta = st.session_state.get("meta_cols", None)
    s6_res = st.session_state.get("s6_ultra_resultado", None)

    if df is None or df.empty or meta is None:
        st.warning("Carregue o histórico no painel '📥 Histórico — Entrada (FLEX)' antes.")
    else:
        if not s6_res or "tabela" not in s6_res or s6_res["tabela"].empty:
            st.warning(
                "Nenhum resultado de S6 Profundo ULTRA encontrado. "
                "Execute o painel '🧬 S6 Profundo & Micro-Leque ULTRA' primeiro."
            )
        else:
            tabela_s6 = s6_res["tabela"]
            idx_alvo = s6_res["idx_alvo"]

            st.markdown(
                f"Último S6 Profundo foi executado para **idx_alvo = {idx_alvo}** "
                f"com {len(tabela_s6)} séries candidatas."
            )

            col_cfg1, col_cfg2 = st.columns(2)
            with col_cfg1:
                n_sim = st.slider(
                    "Número de simulações Monte Carlo:",
                    min_value=100,
                    max_value=2000,
                    value=500,
                    step=100,
                )
            with col_cfg2:
                top_n = st.slider(
                    "Top N séries do S6 a avaliar:",
                    min_value=5,
                    max_value=50,
                    value=20,
                    step=5,
                )

            if st.button("Rodar Monte Carlo Profundo ULTRA"):
                mc_df = monte_carlo_profundo_ultra(
                    df,
                    meta,
                    tabela_s6,
                    n_simulacoes=n_sim,
                    top_n=top_n,
                )

                if mc_df.empty:
                    st.error("Monte Carlo não produziu resultados.")
                else:
                    st.success(
                        f"Monte Carlo Profundo ULTRA executado para Top {top_n} séries de S6."
                    )

                    st.markdown("### 🏆 Séries mais robustas no Monte Carlo")
                    linhas = []
                    for i, row in mc_df.head(15).iterrows():
                        linhas.append(
                            f"S6#{int(row['rank_s6']):02d} | "
                            f"{' '.join(str(x) for x in row['series'])} | "
                            f"hits̄={row['media_acertos']:.3f} | "
                            f"P(≥4)={row['prob_ge4']:.3f} | "
                            f"P(≥5)={row['prob_ge5']:.3f} | "
                            f"P(6)={row['prob_ge6']:.3f}"
                        )
                    st.code("\n".join(linhas), language="text")

                    st.markdown("#### 🔬 Tabela completa Monte Carlo ULTRA")
                    st.dataframe(mc_df)

                    registrar_evento(
                        f"Monte Carlo Profundo ULTRA executado (n_sim={n_sim}, top_n={top_n})."
                    )
            else:
                st.info("Ajuste os parâmetros e clique em **Rodar Monte Carlo Profundo ULTRA**.")


# ------------------------------------------------------------
# Painel: 🧪 QDS REAL & Backtest REAL
# ------------------------------------------------------------

if painel == "🧪 QDS REAL & Backtest REAL":
    st.markdown("## 🧪 QDS REAL & Backtest REAL")

    df = st.session_state.get("df", None)
    meta = st.session_state.get("meta_cols", None)

    if df is None or df.empty or meta is None:
        st.warning("Carregue o histórico no painel '📥 Histórico — Entrada (FLEX)' antes.")
    else:
        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            n_testes_max = st.slider(
                "Máximo de pontos de Backtest:",
                min_value=20,
                max_value=150,
                value=60,
                step=10,
            )
        with col_cfg2:
            janela_max = st.slider(
                "Janela máxima de histórico por teste:",
                min_value=20,
                max_value=80,
                value=40,
                step=5,
            )

        if st.button("Calcular QDS REAL + Backtest REAL agora"):
            resultado = calcular_qds_real_e_backtest(
                df,
                meta,
                n_testes_max=n_testes_max,
                janela_max=janela_max,
            )

            if not resultado.get("ok", False):
                st.error(resultado["msg"])
            else:
                st.success(
                    f"Backtest REAL executado com {resultado['n_testes']} pontos de teste."
                )

                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric(
                        "QDS REAL (%)",
                        f"{resultado['qds_pct']:.2f}%",
                        help="Qualidade Dinâmica de Série: proporção média de acertos em relação ao número de passageiros.",
                    )
                with col_m2:
                    st.metric(
                        "Média de acertos",
                        f"{resultado['media_acertos']:.3f}",
                    )
                with col_m3:
                    st.metric(
                        "Melhor acerto (máx)",
                        f"{resultado['melhor_acerto']}",
                    )

                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    st.metric(
                        "P(≥4 acertos)",
                        f"{resultado['p_ge4']*100:.1f}%",
                    )
                with col_p2:
                    st.metric(
                        "P(≥5 acertos)",
                        f"{resultado['p_ge5']*100:.1f}%",
                    )
                with col_p3:
                    st.metric(
                        "P(6 acertos)",
                        f"{resultado['p_ge6']*100:.1f}%",
                    )

                st.markdown("### Distribuição dos acertos no Backtest REAL")
                linhas = []
                for idx_alvo, acertos in zip(
                    resultado["indices_testados"], resultado["acertos"]
                ):
                    linhas.append(f"idx_alvo={idx_alvo} → acertos={acertos}")
                st.code("\n".join(linhas), language="text")

                registrar_evento(
                    f"QDS REAL + Backtest REAL executados: QDS={resultado['qds_pct']:.2f}%, "
                    f"media_acertos={resultado['media_acertos']:.3f}."
                )
        else:
            cache = st.session_state.get("qds_cache", {}).get("default")
            if cache and cache.get("ok", False):
                st.info(
                    f"Último QDS REAL calculado: {cache['qds_pct']:.2f}% "
                    f"com {cache['n_testes']} pontos de teste."
                )
            else:
                st.info(
                    "Configure os parâmetros e clique em "
                    "**Calcular QDS REAL + Backtest REAL agora**."
                )

# ============================================================
# FIM DA PARTE 3/4
# Próxima parte: Replay LIGHT, Replay ULTRA (horizonte ajustável)
# e Modo TURBO++ ULTRA (integração final de previsão).
# ============================================================
# ============================================================
# PARTE 4/4
# Replay LIGHT, Replay ULTRA com horizonte ajustável,
# Modo TURBO++ ULTRA — Motor Final Consolidado
# ============================================================

# ------------------------------------------------------------
# Função auxiliar: rodar previsão ULTRA completa (single)
# Envolve:
#   - Cálculo de núcleos ULTRA
#   - S6 Profundo
#   - Monte Carlo Profundo
#   - Seleção final (melhor série)
# ------------------------------------------------------------

def previsao_ultra_completa(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    idx_alvo: int,
    n_s6_por_nucleo: int = 40,
    n_mc_sim: int = 400,
    top_n_mc: int = 20,
) -> Dict[str, Any]:
    """
    Executa o pipeline ULTRA completo para UMA série alvo.
    Retorna:
        {
            "ok": bool,
            "idx_alvo": int,
            "serie_alvo": [...],
            "s6_top": DataFrame,
            "mc_top": DataFrame,
            "previsao_final": [...],
            "mensagem": str
        }
    """
    if df is None or df.empty or meta is None:
        return {"ok": False, "mensagem": "Histórico indisponível."}

    nucleos = calcular_nucleos_idx_ipf_ipo_ultra(df, idx_alvo, meta)
    if not nucleos.get("ok", False):
        return {"ok": False, "mensagem": nucleos["msg"]}

    # S6 Profundo
    s6_df = s6_profundo_ultra(
        df, idx_alvo, meta, nucleos,
        n_por_nucleo=n_s6_por_nucleo,
    )
    if s6_df.empty:
        return {"ok": False, "mensagem": "S6 Profundo não produziu séries."}

    # Monte Carlo Profundo
    mc_df = monte_carlo_profundo_ultra(
        df, meta, s6_df,
        n_simulacoes=n_mc_sim,
        top_n=top_n_mc,
    )
    if mc_df.empty:
        return {"ok": False, "mensagem": "Monte Carlo não produziu resultados."}

    # Seleção final = maior prob_ge6 (empate → prob_ge5 → prob_ge4 → média)
    melhor = mc_df.iloc[0]
    previsao_final = melhor["series"]

    return {
        "ok": True,
        "idx_alvo": idx_alvo,
        "serie_alvo": extrair_passageiros_linha(df, idx_alvo),
        "s6_top": s6_df,
        "mc_top": mc_df,
        "previsao_final": previsao_final,
        "mensagem": "",
    }


# ============================================================
# 📅 Replay LIGHT
# ============================================================

if painel == "📅 Replay LIGHT":
    st.markdown("## 📅 Replay LIGHT — Execução rápida")

    df = st.session_state.get("df", None)
    meta = st.session_state.get("meta_cols", None)

    if df is None or df.empty or meta is None:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    idx_max = int(df["idx"].max())
    st.info("O Replay LIGHT executa previsões ULTRA sem Monte Carlo, para alta velocidade.")

    # Parâmetros LIGHT
    n_por_nucleo = st.slider(
        "Séries por núcleo (Micro-Leque LIGHT):",
        min_value=5,
        max_value=40,
        value=20,
        step=5,
    )

    horizonte = st.slider(
        "Número de séries para processar no Replay LIGHT:",
        min_value=5,
        max_value=min(80, idx_max - 1),
        value=20,
        step=1,
    )

    start_idx = idx_max - horizonte + 1
    st.write(f"Executando Replay LIGHT para índices de `{start_idx}` até `{idx_max}`.")

    if st.button("Rodar Replay LIGHT agora"):
        resultados = []
        for idx_alvo in range(start_idx, idx_max + 1):
            nucleos = calcular_nucleos_idx_ipf_ipo_ultra(df, idx_alvo, meta)
            if not nucleos.get("ok", False):
                continue

            s6_df = s6_profundo_ultra(
                df, idx_alvo, meta, nucleos,
                n_por_nucleo=n_por_nucleo,
            )
            if s6_df.empty:
                continue

            # Seleção LIGHT: melhor pelo score_total (S6)
            melhor = s6_df.iloc[0]
            resultados.append({
                "idx": idx_alvo,
                "serie_alvo": extrair_passageiros_linha(df, idx_alvo),
                "previsao": melhor["series"],
                "score": melhor["score_total"],
            })

        if not resultados:
            st.error("Replay LIGHT não gerou resultados.")
        else:
            st.success(f"Replay LIGHT executado para {len(resultados)} séries.")
            linhas = []
            for r in resultados:
                linhas.append(
                    f"idx={r['idx']} | alvo={r['serie_alvo']} | "
                    f"prev={r['previsao']} | score={r['score']:.4f}"
                )
            st.code("\n".join(linhas), language="text")

            st.session_state["replay_light"] = resultados

            registrar_evento(
                f"Replay LIGHT executado para {len(resultados)} séries."
            )
    else:
        st.info("Configure e clique em **Rodar Replay LIGHT agora**.")


# ============================================================
# 📅 Replay ULTRA (Horizonte Ajustável)
# ============================================================

if painel == "📅 Replay ULTRA (Horizonte Ajustável)":
    st.markdown("## 📅 Replay ULTRA — Horizonte Ajustável")

    df = st.session_state.get("df", None)
    meta = st.session_state.get("meta_cols", None)

    if df is None or df.empty or meta is None:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    idx_max = int(df["idx"].max())

    st.info(
        """
O Replay ULTRA executa o **pipeline completo**:
IDX/IPF/IPO → S6 Profundo → Monte Carlo Profundo → Previsão final.

Agora com **horizonte ajustável**, para limitar quantas séries serão processadas
(para evitar sobrecarga).
"""
    )

    horizonte = st.slider(
        "Quantas séries deseja processar?",
        min_value=5,
        max_value=min(100, idx_max - 1),
        value=20,
        step=1,
    )

    start_idx = idx_max - horizonte + 1
    st.write(f"Horizonte selecionado: `{start_idx}` → `{idx_max}`.")

    n_por_nucleo = st.slider("Séries por núcleo (Micro-Leque):", 20, 80, 40, 5)
    n_sim = st.slider("Simulações Monte Carlo:", 200, 2000, 800, 200)
    top_n = st.slider("Top-N Monte Carlo:", 10, 50, 20, 5)

    if st.button("Rodar Replay ULTRA agora"):
        resultados = []
        total = idx_max - start_idx + 1
        barra = st.progress(0)

        for i, idx_alvo in enumerate(range(start_idx, idx_max + 1)):
            barra.progress((i + 1) / total)

            res = previsao_ultra_completa(
                df, meta, idx_alvo,
                n_s6_por_nucleo=n_por_nucleo,
                n_mc_sim=n_sim,
                top_n_mc=top_n,
            )
            if res["ok"]:
                resultados.append(res)

        if not resultados:
            st.error("Replay ULTRA não produziu resultados.")
        else:
            st.success(f"Replay ULTRA completo — {len(resultados)} séries processadas.")

            linhas = []
            for r in resultados:
                linhas.append(
                    f"idx={r['idx_alvo']} | alvo={r['serie_alvo']} | "
                    f"prev={r['previsao_final']}"
                )
            st.code("\n".join(linhas), language="text")

            st.session_state["replay_ultra"] = resultados

            registrar_evento(
                f"Replay ULTRA executado com horizonte={horizonte}."
            )
    else:
        st.info("Configure e clique em **Rodar Replay ULTRA agora**.")


# ============================================================
# 🚀 Modo TURBO++ ULTRA — Previsão final unificada
# ============================================================

if painel == "🚀 Modo TURBO++ ULTRA":
    st.markdown("## 🚀 Modo TURBO++ ULTRA — Motor Final")

    df = st.session_state.get("df", None)
    meta = st.session_state.get("meta_cols", None)

    if df is None or df.empty or meta is None:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    idx_max = int(df["idx"].max())

    st.markdown("### 🎯 Série alvo (última)")
    st.code(resumo_rapido_serie(df, idx_max), language="text")

    st.markdown("---")

    st.markdown("### ⚙️ Parâmetros do motor TURBO++")

    n_por_nucleo = st.slider(
        "Micro-Leque ULTRA — séries por núcleo:",
        min_value=20,
        max_value=80,
        value=40,
        step=5,
    )

    n_sim = st.slider(
        "Monte Carlo Profundo — número de simulações:",
        min_value=300,
        max_value=3000,
        value=1000,
        step=100,
    )

    top_n = st.slider(
        "Top-N do Monte Carlo:",
        min_value=10,
        max_value=50,
        value=20,
        step=5,
    )

    if st.button("EXECUTAR PREVISÃO TURBO++ ULTRA"):
        st.info("Executando motor ULTRA…")

        res = previsao_ultra_completa(
            df,
            meta,
            idx_max,
            n_s6_por_nucleo=n_por_nucleo,
            n_mc_sim=n_sim,
            top_n_mc=top_n,
        )

        if not res["ok"]:
            st.error(res["mensagem"])
        else:
            st.success("Previsão TURBO++ gerada com sucesso!")

            st.markdown("### 🎯 **Previsão Final ULTRA**")
            st.code(" ".join(str(x) for x in res["previsao_final"]), language="text")

            # Interpretar risco global
            regime = st.session_state.get("regime_state", "desconhecido")
            k_est = st.session_state.get("k_star_estado", "desconhecido")

            if regime == "critico" or k_est == "critico":
                st.error("🔴 Ambiente crítico — máxima cautela.")
            elif regime == "atencao" or k_est == "atencao":
                st.warning("🟡 Ambiente moderado — atenção elevada.")
            else:
                st.success("🟢 Ambiente estável — previsão em regime favorável.")

            registrar_evento("Previsão TURBO++ executada.")

    else:
        st.info("Configure os parâmetros e clique em **EXECUTAR PREVISÃO TURBO++ ULTRA**.")


# ============================================================
# FIM DO ARQUIVO COMPLETO — Predict Cars V14-FLEX ULTRA REAL
# (TURBO++)
# ============================================================
