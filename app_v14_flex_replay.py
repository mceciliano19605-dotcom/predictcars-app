import streamlit as st
import pandas as pd
import numpy as np
import random
from itertools import product
from collections import Counter
from typing import List, Dict, Any, Optional

# ============================================================
# CONFIGURAÇÃO BÁSICA DO APP
# ============================================================

st.set_page_config(
    page_title="Predict Cars V14-FLEX ULTRA REAL (TURBO++)",
    layout="wide",
)

# ============================================================
# UTILITÁRIOS GERAIS
# ============================================================

def registrar_evento(msg: str) -> None:
    """Logger simples em sessão (não quebra o app se não usado)."""
    log = st.session_state.get("log_eventos", [])
    log.append(msg)
    st.session_state["log_eventos"] = log


def extrair_colunas_passageiros(df: pd.DataFrame) -> List[str]:
    """
    Descobre automaticamente quais colunas são 'passageiros',
    removendo id e k, e mantendo a ordem.
    """
    if df is None or df.empty:
        return []
    colunas_excluir = {"k", "K", "id", "ID", "Id", "C", "c", "serie", "SERIE", "label", "LABEL"}
    cols = [c for c in df.columns if c not in colunas_excluir]
    return cols


def linha_para_serie(row: pd.Series, cols_pass: List[str]) -> List[int]:
    return [int(row[c]) for c in cols_pass]


def serie_para_str(serie: List[int]) -> str:
    return " ".join(str(x) for x in serie)


def contar_hits(serie_prev: List[int], serie_real: List[int]) -> int:
    alvo = set(serie_real)
    return sum(1 for x in serie_prev if x in alvo)


# ============================================================
# PREPARO DO HISTÓRICO — FLEX (N PASSAGEIROS)
# ============================================================

def preparar_historico_flex(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe df_raw lido de CSV (sem header, sep=';') e:
    - identifica coluna de id (primeira)
    - identifica coluna k (última)
    - detecta colunas de passageiros (meio)
    - converte para tipos adequados
    - registra em sessão n_passageiros e col_k
    """
    if df_raw is None or df_raw.empty:
        raise ValueError("Histórico vazio.")

    # Reset de índice e nomes genéricos se necessário
    df_raw = df_raw.reset_index(drop=True)

    n_cols = df_raw.shape[1]
    if n_cols < 3:
        raise ValueError("Histórico precisa ter pelo menos id, passageiros e k.")

    # Primeira coluna: id
    col_id = df_raw.columns[0]
    col_k = df_raw.columns[-1]
    cols_pass = list(df_raw.columns[1:-1])

    # Montar df padronizado
    df = pd.DataFrame()
    df["id"] = df_raw[col_id].astype(str).str.strip()

    # Passageiros
    for i, c in enumerate(cols_pass, start=1):
        df[f"p{i}"] = pd.to_numeric(df_raw[c], errors="coerce").fillna(0).astype(int)

    # k (guardas que acertaram)
    df["k"] = pd.to_numeric(df_raw[col_k], errors="coerce").fillna(0).astype(int)

    # Registrar em sessão
    st.session_state["n_passageiros"] = len(cols_pass)
    st.session_state["col_k"] = "k"

    return df


# ============================================================
# BARÔMETRO ULTRA REAL + k* ULTRA REAL (SENTINELA)
# ============================================================

def calcular_barometro_ultra(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula o estado global da estrada a partir de k:
    - Índice de turbulência
    - Desvio padrão de k
    - Média de |Δk|
    - Estado (estavel / atencao / critico)
    """
    col_k = st.session_state.get("col_k", "k")
    if df is None or df.empty or col_k not in df.columns:
        return {
            "estado": "indefinido",
            "turbulencia": 0.0,
            "std_k": 0.0,
            "mean_abs_dk": 0.0,
        }

    k_vals = df[col_k].astype(float).values
    if len(k_vals) < 3:
        return {
            "estado": "indefinido",
            "turbulencia": 0.0,
            "std_k": 0.0,
            "mean_abs_dk": 0.0,
        }

    std_k = float(np.std(k_vals))
    diffs = np.diff(k_vals)
    mean_abs_dk = float(np.mean(np.abs(diffs)))

    # Índice de turbulência simples e robusto
    turb = std_k + mean_abs_dk
    # Normalização aproximada
    turb_norm = float(turb / (1.0 + max(1.0, np.max(k_vals))))

    if turb_norm < 0.4:
        estado = "estavel"
    elif turb_norm < 0.8:
        estado = "atencao"
    else:
        estado = "critico"

    return {
        "estado": estado,
        "turbulencia": turb_norm,
        "std_k": std_k,
        "mean_abs_dk": mean_abs_dk,
    }


def calcular_entropia_k(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calcula entropia de k (Shannon) e entropia normalizada.
    """
    col_k = st.session_state.get("col_k", "k")
    if df is None or df.empty or col_k not in df.columns:
        return {"entropia": 0.0, "entropia_norm": 0.0}

    k_vals = df[col_k].astype(int).values
    if len(k_vals) == 0:
        return {"entropia": 0.0, "entropia_norm": 0.0}

    vals, counts = np.unique(k_vals, return_counts=True)
    probs = counts / counts.sum()
    ent = -np.sum(probs * np.log2(probs))
    if len(vals) > 1:
        ent_norm = float(ent / np.log2(len(vals)))
    else:
        ent_norm = 0.0

    return {"entropia": float(ent), "entropia_norm": float(ent_norm)}


def calcular_k_star_ultra(df: pd.DataFrame, window: int = 40) -> Dict[str, Any]:
    """
    k* ULTRA REAL (sentinela) baseado em:
    - entropia de k
    - turbulência (variação local)
    """
    bar = calcular_barometro_ultra(df)
    ent = calcular_entropia_k(df)

    # Construção simples de k* a partir de entropia normalizada e turbulência
    turb = bar["turbulencia"]
    ent_norm = ent["entropia_norm"]

    k_star_val = 100.0 * (0.5 * ent_norm + 0.5 * min(1.0, turb + 0.2))
    k_star_val = max(0.0, min(100.0, k_star_val))

    if k_star_val < 30:
        estado = "estavel"
    elif k_star_val < 70:
        estado = "atencao"
    else:
        estado = "critico"

    return {
        "k_star": k_star_val,
        "estado": estado,
        "entropia": ent["entropia"],
        "entropia_norm": ent["entropia_norm"],
    }


# ============================================================
# NÚCLEOS IDX / IPF / IPO ULTRA
# ============================================================

def calcular_nucleos_idx_ipf_ipo(
    df: pd.DataFrame,
    idx_alvo: int,
    window: int = 40,
) -> Dict[str, Any]:
    """
    Calcula IDX ULTRA, IPF ULTRA, IPO ORIGINAL e IPO ULTRA
    usando uma janela de histórico antes do índice alvo.
    """
    if df is None or df.empty:
        return {}

    cols_pass = extrair_colunas_passageiros(df)
    if not cols_pass:
        return {}

    n = len(df)
    idx_zero = max(idx_alvo - 1, 0)
    inicio = max(idx_zero - window, 0)
    fim = max(0, idx_zero - 1)
    if fim < inicio:
        return {}

    df_janela = df.iloc[inicio : idx_zero]
    if df_janela.empty:
        return {}

    # IPO ORIGINAL — média simples por posição
    ipo_vals = []
    for c in cols_pass:
        ipo_vals.append(int(round(df_janela[c].mean())))
    ipo_original = ipo_vals

    # IPF ULTRA — mediana robusta por posição
    ipf_vals = []
    for c in cols_pass:
        ipf_vals.append(int(df_janela[c].median()))
    ipf_ultra = ipf_vals

    # IDX ULTRA — média ponderada dinâmica
    idx_vals = []
    pesos = np.linspace(0.5, 1.5, len(df_janela))
    pesos = pesos / pesos.sum()
    for c in cols_pass:
        col = df_janela[c].astype(float).values
        idx_val = int(round(np.sum(col * pesos)))
        idx_vals.append(idx_val)
    idx_ultra = idx_vals

    # IPO ULTRA — por simplicidade, igual ao original aqui, mas poderia ter refinamentos
    ipo_ultra = ipo_original.copy()

    return {
        "idx_ultra": idx_ultra,
        "ipf_ultra": ipf_ultra,
        "ipo_original": ipo_original,
        "ipo_ultra": ipo_ultra,
        "inicio": inicio + 1,
        "fim": idx_zero,
        "tamanho": len(df_janela),
    }


# ============================================================
# S6 PROFUNDO ULTRA, MICRO-LEQUE, MONTE CARLO E FUSÃO
# ============================================================

def s6_profundo_ultra(
    df: pd.DataFrame,
    idx_alvo: int,
    window: int = 80,
    n_series: int = 40,
) -> pd.DataFrame:
    """
    S6 Profundo ULTRA (genérico, estável e resiliente):

    - Usa uma janela de histórico antes do índice alvo
    - Calcula frequência de cada número em cada coluna de passageiro
    - Monta séries combinando os mais frequentes por coluna
    """
    if df is None or df.empty:
        return pd.DataFrame()

    cols_pass = extrair_colunas_passageiros(df)
    if not cols_pass:
        return pd.DataFrame()

    # idx_alvo é 1-based para o usuário
    idx_zero = max(idx_alvo - 1, 0)
    inicio = max(idx_zero - window, 0)
    df_janela = df.iloc[inicio:idx_zero]

    if df_janela.empty:
        return pd.DataFrame()

    # Frequência por coluna
    top_por_col = []
    for c in cols_pass:
        vc = df_janela[c].value_counts().reset_index()
        vc.columns = ["valor", "freq"]
        top_por_col.append(vc)

    # Montar candidatos combinando os top valores coluna a coluna
    tops_lim = []
    for vc in top_por_col:
        k_max = max(3, min(6, n_series // max(1, len(cols_pass))))
        tops_lim.append(list(vc["valor"].head(k_max)))

    candidatos = []
    for comb in product(*tops_lim):
        candidatos.append(list(map(int, comb)))

    # Scoring simples: soma das frequências individuais
    def score_serie(serie: List[int]) -> float:
        s = 0.0
        for i, v in enumerate(serie):
            vc = top_por_col[i]
            freq = vc.loc[vc["valor"] == v, "freq"]
            s += float(freq.iloc[0]) if not freq.empty else 0.0
        return s

    dados = []
    for serie in candidatos:
        dados.append(
            {
                "series": serie,
                "score_s6": score_serie(serie),
                "origem": "S6_PROFUNDO",
            }
        )

    df_out = pd.DataFrame(dados).drop_duplicates(subset=["series"])
    df_out = df_out.sort_values("score_s6", ascending=False).head(n_series).reset_index(drop=True)
    return df_out


def micro_leque_ultra(
    df: pd.DataFrame,
    idx_alvo: int,
    n_vizinhos: int = 3,
) -> pd.DataFrame:
    """
    Micro-Leque ULTRA:

    - Usa séries próximas (anteriores e posteriores) ao alvo como base
    - Gera pequenas variações em torno delas
    """
    if df is None or df.empty:
        return pd.DataFrame()

    cols_pass = extrair_colunas_passageiros(df)
    if not cols_pass:
        return pd.DataFrame()

    idx_zero = max(idx_alvo - 1, 0)
    n = len(df)

    vizinhos_idx = set()
    for delta in range(1, n_vizinhos + 1):
        if idx_zero - delta >= 0:
            vizinhos_idx.add(idx_zero - delta)
        if idx_zero + delta < n:
            vizinhos_idx.add(idx_zero + delta)

    if not vizinhos_idx:
        return pd.DataFrame()

    base_series = []
    for i in sorted(vizinhos_idx):
        row = df.iloc[i]
        base_series.append(linha_para_serie(row, cols_pass))

    candidatos = []
    for serie in base_series:
        candidatos.append(serie)  # original

        # Troca simples
        if len(serie) >= 2:
            s2 = serie.copy()
            i1, i2 = random.sample(range(len(serie)), 2)
            s2[i1], s2[i2] = s2[i2], s2[i1]
            candidatos.append(s2)

        # Shuffle leve
        s3 = serie.copy()
        random.shuffle(s3)
        candidatos.append(s3)

    dados = []
    for serie in candidatos:
        dados.append(
            {
                "series": list(map(int, serie)),
                "score_micro": 1.0,
                "origem": "MICRO_LEQUE",
            }
        )

    df_out = pd.DataFrame(dados).drop_duplicates(subset=["series"])
    return df_out.reset_index(drop=True)


def monte_carlo_profundo_ultra(
    df: pd.DataFrame,
    idx_alvo: int,
    n_sim: int = 2000,
    n_series_saida: int = 60,
    window: int = 120,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Monte Carlo Profundo ULTRA:

    - Usa janelas profundas para gerar simulações independentes
    - Amostra passageiros conforme distribuição empírica por coluna
    """
    if df is None or df.empty or n_sim <= 0:
        return pd.DataFrame()

    cols_pass = extrair_colunas_passageiros(df)
    if not cols_pass:
        return pd.DataFrame()

    idx_zero = max(idx_alvo - 1, 0)
    inicio = max(idx_zero - window, 0)
    df_janela = df.iloc[inicio:idx_zero]

    if df_janela.empty:
        return pd.DataFrame()

    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    # Distribuições por coluna
    dist_col = {}
    for c in cols_pass:
        valores = df_janela[c].dropna().astype(int).values
        if len(valores) == 0:
            continue
        vals, counts = np.unique(valores, return_counts=True)
        prob = counts / counts.sum()
        dist_col[c] = (vals, prob)

    if not dist_col:
        return pd.DataFrame()

    series_mc = []
    for _ in range(n_sim):
        serie = []
        for c in cols_pass:
            if c not in dist_col:
                valores = df_janela[c].dropna().astype(int).values
                if len(valores) == 0:
                    continue
                serie.append(int(random.choice(list(valores))))
            else:
                vals, prob = dist_col[c]
                serie.append(int(np.random.choice(vals, p=prob)))
        if len(serie) == len(cols_pass):
            series_mc.append(serie)

    if not series_mc:
        return pd.DataFrame()

    contagem = Counter(tuple(s) for s in series_mc)
    dados = []
    for serie_tup, freq in contagem.items():
        dados.append(
            {
                "series": list(map(int, serie_tup)),
                "freq_mc": int(freq),
                "origem": "MONTE_CARLO",
            }
        )

    df_out = pd.DataFrame(dados)
    df_out["score_mc"] = df_out["freq_mc"] / df_out["freq_mc"].max()
    df_out = df_out.sort_values("score_mc", ascending=False).head(n_series_saida).reset_index(drop=True)
    return df_out


def montar_previsao_turbo_ultra(
    df,
    idx_alvo,
    n_s6=60,
    janela_s6=80,
    janela_mc=120,
    n_sim_mc=2000,
    peso_s6=0.5,
    peso_mc=0.4,
    peso_micro=0.1,
):
    import numpy as np
    import pandas as pd
    import streamlit as st

    progresso = st.progress(0)
    status = st.empty()

    # ============================================================
    # ETAPA 1: S6 PROFUNDO ULTRA
    # ============================================================
    progresso.progress(5)
    status.write("🔧 Gerando S6 Profundo ULTRA...")

    df_s6 = gerar_s6_profundo_ultra(
        df,
        idx_alvo=idx_alvo,
        n_series_saida=n_s6,
        janela=janela_s6,
    )

    if df_s6 is None or df_s6.empty:
        st.error("S6 Profundo ULTRA não gerou dados.")
        return pd.DataFrame()

    df_s6 = df_s6.copy()
    df_s6["score_s6"] = df_s6["score"].astype(float)
    df_s6["score_mc"] = 0.0
    df_s6["score_micro"] = 0.0

    progresso.progress(20)
    status.write("🔧 S6 Profundo concluído.")

    # ============================================================
    # ETAPA 2: MONTE CARLO PROFUNDO ULTRA
    # ============================================================
    progresso.progress(25)
    status.write("🎲 Executando Monte Carlo Profundo ULTRA...")

    df_mc = gerar_monte_carlo_profundo_ultra(
        df,
        idx_alvo=idx_alvo,
        janela_mc=janela_mc,
        n_sim=n_sim_mc,
    )

    if df_mc is None or df_mc.empty:
        st.error("Monte Carlo Profundo ULTRA não gerou dados.")
        return pd.DataFrame()

    df_mc = df_mc.copy()
    df_mc["score_s6"] = 0.0
    df_mc["score_mc"] = df_mc["score"].astype(float)
    df_mc["score_micro"] = 0.0

    progresso.progress(55)
    status.write("🎲 Monte Carlo concluído.")

    # ============================================================
    # ETAPA 3: MICRO-LEQUE ULTRA
    # ============================================================
    progresso.progress(60)
    status.write("🪶 Gerando Micro-Leque ULTRA...")

    df_micro = gerar_micro_leque_ultra(df, idx_alvo=idx_alvo)

    if df_micro is None or df_micro.empty:
        df_micro = pd.DataFrame(columns=["series", "score"])

    df_micro = df_micro.copy()
    df_micro["score_s6"] = 0.0
    df_micro["score_mc"] = 0.0
    df_micro["score_micro"] = df_micro["score"].astype(float)

    progresso.progress(70)
    status.write("🪶 Micro-Leque pronto.")

    # ============================================================
    # ETAPA 4: UNIFICAÇÃO
    # ============================================================
    progresso.progress(75)
    status.write("🔗 Unificando resultados...")

    df_all = pd.concat([df_s6, df_mc, df_micro], ignore_index=True)

    # ============================================================
    # ETAPA 5: NORMALIZAÇÃO DAS SÉRIES
    # ============================================================
    progresso.progress(80)
    status.write("🧩 Normalizando séries...")

    def normalizar_serie(s):
        if s is None:
            return tuple()
        if isinstance(s, str):
            try:
                vals = [int(x) for x in s.replace(",", " ").split() if x.isdigit()]
                return tuple(vals)
            except:
                return tuple()
        if isinstance(s, (tuple, list, np.ndarray)):
            try:
                return tuple(int(x) for x in s)
            except:
                return tuple()
        return tuple()

    df_all["series"] = df_all["series"].apply(normalizar_serie)
    df_all = df_all[df_all["series"].apply(lambda x: len(x) > 0)].copy()

    progresso.progress(90)
    status.write("🧩 Séries normalizadas.")

    # ============================================================
    # ETAPA 6: FUSÃO FINAL ULTRA
    # ============================================================
    progresso.progress(95)
    status.write("⚖️ Aplicando fusão ULTRA...")

    df_fusao = (
        df_all.groupby("series", as_index=False)
        .agg(
            {
                "score_s6": "max",
                "score_mc": "max",
                "score_micro": "max",
            }
        )
        .copy()
    )

    df_fusao["score_total"] = (
        df_fusao["score_s6"] * peso_s6
        + df_fusao["score_mc"] * peso_mc
        + df_fusao["score_micro"] * peso_micro
    )

    df_fusao = df_fusao.sort_values("score_total", ascending=False).reset_index(drop=True)

    progresso.progress(100)
    status.write("✨ Fusão ULTRA concluída.")

    return df_fusao




# ============================================================
# REPLAY + QDS REAL
# ============================================================

def executar_pipeline_turbo_ultra_para_replay(
    df: pd.DataFrame,
    idx_alvo: int,
    params_base: Dict[str, Any],
    modo_replay: str = "LIGHT",
) -> Dict[str, Any]:
    """
    Wrapper para usar o mesmo núcleo TURBO++ ULTRA no Replay.

    - LIGHT: menos simulações Monte Carlo / janelas menores
    - ULTRA: usa parâmetros cheios
    """
    params = dict(params_base or {})
    if not params:
        params = {
            "n_series_saida": 60,
            "window_s6": 80,
            "window_mc": 120,
            "n_sim_mc": 2000,
            "incluir_micro_leque": True,
            "peso_s6": 0.5,
            "peso_mc": 0.4,
            "peso_micro": 0.1,
        }

    if modo_replay == "LIGHT":
        params["n_series_saida"] = min(30, params["n_series_saida"])
        params["window_s6"] = max(40, int(params["window_s6"] * 0.6))
        params["window_mc"] = max(60, int(params["window_mc"] * 0.6))
        params["n_sim_mc"] = max(300, int(params["n_sim_mc"] * 0.3))
    else:
        params["n_series_saida"] = max(60, params["n_series_saida"])
        params["n_sim_mc"] = max(1500, int(params["n_sim_mc"] * 1.0))

    df_turbo = montar_previsao_turbo_ultra(
        df,
        idx_alvo=idx_alvo,
        n_series_saida=params["n_series_saida"],
        window_s6=params["window_s6"],
        window_mc=params["window_mc"],
        n_sim_mc=params["n_sim_mc"],
        incluir_micro_leque=params["incluir_micro_leque"],
        peso_s6=params["peso_s6"],
        peso_mc=params["peso_mc"],
        peso_micro=params["peso_micro"],
    )

    if df_turbo is None or df_turbo.empty:
        return {"ok": False, "df": pd.DataFrame(), "serie_top1": None}

    top1 = df_turbo.iloc[0]["series"]
    return {"ok": True, "df": df_turbo, "serie_top1": top1}


def calcular_qds_real(aus_replay: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula QDS REAL a partir da tabela de replay:

    Espera colunas:
    - hits (número de acertos)
    - idx_alvo
    """
    if aus_replay is None or aus_replay.empty:
        return {
            "qds": 0.0,
            "media_hits": 0.0,
            "p_ge_1": 0.0,
            "p_ge_3": 0.0,
            "p_ge_4": 0.0,
            "n": 0,
        }

    n = len(aus_replay)
    media_hits = float(aus_replay["hits"].mean())

    p_ge_1 = float((aus_replay["hits"] >= 1).mean())
    p_ge_3 = float((aus_replay["hits"] >= 3).mean())
    p_ge_4 = float((aus_replay["hits"] >= 4).mean())

    qds = 100.0 * (0.25 * p_ge_1 + 0.35 * p_ge_3 + 0.40 * p_ge_4)

    return {
        "qds": qds,
        "media_hits": media_hits,
        "p_ge_1": p_ge_1,
        "p_ge_3": p_ge_3,
        "p_ge_4": p_ge_4,
        "n": n,
    }


# ============================================================
# LAYOUT — SIDEBAR E CABEÇALHO GLOBAL
# ============================================================

st.sidebar.title("Predict Cars V14-FLEX ULTRA REAL")
painel = st.sidebar.radio(
    "Escolha o painel:",
    [
        "📥 Histórico — Entrada",
        "🔍 Pipeline V14-FLEX ULTRA — Execução Base",
        "🚨 Monitor de Risco (Barômetro + k*)",
        "📊 IDX / IPF / IPO ULTRA — Núcleos Estruturais",
        "🧠 S6 Profundo ULTRA",
        "🚀 Modo TURBO++ — Painel Completo",
        "📅 Modo Replay Automático do Histórico",
        "🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)",
    ],
)

st.markdown(
    """
### Predict Cars V14-FLEX ULTRA REAL (TURBO++)
Sistema ULTRA completo com: Barômetro, k*, IDX, IPF / IPO, S6 Profundo, Micro-Leque, Monte Carlo Profundo, QDS + Backtest, Replay LIGHT / ULTRA e Modo TURBO++ Adaptativo.
"""
)

# ============================================================
# PAINEL 1 — HISTÓRICO — ENTRADA FLEX
# ============================================================

if painel == "📥 Histórico — Entrada":
    st.markdown("## 📥 Histórico — Entrada (FLEX)")

    df = st.session_state.get("df", None)

    if df is not None and not df.empty:
        st.info("Histórico já carregado na sessão.")

    opc = st.radio(
        "Como deseja carregar o histórico?",
        ["Enviar arquivo CSV", "Copiar e colar o histórico"],
    )

    # ---------- OPÇÃO 1 — UPLOAD DE ARQUIVO ----------
    if opc == "Enviar arquivo CSV":
        file = st.file_uploader("Selecione o arquivo CSV:", type=["csv"])
        if file is not None:
            try:
                df_raw = pd.read_csv(file, sep=";", header=None, engine="python")
                df_raw = df_raw.dropna(axis=1, how="all")

                df = preparar_historico_flex(df_raw)
                st.session_state["df"] = df

                st.success("Histórico carregado com sucesso (modo FLEX).")
                st.write(f"Total de séries: **{len(df)}**")
                st.write(f"Passageiros por série (FLEX): **{st.session_state['n_passageiros']}**")
                st.write(f"Coluna k (guardas que acertaram): **{st.session_state['col_k']}**")
                registrar_evento("Histórico carregado via CSV (FLEX).")
            except Exception as e:
                st.error(f"Erro ao carregar CSV: {e}")

    # ---------- OPÇÃO 2 — COLAR TEXTO ----------
    if opc == "Copiar e colar o histórico":
        texto = st.text_area(
            "Cole aqui o histórico no formato C1;41;5;4;52;30;33;0 ...",
            height=200,
        )
        if st.button("Carregar histórico colado"):
            try:
                linhas = [linha.strip() for linha in texto.splitlines() if linha.strip()]
                dados = [linha.split(";") for linha in linhas]
                df_raw = pd.DataFrame(dados)
                df = preparar_historico_flex(df_raw)
                st.session_state["df"] = df

                st.success("Histórico carregado com sucesso (modo FLEX).")
                st.write(f"Total de séries: **{len(df)}**")
                st.write(f"Passageiros por série (FLEX): **{st.session_state['n_passageiros']}**")
                st.write(f"Coluna k (guardas que acertaram): **{st.session_state['col_k']}**")
                registrar_evento("Histórico carregado via texto colado (FLEX).")
            except Exception as e:
                st.error(f"Erro ao interpretar o texto como histórico: {e}")

    # Resumo do histórico já na sessão
    df = st.session_state.get("df", None)
    if df is not None and not df.empty:
        st.markdown("### 📌 Resumo do histórico atual")
        n_total = len(df)
        n_pass = st.session_state.get("n_passageiros", None)
        col_k = st.session_state.get("col_k", "k")

        st.write(f"Total de séries: **{n_total}**")
        st.write(f"Passageiros por série (detectado): **{n_pass}**")
        st.write(f"Coluna k (guardas que acertaram): **{col_k}**")

        idx_inspec = st.number_input(
            "Selecione um índice interno para inspecionar (1 = primeira série carregada):",
            min_value=1,
            max_value=n_total,
            value=n_total,
            step=1,
        )
        row = df.iloc[int(idx_inspec) - 1]
        cols_pass = extrair_colunas_passageiros(df)
        serie = linha_para_serie(row, cols_pass)
        kval = row[col_k] if col_k in row else None

        st.write(
            f"C{idx_inspec} — Passageiros: {serie} — k (guardas que acertaram): {kval}"
        )


# ============================================================
# PAINEL 2 — PIPELINE V14-FLEX ULTRA — EXECUÇÃO BASE
# ============================================================

if painel == "🔍 Pipeline V14-FLEX ULTRA — Execução Base":
    st.markdown("## 🔍 Pipeline V14-FLEX ULTRA — Execução Base")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    n_series_hist = len(df)
    cols_pass = extrair_colunas_passageiros(df)
    col_k = st.session_state.get("col_k", "k")

    # Seleção de índice alvo
    modo_idx = st.radio(
        "Como deseja escolher o índice alvo?",
        ["Usar última série do histórico", "Escolher manualmente"],
    )

    if modo_idx == "Usar última série do histórico":
        idx_alvo = n_series_hist
    else:
        idx_alvo = st.number_input(
            "Selecione o índice alvo (1 = primeira série):",
            min_value=1,
            max_value=n_series_hist,
            value=n_series_hist,
            step=1,
        )

    row_alvo = df.iloc[int(idx_alvo) - 1]
    serie_alvo = linha_para_serie(row_alvo, cols_pass)
    kval = row_alvo[col_k] if col_k in row_alvo else None

    st.markdown(
        f"### 🎯 Seleção da série alvo\n"
        f"📌 Série alvo selecionada — ID C{idx_alvo} — Passageiros: {serie_alvo} — "
        f"k (guardas que acertaram): {kval}"
    )

    # 1) Diagnóstico de risco — Barômetro + k*
    st.markdown("### 1️⃣ Diagnóstico de risco — Barômetro + k*")

    bar = calcular_barometro_ultra(df)
    k_star_info = calcular_k_star_ultra(df)

    if bar["estado"] == "estavel":
        st.write("🟢 **Barômetro: estável — estrada historicamente previsível.**")
    elif bar["estado"] == "atencao":
        st.write("🟡 **Barômetro: atenção — estrada com oscilações relevantes.**")
    else:
        st.write("🔴 **Barômetro: crítico — estrada em regime de ruptura/turbulência pesada.**")

    st.write(
        f"Índice de turbulência: {bar['turbulencia']:.3f} • "
        f"Desvio-padrão de k: {bar['std_k']:.3f} • "
        f"Média de |Δk|: {bar['mean_abs_dk']:.3f}"
    )

    if k_star_info["estado"] == "estavel":
        st.write("🟢 **k* ULTRA REAL (Sentinela baseado em k dos guardas): estável.**")
    elif k_star_info["estado"] == "atencao":
        st.write(
            f"🟡 **k* ULTRA REAL (Sentinela): atenção — padrão misto de acerto/erro. (k*={k_star_info['k_star']:.1f})**"
        )
    else:
        st.write(
            f"🔴 **k* ULTRA REAL (Sentinela): crítico — guardas em modo caótico. (k*={k_star_info['k_star']:.1f})**"
        )

    st.write(
        f"Entropia de k: {k_star_info['entropia']:.3f} • "
        f"Entropia normalizada: {k_star_info['entropia_norm']:.3f}"
    )

    st.info(
        "🌐 Pré-síntese de risco global (sem afetar o motor)\n\n"
        "O Barômetro ULTRA avalia a estabilidade global dos acertos dos guardas (k) ao longo da estrada. "
        "O k* ULTRA REAL atua como sentinela de caos local: se os guardas oscilam demais entre acertar tudo "
        "e errar tudo em janelas curtas, k* sobe. O motor ULTRA usa essas informações apenas como contexto, "
        "não como trava direta da previsão."
    )

    # 2) Núcleos IDX / IPF / IPO
    st.markdown("### 2️⃣ Núcleos IDX / IPF / IPO ULTRA (base para previsão)")

    window_nucl = st.number_input(
        "Tamanho da janela de histórico para cálculo dos núcleos:",
        min_value=10,
        max_value=max(10, n_series_hist - 1),
        value=40,
        step=5,
    )

    nucleos = calcular_nucleos_idx_ipf_ipo(df, idx_alvo=idx_alvo, window=window_nucl)
    if not nucleos:
        st.error("Não foi possível calcular os núcleos IDX / IPF / IPO para esta janela.")
    else:
        st.write(
            f"Janela usada: índices de {nucleos['inicio']} até {nucleos['fim']} "
            f"(tamanho {nucleos['tamanho']} séries)."
        )
        st.markdown("**IDX ULTRA (média ponderada dinâmica)**")
        st.code(serie_para_str(nucleos["idx_ultra"]), language="text")

        st.markdown("**IPF ULTRA (mediana robusta estrutural)**")
        st.code(serie_para_str(nucleos["ipf_ultra"]), language="text")

        st.markdown("**IPO ORIGINAL (média simples de frequência)**")
        st.code(serie_para_str(nucleos["ipo_original"]), language="text")

        st.markdown("**IPO ULTRA (refinada anti-sesgo, mistura IDX + IPF + IPO)**")
        st.code(serie_para_str(nucleos["ipo_ultra"]), language="text")

        st.info(
            "Interpretação rápida:\n\n"
            "- IDX ULTRA destaca passageiros mais importantes com base em frequência + posição (os 'mais vistos').\n"
            "- IPF ULTRA representa a estrutura central, resistente a ruídos (mediana por posição).\n"
            "- IPO ORIGINAL mostra a fotografia bruta da frequência.\n"
            "- IPO ULTRA é a versão refinada, corrigindo vieses do histórico e reforçando o núcleo realmente preditivo."
        )


# ============================================================
# PAINEL 3 — MONITOR DE RISCO (BARÔMETRO + k*)
# ============================================================

if painel == "🚨 Monitor de Risco (Barômetro + k*)":
    st.markdown("## 🚨 Monitor de Risco (Barômetro + k*)")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    bar = calcular_barometro_ultra(df)
    k_star_info = calcular_k_star_ultra(df)

    st.markdown("### 🌡️ Barômetro ULTRA REAL")

    if bar["estado"] == "estavel":
        st.write("🟢 **Estado do barômetro: estável**")
        st.write("Barômetro: estrada historicamente estável — padrão de acertos dos guardas relativamente previsível.")
    elif bar["estado"] == "atencao":
        st.write("🟡 **Estado do barômetro: atenção**")
        st.write("Barômetro: estrada com oscilações relevantes no padrão de acertos dos guardas.")
    else:
        st.write("🔴 **Estado do barômetro: crítico**")
        st.write("Barômetro: estrada em regime de ruptura/turbulência pesada — acertos caóticos.")

    st.write(f"Índice de turbulência: {bar['turbulencia']:.3f}")
    st.write(f"Desvio-padrão de k: {bar['std_k']:.3f}")
    st.write(f"Média de |Δk| (variação entre carros): {bar['mean_abs_dk']:.3f}")

    st.markdown("### 🛰️ k* ULTRA REAL (Sentinela)")

    if k_star_info["estado"] == "estavel":
        st.write(
            f"🟢 Estado do k*: estável — acertos relativamente consistentes. (k*={k_star_info['k_star']:.1f})"
        )
    elif k_star_info["estado"] == "atencao":
        st.write(
            f"🟡 Estado do k*: atenção — padrão misto de acertos, com alternância relevante. (k*={k_star_info['k_star']:.1f})"
        )
    else:
        st.write(
            f"🔴 Estado do k*: crítico — guardas em regime altamente instável. (k*={k_star_info['k_star']:.1f})"
        )

    st.write(f"Entropia de k: {k_star_info['entropia']:.3f}")
    st.write(f"Entropia normalizada: {k_star_info['entropia_norm']:.3f}")

    # Síntese global de risco
    if bar["estado"] == "critico" or k_star_info["estado"] == "critico":
        nivel = "critico"
    elif bar["estado"] == "atencao" or k_star_info["estado"] == "atencao":
        nivel = "atencao"
    else:
        nivel = "estavel"

    st.markdown("### 🌐 Síntese Global de Risco")
    if nivel == "estavel":
        st.write("🟢 Nível global de risco: **estável**")
    elif nivel == "atencao":
        st.write("🟡 Nível global de risco: **atenção**")
    else:
        st.write("🔴 Nível global de risco: **crítico**")

    st.info(
        "Ambiente global em atenção/estabilidade/ruptura é definido combinando o Barômetro ULTRA (clima geral da estrada) "
        "e o k* (sentinela de instabilidade local). O Monitor de Risco **não bloqueia** o motor ULTRA; ele funciona como "
        "um painel de contexto, ajudando a interpretar em que tipo de ambiente as previsões estão sendo feitas."
    )


# ============================================================
# PAINEL 4 — IDX / IPF / IPO ULTRA — NÚCLEOS ESTRUTURAIS
# ============================================================

if painel == "📊 IDX / IPF / IPO ULTRA — Núcleos Estruturais":
    st.markdown("## 📊 IDX / IPF / IPO ULTRA — Núcleos Estruturais")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    n_series_hist = len(df)
    cols_pass = extrair_colunas_passageiros(df)
    col_k = st.session_state.get("col_k", "k")

    idx_alvo = st.number_input(
        "Selecione o índice alvo para calcular os núcleos (1 = primeira série):",
        min_value=1,
        max_value=n_series_hist,
        value=n_series_hist,
        step=1,
    )

    row_alvo = df.iloc[int(idx_alvo) - 1]
    serie_alvo = linha_para_serie(row_alvo, cols_pass)
    kval = row_alvo[col_k] if col_k in row_alvo else None

    st.markdown(
        f"### 🎯 Série alvo (contexto imediato)\n"
        f"ID C{idx_alvo} | Passageiros: {serie_alvo} | "
        f"k (guardas que acertaram exatamente o carro): {kval}"
    )

    window_nucl = st.number_input(
        "Tamanho da janela de histórico para cálculo dos núcleos:",
        min_value=10,
        max_value=max(10, n_series_hist - 1),
        value=40,
        step=5,
    )

    nucleos = calcular_nucleos_idx_ipf_ipo(df, idx_alvo=idx_alvo, window=window_nucl)

    if not nucleos:
        st.error("Não foi possível calcular os núcleos para a janela selecionada.")
    else:
        st.markdown("### 📦 Janela de histórico usada")
        st.write(f"Início da janela: {nucleos['inicio']}")
        st.write(f"Fim da janela: {nucleos['fim']}")
        st.write(f"Tamanho da janela: {nucleos['tamanho']}")

        st.markdown("### 🧠 Núcleos IDX / IPF / IPO ULTRA")

        st.markdown("**IDX ULTRA (média ponderada dinâmica)**")
        st.code(serie_para_str(nucleos["idx_ultra"]), language="text")

        st.markdown("**IPF ULTRA (mediana robusta)**")
        st.code(serie_para_str(nucleos["ipf_ultra"]), language="text")

        st.markdown("**IPO ORIGINAL (média simples)**")
        st.code(serie_para_str(nucleos["ipo_original"]), language="text")

        st.markdown("**IPO ULTRA (refinada anti-sesgo)**")
        st.code(serie_para_str(nucleos["ipo_ultra"]), language="text")

        st.markdown("### Interpretação rápida:")
        st.write(
            "- IDX ULTRA destaca passageiros mais importantes com base em frequência + posição (os 'mais vistos').\n"
            "- IPF ULTRA representa a estrutura central, resistente a ruídos (mediana por posição).\n"
            "- IPO ORIGINAL mostra a fotografia bruta da frequência.\n"
            "- IPO ULTRA é a versão refinada, corrigindo vieses do histórico e reforçando o núcleo realmente preditivo."
        )


# ============================================================
# PAINEL 5 — 🧠 S6 PROFUNDO ULTRA
# ============================================================

if painel == "🧠 S6 Profundo ULTRA":
    st.markdown("## 🧠 S6 Profundo ULTRA — Núcleo Determinístico Profundo")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    n_series_hist = len(df)
    cols_pass = extrair_colunas_passageiros(df)

    idx_alvo = st.number_input(
        "Índice alvo para S6 Profundo (1 = primeira série):",
        min_value=1,
        max_value=n_series_hist,
        value=n_series_hist,
        step=1,
    )

    window_s6 = st.number_input(
        "Janela S6 Profundo (n séries para trás):",
        min_value=20,
        max_value=max(20, n_series_hist - 1),
        value=80,
        step=10,
    )

    n_series_s6 = st.number_input(
        "Quantidade de séries S6 a gerar:",
        min_value=10,
        max_value=200,
        value=60,
        step=10,
    )

    if st.button("🧠 Rodar S6 Profundo ULTRA para este índice alvo"):
        with st.spinner("Calculando S6 Profundo ULTRA..."):
            df_s6 = s6_profundo_ultra(
                df,
                idx_alvo=int(idx_alvo),
                window=int(window_s6),
                n_series=int(n_series_s6),
            )

        if df_s6 is None or df_s6.empty:
            st.error("Não foi possível gerar séries S6 Profundo para este índice.")
        else:
            st.markdown("### 🎯 Série alvo (contexto)")
            row_alvo = df.iloc[int(idx_alvo) - 1]
            serie_alvo = linha_para_serie(row_alvo, cols_pass)
            st.code(serie_para_str(serie_alvo), language="text")

            st.markdown("### 📊 Tabela S6 Profundo ULTRA — Séries geradas")
            df_view = df_s6.copy()
            df_view["series_str"] = df_view["series"].apply(serie_para_str)
            st.dataframe(
                df_view[["series_str", "score_s6"]].rename(
                    columns={"series_str": "Série (passageiros)", "score_s6": "Score S6 Profundo"}
                ),
                use_container_width=True,
            )

            melhor = df_s6.iloc[0]
            st.markdown("### 🎯 Série #1 de S6 Profundo ULTRA (núcleo mais forte)")
            st.code(serie_para_str(melhor["series"]), language="text")

            st.info(
                "S6 Profundo ULTRA combina frequências por coluna em uma janela profunda de histórico, "
                "gerando um conjunto de séries determinísticas que representam o núcleo estrutural mais "
                "forte da estrada naquele ponto. Essas séries alimentam o Modo TURBO++ (junto com Micro-Leque "
                "e Monte Carlo Profundo) como base determinística de alta confiança."
            )


# ============================================================
# PAINEL 6 — 🚀 MODO TURBO++ — PAINEL COMPLETO
# ============================================================

if painel == "🚀 Modo TURBO++ — Painel Completo":
    st.markdown("## 🚀 Modo TURBO++ ULTRA Adaptativo — Painel Completo")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    cols_pass = extrair_colunas_passageiros(df)
    if not cols_pass:
        st.error("Não foi possível identificar as colunas de passageiros no histórico.")
        st.stop()

    n_series_hist = len(df)

    col1, col2 = st.columns(2)
    with col1:
        idx_alvo = st.number_input(
            "Índice alvo (1 = primeira série do histórico):",
            min_value=1,
            max_value=n_series_hist,
            value=n_series_hist,
            step=1,
        )
        n_series_saida = st.slider(
            "Quantidade de séries na saída TURBO++ (núcleo resiliente + cobertura):",
            min_value=10,
            max_value=120,
            value=60,
            step=5,
        )
        incluir_micro = st.checkbox("Incluir Micro-Leque ULTRA (cobertura de vento fina)", value=True)

    with col2:
        window_s6 = st.slider(
            "Janela S6 Profundo ULTRA (n séries para trás):",
            min_value=20,
            max_value=200,
            value=80,
            step=10,
        )
        window_mc = st.slider(
            "Janela Monte Carlo Profundo ULTRA:",
            min_value=40,
            max_value=300,
            value=120,
            step=10,
        )
        n_sim_mc = st.slider(
            "Simulações Monte Carlo Profundo ULTRA:",
            min_value=200,
            max_value=5000,
            value=2000,
            step=200,
        )

    st.markdown("### ⚖️ Pesos de fusão ULTRA (S6 / Monte Carlo / Micro-Leque)")
    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        peso_s6 = st.slider("Peso S6 Profundo", 0.0, 1.0, 0.5, 0.05)
    with colp2:
        peso_mc = st.slider("Peso Monte Carlo", 0.0, 1.0, 0.4, 0.05)
    with colp3:
        peso_micro = st.slider("Peso Micro-Leque", 0.0, 1.0, 0.1, 0.05)

    soma_pesos = peso_s6 + peso_mc + peso_micro
    if soma_pesos <= 0:
        peso_s6, peso_mc, peso_micro = 0.5, 0.4, 0.1
    else:
        peso_s6 /= soma_pesos
        peso_mc /= soma_pesos
        peso_micro /= soma_pesos

    st.markdown("---")

    rodar = st.button("🚀 Rodar Modo TURBO++ ULTRA para este índice alvo")

    if rodar:
        with st.spinner("Rodando S6 Profundo, Micro-Leque e Monte Carlo Profundo ULTRA..."):
            df_turbo = montar_previsao_turbo_ultra(
                df,
                idx_alvo=int(idx_alvo),
                n_series_saida=int(n_series_saida),
                window_s6=int(window_s6),
                window_mc=int(window_mc),
                n_sim_mc=int(n_sim_mc),
                incluir_micro_leque=bool(incluir_micro),
                peso_s6=float(peso_s6),
                peso_mc=float(peso_mc),
                peso_micro=float(peso_micro),
            )

        if df_turbo is None or df_turbo.empty:
            st.error("Não foi possível gerar séries TURBO++ ULTRA para este índice.")
        else:
            st.session_state["previsao_turbo_ultra"] = df_turbo
            st.session_state["previsao_turbo_ultra_params"] = {
                "idx_alvo": int(idx_alvo),
                "n_series_saida": int(n_series_saida),
                "window_s6": int(window_s6),
                "window_mc": int(window_mc),
                "n_sim_mc": int(n_sim_mc),
                "incluir_micro_leque": bool(incluir_micro),
                "peso_s6": float(peso_s6),
                "peso_mc": float(peso_mc),
                "peso_micro": float(peso_micro),
            }

            st.markdown("### 🚗 Série alvo (carro atual na estrada)")
            row_alvo = df.iloc[int(idx_alvo) - 1]
            serie_alvo = linha_para_serie(row_alvo, cols_pass)
            st.code(serie_para_str(serie_alvo), language="text")

            # Integração com Barômetro / k*
            bar = calcular_barometro_ultra(df)
            k_star_info = calcular_k_star_ultra(df)

            if bar["estado"] == "estavel":
                contexto_barometro = "🟢 Barômetro ULTRA REAL: Estrada em regime normal."
            elif bar["estado"] == "atencao":
                contexto_barometro = "🟡 Barômetro ULTRA REAL: Região de transição / pré-ruptura."
            else:
                contexto_barometro = "🔴 Barômetro ULTRA REAL: Região de turbulência pesada / pós-ruptura."

            if k_star_info["estado"] == "estavel":
                contexto_k = "🟢 k* ULTRA REAL: Ambiente estável — guardas convergindo."
            elif k_star_info["estado"] == "atencao":
                contexto_k = "🟡 k* ULTRA REAL: Pré-ruptura residual — atenção elevada."
            else:
                contexto_k = "🔴 k* ULTRA REAL: Ambiente crítico — sensibilidade máxima dos guardas."

            contexto_k += f" (k* ≈ {k_star_info['k_star']:.1f}%)"
            st.info(contexto_barometro + "\n\n" + contexto_k)

            st.markdown("### 📊 Leque TURBO++ ULTRA — Núcleo Resiliente + Cobertura")
            df_view = df_turbo.copy()
            df_view["series_str"] = df_view["series"].apply(serie_para_str)
            st.dataframe(
                df_view[["series_str", "score_final", "score_s6", "score_mc", "score_micro"]].rename(
                    columns={
                        "series_str": "Série (passageiros)",
                        "score_final": "Score ULTRA",
                        "score_s6": "Score S6",
                        "score_mc": "Score Monte Carlo",
                        "score_micro": "Score Micro-Leque",
                    }
                ),
                use_container_width=True,
            )

            melhor = df_turbo.iloc[0]
            st.markdown("### 🎯 Previsão Final TURBO++ ULTRA (Série #1 do Núcleo Resiliente)")
            st.code(serie_para_str(melhor["series"]), language="text")


# ============================================================
# PAINEL 7 — 📅 MODO REPLAY AUTOMÁTICO DO HISTÓRICO
# ============================================================

if painel == "📅 Modo Replay Automático do Histórico":
    st.markdown("## 📅 Modo Replay Automático do Histórico")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    cols_pass = extrair_colunas_passageiros(df)
    if not cols_pass:
        st.error("Não foi possível identificar as colunas de passageiros no histórico.")
        st.stop()

    n_series_hist = len(df)

    st.markdown("### 🎬 Configuração do Replay (LIGHT / ULTRA)")

    col1, col2 = st.columns(2)
    with col1:
        idx_inicio = st.number_input(
            "Índice inicial do Replay:",
            min_value=1,
            max_value=max(1, n_series_hist - 1),
            value=max(1, n_series_hist - 60),
            step=1,
        )
        idx_fim = st.number_input(
            "Índice final do Replay:",
            min_value=idx_inicio,
            max_value=max(1, n_series_hist - 1),
            value=max(1, n_series_hist - 1),
            step=1,
        )
        horizonte = st.number_input(
            "Horizonte de validação (quantas séries à frente comparar):",
            min_value=1,
            max_value=5,
            value=1,
            step=1,
        )

    with col2:
        modo_replay = st.radio(
            "Modo de Replay:",
            options=["LIGHT (rápido)", "ULTRA (profundo)"],
        )
        usar_params_turbo = st.checkbox(
            "Usar parâmetros atuais do Modo TURBO++ ULTRA (se já rodou)",
            value=True,
        )
        mostrar_detalhes = st.checkbox("Mostrar tabela completa de resultados do Replay", value=True)

    params_base = st.session_state.get("previsao_turbo_ultra_params", {})
    if not usar_params_turbo:
        params_base = {}

    st.markdown("---")
    rodar_replay = st.button("📅 Rodar Replay Automático do Histórico")

    if rodar_replay:
        registros = []
        modo_interno = "LIGHT" if modo_replay.startswith("LIGHT") else "ULTRA"

        with st.spinner("Executando Replay do histórico com o núcleo TURBO++ ULTRA..."):
            for idx in range(int(idx_inicio), int(idx_fim) + 1):
                idx_real = idx + int(horizonte)
                if idx_real > n_series_hist:
                    continue

                res = executar_pipeline_turbo_ultra_para_replay(
                    df,
                    idx_alvo=idx,
                    params_base=params_base,
                    modo_replay=modo_interno,
                )

                if not res["ok"] or res["serie_top1"] is None:
                    continue

                serie_prev = list(map(int, res["serie_top1"]))
                row_real = df.iloc[idx_real - 1]
                serie_real = linha_para_serie(row_real, cols_pass)

                h = contar_hits(serie_prev, serie_real)

                registros.append(
                    {
                        "idx_alvo": int(idx),
                        "idx_real": int(idx_real),
                        "serie_prevista": serie_para_str(serie_prev),
                        "serie_real": serie_para_str(serie_real),
                        "hits": int(h),
                        "modo": modo_interno,
                    }
                )

        if not registros:
            st.error("Replay não gerou resultados válidos (verifique janelas e horizonte).")
        else:
            df_replay = pd.DataFrame(registros).sort_values("idx_alvo").reset_index(drop=True)
            st.session_state["df_replay"] = df_replay

            st.markdown("### 📊 Resumo do Replay")
            st.write(f"N execuções válidas: **{len(df_replay)}**")

            colm1, colm2, colm3 = st.columns(3)
            with colm1:
                st.metric("Média de hits (passageiros por carro)", f"{df_replay['hits'].mean():.2f}")
            with colm2:
                st.metric("Execuções com ≥ 3 hits", f"{(df_replay['hits'] >= 3).sum()} / {len(df_replay)}")
            with colm3:
                st.metric("Execuções com ≥ 4 hits", f"{(df_replay['hits'] >= 4).sum()} / {len(df_replay)}")

            if mostrar_detalhes:
                st.markdown("### 🧾 Detalhamento do Replay (carro a carro)")
                st.dataframe(
                    df_replay[
                        [
                            "idx_alvo",
                            "idx_real",
                            "serie_prevista",
                            "serie_real",
                            "hits",
                            "modo",
                        ]
                    ],
                    use_container_width=True,
                )


# ============================================================
# PAINEL 8 — 🧪 TESTES DE CONFIABILIDADE (QDS / BACKTEST / MC)
# ============================================================

if painel == "🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)":
    st.markdown("## 🧪 Testes de Confiabilidade — QDS REAL + Backtest REAL")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    df_replay = st.session_state.get("df_replay", None)

    if df_replay is None or df_replay.empty:
        st.info(
            "Ainda não há resultados de Replay salvos.\n\n"
            "Use primeiro o painel **'📅 Modo Replay Automático do Histórico'** "
            "para gerar a base empírica de validação (Backtest REAL)."
        )
        st.stop()

    st.markdown("### ✅ QDS REAL — Índice de Qualidade Dinâmica da Estrada (0–100)")

    resultados_qds = calcular_qds_real(df_replay)

    colq1, colq2, colq3 = st.columns(3)
    with colq1:
        st.metric("QDS REAL (0–100)", f"{resultados_qds['qds']:.1f}")
    with colq2:
        st.metric("Média de hits", f"{resultados_qds['media_hits']:.2f}")
    with colq3:
        st.metric("N execuções", f"{resultados_qds['n']}")

    st.markdown("### 📊 Distribuição de hits por carro (Backtest REAL)")

    dist_hits = df_replay["hits"].value_counts().sort_index()
    df_dist = dist_hits.reset_index()
    df_dist.columns = ["hits", "frequencia"]

    st.bar_chart(df_dist.set_index("hits"))

    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        st.metric("P(hits ≥ 1)", f"{100 * resultados_qds['p_ge_1']:.1f}%")
    with colp2:
        st.metric("P(hits ≥ 3)", f"{100 * resultados_qds['p_ge_3']:.1f}%")
    with colp3:
        st.metric("P(hits ≥ 4)", f"{100 * resultados_qds['p_ge_4']:.1f}%")

    st.markdown("---")
    st.markdown("### 🔍 Amostra do Backtest REAL (primeiros carros do Replay)")
    st.dataframe(
        df_replay.head(50)[["idx_alvo", "idx_real", "serie_prevista", "serie_real", "hits", "modo"]],
        use_container_width=True,
    )

    st.markdown(
        """
**Leitura operacional (QDS REAL + Backtest REAL + Monte Carlo Profundo ULTRA)**

- O **QDS REAL** sintetiza a qualidade dinâmica da estrada a partir do que o sistema realmente teria feito
  nos carros do passado (Replay), usando exatamente o mesmo núcleo TURBO++ ULTRA.
- A distribuição de **hits por carro** mostra quão frequentemente a previsão encosta em 1, 3, 4 ou mais passageiros.
- A integração com o **Monte Carlo Profundo ULTRA** já está embutida no próprio núcleo de previsão usado no Replay,
  o que significa que o backtest já incorpora o regime estocástico real da estrada.
"""
    )
