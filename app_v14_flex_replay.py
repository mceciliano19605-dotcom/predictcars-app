import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# ============================================================
# CONFIGURAÇÃO GERAL DO APP FLEX REPLAY
# ============================================================

st.set_page_config(
    page_title="Predict Cars V14-FLEX REPLAY",
    page_icon="🚗",
    layout="wide",
)

# ============================================================
# FUNÇÕES DE APOIO — HISTÓRICO FLEXÍVEL
# ============================================================

def _detectar_sep(linha: str) -> str:
    if ";" in linha:
        return ";"
    if "," in linha:
        return ","
    return ";"


def preparar_historico_flex(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza o histórico para o formato FLEX:
    - Detecta coluna de série (ID).
    - Detecta opcionalmente coluna k.
    - Todas as demais colunas são passageiros (p1..pN).
    - Cria coluna idx (1..N).
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()
    # remove colunas Unnamed
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]

    cols = list(df.columns)
    cols_lower = [str(c).lower() for c in cols]

    # Detecta coluna de série
    id_idx = None
    for i, c in enumerate(cols_lower):
        if ("serie" in c) or c.startswith("c") or ("id" in c):
            id_idx = i
            break
    if id_idx is None:
        id_idx = 0
    id_col = cols[id_idx]

    # Detecta coluna k (opcional)
    k_idx = None
    for i, c in enumerate(cols_lower):
        if i == id_idx:
            continue
        if ("k" in c) or ("risco" in c):
            k_idx = i
            break
    k_col = cols[k_idx] if k_idx is not None else None

    # Passageiros = demais colunas
    passageiro_cols = [c for c in cols if c not in {id_col, k_col}]

    # Monta DF normalizado
    df_norm = pd.DataFrame()
    df_norm["serie"] = df[id_col].astype(str)

    for i, c in enumerate(passageiro_cols):
        pname = f"p{i+1}"
        df_norm[pname] = pd.to_numeric(df[c], errors="coerce")

    if k_col is not None:
        df_norm["k"] = pd.to_numeric(df[k_col], errors="coerce").fillna(0).astype(int)
    else:
        df_norm["k"] = 0

    df_norm["idx"] = np.arange(1, len(df_norm) + 1)
    df_norm = df_norm.sort_values("idx").reset_index(drop=True)
    return df_norm


def parse_historico_texto_flex(conteudo: str) -> pd.DataFrame:
    """
    Converte texto colado em DataFrame FLEX:
    Formato típico:
        C1;41;5;4;52;30;33;0
    Mas aceita número variável de passageiros.
    Regra:
        - 1ª coluna: ID da série
        - Se houver >=3 números após o ID: último é k, demais são passageiros
        - Se houver 1 ou 2 números: todos são passageiros, k=0
    """
    linhas = [l.strip() for l in conteudo.strip().splitlines() if l.strip()]
    if not linhas:
        raise ValueError("Nenhuma linha válida encontrada.")

    sep = _detectar_sep(linhas[0])

    registros: List[Dict[str, Any]] = []
    max_pass = 0

    # 1º passe: medir máximo de passageiros
    temp = []
    for linha in linhas:
        partes = [p.strip() for p in linha.split(sep) if p.strip()]
        if len(partes) < 2:
            continue
        serie = partes[0]
        nums = partes[1:]
        if len(nums) >= 3:
            k_val = nums[-1]
            passageiros = nums[:-1]
        else:
            k_val = "0"
            passageiros = nums
        max_pass = max(max_pass, len(passageiros))
        temp.append((serie, passageiros, k_val))

    if max_pass == 0:
        raise ValueError("Não foi possível detectar passageiros no histórico.")

    # 2º passe: normalizar com p1..pN
    for serie, passageiros, k_val in temp:
        # preenche faltantes repetindo o último
        if not passageiros:
            passageiros = ["0"]
        while len(passageiros) < max_pass:
            passageiros.append(passageiros[-1])
        passageiros = passageiros[:max_pass]

        reg: Dict[str, Any] = {"serie": serie}
        for i, v in enumerate(passageiros):
            reg[f"p{i+1}"] = int(v)
        reg["k"] = int(k_val) if k_val not in ("", None) else 0
        registros.append(reg)

    df_raw = pd.DataFrame(registros)
    df_raw["idx"] = np.arange(1, len(df_raw) + 1)
    cols = ["serie"] + [c for c in df_raw.columns if c.startswith("p")] + ["k", "idx"]
    df_raw = df_raw[cols]
    return df_raw

# ============================================================
# k HISTÓRICO — SENTINELA REATIVO (NÃO INFLUENCIA PREVISÃO)
# ============================================================

def classificar_k_valor(k_val: int) -> str:
    if k_val <= 0:
        return "estavel"
    if k_val == 1:
        return "atencao"
    return "critico"


def estado_k_global(df: pd.DataFrame) -> dict:
    if df.empty or "k" not in df.columns:
        return {
            "estado": "indefinido",
            "contagens": {"estavel": 0, "atencao": 0, "critico": 0},
        }

    estados = df["k"].apply(classificar_k_valor)
    contagens = estados.value_counts().to_dict()
    for chave in ["estavel", "atencao", "critico"]:
        contagens.setdefault(chave, 0)

    if contagens["critico"] > 0:
        estado = "critico"
    elif contagens["atencao"] > 0:
        estado = "atencao"
    else:
        estado = "estavel"

    return {"estado": estado, "contagens": contagens}


def rotulo_k(estado: str) -> str:
    if estado == "estavel":
        return "🟢 Ambiente estável — previsão em regime normal."
    if estado == "atencao":
        return "🟡 Pré-ruptura — usar previsão com atenção."
    if estado == "critico":
        return "🔴 Ambiente crítico — usar previsão com cautela máxima."
    return "⚪ Estado de risco indefinido."

# ============================================================
# k* TURBO++ — SENTINELA PREDITIVO (FLEX)
# ============================================================

def _cols_passageiros(row_or_df) -> List[str]:
    if isinstance(row_or_df, pd.Series):
        cols = [c for c in row_or_df.index if str(c).startswith("p")]
    else:
        cols = [c for c in row_or_df.columns if str(c).startswith("p")]
    cols = sorted(cols, key=lambda x: int(str(x)[1:]) if str(x)[1:].isdigit() else 999)
    return cols


def _extrair_vetor_passageiros(df_linha: pd.Series) -> np.ndarray:
    cols = _cols_passageiros(df_linha)
    vals = [df_linha[c] for c in cols]
    return np.array(vals, dtype=float)


def calcular_metricas_risco_janela(df_janela: pd.DataFrame) -> dict:
    if df_janela is None or df_janela.empty or len(df_janela) < 3:
        return {"turbulencia": 0.0, "dispersao": 0.0}

    diffs = []
    linhas = df_janela.sort_values("idx").reset_index(drop=True)
    for i in range(1, len(linhas)):
        v1 = _extrair_vetor_passageiros(linhas.iloc[i - 1])
        v2 = _extrair_vetor_passageiros(linhas.iloc[i])
        diffs.append(np.linalg.norm(v2 - v1))
    turbulencia = float(np.mean(diffs)) if diffs else 0.0

    mat = np.vstack([_extrair_vetor_passageiros(linhas.iloc[i])
                     for i in range(len(linhas))])
    dispersao = float(np.mean(np.var(mat, axis=0)))

    return {"turbulencia": turbulencia, "dispersao": dispersao}


def normalizar_score(valor: float, referencia_baixa: float, referencia_alta: float) -> float:
    if referencia_alta <= referencia_baixa:
        return 0.0
    score = (valor - referencia_baixa) / (referencia_alta - referencia_baixa)
    score = max(0.0, min(1.0, score))
    return score


def calcular_k_star_estado(
    df: pd.DataFrame,
    idx_alvo: int,
    largura_janela: int = 40,
) -> dict:
    if df.empty:
        return {"estado": "indefinido", "score": 0.0}

    idx_alvo = int(idx_alvo)
    idx_min = int(df["idx"].min())
    idx_max = int(df["idx"].max())
    idx_alvo = max(idx_min + 1, min(idx_alvo, idx_max))

    inicio = max(idx_min, idx_alvo - largura_janela)
    df_janela = df[(df["idx"] >= inicio) & (df["idx"] < idx_alvo)].copy()
    if df_janela.empty:
        return {"estado": "indefinido", "score": 0.0}

    met = calcular_metricas_risco_janela(df_janela)

    score_turb = normalizar_score(met["turbulencia"], 0.0, 60.0)
    score_disp = normalizar_score(met["dispersao"], 0.0, 300.0)

    score = 0.6 * score_turb + 0.4 * score_disp

    if score < 0.33:
        estado = "estavel"
    elif score < 0.66:
        estado = "atencao"
    else:
        estado = "critico"

    return {"estado": estado, "score": float(score)}


def rotulo_k_star(estado: str, score: float) -> str:
    score_pct = int(round(score * 100))
    if estado == "estavel":
        return f"🟢 k*: Ambiente tende a permanecer estável (risco ≈ {score_pct}%)."
    if estado == "atencao":
        return f"🟡 k*: Pré-ruptura estrutural detectada (risco ≈ {score_pct}%)."
    if estado == "critico":
        return f"🔴 k*: Alta probabilidade de ruptura ou turbulência forte (risco ≈ {score_pct}%)."
    return "⚪ k*: Estado preditivo indefinido."

# ============================================================
# MOTOR TURBO++ FLEX — IDX / IPF / IPO / S6
# ============================================================

def construir_janelas(df: pd.DataFrame, tamanho_janela: int) -> pd.DataFrame:
    """
    Constrói janelas IDX:
    - contexto: sequência de tamanho_janela
    - seguidor: série imediatamente seguinte (p1..pN) e k histórico dessa série.
    """
    linhas = df.sort_values("idx").reset_index(drop=True)
    janelas = []
    max_idx = int(linhas["idx"].max())
    min_idx = int(linhas["idx"].min())

    for i in range(min_idx, max_idx - tamanho_janela):
        inicio = i
        fim = i + tamanho_janela - 1
        seguidor_idx = fim + 1
        if seguidor_idx > max_idx:
            break

        contexto = linhas[(linhas["idx"] >= inicio) & (linhas["idx"] <= fim)]
        seguidor = linhas[linhas["idx"] == seguidor_idx]
        if contexto.empty or seguidor.empty:
            continue

        contexto_mat = np.vstack([_extrair_vetor_passageiros(l) for _, l in contexto.iterrows()])
        seg_vec = _extrair_vetor_passageiros(seguidor.iloc[0])

        janelas.append(
            {
                "inicio_idx": inicio,
                "fim_idx": fim,
                "seguidor_idx": seguidor_idx,
                "contexto_mat": contexto_mat,
                "seguidor_vec": seg_vec,
                "seguidor_serie": list(seg_vec.astype(int)),
                "seguidor_k": int(seguidor.iloc[0].get("k", 0)),
            }
        )
    return pd.DataFrame(janelas)


def vetorizar_contexto_atual(df: pd.DataFrame, idx_alvo: int, tamanho_janela: int) -> np.ndarray:
    """
    Constrói o contexto atual: janela imediatamente antes do idx_alvo.
    """
    idx_alvo = int(idx_alvo)
    idx_min = int(df["idx"].min())
    idx_max = int(df["idx"].max())
    idx_alvo = max(idx_min + 1, min(idx_alvo, idx_max))

    fim = idx_alvo - 1
    inicio = max(idx_min, fim - tamanho_janela + 1)

    contexto = df[(df["idx"] >= inicio) & (df["idx"] <= fim)].sort_values("idx")
    if contexto.empty or len(contexto) < 2:
        return None

    contexto_mat = np.vstack([_extrair_vetor_passageiros(l) for _, l in contexto.iterrows()])
    return contexto_mat


def calcular_similaridade_janelas(janelas_df: pd.DataFrame, contexto_atual: np.ndarray) -> pd.DataFrame:
    """
    IDX: mede similaridade entre contexto atual e cada janela histórica.
    """
    if janelas_df.empty or contexto_atual is None:
        return pd.DataFrame()

    scores = []
    for idx, row in janelas_df.iterrows():
        mat = row["contexto_mat"]
        min_len = min(mat.shape[0], contexto_atual.shape[0])
        if min_len <= 0:
            continue

        ca = contexto_atual[-min_len:, :]
        cb = mat[-min_len:, :]

        dif = ca - cb
        dist = np.linalg.norm(dif)
        sim = 1.0 / (1.0 + dist)
        scores.append((idx, sim))

    if not scores:
        return pd.DataFrame()

    idxs, sims = zip(*scores)
    janelas_df = janelas_df.loc[list(idxs)].copy()
    janelas_df["score_idx"] = sims
    janelas_df = janelas_df.sort_values("score_idx", ascending=False).reset_index(drop=True)
    return janelas_df


def gerar_leque_original(df: pd.DataFrame, idx_alvo: int,
                         tamanho_janela: int = 10, top_janelas: int = 40) -> pd.DataFrame:
    """
    Leque ORIGINAL (IPF bruto): pega seguidores das janelas mais semelhantes (IDX).
    """
    if df.empty:
        return pd.DataFrame()

    janelas = construir_janelas(df, tamanho_janela)
    contexto_atual = vetorizar_contexto_atual(df, idx_alvo, tamanho_janela)
    if janelas.empty or contexto_atual is None:
        return pd.DataFrame()

    janelas_sim = calcular_similaridade_janelas(janelas, contexto_atual)
    if janelas_sim.empty:
        return pd.DataFrame()

    janelas_top = janelas_sim.head(top_janelas).copy()

    registros = []
    for _, row in janelas_top.iterrows():
        registros.append(
            {
                "inicio_idx": row["inicio_idx"],
                "fim_idx": row["fim_idx"],
                "seguidor_idx": row["seguidor_idx"],
                "series": row["seguidor_serie"],
                "score_idx": row["score_idx"],
                "k_hist": row["seguidor_k"],  # apenas informativo
            }
        )

    leque = pd.DataFrame(registros)
    return leque


def corrigir_series(series: list) -> list:
    """
    Correções estruturais simples:
    - remove fora de [1, 60]
    - ordena
    (mantém tamanho flexível)
    """
    nums = [int(x) for x in series if 1 <= int(x) <= 60]
    if not nums:
        nums = [1]
    nums = sorted(nums)
    return nums


def gerar_leque_corrigido(leque_original: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica correções estruturais => IPO simplificado.
    """
    if leque_original.empty:
        return pd.DataFrame()

    leque = leque_original.copy()
    leque["series_corr"] = leque["series"].apply(corrigir_series)
    leque["score_ipo"] = leque["score_idx"]

    return leque


def s6_profundo_flat(leque: pd.DataFrame) -> pd.DataFrame:
    """
    S6 Profundo FLEX: achata as séries, agrupa e comprime por:
    - frequência
    - score médio (IDX/IPO)
    - penalização leve por dispersão alta
    """
    if leque.empty:
        return pd.DataFrame()

    registros = []
    for _, row in leque.iterrows():
        s = row.get("series_corr", row["series"])
        chave = tuple(s)
        registros.append(
            {
                "series": chave,
                "score_base": row.get("score_ipo", row.get("score_idx", 0.0)),
            }
        )

    df_flat = pd.DataFrame(registros)
    if df_flat.empty:
        return pd.DataFrame()

    def disp(t):
        arr = np.array(t, dtype=float)
        return float(np.var(arr))

    df_flat["freq"] = 1
    df_agg = (
        df_flat.groupby("series")
        .agg({"score_base": "mean", "freq": "sum"})
        .reset_index()
    )
    df_agg["disp"] = df_agg["series"].apply(disp)

    max_freq = df_agg["freq"].max() or 1.0
    max_disp = df_agg["disp"].max() or 1.0

    df_agg["score"] = (
        0.5 * df_agg["score_base"] +
        0.4 * (df_agg["freq"] / max_freq) -
        0.1 * df_agg["disp"].apply(
            lambda v: normalizar_score(v, 0.0, max_disp)
        )
    )

    df_agg = df_agg.sort_values("score", ascending=False).reset_index(drop=True)
    return df_agg[["series", "score", "freq", "disp"]]


def unir_leques(leque_original: pd.DataFrame, leque_corrigido: pd.DataFrame) -> pd.DataFrame:
    """
    Leque MISTO: une ORIGINAL + CORRIGIDO e reagrupa.
    """
    if leque_original.empty and leque_corrigido.empty:
        return pd.DataFrame()
    if leque_original.empty:
        base = leque_corrigido.copy()
        base["series_corr"] = base["series_corr"]
        base["score_mix"] = base["score_ipo"]
    elif leque_corrigido.empty:
        base = leque_original.copy()
        base["series_corr"] = base["series"]
        base["score_mix"] = base["score_idx"]
    else:
        lo = leque_original.copy()
        lc = leque_corrigido.copy()

        lo["series_corr"] = lo["series"]
        lo["score_mix"] = lo["score_idx"]

        lc["series_corr"] = lc["series_corr"]
        lc["score_mix"] = lc["score_ipo"]

        base = pd.concat([lo, lc], ignore_index=True)

    registros = []
    for _, row in base.iterrows():
        s = row["series_corr"]
        registros.append({"series": tuple(s), "score_base": row["score_mix"]})

    df_flat = pd.DataFrame(registros)
    if df_flat.empty:
        return pd.DataFrame()

    def disp(t):
        arr = np.array(t, dtype=float)
        return float(np.var(arr))

    df_flat["freq"] = 1
    df_agg = (
        df_flat.groupby("series")
        .agg({"score_base": "mean", "freq": "sum"})
        .reset_index()
    )
    df_agg["disp"] = df_agg["series"].apply(disp)

    max_freq = df_agg["freq"].max() or 1.0
    max_disp = df_agg["disp"].max() or 1.0

    df_agg["score"] = (
        0.5 * df_agg["score_base"] +
        0.4 * (df_agg["freq"] / max_freq) -
        0.1 * df_agg["disp"].apply(
            lambda v: normalizar_score(v, 0.0, max_disp)
        )
    )

    df_agg = df_agg.sort_values("score", ascending=False).reset_index(drop=True)
    return df_agg[["series", "score", "freq", "disp"]]


def executar_pipeline_v14_turbo_flex(
    df: pd.DataFrame,
    idx_alvo: int,
    tamanho_janela: int = 10,
    top_janelas: int = 40,
    n_series_final: int = 50,
) -> dict:
    """
    Motor V14 TURBO++ FLEX:
    - IDX: janelas por similaridade
    - IPF: leque ORIGINAL
    - IPO: leque CORRIGIDO
    - S6: achatamento profundo
    - MISTO: fusão final
    """
    resultado = {
        "leque_original": pd.DataFrame(),
        "leque_corrigido": pd.DataFrame(),
        "flat_original": pd.DataFrame(),
        "flat_corrigido": pd.DataFrame(),
        "flat_misto": pd.DataFrame(),
        "previsao_final": None,
    }

    if df.empty:
        return resultado

    leque_orig = gerar_leque_original(df, idx_alvo, tamanho_janela, top_janelas)
    if leque_orig.empty:
        return resultado

    leque_corr = gerar_leque_corrigido(leque_orig)

    flat_orig = s6_profundo_flat(leque_orig)
    flat_corr = s6_profundo_flat(leque_corr)
    flat_mix = unir_leques(leque_orig, leque_corr)

    # Limita quantidade final
    def limitar(df_flat):
        if df_flat is None or df_flat.empty:
            return pd.DataFrame()
        return df_flat.head(n_series_final).copy()

    flat_orig = limitar(flat_orig)
    flat_corr = limitar(flat_corr)
    flat_mix = limitar(flat_mix)

    previsao_final = None
    if flat_mix is not None and not flat_mix.empty:
        previsao_final = list(flat_mix.iloc[0]["series"])

    resultado.update(
        {
            "leque_original": leque_orig,
            "leque_corrigido": leque_corr,
            "flat_original": flat_orig,
            "flat_corrigido": flat_corr,
            "flat_misto": flat_mix,
            "previsao_final": previsao_final,
        }
    )
    return resultado

# ============================================================
# MÓDULOS DE VALIDAÇÃO — QDS / BACKTEST / MONTE CARLO
# ============================================================

def _contar_acertos(series_prev: List[int], series_real: List[int]) -> int:
    return len(set(series_prev) & set(series_real))


def executar_backtest_simples(
    df: pd.DataFrame,
    idx_inicio: int,
    idx_fim: int,
    tamanho_janela: int = 10,
    top_janelas: int = 40,
    n_series_final: int = 50,
    max_passos: int = 50,
) -> pd.DataFrame:
    """
    Backtest simples: roda o motor FLEX em vários idx_alvo históricos
    e mede acertos em relação à série real seguinte.
    """
    if df.empty:
        return pd.DataFrame()

    idx_min = int(df["idx"].min())
    idx_max = int(df["idx"].max())

    idx_inicio = max(idx_inicio, idx_min + 2)
    idx_fim = min(idx_fim, idx_max - 1)
    if idx_inicio >= idx_fim:
        return pd.DataFrame()

    resultados = []
    passos = 0

    for idx_alvo in range(idx_inicio, idx_fim + 1):
        if passos >= max_passos:
            break
        # série real é a própria idx_alvo
        serie_real_row = df[df["idx"] == idx_alvo]
        if serie_real_row.empty:
            continue
        # previsão usando contexto até idx_alvo-1
        res = executar_pipeline_v14_turbo_flex(
            df[df["idx"] < idx_alvo],
            idx_alvo=idx_alvo,
            tamanho_janela=tamanho_janela,
            top_janelas=top_janelas,
            n_series_final=n_series_final,
        )
        prev = res.get("previsao_final", None)
        if not prev:
            continue

        # extrai série real como lista de passageiros
        row = serie_real_row.iloc[0]
        cols_p = _cols_passageiros(serie_real_row)
        serie_real = [int(row[c]) for c in cols_p]

        acertos = _contar_acertos(prev, serie_real)

        resultados.append(
            {
                "idx_alvo": idx_alvo,
                "serie_real": serie_real,
                "previsao": prev,
                "acertos": acertos,
            }
        )
        passos += 1

    if not resultados:
        return pd.DataFrame()

    df_bt = pd.DataFrame(resultados)
    return df_bt


def resumo_qds(df_bt: pd.DataFrame) -> dict:
    """
    QDS simplificado: resume a distribuição de acertos.
    """
    if df_bt.empty:
        return {
            "n_teste": 0,
            "media_acertos": 0.0,
            "pct_0_1": 0.0,
            "pct_2_3": 0.0,
            "pct_4_plus": 0.0,
        }

    n = len(df_bt)
    media = float(df_bt["acertos"].mean())
    pct_0_1 = float((df_bt["acertos"] <= 1).mean() * 100.0)
    pct_2_3 = float(((df_bt["acertos"] >= 2) & (df_bt["acertos"] <= 3)).mean() * 100.0)
    pct_4_plus = float((df_bt["acertos"] >= 4).mean() * 100.0)

    return {
        "n_teste": n,
        "media_acertos": media,
        "pct_0_1": pct_0_1,
        "pct_2_3": pct_2_3,
        "pct_4_plus": pct_4_plus,
    }


def executar_monte_carlo(
    df: pd.DataFrame,
    idx_inicio: int,
    idx_fim: int,
    n_execucoes: int = 20,
    max_passos_backtest: int = 30,
) -> pd.DataFrame:
    """
    Monte Carlo TURBO++ simplificado:
    - Roda vários backtests variando levemente parâmetros de janela/top_janelas.
    - Mede estabilidade das métricas de acerto.
    """
    if df.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(42)
    registros = []

    for i in range(n_execucoes):
        tam_jan = int(rng.integers(8, 14))
        top_jan = int(rng.integers(30, 60))
        df_bt = executar_backtest_simples(
            df,
            idx_inicio=idx_inicio,
            idx_fim=idx_fim,
            tamanho_janela=tam_jan,
            top_janelas=top_jan,
            n_series_final=50,
            max_passos=max_passos_backtest,
        )
        q = resumo_qds(df_bt)
        registros.append(
            {
                "execucao": i + 1,
                "tamanho_janela": tam_jan,
                "top_janelas": top_jan,
                "n_teste": q["n_teste"],
                "media_acertos": q["media_acertos"],
                "pct_0_1": q["pct_0_1"],
                "pct_2_3": q["pct_2_3"],
                "pct_4_plus": q["pct_4_plus"],
            }
        )

    if not registros:
        return pd.DataFrame()
    return pd.DataFrame(registros)

# ============================================================
# LAYOUT — SIDEBAR E ESTADO GLOBAL
# ============================================================

st.sidebar.title("🚗 Predict Cars V14-FLEX REPLAY")
st.sidebar.markdown(
    "Versão FLEX: número variável de passageiros + modo replay automático + validação (QDS / Backtest / Monte Carlo)."
)

painel = st.sidebar.radio(
    "Escolha o painel:",
    [
        "📥 Histórico — Entrada",
        "🔍 Pipeline V14-FLEX (TURBO++)",
        "🚨 Monitor de Risco (k & k*)",
        "🚀 Modo TURBO++ — Painel Completo",
        "📅 Modo Replay Automático do Histórico",
        "🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)",
    ],
)

if "df" not in st.session_state:
    st.session_state["df"] = pd.DataFrame()
if "ultimo_idx_alvo" not in st.session_state:
    st.session_state["ultimo_idx_alvo"] = None
if "k_star_estado" not in st.session_state:
    st.session_state["k_star_estado"] = "indefinido"
if "k_star_score" not in st.session_state:
    st.session_state["k_star_score"] = 0.0
if "resultado_turbo" not in st.session_state:
    st.session_state["resultado_turbo"] = {}
if "backtest_df" not in st.session_state:
    st.session_state["backtest_df"] = pd.DataFrame()
if "monte_carlo_df" not in st.session_state:
    st.session_state["monte_carlo_df"] = pd.DataFrame()
if "replay_df" not in st.session_state:
    st.session_state["replay_df"] = pd.DataFrame()

df_sessao = st.session_state["df"]

# ============================================================
# PAINEL 1 — Histórico — Entrada (FLEX)
# ============================================================

if painel == "📥 Histórico — Entrada":
    st.markdown("## 📥 Histórico — Entrada (FLEX)")

    opc = st.radio(
        "Como deseja carregar o histórico?",
        ["Enviar arquivo CSV", "Copiar e colar o histórico"],
        horizontal=True,
    )

    if opc == "Enviar arquivo CSV":
        file = st.file_uploader("Selecione o arquivo CSV:", type=["csv"])
        if file is not None:
            try:
                df_raw = pd.read_csv(file)
                df = preparar_historico_flex(df_raw)
                st.session_state["df"] = df
                st.success("Histórico FLEX carregado com sucesso!")
            except Exception as e:
                st.error(f"Erro ao carregar CSV: {e}")
    else:
        conteudo = st.text_area(
            "Cole o histórico aqui (ex: C1;41;5;4;52;30;33;0):",
            height=250,
        )
        if st.button("Carregar histórico do texto (FLEX)"):
            try:
                df = parse_historico_texto_flex(conteudo)
                st.session_state["df"] = df
                st.success("Histórico FLEX carregado com sucesso a partir do texto!")
            except Exception as e:
                st.error(f"Erro ao interpretar o texto: {e}")

    df = st.session_state["df"]
    if df is not None and not df.empty:
        st.markdown("### 🔎 Prévia do histórico preparado (FLEX)")
        st.dataframe(df.head(30))

        info_k = estado_k_global(df)
        estado = info_k["estado"]
        contagens = info_k["contagens"]

        st.markdown("### ⚠️ Resumo de risco histórico (k)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Séries estáveis (k=0)", contagens["estavel"])
            st.metric("Séries em atenção (k=1)", contagens["atencao"])
            st.metric("Séries críticas (k≥2)", contagens["critico"])
        with col2:
            st.markdown("**Estado global (k histórico):**")
            if estado == "estavel":
                st.success("🟢 Estável")
            elif estado == "atencao":
                st.warning("🟡 Atenção")
            elif estado == "critico":
                st.error("🔴 Crítico")
            else:
                st.info("⚪ Indefinido")

        st.markdown("### 📈 Evolução de k histórico")
        graf = df[["idx", "k"]].set_index("idx")
        st.line_chart(graf)

        cols_pass = _cols_passageiros(df)
        st.markdown("### ℹ️ Estrutura FLEX detectada")
        st.write(f"Número de passageiros detectados: **{len(cols_pass)}**")
        st.write(f"Colunas de passageiros: {', '.join(cols_pass)}")
    else:
        st.info("Carregue o histórico para continuar.")

# ============================================================
# PAINEL 2 — Pipeline V14-FLEX (TURBO++)
# ============================================================

elif painel == "🔍 Pipeline V14-FLEX (TURBO++)":
    st.markdown("## 🔍 Pipeline V14-FLEX — Execução TURBO++ (núcleo)")

    df = st.session_state.get("df", pd.DataFrame())
    if df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        idx_min = int(df["idx"].min())
        idx_max = int(df["idx"].max())
        idx_alvo = st.number_input(
            "Selecione o índice alvo (idx):",
            min_value=idx_min + 1,
            max_value=idx_max,
            value=idx_max,
            step=1,
        )
    with col2:
        tamanho_janela = st.number_input(
            "Tamanho da janela IDX:",
            min_value=5,
            max_value=30,
            value=10,
            step=1,
        )
    with col3:
        top_janelas = st.number_input(
            "Qtd. janelas similares (IDX):",
            min_value=10,
            max_value=100,
            value=40,
            step=5,
        )

    n_series_final = st.slider(
        "Quantidade de séries finais (S6 Profundo):",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
    )

    if st.button("Executar Pipeline V14-FLEX TURBO++", type="primary"):
        resultado = executar_pipeline_v14_turbo_flex(
            df,
            idx_alvo,
            tamanho_janela=int(tamanho_janela),
            top_janelas=int(top_janelas),
            n_series_final=int(n_series_final),
        )
        st.session_state["resultado_turbo"] = resultado
        st.session_state["ultimo_idx_alvo"] = int(idx_alvo)

        kstar = calcular_k_star_estado(df, idx_alvo)
        st.session_state["k_star_estado"] = kstar["estado"]
        st.session_state["k_star_score"] = kstar["score"]

    resultado = st.session_state.get("resultado_turbo", {})
    leque_orig = resultado.get("leque_original", pd.DataFrame())
    leque_corr = resultado.get("leque_corrigido", pd.DataFrame())
    flat_mix = resultado.get("flat_misto", pd.DataFrame())
    previsao_final = resultado.get("previsao_final", None)

    if leque_orig is None or leque_orig.empty:
        st.info("Execute o pipeline TURBO++ para ver os resultados.")
        st.stop()

    st.markdown("### 📊 Leque ORIGINAL (IPF bruto)")
    st.dataframe(leque_orig[["seguidor_idx", "series", "score_idx", "k_hist"]])

    if leque_corr is not None and not leque_corr.empty:
        st.markdown("### 🔧 Leque CORRIGIDO (IPO simplificado)")
        st.dataframe(leque_corr[["seguidor_idx", "series_corr", "score_ipo"]])

    if flat_mix is not None and not flat_mix.empty:
        st.markdown("### 🧬 S6 Profundo — Leque MISTO (achado e ranqueado)")
        df_view = flat_mix.copy()
        df_view["series_str"] = df_view["series"].apply(lambda s: " ".join(str(x) for x in s))
        st.dataframe(df_view[["series_str", "score", "freq", "disp"]])

    if previsao_final:
        st.markdown("### 🎯 Núcleo TURBO++ FLEX (previsão bruta do motor)")
        st.code(" ".join(str(x) for x in previsao_final), language="text")

    idx_alvo_mem = st.session_state.get("ultimo_idx_alvo", None)
    serie_alvo = df[df["idx"] == idx_alvo_mem] if idx_alvo_mem is not None else pd.DataFrame()
    if not serie_alvo.empty:
        k_real = int(serie_alvo.iloc[0]["k"])
        estado_k_real = classificar_k_valor(k_real)
        st.markdown("### ⚠️ k histórico da série alvo")
        st.info(rotulo_k(estado_k_real))

    k_star_estado = st.session_state["k_star_estado"]
    k_star_score = st.session_state["k_star_score"]
    st.markdown("### ⚡ k* (sentinela preditivo TURBO++)")
    st.info(rotulo_k_star(k_star_estado, k_star_score))

# ============================================================
# PAINEL 3 — Monitor de Risco (k & k*)
# ============================================================

elif painel == "🚨 Monitor de Risco (k & k*)":
    st.markdown("## 🚨 Monitor de Risco (k & k*)")

    df = st.session_state.get("df", pd.DataFrame())
    if df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    info_k = estado_k_global(df)
    estado = info_k["estado"]
    contagens = info_k["contagens"]

    st.markdown("### 🧮 Distribuição de k histórico")
    col1, col2, col3 = st.columns(3)
    col1.metric("Estável (k=0)", contagens["estavel"])
    col2.metric("Atenção (k=1)", contagens["atencao"])
    col3.metric("Crítico (k≥2)", contagens["critico"])

    st.markdown("### 🌡️ Estado global (k histórico)")
    st.write(rotulo_k(estado))

    st.markdown("### 📈 Série temporal de k histórico")
    graf = df[["idx", "k"]].set_index("idx")
    st.line_chart(graf)

    st.markdown("### ⚡ Último k* calculado (sentinela preditivo)")
    k_star_estado = st.session_state.get("k_star_estado", "indefinido")
    k_star_score = st.session_state.get("k_star_score", 0.0)
    st.info(rotulo_k_star(k_star_estado, k_star_score))

    st.markdown("### 🔍 Últimas séries (com k histórico)")
    cols_pass = _cols_passageiros(df)
    vis_cols = ["idx", "serie"] + cols_pass + ["k"]
    st.dataframe(
        df[vis_cols]
        .tail(30)
        .sort_values("idx", ascending=False)
    )

    st.markdown("### ℹ️ Interpretação")
    st.markdown(
        """
        - **k histórico** (reativo): mede o que já aconteceu na estrada.
        - **k\\*** (sentinela preditivo): estima, pela estrutura, se o risco está subindo
          antes de o k real aparecer.
        """
    )

# ============================================================
# PAINEL 4 — Modo TURBO++ — Painel Completo (FLEX)
# ============================================================

elif painel == "🚀 Modo TURBO++ — Painel Completo":
    st.markdown("## 🚀 Modo TURBO++ — Painel Completo (FLEX)")

    df = st.session_state.get("df", pd.DataFrame())
    if df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    idx_alvo_mem = st.session_state.get("ultimo_idx_alvo", None)
    if idx_alvo_mem is None:
        idx_min = int(df["idx"].min()) + 1
        idx_max = int(df["idx"].max())
        idx_escolhido = st.number_input(
            "Índice alvo (idx) para o TURBO++:",
            min_value=idx_min,
            max_value=idx_max,
            value=idx_max,
            step=1,
        )
    else:
        st.markdown(f"**Índice alvo (idx) usado no pipeline TURBO++:** `{idx_alvo_mem}`")
        idx_escolhido = idx_alvo_mem

    n_series_final = st.slider(
        "Quantidade de séries finais (S6 Profundo - painel completo):",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
    )

    st.markdown("---")

    if st.button("Rodar TURBO++ FLEX (painel completo)"):
        resultado = executar_pipeline_v14_turbo_flex(
            df,
            idx_escolhido,
            tamanho_janela=10,
            top_janelas=40,
            n_series_final=int(n_series_final),
        )
        st.session_state["resultado_turbo"] = resultado
        st.session_state["ultimo_idx_alvo"] = int(idx_escolhido)

        kstar = calcular_k_star_estado(df, idx_escolhido)
        st.session_state["k_star_estado"] = kstar["estado"]
        st.session_state["k_star_score"] = kstar["score"]

    resultado = st.session_state.get("resultado_turbo", {})
    leque_orig = resultado.get("leque_original", pd.DataFrame())
    leque_corr = resultado.get("leque_corrigido", pd.DataFrame())
    flat_orig = resultado.get("flat_original", pd.DataFrame())
    flat_corr = resultado.get("flat_corrigido", pd.DataFrame())
    flat_mix = resultado.get("flat_misto", pd.DataFrame())
    previsao_final = resultado.get("previsao_final", None)

    if leque_orig is None or leque_orig.empty:
        st.info("Rode o TURBO++ para ver o painel completo.")
        st.stop()

    st.markdown("### 🧱 Leque ORIGINAL (IPF) — séries brutas por similaridade")
    st.dataframe(leque_orig[["seguidor_idx", "series", "score_idx", "k_hist"]])

    if leque_corr is not None and not leque_corr.empty:
        st.markdown("### 🔧 Leque CORRIGIDO (IPO) — séries estruturalmente ajustadas")
        st.dataframe(leque_corr[["seguidor_idx", "series_corr", "score_ipo"]])

    if flat_orig is not None and not flat_orig.empty:
        st.markdown("### 🧬 S6 Profundo — Leque ORIGINAL achatado")
        df_view = flat_orig.copy()
        df_view["series_str"] = df_view["series"].apply(lambda s: " ".join(str(x) for x in s))
        st.dataframe(df_view[["series_str", "score", "freq", "disp"]])

    if flat_corr is not None and not flat_corr.empty:
        st.markdown("### 🧬 S6 Profundo — Leque CORRIGIDO achatado")
        df_view = flat_corr.copy()
        df_view["series_str"] = df_view["series"].apply(lambda s: " ".join(str(x) for x in s))
        st.dataframe(df_view[["series_str", "score", "freq", "disp"]])

    if flat_mix is not None and not flat_mix.empty:
        st.markdown("### 🧬 S6 Profundo — Leque MISTO (ORIGINAL + CORRIGIDO)")
        df_view = flat_mix.copy()
        df_view["series_str"] = df_view["series"].apply(lambda s: " ".join(str(x) for x in s))
        st.dataframe(df_view[["series_str", "score", "freq", "disp"]])

    st.markdown("### 🎯 Previsão Final TURBO++ FLEX")
    if previsao_final:
        st.code(" ".join(str(x) for x in previsao_final), language="text")
    else:
        st.warning("Nenhuma previsão final pôde ser gerada.")

    idx_alvo_mem = st.session_state.get("ultimo_idx_alvo", None)
    serie_alvo = df[df["idx"] == idx_alvo_mem] if idx_alvo_mem is not None else pd.DataFrame()
    if not serie_alvo.empty:
        k_real = int(serie_alvo.iloc[0]["k"])
        estado_k_real = classificar_k_valor(k_real)
        st.markdown("### 🔁 Comparação com k histórico da série alvo")
        st.info("k histórico (real): " + rotulo_k(estado_k_real))

    k_star_estado = st.session_state["k_star_estado"]
    k_star_score = st.session_state["k_star_score"]
    st.markdown("### ⚡ Contexto de risco (k* preditivo)")
    st.info(rotulo_k_star(k_star_estado, k_star_score))

    st.markdown("---")
    st.markdown(
        """
        ℹ️ Notas:
        - O k histórico e o k* preditivo NUNCA entram na matemática da previsão.
        - A previsão é 100% baseada na estrutura dos passageiros (n FLEX).
        - k e k* servem apenas como contexto de risco para interpretação.
        """
    )

# ============================================================
# PAINEL 5 — Modo Replay Automático do Histórico
# ============================================================

elif painel == "📅 Modo Replay Automático do Histórico":
    st.markdown("## 📅 Modo Replay Automático do Histórico")

    df = st.session_state.get("df", pd.DataFrame())
    if df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    idx_min = int(df["idx"].min())
    idx_max = int(df["idx"].max())

    st.markdown("### 🔧 Faixa de índices para o Replay (scanner do passado)")
    c1, c2 = st.columns(2)
    with c1:
        idx_inicio = st.number_input(
            "Índice inicial (primeiro ponto a ser reconstituído):",
            min_value=idx_min + 2,
            max_value=idx_max - 1,
            value=max(idx_min + 2, idx_max - 100),
            step=1,
        )
    with c2:
        idx_fim = st.number_input(
            "Índice final (último ponto a ser reconstituído):",
            min_value=idx_min + 3,
            max_value=idx_max,
            value=idx_max,
            step=1,
        )

    max_passos_replay = st.slider(
        "Quantidade máxima de pontos a reconstituir (Replay):",
        min_value=10,
        max_value=300,
        value=100,
        step=10,
    )

    st.markdown("### ⚙️ Parâmetros do motor durante o Replay")
    c3, c4 = st.columns(2)
    with c3:
        tamanho_janela_bt = st.number_input(
            "Tamanho da janela IDX (Replay):",
            min_value=5,
            max_value=30,
            value=10,
            step=1,
        )
    with c4:
        top_janelas_bt = st.number_input(
            "Qtd. janelas similares (Replay):",
            min_value=10,
            max_value=100,
            value=40,
            step=5,
        )

    st.markdown("---")

    if st.button("Rodar Replay Automático do Histórico"):
        df_replay = executar_backtest_simples(
            df,
            idx_inicio=int(idx_inicio),
            idx_fim=int(idx_fim),
            tamanho_janela=int(tamanho_janela_bt),
            top_janelas=int(top_janelas_bt),
            n_series_final=50,
            max_passos=int(max_passos_replay),
        )
        st.session_state["replay_df"] = df_replay

        if df_replay.empty:
            st.warning("Replay não gerou resultados (faixa muito pequena ou parâmetros muito restritivos).")
        else:
            st.success(f"Replay executado com {len(df_replay)} pontos reconstituídos.")

    df_replay = st.session_state.get("replay_df", pd.DataFrame())
    if df_replay is not None and not df_replay.empty:
        st.markdown("### 🧭 Replay Histórico — ponto a ponto")
        df_view = df_replay.copy()
        df_view["serie_real_str"] = df_view["serie_real"].apply(lambda s: " ".join(str(x) for x in s))
        df_view["previsao_str"] = df_view["previsao"].apply(lambda s: " ".join(str(x) for x in s))
        st.dataframe(df_view[["idx_alvo", "serie_real_str", "previsao_str", "acertos"]])

        st.markdown("### 📊 Resumo do Replay (QDS sobre o trecho reconstituído)")
        q = resumo_qds(df_replay)
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Nº pontos reconstituídos", q["n_teste"])
        col2.metric("Média de acertos", f"{q['media_acertos']:.2f}")
        col3.metric("% com 0–1 acertos", f"{q['pct_0_1']:.1f}%")
        col4.metric("% com 2–3 acertos", f"{q['pct_2_3']:.1f}%")
        col5.metric("% com ≥4 acertos", f"{q['pct_4_plus']:.1f}%")

        st.markdown(
            """
            ℹ️ Interpretação rápida do Replay:
            - Cada linha representa um dia passado que foi "revivido" como se o motor estivesse naquele ponto.
            - A coluna de acertos mostra quanto o motor teria acertado na série real daquele dia.
            - O resumo QDS acima indica a qualidade média desse trecho da estrada para o seu motor V14-FLEX.
            """
        )
    else:
        st.info("Defina a faixa de índices e rode o Replay para ver os resultados.")

# ============================================================
# PAINEL 6 — Testes de Confiabilidade (QDS / Backtest / Monte Carlo)
# ============================================================

elif painel == "🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)":
    st.markdown("## 🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)")

    df = st.session_state.get("df", pd.DataFrame())
    if df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    idx_min = int(df["idx"].min())
    idx_max = int(df["idx"].max())

    st.markdown("### 🔧 Faixa de índices para teste")
    c1, c2 = st.columns(2)
    with c1:
        idx_inicio = st.number_input(
            "Índice inicial (para backtest / QDS):",
            min_value=idx_min + 2,
            max_value=idx_max - 1,
            value=max(idx_min + 2, idx_max - 50),
            step=1,
        )
    with c2:
        idx_fim = st.number_input(
            "Índice final (para backtest / QDS):",
            min_value=idx_min + 3,
            max_value=idx_max,
            value=idx_max,
            step=1,
        )

    max_passos = st.slider(
        "Máximo de pontos testados (para não ficar pesado):",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
    )

    st.markdown("### 🧮 Parâmetros do motor TURBO++ para o teste")
    c3, c4 = st.columns(2)
    with c3:
        tamanho_janela_bt = st.number_input(
            "Tamanho da janela IDX (backtest/QDS):",
            min_value=5,
            max_value=30,
            value=10,
            step=1,
        )
    with c4:
        top_janelas_bt = st.number_input(
            "Qtd. janelas similares (backtest/QDS):",
            min_value=10,
            max_value=100,
            value=40,
            step=5,
        )

    if st.button("Rodar Backtest + QDS (A + B)"):
        df_bt = executar_backtest_simples(
            df,
            idx_inicio=int(idx_inicio),
            idx_fim=int(idx_fim),
            tamanho_janela=int(tamanho_janela_bt),
            top_janelas=int(top_janelas_bt),
            n_series_final=50,
            max_passos=int(max_passos),
        )
        st.session_state["backtest_df"] = df_bt

        if df_bt.empty:
            st.warning("Backtest não gerou resultados (poucos dados ou parâmetros muito restritivos).")
        else:
            st.success(f"Backtest executado com {len(df_bt)} pontos de teste.")

    df_bt = st.session_state.get("backtest_df", pd.DataFrame())
    if df_bt is not None and not df_bt.empty:
        st.markdown("### (B) Backtest automático — resultados detalhados")
        df_view = df_bt.copy()
        df_view["serie_real_str"] = df_view["serie_real"].apply(lambda s: " ".join(str(x) for x in s))
        df_view["previsao_str"] = df_view["previsao"].apply(lambda s: " ".join(str(x) for x in s))
        st.dataframe(df_view[["idx_alvo", "serie_real_str", "previsao_str", "acertos"]])

        q = resumo_qds(df_bt)
        st.markdown("### (A) QDS V14 — Confiabilidade por série (resumo)")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Nº séries testadas", q["n_teste"])
        col2.metric("Média de acertos", f"{q['media_acertos']:.2f}")
        col3.metric("% com 0–1 acertos", f"{q['pct_0_1']:.1f}%")
        col4.metric("% com 2–3 acertos", f"{q['pct_2_3']:.1f}%")
        col5.metric("% com ≥4 acertos", f"{q['pct_4_plus']:.1f}%")

    st.markdown("---")
    st.markdown("### (C) Monte Carlo TURBO++ — estabilidade do modelo")

    n_execucoes = st.slider(
        "Nº de execuções Monte Carlo:",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
    )
    max_passos_mc = st.slider(
        "Máx. pontos de backtest por execução:",
        min_value=10,
        max_value=100,
        value=30,
        step=10,
    )

    if st.button("Rodar Monte Carlo TURBO++ (C)"):
        df_mc = executar_monte_carlo(
            df,
            idx_inicio=int(idx_inicio),
            idx_fim=int(idx_fim),
            n_execucoes=int(n_execucoes),
            max_passos_backtest=int(max_passos_mc),
        )
        st.session_state["monte_carlo_df"] = df_mc

        if df_mc.empty:
            st.warning("Monte Carlo não gerou resultados.")
        else:
            st.success(f"Monte Carlo executado com {len(df_mc)} execuções.")

    df_mc = st.session_state.get("monte_carlo_df", pd.DataFrame())
    if df_mc is not None and not df_mc.empty:
        st.markdown("### (D) Tudo junto — visão consolidada das execuções Monte Carlo")

        st.dataframe(
            df_mc[
                [
                    "execucao",
                    "tamanho_janela",
                    "top_janelas",
                    "n_teste",
                    "media_acertos",
                    "pct_0_1",
                    "pct_2_3",
                    "pct_4_plus",
                ]
            ]
        )

        st.markdown("#### Resumo estatístico das execuções (Monte Carlo)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Média geral de acertos", f"{df_mc['media_acertos'].mean():.2f}")
        col2.metric("% médio com 0–1 acertos", f"{df_mc['pct_0_1'].mean():.1f}%")
        col3.metric("% médio com 2–3 acertos", f"{df_mc['pct_2_3'].mean():.1f}%")
        col4.metric("% médio com ≥4 acertos", f"{df_mc['pct_4_plus'].mean():.1f}%")

        st.markdown(
            """
            ℹ️ Interpretação rápida:
            - Se a média de acertos se mantém estável entre execuções, o modelo é mais robusto.
            - Se o % de ≥4 acertos varia muito, o modelo é sensível a pequenos ajustes de parâmetros.
            - Use estes sinais para julgar a confiabilidade estrutural do motor V14-FLEX.
            """
        )
