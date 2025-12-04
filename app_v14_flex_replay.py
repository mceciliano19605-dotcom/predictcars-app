# app_v14_flex_replay.py
# Predict Cars V14-FLEX ULTRA REAL (TURBO++)
# Versão COMPLETA e FINAL — sem simplificações
# Inclui: Entrada FLEX, Barômetro ULTRA, k*, IDX ULTRA, IPF/IPO,
# S6 Profundo ULTRA, Micro-Leque ULTRA, Monte Carlo ULTRA (patch),
# Fusão ULTRA, Replay LIGHT, Replay ULTRA, QDS REAL, Modo TURBO++ ULTRA.

import itertools
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st


# ------------------------------ Página ------------------------------

st.set_page_config(
    page_title="Predict Cars V14-FLEX ULTRA REAL (TURBO++)",
    layout="wide",
)

st.markdown(
    """
# Predict Cars V14-FLEX REPLAY — ULTRA REAL (TURBO++)
Versão FLEX: número **variável** de passageiros + Modo TURBO++ ULTRA + Replay + QDS/Backtest/Monte Carlo.
"""
)

# -------------------------------------------------------------------
# DATACLASSES AUXILIARES
# -------------------------------------------------------------------

@dataclass
class BarometroEstado:
    regime: str
    volatilidade_k: float
    media_k: float
    msg: str

@dataclass
class KStarEstado:
    k_star_pct: float
    estado: str
    msg: str

@dataclass
class IDXInfo:
    pesos_posicionais: np.ndarray
    freq_global: Dict[int, float]
    freq_posicional: Dict[int, Dict[int, float]]  # pos -> {valor: freq}

@dataclass
class IPFIPOInfo:
    ipf: List[int]
    ipo: List[int]
    ipo_asb: List[int]


# -------------------------------------------------------------------
# SUPORTE BÁSICO
# -------------------------------------------------------------------

def detectar_num_passageiros(df_raw: pd.DataFrame) -> int:
    if df_raw.shape[1] < 3:
        raise ValueError("Histórico inválido.")
    return df_raw.shape[1] - 2  # ID + passageiros + k


def preparar_historico_v14_flex(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.dropna(axis=1, how="all").copy()
    if df.shape[1] < 3:
        raise ValueError("Histórico inválido após limpeza.")

    n_pass = detectar_num_passageiros(df)
    colunas = ["ID"] + [f"P{i+1}" for i in range(n_pass)] + ["k"]
    df.columns = colunas

    df["ID"] = df["ID"].astype(str).str.strip()
    for c in colunas[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    df["k"] = df["k"].astype(int)
    return df


def obter_num_passageiros(df: pd.DataFrame) -> int:
    return len([c for c in df.columns if c.startswith("P")])


def extrair_series_passageiros(df: pd.DataFrame) -> np.ndarray:
    n = obter_num_passageiros(df)
    cols = [f"P{i+1}" for i in range(n)]
    return df[cols].to_numpy(dtype=int)


def obter_faixa_numerica(df: pd.DataFrame) -> Tuple[int, int]:
    mat = extrair_series_passageiros(df)
    return int(mat.min()), int(mat.max())


# -------------------------------------------------------------------
# BARÔMETRO ULTRA REAL
# -------------------------------------------------------------------

def analisar_barometro_ultra(df: pd.DataFrame, janela: int = 40) -> BarometroEstado:
    k_series = df["k"].astype(float)
    if len(k_series) < 5:
        return BarometroEstado("indefinido", np.nan, np.nan, "Histórico curto.")

    recent = k_series.iloc[-min(janela, len(k_series)):]
    media_k = float(recent.mean())
    vol_k = float(recent.std(ddof=0))

    if vol_k < 0.5 and media_k < 0.5:
        return BarometroEstado("estavel", vol_k, media_k,
            "🟢 Estrada estável — poucos guardas acertando o carro.")
    elif vol_k < 1.0 and media_k < 1.5:
        return BarometroEstado("transicao", vol_k, media_k,
            "🟡 Transição — guardas começando a acertar.")
    elif vol_k < 2.0 or media_k < 3.0:
        return BarometroEstado("turbulento", vol_k, media_k,
            "🟠 Turbulência — blocos previsíveis surgindo.")
    else:
        return BarometroEstado("ruptura", vol_k, media_k,
            "🔴 Ruptura — forte repetição de padrões.")


# -------------------------------------------------------------------
# k* ULTRA REAL
# -------------------------------------------------------------------

def calcular_kstar_ultra(df: pd.DataFrame, janela: int = 80) -> KStarEstado:
    k_series = df["k"].astype(float)
    recent = k_series.iloc[-min(janela, len(k_series)):]
    pct = float((recent > 0).mean())
    media = float(recent.mean())

    k_star_pct = pct * (1 + media / max(1, recent.max()))

    if k_star_pct < 0.20:
        return KStarEstado(k_star_pct, "estavel", "🟢 k*: ambiente estável.")
    elif k_star_pct < 0.50:
        return KStarEstado(k_star_pct, "atencao", "🟡 k*: pré-ruptura.")
    else:
        return KStarEstado(k_star_pct, "critico", "🔴 k*: ambiente crítico.")


# -------------------------------------------------------------------
# IDX ULTRA
# -------------------------------------------------------------------

def construir_idx_ultra(df: pd.DataFrame) -> IDXInfo:
    mat = extrair_series_passageiros(df)
    n_series, n_pass = mat.shape

    valores, cont = np.unique(mat, return_counts=True)
    freq_global = {int(v): float(c) / (n_series * n_pass) for v, c in zip(valores, cont)}

    freq_pos = {}
    for pos in range(n_pass):
        col = mat[:, pos]
        v, c = np.unique(col, return_counts=True)
        freq_pos[pos] = {int(vv): float(cc) / len(col) for vv, cc in zip(v, c)}

    pesos = np.linspace(0.8, 1.2, n_pass)
    pesos = pesos / pesos.sum()

    return IDXInfo(pesos, freq_global, freq_pos)


# -------------------------------------------------------------------
# IPF / IPO / IPO ASB
# -------------------------------------------------------------------

def _top_valores(freq: Dict[int, float], n: int) -> List[int]:
    return [v for v, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:n]]


def construir_ipf_ipo_ultra(df: pd.DataFrame, idx: IDXInfo, n_top_global=20) -> IPFIPOInfo:
    mat = extrair_series_passageiros(df)
    n_series, n_pass = mat.shape

    ipf = [int(np.median(mat[:, pos])) for pos in range(n_pass)]

    top_glob = _top_valores(idx.freq_global, n_top_global)

    ipo = []
    for pos in range(n_pass):
        fpos = idx.freq_posicional.get(pos, {})
        if not fpos:
            ipo.append(ipf[pos])
            continue

        melhor = None
        melhor_score = -1
        for val, f_val in fpos.items():
            f_glob = idx.freq_global.get(val, 0)
            score = 0.6 * f_val + 0.4 * f_glob
            if val in top_glob:
                score *= 1.1
            if score > melhor_score:
                melhor = val
                melhor_score = score
        ipo.append(int(melhor))

    # IPO Anti-SelfBias
    min_val, max_val = obter_faixa_numerica(df)
    ipo_asb = []
    for pos, val in enumerate(ipo):
        peso = idx.pesos_posicionais[pos]
        desloc = 1 if peso < np.median(idx.pesos_posicionais) else -1
        novo = val + desloc
        novo = min(max(novo, min_val), max_val)
        ipo_asb.append(novo)

    return IPFIPOInfo(ipf, ipo, ipo_asb)
# -------------------------------------------------------------------
# S6 PROFUNDO ULTRA
# -------------------------------------------------------------------

def _score_serie_idx(serie: List[int], idx_info: IDXInfo) -> float:
    score = 0.0
    for pos, val in enumerate(serie):
        fglob = idx_info.freq_global.get(int(val), 0.0)
        fpos = idx_info.freq_posicional.get(pos, {}).get(int(val), 0.0)
        w = idx_info.pesos_posicionais[pos]
        score += w * (0.7 * fpos + 0.3 * fglob)
    return score


def gerar_s6_profundo_ultra(
    df: pd.DataFrame,
    idx_info: IDXInfo,
    ipf_ipo: IPFIPOInfo,
    n_series: int = 60,
) -> pd.DataFrame:

    mat = extrair_series_passageiros(df)
    n_pass = mat.shape[1]
    min_val, max_val = obter_faixa_numerica(df)

    bases = [
        ipf_ipo.ipf,
        ipf_ipo.ipo,
        ipf_ipo.ipo_asb,
    ]

    # Variações estruturais
    for base in [ipf_ipo.ipf, ipf_ipo.ipo, ipf_ipo.ipo_asb]:
        for pos in range(n_pass):
            for delta in (-1, 1):
                nova = base.copy()
                nv = nova[pos] + delta
                if min_val <= nv <= max_val:
                    nova[pos] = nv
                    bases.append(nova)

    unicas = []
    vistos = set()
    for s in bases:
        t = tuple(int(x) for x in s)
        if t not in vistos:
            vistos.add(t)
            unicas.append(list(t))

    registros = []
    for s in unicas:
        registros.append({
            "series": s,
            "score_s6": _score_serie_idx(s, idx_info),
            "origem": "S6",
        })

    df_s6 = pd.DataFrame(registros).sort_values("score_s6", ascending=False)
    return df_s6.head(n_series).reset_index(drop=True)


# -------------------------------------------------------------------
# MICRO-LEQUE ULTRA
# -------------------------------------------------------------------

def gerar_micro_leque_ultra(
    df_s6: pd.DataFrame,
    df_hist: pd.DataFrame,
    raio: int = 2,
    max_por_base: int = 10,
) -> pd.DataFrame:

    if df_s6.empty:
        return pd.DataFrame(columns=["series", "score_micro", "origem"])

    min_val, max_val = obter_faixa_numerica(df_hist)
    registros = []

    for _, row in df_s6.iterrows():
        base = row["series"]
        if isinstance(base, tuple):
            base = list(base)
        n_pass = len(base)

        # Núcleo do micro-leque
        registros.append({
            "series": list(base),
            "score_micro": float(n_pass),
            "origem": "MicroLeque",
        })

        count = 0
        for pos in range(n_pass):
            for delta in range(-raio, raio + 1):
                if delta == 0:
                    continue
                nova = base.copy()
                nv = nova[pos] + delta
                if min_val <= nv <= max_val:
                    nova[pos] = nv
                    registros.append({
                        "series": nova,
                        "score_micro": float(n_pass - abs(delta)),
                        "origem": "MicroLeque",
                    })
                    count += 1
                    if count >= max_por_base:
                        break
            if count >= max_por_base:
                break

    df_micro = pd.DataFrame(registros)
    df_micro = normalizar_coluna_series(df_micro)
    if df_micro.empty:
        return df_micro

    df_micro = (
        df_micro.groupby("series", as_index=False)
        .agg(score_micro=("score_micro", "max"), origem=("origem", "first"))
        .sort_values("score_micro", ascending=False)
        .reset_index(drop=True)
    )
    return df_micro


# -------------------------------------------------------------------
# MONTE CARLO PROFUNDO ULTRA (COM PATCH DEFINITIVO)
# -------------------------------------------------------------------

def simular_monte_carlo_ultra(
    df_hist: pd.DataFrame,
    janela_mc: int = 40,
    n_sim: int = 500,
) -> pd.DataFrame:

    if df_hist.empty:
        return pd.DataFrame(columns=["series", "score_mc", "origem"])

    n_pass = obter_num_passageiros(df_hist)
    cols = [f"P{i+1}" for i in range(n_pass)]

    janela_real = min(janela_mc, len(df_hist))
    df_recent = df_hist.iloc[-janela_real:]

    # Distribuições empíricas por posição
    dist_pos = {pos: df_recent[col].to_numpy(dtype=int) for pos, col in enumerate(cols)}

    rng = np.random.default_rng(seed=42)
    simulacoes = []

    for _ in range(n_sim):
        serie = []
        for pos in range(n_pass):
            valores = dist_pos[pos]

            # Se não há valores, preencher com zero
            if len(valores) == 0:
                serie.append(0)
                continue

            val = rng.choice(valores)

            # PATCH — garantir que val nunca seja numpy 0D / array
            if isinstance(val, np.ndarray):
                try:
                    val = val.item()
                except Exception:
                    val = int(val.astype(int))

            serie.append(int(val))

        simulacoes.append(tuple(int(v) for v in serie))

    valores, cont = np.unique(simulacoes, return_counts=True)
    registros = []
    total = float(len(simulacoes))

    for serie, freq in zip(valores, cont):
        registros.append({
            "series": list(serie),
            "score_mc": float(freq) / total,
            "origem": "MonteCarlo",
        })

    df_mc = pd.DataFrame(registros)
    return df_mc.sort_values("score_mc", ascending=False).reset_index(drop=True)


# -------------------------------------------------------------------
# NORMALIZAÇÃO SEGURA DE 'series'
# -------------------------------------------------------------------

def normalizar_coluna_series(df_flat: pd.DataFrame) -> pd.DataFrame:
    if "series" not in df_flat.columns:
        return df_flat

    def _to_tuple(x):
        if x is None:
            return None

        if isinstance(x, (list, tuple, np.ndarray)):
            vals = list(x)
        else:
            try:
                s = str(x)
            except Exception:
                return None
            tokens = s.replace(",", " ").split()
            vals = []
            for t in tokens:
                try:
                    vals.append(int(t))
                except:
                    pass

        vals = [int(v) for v in vals]
        if not vals:
            return None
        return tuple(vals)

    df = df_flat.copy()
    df["series"] = df["series"].apply(_to_tuple)
    return df[df["series"].notna()].copy()


# -------------------------------------------------------------------
# FUSÃO ULTRA (S6 + MC + MICRO-LEQUE)
# -------------------------------------------------------------------

def fundir_leques_ultra(
    df_s6: pd.DataFrame,
    df_mc: pd.DataFrame,
    df_micro: pd.DataFrame,
    peso_s6: float,
    peso_mc: float,
    peso_micro: float,
) -> pd.DataFrame:

    frames = []

    if df_s6 is not None and not df_s6.empty:
        frames.append(df_s6[["series", "score_s6", "origem"]])

    if df_mc is not None and not df_mc.empty:
        frames.append(df_mc[["series", "score_mc", "origem"]])

    if df_micro is not None and not df_micro.empty:
        frames.append(df_micro[["series", "score_micro", "origem"]])

    if not frames:
        return pd.DataFrame(columns=["series", "score_final"])

    df_all = pd.concat(frames, ignore_index=True)
    df_all = normalizar_coluna_series(df_all)

    # Garantir colunas
    for c in ["score_s6", "score_mc", "score_micro"]:
        if c not in df_all.columns:
            df_all[c] = 0.0

    agg = (
        df_all.groupby("series", as_index=False)
        .agg(
            score_s6=("score_s6", "max"),
            score_mc=("score_mc", "max"),
            score_micro=("score_micro", "max"),
        )
    )

    agg["score_final"] = (
        peso_s6 * agg["score_s6"]
        + peso_mc * agg["score_mc"]
        + peso_micro * agg["score_micro"]
    )

    return agg.sort_values("score_final", ascending=False).reset_index(drop=True)
# -------------------------------------------------------------------
# MODO TURBO++ ULTRA — NÚCLEO (COM RETROCOMPATIBILIDADE)
# -------------------------------------------------------------------

def montar_previsao_turbo_ultra(
    df_hist: pd.DataFrame,
    regime_state: str,
    output_mode: str = "top",
    n_s6: Optional[int] = None,
    janela_s6: int = 40,
    janela_mc: int = 40,
    n_sim_mc: int = 500,
    peso_s6: float = 0.5,
    peso_mc: float = 0.3,
    peso_micro: float = 0.2,
    usar_micro_leque: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Núcleo do Modo TURBO++ ULTRA:

    - Constrói IDX ULTRA
    - Constrói IPF / IPO / IPO ASB
    - Gera S6 Profundo ULTRA
    - Gera Monte Carlo Profundo ULTRA
    - Gera Micro-Leque ULTRA (opcional)
    - Realiza Fusão ULTRA (S6 + MC + Micro-Leque)

    Retrocompatibilidade:
    Aceita parâmetros antigos via **kwargs:
    - n_series_saida      -> n_s6
    - window_s6           -> janela_s6
    - window_mc           -> janela_mc
    - incluir_micro_leque -> usar_micro_leque
    """

    # ---------------- RETROCOMPATIBILIDADE ----------------
    if n_s6 is None:
        n_s6 = kwargs.get("n_series_saida", 60)

    if "window_s6" in kwargs:
        try:
            janela_s6 = int(kwargs["window_s6"])
        except Exception:
            pass

    if "window_mc" in kwargs:
        try:
            janela_mc = int(kwargs["window_mc"])
        except Exception:
            pass

    if "incluir_micro_leque" in kwargs:
        try:
            usar_micro_leque = bool(kwargs["incluir_micro_leque"])
        except Exception:
            pass

    # ---------------- BARRA DE PROGRESSO ----------------
    progress_bar = st.progress(0)
    status_text = st.empty()

    # ---------------- PIPELINE ULTRA ----------------
    try:
        status_text.text("Etapa 1/4 — Construindo IDX ULTRA...")
        progress_bar.progress(5)
        idx_info = construir_idx_ultra(df_hist)

        status_text.text("Etapa 2/4 — Construindo IPF / IPO / IPO ASB...")
        progress_bar.progress(20)
        ipf_ipo = construir_ipf_ipo_ultra(df_hist, idx_info)

        status_text.text("Etapa 3/4 — Gerando S6 Profundo ULTRA...")
        progress_bar.progress(40)
        df_s6 = gerar_s6_profundo_ultra(
            df_hist,
            idx_info,
            ipf_ipo,
            n_series=n_s6,
        )

        status_text.text("Etapa 4/4 — Monte Carlo Profundo ULTRA...")
        progress_bar.progress(60)
        df_mc = simular_monte_carlo_ultra(
            df_hist=df_hist,
            janela_mc=janela_mc,
            n_sim=n_sim_mc,
        )

        if usar_micro_leque:
            status_text.text("Etapa extra — Micro-Leque ULTRA...")
            progress_bar.progress(80)
            df_micro = gerar_micro_leque_ultra(
                df_s6=df_s6,
                df_hist=df_hist,
                raio=2,
                max_por_base=12,
            )
        else:
            df_micro = pd.DataFrame(columns=["series", "score_micro", "origem"])

        status_text.text("Fusão ULTRA — S6 + MC + Micro-Leque...")
        progress_bar.progress(90)
        df_fusao = fundir_leques_ultra(
            df_s6=df_s6,
            df_mc=df_mc,
            df_micro=df_micro,
            peso_s6=peso_s6,
            peso_mc=peso_mc,
            peso_micro=peso_micro,
        )

        progress_bar.progress(100)
        status_text.text("Fusão concluída.")

    finally:
        # Mantém a barra cheia ao final
        try:
            progress_bar.progress(100)
        except Exception:
            pass

    if df_fusao is None or df_fusao.empty:
        return pd.DataFrame(columns=["series", "score_final", "score_s6", "score_mc", "score_micro"])

    df_out = df_fusao.copy()
    if output_mode == "top":
        return df_out
    elif output_mode == "detalhado":
        return df_out
    else:
        return df_out


# -------------------------------------------------------------------
# QDS REAL + BACKTEST REAL
# -------------------------------------------------------------------

def calcular_qds_real(
    df_hist: pd.DataFrame,
    horizonte_teste: int = 40,
    n_top: int = 10,
    janela_s6: int = 40,
    janela_mc: int = 40,
    n_sim_mc: int = 400,
) -> Dict[str, Any]:
    """
    QDS REAL / Backtest:
    - Anda pela estrada no passado (janela horizonte_teste).
    - Para cada ponto i, roda o Modo TURBO++ ULTRA usando apenas as séries até i.
    - Verifica se a série i+1 estaria entre as Top-n_top previsões.
    """

    if len(df_hist) < horizonte_teste + 5:
        return {
            "taxa_acerto": float("nan"),
            "n_testes": 0,
            "hits": [],
            "idx_testados": [],
        }

    inicio = max(1, len(df_hist) - horizonte_teste - 1)
    fim = len(df_hist) - 2

    hits = []
    idx_testados = []

    for i in range(inicio, fim + 1):
        df_past = df_hist.iloc[: i + 1].copy()
        alvo = df_hist.iloc[i + 1]

        try:
            bar = analisar_barometro_ultra(df_past)
            df_pred = montar_previsao_turbo_ultra(
                df_hist=df_past,
                regime_state=bar.regime,
                output_mode="top",
                n_s6=n_top * 3,
                janela_s6=janela_s6,
                janela_mc=janela_mc,
                n_sim_mc=n_sim_mc,
                peso_s6=0.5,
                peso_mc=0.3,
                peso_micro=0.2,
            )
        except Exception:
            hits.append(0)
            idx_testados.append(i)
            continue

        if df_pred is None or df_pred.empty:
            hits.append(0)
            idx_testados.append(i)
            continue

        n_pass = obter_num_passageiros(df_hist)
        alvo_tuple = tuple(int(alvo[f"P{j+1}"]) for j in range(n_pass))

        df_pred_norm = normalizar_coluna_series(df_pred)
        candidatos = df_pred_norm["series"].head(n_top).tolist()

        hit = int(alvo_tuple in candidatos)
        hits.append(hit)
        idx_testados.append(i)

    if not hits:
        return {
            "taxa_acerto": float("nan"),
            "n_testes": 0,
            "hits": [],
            "idx_testados": [],
        }

    taxa = float(np.mean(hits))
    return {
        "taxa_acerto": taxa,
        "n_testes": len(hits),
        "hits": hits,
        "idx_testados": idx_testados,
    }


# -------------------------------------------------------------------
# REPLAY LIGHT
# -------------------------------------------------------------------

def resumo_replay_light(df_hist: pd.DataFrame, idx: int) -> Dict[str, Any]:
    """
    Replay LIGHT:
    - Mostra série alvo, k, e o estado local da estrada.
    """
    if idx < 0 or idx >= len(df_hist):
        raise IndexError("Índice fora do histórico.")

    linha = df_hist.iloc[idx]
    n_pass = obter_num_passageiros(df_hist)
    serie = [int(linha[f"P{i+1}"]) for i in range(n_pass)]
    k_val = int(linha["k"])

    janela_local = max(5, min(20, len(df_hist)))
    start = max(0, idx - janela_local // 2)
    end = min(len(df_hist), idx + janela_local // 2)
    df_local = df_hist.iloc[start:end]

    bar_local = analisar_barometro_ultra(df_local)
    kstar_local = calcular_kstar_ultra(df_local)

    return {
        "id": linha["ID"],
        "idx": idx,
        "serie": serie,
        "k": k_val,
        "barometro": bar_local,
        "kstar": kstar_local,
    }


# -------------------------------------------------------------------
# REPLAY ULTRA
# -------------------------------------------------------------------

def executar_replay_ultra(
    df_hist: pd.DataFrame,
    inicio: int,
    fim: int,
    n_top: int = 10,
) -> Dict[str, Any]:
    """
    Replay ULTRA:
    - Percorre do índice 'inicio' até 'fim'.
    - Para cada i, roda TURBO++ ULTRA com histórico até i.
    - Verifica se a série i+1 estaria entre as Top-n_top.
    """

    if inicio < 1:
        inicio = 1
    if fim >= len(df_hist) - 1:
        fim = len(df_hist) - 2
    if inicio >= fim:
        return {"hits": [], "idxs": [], "taxa": float("nan")}

    hits = []
    idxs = []

    total = max(1, fim - inicio + 1)
    progress = st.progress(0)

    for c, i in enumerate(range(inicio, fim + 1), start=1):
        progress.progress(int(100 * c / total))

        df_past = df_hist.iloc[: i + 1].copy()
        alvo = df_hist.iloc[i + 1]

        try:
            bar = analisar_barometro_ultra(df_past)
            df_pred = montar_previsao_turbo_ultra(
                df_hist=df_past,
                regime_state=bar.regime,
                output_mode="top",
                n_s6=n_top * 3,
                janela_s6=40,
                janela_mc=40,
                n_sim_mc=300,
                peso_s6=0.5,
                peso_mc=0.3,
                peso_micro=0.2,
            )
        except Exception:
            hits.append(0)
            idxs.append(i)
            continue

        if df_pred is None or df_pred.empty:
            hits.append(0)
            idxs.append(i)
            continue

        n_pass = obter_num_passageiros(df_hist)
        alvo_tuple = tuple(int(alvo[f"P{j+1}"]) for j in range(n_pass))

        df_pred_norm = normalizar_coluna_series(df_pred)
        candidatos = df_pred_norm["series"].head(n_top).tolist()

        hit = int(alvo_tuple in candidatos)
        hits.append(hit)
        idxs.append(i)

    progress.progress(100)
    taxa = float(np.mean(hits)) if hits else float("nan")
    return {"hits": hits, "idxs": idxs, "taxa": taxa}
# -------------------------------------------------------------------
# ESTADO GLOBAL
# -------------------------------------------------------------------

if "df" not in st.session_state:
    st.session_state["df"] = None


# -------------------------------------------------------------------
# SIDEBAR — NAVEGAÇÃO E PARÂMETROS GLOBAIS
# -------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Navegação")

    painel = st.radio(
        "Escolha o painel:",
        [
            "📥 Histórico — Entrada",
            "🔍 Pipeline V14-FLEX (TURBO++)",
            "🚨 Monitor de Risco (k & k*)",
            "🚀 Modo TURBO++ ULTRA — Painel Completo",
            "📅 Modo Replay Automático do Histórico",
            "🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)",
        ],
    )

    st.markdown("---")
    st.markdown("### Parâmetros Globais (TURBO++ ULTRA)")

    n_s6_global = st.slider("N° base de séries S6 (n_s6)", 20, 200, 60, step=10)
    janela_s6_global = st.slider("Janela S6 (séries)", 10, 200, 40, step=5)
    janela_mc_global = st.slider("Janela Monte Carlo (séries)", 10, 200, 40, step=5)
    n_sim_mc_global = st.slider("N° simulações Monte Carlo (n_sim_mc)", 100, 2000, 500, step=100)

    st.markdown("#### Pesos de Fusão ULTRA")
    peso_s6_global = st.slider("Peso S6", 0.0, 1.0, 0.5, step=0.05)
    peso_mc_global = st.slider("Peso Monte Carlo", 0.0, 1.0, 0.3, step=0.05)
    peso_micro_global = st.slider("Peso Micro-Leque", 0.0, 1.0, 0.2, step=0.05)

    st.markdown("---")
    st.caption("Predict Cars V14-FLEX ULTRA REAL (TURBO++) — Modo Completo.")


# -------------------------------------------------------------------
# PAINEL 1 — HISTÓRICO — ENTRADA FLEX
# -------------------------------------------------------------------

if painel == "📥 Histórico — Entrada":
    st.markdown("## 📥 Histórico — Entrada (FLEX)")

    df = None
    opc = st.radio(
        "Como deseja carregar o histórico?",
        ["Enviar arquivo CSV", "Copiar e colar o histórico"],
    )

    # Upload CSV
    if opc == "Enviar arquivo CSV":
        file = st.file_uploader("Selecione o arquivo CSV:", type=["csv"])
        if file is not None:
            try:
                df_raw = pd.read_csv(file, sep=None, engine="python")
                df = preparar_historico_v14_flex(df_raw)
                st.session_state["df"] = df
                st.success(f"Histórico carregado com sucesso! Séries: {len(df)}.")
                st.write("Prévia (20 primeiras linhas):")
                st.dataframe(df.head(20))
            except Exception as e:
                st.error(f"Erro ao carregar CSV: {e}")

    # Copiar/colar histórico
    else:
        texto = st.text_area(
            "Cole o histórico bruto (com cabeçalho):",
            height=300,
            placeholder="ID;P1;P2;...;k\nC1;41;5;4;52;30;33;0\nC2;9;39;37;49;43;41;1\n...",
        )
        if st.button("Processar histórico colado"):
            try:
                from io import StringIO

                df_raw = pd.read_csv(StringIO(texto), sep=None, engine="python")
                df = preparar_historico_v14_flex(df_raw)
                st.session_state["df"] = df
                st.success(f"Histórico carregado com sucesso! Séries: {len(df)}.")
                st.write("Prévia (20 primeiras linhas):")
                st.dataframe(df.head(20))
            except Exception as e:
                st.error(f"Erro ao processar histórico colado: {e}")

    # Resumo atual
    if st.session_state.get("df") is not None:
        df = st.session_state["df"]
        st.markdown("### Resumo do histórico atual")
        st.write(f"Séries: **{len(df)}** | Passageiros por série: **{obter_num_passageiros(df)}**")
        st.dataframe(df.tail(10))


# -------------------------------------------------------------------
# PAINEL 2 — PIPELINE V14-FLEX (TURBO++)
# -------------------------------------------------------------------

if painel == "🔍 Pipeline V14-FLEX (TURBO++)":
    st.markdown("## 🔍 Pipeline V14-FLEX (TURBO++)")

    df = st.session_state.get("df")
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    st.markdown("### 1. Barômetro ULTRA REAL")
    barometro = analisar_barometro_ultra(df)
    st.write(f"Regime: **{barometro.regime}**")
    st.write(f"Volatilidade de k (janela): `{barometro.volatilidade_k:.3f}`")
    st.write(f"Média de k (janela): `{barometro.media_k:.3f}`")
    st.info(barometro.msg)

    st.markdown("### 2. k* ULTRA REAL")
    kstar = calcular_kstar_ultra(df)
    st.write(f"k*: `{kstar.k_star_pct:.3f}`")
    st.info(kstar.msg)

    st.markdown("### 3. IDX ULTRA")
    idx_info = construir_idx_ultra(df)
    st.write("Pesos posicionais (IDX ULTRA):", idx_info.pesos_posicionais)

    st.markdown("### 4. IPF / IPO / IPO ASB (ULTRA)")
    ipf_ipo = construir_ipf_ipo_ultra(df, idx_info)
    st.write("IPF (mediana estrutural):", ipf_ipo.ipf)
    st.write("IPO (núcleo profissional):", ipf_ipo.ipo)
    st.write("IPO Anti-SelfBias:", ipf_ipo.ipo_asb)

    st.markdown("### 5. S6 Profundo ULTRA — Prévia")
    df_s6_prev = gerar_s6_profundo_ultra(
        df,
        idx_info,
        ipf_ipo,
        n_series=n_s6_global,
    )
    st.dataframe(df_s6_prev.head(20))

    st.markdown("### 6. Monte Carlo Profundo ULTRA — Prévia")
    df_mc_prev = simular_monte_carlo_ultra(
        df_hist=df,
        janela_mc=janela_mc_global,
        n_sim=n_sim_mc_global,
    )
    st.dataframe(df_mc_prev.head(20))

    st.markdown("### 7. Micro-Leque ULTRA — Prévia")
    df_micro_prev = gerar_micro_leque_ultra(
        df_s6=df_s6_prev,
        df_hist=df,
        raio=2,
        max_por_base=6,
    )
    st.dataframe(df_micro_prev.head(20))

    st.markdown("### 8. Fusão ULTRA — Prévia")
    df_fusao_prev = fundir_leques_ultra(
        df_s6=df_s6_prev,
        df_mc=df_mc_prev,
        df_micro=df_micro_prev,
        peso_s6=peso_s6_global,
        peso_mc=peso_mc_global,
        peso_micro=peso_micro_global,
    )
    st.dataframe(df_fusao_prev.head(30))


# -------------------------------------------------------------------
# PAINEL 3 — MONITOR DE RISCO (k & k*)
# -------------------------------------------------------------------

if painel == "🚨 Monitor de Risco (k & k*)":
    st.markdown("## 🚨 Monitor de Risco (k & k*)")

    df = st.session_state.get("df")
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Barômetro ULTRA REAL")
        barometro = analisar_barometro_ultra(df)
        st.metric("Regime atual", barometro.regime)
        st.write(f"Volatilidade de k (janela): `{barometro.volatilidade_k:.3f}`")
        st.write(f"Média de k (janela): `{barometro.media_k:.3f}`")
        st.info(barometro.msg)

    with col2:
        st.markdown("### k* ULTRA REAL")
        kstar = calcular_kstar_ultra(df)
        st.metric("k* (sentinela)", f"{kstar.k_star_pct:.3f}")
        st.info(kstar.msg)

    st.markdown("---")
    st.markdown("### Interpretação do novo k")
    st.write(
        """
- **k** = número de guardas que acertaram exatamente o carro (todos os passageiros).
- **k*** = sensibilidade consolidada da estrada:
    - quanto maior, mais blocos de carros previsíveis se repetem;
    - quanto menor, mais caótica a estrada.
"""
    )


# -------------------------------------------------------------------
# PAINEL 4 — MODO TURBO++ ULTRA — PAINEL COMPLETO
# -------------------------------------------------------------------

if painel == "🚀 Modo TURBO++ ULTRA — Painel Completo":
    st.markdown("## 🚀 Modo TURBO++ ULTRA — Painel Completo")

    df = st.session_state.get("df")
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    st.markdown("### Parâmetros específicos do TURBO++ ULTRA")

    n_saida = st.slider("N° de séries na saída (Top-N final)", 10, 200, 60, step=5)
    usar_micro = st.checkbox("Incluir Micro-Leque ULTRA na fusão", value=True)

    if st.button("Executar Modo TURBO++ ULTRA agora"):
        with st.spinner("Rodando Modo TURBO++ ULTRA (S6 + Monte Carlo + Micro-Leque)..."):
            barometro = analisar_barometro_ultra(df)
            df_turbo = montar_previsao_turbo_ultra(
                df_hist=df,
                regime_state=barometro.regime,
                output_mode="top",
                n_s6=n_s6_global,
                janela_s6=janela_s6_global,
                janela_mc=janela_mc_global,
                n_sim_mc=n_sim_mc_global,
                peso_s6=peso_s6_global,
                peso_mc=peso_mc_global,
                peso_micro=peso_micro_global,
                incluir_micro_leque=usar_micro,   # retrocompatível
                n_series_saida=n_s6_global,       # retrocompatível
            )

        if df_turbo is None or df_turbo.empty:
            st.error("Modo TURBO++ ULTRA não retornou séries de previsão.")
        else:
            st.success("Modo TURBO++ ULTRA executado com sucesso.")

            df_view = df_turbo.copy().head(n_saida)
            st.markdown("### 🎯 Séries de Previsão — Top-N (TURBO++ ULTRA)")
            st.dataframe(df_view)

            serie_final = df_view.iloc[0]["series"]
            if isinstance(serie_final, tuple):
                serie_final = list(serie_final)

            st.markdown("### 🔚 Previsão Final TURBO++ ULTRA")
            st.code(" ".join(str(x) for x in serie_final), language="text")

            kstar = calcular_kstar_ultra(df)
            if kstar.estado == "estavel":
                contexto_k = "🟢 k*: Ambiente estável — previsão em regime normal."
            elif kstar.estado == "atencao":
                contexto_k = "🟡 k*: Pré-ruptura residual — usar previsão com atenção."
            else:
                contexto_k = "🔴 k*: Ambiente crítico — usar previsão com cautela máxima."
            st.info(contexto_k)


# -------------------------------------------------------------------
# PAINEL 5 — MODO REPLAY AUTOMÁTICO DO HISTÓRICO
# -------------------------------------------------------------------

if painel == "📅 Modo Replay Automático do Histórico":
    st.markdown("## 📅 Modo Replay Automático do Histórico")

    df = st.session_state.get("df")
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    n_series = len(df)
    st.write(f"Histórico com **{n_series}** séries.")

    col1, col2 = st.columns(2)

    # Replay LIGHT
    with col1:
        st.markdown("### Replay LIGHT")
        idx_light = st.number_input(
            "Índice da série (0 = primeira linha do DataFrame):",
            min_value=0,
            max_value=max(0, n_series - 1),
            value=max(0, n_series - 2),
            step=1,
        )
        if st.button("Executar Replay LIGHT"):
            try:
                resumo = resumo_replay_light(df, int(idx_light))
                st.markdown("#### Resultado Replay LIGHT")
                st.write(f"ID: **{resumo['id']}** (índice {resumo['idx']})")
                st.write("Série:", resumo["serie"])
                st.write("k:", resumo["k"])
                st.markdown("##### Barômetro local")
                st.info(resumo["barometro"].msg)
                st.markdown("##### k* local")
                st.info(resumo["kstar"].msg)
            except Exception as e:
                st.error(f"Erro no Replay LIGHT: {e}")

    # Replay ULTRA
    with col2:
        st.markdown("### Replay ULTRA — Backtest focal")
        inicio = st.number_input(
            "Início (índice)",
            min_value=1,
            max_value=max(1, n_series - 3),
            value=max(1, n_series - 30),
        )
        fim = st.number_input(
            "Fim (índice)",
            min_value=2,
            max_value=max(2, n_series - 2),
            value=max(2, n_series - 5),
        )
        n_top_replay = st.slider("Top-N para considerar acerto", 5, 50, 10, step=5)

        if st.button("Executar Replay ULTRA"):
            with st.spinner("Executando Replay ULTRA..."):
                try:
                    resultado = executar_replay_ultra(
                        df_hist=df,
                        inicio=int(inicio),
                        fim=int(fim),
                        n_top=int(n_top_replay),
                    )
                    st.success(
                        f"Replay ULTRA concluído. Taxa de acerto ≈ {resultado['taxa']:.3f} em {len(resultado['hits'])} testes."
                    )
                    st.markdown("#### Sequência de acertos (1 = acerto, 0 = erro)")
                    st.write(resultado["hits"])
                except Exception as e:
                    st.error(f"Erro no Replay ULTRA: {e}")


# -------------------------------------------------------------------
# PAINEL 6 — TESTES DE CONFIABILIDADE (QDS / BACKTEST / MONTE CARLO)
# -------------------------------------------------------------------

if painel == "🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)":
    st.markdown("## 🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)")

    df = st.session_state.get("df")
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    horizonte = st.slider("Horizonte de teste (séries)", 10, 200, 40, step=5)
    n_top_qds = st.slider("Top-N para acerto QDS", 5, 60, 10, step=5)

    if st.button("Rodar QDS REAL / Backtest"):
        with st.spinner("Rodando QDS REAL / Backtest no modo TURBO++ ULTRA..."):
            try:
                qds = calcular_qds_real(
                    df_hist=df,
                    horizonte_teste=int(horizonte),
                    n_top=int(n_top_qds),
                    janela_s6=janela_s6_global,
                    janela_mc=janela_mc_global,
                    n_sim_mc=n_sim_mc_global,
                )
                if qds["n_testes"] == 0:
                    st.warning("Histórico insuficiente para QDS REAL.")
                else:
                    st.success(
                        f"QDS REAL executado com sucesso. Taxa de acerto ≈ {qds['taxa_acerto']:.3f} em {qds['n_testes']} testes."
                    )
                    st.markdown("### Curva de acertos (1 = acerto, 0 = erro)")
                    st.write(qds["hits"])
            except Exception as e:
                st.error(f"Erro ao calcular QDS REAL / Backtest: {e}")
