# app_v14_flex_replay.py
# Predict Cars V14-FLEX ULTRA REAL (TURBO++)
# Versão COMPLETA, sem simplificações conceituais:
# - Entrada FLEX (n variável de passageiros, detecção k)
# - Barômetro ULTRA REAL
# - k* ULTRA REAL
# - IDX ULTRA
# - IPF / IPO refinados
# - S6 Profundo ULTRA
# - Micro-Leque ULTRA
# - Monte Carlo Profundo ULTRA
# - QDS REAL + Backtest REAL (estrutura instalada)
# - Replay LIGHT / Replay ULTRA
# - Modo TURBO++ ULTRA Adaptativo (S6 + MC + Micro-Leque)
# - Painel de risco k/k*

import itertools
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st


# -------------------------------------------------------------------
# CONFIGURAÇÃO BÁSICA DA PÁGINA
# -------------------------------------------------------------------

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
# FUNÇÕES BÁSICAS DE SUPORTE
# -------------------------------------------------------------------


def detectar_num_passageiros(df_raw: pd.DataFrame) -> int:
    """
    Detecta automaticamente o número de passageiros (colunas numéricas entre ID e k).
    Convenção:
    - primeira coluna: ID da série (C1, C2, etc.)
    - última coluna: coluna k (número de guardas que acertaram exatamente o carro)
    - colunas intermediárias: passageiros (P1..Pn)
    """
    if df_raw.shape[1] < 3:
        raise ValueError(
            "Histórico inválido: são necessárias pelo menos 3 colunas (ID, passageiros, k)."
        )
    # n_passageiros = colunas - ID - k
    return df_raw.shape[1] - 2


def preparar_historico_v14_flex(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza o histórico para o formato interno do V14-FLEX ULTRA:
    colunas: ['ID', 'P1'..'Pn', 'k'] com n variável.

    Assume:
    - primeira coluna = ID
    - última coluna = k
    - demais = passageiros
    """
    df = df_raw.copy()

    # Remover colunas completamente vazias
    df = df.dropna(axis=1, how="all")

    if df.shape[1] < 3:
        raise ValueError(
            "Histórico inválido após limpeza: são necessárias pelo menos 3 colunas."
        )

    n_pass = detectar_num_passageiros(df)

    colunas = ["ID"] + [f"P{i+1}" for i in range(n_pass)] + ["k"]
    df.columns = colunas

    # Limpar espaços/strings em ID
    df["ID"] = df["ID"].astype(str).str.strip()

    # Converter passageiros e k para inteiros
    for c in colunas[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().reset_index(drop=True).copy()
    df["k"] = df["k"].astype(int)

    return df


def obter_num_passageiros(df: pd.DataFrame) -> int:
    cols = [c for c in df.columns if c.startswith("P")]
    return len(cols)


def extrair_series_passageiros(df: pd.DataFrame) -> np.ndarray:
    """
    Retorna matriz NxM com os passageiros (sem k, sem ID).
    """
    n_pass = obter_num_passageiros(df)
    cols = [f"P{i+1}" for i in range(n_pass)]
    return df[cols].to_numpy(dtype=int)


def obter_faixa_numerica(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Faixa mínima e máxima observada entre todos passageiros (para Micro-Leque etc.).
    """
    mat = extrair_series_passageiros(df)
    return int(mat.min()), int(mat.max())


# -------------------------------------------------------------------
# BARÔMETRO ULTRA REAL — ESTADO DA ESTRADA
# -------------------------------------------------------------------


def analisar_barometro_ultra(df: pd.DataFrame, janela: int = 40) -> BarometroEstado:
    """
    Barômetro ULTRA:
    - Usa janela recente de k.
    - Volatilidade (desvio padrão) e média de k.
    - Classifica regime da estrada em: estável / transição / turbulento / ruptura.
    """
    k_series = df["k"].astype(float)
    if len(k_series) < 5:
        return BarometroEstado(
            regime="indefinido",
            volatilidade_k=float("nan"),
            media_k=float("nan"),
            msg="Histórico muito curto para diagnóstico confiável.",
        )

    janela_real = min(janela, len(k_series))
    k_recent = k_series.iloc[-janela_real:]

    media_k = float(k_recent.mean())
    vol_k = float(k_recent.std(ddof=0))

    # Heurísticas de regime, mantendo a metáfora rica
    if vol_k < 0.5 and media_k < 0.5:
        regime = "estavel"
        msg = "🟢 Estrada estável — poucos guardas acertando exatamente os carros."
    elif vol_k < 1.0 and media_k < 1.5:
        regime = "transicao"
        msg = "🟡 Estrada em transição — surgem nichos onde os guardas começam a acertar."
    elif vol_k < 2.0 or media_k < 3.0:
        regime = "turbulento"
        msg = "🟠 Estrada turbulenta — concentração de carros previsíveis se formando."
    else:
        regime = "ruptura"
        msg = "🔴 Ruptura estrutural — grande bloco de carros previsíveis emergiu."

    return BarometroEstado(
        regime=regime,
        volatilidade_k=vol_k,
        media_k=media_k,
        msg=msg,
    )


# -------------------------------------------------------------------
# k* ULTRA REAL — SENTINELA DOS GUARDAS
# -------------------------------------------------------------------


def calcular_kstar_ultra(df: pd.DataFrame, janela: int = 80) -> KStarEstado:
    """
    k* ULTRA REAL:
    - Calcula o percentual de séries recentes em que k > 0.
    - Usa isso como sensibilidade de guarda: quantos guardas, em média,
      estão acertando exatamente o carro (não só se há acerto).
    - Classificação:
        * < 20%   -> estável (baixo nível de acertos exatos)
        * 20–50%  -> atenção
        * > 50%   -> crítico
    """
    if "k" not in df.columns:
        return KStarEstado(
            k_star_pct=float("nan"),
            estado="indefinido",
            msg="Histórico sem coluna k — impossível estimar k*.",
        )

    k_series = df["k"].astype(float)
    if len(k_series) == 0:
        return KStarEstado(
            k_star_pct=float("nan"),
            estado="indefinido",
            msg="Histórico vazio para k*.",
        )

    janela_real = min(janela, len(k_series))
    recent = k_series.iloc[-janela_real:]

    # k* = porcentagem de séries onde k > 0 ponderada pela média de k
    pct_k_pos = float((recent > 0).mean())
    media_k = float(recent.mean())
    k_star_pct = pct_k_pos * (1.0 + media_k / max(1.0, recent.max()))

    if k_star_pct < 0.20:
        estado = "estavel"
        msg = "🟢 k*: Ambiente estável — poucos guardas acertando exatamente."
    elif k_star_pct < 0.50:
        estado = "atencao"
        msg = "🟡 k*: Pré-ruptura residual — zona de atenção."
    else:
        estado = "critico"
        msg = "🔴 k*: Ambiente crítico — concentração forte de carros previsíveis."

    return KStarEstado(
        k_star_pct=k_star_pct,
        estado=estado,
        msg=msg,
    )


# -------------------------------------------------------------------
# IDX ULTRA — NÚCLEO PONDERADO
# -------------------------------------------------------------------


def construir_idx_ultra(df: pd.DataFrame) -> IDXInfo:
    """
    IDX ULTRA:
    - Matriz de passageiros ao longo da estrada.
    - Frequência global de cada passageiro (em toda estrada).
    - Frequência posicional de cada passageiro (por posição P1..Pn).
    - Pesos posicionais decrescentes, reforçando Pn e Pn-1 etc.
    """
    mat = extrair_series_passageiros(df)
    n_series, n_pass = mat.shape

    # Frequência global
    valores, contagens = np.unique(mat, return_counts=True)
    freq_global = {int(v): float(c) / float(n_series * n_pass) for v, c in zip(valores, contagens)}

    # Frequência posicional
    freq_posicional: Dict[int, Dict[int, float]] = {}
    for pos in range(n_pass):
        col = mat[:, pos]
        v, c = np.unique(col, return_counts=True)
        total = float(len(col))
        freq_posicional[pos] = {int(vv): float(cc) / total for vv, cc in zip(v, c)}

    # Pesos posicionais: últimas posições mais pesadas (carros no fim mais determinantes)
    base = np.linspace(0.8, 1.2, n_pass)
    pesos_posicionais = base / base.sum()

    return IDXInfo(
        pesos_posicionais=pesos_posicionais,
        freq_global=freq_global,
        freq_posicional=freq_posicional,
    )


# -------------------------------------------------------------------
# IPF / IPO ULTRA — NÚCLEOS ESTRUTURAIS
# -------------------------------------------------------------------


def _top_valores_por_freq(freq_dict: Dict[int, float], n_top: int) -> List[int]:
    return [v for v, _ in sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:n_top]]


def construir_ipf_ipo_ultra(
    df: pd.DataFrame,
    idx_info: IDXInfo,
    n_top_global: int = 20,
) -> IPFIPOInfo:
    """
    Constrói IPF (mediana estrutural) e IPO (núcleo profissional) com
    uma variante Anti-SelfBias (ASB).
    """
    mat = extrair_series_passageiros(df)
    n_series, n_pass = mat.shape

    # IPF — escolha de mediana estrutural: mediana por posição
    ipf = [int(np.median(mat[:, pos])) for pos in range(n_pass)]

    # IPO — combinação de valores de maior frequência global
    top_globais = _top_valores_por_freq(idx_info.freq_global, n_top_global)
    ipo = []
    for pos in range(n_pass):
        candidatos_pos = idx_info.freq_posicional.get(pos, {})
        if not candidatos_pos:
            ipo.append(ipf[pos])
            continue

        # mistura global + posicional, ponderada
        melhor_valor = None
        melhor_score = -1.0
        for valor, fpos in candidatos_pos.items():
            fglob = idx_info.freq_global.get(valor, 0.0)
            score = 0.6 * fpos + 0.4 * fglob
            if valor in top_globais:
                score *= 1.1
            if score > melhor_score:
                melhor_score = score
                melhor_valor = valor

        if melhor_valor is None:
            melhor_valor = ipf[pos]

        ipo.append(int(melhor_valor))

    # IPO ASB — Anti-SelfBias:
    # empurra IPO um pouco na direção contrária ao próprio cluster da série alvo.
    # Aqui, como não há alvo explícito, usamos uma leve perturbação por posição.
    ipo_asb = []
    min_val, max_val = obter_faixa_numerica(df)
    for pos, val in enumerate(ipo):
        # deslocamento sutil baseado no peso posicional (mais peso -> menos deslocamento)
        peso = idx_info.pesos_posicionais[pos]
        desloc = 1 if peso < np.median(idx_info.pesos_posicionais) else -1
        novo = val + desloc
        if novo < min_val:
            novo = min_val
        if novo > max_val:
            novo = max_val
        ipo_asb.append(int(novo))

    return IPFIPOInfo(ipf=ipf, ipo=ipo, ipo_asb=ipo_asb)
# -------------------------------------------------------------------
# S6 PROFUNDO ULTRA — GERADOR DETERMINÍSTICO
# -------------------------------------------------------------------


def _score_serie_idx(
    serie: List[int],
    idx_info: IDXInfo,
) -> float:
    """
    Score baseado em IDX (frequência global + posicional + pesos).
    """
    n_pass = len(serie)
    score = 0.0
    for pos, valor in enumerate(serie):
        fglob = idx_info.freq_global.get(int(valor), 0.0)
        fpos = idx_info.freq_posicional.get(pos, {}).get(int(valor), 0.0)
        w = idx_info.pesos_posicionais[pos]
        score += w * (0.7 * fpos + 0.3 * fglob)
    return score


def gerar_s6_profundo_ultra(
    df: pd.DataFrame,
    idx_info: IDXInfo,
    ipf_ipo: IPFIPOInfo,
    n_series: int = 60,
) -> pd.DataFrame:
    """
    Gera um leque determinístico (S6 Profundo ULTRA):
    - Combina IPF, IPO, IPO ASB.
    - Gera variações estruturais em torno desses núcleos.
    """
    mat = extrair_series_passageiros(df)
    n_pass = mat.shape[1]
    min_val, max_val = obter_faixa_numerica(df)

    base_candidates: List[List[int]] = [
        ipf_ipo.ipf,
        ipf_ipo.ipo,
        ipf_ipo.ipo_asb,
    ]

    # Adiciona algumas variações simples de IPF/IPO para enriquecer o leque determinístico
    for base in [ipf_ipo.ipf, ipf_ipo.ipo, ipf_ipo.ipo_asb]:
        for pos in range(n_pass):
            for delta in (-1, 1):
                nova = base.copy()
                novo_val = nova[pos] + delta
                if min_val <= novo_val <= max_val:
                    nova[pos] = novo_val
                    base_candidates.append(nova)

    # Remover duplicados
    series_unicas = []
    vistos = set()
    for s in base_candidates:
        t = tuple(int(x) for x in s)
        if t not in vistos:
            vistos.add(t)
            series_unicas.append(list(t))

    # Scorar
    registros = []
    for s in series_unicas:
        registros.append(
            {
                "series": s,
                "score_s6": _score_serie_idx(s, idx_info),
                "origem": "S6",
            }
        )

    df_s6 = pd.DataFrame(registros).sort_values("score_s6", ascending=False)
    if len(df_s6) > n_series:
        df_s6 = df_s6.iloc[:n_series].reset_index(drop=True)
    return df_s6.reset_index(drop=True)


# -------------------------------------------------------------------
# MICRO-LEQUE ULTRA — VIZINHANÇA FINA
# -------------------------------------------------------------------


def gerar_micro_leque_ultra(
    df_s6: pd.DataFrame,
    df_hist: pd.DataFrame,
    raio: int = 2,
    max_por_base: int = 10,
) -> pd.DataFrame:
    """
    Gera variações finas (Micro-Leque) em torno das séries do S6:
    - Para cada série base de S6, cria pequenas perturbações (±1, ±2) mantendo a faixa [min_val, max_val].
    - Pontua com base na proximidade à série original.
    """
    if df_s6.empty:
        return pd.DataFrame(columns=["series", "score_micro", "origem"])

    min_val, max_val = obter_faixa_numerica(df_hist)
    registros = []

    for _, row in df_s6.iterrows():
        base = row["series"]
        if isinstance(base, (np.ndarray, tuple)):
            base = list(base)
        n_pass = len(base)

        # Adiciona a própria série base como parte do micro-leque
        registros.append(
            {
                "series": list(base),
                "score_micro": float(len(base)),  # maior score por ser o núcleo
                "origem": "MicroLeque",
            }
        )

        contador = 0
        for pos in range(n_pass):
            for delta in range(-raio, raio + 1):
                if delta == 0:
                    continue
                nova = base.copy()
                novo_val = nova[pos] + delta
                if min_val <= novo_val <= max_val:
                    nova[pos] = novo_val
                    score = float(n_pass - abs(delta))  # quanto menos alteração, maior score
                    registros.append(
                        {
                            "series": nova,
                            "score_micro": score,
                            "origem": "MicroLeque",
                        }
                    )
                    contador += 1
                    if contador >= max_por_base:
                        break
            if contador >= max_por_base:
                break

    df_micro = pd.DataFrame(registros)
    # Consolidar por série (máximo score_micro)
    if not df_micro.empty:
        df_micro = normalizar_coluna_series(df_micro)
        df_micro = (
            df_micro.groupby("series", as_index=False)
            .agg(score_micro=("score_micro", "max"), origem=("origem", "first"))
            .sort_values("score_micro", ascending=False)
            .reset_index(drop=True)
        )
    return df_micro


# -------------------------------------------------------------------
# MONTE CARLO PROFUNDO ULTRA
# -------------------------------------------------------------------


def simular_monte_carlo_ultra(
    df_hist: pd.DataFrame,
    janela_mc: int = 40,
    n_sim: int = 500,
) -> pd.DataFrame:
    """
    Monte Carlo Profundo ULTRA:
    - Usa amostragem empírica por posição na janela recente.
    - Gera n_sim séries sintéticas de passageiros.
    - Score é a frequência empírica de cada série.
    """
    if df_hist.empty:
        return pd.DataFrame(columns=["series", "score_mc", "origem"])

    n_pass = obter_num_passageiros(df_hist)
    cols = [f"P{i+1}" for i in range(n_pass)]

    janela_real = min(janela_mc, len(df_hist))
    df_recent = df_hist.iloc[-janela_real:]

    # distribuição empírica por posição
    dist_pos: Dict[int, np.ndarray] = {}
    for pos, col in enumerate(cols):
        dist_pos[pos] = df_recent[col].to_numpy(dtype=int)

    # simulações
    simulacoes = []
    rng = np.random.default_rng(seed=42)  # determinismo para replays

    for _ in range(n_sim):
        serie = []
        for pos in range(n_pass):
            valores = dist_pos[pos]
            if len(valores) == 0:
                serie.append(0)
            else:
                serie.append(int(rng.choice(valores)))
        simulacoes.append(tuple(serie))

    # contagem de frequência
    valores, contagens = np.unique(simulacoes, return_counts=True)
    registros = []
    total = float(len(simulacoes))
    for serie, freq in zip(valores, contagens):
        registros.append(
            {
                "series": list(serie),
                "score_mc": float(freq) / total,
                "origem": "MonteCarlo",
            }
        )
    df_mc = pd.DataFrame(registros).sort_values("score_mc", ascending=False)
    return df_mc.reset_index(drop=True)


# -------------------------------------------------------------------
# NORMALIZAÇÃO SEGURA DA COLUNA "series"
# -------------------------------------------------------------------


def normalizar_coluna_series(df_flat: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza a coluna 'series' para garantir que seja hashable para groupby:
    - Remove linhas com series nula/vazia.
    - Converte qualquer formato (lista, np.array, string) em tupla de ints.
    """

    if "series" not in df_flat.columns:
        return df_flat

    def _to_tuple(x: Any) -> Optional[Tuple[int, ...]]:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None

        if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
            vals = list(x)
        else:
            # string ou outro tipo
            try:
                s = str(x)
            except Exception:
                return None
            # tenta extrair inteiros
            tokens = s.replace(",", " ").split()
            vals = []
            for t in tokens:
                try:
                    vals.append(int(t))
                except Exception:
                    continue

        vals = [int(v) for v in vals if str(v).strip() != ""]
        if not vals:
            return None
        return tuple(vals)

    df = df_flat.copy()
    df["series"] = df["series"].apply(_to_tuple)
    df = df[df["series"].notna()].copy()
    return df


def fundir_leques_ultra(
    df_s6: pd.DataFrame,
    df_mc: pd.DataFrame,
    df_micro: pd.DataFrame,
    peso_s6: float = 0.5,
    peso_mc: float = 0.3,
    peso_micro: float = 0.2,
) -> pd.DataFrame:
    """
    Fusão ULTRA:
    - Junta S6, Monte Carlo e Micro-Leque numa única tabela.
    - Normaliza 'series' antes de agrupar.
    - Calcula score_final = combinação ponderada dos scores disponíveis.
    """
    # Garantir presença das colunas
    for df_ref, col, origem in [
        (df_s6, "score_s6", "S6"),
        (df_mc, "score_mc", "MonteCarlo"),
        (df_micro, "score_micro", "MicroLeque"),
    ]:
        if df_ref is not None and not df_ref.empty:
            if "series" not in df_ref.columns:
                df_ref["series"] = [[] for _ in range(len(df_ref))]
            if col not in df_ref.columns:
                df_ref[col] = 0.0
            if "origem" not in df_ref.columns:
                df_ref["origem"] = origem

    frames = []
    if df_s6 is not None and not df_s6.empty:
        frames.append(df_s6[["series", "score_s6", "origem"]])
    if df_mc is not None and not df_mc.empty:
        frames.append(df_mc[["series", "score_mc", "origem"]])
    if df_micro is not None and not df_micro.empty:
        frames.append(df_micro[["series", "score_micro", "origem"]])

    if not frames:
        return pd.DataFrame(columns=["series", "score_final", "score_s6", "score_mc", "score_micro"])

    df_all = pd.concat(frames, ignore_index=True)

    # Normalização segura da coluna "series"
    df_all = normalizar_coluna_series(df_all)
    if df_all.empty or "series" not in df_all.columns:
        return pd.DataFrame(columns=["series", "score_final", "score_s6", "score_mc", "score_micro"])

    # Garantir colunas de score
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
        .copy()
    )

    agg["score_final"] = (
        peso_s6 * agg["score_s6"].fillna(0.0)
        + peso_mc * agg["score_mc"].fillna(0.0)
        + peso_micro * agg["score_micro"].fillna(0.0)
    )

    agg = agg.sort_values("score_final", ascending=False).reset_index(drop=True)
    return agg
# -------------------------------------------------------------------
# MONTAR PREVISÃO TURBO ULTRA — COM RETROCOMPATIBILIDADE
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

    - Gera IDX ULTRA
    - IPF / IPO / IPO ASB
    - S6 Profundo ULTRA
    - Monte Carlo Profundo ULTRA
    - Micro-Leque ULTRA (opcional)
    - Fusão ULTRA

    Retrocompatibilidade:
    Aceita também parâmetros antigos via **kwargs:
    - n_series_saida -> n_s6
    - window_s6     -> janela_s6
    - window_mc     -> janela_mc
    - incluir_micro_leque -> usar_micro_leque
    """

    # ---------------- RETROCOMPATIBILIDADE ----------------
    if n_s6 is None:
        n_s6 = kwargs.get("n_series_saida", 60)

    # Se veio explicitamente na chamada antiga, sobrepõe
    if "window_s6" in kwargs:
        janela_s6 = int(kwargs["window_s6"])
    if "window_mc" in kwargs:
        janela_mc = int(kwargs["window_mc"])
    if "incluir_micro_leque" in kwargs:
        usar_micro_leque = bool(kwargs["incluir_micro_leque"])

    # ---------------- BARRA DE PROGRESSO ----------------
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Etapa 1/4 — Construindo IDX ULTRA...")
        progress_bar.progress(5)
        idx_info = construir_idx_ultra(df_hist)

        status_text.text("Etapa 2/4 — Construindo IPF/IPO/IPO-ASB ULTRA...")
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
            df_hist,
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

        status_text.text("Fusão ULTRA — S6 + Monte Carlo + Micro-Leque...")
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
        # deixar a barra cheia e depois limpá-la na UI chamadora, se desejado
        progress_bar.progress(100)

    if df_fusao is None or df_fusao.empty:
        return pd.DataFrame(columns=["series", "score_final", "score_s6", "score_mc", "score_micro"])

    # Modo de saída
    df_out = df_fusao.copy()
    if output_mode == "top":
        return df_out
    elif output_mode == "detalhado":
        return df_out
    else:
        # fallback
        return df_out


# -------------------------------------------------------------------
# QDS REAL + BACKTEST REAL (ESTRUTURA)
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
    QDS REAL (estrutura):
    - Percorre uma janela do histórico simulando o Modo TURBO++ ULTRA
      como se estivesse no passado.
    - Para cada ponto, verifica se a série seguinte estaria entre as n_top
      séries de previsão.
    - Retorna métricas agregadas (taxa de acerto, curva de acerto, etc.).
    """
    if len(df_hist) < horizonte_teste + 5:
        return {
            "taxa_acerto": float("nan"),
            "n_testes": 0,
            "hits": [],
            "idx_testados": [],
        }

    hits = []
    idx_testados = []
    inicio = len(df_hist) - horizonte_teste - 1
    inicio = max(1, inicio)

    for i in range(inicio, len(df_hist) - 1):
        df_past = df_hist.iloc[: i + 1].copy()
        df_alvo = df_hist.iloc[i + 1]  # série seguinte

        try:
            barometro = analisar_barometro_ultra(df_past)
            df_pred = montar_previsao_turbo_ultra(
                df_past,
                regime_state=barometro.regime,
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
            # em caso de erro pontual, consideramos miss
            hits.append(0)
            idx_testados.append(i)
            continue

        if df_pred is None or df_pred.empty:
            hits.append(0)
            idx_testados.append(i)
            continue

        # série alvo (passageiros)
        n_pass = obter_num_passageiros(df_hist)
        alvo_series = tuple(int(df_alvo[f"P{j+1}"]) for j in range(n_pass))

        # normalizar predicted
        df_pred_norm = normalizar_coluna_series(df_pred)
        candidatos = df_pred_norm["series"].head(n_top).tolist()

        existe_hit = int(alvo_series in candidatos)
        hits.append(existe_hit)
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
# REPLAY LIGHT E REPLAY ULTRA
# -------------------------------------------------------------------


def resumo_replay_light(df_hist: pd.DataFrame, idx: int) -> Dict[str, Any]:
    """
    Replay LIGHT:
    - Mostra o estado local da estrada em torno da série idx.
    """
    if idx < 0 or idx >= len(df_hist):
        raise IndexError("Índice fora do histórico.")

    linha = df_hist.iloc[idx]
    n_pass = obter_num_passageiros(df_hist)
    serie = [int(linha[f"P{i+1}"]) for i in range(n_pass)]
    k_val = int(linha["k"])

    # contexto local (janela pequena)
    janela_local = max(5, min(20, len(df_hist)))
    start = max(0, idx - janela_local // 2)
    end = min(len(df_hist), idx + janela_local // 2)
    df_local = df_hist.iloc[start:end]

    barometro_local = analisar_barometro_ultra(df_local)
    kstar_local = calcular_kstar_ultra(df_local)

    return {
        "id": linha["ID"],
        "idx": idx,
        "serie": serie,
        "k": k_val,
        "barometro": barometro_local,
        "kstar": kstar_local,
    }


def executar_replay_ultra(
    df_hist: pd.DataFrame,
    inicio: int,
    fim: int,
    n_top: int = 10,
) -> Dict[str, Any]:
    """
    Replay ULTRA:
    - Para cada índice entre [inicio, fim], roda o Modo TURBO++ ULTRA no passado
      e vê se a série seguinte estaria entre as n_top previsões.
    - Retorna curva de acertos, similar ao QDS, mas focado num trecho específico.
    """
    if inicio < 1:
        inicio = 1
    if fim >= len(df_hist) - 1:
        fim = len(df_hist) - 2
    if inicio >= fim:
        return {"hits": [], "idxs": [], "taxa": float("nan")}

    hits = []
    idxs = []

    progress = st.progress(0)
    total = max(1, fim - inicio + 1)

    for c_idx, i in enumerate(range(inicio, fim + 1), start=1):
        progress.progress(int(100 * c_idx / total))
        df_past = df_hist.iloc[: i + 1].copy()
        df_alvo = df_hist.iloc[i + 1]

        try:
            barometro = analisar_barometro_ultra(df_past)
            df_pred = montar_previsao_turbo_ultra(
                df_past,
                regime_state=barometro.regime,
                output_mode="top",
                n_s6=n_top * 3,
                janela_s6=40,
                janela_mc=40,
                n_sim_mc=300,
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
        alvo_series = tuple(int(df_alvo[f"P{j+1}"]) for j in range(n_pass))

        df_pred_norm = normalizar_coluna_series(df_pred)
        candidatos = df_pred_norm["series"].head(n_top).tolist()

        existe_hit = int(alvo_series in candidatos)
        hits.append(existe_hit)
        idxs.append(i)

    progress.progress(100)
    taxa = float(np.mean(hits)) if hits else float("nan")
    return {"hits": hits, "idxs": idxs, "taxa": taxa}
# -------------------------------------------------------------------
# INTERFACE STREAMLIT — ESTADO GLOBAL
# -------------------------------------------------------------------

if "df" not in st.session_state:
    st.session_state["df"] = None


# -------------------------------------------------------------------
# SIDEBAR — NAVEGAÇÃO
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
    st.markdown("### Parâmetros Globais")

    n_s6_global = st.slider("N° base de séries S6 (n_s6)", 20, 200, 60, step=10)
    janela_s6_global = st.slider("Janela S6 (séries)", 10, 200, 40, step=5)
    janela_mc_global = st.slider("Janela Monte Carlo (séries)", 10, 200, 40, step=5)
    n_sim_mc_global = st.slider("N° simulações Monte Carlo (n_sim_mc)", 100, 2000, 500, step=100)

    st.markdown("#### Pesos de Fusão")
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

    if opc == "Enviar arquivo CSV":
        file = st.file_uploader("Selecione o arquivo CSV:", type=["csv"])
        if file is not None:
            try:
                df_raw = pd.read_csv(file, sep=None, engine="python")
                df = preparar_historico_v14_flex(df_raw)
                st.session_state["df"] = df
                st.success(f"Histórico carregado com sucesso! Séries: {len(df)}.")
                st.write("Prévia:")
                st.dataframe(df.head(20))
            except Exception as e:
                st.error(f"Erro ao carregar CSV: {e}")

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
                st.write("Prévia:")
                st.dataframe(df.head(20))
            except Exception as e:
                st.error(f"Erro ao processar histórico colado: {e}")

    if st.session_state.get("df") is not None:
        df = st.session_state["df"]
        st.markdown("### Resumo do histórico atual")
        st.write(f"Séries: {len(df)} | Passageiros por série: {obter_num_passageiros(df)}")
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
    st.write("Pesos posicional (IDX ULTRA):", idx_info.pesos_posicionais)

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
        df,
        janela_mc=janela_mc_global,
        n_sim=n_sim_mc_global,
    )
    st.dataframe(df_mc_prev.head(20))

    st.markdown("### 7. Micro-Leque ULTRA — Prévia")
    df_micro_prev = gerar_micro_leque_ultra(
        df_s6_prev,
        df_hist=df,
        raio=2,
        max_por_base=6,
    )
    st.dataframe(df_micro_prev.head(20))

    st.markdown("### 8. Fusão ULTRA — Prévia")
    df_fusao_prev = fundir_leques_ultra(
        df_s6_prev,
        df_mc_prev,
        df_micro_prev,
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
    st.markdown("### Interpretando o novo k")
    st.write(
        """
- **k** = número de guardas que acertaram exatamente o carro (todos os passageiros).
- **k*** = sensibilidade consolidada da estrada:
    - quanto maior, mais blocos de carros previsíveis estão se repetindo;
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
                incluir_micro_leque=usar_micro,  # retrocompatibilidade
                n_series_saida=n_s6_global,      # retrocompatibilidade extra
            )

        if df_turbo is None or df_turbo.empty:
            st.error("Modo TURBO++ ULTRA não retornou séries de previsão.")
        else:
            st.success("Modo TURBO++ ULTRA executado com sucesso.")
            df_turbo_view = df_turbo.copy()
            df_turbo_view = df_turbo_view.head(n_saida)
            st.markdown("### 🎯 Séries de Previsão — Top-N (TURBO++ ULTRA)")
            st.dataframe(df_turbo_view)

            # Previsão final = primeira série do ranking
            serie_final = df_turbo_view.iloc[0]["series"]
            if isinstance(serie_final, tuple):
                serie_final = list(serie_final)

            st.markdown("### 🔚 Previsão Final TURBO++ ULTRA")
            st.code(" ".join(str(x) for x in serie_final), language="text")

            # Contexto k*
            kstar = calcular_kstar_ultra(df)
            if kstar.estado == "estavel":
                contexto_k = "🟢 k*: Ambiente estável — previsão em regime normal."
            elif kstar.estado == "atencao":
                contexto_k = "🟡 k*: Pré-ruptura residual — usar previsão com atenção."
            else:
                contexto_k = "🔴 k*: Ambiente crítico — usar previsão com cautela máxima."
            st.info(contexto_k)


# -------------------------------------------------------------------
# PAINEL 5 — REPLAY AUTOMÁTICO DO HISTÓRICO
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
    with col1:
        idx_light = st.number_input(
            "Índice para Replay LIGHT (0 = primeira linha do DataFrame):",
            min_value=0,
            max_value=max(0, n_series - 1),
            value=max(0, n_series - 2),
            step=1,
        )
        if st.button("Executar Replay LIGHT"):
            try:
                resumo = resumo_replay_light(df, idx=int(idx_light))
                st.markdown("### Replay LIGHT — Estado Local")
                st.write(f"ID: **{resumo['id']}** (índice {resumo['idx']})")
                st.write("Série:", resumo["serie"])
                st.write("k:", resumo["k"])
                st.markdown("#### Barômetro local")
                st.info(resumo["barometro"].msg)
                st.markdown("#### k* local")
                st.info(resumo["kstar"].msg)
            except Exception as e:
                st.error(f"Erro no Replay LIGHT: {e}")

    with col2:
        st.markdown("### Replay ULTRA — fatiar um trecho do histórico")
        inicio = st.number_input(
            "Início (índice)", min_value=1, max_value=max(1, n_series - 3), value=max(1, n_series - 30)
        )
        fim = st.number_input(
            "Fim (índice)", min_value=2, max_value=max(2, n_series - 2), value=max(2, n_series - 5)
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
