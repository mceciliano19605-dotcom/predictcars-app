# app_v14_flex_replay_adaptativo.py
# Predict Cars V14-FLEX ULTRA REAL (TURBO++)
# Versão REPLAY + TURBO++ ULTRA ADAPTATIVO por k*

import math
import random
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# CONSTANTES E CONFIGURAÇÕES GERAIS
# ============================================================

MAX_PASSAGEIRO = 60  # valor máximo esperado para passageiros (ex.: loteria 60)
MIN_PASSAGEIRO = 1   # valor mínimo

# Sementes fixas para reprodutibilidade básica
RANDOM_SEED = 20251204
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ============================================================
# FUNÇÕES UTILITÁRIAS
# ============================================================

def limpar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    return texto.replace("\r", "\n").strip()


def detectar_separador(linha: str) -> str:
    if ";" in linha:
        return ";"
    if "," in linha:
        return ","
    # fallback: espaço
    return None


def to_int_safe(x: Any) -> Optional[int]:
    try:
        v = int(float(str(x).strip()))
        return v
    except Exception:
        return None


def normalizar_serie_lista(series: List[int]) -> List[int]:
    """Garante inteiros, remove None, ordena e mantém valores dentro de [MIN, MAX]."""
    limpos = []
    for x in series:
        v = to_int_safe(x)
        if v is None:
            continue
        if v < MIN_PASSAGEIRO or v > MAX_PASSAGEIRO:
            continue
        limpos.append(v)
    limpos = sorted(list(set(limpos)))
    return limpos


def series_to_str(s: List[int]) -> str:
    return " ".join(str(x) for x in s)


def series_to_tuple(s: Any) -> Tuple[int, ...]:
    if isinstance(s, (list, tuple, np.ndarray, pd.Series)):
        return tuple(int(x) for x in s)
    # se for string, tentar quebrar em espaços
    if isinstance(s, str):
        parts = s.strip().split()
        return tuple(int(p) for p in parts if p.strip().isdigit())
    return tuple()


def extrair_colunas_passageiros(df: pd.DataFrame) -> List[str]:
    """Detecta automaticamente colunas de passageiros (p1..pN)."""
    # Preferência por nomes p1, p2,... se existirem
    pcols = [c for c in df.columns if str(c).lower().startswith("p")]
    if pcols:
        return pcols

    # Caso não haja p1..pN, pegar todas numéricas menos k
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return []
    # Detectar col k como última numérica ou coluna com valores pequenos
    if "k" in df.columns:
        k_col = "k"
    else:
        # heurística: última coluna numérica como k
        k_col = num_cols[-1]
    pcols = [c for c in num_cols if c != k_col]
    return pcols


def detectar_coluna_k(df: pd.DataFrame) -> str:
    if "k" in df.columns:
        return "k"
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return "k"  # fallback
    # heurística: coluna com menor valor máximo tende a ser k
    stats = [(c, df[c].max()) for c in num_cols]
    stats_sorted = sorted(stats, key=lambda x: x[1])
    return stats_sorted[0][0]


# ============================================================
# PREPARAÇÃO DO HISTÓRICO FLEX (n variável de passageiros + k)
# ============================================================

def preparar_historico_flex_from_csv(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Prepara histórico a partir de um CSV genérico, detectando passageiros + k."""
    df = df_raw.copy()

    # Remover colunas totalmente vazias
    df = df.dropna(how="all", axis=1)

    # Se não tiver numéricas, tentar converter tudo que for possível
    if df.select_dtypes(include=[np.number]).empty:
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="ignore")

    # Detectar coluna k e colunas de passageiros
    k_col = detectar_coluna_k(df)
    if k_col not in df.columns:
        k_col = "k"  # garante a existência
        df[k_col] = 0

    pcols = extrair_colunas_passageiros(df)
    if not pcols:
        # fallback: se só tiver uma numérica, tratar como k
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) == 1:
            df["k"] = df[num_cols[0]]
            pcols = []
        else:
            pcols = [c for c in df.columns if c != k_col]

    # Criar ID se não existir
    if "id" not in df.columns:
        df["id"] = [f"C{i+1}" for i in range(len(df))]

    # Reordenar colunas: id, p1..pN, k
    # Renomear pcols para p1..pN
    pcols_sorted = pcols
    # garantir ordem estável
    pcols_sorted = list(pcols_sorted)
    rename_map = {}
    for i, c in enumerate(pcols_sorted, start=1):
        rename_map[c] = f"p{i}"
    df = df.rename(columns=rename_map)
    pcols_final = [rename_map[c] for c in pcols_sorted]

    # Cast para int sempre que possível
    for c in pcols_final + [k_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    df = df[["id"] + pcols_final + [k_col]].copy()
    df = df.reset_index(drop=True)
    df.columns = ["id"] + pcols_final + ["k"]
    return df


def preparar_historico_flex_from_text(texto: str) -> pd.DataFrame:
    texto = limpar_texto(texto)
    if not texto:
        return pd.DataFrame()

    linhas = [l.strip() for l in texto.split("\n") if l.strip()]
    if not linhas:
        return pd.DataFrame()

    sep = detectar_separador(linhas[0])
    registros = []
    for i, linha in enumerate(linhas):
        if sep:
            partes = [p.strip() for p in linha.split(sep) if p.strip() != ""]
        else:
            partes = [p.strip() for p in linha.replace(",", " ").replace(";", " ").split() if p.strip()]

        if not partes:
            continue

        # Se primeira parte for algo como "C10", considerar ID
        first = partes[0]
        if first.upper().startswith("C") and first[1:].isdigit():
            cid = first
            nums = [to_int_safe(x) for x in partes[1:]]
        else:
            cid = f"C{i+1}"
            nums = [to_int_safe(x) for x in partes]

        nums = [n for n in nums if n is not None]
        if not nums:
            continue

        if len(nums) == 1:
            # apenas k
            passageiros = []
            k_val = nums[0]
        else:
            passageiros = nums[:-1]
            k_val = nums[-1]

        registros.append(
            {
                "id": cid,
                "passageiros": normalizar_serie_lista(passageiros),
                "k": k_val,
            }
        )

    if not registros:
        return pd.DataFrame()

    # Descobrir número máximo de passageiros
    max_len = max(len(r["passageiros"]) for r in registros)
    data_rows = []
    for r in registros:
        linha = {"id": r["id"], "k": r["k"]}
        for i in range(max_len):
            col = f"p{i+1}"
            if i < len(r["passageiros"]):
                linha[col] = r["passageiros"][i]
            else:
                linha[col] = 0
        data_rows.append(linha)

    df = pd.DataFrame(data_rows)
    cols = ["id"] + [c for c in df.columns if c.startswith("p")] + ["k"]
    df = df[cols].copy()
    return df.reset_index(drop=True)


# ============================================================
# BARÔMETRO ULTRA REAL (diagnóstico da estrada)
# ============================================================

def calcular_barometro_ultra_real(df: pd.DataFrame, janela: int = 60) -> Dict[str, Any]:
    """
    Barômetro ULTRA REAL baseado na dinâmica de k.
    Usa uma janela recente para medir:
      - k médio
      - frequência de k=0
      - frequência de k>0
      - volatilidade de k
    """
    if df is None or df.empty or "k" not in df.columns:
        return {
            "estado": "indefinido",
            "k_medio": 0.0,
            "freq_k_zero": 1.0,
            "freq_k_pos": 0.0,
            "vol_k": 0.0,
        }

    sub = df.tail(janela).copy()
    k_vals = sub["k"].astype(float).values
    if len(k_vals) == 0:
        return {
            "estado": "indefinido",
            "k_medio": 0.0,
            "freq_k_zero": 1.0,
            "freq_k_pos": 0.0,
            "vol_k": 0.0,
        }

    k_medio = float(np.mean(k_vals))
    freq_k_zero = float(np.mean(k_vals == 0))
    freq_k_pos = float(np.mean(k_vals > 0))
    vol_k = float(np.std(k_vals))

    # Heurística de regime para o barômetro:
    # - Estrada estável: k_medio razoável e freq_k_zero baixa
    # - Pré-ruptura: k_medio caindo / freq_k_zero intermediária / vol_k moderada
    # - Ruptura: freq_k_zero muito alta, k_medio muito baixo
    if freq_k_zero > 0.8 and k_medio < 0.5:
        estado = "ruptura"
    elif freq_k_zero > 0.5 or vol_k > 2.0:
        estado = "transicao"
    else:
        estado = "estavel"

    return {
        "estado": estado,
        "k_medio": k_medio,
        "freq_k_zero": freq_k_zero,
        "freq_k_pos": freq_k_pos,
        "vol_k": vol_k,
    }


# ============================================================
# k* ULTRA REAL (sentinela dos guardas)
# ============================================================

def calcular_k_estrela(df: pd.DataFrame, janela: int = 80) -> float:
    """
    k* ULTRA REAL: índice em [0, 100] baseado na dinâmica de k.
    Ideia:
      - Peso forte na frequência de k>0
      - Peso moderado no k médio relativo ao máximo observado
      - Penalização se a volatilidade estiver muito alta (instabilidade).
    """
    if df is None or df.empty or "k" not in df.columns:
        return 0.0

    sub = df.tail(janela).copy()
    if sub.empty:
        return 0.0

    k_vals = sub["k"].astype(float).values
    if len(k_vals) == 0:
        return 0.0

    freq_k_pos = float(np.mean(k_vals > 0))
    k_medio = float(np.mean(k_vals))
    k_max = float(np.max(k_vals)) if np.max(k_vals) > 0 else 1.0
    vol_k = float(np.std(k_vals))

    componente_freq = freq_k_pos  # [0,1]
    componente_intensidade = (k_medio / k_max)  # normalizado [0,1]
    # volatilidade alta reduz k* (instabilidade)
    penal_vol = 1.0 / (1.0 + vol_k)  # entre (0,1]

    k_star = 100.0 * (0.6 * componente_freq + 0.3 * componente_intensidade) * penal_vol
    k_star = max(0.0, min(100.0, k_star))
    return float(k_star)


# ============================================================
# DETERMINAÇÃO DE REGIME POR k* + (opcional) QDS local
# ============================================================

def determinar_regime_por_kstar(k_estrela: float, qds_local: Optional[float] = None) -> str:
    """
    Retorna um dos regimes:
      - "padrao"     (verde)
      - "transicao"  (amarelo)
      - "ruptura"    (vermelho)
    Driver principal é k*; QDS local pode apenas *piorar* o regime (nunca melhorar).
    """
    if k_estrela >= 66.0:
        regime = "padrao"
    elif k_estrela >= 33.0:
        regime = "transicao"
    else:
        regime = "ruptura"

    if qds_local is not None:
        # Se QDS muito baixo, forçar ruptura
        if qds_local < 0.05:
            regime = "ruptura"
        elif qds_local < 0.15 and regime == "padrao":
            regime = "transicao"

    return regime


def obter_pesos_por_regime(regime: str) -> Tuple[float, float, float]:
    """
    Retorna (peso_s6, peso_mc, peso_micro) para cada regime.
      - Regime de Padrão (verde):
          S6 ALTO, Micro MÉDIO, Monte Carlo BAIXO
      - Regime de Transição (amarelo):
          S6 MÉDIO, Micro MÉDIO-ALTO, Monte Carlo MÉDIO
      - Regime de Ruptura (vermelho):
          S6 BAIXO, Micro BAIXO-MÉDIO, Monte Carlo ALTO
    """
    if regime == "padrao":
        return (0.7, 0.1, 0.2)
    elif regime == "transicao":
        return (0.33, 0.33, 0.34)
    else:  # ruptura
        return (0.1, 0.7, 0.2)


def descricao_regime(regime: str) -> str:
    if regime == "padrao":
        return "🟢 Regime de padrão — S6 dominante, estrada com padrão profundo."
    elif regime == "transicao":
        return "🟡 Regime de transição — mistura equilibrada, pré-ruptura / mudança de regime."
    else:
        return "🔴 Regime de ruptura — Monte Carlo dominante, foco em previsibilidade curta."


# ============================================================
# QDS REAL (métrica de qualidade dinâmica via backtest simples)
# ============================================================

def calcular_qds_backtest_simples(
    df: pd.DataFrame,
    passenger_cols: List[str],
    janela: int = 40,
    top_n: int = 10,
) -> float:
    """
    QDS REAL aproximado: backtest retroativo básico usando o próprio motor TURBO
    de forma simplificada (apenas Monte Carlo + S6) para medir taxa de acertos
    em janelas recentes.
    IMPORTANTE: Esta função é usada como apoio para o regime, mas o motor
    completo do Replay ULTRA usará uma versão mais detalhada.
    """
    if df is None or df.empty or len(df) < 10:
        return 0.0

    # evitamos loops gigantes
    janela = min(janela, len(df) - 2)
    if janela <= 0:
        return 0.0

    # índice final
    end_idx = len(df) - 2  # último índice em que ainda há um "próximo"
    start_idx = max(0, end_idx - janela + 1)

    hits = 0
    total = 0

    for i in range(start_idx, end_idx + 1):
        df_hist = df.iloc[: i + 1].copy()
        if len(df_hist) < 5:
            continue

        # Série alvo é a última do df_hist; "real" é df.iloc[i+1]
        idx_alvo_local = len(df_hist) - 1

        try:
            from_series = df_hist.iloc[idx_alvo_local][passenger_cols].values.astype(int).tolist()
        except Exception:
            continue

        # Monte Carlo simplificado baseado na janela recente
        candidatos = gerar_previsoes_monte_carlo_simples(
            df_hist,
            passenger_cols=passenger_cols,
            n_series=300,
        )

        # S6 simplificado (similaridade local)
        s6_cands = gerar_previsoes_s6_profundo_simples(
            df_hist,
            passenger_cols=passenger_cols,
            idx_alvo=idx_alvo_local,
            n_top=120,
        )

        # Fusão simples (pesos fixos para QDS)
        df_fusao = fundir_candidatos_basico_para_qds(s6_cands, candidatos, top_n=top_n)
        if df_fusao is None or df_fusao.empty:
            continue

        top_series = [series_to_tuple(s) for s in df_fusao["series"].tolist()[:top_n]]

        real_series = df.iloc[i + 1][passenger_cols].values.astype(int).tolist()
        real_tuple = series_to_tuple(real_series)

        total += 1
        if real_tuple in top_series:
            hits += 1

    if total == 0:
        return 0.0

    return float(hits / total)


# ============================================================
# FUNÇÕES AUXILIARES PARA QDS (Monte Carlo / S6 / Fusão simples)
# ============================================================

def gerar_previsoes_monte_carlo_simples(
    df: pd.DataFrame,
    passenger_cols: List[str],
    n_series: int = 300,
) -> pd.DataFrame:
    """Monte Carlo simplificado para apoio ao QDS."""
    if df.empty:
        return pd.DataFrame(columns=["series", "score_mc"])

    sub = df.tail(80).copy()
    distros = {}
    for c in passenger_cols:
        vals = sub[c].values
        vals = [v for v in vals if MIN_PASSAGEIRO <= v <= MAX_PASSAGEIRO]
        if not vals:
            vals = list(range(MIN_PASSAGEIRO, MAX_PASSAGEIRO + 1))
        distros[c] = vals

    geradas = []
    for _ in range(n_series):
        s = []
        for c in passenger_cols:
            vals = distros[c]
            v = random.choice(vals)
            s.append(v)
        s = normalizar_serie_lista(s)
        if len(s) != len(passenger_cols):
            # se perder tamanho por causa de duplicatas, tentar de novo
            while len(s) < len(passenger_cols):
                v_extra = random.randint(MIN_PASSAGEIRO, MAX_PASSAGEIRO)
                if v_extra not in s:
                    s.append(v_extra)
            s = sorted(s)
        geradas.append(tuple(s))

    # Contagem e score
    cont = {}
    for s in geradas:
        cont[s] = cont.get(s, 0) + 1

    rows = []
    for s, freq in cont.items():
        rows.append(
            {
                "series": list(s),
                "score_mc": 1.0 / (freq + 1e-9),
            }
        )

    df_mc = pd.DataFrame(rows).sort_values("score_mc", ascending=True).reset_index(drop=True)
    return df_mc


def gerar_previsoes_s6_profundo_simples(
    df: pd.DataFrame,
    passenger_cols: List[str],
    idx_alvo: int,
    n_top: int = 120,
) -> pd.DataFrame:
    """S6 profundo simplificado apenas para QDS."""
    if df.empty or idx_alvo <= 0:
        return pd.DataFrame(columns=["series", "score_s6"])

    target = df.iloc[idx_alvo][passenger_cols].values.astype(float)
    registros = []

    for i in range(0, idx_alvo):
        if i + 1 >= len(df):
            break
        atual = df.iloc[i][passenger_cols].values.astype(float)
        prox = df.iloc[i + 1][passenger_cols].values.astype(int).tolist()
        dist = float(np.linalg.norm(atual - target, ord=1))
        registros.append((tuple(prox), dist))

    if not registros:
        return pd.DataFrame(columns=["series", "score_s6"])

    # agrupar por série futura
    dist_map: Dict[Tuple[int, ...], List[float]] = {}
    for serie, d in registros:
        dist_map.setdefault(serie, []).append(d)

    rows = []
    for serie, ds in dist_map.items():
        score = float(np.mean(ds))
        rows.append({"series": list(serie), "score_s6": score})

    df_s6 = pd.DataFrame(rows).sort_values("score_s6", ascending=True)
    df_s6 = df_s6.head(n_top).reset_index(drop=True)
    return df_s6


def fundir_candidatos_basico_para_qds(
    df_s6: pd.DataFrame,
    df_mc: pd.DataFrame,
    top_n: int = 10,
) -> Optional[pd.DataFrame]:
    if (df_s6 is None or df_s6.empty) and (df_mc is None or df_mc.empty):
        return None

    # Ranking interno
    def add_rank(df: pd.DataFrame, col_score: str, col_rank: str) -> pd.DataFrame:
        df = df.copy()
        df[col_rank] = np.arange(1, len(df) + 1)
        return df

    if df_s6 is not None and not df_s6.empty:
        df_s6 = df_s6.sort_values("score_s6", ascending=True).reset_index(drop=True)
        df_s6 = add_rank(df_s6, "score_s6", "rank_s6")
    else:
        df_s6 = pd.DataFrame(columns=["series", "score_s6", "rank_s6"])

    if df_mc is not None and not df_mc.empty:
        df_mc = df_mc.sort_values("score_mc", ascending=True).reset_index(drop=True)
        df_mc = add_rank(df_mc, "score_mc", "rank_mc")
    else:
        df_mc = pd.DataFrame(columns=["series", "score_mc", "rank_mc"])

    # União de séries
    all_keys = set()
    for s in df_s6["series"].tolist():
        all_keys.add(series_to_tuple(s))
    for s in df_mc["series"].tolist():
        all_keys.add(series_to_tuple(s))

    rows = []
    for key in all_keys:
        s_list = list(key)
        row = {"series": s_list}

        # rank S6
        rank_s6 = 9999
        if not df_s6.empty:
            mask = df_s6["series"].apply(lambda x: series_to_tuple(x) == key)
            if mask.any():
                rank_s6 = int(df_s6.loc[mask, "rank_s6"].iloc[0])
        row["rank_s6"] = rank_s6

        # rank MC
        rank_mc = 9999
        if not df_mc.empty:
            mask = df_mc["series"].apply(lambda x: series_to_tuple(x) == key)
            if mask.any():
                rank_mc = int(df_mc.loc[mask, "rank_mc"].iloc[0])
        row["rank_mc"] = rank_mc

        # fusão simples fixa para QDS
        peso_s6 = 0.6
        peso_mc = 0.4
        score_fusao = peso_s6 * rank_s6 + peso_mc * rank_mc
        row["score_fusao_qds"] = score_fusao

        rows.append(row)

    df_fusao = pd.DataFrame(rows).sort_values("score_fusao_qds", ascending=True).reset_index(drop=True)
    return df_fusao.head(max(top_n * 3, top_n))
# ============================================================
# IDX ULTRA (núcleo ponderado da estrada)
# ============================================================

def calcular_idx_ultra(df: pd.DataFrame) -> Dict[str, Any]:
    """
    IDX ULTRA: índice central da estrada, com médias ponderadas por k.
    Retorna:
      - idx_passageiros: média ponderada por posição (p1..pN)
      - idx_global: média de todos os passageiros ponderada
    """
    if df is None or df.empty:
        return {"idx_passageiros": {}, "idx_global": 0.0}

    passenger_cols = [c for c in df.columns if c.startswith("p")]
    if not passenger_cols:
        return {"idx_passageiros": {}, "idx_global": 0.0}

    k_vals = df["k"].astype(float).values
    w = k_vals + 1.0  # todo mundo conta, mas k>0 pesa mais

    idx_pass = {}
    all_vals = []

    for c in passenger_cols:
        vals = df[c].astype(float).values
        all_vals.extend(vals.tolist())
        if np.sum(w) == 0:
            m = float(np.mean(vals))
        else:
            m = float(np.average(vals, weights=w))
        idx_pass[c] = m

    if all_vals:
        if np.sum(w) == 0:
            idx_global = float(np.mean(all_vals))
        else:
            # aproximar usando média de idx_pass
            idx_global = float(np.mean(list(idx_pass.values())))
    else:
        idx_global = 0.0

    return {
        "idx_passageiros": idx_pass,
        "idx_global": idx_global,
    }


# ============================================================
# IPF / IPO REFINADOS (índices de padrão futuro / atual)
# ============================================================

def calcular_ipf_ipo(df: pd.DataFrame) -> Dict[str, Any]:
    """
    IPF (Índice de Padrão Futuro) e IPO (Índice Padrão Atual).
    Implementação simplificada porém real: correlações e tendências entre
    séries consecutivas, ponderadas por k.
    """
    if df is None or df.empty or len(df) < 2:
        return {"ipf": 0.0, "ipo": 0.0}

    passenger_cols = [c for c in df.columns if c.startswith("p")]
    if not passenger_cols:
        return {"ipf": 0.0, "ipo": 0.0}

    k_vals = df["k"].astype(float).values
    w = k_vals + 1.0

    # IPO: estabilidade local do padrão (variação média entre séries consecutivas)
    diffs = []
    for i in range(len(df) - 1):
        a = df.iloc[i][passenger_cols].values.astype(float)
        b = df.iloc[i + 1][passenger_cols].values.astype(float)
        diffs.append(np.linalg.norm(a - b, ord=1))
    if not diffs:
        ipo = 0.0
    else:
        diffs_arr = np.array(diffs)
        ipo = float(1.0 / (1.0 + np.mean(diffs_arr)))  # quanto menor a diferença, maior o IPO

    # IPF: "alinhamento" do futuro com o presente ponderado por k
    # Aqui tomamos a correlação média entre a série e a seguinte
    corrs = []
    for i in range(len(df) - 1):
        a = df.iloc[i][passenger_cols].values.astype(float)
        b = df.iloc[i + 1][passenger_cols].values.astype(float)
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        c = np.corrcoef(a, b)[0, 1]
        corrs.append(c)
    if not corrs:
        ipf = 0.0
    else:
        ipf = float(np.mean(corrs))

    return {"ipf": ipf, "ipo": ipo}


# ============================================================
# S6 PROFUNDO ULTRA — Núcleo determinístico
# ============================================================

def gerar_previsoes_s6_profundo_ultra(
    df: pd.DataFrame,
    idx_alvo: int,
    n_top: int = 200,
) -> pd.DataFrame:
    """
    S6 Profundo ULTRA:
      - baseia-se na similaridade entre a série alvo e séries anteriores
      - utiliza a série seguinte de cada vizinho como candidato futuro
      - agrega as distâncias para formar um score determinístico
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["series", "score_s6", "origem"])

    passenger_cols = [c for c in df.columns if c.startswith("p")]
    if not passenger_cols:
        return pd.DataFrame(columns=["series", "score_s6", "origem"])

    if idx_alvo <= 0 or idx_alvo >= len(df):
        idx_alvo = len(df) - 1

    target = df.iloc[idx_alvo][passenger_cols].values.astype(float)

    registros = []
    for i in range(0, idx_alvo):
        if i + 1 >= len(df):
            break
        atual = df.iloc[i][passenger_cols].values.astype(float)
        prox = df.iloc[i + 1][passenger_cols].values.astype(int).tolist()
        dist = float(np.linalg.norm(atual - target, ord=1))
        cid_atual = df.iloc[i]["id"]
        cid_prox = df.iloc[i + 1]["id"]
        registros.append((tuple(prox), dist, (cid_atual, cid_prox)))

    if not registros:
        return pd.DataFrame(columns=["series", "score_s6", "origem"])

    serie_map: Dict[Tuple[int, ...], List[float]] = {}
    origem_map: Dict[Tuple[int, ...], List[Tuple[str, str]]] = {}

    for serie, dist, origem in registros:
        serie_map.setdefault(serie, []).append(dist)
        origem_map.setdefault(serie, []).append(origem)

    rows = []
    for serie, ds in serie_map.items():
        score = float(np.mean(ds))
        origens = origem_map.get(serie, [])
        rows.append(
            {
                "series": list(serie),
                "score_s6": score,
                "origem": origens,
            }
        )

    df_s6 = pd.DataFrame(rows).sort_values("score_s6", ascending=True)
    df_s6 = df_s6.head(n_top).reset_index(drop=True)
    return df_s6


# ============================================================
# MICRO-LEQUE ULTRA — Variações finas em torno dos núcleos
# ============================================================

def gerar_micro_leque_ultra(
    base_df: pd.DataFrame,
    n_micro_por_serie: int = 15,
) -> pd.DataFrame:
    """
    Micro-Leque ULTRA:
      - recebe séries base (por ex. vindas do S6 ou da fusão parcial)
      - gera pequenas variações locais (±1, ±2) em alguns passageiros
      - mantém o tamanho e a faixa [MIN, MAX]
    """
    if base_df is None or base_df.empty or "series" not in base_df.columns:
        return pd.DataFrame(columns=["series", "score_micro"])

    geradas = []
    for _, row in base_df.iterrows():
        base_series = row["series"]
        base_series = normalizar_serie_lista(base_series)
        if not base_series:
            continue

        for _ in range(n_micro_por_serie):
            s = base_series.copy()
            # escolher 1 ou 2 posições para perturbar
            n_pert = random.choice([1, 2])
            for __ in range(n_pert):
                idx = random.randint(0, len(s) - 1)
                delta = random.choice([-2, -1, 1, 2])
                novo_val = s[idx] + delta
                if novo_val < MIN_PASSAGEIRO:
                    novo_val = MIN_PASSAGEIRO
                if novo_val > MAX_PASSAGEIRO:
                    novo_val = MAX_PASSAGEIRO
                s[idx] = novo_val
            s = normalizar_serie_lista(s)
            if len(s) != len(base_series):
                continue
            geradas.append(tuple(s))

    if not geradas:
        return pd.DataFrame(columns=["series", "score_micro"])

    cont = {}
    for s in geradas:
        cont[s] = cont.get(s, 0) + 1

    rows = []
    for s, freq in cont.items():
        rows.append({"series": list(s), "score_micro": 1.0 / (freq + 1e-9)})

    df_micro = pd.DataFrame(rows).sort_values("score_micro", ascending=True).reset_index(drop=True)
    return df_micro


# ============================================================
# MONTE CARLO PROFUNDO ULTRA — simulações em janela configurável
# ============================================================

def gerar_monte_carlo_profundo_ultra(
    df: pd.DataFrame,
    passenger_cols: List[str],
    n_series: int = 800,
    janela: int = 100,
) -> pd.DataFrame:
    """
    Monte Carlo Profundo ULTRA:
      - usa a janela recente para estimar distribuições empíricas
      - gera muitas séries aleatórias consistentes com a estrada local
      - foca previsibilidade curta (caos / ruptura)
    """
    if df is None or df.empty or not passenger_cols:
        return pd.DataFrame(columns=["series", "score_mc"])

    sub = df.tail(janela).copy()
    if sub.empty:
        sub = df.copy()

    distros = {}
    for c in passenger_cols:
        vals = sub[c].values
        vals = [v for v in vals if MIN_PASSAGEIRO <= v <= MAX_PASSAGEIRO]
        if not vals:
            vals = list(range(MIN_PASSAGEIRO, MAX_PASSAGEIRO + 1))
        distros[c] = vals

    geradas = []
    for _ in range(n_series):
        s = []
        for c in passenger_cols:
            vals = distros[c]
            v = random.choice(vals)
            s.append(v)
        s = normalizar_serie_lista(s)
        if len(s) != len(passenger_cols):
            # se perder tamanho, preenche com valores extras
            while len(s) < len(passenger_cols):
                v_extra = random.randint(MIN_PASSAGEIRO, MAX_PASSAGEIRO)
                if v_extra not in s:
                    s.append(v_extra)
            s = sorted(s)
        geradas.append(tuple(s))

    cont = {}
    for s in geradas:
        cont[s] = cont.get(s, 0) + 1

    rows = []
    for s, freq in cont.items():
        rows.append({"series": list(s), "score_mc": 1.0 / (freq + 1e-9)})

    df_mc = pd.DataFrame(rows).sort_values("score_mc", ascending=True).reset_index(drop=True)
    return df_mc


# ============================================================
# FUSÃO ULTRA ADAPTATIVA (S6 + MC + Micro) por regime
# ============================================================

def adicionar_rank(df: pd.DataFrame, col_score: str, col_rank: str) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(col_score, ascending=True).reset_index(drop=True)
    df[col_rank] = np.arange(1, len(df) + 1)
    return df


def fundir_candidatos_ultra_adaptativo(
    df_s6: pd.DataFrame,
    df_mc: pd.DataFrame,
    df_micro: pd.DataFrame,
    pesos: Tuple[float, float, float],
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Fusão ULTRA adaptativa:
      - combina S6, Monte Carlo e Micro-Leque via ranks e pesos
      - pesos dependem do regime (k* / QDS)
    """
    peso_s6, peso_mc, peso_micro = pesos

    if (df_s6 is None or df_s6.empty) and (df_mc is None or df_mc.empty) and (df_micro is None or df_micro.empty):
        return pd.DataFrame(columns=["series", "score_fusao", "rank_fusao"])

    if df_s6 is not None and not df_s6.empty:
        df_s6 = adicionar_rank(df_s6, "score_s6", "rank_s6")
    else:
        df_s6 = pd.DataFrame(columns=["series", "score_s6", "rank_s6"])

    if df_mc is not None and not df_mc.empty:
        df_mc = adicionar_rank(df_mc, "score_mc", "rank_mc")
    else:
        df_mc = pd.DataFrame(columns=["series", "score_mc", "rank_mc"])

    if df_micro is not None and not df_micro.empty:
        df_micro = adicionar_rank(df_micro, "score_micro", "rank_micro")
    else:
        df_micro = pd.DataFrame(columns=["series", "score_micro", "rank_micro"])

    # União de chaves
    all_keys = set()
    for s in df_s6["series"].tolist():
        all_keys.add(series_to_tuple(s))
    for s in df_mc["series"].tolist():
        all_keys.add(series_to_tuple(s))
    for s in df_micro["series"].tolist():
        all_keys.add(series_to_tuple(s))

    rows = []
    for key in all_keys:
        s_list = list(key)
        row = {"series": s_list}

        rank_s6 = 9999
        if not df_s6.empty:
            mask = df_s6["series"].apply(lambda x: series_to_tuple(x) == key)
            if mask.any():
                rank_s6 = int(df_s6.loc[mask, "rank_s6"].iloc[0])
        row["rank_s6"] = rank_s6

        rank_mc = 9999
        if not df_mc.empty:
            mask = df_mc["series"].apply(lambda x: series_to_tuple(x) == key)
            if mask.any():
                rank_mc = int(df_mc.loc[mask, "rank_mc"].iloc[0])
        row["rank_mc"] = rank_mc

        rank_micro = 9999
        if not df_micro.empty:
            mask = df_micro["series"].apply(lambda x: series_to_tuple(x) == key)
            if mask.any():
                rank_micro = int(df_micro.loc[mask, "rank_micro"].iloc[0])
        row["rank_micro"] = rank_micro

        score = (
            peso_s6 * rank_s6 +
            peso_mc * rank_mc +
            peso_micro * rank_micro
        )
        row["score_fusao"] = float(score)

        rows.append(row)

    df_mix = pd.DataFrame(rows).sort_values("score_fusao", ascending=True).reset_index(drop=True)
    df_mix["rank_fusao"] = np.arange(1, len(df_mix) + 1)
    return df_mix.head(max(top_n * 3, top_n))


# ============================================================
# MOTOR TURBO++ ULTRA ADAPTATIVO (para um índice alvo)
# ============================================================

def executar_turbo_ultra_adaptativo_para_indice(
    df: pd.DataFrame,
    idx_alvo: int,
    top_n: int = 20,
    n_s6: int = 200,
    n_mc: int = 800,
    n_micro: int = 20,
) -> Dict[str, Any]:
    """
    Executa todo o motor TURBO++ ULTRA ADAPTATIVO para um índice alvo:
      - S6 Profundo ULTRA
      - Monte Carlo Profundo ULTRA
      - Micro-Leque ULTRA
      - Fusão adaptativa por regime (k* + QDS)
    Retorna:
      {
        "df_s6": ...,
        "df_mc": ...,
        "df_micro": ...,
        "df_fusao": ...,
        "k_estrela": float,
        "regime": "padrao|transicao|ruptura",
        "pesos": (s6, mc, micro),
        "qds_local": float,
      }
    """
    res = {
        "df_s6": pd.DataFrame(),
        "df_mc": pd.DataFrame(),
        "df_micro": pd.DataFrame(),
        "df_fusao": pd.DataFrame(),
        "k_estrela": 0.0,
        "regime": "ruptura",
        "pesos": (0.1, 0.7, 0.2),
        "qds_local": 0.0,
    }

    if df is None or df.empty:
        return res

    passenger_cols = [c for c in df.columns if c.startswith("p")]
    if not passenger_cols:
        return res

    if idx_alvo < 0:
        idx_alvo = 0
    if idx_alvo >= len(df):
        idx_alvo = len(df) - 1

    # QDS local aproximado em janela curta ao redor do alvo
    # usamos apenas a parte até o índice alvo (para não olhar futuro)
    df_hist = df.iloc[: idx_alvo + 1].copy()
    qds_local = calcular_qds_backtest_simples(
        df_hist,
        passenger_cols=passenger_cols,
        janela=min(20, len(df_hist) - 2) if len(df_hist) > 2 else 0,
        top_n=min(top_n, 15),
    )

    # k* baseado também apenas no histórico disponível até o alvo
    k_star = calcular_k_estrela(df_hist, janela=80)

    regime = determinar_regime_por_kstar(k_star, qds_local=qds_local)
    pesos = obter_pesos_por_regime(regime)

    # S6 Profundo ULTRA
    df_s6 = gerar_previsoes_s6_profundo_ultra(
        df_hist,
        idx_alvo=len(df_hist) - 1,
        n_top=n_s6,
    )

    # Monte Carlo Profundo ULTRA
    df_mc = gerar_monte_carlo_profundo_ultra(
        df_hist,
        passenger_cols=passenger_cols,
        n_series=n_mc,
        janela=120,
    )

    # Micro-Leque ULTRA baseado na saída do S6 (se vazio, cair para MC)
    base_micro = df_s6 if not df_s6.empty else df_mc
    df_micro = gerar_micro_leque_ultra(base_micro, n_micro_por_serie=n_micro)

    # Fusão adaptativa
    df_fusao = fundir_candidatos_ultra_adaptativo(
        df_s6=df_s6,
        df_mc=df_mc,
        df_micro=df_micro,
        pesos=pesos,
        top_n=top_n,
    )

    res.update(
        {
            "df_s6": df_s6,
            "df_mc": df_mc,
            "df_micro": df_micro,
            "df_fusao": df_fusao,
            "k_estrela": k_star,
            "regime": regime,
            "pesos": pesos,
            "qds_local": qds_local,
        }
    )
    return res
# ============================================================
# REPLAY LIGHT — diagnóstico pontual por índice
# ============================================================

def executar_replay_light(
    df: pd.DataFrame,
    idx_alvo: int,
    top_n: int = 20,
) -> Dict[str, Any]:
    """
    Replay LIGHT:
      - Executa o TURBO++ ULTRA ADAPTATIVO em um único ponto do histórico
      - Compara a Previsão Final com a série real seguinte (se existir)
    """
    res_turbo = executar_turbo_ultra_adaptativo_para_indice(
        df=df,
        idx_alvo=idx_alvo,
        top_n=top_n,
        n_s6=200,
        n_mc=800,
        n_micro=20,
    )

    passenger_cols = [c for c in df.columns if c.startswith("p")]
    real_next_series = None
    hit = False

    if idx_alvo + 1 < len(df):
        real_next_series = df.iloc[idx_alvo + 1][passenger_cols].values.astype(int).tolist()
        real_tuple = series_to_tuple(real_next_series)
        if not res_turbo["df_fusao"].empty:
            top_series = [series_to_tuple(s) for s in res_turbo["df_fusao"]["series"].tolist()[:top_n]]
            hit = real_tuple in top_series

    return {
        "turbo": res_turbo,
        "real_next": real_next_series,
        "hit": hit,
    }


# ============================================================
# REPLAY ULTRA / BACKTEST REAL — janela de índices
# ============================================================

def executar_replay_ultra_backtest(
    df: pd.DataFrame,
    idx_inicio: int,
    idx_fim: int,
    top_n: int = 20,
) -> Dict[str, Any]:
    """
    Replay ULTRA / Backtest REAL:
      - para cada índice i em [idx_inicio, idx_fim], executa o TURBO++ ULTRA adaptativo
        usando apenas o histórico até i
      - compara com a série real seguinte, medindo taxa de acerto, regimes, etc.
    """
    passenger_cols = [c for c in df.columns if c.startswith("p")]
    if not passenger_cols:
        return {
            "tabela": pd.DataFrame(),
            "hits": 0,
            "total": 0,
            "taxa_acerto": 0.0,
        }

    idx_inicio = max(0, idx_inicio)
    idx_fim = min(len(df) - 2, idx_fim)  # precisa existir "próximo"
    if idx_fim <= idx_inicio:
        return {
            "tabela": pd.DataFrame(),
            "hits": 0,
            "total": 0,
            "taxa_acerto": 0.0,
        }

    registros = []
    hits = 0
    total = 0

    for i in range(idx_inicio, idx_fim + 1):
        df_hist = df.iloc[: i + 1].copy()
        if len(df_hist) < 6:
            continue

        turbo_res = executar_turbo_ultra_adaptativo_para_indice(
            df=df_hist,
            idx_alvo=len(df_hist) - 1,
            top_n=top_n,
            n_s6=150,
            n_mc=500,
            n_micro=15,
        )

        df_fusao = turbo_res["df_fusao"]
        if df_fusao is None or df_fusao.empty:
            continue

        top_series = [series_to_tuple(s) for s in df_fusao["series"].tolist()[:top_n]]

        real_series = df.iloc[i + 1][passenger_cols].values.astype(int).tolist()
        real_tuple = series_to_tuple(real_series)

        acerto = real_tuple in top_series
        total += 1
        if acerto:
            hits += 1

        melhor = df_fusao.iloc[0]["series"]
        registros.append(
            {
                "id_atual": df.iloc[i]["id"],
                "id_real_prox": df.iloc[i + 1]["id"],
                "serie_real": series_to_str(real_series),
                "melhor_prev": series_to_str(melhor),
                "acerto_topN": acerto,
                "k_estrela_local": turbo_res["k_estrela"],
                "regime_local": turbo_res["regime"],
                "qds_local": turbo_res["qds_local"],
            }
        )

    taxa = float(hits / total) if total > 0 else 0.0
    tabela = pd.DataFrame(registros) if registros else pd.DataFrame()
    return {
        "tabela": tabela,
        "hits": hits,
        "total": total,
        "taxa_acerto": taxa,
    }


# ============================================================
# SUPORTE À INTERFACE — mensagens de contexto por k* / regime
# ============================================================

def mensagem_contexto_kstar(k_star: float, regime: str) -> str:
    base = f"k* = {k_star:.1f}% — "
    if regime == "padrao":
        return base + "Ambiente estável forte, padrão profundo dominante. S6 lidera a fusão."
    elif regime == "transicao":
        return base + "Ambiente de transição / pré-ruptura. Mistura equilibrada de S6, Micro-Leque e Monte Carlo."
    else:
        return base + "Ruptura / macro-caos. Monte Carlo assume protagonismo para capturar previsibilidade curta."


def mensagem_barometro(bar: Dict[str, Any]) -> str:
    estado = bar.get("estado", "indefinido")
    k_medio = bar.get("k_medio", 0.0)
    freq_zero = bar.get("freq_k_zero", 0.0)
    if estado == "estavel":
        return f"🟢 Estrada estável — k médio ≈ {k_medio:.2f}, poucos carros com k=0 ({freq_zero*100:.1f}%)."
    elif estado == "transicao":
        return f"🟡 Estrada em transição — k médio ≈ {k_medio:.2f}, regime misto, atenção à mudança de padrão."
    elif estado == "ruptura":
        return f"🔴 Estrada em ruptura — k médio ≈ {k_medio:.2f}, muitos carros com k=0 ({freq_zero*100:.1f}%)."
    else:
        return "⚪ Barômetro indefinido — histórico insuficiente ou dados inconsistentes."


def mensagem_k_novo_significado() -> str:
    return (
        "📌 **Novo significado de k**:\n\n"
        "- k representa **o número de guardas** que acertaram **exatamente** o carro (todos os passageiros na ordem correta).\n"
        "- k=0: nenhum guarda cravou exatamente aquela série.\n"
        "- k>0: houve guardas que sabiam exatamente quais passageiros estariam naquele carro.\n"
        "- O painel de risco usa a distribuição de k para avaliar raridade, concentração e sensibilidade da estrada."
    )


# ============================================================
# INÍCIO DO APP STREAMLIT
# ============================================================

def configurar_pagina():
    st.set_page_config(
        page_title="Predict Cars V14-FLEX ULTRA REAL (TURBO++ ADAPTATIVO)",
        layout="wide",
    )


def carregar_df_sessao() -> Optional[pd.DataFrame]:
    return st.session_state.get("df", None)


def salvar_df_sessao(df: pd.DataFrame):
    st.session_state["df"] = df


def main_sidebar():
    st.sidebar.markdown("## 🚗 Predict Cars V14-FLEX ULTRA REAL (TURBO++)")
    st.sidebar.markdown("Versão FLEX + REPLAY + TURBO++ ULTRA ADAPTATIVO por k*")

    painel = st.sidebar.radio(
        "Escolha o painel:",
        [
            "📥 Histórico — Entrada",
            "🔍 Pipeline V14-FLEX (TURBO++)",
            "🚨 Monitor de Risco (k & k*)",
            "🚀 Modo TURBO++ ULTRA (Adaptativo)",
            "📅 Modo Replay Automático do Histórico",
            "🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)",
        ],
    )
    return painel


# ============================================================
# PAINEL 1 — Histórico — Entrada (FLEX)
# ============================================================

def painel_historico_entrada():
    st.markdown("## 📥 Histórico — Entrada (FLEX)")
    st.markdown(
        "Entrada FLEX com número variável de passageiros, "
        "detecção automática da coluna k e preparação completa para o pipeline ULTRA."
    )

    df = carregar_df_sessao()

    opc = st.radio(
        "Como deseja carregar o histórico?",
        ["Enviar arquivo CSV", "Copiar e colar o histórico"],
    )

    if opc == "Enviar arquivo CSV":
        file = st.file_uploader("Selecione o arquivo CSV:", type=["csv"])
        if file is not None:
            try:
                df_raw = pd.read_csv(file)
                df = preparar_historico_flex_from_csv(df_raw)
                salvar_df_sessao(df)
                st.success("Histórico carregado e preparado com sucesso!")
                st.write("Prévia do histórico:")
                st.dataframe(df.head(50))
            except Exception as e:
                st.error(f"Erro ao carregar CSV: {e}")

    else:
        texto = st.text_area(
            "Cole o histórico aqui (linhas no formato C1;41;5;4;52;30;33;0, por exemplo):",
            height=240,
        )
        if st.button("Processar histórico colado"):
            try:
                df = preparar_historico_flex_from_text(texto)
                if df is None or df.empty:
                    st.warning("Não foi possível interpretar o histórico. Verifique o formato.")
                else:
                    salvar_df_sessao(df)
                    st.success("Histórico colado e preparado com sucesso!")
                    st.write("Prévia do histórico:")
                    st.dataframe(df.head(50))
            except Exception as e:
                st.error(f"Erro ao processar o texto: {e}")

    if df is not None and not df.empty:
        st.markdown("### 📊 Resumo rápido do histórico")
        st.write(f"Total de séries: **{len(df)}**")
        pcols = [c for c in df.columns if c.startswith("p")]
        st.write(f"Número de passageiros por série (detectado): **{len(pcols)}**")
        st.write("Colunas:", ", ".join(df.columns))


# ============================================================
# PAINEL 2 — Pipeline V14-FLEX (TURBO++) — visão estrutural
# ============================================================

def painel_pipeline_v14_flex():
    st.markdown("## 🔍 Pipeline V14-FLEX (TURBO++) — Execução Estrutural")

    df = carregar_df_sessao()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        return

    pcols = [c for c in df.columns if c.startswith("p")]
    if not pcols:
        st.error("Não foram detectadas colunas de passageiros (p1..pN).")
        return

    idx_alvo = st.number_input(
        "Selecione o índice alvo (1 = primeira série carregada):",
        min_value=1,
        max_value=len(df),
        value=len(df),
        step=1,
    )
    idx_alvo_zero = idx_alvo - 1

    bar = calcular_barometro_ultra_real(df)
    k_star = calcular_k_estrela(df)
    idx_info = calcular_idx_ultra(df)
    ipf_ipo = calcular_ipf_ipo(df)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🌡️ Barômetro ULTRA REAL")
        st.write(mensagem_barometro(bar))
        st.markdown("### 🌟 k* ULTRA REAL")
        st.write(f"k* ≈ **{k_star:.1f}%**")
    with col2:
        st.markdown("### 🧭 IDX ULTRA")
        st.write(f"Índice global (média ponderada): **{idx_info['idx_global']:.2f}**")
        st.markdown("### 📐 IPF / IPO (refinados)")
        st.write(f"IPF ≈ **{ipf_ipo['ipf']:.3f}** — IPO ≈ **{ipf_ipo['ipo']:.3f}**")

    st.markdown("### 🧱 Série alvo (estrutura)")
    st.code(
        series_to_str(df.iloc[idx_alvo_zero][pcols].values.tolist()),
        language="text",
    )

    st.markdown(
        "Este painel mostra o **estado estrutural** da estrada (Barômetro, k*, IDX, IPF/IPO), "
        "que são insumos diretos para o **Modo TURBO++ ULTRA ADAPTATIVO**."
    )


# ============================================================
# PAINEL 3 — Monitor de Risco (k & k*)
# ============================================================

def painel_monitor_risco():
    st.markdown("## 🚨 Monitor de Risco (k & k*)")

    df = carregar_df_sessao()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        return

    bar = calcular_barometro_ultra_real(df)
    k_star = calcular_k_estrela(df)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🌡️ Barômetro ULTRA REAL")
        st.write(mensagem_barometro(bar))
        st.markdown("### 🌟 k* — Sentinela dos guardas")
        st.write(f"k* ≈ **{k_star:.1f}%**")
        regime = determinar_regime_por_kstar(k_star, None)
        st.write(descricao_regime(regime))
    with col2:
        st.markdown("### 📊 Distribuição de k (guardas que acertaram exatamente)")
        hist = df["k"].value_counts().sort_index()
        if not hist.empty:
            st.bar_chart(hist)
        else:
            st.write("Sem dados suficientes para plotar a distribuição de k.")

    st.markdown("### 📌 Interpretação do novo k")
    st.markdown(mensagem_k_novo_significado())
# ============================================================
# PAINEL 4 — Modo TURBO++ ULTRA (Adaptativo por k*)
# ============================================================

def painel_modo_turbo_ultra_adaptativo():
    st.markdown("## 🚀 Modo TURBO++ ULTRA (Adaptativo por k*)")

    df = carregar_df_sessao()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        return

    pcols = [c for c in df.columns if c.startswith("p")]
    if not pcols:
        st.error("Não foram detectadas colunas de passageiros (p1..pN).")
        return

    col_cfg, col_info = st.columns([1, 1.1])

    with col_cfg:
        st.markdown("### ⚙️ Configurações do TURBO++ ULTRA")
        idx_alvo = st.number_input(
            "Índice alvo (1 = primeira série):",
            min_value=1,
            max_value=len(df),
            value=len(df),
            step=1,
        )
        idx_zero = idx_alvo - 1

        top_n = st.slider("Top-N final:", min_value=5, max_value=80, value=20, step=5)
        n_s6 = st.slider("Quantidade de séries S6 Profundo ULTRA:", 50, 400, 200, 50)
        n_mc = st.slider("Quantidade de séries Monte Carlo ULTRA:", 300, 1200, 800, 100)
        n_micro = st.slider("Micro-Leque (variações por série base):", 5, 40, 20, 5)

        rodar = st.button("Executar TURBO++ ULTRA ADAPTATIVO", type="primary")

    with col_info:
        st.markdown("### 🧱 Série alvo (carro atual)")
        st.write(f"ID: **{df.iloc[idx_zero]['id']}**")
        st.code(series_to_str(df.iloc[idx_zero][pcols].values.tolist()), language="text")

    if not rodar:
        st.info("Configure os parâmetros e clique em **Executar TURBO++ ULTRA ADAPTATIVO**.")
        return

    # Execução do motor
    with st.spinner("Rodando S6 Profundo ULTRA, Monte Carlo Profundo ULTRA e Micro-Leque ULTRA..."):
        res = executar_turbo_ultra_adaptativo_para_indice(
            df=df,
            idx_alvo=idx_zero,
            top_n=top_n,
            n_s6=n_s6,
            n_mc=n_mc,
            n_micro=n_micro,
        )

    df_s6 = res["df_s6"]
    df_mc = res["df_mc"]
    df_micro = res["df_micro"]
    df_fusao = res["df_fusao"]
    k_star = res["k_estrela"]
    regime = res["regime"]
    pesos = res["pesos"]
    qds_local = res["qds_local"]

    st.markdown("### 🌟 Contexto adaptativo")
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.metric("k* (sentinela)", f"{k_star:.1f} %")
    with col_c2:
        st.metric("QDS local (janela curta)", f"{qds_local:.3f}")
    with col_c3:
        s6_w, mc_w, micro_w = pesos
        st.write("**Pesos por regime:**")
        st.write(f"S6: **{s6_w:.2f}**  •  Monte Carlo: **{mc_w:.2f}**  •  Micro-Leque: **{micro_w:.2f}**")

    st.info(mensagem_contexto_kstar(k_star, regime))

    st.markdown("### 🧠 S6 Profundo ULTRA — núcleo determinístico")
    if df_s6 is not None and not df_s6.empty:
        st.dataframe(
            df_s6.head(min(30, len(df_s6)))[["series", "score_s6"]],
            use_container_width=True,
        )
    else:
        st.write("Nenhuma série gerada pelo S6 Profundo ULTRA (histórico muito curto).")

    st.markdown("### 🎲 Monte Carlo Profundo ULTRA — motor estocástico")
    if df_mc is not None and not df_mc.empty:
        st.dataframe(
            df_mc.head(min(30, len(df_mc)))[["series", "score_mc"]],
            use_container_width=True,
        )
    else:
        st.write("Nenhuma série gerada pelo Monte Carlo Profundo ULTRA (histórico muito curto).")

    st.markdown("### 🌪️ Micro-Leque ULTRA — variações finas")
    if df_micro is not None and not df_micro.empty:
        st.dataframe(
            df_micro.head(min(30, len(df_micro)))[["series", "score_micro"]],
            use_container_width=True,
        )
    else:
        st.write("Nenhuma série gerada pelo Micro-Leque ULTRA (falta de base ou histórico).")

    st.markdown("### 🔚 Fusão ULTRA ADAPTATIVA — Top-N final")
    if df_fusao is None or df_fusao.empty:
        st.error("Fusão não retornou nenhuma série. Verifique se há histórico suficiente.")
        return

    st.dataframe(
        df_fusao.head(top_n)[["rank_fusao", "series", "score_fusao", "rank_s6", "rank_mc", "rank_micro"]],
        use_container_width=True,
    )

    melhor = df_fusao.iloc[0]["series"]
    st.markdown("### 🎯 Previsão Final TURBO++ ULTRA (Adaptativo)")
    st.code(series_to_str(melhor), language="text")

    # Mensagem de regime
    st.success(descricao_regime(regime))


# ============================================================
# PAINEL 5 — Modo Replay Automático do Histórico
# ============================================================

def painel_replay_automatico():
    st.markdown("## 📅 Modo Replay Automático do Histórico")

    df = carregar_df_sessao()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        return

    pcols = [c for c in df.columns if c.startswith("p")]
    if not pcols:
        st.error("Não foram detectadas colunas de passageiros (p1..pN).")
        return

    st.markdown(
        "O Replay Automático executa o **TURBO++ ULTRA ADAPTATIVO** ao longo de um "
        "intervalo de índices e compara com o que realmente ocorreu, simulando um backtest."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        idx_inicio = st.number_input(
            "Índice inicial (1):",
            min_value=1,
            max_value=max(1, len(df) - 1),
            value=max(1, len(df) - 60),
            step=1,
        )
    with col_b:
        idx_fim = st.number_input(
            "Índice final (precisa ter próximo conhecido):",
            min_value=idx_inicio,
            max_value=len(df) - 1,
            value=len(df) - 1,
            step=1,
        )

    top_n = st.slider("Top-N usado para acerto no Replay ULTRA:", 5, 50, 20, 5)

    if st.button("Executar Replay ULTRA / Backtest REAL"):
        with st.spinner("Executando Replay ULTRA / Backtest REAL..."):
            res = executar_replay_ultra_backtest(
                df=df,
                idx_inicio=idx_inicio - 1,
                idx_fim=idx_fim - 1,
                top_n=top_n,
            )

        tabela = res["tabela"]
        hits = res["hits"]
        total = res["total"]
        taxa = res["taxa_acerto"]

        if tabela is None or tabela.empty:
            st.warning("Nenhum resultado produzido. Tente ajustar a janela ou verifique o histórico.")
            return

        st.markdown("### 📋 Resultado detalhado do Replay ULTRA")
        st.dataframe(tabela, use_container_width=True)

        st.markdown("### 📈 Síntese de desempenho")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tentativas (carros re-jogados)", total)
        with col2:
            st.metric("Acertos em Top-N", hits)
        with col3:
            st.metric("Taxa de acerto", f"{taxa*100:.2f} %")

        st.info(
            "Este Replay ULTRA funciona como um **Backtest REAL focal**, "
            "reproduzindo as decisões que o Modo TURBO++ ULTRA ADAPTATIVO tomaria em cada carro."
        )


# ============================================================
# PAINEL 6 — Testes de Confiabilidade (QDS / Backtest / Monte Carlo)
# ============================================================

def painel_testes_confiabilidade():
    st.markdown("## 🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)")

    df = carregar_df_sessao()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        return

    pcols = [c for c in df.columns if c.startswith("p")]
    if not pcols:
        st.error("Não foram detectadas colunas de passageiros (p1..pN).")
        return

    st.markdown(
        "Este painel consolida a visão de **QDS REAL**, "
        "**Backtest REAL** (via Replay ULTRA) e "
        "**Monte Carlo Profundo ULTRA** em janelas configuráveis."
    )

    janela_qds = st.slider("Janela para QDS REAL (nº de séries recentes):", 20, 200, 60, 10)
    top_n_qds = st.slider("Top-N para acerto no cálculo de QDS:", 5, 50, 20, 5)

    if st.button("Calcular QDS REAL (global da janela)"):
        with st.spinner("Calculando QDS REAL a partir de backtest interno..."):
            qds_val = calcular_qds_backtest_simples(
                df,
                passenger_cols=pcols,
                janela=min(janela_qds, len(df) - 2),
                top_n=top_n_qds,
            )
        st.metric("QDS REAL (janela global)", f"{qds_val:.3f}")
        if qds_val < 0.05:
            st.warning(
                "QDS muito baixo — regime de **ruptura prolongada**. "
                "A estrada não oferece padrão profundo confiável em janelas longas."
            )
        elif qds_val < 0.15:
            st.info(
                "QDS baixo, porém não nulo — regime de **transição / instabilidade**. "
                "Há bolsões de previsibilidade, mas o padrão global ainda é frágil."
            )
        else:
            st.success(
                "QDS moderado/alto — a estrada apresenta **padrão aproveitável** "
                "em janelas longas. S6 e micro-estruturas tendem a funcionar melhor."
            )

    st.markdown("### 🎲 Monte Carlo Profundo ULTRA — visão estatística global")
    if st.button("Gerar amostra Monte Carlo Profundo ULTRA para diagnóstico global"):
        with st.spinner("Gerando amostra global de Monte Carlo Profundo ULTRA..."):
            df_mc = gerar_monte_carlo_profundo_ultra(
                df,
                passenger_cols=pcols,
                n_series=1200,
                janela=200,
            )
        st.write("Prévia das séries mais frequentes (Monte Carlo ULTRA):")
        st.dataframe(df_mc.head(40), use_container_width=True)
        st.info(
            "As séries mais frequentes no Monte Carlo Profundo ULTRA indicam "
            "padrões estatísticos de curto prazo que o modelo está capturando "
            "no regime atual da estrada."
        )


# ============================================================
# MAIN
# ============================================================

def main():
    configurar_pagina()
    painel = main_sidebar()

    if painel == "📥 Histórico — Entrada":
        painel_historico_entrada()
    elif painel == "🔍 Pipeline V14-FLEX (TURBO++)":
        painel_pipeline_v14_flex()
    elif painel == "🚨 Monitor de Risco (k & k*)":
        painel_monitor_risco()
    elif painel == "🚀 Modo TURBO++ ULTRA (Adaptativo)":
        painel_modo_turbo_ultra_adaptativo()
    elif painel == "📅 Modo Replay Automático do Histórico":
        painel_replay_automatico()
    elif painel == "🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)":
        painel_testes_confiabilidade()
    else:
        st.write("Painel não reconhecido.")


if __name__ == "__main__":
    main()
