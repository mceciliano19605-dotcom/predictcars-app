# ============================================================
# Predict Cars V15.5-HÍBRIDO
# Núcleo V14-FLEX ULTRA + k* + Ruído Condicional + QDS REAL +
# Backtest REAL (protegido) + Monte Carlo REAL + AIQ Bridge
# ============================================================

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# CONFIGURAÇÃO DO APP
# ------------------------------------------------------------

st.set_page_config(
    page_title="Predict Cars V15.5-HÍBRIDO",
    page_icon="🚗",
    layout="wide",
)


# ------------------------------------------------------------
# LIMITES / CONSTANTES IMPORTANTES
# ------------------------------------------------------------

LIMITE_REPLAY_HIST = 3000  # limite para não rodar replay completo em estradas enormes

QDS_LABELS = ["PREMIUM", "BOM", "REGULAR", "RUIM"]

QDS_THRESHOLDS = {
    "PREMIUM": 0.85,
    "BOM": 0.70,
    "REGULAR": 0.50,
    "RUIM": 0.0,
}

REGIMES = ["Ultra Estável", "Estável", "Transição", "Turbulento", "Crítico"]


# ------------------------------------------------------------
# DATACLASSES
# ------------------------------------------------------------

@dataclass
class ResumoEstrada:
    n_series: int
    n_passageiros: int
    min_val: int
    max_val: int
    media: float
    desvio: float
    regime_global: str
    k_medio: float
    k_max: int


@dataclass
class ResumoQDS:
    qds_medio: float
    qds_min: float
    qds_max: float
    pct_premium: float
    pct_bom: float
    pct_regular: float
    pct_ruim: float


@dataclass
class ResumoRuido:
    ruido_inicial: float
    ruido_final: float
    pct_pontos_ajustados: float


@dataclass
class ResumoBacktest:
    n_janelas: int
    acertos_totais: int
    acertos_por_serie: float
    hit_rate: float


@dataclass
class ResumoMonteCarlo:
    n_simulacoes: int
    media_acertos: float
    desvio_acertos: float
    melhor_serie_media: float


@dataclass
class ResumoK:
    k_atual: int
    k_star: float
    estado_k: str
    regime_local: str


# ------------------------------------------------------------
# PARSERS FLEX ULTRA
# ------------------------------------------------------------

def _detect_separator(sample_line: str) -> str:
    for sep in [";", ",", "\t", " "]:
        if sep in sample_line:
            return sep
    return ";"


def parse_historico_text(raw: str) -> pd.DataFrame:
    linhas = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not linhas:
        return pd.DataFrame()

    sep = _detect_separator(linhas[0])
    registros = []

    for ln in linhas:
        partes = [p.strip() for p in ln.split(sep) if p.strip()]
        if not partes:
            continue

        if not partes[0].isdigit():
            serie_id = partes[0]
            nums = partes[1:]
        else:
            serie_id = None
            nums = partes

        try:
            nums_int = [int(x) for x in nums]
        except ValueError:
            continue

        if len(nums_int) < 2:
            continue

        *passageiros, k_val = nums_int
        registros.append(
            {"serie_id": serie_id, "passageiros": passageiros, "k": k_val}
        )

    if not registros:
        return pd.DataFrame()

    max_p = max(len(r["passageiros"]) for r in registros)
    linhas_df = []

    for idx, r in enumerate(registros, start=1):
        base = {}
        base["serie"] = r["serie_id"] or f"C{idx}"
        for j in range(max_p):
            col = f"n{j+1}"
            if j < len(r["passageiros"]):
                base[col] = r["passageiros"][j]
            else:
                base[col] = np.nan
        base["k"] = r["k"]
        linhas_df.append(base)

    df = pd.DataFrame(linhas_df)
    num_cols = [c for c in df.columns if c.startswith("n")]
    df[num_cols] = df[num_cols].fillna(-1).astype(int)
    df["k"] = df["k"].astype(int)
    return df


def parse_historico_csv(file) -> pd.DataFrame:
    try:
        df_raw = pd.read_csv(file, sep=None, engine="python")
    except Exception:
        file.seek(0)
        df_raw = pd.read_csv(file, sep=";")

    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    # caso 1 — 1 coluna só, provavelmente tudo junto
    if df_raw.shape[1] == 1:
        colname = df_raw.columns[0]
        raw = "\n".join(str(x) for x in df_raw[colname].astype(str))
        return parse_historico_text(raw)

    cols_lower = [c.lower() for c in df_raw.columns]
    has_serie = any(c.startswith("serie") for c in cols_lower)
    has_k = any(c == "k" for c in cols_lower)

    df = df_raw.copy()

    if has_serie and has_k:
        map_cols = {}
        for c in df.columns:
            cl = c.lower()
            if cl.startswith("serie"):
                map_cols[c] = "serie"
            elif cl == "k":
                map_cols[c] = "k"
        df = df.rename(columns=map_cols)

        num_cols = [c for c in df.columns if c not in ["serie", "k"]]
        num_cols_sorted = sorted(num_cols)
        rename_map = {}
        for i, col in enumerate(num_cols_sorted, start=1):
            rename_map[col] = f"n{i}"
        df = df.rename(columns=rename_map)
    else:
        cols = list(df.columns)
        first_col = cols[0]
        last_col = cols[-1]

        df["serie"] = df[first_col].astype(str)
        df["k"] = df[last_col]

        mid_cols = cols[1:-1]
        rename_map = {}
        for i, col in enumerate(mid_cols, start=1):
            rename_map[col] = f"n{i}"
        df = df.rename(columns=rename_map)

        keep_cols = ["serie"] + [c for c in df.columns if c.startswith("n")] + ["k"]
        df = df[keep_cols]

    num_cols = [c for c in df.columns if c.startswith("n")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(int)
    df["k"] = pd.to_numeric(df["k"], errors="coerce").fillna(0).astype(int)

    if "serie" not in df.columns:
        df.insert(0, "serie", [f"C{i+1}" for i in range(len(df))])

    return df


# ------------------------------------------------------------
# RESUMOS / REGIME / k*
# ------------------------------------------------------------

def classificar_regime_por_k(k_medio: float) -> str:
    if k_medio <= 0.10:
        return "Ultra Estável"
    if k_medio <= 0.25:
        return "Estável"
    if k_medio <= 0.45:
        return "Transição"
    if k_medio <= 0.70:
        return "Turbulento"
    return "Crítico"


def calcular_resumo_estrada(df: pd.DataFrame) -> Optional[ResumoEstrada]:
    if df is None or df.empty:
        return None

    num_cols = [c for c in df.columns if c.startswith("n")]
    if not num_cols:
        return None

    valores = df[num_cols].values.flatten()
    valores = valores[~np.isnan(valores)]
    valores = [int(v) for v in valores if v >= 0]

    if not valores:
        return None

    n_series = len(df)
    n_passageiros = len(num_cols)
    min_val = int(min(valores))
    max_val = int(max(valores))
    media = float(np.mean(valores))
    desvio = float(np.std(valores))

    k_vals = df["k"].astype(int).tolist()
    k_medio = float(np.mean(k_vals)) if k_vals else 0.0
    k_max = int(max(k_vals)) if k_vals else 0

    regime_global = classificar_regime_por_k(k_medio)

    return ResumoEstrada(
        n_series=n_series,
        n_passageiros=n_passageiros,
        min_val=min_val,
        max_val=max_val,
        media=media,
        desvio=desvio,
        regime_global=regime_global,
        k_medio=k_medio,
        k_max=k_max,
    )


def classificar_qds_valor(v: float) -> str:
    if v >= QDS_THRESHOLDS["PREMIUM"]:
        return "PREMIUM"
    if v >= QDS_THRESHOLDS["BOM"]:
        return "BOM"
    if v >= QDS_THRESHOLDS["REGULAR"]:
        return "REGULAR"
    return "RUIM"


def calcular_resumo_qds(series_qds: List[float]) -> Optional[ResumoQDS]:
    if not series_qds:
        return None

    arr = np.array(series_qds, dtype=float)
    qds_medio = float(arr.mean())
    qds_min = float(arr.min())
    qds_max = float(arr.max())

    total = len(arr)
    cat_counts = {cat: 0 for cat in QDS_LABELS}
    for v in arr:
        cat = classificar_qds_valor(float(v))
        cat_counts[cat] += 1

    def pct(x: int) -> float:
        return (x / total) * 100 if total > 0 else 0.0

    return ResumoQDS(
        qds_medio=qds_medio,
        qds_min=qds_min,
        qds_max=qds_max,
        pct_premium=pct(cat_counts["PREMIUM"]),
        pct_bom=pct(cat_counts["BOM"]),
        pct_regular=pct(cat_counts["REGULAR"]),
        pct_ruim=pct(cat_counts["RUIM"]),
    )


def calcular_resumo_ruido(
    ruido_inicial: float,
    ruido_final: float,
    pct_pontos_ajustados: float,
) -> ResumoRuido:
    return ResumoRuido(
        ruido_inicial=float(ruido_inicial),
        ruido_final=float(ruido_final),
        pct_pontos_ajustados=float(pct_pontos_ajustados),
    )


def calcular_resumo_backtest(
    acertos_lista: List[int],
    n_series_por_janela: int,
) -> Optional[ResumoBacktest]:
    if not acertos_lista:
        return None

    n_janelas = len(acertos_lista)
    acertos_totais = sum(acertos_lista)
    acertos_por_serie = acertos_totais / (n_janelas * max(n_series_por_janela, 1))
    hit_rate = acertos_totais / (n_janelas * max(n_series_por_janela, 1))

    return ResumoBacktest(
        n_janelas=n_janelas,
        acertos_totais=acertos_totais,
        acertos_por_serie=acertos_por_serie,
        hit_rate=hit_rate,
    )


def calcular_resumo_monte_carlo(
    matriz_acertos: List[List[int]],
) -> Optional[ResumoMonteCarlo]:
    if not matriz_acertos:
        return None

    medias = [statistics.mean(sim) for sim in matriz_acertos if sim]
    if not medias:
        return None

    media_acertos = float(statistics.mean(medias))
    desvio_acertos = float(statistics.pstdev(medias)) if len(medias) > 1 else 0.0
    melhor_serie_media = float(max(medias))

    return ResumoMonteCarlo(
        n_simulacoes=len(matriz_acertos),
        media_acertos=media_acertos,
        desvio_acertos=desvio_acertos,
        melhor_serie_media=melhor_serie_media,
    )


def calcular_k_star(df: pd.DataFrame, janela: int = 40) -> ResumoK:
    if df is None or df.empty:
        return ResumoK(0, 0.0, "desconhecido", "desconhecido")

    df_ord = df.reset_index(drop=True)
    janela = min(janela, len(df_ord))
    bloco = df_ord.iloc[-janela:].copy()
    k_vals = bloco["k"].astype(int).tolist()
    k_atual = k_vals[-1] if k_vals else 0

    if not k_vals:
        k_star = 0.0
    else:
        positivos = sum(1 for x in k_vals if x > 0)
        k_star = positivos / len(k_vals)

    if k_star <= 0.10:
        estado_k = "estavel"
    elif k_star <= 0.30:
        estado_k = "atencao"
    else:
        estado_k = "critico"

    regime_local = classificar_regime_por_k(k_star)

    return ResumoK(
        k_atual=int(k_atual),
        k_star=float(k_star),
        estado_k=estado_k,
        regime_local=regime_local,
    )


# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------

def init_session_state() -> None:
    defaults = {
        "df": None,
        "df_limpo": None,
        "df_ruido_a": None,
        "df_ruido_b": None,
        "resumo_estrada": None,
        "resumo_k_global": None,
        "lista_qds": [],
        "resumo_qds": None,
        "resumo_ruido": None,
        "resumo_backtest": None,
        "resumo_montecarlo": None,
        "previsao_base_v14": None,
        "previsao_turbo_ultra": None,
        "historico_backtest": None,
        "historico_montecarlo": None,
        "mostrar_debug": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


# ------------------------------------------------------------
# LAYOUT PRINCIPAL / SIDEBAR
# ------------------------------------------------------------

st.title("🚗 Predict Cars V15.5-HÍBRIDO")
st.caption(
    "Núcleo V14-FLEX ULTRA + k* + Ruído Condicional + QDS REAL + "
    "Backtest REAL (protegido) + Monte Carlo REAL + AIQ Bridge (para ChatGPT)."
)

with st.sidebar:
    st.markdown("## 📂 Navegação")

    painel = st.radio(
        "Escolha o painel:",
        options=[
            "📥 Histórico — Entrada FLEX ULTRA (V15-HÍBRIDO)",
            "🔍 Pipeline V14-FLEX ULTRA (V15)",
            "💡 Replay LIGHT",
            "📅 Replay ULTRA",
            "🎯 Replay ULTRA Unitário",
            "🚨 Monitor de Risco (k & k*)",
            "🧪 Testes de Confiabilidade REAL",
            "📊 Ruído Condicional (V15)",
            "🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15)",
            "📄 Relatório Final — AIQ Bridge (para ChatGPT)",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ Opções Globais")
    st.session_state["mostrar_debug"] = st.checkbox(
        "Mostrar debug / tabelas internas", value=False
    )


# ------------------------------------------------------------
# PAINEL 1 — ENTRADA FLEX ULTRA
# ------------------------------------------------------------

if painel == "📥 Histórico — Entrada FLEX ULTRA (V15-HÍBRIDO)":
    st.markdown("## 📥 Histórico — Entrada FLEX ULTRA (V15-HÍBRIDO)")
    st.markdown(
        """
        Ponto de partida do V15.5-HÍBRIDO:

        - Aceita histórico completo da estrada (até 5k+ séries)
        - Detecta automaticamente número de passageiros (n1..nN)
        - Separa `serie` e `k`
        - Calcula:
          - Resumo global da estrada
          - k atual e k* global
          - Regime/barômetro
        """
    )

    with st.expander("📌 Formatos aceitos (FLEX ULTRA)", expanded=False):
        st.markdown(
            """
            **Exemplo com ID:**

            ```text
            C1;41;5;4;52;30;33;0
            C2;9;39;37;49;43;41;1
            C3;36;30;10;11;29;47;2
            ```

            **Exemplo sem ID:**

            ```text
            41;5;4;52;30;33;0
            9;39;37;49;43;41;1
            36;30;10;11;29;47;2
            ```

            **CSV estruturado:**

            ```csv
            serie,n1,n2,n3,n4,n5,n6,k
            C1,41,5,4,52,30,33,0
            C2,9,39,37,49,43,41,1
            C3,36,30,10,11,29,47,2
            ```
            """
        )

    modo_entrada = st.radio(
        "Modo de entrada:",
        ["Upload CSV", "Colar texto"],
        horizontal=True,
    )

    df_result = None

    if modo_entrada == "Upload CSV":
        file = st.file_uploader(
            "Selecione o arquivo de histórico (.csv ou .txt):",
            type=["csv", "txt"],
        )
        if file is not None:
            df_result = parse_historico_csv(file)
    else:
        raw_text = st.text_area(
            "Cole o histórico completo:",
            height=260,
            placeholder="C1;41;5;4;52;30;33;0\nC2;9;39;37;49;43;41;1\n...",
        )
        if raw_text.strip():
            df_result = parse_historico_text(raw_text)

    if df_result is not None and not df_result.empty:
        try:
            df_result = df_result.copy()
            df_result["__idx"] = (
                df_result["serie"].astype(str).str.extract(r"(\d+)").astype(float)
            )
            df_result = df_result.sort_values("__idx").drop(columns=["__idx"])
        except Exception:
            pass

        st.session_state["df"] = df_result
        st.session_state["df_limpo"] = df_result.copy()

        resumo_estrada = calcular_resumo_estrada(df_result)
        resumo_k_global = calcular_k_star(df_result, janela=min(60, len(df_result)))

        st.session_state["resumo_estrada"] = resumo_estrada
        st.session_state["resumo_k_global"] = resumo_k_global

        st.success(
            f"Histórico carregado com sucesso. Total de séries: {len(df_result)}."
        )

        col1, col2, col3 = st.columns(3)

        if resumo_estrada is not None:
            with col1:
                st.markdown("### 🛣️ Estrada Global")
                st.metric("Séries", resumo_estrada.n_series)
                st.metric("Passageiros por série", resumo_estrada.n_passageiros)
                st.metric(
                    "Faixa de valores",
                    f"{resumo_estrada.min_val} — {resumo_estrada.max_val}",
                )

            with col2:
                st.markdown("### 🌡️ Regime / Barômetro")
                st.metric("Regime global", resumo_estrada.regime_global)
                st.metric("k médio", f"{resumo_estrada.k_medio:.2f}")
                st.metric("k máximo", resumo_estrada.k_max)

        if resumo_k_global is not None:
            with col3:
                st.markdown("### 🔭 k* Global (sentinela)")
                st.metric("k atual (última série)", resumo_k_global.k_atual)
                st.metric("k*", f"{resumo_k_global.k_star*100:.1f}%")
                estado_label = {
                    "estavel": "🟢 Ambiente estável",
                    "atencao": "🟡 Pré-ruptura residual",
                    "critico": "🔴 Ambiente crítico",
                }.get(resumo_k_global.estado_k, "⚪ Desconhecido")
                st.write(estado_label)
                st.caption(f"Regime local: **{resumo_k_global.regime_local}**")

        st.markdown("### 🔎 Amostra do histórico normalizado")
        st.dataframe(df_result.head(50), use_container_width=True)

        if st.session_state["mostrar_debug"]:
            st.markdown("#### 🐞 DEBUG — describe()")
            st.write(df_result.describe(include="all"))
    else:
        st.info(
            "Carregue um histórico por arquivo ou texto para iniciar o V15.5-HÍBRIDO."
        )
# ============================================================
# PARTE 2/6 — PIPELINE V14-FLEX ULTRA (V15)
# ============================================================

# ------------------------------------------------------------
# FUNÇÕES DO PIPELINE V14-FLEX ULTRA (V15)
# ------------------------------------------------------------

def get_passenger_cols(df: pd.DataFrame) -> List[str]:
    """
    Retorna as colunas de passageiros (n1..nN) em ordem.
    """
    return sorted(
        [c for c in df.columns if c.startswith("n")],
        key=lambda x: int(x[1:]),
    )


def extrair_janela_hist(
    df: pd.DataFrame,
    idx_alvo: int,
    back: int,
    forward: int = 0,
) -> pd.DataFrame:
    """
    Extrai uma janela do histórico em torno do índice alvo (1-based na interface).

    - back: quantas séries olhar para trás
    - forward: quantas olhar para frente (normalmente 0 para predição)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    n = len(df)
    pos = max(0, min(idx_alvo - 1, n - 1))  # índice interno 0-based

    ini = max(0, pos - back)
    fim = min(n, pos + 1 + forward)
    return df.iloc[ini:fim].copy()


def calcular_matriz_frequencia(
    janela: pd.DataFrame,
    suavizacao: float = 1.0,
) -> Dict[str, Dict[int, float]]:
    """
    Calcula uma matriz de frequências para cada posição (n1..nN) da série.

    Retorno:
        {
            "n1": {valor: probabilidade, ...},
            "n2": {...},
            ...
        }
    """
    if janela is None or janela.empty:
        return {}

    matriz: Dict[str, Dict[int, float]] = {}
    cols = get_passenger_cols(janela)

    for col in cols:
        valores = janela[col].astype(int).tolist()
        contagens: Dict[int, int] = {}

        for v in valores:
            contagens[v] = contagens.get(v, 0) + 1

        total = sum(contagens.values()) + suavizacao * max(len(contagens), 1)

        probs_col: Dict[int, float] = {}
        for v, c in contagens.items():
            probs_col[v] = (c + suavizacao) / total

        matriz[col] = probs_col

    return matriz


def gerar_candidato_serie(
    matriz_freq: Dict[str, Dict[int, float]],
    rng: random.Random,
) -> List[int]:
    """
    Gera uma série candidata baseada na matriz de frequências.

    Usa:
    - random.Random apenas para fallback / ranges
    - np.random.choice para escolher com probabilidades p
    """
    if not matriz_freq:
        # fallback simples se não houver matriz
        return [rng.randint(0, 60) for _ in range(6)]

    serie: List[int] = []

    for col in sorted(matriz_freq.keys(), key=lambda x: int(x[1:])):
        dist = matriz_freq[col]

        if not dist:
            serie.append(rng.randint(0, 60))
            continue

        valores = list(dist.keys())
        probs = np.array([dist[v] for v in valores], dtype=float)

        if probs.sum() <= 0:
            serie.append(rng.randint(0, 60))
            continue

        probs = probs / probs.sum()
        escolha = int(np.random.choice(valores, p=probs))
        serie.append(escolha)

    return serie


def calcular_diversidade_serie(serie: List[int]) -> float:
    """Diversidade simples = proporção de valores distintos."""
    if not serie:
        return 0.0
    return len(set(serie)) / len(serie)


def calcular_qds_candidato(
    serie: List[int],
    matriz_freq: Dict[str, Dict[int, float]],
) -> float:
    """QDS = média das probabilidades posição a posição."""
    if not serie or not matriz_freq:
        return 0.0

    probs_pos: List[float] = []
    cols = sorted(matriz_freq.keys(), key=lambda x: int(x[1:]))

    for idx, col in enumerate(cols):
        dist = matriz_freq[col]
        if idx >= len(serie):
            continue
        v = serie[idx]
        probs_pos.append(float(dist.get(v, 0.0)))

    if not probs_pos:
        return 0.0

    return float(max(0.0, min(1.0, np.mean(probs_pos))))


def calcular_aiq_candidato(
    serie: List[int],
    matriz_freq: Dict[str, Dict[int, float]],
    peso_qds: float = 0.6,
    peso_div: float = 0.4,
) -> Tuple[float, float, float]:
    """
    AIQ = combinação ponderada de QDS e Diversidade.

    Retorna:
        (AIQ, QDS, DIVERSIDADE)
    """
    qds = calcular_qds_candidato(serie, matriz_freq)
    div = calcular_diversidade_serie(serie)

    qds = max(0.0, min(1.0, qds))
    div = max(0.0, min(1.0, div))

    aiq = max(0.0, min(1.0, peso_qds * qds + peso_div * div))
    return float(aiq), float(qds), float(div)


def gerar_leque_candidatos(
    matriz_freq: Dict[str, Dict[int, float]],
    n_series: int,
    seed: int,
) -> List[List[int]]:
    """
    Gera o leque V14-base usando matriz de frequências.
    Usa RNG determinístico por seed.
    """
    rng = random.Random(seed)
    candidatos: List[List[int]] = []
    vistos = set()

    max_tentativas = max(n_series * 10, n_series + 10)

    while len(candidatos) < n_series and max_tentativas > 0:
        s = gerar_candidato_serie(matriz_freq, rng)
        t = tuple(s)
        if t not in vistos:
            vistos.add(t)
            candidatos.append(s)
        max_tentativas -= 1

    return candidatos


def montar_tabela_candidatos(
    candidatos: List[List[int]],
    matriz_freq: Dict[str, Dict[int, float]],
    regime_global: str,
    resumo_k: Optional[ResumoK],
) -> pd.DataFrame:
    """
    Monta o DataFrame com:

    - id
    - series
    - QDS
    - Diversidade
    - AIQ
    - Regime_global
    - Regime_local_k
    """
    registros = []

    for i, serie in enumerate(candidatos, start=1):
        aiq, qds, div = calcular_aiq_candidato(serie, matriz_freq)
        reg_local = resumo_k.regime_local if resumo_k else "desconhecido"

        registros.append(
            {
                "id": i,
                "series": serie,
                "QDS": qds,
                "Diversidade": div,
                "AIQ": aiq,
                "Regime_global": regime_global,
                "Regime_local_k": reg_local,
            }
        )

    if not registros:
        return pd.DataFrame()

    df = pd.DataFrame(registros)
    return df.sort_values(["AIQ", "QDS"], ascending=[False, False]).reset_index(
        drop=True
    )


# ------------------------------------------------------------
# PAINEL 2 — Pipeline V14-FLEX ULTRA (V15)
# ------------------------------------------------------------

if painel == "🔍 Pipeline V14-FLEX ULTRA (V15)":
    st.markdown("## 🔍 Pipeline V14-FLEX ULTRA (V15)")
    st.markdown(
        """
        Este painel executa o **núcleo V14-FLEX ULTRA**:

        1. Seleciona o índice alvo  
        2. Extrai a janela local da estrada  
        3. Calcula a matriz de frequências (n1..nN)  
        4. Gera o leque base de candidatos  
        5. Avalia QDS, Diversidade e AIQ  
        6. Define a **previsão base V14** (pré-ruído, pré-TURBO)
        """
    )

    df_limpo = st.session_state.get("df_limpo", None)
    if df_limpo is None or df_limpo.empty:
        st.warning(
            "Carregue o histórico no painel "
            "'📥 Histórico — Entrada FLEX ULTRA (V15-HÍBRIDO)'."
        )
        st.stop()

    resumo_estrada: Optional[ResumoEstrada] = st.session_state.get("resumo_estrada")
    resumo_k_global: Optional[ResumoK] = st.session_state.get("resumo_k_global")

    if "previsao_base_v14" not in st.session_state:
        st.session_state["previsao_base_v14"] = None

    n_series_hist = len(df_limpo)
    cols_pass = get_passenger_cols(df_limpo)

    # --------------------------------------------------------
    # CONTROLES DO PIPELINE
    # --------------------------------------------------------

    st.markdown("### ⚙️ Controles do Pipeline V14-FLEX ULTRA")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        idx_alvo = st.number_input(
            "Índice alvo (1 = primeira série):",
            min_value=1,
            max_value=n_series_hist,
            value=n_series_hist,
            step=1,
        )
    with col_b:
        janela_back = st.slider(
            "Tamanho da janela para trás:",
            min_value=10,
            max_value=min(300, n_series_hist - 1),
            value=min(60, max(10, n_series_hist - 1)),
            step=5,
        )
    with col_c:
        n_candidatos = st.slider(
            "Tamanho do leque base (V14):",
            min_value=10,
            max_value=200,
            value=60,
            step=5,
        )

    col_d, col_e = st.columns(2)
    with col_d:
        seed_base = st.number_input(
            "Seed V14 (reprodutível):",
            min_value=1,
            max_value=999999,
            value=12345,
        )
    with col_e:
        peso_qds = st.slider(
            "Peso do QDS (AIQ):",
            min_value=0.1,
            max_value=0.9,
            value=0.6,
            step=0.05,
        )

    # --------------------------------------------------------
    # CONTEXTO LOCAL DO ALVO
    # --------------------------------------------------------

    st.markdown("### 🛰️ Contexto local")

    df_janela = extrair_janela_hist(df_limpo, int(idx_alvo), back=int(janela_back))

    if df_janela.empty:
        st.error("Janela vazia. Ajuste os parâmetros.")
        st.stop()

    serie_alvo = df_limpo.iloc[int(idx_alvo) - 1]

    col_s1, col_s2, col_s3 = st.columns([2, 2, 2])

    with col_s1:
        st.markdown("#### 🚗 Série alvo")
        valores = [int(serie_alvo[c]) for c in cols_pass]
        k_val = int(serie_alvo["k"])
        st.code(" ".join(str(x) for x in valores) + f" | k = {k_val}", language="text")

    with col_s2:
        st.markdown("#### 🧭 Janela local")
        st.write(
            f"Séries consideradas: **{len(df_janela)}** "
            f"({df_janela['serie'].iloc[0]} → {df_janela['serie'].iloc[-1]})"
        )
        if resumo_estrada:
            st.metric("Regime global", resumo_estrada.regime_global)
        if resumo_estrada:
            st.metric("k médio global", f"{resumo_estrada.k_medio:.2f}")

    with col_s3:
        st.markdown("#### 🔭 k* local")
        resumo_k_local = calcular_k_star(df_janela, janela=len(df_janela))
        st.metric("k atual", resumo_k_local.k_atual)
        st.metric("k*", f"{resumo_k_local.k_star*100:.1f}%")
        estado_local = {
            "estavel": "🟢 Estável",
            "atencao": "🟡 Pré-ruptura",
            "critico": "🔴 Crítico",
        }.get(resumo_k_local.estado_k, "⚪ Desconhecido")
        st.write(estado_local)
        st.caption(f"Regime local: **{resumo_k_local.regime_local}**")

    if st.session_state.get("mostrar_debug", False):
        st.markdown("#### 🐞 DEBUG — Janela local")
        st.dataframe(df_janela.head(50), use_container_width=True)

    # --------------------------------------------------------
    # MATRIZ DE FREQUÊNCIAS
    # --------------------------------------------------------

    st.markdown("### 📊 Matriz de frequências (V14-FLEX ULTRA)")

    matriz_freq = calcular_matriz_frequencia(df_janela)

    if not matriz_freq:
        st.error("Falha ao gerar matriz de frequências.")
        st.stop()

    preview_cols = st.columns(len(cols_pass))
    for i, col in enumerate(cols_pass):
        with preview_cols[i]:
            st.markdown(f"**{col}**")
            dist = matriz_freq.get(col, {})
            if not dist:
                st.caption("Sem dados.")
            else:
                top_vals = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:5]
                st.caption("\n".join(f"{v}: {p*100:.1f}%" for v, p in top_vals))

    # --------------------------------------------------------
    # LEQUE BASE V14-FLEX ULTRA
    # --------------------------------------------------------

    st.markdown("### 🎯 Leque base V14-FLEX ULTRA")

    candidatos = gerar_leque_candidatos(
        matriz_freq,
        n_series=int(n_candidatos),
        seed=int(seed_base + int(idx_alvo) * 13),
    )

    regime_global_str = resumo_estrada.regime_global if resumo_estrada else "desconhecido"

    df_candidatos = montar_tabela_candidatos(
        candidatos,
        matriz_freq,
        regime_global_str,
        resumo_k_local,
    )

    # Reajuste de AIQ com peso customizado
    if not df_candidatos.empty:
        novas_aiq = []
        for _, row in df_candidatos.iterrows():
            serie = row["series"]
            _, qds_tmp, div_tmp = calcular_aiq_candidato(
                serie,
                matriz_freq,
                peso_qds=peso_qds,
                peso_div=1.0 - peso_qds,
            )
            novas_aiq.append(peso_qds * qds_tmp + (1 - peso_qds) * div_tmp)

        df_candidatos["AIQ"] = novas_aiq
        df_candidatos = df_candidatos.sort_values(
            ["AIQ", "QDS"], ascending=[False, False]
        ).reset_index(drop=True)

    if df_candidatos.empty:
        st.error("Nenhum candidato válido. Ajuste parâmetros.")
        st.stop()

    # Armazena QDS global (para estatísticas futuras)
    lista_qds_global = st.session_state.get("lista_qds", [])
    lista_qds_global.extend(df_candidatos["QDS"].astype(float).tolist())
    st.session_state["lista_qds"] = lista_qds_global

    # Previsão base V14
    melhor = df_candidatos.iloc[0]
    previsao_base = melhor["series"]
    st.session_state["previsao_base_v14"] = previsao_base

    st.markdown("#### 🏁 Previsão base (V14-FLEX ULTRA)")
    st.code(" ".join(str(x) for x in previsao_base), language="text")

    st.caption(
        "Esta é a saída **pura do núcleo V14**, antes do tratamento de ruído "
        "e antes do TURBO++ ULTRA."
    )

    # --------------------------------------------------------
    # EXIBIÇÃO DA TABELA COMPLETA
    # --------------------------------------------------------

    st.markdown("#### 📋 Leque completo (ordenado por AIQ)")

    df_view = df_candidatos.copy()
    df_view["series"] = df_view["series"].apply(lambda x: " ".join(str(v) for v in x))

    st.dataframe(df_view, use_container_width=True)

    if st.session_state.get("mostrar_debug", False):
        st.markdown("#### 🐞 DEBUG — Estatísticas do leque")
        st.write(df_candidatos.describe(include="all"))
# ============================================================
# PARTE 3/6 — Replay LIGHT / Replay ULTRA / Replay ULTRA Unitário
# ============================================================


# ------------------------------------------------------------
# FUNÇÕES DE REPLAY E VALIDAÇÃO
# ------------------------------------------------------------

def medir_acertos(prev: List[int], real: List[int]) -> int:
    """Conta quantos passageiros foram acertados exatamente."""
    if not prev or not real:
        return 0
    return sum(1 for a, b in zip(prev, real) if a == b)


def montar_resultado_replay(
    df: pd.DataFrame,
    df_pred: List[List[int]],
    idx_ini: int,
    cols_pass: List[str],
) -> pd.DataFrame:
    """
    df — dataframe original
    df_pred — lista de previsões geradas
    idx_ini — índice interno 0-based
    """
    registros = []
    for i, prev in enumerate(df_pred):
        alvo = df.iloc[idx_ini + i]
        real = [int(alvo[c]) for c in cols_pass]
        ac = medir_acertos(prev, real)
        registros.append(
            {
                "serie": alvo["serie"],
                "prev": prev,
                "real": real,
                "acertos": ac,
            }
        )
    return pd.DataFrame(registros)


# ------------------------------------------------------------
# PAINEL 3 — Replay LIGHT
# ------------------------------------------------------------

if painel == "💡 Replay LIGHT":
    st.markdown("## 💡 Replay LIGHT — V15.5")
    st.markdown(
        """
        Executa um replay rápido para validar a coerência da estrada
        **sem tratamento de ruído** e **sem motor TURBO**.

        Ideal para testar:
        - estabilidade local
        - comportamento básico do V14-FLEX
        - coerência do histórico
        """
    )

    df_limpo = st.session_state.get("df_limpo")
    if df_limpo is None or df_limpo.empty:
        st.warning("Carregue o histórico no painel inicial.")
        st.stop()

    cols_pass = get_passenger_cols(df_limpo)
    n_hist = len(df_limpo)

    idx_ini = st.number_input(
        "Início do replay (índice):",
        min_value=1,
        max_value=n_hist - 1,
        value=max(1, n_hist - 200),
    )

    n_passos = st.slider(
        "Quantidade de replays (passos):",
        min_value=5,
        max_value=200,
        value=20,
    )

    st.markdown("### 📡 Executar Replay LIGHT")

    if st.button("▶️ Rodar Replay LIGHT"):
        with st.spinner("Executando Replay LIGHT…"):

            preds = []
            for off in range(n_passos):
                pos = int(idx_ini - 1 + off)
                if pos < 0 or pos >= n_hist - 1:
                    break

                df_j = extrair_janela_hist(df_limpo, pos + 1, back=50)
                matriz_freq = calcular_matriz_frequencia(df_j)

                cands = gerar_leque_candidatos(
                    matriz_freq,
                    n_series=40,
                    seed=1234 + pos,
                )
                df_c = montar_tabela_candidatos(
                    cands,
                    matriz_freq,
                    st.session_state["resumo_estrada"].regime_global,
                    calcular_k_star(df_j, janela=len(df_j)),
                )

                if df_c.empty:
                    preds.append([0] * len(cols_pass))
                else:
                    preds.append(df_c.iloc[0]["series"])

            df_rep = montar_resultado_replay(
                df_limpo,
                preds,
                int(idx_ini - 1),
                cols_pass,
            )

        st.success("Replay LIGHT finalizado.")
        df_view = df_rep.copy()
        df_view["prev"] = df_view["prev"].apply(lambda x: " ".join(str(v) for v in x))
        df_view["real"] = df_view["real"].apply(lambda x: " ".join(str(v) for v in x))

        st.dataframe(df_view, use_container_width=True)

        st.markdown("### 📊 Estatísticas")
        st.write(df_rep["acertos"].describe())


# ------------------------------------------------------------
# PAINEL 4 — Replay ULTRA
# ------------------------------------------------------------

if painel == "📅 Replay ULTRA":
    st.markdown("## 📅 Replay ULTRA — V15.5")
    st.markdown(
        """
        Replay robusto do V15.5:

        - Núcleo V14-FLEX ULTRA
        - Sem ruído
        - Sem TURBO++
        - Ideal para comparar com o Replay LIGHT
        """
    )

    df_limpo = st.session_state.get("df_limpo")
    if df_limpo is None or df_limpo.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    cols_pass = get_passenger_cols(df_limpo)
    n_hist = len(df_limpo)

    idx_ini_r = st.number_input(
        "Início do replay (índice):",
        min_value=1,
        max_value=n_hist - 1,
        value=max(1, n_hist - 150),
    )

    n_passos_r = st.slider(
        "Passos de replay:",
        min_value=5,
        max_value=200,
        value=30,
    )

    st.markdown("### 📡 Executar Replay ULTRA")

    if st.button("▶️ Rodar Replay ULTRA"):
        with st.spinner("Executando Replay ULTRA…"):

            preds = []
            for off in range(int(n_passos_r)):
                pos = int(idx_ini_r - 1 + off)
                if pos < 0 or pos >= n_hist - 1:
                    break

                df_j = extrair_janela_hist(df_limpo, pos + 1, back=80)
                matriz_freq = calcular_matriz_frequencia(df_j)

                cands = gerar_leque_candidatos(
                    matriz_freq,
                    n_series=60,
                    seed=777 + pos,
                )
                df_c = montar_tabela_candidatos(
                    cands,
                    matriz_freq,
                    st.session_state["resumo_estrada"].regime_global,
                    calcular_k_star(df_j, janela=len(df_j)),
                )

                if df_c.empty:
                    preds.append([0] * len(cols_pass))
                else:
                    preds.append(df_c.iloc[0]["series"])

            df_rep = montar_resultado_replay(
                df_limpo,
                preds,
                int(idx_ini_r - 1),
                cols_pass,
            )

        st.success("Replay ULTRA finalizado.")
        df_view = df_rep.copy()
        df_view["prev"] = df_view["prev"].apply(lambda x: " ".join(str(v) for v in x))
        df_view["real"] = df_view["real"].apply(lambda x: " ".join(str(v) for v in x))

        st.dataframe(df_view, use_container_width=True)

        st.markdown("### 📊 Estatísticas")
        st.write(df_rep["acertos"].describe())


# ------------------------------------------------------------
# PAINEL 5 — Replay ULTRA Unitário
# ------------------------------------------------------------

if painel == "🎯 Replay ULTRA Unitário":
    st.markdown("## 🎯 Replay ULTRA Unitário — V15.5")
    st.markdown(
        """
        Executa a previsão ULTRA **para uma única série** do histórico,
        permitindo estudar comportamento do modelo ponto a ponto.
        """
    )

    df_limpo = st.session_state.get("df_limpo")

    if df_limpo is None or df_limpo.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    cols_pass = get_passenger_cols(df_limpo)
    n_hist = len(df_limpo)

    idx_alvo_u = st.number_input(
        "Índice alvo do ULTRA Unitário:",
        min_value=1,
        max_value=n_hist - 1,
        value=n_hist - 1,
    )

    if st.button("▶️ Executar ULTRA Unitário"):
        with st.spinner("Executando ULTRA Unitário…"):

            pos = int(idx_alvo_u - 1)
            if pos < 0 or pos >= n_hist - 1:
                st.error("Índice fora do intervalo.")
                st.stop()

            df_j = extrair_janela_hist(df_limpo, pos + 1, back=80)
            matriz_freq = calcular_matriz_frequencia(df_j)

            cands = gerar_leque_candidatos(
                matriz_freq,
                n_series=80,
                seed=999 + pos,
            )

            df_c = montar_tabela_candidatos(
                cands,
                matriz_freq,
                st.session_state["resumo_estrada"].regime_global,
                calcular_k_star(df_j, janela=len(df_j)),
            )

            if df_c.empty:
                st.error("Nenhum candidato encontrado.")
                st.stop()

            prev = df_c.iloc[0]["series"]
            real = [int(df_limpo.iloc[pos][c]) for c in cols_pass]
            acertos = medir_acertos(prev, real)

        st.success("ULTRA Unitário finalizado.")

        st.markdown("### 🎯 Previsão ULTRA Unitário")
        st.code(" ".join(str(x) for x in prev), language="text")

        st.markdown("### 📌 Real")
        st.code(" ".join(str(x) for x in real), language="text")

        st.metric("Acertos", acertos)
# ============================================================
# PARTE 4/6 — Testes de Confiabilidade REAL + Ruído Condicional (V15)
# ============================================================


# ------------------------------------------------------------
# QDS REAL — cálculo
# ------------------------------------------------------------

def calcular_qds_real(df: pd.DataFrame) -> List[float]:
    """
    Calcula o QDS REAL avaliando a coerência do histórico.

    Estratégia:
    - Para cada série, avalia-se seu encaixe na matriz de frequências
      da janela anterior.
    """
    if df is None or df.empty:
        return []

    cols_pass = get_passenger_cols(df)
    n = len(df)
    qds_list = []

    for i in range(1, n):
        df_j = extrair_janela_hist(df, i, back=min(60, i))
        if df_j.empty:
            qds_list.append(0.0)
            continue

        matriz_freq = calcular_matriz_frequencia(df_j)
        alvo = df.iloc[i]
        real = [int(alvo[c]) for c in cols_pass]

        score = calcular_qds_candidato(real, matriz_freq)
        qds_list.append(score)

    return qds_list


# ------------------------------------------------------------
# BACKTEST REAL
# ------------------------------------------------------------

def executar_backtest_real(
    df: pd.DataFrame,
    janela_back: int,
    n_cand: int,
) -> Tuple[List[int], List[List[int]]]:
    """
    Executa o Backtest REAL:

    Para cada posição i:
      - extrai janela (i - janela_back)
      - gera candidatos
      - escolhe melhor
      - compara com real
    """
    cols_pass = get_passenger_cols(df)
    n = len(df)

    acertos_lista = []
    historico_prev = []

    for i in range(janela_back, n - 1):
        df_j = extrair_janela_hist(df, i, back=janela_back)
        if df_j.empty:
            acertos_lista.append(0)
            historico_prev.append([0] * len(cols_pass))
            continue

        matriz_freq = calcular_matriz_frequencia(df_j)
        cands = gerar_leque_candidatos(
            matriz_freq,
            n_series=n_cand,
            seed=2025 + i,
        )
        df_c = montar_tabela_candidatos(
            cands,
            matriz_freq,
            classificar_regime_por_k(df_j["k"].mean()),
            calcular_k_star(df_j, janela=len(df_j)),
        )

        if df_c.empty:
            acertos_lista.append(0)
            historico_prev.append([0] * len(cols_pass))
            continue

        prev = df_c.iloc[0]["series"]
        alvo = df.iloc[i]
        real = [int(alvo[c]) for c in cols_pass]
        ac = medir_acertos(prev, real)

        acertos_lista.append(ac)
        historico_prev.append(prev)

    return acertos_lista, historico_prev


# ------------------------------------------------------------
# MONTE CARLO REAL
# ------------------------------------------------------------

def executar_monte_carlo_real(
    df: pd.DataFrame,
    n_sim: int,
    janela_back: int,
    n_cand: int,
) -> List[List[int]]:
    """
    Monte Carlo REAL:
    Para cada simulação:
      - percorre todo o histórico
      - gera candidatos com seeds diferentes
      - registra acertos
    """
    cols_pass = get_passenger_cols(df)
    n = len(df)

    matriz_acertos = []

    for s in range(n_sim):
        acertos = []
        for i in range(janela_back, n - 1):
            df_j = extrair_janela_hist(df, i, back=janela_back)
            if df_j.empty:
                acertos.append(0)
                continue

            matriz_freq = calcular_matriz_frequencia(df_j)
            cands = gerar_leque_candidatos(
                matriz_freq,
                n_series=n_cand,
                seed=1000 * (s + 1) + i,
            )
            df_c = montar_tabela_candidatos(
                cands,
                matriz_freq,
                classificar_regime_por_k(df_j["k"].mean()),
                calcular_k_star(df_j, janela=len(df_j)),
            )

            if df_c.empty:
                acertos.append(0)
                continue

            prev = df_c.iloc[0]["series"]
            alvo = df.iloc[i]
            real = [int(alvo[c]) for c in cols_pass]
            acertos.append(medir_acertos(prev, real))

        matriz_acertos.append(acertos)

    return matriz_acertos


# ------------------------------------------------------------
# PAINEL 6 — Testes de Confiabilidade REAL
# ------------------------------------------------------------

if painel == "🧪 Testes de Confiabilidade REAL":
    st.markdown("## 🧪 Testes de Confiabilidade REAL — V15.5")
    st.markdown(
        """
        Este painel executa:
        - QDS REAL
        - Backtest REAL
        - Monte Carlo REAL

        Tudo com dados **reais** do histórico,
        sem previsões artificiais e sem ruído.
        """
    )

    df_limpo = st.session_state.get("df_limpo")
    if df_limpo is None or df_limpo.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    cols_pass = get_passenger_cols(df_limpo)
    n_hist = len(df_limpo)

    st.markdown("### ⚙️ Parâmetros")

    col_t1, col_t2, col_t3 = st.columns(3)

    with col_t1:
        janela_back = st.slider(
            "Janela backtest/m.carlo:",
            min_value=10,
            max_value=min(300, n_hist - 1),
            value=min(60, n_hist - 1),
            step=5,
        )
    with col_t2:
        n_cand = st.slider(
            "Tamanho do leque (por janela):",
            min_value=10,
            max_value=200,
            value=60,
            step=5,
        )
    with col_t3:
        n_sim = st.slider(
            "Simulações Monte Carlo:",
            min_value=5,
            max_value=50,
            value=10,
        )

    if st.button("▶️ Executar Testes de Confiabilidade REAL"):
        with st.spinner("Executando QDS / Backtest / Monte Carlo…"):

            # --- QDS REAL ---
            qds_real = calcular_qds_real(df_limpo)
            resumo_qds = calcular_resumo_qds(qds_real)
            st.session_state["resumo_qds"] = resumo_qds

            # --- Backtest REAL ---
            acertos_back, hist_prev = executar_backtest_real(
                df_limpo, janela_back, n_cand
            )
            resumo_back = calcular_resumo_backtest(acertos_back, len(cols_pass))
            st.session_state["resumo_backtest"] = resumo_back
            st.session_state["historico_backtest"] = hist_prev

            # --- Monte Carlo REAL ---
            matriz_acertos = executar_monte_carlo_real(
                df_limpo, n_sim, janela_back, n_cand
            )
            resumo_mc = calcular_resumo_monte_carlo(matriz_acertos)
            st.session_state["resumo_montecarlo"] = resumo_mc
            st.session_state["historico_montecarlo"] = matriz_acertos

        st.success("Testes finalizados.")

        st.markdown("### 📊 QDS REAL")
        if resumo_qds:
            st.metric("QDS médio", f"{resumo_qds.qds_medio:.3f}")
            st.metric("QDS min", f"{resumo_qds.qds_min:.3f}")
            st.metric("QDS max", f"{resumo_qds.qds_max:.3f}")

            col_q1, col_q2, col_q3, col_q4 = st.columns(4)
            col_q1.metric("% PREMIUM", f"{resumo_qds.pct_premium:.1f}%")
            col_q2.metric("% BOM", f"{resumo_qds.pct_bom:.1f}%")
            col_q3.metric("% REGULAR", f"{resumo_qds.pct_regular:.1f}%")
            col_q4.metric("% RUIM", f"{resumo_qds.pct_ruim:.1f}%")

        st.markdown("### 🎯 Backtest REAL")
        if resumo_back:
            st.metric("Janelas", resumo_back.n_janelas)
            st.metric("Acertos totais", resumo_back.acertos_totais)
            st.metric("Acertos por série", f"{resumo_back.acertos_por_serie:.3f}")
            st.metric("Hit-rate", f"{resumo_back.hit_rate:.3f}")

        st.markdown("### 🎲 Monte Carlo REAL")
        if resumo_mc:
            st.metric("Simulações", resumo_mc.n_simulacoes)
            st.metric("Média de acertos", f"{resumo_mc.media_acertos:.3f}")
            st.metric("Desvio", f"{resumo_mc.desvio_acertos:.3f}")
            st.metric("Melhor média", f"{resumo_mc.melhor_serie_media:.3f}")


# ------------------------------------------------------------
# PAINEL 7 — Ruído Condicional (V15)
# ------------------------------------------------------------

def aplicar_ruido_condicional(
    df: pd.DataFrame,
    magnitude: float = 0.15,
) -> Tuple[pd.DataFrame, float, float, float]:
    """
    Aplica ruído condicional:
    - magnitude: força do ajuste (0.0–1.0)
    - retorna df_modificado + estatísticas
    """
    if df is None or df.empty:
        return df.copy(), 0.0, 0.0, 0.0

    cols = get_passenger_cols(df)
    df_mod = df.copy()

    ruido_inicial = 0
    ruido_final = 0
    pontos_aj = 0
    total_pts = len(df_mod) * len(cols)

    for col in cols:
        original = df_mod[col].astype(float).copy()
        ruido_inicial += float((np.std(original)))

        ruido = np.random.normal(loc=0.0, scale=magnitude * np.std(original), size=len(df_mod))
        df_mod[col] = (original + ruido).round().astype(int)

        ruido_final += float((np.std(df_mod[col].astype(float))))
        pontos_aj += sum((df_mod[col] != original).astype(int))

    pct_pontos = (pontos_aj / total_pts) * 100 if total_pts > 0 else 0.0

    return df_mod, float(ruido_inicial), float(ruido_final), float(pct_pontos)


if painel == "📊 Ruído Condicional (V15)":
    st.markdown("## 📊 Ruído Condicional (V15)")
    st.markdown(
        """
        Aplica ruído condicional para gerar:
        - Estrada A (ruído leve)
        - Estrada B (ruído forte)
        """
    )

    df_limpo = st.session_state.get("df_limpo")
    if df_limpo is None or df_limpo.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    magnitude_a = st.slider(
        "Magnitude do Ruído A (leve):",
        min_value=0.01,
        max_value=0.50,
        value=0.08,
        step=0.01,
    )
    magnitude_b = st.slider(
        "Magnitude do Ruído B (forte):",
        min_value=0.02,
        max_value=0.80,
        value=0.20,
        step=0.02,
    )

    if st.button("▶️ Aplicar Ruído"):
        with st.spinner("Gerando Estradas A e B…"):

            dfA, rA_in, rA_out, pA = aplicar_ruido_condicional(df_limpo, magnitude_a)
            dfB, rB_in, rB_out, pB = aplicar_ruido_condicional(df_limpo, magnitude_b)

            st.session_state["df_ruido_a"] = dfA
            st.session_state["df_ruido_b"] = dfB

            resumo_ruido = calcular_resumo_ruido(
                ruido_inicial=(rA_in + rB_in) / 2,
                ruido_final=(rA_out + rB_out) / 2,
                pct_pontos_ajustados=(pA + pB) / 2,
            )
            st.session_state["resumo_ruido"] = resumo_ruido

        st.success("Ruído A/B aplicado com sucesso.")

        st.markdown("### 🛣️ Estrada A (ruído leve)")
        st.dataframe(st.session_state["df_ruido_a"].head(50))

        st.markdown("### 🛣️ Estrada B (ruído forte)")
        st.dataframe(st.session_state["df_ruido_b"].head(50))

        if resumo_ruido:
            st.markdown("### 📊 Estatísticas médias do ruído")
            st.metric("Ruído inicial (médio)", f"{resumo_ruido.ruido_inicial:.3f}")
            st.metric("Ruído final (médio)", f"{resumo_ruido.ruido_final:.3f}")
            st.metric(
                "% pontos ajustados (médio)",
                f"{resumo_ruido.pct_pontos_ajustados:.2f}%",
            )
# ============================================================
# PARTE 5/6 — Modo TURBO++ ULTRA ANTI-RUÍDO (V15.5)
# ============================================================

# ------------------------------------------------------------
# FUNÇÕES AUXILIARES DO TURBO++ ULTRA
# ------------------------------------------------------------

def gerar_matriz_freq_para_df(
    df_base: pd.DataFrame,
    idx_alvo: int,
    janela_back: int,
) -> Dict[str, Dict[int, float]]:
    """
    Extrai janela de df_base e calcula matriz de frequências.
    """
    df_j = extrair_janela_hist(df_base, idx_alvo=idx_alvo, back=janela_back)
    if df_j.empty:
        return {}
    return calcular_matriz_frequencia(df_j)


def gerar_leque_completo(
    df_base: pd.DataFrame,
    idx_alvo: int,
    janela_back: int,
    n_cand: int,
    seed_base: int,
    peso_qds: float,
    resumo_k_local: ResumoK,
    regime_global: str,
) -> pd.DataFrame:
    """
    Gera leque completo a partir de uma estrada base (original ou ruidosa).
    """
    df_j = extrair_janela_hist(df_base, idx_alvo=idx_alvo, back=janela_back)
    if df_j.empty:
        return pd.DataFrame()

    matriz_freq = calcular_matriz_frequencia(df_j)
    if not matriz_freq:
        return pd.DataFrame()

    candidatos = gerar_leque_candidatos(
        matriz_freq,
        n_series=n_cand,
        seed=int(seed_base + idx_alvo * 11),
    )

    df_cand = montar_tabela_candidatos(
        candidatos,
        matriz_freq,
        regime_global,
        resumo_k_local,
    )

    if df_cand.empty:
        return df_cand

    novas_aiq = []
    for _, row in df_cand.iterrows():
        serie = row["series"]
        _, qds_tmp, div_tmp = calcular_aiq_candidato(
            serie,
            matriz_freq,
            peso_qds=peso_qds,
            peso_div=1.0 - peso_qds,
        )
        novas_aiq.append(peso_qds * qds_tmp + (1.0 - peso_qds) * div_tmp)

    df_cand["AIQ"] = novas_aiq
    df_cand = df_cand.sort_values(["AIQ", "QDS"], ascending=[False, False]).reset_index(
        drop=True
    )
    return df_cand


def unir_leques_v15(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """
    Une leques A e B, removendo duplicidades e reordenando por AIQ/QDS.
    """
    if df_a is None or df_a.empty:
        return df_b.copy() if df_b is not None else pd.DataFrame()
    if df_b is None or df_b.empty:
        return df_a.copy()

    df_mix = pd.concat([df_a, df_b], ignore_index=True)

    if "series" not in df_mix.columns:
        return df_mix

    df_mix["serie_str"] = df_mix["series"].apply(lambda x: tuple(x))
    df_mix = df_mix.drop_duplicates(subset=["serie_str"]).drop(columns=["serie_str"])

    if "AIQ" in df_mix.columns and "QDS" in df_mix.columns:
        df_mix = df_mix.sort_values(
            ["AIQ", "QDS"], ascending=[False, False]
        ).reset_index(drop=True)

    return df_mix


# ------------------------------------------------------------
# PAINEL — Modo TURBO++ ULTRA ANTI-RUÍDO (V15)
# ------------------------------------------------------------

if painel == "🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15)":
    st.markdown("## 🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15.5)")
    st.markdown(
        """
        Motor final do **Predict Cars V15.5-HÍBRIDO**:

        - Usa Estrada Original + Estrada A + Estrada B
        - Gera Leque A (ruído leve) e Leque B (ruído forte)
        - Mescla e recalibra (Leque MISTO)
        - Aplica AIQ-HÍBRIDO (QDS + diversidade)
        - Usa k* local como contexto de risco
        - Produz a **Previsão Final TURBO++ ULTRA (V15.5)**
        """
    )

    df_limpo = st.session_state.get("df_limpo")
    df_ra = st.session_state.get("df_ruido_a")
    df_rb = st.session_state.get("df_ruido_b")

    if df_limpo is None or df_limpo.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    if df_ra is None or df_rb is None:
        st.warning(
            "Estradas A/B ainda não existem. "
            "Vá ao painel '📊 Ruído Condicional (V15)' e aplique o ruído."
        )
        st.stop()

    resumo_estrada: Optional[ResumoEstrada] = st.session_state.get("resumo_estrada")
    regime_global_str = (
        resumo_estrada.regime_global if resumo_estrada else "desconhecido"
    )

    n_hist = len(df_limpo)

    # --------------------------------------------------------
    # CONTROLES DO TURBO++
    # --------------------------------------------------------

    st.markdown("### ⚙️ Parâmetros do TURBO++ ULTRA")

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        idx_alvo = st.number_input(
            "Índice alvo:",
            min_value=1,
            max_value=n_hist,
            value=n_hist,
            step=1,
        )
    with col_t2:
        janela_back = st.slider(
            "Janela para trás:",
            min_value=10,
            max_value=min(300, n_hist - 1),
            value=min(60, n_hist - 1),
            step=5,
        )
    with col_t3:
        n_cand = st.slider(
            "Leque A/B (n candidatos):",
            min_value=10,
            max_value=200,
            value=80,
            step=5,
        )

    col_t4, col_t5 = st.columns(2)
    with col_t4:
        peso_qds = st.slider(
            "Peso do QDS (AIQ-HÍBRIDO):",
            min_value=0.1,
            max_value=0.9,
            value=0.65,
            step=0.05,
        )
    with col_t5:
        seed_turbo = st.number_input(
            "Seed base (TURBO):",
            min_value=1,
            max_value=999999,
            value=2025,
        )

    # k* local com base na janela da estrada original
    df_j = extrair_janela_hist(df_limpo, int(idx_alvo), back=int(janela_back))
    resumo_k_local = calcular_k_star(df_j, janela=len(df_j))

    st.markdown("### 🔭 k* Local (para o TURBO)")
    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        st.metric("k atual", resumo_k_local.k_atual)
    with col_k2:
        st.metric("k*", f"{resumo_k_local.k_star*100:.2f}%")
    with col_k3:
        st.metric("Regime local", resumo_k_local.regime_local)

    # --------------------------------------------------------
    # EXECUÇÃO DO TURBO++
    # --------------------------------------------------------

    if st.button("▶️ Executar TURBO++ ULTRA"):
        with st.spinner("Executando TURBO++ ULTRA…"):

            # --- LEQUE A (Estrada A, ruído leve) -------------
            dfA = gerar_leque_completo(
                df_base=df_ra,
                idx_alvo=int(idx_alvo),
                janela_back=int(janela_back),
                n_cand=int(n_cand),
                seed_base=int(seed_turbo),
                peso_qds=float(peso_qds),
                resumo_k_local=resumo_k_local,
                regime_global=regime_global_str,
            )

            # --- LEQUE B (Estrada B, ruído forte) -------------
            dfB = gerar_leque_completo(
                df_base=df_rb,
                idx_alvo=int(idx_alvo),
                janela_back=int(janela_back),
                n_cand=int(n_cand),
                seed_base=int(seed_turbo + 99),
                peso_qds=float(peso_qds),
                resumo_k_local=resumo_k_local,
                regime_global=regime_global_str,
            )

            # --- LEQUE MISTO ---------------------------------
            df_mix = unir_leques_v15(dfA, dfB)

            if df_mix.empty:
                st.error("Falha ao montar Leque MISTO (A+B). Ajuste parâmetros.")
                st.stop()

            melhor = df_mix.iloc[0]
            previsao_final = melhor["series"]
            st.session_state["previsao_turbo_ultra"] = previsao_final

        # ----------------------------------------------------
        # RESULTADOS
        # ----------------------------------------------------
        st.success("TURBO++ ULTRA finalizado.")

        st.markdown("### 🎯 Previsão Final TURBO++ ULTRA — V15.5")
        st.code(" ".join(str(x) for x in previsao_final), language="text")

        # ⚠️ Alerta de risco baseado no k* local
        if resumo_k_local.estado_k == "estavel":
            st.info("🟢 Ambiente estável — previsão em regime normal.")
        elif resumo_k_local.estado_k == "atencao":
            st.warning("🟡 Pré-ruptura residual — usar previsão com atenção.")
        else:
            st.error("🔴 Ambiente crítico — usar previsão com cautela máxima.")

        # ----------------------------------------------------
        # DEBUG OPCIONAL
        # ----------------------------------------------------
        if st.session_state.get("mostrar_debug", False):
            st.markdown("#### 🐞 DEBUG — Leque A")
            dfA_view = dfA.copy()
            if not dfA_view.empty:
                dfA_view["series"] = dfA_view["series"].apply(
                    lambda x: " ".join(str(v) for v in x)
                )
                st.dataframe(dfA_view.head(20), use_container_width=True)
            else:
                st.write("Leque A vazio.")

            st.markdown("#### 🐞 DEBUG — Leque B")
            dfB_view = dfB.copy()
            if not dfB_view.empty:
                dfB_view["series"] = dfB_view["series"].apply(
                    lambda x: " ".join(str(v) for v in x)
                )
                st.dataframe(dfB_view.head(20), use_container_width=True)
            else:
                st.write("Leque B vazio.")

            st.markdown("#### 🐞 DEBUG — Leque MISTO")
            dfM_view = df_mix.copy()
            if not dfM_view.empty:
                dfM_view["series"] = dfM_view["series"].apply(
                    lambda x: " ".join(str(v) for v in x)
                )
                st.dataframe(dfM_view.head(20), use_container_width=True)
            else:
                st.write("Leque MISTO vazio.")
# ============================================================
# PARTE 6/6 — Relatório Final — AIQ Bridge (para ChatGPT)
# ============================================================

# ------------------------------------------------------------
# FUNÇÕES AUXILIARES DO RELATÓRIO
# ------------------------------------------------------------

def formatar_percentual(v: float) -> str:
    if 0.0 <= v <= 1.0:
        return f"{v*100:.2f}%"
    return f"{v:.2f}%"


def gerar_expectativa_acertos(
    regime_local: str,
    k_star: float,
    qds_medio: float,
) -> str:
    """
    Heurística do V15.5 para expectativa de acertos da previsão final.
    """
    if regime_local == "Ultra Estável":
        base = 3.2
    elif regime_local == "Estável":
        base = 2.7
    elif regime_local == "Transição":
        base = 2.2
    elif regime_local == "Turbulento":
        base = 1.7
    else:
        base = 1.2

    ajuste_k = max(0.0, 1.0 - k_star)
    ajuste_qds = max(0.0, min(1.0, qds_medio))

    expectativa = base * (0.7 + 0.3 * ajuste_qds) * (0.6 + 0.4 * ajuste_k)
    return f"{expectativa:.2f} acertos (esperados)"


# ------------------------------------------------------------
# PAINEL — Relatório Final — AIQ Bridge
# ------------------------------------------------------------

if painel == "📄 Relatório Final — AIQ Bridge (para ChatGPT)":
    st.markdown("## 📄 Relatório Final — AIQ Bridge (para ChatGPT)")
    st.markdown(
        """
        Painel oficial do **Predict Cars V15.5-HÍBRIDO** para exportar
        tudo o que o sistema entende da estrada, pronto para colar no ChatGPT.
        """
    )

    df_limpo = st.session_state.get("df_limpo")
    resumo_estrada: Optional[ResumoEstrada] = st.session_state.get("resumo_estrada")
    resumo_k_global: Optional[ResumoK] = st.session_state.get("resumo_k_global")
    resumo_qds: Optional[ResumoQDS] = st.session_state.get("resumo_qds")
    resumo_ruido: Optional[ResumoRuido] = st.session_state.get("resumo_ruido")
    resumo_backtest: Optional[ResumoBacktest] = st.session_state.get("resumo_backtest")
    resumo_montecarlo: Optional[ResumoMonteCarlo] = st.session_state.get(
        "resumo_montecarlo"
    )
    previsao_turbo = st.session_state.get("previsao_turbo_ultra")

    if df_limpo is None or df_limpo.empty:
        st.error("Histórico não carregado. Gere o relatório após carregar o histórico.")
        st.stop()

    # Se QDS REAL ainda não foi calculado, tenta calcular com lista_qds
    if resumo_qds is None:
        lista_qds = st.session_state.get("lista_qds", [])
        if lista_qds:
            resumo_qds = calcular_resumo_qds(lista_qds)
            st.session_state["resumo_qds"] = resumo_qds

    relatorio: List[str] = []
    relatorio.append("# 🔵 Predict Cars V15.5 — AIQ Bridge Report\n")

    # --------------------------------------------------------
    # Estrada Global
    # --------------------------------------------------------
    relatorio.append("## 🛣️ Estrada — Resumo Global\n")

    if resumo_estrada:
        relatorio.append(f"- Total de séries: **{resumo_estrada.n_series}**")
        relatorio.append(f"- Passageiros por série: **{resumo_estrada.n_passageiros}**")
        relatorio.append(
            f"- Faixas de valores (n1..nN): **{resumo_estrada.min_val} — {resumo_estrada.max_val}**"
        )
        relatorio.append(f"- Média global dos passageiros: **{resumo_estrada.media:.2f}**")
        relatorio.append(f"- Desvio-padrão global: **{resumo_estrada.desvio:.2f}**")
        relatorio.append(f"- k médio histórico: **{resumo_estrada.k_medio:.3f}**")
        relatorio.append(f"- k máximo histórico: **{resumo_estrada.k_max}**")
        relatorio.append(
            f"- Regime global (barômetro da estrada): **{resumo_estrada.regime_global}**"
        )
    else:
        relatorio.append("- Resumo global da estrada indisponível.")

    relatorio.append("")

    # --------------------------------------------------------
    # k* Global
    # --------------------------------------------------------
    relatorio.append("## 🔭 k* — Sentinela Global da Estrada\n")

    if resumo_k_global:
        estado_label = {
            "estavel": "🟢 Estável",
            "atencao": "🟡 Pré-ruptura residual",
            "critico": "🔴 Crítico",
        }.get(resumo_k_global.estado_k, "⚪ Indeterminado")

        relatorio.append(f"- k atual (última série): **{resumo_k_global.k_atual}**")
        relatorio.append(f"- k*: **{resumo_k_global.k_star*100:.2f}%**")
        relatorio.append(f"- Regime local (k*): **{resumo_k_global.regime_local}**")
        relatorio.append(f"- Estado de risco: {estado_label}")
    else:
        relatorio.append("- k* global não calculado.")

    relatorio.append("")

    # --------------------------------------------------------
    # QDS GLOBAL
    # --------------------------------------------------------
    relatorio.append("## 📊 QDS — Qualidade Dinâmica da Série\n")

    if resumo_qds:
        relatorio.append(f"- QDS médio: **{resumo_qds.qds_medio:.3f}**")
        relatorio.append(f"- QDS mínimo: **{resumo_qds.qds_min:.3f}**")
        relatorio.append(f"- QDS máximo: **{resumo_qds.qds_max:.3f}**")
        relatorio.append(
            f"- Distribuição: PREMIUM {resumo_qds.pct_premium:.1f}%, "
            f"BOM {resumo_qds.pct_bom:.1f}%, "
            f"REGULAR {resumo_qds.pct_regular:.1f}%, "
            f"RUIM {resumo_qds.pct_ruim:.1f}%"
        )
    else:
        relatorio.append(
            "- QDS global não disponível (execute os Testes de Confiabilidade REAL)."
        )

    relatorio.append("")

    # --------------------------------------------------------
    # Ruído Condicional A/B
    # --------------------------------------------------------
    relatorio.append("## 🎛️ Ruído Condicional — Estradas A/B\n")

    if resumo_ruido:
        relatorio.append(
            f"- Ruído inicial médio (antes dos ajustes): **{resumo_ruido.ruido_inicial:.3f}**"
        )
        relatorio.append(
            f"- Ruído final médio (após ajustes): **{resumo_ruido.ruido_final:.3f}**"
        )
        relatorio.append(
            f"- % de pontos ajustados (médio A/B): **{resumo_ruido.pct_pontos_ajustados:.2f}%**"
        )
    else:
        relatorio.append(
            "- Ruído condicional ainda não aplicado (Estradas A/B não geradas)."
        )

    relatorio.append("")

    # --------------------------------------------------------
    # Backtest REAL
    # --------------------------------------------------------
    relatorio.append("## 🔁 Backtest REAL — Métricas\n")

    if resumo_backtest:
        relatorio.append(f"- Janelas avaliadas: **{resumo_backtest.n_janelas}**")
        relatorio.append(
            f"- Acertos totais (soma de acertos em todas as janelas): "
            f"**{resumo_backtest.acertos_totais}**"
        )
        relatorio.append(
            f"- Acertos médios por série (backtest): **{resumo_backtest.acertos_por_serie:.4f}**"
        )
        relatorio.append(
            f"- Hit-rate global (backtest): **{resumo_backtest.hit_rate*100:.2f}%**"
        )
    else:
        relatorio.append(
            "- Backtest REAL não executado (execute em '🧪 Testes de Confiabilidade REAL')."
        )

    relatorio.append("")

    # --------------------------------------------------------
    # Monte Carlo REAL
    # --------------------------------------------------------
    relatorio.append("## 🎲 Monte Carlo REAL — Métricas\n")

    if resumo_montecarlo:
        relatorio.append(
            f"- Simulações Monte Carlo: **{resumo_montecarlo.n_simulacoes}**"
        )
        relatorio.append(
            f"- Média de acertos (por simulação): **{resumo_montecarlo.media_acertos:.4f}**"
        )
        relatorio.append(
            f"- Desvio dos acertos: **{resumo_montecarlo.desvio_acertos:.4f}**"
        )
        relatorio.append(
            f"- Melhor média de acertos entre as simulações: "
            f"**{resumo_montecarlo.melhor_serie_media:.4f}**"
        )
    else:
        relatorio.append(
            "- Monte Carlo REAL não executado (execute em '🧪 Testes de Confiabilidade REAL')."
        )

    relatorio.append("")

    # --------------------------------------------------------
    # Previsão Final TURBO++ ULTRA (V15.5)
    # --------------------------------------------------------
    relatorio.append("## 🎯 Previsão Final — TURBO++ ULTRA (V15.5)\n")

    if previsao_turbo:
        relatorio.append(
            f"- Série prevista (motor V15.5): **{' '.join(str(x) for x in previsao_turbo)}**"
        )
    else:
        relatorio.append(
            "- Previsão final ainda não gerada (use o painel '🚀 Modo TURBO++ ULTRA ANTI-RUÍDO')."
        )

    relatorio.append("")

    # --------------------------------------------------------
    # Expectativa de Acertos
    # --------------------------------------------------------
    relatorio.append("## 🎯 Expectativa de Acertos (V15.5)\n")

    if resumo_k_global and resumo_qds:
        exp = gerar_expectativa_acertos(
            regime_local=resumo_k_global.regime_local,
            k_star=resumo_k_global.k_star,
            qds_medio=resumo_qds.qds_medio,
        )
        relatorio.append(f"- Expectativa projetada de acertos: **{exp}**")
    else:
        relatorio.append(
            "- Ainda não é possível estimar claramente a expectativa de acertos "
            "(faltam k* global e/ou QDS global)."
        )

    relatorio.append("\n---\n")
    relatorio.append(
        "### ✔ Relatório pronto para ser enviado ao ChatGPT.\n"
        "Use este texto integralmente para análise, refinamento e próximos passos do Predict Cars."
    )

    texto_relatorio = "\n".join(relatorio)

    st.markdown("### 📄 Relatório consolidado")
    st.text_area(
        "Copie o relatório completo abaixo:",
        value=texto_relatorio,
        height=700,
    )
