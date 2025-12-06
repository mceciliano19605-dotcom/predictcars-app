# ============================================================
# Predict Cars V15.5-HÍBRIDO
# Núcleo V14-FLEX ULTRA + k* + Ruído Condicional + QDS REAL +
# Backtest REAL (com proteção p/ históricos grandes) +
# Monte Carlo REAL + AIQ Bridge (para ChatGPT)
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

# Limite de segurança para não rodar REPLAY ULTRA / BACKTEST REAL completo
# em históricos muito grandes (evitar travamento zumbi).
LIMITE_REPLAY_HIST = 3000  # 3k+ séries = desabilita replay/ backtest total

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
# FUNÇÕES UTILITÁRIAS — PARSE DE HISTÓRICO FLEX ULTRA
# ------------------------------------------------------------

def _detect_separator(sample_line: str) -> str:
    """Detecta separador provável entre ; , \t ou espaço."""
    for sep in [";", ",", "\t", " "]:
        if sep in sample_line:
            return sep
    return ";"


def parse_historico_text(raw: str) -> pd.DataFrame:
    """
    Parser FLEX ULTRA para texto colado.

    Aceita:
        C1;41;5;4;52;30;33;0
        41;5;4;52;30;33;0
        etc.
    """
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
            {
                "serie_id": serie_id,
                "passageiros": passageiros,
                "k": k_val,
            }
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
    """
    Parser FLEX ULTRA para arquivo CSV.

    Casos aceitos:
    - serie,n1..nN,k
    - C1;41;5;4;52;30;33;0
    - 41;5;4;52;30;33;0; etc.
    """
    try:
        df_raw = pd.read_csv(file, sep=None, engine="python")
    except Exception:
        file.seek(0)
        df_raw = pd.read_csv(file, sep=";")

    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    # Caso 1 — apenas 1 coluna, pode ser string com tudo dentro
    if df_raw.shape[1] == 1:
        colname = df_raw.columns[0]
        raw = "\n".join(str(x) for x in df_raw[colname].astype(str))
        return parse_historico_text(raw)

    cols_lower = [c.lower() for c in df_raw.columns]
    has_serie = any(c.startswith("serie") for c in cols_lower)
    has_k = any(c == "k" for c in cols_lower)

    df = df_raw.copy()

    if has_serie and has_k:
        # Mapeia para 'serie' e 'k'
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
        # Assume: primeira coluna = ID, última = k
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
# FUNÇÕES DE RESUMO / REGIME / K*
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
    """k* simples: % de séries com k>0 na janela final."""
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
    "Backtest REAL (protegido p/ estradas grandes) + Monte Carlo REAL + "
    "AIQ Bridge (para ChatGPT)."
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
        # Ordena por índice numérico da série, se aplicável
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
        key=lambda x: int(x[1:])
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

        probs: Dict[int, float] = {}
        for v, c in contagens.items():
            probs[v] = (c + suavizacao) / total

        matriz[col] = probs

    return matriz


# ------------------------------------------------------------
# FUNÇÃO CORRIGIDA — GERAR CANDIDATO (AGORA COM NP.CHOICE)
# ------------------------------------------------------------

def gerar_candidato_serie(
    matriz_freq: Dict[str, Dict[int, float]],
    rng: random.Random,
) -> List[int]:
    """
    Gera uma série candidata baseada na matriz de frequências.

    🔧 Correção V15.5:
    - rng.choice() é inválido → substituído por np.random.choice()
    - Mantém rng para randint (consistência determinística)
    """
    if not matriz_freq:
        return [rng.randint(0, 60) for _ in range(6)]

    serie: List[int] = []

    for col in sorted(matriz_freq.keys(), key=lambda x: int(x[1:])):
        dist = matriz_freq[col]

        if not dist:
            serie.append(rng.randint(0, 60))
            continue

        valores = list(dist.keys())
        probs = np.array([dist[v] for v in valores], dtype=float)
        probs = probs / probs.sum()

        # 🔧 CORREÇÃO IMPORTANTE:
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
    """AIQ = combinação ponderada de QDS e Diversidade."""
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
    - série
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
    return df.sort_values(["AIQ", "QDS"], ascending=[False, False]).reset_index(drop=True)


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
        if resumo_k_global:
            st.metric("k médio global", f"{resumo_estrada.k_medio:.2f}" if resumo_estrada else "-")

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
# PARTE 3/6 — Replay LIGHT • Replay ULTRA • Replay Unitário • k*
# ============================================================

# ------------------------------------------------------------
# FUNÇÕES AUXILIARES DO REPLAY
# ------------------------------------------------------------

def replay_calcular_previsao_v14(
    df_limpo: pd.DataFrame,
    idx_alvo: int,
    janela_back: int,
    n_candidatos: int,
    seed_base: int,
    peso_qds: float,
) -> Optional[List[int]]:
    """
    Esta função executa o mesmo pipeline do painel V14-FLEX ULTRA,
    mas devolve apenas a previsão base gerada, para fins de replay.
    """
    try:
        df_janela = extrair_janela_hist(
            df_limpo, idx_alvo=idx_alvo, back=janela_back
        )
        if df_janela.empty:
            return None

        matriz_freq = calcular_matriz_frequencia(df_janela)
        if not matriz_freq:
            return None

        # Determina k* local dentro da janela
        resumo_k_local = calcular_k_star(df_janela, janela=len(df_janela))

        # Regime global
        resumo_estrada = st.session_state.get("resumo_estrada")
        regime_global_str = resumo_estrada.regime_global if resumo_estrada else "desconhecido"

        candidatos = gerar_leque_candidatos(
            matriz_freq,
            n_series=int(n_candidatos),
            seed=int(seed_base + int(idx_alvo) * 13),
        )

        df_cand = montar_tabela_candidatos(
            candidatos,
            matriz_freq,
            regime_global_str,
            resumo_k_local,
        )

        if df_cand.empty:
            return None

        # Recalcula AIQ com peso customizado
        novas_aiq = []
        for _, row in df_cand.iterrows():
            serie = row["series"]
            _, qds_tmp, div_tmp = calcular_aiq_candidato(
                serie,
                matriz_freq,
                peso_qds=peso_qds,
                peso_div=1.0 - peso_qds,
            )
            novas_aiq.append(peso_qds * qds_tmp + (1 - peso_qds) * div_tmp)

        df_cand["AIQ"] = novas_aiq
        df_cand = df_cand.sort_values(["AIQ", "QDS"], ascending=[False, False]).reset_index(drop=True)

        melhor = df_cand.iloc[0]
        return melhor["series"]

    except Exception:
        return None


# ------------------------------------------------------------
# PAINEL — Replay LIGHT
# ------------------------------------------------------------

if painel == "💡 Replay LIGHT":
    st.markdown("## 💡 Replay LIGHT")
    st.markdown(
        """
        Executa o núcleo V14-FLEX ULTRA retroativamente, apenas na **última** série,
        usando os mesmos parâmetros que você definirá aqui.

        🔹 Seguro  
        🔹 Rápido  
        🔹 Funciona mesmo com estradas gigantes (5k+)
        """
    )

    df_limpo = st.session_state.get("df_limpo", None)
    if df_limpo is None or df_limpo.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    n_series_hist = len(df_limpo)
    idx_alvo = n_series_hist  # última série

    st.info(f"Última série = índice {idx_alvo}")

    col_a, col_b = st.columns(2)

    with col_a:
        janela_back = st.slider(
            "Tamanho da janela para trás:",
            min_value=10,
            max_value=min(300, n_series_hist - 1),
            value=min(60, n_series_hist - 1),
            step=5,
        )

    with col_b:
        n_candidatos = st.slider(
            "Tamanho do leque base:",
            min_value=10,
            max_value=200,
            value=60,
            step=5,
        )

    peso_qds = st.slider(
        "Peso do QDS (AIQ):",
        min_value=0.1,
        max_value=0.9,
        value=0.6,
        step=0.05,
    )

    seed_base = st.number_input(
        "Seed:",
        min_value=1,
        max_value=999999,
        value=12345,
    )

    if st.button("▶️ Executar Replay LIGHT"):
        previsao = replay_calcular_previsao_v14(
            df_limpo=df_limpo,
            idx_alvo=idx_alvo,
            janela_back=janela_back,
            n_candidatos=n_candidatos,
            seed_base=seed_base,
            peso_qds=peso_qds,
        )

        if previsao is None:
            st.error("Falha ao calcular previsão LIGHT.")
        else:
            st.success("Replay LIGHT executado.")
            st.code(" ".join(str(x) for x in previsao), language="text")


# ------------------------------------------------------------
# PAINEL — Replay ULTRA
# ------------------------------------------------------------

if painel == "📅 Replay ULTRA":
    st.markdown("## 📅 Replay ULTRA")
    st.markdown(
        """
        Executa o pipeline V14-FLEX ULTRA **janela por janela**, retrocedendo 
        por todo o histórico.

        ⚠️ **Atenção para estradas grandes (>3000 séries):**

        Este painel é automaticamente **DESABILITADO** para rodar integralmente,
        evitando travamento zumbi no Streamlit Cloud.

        Você ainda pode:
        - visualizar controles
        - rodar uma **amostra parcial reduzida**
        - guardar estatísticas locais
        """
    )

    df_limpo = st.session_state.get("df_limpo")
    if df_limpo is None or df_limpo.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    n_hist = len(df_limpo)

    # Informação ao usuário
    if n_hist > LIMITE_REPLAY_HIST:
        st.warning(
            f"""
            🚫 O histórico possui **{n_hist} séries** — acima do limite seguro (**{LIMITE_REPLAY_HIST}**).
            O Replay ULTRA completo está **DESABILITADO** para evitar travamento.
            """
        )
        permitir_execucao = False
    else:
        permitir_execucao = True

    janela_back = st.slider(
        "Tamanho da janela (para trás):",
        min_value=10,
        max_value=min(300, n_hist-1),
        value=min(60, n_hist-1),
        step=5,
    )

    n_candidatos = st.slider(
        "Tamanho do leque base:",
        min_value=10,
        max_value=200,
        value=60,
        step=5,
    )

    seed_base = st.number_input(
        "Seed:",
        min_value=1,
        max_value=999999,
        value=12345,
    )

    peso_qds = st.slider(
        "Peso do QDS (AIQ):",
        min_value=0.1,
        max_value=0.9,
        value=0.6,
        step=0.05,
    )

    # Execução parcial segura
    st.markdown("### Execução parcial (segura)")

    tamanho_amostra = st.slider(
        "Número de séries finais para testar:",
        min_value=5,
        max_value=min(300, n_hist),
        value=min(50, n_hist),
        step=5,
    )

    if st.button("▶️ Rodar Replay ULTRA (parcial)"):
        acertos = 0
        total = 0

        limite_inicial = max(2, n_hist - tamanho_amostra)

        progress = st.progress(0.0)

        for i in range(limite_inicial, n_hist):
            progress.progress((i - limite_inicial) / tamanho_amostra)

            previsao = replay_calcular_previsao_v14(
                df_limpo=df_limpo,
                idx_alvo=i,
                janela_back=janela_back,
                n_candidatos=n_candidatos,
                seed_base=seed_base,
                peso_qds=peso_qds,
            )

            if previsao is None:
                continue

            real = df_limpo.iloc[i - 1]
            valores_real = [int(real[c]) for c in get_passenger_cols(df_limpo)]

            if previsao == valores_real:
                acertos += 1

            total += 1

        st.success("Replay ULTRA parcial concluído.")

        st.metric("Acertos", acertos)
        st.metric("Total avaliado", total)
        if total > 0:
            st.metric("Taxa de acertos (%)", f"{(acertos/total)*100:.2f}%")
        else:
            st.metric("Taxa de acertos (%)", "0.00%")

    # Execução completa (desabilitada p/ histórico grande)
    if not permitir_execucao:
        st.info("Replay ULTRA completo está desabilitado para este histórico.")
    else:
        st.markdown("### Execução completa (somente para estradas menores)")
        if st.button("▶️ Rodar Replay ULTRA (completo)"):
            st.warning("Rodando completo — pode demorar…")

            acertos = 0
            total = 0
            progress = st.progress(0.0)

            for i in range(2, n_hist + 1):
                progress.progress((i - 1) / n_hist)

                previsao = replay_calcular_previsao_v14(
                    df_limpo=df_limpo,
                    idx_alvo=i,
                    janela_back=janela_back,
                    n_candidatos=n_candidatos,
                    seed_base=seed_base,
                    peso_qds=peso_qds,
                )

                if previsao is None:
                    continue

                real = df_limpo.iloc[i - 1]
                valores_real = [int(real[c]) for c in get_passenger_cols(df_limpo)]

                if previsao == valores_real:
                    acertos += 1

                total += 1

            st.success("Replay ULTRA completo finalizado.")
            st.metric("Acertos", acertos)
            st.metric("Total avaliado", total)
            if total > 0:
                st.metric("Taxa de acertos (%)", f"{(acertos/total)*100:.2f}%")


# ------------------------------------------------------------
# PAINEL — Replay ULTRA Unitário
# ------------------------------------------------------------

if painel == "🎯 Replay ULTRA Unitário":
    st.markdown("## 🎯 Replay ULTRA Unitário")

    df_limpo = st.session_state.get("df_limpo", None)
    if df_limpo is None or df_limpo.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    n_hist = len(df_limpo)

    st.markdown(
        "Executa o pipeline completo apenas **para 1 série específica**, "
        "seguindo o mesmo V14-FLEX ULTRA."
    )

    idx_alvo = st.number_input(
        "Índice alvo:",
        min_value=1,
        max_value=n_hist,
        value=n_hist,
        step=1,
    )

    janela_back = st.slider(
        "Janela para trás:",
        min_value=10,
        max_value=min(300, n_hist-1),
        value=min(60, n_hist-1),
        step=5,
    )

    n_candidatos = st.slider(
        "Leque (n candidatos):",
        min_value=10,
        max_value=200,
        value=60,
        step=5,
    )

    seed_base = st.number_input(
        "Seed:",
        min_value=1,
        max_value=999999,
        value=12345,
    )

    peso_qds = st.slider(
        "Peso QDS:",
        min_value=0.1,
        max_value=0.9,
        value=0.6,
        step=0.05,
    )

    if st.button("▶️ Executar Unitário"):
        previsao = replay_calcular_previsao_v14(
            df_limpo=df_limpo,
            idx_alvo=int(idx_alvo),
            janela_back=int(janela_back),
            n_candidatos=int(n_candidatos),
            seed_base=int(seed_base),
            peso_qds=float(peso_qds),
        )

        if previsao is None:
            st.error("Falha ao processar a janela.")
        else:
            st.success("Replay ULTRA Unitário:")
            st.code(" ".join(str(x) for x in previsao), language="text")


# ------------------------------------------------------------
# PAINEL — Monitor de Risco (k & k*)
# ------------------------------------------------------------

if painel == "🚨 Monitor de Risco (k & k*)":
    st.markdown("## 🚨 Monitor de Risco — k & k*")

    df_limpo = st.session_state.get("df_limpo", None)
    if df_limpo is None or df_limpo.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    resumo_k_global: Optional[ResumoK] = st.session_state.get("resumo_k_global")
    resumo_estrada: Optional[ResumoEstrada] = st.session_state.get("resumo_estrada")

    if resumo_k_global is None:
        st.error("k* global não foi calculado.")
        st.stop()

    st.markdown("### 🔭 Estado global")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("k atual", resumo_k_global.k_atual)
        st.metric("k*", f"{resumo_k_global.k_star*100:.1f}%")

    with col2:
        st.metric("Regime (k*)", resumo_k_global.regime_local)

    with col3:
        estado_label = {
            "estavel": "🟢 Ambiente estável",
            "atencao": "🟡 Pré-ruptura residual",
            "critico": "🔴 Ambiente crítico",
        }.get(resumo_k_global.estado_k, "⚪ Desconhecido")
        st.write(estado_label)

    st.markdown("### 🔍 Análise por janela local (unitária)")

    idx_alvo = st.number_input(
        "Índice alvo:",
        min_value=1,
        max_value=len(df_limpo),
        value=len(df_limpo),
        step=1,
    )

    janela_k = st.slider(
        "Janela (para cálculo do k* local):",
        min_value=5,
        max_value=200,
        value=40,
        step=5,
    )

    df_j = extrair_janela_hist(df_limpo, int(idx_alvo), back=int(janela_k))
    if df_j.empty:
        st.error("Janela vazia.")
        st.stop()

    resumo_k_local = calcular_k_star(df_j, janela=int(janela_k))

    st.markdown("### 🔭 k* Local")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("k atual", resumo_k_local.k_atual)
    with col2:
        st.metric("k*", f"{resumo_k_local.k_star*100:.1f}%")
    with col3:
        st.metric("Regime", resumo_k_local.regime_local)

    estado_label_local = {
        "estavel": "🟢 Ambiente estável",
        "atencao": "🟡 Pré-ruptura residual",
        "critico": "🔴 Ambiente crítico",
    }.get(resumo_k_local.estado_k, "⚪ Desconhecido")

    st.write(estado_label_local)

    if st.session_state.get("mostrar_debug", False):
        st.markdown("#### 🐞 DEBUG — k local")
        st.dataframe(df_j.head(50), use_container_width=True)
# ============================================================
# PARTE 4/6 — Testes de Confiabilidade REAL • Ruído Condicional (V15)
# ============================================================

# ------------------------------------------------------------
# FUNÇÕES AUXILIARES — BACKTEST REAL / MONTE CARLO REAL
# ------------------------------------------------------------

def executar_backtest_parcial_real(
    df_limpo: pd.DataFrame,
    janela_back: int,
    n_candidatos: int,
    seed_base: int,
    peso_qds: float,
    n_janelas_max: int,
) -> Optional[ResumoBacktest]:
    """
    Backtest REAL simplificado e controlado:
    - Reaplica o pipeline V14-FLEX ULTRA nas últimas N janelas.
    - Compara a previsão com o real da série.
    - Retorna ResumoBacktest.

    ❗ Protegido para estradas grandes:
       - Em vez de percorrer todas as ~5k séries, utiliza apenas
         as últimas `n_janelas_max` janelas.
    """
    if df_limpo is None or df_limpo.empty:
        return None

    n_hist = len(df_limpo)
    if n_hist < 3:
        return None

    inicio = max(2, n_hist - n_janelas_max)
    acertos_lista: List[int] = []

    for idx in range(inicio, n_hist + 1):
        previsao = replay_calcular_previsao_v14(
            df_limpo=df_limpo,
            idx_alvo=idx,
            janela_back=janela_back,
            n_candidatos=n_candidatos,
            seed_base=seed_base,
            peso_qds=peso_qds,
        )
        if previsao is None:
            continue

        real = df_limpo.iloc[idx - 1]
        valores_real = [int(real[c]) for c in get_passenger_cols(df_limpo)]

        # Contagem de acertos exatos (posição a posição)
        acertos = sum(1 for a, b in zip(previsao, valores_real) if a == b)
        acertos_lista.append(acertos)

    if not acertos_lista:
        return None

    # Cada janela corresponde a 1 "série" de avaliação
    resumo = calcular_resumo_backtest(acertos_lista, n_series_por_janela=1)
    return resumo


def executar_monte_carlo_real(
    df_limpo: pd.DataFrame,
    n_simulacoes: int,
    n_series_amostra: int,
) -> Optional[ResumoMonteCarlo]:
    """
    Monte Carlo REAL com amostragem da estrada:

    - Seleciona aleatoriamente N séries do histórico
    - Gera previsões aleatórias (0..60) com mesmo nº de passageiros
    - Compara com a série real, contando acertos posição a posição
    - Gera uma matriz de acertos [simulação x série] e resume

    ❗ Protegido para estradas grandes através do parâmetro n_series_amostra.
    """
    if df_limpo is None or df_limpo.empty:
        return None

    rng = random.Random(4242)
    cols_pass = get_passenger_cols(df_limpo)
    n_hist = len(df_limpo)
    n_series_amostra = min(n_series_amostra, n_hist)

    matriz_acertos: List[List[int]] = []

    for _ in range(n_simulacoes):
        # Amostra aleatória de índices de séries
        indices = rng.sample(range(n_hist), n_series_amostra)
        acertos_sim: List[int] = []

        for idx in indices:
            linha = df_limpo.iloc[idx]
            reais = [int(linha[c]) for c in cols_pass]

            previsao_aleatoria = [rng.randint(0, 60) for _ in cols_pass]
            acertos = sum(1 for a, b in zip(previsao_aleatoria, reais) if a == b)
            acertos_sim.append(acertos)

        matriz_acertos.append(acertos_sim)

    return calcular_resumo_monte_carlo(matriz_acertos)


# ------------------------------------------------------------
# PAINEL — Testes de Confiabilidade REAL
# ------------------------------------------------------------

if painel == "🧪 Testes de Confiabilidade REAL":
    st.markdown("## 🧪 Testes de Confiabilidade REAL")
    st.markdown(
        """
        Este painel consolida os **testes de robustez** do V15.5:

        - Backtest REAL (com proteção para estradas grandes)
        - Monte Carlo REAL
        """
    )

    df_limpo = st.session_state.get("df_limpo", None)
    if df_limpo is None or df_limpo.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    n_hist = len(df_limpo)
    st.info(f"Total de séries no histórico: **{n_hist}**.")

    # --------------------------------------------------------
    # BACKTEST REAL (PARCIAL / PROTEGIDO)
    # --------------------------------------------------------

    st.markdown("### 🔁 Backtest REAL (parcial e seguro)")

    col_bt1, col_bt2 = st.columns(2)
    with col_bt1:
        janela_back_bt = st.slider(
            "Janela para trás (Backtest):",
            min_value=10,
            max_value=min(300, n_hist - 1),
            value=min(60, n_hist - 1),
            step=5,
        )

    with col_bt2:
        n_janelas_bt = st.slider(
            "Quantidade de janelas (últimas séries avaliadas):",
            min_value=10,
            max_value=min(500, n_hist - 1),
            value=min(200, n_hist - 1),
            step=10,
            help="Backtest parcial sobre as últimas N séries (seguro para 5k+ linhas).",
        )

    col_bt3, col_bt4 = st.columns(2)
    with col_bt3:
        n_candidatos_bt = st.slider(
            "Leque (n candidatos / janela):",
            min_value=10,
            max_value=200,
            value=40,
            step=5,
        )

    with col_bt4:
        peso_qds_bt = st.slider(
            "Peso do QDS no AIQ (Backtest):",
            min_value=0.1,
            max_value=0.9,
            value=0.6,
            step=0.05,
        )

    seed_bt = st.number_input(
        "Seed base (Backtest):",
        min_value=1,
        max_value=999999,
        value=2025,
    )

    if st.button("▶️ Rodar Backtest REAL (parcial)"):
        with st.spinner("Executando Backtest REAL parcial..."):
            resumo_bt = executar_backtest_parcial_real(
                df_limpo=df_limpo,
                janela_back=int(janela_back_bt),
                n_candidatos=int(n_candidatos_bt),
                seed_base=int(seed_bt),
                peso_qds=float(peso_qds_bt),
                n_janelas_max=int(n_janelas_bt),
            )

        if resumo_bt is None:
            st.error("Não foi possível calcular o Backtest REAL.")
        else:
            st.success("Backtest REAL parcial concluído.")
            st.session_state["resumo_backtest"] = resumo_bt

            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            with col_r1:
                st.metric("Janelas avaliadas", resumo_bt.n_janelas)
            with col_r2:
                st.metric("Acertos totais", resumo_bt.acertos_totais)
            with col_r3:
                st.metric("Média de acertos/série", f"{resumo_bt.acertos_por_serie:.3f}")
            with col_r4:
                st.metric("Hit rate (%)", f"{resumo_bt.hit_rate*100:.2f}%")

    # --------------------------------------------------------
    # MONTE CARLO REAL
    # --------------------------------------------------------

    st.markdown("### 🎲 Monte Carlo REAL")

    col_mc1, col_mc2 = st.columns(2)
    with col_mc1:
        n_sim_mc = st.slider(
            "Número de simulações:",
            min_value=100,
            max_value=5000,
            value=800,
            step=100,
        )
    with col_mc2:
        n_series_mc = st.slider(
            "Séries amostradas por simulação:",
            min_value=50,
            max_value=min(600, n_hist),
            value=min(300, n_hist),
            step=50,
            help="Quantidade de séries reais da estrada usadas em cada simulação.",
        )

    if st.button("▶️ Rodar Monte Carlo REAL"):
        with st.spinner("Executando Monte Carlo REAL..."):
            resumo_mc = executar_monte_carlo_real(
                df_limpo=df_limpo,
                n_simulacoes=int(n_sim_mc),
                n_series_amostra=int(n_series_mc),
            )

        if resumo_mc is None:
            st.error("Não foi possível calcular o Monte Carlo REAL.")
        else:
            st.success("Monte Carlo REAL concluído.")
            st.session_state["resumo_montecarlo"] = resumo_mc

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Simulações", resumo_mc.n_simulacoes)
            with col_m2:
                st.metric("Média de acertos", f"{resumo_mc.media_acertos:.4f}")
            with col_m3:
                st.metric("Desvio (acertos)", f"{resumo_mc.desvio_acertos:.4f}")
            with col_m4:
                st.metric(
                    "Melhor média de simulação", f"{resumo_mc.melhor_serie_media:.4f}"
                )


# ------------------------------------------------------------
# FUNÇÕES AUXILIARES — RUÍDO CONDICIONAL (V15)
# ------------------------------------------------------------

def aplicar_ruido_condicional(
    df: pd.DataFrame,
    pct_alvo: float,
    amplitude_base: int,
    seed: int,
) -> Tuple[pd.DataFrame, float]:
    """
    Aplica ruído condicional por célula, baseado em k:

    - Seleciona pct_alvo% das células (n1..nN)
    - Se k == 0 → ruído mais suave (amplitude_base // 2)
    - Se k > 0 → ruído total (amplitude_base)
    - Garante que os valores ficam em [0, 60]
    - Retorna:
        - novo DataFrame
        - % de pontos efetivamente ajustados (pode ser <= pct_alvo)
    """
    df2 = df.copy()
    rng = random.Random(seed)

    cols = get_passenger_cols(df2)
    if not cols:
        return df2, 0.0

    n_lin = len(df2)
    total_cells = n_lin * len(cols)
    if total_cells == 0:
        return df2, 0.0

    n_alvo = int(total_cells * (pct_alvo / 100.0))
    indices = [(i, j) for i in range(n_lin) for j in range(len(cols))]
    rng.shuffle(indices)
    indices_sel = indices[:n_alvo]

    ajustes = 0

    for (i, j) in indices_sel:
        idx_row = df2.index[i]
        col = cols[j]

        val = int(df2.at[idx_row, col])
        if val < 0:
            continue

        k_val = int(df2.at[idx_row, "k"])
        if k_val <= 0:
            amp_efetiva = max(1, amplitude_base // 2)
        else:
            amp_efetiva = amplitude_base

        delta = rng.randint(-amp_efetiva, amp_efetiva)
        novo = max(0, min(60, val + delta))
        if novo != val:
            df2.at[idx_row, col] = novo
            ajustes += 1

    pct_ajuste_real = (ajustes / total_cells) * 100.0
    return df2, pct_ajuste_real


# ------------------------------------------------------------
# PAINEL — Ruído Condicional (V15)
# ------------------------------------------------------------

if painel == "📊 Ruído Condicional (V15)":
    st.markdown("## 📊 Ruído Condicional (V15)")
    st.markdown(
        """
        Este painel aplica **Ruído A** e **Ruído B** sobre a estrada, de forma
        **condicional ao k**:

        - Estrada A: ruído inicial
        - Estrada B: ruído final
        - Ruído mais intenso onde há k>0 (guardas acertando), mais suave onde k=0
        """
    )

    df_limpo = st.session_state.get("df_limpo", None)
    if df_limpo is None or df_limpo.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    n_hist = len(df_limpo)
    st.info(f"Total de séries no histórico: **{n_hist}**.")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        pct_ruido_a = st.slider(
            "Percentual de células alvo — Ruído A:",
            min_value=1.0,
            max_value=50.0,
            value=15.0,
            step=1.0,
        )
        amp_a = st.slider(
            "Amplitude Ruído A:",
            min_value=1,
            max_value=20,
            value=8,
            step=1,
        )
    with col_r2:
        pct_ruido_b = st.slider(
            "Percentual de células alvo — Ruído B:",
            min_value=1.0,
            max_value=80.0,
            value=30.0,
            step=1.0,
        )
        amp_b = st.slider(
            "Amplitude Ruído B:",
            min_value=1,
            max_value=30,
            value=12,
            step=1,
        )

    seed_ruido = st.number_input(
        "Seed base (Ruído):",
        min_value=1,
        max_value=999999,
        value=777,
    )

    if st.button("▶️ Aplicar Ruído Condicional A/B"):
        with st.spinner("Aplicando Ruído A..."):
            df_ra, pct_aj_a = aplicar_ruido_condicional(
                df=df_limpo,
                pct_alvo=float(pct_ruido_a),
                amplitude_base=int(amp_a),
                seed=int(seed_ruido),
            )

        with st.spinner("Aplicando Ruído B sobre a Estrada A..."):
            df_rb, pct_aj_b = aplicar_ruido_condicional(
                df=df_ra,
                pct_alvo=float(pct_ruido_b),
                amplitude_base=int(amp_b),
                seed=int(seed_ruido + 17),
            )

        st.session_state["df_ruido_a"] = df_ra
        st.session_state["df_ruido_b"] = df_rb

        pct_medio_aj = (pct_aj_a + pct_aj_b) / 2.0

        resumo_ruido = calcular_resumo_ruido(
            ruido_inicial=float(amp_a),
            ruido_final=float(amp_b),
            pct_pontos_ajustados=float(pct_medio_aj),
        )
        st.session_state["resumo_ruido"] = resumo_ruido

        st.success("Ruído Condicional A/B aplicado com sucesso.")

        col_x1, col_x2, col_x3 = st.columns(3)
        with col_x1:
            st.metric("Amplitude A", amp_a)
            st.metric("% células alvo (A)", f"{pct_ruido_a:.1f}%")
        with col_x2:
            st.metric("Amplitude B", amp_b)
            st.metric("% células alvo (B)", f"{pct_ruido_b:.1f}%")
        with col_x3:
            st.metric(
                "% pontos ajustados (médio)",
                f"{pct_medio_aj:.2f}%",
            )

        st.markdown("### 🔎 Amostra da Estrada B (pós-ruído)")
        st.dataframe(df_rb.head(50), use_container_width=True)

        if st.session_state.get("mostrar_debug", False):
            st.markdown("#### 🐞 DEBUG — Estrada A (pós-Ruído A)")
            st.dataframe(df_ra.head(30), use_container_width=True)
    else:
        st.info(
            "Configure os parâmetros de ruído e clique em "
            "'Aplicar Ruído Condicional A/B'."
        )
# ============================================================
# PARTE 5/6 — Modo TURBO++ ULTRA ANTI-RUÍDO (V15)
# ============================================================

# ------------------------------------------------------------
# FUNÇÕES AUXILIARES DO TURBO++ ULTRA
# ------------------------------------------------------------

def gerar_matriz_freq_para_df(df: pd.DataFrame, janela_back: int, idx_alvo: int) -> Dict[str, Dict[int, float]]:
    """
    Função interna reutilizável:
    - extrai a janela de df_limpo, df_ruido_a ou df_ruido_b
    - calcula matriz de frequências
    """
    df_j = extrair_janela_hist(df, idx_alvo=idx_alvo, back=janela_back)
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
    Usado para gerar Leque A, Leque B e Leque Misto.
    """
    df_j = extrair_janela_hist(df_base, idx_alvo=idx_alvo, back=janela_back)
    if df_j.empty:
        return pd.DataFrame()

    matriz_freq = calcular_matriz_frequencia(df_j)
    if not matriz_freq:
        return pd.DataFrame()

    # Geração dos candidatos
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

    # Recalcular AIQ com pesos selecionados
    novas_aiq = []
    for _, row in df_cand.iterrows():
        serie = row["series"]
        _, qds_tmp, div_tmp = calcular_aiq_candidato(
            serie,
            matriz_freq,
            peso_qds=peso_qds,
            peso_div=1.0 - peso_qds,
        )
        novas_aiq.append(peso_qds * qds_tmp + (1 - peso_qds) * div_tmp)

    df_cand["AIQ"] = novas_aiq
    df_cand = df_cand.sort_values(["AIQ", "QDS"], ascending=[False, False]).reset_index(drop=True)
    return df_cand


def unir_leques_v15(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """
    Une Leque A + Leque B, eliminando duplicidades e
    recalculando a ordenação final por AIQ → QDS.
    """
    if df_a is None or df_a.empty:
        return df_b.copy()
    if df_b is None or df_b.empty:
        return df_a.copy()

    # Combina
    df_mix = pd.concat([df_a, df_b], ignore_index=True)

    # Remove duplicidades pelo conteúdo da série
    df_mix["serie_str"] = df_mix["series"].apply(lambda x: tuple(x))
    df_mix = df_mix.drop_duplicates(subset=["serie_str"])
    df_mix = df_mix.drop(columns=["serie_str"])

    # Reordena
    df_mix = df_mix.sort_values(["AIQ", "QDS"], ascending=[False, False]).reset_index(drop=True)
    return df_mix


# ------------------------------------------------------------
# PAINEL — Modo TURBO++ ULTRA ANTI-RUÍDO (V15)
# ------------------------------------------------------------

if painel == "🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15)":
    st.markdown("## 🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15)")
    st.markdown(
        """
        Este é o **motor definitivo** do Predict Cars V15.5-HÍBRIDO.

        Aqui o sistema:
        - utiliza a Estrada Original
        - utiliza Estrada A (pós-Ruído A)
        - utiliza Estrada B (pós-Ruído B)
        - gera leques A/B
        - mescla, reordena, compara, calibra
        - aplica AIQ-HÍBRIDO
        - calcula o candidato mais resiliente
        - produz a **PREVISÃO FINAL V15.5**
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
            "Aplique o Ruído Condicional no painel anterior "
            "(Estradas A/B ainda não existem)."
        )
        st.stop()

    resumo_estrada = st.session_state.get("resumo_estrada")
    resumo_k_global = st.session_state.get("resumo_k_global")
    regime_global_str = resumo_estrada.regime_global if resumo_estrada else "desconhecido"

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

    # k local
    df_j = extrair_janela_hist(df_limpo, idx_alvo, back=janela_back)
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

            # --- LEQUE A (Ruído A) --------------------------
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

            # --- LEQUE B (Ruído B) --------------------------
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

            # --- MISTO -------------------------------------
            df_mix = unir_leques_v15(dfA, dfB)

            if df_mix.empty:
                st.error("Falha ao montar Leque MISTO (A+B).")
                st.stop()

            # Previsão Final
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
            dfA_view["series"] = dfA_view["series"].apply(
                lambda x: " ".join(str(v) for v in x)
            )
            st.dataframe(dfA_view.head(20))

            st.markdown("#### 🐞 DEBUG — Leque B")
            dfB_view = dfB.copy()
            dfB_view["series"] = dfB_view["series"].apply(
                lambda x: " ".join(str(v) for v in x)
            )
            st.dataframe(dfB_view.head(20))

            st.markdown("#### 🐞 DEBUG — Leque MISTO")
            dfM_view = df_mix.copy()
            dfM_view["series"] = dfM_view["series"].apply(
                lambda x: " ".join(str(v) for v in x)
            )
            st.dataframe(dfM_view.head(20))



# ============================================================
# PARTE 6/6 — Relatório Final — AIQ Bridge (para ChatGPT)
# ============================================================

# ------------------------------------------------------------
# FUNÇÕES AUXILIARES DO RELATÓRIO
# ------------------------------------------------------------

def formatar_percentual(v: float) -> str:
    return f"{v*100:.2f}%" if 0 <= v <= 1 else f"{v:.2f}%"


def gerar_expectativa_acertos(
    regime_local: str,
    k_star: float,
    qds_medio: float,
) -> str:
    """
    Define a expectativa de acertos da previsão final com base em:
    - regime local (k*)
    - sensibilidade do sistema
    - QDS médio
    """
    # Heurística do V15.5
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
    ajuste_qds = qds_medio

    expectativa = base * (0.7 + 0.3 * ajuste_qds) * (0.6 + 0.4 * ajuste_k)
    return f"{expectativa:.2f} acertos (esperados)"


# ------------------------------------------------------------
# PAINEL — Relatório Final — AIQ Bridge
# ------------------------------------------------------------

if painel == "📄 Relatório Final — AIQ Bridge (para ChatGPT)":
    st.markdown("## 📄 Relatório Final — AIQ Bridge (para ChatGPT)")
    st.markdown(
        """
        Este é o **relatório oficial do Predict Cars V15.5-HÍBRIDO**,
        pronto para ser copiado e colado diretamente no ChatGPT.
        """
    )

    df_limpo = st.session_state.get("df_limpo")
    resumo_estrada = st.session_state.get("resumo_estrada")
    resumo_k_global = st.session_state.get("resumo_k_global")
    resumo_qds: Optional[ResumoQDS] = None

    if df_limpo is None or df_limpo.empty:
        st.error("Histórico não carregado.")
        st.stop()

    # Coleta QDS global acumulado
    lista_qds = st.session_state.get("lista_qds", [])
    if lista_qds:
        resumo_qds = calcular_resumo_qds(lista_qds)

    resumo_ruido = st.session_state.get("resumo_ruido")
    resumo_backtest = st.session_state.get("resumo_backtest")
    resumo_montecarlo = st.session_state.get("resumo_montecarlo")
    previsao_turbo = st.session_state.get("previsao_turbo_ultra")

    # Construção do relatório
    relatorio = []

    relatorio.append("# 🔵 Predict Cars V15.5 — AIQ Bridge Report\n")

    # --------------------------------------------------------
    # Estrada Global
    # --------------------------------------------------------

    relatorio.append("## 🛣️ Estrada — Resumo Global\n")

    if resumo_estrada:
        relatorio.append(f"- Total de séries: **{resumo_estrada.n_series}**")
        relatorio.append(f"- Passageiros por série: **{resumo_estrada.n_passageiros}**")
        relatorio.append(f"- Faixas de valores: **{resumo_estrada.min_val} — {resumo_estrada.max_val}**")
        relatorio.append(f"- Média global (n1..nN): **{resumo_estrada.media:.2f}**")
        relatorio.append(f"- Desvio-padrão: **{resumo_estrada.desvio:.2f}**")
        relatorio.append(f"- k médio: **{resumo_estrada.k_medio:.3f}**")
        relatorio.append(f"- k máximo: **{resumo_estrada.k_max}**")
        relatorio.append(f"- Regime global (barômetro): **{resumo_estrada.regime_global}**")
    else:
        relatorio.append("- Estrada não disponível.")

    relatorio.append("")

    # --------------------------------------------------------
    # k* GLOBAL
    # --------------------------------------------------------

    relatorio.append("## 🔭 k* — Sentinela da Estrada\n")

    if resumo_k_global:
        estado_str = {
            "estavel": "🟢 Estável",
            "atencao": "🟡 Pré-ruptura",
            "critico": "🔴 Crítico",
        }.get(resumo_k_global.estado_k, "⚪ Indeterminado")

        relatorio.append(f"- k atual: **{resumo_k_global.k_atual}**")
        relatorio.append(f"- k*: **{resumo_k_global.k_star*100:.2f}%**")
        relatorio.append(f"- Regime local (k*): **{resumo_k_global.regime_local}**")
        relatorio.append(f"- Estado de risco: {estado_str}")
    else:
        relatorio.append("- k* não disponível.")

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
        relatorio.append("- QDS não disponível.")

    relatorio.append("")

    # --------------------------------------------------------
    # Ruído A/B
    # --------------------------------------------------------

    relatorio.append("## 🎛️ Ruído — Estradas A/B\n")

    if resumo_ruido:
        relatorio.append(f"- Ruído inicial (A): **{resumo_ruido.ruido_inicial}**")
        relatorio.append(f"- Ruído final (B): **{resumo_ruido.ruido_final}**")
        relatorio.append(
            f"- % de pontos ajustados: **{resumo_ruido.pct_pontos_ajustados:.2f}%**"
        )
    else:
        relatorio.append("- Ruído A/B não aplicado.")

    relatorio.append("")

    # --------------------------------------------------------
    # Backtest REAL
    # --------------------------------------------------------

    relatorio.append("## 🔁 Backtest REAL (parcial)\n")

    if resumo_backtest:
        relatorio.append(f"- Janelas avaliadas: **{resumo_backtest.n_janelas}**")
        relatorio.append(f"- Acertos totais: **{resumo_backtest.acertos_totais}**")
        relatorio.append(
            f"- Média de acertos por série: **{resumo_backtest.acertos_por_serie:.4f}**"
        )
        relatorio.append(
            f"- Hit rate: **{resumo_backtest.hit_rate*100:.2f}%**"
        )
    else:
        relatorio.append("- Backtest REAL não executado.")

    relatorio.append("")

    # --------------------------------------------------------
    # Monte Carlo REAL
    # --------------------------------------------------------

    relatorio.append("## 🎲 Monte Carlo REAL\n")

    if resumo_montecarlo:
        relatorio.append(f"- Simulações: **{resumo_montecarlo.n_simulacoes}**")
        relatorio.append(f"- Média de acertos: **{resumo_montecarlo.media_acertos:.4f}**")
        relatorio.append(f"- Desvio dos acertos: **{resumo_montecarlo.desvio_acertos:.4f}**")
        relatorio.append(
            f"- Melhor média de uma simulação: **{resumo_montecarlo.melhor_serie_media:.4f}**"
        )
    else:
        relatorio.append("- Monte Carlo REAL não executado.")

    relatorio.append("")

    # --------------------------------------------------------
    # PREVISÃO FINAL V15.5 (TURBO++ ULTRA)
    # --------------------------------------------------------

    relatorio.append("## 🎯 Previsão Final — TURBO++ ULTRA (V15.5)\n")

    if previsao_turbo:
        relatorio.append(
            f"- Série prevista: **{' '.join(str(x) for x in previsao_turbo)}**"
        )
    else:
        relatorio.append("- Previsão final ainda não gerada.")

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
        relatorio.append(f"- Expectativa projetada: **{exp}**")
    else:
        relatorio.append("- Insuficiente para estimar.")

    relatorio.append("\n---\n")
    relatorio.append("### ✔ Relatório pronto para enviar ao ChatGPT.\n")

    # Exibição final
    texto_relatorio = "\n".join(relatorio)
    st.text_area(
        "📄 Copie o relatório completo abaixo:",
        value=texto_relatorio,
        height=700,
    )
     dfM_view["series"] = dfM_view["series"].apply(lambda x: " ".join(str(v) for v in x))
            st.dataframe(dfM_view.head(20))
