# ============================================================
# Predict Cars V15.5-HÍBRIDO
# Núcleo V14-FLEX ULTRA + k* + Ruído Condicional + QDS REAL +
# Backtest REAL + Monte Carlo REAL + AIQ Bridge (para ChatGPT)
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
# CONFIGURAÇÃO BÁSICA DO APP
# ------------------------------------------------------------

st.set_page_config(
    page_title="Predict Cars V15.5-HÍBRIDO",
    page_icon="🚗",
    layout="wide",
)

# ------------------------------------------------------------
# CONSTANTES / CATEGORIAS DE QUALIDADE / REGIMES
# ------------------------------------------------------------

QDS_LABELS = ["PREMIUM", "BOM", "REGULAR", "RUIM"]

QDS_THRESHOLDS = {
    "PREMIUM": 0.85,
    "BOM": 0.70,
    "REGULAR": 0.50,
    "RUIM": 0.0,
}

REGIMES = ["Ultra Estável", "Estável", "Transição", "Turbulento", "Crítico"]


# ------------------------------------------------------------
# DATACLASSES PARA RESUMOS (RUA PRINCIPAL)
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
# FUNÇÕES UTILITÁRIAS GERAIS
# ------------------------------------------------------------

def _detect_separator(sample_line: str) -> str:
    """
    Detecta separador provável entre ; , \t ou espaço.
    """
    for sep in [";", ",", "\t", " "]:
        if sep in sample_line:
            return sep
    return ";"


def parse_historico_text(raw: str) -> pd.DataFrame:
    """
    Parser FLEX ULTRA para texto colado.
    Aceita linhas no estilo:

    C1;41;5;4;52;30;33;0
    C2;9;39;37;49;43;41;1
    ...

    Ou sem ID da série:

    41;5;4;52;30;33;0
    9;39;37;49;43;41;1
    """
    linhas = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not linhas:
        return pd.DataFrame()

    # Detecta separador na primeira linha não vazia
    sep = _detect_separator(linhas[0])

    registros = []
    for ln in linhas:
        partes = [p.strip() for p in ln.split(sep) if p.strip()]
        if not partes:
            continue

        # Se primeiro campo começa com "C" ou algo não numérico → ID da série
        if not partes[0].isdigit():
            serie_id = partes[0]
            nums = partes[1:]
        else:
            serie_id = None
            nums = partes

        try:
            nums_int = [int(x) for x in nums]
        except ValueError:
            # Alguma linha com lixo → ignora
            continue

        if len(nums_int) < 2:
            # Precisa de pelo menos 1 passageiro + k
            continue

        # Última posição = k (rótulo)
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

    # Normaliza em DataFrame com colunas n1..nN + k
    max_p = max(len(r["passageiros"]) for r in registros)
    linhas_df = []
    for idx, r in enumerate(registros, start=1):
        base = {}
        # ID original ou índice sequencial
        base["serie"] = r["serie_id"] or f"C{idx}"
        # Preenche passageiros
        for j in range(max_p):
            col = f"n{j+1}"
            if j < len(r["passageiros"]):
                base[col] = r["passageiros"][j]
            else:
                # Completa com NaN e depois preenche com -1 (ou outro marcador)
                base[col] = np.nan
        base["k"] = r["k"]
        linhas_df.append(base)

    df = pd.DataFrame(linhas_df)

    # Preenche NaN (caso haja) com -1 para não estragar análises de faixas
    num_cols = [c for c in df.columns if c.startswith("n")]
    df[num_cols] = df[num_cols].fillna(-1).astype(int)

    df["k"] = df["k"].astype(int)
    return df


def parse_historico_csv(file) -> pd.DataFrame:
    """
    Parser FLEX ULTRA para arquivo CSV.
    Aceita:

    - CSV com coluna 'serie' + n1..nX + k
    - CSV com primeira coluna ID (C1...) + 6 passageiros + k
    - CSV somente com passageiros + k (sem ID) → cria 'serie'
    """
    try:
        df_raw = pd.read_csv(file, sep=None, engine="python")
    except Exception:
        # fallback simples: tenta ponto e vírgula
        file.seek(0)
        df_raw = pd.read_csv(file, sep=";")

    # Limpa espaços em nomes de colunas
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    # Se só tem uma coluna → pode ser string inteira com separador; tenta quebrar
    if df_raw.shape[1] == 1:
        colname = df_raw.columns[0]
        # Concatena todas as linhas e reaplica o parser de texto
        raw = "\n".join(str(x) for x in df_raw[colname].astype(str))
        return parse_historico_text(raw)

    # Tenta identificar se já tem 'serie' e 'k'
    cols_lower = [c.lower() for c in df_raw.columns]
    has_serie = any(c.startswith("serie") for c in cols_lower)
    has_k = any(c == "k" for c in cols_lower)

    df = df_raw.copy()

    if has_serie and has_k:
        # Apenas renomeia consistentemente
        map_cols = {}
        for c in df.columns:
            cl = c.lower()
            if cl.startswith("serie"):
                map_cols[c] = "serie"
            elif cl == "k":
                map_cols[c] = "k"
        df = df.rename(columns=map_cols)

        # Identifica colunas de passageiros
        num_cols = [c for c in df.columns if c not in ["serie", "k"]]
        # Garante ordem estável
        num_cols_sorted = sorted(num_cols)
        # Renomeia numéricas para n1..nN
        rename_map = {}
        for i, col in enumerate(num_cols_sorted, start=1):
            rename_map[col] = f"n{i}"
        df = df.rename(columns=rename_map)

    else:
        # Não tem 'serie' ou 'k' claramente definidos
        # Assume: primeira coluna = serie (ou índice), última = k
        cols = list(df.columns)
        first_col = cols[0]
        last_col = cols[-1]

        # Cria 'serie'
        df["serie"] = df[first_col].astype(str)
        # Cria 'k'
        df["k"] = df[last_col]

        # Passageiros = colunas intermediárias
        mid_cols = cols[1:-1]
        rename_map = {}
        for i, col in enumerate(mid_cols, start=1):
            rename_map[col] = f"n{i}"
        df = df.rename(columns=rename_map)

        # Drop colunas antigas de ID e k se forem diferentes dos novos
        for col in cols:
            if col not in rename_map and col not in [first_col, last_col]:
                # Já são numéricas ou extras; permanecem
                continue

        # Mantém apenas série, n1..nN, k
        keep_cols = ["serie"] + [c for c in df.columns if c.startswith("n")] + ["k"]
        df = df[keep_cols]

    # Converte tipos
    num_cols = [c for c in df.columns if c.startswith("n")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(int)
    df["k"] = pd.to_numeric(df["k"], errors="coerce").fillna(0).astype(int)

    # Garante que 'serie' existe
    if "serie" not in df.columns:
        df.insert(0, "serie", [f"C{i+1}" for i in range(len(df))])

    return df


def classificar_regime_por_k(k_medio: float) -> str:
    """
    Classificação de regime global (bem simples, mas suficiente para
    manter a lógica V15.x de barômetro/regime).
    """
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
    """
    Converte um valor de QDS (0..1) para uma categoria.
    """
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
    """
    matriz_acertos: lista de simulações, cada uma com lista de acertos por série.
    """
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
    """
    k* simples: porcentagem de séries com k>0 na janela final.
    Aqui fica o k* local + classificação de regime local.
    """
    if df is None or df.empty:
        return ResumoK(
            k_atual=0,
            k_star=0.0,
            estado_k="desconhecido",
            regime_local="desconhecido",
        )

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

    # Estado por faixas
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
# INICIALIZAÇÃO DO SESSION_STATE
# ------------------------------------------------------------

def init_session_state() -> None:
    """
    Garante que todas as chaves principais existam em st.session_state.
    Isto é importante para o V15.5-HÍBRIDO, pois vários painéis
    vão se alimentando em cadeia (Entrada → Pipeline → Ruído → QDS →
    Backtest → Monte Carlo → TURBO++ ULTRA → Relatório AIQ Bridge).
    """
    defaults = {
        "df": None,                      # histórico original
        "df_limpo": None,                # histórico após limpeza básica
        "df_ruido_a": None,              # histórico após tratamento de ruído A
        "df_ruido_b": None,              # histórico após tratamento de ruído B / condicional
        "resumo_estrada": None,          # ResumoEstrada
        "resumo_k_global": None,         # ResumoK global/local
        "lista_qds": [],                 # lista de valores QDS por série/janela
        "resumo_qds": None,              # ResumoQDS
        "resumo_ruido": None,            # ResumoRuido
        "resumo_backtest": None,         # ResumoBacktest
        "resumo_montecarlo": None,       # ResumoMonteCarlo
        "previsao_turbo_ultra": None,    # lista com série final prevista
        "meta_expectativa_acertos": {},  # dict com expectativa por ambiente
        "historico_backtest": None,      # DataFrame detalhado de backtest real
        "historico_montecarlo": None,    # DataFrame detalhado com simulações
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


# ------------------------------------------------------------
# NAVEGAÇÃO PRINCIPAL (V15.5-HÍBRIDO)
# ------------------------------------------------------------

st.title("🚗 Predict Cars V15.5-HÍBRIDO")
st.caption(
    "Núcleo V14-FLEX ULTRA + k* + Ruído Condicional + QDS REAL + "
    "Backtest REAL + Monte Carlo REAL + AIQ Bridge (para ChatGPT)."
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
    st.markdown("### ⚙️ Opções Globais (visual)")
    mostrar_debug = st.checkbox("Exibir tabelas de debug / inspeção", value=False)
    st.session_state["mostrar_debug"] = mostrar_debug


# ------------------------------------------------------------
# PAINEL 1 — HISTÓRICO — ENTRADA FLEX ULTRA (V15-HÍBRIDO)
# ------------------------------------------------------------

if painel == "📥 Histórico — Entrada FLEX ULTRA (V15-HÍBRIDO)":
    st.markdown("## 📥 Histórico — Entrada FLEX ULTRA (V15-HÍBRIDO)")
    st.markdown(
        """
        Este painel é o **ponto de partida** do V15.5-HÍBRIDO.

        - Aceita histórico no formato **FLEX ULTRA**, por **arquivo CSV** ou **texto colado**.
        - Detecta automaticamente:
            - ID da série (C1, C2, ...)
            - Número de passageiros (n1..nN, N variável)
            - Coluna de rótulo `k`.
        - Gera e guarda na sessão:
            - `df` (histórico bruto normalizado)
            - `df_limpo` (limpeza básica)
            - `resumo_estrada`
            - `resumo_k_global`
        """
    )

    with st.expander("📌 Instruções de formato (FLEX ULTRA)", expanded=False):
        st.markdown(
            """
            Exemplos aceitos:

            **1) Com ID da série**

            ```text
            C1;41;5;4;52;30;33;0
            C2;9;39;37;49;43;41;1
            C3;36;30;10;11;29;47;2
            ```

            **2) Sem ID da série**

            ```text
            41;5;4;52;30;33;0
            9;39;37;49;43;41;1
            36;30;10;11;29;47;2
            ```

            **3) CSV com colunas já nomeadas**

            ```csv
            serie,n1,n2,n3,n4,n5,n6,k
            C1,41,5,4,52,30,33,0
            C2,9,39,37,49,43,41,1
            C3,36,30,10,11,29,47,2
            ```
            """
        )

    modo_entrada = st.radio(
        "Escolha o modo de entrada:",
        options=["Upload CSV", "Colar texto"],
        horizontal=True,
    )

    df_result = None

    if modo_entrada == "Upload CSV":
        file = st.file_uploader(
            "Selecione o arquivo de histórico (.csv):",
            type=["csv", "txt"],
        )
        if file is not None:
            df_result = parse_historico_csv(file)
    else:
        raw_text = st.text_area(
            "Cole aqui o histórico completo:",
            height=240,
            placeholder="C1;41;5;4;52;30;33;0\nC2;9;39;37;49;43;41;1\n...",
        )
        if raw_text.strip():
            df_result = parse_historico_text(raw_text)

    if df_result is not None and not df_result.empty:
        st.success(
            f"Histórico carregado com sucesso! Total de séries: {len(df_result)}."
        )

        # Ordena por índice original caso exista padrão C1, C2, ...
        try:
            df_result = df_result.copy()
            df_result["__idx"] = (
                df_result["serie"].astype(str).str.extract(r"(\d+)").astype(float)
            )
            df_result = df_result.sort_values("__idx").drop(columns=["__idx"])
        except Exception:
            pass

        # Guarda no session_state
        st.session_state["df"] = df_result
        st.session_state["df_limpo"] = df_result.copy()

        # Calcula resumos iniciais da estrada e k*
        resumo_estrada = calcular_resumo_estrada(df_result)
        resumo_k_global = calcular_k_star(df_result, janela=min(60, len(df_result)))

        st.session_state["resumo_estrada"] = resumo_estrada
        st.session_state["resumo_k_global"] = resumo_k_global

        # Mostra quadro-resumo
        if resumo_estrada is not None and resumo_k_global is not None:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### 🛣️ Estrada Global")
                st.metric("Séries no histórico", resumo_estrada.n_series)
                st.metric("Passageiros por série", resumo_estrada.n_passageiros)
                st.metric("Faixa de valores", f"{resumo_estrada.min_val} — {resumo_estrada.max_val}")

            with col2:
                st.markdown("### 🌡️ Regime / Barômetro")
                st.metric("Regime global", resumo_estrada.regime_global)
                st.metric("k médio", f"{resumo_estrada.k_medio:.2f}")
                st.metric("k máximo", resumo_estrada.k_max)

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

        # Preview da tabela
        st.markdown("### 🔎 Amostra do histórico normalizado")
        st.dataframe(df_result.head(50), use_container_width=True)

        if st.session_state["mostrar_debug"]:
            st.markdown("#### 🐞 DEBUG — Info do DataFrame")
            st.write(df_result.describe(include="all"))

    else:
        st.info(
            "Carregue um histórico por **Upload CSV** ou **Cole texto** para "
            "iniciar o pipeline V15.5-HÍBRIDO."
        )

# ------------------------------------------------------------
# (Os demais painéis serão definidos nas próximas partes:)
#
# - 🔍 Pipeline V14-FLEX ULTRA (V15)
# - 💡 Replay LIGHT
# - 📅 Replay ULTRA
# - 🎯 Replay ULTRA Unitário
# - 🚨 Monitor de Risco (k & k*)
# - 🧪 Testes de Confiabilidade REAL
# - 📊 Ruído Condicional (V15)
# - 🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15)
# - 📄 Relatório Final — AIQ Bridge (para ChatGPT)
#
# Eles vão usar todos os resumos / estruturas criados aqui, SEM
# QUALQUER SIMPLIFICAÇÃO, mantendo o jeitão V15.x.
# ------------------------------------------------------------
# ------------------------------------------------------------
# FUNÇÕES DO PIPELINE V14-FLEX ULTRA (V15)
# ------------------------------------------------------------

def get_passenger_cols(df: pd.DataFrame) -> List[str]:
    """
    Retorna as colunas de passageiros (n1..nN) em ordem.
    """
    return sorted([c for c in df.columns if c.startswith("n")],
                  key=lambda x: int(x[1:]))


def extrair_janela_hist(
    df: pd.DataFrame,
    idx_alvo: int,
    back: int,
    forward: int = 0,
) -> pd.DataFrame:
    """
    Extrai janela em torno do índice alvo (1-based na interface).

    - back: quantas séries para trás
    - forward: quantas para frente (normalmente 0 para predição)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    n = len(df)
    pos = max(0, min(idx_alvo - 1, n - 1))  # 0-based interno

    ini = max(0, pos - back)
    fim = min(n, pos + 1 + forward)
    return df.iloc[ini:fim].copy()


def calcular_matriz_frequencia(
    janela: pd.DataFrame,
    suavizacao: float = 1.0,
) -> Dict[str, Dict[int, float]]:
    """
    Calcula uma matriz de frequência (por posição de passageiro).

    Retorna:
        { "n1": {valor: prob, ...}, "n2": {...}, ... }
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

        # Laplace smoothing básico
        total = sum(contagens.values()) + suavizacao * max(len(contagens), 1)
        probs: Dict[int, float] = {}
        for v, c in contagens.items():
            probs[v] = (c + suavizacao) / total

        matriz[col] = probs

    return matriz


def gerar_candidato_serie(
    matriz_freq: Dict[str, Dict[int, float]],
    rng: random.Random,
) -> List[int]:
    """
    Gera uma série candidata amostrando da matriz de frequências
    por posição (n1..nN). Caso alguma posição não tenha histórico,
    sorteia um valor sintético (0..60) como fallback.
    """
    if not matriz_freq:
        # fallback extremo
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

        escolha = rng.choice(valores, p=probs)
        serie.append(int(escolha))

    return serie


def calcular_diversidade_serie(serie: List[int]) -> float:
    """
    Diversidade simples: proporção de valores distintos na série.
    """
    if not serie:
        return 0.0
    return len(set(serie)) / len(serie)


def calcular_qds_candidato(
    serie: List[int],
    matriz_freq: Dict[str, Dict[int, float]],
) -> float:
    """
    QDS do candidato: média das probabilidades (por posição) segundo
    a matriz de frequências. Resultado em [0,1] (já normalizado).
    """
    if not serie or not matriz_freq:
        return 0.0

    probs_pos: List[float] = []
    cols = sorted(matriz_freq.keys(), key=lambda x: int(x[1:]))

    for idx, col in enumerate(cols):
        dist = matriz_freq[col]
        if idx >= len(serie):
            continue
        v = serie[idx]
        p = dist.get(v, 0.0)
        probs_pos.append(float(p))

    if not probs_pos:
        return 0.0

    # Clampa em [0,1]
    qds = float(np.mean(probs_pos))
    return float(max(0.0, min(1.0, qds)))


def calcular_aiq_candidato(
    serie: List[int],
    matriz_freq: Dict[str, Dict[int, float]],
    peso_qds: float = 0.6,
    peso_div: float = 0.4,
) -> Tuple[float, float, float]:
    """
    AIQ = combinação de QDS e Diversidade, ainda em nível de candidato.
    Retorna (AIQ, QDS, diversidade).
    """
    qds = calcular_qds_candidato(serie, matriz_freq)
    diversidade = calcular_diversidade_serie(serie)

    # Normalização de segurança
    qds = max(0.0, min(1.0, qds))
    diversidade = max(0.0, min(1.0, diversidade))

    aiq = peso_qds * qds + peso_div * diversidade
    aiq = max(0.0, min(1.0, aiq))

    return float(aiq), float(qds), float(diversidade)


def gerar_leque_candidatos(
    matriz_freq: Dict[str, Dict[int, float]],
    n_series: int,
    seed: int,
) -> List[List[int]]:
    """
    Gera um leque de candidatos usando a matriz de frequências por posição.

    Usa RNG determinístico por seed para reprodutibilidade (por índice alvo).
    """
    rng = random.Random(seed)
    candidatos: List[List[int]] = []
    vistos = set()

    max_tentativas = max(n_series * 10, n_series + 10)

    while len(candidatos) < n_series and max_tentativas > 0:
        s = gerar_candidato_serie(matriz_freq, rng)
        chave = tuple(s)
        if chave not in vistos:
            vistos.add(chave)
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
    Monta DataFrame com colunas:
    - idx
    - serie (lista)
    - QDS
    - Diversidade
    - AIQ
    - Regime_global
    - Regime_local_k
    """
    registros = []
    for i, serie in enumerate(candidatos, start=1):
        aiq, qds, div = calcular_aiq_candidato(serie, matriz_freq)
        reg_local = resumo_k.regime_local if resumo_k is not None else "desconhecido"

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
    df = df.sort_values(["AIQ", "QDS"], ascending=[False, False]).reset_index(drop=True)
    return df


# ------------------------------------------------------------
# PAINEL 2 — Pipeline V14-FLEX ULTRA (V15)
# ------------------------------------------------------------

if painel == "🔍 Pipeline V14-FLEX ULTRA (V15)":
    st.markdown("## 🔍 Pipeline V14-FLEX ULTRA (V15)")
    st.markdown(
        """
        Este painel executa o **núcleo V14-FLEX ULTRA** em modo de predição:

        1. Seleciona um **índice alvo** no histórico.
        2. Extrai uma **janela local** da estrada.
        3. Calcula a **matriz de frequências** por posição (n1..nN).
        4. Gera um **leque base de candidatos** (V14-FLEX ULTRA).
        5. Avalia **QDS**, diversidade e **AIQ** de cada candidato.
        6. Entrega uma **previsão base V14** (antes do tratamento de ruído profundo).
        """
    )

    df_limpo = st.session_state.get("df_limpo", None)
    if df_limpo is None or df_limpo.empty:
        st.warning(
            "Carregue o histórico primeiro no painel "
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
    # CONTROLES GERAIS DO PIPELINE
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
            "Tamanho da janela para trás (histórico local):",
            min_value=10,
            max_value=min(300, n_series_hist - 1),
            value=min(60, max(10, n_series_hist - 1)),
            step=5,
        )
    with col_c:
        n_candidatos = st.slider(
            "Quantidade de séries no leque base V14:",
            min_value=10,
            max_value=200,
            value=60,
            step=5,
        )

    col_d, col_e = st.columns(2)
    with col_d:
        seed_base = st.number_input(
            "Seed do gerador V14 (reprodutibilidade):",
            min_value=1,
            max_value=999999,
            value=12345,
            step=1,
        )
    with col_e:
        peso_qds = st.slider(
            "Peso do QDS no AIQ (restante é Diversidade):",
            min_value=0.1,
            max_value=0.9,
            value=0.6,
            step=0.05,
        )

    # --------------------------------------------------------
    # CONTEXTO DO ALVO / JANELA LOCAL
    # --------------------------------------------------------

    st.markdown("### 🛰️ Contexto local do alvo")

    df_janela = extrair_janela_hist(df_limpo, int(idx_alvo), back=int(janela_back))
    if df_janela.empty:
        st.error("Janela vazia ou inválida. Ajuste os parâmetros.")
        st.stop()

    # Série alvo (última da janela)
    serie_alvo = df_limpo.iloc[int(idx_alvo) - 1]
    st.markdown(f"**Série alvo:** `{serie_alvo['serie']}` (índice {idx_alvo})")

    col_s1, col_s2, col_s3 = st.columns([2, 2, 2])

    with col_s1:
        st.markdown("#### 🚗 Série alvo (passageiros + k)")
        valores = [int(serie_alvo[c]) for c in cols_pass]
        k_val = int(serie_alvo["k"])
        st.code(
            " ".join(str(x) for x in valores) + f" | k = {k_val}",
            language="text",
        )

    with col_s2:
        st.markdown("#### 🧭 Janela local")
        st.write(
            f"Séries consideradas na janela: **{len(df_janela)}** "
            f"(de {df_janela['serie'].iloc[0]} até {df_janela['serie'].iloc[-1]})."
        )
        if resumo_estrada is not None:
            st.metric("Regime global (estrada)", resumo_estrada.regime_global)
        if resumo_k_global is not None:
            st.metric("k* global", f"{resumo_k_global.k_star*100:.1f}%")

    with col_s3:
        st.markdown("#### 🔭 k* local (na janela)")
        resumo_k_local = calcular_k_star(df_janela, janela=len(df_janela))
        st.metric("k atual (última da janela)", resumo_k_local.k_atual)
        st.metric("k* local", f"{resumo_k_local.k_star*100:.1f}%")
        label_local = {
            "estavel": "🟢 Ambiente estável",
            "atencao": "🟡 Pré-ruptura residual",
            "critico": "🔴 Ambiente crítico",
        }.get(resumo_k_local.estado_k, "⚪ Desconhecido")
        st.write(label_local)
        st.caption(f"Regime local: **{resumo_k_local.regime_local}**")

    if st.session_state["mostrar_debug"]:
        st.markdown("#### 🐞 DEBUG — Janela local (top 30)")
        st.dataframe(df_janela.head(30), use_container_width=True)

    # --------------------------------------------------------
    # MATRIZ DE FREQUÊNCIAS V14-FLEX ULTRA
    # --------------------------------------------------------

    st.markdown("### 📊 Matriz de frequências (base V14-FLEX ULTRA)")
    matriz_freq = calcular_matriz_frequencia(df_janela, suavizacao=1.0)

    if not matriz_freq:
        st.error("Falha ao calcular a matriz de frequências. Verifique o histórico.")
        st.stop()

    # Exibição resumida da matriz (apenas top frequências por posição)
    cols_preview = st.columns(len(cols_pass))
    for i, col in enumerate(cols_pass):
        with cols_preview[i]:
            st.markdown(f"**{col}**")
            dist = matriz_freq.get(col, {})
            if not dist:
                st.write("Sem dados.")
            else:
                # Top 5 valores
                top_vals = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:5]
                linhas = [f"{v}: {p*100:.1f}%" for v, p in top_vals]
                st.caption("\n".join(linhas))

    # --------------------------------------------------------
    # GERAÇÃO DO LEQUE BASE V14-FLEX ULTRA
    # --------------------------------------------------------

    st.markdown("### 🎯 Leque base V14-FLEX ULTRA")

    candidatos = gerar_leque_candidatos(
        matriz_freq=matriz_freq,
        n_series=int(n_candidatos),
        seed=int(seed_base + int(idx_alvo) * 13),
    )

    # Monta tabela com QDS / Diversidade / AIQ
    regime_global_str = resumo_estrada.regime_global if resumo_estrada else "desconhecido"
    df_candidatos = montar_tabela_candidatos(
        candidatos=candidatos,
        matriz_freq=matriz_freq,
        regime_global=regime_global_str,
        resumo_k=resumo_k_local,
    )

    # Reajusta AIQ se o usuário quiser alterar o peso do QDS (sem regenerar o leque)
    if not df_candidatos.empty:
        # recalcula AIQ com peso customizado
        novas_aiq = []
        for _, row in df_candidatos.iterrows():
            serie = row["series"]
            _, qds_tmp, div_tmp = calcular_aiq_candidato(
                serie,
                matriz_freq,
                peso_qds=peso_qds,
                peso_div=1.0 - peso_qds,
            )
            aiq_tmp = peso_qds * qds_tmp + (1.0 - peso_qds) * div_tmp
            novas_aiq.append(aiq_tmp)

        df_candidatos["AIQ"] = novas_aiq
        df_candidatos = df_candidatos.sort_values(
            ["AIQ", "QDS"], ascending=[False, False]
        ).reset_index(drop=True)

    if df_candidatos.empty:
        st.error("Nenhum candidato foi gerado. Ajuste os parâmetros e tente novamente.")
        st.stop()

    # Guarda QDS da geração atual no contexto global (para estatísticas futuras)
    lista_qds_global: List[float] = st.session_state.get("lista_qds", [])
    lista_qds_global.extend(df_candidatos["QDS"].astype(float).tolist())
    st.session_state["lista_qds"] = lista_qds_global

    # Previsão base V14 = melhor AIQ do leque
    melhor = df_candidatos.iloc[0]
    previsao_base = melhor["series"]
    st.session_state["previsao_base_v14"] = previsao_base

    st.markdown("#### 🏁 Previsão base V14-FLEX ULTRA (pré-ruído)")
    st.code(" ".join(str(x) for x in previsao_base), language="text")

    st.caption(
        "Esta é a saída **pura do núcleo V14-FLEX ULTRA**, antes de qualquer "
        "tratamento de ruído condicional profundo ou ajustes avançados de "
        "Backtest/Monte Carlo. Ela será refinada no painel "
        "🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15)."
    )

    # Exibição da tabela de candidatos
    st.markdown("#### 📋 Leque base completo (ordenado por AIQ)")
    df_view = df_candidatos.copy()
    # Converte lista para string para visualização
    df_view["series"] = df_view["series"].apply(
        lambda s: " ".join(str(x) for x in s)
    )

    st.dataframe(df_view, use_container_width=True)

    if st.session_state["mostrar_debug"]:
        st.markdown("#### 🐞 DEBUG — Estatísticas dos candidatos")
        st.write(df_candidatos.describe(include="all"))
# ------------------------------------------------------------
# PAINEL 3 — REPLAY LIGHT
# ------------------------------------------------------------

if painel == "💡 Replay LIGHT":
    st.markdown("## 💡 Replay LIGHT — Inspeção rápida do histórico")

    df_limpo = st.session_state.get("df_limpo", None)
    if df_limpo is None or df_limpo.empty:
        st.warning(
            "Carregue o histórico no painel "
            "'📥 Histórico — Entrada FLEX ULTRA (V15-HÍBRIDO)'."
        )
        st.stop()

    cols_pass = get_passenger_cols(df_limpo)

    st.markdown(
        """
        O Replay LIGHT permite inspecionar rapidamente qualquer série do histórico:

        - Mostra passageiros + k  
        - Mostra o contexto local (janela pequena)  
        - Calcula k* local  
        """
    )

    idx_view = st.number_input(
        "Escolha a série para inspecionar:",
        min_value=1,
        max_value=len(df_limpo),
        value=len(df_limpo),
        step=1,
    )

    serie = df_limpo.iloc[int(idx_view) - 1]
    valores = [int(serie[c]) for c in cols_pass]
    k_val = int(serie["k"])

    st.markdown("### 🚗 Série selecionada")
    st.code(" ".join(str(x) for x in valores) + f"  |  k = {k_val}", language="text")

    # Janela de inspeção curta
    janela_curta = extrair_janela_hist(df_limpo, int(idx_view), back=20)
    resumo_k_local = calcular_k_star(janela_curta, janela=len(janela_curta))

    st.markdown("### 🌡️ Ambiente local")
    st.metric("k* local", f"{resumo_k_local.k_star*100:.1f}%")
    st.write(
        {
            "estavel": "🟢 Ambiente estável",
            "atencao": "🟡 Pré-ruptura residual",
            "critico": "🔴 Ambiente crítico",
        }.get(resumo_k_local.estado_k, "⚪ Desconhecido")
    )

    st.markdown("### 🔎 Janela local (curta)")
    st.dataframe(janela_curta.tail(30), use_container_width=True)


# ------------------------------------------------------------
# PAINEL 4 — REPLAY ULTRA
# ------------------------------------------------------------

if painel == "📅 Replay ULTRA":
    st.markdown("## 📅 Replay ULTRA — Execução completa da estrada")

    df_limpo = st.session_state.get("df_limpo", None)
    if df_limpo is None or df_limpo.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    n_total = len(df_limpo)
    cols_pass = get_passenger_cols(df_limpo)

    st.markdown(
        """
        O Replay ULTRA executa a **estrada inteira** como se estivéssemos no passado,
        rodando:

        - Matriz V14 local  
        - Leque base  
        - Previsão V14  
        - QDS real  
        - (Opcional) Comparação com a série real  
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        janela_back_ultra = st.slider(
            "Janela local para cada passo (Replay ULTRA):",
            min_value=20,
            max_value=200,
            value=60,
            step=5,
        )
    with col2:
        n_candidatos_ultra = st.slider(
            "Tamanho do leque base (V14) em cada passo:",
            min_value=10,
            max_value=200,
            value=60,
            step=5,
        )

    executar = st.button("▶️ Rodar Replay ULTRA")

    if executar:
        progresso = st.progress(0)
        registros = []

        for idx in range(2, n_total + 1):
            progresso.progress(idx / n_total)

            # Janela até a série anterior
            janela_local = extrair_janela_hist(
                df_limpo, idx, back=int(janela_back_ultra)
            )
            matriz_local = calcular_matriz_frequencia(janela_local)

            # Leque base para este passo
            candidatos = gerar_leque_candidatos(
                matriz_freq=matriz_local,
                n_series=int(n_candidatos_ultra),
                seed=123000 + idx * 7,
            )

            resumo_k_local = calcular_k_star(janela_local)
            regime_global = (
                st.session_state["resumo_estrada"].regime_global
                if st.session_state["resumo_estrada"]
                else "desconhecido"
            )

            df_cands = montar_tabela_candidatos(
                candidatos=candidatos,
                matriz_freq=matriz_local,
                regime_global=regime_global,
                resumo_k=resumo_k_local,
            )

            if df_cands.empty:
                continue

            melhor = df_cands.iloc[0]["series"]
            qds_melhor = float(df_cands.iloc[0]["QDS"])

            # Real x previsto
            serie_real = df_limpo.iloc[idx - 1]
            real_vals = [int(serie_real[c]) for c in cols_pass]

            registros.append(
                {
                    "idx": idx,
                    "prev": melhor,
                    "real": real_vals,
                    "k_real": int(serie_real["k"]),
                    "QDS": qds_melhor,
                }
            )

        df_replay = pd.DataFrame(registros)
        st.session_state["historico_backtest"] = df_replay

        st.success("Replay ULTRA concluído!")
        st.dataframe(df_replay, use_container_width=True)


# ------------------------------------------------------------
# PAINEL 5 — REPLAY ULTRA UNITÁRIO
# ------------------------------------------------------------

if painel == "🎯 Replay ULTRA Unitário":
    st.markdown("## 🎯 Replay ULTRA Unitário — Predição isolada do alvo")

    df_limpo = st.session_state.get("df_limpo", None)
    if df_limpo is None or df_limpo.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    n_total = len(df_limpo)
    cols_pass = get_passenger_cols(df_limpo)

    st.markdown(
        """
        O Replay ULTRA Unitário refaz a **predição exata** realizada para um índice histórico,
        aplicando a mesma matriz local, o mesmo leque e os mesmos critérios do V14-FLEX ULTRA.

        Útil para depuração e análise precisa de casos específicos.
        """
    )

    idx_alvo_unit = st.number_input(
        "Índice histórico a ser reavaliado:",
        min_value=1,
        max_value=n_total,
        value=n_total,
        step=1,
    )

    janela_back_u = st.slider(
        "Janela local (Replay Unitário):",
        min_value=20,
        max_value=200,
        value=60,
        step=5,
    )

    rodar_unit = st.button("▶️ Rodar Replay Unitário")

    if rodar_unit:
        janela_local = extrair_janela_hist(
            df_limpo, int(idx_alvo_unit), back=int(janela_back_u)
        )
        matriz_local = calcular_matriz_frequencia(janela_local)
        resumo_k_local = calcular_k_star(janela_local)

        # Leque específico
        candidatos = gerar_leque_candidatos(
            matriz_freq=matriz_local,
            n_series=80,
            seed=999 + int(idx_alvo_unit) * 13,
        )

        regime_global = (
            st.session_state["resumo_estrada"].regime_global
            if st.session_state["resumo_estrada"]
            else "desconhecido"
        )

        df_cands = montar_tabela_candidatos(
            candidatos=candidatos,
            matriz_freq=matriz_local,
            regime_global=regime_global,
            resumo_k=resumo_k_local,
        )

        if df_cands.empty:
            st.error("Falha ao gerar candidatos.")
            st.stop()

        melhor = df_cands.iloc[0]["series"]
        qds_melhor = float(df_cands.iloc[0]["QDS"])

        # REAL:
        serie_real = df_limpo.iloc[int(idx_alvo_unit) - 1]
        real_vals = [int(serie_real[c]) for c in cols_pass]

        st.markdown("### 🔍 Resultado exato (Replay Unitário)")
        st.code(" ".join(str(x) for x in melhor), language="text")

        st.markdown("**QDS da previsão:** " + f"{qds_melhor:.4f}")
        st.markdown("**Série real:**")
        st.code(" ".join(str(x) for x in real_vals), language="text")


# ------------------------------------------------------------
# PAINEL 6 — MONITOR DE RISCO (k & k*)
# ------------------------------------------------------------

if painel == "🚨 Monitor de Risco (k & k*)":
    st.markdown("## 🚨 Monitor de Risco — k & k*")

    df_limpo = st.session_state.get("df_limpo", None)
    if df_limpo is None or df_limpo.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    resumo_k_global: Optional[ResumoK] = st.session_state.get("resumo_k_global")

    st.markdown(
        """
        O Monitor de Risco mostra:

        - k (última série)  
        - k* global  
        - Tendências locais  
        - Regimes  
        """
    )

    if resumo_k_global is None:
        st.error("Resumo global de k* não encontrado.")
        st.stop()

    st.markdown("### 🔭 k* Global")
    st.metric("k atual (última série)", resumo_k_global.k_atual)
    st.metric("k*", f"{resumo_k_global.k_star*100:.1f}%")

    label_global = {
        "estavel": "🟢 Ambiente estável",
        "atencao": "🟡 Pré-ruptura residual",
        "critico": "🔴 Ambiente crítico",
    }.get(resumo_k_global.estado_k, "⚪ Desconhecido")

    st.write(label_global)
    st.caption(f"Regime global: **{resumo_k_global.regime_local}**")

    # Plot simples das últimas janelas de k* locais
    st.markdown("### 📈 k* — análise local por janelas móveis")

    df_tmp = df_limpo.copy()
    valores_k_star = []
    for i in range(20, len(df_tmp) + 1):
        janela = df_tmp.iloc[:i]
        r = calcular_k_star(janela, janela=len(janela))
        valores_k_star.append(r.k_star)

    if valores_k_star:
        st.line_chart(valores_k_star)
    else:
        st.info("Histórico insuficiente para análise dinâmica de k*.")


# ------------------------------------------------------------
# PAINEL 7 — TESTES DE CONFIABILIDADE REAL
# ------------------------------------------------------------

if painel == "🧪 Testes de Confiabilidade REAL":
    st.markdown("## 🧪 Testes de Confiabilidade REAL")

    df_backtest = st.session_state.get("historico_backtest", None)
    if df_backtest is None or df_backtest.empty:
        st.info(
            "Execute primeiro o **📅 Replay ULTRA** para gerar o histórico "
            "necessário para o Backtest REAL."
        )
        st.stop()

    st.markdown(
        """
        Aqui analisamos:

        - Acertos por série  
        - QDS médio do Replay  
        - Estatísticas profundas do V14  
        """
    )

    # Calcula acertos por série
    acertos = []
    for _, row in df_backtest.iterrows():
        prev = row["prev"]
        real = row["real"]
        ac = sum(1 for a, b in zip(prev, real) if a == b)
        acertos.append(ac)

    df_backtest["acertos"] = acertos
    st.dataframe(df_backtest, use_container_width=True)

    # Resumo Backtest REAL
    resumo_bt = calcular_resumo_backtest(acertos, n_series_por_janela=1)
    st.session_state["resumo_backtest"] = resumo_bt

    if resumo_bt:
        st.markdown("### 📊 Resumo do Backtest REAL")
        st.metric("Janelas avaliadas", resumo_bt.n_janelas)
        st.metric("Acertos totais", resumo_bt.acertos_totais)
        st.metric("Média por série", f"{resumo_bt.acertos_por_serie:.2f}")
        st.metric("Hit rate", f"{resumo_bt.hit_rate*100:.2f}%")


# ------------------------------------------------------------
# PAINEL 8 — RUÍDO CONDICIONAL (V15)
# ------------------------------------------------------------

if painel == "📊 Ruído Condicional (V15)":
    st.markdown("## 📊 Ruído Condicional — V15")

    df_limpo = st.session_state.get("df_limpo", None)
    if df_limpo is None or df_limpo.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    st.markdown(
        """
        Tratamento de ruído na estrada:

        - Ruído A (normalização / ajustes simples)  
        - Ruído B (condicional, dependente de regime e k*)  
        """
    )

    # Parâmetros do ruído
    col1, col2, col3 = st.columns(3)
    with col1:
        intensidade_a = st.slider(
            "Intensidade do Ruído A (0–1):", 0.0, 1.0, 0.2, 0.05
        )
    with col2:
        intensidade_b = st.slider(
            "Intensidade do Ruído B (0–1):", 0.0, 1.0, 0.3, 0.05
        )
    with col3:
        uso_regime = st.checkbox("Usar regime/k* para Ruído B", value=True)

    df_ruido_a = df_limpo.copy()
    df_ruido_b = df_limpo.copy()

    # ----- RUÍDO A -----
    for c in get_passenger_cols(df_limpo):
        media = df_limpo[c].mean()
        df_ruido_a[c] = (
            df_limpo[c] + intensidade_a * (media - df_limpo[c])
        ).astype(int)

    # ----- RUÍDO B -----
    resumo_k_global = st.session_state.get("resumo_k_global")
    fator_regime = 1.0
    if uso_regime and resumo_k_global is not None:
        if resumo_k_global.estado_k == "estavel":
            fator_regime = 0.5
        elif resumo_k_global.estado_k == "atencao":
            fator_regime = 1.0
        else:
            fator_regime = 1.5

    for c in get_passenger_cols(df_limpo):
        media = df_limpo[c].mean()
        desvio = df_limpo[c].std() if df_limpo[c].std() > 0 else 1.0
        ruido = (
            intensidade_b
            * fator_regime
            * np.random.normal(loc=0.0, scale=desvio, size=len(df_limpo))
        )
        df_ruido_b[c] = np.clip(df_limpo[c] + ruido, 0, 60).astype(int)

    # Salva no contexto
    st.session_state["df_ruido_a"] = df_ruido_a
    st.session_state["df_ruido_b"] = df_ruido_b

    # Resumo do ruído aplicado
    ruido_inicial = float(intensidade_a)
    ruido_final = float(intensidade_b * fator_regime)
    pct_aj = float((abs(ruido_final - ruido_inicial) / max(ruido_inicial, 0.001)) * 100)

    resumo_ruido = calcular_resumo_ruido(
        ruido_inicial,
        ruido_final,
        pct_aj,
    )
    st.session_state["resumo_ruido"] = resumo_ruido

    st.markdown("### 🔎 Resumo do tratamento de ruído")
    st.metric("Ruído A aplicado", f"{resumo_ruido.ruido_inicial:.2f}")
    st.metric("Ruído B aplicado", f"{resumo_ruido.ruido_final:.2f}")
    st.metric("% Pontos ajustados", f"{resumo_ruido.pct_pontos_ajustados:.1f}%")

    if st.checkbox("Mostrar tabelas modificadas"):
        st.markdown("#### Ruído A")
        st.dataframe(df_ruido_a.head(50), use_container_width=True)
        st.markdown("#### Ruído B")
        st.dataframe(df_ruido_b.head(50), use_container_width=True)
# ------------------------------------------------------------
# PAINEL 9 — MODO TURBO++ ULTRA ANTI-RUÍDO (V15)
# ------------------------------------------------------------

if painel == "🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15)":
    st.markdown("## 🚀 Modo TURBO++ ULTRA ANTI-RUÍDO — V15")

    df_limpo = st.session_state.get("df_limpo", None)
    df_ra = st.session_state.get("df_ruido_a", None)
    df_rb = st.session_state.get("df_ruido_b", None)

    if df_limpo is None or df_limpo.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    if df_ra is None or df_rb is None:
        st.warning(
            "Execute primeiro o painel 📊 Ruído Condicional (V15)."
        )
        st.stop()

    resumo_estrada = st.session_state.get("resumo_estrada", None)
    resumo_k_global = st.session_state.get("resumo_k_global", None)

    st.markdown(
        """
        O TURBO++ ULTRA refaz o núcleo V14 **sobre duas estradas paralelas**:

        - Estrada A (ruído A)
        - Estrada B (ruído B)

        Depois **une os dois leques**, recalcula AIQ-HÍBRIDO e seleciona:

        ### ▶️ A Previsão Final do Motor (TURBO++ ULTRA)
        """
    )

    # --------------------------------------------------------
    # CONTROLES
    # --------------------------------------------------------

    col1, col2 = st.columns(2)
    with col1:
        janela_turbo = st.slider(
            "Janela local (TURBO++):",
            min_value=20,
            max_value=200,
            value=60,
            step=5,
        )
    with col2:
        n_cand_turbo = st.slider(
            "Tamanho do leque (A e B):",
            min_value=20,
            max_value=200,
            value=80,
            step=5,
        )

    seed_turbo = st.number_input(
        "Seed TURBO++ (reprodutível):",
        min_value=1,
        max_value=999999,
        value=2025,
        step=1,
    )

    target_idx = len(df_limpo)
    cols_pass = get_passenger_cols(df_limpo)

    rodar_turbo = st.button("▶️ Rodar TURBO++ ULTRA")

    if rodar_turbo:
        # --------------------------------------------------------
        # MATRIZ FREQUÊNCIA A / B
        # --------------------------------------------------------
        janela_A = extrair_janela_hist(df_ra, target_idx, back=int(janela_turbo))
        janela_B = extrair_janela_hist(df_rb, target_idx, back=int(janela_turbo))

        matriz_A = calcular_matriz_frequencia(janela_A)
        matriz_B = calcular_matriz_frequencia(janela_B)

        # --------------------------------------------------------
        # LEQUE A
        # --------------------------------------------------------
        candidatos_A = gerar_leque_candidatos(
            matriz_freq=matriz_A,
            n_series=int(n_cand_turbo),
            seed=int(seed_turbo + 17),
        )

        resumo_k_A = calcular_k_star(janela_A)
        dfA = montar_tabela_candidatos(
            candidatos_A,
            matriz_A,
            resumo_estrada.regime_global if resumo_estrada else "desconhecido",
            resumo_k_A,
        )

        # --------------------------------------------------------
        # LEQUE B
        # --------------------------------------------------------
        candidatos_B = gerar_leque_candidatos(
            matriz_freq=matriz_B,
            n_series=int(n_cand_turbo),
            seed=int(seed_turbo + 23),
        )

        resumo_k_B = calcular_k_star(janela_B)
        dfB = montar_tabela_candidatos(
            candidatos_B,
            matriz_B,
            resumo_estrada.regime_global if resumo_estrada else "desconhecido",
            resumo_k_B,
        )

        # --------------------------------------------------------
        # UNIR A + B → AIQ-HÍBRIDO
        # --------------------------------------------------------
        dfA["origem"] = "A"
        dfB["origem"] = "B"

        df_mix = pd.concat([dfA, dfB], ignore_index=True)
        df_mix = df_mix.sort_values(["AIQ", "QDS"], ascending=[False, False])

        melhor = df_mix.iloc[0]
        serie_best = melhor["series"]

        st.session_state["previsao_turbo_ultra"] = serie_best

        # --------------------------------------------------------
        # EXIBIÇÃO
        # --------------------------------------------------------
        st.markdown("### 🏁 Previsão Final TURBO++ ULTRA")
        st.code(" ".join(str(x) for x in serie_best), language="text")

        # Ambiente k*
        estado_label = {
            "estavel": "🟢 k*: Ambiente estável — previsão em regime normal.",
            "atencao": "🟡 k*: Pré-ruptura residual — previsão sob atenção.",
            "critico": "🔴 k*: Ambiente crítico — previsão sob cautela máxima.",
        }.get(
            resumo_k_global.estado_k if resumo_k_global else "desconhecido",
            "⚪ Ambiente desconhecido."
        )
        st.write(estado_label)

        st.caption("A previsão final será utilizada no Relatório AIQ Bridge.")


# ------------------------------------------------------------
# PAINEL 10 — RELATÓRIO FINAL — AIQ BRIDGE (para ChatGPT)
# ------------------------------------------------------------

if painel == "📄 Relatório Final — AIQ Bridge (para ChatGPT)":
    st.markdown("## 📄 Relatório Final — AIQ Bridge (para ChatGPT)")
    st.markdown(
        """
        Este é o painel **oficial** do V15.5.

        Ele gera um **relatório completo**, pronto para ser copiado e colado no ChatGPT:

        - Resumo da estrada  
        - Regime, barômetro, k e k*  
        - Dispersão, ruído e ajustes  
        - QDS global  
        - Backtest REAL  
        - Monte Carlo REAL  
        - Previsão Final (TURBO++ ULTRA)  
        - Expectativa de acertos por ambiente  
        """
    )

    df = st.session_state.get("df", None)
    resumo_estrada = st.session_state.get("resumo_estrada", None)
    resumo_k_global = st.session_state.get("resumo_k_global", None)
    resumo_qds = st.session_state.get("resumo_qds", None)
    resumo_ruido = st.session_state.get("resumo_ruido", None)
    resumo_bt = st.session_state.get("resumo_backtest", None)
    resumo_mc = st.session_state.get("resumo_montecarlo", None)
    previsao_final = st.session_state.get("previsao_turbo_ultra", None)

    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    # --------------------------------------------------------
    # CÁLCULO QDS GLOBAL (se ainda não calculado)
    # --------------------------------------------------------

    lista_qds_global = st.session_state.get("lista_qds", [])
    if lista_qds_global:
        resumo_qds = calcular_resumo_qds(lista_qds_global)
        st.session_state["resumo_qds"] = resumo_qds

    # --------------------------------------------------------
    # MONTE CARLO REAL (se o usuário quiser rodar aqui)
    # --------------------------------------------------------

    st.markdown("### 🎲 Monte Carlo REAL")

    n_mc = st.number_input(
        "Quantidade de simulações Monte Carlo:",
        min_value=100,
        max_value=5000,
        value=800,
        step=100,
    )
    rodar_mc = st.button("▶️ Rodar Monte Carlo REAL")

    if rodar_mc:
        cols_pass = get_passenger_cols(df)
        rng = random.Random(4242)
        matriz_acertos = []

        for _ in range(int(n_mc)):
            sim_acertos = []
            for _ in range(len(df)):
                a = rng.randint(0, 60)
                b = rng.randint(0, 60)
                sim_acertos.append(1 if a == b else 0)
            matriz_acertos.append(sim_acertos)

        resumo_mc = calcular_resumo_monte_carlo(matriz_acertos)
        st.session_state["resumo_montecarlo"] = resumo_mc

        st.success("Monte Carlo REAL concluído!")

    # --------------------------------------------------------
    # GERAR RELATÓRIO TEXTUAL (AIQ BRIDGE)
    # --------------------------------------------------------

    st.markdown("### 📄 Relatório consolidado")

    relatorio = []

    relatorio.append("==============================================")
    relatorio.append("PREDICT CARS V15.5 — RELATÓRIO FINAL")
    relatorio.append("==============================================\n")

    # Estrada
    if resumo_estrada:
        relatorio.append("🛣️ **Resumo da Estrada**")
        relatorio.append(f"- Total de séries: {resumo_estrada.n_series}")
        relatorio.append(f"- Passageiros por série: {resumo_estrada.n_passageiros}")
        relatorio.append(f"- Faixa de valores: {resumo_estrada.min_val}–{resumo_estrada.max_val}")
        relatorio.append(f"- Média geral: {resumo_estrada.media:.2f}")
        relatorio.append(f"- Desvio padrão global: {resumo_estrada.desvio:.2f}")
        relatorio.append(f"- Regime global: {resumo_estrada.regime_global}\n")

    # k*
    if resumo_k_global:
        relatorio.append("🔭 **k e k***")
        relatorio.append(f"- k atual (última série): {resumo_k_global.k_atual}")
        relatorio.append(f"- k*: {resumo_k_global.k_star*100:.1f}%")
        relatorio.append(f"- Estado k*: {resumo_k_global.estado_k}")
        relatorio.append(f"- Regime local: {resumo_k_global.regime_local}\n")

    # Ruído
    if resumo_ruido:
        relatorio.append("🌪️ **Ruído Condicional**")
        relatorio.append(f"- Ruído A aplicado: {resumo_ruido.ruido_inicial:.2f}")
        relatorio.append(f"- Ruído B aplicado: {resumo_ruido.ruido_final:.2f}")
        relatorio.append(f"- % de pontos ajustados: {resumo_ruido.pct_pontos_ajustados:.1f}%\n")

    # QDS
    if resumo_qds:
        relatorio.append("📊 **Qualidade Dinâmica de Série (QDS REAL)**")
        relatorio.append(f"- QDS médio: {resumo_qds.qds_medio:.4f}")
        relatorio.append(f"- QDS mínimo: {resumo_qds.qds_min:.4f}")
        relatorio.append(f"- QDS máximo: {resumo_qds.qds_max:.4f}")
        relatorio.append(
            f"- % PREMIUM/BOM/REGULAR/RUIM: "
            f"{resumo_qds.pct_premium:.1f}% / "
            f"{resumo_qds.pct_bom:.1f}% / "
            f"{resumo_qds.pct_regular:.1f}% / "
            f"{resumo_qds.pct_ruim:.1f}%\n"
        )

    # Backtest REAL
    if resumo_bt:
        relatorio.append("🧪 **Backtest REAL**")
        relatorio.append(f"- Janelas avaliadas: {resumo_bt.n_janelas}")
        relatorio.append(f"- Acertos totais: {resumo_bt.acertos_totais}")
        relatorio.append(f"- Média por série: {resumo_bt.acertos_por_serie:.3f}")
        relatorio.append(f"- Hit rate: {resumo_bt.hit_rate*100:.2f}%\n")

    # Monte Carlo REAL
    if resumo_mc:
        relatorio.append("🎲 **Monte Carlo REAL**")
        relatorio.append(f"- Simulações: {resumo_mc.n_simulacoes}")
        relatorio.append(f"- Média de acertos: {resumo_mc.media_acertos:.4f}")
        relatorio.append(f"- Desvio padrão (acertos): {resumo_mc.desvio_acertos:.4f}")
        relatorio.append(f"- Melhor simulação média: {resumo_mc.melhor_serie_media:.4f}\n")

    # Previsão final
    if previsao_final:
        relatorio.append("🎯 **Previsão Final TURBO++ ULTRA (V15.5)**")
        relatorio.append(" ".join(str(x) for x in previsao_final) + "\n")

    # Expectativa de acertos
    if resumo_k_global:
        relatorio.append("📌 **Expectativa de Acertos (por ambiente k*)**")
        if resumo_k_global.estado_k == "estavel":
            relatorio.append("- Ambiente estável → 2–4 acertos típicos.")
        elif resumo_k_global.estado_k == "atencao":
            relatorio.append("- Ambiente de pré-ruptura → 1–3 acertos típicos.")
        else:
            relatorio.append("- Ambiente crítico → 0–2 acertos típicos.")
        relatorio.append("")

    relatorio_texto = "\n".join(relatorio)

    st.text_area(
        "Relatório completo (copie e cole no ChatGPT):",
        relatorio_texto,
        height=600,
    )

    st.success("Relatório Final — AIQ Bridge gerado com sucesso!")
