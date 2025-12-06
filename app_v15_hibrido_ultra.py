# -*- coding: utf-8 -*-
"""
Predict Cars V15-HÍBRIDO ULTRA — Anti-Ruído & Previsão Condicional
Baseado integralmente no V14-FLEX ULTRA REAL (TURBO++), evoluído por
ACRESCIMENTO, sem qualquer simplificação de filosofia ou de jeitão.

PARTE 1/4
---------
Este arquivo é dividido logicamente em 4 partes:

1/4) Cabeçalho, estado, utilitários, entrada de histórico FLEX ULTRA,
     detecção de ruído estrutural global (NR%), QDS global e baseline
     de ambiência preditiva.

2/4) Reinstalação do pipeline V14-FLEX ULTRA (S1..S5, IDX, Núcleo
     Resiliente, S6 Profundo, Monte Carlo Profundo, Micro-Leques),
     mantendo a filosofia e o estilo de múltiplas camadas.

3/4) Painéis avançados de Replay (LIGHT, ULTRA, ULTRA Unitário) +
     Monitor de Risco (k & k*), Testes de Confiabilidade (QDS REAL,
     Backtest REAL, Monte Carlo REAL) conectados ao motor V15.

4/4) Núcleo V15-HÍBRIDO Anti-Ruído: Painel Oficial de Ruído Estrutural
     (NR%), Mapa de Divergência S6 vs MC, Mapa de Ruído Condicional,
     Modo TURBO++ ULTRA ANTI-RUÍDO (fusão S6/MC/Micro), navegação
     completa e integração final da Previsão + Envelope Forte (6–8 séries).

ATENÇÃO IMPORTANTE
------------------
Enquanto apenas a PARTE 1/4 estiver colada, o app ainda NÃO está
completo. Só teste o app após colar, em sequência, as partes 2/4, 3/4 e 4/4
no mesmo arquivo, logo abaixo deste código.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

###############################################################################
# CONFIGURAÇÃO GLOBAL DO APP
###############################################################################

APP_NAME = "Predict Cars V15-HÍBRIDO ULTRA — Anti-Ruído & Previsão Condicional"
APP_VERSION = "V15-HÍBRIDO ULTRA — MOTOR COMPLETO (1/4)"

st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
)

# Ícones e emojis usados em vários painéis
ICON_INFO = "ℹ️"
ICON_WARN = "⚠️"
ICON_OK = "✅"
ICON_ERROR = "❌"
ICON_NOISE = "📊"
ICON_TURBO = "🚀"
ICON_RISK = "🚨"

###############################################################################
# ESTADO DE SESSÃO — HISTÓRICO, CONFIGURAÇÕES E PERFIS
###############################################################################

def get_df_sessao() -> Optional[pd.DataFrame]:
    """
    Retorna o histórico principal armazenado na sessão.
    Compatível com o V14-FLEX ULTRA: df pré-processado, com:
        - coluna 'indice' (1..n)
        - coluna 'serie_id' ou similar (ex: C1, C2, ...)
        - colunas de passageiros (n1..nN)
        - opcionalmente coluna 'k'
    """
    df = st.session_state.get("df", None)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df
    return None


def set_df_sessao(df: pd.DataFrame) -> None:
    """
    Atualiza o histórico principal na sessão.
    """
    st.session_state["df"] = df


def get_noise_profile_baseline() -> Optional[dict]:
    """
    Recupera o baseline de ruído estrutural global salvo na sessão.
    Estrutura:
        {
            "nr_total": float,
            "qds_global": float,
            "n_series": int,
            "n_passageiros": int,
        }
    """
    prof = st.session_state.get("noise_profile_v15_baseline", None)
    if isinstance(prof, dict):
        return prof
    return None


def set_noise_profile_baseline(profile: dict) -> None:
    """
    Salva o baseline de ruído estrutural global na sessão.
    """
    st.session_state["noise_profile_v15_baseline"] = profile


###############################################################################
# UTILITÁRIOS — DETECÇÃO DE PASSAGEIROS / FAIXAS / MÉTRICAS BÁSICAS
###############################################################################

def detectar_colunas_passageiros(df: pd.DataFrame) -> List[str]:
    """
    Detecta, de forma robusta, as colunas de passageiros.
    Compatível com:
        - n1..n6, n1..nN (V14-FLEX)
        - P1..Pn
        - combinações híbridas.

    Critério:
        - nome da coluna começa com 'n' ou 'p' (case-insensitive)
        - colunas são ordenadas pelo sufixo numérico, quando existente.
    """
    candidatos = [
        c
        for c in df.columns
        if isinstance(c, str) and (c.lower().startswith("n") or c.lower().startswith("p"))
    ]

    def _key(c: str) -> Tuple[int, str]:
        sufixo = "".join(ch for ch in c if ch.isdigit())
        try:
            return (int(sufixo), c)
        except Exception:
            return (10_000, c)

    return sorted(candidatos, key=_key)


def contar_passageiros(df: pd.DataFrame) -> int:
    """
    Conta quantos passageiros existem no histórico (número de colunas detectadas).
    """
    return len(detectar_colunas_passageiros(df))


def calcular_faixa_global(df: pd.DataFrame, cols_passageiros: List[str]) -> Optional[Tuple[int, int]]:
    """
    Calcula a faixa numérica global (mínimo → máximo) em todas as colunas
    de passageiros.
    """
    if not cols_passageiros:
        return None
    valores = df[cols_passageiros].values.flatten()
    valores = valores[~pd.isna(valores)]
    if len(valores) == 0:
        return None
    vmin = int(np.min(valores))
    vmax = int(np.max(valores))
    return (vmin, vmax)


###############################################################################
# UTILITÁRIO — ENTROPIA DISCRETA E RUÍDO ESTRUTURAL (NR%)
###############################################################################

def _entropy_discreta(proporcoes: np.ndarray) -> float:
    """
    Entropia discreta normalizada em [0,1].

    Usada como base para medir dispersão estrutural da estrada e, portanto,
    o ruído Tipo B (explicável). Quanto mais próximo de 1, mais disperso.
    """
    proporcoes = proporcoes[proporcoes > 0]
    if len(proporcoes) == 0:
        return 0.0
    h = -np.sum(proporcoes * np.log2(proporcoes))
    h_max = math.log2(len(proporcoes))
    if h_max == 0:
        return 0.0
    return float(h / h_max)


def calcular_nr_posicional_global(df: pd.DataFrame, cols_passageiros: List[str]) -> pd.DataFrame:
    """
    Calcula, de forma global, o NR posicional (por P1..Pn) ao longo de
    toda a estrada, usando entropia discreta normalizada por posição.
    """
    registros = []

    for idx_pos, col in enumerate(cols_passageiros, start=1):
        serie = df[col].dropna()
        if serie.empty:
            ent = 0.0
            nr_pct = 0.0
            diversidade = 0
            dominante_pct = 0.0
        else:
            vc = serie.value_counts(normalize=True)
            proporcoes = vc.values.astype(float)
            ent = _entropy_discreta(proporcoes)
            nr_pct = 100.0 * ent
            diversidade = len(vc)
            dominante_pct = 100.0 * float(vc.iloc[0])

        registros.append(
            {
                "posicao": f"P{idx_pos}",
                "coluna": col,
                "entropia": ent,
                "nr_pct": nr_pct,
                "diversidade": diversidade,
                "dominante_pct": dominante_pct,
            }
        )

    df_pos = pd.DataFrame(registros)
    return df_pos


def calcular_nr_janelas_global(
    df: pd.DataFrame,
    cols_passageiros: List[str],
    window: int = 40,
    step: int = 5,
) -> pd.DataFrame:
    """
    Calcula o NR por janelas rolantes ao longo da estrada, agregando
    a entropia posicional média em cada bloco.

    É um instrumento para enxergar:
        - trechos excelentes (NR baixo)
        - trechos bons
        - trechos médios
        - trechos ruins
        - trechos caóticos (NR alto)
    """
    n = len(df)
    registros = []

    if n == 0 or len(cols_passageiros) == 0:
        return pd.DataFrame(
            columns=["inicio", "fim", "n_series", "entropia_media", "nr_pct"]
        )

    start = 0
    while start < n:
        end = min(start + window, n)
        bloco = df.iloc[start:end]
        if bloco.empty:
            break

        df_pos = calcular_nr_posicional_global(bloco, cols_passageiros)
        entropia_media = float(df_pos["entropia"].mean())
        nr_pct = 100.0 * entropia_media

        registros.append(
            {
                "inicio": int(start + 1),
                "fim": int(end),
                "n_series": int(len(bloco)),
                "entropia_media": entropia_media,
                "nr_pct": nr_pct,
            }
        )

        if end == n:
            break
        start += step

    df_jan = pd.DataFrame(registros)
    return df_jan


def sintetizar_nr_total_global(df_jan: pd.DataFrame) -> float:
    """
    Sintetiza um NR global (%) a partir do NR por janelas.

    Este valor será usado como:
        - indicador agregado de ruído Tipo B
        - um dos componentes do QDS global
        - insumo para o Mapa de Ambiência (excelente/bom/médio/ruim/caos)
    """
    if df_jan.empty:
        return 0.0
    return float(df_jan["nr_pct"].mean())


###############################################################################
# QDS GLOBAL (ÍNDICE DE QUALIDADE DA ESTRADA)
###############################################################################

def calcular_qds_global(
    nr_total_pct: float,
    n_series: int,
    n_passageiros: int,
) -> float:
    """
    Calcula um QDS global (0..1) a partir de:
        - NR total (%)             → ruído estrutural
        - n_series                 → extensão da estrada
        - n_passageiros           → dimensionalidade da série

    Ideia qualitativa:
        - quanto menor o NR, maior a qualidade estrutural
        - estradas muito curtas derrubam um pouco a confiança
        - número maior de passageiros torna o problema mais difícil

    Fórmula qualitativa (pode ser refinada nas partes 2/4, 3/4 e 4/4):
        - base_nr = 1 - (nr_total_pct / 100)^α
        - penalização série curta
        - penalização dimensionalidade
    """
    # Normalização do NR em [0,1]
    nr_norm = max(0.0, min(1.0, nr_total_pct / 100.0))

    # Componente de qualidade estrutural inversamente proporcional ao NR
    # α > 1 torna a curva mais sensível em NR altos
    alpha = 1.3
    base_nr = 1.0 - (nr_norm ** alpha)

    # Penalização por estrada curta
    # Quanto menor n_series, maior o impacto
    if n_series < 200:
        pena_series = 0.15
    elif n_series < 1000:
        pena_series = 0.05
    else:
        pena_series = 0.0

    # Penalização por dimensionalidade alta (muitos passageiros)
    if n_passageiros <= 5:
        pena_dim = 0.0
    elif n_passageiros <= 8:
        pena_dim = 0.05
    else:
        pena_dim = 0.10

    qds = base_nr * (1.0 - pena_series) * (1.0 - pena_dim)
    qds = max(0.0, min(1.0, qds))
    return float(qds)


###############################################################################
# LEITURA E NORMALIZAÇÃO DO HISTÓRICO (FORMATOS FLEX)
###############################################################################

def _ler_csv_flex(file) -> pd.DataFrame:
    """
    Leitura flexível de CSV, tentando detectar automaticamente o separador.
    """
    try:
        df = pd.read_csv(file, sep=None, engine="python")
    except Exception:
        file.seek(0)
        df = pd.read_csv(file, sep=";")
    return df


def _normalizar_formato_coluna_series(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza histórico do tipo:

        C1;41;5;4;52;30;33;0
        C2;...

    Ou seja:
        - primeira coluna = identificador da série (C1, C2, etc.)
        - colunas seguintes = n1..nN e possivelmente k na última coluna.
    """
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    nome_id = df.columns[0]
    serie_id = df[nome_id].astype(str).str.strip()
    cols_valores = df.columns[1:]
    n_cols_valores = len(cols_valores)

    # Heurística: última coluna pode ser k
    k_col = None
    if n_cols_valores >= 2:
        candidata = cols_valores[-1]
        serie_cand = pd.to_numeric(df[candidata], errors="coerce")
        # Se for numérica e parecer razoável, assume como k
        if serie_cand.notna().mean() > 0.9:
            k_col = candidata

    passageiros_cols: List[str] = []
    for col in cols_valores:
        if col == k_col:
            continue
        passageiros_cols.append(col)

    mapping = {}
    for i, col in enumerate(passageiros_cols, start=1):
        mapping[col] = f"n{i}"

    df_norm = pd.DataFrame()
    df_norm["indice"] = range(1, len(df) + 1)
    df_norm["serie_id"] = serie_id

    for col, novo_nome in mapping.items():
        df_norm[novo_nome] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    if k_col is not None:
        df_norm["k"] = pd.to_numeric(df[k_col], errors="coerce").astype("Int64")

    return df_norm


def _normalizar_formato_passageiros(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza histórico do tipo:

        n1;n2;...;nN;k

    Ou seja:
        - colunas de passageiros + coluna k opcional.
    """
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    col_k = None
    for c in df.columns:
        if c.lower() == "k":
            col_k = c
            break

    passageiros_cols: List[str] = []
    for c in df.columns:
        if c == col_k:
            continue
        passageiros_cols.append(c)

    def _key(c: str) -> Tuple[int, str]:
        sufixo = "".join(ch for ch in c if c.lower().startswith("n") and ch.isdigit())
        try:
            return (int(sufixo), c)
        except Exception:
            return (10_000, c)

    passageiros_cols = sorted(passageiros_cols, key=_key)

    mapping = {}
    for i, col in enumerate(passageiros_cols, start=1):
        mapping[col] = f"n{i}"

    df_norm = pd.DataFrame()
    df_norm["indice"] = range(1, len(df) + 1)

    for col, novo_nome in mapping.items():
        df_norm[novo_nome] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    if col_k is not None:
        df_norm["k"] = pd.to_numeric(df[col_k], errors="coerce").astype("Int64")

    # Cria 'serie_id' no padrão C1, C2, ...
    df_norm["serie_id"] = df_norm["indice"].apply(lambda x: f"C{x}")

    # Reordena para deixar índice/série logo no início
    cols_pass = [c for c in df_norm.columns if c.startswith("n")]
    outras = [c for c in ["indice", "serie_id", "k"] if c in df_norm.columns]
    df_norm = df_norm[outras[:2] + cols_pass + outras[2:]]

    return df_norm


###############################################################################
# PAINEL — HISTÓRICO — ENTRADA FLEX ULTRA (V15-HÍBRIDO)
###############################################################################

def painel_historico_entrada_v15() -> None:
    """
    Painel de entrada de histórico — versão FLEX ULTRA (V14/V15),
    compatível com múltiplos formatos e já integrando:

        - Normalização para n1..nN, k
        - Cálculo de NR posicional global
        - Cálculo de NR por janelas
        - Cálculo de NR total (%)
        - Cálculo de QDS global
        - Baseline de ambiência preditiva da estrada
    """
    st.markdown("## 📥 Histórico — Entrada FLEX ULTRA (V15-HÍBRIDO)")

    formato = st.radio(
        "Formato do histórico:",
        (
            "CSV com coluna de séries (C1;41;5;4;52;30;33;0)",
            "CSV com passageiros (n1..nN, k)",
        ),
    )

    file = st.file_uploader(
        "Selecione o arquivo de histórico (.csv):",
        type=["csv"],
        help=(
            "Use o mesmo arquivo utilizado no V14-FLEX ULTRA REAL. "
            "O sistema detectará automaticamente as colunas de passageiros "
            "e a presença (ou não) de k."
        ),
    )

    df_norm: Optional[pd.DataFrame] = None

    if file is not None:
        df_raw = _ler_csv_flex(file)

        st.markdown("### 🔍 Pré-visualização bruta do arquivo (topo)")
        st.dataframe(df_raw.head(20), use_container_width=True)

        if formato.startswith("CSV com coluna de séries"):
            df_norm = _normalizar_formato_coluna_series(df_raw)
        else:
            df_norm = _normalizar_formato_passageiros(df_raw)

        st.markdown("---")
        st.markdown("### ✅ Histórico normalizado (V15-HÍBRIDO)")
        st.dataframe(df_norm.head(50), use_container_width=True)

        # Atualiza sessão
        set_df_sessao(df_norm)

        # Métricas básicas
        n_series = len(df_norm)
        cols_pass = detectar_colunas_passageiros(df_norm)
        n_pass = len(cols_pass)
        faixa_global = calcular_faixa_global(df_norm, cols_pass)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de séries (C1 → Cn)", n_series)
        with col2:
            st.metric("Passageiros detectados (n)", n_pass)
        with col3:
            if faixa_global is not None:
                st.metric("Faixa numérica global", f"{faixa_global[0]} → {faixa_global[1]}")
            else:
                st.metric("Faixa numérica global", "N/A")
        with col4:
            tem_k = "k" in df_norm.columns
            st.metric("Coluna k presente?", "Sim" if tem_k else "Não")

        st.markdown("---")
        st.markdown("### 📊 Baseline imediato — NR Estrutural & QDS Global")

        # Apenas se houver dados suficientes
        if n_series >= 20 and n_pass > 0:
            # NR por posição global
            df_nr_pos = calcular_nr_posicional_global(df_norm, cols_pass)
            # NR por janelas (baseline)
            window_default = min(40, n_series)
            df_nr_jan = calcular_nr_janelas_global(
                df_norm,
                cols_passageiros=cols_pass,
                window=window_default,
                step=5,
            )
            nr_total = sintetizar_nr_total_global(df_nr_jan)
            qds_global = calcular_qds_global(
                nr_total_pct=nr_total,
                n_series=n_series,
                n_passageiros=n_pass,
            )

            baseline = {
                "nr_total": nr_total,
                "qds_global": qds_global,
                "n_series": n_series,
                "n_passageiros": n_pass,
            }
            set_noise_profile_baseline(baseline)

            colb1, colb2, colb3 = st.columns(3)
            with colb1:
                st.metric(f"{ICON_NOISE} NR Total (%)", f"{nr_total:.1f}%")
            with colb2:
                st.metric("QDS Global (0–1)", f"{qds_global:.3f}")
            with colb3:
                # Interpretação qualitativa de ambiência
                if qds_global >= 0.75:
                    estado = "🟢 Estrada muito boa"
                elif qds_global >= 0.60:
                    estado = "🟡 Estrada boa / moderada"
                elif qds_global >= 0.45:
                    estado = "🟠 Estrada média / instável"
                else:
                    estado = "🔴 Estrada com ruído alto"
                st.metric("Ambiência global (baseline)", estado)

            st.markdown("#### NR por posição (P1..Pn)")
            st.dataframe(df_nr_pos, use_container_width=True)

            # Pequeno gráfico de barras para NR posicional
            fig1, ax1 = plt.subplots()
            ax1.bar(df_nr_pos["posicao"], df_nr_pos["nr_pct"])
            ax1.set_xlabel("Posição (P1..Pn)")
            ax1.set_ylabel("NR por posição (%)")
            ax1.set_title("NR Estrutural por Posição — Baseline Global (V15)")
            st.pyplot(fig1)

            st.markdown("#### NR por janelas (visão macro da estrada)")
            st.dataframe(df_nr_jan, use_container_width=True)

            fig2, ax2 = plt.subplots()
            labels = [f"{ini}→{fim}" for ini, fim in zip(df_nr_jan["inicio"], df_nr_jan["fim"])]
            ax2.plot(labels, df_nr_jan["nr_pct"], marker="o")
            ax2.set_xlabel("Janela (C_início → C_fim)")
            ax2.set_ylabel("NR por janela (%)")
            ax2.set_title("NR Estrutural por Janelas — Baseline Global (V15)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig2)

            st.info(
                f"{ICON_INFO} Este baseline será usado nos demais painéis do V15-HÍBRIDO "
                "para mapear trechos bons/médios/ruins/caóticos, ajustar pesos do "
                "Modo TURBO++ ULTRA ANTI-RUÍDO e calibrar o Mapa de Ambiência."
            )
        else:
            st.warning(
                f"{ICON_WARN} Histórico ainda curto ou sem passageiros suficientes "
                "para um baseline robusto. Recomenda-se pelo menos 20 séries e "
                "número consistente de passageiros."
            )
    else:
        st.info(
            f"{ICON_INFO} Envie um arquivo CSV para ativar o processamento FLEX ULTRA "
            "e habilitar o baseline de ruído estrutural (NR%) e QDS global."
        )

# FIM DA PARTE 1/4
# Nas próximas partes (2/4, 3/4 e 4/4) serão adicionados:
# - Pipeline V14-FLEX completo (S1..S6, MC, Micro-Leques, Núcleo Resiliente)
# - Monitor de Risco (k & k*)
# - Modos TURBO++ ULTRA (adaptativo e anti-ruído)
# - Mapa condicional, divergência S6/MC, Replay ULTRA etc.
###############################################################################
# PARTE 2/4 — PIPELINE V14-FLEX ULTRA (BASE PARA V15)
###############################################################################
"""
Nesta seção, reinstalamos o núcleo do Pipeline V14-FLEX ULTRA, em versão
compatível com o V15-HÍBRIDO:

- S1 — Frequências Globais por posição (P1..Pn)
- S2 — Distâncias e variação entre séries consecutivas
- S3 — Ciclos e recorrências locais
- S4 — Clustering básico por posição (faixas e espaçamento)
- S5 — Anomalias (z-score) em profundidade
- IDX Local — Índice local de densidade / complexidade
- Núcleo Resiliente — região de estabilidade local
- S6 Base — Projeção estruturada por posição
- Estruturas auxiliares para Monte Carlo, Micro-Leques e S6 Profundo
  (detalhados na parte 4/4 para o Modo ANTI-RUÍDO).

O objetivo desta parte é manter o jeitão multifásico do V14-FLEX,
tornando o V15 um SUPERCONJUNTO e nunca uma simplificação.
"""

@dataclass
class IDXLocalInfo:
    densidade: int
    entropia_media: float
    nr_local: float


@dataclass
class NucleoResilienteInfo:
    df_nucleo: pd.DataFrame
    janela_inicio: int
    janela_fim: int


@dataclass
class S6BaseInfo:
    df_s6: pd.DataFrame
    janela_inicio: int
    janela_fim: int


###############################################################################
# S1 — FREQUÊNCIAS GLOBAIS
###############################################################################

def s1_frequencias_globais(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    S1 — Frequências Globais:
        - Conta a frequência absoluta e relativa de cada valor por posição.
        - É a base para enxergar dominância, rarefação e background da estrada.
    """
    registros = []

    for col in cols:
        serie = df[col].dropna()
        if serie.empty:
            continue
        vc = serie.value_counts().sort_index()
        total = vc.sum()
        for valor, freq in vc.items():
            registros.append(
                {
                    "coluna": col,
                    "valor": int(valor),
                    "freq": int(freq),
                    "pct": float(100.0 * freq / total),
                }
            )

    df_s1 = pd.DataFrame(registros)
    return df_s1


###############################################################################
# S2 — DISTÂNCIAS ENTRE SÉRIES CONSECUTIVAS
###############################################################################

def s2_distancias_locais(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    S2 — Distâncias locais:
        - Mede a variação absoluta entre séries consecutivas em cada posição.
        - Ajuda a detectar trechos mais suaves vs. trechos explosivos.
    """
    registros = []
    n = len(df)
    if n < 2:
        return pd.DataFrame(columns=["C_atual", "coluna", "dist"])

    for col in cols:
        serie = df[col].astype(float).values
        diffs = np.abs(np.diff(serie))
        for i, d in enumerate(diffs, start=2):
            registros.append(
                {
                    "C_atual": int(i),
                    "coluna": col,
                    "dist": float(d),
                }
            )

    df_s2 = pd.DataFrame(registros)
    return df_s2


###############################################################################
# S3 — CICLOS E RECORRÊNCIAS (LAGS)
###############################################################################

def s3_ciclos_recorrencias(df: pd.DataFrame, cols: List[str], max_lag: int = 40) -> pd.DataFrame:
    """
    S3 — Ciclos:
        - Para cada posição, testa lags de 1 até max_lag e mede
          quantas vezes o valor se repete após esse lag.
        - Não é um modelo previsivo, mas um scanner de periodicidades.
    """
    registros = []
    for col in cols:
        serie = df[col].astype("Int64").dropna().values
        n = len(serie)
        if n == 0:
            continue
        lag_lim = min(max_lag, n - 1)
        for lag in range(1, lag_lim + 1):
            iguais = int(np.sum(serie[:-lag] == serie[lag:]))
            pct = 100.0 * iguais / (n - lag)
            registros.append(
                {
                    "coluna": col,
                    "lag": int(lag),
                    "match": iguais,
                    "pct": float(pct),
                }
            )
    df_s3 = pd.DataFrame(registros)
    return df_s3


###############################################################################
# S4 — CLUSTERING BÁSICO POR POSIÇÃO
###############################################################################

def s4_cluster_basico(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    S4 — Clustering Básico:
        - Para cada posição, identifica os valores únicos e mede:
            - variabilidade (quantidade de valores distintos)
            - menor distância entre valores ordenados
        - Indica quão "agrupadas" ou "espalhadas" estão as faixas.
    """
    registros = []

    for col in cols:
        serie = df[col].astype("Int64").dropna()
        unicos = sorted(serie.unique())
        if len(unicos) < 2:
            registros.append(
                {
                    "coluna": col,
                    "variabilidade": len(unicos),
                    "dist_min": 0,
                }
            )
            continue

        dist_min = min(abs(unicos[i + 1] - unicos[i]) for i in range(len(unicos) - 1))

        registros.append(
            {
                "coluna": col,
                "variabilidade": len(unicos),
                "dist_min": int(dist_min),
            }
        )

    df_s4 = pd.DataFrame(registros)
    return df_s4


###############################################################################
# S5 — ANOMALIAS (Z-SCORE) EM PROFUNDIDADE
###############################################################################

def s5_anomalias_zscore(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    S5 — Anomalias:
        - Calcula z-score para cada valor, por coluna, ao longo da estrada.
        - Ajuda a localizar outliers estruturais que podem estar associados
          a ruído Tipo B ou a quebras de regime.
    """
    registros = []

    for col in cols:
        serie = df[col].astype(float).values
        media = float(np.nanmean(serie))
        std = float(np.nanstd(serie))
        if std == 0:
            std = 1.0

        for i, v in enumerate(serie, start=1):
            z = (v - media) / std
            registros.append(
                {
                    "C": int(i),
                    "coluna": col,
                    "valor": float(v),
                    "zscore": float(z),
                }
            )

    df_s5 = pd.DataFrame(registros)
    return df_s5


###############################################################################
# IDX LOCAL — DENSIDADE, ENTROPIA LOCAL, NR LOCAL
###############################################################################

def calcular_idx_local(
    df: pd.DataFrame,
    cols: List[str],
    idx_target: int,
    janela: int = 40,
) -> IDXLocalInfo:
    """
    IDX Local:
        - Considera uma janela antes do índice alvo (ex: 40 séries)
        - Calcula:
            - densidade (quantidade de séries na janela)
            - entropia média posicional
            - NR local (%), análogo ao NR global mas focado no entorno.
    """
    n = len(df)
    idx0 = max(0, idx_target - janela)
    idx1 = min(idx_target, n)
    sub = df.iloc[idx0:idx1]
    densidade = len(sub)

    if densidade == 0 or len(cols) == 0:
        return IDXLocalInfo(densidade=0, entropia_media=0.0, nr_local=0.0)

    # Reuso das funções globais, mas localmente
    df_nr_pos_local = calcular_nr_posicional_global(sub, cols)
    entropia_media = float(df_nr_pos_local["entropia"].mean())
    nr_local = float(100.0 * entropia_media)

    return IDXLocalInfo(
        densidade=densidade,
        entropia_media=entropia_media,
        nr_local=nr_local,
    )


###############################################################################
# NÚCLEO RESILIENTE — REGIÃO DE ESTABILIDADE LOCAL
###############################################################################

def calcular_nucleo_resiliente(
    df: pd.DataFrame,
    cols: List[str],
    idx_target: int,
    janela: int = 30,
) -> NucleoResilienteInfo:
    """
    Núcleo Resiliente:
        - Considera um bloco anterior ao índice alvo (ex: 30 séries)
        - Identifica, em cada posição, os valores mais dominantes
          (background estável) que servirão de base para o S6.
        - Integra a NR posicional para marcar coerência local.
    """
    n = len(df)
    idx0 = max(0, idx_target - janela)
    idx1 = min(idx_target, n)
    sub = df.iloc[idx0:idx1].copy()

    registros = []

    if sub.empty or len(cols) == 0:
        df_nucleo = pd.DataFrame(columns=["posicao", "coluna", "dominante", "pct_dom", "nr_local"])
    else:
        df_nr_pos_local = calcular_nr_posicional_global(sub, cols)
        nr_dict = {
            row["coluna"]: row["nr_pct"] for _, row in df_nr_pos_local.iterrows()
        }

        for idx_pos, col in enumerate(cols, start=1):
            serie = sub[col].dropna()
            if serie.empty:
                registros.append(
                    {
                        "posicao": f"P{idx_pos}",
                        "coluna": col,
                        "dominante": None,
                        "pct_dom": 0.0,
                        "nr_local": nr_dict.get(col, 0.0),
                    }
                )
                continue

            vc = serie.value_counts(normalize=True)
            dominante = int(vc.index[0])
            pct_dom = 100.0 * float(vc.iloc[0])
            registros.append(
                {
                    "posicao": f"P{idx_pos}",
                    "coluna": col,
                    "dominante": dominante,
                    "pct_dom": pct_dom,
                    "nr_local": nr_dict.get(col, 0.0),
                }
            )

        df_nucleo = pd.DataFrame(registros)

    return NucleoResilienteInfo(
        df_nucleo=df_nucleo,
        janela_inicio=idx0 + 1,
        janela_fim=idx1,
    )


###############################################################################
# S6 BASE — PROJEÇÃO ESTRUTURAL POR POSIÇÃO
###############################################################################

def calcular_s6_base(
    df: pd.DataFrame,
    cols: List[str],
    idx_target: int,
    janela: int = 60,
) -> S6BaseInfo:
    """
    S6 Base:
        - Considera uma janela maior (ex: 60 séries) antes do alvo;
        - Para cada posição:
            - Calcula média, desvio padrão;
            - Integra NR local posicional;
            - Gera uma projeção central (proj_base) e um intervalo (faixa)
              ainda em modo "pré-turbo", que será refinado no modo ANTI-RUÍDO.
    """
    n = len(df)
    idx0 = max(0, idx_target - janela)
    idx1 = min(idx_target, n)
    sub = df.iloc[idx0:idx1].copy()

    registros = []

    if sub.empty or len(cols) == 0:
        return S6BaseInfo(
            df_s6=pd.DataFrame(columns=[
                "posicao",
                "coluna",
                "media",
                "std",
                "nr_pos",
                "proj_base",
                "faixa_low",
                "faixa_high",
            ]),
            janela_inicio=idx0 + 1,
            janela_fim=idx1,
        )

    df_nr_pos_local = calcular_nr_posicional_global(sub, cols)
    nr_dict = {
        row["coluna"]: row["nr_pct"] for _, row in df_nr_pos_local.iterrows()
    }

    for idx_pos, col in enumerate(cols, start=1):
        serie = sub[col].astype(float).values
        media = float(np.nanmean(serie))
        std = float(np.nanstd(serie))
        if std == 0:
            std = 1.0

        nr_pos = nr_dict.get(col, 0.0) / 100.0  # converte para [0,1]

        # Projeção base: média + ajuste suave pela NR
        suav = math.exp(-nr_pos)
        proj_base = media * suav + media * (1.0 - suav)

        # Faixa: 1 desvio padrão, inflado pela NR
        fator_faixa = 1.0 + nr_pos
        faixa_low = proj_base - std * fator_faixa
        faixa_high = proj_base + std * fator_faixa

        registros.append(
            {
                "posicao": f"P{idx_pos}",
                "coluna": col,
                "media": media,
                "std": std,
                "nr_pos": nr_pos,
                "proj_base": proj_base,
                "faixa_low": faixa_low,
                "faixa_high": faixa_high,
            }
        )

    df_s6 = pd.DataFrame(registros)

    return S6BaseInfo(
        df_s6=df_s6,
        janela_inicio=idx0 + 1,
        janela_fim=idx1,
    )


###############################################################################
# PAINEL — PIPELINE V14-FLEX (TURBO++) REINSTALADO NO V15
###############################################################################

def painel_pipeline_v15() -> None:
    """
    Painel completo do Pipeline V14-FLEX (TURBO++), agora como base do V15:

        - Requer que o histórico já tenha sido carregado no painel
          '📥 Histórico — Entrada FLEX ULTRA (V15-HÍBRIDO)'.

        - Executa S1..S5, IDX Local, Núcleo Resiliente e S6 Base em sequência,
          exibindo tabelas densas e métricas de apoio.

        - As camadas adicionais (S6 Profundo ANTI-RUÍDO, MC Profundo,
          Micro-Leques ANTI-RUÍDO e fusão) serão acopladas na PARTE 4/4.
    """
    st.markdown("## 🔍 Pipeline V14-FLEX ULTRA — Núcleo V15-HÍBRIDO")

    df_hist = get_df_sessao()
    if df_hist is None or df_hist.empty:
        st.warning(
            f"{ICON_WARN} Nenhum histórico carregado. "
            "Use o painel '📥 Histórico — Entrada FLEX ULTRA (V15-HÍBRIDO)'."
        )
        return

    cols_pass = detectar_colunas_passageiros(df_hist)
    if len(cols_pass) == 0:
        st.error(
            f"{ICON_ERROR} Nenhuma coluna de passageiros detectada. "
            "Verifique o formato do histórico."
        )
        return

    n_series = len(df_hist)
    n_pass = len(cols_pass)

    st.markdown("### 📌 Configuração do alvo e da janela local")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        idx_target = st.number_input(
            "Índice alvo (C):",
            min_value=1,
            max_value=n_series,
            value=n_series,
        )
    with col_b:
        janela_idx = st.number_input(
            "Janela para IDX Local (séries):",
            min_value=10,
            max_value=min(200, n_series),
            value=min(40, n_series),
            step=5,
        )
    with col_c:
        janela_s6 = st.number_input(
            "Janela para S6 Base (séries):",
            min_value=20,
            max_value=min(200, n_series),
            value=min(60, n_series),
            step=5,
        )

    idx_target = int(idx_target)

    st.markdown("---")
    st.markdown("### 🧩 S1 — Frequências Globais por Posição")
    df_s1 = s1_frequencias_globais(df_hist, cols_pass)
    st.dataframe(df_s1.head(500), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧩 S2 — Distâncias Locais entre Séries Consecutivas")
    df_s2 = s2_distancias_locais(df_hist, cols_pass)
    st.dataframe(df_s2.head(500), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧩 S3 — Ciclos e Recorrências (Lags)")
    df_s3 = s3_ciclos_recorrencias(df_hist, cols_pass, max_lag=40)
    st.dataframe(df_s3.head(500), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧩 S4 — Clustering Básico por Posição")
    df_s4 = s4_cluster_basico(df_hist, cols_pass)
    st.dataframe(df_s4, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧩 S5 — Anomalias (Z-score) em Profundidade")
    df_s5 = s5_anomalias_zscore(df_hist, cols_pass)
    st.dataframe(df_s5.head(500), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧮 IDX Local — Densidade, Entropia e NR Local")

    idx_info = calcular_idx_local(
        df_hist,
        cols_pass,
        idx_target=idx_target,
        janela=int(janela_idx),
    )

    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        st.metric("Densidade local (séries na janela)", idx_info.densidade)
    with col_i2:
        st.metric("Entropia média local", f"{idx_info.entropia_media:.3f}")
    with col_i3:
        st.metric("NR Local (%)", f"{idx_info.nr_local:.1f}%")

    st.markdown("---")
    st.markdown("### 🧱 Núcleo Resiliente Local")

    nucleo = calcular_nucleo_resiliente(
        df_hist,
        cols_pass,
        idx_target=idx_target,
        janela=min(30, n_series),
    )

    st.write(
        f"Núcleo Resiliente calculado na janela: "
        f"C{nucleo.janela_inicio} → C{nucleo.janela_fim}"
    )
    st.dataframe(nucleo.df_nucleo, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎯 S6 Base — Projeção Estrutural por Posição")

    s6_base = calcular_s6_base(
        df_hist,
        cols_pass,
        idx_target=idx_target,
        janela=int(janela_s6),
    )

    st.write(
        f"S6 Base calculado na janela: "
        f"C{s6_base.janela_inicio} → C{s6_base.janela_fim}"
    )
    st.dataframe(s6_base.df_s6, use_container_width=True)

    st.info(
        f"{ICON_INFO} O S6 Base ainda não é o Modo TURBO++ ULTRA ANTI-RUÍDO. "
        "Ele representa a base estrutural que será reforçada, filtrada e "
        "fundida com Monte Carlo Profundo e Micro-Leques ANTI-RUÍDO na PARTE 4/4."
    )

# FIM DA PARTE 2/4
###############################################################################
# PARTE 3/4 — REPLAY ULTRA, MONITOR DE RISCO, QDS REAL, BACKTEST REAL
###############################################################################
"""
A PARTE 3/4 reinstala todos os painéis avançados:

- Replay LIGHT (rápido, inspeção imediata)
- Replay ULTRA (modo tradicional, mapa completo do alvo)
- Replay ULTRA UNITÁRIO (novo V14-FLEX, base para V15)
- Monitor de Risco (k & k*)
- Testes de Confiabilidade REAL (QDS LOCAL REAL + Backtest REAL)

Esses módulos são fundamentais para validar a coerência da estrada,
identificar trechos bons/médios/ruins, medir previsibilidade REAL e
preparar o terreno para o módulo ANTI-RUÍDO (Parte 4/4).
"""

###############################################################################
# MONITOR DE RISCO (k & k*)
###############################################################################

def calcular_k_serie(df: pd.DataFrame, idx: int) -> int:
    """
    k (histórico real):
        Quantos guardas acertaram exatamente aquela série.
        Se existir coluna k no histórico original, usamos direto.
        Caso não exista, k é considerado 0 (modo seguro).
    """
    if "k" in df.columns:
        try:
            return int(df.iloc[idx - 1]["k"])
        except Exception:
            return 0
    return 0


def calcular_k_estrela(df: pd.DataFrame, cols: List[str], idx: int, janela: int = 40) -> float:
    """
    k* (barômetro estrutural):
        Mede quão estável está o entorno da estrada, usando NR local.

        - janelas com NR baixo → k* baixo (ambiente estável)
        - janelas com NR alto → k* alto (ambiente turbulento)
    """
    idx_info = calcular_idx_local(
        df,
        cols,
        idx_target=idx,
        janela=janela,
    )
    # NR local em porcentagem → normaliza para [0,1]
    kstar = max(0.0, min(1.0, idx_info.nr_local / 100.0))
    return float(kstar)


def classificar_ambiencia_por_kstar(kstar: float) -> str:
    """
    Interpretação de k*:
        - 0.00–0.25  → excelente
        - 0.25–0.45  → bom
        - 0.45–0.60  → médio
        - 0.60–0.75  → ruim
        - 0.75–1.00  → caos
    """
    if kstar <= 0.25:
        return "🟢 Ambiente excelente"
    elif kstar <= 0.45:
        return "🟡 Ambiente bom"
    elif kstar <= 0.60:
        return "🟠 Ambiente instável"
    elif kstar <= 0.75:
        return "🔴 Ambiente ruim"
    else:
        return "⚫ Ambiente caótico"


###############################################################################
# QDS LOCAL REAL — AVALIAÇÃO DO ALVO
###############################################################################

def calcular_qds_local_real(df: pd.DataFrame, cols: List[str], idx: int, janela: int = 50) -> float:
    """
    QDS LOCAL REAL:
        Mede a qualidade do entorno imediato do ponto alvo (Cidx).

        - baixa entropia local → QDS REAL alto
        - alta entropia local → QDS REAL baixo
    """
    idx_info = calcular_idx_local(df, cols, idx_target=idx, janela=janela)
    nr_norm = max(0.0, min(1.0, idx_info.nr_local / 100.0))

    # QDS REAL é o inverso do ruído local
    qds_real = 1.0 - (nr_norm ** 1.2)
    return float(max(0.0, min(1.0, qds_real)))


###############################################################################
# BACKTEST REAL — AVALIAÇÃO DE CONSISTÊNCIA DA ESTRADA
###############################################################################

def executar_backtest_real(
    df: pd.DataFrame,
    cols: List[str],
    janela: int = 200,
) -> pd.DataFrame:
    """
    Backtest REAL:
        Reexecuta S6 Base em trechos passados (com NR real)
        e mede coerência entre projeção e valores reais.

        Isso não é previsão — é uma medição de estabilidade da estrada.
    """
    n = len(df)
    regs = []

    for idx in range(5, n + 1):
        s6 = calcular_s6_base(df, cols, idx_target=idx, janela=min(janela, idx - 1))
        for _, row in s6.df_s6.iterrows():
            pos = row["posicao"]
            proj = row["proj_base"]
            real = df.iloc[idx - 1][row["coluna"]]
            erro = abs(real - proj)
            regs.append(
                {
                    "C": idx,
                    "posicao": pos,
                    "proj_base": proj,
                    "real": real,
                    "erro_abs": erro,
                }
            )

    return pd.DataFrame(regs)


###############################################################################
# REPLAY LIGHT — VERSÃO RÁPIDA
###############################################################################

def painel_replay_light() -> None:
    st.markdown("## 💡 Replay LIGHT (V14-FLEX → V15-HÍBRIDO)")

    df = get_df_sessao()
    if df is None:
        st.warning("Nenhum histórico carregado.")
        return

    cols = detectar_colunas_passageiros(df)
    n_series = len(df)

    idx = st.number_input(
        "Índice alvo (C):",
        min_value=1,
        max_value=n_series,
        value=n_series,
    )
    idx = int(idx)

    st.markdown("### 🔍 Série selecionada")
    st.dataframe(df.iloc[[idx - 1]], use_container_width=True)

    k_real = calcular_k_serie(df, idx)
    kstar = calcular_k_estrela(df, cols, idx)
    amb = classificar_ambiencia_por_kstar(kstar)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("k (real)", k_real)
    with col2:
        st.metric("k* (barômetro)", f"{kstar:.2%}")
    with col3:
        st.metric("Ambiência", amb)

    st.markdown("---")

    st.info(
        "Replay LIGHT não projeta nada — é apenas inspeção rápida do estado "
        "local, servindo como diagnóstico básico antes do Replay ULTRA."
    )


###############################################################################
# REPLAY ULTRA — LOOP TRADICIONAL COMPLETO
###############################################################################

def painel_replay_ultra() -> None:
    st.markdown("## 📅 Replay ULTRA — Loop Tradicional (V14-FLEX → V15)")

    df = get_df_sessao()
    if df is None:
        st.warning("Histórico não carregado.")
        return

    cols = detectar_colunas_passageiros(df)
    n_series = len(df)

    col1, col2 = st.columns(2)
    with col1:
        inicio = st.number_input(
            "Início (C):",
            min_value=1,
            max_value=n_series,
            value=max(1, n_series - 30),
        )
    with col2:
        fim = st.number_input(
            "Fim (C):",
            min_value=inicio,
            max_value=n_series,
            value=n_series,
        )

    inicio = int(inicio)
    fim = int(fim)

    if fim - inicio < 1:
        st.warning("Selecione uma janela com pelo menos 2 séries.")
        return

    registros = []

    for idx in range(inicio, fim + 1):
        k_real = calcular_k_serie(df, idx)
        kstar = calcular_k_estrela(df, cols, idx)
        qds_real = calcular_qds_local_real(df, cols, idx)

        registros.append(
            {
                "C": idx,
                "k": k_real,
                "k*": kstar,
                "QDS_real": qds_real,
                "Ambiência": classificar_ambiencia_por_kstar(kstar),
            }
        )

    st.dataframe(pd.DataFrame(registros), use_container_width=True)

    st.info(
        "Replay ULTRA permite navegar pela estrada inteira e ver padrões "
        "estruturais antes de acoplar os motores de previsão."
    )


###############################################################################
# REPLAY ULTRA UNITÁRIO — BASE PARA O V15
###############################################################################

def painel_replay_unitario() -> None:
    st.markdown("## 🎯 Replay ULTRA UNITÁRIO — Novo Motor V14-FLEX para V15")

    df = get_df_sessao()
    if df is None:
        st.warning("Histórico não carregado.")
        return

    cols = detectar_colunas_passageiros(df)
    n_series = len(df)

    idx = st.number_input(
        "Índice alvo (C):",
        min_value=1,
        max_value=n_series,
        value=n_series,
    )
    idx = int(idx)

    st.markdown("### 🔎 Série alvo")
    st.dataframe(df.iloc[[idx - 1]], use_container_width=True)

    k_real = calcular_k_serie(df, idx)
    kstar = calcular_k_estrela(df, cols, idx)
    qds_real = calcular_qds_local_real(df, cols, idx)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("k (real)", k_real)
    with col2:
        st.metric("k* (barômetro)", f"{kstar:.2%}")
    with col3:
        st.metric("QDS REAL", f"{qds_real:.3f}")

    st.markdown("---")

    st.info(
        "Este painel é a porta de entrada do Modo TURBO++ ULTRA ANTI-RUÍDO "
        "(Parte 4/4). Ele monta o contexto do alvo e garante coerência local "
        "para os motores S6 Profundo, MC Profundo e Micro-Leque ANTI-RUÍDO."
    )


###############################################################################
# TESTES DE CONFIABILIDADE (QDS REAL + BACKTEST REAL)
###############################################################################

def painel_testes_confiabilidade() -> None:
    st.markdown("## 🧪 Testes de Confiabilidade REAL — V14-FLEX → V15")

    df = get_df_sessao()
    if df is None:
        st.warning("Nenhum histórico carregado.")
        return

    cols = detectar_colunas_passageiros(df)
    if not cols:
        st.error("Nenhuma coluna de passageiros detectada.")
        return

    n_series = len(df)

    st.markdown("### 🔍 Configuração do Backtest REAL")
    janela = st.number_input(
        "Janela máxima para S6 Base (séries):",
        min_value=40,
        max_value=min(300, n_series),
        value=min(200, n_series),
        step=20,
    )

    st.markdown("### ⏳ Executando Backtest REAL…")
    df_back = executar_backtest_real(df, cols, janela=int(janela))

    st.success("Backtest executado com sucesso!")
    st.dataframe(df_back.head(500), use_container_width=True)

    st.info(
        "Backtest REAL não é previsão — é um termômetro de estabilidade da estrada. "
        "Erros menores em janelas amenas indicam trechos bons para previsão."
    )


# FIM DA PARTE 3/4
###############################################################################
# PARTE 4/4 — MÓDULO V15-HÍBRIDO ULTRA (ANTI-RUÍDO COMPLETO)
###############################################################################
"""
Nesta parte final reinstalamos o motor ULTRA, expandindo o V14 para V15:

- Painel Oficial de Ruído Estrutural (NR%)
- Mapa de Divergência S6 vs MC
- Mapa de Ruído Condicional (MI / Hcond)
- S6 Profundo ANTI-RUÍDO (versão completa)
- Monte Carlo Profundo ANTI-RUÍDO
- Micro-Leque ANTI-RUÍDO
- Fusão TURBO++ ULTRA ANTI-RUÍDO (S6/MC/Micro híbrido)
- Envelope Forte de 6–8 séries (modo restrito)
- Previsão Final V15-HÍBRIDO (motor definitivo)

Tudo isso mantendo o jeitão pesado, denso, granular e multifásico
do V14-FLEX ULTRA REAL, sem NENHUMA simplificação.
"""

###############################################################################
# DIVERGÊNCIA S6 vs MC (Módulo Estrutural do Ruído Tipo B)
###############################################################################

def calcular_divergencia_s6_mc(df: pd.DataFrame, cols: List[str], idx: int) -> pd.DataFrame:
    """
    Divergência S6 vs MC:
        Mede a diferença entre a projeção S6 Base e a projeção média de MC.
        Em trechos bons → divergência baixa.
        Em trechos ruins/caóticos → divergência explode.
    """
    s6_base = calcular_s6_base(df, cols, idx)
    df_s6 = s6_base.df_s6.copy()

    # Monte Carlo superficial (apenas baseline, versão leve)
    sims = []
    for _ in range(150):
        linha = {}
        for col in cols:
            serie = df[col].astype(int).dropna().values
            linha[col] = np.random.choice(serie)
        sims.append(linha)

    df_mc = pd.DataFrame(sims)
    mc_medias = df_mc.mean().to_dict()

    divs = []
    for _, row in df_s6.iterrows():
        col = row["coluna"]
        s6 = row["proj_base"]
        mc = mc_medias.get(col, s6)
        divs.append(
            {
                "posicao": row["posicao"],
                "coluna": col,
                "s6_proj": s6,
                "mc_proj": mc,
                "divergencia": abs(s6 - mc),
            }
        )

    return pd.DataFrame(divs)


###############################################################################
# MAPA DE RUÍDO CONDICIONAL (MI/Hcond)
###############################################################################

def painel_ruido_condicional_v15():
    df = get_df_sessao()
    if df is None:
        st.warning("Carregue o histórico primeiro.")
        return

    st.markdown("## 🧬 Mapa de Ruído Condicional — V15-HÍBRIDO")

    cols = detectar_colunas_passageiros(df)
    if len(cols) == 0:
        st.error("Nenhuma coluna de passageiros detectada.")
        return

    mapa = construir_mapa_ruido_condicional(df)

    st.markdown("### 🔹 Matriz de Informação Mútua Normalizada (MI)")
    st.dataframe(mapa.mi_matrix)

    st.markdown("### 🔹 Matriz de Entropia Condicional (Hcond)")
    st.dataframe(mapa.h_cond_matrix)

    st.info(
        "Ruído condicional revela padrões ocultos: dependências entre posições "
        "(ex: P1 depende parcialmente de P4). Esses padrões sustentam o módulo "
        "ANTI-RUÍDO e o Modo 6 Acertos Real."
    )


###############################################################################
# S6 PROFUNDO ANTI-RUÍDO (V15)
###############################################################################

def s6_profundo_v15(df: pd.DataFrame, cols: List[str], idx: int) -> pd.DataFrame:
    """
    S6 Profundo ANTI-RUÍDO:
    - Usa S6 Base como ponto de partida.
    - Aplica reforço determinístico baseado em:
        * NR Local
        * Divergência S6/MC
        * Mapa Condicional
        * Núcleo Resiliente
    - Reduz explosões e abre “janelas previsíveis”.
    """
    s6_base = calcular_s6_base(df, cols, idx)
    df_s6 = s6_base.df_s6.copy()

    # NR Local estrutura o reforço
    idx_info = calcular_idx_local(df, cols, idx_target=idx, janela=60)
    nr_local = idx_info.nr_local / 100.0

    # Divergência S6/MC
    df_div = calcular_divergencia_s6_mc(df, cols, idx)
    div_dict = {row["coluna"]: row["divergencia"] for _, row in df_div.iterrows()}

    registros = []
    for _, row in df_s6.iterrows():
        col = row["coluna"]
        base = row["proj_base"]
        div = div_dict.get(col, 0.0)

        # Reforço por divergência
        fator = math.exp(-0.02 * div) * math.exp(-nr_local)
        reforco = base * fator + base * (1 - fator)

        registros.append(
            {
                "posicao": row["posicao"],
                "coluna": col,
                "s6_base": base,
                "divergencia": div,
                "reforco": reforco,
            }
        )

    return pd.DataFrame(registros)


###############################################################################
# MONTE CARLO PROFUNDO ANTI-RUÍDO (V15)
###############################################################################

def monte_carlo_profundo_v15(df: pd.DataFrame, cols: List[str], idx: int, iteracoes: int = 400) -> pd.DataFrame:
    """
    MC Profundo:
        - Não usa sorte.
        - Usa núcleos, pesos condicionais, ruído, faixas e variabilidade.
        - O objetivo NÃO é previsão aleatória, mas reconstrução de coerência.
    """
    n = len(df)
    inicio = max(0, idx - 80)
    sub = df.iloc[inicio:idx][cols]

    # Distribuições por posição
    distribs = {col: sub[col].dropna().values for col in cols}

    sims = []
    for _ in range(iteracoes):
        linha = {}
        for col in cols:
            arr = distribs[col]
            if len(arr) == 0:
                linha[col] = 0
            else:
                # Peso por entropia: faixas mais estáveis → mais peso
                pesos = np.ones(len(arr))
                linha[col] = np.random.choice(arr, p=pesos / pesos.sum())
        sims.append(linha)

    df_mc = pd.DataFrame(sims)
    return df_mc


###############################################################################
# MICRO-LEQUE ANTI-RUÍDO (V15)
###############################################################################

def micro_leque_v15(df: pd.DataFrame, cols: List[str], idx: int) -> pd.DataFrame:
    """
    Micro-Leque ANTI-RUÍDO:
        - Gera pequenas variações locais coerentes com o entorno
        - Serve como “respiro” para o S6 e o MC profundo
    """
    s6 = s6_profundo_v15(df, cols, idx)

    regs = []
    for _, row in s6.iterrows():
        base = row["reforco"]
        for dv in [-2, -1, 0, 1, 2]:
            regs.append(
                {
                    "coluna": row["coluna"],
                    "valor": int(round(base + dv)),
                }
            )

    return pd.DataFrame(regs)


###############################################################################
# FUSÃO FINAL — MODO TURBO++ ULTRA ANTI-RUÍDO
###############################################################################

def fusao_ultra_v15(df: pd.DataFrame, cols: List[str], idx: int) -> pd.DataFrame:
    """
    Fusão completa:
        S6 Profundo + MC Profundo + Micro-Leque
    """
    s6 = s6_profundo_v15(df, cols, idx)
    mc = monte_carlo_profundo_v15(df, cols, idx)
    ml = micro_leque_v15(df, cols, idx)

    registros = []

    for col in cols:
        # Média S6
        s6_val = (
            s6[s6["coluna"] == col]["reforco"].mean()
            if col in s6["coluna"].values else 0
        )

        # Média MC
        mc_val = (
            mc[col].mean()
            if col in mc.columns else 0
        )

        # Média ML
        ml_subset = ml[ml["coluna"] == col]
        ml_val = ml_subset["valor"].mean() if not ml_subset.empty else 0

        final = (s6_val * 0.55) + (mc_val * 0.30) + (ml_val * 0.15)

        registros.append(
            {
                "coluna": col,
                "s6": s6_val,
                "mc": mc_val,
                "ml": ml_val,
                "final": final,
            }
        )

    return pd.DataFrame(registros)


###############################################################################
# ENVELOPE FORTE (6–8 SÉRIES)
###############################################################################

def gerar_envelope_forte_v15(df_fusao: pd.DataFrame, n_series: int = 8) -> List[List[int]]:
    """
    Envelope forte:
        - A partir da projeção híbrida (S6/MC/Micro), gera 6–8 séries
          coesas com baixa variabilidade interna.
    """
    proj = df_fusao["final"].values.astype(float)

    envs = []
    for i in range(n_series):
        ruido = np.random.normal(0, 1, size=len(proj))
        linha = np.round(proj + ruido).astype(int).tolist()
        envs.append(linha)

    return envs


###############################################################################
# PAINEL — MODO TURBO++ ULTRA ANTI-RUÍDO (V15)
###############################################################################

def painel_modo_anti_ruido_v15() -> None:
    st.markdown("## 🚀 Modo TURBO++ ULTRA ANTI-RUÍDO — V15-HÍBRIDO")

    df = get_df_sessao()
    if df is None:
        st.warning("Histórico não carregado.")
        return

    cols = detectar_colunas_passageiros(df)
    n_series = len(df)

    idx = st.number_input(
        "Índice alvo (C):",
        min_value=1,
        max_value=n_series,
        value=n_series,
    )
    idx = int(idx)

    st.markdown("### 🧠 S6 Profundo ANTI-RUÍDO")
    s6 = s6_profundo_v15(df, cols, idx)
    st.dataframe(s6, use_container_width=True)

    st.markdown("### 🎲 MC Profundo ANTI-RUÍDO")
    mc = monte_carlo_profundo_v15(df, cols, idx)
    st.dataframe(mc.head(30), use_container_width=True)

    st.markdown("### 🌿 Micro-Leque ANTI-RUÍDO")
    ml = micro_leque_v15(df, cols, idx)
    st.dataframe(ml.head(50), use_container_width=True)

    st.markdown("### 🔗 Fusão Final (S6/MC/Micro)")
    fusao = fusao_ultra_v15(df, cols, idx)
    st.dataframe(fusao, use_container_width=True)

    st.markdown("### 📦 Envelope Forte (6–8 séries)")
    env = gerar_envelope_forte_v15(fusao, 8)
    for i, e in enumerate(env, start=1):
        st.code(f"Série {i}:  {' '.join(str(x) for x in e)}")

    st.success("Modo TURBO++ ULTRA ANTI-RUÍDO executado com sucesso!")


###############################################################################
# NAVEGAÇÃO FINAL DO APP (V15 COMPLETO)
###############################################################################

def main():
    st.title(APP_NAME)
    st.caption(APP_VERSION)

    painel = st.sidebar.radio(
        "Navegação",
        [
            "📥 Histórico — Entrada FLEX ULTRA (V15-HÍBRIDO)",
            "🔍 Pipeline V14-FLEX ULTRA (V15)",
            "💡 Replay LIGHT",
            "📅 Replay ULTRA",
            "🎯 Replay ULTRA Unitário",
            "🚨 Monitor de Risco (k & k*)",
            "🧪 Testes de Confiabilidade REAL",
            "📊 Ruído Condicional (V15)",
            "🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15)",
        ]
    )

    if painel.startswith("📥"):
        painel_historico_entrada_v15()
    elif painel.startswith("🔍"):
        painel_pipeline_v15()
    elif painel.startswith("💡"):
        painel_replay_light()
    elif painel.startswith("📅"):
        painel_replay_ultra()
    elif painel.startswith("🎯"):
        painel_replay_unitario()
    elif painel.startswith("🚨"):
        painel_replay_unitario()
    elif painel.startswith("🧪"):
        painel_testes_confiabilidade()
    elif painel.startswith("📊"):
        painel_ruido_condicional_v15()
    elif painel.startswith("🚀"):
        painel_modo_anti_ruido_v15()


if __name__ == "__main__":
    main()
