# ============================================================
# Predict Cars — V15.5.2-HÍBRIDO ANTI-ZUMBI (JUNÇÃO TOTAL)
# ============================================================
# Arquitetura consolidada: V13.8 → V14 → V14-FLEX → V15 → V15-HÍBRIDO
# Este arquivo unifica absolutamente TODOS os módulos históricos:
# - Entrada FLEX ULTRA (upload + texto, variável n1..nN, k opcional)
# - Pipeline S1–S7 completo (limpeza → normalização → métricas →
#   IDX → Núcleo Resiliente → S6 Profundo → S7 Final)
# - QDS Global + QDS Local
# - TVF (Top Variability Filter) integrado
# - Backtest Interno + Backtest do Futuro
# - Monte Carlo Profundo
# - K real, k*, k preditivo, Barômetro e Regimes
# - Replay LIGHT, Replay ULTRA, Replay ULTRA Unitário
# - Modo TURBO++ ULTRA com ajuste de ruído por regime
# - Ruído Condicional V15
# - Modo 6 acertos preparado
# - Modo FLEX: número variável de passageiros
# - Proteções ANTI-ZUMBI (limites, blocos, watchdogs, timeout por painel)
# ============================================================

import io
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# CONFIGURAÇÃO GLOBAL DO APP
# ============================================================

st.set_page_config(
    page_title="Predict Cars V15.5.2-HÍBRIDO ANTI-ZUMBI",
    layout="wide",
)

# ============================================================
# CONSTANTES DO SISTEMA (ANTI-ZUMBI + LIMITES GERAIS)
# ============================================================

# Limite automático padrão para segurança geral
MAX_LINHAS_AUTO = 2500

# Hard limit absoluto para evitar travamento completo
MAX_LINHAS_ABSOLUTO = 8000

# Tempo máximo recomendado por painel, usado no medidor de tempo
TEMPO_MAX_SEGUNDOS = 25

# Limite de blocos nos loops grandes (Replay ULTRA, Backtest, etc.)
MAX_BLOCOS_REPLAY = 2000
MAX_SIMULACOES_TESTES = 800

# Faixas permitidas dos passageiros
VALOR_MIN_PASSAGEIRO = 0
VALOR_MAX_PASSAGEIRO = 60

# Seeds internas — Monte Carlo e Turbo++ estáveis
RNG_SEMENTE_TURBO = 42
RNG_SEMENTE_MONTECARLO = 123

# ============================================================
# ANTI-ZUMBI: MEDIDOR DE TEMPO
# ============================================================

@contextmanager
def medidor_tempo(painel: str):
    """
    Mede o tempo de execução de um painel.
    Se passar do limite recomendado, exibe alerta ANTI-ZUMBI.
    """
    inicio = time.time()
    try:
        yield
    finally:
        dur = time.time() - inicio
        if dur > TEMPO_MAX_SEGUNDOS:
            st.warning(
                f"⏱️ Painel **{painel}** levou {dur:.1f}s. "
                "O sistema está em modo ANTI-ZUMBI – considere reduzir a janela."
            )

# ============================================================
# ANTI-ZUMBI: LIMITADOR DE DF
# ============================================================

def limitar_df(df: pd.DataFrame, max_linhas: int, contexto: str) -> pd.DataFrame:
    """
    Limita o tamanho do DataFrame usado pelos módulos internos.
    Proteção anti-zumbi: evita congelamento por datasets gigantes.
    """
    if df is None or df.empty:
        return df
    n = len(df)
    if n > max_linhas:
        st.warning(
            f"🧯 ANTI-ZUMBI ativado [{contexto}]: histórico possui {n} linhas. "
            f"Usando apenas as **{max_linhas}** mais recentes."
        )
        return df.tail(max_linhas).copy()
    return df

# ============================================================
# ANTI-ZUMBI: SECURE WRAPPER SESSION_STATE
# ============================================================

def init_session_state():
    """
    Inicializa todas as chaves importantes do sistema.
    Este método garante que mudanças entre painéis não provoquem
    comportamento zumbi ou perda silenciosa de variáveis.
    """
    defaults = {
        "df": None,
        "max_linhas_user": MAX_LINHAS_AUTO,

        # Estrutura S1–S7
        "df_s1": None,
        "df_s2": None,
        "df_s3": None,
        "df_s4": None,
        "df_s5": None,
        "df_s6": None,
        "df_s7": None,

        # Métricas globais
        "qds_global": None,
        "qds_local": None,
        "disp_global": None,
        "regime_estrada": None,
        "k_star_qual": None,

        # Métricas de risco
        "k_medio": None,
        "k_max": None,

        # Resultados avançados
        "resultado_backtest": None,
        "resultado_montecarlo": None,

        # Resultado TURBO++
        "leque_turbo_ultra": None,
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# Inicialização obrigatória
init_session_state()

# ============================================================
# ANTI-ZUMBI: OBTÉM DF COM SEGURANÇA
# ============================================================

def obter_df_seguro() -> Optional[pd.DataFrame]:
    """
    Retorna o histórico carregado, já com:
    - Limite escolhido pelo usuário
    - Ou limite automático ANTI-ZUMBI
    """
    df = st.session_state.get("df")
    if df is None or df.empty:
        return None

    max_user = st.session_state.get("max_linhas_user", 0)

    if isinstance(max_user, int) and max_user > 0:
        return limitar_df(df, max_user, "Limite do usuário")

    return limitar_df(df, MAX_LINHAS_AUTO, "Modo automático")
# ============================================================
# PARTE 2/24 — ENTRADA FLEX ULTRA (UPLOAD + TEXTO)
# ============================================================
# - Suporta:
#   * Upload de CSV
#   * Colar texto CSV
#   * Separador ; ou ,
#   * Qualquer quantidade de passageiros (n1..nN)
#   * k opcional na última coluna
#   * Geração de id (C1, C2, ...)
# ============================================================

def detectar_sep(conteudo: str) -> str:
    """
    Detecta separador predominante (; ou ,).
    Se houver empate, prioriza ';' (padrão do seu histórico).
    """
    if conteudo.count(";") >= conteudo.count(","):
        return ";"
    return ","


def _ler_csv_generico(conteudo: str) -> pd.DataFrame:
    """
    Lê texto CSV genérico sem header, usando o separador detectado.
    """
    sep = detectar_sep(conteudo)
    buffer = io.StringIO(conteudo)
    df_raw = pd.read_csv(buffer, sep=sep, header=None)
    return df_raw


def normalizar_historico(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza o histórico para o formato FLEX ULTRA:

    - id: identificador da série (C1, C2, ...)
    - n1..nN: passageiros (número variável de colunas)
    - k (opcional): número de guardas que acertaram exatamente (inteiro >= 0)

    Regras:
    - Se existir uma última coluna que seja toda numérica e >= 0 → interpretada como k.
    - Caso contrário, todas as colunas depois da primeira são passageiros.
    """
    if df_raw is None or df_raw.empty:
        raise ValueError("Histórico vazio após leitura do CSV.")

    df = df_raw.copy()
    n_cols = df.shape[1]

    if n_cols < 2:
        raise ValueError(
            "Histórico precisa ter pelo menos 2 colunas (id + passageiros). "
            "Exemplo típico: C1;41;5;4;52;30;33;0"
        )

    # Nomeia colunas genéricas inicialmente
    df.columns = [f"col_{i}" for i in range(1, n_cols + 1)]

    col_id = "col_1"
    outras = [c for c in df.columns if c != col_id]

    # Tenta detectar k como última coluna inteira não-negativa
    col_k = None
    if len(outras) >= 2:
        ultima = outras[-1]
        serie_ult = pd.to_numeric(df[ultima], errors="coerce")
        if serie_ult.notna().all() and (serie_ult >= 0).all():
            # Boa candidata a k
            col_k = ultima
            col_pass = outras[:-1]
        else:
            col_pass = outras
    else:
        col_pass = outras

    # Mapa de renomeação
    rename_map = {col_id: "id"}
    for i, c in enumerate(col_pass, start=1):
        rename_map[c] = f"n{i}"
    if col_k is not None:
        rename_map[col_k] = "k"

    df = df.rename(columns=rename_map)

    # Garante que colunas n* sejam numéricas
    col_nums = [c for c in df.columns if c.startswith("n")]
    for c in col_nums:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Garante que k (se existir) seja inteiro >= 0
    if "k" in df.columns:
        df["k"] = pd.to_numeric(df["k"], errors="coerce").fillna(0)
        df["k"] = df["k"].clip(lower=0).astype(int)

    # Remove linhas sem nenhum passageiro válido
    df = df.dropna(subset=col_nums, how="all").reset_index(drop=True)

    # Gera id se vier vazio ou inutilizável
    if "id" not in df.columns or df["id"].isna().all():
        df["id"] = [f"C{i}" for i in range(1, len(df) + 1)]
    else:
        # Normaliza id para string (por segurança)
        df["id"] = df["id"].astype(str)

    return df


def carregar_historico_upload(arquivo, formato: str) -> pd.DataFrame:
    """
    Carrega histórico a partir de upload de arquivo .csv.
    O parâmetro 'formato' é mantido para compatibilidade com o UI,
    mas a normalização é sempre FLEX ULTRA.
    """
    if arquivo is None:
        raise ValueError("Nenhum arquivo selecionado.")

    conteudo = arquivo.read().decode("utf-8").strip()
    if not conteudo:
        raise ValueError("Arquivo vazio ou não pôde ser lido.")

    df_raw = _ler_csv_generico(conteudo)
    df_norm = normalizar_historico(df_raw)
    return df_norm


def carregar_historico_texto(texto: str, formato: str) -> pd.DataFrame:
    """
    Carrega histórico a partir de texto colado.
    Espera o mesmo formato do arquivo (CSV compatível).
    """
    if not texto or not texto.strip():
        raise ValueError("Texto do histórico está vazio.")

    conteudo = texto.strip()
    df_raw = _ler_csv_generico(conteudo)
    df_norm = normalizar_historico(df_raw)
    return df_norm
# ============================================================
# PARTE 3/24 — ESTRUTURA DA ESTRADA + PIPELINE S1–S3
# ============================================================

@dataclass
class EstradaContext:
    """
    Estrutura central da estrada no Predict Cars.
    Carrega o estado e cada camada do pipeline S1–S7.
    """
    df_base: pd.DataFrame
    col_pass: List[str] = field(default_factory=list)

    # Camadas S1–S7
    df_s1: Optional[pd.DataFrame] = None
    df_s2: Optional[pd.DataFrame] = None
    df_s3: Optional[pd.DataFrame] = None
    df_s4: Optional[pd.DataFrame] = None
    df_s5: Optional[pd.DataFrame] = None
    df_s6: Optional[pd.DataFrame] = None
    df_s7: Optional[pd.DataFrame] = None

    # Métricas e dados extras
    qds_global: Optional[float] = None
    info_extra: Dict[str, Any] = field(default_factory=dict)

    def detectar_col_pass(self) -> List[str]:
        """
        Detecta e devolve as colunas n* (passageiros).
        """
        if not self.col_pass:
            self.col_pass = [c for c in self.df_base.columns if c.startswith("n")]
        return self.col_pass


# ============================================================
# S1 — LIMPEZA PROFUNDA + CLIPPING (V14 original + V15-Híbrido)
# ============================================================

def s1_filtrar_valores(context: EstradaContext) -> pd.DataFrame:
    """
    S1 — Limpeza profunda da estrada:
    - remove linhas totalmente vazias nos passageiros
    - converte passageiros para numérico
    - aplica clipping rígido (0–60)
    - mantém id, k e demais colunas auxiliares
    """
    df = context.df_base.copy()
    cols = context.detectar_col_pass()

    # remove linhas sem nenhum valor válido
    df = df.dropna(subset=cols, how="all").reset_index(drop=True)

    # converte para numérico e clippa
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].clip(VALOR_MIN_PASSAGEIRO, VALOR_MAX_PASSAGEIRO)

    context.df_s1 = df
    st.session_state["df_s1"] = df
    return df


# ============================================================
# S2 — NORMALIZAÇÃO (V14 original: centragem + escala)
# ============================================================

def s2_normalizar_basico(context: EstradaContext) -> pd.DataFrame:
    """
    S2 — Normalização real (baseline V14):
    - (x - média) / desvio
    - evita divisão por zero
    - transforma dados em escala comparável para S3–S7
    """
    if context.df_s1 is None:
        s1_filtrar_valores(context)

    df = context.df_s1.copy()
    cols = context.detectar_col_pass()

    for c in cols:
        serie = df[c].astype(float)
        mu = serie.mean()
        sd = serie.std(ddof=1)
        if sd == 0:
            sd = 1.0  # fallback seguro
        df[c] = (serie - mu) / sd

    context.df_s2 = df
    st.session_state["df_s2"] = df
    return df


# ============================================================
# S3 — MÉTRICAS LOCAIS (V14: diff, absdiff, volatilidade local)
# ============================================================

def s3_metricas_locais(context: EstradaContext) -> pd.DataFrame:
    """
    S3 — Métricas locais reais:
    - diferenças entre séries consecutivas
    - magnitudes absolutas
    - volatilidade inicial por passageiro
    """
    if context.df_s2 is None:
        s2_normalizar_basico(context)

    df = context.df_s2.copy()
    cols = context.detectar_col_pass()

    for c in cols:
        df[f"{c}_diff"] = df[c].diff()
        df[f"{c}_absdiff"] = df[f"{c}_diff"].abs()

    context.df_s3 = df
    st.session_state["df_s3"] = df
    return df
# ============================================================
# PARTE 4/24 — S4 (RESUMO GLOBAL) + S5 (IDX + NÚCLEO RESILIENTE)
# ============================================================

def s4_resumo_global(context: EstradaContext) -> pd.DataFrame:
    """
    S4 — Resumo global da estrada (escala normalizada, baseline V14/V15):
    - Calcula estatísticas por passageiro em S2/S3:
      * média
      * desvio padrão
      * mínimo
      * máximo
      * ruído médio (|diff|)
      * ruído p95 (|diff|)
    - Serve de base para o IDX/Núcleo Resiliente e para S6/S7.
    """
    if context.df_s3 is None:
        s3_metricas_locais(context)

    df = context.df_s3
    cols = context.detectar_col_pass()

    stats = []
    for c in cols:
        serie = df[c].dropna().astype(float)
        if serie.empty:
            continue

        d = {
            "passageiro": c,
            "media": float(serie.mean()),
            "desvio": float(serie.std(ddof=1)) if len(serie) > 1 else 0.0,
            "min": float(serie.min()),
            "max": float(serie.max()),
        }

        # Ruído local baseado nas diferenças absolutas
        col_absdiff = f"{c}_absdiff"
        if col_absdiff in df.columns:
            diffs = df[col_absdiff].dropna().astype(float)
            if not diffs.empty:
                d["ruido_medio"] = float(diffs.mean())
                d["ruido_p95"] = float(diffs.quantile(0.95))
            else:
                d["ruido_medio"] = 0.0
                d["ruido_p95"] = 0.0
        else:
            d["ruido_medio"] = 0.0
            d["ruido_p95"] = 0.0

        stats.append(d)

    df_stats = pd.DataFrame(stats)
    context.df_s4 = df_stats
    st.session_state["df_s4"] = df_stats
    return df_stats


def s5_idx_nucleo_resiliente(context: EstradaContext) -> pd.DataFrame:
    """
    S5 — IDX + Núcleo Resiliente (baseline V14/V15):

    Ideia:
    - Quanto MENOR o ruído médio, MAIS resiliente é o passageiro.
    - Constrói um score de resiliência por passageiro.
    - A partir desses scores, gera um IDX global da estrada.

    Saídas:
    - df_s5 com:
      * passageiro
      * ruido_medio, ruido_p95
      * score_resiliencia
      * idx_local (normalizado 0–1)
    - info_extra["idx_global_resiliencia"]
    """
    if context.df_s4 is None:
        s4_resumo_global(context)

    stats = context.df_s4.copy()
    if stats.empty:
        context.df_s5 = stats
        st.session_state["df_s5"] = stats
        context.info_extra["idx_global_resiliencia"] = None
        return stats

    # Evita zeros e NaN em ruido_medio
    if "ruido_medio" not in stats.columns:
        stats["ruido_medio"] = 0.0
    stats["ruido_medio"] = stats["ruido_medio"].fillna(0.0)

    # Score de resiliência: 1 / (1 + ruído)
    stats["score_resiliencia"] = 1.0 / (1.0 + stats["ruido_medio"])

    # Normaliza score_resiliencia em [0, 1] (idx_local)
    sr = stats["score_resiliencia"]
    sr_min = float(sr.min())
    sr_max = float(sr.max()) if float(sr.max()) != float(sr.min()) else sr_min + 1e-9
    stats["idx_local"] = (sr - sr_min) / (sr_max - sr_min)

    # IDX global da estrada = média dos idx_local
    idx_global = float(stats["idx_local"].mean())
    context.info_extra["idx_global_resiliencia"] = idx_global

    context.df_s5 = stats
    st.session_state["df_s5"] = stats
    return stats
# ============================================================
# PARTE 5/24 — S6 PROFUNDO + S7 FINAL (QDS Global, Regime, k*)
# ============================================================

def s6_profundo(context: EstradaContext) -> pd.DataFrame:
    """
    S6 PROFUNDO — versão consolidada (V14 TURBO++ + V15-HÍBRIDO):

    - A estrada é dividida em janelas móveis (tamanho adaptativo).
    - Para cada janela:
        * calcula dispersão média das colunas n* (na escala normalizada S2/S3)
        * classifica a janela em regime:
            ▢ 🟢 estável
            ▢ 🟡 moderado
            ▢ 🔴 turbulento
    - Retorna um mapa completo de regimes da estrada.
    - O S6 é uma das principais entradas para:
        * QDS Global
        * k* qualitativo
        * Modo TURBO++ ULTRA adaptativo
        * Replay ULTRA
    """
    if context.df_s3 is None:
        s3_metricas_locais(context)

    df = context.df_s3.copy()
    cols = context.detectar_col_pass()

    if df.empty or not cols:
        context.df_s6 = pd.DataFrame([])
        st.session_state["df_s6"] = context.df_s6
        return context.df_s6

    # Tamanho adaptativo da janela (garante estabilidade)
    janela = min(120, max(30, len(df) // 10))

    resultados = []
    for inicio in range(0, len(df), janela):
        fim = min(len(df), inicio + janela)
        sub = df.iloc[inicio:fim]

        # desvio médio de todos os passageiros
        desvios = []
        for c in cols:
            serie = sub[c].dropna().astype(float)
            if len(serie) > 1:
                desvios.append(float(serie.std(ddof=1)))
        disp = float(np.mean(desvios)) if desvios else 0.0

        # Classificação do regime para esta janela
        if disp < 0.6:
            regime = "🟢 estável"
        elif disp < 1.2:
            regime = "🟡 moderado"
        else:
            regime = "🔴 turbulento"

        resultados.append(
            {
                "inicio": inicio + 1,
                "fim": fim,
                "qtd": fim - inicio,
                "disp_s6": disp,
                "regime_s6": regime,
            }
        )

    df_s6 = pd.DataFrame(resultados)
    context.df_s6 = df_s6
    st.session_state["df_s6"] = df_s6
    return df_s6


def s7_camada_final(context: EstradaContext) -> pd.DataFrame:
    """
    S7 — Camada Final (consolidação total):
    Integra:
    - Estatísticas globais (S4)
    - IDX + Núcleo Resiliente (S5)
    - Regimes por janelas (S6)
    - Calcula QDS Global real
    - Define regime geral da estrada
    - Define k* qualitativo (sentinela preditivo)
    """
    # Garante que todas as camadas anteriores existam
    if context.df_s5 is None:
        s5_idx_nucleo_resiliente(context)
    if context.df_s6 is None:
        s6_profundo(context)

    stats = context.df_s5.copy()
    s6 = context.df_s6.copy()

    # ============================================================
    # QDS Global (Qualidade Dinâmica da Série)
    # ------------------------------------------------------------
    # Quanto maior a dispersão do S6, pior a qualidade dinâmica.
    # Fórmula base do V14:
    #     QDS = 100 - (disp_medio_S6 * 25)
    # ============================================================
    if not s6.empty:
        disp_s6_medio = float(s6["disp_s6"].mean())
        qds_global = max(0.0, 100.0 - disp_s6_medio * 25.0)
    else:
        qds_global = 50.0  # fallback neutro

    context.qds_global = qds_global
    st.session_state["qds_global"] = qds_global

    # ============================================================
    # Regime geral da estrada
    # ============================================================
    disp_global = disp_s6_medio if not s6.empty else 1.0

    if disp_global < 0.6:
        regime = "🟢 Estrada estável"
        k_star_info = "k*: ambiente estável — regime normal."
    elif disp_global < 1.2:
        regime = "🟡 Estrada moderada"
        k_star_info = "k*: turbulência moderada — ajustes recomendados."
    else:
        regime = "🔴 Estrada turbulenta"
        k_star_info = "k*: turbulência forte — leques amplos recomendados."

    context.info_extra["regime_estrada"] = regime
    context.info_extra["k_star_qual"] = k_star_info

    st.session_state["regime_estrada"] = regime
    st.session_state["k_star_qual"] = k_star_info

    # ============================================================
    # Consolidação final
    # ============================================================
    df_final = stats.copy()
    df_final["qds_global"] = qds_global
    df_final["regime"] = regime

    context.df_s7 = df_final
    st.session_state["df_s7"] = df_final
    return df_final
# ============================================================
# PARTE 6/24 — k REAL + INTEGRAÇÃO FINAL DA ESTRADA
# ============================================================

def calcular_k_real(df: pd.DataFrame) -> Tuple[Optional[int], Optional[float]]:
    """
    Calcula o k real existente no histórico:
    - k_max  = maior k observado
    - k_medio = média dos ks
    Se não houver coluna 'k', retorna (None, None).

    Este k real se integra com:
    - Monitor de Risco
    - Modo TURBO++ (modo k vs modo k*)
    - Replay ULTRA / Unitário
    """
    if "k" not in df.columns:
        st.session_state["k_max"] = None
        st.session_state["k_medio"] = None
        return None, None

    serie = pd.to_numeric(df["k"], errors="coerce").dropna()
    if serie.empty:
        st.session_state["k_max"] = None
        st.session_state["k_medio"] = None
        return None, None

    k_max = int(serie.max())
    k_medio = float(serie.mean())

    st.session_state["k_max"] = k_max
    st.session_state["k_medio"] = k_medio

    return k_max, k_medio


def analisar_estrada_completa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função-mãe que executa o pipeline completo S1–S7:

        S1 — limpeza
        S2 — normalização
        S3 — métricas locais
        S4 — estatísticas globais
        S5 — IDX + Núcleo Resiliente
        S6 — Profundo (regimes)
        S7 — Final (QDS, regime global, k* qualitativo)

    Também atualiza:
        - disp_global (nível real de dispersão da estrada)
        - regime_estrada
        - k_star_qual
        - qds_global

    Esta função é usada por:
        - Pipeline principal
        - Replay LIGHT / ULTRA / Unitário
        - Monitor de Risco
        - Modo TURBO++ ULTRA adaptativo
    """
    context = EstradaContext(df_base=df)

    # Executa as camadas de forma sequencial
    s1_filtrar_valores(context)
    s2_normalizar_basico(context)
    s3_metricas_locais(context)
    stats = s4_resumo_global(context)
    s5_idx_nucleo_resiliente(context)
    s6_profundo(context)
    s7_camada_final(context)

    # ============================================================
    # Dispersão Global Verdadeira (baseada em S4)
    # ============================================================
    if not stats.empty and "desvio" in stats.columns:
        disp_global = float(stats["desvio"].mean())
    else:
        disp_global = 1.0

    st.session_state["disp_global"] = disp_global

    # Regime e k* já definidos no S7
    regime = context.info_extra.get("regime_estrada", None)
    k_star = context.info_extra.get("k_star_qual", None)

    if regime:
        st.session_state["regime_estrada"] = regime
    if k_star:
        st.session_state["k_star_qual"] = k_star

    return stats
# ============================================================
# PARTE 7/24 — BACKTEST INTERNO, BACKTEST DO FUTURO,
#                MONTE CARLO PROFUNDO, QDS LOCAL
# ============================================================

def backtest_interno(
    df: pd.DataFrame,
    passo: int = 10,
    janela: int = 80,
) -> pd.DataFrame:
    """
    Backtest Interno — versão alinhada ao V14/V15:

    Ideia central:
    - Percorre a estrada em janelas de tamanho fixo.
    - Em cada janela:
        * mede dispersão média dos passageiros
        * converte em uma "qualidade simulada" (escala 0–100)
    - Não mexe nas previsões do pipeline; apenas mede
      o quanto aquele trecho da estrada seria "bom" para atacar.

    Saída:
    - DataFrame com colunas:
        * inicio, fim, qtd
        * disp_backtest
        * qualidade_simulada_%
    """
    col_pass = [c for c in df.columns if c.startswith("n")]
    if not col_pass:
        raise ValueError("Histórico sem colunas de passageiros (n1..nN) para backtest interno.")

    resultados = []
    total = len(df)
    idx = 0

    while idx + janela <= total:
        sub = df.iloc[idx: idx + janela][col_pass].astype(float)

        desvios = sub.std(ddof=1)
        disp = float(desvios.mean())

        # Qualidade: quanto menor a dispersão, maior a "qualidade simulada"
        qualidade = max(0.0, 100.0 - disp * 4.0)

        resultados.append(
            {
                "inicio": idx + 1,
                "fim": idx + janela,
                "qtd": janela,
                "disp_backtest": disp,
                "qualidade_simulada_%": round(qualidade, 1),
            }
        )

        idx += passo
        if len(resultados) >= MAX_SIMULACOES_TESTES:
            break

    df_bt = pd.DataFrame(resultados)
    st.session_state["resultado_backtest"] = df_bt
    return df_bt


def backtest_do_futuro(
    df: pd.DataFrame,
    janela_hist: int = 80,
    horizonte_futuro: int = 20,
    passo: int = 10,
) -> pd.DataFrame:
    """
    Backtest do Futuro — versão conceitual original do V14:

    Ideia:
    - Usa janelas históricas recentes (janela_hist).
    - Para cada posição possível:
        * considera janela_hist como "histórico"
        * e horizonte_futuro como "futuro"
        * mede como o regime do histórico se conecta com o ruído do futuro.

    Saída:
    - DataFrame com:
        * inicio_hist, fim_hist
        * inicio_fut, fim_fut
        * disp_hist, disp_fut
        * compatibilidade_%
    """
    col_pass = [c for c in df.columns if c.startswith("n")]
    if not col_pass:
        raise ValueError("Histórico sem colunas de passageiros (n1..nN) para Backtest do Futuro.")

    resultados = []
    total = len(df)
    idx = 0

    while idx + janela_hist + horizonte_futuro <= total:
        hist = df.iloc[idx: idx + janela_hist][col_pass].astype(float)
        fut = df.iloc[idx + janela_hist: idx + janela_hist + horizonte_futuro][col_pass].astype(float)

        disp_hist = float(hist.std(ddof=1).mean())
        disp_fut = float(fut.std(ddof=1).mean())

        # Compatibilidade: quanto mais parecidos os regimes, maior compatibilidade
        delta = abs(disp_hist - disp_fut)
        compat = max(0.0, 100.0 - delta * 15.0)

        resultados.append(
            {
                "inicio_hist": idx + 1,
                "fim_hist": idx + janela_hist,
                "inicio_fut": idx + janela_hist + 1,
                "fim_fut": idx + janela_hist + horizonte_futuro,
                "disp_hist": disp_hist,
                "disp_fut": disp_fut,
                "compatibilidade_%": round(compat, 1),
            }
        )

        idx += passo
        if len(resultados) >= MAX_SIMULACOES_TESTES:
            break

    df_bf = pd.DataFrame(resultados)
    # Guardar em session_state se necessário no futuro
    st.session_state["resultado_backtest_futuro"] = df_bf
    return df_bf


def simular_monte_carlo_profundo(
    df: pd.DataFrame,
    n_universos: int = 500,
    tamanho_amostra: int = 50,
) -> pd.DataFrame:
    """
    Monte Carlo Profundo:

    - Cria "universos" amostrais da estrada.
    - Em cada universo:
        * amostra tamanho_amostra séries aleatórias
        * mede ruído médio dos passageiros
    - Retorna distribuição de ruído global.

    Saída:
    - DataFrame com:
        * universo
        * ruido_medio
    """
    col_pass = [c for c in df.columns if c.startswith("n")]
    if not col_pass:
        raise ValueError("Histórico sem colunas de passageiros (n1..nN) para Monte Carlo Profundo.")

    rng = np.random.default_rng(RNG_SEMENTE_MONTECARLO)
    n = len(df)
    tamanho_amostra = min(tamanho_amostra, n)

    resultados = []
    for u in range(int(n_universos)):
        idxs = rng.integers(0, n, size=tamanho_amostra)
        sub = df.iloc[idxs][col_pass].astype(float)
        desvios = sub.std(ddof=1)
        ruido = float(desvios.mean())
        resultados.append(
            {
                "universo": u + 1,
                "ruido_medio": ruido,
            }
        )

        if len(resultados) >= MAX_SIMULACOES_TESTES:
            break

    df_mc = pd.DataFrame(resultados)
    st.session_state["resultado_montecarlo"] = df_mc
    return df_mc


def calcular_qds_local(df_bt: pd.DataFrame) -> Optional[float]:
    """
    QDS Local:
    - A partir do Backtest Interno, extrai uma QDS média local.
    - Mede a "qualidade dinâmica" nos trechos testados.
    """
    if df_bt is None or df_bt.empty:
        st.session_state["qds_local"] = None
        return None

    if "qualidade_simulada_%" not in df_bt.columns:
        st.session_state["qds_local"] = None
        return None

    qds = float(df_bt["qualidade_simulada_%"].mean())
    st.session_state["qds_local"] = qds
    return qds
# ============================================================
# PARTE 8/24 — MODO TURBO++ ULTRA + TVF
# ============================================================

def _calibrar_ruido_por_regime() -> float:
    """
    Define um fator de ruído base para o TURBO++ a partir de:
    - disp_global (S4/S7)
    - qds_global
    - regime_estrada

    Quanto melhor o ambiente (alta QDS, estrada estável),
    menor o ruído; quanto pior, maior o ruído.
    """
    disp_global = st.session_state.get("disp_global", 1.0)
    qds_global = st.session_state.get("qds_global", 50.0)
    regime = st.session_state.get("regime_estrada", "🟡 Estrada moderada")

    # Base neutra
    base = 1.0

    # Ajuste por dispersão
    if disp_global < 0.6:
        base *= 0.7
    elif disp_global > 1.4:
        base *= 1.4

    # Ajuste por QDS
    if qds_global > 75:
        base *= 0.8
    elif qds_global < 40:
        base *= 1.3

    # Ajuste por regime qualitativo
    if "🟢" in regime:
        base *= 0.8
    elif "🔴" in regime:
        base *= 1.3

    return max(0.3, min(base, 3.0))


def _score_tvf_serie(
    valores: np.ndarray,
    col_pass: List[str],
    df_stats_s4: pd.DataFrame,
    qds_global: float,
    modo_k: str,
) -> float:
    """
    Calcula um score TVF (Top Variability Filter) para uma série do leque.

    Ideia:
    - Penalizar séries muito "distantes" do perfil médio da estrada.
    - Recompensar séries mais alinhadas com o núcleo resiliente.
    - Ajustar levemente pelo QDS global e pelo modo k/k̂.
    """
    # Cria dicionário passageiro->valor para facilitar
    d_val = {c: v for c, v in zip(col_pass, valores)}

    # junta com stats de S4
    desvios_norm = []
    for _, row in df_stats_s4.iterrows():
        p = row["passageiro"]
        if p not in d_val:
            continue
        # aqui consideramos distância em relação à média normalizada
        # (quanto mais distante, maior o "custo")
        # stats já estão na escala normalizada (S2/S3)
        media = row["media"]
        desvio = row["desvio"] if row["desvio"] != 0 else 1.0
        z = abs((d_val[p] - media) / desvio)
        desvios_norm.append(z)

    if not desvios_norm:
        base = 0.5
    else:
        media_z = float(np.mean(desvios_norm))
        # Quanto menor o z médio, melhor
        base = 1.0 / (1.0 + media_z)

    # Ajuste por QDS Global
    if qds_global > 75:
        base *= 1.05
    elif qds_global < 40:
        base *= 0.95

    # Ajuste leve por modo k
    if "k̂" in modo_k:
        base *= 1.02  # modo preditivo ganha leve peso
    else:
        base *= 0.98

    return float(base)


def gerar_leque_turbo_ultra(
    df: pd.DataFrame,
    idx_alvo_zero: int,
    n_series: int,
    modo_k: str,
    confiab_min: int,
    usar_barometro: bool = True,
) -> pd.DataFrame:
    """
    Núcleo completo do Modo TURBO++ ULTRA:

    - Toma uma série alvo (idx_alvo_zero) como base.
    - Usa a estrada (S1–S7 + QDS + regime) para calibrar ruído.
    - Gera um leque de séries, com:
        * n1..nN
        * confianca_%
        * modo_k (k* vs k̂)
        * score_tvf
    - Aplica TVF para selecionar as top N séries finais.

    Observações:
    - Não altera o histórico.
    - Depende de:
        * analisar_estrada_completa() já ter sido executado
          OU é executada aqui internamente, se ainda não houver S1–S7.
    """
    if df is None or df.empty:
        raise ValueError("Histórico vazio para gerar leque TURBO++ ULTRA.")

    df = df.reset_index(drop=True)
    df = limitar_df(df, MAX_LINHAS_AUTO, "Leque TURBO++")

    col_pass = [c for c in df.columns if c.startswith("n")]
    if not col_pass:
        raise ValueError("Histórico sem colunas de passageiros (n1..nN) para TURBO++ ULTRA.")

    if not (0 <= idx_alvo_zero < len(df)):
        raise ValueError("Índice alvo fora do intervalo do histórico carregado.")

    # Garante que a estrada esteja analisada
    if st.session_state.get("df_s7") is None:
        analisar_estrada_completa(df)

    df_stats_s4 = st.session_state.get("df_s4", None)
    if df_stats_s4 is None or df_stats_s4.empty:
        # Se por algum motivo não houver S4, reexecuta a análise
        analisar_estrada_completa(df)
        df_stats_s4 = st.session_state.get("df_s4", None)

    if df_stats_s4 is None or df_stats_s4.empty:
        raise ValueError("Não foi possível obter estatísticas S4 para TVF.")

    qds_global = st.session_state.get("qds_global", 50.0)

    # Série alvo em escala original (passageiros clampados 0–60)
    base = df.loc[idx_alvo_zero, col_pass].astype(float).values
    base = np.clip(base, VALOR_MIN_PASSAGEIRO, VALOR_MAX_PASSAGEIRO)

    rng = np.random.default_rng(RNG_SEMENTE_TURBO)

    # Intensidade de ruído calibrada
    if usar_barometro:
        fator_ruido = _calibrar_ruido_por_regime()
    else:
        fator_ruido = 1.0

    previsoes_raw = []
    oversampling = max(2, int(1.8 * n_series))

    for _ in range(oversampling):
        # Ruído gaussian com escala ajustada ao regime/QDS
        ruido = rng.normal(loc=0.0, scale=fator_ruido * 2.0, size=len(col_pass))
        serie = np.clip(base + ruido, VALOR_MIN_PASSAGEIRO, VALOR_MAX_PASSAGEIRO)
        serie = np.round(serie).astype(int)

        # Confiabilidade inversamente proporcional à intensidade do ruído
        intensidade = float(np.abs(ruido).mean())
        confianca = max(5.0, 100.0 - intensidade * 3.5)

        if confianca < confiab_min:
            continue

        # Score TVF da série
        # Para o TVF usamos a série em "espaço normalizado" relativo a S4.
        # Aqui, como S4 já está em escala normalizada, usaremos os valores
        # como "pseudo-normalizados" — coerente com o espírito do V14/V15.
        serie_float = serie.astype(float)
        score_tvf = _score_tvf_serie(
            valores=serie_float,
            col_pass=col_pass,
            df_stats_s4=df_stats_s4,
            qds_global=qds_global,
            modo_k=modo_k,
        )

        previsoes_raw.append(
            {
                **{c: int(v) for c, v in zip(col_pass, serie)},
                "confianca_%": round(confianca, 1),
                "score_tvf": round(score_tvf, 6),
                "modo_k": modo_k,
            }
        )

    if not previsoes_raw:
        return pd.DataFrame([])

    df_raw = pd.DataFrame(previsoes_raw)

    # ============================================================
    # TVF — Top Variability Filter
    # ------------------------------------------------------------
    # Ordena combinações por:
    #   1) score_tvf (desc)
    #   2) confianca_% (desc)
    # e mantém somente as n_series primeiras.
    # ============================================================
    df_raw = df_raw.sort_values(
        by=["score_tvf", "confianca_%"],
        ascending=[False, False],
        ignore_index=True,
    )

    df_final = df_raw.head(int(n_series)).copy()
    df_final.insert(0, "rank", np.arange(1, len(df_final) + 1))

    st.session_state["leque_turbo_ultra"] = df_final
    return df_final
# ============================================================
# PARTE 9/24 — RUÍDO CONDICIONAL V15 (LOCAL / GLOBAL / REGIME)
# ============================================================

def calcular_ruido_condicional_global(df: pd.DataFrame) -> Dict[str, float]:
    """
    Ruído Condicional Global (V15):
    Mede o ruído por passageiro considerando:
    - variação (diff)
    - z-score condicional
    - regime contextual (S6)
    Retorna um dicionário passageiro -> ruído global.
    """
    col_pass = [c for c in df.columns if c.startswith("n")]
    if not col_pass:
        return {}

    # Diferenças
    ruido = {}
    for c in col_pass:
        diffs = df[c].diff().abs().dropna()
        if diffs.empty:
            ruido[c] = 0.0
            continue

        # ruído condicional z = |diff| / (1 + média absoluta)
        base = df[c].abs().mean()
        z = diffs.mean() / (1.0 + base)
        ruido[c] = float(z)

    return ruido


def calcular_ruido_condicional_movel(
    df: pd.DataFrame,
    janela: int = 40
) -> pd.DataFrame:
    """
    Ruído Condicional Móvel:
    - Janela deslizante (tamanho adaptativo)
    - Calcula ruído condicional local por passageiro
    - Produz mapa temporal de ruídos
    """
    col_pass = [c for c in df.columns if c.startswith("n")]
    if not col_pass:
        return pd.DataFrame([])

    janela = min(max(20, janela), 200)  # limites seguros

    resultados = []

    for ini in range(0, len(df), janela):
        fim = min(len(df), ini + janela)
        sub = df.iloc[ini:fim]

        ruido_local = {}
        for c in col_pass:
            diffs = sub[c].diff().abs().dropna()
            if diffs.empty:
                ruido_local[c] = 0.0
            else:
                base = sub[c].abs().mean()
                z = diffs.mean() / (1.0 + base)
                ruido_local[c] = float(z)

        disp_local = float(np.mean([v for v in ruido_local.values()]))

        resultados.append(
            {
                "inicio": ini + 1,
                "fim": fim,
                "qtd": fim - ini,
                "disp_ruido_cond": disp_local,
                **{c: ruido_local[c] for c in col_pass},
            }
        )

        if len(resultados) > MAX_SIMULACOES_TESTES:
            break

    return pd.DataFrame(resultados)


def estimar_ruido_por_regime(
    df_s6: pd.DataFrame,
    ruido_global: Dict[str, float],
) -> Dict[str, float]:
    """
    Estima o ruído condicional por regime (V15):
    - Usa o mapa S6 para reforçar/penalizar ruído baseado no regime.
    - Retorna um dicionário passageiro -> ruído ajustado.
    """
    if df_s6 is None or df_s6.empty:
        return ruido_global

    fatores = []
    for _, row in df_s6.iterrows():
        reg = row.get("regime_s6", "")
        if "🟢" in reg:
            fatores.append(0.85)
        elif "🟡" in reg:
            fatores.append(1.0)
        elif "🔴" in reg:
            fatores.append(1.25)

    if not fatores:
        fator_medio = 1.0
    else:
        fator_medio = float(np.mean(fatores))

    # aplica o fator médio ao ruído global
    ajustado = {c: float(v) * fator_medio for c, v in ruido_global.items()}
    return ajustado
# ============================================================
# PARTE 10/24 — NÚCLEO DE REPLAY (LIGHT / ULTRA / UNITÁRIO)
# ============================================================

def replay_light_core(
    df: pd.DataFrame,
    idx_alvo_zero: int,
    janela_contexto: int = 10,
) -> Dict[str, Any]:
    """
    Núcleo do Replay LIGHT (sem UI):

    - Seleciona uma série alvo (idx_alvo_zero).
    - Mostra a linha alvo.
    - Mostra contexto ± janela_contexto.
    - Opcionalmente, pode reutilizar a análise de estrada completa.

    Retorna dict com:
        - df_alvo
        - df_contexto
    """
    if df is None or df.empty:
        raise ValueError("Histórico vazio no Replay LIGHT.")

    df = df.reset_index(drop=True)
    if not (0 <= idx_alvo_zero < len(df)):
        raise ValueError("Índice alvo fora do intervalo no Replay LIGHT.")

    idx0 = idx_alvo_zero
    df_alvo = df.iloc[[idx0]].copy()

    i_ini = max(0, idx0 - janela_contexto)
    i_fim = min(len(df), idx0 + janela_contexto + 1)
    df_contexto = df.iloc[i_ini:i_fim].copy()

    return {
        "df_alvo": df_alvo,
        "df_contexto": df_contexto,
    }


def replay_ultra_blocos_core(
    df: pd.DataFrame,
    tamanho_bloco: int = 100,
    passo: int = 50,
) -> pd.DataFrame:
    """
    Núcleo do Replay ULTRA (sem UI):

    - Varre a estrada em blocos de tamanho_bloco.
    - Para cada bloco:
        * Executa a análise completa da estrada naquele trecho.
        * Extrai dispersão global local e regime.

    Saída:
    - DataFrame com:
        * inicio, fim, qtd
        * disp_local
        * regime_local
        * qds_global (para o bloco)
    """
    if df is None or df.empty:
        raise ValueError("Histórico vazio no Replay ULTRA.")

    df = df.reset_index(drop=True)
    df = limitar_df(df, MAX_LINHAS_AUTO, "Replay ULTRA")

    resultados = []
    total = len(df)
    tamanho_bloco = max(10, min(tamanho_bloco, 500))
    passo = max(5, min(passo, 500))

    contador_blocos = 0
    for inicio in range(0, total, passo):
        fim = inicio + tamanho_bloco
        if inicio >= total:
            break
        fim = min(total, fim)

        sub = df.iloc[inicio:fim].copy()
        if sub.empty:
            continue

        # Analisa estrada localmente (S1–S7) neste bloco
        stats_local = analisar_estrada_completa(sub)
        disp_local = st.session_state.get("disp_global", None)
        regime_local = st.session_state.get("regime_estrada", None)
        qds_bloco = st.session_state.get("qds_global", None)

        resultados.append(
            {
                "inicio": inicio + 1,
                "fim": fim,
                "qtd": fim - inicio,
                "disp_local": disp_local,
                "regime_local": regime_local,
                "qds_bloco": qds_bloco,
            }
        )

        contador_blocos += 1
        if contador_blocos >= MAX_BLOCOS_REPLAY:
            break

    return pd.DataFrame(resultados)


def replay_unitario_core(
    df: pd.DataFrame,
    idx_alvo_zero: int,
    janela_local: int = 20,
) -> Dict[str, Any]:
    """
    Núcleo do Replay ULTRA Unitário (sem UI):

    - Foca em 1 série alvo (idx_alvo_zero).
    - Considera uma janela local ± janela_local ao redor do alvo.
    - Analisa o regime local daquela janela via S1–S7.
    - Retorna:
        * df_alvo (linha alvo)
        * df_local (janela)
        * stats_local (S4 da janela)
        * regime_local
        * qds_local (QDS Global da janela)
    """
    if df is None or df.empty:
        raise ValueError("Histórico vazio no Replay ULTRA Unitário.")

    df = df.reset_index(drop=True)
    if not (0 <= idx_alvo_zero < len(df)):
        raise ValueError("Índice alvo fora do intervalo no Replay ULTRA Unitário.")

    idx0 = idx_alvo_zero
    i_ini = max(0, idx0 - janela_local)
    i_fim = min(len(df), idx0 + janela_local + 1)

    df_local = df.iloc[i_ini:i_fim].copy()
    df_local = limitar_df(df_local, MAX_LINHAS_AUTO, "Replay ULTRA Unitário")

    # Analisa estrada local na janela
    stats_local = analisar_estrada_completa(df_local)
    regime_local = st.session_state.get("regime_estrada", None)
    qds_local = st.session_state.get("qds_global", None)

    df_alvo = df.iloc[[idx0]].copy()

    return {
        "df_alvo": df_alvo,
        "df_local": df_local,
        "stats_local": stats_local,
        "regime_local": regime_local,
        "qds_local": qds_local,
    }
# ============================================================
# PARTE 11/24 — MONITOR DE RISCO (k & k*) + INTERPRETAÇÃO
# ============================================================

def avaliar_risco_k(k_medio: Optional[float], k_max: Optional[int]) -> str:
    """
    Avalia risco baseado em k real.
    Quanto maior o k (médio ou máximo), maior a sensibilidade da estrada.
    """
    if k_medio is None or k_max is None:
        return "Indefinido — histórico sem coluna k."

    if k_max >= 10:
        return "🔴 k muito alto — forte sensibilidade dos guardas."
    elif k_max >= 5:
        return "🟡 k elevado — atenção ao regime."
    elif k_medio >= 2:
        return "🟡 k moderado — comportamento relevante."
    else:
        return "🟢 k baixo — impacto reduzido no ambiente."


def interpretar_regime_estrada(regime: Optional[str]) -> str:
    """
    Reforço textual do regime da estrada (S7).
    """
    if not regime:
        return "Indefinido — execute a análise da estrada."

    if "🟢" in regime:
        return "🟢 Estrada estável — previsões mais concentradas."
    if "🟡" in regime:
        return "🟡 Estrada moderada — equilíbrio entre ruído e estabilidade."
    if "🔴" in regime:
        return "🔴 Estrada turbulenta — previsões mais amplas recomendadas."
    return regime


def interpretar_qds_global(qds: Optional[float]) -> str:
    """
    Interpreta a QDS global.
    QDS = 100 => estrada perfeita
    QDS = 0   => estrada extremamente caótica
    """
    if qds is None:
        return "Indefinido — execute o pipeline S1–S7."

    if qds >= 80:
        return f"🟢 QDS Global: {qds:.1f}% — alta qualidade dinâmica."
    elif qds >= 60:
        return f"🟡 QDS Global: {qds:.1f}% — qualidade intermediária."
    elif qds >= 40:
        return f"🟠 QDS Global: {qds:.1f}% — atenção reforçada."
    else:
        return f"🔴 QDS Global: {qds:.1f}% — estrada altamente instável."


def interpretar_ruido_condicional(ruido_cond: Dict[str, float]) -> str:
    """
    Interpretação textual do ruído condicional global.
    """
    if not ruido_cond:
        return "Ruído global não pôde ser estimado."

    valores = list(ruido_cond.values())
    ruido_m = float(np.mean(valores))

    if ruido_m < 0.03:
        return "🟢 Ruído condicional muito baixo — excelente estabilidade."
    elif ruido_m < 0.06:
        return "🟡 Ruído condicional moderado — cenário controlado."
    elif ruido_m < 0.10:
        return "🟠 Ruído condicional elevado — atenção às variações bruscas."
    else:
        return "🔴 Ruído condicional muito alto — risco significativo."


def consolidar_monitor_risco(df: pd.DataFrame) -> Dict[str, str]:
    """
    Consolida tudo para o Monitor de Risco 🚨:

    - k_medio, k_max
    - regime da estrada (S7)
    - k* qualitativo
    - QDS global
    - Ruído condicional global (V15)
    - Nível de dispersão global (S4/S7)
    """
    df = df.reset_index(drop=True)

    # === 1. k real ===
    k_max = st.session_state.get("k_max", None)
    k_medio = st.session_state.get("k_medio", None)
    risco_k = avaliar_risco_k(k_medio, k_max)

    # === 2. regime ===
    regime = st.session_state.get("regime_estrada", None)
    risco_regime = interpretar_regime_estrada(regime)

    # === 3. k* qualitativo ===
    k_star_info = st.session_state.get("k_star_qual", "Indefinido.")

    # === 4. QDS Global ===
    qds_global = st.session_state.get("qds_global", None)
    risco_qds = interpretar_qds_global(qds_global)

    # === 5. ruído condicional ===
    ruido_global = calcular_ruido_condicional_global(df)
    risco_ruido = interpretar_ruido_condicional(ruido_global)

    # === 6. dispersão global ===
    disp_global = st.session_state.get("disp_global", None)
    if disp_global is None:
        risco_disp = "Dispersão desconhecida."
    elif disp_global < 0.6:
        risco_disp = f"🟢 Dispersão Global: {disp_global:.3f} (baixa)."
    elif disp_global < 1.2:
        risco_disp = f"🟡 Dispersão Global: {disp_global:.3f} (média)."
    else:
        risco_disp = f"🔴 Dispersão Global: {disp_global:.3f} (alta)."

    # Consolidação
    resumo = {
        "risco_k": risco_k,
        "risco_regime": risco_regime,
        "k_star_info": k_star_info,
        "risco_qds": risco_qds,
        "risco_ruido": risco_ruido,
        "risco_disp": risco_disp,
    }

    # guarda no session_state
    st.session_state["monitor_risco_resumo"] = resumo
    return resumo
# ============================================================
# PARTE 12/24 — PAINEL: 📥 Histórico — Entrada FLEX ULTRA (UI)
# ============================================================

def painel_historico():
    st.markdown("## 📥 Histórico — Entrada FLEX ULTRA (V15.5.2-HÍBRIDO)")

    col_a, col_b = st.columns(2)

    # ------------------------------------------------------------
    # COLUNA A — Upload de arquivo
    # ------------------------------------------------------------
    with col_a:
        st.subheader("1) Upload de arquivo (.csv)")

        formato = st.selectbox(
            "Formato do histórico:",
            [
                "CSV FLEX (id + n1..nN [+k])",
                "CSV com coluna de séries (equivalente a FLEX)",
            ],
            help=(
                "Ambas as opções levam para o mesmo normalizador FLEX ULTRA.\n"
                "A diferença é apenas descritiva."
            ),
        )

        arquivo = st.file_uploader(
            "Selecione o arquivo de histórico (.csv):",
            type=["csv"],
        )

        if st.button("Carregar histórico do arquivo", use_container_width=True):
            if not arquivo:
                st.warning("Selecione um arquivo antes de carregar.")
            else:
                try:
                    with st.spinner("Lendo e normalizando histórico (UPLOAD)..."):
                        df = carregar_historico_upload(arquivo, formato)
                        # aplica ANTI-ZUMBI hard limit aqui também
                        df = limitar_df(df, MAX_LINHAS_ABSOLUTO, "Upload de histórico")
                        st.session_state["df"] = df
                        # reset de derivados
                        st.session_state["df_s1"] = None
                        st.session_state["df_s2"] = None
                        st.session_state["df_s3"] = None
                        st.session_state["df_s4"] = None
                        st.session_state["df_s5"] = None
                        st.session_state["df_s6"] = None
                        st.session_state["df_s7"] = None
                        st.session_state["qds_global"] = None
                        st.session_state["disp_global"] = None
                        st.session_state["regime_estrada"] = None
                        st.session_state["k_star_qual"] = None
                        st.session_state["k_max"] = None
                        st.session_state["k_medio"] = None
                        st.success(f"Histórico carregado com **{len(df)}** linhas.")
                except Exception as e:
                    st.error(f"Erro ao carregar histórico do arquivo: {e}")

    # ------------------------------------------------------------
    # COLUNA B — Texto colado
    # ------------------------------------------------------------
    with col_b:
        st.subheader("2) Colar texto (CSV)")

        texto = st.text_area(
            "Cole aqui o histórico em formato CSV (C1;41;5;4;52;30;33;0 ...):",
            height=200,
        )

        if st.button("Carregar histórico do texto", use_container_width=True):
            if not texto.strip():
                st.warning("Cole o texto do histórico antes de carregar.")
            else:
                try:
                    with st.spinner("Lendo e normalizando histórico (TEXTO)..."):
                        df = carregar_historico_texto(texto, formato)
                        df = limitar_df(df, MAX_LINHAS_ABSOLUTO, "Texto de histórico")
                        st.session_state["df"] = df
                        # reset de derivados (mesmo do upload)
                        st.session_state["df_s1"] = None
                        st.session_state["df_s2"] = None
                        st.session_state["df_s3"] = None
                        st.session_state["df_s4"] = None
                        st.session_state["df_s5"] = None
                        st.session_state["df_s6"] = None
                        st.session_state["df_s7"] = None
                        st.session_state["qds_global"] = None
                        st.session_state["disp_global"] = None
                        st.session_state["regime_estrada"] = None
                        st.session_state["k_star_qual"] = None
                        st.session_state["k_max"] = None
                        st.session_state["k_medio"] = None
                        st.success(f"Histórico carregado com **{len(df)}** linhas.")
                except Exception as e:
                    st.error(f"Erro ao carregar histórico do texto: {e}")

    st.markdown("---")

    # ------------------------------------------------------------
    # CONTROLE ANTI-ZUMBI — Limite de linhas em uso
    # ------------------------------------------------------------
    st.markdown("### 🧯 Controle ANTI-ZUMBI — Limite de linhas em uso")

    col_l1, col_l2 = st.columns([2, 1])

    with col_l1:
        max_user = st.number_input(
            "Máximo de linhas do histórico para usar nos cálculos (0 = automático):",
            min_value=0,
            max_value=MAX_LINHAS_ABSOLUTO,
            value=st.session_state.get("max_linhas_user", MAX_LINHAS_AUTO),
            step=100,
            help=(
                "Este limite vale para os módulos internos (S1–S7, Replays, TURBO++, "
                "Backtests, Monte Carlo etc.).\n\n"
                "0 = modo automático (usa até "
                f"{MAX_LINHAS_AUTO} linhas)."
            ),
        )
        st.session_state["max_linhas_user"] = int(max_user)

    with col_l2:
        df_current = st.session_state.get("df", None)
        if df_current is not None and not df_current.empty:
            st.info(
                f"Histórico carregado: **{len(df_current)}** linhas.\n\n"
                f"Limite em uso: "
                f"**{st.session_state['max_linhas_user'] or MAX_LINHAS_AUTO}** linhas."
            )
        else:
            st.info("Nenhum histórico carregado ainda.")

    # ------------------------------------------------------------
    # VISÃO RÁPIDA DO HISTÓRICO
    # ------------------------------------------------------------
    st.markdown("### 🔎 Visão rápida do histórico carregado")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue um histórico para visualizar.")
        return

    col_prev1, col_prev2 = st.columns([3, 1])

    with col_prev1:
        st.dataframe(df.head(20), use_container_width=True)

    with col_prev2:
        colunas = list(df.columns)
        texto_cols = "\n".join([f"- {c}" for c in colunas])
        st.markdown("**Colunas detectadas:**")
        st.code(texto_cols)

        if "k" in df.columns:
            st.success("Coluna **k** detectada no histórico (número de guardas).")
        else:
            st.info("Nenhuma coluna **k** detectada. k será tratado como ausente.")

        st.caption(
            "A partir deste painel, os demais módulos (Pipeline, Replay, TURBO++, "
            "Monitor de Risco, Testes, Ruído Condicional) usarão este histórico "
            "com o limite ANTI-ZUMBI configurado acima."
        )
# ============================================================
# PARTE 13/24 — PAINEL: 🔍 Pipeline V15.5.2 (S1–S7 Completo)
# ============================================================

def painel_pipeline():
    st.markdown("## 🔍 Pipeline V15.5.2 — Execução S1–S7 Completo")
    st.caption(
        "Pipeline híbrido com camadas S1–S7, análise global, regime, QDS, "
        "k*, dispersão, e normalização total FLEX ULTRA."
    )

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue um histórico no painel 📥 Histórico — Entrada.")
        return

    # --------------------------------------------------------
    # Aplicar limite ANTI-ZUMBI ao histórico
    # --------------------------------------------------------
    max_linhas_user = st.session_state.get("max_linhas_user", MAX_LINHAS_AUTO)
    df_lim = limitar_df(df, max_linhas_user, "Pipeline")

    st.info(
        f"Usando **{len(df_lim)}** linhas do histórico para rodar o Pipeline "
        "(limite ANTI-ZUMBI já aplicado)."
    )

    # --------------------------------------------------------
    # Botão de execução
    # --------------------------------------------------------
    if st.button("🚀 Executar Pipeline (S1–S7)", use_container_width=True):
        try:
            with st.spinner("Executando S1 → S7 (pipeline completo)..."):
                stats = analisar_estrada_completa(df_lim)

                # Registrar DataFrames S1–S7 para visualização
                st.session_state["df_s1"] = st.session_state.get("df_s1")
                st.session_state["df_s2"] = st.session_state.get("df_s2")
                st.session_state["df_s3"] = st.session_state.get("df_s3")
                st.session_state["df_s4"] = st.session_state.get("df_s4")
                st.session_state["df_s5"] = st.session_state.get("df_s5")
                st.session_state["df_s6"] = st.session_state.get("df_s6")
                st.session_state["df_s7"] = st.session_state.get("df_s7")

                # Atualizar k real
                calcular_k_real(df_lim)

                st.success("Pipeline executado com sucesso. Camadas disponíveis abaixo.")
        except Exception as e:
            st.error(f"Erro ao executar o Pipeline: {e}")
            return

    st.markdown("---")

    # --------------------------------------------------------
    # VISUALIZAÇÃO DAS CAMADAS
    # --------------------------------------------------------
    st.markdown("### 📌 Camadas S1–S7 disponíveis")

    abas = st.tabs([
        "S1 — Filtro e Clamping",
        "S3/S4 — Métricas Locais & Estatísticas Globais",
        "S6 — Profundo (Regimes)",
        "S7 — Final (QDS, k*, Regime)",
        "Resumo Global"
    ])

    # --------------------- S1 ---------------------------
    with abas[0]:
        df_s1 = st.session_state.get("df_s1", None)
        if df_s1 is None or df_s1.empty:
            st.info("Execute o Pipeline para gerar S1.")
        else:
            st.dataframe(df_s1.head(50), use_container_width=True)
            st.caption("Clamping, limpeza e padronização mínima.")

    # -------------------- S3/S4 -------------------------
    with abas[1]:
        df_s3 = st.session_state.get("df_s3", None)
        df_s4 = st.session_state.get("df_s4", None)
        col_s3, col_s4 = st.columns(2)

        with col_s3:
            st.markdown("#### 🔹 S3 — Métricas Locais")
            if df_s3 is None or df_s3.empty:
                st.info("Execute o Pipeline para gerar S3.")
            else:
                st.dataframe(df_s3.head(50), use_container_width=True)

        with col_s4:
            st.markdown("#### 🔹 S4 — Estatísticas Globais")
            if df_s4 is None or df_s4.empty:
                st.info("Execute o Pipeline para gerar S4.")
            else:
                st.dataframe(df_s4, use_container_width=True)

    # ---------------------- S6 --------------------------
    with abas[2]:
        df_s6 = st.session_state.get("df_s6", None)
        if df_s6 is None or df_s6.empty:
            st.info("Execute o Pipeline para gerar S6.")
        else:
            st.dataframe(df_s6.head(50), use_container_width=True)
            st.caption("Classificação de regimes locais — estabilidade/turbulência.")

    # ---------------------- S7 --------------------------
    with abas[3]:
        df_s7 = st.session_state.get("df_s7", None)
        if df_s7 is None or df_s7.empty:
            st.info("Execute o Pipeline para gerar S7.")
        else:
            st.dataframe(df_s7.head(50), use_container_width=True)
            st.caption(
                "QDS Global, k* qualitativo, síntese final do regime e métricas de estabilidade."
            )

        st.markdown("#### 🔸 Indicadores Globais (QDS / Regime / k*)")

        qds_global = st.session_state.get("qds_global", None)
        regime = st.session_state.get("regime_estrada", None)
        k_star = st.session_state.get("k_star_qual", None)

        st.write(f"**QDS Global:** {qds_global}")
        st.write(f"**Regime da estrada:** {regime}")
        st.write(f"**k*** (sentinela qualitativo): {k_star}")

    # -------------------- Resumo Global ----------------
    with abas[4]:
        disp = st.session_state.get("disp_global", None)
        kmax = st.session_state.get("k_max", None)
        kmed = st.session_state.get("k_medio", None)

        st.metric("Dispersão Global (S4)", f"{disp:.4f}" if disp else "—")
        st.metric("k máximo observado", f"{kmax}" if kmax is not None else "—")
        st.metric("k médio observado", f"{kmed:.2f}" if kmed is not None else "—")

        st.caption(
            "Esta aba sintetiza os principais indicadores que influenciam o TURBO++, "
            "o Monitor de Risco e o modo 6 acertos."
        )
# ============================================================
# PARTE 14/24 — PAINEL: 🎬 Replay LIGHT (UI)
# ============================================================

def painel_replay_light():
    st.markdown("## 🎬 Replay LIGHT — Visão Rápida da Estrada")
    st.caption(
        "Mostra a série alvo e o contexto ±N ao redor dela, sem alterar o pipeline.\n"
        "Útil para validar comportamento imediato do histórico."
    )

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue um histórico primeiro no painel 📥 Histórico — Entrada.")
        return

    df = df.reset_index(drop=True)

    # -----------------------------
    # Anti-zumbi base
    # -----------------------------
    max_linhas_user = st.session_state.get("max_linhas_user", MAX_LINHAS_AUTO)
    df_lim = limitar_df(df, max_linhas_user, "Replay LIGHT")
    n_total = len(df_lim)

    st.info(f"Replay LIGHT usando **{n_total}** linhas do histórico (ANTI-ZUMBI ativo).")

    # -----------------------------
    # Seleção da série alvo
    # -----------------------------
    col1, col2 = st.columns([2, 1])

    with col1:
        idx_alvo = st.number_input(
            "Selecione a série alvo (1 = primeira linha após o limite ANTI-ZUMBI):",
            min_value=1,
            max_value=n_total,
            value=st.session_state.get("idx_replay_light", n_total),
            help="Você pode olhar qualquer ponto da estrada."
        )
        st.session_state["idx_replay_light"] = idx_alvo

    with col2:
        janela_contexto = st.number_input(
            "Janela de contexto (±N):",
            min_value=3,
            max_value=200,
            value=20,
            help="Define quantas séries antes e depois serão exibidas."
        )

    # -----------------------------
    # Botão de execução
    # -----------------------------
    if st.button("🎬 Executar Replay LIGHT", use_container_width=True):
        try:
            with st.spinner("Carregando Replay LIGHT..."):
                resultado = replay_light_core(
                    df_lim,
                    idx_alvo_zero=idx_alvo - 1,
                    janela_contexto=janela_contexto,
                )

                df_alvo = resultado["df_alvo"]
                df_contexto = resultado["df_contexto"]

                st.success("Replay LIGHT carregado com sucesso.")
        except Exception as e:
            st.error(f"Erro ao executar Replay LIGHT: {e}")
            return

        st.markdown("---")

        # -----------------------------
        # EXIBIÇÃO DO RESULTADO
        # -----------------------------
        st.markdown("### 🎯 Série Alvo")

        st.dataframe(df_alvo, use_container_width=True)

        st.markdown("### 🌄 Contexto da Estrada (± janela)")

        st.dataframe(df_contexto, use_container_width=True)

        st.caption(
            "Esta visualização permite entender o comportamento local da estrada "
            "antes de executar o Replay ULTRA, TURBO++ ou o modo 6 acertos."
        )
# ============================================================
# PARTE 15/24 — PAINEL: 🎥 Replay ULTRA (UI)
# ============================================================

def painel_replay_ultra():
    st.markdown("## 🎥 Replay ULTRA — Mapa de Regimes da Estrada")
    st.caption(
        "Varre a estrada em blocos e mede, em cada bloco, a dispersão, o regime e a QDS local.\n"
        "Usa o mesmo pipeline S1–S7, em modo 'scanner' da estrada."
    )

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue um histórico primeiro no painel 📥 Histórico — Entrada.")
        return

    df = df.reset_index(drop=True)

    # -----------------------------
    # Anti-zumbi e limites
    # -----------------------------
    max_linhas_user = st.session_state.get("max_linhas_user", MAX_LINHAS_AUTO)
    df_lim = limitar_df(df, max_linhas_user, "Replay ULTRA")
    n_total = len(df_lim)

    st.info(
        f"Replay ULTRA usando **{n_total}** linhas do histórico "
        "(limite ANTI-ZUMBI já aplicado)."
    )

    # -----------------------------
    # Parâmetros do scanner
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        tamanho_bloco = st.number_input(
            "Tamanho do bloco (nº de séries por bloco):",
            min_value=20,
            max_value=500,
            value=min(120, max(60, n_total // 5)),
            step=10,
            help="Quanto maior o bloco, mais suave o mapa; quanto menor, mais sensível."
        )

    with col2:
        passo = st.number_input(
            "Passo entre blocos:",
            min_value=5,
            max_value=300,
            value=min(80, max(20, tamanho_bloco // 2)),
            step=5,
            help="Define de quantas em quantas séries o bloco 'anda' ao longo da estrada."
        )

    with col3:
        usar_medidor_tempo = st.checkbox(
            "Ativar medidor de tempo ANTI-ZUMBI",
            value=True,
            help="Recomenda-se manter ligado em históricos grandes."
        )

    # -----------------------------
    # Botão de execução
    # -----------------------------
    if st.button("🎥 Executar Replay ULTRA", use_container_width=True):
        try:
            if usar_medidor_tempo:
                ctx = medidor_tempo("Replay ULTRA")
            else:
                # contexto 'buraco negro' (não mede tempo)
                @contextmanager
                def _dummy():
                    yield
                ctx = _dummy()

            with ctx:
                with st.spinner("Varredo a estrada em blocos..."):
                    df_blocos = replay_ultra_blocos_core(
                        df_lim,
                        tamanho_bloco=int(tamanho_bloco),
                        passo=int(passo),
                    )

            if df_blocos is None or df_blocos.empty:
                st.warning("Nenhum bloco foi gerado no Replay ULTRA.")
                return

            st.success(f"Replay ULTRA gerou **{len(df_blocos)}** blocos de análise.")

        except Exception as e:
            st.error(f"Erro ao executar Replay ULTRA: {e}")
            return

        st.markdown("---")

        # -----------------------------
        # EXIBIÇÃO DOS BLOCOS
        # -----------------------------
        st.markdown("### 🗺️ Mapa de Blocos (Regimes & QDS)")

        st.dataframe(df_blocos, use_container_width=True)

        st.caption(
            "Cada linha representa um bloco da estrada, com sua dispersão local, "
            "regime (🟢/🟡/🔴) e QDS aproximada do trecho. "
            "Este mapa é base para encontrar trechos bons para ataque (ex.: modo 6 acertos)."
        )

        # -----------------------------
        # RESUMO SINTÉTICO
        # -----------------------------
        st.markdown("### 📊 Resumo dos Regimes Encontrados")

        regimes = df_blocos["regime_local"].value_counts(dropna=False)
        for reg, qtd in regimes.items():
            reg_label = reg if isinstance(reg, str) else "Indefinido"
            st.write(f"- **{reg_label}**: {qtd} bloco(s)")

        qds_vals = df_blocos["qds_bloco"].dropna()
        if not qds_vals.empty:
            st.write(
                f"QDS média dos blocos: **{float(qds_vals.mean()):.1f}%** "
                f"(mín: {float(qds_vals.min()):.1f}%, máx: {float(qds_vals.max()):.1f}%)"
            )
# ============================================================
# PARTE 16/24 — PAINEL: 🎯 Replay ULTRA Unitário (UI)
# ============================================================

def painel_replay_unitario():
    st.markdown("## 🎯 Replay ULTRA Unitário — Análise Local da Estrada")
    st.caption(
        "Foca em uma única série alvo e analisa profundamente a janela local "
        "ao redor dela usando o pipeline S1–S7 completo."
    )

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue um histórico primeiro no painel 📥 Histórico — Entrada.")
        return

    df = df.reset_index(drop=True)

    # -----------------------------
    # Anti-zumbi
    # -----------------------------
    max_linhas_user = st.session_state.get("max_linhas_user", MAX_LINHAS_AUTO)
    df_lim = limitar_df(df, max_linhas_user, "Replay Unitário")
    n_total = len(df_lim)

    st.info(
        f"Replay ULTRA Unitário usando **{n_total}** linhas do histórico "
        "(limite ANTI-ZUMBI ativo)."
    )

    # -----------------------------
    # Parâmetros de entrada
    # -----------------------------
    col1, col2 = st.columns([2, 1])

    with col1:
        idx_alvo = st.number_input(
            "Selecione a série alvo (1-based):",
            min_value=1,
            max_value=n_total,
            value=st.session_state.get("idx_replay_unitario", n_total),
            help="Esta é a série alvo para análise profunda."
        )
        st.session_state["idx_replay_unitario"] = idx_alvo

    with col2:
        janela_local = st.number_input(
            "Janela local (±N):",
            min_value=5,
            max_value=200,
            value=30,
            help="Define o intervalo local a ser analisado ao redor da série alvo."
        )

    # -----------------------------
    # Botão de execução
    # -----------------------------
    if st.button("🔍 Executar Replay ULTRA Unitário", use_container_width=True):
        try:
            with st.spinner("Executando Replay ULTRA Unitário..."):
                resultado = replay_unitario_core(
                    df_lim,
                    idx_alvo_zero=idx_alvo - 1,
                    janela_local=janela_local,
                )

                df_alvo = resultado["df_alvo"]
                df_local = resultado["df_local"]
                stats_local = resultado["stats_local"]
                regime_local = resultado["regime_local"]
                qds_local = resultado["qds_local"]
        except Exception as e:
            st.error(f"Erro ao executar Replay ULTRA Unitário: {e}")
            return

        st.success("Replay ULTRA Unitário executado com sucesso.")
        st.markdown("---")

        # -----------------------------------------------------------
        # 1. Série Alvo
        # -----------------------------------------------------------
        st.markdown("### 🎯 Série Alvo")
        st.dataframe(df_alvo, use_container_width=True)

        st.markdown("---")

        # -----------------------------------------------------------
        # 2. Janela Local (± janela)
        # -----------------------------------------------------------
        st.markdown("### 🌐 Janela Local (ambiente da estrada)")
        st.dataframe(df_local, use_container_width=True)
        st.caption(
            "Este é o trecho real da estrada usado para calcular regime local, "
            "QDS local, estabilidade e dispersão."
        )

        st.markdown("---")

        # -----------------------------------------------------------
        # 3. Estatísticas S4 Local
        # -----------------------------------------------------------
        st.markdown("### 📊 Estatísticas S4 — Janela Local")
        st.dataframe(stats_local, use_container_width=True)

        st.caption(
            "Estas estatísticas vêm da camada S4 aplicada **somente** à janela local, "
            "mostrando médias, desvios e estabilidade dos passageiros localmente."
        )

        st.markdown("---")

        # -----------------------------------------------------------
        # 4. Indicadores Gerais Locais
        # -----------------------------------------------------------
        st.markdown("### 🧭 Indicadores Locais (Regime / QDS / Dispersão)")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Regime Local", value=str(regime_local))

        with col_b:
            if qds_local is not None:
                st.metric("QDS Local (%)", value=f"{qds_local:.2f}%")
            else:
                st.metric("QDS Local (%)", value="—")

        with col_c:
            disp_local = st.session_state.get("disp_global", None)
            if disp_local is not None:
                st.metric("Dispersão Local", value=f"{disp_local:.4f}")
            else:
                st.metric("Dispersão Local", value="—")

        st.caption(
            "Estes indicadores são fundamentais para entender a estabilidade do trecho "
            "no qual sua série alvo está localizada."
        )
# ============================================================
# PARTE 17/24 — PAINEL: 🚨 Monitor de Risco (k & k*) — UI
# ============================================================

def painel_monitor_risco():
    st.markdown("## 🚨 Monitor de Risco — k & k* (V15.5.2-Híbrido)")
    st.caption(
        "Avaliação dinâmica da estrada usando k real, k*, regime global, QDS e ruído condicional.\n"
        "Usado por TURBO++, Replay ULTRA e modo 6 acertos."
    )

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue um histórico no painel 📥 Histórico — Entrada.")
        return

    df = df.reset_index(drop=True)

    # -----------------------------
    # Anti-zumbi aplicado ao monitor
    # -----------------------------
    max_linhas_user = st.session_state.get("max_linhas_user", MAX_LINHAS_AUTO)
    df_lim = limitar_df(df, max_linhas_user, "Monitor de Risco")

    st.info(
        f"Monitor de Risco usando **{len(df_lim)}** linhas do histórico "
        "(limite ANTI-ZUMBI ativo)."
    )

    # -----------------------------
    # Botão para avaliar risco
    # -----------------------------
    if st.button("🚨 Atualizar Monitor de Risco", use_container_width=True):
        try:
            with st.spinner("Calculando riscos (k, k*, regime, QDS, ruído)..."):
                resumo = consolidar_monitor_risco(df_lim)
            st.success("Monitor de risco atualizado com sucesso.")
        except Exception as e:
            st.error(f"Erro ao atualizar Monitor de Risco: {e}")
            return
    else:
        resumo = st.session_state.get("monitor_risco_resumo", None)
        if resumo is None:
            st.info("Clique em **Atualizar Monitor de Risco** para gerar os indicadores.")
            return

    st.markdown("---")

    # -----------------------------
    # Indicadores principais
    # -----------------------------
    st.markdown("### 🔎 Indicadores Principais")

    col1, col2, col3 = st.columns(3)

    with col1:
        kmax = st.session_state.get("k_max", None)
        kmed = st.session_state.get("k_medio", None)
        st.metric("k máximo", value=f"{kmax}" if kmax is not None else "—")
        st.metric("k médio", value=f"{kmed:.2f}" if kmed is not None else "—")

    with col2:
        regime = st.session_state.get("regime_estrada", None)
        st.metric("Regime da Estrada", value=str(regime))

    with col3:
        qds = st.session_state.get("qds_global", None)
        if qds is not None:
            st.metric("QDS Global (%)", value=f"{qds:.2f}%")
        else:
            st.metric("QDS Global (%)", value="—")

    st.markdown("---")

    # -----------------------------
    # Narrativa de risco estruturada
    # -----------------------------
    st.markdown("### 📢 Narrativa de Risco Completa")

    risco_k = resumo.get("risco_k", "")
    risco_reg = resumo.get("risco_regime", "")
    risco_qds = resumo.get("risco_qds", "")
    risco_ruido = resumo.get("risco_ruido", "")
    risco_disp = resumo.get("risco_disp", "")
    k_star = resumo.get("k_star_info", "")

    st.write(f"**k real** → {risco_k}")
    st.write(f"**Regime atual** → {risco_reg}")
    st.write(f"**k*** (sentinela) → {k_star}")
    st.write(f"**QDS Global** → {risco_qds}")
    st.write(f"**Ruído Condicional** → {risco_ruido}")
    st.write(f"**Dispersão Global** → {risco_disp}")

    st.markdown("---")

    # -----------------------------
    # Interpretação estratégica final
    # -----------------------------
    st.markdown("### 🧠 Interpretação Estratégica (para TURBO++, Replay e 6 acertos)")

    interpretacao = []

    # A) k real
    if "🔴" in risco_k:
        interpretacao.append("⚠️ k muito alto: previsões podem exigir leques mais amplos.")
    elif "🟡" in risco_k:
        interpretacao.append("🔶 k moderado/alto: ajuste leve de ruído recomendado.")
    else:
        interpretacao.append("🟢 k baixo: estradas menos sensíveis ao modo k.")

    # B) regime
    if "🔴" in risco_reg:
        interpretacao.append("🚨 Estrada turbulenta: regimes ruins para modo 6 acertos.")
    elif "🟡" in risco_reg:
        interpretacao.append("🔶 Regime misto: atenção ao TVF e QDS.")
    else:
        interpretacao.append("🟢 Estrada estável: excelente para previsões concentradas.")

    # C) QDS
    if "🔴" in risco_qds:
        interpretacao.append("🔥 QDS baixa: previsões devem ser conservadoras.")
    elif "🟡" in risco_qds or "🟠" in risco_qds:
        interpretacao.append("🔶 QDS intermediária: recomenda-se testes adicionais.")
    else:
        interpretacao.append("🟢 QDS alta: ambiente favorável para ataques.")

    # D) ruído
    if "🔴" in risco_ruido:
        interpretacao.append("🔥 Ruído muito alto: sugerido evitar previsões agressivas.")
    elif "🟠" in risco_ruido:
        interpretacao.append("🔶 Ruído elevado: preferir modo k̂ para estabilização.")
    else:
        interpretacao.append("🟢 Ruído baixo: ótimo para modo k*.")

    # Final — juntar
    st.write("\n".join(f"- {p}" for p in interpretacao))
# ============================================================
# PARTE 18/24 — PAINEL: 🚀 Modo TURBO++ ULTRA (UI)
# ============================================================

def painel_turbo_ultra():
    st.markdown("## 🚀 Modo TURBO++ ULTRA — Previsões Avançadas (V15.5.2)")
    st.caption(
        "Gerador de previsões avançadas com TVF, ruído calibrado por regime, "
        "QDS Global, k/k̂, e filtros automáticos de confiabilidade.\n"
        "Este é o módulo oficial de ataque do sistema híbrido."
    )

    # -------------------------------------------------------
    # Verificar histórico
    # -------------------------------------------------------
    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue um histórico primeiro no painel 📥 Histórico — Entrada.")
        return

    df = df.reset_index(drop=True)

    # -------------------------------------------------------
    # ANTI-ZUMBI
    # -------------------------------------------------------
    max_linhas_user = st.session_state.get("max_linhas_user", MAX_LINHAS_AUTO)
    df_lim = limitar_df(df, max_linhas_user, "TURBO++ ULTRA")
    n_total = len(df_lim)

    st.info(
        f"Modo TURBO++ usando **{n_total}** linhas do histórico "
        "(ANTI-ZUMBI ativo)."
    )

    colA, colB, colC = st.columns(3)

    # -------------------------------------------------------
    # A) Seleção da série alvo
    # -------------------------------------------------------
    with colA:
        idx_alvo = st.number_input(
            "Índice alvo (1-based):",
            min_value=1,
            max_value=n_total,
            value=st.session_state.get("idx_turbo_ultra", n_total),
            help="Série alvo para base das previsões."
        )
        st.session_state["idx_turbo_ultra"] = idx_alvo

    # -------------------------------------------------------
    # B) Quantidade de séries no leque final
    # -------------------------------------------------------
    with colB:
        n_series = st.number_input(
            "Quantidade final de séries (Top N):",
            min_value=3,
            max_value=300,
            value=50,
            step=1,
            help="Após o TVF, somente as Top N serão mantidas."
        )

    # -------------------------------------------------------
    # C) Confiabilidade mínima
    # -------------------------------------------------------
    with colC:
        confiab_min = st.number_input(
            "Confiabilidade mínima (%):",
            min_value=5,
            max_value=100,
            value=40,
            help="Séries com confiabilidade abaixo deste valor são descartadas."
        )

    st.markdown("---")

    # -------------------------------------------------------
    # Modo k
    # -------------------------------------------------------
    modo_k = st.radio(
        "Modo k:",
        options=["Usar k atual (k*)", "Usar k preditivo (k̂)"],
        index=0,
        help="k* = ambiente atual. k̂ = modo preditivo baseado no futuro estimado."
    )

    # -------------------------------------------------------
    # Barômetro / Regime
    # -------------------------------------------------------
    usar_barometro = st.checkbox(
        "Usar Barômetro / Regime para calibrar ruído",
        value=True,
        help="Recomendado. Desative apenas para análises experimentais."
    )

    st.markdown("---")

    # -------------------------------------------------------
    # Botão principal
    # -------------------------------------------------------
    if st.button("🚀 Executar TURBO++ ULTRA", use_container_width=True):
        try:
            with st.spinner("Gerando leque TURBO++ ULTRA..."):
                df_leque = gerar_leque_turbo_ultra(
                    df_lim,
                    idx_alvo_zero=idx_alvo - 1,
                    n_series=int(n_series),
                    modo_k="k*" if "k*" in modo_k else "k̂",
                    confiab_min=float(confiab_min),
                    usar_barometro=usar_barometro,
                )

            if df_leque is None or df_leque.empty:
                st.warning("Nenhuma série atendeu aos critérios de confiabilidade/TVF.")
                return

            st.session_state["leque_turbo_ultra"] = df_leque
            st.success(f"TURBO++ ULTRA gerou **{len(df_leque)}** séries finais.")

        except Exception as e:
            st.error(f"Erro ao executar TURBO++ ULTRA: {e}")
            return

        st.markdown("---")

        # -------------------------------------------------------
        # EXIBIÇÃO DA TABELA FINAL
        # -------------------------------------------------------
        st.markdown("### 🏁 Previsão Final — TURBO++ ULTRA (Top N)")

        st.dataframe(df_leque, use_container_width=True)

        st.caption(
            "As séries acima já passaram pelo TVF (Top Variability Filter) e "
            "pelo filtro de confiabilidade mínima. "
            "São as **previsões finais oficiais**."
        )

        # -------------------------------------------------------
        # AMOSTRA FINAL (por estética)
        # -------------------------------------------------------
        st.markdown("### 🔚 Previsão recomendada (melhor posição)")

        melhor = df_leque.iloc[0]
        passageiros_final = [melhor[c] for c in melhor.index if c.startswith("n")]
        st.code(f"{' '.join(str(x) for x in passageiros_final)}")

        st.caption(
            "Esta é a série mais bem ranqueada pelo TVF e pela confiabilidade — "
            "geralmente a que o sistema recomenda como previsão principal."
        )
# ============================================================
# PARTE 19/24 — PAINEL: 🧪 Testes de Confiabilidade REAL (UI)
# ============================================================

def painel_testes_confiabilidade():
    st.markdown("## 🧪 Testes de Confiabilidade REAL (V15.5.2-Híbrido)")
    st.caption(
        "Executa análises profundas da estrada: Backtest Interno, Backtest do Futuro, "
        "Monte Carlo Profundo e QDS Local.\n"
        "Fundamental para validar previsões do TURBO++ ULTRA e do modo 6 acertos."
    )

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue um histórico no painel 📥 Histórico — Entrada.")
        return

    df = df.reset_index(drop=True)

    # ------------------------------------------------------------
    # Aplicar ANTI-ZUMBI
    # ------------------------------------------------------------
    max_linhas_user = st.session_state.get("max_linhas_user", MAX_LINHAS_AUTO)
    df_lim = limitar_df(df, max_linhas_user, "Testes de Confiabilidade")
    n_total = len(df_lim)

    st.info(
        f"Testes usarão **{n_total}** linhas do histórico "
        "(ANTI-ZUMBI ativo)."
    )

    st.markdown("---")

    # ------------------------------------------------------------
    # Abas de Testes
    # ------------------------------------------------------------
    aba1, aba2, aba3 = st.tabs([
        "🔎 Backtest Interno",
        "📅 Backtest do Futuro",
        "🌌 Monte Carlo Profundo"
    ])

    # ============================================================
    # 🔎 ABA 1 — BACKTEST INTERNO
    # ============================================================
    with aba1:
        st.markdown("### 🔎 Backtest Interno — Qualidade Dinâmica Local")

        colA, colB = st.columns(2)
        with colA:
            janela = st.number_input(
                "Tamanho da janela (80 recomendado):",
                min_value=20,
                max_value=500,
                value=80,
                step=10,
            )
        with colB:
            passo = st.number_input(
                "Passo entre janelas:",
                min_value=5,
                max_value=200,
                value=10,
                step=5,
            )

        if st.button("🎛️ Executar Backtest Interno", key="bt_interno", use_container_width=True):
            try:
                with st.spinner("Executando Backtest Interno..."):
                    df_bt = backtest_interno(
                        df_lim,
                        passo=int(passo),
                        janela=int(janela),
                    )

                    st.session_state["resultado_backtest"] = df_bt
                    qds_local = calcular_qds_local(df_bt)
            except Exception as e:
                st.error(f"Erro no Backtest Interno: {e}")
                return

            st.success("Backtest Interno concluído.")
            st.markdown("---")

            st.markdown("### 📊 Resultado do Backtest Interno")
            st.dataframe(df_bt, use_container_width=True)

            st.markdown("#### 🔵 QDS Local obtida a partir do Backtest Interno")
            if qds_local is not None:
                st.metric("QDS Local (%)", f"{qds_local:.2f}%")
            else:
                st.metric("QDS Local (%)", "—")

    # ============================================================
    # 📅 ABA 2 — BACKTEST DO FUTURO
    # ============================================================
    with aba2:
        st.markdown("### 📅 Backtest do Futuro — Compatibilidade de Regimes")

        colC, colD, colE = st.columns(3)

        with colC:
            janela_hist = st.number_input(
                "Tamanho da janela histórica:",
                min_value=20,
                max_value=500,
                value=80,
                step=10
            )
        with colD:
            janela_fut = st.number_input(
                "Tamanho da janela futura:",
                min_value=5,
                max_value=200,
                value=20,
                step=5
            )
        with colE:
            passo_bf = st.number_input(
                "Passo entre testes:",
                min_value=5,
                max_value=200,
                value=10,
                step=5
            )

        if st.button("📅 Executar Backtest do Futuro", key="bt_futuro", use_container_width=True):
            try:
                with st.spinner("Executando Backtest do Futuro..."):
                    df_bf = backtest_do_futuro(
                        df_lim,
                        janela_hist=int(janela_hist),
                        horizonte_futuro=int(janela_fut),
                        passo=int(passo_bf)
                    )

                    st.session_state["resultado_backtest_futuro"] = df_bf

            except Exception as e:
                st.error(f"Erro no Backtest do Futuro: {e}")
                return

            st.success("Backtest do Futuro concluído.")
            st.markdown("---")

            st.markdown("### 📊 Resultado do Backtest do Futuro")
            st.dataframe(df_bf, use_container_width=True)

            st.caption(
                "Compatibilidade alta indica que o trecho histórico é bom preditor do futuro.\n"
                "Compatibilidade baixa indica ruptura ou mudança de regime."
            )

    # ============================================================
    # 🌌 ABA 3 — MONTE CARLO PROFUNDO
    # ============================================================
    with aba3:
        st.markdown("### 🌌 Monte Carlo Profundo — Distribuição do Ruído Global")

        colF, colG = st.columns(2)

        with colF:
            n_universos = st.number_input(
                "Número de universos simulados:",
                min_value=50,
                max_value=2000,
                value=500,
                step=50,
                help="Quanto maior, mais precisa a estimativa."
            )

        with colG:
            tam_amostra = st.number_input(
                "Tamanho da amostra por universo:",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
            )

        if st.button("🌌 Executar Monte Carlo Profundo", key="bt_mc", use_container_width=True):
            try:
                with st.spinner("Executando Monte Carlo Profundo..."):
                    df_mc = simular_monte_carlo_profundo(
                        df_lim,
                        n_universos=int(n_universos),
                        tamanho_amostra=int(tam_amostra),
                    )

                    st.session_state["resultado_montecarlo"] = df_mc

            except Exception as e:
                st.error(f"Erro no Monte Carlo Profundo: {e}")
                return

            st.success("Monte Carlo Profundo concluído.")
            st.markdown("---")

            st.markdown("### 📊 Resultados do Monte Carlo Profundo")
            st.dataframe(df_mc, use_container_width=True)

            st.caption(
                "Cada universo representa uma versão alternativa da estrada.\n"
                "A distribuição do ruído médio indica estabilidade ou caos global."
            )
# ============================================================
# PARTE 20/24 — PAINEL: 📊 Ruído Condicional (UI Completa)
# ============================================================

def painel_ruido_condicional():
    st.markdown("## 📊 Ruído Condicional (V15.5.2-Híbrido)")
    st.caption(
        "Analisa a variação local e global dos passageiros usando diffs absolutos, "
        "z-score condicional, regime e mapa temporal.\n"
        "Base para validar rupturas, estabilidade e trechos bons para ataques."
    )

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel 📥 Histórico — Entrada.")
        return

    df = df.reset_index(drop=True)

    # ------------------------------------------------------------
    # Aplicar ANTI-ZUMBI
    # ------------------------------------------------------------
    max_linhas_user = st.session_state.get("max_linhas_user", MAX_LINHAS_AUTO)
    df_lim = limitar_df(df, max_linhas_user, "Ruído Condicional")
    n_total = len(df_lim)

    st.info(
        f"Ruído Condicional será calculado usando **{n_total}** linhas "
        "(ANTI-ZUMBI ativo)."
    )

    st.markdown("---")

    # ------------------------------------------------------------
    # Abas para análise
    # ------------------------------------------------------------
    aba1, aba2, aba3 = st.tabs([
        "🌐 Ruído Global",
        "📈 Ruído Móvel (Mapa Temporal)",
        "🔮 Ruído por Regime (S6)"
    ])

    # ============================================================
    # 🌐 ABA 1 — RUÍDO GLOBAL
    # ============================================================
    with aba1:
        st.markdown("### 🌐 Ruído Condicional Global")

        if st.button("Calcular Ruído Global", use_container_width=True):
            try:
                with st.spinner("Calculando ruído condicional global..."):
                    ruido_global = calcular_ruido_condicional_global(df_lim)
                    st.session_state["ruido_global"] = ruido_global
            except Exception as e:
                st.error(f"Erro ao calcular ruído global: {e}")
                return

            st.success("Ruído Condicional Global calculado.")

            if not ruido_global:
                st.warning("Não foi possível calcular ruído global.")
            else:
                valores = list(ruido_global.values())
                ruido_medio = float(np.mean(valores))

                st.metric("Ruído Médio Global", f"{ruido_medio:.6f}")

                st.markdown("#### 🔹 Ruído por passageiro")
                st.code(
                    "\n".join([f"{c}: {v:.6f}" for c, v in ruido_global.items()])
                )

                st.caption("Valores menores indicam maior estabilidade.")

    # ============================================================
    # 📈 ABA 2 — RUÍDO MÓVEL (MAPA TEMPORIAL)
    # ============================================================
    with aba2:
        st.markdown("### 📈 Ruído Condicional Móvel — Mapa Temporal (V15)")

        janela_movel = st.number_input(
            "Tamanho da janela móvel:",
            min_value=10,
            max_value=300,
            value=40,
            step=5,
            help="Usado para calcular estabilidade local ao longo da estrada."
        )

        if st.button("Gerar Mapa de Ruído Móvel", use_container_width=True):
            try:
                with st.spinner("Construindo mapa temporal de ruído..."):
                    df_ruido_movel = calcular_ruido_condicional_movel(
                        df_lim,
                        janela=int(janela_movel)
                    )
                    st.session_state["ruido_movel"] = df_ruido_movel
            except Exception as e:
                st.error(f"Erro ao calcular ruído móvel: {e}")
                return

            if df_ruido_movel is None or df_ruido_movel.empty:
                st.warning("Falha ao gerar ruído móvel.")
                return

            st.success("Mapa de ruído móvel gerado.")
            st.markdown("### 🗺️ Mapa de Ruído Local (por bloco)")

            st.dataframe(df_ruido_movel, use_container_width=True)

            st.caption(
                "Cada linha representa um trecho da estrada com sua estabilidade local.\n"
                "Trechos com menor 'disp_ruido_cond' são mais estáveis."
            )

            disp_vals = df_ruido_movel["disp_ruido_cond"].dropna()
            if not disp_vals.empty:
                st.markdown("### 📊 Sumário da Dispersão Local")
                st.write(
                    f"- Média: **{disp_vals.mean():.6f}**\n"
                    f"- Mínima: **{disp_vals.min():.6f}**\n"
                    f"- Máxima: **{disp_vals.max():.6f}**\n"
                )

    # ============================================================
    # 🔮 ABA 3 — RUÍDO POR REGIME
    # ============================================================
    with aba3:
        st.markdown("### 🔮 Ruído Condicional por Regime (S6)")

        df_s6 = st.session_state.get("df_s6", None)
        ruido_global = st.session_state.get("ruido_global", None)

        if df_s6 is None or df_s6.empty:
            st.info("Execute o Pipeline (S1–S7) para gerar o S6 (regimes).")
        elif ruido_global is None:
            st.info("Calcule primeiro o Ruído Global (Aba 1).")
        else:
            if st.button("Calcular Ruído por Regime", use_container_width=True):
                try:
                    with st.spinner("Ajustando ruído global por regime..."):
                        ruido_regime = estimar_ruido_por_regime(
                            df_s6,
                            ruido_global
                        )
                        st.session_state["ruido_regime"] = ruido_regime
                except Exception as e:
                    st.error(f"Erro ao calcular ruído por regime: {e}")
                    return

                st.success("Ruído por Regime calculado.")
                st.markdown("### 🔹 Ruído Condicional Ajustado por Regime")

                st.code(
                    "\n".join(
                        [f"{c}: {v:.6f}" for c, v in ruido_regime.items()]
                    )
                )

                st.caption(
                    "Regimes 🟢 reduzem ruído — regimes 🔴 aumentam ruído.\n"
                    "Este ruído ajustado é usado internamente pelo TURBO++ ULTRA."
                )
# ============================================================
# PARTE 21/24 — PAINEL DE NAVEGAÇÃO GERAL (UI PRINCIPAL)
# ============================================================

def painel_navegacao():
    st.sidebar.markdown("## 📂 Navegação — Predict Cars V15.5.2-Híbrido")
    st.sidebar.caption(
        "Menu principal com todos os módulos do sistema.\n"
        "Escolha um painel abaixo para visualizar."
    )

    painel_escolhido = st.sidebar.radio(
        "Painéis disponíveis:",
        [
            "📥 Histórico — Entrada",
            "🔍 Pipeline V15.5.2 (S1–S7)",
            "🎬 Replay LIGHT",
            "🎥 Replay ULTRA",
            "🎯 Replay ULTRA Unitário",
            "🚨 Monitor de Risco (k & k*)",
            "🚀 Modo TURBO++ ULTRA",
            "🧪 Testes de Confiabilidade REAL",
            "📊 Ruído Condicional (V15)",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Estado do App")

    df = st.session_state.get("df", None)
    if df is not None and not df.empty:
        st.sidebar.success(f"Histórico carregado: **{len(df)} linhas**")
    else:
        st.sidebar.warning("Nenhum histórico carregado.")

    st.sidebar.markdown("---")
    st.sidebar.caption("Predict Cars V15.5.2 — Núcleo Híbrido Anti-Zumbi")

    # --------------------------------------------------------
    # Roteamento interno do app
    # --------------------------------------------------------
    if painel_escolhido == "📥 Histórico — Entrada":
        painel_historico()

    elif painel_escolhido == "🔍 Pipeline V15.5.2 (S1–S7)":
        painel_pipeline()

    elif painel_escolhido == "🎬 Replay LIGHT":
        painel_replay_light()

    elif painel_escolhido == "🎥 Replay ULTRA":
        painel_replay_ultra()

    elif painel_escolhido == "🎯 Replay ULTRA Unitário":
        painel_replay_unitario()

    elif painel_escolhido == "🚨 Monitor de Risco (k & k*)":
        painel_monitor_risco()

    elif painel_escolhido == "🚀 Modo TURBO++ ULTRA":
        painel_turbo_ultra()

    elif painel_escolhido == "🧪 Testes de Confiabilidade REAL":
        painel_testes_confiabilidade()

    elif painel_escolhido == "📊 Ruído Condicional (V15)":
        painel_ruido_condicional()
# ============================================================
# PARTE 22/24 — FUNÇÃO PRINCIPAL main() E BOOT INICIAL
# ============================================================

def inicializar_estado():
    """
    Inicializa todas as variáveis necessárias no session_state.
    Evita erros de chave inexistente e prepara o app para todos os módulos.
    """

    defaults = {
        # Histórico
        "df": None,

        # Camadas S1–S7
        "df_s1": None,
        "df_s2": None,
        "df_s3": None,
        "df_s4": None,
        "df_s5": None,
        "df_s6": None,
        "df_s7": None,

        # Indicadores globais
        "qds_global": None,
        "disp_global": None,
        "regime_estrada": None,
        "k_star_qual": None,

        # k real
        "k_max": None,
        "k_medio": None,

        # Replay
        "idx_replay_light": None,
        "idx_replay_unitario": None,

        # TURBO++
        "idx_turbo_ultra": None,
        "leque_turbo_ultra": None,

        # Testes de confiabilidade
        "resultado_backtest": None,
        "resultado_backtest_futuro": None,
        "resultado_montecarlo": None,

        # Ruído
        "ruido_global": None,
        "ruido_movel": None,
        "ruido_regime": None,

        # Anti-zumbi
        "max_linhas_user": MAX_LINHAS_AUTO,

        # Resumo de risco
        "monitor_risco_resumo": None,
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def configurar_pagina():
    """
    Configurações gerais da página do Streamlit.
    """
    st.set_page_config(
        page_title="Predict Cars V15.5.2 — Híbrido Anti-Zumbi",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        /* Melhorar visual das tabelas e cabeçalhos */
        .css-1d391kg, .css-1offfwp {
            font-size: 15px !important;
        }
        .stButton>button {
            font-weight: 600;
            border-radius: 6px;
        }
        .stMetric {
            font-weight: 700;
        }
        .css-12w0qpk {
            width: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    """
    Função principal do app:
    - inicializa estado (session_state)
    - configura a página
    - exibe título e descrição
    - chama o painel de navegação (menu principal)
    """

    # 1) Configurar aparência geral
    configurar_pagina()

    # 2) Inicializar session_state
    inicializar_estado()

    # 3) Cabeçalho principal
    st.markdown("# 🚗 Predict Cars V15.5.2 — Híbrido Anti-Zumbi")
    st.caption(
        "Sistema completo de previsão, análise da estrada, testes dinâmicos, "
        "regimes, ruído, TURBO++ ULTRA, TVF e modo k/k̂."
    )

    st.markdown("---")

    # 4) Iniciar painel de navegação
    painel_navegacao()
# ============================================================
# PARTE 23/24 — CABEÇALHO COMPLETO DO ARQUIVO
# Imports + Constantes + Seeds + ANTI-ZUMBI + Utilidades
# ============================================================

# -----------------------------
# IMPORTS PRINCIPAIS DO APP
# -----------------------------
import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import re

from contextlib import contextmanager

# -----------------------------
# CONSTANTES GERAIS DO SISTEMA
# -----------------------------
VALOR_MIN_PASSAGEIRO = 0
VALOR_MAX_PASSAGEIRO = 60

# Anti-zumbi — limites duros
MAX_LINHAS_ABSOLUTO = 20000      # limite físico total absoluto
MAX_LINHAS_AUTO = 6000           # limite automático seguro
MAX_SIMULACOES_TESTES = 2000     # evita Monte Carlo explosivo
MAX_BLOCOS_REPLAY = 600          # limite do Replay ULTRA em blocos

# Seeds (determinismo)
RNG_SEMENTE_TURBO = 1942
RNG_SEMENTE_MONTECARLO = 2718

# ------------------------------------------------------------
# FUNÇÃO: limitar_df (ANTI-ZUMBI central)
# ------------------------------------------------------------
def limitar_df(df: pd.DataFrame, limite_user: int, origem: str) -> pd.DataFrame:
    """
    Aplica o limite ANTI-ZUMBI ao DataFrame.
    - limite_user = valor configurado pelo usuário
    - MAX_LINHAS_AUTO = limite automático seguro
    - MAX_LINHAS_ABSOLUTO = limite físico absoluto

    Retorna um DF cortado no topo com segurança.
    """
    if df is None or df.empty:
        return df

    limite_efetivo = limite_user if limite_user > 0 else MAX_LINHAS_AUTO
    limite_efetivo = min(limite_efetivo, MAX_LINHAS_ABSOLUTO)

    if len(df) <= limite_efetivo:
        return df

    df_lim = df.head(limite_efetivo).copy()
    return df_lim


# ------------------------------------------------------------
# MEDIDOR DE TEMPO (anti zumbi / profilaxia)
# ------------------------------------------------------------
@contextmanager
def medidor_tempo(nome: str = "Bloco"):
    """
    Um medidor simples de tempo para evitar execuções ocultas longas.
    Usado em Replay ULTRA e grandes loops.
    """
    inicio = time.time()
    yield
    dur = time.time() - inicio
    st.info(f"{nome} executado em {dur:.2f} segundos.")


# ============================================================
# PARTE 23 — PARSER E CARREGAMENTO DO HISTÓRICO FLEX ULTRA
# ============================================================

def _detectar_delimitador(texto: str) -> str:
    """
    Detecta o delimitador do CSV textual: ';' ou ','.
    """
    if ";" in texto:
        return ";"
    return ","


def carregar_historico_texto(texto: str, formato: str) -> pd.DataFrame:
    """
    Carrega CSV colado diretamente.
    Suporta:
    - CSV FLEX (id; n1; n2; ...; k)
    - CSV com coluna de séries
    """
    texto = texto.strip()
    if not texto:
        raise ValueError("Texto vazio para carregar histórico.")

    delim = _detectar_delimitador(texto)

    df = pd.read_csv(io.StringIO(texto), sep=delim, header=None)
    df = _normalizar_historico_flex(df)
    return df


def carregar_historico_upload(arquivo, formato: str) -> pd.DataFrame:
    """
    Carrega arquivo CSV.
    Suporta formatos FLEX ULTRA.
    """
    raw = arquivo.read().decode("utf-8", errors="ignore")
    if not raw.strip():
        raise ValueError("Arquivo vazio ou ilegível.")

    delim = _detectar_delimitador(raw)
    df = pd.read_csv(io.StringIO(raw), sep=delim, header=None)
    df = _normalizar_historico_flex(df)
    return df


# ------------------------------------------------------------
# NORMALIZADOR FLEX ULTRA (núcleo)
# ------------------------------------------------------------
def _normalizar_historico_flex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Versão robusta para colagem de texto (HOTFIX V15.5.2)
    - Mantém 'C' no id mesmo se o navegador remover
    - Força coluna 0 a ser string
    - Detecta corretamente k na última coluna
    - Suporta histórico grande colado
    """

    df = df.copy()

    # Garantir que a coluna 0 existe
    if df.shape[1] < 2:
        raise ValueError("Histórico inválido: não há colunas suficientes.")

    # Forçar primeira coluna a STRING
    df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()

    # Restaurar prefixo C se removido
    def restaurar_id(x):
        x = str(x).strip()

        # remover BOM invisível
        x = x.replace("\ufeff", "")

        # já tem C
        if x.startswith("C"):
            return x

        # está só com número
        if x.isdigit():
            return f"C{x}"

        return x

    df["id_raw"] = df.iloc[:, 0].apply(restaurar_id)

    # Extrair número do ID
    import re
    def extrair_num(x):
        m = re.search(r"\d+", x)
        if m:
            return int(m.group())
        return None

    df["id"] = df["id_raw"].apply(extrair_num)

    # Detectar colunas de dados (passageiros + k)
    colunas_dados = list(df.columns[1:])  # todas depois da primeira

    # Verificar se última coluna é k
    ultima = colunas_dados[-1]
    ultima_vals = pd.to_numeric(df[ultima], errors="coerce")

    tem_k = False
    if ultima_vals.notnull().all():
        if ultima_vals.max() <= 20:
            tem_k = True

    passageiros_cols = colunas_dados[:-1] if tem_k else colunas_dados
    k_col = ultima if tem_k else None

    # Criar DF final
    final_cols = ["id"]
    rename_map = {}

    # renomear passageiros
    for i, c in enumerate(passageiros_cols):
        rename_map[c] = f"n{i+1}"
        final_cols.append(f"n{i+1}")

    # renomear k
    if tem_k:
        rename_map[k_col] = "k"
        final_cols.append("k")

    df = df.rename(columns=rename_map)
    df = df[final_cols].copy()

    # converter para int
    for c in df.columns:
        if c.startswith("n"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(0, 60).astype(int)
        if c == "k":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    df = df.dropna().reset_index(drop=True)
    return df



# ============================================================
# UTILIDADES S1–S7 (placeholder estrutural — preenchidas nas partes anteriores)
# ============================================================

# Aqui NÃO repetimos os S1–S7, Replay, TVF, TURBO++, etc.,
# pois já foram definidos nas Partes 1–22.
# Apenas garantimos que o cabeçalho contenha as bases necessárias.
# ============================================================
# PARTE 24/24 — RODAPÉ FINAL DO ARQUIVO
# ============================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Erro fatal na execução do app: {e}")
        st.stop()
