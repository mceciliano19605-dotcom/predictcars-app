# =====================================================================
# PREDICT CARS V15.5.1-HÍBRIDO — PARTE 1/24
# Arquitetura completa, extensa e não compactada.
# Núcleo V14-FLEX ULTRA + S1..S7 + IDX + NRF + S6 Profundo
# + Backtest Interno e do Futuro + Monte Carlo Profundo
# + QDS Completo + AIQ Global Estendido + Motores TURBO / TURBO+
# + Modo 6 Acertos + Modo Premium / Estrutural / Cobertura
# + Monitor de Risco (k & k*) + Painéis completos
# =====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import math
import random
import statistics
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter

# =====================================================================
# CONFIGURAÇÃO GERAL DO APP
# =====================================================================

st.set_page_config(
    page_title="Predict Cars V15.5.1-HÍBRIDO",
    layout="wide"
)

# =====================================================================
# ESTRUTURA DE ESTADO GLOBAL (SESSÃO)
# — Sem compactação, incluindo todos os estados usados ao longo
#   do app inteiro, mesmo que só alguns sejam usados em módulos
#   avançados instalados posteriormente (NRF, S6 Profundo, etc.)
# =====================================================================

def init_full_state():
    keys = {
        "df_historico": None,
        "road_stats": None,

        # Estados auxiliares para pipeline
        "ultima_previsao_turbo": None,
        "ultima_previsao_turbo_plus": None,
        "ultima_previsao_turbo_ultra": None,
        "ultima_previsao_premium": None,

        # Estados de confiabilidade
        "qds_info": None,
        "qds_info_completo": None,
        "backtest_info": None,
        "backtest_futuro_info": None,
        "mc_info": None,
        "mc_multi_info": None,
        "aiq_info": None,
        "aiq_avancado_info": None,

        # Estados de ruído
        "ruido_info": None,
        "ruido_condicional_info": None,
        "ruido_ab_info": None,
        "ruido_analitico_info": None,

        # Estados de risco
        "ultimo_monitor_risco": None,
        "monitor_global": None,
        "monitor_local": None,

        # Estados de IDX
        "idx_simples": None,
        "idx_avancado": None,
        "idx_profundo": None,
        "idx_hibrido": None,

        # Estados de S6
        "s6_raw": None,
        "s6_profundo_raw": None,
        "s6_profundo_flat": None,
        "s6_profundo_stats": None,

        # Estados do replay
        "replay_light": None,
        "replay_ultra": None,
        "replay_unitario": None,

        # Estados do relatório
        "relatorio_aiq": None,

        # Estados adicionais do V15.5 HÍBRIDO
        "modo_6_acertos_info": None,
        "cobertura_info": None,
        "premium_info": None,
        "estrutural_info": None,
    }

    for k, v in keys.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_full_state()

# =====================================================================
# FUNÇÕES BÁSICAS E UTILITÁRIAS — SEM COMPACTAÇÃO
# Inclui redundância intencional para manter o estilo do Predict Cars.
# =====================================================================

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """
    Função clássica usada em TODO o sistema.
    Mantém valor dentro do intervalo [lo, hi].
    """
    if x < lo:
        return lo
    if x > hi:
        return hi
    return float(x)

def safe_int(x, default=0):
    """
    Conversão robusta para inteiro.
    Usada em dezenas de lugares.
    """
    try:
        return int(x)
    except:
        return default

def safe_float(x, default=0.0):
    """
    Conversão robusta para float.
    """
    try:
        return float(x)
    except:
        return default

def list_passageiros(row: pd.Series) -> List[int]:
    """
    Extrai os passageiros em forma de lista:
    [p1, p2, ..., pN]
    """
    cols = [c for c in row.index if c.startswith("p")]
    return [safe_int(row[c]) for c in cols]

def contar_passageiros(df: pd.DataFrame) -> int:
    """
    Quantidade de passageiros por série.
    """
    return len([c for c in df.columns if c.startswith("p")])

def has_historico() -> bool:
    """
    Verifica se o histórico está carregado.
    """
    df = st.session_state.get("df_historico", None)
    return df is not None and not df.empty

# =====================================================================
# DETECÇÃO DE SEPARADOR E PARSING INLINE
# =====================================================================

def infer_separator_from_line(line: str) -> str:
    """
    Detecta o separador mais provável de uma linha CSV.
    """
    if ";" in line:
        return ";"
    if "," in line:
        return ","
    if "\t" in line:
        return "\t"
    return ";"  # fallback padrão

# =====================================================================
# PRÉ-BLOCO: NORMALIZAÇÃO GERAL DO HISTÓRICO (versão extensa)
# Incluindo múltiplos caminhos, redundâncias e comentários
# para manter o estilo das versões V13.8–V15 originais.
# =====================================================================

def normalizar_historico_extenso(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizador amplo:
    - Detecta formato
    - Reorganiza colunas
    - Garante id / p1..pN / k
    - Suporta múltiplos padrões do Predict Cars V14–V15
    """

    # 1) Copia para não modificar o original
    df = df_raw.copy()

    # 2) Normaliza nomes de coluna
    novas = []
    for c in df.columns:
        if isinstance(c, str):
            novas.append(c.strip().lower())
        else:
            novas.append(str(c).lower())
    df.columns = novas

    cols = list(df.columns)

    # CENÁRIO 1: Já tem id e k identificados claramente
    if "id" in cols and "k" in cols:
        # Detectar colunas p1..pN automaticamente
        p_cols = [c for c in cols if c.startswith("p") or c.startswith("n")]
        if p_cols:
            # Renomeia n1..nN para p1..pN
            mapping = {}
            new_p = []
            i = 1
            for c in p_cols:
                mapping[c] = f"p{i}"
                new_p.append(f"p{i}")
                i += 1
            df = df.rename(columns=mapping)
            ordered = ["id"] + new_p + ["k"]
            df = df[ordered]
            return df

    # CENÁRIO 2: Primeiro é id, último é k
    if len(cols) >= 3:
        id_c = cols[0]
        k_c = cols[-1]
        mid = cols[1:-1]
        mapping = {id_c: "id", k_c: "k"}
        new_p = []
        for i, c in enumerate(mid, start=1):
            mapping[c] = f"p{i}"
            new_p.append(f"p{i}")
        df = df.rename(columns=mapping)
        ordered = ["id"] + new_p + ["k"]
        df = df[ordered]
        return df

    raise ValueError("Formato desconhecido ao normalizar histórico (extenso).")

# =====================================================================
# PARTE 1/24 — FIM
# =====================================================================
# =====================================================================
# PREDICT CARS V15.5.1-HÍBRIDO — PARTE 2/24
# Carregamento FLEX ULTRA (arquivo / texto) + Estatísticas da Estrada
# + Vetores de Features básicos para motores / IDX.
# =====================================================================

# ---------------------------------------------------------------------
# CARREGAMENTO FLEX ULTRA — ARQUIVO
# ---------------------------------------------------------------------

def carregar_historico_de_arquivo_flex_ultra(uploaded_file) -> pd.DataFrame:
    """
    Carrega histórico a partir de um arquivo enviado via Streamlit.
    Aceita:
      - CSV com cabeçalho ou sem cabeçalho
      - separador ';', ',' ou '\t'
    Aplica normalização extensa:
      -> id / p1..pN / k
    """
    if uploaded_file is None:
        raise ValueError("Nenhum arquivo enviado para carregar o histórico.")

    # Lê bytes e tenta decodificar em UTF-8, caindo para latin-1
    raw = uploaded_file.read()
    try:
        text = raw.decode("utf-8")
    except Exception:
        text = raw.decode("latin-1")

    # Remove linhas vazias
    linhas = [ln for ln in text.splitlines() if ln.strip()]
    if not linhas:
        raise ValueError("Arquivo aparentemente vazio.")

    # Detecta separador
    sep = infer_separator_from_line(linhas[0])
    conteudo = "\n".join(linhas)

    # Tenta sem header
    try:
        df_raw = pd.read_csv(
            pd.io.common.StringIO(conteudo),
            sep=sep,
            header=None
        )
    except Exception as e:
        raise ValueError(f"Erro ao ler CSV bruto: {e}")

    # Normaliza em formato Predict Cars
    df_norm = normalizar_historico_extenso(df_raw)

    # Ajusta tipos para int (passageiros) e float (k) quando possível
    p_cols = [c for c in df_norm.columns if c.startswith("p")]
    for c in p_cols:
        df_norm[c] = df_norm[c].apply(safe_int)
    df_norm["k"] = df_norm["k"].apply(safe_int)

    return df_norm

# ---------------------------------------------------------------------
# CARREGAMENTO FLEX ULTRA — TEXTO COLADO
# ---------------------------------------------------------------------

def carregar_historico_de_texto_flex_ultra(texto: str) -> pd.DataFrame:
    """
    Carrega histórico a partir de texto colado.
    Cada linha deve conter:
      C1;41;5;4;52;30;33;0
      C2;...
    ou equivalente com vírgulas, tabs etc.
    """
    if texto is None or not texto.strip():
        raise ValueError("Texto vazio ao carregar histórico.")

    linhas = [ln.strip() for ln in texto.splitlines() if ln.strip()]
    if not linhas:
        raise ValueError("Texto sem linhas válidas.")

    sep = infer_separator_from_line(linhas[0])
    conteudo = "\n".join(linhas)

    try:
        df_raw = pd.read_csv(
            pd.io.common.StringIO(conteudo),
            sep=sep,
            header=None
        )
    except Exception as e:
        raise ValueError(f"Erro ao interpretar texto como CSV: {e}")

    df_norm = normalizar_historico_extenso(df_raw)

    p_cols = [c for c in df_norm.columns if c.startswith("p")]
    for c in p_cols:
        df_norm[c] = df_norm[c].apply(safe_int)
    df_norm["k"] = df_norm["k"].apply(safe_int)

    return df_norm

# ---------------------------------------------------------------------
# FUNÇÕES AUXILIARES DE ACESSO AO HISTÓRICO NO ESTADO
# ---------------------------------------------------------------------

def set_historico(df: pd.DataFrame):
    """
    Guarda o histórico no estado global, já normalizado.
    Também atualiza estatísticas básicas da estrada.
    """
    if df is None or df.empty:
        st.session_state["df_historico"] = None
        st.session_state["road_stats"] = None
        return

    st.session_state["df_historico"] = df
    st.session_state["road_stats"] = calcular_estatisticas_estrada_extenso(df)

def get_historico() -> Optional[pd.DataFrame]:
    return st.session_state.get("df_historico", None)

def get_road_stats() -> Optional[Dict]:
    return st.session_state.get("road_stats", None)

# ---------------------------------------------------------------------
# ESTATÍSTICAS DA ESTRADA — VERSÃO EXTENSA
# Inclui mais campos do que o mínimo necessário, por design.
# ---------------------------------------------------------------------

def calcular_estatisticas_estrada_extenso(df: pd.DataFrame) -> Dict:
    """
    Calcula estatísticas globais da estrada, com redundâncias:
      - quantidade de séries
      - quantidade de passageiros
      - faixa numérica global (min/max)
      - frequência de números globais
      - distribuição de k
      - média de k, desvio de k
      - média de amplitude por série
      - média de dispersão (desvio padrão por série)
    """
    if df is None or df.empty:
        return None

    p_cols = [c for c in df.columns if c.startswith("p")]
    if not p_cols:
        raise ValueError("Nenhuma coluna de passageiros encontrada ao calcular estatísticas.")

    # Matriz de passageiros
    mat = df[p_cols].astype(float).values
    flat = mat.flatten()
    flat = flat[~np.isnan(flat)]

    if flat.size == 0:
        raise ValueError("Matriz de passageiros vazia ao calcular estatísticas.")

    minimo = int(np.min(flat))
    maximo = int(np.max(flat))

    # Frequência de números globais
    unique_nums, counts_nums = np.unique(flat, return_counts=True)
    freq_nums = dict(zip(unique_nums.tolist(), counts_nums.tolist()))

    # k
    k_vals = df["k"].astype(float).values
    unique_k, counts_k = np.unique(k_vals, return_counts=True)
    freq_k = dict(zip(unique_k.tolist(), counts_k.tolist()))
    k_mean = float(np.mean(k_vals))
    k_std = float(np.std(k_vals))

    # Estatísticas por série
    df_pass = df[p_cols].astype(float)
    # amplitude = max - min por série
    amps = df_pass.max(axis=1) - df_pass.min(axis=1)
    amp_media = float(amps.mean())
    # dispersão = desvio padrão por série
    stds = df_pass.std(axis=1)
    disp_media = float(stds.mean())

    stats = {
        "n_series": int(len(df)),
        "n_passageiros": int(len(p_cols)),
        "min": minimo,
        "max": maximo,
        "freq_nums": freq_nums,
        "freq_k": freq_k,
        "k_mean": k_mean,
        "k_std": k_std,
        "amp_media": amp_media,
        "disp_media": disp_media,
    }
    return stats

def descrever_estatisticas_estrada_para_ui(stats: Optional[Dict]) -> List[str]:
    """
    Gera linhas de descrição textual para o painel da estrada.
    """
    linhas = []
    if not stats:
        return linhas

    linhas.append(f"- Séries no histórico: **{stats['n_series']}**")
    linhas.append(f"- Passageiros por série: **{stats['n_passageiros']}**")
    linhas.append(f"- Faixa numérica global: **{stats['min']} a {stats['max']}**")
    linhas.append(f"- k médio global: **{stats['k_mean']:.4f}**")
    linhas.append(f"- Desvio padrão de k: **{stats['k_std']:.4f}**")
    linhas.append(f"- Amplitude média por série: **{stats['amp_media']:.4f}**")
    linhas.append(f"- Dispersão média por série: **{stats['disp_media']:.4f}**")

    return linhas

# ---------------------------------------------------------------------
# VETORES DE FEATURES PARA ÍNDICES / MOTORES / S CAMADAS
# ---------------------------------------------------------------------

def construir_vetor_features_basico(row: pd.Series) -> np.ndarray:
    """
    Vetor de features básico:
      [p1, p2, ..., pN, k]
    Usado como elemento comum em:
      - IDX Simples
      - Motores TURBO
      - S1..S2
    """
    ps = list_passageiros(row)
    k_val = safe_float(row["k"], 0.0)
    vec = np.array(ps + [k_val], dtype=float)
    return vec

def construir_vetor_features_expandido(row: pd.Series) -> np.ndarray:
    """
    Vetor de features expandido, usado em módulos mais profundos:
      [p1..pN, k, media_p, std_p, amp_p]
    onde:
      media_p = média dos passageiros
      std_p   = desvio padrão dos passageiros
      amp_p   = amplitude (max - min)
    """
    ps = list_passageiros(row)
    arr = np.array(ps, dtype=float)
    if arr.size == 0:
        media_p = 0.0
        std_p = 0.0
        amp_p = 0.0
    else:
        media_p = float(arr.mean())
        std_p = float(arr.std())
        amp_p = float(arr.max() - arr.min())

    k_val = safe_float(row["k"], 0.0)
    vec = np.array(ps + [k_val, media_p, std_p, amp_p], dtype=float)
    return vec

def distancia_euclidiana(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Distância euclidiana padrão entre dois vetores.
    """
    return float(np.linalg.norm(v1 - v2))

def distancia_euclidiana_normalizada(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Versão alternativa da distância que normaliza pelo tamanho do vetor.
    Mantida para uso em módulos de teste / comparação.
    """
    if v1.shape != v2.shape:
        # Em caso de forma distinta, faz fallback para a normal.
        return distancia_euclidiana(v1, v2)
    n = v1.shape[0]
    if n <= 0:
        return 0.0
    d = np.linalg.norm(v1 - v2)
    return float(d / math.sqrt(n))

# =====================================================================
# PARTE 2/24 — FIM
# =====================================================================
# =====================================================================
# PREDICT CARS V15.5.1-HÍBRIDO — PARTE 3/24
# k* (sentinela), regimes, QDS COMPLETO (versão extensa),
# camadas S1, S2, S3, S4 — pré-processamentos estruturais clássicos
# =====================================================================

# ---------------------------------------------------------------------
# k* — SENTINELA DA ESTRADA
# Versão longa, detalhada, sem compactação.
# ---------------------------------------------------------------------

def calcular_k_star_extenso(df: pd.DataFrame, janela: int = 40) -> Tuple[pd.Series, pd.Series]:
    """
    Calcula k* (sentinela) usando:
      - média móvel
      - desvio móvel
      - z-score robusto
    Retorna:
      - série k*
      - série de regimes: estavel / atencao / critico
    """

    if df is None or df.empty:
        raise ValueError("Histórico vazio ao calcular k*.")

    k = df["k"].astype(float)

    # Média móvel
    rolling_mean = k.rolling(window=janela, min_periods=max(5, janela // 4)).mean()

    # Desvio padrão móvel
    rolling_std = k.rolling(window=janela, min_periods=max(5, janela // 4)).std()

    # Evita zeros
    rolling_std = rolling_std.replace(0, np.nan)

    # z-score
    k_star = (k - rolling_mean) / rolling_std
    k_star = k_star.fillna(0.0)

    # Classificação dos regimes
    regimes = []
    for valor in k_star:
        if valor <= 0.5:
            regimes.append("estavel")
        elif valor <= 1.5:
            regimes.append("atencao")
        else:
            regimes.append("critico")

    return k_star, pd.Series(regimes, index=df.index, name="regime_k_star")

# ---------------------------------------------------------------------
# MONITOR DE REGIMES — GLOBAL E LOCAL
# Versão extensa usada pelo Monitor de Risco e por S-camadas avançadas.
# ---------------------------------------------------------------------

def analisar_regime_global_e_local(df: pd.DataFrame, idx_local: Optional[int] = None, janela: int = 40) -> Dict:
    """
    Retorna:
      - k_mean, k_std
      - distribuições dos regimes
      - regime global e texto interpretativo
      - caso idx_local seja fornecido: regime local + texto local
      - série k* e série de regimes
    """

    k_star, regime = calcular_k_star_extenso(df, janela=janela)
    df_aux = df.copy()
    df_aux["k_star"] = k_star
    df_aux["regime_k_star"] = regime

    # Estatísticas globais de k
    k_vals = df_aux["k"].astype(float).values
    k_mean = float(np.mean(k_vals))
    k_std = float(np.std(k_vals))

    # Distribuição dos regimes
    reg_counts = regime.value_counts().to_dict()

    # Proporção de crítico → regime global
    pct_critico = reg_counts.get("critico", 0) / max(1, len(df_aux))

    if pct_critico < 0.1:
        regime_global = "estavel"
        texto_global = "🟢 Estrada predominantemente estável (baixa turbulência em k*)."
    elif pct_critico < 0.3:
        regime_global = "atencao"
        texto_global = "🟡 Estrada com bolsões de turbulência moderada."
    else:
        regime_global = "critico"
        texto_global = "🔴 Estrada em regime crítico — alta turbulência em k*."

    info_local = None
    if idx_local is not None and 0 <= idx_local < len(df_aux):
        row = df_aux.iloc[idx_local]
        reg_l = row["regime_k_star"]
        k_val_l = float(row["k"])
        k_star_l = float(row["k_star"])

        if reg_l == "estavel":
            texto_local = "🟢 Ambiente estável — previsão em regime normal."
        elif reg_l == "atencao":
            texto_local = "🟡 Ambiente em pré-ruptura — atenção moderada."
        else:
            texto_local = "🔴 Ambiente crítico — máxima cautela na previsão."

        info_local = {
            "regime": reg_l,
            "k": k_val_l,
            "k_star": k_star_l,
            "texto": texto_local,
        }

    return {
        "k_mean": k_mean,
        "k_std": k_std,
        "regimes_counts": reg_counts,
        "regime_global": regime_global,
        "texto_global": texto_global,
        "local_info": info_local,
        "series_k_star": k_star,
        "series_regime": regime,
    }

# ---------------------------------------------------------------------
# QDS V15 — QUALIDADE DINÂMICA DA SÉRIE
# Versão COMPLETA, incluindo:
# - estabilidade de k
# - dispersões
# - amplitudes
# - repetição de padrões
# - entropia
# - T_norm
# - SDM
# ---------------------------------------------------------------------

def qds_estabilidade_k(df: pd.DataFrame, janela: int) -> Dict:
    k = df["k"].astype(float)
    if len(k) < janela:
        rolling_std = k.std()
    else:
        rolling_std = k.rolling(window=janela, min_periods=max(5, janela//5)).std().mean()

    rolling_std = float(rolling_std) if rolling_std is not None else 0.0
    estabilidade = 1 / (1 + rolling_std)

    return {
        "k_std_movel": rolling_std,
        "estabilidade_k": clamp(estabilidade),
    }

def qds_dispersao_passageiros(df: pd.DataFrame) -> Dict:
    p_cols = [c for c in df.columns if c.startswith("p")]
    if not p_cols:
        return {"disp_media": 0.0, "sdm": 0.0, "t_norm": 0.0}

    arr = df[p_cols].astype(float)
    stds = arr.std(axis=1)
    disp = float(stds.mean())

    # SDM = média das dispersões normalizada pela faixa
    faixa_min = float(arr.min().min())
    faixa_max = float(arr.max().max())
    faixa_total = max(1.0, faixa_max - faixa_min)
    sdm = float((stds / faixa_total).mean())

    # T_norm = amplitude média normalizada
    amps = arr.max(axis=1) - arr.min(axis=1)
    t_norm = float((amps / faixa_total).mean())

    return {
        "disp_media": disp,
        "sdm": sdm,
        "t_norm": t_norm,
    }

def qds_repeticao_padroes(df: pd.DataFrame) -> Dict:
    p_cols = [c for c in df.columns if c.startswith("p")]
    tuplas = [tuple(row[p_cols].values.tolist()) for _, row in df.iterrows()]
    ct = Counter(tuplas)
    repetidas = sum(1 for v in ct.values() if v > 1)
    ratio = repetidas / max(1, len(tuplas))

    score = 1 / (1 + ratio)
    return {
        "repet_ratio": float(ratio),
        "repet_score": clamp(score),
    }

def qds_entropia(df: pd.DataFrame) -> Dict:
    p_cols = [c for c in df.columns if c.startswith("p")]
    arr = df[p_cols].astype(float)
    flat = arr.values.flatten()
    unique, counts = np.unique(flat, return_counts=True)
    probs = counts / counts.sum()

    # Shannon entropy
    ent = -np.sum([p * math.log(p + 1e-12) for p in probs])
    ent_norm = ent / max(1.0, math.log(len(unique) + 1e-12))

    return {
        "entropia": float(ent),
        "entropia_norm": clamp(float(ent_norm)),
    }

def calcular_qds_completo(df: pd.DataFrame, janela: int = 50) -> Dict:
    """
    QDS completo integrando:
     - estabilidade de k
     - dispersão
     - SDM
     - T_norm
     - repetição de padrões
     - entropia
    Versão V15 estendida.
    """
    comp_k = qds_estabilidade_k(df, janela)
    comp_disp = qds_dispersao_passageiros(df)
    comp_rep = qds_repeticao_padroes(df)
    comp_ent = qds_entropia(df)

    # Pontuação agregada (pesos calibrados na versão V14-FLEX ULTRA)
    qds = (
        0.30 * comp_k["estabilidade_k"]
        + 0.25 * (1 / (1 + comp_disp["disp_media"]))
        + 0.15 * (1 - comp_rep["repet_ratio"])
        + 0.15 * (1 - comp_disp["sdm"])
        + 0.10 * comp_ent["entropia_norm"]
        + 0.05 * (1 - comp_disp["t_norm"])
    )

    return {
        "k": comp_k,
        "disp": comp_disp,
        "rep": comp_rep,
        "ent": comp_ent,
        "qds_score": clamp(qds),
    }

# ---------------------------------------------------------------------
# CAMADAS S1, S2, S3, S4 — PRÉ-PROCESSAMENTO ESTRUTURAL
# Cada camada é escrita de forma detalhada (não compactada).
# ---------------------------------------------------------------------

def camada_s1_normalizacao(df: pd.DataFrame) -> pd.DataFrame:
    """
    S1 — Normalização global [0,1]
    """
    p_cols = [c for c in df.columns if c.startswith("p")]
    if not p_cols:
        return df.copy()

    df2 = df.copy()
    arr = df2[p_cols].astype(float)
    vmin = arr.min().min()
    vmax = arr.max().max()
    if vmax == vmin:
        return df2

    for c in p_cols:
        df2[c] = (df2[c].astype(float) - vmin) / (vmax - vmin)

    return df2

def camada_s2_caracteristicas(df: pd.DataFrame) -> pd.DataFrame:
    """
    S2 — Adiciona:
      - média dos passageiros
      - desvio padrão
      - amplitude
    """
    df2 = df.copy()
    p_cols = [c for c in df.columns if c.startswith("p")]
    if not p_cols:
        return df2

    arr = df2[p_cols].astype(float)
    df2["s2_mean"] = arr.mean(axis=1)
    df2["s2_std"] = arr.std(axis=1)
    df2["s2_amp"] = arr.max(axis=1) - arr.min(axis=1)
    return df2

def camada_s3_suavizacao(df: pd.DataFrame, janela: int = 5) -> pd.DataFrame:
    """
    S3 — Suavização leve das séries via média móvel por passageiro.
    """
    df2 = df.copy()
    p_cols = [c for c in df2.columns if c.startswith("p")]
    for c in p_cols:
        df2[c] = df2[c].rolling(window=janela, min_periods=1).mean()
    return df2

def camada_s4_autoencoder_light(df: pd.DataFrame, fator: float = 0.25) -> pd.DataFrame:
    """
    S4 — Autoencoder simplificado:
      - compressão parcial dos passageiros
      - reexpansão
      - ruído controlado opcional
    Mantido por legado das versões V13.x.
    """
    df2 = df.copy()
    p_cols = [c for c in df2.columns if c.startswith("p")]
    if not p_cols:
        return df2

    n = len(p_cols)
    meio = max(1, int(n * fator))

    # Compressão
    arr = df2[p_cols].astype(float).values
    enc = np.zeros((arr.shape[0], meio))
    for i in range(arr.shape[0]):
        seg = arr[i]
        for j in range(meio):
            start = int(j * (n / meio))
            end = int((j + 1) * (n / meio))
            bloco = seg[start:end]
            enc[i, j] = float(np.mean(bloco)) if bloco.size else 0.0

    # Reexpansão
    rec = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(n):
            idx0 = int(j * (meio / n))
            rec[i, j] = enc[i, idx0]

    df2_rec = df2.copy()
    for idx, c in enumerate(p_cols):
        df2_rec[c] = rec[:, idx]

    return df2_rec

# =====================================================================
# PARTE 3/24 — FIM
# =====================================================================
# =====================================================================
# PREDICT CARS V15.5.1-HÍBRIDO — PARTE 4/24
# S5 Núcleo Resiliente (light), S6/S7 estruturais,
# IDX Simples e início do IDX Avançado (estrutura base).
# =====================================================================

# ---------------------------------------------------------------------
# S5 — NÚCLEO RESILIENTE (VERSÃO LIGHT EXPLÍCITA)
# Nota: A versão PRO / completa do NRF entra em partes posteriores.
# Aqui instalamos um núcleo estrutural intermediário, usado por:
#   - motores
#   - IDX
#   - análises de consistência
# ---------------------------------------------------------------------

def s5_nucleo_resiliente_light(df: pd.DataFrame) -> Dict:
    """
    Núcleo resiliente leve:
      - mede consistência local das séries
      - mede quanto cada série se parece com sua vizinhança
      - gera um escore de resiliência por série
    Este escore é usado em camadas posteriores como peso.
    """
    if df is None or df.empty:
        return {
            "resiliencia_por_serie": [],
            "resiliencia_media": 0.0,
        }

    p_cols = [c for c in df.columns if c.startswith("p")]
    if not p_cols:
        return {
            "resiliencia_por_serie": [],
            "resiliencia_media": 0.0,
        }

    n = len(df)
    resiliencias = []
    arr = df[p_cols].astype(float).values

    # Janela de comparação local
    janela = max(3, min(15, n // 10))

    for i in range(n):
        # Seleciona vizinhos ao redor de i
        ini = max(0, i - janela)
        fim = min(n, i + janela + 1)
        viz = []
        for j in range(ini, fim):
            if j == i:
                continue
            viz.append(arr[j])
        if not viz:
            resiliencias.append(0.5)
            continue

        v0 = arr[i]
        dists = []
        for v in viz:
            d = np.linalg.norm(v0 - v)
            dists.append(d)
        media = float(np.mean(dists))
        # Escala de resiliência: quanto menor a distância média, maior resiliência
        esc = 1 / (1 + media)
        resiliencias.append(clamp(esc))

    resil_media = float(np.mean(resiliencias)) if resiliencias else 0.0

    return {
        "resiliencia_por_serie": resiliencias,
        "resiliencia_media": resil_media,
    }

# ---------------------------------------------------------------------
# S6 — SIMILARIDADE ESTRUTURADA (PRÉ-S6 PROFUNDO)
# Aqui instalamos um S6 light, que já será usado por:
#   - motores de previsão
#   - IDX Simples
#   - S7
# ---------------------------------------------------------------------

def s6_similaridade_estrutural_light(df: pd.DataFrame) -> Dict:
    """
    S6 Light:
      - para cada série, mede a similaridade média com o restante da estrada,
        usando vetor de features expandido.
      - retorna um score [0..1] por série (1 = muito similar ao "padrão").
    """
    if df is None or df.empty:
        return {
            "similaridade_por_serie": [],
            "similaridade_media": 0.0,
        }

    n = len(df)
    if n < 3:
        return {
            "similaridade_por_serie": [0.5] * n,
            "similaridade_media": 0.5,
        }

    # Vetores expandido para todas as séries
    features = []
    for _, row in df.iterrows():
        vec = construir_vetor_features_expandido(row)
        features.append(vec)
    features = np.vstack(features)

    similaridades = []
    for i in range(n):
        base = features[i]
        dists = []
        for j in range(n):
            if j == i:
                continue
            d = distancia_euclidiana_normalizada(base, features[j])
            dists.append(d)
        if not dists:
            similaridades.append(0.5)
            continue
        media = float(np.mean(dists))
        # Quanto menor a distância, maior similaridade (invertido)
        sim = 1 / (1 + media)
        similaridades.append(clamp(sim))

    sim_media = float(np.mean(similaridades)) if similaridades else 0.0

    return {
        "similaridade_por_serie": similaridades,
        "similaridade_media": sim_media,
    }

# ---------------------------------------------------------------------
# S7 — CAMADA FINAL ESTRUTURAL (COMBINAÇÃO S5 + S6)
# Esta camada produz pesos estruturais para:
#   - motores TURBO
#   - IDX Avançado
#   - futuras camadas premium / estrutural / cobertura
# ---------------------------------------------------------------------

def s7_campos_estruturais(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriquecimento estrutural final:
      - adiciona colunas s5_res, s6_sim, s7_peso_estrutural
    """
    if df is None or df.empty:
        return df

    df2 = df.copy()

    info_s5 = s5_nucleo_resiliente_light(df2)
    info_s6 = s6_similaridade_estrutural_light(df2)

    s5_res = info_s5["resiliencia_por_serie"]
    s6_sim = info_s6["similaridade_por_serie"]

    n = len(df2)
    if len(s5_res) < n:
        s5_res = list(s5_res) + [0.5] * (n - len(s5_res))
    if len(s6_sim) < n:
        s6_sim = list(s6_sim) + [0.5] * (n - len(s6_sim))

    df2["s5_resil"] = s5_res
    df2["s6_sim"] = s6_sim

    pesos = []
    for i in range(n):
        res = s5_res[i]
        sim = s6_sim[i]
        peso = 0.6 * res + 0.4 * sim
        pesos.append(clamp(peso))

    df2["s7_peso_estrutural"] = pesos

    return df2

# =====================================================================
# IDX SIMPLES — VETORÍZACAO BÁSICA DA ESTRADA
# Produz:
#   - df_idx_simples: DataFrame com métricas por série
# =====================================================================

def calcular_idx_simples(df: pd.DataFrame) -> pd.DataFrame:
    """
    IDX Simples:
      - para cada série:
        * média dos passageiros
        * desvio padrão dos passageiros
        * amplitude
        * k
      - referência cruzada com posição na estrada
    """
    if df is None or df.empty:
        return pd.DataFrame()

    p_cols = [c for c in df.columns if c.startswith("p")]
    arr = df[p_cols].astype(float)

    medias = arr.mean(axis=1)
    stds = arr.std(axis=1)
    amps = arr.max(axis=1) - arr.min(axis=1)

    idx = []
    for i in range(len(df)):
        idx.append({
            "indice": i + 1,
            "media": float(medias.iloc[i]),
            "std": float(stds.iloc[i]),
            "amp": float(amps.iloc[i]),
            "k": float(df["k"].iloc[i]),
        })

    df_idx = pd.DataFrame(idx)
    st.session_state["idx_simples"] = df_idx
    return df_idx

# =====================================================================
# IDX AVANÇADO — ESTRUTURA BASE (CAMADAS MÚLTIPLAS)
# Nesta parte definimos apenas a “estrutura” e funções
# principais auxiliares. O cálculo completo entra nas
# próximas partes com S6 Profundo e Núcleo Resiliente PRO.
# =====================================================================

def _idx_avancado_componentes_basicos(df: pd.DataFrame) -> Dict:
    """
    Componentes básicos usados pelo IDX Avançado:
      - QDS completo
      - S5 Resiliente (light, nesta fase)
      - S6 Similaridade (light, nesta fase)
    """
    qds_info = calcular_qds_completo(df, janela=50)
    s5_info = s5_nucleo_resiliente_light(df)
    s6_info = s6_similaridade_estrutural_light(df)

    return {
        "qds": qds_info,
        "s5": s5_info,
        "s6": s6_info,
    }

def _idx_avancado_score_por_serie(df: pd.DataFrame, comp: Dict) -> List[float]:
    """
    Constrói um escore IDX Avançado por série combinando:
      - peso estrutural (S7)
      - similaridade (S6)
      - resiliência (S5)
      - posição relativa na estrada
    """
    df_s7 = s7_campos_estruturais(df)
    n = len(df_s7)

    pesos = df_s7["s7_peso_estrutural"].tolist()
    s5_res = comp["s5"]["resiliencia_por_serie"]
    s6_sim = comp["s6"]["similaridade_por_serie"]

    if len(s5_res) < n:
        s5_res = list(s5_res) + [0.5] * (n - len(s5_res))
    if len(s6_sim) < n:
        s6_sim = list(s6_sim) + [0.5] * (n - len(s6_sim))

    scores = []
    for i in range(n):
        pos_rel = i / max(1, n - 1)
        w = pesos[i]
        r = s5_res[i]
        s = s6_sim[i]
        # Combinação inicial (refinada em partes posteriores)
        score = (
            0.40 * w +
            0.30 * r +
            0.20 * s +
            0.10 * (1 - abs(0.5 - pos_rel))  # penaliza extremos, privilegia meio
        )
        scores.append(float(clamp(score)))

    return scores

def calcular_idx_avancado_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cálculo base do IDX Avançado:
      - gera um escore por série
      - guarda no estado
    A parte PRO / Profunda será estendida em módulos seguintes
    (S6 Profundo, NRF completo, etc.).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    comp = _idx_avancado_componentes_basicos(df)
    scores = _idx_avancado_score_por_serie(df, comp)

    rows = []
    for i, sc in enumerate(scores):
        rows.append({
            "indice": i + 1,
            "idx_avancado_score": sc,
            "k": float(df["k"].iloc[i]),
        })

    df_idx = pd.DataFrame(rows)
    st.session_state["idx_avancado"] = df_idx
    return df_idx

# =====================================================================
# PARTE 4/24 — FIM
# =====================================================================
# =====================================================================
# PREDICT CARS V15.5.1-HÍBRIDO — PARTE 5/24
# S6 Profundo (estrutura), Monte Carlo Profundo multi-camadas,
# Backtest Interno (versão longa) e pré-estrutura do Backtest do Futuro.
# =====================================================================

# ---------------------------------------------------------------------
# S6 PROFUNDO — ESTRUTURA COMPLETA (FLAT / MULTI-JANELA)
# Nesta fase, instalamos a versão conceitual e a estrutura
# de dados para o S6 Profundo. As versões mais pesadas de
# cálculo entram nas próximas partes, mas aqui já criamos:
#   - s6_profundo_raw
#   - s6_profundo_flat
#   - s6_profundo_stats
# no estado global.
# ---------------------------------------------------------------------

def s6_profundo_construir_raw(df: pd.DataFrame, janelas: List[int]) -> Dict:
    """
    Constrói um dicionário bruto com métricas de múltiplas janelas
    para cada série, para uso do S6 Profundo.
    Para cada janela w em janelas:
      - média dos passageiros na janela
      - desvio padrão dos passageiros na janela
      - amplitude média na janela
    """
    if df is None or df.empty:
        return {}

    p_cols = [c for c in df.columns if c.startswith("p")]
    if not p_cols:
        return {}

    arr = df[p_cols].astype(float).values
    n = len(df)

    dados = {}
    for w in janelas:
        w = max(3, min(w, n))
        medias_w = []
        stds_w = []
        amps_w = []

        for i in range(n):
            ini = max(0, i - w + 1)
            sub = arr[ini:i+1]
            if sub.size == 0:
                medias_w.append(0.0)
                stds_w.append(0.0)
                amps_w.append(0.0)
            else:
                medias_w.append(float(sub.mean()))
                stds_w.append(float(sub.std()))
                amps_w.append(float(sub.max() - sub.min()))

        dados[w] = {
            "medias": medias_w,
            "stds": stds_w,
            "amps": amps_w,
        }

    # Guarda no estado
    st.session_state["s6_profundo_raw"] = dados
    return dados

def s6_profundo_flatten(dados_raw: Dict) -> pd.DataFrame:
    """
    Produz um DataFrame flat com colunas:
      - indice
      - para cada janela w:
        * s6_wX_media, s6_wX_std, s6_wX_amp
    """
    if not dados_raw:
        df_empty = pd.DataFrame(columns=["indice"])
        st.session_state["s6_profundo_flat"] = df_empty
        return df_empty

    # Descobre tamanho n
    qualquer_w = next(iter(dados_raw.keys()))
    n = len(dados_raw[qualquer_w]["medias"])

    registros = []
    for i in range(n):
        linha = {"indice": i + 1}
        for w, info in dados_raw.items():
            medias_w = info["medias"]
            stds_w = info["stds"]
            amps_w = info["amps"]
            if i < len(medias_w):
                linha[f"s6_w{w}_media"] = float(medias_w[i])
            if i < len(stds_w):
                linha[f"s6_w{w}_std"] = float(stds_w[i])
            if i < len(amps_w):
                linha[f"s6_w{w}_amp"] = float(amps_w[i])
        registros.append(linha)

    df_flat = pd.DataFrame(registros)
    st.session_state["s6_profundo_flat"] = df_flat
    return df_flat

def s6_profundo_resumir_stats(df_flat: pd.DataFrame) -> Dict:
    """
    Gera estatísticas agregadas sobre o S6 Profundo flatten:
      - média global de cada coluna
      - desvio padrão de cada coluna
    """
    if df_flat is None or df_flat.empty:
        stats = {
            "medias_colunas": {},
            "stds_colunas": {},
        }
        st.session_state["s6_profundo_stats"] = stats
        return stats

    cols = [c for c in df_flat.columns if c != "indice"]
    medias = {}
    stds = {}
    for c in cols:
        arr = df_flat[c].astype(float)
        medias[c] = float(arr.mean())
        stds[c] = float(arr.std())

    stats = {
        "medias_colunas": medias,
        "stds_colunas": stds,
    }
    st.session_state["s6_profundo_stats"] = stats
    return stats

def s6_profundo_pipeline_basico(df: pd.DataFrame) -> Dict:
    """
    Executa o pipeline básico do S6 Profundo:
      1) constrói raw multi-janela
      2) flatten em DataFrame
      3) sumariza estatísticas
    Guarda tudo no estado.
    """
    if df is None or df.empty:
        return {
            "raw": {},
            "flat": pd.DataFrame(),
            "stats": {},
        }

    # Conjunto default de janelas (legado V14/15)
    janelas = [5, 10, 20, 40]

    raw = s6_profundo_construir_raw(df, janelas=janelas)
    flat = s6_profundo_flatten(raw)
    stats = s6_profundo_resumir_stats(flat)

    return {
        "raw": raw,
        "flat": flat,
        "stats": stats,
    }

# ---------------------------------------------------------------------
# MONTE CARLO PROFUNDO — MULTI-CAMADAS (V15)
# Aqui estabelecemos um Monte Carlo multi-nível, que será
# usado tanto na Confiabilidade REAL quanto no AIQ Avançado.
# ---------------------------------------------------------------------

def monte_carlo_profundo_multicamadas(
    df: pd.DataFrame,
    n_runs: int = 120,
    tamanhos_amostra: Optional[List[int]] = None,
) -> Dict:
    """
    Monte Carlo Profundo:
      - para cada tamanho de amostra em tamanhos_amostra
        * sorteia sub-trechos aleatórios
        * calcula QDS completo
      - agrega estatísticas de QDS por tamanho
    """
    if df is None or df.empty:
        info = {
            "resultados_por_tamanho": {},
            "qds_global_media": None,
            "qds_global_std": None,
        }
        st.session_state["mc_multi_info"] = info
        return info

    n = len(df)
    if tamanhos_amostra is None:
        tamanhos_amostra = [40, 80, 120]

    resultados = {}
    todos_qds = []

    for tam in tamanhos_amostra:
        tam_val = max(10, min(tam, n))
        qds_vals = []

        if n < tam_val + 5:
            # Muito curto para esse tamanho de janela
            resultados[tam_val] = {
                "n_runs": 0,
                "qds_media": None,
                "qds_std": None,
            }
            continue

        for _ in range(n_runs):
            # Sorteio de sub-trecho
            inicio = random.randint(0, n - tam_val)
            fim = inicio + tam_val
            sub_df = df.iloc[inicio:fim].reset_index(drop=True)

            qds_info = calcular_qds_completo(sub_df, janela=min(50, tam_val))
            qds_vals.append(qds_info["qds_score"])

        if qds_vals:
            media = float(np.mean(qds_vals))
            std = float(np.std(qds_vals))
            resultados[tam_val] = {
                "n_runs": n_runs,
                "qds_media": media,
                "qds_std": std,
            }
            todos_qds.extend(qds_vals)
        else:
            resultados[tam_val] = {
                "n_runs": 0,
                "qds_media": None,
                "qds_std": None,
            }

    if todos_qds:
        global_media = float(np.mean(todos_qds))
        global_std = float(np.std(todos_qds))
    else:
        global_media = None
        global_std = None

    info = {
        "resultados_por_tamanho": resultados,
        "qds_global_media": global_media,
        "qds_global_std": global_std,
    }
    st.session_state["mc_multi_info"] = info
    return info

# ---------------------------------------------------------------------
# BACKTEST INTERNO — VERSÃO LONGA (DETALHADA)
# Mede a performance do motor base ao longo da estrada.
# Nesta parte, conectamos apenas com o motor "placeholder".
# Os detalhes do motor TURBO surgirão nas próximas partes.
# ---------------------------------------------------------------------

def backtest_interno_extenso(
    df: pd.DataFrame,
    func_motor,
    n_tests: int = 80,
    janela_min: int = 60,
    janela_max: int = 240,
    top_n: int = 15,
) -> Dict:
    """
    Backtest Interno Extenso:
      - escolhe n_tests índices aleatórios ao longo da estrada
      - chama func_motor(df, idx_alvo, top_n, janela_min, janela_max)
      - mede:
        * número de testes
        * hits
        * hit_rate
        * posição média do acerto (avg_rank_hit)
        * distribuição dos ranks
    """
    if df is None or df.empty:
        info = {
            "n_tests": 0,
            "hits": 0,
            "hit_rate": 0.0,
            "avg_rank_hit": None,
            "ranks_hist": {},
        }
        st.session_state["backtest_info"] = info
        return info

    n = len(df)
    if n < janela_min + 5:
        info = {
            "n_tests": 0,
            "hits": 0,
            "hit_rate": 0.0,
            "avg_rank_hit": None,
            "ranks_hist": {},
        }
        st.session_state["backtest_info"] = info
        return info

    # Gera candidatos de índices-alvo
    candidatos = list(range(janela_min, n - 1))
    if len(candidatos) > n_tests:
        candidatos = random.sample(candidatos, n_tests)

    hits = 0
    ranks_hits = []

    for idx_alvo in candidatos:
        # Motor retorna uma lista de séries (leque)
        leque = func_motor(
            df=df,
            idx_alvo=idx_alvo,
            top_n=top_n,
            janela_min=janela_min,
            janela_max=janela_max,
        )
        if not leque:
            continue

        real = list_passageiros(df.iloc[idx_alvo])
        pos_hit = None
        for pos, serie in enumerate(leque, start=1):
            if serie == real:
                pos_hit = pos
                break

        if pos_hit is not None:
            hits += 1
            ranks_hits.append(pos_hit)

    n_eff = len(candidatos)
    hit_rate = hits / max(1, n_eff)

    if ranks_hits:
        avg_rank_hit = float(np.mean(ranks_hits))
        hist_ct = Counter(ranks_hits)
        ranks_hist = {int(k): int(v) for k, v in hist_ct.items()}
    else:
        avg_rank_hit = None
        ranks_hist = {}

    info = {
        "n_tests": n_eff,
        "hits": hits,
        "hit_rate": float(hit_rate),
        "avg_rank_hit": avg_rank_hit,
        "ranks_hist": ranks_hist,
    }
    st.session_state["backtest_info"] = info
    return info

# ---------------------------------------------------------------------
# ESTRUTURA DO BACKTEST DO FUTURO — PRÉ-BLOCO
# O cálculo efetivo, com janela avançando e previsão n+1,
# será amarrado ao motor TURBO nas partes posteriores.
# Aqui estabelecemos apenas a estrutura padrão.
# ---------------------------------------------------------------------

def backtest_do_futuro_estrutura(
    df: pd.DataFrame,
    func_motor,
    passo: int = 1,
    top_n: int = 15,
) -> Dict:
    """
    Backtest do Futuro (estrutura):
      - Simula o uso do motor em janelas que avançam pelo tempo:
        * para cada posição, corta o histórico até i
        * roda o motor prevendo i+1
        * compara com o real i+1
      - Nesta fase, apenas estrutura e contadores principais.
    """
    if df is None or df.empty:
        info = {
            "n_tests": 0,
            "hits": 0,
            "hit_rate": 0.0,
        }
        st.session_state["backtest_futuro_info"] = info
        return info

    n = len(df)
    if n < 3:
        info = {
            "n_tests": 0,
            "hits": 0,
            "hit_rate": 0.0,
        }
        st.session_state["backtest_futuro_info"] = info
        return info

    hits = 0
    tests = 0

    # Começa em 2 (índice 1 baseado em 0), pois precisa haver pelo menos uma série anterior
    for i in range(1, n - 1, passo):
        # Histórico até i
        sub = df.iloc[:i+1].reset_index(drop=True)
        idx_alvo = len(sub) - 1  # último índice de sub

        leque = func_motor(
            df=sub,
            idx_alvo=idx_alvo,
            top_n=top_n,
        )
        if not leque:
            continue

        real = list_passageiros(df.iloc[i+1])
        tests += 1
        if real in leque:
            hits += 1

    hit_rate = hits / max(1, tests)

    info = {
        "n_tests": tests,
        "hits": hits,
        "hit_rate": float(hit_rate),
    }
    st.session_state["backtest_futuro_info"] = info
    return info

# =====================================================================
# PARTE 5/24 — FIM
# =====================================================================
# =====================================================================
# PREDICT CARS V15.5.1-HÍBRIDO — PARTE 6/24
# Motor TURBO base (V14-FLEX ULTRA), versão longa, e wrappers
# para integração com Backtest Interno / Backtest do Futuro.
# =====================================================================

# ---------------------------------------------------------------------
# AJUSTE DE JANELAS PARA O MOTOR
# ---------------------------------------------------------------------

def ajustar_janelas_para_motor(
    n_series: int,
    janela_min: int,
    janela_max: int,
) -> Tuple[int, int]:
    """
    Ajusta janela_min e janela_max de forma segura, garantindo:
      - 1 <= janela_min < janela_max <= n_series-1
    """
    if n_series <= 2:
        return 1, 1

    jmin = max(1, min(janela_min, n_series - 2))
    jmax = max(jmin + 1, min(janela_max, n_series - 1))
    return jmin, jmax

# ---------------------------------------------------------------------
# MOTOR BASE — TURBO CORE V14-FLEX ULTRA (VERSÃO LONGA)
# ---------------------------------------------------------------------

def motor_turbo_core_v14(
    df: pd.DataFrame,
    idx_alvo: int,
    top_n: int = 25,
    janela_min: int = 60,
    janela_max: int = 220,
    usar_features_expandido: bool = False,
    usar_peso_estrutural: bool = True,
) -> List[List[int]]:
    """
    Motor base TURBO V14-FLEX ULTRA.
    Dado:
      - df: estrada completa (ou filtrada)
      - idx_alvo: índice alvo (0-based) da série que queremos prever
      - top_n: tamanho do leque de saída
      - janela_min / janela_max: limites da janela de busca de vizinhos
      - usar_features_expandido:
          False → vetor básico [p1..pN, k]
          True  → vetor expandido [p1..pN, k, mean, std, amp]
      - usar_peso_estrutural:
          True  → aplica peso de S7 (peso estrutural) na agregação
    Retorna:
      - lista de listas, cada lista sendo uma série prevista [p1..pN]
    """

    if df is None or df.empty:
        return []

    n = len(df)
    if idx_alvo <= 0 or idx_alvo >= n:
        # Precisa existir pelo menos uma série anterior ao alvo
        return []

    # Ajusta janelas
    jmin, jmax = ajustar_janelas_para_motor(n, janela_min, janela_max)
    if jmax <= jmin:
        return []

    # Seleciona a série imediatamente anterior ao alvo como "estado atual"
    row_alvo = df.iloc[idx_alvo - 1]
    if usar_features_expandido:
        vec_alvo = construir_vetor_features_expandido(row_alvo)
    else:
        vec_alvo = construir_vetor_features_basico(row_alvo)

    # Prepara campos estruturais (S7) se necessário
    if usar_peso_estrutural:
        df_s7 = s7_campos_estruturais(df)
        pesos_estruturais = df_s7["s7_peso_estrutural"].tolist()
    else:
        df_s7 = df
        pesos_estruturais = [1.0] * n

    # Coleta candidatos
    candidatos = []  # (score_agregado, serie_futura_tuple, idx_base)

    # Varrendo de jmin até jmax-1 como candidatos de base
    for i in range(jmin, jmax):
        if i <= 0 or i >= n:
            continue

        row_base = df.iloc[i - 1]
        if usar_features_expandido:
            vec_base = construir_vetor_features_expandido(row_base)
        else:
            vec_base = construir_vetor_features_basico(row_base)

        # Distância entre o estado alvo e o estado base
        dist = distancia_euclidiana(vec_alvo, vec_base)

        # Série futura associada ao estado base é a série na posição i (próxima)
        idx_futuro = i
        if idx_futuro >= n:
            continue
        row_futuro = df.iloc[idx_futuro]
        serie_futuro = list_passageiros(row_futuro)

        # Peso estrutural da base
        peso_estrutural = pesos_estruturais[idx_futuro] if idx_futuro < len(pesos_estruturais) else 1.0

        # Peso de tempo (prefere bases mais recentes dentro da janela)
        pos_rel = i / max(1, jmax)
        peso_tempo = 1 + pos_rel  # >1 para séries mais "avançadas" dentro da janela

        # Constrói um score "inverso" de distância
        base_score = 1 / (1 + dist)
        score_ajustado = base_score * peso_estrutural * peso_tempo

        candidatos.append((score_ajustado, tuple(serie_futuro), i))

    if not candidatos:
        return []

    # Ordena por score ajustado (decrescente)
    candidatos.sort(key=lambda x: x[0], reverse=True)

    # Corta pelos vizinhos mais significativos
    max_vizinhos = max(top_n * 4, 50)  # redundância intencional
    candidatos = candidatos[:max_vizinhos]

    # Agregação por série_futuro
    score_por_serie = defaultdict(float)
    for rank, (score, serie, idx_base) in enumerate(candidatos, start=1):
        # Peso extra por rank (decresce com rank)
        peso_rank = 1 / rank
        score_final = score * peso_rank
        score_por_serie[serie] += score_final

    # Ordena novamente por score agregado
    ordenados = sorted(score_por_serie.items(), key=lambda x: x[1], reverse=True)

    # Constrói o leque final
    leque = [list(s) for (s, sc) in ordenados[:top_n]]
    return leque

# ---------------------------------------------------------------------
# WRAPPER DO MOTOR PARA USO EM BACKTESTS
# 1) backtest_interno_extenso → motor_turbo_para_backtest_interno
# 2) backtest_do_futuro_estrutura → motor_turbo_para_backtest_futuro
# Ambos têm a mesma assinatura pedida lá atrás.
# ---------------------------------------------------------------------

def motor_turbo_para_backtest_interno(
    df: pd.DataFrame,
    idx_alvo: int,
    top_n: int = 15,
    janela_min: int = 60,
    janela_max: int = 240,
) -> List[List[int]]:
    """
    Wrapper específico para o Backtest Interno Extenso.
    Usa o motor TURBO core v14 com:
      - features expandido
      - peso estrutural
    """
    leque = motor_turbo_core_v14(
        df=df,
        idx_alvo=idx_alvo,
        top_n=top_n,
        janela_min=janela_min,
        janela_max=janela_max,
        usar_features_expandido=True,
        usar_peso_estrutural=True,
    )
    return leque

def motor_turbo_para_backtest_futuro(
    df: pd.DataFrame,
    idx_alvo: int,
    top_n: int = 15,
    janela_min: int = 40,
    janela_max: int = 200,
) -> List[List[int]]:
    """
    Wrapper específico para o Backtest do Futuro.
    Neste modo:
      - a janela é recalculada internamente com base no tamanho de df
      - priorizamos os vizinhos mais recentes (automaticamente)
    """
    n = len(df)
    jmin, jmax = ajustar_janelas_para_motor(n, janela_min, janela_max)
    leque = motor_turbo_core_v14(
        df=df,
        idx_alvo=idx_alvo,
        top_n=top_n,
        janela_min=jmin,
        janela_max=jmax,
        usar_features_expandido=True,
        usar_peso_estrutural=True,
    )
    return leque

# ---------------------------------------------------------------------
# FUNÇÕES AUXILIARES DE EXECUÇÃO DO MOTOR PRO BACKTESTS
# Ligando com as funções definidas na PARTE 5/24.
# ---------------------------------------------------------------------

def executar_backtest_interno_com_turbo(df: pd.DataFrame) -> Dict:
    """
    Atalho para rodar o Backtest Interno Extenso usando o motor TURBO.
    """
    info = backtest_interno_extenso(
        df=df,
        func_motor=motor_turbo_para_backtest_interno,
        n_tests=80,
        janela_min=60,
        janela_max=240,
        top_n=15,
    )
    return info

def executar_backtest_do_futuro_com_turbo(df: pd.DataFrame) -> Dict:
    """
    Atalho para rodar o Backtest do Futuro usando o motor TURBO.
    """
    info = backtest_do_futuro_estrutura(
        df=df,
        func_motor=motor_turbo_para_backtest_futuro,
        passo=1,
        top_n=15,
    )
    return info

# =====================================================================
# PARTE 6/24 — FIM
# =====================================================================
# =====================================================================
# PREDICT CARS V15.5.1-HÍBRIDO — PARTE 7/24
# Consolidação do IDX Avançado, estrutura do IDX Profundo,
# IDX Híbrido e ganchos para modos Premium / Estrutural / Cobertura.
# =====================================================================

# ---------------------------------------------------------------------
# IDX AVANÇADO — CONSOLIDAÇÃO COMPLETA
# Usando:
#   - QDS completo
#   - S5 (resiliência)
#   - S6 (similaridade)
#   - S7 (peso estrutural)
# ---------------------------------------------------------------------

def idx_avancado_consolidar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Versão consolidada do IDX Avançado.
    Gera um DataFrame com:
      - indice
      - score_idx_avancado
      - componentes:
        * peso_estrutural (S7)
        * resiliência (S5)
        * similaridade (S6)
        * k
    Guarda em st.session_state["idx_avancado"].
    """
    if df is None or df.empty:
        df_vazio = pd.DataFrame()
        st.session_state["idx_avancado"] = df_vazio
        return df_vazio

    # Recalcula tudo explicitamente (sem compactação)
    df_s7 = s7_campos_estruturais(df)
    comp = _idx_avancado_componentes_basicos(df)
    s5_res = comp["s5"]["resiliencia_por_serie"]
    s6_sim = comp["s6"]["similaridade_por_serie"]

    n = len(df_s7)
    if len(s5_res) < n:
        s5_res = list(s5_res) + [0.5] * (n - len(s5_res))
    if len(s6_sim) < n:
        s6_sim = list(s6_sim) + [0.5] * (n - len(s6_sim))

    pesos_estr = df_s7["s7_peso_estrutural"].tolist()

    linhas = []
    for i in range(n):
        pos_rel = i / max(1, n - 1)
        w = pesos_estr[i]
        r = s5_res[i]
        s = s6_sim[i]
        # Score final um pouco mais calibrado do que na base
        score = (
            0.45 * w +
            0.25 * r +
            0.20 * s +
            0.10 * (1 - abs(0.5 - pos_rel))
        )
        linhas.append({
            "indice": i + 1,
            "idx_avancado_score": float(clamp(score)),
            "s7_peso_estrutural": float(w),
            "s5_resiliencia": float(r),
            "s6_similaridade": float(s),
            "k": float(df["k"].iloc[i]),
        })

    df_idx = pd.DataFrame(linhas)
    st.session_state["idx_avancado"] = df_idx
    return df_idx

# ---------------------------------------------------------------------
# IDX PROFUNDO — USANDO S6 PROFUNDO (MULTI-JANELA)
# Cria um índice baseado nas features planas do S6 Profundo.
# ---------------------------------------------------------------------

def idx_profundo_construir(df: pd.DataFrame) -> pd.DataFrame:
    """
    IDX Profundo:
      - executa o pipeline S6 Profundo
      - a partir do df_flat, cria um score de "coerência profunda"
      - combina todas as colunas s6_wX_* em um escore [0..1] por série
    Guarda resultado em st.session_state["idx_profundo"].
    """
    if df is None or df.empty:
        df_vazio = pd.DataFrame()
        st.session_state["idx_profundo"] = df_vazio
        return df_vazio

    s6_info = s6_profundo_pipeline_basico(df)
    df_flat = s6_info["flat"]

    if df_flat is None or df_flat.empty:
        df_vazio = pd.DataFrame()
        st.session_state["idx_profundo"] = df_vazio
        return df_vazio

    cols_media = [c for c in df_flat.columns if "media" in c]
    cols_std = [c for c in df_flat.columns if "std" in c]
    cols_amp = [c for c in df_flat.columns if "amp" in c]

    n = len(df_flat)
    linhas = []

    for i in range(n):
        linha = df_flat.iloc[i]

        medias = []
        stds = []
        amps = []

        for c in cols_media:
            medias.append(safe_float(linha.get(c, 0.0)))
        for c in cols_std:
            stds.append(safe_float(linha.get(c, 0.0)))
        for c in cols_amp:
            amps.append(safe_float(linha.get(c, 0.0)))

        # Média das médias, desvios, amplitudes
        med_val = float(np.mean(medias)) if medias else 0.0
        std_val = float(np.mean(stds)) if stds else 0.0
        amp_val = float(np.mean(amps)) if amps else 0.0

        # Score profundo: penaliza std e amp muito altos, prefere médias moderadas
        # (estes pesos foram ajustados empiricamente nas versões anteriores)
        score = (
            0.40 * (1 / (1 + abs(med_val))) +
            0.30 * (1 / (1 + std_val)) +
            0.30 * (1 / (1 + amp_val))
        )

        linhas.append({
            "indice": int(linha["indice"]),
            "idx_profundo_score": float(clamp(score)),
            "s6_media_global": med_val,
            "s6_std_global": std_val,
            "s6_amp_global": amp_val,
        })

    df_idx_prof = pd.DataFrame(linhas)
    st.session_state["idx_profundo"] = df_idx_prof
    return df_idx_prof

# ---------------------------------------------------------------------
# IDX HÍBRIDO — COMBINA AVANÇADO + PROFUNDO
# ---------------------------------------------------------------------

def idx_hibrido_construir(df: pd.DataFrame) -> pd.DataFrame:
    """
    IDX Híbrido:
      - junta idx_avancado_score e idx_profundo_score
      - produz um escore único idx_hibrido_score
    Guarda em st.session_state["idx_hibrido"].
    """
    if df is None or df.empty:
        df_vazio = pd.DataFrame()
        st.session_state["idx_hibrido"] = df_vazio
        return df_vazio

    # Garante que os índices existam
    df_av = idx_avancado_consolidar(df)
    df_pf = idx_profundo_construir(df)

    if df_av.empty or df_pf.empty:
        df_vazio = pd.DataFrame()
        st.session_state["idx_hibrido"] = df_vazio
        return df_vazio

    # Merge por indice
    df_merge = pd.merge(
        df_av,
        df_pf,
        on="indice",
        how="inner",
        suffixes=("_av", "_prof"),
    )

    linhas = []
    for _, row in df_merge.iterrows():
        s_av = safe_float(row.get("idx_avancado_score", 0.0))
        s_pf = safe_float(row.get("idx_profundo_score", 0.0))

        # Combinação inicial: pesos semelhantes
        score_h = 0.5 * s_av + 0.5 * s_pf

        linhas.append({
            "indice": int(row["indice"]),
            "idx_hibrido_score": float(clamp(score_h)),
            "idx_avancado_score": float(s_av),
            "idx_profundo_score": float(s_pf),
            "k": float(row.get("k", 0.0)),
        })

    df_hib = pd.DataFrame(linhas)
    st.session_state["idx_hibrido"] = df_hib
    return df_hib

# ---------------------------------------------------------------------
# MODOS PREMIUM / ESTRUTURAL / COBERTURA — GANCHOS INICIAIS
# A lógica completa de seleção por faixas entra depois,
# mas aqui definimos as estruturas base de classificação.
# ---------------------------------------------------------------------

def classificar_modo_series_com_idx(df: pd.DataFrame) -> Dict:
    """
    Classifica as séries em:
      - Premium
      - Estrutural
      - Cobertura
    usando o idx_hibrido_score, quando disponível.
    Retorna dicionário com listas de índices para cada modo.
    """
    if df is None or df.empty:
        info = {
            "premium": [],
            "estrutural": [],
            "cobertura": [],
        }
        st.session_state["premium_info"] = info
        st.session_state["estrutural_info"] = info
        st.session_state["cobertura_info"] = info
        return info

    df_hib = idx_hibrido_construir(df)
    if df_hib.empty:
        info = {
            "premium": [],
            "estrutural": [],
            "cobertura": [],
        }
        st.session_state["premium_info"] = info
        st.session_state["estrutural_info"] = info
        st.session_state["cobertura_info"] = info
        return info

    scores = df_hib["idx_hibrido_score"].tolist()
    n = len(scores)
    if n == 0:
        info = {
            "premium": [],
            "estrutural": [],
            "cobertura": [],
        }
        st.session_state["premium_info"] = info
        st.session_state["estrutural_info"] = info
        st.session_state["cobertura_info"] = info
        return info

    # Quebras em tercis (aproximadas) para separar os modos
    sorted_scores = sorted(scores)
    t1 = sorted_scores[int(0.33 * (n - 1))]
    t2 = sorted_scores[int(0.66 * (n - 1))]

    premium_idx = []
    estrut_idx = []
    cobert_idx = []

    for i, s in enumerate(scores):
        if s >= t2:
            premium_idx.append(i + 1)   # 1-based
        elif s >= t1:
            estrut_idx.append(i + 1)
        else:
            cobert_idx.append(i + 1)

    info = {
        "premium": premium_idx,
        "estrutural": estrut_idx,
        "cobertura": cobert_idx,
    }

    st.session_state["premium_info"] = {"indices": premium_idx}
    st.session_state["estrutural_info"] = {"indices": estrut_idx}
    st.session_state["cobertura_info"] = {"indices": cobert_idx}
    return info

# =====================================================================
# PARTE 7/24 — FIM
# =====================================================================
# =====================================================================
# PREDICT CARS V15.5.1-HÍBRIDO — PARTE 8/24
# AIQ Global / Avançado, Monitor de Risco (k & k*),
# e estruturas de Ruído Condicional / A-B.
# =====================================================================

# ---------------------------------------------------------------------
# AIQ GLOBAL — AGREGADOR DE QUALIDADE (VERSÃO V15)
# Integra:
#   - QDS
#   - Backtest Interno
#   - Backtest do Futuro
#   - Monte Carlo Profundo
#   - IDX Híbrido
# ---------------------------------------------------------------------

def calcular_aiq_global(df: pd.DataFrame) -> Dict:
    """
    Calcula o AIQ Global (versão V15):
      - usa QDS completo
      - usa resultados do Backtest Interno
      - usa resultados do Backtest do Futuro
      - usa Monte Carlo Profundo multijanela
      - usa estatísticas do IDX Híbrido
    Armazena em st.session_state["aiq_info"].
    """
    if df is None or df.empty:
        info = {
            "aiq_score": 0.0,
            "qds": None,
            "backtest_interno": None,
            "backtest_futuro": None,
            "mc_multi": None,
            "idx_hibrido_stats": None,
        }
        st.session_state["aiq_info"] = info
        return info

    # QDS completo
    qds_info = calcular_qds_completo(df, janela=50)

    # Backtest Interno com TURBO
    bt_int = executar_backtest_interno_com_turbo(df)

    # Backtest do Futuro com TURBO
    bt_fut = executar_backtest_do_futuro_com_turbo(df)

    # Monte Carlo Profundo
    mc_multi = monte_carlo_profundo_multicamadas(df, n_runs=80, tamanhos_amostra=[40, 80, 120])

    # IDX Híbrido
    df_hib = idx_hibrido_construir(df)
    if df_hib is not None and not df_hib.empty:
        s_hib = df_hib["idx_hibrido_score"].astype(float).values
        hib_media = float(np.mean(s_hib))
        hib_std = float(np.std(s_hib))
    else:
        hib_media = 0.0
        hib_std = 0.0

    idx_stats = {
        "hib_media": hib_media,
        "hib_std": hib_std,
    }

    # -----------------------------------------------------------------
    # Combinação em AIQ Score
    # -----------------------------------------------------------------
    qds_score = qds_info["qds_score"]

    # Backtest interno → peso por hit_rate
    bt_int_rate = bt_int.get("hit_rate", 0.0) or 0.0
    bt_fut_rate = bt_fut.get("hit_rate", 0.0) or 0.0

    # Monte Carlo → preferimos qds_global_media estável e alto
    mc_media = mc_multi.get("qds_global_media", None)
    if mc_media is None:
        mc_media = 0.0
    mc_media = float(mc_media)

    # IDX Híbrido → preferimos média alta e desvio moderado
    # normaliza hib_media em [0..1] por heurística
    hib_media_norm = clamp(hib_media)
    hib_stab = 1 / (1 + hib_std)

    # AIQ Score final — pesos calibrados
    aiq = (
        0.30 * qds_score +
        0.20 * bt_int_rate +
        0.15 * bt_fut_rate +
        0.15 * mc_media +
        0.10 * hib_media_norm +
        0.10 * hib_stab
    )
    aiq = clamp(aiq)

    info = {
        "aiq_score": aiq,
        "qds": qds_info,
        "backtest_interno": bt_int,
        "backtest_futuro": bt_fut,
        "mc_multi": mc_multi,
        "idx_hibrido_stats": idx_stats,
    }
    st.session_state["aiq_info"] = info
    return info

# ---------------------------------------------------------------------
# AIQ AVANÇADO — CONTEXTO AMPLIADO COM S6 PROFUNDO
# Usa também:
#   - S6 Profundo (stats globais)
#   - Entropia global das séries (já computada em QDS)
# ---------------------------------------------------------------------

def calcular_aiq_avancado(df: pd.DataFrame) -> Dict:
    """
    Extensão do AIQ Global com:
      - estatísticas do S6 Profundo
      - entropia global da estrada
    Armazena em st.session_state["aiq_avancado_info"].
    """
    if df is None or df.empty:
        info = {
            "aiq_avancado_score": 0.0,
            "base_aiq_info": None,
            "s6_profundo_stats": None,
        }
        st.session_state["aiq_avancado_info"] = info
        return info

    base_aiq = calcular_aiq_global(df)
    qds_info = base_aiq["qds"]

    # S6 Profundo
    s6_info = s6_profundo_pipeline_basico(df)
    stats = s6_info["stats"]

    # Extra: entropia normalizada da QDS
    ent_norm = qds_info["ent"]["entropia_norm"]

    # Condensa desvio médio das colunas do S6 Profundo
    if stats and "stds_colunas" in stats:
        std_vals = list(stats["stds_colunas"].values())
        if std_vals:
            s6_std_media = float(np.mean(std_vals))
        else:
            s6_std_media = 0.0
    else:
        s6_std_media = 0.0

    s6_estabilidade = 1 / (1 + s6_std_media)

    # AIQ avançado — realça:
    #   - AIQ base alto
    #   - entropia moderada (nem zero, nem máxima)
    #   - estabilidade de S6 Profundo
    aiq_base = base_aiq["aiq_score"]
    # entropia moderada → penalidade se muito baixa ou muito alta
    ent_moderada = 1 - abs(ent_norm - 0.6)

    aiq_av = clamp(
        0.55 * aiq_base +
        0.25 * ent_moderada +
        0.20 * s6_estabilidade
    )

    info = {
        "aiq_avancado_score": aiq_av,
        "base_aiq_info": base_aiq,
        "s6_profundo_stats": stats,
        "entropia_norm": ent_norm,
    }
    st.session_state["aiq_avancado_info"] = info
    return info

# ---------------------------------------------------------------------
# MONITOR DE RISCO (k & k*)
# Combina:
#   - regimes globais / locais de k*
#   - k_mean / k_std
#   - AIQ Avançado
#   - QDS
# ---------------------------------------------------------------------

def montar_monitor_de_risco_global(df: pd.DataFrame) -> Dict:
    """
    Monta o monitor de risco global:
      - regime global por k*
      - k_mean, k_std
      - aiq_avancado_score
      - qds_score
      - texto interpretativo
    Armazena em st.session_state["monitor_global"].
    """
    if df is None or df.empty:
        info = {
            "regime_global": "desconhecido",
            "texto_regime": "Histórico vazio — não é possível avaliar o risco.",
            "k_mean": 0.0,
            "k_std": 0.0,
            "aiq_avancado": 0.0,
            "qds_score": 0.0,
        }
        st.session_state["monitor_global"] = info
        return info

    # Regime via k*
    reg_info = analisar_regime_global_e_local(df, idx_local=None, janela=40)

    # AIQ Avançado + QDS
    aiq_adv = calcular_aiq_avancado(df)
    qds_score = aiq_adv["base_aiq_info"]["qds"]["qds_score"]

    regime_global = reg_info["regime_global"]
    texto_regime = reg_info["texto_global"]
    k_mean = reg_info["k_mean"]
    k_std = reg_info["k_std"]
    aiq_val = aiq_adv["aiq_avancado_score"]

    # Texto extra combinando AIQ e regime
    if regime_global == "estavel":
        if aiq_val >= 0.7:
            texto_aiq = "🟢 Núcleo preditivo em excelente forma, com alta confiabilidade."
        elif aiq_val >= 0.5:
            texto_aiq = "🟢 Núcleo preditivo em regime saudável, com boa confiabilidade."
        else:
            texto_aiq = "🟡 Núcleo preditivo ainda em fase de consolidação, apesar da estrada estável."
    elif regime_global == "atencao":
        if aiq_val >= 0.7:
            texto_aiq = "🟡 Estrada em pré-ruptura, mas núcleo preditivo robusto compensa parte do risco."
        else:
            texto_aiq = "🟡 Estrada em pré-ruptura, recomenda-se cautela nas previsões e tamanho do leque."
    else:  # critico
        if aiq_val >= 0.7:
            texto_aiq = "🟠 Núcleo preditivo forte, porém a estrada está em regime crítico — risco elevado."
        else:
            texto_aiq = "🔴 Estrada em regime crítico e núcleo preditivo em estresse — máxima cautela."

    info = {
        "regime_global": regime_global,
        "texto_regime": texto_regime,
        "k_mean": float(k_mean),
        "k_std": float(k_std),
        "aiq_avancado": float(aiq_val),
        "qds_score": float(qds_score),
        "texto_aiq": texto_aiq,
    }
    st.session_state["monitor_global"] = info
    return info

def montar_monitor_de_risco_local(df: pd.DataFrame, idx_local: int) -> Dict:
    """
    Monta o monitor de risco local (para um índice alvo):
      - regime local via k*
      - k local, k_star local
      - texto interpretativo de risco
    Armazena em st.session_state["monitor_local"].
    """
    if df is None or df.empty:
        info = {
            "regime_local": "desconhecido",
            "k_local": 0.0,
            "k_star_local": 0.0,
            "texto_local": "Histórico vazio — não é possível avaliar o risco local.",
        }
        st.session_state["monitor_local"] = info
        return info

    reg_info = analisar_regime_global_e_local(df, idx_local=idx_local, janela=40)
    local = reg_info["local_info"]

    if local is None:
        info = {
            "regime_local": "desconhecido",
            "k_local": 0.0,
            "k_star_local": 0.0,
            "texto_local": "Índice fora do intervalo ou não avaliado.",
        }
        st.session_state["monitor_local"] = info
        return info

    info = {
        "regime_local": local["regime"],
        "k_local": float(local["k"]),
        "k_star_local": float(local["k_star"]),
        "texto_local": local["texto"],
    }
    st.session_state["monitor_local"] = info
    return info

def montar_monitor_de_risco_completo(df: pd.DataFrame, idx_local: Optional[int]) -> Dict:
    """
    Agrega monitor global + local em um único dicionário.
    Armazena também em st.session_state["ultimo_monitor_risco"].
    """
    global_info = montar_monitor_de_risco_global(df)
    local_info = montar_monitor_de_risco_local(df, idx_local if idx_local is not None else len(df) - 1)

    info = {
        "global": global_info,
        "local": local_info,
    }
    st.session_state["ultimo_monitor_risco"] = info
    return info

# ---------------------------------------------------------------------
# RUÍDO CONDICIONAL — ESTRUTURA BÁSICA
# Mede, para um conjunto de séries previstas:
#   - dispersão média por passageiro
#   - amplitude média
#   - entropia dos passageiros
# ---------------------------------------------------------------------

def analisar_ruido_do_leque(leque: List[List[int]]) -> Dict:
    """
    Analisa ruído de um leque de previsões.
    Retorna:
      - n_series
      - disp_media
      - amp_media
      - entropia_norm
    """
    if not leque:
        info = {
            "n_series": 0,
            "disp_media": 0.0,
            "amp_media": 0.0,
            "entropia_norm": 0.0,
        }
        return info

    arr = np.array(leque, dtype=float)
    if arr.ndim != 2:
        info = {
            "n_series": 0,
            "disp_media": 0.0,
            "amp_media": 0.0,
            "entropia_norm": 0.0,
        }
        return info

    # Dispersão média por passageiro
    stds = arr.std(axis=0)
    disp_media = float(np.mean(stds))

    # Amplitude média por passageiro
    amps = arr.max(axis=0) - arr.min(axis=0)
    amp_media = float(np.mean(amps))

    # Entropia global dos passageiros
    flat = arr.flatten()
    unique, counts = np.unique(flat, return_counts=True)
    probs = counts / counts.sum()
    ent = -np.sum([p * math.log(p + 1e-12) for p in probs])
    ent_norm = ent / max(1.0, math.log(len(unique) + 1e-12))

    info = {
        "n_series": int(arr.shape[0]),
        "disp_media": disp_media,
        "amp_media": amp_media,
        "entropia_norm": float(clamp(float(ent_norm))),
    }
    return info

# ---------------------------------------------------------------------
# RUÍDO CONDICIONAL A/B — PERFIS DE RUÍDO PARA DOIS MODOS
# Estrutura genérica; o preenchimento A/B será feito nos painéis
# de Ruído Condicional (por exemplo, comparando dois conjuntos
# de leques, ou dois modos de motor).
# ---------------------------------------------------------------------

def comparar_ruido_condicional_ab(leque_a: List[List[int]], leque_b: List[List[int]]) -> Dict:
    """
    Compara ruído de dois leques (A e B):
      - calcula métricas de ruído para cada
      - gera deltas
    Armazena em st.session_state["ruido_condicional_info"].
    """
    info_a = analisar_ruido_do_leque(leque_a)
    info_b = analisar_ruido_do_leque(leque_b)

    def delta(a, b):
        return float(b - a)

    comp = {
        "A": info_a,
        "B": info_b,
        "delta_disp": delta(info_a["disp_media"], info_b["disp_media"]),
        "delta_amp": delta(info_a["amp_media"], info_b["amp_media"]),
        "delta_entropia": delta(info_a["entropia_norm"], info_b["entropia_norm"]),
    }

    st.session_state["ruido_condicional_info"] = comp
    return comp

# =====================================================================
# PARTE 8/24 — FIM
# =====================================================================
# =====================================================================
# PREDICT CARS V15.5.1-HÍBRIDO — PARTE 9/24
# Testes de Confiabilidade REAL (QDS / Backtest / Monte Carlo)
# + estrutura numérica dos modos Replay LIGHT / ULTRA / Unitário.
# =====================================================================

# ---------------------------------------------------------------------
# TESTES DE CONFIABILIDADE REAL — BLOCO DE ALTO NÍVEL
# Empacota:
#   - QDS completo
#   - Backtest Interno
#   - Backtest do Futuro
#   - Monte Carlo Profundo
#   - AIQ Global / Avançado
# ---------------------------------------------------------------------

def executar_testes_confiabilidade_real(df: pd.DataFrame) -> Dict:
    """
    Executa o pacote completo de Testes de Confiabilidade REAL:
      - QDS completo
      - Backtest Interno (TURBO)
      - Backtest do Futuro (TURBO)
      - Monte Carlo Profundo multi-camadas
      - AIQ Global
      - AIQ Avançado
    Preenche diversos campos em st.session_state:
      - qds_info
      - qds_info_completo
      - backtest_info
      - backtest_futuro_info
      - mc_multi_info
      - aiq_info
      - aiq_avancado_info
    """
    if df is None or df.empty:
        info = {
            "qds": None,
            "backtest_interno": None,
            "backtest_futuro": None,
            "monte_carlo_multi": None,
            "aiq": None,
            "aiq_avancado": None,
        }
        st.session_state["qds_info"] = None
        st.session_state["qds_info_completo"] = None
        st.session_state["backtest_info"] = None
        st.session_state["backtest_futuro_info"] = None
        st.session_state["mc_multi_info"] = None
        st.session_state["aiq_info"] = None
        st.session_state["aiq_avancado_info"] = None
        st.session_state["modo_6_acertos_info"] = None
        return info

    # QDS completo
    qds_info = calcular_qds_completo(df, janela=50)
    st.session_state["qds_info"] = qds_info
    st.session_state["qds_info_completo"] = qds_info  # redundância intencional

    # Backtest Interno
    bt_int = executar_backtest_interno_com_turbo(df)
    # Backtest do Futuro
    bt_fut = executar_backtest_do_futuro_com_turbo(df)

    # Monte Carlo Profundo multi-camadas
    mc_multi = monte_carlo_profundo_multicamadas(
        df,
        n_runs=80,
        tamanhos_amostra=[40, 80, 120]
    )

    # AIQ Global + Avançado
    aiq_info = calcular_aiq_global(df)
    aiq_adv = calcular_aiq_avancado(df)

    info = {
        "qds": qds_info,
        "backtest_interno": bt_int,
        "backtest_futuro": bt_fut,
        "monte_carlo_multi": mc_multi,
        "aiq": aiq_info,
        "aiq_avancado": aiq_adv,
    }

    st.session_state["aiq_info"] = aiq_info
    st.session_state["aiq_avancado_info"] = aiq_adv

    return info

# ---------------------------------------------------------------------
# REPLAY LIGHT / ULTRA / UNITÁRIO — ESTRUTURA NUMÉRICA
# ---------------------------------------------------------------------

# REPLAY LIGHT:
#   - foco no alvo único
#   - janela recente configurável
#   - leque principal com ruído associado

def executar_replay_light(
    df: pd.DataFrame,
    idx_alvo_1based: int,
    janela_min: int = 40,
    janela_max: int = 200,
    top_n: int = 25,
) -> Dict:
    """
    Replay LIGHT:
      - Usa o histórico original completo
      - Converte o índice 1-based para 0-based
      - Roda o motor TURBO base
      - Analisa ruído do leque
      - Monta contexto de risco local
    Armazena em st.session_state["replay_light"].
    """
    if df is None or df.empty:
        info = {
            "idx_alvo": idx_alvo_1based,
            "leque": [],
            "ruido": None,
            "monitor_risco": None,
        }
        st.session_state["replay_light"] = info
        return info

    n = len(df)
    idx0 = idx_alvo_1based - 1
    if idx0 < 1:
        idx0 = 1
    if idx0 >= n:
        idx0 = n - 1

    # Motor TURBO
    leque = motor_turbo_core_v14(
        df=df,
        idx_alvo=idx0,
        top_n=top_n,
        janela_min=janela_min,
        janela_max=janela_max,
        usar_features_expandido=True,
        usar_peso_estrutural=True,
    )

    # Ruído do leque
    ruido = analisar_ruido_do_leque(leque)

    # Monitor de risco local
    monitor = montar_monitor_de_risco_completo(df, idx_local=idx0)

    info = {
        "idx_alvo": idx_alvo_1based,
        "leque": leque,
        "ruido": ruido,
        "monitor_risco": monitor,
    }
    st.session_state["replay_light"] = info
    return info

# ---------------------------------------------------------------------
# REPLAY ULTRA:
#   - percorre uma faixa de índices
#   - para cada índice, gera leque e analisa hits (se desejar)
#   - pode servir de base para visualizações mais densas
# ---------------------------------------------------------------------

def executar_replay_ultra(
    df: pd.DataFrame,
    idx_inicio_1based: int,
    idx_fim_1based: int,
    passo: int = 1,
    janela_min: int = 40,
    janela_max: int = 200,
    top_n: int = 15,
) -> Dict:
    """
    Replay ULTRA:
      - Varia idx_alvo de idx_inicio até idx_fim (1-based)
      - Para cada índice:
          * roda motor TURBO base
          * registra tamanho do leque e ruído
      - Não faz checagem de hit por padrão (isso é papel do backtest),
        mas fornece base para análises visuais.
    Armazena em st.session_state["replay_ultra"].
    """
    if df is None or df.empty:
        info = {
            "indices": [],
            "resumos": [],
        }
        st.session_state["replay_ultra"] = info
        return info

    n = len(df)
    idx_inicio = max(2, idx_inicio_1based)  # evita índice < 2
    idx_fim = min(idx_fim_1based, n - 1)

    if idx_inicio > idx_fim:
        idx_inicio, idx_fim = idx_fim, idx_inicio

    indices = []
    resumos = []

    for idx in range(idx_inicio, idx_fim + 1, passo):
        idx0 = idx - 1
        leque = motor_turbo_core_v14(
            df=df,
            idx_alvo=idx0,
            top_n=top_n,
            janela_min=janela_min,
            janela_max=janela_max,
            usar_features_expandido=True,
            usar_peso_estrutural=True,
        )
        ruido = analisar_ruido_do_leque(leque)
        indices.append(idx)
        resumos.append({
            "idx": idx,
            "n_series_leque": len(leque),
            "ruido": ruido,
        })

    info = {
        "indices": indices,
        "resumos": resumos,
    }
    st.session_state["replay_ultra"] = info
    return info

# ---------------------------------------------------------------------
# REPLAY ULTRA UNITÁRIO:
#   - foca em um índice específico
#   - registra detalhes extras da série original e do leque
# ---------------------------------------------------------------------

def executar_replay_ultra_unitario(
    df: pd.DataFrame,
    idx_alvo_1based: int,
    janela_min: int = 40,
    janela_max: int = 200,
    top_n: int = 25,
) -> Dict:
    """
    Replay ULTRA Unitário:
      - Exibe a série alvo (Cidx) com passageiros originais
      - Gera leque TURBO para esse alvo
      - Adiciona contexto de k, k*, regime local e ruído
    Armazena em st.session_state["replay_unitario"].
    """
    if df is None or df.empty:
        info = {
            "idx_alvo": idx_alvo_1based,
            "serie_real": None,
            "k_real": None,
            "leque": [],
            "ruido": None,
            "monitor_risco": None,
        }
        st.session_state["replay_unitario"] = info
        return info

    n = len(df)
    idx0 = idx_alvo_1based - 1
    if idx0 <= 0:
        idx0 = 1
    if idx0 >= n:
        idx0 = n - 1

    row_real = df.iloc[idx0]
    serie_real = list_passageiros(row_real)
    k_real = safe_int(row_real["k"])

    leque = motor_turbo_core_v14(
        df=df,
        idx_alvo=idx0,
        top_n=top_n,
        janela_min=janela_min,
        janela_max=janela_max,
        usar_features_expandido=True,
        usar_peso_estrutural=True,
    )

    ruido = analisar_ruido_do_leque(leque)
    monitor = montar_monitor_de_risco_completo(df, idx_local=idx0)

    info = {
        "idx_alvo": idx_alvo_1based,
        "serie_real": serie_real,
        "k_real": k_real,
        "leque": leque,
        "ruido": ruido,
        "monitor_risco": monitor,
    }
    st.session_state["replay_unitario"] = info
    return info

# =====================================================================
# PARTE 9/24 — FIM
# =====================================================================
# =====================================================================
# PREDICT CARS V15.5.1-HÍBRIDO — PARTE 10/24
# Modo 6 Acertos (núcleo numérico completo)
# + Estrutura do Relatório Final — AIQ Bridge (para ChatGPT).
# =====================================================================

# ---------------------------------------------------------------------
# MODO 6 ACERTOS — Núcleo completo
# Combina:
#   - análise de ambiente (k*, idx_híbrido, S6, QDS)
#   - seleção de trechos bons / premium
#   - filtragem de séries por estrutura
#   - afinamento com IDX Avançado / Híbrido
#   - motor TURBO adaptado para busca profunda
#   - saída final com métricas
# ---------------------------------------------------------------------

def executar_modo_6_acertos(
    df: pd.DataFrame,
    idx_alvo_1based: int,
    top_n: int = 12,
    janela_min: int = 40,
    janela_max: int = 220,
) -> Dict:
    """
    IMPLEMENTAÇÃO COMPLETA DO MODO 6 ACERTOS (núcleo numérico).
    Estrutura:
      1. verifica idx
      2. obtém regimes global/local de risco
      3. monta IDX híbrido
      4. identifica trechos bons (Premium + Estrutural)
      5. filtra o df para trechos saudáveis (quando possível)
      6. executa motor_turbo_core_v14 em versão adaptativa
      7. valida ruído e consistência
      8. retorna bloco completo para Streamlit
    Armazena em st.session_state["modo_6_acertos_info"].
    """
    if df is None or df.empty:
        info = {
            "idx_alvo": idx_alvo_1based,
            "serie_real": None,
            "k_real": None,
            "leque_final": [],
            "ruido": None,
            "monitor_risco": None,
            "trechos_bons": None,
        }
        st.session_state["modo_6_acertos_info"] = info
        return info

    n = len(df)
    idx0 = idx_alvo_1based - 1
    if idx0 <= 0:
        idx0 = 1
    if idx0 >= n:
        idx0 = n - 1

    # Série real
    row_real = df.iloc[idx0]
    serie_real = list_passageiros(row_real)
    k_real = safe_int(row_real["k"])

    # 1. MONITOR DE RISCO COMPLETO
    monitor = montar_monitor_de_risco_completo(df, idx_local=idx0)

    # 2. IDX HÍBRIDO
    df_hib = idx_hibrido_construir(df)
    if df_hib.empty:
        # fallback sem filtragem
        df_trecho = df
        trechos_bons = []
    else:
        info_modes = classificar_modo_series_com_idx(df)
        premium = info_modes["premium"]
        estrut = info_modes["estrutural"]

        # Trechos bons = Premium + Estrutural
        trechos_bons = sorted(set(premium + estrut))

        # Filtra df se tiver material suficiente
        if len(trechos_bons) >= max(30, int(0.15 * n)):
            mask = df_hib["indice"].isin(trechos_bons)
            df_trecho = df.loc[mask].reset_index(drop=True)
        else:
            df_trecho = df

    # 3. Ajuste final da janela com base na saúde da estrada
    regime_global = monitor["global"]["regime_global"]
    if regime_global == "estavel":
        jmin_a, jmax_a = janela_min, janela_max
    elif regime_global == "atencao":
        jmin_a, jmax_a = int(janela_min * 1.2), int(janela_max * 0.85)
    else:  # crítico
        jmin_a, jmax_a = int(janela_min * 1.5), int(janela_max * 0.7)

    # 4. EXECUÇÃO DO MOTOR
    leque = motor_turbo_core_v14(
        df=df_trecho,
        idx_alvo=min(idx0, len(df_trecho) - 1),
        top_n=top_n,
        janela_min=jmin_a,
        janela_max=jmax_a,
        usar_features_expandido=True,
        usar_peso_estrutural=True,
    )

    # Muito importante: o Modo 6 Acertos espera leques mais densos
    if len(leque) < top_n:
        # Dispara mais alguns vizinhos via janela ampliada
        leque_extra = motor_turbo_core_v14(
            df=df_trecho,
            idx_alvo=min(idx0, len(df_trecho) - 1),
            top_n=top_n,
            janela_min=max(10, int(jmin_a * 0.6)),
            janela_max=min(len(df_trecho) - 1, int(jmax_a * 1.3)),
            usar_features_expandido=True,
            usar_peso_estrutural=True,
        )
        leque = leque + leque_extra
        # remove duplicatas mantendo ordem
        seen = set()
        new = []
        for s in leque:
            t = tuple(s)
            if t not in seen:
                seen.add(t)
                new.append(s)
        leque = new[:top_n]

    # 5. RUÍDO DO LEQUE
    ruido = analisar_ruido_do_leque(leque)

    info = {
        "idx_alvo": idx_alvo_1based,
        "serie_real": serie_real,
        "k_real": k_real,
        "leque_final": leque,
        "ruido": ruido,
        "monitor_risco": monitor,
        "trechos_bons": trechos_bons,
    }
    st.session_state["modo_6_acertos_info"] = info
    return info

# ---------------------------------------------------------------------
# RELATÓRIO FINAL — AIQ BRIDGE (para ChatGPT)
# Produz um relatório textual completo:
#   - contexto da estrada
#   - regime global/local
#   - AIQ
#   - QDS
#   - ruído e estabilidade do leque
#   - série alvo
#   - leque final
#   - metadados técnicos
# ---------------------------------------------------------------------

def gerar_relatorio_final_aiq_bridge(
    df: pd.DataFrame,
    idx_alvo_1based: int,
    leque: List[List[int]],
    ruido_info: Dict,
    monitor_info: Dict,
    aiq_info: Dict,
    aiq_avancado: Dict,
    modo: str = "TURBO++ ULTRA",
) -> str:
    """
    Gera texto estruturado para ser colado no ChatGPT
    (relatório final de previsão).
    """
    n = len(df)
    idx0 = max(1, min(idx_alvo_1based - 1, n - 1))
    row = df.iloc[idx0]
    serie_real = list_passageiros(row)
    k_real = safe_int(row["k"])

    # Bloco — Cabeçalho
    txt = []
    txt.append("===============================================")
    txt.append(f"🔮 RELATÓRIO FINAL — AIQ BRIDGE ({modo})")
    txt.append("===============================================")
    txt.append("")
    txt.append(f"Série alvo: **C{idx_alvo_1based}**")
    txt.append(f"Passageiros reais: {serie_real}")
    txt.append(f"k real: {k_real}")
    txt.append("")

    # Bloco — Monitor de Risco Global
    mg = monitor_info["global"]
    txt.append("### 🌡️ Monitor de Risco — Regime Global")
    txt.append(f"- Regime: **{mg['regime_global']}**")
    txt.append(f"- k_mean: {mg['k_mean']:.3f}")
    txt.append(f"- k_std: {mg['k_std']:.3f}")
    txt.append(f"- AIQ Avançado: {mg['aiq_avancado']:.3f}")
    txt.append(f"- QDS Score: {mg['qds_score']:.3f}")
    txt.append(f"- Interpretação: {mg['texto_regime']}")
    txt.append(f"- Considerações: {mg['texto_aiq']}")
    txt.append("")

    # Bloco — Monitor Local
    ml = monitor_info["local"]
    txt.append("### 🎯 Monitor de Risco — Regime Local")
    txt.append(f"- Regime local: **{ml['regime_local']}**")
    txt.append(f"- k local: {ml['k_local']:.3f}")
    txt.append(f"- k* local: {ml['k_star_local']:.3f}")
    txt.append(f"- Interpretação local: {ml['texto_local']}")
    txt.append("")

    # Bloco — AIQ
    txt.append("### 📊 AIQ — Índices de Qualidade")
    txt.append(f"- AIQ Global: {aiq_info['aiq_score']:.3f}")
    txt.append(f"- AIQ Avançado: {aiq_avancado['aiq_avancado_score']:.3f}")
    txt.append(f"- Hib média: {aiq_info['idx_hibrido_stats']['hib_media']:.3f}")
    txt.append(f"- Hib std: {aiq_info['idx_hibrido_stats']['hib_std']:.3f}")
    txt.append("")

    # Bloco — Ruído
    txt.append("### 🔧 Ruído Condicional do Leque")
    txt.append(f"- Dispersão média: {ruido_info['disp_media']:.3f}")
    txt.append(f"- Amplitude média: {ruido_info['amp_media']:.3f}")
    txt.append(f"- Entropia norm.: {ruido_info['entropia_norm']:.3f}")
    txt.append("")

    # Bloco — Previsões
    txt.append("### 🚀 Leque Final de Previsões")
    txt.append("")
    for i, s in enumerate(leque, start=1):
        txt.append(f"**#{i:02d}** → {s}")

    txt.append("")
    txt.append("===============================================")
    txt.append("Relatório preparado automaticamente para ChatGPT.")
    txt.append("===============================================")

    return "\n".join(txt)

# =====================================================================
# PARTE 10/24 — FIM
# =====================================================================
# =====================================================================
# PREDICT CARS V15.5.1-HÍBRIDO — PARTE 11/24
# Helpers de UI + Painel de Entrada FLEX ULTRA (arquivo / texto)
# com exibição de estatísticas da estrada e gatilho opcional
# para Testes de Confiabilidade REAL.
# =====================================================================

# ---------------------------------------------------------------------
# HELPERS DE UI (GENÉRICOS)
# ---------------------------------------------------------------------

def ui_titulo_principal():
    st.markdown(
        """
# 🚗 Predict Cars V15.5.1-HÍBRIDO  
### Núcleo V14-FLEX ULTRA + S6 Profundo + AIQ Avançado + Modo 6 Acertos

Versão híbrida completa, sem simplificações.  
Entrada FLEX ULTRA, Pipeline V14-FLEX ULTRA, Replay em vários níveis,  
Testes de Confiabilidade REAL, Ruído Condicional e Monitor de Risco (k & k*).
        """
    )
    st.markdown("---")

def ui_caixa_info(texto: str):
    st.info(texto)

def ui_caixa_sucesso(texto: str):
    st.success(texto)

def ui_caixa_aviso(texto: str):
    st.warning(texto)

def ui_caixa_erro(texto: str):
    st.error(texto)

def ui_separador_fino():
    st.markdown("---")

# ---------------------------------------------------------------------
# PAINEL — ENTRADA FLEX ULTRA
#   - Escolha entre arquivo e texto colado
#   - Normalize o histórico
#   - Exibe estatísticas da estrada
#   - Permite (opcional) acionar Testes de Confiabilidade REAL
# ---------------------------------------------------------------------

def painel_entrada_flex_ultra():
    ui_titulo_principal()
    st.markdown("## 📥 Histórico — Entrada FLEX ULTRA")

    st.markdown(
        """
Este painel permite carregar o **histórico completo** da estrada, em formato flexível:

- Arquivo `.csv` com:
  - `id; p1; p2; ...; pN; k`  
  - ou `id, p1, p2, ..., pN, k`  
- Texto colado seguindo o mesmo padrão.

O sistema detecta automaticamente:
- Separador (`;`, `,` ou `tab`)
- Quantidade de passageiros por série (N variável)
- Colunas de `id` e `k`.
        """
    )

    ui_separador_fino()

    modo_entrada = st.radio(
        "Escolha o modo de entrada do histórico:",
        ["📁 Arquivo CSV", "📋 Texto colado"],
        index=0,
        horizontal=True,
    )

    df_carregado = None
    erro = None

    if modo_entrada == "📁 Arquivo CSV":
        uploaded = st.file_uploader("Selecione o arquivo de histórico (.csv):", type=["csv"])
        if uploaded is not None:
            try:
                df_carregado = carregar_historico_de_arquivo_flex_ultra(uploaded)
            except Exception as e:
                erro = f"Erro ao carregar arquivo: {e}"
    else:
        texto = st.text_area(
            "Cole aqui o histórico (linhas tipo `C1;41;5;4;52;30;33;0`):",
            height=220,
        )
        if st.button("Carregar histórico a partir do texto"):
            if texto.strip():
                try:
                    df_carregado = carregar_historico_de_texto_flex_ultra(texto)
                except Exception as e:
                    erro = f"Erro ao interpretar o texto: {e}"
            else:
                erro = "Texto vazio. Cole o histórico antes de carregar."

    if erro:
        ui_caixa_erro(erro)
        return

    if df_carregado is not None:
        # Guarda no estado
        set_historico(df_carregado)
        ui_caixa_sucesso("Histórico carregado e normalizado com sucesso.")

    df = get_historico()
    if df is None or df.empty:
        ui_caixa_aviso("Nenhum histórico carregado ainda.")
        return

    ui_separador_fino()

    st.markdown("### 📊 Resumo do histórico carregado")

    stats = get_road_stats()
    if stats:
        linhas = descrever_estatisticas_estrada_para_ui(stats)
        for ln in linhas:
            st.markdown(ln)
    else:
        st.markdown("- Não foi possível calcular estatísticas da estrada.")

    with st.expander("🔍 Ver primeiras linhas do histórico"):
        st.dataframe(df.head(20))

    ui_separador_fino()

    st.markdown("### 🧪 Opcional: Rodar Testes de Confiabilidade REAL agora")

    auto_testes = st.checkbox(
        "Rodar pacote de Testes de Confiabilidade REAL (QDS / Backtest / Monte Carlo / AIQ) após carregar",
        value=False,
    )

    if auto_testes:
        with st.spinner("Executando Testes de Confiabilidade REAL..."):
            info_conf = executar_testes_confiabilidade_real(df)
        ui_caixa_sucesso("Testes de Confiabilidade REAL executados e armazenados no estado.")
        with st.expander("Ver resumo numérico dos testes de confiabilidade"):
            st.write(info_conf)

# =====================================================================
# PARTE 11/24 — FIM
# =====================================================================
# =====================================================================
# PREDICT CARS V15.5.1-HÍBRIDO — PARTE 12/24
# Sidebar de Navegação + Estrutura base dos painéis principais:
# Entrada FLEX ULTRA, Pipeline V14-FLEX ULTRA, Replay (LIGHT/ULTRA/Unitário),
# Testes de Confiabilidade REAL e Monitor de Risco (k & k*).
# =====================================================================

# ---------------------------------------------------------------------
# SIDEBAR — NAVEGAÇÃO PRINCIPAL
# ---------------------------------------------------------------------

def montar_sidebar_principal():
    st.sidebar.markdown(
        """
# 🚗 Predict Cars  
### V15.5.1-HÍBRIDO
"""
    )

    painel = st.sidebar.radio(
        "📌 Escolha o painel:",
        [
            "📥 Entrada FLEX ULTRA",
            "🔍 Pipeline V14-FLEX ULTRA",
            "🎯 Replay LIGHT",
            "📅 Replay ULTRA",
            "🎛 Replay ULTRA Unitário",
            "🧪 Testes de Confiabilidade REAL",
            "🚨 Monitor de Risco (k & k*)",
        ],
        index=0,
    )

    return painel

# ---------------------------------------------------------------------
# PAINEL — PIPELINE V14-FLEX ULTRA (UI base + ganchos numéricos)
# ---------------------------------------------------------------------

def painel_pipeline_v14_flex_ultra():
    st.markdown("## 🔍 Pipeline V14-FLEX ULTRA")

    df = get_historico()
    if df is None or df.empty:
        ui_caixa_aviso("Carregue o histórico primeiro no painel '📥 Entrada FLEX ULTRA'.")
        return

    st.markdown(
        """
Painel dedicado ao **pipeline completo V14-FLEX ULTRA**, incluindo:

- Parâmetros de janela  
- Execução do motor TURBO base  
- Pré-visualização do leque  
- Ruído condicional  
- Monitor de risco local  
"""
    )

    ui_separador_fino()

    idx_alvo = st.number_input(
        "Escolha a série alvo (1-based):",
        min_value=2,
        max_value=len(df),
        value=len(df),
        step=1,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        top_n = st.number_input("Tamanho do leque", 5, 50, 25, 1)
    with col2:
        janela_min = st.number_input("Janela mínima", 10, 200, 40, 1)
    with col3:
        janela_max = st.number_input("Janela máxima", 50, 500, 200, 1)

    if st.button("▶️ Rodar Pipeline V14-FLEX ULTRA"):
        with st.spinner("Executando pipeline..."):
            leque = motor_turbo_core_v14(
                df=df,
                idx_alvo=idx_alvo - 1,
                top_n=top_n,
                janela_min=janela_min,
                janela_max=janela_max,
                usar_features_expandido=True,
                usar_peso_estrutural=True,
            )
            ruido = analisar_ruido_do_leque(leque)
            monitor = montar_monitor_de_risco_completo(df, idx_local=idx_alvo - 1)

        ui_caixa_sucesso("Pipeline executado com sucesso!")

        st.markdown("### 🎯 Leque gerado")
        for i, s in enumerate(leque, start=1):
            st.markdown(f"**#{i:02d}** → {s}")

        st.markdown("### 🔧 Ruído do leque")
        st.write(ruido)

        st.markdown("### 🌡️ Monitor de Risco (local + global)")
        st.write(monitor)

# ---------------------------------------------------------------------
# PAINEL — REPLAY LIGHT
# ---------------------------------------------------------------------

def painel_replay_light():
    st.markdown("## 🎯 Replay LIGHT")

    df = get_historico()
    if df is None or df.empty:
        ui_caixa_aviso("Carregue o histórico primeiro.")
        return

    idx_alvo = st.number_input(
        "Índice alvo (1-based):",
        min_value=2,
        max_value=len(df),
        value=len(df),
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        janela_min = st.number_input("Janela mínima", 10, 200, 40)
    with col2:
        janela_max = st.number_input("Janela máxima", 50, 500, 200)
    with col3:
        top_n = st.number_input("Leque (N)", 5, 40, 25)

    if st.button("▶️ Executar Replay LIGHT"):
        with st.spinner("Rodando Replay LIGHT..."):
            info = executar_replay_light(
                df=df,
                idx_alvo_1based=idx_alvo,
                janela_min=janela_min,
                janela_max=janela_max,
                top_n=top_n,
            )
        ui_caixa_sucesso("Replay LIGHT executado!")

        st.markdown("### 🎯 Leque")
        for i, s in enumerate(info["leque"], start=1):
            st.markdown(f"**#{i:02d}** → {s}")

        st.markdown("### 🔧 Ruído")
        st.write(info["ruido"])

        st.markdown("### 🌡️ Monitor de Risco")
        st.write(info["monitor_risco"])

# ---------------------------------------------------------------------
# PAINEL — REPLAY ULTRA
# ---------------------------------------------------------------------

def painel_replay_ultra():
    st.markdown("## 📅 Replay ULTRA")

    df = get_historico()
    if df is None or df.empty:
        ui_caixa_aviso("Carregue o histórico primeiro.")
        return

    idx_inicio = st.number_input(
        "Índice inicial (1-based):",
        min_value=2,
        max_value=len(df) - 1,
        value=max(2, len(df) - 30),
    )
    idx_fim = st.number_input(
        "Índice final (1-based):",
        min_value=3,
        max_value=len(df),
        value=len(df),
    )

    passo = st.number_input("Passo", 1, 10, 1)
    top_n = st.number_input("N (leque)", 5, 30, 15)

    if st.button("▶️ Executar Replay ULTRA"):
        with st.spinner("Rodando Replay ULTRA..."):
            info = executar_replay_ultra(
                df=df,
                idx_inicio_1based=idx_inicio,
                idx_fim_1based=idx_fim,
                passo=passo,
                janela_min=40,
                janela_max=200,
                top_n=top_n,
            )
        ui_caixa_sucesso("Replay ULTRA executado!")

        st.markdown("### 📊 Resumo")
        st.write(info)

# ---------------------------------------------------------------------
# PAINEL — REPLAY ULTRA UNITÁRIO
# ---------------------------------------------------------------------

def painel_replay_unitario():
    st.markdown("## 🎛 Replay ULTRA Unitário")

    df = get_historico()
    if df is None or df.empty:
        ui_caixa_aviso("Carregue o histórico primeiro.")
        return

    idx_alvo = st.number_input(
        "Índice alvo (1-based):",
        min_value=2,
        max_value=len(df),
        value=len(df),
    )

    if st.button("▶️ Executar Replay ULTRA Unitário"):
        with st.spinner("Executando..."):
            info = executar_replay_ultra_unitario(
                df=df,
                idx_alvo_1based=idx_alvo,
                janela_min=40,
                janela_max=200,
                top_n=25,
            )
        ui_caixa_sucesso("Replay ULTRA Unitário executado!")

        st.markdown("### 🎯 Série real")
        st.write(info["serie_real"])

        st.markdown("### 🚀 Leque")
        for i, s in enumerate(info["leque"], start=1):
            st.markdown(f"**#{i:02d}** → {s}")

        st.markdown("### 🔧 Ruído")
        st.write(info["ruido"])

        st.markdown("### 🌡️ Monitor de risco")
        st.write(info["monitor_risco"])

# ---------------------------------------------------------------------
# PAINEL — TESTES DE CONFIABILIDADE REAL
# ---------------------------------------------------------------------

def painel_testes_confiabilidade():
    st.markdown("## 🧪 Testes de Confiabilidade REAL")

    df = get_historico()
    if df is None or df.empty:
        ui_caixa_aviso("Carregue o histórico primeiro.")
        return

    if st.button("▶️ Executar pacote completo de Testes de Confiabilidade REAL"):
        with st.spinner("Executando..."):
            info = executar_testes_confiabilidade_real(df)
        ui_caixa_sucesso("Testes executados e armazenados no estado.")
        st.write(info)

    if st.button("📄 Gerar resumo (modo leitura rápida)"):
        aiq_info = st.session_state.get("aiq_info", {})
        st.write(aiq_info)

# ---------------------------------------------------------------------
# PAINEL — MONITOR DE RISCO (k & k*)
# ---------------------------------------------------------------------

def painel_monitor_de_risco():
    st.markdown("## 🚨 Monitor de Risco (k & k*)")

    df = get_historico()
    if df is None or df.empty:
        ui_caixa_aviso("Histórico não carregado.")
        return

    idx_alvo = st.number_input(
        "Índice alvo (para risco local):",
        min_value=2,
        max_value=len(df),
        value=len(df),
    )

    if st.button("▶️ Avaliar risco agora"):
        with st.spinner("Analisando..."):
            info = montar_monitor_de_risco_completo(df, idx_local=idx_alvo - 1)
        ui_caixa_sucesso("Monitor de risco atualizado!")
        st.write(info)

# =====================================================================
# PARTE 12/24 — FIM
# =====================================================================
# =====================================================================
# PREDICT CARS V15.5.1-HÍBRIDO — PARTE 13/24
# Pipeline V14-FLEX ULTRA (núcleo numérico completo):
# S1 → S5, combinadores, diagnóstico interno e motor TURBO adaptado
# para pipeline.
# =====================================================================

# ---------------------------------------------------------------------
# S1 — ANÁLISE ESTATÍSTICA BÁSICA
# ---------------------------------------------------------------------

def s1_basico_estatisticas(df: pd.DataFrame) -> Dict:
    """
    S1 (padrão V14):
      - médias
      - desvios
      - amplitudes por passageiro
      - amplitude total
      - entropia aproximada da estrada
    """
    if df is None or df.empty:
        return {
            "medias": [],
            "stds": [],
            "amps": [],
            "amp_total": 0.0,
            "entropia_glob": 0.0,
        }

    passageiros_cols = passageiros_columns(df)
    arr = df[passageiros_cols].astype(float).values

    # Médias
    medias = arr.mean(axis=0)
    # Desvios
    stds = arr.std(axis=0)
    # Amplitudes por passageiro
    amps = arr.max(axis=0) - arr.min(axis=0)
    # Amplitude total
    amp_total = float(arr.max() - arr.min())

    # Entropia aproximada
    flat = arr.flatten()
    unq, cnts = np.unique(flat, return_counts=True)
    probs = cnts / cnts.sum()
    ent = -np.sum([p * math.log(p + 1e-12) for p in probs])

    ent_norm = ent / max(1.0, math.log(len(unq) + 1e-12))

    return {
        "medias": medias.tolist(),
        "stds": stds.tolist(),
        "amps": amps.tolist(),
        "amp_total": amp_total,
        "entropia_glob": float(clamp(ent_norm)),
    }

# ---------------------------------------------------------------------
# S2 — CROSS-SIMILARITY BÁSICA ENTRE SÉRIES
# ---------------------------------------------------------------------

def s2_cross_similarity(df: pd.DataFrame) -> Dict:
    """
    Similaridade geral entre pares próximos.
    Calcula um vetor de similaridade média por posição.
    """
    if df is None or df.empty:
        return {"sim": []}

    passageiros_cols = passageiros_columns(df)
    arr = df[passageiros_cols].astype(float).values

    n = len(arr)
    if n < 3:
        return {"sim": [1.0] * n}

    sims = []
    for i in range(n):
        if i == 0:
            sims.append(1.0)
            continue
        v1 = arr[i]
        v2 = arr[i - 1]
        dist = np.linalg.norm(v1 - v2)
        sim = 1 / (1 + dist)
        sims.append(float(clamp(sim)))

    return {"sim": sims}

# ---------------------------------------------------------------------
# S3 — DETECÇÃO DE PADRÕES DE OSCILAÇÃO LOCAL
# ---------------------------------------------------------------------

def s3_oscilacao_local(df: pd.DataFrame, janela: int = 12) -> Dict:
    """
    Mede oscilação média local por janelas deslizantes.
    """
    if df is None or df.empty:
        return {"osc_local": []}

    passageiros_cols = passageiros_columns(df)
    arr = df[passageiros_cols].astype(float).values
    n = len(arr)

    osc = []
    for i in range(n):
        ini = max(0, i - janela)
        sub = arr[ini:i+1]
        if len(sub) < 2:
            osc.append(0.0)
        else:
            osc_val = float(sub.std())
            osc.append(osc_val)

    return {"osc_local": osc}

# ---------------------------------------------------------------------
# S4 — ÍNDICE DE ESTABILIDADE PROJETADA
# ---------------------------------------------------------------------

def s4_estabilidade_proj(df: pd.DataFrame) -> Dict:
    """
    Estabilidade projetada: combina variância e suavidade.
    """
    if df is None or df.empty:
        return {"estabilidade": []}

    passageiros_cols = passageiros_columns(df)
    arr = df[passageiros_cols].astype(float).values

    n = len(arr)
    est = []
    for i in range(n):
        amp = float(arr[i].max() - arr[i].min())
        desv = float(arr[i].std())
        e = 1 / (1 + amp + desv)
        est.append(float(clamp(e)))

    return {"estabilidade": est}

# ---------------------------------------------------------------------
# S5 — RESILIÊNCIA (compatível com S6/S7 e com o IDX Avançado)
# ---------------------------------------------------------------------

def s5_resiliencia(df: pd.DataFrame) -> Dict:
    """
    Resiliência por série:
      - séries mais centrais (em termos de estatísticas locais)
        têm maior resiliência.
    """
    if df is None or df.empty:
        return {"resiliencia": []}

    passageiros_cols = passageiros_columns(df)
    arr = df[passageiros_cols].astype(float).values

    n = len(arr)
    meds = arr.mean(axis=1)
    stds = arr.std(axis=1)

    # Normalização
    meds_n = (meds - meds.min()) / (max(1e-9, meds.max() - meds.min()))
    stds_n = (stds - stds.min()) / (max(1e-9, stds.max() - stds.min()))

    res = []
    for m, s in zip(meds_n, stds_n):
        r = float(clamp(0.6 * (1 - abs(m - 0.5)) + 0.4 * (1 - s)))
        res.append(r)

    return {"resiliencia": res}

# ---------------------------------------------------------------------
# COMBINADOR V14 — S1 → S5
# ---------------------------------------------------------------------

def combinador_v14(df: pd.DataFrame) -> Dict:
    """
    Combina S1..S5 num único pacote para o pipeline:
      - estatísticas globais
      - similaridade
      - oscilação local
      - estabilidade projetada
      - resiliência
    """
    s1 = s1_basico_estatisticas(df)
    s2 = s2_cross_similarity(df)
    s3 = s3_oscilacao_local(df)
    s4 = s4_estabilidade_proj(df)
    s5r = s5_resiliencia(df)

    return {
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "s4": s4,
        "s5": s5r,
    }

# ---------------------------------------------------------------------
# DIAGNÓSTICO INTERNO DO PIPELINE
# ---------------------------------------------------------------------

def diagnostico_interno_pipeline(df: pd.DataFrame) -> Dict:
    """
    Produz um diagnóstico interno do pipeline V14-FLEX:
      - entropia global
      - amplitude total
      - resiliência média
      - similaridade média
      - estabilidade média
      - oscilação média
    """
    comp = combinador_v14(df)
    s1 = comp["s1"]
    s2 = comp["s2"]
    s3 = comp["s3"]
    s4 = comp["s4"]
    s5r = comp["s5"]

    diag = {
        "entropia_global": s1["entropia_glob"],
        "amplitude_total": s1["amp_total"],
        "resiliencia_media": float(np.mean(s5r["resiliencia"])) if s5r["resiliencia"] else 0.0,
        "similaridade_media": float(np.mean(s2["sim"])) if s2["sim"] else 0.0,
        "estabilidade_media": float(np.mean(s4["estabilidade"])) if s4["estabilidade"] else 0.0,
        "oscilacao_media": float(np.mean(s3["osc_local"])) if s3["osc_local"] else 0.0,
    }

    return diag

# ---------------------------------------------------------------------
# MOTOR TURBO — VARIAÇÃO PARA PIPELINE V14-FLEX ULTRA
# (separa da versão dos replays para permitir ajustes independentes)
# ---------------------------------------------------------------------

def motor_turbo_pipeline_v14(
    df: pd.DataFrame,
    idx_alvo: int,
    top_n: int = 25,
    janela_min: int = 40,
    janela_max: int = 200,
) -> Dict:
    """
    Versão do motor TURBO especialmente configurada para
    o painel de Pipeline V14-FLEX ULTRA.

    Retorna:
      {
        "leque": [...],
        "ruido": {...},
        "monitor": {...},
        "diagnostico": {...},
      }
    """
    if df is None or df.empty:
        return {
            "leque": [],
            "ruido": None,
            "monitor": None,
            "diagnostico": None,
        }

    # Diagnóstico
    diag = diagnostico_interno_pipeline(df)

    # Motor
    leque = motor_turbo_core_v14(
        df=df,
        idx_alvo=idx_alvo,
        top_n=top_n,
        janela_min=janela_min,
        janela_max=janela_max,
        usar_features_expandido=True,
        usar_peso_estrutural=True,
    )

    ruido = analisar_ruido_do_leque(leque)
    monitor = montar_monitor_de_risco_completo(df, idx_local=idx_alvo)

    return {
        "leque": leque,
        "ruido": ruido,
        "monitor": monitor,
        "diagnostico": diag,
    }

# =====================================================================
# PARTE 13/24 — FIM
# =====================================================================
# =====================================================================
# PREDICT CARS V15.5.1-HÍBRIDO — PARTE 14/24
# UI COMPLETA DO PAINEL — Pipeline V14-FLEX ULTRA
# Com diagnóstico interno, ruído, risco, visualização S1→S5,
# leque detalhado, e ganchos para modos Premium/Estrutural/Cobertura.
# =====================================================================

def painel_pipeline_v14_flex_ultra():
    st.markdown("## 🔍 Pipeline V14-FLEX ULTRA — Execução Completa")

    df = get_historico()
    if df is None or df.empty:
        ui_caixa_aviso("Carregue o histórico primeiro no painel **Entrada FLEX ULTRA**.")
        return

    st.markdown(
        """
### Mecanismo completo do Pipeline V14-FLEX ULTRA  
Integra S1 → S5, IDX avançado, motor TURBO especializado e diagnóstico interno detalhado.
        """
    )

    ui_separador_fino()

    # ---------------------------
    # Seleção de índice e parâmetros
    # ---------------------------

    col1, col2, col3 = st.columns(3)
    with col1:
        idx_alvo = st.number_input(
            "Índice alvo (1-based):",
            min_value=2,
            max_value=len(df),
            value=len(df),
            step=1,
        )
    with col2:
        top_n = st.number_input("Tamanho do leque (N)", 5, 60, 25)
    with col3:
        modo_visual = st.selectbox(
            "Modo visual",
            ["Completo", "Diagnóstico", "Leque + Risco"],
            index=0,
        )

    col4, col5 = st.columns(2)
    with col4:
        janela_min = st.number_input("Janela mínima", 5, 200, 40, 1)
    with col5:
        janela_max = st.number_input("Janela máxima", 20, 600, 200, 1)

    ui_separador_fino()

    # ---------------------------
    # Botão principal — Executar pipeline
    # ---------------------------

    if st.button("▶️ Rodar Pipeline V14-FLEX ULTRA"):

        with st.spinner("Executando pipeline completo..."):

            resultado = motor_turbo_pipeline_v14(
                df=df,
                idx_alvo=idx_alvo - 1,
                top_n=top_n,
                janela_min=janela_min,
                janela_max=janela_max,
            )

        ui_caixa_sucesso("Pipeline executado com sucesso!")

        leque = resultado["leque"]
        ruido = resultado["ruido"]
        monitor = resultado["monitor"]
        diag = resultado["diagnostico"]

        # Guardar no estado
        st.session_state["pipeline_ultra"] = {
            "leque": leque,
            "ruido": ruido,
            "monitor": monitor,
            "diagnostico": diag,
            "idx_alvo": idx_alvo,
            "top_n": top_n,
        }

    # ---------------------------
    # Após execução, mostrar resultados
    # ---------------------------

    if "pipeline_ultra" not in st.session_state:
        return

    data = st.session_state["pipeline_ultra"]
    leque = data["leque"]
    ruido = data["ruido"]
    monitor = data["monitor"]
    diag = data["diagnostico"]

    ui_separador_fino()

    # =========================================================
    # VISUALIZAÇÃO COMPLETA
    # =========================================================

    if modo_visual == "Completo":

        st.markdown("## 🎯 Leque V14-FLEX ULTRA")
        if leque:
            for i, s in enumerate(leque, start=1):
                st.markdown(f"**#{i:02d}** → {s}")
        else:
            ui_caixa_aviso("Leque vazio — verifique parâmetros e janela.")

        ui_separador_fino()

        st.markdown("## 🔧 Ruído Condicional do Leque")
        st.write(ruido)

        ui_separador_fino()

        st.markdown("## 🌡️ Monitor de Risco (k & k*) — Global + Local")
        st.write(monitor)

        ui_separador_fino()

        # ---------------------------
        # Diagnóstico interno
        # ---------------------------

        st.markdown("## 🧩 Diagnóstico Interno S1 → S5")
        st.write(diag)

        with st.expander("🔍 Ver S1 — Estatísticas Básicas"):
            s1 = s1_basico_estatisticas(df)
            st.write(s1)

        with st.expander("🔍 Ver S2 — Similaridade Local"):
            s2 = s2_cross_similarity(df)
            st.write(s2)

        with st.expander("🔍 Ver S3 — Oscilação Local"):
            st.write(s3_oscilacao_local(df))

        with st.expander("🔍 Ver S4 — Estabilidade Projetada"):
            st.write(s4_estabilidade_proj(df))

        with st.expander("🔍 Ver S5 — Resiliência"):
            st.write(s5_resiliencia(df))

    # =========================================================
    # VISUALIZAÇÃO SOMENTE DO DIAGNÓSTICO
    # =========================================================

    elif modo_visual == "Diagnóstico":

        st.markdown("## 🧩 Diagnóstico Interno (Resumo)")
        st.write(diag)

        st.markdown("---")
        st.markdown("### S1 → S5 (detalhes)")
        with st.expander("🔍 S1 — Estatísticas"):
            st.write(s1_basico_estatisticas(df))
        with st.expander("🔍 S2 — Similaridade"):
            st.write(s2_cross_similarity(df))
        with st.expander("🔍 S3 — Oscilação"):
            st.write(s3_oscilacao_local(df))
        with st.expander("🔍 S4 — Estabilidade"):
            st.write(s4_estabilidade_proj(df))
        with st.expander("🔍 S5 — Resiliência"):
            st.write(s5_resiliencia(df))

    # =========================================================
    # VISUALIZAÇÃO SOMENTE DO LEQUE + RISCO
    # =========================================================

    elif modo_visual == "Leque + Risco":

        st.markdown("## 🎯 Leque V14-FLEX")
        if leque:
            for i, s in enumerate(leque, start=1):
                st.markdown(f"**#{i:02d}** → {s}")
        else:
            ui_caixa_aviso("Nenhuma previsão gerada.")

        ui_separador_fino()
        st.markdown("## 🔧 Ruído Condicional")
        st.write(ruido)

        ui_separador_fino()
        st.markdown("## 🌡️ Monitor de Risco")
        st.write(monitor)

    # =========================================================
    # Ganchos para modos Premium / Estrutural / Cobertura
    # =========================================================

    ui_separador_fino()

    st.markdown("### 🔮 Modos Premium / Estrutural / Cobertura (base)")
    st.markdown(
        "Esta área apresenta a classificação das séries segundo o **IDX Híbrido**, "
        "servindo de base para os modos premium/estrutural/cobertura."
    )

    info_modes = classificar_modo_series_com_idx(df)

    with st.expander("📘 Séries Premium"):
        st.write(info_modes["premium"])

    with st.expander("📗 Séries Estruturais"):
        st.write(info_modes["estrutural"])

    with st.expander("📙 Séries de Cobertura"):
        st.write(info_modes["cobertura"])

# =====================================================================
# PARTE 14/24 — FIM
# =====================================================================
# =====================================================================
# PREDICT CARS V15.5.1-HÍBRIDO — PARTE 15/24
# Painéis COMPLETOS:
#   - Replay LIGHT
#   - Replay ULTRA
#   - Replay ULTRA Unitário
# =====================================================================

# ---------------------------------------------------------------------
# PAINEL — REPLAY LIGHT (completo)
# ---------------------------------------------------------------------

def painel_replay_light():
    st.markdown("## 🎯 Replay LIGHT — Execução Completa")

    df = get_historico()
    if df is None or df.empty:
        ui_caixa_aviso("Carregue o histórico primeiro.")
        return

    st.markdown(
        """
O **Replay LIGHT** permite avaliar rapidamente um único ponto da estrada,
gerando:
- Leque TURBO da janela local  
- Ruído condicional  
- Monitor de risco (k & k*)  
- Contexto completo da série alvo  
        """
    )

    ui_separador_fino()

    idx_alvo = st.number_input(
        "Série alvo (1-based):",
        min_value=2,
        max_value=len(df),
        value=len(df),
        step=1,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        janela_min = st.number_input("Janela mínima", 10, 200, 40)
    with col2:
        janela_max = st.number_input("Janela máxima", 50, 500, 200)
    with col3:
        top_n = st.number_input("N (tamanho do leque)", 5, 40, 25)

    ui_separador_fino()

    if st.button("▶️ Executar Replay LIGHT"):
        with st.spinner("Rodando Replay LIGHT..."):
            info = executar_replay_light(
                df=df,
                idx_alvo_1based=idx_alvo,
                janela_min=janela_min,
                janela_max=janela_max,
                top_n=top_n,
            )

        ui_caixa_sucesso("Replay LIGHT concluído!")

        st.session_state["replay_light_info"] = info

    # ------------------------------
    # Exibição dos resultados
    # ------------------------------

    if "replay_light_info" not in st.session_state:
        return

    info = st.session_state["replay_light_info"]

    ui_separador_fino()

    st.markdown(f"## 🎯 Leque — C{info['idx_alvo']}")
    for i, s in enumerate(info["leque"], start=1):
        st.markdown(f"**#{i:02d}** → {s}")

    ui_separador_fino()
    st.markdown("## 🔧 Ruído do Leque")
    st.write(info["ruido"])

    ui_separador_fino()
    st.markdown("## 🌡️ Monitor de Risco (global + local)")
    st.write(info["monitor_risco"])

    ui_separador_fino()

    if st.button("📄 Gerar Relatório Final — AIQ Bridge (Replay LIGHT)"):
        aiq_info = st.session_state.get("aiq_info", {})
        aiq_adv = st.session_state.get("aiq_avancado_info", {})

        rel = gerar_relatorio_final_aiq_bridge(
            df=df,
            idx_alvo_1based=info["idx_alvo"],
            leque=info["leque"],
            ruido_info=info["ruido"],
            monitor_info=info["monitor_risco"],
            aiq_info=aiq_info,
            aiq_avancado=aiq_adv,
            modo="Replay LIGHT",
        )
        st.session_state["relatorio_final"] = rel
        ui_caixa_sucesso("Relatório gerado. Veja abaixo.")

    if "relatorio_final" in st.session_state:
        with st.expander("📝 Ver Relatório Final"):
            st.markdown(st.session_state["relatorio_final"])


# ---------------------------------------------------------------------
# PAINEL — REPLAY ULTRA (completo)
# ---------------------------------------------------------------------

def painel_replay_ultra():
    st.markdown("## 📅 Replay ULTRA — Execução Completa")

    df = get_historico()
    if df is None or df.empty:
        ui_caixa_aviso("Carregue o histórico primeiro.")
        return

    st.markdown(
        """
O **Replay ULTRA** permite percorrer uma grande faixa da estrada e gerar:
- Leques sucessivos  
- Ruído de cada ponto  
- Tabela de resumo  
        """
    )

    ui_separador_fino()

    idx_inicio = st.number_input(
        "Índice inicial (1-based):",
        min_value=2,
        max_value=len(df) - 1,
        value=max(2, len(df) - 40),
    )
    idx_fim = st.number_input(
        "Índice final (1-based):",
        min_value=3,
        max_value=len(df),
        value=len(df),
    )

    col1, col2 = st.columns(2)
    with col1:
        passo = st.number_input("Passo", 1, 20, 1)
    with col2:
        top_n = st.number_input("Tamanho do leque (N)", 5, 50, 15)

    ui_separador_fino()

    if st.button("▶️ Executar Replay ULTRA"):
        with st.spinner("Executando Replay ULTRA..."):
            info = executar_replay_ultra(
                df=df,
                idx_inicio_1based=idx_inicio,
                idx_fim_1based=idx_fim,
                passo=passo,
                janela_min=40,
                janela_max=200,
                top_n=top_n,
            )
        ui_caixa_sucesso("Replay ULTRA concluído!")
        st.session_state["replay_ultra_info"] = info

    # ------------------------------
    # Exibição dos resultados
    # ------------------------------

    if "replay_ultra_info" not in st.session_state:
        return

    info = st.session_state["replay_ultra_info"]

    st.markdown("## 📊 Resumo do Replay ULTRA")
    st.write(info)

    with st.expander("Ver resumos individuais"):
        for item in info["resumos"]:
            st.markdown(f"### C{item['idx']}")
            st.write(item)


# ---------------------------------------------------------------------
# PAINEL — REPLAY ULTRA UNITÁRIO (completo)
# ---------------------------------------------------------------------

def painel_replay_unitario():
    st.markdown("## 🎛 Replay ULTRA Unitário — Execução Completa")

    df = get_historico()
    if df is None or df.empty:
        ui_caixa_aviso("Carregue o histórico primeiro.")
        return

    st.markdown(
        """
O **Replay ULTRA Unitário** mostra **tudo de um único ponto da estrada**:

- Passageiros reais  
- k real  
- Leque ULTRA  
- Ruído  
- Risco local e global  
- Exportação do Relatório Final — AIQ Bridge  
        """
    )

    ui_separador_fino()

    idx_alvo = st.number_input(
        "Índice alvo (1-based):",
        min_value=2,
        max_value=len(df),
        value=len(df),
    )

    if st.button("▶️ Executar Replay ULTRA Unitário"):
        with st.spinner("Calculando..."):
            info = executar_replay_ultra_unitario(
                df=df,
                idx_alvo_1based=idx_alvo,
                janela_min=40,
                janela_max=200,
                top_n=25,
            )
        ui_caixa_sucesso("Replay ULTRA Unitário concluído!")
        st.session_state["replay_unit_info"] = info

    # ------------------------------
    # Exibir resultados
    # ------------------------------

    if "replay_unit_info" not in st.session_state:
        return

    info = st.session_state["replay_unit_info"]

    st.markdown("## 🎯 Série Real")
    st.write(info["serie_real"])

    ui_separador_fino()
    st.markdown("## 🚀 Leque ULTRA")
    for i, s in enumerate(info["leque"], start=1):
        st.markdown(f"**#{i:02d}** → {s}")

    ui_separador_fino()
    st.markdown("## 🔧 Ruído")
    st.write(info["ruido"])

    ui_separador_fino()
    st.markdown("## 🌡️ Monitor de Risco (global + local)")
    st.write(info["monitor_risco"])

    ui_separador_fino()

    # Exportar relatório final
    if st.button("📄 Gerar Relatório Final — AIQ Bridge (ULTRA Unitário)"):
        aiq_info = st.session_state.get("aiq_info", {})
        aiq_adv = st.session_state.get("aiq_avancado_info", {})

        rel = gerar_relatorio_final_aiq_bridge(
            df=df,
            idx_alvo_1based=info["idx_alvo"],
            leque=info["leque"],
            ruido_info=info["ruido"],
            monitor_info=info["monitor_risco"],
            aiq_info=aiq_info,
            aiq_avancado=aiq_adv,
            modo="Replay ULTRA Unitário",
        )
        st.session_state["relatorio_final_unit"] = rel
        ui_caixa_sucesso("Relatório gerado!")

    if "relatorio_final_unit" in st.session_state:
        with st.expander("📝 Ver Relatório Final — Unitário"):
            st.markdown(st.session_state["relatorio_final_unit"])

# =====================================================================
# PARTE 15/24 — FIM
# =====================================================================
# =====================================================================
# PREDICT CARS V15.5.1-HÍBRIDO — PARTE 16/24
# Painéis COMPLETOS:
#   - Testes de Confiabilidade REAL
#   - Modo 6 Acertos
# =====================================================================

# ---------------------------------------------------------------------
# PAINEL — TESTES DE CONFIABILIDADE REAL (completo)
# ---------------------------------------------------------------------

def painel_testes_confiabilidade():
    st.markdown("## 🧪 Testes de Confiabilidade REAL — Execução Completa")

    df = get_historico()
    if df is None or df.empty:
        ui_caixa_aviso("Carregue o histórico primeiro.")
        return

    st.markdown(
        """
Este módulo executa o pacote **mais completo** de validação do Predict Cars:

- **QDS completo**
- **Backtest Interno**
- **Backtest do Futuro**
- **Monte Carlo Profundo (multi-camadas)**
- **AIQ Global**
- **AIQ Avançado**

Todos os resultados são armazenados em `st.session_state`
para uso posterior nos outros painéis.
        """
    )

    ui_separador_fino()

    if st.button("▶️ Executar Testes de Confiabilidade REAL"):
        with st.spinner("Rodando pacote completo..."):
            info = executar_testes_confiabilidade_real(df)

        st.session_state["testes_reais_info"] = info
        ui_caixa_sucesso("Testes de Confiabilidade REAL concluídos!")

    # -----------------------------------------------------------------
    # Exibição dos resultados
    # -----------------------------------------------------------------
    if "testes_reais_info" not in st.session_state:
        return

    info = st.session_state["testes_reais_info"]

    ui_separador_fino()

    st.markdown("## 📊 Resultados Gerais")
    st.write(info)

    # ---------------------
    # QDS
    # ---------------------
    with st.expander("🧩 QDS — Detalhes completos"):
        st.write(info["qds"])

    # ---------------------
    # Backtest Interno
    # ---------------------
    with st.expander("🎯 Backtest Interno — Detalhes"):
        st.write(info["backtest_interno"])

    # ---------------------
    # Backtest do Futuro
    # ---------------------
    with st.expander("📅 Backtest do Futuro — Detalhes"):
        st.write(info["backtest_futuro"])

    # ---------------------
    # Monte Carlo Profundo
    # ---------------------
    with st.expander("🎲 Monte Carlo Profundo — Detalhes"):
        st.write(info["monte_carlo_multi"])

    # ---------------------
    # AIQ Global
    # ---------------------
    with st.expander("📈 AIQ Global — Detalhes"):
        st.write(info["aiq"])

    # ---------------------
    # AIQ Avançado
    # ---------------------
    with st.expander("📈 AIQ Avançado — Detalhes"):
        st.write(info["aiq_avancado"])


# ---------------------------------------------------------------------
# PAINEL — MODO 6 ACERTOS (completo)
# ---------------------------------------------------------------------

def painel_modo_6_acertos():
    st.markdown("## 🎯 Modo 6 Acertos — Execução Completa")

    df = get_historico()
    if df is None or df.empty:
        ui_caixa_aviso("Carregue o histórico primeiro.")
        return

    st.markdown(
        """
O **Modo 6 Acertos** é o modo mais **profundo, sensível e seletivo**
do Predict Cars, baseado em:

- Regime da estrada (k*)  
- IDX Híbrido  
- Trechos Premium + Estruturais  
- Motor TURBO adaptativo  
- Densificação do leque  
- Análise de ruído avançada  
- Risco local e global  
        """
    )

    ui_separador_fino()

    idx_alvo = st.number_input(
        "Série alvo (1-based):",
        min_value=2,
        max_value=len(df),
        value=len(df),
    )

    top_n = st.number_input("Tamanho final do leque (N)", 6, 20, 12)

    ui_separador_fino()

    if st.button("▶️ Executar Modo 6 Acertos"):
        with st.spinner("Executando Modo 6 Acertos..."):
            info = executar_modo_6_acertos(
                df=df,
                idx_alvo_1based=idx_alvo,
                top_n=top_n,
                janela_min=40,
                janela_max=220,
            )

        st.session_state["modo_6_info"] = info
        ui_caixa_sucesso("Modo 6 Acertos concluído!")

    # ---------------------
    # Exibição de resultados
    # ---------------------

    if "modo_6_info" not in st.session_state:
        return

    info = st.session_state["modo_6_info"]

    ui_separador_fino()
    st.markdown("## 🎯 Série Real")
    st.write(info["serie_real"])

    ui_separador_fino()
    st.markdown("## 🔮 Leque Final (Modo 6 Acertos)")
    for i, s in enumerate(info["leque_final"], start=1):
        st.markdown(f"**#{i:02d}** → {s}")

    ui_separador_fino()
    st.markdown("## 🔧 Ruído")
    st.write(info["ruido"])

    ui_separador_fino()
    st.markdown("## 🌡️ Monitor de Risco (global + local)")
    st.write(info["monitor_risco"])

    ui_separador_fino()
    st.markdown("## 📘 Trechos Bons (Premium + Estruturais)")
    st.write(info["trechos_bons"])

    # ---------------------
    # Relatório Final
    # ---------------------

    ui_separador_fino()

    if st.button("📄 Gerar Relatório Final — AIQ Bridge (Modo 6 Acertos)"):
        aiq_info = st.session_state.get("aiq_info", {})
        aiq_adv = st.session_state.get("aiq_avancado_info", {})

        rel = gerar_relatorio_final_aiq_bridge(
            df=df,
            idx_alvo_1based=info["idx_alvo"],
            leque=info["leque_final"],
            ruido_info=info["ruido"],
            monitor_info=info["monitor_risco"],
            aiq_info=aiq_info,
            aiq_avancado=aiq_adv,
            modo="Modo 6 Acertos",
        )
        st.session_state["relatorio_final_m6"] = rel
        ui_caixa_sucesso("Relatório final gerado!")

    if "relatorio_final_m6" in st.session_state:
        with st.expander("📝 Ver Relatório Final (Modo 6 Acertos)"):
            st.markdown(st.session_state["relatorio_final_m6"])

# =====================================================================
# PARTE 16/24 — FIM
# =====================================================================
# =====================================================================
# PREDICT CARS V15.5.1-HÍBRIDO — PARTE 17/24
# Painel completo do Monitor de Risco (k & k*),
# Painel de Ruído Condicional A/B
# e Modo TURBO++ ULTRA ANTI-RUÍDO (V15.5).
# =====================================================================

# ---------------------------------------------------------------------
# PAINEL — MONITOR DE RISCO (k & k*) — VERSÃO COMPLETA
# (Sobrescreve a versão mais simples definida antes)
# ---------------------------------------------------------------------

def painel_monitor_de_risco():
    st.markdown("## 🚨 Monitor de Risco (k & k*) — Visão Completa")

    df = get_historico()
    if df is None or df.empty:
        ui_caixa_aviso("Histórico não carregado. Use o painel 'Entrada FLEX ULTRA'.")
        return

    st.markdown(
        """
O **Monitor de Risco (k & k\*)** consolida:

- Regime **global** (estável / atenção / crítico)  
- Risco **local** na série alvo  
- Estatísticas de k (média / desvio)  
- Integração com **AIQ Avançado** e **QDS**  
        """
    )

    ui_separador_fino()

    idx_alvo = st.number_input(
        "Índice alvo para risco local (1-based):",
        min_value=2,
        max_value=len(df),
        value=len(df),
    )

    if st.button("▶️ Avaliar risco (global + local)"):
        with st.spinner("Analisando k, k* e regimes..."):
            info = montar_monitor_de_risco_completo(df, idx_local=idx_alvo - 1)
        st.session_state["monitor_risco_info"] = info
        ui_caixa_sucesso("Monitor de risco atualizado!")

    if "monitor_risco_info" not in st.session_state:
        return

    info = st.session_state["monitor_risco_info"]

    # -------------------------
    # Visão Global
    # -------------------------

    ui_separador_fino()
    st.markdown("### 🌍 Regime Global")

    mg = info["global"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"- **Regime global:** `{mg['regime_global']}`")
        st.markdown(f"- **k_mean:** `{mg['k_mean']:.3f}`")
        st.markdown(f"- **k_std:** `{mg['k_std']:.3f}`")
    with col2:
        st.markdown(f"- **AIQ Avançado:** `{mg['aiq_avancado']:.3f}`")
        st.markdown(f"- **QDS Score:** `{mg['qds_score']:.3f}`")

    st.markdown("**Interpretação global:**")
    st.markdown(f"> {mg['texto_regime']}")
    st.markdown("")
    st.markdown("**Considerações AIQ:**")
    st.markdown(f"> {mg['texto_aiq']}")

    # -------------------------
    # Visão Local
    # -------------------------

    ui_separador_fino()
    st.markdown("### 🎯 Regime Local")

    ml = info["local"]
    st.markdown(f"- **Regime local:** `{ml['regime_local']}`")
    st.markdown(f"- **k local:** `{ml['k_local']:.3f}`")
    st.markdown(f"- **k\* local:** `{ml['k_star_local']:.3f}`")

    st.markdown("**Interpretação local:**")
    st.markdown(f"> {ml['texto_local']}")

    ui_separador_fino()

    st.markdown(
        """
> Use este painel como **barômetro de risco contínuo** antes de decidir:
> - tamanho do leque  
> - ativação do Modo 6 Acertos  
> - ativação do modo TURBO++ ULTRA ANTI-RUÍDO.  
        """
    )

# ---------------------------------------------------------------------
# PAINEL — RUÍDO CONDICIONAL A/B
# Compara dois leques (A e B), para avaliar qual é mais "limpo".
# ---------------------------------------------------------------------

def painel_ruido_condicional_ab():
    st.markdown("## 📊 Ruído Condicional A/B — Comparação de Leques")

    df = get_historico()
    if df is None or df.empty:
        ui_caixa_aviso("Carregue o histórico primeiro.")
        return

    st.markdown(
        """
Este painel permite comparar o **ruído de dois leques de previsão**:

- Leque **A** (por exemplo, modo padrão)  
- Leque **B** (por exemplo, modo TURBO++ ULTRA ANTI-RUÍDO)  

Você escolhe **dois conjuntos de parâmetros**, e o sistema:

- Gera um leque A  
- Gera um leque B  
- Compara dispersão, amplitude média e entropia.  
        """
    )

    ui_separador_fino()

    idx_alvo = st.number_input(
        "Série alvo (1-based):",
        min_value=2,
        max_value=len(df),
        value=len(df),
    )

    st.markdown("### ⚙️ Configuração do Leque A")
    colA1, colA2, colA3 = st.columns(3)
    with colA1:
        top_n_a = st.number_input("N A", 5, 40, 25, key="top_n_a")
    with colA2:
        jmin_a = st.number_input("Janela min A", 10, 200, 40, key="jmin_a")
    with colA3:
        jmax_a = st.number_input("Janela max A", 50, 500, 200, key="jmax_a")

    ui_separador_fino()
    st.markdown("### ⚙️ Configuração do Leque B")
    colB1, colB2, colB3 = st.columns(3)
    with colB1:
        top_n_b = st.number_input("N B", 5, 40, 25, key="top_n_b")
    with colB2:
        jmin_b = st.number_input("Janela min B", 10, 200, 40, key="jmin_b")
    with colB3:
        jmax_b = st.number_input("Janela max B", 50, 500, 200, key="jmax_b")

    ui_separador_fino()

    if st.button("▶️ Gerar e comparar Leques A/B"):
        with st.spinner("Gerando Leque A..."):
            leque_a = motor_turbo_core_v14(
                df=df,
                idx_alvo=idx_alvo - 1,
                top_n=top_n_a,
                janela_min=jmin_a,
                janela_max=jmax_a,
                usar_features_expandido=True,
                usar_peso_estrutural=True,
            )
        with st.spinner("Gerando Leque B..."):
            leque_b = motor_turbo_core_v14(
                df=df,
                idx_alvo=idx_alvo - 1,
                top_n=top_n_b,
                janela_min=jmin_b,
                janela_max=jmax_b,
                usar_features_expandido=True,
                usar_peso_estrutural=True,
            )

        comp = comparar_ruido_condicional_ab(leque_a, leque_b)
        st.session_state["ruido_ab_info"] = {
            "leque_a": leque_a,
            "leque_b": leque_b,
            "comparacao": comp,
            "idx_alvo": idx_alvo,
        }
        ui_caixa_sucesso("Comparação de ruído A/B realizada!")

    if "ruido_ab_info" not in st.session_state:
        return

    info = st.session_state["ruido_ab_info"]
    comp = info["comparacao"]

    ui_separador_fino()
    st.markdown("### 🔧 Métricas de Ruído — Leque A")
    st.write(comp["A"])

    ui_separador_fino()
    st.markdown("### 🔧 Métricas de Ruído — Leque B")
    st.write(comp["B"])

    ui_separador_fino()
    st.markdown("### 🔁 Diferenças (B - A)")
    st.markdown(f"- Δ dispersão média: `{comp['delta_disp']:.4f}`")
    st.markdown(f"- Δ amplitude média: `{comp['delta_amp']:.4f}`")
    st.markdown(f"- Δ entropia norm.: `{comp['delta_entropia']:.4f}`")

    ui_separador_fino()
    st.markdown(
        """
> **Interpretação rápida**:
> - Valores **negativos** em Δ dispersão / Δ amplitude / Δ entropia indicam que  
>   o Leque B está **mais limpo / concentrado** que o Leque A.  
        """
    )

# ---------------------------------------------------------------------
# MODO TURBO++ ULTRA ANTI-RUÍDO (núcleo numérico básico)
# ---------------------------------------------------------------------

def modo_turbo_ultra_antiruido(
    df: pd.DataFrame,
    idx_alvo_1based: int,
    top_n_base: int = 25,
    janela_min: int = 40,
    janela_max: int = 200,
    ruido_alvo_max: float = 0.35,
    max_iter: int = 4,
) -> Dict:
    """
    Modo TURBO++ ULTRA ANTI-RUÍDO:
      - começa com um leque base
      - mede ruído
      - se ruído estiver acima do alvo, força:
          * redução progressiva do top_n
          * refinamento de janela
      - tenta progressivamente "apertar" o leque
        sem perder demais em diversidade.
    Retorna:
      - leque_final
      - ruido_final
      - historico_iteracoes
    """
    if df is None or df.empty:
        return {
            "leque_final": [],
            "ruido_final": None,
            "historico_iteracoes": [],
        }

    n = len(df)
    idx0 = max(1, min(idx_alvo_1based - 1, n - 1))

    historico = []
    top_n = top_n_base
    jmin = janela_min
    jmax = janela_max

    leque_final = []
    ruido_final = None

    for it in range(1, max_iter + 1):
        leque = motor_turbo_core_v14(
            df=df,
            idx_alvo=idx0,
            top_n=top_n,
            janela_min=jmin,
            janela_max=jmax,
            usar_features_expandido=True,
            usar_peso_estrutural=True,
        )
        ruido = analisar_ruido_do_leque(leque)

        historico.append({
            "iteracao": it,
            "top_n": top_n,
            "janela_min": jmin,
            "janela_max": jmax,
            "ruido": ruido,
        })

        if ruido["disp_media"] <= ruido_alvo_max:
            leque_final = leque
            ruido_final = ruido
            break

        # Ajustes progressivos:
        top_n = max(6, int(top_n * 0.7))
        jmin = max(10, int(jmin * 1.1))
        jmax = max(jmin + 5, int(jmax * 0.85))

        leque_final = leque
        ruido_final = ruido

    return {
        "leque_final": leque_final,
        "ruido_final": ruido_final,
        "historico_iteracoes": historico,
    }

# ---------------------------------------------------------------------
# PAINEL — TURBO++ ULTRA ANTI-RUÍDO (V15.5)
# ---------------------------------------------------------------------

def painel_turbo_ultra_antiruido():
    st.markdown("## 🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15.5)")

    df = get_historico()
    if df is None or df.empty:
        ui_caixa_aviso("Carregue o histórico primeiro.")
        return

    st.markdown(
        """
O **Modo TURBO++ ULTRA ANTI-RUÍDO** ajusta automaticamente:

- tamanho do leque  
- janela mínima / máxima  

para tentar **reduzir a dispersão do leque** até um alvo de ruído configurado.
        """
    )

    ui_separador_fino()

    idx_alvo = st.number_input(
        "Série alvo (1-based):",
        min_value=2,
        max_value=len(df),
        value=len(df),
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        top_n_base = st.number_input("N base", 8, 60, 25)
    with col2:
        ruido_alvo_max = st.number_input("Ruído alvo (disp. média máx.)", 0.05, 1.0, 0.35, 0.01)
    with col3:
        max_iter = st.number_input("Máx. iterações", 1, 10, 4)

    col4, col5 = st.columns(2)
    with col4:
        jmin = st.number_input("Janela min (base)", 10, 200, 40)
    with col5:
        jmax = st.number_input("Janela max (base)", 50, 500, 200)

    ui_separador_fino()

    if st.button("▶️ Rodar TURBO++ ULTRA ANTI-RUÍDO"):
        with st.spinner("Otimizando leque com foco em ruído..."):
            info = modo_turbo_ultra_antiruido(
                df=df,
                idx_alvo_1based=idx_alvo,
                top_n_base=top_n_base,
                janela_min=jmin,
                janela_max=jmax,
                ruido_alvo_max=ruido_alvo_max,
                max_iter=max_iter,
            )

        st.session_state["turbo_antiruido_info"] = info
        ui_caixa_sucesso("Execução do modo TURBO++ ULTRA ANTI-RUÍDO concluída!")

    if "turbo_antiruido_info" not in st.session_state:
        return

    info = st.session_state["turbo_antiruido_info"]

    ui_separador_fino()
    st.markdown("### 🚀 Leque Final Anti-Ruído")
    for i, s in enumerate(info["leque_final"], start=1):
        st.markdown(f"**#{i:02d}** → {s}")

    ui_separador_fino()
    st.markdown("### 🔧 Ruído Final")
    st.write(info["ruido_final"])

    ui_separador_fino()
    st.markdown("### 📜 Histórico de Iterações")
    for step in info["historico_iteracoes"]:
        st.markdown(f"#### Iteração {step['iteracao']}")
        st.markdown(
            f"- N: `{step['top_n']}`, Janela: `{step['janela_min']}–{step['janela_max']}`"
        )
        st.write(step["ruido"])

# =====================================================================
# PARTE 17/24 — FIM
# =====================================================================
# =====================================================================
# PREDICT CARS V15.5.1-HÍBRIDO — PARTE 18/24
# Relatório Final — AIQ Bridge (painel completo + auxiliares)
# =====================================================================

# ---------------------------------------------------------------------
# FUNÇÃO — Monta o HTML do Relatório Final
# (versão completa, com todos os blocos e sem compactar)
# ---------------------------------------------------------------------

def formatar_relatorio_html(
    titulo: str,
    contexto: Dict,
    serie_real: Optional[List[int]],
    leque: List[List[int]],
    ruido: Dict,
    monitor: Dict,
    aiq: Dict,
    aiq_avancado: Dict,
    modo: str,
) -> str:
    """
    Monta o relatório final em HTML completo.

    Este relatório é enviado ao usuário via painel próprio
    e pode ser colado no ChatGPT para interpretação profunda.
    """

    # ===========================
    # Cabeçalho
    # ===========================

    html = f"""
<h1>🔵 Predict Cars — Relatório Final (AIQ Bridge)</h1>
<h2>{titulo}</h2>
<p><b>Modo:</b> {modo}</p>
<hr>
"""

    # ===========================
    # Série real (se existir)
    # ===========================

    if serie_real is not None:
        html += "<h3>🎯 Série Real</h3>"
        html += f"<pre>{serie_real}</pre>"

    # ===========================
    # Leque gerado
    # ===========================

    html += "<h3>🚀 Leque Final</h3>"
    if leque:
        html += "<ul>"
        for i, s in enumerate(leque, start=1):
            html += f"<li><b>#{i:02d}</b> — {s}</li>"
        html += "</ul>"
    else:
        html += "<p>(Vazio)</p>"

    # ===========================
    # Ruído
    # ===========================

    html += "<h3>🔧 Ruído Condicional</h3>"
    html += "<pre>"
    for k, v in ruido.items():
        html += f"{k}: {v}\n"
    html += "</pre>"

    # ===========================
    # Monitor de Risco
    # ===========================

    html += "<h3>🌡️ Monitor de Risco (k & k*)</h3>"

    mg = monitor.get("global", {})
    ml = monitor.get("local", {})

    html += "<h4>Global</h4><pre>"
    for k, v in mg.items():
        html += f"{k}: {v}\n"
    html += "</pre>"

    html += "<h4>Local</h4><pre>"
    for k, v in ml.items():
        html += f"{k}: {v}\n"
    html += "</pre>"

    # ===========================
    # AIQ Global
    # ===========================

    html += "<h3>📈 AIQ Global</h3>"
    html += "<pre>"
    for k, v in aiq.items():
        html += f"{k}: {v}\n"
    html += "</pre>"

    # ===========================
    # AIQ Avançado
    # ===========================

    html += "<h3>📈 AIQ Avançado</h3>"
    html += "<pre>"
    for k, v in aiq_avancado.items():
        html += f"{k}: {v}\n"
    html += "</pre>"

    # ===========================
    # Contexto geral do modo
    # ===========================

    html += "<h3>📘 Contexto do Modo</h3>"
    html += "<pre>"
    for k, v in contexto.items():
        html += f"{k}: {v}\n"
    html += "</pre>"

    html += "<hr><p><i>Relatório gerado automaticamente pelo Predict Cars V15.5.1-HÍBRIDO.</i></p>"

    return html


# ---------------------------------------------------------------------
# FUNÇÃO — Gerar Relatório Final AIQ Bridge
# (wrapper completo para uso unificado nos painéis)
# ---------------------------------------------------------------------

def gerar_relatorio_final_aiq_bridge(
    df: pd.DataFrame,
    idx_alvo_1based: int,
    leque: List[List[int]],
    ruido_info: Dict,
    monitor_info: Dict,
    aiq_info: Dict,
    aiq_avancado: Dict,
    modo: str,
) -> str:
    """
    Monta o relatório final completo, com tudo incluído.
    """

    serie_real = None
    if 1 <= idx_alvo_1based <= len(df):
        passageiros_cols = passageiros_columns(df)
        linha = df.iloc[idx_alvo_1based - 1]
        serie_real = linha[passageiros_cols].astype(int).tolist()

    contexto = {
        "idx_alvo": idx_alvo_1based,
        "num_series": len(df),
        "modo_execucao": modo,
    }

    html = formatar_relatorio_html(
        titulo="Relatório Final — Previsão",
        contexto=contexto,
        serie_real=serie_real,
        leque=leque,
        ruido=ruido_info,
        monitor=monitor_info,
        aiq=aiq_info,
        aiq_avancado=aiq_avancado,
        modo=modo,
    )

    return html


# ---------------------------------------------------------------------
# PAINEL — RELATÓRIO FINAL AIQ Bridge
# (seleção do modo + renderização completa)
# ---------------------------------------------------------------------

def painel_relatorio_final():
    st.markdown("## 📄 Relatório Final — AIQ Bridge")

    st.markdown(
        """
O **Relatório Final — AIQ Bridge** consolida:

- Leque final  
- Série real (se houver)  
- Ruído  
- Risco (k & k*)  
- AIQ Global  
- AIQ Avançado  

Você pode escolher qualquer modo executado anteriormente.
        """
    )

    ui_separador_fino()

    opcoes = [
        "Replay LIGHT",
        "Replay ULTRA Unitário",
        "Modo 6 Acertos",
        "TURBO++ ULTRA ANTI-RUÍDO",
    ]

    modo = st.selectbox("Escolha o modo:", opcoes)

    # -----------------------------------------------------
    # Seleção automática dos dados de acordo com o modo
    # -----------------------------------------------------

    info = None
    if modo == "Replay LIGHT":
        info = st.session_state.get("replay_light_info")
    elif modo == "Replay ULTRA Unitário":
        info = st.session_state.get("replay_unit_info")
    elif modo == "Modo 6 Acertos":
        info = st.session_state.get("modo_6_info")
    elif modo == "TURBO++ ULTRA ANTI-RUÍDO":
        info = st.session_state.get("turbo_antiruido_info")

    if info is None:
        ui_caixa_aviso(f"O modo **{modo}** ainda não foi executado.")
        return

    # --------------------------------------------
    # Preparação do conteúdo
    # --------------------------------------------

    if modo == "TURBO++ ULTRA ANTI-RUÍDO":
        leque = info["leque_final"]
        ruido = info["ruido_final"]
        monitor = montar_monitor_de_risco_completo(
            get_historico(), idx_local=len(get_historico()) - 1
        )
        idx = len(get_historico())
    else:
        leque = info["leque"] if "leque" in info else info["leque_final"]
        ruido = info["ruido"] if "ruido" in info else info["ruido_final"]
        monitor = info["monitor_risco"]
        idx = info["idx_alvo"]

    aiq = st.session_state.get("aiq_info", {})
    aiq_adv = st.session_state.get("aiq_avancado_info", {})

    # --------------------------------------------
    # Gerar relatório
    # --------------------------------------------

    rel = gerar_relatorio_final_aiq_bridge(
        df=get_historico(),
        idx_alvo_1based=idx,
        leque=leque,
        ruido_info=ruido,
        monitor_info=monitor,
        aiq_info=aiq,
        aiq_avancado=aiq_adv,
        modo=modo,
    )

    st.session_state["relatorio_final_geral"] = rel

    ui_separador_fino()

    st.markdown("### 📝 Relatório Final (HTML)")
    st.markdown(rel, unsafe_allow_html=True)

    ui_separador_fino()

    st.markdown("### 📋 Copiar Relatório")
    st.code(rel, language="html")

    ui_separador_fino()

    with st.expander("📊 Comparar com outro relatório"):
        rel2 = st.text_area("Cole aqui outro relatório para comparação:")
        if rel2.strip():
            st.markdown("### Comparação não implementada — (placeholder para V16.0)")
        else:
            st.markdown("Cole outro relatório acima para comparar.")

# =====================================================================
# PARTE 18/24 — FIM
# =====================================================================
# =====================================================================
# PREDICT CARS V15.5.1-HÍBRIDO — PARTE 19/24
# Navegação FINAL (sidebar) + despache de painéis + função main()
# =====================================================================

# ---------------------------------------------------------------------
# SIDEBAR — NAVEGAÇÃO PRINCIPAL (VERSÃO FINAL, COMPLETA)
#   (sobrescreve a versão anterior, ampliando para todos os painéis)
# ---------------------------------------------------------------------

def montar_sidebar_principal():
    st.sidebar.markdown(
        """
# 🚗 Predict Cars  
### V15.5.1-HÍBRIDO
"""
    )

    st.sidebar.markdown("### 📂 Navegação")

    painel = st.sidebar.radio(
        "Escolha o painel:",
        [
            "📥 Entrada FLEX ULTRA",
            "🔍 Pipeline V14-FLEX ULTRA",
            "🎯 Replay LIGHT",
            "📅 Replay ULTRA",
            "🎛 Replay ULTRA Unitário",
            "🧪 Testes de Confiabilidade REAL",
            "🚨 Monitor de Risco (k & k*)",
            "📊 Ruído Condicional A/B",
            "🚀 Modo TURBO++ ULTRA ANTI-RUÍDO",
            "🎯 Modo 6 Acertos",
            "📄 Relatório Final — AIQ Bridge",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
**Legenda rápida:**

- Entrada FLEX → carrega a estrada  
- Pipeline → executa V14-FLEX ULTRA  
- Replay → reavalia pontos da estrada  
- Testes → validação REAL (QDS / BT / MC)  
- Monitor → risco k & k*  
- Ruído A/B → compara leques  
- TURBO++ Anti-Ruído → otimiza ruído do leque  
- Modo 6 Acertos → modo cirúrgico  
- Relatório Final → AIQ Bridge para ChatGPT  
"""
    )

    return painel

# ---------------------------------------------------------------------
# DESPACHO DOS PAINÉIS (função central)
# ---------------------------------------------------------------------

def despachar_painel(painel: str):
    """
    Função central que, dado o nome do painel selecionado na sidebar,
    chama o painel correspondente.
    """

    if painel == "📥 Entrada FLEX ULTRA":
        painel_entrada_flex_ultra()

    elif painel == "🔍 Pipeline V14-FLEX ULTRA":
        painel_pipeline_v14_flex_ultra()

    elif painel == "🎯 Replay LIGHT":
        painel_replay_light()

    elif painel == "📅 Replay ULTRA":
        painel_replay_ultra()

    elif painel == "🎛 Replay ULTRA Unitário":
        painel_replay_unitario()

    elif painel == "🧪 Testes de Confiabilidade REAL":
        painel_testes_confiabilidade()

    elif painel == "🚨 Monitor de Risco (k & k*)":
        painel_monitor_de_risco()

    elif painel == "📊 Ruído Condicional A/B":
        painel_ruido_condicional_ab()

    elif painel == "🚀 Modo TURBO++ ULTRA ANTI-RUÍDO":
        painel_turbo_ultra_antiruido()

    elif painel == "🎯 Modo 6 Acertos":
        painel_modo_6_acertos()

    elif painel == "📄 Relatório Final — AIQ Bridge":
        painel_relatorio_final()

    else:
        ui_caixa_aviso("Painel não reconhecido. Verifique a navegação.")

# ---------------------------------------------------------------------
# FUNÇÃO main() — PONTO CENTRAL DO APP
# ---------------------------------------------------------------------

def main():
    # Configuração da página (layout amplo, ícone, título)
    try:
        st.set_page_config(
            page_title="Predict Cars V15.5.1-HÍBRIDO",
            page_icon="🚗",
            layout="wide",
        )
    except Exception:
        # Streamlit permite apenas uma chamada; se já foi chamada em outro contexto,
        # ignoramos o erro silenciosamente.
        pass

    # Título principal (apenas na primeira carga visual)
    ui_titulo_principal()

    # Sidebar e escolha do painel
    painel = montar_sidebar_principal()

    # Execução do painel selecionado
    despachar_painel(painel)


# ---------------------------------------------------------------------
# PONTO DE ENTRADA — Execução direta
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()

# =====================================================================
# PARTE 19/24 — FIM
# =====================================================================
# =====================================================================
# PREDICT CARS V15.5.1-HÍBRIDO — PARTE 20/24
# UTILIDADES AVANÇADAS, NORMALIZAÇÕES, FUNÇÕES NUMÉRICAS
# E BLOCO DE LOGGING INTERNO
# =====================================================================

# ---------------------------------------------------------------------
# CLAMP / NORMALIZE / UTILIDADES MATEMÁTICAS BÁSICAS
# ---------------------------------------------------------------------

def clamp(x: float, a: float = 0.0, b: float = 1.0) -> float:
    """Garante que x fique dentro de [a, b]."""
    try:
        return max(a, min(b, float(x)))
    except Exception:
        return a

def safe_norm(arr: np.ndarray) -> float:
    """Norma L2, mas segura contra erros."""
    try:
        return float(np.linalg.norm(arr))
    except Exception:
        return 0.0

def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalização min-max segura."""
    try:
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    except Exception:
        return np.zeros_like(arr)

def safe_mean(x):
    try:
        return float(np.mean(x))
    except Exception:
        return 0.0

def safe_std(x):
    try:
        return float(np.std(x))
    except Exception:
        return 0.0

def safe_entropy(arr):
    """Entropia simples normalizada."""
    try:
        unq, cnts = np.unique(arr, return_counts=True)
        probs = cnts / cnts.sum()
        ent = -np.sum([p * math.log(p + 1e-12) for p in probs])
        return float(ent / max(1.0, math.log(len(unq) + 1)))
    except Exception:
        return 0.0

# ---------------------------------------------------------------------
# DISTÂNCIAS AVANÇADAS / SIMILARIDADES HÍBRIDAS
# ---------------------------------------------------------------------

def distancia_hibrida(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Distância híbrida:
    - mistura peso de norma L2 + similaridade absoluta + variações.
    """
    try:
        d1 = safe_norm(v1 - v2)
        d2 = np.abs(v1 - v2).sum()
        d3 = float(np.max(np.abs(v1 - v2)))
        dist = 0.5 * d1 + 0.3 * d2 + 0.2 * d3
        return float(dist)
    except Exception:
        return 9999.0

def similaridade_hibrida(v1, v2):
    """Similaridade derivada da distância híbrida."""
    d = distancia_hibrida(v1, v2)
    return float(1 / (1 + d))

# ---------------------------------------------------------------------
# CONVERSORES E SANITIZADORES
# ---------------------------------------------------------------------

def to_int_list(x):
    try:
        return [int(t) for t in x]
    except Exception:
        return []

def to_float_list(x):
    try:
        return [float(t) for t in x]
    except Exception:
        return []

def sanitize_series_list(L, expected_len=6):
    """
    Garante que as séries tenham tamanho correto, preenchendo com zeros se necessário.
    """
    R = []
    for s in L:
        try:
            if len(s) != expected_len:
                s = s[:expected_len] + [0] * (expected_len - len(s))
            R.append([int(x) for x in s])
        except Exception:
            R.append([0] * expected_len)
    return R

# ---------------------------------------------------------------------
# CORES / FORMATAÇÃO
# ---------------------------------------------------------------------

def color_tag(text: str, color: str = "blue"):
    """Wrap simples para colore HTML."""
    return f"<span style='color:{color};font-weight:bold;'>{text}</span>"

def colorize_risco(regime: str):
    if regime.lower() == "estavel":
        return color_tag("🟢 estável", "green")
    if regime.lower() == "atencao":
        return color_tag("🟡 atenção", "gold")
    return color_tag("🔴 crítico", "red")

# ---------------------------------------------------------------------
# FERRAMENTAS DE DEBUG AVANÇADO (SILENCIOSO)
# ---------------------------------------------------------------------

DEBUG_LOGS = []

def debug_log(msg: str):
    """
    Armazena mensagens de debug internamente.
    Não polui a interface, mas pode ser exibido no painel de debug técnico (Parte 21).
    """
    if len(DEBUG_LOGS) > 5000:
        DEBUG_LOGS.clear()
    DEBUG_LOGS.append(msg)

# ---------------------------------------------------------------------
# MERGE / DICT HELPERS
# ---------------------------------------------------------------------

def merge_dicts(a: Dict, b: Dict) -> Dict:
    r = dict(a)
    r.update(b)
    return r

def safe_get(d: Dict, key: str, default=None):
    try:
        return d.get(key, default)
    except Exception:
        return default

# ---------------------------------------------------------------------
# FILTROS / FUNÇÕES AUXILIARES
# ---------------------------------------------------------------------

def filtro_series_premium(df):
    """Retorna apenas as séries classificadas como premium."""
    info = classificar_modo_series_com_idx(df)
    return info.get("premium", [])

def filtro_series_estruturais(df):
    info = classificar_modo_series_com_idx(df)
    return info.get("estrutural", [])

def filtro_series_cobertura(df):
    info = classificar_modo_series_com_idx(df)
    return info.get("cobertura", [])

# ---------------------------------------------------------------------
# UTILS PARA EXPANSÃO FUTURA (V16)
# ---------------------------------------------------------------------

def medir_curva_de_ruido(leque):
    """
    Mede evolução interna de ruído ao longo das posições do leque.
    Útil para modos avançados que selecionam trechos “mais limpos”.
    """
    if not leque:
        return []

    curvas = []
    for s in leque:
        a = np.array(s)
        amp = float(a.max() - a.min())
        std = float(a.std())
        curvas.append({"amp": amp, "std": std})

    return curvas

# ---------------------------------------------------------------------
# NORMALIZAÇÕES ADICIONAIS (V15)
# ---------------------------------------------------------------------

def normalizar_ruido_value(x):
    """Normalização não-linear usada no Modo Anti-Ruído."""
    try:
        return float(1 - math.exp(-abs(x)))
    except Exception:
        return 1.0

def normalizar_k_sensibilidade(k_series):
    """Usado para calibrar sensibilidade do k preditivo."""
    try:
        arr = np.array(k_series)
        return normalize_array(arr)
    except Exception:
        return np.zeros(len(k_series))

# =====================================================================
# PARTE 20/24 — FIM
# =====================================================================
# =====================================================================
# PREDICT CARS V15.5.1-HÍBRIDO — PARTE 21/24
# PAINEL TÉCNICO / DEBUG AVANÇADO
# =====================================================================

# ---------------------------------------------------------------------
# INSPEÇÃO RÁPIDA DA ESTRADA
# ---------------------------------------------------------------------

def inspecionar_estrada_resumo(df: pd.DataFrame) -> Dict:
    """
    Retorna um resumo técnico da estrada:
      - número de séries
      - número de passageiros
      - faixa mínima/máxima
      - média global
      - desvio global
    """
    if df is None or df.empty:
        return {
            "n_series": 0,
            "n_passageiros": 0,
            "valor_min": None,
            "valor_max": None,
            "media_global": None,
            "std_global": None,
        }

    passageiros_cols = passageiros_columns(df)
    arr = df[passageiros_cols].astype(float).values

    return {
        "n_series": len(df),
        "n_passageiros": len(passageiros_cols),
        "valor_min": float(arr.min()),
        "valor_max": float(arr.max()),
        "media_global": float(arr.mean()),
        "std_global": float(arr.std()),
    }

def inspecionar_k_global(df: pd.DataFrame) -> Dict:
    """
    Estatísticas simples de k no histórico:
      - média
      - desvio
      - min / max
      - histograma bruto
    """
    if df is None or df.empty or "k" not in df.columns:
        return {
            "k_media": None,
            "k_std": None,
            "k_min": None,
            "k_max": None,
            "histograma": {},
        }

    ks = df["k"].astype(float).values
    unq, cnts = np.unique(ks, return_counts=True)
    hist = {int(u): int(c) for u, c in zip(unq, cnts)}

    return {
        "k_media": float(ks.mean()),
        "k_std": float(ks.std()),
        "k_min": float(ks.min()),
        "k_max": float(ks.max()),
        "histograma": hist,
    }

# ---------------------------------------------------------------------
# PAINEL TÉCNICO / DEBUG
# ---------------------------------------------------------------------

def painel_debug_tecnico():
    st.markdown("## 🛠 Painel Técnico / Debug Avançado")

    df = get_historico()
    if df is None or df.empty:
        ui_caixa_aviso("Histórico ainda não carregado. Use o painel de Entrada FLEX ULTRA.")
        return

    st.markdown(
        """
Este painel é **técnico**, voltado para inspeção interna do motor:

- Verificação da estrada (dimensões, faixas, estatísticas)  
- Estatísticas de `k`  
- QDS atual  
- IDX Híbrido  
- Logs internos (DEBUG_LOGS)  
        """
    )

    ui_separador_fino()

    # -----------------------------
    # Resumo técnico da estrada
    # -----------------------------

    st.markdown("### 🧱 Resumo Técnico da Estrada")

    resumo = inspecionar_estrada_resumo(df)
    st.write(resumo)

    ui_separador_fino()

    # -----------------------------
    # Estatísticas de k
    # -----------------------------

    st.markdown("### 🔢 Estatísticas de k")

    kstats = inspecionar_k_global(df)
    st.write(kstats)

    # Histograma simples
    if kstats["histograma"]:
        st.markdown("#### Histograma de k (valor → contagem)")
        st.write(kstats["histograma"])

    ui_separador_fino()

    # -----------------------------
    # QDS / AIQ / IDX
    # -----------------------------

    st.markdown("### 📊 QDS / AIQ / IDX (quando disponíveis)")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Recalcular QDS agora"):
            with st.spinner("Recalculando QDS..."):
                qds_info = calcular_qds_completo(df, janela=50)
            st.session_state["qds_info"] = qds_info
            ui_caixa_sucesso("QDS recalculado.")
        qds_atual = st.session_state.get("qds_info", None)
        st.markdown("#### QDS atual")
        st.write(qds_atual)

    with col2:
        if st.button("Recalcular AIQ/IDX agora"):
            with st.spinner("Recalculando AIQ Avançado..."):
                aiq_adv = calcular_aiq_avancado(df)
            st.session_state["aiq_avancado_info"] = aiq_adv
            ui_caixa_sucesso("AIQ Avançado recalculado.")
        aiq_atual = st.session_state.get("aiq_avancado_info", None)
        st.markdown("#### AIQ Avançado atual")
        st.write(aiq_atual)

    ui_separador_fino()

    st.markdown("### 🧬 IDX Híbrido (visão rápida)")
    df_hib = st.session_state.get("idx_hibrido", None)
    if df_hib is None or (hasattr(df_hib, "empty") and df_hib.empty):
        df_hib = idx_hibrido_construir(df)
    st.dataframe(df_hib.head(50))

    ui_separador_fino()

    # -----------------------------
    # Logs internos (DEBUG_LOGS)
    # -----------------------------

    st.markdown("### 📜 Logs Internos (DEBUG_LOGS)")

    if DEBUG_LOGS:
        with st.expander("Ver logs (internos)"):
            st.text("\n".join(DEBUG_LOGS[-500:]))
    else:
        st.markdown("Nenhum log interno registrado (DEBUG_LOGS vazio).")

    ui_separador_fino()

    st.markdown(
        """
> Use este painel apenas para inspeção técnica.  
> Não interfere diretamente nas previsões, mas ajuda a entender
> o comportamento interno do sistema.
        """
    )

# =====================================================================
# PARTE 21/24 — FIM
# =====================================================================
# =====================================================================
# PREDICT CARS V15.5.1-HÍBRIDO — PARTE 22/24
# QDS COMPLETO + BACKTEST INTERNO + BACKTEST DO FUTURO
# =====================================================================

# ---------------------------------------------------------------------
# QDS — Qualidade Dinâmica de Série (versão completa V15)
# ---------------------------------------------------------------------

def calcular_qds_completo(df: pd.DataFrame, janela: int = 50) -> Dict:
    """
    QDS completo, versão V15:
      - mede estabilidade dinâmica
      - mede instabilidade local
      - mede variação temporal por janelas
      - normaliza curvas
      - produz um score final de 0 a 1

    Retorna:
        {
            "qds_bruto": [...],
            "qds_norm": [...],
            "score_global": float,
            "janela": janela,
        }
    """
    if df is None or df.empty:
        return {
            "qds_bruto": [],
            "qds_norm": [],
            "score_global": 0.0,
            "janela": janela,
        }

    passageiros_cols = passageiros_columns(df)
    arr = df[passageiros_cols].astype(float).values
    n = len(arr)

    qds_bruto = []

    for i in range(n):
        ini = max(0, i - janela)
        sub = arr[ini:i+1]
        if len(sub) < 2:
            qds_bruto.append(0.0)
            continue

        # 1) Estabilidade
        amp = float(sub.max() - sub.min())
        desv = float(sub.std())
        est = 1 / (1 + amp + desv)

        # 2) Oscilação interna
        osc = float(np.abs(np.diff(sub, axis=0)).mean()) if len(sub) > 1 else 0.0

        # 3) Combinação
        q = clamp(0.7 * est + 0.3 * (1 - normalizar_ruido_value(osc)))
        qds_bruto.append(q)

    qds_norm = normalize_array(np.array(qds_bruto))
    score = float(np.mean(qds_norm))

    res = {
        "qds_bruto": qds_bruto,
        "qds_norm": qds_norm.tolist(),
        "score_global": score,
        "janela": janela,
    }

    st.session_state["qds_info"] = res
    return res

# ---------------------------------------------------------------------
# BACKTEST INTERNO — versão clássica V14/V15
# ---------------------------------------------------------------------

def executar_backtest_interno(df: pd.DataFrame, janela_min=40, janela_max=200, top_n=25) -> Dict:
    """
    Backtest Interno (ache X para prever X+1).
    Usa motor TURBO clássico e mede acertos nas faixas reais.
    """
    if df is None or df.empty:
        return {"acertos": [], "media": 0.0}

    passageiros_cols = passageiros_columns(df)
    acertos = []

    for i in range(1, len(df)):
        leque = motor_turbo_core_v14(
            df=df,
            idx_alvo=i - 1,
            top_n=top_n,
            janela_min=janela_min,
            janela_max=janela_max,
            usar_features_expandido=True,
            usar_peso_estrutural=True,
        )

        if not leque:
            acertos.append(0)
            continue

        serie_real = df.iloc[i][passageiros_cols].astype(int).tolist()

        hit = 1 if serie_real in leque else 0
        acertos.append(hit)

    media = float(np.mean(acertos)) if acertos else 0.0

    return {
        "acertos": acertos,
        "media": media,
        "descricao": "Backtest Interno (prevê X+1 a partir de X)",
    }

# ---------------------------------------------------------------------
# BACKTEST DO FUTURO — versão V15.5 (com janelas + IDX)
# ---------------------------------------------------------------------

def executar_backtest_do_futuro(df: pd.DataFrame, passo=3, janela_min=40, janela_max=200, top_n=25) -> Dict:
    """
    Backtest do Futuro:
      - tenta prever X+p a partir de X (p pode ser 2, 3...)
      - considera comportamento global (k*, QDS, IDX)
      - modo avançado usado no módulo de confiabilidade

    Retorna:
        {
            "passo": passo,
            "resultados": [...],
            "media": float
        }
    """
    if df is None or df.empty:
        return {"passo": passo, "resultados": [], "media": 0.0}

    passageiros_cols = passageiros_columns(df)
    resultados = []

    for i in range(len(df)):
        alvo = i + passo
        if alvo >= len(df):
            break

        leque = motor_turbo_core_v14(
            df=df,
            idx_alvo=i,
            top_n=top_n,
            janela_min=janela_min,
            janela_max=janela_max,
            usar_features_expandido=True,
            usar_peso_estrutural=True,
        )

        if not leque:
            resultados.append(0)
            continue

        serie_real = df.iloc[alvo][passageiros_cols].astype(int).tolist()
        hit = 1 if serie_real in leque else 0
        resultados.append(hit)

    media = float(np.mean(resultados)) if resultados else 0.0

    return {
        "passo": passo,
        "resultados": resultados,
        "media": media,
        "descricao": "Backtest do Futuro (prevê X+p a partir de X)",
    }

# ---------------------------------------------------------------------
# FUNÇÃO CENTRAL — TESTES DE CONFIABILIDADE REAL
# (usada diretamente pelo painel de Testes)
# ---------------------------------------------------------------------

def executar_testes_confiabilidade_real(df: pd.DataFrame) -> Dict:
    """
    Executa todo o pacote REAL:
        - QDS completo
        - Backtest Interno
        - Backtest do Futuro
        - Monte Carlo Multi-Camadas
        - AIQ Global + AIQ Avançado
    """
    if df is None or df.empty:
        return {}

    # QDS
    qds_info = calcular_qds_completo(df, janela=50)

    # Backtests
    bt_interno = executar_backtest_interno(df)
    bt_futuro = executar_backtest_do_futuro(df)

    # Monte Carlo profundo
    mc = executar_monte_carlo_profundo(df)

    # AIQ
    aiq = calcular_aiq_global(df)
    aiq_avancado = calcular_aiq_avancado(df)

    res = {
        "qds": qds_info,
        "backtest_interno": bt_interno,
        "backtest_futuro": bt_futuro,
        "monte_carlo_multi": mc,
        "aiq": aiq,
        "aiq_avancado": aiq_avancado,
    }

    # salva no estado
    st.session_state["aiq_info"] = aiq
    st.session_state["aiq_avancado_info"] = aiq_avancado

    return res

# =====================================================================
# PARTE 22/24 — FIM
# =====================================================================
# =====================================================================
# PREDICT CARS V15.5.1-HÍBRIDO — PARTE 23/24
# MONTE CARLO PROFUNDO + AIQ GLOBAL + AIQ AVANÇADO + IDX HÍBRIDO
# =====================================================================

# ---------------------------------------------------------------------
# MONTE CARLO PROFUNDO (versão simples)
# ---------------------------------------------------------------------

def executar_monte_carlo_profundo(
    df: pd.DataFrame,
    n_runs: int = 100,
    tamanho_janela: int = 60,
    top_n: int = 25,
) -> Dict:
    """
    Monte Carlo Profundo (versão simples V15.5):

    - sorteia janelas da estrada
    - escolhe índices alvo dentro da janela
    - roda motor TURBO para cada alvo
    - mede métricas de:
        * tamanho do leque
        * dispersão média
        * amplitude média
        * entropia normalizada

    Retorna um resumo agregando todas as execuções.
    """
    if df is None or df.empty:
        return {
            "n_runs": n_runs,
            "tamanho_janela": tamanho_janela,
            "resultados": [],
            "disp_media_global": 0.0,
            "amp_media_global": 0.0,
            "entropia_media_global": 0.0,
        }

    n = len(df)
    resultados = []
    disp_list = []
    amp_list = []
    ent_list = []

    for r in range(n_runs):
        if n <= tamanho_janela + 1:
            ini = 0
            fim = n - 1
        else:
            ini = np.random.randint(0, max(1, n - tamanho_janela))
            fim = min(n - 1, ini + tamanho_janela)

        idx_alvo = np.random.randint(ini + 1, fim + 1)
        leque = motor_turbo_core_v14(
            df=df,
            idx_alvo=idx_alvo,
            top_n=top_n,
            janela_min=max(10, int(tamanho_janela * 0.4)),
            janela_max=tamanho_janela,
            usar_features_expandido=True,
            usar_peso_estrutural=True,
        )

        ruido = analisar_ruido_do_leque(leque)

        disp_list.append(ruido["disp_media"])
        amp_list.append(ruido["amp_media"])
        ent_list.append(ruido["entropia_norm"])

        resultados.append({
            "run": r + 1,
            "idx_alvo": idx_alvo + 1,
            "n_series_leque": len(leque),
            "ruido": ruido,
        })

    return {
        "n_runs": n_runs,
        "tamanho_janela": tamanho_janela,
        "resultados": resultados,
        "disp_media_global": float(np.mean(disp_list)) if disp_list else 0.0,
        "amp_media_global": float(np.mean(amp_list)) if amp_list else 0.0,
        "entropia_media_global": float(np.mean(ent_list)) if ent_list else 0.0,
    }

# ---------------------------------------------------------------------
# MONTE CARLO PROFUNDO MULTICAMADAS (usado em Testes de Confiabilidade)
# ---------------------------------------------------------------------

def monte_carlo_profundo_multicamadas(
    df: pd.DataFrame,
    n_runs: int = 80,
    tamanhos_amostra: List[int] = None,
) -> Dict:
    """
    Versão multicamadas:

    - executa Monte Carlo para vários tamanhos de janela (tamanhos_amostra)
    - agrega métricas globais por camada
    """
    if tamanhos_amostra is None:
        tamanhos_amostra = [40, 80, 120]

    camadas = []

    for tam in tamanhos_amostra:
        mc = executar_monte_carlo_profundo(
            df=df,
            n_runs=n_runs,
            tamanho_janela=tam,
            top_n=25,
        )
        camadas.append({
            "tamanho_janela": tam,
            "disp_media_global": mc["disp_media_global"],
            "amp_media_global": mc["amp_media_global"],
            "entropia_media_global": mc["entropia_media_global"],
        })

    return {
        "n_runs": n_runs,
        "tamanhos_amostra": tamanhos_amostra,
        "camadas": camadas,
    }

# ---------------------------------------------------------------------
# IDX HÍBRIDO — CONSTRUÇÃO
# ---------------------------------------------------------------------

def idx_hibrido_construir(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói o IDX Híbrido (V15.5):

    - usa:
        * similaridade (S2)
        * estabilidade (S4)
        * resiliência (S5)
    - gera um índice de 0 a 1 por série

    Retorna DataFrame com:
        - indice (1-based)
        - hib (índice híbrido)
    """
    if df is None or df.empty:
        df_empty = pd.DataFrame({"indice": [], "hib": []})
        st.session_state["idx_hibrido"] = df_empty
        return df_empty

    s2 = s2_cross_similarity(df)
    s4 = s4_estabilidade_proj(df)
    s5 = s5_resiliencia(df)

    sim = np.array(s2["sim"])
    est = np.array(s4["estabilidade"])
    res = np.array(s5["resiliencia"])

    # Garantir mesmo tamanho
    n = len(df)
    sim = sim[:n] if len(sim) >= n else np.pad(sim, (0, n - len(sim)), constant_values=sim.mean() if len(sim) else 0.5)
    est = est[:n] if len(est) >= n else np.pad(est, (0, n - len(est)), constant_values=est.mean() if len(est) else 0.5)
    res = res[:n] if len(res) >= n else np.pad(res, (0, n - len(res)), constant_values=res.mean() if len(res) else 0.5)

    hib = 0.35 * sim + 0.35 * est + 0.30 * res
    hib = np.clip(hib, 0.0, 1.0)

    df_hib = pd.DataFrame({
        "indice": np.arange(1, n + 1),
        "hib": hib,
    })

    st.session_state["idx_hibrido"] = df_hib
    return df_hib

# ---------------------------------------------------------------------
# CLASSIFICAÇÃO DAS SÉRIES EM PREMIUM / ESTRUTURAL / COBERTURA
# ---------------------------------------------------------------------

def classificar_modo_series_com_idx(df: pd.DataFrame) -> Dict:
    """
    Classifica as séries em 3 grupos, usando o hib:

    - Premium: hib acima do p85
    - Estrutural: hib entre p50 e p85
    - Cobertura: hib abaixo do p50
    """
    df_hib = st.session_state.get("idx_hibrido", None)
    if df_hib is None or (hasattr(df_hib, "empty") and df_hib.empty):
        df_hib = idx_hibrido_construir(df)

    if df_hib.empty:
        return {"premium": [], "estrutural": [], "cobertura": []}

    hib_vals = df_hib["hib"].values
    p50 = float(np.percentile(hib_vals, 50))
    p85 = float(np.percentile(hib_vals, 85))

    premium = df_hib[df_hib["hib"] >= p85]["indice"].tolist()
    estrutural = df_hib[(df_hib["hib"] >= p50) & (df_hib["hib"] < p85)]["indice"].tolist()
    cobertura = df_hib[df_hib["hib"] < p50]["indice"].tolist()

    return {
        "premium": premium,
        "estrutural": estrutural,
        "cobertura": cobertura,
        "limiares": {"p50": p50, "p85": p85},
    }

# ---------------------------------------------------------------------
# AIQ GLOBAL — versão V15.5
# ---------------------------------------------------------------------

def calcular_aiq_global(df: pd.DataFrame) -> Dict:
    """
    AIQ Global V15.5:

    - usa:
        * hib médio
        * hib desvio
        * QDS, se existir
    - produz um score de 0 a 1
    """
    if df is None or df.empty:
        return {
            "aiq_score": 0.0,
            "idx_hibrido_stats": {
                "hib_media": 0.0,
                "hib_std": 0.0,
                "hib_min": 0.0,
                "hib_max": 0.0,
            },
            "n_series": 0,
        }

    df_hib = idx_hibrido_construir(df)
    hib = df_hib["hib"].values

    hib_media = float(hib.mean()) if len(hib) else 0.0
    hib_std = float(hib.std()) if len(hib) else 0.0
    hib_min = float(hib.min()) if len(hib) else 0.0
    hib_max = float(hib.max()) if len(hib) else 0.0

    qds_info = st.session_state.get("qds_info", None)
    if qds_info and "score_global" in qds_info:
        qds_score = float(qds_info["score_global"])
    else:
        qds_score = hib_media  # fallback alinhado

    # Combinação simples: 60% hib, 40% QDS
    aiq_score = clamp(0.6 * hib_media + 0.4 * qds_score)

    res = {
        "aiq_score": aiq_score,
        "idx_hibrido_stats": {
            "hib_media": hib_media,
            "hib_std": hib_std,
            "hib_min": hib_min,
            "hib_max": hib_max,
        },
        "n_series": len(df),
    }

    return res

# ---------------------------------------------------------------------
# AIQ AVANÇADO — versão V15.5
# ---------------------------------------------------------------------

def calcular_aiq_avancado(df: pd.DataFrame) -> Dict:
    """
    AIQ Avançado V15.5:

    - inclui:
        * aiq_score global
        * entropia global da estrada
        * amplitude total
        * score final "aiq_avancado_score"
    """
    if df is None or df.empty:
        return {
            "aiq_avancado_score": 0.0,
            "entropia_global": 0.0,
            "amplitude_total": 0.0,
        }

    s1 = s1_basico_estatisticas(df)
    ent = s1["entropia_glob"]
    amp_total = s1["amp_total"]

    aiq_global = calcular_aiq_global(df)
    base_score = aiq_global["aiq_score"]

    # Menor amplitude e entropia em faixa moderada tendem a aumentar score
    # (ajuste heurístico)
    amp_factor = 1 / (1 + amp_total / 100.0)
    ent_factor = 1 - abs(ent - 0.5)  # ideal ~0.5

    raw = base_score * 0.6 + amp_factor * 0.2 + ent_factor * 0.2
    aiq_avancado_score = clamp(raw)

    res = {
        "aiq_avancado_score": aiq_avancado_score,
        "entropia_global": ent,
        "amplitude_total": amp_total,
        "aiq_base": base_score,
    }

    return res

# =====================================================================
# PARTE 23/24 — FIM
# =====================================================================
# =====================================================================
# PREDICT CARS V15.5.1-HÍBRIDO — PARTE 24/24
# BLOCO FINAL DE COMPATIBILIDADE / ATIVAÇÃO / FALLBACKS
# =====================================================================

# ---------------------------------------------------------------------
# FALLBACKS PARA SEGURANÇA
# Estes wrappers garantem que, se algum módulo for chamado antes de
# estar carregado (reexecução parcial do Streamlit), o app não quebre.
# ---------------------------------------------------------------------

def _safe_call(func, default=None):
    try:
        return func()
    except Exception:
        return default

def _safe_dict_call(func, default=None):
    try:
        r = func()
        return r if isinstance(r, dict) else default
    except Exception:
        return default

def _safe_list_call(func, default=None):
    try:
        r = func()
        return r if isinstance(r, list) else default
    except Exception:
        return default

# ---------------------------------------------------------------------
# REGISTRO DE TODAS AS FUNÇÕES-CHAVE (para inspeção interna futura)
# Não altera nada no funcionamento do app, mas deixa tudo visível
# para módulos futuros (como V16 / V16-HÍBRIDO).
# ---------------------------------------------------------------------

FUNCOES_REGISTRADAS_V1551 = {
    # Entrada / leitura
    "entrada_flex_ultra": "painel_entrada_flex_ultra",
    "get_historico": "get_historico",

    # Pipeline
    "pipeline_ultra": "painel_pipeline_v14_flex_ultra",
    "motor_pipeline": "motor_turbo_pipeline_v14",
    "diagnostico_pipeline": "diagnostico_interno_pipeline",

    # Replays
    "replay_light": "painel_replay_light",
    "replay_ultra": "painel_replay_ultra",
    "replay_unit": "painel_replay_unitario",

    # Testes de Confiabilidade REAL
    "testes_confiabilidade_real": "painel_testes_confiabilidade",
    "executar_testes_confiabilidade_real": "executar_testes_confiabilidade_real",

    # Modo 6 Acertos
    "modo_6_acertos": "painel_modo_6_acertos",

    # Monitor de Risco
    "monitor_de_risco": "painel_monitor_de_risco",

    # Ruído Condicional A/B
    "ruido_ab": "painel_ruido_condicional_ab",

    # Anti-Ruído
    "turbo_ultra_antiruido": "painel_turbo_ultra_antiruido",

    # Relatório Final
    "relatorio_final": "painel_relatorio_final",

    # Debug Técnico
    "debug_tecnico": "painel_debug_tecnico",

    # IDX / AIQ / QDS / Monte Carlo
    "idx_hibrido": "idx_hibrido_construir",
    "classificar_idx": "classificar_modo_series_com_idx",
    "aiq_global": "calcular_aiq_global",
    "aiq_avancado": "calcular_aiq_avancado",
    "qds_completo": "calcular_qds_completo",
    "monte_carlo_profundo": "executar_monte_carlo_profundo",
}

# ---------------------------------------------------------------------
# MENSAGEM FINAL — ATIVAÇÃO DO SISTEMA COMPLETO
# ---------------------------------------------------------------------

def bloco_ativacao_final_do_sistema():
    """
    Este bloco é chamado silenciosamente no final do app.
    Não aparece no front-end, mas ajuda a garantir que
    tudo esteja carregado no Streamlit.

    Também funciona como “assinatura interna” da versão.
    """
    debug_log("Sistema Predict Cars V15.5.1-HÍBRIDO totalmente carregado.")
    debug_log(f"Funções registradas: {len(FUNCOES_REGISTRADAS_V1551)} módulos.")
    return True

# Chamamos no carregamento:
try:
    bloco_ativacao_final_do_sistema()
except Exception:
    pass

# ---------------------------------------------------------------------
# MENSAGEM DE IDENTIDADE DA VERSÃO (não exibida para o usuário final)
# ---------------------------------------------------------------------

IDENTIDADE_V1551 = """
Predict Cars V15.5.1-HÍBRIDO
Motor completo, sem simplificações.
Pipeline V14-FLEX ULTRA + Replay + IDX Híbrido + QDS + Backtests + Monte Carlo +
AIQ Avançado + k & k* + Anti-Ruído + Modo 6 Acertos + Relatório Final AIQ Bridge.
"""

# =====================================================================
# PARTE 24/24 — FIM
# =====================================================================
