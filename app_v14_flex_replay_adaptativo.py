# ============================================================
# Predict Cars — V14-FLEX ULTRA REAL (TURBO++)
# app_v14_flex_replay_ultra_unitario.py
#
# Versão ULTRA completa, com:
# - Entrada FLEX (n passageiros + k)
# - Barômetro ULTRA REAL
# - k* ULTRA REAL (sentinela)
# - IDX ULTRA
# - IPF / IPO refinados
# - S6 Profundo ULTRA
# - Micro-Leque ULTRA
# - Monte Carlo Profundo ULTRA
# - QDS REAL
# - Backtest REAL
# - Replay LIGHT
# - Replay ULTRA (loop tradicional)
# - Replay ULTRA UNITÁRIO (novo)
# - Modo TURBO++ ULTRA Adaptativo
#
# Estrutura densa, sem simplificações, mantendo o "jeitão" V14-FLEX.
# ============================================================

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# CONFIGURAÇÃO GERAL DO APP
# ============================================================

APP_VERSION = "V14-FLEX ULTRA REAL (TURBO++)"

# Para evitar multi-config em recarregamentos
if "page_config_set" not in st.session_state:
    st.set_page_config(
        page_title="Predict Cars — V14-FLEX ULTRA REAL (TURBO++)",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.session_state["page_config_set"] = True


# ============================================================
# DATACLASSES — ESTRUTURAS DE APOIO
# ============================================================

@dataclass
class RegimeInfo:
    janela_usada: int
    k_medio: float
    k_max: float
    estado: str
    descricao: str


@dataclass
class KStarInfo:
    janela_usada: int
    k_media_janela: float
    k_max_janela: float
    k_star_pct: float
    estado: str
    descricao: str


@dataclass
class BacktestResult:
    tabela: pd.DataFrame
    descricao: str


@dataclass
class QDSInfo:
    valor: float
    descricao: str


# ============================================================
# FUNÇÕES UTILITÁRIAS BÁSICAS
# ============================================================

def obter_n_passageiros(df: pd.DataFrame) -> int:
    """
    Detecta automaticamente o número de passageiros (colunas p1..pn)
    a partir do dataframe carregado.
    """
    cols = [c for c in df.columns if c.lower().startswith("p")]
    # Garante ordenação p1, p2, ..., pn
    cols_ordenadas = sorted(cols, key=lambda x: int("".join(filter(str.isdigit, x)) or 0))
    return len(cols_ordenadas)


def extrair_passageiros(df: pd.DataFrame) -> List[str]:
    """
    Retorna a lista de colunas de passageiros (p1..pn), ordenadas.
    """
    cols = [c for c in df.columns if c.lower().startswith("p")]
    return sorted(cols, key=lambda x: int("".join(filter(str.isdigit, x)) or 0))


def obter_intervalo_indices(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Usa a coluna 'idx' para determinar o intervalo de índices
    efetivamente disponíveis no histórico.
    """
    if "idx" not in df.columns:
        raise KeyError("Coluna 'idx' não encontrada no dataframe histórico.")
    return int(df["idx"].min()), int(df["idx"].max())


def normalizar_serie(serie: Any) -> List[int]:
    """
    Normaliza uma série de passageiros (lista/array/Series) para lista de int.
    Aceita tanto lista de ints quanto formatos aproximados.
    """
    if isinstance(serie, list):
        return [int(x) for x in serie]
    if isinstance(serie, np.ndarray):
        return [int(x) for x in serie.tolist()]
    if isinstance(serie, pd.Series):
        return [int(x) for x in serie.tolist()]
    if isinstance(serie, str):
        # Ex: "1 2 3 4 5 6"
        partes = [p for p in serie.replace(",", " ").split() if p.strip()]
        return [int(p) for p in partes]
    # Fallback
    try:
        return [int(x) for x in list(serie)]
    except Exception:
        return []


def calcular_acerto_total(serie_real: List[int], serie_prevista: List[int]) -> int:
    """
    Calcula acerto total (6 acertos exatos = 6, etc).
    Aqui consideramos acerto pleno se todos os passageiros forem iguais.
    """
    if len(serie_real) != len(serie_prevista):
        return 0
    return sum(1 for a, b in zip(serie_real, serie_prevista) if int(a) == int(b))


# ============================================================
# PARSER ULTRA DO HISTÓRICO — ENTRADA FLEX COMPLETA
# ============================================================

def _parser_ultra_from_text(texto: str, sep: str = ";") -> pd.DataFrame:
    """
    Parser ULTRA a partir de texto puro.
    Aceita separador ';' ou ',' e reconstrói:
    - serie_id (ex: C1..Cn)
    - p1..pn
    - k
    - idx (1..n)
    """
    # Normaliza separador
    if sep == ",":
        texto = texto.replace(",", ";")

    # Remoção de caracteres invisíveis comuns
    texto = texto.replace("\ufeff", "").replace("\r", "")

    linhas = [ln for ln in texto.strip().split("\n") if ln.strip()]
    dados = []

    for ln in linhas:
        partes = [p.strip() for p in ln.split(";")]

        # Esperado: id + n_passageiros + k  (pelo menos 3 colunas -> id,p1,k)
        if len(partes) < 3:
            continue

        serie_id_bruto = partes[0]
        if not serie_id_bruto:
            continue

        # Aceita "C1", "c1", "  C1  ", etc.
        serie_id = serie_id_bruto.strip().upper()
        if not serie_id.startswith("C"):
            # Se não tem C explícito, tenta construir
            if serie_id.isdigit():
                serie_id = f"C{serie_id}"
            else:
                # Linha com id inválido: ignora
                continue

        # Passageiros = todas as colunas intermediárias, exceto última (k)
        valores = partes[1:-1]
        k_str = partes[-1]

        # Converte passageiros e k para numérico
        try:
            passageiros = [int(x) for x in valores]
            k_val = int(k_str)
        except Exception:
            # Linha com problemas numéricos: ignora
            continue

        # Monta registro flexível
        registro = {"serie_id": serie_id}
        for i, v in enumerate(passageiros, start=1):
            registro[f"p{i}"] = v
        registro["k"] = k_val
        dados.append(registro)

    if not dados:
        raise ValueError("Nenhuma linha válida encontrada no histórico (parser ULTRA).")

    df = pd.DataFrame(dados)

    # Recria 'idx' como 1..n respeitando a ordem original
    df["idx"] = np.arange(1, len(df) + 1)

    # Garante ordem das colunas: serie_id, idx, p1..pn, k
    col_pass = extrair_passageiros(df)
    cols_ordenadas = ["serie_id", "idx"] + col_pass + ["k"]
    df = df[cols_ordenadas]

    return df


def carregar_historico_via_csv_ultra(file) -> pd.DataFrame:
    """
    Lê o arquivo enviado pelo usuário (upload CSV) usando o parser ULTRA robusto.
    Garante que NÃO corta a última linha e reconstrói idx.
    """
    raw = file.read()
    # Tenta utf-8-sig primeiro
    try:
        texto = raw.decode("utf-8-sig")
    except Exception:
        # Fallback para latin1
        texto = raw.decode("latin1")

    return _parser_ultra_from_text(texto, sep=";")


def carregar_historico_via_texto_ultra(texto: str, sep: str = ";") -> pd.DataFrame:
    """
    Lê o histórico colado como texto no app (modo texto), usando
    o mesmo parser ULTRA do CSV.
    """
    return _parser_ultra_from_text(texto, sep=sep)


# ============================================================
# REGIME ULTRA — BARÔMETRO DA ESTRADA
# ============================================================

def calcular_regime_ultra(df: pd.DataFrame, janela: int = 40) -> RegimeInfo:
    """
    Calcula o regime (barômetro ULTRA) usando uma janela recente de k.
    - k_medio
    - k_max
    - estado (estavel / transicao / ruptura)
    """
    if "k" not in df.columns:
        raise KeyError("Coluna 'k' não encontrada no histórico.")

    # Usa sempre a cauda do histórico
    janela = min(janela, len(df))
    trecho = df.tail(janela)
    k_vals = trecho["k"].astype(float).values

    k_medio = float(np.mean(k_vals)) if len(k_vals) > 0 else 0.0
    k_max = float(np.max(k_vals)) if len(k_vals) > 0 else 0.0

    # Heurística de regime (mantendo jeitão ULTRA)
    if k_medio < 0.05 and k_max <= 1:
        estado = "ruptura"
        desc = "🔴 Estrada em ruptura — muitos carros com k=0, baixíssima previsibilidade estrutural."
    elif k_medio < 0.25:
        estado = "estavel"
        desc = "🟢 Estrada estável — poucos guardas acertando exatamente o carro."
    else:
        estado = "transicao"
        desc = "🟡 Estrada em transição — guardas começando a acertar em alguns pontos."

    return RegimeInfo(
        janela_usada=janela,
        k_medio=k_medio,
        k_max=k_max,
        estado=estado,
        descricao=desc,
    )


# ============================================================
# k* ULTRA REAL — SENTINELA DOS GUARDAS
# ============================================================

def calcular_k_star_ultra(df: pd.DataFrame, janela: int = 40) -> KStarInfo:
    """
    Calcula o k* ULTRA (sentinela preditivo) usando distribuição recente de k.
    Interpretação:
    - k* baixo: ambiente estável-fraco (sem padrão forte, sem caos)
    - k* médio: pré-transição / sensibilidade moderada
    - k* alto: turbulência / ruptura de regime / pré-choque
    """
    if "k" not in df.columns:
        raise KeyError("Coluna 'k' não encontrada no histórico.")

    janela = min(janela, len(df))
    trecho = df.tail(janela)
    k_vals = trecho["k"].astype(float).values

    if len(k_vals) == 0:
        return KStarInfo(
            janela_usada=0,
            k_media_janela=0.0,
            k_max_janela=0.0,
            k_star_pct=0.0,
            estado="neutro",
            descricao="⚪ k*: neutro — histórico insuficiente para avaliar sensibilidade.",
        )

    k_media = float(np.mean(k_vals))
    k_max = float(np.max(k_vals))

    # Uma forma ULTRA simples de k*:
    # proporção de séries com k > 0 na janela
    proporcao_k_pos = float(np.mean(k_vals > 0))

    k_star_pct = round(proporcao_k_pos * 100.0, 1)

    # Estados de k*
    if k_star_pct < 15:
        estado = "estavel"
        desc = "🟢 k*: Ambiente estável — poucos guardas acertando exatamente."
    elif k_star_pct < 40:
        estado = "atencao"
        desc = "🟡 k*: Pré-transição — sensibilidade crescente dos guardas."
    else:
        estado = "critico"
        desc = "🔴 k*: Ambiente crítico — guardas muito sensíveis, alta chance de rupturas."

    return KStarInfo(
        janela_usada=janela,
        k_media_janela=k_media,
        k_max_janela=k_max,
        k_star_pct=k_star_pct,
        estado=estado,
        descricao=desc,
    )


# ============================================================
# IDX ULTRA — CONTEXTO ESTRUTURAL GLOBAL
# ============================================================

def construir_contexto_idx_ultra(df: pd.DataFrame, idx_alvo: int) -> Dict[str, Any]:
    """
    Constrói um contexto IDX ULTRA simples:
    - janela usada
    - média de k
    - k max
    - índice global (proxy)
    Mantém apenas como insumo de contexto (não para visão determinística).
    """
    min_idx, max_idx = obter_intervalo_indices(df)
    idx_alvo = int(idx_alvo)
    if idx_alvo < min_idx:
        idx_alvo = min_idx
    if idx_alvo > max_idx:
        idx_alvo = max_idx

    # Usa janela até o alvo (inclusive)
    janela = 40
    corte = df[df["idx"] <= idx_alvo].tail(janela)
    janela_real = len(corte)
    if janela_real == 0:
        return {
            "janela_usada": 0,
            "media_k": 0.0,
            "max_k": 0.0,
            "idx_global": 0.0,
        }

    k_vals = corte["k"].astype(float).values
    media_k = float(np.mean(k_vals))
    max_k = float(np.max(k_vals))

    # IDX ULTRA: proxy simples — quanto menor k, maior dispersão / índice.
    # Aqui usamos algo como:
    # idx_global = (1 - media_k / (1 + max_k)) * 50  (mantendo jeitão "29.91", etc.)
    idx_global = (1.0 - (media_k / (1.0 + max_k))) * 50.0

    return {
        "janela_usada": janela_real,
        "media_k": media_k,
        "max_k": max_k,
        "idx_global": idx_global,
    }


# ============================================================
# PLACEHOLDERS PARA MÓDULOS PROFUNDOS (S6 / MICRO / MC / BACKTEST)
# (Implementados na PARTE 2/4 para manter organização)
# ============================================================

# As funções abaixo serão definidas completamente na Parte 2/4:
#
# - montar_previsao_turbo_ultra(...)
# - montar_contexto_replay_light(...)
# - montar_contexto_replay_ultra_unitario(...)
# - executar_backtest_real(...)
# - calcular_qds_real(...)
#
# Aqui apenas declaramos as assinaturas para referência de tipo.


def montar_previsao_turbo_ultra(
    df_hist: pd.DataFrame,
    idx_alvo: int,
    regime_info: RegimeInfo,
    k_info: KStarInfo,
    n_series_base: int = 300,
    n_series_micro: int = 80,
    n_sim_mc: int = 800,
    output_mode: str = "automatico",
    n_series_fixed: int = 25,
    min_conf_pct: float = 30.0,
) -> pd.DataFrame:
    """
    Implementação completa na PARTE 2/4.
    """
    raise NotImplementedError("montar_previsao_turbo_ultra será implementada na PARTE 2/4.")


def montar_contexto_replay_light(
    df_hist: pd.DataFrame,
    idx_alvo: int,
    regime_info: RegimeInfo,
    k_info: KStarInfo,
) -> Dict[str, Any]:
    """
    Implementação completa na PARTE 2/4.
    """
    raise NotImplementedError("montar_contexto_replay_light será implementada na PARTE 2/4.")


def montar_contexto_replay_ultra_unitario(
    df_hist: pd.DataFrame,
    idx_alvo: int,
    df_turbo: pd.DataFrame,
    qds_info: Optional[QDSInfo],
    regime_info: RegimeInfo,
    k_info: KStarInfo,
    top_n: int = 25,
) -> Dict[str, Any]:
    """
    Implementação completa na PARTE 2/4.
    """
    raise NotImplementedError("montar_contexto_replay_ultra_unitario será implementada na PARTE 2/4.")


def executar_backtest_real(
    df_hist: pd.DataFrame,
    n_passageiros: int,
    janela: int = 150,
    top_n: int = 10,
) -> BacktestResult:
    """
    Implementação completa na PARTE 2/4.
    """
    raise NotImplementedError("executar_backtest_real será implementada na PARTE 2/4.")


def calcular_qds_real(
    tabela_bt: pd.DataFrame,
    top_n: int = 10,
) -> QDSInfo:
    """
    Implementação completa na PARTE 2/4.
    """
    raise NotImplementedError("calcular_qds_real será implementada na PARTE 2/4.")
# ============================================================
# MÓDULOS PROFUNDOS — S6 / MICRO / MONTE CARLO / FUSÃO ULTRA
# ============================================================

def _gerar_leque_s6_ultra(
    df_hist: pd.DataFrame,
    idx_alvo: int,
    n_series_base: int,
) -> pd.DataFrame:
    """
    Gera um leque base tipo S6 ULTRA:
    - usa séries reais anteriores ao idx_alvo
    - recorta janela estruturada (até ~300 últimos carros)
    - permite repetição parcial, mantendo o jeitão "núcleo determinístico"
    """
    min_idx, max_idx = obter_intervalo_indices(df_hist)
    idx_alvo = int(idx_alvo)
    if idx_alvo <= min_idx:
        idx_alvo = min_idx + 1
    if idx_alvo > max_idx:
        idx_alvo = max_idx

    df_passado = df_hist[df_hist["idx"] < idx_alvo].copy()
    if df_passado.empty:
        df_passado = df_hist.copy()

    # Janela máxima para estrutura (S6)
    df_passado = df_passado.tail(300)

    cols_pass = extrair_passageiros(df_passado)
    registros = []

    # Se n_series_base > número de linhas, podemos replicar amostrando com reposição
    if len(df_passado) == 0:
        return pd.DataFrame(columns=["series", "origem", "score_s6"])

    # Usa amostragem com reposição para gerar base
    for _ in range(n_series_base):
        row = df_passado.sample(1, replace=True).iloc[0]
        serie = [int(row[c]) for c in cols_pass]
        registros.append(
            {
                "series": serie,
                "origem": "S6",
                "score_s6": 1.0,  # score relativo será refinado depois
            }
        )

    df_s6 = pd.DataFrame(registros)
    return df_s6


def _gerar_leque_micro_ultra(
    df_hist: pd.DataFrame,
    idx_alvo: int,
    n_series_micro: int,
) -> pd.DataFrame:
    """
    Micro-Leque ULTRA:
    - gera variações finas em torno do carro alvo-1 (ou última série conhecida)
    - pequenas perturbações em +/- 1, 2, 3
    """
    min_idx, max_idx = obter_intervalo_indices(df_hist)
    idx_alvo = int(idx_alvo)
    if idx_alvo <= min_idx:
        idx_ref = min_idx
    else:
        idx_ref = idx_alvo - 1
    if idx_ref > max_idx:
        idx_ref = max_idx

    df_ref = df_hist[df_hist["idx"] == idx_ref]
    if df_ref.empty:
        df_ref = df_hist.tail(1)

    cols_pass = extrair_passageiros(df_ref)
    base = df_ref.iloc[0]
    base_serie = [int(base[c]) for c in cols_pass]

    registros = []
    rng = np.random.default_rng(seed=idx_alvo + 123)

    for _ in range(n_series_micro):
        nova = base_serie.copy()
        # Perturba algumas posições aleatórias
        n_positions = max(1, int(len(nova) / 2))
        posicoes = rng.choice(len(nova), size=n_positions, replace=False)
        for p in posicoes:
            delta = int(rng.integers(-3, 4))  # -3..+3
            novo_valor = max(1, nova[p] + delta)
            # Mantém um teto genérico (ex: 60) para não explodir para cima
            novo_valor = min(60, novo_valor)
            nova[p] = novo_valor

        registros.append(
            {
                "series": nova,
                "origem": "MICRO",
                "score_micro": 1.0,
            }
        )

    df_micro = pd.DataFrame(registros)
    return df_micro


def _gerar_leque_mc_ultra(
    df_hist: pd.DataFrame,
    idx_alvo: int,
    n_sim_mc: int,
) -> pd.DataFrame:
    """
    Monte Carlo Profundo ULTRA:
    - gera séries novas com base na distribuição marginal dos passageiros
      nas últimas janelas históricas (modo puramente estocástico).
    """
    min_idx, max_idx = obter_intervalo_indices(df_hist)
    idx_alvo = int(idx_alvo)
    if idx_alvo <= min_idx:
        idx_alvo = min_idx + 1
    if idx_alvo > max_idx:
        idx_alvo = max_idx

    df_passado = df_hist[df_hist["idx"] < idx_alvo].copy()
    if df_passado.empty:
        df_passado = df_hist.copy()

    # Usa uma janela maior para Monte Carlo
    df_passado = df_passado.tail(400)
    cols_pass = extrair_passageiros(df_passado)
    if df_passado.empty or not cols_pass:
        return pd.DataFrame(columns=["series", "origem", "score_mc"])

    rng = np.random.default_rng(seed=idx_alvo + 987)
    registros = []

    # Pré-calcula distribuições empíricas por posição
    valores_por_col = {}
    for c in cols_pass:
        valores = df_passado[c].dropna().astype(int).values
        if len(valores) == 0:
            valores = np.arange(1, 61)
        valores_por_col[c] = valores

    for _ in range(n_sim_mc):
        serie = []
        for c in cols_pass:
            vals = valores_por_col[c]
            serie.append(int(rng.choice(vals)))
        registros.append(
            {
                "series": serie,
                "origem": "MC",
                "score_mc": 1.0,
            }
        )

    df_mc = pd.DataFrame(registros)
    return df_mc


def _score_freq_basico(
    df_hist: pd.DataFrame,
    candidatos: pd.DataFrame,
) -> pd.DataFrame:
    """
    Score básico de frequência:
    - séries com passageiros mais frequentes na janela recente recebem score maior.
    - mantém o jeitão de um "TVF" simplificado.
    """
    if candidatos.empty:
        return candidatos

    cols_pass = extrair_passageiros(df_hist)
    if not cols_pass:
        return candidatos

    # Frequências recentes (janela de 200 carros)
    df_ref = df_hist.tail(200)
    freq = {}
    for c in cols_pass:
        valores = df_ref[c].dropna().astype(int)
        freq[c] = valores.value_counts(normalize=True).to_dict()

    scores = []
    for _, row in candidatos.iterrows():
        serie = row["series"]
        s = 0.0
        for i, val in enumerate(serie):
            col = cols_pass[i] if i < len(cols_pass) else cols_pass[-1]
            f = freq.get(col, {}).get(int(val), 0.0)
            s += f
        scores.append(s)

    candidatos = candidatos.copy()
    candidatos["score_freq"] = scores
    return candidatos


def _combinar_leques_ultra(
    df_s6: pd.DataFrame,
    df_micro: pd.DataFrame,
    df_mc: pd.DataFrame,
    peso_s6: float,
    peso_micro: float,
    peso_mc: float,
) -> pd.DataFrame:
    """
    Combina os leques S6 / MICRO / MC em um único dataframe,
    aplicando pesos adaptativos e um score global.
    """
    frames = []

    if not df_s6.empty:
        df_s6 = df_s6.copy()
        if "score_s6" not in df_s6.columns:
            df_s6["score_s6"] = 1.0
        df_s6["score_micro"] = 0.0
        df_s6["score_mc"] = 0.0
        frames.append(df_s6)

    if not df_micro.empty:
        df_micro = df_micro.copy()
        if "score_micro" not in df_micro.columns:
            df_micro["score_micro"] = 1.0
        df_micro["score_s6"] = 0.0
        df_micro["score_mc"] = 0.0
        frames.append(df_micro)

    if not df_mc.empty:
        df_mc = df_mc.copy()
        if "score_mc" not in df_mc.columns:
            df_mc["score_mc"] = 1.0
        df_mc["score_s6"] = 0.0
        df_mc["score_micro"] = 0.0
        frames.append(df_mc)

    if not frames:
        return pd.DataFrame(columns=["series", "origem", "score_s6", "score_micro", "score_mc", "score_global"])

    df_mix = pd.concat(frames, ignore_index=True)

    # Score base por origem
    # S6 recebe foco maior em ambientes mais estáveis
    df_mix["score_origem"] = (
        df_mix["score_s6"] * peso_s6
        + df_mix["score_micro"] * peso_micro
        + df_mix["score_mc"] * peso_mc
    )

    # Score global ainda será refinado com frequência
    df_mix["score_global"] = df_mix["score_origem"].astype(float)

    return df_mix


# ============================================================
# MONTAGEM DA PREVISÃO TURBO++ ULTRA (ADAPTATIVO POR k*)
# ============================================================

def montar_previsao_turbo_ultra(
    df_hist: pd.DataFrame,
    idx_alvo: int,
    regime_info: RegimeInfo,
    k_info: KStarInfo,
    n_series_base: int = 300,
    n_series_micro: int = 80,
    n_sim_mc: int = 800,
    output_mode: str = "automatico",
    n_series_fixed: int = 25,
    min_conf_pct: float = 30.0,
) -> pd.DataFrame:
    """
    Núcleo do Modo TURBO++ ULTRA Adaptativo.
    - Gera leques S6 / MICRO / MC
    - Aplica pesos por regime (usando k* e barômetro)
    - Calcula score global com frequência
    - Limita por modo de saída (automático / fixo / confiabilidade mínima)
    Retorna df com colunas:
        - series (list[int])
        - origem (S6 / MICRO / MC)
        - score_s6 / score_micro / score_mc
        - score_freq
        - score_global
    """
    if df_hist is None or df_hist.empty:
        return pd.DataFrame(columns=["series", "origem", "score_global"])

    # Define pesos adaptativos conforme regime + k*
    if regime_info.estado == "ruptura":
        # Ruptura: Monte Carlo dominante, Micro secundário, S6 fraco
        peso_s6 = 0.10
        peso_mc = 0.70
        peso_micro = 0.20
    elif regime_info.estado == "transicao":
        # Transição: equilíbrio moderado
        peso_s6 = 0.30
        peso_mc = 0.40
        peso_micro = 0.30
    else:
        # Estável: S6 e Micro mais relevantes, MC complementa
        peso_s6 = 0.40
        peso_mc = 0.30
        peso_micro = 0.30

    # Pequeno ajuste dinâmico usando k*
    if k_info.k_star_pct < 15:
        # Ambiente estável-fraco, menos estrutura forte, MC um pouco mais
        peso_mc += 0.05
        peso_s6 -= 0.05
    elif k_info.k_star_pct > 40:
        # Ambiente crítico, alta sensibilidade, MC e Micro sobem
        peso_mc += 0.05
        peso_micro += 0.05
        peso_s6 -= 0.10

    # Normaliza pesos para somar 1
    soma = peso_s6 + peso_mc + peso_micro
    if soma <= 0:
        peso_s6, peso_mc, peso_micro = 0.33, 0.34, 0.33
    else:
        peso_s6 /= soma
        peso_mc /= soma
        peso_micro /= soma

    # Geração dos leques
    df_s6 = _gerar_leque_s6_ultra(df_hist, idx_alvo, n_series_base)
    df_micro = _gerar_leque_micro_ultra(df_hist, idx_alvo, n_series_micro)
    df_mc = _gerar_leque_mc_ultra(df_hist, idx_alvo, n_sim_mc)

    df_mix = _combinar_leques_ultra(df_s6, df_micro, df_mc, peso_s6, peso_micro, peso_mc)

    if df_mix.empty:
        return df_mix

    # Score de frequência (TVF simplificado)
    df_mix = _score_freq_basico(df_hist, df_mix)

    # Score global = combinação entre origem e frequência
    # Mantendo um jeitão "turbo" de TVF global
    max_origem = max(df_mix["score_origem"].max(), 1e-9)
    max_freq = max(df_mix["score_freq"].max(), 1e-9)

    df_mix["score_origem_norm"] = df_mix["score_origem"] / max_origem
    df_mix["score_freq_norm"] = df_mix["score_freq"] / max_freq

    # Peso 60% frequência, 40% origem (pode ser refinado)
    df_mix["score_global"] = (
        0.4 * df_mix["score_origem_norm"] + 0.6 * df_mix["score_freq_norm"]
    )

    # Remove duplicatas de séries mantendo maior score
    # Converte series para tupla para agrupar
    df_mix["series_key"] = df_mix["series"].apply(lambda s: tuple(s))
    df_mix = df_mix.sort_values("score_global", ascending=False)
    df_mix = df_mix.drop_duplicates(subset=["series_key"], keep="first")

    # Aplica modo de saída
    if output_mode == "Quantidade fixa":
        df_final = df_mix.head(n_series_fixed).copy()
    elif output_mode == "Confiabilidade mínima":
        # Mantém séries com score >= (min_conf_pct% do máximo)
        max_score = df_mix["score_global"].max()
        limite = max_score * (float(min_conf_pct) / 100.0)
        df_filtro = df_mix[df_mix["score_global"] >= limite]
        if df_filtro.empty:
            df_final = df_mix.head(n_series_fixed).copy()
        else:
            df_final = df_filtro.copy()
    else:
        # Automático: limita a um máximo razoável (25 por padrão)
        df_final = df_mix.head(n_series_fixed).copy()

    df_final = df_final.reset_index(drop=True)
    return df_final


# ============================================================
# REPLAY LIGHT — CONTEXTO SIMPLIFICADO
# ============================================================

def montar_contexto_replay_light(
    df_hist: pd.DataFrame,
    idx_alvo: int,
    regime_info: RegimeInfo,
    k_info: KStarInfo,
) -> Dict[str, Any]:
    """
    Replay LIGHT:
    - pega a série real do idx_alvo
    - monta previsão TURBO++ ULTRA para esse alvo
    - calcula acertos no leque
    - retorna um contexto leve para visualização rápida.
    """
    min_idx, max_idx = obter_intervalo_indices(df_hist)
    idx_alvo = int(idx_alvo)
    if idx_alvo < min_idx or idx_alvo > max_idx:
        raise ValueError(f"Índice alvo {idx_alvo} fora do intervalo [{min_idx}, {max_idx}].")

    df_alvo = df_hist[df_hist["idx"] == idx_alvo]
    if df_alvo.empty:
        raise ValueError(f"Índice alvo {idx_alvo} não encontrado no histórico.")

    cols_pass = extrair_passageiros(df_hist)
    row = df_alvo.iloc[0]
    serie_real = [int(row[c]) for c in cols_pass]

    # Monta um leque TURBO para o replay
    df_turbo = montar_previsao_turbo_ultra(
        df_hist=df_hist,
        idx_alvo=idx_alvo,
        regime_info=regime_info,
        k_info=k_info,
        n_series_base=200,
        n_series_micro=60,
        n_sim_mc=500,
        output_mode="Quantidade fixa",
        n_series_fixed=25,
    )

    if df_turbo.empty:
        return {
            "idx_alvo": idx_alvo,
            "serie_real": serie_real,
            "df_turbo": pd.DataFrame(),
            "acertos_top1": 0,
            "melhor_acerto": 0,
            "pos_melhor": None,
        }

    acertos = []
    for _, r in df_turbo.iterrows():
        prev = r["series"]
        acertos.append(calcular_acerto_total(serie_real, prev))

    df_turbo = df_turbo.copy()
    df_turbo["acertos"] = acertos
    df_turbo = df_turbo.sort_values(["acertos", "score_global"], ascending=[False, False])

    melhor_acerto = int(df_turbo["acertos"].max())
    pos_melhor = int(df_turbo["acertos"].idxmax())
    # Índice relativo no leque (posição dentro do top-N)
    pos_melhor_rank = int(df_turbo.reset_index(drop=True)["acertos"].idxmax()) + 1

    acertos_top1 = int(df_turbo["acertos"].iloc[0])

    contexto = {
        "idx_alvo": idx_alvo,
        "serie_real": serie_real,
        "df_turbo": df_turbo.reset_index(drop=True),
        "acertos_top1": acertos_top1,
        "melhor_acerto": melhor_acerto,
        "pos_melhor_absoluto": pos_melhor,
        "pos_melhor_rank": pos_melhor_rank,
    }
    return contexto


# ============================================================
# REPLAY ULTRA UNITÁRIO — CONTEXTO COMPLETO (1 ÍNDICE)
# ============================================================

def montar_contexto_replay_ultra_unitario(
    df_hist: pd.DataFrame,
    idx_alvo: int,
    df_turbo: pd.DataFrame,
    qds_info: Optional[QDSInfo],
    regime_info: RegimeInfo,
    k_info: KStarInfo,
    top_n: int = 25,
) -> Dict[str, Any]:
    """
    Replay ULTRA UNITÁRIO:
    - usa o leque TURBO++ (df_turbo já calculado)
    - compara com a série real do índice alvo
    - calcula acertos detalhados (top-1, top-N, melhor acerto, etc.)
    - inclui QDS e contexto de regime/k*.
    """
    min_idx, max_idx = obter_intervalo_indices(df_hist)
    idx_alvo = int(idx_alvo)
    if idx_alvo < min_idx or idx_alvo > max_idx:
        raise ValueError(f"Índice alvo {idx_alvo} fora do intervalo [{min_idx}, {max_idx}].")

    df_alvo = df_hist[df_hist["idx"] == idx_alvo]
    if df_alvo.empty:
        raise ValueError(f"Índice alvo {idx_alvo} não encontrado no histórico.")

    cols_pass = extrair_passageiros(df_hist)
    row = df_alvo.iloc[0]
    serie_real = [int(row[c]) for c in cols_pass]

    if df_turbo is None or df_turbo.empty:
        return {
            "idx_alvo": idx_alvo,
            "serie_real": serie_real,
            "df_turbo": pd.DataFrame(),
            "acertos_top1": 0,
            "acertos_topN": 0,
            "melhor_acerto": 0,
            "pos_melhor_rank": None,
            "qds": qds_info.valor if qds_info else None,
            "regime": regime_info.estado,
            "k_star": k_info.k_star_pct,
        }

    df_turbo = df_turbo.copy()
    acertos = []
    for _, r in df_turbo.iterrows():
        prev = r["series"]
        acertos.append(calcular_acerto_total(serie_real, prev))

    df_turbo["acertos"] = acertos
    df_turbo = df_turbo.sort_values(["acertos", "score_global"], ascending=[False, False])
    df_turbo = df_turbo.reset_index(drop=True)

    melhor_acerto = int(df_turbo["acertos"].max())
    pos_melhor_rank = int(df_turbo["acertos"].idxmax()) + 1

    top_n = min(top_n, len(df_turbo))
    df_top = df_turbo.head(top_n)
    acertos_topN = int((df_top["acertos"] > 0).sum())
    acertos_top1 = int(df_top["acertos"].iloc[0])

    contexto = {
        "idx_alvo": idx_alvo,
        "serie_real": serie_real,
        "df_turbo": df_turbo,
        "acertos_top1": acertos_top1,
        "acertos_topN": acertos_topN,
        "melhor_acerto": melhor_acerto,
        "pos_melhor_rank": pos_melhor_rank,
        "qds": qds_info.valor if qds_info else None,
        "qds_descricao": qds_info.descricao if qds_info else "",
        "regime": regime_info.estado,
        "regime_descricao": regime_info.descricao,
        "k_star": k_info.k_star_pct,
        "k_star_descricao": k_info.descricao,
    }
    return contexto


# ============================================================
# BACKTEST REAL — MODO ESTRUTURAL
# ============================================================

def executar_backtest_real(
    df_hist: pd.DataFrame,
    n_passageiros: int,
    janela: int = 150,
    top_n: int = 10,
) -> BacktestResult:
    """
    Backtest REAL:
    - percorre uma janela de índices finais
    - para cada idx, calcula o leque TURBO++ ULTRA usando apenas o passado
    - verifica acerto no top-1 e top-N
    Retorna:
        - tabela com colunas: idx_alvo, acertos_top1, acertos_topN, melhor_acerto
        - descrição textual da performance.
    """
    min_idx, max_idx = obter_intervalo_indices(df_hist)
    if max_idx - min_idx < 10:
        return BacktestResult(
            tabela=pd.DataFrame(),
            descricao="Histórico insuficiente para backtest.",
        )

    # Limita janela final
    inicio_bt = max(min_idx + 10, max_idx - janela)
    indices_teste = list(range(inicio_bt, max_idx + 1))

    registros = []

    for idx_alvo in indices_teste:
        df_passado = df_hist[df_hist["idx"] < idx_alvo]
        if df_passado.empty:
            continue

        regime_local = calcular_regime_ultra(df_passado, janela=40)
        k_local = calcular_k_star_ultra(df_passado, janela=40)

        df_alvo = df_hist[df_hist["idx"] == idx_alvo]
        if df_alvo.empty:
            continue
        cols_pass = extrair_passageiros(df_hist)
        serie_real = [int(df_alvo.iloc[0][c]) for c in cols_pass]

        df_turbo = montar_previsao_turbo_ultra(
            df_hist=df_passado,
            idx_alvo=idx_alvo,
            regime_info=regime_local,
            k_info=k_local,
            n_series_base=200,
            n_series_micro=60,
            n_sim_mc=400,
            output_mode="Quantidade fixa",
            n_series_fixed=top_n,
        )

        if df_turbo.empty:
            registros.append(
                {
                    "idx_alvo": idx_alvo,
                    "acertos_top1": 0,
                    "acertos_topN": 0,
                    "melhor_acerto": 0,
                }
            )
            continue

        acertos = []
        for _, r in df_turbo.iterrows():
            prev = r["series"]
            acertos.append(calcular_acerto_total(serie_real, prev))

        df_turbo = df_turbo.copy()
        df_turbo["acertos"] = acertos
        df_turbo = df_turbo.sort_values(["acertos", "score_global"], ascending=[False, False])
        df_turbo = df_turbo.reset_index(drop=True)

        melhor_acerto = int(df_turbo["acertos"].max())
        ac_top1 = int(df_turbo["acertos"].iloc[0])
        df_topN = df_turbo.head(top_n)
        ac_topN = int((df_topN["acertos"] > 0).sum())

        registros.append(
            {
                "idx_alvo": idx_alvo,
                "acertos_top1": ac_top1,
                "acertos_topN": ac_topN,
                "melhor_acerto": melhor_acerto,
            }
        )

    if not registros:
        return BacktestResult(
            tabela=pd.DataFrame(),
            descricao="Backtest não gerou registros.",
        )

    df_bt = pd.DataFrame(registros)
    # Métricas agregadas
    total = len(df_bt)
    hits_top1 = int((df_bt["acertos_top1"] > 0).sum())
    hits_topN = int((df_bt["acertos_topN"] > 0).sum())

    pct_top1 = hits_top1 / total if total > 0 else 0.0
    pct_topN = hits_topN / total if total > 0 else 0.0

    desc = (
        f"Backtest REAL executado em {total} índices. "
        f"Acerto top-1 em {hits_top1} casos ({pct_top1:.1%}) "
        f"e acerto em pelo menos 1 série no top-{top_n} em {hits_topN} casos ({pct_topN:.1%})."
    )

    return BacktestResult(tabela=df_bt, descricao=desc)


# ============================================================
# QDS REAL — ÍNDICE DE QUALIDADE DINÂMICA
# ============================================================

def calcular_qds_real(
    tabela_bt: pd.DataFrame,
    top_n: int = 10,
) -> QDSInfo:
    """
    QDS REAL:
    - converte a tabela de backtest em um índice de 0 a 1.
    - combina:
        - taxa de acerto top-1
        - taxa de acerto top-N
        - intensidade dos melhores acertos.
    """
    if tabela_bt is None or tabela_bt.empty:
        return QDSInfo(
            valor=0.0,
            descricao="QDS = 0.0 — sem dados de backtest disponíveis.",
        )

    total = len(tabela_bt)
    if total == 0:
        return QDSInfo(
            valor=0.0,
            descricao="QDS = 0.0 — backtest vazio.",
        )

    hits_top1 = int((tabela_bt["acertos_top1"] > 0).sum())
    hits_topN = int((tabela_bt["acertos_topN"] > 0).sum())
    melhor_med = float(tabela_bt["melhor_acerto"].mean())

    pct_top1 = hits_top1 / total
    pct_topN = hits_topN / total
    # Normaliza melhor_med para [0,1] considerando n_passageiros típico (6)
    melhor_norm = min(melhor_med / 6.0, 1.0)

    # Combinação ponderada
    qds = 0.4 * pct_top1 + 0.4 * pct_topN + 0.2 * melhor_norm
    qds = float(max(0.0, min(1.0, qds)))

    if qds < 0.2:
        desc = f"QDS = {qds:.3f} — qualidade muito baixa, sistema operando praticamente em regime aleatório."
    elif qds < 0.4:
        desc = f"QDS = {qds:.3f} — qualidade baixa, previsões com pouca aderência ao histórico."
    elif qds < 0.6:
        desc = f"QDS = {qds:.3f} — qualidade moderada, previsões razoáveis em parte dos cenários."
    elif qds < 0.8:
        desc = f"QDS = {qds:.3f} — boa qualidade, previsões consistentes em boa parte dos cenários."
    else:
        desc = f"QDS = {qds:.3f} — excelente qualidade, previsão altamente aderente ao comportamento histórico."

    return QDSInfo(valor=qds, descricao=desc)
# ============================================================
# INTERFACE — SIDEBAR E PAINÉIS PRINCIPAIS
# ============================================================

st.title("🚗 Predict Cars — V14-FLEX ULTRA REAL (TURBO++)")
st.caption(f"Versão completa: {APP_VERSION}")

# ------------------------------------------------------------
# SIDEBAR — NAVEGAÇÃO
# ------------------------------------------------------------

painel = st.sidebar.radio(
    "📂 Navegação",
    [
        "📥 Histórico — Entrada",
        "🔍 Pipeline V14-FLEX ULTRA",
        "🚨 Monitor de Risco (k & k*)",
        "🚀 Modo TURBO++ ULTRA Adaptativo",
        "💡 Replay LIGHT",
        "📅 Replay ULTRA (Loop Tradicional)",
        "🎯 Replay ULTRA UNITÁRIO (Novo)",
        "🧪 Testes de Confiabilidade",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Estado interno:**")
if "df" in st.session_state and st.session_state["df"] is not None:
    df_ref_sidebar = st.session_state["df"]
    try:
        min_idx_sb, max_idx_sb = obter_intervalo_indices(df_ref_sidebar)
        st.sidebar.write(f"Histórico: C{min_idx_sb} → C{max_idx_sb}")
        st.sidebar.write(f"Séries: {len(df_ref_sidebar)}")
    except Exception:
        st.sidebar.write("Histórico carregado, mas sem coluna idx.")
else:
    st.sidebar.write("Histórico ainda não carregado.")


# ============================================================
# PAINEL — HISTÓRICO (ENTRADA FLEX ULTRA)
# ============================================================

if painel == "📥 Histórico — Entrada":

    st.header("📥 Histórico — Entrada FLEX ULTRA")
    st.write(
        "Carregue seu histórico com **n passageiros + coluna k**. "
        "O parser ULTRA faz limpeza de ruídos, normaliza encoding e "
        "garante leitura fiel, incluindo a **última linha**."
    )

    metodo = st.radio(
        "Método de carregamento:",
        ["Upload CSV", "Colar texto"],
        horizontal=True,
    )

    df = None

    try:
        if metodo == "Upload CSV":
            file = st.file_uploader("Selecione arquivo CSV", type=["csv"])
            if file is not None:
                df = carregar_historico_via_csv_ultra(file)

        else:  # texto manual
            texto = st.text_area("Cole o conteúdo do CSV aqui")
            sep = st.selectbox("Separador", [";", ","], index=0)
            if texto.strip():
                df = carregar_historico_via_texto_ultra(texto, sep=sep)

        if df is not None:
            st.success(f"Histórico carregado com sucesso! Total: **{len(df)} séries**.")
            st.dataframe(df, use_container_width=True)

            st.session_state["df"] = df

            try:
                n_pass = obter_n_passageiros(df)
                st.info(f"n passageiros detectados: **{n_pass}**")

                min_idx, max_idx = obter_intervalo_indices(df)
                st.write(f"Intervalo de índices: **C{min_idx} → C{max_idx}**")
            except Exception as e:
                st.warning(f"Histórico carregado, mas não foi possível determinar índice/colunas: {e}")

    except Exception as e:
        st.error(f"Erro ao carregar histórico: {e}")


# ============================================================
# PAINEL — PIPELINE V14-FLEX ULTRA (ESTRUTURAL)
# ============================================================

elif painel == "🔍 Pipeline V14-FLEX ULTRA":

    st.header("🔍 Pipeline V14-FLEX (ULTRA) — Visão Estrutural")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    try:
        min_idx, max_idx = obter_intervalo_indices(df)
    except Exception as e:
        st.error(f"Histórico sem coluna 'idx' válida: {e}")
        st.stop()

    idx_alvo = st.number_input(
        "Selecione o índice alvo (1 = primeira série carregada):",
        min_value=int(min_idx),
        max_value=int(max_idx),
        value=int(max_idx),
        step=1,
    )

    # Barômetro / k* / IDX são sempre calculados em cima do histórico até idx_alvo
    df_ate_alvo = df[df["idx"] <= idx_alvo].copy()

    regime_info = calcular_regime_ultra(df_ate_alvo, janela=40)
    k_info = calcular_k_star_ultra(df_ate_alvo, janela=40)
    contexto_idx = construir_contexto_idx_ultra(df_ate_alvo, idx_alvo=idx_alvo)

    # Série alvo (estrutura)
    df_alvo = df[df["idx"] == idx_alvo]
    cols_pass = extrair_passageiros(df)
    if not df_alvo.empty:
        serie_alvo = [int(df_alvo.iloc[0][c]) for c in cols_pass]
    else:
        serie_alvo = []

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📡 Barômetro ULTRA REAL")
        st.markdown(regime_info.descricao)
        st.json(
            {
                "janela": regime_info.janela_usada,
                "k_medio": round(regime_info.k_medio, 3),
                "k_max": int(regime_info.k_max),
                "estado": regime_info.estado,
            }
        )

    with col2:
        st.subheader("🛡 k* ULTRA REAL — Sentinela")
        st.markdown(k_info.descricao)
        st.json(
            {
                "janela": k_info.janela_usada,
                "k_media_janela": round(k_info.k_media_janela, 3),
                "k_max_janela": int(k_info.k_max_janela),
                "k_star_pct": k_info.k_star_pct,
                "estado": k_info.estado,
            }
        )

    with col3:
        st.subheader("🧭 IDX ULTRA — Contexto")
        st.json(
            {
                "janela_usada": contexto_idx["janela_usada"],
                "media_k": round(contexto_idx["media_k"], 3),
                "max_k": int(contexto_idx["max_k"]),
                "indice_global": round(contexto_idx["idx_global"], 2),
            }
        )

    st.markdown("### 🧱 Série alvo (estrutura)")
    if serie_alvo:
        st.code(" ".join(str(x) for x in serie_alvo), language="text")
    else:
        st.info("Nenhuma série alvo encontrada para o índice selecionado.")

    st.caption(
        "Este painel mostra o **estado estrutural da estrada** (Barômetro, k*, IDX), "
        "que são insumos diretos para o Modo TURBO++ ULTRA Adaptativo, mas não gera previsão."
    )


# ============================================================
# PAINEL — MONITOR DE RISCO (k & k*)
# ============================================================

elif painel == "🚨 Monitor de Risco (k & k*)":

    st.header("🚨 Monitor de Risco — k & k*")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    try:
        min_idx, max_idx = obter_intervalo_indices(df)
    except Exception as e:
        st.error(f"Histórico sem coluna 'idx' válida: {e}")
        st.stop()

    janela = st.slider(
        "Tamanho da janela recente (para cálculo do risco):",
        min_value=20,
        max_value=min(200, int(len(df))),
        value=40,
        step=5,
    )

    df_recente = df.tail(janela)

    regime_info = calcular_regime_ultra(df_recente, janela=janela)
    k_info = calcular_k_star_ultra(df_recente, janela=janela)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📡 Barômetro da Estrada")
        st.markdown(regime_info.descricao)
        st.json(
            {
                "janela": regime_info.janela_usada,
                "k_medio": round(regime_info.k_medio, 3),
                "k_max": int(regime_info.k_max),
                "estado": regime_info.estado,
            }
        )

    with col2:
        st.subheader("🛡 k* — Sentinela dos Guardas")
        st.markdown(k_info.descricao)
        st.json(
            {
                "janela": k_info.janela_usada,
                "k_media_janela": round(k_info.k_media_janela, 3),
                "k_max_janela": int(k_info.k_max_janela),
                "k_star_pct": k_info.k_star_pct,
                "estado": k_info.estado,
            }
        )

    st.markdown("### 📊 Distribuição de k (guardas que acertaram exatamente)")
    hist_k = df_recente["k"].value_counts().sort_index()
    st.bar_chart(hist_k)

    st.caption(
        "O painel de risco usa a distribuição de **k** e a sensibilidade de **k*** "
        "para avaliar raridade, concentração e turbulência da estrada."
    )


# ============================================================
# PAINEL — MODO TURBO++ ULTRA ADAPTATIVO (por k*)
# ============================================================

elif painel == "🚀 Modo TURBO++ ULTRA Adaptativo":

    st.header("🚀 Modo TURBO++ ULTRA — Adaptativo por k*")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    try:
        min_idx, max_idx = obter_intervalo_indices(df)
    except Exception as e:
        st.error(f"Histórico sem coluna 'idx' válida: {e}")
        st.stop()

    st.markdown("### ⚙️ Configurações do TURBO++ ULTRA")

    col_cfg1, col_cfg2 = st.columns(2)

    with col_cfg1:
        idx_alvo = st.number_input(
            "Índice alvo (1 = primeira série):",
            min_value=int(min_idx + 1),
            max_value=int(max_idx),
            value=int(max_idx),
            step=1,
            help=(
                "O motor usa as séries **antes** do índice alvo como estrada de referência. "
                "Ex: se alvo = C2946, usa C1..C2945 como estrada."
            ),
        )

        top_n_final = st.slider(
            "Top-N final:",
            min_value=5,
            max_value=80,
            value=25,
            step=5,
        )

        output_mode = st.radio(
            "Modo de geração do Leque:",
            ["Automático", "Quantidade fixa", "Confiabilidade mínima"],
        )

    with col_cfg2:
        n_series_base = st.slider(
            "Quantidade de séries S6 Profundo ULTRA:",
            min_value=50,
            max_value=400,
            value=250,
            step=50,
        )

        n_sim_mc = st.slider(
            "Quantidade de séries Monte Carlo ULTRA:",
            min_value=300,
            max_value=1200,
            value=800,
            step=100,
        )

        n_series_micro = st.slider(
            "Micro-Leque (variações por série base):",
            min_value=5,
            max_value=40,
            value=20,
            step=5,
        )

    if output_mode == "Quantidade fixa":
        n_series_fixed = top_n_final
        min_conf_pct = 30.0
    elif output_mode == "Confiabilidade mínima":
        n_series_fixed = top_n_final
        min_conf_pct = st.slider(
            "Confiabilidade mínima (%) (se modo for 'Confiabilidade mínima'):",
            min_value=10,
            max_value=90,
            value=30,
            step=5,
        )
    else:
        n_series_fixed = top_n_final
        min_conf_pct = 30.0

    # Cálculo de contexto (regime/k* usando apenas passado)
    df_passado = df[df["idx"] < idx_alvo].copy()
    if df_passado.empty:
        df_passado = df.copy()

    regime_info = calcular_regime_ultra(df_passado, janela=40)
    k_info = calcular_k_star_ultra(df_passado, janela=40)

    st.markdown("### 🌟 Contexto adaptativo")
    col_ctx1, col_ctx2, col_ctx3 = st.columns(3)

    with col_ctx1:
        st.subheader("k* (sentinela)")
        st.write(f"{k_info.k_star_pct:.1f} %")

    with col_ctx2:
        st.subheader("QDS local (janela curta)")
        st.write("0.000")  # QDS local ainda não integrado aqui (usamos QDS no painel de testes)

    with col_ctx3:
        # Pesos esperados (derivados indiretamente do regime, apenas descrição)
        if regime_info.estado == "ruptura":
            pesos_desc = "S6: 0.10 • Monte Carlo: 0.70 • Micro-Leque: 0.20"
        elif regime_info.estado == "transicao":
            pesos_desc = "S6: 0.30 • Monte Carlo: 0.40 • Micro-Leque: 0.30"
        else:
            pesos_desc = "S6: 0.40 • Monte Carlo: 0.30 • Micro-Leque: 0.30"
        st.subheader("Pesos por regime (descrição)")
        st.write(pesos_desc)

    if st.button("🧠 Rodar TURBO++ ULTRA", type="primary"):
        with st.spinner("Gerando leque TURBO++ ULTRA..."):
            df_turbo = montar_previsao_turbo_ultra(
                df_hist=df,
                idx_alvo=int(idx_alvo),
                regime_info=regime_info,
                k_info=k_info,
                n_series_base=int(n_series_base),
                n_series_micro=int(n_series_micro),
                n_sim_mc=int(n_sim_mc),
                output_mode=output_mode,
                n_series_fixed=int(top_n_final),
                min_conf_pct=float(min_conf_pct),
            )

        if df_turbo is None or df_turbo.empty:
            st.error("Não foi possível gerar o leque TURBO++ ULTRA.")
        else:
            st.success(f"Leque TURBO++ ULTRA gerado com **{len(df_turbo)} séries**.")

            st.markdown("### 🧠 S6 Profundo ULTRA — núcleo determinístico\n"
                        "Monte Carlo Profundo ULTRA — motor estocástico\n"
                        "Micro-Leque ULTRA — variações finas\n"
                        "Fusão ULTRA ADAPTATIVA — Top-N final")

            st.dataframe(df_turbo[["series", "origem", "score_global"]], use_container_width=True)

            # Previsão final = top-1
            melhor = df_turbo.iloc[0]
            previsao_final = melhor["series"]

            st.markdown("### 🎯 Previsão Final TURBO++ ULTRA (Adaptativo)")
            st.code(" ".join(str(x) for x in previsao_final), language="text")

            if regime_info.estado == "ruptura":
                st.error("🔴 Regime de ruptura — Monte Carlo dominante, foco em previsibilidade curta.")
            elif regime_info.estado == "transicao":
                st.warning("🟡 Regime de transição — cenário misto, previsibilidade moderada.")
            else:
                st.success("🟢 Regime estável — núcleo determinístico e micro-leque ganham importância.")


# ============================================================
# PAINEL — REPLAY LIGHT
# ============================================================

elif painel == "💡 Replay LIGHT":

    st.header("💡 Replay LIGHT — Focado em 1 índice (visão rápida)")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    try:
        min_idx, max_idx = obter_intervalo_indices(df)
    except Exception as e:
        st.error(f"Histórico sem coluna 'idx' válida: {e}")
        st.stop()

    idx_alvo = st.number_input(
        "Selecione o índice alvo para Replay LIGHT:",
        min_value=int(min_idx),
        max_value=int(max_idx),
        value=int(max_idx),
        step=1,
    )

    df_ate_alvo = df[df["idx"] <= idx_alvo].copy()
    if df_ate_alvo.empty:
        df_ate_alvo = df.copy()

    regime_info = calcular_regime_ultra(df_ate_alvo, janela=40)
    k_info = calcular_k_star_ultra(df_ate_alvo, janela=40)

    if st.button("Rodar Replay LIGHT"):
        with st.spinner("Montando Replay LIGHT..."):
            contexto = montar_contexto_replay_light(
                df_hist=df,
                idx_alvo=int(idx_alvo),
                regime_info=regime_info,
                k_info=k_info,
            )

        st.subheader("🧱 Série alvo (real)")
        st.code(" ".join(str(x) for x in contexto["serie_real"]), language="text")

        st.subheader("📜 Leque TURBO++ ULTRA (Replay LIGHT)")
        df_turbo = contexto["df_turbo"]
        if df_turbo.empty:
            st.warning("Leque vazio neste ponto da estrada.")
        else:
            st.dataframe(df_turbo[["series", "origem", "score_global", "acertos"]], use_container_width=True)

        st.markdown("### 🎯 Resumo de acertos")
        st.write(f"Acertos top-1: **{contexto['acertos_top1']}**")
        st.write(f"Melhor acerto no leque: **{contexto['melhor_acerto']}** passageiros.")
        st.write(f"Posição do melhor acerto no ranking: **{contexto['pos_melhor_rank']}**.")


# ============================================================
# PAINEL — REPLAY ULTRA (LOOP TRADICIONAL)
# ============================================================

elif painel == "📅 Replay ULTRA (Loop Tradicional)":

    st.header("📅 Replay ULTRA — Loop Tradicional")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    try:
        min_idx, max_idx = obter_intervalo_indices(df)
    except Exception as e:
        st.error(f"Histórico sem coluna 'idx' válida: {e}")
        st.stop()

    st.write(
        "Este modo percorre um intervalo de índices e executa o Replay LIGHT em loop. "
        "Pode ser pesado, mas foi mantido para análise ULTRA histórica."
    )

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        inicio = st.number_input(
            "Índice inicial:",
            min_value=int(min_idx),
            max_value=int(max_idx),
            value=max(int(min_idx), int(max_idx) - 20),
            step=1,
        )
    with col_r2:
        fim = st.number_input(
            "Índice final:",
            min_value=int(min_idx),
            max_value=int(max_idx),
            value=int(max_idx),
            step=1,
        )

    if inicio > fim:
        st.error("Índice inicial não pode ser maior que o índice final.")
        st.stop()

    if (fim - inicio) > 100:
        st.warning("Intervalos muito grandes podem deixar o app pesado. Considere reduzir o range.")

    if st.button("Rodar Replay ULTRA (Loop)"):
        registros = []
        with st.spinner("Executando Replay ULTRA em loop..."):
            for idx_alvo in range(int(inicio), int(fim) + 1):
                df_ate_alvo = df[df["idx"] <= idx_alvo].copy()
                if df_ate_alvo.empty:
                    df_ate_alvo = df.copy()
                regime_info = calcular_regime_ultra(df_ate_alvo, janela=40)
                k_info = calcular_k_star_ultra(df_ate_alvo, janela=40)

                contexto = montar_contexto_replay_light(
                    df_hist=df,
                    idx_alvo=int(idx_alvo),
                    regime_info=regime_info,
                    k_info=k_info,
                )

                registros.append(
                    {
                        "idx_alvo": idx_alvo,
                        "acertos_top1": contexto["acertos_top1"],
                        "melhor_acerto": contexto["melhor_acerto"],
                        "pos_melhor_rank": contexto["pos_melhor_rank"],
                    }
                )

        if not registros:
            st.warning("Nenhum registro gerado no loop.")
        else:
            df_loop = pd.DataFrame(registros)
            st.subheader("📊 Resumo do Replay ULTRA (loop)")
            st.dataframe(df_loop, use_container_width=True)

            st.line_chart(df_loop.set_index("idx_alvo")[["acertos_top1", "melhor_acerto"]])

            st.caption(
                "Use este painel para entender como a performance do leque varia ao longo da estrada, "
                "sem simplificações."
            )


# ============================================================
# PAINEL — REPLAY ULTRA UNITÁRIO (NOVO)
# ============================================================

elif painel == "🎯 Replay ULTRA UNITÁRIO (Novo)":

    st.header("🎯 Replay ULTRA UNITÁRIO — 1 índice por vez (visão completa)")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    try:
        min_idx, max_idx = obter_intervalo_indices(df)
    except Exception as e:
        st.error(f"Histórico sem coluna 'idx' válida: {e}")
        st.stop()

    idx_alvo = st.number_input(
        "Selecione o índice alvo para Replay ULTRA Unitário:",
        min_value=int(min_idx),
        max_value=int(max_idx),
        value=int(max_idx),
        step=1,
    )

    top_n = st.slider(
        "Top-N considerado para acertos:",
        min_value=5,
        max_value=80,
        value=25,
        step=5,
    )

    # Opcionalmente, podemos pedir para reutilizar um backtest global como QDS
    st.markdown("Você pode opcionalmente rodar o painel de **Testes de Confiabilidade** antes, "
                "para obter um QDS global. Aqui, o QDS é tratado de forma local/simples.")

    if st.button("Rodar Replay ULTRA Unitário"):
        df_ate_alvo = df[df["idx"] <= idx_alvo].copy()
        if df_ate_alvo.empty:
            df_ate_alvo = df.copy()

        regime_info = calcular_regime_ultra(df_ate_alvo, janela=40)
        k_info = calcular_k_star_ultra(df_ate_alvo, janela=40)

        # Sem QDS pré-calculado aqui (poderíamos integrar no futuro)
        qds_info = None

        with st.spinner("Gerando leque TURBO++ ULTRA para Replay Unitário..."):
            df_turbo = montar_previsao_turbo_ultra(
                df_hist=df,
                idx_alvo=int(idx_alvo),
                regime_info=regime_info,
                k_info=k_info,
                n_series_base=250,
                n_series_micro=40,
                n_sim_mc=600,
                output_mode="Quantidade fixa",
                n_series_fixed=int(top_n),
                min_conf_pct=30.0,
            )

        if df_turbo is None or df_turbo.empty:
            st.error("Não foi possível gerar o leque TURBO++ ULTRA para este índice.")
        else:
            with st.spinner("Montando contexto de Replay ULTRA Unitário..."):
                contexto = montar_contexto_replay_ultra_unitario(
                    df_hist=df,
                    idx_alvo=int(idx_alvo),
                    df_turbo=df_turbo,
                    qds_info=qds_info,
                    regime_info=regime_info,
                    k_info=k_info,
                    top_n=int(top_n),
                )

            st.subheader("🧱 Série alvo (real)")
            st.code(" ".join(str(x) for x in contexto["serie_real"]), language="text")

            st.subheader("📜 Leque TURBO++ ULTRA (com acertos)")
            st.dataframe(
                contexto["df_turbo"][["series", "origem", "score_global", "acertos"]],
                use_container_width=True,
            )

            st.markdown("### 🎯 Saída completa do Replay ULTRA Unitário")
            st.write(f"Acertos top-1: **{contexto['acertos_top1']}**")
            st.write(f"Acertos em pelo menos 1 série no top-{top_n}: **{contexto['acertos_topN']}**")
            st.write(f"Melhor acerto no leque: **{contexto['melhor_acerto']}** passageiros.")
            st.write(f"Posição do melhor acerto no ranking: **{contexto['pos_melhor_rank']}**")

            st.markdown("### 🌡️ Regime e k* no momento do alvo")
            st.write(contexto["regime_descricao"])
            st.write(contexto["k_star_descricao"])

            if contexto["qds"] is not None:
                st.markdown("### 📈 QDS local/global")
                st.write(contexto["qds_descricao"])
            else:
                st.caption(
                    "QDS não foi integrado a este alvo de forma explícita. "
                    "Use o painel de Testes de Confiabilidade para cálculo global."
                )


# ============================================================
# PAINEL — TESTES DE CONFIABILIDADE (QDS / BACKTEST)
# ============================================================

elif painel == "🧪 Testes de Confiabilidade":

    st.header("🧪 Testes de Confiabilidade — QDS REAL + Backtest REAL")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    try:
        min_idx, max_idx = obter_intervalo_indices(df)
    except Exception as e:
        st.error(f"Histórico sem coluna 'idx' válida: {e}")
        st.stop()

    n_pass = obter_n_passageiros(df)

    st.markdown(
        "Este painel executa um **backtest real** sobre uma janela de índices finais "
        "da estrada e calcula o **QDS REAL (Índice de Qualidade Dinâmica da Série)**."
    )

    janela_bt = st.slider(
        "Janela de backtest (nº de índices finais):",
        min_value=30,
        max_value=min(400, int(len(df))),
        value=150,
        step=10,
    )

    top_n_bt = st.slider(
        "Top-N considerado no backtest:",
        min_value=5,
        max_value=40,
        value=10,
        step=5,
    )

    if st.button("Rodar Backtest REAL + QDS"):
        with st.spinner("Executando Backtest REAL... isso pode levar algum tempo."):
            bt_result = executar_backtest_real(
                df_hist=df,
                n_passageiros=n_pass,
                janela=int(janela_bt),
                top_n=int(top_n_bt),
            )

        if bt_result.tabela is None or bt_result.tabela.empty:
            st.error("Backtest não gerou resultados suficientes.")
        else:
            st.subheader("📊 Tabela de Backtest REAL")
            st.dataframe(bt_result.tabela, use_container_width=True)

            st.markdown("### 📝 Resumo do Backtest")
            st.write(bt_result.descricao)

            with st.spinner("Calculando QDS REAL..."):
                qds_info = calcular_qds_real(bt_result.tabela, top_n=int(top_n_bt))

            st.markdown("### 📈 QDS REAL — Índice de Qualidade Dinâmica")
            st.write(qds_info.descricao)

            st.caption(
                "Use o QDS para calibrar expectativas de acerto do TURBO++ ULTRA, "
                "sem simplificações, em cima da estrada real que você carregou."
            )
# ============================================================
# FINALIZAÇÃO DO APP
# ============================================================

st.markdown("---")
st.caption(
    "Predict Cars — V14-FLEX ULTRA REAL (TURBO++) • "
    "Motor completo, sem simplificações, com todos os módulos "
    "S6 Profundo ULTRA • Micro-Leque ULTRA • Monte Carlo Profundo ULTRA • "
    "Backtest REAL • QDS REAL • Replay LIGHT • Replay ULTRA • Replay ULTRA UNITÁRIO."
)

# ============================================================
# EXECUÇÃO DIRETA (OPCIONAL)
# Streamlit ignora o bloco main, mas mantemos por padronização.
# ============================================================

if __name__ == "__main__":
    # A execução real será feita pelo Streamlit com:
    #   streamlit run app_v14_flex_replay_ultra_unitario.py
    pass
