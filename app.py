# =========================================================
# PREDICT CARS V13.8-TURBO — app.py CONSOLIDADO
# Versão completa (BLOCOS 1 a 14 integrados)
# =========================================================

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURAÇÃO GLOBAL DO APP
# =========================================================

st.set_page_config(
    page_title="Predict Cars V13.8-TURBO",
    page_icon="🚗",
    layout="wide",
)

# =========================================================
# TIPOS BÁSICOS
# =========================================================

@dataclass
class RegimeState:
    nome: str
    dispersao: float
    amplitude: float
    vibracao: float
    pares: List[Tuple[int, int]]

def formatar_serie_para_texto(s):
    # Caso 1 — já é string
    if isinstance(s, str):
        # Tenta dividir em números se for algo como "8 15 23"
        partes = [p.strip() for p in s.replace(",", " ").split() if p.strip()]
        numeros = []
        for p in partes:
            try:
                numeros.append(str(int(p)))
            except:
                continue
        return " ".join(numeros)

    # Caso 2 — lista ou tupla
    if isinstance(s, (list, tuple)):
        numeros = []
        for x in s:
            try:
                numeros.append(str(int(x)))
            except:
                continue
        return " ".join(numeros)

    # Caso 3 — qualquer outra coisa
    try:
        return str(int(s))
    except:
        return ""


# =========================================================
# FUNÇÕES BÁSICAS — PARSING DO HISTÓRICO
# =========================================================

def parse_line_to_series(line: str) -> Optional[List[int]]:
    """
    Converte uma linha do arquivo em lista de inteiros.
    Aceita formatos:
    - C1234; n1; n2; n3; n4; n5; k
    - n1; n2; n3; n4; n5; k
    - n1 n2 n3 n4 n5 k
    """
    if not line.strip():
        return None

    # Troca vírgulas por ponto e vírgula, normaliza separadores
    line = line.replace(",", ";").replace("\t", ";")
    # Se não houver ';', tenta separar por espaço
    if ";" not in line:
        parts = line.split()
    else:
        parts = [p.strip() for p in line.split(";") if p.strip()]

    if not parts:
        return None

    # Ignora prefixos tipo 'C1234'
    if parts[0].upper().startswith("C") and len(parts) > 1:
        parts = parts[1:]

    # Espera pelo menos 6 números (5+1 ou 6+1, etc.)
    nums = []
    for p in parts:
        try:
            nums.append(int(p))
        except ValueError:
            # ignora tokens não numéricos
            continue

    if len(nums) < 6:
        return None

    # Considera sempre os 6 primeiros como passageiros, o último como k
    # Se houver mais de 7 colunas, corta.
    return nums[:7]


def history_to_dataframe(text: str) -> pd.DataFrame:
    """
    Converte texto bruto em DataFrame no formato interno:
    colunas: n1..n6, k
    Sem índice Cxxxx (não é necessário para o app).
    """
    linhas = text.splitlines()
    registros = []

    for line in linhas:
        serie = parse_line_to_series(line)
        if serie is None:
            continue
        # se vier exatamente 6, assume k = 0
        if len(serie) == 6:
            serie.append(0)
        # garante tamanho 7
        if len(serie) > 7:
            serie = serie[:7]
        registros.append(serie)

    if not registros:
        return pd.DataFrame(columns=["n1", "n2", "n3", "n4", "n5", "n6", "k"])

    df = pd.DataFrame(
        registros,
        columns=["n1", "n2", "n3", "n4", "n5", "n6", "k"],
    )
    return df


# =========================================================
# FUNÇÕES BÁSICAS — ESTADO DA ESTRADA (REGIME)
# =========================================================

def calcular_regime(df: pd.DataFrame) -> Optional[RegimeState]:
    """
    Estima o regime atual com base nas últimas linhas do histórico.
    Heurística simplificada:
    - dispersão média
    - amplitude média
    - vibração (variação da dispersão)
    - pares frequentes nas últimas linhas
    """
    if df is None or df.empty:
        return None

    # Considera um trecho recente (ex.: últimas 80 linhas ou menos)
    trecho = df.tail(min(80, len(df)))
    numeros = trecho[["n1", "n2", "n3", "n4", "n5", "n6"]].values

    dispersoes = np.std(numeros, axis=1)
    amplitudes = np.max(numeros, axis=1) - np.min(numeros, axis=1)
    vib = np.abs(np.diff(dispersoes)).mean() if len(dispersoes) > 1 else 0.0

    disp_med = float(np.mean(dispersoes))
    amp_med = float(np.mean(amplitudes))

    # Geração simples de pares
    contagem_pares: Dict[Tuple[int, int], int] = {}
    for linha in numeros:
        linha_ordenada = sorted(set(linha.tolist()))
        for i in range(len(linha_ordenada)):
            for j in range(i + 1, len(linha_ordenada)):
                par = (linha_ordenada[i], linha_ordenada[j])
                contagem_pares[par] = contagem_pares.get(par, 0) + 1

    pares_ativos = sorted(
        contagem_pares.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:10]
    pares_ativos = [p[0] for p in pares_ativos]

    # Regra simples para nome do regime
    if disp_med < 10 and amp_med < 25 and vib < 3:
        nome = "Resiliente"
    elif disp_med < 16 and amp_med < 35:
        nome = "Intermediário"
    else:
        nome = "Turbulento"

    return RegimeState(
        nome=nome,
        dispersao=disp_med,
        amplitude=amp_med,
        vibracao=float(vib),
        pares=pares_ativos,
    )


# =========================================================
# INICIALIZAÇÃO DE ESTADO (session_state)
# =========================================================

def ensure_session_defaults():
    """
    Garante chaves básicas no session_state.
    """
    defaults = {
        "df": pd.DataFrame(),
        "regime_state": None,
        "idx_result": pd.DataFrame(),
        "nucleo_ipf": None,
        "nucleo_ipo": None,
        "ajustes_log": [],
        "dependencias": None,
        "s6_df": pd.DataFrame(),
        "mc_df": pd.DataFrame(),
        "backtest_interno": pd.DataFrame(),
        "btf_raw": pd.DataFrame(),
        "leque_turbo": {},
        "logs_tecnicos": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


ensure_session_defaults()

# =========================================================
# LAYOUT PRINCIPAL — CABEÇALHO E ENTRADA DE HISTÓRICO
# =========================================================

st.title("🚗 Predict Cars V13.8-TURBO")
st.caption("Modo Ultra-Híbrido TURBO — Núcleo Resiliente + Leque Estrutural + Backtest do Futuro")

st.markdown("### 📥 Entrada de Histórico")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Carregar arquivo de histórico (.txt ou .csv)",
        type=["txt", "csv"],
    )

with col2:
    text_input = st.text_area(
        "Ou colar o histórico aqui (linhas com 6 passageiros + k)",
        height=200,
    )

df: pd.DataFrame

if uploaded_file is not None:
    raw_bytes = uploaded_file.read()
    raw_text = raw_bytes.decode("utf-8", errors="ignore")
    df = history_to_dataframe(raw_text)
    st.session_state["df"] = df
elif text_input.strip():
    df = history_to_dataframe(text_input)
    st.session_state["df"] = df
else:
    df = st.session_state.get("df", pd.DataFrame())

if df is not None and not df.empty:
    st.success(f"Histórico carregado com {len(df)} séries.")
    st.dataframe(df.tail(10), use_container_width=True)
else:
    st.info("Nenhum histórico válido carregado ainda.")

# =========================================================
# CÁLCULO DO ESTADO DA ESTRADA (REGIME)
# =========================================================

if not df.empty:
    regime_state = calcular_regime(df)
    st.session_state["regime_state"] = regime_state
else:
    regime_state = None

# =========================================================
# CONTROLES GERAIS (SIDEBAR)
# =========================================================

st.sidebar.markdown("## ⚙️ Controles Gerais")

# Modo de geração do leque final
output_mode = st.sidebar.radio(
    "Modo de geração do Leque:",
    options=[
        "Automático (por regime)",
        "Quantidade fixa",
        "Confiabilidade mínima",
    ],
    index=0,
)

n_series_fixed = st.sidebar.slider(
    "Quantidade total de séries (se modo for 'Quantidade fixa')",
    min_value=5,
    max_value=25,
    value=12,
)

min_conf_pct = st.sidebar.slider(
    "Confiabilidade mínima (%) (se modo for 'Confiabilidade mínima')",
    min_value=30,
    max_value=85,
    value=55,
)

modo_k = st.sidebar.radio(
    "Modo k:",
    ["Usar k atual (k*)", "Usar k preditivo (k̂)"],
    index=0,
)
if modo_k == "Usar k atual (k*)":
    k_ativo = k_estado
else:
    k_ativo = k_pred

# =========================================================
# MENU DE NAVEGAÇÃO (PAINÉIS PRINCIPAIS)
# =========================================================

st.sidebar.markdown("## 📂 Navegação")

painel = st.sidebar.radio(
    "Escolha o painel:",
    [
        "Histórico",
        "Estado Atual",
        "IDX Avançado",
        "Núcleo IPF / IPO",
        "Ajustes (ASB / ADN / ICA / HLA)",
        "Dependências Ocultas",
        "S6 Profundo",
        "Monte Carlo Profundo",
        "Backtest Interno",
        "Backtest do Futuro",
        "Leque TURBO",
        "Saída Final Controlada",
        "S1–S5 + Ajuste Fino",
        "Logs Técnicos",
        "Diagnóstico Profundo",
        "Exportar Resultados",
        "Exportar Sessão Completa",
        "Comparação k* vs k̂",
    ],
    index=0,
)


# A partir da PARTE 2/7, cada painel será implementado
# com base na variável `painel` e no DataFrame `df`.
# =========================================================
# PAINEL: HISTÓRICO
# =========================================================

if painel == "Histórico":
    st.markdown("## 📜 Histórico Carregado")
    if df.empty:
        st.warning("Nenhum histórico carregado.")
    else:
        st.dataframe(df, use_container_width=True)
        st.markdown("### 🔍 Últimas 15 séries")
        st.dataframe(df.tail(15), use_container_width=True)
    st.stop()

# =========================================================
# PAINEL: ESTADO ATUAL (REGIME)
# =========================================================

if painel == "Estado Atual":
    st.markdown("## 🌡️ Estado da Estrada (Regime)")
    if regime_state is None:
        st.warning("Regime não pôde ser calculado — carregue histórico válido.")
        st.stop()

    # =========================================================
    # SENSOR AMBIENTAL k* — ESTADO ATUAL (MODO SIMPLES)
    # =========================================================
    try:
        # Histórico completo
        df_hist = df.copy()

        # Função para renomear colunas corretamente
        if df_hist.shape[1] >= 8:
            df_hist.columns = ["id", "n1", "n2", "n3", "n4", "n5", "n6", "k"]
        else:
            # fallback: se vier sem ID
            if df_hist.shape[1] == 7:
                df_hist.columns = ["n1", "n2", "n3", "n4", "n5", "n6", "k"]
                df_hist["id"] = None
            else:
                df_hist["k"] = 0  # pior caso

        # Últimos valores de k
        ultimos_k = df_hist["k"].tail(5).tolist()

        # Detecta ruptura recente (k != 0)
        ruptura_recente = (df_hist["k"].iloc[-1] != 0)

        # Lógica do sensor
        if ruptura_recente:
            k_estado = "critico"
        else:
            if any(k != 0 for k in ultimos_k):
                k_estado = "atencao"
            else:
                k_estado = "estavel"
        k_pred = calcular_k_pred(k_estado, df_hist)

        texto_k_atual = contexto_k_texto(k_estado, prefixo="k*")
        texto_k_pred  = contexto_k_texto(k_pred,    prefixo="k̂")
        
        # Exibir badge ambiental no Estado Atual
        st.markdown("### 🌡️ Estado Ambiental da Estrada (k*) — Estado Atual")
        st.markdown(texto_k_atual)
        st.markdown(texto_k_pred)
    except Exception as e:
        st.error(f"Erro no sensor k* (Estado Atual): {e}")
    st.subheader("Resumo do Regime Atual")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Regime", regime_state.nome)
    c2.metric("Dispersão", f"{regime_state.dispersao:.2f}")
    c3.metric("Amplitude", f"{regime_state.amplitude:.2f}")
    c4.metric("Vibração", f"{regime_state.vibracao:.2f}")

    st.markdown("### 🔗 Pares Ativos (mais frequentes recentemente)")
    pares_df = pd.DataFrame(regime_state.pares, columns=["p1", "p2"])
    st.dataframe(pares_df, use_container_width=True)

    st.stop()

# =========================================================
# MÓDULO IDX AVANÇADO — SIMILARIDADE E NÚCLEOS INICIAIS
# =========================================================

def calcular_similaridade(linha_a: np.ndarray, linha_b: np.ndarray) -> float:
    """
    Similaridade simples entre duas séries:
    - 1 / (1 + soma das distâncias absolutas)
    Quanto maior, mais parecido.
    """
    return 1.0 / (1.0 + np.sum(np.abs(linha_a - linha_b)))


def executar_idx_avancado(df: pd.DataFrame, n_top: int = 40) -> pd.DataFrame:
    """
    Identifica as séries historicamente mais parecidas com a última série.
    Retorna DataFrame com:
    - índice original
    - similaridade
    - n1..n6
    """
    if df.empty:
        return pd.DataFrame()

    ultima = df[["n1", "n2", "n3", "n4", "n5", "n6"]].values[-1]
    similares = []

    for idx in range(len(df) - 1):
        atual = df.iloc[idx][["n1", "n2", "n3", "n4", "n5", "n6"]].values
        sim = calcular_similaridade(ultima, atual)
        similares.append(
            (idx, sim) + tuple(df.iloc[idx][["n1", "n2", "n3", "n4", "n5", "n6"]].values)
        )

    cols = ["idx", "similaridade", "n1", "n2", "n3", "n4", "n5", "n6"]
    df_sim = pd.DataFrame(similares, columns=cols)
    df_sim = df_sim.sort_values("similaridade", ascending=False).head(n_top)

    return df_sim


# =========================================================
# MÓDULOS IPF / IPO BÁSICOS (primeira camada)
# =========================================================

def extrair_ipf(df_idx: pd.DataFrame) -> Optional[List[int]]:
    """
    IPF (IDX Puro Focado simplificado):
    Extrai núcleo como média ponderada dos top-idx por similaridade.
    """
    if df_idx.empty:
        return None

    pesos = df_idx["similaridade"].values
    nums = df_idx[["n1", "n2", "n3", "n4", "n5", "n6"]].values

    media = np.average(nums, weights=pesos, axis=0)
    nucleo = [int(round(x)) for x in media]
    # Garante números distintos
    nucleo = list(sorted(set(nucleo)))
    while len(nucleo) < 6:
        nucleo.append(nucleo[-1] + 1)
    return nucleo[:6]


def aplicar_ipo_profundo(nucleo: List[int], regime: RegimeState) -> List[int]:
    """
    IPO Profundo simplificado:
    - Ajusta faixa dominante;
    - Suaviza extremos incoerentes com regime;
    - Garante coerência estrutural mínima.
    """
    if nucleo is None:
        return []

    ordenado = sorted(nucleo)
    faixas = np.array(ordenado)

    # Ajuste leve conforme regime
    if regime.nome == "Resiliente":
        faixas = np.clip(faixas, 1, 70)
    elif regime.nome == "Intermediário":
        faixas = np.clip(faixas, 5, 75)
    else:  # Turbulento
        faixas = np.clip(faixas, 10, 80)

    return sorted(set(int(x) for x in faixas))[:6]


# =========================================================
# PAINEL: IDX AVANÇADO
# =========================================================

if painel == "IDX Avançado":
    st.markdown("## 🔎 IDX Avançado")
    if df.empty:
        st.warning("Nenhum histórico carregado.")
        st.stop()

    df_idx = executar_idx_avancado(df)
    st.session_state["idx_result"] = df_idx

    st.markdown("### Top séries similares (IDX)")
    st.dataframe(df_idx, use_container_width=True)
    # =========================================================
    # SENSOR AMBIENTAL k* — IDX AVANÇADO (MODO SIMPLES)
    # =========================================================
    try:
        # Histórico completo
        df_hist = df.copy()

        # Função para renomear colunas corretamente
        if df_hist.shape[1] >= 8:
            df_hist.columns = ["id", "n1", "n2", "n3", "n4", "n5", "n6", "k"]
        else:
            if df_hist.shape[1] == 7:
                df_hist.columns = ["n1", "n2", "n3", "n4", "n5", "n6", "k"]
                df_hist["id"] = None
            else:
                df_hist["k"] = 0

        # Últimos valores de k
        ultimos_k = df_hist["k"].tail(5).tolist()

        # Detecta ruptura recente
        ruptura_recente = (df_hist["k"].iloc[-1] != 0)

        # Lógica do sensor
        if ruptura_recente:
            k_estado = "critico"
        else:
            if any(k != 0 for k in ultimos_k):
                k_estado = "atencao"
            else:
                k_estado = "estavel"

        # Exibir badge no IDX
        st.markdown("### 🌡️ Estado Ambiental da Estrada (k*) — IDX Avançado")
        st.markdown(contexto_k_texto(k_estado, prefixo="k*"))
        st.markdown(texto_k_pred)
    
    except Exception as e:
        st.error(f"Erro no sensor k* (IDX Avançado): {e}")

    st.stop()
# =========================================================
# FUNÇÕES DE AJUSTE (ASB / ADN / ICA / HLA)
# =========================================================

def aplicar_asb_antibias(nucleo: List[int], regime: RegimeState) -> List[int]:
    """
    ASB — Anti-SelfBias simplificado:
    - Permite repetição somente quando coerente com o regime;
    - Evita compressão artificial de faixa.
    """
    if nucleo is None:
        return []

    base = sorted(nucleo)

    # Evita compressão artificial
    diffs = np.diff(base)
    if np.any(diffs < 2):
        base = [base[0]] + [base[i] + 2 for i in range(1, len(base))]

    # Regras por regime
    if regime.nome == "Resiliente":
        return base  # repetição natural é permitida
    elif regime.nome == "Intermediário":
        # leve expansão
        return [min(80, x + 1) for x in base]
    else:
        # turbulência → evitar repetições e zonas muito estreitas
        return sorted(set([min(80, x + 2) for x in base]))[:6]


def aplicar_adn(nucleo: List[int], modo: str = "leve") -> List[int]:
    """
    ADN (Ajuste Dinâmico):
    - leve → corrige ruídos sem alterar essência
    - médio → substitui elementos fracos
    - profundo → reavalia microestruturas (simplificado)
    """
    if nucleo is None:
        return []

    base = sorted(nucleo)

    if modo == "leve":
        return base

    if modo == "médio":
        # substitui o menor elemento por +1
        base[0] = min(80, base[0] + 1)
        return sorted(base)

    if modo == "profundo":
        # desloca todo o núcleo para a faixa seguinte
        return sorted([min(80, x + 2) for x in base])

    return base


def aplicar_ica_profundo(nucleo: List[int]) -> List[int]:
    """
    ICA Profundo (Iterative Core Adjustment):
    - reforça coerência entre posições adjacentes;
    - evita saltos incoerentes.
    """
    if nucleo is None:
        return []

    base = sorted(nucleo)

    for i in range(1, len(base)):
        if base[i] - base[i - 1] > 15:
            base[i] = base[i - 1] + 10

    return sorted(set(base))[:6]


def aplicar_hla_profundo(nucleo: List[int]) -> List[int]:
    """
    HLA Profundo:
    - poda incoerências de dispersão;
    - reequilibra extremos.
    """
    if nucleo is None:
        return []

    base = sorted(nucleo)

    # força extremos a serem coerentes
    if base[-1] - base[0] > 60:
        base[-1] = base[0] + 45

    return sorted(set(base))[:6]


# =========================================================
# PIPELINE COMPLETO DO NÚCLEO (IPF + IPO + ASB + ADN + ICA + HLA)
# =========================================================

def gerar_nucleo_resiliente(df: pd.DataFrame, regime: RegimeState) -> List[int]:
    """
    Pipeline resumido para gerar o Núcleo Resiliente completo.
    """
    if df.empty:
        return []

    # Etapa 1: IDX
    df_idx = executar_idx_avancado(df)
    ipf = extrair_ipf(df_idx)
    if ipf is None:
        return []

    # Etapa 2: IPO
    ipo = aplicar_ipo_profundo(ipf, regime)

    # Etapa 3: ASB (Anti-SelfBias)
    asb = aplicar_asb_antibias(ipo, regime)

    # Etapa 4: ADN (modo médio por padrão)
    adn = aplicar_adn(asb, modo="médio")

    # Etapa 5: ICA / HLA
    ica = aplicar_ica_profundo(adn)
    hla = aplicar_hla_profundo(ica)

    return sorted(set(hla))[:6]


# =========================================================
# PAINEL: NÚCLEO IPF / IPO
# =========================================================

if painel == "Núcleo IPF / IPO":
    st.markdown("## 🧬 Núcleo IPF / IPO")
    if df.empty:
        st.warning("Carregue um histórico válido.")
        st.stop()

    df_idx = st.session_state.get("idx_result", executar_idx_avancado(df))
    ipf = extrair_ipf(df_idx)
    ipo = aplicar_ipo_profundo(ipf, regime_state)

    st.session_state["nucleo_ipf"] = ipf
    st.session_state["nucleo_ipo"] = ipo

    st.markdown("### 🔹 Núcleo IPF (puro focado)")
    st.write(ipf)

    st.markdown("### 🔹 Núcleo IPO (otimizado)")
    st.write(ipo)

    st.stop()


# =========================================================
# PAINEL: AJUSTES (ASB / ADN / ICA / HLA)
# =========================================================

if painel == "Ajustes (ASB / ADN / ICA / HLA)":
    st.markdown("## 🔧 Ajustes do Núcleo (ASB, ADN, ICA, HLA)")

    if df.empty:
        st.warning("Carregue um histórico antes de visualizar ajustes.")
        st.stop()

    # Etapas
    df_idx = st.session_state.get("idx_result", executar_idx_avancado(df))
    ipf = extrair_ipf(df_idx)
    ipo = aplicar_ipo_profundo(ipf, regime_state)

    asb = aplicar_asb_antibias(ipo, regime_state)
    adn = aplicar_adn(asb, modo="médio")
    ica = aplicar_ica_profundo(adn)
    hla = aplicar_hla_profundo(ica)

    st.markdown("### 🔹 IPF → IPO → ASB → ADN → ICA → HLA")
    st.write({
        "IPF": ipf,
        "IPO": ipo,
        "ASB": asb,
        "ADN (médio)": adn,
        "ICA": ica,
        "HLA": hla,
    })

    st.stop()
# =========================================================
# DEPENDÊNCIAS OCULTAS
# =========================================================

def calcular_dependencias_ocultas(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Dependências ocultas (versão simplificada):
    - pares naturais
    - pares ocultos
    - pesos leves / médios / pesados
    - vibração histórica
    """
    if df.empty:
        return {}

    numeros = df[["n1", "n2", "n3", "n4", "n5", "n6"]].values

    # Contagem simples de pares
    contagem: Dict[Tuple[int, int], int] = {}
    for linha in numeros:
        linha_ord = sorted(set(linha.tolist()))
        for i in range(len(linha_ord)):
            for j in range(i + 1, len(linha_ord)):
                par = (linha_ord[i], linha_ord[j])
                contagem[par] = contagem.get(par, 0) + 1

    pares_ordenados = sorted(contagem.items(), key=lambda x: x[1], reverse=True)
    naturais = pares_ordenados[:15]
    ocultos = pares_ordenados[15:40]

    # vibração histórica simples
    dispersoes = np.std(numeros, axis=1)
    vibracao = float(np.mean(np.abs(np.diff(dispersoes)))) if len(dispersoes) > 1 else 0.0

    dependencias = {
        "pares_naturais": naturais,
        "pares_ocultos": ocultos,
        "vibracao": vibracao,
    }

    return dependencias


# =========================================================
# PAINEL: DEPENDÊNCIAS OCULTAS
# =========================================================

if painel == "Dependências Ocultas":
    st.markdown("## 🧩 Dependências Ocultas")
    if df.empty:
        st.warning("Carregue histórico primeiro.")
        st.stop()

    dep = calcular_dependencias_ocultas(df)
    st.session_state["dependencias"] = dep

    st.markdown("### 🔸 Pares Naturais")
    df_nat = pd.DataFrame(dep["pares_naturais"], columns=["par", "freq"])
    st.dataframe(df_nat, use_container_width=True)

    st.markdown("### 🔸 Pares Ocultos")
    df_oc = pd.DataFrame(dep["pares_ocultos"], columns=["par", "freq"])
    st.dataframe(df_oc, use_container_width=True)

    st.markdown("### 🔸 Vibração Histórica")
    st.write(dep["vibracao"])

    st.stop()


# =========================================================
# MÓDULO S6 — MODOS DE 6 ACERTOS PROFUNDO
# =========================================================

def gerar_s6_profundo(df: pd.DataFrame, nucleo: List[int], regime: RegimeState) -> pd.DataFrame:
    """
    S6 Profundo simplificado:
    - gera séries vizinhas do núcleo
    - usa microperturbações coerentes com o regime
    """
    if df.empty or nucleo is None:
        return pd.DataFrame()

    base = sorted(nucleo)
    candidatos = []

    for desloc in [-3, -2, -1, 1, 2, 3]:
        nova = [min(80, max(1, x + desloc)) for x in base]
        nova = sorted(set(nova))[:6]
        candidatos.append(nova)

    linhas = []
    for idx, serie in enumerate(candidatos):
        linhas.append([idx] + serie)

    cols = ["id", "n1", "n2", "n3", "n4", "n5", "n6"]
    return pd.DataFrame(linhas, columns=cols)


# =========================================================
# PAINEL: S6 PROFUNDO
# =========================================================

if painel == "S6 Profundo":
    st.markdown("## 🎯 S6 Profundo — Zonas de Convergência")
    if df.empty:
        st.warning("Carregue histórico primeiro.")
        st.stop()

    nucleo = gerar_nucleo_resiliente(df, regime_state)
    s6 = gerar_s6_profundo(df, nucleo, regime_state)

    st.session_state["s6_df"] = s6

    st.markdown("### 🔹 Núcleo Resiliente")
    st.write(nucleo)

    st.markdown("### 🔹 Séries S6 Geradas")
    st.dataframe(s6, use_container_width=True)

    st.stop()


# =========================================================
# MÓDULO MONTE CARLO PROFUNDO
# =========================================================

def gerar_monte_carlo(df: pd.DataFrame, nucleo: List[int], regime: RegimeState, n_sim=50) -> pd.DataFrame:
    """
    Monte Carlo Profundo:
    - perturba o núcleo de forma leve
    - gera variações coerentes com a estrada
    """
    if df.empty or nucleo is None:
        return pd.DataFrame()

    linhas = []
    for i in range(n_sim):
        var = []
        for x in nucleo:
            ruido = np.random.randint(-2, 3)
            novo = min(80, max(1, x + ruido))
            var.append(novo)
        var = sorted(set(var))[:6]
        linhas.append([i] + var)

    cols = ["sim_id", "n1", "n2", "n3", "n4", "n5", "n6"]
    return pd.DataFrame(linhas, columns=cols)


# =========================================================
# PAINEL: MONTE CARLO PROFUNDO
# =========================================================

if painel == "Monte Carlo Profundo":
    st.markdown("## 🎲 Monte Carlo Profundo")
    if df.empty:
        st.warning("Carregue histórico primeiro.")
        st.stop()

    nucleo = gerar_nucleo_resiliente(df, regime_state)
    mc = gerar_monte_carlo(df, nucleo, regime_state, n_sim=80)

    st.session_state["mc_df"] = mc

    st.markdown("### 🔹 Núcleo Resiliente Usado")
    st.write(nucleo)

    st.markdown("### 🔹 Simulações Monte Carlo")
    st.dataframe(mc, use_container_width=True)

    st.stop()
# =========================================================
# BACKTEST INTERNO (Simulação Retroativa)
# =========================================================

def executar_backtest_interno(df: pd.DataFrame, nucleo: List[int]) -> pd.DataFrame:
    """
    Backtest Interno:
    - testa o núcleo atual contra trechos passados semelhantes.
    - mede coerência estrutural retrospectiva (simplificado).
    """
    if df.empty or nucleo is None:
        return pd.DataFrame()

    ultimas = df.tail(80)[["n1", "n2", "n3", "n4", "n5", "n6"]].values
    nuc = np.array(nucleo)

    linhas = []
    for idx, linha in enumerate(ultimas):
        acertos = len(set(linha.tolist()) & set(nucleo))
        linhas.append([idx] + linha.tolist() + [acertos])

    cols = ["id", "n1", "n2", "n3", "n4", "n5", "n6", "acertos"]
    return pd.DataFrame(linhas, columns=cols)


# =========================================================
# PAINEL: BACKTEST INTERNO
# =========================================================

if painel == "Backtest Interno":
    st.markdown("## 🕒 Backtest Interno (Retroativo)")
    if df.empty:
        st.warning("Carregue histórico.")
        st.stop()

    nucleo = gerar_nucleo_resiliente(df, regime_state)
    bt = executar_backtest_interno(df, nucleo)

    st.session_state["backtest_interno"] = bt

    st.markdown("### 🔹 Núcleo Resiliente")
    st.write(nucleo)

    st.markdown("### 🔹 Backtest Interno — Últimas 80 séries")
    st.dataframe(bt, use_container_width=True)

    st.stop()


# =========================================================
# BACKTEST DO FUTURO (BTF)
# =========================================================

def executar_backtest_do_futuro(df: pd.DataFrame, nucleo: List[int]) -> pd.DataFrame:
    """
    Backtest do Futuro:
    - simula como o núcleo atual se comportaria em trechos passados longos.
    - valida coerência retrospectiva (BTF oficial).
    """
    if df.empty or nucleo is None:
        return pd.DataFrame()

    linhas = []
    for idx in range(len(df) - 1):
        real = df.iloc[idx + 1][["n1", "n2", "n3", "n4", "n5", "n6"]].values
        acertos = len(set(real.tolist()) & set(nucleo))
        linhas.append([idx] + real.tolist() + [acertos])

    cols = ["id", "real1", "real2", "real3", "real4", "real5", "real6", "acertos"]
    return pd.DataFrame(linhas, columns=cols)


# =========================================================
# PAINEL: BACKTEST DO FUTURO
# =========================================================

if painel == "Backtest do Futuro":
    st.markdown("## 🔮 Backtest do Futuro (Coerência Retroativa)")

    if df.empty:
        st.warning("Carregue o histórico.")
        st.stop()

    nucleo = gerar_nucleo_resiliente(df, regime_state)
    btf = executar_backtest_do_futuro(df, nucleo)

    st.session_state["btf_raw"] = btf

    st.markdown("### 🔹 Núcleo Usado no BTF")
    st.write(nucleo)

    st.markdown("### 🔹 Backtest do Futuro — Acurácia Estrutural")
    st.dataframe(btf.tail(50), use_container_width=True)

    st.stop()


# =========================================================
# LEQUE TURBO — BASE (pré-séries antes do controle final)
# =========================================================

def gerar_series_base(df: pd.DataFrame, regime: RegimeState) -> Dict[str, List[List[int]]]:
    """
    Gera:
    - Núcleo Final Turbo
    - Séries Premium iniciais
    - Séries Estruturais iniciais
    - Séries de Cobertura (básico)
    OBS: O refinamento final vem na PARTE 6 e 7.
    """
    nucleo = gerar_nucleo_resiliente(df, regime)

    if not nucleo:
        return {
            "nucleo": [],
            "premium": [],
            "estruturais": [],
            "cobertura": [],
        }

    base = sorted(nucleo)

    # Séries Premium (leve variação)
    premium = []
    for offset in [-1, 1]:
        p = [min(80, max(1, x + offset)) for x in base]
        p = sorted(set(p))[:6]
        premium.append(p)

    # Séries Estruturais (duas variantes)
    estruturais = []
    e1 = [min(80, x + 2) for x in base]
    e2 = [max(1, x - 2) for x in base]
    estruturais.append(sorted(set(e1))[:6])
    estruturais.append(sorted(set(e2))[:6])

    # Cobertura (perturbações mais amplas)
    cobertura = []
    c1 = [min(80, x + 3) for x in base]
    c2 = [max(1, x - 3) for x in base]
    cobertura.append(sorted(set(c1))[:6])
    cobertura.append(sorted(set(c2))[:6])

    return {
        "nucleo": base,
        "premium": premium,
        "estruturais": estruturais,
        "cobertura": cobertura,
    }


# =========================================================
# PAINEL: LEQUE TURBO (BASE)
# =========================================================

if painel == "Leque TURBO":
    st.markdown("## 🚀 Leque TURBO — Base Estrutural")

    if df.empty:
        st.warning("Carregue histórico.")
        st.stop()

    leque = gerar_series_base(df, regime_state)
    st.session_state["leque_turbo"] = leque

    st.markdown("### 🔹 Núcleo Final (base)")
    st.write(leque["nucleo"])

    st.markdown("### 🔹 Séries Premium (base)")
    st.write(leque["premium"])

    st.markdown("### 🔹 Séries Estruturais (base)")
    st.write(leque["estruturais"])

    st.markdown("### 🔹 Séries de Cobertura (base)")
    st.write(leque["cobertura"])

    st.stop()
# =========================================================
# LEQUE TURBO — FLAT TABLE + MODOS DE SAÍDA
# =========================================================

def build_flat_series_table(leque: Dict[str, Any]) -> pd.DataFrame:
    """
    Constrói uma tabela plana com todas as séries do Leque TURBO.
    Colunas:
    - category
    - series (lista de ints)
    - coherence (0 a 1)
    - expected_hits (1 a 6)
    """
    rows = []

    if not leque:
        return pd.DataFrame(columns=["category", "series", "coherence", "expected_hits"])

    # Núcleo principal
    nucleo = leque.get("nucleo", [])
    if nucleo:
        rows.append({
            "category": "NÚCLEO TURBO",
            "series": sorted(nucleo),
            "coherence": 0.90,
            "expected_hits": 4,
        })

    # Premium
    for serie in leque.get("premium", []):
        rows.append({
            "category": "Premium",
            "series": sorted(serie),
            "coherence": 0.82,
            "expected_hits": 3,
        })

    # Estruturais
    for serie in leque.get("estruturais", []):
        rows.append({
            "category": "Estrutural",
            "series": sorted(serie),
            "coherence": 0.74,
            "expected_hits": 2,
        })

    # Cobertura
    for serie in leque.get("cobertura", []):
        rows.append({
            "category": "Cobertura",
            "series": sorted(serie),
            "coherence": 0.65,
            "expected_hits": 1,
        })

    # S6 (se disponível em sessão)
    s6_df = st.session_state.get("s6_df", pd.DataFrame())
    if not s6_df.empty:
        for _, row in s6_df.iterrows():
            serie = [int(row[c]) for c in ["n1", "n2", "n3", "n4", "n5", "n6"]]
            rows.append({
                "category": "S6",
                "series": sorted(serie),
                "coherence": 0.87,
                "expected_hits": 5,
            })

    if not rows:
        return pd.DataFrame(columns=["category", "series", "coherence", "expected_hits"])

    flat_df = pd.DataFrame(rows)
    return flat_df


def limit_by_mode(
    flat_df: pd.DataFrame,
    regime: Optional[RegimeState],
    output_mode: str,
    n_series_fixed: int,
    min_conf_pct: int,
) -> pd.DataFrame:
    """
    Aplica os três modos de controle:
    - Automático (por regime)
    - Quantidade fixa
    - Confiabilidade mínima
    """
    if flat_df.empty:
        return flat_df

    df_sorted = flat_df.sort_values("coherence", ascending=False).reset_index(drop=True)

    # Modo automático por regime
    if output_mode.startswith("Automático"):
        if regime is None:
            target = min(12, len(df_sorted))
            return df_sorted.head(target)

        if regime.nome == "Resiliente":
            target = 10
            conf_min = 0.70
        elif regime.nome == "Intermediário":
            target = 12
            conf_min = 0.60
        else:  # Turbulento
            target = 15
            conf_min = 0.50

        filtrado = df_sorted[df_sorted["coherence"] >= conf_min].head(target)
        if filtrado.empty:
            return df_sorted.head(target)
        return filtrado

    # Modo quantidade fixa
    if output_mode.startswith("Quantidade"):
        return df_sorted.head(min(n_series_fixed, len(df_sorted)))

    # Modo confiabilidade mínima
    if output_mode.startswith("Confiabilidade"):
        thr = min_conf_pct / 100.0
        filtrado = df_sorted[df_sorted["coherence"] >= thr]
        if filtrado.empty:
            # fallback
            return df_sorted.head(8)
        return filtrado.reset_index(drop=True)

    return df_sorted



# =========================================================
# LOGS TÉCNICOS — REGISTRO E PAINEL
# =========================================================

def add_log(etapa: str, dados: Any):
    """
    Registra um log técnico no session_state["logs_tecnicos"].
    Pode ser chamado em qualquer etapa do pipeline.
    """
    if "logs_tecnicos" not in st.session_state:
        st.session_state["logs_tecnicos"] = []
    st.session_state["logs_tecnicos"].append(
        {
            "etapa": etapa,
            "dados": dados,
        }
    )


if painel == "Logs Técnicos":
    st.markdown("## 🧰 Logs Técnicos — Pipeline V13.8-TURBO")

    logs = st.session_state.get("logs_tecnicos", [])

    if not logs:
        st.info("Nenhum log técnico registrado ainda.")
        st.stop()

    for registro in logs:
        with st.expander(f"Etapa: {registro['etapa']}"):
            st.write(registro["dados"])

    st.stop()


# =========================================================
# DIAGNÓSTICO PROFUNDO — GRÁFICOS E ESTABILIDADE
# =========================================================

def plot_line(data, title, ylabel):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(data)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Índice")
    ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig)


def plot_hist(data, title, xlabel):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(data, bins=20, edgecolor="black", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequência")
    ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig)


def calcular_indice_estabilidade(regime_state: Optional[RegimeState]) -> Optional[float]:
    """
    Índice composto de estabilidade da estrada:
    ~1.0 → muito estável
    ~0.5 → intermediário
    ~0.0 → instável / turbulento
    """
    if not regime_state:
        return None

    disp_peso = max(0.0, 1.0 - regime_state.dispersao / 40.0)
    amp_peso = max(0.0, 1.0 - regime_state.amplitude / 60.0)
    vib_peso = max(0.0, 1.0 - regime_state.vibracao / 30.0)
    par_peso = min(1.0, len(regime_state.pares) / 10.0)

    score = (disp_peso + amp_peso + vib_peso + par_peso) / 4.0
    return score


if painel == "Diagnóstico Profundo":
    st.markdown("## 🧭 Diagnóstico Profundo — Estrutura da Estrada")

    if df.empty:
        st.warning("Carregue um histórico para visualizar diagnóstico.")
        st.stop()

    # Curvas estruturais básicas
    st.markdown("### 📈 Dispersão e Amplitude ao Longo do Tempo")

    dispersoes = df.apply(
        lambda row: np.std([row["n1"], row["n2"], row["n3"], row["n4"], row["n5"], row["n6"]]),
        axis=1,
    )
    amplitudes = df.apply(
        lambda row: max([row["n1"], row["n2"], row["n3"], row["n4"], row["n5"], row["n6"]])
        - min([row["n1"], row["n2"], row["n3"], row["n4"], row["n5"], row["n6"]]),
        axis=1,
    )

    plot_line(dispersoes, "Dispersão das Séries", "Dispersão")
    plot_line(amplitudes, "Amplitude das Séries", "Amplitude")

    # Vibração
    st.markdown("### 🌐 Vibração Estrutural")
    vib = np.abs(dispersoes.diff().fillna(0))
    plot_line(vib, "Variação da Dispersão (Vibração)", "Vibração")

    # Backtest Interno — distribuição de acertos
    st.markdown("### 🎯 Distribuição de Acertos — Backtest Interno")
    bti = st.session_state.get("backtest_interno", pd.DataFrame())
    if not bti.empty and "acertos" in bti.columns:
        plot_hist(bti["acertos"], "Distribuição de Acertos (Backtest Interno)", "Acertos")
    else:
        st.info("Backtest Interno ainda não foi executado.")

    # Backtest do Futuro — distribuição de acertos
    st.markdown("### 🔮 Distribuição de Acertos — Backtest do Futuro")
    btf = st.session_state.get("btf_raw", pd.DataFrame())
    if not btf.empty and "acertos" in btf.columns:
        plot_hist(btf["acertos"], "Distribuição de Acertos (Backtest do Futuro)", "Acertos")
    else:
        st.info("Backtest do Futuro ainda não foi executado.")

    # Estabilidade global
    st.markdown("### 🧩 Índice Global de Estabilidade")
    est = calcular_indice_estabilidade(regime_state)
    if est is None:
        st.info("Regime ainda não calculado.")
    else:
        st.metric("Estabilidade Estrutural", f"{est * 100:.1f}%")
        if est >= 0.75:
            st.success("Estrada ESTÁVEL (tendência resiliente).")
        elif est >= 0.50:
            st.warning("Estrada MODERADA (estado intermediário).")
        else:
            st.error("Estrada INSTÁVEL / Turbulenta.")

    st.stop()
# =========================================================
# PAINEL — EXPORTAR RESULTADOS (TXT / CSV)
# =========================================================

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def text_to_bytes(text: str) -> bytes:
    return text.encode("utf-8")


if painel == "Exportar Resultados":
    st.markdown("## 📤 Exportar Resultados (TXT / CSV)")

    # Usamos o leque final controlado
    leque = gerar_series_base(df, regime_state)
    flat_df = build_flat_series_table(leque)
    controlled_df = limit_by_mode(
        flat_df,
        regime_state,
        output_mode,
        n_series_fixed,
        min_conf_pct,
    )

    if controlled_df.empty:
        st.warning("Nenhuma série disponível para exportação.")
        st.stop()

    # CSV
    st.markdown("### 🟦 Baixar CSV (Leque Final)")
    csv_bytes = df_to_csv_bytes(controlled_df)
    st.download_button(
        "📥 Download CSV",
        data=csv_bytes,
        file_name="leque_turbo.csv",
        mime="text/csv",
    )

    # TXT puro
    st.markdown("### 🟩 Baixar TXT (Lista Pura)")
    lista_pura = []
    for i, (_, row) in enumerate(controlled_df.iterrows()):
        ss = " ".join(str(x) for x in row["series"])
        lista_pura.append(f"{i+1}) {ss}")

    txt_bytes = text_to_bytes("\n".join(lista_pura))

    st.download_button(
        "📥 Download TXT",
        data=txt_bytes,
        file_name="lista_pura.txt",
        mime="text/plain",
    )

    st.stop()


# =========================================================
# PAINEL — EXPORTAR SESSÃO COMPLETA (ZIP)
# =========================================================

def build_session_zip() -> bytes:
    """
    Gera um ZIP com:
    - histórico carregado
    - regime
    - núcleo
    - leque final
    - lista pura
    - S6
    - Monte Carlo
    - backtests
    - logs técnicos
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:

        # Histórico
        if not df.empty:
            z.writestr("historico.csv", df.to_csv(index=False))

        # Regime
        if regime_state:
            z.writestr(
                "regime.json",
                json.dumps(regime_state.__dict__, indent=2),
            )

        # Núcleo Resiliente
        nucleo = gerar_nucleo_resiliente(df, regime_state)
        z.writestr("nucleo.json", json.dumps(nucleo))

        # Leque Final (CSV)
        leque_final = gerar_series_base(df, regime_state)
        flat_df = build_flat_series_table(leque_final)
        z.writestr("leque_flat.csv", flat_df.to_csv(index=False))

        # Lista pura
        lista_pura = [
            f"{i+1}) " + " ".join(str(x) for x in row["series"])
            for i, (_, row) in enumerate(flat_df.iterrows())
        ]
        z.writestr("lista_pura.txt", "\n".join(lista_pura))

        # S6
        s6_df = st.session_state.get("s6_df", pd.DataFrame())
        if not s6_df.empty:
            z.writestr("s6.csv", s6_df.to_csv(index=False))

        # Monte Carlo
        mc_df = st.session_state.get("mc_df", pd.DataFrame())
        if not mc_df.empty:
            z.writestr("monte_carlo.csv", mc_df.to_csv(index=False))

        # Backtest Interno
        bti = st.session_state.get("backtest_interno", pd.DataFrame())
        if not bti.empty:
            z.writestr("backtest_interno.csv", bti.to_csv(index=False))

        # Backtest do Futuro
        btf = st.session_state.get("btf_raw", pd.DataFrame())
        if not btf.empty:
            z.writestr("backtest_futuro.csv", btf.to_csv(index=False))

        # Logs técnicos
        logs = st.session_state.get("logs_tecnicos", [])
        z.writestr("logs_tecnicos.json", json.dumps(logs, indent=2))

    buffer.seek(0)
    return buffer.read()


if painel == "Exportar Sessão Completa":
    st.markdown("## 📦 Exportar Sessão Completa (ZIP)")

    if df.empty:
        st.warning("Carregue histórico para exportar sessão.")
        st.stop()

    zip_bytes = build_session_zip()

    st.download_button(
        "📥 Baixar ZIP Completo",
        data=zip_bytes,
        file_name="predictcars_v13.8_turbo_session.zip",
        mime="application/zip",
    )

    st.stop()
# =========================================================
# MÓDULO S — Protocolos S1–S5 + Ajuste Fino Global (AFG)
# =========================================================
# Objetivo:
# - Detectar conflitos estruturais NG vs NL
# - Aplicar correções macro (S1–S4)
# - Aplicar ajuste fino (S5) em faixas críticas
# - Permitir comparar:
#   • Leque ORIGINAL (sem correção)
#   • Leque CORRIGIDO (com S1–S5 + AFG)
#   em um painel dedicado.
# =========================================================


# ---------------------------------------------------------
# 1) Núcleos Global (NG) e Local (NL)
# ---------------------------------------------------------

def gerar_nucleo_global(df: pd.DataFrame, regime: RegimeState) -> List[int]:
    """NG — Núcleo Global (usa todo o histórico via V13.8-TURBO)."""
    return gerar_nucleo_resiliente(df, regime)


def gerar_nucleo_local(df: pd.DataFrame, regime: RegimeState, janela: int = 40) -> List[int]:
    """
    NL — Núcleo Local:
    - usa apenas as últimas N séries (`janela`)
    - captura motorista de curto trecho / comportamento local
    """
    if df.empty:
        return []
    trecho = df.tail(min(janela, len(df)))
    return gerar_nucleo_resiliente(trecho, regime)


# ---------------------------------------------------------
# 2) Métricas de Conflito: MUC, dispersão, zona final
# ---------------------------------------------------------

@dataclass
class SMetricasConflito:
    muc: float
    d_faixas: float
    d_clusters: float
    d_mediana: float
    d_disp: float
    d_zona: float
    aciona_s1: bool
    aciona_s2: bool
    aciona_s3: bool
    aciona_s4: bool


def _faixa_media(serie: List[int]) -> float:
    if not serie:
        return 0.0
    return float(np.mean(serie))


def _mediana_serie(serie: List[int]) -> float:
    if not serie:
        return 0.0
    return float(np.median(serie))


def _pseudo_cluster(serie: List[int]) -> float:
    """
    Proxy simples para clusters / motoristas:
    - calcula média dos deltas entre vizinhos.
    """
    if len(serie) < 2:
        return 0.0
    arr = np.array(sorted(serie))
    deltas = np.diff(arr)
    return float(np.mean(deltas))


def calcular_metricas_conflito_s(
    df: pd.DataFrame,
    regime: Optional[RegimeState],
    janela_local: int = 40,
) -> Optional[SMetricasConflito]:
    """
    Calcula:
    - NG (núcleo global)
    - NL (núcleo local)
    - D_faixas, D_clusters, D_mediana
    - D_disp (dispersão prevista vs real)
    - D_zona (zona final prevista vs real)
    - MUC = média das três primeiras
    Define gatilhos para S1–S4 com limiares heurísticos.
    """
    if df.empty or regime is None:
        return None

    ng = gerar_nucleo_global(df, regime)
    nl = gerar_nucleo_local(df, regime, janela_local)

    if not ng or not nl:
        return None

    # faixas
    d_faixas = abs(_faixa_media(ng) - _faixa_media(nl))

    # clusters aproximados
    d_clusters = abs(_pseudo_cluster(ng) - _pseudo_cluster(nl))

    # mediana
    d_mediana = abs(_mediana_serie(ng) - _mediana_serie(nl))

    muc = (d_faixas + d_clusters + d_mediana) / 3.0

    # Dispersão prevista x real (últimas séries)
    nums = df[["n1", "n2", "n3", "n4", "n5", "n6"]].values
    disp_real = float(np.mean(np.std(nums, axis=1)))
    disp_prev = float(np.std(np.array(ng)))
    d_disp = abs(disp_prev - disp_real)

    # Zona final (último passageiro da série / núcleo)
    ultimo_real = float(np.mean(nums[:, -1]))  # média dos últimos passageiros
    ultimo_prev = float(sorted(ng)[-1])
    d_zona = abs(ultimo_prev - ultimo_real)

    # Limiar heurístico (podem ser recalibrados via backtest)
    theta_global = 6.0   # conflito forte NG vs NL
    theta_local = 4.0    # conflito local acentuado
    theta_disp = 5.0     # dispersão atípica
    theta_zf = 5.0       # zona final desalinhada

    aciona_s1 = muc > theta_global
    aciona_s2 = (muc > theta_local) and not aciona_s1
    aciona_s3 = d_disp > theta_disp
    aciona_s4 = d_zona > theta_zf

    return SMetricasConflito(
        muc=muc,
        d_faixas=d_faixas,
        d_clusters=d_clusters,
        d_mediana=d_mediana,
        d_disp=d_disp,
        d_zona=d_zona,
        aciona_s1=aciona_s1,
        aciona_s2=aciona_s2,
        aciona_s3=aciona_s3,
        aciona_s4=aciona_s4,
    )


# ---------------------------------------------------------
# 3) Macrocorreções S1–S4 sobre o núcleo / leque
# ---------------------------------------------------------

def aplicar_anti_s1(nucleo: List[int]) -> List[int]:
    """
    Anti-S1 — Núcleo supercomprimido:
    - aumenta levemente dispersão
    - reintroduz "segunda força" via deslocamento suave
    """
    if not nucleo:
        return []
    base = sorted(nucleo)
    # empurra alguns elementos para abrir a faixa
    ajustado = []
    for i, x in enumerate(base):
        if i == 0:
            ajustado.append(x)
        else:
            if x - ajustado[-1] < 3:
                ajustado.append(ajustado[-1] + 3)
            else:
                ajustado.append(x)
    return sorted(set(min(80, max(1, v)) for v in ajustado))[:6]


def aplicar_anti_s2(ng: List[int], nl: List[int]) -> List[int]:
    """
    Anti-S2 — Motorista de curto trecho:
    - Núcleo final = interseção NG∩NL + 1–2 dominantes locais
    """
    if not ng or not nl:
        return sorted(set(ng or nl))[:6]

    inter = sorted(set(ng) & set(nl))
    locais = [x for x in nl if x not in inter]

    # garante interseção
    resultado = inter.copy()

    # adiciona até 2 dominantes locais
    for x in locais:
        if len(resultado) >= 6:
            break
        resultado.append(x)

    # completa, se faltar, com NG
    for x in ng:
        if len(resultado) >= 6:
            break
        if x not in resultado:
            resultado.append(x)

    return sorted(set(resultado))[:6]


def aplicar_anti_s3(nucleo: List[int], disp_alvo: float) -> List[int]:
    """
    Anti-S3 — Dispersão atípica:
    - ajusta extremos para aproximar dispersão de disp_alvo.
    """
    if not nucleo:
        return []

    base = sorted(nucleo)
    disp_atual = float(np.std(np.array(base)))
    # heurística simples: se muito menor, abre extremos; se muito maior, puxa
    if disp_atual < disp_alvo:
        base[0] = max(1, base[0] - 1)
        base[-1] = min(80, base[-1] + 1)
    elif disp_atual > disp_alvo:
        base[0] = min(base[-1], base[0] + 1)
        base[-1] = max(base[0], base[-1] - 1)

    return sorted(set(base))[:6]


def aplicar_anti_s4(nucleo: List[int], ultimo_real: float) -> List[int]:
    """
    Anti-S4 — Zona final desalinhada:
    - ajusta o último passageiro em direção à média real da cauda.
    """
    if not nucleo:
        return []
    base = sorted(nucleo)
    alvo = int(round(ultimo_real))
    # move só o último elemento
    base[-1] = min(80, max(1, alvo))
    return sorted(set(base))[:6]


def aplicar_macro_s1_s4(
    df: pd.DataFrame,
    regime: RegimeState,
    metricas: SMetricasConflito,
) -> List[int]:
    """
    Aplica S1–S4 sobre o núcleo global, retornando núcleo macro-corrigido.
    """
    ng = gerar_nucleo_global(df, regime)
    nl = gerar_nucleo_local(df, regime, janela=40)
    if not ng:
        return []

    nuc = ng.copy()

    # Dispersão real alvo
    nums = df[["n1", "n2", "n3", "n4", "n5", "n6"]].values
    disp_real = float(np.mean(np.std(nums, axis=1)))
    ultimo_real = float(np.mean(nums[:, -1]))

    if metricas.aciona_s1:
        nuc = aplicar_anti_s1(nuc)

    if metricas.aciona_s2:
        nuc = aplicar_anti_s2(nuc, nl)

    if metricas.aciona_s3:
        nuc = aplicar_anti_s3(nuc, disp_real)

    if metricas.aciona_s4:
        nuc = aplicar_anti_s4(nuc, ultimo_real)

    return sorted(set(nuc))[:6]


# ---------------------------------------------------------
# 4) Ajuste Fino Global (AFG) + S5 (permutações finas)
# ---------------------------------------------------------

def identificar_faixas_criticas(serie: List[int]) -> List[Tuple[int, int]]:
    """
    Identifica pares de valores muito próximos (candidatos equivalentes).
    Ex.: (30, 32), (45, 47) etc.
    """
    if len(serie) < 2:
        return []
    arr = sorted(serie)
    criticos = []
    for i in range(len(arr) - 1):
        if abs(arr[i+1] - arr[i]) <= 2:
            criticos.append((arr[i], arr[i+1]))
    return criticos


def aplicar_s5_permuta_fina(serie: List[int]) -> List[List[int]]:
    """
    S5 — Permutações finas em faixas críticas:
    - troca apenas dois elementos em faixas críticas
    - mantém sempre 6 passageiros
    - NÃO usa set(), NÃO redefine, NÃO corta lista
    """
    base = sorted(serie)
    criticos = identificar_faixas_criticas(base)

    if not criticos:
        return []

    variacoes = []

    # limite de no máximo 2 variações
    for par in criticos[:2]:
        a, b = par

        # só cria permutação se ambos estão na série
        if a in base and b in base:
            s = base.copy()
            i = s.index(a)
            j = s.index(b)

            # troca simples (permuta controlada)
            s[i], s[j] = s[j], s[i]

            # garantir que continua com 6 números ordenados
            variacoes.append(sorted(s))

    # remover duplicações
    limpas = []
    for v in variacoes:
        if v not in limpas:
            limpas.append(v)

    return limpas



def aplicar_ajuste_fino_global(
    flat_df: pd.DataFrame,
    score_min: float = 0.70,
) -> pd.DataFrame:
    """
    AFG:
    - atua somente em séries com coherence >= score_min
    - aplica S5 para gerar variações finas em faixas críticas
    - não altera séries base, apenas adiciona variações derivadas
    """
    if flat_df.empty:
        return flat_df

    linhas = []
    # Copia todas as séries originais
    for _, row in flat_df.iterrows():
        linhas.append(row.to_dict())

    # Ajuste fino apenas nas séries mais fortes
    foco = flat_df[flat_df["coherence"] >= score_min]

    for _, row in foco.iterrows():
        serie_base = row["series"]
        variacoes = aplicar_s5_permuta_fina(serie_base)
        for v in variacoes:
            novo = row.to_dict()
            novo["series"] = sorted(v)
            # leve ajuste na coherence / expected_hits (refinamento)
            novo["coherence"] = min(1.0, float(novo["coherence"]) + 0.02)
            novo["expected_hits"] = min(6, int(novo["expected_hits"]) + 0)
            novo["category"] = f"{row['category']}+S5"
            linhas.append(novo)
 
    # Padronizar e filtrar séries válidas antes de remover duplicatas
    df_temp = pd.DataFrame(linhas)

    # 1) manter apenas séries "de verdade":
    #    - listas/tuplas
    #    - com exatamente 6 números
    #    (descarta vetores estranhos tipo 0 1 1 1 2 3 3 3 3 3 4 6)
    def _serie_valida(s):
        if isinstance(s, (list, tuple)):
            if len(s) != 6:
                return False
            # opcional: garantir que são inteiros entre 1 e 80
            try:
                return all(isinstance(x, (int, float)) and 1 <= int(x) <= 80 for x in s)
            except Exception:
                return False
        return False

    df_temp = df_temp[df_temp["series"].apply(_serie_valida)].copy()

    # 2) converter a lista em string para poder usar drop_duplicates
    df_temp["series"] = df_temp["series"].apply(
        lambda s: " ".join(str(int(x)) for x in s)  # garante "n1 n2 n3 n4 n5 n6"
    )

    # 3) remover duplicatas e ordenar por coherence
    df_out = df_temp.drop_duplicates(subset=["category", "series"])
    df_out = df_out.sort_values("coherence", ascending=False).reset_index(drop=True)
    return df_out




# ---------------------------------------------------------
# 5) Construção do Leque CORRIGIDO (S1–S5)
# ---------------------------------------------------------

def gerar_leque_corrigido(
    df: pd.DataFrame,
    regime: RegimeState,
) -> Dict[str, Any]:
    """
    Gera um leque corrigido:
    - núcleo passa por S1–S4
    - séries derivadas passam por AFG + S5
    """
    metricas = calcular_metricas_conflito_s(df, regime)
    if metricas is None:
        # fallback: usa leque base
        return gerar_series_base(df, regime)

    # núcleo corrigido
    nuc_corrigido = aplicar_macro_s1_s4(df, regime, metricas)

    # constrói leque base a partir do núcleo corrigido
    if not nuc_corrigido:
        leque_base = gerar_series_base(df, regime)
    else:
        # reaproveita a lógica de gerar_series_base, mas com núcleo injetado
        leque_base = gerar_series_base(df, regime)
        leque_base["nucleo"] = nuc_corrigido

    # flat base
    flat_base = build_flat_series_table(leque_base)
    # aplica ajuste fino (AFG + S5)
    flat_corrigido = aplicar_ajuste_fino_global(flat_base, score_min=0.70)

    # reconstrói dicionário de listas para o painel de comparação
    leque_out = {
        "nucleo": nuc_corrigido,
        "premium": [],
        "estruturais": [],
        "cobertura": [],
        "s6": [],
    }
    # Função auxiliar para garantir séries com exatamente 6 números
    def _converter_serie(s):
        # caso 1 — se vier como string: "12 22 30 35 40 54"
        if isinstance(s, str):
            try:
                nums = [int(x) for x in s.split() if x.isdigit()]
                if len(nums) >= 6:
                    return sorted(nums[:6])
            except:
                pass

        # caso 2 — se vier como lista/tupla
        if isinstance(s, (list, tuple)):
            try:
                nums = [int(x) for x in s]
                if len(nums) >= 6:
                    return sorted(nums[:6])
            except:
                pass

        # fallback — inválido
        return []

    for _, row in flat_corrigido.iterrows():
        cat = row["category"]
        serie = _converter_serie(row["series"])
        if cat.startswith("NÚCLEO"):
            leque_out["nucleo"] = serie
        elif cat.startswith("Premium"):
            leque_out["premium"].append(serie)
        elif cat.startswith("Estrutural"):
            leque_out["estruturais"].append(serie)
        elif cat.startswith("Cobertura"):
            leque_out["cobertura"].append(serie)
        elif cat.startswith("S6"):
            leque_out.setdefault("s6", []).append(serie)

    return leque_out
# ---------------------------------------------------------
# FUNÇÃO — UNIR LEQUES ORIGINAL + CORRIGIDO (TURBO++)
# ---------------------------------------------------------

def unir_leques(flat_original: pd.DataFrame, flat_corr: pd.DataFrame) -> pd.DataFrame:
    """
    Junta os dois leques:
    - mantém categoria original
    - remove duplicatas
    - reordena por coherence (maior primeiro)
    """
    if flat_original is None or flat_original.empty:
        return flat_corr.copy()

    if flat_corr is None or flat_corr.empty:
        return flat_original.copy()

    df_mix = pd.concat([flat_original, flat_corr], ignore_index=True)

    # remover duplicatas por série (string ou lista coerente)
    def _key(s):
        if isinstance(s, (list, tuple)):
            return " ".join(str(int(x)) for x in s)
        if isinstance(s, str):
            return s.strip()
        return str(s)

    df_mix["serie_key"] = df_mix["series"].apply(_key)

    df_mix = df_mix.drop_duplicates(subset=["serie_key"])
    df_mix = df_mix.sort_values("coherence", ascending=False).reset_index(drop=True)
    df_mix = df_mix.drop(columns=["serie_key"])

    return df_mix



# =========================================================
# PAINEL — SAÍDA FINAL CONTROLADA
# =========================================================

if painel == "Saída Final Controlada":
    st.markdown("## 🎯 Saída Final Controlada — Leque TURBO")

    if df.empty:
        st.warning("Carregue um histórico para gerar o leque.")
        st.stop()

    # Gera/Regera o leque base
    leque = gerar_series_base(df, regime_state)
    st.session_state["leque_turbo"] = leque
    # Leque corrigido S1–S5
    leque_corrigido = gerar_leque_corrigido(df, regime_state)
    flat_corr = build_flat_series_table(leque_corrigido).copy()

    # Criar Leque MISTO (ORIGINAL + CORRIGIDO)
    leque_original = gerar_series_base(df, regime_state)
    flat_original = build_flat_series_table(leque_original)

    flat_mix = unir_leques(flat_original, flat_corr)

    # Aplicar modo de saída sobre o MIX
    flat_df = limit_by_mode(
        flat_mix,
        regime_state,
        output_mode,
        n_series_fixed,
        min_conf_pct,
    )


    flat_df = flat_corr.copy()
    if flat_df.empty:
        st.warning("Não foi possível gerar séries a partir do histórico atual.")
        st.stop()
    # Criar Leque MISTO TURBO++
    flat_mix = unir_leques(flat_df, flat_corr)
    
    # Aplica modo de saída
    controlled_df = limit_by_mode(
        flat_mix,
        regime_state,
        output_mode,
        n_series_fixed,
        min_conf_pct,
    )

    # Monta tabela final
    def montar_tabela_final(df_in: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "Rank": i + 1,
                "Categoria": row["category"],
                "Série": row["series"],
                "Confiabilidade (%)": int(round(row["coherence"] * 100)),
                "Acertos Esperados": int(row["expected_hits"]),
            }
            for i, (_, row) in enumerate(df_in.iterrows())
        ])
    # Núcleo Resiliente Final (NRF)
    try:
        nucleo_resiliente = None

        # pega a melhor série do controlled_df (rank 1)
        if not controlled_df.empty:
            melhor = controlled_df.iloc[0]
            nucleo_resiliente = melhor["series"]

        st.markdown("### ⭐ Núcleo Resiliente Final (NRF)")
        if nucleo_resiliente:
            st.code(" ".join(str(x) for x in nucleo_resiliente), language="text")
        else:
            st.write("Núcleo não disponível.")

    except Exception as e:
        st.error(f"Erro ao gerar Núcleo Resiliente Final: {e}")

    # =========================================================
    # SENSOR AMBIENTAL k* — MODO SIMPLES (Compatível com V13.8)
    # =========================================================
    try:
        # Histórico completo
        df_hist = df.copy()

        # Função para renomear colunas corretamente
        if df_hist.shape[1] >= 8:
            df_hist.columns = ["id", "n1", "n2", "n3", "n4", "n5", "n6", "k"]
        else:
            # fallback: se vier sem ID
            if df_hist.shape[1] == 7:
                df_hist.columns = ["n1", "n2", "n3", "n4", "n5", "n6", "k"]
                df_hist["id"] = None
            else:
                df_hist["k"] = 0  # pior caso

        # Últimos valores de k
        ultimos_k = df_hist["k"].tail(5).tolist()

        # Detecta ruptura recente (k != 0)
        ruptura_recente = (df_hist["k"].iloc[-1] != 0)

        # Lógica do sensor
        if ruptura_recente:
            k_estado = "critico"
        else:
            if any(k != 0 for k in ultimos_k):
                k_estado = "atencao"
            else:
                k_estado = "estavel"

        # Exibir badge ambiental
        st.markdown("### 🌡️ Estado Ambiental da Estrada (k*)")
        st.markdown(contexto_k_texto(k_estado, prefixo="k*"))

    except Exception as e:
        st.error(f"Erro no sensor k* simples: {e}")

def contexto_k_texto(k_estado: str, prefixo: str = "k*") -> str:
    """
    Gera o texto padrão para o estado k* ou k̂.
    k_estado: "estavel", "atencao" ou "critico"
    prefixo: rótulo exibido (ex.: "k*", "k̂", "k efetivo")
    """
    if k_estado == "estavel":
        return f"🟢 {prefixo}: Ambiente estável — previsão em regime normal."
    elif k_estado == "atencao":
        return f"🟡 {prefixo}: Pré-ruptura residual — usar previsão com atenção."
    else:
        return f"🔴 {prefixo}: Ambiente crítico — usar previsão com cautela máxima."

def calcular_sdm(df, janela=8):
    """
    SDM — Similaridade Dinâmica do Momento.
    Mede quão parecidos são os últimos trechos entre si.
    Retorna valor entre 0 e 1.
    """

    try:
        if df is None or df.empty or len(df) < janela + 1:
            return 0.5  # neutro

        # Extrair últimos n1..n6
        recentes = df[["n1","n2","n3","n4","n5","n6"]].tail(janela + 1).values

        atual = recentes[-1]
        anteriores = recentes[:-1]

        sims = []
        for linha in anteriores:
            dist = abs(linha - atual).sum()
            sim = 1 / (1 + dist)
            sims.append(sim)

        return float(sum(sims) / len(sims))

    except:
        return 0.5  # fallback neutro
def calcular_t_norm(df, janela=10):
    """
    T_norm — Turbulência Normalizada.
    Mede quanta oscilação existe nos últimos trechos.
    Retorna valor entre 0 e 1.
    """

    try:
        if df is None or df.empty or len(df) < janela:
            return 0.5  # neutro

        # pegar últimos trechos n1..n6
        bloco = df[["n1","n2","n3","n4","n5","n6"]].tail(janela).values

        # medir dispersão média entre trechos
        variacoes = []
        for i in range(1, len(bloco)):
            dist = abs(bloco[i] - bloco[i-1]).sum()
            variacoes.append(dist)

        media = sum(variacoes) / len(variacoes)

        # normaliza entre 0 e 1 usando função suave
        t_norm = 1 - (1 / (1 + media))

        return float(t_norm)

    except:
        return 0.5  # fallback neutro
def calcular_entropia_k(df, janela=10):
    """
    Entropia direcional do k.
    Mede a irregularidade do comportamento do k recente.
    Retorna valor entre 0 e 1.
    """

    try:
        if df is None or df.empty or "k" not in df.columns:
            return 0.5  # neutro

        # últimos valores de k
        k_vals = df["k"].tail(janela).tolist()

        if len(k_vals) <= 1:
            return 0.5

        # contar mudanças de estado
        mudancas = 0
        for i in range(1, len(k_vals)):
            if (k_vals[i] != 0) != (k_vals[i-1] != 0):
                mudancas += 1

        entropia = mudancas / (len(k_vals) - 1)

        return float(entropia)

    except:
        return 0.5
def calcular_tendencia_k(df, janela=12):
    """
    Tendência do k (k-slope).
    Indica se o ambiente está melhorando (+), piorando (-) ou neutro.
    Retorna: -1, 0 ou +1.
    """

    try:
        if df is None or df.empty or "k" not in df.columns:
            return 0  # neutro

        k_vals = df["k"].tail(janela).tolist()

        if len(k_vals) < 3:
            return 0

        # Converter k em binário (0 = estável, 1 = alerta/ruptura)
        binario = [1 if k != 0 else 0 for k in k_vals]

        # Eixo x (0,1,2,...)
        xs = list(range(len(binario)))

        # Cálculo do slope simples (regressão linear de 1 variável)
        n = len(xs)
        media_x = sum(xs) / n
        media_y = sum(binario) / n

        num = sum((xs[i] - media_x) * (binario[i] - media_y) for i in range(n))
        den = sum((xs[i] - media_x) ** 2 for i in range(n))

        slope = num / den if den != 0 else 0

        # Interpretação da tendência
        if slope > 0.05:
            return +1  # piorando
        elif slope < -0.05:
            return -1  # melhorando
        else:
            return 0   # neutro

    except:
        return 0


def calcular_k_pred(k_estado_atual: str, df):
    """
    k preditivo básico (k̂) — versão inicial.
    Nesta fase, apenas retorna o próprio k_estado_atual.
    Depois, iremos substituir pela versão real com SDM, T_norm, entropia e tendência.
    """
    try:
        # 1) Calcular sensores avançados
        sdm = calcular_sdm(df)
        tnorm = calcular_t_norm(df)
        ent = calcular_entropia_k(df)
        trend = calcular_tendencia_k(df)

        # 2) Score bruto
        score = (
            0.30 * (1 - sdm) +      # menor similaridade → mais crítico
            0.30 * tnorm +         # mais turbulência → pior
            0.25 * ent +           # mais entropia → pior
            0.15 * (trend + 1)/2   # trend: -1,0,+1 → normaliza p/ 0..1
        )

        # 3) Classificação por faixas
        if score < 0.33:
            return "estavel"
        elif score < 0.66:
            return "atencao"
        else:
            return "critico"

    except:
        return k_estado_atual

   

def ajustar_n_series_por_k(k_ativo: str, n_series_base: int) -> int:
    """
    Ajuste simples do tamanho do leque com base no k_ativo.
    - estavel  → mantém o tamanho original
    - atencao  → reduz levemente o leque
    - critico  → reduz mais o leque, focando nas séries mais fortes
    Sempre respeita um mínimo de 5 séries.
    """
    n = n_series_base

    if k_ativo == "atencao":
        n = max(5, n - 2)
    elif k_ativo == "critico":
        n = max(5, n - 4)

    return n

# Previsão Final TURBO
try:
    previsao_final = None
    if not controlled_df.empty:
        melhor = controlled_df.iloc[0]
        previsao_final = melhor["series"]
    prefixo_k = "k*" if k_ativo == k_estado else "k̂"
    contexto_k = contexto_k_texto(k_ativo, prefixo=prefixo_k)
    st.markdown("### 🎯 Previsão Final TURBO")
    if previsao_final:
        st.code(" ".join(str(x) for x in previsao_final), language="text")
        st.info(contexto_k)
    else:
        st.write("Previsão não disponível.")

except Exception as e:
    st.error(f"Erro ao gerar Previsão Final TURBO: {e}")

# Listas Auxiliares TURBO
try:
    st.markdown("### 🧩 Listas Auxiliares (Premium / Estruturais / Cobertura)")

    lista_premium = []
    lista_estruturais = []
    lista_cobertura = []

    for _, row in controlled_df.iterrows():
        cat = row["category"]
        ss = " ".join(str(x) for x in row["series"])

        # Ajuste leve por k_ativo (modo simples)
        if k_ativo == "critico":
            # Em ambiente crítico, remover Cobertura (focar no núcleo forte)
            if "Cobertura" in cat:
                continue
        elif k_ativo == "atencao":
            # Em atenção, reduzir Cobertura (deixa passar só parte)
            import random
            if "Cobertura" in cat and random.random() < 0.5:
                continue

        if cat.startswith("Premium"):
            lista_premium.append(ss)
        elif cat.startswith("Estrutural"):
            lista_estruturais.append(ss)
        elif cat.startswith("Cobertura"):
            lista_cobertura.append(ss)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### ⭐ Premium")
        st.text_area("Premium", value="\n".join(lista_premium), height=200)

    with col2:
        st.markdown("#### 🧱 Estruturais")
        st.text_area("Estruturais", value="\n".join(lista_estruturais), height=200)

    with col3:
        st.markdown("#### 🌐 Cobertura")
        st.text_area("Cobertura", value="\n".join(lista_cobertura), height=200)

except Exception as e:
    st.error(f"Erro ao gerar listas auxiliares: {e}")


# Lista Pura Final TURBO
try:
    st.markdown("### 📋 Lista Pura Final (Numerada)")

    lista_final = []
    for i, (_, row) in enumerate(controlled_df.iterrows()):
        ss = " ".join(str(x) for x in row["series"])
        lista_final.append(f"{i + 1}) {ss}")

    st.text_area(
        "Lista Pura Final",
        value="\n".join(lista_final),
        height=220,
    )

except Exception as e:
    st.error(f"Erro ao gerar Lista Pura Final: {e}")

# Monta tabela para exibição
st.markdown("### 📦 Leque Final — TURBO")
st.dataframe(
    montar_tabela_final(controlled_df),
    use_container_width=True
)

# BOTÃO — EXPORTAR PREVISÃO TURBO++
if not controlled_df.empty:
    pass

    try:
        texto_exportar = "\n".join(
            " ".join(str(x) for x in row["series"])
            for _, row in controlled_df.iterrows()
        )

        st.markdown("### 📤 Exportar Previsão TURBO++")
        st.download_button(
            label="📥 Baixar arquivo .txt com as séries (TURBO++)",
            data=texto_exportar,
            file_name="previsao_turbo_plus.txt",
            mime="text/plain",
        )

    except Exception as e:
        st.error(f"Erro ao exportar arquivo TURBO++: {e}")

st.stop()


# ---------------------------------------------------------
# 6) Painel S1–S5 + Ajuste Fino — Comparação Original vs Corrigido
# ---------------------------------------------------------

if painel == "S1–S5 + Ajuste Fino":
    st.markdown("## 🌀 Protocolos S1–S5 + Ajuste Fino Global")

    if df.empty or regime_state is None:
        st.warning("Carregue o histórico para ativar os protocolos S1–S5.")
        st.stop()

    # Modo de visualização
    modo_corr = st.radio(
        "Modo de visualização:",
        [
            "Somente Leque Original (sem correção)",
            "Somente Leque Corrigido (S1–S5 + AFG)",
            "Comparar Lado a Lado",
        ],
        index=2,
    )

    # Métricas de conflito
    metricas = calcular_metricas_conflito_s(df, regime_state)
    if metricas is None:
        st.info("Não foi possível calcular métricas de conflito. Usando apenas leque original.")
        metricas = None

    st.markdown("### 📊 Métrica Universal de Conflito (MUC) e derivados")

    col_m1, col_m2, col_m3 = st.columns(3)
    if metricas:
        col_m1.metric("MUC", f"{metricas.muc:.2f}")
        col_m2.metric("D_faixas", f"{metricas.d_faixas:.2f}")
        col_m3.metric("D_clusters", f"{metricas.d_clusters:.2f}")

        col_m4, col_m5, col_m6 = st.columns(3)
        col_m4.metric("D_mediana", f"{metricas.d_mediana:.2f}")
        col_m5.metric("D_disp", f"{metricas.d_disp:.2f}")
        col_m6.metric("D_zona", f"{metricas.d_zona:.2f}")

        st.markdown("#### 🔔 Gatilhos S1–S4")
        gatilhos = {
            "S1 — Núcleo supercomprimido": metricas.aciona_s1,
            "S2 — Motorista de curto trecho": metricas.aciona_s2,
            "S3 — Dispersão atípica": metricas.aciona_s3,
            "S4 — Zona final desalinhada": metricas.aciona_s4,
        }
        for nome, flag in gatilhos.items():
            if flag:
                st.error(f"{nome} — ATIVADO")
            else:
                st.success(f"{nome} — Inativo")
    else:
        st.write("Métricas indisponíveis nesta configuração de histórico.")

    st.markdown("---")

    # Leque ORIGINAL
    leque_original = gerar_series_base(df, regime_state)
    flat_original = build_flat_series_table(leque_original)
    flat_original = limit_by_mode(
        flat_original,
        regime_state,
        output_mode,
        n_series_fixed,
        min_conf_pct,
    )

    # Leque CORRIGIDO
    leque_corrigido = gerar_leque_corrigido(df, regime_state)
    flat_corr = build_flat_series_table(leque_corrigido).copy()

    # Leque MISTO (ORIGINAL + CORRIGIDO)
    flat_mix = unir_leques(flat_original, flat_corr)

    # Aplicar modo de saída no MIX

    n_series_ajustado = ajustar_n_series_por_k(k_ativo, n_series_fixed)

    flat_corrigido = limit_by_mode(
        flat_mix,
        regime_state,
        output_mode,
        n_series_ajustado,
        min_conf_pct,
    )




    def montar_tabela(flat_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "Rank": i + 1,
                "Categoria": row["category"],
                "Série": row["series"],
                "Confiabilidade (%)": int(round(row["coherence"] * 100)),
                "Acertos Esperados": int(row["expected_hits"]),
            }
            for i, (_, row) in enumerate(flat_df.iterrows())
        ])

    if modo_corr.startswith("Somente Leque Original"):
        st.markdown("### 🎯 Leque Original (sem correções S1–S5)")
        st.dataframe(montar_tabela(flat_original), use_container_width=True)

    elif modo_corr.startswith("Somente Leque Corrigido"):
        st.markdown("### 🎯 Leque Corrigido (S1–S5 + AFG)")
        st.dataframe(montar_tabela(flat_corrigido), use_container_width=True)

    else:
        st.markdown("### 🆚 Comparação Lado a Lado")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 🎯 Leque Original")
            st.dataframe(montar_tabela(flat_original), use_container_width=True)

        with c2:
            st.markdown("#### 🎯 Leque Corrigido (S1–S5 + AFG)")
            st.dataframe(montar_tabela(flat_corrigido), use_container_width=True)
    # Listas puras (para copiar)
    st.markdown("---")
    st.markdown("### 📋 Listas Puras — Original vs Corrigido")

    lista_orig = [
        f"{i+1}) " + formatar_serie_para_texto(row["series"])
        for i, (_, row) in enumerate(flat_original.iterrows())
    ]

    lista_corr = [
        f"{i+1}) " + formatar_serie_para_texto(row["series"])
        for i, (_, row) in enumerate(flat_corrigido.iterrows())
    ]

    col_o, col_c = st.columns(2)
    with col_o:
        st.markdown("#### Original")
        st.text_area(
            "Lista Pura Original",
            value="\n".join(lista_orig),
            height=300,
        )

    with col_c:
        st.markdown("#### Corrigido (S1–S5)")
        st.text_area(
            "Lista Pura Corrigida",
            value="\n".join(lista_corr),
            height=300,
        )

      

    st.stop()
# ---------------------------------------------------------
# 7) Painel TURBO — Saída Final V13.8
# ---------------------------------------------------------

if painel == "Saída Turbo V13.8":
    st.markdown("## 🚀 Predict Cars V13.8 — Saída Turbo Final")

    if df.empty or regime_state is None:
        st.warning("Carregue o histórico para ativar a Saída Turbo.")
        st.stop()

    st.info("Painel Turbo instalado. Falta ativar o motor interno (Passo 4).")

# =========================================================
# PAINEL: COMPARAÇÃO k* vs k̂
# =========================================================

if painel == "Comparação k* vs k̂":
    st.markdown("## ⚖️ Comparação entre k* (atual) e k̂ (preditivo)")

    if df.empty:
        st.warning("Histórico vazio — carregue um arquivo para comparar.")
        st.stop()

    # --- Contexto com k atual (k*) ---
    prefixo_kA = "k*"
    contextoA = contexto_k_texto(k_estado, prefixo=prefixo_kA)

    # --- Contexto com k preditivo (k̂) ---
    prefixo_kB = "k̂"
    contextoB = contexto_k_texto(k_pred, prefixo=prefixo_kB)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔵 Previsão com k atual (k*)")
        st.markdown(contextoA)

    with col2:
        st.markdown("### 🟣 Previsão com k preditivo (k̂)")
        st.markdown(contextoB)

    st.info("Este painel compara exclusivamente o estado ambiental. "
            "A previsão numérica permanece igual por enquanto.")

    st.stop()

# ---------------------------------------------------------
# Função auxiliar — Normalizar Série
# ---------------------------------------------------------

import pandas as pd
from collections.abc import Iterable

def normalizar_serie(serie):
    if serie is None or (isinstance(serie, float) and pd.isna(serie)):
        return ""

    nums = []

    if isinstance(serie, str):
        cleaned = (serie.replace("["," ").replace("]"," ")
                         .replace("(" , " ").replace(")" , " ")
                         .replace(",", " ").replace(";", " "))
        for t in cleaned.split():
            try: nums.append(int(t))
            except: pass

    elif isinstance(serie, (list, tuple, set)):
        try: nums = [int(x) for x in serie]
        except: return str(serie)

    else:
        try:
            if isinstance(serie, Iterable):
                nums = [int(x) for x in serie]
            else:
                return str(serie)
        except:
            return str(serie)

    if not nums:
        return str(serie)

    nums = sorted(dict.fromkeys(nums))
    return " ".join(str(n) for n in nums)

