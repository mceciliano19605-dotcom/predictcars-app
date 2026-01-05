# ============================================================
# PARTE 1/8 — INÍCIO
# ============================================================
"""
📜 BLOCO CANÔNICO DE GOVERNANÇA — PREDICTCARS
============================================================

⚠️ ESTE BLOCO É PARTE ESTRUTURAL DO SISTEMA
⚠️ NÃO EXECUTA
⚠️ NÃO ALTERA COMPORTAMENTO
⚠️ NÃO PODE SER REMOVIDO, REDUZIDO OU REINTERPRETADO

Este bloco existe para garantir CONTINUIDADE, GOVERNANÇA,
RASTREABILIDADE e SEGURANÇA EVOLUTIVA do PredictCars.

Ele substitui QUALQUER dependência de:
- memória do operador
- memória de chats
- explicações externas
- “contexto implícito”

============================================================
🔒 ESTADO BASELINE DO SISTEMA
============================================================

Sistema: PredictCars
Versão-base: V15.7 MAX + V16 Premium
Estado: ESTÁVEL (BASELINE A)

Este arquivo representa um sistema:
- operacional
- validado
- em uso real
- sem regressões conhecidas

Qualquer evolução futura DEVE:
- partir deste estado
- ser ADITIVA
- ser auditada
- preservar o “jeitão” histórico do PredictCars

============================================================
🧭 MAPA CANÔNICO DOS PAINÉIS (ORDEM OPERACIONAL)
============================================================

A navegação do PredictCars segue UM TRILHO ÚNICO.
Os painéis abaixo NÃO são concorrentes.

A exclusividade é garantida por:
- seleção única via sidebar.radio
- roteador sequencial
- encerramento explícito via st.stop()

ORDEM OPERACIONAL CANÔNICA:

0️⃣ 📁 Carregar Histórico
1️⃣ 🧭 Modo Guiado Oficial — PredictCars
2️⃣ 🧭 Checklist Operacional — Decisão (AGORA)

3️⃣ 🛣️ Pipeline V14-FLEX ULTRA
4️⃣ 📊 Sentinelas (k / k*)
5️⃣ 🌊 ECO / PRÉ-ECO / Estado

6️⃣ ⚙️ Modo 6
7️⃣ ⚙️ Modo TURBO++ HÍBRIDO
8️⃣ ⚙️ Modo TURBO++ ULTRA

9️⃣ 📡 Painel de Ruído Condicional
🔟 📉 Painel de Divergência S6 vs MC

1️⃣1️⃣ 🔵 Modo Especial — Evento Condicionado
1️⃣2️⃣ 🧠 V16 Premium (painéis retrospectivos e observacionais)

============================================================
🚫 REGRA ABSOLUTA — NÃO EXISTE “CONCORRÊNCIA DE PAINÉIS”
============================================================

O uso de múltiplos blocos:

    if painel == "...":
        ...
        st.stop()

NÃO gera concorrência, pois:

- apenas UM valor de `painel` existe por execução
- st.stop() encerra o fluxo imediatamente
- nenhum painel abaixo é executado

⚠️ Qualquer remoção de st.stop() em painéis existentes
é considerada ALTERAÇÃO ESTRUTURAL CRÍTICA.

============================================================
🔴 PAINÉIS CRÍTICOS — NÃO TOCAR
============================================================

Os painéis abaixo são ÁREAS CRÍTICAS DO SISTEMA:

- 🧭 Checklist Operacional — Decisão (AGORA)
- 🌊 ECO / PRÉ-ECO / Estado
- 📊 Sentinelas (k / k*)
- ⚙️ Modo 6
- ⚙️ Modo TURBO++ ULTRA
- 📄 Relatório Final

Regras:
- NÃO automatizar decisão
- NÃO alterar sem auditoria formal
- NÃO “corrigir resultado”
- NÃO introduzir heurística silenciosa

============================================================
🟡 PAINÉIS OBSERVACIONAIS
============================================================

- Modo Guiado Oficial
- Painel de Ruído Condicional
- Painel de Divergência S6 vs MC
- MVPs do Modo Especial
- Painéis V16 retrospectivos

Regras:
- podem evoluir conceitualmente
- NÃO podem decidir
- NÃO podem gerar listas
- NÃO podem aprender

============================================================
🟢 PAINÉIS EXPLORATÓRIOS
============================================================

- Visualizações auxiliares
- Diagnósticos paralelos
- Leituras de apoio

Regras:
- nunca substituem painéis críticos
- nunca alteram fluxo principal
- podem ser removidos se inúteis

============================================================
🧠 PRINCÍPIOS DE EVOLUÇÃO (NORMATIVOS)
============================================================

- Toda evolução é ADITIVA
- Nada é substitutivo
- Leitura precede ação
- Observação precede decisão
- Decisão é HUMANA
- Aprendizado ocorre APÓS o evento
- Correções ocorrem NA ORIGEM, nunca por filtro tardio

============================================================
📌 REGRA FINAL
============================================================

Se houver dúvida entre:
- “melhorar”
- “refatorar”
- “organizar”
- “otimizar”

A ação padrão é:
❌ NÃO MEXER

Somente evoluir quando:
- o sistema pedir
- os dados sustentarem
- o impacto for analisado

============================================================
📜 FIM DO BLOCO CANÔNICO DE GOVERNANÇA — PREDICTCARS
============================================================
"""

import streamlit as st
# =====================================================================
# 📜 TEXTO CANÔNICO — CONTRATO OPERACIONAL DO PREDICTCARS
# Versão: V15.7 MAX
#
# ESTE BLOCO É A ÂNCORA DO SISTEMA.
# Ele define COMO o PredictCars deve ser usado, evoluído e interpretado.
#
# REGRA ABSOLUTA:
# - A decisão acontece UMA ÚNICA VEZ, ANTES do evento
# - O aprendizado acontece SOMENTE DEPOIS do evento
# - Nunca ao mesmo tempo
#
# SEQUÊNCIA OFICIAL — AGORA (DECIDIR E JOGAR)
# A0: Carregar Histórico
# A1: Leitura Estrutural do Ambiente (k*, risco, histórico)
# A2: Qualidade do Sinal (ruído, divergência)
# A3: Ritmo do Alvo (replays)
# A4: DECISÃO ÚNICA (Checklist Operacional)
# A5: Motor (Pipeline / TURBO)
# A6: EIXO 2 — Momento & Antecipação (Backtest do Pacote)
# A7: Execução (Modo 6, Relatório)
#
# SEQUÊNCIA OFICIAL — DEPOIS (APRENDER)
# B0: Incorporação da Série Real
# B1: Diagnóstico de Regime
# B2: EIXO 3 — Passageiros (ECO / ANTI-EXATO)
# B3: Ajuste de Lente (humano, não automático)
#
# OS 3 EIXOS DO SISTEMA
# EIXO 1: Estrutura das Listas (higiene, coerência, prontidão)
# EIXO 2: Momento & Antecipação (ritmo + backtest)
# EIXO 3: Aprendizado (regime + passageiros)
#
# MODO ESPECIAL:
# - Nunca substitui o modo normal
# - Nunca cria listas do zero
# - Atua apenas sobre pacotes já gerados
#
# QUALQUER EVOLUÇÃO FUTURA DO PREDICTCARS
# DEVE RESPEITAR ESTE CONTRATO.
# =====================================================================

st.sidebar.warning("Rodando arquivo: app_v15_7_MAX.py")
# ============================================================
# Predict Cars V15.7 MAX — V16 PREMIUM PROFUNDO
# Núcleo + Coberturas + Interseção Estatística
# Pipeline V14-FLEX ULTRA + Replay LIGHT/ULTRA + TURBO++ HÍBRIDO
# + TURBO++ ULTRA + Painel de Ruído Condicional
# + Painel de Divergência S6 vs MC + Monitor de Risco (k & k*)
# + Testes de Confiabilidade REAL + Modo 6 Acertos V15.7 MAX
# + Relatório Final COMPLETO V15.7 MAX
# Arquivo oficial: app_v15_7_MAX.py
# ============================================================
import math
import itertools
import textwrap
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# DEBUG TEMPORÁRIO — PROVA DE EXECUÇÃO DO ARQUIVO
st.sidebar.caption("🧪 DEBUG: arquivo carregado")
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# V16 — GUARDA CANÔNICA (ANTI-NAMEERROR) — TOPO DO ARQUIVO
# (DESATIVADA — substituída pela CAMADA D real)
# Mantida apenas como registro histórico
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# def v16_registrar_estado_alvo():
#     return {
#         "tipo": "indefinido",
#         "velocidade": "indefinida",
#         "comentario": "Estado ainda não disponível (carregue histórico e rode Sentinelas/Pipeline).",
#     }

# def v16_registrar_expectativa():
#     return {
#         "previsibilidade": "indefinida",
#         "erro_esperado": "indefinido",
#         "chance_janela_ouro": "baixa",
#         "comentario": "Expectativa ainda não disponível (carregue histórico e rode Sentinelas/Pipeline).",
#     }

# def v16_registrar_volume_e_confiabilidade():
#     return {
#         "minimo": "-",
#         "recomendado": "-",
#         "maximo_tecnico": "-",
#         "confiabilidades_estimadas": {},
#         "comentario": "Volume ainda não disponível (carregue histórico e rode Sentinelas/Pipeline).",
#     }

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# FIM — V16 — GUARDA CANÔNICA (ANTI-NAMEERROR) — DESATIVADA
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



# ============================================================
# FUNÇÃO — CARREGAMENTO UNIVERSAL DE HISTÓRICO (FLEX ULTRA)
# REGRA FIXA:
# - Último valor da linha = k
# - Quantidade de passageiros é LIVRE
# - Universo é derivado do histórico (SANIDADE)
# ============================================================

def carregar_historico_universal(linhas):
    """
    Formato esperado (exemplos válidos):
    C10;20;32;49;54;62;0
    C5790;4;5;6;23;35;43;0
    C15;01;02;03;04;05;06;07;08;09;10;1
    """

    registros = []
    universo_detectado = []

    for idx, linha in enumerate(linhas, start=1):
        linha = linha.strip()

        if not linha:
            continue

        partes = linha.split(";")

        if len(partes) < 3:
            raise ValueError(f"Linha {idx} inválida (campos insuficientes): {linha}")

        try:
            valores = partes[1:]          # ignora identificador
            k = int(valores[-1])          # último valor é k
            passageiros = [int(x) for x in valores[:-1]]
        except ValueError:
            raise ValueError(f"Linha {idx} contém valores não numéricos: {linha}")

        if not passageiros:
            raise ValueError(f"Linha {idx} sem passageiros válidos: {linha}")

        # coleta universo real
        universo_detectado.extend(passageiros)

        registro = {f"p{i+1}": p for i, p in enumerate(passageiros)}
        registro["k"] = k
        registro["serie"] = idx

        registros.append(registro)

    if not registros:
        raise ValueError("Histórico vazio ou inválido.")

    df = pd.DataFrame(registros)

    # ------------------------------------------------------------
    # SANIDADE DO UNIVERSO — CANÔNICA
    # ------------------------------------------------------------
    try:
        universo_max = int(max(universo_detectado))
        st.session_state["universo_min"] = 1
        st.session_state["universo_max"] = universo_max
    except Exception:
        st.session_state["universo_min"] = 1
        st.session_state["universo_max"] = None

    return df



# ============================================================
# V16 PREMIUM — IMPORTAÇÃO OFICIAL
# (Não altera nada do V15.7, apenas registra os painéis novos)
# ============================================================
from app_v16_premium import (
    v16_obter_paineis,
    v16_renderizar_painel,
)

# ============================================================
# Configuração da página (obrigatório V15.7 MAX)
# ============================================================
st.set_page_config(
    page_title="Predict Cars V15.7 MAX — V16 Premium",
    page_icon="🚗",
    layout="wide",
)

# ============================================================
# Estilos globais — preservando jeitão V14-FLEX + V15.6 MAX
# ============================================================
st.markdown(
    """
    <style>
    .big-title { font-size: 32px; font-weight: bold; }
    .sub-title { font-size: 22px; font-weight: bold; margin-top: 25px; }
    .danger { color: red; font-weight: bold; }
    .success { color: green; font-weight: bold; }
    .warning { color: orange; font-weight: bold; }
    .gray-text { color: #888; }
    .info-box {
        background-color: #f5f5f5;
        padding: 10px 15px;
        border-left: 4px solid #4c8bf5;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Sessão Streamlit — persistência para V15.7 MAX
# ============================================================

# Inicialização de estado
if "historico_df" not in st.session_state:
    st.session_state["historico_df"] = None

if "ultima_previsao" not in st.session_state:
    st.session_state["ultima_previsao"] = None

if "sentinela_kstar" not in st.session_state:
    st.session_state["sentinela_kstar"] = None

if "diagnostico_risco" not in st.session_state:
    st.session_state["diagnostico_risco"] = None

if "n_alvo" not in st.session_state:
    st.session_state["n_alvo"] = None


# ============================================================
# DETECÇÃO CANÔNICA DE n_alvo (PASSAGEIROS REAIS DA RODADA)
# REGRA FIXA:
# - Última coluna SEMPRE é k
# - Todas as colunas p* anteriores são passageiros
# - n_alvo é definido pela ÚLTIMA SÉRIE VÁLIDA
# ============================================================

def detectar_n_alvo(historico_df):
    if historico_df is None or historico_df.empty:
        return None

    col_pass = [c for c in historico_df.columns if c.startswith("p")]
    if not col_pass:
        return None

    ultima_linha = historico_df[col_pass].iloc[-1]
    return int(ultima_linha.dropna().shape[0])


# Atualização automática de n_alvo
if st.session_state.get("historico_df") is not None:
    st.session_state["n_alvo"] = detectar_n_alvo(
        st.session_state["historico_df"]
    )

# ============================================================
# V16 PREMIUM — INFRAESTRUTURA UNIVERSAL
# (REGRAS CANÔNICAS + ORÇAMENTO CONDICIONADO)
# ============================================================

# -----------------------------
# REGRA CANÔNICA: LISTAS >= n_real
# -----------------------------
def validar_lista_vs_n_real(lista, n_real):
    return isinstance(lista, (list, tuple)) and len(lista) >= int(n_real)

# -----------------------------
# ORÇAMENTOS CONDICIONADOS (TABELAS)
# -----------------------------
ORCAMENTOS_CONDICIONADOS = {
    5: {
        5: 3,
        6: 18,
        7: 63,
        8: 168,
        9: 378,
        10: 756,
    },
    6: {
        6: 6,
        7: 42,
        8: 168,
        9: 504,
        10: 1260,
        11: 2772,
    },
    15: {
        15: 3.5,
        16: 56,
        17: 476,
    },
}

# -----------------------------
# RESOLUÇÃO DE ORÇAMENTO
# -----------------------------
def resolver_orcamento(n_real, tamanho_lista, orcamento_manual=None):
    """
    Prioridade:
    1) Orçamento manual (se fornecido)
    2) Tabela condicionada por n_real
    3) None (não avalia custo)
    """
    if orcamento_manual is not None:
        try:
            return float(orcamento_manual)
        except Exception:
            return None

    tabela = ORCAMENTOS_CONDICIONADOS.get(int(n_real))
    if not tabela:
        return None

    return tabela.get(int(tamanho_lista))

# -----------------------------
# AVALIAÇÃO UNIVERSAL (OBSERVACIONAL)
# -----------------------------
def avaliar_listas_universal(listas, alvo_real, n_real, orcamento_manual=None):
    """
    Retorna métricas OBSERVACIONAIS:
    - acertos / n_real
    - custo (se disponível)
    """
    resultados = []
    alvo_set = set(int(v) for v in alvo_real if int(v) > 0)

    for idx, lst in enumerate(listas, start=1):
        if not validar_lista_vs_n_real(lst, n_real):
            continue

        lst_set = set(int(v) for v in lst if int(v) > 0)
        acertos = len(alvo_set.intersection(lst_set))
        custo = resolver_orcamento(n_real, len(lst), orcamento_manual)

        resultados.append({
            "lista_id": idx,
            "tamanho_lista": len(lst),
            "acertos": acertos,
            "n_real": int(n_real),
            "score": f"{acertos}/{int(n_real)}",
            "custo": custo,
        })

    return resultados





# ============================================================
# GUARDAS DE SEGURANÇA POR n_alvo
# (INFRAESTRUTURA — NÃO APLICADA A NENHUM PAINEL)
# ============================================================

def guarda_n_alvo(n_esperado, nome_modulo):
    n_alvo = st.session_state.get("n_alvo")

    if n_alvo is None:
        st.warning(
            f"⚠️ {nome_modulo}: n_alvo não detectado. "
            f"Carregue um histórico válido antes de executar este painel."
        )
        return False

    if n_alvo != n_esperado:
        st.warning(
            f"🚫 {nome_modulo} BLOQUEADO\n\n"
            f"n detectado = {n_alvo}\n"
            f"n esperado por este módulo = {n_esperado}\n\n"
            f"Este painel assume n fixo e foi bloqueado para evitar "
            f"cálculo incorreto ou truncamento silencioso."
        )
        return False

    return True



# ============================================================
# V16 PREMIUM — INSTRUMENTAÇÃO RETROSPECTIVA (ERRO POR REGIME)
# (PAINEL OBSERVACIONAL PERMANENTE — NÃO MUDA MOTOR)
# ============================================================

def _pc16_normalizar_series_6(historico_df: pd.DataFrame) -> np.ndarray:
    """
    Extrai exatamente as colunas p1..p6 do histórico V15.7 MAX.
    Retorna matriz shape (N, 6) com cada série ordenada.
    """
    if historico_df is None or historico_df.empty:
        return np.zeros((0, 6), dtype=float)

    colunas_esperadas = ["p1", "p2", "p3", "p4", "p5", "p6"]
    for c in colunas_esperadas:
        if c not in historico_df.columns:
            return np.zeros((0, 6), dtype=float)

    try:
        dfp = historico_df[colunas_esperadas].astype(float).dropna()
    except Exception:
        return np.zeros((0, 6), dtype=float)

    if len(dfp) < 10:
        return np.zeros((0, 6), dtype=float)

    arr = dfp.values
    arr.sort(axis=1)
    return arr



def _pc16_distancia_media(v: np.ndarray, centro: np.ndarray) -> float:
    """
    Distância média absoluta (L1 média) entre vetor de 6 e centro de 6.
    """
    return float(np.mean(np.abs(v - centro)))



def pc16_calcular_continuidade_por_janelas(
    historico_df: pd.DataFrame,
    janela: int = 60,
    step: int = 1,
    usar_quantis: bool = True
) -> Dict[str, Any]:
    """
    Analisa retrospectivamente o histórico em janelas móveis.
    Para cada janela [t-janela, t), calcula:
      - 'dx_janela': dispersão média das séries da janela em relação ao centróide da janela
      - 'erro_prox': erro da PRÓXIMA série (t) em relação ao centróide da janela (proxy de 'erro contido')
    Classifica regime por dx_janela (ECO / PRE / RUIM) e compara erro_prox por regime.

    Retorna dict com DataFrame e resumo.
    """
    X = _pc16_normalizar_series_6(historico_df)
    n = X.shape[0]
    if n < (janela + 5):
        return {
            "ok": False,
            "motivo": f"Histórico insuficiente para janela={janela}. Séries válidas: {n}.",
            "df": pd.DataFrame(),
            "resumo": {}
        }

    rows = []
    # percorre janelas, garantindo que exista a "próxima" série t
    for t in range(janela, n - 1, step):
        bloco = X[t - janela:t, :]
        centro = np.mean(bloco, axis=0)

        # dx_janela: média das distâncias das séries da janela ao centróide
        dists = [ _pc16_distancia_media(bloco[i], centro) for i in range(bloco.shape[0]) ]
        dx_janela = float(np.mean(dists))

        # erro_prox: distância da série seguinte (t) ao centróide da janela
        prox = X[t, :]
        erro_prox = _pc16_distancia_media(prox, centro)

        rows.append({
            "t": t,  # índice da série (0-based dentro do array)
            "dx_janela": dx_janela,
            "erro_prox": erro_prox
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return {
            "ok": False,
            "motivo": "Não foi possível gerar janelas (df vazio).",
            "df": pd.DataFrame(),
            "resumo": {}
        }

    # Classificação de regime (ECO/PRE/RUIM) baseada em dx_janela
    if usar_quantis:
        q1 = float(df["dx_janela"].quantile(0.33))
        q2 = float(df["dx_janela"].quantile(0.66))
    else:
        # fallback conservador: thresholds fixos (raramente usado)
        q1, q2 = 0.30, 0.45

    def _rotulo(dx: float) -> str:
        if dx <= q1:
            return "ECO"
        elif dx <= q2:
            return "PRE"
        return "RUIM"

    df["regime"] = df["dx_janela"].apply(_rotulo)

    # Métricas resumo
    resumo = {}
    for reg in ["ECO", "PRE", "RUIM"]:
        sub = df[df["regime"] == reg]
        if len(sub) == 0:
            resumo[reg] = {"n": 0}
            continue

        resumo[reg] = {
            "n": int(len(sub)),
            "dx_janela_medio": float(sub["dx_janela"].mean()),
            "erro_prox_medio": float(sub["erro_prox"].mean()),
            "erro_prox_mediana": float(sub["erro_prox"].median()),
        }

    # Métrica única que queremos: diferença ECO vs RUIM no erro_prox médio
    if resumo.get("ECO", {}).get("n", 0) > 0 and resumo.get("RUIM", {}).get("n", 0) > 0:
        diff = resumo["RUIM"]["erro_prox_medio"] - resumo["ECO"]["erro_prox_medio"]
    else:
        diff = None

    resumo_geral = {
        "janela": int(janela),
        "step": int(step),
        "q1_dx": q1,
        "q2_dx": q2,
        "diff_ruim_menos_eco_no_erro": diff,
        "n_total_janelas": int(len(df))
    }

    return {
        "ok": True,
        "motivo": "",
        "df": df,
        "resumo": resumo,
        "resumo_geral": resumo_geral
    }



# ============================================================
# Função utilitária — formatador geral
# ============================================================
def formatar_lista_passageiros(lista: List[int]) -> str:
    """Formata lista no padrão compacto V15.7 MAX"""
    return ", ".join(str(x) for x in lista)

# ============================================================
# Parsing FLEX ULTRA — versão robusta V15.7 MAX
# ============================================================
def analisar_historico_flex_ultra(conteudo: str) -> pd.DataFrame:
    """
    Parser oficial V15.7 MAX — leitura de histórico com:
    - prefixo C1, C2, C3 ...
    - 5 ou 6 passageiros
    - sensor k sempre na última coluna
    """
    linhas = conteudo.strip().split("\n")
    registros = []

    for linha in linhas:
        partes = linha.replace(" ", "").split(";")
        if len(partes) < 7:
            continue

        try:
            serie = partes[0]
            nums = list(map(int, partes[1:-1]))
            k_val = int(partes[-1])
            registros.append([serie] + nums + [k_val])
        except:
            continue

    colunas = ["serie", "p1", "p2", "p3", "p4", "p5", "p6", "k"]
    df = pd.DataFrame(registros, columns=colunas[: len(registros[0])])

    return df

# ============================================================
# Utilitários de texto e apresentação — V15.7 MAX
# ============================================================
def texto_em_blocos(texto: str, largura: int = 100) -> List[str]:
    if not texto:
        return []
    return textwrap.wrap(texto, width=largura)


def exibir_bloco_mensagem(
    titulo: str,
    corpo: str,
    tipo: str = "info",
) -> None:

    blocos = texto_em_blocos(corpo, largura=110)

    if tipo == "info":
        st.info(f"**{titulo}**\n\n" + "\n\n".join(blocos))
    elif tipo == "warning":
        st.warning(f"**{titulo}**\n\n" + "\n\n".join(blocos))
    elif tipo == "error":
        st.error(f"**{titulo}**\n\n" + "\n\n".join(blocos))
    elif tipo == "success":
        st.success(f"**{titulo}**\n\n" + "\n\n".join(blocos))
    else:
        st.markdown(
            f"""
            <div class="info-box">
                <div class="sub-title">{titulo}</div>
                <p>{"<br>".join(blocos)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# Configurações Anti-Zumbi — limites globais
# ============================================================
LIMITE_SERIES_REPLAY_ULTRA: int = 8000
LIMITE_SERIES_TURBO_ULTRA: int = 8000
LIMITE_PREVISOES_TURBO: int = 600
LIMITE_PREVISOES_MODO_6: int = 800


def limitar_operacao(
    qtd_series: int,
    limite_series: int,
    contexto: str = "",
    painel: str = "",
) -> bool:

    if qtd_series is None:
        return True

    if qtd_series <= limite_series:
        return True

    msg = (
        f"🔒 **Operação bloqueada pela Proteção Anti-Zumbi ({contexto}).**\n\n"
        f"- Séries detectadas: **{qtd_series}**\n"
        f"- Limite seguro: **{limite_series}**\n"
        f"Painel: **{painel}**\n\n"
        "👉 Evitamos travamento no Streamlit."
    )
    exibir_bloco_mensagem("Proteção Anti-Zumbi", msg, tipo="warning")
    return False


# ============================================================
# NÚCLEO V16 — Premium Profundo (Diagnóstico & Calibração)
# Compatível com V15.7 MAX, 100% opcional e retrocompatível
# ============================================================
from typing import Dict, Any, Optional, Tuple  # Reimportar não faz mal


def v16_identificar_df_base() -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    """
    Tenta descobrir qual DataFrame de histórico está ativo no app.
    Busca em chaves comuns do st.session_state para não quebrar nada.
    Se não encontrar nada, retorna (None, None).
    """
    candidatos = []
    for chave in ["historico_df", "df_historico", "df_base", "df", "df_hist"]:
        if chave in st.session_state:
            objeto = st.session_state[chave]
            if isinstance(objeto, pd.DataFrame) and not objeto.empty:
                candidatos.append((chave, objeto))

    if not candidatos:
        return None, None

    chave_escolhida, df_escolhido = candidatos[0]
    return chave_escolhida, df_escolhido


def v16_resumo_basico_historico(
    df: pd.DataFrame,
    limite_linhas: int = 3000,
) -> Dict[str, Any]:
    """
    Gera um resumo leve do histórico para diagnóstico:
    - Quantidade total de séries
    - Janela usada para diagnóstico (anti-zumbi)
    - Distribuição de k (se existir)
    - Presença de colunas relevantes (k*, NR%, QDS)
    Tudo protegido contra KeyError e DataFrames pequenos.
    """
    resumo: Dict[str, Any] = {}

    n_total = int(len(df))
    if n_total <= 0:
        resumo["n_total"] = 0
        resumo["n_usado"] = 0
        resumo["colunas"] = list(df.columns)
        resumo["dist_k"] = {}
        resumo["info_extra"] = {}
        return resumo

    limite_seguro = max(100, min(limite_linhas, n_total))
    df_uso = df.tail(limite_seguro).copy()

    resumo["n_total"] = n_total
    resumo["n_usado"] = int(len(df_uso))
    resumo["colunas"] = list(df_uso.columns)

    dist_k: Dict[Any, int] = {}
    if "k" in df_uso.columns:
        try:
            contagem_k = df_uso["k"].value_counts().sort_index()
            for k_val, qtd in contagem_k.items():
                dist_k[int(k_val)] = int(qtd)
        except Exception:
            dist_k = {}
    resumo["dist_k"] = dist_k

    info_extra: Dict[str, Any] = {}
    for col in df_uso.columns:
        col_lower = str(col).lower()
        if "k*" in col_lower or "k_est" in col_lower or "kstar" in col_lower:
            info_extra["tem_k_estrela"] = True
        if "nr" in col_lower and "%" in col_lower:
            info_extra["tem_nr_percent"] = True
        if "qds" in col_lower:
            info_extra["tem_qds"] = True
    resumo["info_extra"] = info_extra

    return resumo


def v16_mapear_confiabilidade_session_state() -> Dict[str, Any]:
    """
    Varre st.session_state e tenta localizar informações de confiabilidade,
    QDS, k*, NR%, etc., sem assumir nomes fixos.
    Não quebra o app se nada for encontrado.
    """
    mapeamento: Dict[str, Any] = {}

    try:
        for chave, valor in st.session_state.items():
            nome_lower = str(chave).lower()
            if any(token in nome_lower for token in ["confiab", "qds", "k_estrela", "k*", "nr%", "ruido"]):
                if isinstance(valor, (int, float, str)):
                    mapeamento[chave] = valor
                elif isinstance(valor, dict):
                    mapeamento[chave] = {"tipo": "dict", "tamanho": len(valor)}
                elif isinstance(valor, pd.DataFrame):
                    mapeamento[chave] = {
                        "tipo": "DataFrame",
                        "linhas": len(valor),
                        "colunas": list(valor.columns)[:10],
                    }
                else:
                    mapeamento[chave] = {"tipo": type(valor).__name__}
    except Exception:
        pass

    return mapeamento


# ============================================================
# Métricas básicas do histórico — V15.7 MAX
# ============================================================
def calcular_metricas_basicas_historico(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula métricas simples do histórico para uso em:
    - Painel de carregamento
    - Monitor de Risco — k & k*
    Tudo de forma leve, sem quebrar se faltarem colunas.
    """
    metricas: Dict[str, Any] = {}

    if df is None or df.empty:
        metricas["qtd_series"] = 0
        metricas["min_k"] = None
        metricas["max_k"] = None
        metricas["media_k"] = 0.0
        return metricas

    metricas["qtd_series"] = int(len(df))

    if "k" in df.columns:
        try:
            k_vals = df["k"].astype(float)
            metricas["min_k"] = float(k_vals.min())
            metricas["max_k"] = float(k_vals.max())
            metricas["media_k"] = float(k_vals.mean())
        except Exception:
            metricas["min_k"] = None
            metricas["max_k"] = None
            metricas["media_k"] = 0.0
    else:
        metricas["min_k"] = None
        metricas["max_k"] = None
        metricas["media_k"] = 0.0

    return metricas


def exibir_resumo_inicial_historico(metricas: Dict[str, Any]) -> None:
    """
    Exibe um resumo amigável logo após o carregamento do histórico.
    Usado no Painel 1 (Carregar Histórico) e como base para o Monitor de Risco.
    """
    qtd_series = metricas.get("qtd_series", 0)
    min_k = metricas.get("min_k")
    max_k = metricas.get("max_k")
    media_k = metricas.get("media_k", 0.0)

    corpo = (
        f"- Séries carregadas: **{qtd_series}**\n"
        f"- k mínimo: **{min_k}** · k máximo: **{max_k}** · k médio: **{media_k:.2f}**\n"
    )

    exibir_bloco_mensagem(
        "Resumo inicial do histórico (V15.7 MAX)",
        corpo,
        tipo="info",
    )

# ============================================================
# Cabeçalho visual principal
# ============================================================
st.markdown(
    '<div class="big-title">🚗 Predict Cars V15.7 MAX — V16 PREMIUM PROFUNDO</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p class="gray-text">
    Núcleo + Coberturas + Interseção Estatística · Pipeline V14-FLEX ULTRA ·
    Replay LIGHT/ULTRA · TURBO++ HÍBRIDO · TURBO++ ULTRA · Monitor de Risco (k & k*) ·
    Painel de Ruído Condicional · Divergência S6 vs MC · Testes de Confiabilidade REAL ·
    Modo 6 Acertos V15.7 MAX · Relatório Final Integrado.
    </p>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Construção da  — V15.7 MAX
# (LAUDO DE CÓDIGO — BLOCO 1-FIX | ORDEM OPERACIONAL FINAL)
# ============================================================

def construir_navegacao_v157() -> str:

    st.sidebar.markdown("## 🚦  PredictCars V15.7 MAX")
    st.sidebar.markdown("📌 Selecione o painel:")

    n_alvo = st.session_state.get("n_alvo")

    # =========================================================
    # ORDEM OPERACIONAL DEFINITIVA — USO DIÁRIO
    # =========================================================
    opcoes = [

        # -----------------------------------------------------
        # BLOCO 0 — ENTRADA
        # -----------------------------------------------------
        "📁 Carregar Histórico (Arquivo)",
        "📄 Carregar Histórico (Colar)",

        # -----------------------------------------------------
        # BLOCO 1 — ORIENTAÇÃO
        # -----------------------------------------------------
        "🧭 Modo Guiado Oficial — PredictCars",

        # -----------------------------------------------------
        # BLOCO 2 — LEITURA DO AMBIENTE
        # -----------------------------------------------------
        "🛰️ Sentinelas — k* (Ambiente de Risco)",
        "🧭 Monitor de Risco — k & k*",
        "📊 Observação Histórica — Eventos k",
        "⏱️ Duração da Janela — Análise Histórica",

        # -----------------------------------------------------
        # BLOCO 3 — QUALIDADE DO SINAL
        # -----------------------------------------------------
        "📡 Painel de Ruído Condicional",
        "📉 Painel de Divergência S6 vs MC",

        # -----------------------------------------------------
        # BLOCO 4 — RITMO DO ALVO
        # -----------------------------------------------------
        "🔁 Replay LIGHT",
        "🔁 Replay ULTRA",
        "🧪 Replay Curto — Expectativa 1–3 Séries",

        # -----------------------------------------------------
        # BLOCO 5 — EIXO 1 | ESTRUTURA DAS LISTAS
        # -----------------------------------------------------
        "🧼 B1 — Higiene de Passageiros",
        "🧩 B2 — Coerência Interna das Listas",
        "🟢 B3 — Prontidão (Refinamento)",
        "🟣 B4 — Refinamento Leve de Passageiros",

        # -----------------------------------------------------
        # BLOCO 6 — DECISÃO ÚNICA
        # -----------------------------------------------------
        "🧭 Checklist Operacional — Decisão (AGORA)",

        # -----------------------------------------------------
        # BLOCO 7 — MOTOR
        # -----------------------------------------------------
        "🛣️ Pipeline V14-FLEX ULTRA",
        "⚙️ Modo TURBO++ HÍBRIDO",
        "⚙️ Modo TURBO++ ULTRA",

        # -----------------------------------------------------
        # BLOCO 7.5 — EIXO 2 | MOMENTO & ANTECIPAÇÃO
        # -----------------------------------------------------
        "📊 V16 Premium — Backtest Rápido do Pacote (N=60)",

        # -----------------------------------------------------
        # BLOCO 8 — EXECUÇÃO
        # -----------------------------------------------------
        "🎯 Modo 6 Acertos — Execução",
        "🧪 Testes de Confiabilidade REAL",
        "📘 Relatório Final",

        # -----------------------------------------------------
        # BLOCO 9 — EXTENSÃO
        # -----------------------------------------------------
        "🔵 MODO ESPECIAL — Evento Condicionado",

        # -----------------------------------------------------
        # BLOCO 10 — CAMADA UNIVERSAL
        # -----------------------------------------------------
        "💰 MVP-U2 — Orçamento Universal",
        "🧩 MVP-U3 — Cobertura Universal",
        "📈 MVP-U4 — Eficiência Marginal por Custo",

        # -----------------------------------------------------
        # BLOCO 11 — DEPOIS | APRENDIZADO (EIXO 3)
        # -----------------------------------------------------
        "🧠 Memória Operacional (Observacional)",
        "🧠 Memória Operacional — Registro Semi-Automático",
        "🧠 Laudo Operacional V16",
        "🧠 Diagnóstico ECO & Estado (V16)",
        "📊 V16 Premium — Erro por Regime (Retrospectivo)",
        "📊 V16 Premium — EXATO por Regime (Proxy)",
        "📊 V16 Premium — PRÉ-ECO → ECO (Persistência & Continuidade)",
        "📊 V16 Premium — Passageiros Recorrentes em ECO (Interseção)",
        "📊 V16 Premium — PRÉ-ECO | Contribuição de Passageiros",
        "📊 V16 Premium — ANTI-EXATO | Passageiros Nocivos",
        "🎯 Compressão do Alvo — Observacional (V16)",
        "🔮 V16 Premium Profundo — Diagnóstico & Calibração",
    ]

    # ------------------------------------------------------------
    # INSERÇÃO CONDICIONAL — MODO N EXPERIMENTAL (n≠6)
    # ------------------------------------------------------------
    if (n_alvo is not None) and (int(n_alvo) != 6):
        opcoes.insert(
            opcoes.index("🎯 Modo 6 Acertos — Execução"),
            "🧪 Modo N Experimental (n≠6)"
        )

    painel = st.sidebar.radio("", opcoes, index=0, key="NAV_V157_CANONICA")
    return painel


# ============================================================
# FIM — Construção da  — V15.7 MAX
# ============================================================



# ============================================================
# Ativação da Navegação — V15.7 MAX
# ============================================================

painel = construir_navegacao_v157()
st.sidebar.caption(f"Painel ativo: {painel}")

# ============================================================
# DEBUG — CARIMBO DE BUILD (QUAL ARQUIVO REALMENTE ESTÁ RODANDO)
# ============================================================
try:
    st.sidebar.markdown("---")
    st.sidebar.caption("✅ BUILD-ID: NAV_ORDEM_FINAL_2025-12-29A")
    st.sidebar.caption(f"📄 __file__: {__file__}")
    st.sidebar.caption(f"🔎 Primeiro item NAV: {construir_navegacao_v157.__name__}")
    st.sidebar.caption("🧭 TOP-5: (debug desativado — não chamar construir_navegacao_v157() aqui)")
except Exception as _e:
    st.sidebar.caption(f"⚠️ DEBUG build falhou: {_e}")



# ============================================================
# DEBUG MINIMAL — CONFIRMA PAINEL ATIVO
# (manter por enquanto para auditoria)
# ============================================================
st.sidebar.caption(f"Painel ativo: {painel}")



# ============================================================
# MODO ESPECIAL — EVENTO CONDICIONADO (C2955)
# AVALIAÇÃO MULTI-ORÇAMENTO | OBSERVACIONAL | 6 OU NADA
# ============================================================

def pc_especial_avaliar_pacote_contem_6(carro, alvo):
    """
    Retorna True se o carro contém TODOS os 6 números do alvo.
    Régua BINÁRIA: 6 ou nada.
    """
    try:
        return set(alvo).issubset(set(carro))
    except Exception:
        return False


def pc_especial_avaliar_historico_pacote(historico_df, pacote):
    """
    Percorre o histórico rodada a rodada e verifica se,
    em alguma rodada, algum carro do pacote contém os 6.
    Retorna contagem de sucessos.
    """
    if historico_df is None or historico_df.empty:
        return {
            "rodadas": 0,
            "sucessos": 0,
        }

    col_pass = [c for c in historico_df.columns if c.startswith("p")]
    rodadas = 0
    sucessos = 0

    for _, row in historico_df.iterrows():
        try:
            alvo = [int(row[c]) for c in col_pass[:6]]
        except Exception:
            continue

        rodadas += 1

        for carro in pacote:
            if pc_especial_avaliar_pacote_contem_6(carro, alvo):
                sucessos += 1
                break  # sucesso binário por rodada

    return {
        "rodadas": rodadas,
        "sucessos": sucessos,
    }


# ============================================================
# 🔵 MODO ESPECIAL — MVP2 (2–6 acertos + Estado do Alvo PROXY)
# OBSERVACIONAL | NÃO decide | NÃO gera pacotes | NÃO aprende
# ============================================================

def _pc_contar_hits_lista_vs_alvo(lista, alvo_set):
    """
    Retorna quantidade de acertos (interseção) entre uma lista (carro) e o alvo (set).
    """
    try:
        s = set(int(x) for x in lista)
    except Exception:
        return 0
    return len(s & alvo_set)


def _pc_melhor_hit_do_pacote(pacote_listas, alvo_set):
    """
    Dado um pacote (listas de previsão), retorna o MELHOR hit (0..6) encontrado contra o alvo.
    """
    if not pacote_listas:
        return 0

    best = 0
    for lst in pacote_listas:
        h = _pc_contar_hits_lista_vs_alvo(lst, alvo_set)
        if h > best:
            best = h
            if best >= 6:
                break
    return best


def _pc_extrair_carro_row(row):
    """
    Extrai os 6 passageiros da linha do histórico.
    Espera colunas p1..p6 (padrão do PredictCars).
    """
    try:
        return [int(row[f"p{i}"]) for i in range(1, 7)]
    except Exception:
        return None


def _pc_distancia_carros(carro_a, carro_b):
    """
    Distância simples entre dois carros (proxy):
    número de passageiros diferentes.
    """
    if carro_a is None or carro_b is None:
        return None
    try:
        return len(set(carro_a) ^ set(carro_b))
    except Exception:
        return None


def _pc_estado_alvo_proxy(dist):
    """
    Classificação simples do estado do alvo (proxy),
    baseada na distância entre carros consecutivos.
    """
    if dist is None:
        return "None"

    try:
        d = float(dist)
    except Exception:
        return "None"

    if d <= 1:
        return "parado"
    elif d <= 3:
        return "movimento_lento"
    else:
        return "movimento_brusco"


def pc_modo_especial_mvp2_avaliar_pacote(df_hist, pacote_listas):
    """
    MVP2:
    - Para cada série do histórico, computa:
        estado_alvo_proxy (parado/lento/brusco/None)
        melhor_hit (0..6) do pacote contra o alvo daquela série
    - Consolida em tabela: Estado x Hits(2..6) [contagem EXATA]
    Retorna (df_resumo, total_series_avaliadas).
    """
    if df_hist is None or df_hist.empty:
        return pd.DataFrame(), 0

    if not pacote_listas:
        return pd.DataFrame(), int(len(df_hist))

    cont = {
        "parado": {2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        "movimento_lento": {2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        "movimento_brusco": {2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        "None": {2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
    }

    rows = list(df_hist.iterrows())
    carro_prev = None

    for _, row in rows:
        carro_atual = _pc_extrair_carro_row(row)

        dist = (
            _pc_distancia_carros(carro_prev, carro_atual)
            if carro_prev is not None and carro_atual is not None
            else None
        )

        estado = _pc_estado_alvo_proxy(dist)
        estado_key = estado if estado in cont else "None"

        if carro_atual is None:
            carro_prev = carro_atual
            continue

        alvo_set = set(carro_atual)
        best_hit = _pc_melhor_hit_do_pacote(pacote_listas, alvo_set)

        if best_hit in [2, 3, 4, 5, 6]:
            cont[estado_key][best_hit] += 1

        carro_prev = carro_atual

    out = []
    for estado_key in ["parado", "movimento_lento", "movimento_brusco", "None"]:
        linha = {"Estado": estado_key}
        for h in [2, 3, 4, 5, 6]:
            linha[str(h)] = int(cont[estado_key][h])
        out.append(linha)

    df_out = pd.DataFrame(out)

    ordem = {"parado": 0, "movimento_lento": 1, "movimento_brusco": 2, "None": 3}
    df_out["__ord"] = df_out["Estado"].map(ordem).fillna(9).astype(int)
    df_out = df_out.sort_values("__ord").drop(columns=["__ord"])

    return df_out, int(len(df_hist))

# ============================================================
# 🔵 FIM — FUNÇÕES DO MODO ESPECIAL MVP2
# ============================================================


# ============================================================
# PAINEL — 🔵 MODO ESPECIAL (Evento Condicionado C2955)
# Avaliação MULTI-ORÇAMENTO | Observacional
# ============================================================

if painel == "🔵 MODO ESPECIAL — Evento Condicionado":

    st.markdown("## 🔵 MODO ESPECIAL — Evento Condicionado (C2955)")
    st.caption(
        "Avaliação OBSERVACIONAL de pacotes já gerados.\n\n"
        "✔ Régua extrema: **6 ou nada** (MVP1)\n"
        "✔ Avaliação realista: **2–6 por estado do alvo** (MVP2)\n"
        "✔ Sem aprendizado\n"
        "✔ Sem interferência no Modo Normal\n"
        "✔ Decisão HUMANA (Rogério + Auri)"
    )

    historico_df = st.session_state.get("historico_df")

    # ============================================================
    # 🔵 SELETOR DE FONTE DO PACOTE (TURBO × MODO 6)
    # OBSERVACIONAL | NÃO decide | NÃO aprende | NÃO interfere
    # ============================================================

    pacote_turbo_raw = st.session_state.get("ultima_previsao")

    pacote_m6_total = (
        st.session_state.get("modo6_listas_totais")
        or st.session_state.get("modo6_listas")
        or []
    )

    pacote_m6_top10 = st.session_state.get("modo6_listas_top10") or []

    fontes = []
    if pacote_turbo_raw:
        fontes.append("TURBO (núcleo)")
    if pacote_m6_total:
        fontes.append("MODO 6 (TOTAL)")
    if pacote_m6_top10:
        fontes.append("MODO 6 (TOP 10)")
    if pacote_turbo_raw and pacote_m6_total:
        fontes.append("MIX (TURBO + M6 TOTAL)")

    if not fontes:
        exibir_bloco_mensagem(
            "Pré-requisitos ausentes",
            "É necessário:\n"
            "- Histórico carregado\n"
            "- Pacotes gerados pelo TURBO ou Modo 6",
            tipo="warning",
        )
        st.stop()

    idx_default = fontes.index("MODO 6 (TOTAL)") if "MODO 6 (TOTAL)" in fontes else 0

    fonte_escolhida = st.selectbox(
        "Fonte do pacote para avaliação (observacional):",
        options=fontes,
        index=idx_default,
    )

    # -----------------------------
    # Construção do pacote ativo
    # -----------------------------
    if fonte_escolhida == "TURBO (núcleo)":
        pacotes_raw = pacote_turbo_raw
    elif fonte_escolhida == "MODO 6 (TOTAL)":
        pacotes_raw = pacote_m6_total
    elif fonte_escolhida == "MODO 6 (TOP 10)":
        pacotes_raw = pacote_m6_top10
    else:
        mix = []

        if isinstance(pacote_turbo_raw, list):
            if pacote_turbo_raw and isinstance(pacote_turbo_raw[0], int):
                mix.append(pacote_turbo_raw)
            else:
                mix.extend(pacote_turbo_raw)

        if isinstance(pacote_m6_total, list):
            mix.extend(pacote_m6_total)

        pacotes_raw = mix

    # ============================================================
    # ✅ NORMALIZAÇÃO FINAL — LISTA DE LISTAS
    # ============================================================
    if pacotes_raw is None:
        pacotes = []
    elif isinstance(pacotes_raw, list) and pacotes_raw and isinstance(pacotes_raw[0], int):
        pacotes = [pacotes_raw]
    elif isinstance(pacotes_raw, list):
        pacotes = pacotes_raw
    else:
        pacotes = []

    st.caption(
        f"Pacote ativo: **{fonte_escolhida}** | "
        f"Listas avaliadas: **{len(pacotes)}**"
    )

    if historico_df is None or historico_df.empty or not pacotes:
        exibir_bloco_mensagem(
            "Pré-requisitos ausentes",
            "Histórico vazio ou pacote inválido.",
            tipo="warning",
        )
        st.stop()

    # ============================================================
    # 🔵 MVP4 — ANÁLISE DE COMPOSIÇÕES DE COBERTURA (OBSERVACIONAL)
    # Núcleo / Fronteira automáticos — NÃO executa
    # ============================================================

    st.markdown("### 🔵 MVP4 — Análise de Composições de Cobertura")
    st.caption(
        "Painel analítico: sugere **composições candidatas** (6×6 até 1×9),\n"
        "com base em núcleo/fronteira extraídos automaticamente.\n"
        "❌ Não gera listas | ❌ Não decide | ❌ Não interfere"
    )

    from collections import Counter
    from math import comb

    todas = [n for lista in pacotes for n in lista]
    freq = Counter(todas)

    nucleo = sorted([n for n, c in freq.items() if c >= 3])
    fronteira = sorted([n for n, c in freq.items() if c == 2])
    ruido = sorted([n for n, c in freq.items() if c == 1])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🧱 Núcleo**")
        st.write(nucleo if nucleo else "—")
        if len(nucleo) < 4:
            st.warning("Núcleo fraco (<4).")
        if len(nucleo) > 5:
            st.warning("Núcleo grande (>5).")

    with col2:
        st.markdown("**🟡 Fronteira**")
        st.write(fronteira if fronteira else "—")
        if len(fronteira) > 6:
            st.warning("Fronteira extensa (ambiguidade elevada).")

    with col3:
        st.markdown("**🔴 Ruído**")
        st.write(ruido if ruido else "—")
        st.caption("Ruído excluído de carros >6.")

    st.markdown("#### 📦 Composições Candidatas (comparação teórica)")

    composicoes = [
        ("C1 — Foco puro", [(6, 6)]),
        ("C2 — Proteção leve", [(6, 4), (7, 1)]),
        ("C3 — Proteção + ambiguidade", [(6, 2), (7, 1), (8, 1)]),
        ("C4 — Envelope compacto", [(8, 1)]),
        ("C5 — Envelope amplo", [(9, 1)]),
    ]

    for nome, mix in composicoes:
        custo = 0
        combs = 0
        for m, q in mix:
            c = comb(m, 6)
            custo += c * 6 * q
            combs += c * q

        with st.expander(f"📘 {nome}"):
            st.write(f"Mix: {mix}")
            st.write(f"• Combinações de 6 cobertas: **{combs}**")
            st.write(f"• Custo teórico (régua): **{custo}**")

            if len(nucleo) < 4:
                st.warning("⚠️ Núcleo fraco — envelope pode diluir sinal.")
            if len(fronteira) > 6:
                st.warning("⚠️ Fronteira grande — risco de ilusão de cobertura.")

    # ============================================================
    # MVP2 — Avaliação 2–6 × Estado do Alvo (OBSERVACIONAL)
    # ============================================================

    st.markdown("### 📊 Resultado comparativo — MVP2 (2–6 × Estado do Alvo)")
    st.caption(
        "Leitura realista de aproximação.\n"
        "🟢 parado | 🟡 movimento lento | 🔴 movimento brusco\n"
        "O sistema **não decide**."
    )

    linhas = []

    orcamentos_disponiveis = [6, 42, 168, 504, 1260, 2772]

    orcamentos_sel = st.multiselect(
        "Selecione os orçamentos a avaliar (observacional):",
        options=orcamentos_disponiveis,
        default=[42],
    )

    if not orcamentos_sel:
        st.warning("Selecione ao menos um orçamento.")
        st.stop()

    for orc in orcamentos_sel:
        df_mvp2, total_series = pc_modo_especial_mvp2_avaliar_pacote(
            df_hist=historico_df,
            pacote_listas=pacotes,
        )

        if df_mvp2 is None or df_mvp2.empty:
            linhas.append({
                "Orçamento": orc,
                "Estado": "N/A",
                "Séries": int(total_series),
                "2": 0, "3": 0, "4": 0, "5": 0, "6": 0
            })
            continue

        for _, r in df_mvp2.iterrows():
            linhas.append({
                "Orçamento": int(orc),
                "Estado": str(r["Estado"]),
                "Séries": int(total_series),
                "2": int(r["2"]),
                "3": int(r["3"]),
                "4": int(r["4"]),
                "5": int(r["5"]),
                "6": int(r["6"]),
            })

    df_cmp = pd.DataFrame(linhas)
    st.dataframe(df_cmp, use_container_width=True, height=420)

    st.info(
        "📌 Interpretação HUMANA:\n"
        "- 🟢 Mais 4/5 em 'parado' → janela boa\n"
        "- 🟡 Predomínio de 3/4 → cautela\n"
        "- 🔴 Quase só 2/3 → reduzir agressividade\n"
        "- 6 é raro; 4/5 indicam proximidade real"
    )




# ============================================================
# CAMADA A — ESTADO DO ALVO (V16)
# Observador puro — NÃO decide, NÃO bloqueia, NÃO gera previsões
# ============================================================

def v16_diagnosticar_eco_estado():
    """
    Diagnóstico OBSERVACIONAL enriquecido (ECO A):
    - ECO: força + persistência + acionabilidade
    - ESTADO: parado / movimento_lento / movimento_brusco + confiabilidade
    NÃO altera motores | NÃO decide | NÃO bloqueia
    """

    historico_df = st.session_state.get("historico_df")

    # -----------------------------
    # Fallback seguro
    # -----------------------------
    if historico_df is None or historico_df.empty:
        diag = {
            "eco_forca": "indefinido",
            "eco_persistencia": "indefinida",
            "eco_acionabilidade": "não_acionável",
            "estado": "indefinido",
            "estado_confiavel": False,
            "contradicoes": ["histórico insuficiente"],
            "leitura_geral": "Histórico insuficiente para diagnóstico.",
        }
        st.session_state["diagnostico_eco_estado_v16"] = diag
        return diag

    # =========================================================
    # ECO — sinais já existentes
    # =========================================================
    k_star = st.session_state.get("sentinela_kstar")
    nr_pct = st.session_state.get("nr_percent")
    divergencia = st.session_state.get("div_s6_mc")

    sinais = 0
    motivos = []

    if isinstance(k_star, (int, float)) and k_star < 0.15:
        sinais += 1
        motivos.append("k* favorável")

    if isinstance(nr_pct, (int, float)) and nr_pct < 30:
        sinais += 1
        motivos.append("ruído controlado")

    if isinstance(divergencia, (int, float)) and divergencia < 5:
        sinais += 1
        motivos.append("baixa divergência")

    if sinais >= 3:
        eco_forca = "forte"
    elif sinais == 2:
        eco_forca = "médio"
    else:
        eco_forca = "fraco"

    # Persistência curta
    hist_eco = st.session_state.get("historico_eco_v16", [])
    hist_eco.append(eco_forca)
    hist_eco = hist_eco[-5:]
    st.session_state["historico_eco_v16"] = hist_eco

    eco_persistencia = "persistente" if hist_eco.count(eco_forca) >= 3 else "instável"

    # =========================================================
    # CONTRADIÇÕES (leitura turva)
    # =========================================================
    contradicoes = []
    if eco_forca in ("fraco", "médio") and isinstance(divergencia, (int, float)) and divergencia > 20:
        contradicoes.append("divergência elevada")
    if eco_persistencia == "persistente" and isinstance(divergencia, (int, float)) and divergencia > 30:
        contradicoes.append("persistência enganosa")

    # =========================================================
    # ACIONABILIDADE (OBSERVACIONAL)
    # =========================================================
    if eco_forca == "forte" and eco_persistencia == "persistente" and not contradicoes:
        eco_acionabilidade = "favorável"
    elif eco_forca in ("médio", "forte") and not contradicoes:
        eco_acionabilidade = "cautela"
    else:
        eco_acionabilidade = "não_acionável"

    # =========================================================
    # ESTADO DO ALVO (proxy existente)
    # =========================================================
    estado_proxy = None
    try:
        col_pass = [c for c in historico_df.columns if c.startswith("p")]
        if len(col_pass) >= 6 and len(historico_df) >= 2:
            a = [int(historico_df.iloc[-1][c]) for c in col_pass[:6]]
            b = [int(historico_df.iloc[-2][c]) for c in col_pass[:6]]
            dist = len(set(a) ^ set(b))
            if dist <= 1:
                estado_proxy = "parado"
            elif dist <= 3:
                estado_proxy = "movimento_lento"
            else:
                estado_proxy = "movimento_brusco"
    except Exception:
        estado_proxy = None

    if estado_proxy is None:
        estado = "indefinido"
        estado_confiavel = False
    else:
        estado = estado_proxy
        hist_estado = st.session_state.get("historico_estado_v16", [])
        hist_estado.append(estado)
        hist_estado = hist_estado[-5:]
        st.session_state["historico_estado_v16"] = hist_estado
        estado_confiavel = hist_estado.count(estado) >= 3

    # =========================================================
    # LEITURA FINAL (MASTIGADA)
    # =========================================================
    leitura = (
        f"ECO {eco_forca}, {eco_persistencia}, {eco_acionabilidade}. "
        f"Estado {estado}. "
        f"{'Confiável' if estado_confiavel else 'Em transição'}."
    )
    if contradicoes:
        leitura += " Atenção: " + "; ".join(contradicoes) + "."

    diagnostico = {
        "eco_forca": eco_forca,
        "eco_persistencia": eco_persistencia,
        "eco_acionabilidade": eco_acionabilidade,
        "estado": estado,
        "estado_confiavel": estado_confiavel,
        "contradicoes": contradicoes,
        "leitura_geral": leitura,
        "motivos_eco": motivos,
    }

    st.session_state["diagnostico_eco_estado_v16"] = diagnostico
    return diagnostico

# ============================================================
# ATIVAÇÃO SILENCIOSA — DIAGNÓSTICO ECO & ESTADO (V16)
# ============================================================
if "historico_df" in st.session_state:
    try:
        v16_diagnosticar_eco_estado()
    except Exception:
        pass




# ============================================================
# CAMADA B — EXPECTATIVA DE CURTO PRAZO (V16)
# Laudo observacional: horizonte 1–3 séries (NÃO decide)
# ============================================================


def v16_calcular_expectativa_curto_prazo(
    df: Optional[pd.DataFrame],
    estado_alvo: Optional[Dict[str, Any]],
    k_star: Optional[float],
    nr_percent: Optional[float],
    divergencia: Optional[float],
) -> Dict[str, Any]:

    if df is None or df.empty:
        return {
            "horizonte": "1–3 séries",
            "previsibilidade": "indefinida",
            "erro_esperado": "indefinido",
            "chance_janela_ouro": "baixa",
            "comentario": "Histórico insuficiente para expectativa.",
        }

    k = float(k_star) if isinstance(k_star, (int, float)) else 0.25
    nr = float(nr_percent) if isinstance(nr_percent, (int, float)) else 35.0
    div = float(divergencia) if isinstance(divergencia, (int, float)) else 4.0

    tipo = (estado_alvo or {}).get("tipo", "movimento_lento")

    # Índice simples de previsibilidade
    risco_norm = min(1.0, (nr / 70.0) * 0.4 + (div / 10.0) * 0.3 + (k / 0.5) * 0.3)
    previsibilidade_score = max(0.0, 1.0 - risco_norm)

    if previsibilidade_score >= 0.65:
        previsibilidade = "alta"
        erro = "baixo"
    elif previsibilidade_score >= 0.40:
        previsibilidade = "média"
        erro = "médio"
    else:
        previsibilidade = "baixa"
        erro = "alto"

    # Chance de janela de ouro (qualitativa)
    if tipo == "parado" and previsibilidade_score >= 0.60:
        chance_ouro = "alta"
    elif tipo == "movimento_lento" and previsibilidade_score >= 0.45:
        chance_ouro = "média"
    else:
        chance_ouro = "baixa"

    comentario = (
        f"Alvo {tipo}. Previsibilidade {previsibilidade}. "
        f"Erro esperado {erro}. Chance de janela de ouro {chance_ouro}."
    )

    return {
        "horizonte": "1–3 séries",
        "previsibilidade": previsibilidade,
        "erro_esperado": erro,
        "chance_janela_ouro": chance_ouro,
        "score_previsibilidade": round(previsibilidade_score, 4),
        "comentario": comentario,
    }


def v16_registrar_expectativa():
    estado = st.session_state.get("estado_alvo_v16")
    expectativa = v16_calcular_expectativa_curto_prazo(
        st.session_state.get("historico_df"),
        estado,
        st.session_state.get("sentinela_kstar"),
        st.session_state.get("nr_percent"),
        st.session_state.get("div_s6_mc"),
    )
    st.session_state["expectativa_v16"] = expectativa
    return expectativa

# ============================================================
# CAMADA C — VOLUME & CONFIABILIDADE (V16)
# Sistema INFORMA; humano DECIDE
# ============================================================

def v16_estimativa_confiabilidade_por_volume(
    estado_alvo: Optional[Dict[str, Any]],
    expectativa: Optional[Dict[str, Any]],
    base_confiabilidade: Optional[float] = None,
) -> Dict[int, float]:
    """
    Retorna um mapa {volume: confiabilidade_estimada}.
    Não bloqueia execução; apenas informa trade-offs.
    """
    tipo = (estado_alvo or {}).get("tipo", "movimento_lento")
    score_prev = (expectativa or {}).get("score_previsibilidade", 0.4)

    # Base de confiabilidade (fallback seguro)
    base = float(base_confiabilidade) if isinstance(base_confiabilidade, (int, float)) else score_prev

    # Ajuste por tipo de alvo
    if tipo == "parado":
        fator = 1.15
    elif tipo == "movimento_lento":
        fator = 1.00
    else:
        fator = 0.80

    volumes = [3, 6, 12, 20, 30, 50, 80]
    estimativas: Dict[int, float] = {}

    for v in volumes:
        # Ganho marginal decrescente
        ganho = 1.0 - (1.0 / max(1.0, np.log(v + 1)))
        conf = base * fator * ganho
        estimativas[v] = round(max(0.05, min(0.95, conf)), 3)

    return estimativas


def v16_calcular_volume_operacional(
    estado_alvo: Optional[Dict[str, Any]],
    expectativa: Optional[Dict[str, Any]],
    confiabilidades: Dict[int, float],
) -> Dict[str, Any]:
    """
    Consolida recomendações de volume sem impor decisão.
    """
    tipo = (estado_alvo or {}).get("tipo", "movimento_lento")
    prev = (expectativa or {}).get("previsibilidade", "média")

    # Volume recomendado por heurística qualitativa
    if tipo == "parado" and prev == "alta":
        recomendado = 30
    elif tipo == "movimento_lento":
        recomendado = 20
    else:
        recomendado = 6

    # Limites técnicos (anti-zumbi conceitual, não bloqueante)
    minimo = 3
    maximo = max(confiabilidades.keys()) if confiabilidades else 30

    return {
        "minimo": minimo,
        "recomendado": recomendado,
        "maximo_tecnico": maximo,
        "confiabilidades_estimadas": confiabilidades,
        "comentario": (
            "O sistema informa volumes e confiabilidades. "
            "A decisão final de quantas previsões gerar é do usuário."
        ),
    }


def v16_registrar_volume_e_confiabilidade():
    estado = st.session_state.get("estado_alvo_v16")
    expectativa = st.session_state.get("expectativa_v16")

    confiabs = v16_estimativa_confiabilidade_por_volume(
        estado_alvo=estado,
        expectativa=expectativa,
        base_confiabilidade=(expectativa or {}).get("score_previsibilidade"),
    )

    volume_op = v16_calcular_volume_operacional(
        estado_alvo=estado,
        expectativa=expectativa,
        confiabilidades=confiabs,
    )

    st.session_state["volume_operacional_v16"] = volume_op
    return volume_op



# ============================================================
# PARTE 1/8 — FIM
# ============================================================
# ============================================================
# PARTE 2/8 — INÍCIO
# ============================================================

# ============================================================
# >>> FUNÇÃO AUXILIAR — AJUSTE DE AMBIENTE PARA MODO 6
# (UNIVERSAL — respeita o fenômeno detectado)
# ============================================================

def ajustar_ambiente_modo6(
    *,
    df,
    k_star,
    nr_pct,
    divergencia_s6_mc,
    risco_composto,
    previsibilidade="baixa",
):
    """
    Ajusta volumes do Modo 6 sem bloquear execução.
    Sempre retorna configuração válida.

    BLOCO UNIVERSAL C:
    - Não assume n = 6
    - Lê PC_N_EFETIVO e PC_UNIVERSO_ATIVO se existirem
    - Não força alteração de comportamento
    """

    # --------------------------------------------------------
    # Leitura do fenômeno ativo (Blocos A + B + C)
    # --------------------------------------------------------
    pc_n_efetivo = st.session_state.get("PC_N_EFETIVO")
    pc_universo = st.session_state.get("PC_UNIVERSO_ATIVO")

    # --------------------------------------------------------
    # Valores base (comportamento LEGADO preservado)
    # --------------------------------------------------------
    volume_min = 3
    volume_recomendado = 6
    volume_max = 80

    # --------------------------------------------------------
    # Ajuste simples por previsibilidade (V16)
    # --------------------------------------------------------
    if previsibilidade == "alta":
        volume_min = 6
        volume_recomendado = 12
        volume_max = 40
    elif previsibilidade == "baixa":
        volume_min = 3
        volume_recomendado = 6
        volume_max = 20

    # --------------------------------------------------------
    # Ajuste UNIVERSAL SUAVE (não forçador)
    # --------------------------------------------------------
    aviso_universal = ""

    if pc_n_efetivo is not None:
        aviso_universal += f" | Fenômeno n={pc_n_efetivo}"

        # Regra conservadora:
        # quanto maior n, menor o volume máximo recomendado
        if pc_n_efetivo > 6:
            volume_max = min(volume_max, 20)
            volume_recomendado = min(volume_recomendado, 6)
            aviso_universal += " (redução preventiva)"

        elif pc_n_efetivo < 6:
            # Fenômenos menores toleram leve expansão
            volume_max = min(volume_max, 40)
            aviso_universal += " (fenômeno compacto)"

    if pc_universo is not None:
        u_min, u_max = pc_universo
        aviso_universal += f" | Univ:{u_min}-{u_max}"

    # --------------------------------------------------------
    # Retorno PADRÃO (compatível com todo o app)
    # --------------------------------------------------------
    return {
        "volume_min": volume_min,
        "volume_recomendado": volume_recomendado,
        "volume_max": volume_max,
        "confiabilidade_estimada": 0.05,
        "aviso_curto": (
            f"Modo 6 ativo | Volumes: "
            f"{volume_min}/{volume_recomendado}/{volume_max}"
            f"{aviso_universal}"
        ),
    }

# ============================================================
# <<< FIM — FUNÇÃO AUXILIAR — AJUSTE DE AMBIENTE PARA MODO 6
# ============================================================


# ============================================================
# GATILHO ECO — OBSERVADOR PASSIVO (V16 PREMIUM)
# NÃO decide | NÃO expande | NÃO altera volumes
# Apenas sinaliza prontidão para ECO
# (UNIVERSAL — consciente do fenômeno)
# ============================================================

def avaliar_gatilho_eco(
    k_star_atual: float,
    nr_pct: float,
    divergencia_s6_mc: float,
):
    """
    Avalia se o ambiente está tecnicamente pronto para ECO.
    BLOCO UNIVERSAL C:
    - Leitura do fenômeno ativo
    - Nenhuma decisão automática
    """

    pc_n_efetivo = st.session_state.get("PC_N_EFETIVO")
    pc_universo = st.session_state.get("PC_UNIVERSO_ATIVO")

    pronto_eco = False
    motivos = []

    # --------------------------------------------------------
    # Critérios técnicos (LEGADOS)
    # --------------------------------------------------------
    if k_star_atual < 0.15:
        motivos.append("k* favorável")

    if nr_pct < 0.30:
        motivos.append("ruído controlado")

    if divergencia_s6_mc < 5.0:
        motivos.append("baixa divergência S6 vs MC")

    if len(motivos) >= 2:
        pronto_eco = True

    # --------------------------------------------------------
    # Informação universal (observacional)
    # --------------------------------------------------------
    info_universal = ""

    if pc_n_efetivo is not None:
        info_universal += f" | Fenômeno n={pc_n_efetivo}"

    if pc_universo is not None:
        u_min, u_max = pc_universo
        info_universal += f" | Univ:{u_min}-{u_max}"

    return {
        "pronto_eco": pronto_eco,
        "motivos": motivos,
        "mensagem": (
            "ECO tecnicamente possível"
            if pronto_eco
            else "ECO ainda não recomendado"
        )
        + info_universal,
    }

# ============================================================
# <<< FIM — GATILHO ECO — OBSERVADOR PASSIVO (V16 PREMIUM)
# ============================================================




# ============================================================
# Painel 1 — 📁 Carregar Histórico (Arquivo)
# ============================================================
if painel == "📁 Carregar Histórico (Arquivo)":

    st.markdown("## 📁 Carregar Histórico — V15.7 MAX")

    st.markdown(
        "Envie um arquivo de histórico em formato **FLEX ULTRA**.\n\n"
        "📌 Regra universal: o **último valor da linha é sempre k**, "
        "independente da quantidade de passageiros."
    )

    arquivo = st.file_uploader(
        "Envie o arquivo de histórico",
        type=["txt", "csv"],
    )

    if arquivo is None:
        exibir_bloco_mensagem(
            "Aguardando arquivo de histórico",
            "Envie seu arquivo para iniciar o processamento do PredictCars.",
            tipo="info",
        )
        st.stop()

    try:
        conteudo = arquivo.getvalue().decode("utf-8")
        linhas = conteudo.strip().split("\n")

        if not limitar_operacao(
            len(linhas),
            limite_series=LIMITE_SERIES_REPLAY_ULTRA,
            contexto="Carregar Histórico (Arquivo)",
            painel="📁 Carregar Histórico (Arquivo)",
        ):
            st.stop()

        df = carregar_historico_universal(linhas)

    except Exception as erro:
        exibir_bloco_mensagem(
            "Erro ao processar histórico",
            f"Detalhes técnicos: {erro}",
            tipo="error",
        )
        st.stop()

    st.session_state["historico_df"] = df

    metricas = calcular_metricas_basicas_historico(df)
    exibir_resumo_inicial_historico(metricas)

    # ============================================================
    # 🌐 BLOCO UNIVERSAL A — DETECTOR DO FENÔMENO
    # ============================================================

    st.markdown("### 🌐 Perfil do Fenômeno (detecção automática)")
    st.caption(
        "Detecção automática do formato real do fenômeno.\n"
        "✔ Última coluna = k\n"
        "✔ Quantidade de passageiros livre\n"
        "✔ Universo variável\n"
        "❌ Não há decisão automática"
    )

    import hashlib

    colunas = list(df.columns)
    col_id = colunas[0]
    col_k = colunas[-1]
    col_passageiros = colunas[1:-1]

    passageiros_por_linha = []
    todos_passageiros = []

    for _, row in df.iterrows():
        valores = [int(v) for v in row[col_passageiros] if pd.notna(v)]
        passageiros_por_linha.append(len(valores))
        todos_passageiros.extend(valores)

    n_set = sorted(set(passageiros_por_linha))
    mix_n_detectado = len(n_set) > 1
    n_passageiros = n_set[0] if not mix_n_detectado else None

    universo_min = int(min(todos_passageiros)) if todos_passageiros else None
    universo_max = int(max(todos_passageiros)) if todos_passageiros else None
    universo_set = sorted(set(todos_passageiros))

    hash_base = f"{n_set}-{universo_min}-{universo_max}"
    fenomeno_id = hashlib.md5(hash_base.encode()).hexdigest()[:8]

    st.session_state["pc_n_passageiros"] = n_passageiros
    st.session_state["pc_n_set_detectado"] = n_set
    st.session_state["pc_mix_n_detectado"] = mix_n_detectado
    st.session_state["pc_universo_min"] = universo_min
    st.session_state["pc_universo_max"] = universo_max
    st.session_state["pc_universo_set"] = universo_set
    st.session_state["pc_fenomeno_id"] = fenomeno_id

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📐 Estrutura**")
        st.write(f"Passageiros por série (n): **{n_set}**")
        if mix_n_detectado:
            st.warning("Mistura de n detectada no mesmo histórico.")
        st.write(f"Coluna ID: `{col_id}`")
        st.write(f"Coluna k: `{col_k}`")

    with col2:
        st.markdown("**🌍 Universo observado**")
        st.write(f"Mínimo: **{universo_min}**")
        st.write(f"Máximo: **{universo_max}**")
        st.write(f"Total distintos: **{len(universo_set)}**")

    st.markdown("**🆔 Fenômeno ID (auditoria)**")
    st.code(fenomeno_id)

    # ============================================================
    # 🌐 BLOCO UNIVERSAL B — PARAMETRIZAÇÃO DO FENÔMENO
    # ============================================================

    st.markdown("### 🌐 Parâmetros Ativos do Fenômeno")
    st.caption(
        "Parâmetros universais derivados do histórico.\n"
        "✔ Não executa\n"
        "✔ Não interfere\n"
        "✔ Não altera módulos existentes"
    )

    if not mix_n_detectado:
        pc_n_alvo = n_passageiros
        pc_n_status = "fixo"
    else:
        pc_n_alvo = None
        pc_n_status = "misto"

    st.session_state["pc_n_alvo"] = pc_n_alvo
    st.session_state["pc_range_min"] = universo_min
    st.session_state["pc_range_max"] = universo_max

    if pc_n_alvo:
        st.session_state["pc_regua_extrema"] = f"{pc_n_alvo} ou nada"
        st.session_state["pc_regua_mvp2"] = f"2–{pc_n_alvo}"
    else:
        st.session_state["pc_regua_extrema"] = "indefinida"
        st.session_state["pc_regua_mvp2"] = "indefinida"

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**🎯 n alvo**")
        st.write(f"Status: **{pc_n_status}**")
        st.write(f"n alvo: **{pc_n_alvo if pc_n_alvo else 'MISTO'}**")

    with col4:
        st.markdown("**📏 Universo ativo**")
        st.write(f"{universo_min} – {universo_max}")
        st.write("Origem: histórico observado")

    if mix_n_detectado:
        st.warning(
            "⚠️ Histórico contém mistura de quantidades de passageiros.\n\n"
            "Recomenda-se separar fenômenos antes de previsões."
        )

    if pc_n_alvo and pc_n_alvo != 6:
        st.info(
            f"ℹ️ Fenômeno com n = {pc_n_alvo} detectado.\n"
            "Módulos legados ainda podem assumir n=6.\n"
            "➡️ Próximo passo: BLOCO UNIVERSAL C."
        )

    st.success("Perfil e parâmetros do fenômeno definidos.")

    st.success("Histórico carregado com sucesso!")
    st.dataframe(df.head(20))


# ============================================================
# Painel 1B — 📄 Carregar Histórico (Colar)
# ============================================================
if "Carregar Histórico (Colar)" in painel:

    st.markdown("## 📄 Carregar Histórico — Copiar e Colar (V15.7 MAX)")

    texto = st.text_area(
        "Cole aqui o histórico completo",
        height=320,
        key="pc_colar_texto_simples",
    )

    clicked = st.button(
        "📥 Processar Histórico (Copiar e Colar)",
        key="pc_colar_btn_simples",
    )

    if clicked:

        st.write("PROCESSANDO HISTÓRICO...")

        if not texto.strip():
            st.error("Histórico vazio")
            st.stop()

        linhas = texto.strip().split("\n")

        df = carregar_historico_universal(linhas)

        st.session_state["historico_df"] = df

        st.success(f"Histórico carregado com sucesso: {len(df)} séries")





# ============================================================
# BLOCO — OBSERVADOR HISTÓRICO DE EVENTOS k (V16)
# FASE 1 — OBSERVAÇÃO PURA | SEM IMPACTO OPERACIONAL
# ============================================================






# ============================================================
# PAINEL — 📊 V16 PREMIUM — ERRO POR REGIME (RETROSPECTIVO)
# (INSTRUMENTAÇÃO: mede continuidade do erro por janelas)
# ============================================================

# ============================================================
# PAINEL — 🧠 Diagnóstico ECO & Estado (V16)
# Observacional | NÃO decide | NÃO altera motores
# ============================================================

elif painel == "🧠 Diagnóstico ECO & Estado (V16)":

    st.markdown("## 🧠 Diagnóstico ECO & Estado — V16")
    st.caption("Leitura mastigada do ambiente e do alvo. Observacional.")

    diag = st.session_state.get("diagnostico_eco_estado_v16")

    if not diag:
        st.info("Diagnóstico ainda não disponível. Carregue um histórico.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌊 ECO")
        eco_forca = diag.get("eco_forca") or "fraco"
        st.write(f"Força: **{eco_forca}**")
        st.write(f"Persistência: **{diag.get('eco_persistencia')}**")
        st.write(f"Acionabilidade: **{diag.get('eco_acionabilidade')}**")

        motivos = diag.get("motivos_eco", [])
        if motivos:
            st.caption("Motivos: " + ", ".join(motivos))

        contradicoes = diag.get("contradicoes", [])
        if contradicoes:
            st.warning("⚠️ Contradições: " + "; ".join(contradicoes))

    with col2:
        st.markdown("### 🐟 Estado do Alvo")
        st.write(f"Estado: **{diag.get('estado')}**")
        st.write(
            "Confiabilidade: "
            f"**{'alta' if diag.get('estado_confiavel') else 'baixa / transição'}**"
        )

    st.markdown("### 🧠 Leitura Geral")
    st.success(diag.get("leitura_geral", "—"))

elif painel == "📊 V16 Premium — Erro por Regime (Retrospectivo)":

    st.subheader("📊 V16 Premium — Erro por Regime (Retrospectivo)")
    st.caption(
        "Instrumentação retrospectiva: janelas móveis → regime (ECO/PRE/RUIM) "
        "por dispersão da janela e erro da PRÓXIMA série como proxy de 'erro contido'. "
        "Não altera motor. Não escolhe passageiros."
    )

    # ============================================================
    # Localização ROBUSTA do histórico (padrão oficial V16)
    # ============================================================
    _, historico_df = v16_identificar_df_base()

    if historico_df is None or historico_df.empty:
        st.warning(
            "Histórico não encontrado no estado atual do app.\n\n"
            "👉 Recarregue o histórico e volte diretamente a este painel."
        )
        st.stop()

    if len(historico_df) < 100:
        st.warning(
            f"Histórico muito curto para análise retrospectiva.\n\n"
            f"Séries detectadas: {len(historico_df)}"
        )
        st.stop()

    # 🔒 Anti-zumbi automático (painel leve, invisível)
    janela = 60
    step = 1

    with st.spinner("Calculando análise retrospectiva por janelas (V16 Premium)..."):
        out = pc16_calcular_continuidade_por_janelas(
            historico_df=historico_df,
            janela=janela,
            step=step,
            usar_quantis=True
        )

    if not out.get("ok", False):
        st.error(f"Falha na análise: {out.get('motivo','Erro desconhecido')}")
        st.stop()

    resumo_geral = out.get("resumo_geral", {})
    resumo = out.get("resumo", {})
    df = out.get("df", pd.DataFrame())

    # ============================================================
    # RESULTADO OBJETIVO
    # ============================================================
    st.markdown("### ✅ Resultado objetivo — Continuidade do erro")

    diff = resumo_geral.get("diff_ruim_menos_eco_no_erro", None)
    if diff is None:
        st.info(
            "Ainda não há base suficiente para comparar ECO vs RUIM.\n\n"
            "Isso ocorre quando algum regime tem poucas janelas."
        )
    else:
        st.write(
            f"**Diferença RUIM − ECO no erro médio (erro_prox):** "
            f"`{diff:.6f}`\n\n"
            "➡️ Valores positivos indicam erro menor em ECO."
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de janelas", str(resumo_geral.get("n_total_janelas", "—")))
    col2.metric("Janela (W)", str(resumo_geral.get("janela", "—")))
    col3.metric("q1 dx (ECO ≤)", f"{resumo_geral.get('q1_dx', 0):.6f}")
    col4.metric("q2 dx (PRE ≤)", f"{resumo_geral.get('q2_dx', 0):.6f}")

    # ============================================================
    # TABELA POR REGIME
    # ============================================================
    st.markdown("### 🧭 Tabela por Regime (ECO / PRE / RUIM)")

    linhas = []
    for reg in ["ECO", "PRE", "RUIM"]:
        r = resumo.get(reg, {"n": 0})
        linhas.append({
            "Regime": reg,
            "n_janelas": r.get("n", 0),
            "dx_janela_medio": r.get("dx_janela_medio"),
            "erro_prox_medio": r.get("erro_prox_medio"),
            "erro_prox_mediana": r.get("erro_prox_mediana"),
        })

    df_reg = pd.DataFrame(linhas)
    st.dataframe(df_reg, use_container_width=True)

    # ============================================================
    # AUDITORIA LEVE
    # ============================================================
    st.markdown("### 🔎 Amostra das janelas (auditoria leve)")
    st.caption(
        "Exibe as primeiras linhas apenas para validação conceitual. "
        "`t` é um índice interno (0-based)."
    )
    st.dataframe(df.head(50), use_container_width=True)

    # ============================================================
    # LEITURA OPERACIONAL
    # ============================================================
    st.markdown("### 🧠 Leitura operacional (objetiva)")
    st.write(
        "- Se **ECO** apresentar **erro_prox_medio** consistentemente menor que **RUIM**, "
        "isso sustenta matematicamente que, em estados ECO, **o erro tende a permanecer contido**.\n"
        "- Este painel **não escolhe passageiros**.\n"
        "- Ele **autoriza** (ou não) a fase seguinte: **concentração para buscar 6**, "
        "sem alterar motor ou fluxo."
    )




# ============================================================
# PAINEL V16 — 🎯 Compressão do Alvo (OBSERVACIONAL)
# Leitura pura | NÃO prevê | NÃO decide | NÃO altera motores
# ============================================================

if painel == "🎯 Compressão do Alvo (Observacional)":

    st.markdown("## 🎯 Compressão do Alvo — Leitura Observacional (V16)")
    st.caption(
        "Este painel mede **se o erro provável está comprimindo**.\n\n"
        "⚠️ Não prevê números, não sugere volume, não altera o fluxo."
    )

    # -----------------------------
    # Coleta de sinais já existentes
    # -----------------------------
    nr = st.session_state.get("nr_percent")
    div = st.session_state.get("div_s6_mc")
    k_star = st.session_state.get("sentinela_kstar")
    risco = (st.session_state.get("diagnostico_risco") or {}).get("indice_risco")

    df = st.session_state.get("historico_df")

    if df is None or nr is None or div is None or k_star is None or risco is None:
        exibir_bloco_mensagem(
            "Pré-requisitos ausentes",
            "Execute os painéis de Sentinela, Ruído, Divergência e Monitor de Risco.",
            tipo="warning",
        )
        st.stop()

    # -----------------------------
    # 1) Estabilidade do ruído
    # -----------------------------
    nr_ok = nr < 45.0

    # -----------------------------
    # 2) Convergência dos motores
    # -----------------------------
    div_ok = div < 5.0

    # -----------------------------
    # 3) Regime não-hostil
    # -----------------------------
    risco_ok = risco < 0.55

    # -----------------------------
    # 4) k como marcador NORMAL (não extremo)
    # -----------------------------
    k_ok = 0.10 <= k_star <= 0.35

    # -----------------------------
    # 5) Repetição estrutural (passageiros)
    # -----------------------------
    col_pass = [c for c in df.columns if c.startswith("p")]
    ultimos = df[col_pass].iloc[-10:].values

    repeticoes = []
    for i in range(len(ultimos) - 1):
        repeticoes.append(len(set(ultimos[i]) & set(ultimos[i + 1])))

    repeticao_media = float(np.mean(repeticoes)) if repeticoes else 0.0
    repeticao_ok = repeticao_media >= 2.5

    # -----------------------------
    # Consolidação OBSERVACIONAL
    # -----------------------------
    sinais = {
        "NR% estável": nr_ok,
        "Convergência S6 × MC": div_ok,
        "Risco controlado": risco_ok,
        "k em faixa normal": k_ok,
        "Repetição estrutural": repeticao_ok,
    }

    positivos = sum(1 for v in sinais.values() if v)

    # -----------------------------
    # Exibição
    # -----------------------------
    st.markdown("### 📊 Sinais de Compressão do Erro")

    for nome, ok in sinais.items():
        st.markdown(
            f"- {'🟢' if ok else '🔴'} **{nome}**"
        )

    st.markdown("### 🧠 Leitura Consolidada")

    if positivos >= 4:
        leitura = (
            "🟢 **Alta compressão do erro provável**.\n\n"
            "O alvo está mais bem definido do que o normal.\n"
            "Se houver PRÉ-ECO / ECO, a convicção operacional aumenta."
        )
    elif positivos == 3:
        leitura = (
            "🟡 **Compressão parcial**.\n\n"
            "Há foco emergente, mas ainda com dispersão residual."
        )
    else:
        leitura = (
            "🔴 **Sem compressão clara**.\n\n"
            "Erro ainda espalhado. Operar com cautela."
        )

    exibir_bloco_mensagem(
        "Compressão do Alvo — Diagnóstico",
        leitura,
        tipo="info",
    )

    st.caption(
        f"Sinais positivos: {positivos}/5 | "
        "Este painel **não autoriza nem bloqueia** nenhuma ação."
    )

# ============================================================
# FIM — PAINEL V16 — COMPRESSÃO DO ALVO (OBSERVACIONAL)
# ============================================================


# ============================================================
# BLOCO — OBSERVADOR HISTÓRICO DE EVENTOS k (V16)
# FASE 2 — REPLAY HISTÓRICO OBSERVACIONAL (MEMÓRIA REAL)
# NÃO decide | NÃO prevê | NÃO altera motores | NÃO altera volumes
# ============================================================

def v16_replay_historico_observacional(
    *,
    df,
    matriz_norm,
    janela_max=800,
):
    """
    Replay histórico OBSERVACIONAL.
    Executa leitura silenciosa série-a-série para preencher memória
    e eliminar campos None no Observador Histórico.

    - Usa somente dados já calculados
    - NÃO reexecuta motores pesados
    - NÃO interfere no fluxo operacional
    """

    if df is None or matriz_norm is None:
        return []

    n_total = len(df)
    inicio = max(0, n_total - int(janela_max))

    registros = []

    col_pass = [c for c in df.columns if c.startswith("p")]

    for idx in range(inicio, n_total):

        # --- NR% local (réplica leve) ---
        try:
            m = matriz_norm[: idx + 1]
            variancias = np.var(m, axis=1)
            ruido_A = float(np.mean(variancias))
            saltos = [
                np.linalg.norm(m[i] - m[i - 1])
                for i in range(1, len(m))
            ]
            ruido_B = float(np.mean(saltos)) if saltos else 0.0
            nr_pct = float(
                (0.55 * min(1.0, ruido_A / 0.08) +
                 0.45 * min(1.0, ruido_B / 1.20)) * 100.0
            )
        except Exception:
            nr_pct = None

        # --- Divergência local S6 vs MC (proxy leve) ---
        try:
            base = m[-1]
            candidatos = m[-10:] if len(m) >= 10 else m
            divergencia = float(
                np.linalg.norm(np.mean(candidatos, axis=0) - base)
            )
        except Exception:
            divergencia = None

        # --- Velocidade / estado do alvo (heurística coerente) ---
        try:
            vel = float(
                (nr_pct / 100.0 if nr_pct is not None else 0.5) +
                (divergencia / 15.0 if divergencia is not None else 0.5)
            ) / 2.0
        except Exception:
            vel = None

        if vel is None:
            estado = None
        elif vel < 0.30:
            estado = "parado"
        elif vel < 0.55:
            estado = "movimento_lento"
        elif vel < 0.80:
            estado = "movimento_rapido"
        else:
            estado = "disparado"

        # --- k histórico ---
        try:
            k_val = int(df.iloc[idx].get("k", 0))
        except Exception:
            k_val = 0

        registros.append({
            "serie_id": idx,
            "k_valor": k_val,
            "estado_alvo": estado,
            "nr_percent": nr_pct,
            "div_s6_mc": divergencia,
        })

    return registros


# ============================================================
# EXECUÇÃO AUTOMÁTICA — REPLAY OBSERVACIONAL (SE HISTÓRICO EXISTIR)
# ============================================================

if (
    "historico_df" in st.session_state
    and "pipeline_matriz_norm" in st.session_state
):
    registros_obs = v16_replay_historico_observacional(
        df=st.session_state.get("historico_df"),
        matriz_norm=st.session_state.get("pipeline_matriz_norm"),
        janela_max=800,  # DECISÃO DO COMANDO
    )

    st.session_state["observador_historico_v16"] = registros_obs

# ============================================================
# FIM — BLOCO OBSERVADOR HISTÓRICO (V16) — FASE 2
# ============================================================



# ============================================================
# BLOCO — OBSERVAÇÃO HISTÓRICA OFFLINE (V16)
# OPÇÃO B MÍNIMA | LEITURA PURA | NÃO DECIDE | NÃO OPERA
# ============================================================

def _pc_distancia_carros_offline(a, b):
    """
    Distância simples entre dois carros (listas de 6):
    quantos passageiros mudaram (0..6).
    Observacional, robusto e defensivo.
    """
    try:
        sa = set(int(x) for x in a)
        sb = set(int(x) for x in b)
        inter = len(sa & sb)
        return max(0, 6 - inter)
    except Exception:
        return None


def _pc_estado_alvo_proxy_offline(dist):
    """
    Mapeia distância (0..6) em estado do alvo (proxy observacional).
    NÃO é o estado V16 online. Uso EXCLUSIVO histórico.
    """
    if dist is None:
        return None
    if dist <= 1:
        return "parado"
    if dist <= 3:
        return "movimento_lento"
    if dist <= 5:
        return "movimento"
    return "movimento_brusco"


def _pc_extrair_carro_offline(row):
    """
    Extrai os 6 passageiros de uma linha do histórico.
    Compatível com p1..p6 ou colunas numéricas genéricas.
    """
    cols_p = ["p1", "p2", "p3", "p4", "p5", "p6"]
    if all(c in row.index for c in cols_p):
        return [row[c] for c in cols_p]

    candidatos = []
    for c in row.index:
        if str(c).lower() == "k":
            continue
        try:
            candidatos.append(int(row[c]))
        except Exception:
            continue

    return candidatos[:6] if len(candidatos) >= 6 else None


def construir_contexto_historico_offline_v16(df):
    """
    Constrói CONTEXTO HISTÓRICO OFFLINE mínimo:
    - estado_alvo_proxy_historico
    - delta_k_historico
    - eventos_k_historico (enriquecido)
    NÃO interfere em motores, painéis ou decisões.
    """

    if df is None or df.empty:
        return

    estado_proxy_hist = {}
    delta_k_hist = {}
    eventos_k = []

    carro_prev = None
    ultima_pos_k = None

    for pos, (idx, row) in enumerate(df.iterrows()):
        carro_atual = _pc_extrair_carro_offline(row)

        dist = (
            _pc_distancia_carros_offline(carro_prev, carro_atual)
            if carro_prev is not None and carro_atual is not None
            else None
        )

        estado_proxy = _pc_estado_alvo_proxy_offline(dist)
        estado_proxy_hist[idx] = estado_proxy

        # Evento k (observacional)
        try:
            k_val = int(row.get("k", 0))
        except Exception:
            k_val = 0

        if k_val > 0:
            delta = None if ultima_pos_k is None else int(pos - ultima_pos_k)
            delta_k_hist[idx] = delta

            eventos_k.append({
                "serie_id": idx,
                "pos": int(pos),
                "k_valor": int(k_val),
                "delta_series": delta,
                "estado_alvo_proxy": estado_proxy,
            })

            ultima_pos_k = pos

        carro_prev = carro_atual

    # Persistência PASSIVA (session_state)
    st.session_state["estado_alvo_proxy_historico"] = estado_proxy_hist
    st.session_state["delta_k_historico"] = delta_k_hist
    st.session_state["eventos_k_historico"] = eventos_k


# ============================================================
# EXECUÇÃO AUTOMÁTICA OFFLINE (SE HISTÓRICO EXISTIR)
# NÃO BLOQUEIA | NÃO DECIDE | NÃO OPERA
# ============================================================

if "historico_df" in st.session_state:
    try:
        construir_contexto_historico_offline_v16(
            st.session_state.get("historico_df")
        )
    except Exception:
        pass

# ============================================================
# FIM — OBSERVAÇÃO HISTÓRICA OFFLINE (V16) — OPÇÃO B MÍNIMA
# ============================================================

def extrair_eventos_k_historico(
    df,
    estados_alvo=None,
    k_star_series=None,
    nr_percent_series=None,
    divergencia_series=None,
    pre_eco_series=None,
    eco_series=None,
):
    """
    Extrai eventos k do histórico com contexto.
    NÃO decide, NÃO filtra operacionalmente, NÃO altera motores.
    Retorna lista de dicionários observacionais.
    """

    if df is None or df.empty:
        return []

    eventos = []
    ultima_serie_k = None

    for idx, row in df.iterrows():
        # Espera-se que o histórico tenha coluna 'k'
        k_valor = row.get("k", 0)

        if k_valor and k_valor > 0:
            # Delta desde último k
            if ultima_serie_k is None:
                delta = None
            else:
                delta = idx - ultima_serie_k

            evento = {
                "serie_id": idx,
                "k_valor": int(k_valor),
                "delta_series": delta,
                "estado_alvo": (
                    estados_alvo.get(idx)
                    if isinstance(estados_alvo, dict)
                    else None
                ),
                "k_star": (
                    k_star_series.get(idx)
                    if isinstance(k_star_series, dict)
                    else None
                ),
                "nr_percent": (
                    nr_percent_series.get(idx)
                    if isinstance(nr_percent_series, dict)
                    else None
                ),
                "div_s6_mc": (
                    divergencia_series.get(idx)
                    if isinstance(divergencia_series, dict)
                    else None
                ),
                "pre_eco": (
                    pre_eco_series.get(idx)
                    if isinstance(pre_eco_series, dict)
                    else False
                ),
                "eco": (
                    eco_series.get(idx)
                    if isinstance(eco_series, dict)
                    else False
                ),
            }

            eventos.append(evento)
            ultima_serie_k = idx

    return eventos


# ============================================================
# EXECUÇÃO AUTOMÁTICA (APENAS SE HISTÓRICO EXISTIR)
# ============================================================

if "historico_df" in st.session_state:
    df_hist = st.session_state.get("historico_df")

    eventos_k = extrair_eventos_k_historico(
        df=df_hist,
        estados_alvo=st.session_state.get("estado_alvo_historico"),
        k_star_series=st.session_state.get("kstar_historico"),
        nr_percent_series=st.session_state.get("nr_historico"),
        divergencia_series=st.session_state.get("div_s6_mc_historico"),
        pre_eco_series=st.session_state.get("pre_eco_historico"),
        eco_series=st.session_state.get("eco_historico"),
    )

    st.session_state["eventos_k_historico"] = eventos_k

# ============================================================
# BLOCO — FIM OBSERVADOR HISTÓRICO DE EVENTOS k
# ============================================================

# ============================================================
# Painel — 📊 Observador Histórico de Eventos k (V16)
# FASE 1 — OBSERVAÇÃO PURA | NÃO DECIDE | NÃO OPERA
# ============================================================

if painel == "📊 Observador k — Histórico":

    st.markdown("## 📊 Observador Histórico de Eventos k")
    st.caption(
        "Leitura puramente observacional. "
        "Este painel **não influencia** previsões, volumes ou decisões."
    )

    eventos = st.session_state.get("eventos_k_historico")

    if not eventos:
        exibir_bloco_mensagem(
            "Nenhum evento k disponível",
            "Carregue um histórico válido para observar eventos k.",
            tipo="info",
        )
        st.stop()

    df_k = pd.DataFrame(eventos)

    st.markdown("### 🔍 Tabela de Eventos k (Histórico)")
    st.dataframe(
        df_k,
        use_container_width=True,
        height=420,
    )

    # Métricas simples (somente leitura)
    st.markdown("### 📈 Métricas Observacionais Básicas")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total de eventos k",
            len(df_k),
        )

    with col2:
        delta_vals = df_k["delta_series"].dropna()
        st.metric(
            "Δ médio entre ks",
            round(delta_vals.mean(), 2) if not delta_vals.empty else "—",
        )

    with col3:
        st.metric(
            "Δ mínimo observado",
            int(delta_vals.min()) if not delta_vals.empty else "—",
        )

    st.info(
        "Interpretação é humana. "
        "Nenhum uso operacional é feito a partir destes dados."
    )

# ============================================================
# FIM — Painel Observador Histórico de Eventos k
# ============================================================

# ============================================================
# Painel — 🎯 Compressão do Alvo — Observacional (V16)
# LEITURA PURA | NÃO DECIDE | NÃO ALTERA MOTORES
# Objetivo: medir se o alvo está REALMENTE "na mira"
# ============================================================

if painel == "🎯 Compressão do Alvo — Observacional (V16)":

    st.markdown("## 🎯 Compressão do Alvo — Observacional (V16)")
    st.caption(
        "Painel **observacional puro**.\n\n"
        "Ele NÃO gera previsões, NÃO altera volumes e NÃO interfere no fluxo.\n"
        "Serve para responder: **o alvo está realmente comprimido / na mira?**"
    )

    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")

    if df is None or matriz_norm is None:
        exibir_bloco_mensagem(
            "Pipeline incompleto",
            "Execute **Carregar Histórico** e **Pipeline V14-FLEX ULTRA** antes.",
            tipo="warning",
        )
        st.stop()

    # ------------------------------------------------------------
    # Parâmetros fixos (observacionais)
    # ------------------------------------------------------------
    JANELA_ANALISE = 120   # últimas séries
    JANELA_LOCAL = 8       # microjanela para dispersão
    LIMIAR_COMPRESSAO = 0.65  # heurístico (não decisório)

    n = len(matriz_norm)
    if n < JANELA_ANALISE + JANELA_LOCAL:
        exibir_bloco_mensagem(
            "Histórico insuficiente",
            "São necessárias mais séries para analisar compressão do alvo.",
            tipo="warning",
        )
        st.stop()

    # ------------------------------------------------------------
    # Cálculo da compressão
    # ------------------------------------------------------------
    dispersoes = []
    centroides = []

    for i in range(n - JANELA_ANALISE, n):
        janela = matriz_norm[max(0, i - JANELA_LOCAL): i + 1]
        centro = np.mean(janela, axis=0)
        centroides.append(centro)

        dist = np.mean(
            [np.linalg.norm(linha - centro) for linha in janela]
        )
        dispersoes.append(dist)

    dispersao_media = float(np.mean(dispersoes))
    dispersao_std = float(np.std(dispersoes))

    # Compressão relativa (quanto menor a dispersão, maior a compressão)
    compressao_score = 1.0 - min(1.0, dispersao_media / (dispersao_media + dispersao_std + 1e-6))
    compressao_score = float(round(compressao_score, 4))

    # ------------------------------------------------------------
    # Interpretação QUALITATIVA (não decisória)
    # ------------------------------------------------------------
    if compressao_score >= 0.75:
        leitura = "🟢 Alvo fortemente comprimido"
        comentario = (
            "O histórico recente mostra **alta repetição estrutural**.\n"
            "O sistema está operando em zona de foco.\n"
            "Quando combinado com PRÉ-ECO / ECO, **permite acelerar**."
        )
    elif compressao_score >= LIMIAR_COMPRESSAO:
        leitura = "🟡 Compressão moderada"
        comentario = (
            "Existe coerência estrutural, mas ainda com respiração.\n"
            "Bom para operação equilibrada."
        )
    else:
        leitura = "🔴 Alvo disperso"
        comentario = (
            "Alta variabilidade estrutural.\n"
            "Mesmo que k apareça, **não indica alvo na mira**."
        )

    # ------------------------------------------------------------
    # Exibição
    # ------------------------------------------------------------
    st.markdown("### 📐 Métrica de Compressão do Alvo")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Score de Compressão", compressao_score)

    with col2:
        st.metric("Dispersão média", round(dispersao_media, 4))

    with col3:
        st.metric("Volatilidade da dispersão", round(dispersao_std, 4))

    exibir_bloco_mensagem(
        "Leitura Observacional",
        f"**{leitura}**\n\n{comentario}",
        tipo="info",
    )

    st.info(
        "📌 Interpretação correta:\n"
        "- **Compressão NÃO prevê**\n"
        "- **Compressão NÃO decide**\n"
        "- Compressão **aumenta convicção** quando outros sinais já são positivos\n"
        "- Serve para **pisar mais fundo**, não para apertar o gatilho sozinho"
    )

# ============================================================
# FIM — Painel 🎯 Compressão do Alvo — Observacional (V16)
# ============================================================



# ============================================================
# Observação Histórica — Eventos k (V16)
# Leitura passiva do histórico. Não interfere em decisões.
# + CRUZAMENTO k × ESTADO DO ALVO (PROXY)
# ============================================================

def _pc_distancia_carros(a, b):
    """
    Distância simples entre dois carros (listas de 6):
    quantos passageiros mudaram (0..6).
    """
    try:
        sa = set([int(x) for x in a])
        sb = set([int(x) for x in b])
        inter = len(sa & sb)
        return max(0, 6 - inter)
    except Exception:
        return None


def _pc_estado_alvo_proxy(dist):
    """
    Mapeia distância (0..6) em estado do alvo (proxy observacional).
    """
    if dist is None:
        return None
    if dist <= 1:
        return "parado"
    if dist <= 3:
        return "movimento_lento"
    if dist <= 5:
        return "movimento"
    return "movimento_brusco"


def _pc_extrair_carro_row(row):
    """
    Extrai os 6 passageiros da linha do df.
    Tentativa 1: colunas numéricas (6 colunas)
    Tentativa 2: colunas p1..p6 (se existir)
    """
    # Caso já tenha colunas p1..p6
    cols_p = ["p1", "p2", "p3", "p4", "p5", "p6"]
    if all(c in row.index for c in cols_p):
        return [row[c] for c in cols_p]

    # Caso seja DF com colunas misturadas: pega primeiros 6 inteiros que não sejam 'k'
    candidatos = []
    for c in row.index:
        if str(c).lower() == "k":
            continue
        try:
            v = int(row[c])
            candidatos.append(v)
        except Exception:
            continue

    if len(candidatos) >= 6:
        return candidatos[:6]

    return None


def extrair_eventos_k_historico_com_proxy(df):
    """
    Eventos k + delta + estado do alvo (proxy) calculado do próprio histórico.
    NÃO depende de estado_alvo_historico/kstar_historico/etc.
    """
    if df is None or df.empty:
        return [], {}

    eventos = []
    ultima_pos_k = None

    # Para estatística
    cont_estados = {"parado": 0, "movimento_lento": 0, "movimento": 0, "movimento_brusco": 0, "None": 0}

    # Vamos usar posição sequencial (0..n-1) para delta
    rows = list(df.iterrows())

    carro_prev = None

    for pos, (idx, row) in enumerate(rows):
        k_val = row.get("k", 0)
        carro_atual = _pc_extrair_carro_row(row)

        dist = _pc_distancia_carros(carro_prev, carro_atual) if (carro_prev is not None and carro_atual is not None) else None
        estado = _pc_estado_alvo_proxy(dist)

        # Contagem estados (para todas as séries, não só eventos k)
        if estado is None:
            cont_estados["None"] += 1
        else:
            cont_estados[estado] += 1

        # Evento k
        try:
            k_int = int(k_val) if k_val is not None else 0
        except Exception:
            k_int = 0

        if k_int > 0:
            delta = None if ultima_pos_k is None else int(pos - ultima_pos_k)

            eventos.append({
                "serie_id": idx,
                "pos": int(pos),
                "k_valor": int(k_int),
                "delta_series": delta,
                "distancia_prev": dist,
                "estado_alvo_proxy": estado,
            })

            ultima_pos_k = pos

        carro_prev = carro_atual

    return eventos, cont_estados


# ============================================================
# PAINEL (VISUALIZAÇÃO)
# ============================================================

if painel == "Observação Histórica — Eventos k":

    st.markdown("## Observação Histórica — Eventos k")
    st.caption("Leitura passiva do histórico. Não interfere em decisões.")

    df_hist = st.session_state.get("historico_df")

    if df_hist is None or df_hist.empty:
        exibir_bloco_mensagem(
            "Histórico ausente",
            "Carregue o histórico primeiro (Painel 1 / 1B).",
            tipo="warning",
        )
        st.stop()

    eventos_k, cont_estados = extrair_eventos_k_historico_com_proxy(df_hist)
    st.session_state["eventos_k_historico"] = eventos_k

    # ===========================
    # Resumo estatístico
    # ===========================
    total_eventos = len(eventos_k)

    deltas = [e["delta_series"] for e in eventos_k if isinstance(e.get("delta_series"), int)]
    delta_medio = round(sum(deltas) / max(1, len(deltas)), 2) if deltas else None
    max_k = max([e.get("k_valor", 0) for e in eventos_k], default=0)

    st.markdown("### Resumo Estatístico Simples")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total de eventos k", f"{total_eventos}")
    c2.metric("Δ médio entre ks", f"{delta_medio}" if delta_medio is not None else "—")
    c3.metric("Máx k observado", f"{max_k}")

    st.markdown("### Distribuição do Estado do Alvo (PROXY no histórico)")
    total_series = sum(cont_estados.values()) if isinstance(cont_estados, dict) else 0
    if total_series > 0:
        corpo = (
            f"- parado: **{cont_estados.get('parado', 0)}**\n"
            f"- movimento_lento: **{cont_estados.get('movimento_lento', 0)}**\n"
            f"- movimento: **{cont_estados.get('movimento', 0)}**\n"
            f"- movimento_brusco: **{cont_estados.get('movimento_brusco', 0)}**\n"
        )
        exibir_bloco_mensagem("Estado do alvo (proxy)", corpo, tipo="info")
    else:
        st.info("Não foi possível calcular distribuição de estado (proxy).")

    # ===========================
    # Tabela de eventos k
    # ===========================
    st.markdown("### 📋 Tabela de Eventos k (com estado proxy)")
    if total_eventos == 0:
        st.info("Nenhum evento k encontrado no histórico.")
        st.stop()

    mostrar = st.slider(
        "Quantos eventos k mostrar (mais recentes)?",
        min_value=20,
        max_value=min(300, total_eventos),
        value=min(80, total_eventos),
        step=10,
    )

    # Mostra os mais recentes
    df_evt = pd.DataFrame(eventos_k[-mostrar:])
    st.dataframe(df_evt, use_container_width=True)

    st.caption("Obs.: estado_alvo_proxy é calculado por mudança entre carros consecutivos (distância 0..6).")
    st.caption("k*/NR%/div/PRÉ-ECO/ECO ainda não estão historificados por série — isso é a próxima evolução (opcional).")

# ============================================================
# FIM — Observação Histórica — Eventos k (V16)
# ============================================================

        

# ============================================================
# Painel 2 — 🛰️ Sentinelas — k* (Ambiente de Risco)
# ============================================================

if painel == "🛰️ Sentinelas — k* (Ambiente de Risco)":

    st.markdown("## 🛰️ Sentinelas — k* (Ambiente de Risco) — V15.7 MAX")

    df = st.session_state.get("historico_df")

    if df is None:
        exibir_bloco_mensagem(
            "Histórico não carregado",
            "Vá primeiro ao painel **📁 Carregar Histórico**.",
            tipo="warning",
        )
    else:
        qtd_series = len(df)

        # Parâmetros do k*
        janela_curta = 12
        janela_media = 30
        janela_longa = 60

        # Anti-zumbi aplicado antes de cálculos longos
        if not limitar_operacao(
            qtd_series,
            limite_series=LIMITE_SERIES_REPLAY_ULTRA,
            contexto="Sentinela k*",
            painel="🛰️ Sentinelas — k*",
        ):
            st.stop()

        # -------------------------------------------
        # Cálculo do k* — versão V15.7 MAX / V16 Premium
        # -------------------------------------------
        try:
            k_vals = df["k"].astype(int).values

            def media_movel(vetor, janela):
                if len(vetor) < janela:
                    return np.mean(vetor)
                return np.mean(vetor[-janela:])

            k_curto = media_movel(k_vals, janela_curta)
            k_medio = media_movel(k_vals, janela_media)
            k_longo = media_movel(k_vals, janela_longa)

            # Fórmula nova do k* — ponderada
            k_star = (
                0.50 * k_curto
                + 0.35 * k_medio
                + 0.15 * k_longo
            )

        except Exception as erro:
            exibir_bloco_mensagem(
                "Erro no cálculo do k*",
                f"Ocorreu um erro interno: {erro}",
                tipo="error",
            )
            st.stop()

        # Guarda na sessão
        st.session_state["sentinela_kstar"] = k_star

        # Exibição amigável
        st.markdown(f"### 🌡️ k* calculado: **{k_star:.4f}**")

        # Diagnóstico de regime
        if k_star < 0.15:
            regime = "🟢 Ambiente Estável (Regime de Padrão)"
        elif k_star < 0.30:
            regime = "🟡 Pré-Ruptura (Atenção)"
        else:
            regime = "🔴 Ambiente de Ruptura (Alta Turbulência)"

        exibir_bloco_mensagem(
            "Diagnóstico do Ambiente",
            f"O regime identificado para o histórico atual é:\n\n{regime}",
            tipo="info",
        )

# ============================================================
# Painel X — 📊 Observação Histórica — Eventos k (V16)
# ============================================================

if painel == "📊 Observação Histórica — Eventos k":

    st.markdown("## 📊 Observação Histórica — Eventos k")
    st.caption("Leitura passiva do histórico. Não interfere em decisões.")

    eventos = st.session_state.get("eventos_k_historico", [])

    if not eventos:
        st.info("Nenhum evento k encontrado no histórico carregado.")
        st.stop()

    df_eventos = pd.DataFrame(eventos)

    st.markdown("### 📋 Tabela de Eventos k")
    st.dataframe(df_eventos, use_container_width=True)

    # Resumo rápido
    st.markdown("### 📈 Resumo Estatístico Simples")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total de eventos k", len(df_eventos))

    with col2:
        media_delta = (
            df_eventos["delta_series"].dropna().mean()
            if "delta_series" in df_eventos
            else None
        )
        st.metric(
            "Δ médio entre ks",
            f"{media_delta:.2f}" if media_delta else "—",
        )

    with col3:
        st.metric(
            "Máx k observado",
            df_eventos["k_valor"].max() if "k_valor" in df_eventos else "—",
        )

# ============================================================
# FIM — Painel X — Observação Histórica — Eventos k
# ============================================================


# ============================================================
# Painel 3 — 🛣️ Pipeline V14-FLEX ULTRA (Preparação)
# ============================================================
if painel == "🛣️ Pipeline V14-FLEX ULTRA":

    st.markdown("## 🛣️ Pipeline V14-FLEX ULTRA — V15.7 MAX")

    df = st.session_state.get("historico_df")

    if df is None:
        exibir_bloco_mensagem(
            "Histórico não carregado",
            "Vá ao painel **📁 Carregar Histórico** antes de continuar.",
            tipo="warning",
        )
        st.stop()

    qtd_series = len(df)

    # Proteção anti-zumbi do pipeline — mais duro que o k*
    if not limitar_operacao(
        qtd_series,
        limite_series=LIMITE_SERIES_REPLAY_ULTRA,
        contexto="Pipeline V14-FLEX ULTRA",
        painel="🛣️ Pipeline",
    ):
        st.stop()

    st.info("Iniciando processamento do Pipeline FLEX ULTRA...")

    col_pass = [c for c in df.columns if c.startswith("p")]
    matriz = df[col_pass].astype(float).values

    # ============================================================
    # Normalização
    # ============================================================
    try:
        minimo = matriz.min()
        maximo = matriz.max()
        amplitude = maximo - minimo if maximo != minimo else 1.0

        matriz_norm = (matriz - minimo) / amplitude

    except Exception as erro:
        exibir_bloco_mensagem(
            "Erro na normalização",
            f"Detalhes técnicos: {erro}",
            tipo="error",
        )
        st.stop()

    # ============================================================
    # Estatísticas da estrada (FLEX ULTRA)
    # ============================================================
    medias = np.mean(matriz_norm, axis=1)
    desvios = np.std(matriz_norm, axis=1)

    media_geral = float(np.mean(medias))
    desvio_geral = float(np.mean(desvios))

    # Classificação simples de regime da estrada
    if media_geral < 0.35:
        estrada = "🟦 Estrada Fria (Baixa energia)"
    elif media_geral < 0.65:
        estrada = "🟩 Estrada Neutra / Estável"
    else:
        estrada = "🟥 Estrada Quente (Alta volatilidade)"

    # ============================================================
    # Clusterização leve (DX — motor original FLEX ULTRA)
    # ============================================================
    try:
        from sklearn.cluster import KMeans

        n_clusters = 3
        modelo = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
        clusters = modelo.fit_predict(matriz_norm)

        centroides = modelo.cluster_centers_

    except Exception:
        clusters = np.zeros(len(matriz_norm))
        centroides = np.zeros((1, matriz_norm.shape[1]))

# ============================================================
# PARTE 2/8 — FIM
# ============================================================
# ============================================================
# PARTE 3/8 — INÍCIO
# ============================================================

    # ============================================================
    # Exibição final do pipeline
    # ============================================================
    st.markdown("### 📌 Diagnóstico do Pipeline FLEX ULTRA")

    corpo = (
        f"- Séries carregadas: **{qtd_series}**\n"
        f"- Passageiros por carro (n): **{len(col_pass)}**\n"
        f"- Energia média da estrada: **{media_geral:.4f}**\n"
        f"- Volatilidade média: **{desvio_geral:.4f}**\n"
        f"- Regime detectado: {estrada}\n"
        f"- Clusters formados: **{int(max(clusters)+1)}**"
    )

    exibir_bloco_mensagem(
        "Resumo do Pipeline FLEX ULTRA",
        corpo,
        tipo="info",
    )

    # ============================================================
    # Salvando na sessão para módulos seguintes (CANÔNICO)
    # ============================================================
    st.session_state["pipeline_col_pass"] = col_pass
    st.session_state["pipeline_clusters"] = clusters
    st.session_state["pipeline_centroides"] = centroides
    st.session_state["pipeline_matriz_norm"] = matriz_norm
    st.session_state["pipeline_estrada"] = estrada

    st.success("Pipeline FLEX ULTRA concluído com sucesso!")

# ============================================================
# PARTE 3/8 — FIM
# ============================================================


# ============================================================
# Painel 4 — 🔁 Replay LIGHT
# ============================================================
if painel == "🔁 Replay LIGHT":

    st.markdown("## 🔁 Replay LIGHT — V15.7 MAX")

    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")

    if df is None or matriz_norm is None:
        exibir_bloco_mensagem(
            "Pipeline não encontrado",
            "Execute primeiro o painel **🛣️ Pipeline V14-FLEX ULTRA**.",
            tipo="warning",
        )
        st.stop()

    qtd_series = len(df)

    # Anti-zumbi para replays leves
    if not limitar_operacao(
        qtd_series,
        limite_series=LIMITE_SERIES_REPLAY_ULTRA,
        contexto="Replay LIGHT",
        painel="🔁 Replay LIGHT",
    ):
        st.stop()

    st.info("Executando Replay LIGHT...")

    try:
        # DX leve = simples proximidade média entre séries vizinhas
        proximidades = []
        for i in range(1, len(matriz_norm)):
            dist = np.linalg.norm(matriz_norm[i] - matriz_norm[i - 1])
            proximidades.append(dist)

        media_proximidade = float(np.mean(proximidades))
        desvio_proximidade = float(np.std(proximidades))

    except Exception as erro:
        exibir_bloco_mensagem(
            "Erro no Replay LIGHT",
            f"Detalhes técnicos: {erro}",
            tipo="error",
        )
        st.stop()

    corpo = (
        f"- Séries analisadas: **{qtd_series}**\n"
        f"- Proximidade média (DX Light): **{media_proximidade:.4f}**\n"
        f"- Desvio de proximidade: **{desvio_proximidade:.4f}**\n"
        "\nValores mais altos indicam maior irregularidade."
    )

    exibir_bloco_mensagem(
        "Resumo do Replay LIGHT",
        corpo,
        tipo="info",
    )

    st.success("Replay LIGHT concluído!")

# ============================================================
# Painel 5 — 🔁 Replay ULTRA
# ============================================================
if painel == "🔁 Replay ULTRA":

    st.markdown("## 🔁 Replay ULTRA — V15.7 MAX")

    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")

    if df is None or matriz_norm is None:
        exibir_bloco_mensagem(
            "Pipeline não encontrado",
            "Execute primeiro o painel **🛣️ Pipeline V14-FLEX ULTRA**.",
            tipo="warning",
        )
        st.stop()

    qtd_series = len(df)

    # Proteção anti-zumbi — Replay ULTRA é mais pesado
    if not limitar_operacao(
        qtd_series,
        limite_series=LIMITE_SERIES_REPLAY_ULTRA,
        contexto="Replay ULTRA completo",
        painel="🔁 Replay ULTRA",
    ):
        st.stop()

    st.info("Executando Replay ULTRA...")

    try:
        # DX Ultra = distância média entre cada série e o centróide global
        centr_global = np.mean(matriz_norm, axis=0)
        distancias = [
            np.linalg.norm(linha - centr_global) for linha in matriz_norm
        ]

        media_dx = float(np.mean(distancias))
        desvio_dx = float(np.std(distancias))

    except Exception as erro:
        exibir_bloco_mensagem(
            "Erro no Replay ULTRA",
            f"Detalhes técnicos: {erro}",
            tipo="error",
        )
        st.stop()

    corpo = (
        f"- Séries analisadas: **{qtd_series}**\n"
        f"- Distância média ao centróide (DX Ultra): **{media_dx:.4f}**\n"
        f"- Dispersão DX Ultra: **{desvio_dx:.4f}**\n"
        "\nValores maiores indicam estrada mais caótica."
    )

    exibir_bloco_mensagem(
        "Resumo do Replay ULTRA",
        corpo,
        tipo="info",
    )

    st.success("Replay ULTRA concluído!")

# ============================================================
# PARTE 3/8 — FIM
# ============================================================
# ============================================================
# PARTE 4/8 — INÍCIO
# ============================================================

# ============================================================
# Painel 6 — ⚙️ Modo TURBO++ HÍBRIDO
# ============================================================
if painel == "⚙️ Modo TURBO++ HÍBRIDO":

    st.markdown("## ⚙️ Modo TURBO++ HÍBRIDO — V15.7 MAX")

    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")

    if df is None or matriz_norm is None:
        exibir_bloco_mensagem(
            "Pipeline incompleto",
            "Execute o painel **🛣️ Pipeline V14-FLEX ULTRA**.",
            tipo="warning",
        )
        st.stop()

    qtd_series = len(df)

    # Anti-zumbi leve
    if not limitar_operacao(
        qtd_series,
        limite_series=LIMITE_PREVISOES_TURBO,
        contexto="TURBO++ HÍBRIDO",
        painel="⚙️ Modo TURBO++ HÍBRIDO",
    ):
        st.stop()

    st.info("Executando Modo TURBO++ HÍBRIDO...")

    # ============================================================
    # MOTOR HÍBRIDO — DX Light + S6 Light + Monte Carlo Light
    # ============================================================
    try:
        # DX Light — proximidade final
        vetor_final = matriz_norm[-1]
        distancias = [
            np.linalg.norm(vetor_final - linha) for linha in matriz_norm[:-1]
        ]

        # S6 Light — estatística simples dos passageiros
        col_pass = [c for c in df.columns if c.startswith("p")]
        ult = df[col_pass].iloc[-1].values

        s6_scores = []
        for idx in range(len(df) - 1):
            candidato = df[col_pass].iloc[idx].values
            intersec = len(set(candidato) & set(ult))
            s6_scores.append(intersec)

        # Monte Carlo Light — sorteio ponderado
        pesos_mc = np.array([1 / (1 + d) for d in distancias])
        pesos_mc = pesos_mc / pesos_mc.sum()

        escolha_idx = np.random.choice(len(pesos_mc), p=pesos_mc)
        previsao_mc = df[col_pass].iloc[escolha_idx].values.tolist()

        # Consolidação leve
        s6_melhor = df[col_pass].iloc[np.argmax(s6_scores)].values.tolist()
        dx_melhor = df[col_pass].iloc[np.argmin(distancias)].values.tolist()

        # Combinação híbrida
        previsao_final = list(
            np.round(
                0.4 * np.array(dx_melhor)
                + 0.3 * np.array(s6_melhor)
                + 0.3 * np.array(previsao_mc)
            )
        )
        previsao_final = [int(x) for x in previsao_final]

    except Exception as erro:
        exibir_bloco_mensagem(
            "Erro no TURBO++ HÍBRIDO",
            f"Detalhes: {erro}",
            tipo="error",
        )
        st.stop()

    # ============================================================
    # Exibição final
    # ============================================================
    st.markdown("### 🔮 Previsão HÍBRIDA (TURBO++)")
    st.success(f"**{formatar_lista_passageiros(previsao_final)}**")

    st.session_state["ultima_previsao"] = previsao_final

# ============================================================
# BLOCO 1/4 — ORQUESTRADOR DE TENTATIVA (V16) — INVISÍVEL
# Objetivo: traduzir diagnóstico (alvo/risco/confiabilidade) em
# "configuração de tentativa" para o Modo 6 (sem decidir listas).
# LISTAS SEMPRE EXISTEM: este orquestrador NUNCA retorna volume 0.
# ============================================================

from typing import Dict, Any, Optional


# ------------------------------------------------------------
# HELPERS (V16) — clamp + safe float
# ------------------------------------------------------------

def _clamp_v16(x: float, lo: float, hi: float) -> float:
    try:
        x = float(x)
    except Exception:
        x = lo
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _safe_float_v16(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


# ------------------------------------------------------------
# ORQUESTRADOR DE TENTATIVA (V16) — núcleo conceitual
# ------------------------------------------------------------

def orquestrar_tentativa_v16(
    *,
    series_count: int,
    alvo_tipo: Optional[str] = None,          # "parado" | "movimento_lento" | "movimento_rapido"
    alvo_velocidade: Optional[float] = None,  # ex: 0.9319 (se disponível)
    k_star: Optional[float] = None,           # ex: 0.2083
    nr_pct: Optional[float] = None,           # ex: 67.87  (0..100)
    divergencia_s6_mc: Optional[float] = None,# ex: 14.0480
    risco_composto: Optional[float] = None,   # ex: 0.7560  (0..1)
    confiabilidade_estimada: Optional[float] = None,  # 0..1 (se você já tiver)
    # --- Limites técnicos (anti-zumbi) ---
    limite_seguro_series_modo6: int = 800,    # padrão atual (já visto no app)
    # --- Volumes base (pode ser ajustado depois, mas COMEÇA CONSERVADOR) ---
    volume_min_base: int = 3,
    volume_rec_base: int = 6,
    volume_max_base: int = 80,
) -> Dict[str, Any]:
    """
    Retorna um dicionário com a "configuração de tentativa" (invisível),
    para o Modo 6 usar como guia de volume e forma (diversidade/variação).

    ✅ Regras implementadas aqui:
    - Objetivo único: tentar cravar 6 passageiros (não decide, só orienta).
    - Listas SEMPRE existem -> volume_min >= 1 (nunca 0).
    - Confiabilidade alta => explorar (mandar bala com critério).
    - Confiabilidade baixa => tentar com critério (degradado, mas não zero).
    - Anti-zumbi não censura: limita teto, mas não zera.
    """

    # -----------------------------
    # Sanitização básica
    # -----------------------------
    try:
        series_count = int(series_count)
    except Exception:
        series_count = 0

    k_star = _safe_float_v16(k_star, 0.0)
    nr_pct = _safe_float_v16(nr_pct, 0.0)
    divergencia_s6_mc = _safe_float_v16(divergencia_s6_mc, 0.0)
    risco_composto = _safe_float_v16(risco_composto, 0.0)

    # Normalizações defensivas
    nr_norm = _clamp_v16(nr_pct / 100.0, 0.0, 1.0)             # 0..1
    risco_norm = _clamp_v16(risco_composto, 0.0, 1.0)          # 0..1
    k_norm = _clamp_v16(k_star / 0.35, 0.0, 1.0)               # 0..1 (0.35 ~ teto típico de alerta)
    div_norm = _clamp_v16(divergencia_s6_mc / 15.0, 0.0, 1.0)  # 0..1 (15 ~ divergência crítica)

    # -----------------------------
    # Inferência do tipo de alvo (se não vier do Laudo)
    # -----------------------------
    alvo_tipo_norm = (alvo_tipo or "").strip().lower()

    if not alvo_tipo_norm:
        v = _safe_float_v16(alvo_velocidade, 0.0)
        # Heurística simples (pode refinar depois):
        # - <0.35: parado/lento
        # - 0.35..0.70: movimento_lento
        # - >0.70: movimento_rapido
        if v <= 0.35:
            alvo_tipo_norm = "parado"
        elif v <= 0.70:
            alvo_tipo_norm = "movimento_lento"
        else:
            alvo_tipo_norm = "movimento_rapido"

    if alvo_tipo_norm in ("lento", "movimento lento", "movimento-lento"):
        alvo_tipo_norm = "movimento_lento"
    if alvo_tipo_norm in ("rapido", "rápido", "movimento rapido", "movimento-rápido", "movimento_rapido"):
        alvo_tipo_norm = "movimento_rapido"
    if alvo_tipo_norm in ("parado", "estavel", "estável"):
        alvo_tipo_norm = "parado"

    if alvo_tipo_norm not in ("parado", "movimento_lento", "movimento_rapido"):
        alvo_tipo_norm = "movimento_rapido"  # default seguro: tratar como difícil

    # -----------------------------
    # Construção de uma "confiabilidade estimada" interna (se não vier)
    # -----------------------------
    # Ideia: confiabilidade cai com ruído, risco, k* alto e divergência alta.
    # (Não é promessa, é régua de orientação de intensidade.)
    if confiabilidade_estimada is None:
        penal = 0.40 * nr_norm + 0.25 * risco_norm + 0.20 * div_norm + 0.15 * k_norm
        conf = 1.0 - _clamp_v16(penal, 0.0, 1.0)
    else:
        conf = _clamp_v16(_safe_float_v16(confiabilidade_estimada, 0.0), 0.0, 1.0)

    # -----------------------------
    # Definição do "modo de tentativa" (conceito → controle interno)
    # -----------------------------
    # - exploração_intensa: alta confiança (mandar bala com critério)
    # - tentativa_controlada: meio termo
    # - tentativa_degradada: baixa confiança / alvo rápido / ambiente hostil
    if conf >= 0.55 and risco_norm <= 0.55 and nr_norm <= 0.55 and div_norm <= 0.60:
        modo = "exploracao_intensa"
    elif conf >= 0.30 and risco_norm <= 0.75 and nr_norm <= 0.75:
        modo = "tentativa_controlada"
    else:
        modo = "tentativa_degradada"

    # Alvo rápido puxa para degradado, a menos que seja realmente "bom"
    if alvo_tipo_norm == "movimento_rapido" and modo != "exploracao_intensa":
        modo = "tentativa_degradada"

    # -----------------------------
    # Volumes base (sempre > 0)
    # -----------------------------
    vol_min = max(1, int(volume_min_base))
    vol_rec = max(vol_min, int(volume_rec_base))
    vol_max = max(vol_rec, int(volume_max_base))

    # -----------------------------
    # Ajuste de intensidade por modo + confiabilidade
    # -----------------------------
    # Observação: "mandar bala" = aumentar volume e variação interna,
    # mas SEM explodir sem critério.
    if modo == "exploracao_intensa":
        # Escala com conf (0.55..1.0) -> multiplicador (1.1..1.9)
        mult = 1.1 + 0.8 * _clamp_v16((conf - 0.55) / 0.45, 0.0, 1.0)
        vol_rec = int(max(vol_rec, round(vol_rec * mult)))
        vol_max = int(max(vol_max, round(vol_max * mult)))

        diversidade = 0.55  # moderada (refino + variação)
        variacao_interna = 0.75
        aviso_curto = "🟢 Exploração intensa: mandar bala com critério (janela favorável)."

    elif modo == "tentativa_controlada":
        # Escala suave com conf (0.30..0.55) -> multiplicador (0.95..1.20)
        mult = 0.95 + 0.25 * _clamp_v16((conf - 0.30) / 0.25, 0.0, 1.0)
        vol_rec = int(max(vol_rec, round(vol_rec * mult)))
        vol_max = int(max(vol_max, round(vol_max * mult)))

        # diversidade depende do alvo
        if alvo_tipo_norm == "parado":
            diversidade = 0.35  # mais próximo (ajuste fino)
            variacao_interna = 0.60
        elif alvo_tipo_norm == "movimento_lento":
            diversidade = 0.50  # cercamento
            variacao_interna = 0.55
        else:
            diversidade = 0.65  # já puxa para hipóteses
            variacao_interna = 0.45

        aviso_curto = "🟡 Tentativa controlada: cercar com critério (sem exagero)."

    else:
        # Degradado: volume controlado, diversidade alta (hipóteses)
        # Garante mínimo, limita teto e aumenta diversidade.
        # Se conf for muito baixa, não adianta inflar volume: mantém enxuto.
        if conf <= 0.10:
            vol_rec = max(vol_min, min(vol_rec, 6))
            vol_max = max(vol_rec, min(vol_max, 12))
        elif conf <= 0.20:
            vol_rec = max(vol_min, min(vol_rec, 8))
            vol_max = max(vol_rec, min(vol_max, 18))
        else:
            vol_rec = max(vol_min, min(vol_rec, 10))
            vol_max = max(vol_rec, min(vol_max, 24))

        diversidade = 0.85  # alto (ali, lá, acolá)
        variacao_interna = 0.35
        aviso_curto = "🔴 Tentativa degradada: hipóteses espalhadas (chance baixa, mas listas existem)."

    # -----------------------------
    # Anti-zumbi como LIMITADOR (não censura)
    # -----------------------------
    # Se o histórico excede o limite seguro do modo 6:
    # - não bloqueia
    # - apenas derruba o teto e puxa recomendado para um patamar seguro
    # Mantém volume_min > 0 SEMPRE.
    if series_count > int(limite_seguro_series_modo6):
        # Fator de penalização pelo excesso de séries (piora custo)
        excesso = series_count - int(limite_seguro_series_modo6)
        fator = _clamp_v16(1.0 - (excesso / max(1.0, float(limite_seguro_series_modo6))) * 0.60, 0.25, 1.0)

        teto_seguro = int(max(vol_rec, round(vol_max * fator)))
        teto_seguro = int(_clamp_v16(teto_seguro, max(vol_rec, vol_min), vol_max))

        # puxa recomendado junto do teto seguro (mas nunca abaixo do mínimo)
        vol_max = max(vol_rec, teto_seguro)
        vol_rec = max(vol_min, min(vol_rec, vol_max))

        aviso_curto += " 🔒 Anti-Zumbi: volume limitado (sem bloquear geração)."

    # -----------------------------
    # Garantias finais (invioláveis)
    # -----------------------------
    vol_min = max(1, int(vol_min))
    vol_rec = max(vol_min, int(vol_rec))
    vol_max = max(vol_rec, int(vol_max))

    diversidade = _clamp_v16(diversidade, 0.10, 0.95)
    variacao_interna = _clamp_v16(variacao_interna, 0.10, 0.95)

    return {
        "modo_tentativa": modo,
        "alvo_tipo": alvo_tipo_norm,
        "confiabilidade_estimada": float(conf),
        "volume_min": int(vol_min),
        "volume_recomendado": int(vol_rec),
        "volume_max": int(vol_max),
        "diversidade": float(diversidade),
        "variacao_interna": float(variacao_interna),
        "aviso_curto": str(aviso_curto),
        "debug": {
            "nr_norm": float(nr_norm),
            "risco_norm": float(risco_norm),
            "k_norm": float(k_norm),
            "div_norm": float(div_norm),
            "series_count": int(series_count),
            "limite_seguro_series_modo6": int(limite_seguro_series_modo6),
        },
    }

# ============================================================
# BLOCO 2/4 — PONTE ORQUESTRADOR → TURBO++ ULTRA (V16)
# Objetivo: coletar diagnósticos existentes do app (Laudo/Risco)
# e preparar a configuração de tentativa para o Modo 6,
# SEM alterar UI e SEM decidir listas.
# ============================================================

def preparar_tentativa_turbo_ultra_v16(
    *,
    df,
    series_count: int,
    alvo_tipo: Optional[str] = None,
    alvo_velocidade: Optional[float] = None,
    k_star: Optional[float] = None,
    nr_pct: Optional[float] = None,
    divergencia_s6_mc: Optional[float] = None,
    risco_composto: Optional[float] = None,
    confiabilidade_estimada: Optional[float] = None,
    limite_seguro_series_modo6: int = 800,
) -> Dict[str, Any]:
    """
    Ponte invisível:
    - lê informações já calculadas no app
    - chama o Orquestrador de Tentativa (BLOCO 1)
    - devolve um dicionário pronto para o TURBO++ ULTRA usar

    NÃO gera listas
    NÃO executa motores
    NÃO decide nada
    """

    # Defesa básica
    try:
        series_count = int(series_count)
    except Exception:
        series_count = 0

    # Chamada central ao Orquestrador
    cfg = orquestrar_tentativa_v16(
        series_count=series_count,
        alvo_tipo=alvo_tipo,
        alvo_velocidade=alvo_velocidade,
        k_star=k_star,
        nr_pct=nr_pct,
        divergencia_s6_mc=divergencia_s6_mc,
        risco_composto=risco_composto,
        confiabilidade_estimada=confiabilidade_estimada,
        limite_seguro_series_modo6=limite_seguro_series_modo6,
    )

    # Normalização final (garantia extra)
    cfg["volume_min"] = max(1, int(cfg.get("volume_min", 1)))
    cfg["volume_recomendado"] = max(
        cfg["volume_min"],
        int(cfg.get("volume_recomendado", cfg["volume_min"]))
    )
    cfg["volume_max"] = max(
        cfg["volume_recomendado"],
        int(cfg.get("volume_max", cfg["volume_recomendado"]))
    )

    return cfg

# ============================================================
# >>> INÍCIO — BLOCO 3/4 — ORQUESTRADOR → TURBO++ ULTRA (V16)
# Camada invisível de conexão (não é painel, não gera listas)
# ============================================================

def _injetar_cfg_tentativa_turbo_ultra_v16(
    *,
    df,
    qtd_series: int,
    k_star,
    limite_series_padrao: int,
):
    """
    Injeta no session_state a configuração de tentativa calculada
    pelo Orquestrador (BLOCO 1 + BLOCO 2), sem bloquear execução.
    """

    # Coleta informações já existentes
    laudo_v16 = st.session_state.get("laudo_operacional_v16", {}) or {}

    alvo_tipo = laudo_v16.get("estado_alvo") or laudo_v16.get("alvo_tipo")
    alvo_velocidade = laudo_v16.get("velocidade_estimada")

    nr_pct = st.session_state.get("nr_pct")
    divergencia_s6_mc = st.session_state.get("divergencia_s6_mc")
    risco_composto = st.session_state.get("indice_risco")

    cfg = preparar_tentativa_turbo_ultra_v16(
        df=df,
        series_count=qtd_series,
        alvo_tipo=alvo_tipo,
        alvo_velocidade=alvo_velocidade,
        k_star=k_star,
        nr_pct=nr_pct,
        divergencia_s6_mc=divergencia_s6_mc,
        risco_composto=risco_composto,
        limite_seguro_series_modo6=limite_series_padrao,
    )

    # Guarda para uso posterior
    st.session_state["cfg_tentativa_turbo_ultra"] = cfg

    # Aviso curto (informativo, não bloqueante)
    aviso = cfg.get("aviso_curto")
    if aviso:
        st.caption(aviso)

    # Define limite efetivo (anti-zumbi vira limitador, não censura)
    limite_efetivo = min(
        limite_series_padrao,
        int(cfg.get("volume_max", limite_series_padrao))
    )

    return limite_efetivo


# ============================================================
# <<< FIM — BLOCO 3/4 — ORQUESTRADOR → TURBO++ ULTRA (V16)
# ============================================================

# ============================================================
# >>> PAINEL 7 — ⚙️ Modo TURBO++ ULTRA (MVP3 — VOLUME POR ORÇAMENTO)
# ============================================================

if painel == "⚙️ Modo TURBO++ ULTRA":

    st.markdown("## ⚙️ Modo TURBO++ ULTRA — MVP3")
    st.caption(
        "Exploração controlada.\n\n"
        "✔ Motor original preservado\n"
        "✔ Anti-zumbi respeitado\n"
        "✔ Volume liberado por orçamento\n"
        "✔ Falha silenciosa permitida\n"
        "✔ Sem decisão automática"
    )

    # ------------------------------------------------------------
    # BUSCA DE ESTADO (SOMENTE AQUI)
    # ------------------------------------------------------------
    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")

    _kstar_raw = st.session_state.get("sentinela_kstar")
    k_star = float(_kstar_raw) if isinstance(_kstar_raw, (int, float)) else 0.0

    if df is None or df.empty or matriz_norm is None:
        exibir_bloco_mensagem(
            "Pipeline incompleto",
            "Carregue o histórico e execute **🛣️ Pipeline V14-FLEX ULTRA**.",
            tipo="warning",
        )
        st.stop()

    qtd_series = len(df)

    # ------------------------------------------------------------
    # 🔒 MARCAÇÃO OFICIAL — TURBO EXECUTADO (ANTES DO ANTI-ZUMBI)
    # (Se o Anti-Zumbi bloquear com st.stop, o Modo 6 NÃO fica travado)
    # ------------------------------------------------------------
    st.session_state["turbo_ultra_executado"] = True
    st.session_state["turbo_executado"] = True
    st.session_state["turbo_ultra_rodou"] = True
    st.session_state["motor_turbo_executado"] = True

    # ------------------------------------------------------------
    # ANTI-ZUMBI — LIMITADOR OFICIAL
    # (COMPORTAMENTO OFICIAL PRESERVADO)
    # ------------------------------------------------------------
    LIMITE_SERIES_TURBO_ULTRA_EFETIVO = _injetar_cfg_tentativa_turbo_ultra_v16(
        df=df,
        qtd_series=qtd_series,
        k_star=k_star,
        limite_series_padrao=LIMITE_SERIES_TURBO_ULTRA,
    )

    limitar_operacao(
        qtd_series,
        limite_series=LIMITE_SERIES_TURBO_ULTRA_EFETIVO,
        contexto="TURBO++ ULTRA",
        painel="⚙️ Modo TURBO++ ULTRA",
    )
    # ⬆️ se bloquear, a própria função já dá st.stop()
    # ✅ e a marcação acima já foi gravada, então o Modo 6 destrava.

    # ------------------------------------------------------------
    # ORÇAMENTO
    # ------------------------------------------------------------
    orcamentos_disponiveis = [6, 42, 168, 504, 1260, 2772]

    orcamento = st.selectbox(
        "Selecione o orçamento para o TURBO++ ULTRA:",
        options=orcamentos_disponiveis,
        index=1,
    )

    mapa_execucoes = {
        6: 1,
        42: 1,
        168: 3,
        504: 6,
        1260: 10,
        2772: 20,
    }

    n_exec = mapa_execucoes.get(int(orcamento), 1)

    st.info(
        f"🔢 Orçamento selecionado: **{orcamento}**\n\n"
        f"▶️ Execuções do TURBO++ ULTRA: **{n_exec}**"
    )

    # ------------------------------------------------------------
    # EXECUÇÃO SEGURA DO TURBO++ ULTRA
    # ------------------------------------------------------------
    st.info("Executando Modo TURBO++ ULTRA...")

    todas_listas = []

    for _ in range(n_exec):
        try:
            lista = turbo_ultra_v15_7(
                df=df,
                matriz_norm=matriz_norm,
                k_star=k_star,
            )
            if isinstance(lista, list) and len(lista) >= 6:
                todas_listas.append(lista)
        except Exception:
            pass

    # ------------------------------------------------------------
    # FECHAMENTO DE ESTADO DO PIPELINE
    # ------------------------------------------------------------
    st.session_state["pipeline_flex_ultra_concluido"] = True
    st.session_state["turbo_ultra_listas_leves"] = todas_listas.copy()
    st.session_state["ultima_previsao"] = todas_listas

    if not todas_listas:
        st.warning(
            "Nenhuma lista foi gerada nesta condição.\n\n"
            "Isso é um **resultado válido**.\n"
            "O motor foi executado (ou bloqueado) e falhou silenciosamente."
        )
        st.stop()

    st.success(
        f"✅ TURBO++ ULTRA executado com sucesso.\n\n"
        f"📦 Listas geradas: **{len(todas_listas)}**"
    )

    st.markdown("### 🔮 Listas geradas (amostra)")
    st.write(todas_listas[: min(5, len(todas_listas))])

# ============================================================
# <<< FIM — PAINEL 7 — ⚙️ Modo TURBO++ ULTRA (MVP3)
# ============================================================





# ============================================================
# MOTORES PROFUNDOS (PUROS)
# NÃO executam sozinhos
# NÃO acessam session_state
# NÃO exibem nada
# ============================================================

# --- S6 PROFUNDO ---
def s6_profundo_V157(df_local, col_pass, idx_alvo):
    ult_local = df_local[col_pass].iloc[idx_alvo].values
    scores_local = []
    for i_local in range(len(df_local) - 1):
        base_local = df_local[col_pass].iloc[i_local].values
        inter_local = len(set(base_local) & set(ult_local))
        scores_local.append(inter_local)
    melhores_idx_local = np.argsort(scores_local)[-25:]
    candidatos_local = df_local[col_pass].iloc[melhores_idx_local].values
    return candidatos_local


# --- MICRO-LEQUE PROFUNDO ---
def micro_leque_profundo(base, profundidade=20):
    leque = []
    for delta in range(-profundidade, profundidade + 1):
        novo = [max(1, min(60, x + delta)) for x in base]
        leque.append(novo)
    return np.array(leque)


# --- MONTE CARLO PROFUNDO ---
def monte_carlo_profundo(base, n=800):
    sims = []
    for _ in range(n):
        ruido = np.random.randint(-5, 6, size=len(base))
        candidato = base + ruido
        candidato = np.clip(candidato, 1, 60)
        sims.append(candidato.tolist())
    return sims




# ============================================================
# Painel 8 — 📡 Painel de Ruído Condicional
# ============================================================

if painel == "📡 Painel de Ruído Condicional":

    st.markdown("## 📡 Painel de Ruído Condicional — V15.7 MAX")

    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")

    if df is None or matriz_norm is None:
        exibir_bloco_mensagem(
            "Pipeline não encontrado",
            "Execute primeiro **📁 Carregar Histórico** e **🛣️ Pipeline V14-FLEX ULTRA**.",
            tipo="warning",
        )
        st.stop()

    qtd_series = len(df)

    if not limitar_operacao(
        qtd_series,
        limite_series=LIMITE_SERIES_REPLAY_ULTRA,
        contexto="Ruído Condicional",
        painel="📡 Painel de Ruído Condicional",
    ):
        st.stop()

    st.info("Calculando indicadores de ruído condicional...")

# ============================================================
# PARTE 4/8 — FIM
# ============================================================
# ============================================================
# PARTE 5/8 — INÍCIO
# ============================================================

    try:
        # Ruído Tipo A: dispersão intra-série (variação entre passageiros)
        variancias_intra = np.var(matriz_norm, axis=1)
        ruido_A_medio = float(np.mean(variancias_intra))

        # Ruído Tipo B: salto entre séries consecutivas (DX Light já usado)
        saltos = []
        for i in range(1, len(matriz_norm)):
            dist = np.linalg.norm(matriz_norm[i] - matriz_norm[i - 1])
            saltos.append(dist)
        ruido_B_medio = float(np.mean(saltos))

        # Normalização aproximada dos ruídos em [0,1]
        # (evitando divisão por zero)
        ruido_A_norm = min(1.0, ruido_A_medio / 0.08)   # escala empírica
        ruido_B_norm = min(1.0, ruido_B_medio / 1.20)   # escala empírica

        nr_percent = float((0.55 * ruido_A_norm + 0.45 * ruido_B_norm) * 100.0)

    except Exception as erro:
        exibir_bloco_mensagem(
            "Erro no cálculo de ruído",
            f"Detalhes técnicos: {erro}",
            tipo="error",
        )
        st.stop()

    # Classificação simples do NR%
    if nr_percent < 20:
        classe = "🟢 Baixo Ruído (Ambiente limpo)"
    elif nr_percent < 40:
        classe = "🟡 Ruído Moderado (Cuidado)"
    elif nr_percent < 60:
        classe = "🟠 Ruído Elevado (Atenção forte)"
    else:
        classe = "🔴 Ruído Crítico (Alta contaminação)"

    corpo = (
        f"- Séries analisadas: **{qtd_series}**\n"
        f"- Ruído Tipo A (intra-série, médio): **{ruido_A_medio:.4f}**\n"
        f"- Ruído Tipo B (entre séries, médio): **{ruido_B_medio:.4f}**\n"
        f"- NR% (Ruído Condicional Normalizado): **{nr_percent:.2f}%**\n"
        f"- Classe de ambiente: {classe}"
    )

    exibir_bloco_mensagem(
        "Resumo do Ruído Condicional",
        corpo,
        tipo="info",
    )

    st.session_state["nr_percent"] = nr_percent
    st.success("Cálculo de Ruído Condicional concluído!")


# ============================================================
# Painel 9 — 📉 Painel de Divergência S6 vs MC
# ============================================================
if painel == "📉 Painel de Divergência S6 vs MC":

    st.markdown("## 📉 Painel de Divergência S6 vs MC — V15.7 MAX")

    divergencia = st.session_state.get("div_s6_mc", None)

    if divergencia is None:
        exibir_bloco_mensagem(
            "Divergência não calculada",
            "Execute o painel **⚙️ Modo TURBO++ ULTRA** para gerar a divergência S6 vs MC.",
            tipo="warning",
        )
        st.stop()

    # Classificação da divergência
    if divergencia < 2.0:
        classe = "🟢 Alta Convergência (S6 ≈ MC)"
        comentario = (
            "Os motores S6 Profundo e Monte Carlo Profundo estão altamente alinhados. "
            "O núcleo preditivo é mais confiável, favorecendo decisões mais agressivas."
        )
    elif divergencia < 5.0:
        classe = "🟡 Convergência Parcial"
        comentario = (
            "Há uma diferença moderada entre S6 e Monte Carlo. "
            "As decisões permanecem utilizáveis, mas requerem atenção adicional."
        )
    else:
        classe = "🔴 Alta Divergência (S6 distante de MC)"
        comentario = (
            "Os motores S6 e Monte Carlo estão em desacordo significativo. "
            "A recomendação é reduzir agressividade, aumentar coberturas ou aguardar estabilização."
        )

    corpo = (
        f"- Divergência S6 vs MC (norma): **{divergencia:.4f}**\n"
        f"- Classe de alinhamento: {classe}\n\n"
        f"{comentario}"
    )

    exibir_bloco_mensagem(
        "Resumo da Divergência S6 vs MC",
        corpo,
        tipo="info",
    )

    st.success("Análise de divergência concluída!")

# ============================================================
# PAINEL — 🧼 B1 | Higiene de Passageiros (V16)
# Observacional | NÃO decide | NÃO altera motores
# ============================================================

elif painel == "🧼 B1 — Higiene de Passageiros":

    st.markdown("## 🧼 B1 — Higiene de Passageiros (V16)")
    st.caption(
        "Leitura observacional para identificar passageiros resilientes e nocivos.\n"
        "Não remove números. Não decide listas. Preparação para Perna B."
    )

    df = st.session_state.get("historico_df")

    if df is None or df.empty:
        st.info("Histórico não carregado.")
        st.stop()

    # ------------------------------------------------------------
    # Detecta colunas de passageiros (n-base)
    # ------------------------------------------------------------
    col_pass = [c for c in df.columns if c.startswith("p")]

    if len(col_pass) < 6:
        st.warning("Não foi possível detectar passageiros suficientes.")
        st.stop()

    # ------------------------------------------------------------
    # Frequência simples dos passageiros
    # ------------------------------------------------------------
    freq = {}

    for _, row in df.iterrows():
        for c in col_pass:
            try:
                v = int(row[c])
                if v > 0:
                    freq[v] = freq.get(v, 0) + 1
            except Exception:
                pass

    if not freq:
        st.warning("Frequência de passageiros vazia.")
        st.stop()

    total_series = len(df)

    # ------------------------------------------------------------
    # Métricas observacionais
    # ------------------------------------------------------------
    dados = []

    for p, f in freq.items():
        taxa = f / total_series

        # heurísticas simples (OBSERVAÇÃO)
        resiliente = taxa >= 0.18
        nocivo = taxa <= 0.05

        dados.append({
            "Passageiro": p,
            "Ocorrências": f,
            "Taxa": round(taxa, 4),
            "Resiliente": "✅" if resiliente else "",
            "Nocivo": "⚠️" if nocivo else "",
        })

    df_pass = pd.DataFrame(dados).sort_values(
        by="Taxa", ascending=False
    )

    # ------------------------------------------------------------
    # Exibição
    # ------------------------------------------------------------
    st.markdown("### 📊 Leitura de Frequência dos Passageiros")
    st.dataframe(df_pass, use_container_width=True)

    # ------------------------------------------------------------
    # Síntese mastigada (NÃO decisória)
    # ------------------------------------------------------------
    resilientes = df_pass[df_pass["Resiliente"] == "✅"]["Passageiro"].tolist()
    nocivos = df_pass[df_pass["Nocivo"] == "⚠️"]["Passageiro"].tolist()

    leitura = (
        f"- Passageiros resilientes (recorrência alta): {resilientes[:12]}\n"
        f"- Passageiros potencialmente nocivos (recorrência muito baixa): {nocivos[:12]}\n\n"
        "⚠️ Nenhum passageiro foi removido.\n"
        "⚠️ Esta leitura serve apenas como preparação para refinamento futuro."
    )

    exibir_bloco_mensagem(
        "🧠 Leitura Observacional — Higiene de Passageiros",
        leitura,
        tipo="info",
    )

    # ------------------------------------------------------------
    # Registro silencioso para Perna B
    # ------------------------------------------------------------
    st.session_state["b1_resilientes"] = resilientes
    st.session_state["b1_nocivos"] = nocivos

    st.success("B1 concluído — leitura registrada com sucesso.")

# ============================================================
# <<< FIM — PAINEL 🧼 B1 | Higiene de Passageiros
# ============================================================

# ============================================================
# PAINEL — 🧩 B2 | Coerência Interna das Listas (V16)
# Observacional | NÃO decide | NÃO altera motores
# ============================================================

elif painel == "🧩 B2 — Coerência Interna das Listas":

    st.markdown("## 🧩 B2 — Coerência Interna das Listas (V16)")
    st.caption(
        "Leitura observacional de coesão e conflitos internos das listas.\n"
        "Não filtra, não prioriza, não decide."
    )

    # ------------------------------------------------------------
    # Fonte das listas (preferência: Modo 6)
    # ------------------------------------------------------------
    listas = (
        st.session_state.get("modo6_listas_totais")
        or st.session_state.get("modo6_listas")
        or []
    )

    if not listas:
        st.info("Nenhuma lista disponível para análise. Execute o Modo 6.")
        st.stop()

    # ------------------------------------------------------------
    # Universo e estatísticas globais
    # ------------------------------------------------------------
    todas = [x for lst in listas for x in lst if isinstance(x, int)]
    if not todas:
        st.warning("Listas inválidas para análise.")
        st.stop()

    freq_global = pd.Series(todas).value_counts(normalize=True)

    # ------------------------------------------------------------
    # Métricas por lista
    # ------------------------------------------------------------
    linhas = []

    for i, lst in enumerate(listas, start=1):
        lst = [int(x) for x in lst if isinstance(x, int)]
        if not lst:
            continue

        # Coesão: média da frequência global dos elementos
        coesao = float(freq_global.loc[lst].mean()) if set(lst).issubset(freq_global.index) else 0.0

        # Conflito simples: proporção de pares muito raros juntos
        pares = [(a, b) for a in lst for b in lst if a < b]
        raros = 0
        for a, b in pares:
            fa = freq_global.get(a, 0.0)
            fb = freq_global.get(b, 0.0)
            if fa < 0.05 and fb < 0.05:
                raros += 1

        conflito = raros / max(1, len(pares))

        linhas.append({
            "Lista": i,
            "Coesão (↑ melhor)": round(coesao, 4),
            "Conflito (↓ melhor)": round(conflito, 4),
        })

    df_b2 = pd.DataFrame(linhas)

    # ------------------------------------------------------------
    # Exibição
    # ------------------------------------------------------------
    st.markdown("### 📊 Coesão e Conflito por Lista")
    st.dataframe(df_b2, use_container_width=True)

    # ------------------------------------------------------------
    # Síntese mastigada (NÃO decisória)
    # ------------------------------------------------------------
    leitura = (
        "- **Coesão** alta indica elementos com histórico compatível.\n"
        "- **Conflito** alto indica combinações raras juntas.\n\n"
        "⚠️ Nenhuma lista foi removida ou priorizada.\n"
        "⚠️ Use esta leitura apenas para preparação."
    )

    exibir_bloco_mensagem(
        "🧠 Leitura Observacional — Coerência Interna",
        leitura,
        tipo="info",
    )

    # Registro silencioso
    st.session_state["b2_coerencia_df"] = df_b2

    st.success("B2 concluído — leitura registrada com sucesso.")

# ============================================================
# <<< FIM — PAINEL 🧩 B2 | Coerência Interna das Listas
# ============================================================

# ============================================================
# PAINEL — 🟢 B3 | Prontidão para Refinamento (V16)
# Observacional | NÃO decide | NÃO altera motores
# ============================================================

elif painel == "🟢 B3 — Prontidão (Refinamento)":

    st.markdown("## 🟢 B3 — Prontidão para Refinamento (V16)")
    st.caption(
        "Avalia se o contexto permite avançar da leitura (Perna B) "
        "para refinamento de passageiros. Não executa nada."
    )

    # ------------------------------------------------------------
    # Leituras já consolidadas
    # ------------------------------------------------------------
    diag = st.session_state.get("diagnostico_eco_estado_v16", {})
    df_b2 = st.session_state.get("b2_coerencia_df")

    if not diag or df_b2 is None or df_b2.empty:
        st.info(
            "Leituras insuficientes para avaliar prontidão.\n\n"
            "Execute B1, B2 e Diagnóstico ECO & Estado."
        )
        st.stop()

    eco = diag.get("eco")
    eco_persist = diag.get("eco_persistencia")
    acion = diag.get("eco_acionabilidade")
    estado = diag.get("estado")
    estado_ok = diag.get("estado_confiavel")

    # ------------------------------------------------------------
    # Heurísticas de prontidão (OBSERVAÇÃO)
    # ------------------------------------------------------------
    sinais_ok = []

    if eco in ("médio", "forte"):
        sinais_ok.append("ECO ≥ médio")

    if eco == "fraco" and eco_persist == "persistente" and estado in ("parado", "movimento_lento"):
        sinais_ok.append("ECO fraco porém estável com estado calmo")

    if estado_ok and estado in ("parado", "movimento_lento"):
        sinais_ok.append("Estado desacelerado e confiável")

    # Coesão média das listas
    coesao_media = float(df_b2["Coesão (↑ melhor)"].mean())

    if coesao_media >= 0.12:
        sinais_ok.append("Coesão média aceitável")

    # ------------------------------------------------------------
    # Veredito OBSERVACIONAL
    # ------------------------------------------------------------
    pronto = len(sinais_ok) >= 3

    if pronto:
        status = "🟢 PRONTO PARA REFINAMENTO"
        detalhe = (
            "O contexto permite iniciar refinamento controlado de passageiros.\n"
            "A Perna B pode evoluir para ações leves (sem afunilar)."
        )
        tipo = "success"
    else:
        status = "🟡 AINDA EM PREPARAÇÃO"
        detalhe = (
            "O contexto ainda pede dispersão.\n"
            "Continue observando e acumulando leitura."
        )
        tipo = "info"

    corpo = (
        f"**Status:** {status}\n\n"
        f"**Sinais atendidos:** {sinais_ok if sinais_ok else 'Nenhum'}\n\n"
        f"**Coesão média das listas:** {coesao_media:.4f}\n\n"
        f"⚠️ Este painel **não executa refinamento**.\n"
        f"⚠️ Serve apenas para indicar **prontidão**."
    )

    exibir_bloco_mensagem(
        "🧠 Veredito de Prontidão — Perna B",
        corpo,
        tipo=tipo,
    )

    # Registro silencioso
    st.session_state["b3_pronto_refinar"] = pronto

    st.success("B3 concluído — prontidão avaliada.")

# ============================================================
# <<< FIM — PAINEL 🟢 B3 | Prontidão para Refinamento
# ============================================================

# ============================================================
# PAINEL — 🟣 B4 | Refinamento Leve de Passageiros (V16)
# Ajuste leve | Reversível | NÃO decide | NÃO afunila
# ============================================================

elif painel == "🟣 B4 — Refinamento Leve de Passageiros":

    st.markdown("## 🟣 B4 — Refinamento Leve de Passageiros (V16)")
    st.caption(
        "Aplica ajustes leves e reversíveis nos passageiros das listas.\n"
        "Não reduz volume, não prioriza, não decide."
    )

    # ------------------------------------------------------------
    # Pré-condições
    # ------------------------------------------------------------
    pronto = st.session_state.get("b3_pronto_refinar", False)
    listas = (
        st.session_state.get("modo6_listas_totais")
        or st.session_state.get("modo6_listas")
        or []
    )

    resilientes = st.session_state.get("b1_resilientes", [])
    nocivos = st.session_state.get("b1_nocivos", [])

    if not listas:
        st.info("Nenhuma lista disponível. Execute o Modo 6.")
        st.stop()

    if not pronto:
        st.warning(
            "Contexto ainda não marcado como pronto para refinamento.\n"
            "Este painel é **apenas demonstrativo** neste estado."
        )

    # ------------------------------------------------------------
    # Universo de referência
    # ------------------------------------------------------------
    universo = sorted({int(x) for lst in listas for x in lst if isinstance(x, int)})
    if not universo:
        st.warning("Universo inválido para refinamento.")
        st.stop()

    rng = np.random.default_rng(42)

    # ------------------------------------------------------------
    # Refinamento leve (heurístico, reversível)
    # ------------------------------------------------------------
    listas_refinadas = []

    for lst in listas:
        nova = list(lst)

        # substitui no máx. 1 passageiro nocivo por um resiliente
        candidatos_nocivos = [x for x in nova if x in nocivos]
        candidatos_resilientes = [x for x in resilientes if x not in nova]

        if candidatos_nocivos and candidatos_resilientes:
            sai = rng.choice(candidatos_nocivos)
            entra = rng.choice(candidatos_resilientes)
            nova = [entra if x == sai else x for x in nova]

        listas_refinadas.append(sorted(set(nova)))

    # ------------------------------------------------------------
    # Exibição comparativa (leitura)
    # ------------------------------------------------------------
    st.markdown("### 🔍 Comparação — Antes × Depois (amostra)")
    limite = min(10, len(listas))

    for i in range(limite):
        col1, col2 = st.columns(2)
        with col1:
            st.code(f"Antes {i+1}: {sorted(listas[i])}", language="python")
        with col2:
            st.code(f"Depois {i+1}: {listas_refinadas[i]}", language="python")

    # ------------------------------------------------------------
    # Síntese observacional
    # ------------------------------------------------------------
    leitura = (
        "- Ajuste máximo: **1 passageiro por lista**\n"
        "- Volume total preservado\n"
        "- Refinamento **reversível**\n"
        "- Uso de passageiros resilientes\n\n"
        "⚠️ As listas refinadas **não substituem** as originais.\n"
        "⚠️ Cabe ao operador decidir se usa esta leitura."
    )

    exibir_bloco_mensagem(
        "🧠 Leitura Observacional — Refinamento Leve",
        leitura,
        tipo="info",
    )

    # Registro silencioso (não substitui listas)
    st.session_state["b4_listas_refinadas"] = listas_refinadas

    st.success("B4 concluído — refinamento leve avaliado.")

# ============================================================
# <<< FIM — PAINEL 🟣 B4 | Refinamento Leve de Passageiros
# ============================================================


# ============================================================
# Painel 10 — 🧭 Monitor de Risco — k & k*
# ============================================================
if painel == "🧭 Monitor de Risco — k & k*":

    st.markdown("## 🧭 Monitor de Risco — k & k* — V15.7 MAX")

    df = st.session_state.get("historico_df")
    k_star = st.session_state.get("sentinela_kstar")
    nr_percent = st.session_state.get("nr_percent")
    divergencia = st.session_state.get("div_s6_mc")

    if df is None:
        exibir_bloco_mensagem(
            "Histórico não carregado",
            "Vá ao painel **📁 Carregar Histórico** antes.",
            tipo="warning",
        )
        st.stop()

    metricas = calcular_metricas_basicas_historico(df)

    qtd_series = metricas.get("qtd_series", 0)
    min_k = metricas.get("min_k")
    max_k = metricas.get("max_k")
    media_k = metricas.get("media_k")

    # Garantias (se sentinelas/ruído/divergência não tiverem sido rodados)
    if k_star is None:
        k_star = 0.25  # valor neutro
    if nr_percent is None:
        nr_percent = 35.0  # ruído moderado default
    if divergencia is None:
        divergencia = 4.0  # divergência intermediária

    # Índice de risco composto (escala 0 a 1)
    # k* alto, NR% alto e divergência alta => risco maior
    kstar_norm = min(1.0, k_star / 0.50)
    nr_norm = min(1.0, nr_percent / 70.0)
    div_norm = min(1.0, divergencia / 8.0)

    indice_risco = float(0.40 * kstar_norm + 0.35 * nr_norm + 0.25 * div_norm)

    # Classificação de risco
    if indice_risco < 0.30:
        classe_risco = "🟢 Risco Baixo (Janela Favorável)"
        recomendacao = (
            "O ambiente está favorável para decisões mais agressivas, "
            "com menor necessidade de coberturas pesadas."
        )
    elif indice_risco < 0.55:
        classe_risco = "🟡 Risco Moderado"
        recomendacao = (
            "Ambiente misto. Recomenda-se equilíbrio entre núcleo e coberturas, "
            "com atenção à divergência e ao ruído."
        )
    elif indice_risco < 0.80:
        classe_risco = "🟠 Risco Elevado"
        recomendacao = (
            "Ambiente turbulento. Aumentar coberturas, reduzir exposição e "
            "observar de perto os painéis de Ruído e Divergência."
        )
    else:
        classe_risco = "🔴 Risco Crítico"
        recomendacao = (
            "Condição crítica. Sugere-se extrema cautela, priorizando preservação e "
            "eventualmente aguardando melhoria do regime antes de decisões mais fortes."
        )

    corpo = (
        f"- Séries no histórico: **{qtd_series}**\n"
        f"- k mínimo: **{min_k}** · k máximo: **{max_k}** · k médio: **{media_k:.2f}**\n"
        f"- k* (sentinela): **{k_star:.4f}**\n"
        f"- NR% (Ruído Condicional): **{nr_percent:.2f}%**\n"
        f"- Divergência S6 vs MC: **{divergencia:.4f}**\n"
        f"- Índice composto de risco: **{indice_risco:.4f}**\n"
        f"- Classe de risco: {classe_risco}\n\n"
        f"{recomendacao}"
    )

    exibir_bloco_mensagem(
        "Resumo do Monitor de Risco — k & k*",
        corpo,
        tipo="info",
    )

    st.session_state["diagnostico_risco"] = {
        "indice_risco": indice_risco,
        "classe_risco": classe_risco,
        "k_star": k_star,
        "nr_percent": nr_percent,
        "divergencia": divergencia,
    }

    st.success("Monitor de Risco atualizado com sucesso!")

# ============================================================
# PARTE 5/8 — FIM
# ============================================================
# ============================================================
# PARTE 6/8 — INÍCIO
# ============================================================

# ============================================================
# BLOCO V16 — PROTOCOLO PRÉ-ECO / ECO
# Observador tático — AJUSTA POSTURA PARA A PRÓXIMA SÉRIE
# NÃO prevê, NÃO altera motor, NÃO bloqueia
# ============================================================

def v16_avaliar_pre_eco_eco():
    """
    Usa SOMENTE o estado ATUAL (última série do histórico)
    para definir a postura de ataque da PRÓXIMA série.
    """

    k_star = st.session_state.get("sentinela_kstar")
    nr = st.session_state.get("nr_percent")
    div = st.session_state.get("div_s6_mc")
    risco = (st.session_state.get("diagnostico_risco") or {}).get("indice_risco")

    # Defaults defensivos
    k_star = float(k_star) if isinstance(k_star, (int, float)) else 0.30
    nr = float(nr) if isinstance(nr, (int, float)) else 50.0
    div = float(div) if isinstance(div, (int, float)) else 6.0
    risco = float(risco) if isinstance(risco, (int, float)) else 0.60

    sinais_ok = 0

    if k_star <= 0.30:
        sinais_ok += 1
    if nr <= 45.0:
        sinais_ok += 1
    if div <= 6.0:
        sinais_ok += 1
    if risco <= 0.55:
        sinais_ok += 1

    # Classificação
    if sinais_ok >= 3:
        status = "PRE_ECO_ATIVO"
        postura = "ATIVA"
        comentario = (
            "🟡 PRÉ-ECO detectado — ambiente NÃO piora.\n"
            "Postura ativa para a próxima série.\n"
            "Modo 6 ligado, volume moderado."
        )
    else:
        status = "SEM_ECO"
        postura = "DEFENSIVA"
        comentario = (
            "🔴 Nenhum pré-eco — ambiente instável.\n"
            "Operar apenas com coberturas."
        )

    resultado = {
        "status": status,
        "postura": postura,
        "sinais_ok": sinais_ok,
        "comentario": comentario,
    }

    st.session_state["v16_pre_eco"] = resultado
    return resultado

# ============================================================
# FUNÇÃO — SANIDADE FINAL DAS LISTAS (DISPONÍVEL AO MODO 6)
# Remove listas inválidas, duplicatas e permutações
# Válido para V15.7 MAX e V16 Premium
# ============================================================

def sanidade_final_listas(listas):
    """
    Sanidade final das listas de previsão.
    Regras:
    - Remove listas com números repetidos internamente
    - Remove permutações (ordem diferente, mesmos números)
    - Remove duplicatas exatas
    - Garante apenas listas válidas com 6 números distintos
    """
    if not listas:
        return []

    listas_saneadas = []
    vistos = set()

    for lista in listas:
        try:
            nums = [int(x) for x in lista]
        except Exception:
            continue

        # exatamente 6 números distintos
        if len(nums) != 6 or len(set(nums)) != 6:
            continue

        chave = tuple(sorted(nums))
        if chave in vistos:
            continue

        vistos.add(chave)
        listas_saneadas.append(nums)

    return listas_saneadas

# ============================================================
# FIM — FUNÇÃO SANIDADE FINAL DAS LISTAS
# ============================================================

# ============================================================
# B0 — SANIDADE DE UNIVERSO (V16)
# Observacional + corretivo leve
# Garante que nenhum passageiro fora do universo real apareça
# NÃO altera motores | NÃO decide | NÃO bloqueia
# ============================================================

def v16_sanidade_universo_listas(listas, historico_df):
    """
    Remove / ajusta números fora do universo real observado no histórico.
    Universo é inferido EXCLUSIVAMENTE do histórico carregado.
    """

    if historico_df is None or historico_df.empty:
        return listas  # sem histórico, não mexe

    col_pass = [c for c in historico_df.columns if c.startswith("p")]
    valores = []

    for _, row in historico_df.iterrows():
        for c in col_pass:
            try:
                v = int(row[c])
                if v > 0:
                    valores.append(v)
            except Exception:
                pass

    if not valores:
        return listas

    umin = min(valores)
    umax = max(valores)

    listas_sanas = []

    for lst in listas:
        nova = []
        for v in lst:
            try:
                iv = int(v)
                if iv < umin:
                    iv = umin
                elif iv > umax:
                    iv = umax
                nova.append(iv)
            except Exception:
                pass

        nova = list(dict.fromkeys(nova))  # remove duplicatas mantendo ordem
        listas_sanas.append(nova)

    return listas_sanas


# ============================================================
# >>> INÍCIO — BLOCO DO PAINEL 6 — MODO 6 ACERTOS (PRÉ-ECO)
# ============================================================

if painel == "🎯 Modo 6 Acertos — Execução":

    st.markdown("## 🎯 Modo 6 Acertos — Execução")

    df = st.session_state.get("historico_df")

    # ------------------------------------------------------------
    # k* (fallback seguro)
    # ------------------------------------------------------------
    _kstar_raw = st.session_state.get("sentinela_kstar")
    k_star = float(_kstar_raw) if isinstance(_kstar_raw, (int, float)) else 0.0

    nr_pct = st.session_state.get("nr_percent")
    divergencia_s6_mc = st.session_state.get("div_s6_mc")
    risco_composto = st.session_state.get("indice_risco")
    ultima_prev = st.session_state.get("ultima_previsao")

    # ------------------------------------------------------------
    # GUARDA — CRITÉRIO MÍNIMO (ORIGINAL PRESERVADO)
    # ------------------------------------------------------------
    pipeline_ok = st.session_state.get("pipeline_flex_ultra_concluido") is True

    turbo_executado_ok = any([
        st.session_state.get("turbo_ultra_executado"),
        st.session_state.get("turbo_executado"),
        st.session_state.get("turbo_ultra_rodou"),
        st.session_state.get("motor_turbo_executado"),
    ])

    if df is None or df.empty or not pipeline_ok or not turbo_executado_ok:
        exibir_bloco_mensagem(
            "Pipeline incompleto",
            "É necessário:\n"
            "- Histórico carregado\n"
            "- Pipeline V14-FLEX ULTRA executado\n"
            "- TURBO++ ULTRA executado ao menos uma vez (bloqueio é válido)\n\n"
            "ℹ️ O TURBO pode se recusar a gerar listas — isso é válido.",
            tipo="warning",
        )
        st.stop()

    # ------------------------------------------------------------
    # AJUSTE DE AMBIENTE (PRÉ-ECO) — ORIGINAL
    # ------------------------------------------------------------
    config = ajustar_ambiente_modo6(
        df=df,
        k_star=k_star,
        nr_pct=nr_pct,
        divergencia_s6_mc=divergencia_s6_mc,
        risco_composto=risco_composto,
        previsibilidade="alta",
    )

    st.caption(config["aviso_curto"] + " | PRÉ-ECO técnico ativo")

    volume = int(config["volume_recomendado"])
    volume = max(1, min(volume, int(config["volume_max"])))


    # ------------------------------------------------------------
    # DETECÇÃO DO FENÔMENO (n + UNIVERSO REAL)
    # ------------------------------------------------------------
    col_pass = [c for c in df.columns if c.startswith("p")]

    universo_tmp = []
    contagens = []

    for _, row in df.iterrows():
        vals = [int(row[c]) for c in col_pass if pd.notna(row[c])]
        if vals:
            contagens.append(len(vals))
            universo_tmp.extend(vals)

    if not contagens or not universo_tmp:
        st.warning("Não foi possível detectar universo válido no histórico.")
        st.stop()

    n_real = int(pd.Series(contagens).mode().iloc[0])
    st.session_state["n_alvo"] = n_real

    universo = sorted({v for v in universo_tmp if v > 0})
    umin, umax = min(universo), max(universo)   # EX: 1–50 (REAL)


    # ------------------------------------------------------------
    # REPRODUTIBILIDADE (ORIGINAL)
    # ------------------------------------------------------------
    seed = abs(hash(f"PC-M6-{len(df)}-{n_real}-{umin}-{umax}")) % (2**32)
    rng = np.random.default_rng(seed)


    # ------------------------------------------------------------
    # MAPA DE ÍNDICES (CORREÇÃO ESTRUTURAL)
    # ------------------------------------------------------------
    universo_idx = list(range(len(universo)))
    valor_por_idx = {i: universo[i] for i in universo_idx}
    idx_por_valor = {v: i for i, v in valor_por_idx.items()}


    def ajustar_para_n(lista):
        out_idx = []

        for v in lista:
            if v in idx_por_valor:
                idx = idx_por_valor[v]
                if idx not in out_idx:
                    out_idx.append(idx)

        while len(out_idx) < n_real:
            cand = rng.choice(universo_idx)
            if cand not in out_idx:
                out_idx.append(cand)

        return out_idx[:n_real]


    # ------------------------------------------------------------
    # BASE ULTRA (ORIGINAL, MAS EM ÍNDICES)
    # ------------------------------------------------------------
    if ultima_prev:
        base_vals = ultima_prev if isinstance(ultima_prev[0], int) else ultima_prev[0]
        base_idx = ajustar_para_n(base_vals)
    else:
        base_idx = rng.choice(universo_idx, size=n_real, replace=False).tolist()


    # ------------------------------------------------------------
    # GERAÇÃO PRÉ-ECO (SEM POSSIBILIDADE DE SAIR DO UNIVERSO)
    # ------------------------------------------------------------
    listas_brutas = []

    for _ in range(volume):
        ruido = rng.integers(-3, 4, size=n_real)  # deslocamento leve
        nova_idx = [
            max(0, min(len(universo_idx) - 1, idx + r))
            for idx, r in zip(base_idx, ruido)
        ]
        nova = [valor_por_idx[i] for i in nova_idx]
        listas_brutas.append(nova)


    # ------------------------------------------------------------
    # 🔒 FILTRO FINAL DE DOMÍNIO (ANTI-RESÍDUO)  ← CORREÇÃO
    # ------------------------------------------------------------
    listas_filtradas = []
    descartadas = 0

    for lista in listas_brutas:
        if all(umin <= int(v) <= umax for v in lista):
            listas_filtradas.append(lista)
        else:
            descartadas += 1

    if descartadas > 0:
        st.warning(
            f"⚠️ {descartadas} lista(s) descartada(s) por violar o domínio "
            f"dos passageiros ({umin}–{umax})."
        )

    listas_brutas = listas_filtradas


    # ------------------------------------------------------------
    # SANIDADE FINAL — SOMENTE ESTRUTURAL (ORIGINAL)
    # ------------------------------------------------------------
    listas_totais = sanidade_final_listas(listas_brutas)

    listas_top10 = listas_totais[:10]

    st.session_state["modo6_listas_totais"] = listas_totais
    st.session_state["modo6_listas_top10"] = listas_top10
    st.session_state["modo6_listas"] = listas_totais

    st.success(
        f"Modo 6 (PRÉ-ECO | n-base={n_real}) — "
        f"{len(listas_totais)} listas totais | "
        f"{len(listas_top10)} priorizadas (Top 10)."
    )

# ============================================================
# <<< FIM — BLOCO DO PAINEL 6 — MODO 6 ACERTOS (PRÉ-ECO)
# ============================================================





# ============================================================
# 🧪 Modo N Experimental (n≠6)
# (LAUDO DE CÓDIGO — FASE 1 / BLOCO 2)
#
# OBJETIVO:
# - Roteamento mínimo + guardas explícitas
# - Avisos claros de EXPERIMENTAL
# - ZERO lógica de geração
#
# BLINDAGEM:
# - NÃO altera Modo 6
# - NÃO altera TURBO
# - NÃO altera ECO/PRÉ-ECO
# - NÃO escreve em session_state (somente leitura)
# ============================================================

elif painel == "🧪 Modo N Experimental (n≠6)":

    st.header("🧪 Modo N Experimental (n≠6)")
    st.warning(
        "EXPERIMENTAL — Este painel é isolado. "
        "Não substitui o Modo 6, não altera TURBO, "
        "não aprende e pode recusar geração."
    )

    # ------------------------------
    # Guardas canônicas (EVIDÊNCIA REAL)
    # ------------------------------
    historico_df = st.session_state.get("historico_df")
    n_alvo = st.session_state.get("n_alvo")
    k_calculado = st.session_state.get("k_calculado") or st.session_state.get("k_star")

    # Evidências indiretas do pipeline (como ele REALMENTE funciona)
    estrada_regime = st.session_state.get("estrada_regime")
    energia_media = st.session_state.get("energia_media")
    clusters_formados = st.session_state.get("clusters_formados")

    # Guarda 1 — histórico
    if historico_df is None or historico_df.empty:
        st.error("Pré-requisito ausente: histórico não carregado.")
        st.stop()

    # Guarda 2 — n_alvo válido e diferente de 6
    try:
        n_int = int(n_alvo)
    except Exception:
        st.error("Pré-requisito ausente: n_alvo inválido.")
        st.stop()

    if n_int == 6:
        st.info("Este painel é exclusivo para n≠6. Para n=6, utilize o Modo 6.")
        st.stop()

    # Guarda 3 — pipeline (por evidência observada)
    if estrada_regime is None and energia_media is None and clusters_formados is None:
        st.error("Pré-requisito ausente: Pipeline V14-FLEX ULTRA não concluído.")
        st.stop()

    # Guarda 4 — sentinelas
    if k_calculado is None:
        st.error("Pré-requisito ausente: Sentinelas (k/k*) não calculadas.")
        st.stop()

    # ------------------------------
    # Estado observado (laudo)
    # ------------------------------
    st.subheader("📋 Estado Observado (Laudo)")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("n do Fenômeno", n_int)
    with col2:
        st.metric("Séries", len(historico_df))
    with col3:
        st.metric("Pipeline", "CONCLUÍDO (por evidência)")
    with col4:
        st.metric(
            "Sentinela",
            f"{k_calculado:.4f}" if isinstance(k_calculado, (int, float)) else str(k_calculado),
        )

    st.markdown("---")

    st.info(
        "Este é o **BLOCO 2 (Laudo de Código)**.\n\n"
        "➡️ Nenhuma lista é gerada aqui.\n"
        "➡️ O próximo bloco ativará a lógica EXPERIMENTAL de geração, "
        "usando U2/U3/U4 como autorizadores."
    )



# ============================================================
# 🧪 Modo N Experimental (n≠6)
# BLOCO 3 — GERAÇÃO MÍNIMA EXPERIMENTAL (n=5)
# ============================================================

    st.markdown("### 🔬 Geração Experimental — n≠6")
    st.caption(
        "Modo EXPERIMENTAL. Geração mínima, consciente e auditável. "
        "Não substitui o Modo 6."
    )

    # ------------------------------------------------------------
    # Fonte canônica do pacote (somente leitura)
    # ------------------------------------------------------------
    listas_base = st.session_state.get("modo6_listas_totais", [])

    # ------------------------------------------------------------
    # Autorizadores (MVP-U2 / U3 / U4)
    # ------------------------------------------------------------
    autorizacao = {
        "orcamento_ok": False,
        "cobertura_ok": False,
        "eficiencia_ok": False,
    }

    # Autorização mínima por orçamento (U2)
    orcamento_manual = st.session_state.get("orcamento_manual_universal")
    if isinstance(orcamento_manual, (int, float)) and orcamento_manual > 0:
        autorizacao["orcamento_ok"] = True

    # Autorização mínima por cobertura (U3)
    if listas_base and len(listas_base) >= 1:
        autorizacao["cobertura_ok"] = True

    # Autorização mínima por eficiência (U4)
    # (critério mínimo: ao menos 1 lista viável)
    if autorizacao["orcamento_ok"] and autorizacao["cobertura_ok"]:
        autorizacao["eficiencia_ok"] = True

    # ------------------------------------------------------------
    # Decisão EXPERIMENTAL (sem fallback)
    # ------------------------------------------------------------
    if not all(autorizacao.values()):
        st.warning(
            "Geração NÃO autorizada pelos MVPs (U2/U3/U4).\n\n"
            "➡️ Resultado válido.\n"
            "➡️ Nenhuma lista foi gerada."
        )
    else:
        # --------------------------------------------------------
        # Geração mínima (1 a 3 listas) — n-base
        # --------------------------------------------------------
        max_listas = min(3, len(listas_base))
        listas_n = [sorted(lst)[:n_int] for lst in listas_base[:max_listas]]

        st.success(f"Geração EXPERIMENTAL autorizada — {len(listas_n)} lista(s).")

        for i, lst in enumerate(listas_n, start=1):
            st.code(f"Lista N{i}: {lst}", language="python")

        # --------------------------------------------------------
        # Mini-laudo automático
        # --------------------------------------------------------
        st.markdown("#### 📄 Mini-Laudo (Automático)")
        st.write(
            {
                "modo": "Modo N Experimental",
                "n": n_int,
                "listas_geradas": len(listas_n),
                "orcamento_manual": orcamento_manual,
                "regime": "OBSERVADO",
                "status": "GERADO" if listas_n else "RECUSADO",
            }
        )



# ============================================================
# 📊 V16 PREMIUM — MVP-U2 | ORÇAMENTO UNIVERSAL (OBSERVACIONAL)
# ============================================================
if painel == "📊 V16 Premium — Orçamento Universal":

    st.title("📊 MVP-U2 — Orçamento Universal (Observacional)")
    st.caption(
        "Observacional • Não gera listas • Não decide\n"
        "Avalia custo real dos pacotes já gerados (Modo 6 / Universal)."
    )

    listas = st.session_state.get("modo6_listas_totais", [])
    n_alvo = st.session_state.get("n_alvo")

    if not listas or n_alvo is None:
        st.warning(
            "Pacote indisponível.\n\n"
            "Execute primeiro:\n"
            "• Pipeline\n"
            "• Modo 6 (Painel 11)"
        )
        st.stop()

    st.markdown("---")

    # --------------------------------------------------------
    # TABELA DE CUSTO UNIVERSAL (CANÔNICA)
    # --------------------------------------------------------
    TABELA_CUSTO = {
        5:  {5: 3,   6: 18,   7: 63,   8: 168,   9: 378,   10: 756},
        6:  {6: 6,   7: 42,   8: 168,  9: 504,   10: 1260, 11: 2772},
        15: {15: 3.5, 16: 56, 17: 476},
    }

    st.markdown("### 📐 Tabela canônica de custo (fixa)")
    st.json(TABELA_CUSTO)

    st.markdown("---")

    # --------------------------------------------------------
    # Entrada de orçamento manual (opcional)
    # --------------------------------------------------------
    orcamento_manual = st.number_input(
        "Orçamento manual (opcional)",
        min_value=0.0,
        value=0.0,
        step=1.0,
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Avaliação das listas
    # --------------------------------------------------------
    linhas = []

    for i, lista in enumerate(listas, start=1):
        tamanho = len(lista)

        custo = None
        if n_alvo in TABELA_CUSTO and tamanho in TABELA_CUSTO[n_alvo]:
            custo = TABELA_CUSTO[n_alvo][tamanho]

        linhas.append({
            "lista_id": i,
            "n_lista": tamanho,
            "custo_estimado": custo,
            "cabe_no_orcamento_manual": (
                custo is not None and orcamento_manual > 0 and custo <= orcamento_manual
            ),
        })

    df_orc = pd.DataFrame(linhas)

    st.markdown("### 📊 Avaliação observacional do pacote")
    st.dataframe(df_orc, use_container_width=True, hide_index=True)

    st.markdown(
        """
🧠 **Leitura correta**
- Custo **None** = combinação não prevista na tabela
- Painel **não filtra**, **não decide**, **não prioriza**
- Serve apenas para **decisão HUMANA**
"""
    )

# ============================================================
# MVP-U3 — COBERTURA UNIVERSAL (OBSERVACIONAL)
# NÃO GERA LISTAS • NÃO DECIDE • NÃO ALTERA MOTOR
# ============================================================
if painel == "🧩 MVP-U3 — Cobertura Universal":

    st.markdown("## 🧩 MVP-U3 — Cobertura Universal (Observacional)")
    st.caption(
        "Avalia cobertura, redundância e custo teórico do pacote ATUAL.\n"
        "Funciona para qualquer n_alvo (5, 6, 15, etc.).\n"
        "❌ Não gera listas • ❌ Não decide • ✅ Apenas mede"
    )

    # ------------------------------------------------------------
    # Recuperação segura do histórico
    # ------------------------------------------------------------
    historico_df = st.session_state.get("historico_df")
    if historico_df is None or historico_df.empty:
        st.warning("Histórico não encontrado. Carregue o histórico antes.")
        st.stop()

    # ------------------------------------------------------------
    # Recuperação do pacote congelado
    # ------------------------------------------------------------
    pacote = (
        st.session_state.get("pacote_listas_atual")
        or st.session_state.get("modo6_listas_totais")
    )

    if not pacote:
        st.warning("Nenhum pacote de listas disponível para avaliação.")
        st.stop()

    # ------------------------------------------------------------
    # Detecção canônica de n_alvo
    # ------------------------------------------------------------
    n_alvo = st.session_state.get("n_alvo")
    if not n_alvo or n_alvo <= 0:
        st.warning("n_alvo não detectado. Execute o carregamento do histórico.")
        st.stop()

    # ------------------------------------------------------------
    # Universo real observado no histórico
    # ------------------------------------------------------------
    col_pass = [c for c in historico_df.columns if c.startswith("p")]
    universo = sorted(
        {
            int(v)
            for _, row in historico_df.iterrows()
            for v in row[col_pass]
            if pd.notna(v) and int(v) > 0
        }
    )

    if not universo:
        st.warning("Universo vazio. Histórico inválido.")
        st.stop()

    # ------------------------------------------------------------
    # Métricas de cobertura
    # ------------------------------------------------------------
    total_listas = len(pacote)

    tamanhos = [len(set(lst)) for lst in pacote]
    validas = [lst for lst in pacote if len(set(lst)) >= n_alvo]

    cobertura_unica = set()
    for lst in validas:
        cobertura_unica.update(lst)

    taxa_validas = len(validas) / total_listas if total_listas else 0.0
    cobertura_pct = (
        len(cobertura_unica) / len(universo) * 100 if universo else 0.0
    )

    # Redundância média
    freq = {}
    for lst in validas:
        for x in lst:
            freq[x] = freq.get(x, 0) + 1

    redundancia_media = (
        sum(freq.values()) / len(freq) if freq else 0.0
    )

    # ------------------------------------------------------------
    # Exibição — Métricas principais
    # ------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Listas totais", total_listas)
    col2.metric("Listas ≥ n_alvo", f"{len(validas)} ({taxa_validas*100:.1f}%)")
    col3.metric("Cobertura do universo", f"{cobertura_pct:.1f}%")
    col4.metric("Redundância média", f"{redundancia_media:.2f}")

    # ------------------------------------------------------------
    # Diagnóstico textual (OBSERVACIONAL)
    # ------------------------------------------------------------
    st.markdown("### 🧠 Leitura observacional")

    if taxa_validas < 0.6:
        st.warning(
            "Poucas listas atingem o tamanho mínimo do fenômeno.\n"
            "Cobertura estrutural fraca."
        )
    elif cobertura_pct < 40:
        st.warning(
            "Cobertura baixa do universo observado.\n"
            "Pacote concentrado demais."
        )
    else:
        st.success(
            "Cobertura estrutural aceitável para o fenômeno atual.\n"
            "Pacote coerente sob critério universal."
        )

    st.info(
        "📌 Este painel NÃO decide execução.\n"
        "Use apenas como régua de cobertura e redundância."
    )

# ============================================================
# <<< FIM — MVP-U3 — COBERTURA UNIVERSAL
# ============================================================

# ============================================================
# MVP-U4 — EFICIÊNCIA MARGINAL POR CUSTO (OBSERVACIONAL)
# NÃO GERA LISTAS • NÃO DECIDE • NÃO ALTERA MOTOR
# ============================================================
if painel == "📈 MVP-U4 — Eficiência Marginal por Custo":

    st.markdown("## 📈 MVP-U4 — Eficiência Marginal por Custo (Observacional)")
    st.caption(
        "Avalia quanto de cobertura adicional é obtida por unidade extra de orçamento.\n"
        "Depende de U2 (Orçamento) e U3 (Cobertura).\n"
        "❌ Não gera listas • ❌ Não decide • ✅ Apenas mede"
    )

    # ------------------------------------------------------------
    # Recuperação do histórico e n_alvo
    # ------------------------------------------------------------
    historico_df = st.session_state.get("historico_df")
    n_alvo = st.session_state.get("n_alvo")

    if historico_df is None or historico_df.empty or not n_alvo:
        st.warning("Histórico ou n_alvo indisponível. Carregue o histórico primeiro.")
        st.stop()

    # ------------------------------------------------------------
    # Recuperação do pacote congelado
    # ------------------------------------------------------------
    pacote = (
        st.session_state.get("pacote_listas_atual")
        or st.session_state.get("modo6_listas_totais")
    )

    if not pacote:
        st.warning("Nenhum pacote disponível para análise.")
        st.stop()

    # ------------------------------------------------------------
    # Universo real observado
    # ------------------------------------------------------------
    col_pass = [c for c in historico_df.columns if c.startswith("p")]
    universo = sorted(
        {
            int(v)
            for _, row in historico_df.iterrows()
            for v in row[col_pass]
            if pd.notna(v) and int(v) > 0
        }
    )

    if not universo:
        st.warning("Universo vazio. Histórico inválido.")
        st.stop()

    universo_size = len(universo)

    # ------------------------------------------------------------
    # Tabela canônica de custo (mesma do MVP-U2)
    # ------------------------------------------------------------
    TABELA_CUSTO = {
        5:  {5: 3,   6: 18,   7: 63,   8: 168,   9: 378,   10: 756},
        6:  {6: 6,   7: 42,   8: 168,  9: 504,   10: 1260, 11: 2772},
        15: {15: 3.5, 16: 56, 17: 476},
    }

    # ------------------------------------------------------------
    # Agrupamento por tamanho de lista (≥ n_alvo)
    # ------------------------------------------------------------
    grupos = {}
    for lst in pacote:
        if len(set(lst)) >= n_alvo:
            k = len(set(lst))
            grupos.setdefault(k, []).append(lst)

    if not grupos:
        st.warning("Nenhuma lista válida (≥ n_alvo) encontrada.")
        st.stop()

    # ------------------------------------------------------------
    # Cálculo de cobertura por grupo
    # ------------------------------------------------------------
    linhas = []

    for tamanho, listas in sorted(grupos.items()):
        cobertura = set()
        for lst in listas:
            cobertura.update(lst)

        cobertura_pct = len(cobertura) / universo_size * 100

        custo = None
        if n_alvo in TABELA_CUSTO and tamanho in TABELA_CUSTO[n_alvo]:
            custo = TABELA_CUSTO[n_alvo][tamanho]

        linhas.append({
            "n_lista": tamanho,
            "cobertura_pct": cobertura_pct,
            "custo": custo,
        })

    df = pd.DataFrame(linhas).sort_values("n_lista").reset_index(drop=True)

    if df.empty:
        st.warning("Não foi possível calcular métricas.")
        st.stop()

    # ------------------------------------------------------------
    # Base = menor tamanho válido
    # ------------------------------------------------------------
    base = df.iloc[0]
    base_cob = base["cobertura_pct"]
    base_custo = base["custo"]

    # ------------------------------------------------------------
    # Eficiência marginal
    # ------------------------------------------------------------
    em_linhas = []
    for _, row in df.iterrows():
        if row["custo"] is None or base_custo is None or row["custo"] == base_custo:
            em = None
            dc = None
            dd = None
        else:
            dc = row["custo"] - base_custo
            dd = row["cobertura_pct"] - base_cob
            em = dd / dc if dc > 0 else None

        em_linhas.append({
            "n_lista": row["n_lista"],
            "cobertura_pct": round(row["cobertura_pct"], 2),
            "custo": row["custo"],
            "Δcobertura": round(dd, 2) if dd is not None else None,
            "Δcusto": dc,
            "eficiencia_marginal": round(em, 4) if em is not None else None,
        })

    df_em = pd.DataFrame(em_linhas)

    # ------------------------------------------------------------
    # Exibição
    # ------------------------------------------------------------
    st.markdown("### 📊 Eficiência marginal por tamanho de lista")
    st.dataframe(df_em, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------
    # Leitura observacional
    # ------------------------------------------------------------
    st.markdown("### 🧠 Leitura observacional")

    valid_em = df_em.dropna(subset=["eficiencia_marginal"])
    if valid_em.empty:
        st.info("Eficiência marginal não disponível para comparação.")
    else:
        melhor = valid_em.sort_values("eficiencia_marginal", ascending=False).iloc[0]
        st.success(
            f"Maior eficiência marginal em n_lista = {int(melhor['n_lista'])} "
            f"(EM = {melhor['eficiencia_marginal']})."
        )

        baixos = valid_em[valid_em["eficiencia_marginal"] < 0.01]
        if not baixos.empty:
            st.warning(
                "Retorno decrescente detectado em alguns tamanhos:\n"
                + ", ".join(str(int(x)) for x in baixos["n_lista"].tolist())
            )

    st.info(
        "📌 Este painel é apenas observacional.\n"
        "Use para decidir até onde vale a pena aumentar o orçamento."
    )

# ============================================================
# <<< FIM — MVP-U4 — EFICIÊNCIA MARGINAL POR CUSTO
# ============================================================


# ============================================================
# Painel 12 — 🧪 Testes de Confiabilidade REAL
# ============================================================
if painel == "🧪 Testes de Confiabilidade REAL":

    st.markdown("## 🧪 Testes de Confiabilidade REAL — V15.7 MAX")

    df = st.session_state.get("historico_df")
    listas_m6 = st.session_state.get("modo6_listas")
    ultima_prev = st.session_state.get("ultima_previsao")

    if df is None or listas_m6 is None or ultima_prev is None:
        exibir_bloco_mensagem(
            "Pré-requisitos não atendidos",
            "Execute o pipeline até o Modo 6 Acertos.",
            tipo="warning",
        )
        st.stop()

    qtd_series = len(df)
    if qtd_series < 15:
        exibir_bloco_mensagem(
            "Histórico insuficiente",
            "São necessárias pelo menos 15 séries para validar a confiabilidade.",
            tipo="warning",
        )
        st.stop()

    st.info("Executando avaliação REAL de confiabilidade...")

    col_pass = [c for c in df.columns if c.startswith("p")]

    # Janela de teste recente
    janela = df[col_pass].iloc[-12:].values

    # ============================================================
    # Medição de acertos reais
    # ============================================================
    def acertos(lista, alvo):
        return len(set(lista) & set(alvo))

    acertos_nucleo = []
    acertos_coberturas = []

    for alvo in janela:
        # núcleo
        ac_nuc = acertos(ultima_prev, alvo)
        acertos_nucleo.append(ac_nuc)

        # coberturas
        max_cov = 0
        for lst in listas_m6:
            ac_lst = acertos(lst, alvo)
            if ac_lst > max_cov:
                max_cov = ac_lst
        acertos_coberturas.append(max_cov)

    # Médias
    media_nucleo = float(np.mean(acertos_nucleo))
    media_cob = float(np.mean(acertos_coberturas))

    # ============================================================
    # Diagnóstico
    # ============================================================
    corpo = (
        f"- Janela avaliada: **12 séries recentes**\n"
        f"- Média de acertos do Núcleo: **{media_nucleo:.2f}**\n"
        f"- Média de acertos das Coberturas: **{media_cob:.2f}**\n"
        "\n"
        "Coberturas devem superar o núcleo em ambientes turbulentos.\n"
        "Se o núcleo supera as coberturas, o ambiente está mais limpo."
    )

    exibir_bloco_mensagem(
        "Resumo da Confiabilidade REAL",
        corpo,
        tipo="info",
    )

    st.success("Teste de Confiabilidade REAL concluído com sucesso!")

# ============================================================
# BLOCO — SANIDADE FINAL DAS LISTAS DE PREVISÃO
# (Elimina permutações, duplicatas por conjunto
#  E listas com números repetidos internos)
# Válido para V15.7 MAX e V16 Premium
# ============================================================

def sanidade_final_listas(listas):
    """
    Sanidade final das listas de previsão.
    Regras:
    - Remove listas com números repetidos internamente
    - Remove permutações (ordem diferente, mesmos números)
    - Remove duplicatas exatas
    - Garante apenas listas válidas com 6 números distintos
    """

    listas_saneadas = []
    vistos = set()

    for lista in listas:
        try:
            nums = [int(x) for x in lista]
        except Exception:
            continue

        # 🔒 REGRA CRÍTICA — exatamente 6 números distintos
        if len(nums) != 6:
            continue

        if len(set(nums)) != 6:
            # Exemplo eliminado: [11, 12, 32, 32, 37, 42]
            continue

        # Normaliza ordem para detectar permutações
        chave = tuple(sorted(nums))

        if chave in vistos:
            continue

        vistos.add(chave)
        listas_saneadas.append(nums)

    return listas_saneadas


# ============================================================
# APLICAÇÃO AUTOMÁTICA DA SANIDADE (SE LISTAS EXISTIREM)
# ============================================================

# Sanear listas do Modo 6 (V15.7)
if "modo6_listas" in st.session_state:
    st.session_state["modo6_listas"] = sanidade_final_listas(
        st.session_state.get("modo6_listas", []),
    )

# Sanear Execução V16 (se existir)
if "v16_execucao" in st.session_state:
    exec_v16 = st.session_state.get("v16_execucao", {})

    for chave in ["C2", "C3", "todas_listas"]:
        if chave in exec_v16:
            exec_v16[chave] = sanidade_final_listas(
                exec_v16.get(chave, []),
            )

    st.session_state["v16_execucao"] = exec_v16

# ============================================================
# PARTE 6/8 — FIM
# ============================================================



# ============================================================
# PARTE 7/8 — INÍCIO
# ============================================================

# ============================================================
# Painel — 🧪 Replay Curto — Expectativa 1–3 Séries (V16)
# Diagnóstico apenas | NÃO gera previsões | NÃO altera fluxo
# ============================================================
if painel == "🧪 Replay Curto — Expectativa 1–3 Séries":

    st.markdown("## 🧪 Replay Curto — Expectativa 1–3 Séries (Diagnóstico)")
    st.caption(
        "Validação no passado da expectativa de curto prazo (1–3 séries). "
        "Este painel **não prevê números** e **não altera decisões**."
    )

    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")

    if df is None or matriz_norm is None:
        exibir_bloco_mensagem(
            "Pipeline incompleto",
            "Execute **Carregar Histórico** e **Pipeline V14-FLEX ULTRA**.",
            tipo="warning",
        )
        st.stop()

    # -------------------------------
    # Parâmetros FIXOS (sem bifurcação)
    # -------------------------------
    JANELA_REPLAY = 80       # pontos do passado
    HORIZONTE = 3            # 1–3 séries
    LIMIAR_NR = 0.02         # queda mínima de NR% para considerar melhora
    LIMIAR_DIV = 0.50        # queda mínima de divergência para considerar melhora

    n = len(df)
    if n < JANELA_REPLAY + HORIZONTE + 5:
        exibir_bloco_mensagem(
            "Histórico insuficiente",
            "É necessário mais histórico para o replay curto.",
            tipo="warning",
        )
        st.stop()

    # -------------------------------
    # Helpers locais (diagnóstico)
    # -------------------------------
    col_pass = [c for c in df.columns if c.startswith("p")]

    def calc_nr_local(matriz):
        # NR% aproximado (mesma lógica do painel, versão local)
        variancias = np.var(matriz, axis=1)
        ruido_A = float(np.mean(variancias))
        saltos = []
        for i in range(1, len(matriz)):
            saltos.append(np.linalg.norm(matriz[i] - matriz[i - 1]))
        ruido_B = float(np.mean(saltos)) if saltos else 0.0
        return (0.55 * min(1.0, ruido_A / 0.08) + 0.45 * min(1.0, ruido_B / 1.20))

    def calc_div_local(base, candidatos):
        return float(np.linalg.norm(np.mean(candidatos, axis=0) - base))

    def estado_sinal(nr_deriv, div_deriv, vel):
        # 🟢 melhora curta
        if nr_deriv < -LIMIAR_NR and div_deriv < -LIMIAR_DIV and vel < 0.75:
            return "🟢 Melhora curta"
        # 🔴 continuidade ruim
        if nr_deriv > 0 or div_deriv > 0 or vel >= 0.80:
            return "🔴 Continuidade ruim"
        # 🟡 transição
        return "🟡 Respiração / Transição"

    # -------------------------------
    # Replay
    # -------------------------------
    resultados = []
    base_ini = n - JANELA_REPLAY - HORIZONTE

    for i in range(base_ini, n - HORIZONTE):
        # Janela até o ponto i
        matriz_i = matriz_norm[: i + 1]
        nr_i = calc_nr_local(matriz_i)

        # Divergência local (proxy simples)
        base = matriz_i[-1]
        candidatos = matriz_i[-10:] if len(matriz_i) >= 10 else matriz_i
        div_i = calc_div_local(base, candidatos)

        # Velocidade (proxy simples)
        vel = float(np.mean(np.std(matriz_i[-5:], axis=1)))

        # Próximo trecho (1–3)
        matriz_f = matriz_norm[: i + 1 + HORIZONTE]
        nr_f = calc_nr_local(matriz_f)
        base_f = matriz_f[-1]
        candidatos_f = matriz_f[-10:] if len(matriz_f) >= 10 else matriz_f
        div_f = calc_div_local(base_f, candidatos_f)

        nr_deriv = nr_f - nr_i
        div_deriv = div_f - div_i

        estado = estado_sinal(nr_deriv, div_deriv, vel)

        melhora_real = (nr_deriv < -LIMIAR_NR) or (div_deriv < -LIMIAR_DIV)

        resultados.append({
            "estado": estado,
            "melhora_real": melhora_real
        })

    # -------------------------------
    # Consolidação
    # -------------------------------
    df_res = pd.DataFrame(resultados)
    resumo = (
        df_res.groupby("estado")["melhora_real"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={
            "count": "Ocorrências",
            "mean": "Taxa de Melhora"
        })
    )
    resumo["Taxa de Melhora"] = (resumo["Taxa de Melhora"] * 100).round(1)

    st.markdown("### 📊 Resultado do Replay Curto (passado)")
    st.dataframe(resumo, use_container_width=True)

    st.info(
        "Este painel valida **se o estado 🟢 precede melhora real** no curto prazo "
        "(1–3 séries) **mais vezes que o acaso**. "
        "Ele **não prevê o futuro**, apenas qualifica a expectativa."
    )

# ============================================================
# CAMADA B1 — PRIORIZAÇÃO OBSERVACIONAL DE LISTAS (V16)
# NÃO gera | NÃO decide | NÃO altera motores
# ============================================================

def v16_priorizar_listas_por_contexto(listas):
    """
    Ordena listas existentes usando ECO + Estado.
    Apenas PRIORIZA — não remove, não cria, não decide.
    """

    diag = st.session_state.get("diagnostico_eco_estado_v16", {})
    eco_acion = diag.get("eco_acionabilidade", "não_acionável")
    estado = diag.get("estado", "indefinido")

    if not listas or not isinstance(listas, list):
        return listas

    def score_lista(lst):
        score = 0

        # Preferência leve quando ambiente é favorável
        if eco_acion == "favorável":
            score += 2
        elif eco_acion == "cautela":
            score += 1

        # Penalização leve em movimento brusco
        if estado == "movimento_brusco":
            score -= 1

        # Listas mais compactas tendem a ser mais estáveis
        try:
            if len(set(lst)) <= len(lst):
                score += 1
        except Exception:
            pass

        return score

    try:
        listas_ordenadas = sorted(listas, key=score_lista, reverse=True)
        return listas_ordenadas
    except Exception:
        return listas


# ============================================================
# >>> PAINEL X — 🧠 Memória Operacional (Observacional)
# ============================================================
if painel == "🧠 Memória Operacional (Observacional)":

    st.markdown("## 🧠 Memória Operacional — Observacional (Passiva)")

    st.caption(
        "Este painel é **estritamente observacional**.\n\n"
        "📌 NÃO decide\n"
        "📌 NÃO bloqueia\n"
        "📌 NÃO altera geração\n"
        "📌 NÃO interfere em volumes\n\n"
        "Serve apenas para **registro e leitura disciplinada** do que já ocorreu."
    )

    # ------------------------------------------------------------
    # Inicialização segura da Memória Operacional
    # ------------------------------------------------------------
    if "memoria_operacional" not in st.session_state:
        st.session_state["memoria_operacional"] = []

    mo = st.session_state.get("memoria_operacional", [])

    # ------------------------------------------------------------
    # Contexto atual (somente leitura)
    # ------------------------------------------------------------
    fenomeno_id = st.session_state.get("fenomeno_id", "N/D")
    alvo_atual = st.session_state.get("n_alvo", "N/D")
    eco_status = st.session_state.get("eco_status", "N/D")
    estado_status = st.session_state.get("estado_atual", "N/D")

    exibir_bloco_mensagem(
        "🧭 Contexto Atual (Leitura)",
        f"- Fenômeno ID: **{fenomeno_id}**\n"
        f"- Alvo atual: **{alvo_atual}**\n"
        f"- ECO: **{eco_status}**\n"
        f"- Estado: **{estado_status}**",
        tipo="info",
    )

    # ------------------------------------------------------------
    # Registro manual da tentativa (PASSIVO)
    # ------------------------------------------------------------
    st.markdown("### 📝 Registrar tentativa (Manual | Passivo)")

    st.caption(
        "O registro é **consciente e manual**.\n\n"
        "📌 Não ativa modo algum\n"
        "📌 Não altera comportamento do sistema\n"
        "📌 Não dispara nenhuma decisão automática"
    )

    with st.form("form_registro_memoria_operacional"):
        descricao = st.text_input(
            "Descrição curta da tentativa (ex.: geração normal, pós-TURBO, leitura prata)",
            value="",
        )

        registrar = st.form_submit_button("Registrar tentativa na Memória Operacional")

    if registrar:
        registro = {
            "fenomeno_id": fenomeno_id,
            "alvo": alvo_atual,
            "eco": eco_status,
            "estado": estado_status,
            "descricao": descricao or "N/D",
        }

        mo.append(registro)
        st.session_state["memoria_operacional"] = mo

        st.success("Registro adicionado à Memória Operacional.")

    # ------------------------------------------------------------
    # Visualização da Memória Operacional
    # ------------------------------------------------------------
    st.markdown("### 📚 Registros na Memória Operacional")

    if not mo:
        st.info("Nenhum registro na Memória Operacional até o momento.")
    else:
        for i, reg in enumerate(mo, 1):
            st.markdown(
                f"**{i:02d})** "
                f"Fenômeno: `{reg.get('fenomeno_id')}` | "
                f"Alvo: `{reg.get('alvo')}` | "
                f"ECO: `{reg.get('eco')}` | "
                f"Estado: `{reg.get('estado')}`\n\n"
                f"↳ {reg.get('descricao')}"
            )

    st.caption(
        "📌 A Memória Operacional é **volátil nesta fase**.\n"
        "📌 Nenhuma lógica automática depende dela.\n"
        "📌 Pode ser removida sem impacto no sistema."
    )

# ============================================================
# <<< FIM — PAINEL X — 🧠 Memória Operacional
# ============================================================

# ============================================================
# >>> PAINEL Y — 🧠 Memória Operacional — Registro Semi-Automático
# ============================================================
if painel == "🧠 Memória Operacional — Registro Semi-Automático":

    st.markdown("## 🧠 Memória Operacional — Registro Semi-Automático (Passivo)")

    st.caption(
        "Este painel **NÃO decide** e **NÃO bloqueia**.\n\n"
        "Ele apenas **sugere um registro** com base no estado já calculado.\n"
        "📌 O operador **confirma explicitamente**.\n"
        "📌 Nenhuma lógica do sistema é alterada."
    )

    # ------------------------------------------------------------
    # Inicialização segura da MO (compartilhada)
    # ------------------------------------------------------------
    if "memoria_operacional" not in st.session_state:
        st.session_state["memoria_operacional"] = []

    mo = st.session_state.get("memoria_operacional", [])

    # ------------------------------------------------------------
    # Captura PASSIVA do estado atual (somente leitura)
    # ------------------------------------------------------------
    fenomeno_id = st.session_state.get("fenomeno_id", "N/D")
    alvo_atual = st.session_state.get("n_alvo", "N/D")

    eco_status = st.session_state.get("eco_status", "N/D")
    estado_status = st.session_state.get("estado_atual", "N/D")

    pipeline_status = (
        "CONCLUÍDO" if st.session_state.get("pipeline_flex_ultra_concluido") else "N/D"
    )

    qtd_listas_m6 = len(
        st.session_state.get("modo6_listas_totais")
        or st.session_state.get("modo6_listas")
        or []
    )

    qtd_listas_ultra = len(st.session_state.get("turbo_ultra_listas_leves") or [])

    exibir_bloco_mensagem(
        "📋 Estado Capturado (Leitura)",
        f"- Fenômeno ID: **{fenomeno_id}**\n"
        f"- Alvo: **{alvo_atual}**\n"
        f"- ECO: **{eco_status}**\n"
        f"- Estado: **{estado_status}**\n"
        f"- Pipeline FLEX ULTRA: **{pipeline_status}**\n"
        f"- Listas Modo 6: **{qtd_listas_m6}**\n"
        f"- Listas TURBO ULTRA: **{qtd_listas_ultra}**",
        tipo="info",
    )

    # ------------------------------------------------------------
    # Sugestão de registro (NÃO automática)
    # ------------------------------------------------------------
    st.markdown("### 📝 Sugestão de Registro (Confirmação Manual)")

    sugestao = (
        f"Geração executada | ECO={eco_status} | "
        f"Estado={estado_status} | "
        f"Listas(M6={qtd_listas_m6}, ULTRA={qtd_listas_ultra})"
    )

    with st.form("form_registro_semi_automatico"):
        descricao = st.text_area(
            "Descrição do registro (editável antes de confirmar):",
            value=sugestao,
            height=100,
        )

        confirmar = st.form_submit_button(
            "Confirmar registro na Memória Operacional"
        )

    if confirmar:
        registro = {
            "fenomeno_id": fenomeno_id,
            "alvo": alvo_atual,
            "eco": eco_status,
            "estado": estado_status,
            "descricao": descricao or "N/D",
        }

        mo.append(registro)
        st.session_state["memoria_operacional"] = mo

        st.success("Registro semi-automático adicionado à Memória Operacional.")

    # ------------------------------------------------------------
    # Visualização rápida (últimos registros)
    # ------------------------------------------------------------
    st.markdown("### 📚 Últimos registros (leitura rápida)")

    if not mo:
        st.info("Nenhum registro ainda.")
    else:
        for i, reg in enumerate(mo[-5:], 1):
            st.markdown(
                f"**{i:02d})** "
                f"Fenômeno `{reg.get('fenomeno_id')}` | "
                f"Alvo `{reg.get('alvo')}` | "
                f"ECO `{reg.get('eco')}` | "
                f"Estado `{reg.get('estado')}`\n\n"
                f"↳ {reg.get('descricao')}"
            )

    st.caption(
        "📌 Registro **semi-automático** = estado sugerido + confirmação humana.\n"
        "📌 Nenhuma decisão é tomada pelo sistema."
    )

# ============================================================
# <<< FIM — PAINEL Y — 🧠 Memória Operacional — Registro Semi-Automático
# ============================================================



# ============================================================
# >>> PAINEL 13 — 📘 Relatório Final — V15.7 MAX (Premium)
# ============================================================
if painel == "📘 Relatório Final":

    st.markdown("## 📘 Relatório Final — V15.7 MAX — V16 Premium Profundo")

    # ------------------------------------------------------------
    # 🧲 BLOCO 0 — SUGADOR DE ESTADO CONSOLIDADO
    # ------------------------------------------------------------
    historico_df = st.session_state.get("historico_df")
    n_alvo = st.session_state.get("n_alvo")

    pipeline_status = st.session_state.get("pipeline_flex_ultra_concluido")
    ultima_prev = st.session_state.get("ultima_previsao")

    listas_m6_totais = (
        st.session_state.get("modo6_listas_totais")
        or st.session_state.get("modo6_listas")
        or []
    )

    listas_ultra = st.session_state.get("turbo_ultra_listas_leves") or []

    # Validação mínima
    if not listas_m6_totais:
        exibir_bloco_mensagem(
            "Sem pacote do Modo 6",
            "Execute o painel **🎯 Modo 6 Acertos — Execução** antes.",
            tipo="warning",
        )
        st.stop()

    # ------------------------------------------------------------
    # Estado consolidado
    # ------------------------------------------------------------
    linhas = []

    if historico_df is not None:
        linhas.append(f"- Séries carregadas: **{len(historico_df)}**")

    if n_alvo is not None:
        linhas.append(f"- Passageiros por carro (n): **{n_alvo}**")

    if pipeline_status is True:
        linhas.append("- Pipeline FLEX ULTRA: ✅ **CONCLUÍDO**")

    exibir_bloco_mensagem(
        "🧲 Estado Consolidado da Rodada",
        "\n".join(linhas),
        tipo="info",
    )

    # ============================================================
    # 🧠 DIAGNÓSTICO CONSOLIDADO DA RODADA (NOVO — ADITIVO)
    # ============================================================
    eco_consolidado = st.session_state.get("eco_status", "DESCONHECIDO")
    estado_consolidado = st.session_state.get("estado_atual", "DESCONHECIDO")

    st.markdown("### 🧠 Diagnóstico Consolidado da Rodada")

    st.info(
        f"**ECO:** {eco_consolidado}\n\n"
        f"**Estado do alvo:** {estado_consolidado}"
    )

    st.caption(
        "Leitura consolidada do sistema nesta rodada.\n"
        "Não gera decisão automática."
    )

    # ------------------------------------------------------------
    # Núcleo TURBO (se existir)
    # ------------------------------------------------------------
    st.markdown("### 🔮 Previsão Principal (Núcleo — TURBO++ ULTRA)")

    if ultima_prev:
        st.success(formatar_lista_passageiros(ultima_prev))
    else:
        st.info(
            "Nenhuma previsão TURBO disponível nesta rodada "
            "(isso é válido em regime estável)."
        )

    # ------------------------------------------------------------
    # 🛡️ Pacote Prioritário — Top 10 (Modo 6)
    # ------------------------------------------------------------
    st.markdown("### 🛡️ Pacote Prioritário (Top 10) — Modo 6")

    top10 = listas_m6_totais[:10]
    for i, lst in enumerate(top10, 1):
        st.markdown(f"**{i:02d})** {formatar_lista_passageiros(lst)}")

    # ------------------------------------------------------------
    # 📊 EIXO 1 — CONTRIBUIÇÃO DE PASSAGEIROS (OBSERVACIONAL)
    # ------------------------------------------------------------
    try:
        listas_pacote_eixo1 = listas_m6_totais[:]

        historico_label = (
            f"C1 → C{len(historico_df)}"
            if historico_df is not None
            else "Histórico indefinido"
        )

        eixo1_resultado = calcular_eixo1_contribuicao(
            listas_pacote=listas_pacote_eixo1,
            historico_label=historico_label,
            modo_geracao="Modo 6",
            n_base=n_alvo or 6,
            eco_status=st.session_state.get("eco_status", "DESCONHECIDO"),
            estado_status=st.session_state.get("estado_atual", "DESCONHECIDO"),
        )
    except Exception:
        eixo1_resultado = None

    if eixo1_resultado:
        st.markdown("### 📊 Eixo 1 — Contribuição de Passageiros (Observacional)")

        st.write(
            f"**Núcleo local detectado:** "
            f"{'SIM' if eixo1_resultado['nucleo']['detectado'] else 'NÃO'} "
            f"({eixo1_resultado['nucleo']['tipo']})"
        )

        st.write(
            "**Estruturais do pacote:** "
            + (
                ", ".join(map(str, eixo1_resultado["papeis"]["estruturais"]))
                if eixo1_resultado["papeis"]["estruturais"]
                else "—"
            )
        )

        st.write(
            "**Contribuintes:** "
            + (
                ", ".join(map(str, eixo1_resultado["papeis"]["contribuintes"]))
                if eixo1_resultado["papeis"]["contribuintes"]
                else "—"
            )
        )

        st.write(
            "**Leitura sintética:** "
            + " ".join(eixo1_resultado["leitura_sintetica"])
        )

        st.caption(eixo1_resultado["trava"])

    # ------------------------------------------------------------
    # 📦 Pacote Operacional TOTAL (Modo 6 + TURBO ULTRA)
    # ------------------------------------------------------------
    pacote_operacional = listas_m6_totais.copy()

    for lst in listas_ultra:
        if lst not in pacote_operacional:
            pacote_operacional.append(lst)

    try:
        pacote_operacional = v16_priorizar_listas_por_contexto(pacote_operacional)
    except Exception:
        pass

    total_listas = len(pacote_operacional)

    # ------------------------------------------------------------
    # 🔥 MANDAR BALA — POSTURA OPERACIONAL
    # ------------------------------------------------------------
    st.markdown("### 🔥 Mandar Bala — Postura Operacional (Ação Consciente)")

    qtd_bala = st.slider(
        "Quantas listas você quer levar para a ação nesta rodada?",
        min_value=1,
        max_value=total_listas,
        value=min(10, total_listas),
        step=1,
        key="slider_mandar_bala_restaurado",
    )

    for i, lst in enumerate(pacote_operacional[:qtd_bala], 1):
        st.markdown(f"**🔥 {i:02d})** {formatar_lista_passageiros(lst)}")

    exibir_bloco_mensagem(
        "🧩 Fechamento Operacional",
        f"- Listas disponíveis: **{total_listas}**\n"
        f"- Listas levadas para ação: **{qtd_bala}**\n\n"
        "📌 O sistema **não decide**. O operador **assume a postura**.",
        tipo="success",
    )

    # ============================================================
    # 🧠 RF-GOV — GOVERNANÇA INFORMATIVA (AVISOS | SEM EFEITO)
    # ============================================================
    try:
        st.markdown("### 🧠 RF-GOV — Governança Informativa")

        fenomeno_id = st.session_state.get("fenomeno_id", "N/D")
        alvo_atual = st.session_state.get("n_alvo", "N/D")

        eco_status = st.session_state.get("eco_status", "N/D")
        estado_status = st.session_state.get("estado_atual", "N/D")

        mo = st.session_state.get("memoria_operacional", [])
        tentativas_mesmo_alvo = [r for r in mo if r.get("alvo") == alvo_atual]

        avisos = []

        if len(tentativas_mesmo_alvo) >= 2:
            avisos.append(
                "⚠️ Múltiplas tentativas recentes para o mesmo alvo registradas."
            )

        if eco_status in ("RUIM", "DESCONHECIDO"):
            avisos.append("ℹ️ ECO desfavorável ou indefinido.")

        if estado_status in ("RÁPIDO", "INSTÁVEL"):
            avisos.append("ℹ️ Estado do alvo indica instabilidade.")

        st.info(
            f"**Fenômeno ID:** {fenomeno_id}\n\n"
            f"**Alvo:** {alvo_atual}\n\n"
            f"**ECO:** {eco_status}\n"
            f"**Estado:** {estado_status}"
        )

        for a in avisos:
            st.warning(a)

        if not avisos:
            st.success("Nenhum alerta relevante de governança nesta rodada.")

    except Exception:
        st.caption("RF-GOV indisponível nesta execução.")

    st.success("Relatório Final gerado com sucesso!")

# ============================================================
# <<< FIM — PAINEL 13 — 📘 Relatório Final
# ============================================================












# ============================================================
# Painel — ⏱️ DURAÇÃO DA JANELA — ANÁLISE HISTÓRICA (V16)
# Diagnóstico PURO | Mede quantas séries janelas favoráveis duraram
# NÃO prevê | NÃO decide | NÃO altera motores
# ============================================================

# ============================================================
# Painel — 🔍 Cruzamento Histórico do k (Observacional)
# V16 | LEITURA PURA | NÃO DECIDE | NÃO ALTERA MOTORES
# ============================================================

if painel == "🔍 Cruzamento Histórico do k":

    st.markdown("## 🔍 Cruzamento Histórico do k")
    st.caption(
        "Leitura observacional do histórico. "
        "Este painel NÃO interfere em decisões, volumes ou modos."
    )

    eventos = st.session_state.get("eventos_k_historico", [])

    if not eventos:
        exibir_bloco_mensagem(
            "Nenhum evento k encontrado",
            "Carregue o histórico para analisar os eventos k.",
            tipo="warning",
        )
        st.stop()

    df_k = pd.DataFrame(eventos)

    # ============================================================
    # FILTROS SIMPLES (OBSERVACIONAIS)
    # ============================================================
    st.markdown("### 🎛️ Filtros Observacionais")

    col1, col2, col3 = st.columns(3)

    with col1:
        filtro_estado = st.multiselect(
            "Estado do alvo",
            options=sorted(df_k["estado_alvo"].dropna().unique().tolist()),
            default=None,
        )

    with col2:
        filtro_pre_eco = st.selectbox(
            "PRÉ-ECO",
            options=["Todos", "Sim", "Não"],
            index=0,
        )

    with col3:
        filtro_eco = st.selectbox(
            "ECO",
            options=["Todos", "Sim", "Não"],
            index=0,
        )

    df_f = df_k.copy()

    if filtro_estado:
        df_f = df_f[df_f["estado_alvo"].isin(filtro_estado)]

    if filtro_pre_eco != "Todos":
        df_f = df_f[df_f["pre_eco"] == (filtro_pre_eco == "Sim")]

    if filtro_eco != "Todos":
        df_f = df_f[df_f["eco"] == (filtro_eco == "Sim")]

    # ============================================================
    # MÉTRICAS RESUMIDAS
    # ============================================================
    st.markdown("### 📊 Resumo Estatístico")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Eventos k", len(df_f))

    with col2:
        st.metric(
            "Δ médio entre ks",
            round(df_f["delta_series"].dropna().mean(), 2)
            if "delta_series" in df_f else "—",
        )

    with col3:
        st.metric(
            "k médio",
            round(df_f["k_valor"].mean(), 2)
            if "k_valor" in df_f else "—",
        )

    with col4:
        st.metric(
            "Máx k observado",
            int(df_f["k_valor"].max())
            if "k_valor" in df_f else "—",
        )

    # ============================================================
    # TABELA FINAL (LEITURA CRUA)
    # ============================================================
    st.markdown("### 📋 Eventos k — Histórico")

    st.dataframe(
        df_f[
            [
                "serie_id",
                "k_valor",
                "delta_series",
                "estado_alvo",
                "k_star",
                "nr_percent",
                "div_s6_mc",
                "pre_eco",
                "eco",
            ]
        ].sort_values("serie_id"),
        use_container_width=True,
    )

# ============================================================
# FIM — Painel Cruzamento Histórico do k
# ============================================================


if painel == "⏱️ Duração da Janela — Análise Histórica":

    st.markdown("## ⏱️ Duração da Janela — Análise Histórica")

    st.info(
        "Este painel mede, **no passado**, quantas séries consecutivas "
        "as janelas favoráveis **REALMENTE duraram**, após serem confirmadas.\n\n"
        "📌 Definição usada:\n"
        "- Abertura: melhora conjunta (NR%, divergência, k*, desempenho real)\n"
        "- Fechamento: perda clara dessa coerência\n\n"
        "⚠️ Este painel NÃO prevê entrada de janela."
    )

    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")

    if df is None or matriz_norm is None:
        exibir_bloco_mensagem(
            "Pipeline incompleto",
            "Execute **Carregar Histórico** e **Pipeline V14-FLEX ULTRA**.",
            tipo="warning",
        )
        st.stop()

    # ------------------------------------------------------------
    # Parâmetros FIXOS (diagnóstico histórico)
    # ------------------------------------------------------------
    JANELA_ANALISE = 200
    LIMIAR_NR_QUEDA = 0.02
    LIMIAR_DIV_QUEDA = 0.50

    col_pass = [c for c in df.columns if c.startswith("p")]

    # Helpers locais (réplicas leves, sem tocar no motor)
    def _nr_local(m):
        variancias = np.var(m, axis=1)
        ruido_A = float(np.mean(variancias))
        saltos = [
            np.linalg.norm(m[i] - m[i - 1]) for i in range(1, len(m))
        ]
        ruido_B = float(np.mean(saltos)) if saltos else 0.0
        return 0.55 * min(1.0, ruido_A / 0.08) + 0.45 * min(1.0, ruido_B / 1.20)

    def _div_local(m):
        base = m[-1]
        candidatos = m[-10:] if len(m) >= 10 else m
        return float(np.linalg.norm(np.mean(candidatos, axis=0) - base))

    resultados = []
    n = len(matriz_norm)

    for i in range(max(30, n - JANELA_ANALISE), n - 3):
        m_i = matriz_norm[: i + 1]
        m_f = matriz_norm[: i + 4]

        nr_i = _nr_local(m_i)
        nr_f = _nr_local(m_f)
        div_i = _div_local(m_i)
        div_f = _div_local(m_f)

        abriu = (nr_f - nr_i) < -LIMIAR_NR_QUEDA and (div_f - div_i) < -LIMIAR_DIV_QUEDA

        if abriu:
            duracao = 1
            for j in range(i + 1, n - 1):
                m_j = matriz_norm[: j + 1]
                if _nr_local(m_j) <= nr_f and _div_local(m_j) <= div_f:
                    duracao += 1
                else:
                    break

            resultados.append(duracao)

    if not resultados:
        st.warning("Nenhuma janela favorável clara detectada no período analisado.")
        st.stop()

    df_res = pd.DataFrame({"Duração (séries)": resultados})

    st.markdown("### 📊 Distribuição Histórica da Duração das Janelas")
    st.dataframe(df_res.describe(), use_container_width=True)

    st.info(
        f"📌 Total de janelas detectadas: **{len(resultados)}**\n\n"
        "Este painel responde:\n"
        "👉 *Quando a janela abre, ela costuma durar quantas séries?*\n\n"
        "Use isso para **decidir até quando mandar bala**."
    )

# ============================================================
# V16 — CAMADA D
# Estado do Alvo · Expectativa · Volume × Confiabilidade
# (FIX: usa divergência correta do Monitor de Risco)
# ============================================================

def _v16_get_nr_div_risco():
    """
    Leitura segura e compatível com o app:
    - NR% vem do Ruído Condicional (nr_percent) OU do Monitor (diagnostico_risco.nr_percent)
    - Divergência vem do Monitor (diagnostico_risco.divergencia) OU fallback (div_s6_mc)
    - Risco vem do Monitor (diagnostico_risco.indice_risco)
    """
    risco_pack = st.session_state.get("diagnostico_risco") or {}

    nr = st.session_state.get("nr_percent")
    if nr is None:
        nr = risco_pack.get("nr_percent")

    # ⚠️ FIX PRINCIPAL: no seu app a divergência oficial está aqui:
    div = risco_pack.get("divergencia")
    if div is None:
        # fallback legado (se existir em alguma variação do app)
        div = st.session_state.get("div_s6_mc")

    indice_risco = risco_pack.get("indice_risco")

    return nr, div, indice_risco


def v16_registrar_estado_alvo():
    """
    Classifica o estado do alvo com base em:
    - NR%
    - Divergência S6 vs MC
    - Índice de risco (composto)
    """
    nr, div, risco = _v16_get_nr_div_risco()

    if nr is None or div is None or risco is None:
        estado = {
            "tipo": "indefinido",
            "velocidade": "indefinida",
            "comentario": "Histórico/monitor insuficiente para classificar o alvo (rode Monitor de Risco e Ruído Condicional).",
        }
        st.session_state["estado_alvo_v16"] = estado
        return estado

    # velocidade ∈ [~0, ~1+] (heurística)
    velocidade = round((float(nr) / 100.0 + float(div) / 15.0 + float(risco)) / 3.0, 3)

    if velocidade < 0.30:
        tipo = "alvo_parado"
        comentario = "🎯 Alvo praticamente parado — oportunidade rara. Volume alto recomendado."
    elif velocidade < 0.55:
        tipo = "movimento_lento"
        comentario = "🎯 Alvo em movimento lento — alternar rajadas e coberturas."
    elif velocidade < 0.80:
        tipo = "movimento_rapido"
        comentario = "⚠️ Alvo em movimento rápido — reduzir agressividade."
    else:
        tipo = "disparado"
        comentario = "🚨 Alvo disparado — ambiente hostil. Operar apenas de forma respiratória."

    estado = {
        "tipo": tipo,
        "velocidade": velocidade,
        "comentario": comentario,
    }

    st.session_state["estado_alvo_v16"] = estado
    return estado


def v16_registrar_expectativa():
    """
    Estima expectativa de curto prazo (1–3 séries)
    com base em microjanelas, ruído e divergência.
    """
    micro = st.session_state.get("v16_microdiag") or {}
    nr, div, _ = _v16_get_nr_div_risco()

    if not micro or nr is None or div is None:
        expectativa = {
            "previsibilidade": "indefinida",
            "erro_esperado": "indefinido",
            "chance_janela_ouro": "baixa",
            "comentario": "Expectativa indisponível (rode Microjanelas V16 e garanta NR/divergência).",
        }
        st.session_state["expectativa_v16"] = expectativa
        return expectativa

    score = float(micro.get("score_melhor", 0.0) or 0.0)
    janela_ouro = bool(micro.get("janela_ouro", False))

    if janela_ouro and score >= 0.80 and float(nr) < 40.0 and float(div) < 5.0:
        expectativa = {
            "previsibilidade": "alta",
            "erro_esperado": "baixo",
            "chance_janela_ouro": "alta",
            "comentario": "🟢 Forte expectativa positiva nas próximas 1–3 séries.",
        }
    elif score >= 0.50 and float(nr) < 60.0:
        expectativa = {
            "previsibilidade": "moderada",
            "erro_esperado": "moderado",
            "chance_janela_ouro": "média",
            "comentario": "🟡 Ambiente misto. Oportunidades pontuais podem surgir no curto prazo.",
        }
    else:
        expectativa = {
            "previsibilidade": "baixa",
            "erro_esperado": "alto",
            "chance_janela_ouro": "baixa",
            "comentario": "🔴 Baixa previsibilidade nas próximas 1–3 séries (ruído/divergência dominantes).",
        }

    st.session_state["expectativa_v16"] = expectativa
    return expectativa


def v16_registrar_volume_e_confiabilidade():
    """
    Relaciona quantidade de previsões com confiabilidade estimada.
    O sistema informa — a decisão é do operador.
    """
    risco_pack = st.session_state.get("diagnostico_risco") or {}
    indice = risco_pack.get("indice_risco")

    if indice is None:
        volume_op = {
            "minimo": 3,
            "recomendado": 6,
            "maximo_tecnico": 20,
            "confiabilidades_estimadas": {},
            "comentario": "Confiabilidade não calculada (rode o Monitor de Risco).",
        }
        st.session_state["volume_operacional_v16"] = volume_op
        return volume_op

    indice = float(indice)
    conf_base = max(0.05, 1.0 - indice)

    volumes = [3, 6, 10, 20, 40, 80]
    confs = {}
    for v in volumes:
        confs[v] = round(max(0.01, conf_base - v * 0.003), 3)

    recomendado = 20 if conf_base > 0.35 else 6

    volume_op = {
        "minimo": 3,
        "recomendado": int(recomendado),
        "maximo_tecnico": 80,
        "confiabilidades_estimadas": confs,
        "comentario": (
            "O sistema informa volumes e confiabilidades estimadas. "
            "A decisão final de quantas previsões gerar é do operador."
        ),
    }

    st.session_state["volume_operacional_v16"] = volume_op
    return volume_op




# ============================================================
# Painel X — 🧠 Laudo Operacional V16 (Estado, Expectativa, Volume)
# ============================================================

if painel == "🧠 Laudo Operacional V16":

    st.markdown("## 🧠 Laudo Operacional V16 — Leitura do Ambiente")

    # --------------------------------------------------------
    # Leitura segura (usa Camada D se existir, senão guarda)
    # --------------------------------------------------------
    try:
        estado = v16_registrar_estado_alvo()
    except Exception:
        estado = {
            "tipo": "indefinido",
            "velocidade": "indefinida",
            "comentario": "Estado ainda não disponível.",
        }

    try:
        expectativa = v16_registrar_expectativa()
    except Exception:
        expectativa = {
            "previsibilidade": "indefinida",
            "erro_esperado": "indefinido",
            "chance_janela_ouro": "baixa",
            "comentario": "Expectativa ainda não disponível.",
        }

    try:
        volume_op = v16_registrar_volume_e_confiabilidade()
    except Exception:
        volume_op = {
            "minimo": "-",
            "recomendado": "-",
            "maximo_tecnico": "-",
            "confiabilidades_estimadas": {},
            "comentario": "Volume ainda não disponível.",
        }

    # --------------------------------------------------------
    # 1) Estado do Alvo
    # --------------------------------------------------------
    st.markdown("### 🎯 Estado do Alvo")
    st.info(
        f"Tipo: **{estado.get('tipo')}**  \n"
        f"Velocidade estimada: **{estado.get('velocidade')}**  \n"
        f"Comentário: {estado.get('comentario')}"
    )

    # --------------------------------------------------------
    # 2) Expectativa de Curto Prazo
    # --------------------------------------------------------
    st.markdown("### 🔮 Expectativa (1–3 séries)")
    st.info(
        f"Previsibilidade: **{expectativa.get('previsibilidade')}**  \n"
        f"Erro esperado: **{expectativa.get('erro_esperado')}**  \n"
        f"Chance de janela de ouro: **{expectativa.get('chance_janela_ouro')}**  \n\n"
        f"{expectativa.get('comentario')}"
    )

    # --------------------------------------------------------
    # 3) Volume x Confiabilidade
    # --------------------------------------------------------
    st.markdown("### 📊 Volume × Confiabilidade (informativo)")

    confs = volume_op.get("confiabilidades_estimadas", {})
    if isinstance(confs, dict) and confs:
        df_conf = pd.DataFrame(
            [{"Previsões": k, "Confiabilidade estimada": v} for k, v in confs.items()]
        )
        st.dataframe(df_conf, use_container_width=True)

    st.warning(
        f"📌 Volume mínimo: **{volume_op.get('minimo')}**  \n"
        f"📌 Volume recomendado: **{volume_op.get('recomendado')}**  \n"
        f"📌 Volume máximo técnico: **{volume_op.get('maximo_tecnico')}**  \n\n"
        f"{volume_op.get('comentario')}"
    )

    st.success(
        "O PredictCars informa o ambiente e os trade-offs.\n"
        "A decisão final de quantas previsões gerar é do operador."
    )



# ============================================================
# PARTE 7/8 — FIM
# ============================================================

# ============================================================
# PARTE 8/8 — INÍCIO
# ============================================================


# ============================================================
# 🔥 HOTFIX DEFINITIVO — EXATO PROXY (NORMALIZAÇÃO TOTAL)
# NÃO PROCURAR FUNÇÃO
# NÃO SUBSTITUIR CÓDIGO EXISTENTE
# ESTE BLOCO SOBRESCREVE O COMPORTAMENTO INTERNAMENTE
# ============================================================

def _v16_exato_proxy__normalizar_serie(valor):
    """
    Converte qualquer coisa em inteiro válido de passageiro.
    Aceita:
    - int
    - float
    - string ('12', '12.0', ' 12 ')
    Retorna None se inválido.
    """
    try:
        if valor is None:
            return None
        if isinstance(valor, str):
            valor = valor.strip().replace(",", ".")
        v = int(float(valor))
        return v
    except Exception:
        return None


def _v16_exato_proxy__topk_frequentes_FIX(window_df: pd.DataFrame, cols_pass: list, top_k: int) -> set:
    freq = {}
    for c in cols_pass:
        for v in window_df[c].values:
            vv = _v16_exato_proxy__normalizar_serie(v)
            if vv is not None:
                freq[vv] = freq.get(vv, 0) + 1
    if not freq:
        return set()
    return set(k for k, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:top_k])


def _v16_exato_proxy__serie_set_FIX(df_row: pd.Series, cols_pass: list) -> set:
    out = set()
    for c in cols_pass:
        vv = _v16_exato_proxy__normalizar_serie(df_row[c])
        if vv is not None:
            out.add(vv)
    return out


# 🔒 SOBRESCREVE FUNÇÕES USADAS PELO PAINEL (SEM VOCÊ CAÇAR NADA)
try:
    v16_exato_proxy__topk_frequentes = _v16_exato_proxy__topk_frequentes_FIX
    v16_exato_proxy__serie_set = _v16_exato_proxy__serie_set_FIX
except Exception:
    pass

# ============================================================
# 🔥 FIM HOTFIX DEFINITIVO — EXATO PROXY (NORMALIZAÇÃO TOTAL)
# ============================================================

# ============================================================
# 📊 BLOCO NOVO — V16 PREMIUM — EXATO POR REGIME (PROXY)
# VERSÃO FORÇADA — NÃO FICA EM BRANCO
# ============================================================

V16_PAINEL_EXATO_PROXY_NOME = "📊 V16 Premium — EXATO por Regime (Proxy)"


def v16_painel_exato_por_regime_proxy():
    st.markdown("## 📊 V16 Premium — EXATO por Regime (Proxy)")

    # --------------------------------------------------------
    # 0) Obter histórico BASE (FORÇADO)
    # --------------------------------------------------------
    try:
        nome_df, df_base = v16_identificar_df_base()
    except Exception:
        df_base = None

    if df_base is None or len(df_base) == 0:
        st.error("❌ Histórico não disponível. Painel abortado.")
        return

    st.success(f"✔ Histórico detectado: {len(df_base)} séries")

    # --------------------------------------------------------
    # 1) Extração FORÇADA dos passageiros
    # Regra: colunas 1..6
    # --------------------------------------------------------
    cols = list(df_base.columns)
    if len(cols) < 7:
        st.error("❌ Histórico não tem colunas suficientes.")
        return

    cols_pass = cols[1:7]
    st.info(f"Passageiros usados: {cols_pass}")

    # --------------------------------------------------------
    # 2) Normalização TOTAL
    # --------------------------------------------------------
    def norm(v):
        try:
            return int(float(str(v).strip()))
        except Exception:
            return None

    # --------------------------------------------------------
    # 3) Parâmetros FIXOS
    # --------------------------------------------------------
    W = 60
    TOP_K = 12

    if len(df_base) <= W:
        st.error("❌ Histórico insuficiente para janela W=60.")
        return

    # --------------------------------------------------------
    # 4) Loop FORÇADO (sem filtros que zeram tudo)
    # --------------------------------------------------------
    registros = []

    for t in range(W, len(df_base)):
        janela = df_base.iloc[t - W : t]
        prox = df_base.iloc[t]

        freq = {}
        for c in cols_pass:
            for v in janela[c].values:
                vv = norm(v)
                if vv is not None:
                    freq[vv] = freq.get(vv, 0) + 1

        if not freq:
            continue

        topk = set(k for k, _ in sorted(freq.items(), key=lambda x: -x[1])[:TOP_K])

        real = set()
        for c in cols_pass:
            vv = norm(prox[c])
            if vv is not None:
                real.add(vv)

        hits = len(topk & real)

        # regime SIMPLES (FORÇADO)
        if hits >= 3:
            regime = "ECO"
        elif hits >= 2:
            regime = "PRÉ-ECO"
        else:
            regime = "RUIM"

        registros.append(
            {"regime": regime, "hits": hits}
        )

    if not registros:
        st.error("❌ Nenhum registro gerado.")
        return

    df = pd.DataFrame(registros)

    # --------------------------------------------------------
    # 5) RESULTADOS GARANTIDOS
    # --------------------------------------------------------
    resumo = []
    for r in ["ECO", "PRÉ-ECO", "RUIM"]:
        sub = df[df["regime"] == r]
        resumo.append({
            "Regime": r,
            "Eventos": len(sub),
            "Hits ≥2 (%)": round((sub["hits"] >= 2).mean() * 100, 2) if len(sub) else 0.0,
            "Hits ≥3 (%)": round((sub["hits"] >= 3).mean() * 100, 2) if len(sub) else 0.0,
        })

    df_out = pd.DataFrame(resumo)

    st.markdown("### 📊 Resultado (FORÇADO)")
    st.dataframe(df_out, use_container_width=True)

    st.success("✅ Painel executado com sucesso (versão forçada).")


def v16_registrar_painel_exato_proxy__no_router():
    if st.session_state.get("_v16_exato_proxy_router_ok", False):
        return

    g = globals()

    if "v16_obter_paineis" in g:
        orig = g["v16_obter_paineis"]

        def novo():
            try:
                lst = list(orig())
            except Exception:
                lst = []
            if V16_PAINEL_EXATO_PROXY_NOME not in lst:
                lst.append(V16_PAINEL_EXATO_PROXY_NOME)
            return lst

        g["v16_obter_paineis"] = novo

    if "v16_renderizar_painel" in g:
        orig_r = g["v16_renderizar_painel"]

        def render(p):
            if p == V16_PAINEL_EXATO_PROXY_NOME:
                return v16_painel_exato_por_regime_proxy()
            return orig_r(p)

        g["v16_renderizar_painel"] = render

    st.session_state["_v16_exato_proxy_router_ok"] = True


try:
    v16_registrar_painel_exato_proxy__no_router()
except Exception:
    pass

# ============================================================
# 📊 FIM DO BLOCO NOVO — V16 PREMIUM — EXATO POR REGIME (PROXY)
# ============================================================

# ============================================================
# 📊 BLOCO NOVO — V16 PREMIUM — PRÉ-ECO → ECO (PERSISTÊNCIA & CONTINUIDADE)
# (COLAR ENTRE: FIM DO EXATO PROXY  e  INÍCIO DO V16 PREMIUM PROFUNDO)
# ============================================================

V16_PAINEL_PRE_ECO_PERSIST_NOME = "📊 V16 Premium — PRÉ-ECO → ECO (Persistência & Continuidade)"


def v16_painel_pre_eco_persistencia_continuidade():
    st.markdown("## 📊 V16 Premium — PRÉ-ECO → ECO (Persistência & Continuidade)")
    st.markdown(
        """
Este painel é **100% observacional** e **retrospectivo**.

Ele responde:
- ✅ Qual % de **PRÉ-ECO** vira **ECO** em **1–3 séries**?
- ✅ Como separar **PRÉ-ECO fraco** vs **PRÉ-ECO forte**?
- ✅ Quais são os **últimos PRÉ-ECO fortes** (para prontidão humana)?

**Sem mudar motor. Sem decidir operação.**
        """
    )

    # --------------------------------------------------------
    # 0) Histórico base (obrigatório)
    # --------------------------------------------------------
    try:
        nome_df, df_base = v16_identificar_df_base()
    except Exception:
        nome_df, df_base = None, None

    if df_base is None or len(df_base) == 0:
        st.warning("⚠️ Histórico não disponível. Carregue o histórico e volte aqui.")
        return

    cols = list(df_base.columns)
    if len(cols) < 7:
        st.error("❌ Histórico não tem colunas suficientes (precisa: série + 6 passageiros).")
        return

    cols_pass = cols[1:7]

    st.success(f"✔ Histórico detectado: {len(df_base)} séries")
    st.info(f"Passageiros usados: {cols_pass}")

    # --------------------------------------------------------
    # 1) Normalização TOTAL (robusta)
    # --------------------------------------------------------
    def norm(v):
        try:
            return int(float(str(v).strip().replace(",", ".")))
        except Exception:
            return None

    # --------------------------------------------------------
    # 2) Parâmetros FIXOS (decisão sem bifurcação)
    # --------------------------------------------------------
    W = 60
    TOP_K = 12
    LOOKAHEAD_MAX = 3
    RUN_BACK = 5
    MAX_JANELAS = 4000  # anti-zumbi interno

    if len(df_base) <= W + LOOKAHEAD_MAX:
        st.error(f"❌ Histórico insuficiente para W={W} + lookahead.")
        return

    # Anti-zumbi: só últimas MAX_JANELAS
    t_final = len(df_base) - 1
    t_inicial = max(W, t_final - MAX_JANELAS)

    st.markdown("### ⚙️ Parâmetros (fixos)")
    st.code(
        f"W = {W}\nTOP_K = {TOP_K}\nLOOKAHEAD_MAX = {LOOKAHEAD_MAX}\nRUN_BACK = {RUN_BACK}\nMAX_JANELAS = {MAX_JANELAS}",
        language="python",
    )

    st.info(f"🧱 Anti-zumbi interno: analisando t={t_inicial} até t={t_final} (máx {MAX_JANELAS} janelas).")

    # --------------------------------------------------------
    # 3) Funções internas (dx, topk, real, hits)
    # --------------------------------------------------------
    def dx_janela(window_df):
        vals = []
        for c in cols_pass:
            s = [norm(x) for x in window_df[c].values]
            s = [x for x in s if x is not None]
            if len(s) >= 2:
                vals.append(float(np.std(s, ddof=1)))
        if not vals:
            return None
        return float(np.mean(vals))

    def topk_frequentes(window_df):
        freq = {}
        for c in cols_pass:
            for x in window_df[c].values:
                vv = norm(x)
                if vv is not None:
                    freq[vv] = freq.get(vv, 0) + 1
        if not freq:
            return set()
        return set(k for k, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:TOP_K])

    def serie_real_set(df_row):
        out = set()
        for c in cols_pass:
            vv = norm(df_row[c])
            if vv is not None:
                out.add(vv)
        return out

    # --------------------------------------------------------
    # 4) Primeiro passe: dx_list para quantis ECO/PRE/RUIM
    # --------------------------------------------------------
    dx_list = []
    dx_por_t = {}
    for t in range(t_inicial, t_final + 1):
        wdf = df_base.iloc[t - W : t]
        dx = dx_janela(wdf)
        if dx is not None:
            dx_list.append(dx)
            dx_por_t[t] = dx

    if len(dx_list) < 80:
        st.error(f"❌ Poucas janelas válidas para quantis. Válidas: {len(dx_list)}")
        return

    q1 = float(np.quantile(dx_list, 0.33))
    q2 = float(np.quantile(dx_list, 0.66))

    st.markdown("### 🧭 Regimes por quantis (dx_janela)")
    st.info(
        f"q1 (ECO ≤): **{q1:.6f}**  \n"
        f"q2 (PRÉ-ECO ≤): **{q2:.6f}**  \n\n"
        "Regra: dx ≤ q1 → ECO | dx ≤ q2 → PRÉ-ECO | dx > q2 → RUIM"
    )

    # --------------------------------------------------------
    # 5) Segundo passe: regime + hits por t
    # --------------------------------------------------------
    registros = []
    regime_por_t = {}
    hits_por_t = {}

    for t in range(t_inicial, t_final + 1):
        if t not in dx_por_t:
            continue

        dx = dx_por_t[t]
        if dx <= q1:
            regime = "ECO"
        elif dx <= q2:
            regime = "PRÉ-ECO"
        else:
            regime = "RUIM"

        wdf = df_base.iloc[t - W : t]
        top = topk_frequentes(wdf)
        real = serie_real_set(df_base.iloc[t])
        hits = int(len(top & real))

        regime_por_t[t] = regime
        hits_por_t[t] = hits

        registros.append({"t": int(t), "dx": float(dx), "regime": regime, "hits": hits})

    if not registros:
        st.error("❌ Não houve registros válidos.")
        return

    df = pd.DataFrame(registros)

    # --------------------------------------------------------
    # 6) Persistência PRÉ-ECO (run_len_pre)
    # --------------------------------------------------------
    run_len_pre = {}
    current = 0
    for t in sorted(regime_por_t.keys()):
        if regime_por_t[t] == "PRÉ-ECO":
            current += 1
        else:
            current = 0
        run_len_pre[t] = current

    # --------------------------------------------------------
    # 7) PRÉ-ECO → ECO em 1..3 séries (taxas)
    # --------------------------------------------------------
    total_pre = 0
    vira_eco_1 = 0
    vira_eco_2 = 0
    vira_eco_3 = 0

    eventos_pre = []

    for t in sorted(regime_por_t.keys()):
        if regime_por_t[t] != "PRÉ-ECO":
            continue

        total_pre += 1

        r1 = regime_por_t.get(t + 1)
        r2 = regime_por_t.get(t + 2)
        r3 = regime_por_t.get(t + 3)

        ok1 = (r1 == "ECO")
        ok2 = (r1 == "ECO") or (r2 == "ECO")
        ok3 = (r1 == "ECO") or (r2 == "ECO") or (r3 == "ECO")

        vira_eco_1 += 1 if ok1 else 0
        vira_eco_2 += 1 if ok2 else 0
        vira_eco_3 += 1 if ok3 else 0

        # dx trend e repetição de hits>=2 (últimos RUN_BACK)
        ts = [x for x in range(t - (RUN_BACK - 1), t + 1) if x in dx_por_t and x in hits_por_t and x in regime_por_t]
        dx_seq = [dx_por_t[x] for x in ts]
        hit_seq = [hits_por_t[x] for x in ts]
        hits_2plus = sum(1 for h in hit_seq if h >= 2)

        dx_trend = "estável"
        if len(dx_seq) >= 2:
            if dx_seq[-1] < dx_seq[0]:
                dx_trend = "caindo"
            elif dx_seq[-1] > dx_seq[0]:
                dx_trend = "subindo"

        # Score simples (informativo): persistência + hits repetidos + dx caindo
        score = 0
        score += min(run_len_pre.get(t, 0), 12)            # 0..12
        score += hits_2plus                               # 0..5
        score += 2 if dx_trend == "caindo" else 0
        score -= 2 if dx_trend == "subindo" else 0
        score += 1 if ok3 else 0

        eventos_pre.append(
            {
                "t": int(t),
                "run_len_pre": int(run_len_pre.get(t, 0)),
                "hits_t": int(hits_por_t.get(t, 0)),
                "hits_2plus_ult5": int(hits_2plus),
                "dx_trend_ult5": dx_trend,
                "vira_ECO_em_1": bool(ok1),
                "vira_ECO_em_2": bool(ok2),
                "vira_ECO_em_3": bool(ok3),
                "score_pre_forte": int(score),
            }
        )

    if total_pre == 0:
        st.error("❌ Não houve eventos PRÉ-ECO para avaliar.")
        return

    taxa1 = vira_eco_1 / total_pre
    taxa2 = vira_eco_2 / total_pre
    taxa3 = vira_eco_3 / total_pre

    st.markdown("### ✅ Taxas PRÉ-ECO → ECO (objetivas)")
    st.dataframe(
        pd.DataFrame(
            [{
                "Eventos PRÉ-ECO": int(total_pre),
                "Vira ECO em 1": round(taxa1, 4),
                "Vira ECO em 2": round(taxa2, 4),
                "Vira ECO em 3": round(taxa3, 4),
            }]
        ),
        use_container_width=True
    )

    # --------------------------------------------------------
    # 8) Top PRÉ-ECO fortes recentes (guia humano)
    # --------------------------------------------------------
    df_evt = pd.DataFrame(eventos_pre).sort_values(["t"], ascending=True)

    # Top 10 recentes com maior score
    df_top = (
        df_evt.sort_values(["score_pre_forte", "t"], ascending=[False, False])
        .head(10)
        .copy()
    )

    st.markdown("### 🟡 Top 10 PRÉ-ECO fortes (recentes / score)")
    st.dataframe(df_top, use_container_width=True)

    st.success(
        "✅ Painel PRÉ-ECO → ECO executado.\n"
        "Ele mede persistência/continuidade — a decisão de prontidão continua humana."
    )


def v16_registrar_painel_pre_eco_persist__no_router():
    """
    Integra este painel ao roteador V16 (idempotente).
    """
    if st.session_state.get("_v16_pre_eco_persist_router_ok", False):
        return

    g = globals()

    if "v16_obter_paineis" in g and callable(g["v16_obter_paineis"]):
        _orig_obter = g["v16_obter_paineis"]

        def _wrap_v16_obter_paineis__pre_eco():
            try:
                lst = list(_orig_obter())
            except Exception:
                lst = []
            if V16_PAINEL_PRE_ECO_PERSIST_NOME not in lst:
                lst.append(V16_PAINEL_PRE_ECO_PERSIST_NOME)
            return lst

        g["v16_obter_paineis"] = _wrap_v16_obter_paineis__pre_eco

    if "v16_renderizar_painel" in g and callable(g["v16_renderizar_painel"]):
        _orig_render = g["v16_renderizar_painel"]

        def _wrap_v16_renderizar_painel__pre_eco(painel_nome: str):
            if painel_nome == V16_PAINEL_PRE_ECO_PERSIST_NOME:
                return v16_painel_pre_eco_persistencia_continuidade()
            return _orig_render(painel_nome)

        g["v16_renderizar_painel"] = _wrap_v16_renderizar_painel__pre_eco

    st.session_state["_v16_pre_eco_persist_router_ok"] = True


# Registrar no router imediatamente (sem mexer em menu/motor)
try:
    v16_registrar_painel_pre_eco_persist__no_router()
except Exception:
    pass

# ============================================================
# 📊 FIM DO BLOCO NOVO — V16 PREMIUM — PRÉ-ECO → ECO (PERSISTÊNCIA & CONTINUIDADE)
# ============================================================

# ============================================================
# 📊 BLOCO NOVO — V16 PREMIUM — PASSAGEIROS RECORRENTES EM ECO (INTERSEÇÃO)
# (COLAR IMEDIATAMENTE ANTES DE: "INÍCIO DO PAINEL V16 PREMIUM PROFUNDO  (COLAR AQUI)")
# ============================================================

V16_PAINEL_ECO_RECORRENTES_NOME = "📊 V16 Premium — Passageiros Recorrentes em ECO (Interseção)"


def v16_painel_passageiros_recorrentes_eco_intersecao():
    st.markdown("## 📊 V16 Premium — Passageiros Recorrentes em ECO (Interseção)")
    st.markdown(
        """
Este painel é **100% observacional** e **retrospectivo**.

Ele responde:
- ✅ Em **trechos ECO**, quais passageiros aparecem de forma **recorrente** (persistência)?
- ✅ Em blocos ECO **consecutivos**, qual é a **interseção** real dos TOP-K por janela?
- ✅ Quais são os **passageiros ECO-resilientes** (candidatos estruturais para EXATO)?

**Sem mudar motor. Sem decidir operação.**
        """
    )

    # --------------------------------------------------------
    # 0) Histórico base (robusto, sem caça)
    # --------------------------------------------------------
    try:
        nome_df, df_base = v16_identificar_df_base()
    except Exception:
        nome_df, df_base = None, None

    if df_base is None or len(df_base) == 0:
        st.warning("⚠️ Histórico não disponível. Carregue o histórico e volte aqui.")
        return

    cols = list(df_base.columns)
    if len(cols) < 7:
        st.error("❌ Histórico insuficiente: precisa de (série + 6 passageiros).")
        return

    cols_pass = cols[1:7]
    st.success(f"✔ Histórico detectado: {len(df_base)} séries")
    st.info(f"Passageiros usados: {cols_pass}")

    # --------------------------------------------------------
    # 1) Normalização TOTAL (robusta)
    # --------------------------------------------------------
    def norm(v):
        try:
            return int(float(str(v).strip().replace(",", ".")))
        except Exception:
            return None

    # --------------------------------------------------------
    # 2) Parâmetros FIXOS (sem bifurcação)
    # --------------------------------------------------------
    W = 60
    TOP_K = 12
    RUN_MIN = 3            # só consideramos "bloco ECO" com pelo menos 3 janelas ECO consecutivas
    MAX_JANELAS = 4000     # anti-zumbi interno

    if len(df_base) <= W + 5:
        st.error(f"❌ Histórico insuficiente para W={W}.")
        return

    t_final = len(df_base) - 1
    t_inicial = max(W, t_final - MAX_JANELAS)

    st.markdown("### ⚙️ Parâmetros (fixos)")
    st.code(
        f"W = {W}\nTOP_K = {TOP_K}\nRUN_MIN = {RUN_MIN}\nMAX_JANELAS = {MAX_JANELAS}",
        language="python",
    )
    st.info(f"🧱 Anti-zumbi interno: analisando t={t_inicial} até t={t_final} (máx {MAX_JANELAS} janelas).")

    # --------------------------------------------------------
    # 3) Funções internas (dx, topk)
    # --------------------------------------------------------
    def dx_janela(window_df):
        vals = []
        for c in cols_pass:
            s = [norm(x) for x in window_df[c].values]
            s = [x for x in s if x is not None]
            if len(s) >= 2:
                vals.append(float(np.std(s, ddof=1)))
        if not vals:
            return None
        return float(np.mean(vals))

    def topk_frequentes(window_df):
        freq = {}
        for c in cols_pass:
            for x in window_df[c].values:
                vv = norm(x)
                if vv is not None:
                    freq[vv] = freq.get(vv, 0) + 1
        if not freq:
            return set()
        ordenado = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        return set(k for k, _ in ordenado[:TOP_K])

    # --------------------------------------------------------
    # 4) Primeiro passe: dx por t + quantis para ECO/PRE/RUIM
    # --------------------------------------------------------
    dx_list = []
    dx_por_t = {}
    for t in range(t_inicial, t_final + 1):
        wdf = df_base.iloc[t - W : t]
        dx = dx_janela(wdf)
        if dx is not None:
            dx_list.append(dx)
            dx_por_t[t] = dx

    if len(dx_list) < 80:
        st.error(f"❌ Poucas janelas válidas para quantis. Válidas: {len(dx_list)}")
        return

    q1 = float(np.quantile(dx_list, 0.33))
    q2 = float(np.quantile(dx_list, 0.66))

    st.markdown("### 🧭 Regimes por quantis (dx_janela)")
    st.info(
        f"q1 (ECO ≤): **{q1:.6f}**  \n"
        f"q2 (PRÉ-ECO ≤): **{q2:.6f}**  \n\n"
        "Regra: dx ≤ q1 → ECO | dx ≤ q2 → PRÉ-ECO | dx > q2 → RUIM"
    )

    # --------------------------------------------------------
    # 5) Segundo passe: regime por t + TOP-K por t (apenas ECO)
    # --------------------------------------------------------
    regime_por_t = {}
    top_por_t = {}

    for t in range(t_inicial, t_final + 1):
        dx = dx_por_t.get(t)
        if dx is None:
            continue

        if dx <= q1:
            regime = "ECO"
        elif dx <= q2:
            regime = "PRÉ-ECO"
        else:
            regime = "RUIM"

        regime_por_t[t] = regime

        if regime == "ECO":
            wdf = df_base.iloc[t - W : t]
            top_por_t[t] = topk_frequentes(wdf)

    if not top_por_t:
        st.warning("⚠️ Nenhuma janela ECO detectada neste recorte.")
        return

    # --------------------------------------------------------
    # 6) Detectar blocos ECO consecutivos (runs)
    # --------------------------------------------------------
    ts_eco = sorted(top_por_t.keys())

    runs = []
    start = ts_eco[0]
    prev = ts_eco[0]
    for t in ts_eco[1:]:
        if t == prev + 1:
            prev = t
        else:
            runs.append((start, prev))
            start = t
            prev = t
    runs.append((start, prev))

    # filtrar runs curtos
    runs = [r for r in runs if (r[1] - r[0] + 1) >= RUN_MIN]

    st.markdown("### 🟢 Blocos ECO consecutivos (detectados)")
    st.info(
        f"Total de runs ECO (≥ {RUN_MIN}): **{len(runs)}**  \n"
        f"Total de janelas ECO: **{len(ts_eco)}**"
    )

    if not runs:
        st.warning("⚠️ Existem janelas ECO, mas nenhuma sequência ECO longa o suficiente (RUN_MIN).")
        return

    # --------------------------------------------------------
    # 7) Para cada run ECO: interseções cumulativas e persistência
    # --------------------------------------------------------
    resumo_runs = []
    contagem_passageiros_eco = {}  # persistência global em ECO (conta presença em TOP-K por janela)
    total_janelas_eco = 0

    for (a, b) in runs:
        ts = list(range(a, b + 1))
        sets = [top_por_t[t] for t in ts if t in top_por_t]
        if len(sets) < RUN_MIN:
            continue

        # persistência global
        for s in sets:
            for p in s:
                contagem_passageiros_eco[p] = contagem_passageiros_eco.get(p, 0) + 1

        total_janelas_eco += len(sets)

        # interseções cumulativas (2..min(6, len))
        inter_2 = None
        inter_3 = None
        inter_4 = None
        inter_5 = None
        inter_6 = None

        def inter_size(n):
            if len(sets) < n:
                return None
            inter = sets[0].copy()
            for i in range(1, n):
                inter &= sets[i]
            return len(inter)

        inter_2 = inter_size(2)
        inter_3 = inter_size(3)
        inter_4 = inter_size(4)
        inter_5 = inter_size(5)
        inter_6 = inter_size(6)

        # score simples do run (informativo): inter_3 e inter_4 pesam mais
        score_run = 0
        if inter_2 is not None: score_run += inter_2
        if inter_3 is not None: score_run += 2 * inter_3
        if inter_4 is not None: score_run += 3 * inter_4

        resumo_runs.append(
            {
                "t_ini": int(a),
                "t_fim": int(b),
                "len_run": int(b - a + 1),
                "inter_2": inter_2 if inter_2 is not None else 0,
                "inter_3": inter_3 if inter_3 is not None else 0,
                "inter_4": inter_4 if inter_4 is not None else 0,
                "inter_5": inter_5 if inter_5 is not None else 0,
                "inter_6": inter_6 if inter_6 is not None else 0,
                "score_run": int(score_run),
            }
        )

    if not resumo_runs:
        st.warning("⚠️ Não consegui consolidar runs ECO (depois de filtros).")
        return

    df_runs = pd.DataFrame(resumo_runs).sort_values(["score_run", "len_run", "t_fim"], ascending=[False, False, False])

    st.markdown("### 📊 Runs ECO — Interseção TOP-K (cumulativa)")
    st.dataframe(df_runs, use_container_width=True)

    # --------------------------------------------------------
    # 8) Passageiros ECO-resilientes (persistência global em ECO)
    # --------------------------------------------------------
    st.markdown("### 🎯 Passageiros ECO-resilientes (persistência em TOP-K durante ECO)")

    if total_janelas_eco <= 0:
        st.warning("⚠️ Total de janelas ECO inválido.")
        return

    itens = []
    for p, cnt in contagem_passageiros_eco.items():
        itens.append(
            {
                "passageiro": int(p),
                "presencas_em_ECO": int(cnt),
                "taxa_presenca_ECO": round(float(cnt) / float(total_janelas_eco), 4),
            }
        )

    df_p = pd.DataFrame(itens).sort_values(["taxa_presenca_ECO", "presencas_em_ECO", "passageiro"], ascending=[False, False, True])

    st.info(f"Total de janelas ECO consideradas (em runs): **{total_janelas_eco}**")
    st.dataframe(df_p.head(25), use_container_width=True)

    # lista curta (top 12)
    top12 = df_p.head(12)["passageiro"].tolist()
    st.success("✅ Lista curta (TOP 12 ECO-resilientes) — informativa (não é previsão):")
    st.code(", ".join(str(x) for x in top12))

    st.success(
        "✅ Painel Passageiros Recorrentes em ECO executado.\n"
        "Ele mede persistência/interseção — a decisão de ataque e montagem para 6 continua humana."
    )


# ============================================================
# 📊 FIM DO BLOCO NOVO — V16 PREMIUM — PASSAGEIROS RECORRENTES EM ECO (INTERSEÇÃO)
# ============================================================


# ============================================================
# INÍCIO DO PAINEL V16 PREMIUM PROFUNDO  (COLAR AQUI)
# ============================================================

# ============================================================
# PAINEL — 🔮 V16 Premium Profundo — Diagnóstico & Calibração
# ============================================================
if painel == "🔮 V16 Premium Profundo — Diagnóstico & Calibração":
    st.markdown("## 🔮 V16 Premium Profundo — Diagnóstico & Calibração")
    st.markdown(
        """
        Este painel **não altera nada do fluxo V15.7 MAX**.

        Ele serve para:
        - 📊 **Inspecionar o histórico ativo** (tamanho, colunas, distribuição de k),
        - 🛡️ **Verificar rapidamente o regime de risco potencial** para o TURBO++ e Modo 6 Acertos,
        - 📐 **Organizar informações de confiabilidade/QDS/k*** já calculadas em outros painéis.

        Tudo com **anti-zumbi interno**, rodando apenas em uma janela segura do histórico.
        """
    )

    # --------------------------------------------------------
    # 1) Descobrir automaticamente qual DF de histórico usar
    # --------------------------------------------------------
    nome_df, df_base = v16_identificar_df_base()

    if df_base is None:
        st.warning(
            "⚠️ Não encontrei nenhum DataFrame de histórico ativo em `st.session_state`.\n\n"
            "Use primeiro um painel que carregue o histórico (por exemplo, **Carregar Histórico**), "
            "e depois volte aqui."
        )
        st.stop()

    st.info(
        f"📁 DataFrame detectado para diagnóstico: **{nome_df}**  \n"
        f"Séries totais disponíveis: **{len(df_base)}**"
    )

    # --------------------------------------------------------
    # 2) Controle Anti-Zumbi V16 (apenas para este painel)
    # --------------------------------------------------------
    n_total = int(len(df_base))
    limite_max_slider = int(min(6000, max(500, n_total)))

    st.markdown("### 🛡️ Anti-zumbi V16 — Janela de Diagnóstico")

    limite_linhas = st.slider(
        "Quantidade máxima de séries a considerar no diagnóstico (janela final do histórico):",
        min_value=200,
        max_value=limite_max_slider,
        value=min(2000, limite_max_slider),
        step=100,
    )

    # --------------------------------------------------------
    # 3) Resumo básico do histórico (janela segura)
    # --------------------------------------------------------
    resumo = v16_resumo_basico_historico(df_base, limite_linhas=limite_linhas)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Séries totais no histórico", resumo.get("n_total", 0))
    with col2:
        st.metric("Séries usadas no diagnóstico", resumo.get("n_usado", 0))
    with col3:
        st.metric("Qtd. de colunas detectadas", len(resumo.get("colunas", [])))

    st.markdown("### 🧬 Colunas detectadas na janela de diagnóstico")
    st.write(resumo.get("colunas", []))

    # Distribuição de k (se existir)
    dist_k = resumo.get("dist_k", {})
    if dist_k:
        st.markdown("### 🎯 Distribuição de k (janela final do histórico)")
        df_k = pd.DataFrame(
            {"k": list(dist_k.keys()), "qtd": list(dist_k.values())}
        ).sort_values("k")
        df_k["proporção (%)"] = (df_k["qtd"] / df_k["qtd"].sum() * 100).round(2)
        st.dataframe(df_k, use_container_width=True)
    else:
        st.info("ℹ️ Não foi possível calcular a distribuição de k.")

    # --------------------------------------------------------
    # 4) Mapa rápido de confiabilidade / QDS / k*
    # --------------------------------------------------------
    st.markdown("### 🧠 Mapa rápido de confiabilidade (session_state)")

    with st.expander("Ver variáveis relevantes detectadas"):
        mapeamento_conf = v16_mapear_confiabilidade_session_state()
        if not mapeamento_conf:
            st.write("Nenhuma variável relevante encontrada.")
        else:
            st.json(mapeamento_conf)

    # --------------------------------------------------------
    # 5) Interpretação qualitativa do regime
    # --------------------------------------------------------
    st.markdown("### 🩺 Interpretação qualitativa do regime")
    comentario_regime = []

    if dist_k:
        total_k = sum(dist_k.values())
        proporcao_k_alto = round(
            sum(qtd for k_val, qtd in dist_k.items() if k_val >= 3) / total_k * 100,
            2,
        )
        proporcao_k_baixo = round(
            sum(qtd for k_val, qtd in dist_k.items() if k_val <= 1) / total_k * 100,
            2,
        )

        comentario_regime.append(f"- k ≥ 3: **{proporcao_k_alto}%**")
        comentario_regime.append(f"- k ≤ 1: **{proporcao_k_baixo}%**")

        if proporcao_k_alto >= 35:
            comentario_regime.append("- 🟢 Regime mais estável.")
        elif proporcao_k_baixo >= 50:
            comentario_regime.append("- 🔴 Regime turbulento.")
        else:
            comentario_regime.append("- 🟡 Regime intermediário.")
    else:
        comentario_regime.append("- ℹ️ Sem dados suficientes para avaliar o regime.")

    st.markdown("\n".join(comentario_regime))

    st.success("Painel V16 Premium Profundo executado com sucesso!")

# ======================================================================
# 📊 V16 PREMIUM — PRÉ-ECO | CONTRIBUIÇÃO DE PASSAGEIROS (OBSERVACIONAL)
# (CTRL+F ESTE BLOCO)
# ======================================================================

def _v16_laplace_rate(sucessos: int, total: int, alpha: int = 1) -> float:
    # Suavização Laplace: (a+α)/(A+2α)
    if total <= 0:
        return 0.0
    return float((sucessos + alpha) / (total + 2 * alpha))

def _v16_wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    # Wilson score interval para proporção
    if n <= 0:
        return (0.0, 1.0)
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2*n)) / denom
    margin = (z / denom) * math.sqrt((p*(1-p)/n) + (z**2)/(4*(n**2)))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (lo, hi)

def _v16_delta_ci_worstcase(p1_ci: Tuple[float, float], p0_ci: Tuple[float, float]) -> Tuple[float, float]:
    # IC conservador para Δ = P1 - P0 usando pior caso:
    # Δ_lo = P1_lo - P0_hi ; Δ_hi = P1_hi - P0_lo
    return (p1_ci[0] - p0_ci[1], p1_ci[1] - p0_ci[0])

def _v16_safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default

def _v16_build_pre_eco_mask(df_ctx: pd.DataFrame,
                           teto_nr: float,
                           teto_div: float,
                           kstar_delta_max: float = 0.0) -> pd.Series:
    """
    PRÉ-ECO = prontidão objetiva:
      - NR% não explode
      - Divergência não hostil
      - k* não piora (Δk* <= kstar_delta_max)
      - Laudo não hostil (se existir coluna)
    """
    # Colunas esperadas (se existirem): 'kstar', 'nr', 'div', 'laudo_hostil'
    nr = df_ctx["nr"] if "nr" in df_ctx.columns else pd.Series([np.nan]*len(df_ctx))
    div = df_ctx["div"] if "div" in df_ctx.columns else pd.Series([np.nan]*len(df_ctx))
    kstar = df_ctx["kstar"] if "kstar" in df_ctx.columns else pd.Series([np.nan]*len(df_ctx))

    # Δk*
    kstar_prev = kstar.shift(1)
    dk = (kstar - kstar_prev)

    ok_nr = nr.apply(lambda v: _v16_safe_float(v, 999.0) <= teto_nr)
    ok_div = div.apply(lambda v: _v16_safe_float(v, 999.0) <= teto_div)
    ok_k = dk.apply(lambda v: _v16_safe_float(v, 999.0) <= kstar_delta_max)

    if "laudo_hostil" in df_ctx.columns:
        # laudo_hostil True = hostil, então queremos False
        ok_laudo = (~df_ctx["laudo_hostil"].fillna(False)).astype(bool)
    else:
        ok_laudo = pd.Series([True]*len(df_ctx))

    preeco = (ok_nr & ok_div & ok_k & ok_laudo)
    return preeco

def _v16_hits_exatos(car_a: List[int], car_b: List[int]) -> int:
    # acertos exatos = interseção simples
    sa = set(car_a)
    sb = set(car_b)
    return len(sa.intersection(sb))

def _v16_extract_car_numbers(row: Any) -> List[int]:
    """
    Extrator robusto: tenta pegar lista/tupla/np.array; se for string, tenta parsear dígitos.
    Mantém só ints >=0.
    """
    if row is None:
        return []
    if isinstance(row, (list, tuple, np.ndarray)):
        out = []
        for v in row:
            try:
                out.append(int(v))
            except Exception:
                pass
        return out
    if isinstance(row, str):
        # Extrai números inteiros de uma string
        nums = []
        cur = ""
        for ch in row:
            if ch.isdigit():
                cur += ch
            else:
                if cur != "":
                    nums.append(int(cur))
                    cur = ""
        if cur != "":
            nums.append(int(cur))
        return nums
    # fallback
    try:
        return [int(row)]
    except Exception:
        return []

def _v16_compute_contrib_table(historico_carros: List[List[int]],
                               df_ctx: pd.DataFrame,
                               preeco_mask: pd.Series,
                               w: int = 60,
                               alpha: int = 1,
                               amin: int = 6,
                               bmin: int = 20) -> pd.DataFrame:
    """
    Contribuição de passageiros no PRÉ-ECO:
      Para cada t (dentro janela), observa passageiros do carro real em t,
      e mede hit2/hit3 do próximo alvo (t+1).
    """
    n = len(historico_carros)
    if n < (w + 2):
        return pd.DataFrame()

    # Índices válidos: precisamos de t e t+1 existirem
    t_ini = max(0, n - w - 2)
    t_fim = n - 2  # último t que ainda tem t+1

    # Subconjunto analisado
    idxs = list(range(t_ini, t_fim + 1))

    # PRÉ-ECO alinhado em t
    preeco_sub = preeco_mask.iloc[idxs].reset_index(drop=True) if len(preeco_mask) >= n else pd.Series([False]*len(idxs))

    # Monta targets hit2/hit3 do alvo (t+1) com referência no t?
    # Aqui seguimos a definição observacional: hits exatos entre carro(t) e carro(t+1).
    # (Não é acerto do sistema; é dinâmica do alvo entre séries consecutivas.)
    hit2 = []
    hit3 = []
    passageiros_t = []

    for t in idxs:
        car_t = historico_carros[t]
        car_next = historico_carros[t+1]
        h = _v16_hits_exatos(car_t, car_next)
        hit2.append(1 if h >= 2 else 0)
        hit3.append(1 if h >= 3 else 0)
        passageiros_t.append(set(car_t))

    # Filtra só PRÉ-ECO
    rows = []
    for i, t in enumerate(idxs):
        if bool(preeco_sub.iloc[i]):
            rows.append((i, passageiros_t[i], hit2[i], hit3[i]))

    if len(rows) < 5:
        return pd.DataFrame()

    # Universo de passageiros observados no PRÉ-ECO
    universo = set()
    for _, ps, _, _ in rows:
        universo |= set(ps)
    universo = sorted(list(universo))

    # Base rates (para suporte)
    base_hit2 = sum(r[2] for r in rows) / max(1, len(rows))
    base_hit3 = sum(r[3] for r in rows) / max(1, len(rows))

    # Para cada passageiro p: conta A/B/a/b para hit2 e hit3
    data = []
    for p in universo:
        A = 0
        B = 0

        a2 = 0
        b2 = 0
        a3 = 0
        b3 = 0

        for _, ps, y2, y3 in rows:
            if p in ps:
                A += 1
                a2 += y2
                a3 += y3
            else:
                B += 1
                b2 += y2
                b3 += y3

        # Gates
        if A < amin or B < bmin:
            cls = "INSUFICIENTE"
        else:
            cls = "PENDENTE"  # define abaixo

        # Taxas suavizadas
        p1_2 = _v16_laplace_rate(a2, A, alpha=alpha)
        p0_2 = _v16_laplace_rate(b2, B, alpha=alpha)
        p1_3 = _v16_laplace_rate(a3, A, alpha=alpha)
        p0_3 = _v16_laplace_rate(b3, B, alpha=alpha)

        # Lifts
        lift2 = (p1_2 / p0_2) if p0_2 > 0 else np.nan
        lift3 = (p1_3 / p0_3) if p0_3 > 0 else np.nan

        # IC Wilson para proporções (usando p sem Laplace para CI, mais “puro”)
        raw_p1_2 = (a2 / A) if A > 0 else 0.0
        raw_p0_2 = (b2 / B) if B > 0 else 0.0
        raw_p1_3 = (a3 / A) if A > 0 else 0.0
        raw_p0_3 = (b3 / B) if B > 0 else 0.0

        ci_p1_2 = _v16_wilson_ci(raw_p1_2, A)
        ci_p0_2 = _v16_wilson_ci(raw_p0_2, B)
        ci_p1_3 = _v16_wilson_ci(raw_p1_3, A)
        ci_p0_3 = _v16_wilson_ci(raw_p0_3, B)

        # Δ e IC conservador
        d2 = p1_2 - p0_2
        d3 = p1_3 - p0_3

        ci_d2 = _v16_delta_ci_worstcase(ci_p1_2, ci_p0_2)
        ci_d3 = _v16_delta_ci_worstcase(ci_p1_3, ci_p0_3)

        # Score (z aprox): z = Δ / SE(Δ) (SE aprox com raw, para não “embelezar”)
        se2 = math.sqrt((raw_p1_2*(1-raw_p1_2)/max(1, A)) + (raw_p0_2*(1-raw_p0_2)/max(1, B)))
        se3 = math.sqrt((raw_p1_3*(1-raw_p1_3)/max(1, A)) + (raw_p0_3*(1-raw_p0_3)/max(1, B)))

        z2 = ( (raw_p1_2 - raw_p0_2) / se2 ) if se2 > 0 else 0.0
        z3 = ( (raw_p1_3 - raw_p0_3) / se3 ) if se3 > 0 else 0.0

        score = (2.0 * z3) + (1.0 * z2)

        # Classificação (só se não for insuficiente)
        if cls != "INSUFICIENTE":
            # Regras conservadoras (fixas)
            leader = (ci_d3[0] > 0.0) and (not np.isnan(lift3)) and (lift3 >= 1.10) and (score >= 1.0)
            discard = (ci_d3[1] < 0.0) and (not np.isnan(lift3)) and (lift3 <= 0.90) and (score <= -1.0)

            if leader:
                cls = "LÍDER"
            elif discard:
                cls = "DESCARTÁVEL"
            else:
                cls = "NEUTRO"

        data.append({
            "passageiro": int(p),
            "A_presente": int(A),
            "a_hit2": int(a2),
            "a_hit3": int(a3),
            "B_ausente": int(B),
            "b_hit2": int(b2),
            "b_hit3": int(b3),
            "P1_hit2": float(p1_2),
            "P0_hit2": float(p0_2),
            "Δ_hit2": float(d2),
            "Lift_hit2": float(lift2) if not np.isnan(lift2) else np.nan,
            "ICΔ_hit2_lo": float(ci_d2[0]),
            "ICΔ_hit2_hi": float(ci_d2[1]),
            "P1_hit3": float(p1_3),
            "P0_hit3": float(p0_3),
            "Δ_hit3": float(d3),
            "Lift_hit3": float(lift3) if not np.isnan(lift3) else np.nan,
            "ICΔ_hit3_lo": float(ci_d3[0]),
            "ICΔ_hit3_hi": float(ci_d3[1]),
            "z_hit2": float(z2),
            "z_hit3": float(z3),
            "score": float(score),
            "classe": cls,
            "base_hit2_preEco": float(base_hit2),
            "base_hit3_preEco": float(base_hit3),
        })

    df = pd.DataFrame(data)
    if df.empty:
        return df

    # Ordenação: primeiro líderes por score, depois neutros, depois descartáveis, depois insuficientes
    ordem = {"LÍDER": 0, "NEUTRO": 1, "DESCARTÁVEL": 2, "INSUFICIENTE": 3}
    df["ordem_classe"] = df["classe"].map(ordem).fillna(9).astype(int)
    df = df.sort_values(by=["ordem_classe", "score"], ascending=[True, False]).drop(columns=["ordem_classe"])
    return df

def _v16_pairwise_coocorrencia(preeco_rows_passageiros: List[set], top_n: int = 25) -> pd.DataFrame:
    """
    Coocorrência (Jaccard) entre passageiros dentro do PRÉ-ECO.
    Retorna top pares com maior Jaccard (para alertar líder condicionado).
    """
    if len(preeco_rows_passageiros) < 8:
        return pd.DataFrame()

    # Universo
    uni = set()
    for s in preeco_rows_passageiros:
        uni |= set(s)
    uni = sorted(list(uni))

    # Contagens de presença
    pres = {p: 0 for p in uni}
    for s in preeco_rows_passageiros:
        for p in s:
            pres[p] += 1

    # Pairs
    pairs = []
    uni_len = len(uni)
    for i in range(uni_len):
        p = uni[i]
        for j in range(i+1, uni_len):
            q = uni[j]
            inter = 0
            union = 0
            for s in preeco_rows_passageiros:
                ip = (p in s)
                iq = (q in s)
                if ip or iq:
                    union += 1
                    if ip and iq:
                        inter += 1
            if union > 0:
                jac = inter / union
                if jac > 0:
                    pairs.append((p, q, inter, union, jac))

    if not pairs:
        return pd.DataFrame()

    dfp = pd.DataFrame(pairs, columns=["p", "q", "inter", "union", "jaccard"])
    dfp = dfp.sort_values(by="jaccard", ascending=False).head(top_n)
    return dfp

# ----------------------------------------------------------------------
# 📊 PAINEL — V16 PREMIUM — PRÉ-ECO | CONTRIBUIÇÃO DE PASSAGEIROS
# ----------------------------------------------------------------------
if "painel" in locals() and painel == "📊 V16 Premium — PRÉ-ECO | Contribuição de Passageiros":
    st.title("📊 V16 Premium — PRÉ-ECO | Contribuição de Passageiros")
    st.caption("Observacional, retrospectivo, objetivo e replicável. ❌ Sem motor. ❌ Sem listas. ✅ Só EXATO (Hit2/Hit3).")

    # -----------------------------
    # Parâmetros FIXOS (comando)
    # -----------------------------
    W_FIXO = 60
    ALPHA = 1
    AMIN = 6
    BMIN = 20

    with st.expander("🔒 Critério fixo (transparência total)", expanded=True):
        st.markdown(
            f"""
- **Janela W:** `{W_FIXO}` (fixo)
- **Suavização Laplace α:** `{ALPHA}` (fixo)
- **Amin / Bmin:** `{AMIN}` / `{BMIN}` (fixo)
- **Foco:** Hit3 (peso 2) + Hit2 (peso 1) → **score**
- **PRÉ-ECO:** filtro objetivo (NR, divergência, Δk*, laudo hostil se existir)
"""
        )

    # -----------------------------
    # Coleta do histórico (somente leitura)
    # -----------------------------
    # Tentamos chaves prováveis sem quebrar o app
    historico_carros = None

    # Opção 1: já existe lista pronta em session_state
    for k in ["historico_carros", "historico", "carros_historico", "dados_historico_carros"]:
        if k in st.session_state and st.session_state[k] is not None:
            historico_carros = st.session_state[k]
            break

    # Opção 2: tenta montar a partir de um DataFrame de histórico
    if historico_carros is None:
        for kdf in ["df_historico", "df", "dados", "historico_df"]:
            if kdf in st.session_state and isinstance(st.session_state[kdf], pd.DataFrame):
                dfh = st.session_state[kdf].copy()
                # Tenta inferir colunas com números
                cols_num = [c for c in dfh.columns if str(c).lower().strip() in ["n1","n2","n3","n4","n5","n6","a","b","c","d","e","f"]]
                if len(cols_num) >= 5:
                    historico_carros = []
                    for _, r in dfh.iterrows():
                        car = []
                        for c in cols_num[:6]:
                            try:
                                car.append(int(r[c]))
                            except Exception:
                                pass
                        historico_carros.append(car)
                break

    if not historico_carros or len(historico_carros) < (W_FIXO + 2):
        st.warning("Histórico insuficiente para o painel (precisa de W+2 séries). Carregue histórico completo e rode novamente.")
        st.stop()

    n_total = len(historico_carros)
    st.info(f"📁 Histórico detectado: **{n_total} séries**. Janela analisada: **últimas {W_FIXO} séries úteis (com alvo t+1)**.")

    # -----------------------------
    # Contexto de métricas (k*, NR, diverg, laudo)
    # -----------------------------
    # Este painel NÃO inventa métricas: ele lê o que existir.
    # Se não existir, ele opera com defaults conservadores → PRÉ-ECO vira “raríssimo” (ou vazio).
    df_ctx = pd.DataFrame({"idx": list(range(n_total))})

    # Tenta puxar séries de k*, NR, divergência, laudo hostil (se já existirem no seu app)
    # Chaves prováveis (mantendo robusto)
    series_map = [
        ("kstar", ["kstar_series", "serie_kstar", "kstar_hist", "kstar_por_serie"]),
        ("nr",    ["nr_series", "serie_nr", "nr_hist", "nr_por_serie"]),
        ("div",   ["div_series", "serie_div", "div_hist", "divergencia_series", "div_s6_mc_series"]),
        ("laudo_hostil", ["laudo_hostil_series", "serie_laudo_hostil"]),
    ]

    for col, keys in series_map:
        val = None
        for kk in keys:
            if kk in st.session_state and st.session_state[kk] is not None:
                val = st.session_state[kk]
                break
        if val is not None:
            try:
                s = pd.Series(list(val))
                if len(s) >= n_total:
                    s = s.iloc[:n_total]
                else:
                    # completa com NaN
                    s = s.reindex(range(n_total))
                df_ctx[col] = s
            except Exception:
                pass

    # Tetos PRÉ-ECO (fixos/visíveis — mas não “otimizáveis”)
    # Se você já tiver tetos globais no app, você pode substituir por leitura deles.
    teto_nr = 0.20
    teto_div = 0.35

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("🔎 Teto NR% (PRÉ-ECO)", f"{teto_nr:.2f}")
    with colB:
        st.metric("🔎 Teto Diverg (PRÉ-ECO)", f"{teto_div:.2f}")
    with colC:
        st.metric("🔎 Δk* máx (PRÉ-ECO)", "≤ 0.00")

    preeco_mask = _v16_build_pre_eco_mask(df_ctx=df_ctx, teto_nr=teto_nr, teto_div=teto_div, kstar_delta_max=0.0)

    # Aplica janela W (final do histórico)
    t_ini = max(0, n_total - W_FIXO - 2)
    t_fim = n_total - 2
    preeco_sub = preeco_mask.iloc[t_ini:t_fim+1].reset_index(drop=True)

    qtd_preeco = int(preeco_sub.sum())
    st.success(f"🟡 Rodadas PRÉ-ECO detectadas (na janela): **{qtd_preeco}** / {len(preeco_sub)}")

    if qtd_preeco < 5:
        st.warning("PRÉ-ECO muito raro nesta janela (ou métricas ausentes). O painel mantém honestidade: sem base, sem classificação forte.")
        # ainda assim tentamos rodar; provavelmente vai dar vazio/insuficiente.

    # -----------------------------
    # Calcula tabela de contribuição
    # -----------------------------
    df_contrib = _v16_compute_contrib_table(
        historico_carros=historico_carros,
        df_ctx=df_ctx,
        preeco_mask=preeco_mask,
        w=W_FIXO,
        alpha=ALPHA,
        amin=AMIN,
        bmin=BMIN
    )

    if df_contrib.empty:
        st.warning("Sem dados suficientes para medir contribuição (PRÉ-ECO insuficiente ou janela curta).")
        st.stop()

    # -----------------------------
    # Visões (Líder / Neutro / Descartável / Insuficiente)
    # -----------------------------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🏁 LÍDERES", int((df_contrib["classe"] == "LÍDER").sum()))
    with c2:
        st.metric("⚪ NEUTROS", int((df_contrib["classe"] == "NEUTRO").sum()))
    with c3:
        st.metric("❌ DESCARTÁVEIS", int((df_contrib["classe"] == "DESCARTÁVEL").sum()))
    with c4:
        st.metric("🟡 INSUF.", int((df_contrib["classe"] == "INSUFICIENTE").sum()))

    st.markdown("### 🧾 Tabela completa (ordenada por classe → score)")
    st.dataframe(
        df_contrib,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    st.markdown("### 🏁 Top LÍDERES (PRÉ-ECO)")
    st.dataframe(
        df_contrib[df_contrib["classe"] == "LÍDER"].head(25),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### ❌ Top DESCARTÁVEIS (PRÉ-ECO)")
    st.dataframe(
        df_contrib[df_contrib["classe"] == "DESCARTÁVEL"].head(25),
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------
    # Coocorrência (Líder condicionado)
    # -----------------------------
    st.markdown("---")
    st.markdown("### 🔗 Coocorrência (Jaccard) — alerta de “líder condicionado”")

    # Reconstroi sets PRÉ-ECO na janela
    idxs = list(range(t_ini, t_fim + 1))
    preeco_rows_sets = []
    for t in idxs:
        if bool(preeco_mask.iloc[t]):
            preeco_rows_sets.append(set(historico_carros[t]))

    df_pairs = _v16_pairwise_coocorrencia(preeco_rows_sets, top_n=30)
    if df_pairs.empty:
        st.info("Coocorrência insuficiente para análise robusta nesta janela (ou PRÉ-ECO raro).")
    else:
        st.dataframe(df_pairs, use_container_width=True, hide_index=True)
        st.caption("Quanto maior o Jaccard, mais “colados” os passageiros aparecem. Isso NÃO é corte — é alerta observacional.")

    st.markdown("---")
    st.caption("🔒 Este painel é 100% observacional: não gera listas, não decide, não altera motor. Ele mede contribuição condicional no PRÉ-ECO (Hit2/Hit3).")

# ============================================================
# 📊 V16 PREMIUM — ANTI-EXATO | PASSAGEIROS NOCIVOS CONSISTENTES
# ============================================================
if painel == "📊 V16 Premium — ANTI-EXATO | Passageiros Nocivos":

    st.title("📊 V16 Premium — ANTI-EXATO | Passageiros Nocivos Consistentes")
    st.caption(
        "Observacional • Retrospectivo • Objetivo\n"
        "Identifica passageiros que REDUZEM a chance de EXATO (≥2 / ≥3).\n"
        "❌ Não gera listas • ❌ Não decide • ✅ Apoia limpeza do Modo 6"
    )

    # --------------------------------------------------------
    # Parâmetros FIXOS (canônicos)
    # --------------------------------------------------------
    W = 60
    ALPHA = 1
    AMIN = 12
    BMIN = 40

    st.markdown(
        f"""
**Critério fixo**
- Janela: **{W}**
- Suavização Laplace: **α = {ALPHA}**
- Amostra mínima: **A ≥ {AMIN}**, **B ≥ {BMIN}**
- Evento-alvo: **Hit3 (principal)** + Hit2 (suporte)
"""
    )

    # --------------------------------------------------------
    # Fonte CANÔNICA de passageiros (Pipeline V14-FLEX ULTRA)
    # --------------------------------------------------------
    if "pipeline_col_pass" not in st.session_state:
        st.warning(
            "Fonte canônica de passageiros não encontrada.\n\n"
            "Execute primeiro o painel 🛣️ Pipeline V14-FLEX ULTRA."
        )
        st.stop()

    col_pass = st.session_state["pipeline_col_pass"]

    nome_df, df_base = v16_identificar_df_base()
    if df_base is None:
        st.warning("Histórico não encontrado. Carregue o histórico e rode o Pipeline.")
        st.stop()

    if len(col_pass) < 6:
        st.warning("Fonte de passageiros inválida (menos de 6 colunas).")
        st.stop()

    historico = df_base[col_pass].astype(int).values.tolist()
    n = len(historico)

    if n < (W + 2):
        st.warning("Histórico insuficiente para análise ANTI-EXATO.")
        st.stop()

    # --------------------------------------------------------
    # Construção das janelas móveis
    # --------------------------------------------------------
    def contar_hits(car_a, car_b):
        return len(set(car_a).intersection(set(car_b)))

    resultados = []

    for t in range(n - W - 1, n - 1):
        janela = historico[t - W + 1 : t + 1]
        alvo = historico[t + 1]

        for car in janela:
            hits = contar_hits(car, alvo)
            resultados.append({
                "passageiros": car,
                "hit2": 1 if hits >= 2 else 0,
                "hit3": 1 if hits >= 3 else 0,
            })

    df = pd.DataFrame(resultados)

    universo = sorted({p for car in df["passageiros"] for p in car})

    linhas = []

    for p in universo:
        presente = df["passageiros"].apply(lambda x: p in x)

        A = int(presente.sum())
        B = int((~presente).sum())

        if A < AMIN or B < BMIN:
            classe = "INSUFICIENTE"
        else:
            a3 = df.loc[presente, "hit3"].sum()
            b3 = df.loc[~presente, "hit3"].sum()

            p1 = (a3 + ALPHA) / (A + 2 * ALPHA)
            p0 = (b3 + ALPHA) / (B + 2 * ALPHA)

            delta = p1 - p0
            lift = p1 / p0 if p0 > 0 else 1.0

            if delta < 0 and lift <= 0.92:
                classe = "NOCIVO CONSISTENTE"
            else:
                classe = "NEUTRO"

        linhas.append({
            "passageiro": p,
            "A_presente": A,
            "B_ausente": B,
            "classe": classe,
        })

    df_out = pd.DataFrame(linhas).sort_values("classe")

    st.markdown("### 🧾 Classificação de Passageiros")
    st.dataframe(df_out, use_container_width=True, hide_index=True)

    st.markdown(
        """
🧠 **Como usar este painel**
- Passageiros **NOCIVOS CONSISTENTES** são candidatos a **EVITAR** no Modo 6
- Não é corte automático
- Serve para **limpar listas**, não para criar novas
"""
    )

# ============================================================
# PAINEL — 🌐 MODO UNIVERSAL — AVALIAÇÃO OBSERVACIONAL
# (N-AGNÓSTICO • SEM DECISÃO • SEM GERAÇÃO)
# ============================================================

if painel == "🌐 Modo Universal — Avaliação Observacional":

    st.title("🌐 Modo Universal — Avaliação Observacional")
    st.caption(
        "Observacional • N-agnóstico • Sem geração • Sem decisão\n"
        "Avalia listas existentes contra o alvo real (n_real)."
    )

    df = st.session_state.get("historico_df")
    n_real = st.session_state.get("n_alvo")
    listas = st.session_state.get("modo6_listas_totais") or []

    if df is None or n_real is None:
        st.warning(
            "Histórico ou n_real não disponível.\n\n"
            "Carregue o histórico antes de usar este painel."
        )
        st.stop()

    # -----------------------------
    # Alvo real (última série válida)
    # -----------------------------
    col_pass = [c for c in df.columns if c.startswith("p")]
    alvo_real = (
        df[col_pass]
        .iloc[-1]
        .dropna()
        .astype(int)
        .tolist()
    )

    if not listas:
        st.info("Nenhuma lista disponível para avaliação.")
        st.stop()

    # -----------------------------
    # Orçamento manual (opcional)
    # -----------------------------
    st.subheader("🔢 Orçamento (opcional)")
    orcamento_manual = st.text_input(
        "Informe um orçamento manual (opcional)",
        value="",
        help="Se preenchido, substitui a tabela condicionada."
    )
    if orcamento_manual == "":
        orcamento_manual = None

    # -----------------------------
    # Avaliação observacional
    # -----------------------------
    resultados = avaliar_listas_universal(
        listas=listas,
        alvo_real=alvo_real,
        n_real=n_real,
        orcamento_manual=orcamento_manual,
    )

    if not resultados:
        st.info("Nenhuma lista válida para avaliação (listas < n_real são ignoradas).")
        st.stop()

    df_out = pd.DataFrame(resultados)

    st.subheader("📊 Resultados (acertos / n_real)")
    st.dataframe(df_out, use_container_width=True, hide_index=True)

    st.caption(
        "Leitura sempre relativa ao n_real.\n"
        "Listas com tamanho menor que n_real são descartadas automaticamente."
    )



# ============================================================
# PAINEL V16 PREMIUM — BACKTEST RÁPIDO DO PACOTE (N = 60)
# ============================================================
if painel == "📊 V16 Premium — Backtest Rápido do Pacote (N=60)":

    st.subheader("📊 V16 Premium — Backtest Rápido do Pacote (N = 60)")
    st.caption(
        "Ensaio estatístico do pacote ATUAL de listas sobre os últimos 60 alvos. "
        "Não é previsão. Não decide volume. Mede apenas resistência sob pressão."
    )

    # ------------------------------------------------------------
    # Recuperação segura do histórico
    # ------------------------------------------------------------
    historico_df = st.session_state.get("historico_df")

    if historico_df is None or historico_df.empty:
        st.warning("Histórico não encontrado. Carregue o histórico antes.")
        st.stop()

    if historico_df.shape[0] < 60:
        st.warning("Histórico insuficiente para backtest (mínimo: 60 séries).")
        st.stop()

    # ------------------------------------------------------------
    # Recuperação do pacote congelado
    # ------------------------------------------------------------
    pacote = st.session_state.get("pacote_listas_atual")

    if not pacote:
        st.warning("Nenhum pacote de listas foi registrado ainda.")
        st.stop()

    # ------------------------------------------------------------
    # Identificação das colunas de passageiros
    # ------------------------------------------------------------
    colunas_passageiros = [c for c in historico_df.columns if c.lower().startswith("p")]

    if not colunas_passageiros:
        st.error("Não foi possível identificar colunas de passageiros no histórico.")
        st.stop()

    # ------------------------------------------------------------
    # Preparação do histórico (últimos 60 alvos)
    # ------------------------------------------------------------
    ultimos_60 = historico_df.tail(60)

    resultados = {
        ">=3": 0,
        ">=4": 0,
        ">=5": 0,
        ">=6": 0,
    }

    total_testes = 0

    # ------------------------------------------------------------
    # Execução do backtest
    # ------------------------------------------------------------
    for _, linha in ultimos_60.iterrows():

        # Alvo reconstruído a partir das colunas reais
        alvo = set(int(linha[c]) for c in colunas_passageiros if pd.notna(linha[c]))

        for lista in pacote:
            acertos = len(set(lista) & alvo)
            total_testes += 1

            if acertos >= 3:
                resultados[">=3"] += 1
            if acertos >= 4:
                resultados[">=4"] += 1
            if acertos >= 5:
                resultados[">=5"] += 1
            if acertos >= 6:
                resultados[">=6"] += 1

    # ------------------------------------------------------------
    # Cálculo das porcentagens
    # ------------------------------------------------------------
    perc = {
        k: (v / total_testes) * 100 if total_testes > 0 else 0.0
        for k, v in resultados.items()
    }

    # ------------------------------------------------------------
    # Exibição
    # ------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("≥ 3 acertos", f"{perc['>=3']:.2f}%")
    col2.metric("≥ 4 acertos", f"{perc['>=4']:.2f}%")
    col3.metric("≥ 5 acertos", f"{perc['>=5']:.2f}%")
    col4.metric("≥ 6 acertos", f"{resultados['>=6']} ocorrências")

    st.info(
        "📌 Interpretação correta:\n"
        "- Percentuais baixos indicam palco escorregadio\n"
        "- Percentuais estáveis indicam pacote resiliente\n"
        "- Isso NÃO prevê o próximo alvo\n"
        "- Serve apenas para calibrar postura e volume"
    )


# ============================================================
# PAINEL — 🧭 CHECKLIST OPERACIONAL — DECISÃO (AGORA)
# ============================================================
if painel == "🧭 Checklist Operacional — Decisão (AGORA)":

    st.markdown("## 🧭 Checklist Operacional — Decisão (AGORA)")
    st.caption(
        "Checklist obrigatório ANTES do Modo 6 / Mandar Bala.\n"
        "Não calcula, não cria listas, não decide automaticamente."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # 1) Estrada
    # --------------------------------------------------------
    st.markdown("### 1️⃣ Estrada permite ataque?")
    st.markdown(
        "- k* **não piorou**\n"
        "- NR% **não explodiu**\n"
        "- Divergência **não disparou**"
    )
    estrada_ok = st.radio(
        "Resultado da leitura da estrada:",
        ["SIM", "NÃO"],
        horizontal=True,
        key="CHECKLIST_ESTRADA_OK",
    )

    # --------------------------------------------------------
    # 2) Regime
    # --------------------------------------------------------
    st.markdown("### 2️⃣ Regime jogável?")
    regime = st.radio(
        "Regime identificado:",
        ["OURO", "PRATA", "RUIM"],
        horizontal=True,
        key="CHECKLIST_REGIME",
    )

    # --------------------------------------------------------
    # 3) Eixo
    # --------------------------------------------------------
    st.markdown("### 3️⃣ Existe eixo claro nas listas?")
    eixo = st.radio(
        "Eixo identificado:",
        ["SIM", "NÃO"],
        horizontal=True,
        key="CHECKLIST_EIXO",
    )

    # --------------------------------------------------------
    # 4) Nocivos
    # --------------------------------------------------------
    st.markdown("### 4️⃣ Nocivos concentrados nas mesmas listas?")
    nocivos = st.radio(
        "Nocivos:",
        ["SIM", "NÃO"],
        horizontal=True,
        key="CHECKLIST_NOCIVOS",
    )

    st.markdown("---")

    # --------------------------------------------------------
    # 5) Decisão humana
    # --------------------------------------------------------
    st.markdown("### 5️⃣ Decisão final (humana)")
    acao = st.radio(
        "Ação escolhida:",
        [
            "CONCENTRAR (6–8 listas)",
            "EQUILIBRAR (8–10 listas)",
            "EXPANDIR COM CRITÉRIO (10–12 listas)",
            "SEGURAR / NÃO ESCALAR",
        ],
        key="CHECKLIST_ACAO",
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Síntese
    # --------------------------------------------------------
    st.markdown("### 🧾 Síntese da decisão")
    st.write(
        {
            "Estrada OK": estrada_ok,
            "Regime": regime,
            "Eixo": eixo,
            "Nocivos concentrados": nocivos,
            "Ação escolhida": acao,
        }
    )

    st.success(
        "Checklist concluído. "
        "A decisão da rodada está FECHADA aqui. "
        "Prossiga para o Modo 6 e execução."
    )


# ============================================================
# PAINEL — 🧭 MODO GUIADO OFICIAL — PREDICTCARS
# ============================================================
if painel == "🧭 Modo Guiado Oficial — PredictCars":

    st.markdown("## 🧭 Modo Guiado Oficial — PredictCars")
    st.caption(
        "Guia operacional único · uso diário · contrato de uso do sistema.\n"
        "Não executa, não calcula, não decide — apenas orienta a sequência correta."
    )

    st.markdown("---")

    st.markdown("""
🧭 **MODO GUIADO OFICIAL — CONTRATO OPERACIONAL**

Este painel descreve **COMO o PredictCars deve ser usado**.
Ele existe para evitar decisões fora de ordem e misturas perigosas
entre leitura, decisão, execução e aprendizado.

━━━━━━━━━━━━━━━━━━━━
🔵 **AGORA — DECIDIR E JOGAR**
━━━━━━━━━━━━━━━━━━━━

**1️⃣ ENTRADA**
- 📁 Carregar Histórico (Arquivo ou Colar)

**2️⃣ EIXO 1 — ESTRUTURA DO AMBIENTE**
*(saúde da estrada · não números)*

Painéis:
- 🛰️ Sentinelas — k*
- 🧭 Monitor de Risco — k & k*
- 📡 Painel de Ruído Condicional
- 📉 Painel de Divergência S6 vs MC

Pergunta respondida:
- O ambiente permite ataque?

---

**3️⃣ EIXO 2 — MOMENTO & ANTECIPAÇÃO**
*(ritmo do alvo + evidência recente)*

Painéis:
- 🔁 Replay LIGHT
- 🔁 Replay ULTRA
- 🧪 Replay Curto — Expectativa 1–3 Séries
- 📊 V16 Premium — Backtest Rápido do Pacote (N=60)

Pergunta respondida:
- O momento favorece agir agora?

---

**4️⃣ DECISÃO ÚNICA (HUMANA)**
*(registrada · sem retorno)*

Painel:
- 🧭 Checklist Operacional — Decisão (AGORA)

Aqui você define:
- atacar ou não
- concentrar, equilibrar ou expandir
- volume de listas

📌 **Depois disso, não se volta atrás.**

---

**5️⃣ MOTOR**
- 🛣️ Pipeline V14-FLEX ULTRA
- ⚙️ Modo TURBO++ HÍBRIDO
- ⚙️ Modo TURBO++ ULTRA

---

**6️⃣ EXECUÇÃO**
- 🎯 Modo 6 Acertos — Execução
- 🧪 Testes de Confiabilidade REAL
- 📘 Relatório Final
- 🔥 Mandar Bala

━━━━━━━━━━━━━━━━━━━━
🟣 **EXTENSÃO CONDICIONAL — MODO ESPECIAL**
━━━━━━━━━━━━━━━━━━━━

Use **somente após** concluir o fluxo acima.

- 🔵 MODO ESPECIAL — Evento Condicionado
- Atua sobre pacotes já gerados
- Não cria listas novas
- Útil apenas para eventos únicos

━━━━━━━━━━━━━━━━━━━━
🟢 **DEPOIS — APRENDER**
━━━━━━━━━━━━━━━━━━━━

Painéis:
- 📊 V16 Premium — ANTI-EXATO | Passageiros Nocivos
- 📊 EXATO / ECO / Regime

📌 Aprendizado **somente para a próxima rodada**.

━━━━━━━━━━━━━━━━━━━━
🧱 **OS 3 EIXOS DO SISTEMA**
━━━━━━━━━━━━━━━━━━━━

- **Eixo 1** — Estrutura das Listas  
- **Eixo 2** — Momento & Antecipação  
- **Eixo 3** — Aprendizado  

━━━━━━━━━━━━━━━━━━━━
📜 **REGRA FINAL**
━━━━━━━━━━━━━━━━━━━━

A decisão acontece **ANTES**.  
O aprendizado acontece **DEPOIS**.  
**Nunca ao mesmo tempo.**
""")

    st.success(
        "Modo Guiado carregado com sucesso.\n"
        "Este painel é o contrato oficial de uso do PredictCars."
    )

# ============================================================
# <<< FIM — PAINEL 🧭 MODO GUIADO OFICIAL — PREDICTCARS
# ============================================================





# ============================================================
# ROTEADOR V16 PREMIUM — EXECUÇÃO DOS PAINÉIS (DEFINITIVO)
# ============================================================

# ------------------------------------------------------------
# ORIENTAÇÃO / USO
# ------------------------------------------------------------
if painel == "🧭 Modo Guiado Oficial — PredictCars":
    st.markdown("## 🧭 Modo Guiado Oficial — PredictCars")
    st.info(
        "Este painel apenas orienta o uso correto do sistema.\n"
        "Siga a sequência indicada no menu."
    )
    st.stop()

# ------------------------------------------------------------
# DECISÃO OPERACIONAL (AGORA)
# ------------------------------------------------------------
if painel == "🧭 Checklist Operacional — Decisão (AGORA)":
    st.markdown("## 🧭 Checklist Operacional — Decisão (AGORA)")
    st.caption(
        "Checklist obrigatório ANTES do Modo 6 / Mandar Bala.\n"
        "Não calcula, não cria listas, não decide automaticamente."
    )

    st.markdown("---")

    st.markdown("### 1️⃣ Estrada permite ataque?")
    estrada_ok = st.radio(
        "Resultado da leitura da estrada:",
        ["SIM", "NÃO"],
        horizontal=True,
    )

    st.markdown("### 2️⃣ Regime jogável?")
    regime = st.radio(
        "Regime identificado:",
        ["OURO", "PRATA", "RUIM"],
        horizontal=True,
    )

    st.markdown("### 3️⃣ Existe eixo claro nas listas?")
    eixo = st.radio(
        "Eixo identificado:",
        ["SIM", "NÃO"],
        horizontal=True,
    )

    st.markdown("### 4️⃣ Nocivos concentrados nas mesmas listas?")
    nocivos = st.radio(
        "Nocivos:",
        ["SIM", "NÃO"],
        horizontal=True,
    )

    st.markdown("### 5️⃣ Decisão final (humana)")
    acao = st.radio(
        "Ação escolhida:",
        [
            "CONCENTRAR (6–8 listas)",
            "EQUILIBRAR (8–10 listas)",
            "EXPANDIR COM CRITÉRIO (10–12 listas)",
            "SEGURAR / NÃO ESCALAR",
        ],
    )

    st.markdown("---")
    st.markdown("### 🧾 Síntese da decisão")
    st.write(
        {
            "Estrada OK": estrada_ok,
            "Regime": regime,
            "Eixo": eixo,
            "Nocivos concentrados": nocivos,
            "Ação escolhida": acao,
        }
    )

    st.success(
        "Checklist concluído. "
        "A decisão da rodada está FECHADA aqui. "
        "Prossiga para o Modo 6 e execução."
    )
    st.stop()

# ------------------------------------------------------------
# EIXO 2 — MOMENTO & ANTECIPAÇÃO
# ------------------------------------------------------------
if painel == "📊 V16 Premium — Backtest Rápido do Pacote (N=60)":
    st.markdown("## 📊 Backtest Rápido do Pacote (N=60)")
    st.caption(
        "Avaliação observacional do pacote atual.\n"
        "Não decide. Apenas informa."
    )
    st.stop()

# ------------------------------------------------------------
# RITMO DO ALVO (SUPORTE)
# ------------------------------------------------------------
if painel == "🧪 Replay Curto — Expectativa 1–3 Séries":
    st.markdown("## 🧪 Replay Curto — Expectativa 1–3 Séries")
    st.stop()

if painel == "⏱️ Duração da Janela — Análise Histórica":
    st.markdown("## ⏱️ Duração da Janela — Análise Histórica")
    st.stop()

# ------------------------------------------------------------
# V16 PREMIUM — APRENDIZADO (DEPOIS)
# ------------------------------------------------------------

if painel == "🧠 Laudo Operacional V16":
    # ROTA DUPLICADA DESATIVADA (há um painel Laudo V16 completo em outro ponto do app)
    # Mantemos apenas UM ponto de execução do Laudo para evitar dupla execução/deriva.
    st.stop()


if painel == "📊 V16 Premium — Erro por Regime (Retrospectivo)":
    v16_painel_erro_por_regime_retrospectivo()
    st.stop()

if painel == "📊 V16 Premium — EXATO por Regime (Proxy)":
    v16_painel_exato_por_regime_proxy()
    st.stop()

if painel == "📊 V16 Premium — PRÉ-ECO → ECO (Persistência & Continuidade)":
    v16_painel_pre_eco_persistencia_continuidade()
    st.stop()

if painel == "📊 V16 Premium — Passageiros Recorrentes em ECO (Interseção)":
    v16_painel_passageiros_recorrentes_eco_intersecao()
    st.stop()

if painel == "🎯 Compressão do Alvo — Observacional (V16)":
    v16_painel_compressao_alvo()
    st.stop()

if painel == "🔮 V16 Premium Profundo — Diagnóstico & Calibração":
    v16_painel_premium_profundo()
    st.stop()

# ============================================================
# FIM DO ROTEADOR V16 PREMIUM — EXECUÇÃO DOS PAINÉIS
# ============================================================
