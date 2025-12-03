# Predict Cars V14-FLEX REPLAY
# App completo com:
# - Pipeline V14-FLEX (IPF → IPO → S6 → Modo E)
# - Monitor de Risco (k / k*)
# - Modo TURBO++ — Painel Completo
# - Replay Automático
# - Testes de Confiabilidade (empírico)
# - Painel: Séries Alternativas Inteligentes V14-FLEX
#   (Modo Automático + Modo Avançado por Confiabilidade)

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Tuple

# ============================================================
# CONFIGURAÇÃO BÁSICA
# ============================================================

st.set_page_config(
    page_title="Predict Cars V14-FLEX REPLAY",
    layout="wide",
)

st.markdown(
    """
# Predict Cars V14-FLEX REPLAY
Versão FLEX: número variável de passageiros + modo replay automático + validação empírica.
"""
)

# ============================================================
# CONSTANTES E FUNÇÕES AUXILIARES
# ============================================================

NUM_MIN = 1
NUM_MAX = 60


def _coerce_int(x: Any) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return 0


# ------------------------------------------------------------
# PARSER FLEX - CSV
# ------------------------------------------------------------

def preparar_historico_V14(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Converte o CSV cru para o formato padrão do V14-FLEX:

    colunas:
    - id (str)
    - passageiros (list[int])
    - k (int)
    - n_passageiros (int)
    - idx_numeric (int)
    """
    linhas = []
    for i, row in enumerate(df_raw.itertuples(index=False)):  # type: ignore
        valores = list(row)
        if not valores:
            continue

        s0 = str(valores[0]).strip()
        if s0 and not s0.isdigit():
            id_serie = s0
            resto = valores[1:]
        else:
            id_serie = f"C{i+1}"
            resto = valores

        if len(resto) < 2:
            continue

        k_val = _coerce_int(resto[-1])
        passageiros = [_coerce_int(x) for x in resto[:-1]]
        passageiros = [p for p in passageiros if p > 0]

        if not passageiros:
            continue

        linhas.append(
            {
                "id": id_serie,
                "passageiros": passageiros,
                "k": k_val,
                "n_passageiros": len(passageiros),
            }
        )

    df = pd.DataFrame(linhas)
    if not df.empty:
        df["idx_numeric"] = range(1, len(df) + 1)
    return df


# ------------------------------------------------------------
# PARSER FLEX - TEXTO
# ------------------------------------------------------------

def preparar_historico_de_texto(texto: str) -> pd.DataFrame:
    """
    Converte histórico colado em texto em DataFrame padrão V14-FLEX.

    Aceita linhas do tipo:
    C1;41;5;4;52;30;33;0
    41;5;4;52;30;33;0
    41,5,4,52,30,33,0
    41 5 4 52 30 33 0
    """
    linhas = []
    for i, raw_line in enumerate(texto.splitlines()):
        linha = raw_line.strip()
        if not linha:
            continue

        if ";" in linha:
            partes = [p.strip() for p in linha.split(";")]
        elif "," in linha:
            partes = [p.strip() for p in linha.split(",")]
        else:
            partes = [p.strip() for p in linha.split()]

        if not partes:
            continue

        s0 = partes[0]
        if s0 and not s0.isdigit():
            id_serie = s0
            resto = partes[1:]
        else:
            id_serie = f"C{i+1}"
            resto = partes

        if len(resto) < 2:
            continue

        k_val = _coerce_int(resto[-1])
        passageiros = [_coerce_int(x) for x in resto[:-1]]
        passageiros = [p for p in passageiros if p > 0]
        if not passageiros:
            continue

        linhas.append(
            {
                "id": id_serie,
                "passageiros": passageiros,
                "k": k_val,
                "n_passageiros": len(passageiros),
            }
        )

    df = pd.DataFrame(linhas)
    if not df.empty:
        df["idx_numeric"] = range(1, len(df) + 1)
    return df


# ============================================================
# MÓDULO DE RISCO — k & k*
# ============================================================

def avaliar_risco_k(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Avalia risco histórico (k) e risco preditivo agregado (k*).
    Retorna (desc_k, desc_k_star) em Markdown.
    """
    if df.empty or "k" not in df.columns:
        return (
            "⚠️ k histórico da série alvo\nDados insuficientes para avaliar.",
            "⚡ k* (sentinela preditivo)\nDados insuficientes para projetar risco.",
        )

    # k histórico da última série
    k_ultimo = int(df.iloc[-1]["k"])
    if k_ultimo <= 0:
        desc_k = (
            "⚠️ k histórico da série alvo\n"
            "🟢 Ambiente estável — previsão em regime normal."
        )
    elif k_ultimo == 1:
        desc_k = (
            "⚠️ k histórico da série alvo\n"
            "🟡 Ambiente em atenção — sinais de pré-ruptura local."
        )
    else:
        desc_k = (
            "⚠️ k histórico da série alvo\n"
            "🔴 Ambiente crítico — trecho turbulento da estrada."
        )

    # k* preditivo baseado na frequência de k>0 nas últimas N séries
    n_janela = min(50, len(df))
    sub = df.tail(n_janela)
    proporcao_eventos = float((sub["k"] > 0).mean())
    risco_pct = int(round(100 * proporcao_eventos))

    if risco_pct <= 15:
        desc_k_star = (
            "⚡ k* (sentinela preditivo TURBO++)\n"
            f"🟢 k*: Ambiente tende a permanecer estável (risco ≈ {risco_pct}%)."
        )
    elif risco_pct <= 40:
        desc_k_star = (
            "⚡ k* (sentinela preditivo TURBO++)\n"
            f"🟡 k*: Ambiente com ruído moderado (risco ≈ {risco_pct}%)."
        )
    else:
        desc_k_star = (
            "⚡ k* (sentinela preditivo TURBO++)\n"
            f"🔴 k*: Ambiente com alta turbulência (risco ≈ {risco_pct}%)."
        )

    return desc_k, desc_k_star


# ============================================================
# PIPELINE V14-FLEX — IPF, IPO, S6, MODO E
# ============================================================

def extrair_contexto(df: pd.DataFrame, idx_alvo: int, janela: int = 30) -> pd.DataFrame:
    """
    Extrai janela de contexto antes da série alvo (não inclui a série alvo).
    idx_alvo é 1-based.
    """
    if df.empty:
        return df

    pos = max(0, min(len(df) - 1, idx_alvo - 1))
    inicio = max(0, pos - janela)
    return df.iloc[inicio:pos].copy()


def gerar_leque_original(contexto: pd.DataFrame) -> List[int]:
    """
    IPF simplificado: conta frequências dos passageiros na janela de contexto
    e monta um leque ORIGINAL.
    """
    if contexto.empty or "passageiros" not in contexto.columns:
        return []

    cont: Dict[int, int] = {}
    for passageiros in contexto["passageiros"]:
        for p in passageiros:
            if NUM_MIN <= p <= NUM_MAX:
                cont[p] = cont.get(p, 0) + 1

    if not cont:
        return []

    ordenado = sorted(cont.items(), key=lambda kv: (-kv[1], kv[0]))
    numeros = [n for n, _ in ordenado[:25]]
    return sorted(set(numeros))


def gerar_leque_corrigido(contexto: pd.DataFrame, leque_original: List[int]) -> List[int]:
    """
    IPO simplificado: reforça números dos últimos trechos,
    removendo os muito raros do leque original.
    """
    if contexto.empty or not leque_original:
        return leque_original

    recente = contexto.tail(min(10, len(contexto)))
    cont: Dict[int, int] = {}
    for passageiros in recente["passageiros"]:
        for p in passageiros:
            if NUM_MIN <= p <= NUM_MAX:
                cont[p] = cont.get(p, 0) + 1

    filtrado = [n for n in leque_original if cont.get(n, 0) >= 1]

    if not filtrado:
        filtrado = leque_original

    return sorted(set(filtrado))


def gerar_leque_misto(leque_original: List[int], leque_corrigido: List[int]) -> List[int]:
    """
    S6 Profundo simplificado: união ORIGINAL + CORRIGIDO.
    """
    return sorted(set(leque_original) | set(leque_corrigido))


def selecionar_serie_final_modo_E(leque_final: List[int]) -> List[int]:
    """
    Seleciona a série final (6 passageiros) a partir do Leque Final.

    Modo E — MIX Inteligente (A + B + D):
    - recorte central
    - espalhamento pelo leque
    - remoção de extremos e ruídos óbvios
    """
    if not leque_final:
        return []

    numeros = sorted(set(leque_final))
    n = len(numeros)

    if n <= 6:
        return numeros

    # Recorte central se houver muitos números
    if n > 10:
        corte = max(1, int(0.2 * n))
        centro = numeros[corte:-corte]
        if centro:
            numeros = centro
            n = len(numeros)

    # índices relativos para espalhar no leque
    indices_relativos = [0.12, 0.30, 0.48, 0.62, 0.78, 0.90]
    escolhidos: List[int] = []
    usados = set()

    for rel in indices_relativos:
        idx = int(round(rel * (n - 1)))
        idx = max(0, min(n - 1, idx))
        v = numeros[idx]
        if v not in usados:
            escolhidos.append(v)
            usados.add(v)

    # completar se faltar
    if len(escolhidos) < 6:
        for v in numeros:
            if v not in usados:
                escolhidos.append(v)
                usados.add(v)
                if len(escolhidos) == 6:
                    break

    return sorted(escolhidos)


def executar_pipeline_v14_flex(
    df: pd.DataFrame,
    idx_alvo: int,
    janela: int = 30,
) -> Dict[str, Any]:
    """
    Executa o pipeline V14-FLEX completo para uma série alvo (idx_alvo, 1-based).
    Retorna um dict com todos os elementos relevantes.
    """
    if df.empty:
        return {}

    idx_alvo = int(idx_alvo)
    pos = max(0, min(len(df) - 1, idx_alvo - 1))
    alvo_row = df.iloc[pos]

    contexto = extrair_contexto(df, idx_alvo, janela)
    leque_original = gerar_leque_original(contexto)
    leque_corrigido = gerar_leque_corrigido(contexto, leque_original)
    leque_misto = gerar_leque_misto(leque_original, leque_corrigido)
    leque_final = leque_misto.copy()
    serie_final = selecionar_serie_final_modo_E(leque_final)

    desc_k, desc_k_star = avaliar_risco_k(df.iloc[: pos + 1])

    return {
        "id_alvo": alvo_row["id"],
        "passageiros_alvo": alvo_row["passageiros"],
        "k_alvo": int(alvo_row["k"]),
        "leque_original": leque_original,
        "leque_corrigido": leque_corrigido,
        "leque_misto": leque_misto,
        "leque_final": leque_final,
        "serie_final": serie_final,
        "desc_k": desc_k,
        "desc_k_star": desc_k_star,
    }


# ============================================================
# MÓDULO — SÉRIES ALTERNATIVAS INTELIGENTES V14-FLEX
# ============================================================

def estimar_confiabilidade_heuristica(tipo: str, tamanho_leque: int) -> Dict[str, Any]:
    """
    Estima confiabilidade de forma heurística com base no tipo de série
    e no tamanho do leque final.
    """
    base_map = {
        "principal": 0.75,
        "conservadora": 0.80,
        "intermediaria": 0.70,
        "agressiva": 0.60,
        "cluster": 0.68,
        "asb": 0.70,
    }
    base = base_map.get(tipo, 0.65)

    # ajuste por tamanho de leque
    if tamanho_leque <= 15:
        base += 0.05
    elif tamanho_leque >= 30:
        base -= 0.05

    prob = max(0.40, min(0.95, base))

    if prob >= 0.78:
        nivel = "Alta"
        faixa_acertos = "3–5 acertos prováveis em cenários típicos."
    elif prob >= 0.65:
        nivel = "Intermediária"
        faixa_acertos = "2–4 acertos prováveis."
    else:
        nivel = "Baixa"
        faixa_acertos = "1–3 acertos prováveis."

    return {
        "prob": prob,
        "nivel": nivel,
        "faixa_acertos": faixa_acertos,
    }


def _escolher_seis(numeros: List[int], indices_relativos: List[float]) -> List[int]:
    """
    Escolhe 6 números de um leque usando índices relativos (0–1).
    """
    if not numeros:
        return []

    nums = sorted(set(numeros))
    n = len(nums)
    if n <= 6:
        return nums

    usados = set()
    escolhidos: List[int] = []
    for rel in indices_relativos:
        idx = int(round(rel * (n - 1)))
        idx = max(0, min(n - 1, idx))
        v = nums[idx]
        if v not in usados:
            escolhidos.append(v)
            usados.add(v)

    if len(escolhidos) < 6:
        for v in nums:
            if v not in usados:
                escolhidos.append(v)
                usados.add(v)
                if len(escolhidos) == 6:
                    break

    return sorted(escolhidos)


def gerar_series_alternativas_inteligentes(
    leque_final: List[int],
    serie_principal: List[int],
) -> List[Dict[str, Any]]:
    """
    Gera séries alternativas A–E a partir do leque_final e da série principal.

    Retorna lista de dicts com:
    - nome
    - tipo
    - serie (list[int])
    - descricao
    - confiabilidade (dict com prob, nivel, faixa_acertos)
    """
    if not leque_final:
        return []

    nums = sorted(set(leque_final))
    tam = len(nums)

    # Principal (Modo E)
    conf_principal = estimar_confiabilidade_heuristica("principal", tam)
    series: List[Dict[str, Any]] = [
        {
            "nome": "Série Principal (Modo E)",
            "tipo": "principal",
            "serie": serie_principal,
            "descricao": "Equilíbrio geral do leque — MIX Inteligente (A+B+D).",
            "confiabilidade": conf_principal,
        }
    ]

    # A) Conservadora — foco ainda mais central
    if tam > 10:
        corte = max(1, int(0.25 * tam))
        centro = nums[corte:-corte] or nums
    else:
        centro = nums

    serie_A = _escolher_seis(centro, [0.18, 0.30, 0.42, 0.58, 0.70, 0.82])
    conf_A = estimar_confiabilidade_heuristica("conservadora", tam)
    series.append(
        {
            "nome": "Série A — Conservadora",
            "tipo": "conservadora",
            "serie": serie_A,
            "descricao": "Núcleo mais central do leque, priorizando estabilidade.",
            "confiabilidade": conf_A,
        }
    )

    # B) Intermediária — variação suave em torno do leque inteiro
    indices_B = [0.12, 0.28, 0.44, 0.60, 0.76, 0.90]
    serie_B = _escolher_seis(nums, indices_B)
    conf_B = estimar_confiabilidade_heuristica("intermediaria", tam)
    series.append(
        {
            "nome": "Série B — Intermediária Estruturada",
            "tipo": "intermediaria",
            "serie": serie_B,
            "descricao": "Combina estrutura central com abertura para faixas vizinhas.",
            "confiabilidade": conf_B,
        }
    )

    # C) Agressiva — usa bordas e meio
    indices_C = [0.0, 0.18, 0.36, 0.64, 0.82, 1.0]
    serie_C = _escolher_seis(nums, indices_C)
    conf_C = estimar_confiabilidade_heuristica("agressiva", tam)
    series.append(
        {
            "nome": "Série C — Agressiva",
            "tipo": "agressiva",
            "serie": serie_C,
            "descricao": "Explora bordas e zonas menos óbvias do leque para cenários extremos.",
            "confiabilidade": conf_C,
        }
    )

    # D) Cluster Puro — foco na faixa mais densa
    if tam >= 8:
        inicio = int(0.30 * tam)
        fim = int(0.70 * tam)
        cluster = nums[inicio:fim] or nums
    else:
        cluster = nums

    serie_D = _escolher_seis(cluster, [0.05, 0.25, 0.45, 0.60, 0.80, 0.95])
    conf_D = estimar_confiabilidade_heuristica("cluster", tam)
    series.append(
        {
            "nome": "Série D — Cluster Puro",
            "tipo": "cluster",
            "serie": serie_D,
            "descricao": "Foca na faixa mais densa do leque, simulando o cluster dominante.",
            "confiabilidade": conf_D,
        }
    )

    # E) Anti-SelfBias — desloca padrão central
    indices_E = [0.10, 0.32, 0.40, 0.55, 0.68, 0.88]
    serie_E = _escolher_seis(nums, indices_E)
    conf_E = estimar_confiabilidade_heuristica("asb", tam)
    series.append(
        {
            "nome": "Série E — Anti-SelfBias",
            "tipo": "asb",
            "serie": serie_E,
            "descricao": "Quebra padrões óbvios do leque para reduzir enviesamento.",
            "confiabilidade": conf_E,
        }
    )

    return series


# ============================================================
# PAINEL 1 — HISTÓRICO — ENTRADA
# ============================================================

def painel_historico_entrada() -> None:
    st.markdown("## 📥 Histórico — Entrada")

    df = st.session_state.get("df")
    if df is not None and not df.empty:
        st.success("Histórico já carregado na sessão.")
        st.dataframe(df[["id", "passageiros", "k", "n_passageiros"]])

    opc = st.radio(
        "Como deseja carregar o histórico?",
        ["Enviar arquivo CSV", "Copiar e colar o histórico"],
    )

    if opc == "Enviar arquivo CSV":
        file = st.file_uploader("Selecione o arquivo CSV:", type=["csv"])
        if file is not None:
            try:
                df_raw = pd.read_csv(file, header=None, sep=None, engine="python")
                df = preparar_historico_V14(df_raw)
                st.session_state["df"] = df
                st.success("Histórico carregado com sucesso!")
                st.dataframe(df[["id", "passageiros", "k", "n_passageiros"]])
            except Exception as e:
                st.error(f"Erro ao carregar CSV: {e}")
    else:
        texto = st.text_area(
            "Cole o histórico aqui (uma série por linha):",
            height=200,
            placeholder="Exemplo:\nC1;41;5;4;52;30;33;0\nC2;9;39;37;49;43;41;1\n...",
        )
        if st.button("Processar histórico colado"):
            if not texto.strip():
                st.warning("Cole algum conteúdo antes de processar.")
            else:
                try:
                    df = preparar_historico_de_texto(texto)
                    if df.empty:
                        st.error("Não foi possível interpretar o histórico.")
                    else:
                        st.session_state["df"] = df
                        st.success("Histórico carregado com sucesso!")
                        st.dataframe(df[["id", "passageiros", "k", "n_passageiros"]])
                except Exception as e:
                    st.error(f"Erro ao processar texto: {e}")


# ============================================================
# PAINEL 2 — PIPELINE V14-FLEX (TURBO++)
# ============================================================

def painel_pipeline_v14_flex() -> None:
    st.markdown("## 🔍 Pipeline V14-FLEX (TURBO++)")

    df = st.session_state.get("df")
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Seleção da série alvo")
        idx_min = 1
        idx_max = len(df)
        idx_alvo = st.number_input(
            "Índice alvo (1 = primeira série carregada):",
            min_value=idx_min,
            max_value=idx_max,
            value=idx_max,
            step=1,
        )
        alvo_row = df.iloc[int(idx_alvo) - 1]
        st.markdown(f"**ID alvo:** {alvo_row['id']}")
        st.markdown(f"**Passageiros alvo:** {alvo_row['passageiros']}")
        st.markdown(f"**k alvo:** {int(alvo_row['k'])}")

    with col2:
        st.write("### Configuração da janela de contexto")
        janela = st.slider(
            "Janela de contexto (séries anteriores usadas no IPF/IPO):",
            min_value=10,
            max_value=100,
            value=min(30, len(df) - 1 if len(df) > 1 else 10),
            step=1,
        )

    if st.button("Executar Pipeline V14-FLEX TURBO++"):
        with st.spinner("Rodando pipeline V14-FLEX TURBO++..."):
            resultado = executar_pipeline_v14_flex(df, idx_alvo, janela=janela)

        if not resultado:
            st.error("Não foi possível executar o pipeline.")
            return

        st.markdown("### Estrutura dos Leques")

        st.write("Leque ORIGINAL (IPF bruto)")
        st.code(" ".join(str(x) for x in resultado["leque_original"]), language="text")

        st.write("🔧 Leque CORRIGIDO (IPO simplificado)")
        st.code(" ".join(str(x) for x in resultado["leque_corrigido"]), language="text")

        st.write("🧬 S6 Profundo — Leque MISTO (achado e ranqueado)")
        st.code(" ".join(str(x) for x in resultado["leque_misto"]), language="text")

        st.write("🎯 Núcleo TURBO++ FLEX (previsão bruta do motor)")
        st.code(" ".join(str(x) for x in resultado["leque_final"]), language="text")

        st.markdown("---")
        st.markdown(resultado["desc_k"])
        st.markdown("")
        st.markdown(resultado["desc_k_star"])

        st.markdown("---")
        st.markdown("### 🎯 Previsão Final TURBO++ FLEX (Modo E)")
        serie_final = resultado["serie_final"]
        if serie_final:
            st.code(" ".join(str(x) for x in serie_final), language="text")
            st.success("Série final gerada com sucesso.")
        else:
            st.warning("Não foi possível gerar a série final a partir do leque.")

        st.session_state["ultimo_pipeline"] = {
            "idx_alvo": int(idx_alvo),
            "resultado": resultado,
        }


# ============================================================
# PAINEL 3 — MONITOR DE RISCO (k & k*)
# ============================================================

def painel_monitor_risco() -> None:
    st.markdown("## 🚨 Monitor de Risco (k & k*)")

    df = st.session_state.get("df")
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        return

    desc_k, desc_k_star = avaliar_risco_k(df)

    st.markdown("### Visão geral do risco")
    st.markdown(desc_k)
    st.markdown("")
    st.markdown(desc_k_star)

    st.markdown("---")
    st.markdown("### Distribuição de k no histórico")
    st.dataframe(df[["id", "k"]].reset_index(drop=True))


# ============================================================
# PAINEL 4 — MODO TURBO++ — PAINEL COMPLETO
# ============================================================

def painel_modo_turbo_completo() -> None:
    st.markdown("## 🚀 Modo TURBO++ — Painel Completo")

    df = st.session_state.get("df")
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        return

    ultimo = st.session_state.get("ultimo_pipeline")
    if not ultimo:
        st.info(
            "Execute primeiro o painel '🔍 Pipeline V14-FLEX (TURBO++)' "
            "para popular este modo."
        )
        return

    resultado = ultimo["resultado"]

    st.markdown(f"### Série alvo: **{resultado['id_alvo']}**")
    st.markdown(f"Passageiros alvo: **{resultado['passageiros_alvo']}**")
    st.markdown(f"k alvo: **{resultado['k_alvo']}**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Leque ORIGINAL (IPF)")
        st.code(" ".join(str(x) for x in resultado["leque_original"]), language="text")

        st.markdown("#### Leque CORRIGIDO (IPO)")
        st.code(" ".join(str(x) for x in resultado["leque_corrigido"]), language="text")

    with col2:
        st.markdown("#### Leque MISTO (S6 Profundo)")
        st.code(" ".join(str(x) for x in resultado["leque_misto"]), language="text")

        st.markdown("#### Núcleo TURBO++ FLEX (Leque final)")
        st.code(" ".join(str(x) for x in resultado["leque_final"]), language="text")

    st.markdown("---")
    st.markdown("### 🎯 Previsão Final TURBO++ FLEX (Modo E)")
    serie_final = resultado["serie_final"]
    if serie_final:
        st.code(" ".join(str(x) for x in serie_final), language="text")
    else:
        st.warning("Série final não disponível.")

    st.markdown("---")
    st.markdown("### Contexto de risco")
    st.markdown(resultado["desc_k"])
    st.markdown("")
    st.markdown(resultado["desc_k_star"])


# ============================================================
# PAINEL 5 — MODO REPLAY AUTOMÁTICO DO HISTÓRICO
# ============================================================

def calcular_acertos(p_real: List[int], p_prev: List[int]) -> int:
    return len(set(p_real) & set(p_prev))


def painel_modo_replay() -> None:
    st.markdown("## 📅 Modo Replay Automático do Histórico")

    df = st.session_state.get("df")
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        return

    st.markdown(
        "Neste modo, o V14-FLEX REPLAY simula previsões ao longo do histórico "
        "e mede os acertos de forma empírica."
    )

    if len(df) < 3:
        st.warning("Histórico muito curto para replay automático.")
        return

    idx_min = 2
    idx_max = len(df) - 1

    col1, col2 = st.columns(2)
    with col1:
        inicio = st.number_input(
            "Índice inicial para replay (previsão para a próxima série):",
            min_value=idx_min,
            max_value=idx_max,
            value=idx_min,
            step=1,
        )
    with col2:
        fim = st.number_input(
            "Índice final para replay:",
            min_value=inicio,
            max_value=idx_max,
            value=idx_max,
            step=1,
        )

    if st.button("Executar Replay Automático"):
        resultados = []
        total = 0

        with st.spinner("Executando replay ao longo do histórico..."):
            for idx_alvo in range(int(inicio), int(fim) + 1):
                # previsão para série idx_alvo usando contexto até idx_alvo-1
                resultado = executar_pipeline_v14_flex(df, idx_alvo - 1)
                if not resultado:
                    continue

                real_row = df.iloc[idx_alvo - 1]
                passageiros_reais = list(real_row["passageiros"])
                serie_prev = resultado["serie_final"]
                acertos = calcular_acertos(passageiros_reais, serie_prev)

                resultados.append(
                    {
                        "idx_prev": idx_alvo - 1,
                        "idx_real": idx_alvo,
                        "id_prev": resultado["id_alvo"],
                        "id_real": real_row["id"],
                        "prev": serie_prev,
                        "real": passageiros_reais,
                        "acertos": acertos,
                    }
                )
                total += 1

        if not resultados:
            st.error("Replay não gerou resultados.")
            return

        df_res = pd.DataFrame(resultados)
        st.session_state["replay_resultados"] = df_res

        st.success(f"Replay concluído com {total} previsões.")
        st.markdown("### Amostra de resultados do Replay")
        st.dataframe(df_res.head(50))

        st.markdown("---")
        st.markdown("### Estatísticas rápidas")
        media_acertos = float(df_res["acertos"].mean())
        st.markdown(f"**Média de acertos por série:** {media_acertos:.2f}")

        for n in [2, 3, 4, 5, 6]:
            pct = 100 * float((df_res["acertos"] >= n).mean())
            st.markdown(f"Séries com **≥{n} acertos**: {pct:.1f}%")


# ============================================================
# PAINEL 6 — TESTES DE CONFIABILIDADE (EMPÍRICO)
# ============================================================

def painel_testes_confiabilidade() -> None:
    st.markdown("## 🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)")

    df_replay = st.session_state.get("replay_resultados")
    if df_replay is None or df_replay.empty:
        st.info(
            "Execute primeiro o '📅 Modo Replay Automático do Histórico' "
            "para gerar dados de backtest."
        )
        return

    st.markdown("### Visão geral dos resultados do Replay")
    st.dataframe(df_replay)

    st.markdown("---")
    st.markdown("### Métricas de confiabilidade empírica")

    media_acertos = float(df_replay["acertos"].mean())
    st.markdown(f"**Média de acertos por série:** {media_acertos:.2f}")

    for n in [2, 3, 4, 5, 6]:
        pct = 100 * float((df_replay["acertos"] >= n).mean())
        st.markdown(f"- Séries com **≥{n} acertos**: {pct:.1f}%")

    if media_acertos < 2.0:
        nivel = "Baixa"
        cor = "🔴"
    elif media_acertos < 3.5:
        nivel = "Intermediária"
        cor = "🟡"
    else:
        nivel = "Alta"
        cor = "🟢"

    st.markdown("---")
    st.markdown(f"### {cor} Nível de confiabilidade empírica: **{nivel}**")
    st.markdown(
        "Este painel usa apenas os resultados do Replay (backtest interno) como base "
        "para a confiabilidade. Módulos QDS / Backtest avançado / Monte Carlo "
        "podem ser acoplados futuramente em cima destas métricas."
    )


# ============================================================
# PAINEL 7 — SÉRIES ALTERNATIVAS INTELIGENTES V14-FLEX
# ============================================================

def painel_series_alternativas_inteligentes() -> None:
    st.markdown("## 🎛 Séries Alternativas Inteligentes V14-FLEX")

    df = st.session_state.get("df")
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        return

    ultimo = st.session_state.get("ultimo_pipeline")
    if not ultimo or "resultado" not in ultimo:
        st.info(
            "Execute primeiro o painel '🔍 Pipeline V14-FLEX (TURBO++)' "
            "para gerar o leque final e a série principal."
        )
        return

    resultado = ultimo["resultado"]
    leque_final = resultado.get("leque_final", [])
    serie_principal = resultado.get("serie_final", [])

    if not leque_final or not serie_principal:
        st.warning("Não há leque final ou série principal disponíveis.")
        return

    series_alt = gerar_series_alternativas_inteligentes(leque_final, serie_principal)
    if not series_alt:
        st.warning("Não foi possível gerar séries alternativas.")
        return

    modo = st.radio(
        "Selecione o modo de visualização:",
        ["🔹 Modo Automático", "🧪 Modo Avançado por Confiabilidade"],
    )

    st.markdown("### Contexto atual da estrada")
    st.markdown(resultado.get("desc_k", ""))
    st.markdown("")
    st.markdown(resultado.get("desc_k_star", ""))

    st.markdown("---")

    if modo == "🔹 Modo Automático":
        st.markdown("### Séries sugeridas automaticamente pelo motor V14-FLEX")

        for s in series_alt:
            conf = s["confiabilidade"]
            serie = s["serie"]
            if not serie:
                continue

            with st.expander(f"{s['nome']}"):
                st.code(" ".join(str(x) for x in serie), language="text")
                st.markdown(f"**Estilo:** {s['descricao']}")
                st.markdown(
                    f"**Confiabilidade estimada:** {conf['nivel']} "
                    f"(~{conf['prob']*100:.0f}%)"
                )
                st.markdown(f"**Acertos prováveis:** {conf['faixa_acertos']}")
                st.markdown(
                    "Obs.: estimativas baseadas em heurísticas internas do V14-FLEX, "
                    "levando em conta o leque final e o regime atual."
                )
    else:
        st.markdown("### 🧪 Modo Avançado por Confiabilidade")

        conf_desejada = st.slider(
            "Confiabilidade desejada (estimativa aproximada):",
            min_value=50,
            max_value=95,
            value=75,
            step=1,
            help=(
                "O sistema tentará selecionar séries cuja confiabilidade heurística "
                "seja próxima ou acima deste valor."
            ),
        )

        max_series = st.slider(
            "Número máximo de séries a exibir:",
            min_value=1,
            max_value=10,
            value=min(5, len(series_alt)),
            step=1,
        )

        if st.button("Calcular séries recomendadas"):
            alvo = conf_desejada / 100.0

            ordenadas = sorted(
                series_alt,
                key=lambda s: abs(s["confiabilidade"]["prob"] - alvo),
            )

            selecionadas = []
            for s in ordenadas:
                if len(selecionadas) >= max_series:
                    break
                selecionadas.append(s)

            if not selecionadas:
                st.warning("Nenhuma série pôde ser selecionada para esse nível.")
                return

            probs = [s["confiabilidade"]["prob"] for s in selecionadas]
            prob_media = sum(probs) / len(probs)

            st.markdown(
                f"**Séries selecionadas:** {len(selecionadas)} "
                f"(confiabilidade média ~{prob_media*100:.0f}%)."
            )
            st.markdown(
                "As séries abaixo são as mais alinhadas com o nível de confiabilidade "
                "solicitado, dentro das heurísticas internas do V14-FLEX."
            )

            st.markdown("---")

            for s in selecionadas:
                conf = s["confiabilidade"]
                serie = s["serie"]
                if not serie:
                    continue

                with st.expander(f"{s['nome']}"):
                    st.code(" ".join(str(x) for x in serie), language="text")
                    st.markdown(f"**Estilo:** {s['descricao']}")
                    st.markdown(
                        f"**Confiabilidade estimada:** {conf['nivel']} "
                        f"(~{conf['prob']*100:.0f}%)"
                    )
                    st.markdown(f"**Acertos prováveis:** {conf['faixa_acertos']}")


# ============================================================
# ROTEADOR PRINCIPAL DE PAINÉIS
# ============================================================

painel = st.radio(
    "Escolha o painel:",
    [
        "📥 Histórico — Entrada",
        "🔍 Pipeline V14-FLEX (TURBO++)",
        "🚨 Monitor de Risco (k & k*)",
        "🚀 Modo TURBO++ — Painel Completo",
        "📅 Modo Replay Automático do Histórico",
        "🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)",
        "🎛 Séries Alternativas Inteligentes V14-FLEX",
    ],
)

if painel == "📥 Histórico — Entrada":
    painel_historico_entrada()
elif painel == "🔍 Pipeline V14-FLEX (TURBO++)":
    painel_pipeline_v14_flex()
elif painel == "🚨 Monitor de Risco (k & k*)":
    painel_monitor_risco()
elif painel == "🚀 Modo TURBO++ — Painel Completo":
    painel_modo_turbo_completo()
elif painel == "📅 Modo Replay Automático do Histórico":
    painel_modo_replay()
elif painel == "🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)":
    painel_testes_confiabilidade()
elif painel == "🎛 Séries Alternativas Inteligentes V14-FLEX":
    painel_series_alternativas_inteligentes()
