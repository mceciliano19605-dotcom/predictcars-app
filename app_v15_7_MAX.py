# ============================================================
# PARTE 1/8 — INÍCIO
# ============================================================

import streamlit as st
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
if "historico_df" not in st.session_state:
    st.session_state["historico_df"] = None

if "ultima_previsao" not in st.session_state:
    st.session_state["ultima_previsao"] = None

if "sentinela_kstar" not in st.session_state:
    st.session_state["sentinela_kstar"] = None

if "diagnostico_risco" not in st.session_state:
    st.session_state["diagnostico_risco"] = None

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
# Construção da Navegação — V15.7 MAX
# ============================================================
def construir_navegacao_v157() -> str:

    st.sidebar.markdown("## 🚦 Navegação PredictCars V15.7 MAX")

    # ------------------------------------------------------------
    # Painéis originais do V15.7 MAX (BASE)
    # ------------------------------------------------------------
    opcoes_base = [
        "📁 Carregar Histórico (Arquivo)",
        "📄 Carregar Histórico (Colar)",
        "🛰️ Sentinelas — k* (Ambiente de Risco)",
        "🛣️ Pipeline V14-FLEX ULTRA",
        "🔁 Replay LIGHT",
        "🔁 Replay ULTRA",
        "⚙️ Modo TURBO++ HÍBRIDO",
        "⚙️ Modo TURBO++ ULTRA",
        "📡 Painel de Ruído Condicional",
        "📉 Painel de Divergência S6 vs MC",
        "🧭 Monitor de Risco — k & k*",
        "🎯 Modo 6 Acertos — Execução",
        "🧪 Testes de Confiabilidade REAL",
        "📘 Relatório Final",
        "🔮 V16 Premium Profundo — Diagnóstico & Calibração",
         "🧠 Laudo Operacional V16",
    ]

    # ============================================================
    # INTEGRAÇÃO OFICIAL V16 PREMIUM — PAINÉIS ADICIONAIS
    # (Os painéis abaixo são SOMADOS ao menu do V15.7 MAX)
    # ============================================================
    try:
        opcoes_v16 = v16_obter_paineis()
    except Exception:
        opcoes_v16 = []

    # Combinação final
    opcoes = opcoes_base + opcoes_v16

    # ------------------------------------------------------------
    # Renderização do menu
    # ------------------------------------------------------------
    painel = st.sidebar.selectbox(
        "Selecione um painel:",
        opcoes,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <p class="gray-text">
        PredictCars V15.7 MAX · V16 Premium Profundo<br>
        Núcleo + Coberturas + Interseção Estatística
        </p>
        """,
        unsafe_allow_html=True,
    )

    return painel



# ============================================================
# Ativação da Navegação
# ============================================================
painel = construir_navegacao_v157()

# ============================================================
# CAMADA A — ESTADO DO ALVO (V16)
# Observador puro — NÃO decide, NÃO bloqueia, NÃO gera previsões
# ============================================================

def v16_estimar_volatilidade_local(df: Optional[pd.DataFrame], janela: int = 30) -> float:
    try:
        matriz_norm = st.session_state.get("pipeline_matriz_norm")
        if isinstance(matriz_norm, np.ndarray) and len(matriz_norm) >= janela:
            bloco = matriz_norm[-janela:]
            return float(np.mean(np.std(bloco, axis=1)))
    except Exception:
        pass

    if df is None or df.empty:
        return 0.0

    try:
        col_pass = [c for c in df.columns if str(c).startswith("p")]
        bloco = df[col_pass].tail(janela).astype(float).values
        return float(np.mean(np.std(bloco, axis=1)))
    except Exception:
        return 0.0


def v16_calcular_estado_alvo(
    df: Optional[pd.DataFrame],
    nr_percent: Optional[float],
    divergencia: Optional[float],
) -> Dict[str, Any]:

    nr = float(nr_percent) if isinstance(nr_percent, (int, float)) else 35.0
    div = float(divergencia) if isinstance(divergencia, (int, float)) else 4.0
    vol = v16_estimar_volatilidade_local(df, janela=30)

    nr_norm = min(1.0, nr / 70.0)
    div_norm = min(1.0, div / 10.0)
    vol_norm = min(1.0, vol / 0.35)

    velocidade = float(0.45 * nr_norm + 0.35 * div_norm + 0.20 * vol_norm)

    if velocidade < 0.33:
        tipo = "parado"
        comentario = "🎯 Alvo estável — erro tende a ser por pouco. Volume alto faz sentido."
    elif velocidade < 0.66:
        tipo = "movimento_lento"
        comentario = "🎯 Alvo em movimento lento — alternar rajadas e coberturas."
    else:
        tipo = "movimento_rapido"
        comentario = "⚠️ Alvo rápido — ambiente difícil. Operação respiratória."

    return {
        "tipo": tipo,
        "velocidade": round(velocidade, 4),
        "vol_local": round(vol, 4),
        "nr_percent": nr,
        "divergencia": div,
        "comentario": comentario,
    }


def v16_registrar_estado_alvo():
    estado = v16_calcular_estado_alvo(
        st.session_state.get("historico_df"),
        st.session_state.get("nr_percent"),
        st.session_state.get("div_s6_mc"),
    )
    st.session_state["estado_alvo_v16"] = estado
    return estado

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
# Painel 1 — 📁 Carregar Histórico (Arquivo)
# ============================================================
if painel == "📁 Carregar Histórico (Arquivo)":

    st.markdown("## 📁 Carregar Histórico — V15.7 MAX")

    arquivo = st.file_uploader(
        "Envie o arquivo de histórico (formato FLEX ULTRA)",
        type=["txt", "csv"],
    )

    if arquivo is not None:
        conteudo = arquivo.getvalue().decode("utf-8")
        df = analisar_historico_flex_ultra(conteudo)

        st.session_state["historico_df"] = df

        metricas = calcular_metricas_basicas_historico(df)
        exibir_resumo_inicial_historico(metricas)

        st.success("Histórico carregado com sucesso!")
        st.dataframe(df.head(20))

    else:
        exibir_bloco_mensagem(
            "Aguardando arquivo de histórico",
            "Envie seu arquivo para iniciar o processamento do PredictCars V15.7 MAX.",
            tipo="info",
        )

# ============================================================
# Painel 1B — 📄 Carregar Histórico (Colar)
# ============================================================
if painel == "📄 Carregar Histórico (Colar)":

    st.markdown("## 📄 Carregar Histórico — Copiar e Colar (V15.7 MAX)")

    st.markdown(
        "Cole abaixo o conteúdo completo do histórico em formato **FLEX ULTRA** "
        "(linhas como `C123;12;34;56;23;45;2`)."
    )

    texto = st.text_area(
        "Cole aqui o histórico completo",
        height=300,
        placeholder="C1;41;5;4;52;30;33;0\nC2;9;39;37;49;43;41;1\n..."
    )

    if st.button("📥 Processar Histórico (Copiar e Colar)"):

        linhas = texto.strip().split("\n")

        if not limitar_operacao(
            len(linhas),
            limite_series=LIMITE_SERIES_REPLAY_ULTRA,
            contexto="Carregar Histórico (Copiar e Colar)",
            painel="📄 Carregar Histórico (Copiar e Colar)",
        ):
            st.stop()

        if not texto.strip():
            exibir_bloco_mensagem(
                "Nenhum dado encontrado",
                "Cole o conteúdo do histórico FLEX ULTRA para continuar.",
                tipo="warning",
            )
            st.stop()

        try:
            conteudo = "\n".join(linhas)
            df = analisar_historico_flex_ultra(conteudo)
        except Exception as erro:
            exibir_bloco_mensagem(
                "Erro ao processar histórico",
                f"Detalhes técnicos: {erro}",
                tipo="error",
            )
            st.stop()

        st.session_state["historico_df"] = df

        exibir_bloco_mensagem(
            "Histórico carregado com sucesso!",
            f"Séries carregadas: **{len(df)}**\n\n"
            "Agora prossiga para o painel **🛣️ Pipeline V14-FLEX ULTRA**.",
            tipo="success",
        )

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

    # Salvando na sessão para módulos seguintes
    st.session_state["pipeline_clusters"] = clusters
    st.session_state["pipeline_centroides"] = centroides
    st.session_state["pipeline_matriz_norm"] = matriz_norm
    st.session_state["pipeline_estrada"] = estrada

    st.success("Pipeline FLEX ULTRA concluído com sucesso!")

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
# Painel 7 — ⚙️ Modo TURBO++ ULTRA
# ============================================================
if painel == "⚙️ Modo TURBO++ ULTRA":

    st.markdown("## ⚙️ Modo TURBO++ ULTRA — V15.7 MAX")

    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")
    k_star = st.session_state.get("sentinela_kstar")

    if df is None or matriz_norm is None:
        exibir_bloco_mensagem(
            "Pipeline incompleto",
            "Execute o painel **🛣️ Pipeline V14-FLEX ULTRA**.",
            tipo="warning",
        )
        st.stop()

    qtd_series = len(df)

    # Anti-zumbi forte — TURBO++ ULTRA é mais pesado
    if not limitar_operacao(
        qtd_series,
        limite_series=LIMITE_SERIES_TURBO_ULTRA,
        contexto="TURBO++ ULTRA",
        painel="⚙️ Modo TURBO++ ULTRA",
    ):
        st.stop()

    if k_star is None:
        exibir_bloco_mensagem(
            "k* não encontrado",
            "Vá ao painel **🛰️ Sentinelas — k*** antes.",
            tipo="warning",
        )
        st.stop()

    st.info("Executando Modo TURBO++ ULTRA...")

    col_pass = [c for c in df.columns if c.startswith("p")]

    # ============================================================
    # MOTORES PROFUNDOS
    # ============================================================
    # --- S6 PROFUNDO ---
    def s6_profundo_V157(df_local, idx_alvo):
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
    # ORQUESTRAÇÃO ULTRA
    # ============================================================
    try:
        base = df[col_pass].iloc[-1].values

        candidatos_s6 = s6_profundo_V157(df, -1)

        ml = micro_leque_profundo(base, profundidade=15)

        mc = monte_carlo_profundo(base, n=1200)

        # Pesos guiados por k*
        peso_s6 = 0.55 - (k_star * 0.15)
        peso_mc = 0.30 + (k_star * 0.20)
        peso_ml = 1.0 - (peso_s6 + peso_mc)

        # Interseção estatística
        todos = np.vstack([
            candidatos_s6,
            ml,
            np.array(mc)
        ])

        previsao_raw = (
            peso_s6 * candidatos_s6.mean(axis=0)
            + peso_mc * np.mean(mc, axis=0)
            + peso_ml * ml.mean(axis=0)
        )

        previsao_final = [int(round(x)) for x in previsao_raw]

        # Divergência S6 vs MC
        divergencia = np.linalg.norm(
            candidatos_s6.mean(axis=0) - np.mean(mc, axis=0)
        )

    except Exception as erro:
        exibir_bloco_mensagem(
            "Erro no motor TURBO++ ULTRA",
            f"Detalhes técnicos: {erro}",
            tipo="error",
        )
        st.stop()

    # ============================================================
    # Exibição final
    # ============================================================
    st.markdown("### 🔮 Previsão ULTRA (TURBO++)")
    st.success(f"**{formatar_lista_passageiros(previsao_final)}**")

    st.markdown("### 🔎 Divergência S6 vs MC")
    st.info(f"**{divergencia:.4f}**")

    st.session_state["ultima_previsao"] = previsao_final
    st.session_state["div_s6_mc"] = divergencia

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
# Painel 11 — 🎯 Modo 6 Acertos — Execução (V15.7 MAX)
# ============================================================
if painel == "🎯 Modo 6 Acertos — Execução":

    st.markdown("## 🎯 Modo 6 Acertos — Execução — V15.7 MAX")

    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")
    ultima_prev = st.session_state.get("ultima_previsao")
    risco = st.session_state.get("diagnostico_risco", {})

    if df is None or matriz_norm is None or ultima_prev is None:
        exibir_bloco_mensagem(
            "Pré-requisitos não atendidos",
            "Execute TURBO++ ULTRA antes, para gerar a previsão base.",
            tipo="warning",
        )
        st.stop()

    qtd_series = len(df)

    # Anti-zumbi rigoroso — Modo 6 Acertos gera MUITAS listas
    if not limitar_operacao(
        qtd_series,
        limite_series=LIMITE_PREVISOES_MODO_6,
        contexto="Modo 6 Acertos",
        painel="🎯 Modo 6 Acertos — Execução",
    ):
        st.stop()

    st.info("Executando Modo 6 Acertos (versão Premium)...")

    # ============================================================
    # Núcleo (TURBO++ ULTRA)
    # ============================================================
    base = np.array(ultima_prev)

    # ============================================================
    # Coberturas Estatísticas Premium
    # ============================================================
    def gerar_coberturas(base_local):
        coberturas_local = []

        # Camada 1 — deslocamentos leves
        for d in [-2, -1, 1, 2]:
            cob = np.clip(base_local + d, 1, 60)
            coberturas_local.append(cob.tolist())

        # Camada 2 — reembaralhamentos leves
        for _ in range(6):
            emb = np.random.permutation(base_local)
            coberturas_local.append(emb.tolist())

        # Camada 3 — ruído adaptado ao risco
        if isinstance(risco, dict):
            indice_risco_local = risco.get("indice_risco", 0.4)
        else:
            indice_risco_local = 0.4

        amplitude = 3 + int(indice_risco_local * 5)

        for _ in range(10):
            ruido = np.random.randint(
                -amplitude,
                amplitude + 1,
                size=len(base_local)
            )
            cob = np.clip(base_local + ruido, 1, 60)
            coberturas_local.append(cob.tolist())


        # Remove duplicatas mantendo ordem
        unicos = []
        vistos = set()
        for lista in coberturas_local:
            t = tuple(lista)
            if t not in vistos:
                vistos.add(t)
                unicos.append(lista)

        return unicos

    coberturas = gerar_coberturas(base)

    # ============================================================
    # Interseção estatística (núcleo + coberturas)
    # ============================================================
    todas = [base.tolist()] + coberturas
    todas = [list(map(int, x)) for x in todas]

    # Ordenação por similaridade ao núcleo
    def similaridade(a, b):
        return len(set(a) & set(b))

    todas_ordenadas = sorted(
        todas,
        key=lambda x: similaridade(base, x),
        reverse=True,
    )

    # Seleção final
    listas_finais = todas_ordenadas[:20]

    # ============================================================
    # Exibição do resultado
    # ============================================================
    st.markdown("### 🔮 Núcleo + Coberturas (Top 20)")
    for i, lst in enumerate(listas_finais, 1):
        st.markdown(f"**{i:02d})** {formatar_lista_passageiros(lst)}")

    st.session_state["modo6_listas"] = listas_finais
    st.success("Modo 6 Acertos concluído!")


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
# (Elimina permutações, duplicatas por conjunto e falsas coberturas)
# Válido para V15.7 MAX e V16 Premium
# ============================================================

def _sanear_listas_previsao(
    listas,
    min_diferencas: int = 1,
    referencia: list = None,
):
    """
    Aplica saneamento final nas listas de previsão:
    - Normaliza (ordena)
    - Remove duplicatas por CONJUNTO
    - Remove listas idênticas à referência (6/6)
    - Exige diversidade mínima (min_diferencas)
    """

    if not listas:
        return []

    saneadas = []
    vistos = set()

    ref_set = set(referencia) if referencia else None

    for lst in listas:
        try:
            lst_int = [int(x) for x in lst]
        except Exception:
            continue

        lst_norm = tuple(sorted(lst_int))

        # Remove duplicatas por conjunto
        if lst_norm in vistos:
            continue

        # Remove cópia total da referência (6/6)
        if ref_set is not None:
            inter = len(set(lst_norm) & ref_set)
            if inter == len(ref_set):
                continue
            # Exige diversidade mínima
            if (len(ref_set) - inter) < min_diferencas:
                continue

        vistos.add(lst_norm)
        saneadas.append(list(lst_norm))

    return saneadas


# ============================================================
# APLICAÇÃO AUTOMÁTICA DA SANIDADE (SE LISTAS EXISTIREM)
# ============================================================

# Sanear listas do Modo 6 (V15.7)
if "modo6_listas" in st.session_state:
    base_ref = st.session_state.get("ultima_previsao")
    st.session_state["modo6_listas"] = _sanear_listas_previsao(
        st.session_state.get("modo6_listas", []),
        min_diferencas=1,
        referencia=base_ref,
    )

# Sanear Execução V16 (se existir)
if "v16_execucao" in st.session_state:
    exec_v16 = st.session_state.get("v16_execucao", {})
    base_ref = exec_v16.get("C1")

    for chave in ["C2", "C3", "todas_listas"]:
        if chave in exec_v16:
            exec_v16[chave] = _sanear_listas_previsao(
                exec_v16.get(chave, []),
                min_diferencas=1,
                referencia=base_ref,
            )

    st.session_state["v16_execucao"] = exec_v16

# ============================================================
# PARTE 6/8 — FIM
# ============================================================
# ============================================================
# PARTE 7/8 — INÍCIO
# ============================================================

# ============================================================
# Painel 13 — 📘 Relatório Final — V15.7 MAX (Premium)
# ============================================================
if painel == "📘 Relatório Final":

    st.markdown("## 📘 Relatório Final — V15.7 MAX — V16 Premium Profundo")

    ultima_prev = st.session_state.get("ultima_previsao")
    listas_m6 = st.session_state.get("modo6_listas")
    risco = st.session_state.get("diagnostico_risco")
    nr_percent = st.session_state.get("nr_percent")
    k_star = st.session_state.get("sentinela_kstar")
    divergencia = st.session_state.get("div_s6_mc")

    if ultima_prev is None:
        exibir_bloco_mensagem(
            "Nenhuma previsão encontrada",
            "Execute o painel **⚙️ Modo TURBO++ ULTRA** antes.",
            tipo="warning",
        )
        st.stop()

    if listas_m6 is None:
        exibir_bloco_mensagem(
            "Modo 6 Acertos ainda não executado",
            "Vá ao painel **🎯 Modo 6 Acertos — Execução**.",
            tipo="warning",
        )
        st.stop()

    if risco is None:
        risco = {
            "indice_risco": 0.45,
            "classe_risco": "🟡 Risco Moderado",
            "k_star": k_star or 0.25,
            "nr_percent": nr_percent or 35.0,
            "divergencia": divergencia or 4.0,
        }

    # ============================================================
    # 1) Previsão principal (Núcleo)
    # ============================================================
    st.markdown("### 🔮 Previsão Principal (Núcleo — TURBO++ ULTRA)")
    st.success(formatar_lista_passageiros(ultima_prev))

    # ============================================================
    # 2) Coberturas (Top 10)
    # ============================================================
    st.markdown("### 🛡️ Coberturas Selecionadas (Top 10)")
    for i, lst in enumerate(listas_m6[:10], 1):
        st.markdown(f"**{i:02d})** {formatar_lista_passageiros(lst)}")

    # ============================================================
    # 3) Indicadores Premium — Ambiente e Risco
    # ============================================================
    st.markdown("### 🌐 Indicadores do Ambiente (k*, NR%, Divergência)")

    corpo = (
        f"- 🌡️ **k\\*** (sentinela): **{risco['k_star']:.4f}**\n"
        f"- 📡 **NR%** (ruído condicional): **{risco['nr_percent']:.2f}%**\n"
        f"- 📉 **Divergência S6 vs MC**: **{risco['divergencia']:.4f}**\n"
    )

    exibir_bloco_mensagem(
        "Indicadores do Ambiente — Premium",
        corpo,
        tipo="info",
    )

    # ============================================================
    # 4) Diagnóstico de Risco Composto
    # ============================================================
    st.markdown("### 🧭 Diagnóstico de Risco Composto")

    indice_risco = risco["indice_risco"]
    classe_risco = risco["classe_risco"]

    corpo = (
        f"- Índice Composto de Risco: **{indice_risco:.4f}**\n"
        f"- Classe de Risco: {classe_risco}\n"
    )

    exibir_bloco_mensagem(
        "Resumo do Risco Composto",
        corpo,
        tipo="info",
    )

    # ============================================================
    # 5) Orientação Final — Premium
    # ============================================================
    st.markdown("### 🧩 Orientação Final — V16 Premium")

    if indice_risco < 0.30:
        orientacao = (
            "🟢 **Ambiente favorável** — Combinação de Núcleo + Coberturas leves.\n"
            "A agressividade pode ser moderada → priorizar listas mais enxutas."
        )
    elif indice_risco < 0.55:
        orientacao = (
            "🟡 **Ambiente equilibrado** — Núcleo ainda opera bem.\n"
            "Manter coberturas e reforçar listas auxiliares."
        )
    elif indice_risco < 0.80:
        orientacao = (
            "🟠 **Ambiente turbulento** — Priorizar coberturas e reduzir peso do núcleo.\n"
            "Avaliar divergência e ruído antes de decisões finais."
        )
    else:
        orientacao = (
            "🔴 **Ambiente crítico** — Operar com máxima cautela, priorizando estabilização.\n"
            "Evitar agressividade e monitorar S6 vs MC."
        )

    exibir_bloco_mensagem(
        "Orientação Premium",
        orientacao,
        tipo="info",
    )

    st.success("Relatório Final gerado com sucesso!")
# ============================================================
# Painel X — 🧠 Laudo Operacional V16 (Estado, Expectativa, Volume)
# ============================================================

if painel == "🧠 Laudo Operacional V16":

    st.markdown("## 🧠 Laudo Operacional V16 — Leitura do Ambiente")

    # Garantir registros atualizados
    estado = v16_registrar_estado_alvo()
    expectativa = v16_registrar_expectativa()
    volume_op = v16_registrar_volume_e_confiabilidade()

    # --------------------------------------------------------
    # 1) Estado do Alvo
    # --------------------------------------------------------
    st.markdown("### 🎯 Estado do Alvo")
    st.info(
        f"Tipo: **{estado['tipo']}**  \n"
        f"Velocidade estimada: **{estado['velocidade']}**  \n"
        f"Comentário: {estado['comentario']}"
    )

    # --------------------------------------------------------
    # 2) Expectativa de Curto Prazo
    # --------------------------------------------------------
    st.markdown("### 🔮 Expectativa (1–3 séries)")
    st.info(
        f"Previsibilidade: **{expectativa['previsibilidade']}**  \n"
        f"Erro esperado: **{expectativa['erro_esperado']}**  \n"
        f"Chance de janela de ouro: **{expectativa['chance_janela_ouro']}**  \n\n"
        f"{expectativa['comentario']}"
    )

    # --------------------------------------------------------
    # 3) Volume x Confiabilidade
    # --------------------------------------------------------
    st.markdown("### 📊 Volume × Confiabilidade (informativo)")

    confs = volume_op.get("confiabilidades_estimadas", {})
    if confs:
        df_conf = pd.DataFrame(
            [{"Previsões": k, "Confiabilidade estimada": v} for k, v in confs.items()]
        )
        st.dataframe(df_conf, use_container_width=True)

    st.warning(
        f"📌 Volume mínimo: **{volume_op['minimo']}**  \n"
        f"📌 Volume recomendado: **{volume_op['recomendado']}**  \n"
        f"📌 Volume máximo técnico: **{volume_op['maximo_tecnico']}**  \n\n"
        f"{volume_op['comentario']}"
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


# ============================================================
# ROTEADOR V16 PREMIUM — EXECUÇÃO DOS PAINÉIS
# ============================================================
try:
    _opcoes_v16_router = v16_obter_paineis()
except Exception:
    _opcoes_v16_router = []

if painel in _opcoes_v16_router:
    v16_renderizar_painel(painel)
    st.stop()

# ============================================================
# PARTE 8/8 — FIM
# ============================================================
