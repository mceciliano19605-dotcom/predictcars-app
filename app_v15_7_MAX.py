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

    opcoes = [
        "📁 Carregar Histórico",
        "📄 Carregar Histórico (Copiar e Colar)",
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
    ]

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
# Painel 1 — 📁 Carregar Histórico
# ============================================================
if painel == "📁 Carregar Histórico":

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
# Painel 1B — 📄 Carregar Histórico (Copiar e Colar)
# ============================================================
if painel == "📄 Carregar Histórico (Copiar e Colar)":

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
            df = analisar_historico_flex_ultra(linhas)
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
            np.round(0.4 * np.array(dx_melhor)
                     + 0.3 * np.array(s6_melhor)
                     + 0.3 * np.array(previsao_mc))
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
    def s6_profundo_V157(df, idx_alvo):
        ult = df[col_pass].iloc[idx_alvo].values
        scores = []
        for i in range(len(df) - 1):
            base = df[col_pass].iloc[i].values
            inter = len(set(base) & set(ult))
            scores.append(inter)
        melhores_idx = np.argsort(scores)[-25:]
        candidatos = df[col_pass].iloc[melhores_idx].values
        return candidatos

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
    def gerar_coberturas(base):
        coberturas = []

        # Camada 1 — deslocamentos leves
        for d in [-2, -1, 1, 2]:
            cob = np.clip(base + d, 1, 60)
            coberturas.append(cob.tolist())

        # Camada 2 — reembaralhamentos leves
        for _ in range(6):
            emb = np.random.permutation(base)
            coberturas.append(emb.tolist())

        # Camada 3 — ruído adaptado ao risco
        indice_risco = risco.get("indice_risco", 0.4)
        amplitude = 3 + int(indice_risco * 5)

        for _ in range(10):
            ruido = np.random.randint(-amplitude, amplitude + 1, size=len(base))
            cob = np.clip(base + ruido, 1, 60)
            coberturas.append(cob.tolist())

        # Remove duplicatas mantendo ordem
        unicos = []
        vistos = set()
        for lista in coberturas:
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
