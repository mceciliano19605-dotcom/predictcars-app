import streamlit as st
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURAÇÃO GERAL DO APP
# ============================================================

st.set_page_config(
    page_title="Predict Cars V14 TURBO++",
    page_icon="🚗",
    layout="wide",
)

# ============================================================
# FUNÇÕES DE APOIO — HISTÓRICO E PREPARO
# ============================================================

def _detectar_sep(linha: str) -> str:
    if ";" in linha:
        return ";"
    if "," in linha:
        return ","
    return ";"


def preparar_historico_V14(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza o histórico para o formato padrão V14:
    colunas: ['serie', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'k', 'idx'].
    """
    df = df_raw.copy()

    # Remove colunas vazias típicas de CSV
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]

    # Caso mais comum: 7 ou 8 colunas
    if df.shape[1] in (7, 8):
        cols_novas = ["serie", "p1", "p2", "p3", "p4", "p5", "p6"]
        if df.shape[1] == 8:
            cols_novas.append("k")
        df.columns = cols_novas
    else:
        # Tenta mapear por nomes aproximados
        colunas = [c.lower() for c in df.columns]
        mapa = {}
        for i, c in enumerate(colunas):
            if "c" in c or "id" in c or "serie" in c:
                mapa["serie"] = df.columns[i]
            elif any(x in c for x in ["k", "risco"]):
                mapa["k"] = df.columns[i]

        # passageiros: pega os 6 primeiros que não forem 'serie' nem 'k'
        restantes = [c for c in df.columns if c not in mapa.values()]
        while len(restantes) < 6:
            restantes.append(restantes[-1])
        mapa["p1"] = restantes[0]
        mapa["p2"] = restantes[1]
        mapa["p3"] = restantes[2]
        mapa["p4"] = restantes[3]
        mapa["p5"] = restantes[4]
        mapa["p6"] = restantes[5]

        ordem = ["serie", "p1", "p2", "p3", "p4", "p5", "p6"]
        if "k" in mapa:
            ordem.append("k")
        df = df.rename(columns={v: k for k, v in mapa.items()})[ordem]

    # Tipos
    df["serie"] = df["serie"].astype(str)
    for c in ["p1", "p2", "p3", "p4", "p5", "p6"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "k" not in df.columns:
        df["k"] = 0
    else:
        df["k"] = pd.to_numeric(df["k"], errors="coerce").fillna(0).astype(int)

    # Índice sequencial (estrada)
    df["idx"] = np.arange(1, len(df) + 1)
    df = df.sort_values("idx").reset_index(drop=True)

    return df


def parse_historico_texto(conteudo: str) -> pd.DataFrame:
    """
    Converte texto colado no formato:
    C1;41;5;4;52;30;33;0
    C2;9;39;37;49;43;41;1
    ...
    em DataFrame preparado.
    """
    linhas = [l.strip() for l in conteudo.strip().splitlines() if l.strip()]
    if not linhas:
        raise ValueError("Nenhuma linha válida encontrada.")

    sep = _detectar_sep(linhas[0])
    registros = []
    for linha in linhas:
        partes = [p.strip() for p in linha.split(sep) if p.strip()]
        if len(partes) < 7:
            continue
        serie = partes[0]
        numeros = partes[1:]
        numeros += ["0"] * max(0, 7 - len(numeros))
        p1, p2, p3, p4, p5, p6 = numeros[:6]
        k = numeros[6] if len(numeros) >= 7 else "0"
        registros.append(
            {
                "serie": serie,
                "p1": int(p1),
                "p2": int(p2),
                "p3": int(p3),
                "p4": int(p4),
                "p5": int(p5),
                "p6": int(p6),
                "k": int(k),
            }
        )

    df_raw = pd.DataFrame(registros)
    return preparar_historico_V14(df_raw)


# ============================================================
# k HISTÓRICO — SENTINELA REATIVO
# ============================================================

def classificar_k_valor(k_val: int) -> str:
    """
    Converte o valor de k em estado de risco:
    0 -> estavel
    1 -> atencao
    >=2 -> critico
    """
    if k_val <= 0:
        return "estavel"
    if k_val == 1:
        return "atencao"
    return "critico"


def estado_k_global(df: pd.DataFrame) -> dict:
    """
    Resumo global do k histórico (reativo).
    """
    if df.empty or "k" not in df.columns:
        return {
            "estado": "indefinido",
            "contagens": {"estavel": 0, "atencao": 0, "critico": 0},
        }

    estados = df["k"].apply(classificar_k_valor)
    contagens = estados.value_counts().to_dict()
    for chave in ["estavel", "atencao", "critico"]:
        contagens.setdefault(chave, 0)

    # Puxa o pior estado que aparece
    if contagens["critico"] > 0:
        estado = "critico"
    elif contagens["atencao"] > 0:
        estado = "atencao"
    else:
        estado = "estavel"

    return {"estado": estado, "contagens": contagens}


def rotulo_k(estado: str) -> str:
    """
    Frase amigável para o estado de risco.
    """
    if estado == "estavel":
        return "🟢 Ambiente estável — previsão em regime normal."
    if estado == "atencao":
        return "🟡 Pré-ruptura — usar previsão com atenção."
    if estado == "critico":
        return "🔴 Ambiente crítico — usar previsão com cautela máxima."
    return "⚪ Estado de risco indefinido."


# ============================================================
# k* TURBO++ — SENTINELA PREDITIVO (MÓDULO SIMPLIFICADO)
# ============================================================

def _extrair_vetor_passageiros(df_linha: pd.Series) -> np.ndarray:
    return np.array([df_linha["p1"], df_linha["p2"], df_linha["p3"],
                     df_linha["p4"], df_linha["p5"], df_linha["p6"]], dtype=float)


def calcular_metricas_risco_janela(df_janela: pd.DataFrame) -> dict:
    """
    Calcula métricas de risco estruturais em uma janela (trecho da estrada).
    Produz:
    - turbulencia_media (mudança entre séries consecutivas)
    - dispersao_media (variância média dos passageiros)
    """
    if df_janela is None or df_janela.empty or len(df_janela) < 3:
        return {"turbulencia": 0.0, "dispersao": 0.0}

    # Turbulência: média da distância entre séries consecutivas
    diffs = []
    linhas = df_janela.sort_values("idx").reset_index(drop=True)
    for i in range(1, len(linhas)):
        v1 = _extrair_vetor_passageiros(linhas.iloc[i - 1])
        v2 = _extrair_vetor_passageiros(linhas.iloc[i])
        diffs.append(np.linalg.norm(v2 - v1))
    turbulencia = float(np.mean(diffs)) if diffs else 0.0

    # Dispersão: variância média por passageiro
    mat = np.vstack([_extrair_vetor_passageiros(linhas.iloc[i])
                     for i in range(len(linhas))])
    dispersao = float(np.mean(np.var(mat, axis=0)))

    return {"turbulencia": turbulencia, "dispersao": dispersao}


def normalizar_score(valor: float, referencia_baixa: float, referencia_alta: float) -> float:
    """
    Normaliza valor para [0,1] dado um intervalo aproximado.
    """
    if referencia_alta <= referencia_baixa:
        return 0.0
    score = (valor - referencia_baixa) / (referencia_alta - referencia_baixa)
    score = max(0.0, min(1.0, score))
    return score


def calcular_k_star_estado(
    df: pd.DataFrame,
    idx_alvo: int,
    largura_janela: int = 40,
) -> dict:
    """
    k* TURBO++ (versão simplificada):
    - analisa um trecho antes do idx_alvo,
    - mede turbulência e dispersão,
    - gera um score de risco [0,1],
    - converte em estado: estavel / atencao / critico.
    """
    if df.empty:
        return {"estado": "indefinido", "score": 0.0}

    idx_alvo = int(idx_alvo)
    idx_min = int(df["idx"].min())
    idx_max = int(df["idx"].max())
    idx_alvo = max(idx_min + 1, min(idx_alvo, idx_max))

    # Janela: pega até "largura_janela" séries antes do alvo
    inicio = max(idx_min, idx_alvo - largura_janela)
    df_janela = df[(df["idx"] >= inicio) & (df["idx"] < idx_alvo)].copy()

    if df_janela.empty:
        return {"estado": "indefinido", "score": 0.0}

    met = calcular_metricas_risco_janela(df_janela)

    # Normalizações heurísticas (ajustáveis):
    # Turbulência típica vai de 0 a ~60 (palpite), dispersão 0 a ~300 (palpite).
    score_turb = normalizar_score(met["turbulencia"], 0.0, 60.0)
    score_disp = normalizar_score(met["dispersao"], 0.0, 300.0)

    # Score combinado: média ponderada
    score = 0.6 * score_turb + 0.4 * score_disp

    # Mapeia para estados
    if score < 0.33:
        estado = "estavel"
    elif score < 0.66:
        estado = "atencao"
    else:
        estado = "critico"

    return {"estado": estado, "score": float(score)}


def rotulo_k_star(estado: str, score: float) -> str:
    """
    Mensagem interpretável para o k* (sentinela preditivo).
    """
    score_pct = int(round(score * 100))
    if estado == "estavel":
        return f"🟢 k*: Ambiente tende a permanecer estável (risco ≈ {score_pct}%)."
    if estado == "atencao":
        return f"🟡 k*: Pré-ruptura estrutural detectada (risco ≈ {score_pct}%)."
    if estado == "critico":
        return f"🔴 k*: Alta probabilidade de ruptura ou turbulência forte (risco ≈ {score_pct}%)."
    return "⚪ k*: Estado preditivo indefinido."


# ============================================================
# PIPELINE V14 — PLACEHOLDER ESTRUTURAL (SUBSTITUÍVEL)
# ============================================================

def executar_pipeline_v14_simples(
    df: pd.DataFrame,
    idx_alvo: int,
    n_series: int = 20,
) -> pd.DataFrame:
    """
    Pipeline V14 simples — versão estrutural.
    Hoje: usa janelas recentes como candidatos.
    Depois você pode plugar o motor real V14 TURBO++.
    """
    if df.empty:
        return pd.DataFrame()

    idx_alvo = int(idx_alvo)
    idx_min = int(df["idx"].min())
    idx_max = int(df["idx"].max())
    idx_alvo = max(idx_min + 1, min(idx_alvo, idx_max))

    # Histórico antes do alvo
    df_hist = df[df["idx"] < idx_alvo].copy()
    if df_hist.empty:
        df_hist = df.copy()

    # Pega as N últimas séries como candidatos
    n_series = int(n_series)
    candidatos = df_hist.tail(n_series).copy()

    # Constrói vetor "series" e score simples
    candidatos["series"] = candidatos.apply(
        lambda row: [int(row[c]) for c in ["p1", "p2", "p3", "p4", "p5", "p6"]],
        axis=1,
    )

    max_idx_hist = candidatos["idx"].max()
    candidatos["score"] = 1.0 - (max_idx_hist - candidatos["idx"]) / max(1, max_idx_hist)

    # k_previsto histórico herdado (placeholder)
    candidatos["k_previsto"] = candidatos["k"].astype(int)

    candidatos = candidatos.sort_values("score", ascending=False).reset_index(drop=True)

    return candidatos[["serie", "idx", "series", "score", "k_previsto"]]


def extrair_previsao_final(df_candidatos: pd.DataFrame):
    """
    Escolhe a melhor série candidata como previsão final.
    """
    if df_candidatos is None or df_candidatos.empty:
        return None
    melhor = df_candidatos.iloc[0]
    return melhor["series"]


def estimar_k_previsto(df_candidatos: pd.DataFrame) -> int:
    """
    Estima k previsto (reativo) usando o melhor candidato.
    """
    if df_candidatos is None or df_candidatos.empty:
        return 0
    melhor = df_candidatos.iloc[0]
    return int(melhor.get("k_previsto", 0))


# ============================================================
# LAYOUT — SIDEBAR E ESTADO GLOBAL
# ============================================================

st.sidebar.title("🚗 Predict Cars V14 TURBO++")
st.sidebar.markdown("Versão com **k (histórico)** e **k\* (sentinela preditivo)**.")

painel = st.sidebar.radio(
    "Escolha o painel:",
    [
        "📥 Histórico — Entrada",
        "🔍 Pipeline V14 (Simples)",
        "🚨 Monitor de Risco (k & k*)",
        "🚀 Modo TURBO++ — Previsão Final",
    ],
)

# Inicializa sessão
if "df" not in st.session_state:
    st.session_state["df"] = pd.DataFrame()
if "df_candidatos" not in st.session_state:
    st.session_state["df_candidatos"] = pd.DataFrame()
if "ultimo_idx_alvo" not in st.session_state:
    st.session_state["ultimo_idx_alvo"] = None
if "k_star_estado" not in st.session_state:
    st.session_state["k_star_estado"] = "indefinido"
if "k_star_score" not in st.session_state:
    st.session_state["k_star_score"] = 0.0

df_sessao = st.session_state["df"]  # atalho local


# ============================================================
# PAINEL 1 — Histórico — Entrada
# ============================================================

if painel == "📥 Histórico — Entrada":
    st.markdown("## 📥 Histórico — Entrada")

    df = None

    opc = st.radio(
        "Como deseja carregar o histórico?",
        ["Enviar arquivo CSV", "Copiar e colar o histórico"],
        horizontal=True,
    )

    # ---------- OPÇÃO 1 — UPLOAD ----------
    if opc == "Enviar arquivo CSV":
        file = st.file_uploader("Selecione o arquivo CSV:", type=["csv"])
        if file is not None:
            try:
                df_raw = pd.read_csv(file)
                df = preparar_historico_V14(df_raw)
                st.session_state["df"] = df
                st.success("Histórico carregado com sucesso!")
            except Exception as e:
                st.error(f"Erro ao carregar CSV: {e}")

    # ---------- OPÇÃO 2 — TEXTO ----------
    else:
        conteudo = st.text_area(
            "Cole o histórico aqui (ex: C1;41;5;4;52;30;33;0):",
            height=250,
        )
        if st.button("Carregar histórico do texto"):
            try:
                df = parse_historico_texto(conteudo)
                st.session_state["df"] = df
                st.success("Histórico carregado com sucesso a partir do texto!")
            except Exception as e:
                st.error(f"Erro ao interpretar o texto: {e}")

    df = st.session_state["df"]

    if df is not None and not df.empty:
        st.markdown("### 🔎 Prévia do histórico preparado (V14)")
        st.dataframe(df.head(30))

        info_k = estado_k_global(df)
        estado = info_k["estado"]
        contagens = info_k["contagens"]

        st.markdown("### ⚠️ Resumo de risco histórico (k)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Séries estáveis (k=0)", contagens["estavel"])
            st.metric("Séries em atenção (k=1)", contagens["atencao"])
            st.metric("Séries críticas (k≥2)", contagens["critico"])
        with col2:
            st.markdown("**Estado global (k histórico):**")
            if estado == "estavel":
                st.success("🟢 Estável")
            elif estado == "atencao":
                st.warning("🟡 Atenção")
            elif estado == "critico":
                st.error("🔴 Crítico")
            else:
                st.info("⚪ Indefinido")

        st.markdown("### 📈 Evolução de k histórico")
        graf = df[["idx", "k"]].set_index("idx")
        st.line_chart(graf)

    else:
        st.info("Carregue o histórico para continuar.")


# ============================================================
# PAINEL 2 — Pipeline V14 (Simples)
# ============================================================

elif painel == "🔍 Pipeline V14 (Simples)":
    st.markdown("## 🔍 Pipeline V14 — Execução Simples")

    df = st.session_state.get("df", pd.DataFrame())
    if df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        idx_min = int(df["idx"].min())
        idx_max = int(df["idx"].max())
        idx_alvo = st.number_input(
            "Selecione o índice alvo (idx):",
            min_value=idx_min + 1,
            max_value=idx_max,
            value=idx_max,
            step=1,
        )
    with col2:
        n_series = st.number_input(
            "Quantidade de séries candidatas:",
            min_value=5,
            max_value=200,
            value=30,
            step=5,
        )
    with col3:
        executar = st.button("Executar Pipeline V14 (Simples)", type="primary")

    if executar:
        df_cand = executar_pipeline_v14_simples(df, idx_alvo, n_series)
        st.session_state["df_candidatos"] = df_cand
        st.session_state["ultimo_idx_alvo"] = int(idx_alvo)

        # Atualiza k* para este alvo
        kstar = calcular_k_star_estado(df, idx_alvo)
        st.session_state["k_star_estado"] = kstar["estado"]
        st.session_state["k_star_score"] = kstar["score"]

    df_cand = st.session_state.get("df_candidatos", pd.DataFrame())
    idx_alvo_mem = st.session_state.get("ultimo_idx_alvo", None)

    if df_cand is not None and not df_cand.empty:
        st.markdown("### 📊 Séries candidatas — Pipeline V14")
        st.dataframe(df_cand)

        # k real da série alvo (histórico)
        serie_alvo = df[df["idx"] == idx_alvo_mem] if idx_alvo_mem is not None else pd.DataFrame()
        if not serie_alvo.empty:
            k_real = int(serie_alvo.iloc[0]["k"])
            estado_k_real = classificar_k_valor(k_real)
            st.markdown("### ⚠️ k histórico da série alvo")
            st.write(f"**Série alvo:** {serie_alvo.iloc[0]['serie']}")
            st.info(rotulo_k(estado_k_real))

        # k previsto reativo (herdado do melhor candidato)
        k_prev = estimar_k_previsto(df_cand)
        estado_k_prev = classificar_k_valor(k_prev)
        st.markdown("### 🧭 k previsto (reativo, baseado nos candidatos)")
        st.info(rotulo_k(estado_k_prev))

        # k* preditivo (sentinela estrutural)
        k_star_estado = st.session_state["k_star_estado"]
        k_star_score = st.session_state["k_star_score"]
        st.markdown("### ⚡ k* (sentinela preditivo TURBO++)")
        st.info(rotulo_k_star(k_star_estado, k_star_score))

    else:
        st.info("Execute o pipeline para ver as séries candidatas.")


# ============================================================
# PAINEL 3 — Monitor de Risco (k & k*)
# ============================================================

elif painel == "🚨 Monitor de Risco (k & k*)":
    st.markdown("## 🚨 Monitor de Risco (k & k*)")

    df = st.session_state.get("df", pd.DataFrame())
    if df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    info_k = estado_k_global(df)
    estado = info_k["estado"]
    contagens = info_k["contagens"]

    st.markdown("### 🧮 Distribuição de k histórico")
    col1, col2, col3 = st.columns(3)
    col1.metric("Estável (k=0)", contagens["estavel"])
    col2.metric("Atenção (k=1)", contagens["atencao"])
    col3.metric("Crítico (k≥2)", contagens["critico"])

    st.markdown("### 🌡️ Estado global (k histórico)")
    st.write(rotulo_k(estado))

    st.markdown("### 📈 Série temporal de k histórico")
    graf = df[["idx", "k"]].set_index("idx")
    st.line_chart(graf)

    st.markdown("### ⚡ Último k* calculado (sentinela preditivo)")
    k_star_estado = st.session_state.get("k_star_estado", "indefinido")
    k_star_score = st.session_state.get("k_star_score", 0.0)
    st.info(rotulo_k_star(k_star_estado, k_star_score))

    st.markdown("### 🔍 Últimas séries (com k histórico)")
    st.dataframe(
        df[["idx", "serie", "p1", "p2", "p3", "p4", "p5", "p6", "k"]]
        .tail(30)
        .sort_values("idx", ascending=False)
    )

    st.markdown("### ℹ️ Interpretação")
    st.markdown(
        """
        - **k histórico** (reativo): mede o que já aconteceu na estrada.
        - **k\*** (sentinela preditivo): estima, pela estrutura, se o risco está subindo
          antes de o k real aparecer.
        """
    )


# ============================================================
# PAINEL 4 — Modo TURBO++ — Previsão Final
# ============================================================

elif painel == "🚀 Modo TURBO++ — Previsão Final":
    st.markdown("## 🚀 Modo TURBO++ — Previsão Final")

    df = st.session_state.get("df", pd.DataFrame())
    if df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    df_cand = st.session_state.get("df_candidatos", pd.DataFrame())
    idx_alvo_mem = st.session_state.get("ultimo_idx_alvo", None)

    col1, col2 = st.columns(2)
    with col1:
        if idx_alvo_mem is None:
            idx_min = int(df["idx"].min()) + 1
            idx_max = int(df["idx"].max())
            idx_escolhido = st.number_input(
                "Índice alvo (idx) para o TURBO++:",
                min_value=idx_min,
                max_value=idx_max,
                value=idx_max,
                step=1,
            )
        else:
            st.markdown(f"**Índice alvo (idx) usado no pipeline simples:** `{idx_alvo_mem}`")
            idx_escolhido = idx_alvo_mem
    with col2:
        n_series_turbo = st.number_input(
            "Quantidade de séries na base TURBO++:",
            min_value=10,
            max_value=300,
            value=50,
            step=10,
        )

    st.markdown("---")

    if st.button("Rodar TURBO++"):
        df_cand = executar_pipeline_v14_simples(df, idx_escolhido, n_series_turbo)
        st.session_state["df_candidatos"] = df_cand
        st.session_state["ultimo_idx_alvo"] = int(idx_escolhido)

        # Atualiza k* para este alvo
        kstar = calcular_k_star_estado(df, idx_escolhido)
        st.session_state["k_star_estado"] = kstar["estado"]
        st.session_state["k_star_score"] = kstar["score"]

    df_cand = st.session_state.get("df_candidatos", pd.DataFrame())

    if df_cand is None or df_cand.empty:
        st.info("Rode o TURBO++ para gerar a previsão final.")
        st.stop()

    st.markdown("### 📊 Base interna TURBO++ (candidatos)")
    st.dataframe(df_cand)

    previsao_final = extrair_previsao_final(df_cand)
    k_prev = estimar_k_previsto(df_cand)
    estado_k_prev = classificar_k_valor(k_prev)
    k_star_estado = st.session_state["k_star_estado"]
    k_star_score = st.session_state["k_star_score"]

    st.markdown("### 🎯 Previsão Final TURBO++")
    if previsao_final:
        st.code(" ".join(str(x) for x in previsao_final), language="text")
    else:
        st.warning("Nenhuma previsão final pôde ser gerada.")

    st.markdown("### ⚠️ Contexto de risco (k histórico + k* preditivo)")
    st.info("**k previsto (reativo, herdado dos candidatos):** " + rotulo_k(estado_k_prev))
    st.info("**k\* (sentinela preditivo TURBO++):** " + rotulo_k_star(k_star_estado, k_star_score))

    # Comparação opcional com k real do alvo
    idx_alvo_mem = st.session_state.get("ultimo_idx_alvo", None)
    serie_alvo = df[df["idx"] == idx_alvo_mem] if idx_alvo_mem is not None else pd.DataFrame()
    if not serie_alvo.empty:
        k_real = int(serie_alvo.iloc[0]["k"])
        estado_k_real = classificar_k_valor(k_real)
        st.markdown("### 🔁 Comparação com k histórico da série alvo")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**k histórico (real):**")
            st.write(rotulo_k(estado_k_real))
        with col2:
            st.markdown("**k\* preditivo (TURBO++):**")
            st.write(rotulo_k_star(k_star_estado, k_star_score))

    st.markdown("---")
    st.markdown(
        """
        ℹ️ **Notas desta versão única V14 TURBO++**

        - O motor de previsão ainda está em modo estrutural (baseado em janelas recentes).
        - O fluxo já está pronto para receber o motor completo V14 (IDX/IPF/IPO/S6).
        - O k histórico atua como sentinela reativo.
        - O k\* atua como sentinela preditivo, antecipando risco estrutural.
        """
    )
