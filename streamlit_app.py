import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any

# ============================================================
# CONFIGURAÇÃO INICIAL
# ============================================================
st.set_page_config(
    page_title="Predict Cars V14 TURBO++",
    layout="wide",
)

st.markdown(
    """
    # Predict Cars V14 TURBO++ — Modo Estrutural + k como Risco

    Núcleo V14 + S6/S7 + TVF + Backtest Básico + AIQ + QDS + k* (camada de risco)

    - A **previsão estrutural** (listas e séries) é feita **sem o k**.
    - O **k** é usado apenas como **farol de risco** (clima / confiança).
    """
)

# ============================================================
# FUNÇÕES DE SUPORTE
# ============================================================

def preparar_historico(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara o histórico para o formato interno:
    colunas: idx, p1..p6, k
    
    Aceita:
    - CSV com ; ou , separando
    - Primeira coluna podendo ser 'C1', 'C2' etc ou apenas índices numéricos
    - Última coluna sendo k (0/1) ou ausente (nesse caso assume k=0)
    """
    df = df_raw.copy()

    # Se tiver apenas 7 colunas: idx + 6 passageiros
    # Se tiver 8 colunas: idx + 6 passageiros + k
    # Tenta identificar automaticamente.
    n_cols = df.shape[1]
    if n_cols < 7:
        raise ValueError("Histórico precisa ter pelo menos 7 colunas (idx + 6 passageiros).")

    # Renomear colunas genéricas
    cols = list(df.columns)
    rename_map = {}

    # idx na primeira coluna
    rename_map[cols[0]] = "idx"

    # passageiros
    for i in range(1, 7):
        rename_map[cols[i]] = f"p{i}"

    # k (se existir)
    if n_cols >= 8:
        rename_map[cols[7]] = "k"

    df = df.rename(columns=rename_map)

    # Se não tiver k, cria k=0
    if "k" not in df.columns:
        df["k"] = 0

    # Normalizar idx: remover prefixo tipo 'C'
    def parse_idx(x):
        try:
            if isinstance(x, str) and x.upper().startswith("C"):
                return int(x[1:])
            return int(x)
        except Exception:
            return np.nan

    df["idx"] = df["idx"].apply(parse_idx)
    df = df.dropna(subset=["idx"])
    df["idx"] = df["idx"].astype(int)

    # Ordenar por idx
    df = df.sort_values("idx").reset_index(drop=True)

    # Garantir tipos numéricos para passageiros e k
    for col in [f"p{i}" for i in range(1, 7)]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["k"] = pd.to_numeric(df["k"], errors="coerce").fillna(0).astype(int)

    return df


def extrair_vetor(df: pd.DataFrame, idx_alvo: int) -> np.ndarray:
    """Retorna o vetor [p1..p6] do índice alvo."""
    linha = df[df["idx"] == idx_alvo]
    if linha.empty:
        raise ValueError(f"Índice {idx_alvo} não encontrado no histórico.")
    return linha[[f"p{i}" for i in range(1, 7)]].values[0]


def calcular_distancias(df: pd.DataFrame, idx_alvo: int) -> pd.DataFrame:
    """
    Calcula distância estrutural simples (euclidiana) entre a série alvo e todas as anteriores.
    Não usa k.
    """
    alvo = extrair_vetor(df, idx_alvo)
    candidatos = df[df["idx"] < idx_alvo].copy()  # só usa passado

    if candidatos.empty:
        raise ValueError("Não há séries anteriores suficientes para calcular IDX/S6.")

    mat = candidatos[[f"p{i}" for i in range(1, 7)]].values
    dists = np.linalg.norm(mat - alvo, axis=1)

    candidatos["dist"] = dists
    # Dispersão (max-min) da série candidata
    candidatos["disp"] = mat.max(axis=1) - mat.min(axis=1)
    return candidatos


def construir_s6_s7(
    df: pd.DataFrame, idx_alvo: int, s6_max: int, s7_disp_max: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constrói S6 (vizinhança) e S7 (filtro por dispersão).
    """
    base = calcular_distancias(df, idx_alvo)
    s6 = base.sort_values("dist").head(s6_max).reset_index(drop=True)
    s7 = s6[s6["disp"] <= s7_disp_max].reset_index(drop=True)
    return s6, s7


def calcular_tvf(s7: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula um TVF adaptativo simples:
    - score_tvf = 1 / (1 + dist) * (1 / (1 + disp_normalizada))
    """
    df = s7.copy()
    if df.empty:
        return df

    # Normalizar dist e disp para [0,1] (evita explosões)
    df["dist_n"] = (df["dist"] - df["dist"].min()) / (df["dist"].max() - df["dist"].min() + 1e-9)
    df["disp_n"] = (df["disp"] - df["disp"].min()) / (df["disp"].max() - df["disp"].min() + 1e-9)

    df["score_tvf"] = (1.0 / (1.0 + df["dist_n"])) * (1.0 / (1.0 + df["disp_n"]))

    # Série de previsão = próxima linha do histórico após cada vizinho (se existir)
    series_list = []
    for _, row in df.iterrows():
        idx_viz = int(row["idx"])
        prox = df_sessao[df_sessao["idx"] == idx_viz + 1]
        if prox.empty:
            series_list.append(None)
        else:
            series_list.append(
                list(prox[[f"p{i}" for i in range(1, 7)]].values[0]) + [int(prox["k"].values[0])]
            )

    df["series"] = series_list
    df = df[~df["series"].isna()].reset_index(drop=True)

    return df.sort_values("score_tvf", ascending=False).reset_index(drop=True)


def calcular_se_iaq(s6: pd.DataFrame) -> Tuple[float, float]:
    """
    SE (sensibilidade estrutural) e IAQ (índice de alinhamento qualitativo simples).
    Versão simplificada baseada em variância e consistência.
    """
    if s6.empty:
        return 0.0, 0.0

    # Quanto menor a média de distâncias, maior o SE
    dist = s6["dist"].values
    se = float(max(0.0, 100.0 - 100.0 * dist.mean() / (dist.max() + 1e-9)))

    # IAQ: combinação da variância de dispersão + compactação de distâncias
    disp = s6["disp"].values
    if len(disp) > 1:
        var_disp = np.var(disp)
    else:
        var_disp = 0.0
    iaq = float(max(0.0, 100.0 - 50.0 * var_disp / (disp.max() - disp.min() + 1e-9)))

    return se, iaq


def calcular_k_estado(df: pd.DataFrame, janela: int = 50) -> Tuple[str, float]:
    """
    Calcula o estado do k* com base nos últimos 'janela' valores de k.
    - retorna (estado, frequencia_1_em_%)
    """
    if df.empty:
        return "desconhecido", 0.0

    ult = df.tail(janela)
    freq1 = ult["k"].mean() * 100.0  # porcentagem de k=1

    if freq1 < 10:
        estado = "estavel"
    elif freq1 < 25:
        estado = "atencao"
    else:
        estado = "critico"

    return estado, float(freq1)


def calcular_qds(se: float, iaq: float, k_estado: str) -> float:
    """
    QDS simples baseado em SE, IAQ e penalização pelo k*.
    - SE e IAQ são 0-100.
    - Penaliza um pouco se k_estado é crítico.
    """
    base = 0.6 * se + 0.4 * iaq  # peso maior para SE
    if k_estado == "estavel":
        fator = 1.0
    elif k_estado == "atencao":
        fator = 0.9
    else:
        fator = 0.75  # penaliza, mas NÃO altera a previsão, só a confiança

    return float(base * fator / 100.0 * 100.0)  # mantém escala 0-100


def descrever_regime(se: float, iaq: float) -> str:
    """
    Regime da estrada baseado em SE e IAQ.
    """
    if se >= 65 and iaq >= 75:
        return "Estrada em regime normal (núcleo estável)."
    elif se >= 50 and iaq >= 60:
        return "Estrada em regime intermediário (atenção moderada)."
    else:
        return "Estrada em regime turbulento / instável."


def descrever_k_estado(k_estado: str, freq1: float) -> str:
    if k_estado == "estavel":
        return f"🟢 k*: Ambiente estável — poucos eventos críticos ({freq1:.1f}% de k=1)."
    elif k_estado == "atencao":
        return f"🟡 k*: Pré-ruptura / atenção — moderada frequência de k=1 ({freq1:.1f}%)."
    else:
        return f"🔴 k*: Ambiente crítico — alta frequência de k=1 ({freq1:.1f}%)."


def descrever_qds(qds: float) -> str:
    if qds >= 70:
        return f"🟢 QDS alto ({qds:.1f}) — cenário forte para uso estrutural da previsão."
    elif qds >= 40:
        return f"🟡 QDS médio ({qds:.1f}) — use a previsão como apoio, com atenção."
    else:
        return f"🔴 QDS baixo ({qds:.1f}) — usar apenas como referência qualitativa."


# ============================================================
# ESTADO GLOBAL DO HISTÓRICO
# ============================================================

if "df" not in st.session_state:
    st.session_state["df"] = None

# Atalho interno para funções que usam o histórico
df_sessao: pd.DataFrame = st.session_state["df"] if st.session_state["df"] is not None else pd.DataFrame()

# ============================================================
# MENU DE PAINÉIS
# ============================================================

painel = st.sidebar.selectbox(
    "Painéis",
    [
        "📥 Histórico — Entrada",
        "🧠 Núcleo Estrutural — IDX / S6 / S7",
        "🎯 Previsão Estrutural Pura",
        "🌡 Painel de Risco k* / QDS",
        "🔍 Pipeline V14 — Execução Completa (Automático)",
        "📜 Logs / Debug Básico",
    ],
)


# ============================================================
# PAINEL 1 — HISTÓRICO — ENTRADA
# ============================================================

if painel == "📥 Histórico — Entrada":
    st.markdown("## 📥 Histórico — Entrada")

    df = None

    opc = st.radio(
        "Como deseja carregar o histórico?",
        ["Enviar arquivo CSV", "Copiar e colar o histórico (texto bruto)"],
    )

    if opc == "Enviar arquivo CSV":
        file = st.file_uploader("Selecione o arquivo CSV:", type=["csv"])
        if file is not None:
            try:
                # tenta ; depois ,
                try:
                    df_raw = pd.read_csv(file, sep=";")
                except Exception:
                    file.seek(0)
                    df_raw = pd.read_csv(file, sep=",")
                df = preparar_historico(df_raw)
                st.success("Histórico carregado e preparado com sucesso!")
                st.session_state["df"] = df
                st.write("Prévia do histórico preparado:")
                st.dataframe(df.head(20))
            except Exception as e:
                st.error(f"Erro ao carregar CSV: {e}")

    else:
        txt = st.text_area(
            "Cole aqui o histórico (linhas como C1;41;5;4;52;30;33;0):",
            height=200,
        )
        if st.button("Processar histórico colado"):
            if not txt.strip():
                st.warning("Cole algum conteúdo primeiro.")
            else:
                try:
                    # Converte texto para CSV temporário em memória
                    linhas = [l.strip() for l in txt.splitlines() if l.strip()]
                    data = [l.split(";") for l in linhas]
                    df_raw = pd.DataFrame(data)
                    df = preparar_historico(df_raw)
                    st.success("Histórico colado processado com sucesso!")
                    st.session_state["df"] = df
                    st.write("Prévia do histórico preparado:")
                    st.dataframe(df.head(20))
                except Exception as e:
                    st.error(f"Erro ao processar histórico colado: {e}")

    if st.session_state["df"] is not None:
        st.markdown("### ✅ Histórico atual na sessão")
        st.dataframe(st.session_state["df"].tail(10))


# ============================================================
# PAINEL 2 — NÚCLEO ESTRUTURAL — IDX / S6 / S7
# ============================================================

elif painel == "🧠 Núcleo Estrutural — IDX / S6 / S7":
    st.markdown("## 🧠 Núcleo Estrutural — IDX / S6 / S7")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    col_esq, col_dir = st.columns([1, 1.2])

    with col_esq:
        idx_min = int(df["idx"].min())
        idx_max = int(df["idx"].max()) - 1  # para prever próxima
        idx_alvo = st.number_input(
            "Selecione o índice alvo (C atual):",
            min_value=idx_min,
            max_value=idx_max,
            value=idx_max,
            step=1,
        )

        s6_max = st.number_input(
            "Máx. séries em S6 (vizinhança):",
            min_value=16,
            max_value=2048,
            value=512,
            step=16,
        )
        s7_disp_max = st.number_input(
            "Dispersão máxima em S7 (max - min):",
            min_value=10.0,
            max_value=70.0,
            value=45.0,
            step=1.0,
        )

        if st.button("Calcular Núcleo Estrutural (IDX / S6 / S7)"):
            try:
                s6, s7 = construir_s6_s7(df, idx_alvo, s6_max, s7_disp_max)
                se, iaq = calcular_se_iaq(s6)

                st.session_state["idx_result"] = {
                    "idx_alvo": idx_alvo,
                    "s6": s6,
                    "s7": s7,
                    "se": se,
                    "iaq": iaq,
                }
                st.success("Núcleo Estrutural calculado com sucesso!")
            except Exception as e:
                st.error(f"Erro ao calcular IDX/S6/S7: {e}")

    with col_dir:
        if "idx_result" in st.session_state:
            res = st.session_state["idx_result"]
            idx_alvo = res["idx_alvo"]
            s6 = res["s6"]
            s7 = res["s7"]
            se = res["se"]
            iaq = res["iaq"]

            st.markdown(f"### 🏁 Série Alvo (C{idx_alvo})")
            alvo_vec = extrair_vetor(df, idx_alvo)
            st.code(" ".join(str(int(x)) for x in alvo_vec), language="text")

            st.markdown("### 🔍 Visão da Vizinhança")
            st.write(f"Séries em S6: **{len(s6)}**")
            st.write(f"Séries em S7 (após filtro dispersão): **{len(s7)}**")

            st.markdown("### 📊 Métricas Estruturais")
            st.write(f"Sensibilidade Estrutural (SE): **{se:.1f}**")
            st.write(f"IAQ (qualidade estrutural): **{iaq:.1f}**")
            st.info(descrever_regime(se, iaq))

            with st.expander("Ver detalhes de S6 (vizinhança completa)"):
                st.dataframe(s6[["idx", "p1", "p2", "p3", "p4", "p5", "p6", "dist", "disp"]].head(50))

            with st.expander("Ver detalhes de S7 (após filtro de dispersão)"):
                st.dataframe(s7[["idx", "p1", "p2", "p3", "p4", "p5", "p6", "dist", "disp"]].head(50))
        else:
            st.info("Calcule o núcleo estrutural à esquerda para ver os resultados aqui.")


# ============================================================
# PAINEL 3 — PREVISÃO ESTRUTURAL PURA (SEM k)
# ============================================================

elif painel == "🎯 Previsão Estrutural Pura":
    st.markdown("## 🎯 Previsão Estrutural Pura (SEM k)")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    idx_min = int(df["idx"].min())
    idx_max = int(df["idx"].max()) - 1
    idx_alvo = st.number_input(
        "Selecione o índice alvo (C atual):",
        min_value=idx_min,
        max_value=idx_max,
        value=idx_max,
        step=1,
    )

    s6_max = st.number_input(
        "Máx. séries em S6 (vizinhança base):",
        min_value=16,
        max_value=2048,
        value=512,
        step=16,
    )
    s7_disp_max = st.number_input(
        "Dispersão máxima em S7 (max - min) — base:",
        min_value=10.0,
        max_value=70.0,
        value=45.0,
        step=1.0,
    )
    top_n = st.number_input(
        "Top N final pelo TVF (puro):",
        min_value=10,
        max_value=256,
        value=64,
        step=2,
    )

    if st.button("Gerar Previsão Estrutural Pura"):
        try:
            global df_sessao
            df_sessao = df  # garante visibilidade em calcular_tvf
            s6, s7 = construir_s6_s7(df, idx_alvo, s6_max, s7_disp_max)
            se, iaq = calcular_se_iaq(s6)
            ranking = calcular_tvf(s7)

            if ranking.empty:
                st.error("Nenhuma série candidata encontrada (S7 vazio). Ajuste S6/S7.")
                st.stop()

            ranking_top = ranking.head(int(top_n))

            st.markdown(f"### 🏁 Série Alvo (C{idx_alvo})")
            alvo_vec = extrair_vetor(df, idx_alvo)
            st.code(" ".join(str(int(x)) for x in alvo_vec), language="text")

            st.markdown("### 📊 Núcleo Estrutural (SE / IAQ)")
            st.write(f"SE: **{se:.1f}**")
            st.write(f"IAQ: **{iaq:.1f}**")
            st.info(descrever_regime(se, iaq))

            st.markdown("### 📈 Ranking de Séries (TVF Puro — SEM k)")
            st.write("Top séries segundo TVF (estrutura pura, sem impacto de k*):")
            df_view = ranking_top[["series", "dist", "disp", "score_tvf"]].copy()
            st.dataframe(df_view)

            melhor = ranking_top.iloc[0]
            previsao = melhor["series"]

            st.markdown("### 🎯 Previsão Estrutural Pura — Série #1")
            st.code(" ".join(str(int(x)) for x in previsao[:-1]) + f"  k={previsao[-1]}", language="text")

            st.session_state["previsao_pura"] = {
                "idx_alvo": idx_alvo,
                "se": se,
                "iaq": iaq,
                "ranking": ranking_top,
                "previsao": previsao,
            }

        except Exception as e:
            st.error(f"Erro ao gerar previsão estrutural pura: {e}")


# ============================================================
# PAINEL 4 — PAINEL DE RISCO k* / QDS
# ============================================================

elif painel == "🌡 Painel de Risco k* / QDS":
    st.markdown("## 🌡 Painel de Risco — k* / QDS")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    janela_k = st.number_input(
        "Janela para cálculo de k* (nº de séries recentes):",
        min_value=20,
        max_value=200,
        value=50,
        step=5,
    )

    k_estado, freq1 = calcular_k_estado(df, janela=int(janela_k))
    st.markdown("### 🔎 Estado do k* (clima)")
    st.info(descrever_k_estado(k_estado, freq1))

    st.markdown("### 📊 QDS Dinâmico (apenas exemplo global)")
    # Usa última previsão pura (se existir) para um QDS mais coerente
    if "previsao_pura" in st.session_state:
        se = st.session_state["previsao_pura"]["se"]
        iaq = st.session_state["previsao_pura"]["iaq"]
    else:
        # fallback: calcula sobre último índice
        idx_alvo = int(df["idx"].max()) - 1
        s6, _ = construir_s6_s7(df, idx_alvo, 128, 45.0)
        se, iaq = calcular_se_iaq(s6)

    qds = calcular_qds(se, iaq, k_estado)
    st.write(f"SE: **{se:.1f}** — IAQ: **{iaq:.1f}** — QDS: **{qds:.1f}**")
    st.info(descrever_qds(qds))

    st.markdown(
        """
        **Importante:**  
        - O k* e o QDS **não alteram mais as séries de previsão**.  
        - Eles servem apenas como **camada de risco / confiança** para interpretar as listas estruturais.
        """
    )


# ============================================================
# PAINEL 5 — PIPELINE V14 — EXECUÇÃO COMPLETA (Automático)
# ============================================================

elif painel == "🔍 Pipeline V14 — Execução Completa (Automático)":
    st.markdown("## 🔍 Pipeline V14 — Execução Completa (Automático)")
    st.write("Núcleo V14 + S6/S7 + TVF Puro + k* como camada de risco (sem deformar a previsão).")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    col1, col2 = st.columns([1, 1.4])

    with col1:
        idx_min = int(df["idx"].min())
        idx_max = int(df["idx"].max()) - 1
        idx_alvo = st.number_input(
            "Selecione o índice alvo (C atual):",
            min_value=idx_min,
            max_value=idx_max,
            value=idx_max,
            step=1,
        )

        s6_base = st.number_input(
            "Máx. séries em S6 (vizinhança base):",
            min_value=16,
            max_value=2048,
            value=512,
            step=16,
        )
        s7_disp_base = st.number_input(
            "Dispersão máxima em S7 (max - min) — base:",
            min_value=10.0,
            max_value=70.0,
            value=45.0,
            step=1.0,
        )
        top_n_base = st.number_input(
            "Top N final pelo TVF — base:",
            min_value=10,
            max_value=256,
            value=128,
            step=2,
        )

        if st.button("🚀 Rodar Pipeline V14 TURBO++ (Automático)"):
            try:
                global df_sessao
                df_sessao = df

                # Núcleo estrutural
                s6, s7 = construir_s6_s7(df, idx_alvo, s6_base, s7_disp_base)
                se, iaq = calcular_se_iaq(s6)

                # Ranking TVF puro
                ranking = calcular_tvf(s7)
                if ranking.empty:
                    st.error("Nenhuma série passou por S7 — ajuste parâmetros.")
                    st.stop()
                ranking_top = ranking.head(int(top_n_base))

                # Previsão pura
                melhor = ranking_top.iloc[0]
                previsao_pura = melhor["series"]

                # k* e QDS
                k_estado, freq1 = calcular_k_estado(df, janela=50)
                qds = calcular_qds(se, iaq, k_estado)

                st.session_state["pipeline_v14"] = {
                    "idx_alvo": idx_alvo,
                    "s6": s6,
                    "s7": s7,
                    "se": se,
                    "iaq": iaq,
                    "ranking": ranking_top,
                    "previsao_pura": previsao_pura,
                    "k_estado": k_estado,
                    "freq1": freq1,
                    "qds": qds,
                    "s6_final": len(s6),
                    "s7_final": len(s7),
                    "top_n_final": int(top_n_base),
                }

                st.success("Pipeline V14 TURBO++ executado com sucesso!")
            except Exception as e:
                st.error(f"Erro ao executar pipeline: {e}")

    with col2:
        if "pipeline_v14" in st.session_state:
            res = st.session_state["pipeline_v14"]
            idx_alvo = res["idx_alvo"]
            se = res["se"]
            iaq = res["iaq"]
            ranking_top = res["ranking"]
            previsao_pura = res["previsao_pura"]
            k_estado = res["k_estado"]
            freq1 = res["freq1"]
            qds = res["qds"]
            s6_final = res["s6_final"]
            s7_final = res["s7_final"]
            top_n_final = res["top_n_final"]

            st.markdown(f"### 🏁 Série Alvo (C{idx_alvo})")
            alvo_vec = extrair_vetor(df, idx_alvo)
            st.code(" ".join(str(int(x)) for x in alvo_vec), language="text")

            st.markdown("### 🧪 Diagnóstico Automático da Estrada")
            st.write(f"Séries em S6: **{s6_final}**")
            st.write(f"Séries após S7: **{s7_final}**")
            st.write(f"Sensibilidade Estrutural (SE): **{se:.1f}**")
            st.write(f"IAQ: **{iaq:.1f}**")
            st.info(descrever_regime(se, iaq))

            st.markdown("### 🌡 Clima — k* / QDS (Camada de Risco)")
            st.info(descrever_k_estado(k_estado, freq1))
            st.info(descrever_qds(qds))

            st.markdown(
                f"Parâmetros finais usados (Auto-Mist estrutura pura): S6 = {s6_final}, "
                f"S7 (disp máx) = {float(df['p1'].max() - df['p1'].min()):.1f if not df.empty else 0.0}, "
                f"Top N TVF = {top_n_final}"
            )

            st.markdown("### 📈 Ranking de Séries (TVF Puro — máx. 20)")
            df_view = ranking_top[["series", "dist", "disp", "score_tvf"]].head(20)
            st.dataframe(df_view)

            st.markdown("### 🎯 Previsão Final TURBO++ (Estrutural Pura + Risco Separado)")
            st.code(" ".join(str(int(x)) for x in previsao_pura[:-1]) + f"  k={previsao_pura[-1]}", language="text")

            st.markdown("#### Contexto da previsão:")
            st.write(descrever_regime(se, iaq))
            st.write(descrever_k_estado(k_estado, freq1))
            st.write(descrever_qds(qds))

        else:
            st.info("Rode o pipeline na coluna da esquerda para ver os resultados aqui.")


# ============================================================
# PAINEL 6 — LOGS / DEBUG BÁSICO
# ============================================================

elif painel == "📜 Logs / Debug Básico":
    st.markdown("## 📜 Logs / Debug Básico")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Histórico ainda não carregado.")
    else:
        st.markdown("### 📌 Informações gerais do histórico")
        st.write(f"Nº de linhas: **{len(df)}**")
        st.write(f"Intervalo de índices: **C{int(df['idx'].min())}** até **C{int(df['idx'].max())}**")
        st.write("Prévia (últimas 20 linhas):")
        st.dataframe(df.tail(20))

    if "pipeline_v14" in st.session_state:
        st.markdown("### 🧾 Última execução do Pipeline V14")
        st.json(
            {
                "idx_alvo": st.session_state["pipeline_v14"]["idx_alvo"],
                "s6_final": st.session_state["pipeline_v14"]["s6_final"],
                "s7_final": st.session_state["pipeline_v14"]["s7_final"],
                "se": st.session_state["pipeline_v14"]["se"],
                "iaq": st.session_state["pipeline_v14"]["iaq"],
                "k_estado": st.session_state["pipeline_v14"]["k_estado"],
                "qds": st.session_state["pipeline_v14"]["qds"],
            }
        )
    else:
        st.info("Nenhuma execução do Pipeline V14 registrada ainda.")
