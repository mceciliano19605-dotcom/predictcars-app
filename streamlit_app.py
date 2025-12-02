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
    colunas: ['serie', 'p1'..'p6', 'k', 'idx'].
    """
    df = df_raw.copy()
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]

    if df.shape[1] in (7, 8):
        cols_novas = ["serie", "p1", "p2", "p3", "p4", "p5", "p6"]
        if df.shape[1] == 8:
            cols_novas.append("k")
        df.columns = cols_novas
    else:
        colunas = [c.lower() for c in df.columns]
        mapa = {}
        for i, c in enumerate(colunas):
            if "c" in c or "id" in c or "serie" in c:
                mapa["serie"] = df.columns[i]
            elif any(x in c for x in ["k", "risco"]):
                mapa["k"] = df.columns[i]

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

    df["serie"] = df["serie"].astype(str)
    for c in ["p1", "p2", "p3", "p4", "p5", "p6"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "k" not in df.columns:
        df["k"] = 0
    else:
        df["k"] = pd.to_numeric(df["k"], errors="coerce").fillna(0).astype(int)

    df["idx"] = np.arange(1, len(df) + 1)
    df = df.sort_values("idx").reset_index(drop=True)
    return df


def parse_historico_texto(conteudo: str) -> pd.DataFrame:
    """
    Converte texto colado no formato:
    C1;41;5;4;52;30;33;0
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
# k HISTÓRICO — SENTINELA REATIVO (NÃO INFLUENCIA PREVISÃO)
# ============================================================

def classificar_k_valor(k_val: int) -> str:
    if k_val <= 0:
        return "estavel"
    if k_val == 1:
        return "atencao"
    return "critico"


def estado_k_global(df: pd.DataFrame) -> dict:
    if df.empty or "k" not in df.columns:
        return {
            "estado": "indefinido",
            "contagens": {"estavel": 0, "atencao": 0, "critico": 0},
        }

    estados = df["k"].apply(classificar_k_valor)
    contagens = estados.value_counts().to_dict()
    for chave in ["estavel", "atencao", "critico"]:
        contagens.setdefault(chave, 0)

    if contagens["critico"] > 0:
        estado = "critico"
    elif contagens["atencao"] > 0:
        estado = "atencao"
    else:
        estado = "estavel"

    return {"estado": estado, "contagens": contagens}


def rotulo_k(estado: str) -> str:
    if estado == "estavel":
        return "🟢 Ambiente estável — previsão em regime normal."
    if estado == "atencao":
        return "🟡 Pré-ruptura — usar previsão com atenção."
    if estado == "critico":
        return "🔴 Ambiente crítico — usar previsão com cautela máxima."
    return "⚪ Estado de risco indefinido."


# ============================================================
# k* TURBO++ — SENTINELA PREDITIVO (NÃO INFLUENCIA PREVISÃO)
# ============================================================

def _extrair_vetor_passageiros(df_linha: pd.Series) -> np.ndarray:
    return np.array(
        [df_linha["p1"], df_linha["p2"], df_linha["p3"],
         df_linha["p4"], df_linha["p5"], df_linha["p6"]],
        dtype=float
    )


def calcular_metricas_risco_janela(df_janela: pd.DataFrame) -> dict:
    if df_janela is None or df_janela.empty or len(df_janela) < 3:
        return {"turbulencia": 0.0, "dispersao": 0.0}

    diffs = []
    linhas = df_janela.sort_values("idx").reset_index(drop=True)
    for i in range(1, len(linhas)):
        v1 = _extrair_vetor_passageiros(linhas.iloc[i - 1])
        v2 = _extrair_vetor_passageiros(linhas.iloc[i])
        diffs.append(np.linalg.norm(v2 - v1))
    turbulencia = float(np.mean(diffs)) if diffs else 0.0

    mat = np.vstack([_extrair_vetor_passageiros(linhas.iloc[i])
                     for i in range(len(linhas))])
    dispersao = float(np.mean(np.var(mat, axis=0)))

    return {"turbulencia": turbulencia, "dispersao": dispersao}


def normalizar_score(valor: float, referencia_baixa: float, referencia_alta: float) -> float:
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
    if df.empty:
        return {"estado": "indefinido", "score": 0.0}

    idx_alvo = int(idx_alvo)
    idx_min = int(df["idx"].min())
    idx_max = int(df["idx"].max())
    idx_alvo = max(idx_min + 1, min(idx_alvo, idx_max))

    inicio = max(idx_min, idx_alvo - largura_janela)
    df_janela = df[(df["idx"] >= inicio) & (df["idx"] < idx_alvo)].copy()
    if df_janela.empty:
        return {"estado": "indefinido", "score": 0.0}

    met = calcular_metricas_risco_janela(df_janela)

    score_turb = normalizar_score(met["turbulencia"], 0.0, 60.0)
    score_disp = normalizar_score(met["dispersao"], 0.0, 300.0)

    score = 0.6 * score_turb + 0.4 * score_disp

    if score < 0.33:
        estado = "estavel"
    elif score < 0.66:
        estado = "atencao"
    else:
        estado = "critico"

    return {"estado": estado, "score": float(score)}


def rotulo_k_star(estado: str, score: float) -> str:
    score_pct = int(round(score * 100))
    if estado == "estavel":
        return f"🟢 k*: Ambiente tende a permanecer estável (risco ≈ {score_pct}%)."
    if estado == "atencao":
        return f"🟡 k*: Pré-ruptura estrutural detectada (risco ≈ {score_pct}%)."
    if estado == "critico":
        return f"🔴 k*: Alta probabilidade de ruptura ou turbulência forte (risco ≈ {score_pct}%)."
    return "⚪ k*: Estado preditivo indefinido."


# ============================================================
# MOTOR TURBO++ REAL — IDX / IPF / IPO / S6
# (k NÃO É USADO EM NENHUM CÁLCULO)
# ============================================================

def construir_janelas(df: pd.DataFrame, tamanho_janela: int) -> pd.DataFrame:
    """
    Constrói janelas IDX:
    - contexto: sequência de tamanho_janela
    - seguidor: série imediatamente seguinte (p1..p6) e k histórico dessa série.
    """
    linhas = df.sort_values("idx").reset_index(drop=True)
    janelas = []
    max_idx = int(linhas["idx"].max())
    min_idx = int(linhas["idx"].min())

    for i in range(min_idx, max_idx - tamanho_janela):
        inicio = i
        fim = i + tamanho_janela - 1
        seguidor_idx = fim + 1
        if seguidor_idx > max_idx:
            break

        contexto = linhas[(linhas["idx"] >= inicio) & (linhas["idx"] <= fim)]
        seguidor = linhas[linhas["idx"] == seguidor_idx]
        if contexto.empty or seguidor.empty:
            continue

        contexto_mat = np.vstack([_extrair_vetor_passageiros(l) for _, l in contexto.iterrows()])
        seg_vec = _extrair_vetor_passageiros(seguidor.iloc[0])

        janelas.append(
            {
                "inicio_idx": inicio,
                "fim_idx": fim,
                "seguidor_idx": seguidor_idx,
                "contexto_mat": contexto_mat,
                "seguidor_vec": seg_vec,
                "seguidor_serie": [
                    int(seguidor.iloc[0]["p1"]),
                    int(seguidor.iloc[0]["p2"]),
                    int(seguidor.iloc[0]["p3"]),
                    int(seguidor.iloc[0]["p4"]),
                    int(seguidor.iloc[0]["p5"]),
                    int(seguidor.iloc[0]["p6"]),
                ],
                "seguidor_k": int(seguidor.iloc[0]["k"]),
            }
        )
    return pd.DataFrame(janelas)


def vetorizar_contexto_atual(df: pd.DataFrame, idx_alvo: int, tamanho_janela: int) -> np.ndarray:
    """
    Constrói o contexto atual: janela imediatamente antes do idx_alvo.
    """
    idx_alvo = int(idx_alvo)
    idx_min = int(df["idx"].min())
    idx_max = int(df["idx"].max())
    idx_alvo = max(idx_min + 1, min(idx_alvo, idx_max))

    fim = idx_alvo - 1
    inicio = max(idx_min, fim - tamanho_janela + 1)

    contexto = df[(df["idx"] >= inicio) & (df["idx"] <= fim)].sort_values("idx")
    if contexto.empty or len(contexto) < 2:
        return None

    contexto_mat = np.vstack([_extrair_vetor_passageiros(l) for _, l in contexto.iterrows()])
    return contexto_mat


def calcular_similaridade_janelas(janelas_df: pd.DataFrame, contexto_atual: np.ndarray) -> pd.DataFrame:
    """
    IDX: mede similaridade entre contexto atual e cada janela histórica.
    """
    if janelas_df.empty or contexto_atual is None:
        return pd.DataFrame()

    scores = []
    for idx, row in janelas_df.iterrows():
        mat = row["contexto_mat"]
        min_len = min(mat.shape[0], contexto_atual.shape[0])
        if min_len <= 0:
            continue

        ca = contexto_atual[-min_len:, :]
        cb = mat[-min_len:, :]

        dif = ca - cb
        dist = np.linalg.norm(dif)
        sim = 1.0 / (1.0 + dist)
        scores.append((idx, sim))

    if not scores:
        return pd.DataFrame()

    idxs, sims = zip(*scores)
    janelas_df = janelas_df.loc[list(idxs)].copy()
    janelas_df["score_idx"] = sims
    janelas_df = janelas_df.sort_values("score_idx", ascending=False).reset_index(drop=True)
    return janelas_df


def gerar_leque_original(df: pd.DataFrame, idx_alvo: int,
                         tamanho_janela: int = 10, top_janelas: int = 40) -> pd.DataFrame:
    """
    Leque ORIGINAL (IPF bruto): pega seguidores das janelas mais semelhantes (IDX).
    """
    if df.empty:
        return pd.DataFrame()

    janelas = construir_janelas(df, tamanho_janela)
    contexto_atual = vetorizar_contexto_atual(df, idx_alvo, tamanho_janela)
    if janelas.empty or contexto_atual is None:
        return pd.DataFrame()

    janelas_sim = calcular_similaridade_janelas(janelas, contexto_atual)
    if janelas_sim.empty:
        return pd.DataFrame()

    janelas_top = janelas_sim.head(top_janelas).copy()

    registros = []
    for _, row in janelas_top.iterrows():
        registros.append(
            {
                "inicio_idx": row["inicio_idx"],
                "fim_idx": row["fim_idx"],
                "seguidor_idx": row["seguidor_idx"],
                "series": row["seguidor_serie"],
                "score_idx": row["score_idx"],
                "k_hist": row["seguidor_k"],  # apenas informativo
            }
        )

    leque = pd.DataFrame(registros)
    return leque


def corrigir_series(series: list) -> list:
    """
    Correções estruturais simples:
    - ordena
    - remove fora de [1, 60]
    """
    nums = [int(x) for x in series]
    nums = [n for n in nums if 1 <= n <= 60]
    while len(nums) < 6:
        nums.append(nums[-1] if nums else 1)
    nums = nums[:6]
    nums = sorted(nums)
    return nums


def gerar_leque_corrigido(leque_original: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica correções estruturais => IPO simplificado.
    """
    if leque_original.empty:
        return pd.DataFrame()

    leque = leque_original.copy()
    leque["series_corr"] = leque["series"].apply(corrigir_series)
    leque["score_ipo"] = leque["score_idx"]

    return leque


def s6_profundo_flat(leque: pd.DataFrame) -> pd.DataFrame:
    """
    S6 Profundo: achata as séries, agrupa e comprime por:
    - frequência
    - score médio (IDX/IPO)
    - penalização leve por dispersão alta
    """
    if leque.empty:
        return pd.DataFrame()

    registros = []
    for _, row in leque.iterrows():
        s = row.get("series_corr", row["series"])
        chave = tuple(s)
        registros.append(
            {
                "series": chave,
                "score_base": row.get("score_ipo", row.get("score_idx", 0.0)),
            }
        )

    df_flat = pd.DataFrame(registros)
    if df_flat.empty:
        return pd.DataFrame()

    # Dispersão (internamente, sem k)
    def disp(t):
        arr = np.array(t, dtype=float)
        return float(np.var(arr))

    df_flat["freq"] = 1
    df_agg = (
        df_flat.groupby("series")
        .agg({"score_base": "mean", "freq": "sum"})
        .reset_index()
    )
    df_agg["disp"] = df_agg["series"].apply(disp)

    # Score final: combina frequência, score_base e dispersão
    df_agg["score"] = (
       0.5 * df_agg["score_base"] +
       0.4 * (df_agg["freq"] / df_agg["freq"].max()) -
       0.1 * df_agg["disp"].apply(
           lambda v: normalizar_score(v, 0.0, df_agg["disp"].max() or 1.0)
       )
   )
 


    df_agg = df_agg.sort_values("score", ascending=False).reset_index(drop=True)
    return df_agg[["series", "score", "freq", "disp"]]


def unir_leques(leque_original: pd.DataFrame, leque_corrigido: pd.DataFrame) -> pd.DataFrame:
    """
    Leque MISTO: une ORIGINAL + CORRIGIDO e reagrupa.
    """
    if leque_original.empty and leque_corrigido.empty:
        return pd.DataFrame()
    if leque_original.empty:
        base = leque_corrigido.copy()
        base["series_corr"] = base["series_corr"]
        base["score_mix"] = base["score_ipo"]
    elif leque_corrigido.empty:
        base = leque_original.copy()
        base["series_corr"] = base["series"]
        base["score_mix"] = base["score_idx"]
    else:
        lo = leque_original.copy()
        lc = leque_corrigido.copy()

        lo["series_corr"] = lo["series"]
        lo["score_mix"] = lo["score_idx"]

        lc["series_corr"] = lc["series_corr"]
        lc["score_mix"] = lc["score_ipo"]

        base = pd.concat([lo, lc], ignore_index=True)

    registros = []
    for _, row in base.iterrows():
        s = row["series_corr"]
        registros.append({"series": tuple(s), "score_base": row["score_mix"]})

    df_flat = pd.DataFrame(registros)
    if df_flat.empty:
        return pd.DataFrame()

    df_flat["freq"] = 1
    df_agg = (
        df_flat.groupby("series")
        .agg({"score_base": "mean", "freq": "sum"})
        .reset_index()
    )

    def disp(t):
        arr = np.array(t, dtype=float)
        return float(np.var(arr))

    df_agg["disp"] = df_agg["series"].apply(disp)
    df_agg["score"] = (
        0.5 * df_agg["score_base"] +
        0.4 * (df_agg["freq"] / df_agg["freq"].max()) -
        0.1 * df_agg["disp"].apply(
            lambda v: normalizar_score(v, 0.0, df_agg["disp"].max() or 1.0)
        )
     )   
            
    df_agg = df_agg.sort_values("score", ascending=False).reset_index(drop=True)
    return df_agg[["series", "score", "freq", "disp"]]


def executar_pipeline_v14_turbo(
    df: pd.DataFrame,
    idx_alvo: int,
    tamanho_janela: int = 10,
    top_janelas: int = 40,
    n_series_final: int = 50,
) -> dict:
    """
    Motor V14 TURBO++:
    - IDX: janelas por similaridade
    - IPF: leque ORIGINAL
    - IPO: leque CORRIGIDO
    - S6: achatamento profundo
    - MISTO: fusão final
    """
    resultado = {
        "leque_original": pd.DataFrame(),
        "leque_corrigido": pd.DataFrame(),
        "flat_original": pd.DataFrame(),
        "flat_corrigido": pd.DataFrame(),
        "flat_misto": pd.DataFrame(),
        "previsao_final": None,
    }

    if df.empty:
        return resultado

    leque_orig = gerar_leque_original(df, idx_alvo, tamanho_janela, top_janelas)
    if leque_orig.empty:
        return resultado

    leque_corr = gerar_leque_corrigido(leque_orig)

    flat_orig = s6_profundo_flat(leque_orig)
    flat_corr = s6_profundo_flat(leque_corr)
    flat_mix = unir_leques(leque_orig, leque_corr)

    # Limita quantidade final
    def limitar(df_flat):
        if df_flat is None or df_flat.empty:
            return pd.DataFrame()
        return df_flat.head(n_series_final).copy()

    flat_orig = limitar(flat_orig)
    flat_corr = limitar(flat_corr)
    flat_mix = limitar(flat_mix)

    previsao_final = None
    if flat_mix is not None and not flat_mix.empty:
        previsao_final = list(flat_mix.iloc[0]["series"])

    resultado.update(
        {
            "leque_original": leque_orig,
            "leque_corrigido": leque_corr,
            "flat_original": flat_orig,
            "flat_corrigido": flat_corr,
            "flat_misto": flat_mix,
            "previsao_final": previsao_final,
        }
    )
    return resultado


# ============================================================
# LAYOUT — SIDEBAR E ESTADO GLOBAL
# ============================================================

st.sidebar.title("🚗 Predict Cars V14 TURBO++")
st.sidebar.markdown("Versão com k (histórico) e k* (sentinela preditivo).")

painel = st.sidebar.radio(
    "Escolha o painel:",
    [
        "📥 Histórico — Entrada",
        "🔍 Pipeline V14 (Simples/TURBO++)",
        "🚨 Monitor de Risco (k & k*)",
        "🚀 Modo TURBO++ — Painel Completo",
    ],
)

if "df" not in st.session_state:
    st.session_state["df"] = pd.DataFrame()
if "ultimo_idx_alvo" not in st.session_state:
    st.session_state["ultimo_idx_alvo"] = None
if "k_star_estado" not in st.session_state:
    st.session_state["k_star_estado"] = "indefinido"
if "k_star_score" not in st.session_state:
    st.session_state["k_star_score"] = 0.0
if "resultado_turbo" not in st.session_state:
    st.session_state["resultado_turbo"] = {}

df_sessao = st.session_state["df"]


# ============================================================
# PAINEL 1 — Histórico — Entrada
# ============================================================

if painel == "📥 Histórico — Entrada":
    st.markdown("## 📥 Histórico — Entrada")

    opc = st.radio(
        "Como deseja carregar o histórico?",
        ["Enviar arquivo CSV", "Copiar e colar o histórico"],
        horizontal=True,
    )

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
# PAINEL 2 — Pipeline V14 (Simples/TURBO++)
# ============================================================

elif painel == "🔍 Pipeline V14 (Simples/TURBO++)":
    st.markdown("## 🔍 Pipeline V14 — Execução TURBO++ (núcleo)")

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
        tamanho_janela = st.number_input(
            "Tamanho da janela IDX:",
            min_value=5,
            max_value=30,
            value=10,
            step=1,
        )
    with col3:
        top_janelas = st.number_input(
            "Qtd. janelas similares (IDX):",
            min_value=10,
            max_value=100,
            value=40,
            step=5,
        )

    n_series_final = st.slider(
        "Quantidade de séries finais (S6 Profundo):",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
    )

    if st.button("Executar Pipeline V14 TURBO++", type="primary"):
        resultado = executar_pipeline_v14_turbo(
            df,
            idx_alvo,
            tamanho_janela=int(tamanho_janela),
            top_janelas=int(top_janelas),
            n_series_final=int(n_series_final),
        )
        st.session_state["resultado_turbo"] = resultado
        st.session_state["ultimo_idx_alvo"] = int(idx_alvo)

        kstar = calcular_k_star_estado(df, idx_alvo)
        st.session_state["k_star_estado"] = kstar["estado"]
        st.session_state["k_star_score"] = kstar["score"]

    resultado = st.session_state.get("resultado_turbo", {})
    leque_orig = resultado.get("leque_original", pd.DataFrame())
    leque_corr = resultado.get("leque_corrigido", pd.DataFrame())
    flat_mix = resultado.get("flat_misto", pd.DataFrame())
    previsao_final = resultado.get("previsao_final", None)

    if leque_orig is None or leque_orig.empty:
        st.info("Execute o pipeline TURBO++ para ver os resultados.")
        st.stop()

    st.markdown("### 📊 Leque ORIGINAL (IPF bruto)")
    st.dataframe(leque_orig[["seguidor_idx", "series", "score_idx", "k_hist"]])

    if leque_corr is not None and not leque_corr.empty:
        st.markdown("### 🔧 Leque CORRIGIDO (IPO simplificado)")
        st.dataframe(leque_corr[["seguidor_idx", "series_corr", "score_ipo"]])

    if flat_mix is not None and not flat_mix.empty:
        st.markdown("### 🧬 S6 Profundo — Leque MISTO (achado e ranqueado)")
        df_view = flat_mix.copy()
        df_view["series_str"] = df_view["series"].apply(lambda s: " ".join(str(x) for x in s))
        st.dataframe(df_view[["series_str", "score", "freq", "disp"]])

    if previsao_final:
        st.markdown("### 🎯 Núcleo TURBO++ (previsão bruta do motor)")
        st.code(" ".join(str(x) for x in previsao_final), language="text")

    idx_alvo_mem = st.session_state.get("ultimo_idx_alvo", None)
    serie_alvo = df[df["idx"] == idx_alvo_mem] if idx_alvo_mem is not None else pd.DataFrame()
    if not serie_alvo.empty:
        k_real = int(serie_alvo.iloc[0]["k"])
        estado_k_real = classificar_k_valor(k_real)
        st.markdown("### ⚠️ k histórico da série alvo")
        st.info(rotulo_k(estado_k_real))

    k_star_estado = st.session_state["k_star_estado"]
    k_star_score = st.session_state["k_star_score"]
    st.markdown("### ⚡ k* (sentinela preditivo TURBO++)")
    st.info(rotulo_k_star(k_star_estado, k_star_score))


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
# PAINEL 4 — Modo TURBO++ — Painel Completo
# ============================================================

elif painel == "🚀 Modo TURBO++ — Painel Completo":
    st.markdown("## 🚀 Modo TURBO++ — Painel Completo")

    df = st.session_state.get("df", pd.DataFrame())
    if df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    idx_alvo_mem = st.session_state.get("ultimo_idx_alvo", None)
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
        st.markdown(f"**Índice alvo (idx) usado no pipeline TURBO++:** `{idx_alvo_mem}`")
        idx_escolhido = idx_alvo_mem

    n_series_final = st.slider(
        "Quantidade de séries finais (S6 Profundo - painel completo):",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
    )

    st.markdown("---")

    if st.button("Rodar TURBO++ (painel completo)"):
        resultado = executar_pipeline_v14_turbo(
            df,
            idx_escolhido,
            tamanho_janela=10,
            top_janelas=40,
            n_series_final=int(n_series_final),
        )
        st.session_state["resultado_turbo"] = resultado
        st.session_state["ultimo_idx_alvo"] = int(idx_escolhido)

        kstar = calcular_k_star_estado(df, idx_escolhido)
        st.session_state["k_star_estado"] = kstar["estado"]
        st.session_state["k_star_score"] = kstar["score"]

    resultado = st.session_state.get("resultado_turbo", {})
    leque_orig = resultado.get("leque_original", pd.DataFrame())
    leque_corr = resultado.get("leque_corrigido", pd.DataFrame())
    flat_orig = resultado.get("flat_original", pd.DataFrame())
    flat_corr = resultado.get("flat_corrigido", pd.DataFrame())
    flat_mix = resultado.get("flat_misto", pd.DataFrame())
    previsao_final = resultado.get("previsao_final", None)

    if leque_orig is None or leque_orig.empty:
        st.info("Rode o TURBO++ para ver o painel completo.")
        st.stop()

    st.markdown("### 🧱 Leque ORIGINAL (IPF) — séries brutas por similaridade")
    st.dataframe(leque_orig[["seguidor_idx", "series", "score_idx", "k_hist"]])

    if leque_corr is not None and not leque_corr.empty:
        st.markdown("### 🔧 Leque CORRIGIDO (IPO) — séries estruturalmente ajustadas")
        st.dataframe(leque_corr[["seguidor_idx", "series_corr", "score_ipo"]])

    if flat_orig is not None and not flat_orig.empty:
        st.markdown("### 🧬 S6 Profundo — Leque ORIGINAL achatado")
        df_view = flat_orig.copy()
        df_view["series_str"] = df_view["series"].apply(lambda s: " ".join(str(x) for x in s))
        st.dataframe(df_view[["series_str", "score", "freq", "disp"]])

    if flat_corr is not None and not flat_corr.empty:
        st.markdown("### 🧬 S6 Profundo — Leque CORRIGIDO achatado")
        df_view = flat_corr.copy()
        df_view["series_str"] = df_view["series"].apply(lambda s: " ".join(str(x) for x in s))
        st.dataframe(df_view[["series_str", "score", "freq", "disp"]])

    if flat_mix is not None and not flat_mix.empty:
        st.markdown("### 🧬 S6 Profundo — Leque MISTO (ORIGINAL + CORRIGIDO)")
        df_view = flat_mix.copy()
        df_view["series_str"] = df_view["series"].apply(lambda s: " ".join(str(x) for x in s))
        st.dataframe(df_view[["series_str", "score", "freq", "disp"]])

    st.markdown("### 🎯 Previsão Final TURBO++")
    if previsao_final:
        st.code(" ".join(str(x) for x in previsao_final), language="text")
    else:
        st.warning("Nenhuma previsão final pôde ser gerada.")

    idx_alvo_mem = st.session_state.get("ultimo_idx_alvo", None)
    serie_alvo = df[df["idx"] == idx_alvo_mem] if idx_alvo_mem is not None else pd.DataFrame()
    if not serie_alvo.empty:
        k_real = int(serie_alvo.iloc[0]["k"])
        estado_k_real = classificar_k_valor(k_real)
        st.markdown("### 🔁 Comparação com k histórico da série alvo")
        st.info("k histórico (real): " + rotulo_k(estado_k_real))

    k_star_estado = st.session_state["k_star_estado"]
    k_star_score = st.session_state["k_star_score"]
    st.markdown("### ⚡ Contexto de risco (k* preditivo)")
    st.info(rotulo_k_star(k_star_estado, k_star_score))

    st.markdown("---")
    st.markdown(
        """
        ℹ️ Notas:
        - O k histórico e o k* preditivo NUNCA entram na matemática da previsão.
        - A previsão é 100% baseada na estrutura dos 6 passageiros.
        - k e k* servem apenas como contexto de risco para interpretação.
        """
    )
