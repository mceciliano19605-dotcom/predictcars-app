# app_v14_auto.py
# Predict Cars V14 TURBO++ — Modo Automático
# Núcleo V14 + S6/S7 + TVF Adaptativo + QDS Dinâmico + k* Sensível + Auto-Mist

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

# ============================================================
# CONFIG BÁSICA DO APP
# ============================================================

st.set_page_config(
    page_title="Predict Cars V14 TURBO++ (Auto)",
    layout="wide",
)

st.title("🚗 Predict Cars V14 TURBO++ — Modo Automático")
st.caption("Núcleo V14 + S6/S7 + TVF Adaptativo + QDS Dinâmico + k* Sensível + Auto-Mist")

# ============================================================
# FUNÇÕES AUXILIARES — HISTÓRICO
# ============================================================

def parse_pasted_history(text: str) -> pd.DataFrame:
    """
    Converte texto colado (linhas tipo:
    C1;41;5;4;52;30;33;0
    ou
    1;41;5;4;52;30;33;0
    ) em DataFrame numérico.
    """
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # aceita ; ou ,
        if ";" in line:
            parts = line.split(";")
        else:
            parts = line.split(",")

        # tenta tirar prefixo C
        first = parts[0].strip()
        if first.upper().startswith("C"):
            try:
                idx = int(first[1:])
            except ValueError:
                idx = np.nan
        else:
            try:
                idx = int(first)
            except ValueError:
                idx = np.nan

        nums = []
        for p in parts[1:]:
            p = p.strip()
            if p == "":
                nums.append(np.nan)
            else:
                try:
                    nums.append(int(p))
                except ValueError:
                    try:
                        nums.append(float(p))
                    except ValueError:
                        nums.append(np.nan)

        rows.append([idx] + nums)

    if not rows:
        return pd.DataFrame()

    max_len = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_len:
            r.append(np.nan)

    cols = ["idx"] + [f"v{i}" for i in range(1, max_len)]
    df = pd.DataFrame(rows, columns=cols)

    # última coluna pode ser k (0/1, etc.)
    # vamos assumir: idx | p1..pN | k (opcional)
    return df


def prepare_history(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza o histórico:
    - garante coluna 'idx' como índice de série
    - detecta passageiros e possível coluna k.
    """
    df = df_raw.copy()

    # se não houver 'idx', assume primeira coluna
    if "idx" not in df.columns:
        df.columns = ["idx"] + [f"v{i}" for i in range(1, df.shape[1])]

    # garante ordenação por idx
    df = df.sort_values("idx").reset_index(drop=True)

    # tenta identificar se última coluna é k (0/1)
    if df.shape[1] >= 3:
        last_col = df.columns[-1]
        # heurística: se os valores são 0/1 ou poucos inteiros pequenos, tratamos como k
        vals = df[last_col].dropna().unique()
        if len(vals) <= 5 and set(vals).issubset({0, 1, 2, 3}):
            # ok, trata como k
            df.rename(columns={last_col: "k"}, inplace=True)
        else:
            # sem k claro -> deixa sem renomear
            pass

    return df


def get_passenger_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c not in ("idx", "k")]
    return cols


# ============================================================
# FUNÇÕES DE MÉTRICA E PIPELINE
# ============================================================

def series_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distância média absoluta entre duas séries."""
    return float(np.mean(np.abs(a - b)))


def build_context_dataset(df: pd.DataFrame, passenger_cols: List[str]) -> pd.DataFrame:
    """
    Constrói conjunto de contextos:
    cada linha i representa o contexto C_i com o próximo C_{i+1} como alvo (predição).
    """
    contexts = []
    for i in range(len(df) - 1):
        row_now = df.iloc[i]
        row_next = df.iloc[i + 1]
        context = {
            "ctx_idx": int(row_now["idx"]),
            "next_idx": int(row_next["idx"]),
        }
        for c in passenger_cols:
            context[f"ctx_{c}"] = row_now[c]
            context[f"next_{c}"] = row_next[c]
        if "k" in df.columns:
            context["ctx_k"] = row_now.get("k", np.nan)
            context["next_k"] = row_next.get("k", np.nan)
        contexts.append(context)

    return pd.DataFrame(contexts)


def compute_k_state(df: pd.DataFrame, window: int = 50) -> str:
    """
    Estado de k* baseado nos últimos valores da coluna k (se existir).
    Retorna: 'estavel', 'atencao', 'critico'.
    """
    if "k" not in df.columns:
        return "desconhecido"

    recent = df["k"].dropna().tail(window)
    if recent.empty:
        return "desconhecido"

    # assume k como 0/1 (ou contagens pequenas)
    rate = np.mean(recent.values > 0)

    if rate < 0.2:
        return "estavel"
    elif rate < 0.4:
        return "atencao"
    else:
        return "critico"


def compute_structural_sensitivity(distances: np.ndarray, s7_survivors: int, s6_total: int) -> float:
    """
    Sensibilidade Estrutural (SE): 0 a 100.
    Leva em conta:
    - proximidade média das vizinhanças
    - taxa de sobrevivência em S7
    """
    if len(distances) == 0:
        return 0.0

    avg_dist = float(np.mean(distances))
    max_dist = float(np.max(distances)) if np.max(distances) > 0 else 1.0

    # quanto menor a distância, maior a coerência
    coh_dist = 1.0 - min(avg_dist / max_dist, 1.0)

    surv_ratio = s7_survivors / max(s6_total, 1)
    surv_norm = min(surv_ratio / 0.3, 1.0)  # 30% de sobrevivência vira 1.0

    se = 100.0 * (0.6 * coh_dist + 0.4 * surv_norm)
    return float(se)


def classify_regime(se: float, s7_survivors: int) -> str:
    """
    Classifica o regime da estrada:
    - 'normal'
    - 'pre-ruptura'
    - 'ruptura'
    """
    if s7_survivors == 0 or se < 25:
        return "ruptura"
    elif se < 55:
        return "pre-ruptura"
    else:
        return "normal"


def compute_iaq(s6_total: int, s7_survivors: int, se: float) -> float:
    """
    IAQ — Índice de Alinhamento de Quadro (0 a 100).
    Considera:
    - relação S7/S6
    - sensibilidade estrutural
    """
    surv_ratio = s7_survivors / max(s6_total, 1)
    surv_norm = min(surv_ratio / 0.4, 1.0)  # 40% -> 1.0

    se_norm = se / 100.0

    iaq = 100.0 * (0.5 * surv_norm + 0.5 * se_norm)
    return float(iaq)


def compute_qds(iaq: float, se: float, k_state: str) -> float:
    """
    QDS Dinâmico (0 a 100).
    Combina:
    - IAQ
    - Sensibilidade estrutural
    - Estado de k*
    """
    k_factor = 1.0
    if k_state == "estavel":
        k_factor = 1.0
    elif k_state == "atencao":
        k_factor = 0.8
    elif k_state == "critico":
        k_factor = 0.6
    else:
        k_factor = 0.9  # desconhecido

    base = 0.5 * (iaq + se)
    qds = base * k_factor / 100.0 * 100.0
    return float(qds)


def adapt_tvf(regime: str, base_topn: int) -> int:
    """
    TVF Adaptativo: ajusta Top N final conforme o regime.
    """
    if regime == "normal":
        return base_topn
    elif regime == "pre-ruptura":
        return max(16, base_topn // 2)
    else:  # ruptura
        return max(8, base_topn // 4)


def adapt_s6_s7(regime: str, base_s6: int, base_disp: float) -> Tuple[int, float]:
    """
    Ajusta automaticamente S6 (nº de vizinhos) e S7 (dispersão máxima)
    conforme o regime.
    """
    if regime == "normal":
        return base_s6, base_disp
    elif regime == "pre-ruptura":
        return int(base_s6 * 1.5), base_disp + 15.0
    else:  # ruptura
        return int(base_s6 * 2.0), base_disp + 25.0


def apply_s6_s7_tvf(
    ctx_df: pd.DataFrame,
    df: pd.DataFrame,
    idx_alvo: int,
    passenger_cols: List[str],
    s6_limit: int,
    disp_limit: float,
    topn_tvf: int,
) -> Dict[str, Any]:
    """
    Pipeline S6 -> S7 -> TVF para um dado conjunto de parâmetros.
    Retorna infos para cálculo de QDS e previsões.
    """
    # contexto alvo (linha com idx == idx_alvo)
    row_target = df[df["idx"] == idx_alvo]
    if row_target.empty:
        raise ValueError(f"Índice alvo {idx_alvo} não encontrado no histórico.")

    target_vals = row_target.iloc[0][passenger_cols].values.astype(float)

    # candidatos de contexto (todos ctx_idx com próximo conhecido)
    ctx = ctx_df.copy()

    # distância estrutural
    ctx_pass_cols = [f"ctx_{c}" for c in passenger_cols]
    ctx_matrix = ctx[ctx_pass_cols].values.astype(float)

    dists = np.array([series_distance(target_vals, v) for v in ctx_matrix])
    ctx["dist"] = dists

    # S6 — pega vizinhos mais próximos
    ctx_s6 = ctx.sort_values("dist").head(s6_limit).reset_index(drop=True)
    s6_total = len(ctx_s6)

    # monta as séries previstas (S6) a partir do próximo trecho
    series_candidates = []
    for _, r in ctx_s6.iterrows():
        next_vals = [r[f"next_{c}"] for c in passenger_cols]
        series_candidates.append(next_vals)
    series_candidates = np.array(series_candidates, dtype=float)

    # S7 — filtro de dispersão (max - min <= disp_limit)
    dispersions = series_candidates.max(axis=1) - series_candidates.min(axis=1)
    mask_s7 = dispersions <= disp_limit
    s7_indices = np.where(mask_s7)[0]

    s7_total = len(s7_indices)
    if s7_total == 0:
        survivors_df = pd.DataFrame(
            columns=["series", "dist", "disp", "score_tvf"]
        )
        return {
            "target_vals": target_vals,
            "s6_total": s6_total,
            "s7_total": s7_total,
            "distances": dists,
            "survivors_df": survivors_df,
        }

    survivors = series_candidates[s7_indices]
    survivors_dist = dists[s7_indices]
    survivors_disp = dispersions[s7_indices]

    # TVF — scoring simples: combina distância e dispersão
    dist_norm = survivors_dist / (survivors_dist.max() if survivors_dist.max() > 0 else 1.0)
    disp_norm = survivors_disp / (survivors_disp.max() if survivors_disp.max() > 0 else 1.0)

    score_tvf = (0.6 * (1 - dist_norm) + 0.4 * (1 - disp_norm))  # maior = melhor

    survivors_df = pd.DataFrame({
        "series": [list(map(int, s)) for s in survivors],
        "dist": survivors_dist,
        "disp": survivors_disp,
        "score_tvf": score_tvf,
    }).sort_values("score_tvf", ascending=False).reset_index(drop=True)

    survivors_df = survivors_df.head(topn_tvf)

    return {
        "target_vals": target_vals,
        "s6_total": s6_total,
        "s7_total": s7_total,
        "distances": dists,
        "survivors_df": survivors_df,
    }


def run_adaptive_pipeline(
    df: pd.DataFrame,
    idx_alvo: int,
    base_s6: int = 512,
    base_disp: float = 45.0,
    base_topn: int = 128,
) -> Dict[str, Any]:
    """
    Roda pipeline adaptativo:
    1) Tenta com parâmetros base
    2) Diagnostica regime
    3) Se necessário, ajusta S6/S7/TVF e roda de novo (Auto-Mist)
    4) Calcula SE, IAQ, QDS, k* sensível
    """
    passenger_cols = get_passenger_columns(df)
    ctx_df = build_context_dataset(df, passenger_cols)
    k_state_global = compute_k_state(df)

    logs = []

    # --------- PASSO 1: parâmetros base ----------
    res_base = apply_s6_s7_tvf(
        ctx_df, df, idx_alvo, passenger_cols,
        s6_limit=base_s6,
        disp_limit=base_disp,
        topn_tvf=base_topn,
    )

    se_base = compute_structural_sensitivity(
        res_base["distances"],
        res_base["s7_total"],
        res_base["s6_total"],
    )
    regime_base = classify_regime(se_base, res_base["s7_total"])
    iaq_base = compute_iaq(res_base["s6_total"], res_base["s7_total"], se_base)
    qds_base = compute_qds(iaq_base, se_base, k_state_global)

    logs.append({
        "etapa": "Base",
        "s6": base_s6,
        "disp": base_disp,
        "topn": base_topn,
        "se": se_base,
        "iaq": iaq_base,
        "qds": qds_base,
        "regime": regime_base,
        "s6_total": res_base["s6_total"],
        "s7_total": res_base["s7_total"],
    })

    # se já é regime normal com S7 aceitável, usamos esse resultado
    if (regime_base == "normal") and (res_base["s7_total"] > 0):
        final_res = res_base
        final_regime = regime_base
        final_se = se_base
        final_iaq = iaq_base
        final_qds = qds_base
        final_params = (base_s6, base_disp, base_topn)
    else:
        # --------- PASSO 2: parâmetros adaptados ----------
        adapt_s6, adapt_disp = adapt_s6_s7(regime_base, base_s6, base_disp)
        adapt_topn = adapt_tvf(regime_base, base_topn)

        res_adapt = apply_s6_s7_tvf(
            ctx_df, df, idx_alvo, passenger_cols,
            s6_limit=adapt_s6,
            disp_limit=adapt_disp,
            topn_tvf=adapt_topn,
        )

        se_adapt = compute_structural_sensitivity(
            res_adapt["distances"],
            res_adapt["s7_total"],
            res_adapt["s6_total"],
        )
        regime_adapt = classify_regime(se_adapt, res_adapt["s7_total"])
        iaq_adapt = compute_iaq(res_adapt["s6_total"], res_adapt["s7_total"], se_adapt)
        qds_adapt = compute_qds(iaq_adapt, se_adapt, k_state_global)

        logs.append({
            "etapa": "Adaptado",
            "s6": adapt_s6,
            "disp": adapt_disp,
            "topn": adapt_topn,
            "se": se_adapt,
            "iaq": iaq_adapt,
            "qds": qds_adapt,
            "regime": regime_adapt,
            "s6_total": res_adapt["s6_total"],
            "s7_total": res_adapt["s7_total"],
        })

        # escolhe o melhor resultado (maior QDS, desde que haja pelo menos 1 sobrevivente ou seja melhor que base)
        if (res_adapt["s7_total"] > 0 and qds_adapt >= qds_base) or res_base["s7_total"] == 0:
            final_res = res_adapt
            final_regime = regime_adapt
            final_se = se_adapt
            final_iaq = iaq_adapt
            final_qds = qds_adapt
            final_params = (adapt_s6, adapt_disp, adapt_topn)
        else:
            final_res = res_base
            final_regime = regime_base
            final_se = se_base
            final_iaq = iaq_base
            final_qds = qds_base
            final_params = (base_s6, base_disp, base_topn)

    survivors_df = final_res["survivors_df"]

    if survivors_df.empty:
        previsao_final = None
    else:
        # Previsão Final TURBO++ = melhor série no TVF adaptativo
        previsao_final = survivors_df.iloc[0]["series"]

    return {
        "idx_alvo": idx_alvo,
        "passenger_cols": passenger_cols,
        "target_vals": final_res["target_vals"],
        "survivors_df": survivors_df,
        "previsao_final": previsao_final,
        "regime": final_regime,
        "se": final_se,
        "iaq": final_iaq,
        "qds": final_qds,
        "k_state": k_state_global,
        "params": final_params,
        "logs": logs,
        "s6_total": final_res["s6_total"],
        "s7_total": final_res["s7_total"],
    }


def format_k_state(k_state: str) -> str:
    if k_state == "estavel":
        return "🟢 k*: Ambiente estável."
    elif k_state == "atencao":
        return "🟡 k*: Pré-ruptura residual — atenção."
    elif k_state == "critico":
        return "🔴 k*: Ambiente crítico — máxima cautela."
    else:
        return "⚪ k*: Estado de k* desconhecido."


def format_regime(regime: str) -> str:
    if regime == "normal":
        return "🟢 Regime: Estrada em regime normal (núcleo estável)."
    elif regime == "pre-ruptura":
        return "🟡 Regime: Pré-ruptura — vizinhança estrutural limitada."
    else:
        return "🔴 Regime: Ruptura estrutural — histórico pouco confiável para estrutura fina."


def format_qds_label(qds: float) -> str:
    if qds >= 70:
        return "🟢 QDS alto — previsão estrutural forte."
    elif qds >= 40:
        return "🟡 QDS médio — usar com atenção."
    else:
        return "🔴 QDS baixo — usar apenas como referência."


# ============================================================
# PAINEL 1 — Histórico — Entrada
# ============================================================

st.sidebar.header("Painéis")
painel = st.sidebar.radio(
    "Escolha o painel:",
    ["📥 Histórico — Entrada", "🧠 Pipeline V14 — Execução Automática"],
)

if "df_hist" not in st.session_state:
    st.session_state["df_hist"] = None

if painel == "📥 Histórico — Entrada":
    st.markdown("## 📥 Histórico — Entrada")

    df_hist = None

    opc = st.radio(
        "Como deseja carregar o histórico?",
        ["Enviar arquivo CSV", "Copiar e colar o histórico"],
    )

    if opc == "Enviar arquivo CSV":
        file = st.file_uploader("Selecione o arquivo CSV:", type=["csv"])
        if file is not None:
            try:
                df_raw = pd.read_csv(file, sep=None, engine="python")
                df_hist = prepare_history(df_raw)
                st.success("Histórico carregado e preparado com sucesso!")
                st.write("Prévia do histórico (5 últimas linhas):")
                st.dataframe(df_hist.tail(5), use_container_width=True)
                st.session_state["df_hist"] = df_hist
            except Exception as e:
                st.error(f"Erro ao carregar CSV: {e}")

    else:
        text = st.text_area(
            "Cole aqui o histórico (linhas tipo C1;41;5;4;52;30;33;0):",
            height=250,
        )
        if st.button("Processar histórico colado"):
            try:
                df_raw = parse_pasted_history(text)
                df_hist = prepare_history(df_raw)
                st.success("Histórico colado e preparado com sucesso!")
                st.write("Prévia do histórico (5 últimas linhas):")
                st.dataframe(df_hist.tail(5), use_container_width=True)
                st.session_state["df_hist"] = df_hist
            except Exception as e:
                st.error(f"Erro ao processar texto colado: {e}")

# ============================================================
# PAINEL 2 — Pipeline V14 — Execução Automática
# ============================================================

if painel == "🧠 Pipeline V14 — Execução Automática":
    st.markdown("## 🧠 Pipeline V14 — Execução Completa (Automática)")

    df_hist = st.session_state.get("df_hist", None)

    if df_hist is None or df_hist.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    passenger_cols = get_passenger_columns(df_hist)

    col1, col2, col3 = st.columns(3)
    with col1:
        idx_min = int(df_hist["idx"].min())
        idx_max = int(df_hist["idx"].max() - 1)  # não deixa escolher o último (precisa do próximo para contexto)
        idx_alvo = st.number_input(
            "Selecione o índice alvo (C atual):",
            min_value=idx_min,
            max_value=idx_max,
            value=idx_max,
            step=1,
        )
    with col2:
        base_s6 = st.number_input(
            "Máx. séries em S6 (vizinhança base):",
            min_value=32,
            max_value=4096,
            value=512,
            step=32,
        )
    with col3:
        base_disp = st.number_input(
            "Dispersão máxima em S7 (max - min) — base:",
            min_value=10.0,
            max_value=100.0,
            value=45.0,
            step=1.0,
        )

    col4, col5 = st.columns(2)
    with col4:
        base_topn = st.number_input(
            "Top N final pelo TVF — base:",
            min_value=8,
            max_value=512,
            value=128,
            step=8,
        )
    with col5:
        auto_run = st.checkbox("Ativar Auto-Mist / QDS Dinâmico (sempre ligado)", value=True)

    st.markdown("---")

    if st.button("🚀 Rodar Pipeline V14 TURBO++ (Automático)"):
        try:
            result = run_adaptive_pipeline(
                df_hist,
                idx_alvo=idx_alvo,
                base_s6=base_s6,
                base_disp=base_disp,
                base_topn=base_topn,
            )

            # ==================================================
            # SEÇÃO 1 — Série Alvo e Núcleo
            # ==================================================
            st.subheader("🏁 Série Alvo (C atual)")

            row_target = df_hist[df_hist["idx"] == idx_alvo].iloc[0]
            st.write(f"Índice alvo: **C{idx_alvo}**")

            serie_alvo = [int(row_target[c]) for c in passenger_cols]
            st.code(" ".join(str(x) for x in serie_alvo), language="text")

            # ==================================================
            # SEÇÃO 2 — Diagnóstico Automático
            # ==================================================
            st.subheader("🧪 Diagnóstico Automático da Estrada")

            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Séries em S6", result["s6_total"])
            with col_b:
                st.metric("Séries após S7", result["s7_total"])
            with col_c:
                st.metric("Sensibilidade Estrutural (SE)", f"{result['se']:.1f}")
            with col_d:
                st.metric("IAQ", f"{result['iaq']:.1f}")

            st.markdown(format_regime(result["regime"]))
            st.markdown(format_k_state(result["k_state"]))

            st.subheader("📊 QDS Dinâmico")
            st.metric("QDS — Qualidade Dinâmica da Série", f"{result['qds']:.1f}")
            st.markdown(format_qds_label(result["qds"]))

            s6_final, disp_final, topn_final = result["params"]
            st.markdown(
                f"**Parâmetros finais usados (após Auto-Mist):** "
                f"S6 = {s6_final}, S7 (disp máx) = {disp_final:.1f}, Top N TVF = {topn_final}"
            )

            # Logs das etapas (base vs adaptado)
            with st.expander("Ver detalhes do Auto-Mist (etapas base vs adaptada)"):
                logs_df = pd.DataFrame(result["logs"])
                st.dataframe(logs_df, use_container_width=True)

            # ==================================================
            # SEÇÃO 3 — Ranking de Séries (TVF Adaptativo)
            # ==================================================
            st.subheader("📈 Ranking de Séries (TVF Adaptativo)")

            survivors_df = result["survivors_df"]

            if survivors_df.empty:
                st.warning(
                    "Nenhuma série passou pelos filtros S6/S7 mesmo com ajustes automáticos.\n\n"
                    "Isso indica **ruptura estrutural severa**. "
                    "Use apenas análises qualitativas — previsões numéricas não são confiáveis aqui."
                )
            else:
                # Mostra top 20
                st.write("Top séries segundo TVF adaptativo (máx. 20):")
                show_df = survivors_df.head(20).copy()
                show_df["series"] = show_df["series"].apply(
                    lambda s: " ".join(str(int(x)) for x in s)
                )
                st.dataframe(show_df, use_container_width=True)

            # ==================================================
            # SEÇÃO 4 — Previsão Final TURBO++
            # ==================================================
            st.subheader("🎯 Previsão Final TURBO++")

            if result["previsao_final"] is None:
                st.error(
                    "❌ Não foi possível gerar uma Previsão Final TURBO++ com QDS mínimo aceitável.\n\n"
                    "Motivo provável: ruptura estrutural do trecho atual + histórico pouco informativo."
                )
            else:
                serie_prev = result["previsao_final"]
                st.code(" ".join(str(int(x)) for x in serie_prev), language="text")

                resumo_k = format_k_state(result["k_state"])
                resumo_regime = format_regime(result["regime"])
                resumo_qds = format_qds_label(result["qds"])

                st.markdown("**Contexto da previsão:**")
                st.markdown(f"- {resumo_regime}")
                st.markdown(f"- {resumo_k}")
                st.markdown(f"- {resumo_qds}")

                st.info(
                    "Interpretação operacional:\n"
                    "• Se QDS ≥ 70 → cenário forte para uso estrutural da previsão.\n"
                    "• Se 40 ≤ QDS < 70 → use como apoio, combinando com outras leituras.\n"
                    "• Se QDS < 40 → usar apenas como referência qualitativa, sem confiar em acertos diretos."
                )

        except Exception as e:
            st.error(f"Erro ao rodar o pipeline: {e}")
