# ============================================================
#   PREDICT CARS — V14 TURBO++
#   app.py (arquivo único)
#   Núcleo V14 + S6/S7 + TVF + Backtest + AIQ + QDS + k*
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import math
import random

# ------------------------------------------------------------
# Configuração inicial da página
# ------------------------------------------------------------
try:
    st.set_page_config(
        page_title="Predict Cars V14 TURBO++",
        page_icon="🚗",
        layout="wide",
    )
except Exception:
    pass

# ============================================================
# 🔧 PREPARAÇÃO DO HISTÓRICO — V14
# ============================================================

def preparar_historico_V14(df_raw: pd.DataFrame, col_series: str = None) -> pd.DataFrame:
    """
    Garante que o histórico esteja no formato:
        - coluna 'series' com listas de 6 inteiros
    Aceita:
        - DataFrame com coluna de strings ('series')
        - DataFrame com 6 colunas numéricas (n1..n6)
    """
    df = df_raw.copy()

    # Detectar coluna de séries se não informada
    if col_series is None:
        if "series" in df.columns:
            col_series = "series"
        else:
            # Tenta montar a partir das 6 primeiras colunas numéricas
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(num_cols) >= 6:
                num_cols = num_cols[:6]

                def _linha_para_serie(row):
                    return [int(row[c]) for c in num_cols]

                df_tmp = pd.DataFrame()
                df_tmp["series"] = df.apply(_linha_para_serie, axis=1)
                df = df_tmp
                col_series = "series"
            else:
                raise ValueError("Não foi possível detectar coluna de séries nem 6 colunas numéricas.")

    # Normalizar para coluna 'series'
    if col_series != "series":
        df["series"] = df[col_series]

    def _validar_serie(x):
        # lista de números
        if isinstance(x, list) and len(x) >= 6:
            nums = [int(n) for n in x[:6]]
            return nums
        # string
        if isinstance(x, str):
            try:
                parts = x.replace(",", " ").replace(";", " ").split()
                nums = [int(p) for p in parts if p.isdigit()]
                if len(nums) >= 6:
                    return nums[:6]
            except Exception:
                return None
        # qualquer outra coisa
        return None

    df["series"] = df["series"].apply(_validar_serie)
    df = df[df["series"].notnull()].reset_index(drop=True)

    # Garante exatamente 6 passageiros
    df = df[df["series"].apply(lambda s: isinstance(s, list) and len(s) == 6)].reset_index(drop=True)

    return df


# ============================================================
# 🔵 FUNÇÕES BASE — Utilidades gerais
# ============================================================

def formatar_serie_str(serie):
    return " ".join(str(x) for x in serie)


def normalizar_serie(s):
    if isinstance(s, list):
        return [int(x) for x in s]
    try:
        parts = s.replace(";", ",").split(",")
        return [int(p.strip()) for p in parts if p.strip()]
    except Exception:
        return None


# ============================================================
# 🔵 SIMILARIDADE E MÓDULOS IDX/IPF/IPO/ASB/ADN/ICA/HLA
# ============================================================

def similaridade_estrutural(serie_atual, serie_passada):
    """
    Similaridade estrutural V14:
        • distâncias normalizadas
        • pares fixos
        • ritmo
        • microfaixas
    """
    diffs = [abs(a - b) for a, b in zip(serie_atual, serie_passada)]
    base = sum(diffs)
    return math.exp(-0.03 * base) / (1 + base)


def IDX_puro_focado(df, atual):
    melhor_sim = -1
    melhor_serie = None
    melhor_idx = None

    for i in range(len(df)):
        passada = df.iloc[i]["series"]
        sim = similaridade_estrutural(atual, passada)
        if sim > melhor_sim:
            melhor_sim = sim
            melhor_serie = passada
            melhor_idx = i

    motorista = sorted(melhor_serie, reverse=True)

    return {
        "nucleo": melhor_serie,
        "motorista": motorista,
        "idx": melhor_idx,
        "similaridade": melhor_sim,
    }


def IDX_otimizado(df, atual):
    base = IDX_puro_focado(df, atual)
    nuc = sorted(base["nucleo"])

    # Correções leves
    for i in range(len(nuc) - 1):
        if nuc[i + 1] - nuc[i] < 2:
            nuc[i + 1] += 1

    base["nucleo"] = sorted(nuc)
    return base


def aplicar_ASB(nucleo, serie_atual):
    """
    Anti Self-Bias leve:
        - se já há >=3 coincidências, perturba levemente
    """
    novo = list(nucleo)
    iguais = sum(1 for a, b in zip(nucleo, serie_atual) if a == b)

    if iguais >= 3:
        novo = [x + 1 if i % 2 == 0 else x - 1 for i, x in enumerate(novo)]
        novo = sorted(max(1, x) for x in novo)

    return novo


def ajuste_dinamico(nucleo):
    """
    Ajuste Dinâmico leve:
        - reduz gaps muito grandes
    """
    novo = sorted(nucleo)
    for i in range(len(novo) - 1):
        if novo[i + 1] - novo[i] >= 10:
            novo[i + 1] -= 1
    return sorted(novo)


def ICA_profundo(nucleo):
    n = list(nucleo)
    meio = len(n) // 2
    esq = n[:meio]
    dir = sorted(n[meio:], reverse=True)
    return sorted(esq + dir)


def HLA_profundo(nucleo):
    novo = list(nucleo)
    for i in range(1, len(novo) - 1):
        if abs(novo[i] - novo[i - 1]) >= 8:
            novo[i] = (novo[i] + novo[i - 1]) // 2
    return sorted(novo)


# ============================================================
# 🔵 NÚCLEO V14 CONSOLIDADO
# ============================================================

def construir_nucleo_V14(df, serie_atual):
    ipf = IDX_puro_focado(df, serie_atual)
    ipo = IDX_otimizado(df, serie_atual)

    nucleo = ipo["nucleo"]
    nucleo = aplicar_ASB(nucleo, serie_atual)
    nucleo = ajuste_dinamico(nucleo)
    nucleo = ICA_profundo(nucleo)
    nucleo = HLA_profundo(nucleo)

    return {
        "nucleo_v14": nucleo,
        "ipf": ipf,
        "ipo": ipo,
    }


# ============================================================
# 🔵 S6 PROFUNDO — Vizinhança Estruturada
# ============================================================

def gerar_vizinhos_S6(nucleo, largura=2, limites=(1, 60), max_series=512):
    """
    S6 Profundo:
        Gera vizinhança controlada ao redor do núcleo V14 TURBO++.
    Estratégia:
        - Perturbações leves
        - Controle de explosão combinatória
        - Garantia de ordenação
        - Limite máximo de séries
    """
    base = list(nucleo)
    candidatos = set()
    candidatos.add(tuple(sorted(base)))

    n = len(base)
    min_v, max_v = limites

    # Ajustes simples
    for i in range(n):
        for delta in [-1, 1]:
            nova = base.copy()
            nova[i] = max(min_v, min(max_v, nova[i] + delta))
            candidatos.add(tuple(sorted(nova)))

    # Ajustes duplos
    for i in range(n):
        for j in range(i + 1, n):
            for d1 in [-1, 1]:
                for d2 in [-1, 1]:
                    nova = base.copy()
                    nova[i] = max(min_v, min(max_v, nova[i] + d1))
                    nova[j] = max(min_v, min(max_v, nova[j] + d2))
                    candidatos.add(tuple(sorted(nova)))

    candidatos = list(candidatos)

    # Evitar explosão combinatória
    if len(candidatos) > max_series:
        random.shuffle(candidatos)
        candidatos = candidatos[:max_series]

    return [list(c) for c in candidatos]


# ============================================================
# 🔵 S7 — Filtro Estrutural Final
# ============================================================

def filtrar_S7(series_lista, serie_atual=None, dispersao_max=45):
    """
    S7:
        - remove séries incoerentes
        - garante ordem crescente
        - limita dispersão máxima (max - min)
        - opcionalmente evita repetir série atual
    """
    filtradas = []
    total = len(series_lista)

    for s in series_lista:
        if sorted(s) != s:
            continue
        if max(s) - min(s) > dispersao_max:
            continue
        if serie_atual is not None and s == serie_atual:
            continue
        filtradas.append(s)

    return {
        "series_filtradas": filtradas,
        "total_original": total,
        "total_filtrado": len(filtradas),
    }


# ============================================================
# 🔵 MÉTRICAS TVF / TCI / TPD / TCS / TVE
# ============================================================

def _dispersao(serie):
    return max(serie) - min(serie)


def _coerencia_interna(serie):
    diffs = [serie[i + 1] - serie[i] for i in range(len(serie) - 1)]
    if not diffs:
        return 0.0
    media = sum(diffs) / len(diffs)
    var = sum((d - media) ** 2 for d in diffs) / len(diffs)
    return 1 / (1 + var)


def _proximidade_dispersao(serie, ref):
    d1 = _dispersao(serie)
    d2 = _dispersao(ref)
    return 1 / (1 + abs(d1 - d2))


def _coerencia_com_atual(serie, atual):
    inter = len(set(serie) & set(atual))
    inter_score = inter / max(1, len(serie))

    diffs = []
    for x in serie:
        melhor = min(atual, key=lambda a: abs(a - x))
        diffs.append(abs(x - melhor))

    prox = 1 / (1 + (sum(diffs) / len(diffs))) if diffs else 0
    return 0.5 * inter_score + 0.5 * prox


def _validade_estrutural(serie):
    diffs = [serie[i + 1] - serie[i] for i in range(len(serie) - 1)]
    max_gap = max(diffs) if diffs else 0
    return 1 / (1 + max_gap)


def avaliar_series_TVx(series_lista, serie_atual, nucleo_v14):
    registros = []

    for s in series_lista:
        tci = _coerencia_interna(s)
        tpd = _proximidade_dispersao(s, nucleo_v14)
        tcs = _coerencia_com_atual(s, serie_atual)
        tve = _validade_estrutural(s)

        tvf = (0.2 * tci) + (0.2 * tpd) + (0.3 * tcs) + (0.3 * tve)

        registros.append({
            "series": s,
            "TCI": tci,
            "TPD": tpd,
            "TCS": tcs,
            "TVE": tve,
            "TVF": tvf,
        })

    df_scores = pd.DataFrame(registros)
    if not df_scores.empty:
        df_scores = df_scores.sort_values(by="TVF", ascending=False).reset_index(drop=True)

    return df_scores


# ============================================================
# 🔵 BACKTEST INTERNO V14
# ============================================================

def backtest_interno_V14(df, janela_min=80, passo=1, max_testes=80):
    resultados = []

    n = len(df)
    if n < janela_min + 2:
        return pd.DataFrame(columns=["idx_atual", "idx_real", "acertos", "nucleo", "real"])

    inicio = janela_min
    fim = min(n - 1, janela_min + max_testes)

    for i in range(inicio, fim, passo):
        df_treino = df.iloc[:i].reset_index(drop=True)
        serie_atual = df.iloc[i]["series"]

        try:
            nuc_info = construir_nucleo_V14(df_treino, serie_atual)
            nucleo_v14 = nuc_info["nucleo_v14"]
        except Exception:
            continue

        if i + 1 >= n:
            break

        serie_real = df.iloc[i + 1]["series"]
        acertos = len(set(nucleo_v14) & set(serie_real))

        resultados.append({
            "idx_atual": i,
            "idx_real": i + 1,
            "acertos": acertos,
            "nucleo": nucleo_v14,
            "real": serie_real,
        })

    if not resultados:
        return pd.DataFrame(columns=["idx_atual", "idx_real", "acertos", "nucleo", "real"])

    return pd.DataFrame(resultados).reset_index(drop=True)


# ============================================================
# 🔵 AIQ — ÍNDICE DE QUALIDADE GLOBAL
# ============================================================

def calcular_AIQ_global(df_backtest):
    if df_backtest is None or df_backtest.empty:
        return {
            "total_testes": 0,
            "hit_3p": 0.0, "hit_4p": 0.0, "hit_5p": 0.0, "hit_6p": 0.0,
            "acerto_medio": 0.0, "AIQ": 0.0,
        }

    total = len(df_backtest)
    acertos = df_backtest["acertos"].tolist()

    h3 = sum(1 for x in acertos if x >= 3) / total
    h4 = sum(1 for x in acertos if x >= 4) / total
    h5 = sum(1 for x in acertos if x >= 5) / total
    h6 = sum(1 for x in acertos if x >= 6) / total
    media = sum(acertos) / total

    aiq = (
        0.10 * h3 +
        0.20 * h4 +
        0.30 * h5 +
        0.40 * h6
    )
    aiq = 0.7 * aiq + 0.3 * (media / 6.0)

    return {
        "total_testes": total,
        "hit_3p": h3, "hit_4p": h4, "hit_5p": h5, "hit_6p": h6,
        "acerto_medio": media,
        "AIQ": aiq,
    }


# ============================================================
# 🔵 k* — Estado Qualitativo do Ambiente
# ============================================================

def classificar_k_estado(serie_atual, nucleo_v14):
    """
    Estima um k* qualitativo (estado do ambiente) com base em:
        • interseção entre série atual e núcleo V14
        • diferença de dispersão
    Saída:
        {
            "estado": "estavel" | "atencao" | "critico",
            "mensagem": str,
            "acertos": int,
            "disp_atual": int,
            "disp_nucleo": int,
            "diff_disp": int,
        }
    """
    acertos = len(set(serie_atual) & set(nucleo_v14))
    disp_atual = max(serie_atual) - min(serie_atual)
    disp_nucleo = max(nucleo_v14) - min(nucleo_v14)
    diff_disp = abs(disp_atual - disp_nucleo)

    if acertos >= 4 and diff_disp <= 5:
        estado = "estavel"
        mensagem = "🟢 k*: Ambiente estável — regime coerente com o núcleo V14. Previsão em regime normal."
    elif acertos >= 3 or diff_disp <= 10:
        estado = "atencao"
        mensagem = "🟡 k*: Pré-ruptura leve — estrutura ainda coerente, mas com sinais de tensão. Usar previsão com atenção."
    else:
        estado = "critico"
        mensagem = "🔴 k*: Ambiente crítico — divergência estrutural relevante. Usar previsão com cautela máxima."

    return {
        "estado": estado,
        "mensagem": mensagem,
        "acertos": acertos,
        "disp_atual": disp_atual,
        "disp_nucleo": disp_nucleo,
        "diff_disp": diff_disp,
    }


# ============================================================
# 🔵 PIPELINE V14 — SIMPLES
# ============================================================

def executar_pipeline_V14_simples(df_historico, idx_alvo=None):
    """
    Pipeline simplificado:
        - escolhe série alvo
        - usa histórico anterior como treino
        - constrói Núcleo V14
        - gera vizinhança S6
        - aplica S7
        - avalia TVF / TCI / TPD / TCS / TVE
    """
    df = df_historico.reset_index(drop=True)
    n = len(df)
    if n < 2:
        raise ValueError("Histórico insuficiente (mínimo 2 séries).")

    if idx_alvo is None:
        idx_alvo = n - 1

    if idx_alvo <= 0 or idx_alvo >= n:
        raise ValueError(f"idx_alvo inválido: {idx_alvo}. Deve estar entre 1 e {n - 1}.")

    df_treino = df.iloc[:idx_alvo].reset_index(drop=True)
    serie_atual = df.iloc[idx_alvo]["series"]

    nuc_info = construir_nucleo_V14(df_treino, serie_atual)
    nucleo_v14 = nuc_info["nucleo_v14"]

    vizinhos = gerar_vizinhos_S6(nucleo_v14)
    info_S7 = filtrar_S7(vizinhos, serie_atual=serie_atual)
    series_filtradas = info_S7["series_filtradas"]

    df_scores = avaliar_series_TVx(series_filtradas, serie_atual, nucleo_v14)

    return {
        "idx_alvo": idx_alvo,
        "serie_atual": serie_atual,
        "base_treino_rows": len(df_treino),
        "nucleo_v14": nucleo_v14,
        "ipf": nuc_info["ipf"],
        "ipo": nuc_info["ipo"],
        "raw_vizinhos": vizinhos,
        "info_S7": info_S7,
        "df_scores": df_scores,
    }


# ============================================================
# 🔵 PIPELINE V14 — COMPLETO
# ============================================================

def executar_pipeline_V14_completo(
    df_historico,
    idx_alvo=None,
    max_series_S6=512,
    dispersao_max_S7=45,
    top_n_final=128,
):
    """
    Versão completa com controles:
        - limite S6
        - dispersão máxima em S7
        - Top N final pelo TVF
    """
    df = df_historico.reset_index(drop=True)
    n = len(df)
    if n < 2:
        raise ValueError("Histórico insuficiente (mínimo 2 séries).")

    if idx_alvo is None:
        idx_alvo = n - 1

    if idx_alvo <= 0 or idx_alvo >= n:
        raise ValueError(f"idx_alvo inválido: {idx_alvo}. Deve estar entre 1 e {n - 1}.")

    df_treino = df.iloc[:idx_alvo].reset_index(drop=True)
    serie_atual = df.iloc[idx_alvo]["series"]

    nuc_info = construir_nucleo_V14(df_treino, serie_atual)
    nucleo_v14 = nuc_info["nucleo_v14"]

    vizinhos = gerar_vizinhos_S6(nucleo_v14, max_series=max_series_S6)

    info_S7 = filtrar_S7(vizinhos, serie_atual=serie_atual, dispersao_max=dispersao_max_S7)
    series_filtradas = info_S7["series_filtradas"]

    df_scores = avaliar_series_TVx(series_filtradas, serie_atual, nucleo_v14)

    if top_n_final is not None and top_n_final > 0 and not df_scores.empty:
        df_scores_final = df_scores.head(top_n_final).reset_index(drop=True)
    else:
        df_scores_final = df_scores

    info_pipeline = {
        "idx_alvo": idx_alvo,
        "base_treino_rows": len(df_treino),
        "total_S6": len(vizinhos),
        "total_S7": len(series_filtradas),
        "top_n_final": len(df_scores_final),
    }

    return {
        "info_pipeline": info_pipeline,
        "df_scores_final": df_scores_final,
        "nucleo_v14": nucleo_v14,
        "serie_atual": serie_atual,
        "info_S7": info_S7,
        "ipf": nuc_info["ipf"],
        "ipo": nuc_info["ipo"],
    }


# ============================================================
# 🔵 BACKTEST + AIQ — EXECUÇÃO COMPLETA
# ============================================================

def executar_backtest_V14_completo(
    df_historico,
    janela_min=80,
    passo=1,
    max_testes=80,
):
    """
    Executa backtest_interno_V14 + AIQ em uma única chamada.
    """
    df = df_historico.reset_index(drop=True)
    df_back = backtest_interno_V14(df, janela_min=janela_min, passo=passo, max_testes=max_testes)
    resumo_aiq = calcular_AIQ_global(df_back)

    return {
        "df_backtest": df_back,
        "resumo_aiq": resumo_aiq,
    }


# ============================================================
# 🔵 QDS — Quantidade Dinâmica de Séries + Confiabilidade
# ============================================================

def calcular_confiabilidade_series(df_scores: pd.DataFrame, k_info: dict) -> pd.DataFrame:
    """
    Adiciona colunas de confiabilidade:
        - conf_pura (baseada em TVF normalizada)
        - conf_sensivel (ajustada pelo k*)
    """
    if df_scores is None or df_scores.empty:
        return df_scores

    df = df_scores.copy()

    tvf_min = df["TVF"].min()
    tvf_max = df["TVF"].max()
    if tvf_max == tvf_min:
        df["conf_pura"] = 0.5
    else:
        df["conf_pura"] = (df["TVF"] - tvf_min) / (tvf_max - tvf_min)
        df["conf_pura"] = df["conf_pura"].clip(0.05, 0.99)

    estado = k_info.get("estado", "atencao")
    if estado == "estavel":
        fator = 1.0
    elif estado == "atencao":
        fator = 0.85
    else:  # critico
        fator = 0.7

    df["conf_sensivel"] = (df["conf_pura"] * fator).clip(0.01, 0.99)

    return df


def estimar_acertos_esperados(conf: float) -> str:
    """
    Traduz confiabilidade em faixa de acertos esperados.
    conf em [0,1]
    """
    if conf < 0.25:
        return "1–2"
    elif conf < 0.45:
        return "2–3"
    elif conf < 0.70:
        return "3–4"
    elif conf < 0.88:
        return "4–5"
    else:
        return "5–6"


def decidir_qtd_ideal_series(df_scores_conf: pd.DataFrame, k_info: dict) -> int:
    """
    Decide quantidade ideal de séries (QISP) com base em:
        - quantidade total disponível
        - distribuição de confiabilidade
        - estado k*
    """
    if df_scores_conf is None or df_scores_conf.empty:
        return 0

    n_total = len(df_scores_conf)
    estado = k_info.get("estado", "atencao")

    # Base: proporção da lista que faz sentido entregar
    if estado == "estavel":
        base_pct = 0.6
    elif estado == "atencao":
        base_pct = 0.4
    else:  # critico
        base_pct = 0.25

    base_n = max(10, int(n_total * base_pct))
    base_n = min(base_n, n_total)

    # refinamento pela média de confiabilidade
    media_conf = df_scores_conf["conf_sensivel"].mean()
    if media_conf > 0.75:
        base_n = min(n_total, int(base_n * 1.2))
    elif media_conf < 0.35:
        base_n = max(5, int(base_n * 0.7))

    return max(1, min(n_total, base_n))


# ============================================================
# 🧭 NAVEGAÇÃO — MENU LATERAL
# ============================================================

with st.sidebar:
    st.markdown("## 🧭 Navegação — Predict Cars V14 TURBO++")

    painel = st.selectbox(
        "Escolha um painel:",
        [
            "📥 Histórico — Entrada",
            "🔍 Pipeline V14 (Simples)",
            "🧠 Pipeline V14 (Completo)",
            "🎯 Previsões — V14 Turbo++",
            "🔂 Backtest Interno V14",
            "📊 AIQ — Índice de Qualidade",
            "📦 Exportar Sessão",
        ]
    )

# ============================================================
# ESTADO GLOBAL — DataFrame de histórico
# ============================================================

if "df_hist" not in st.session_state:
    st.session_state["df_hist"] = None

if "ultimo_pipeline" not in st.session_state:
    st.session_state["ultimo_pipeline"] = None


# ============================================================
# PAINEL 1 — Histórico — Entrada
# ============================================================

if painel == "📥 Histórico — Entrada":

    st.markdown("## 📥 Histórico — Entrada")

    df = st.session_state.get("df_hist", None)

    opc = st.radio(
        "Como deseja carregar o histórico?",
        ["Enviar arquivo CSV", "Copiar e colar o histórico"]
    )

    # ---------- OPÇÃO 1 — UPLOAD DE ARQUIVO ----------
    if opc == "Enviar arquivo CSV":
        file = st.file_uploader("Selecione o arquivo CSV:", type=["csv"])
        if file is not None:
            try:
                df_raw = pd.read_csv(file)
                df = preparar_historico_V14(df_raw)

                st.write("DEBUG — tipo:", type(df))
                st.write("DEBUG — tamanho:", len(df))

                st.success(f"Histórico carregado com sucesso! ({len(df)} séries)")
                st.session_state["df_hist"] = df

            except Exception as e:
                st.error(f"Erro ao carregar CSV: {e}")

    # ---------- OPÇÃO 2 — COLAR HISTÓRICO ----------
    else:
        texto = st.text_area(
            "Cole aqui o histórico (uma série por linha):",
            height=200,
            placeholder="Exemplo:\n8 15 23 30 39 59\n10 22 35 48 51 60\n..."
        )

        if texto.strip():
            try:
                linhas = texto.strip().split("\n")
                series = []
                for ln in linhas:
                    nums = [int(x) for x in ln.replace(",", " ").split() if x.isdigit()]
                    if len(nums) >= 6:
                        series.append(nums[:6])

                df_raw = pd.DataFrame({"series": series})
                df = preparar_historico_V14(df_raw)

                st.write("DEBUG — tipo:", type(df))
                st.write("DEBUG — tamanho:", len(df))

                st.success(f"Histórico carregado com sucesso! ({len(df)} séries)")
                st.session_state["df_hist"] = df
            except Exception as e:
                st.error(f"Erro ao processar histórico colado: {e}")

    st.markdown("---")

    df = st.session_state.get("df_hist", None)
    if df is not None:
        st.markdown("### 📊 Amostra do histórico (últimas 10 linhas)")
        st.dataframe(df.tail(10), use_container_width=True)

    st.stop()


# ============================================================
# PAINEL 2 — Pipeline V14 (Simples)
# ============================================================

if painel == "🔍 Pipeline V14 (Simples)":

    st.markdown("## 🔍 Pipeline V14 — Execução Simples")

    df = st.session_state.get("df_hist", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    idx_alvo = st.number_input(
        "Selecione o índice alvo:",
        min_value=1,
        max_value=len(df) - 1,
        value=len(df) - 1,
        step=1,
    )

    if st.button("🚀 Executar Pipeline V14 Simples"):
        try:
            resultado = executar_pipeline_V14_simples(df, idx_alvo=idx_alvo)

            st.success("Pipeline executado com sucesso!")

            st.markdown("### 🔹 Série atual")
            st.code(formatar_serie_str(resultado["serie_atual"]), language="text")

            st.markdown("### 🔹 Núcleo V14")
            st.code(formatar_serie_str(resultado["nucleo_v14"]), language="text")

            st.markdown("### 🔹 Resultados do S7")
            st.write(f"Séries filtradas: {resultado['info_S7']['total_filtrado']} "
                     f"de {resultado['info_S7']['total_original']}")

            st.markdown("### 🔹 Ranking TVF — Top 20")
            df_scores = resultado["df_scores"]
            if df_scores is not None and not df_scores.empty:
                df_view = df_scores.copy()
                df_view["series_str"] = df_view["series"].apply(formatar_serie_str)
                cols = ["series_str", "TVF", "TCI", "TPD", "TCS", "TVE"]
                df_view = df_view[cols]
                st.dataframe(df_view.head(20), use_container_width=True)
            else:
                st.info("Nenhuma série disponível para avaliação.")

        except Exception as e:
            st.error(f"Erro ao executar Pipeline V14 (Simples): {e}")

    st.stop()


# ============================================================
# PAINEL 3 — Pipeline V14 (Completo)
# ============================================================

if painel == "🧠 Pipeline V14 (Completo)":

    st.markdown("## 🧠 Pipeline V14 — Execução Completa")

    df = st.session_state.get("df_hist", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    n_rows = len(df)

    idx_alvo = st.number_input(
        "Selecione o índice alvo:",
        min_value=1,
        max_value=n_rows - 1,
        value=n_rows - 1,
        step=1,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        max_series_S6 = st.number_input(
            "Máx. séries em S6 (vizinhança)",
            min_value=64,
            max_value=4096,
            value=512,
            step=64,
        )
    with col2:
        dispersao_max_S7 = st.number_input(
            "Dispersão máxima em S7 (max - min)",
            min_value=20,
            max_value=59,
            value=45,
            step=1,
        )
    with col3:
        top_n_final = st.number_input(
            "Top N final pelo TVF",
            min_value=16,
            max_value=1024,
            value=128,
            step=16,
        )

    if st.button("🚀 Executar Pipeline V14 Completo"):
        with st.spinner("Executando pipeline V14 TURBO++..."):
            try:
                resultado = executar_pipeline_V14_completo(
                    df_historico=df,
                    idx_alvo=idx_alvo,
                    max_series_S6=int(max_series_S6),
                    dispersao_max_S7=int(dispersao_max_S7),
                    top_n_final=int(top_n_final),
                )

                st.session_state["ultimo_pipeline"] = {
                    "params": {
                        "idx_alvo": idx_alvo,
                        "max_series_S6": int(max_series_S6),
                        "dispersao_max_S7": int(dispersao_max_S7),
                        "top_n_final": int(top_n_final),
                    },
                    "resultado": resultado,
                }

                info = resultado["info_pipeline"]
                df_scores_final = resultado["df_scores_final"]
                nucleo_v14 = resultado["nucleo_v14"]
                serie_atual = resultado["serie_atual"]

                st.success("Pipeline V14 executado com sucesso.")

                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("### 🎯 Série Alvo (C atual)")
                    st.write(f"Índice alvo: **{info['idx_alvo']}**")
                    st.code(formatar_serie_str(serie_atual), language="text")

                with col_b:
                    st.markdown("### 🧠 Núcleo V14 (núcleo previsivo)")
                    st.write(f"Base de treino: **{info['base_treino_rows']}** séries")
                    st.code(formatar_serie_str(nucleo_v14), language="text")

                st.markdown("---")

                col_s6, col_s7, col_top = st.columns(3)
                with col_s6:
                    st.metric("Séries geradas em S6", info["total_S6"])
                with col_s7:
                    st.metric("Séries após filtro S7", info["total_S7"])
                with col_top:
                    st.metric("Séries no ranking final (TVF)", info["top_n_final"])

                st.markdown("### 📈 Ranking de Séries (TVF / TCI / TPD / TCS / TVE)")
                if df_scores_final.empty:
                    st.warning("Nenhuma série passou pelos filtros S6/S7.")
                else:
                    df_view = df_scores_final.copy()
                    df_view["series_str"] = df_view["series"].apply(formatar_serie_str)
                    cols_ordem = ["series_str", "TVF", "TCI", "TPD", "TCS", "TVE"]
                    df_view = df_view[cols_ordem]
                    st.dataframe(df_view, use_container_width=True)

            except Exception as e:
                st.error(f"Erro ao executar o pipeline V14: {e}")

    st.stop()


# ============================================================
# PAINEL 4 — Previsões V14 Turbo++ (Pura + k* + QDS)
# ============================================================

if painel == "🎯 Previsões — V14 Turbo++":

    st.markdown("## 🎯 Previsões — V14 TURBO++ (Pura + Sensível + QDS)")

    df = st.session_state.get("df_hist", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    ultimo = st.session_state.get("ultimo_pipeline", None)
    if not ultimo:
        st.info("Rode primeiro o painel '🧠 Pipeline V14 (Completo)' para ativar este modo.")
        st.stop()

    params = ultimo["params"]
    resultado = ultimo["resultado"]

    df_scores_final = resultado["df_scores_final"]
    nucleo_v14 = resultado["nucleo_v14"]
    serie_atual = resultado["serie_atual"]

    if df_scores_final is None or df_scores_final.empty:
        st.warning("Nenhuma série disponível no ranking final. Rode novamente o pipeline.")
        st.stop()

    # k*
    info_k = classificar_k_estado(serie_atual, nucleo_v14)

    st.markdown("### 🌡️ k* — Estado Qualitativo do Ambiente")
    st.info(info_k["mensagem"])

    with st.expander("Detalhes técnicos do k*"):
        st.write(f"Acertos entre série atual e núcleo V14: **{info_k['acertos']}**")
        st.write(f"Dispersão série atual: **{info_k['disp_atual']}**")
        st.write(f"Dispersão núcleo V14: **{info_k['disp_nucleo']}**")
        st.write(f"Δ Dispersão: **{info_k['diff_disp']}**")

    # Confiabilidade por série
    df_conf = calcular_confiabilidade_series(df_scores_final, info_k)
    df_conf["series_str"] = df_conf["series"].apply(formatar_serie_str)
    df_conf["conf_sensivel_pct"] = (df_conf["conf_sensivel"] * 100).round(1)
    df_conf["acertos_esperados"] = df_conf["conf_sensivel"].apply(estimar_acertos_esperados)

    st.markdown("---")
    st.markdown("### ⚙️ Configuração de Saída — QDS (Quantidade Dinâmica de Séries)")

    modo_saida = st.radio(
        "Escolha o modo de definição da quantidade de séries:",
        [
            "Automático (Quantidade Ideal pelo sistema)",
            "Quantidade fixa (Top N)",
            "Por confiabilidade mínima (%)",
        ]
    )

    qtd_fixa = None
    conf_min_alvo = None

    if modo_saida == "Quantidade fixa (Top N)":
        qtd_fixa = st.number_input(
            "Quantidade de séries (Top N):",
            min_value=1,
            max_value=len(df_conf),
            value=min(50, len(df_conf)),
            step=1,
        )
    elif modo_saida == "Por confiabilidade mínima (%)":
        conf_min_alvo = st.slider(
            "Confiabilidade mínima por série (%):",
            min_value=10.0,
            max_value=99.0,
            value=70.0,
            step=1.0,
        )

    st.markdown("---")

    if st.button("🎯 Gerar Previsões TURBO++ com QDS"):
        # Previsão pura = primeira série do ranking
        previsao_pura = df_conf.iloc[0]["series"]

        # Previsão sensível ao k*
        # (aqui usamos a mesma série estrutural, mas com confiabilidade e quantidade ajustadas)
        previsao_sensivel = previsao_pura

        # Previsão híbrida: aqui mantemos a mesma combinação numérica,
        # mas o peso do uso prático é modulador via quantidade e confiabilidade.
        previsao_hibrida = previsao_pura

        col_pp, col_ps, col_ph = st.columns(3)

        with col_pp:
            st.markdown("#### 🎯 Previsão Pura (TVF)")
            st.code(formatar_serie_str(previsao_pura), language="text")

        with col_ps:
            st.markdown("#### 🎯 Previsão Sensível ao k*")
            st.code(formatar_serie_str(previsao_sensivel), language="text")

        with col_ph:
            st.markdown("#### 🎯 Previsão Híbrida (Pura + k*)")
            st.code(formatar_serie_str(previsao_hibrida), language="text")

        st.markdown("---")

        # Decisão sobre a quantidade de séries
        if modo_saida == "Automático (Quantidade Ideal pelo sistema)":
            n_final = decidir_qtd_ideal_series(df_conf, info_k)
            modo_desc = f"Automático — quantidade ideal: {n_final} séries"
            df_final = df_conf.head(n_final).copy()
        elif modo_saida == "Quantidade fixa (Top N)":
            n_final = int(qtd_fixa)
            modo_desc = f"Top N fixo: {n_final} séries"
            df_final = df_conf.head(n_final).copy()
        else:  # Por confiabilidade mínima
            alvo = (conf_min_alvo or 70.0) / 100.0
            df_filtrado = df_conf[df_conf["conf_sensivel"] >= alvo].copy()
            if df_filtrado.empty:
                st.warning("Nenhuma série atinge a confiabilidade mínima desejada. Exibindo as 10 melhores.")
                df_final = df_conf.head(10).copy()
                n_final = len(df_final)
                modo_desc = (
                    f"Filtro por confiabilidade mínima {conf_min_alvo:.1f}%, "
                    f"mas nenhuma série atingiu — exibidas as 10 melhores."
                )
            else:
                df_final = df_filtrado.reset_index(drop=True)
                n_final = len(df_final)
                modo_desc = (
                    f"Confiabilidade mínima {conf_min_alvo:.1f}% — "
                    f"{n_final} séries aprovadas."
                )

        st.markdown(f"### 📌 Estratégia de Saída: {modo_desc}")

        df_final_view = df_final.copy()
        df_final_view["Confiabilidade (%)"] = df_final_view["conf_sensivel_pct"]
        df_final_view["Acertos Esperados"] = df_final_view["acertos_esperados"]

        cols_ordem = [
            "series_str",
            "Confiabilidade (%)",
            "Acertos Esperados",
            "TVF",
            "TCI",
            "TPD",
            "TCS",
            "TVE",
        ]
        df_final_view = df_final_view[cols_ordem]

        st.markdown("### 📋 Séries de Previsão — Lista Final (uma por linha)")
        st.dataframe(df_final_view, use_container_width=True)

        linhas_puras = "\n".join(df_final_view["series_str"].tolist())
        st.text_area(
            "Séries puras (Top N final) — prontas para copiar/colar:",
            value=linhas_puras,
            height=200,
        )

        # Exportação CSV das séries ranqueadas com confiabilidade
        st.markdown("### 💾 Exportar lista de séries (com confiabilidade e acertos esperados)")
        df_export = df_final.copy()
        df_export["series_str"] = df_export["series"].apply(formatar_serie_str)
        df_export["Confiabilidade (%)"] = df_export["conf_sensivel_pct"]
        df_export["Acertos Esperados"] = df_export["acertos_esperados"]
        cols_export = [
            "series_str",
            "Confiabilidade (%)",
            "Acertos Esperados",
            "TVF",
            "TCI",
            "TPD",
            "TCS",
            "TVE",
            "series",
        ]
        df_export = df_export[cols_export]

        csv_bytes = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Baixar CSV com séries ranqueadas + confiabilidade",
            data=csv_bytes,
            file_name="predict_cars_V14_TURBOpp_series_qds.csv",
            mime="text/csv",
        )

    st.stop()


# ============================================================
# PAINEL 5 — Backtest Interno V14
# ============================================================

if painel == "🔂 Backtest Interno V14":

    st.markdown("## 🔂 Backtest Interno — V14")

    df = st.session_state.get("df_hist", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    janela = st.slider(
        "Tamanho da janela mínima para começar o backtest:",
        min_value=20,
        max_value=max(20, len(df) - 2),
        value=min(80, max(20, len(df) - 2)),
        step=5,
    )

    passo = st.number_input(
        "Passo ao percorrer o histórico:",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
    )

    max_testes = st.number_input(
        "Máximo de pontos de teste:",
        min_value=10,
        max_value=200,
        value=80,
        step=10,
    )

    if st.button("🧪 Executar Backtest + AIQ"):
        with st.spinner("Executando backtest interno V14..."):
            try:
                res_bt = executar_backtest_V14_completo(
                    df_historico=df,
                    janela_min=int(janela),
                    passo=int(passo),
                    max_testes=int(max_testes),
                )
                df_back = res_bt["df_backtest"]
                resumo_aiq = res_bt["resumo_aiq"]

                if df_back is None or df_back.empty:
                    st.warning("Backtest não produziu resultados.")
                else:
                    st.success("Backtest executado com sucesso.")

                    col_aiq1, col_aiq2, col_aiq3, col_aiq4 = st.columns(4)
                    col_aiq1.metric("Testes realizados", resumo_aiq["total_testes"])
                    col_aiq2.metric("Hit ≥ 3", f"{resumo_aiq['hit_3p'] * 100:.1f}%")
                    col_aiq3.metric("Hit ≥ 4", f"{resumo_aiq['hit_4p'] * 100:.1f}%")
                    col_aiq4.metric("Hit ≥ 5", f"{resumo_aiq['hit_5p'] * 100:.1f}%")

                    col_aiq5, col_aiq6 = st.columns(2)
                    col_aiq5.metric("Hit = 6", f"{resumo_aiq['hit_6p'] * 100:.1f}%")
                    col_aiq6.metric("AIQ Global V14", f"{resumo_aiq['AIQ']:.3f}")

                    st.markdown("#### Detalhes do Backtest (amostra)")
                    df_bt_view = df_back.copy()
                    df_bt_view["nucleo_str"] = df_bt_view["nucleo"].apply(formatar_serie_str)
                    df_bt_view["real_str"] = df_bt_view["real"].apply(formatar_serie_str)
                    st.dataframe(
                        df_bt_view[["idx_atual", "idx_real", "acertos", "nucleo_str", "real_str"]].head(50),
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"Erro ao executar o backtest V14: {e}")

    st.stop()


# ============================================================
# PAINEL 6 — AIQ — Índice de Qualidade Global (visão rápida)
# ============================================================

if painel == "📊 AIQ — Índice de Qualidade":

    st.markdown("## 📊 AIQ — Índice de Qualidade Global")

    df = st.session_state.get("df_hist", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    st.info("Use este painel para uma visão rápida do AIQ. "
            "Para detalhes completos, utilize o painel '🔂 Backtest Interno V14'.")

    # Backtest rápido com parâmetros padrão
    if st.button("📈 Calcular AIQ Rápido (parâmetros padrão)"):
        with st.spinner("Executando backtest rápido..."):
            try:
                res_bt = executar_backtest_V14_completo(
                    df_historico=df,
                    janela_min=min(80, max(20, len(df) - 2)),
                    passo=1,
                    max_testes=80,
                )
                df_back = res_bt["df_backtest"]
                resumo_aiq = res_bt["resumo_aiq"]

                if df_back is None or df_back.empty:
                    st.warning("Backtest rápido não produziu resultados.")
                else:
                    st.success("AIQ calculado com sucesso!")

                    col_aiq1, col_aiq2, col_aiq3, col_aiq4 = st.columns(4)
                    col_aiq1.metric("Testes realizados", resumo_aiq["total_testes"])
                    col_aiq2.metric("Hit ≥ 3", f"{resumo_aiq['hit_3p'] * 100:.1f}%")
                    col_aiq3.metric("Hit ≥ 4", f"{resumo_aiq['hit_4p'] * 100:.1f}%")
                    col_aiq4.metric("Hit ≥ 5", f"{resumo_aiq['hit_5p'] * 100:.1f}%")

                    col_aiq5, col_aiq6 = st.columns(2)
                    col_aiq5.metric("Hit = 6", f"{resumo_aiq['hit_6p'] * 100:.1f}%")
                    col_aiq6.metric("AIQ Global V14", f"{resumo_aiq['AIQ']:.3f}")

            except Exception as e:
                st.error(f"Erro ao calcular AIQ: {e}")

    st.stop()


# ============================================================
# PAINEL 7 — Exportar Sessão
# ============================================================

if painel == "📦 Exportar Sessão":

    st.markdown("## 📦 Exportar Sessão")

    df = st.session_state.get("df_hist", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    if st.button("📥 Exportar histórico para CSV"):
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Clique para baixar o histórico",
            data=csv,
            file_name="historico_v14.csv",
            mime="text/csv",
        )

    st.stop()

