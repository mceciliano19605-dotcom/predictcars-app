# ============================================================
#   PREDICT CARS — V14 TURBO++
#   app_v14_turbo_test.py
#   Núcleo V14 + S6/S7 + TVF + Backtest + AIQ + UI em painéis
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
# 🔧 HELPERS BÁSICOS
# ============================================================

def formatar_serie_str(serie):
    """Converte [8, 15, 23, 30, 39, 59] -> '8 15 23 30 39 59'."""
    return " ".join(str(x) for x in serie)


def parse_serie_str(s):
    """
    Converte uma string em série (lista de 6 inteiros), aceitando:
      - '8 15 23 30 39 59'
      - '8,15,23,30,39,59'
      - '8; 15; 23; 30; 39; 59'
      - 'C2943; 8; 15; 23; 30; 39; 59; k'
    Retorna:
      - [n1, n2, n3, n4, n5, n6] ou None se inválido.
    """
    if not isinstance(s, str):
        return None

    s = s.strip()
    if not s:
        return None

    # Normaliza separadores
    for ch in [",", ";", "|", "\t"]:
        s = s.replace(ch, " ")

    tokens = s.split()
    nums = []
    for t in tokens:
        if t.isdigit():
            nums.append(int(t))

    if len(nums) < 6:
        return None

    # Usa apenas os 6 primeiros (k ou extras ficam de fora)
    return nums[:6]


def parse_texto_historico(texto):
    """
    Converte um bloco de texto em lista de séries.
    Cada linha pode ter:
        '8 15 23 30 39 59'
        '8;15;23;30;39;59'
        'C2943; 8; 15; 23; 30; 39; 59; 1'
    """
    linhas = texto.strip().splitlines()
    series = []

    for ln in linhas:
        s = parse_serie_str(ln)
        if s is not None:
            series.append(s)

    return series


# ============================================================
# 🔵 PREPARAÇÃO DO HISTÓRICO — V14
# ============================================================

def preparar_historico_V14(
    df_raw: pd.DataFrame,
    modo: str = "coluna",
    col_series: str = "series",
    col_passageiros=None,
) -> pd.DataFrame:
    """
    Garante que o histórico fique no formato:
        df["series"] = [n1, n2, n3, n4, n5, n6] (lista de 6 ints)

    Parâmetros:
      - modo = "coluna":
            usa df[col_series] (string, lista, etc.)
      - modo = "colunas":
            usa lista col_passageiros = [c1,...,c6]
    """
    df = df_raw.copy()

    if modo == "colunas":
        if not col_passageiros or len(col_passageiros) != 6:
            raise ValueError("col_passageiros deve ter exatamente 6 colunas.")

        def _linha_para_serie(row):
            vals = []
            for c in col_passageiros:
                vals.append(int(row[c]))
            return vals

        df["series"] = df.apply(_linha_para_serie, axis=1)
        # Depois normaliza como se fosse coluna única
        col_series = "series"
        modo = "coluna"

    if modo == "coluna":
        if col_series not in df.columns:
            raise ValueError(f"DataFrame histórico precisa ter coluna '{col_series}'.")

        def _normalizar_celula(x):
            # Já é lista ou array?
            if isinstance(x, (list, tuple, np.ndarray)):
                vals = []
                for n in x:
                    if isinstance(n, (int, float)):
                        vals.append(int(n))
                return vals if len(vals) == 6 else None

            # É número isolado (provavelmente não serve)
            if isinstance(x, (int, float)):
                # Não tem como montar 6 passageiros a partir de 1 só
                return None

            # String: tenta interpretar
            if isinstance(x, str):
                return parse_serie_str(x)

            return None

        df["series"] = df[col_series].apply(_normalizar_celula)
        df = df[df["series"].apply(lambda v: isinstance(v, list) and len(v) == 6)]
        df = df.reset_index(drop=True)

        # Mantém apenas a coluna 'series' (para o núcleo)
        df = df[["series"]]

        return df

    raise ValueError(f"Modo inválido em preparar_historico_V14: {modo}")


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
    novo = list(nucleo)
    iguais = sum(1 for a, b in zip(nucleo, serie_atual) if a == b)

    if iguais >= 3:
        novo = [x + 1 if i % 2 == 0 else x - 1 for i, x in enumerate(novo)]
        novo = sorted(max(1, x) for x in novo)

    return novo


def ajuste_dinamico(nucleo):
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
# 🔵 S6 PROFUNDO — Geração de Vizinhança Estruturada
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
# 🔵 EXECUÇÃO BÁSICA DO V14 — PIPELINE SIMPLES
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
        raise ValueError(f"idx_alvo inválido: {idx_alvo}. Deve estar entre 1 e {n-1}.")

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
# 🔵 EXECUÇÃO COMPLETA DO V14 — COM CONTROLES
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
        raise ValueError(f"idx_alvo inválido: {idx_alvo}. Deve estar entre 1 e {n-1}.")

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
# 🔵 k* — Estado Qualitativo do Ambiente
# ============================================================

def classificar_k_estado(serie_atual, nucleo_v14):
    """
    Estima um k* qualitativo (estado do ambiente) com base em:
        • interseção entre série atual e núcleo V14
        • diferença de dispersão
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

st.title("🚗 Predict Cars V14 TURBO++")
st.caption("Núcleo V14 + S6/S7 + TVF + Backtest Interno + AIQ Global")
st.markdown("---")

# ============================================================
# PAINEL 1 — Histórico — Entrada
# ============================================================

if painel == "📥 Histórico — Entrada":

    st.markdown("## 📥 Histórico — Entrada")

    df = st.session_state.get("df", None)

    opc = st.radio(
        "Como deseja carregar o histórico?",
        ["Enviar arquivo CSV", "Copiar e colar o histórico"]
    )

    df_hist = None
    erro_hist = None

    # ---------- OPÇÃO 1 — UPLOAD DE ARQUIVO ----------
    if opc == "Enviar arquivo CSV":
        file = st.file_uploader("Selecione o arquivo CSV:", type=["csv"])

        if file is not None:
            try:
                df_raw = pd.read_csv(file)

                st.markdown("### ⚙️ Formato do CSV")
                modo_csv = st.radio(
                    "Como o histórico está estruturado?",
                    ["CSV com coluna de séries (ex: 'series')", "CSV com 6 colunas de passageiros"],
                )

                if modo_csv.startswith("CSV com coluna"):
                    col_series = st.selectbox(
                        "Coluna com as séries (string/lista):",
                        options=list(df_raw.columns),
                    )
                    df_hist = preparar_historico_V14(
                        df_raw,
                        modo="coluna",
                        col_series=col_series,
                    )

                else:
                    colunas_num = st.multiselect(
                        "Selecione as 6 colunas dos passageiros (na ordem):",
                        options=list(df_raw.columns),
                        default=list(df_raw.columns)[:6] if len(df_raw.columns) >= 6 else [],
                    )

                    if len(colunas_num) == 6:
                        df_hist = preparar_historico_V14(
                            df_raw,
                            modo="colunas",
                            col_passageiros=colunas_num,
                        )
                    else:
                        erro_hist = "Selecione exatamente 6 colunas para montar as séries."

            except Exception as e:
                erro_hist = f"Erro ao carregar o histórico em CSV: {e}"

    # ---------- OPÇÃO 2 — COLAR HISTÓRICO ----------
    else:
        texto = st.text_area(
            "Cole aqui o histórico (uma série por linha):",
            height=220,
            placeholder=(
                "Exemplos de formatos aceitos:\n"
                "8 15 23 30 39 59\n"
                "8,15,23,30,39,59\n"
                "8; 15; 23; 30; 39; 59\n"
                "C2943; 8; 15; 23; 30; 39; 59; 1"
            ),
        )

        if texto.strip():
            try:
                series = parse_texto_historico(texto)
                if not series:
                    erro_hist = "Nenhuma série válida encontrada no texto colado."
                else:
                    df_tmp = pd.DataFrame({"series": series})
                    df_hist = preparar_historico_V14(df_tmp, modo="coluna", col_series="series")
            except Exception as e:
                erro_hist = f"Erro ao processar histórico colado: {e}"

    # ---------- VISÃO / DEBUG ----------
    st.markdown("---")
    st.markdown("### 🔎 Resultado do processamento")

    if erro_hist:
        st.error(erro_hist)
    elif df_hist is None:
        st.info("Carregue um CSV ou cole o histórico para ver o resultado.")
    else:
        st.success(f"Histórico carregado com sucesso ({len(df_hist)} séries).")
        st.write("DEBUG — tipo:", type(df_hist))
        st.write("DEBUG — tamanho:", len(df_hist))
        st.dataframe(df_hist.tail(10), use_container_width=True)

        # Salva no session_state para os demais painéis
        st.session_state["df"] = df_hist

    st.stop()


# ============================================================
# PAINEL 2 — Pipeline V14 (Simples)
# ============================================================

if painel == "🔍 Pipeline V14 (Simples)":

    st.markdown("## 🔍 Pipeline V14 — Execução Simples")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    n = len(df)

    idx_alvo = st.number_input(
        "Selecione o índice alvo:",
        min_value=1,
        max_value=n - 1,
        value=n - 1,
        step=1,
    )

    try:
        resultado = executar_pipeline_V14_simples(df, idx_alvo=idx_alvo)

        st.success("Pipeline executado com sucesso!")

        st.markdown("### 🔹 Série atual")
        st.code(formatar_serie_str(resultado["serie_atual"]), language="text")

        st.markdown("### 🔹 Núcleo V14")
        st.code(formatar_serie_str(resultado["nucleo_v14"]), language="text")

        st.markdown("### 🔹 Resultados do S7")
        info_S7 = resultado["info_S7"]
        st.write(
            f"Séries filtradas: {info_S7['total_filtrado']} "
            f"de {info_S7['total_original']}"
        )

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

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    n_series = len(df)

    idx_alvo = st.slider(
        "Índice da série alvo (0 = primeira linha, n-1 = última):",
        min_value=1,
        max_value=n_series - 1,
        value=n_series - 1,
        step=1,
        help="A série alvo é a 'C atual'. O histórico até a série anterior é usado como base de treino.",
    )

    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        max_series_S6 = st.number_input(
            "Máx. séries em S6 (vizinhança)",
            min_value=64,
            max_value=4096,
            value=512,
            step=64,
        )

    with col_p2:
        dispersao_max_S7 = st.number_input(
            "Dispersão máxima em S7 (max - min)",
            min_value=20,
            max_value=59,
            value=45,
            step=1,
        )

    with col_p3:
        top_n_final = st.number_input(
            "Top N final pelo TVF",
            min_value=16,
            max_value=1024,
            value=128,
            step=16,
        )

    st.markdown("---")

    rodar_pipeline = st.button("🚀 Rodar Pipeline V14 Completo")

    if rodar_pipeline:
        with st.spinner("Executando pipeline V14 TURBO++..."):
            try:
                resultado = executar_pipeline_V14_completo(
                    df_historico=df,
                    idx_alvo=idx_alvo,
                    max_series_S6=max_series_S6,
                    dispersao_max_S7=dispersao_max_S7,
                    top_n_final=top_n_final,
                )

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
# PAINEL 4 — Previsões V14 Turbo++
# ============================================================

if painel == "🎯 Previsões — V14 Turbo++":

    st.markdown("## 🎯 Previsão Final TURBO++ + k* + Séries Puras")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    n_series = len(df)

    idx_alvo = st.slider(
        "Índice da série alvo (0 = primeira linha, n-1 = última):",
        min_value=1,
        max_value=n_series - 1,
        value=n_series - 1,
        step=1,
    )

    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        max_series_S6 = st.number_input(
            "Máx. séries em S6 (vizinhança)",
            min_value=64,
            max_value=4096,
            value=512,
            step=64,
        )

    with col_p2:
        dispersao_max_S7 = st.number_input(
            "Dispersão máxima em S7 (max - min)",
            min_value=20,
            max_value=59,
            value=45,
            step=1,
        )

    with col_p3:
        top_n_final = st.number_input(
            "Top N final pelo TVF",
            min_value=16,
            max_value=1024,
            value=128,
            step=16,
        )

    gerar_prev_final = st.button("🎯 Gerar Previsão Final TURBO++")

    if gerar_prev_final:
        with st.spinner("Calculando Previsão Final TURBO++..."):
            try:
                resultado_final = executar_pipeline_V14_completo(
                    df_historico=df,
                    idx_alvo=idx_alvo,
                    max_series_S6=max_series_S6,
                    dispersao_max_S7=dispersao_max_S7,
                    top_n_final=top_n_final,
                )

                info_pf = resultado_final["info_pipeline"]
                df_scores_pf = resultado_final["df_scores_final"]
                nucleo_v14_pf = resultado_final["nucleo_v14"]
                serie_atual_pf = resultado_final["serie_atual"]

                if df_scores_pf is None or df_scores_pf.empty:
                    st.warning("Nenhuma série passou pelos filtros S6/S7 — não foi possível gerar uma Previsão Final TURBO++.")
                else:
                    melhor = df_scores_pf.iloc[0]
                    previsao_final = melhor["series"]

                    st.success("Previsão Final TURBO++ gerada com sucesso.")

                    col_pf1, col_pf2 = st.columns(2)

                    with col_pf1:
                        st.markdown("##### 🎯 Série Alvo (C atual)")
                        st.write(f"Índice alvo: **{info_pf['idx_alvo']}**")
                        st.code(formatar_serie_str(serie_atual_pf), language="text")

                    with col_pf2:
                        st.markdown("##### 🧠 Núcleo V14 (núcleo previsivo)")
                        st.code(formatar_serie_str(nucleo_v14_pf), language="text")

                    st.markdown("### 🎯 Previsão Final TURBO++ (6 passageiros)")
                    st.code(formatar_serie_str(previsao_final), language="text")

                    # ---------------- k* ----------------
                    info_k = classificar_k_estado(serie_atual_pf, nucleo_v14_pf)

                    st.markdown("### 🌡️ k* — Estado Qualitativo do Ambiente")
                    st.info(info_k["mensagem"])

                    with st.expander("Detalhes técnicos do k* (debug)"):
                        st.write(f"Acertos entre série atual e núcleo V14: **{info_k['acertos']}**")
                        st.write(f"Dispersão série atual: **{info_k['disp_atual']}**")
                        st.write(f"Dispersão núcleo V14: **{info_k['disp_nucleo']}**")
                        st.write(f"Δ Dispersão: **{info_k['diff_disp']}**")

                    # ---------------- Séries puras ----------------
                    st.markdown("### 📋 Séries Puras (Top N pelo TVF) — prontas para copiar/colar")

                    linhas_puras = []
                    for _, row in df_scores_pf.iterrows():
                        s = row["series"]
                        linhas_puras.append(formatar_serie_str(s))

                    bloco_series_puras = "\n".join(linhas_puras)

                    st.text_area(
                        "Séries puras (uma por linha):",
                        value=bloco_series_puras,
                        height=200,
                    )

                    # ---------------- Exportação CSV ----------------
                    st.markdown("### 💾 Exportar séries ranqueadas")

                    df_export = df_scores_pf.copy()
                    df_export["series_str"] = df_export["series"].apply(formatar_serie_str)
                    cols = ["series_str", "TVF", "TCI", "TPD", "TCS", "TVE", "series"]
                    df_export = df_export[cols]

                    csv_bytes = df_export.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="💾 Baixar séries ranqueadas em CSV",
                        data=csv_bytes,
                        file_name="predict_cars_V14_TURBO_pp_series.csv",
                        mime="text/csv",
                    )

            except Exception as e:
                st.error(f"Erro ao gerar a Previsão Final TURBO++: {e}")

    st.stop()


# ============================================================
# PAINEL 5 — Backtest Interno V14
# ============================================================

if painel == "🔂 Backtest Interno V14":

    st.markdown("## 🔂 Backtest Interno — V14")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    if len(df) < 30:
        st.info("Recomendado pelo menos ~100 séries para um backtest mais robusto, mas você pode testar com menos.")
    
    col_b1, col_b2, col_b3 = st.columns(3)

    with col_b1:
        janela_min_bt = st.number_input(
            "Janela mínima para começar o backtest",
            min_value=10,
            max_value=max(10, len(df) - 2),
            value=min(80, max(10, len(df) - 2)),
            step=5,
        )

    with col_b2:
        passo_bt = st.number_input(
            "Passo ao percorrer o histórico",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
        )

    with col_b3:
        max_testes_bt = st.number_input(
            "Máximo de pontos de teste",
            min_value=10,
            max_value=200,
            value=min(80,  max(10, len(df) - 2)),
            step=10,
        )

    rodar_backtest = st.button("🧪 Rodar Backtest Interno + AIQ")

    if rodar_backtest:
        with st.spinner("Executando backtest interno V14..."):
            try:
                res_bt = executar_backtest_V14_completo(
                    df_historico=df,
                    janela_min=int(janela_min_bt),
                    passo=int(passo_bt),
                    max_testes=int(max_testes_bt),
                )
                df_back = res_bt["df_backtest"]
                resumo_aiq = res_bt["resumo_aiq"]

                if df_back is None or df_back.empty:
                    st.warning("Backtest não produziu resultados.")
                else:
                    st.success("Backtest executado com sucesso.")

                    st.markdown("### 📊 Resumo do AIQ Global")
                    col_aiq1, col_aiq2, col_aiq3, col_aiq4 = st.columns(4)
                    col_aiq1.metric("Testes realizados", resumo_aiq["total_testes"])
                    col_aiq2.metric("Hit ≥ 3", f"{resumo_aiq['hit_3p']*100:.1f}%")
                    col_aiq3.metric("Hit ≥ 4", f"{resumo_aiq['hit_4p']*100:.1f}%")
                    col_aiq4.metric("Hit ≥ 5", f"{resumo_aiq['hit_5p']*100:.1f}%")

                    col_aiq5, col_aiq6 = st.columns(2)
                    col_aiq5.metric("Hit = 6", f"{resumo_aiq['hit_6p']*100:.1f}%")
                    col_aiq6.metric("AIQ Global V14", f"{resumo_aiq['AIQ']:.3f}")

                    st.markdown("### 📋 Detalhes do Backtest (amostra)")
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
# PAINEL 6 — AIQ — Índice de Qualidade
# ============================================================

if painel == "📊 AIQ — Índice de Qualidade":

    st.markdown("## 📊 AIQ — Índice de Qualidade Global")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    if len(df) < 50:
        st.info("Para um AIQ mais estável, recomendo pelo menos ~100 séries, mas você pode testar com menos.")

    janela_min_bt = st.number_input(
        "Janela mínima para começar o backtest (apenas para cálculo do AIQ)",
        min_value=10,
        max_value=max(10, len(df) - 2),
        value=min(80, max(10, len(df) - 2)),
        step=5,
    )

    calcular_aiq = st.button("📈 Calcular AIQ Global V14")

    if calcular_aiq:
        with st.spinner("Executando backtest interno para cálculo do AIQ..."):
            try:
                res_bt = executar_backtest_V14_completo(
                    df_historico=df,
                    janela_min=int(janela_min_bt),
                    passo=1,
                    max_testes=80,
                )
                resumo_aiq = res_bt["resumo_aiq"]

                if resumo_aiq["total_testes"] == 0:
                    st.warning("Backtest não produziu resultados suficientes para calcular o AIQ.")
                else:
                    st.success("AIQ Global V14 calculado com sucesso.")

                    col_aiq1, col_aiq2, col_aiq3, col_aiq4 = st.columns(4)
                    col_aiq1.metric("Testes realizados", resumo_aiq["total_testes"])
                    col_aiq2.metric("Hit ≥ 3", f"{resumo_aiq['hit_3p']*100:.1f}%")
                    col_aiq3.metric("Hit ≥ 4", f"{resumo_aiq['hit_4p']*100:.1f}%")
                    col_aiq4.metric("Hit ≥ 5", f"{resumo_aiq['hit_5p']*100:.1f}%")

                    col_aiq5, col_aiq6 = st.columns(2)
                    col_aiq5.metric("Hit = 6", f"{resumo_aiq['hit_6p']*100:.1f}%")
                    col_aiq6.metric("AIQ Global V14", f"{resumo_aiq['AIQ']:.3f}")

            except Exception as e:
                st.error(f"Erro ao calcular o AIQ Global V14: {e}")

    st.stop()


# ============================================================
# PAINEL 7 — Exportar Sessão
# ============================================================

if painel == "📦 Exportar Sessão":

    st.markdown("## 📦 Exportar Sessão")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    st.markdown("### 💾 Exportar histórico atual para CSV")

    if st.button("📥 Gerar arquivo CSV do histórico"):
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Clique para baixar o histórico",
            data=csv,
            file_name="historico_v14.csv",
            mime="text/csv",
        )

    st.stop()

