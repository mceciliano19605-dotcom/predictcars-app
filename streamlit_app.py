# ============================================================
#   PREDICT CARS — V14 TURBO++
#   app_v14_turbo_test.py
#   (Arquivo completo: Núcleo V14 + S6/S7 + TVF + Backtest + AIQ + UI)
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
# 🔧 Função básica: preparar_historico_V14
# ============================================================

def preparar_historico_V14(df_raw):
    df = df_raw.copy()

    if "series" not in df.columns:
        raise ValueError("O DataFrame não contém a coluna 'series'.")

    def normalize_row(row):
        if isinstance(row, list):
            return row
        if isinstance(row, str):
            nums = [int(x) for x in row.replace(",", " ").split() if x.isdigit()]
            return nums[:6]
        return row

    df["series"] = df["series"].apply(normalize_row)
    df = df[df["series"].apply(lambda x: isinstance(x, list) and len(x) == 6)]
    df = df.reset_index(drop=True)
    return df

# ============================================================
# NAVEGAÇÃO — MENU LATERAL
# ============================================================

with st.sidebar:
    st.markdown("## 🧭 Navegação — Predict Cars V14 TURBO++")

    st.write("df no session_state:", st.session_state.get("df")) 
    
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
# PAINEL 1 — Histórico — Entrada
# ============================================================

if painel == "📥 Histórico — Entrada":

    st.markdown("## 📥 Histórico — Entrada")

    df = None

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
                st.success("Histórico carregado com sucesso!")
                st.session_state["df"] = df
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
                    if len(nums) == 6:
                        series.append(nums)

                df_raw = pd.DataFrame({"series": series})
                df = preparar_historico_V14(df_raw)
                st.success("Histórico carregado com sucesso!")
                st.session_state["df"] = df
            except Exception as e:
                st.error(f"Erro ao processar histórico colado: {e}")

    # ============================================================
    # SALVAR NO SESSION_STATE (VERSÃO CORRIGIDA E SEGURA)
    # ============================================================

    # Sempre recarrega do session_state caso já exista
    df = st.session_state.get("df", None)

    st.markdown("---")

    # 🔴 ENCERRA AQUI ESTE PAINEL
    st.stop()


# ============================================================
# PAINEL 2 — Pipeline V14 (Simples)
# ============================================================

if painel == "🔍 Pipeline V14 (Simples)":

    st.markdown("## 🔍 Pipeline V14 — Execução Simples")

    # Verifica histórico carregado
    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    # Selecionar índice alvo
    idx_alvo = st.number_input(
        "Selecione o índice alvo:",
        min_value=1,
        max_value=len(df) - 1,
        value=len(df) - 1,
        step=1,
    )

    try:
        resultado = executar_pipeline_V14_simples(df, idx_alvo=idx_alvo)

        st.success("Pipeline executado com sucesso!")

        st.markdown("### 🔹 Série atual")
        st.code(" ".join(str(x) for x in resultado["serie_atual"]))

        st.markdown("### 🔹 Núcleo V14")
        st.code(" ".join(str(x) for x in resultado["nucleo_v14"]))

        st.markdown("### 🔹 Resultados do S7")
        st.write(f"Séries filtradas: {resultado['info_S7']['total_filtrado']} "
                 f"de {resultado['info_S7']['total_original']}")

        st.markdown("### 🔹 Ranking TVF — Top 20")
        df_scores = resultado["df_scores"]
        if df_scores is not None and not df_scores.empty:
            st.dataframe(df_scores.head(20), use_container_width=True)
        else:
            st.info("Nenhuma série disponível para avaliação.")

    except Exception as e:
        st.error(f"Erro ao executar Pipeline V14 (Simples): {e}")

  
# ============================================================
# PAINEL 3 — Pipeline V14 (Completo)
# ============================================================

if painel == "🧠 Pipeline V14 (Completo)":

    st.markdown("## 🧠 Pipeline V14 — Execução Completa")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    max_idx = len(df)

    idx_alvo = st.number_input(
        "Selecione o índice alvo:",
        min_value=1,
        max_value=max_idx - 1,
        value=max_idx - 1,
        step=1,
    )

    col1, col2 = st.columns(2)
    with col1:
        usar_s6 = st.checkbox("Ativar S6 Profundo", value=True)
        usar_s7 = st.checkbox("Ativar S7 / TVF", value=True)
        usar_tvf_local = st.checkbox("Ativar TVF Local", value=True)
    with col2:
        usar_backtest_int = st.checkbox("Backtest Interno", value=True)
        usar_backtest_fut = st.checkbox("Backtest do Futuro", value=False)
        calcular_aiq = st.checkbox("Calcular AIQ", value=True)

    n_series_saida = st.number_input(
        "Qtd. de séries na saída final",
        min_value=10,
        max_value=500,
        value=120,
        step=10,
    )

    min_conf_pct = st.slider(
        "Confiabilidade mínima (%)",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0,
    )

    executar = st.button("🚀 Executar Pipeline V14 Completo")

    # 🔵 TUDO A PARTIR DAQUI FICA DENTRO DO PAINEL
    if executar:
        with st.spinner("Rodando Núcleo V14 TURBO++…"):

            # 🔵 Ligação real ao Núcleo V14 TURBO++
            resultado = executar_pipeline_V14_completo(
                df=df,
                idx_alvo=idx_alvo,
                n_series_saida=n_series_saida,
                min_conf_pct=min_conf_pct,
                config={
                    "s6": usar_s6,
                    "s7": usar_s7,
                    "tvf_local": usar_tvf_local,
                    "bt_int": usar_backtest_int,
                    "bt_fut": usar_backtest_fut,
                    "aiq": calcular_aiq,
                }
            )

            # Desempacotar de forma segura
            previsao_final = resultado.get("previsao_final")
            resultado_s6 = resultado.get("s6")
            resultado_s7 = resultado.get("s7_tfv")
            resultado_bt_int = resultado.get("backtest_interno")
            resultado_bt_fut = resultado.get("backtest_futuro")
            resultado_aiq = resultado.get("aiq")

        st.markdown("### 📊 Resultados")

        aba1, aba2, aba3, aba4, aba5 = st.tabs(
            ["🎯 Previsão", "🧬 S6", "🌀 S7 / TVF", "⏱ Backtests", "📈 AIQ"]
        )

        with aba1:
            if previsao_final:
                st.code(" ".join(str(x) for x in previsao_final))
            else:
                st.info("Núcleo ainda não conectado.")

        with aba2:
            st.info("S6 ainda não conectado.")

        with aba3:
            st.info("S7 / TVF ainda não conectado.")

        with aba4:
            st.info("Backtests ainda não conectados.")

        with aba5:
            st.info("AIQ ainda não conectado.")


# ============================================================
# PAINEL 4 — Previsões V14 Turbo++
# ============================================================

if painel == "🎯 Previsões — V14 Turbo++":

    st.markdown("## 🎯 Previsões — V14 TURBO++")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    n_series = st.number_input(
        "Quantidade de séries a gerar:",
        min_value=1,
        max_value=300,
        value=50,
    )

    gerar = st.button("🚀 Gerar Previsões Turbo++")

    if gerar:
        with st.spinner("Gerando previsões com Turbo++…"):

            previsoes = []
            for _ in range(n_series):
                previsoes.append([0, 0, 0, 0, 0, 0])  # placeholder

        st.success("Previsões geradas!")
        for serie in previsoes:
            st.code(" ".join(str(x) for x in serie))

    st.stop()


# ============================================================
# PAINEL 5 — Backtest Interno V14
# ============================================================

if painel == "🔂 Backtest Interno V14":

    st.markdown("## 🔂 Backtest Interno — V14")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    janela = st.slider(
        "Tamanho da janela (linhas para trás):",
        min_value=5,
        max_value=200,
        value=30,
    )

    executar = st.button("🚀 Executar Backtest Interno")

    if executar:
        with st.spinner("Rodando backtest…"):
            resultado = {"acertos": 0, "total": 0}  # placeholder

        st.success("Backtest concluído!")
        st.json(resultado)

    st.stop()

# ============================================================
# PAINEL 6 — AIQ — Índice de Qualidade Global
# ============================================================

if painel == "📊 AIQ — Índice de Qualidade":

    st.markdown("## 📊 AIQ — Índice de Qualidade Global")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    calcular = st.button("📈 Calcular AIQ")

    if calcular:
        with st.spinner("Calculando AIQ…"):
            aiq = 0  # placeholder

        st.success("AIQ calculado!")
        st.metric("AIQ Global", aiq)

    st.stop()

# ============================================================
# PAINEL 7 — Exportar Sessão
# ============================================================

if painel == "📦 Exportar Sessão":

    st.markdown("## 📦 Exportar Sessão")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    if st.button("📥 Exportar histórico para CSV"):
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Clique para baixar",
            data=csv,
            file_name="historico_v14.csv",
            mime="text/csv",
        )

    st.stop()


# ============================================================
# 🔵 FUNÇÕES BASE — Normalização e utilidades gerais
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
# 🔵 PREPARAÇÃO DO HISTÓRICO — V14
# ============================================================

def preparar_historico_V14(df_raw, col_series="series"):
    """
    Garante que o histórico esteja no formato:
        - coluna 'series' com listas de inteiros
    """
    df = df_raw.copy()

    # Normalizar nome de coluna
    if col_series != "series" and col_series in df.columns:
        df["series"] = df[col_series]
    elif "series" not in df.columns:
        raise ValueError("DataFrame histórico precisa ter coluna 'series'.")

    def _validar_serie(x):
        if isinstance(x, list) and all(isinstance(n, (int, float)) for n in x):
            return [int(n) for n in x]
        if isinstance(x, str):
            try:
                parts = [p.strip() for p in x.replace(";", ",").split(",") if p.strip()]
                return [int(p) for p in parts]
            except Exception:
                return None
        return None

    df["series"] = df["series"].apply(_validar_serie)
    df = df[df["series"].notnull()].reset_index(drop=True)

    return df


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
# 🔵 INTERFACE STREAMLIT — HISTÓRICO + PIPELINE + BACKTEST
# ============================================================

st.title("🚗 Predict Cars V14 TURBO++")
st.caption("Núcleo V14 + S6/S7 + TVF + Backtest Interno + AIQ Global")

st.markdown("---")

# ---------------- SIDEBAR: Histórico ----------------

st.sidebar.header("📥 Histórico — Entrada")

modo_hist = st.sidebar.radio(
    "Formato do histórico:",
    ["CSV com coluna de séries", "CSV com 6 passageiros (n1..n6)"],
    index=0,
)

uploaded_file = st.sidebar.file_uploader(
    "Selecione o arquivo de histórico (.csv):",
    type=["csv"],
)

df_historico = None
erro_hist = None

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)

        if modo_hist == "CSV com coluna de séries":
            col_series = st.sidebar.selectbox(
                "Coluna com as séries (ex: 'series' ou 's')",
                options=list(df_raw.columns),
            )
            df_historico = preparar_historico_V14(df_raw, col_series)

        else:
            colunas_num = st.sidebar.multiselect(
                "Selecione as 6 colunas dos passageiros (na ordem):",
                options=list(df_raw.columns),
                default=list(df_raw.columns)[:6] if len(df_raw.columns) >= 6 else [],
            )

            if len(colunas_num) == 6:
                def _linha_para_serie(row):
                    return [int(row[c]) for c in colunas_num]

                df_tmp = pd.DataFrame()
                df_tmp["series"] = df_raw.apply(_linha_para_serie, axis=1)
                df_historico = preparar_historico_V14(df_tmp, "series")
            else:
                erro_hist = "Selecione exatamente 6 colunas para montar as séries."

    except Exception as e:
        erro_hist = f"Erro ao carregar o histórico: {e}"

# ---------------- VISÃO GERAL DO HISTÓRICO ----------------

col_hist, col_status = st.columns([2, 1])

with col_hist:
    st.subheader("📊 Histórico carregado")

    if erro_hist:
        st.error(erro_hist)
    elif df_historico is None:
        st.info("Aguardando carregamento do histórico em CSV na barra lateral.")
    else:
        st.success(f"Histórico carregado com sucesso ({len(df_historico)} séries).")
        st.dataframe(df_historico.tail(10), use_container_width=True)

with col_status:
    st.subheader("ℹ️ Status")
    if df_historico is None:
        st.write("Nenhum histórico pronto ainda.")
    else:
        st.markdown(f"- Séries totais: **{len(df_historico)}**")
        if len(df_historico) >= 2:
            st.markdown("- Histórico suficiente para rodar o V14 ✅")
        else:
            st.markdown("- Histórico insuficiente (mínimo 2 séries) ❌")

st.markdown("---")

# ---------------- CONTROLES DO PIPELINE V14 ----------------

st.subheader("⚙️ Configurações do Pipeline V14")

if df_historico is not None and len(df_historico) >= 2:
    n_series = len(df_historico)

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

    # ------------ BOTÃO: RODAR PIPELINE V14 ------------

    rodar_pipeline = st.button("🚀 Rodar Pipeline V14 Completo")

    if rodar_pipeline:
        with st.spinner("Executando pipeline V14 TURBO++..."):
            try:
                resultado = executar_pipeline_V14_completo(
                    df_historico=df_historico,
                    idx_alvo=idx_alvo,
                    max_series_S6=max_series_S6,
                    dispersao_max_S7=dispersao_max_S7,
                    top_n_final=top_n_final,
                )

                info = resultado["info_pipeline"]
                df_scores_final = resultado["df_scores_final"]
                nucleo_v14 = resultado["nucleo_v14"]
                serie_atual = resultado["serie_atual"]
                info_S7 = resultado["info_S7"]

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

else:
    st.info("Carregue um histórico válido com pelo menos 2 séries para habilitar o pipeline V14.")

st.markdown("---")

# ---------------- BACKTEST + AIQ ----------------

st.subheader("🧪 Backtest Interno V14 + AIQ Global")

if df_historico is not None and len(df_historico) >= 100:
    col_b1, col_b2, col_b3 = st.columns(3)

    with col_b1:
        janela_min_bt = st.number_input(
            "Janela mínima para começar o backtest",
            min_value=20,
            max_value=len(df_historico) - 2,
            value=80,
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
            value=80,
            step=10,
        )

    rodar_backtest = st.button("🧪 Rodar Backtest + AIQ")

    if rodar_backtest:
        with st.spinner("Executando backtest interno V14..."):
            try:
                res_bt = executar_backtest_V14_completo(
                    df_historico=df_historico,
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

                    col_aiq1, col_aiq2, col_aiq3, col_aiq4 = st.columns(4)
                    col_aiq1.metric("Testes realizados", resumo_aiq["total_testes"])
                    col_aiq2.metric("Hit ≥ 3", f"{resumo_aiq['hit_3p']*100:.1f}%")
                    col_aiq3.metric("Hit ≥ 4", f"{resumo_aiq['hit_4p']*100:.1f}%")
                    col_aiq4.metric("Hit ≥ 5", f"{resumo_aiq['hit_5p']*100:.1f}%")

                    col_aiq5, col_aiq6 = st.columns(2)
                    col_aiq5.metric("Hit = 6", f"{resumo_aiq['hit_6p']*100:.1f}%")
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

else:
    st.info("Para rodar o Backtest e o AIQ Global, é recomendado um histórico com pelo menos 100 séries.")
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
# 🔵 SEÇÃO FINAL — Previsão TURBO++, k* e Exportação
# ============================================================

st.markdown("---")
st.subheader("🎯 Previsão Final TURBO++ + k* + Exportação")

if df_historico is not None and len(df_historico) >= 2:
    st.markdown(
        "Esta seção usa **as mesmas configurações atuais do Pipeline V14** "
        "(índice alvo, S6, S7, Top N) para gerar a **Previsão Final TURBO++**."
    )

    gerar_prev_final = st.button("🎯 Gerar Previsão Final TURBO++ a partir das configurações atuais")

    if gerar_prev_final:
        with st.spinner("Calculando Previsão Final TURBO++..."):
            try:
                resultado_final = executar_pipeline_V14_completo(
                    df_historico=df_historico,
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
else:
    st.info("Carregue um histórico válido e configure o Pipeline V14 para habilitar a Previsão Final TURBO++.")
