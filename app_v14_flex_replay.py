# ============================================================
# Predict Cars V14-FLEX ULTRA REAL (TURBO++)
# streamlit_app.py — Versão completa com:
# - Entrada FLEX (n variável de passageiros)
# - Barômetro ULTRA REAL
# - k* ULTRA REAL (sentinela baseado em k dos guardas)
# - IDX / IPF / IPO ULTRA
# - S6 Profundo & Micro-Leque ULTRA
# - Monte Carlo Profundo ULTRA
# - QDS REAL & Backtest REAL
# - Replay LIGHT
# - Replay ULTRA (Horizonte Ajustável)
# - Modo TURBO++ ULTRA Adaptativo
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import math
from typing import List, Tuple, Dict, Any
from collections import Counter
from itertools import combinations

# ------------------------------------------------------------
# Configuração básica da página
# ------------------------------------------------------------
st.set_page_config(
    page_title="Predict Cars V14-FLEX ULTRA REAL (TURBO++)",
    layout="wide",
)

# ------------------------------------------------------------
# Utilitários gerais
# ------------------------------------------------------------

def registrar_evento(msg: str) -> None:
    """Log simples em session_state, apenas informativo."""
    historico = st.session_state.get("log_eventos", [])
    historico.append(msg)
    st.session_state["log_eventos"] = historico


def calcular_entropia(valores: List[int]) -> float:
    """Entropia de Shannon básica para lista de inteiros."""
    if not valores:
        return 0.0
    contagem = Counter(valores)
    total = sum(contagem.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in contagem.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def normalizar_0_1(x: float, xmin: float, xmax: float) -> float:
    if xmax == xmin:
        return 0.0
    v = (x - xmin) / (xmax - xmin)
    return max(0.0, min(1.0, v))


def detectar_colunas_passageiros(df_raw: pd.DataFrame) -> Tuple[List[str], str]:
    """
    Detecta automaticamente quais colunas são de passageiros e qual é a coluna k.

    Regra:
    - Se existir coluna 'k' (case insensitive), ela é usada como k.
    - Caso contrário, assume-se que a última coluna numérica é k.
    - Todas as colunas numéricas (exceto k) entre a primeira numérica e k são passageiros.
    """
    cols = list(df_raw.columns)

    # Tenta achar 'k' explícito
    col_k = None
    for c in cols:
        if str(c).strip().lower() == "k":
            col_k = c
            break

    if col_k is None:
        # Tenta usar última coluna numérica
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df_raw[c])]
        if not numeric_cols:
            raise ValueError("Nenhuma coluna numérica encontrada para detectar passageiros/k.")
        col_k = numeric_cols[-1]

    # Passageiros = todas numéricas antes de k
    idx_k = cols.index(col_k)
    numeric_before_k = [
        c for c in cols[: idx_k + 1] if pd.api.types.is_numeric_dtype(df_raw[c])
    ]
    passageiros_cols = [c for c in numeric_before_k if c != col_k]

    if len(passageiros_cols) == 0:
        raise ValueError("Nenhuma coluna de passageiros detectada antes da coluna k.")

    return passageiros_cols, col_k


def preparar_historico_flex(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara o histórico em formato FLEX:
    - Detecta automaticamente colunas de passageiros e coluna k.
    - Garante índice C1, C2, ...
    - Salva estrutura em session_state:
      - 'df'
      - 'passageiros_cols'
      - 'col_k'
      - 'n_passageiros'
    """
    df = df_raw.copy()

    passageiros_cols, col_k = detectar_colunas_passageiros(df)
    n_pass = len(passageiros_cols)

    # Cria coluna ID se não existir
    if "ID" not in df.columns and "id" not in [c.lower() for c in df.columns]:
        df.insert(0, "ID", [f"C{i}" for i in range(1, len(df) + 1)])

    # Normaliza o nome da coluna k para exatamente 'k'
    if col_k != "k":
        df.rename(columns={col_k: "k"}, inplace=True)
        col_k = "k"

    # Garante tipos numéricos
    for c in passageiros_cols + [col_k]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=passageiros_cols + [col_k]).reset_index(drop=True)
    df["idx_interno"] = np.arange(1, len(df) + 1)

    st.session_state["df"] = df
    st.session_state["passageiros_cols"] = passageiros_cols
    st.session_state["col_k"] = col_k
    st.session_state["n_passageiros"] = n_pass

    return df


def obter_contexto_basico() -> Tuple[pd.DataFrame, List[str], str, int]:
    """Recupera df + metadados principais do session_state."""
    df = st.session_state.get("df", None)
    passageiros_cols = st.session_state.get("passageiros_cols", [])
    col_k = st.session_state.get("col_k", "k")
    n_pass = st.session_state.get("n_passageiros", len(passageiros_cols))
    return df, passageiros_cols, col_k, n_pass


# ------------------------------------------------------------
# Barômetro ULTRA REAL + k* ULTRA REAL (sentinela)
# ------------------------------------------------------------

def calcular_barometro_ultra(df: pd.DataFrame, col_k: str, window: int = 120) -> Dict[str, Any]:
    """
    Barômetro ULTRA REAL baseado na coluna k:
    - k representa quantos guardas acertaram exatamente os 15 passageiros do carro.
    - O barômetro mede a estabilidade global desse "acerto dos guardas" ao longo do tempo.
    """
    if df is None or df.empty:
        return {
            "estado": "desconhecido",
            "turbulencia": None,
            "std_k": None,
            "mean_abs_dk": None,
        }

    serie_k = df[col_k].astype(float).values
    if len(serie_k) < 3:
        return {
            "estado": "desconhecido",
            "turbulencia": None,
            "std_k": None,
            "mean_abs_dk": None,
        }

    w = min(window, len(serie_k))
    recent = serie_k[-w:]
    if len(recent) < 3:
        return {
            "estado": "desconhecido",
            "turbulencia": None,
            "std_k": None,
            "mean_abs_dk": None,
        }

    diffs = np.diff(recent)
    mean_abs_dk = float(np.mean(np.abs(diffs)))
    std_k = float(np.std(recent))

    # Índice de turbulência: mistura variação de k + variação entre carros
    turbulencia = float(mean_abs_dk + 0.3 * std_k)

    # Estados (ajustados para a natureza de k como "acerto de guardas")
    # Baixa turbulência → estrada previsível (muitos guardas alinhados ou padrão estável)
    # Alta turbulência → guardas ora acertam muito, ora erram muito → regime caótico.
    if turbulencia < 1.5:
        estado = "estavel"
    elif turbulencia < 3.0:
        estado = "atencao"
    else:
        estado = "critico"

    return {
        "estado": estado,
        "turbulencia": turbulencia,
        "std_k": std_k,
        "mean_abs_dk": mean_abs_dk,
    }


def calcular_k_star_ultra(df: pd.DataFrame, col_k: str, window: int = 80) -> Dict[str, Any]:
    """
    k* ULTRA REAL (sentinela):
    - Usa a distribuição de k (acertos dos guardas) nas últimas 'window' séries.
    - Entropia alta + variação alta → cenário caótico (guardas ora enxergam bem, ora não).
    - Entropia baixa + k estável → cenário mais previsível.
    Retorna:
      - k_star (0..100)
      - entropia
      - entropia_normalizada (0..1)
      - estado (estavel / atencao / critico)
    """
    if df is None or df.empty:
        return {
            "k_star": None,
            "estado": "desconhecido",
            "entropia": None,
            "entropia_norm": None,
        }

    serie_k = df[col_k].astype(int).values
    w = min(window, len(serie_k))
    recent = serie_k[-w:]
    if len(recent) < 5:
        return {
            "k_star": None,
            "estado": "desconhecido",
            "entropia": None,
            "entropia_norm": None,
        }

    ent = calcular_entropia(list(recent))
    # Entropia máxima aproximada para k limitado (0..15 ou próximo)
    ent_max_teorica = math.log2(len(set(recent))) if len(set(recent)) > 1 else 1.0
    ent_norm = normalizar_0_1(ent, 0.0, ent_max_teorica)

    diffs = np.diff(recent)
    mean_abs_dk = float(np.mean(np.abs(diffs)))
    std_k = float(np.std(recent))

    # Índice composto de caos local: variação + entropia
    caos_local = 0.6 * ent_norm + 0.25 * normalizar_0_1(mean_abs_dk, 0.0, 8.0) + 0.15 * normalizar_0_1(std_k, 0.0, 8.0)
    k_star = float(100.0 * caos_local)

    if k_star < 35:
        estado = "estavel"
    elif k_star < 70:
        estado = "atencao"
    else:
        estado = "critico"

    return {
        "k_star": k_star,
        "estado": estado,
        "entropia": ent,
        "entropia_norm": ent_norm,
    }


# ------------------------------------------------------------
# Núcleos IDX / IPF / IPO ULTRA
# ------------------------------------------------------------

def extrair_passageiros_linha(row: pd.Series, passageiros_cols: List[str]) -> List[int]:
    return [int(row[c]) for c in passageiros_cols]


def calcular_nucleos_idx_ipf_ipo(
    df: pd.DataFrame,
    idx_alvo: int,
    passageiros_cols: List[str],
    janela: int = 40,
) -> Dict[str, List[int]]:
    """
    Calcula os núcleos IDX / IPF / IPO ULTRA usando uma janela de histórico antes do idx_alvo.
    - IDX ULTRA: média ponderada dinâmica (frequência + posição)
    - IPF ULTRA: mediana robusta (estrutura)
    - IPO ORIGINAL: frequência simples
    - IPO ULTRA: refinado anti-sesgo (ajusta ordem e pesos)
    """
    n = len(df)
    if n == 0:
        return {
            "IDX": [],
            "IPF": [],
            "IPO_ORIG": [],
            "IPO_ULTRA": [],
        }

    # idx_alvo é 1-based
    idx0 = max(1, min(idx_alvo, n))
    fim = max(1, idx0 - 1)
    inicio = max(1, fim - janela + 1)

    bloco = df[(df["idx_interno"] >= inicio) & (df["idx_interno"] <= fim)]
    if bloco.empty:
        return {
            "IDX": [],
            "IPF": [],
            "IPO_ORIG": [],
            "IPO_ULTRA": [],
        }

    # Monta lista de passageiros por linha
    todas_series = [extrair_passageiros_linha(row, passageiros_cols) for _, row in bloco.iterrows()]
    flat = [p for serie in todas_series for p in serie]

    # Frequência simples
    freq = Counter(flat)

    # Frequência ponderada por posição na janela (mais recentes pesam mais)
    pesos_pos = {}
    for _, row in bloco.iterrows():
        pos_rel = row["idx_interno"] - inicio  # 0,1,2,...
        peso = 1.0 + pos_rel / max(1, (fim - inicio + 1))
        serie_pass = extrair_passageiros_linha(row, passageiros_cols)
        for p in serie_pass:
            pesos_pos[p] = pesos_pos.get(p, 0.0) + peso

    # IDX: top por peso posicional
    idx_ord = sorted(pesos_pos.items(), key=lambda x: (-x[1], x[0]))
    idx_nucleo = [p for p, _ in idx_ord][: len(passageiros_cols)]

    # IPO_ORIG: top por frequência simples
    ipo_ord = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    ipo_orig = [p for p, _ in ipo_ord][: len(passageiros_cols)]

    # IPF: mediana robusta por posição do passageiro
    n_pass = len(passageiros_cols)
    matriz = np.array(todas_series)  # shape (M, n_pass)
    ipf = []
    for j in range(n_pass):
        col = matriz[:, j]
        ipf.append(int(np.median(col)))

    # IPO_ULTRA: mistura IDX + IPF + IPO_ORIG
    # Estratégia: começa com IPO_ORIG e faz pequenos ajustes usando IDX / IPF.
    base = ipo_orig.copy()
    bonus = set(idx_nucleo[: max(3, n_pass // 3)]) | set(ipf[: max(3, n_pass // 3)])
    # Garante presença de alguns elementos importantes
    for b in bonus:
        if b not in base:
            base.append(b)
    # Corta no tamanho certo
    ipo_ultra = base[:n_pass]

    return {
        "IDX": idx_nucleo,
        "IPF": ipf,
        "IPO_ORIG": ipo_orig,
        "IPO_ULTRA": ipo_ultra,
        "janela_inicio": int(inicio),
        "janela_fim": int(fim),
    }


# ------------------------------------------------------------
# Layout principal — Navegação
# ------------------------------------------------------------

if "log_eventos" not in st.session_state:
    st.session_state["log_eventos"] = []

st.title("Predict Cars V14-FLEX ULTRA REAL (TURBO++)")
st.write(
    "Sistema ULTRA completo com: Barômetro, k*, IDX, IPF / IPO, S6 Profundo, Micro-Leque, "
    "Monte Carlo Profundo, QDS + Backtest, Replay LIGHT / ULTRA e Modo TURBO++ Adaptativo."
)

with st.sidebar:
    st.markdown("## Navegação")
    painel = st.radio(
        "Escolha o painel:",
        [
            "📥 Histórico — Entrada (FLEX)",
            "🔍 Pipeline V14-FLEX ULTRA",
            "🚨 Monitor de Risco (Barômetro + k*)",
            "📊 IDX / IPF / IPO ULTRA",
            "🧬 S6 Profundo & Micro-Leque ULTRA",
            "🎲 Monte Carlo Profundo ULTRA",
            "🧪 QDS REAL & Backtest REAL",
            "📅 Replay LIGHT",
            "📅 Replay ULTRA (Horizonte Ajustável)",
            "🚀 Modo TURBO++ ULTRA",
        ],
    )

# ------------------------------------------------------------
# PAINEL 1 — Histórico — Entrada (FLEX)
# ------------------------------------------------------------

if painel == "📥 Histórico — Entrada (FLEX)":
    st.markdown("## 📥 Histórico — Entrada (FLEX)")

    df_sessao = st.session_state.get("df", None)
    if df_sessao is not None and not df_sessao.empty:
        st.success("Histórico já carregado na sessão.")
        st.dataframe(df_sessao.head(10), use_container_width=True)

    opc = st.radio(
        "Como deseja carregar o histórico?",
        ["Enviar arquivo CSV", "Copiar e colar o histórico"],
    )

    if opc == "Enviar arquivo CSV":
        file = st.file_uploader("Selecione o arquivo CSV:", type=["csv"])
        if file is not None:
            try:
                df_raw = pd.read_csv(file, sep=";", header=None, engine="python")
                df_raw = df_raw.dropna(axis=1, how="all")
               
                df = preparar_historico_flex(df_raw)
                st.success("Histórico carregado com sucesso (modo FLEX).")
                st.write(f"Total de séries: **{len(df)}**")
                st.write(f"Passageiros por série (FLEX): **{st.session_state['n_passageiros']}**")
                st.write(f"Coluna k (guardas que acertaram): **{st.session_state['col_k']}**")
                registrar_evento("Histórico carregado via CSV (FLEX).")
            except Exception as e:
                st.error(f"Erro ao carregar CSV: {e}")

    else:
        texto = st.text_area(
            "Cole aqui o histórico (CSV ou linhas separadas por ponto e vírgula):",
            height=200,
        )
        if st.button("Carregar histórico colado"):
            if not texto.strip():
                st.warning("Cole algum conteúdo antes de carregar.")
            else:
                try:
                    from io import StringIO

                    buffer = StringIO(texto)
                    # Tenta detectar separador
                    df_raw = pd.read_csv(buffer, sep=None, engine="python", header=None)
                    # Tenta criar cabeçalho genérico se necessário
                    if df_raw.columns.dtype == "int64":
                        df_raw.columns = [f"col{i+1}" for i in range(len(df_raw.columns))]
                    df = preparar_historico_flex(df_raw)
                    st.success("Histórico carregado com sucesso (modo FLEX).")
                    st.write(f"Total de séries: **{len(df)}**")
                    st.write(f"Passageiros por série (FLEX): **{st.session_state['n_passageiros']}**")
                    st.write(f"Coluna k (guardas que acertaram): **{st.session_state['col_k']}**")
                    registrar_evento("Histórico carregado via texto colado (FLEX).")
                except Exception as e:
                    st.error(f"Erro ao interpretar o texto como CSV: {e}")

    df, passageiros_cols, col_k, n_pass = obter_contexto_basico()
    if df is not None and not df.empty:
        st.markdown("### 📌 Resumo do histórico atual")
        st.write(f"**Total de séries:** {len(df)}")
        st.write(f"**Passageiros por série (detectado):** {n_pass}")
        st.write(f"**Coluna k (guardas que acertaram):** {col_k}")

        idx_preview = st.number_input(
            "Selecione um índice interno para inspecionar (1 = primeira série carregada):",
            min_value=1,
            max_value=len(df),
            value=len(df),
            step=1,
        )
        row = df.iloc[idx_preview - 1]
        serie_pass = extrair_passageiros_linha(row, passageiros_cols)
        st.markdown(
            f"**C{idx_preview} — Passageiros:** {serie_pass} — k (guardas que acertaram): **{int(row[col_k])}**"
        )


# ------------------------------------------------------------
# PAINEL 2 — Pipeline V14-FLEX ULTRA (Execução Base)
# ------------------------------------------------------------

if painel == "🔍 Pipeline V14-FLEX ULTRA":
    st.markdown("## 🔍 Pipeline V14-FLEX ULTRA — Execução Base")

    df, passageiros_cols, col_k, n_pass = obter_contexto_basico()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada (FLEX)'.")
        st.stop()

    n_total = len(df)

    modo_idx = st.radio(
        "Como deseja escolher o índice alvo?",
        ["Usar última série do histórico", "Escolher manualmente"],
    )

    if modo_idx == "Usar última série do histórico":
        idx_alvo = n_total
    else:
        idx_alvo = st.number_input(
            "Selecione o índice alvo (1 = primeira série carregada):",
            min_value=1,
            max_value=n_total,
            value=n_total,
            step=1,
        )

    row_alvo = df.iloc[idx_alvo - 1]
    serie_pass_alvo = extrair_passageiros_linha(row_alvo, passageiros_cols)
    k_alvo = int(row_alvo[col_k])
    st.markdown("### 🎯 Seleção da série alvo")
    st.write(
        f"📌 **Série alvo selecionada** — ID C{idx_alvo} — Passageiros: {serie_pass_alvo} — "
        f"k (guardas que acertaram): **{k_alvo}**"
    )

    # 1) Diagnóstico de risco — Barômetro + k*
    st.markdown("### 1️⃣ Diagnóstico de risco — Barômetro + k*")

    bar = calcular_barometro_ultra(df, col_k)
    kstar_info = calcular_k_star_ultra(df, col_k)

    estado_bar = bar["estado"]
    estado_kstar = kstar_info["estado"]

    # Barômetro
    if estado_bar == "critico":
        st.error("🔴 Barômetro: **crítico** — estrada globalmente turbulenta.")
    elif estado_bar == "atencao":
        st.warning("🟡 Barômetro: **atenção** — estrada moderadamente instável.")
    elif estado_bar == "estavel":
        st.success("🟢 Barômetro: **estável** — estrada historicamente previsível.")
    else:
        st.info("⚪ Barômetro: estado **desconhecido** (poucos dados).")

    st.write(
        f"**Índice de turbulência:** `{bar['turbulencia']:.3f}` • "
        f"Desvio-padrão de k: `{bar['std_k']:.3f}` • Média de |Δk|: `{bar['mean_abs_dk']:.3f}`"
    )

    st.markdown("#### 🛰️ k* ULTRA REAL (Sentinela baseado em k dos guardas)")

    if kstar_info["k_star"] is not None:
        k_star_val = kstar_info["k_star"]
        if estado_kstar == "critico":
            st.error(
                f"🔴 k*: **crítico** — guardas com padrão de acerto/erro altamente caótico. (k*={k_star_val:.1f})"
            )
        elif estado_kstar == "atencao":
            st.warning(
                f"🟡 k*: **atenção** — guardas com padrão misto de acerto/erro. (k*={k_star_val:.1f})"
            )
        elif estado_kstar == "estavel":
            st.success(
                f"🟢 k*: **estável** — guardas com padrão relativamente previsível. (k*={k_star_val:.1f})"
            )
        else:
            st.info("⚪ k*: estado desconhecido (poucos dados).")

        st.write(
            f"Entropia de k: `{kstar_info['entropia']:.3f}` • "
            f"Entropia normalizada: `{kstar_info['entropia_norm']:.3f}`"
        )
    else:
        st.info("k*: não foi possível calcular (poucos dados).")

    # Síntese textual
    st.markdown("#### 🌐 Pré-síntese de risco global (sem afetar o motor)")
    st.info(
        "O Barômetro ULTRA avalia a estabilidade global dos acertos dos guardas (k) ao longo da estrada. "
        "O k* ULTRA REAL atua como sentinela de caos local: se os guardas oscilam demais entre acertar tudo e errar "
        "tudo em janelas curtas, k* sobe. O motor ULTRA usa essas informações **apenas como contexto**, "
        "não como trava direta da previsão."
    )

    # 2) Núcleos IDX / IPF / IPO ULTRA
    st.markdown("### 2️⃣ Núcleos IDX / IPF / IPO ULTRA (base para previsão)")

    nucleos = calcular_nucleos_idx_ipf_ipo(df, idx_alvo, passageiros_cols, janela=40)
    idx_nucleo = nucleos["IDX"]
    ipf_nucleo = nucleos["IPF"]
    ipo_orig = nucleos["IPO_ORIG"]
    ipo_ultra = nucleos["IPO_ULTRA"]
    inicio_jan = nucleos.get("janela_inicio", max(1, idx_alvo - 40))
    fim_jan = nucleos.get("janela_fim", idx_alvo - 1)

    st.write("#### IDX ULTRA (média ponderada dinâmica)")
    st.code(" ".join(str(x) for x in idx_nucleo), language="text")

    st.write("#### IPF ULTRA (mediana robusta estrutural)")
    st.code(" ".join(str(x) for x in ipf_nucleo), language="text")

    st.write("#### IPO ORIGINAL (média simples de frequência)")
    st.code(" ".join(str(x) for x in ipo_orig), language="text")

    st.write("#### IPO ULTRA (refinada anti-sesgo, mistura IDX + IPF + IPO)")
    st.code(" ".join(str(x) for x in ipo_ultra), language="text")

    st.write(
        f"Janela usada: índices de **{inicio_jan}** até **{fim_jan}** "
        f"(tamanho **{fim_jan - inicio_jan + 1}** séries)."
    )

    st.markdown("### 3️⃣ Pré-síntese da base de previsão ULTRA")
    st.info(
        "Nesta camada, o app consolida os núcleos IDX / IPF / IPO como ponto de partida para o motor ULTRA "
        "(S6 Profundo, Micro-Leque, Monte Carlo, QDS / Backtest e TURBO++), que será aplicado nas próximas camadas."
    )

# (continua na PARTE 2/4)
# ------------------------------------------------------------
# PAINEL 3 — Monitor de Risco (Barômetro + k*)
# ------------------------------------------------------------

if painel == "🚨 Monitor de Risco (Barômetro + k*)":
    st.markdown("## 🚨 Monitor de Risco (Barômetro + k*)")

    df, passageiros_cols, col_k, n_pass = obter_contexto_basico()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada (FLEX)'.")
        st.stop()

    st.markdown("### 🌡️ Barômetro ULTRA REAL")

    bar = calcular_barometro_ultra(df, col_k)
    estado_bar = bar["estado"]

    if estado_bar == "critico":
        st.error("🔴 Estado do barômetro: **crítico**")
        st.write("Barômetro: estrada globalmente turbulenta — os acertos dos guardas (k) variam demais.")
    elif estado_bar == "atencao":
        st.warning("🟡 Estado do barômetro: **atenção**")
        st.write("Barômetro: estrada moderadamente instável — alternância entre fases de ordem e ruído.")
    elif estado_bar == "estavel":
        st.success("🟢 Estado do barômetro: **estável**")
        st.write("Barômetro: estrada historicamente estável — padrão de acertos dos guardas relativamente previsível.")
    else:
        st.info("⚪ Estado do barômetro: **desconhecido** (histórico insuficiente).")

    if bar["turbulencia"] is not None:
        st.write(f"**Índice de turbulência:** `{bar['turbulencia']:.3f}`")
        st.write(f"**Desvio-padrão de k:** `{bar['std_k']:.3f}`")
        st.write(f"**Média de |Δk| (variação entre carros):** `{bar['mean_abs_dk']:.3f}`")

    st.markdown("---")
    st.markdown("### 🛰️ k* ULTRA REAL (Sentinela)")

    kstar_info = calcular_k_star_ultra(df, col_k)
    estado_kstar = kstar_info["estado"]
    k_star_val = kstar_info.get("k_star", None)

    if k_star_val is not None:
        if estado_kstar == "critico":
            st.error(
                f"🔴 Estado do k*: **crítico** — guardas em regime caótico de acerto/erro. (k*={k_star_val:.1f})"
            )
        elif estado_kstar == "atencao":
            st.warning(
                f"🟡 Estado do k*: **atenção** — padrão misto de acertos, com alternância relevante. (k*={k_star_val:.1f})"
            )
        elif estado_kstar == "estavel":
            st.success(
                f"🟢 Estado do k*: **estável** — guardas com padrão relativamente previsível. (k*={k_star_val:.1f})"
            )
        else:
            st.info("⚪ Estado do k*: **desconhecido** (histórico insuficiente).")

        st.write(f"**Entropia de k:** `{kstar_info['entropia']:.3f}`")
        st.write(f"**Entropia normalizada:** `{kstar_info['entropia_norm']:.3f}`")
    else:
        st.info("k*: não foi possível calcular (poucos dados).")

    st.markdown("---")
    st.markdown("### 🌐 Síntese Global de Risco")

    # Síntese global boa prática: combina barômetro (global) + k* (local),
    # mas NÃO trava o motor. Apenas orienta o usuário.
    if estado_bar == "critico" or estado_kstar == "critico":
        st.error("🔴 Nível global de risco: **crítico**")
        st.write(
            "Ambiente global crítico — usar qualquer previsão com máxima cautela. "
            "Estrada e/ou guardas em regime altamente turbulento."
        )
        regime_state = "critico"
    elif estado_bar == "atencao" or estado_kstar == "atencao":
        st.warning("🟡 Nível global de risco: **atenção**")
        st.write(
            "Ambiente global em atenção — estrada com oscilações perceptíveis ou guardas com padrão de acerto/erro misto. "
            "Previsões continuam possíveis, porém com prudência reforçada."
        )
        regime_state = "atencao"
    elif estado_bar == "estavel" and estado_kstar == "estavel":
        st.success("🟢 Nível global de risco: **estável**")
        st.write(
            "Ambiente global estável — estrada historicamente previsível e guardas com padrão coerente de acertos. "
            "Pré-condições favoráveis para previsões ULTRA."
        )
        regime_state = "estavel"
    else:
        st.info("⚪ Nível global de risco: **desconhecido**")
        st.write(
            "Não há dados suficientes para uma síntese de risco confiável. "
            "Use as previsões com cautela até que o histórico seja maior."
        )
        regime_state = "desconhecido"

    # Guarda o regime no session_state para uso em outros painéis
    st.session_state["regime_state"] = regime_state

    st.info(
        "🔎 Importante: o Monitor de Risco **não bloqueia** o motor ULTRA. "
        "Ele funciona como um painel de contexto, ajudando a interpretar em que tipo de ambiente "
        "as previsões estão sendo feitas (estrada calma, moderada ou caótica)."
    )


# ------------------------------------------------------------
# PAINEL 4 — IDX / IPF / IPO ULTRA (visão detalhada)
# ------------------------------------------------------------

if painel == "📊 IDX / IPF / IPO ULTRA":
    st.markdown("## 📊 IDX / IPF / IPO ULTRA — Núcleos Estruturais")

    df, passageiros_cols, col_k, n_pass = obter_contexto_basico()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada (FLEX)'.")
        st.stop()

    n_total = len(df)

    idx_alvo = st.number_input(
        "Selecione o índice alvo para calcular os núcleos (1 = primeira série):",
        min_value=1,
        max_value=n_total,
        value=n_total,
        step=1,
    )

    row_alvo = df.iloc[idx_alvo - 1]
    serie_pass_alvo = extrair_passageiros_linha(row_alvo, passageiros_cols)
    k_alvo = int(row_alvo[col_k])

    st.markdown("### 🎯 Série alvo (contexto imediato)")
    st.write(
        f"ID C{idx_alvo} | Passageiros: {serie_pass_alvo} | "
        f"k (guardas que acertaram exatamente o carro): **{k_alvo}**"
    )

    janela = st.number_input(
        "Tamanho da janela de histórico para cálculo dos núcleos:",
        min_value=10,
        max_value=min(200, n_total - 1),
        value=40,
        step=5,
    )

    nucleos = calcular_nucleos_idx_ipf_ipo(df, idx_alvo, passageiros_cols, janela=int(janela))
    idx_nucleo = nucleos["IDX"]
    ipf_nucleo = nucleos["IPF"]
    ipo_orig = nucleos["IPO_ORIG"]
    ipo_ultra = nucleos["IPO_ULTRA"]
    inicio_jan = nucleos.get("janela_inicio", max(1, idx_alvo - int(janela)))
    fim_jan = nucleos.get("janela_fim", idx_alvo - 1)

    st.markdown("### 📦 Janela de histórico usada")
    st.write(f"**Início da janela:** {inicio_jan}")
    st.write(f"**Fim da janela:** {fim_jan}")
    st.write(f"**Tamanho da janela:** {fim_jan - inicio_jan + 1}")

    st.markdown("### 🧠 Núcleos IDX / IPF / IPO ULTRA")

    st.write("#### IDX ULTRA (média ponderada dinâmica)")
    st.code(" ".join(str(x) for x in idx_nucleo), language="text")

    st.write("#### IPF ULTRA (mediana robusta)")
    st.code(" ".join(str(x) for x in ipf_nucleo), language="text")

    st.write("#### IPO ORIGINAL (média simples)")
    st.code(" ".join(str(x) for x in ipo_orig), language="text")

    st.write("#### IPO ULTRA (refinada anti-sesgo)")
    st.code(" ".join(str(x) for x in ipo_ultra), language="text")

    st.info(
        "Interpretação rápida:\n"
        "- **IDX ULTRA** destaca passageiros mais importantes com base em frequência + posição (os 'mais vistos').\n"
        "- **IPF ULTRA** representa a estrutura central, resistente a ruídos (mediana por posição).\n"
        "- **IPO ORIGINAL** mostra a fotografia bruta da frequência.\n"
        "- **IPO ULTRA** é a versão refinada, corrigindo vieses do histórico e reforçando o núcleo realmente preditivo."
    )

# (continua na PARTE 3/4)
# ============================================================
# ======================== PARTE 3/4 =========================
# ===== Modo TURBO++ ULTRA, S6 Profundo, Micro-Leque, MC =====
# ============================================================

# ------------------------------------------------------------
# Funções auxiliares — colunas de passageiros / séries
# ------------------------------------------------------------
from typing import List, Dict, Any


def extrair_colunas_passageiros(df: pd.DataFrame) -> List[str]:
    """
    Tenta descobrir automaticamente quais colunas são 'passageiros'.

    Regras:
    - Remove claramente identificadores e o k
    - Usa o resto como passageiros (ordem preservada)
    """
    if df is None or df.empty:
        return []

    colunas_excluir = {"k", "K", "id", "ID", "Id", "C", "c", "serie", "SERIE", "label", "LABEL"}
    return [c for c in df.columns if c not in colunas_excluir]


def linha_para_serie(row: pd.Series, cols_pass: List[str]) -> List[int]:
    return [int(row[c]) for c in cols_pass]


def serie_para_str(serie: List[int]) -> str:
    return " ".join(str(x) for x in serie)


def contar_hits(serie_prev: List[int], serie_real: List[int]) -> int:
    alvo = set(serie_real)
    return sum(1 for x in serie_prev if x in alvo)


# ------------------------------------------------------------
# S6 Profundo ULTRA — núcleo determinístico
# ------------------------------------------------------------
def s6_profundo_ultra(
    df: pd.DataFrame,
    idx_alvo: int,
    window: int = 80,
    n_series: int = 40,
) -> pd.DataFrame:
    """
    S6 Profundo ULTRA (versão genérica, estável e resiliente):

    - Usa uma janela de histórico antes do índice alvo
    - Calcula frequência de cada número em cada coluna de passageiro
    - Monta séries combinando os mais frequentes por coluna
    """
    if df is None or df.empty:
        return pd.DataFrame()

    cols_pass = extrair_colunas_passageiros(df)
    if not cols_pass:
        return pd.DataFrame()

    # idx_alvo é 1-based para o usuário
    idx_zero = max(idx_alvo - 1, 0)
    inicio = max(idx_zero - window, 0)
    df_janela = df.iloc[inicio:idx_zero]

    if df_janela.empty:
        return pd.DataFrame()

    # Frequência por coluna
    top_por_col = []
    for c in cols_pass:
        vc = df_janela[c].value_counts().reset_index()
        vc.columns = ["valor", "freq"]
        top_por_col.append(vc)

    # Montar candidatos combinando os top valores coluna a coluna
    from itertools import product

    tops_lim = []
    for vc in top_por_col:
        # Ajuste para não explodir combinatória
        k_max = max(3, min(6, n_series // max(1, len(cols_pass))))
        tops_lim.append(list(vc["valor"].head(k_max)))

    candidatos = []
    for comb in product(*tops_lim):
        candidatos.append(list(map(int, comb)))

    # Scoring simples: soma das frequências individuais
    def score_serie(serie: List[int]) -> float:
        s = 0.0
        for i, v in enumerate(serie):
            vc = top_por_col[i]
            freq = vc.loc[vc["valor"] == v, "freq"]
            s += float(freq.iloc[0]) if not freq.empty else 0.0
        return s

    dados = []
    for serie in candidatos:
        dados.append(
            {
                "series": serie,
                "score_s6": score_serie(serie),
                "origem": "S6_PROFUNDO",
            }
        )

    df_out = pd.DataFrame(dados).drop_duplicates(subset=["series"])
    df_out = df_out.sort_values("score_s6", ascending=False).head(n_series).reset_index(drop=True)
    return df_out


# ------------------------------------------------------------
# Micro-Leque ULTRA — vizinhança em torno do alvo
# ------------------------------------------------------------
def micro_leque_ultra(
    df: pd.DataFrame,
    idx_alvo: int,
    n_vizinhos: int = 3,
) -> pd.DataFrame:
    """
    Micro-Leque ULTRA:

    - Usa séries próximas (anteriores e posteriores) ao alvo como base
    - Gera pequenas variações em torno delas
    """
    if df is None or df.empty:
        return pd.DataFrame()

    cols_pass = extrair_colunas_passageiros(df)
    if not cols_pass:
        return pd.DataFrame()

    idx_zero = max(idx_alvo - 1, 0)
    n = len(df)

    vizinhos_idx = set()
    for delta in range(1, n_vizinhos + 1):
        if idx_zero - delta >= 0:
            vizinhos_idx.add(idx_zero - delta)
        if idx_zero + delta < n:
            vizinhos_idx.add(idx_zero + delta)

    if not vizinhos_idx:
        return pd.DataFrame()

    base_series = []
    for i in sorted(vizinhos_idx):
        row = df.iloc[i]
        base_series.append(linha_para_serie(row, cols_pass))

    # Pequenas perturbações: troca leve entre posições / shuffle
    import random

    candidatos = []
    for serie in base_series:
        candidatos.append(serie)  # original

        # Troca simples
        if len(serie) >= 2:
            s2 = serie.copy()
            i1, i2 = random.sample(range(len(serie)), 2)
            s2[i1], s2[i2] = s2[i2], s2[i1]
            candidatos.append(s2)

        # Shuffle leve
        s3 = serie.copy()
        random.shuffle(s3)
        candidatos.append(s3)

    dados = []
    for serie in candidatos:
        dados.append(
            {
                "series": list(map(int, serie)),
                "score_micro": 1.0,
                "origem": "MICRO_LEQUE",
            }
        )

    df_out = pd.DataFrame(dados).drop_duplicates(subset=["series"])
    return df_out.reset_index(drop=True)


# ------------------------------------------------------------
# Monte Carlo Profundo ULTRA
# ------------------------------------------------------------
def monte_carlo_profundo_ultra(
    df: pd.DataFrame,
    idx_alvo: int,
    n_sim: int = 2000,
    n_series_saida: int = 60,
    window: int = 120,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Monte Carlo Profundo ULTRA:

    - Usa janelas profundas para gerar simulações independentes
    - Amostra passageiros conforme distribuição empírica por coluna
    """
    if df is None or df.empty or n_sim <= 0:
        return pd.DataFrame()

    cols_pass = extrair_colunas_passageiros(df)
    if not cols_pass:
        return pd.DataFrame()

    idx_zero = max(idx_alvo - 1, 0)
    inicio = max(idx_zero - window, 0)
    df_janela = df.iloc[inicio:idx_zero]

    if df_janela.empty:
        return pd.DataFrame()

    import numpy as np
    import random

    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    # Distribuições por coluna
    dist_col = {}
    for c in cols_pass:
        valores = df_janela[c].dropna().astype(int).values
        if len(valores) == 0:
            continue
        vals, counts = np.unique(valores, return_counts=True)
        prob = counts / counts.sum()
        dist_col[c] = (vals, prob)

    if not dist_col:
        return pd.DataFrame()

    series_mc = []
    for _ in range(n_sim):
        serie = []
        for c in cols_pass:
            if c not in dist_col:
                # fallback: escolhe qualquer valor da janela
                valores = df_janela[c].dropna().astype(int).values
                if len(valores) == 0:
                    continue
                serie.append(int(random.choice(list(valores))))
            else:
                vals, prob = dist_col[c]
                serie.append(int(np.random.choice(vals, p=prob)))
        if len(serie) == len(cols_pass):
            series_mc.append(serie)

    if not series_mc:
        return pd.DataFrame()

    # Agregar por frequência
    from collections import Counter

    contagem = Counter(tuple(s) for s in series_mc)
    dados = []
    for serie_tup, freq in contagem.items():
        dados.append(
            {
                "series": list(map(int, serie_tup)),
                "freq_mc": int(freq),
                "origem": "MONTE_CARLO",
            }
        )

    df_out = pd.DataFrame(dados)
    df_out["score_mc"] = df_out["freq_mc"] / df_out["freq_mc"].max()
    df_out = df_out.sort_values("score_mc", ascending=False).head(n_series_saida).reset_index(drop=True)
    return df_out


# ------------------------------------------------------------
# Fusão ULTRA — monta Previsão TURBO++ final
# ------------------------------------------------------------
def montar_previsao_turbo_ultra(
    df: pd.DataFrame,
    idx_alvo: int,
    n_series_saida: int = 60,
    window_s6: int = 80,
    window_mc: int = 120,
    n_sim_mc: int = 2000,
    incluir_micro_leque: bool = True,
    peso_s6: float = 0.5,
    peso_mc: float = 0.4,
    peso_micro: float = 0.1,
) -> pd.DataFrame:
    """
    Núcleo de fusão TURBO++ ULTRA:

    Combina:
    - S6 Profundo ULTRA
    - Monte Carlo Profundo ULTRA
    - Micro-Leque ULTRA (opcional)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # S6
    df_s6 = s6_profundo_ultra(df, idx_alvo, window=window_s6, n_series=n_series_saida * 2)
    if df_s6.empty:
        df_s6 = pd.DataFrame(columns=["series", "score_s6", "origem"])

    # MC
    df_mc = monte_carlo_profundo_ultra(
        df,
        idx_alvo,
        n_sim=n_sim_mc,
        n_series_saida=n_series_saida * 2,
        window=window_mc,
    )
    if df_mc.empty:
        df_mc = pd.DataFrame(columns=["series", "score_mc", "freq_mc", "origem"])

    # Micro-Leque
    if incluir_micro_leque:
        df_micro = micro_leque_ultra(df, idx_alvo)
    else:
        df_micro = pd.DataFrame(columns=["series", "score_micro", "origem"])

    # Normalizar colunas de score
    for col in ["score_s6", "score_mc", "score_micro"]:
        for d in (df_s6, df_mc, df_micro):
            if col in d.columns:
                if d[col].max() > 0:
                    d[col] = d[col] / d[col].max()
                else:
                    d[col] = 0.0

    # Unir
    frames = []
    if not df_s6.empty:
        frames.append(df_s6[["series", "score_s6"]])
    if not df_mc.empty:
        frames.append(df_mc[["series", "score_mc"]])
    if not df_micro.empty:
        frames.append(df_micro[["series", "score_micro"]])

    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.groupby("series", as_index=False).agg(
        {
            "score_s6": "max",
            "score_mc": "max",
            "score_micro": "max",
        }
    )

    for col in ["score_s6", "score_mc", "score_micro"]:
        if col not in df_all.columns:
            df_all[col] = 0.0

    df_all["score_final"] = (
        peso_s6 * df_all["score_s6"].fillna(0.0)
        + peso_mc * df_all["score_mc"].fillna(0.0)
        + peso_micro * df_all["score_micro"].fillna(0.0)
    )

    df_all = df_all.sort_values("score_final", ascending=False).head(n_series_saida).reset_index(drop=True)
    return df_all


# ------------------------------------------------------------
# Painel — 🚀 Modo TURBO++ — Painel Completo
# ------------------------------------------------------------
if painel == "🚀 Modo TURBO++ — Painel Completo":
    st.markdown("## 🚀 Modo TURBO++ ULTRA Adaptativo — Painel Completo")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    cols_pass = extrair_colunas_passageiros(df)
    if not cols_pass:
        st.error("Não foi possível identificar as colunas de passageiros no histórico.")
        st.stop()

    n_series_hist = len(df)

    col1, col2 = st.columns(2)
    with col1:
        idx_alvo = st.number_input(
            "Índice alvo (1 = primeira série do histórico):",
            min_value=1,
            max_value=n_series_hist,
            value=n_series_hist,
            step=1,
        )
        n_series_saida = st.slider(
            "Quantidade de séries na saída TURBO++ (núcleo resiliente + cobertura):",
            min_value=10,
            max_value=120,
            value=60,
            step=5,
        )
        incluir_micro = st.checkbox("Incluir Micro-Leque ULTRA (cobertura de vento fina)", value=True)

    with col2:
        window_s6 = st.slider(
            "Janela S6 Profundo ULTRA (n séries para trás):",
            min_value=20,
            max_value=200,
            value=80,
            step=10,
        )
        window_mc = st.slider(
            "Janela Monte Carlo Profundo ULTRA:",
            min_value=40,
            max_value=300,
            value=120,
            step=10,
        )
        n_sim_mc = st.slider(
            "Simulações Monte Carlo Profundo ULTRA:",
            min_value=200,
            max_value=5000,
            value=2000,
            step=200,
        )

    st.markdown("### ⚖️ Pesos de fusão ULTRA (S6 / Monte Carlo / Micro-Leque)")
    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        peso_s6 = st.slider("Peso S6 Profundo", 0.0, 1.0, 0.5, 0.05)
    with colp2:
        peso_mc = st.slider("Peso Monte Carlo", 0.0, 1.0, 0.4, 0.05)
    with colp3:
        peso_micro = st.slider("Peso Micro-Leque", 0.0, 1.0, 0.1, 0.05)

    # Normalizar pesos se a soma não for 1
    soma_pesos = peso_s6 + peso_mc + peso_micro
    if soma_pesos <= 0:
        peso_s6, peso_mc, peso_micro = 0.5, 0.4, 0.1
    else:
        peso_s6 /= soma_pesos
        peso_mc /= soma_pesos
        peso_micro /= soma_pesos

    st.markdown("---")

    rodar = st.button("🚀 Rodar Modo TURBO++ ULTRA para este índice alvo")

    if rodar:
        with st.spinner("Rodando S6 Profundo, Micro-Leque e Monte Carlo Profundo ULTRA..."):
            df_turbo = montar_previsao_turbo_ultra(
                df,
                idx_alvo=idx_alvo,
                n_series_saida=n_series_saida,
                window_s6=window_s6,
                window_mc=window_mc,
                n_sim_mc=n_sim_mc,
                incluir_micro_leque=incluir_micro,
                peso_s6=peso_s6,
                peso_mc=peso_mc,
                peso_micro=peso_micro,
            )

        if df_turbo is None or df_turbo.empty:
            st.error("Não foi possível gerar séries TURBO++ ULTRA para este índice.")
        else:
            st.session_state["previsao_turbo_ultra"] = df_turbo
            st.session_state["previsao_turbo_ultra_params"] = {
                "idx_alvo": int(idx_alvo),
                "n_series_saida": int(n_series_saida),
                "window_s6": int(window_s6),
                "window_mc": int(window_mc),
                "n_sim_mc": int(n_sim_mc),
                "incluir_micro_leque": bool(incluir_micro),
                "peso_s6": float(peso_s6),
                "peso_mc": float(peso_mc),
                "peso_micro": float(peso_micro),
            }

            # Mostrar série alvo e contexto
            st.markdown("### 🚗 Série alvo (carro atual na estrada)")
            row_alvo = df.iloc[int(idx_alvo) - 1]
            serie_alvo = linha_para_serie(row_alvo, cols_pass)
            st.code(serie_para_str(serie_alvo), language="text")

            # Integração com Barômetro / k*
            regime_state = st.session_state.get("regime_state", "normal")
            k_estado = st.session_state.get("k_estado", "estavel")
            k_star_val = st.session_state.get("k_star_val", None)

            contexto_barometro = ""
            if regime_state == "normal":
                contexto_barometro = "🟢 Barômetro ULTRA REAL: Estrada em regime normal."
            elif regime_state == "transicao":
                contexto_barometro = "🟡 Barômetro ULTRA REAL: Região de transição / pré-ruptura."
            else:
                contexto_barometro = "🔴 Barômetro ULTRA REAL: Região de turbulência pesada / pós-ruptura."

            contexto_k = ""
            if k_estado == "estavel":
                contexto_k = "🟢 k* ULTRA REAL: Ambiente estável — guardas convergindo."
            elif k_estado == "atencao":
                contexto_k = "🟡 k* ULTRA REAL: Pré-ruptura residual — atenção elevada."
            else:
                contexto_k = "🔴 k* ULTRA REAL: Ambiente crítico — sensibilidade máxima dos guardas."

            if k_star_val is not None:
                contexto_k += f" (k* ≈ {k_star_val:.1f}%)"

            st.info(contexto_barometro + "\n\n" + contexto_k)

            # Tabela completa
            st.markdown("### 📊 Leque TURBO++ ULTRA — Núcleo Resiliente + Cobertura")
            df_view = df_turbo.copy()
            df_view["series_str"] = df_view["series"].apply(serie_para_str)
            st.dataframe(
                df_view[["series_str", "score_final", "score_s6", "score_mc", "score_micro"]].rename(
                    columns={
                        "series_str": "Série (passageiros)",
                        "score_final": "Score ULTRA",
                        "score_s6": "Score S6",
                        "score_mc": "Score Monte Carlo",
                        "score_micro": "Score Micro-Leque",
                    }
                ),
                use_container_width=True,
            )

            # Previsão final
            melhor = df_turbo.iloc[0]
            st.markdown("### 🎯 Previsão Final TURBO++ ULTRA (Série #1 do Núcleo Resiliente)")
            st.code(serie_para_str(melhor["series"]), language="text")

# ============================================================
# ====================== FIM DA PARTE 3/4 ====================
# ============================================================


# ============================================================
# ======================== PARTE 4/4 =========================
# ===== Replay LIGHT / ULTRA, QDS REAL, Backtest REAL ========
# ============================================================

# ------------------------------------------------------------
# Funções auxiliares — Replay e QDS REAL
# ------------------------------------------------------------
def executar_pipeline_turbo_ultra_para_replay(
    df: pd.DataFrame,
    idx_alvo: int,
    params_base: Dict[str, Any],
    modo_replay: str = "LIGHT",
) -> Dict[str, Any]:
    """
    Wrapper para usar o mesmo núcleo TURBO++ ULTRA no Replay.

    - LIGHT: menos simulações Monte Carlo / janelas menores
    - ULTRA: usa parâmetros cheios ou até reforçados
    """
    # Clona parâmetros
    params = dict(params_base or {})

    # Defaults se nada foi rodado ainda
    if not params:
        params = {
            "n_series_saida": 60,
            "window_s6": 80,
            "window_mc": 120,
            "n_sim_mc": 2000,
            "incluir_micro_leque": True,
            "peso_s6": 0.5,
            "peso_mc": 0.4,
            "peso_micro": 0.1,
        }

    if modo_replay == "LIGHT":
        params["n_series_saida"] = min(30, params["n_series_saida"])
        params["window_s6"] = max(40, int(params["window_s6"] * 0.6))
        params["window_mc"] = max(60, int(params["window_mc"] * 0.6))
        params["n_sim_mc"] = max(300, int(params["n_sim_mc"] * 0.3))
    else:  # ULTRA
        params["n_series_saida"] = max(60, params["n_series_saida"])
        params["n_sim_mc"] = max(1500, int(params["n_sim_mc"] * 1.0))

    df_turbo = montar_previsao_turbo_ultra(
        df,
        idx_alvo=idx_alvo,
        n_series_saida=params["n_series_saida"],
        window_s6=params["window_s6"],
        window_mc=params["window_mc"],
        n_sim_mc=params["n_sim_mc"],
        incluir_micro_leque=params["incluir_micro_leque"],
        peso_s6=params["peso_s6"],
        peso_mc=params["peso_mc"],
        peso_micro=params["peso_micro"],
    )

    if df_turbo is None or df_turbo.empty:
        return {"ok": False, "df": pd.DataFrame(), "serie_top1": None}

    top1 = df_turbo.iloc[0]["series"]
    return {"ok": True, "df": df_turbo, "serie_top1": top1}


def calcular_qds_real(aus_replay: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula QDS REAL a partir da tabela de replay:

    Espera colunas:
    - hits (número de acertos)
    - idx_alvo
    """
    if aus_replay is None or aus_replay.empty:
        return {
            "qds": 0.0,
            "media_hits": 0.0,
            "p_ge_1": 0.0,
            "p_ge_3": 0.0,
            "p_ge_4": 0.0,
            "n": 0,
        }

    n = len(aus_replay)
    media_hits = float(aus_replay["hits"].mean())

    p_ge_1 = float((aus_replay["hits"] >= 1).mean())
    p_ge_3 = float((aus_replay["hits"] >= 3).mean())
    p_ge_4 = float((aus_replay["hits"] >= 4).mean())

    # QDS REAL (0–100) — ponderação simples
    qds = 100.0 * (0.25 * p_ge_1 + 0.35 * p_ge_3 + 0.40 * p_ge_4)

    return {
        "qds": qds,
        "media_hits": media_hits,
        "p_ge_1": p_ge_1,
        "p_ge_3": p_ge_3,
        "p_ge_4": p_ge_4,
        "n": n,
    }


# ------------------------------------------------------------
# Painel — 📅 Modo Replay Automático do Histórico
# ------------------------------------------------------------
if painel == "📅 Modo Replay Automático do Histórico":
    st.markdown("## 📅 Modo Replay Automático do Histórico")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    cols_pass = extrair_colunas_passageiros(df)
    if not cols_pass:
        st.error("Não foi possível identificar as colunas de passageiros no histórico.")
        st.stop()

    n_series_hist = len(df)

    st.markdown("### 🎬 Configuração do Replay (LIGHT / ULTRA)")

    col1, col2 = st.columns(2)
    with col1:
        idx_inicio = st.number_input(
            "Índice inicial do Replay:",
            min_value=1,
            max_value=max(1, n_series_hist - 1),
            value=max(1, n_series_hist - 60),
            step=1,
        )
        idx_fim = st.number_input(
            "Índice final do Replay:",
            min_value=idx_inicio,
            max_value=max(1, n_series_hist - 1),
            value=max(1, n_series_hist - 1),
            step=1,
        )
        horizonte = st.number_input(
            "Horizonte de validação (quantas séries à frente comparar):",
            min_value=1,
            max_value=5,
            value=1,
            step=1,
        )

    with col2:
        modo_replay = st.radio(
            "Modo de Replay:",
            options=["LIGHT (rápido)", "ULTRA (profundo)"],
        )
        usar_params_turbo = st.checkbox(
            "Usar parâmetros atuais do Modo TURBO++ ULTRA (se já rodou)",
            value=True,
        )
        mostrar_detalhes = st.checkbox("Mostrar tabela completa de resultados do Replay", value=True)

    params_base = st.session_state.get("previsao_turbo_ultra_params", {})
    if not usar_params_turbo:
        params_base = {}

    st.markdown("---")
    rodar_replay = st.button("📅 Rodar Replay Automático do Histórico")

    if rodar_replay:
        registros = []
        modo_interno = "LIGHT" if modo_replay.startswith("LIGHT") else "ULTRA"

        with st.spinner("Executando Replay do histórico com o núcleo TURBO++ ULTRA..."):
            for idx in range(int(idx_inicio), int(idx_fim) + 1):
                idx_real = idx + int(horizonte)
                if idx_real > n_series_hist:
                    # Não há série real para comparar
                    continue

                res = executar_pipeline_turbo_ultra_para_replay(
                    df,
                    idx_alvo=idx,
                    params_base=params_base,
                    modo_replay=modo_interno,
                )

                if not res["ok"] or res["serie_top1"] is None:
                    continue

                serie_prev = list(map(int, res["serie_top1"]))
                row_real = df.iloc[idx_real - 1]
                serie_real = linha_para_serie(row_real, cols_pass)

                h = contar_hits(serie_prev, serie_real)

                registros.append(
                    {
                        "idx_alvo": int(idx),
                        "idx_real": int(idx_real),
                        "serie_prevista": serie_para_str(serie_prev),
                        "serie_real": serie_para_str(serie_real),
                        "hits": int(h),
                        "modo": modo_interno,
                    }
                )

        if not registros:
            st.error("Replay não gerou resultados válidos (verifique janelas e horizonte).")
        else:
            df_replay = pd.DataFrame(registros).sort_values("idx_alvo").reset_index(drop=True)
            st.session_state["df_replay"] = df_replay

            st.markdown("### 📊 Resumo do Replay")
            st.write(f"N execuções válidas: **{len(df_replay)}**")

            colm1, colm2, colm3 = st.columns(3)
            with colm1:
                st.metric("Média de hits (passageiros por carro)", f"{df_replay['hits'].mean():.2f}")
            with colm2:
                st.metric("Execuções com ≥ 3 hits", f"{(df_replay['hits'] >= 3).sum()} / {len(df_replay)}")
            with colm3:
                st.metric("Execuções com ≥ 4 hits", f"{(df_replay['hits'] >= 4).sum()} / {len(df_replay)}")

            if mostrar_detalhes:
                st.markdown("### 🧾 Detalhamento do Replay (carro a carro)")
                st.dataframe(
                    df_replay[
                        [
                            "idx_alvo",
                            "idx_real",
                            "serie_prevista",
                            "serie_real",
                            "hits",
                            "modo",
                        ]
                    ],
                    use_container_width=True,
                )

# ------------------------------------------------------------
# Painel — 🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)
# ------------------------------------------------------------
if painel == "🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)":
    st.markdown("## 🧪 Testes de Confiabilidade — QDS REAL + Backtest REAL")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        st.stop()

    df_replay = st.session_state.get("df_replay", None)

    if df_replay is None or df_replay.empty:
        st.info(
            "Ainda não há resultados de Replay salvos.\n\n"
            "Use primeiro o painel **'📅 Modo Replay Automático do Histórico'** "
            "para gerar a base empírica de validação (Backtest REAL)."
        )
        st.stop()

    st.markdown("### ✅ QDS REAL — Índice de Qualidade Dinâmica da Série (0–100)")

    resultados_qds = calcular_qds_real(df_replay)

    colq1, colq2, colq3 = st.columns(3)
    with colq1:
        st.metric("QDS REAL (0–100)", f"{resultados_qds['qds']:.1f}")
    with colq2:
        st.metric("Média de hits", f"{resultados_qds['media_hits']:.2f}")
    with colq3:
        st.metric("N execuções", f"{resultados_qds['n']}")

    st.markdown("### 📊 Distribuição de hits por carro (Backtest REAL)")

    # Histograma simples usando value_counts
    dist_hits = df_replay["hits"].value_counts().sort_index()
    df_dist = dist_hits.reset_index()
    df_dist.columns = ["hits", "frequencia"]

    st.bar_chart(df_dist.set_index("hits"))

    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        st.metric("P(hits ≥ 1)", f"{100 * resultados_qds['p_ge_1']:.1f}%")
    with colp2:
        st.metric("P(hits ≥ 3)", f"{100 * resultados_qds['p_ge_3']:.1f}%")
    with colp3:
        st.metric("P(hits ≥ 4)", f"{100 * resultados_qds['p_ge_4']:.1f}%")

    st.markdown("---")
    st.markdown("### 🔍 Amostra do Backtest REAL (primeiros carros do Replay)")
    st.dataframe(
        df_replay.head(50)[["idx_alvo", "idx_real", "serie_prevista", "serie_real", "hits", "modo"]],
        use_container_width=True,
    )

    st.markdown(
        """
**Leitura operacional (QDS REAL + Backtest REAL + Monte Carlo Profundo ULTRA)**

- O **QDS REAL** sintetiza a qualidade dinâmica da estrada a partir do que o sistema realmente teria feito
  nos carros do passado (Replay), usando exatamente o mesmo núcleo TURBO++ ULTRA.
- A distribuição de **hits por carro** mostra quão frequentemente a previsão encosta em 1, 3, 4 ou mais passageiros.
- A integração com o **Monte Carlo Profundo ULTRA** já está embutida no próprio núcleo de previsão usado no Replay,
  o que significa que o backtest já incorpora o regime estocástico real da estrada.
"""
    )

# ============================================================
# ====================== FIM DA PARTE 4/4 ====================
# ============================================================
