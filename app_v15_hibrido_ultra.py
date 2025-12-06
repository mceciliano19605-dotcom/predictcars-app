# ============================================================
# Predict Cars V15.4-HÍBRIDO
# Núcleo V14-FLEX ULTRA + Ruído A/B + QDS REAL + Backtest REAL por QDS
# + Monte Carlo REAL por QDS + Expectativa de Acertos por Ambiente
# ============================================================

import math
import json
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# CONFIGURAÇÃO BÁSICA DO APP
# ------------------------------------------------------------

st.set_page_config(
    page_title="Predict Cars V15.4-HÍBRIDO",
    layout="wide",
)

st.markdown(
    """
# 🚗 Predict Cars V15.4-HÍBRIDO

Núcleo V14-FLEX ULTRA + Ruído Tipo A/B + QDS REAL + Backtest REAL por QDS + Monte Carlo REAL por ambiente.

Versão **V15.4**: mantém 100% do jeitão V15.2/V15.3, sem simplificações, incluindo:
- Entrada FLEX ULTRA (arquivo + texto), com número variável de passageiros (n1..nN);
- df original + df_limpo (pós-ruído Tipo A);
- k como sentinela de acertos (guardas);
- Pipeline V14-FLEX ULTRA;
- k* como barômetro da estrada;
- TURBO++ ULTRA com ruído Tipo B;
- QDS REAL ao longo da estrada;
- Backtest REAL segmentado por QDS;
- Monte Carlo REAL por QDS (V15.4) e Expectativa de acertos por ambiente;
- Testes de Confiabilidade REAL consolidando tudo.
"""
)

# ------------------------------------------------------------
# ESTADO GLOBAL NA SESSÃO
# ------------------------------------------------------------

for key in [
    "df",
    "df_limpo",
    "ruido_stats",
    "qds_stats",
    "qds_config",
    "df_qds",
    "df_backtest",
    "backtest_stats",
    "mc_stats",
    "df_mc",
]:
    if key not in st.session_state:
        st.session_state[key] = None


# ------------------------------------------------------------
# FUNÇÕES DE SUPORTE PARA ENTRADA FLEX ULTRA
# ------------------------------------------------------------

def detectar_delimitador(linha: str) -> str:
    """Detecta delimitador predominante em uma linha: ';' ou ','."""
    if linha.count(";") >= linha.count(","):
        return ";"
    return ","


def parse_linha_flex(linha: str, delim: str) -> Optional[List[str]]:
    """
    Faz o parse de uma linha de histórico flexível.
    Ex:
        C1;41;5;4;52;30;33;0
        C2;9;39;37;49;43;41;1
        41;5;4;52;30;33;0
    Retorna lista de tokens já stripados ou None se linha vazia/inválida.
    """
    linha = linha.strip()
    if not linha:
        return None
    partes = [p.strip() for p in linha.split(delim)]
    if all(p == "" for p in partes):
        return None
    return partes


def construir_df_a_partir_de_linhas(linhas: List[str]) -> pd.DataFrame:
    """
    Constrói DataFrame a partir de linhas textuais FLEX ULTRA.

    Regras:
      - Primeira coluna pode ser um identificador de série (C1, C2, etc.) ou já ser n1;
      - Última coluna é k (int);
      - As colunas internas são passageiros (n1..nN);
      - Número de passageiros é determinado pela linha com maior quantidade de campos.

    Retorno:
      df com colunas: ['serie_id', 'n1'..'nN', 'k']
    """
    # limpar linhas
    linhas_validas = []
    for ln in linhas:
        ln = ln.strip()
        if ln:
            linhas_validas.append(ln)

    if not linhas_validas:
        raise ValueError("Nenhuma linha válida encontrada no texto.")

    # detectar delimitador pela primeira linha não vazia
    delim = detectar_delimitador(linhas_validas[0])

    registros = []
    max_pass = 0

    for ln in linhas_validas:
        partes = parse_linha_flex(ln, delim)
        if not partes:
            continue

        # Considerar 2 formatos:
        # 1) Cid; n1; n2; ...; nN; k
        # 2) n1; n2; ...; nN; k
        # Detectar se o primeiro token é algo tipo 'C123'
        primeiro = partes[0]
        possui_id_explicito = (
            (primeiro.upper().startswith("C") and len(partes) >= 3)
            or (not primeiro.replace(".", "", 1).isdigit())
        )

        if possui_id_explicito:
            serie_id = primeiro
            valores = partes[1:]
        else:
            serie_id = None
            valores = partes

        if len(valores) < 2:
            # precisa ter pelo menos 1 passageiro + k
            continue

        # último é k
        k_str = valores[-1]
        passageiros_str = valores[:-1]

        try:
            k_val = int(float(k_str))
        except Exception:
            k_val = 0

        passageiros = []
        for v in passageiros_str:
            try:
                passageiros.append(int(float(v)))
            except Exception:
                passageiros.append(0)

        max_pass = max(max_pass, len(passageiros))

        registros.append(
            {
                "serie_id": serie_id,
                "passageiros": passageiros,
                "k": k_val,
            }
        )

    if not registros:
        raise ValueError("Não foi possível montar registros válidos a partir do texto.")

    # Normalizar quantidade de passageiros (preencher com NaN)
    for r in registros:
        if len(r["passageiros"]) < max_pass:
            r["passageiros"] = r["passageiros"] + [np.nan] * (max_pass - len(r["passageiros"]))

    dados = {
        "serie_id": [],
        "k": [],
    }
    for j in range(1, max_pass + 1):
        dados[f"n{j}"] = []

    for idx, r in enumerate(registros, start=1):
        sid = r["serie_id"] if r["serie_id"] is not None else f"C{idx}"
        dados["serie_id"].append(sid)
        dados["k"].append(int(r["k"]))
        for j, val in enumerate(r["passageiros"], start=1):
            dados[f"n{j}"].append(val)

    df = pd.DataFrame(dados)
    return df


def carregar_csv_flex(arquivo) -> pd.DataFrame:
    """
    Carrega um CSV flexível.

    Formatos aceitos:
      - com coluna de série explícita (C1, C2,...)
      - com colunas n1..nN + k
      - com colunas sem nome claro, mas última coluna sendo k.

    Garante saída no formato:
      ['serie_id', 'n1'..'nN', 'k']
    """
    try:
        df_raw = pd.read_csv(arquivo, sep=None, engine="python")
    except Exception:
        arquivo.seek(0)
        df_raw = pd.read_csv(arquivo, sep=";")

    if df_raw is None or df_raw.empty:
        raise ValueError("CSV vazio ou não pôde ser carregado.")

    df = df_raw.copy()

    # Tentar detectar coluna que representa série_id
    col_id = None
    for c in df.columns:
        if str(c).lower().startswith("c") or str(c).lower().startswith("id"):
            col_id = c
            break

    # Tentar detectar coluna k
    col_k = None
    for c in df.columns:
        if str(c).lower() == "k":
            col_k = c
            break

    if col_k is None:
        # se não houver coluna k explícita, assumir última coluna
        col_k = df.columns[-1]

    # Passageiros: todas as demais colunas numéricas, exceto col_id e col_k
    cols_pass = [c for c in df.columns if c not in [col_id, col_k]]

    # Ordenar passageiros pela ordem original de aparição
    df_pass = df[cols_pass].copy()
    max_pass = len(cols_pass)

    dados = {
        "serie_id": [],
        "k": [],
    }
    for j in range(1, max_pass + 1):
        dados[f"n{j}"] = []

    for idx, row in df.iterrows():
        # serie_id
        sid = None
        if col_id is not None:
            sid = row[col_id]
        if sid is None or (isinstance(sid, float) and math.isnan(sid)):
            sid = f"C{idx + 1}"

        # k
        try:
            k_val = int(float(row[col_k]))
        except Exception:
            k_val = 0

        # passageiros
        passageiros = []
        for c in cols_pass:
            try:
                passageiros.append(int(float(row[c])))
            except Exception:
                passageiros.append(np.nan)

        dados["serie_id"].append(sid)
        dados["k"].append(k_val)
        for j, val in enumerate(passageiros, start=1):
            dados[f"n{j}"].append(val)

    df_out = pd.DataFrame(dados)
    return df_out


def garantir_serie_id(df: pd.DataFrame) -> pd.DataFrame:
    """Garante coluna 'serie_id' no formato C1..Cn."""
    df = df.copy()
    if "serie_id" not in df.columns:
        df["serie_id"] = [f"C{i}" for i in range(1, len(df) + 1)]
    else:
        df["serie_id"] = df["serie_id"].fillna("").astype(str)
        df.loc[df["serie_id"] == "", "serie_id"] = [
            f"C{i}" for i in range(1, len(df) + 1)
        ]
    return df


def get_df_base() -> Optional[pd.DataFrame]:
    """
    Retorna o DataFrame base a ser usado pelos painéis:
      - df_limpo (se existir)
      - caso contrário, df original.
    """
    df_limpo = st.session_state.get("df_limpo", None)
    if df_limpo is not None and isinstance(df_limpo, pd.DataFrame) and not df_limpo.empty:
        return df_limpo
    df = st.session_state.get("df", None)
    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
        return df
    return None


# ------------------------------------------------------------
# BARRA LATERAL — NAVEGAÇÃO ENTRE PAINÉIS
# ------------------------------------------------------------

st.sidebar.markdown("## 📂 Navegação")

painel = st.sidebar.radio(
    "Escolha o painel:",
    (
        "📥 Histórico — Entrada FLEX ULTRA (V15.4-HÍBRIDO)",
        "🔍 Pipeline V14-FLEX ULTRA (V15.4)",
        "📈 QDS REAL — Qualidade Dinâmica da Estrada (V15.4)",
        "🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15.4)",
        "💡 Replay LIGHT",
        "📅 Replay ULTRA",
        "🎯 Replay ULTRA Unitário",
        "🚨 Monitor de Risco (k & k*)",
        "📊 Ruído Condicional (V15.4)",
        "🧹 Tratamento de Ruído Tipo A+B (V15.4)",
        "📉 Backtest REAL por QDS (V15.4)",
        "🎲 Monte Carlo REAL por QDS (V15.4)",
        "📊 Expectativa de Acertos por Ambiente (V15.4)",
        "🧪 Testes de Confiabilidade REAL",
    ),
)


# ------------------------------------------------------------
# RESUMO GLOBAL DO HISTÓRICO (TOPO DOS PAINÉIS)
# ------------------------------------------------------------

df_base_tmp = get_df_base()
if df_base_tmp is not None and not df_base_tmp.empty:
    cols_pass_tmp = [c for c in df_base_tmp.columns if c.startswith("n")]
    n_series_tmp = len(df_base_tmp)
    max_pass_tmp = len(cols_pass_tmp)
    if "k" in df_base_tmp.columns:
        k_vals_tmp = df_base_tmp["k"].astype(int)
        k_zero = int((k_vals_tmp == 0).sum())
        k_pos = int((k_vals_tmp > 0).sum())
        resumo_k = f"k = 0 em {k_zero} séries | k > 0 em {k_pos} séries"
    else:
        resumo_k = "Coluna 'k' não encontrada."

    st.markdown(
        f"""
**📊 Resumo rápido do histórico (base atual):**  
Séries: **{n_series_tmp}** | Passageiros por série (máx): **{max_pass_tmp}** | {resumo_k}
"""
    )

    if st.session_state.get("df_limpo", None) is not None:
        st.success("✔ Histórico pós-tratamento de ruído (Tipo A) em uso como estrada principal.")
    else:
        st.info("ℹ Usando histórico original (sem df_limpo ativo).")


# ============================================================
# PAINEL 1 — 📥 Histórico — Entrada FLEX ULTRA (V15.4-HÍBRIDO)
# ============================================================

if painel == "📥 Histórico — Entrada FLEX ULTRA (V15.4-HÍBRIDO)":

    st.markdown("## 📥 Histórico — Entrada FLEX ULTRA (V15.4-HÍBRIDO)")
    st.markdown(
        "Entrada híbrida, mantendo o jeitão completo:\n\n"
        "- **Upload de arquivo CSV** (histórico completo);\n"
        "- **Entrada por texto** (copiar/colar), aceitando formato flexível (C1; n1;..;k);\n"
        "- Número variável de passageiros (n1..nN) detectado automaticamente;\n"
        "- `serie_id` garantido (C1..Cn);\n"
        "- Coluna `k` sempre presente.\n\n"
        "Nada simplificado: o motor V15.4 usa exatamente o mesmo padrão estrutural "
        "que as versões V15.2/V15.3, com df original + df_limpo."
    )

    st.markdown("### 📂 Escolha a forma de entrada do histórico")

    modo_entrada = st.radio(
        "Modo de entrada do histórico:",
        ("Arquivo CSV", "Texto (copiar/colar)"),
        key="modo_entrada_v154",
    )

    df_result = None

    if modo_entrada == "Arquivo CSV":
        st.markdown("### 📁 Upload de arquivo CSV")
        arquivo = st.file_uploader("Selecione o arquivo de histórico (.csv):", type=["csv"])
        if arquivo is not None:
            if st.button("Carregar histórico a partir do CSV", type="primary", key="btn_load_csv_v154"):
                try:
                    df_result = carregar_csv_flex(arquivo)
                    df_result = garantir_serie_id(df_result)

                    st.session_state["df"] = df_result
                    st.session_state["df_limpo"] = None
                    st.session_state["ruido_stats"] = None
                    st.session_state["qds_stats"] = None
                    st.session_state["qds_config"] = None
                    st.session_state["df_qds"] = None
                    st.session_state["df_backtest"] = None
                    st.session_state["backtest_stats"] = None
                    st.session_state["mc_stats"] = None
                    st.session_state["df_mc"] = None

                    st.success("Histórico carregado com sucesso a partir do CSV.")
                except Exception as e:
                    st.error(f"Erro ao carregar CSV: {e}")

    else:
        st.markdown("### 📝 Entrada por texto (copiar/colar) — FLEX ULTRA")
        st.markdown(
            "Cole abaixo o histórico no formato flexível, por exemplo:\n\n"
            "`C1;41;5;4;52;30;33;0`\n"
            "`C2;9;39;37;49;43;41;1`\n\n"
            "Ou apenas os passageiros + k:\n\n"
            "`41;5;4;52;30;33;0`"
        )

        texto = st.text_area(
            "Cole o histórico aqui:",
            height=250,
            key="texto_hist_v154",
        )

        if st.button("Carregar histórico a partir do texto", type="primary", key="btn_load_text_v154"):
            try:
                linhas = texto.splitlines()
                df_result = construir_df_a_partir_de_linhas(linhas)
                df_result = garantir_serie_id(df_result)

                st.session_state["df"] = df_result
                st.session_state["df_limpo"] = None
                st.session_state["ruido_stats"] = None
                st.session_state["qds_stats"] = None
                st.session_state["qds_config"] = None
                st.session_state["df_qds"] = None
                st.session_state["df_backtest"] = None
                st.session_state["backtest_stats"] = None
                st.session_state["mc_stats"] = None
                st.session_state["df_mc"] = None

                st.success("Histórico carregado com sucesso a partir do texto.")
            except Exception as e:
                st.error(f"Erro ao processar o texto: {e}")

    # --------------------------------------------------------
    # Resumo do histórico carregado
    # --------------------------------------------------------
    df_loaded = st.session_state.get("df", None)
    if df_loaded is not None and isinstance(df_loaded, pd.DataFrame) and not df_loaded.empty:
        cols_pass = [c for c in df_loaded.columns if c.startswith("n")]
        n_series = len(df_loaded)
        max_pass = len(cols_pass)
        st.markdown("### 📊 Resumo do histórico carregado")
        if "k" in df_loaded.columns:
            k_vals = df_loaded["k"].astype(int)
            k_zero = int((k_vals == 0).sum())
            k_pos = int((k_vals > 0).sum())
            st.write(
                {
                    "Número total de séries": n_series,
                    "Passageiros por série (máximo detectado)": max_pass,
                    "k = 0 (contagem)": k_zero,
                    "k > 0 (contagem)": k_pos,
                }
            )
        else:
            st.write(
                {
                    "Número total de séries": n_series,
                    "Passageiros por série (máximo detectado)": max_pass,
                    "k": "coluna 'k' ausente",
                }
            )

        with st.expander("Prévia das primeiras 30 séries", expanded=False):
            st.dataframe(df_loaded.head(30))

        st.info(
            "A partir daqui, todos os painéis (Pipeline, QDS REAL, TURBO++ ULTRA, "
            "Replay, Ruído, Backtest, Monte Carlo etc.) usarão este histórico como base. "
            "Se você aplicar o Tratamento de Ruído Tipo A, um df_limpo será criado e "
            "passará a ser a estrada principal."
        )
# ============================================================
# PARTE 2/4 — PIPELINE V14-FLEX ULTRA (S1–S5), CLIMA, k*, QDS NÚCLEO
# ============================================================

# ------------------------------------------------------------
# FUNÇÕES BÁSICAS DE SÉRIE / PASSAGEIROS
# ------------------------------------------------------------

def extrair_colunas_passageiros(df: pd.DataFrame) -> List[str]:
    """Retorna a lista de colunas n1..nN presentes no DataFrame."""
    return [c for c in df.columns if c.startswith("n")]


def normalizar_serie(serie: List[int]) -> List[int]:
    """Normaliza uma série para lista de int (removendo NaN, cast seguro)."""
    out = []
    for v in serie:
        try:
            if pd.isna(v):
                continue
            out.append(int(v))
        except Exception:
            continue
    return out


# ------------------------------------------------------------
# BARÔMETRO LOCAL / CLIMA / k* (REGIME DA ESTRADA)
# ------------------------------------------------------------

def calcular_barometro_local(df: pd.DataFrame, janela: int = 40) -> dict:
    """
    Calcula barômetro da estrada numa janela final.

    - média da diferença absoluta média entre séries consecutivas (nos passageiros)
    - percentual de k>0 na janela
    """
    df = df.copy()
    cols_pass = extrair_colunas_passageiros(df)
    if not cols_pass or len(df) < 2:
        return {
            "media_diferenca": 0.0,
            "pct_k_positivo": 0.0,
        }

    df_janela = df.tail(janela).copy()
    matriz = df_janela[cols_pass].astype(float).to_numpy()
    diffs = np.abs(np.diff(matriz, axis=0)).mean(axis=1)
    media_dif = float(np.mean(diffs))

    if "k" in df_janela.columns:
        k_vals = df_janela["k"].astype(int)
        pct_k_pos = 100.0 * float((k_vals > 0).mean())
    else:
        pct_k_pos = 0.0

    return {
        "media_diferenca": media_dif,
        "pct_k_positivo": pct_k_pos,
    }


def classificar_clima(bar: dict) -> str:
    """Classifica o clima da estrada em função do barômetro."""
    media_dif = float(bar.get("media_diferenca", 0.0))
    pct_k_pos = float(bar.get("pct_k_positivo", 0.0))

    # Heurística estável (mesma ideia de V15.2/V15.3)
    if media_dif <= 8.0 and pct_k_pos <= 10.0:
        return "🟢 Estrada estável — poucas variações bruscas."
    elif media_dif <= 12.0 and pct_k_pos <= 25.0:
        return "🟡 Estrada em atenção — variações moderadas e alguns guardas acertando."
    else:
        return "🔴 Estrada turbulenta — muitas variações e/ou muitos guardas acertando."


def classificar_k_ambiente(df: pd.DataFrame, janela: int = 40) -> Tuple[str, str]:
    """
    Produz o estado k* (estável / atenção / crítico) com mensagem de contexto.
    Baseia-se na densidade de k>0 na janela final.
    """
    if "k" not in df.columns or len(df) < 2:
        return "estavel", "🟢 k*: Ambiente estável — sem leituras suficientes."

    df_janela = df.tail(janela).copy()
    k_vals = df_janela["k"].astype(int)
    pct_pos = 100.0 * float((k_vals > 0).mean())

    if pct_pos <= 10.0:
        estado = "estavel"
        msg = "🟢 k*: Ambiente estável — regime normal."
    elif pct_pos <= 25.0:
        estado = "atencao"
        msg = "🟡 k*: Pré-ruptura residual — usar previsões com atenção."
    else:
        estado = "critico"
        msg = "🔴 k*: Ambiente crítico — usar previsões com cautela máxima."

    return estado, msg


def detectar_regime(df: pd.DataFrame) -> Tuple[str, str, dict, Tuple[str, str]]:
    """
    Junta barômetro + k* num único pacote:
      - clima textual
      - estado k* (estavel/atencao/critico)
      - barômetro detalhado
      - tupla (k_estado, mensagem k*)
    """
    bar = calcular_barometro_local(df)
    clima = classificar_clima(bar)
    k_estado, k_msg = classificar_k_ambiente(df)
    return clima, k_estado, bar, (k_estado, k_msg)


# ------------------------------------------------------------
# S1–S5 (GERAÇÃO DE LEQUES BÁSICOS) — V14-FLEX ULTRA
# ------------------------------------------------------------

def s1_vizinhanca_simples(df: pd.DataFrame, n_out: int = 80) -> List[List[int]]:
    """
    S1 — Vizinhança simples da última série:
      - gera séries perturbando pouco os passageiros da última série.
    """
    cols_pass = extrair_colunas_passageiros(df)
    if not cols_pass:
        return []

    ultima = df[cols_pass].iloc[-1].astype(int).to_numpy()
    base = ultima.copy()
    out = []

    # pequenas variações em torno dos passageiros
    for delta in [-2, -1, 0, 1, 2]:
        for i in range(len(base)):
            v = base.copy()
            v[i] = max(0, v[i] + delta)
            out.append(normalizar_serie(v))

    # complemento pseudo-aleatório (determinístico via np.random default)
    while len(out) < n_out:
        v = base.copy()
        idx = np.random.randint(0, len(v))
        passo = np.random.choice([-3, -2, -1, 0, 1, 2, 3])
        v[idx] = max(0, v[idx] + passo)
        out.append(normalizar_serie(v))

    return out[:n_out]


def s2_referencia_historica(df: pd.DataFrame, n_out: int = 80) -> List[List[int]]:
    """
    S2 — Referência histórica:
      - seleciona séries históricas recentes como base.
    """
    cols_pass = extrair_colunas_passageiros(df)
    if not cols_pass:
        return []

    n_total = len(df)
    janela = min(200, n_total)
    df_jan = df.tail(janela)
    # escolhe algumas séries espaçadas
    idxs = np.linspace(0, len(df_jan) - 1, min(n_out, len(df_jan)), dtype=int)
    out = []
    for ix in idxs:
        v = df_jan[cols_pass].iloc[ix].tolist()
        out.append(normalizar_serie(v))
    return out[:n_out]


def s3_mediana_local(df: pd.DataFrame, n_out: int = 80, janela: int = 30) -> List[List[int]]:
    """
    S3 — Mediana local:
      - usa a mediana recente como pivô, com pequenas variações.
    """
    cols_pass = extrair_colunas_passageiros(df)
    if not cols_pass:
        return []

    df_jan = df.tail(janela)
    mediana = df_jan[cols_pass].median().astype(float).to_numpy()
    base = mediana.copy()
    out = []

    for delta in [-1, 0, 1]:
        for i in range(len(base)):
            v = base.copy()
            v[i] = max(0, v[i] + delta)
            out.append(normalizar_serie(v))

    while len(out) < n_out:
        v = base.copy()
        idx = np.random.randint(0, len(v))
        passo = np.random.choice([-2, -1, 0, 1, 2])
        v[idx] = max(0, v[idx] + passo)
        out.append(normalizar_serie(v))

    return out[:n_out]


def s4_mix_historico(df: pd.DataFrame, n_out: int = 80) -> List[List[int]]:
    """
    S4 — Mix histórico:
      - constrói novas séries combinando passageiros de diferentes séries.
    """
    cols_pass = extrair_colunas_passageiros(df)
    if not cols_pass or len(df) < 2:
        return []

    matriz = df[cols_pass].astype(int).to_numpy()
    n_series, n_pass = matriz.shape
    out = []

    for _ in range(n_out):
        nova = []
        for j in range(n_pass):
            idx = np.random.randint(0, n_series)
            nova.append(int(matriz[idx, j]))
        out.append(normalizar_serie(nova))

    return out


def s5_estrutura_risco(df: pd.DataFrame, n_out: int = 80) -> List[List[int]]:
    """
    S5 — Estrutura de risco:
      - prioriza números que aparecem com mais frequência recente,
        montando séries com base em distribuição empírica por posição.
    """
    cols_pass = extrair_colunas_passageiros(df)
    if not cols_pass:
        return []

    df_rec = df.tail(300).copy()
    out = []

    # histogramas por posição
    histos = {}
    for c in cols_pass:
        valores = df_rec[c].dropna().astype(int)
        if valores.empty:
            histos[c] = [0]
        else:
            contagem = valores.value_counts().sort_index()
            histos[c] = contagem.index.tolist()

    for _ in range(n_out):
        nova = []
        for c in cols_pass:
            possiveis = histos.get(c, [0])
            if possiveis:
                nova.append(int(np.random.choice(possiveis)))
            else:
                nova.append(0)
        out.append(normalizar_serie(nova))

    return out


# ------------------------------------------------------------
# LEQUE ORIGINAL / CORRIGIDO / MIX (NÚCLEO V14-FLEX ULTRA)
# ------------------------------------------------------------

def gerar_series_base(df: pd.DataFrame, regime_state: str, n_out: int = 200) -> List[List[int]]:
    """
    Gera leque ORIGINAL (S1–S5 reunidos).

    regime_state (estavel/atencao/critico) pode modular n_out,
    mas aqui mantemos a mesma lógica robusta do V15.x.
    """
    n_out_local = int(n_out)

    s1 = s1_vizinhanca_simples(df, n_out=n_out_local)
    s2 = s2_referencia_historica(df, n_out=n_out_local)
    s3 = s3_mediana_local(df, n_out=n_out_local)
    s4 = s4_mix_historico(df, n_out=n_out_local)
    s5 = s5_estrutura_risco(df, n_out=n_out_local)

    todas = s1 + s2 + s3 + s4 + s5

    # remover duplicadas preservando ordem
    seen = set()
    unicas = []
    for serie in todas:
        t = tuple(serie)
        if t not in seen:
            seen.add(t)
            unicas.append(serie)

    return unicas


def gerar_leque_corrigido(df: pd.DataFrame, regime_state: str, n_out: int = 200) -> List[List[int]]:
    """
    Gera leque CORRIGIDO (ajustes adicionais).
    Pode reforçar algumas estruturas da S3 e S5, imitando S6/S7 estrutural simplificado
    (sem nunca reduzir a complexidade do motor já existente).
    """
    cols_pass = extrair_colunas_passageiros(df)
    if not cols_pass:
        return []

    # base de mediana + mix, com ajustes
    base_med = s3_mediana_local(df, n_out=n_out // 2, janela=50)
    base_mix = s5_estrutura_risco(df, n_out=n_out // 2)

    # Ajustes leves por regime
    fator = 1
    if regime_state == "atencao":
        fator = 2
    elif regime_state == "critico":
        fator = 3

    ajustadas = []
    for serie in base_med + base_mix:
        for _ in range(fator):
            v = serie.copy()
            if len(v) > 0:
                idx = np.random.randint(0, len(v))
                passo = np.random.choice([-1, 0, 1])
                v[idx] = max(0, v[idx] + passo)
            ajustadas.append(normalizar_serie(v))

    # remover duplicadas
    seen = set()
    unicas = []
    for serie in ajustadas:
        t = tuple(serie)
        if t not in seen:
            seen.add(t)
            unicas.append(serie)

    return unicas[:n_out]


def build_flat_series_table(series_list: List[List[int]]) -> pd.DataFrame:
    """
    Constrói DataFrame plano a partir de lista de séries (lista de listas).

    Output:
      - colunas n1..nN
      - coluna 'series' com a lista original
    """
    if not series_list:
        return pd.DataFrame()

    max_len = max(len(s) for s in series_list)
    dados = {f"n{i}": [] for i in range(1, max_len + 1)}
    dados["series"] = []

    for s in series_list:
        s_norm = normalizar_serie(s)
        dados["series"].append(s_norm)
        for i in range(1, max_len + 1):
            if i <= len(s_norm):
                dados[f"n{i}"].append(s_norm[i - 1])
            else:
                dados[f"n{i}"].append(np.nan)

    df_flat = pd.DataFrame(dados)
    return df_flat


# ------------------------------------------------------------
# NÚCLEO QDS REAL (V15.4) — MESMO JEITÃO DO V15.3
# ------------------------------------------------------------

def calcular_qds_estrada(df: pd.DataFrame, window_tam: int = 40) -> Tuple[pd.DataFrame, dict]:
    """
    Calcula QDS REAL ao longo da estrada:

      - para cada índice i, considera janela anterior [i - window_tam + 1 .. i]
      - mede dispersão local (diferença média entre séries consecutivas dentro da janela)
      - mede percentual de k>0 na janela
      - usa heurística para converter em score QDS (0–100)
      - classifica em níveis: PREMIUM / BOM / REGULAR / RUIM

    Retorna:
      - df_qds (indexado pelo índice da série, 1..N)
      - stats agregadas (médias, percentuais por nível)
    """
    if df is None or df.empty:
        raise ValueError("Histórico vazio para cálculo de QDS.")

    df = df.copy()
    df = garantir_serie_id(df)
    cols_pass = extrair_colunas_passageiros(df)
    n_total = len(df)

    registros = []

    for idx in range(1, n_total + 1):
        ini = max(1, idx - window_tam + 1)
        fim = idx

        df_win = df.iloc[ini - 1 : fim].copy()
        if len(df_win) < 2 or not cols_pass:
            qds_score = 0.0
            nivel = "RUIM"
        else:
            matriz = df_win[cols_pass].astype(float).to_numpy()
            diffs = np.abs(np.diff(matriz, axis=0)).mean(axis=1)
            media_dif = float(np.mean(diffs))

            if "k" in df_win.columns:
                k_vals = df_win["k"].astype(int)
                pct_k_pos = 100.0 * float((k_vals > 0).mean())
            else:
                pct_k_pos = 0.0

            # Escalonamento empírico de QDS:
            # - baixa dispersão + k>0 moderado → QDS alto
            # - alta dispersão + k>0 raro → QDS baixo
            # Limitamos dispersão numa faixa típica [0..20]
            disp_norm = max(0.0, min(20.0, media_dif))
            # Quanto menor a dispersão, melhor (inversão)
            comp_disp = 1.0 - disp_norm / 20.0  # 0..1

            # k_norm: queremos penalizar extremos muito altos de k>0 também
            k_norm = max(0.0, min(50.0, pct_k_pos)) / 50.0  # 0..1

            # Combinação linear com pesos
            raw = 0.6 * comp_disp + 0.4 * (1.0 - abs(k_norm - 0.3) / 0.3)
            qds_score = max(0.0, min(1.0, raw)) * 100.0

            if qds_score >= 80.0:
                nivel = "PREMIUM"
            elif qds_score >= 60.0:
                nivel = "BOM"
            elif qds_score >= 40.0:
                nivel = "REGULAR"
            else:
                nivel = "RUIM"

        registros.append(
            {
                "idx": idx,
                "serie_id": df.iloc[idx - 1]["serie_id"],
                "qds_score": qds_score,
                "nivel_qds": nivel,
            }
        )

    df_qds = pd.DataFrame(registros).set_index("idx")

    # Stats agregadas
    qds_vals = df_qds["qds_score"].values
    stats = {
        "window_tam": int(window_tam),
        "qds_media": float(np.mean(qds_vals)),
        "qds_min": float(np.min(qds_vals)),
        "qds_max": float(np.max(qds_vals)),
    }

    niveis = ["PREMIUM", "BOM", "REGULAR", "RUIM"]
    stats_niveis = {}
    for nivel in niveis:
        bloco = df_qds[df_qds["nivel_qds"] == nivel]
        pct = 0.0
        if len(df_qds) > 0:
            pct = 100.0 * len(bloco) / len(df_qds)
        stats_niveis[nivel] = {
            "pontos": int(len(bloco)),
            "pct": pct,
        }

    stats["pct_premium"] = stats_niveis["PREMIUM"]["pct"]
    stats["pct_bom_ou_melhor"] = stats_niveis["PREMIUM"]["pct"] + stats_niveis["BOM"]["pct"]
    stats["por_nivel"] = stats_niveis

    return df_qds, stats


# ============================================================
# PAINEL — 🔍 Pipeline V14-FLEX ULTRA (V15.4)
# ============================================================

if painel == "🔍 Pipeline V14-FLEX ULTRA (V15.4)":

    st.markdown("## 🔍 Pipeline V14-FLEX ULTRA (V15.4)")
    st.markdown(
        "Execução do pipeline base (S1–S5) sobre a estrada atual (df ou df_limpo), "
        "com clima, barômetro e k* integrados.\n\n"
        "Nada simplificado: este painel mostra a visão bruta do núcleo V14-FLEX ULTRA "
        "que serve de base para o TURBO++ ULTRA, QDS REAL, Backtest e Monte Carlo."
    )

    df = get_df_base()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel de Entrada FLEX ULTRA.")
        st.stop()

    clima, k_estado, bar, (k_st, k_msg) = detectar_regime(df)

    col_esq, col_dir = st.columns(2)

    with col_esq:
        st.markdown("### 🌡️ Clima Local (Barômetro da Estrada)")
        st.info(clima)
        st.markdown("#### 🔍 Estatísticas da janela local")
        st.write(bar)

    with col_dir:
        st.markdown("### ⭐ Estado k* Local")
        st.info(k_msg)

    st.markdown("---")
    st.markdown("### 🛠️ Execução S1–S5 (faixas iniciais)")

    n_total = len(df)
    st.write(f"Histórico atual contém **{n_total} séries**.")

    idx_alvo = st.number_input(
        "Escolha o índice alvo (1 = primeira série carregada):",
        min_value=1,
        max_value=n_total,
        value=n_total,
        step=1,
        key="pipeline_idx_v154",
    )

    n_out_base = st.number_input(
        "Tamanho base de saída de cada módulo (S1–S5):",
        min_value=20,
        max_value=400,
        value=80,
        step=10,
        key="pipeline_nout_v154",
    )

    if st.button("Rodar Pipeline V14-FLEX ULTRA (S1–S5)", type="primary", key="btn_pipeline_v154"):
        df_sub = df.iloc[:idx_alvo].copy()
        clima_local, k_estado_local, bar_local, (k_st_local, k_msg_local) = detectar_regime(df_sub)

        st.markdown("### ℹ️ Contexto do ponto alvo")
        serie_id = df_sub.iloc[-1].get("serie_id", f"C{idx_alvo}")
        st.write(f"ID alvo: **{serie_id}** (índice {idx_alvo})")
        st.info(clima_local)
        st.info(k_msg_local)

        with st.spinner("Gerando leques S1–S5 para o ponto alvo..."):
            s1 = s1_vizinhanca_simples(df_sub, n_out=int(n_out_base))
            s2 = s2_referencia_historica(df_sub, n_out=int(n_out_base))
            s3 = s3_mediana_local(df_sub, n_out=int(n_out_base))
            s4 = s4_mix_historico(df_sub, n_out=int(n_out_base))
            s5 = s5_estrutura_risco(df_sub, n_out=int(n_out_base))

        st.success("Leques S1–S5 gerados com sucesso.")

        st.markdown("### 📊 S1 — Vizinhança simples (amostra)")
        st.dataframe(pd.DataFrame({"series": [normalizar_serie(x) for x in s1[:30]]}))
        st.markdown("### 📊 S2 — Referência histórica (amostra)")
        st.dataframe(pd.DataFrame({"series": [normalizar_serie(x) for x in s2[:30]]}))
        st.markdown("### 📊 S3 — Mediana local (amostra)")
        st.dataframe(pd.DataFrame({"series": [normalizar_serie(x) for x in s3[:30]]}))
        st.markdown("### 📊 S4 — Mix histórico (amostra)")
        st.dataframe(pd.DataFrame({"series": [normalizar_serie(x) for x in s4[:30]]}))
        st.markdown("### 📊 S5 — Estrutura de risco (amostra)")
        st.dataframe(pd.DataFrame({"series": [normalizar_serie(x) for x in s5[:30]]}))

        st.info(
            "O leque ORIGINAL usado pelo TURBO++ ULTRA será a união controlada de "
            "S1–S5, normalizada e transformada em tabela plana. O leque CORRIGIDO "
            "aplicará ajustes adicionais (S6/S7) e o MIX combinará ambos."
        )


# ============================================================
# PAINEL — 📈 QDS REAL — Qualidade Dinâmica da Estrada (V15.4)
# (INTERFACE — NÚCLEO JÁ DEFINIDO ACIMA)
# ============================================================

if painel == "📈 QDS REAL — Qualidade Dinâmica da Estrada (V15.4)":

    st.markdown("## 📈 QDS REAL — Qualidade Dinâmica da Estrada (V15.4)")
    st.markdown(
        "Mede a **Qualidade Dinâmica da Série** ao longo da estrada, combinando:\n\n"
        "- dispersão local (diferença média entre séries consecutivas);\n"
        "- percentual de k>0 na janela;\n"
        "- comportamento do k atual.\n\n"
        "Produz um score QDS (0–100) e classifica trechos como "
        "**PREMIUM / BOM / REGULAR / RUIM**.\n"
        "Este QDS será usado diretamente no **Backtest REAL por QDS (V15.4)** "
        "e no **Monte Carlo REAL por QDS (V15.4)**."
    )

    df_base = get_df_base()
    if df_base is None or df_base.empty:
        st.warning("Carregue o histórico primeiro no painel de Entrada FLEX ULTRA.")
        st.stop()

    st.markdown("### ⚙️ Parâmetros do QDS REAL")
    col_w, col_dummy = st.columns([1, 1])
    with col_w:
        window_tam = st.slider(
            "Tamanho da janela para cálculo local (séries):",
            min_value=10,
            max_value=200,
            value=40,
            step=5,
        )

    if st.button("Calcular QDS REAL da estrada", type="primary", key="btn_qds_real_v154"):
        with st.spinner("Calculando QDS REAL ao longo da estrada..."):
            df_qds, stats = calcular_qds_estrada(df_base, window_tam=int(window_tam))

        st.session_state["qds_stats"] = stats
        st.session_state["qds_config"] = {"window_tam": int(window_tam)}
        st.session_state["df_qds"] = df_qds

        st.success("QDS REAL calculado com sucesso.")

        st.markdown("### 📊 Estatísticas agregadas de QDS")
        st.write(
            {
                "Tamanho da janela": stats["window_tam"],
                "QDS médio": stats["qds_media"],
                "QDS mínimo": stats["qds_min"],
                "QDS máximo": stats["qds_max"],
                "% de trechos PREMIUM": f"{stats['pct_premium']:.2f}%",
                "% de trechos BOM ou melhor": f"{stats['pct_bom_ou_melhor']:.2f}%",
            }
        )

        st.markdown("### 📈 Amostra da curva QDS ao longo da estrada (últimas 200 séries)")
        st.dataframe(df_qds.tail(200))

        with st.expander("Visualização completa do QDS (tabela completa)", expanded=False):
            st.dataframe(df_qds)

        st.info(
            "Trechos **PREMIUM** indicam janelas onde o TURBO++ ULTRA tende a operar "
            "com maior consistência. Trechos **RUIM** indicam ambientes onde a estrada "
            "está frágil, mesmo após o tratamento de ruído. O Backtest REAL e o "
            "Monte Carlo REAL usarão essas faixas para comparar a performance do motor "
            "por ambiente."
        )
    else:
        stats = st.session_state.get("qds_stats", None)
        df_qds = st.session_state.get("df_qds", None)
        if stats is not None and df_qds is not None:
            st.markdown("### 📊 Estatísticas agregadas de QDS (último cálculo)")
            st.write(
                {
                    "Tamanho da janela": stats["window_tam"],
                    "QDS médio": stats["qds_media"],
                    "QDS mínimo": stats["qds_min"],
                    "QDS máximo": stats["qds_max"],
                    "% de trechos PREMIUM": f"{stats['pct_premium']:.2f}%",
                    "% de trechos BOM ou melhor": f"{stats['pct_bom_ou_melhor']:.2f}%",
                }
            )
            st.markdown("### 📈 Amostra da curva QDS (últimas 200 séries)")
            st.dataframe(df_qds.tail(200))
        else:
            st.info(
                "Configure a janela e clique em **'Calcular QDS REAL da estrada'** para "
                "gerar o mapa de qualidade dinâmica. O Backtest REAL e o Monte Carlo REAL "
                "dependem deste cálculo."
            )
# ============================================================
# PARTE 3/4 — TURBO++ ULTRA, RUÍDO A/B, REPLAY, MONITOR, RUÍDO CONDICIONAL
# ============================================================

# ------------------------------------------------------------
# MÉTRICAS DE SÉRIE / TVF / RUÍDO TIPO B
# ------------------------------------------------------------

def calcular_discrepancia_para_ultima(df: pd.DataFrame, flat_df: pd.DataFrame) -> pd.Series:
    """
    Calcula a discrepância (distância) de cada série candidata para a última série da estrada.
    Usado como base para TVF (quanto menor a discrepância, maior a TVF).
    """
    cols_pass = extrair_colunas_passageiros(df)
    if not cols_pass or flat_df.empty:
        return pd.Series([0.0] * len(flat_df), index=flat_df.index)

    ultima = df[cols_pass].iloc[-1].astype(float).to_numpy()

    dists = []
    for _, row in flat_df.iterrows():
        serie_vals = []
        for c in cols_pass:
            serie_vals.append(row.get(c, np.nan))
        serie_arr = np.array(serie_vals, dtype=float)
        mask = ~np.isnan(serie_arr)
        if not mask.any():
            dists.append(0.0)
            continue
        diff = np.abs(ultima[mask] - serie_arr[mask])
        dists.append(float(diff.mean()))
    return pd.Series(dists, index=flat_df.index)


def calcular_ruido_tipo_b_para_series(flat_df: pd.DataFrame) -> pd.Series:
    """
    Calcula um fator de ruído Tipo B por série candidata.
    Ideia: séries com dispersão interna muito alta (por posição) são penalizadas.

    Retorna um fator entre ~0.3 e 1.0, onde:
      - 1.0 ~ série "suave"
      - valores menores ~ série mais ruidosa
    """
    cols_pass = [c for c in flat_df.columns if c.startswith("n")]
    if not cols_pass:
        return pd.Series([1.0] * len(flat_df), index=flat_df.index)

    dispersoes = []
    for _, row in flat_df.iterrows():
        vals = []
        for c in cols_pass:
            v = row.get(c, np.nan)
            if not pd.isna(v):
                vals.append(float(v))
        if len(vals) <= 1:
            dispersoes.append(0.0)
        else:
            vals_arr = np.array(vals)
            dispersoes.append(float(np.std(vals_arr)))

    disp_series = pd.Series(dispersoes, index=flat_df.index)
    if disp_series.max() <= 0:
        return pd.Series([1.0] * len(flat_df), index=flat_df.index)

    # normaliza dispersão para [0..1]
    disp_norm = disp_series / disp_series.max()
    # mapeia para fator de ruído: mais dispersão => fator menor
    ruido_factor = 1.0 - 0.7 * disp_norm  # mínimo ~0.3
    ruido_factor = ruido_factor.clip(lower=0.3, upper=1.0)
    return ruido_factor


def avaliar_series_tvf(
    df: pd.DataFrame,
    flat_df: pd.DataFrame,
    regime_state: str,
) -> pd.DataFrame:
    """
    Avalia cada série candidata, produzindo:

      - discrepância para a última série da estrada
      - TVF_base (0–100) relativa
      - conf_pct (similar à TVF_base)
      - ruido_factor (Tipo B)
      - tvf_ajustada (TVF_base * ruido_factor)

    Mantém o mesmo jeitão V15.x: melhor TVF_ajustada no topo.
    """
    if flat_df is None or flat_df.empty:
        return flat_df

    flat = flat_df.copy()

    # Discrepância (distância média absoluta)
    discrep = calcular_discrepancia_para_ultima(df, flat)
    flat["discrepancia"] = discrep

    if discrep.max() > 0:
        # menor discrepância => maior TVF
        tvf_base = (1.0 - discrep / discrep.max()) * 100.0
    else:
        tvf_base = pd.Series([100.0] * len(flat), index=flat.index)

    flat["tvf_base"] = tvf_base
    flat["conf_pct"] = tvf_base  # jeitão: usar TVF como confiança básica

    # Ruído Tipo B
    ruido_factor = calcular_ruido_tipo_b_para_series(flat)
    flat["ruido_factor"] = ruido_factor

    # TVF ajustada pelo ruído
    flat["tvf_ajustada"] = flat["tvf_base"] * flat["ruido_factor"]

    # Ordenar por tvf_ajustada desc
    flat = flat.sort_values(by="tvf_ajustada", ascending=False).reset_index(drop=True)

    return flat


# ------------------------------------------------------------
# LIMITAÇÃO POR MODO DE SAÍDA (Automático / Quantidade fixa / Confiabilidade mínima)
# ------------------------------------------------------------

def limit_by_mode(
    flat_df: pd.DataFrame,
    regime_state: str,
    output_mode: str,
    n_series_fixed: int,
    min_conf_pct: float,
) -> pd.DataFrame:
    """
    Aplica modo de saída:

      - 'Automático (por regime)': ajusta total de séries conforme regime_state;
      - 'Quantidade fixa': usa n_series_fixed;
      - 'Confiabilidade mínima': filtra por conf_pct >= min_conf_pct.
    """
    if flat_df is None or flat_df.empty:
        return flat_df

    df_out = flat_df.copy()

    if output_mode == "Automático (por regime)":
        # Heurística por regime
        if regime_state == "estavel":
            n_final = min(25, len(df_out))
        elif regime_state == "atencao":
            n_final = min(35, len(df_out))
        else:
            n_final = min(50, len(df_out))
        df_out = df_out.head(n_final)

    elif output_mode == "Quantidade fixa":
        n_final = max(1, min(int(n_series_fixed), len(df_out)))
        df_out = df_out.head(n_final)

    elif output_mode == "Confiabilidade mínima":
        df_out = df_out[df_out["conf_pct"] >= float(min_conf_pct)].copy()
        if df_out.empty:
            # fallback: ao menos uma série
            df_out = flat_df.head(1).copy()

    df_out = df_out.reset_index(drop=True)
    return df_out


# ------------------------------------------------------------
# LEQUE ORIGINAL / CORRIGIDO / MISTO PARA TURBO++ ULTRA
# ------------------------------------------------------------

def montar_leques_turbo(
    df: pd.DataFrame,
    regime_state: str,
    n_out_base: int = 200,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Monta:

      - df_original: leque ORIGINAL (S1–S5) em tabela plana;
      - df_corrigido: leque CORRIGIDO;
      - df_mix: união ORIGINAL+CORRIGIDO, com marcação de origem.
    """
    leque_original = gerar_series_base(df, regime_state, n_out=n_out_base)
    leque_corrigido = gerar_leque_corrigido(df, regime_state, n_out=n_out_base)

    df_orig = build_flat_series_table(leque_original)
    df_corr = build_flat_series_table(leque_corrigido)

    if not df_orig.empty:
        df_orig["origem"] = "ORIGINAL"
    if not df_corr.empty:
        df_corr["origem"] = "CORRIGIDO"

    # MIX: união dos dois
    df_mix = pd.concat([df_orig, df_corr], ignore_index=True)

    # Remover séries duplicadas em 'series'
    if "series" in df_mix.columns:
        df_mix = df_mix.drop_duplicates(subset=["series"], keep="first").reset_index(drop=True)

    return df_orig, df_corr, df_mix


# ------------------------------------------------------------
# TURBO++ ULTRA — NÚCLEO (MONTAR PREVISÃO)
# ------------------------------------------------------------

def montar_previsao_turbo_ultra(
    df: pd.DataFrame,
    idx_alvo: int,
    output_mode: str,
    n_series_fixed: int,
    min_conf_pct: float,
) -> Tuple[pd.DataFrame, Optional[List[int]], dict]:
    """
    Núcleo TURBO++ ULTRA V15.4:

      - usa estrada até idx_alvo;
      - detecta regime (clima + k*);
      - monta leques ORIGINAL + CORRIGIDO + MIX;
      - avalia tvf + ruído Tipo B;
      - aplica limit_by_mode;
      - escolhe melhor série como previsão final.

    Retorna:
      - df_controlado (leque final),
      - previsao_final (lista de ints) ou None,
      - contexto (dict com clima, k*, barômetro, etc.).
    """
    if df is None or df.empty:
        return pd.DataFrame(), None, {}

    idx_alvo = int(idx_alvo)
    if idx_alvo < 1 or idx_alvo > len(df):
        idx_alvo = len(df)

    df_sub = df.iloc[:idx_alvo].copy()
    clima, k_estado, bar, (k_st, k_msg) = detectar_regime(df_sub)

    regime_state = k_estado  # usamos o k* como regime base

    df_orig, df_corr, df_mix = montar_leques_turbo(
        df_sub,
        regime_state=regime_state,
        n_out_base=200,
    )

    if df_mix.empty:
        return df_mix, None, {
            "clima": clima,
            "k_msg": k_msg,
            "barometro": bar,
            "regime_state": regime_state,
        }

    df_avaliado = avaliar_series_tvf(df_sub, df_mix, regime_state=regime_state)

    df_controlado = limit_by_mode(
        df_avaliado,
        regime_state=regime_state,
        output_mode=output_mode,
        n_series_fixed=n_series_fixed,
        min_conf_pct=min_conf_pct,
    )

    previsao_final = None
    ruido_fator_final = None
    origem_final = None
    if not df_controlado.empty:
        melhor = df_controlado.iloc[0]
        previsao_final = normalizar_serie(melhor["series"])
        ruido_fator_final = float(melhor.get("ruido_factor", 1.0))
        origem_final = str(melhor.get("origem", "MISTO"))

    contexto = {
        "clima": clima,
        "k_msg": k_msg,
        "barometro": bar,
        "regime_state": regime_state,
        "ruido_fator_final": ruido_fator_final,
        "origem_final": origem_final,
    }

    return df_controlado, previsao_final, contexto


# ============================================================
# PAINEL — 🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15.4)
# ============================================================

if painel == "🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15.4)":

    st.markdown("## 🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15.4)")
    st.markdown(
        "Núcleo V14-FLEX ULTRA + Leque ORIGINAL/CORRIGIDO/MISTO + TVF + k* adaptativo + Ruído Tipo B.\n\n"
        "Este é o motor que será usado pelo **Backtest REAL por QDS** e pelo "
        "**Monte Carlo REAL por QDS (V15.4)**."
    )

    df = get_df_base()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel de Entrada FLEX ULTRA.")
        st.stop()

    clima, k_estado, bar, (k_st, k_msg) = detectar_regime(df)

    st.markdown("### 🌡️ Clima da Estrada (base atual)")
    st.info(clima)

    st.markdown("### ⭐ k* — Sentinela do Ambiente")
    st.info(k_msg)

    st.markdown("### ⚙️ Controles do Leque TURBO++ ULTRA")

    n_total = len(df)
    idx_alvo = st.number_input(
        "Índice alvo (1 = primeira série carregada):",
        min_value=1,
        max_value=n_total,
        value=n_total,
        step=1,
        key="turbo_idx_v154",
    )

    output_mode = st.radio(
        "Modo de geração do Leque:",
        ("Automático (por regime)", "Quantidade fixa", "Confiabilidade mínima"),
        key="turbo_mode_v154",
    )

    n_series_fixed = st.slider(
        "Quantidade total de séries (se modo for 'Quantidade fixa')",
        min_value=5,
        max_value=100,
        value=25,
        step=5,
        key="turbo_nfixed_v154",
    )

    min_conf_pct = st.slider(
        "Confiabilidade mínima (%) (se modo for 'Confiabilidade mínima')",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
        key="turbo_minconf_v154",
    )

    if st.button("Gerar Leque TURBO++ ULTRA e Previsão", type="primary", key="btn_turbo_v154"):
        with st.spinner("Gerando leque TURBO++ ULTRA..."):
            df_controlado, previsao_final, contexto = montar_previsao_turbo_ultra(
                df,
                idx_alvo=idx_alvo,
                output_mode=output_mode,
                n_series_fixed=int(n_series_fixed),
                min_conf_pct=float(min_conf_pct),
            )

        if df_controlado is None or df_controlado.empty:
            st.error("Não foi possível gerar o leque TURBO++ ULTRA para este ponto.")
            st.stop()

        st.success(f"Leque TURBO++ ULTRA gerado com sucesso: {len(df_controlado)} séries após controle.")

        st.markdown("### 📊 Leque TURBO++ ULTRA — Séries Candidatas Controladas")
        st.dataframe(
            df_controlado[
                [c for c in df_controlado.columns if c.startswith("n")]
                + ["origem", "discrepancia", "tvf_base", "conf_pct", "ruido_factor", "tvf_ajustada"]
            ].head(100)
        )

        st.markdown("### 🎯 Previsão Final TURBO++ ULTRA")
        if previsao_final:
            st.code(" ".join(str(x) for x in previsao_final), language="text")
        else:
            st.warning("Nenhuma previsão final foi selecionada (leque vazio).")

        k_msg_local = contexto.get("k_msg", "k* não calculado.")
        st.info(k_msg_local)

        origem_final = contexto.get("origem_final", "MISTO")
        ruido_fator_final = contexto.get("ruido_fator_final", None)
        tvf_top = float(df_controlado.iloc[0]["tvf_ajustada"])

        if ruido_fator_final is not None:
            st.markdown(
                f"Origem = **{origem_final}**, TVF ≈ **{tvf_top:.2f}**, "
                f"Ruído fator ≈ **{ruido_fator_final:.2f}**."
            )
        else:
            st.markdown(f"Origem = **{origem_final}**, TVF ≈ **{tvf_top:.2f}**.")


# ============================================================
# PAINEL — 💡 Replay LIGHT
# ============================================================

if painel == "💡 Replay LIGHT":

    st.markdown("## 💡 Replay LIGHT")
    st.markdown(
        "Simula o que o **TURBO++ ULTRA** teria feito em um ponto específico do histórico, "
        "usando o mesmo motor (TVF + ruído Tipo B + k* adaptativo)."
    )

    df = get_df_base()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel de Entrada FLEX ULTRA.")
        st.stop()

    n_total = len(df)
    st.write(f"Histórico atual contém **{n_total} séries**.")

    idx_alvo = st.number_input(
        "Escolha o índice alvo (1 = primeira série carregada):",
        min_value=1,
        max_value=n_total,
        value=n_total,
        step=1,
        key="replay_light_idx_v154",
    )

    output_mode = st.radio(
        "Modo de geração do Leque (para o Replay LIGHT):",
        ("Automático (por regime)", "Quantidade fixa", "Confiabilidade mínima"),
        key="replay_light_mode_v154",
    )

    n_series_fixed = st.slider(
        "Quantidade total de séries (se modo for 'Quantidade fixa')",
        min_value=5,
        max_value=100,
        value=25,
        step=5,
        key="replay_light_nfixed_v154",
    )

    min_conf_pct = st.slider(
        "Confiabilidade mínima (%) (se modo for 'Confiabilidade mínima')",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
        key="replay_light_minconf_v154",
    )

    if st.button("Rodar Replay LIGHT para esse ponto", type="primary", key="btn_replay_light_v154"):
        df_sub = df.iloc[:idx_alvo].copy()
        clima_local, k_estado_local, bar_local, (k_st_local, k_msg_local) = detectar_regime(df_sub)

        st.markdown("### ℹ️ Contexto do ponto alvo (base atual)")
        serie_id = df_sub.iloc[-1].get("serie_id", f"C{idx_alvo}")
        st.write(f"ID alvo: **{serie_id}** (índice {idx_alvo})")
        st.info(clima_local)
        st.info(k_msg_local)

        with st.spinner("Gerando Replay LIGHT (TURBO++ ULTRA) para o ponto alvo..."):
            df_controlado, previsao_final, contexto = montar_previsao_turbo_ultra(
                df,
                idx_alvo=idx_alvo,
                output_mode=output_mode,
                n_series_fixed=int(n_series_fixed),
                min_conf_pct=float(min_conf_pct),
            )

        if df_controlado is None or df_controlado.empty:
            st.error("Não foi possível gerar o Replay LIGHT para este ponto.")
            st.stop()

        st.success(f"Replay LIGHT gerado com sucesso: {len(df_controlado)} séries no leque controlado.")

        st.markdown("### 📊 Leque resultante do Replay LIGHT (top 30)")
        st.dataframe(
            df_controlado[
                [c for c in df_controlado.columns if c.startswith("n")]
                + ["origem", "discrepancia", "tvf_base", "conf_pct", "ruido_factor", "tvf_ajustada"]
            ].head(30)
        )

        st.markdown("### 🎯 Previsão que teria sido feita nesse ponto")
        if previsao_final:
            st.code(" ".join(str(x) for x in previsao_final), language="text")
        else:
            st.warning("Nenhuma previsão final foi selecionada (leque vazio).")

        k_msg_out = contexto.get("k_msg", k_msg_local)
        st.info(k_msg_out)

        origem_final = contexto.get("origem_final", "MISTO")
        ruido_fator_final = contexto.get("ruido_fator_final", None)
        tvf_top = float(df_controlado.iloc[0]["tvf_ajustada"])

        if ruido_fator_final is not None:
            st.markdown(
                f"Origem = **{origem_final}**, TVF ≈ **{tvf_top:.2f}**, "
                f"Ruído fator ≈ **{ruido_fator_final:.2f}**."
            )
        else:
            st.markdown(f"Origem = **{origem_final}**, TVF ≈ **{tvf_top:.2f}**.")


# ============================================================
# PAINEL — 📅 Replay ULTRA
# ============================================================

if painel == "📅 Replay ULTRA":

    st.markdown("## 📅 Replay ULTRA")
    st.markdown(
        "Simulação mais pesada: percorre um intervalo de índices, rodando o mesmo "
        "motor TURBO++ ULTRA em cada ponto, para enxergar o comportamento do leque "
        "ao longo da estrada."
    )

    df = get_df_base()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel de Entrada FLEX ULTRA.")
        st.stop()

    n_total = len(df)
    st.write(f"Histórico atual contém **{n_total} séries**.")

    idx_ini = st.number_input(
        "Índice inicial do Replay ULTRA:",
        min_value=1,
        max_value=n_total,
        value=max(1, n_total - 50),
        step=1,
        key="replay_ultra_ini_v154",
    )

    idx_fim = st.number_input(
        "Índice final do Replay ULTRA:",
        min_value=1,
        max_value=n_total,
        value=n_total,
        step=1,
        key="replay_ultra_fim_v154",
    )

    if idx_fim < idx_ini:
        idx_fim = idx_ini

    output_mode = st.radio(
        "Modo de geração do Leque (Replay ULTRA):",
        ("Automático (por regime)", "Quantidade fixa", "Confiabilidade mínima"),
        key="replay_ultra_mode_v154",
    )

    n_series_fixed = st.slider(
        "Quantidade total de séries (se modo for 'Quantidade fixa')",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        key="replay_ultra_nfixed_v154",
    )

    min_conf_pct = st.slider(
        "Confiabilidade mínima (%) (se modo for 'Confiabilidade mínima')",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
        key="replay_ultra_minconf_v154",
    )

    if st.button("Rodar Replay ULTRA no intervalo", type="primary", key="btn_replay_ultra_v154"):
        registros = []
        with st.spinner("Executando Replay ULTRA ao longo do intervalo..."):
            for idx in range(int(idx_ini), int(idx_fim) + 1):
                df_controlado, previsao_final, contexto = montar_previsao_turbo_ultra(
                    df,
                    idx_alvo=idx,
                    output_mode=output_mode,
                    n_series_fixed=int(n_series_fixed),
                    min_conf_pct=float(min_conf_pct),
                )

                if previsao_final is None:
                    previsao_final = []

                registros.append(
                    {
                        "idx": idx,
                        "serie_id": df.iloc[idx - 1].get("serie_id", f"C{idx}"),
                        "previsao": previsao_final,
                        "origem": contexto.get("origem_final", "MISTO"),
                        "ruido_factor": contexto.get("ruido_fator_final", None),
                    }
                )

        df_replay = pd.DataFrame(registros).set_index("idx")
        st.success("Replay ULTRA concluído.")
        st.dataframe(df_replay.head(200))

        st.info(
            "O Replay ULTRA permite enxergar a consistência do motor ponto a ponto, "
            "antes de rodar Backtests e Monte Carlo. Você pode exportar esta tabela "
            "para análises adicionais."
        )


# ============================================================
# PAINEL — 🎯 Replay ULTRA Unitário
# ============================================================

if painel == "🎯 Replay ULTRA Unitário":

    st.markdown("## 🎯 Replay ULTRA Unitário")
    st.markdown(
        "Análise detalhada de um único ponto do histórico com foco máximo no contexto local "
        "e no leque TURBO++ ULTRA desse ponto."
    )

    df = get_df_base()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel de Entrada FLEX ULTRA.")
        st.stop()

    n_total = len(df)
    st.write(f"Histórico atual contém **{n_total} séries**.")

    idx_alvo = st.number_input(
        "Escolha o índice alvo para análise ULTRA:",
        min_value=1,
        max_value=n_total,
        value=n_total,
        step=1,
        key="replay_ultra_unit_idx_v154",
    )

    output_mode = st.radio(
        "Modo de geração do Leque (para este ponto ULTRA):",
        ("Automático (por regime)", "Quantidade fixa", "Confiabilidade mínima"),
        key="replay_ultra_unit_mode_v154",
    )

    n_series_fixed = st.slider(
        "Quantidade total de séries (se modo for 'Quantidade fixa')",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        key="replay_ultra_unit_nfixed_v154",
    )

    min_conf_pct = st.slider(
        "Confiabilidade mínima (%) (se modo for 'Confiabilidade mínima')",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
        key="replay_ultra_unit_minconf_v154",
    )

    if st.button("Rodar Replay ULTRA Unitário", type="primary", key="btn_replay_ultra_unit_v154"):
        df_sub = df.iloc[:idx_alvo].copy()
        clima_local, k_estado_local, bar_local, (k_st_local, k_msg_local) = detectar_regime(df_sub)

        st.markdown("### ℹ️ Contexto completo do ponto ULTRA")
        serie_id = df_sub.iloc[-1].get("serie_id", f"C{idx_alvo}")
        st.write(f"ID alvo: **{serie_id}** (índice {idx_alvo})")
        st.info(clima_local)
        st.info(k_msg_local)

        st.markdown("#### Barômetro local:")
        st.write(bar_local)

        with st.spinner("Gerando Leque TURBO++ ULTRA para o ponto ULTRA..."):
            df_controlado, previsao_final, contexto = montar_previsao_turbo_ultra(
                df,
                idx_alvo=idx_alvo,
                output_mode=output_mode,
                n_series_fixed=int(n_series_fixed),
                min_conf_pct=float(min_conf_pct),
            )

        if df_controlado is None or df_controlado.empty:
            st.error("Não foi possível gerar o leque ULTRA para este ponto.")
            st.stop()

        st.success(f"Leque TURBO++ ULTRA gerado para o ponto ULTRA: {len(df_controlado)} séries.")

        st.markdown("### 📊 Leque ULTRA (top 40)")
        st.dataframe(
            df_controlado[
                [c for c in df_controlado.columns if c.startswith("n")]
                + ["origem", "discrepancia", "tvf_base", "conf_pct", "ruido_factor", "tvf_ajustada"]
            ].head(40)
        )

        st.markdown("### 🎯 Previsão ULTRA para este ponto")
        if previsao_final:
            st.code(" ".join(str(x) for x in previsao_final), language="text")
        else:
            st.warning("Nenhuma previsão final foi selecionada (leque vazio).")

        k_msg_out = contexto.get("k_msg", k_msg_local)
        st.info(k_msg_out)

        origem_final = contexto.get("origem_final", "MISTO")
        ruido_fator_final = contexto.get("ruido_fator_final", None)
        tvf_top = float(df_controlado.iloc[0]["tvf_ajustada"])

        if ruido_fator_final is not None:
            st.markdown(
                f"Origem = **{origem_final}**, TVF ≈ **{tvf_top:.2f}**, "
                f"Ruído fator ≈ **{ruido_fator_final:.2f}**."
            )
        else:
            st.markdown(f"Origem = **{origem_final}**, TVF ≈ **{tvf_top:.2f}**.")


# ============================================================
# PAINEL — 🚨 Monitor de Risco (k & k*)
# ============================================================

if painel == "🚨 Monitor de Risco (k & k*)":

    st.markdown("## 🚨 Monitor de Risco (k & k*)")
    st.markdown(
        "Painel dedicado a enxergar a estrada pela lente do **k** e do **k***, "
        "com foco em rupturas, pré-rupturas e regimes estáveis."
    )

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico original no painel de Entrada FLEX ULTRA.")
        st.stop()

    clima, k_estado, bar, (k_st, k_msg) = detectar_regime(df)

    st.markdown("### 🌡️ Clima atual da estrada (base atual)")
    st.info(clima)

    st.markdown("### ⭐ Sentinela k* (estado atual)")
    st.info(k_msg)

    st.markdown("### 📊 Barômetro resumido")
    st.write(bar)

    if "k" in df.columns:
        k_vals = df["k"].astype(int)
        stats_k = {
            "k mínimo": int(k_vals.min()),
            "k máximo": int(k_vals.max()),
            "k médio": float(k_vals.mean()),
        }
        st.markdown("### Estatísticas básicas de k (histórico original)")
        st.write(stats_k)

        # Histograma simples de k
        with st.expander("📈 Distribuição de k no histórico original (tabela)", expanded=False):
            dist_k = k_vals.value_counts().sort_index()
            st.dataframe(dist_k)
    else:
        st.warning("Histórico original não possui coluna 'k' explícita.")


# ============================================================
# PAINEL — 📊 Ruído Condicional (V15.4)
# ============================================================

if painel == "📊 Ruído Condicional (V15.4)":

    st.markdown("## 📊 Ruído Condicional (V15.4)")
    st.markdown(
        "Monitor para enxergar como a estrada reage a diferentes regimes, "
        "abrindo espaço para filtros anti-ruído condicionais ao ambiente.\n\n"
        "Agora integrado ao contexto de **QDS REAL** (V15.4) e ao motor TURBO++ ULTRA."
    )

    df = get_df_base()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel de Entrada FLEX ULTRA.")
        st.stop()

    clima, k_estado, bar, (k_st, k_msg) = detectar_regime(df)

    st.markdown("### 🌡️ Clima e k* (base atual)")
    st.info(clima)
    st.info(k_msg)

    if "k" in df.columns:
        k_vals = df["k"].astype(int)
        pct_zero = 100.0 * float((k_vals == 0).mean())
        pct_pos = 100.0 * float((k_vals > 0).mean())
    else:
        pct_zero = pct_pos = 0.0

    st.markdown("### 🔎 Indicadores simples de ruído (versão inicial)")
    st.write(
        {
            "Séries sem acerto (k = 0)": f"{pct_zero:.1f}%",
            "Séries com acerto (k > 0)": f"{pct_pos:.1f}%",
        }
    )

    stats_qds = st.session_state.get("qds_stats", None)
    if stats_qds is not None:
        st.markdown("### 📈 QDS REAL como filtro condicional de ruído")
        st.write(
            {
                "QDS médio": stats_qds["qds_media"],
                "% PREMIUM": f"{stats_qds['pct_premium']:.2f}%",
                "% BOM ou melhor": f"{stats_qds['pct_bom_ou_melhor']:.2f}%",
            }
        )
        st.info(
            "Trechos com **QDS alto** e **k*** estável tendem a ser regiões com ruído efetivo mais "
            "controlado, ideais para estratégias mais agressivas (como modos de 5/6 acertos)."
        )
    else:
        st.info(
            "Calcule o **QDS REAL** no painel correspondente para habilitar a visão "
            "condicional de ruído por ambiente."
        )


# ============================================================
# TRATAMENTO DE RUÍDO TIPO A (LIMPEZA / SUAVIZAÇÃO) + PAINEL
# ============================================================

def tratar_ruido_tipo_a(
    df: pd.DataFrame,
    window: int = 7,
    limiar_sigma: float = 3.0,
) -> Tuple[pd.DataFrame, dict]:
    """
    Tratamento de Ruído Tipo A:

      - Calcula dispersão média entre séries (diferença média entre vizinhos);
      - Aplica suavização em pontos considerados outliers (baseados em MAD);
      - Retorna df_limpo + estatísticas antes/depois.

    Não simplifica o histórico: mantém mesma estrutura (serie_id, n1..nN, k).
    """
    if df is None or df.empty:
        raise ValueError("Histórico vazio para tratamento de ruído Tipo A.")

    df = df.copy()
    df = garantir_serie_id(df)
    cols_pass = extrair_colunas_passageiros(df)
    if not cols_pass:
        raise ValueError("Histórico sem colunas de passageiros para tratamento de ruído.")

    matriz = df[cols_pass].astype(float).to_numpy()
    n, m = matriz.shape

    # Dispersão antes (diferença média entre séries consecutivas)
    if n > 1:
        diffs_before = np.abs(np.diff(matriz, axis=0)).mean(axis=1)
        disp_before = float(diffs_before.mean())
    else:
        diffs_before = np.array([0.0])
        disp_before = 0.0

    # Cálculo de MAD (desvio absoluto mediano) na vizinhança por posição
    matriz_limpa = matriz.copy()
    window = max(3, int(window))
    half = window // 2

    ajustes = 0
    for i in range(n):
        i_ini = max(0, i - half)
        i_fim = min(n, i + half + 1)
        bloco = matriz[i_ini:i_fim, :]

        med = np.nanmedian(bloco, axis=0)
        abs_dev = np.abs(bloco - med)
        mad = np.nanmedian(abs_dev, axis=0)
        sigma_est = 1.4826 * mad  # aproximação robusta

        for j in range(m):
            val = matriz[i, j]
            mu = med[j]
            sig = sigma_est[j]
            if np.isnan(val) or np.isnan(mu) or np.isnan(sig) or sig == 0:
                continue
            z = np.abs(val - mu) / sig
            if z > limiar_sigma:
                matriz_limpa[i, j] = mu
                ajustes += 1

    # Dispersão depois
    if n > 1:
        diffs_after = np.abs(np.diff(matriz_limpa, axis=0)).mean(axis=1)
        disp_after = float(diffs_after.mean())
    else:
        diffs_after = np.array([0.0])
        disp_after = 0.0

    df_limpo = df.copy()
    for idx_c, c in enumerate(cols_pass):
        df_limpo[c] = matriz_limpa[:, idx_c]

    total_pontos = n * m
    pct_ajustado = 100.0 * ajustes / total_pontos if total_pontos > 0 else 0.0

    stats = {
        "Window": int(window),
        "Limiar sigma": float(limiar_sigma),
        "Dispersão média (antes)": float(disp_before),
        "Dispersão média (depois)": float(disp_after),
        "% de pontos ajustados (n1..nN)": f"{pct_ajustado:.3f}%",
    }

    return df_limpo, stats


if painel == "🧹 Tratamento de Ruído Tipo A+B (V15.4)":

    st.markdown("## 🧹 Tratamento de Ruído Tipo A+B (V15.4)")
    st.markdown(
        "Tipo A: limpeza/suavização do histórico (**df_limpo**).  \n"
        "Tipo B: penalização de séries ruidosas no TURBO++ (**ruido_factor** já integrado ao TVF).\n\n"
        "V15.4: este painel também alimenta o contexto do **QDS REAL**, do **Backtest REAL** "
        "e do **Monte Carlo REAL por QDS**."
    )

    df_orig = st.session_state.get("df", None)
    if df_orig is None or df_orig.empty:
        st.warning("Carregue o histórico original primeiro no painel de Entrada FLEX ULTRA.")
        st.stop()

    cols_pass = extrair_colunas_passageiros(df_orig)
    if not cols_pass:
        st.warning("Histórico original não possui colunas de passageiros identificadas (n1..nN).")
        st.stop()

    # Dispersão original
    matriz = df_orig[cols_pass].astype(float).to_numpy()
    if len(matriz) > 1:
        diffs_orig = np.abs(np.diff(matriz, axis=0)).mean(axis=1)
        disp_orig = float(diffs_orig.mean())
    else:
        disp_orig = 0.0

    st.markdown("### 🔎 Situação atual do ruído (histórico original)")
    st.write({"Dispersão média entre séries (original)": disp_orig})

    st.markdown("### ⚙️ Parâmetros do Tratamento de Ruído Tipo A")
    col_w, col_s = st.columns(2)
    with col_w:
        window = st.slider(
            "Janela (tamanho da vizinhança)",
            min_value=3,
            max_value=31,
            value=7,
            step=2,
            key="ruidoA_window_v154",
        )
    with col_s:
        limiar_sigma = st.slider(
            "Limiar de ruído (multiplicador de MAD)",
            min_value=2.0,
            max_value=6.0,
            value=3.0,
            step=0.5,
            key="ruidoA_sigma_v154",
        )

    if st.button("Aplicar Tratamento de Ruído Tipo A", type="primary", key="btn_ruidoA_v154"):
        with st.spinner("Aplicando tratamento de ruído Tipo A (limpeza/suavização)..."):
            df_limpo, stats_ruido = tratar_ruido_tipo_a(
                df_orig,
                window=int(window),
                limiar_sigma=float(limiar_sigma),
            )

        st.session_state["df_limpo"] = df_limpo
        st.session_state["ruido_stats"] = stats_ruido

        st.success("Tratamento de Ruído Tipo A aplicado com sucesso.")

        st.markdown("### 📊 Métricas antes/depois")
        st.write(stats_ruido)

        st.markdown("### Prévia do histórico pós-tratamento (df_limpo)")
        st.dataframe(df_limpo.head(30))

        st.info(
            "A partir de agora, todos os painéis que usam o histórico base "
            "(Pipeline, TURBO, Replay, QDS, Backtest, Monte Carlo etc.) "
            "passarão a usar **df_limpo** como estrada principal."
        )
    else:
        stats_ruido = st.session_state.get("ruido_stats", None)
        if stats_ruido is not None:
            st.markdown("### 📊 Métricas antes/depois (último tratamento aplicado)")
            st.write(stats_ruido)
            df_limpo_prev = st.session_state.get("df_limpo", None)
            if df_limpo_prev is not None:
                st.markdown("### Prévia do histórico pós-tratamento (df_limpo)")
                st.dataframe(df_limpo_prev.head(30))
            st.success("✅ Tratamento ativo — df_limpo está em uso como estrada principal.")
        else:
            st.info(
                "Configure os parâmetros e clique em **'Aplicar Tratamento de Ruído Tipo A'** "
                "para gerar o histórico suavizado (df_limpo). O Ruído Tipo B já está integrado "
                "ao motor TURBO++ (ruido_factor nas séries candidatas)."
            )
# ============================================================
# PARTE 4/4 — BACKTEST REAL por QDS, MONTE CARLO REAL, EXPECTATIVA, CONFIABILIDADE
# ============================================================

# ------------------------------------------------------------
# BACKTEST REAL por QDS (NÚCLEO)
# ------------------------------------------------------------

def rodar_backtest_real_por_qds(
    df_base: pd.DataFrame,
    df_qds: pd.DataFrame,
    idx_ini: int,
    idx_fim: int,
    output_mode: str,
    n_series_fixed: int,
    min_conf_pct: float,
) -> Tuple[pd.DataFrame, dict]:
    """
    Backtest REAL por QDS (V15.4):

      - percorre um intervalo [idx_ini .. idx_fim] da estrada;
      - para cada índice i, usa o histórico até i-1 para prever a série i;
      - compara a previsão com a verdade real (série i do histórico);
      - calcula:
          • acertos por passageiro;
          • acerto total (série exatamente igual);
          • nível de QDS naquele ponto (PREMIUM/BOM/REGULAR/RUIM);
      - agrega estatísticas globais e por nível de QDS.

    Usa o mesmo motor TURBO++ ULTRA V15.4.
    """
    df_base = df_base.copy()
    df_qds = df_qds.copy()

    df_base = garantir_serie_id(df_base)
    cols_pass = extrair_colunas_passageiros(df_base)
    if not cols_pass:
        raise ValueError("Histórico base sem colunas de passageiros (n1..nN) para backtest.")

    n_total = len(df_base)
    idx_ini = max(1, int(idx_ini))
    idx_fim = min(n_total, int(idx_fim))
    if idx_fim <= idx_ini:
        idx_fim = idx_ini

    registros = []
    total_pontos = 0
    soma_acertos = 0
    soma_acertos_total = 0

    # estrutura por nível de QDS
    niveis_qds = ["PREMIUM", "BOM", "REGULAR", "RUIM", "SEM_QDS"]
    agg_nivel = {nivel: {"pontos": 0, "soma_acertos": 0.0, "soma_total": 0} for nivel in niveis_qds}

    for idx in range(idx_ini, idx_fim + 1):
        # precisamos ter pelo menos 1 série antes para servir de histórico
        if idx <= 1:
            continue

        # histórico até i-1
        idx_hist = idx - 1
        df_hist = df_base.iloc[:idx_hist].copy()

        # previsão para a série idx (target)
        df_controlado, previsao_final, contexto = montar_previsao_turbo_ultra(
            df_base,
            idx_alvo=idx_hist,
            output_mode=output_mode,
            n_series_fixed=int(n_series_fixed),
            min_conf_pct=float(min_conf_pct),
        )

        # série real (idx)
        row_real = df_base.iloc[idx - 1]
        real_vals = [int(row_real[c]) for c in cols_pass]

        if previsao_final is None:
            previsao_final = []

        n_comp = min(len(real_vals), len(previsao_final))
        acertos_pass = 0
        for j in range(n_comp):
            if previsao_final[j] == real_vals[j]:
                acertos_pass += 1

        acerto_total = 1 if acertos_pass == n_comp and n_comp > 0 else 0

        # nível de QDS no ponto idx
        if idx in df_qds.index:
            nivel_qds = df_qds.loc[idx, "nivel_qds"]
            if nivel_qds not in niveis_qds:
                nivel_qds = "SEM_QDS"
        else:
            nivel_qds = "SEM_QDS"

        qds_val = float(df_qds.loc[idx, "qds_score"]) if idx in df_qds.index else None

        total_pontos += 1
        soma_acertos += acertos_pass
        soma_acertos_total += acerto_total

        agg = agg_nivel[nivel_qds]
        agg["pontos"] += 1
        agg["soma_acertos"] += acertos_pass
        agg["soma_total"] += acerto_total

        registros.append(
            {
                "idx": idx,
                "serie_id": row_real.get("serie_id", f"C{idx}"),
                "nivel_qds": nivel_qds,
                "qds_score": qds_val,
                "previsao": previsao_final,
                "real": real_vals,
                "acertos_passageiros": acertos_pass,
                "acerto_total": acerto_total,
            }
        )

    if total_pontos == 0:
        raise ValueError("Nenhum ponto válido para backtest no intervalo especificado.")

    media_acertos = soma_acertos / total_pontos
    taxa_total = 100.0 * soma_acertos_total / total_pontos

    stats_globais = {
        "Intervalo analisado": [idx_ini, idx_fim],
        "Total de pontos": total_pontos,
        "Média de acertos por série (passageiros)": media_acertos,
        "Taxa de acerto total (série exata)": f"{taxa_total:.2f}%",
    }

    stats_por_nivel = {}
    for nivel in niveis_qds:
        agg = agg_nivel[nivel]
        if agg["pontos"] > 0:
            media_nivel = agg["soma_acertos"] / agg["pontos"]
            taxa_nivel = 100.0 * agg["soma_total"] / agg["pontos"]
        else:
            media_nivel = None
            taxa_nivel = None
        stats_por_nivel[nivel] = {
            "pontos": agg["pontos"],
            "media_acertos_pass": media_nivel,
            "taxa_acerto_total": taxa_nivel,
        }

    stats_globais["por_nivel_qds"] = stats_por_nivel

    df_backtest = pd.DataFrame(registros).set_index("idx")
    return df_backtest, stats_globais


# ============================================================
# PAINEL — 📉 Backtest REAL por QDS (V15.4)
# ============================================================

if painel == "📉 Backtest REAL por QDS (V15.4)":

    st.markdown("## 📉 Backtest REAL por QDS (V15.4)")
    st.markdown(
        "Aqui o Predict Cars V15.4 simula o que teria feito em cada ponto da estrada, "
        "usando o motor **TURBO++ ULTRA (V15.4)** e comparando com a verdade real do histórico.\n\n"
        "O resultado é segmentado por nível de **QDS** (PREMIUM / BOM / REGULAR / RUIM), "
        "para mapear a performance do motor por ambiente."
    )

    df_base = get_df_base()
    if df_base is None or df_base.empty:
        st.warning("Carregue o histórico primeiro no painel de Entrada FLEX ULTRA.")
        st.stop()

    df_qds = st.session_state.get("df_qds", None)
    if df_qds is None or df_qds.empty:
        st.warning("Calcule primeiro o **QDS REAL** no painel correspondente antes de rodar o Backtest REAL.")
        st.stop()

    n_total = len(df_base)
    st.write(f"Histórico atual contém **{n_total} séries**.")

    idx_ini = st.number_input(
        "Índice inicial do backtest:",
        min_value=1,
        max_value=n_total,
        value=max(2, n_total - 200),
        step=1,
        key="bt_qds_ini_v154",
    )

    idx_fim = st.number_input(
        "Índice final do backtest:",
        min_value=1,
        max_value=n_total,
        value=n_total,
        step=1,
        key="bt_qds_fim_v154",
    )

    if idx_fim < idx_ini:
        idx_fim = idx_ini

    st.markdown("### ⚙️ Configuração do motor durante o Backtest")

    output_mode = st.radio(
        "Modo de geração do Leque (durante o backtest):",
        ("Automático (por regime)", "Quantidade fixa", "Confiabilidade mínima"),
        key="bt_qds_mode_v154",
    )

    n_series_fixed = st.slider(
        "Quantidade total de séries (se modo for 'Quantidade fixa')",
        min_value=5,
        max_value=100,
        value=25,
        step=5,
        key="bt_qds_nfixed_v154",
    )

    min_conf_pct = st.slider(
        "Confiabilidade mínima (%) (se modo for 'Confiabilidade mínima')",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
        key="bt_qds_minconf_v154",
    )

    if st.button("Rodar Backtest REAL por trecho de QDS", type="primary", key="btn_bt_qds_v154"):
        with st.spinner("Executando Backtest REAL por QDS ao longo do intervalo..."):
            df_backtest, stats_bt = rodar_backtest_real_por_qds(
                df_base=df_base,
                df_qds=df_qds,
                idx_ini=int(idx_ini),
                idx_fim=int(idx_fim),
                output_mode=output_mode,
                n_series_fixed=int(n_series_fixed),
                min_conf_pct=float(min_conf_pct),
            )

        st.session_state["df_backtest"] = df_backtest
        st.session_state["backtest_stats"] = stats_bt

        st.success("Backtest REAL por trecho de QDS concluído.")

        st.markdown("### 📊 Estatísticas globais do Backtest")
        st.write(
            {
                "Intervalo analisado": stats_bt["Intervalo analisado"],
                "Total de pontos": stats_bt["Total de pontos"],
                "Média de acertos por série (passageiros)": stats_bt["Média de acertos por série (passageiros)"],
                "Taxa de acerto total (série exata)": stats_bt["Taxa de acerto total (série exata)"],
            }
        )

        st.markdown("### 📊 Estatísticas por nível de QDS")
        stats_nivel = stats_bt["por_nivel_qds"]
        st.write(stats_nivel)

        with st.expander("📋 Tabela detalhada do Backtest (amostra)", expanded=False):
            st.dataframe(df_backtest.head(200))

        st.info(
            "Use este painel para entender **onde** o motor acerta mais: "
            "se em ambientes **PREMIUM**, **BOM**, **REGULAR** ou **RUIM**. "
            "Isso alimenta diretamente os painéis de **Expectativa de Acertos** e **Monte Carlo REAL por QDS**."
        )
    else:
        stats_bt = st.session_state.get("backtest_stats", None)
        df_backtest_prev = st.session_state.get("df_backtest", None)
        if stats_bt is not None and df_backtest_prev is not None:
            st.markdown("### 📊 Estatísticas globais do Backtest (última execução)")
            st.write(
                {
                    "Intervalo analisado": stats_bt["Intervalo analisado"],
                    "Total de pontos": stats_bt["Total de pontos"],
                    "Média de acertos por série (passageiros)": stats_bt["Média de acertos por série (passageiros)"],
                    "Taxa de acerto total (série exata)": stats_bt["Taxa de acerto total (série exata)"],
                }
            )
            st.markdown("### 📊 Estatísticas por nível de QDS (última execução)")
            st.write(stats_bt["por_nivel_qds"])
        else:
            st.info(
                "Defina o intervalo, configure o motor e clique em "
                "**'Rodar Backtest REAL por trecho de QDS'** para gerar as métricas."
            )


# ------------------------------------------------------------
# MONTE CARLO REAL por QDS (NÚCLEO)
# ------------------------------------------------------------

def rodar_monte_carlo_real_por_qds(
    df_base: pd.DataFrame,
    df_qds: pd.DataFrame,
    idx_ini: int,
    idx_fim: int,
    n_sims: int,
    output_mode: str,
    n_series_fixed: int,
    min_conf_pct: float,
) -> Tuple[pd.DataFrame, dict]:
    """
    Monte Carlo REAL por QDS (V15.4):

      - sorteia pontos da estrada dentro de [idx_ini .. idx_fim];
      - em cada sorteio:
          • usa histórico até i-1 para prever a série i;
          • compara com a verdade real da série i;
          • registra nível de QDS, acertos, acerto total;
      - repete o processo n_sims vezes, com aleatoriedade proveniente do próprio motor
        (S4, S5, leque corrigido, etc.);
      - agrega estatísticas por nível de QDS.

    Complementa o Backtest REAL, adicionando distribuição ao redor da performance por ambiente.
    """
    df_base = df_base.copy()
    df_qds = df_qds.copy()
    df_base = garantir_serie_id(df_base)

    cols_pass = extrair_colunas_passageiros(df_base)
    if not cols_pass:
        raise ValueError("Histórico base sem colunas de passageiros (n1..nN) para Monte Carlo.")

    n_total = len(df_base)
    idx_ini = max(1, int(idx_ini))
    idx_fim = min(n_total, int(idx_fim))
    if idx_fim <= idx_ini:
        idx_fim = idx_ini

    indices_validos = [i for i in range(idx_ini, idx_fim + 1) if i > 1]
    indices_validos = [i for i in indices_validos if i in df_qds.index]

    if not indices_validos:
        raise ValueError("Nenhum índice válido (com QDS) disponível no intervalo para Monte Carlo.")

    niveis_qds = ["PREMIUM", "BOM", "REGULAR", "RUIM", "SEM_QDS"]
    agg_nivel = {nivel: {"pontos": 0, "soma_acertos": 0.0, "soma_total": 0} for nivel in niveis_qds}

    registros = []

    n_sims = int(n_sims)
    for sim in range(n_sims):
        idx = int(np.random.choice(indices_validos))
        if idx <= 1:
            continue

        idx_hist = idx - 1
        df_hist = df_base.iloc[:idx_hist].copy()

        df_controlado, previsao_final, contexto = montar_previsao_turbo_ultra(
            df_base,
            idx_alvo=idx_hist,
            output_mode=output_mode,
            n_series_fixed=int(n_series_fixed),
            min_conf_pct=float(min_conf_pct),
        )

        row_real = df_base.iloc[idx - 1]
        real_vals = [int(row_real[c]) for c in cols_pass]

        if previsao_final is None:
            previsao_final = []

        n_comp = min(len(real_vals), len(previsao_final))
        acertos_pass = 0
        for j in range(n_comp):
            if previsao_final[j] == real_vals[j]:
                acertos_pass += 1

        acerto_total = 1 if acertos_pass == n_comp and n_comp > 0 else 0

        if idx in df_qds.index:
            nivel_qds = df_qds.loc[idx, "nivel_qds"]
            if nivel_qds not in niveis_qds:
                nivel_qds = "SEM_QDS"
        else:
            nivel_qds = "SEM_QDS"

        qds_val = float(df_qds.loc[idx, "qds_score"]) if idx in df_qds.index else None

        agg = agg_nivel[nivel_qds]
        agg["pontos"] += 1
        agg["soma_acertos"] += acertos_pass
        agg["soma_total"] += acerto_total

        registros.append(
            {
                "sim": sim + 1,
                "idx": idx,
                "serie_id": row_real.get("serie_id", f"C{idx}"),
                "nivel_qds": nivel_qds,
                "qds_score": qds_val,
                "previsao": previsao_final,
                "real": real_vals,
                "acertos_passageiros": acertos_pass,
                "acerto_total": acerto_total,
            }
        )

    total_sims = len(registros)
    if total_sims == 0:
        raise ValueError("Nenhuma simulação válida foi registrada no Monte Carlo.")

    # Agregados por nível
    stats_por_nivel = {}
    soma_acertos_global = 0.0
    soma_total_global = 0

    for nivel, agg in agg_nivel.items():
        if agg["pontos"] > 0:
            media_nivel = agg["soma_acertos"] / agg["pontos"]
            taxa_nivel = 100.0 * agg["soma_total"] / agg["pontos"]
        else:
            media_nivel = None
            taxa_nivel = None

        stats_por_nivel[nivel] = {
            "pontos": agg["pontos"],
            "media_acertos_pass": media_nivel,
            "taxa_acerto_total": taxa_nivel,
        }

        soma_acertos_global += agg["soma_acertos"]
        soma_total_global += agg["soma_total"]

    media_global = soma_acertos_global / total_sims
    taxa_global = 100.0 * soma_total_global / total_sims

    stats_globais = {
        "Intervalo amostrado": [idx_ini, idx_fim],
        "Total de simulações": total_sims,
        "Média de acertos por série (passageiros)": media_global,
        "Taxa de acerto total (série exata)": f"{taxa_global:.2f}%",
        "por_nivel_qds": stats_por_nivel,
    }

    df_mc = pd.DataFrame(registros)
    return df_mc, stats_globais


# ============================================================
# PAINEL — 🎲 Monte Carlo REAL por QDS (V15.4)
# ============================================================

if painel == "🎲 Monte Carlo REAL por QDS (V15.4)":

    st.markdown("## 🎲 Monte Carlo REAL por QDS (V15.4)")
    st.markdown(
        "Este painel roda um **Monte Carlo REAL** usando o motor **TURBO++ ULTRA (V15.4)**, "
        "segmentado por nível de **QDS**.\n\n"
        "Ao invés de percorrer todos os pontos de forma determinística (como no Backtest REAL), "
        "aqui sorteamos pontos dentro de um intervalo e deixamos o motor agir com sua aleatoriedade "
        "interna (S4, S5, leque corrigido etc.), medindo:\n\n"
        "- média de acertos por série (por ambiente);\n"
        "- taxa de acerto total da série por ambiente (PREMIUM/BOM/REGULAR/RUIM)."
    )

    df_base = get_df_base()
    if df_base is None or df_base.empty:
        st.warning("Carregue o histórico primeiro no painel de Entrada FLEX ULTRA.")
        st.stop()

    df_qds = st.session_state.get("df_qds", None)
    if df_qds is None or df_qds.empty:
        st.warning("Calcule primeiro o **QDS REAL** no painel correspondente antes de rodar o Monte Carlo REAL.")
        st.stop()

    n_total = len(df_base)
    st.write(f"Histórico atual contém **{n_total} séries**.")

    idx_ini = st.number_input(
        "Índice inicial para amostragem (Monte Carlo):",
        min_value=1,
        max_value=n_total,
        value=max(2, n_total - 400),
        step=1,
        key="mc_qds_ini_v154",
    )

    idx_fim = st.number_input(
        "Índice final para amostragem (Monte Carlo):",
        min_value=1,
        max_value=n_total,
        value=n_total,
        step=1,
        key="mc_qds_fim_v154",
    )

    if idx_fim < idx_ini:
        idx_fim = idx_ini

    n_sims = st.slider(
        "Número de simulações (amostras Monte Carlo):",
        min_value=50,
        max_value=1000,
        value=300,
        step=50,
        key="mc_qds_nsims_v154",
    )

    st.markdown("### ⚙️ Configuração do motor durante o Monte Carlo")

    output_mode = st.radio(
        "Modo de geração do Leque (Monte Carlo):",
        ("Automático (por regime)", "Quantidade fixa", "Confiabilidade mínima"),
        key="mc_qds_mode_v154",
    )

    n_series_fixed = st.slider(
        "Quantidade total de séries (se modo for 'Quantidade fixa')",
        min_value=5,
        max_value=100,
        value=25,
        step=5,
        key="mc_qds_nfixed_v154",
    )

    min_conf_pct = st.slider(
        "Confiabilidade mínima (%) (se modo for 'Confiabilidade mínima')",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
        key="mc_qds_minconf_v154",
    )

    if st.button("Rodar Monte Carlo REAL por QDS", type="primary", key="btn_mc_qds_v154"):
        with st.spinner("Executando Monte Carlo REAL por QDS..."):
            df_mc, stats_mc = rodar_monte_carlo_real_por_qds(
                df_base=df_base,
                df_qds=df_qds,
                idx_ini=int(idx_ini),
                idx_fim=int(idx_fim),
                n_sims=int(n_sims),
                output_mode=output_mode,
                n_series_fixed=int(n_series_fixed),
                min_conf_pct=float(min_conf_pct),
            )

        st.session_state["df_mc"] = df_mc
        st.session_state["mc_stats"] = stats_mc

        st.success("Monte Carlo REAL por QDS concluído.")

        st.markdown("### 📊 Estatísticas globais do Monte Carlo")
        st.write(
            {
                "Intervalo amostrado": stats_mc["Intervalo amostrado"],
                "Total de simulações": stats_mc["Total de simulações"],
                "Média de acertos por série (passageiros)": stats_mc["Média de acertos por série (passageiros)"],
                "Taxa de acerto total (série exata)": stats_mc["Taxa de acerto total (série exata)"],
            }
        )

        st.markdown("### 📊 Estatísticas por nível de QDS (Monte Carlo)")
        st.write(stats_mc["por_nivel_qds"])

        with st.expander("📋 Tabela detalhada de simulações (amostra)", expanded=False):
            st.dataframe(df_mc.head(300))

        st.info(
            "O Monte Carlo REAL por QDS mostra como o motor se comporta **em média** "
            "em cada ambiente, levando em conta a aleatoriedade do próprio leque "
            "(principalmente S4/S5 e leque CORRIGIDO)."
        )
    else:
        stats_mc = st.session_state.get("mc_stats", None)
        df_mc_prev = st.session_state.get("df_mc", None)
        if stats_mc is not None and df_mc_prev is not None:
            st.markdown("### 📊 Estatísticas globais do Monte Carlo (última execução)")
            st.write(
                {
                    "Intervalo amostrado": stats_mc["Intervalo amostrado"],
                    "Total de simulações": stats_mc["Total de simulações"],
                    "Média de acertos por série (passageiros)": stats_mc["Média de acertos por série (passageiros)"],
                    "Taxa de acerto total (série exata)": stats_mc["Taxa de acerto total (série exata)"],
                }
            )
            st.markdown("### 📊 Estatísticas por nível de QDS (última execução)")
            st.write(stats_mc["por_nivel_qds"])
        else:
            st.info(
                "Defina o intervalo, o número de simulações e clique em "
                "**'Rodar Monte Carlo REAL por QDS'** para gerar as métricas."
            )


# ============================================================
# PAINEL — 📊 Expectativa de Acertos por Ambiente (V15.4)
# ============================================================

if painel == "📊 Expectativa de Acertos por Ambiente (V15.4)":

    st.markdown("## 📊 Expectativa de Acertos por Ambiente (V15.4)")
    st.markdown(
        "Este painel consolida o **Backtest REAL por QDS** e o **Monte Carlo REAL por QDS**, "
        "mostrando a expectativa de acertos por ambiente:\n\n"
        "- PREMIUM\n- BOM\n- REGULAR\n- RUIM\n- SEM_QDS (casos não classificados)\n\n"
        "A ideia é responder com números:\n\n"
        "- Quantos passageiros, em média, o motor acerta em cada ambiente?\n"
        "- Qual a probabilidade de acertar a série inteira em cada ambiente?\n"
        "- Como o comportamento esperado muda quando consideramos a aleatoriedade (Monte Carlo)?"
    )

    bt_stats = st.session_state.get("backtest_stats", None)
    mc_stats = st.session_state.get("mc_stats", None)

    if bt_stats is None or "por_nivel_qds" not in bt_stats:
        st.warning("Backtest REAL por QDS ainda não foi executado. Rode o painel correspondente primeiro.")
        st.stop()

    if mc_stats is None or "por_nivel_qds" not in mc_stats:
        st.warning("Monte Carlo REAL por QDS ainda não foi executado. Rode o painel correspondente primeiro.")
        st.stop()

    niveis = ["PREMIUM", "BOM", "REGULAR", "RUIM", "SEM_QDS"]
    linhas = []
    for nivel in niveis:
        bt_n = bt_stats["por_nivel_qds"].get(nivel, {})
        mc_n = mc_stats["por_nivel_qds"].get(nivel, {})

        linhas.append(
            {
                "Nível QDS": nivel,
                "Backtest - pontos": bt_n.get("pontos", 0),
                "Backtest - média acertos/pass": bt_n.get("media_acertos_pass", None),
                "Backtest - taxa série exata (%)": bt_n.get("taxa_acerto_total", None),
                "Monte Carlo - pontos": mc_n.get("pontos", 0),
                "Monte Carlo - média acertos/pass": mc_n.get("media_acertos_pass", None),
                "Monte Carlo - taxa série exata (%)": mc_n.get("taxa_acerto_total", None),
            }
        )

    df_expect = pd.DataFrame(linhas)

    st.markdown("### 📈 Tabela consolidada de expectativa por ambiente")
    st.dataframe(df_expect)

    st.info(
        "Interpretação sugerida:\n\n"
        "- Ambientes com **média alta de acertos/pass** e **taxa de série exata maior** "
        "tendem a ser melhores candidatos para estratégias mais agressivas;\n"
        "- Ambientes com média baixa e taxa de série exata próxima de zero indicam "
        "zonas de alta turbulência, mesmo após tratamento de ruído e QDS;\n"
        "- Compare Backtest (determinístico) com Monte Carlo (aleatório) para entender "
        "se o motor é estável ou se oscila demais em cada nível de QDS."
    )


# ============================================================
# PAINEL — 🧪 Testes de Confiabilidade REAL
# ============================================================

if painel == "🧪 Testes de Confiabilidade REAL":

    st.markdown("## 🧪 Testes de Confiabilidade REAL (V15.4)")
    st.markdown(
        "Painel conceitual que consolida os efeitos de:\n\n"
        "- Tratamento de Ruído Tipo A (df_limpo);\n"
        "- Ruído Tipo B (penalização no TURBO++);\n"
        "- QDS REAL (mapa de qualidade da estrada);\n"
        "- Backtest REAL por QDS (V15.4);\n"
        "- Monte Carlo REAL por QDS (V15.4).\n\n"
        "Estrutura mantida no jeitão, pronta para receber no futuro:\n"
        "- Monte Carlo profundo por regime;\n"
        "- Estratégias explícitas de ataque/defesa por ambiente;\n"
        "- Painéis dedicados de expectativa de acertos por modo de operação."
    )

    ruido_stats = st.session_state.get("ruido_stats", None)
    qds_stats = st.session_state.get("qds_stats", None)
    bt_stats = st.session_state.get("backtest_stats", None)
    mc_stats = st.session_state.get("mc_stats", None)

    st.markdown("### 🔎 Efeito atual do Tratamento de Ruído Tipo A")
    if ruido_stats is not None:
        st.write(ruido_stats)
    else:
        st.info("Tratamento de Ruído Tipo A ainda não foi aplicado.")

    st.markdown("### 📈 Resumo do QDS REAL (para apoiar decisões por ambiente)")
    if qds_stats is not None:
        st.write(
            {
                "Tamanho da janela": qds_stats["window_tam"],
                "QDS médio": qds_stats["qds_media"],
                "QDS mínimo": qds_stats["qds_min"],
                "QDS máximo": qds_stats["qds_max"],
                "% de trechos PREMIUM": f"{qds_stats['pct_premium']:.2f}%",
                "% de trechos BOM ou melhor": f"{qds_stats['pct_bom_ou_melhor']:.2f}%",
            }
        )
    else:
        st.info("QDS REAL ainda não foi calculado.")

    st.markdown("### 📉 Resumo do Backtest REAL por QDS (V15.4)")
    if bt_stats is not None:
        st.write(
            {
                "Intervalo analisado": bt_stats["Intervalo analisado"],
                "Total de pontos": bt_stats["Total de pontos"],
                "Média de acertos por série (passageiros)": bt_stats["Média de acertos por série (passageiros)"],
                "Taxa de acerto total (série exata)": bt_stats["Taxa de acerto total (série exata)"],
                "Desempenho por nível de QDS": bt_stats["por_nivel_qds"],
            }
        )
    else:
        st.info("Backtest REAL por QDS ainda não foi executado.")

    st.markdown("### 🎲 Resumo do Monte Carlo REAL por QDS (V15.4)")
    if mc_stats is not None:
        st.write(
            {
                "Intervalo amostrado": mc_stats["Intervalo amostrado"],
                "Total de simulações": mc_stats["Total de simulações"],
                "Média de acertos por série (passageiros)": mc_stats["Média de acertos por série (passageiros)"],
                "Taxa de acerto total (série exata)": mc_stats["Taxa de acerto total (série exata)"],
                "Desempenho por nível de QDS": mc_stats["por_nivel_qds"],
            }
        )
    else:
        st.info("Monte Carlo REAL por QDS ainda não foi executado.")

    st.markdown("### 🧷 Situação atual consolidada")
    st.markdown(
        "- Motor **TURBO++ ULTRA (V15.4)** produz leques com TVF + ajuste de ruído Tipo B;\n"
        "- Tratamento de Ruído Tipo A suaviza a estrada base (df_limpo) com controle de dispersão;\n"
        "- **QDS REAL** mapeia a saúde da estrada (PREMIUM/BOM/REGULAR/RUIM);\n"
        "- **Backtest REAL por QDS** mede o desempenho efetivo do motor em cada faixa de ambiente;\n"
        "- **Monte Carlo REAL por QDS** adiciona a camada de variabilidade estatística nessas faixas."
    )

    st.info(
        "A partir desta base, próximas versões (V15.5, V16...) podem incluir:\n"
        "- Modos diferenciados de ataque por ambiente (por exemplo, usar o modo 6 acertos apenas "
        "quando QDS ≥ limiar e k* estável);\n"
        "- Monte Carlo estratificado por regime de k* (estável/atenção/crítico);\n"
        "- Painéis dedicados de 'planejamento de ataque' baseados na expectativa de acertos por faixa."
    )
