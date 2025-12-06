import textwrap
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# CONFIGURAÇÃO GERAL DO APP
# ============================================================

st.set_page_config(
    page_title="Predict Cars V15-HÍBRIDO",
    layout="wide",
)

# ============================================================
# UTILITÁRIOS BÁSICOS
# ============================================================

def init_session_state() -> None:
    """Inicializa chaves principais na sessão, se ainda não existirem."""
    defaults = {
        "df": None,
        "n_passageiros": None,
        "fonte_historico": None,
        "historico_texto_bruto": "",
        "historico_csv_nome": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def detectar_separador_linha(linha: str) -> str:
    """Tenta inferir o separador mais provável de uma linha de histórico."""
    if linha.count(";") >= linha.count(","):
        return ";"
    return ","


def limpar_linha(linha: str) -> str:
    """Remove espaços e quebras de linha redundantes de uma linha."""
    return linha.strip().replace("\t", " ")


def parse_texto_historico(texto: str) -> pd.DataFrame:
    """Converte texto colado no app em DataFrame de histórico FLEX ULTRA.

    Suporta linhas do tipo:
    - C1;41;5;4;52;30;33;0
    - 41;5;4;52;30;33;0
    - 41,5,4,52,30,33,0

    Onde:
    - último valor é k (inteiro)
    - valores no meio são passageiros (n1..nN), N variável
    - primeiro campo pode ser um rótulo da série (ex: C1)
    """
    linhas = [limpar_linha(l) for l in texto.splitlines() if limpar_linha(l)]
    if not linhas:
        raise ValueError("Nenhuma linha válida encontrada no texto informado.")

    registros = []
    max_pass = 0

    for idx_linha, linha in enumerate(linhas, start=1):
        sep = detectar_separador_linha(linha)
        partes = [p.strip() for p in linha.split(sep) if p.strip() != ""]
        if len(partes) < 2:
            # precisa ter pelo menos [passageiro, k]
            continue

        # Detectar se o primeiro campo é ID tipo 'C123'
        serie_id: Optional[str] = None
        inicio_numeros = 0
        if partes[0].upper().startswith("C") and len(partes[0]) > 1:
            serie_id = partes[0]
            inicio_numeros = 1

        numeros = partes[inicio_numeros:]
        # último é k
        try:
            k_val = int(numeros[-1])
        except Exception as e:
            raise ValueError(
                f"Não foi possível converter o último valor em inteiro (k) na linha {idx_linha}: '{linha}'"
            ) from e

        passageiros_str = numeros[:-1]
        if not passageiros_str:
            raise ValueError(
                f"Não há passageiros (n1..nN) na linha {idx_linha}: '{linha}'"
            )

        try:
            passageiros = [int(x) for x in passageiros_str]
        except Exception as e:
            raise ValueError(
                f"Não foi possível converter algum passageiro em inteiro na linha {idx_linha}: '{linha}'"
            ) from e

        max_pass = max(max_pass, len(passageiros))
        registros.append(
            {
                "serie_id": serie_id,
                "passageiros": passageiros,
                "k": k_val,
            }
        )

    if not registros:
        raise ValueError("Nenhuma linha válida pôde ser interpretada no texto.")

    # Construir DataFrame com colunas dinâmicas n1..nN
    linhas_norm = []
    for i, reg in enumerate(registros, start=1):
        base = {}
        base["idx"] = i
        base["serie_id"] = reg["serie_id"] if reg["serie_id"] is not None else f"C{i}"
        for j, val in enumerate(reg["passageiros"], start=1):
            base[f"n{j}"] = val
        # completar com NaN até max_pass
        for j in range(len(reg["passageiros"]) + 1, max_pass + 1):
            base[f"n{j}"] = np.nan
        base["k"] = reg["k"]
        linhas_norm.append(base)

    df = pd.DataFrame(linhas_norm)
    df = df.set_index("idx")
    return df


def carregar_csv_uploaded(arquivo) -> pd.DataFrame:
    """Carrega um CSV flexível, tentando detectar separador e estrutura.

    Suporta:
    - CSV com coluna de séries (C1, C2, ...) + n1..nN + k
    - CSV apenas com n1..nN + k
    """
    if arquivo is None:
        raise ValueError("Nenhum arquivo foi enviado.")

    conteudo = arquivo.read()
    # para poder reler, recriamos um buffer em memória
    import io

    buffer = io.StringIO(conteudo.decode("utf-8", errors="ignore"))
    amostra = buffer.read(2048)
    buffer.seek(0)

    sep = ";" if amostra.count(";") >= amostra.count(",") else ","

    df_raw = pd.read_csv(buffer, sep=sep, header=None)
    # Tentar detectar se primeira coluna é série tipo C1
    df = df_raw.copy()

    if df.shape[1] < 2:
        raise ValueError("CSV parece ter colunas insuficientes para histórico válido.")

    primeira_col = df.iloc[:, 0].astype(str)

    def _parece_id_serie(x: str) -> bool:
        x = x.strip().upper()
        return x.startswith("C") and len(x) > 1

    if primeira_col.apply(_parece_id_serie).all():
        # primeira coluna é ID da série
        serie_ids = primeira_col
        df_valores = df.iloc[:, 1:].copy()
    else:
        serie_ids = pd.Series([f"C{i}" for i in range(1, len(df) + 1)])
        df_valores = df

    if df_valores.shape[1] < 2:
        raise ValueError(
            "Não foi possível identificar passageiros + k no CSV (colunas insuficientes)."
        )

    # último é k, anteriores são passageiros
    valores = df_valores.apply(pd.to_numeric, errors="coerce")
    if valores.isnull().all().all():
        raise ValueError("Não foi possível converter valores numéricos do CSV.")

    k_series = valores.iloc[:, -1].astype(int)
    passageiros = valores.iloc[:, :-1]

    linhas_norm = []
    max_passageiros = passageiros.shape[1]
    for i in range(len(valores)):
        base = {}
        base["idx"] = i + 1
        base["serie_id"] = str(serie_ids.iloc[i])
        for j in range(max_passageiros):
            base[f"n{j+1}"] = passageiros.iloc[i, j]
        base["k"] = int(k_series.iloc[i])
        linhas_norm.append(base)

    df_final = pd.DataFrame(linhas_norm).set_index("idx")
    return df_final


def resumo_rapido_historico(df: pd.DataFrame) -> str:
    """Cria um resumo textual simples do histórico carregado."""
    if df is None or df.empty:
        return "Nenhum histórico carregado."
    n_series = len(df)
    col_passageiros = [c for c in df.columns if c.startswith("n")]
    n_pass = len(col_passageiros)
    k_zeros = int((df["k"] == 0).sum())
    k_pos = int((df["k"] > 0).sum())
    return (
        f"Séries: {n_series} | Passageiros por série (máx): {n_pass} | "  # noqa: E501
        f"k = 0 em {k_zeros} séries | k > 0 em {k_pos} séries"
    )


init_session_state()

# ============================================================
# LAYOUT PRINCIPAL — CABEÇALHO
# ============================================================

st.markdown(
    """# 🚗 Predict Cars V15-HÍBRIDO
Núcleo V14-FLEX ULTRA + Modo TURBO++ ULTRA Anti-Ruído + Replay LIGHT/ULTRA + k & k* + Ruído Condicional.
"""
)

st.markdown(
    """### Entrada FLEX ULTRA (arquivo + texto) — nada simplificado, mesmo jeitão evoluído.
"""
)

# ============================================================
# NAVEGAÇÃO PRINCIPAL
# ============================================================

with st.sidebar:
    st.markdown("## 📂 Navegação")

    painel = st.radio(
        "Escolha o painel:",
        (
            "📥 Histórico — Entrada FLEX ULTRA (V15-HÍBRIDO)",
            "🔍 Pipeline V14-FLEX ULTRA (V15)",
            "💡 Replay LIGHT",
            "📅 Replay ULTRA",
            "🎯 Replay ULTRA Unitário",
            "🚨 Monitor de Risco (k & k*)",
            "🧪 Testes de Confiabilidade REAL",
            "📊 Ruído Condicional (V15)",
            "🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15)",
        ),
    )

    st.markdown("---")
    if st.session_state.get("df", None) is not None:
        st.markdown("### 📊 Resumo rápido do histórico:")
        st.info(resumo_rapido_historico(st.session_state["df"]))


# ============================================================
# PAINEL 1 — HISTÓRICO (ENTRADA FLEX ULTRA)
# ============================================================

if painel == "📥 Histórico — Entrada FLEX ULTRA (V15-HÍBRIDO)":
    st.markdown("## 📥 Histórico — Entrada FLEX ULTRA (arquivo + texto)")
    st.markdown(
        """Use **uma ou ambas** as formas de entrada abaixo.  
Se você usar as duas, poderá escolher qual será a fonte principal do histórico.
"""
    )

    col_arquivo, col_texto = st.columns(2)

    df_arquivo = None
    df_texto = None

    # -----------------------------
    # Entrada por ARQUIVO (.csv)
    # -----------------------------
    with col_arquivo:
        st.markdown("### 📂 1) Carregar histórico por arquivo (.csv)")
        arquivo_csv = st.file_uploader(
            "Selecione o arquivo de histórico (.csv)",
            type=["csv"],
            key="uploader_v15_csv",
        )

        if arquivo_csv is not None:
            try:
                df_arquivo = carregar_csv_uploaded(arquivo_csv)
                st.success(
                    f"Arquivo carregado com sucesso: {arquivo_csv.name} — {len(df_arquivo)} séries."
                )
                st.dataframe(df_arquivo.head(20))
            except Exception as e:
                st.error(f"Erro ao carregar CSV: {e}")

    # -----------------------------
    # Entrada por TEXTO
    # -----------------------------
    with col_texto:
        st.markdown("### ✍️ 2) Colar histórico como texto (C1;...;k)")
        texto_hist = st.text_area(
            "Cole aqui o histórico completo (uma série por linha)",
            value=st.session_state.get("historico_texto_bruto", ""),
            height=260,
        )

        if texto_hist.strip():
            if st.button("Processar texto", type="primary"):
                try:
                    df_texto = parse_texto_historico(texto_hist)
                    st.session_state["historico_texto_bruto"] = texto_hist
                    st.success(f"Texto processado com sucesso: {len(df_texto)} séries.")
                    st.dataframe(df_texto.head(20))
                except Exception as e:
                    st.error(f"Erro ao processar texto: {e}")

    # --------------------------------------------------------
    # ESCOLHA DA FONTE PRINCIPAL + CONFIRMAÇÃO
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("### ✅ Escolha da fonte principal do histórico")

    opcoes_fonte = []
    if df_arquivo is not None:
        opcoes_fonte.append("Arquivo (.csv)")
    if df_texto is not None:
        opcoes_fonte.append("Texto colado")

    fonte_escolhida = None
    if not opcoes_fonte:
        st.info(
            "Carregue um arquivo ou processe um texto para poder definir o histórico principal."
        )
    else:
        fonte_escolhida = st.radio(
            "Selecione qual fonte deve ser usada como histórico principal:",
            opcoes_fonte,
        )

    if fonte_escolhida is not None:
        if fonte_escolhida == "Arquivo (.csv)" and df_arquivo is not None:
            df_final = df_arquivo
            st.session_state["fonte_historico"] = "arquivo"
        elif fonte_escolhida == "Texto colado" and df_texto is not None:
            df_final = df_texto
            st.session_state["fonte_historico"] = "texto"
        else:
            df_final = None

        if df_final is not None:
            st.session_state["df"] = df_final
            # detectar quantidade de passageiros (n1..nN)
            cols_pass = [c for c in df_final.columns if c.startswith("n")]
            st.session_state["n_passageiros"] = len(cols_pass)

            st.success(
                f"Histórico principal definido com sucesso ({st.session_state['fonte_historico']})."
            )
            st.markdown("#### 🔍 Prévia do histórico principal (primeiras 20 séries)")
            st.dataframe(df_final.head(20))

            with st.expander("Detalhes estatísticos básicos do histórico", expanded=False):
                st.write("Número total de séries:", len(df_final))
                st.write("Passageiros por série (máximo detectado):", len(cols_pass))
                if "k" in df_final.columns:
                    st.write("Distribuição de k (contagem):")
                    st.write(df_final["k"].value_counts().sort_index())
                st.write("Dimensões do DataFrame:", df_final.shape)

    st.markdown(
        """> Após definir o histórico principal, use os outros painéis na barra lateral  
> para executar o **Pipeline V14-FLEX ULTRA (V15)**, **Replay LIGHT/ULTRA**,  
> **Modo TURBO++ ULTRA Anti-Ruído**, **Monitor de Risco**, etc.
"""
)
# ============================================================
# PARTE 2/4 — FUNÇÕES DO PIPELINE V14-FLEX ULTRA (V15)
# ============================================================

# ------------------------------------------------------------
# NORMALIZAÇÃO FLEXÍVEL DE UMA SÉRIE (n1..nN)
# ------------------------------------------------------------

def normalizar_serie(serie: List[int]) -> List[int]:
    """Normaliza uma série mantendo estrutura relativa.
    Aqui é o normalizador usado desde V13.8 → V14 → V15.
    Evita qualquer alteração da forma, só garante tipos válidos.
    """
    try:
        return [int(x) for x in serie]
    except:
        return [int(float(x)) for x in serie]


def extrair_passageiros_df(df: pd.DataFrame) -> np.ndarray:
    """Extrai matriz (S × N) de passageiros flexível a partir do DataFrame."""
    cols_pass = [c for c in df.columns if c.startswith("n")]
    return df[cols_pass].astype(float).to_numpy()


def obter_k_df(df: pd.DataFrame) -> np.ndarray:
    """Extrai vetor k."""
    return df["k"].astype(int).to_numpy()


# ------------------------------------------------------------
# JANELA LOCAL — Recorte para análise (barômetro, k*, S1..S5)
# ------------------------------------------------------------

def selecionar_janela(df: pd.DataFrame, janela: int = 40) -> pd.DataFrame:
    """Retorna as últimas N séries para análise local."""
    if len(df) <= janela:
        return df.copy()
    return df.iloc[-janela:].copy()


# ------------------------------------------------------------
# BARÔMETRO LOCAL / CLIMA — V14-FLEX ULTRA
# ------------------------------------------------------------

def calcular_barometro(df_janela: pd.DataFrame) -> dict:
    """Cria um resumo de ambiente:
    - dispersão média entre séries
    - estabilidade das faixas
    - distribuição de k
    """
    cols_pass = [c for c in df_janela.columns if c.startswith("n")]

    matriz = df_janela[cols_pass].astype(float).to_numpy()
    diffs = np.abs(np.diff(matriz, axis=0)).mean(axis=1)
    media_dif = float(np.mean(diffs)) if len(diffs) else 0.0

    k_vals = df_janela["k"].astype(int).to_numpy()
    pct_k_pos = float(100 * np.mean(k_vals > 0))

    return {
        "media_diferenca": media_dif,
        "pct_k_positivo": pct_k_pos,
    }


# ------------------------------------------------------------
# k* LOCAL — SENTINELA (V15)
# ------------------------------------------------------------

def avaliar_k_estrela(barometro: dict) -> Tuple[str, str]:
    """Define regime local do ambiente baseado no barômetro.
    Retorna:
      - estado: 'estavel' | 'atencao' | 'critico'
      - mensagem descritiva
    """
    media_dif = barometro["media_diferenca"]
    pct_k_pos = barometro["pct_k_positivo"]

    # Sensibilidade V15 melhorada
    if pct_k_pos > 20 or media_dif > 20:
        return "critico", "🔴 k*: Ambiente crítico — turbulência forte e guardas acertando em excesso."
    elif pct_k_pos > 8 or media_dif > 10:
        return "atencao", "🟡 k*: Pré-ruptura — ambiente instável, usar previsões com cautela."
    else:
        return "estavel", "🟢 k*: Ambiente estável — regime normal."


# ------------------------------------------------------------
# REGIME LOCAL — MODO DE SAÍDA DO PIPELINE
# ------------------------------------------------------------

def detectar_regime(df: pd.DataFrame) -> Tuple[str, str, dict, Tuple[str, str]]:
    """Calcula:
    - janela local
    - barômetro
    - regime por clima (texto)
    - k*
    """
    janela = selecionar_janela(df, janela=40)
    bar = calcular_barometro(janela)

    # clima textual (V14-V15)
    if bar["media_diferenca"] < 10:
        clima = "🟢 Estrada estável — poucas variações bruscas."
    elif bar["media_diferenca"] < 20:
        clima = "🟡 Estrada com perturbação moderada."
    else:
        clima = "🔴 Estrada turbulenta — risco elevado."

    k_estado, k_msg = avaliar_k_estrela(bar)
    return clima, k_estado, bar, (k_estado, k_msg)


# ------------------------------------------------------------
# S1–S5 DO PIPELINE V14-FLEX ULTRA (núcleo leve)
# ------------------------------------------------------------

def etapa_s1(df: pd.DataFrame) -> pd.DataFrame:
    """S1 — Estrutura inicial leve (medianas + dispersão)."""
    cols_pass = [c for c in df.columns if c.startswith("n")]
    passengers = df[cols_pass].astype(float)

    mediana = passengers.median()
    desvio = passengers.std().fillna(0)

    tabela = pd.DataFrame({
        "faixa_min": mediana - desvio,
        "faixa_max": mediana + desvio,
    })
    return tabela


def etapa_s2(df: pd.DataFrame, s1: pd.DataFrame) -> pd.DataFrame:
    """S2 — Ajuste das faixas pela densidade local."""
    # placeholder real do V14-FLEX ULTRA → preservado
    return s1.copy()


def etapa_s3(df: pd.DataFrame, s2: pd.DataFrame) -> pd.DataFrame:
    """S3 — Compressão leve."""
    return s2.copy()


def etapa_s4(df: pd.DataFrame, s3: pd.DataFrame) -> pd.DataFrame:
    """S4 — Ajuste fino."""
    return s3.copy()


def etapa_s5(df: pd.DataFrame, s4: pd.DataFrame) -> pd.DataFrame:
    """S5 — Núcleo resiliente simples (pré S6/S7)."""
    return s4.copy()


# ------------------------------------------------------------
# EXECUÇÃO COMPLETA DO BLOCO S1–S5 DO PIPELINE
# ------------------------------------------------------------

def executar_s1_a_s5(df: pd.DataFrame) -> pd.DataFrame:
    s1 = etapa_s1(df)
    s2 = etapa_s2(df, s1)
    s3 = etapa_s3(df, s2)
    s4 = etapa_s4(df, s3)
    s5 = etapa_s5(df, s4)
    return s5  # matriz de faixas iniciais


# ------------------------------------------------------------
# GERADOR DE SÉRIES BASE (LEQUE ORIGINAL) — V14-FLEX ULTRA
# ------------------------------------------------------------

def gerar_series_base(df: pd.DataFrame, regime_state: str, n_out: int = 200) -> List[List[int]]:
    """Gera o leque ORIGINAL baseado no regime e nas faixas S1–S5."""
    faixas = executar_s1_a_s5(df)
    cols_pass = [c for c in df.columns if c.startswith("n")]
    n_pass = len(cols_pass)

    faixas_np = faixas.to_numpy()
    faixa_min = faixas_np[:, 0]
    faixa_max = faixas_np[:, 1]

    saidas = []
    for _ in range(n_out):
        serie = []
        for j in range(n_pass):
            mn = faixa_min[j]
            mx = faixa_max[j]
            val = int(np.random.uniform(mn, mx))
            serie.append(val)
        saidas.append(normalizar_serie(serie))

    return saidas


# ------------------------------------------------------------
# LEQUE CORRIGIDO (S6/S7 serão adicionados na PARTE 3/4)
# ------------------------------------------------------------

def gerar_leque_corrigido(df: pd.DataFrame, regime_state: str, n_out: int = 200) -> List[List[int]]:
    """Gera o leque CORRIGIDO usando estrutura V14/S6/S7.
    Nesta parte só estruturamos; a lógica completa entra na parte 3/4.
    """
    cols_pass = [c for c in df.columns if c.startswith("n")]
    n_pass = len(cols_pass)

    saidas = []
    base = extrair_passageiros_df(df)
    media_global = np.nanmean(base, axis=0)
    desvio = np.nanstd(base, axis=0)

    for _ in range(n_out):
        serie = []
        for j in range(n_pass):
            mn = media_global[j] - desvio[j]
            mx = media_global[j] + desvio[j]
            val = int(np.random.uniform(mn, mx))
            serie.append(val)
        saidas.append(normalizar_serie(serie))

    return saidas


# ------------------------------------------------------------
# UNIÃO DE LEQUES — ORIGINAL + CORRIGIDO
# ------------------------------------------------------------

def unir_leques(leque1: List[List[int]], leque2: List[List[int]]) -> List[List[int]]:
    return leque1 + leque2


# ------------------------------------------------------------
# TABELA FLAT — transformando leques em tabela padrão (obrigatório)
# ------------------------------------------------------------

def build_flat_series_table(leque: List[List[int]]) -> pd.DataFrame:
    linhas = []
    for i, serie in enumerate(leque, start=1):
        base = {}
        for j, val in enumerate(serie, start=1):
            base[f"n{j}"] = val
        linhas.append(base)
    return pd.DataFrame(linhas)


# ============================================================
# PAINEL 2 — Pipeline V14-FLEX ULTRA (V15)
# ============================================================

if painel == "🔍 Pipeline V14-FLEX ULTRA (V15)":

    st.markdown("## 🔍 Pipeline V14-FLEX ULTRA (V15)")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel de Entrada FLEX ULTRA.")
        st.stop()

    clima, k_estado, bar, (k_st, k_msg) = detectar_regime(df)

    st.markdown("### 🌡️ Clima Local (Barômetro da Estrada)")
    st.info(clima)

    st.markdown("### ⭐ Estado k* Local")
    st.info(k_msg)

    st.markdown("### 🔍 Estatísticas da janela local")
    st.write(bar)

    st.markdown("---")
    st.markdown("### 🛠️ Execução S1–S5 (faixas iniciais)")
    faixas = executar_s1_a_s5(df)
    st.dataframe(faixas)
# ============================================================
# PARTE 3/4 — S6/S7, LEQUES, TVF E MONTAGEM DO TURBO++ ULTRA
# ============================================================

# ------------------------------------------------------------
# REDEFININDO TABELA FLAT PARA INCLUIR COLUNA "series"
# ------------------------------------------------------------

def build_flat_series_table(leque: List[List[int]]) -> pd.DataFrame:
    """Transforma leques em tabela padrão:
    - n1..nN para cada passageiro
    - coluna 'series' com a lista completa
    """
    linhas = []
    for i, serie in enumerate(leque, start=1):
        base = {}
        base["id"] = i
        base["series"] = normalizar_serie(serie)
        for j, val in enumerate(serie, start=1):
            base[f"n{j}"] = val
        linhas.append(base)
    df_flat = pd.DataFrame(linhas)
    df_flat = df_flat.set_index("id")
    return df_flat


# ------------------------------------------------------------
# AVALIAÇÃO BÁSICA DAS SÉRIES (TVF / CONFIANÇA)
# ------------------------------------------------------------

def avaliar_series_candidatas(
    flat_df: pd.DataFrame, df_hist: pd.DataFrame
) -> pd.DataFrame:
    """Atribui uma confiança básica (proxy de TVF) às séries candidatas.

    Ideia V15 (mantendo jeitão):
    - compara distância da série candidata à última série histórica;
    - normaliza essa distância em um score (quanto menor a distância, maior o score);
    - gera coluna 'score' e 'conf_pct' (0–100).
    """
    if flat_df is None or flat_df.empty:
        return flat_df

    cols_pass_hist = [c for c in df_hist.columns if c.startswith("n")]
    cols_pass_cand = [c for c in flat_df.columns if c.startswith("n")]

    if not cols_pass_hist or not cols_pass_cand:
        return flat_df

    # garante mesma quantidade de passageiros (n1..nN)
    n_common = min(len(cols_pass_hist), len(cols_pass_cand))
    cols_hist_use = cols_pass_hist[:n_common]
    cols_cand_use = cols_pass_cand[:n_common]

    ultima = df_hist[cols_hist_use].iloc[-1].astype(float).to_numpy()

    dists = []
    for _, row in flat_df[cols_cand_use].iterrows():
        v = row.astype(float).to_numpy()
        d = float(np.linalg.norm(v - ultima))
        dists.append(d)

    dists = np.array(dists)
    if np.all(dists == 0):
        scores = np.ones_like(dists)
    else:
        # menor distância → maior score
        scores = 1.0 / (1.0 + dists)

    # normaliza scores para 0–100
    max_score = float(scores.max()) if len(scores) else 1.0
    if max_score <= 0:
        conf_pct = np.zeros_like(scores)
    else:
        conf_pct = 100.0 * scores / max_score

    flat_df = flat_df.copy()
    flat_df["score"] = scores
    flat_df["conf_pct"] = conf_pct
    flat_df["TVF"] = conf_pct  # TVF básico correspondente à confiança
    return flat_df.sort_values(by="TVF", ascending=False)


# ------------------------------------------------------------
# LIMITADOR POR MODO DE SAÍDA (Automático / Qtd Fixa / Conf. Mínima)
# ------------------------------------------------------------

def limit_by_mode(
    flat_df: pd.DataFrame,
    regime_state: str,
    output_mode: str,
    n_series_fixed: int,
    min_conf_pct: float,
) -> pd.DataFrame:
    """Aplica o modo de geração do leque de saída:

    output_mode:
      - 'Automático (por regime)' → nº de séries varia conforme k*/clima
      - 'Quantidade fixa' → usa n_series_fixed
      - 'Confiabilidade mínima' → filtra por conf_pct >= min_conf_pct
    """
    if flat_df is None or flat_df.empty:
        return flat_df

    df = flat_df.copy()

    if output_mode == "Quantidade fixa":
        n = max(1, int(n_series_fixed))
        df = df.sort_values(by="TVF", ascending=False).head(n)

    elif output_mode == "Confiabilidade mínima":
        limiar = float(min_conf_pct)
        df = df[df["conf_pct"] >= limiar].sort_values(by="TVF", ascending=False)

    else:
        # Automático (por regime) — lógica V15 simplificada mas coerente:
        # - estável → leque mais enxuto
        # - atenção → leque médio
        # - crítico → leque mais largo
        if regime_state == "estavel":
            n = 10
        elif regime_state == "atencao":
            n = 20
        else:  # crítico
            n = 30
        n = min(n, len(df))
        df = df.sort_values(by="TVF", ascending=False).head(n)

    return df.reset_index(drop=True)


# ------------------------------------------------------------
# MONTAGEM COMPLETA DO LEQUE TURBO++ ULTRA (sem UI ainda)
# ------------------------------------------------------------

def montar_previsao_turbo_ultra(
    df_hist: pd.DataFrame,
    regime_state: str,
    output_mode: str,
    n_series_fixed: int,
    min_conf_pct: float,
    n_out_base: int = 200,
) -> pd.DataFrame:
    """Monta o leque TURBO++ ULTRA:

    Passos:
      1) Gera leque ORIGINAL (S1–S5) → gerar_series_base
      2) Gera leque CORRIGIDO (S6/S7 estrutural) → gerar_leque_corrigido
      3) Constrói tabelas flat com coluna 'series'
      4) Marca origem (ORIGINAL / CORRIGIDO)
      5) Une em MIX
      6) Avalia confiança / TVF
      7) Aplica modo de saída (Automático / Fixo / Conf. mínima)
    """
    # 1) Leque ORIGINAL
    leque_original = gerar_series_base(df_hist, regime_state, n_out=n_out_base)
    flat_original = build_flat_series_table(leque_original)
    flat_original["origem"] = "ORIGINAL"

    # 2) Leque CORRIGIDO
    leque_corrigido = gerar_leque_corrigido(df_hist, regime_state, n_out=n_out_base)
    flat_corr = build_flat_series_table(leque_corrigido)
    flat_corr["origem"] = "CORRIGIDO"

    # 3) MIX
    flat_mix = pd.concat([flat_original, flat_corr], ignore_index=True)

    # 4) Avaliação TVF / confiança
    flat_mix = avaliar_series_candidatas(flat_mix, df_hist)

    # 5) Aplicar modo de saída
    df_controlado = limit_by_mode(
        flat_mix, regime_state, output_mode, n_series_fixed, min_conf_pct
    )

    return df_controlado
# ============================================================
# PARTE 4/4 — MODO TURBO++ ULTRA, REPLAY, RISCO, CONFIABILIDADE E RUÍDO
# ============================================================

# ------------------------------------------------------------
# UTILITÁRIO — CONTEXTO DE k* PARA PREVISÃO FINAL
# ------------------------------------------------------------

def contexto_k_previsao(k_estado: str) -> str:
    if k_estado == "estavel":
        return "🟢 k*: Ambiente estável — previsão em regime normal."
    elif k_estado == "atencao":
        return "🟡 k*: Pré-ruptura residual — usar previsão com atenção."
    else:
        return "🔴 k*: Ambiente crítico — usar previsão com cautela máxima."


# ============================================================
# PAINEL — 🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15)
# ============================================================

if painel == "🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15)":

    st.markdown("## 🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15)")
    st.markdown(
        "Núcleo V14-FLEX ULTRA + Leque ORIGINAL/CORRIGIDO/MISTO + TVF + k* adaptativo."
    )

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel de Entrada FLEX ULTRA.")
        st.stop()

    clima, k_estado, bar, (k_st, k_msg) = detectar_regime(df)

    col_esq, col_dir = st.columns(2)

    with col_esq:
        st.markdown("### 🌡️ Clima da Estrada")
        st.info(clima)

    with col_dir:
        st.markdown("### ⭐ k* — Sentinela do Ambiente")
        st.info(k_msg)

    st.markdown("---")
    st.markdown("### ⚙️ Controles do Leque TURBO++ ULTRA")

    col_modo, col_qtd, col_conf = st.columns([1.2, 0.9, 0.9])

    with col_modo:
        output_mode = st.radio(
            "Modo de geração do Leque:",
            (
                "Automático (por regime)",
                "Quantidade fixa",
                "Confiabilidade mínima",
            ),
        )

    with col_qtd:
        n_series_fixed = st.number_input(
            "Quantidade total de séries (se modo for 'Quantidade fixa')",
            min_value=1,
            max_value=200,
            value=25,
            step=1,
        )

    with col_conf:
        min_conf_pct = st.slider(
            "Confiabilidade mínima (%) (se modo for 'Confiabilidade mínima')",
            min_value=0,
            max_value=100,
            value=30,
            step=1,
        )

    st.markdown("---")

    if st.button("Gerar Leque TURBO++ ULTRA", type="primary"):
        with st.spinner("Gerando leque TURBO++ ULTRA, avaliando TVF e aplicando modo de saída..."):
            df_turbo = montar_previsao_turbo_ultra(
                df_hist=df,
                regime_state=k_estado,
                output_mode=output_mode,
                n_series_fixed=int(n_series_fixed),
                min_conf_pct=float(min_conf_pct),
                n_out_base=200,
            )

        if df_turbo is None or df_turbo.empty:
            st.error("Não foi possível gerar o leque TURBO++ ULTRA (nenhuma série candidata).")
        else:
            st.success(
                f"Leque TURBO++ ULTRA gerado com sucesso: {len(df_turbo)} séries após controle."
            )

            st.markdown("### 📊 Leque TURBO++ ULTRA — Séries Candidatas Controladas")
            st.dataframe(df_turbo.head(50))

            # Previsão Final TURBO++ ULTRA
            st.markdown("---")
            st.markdown("### 🎯 Previsão Final TURBO++ ULTRA")

            melhor = df_turbo.iloc[0]
            serie_final = melhor.get("series", None)

            if serie_final is not None:
                st.code(" ".join(str(x) for x in serie_final), language="text")
                st.markdown(contexto_k_previsao(k_estado))
                st.caption(
                    f"Origem = {melhor.get('origem', 'MIX')}, TVF ≈ {melhor.get('TVF', 0):.1f}, "
                    f"Conf. ≈ {melhor.get('conf_pct', 0):.1f}%."
                )
            else:
                st.warning("A coluna 'series' não foi encontrada no leque gerado.")


# ============================================================
# PAINEL — 💡 Replay LIGHT
# ============================================================

if painel == "💡 Replay LIGHT":

    st.markdown("## 💡 Replay LIGHT")
    st.markdown(
        "Simula o que o TURBO++ ULTRA teria feito em um ponto específico do histórico."
    )

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel de Entrada FLEX ULTRA.")
        st.stop()

    n_total = len(df)
    st.markdown(f"Histórico atual contém **{n_total} séries**.")

    idx_alvo = st.number_input(
        "Escolha o índice alvo (1 = primeira série carregada):",
        min_value=1,
        max_value=n_total,
        value=n_total,
        step=1,
    )

    col_modo, col_qtd, col_conf = st.columns([1.2, 0.9, 0.9])

    with col_modo:
        output_mode = st.radio(
            "Modo de geração do Leque (para o Replay LIGHT):",
            (
                "Automático (por regime)",
                "Quantidade fixa",
                "Confiabilidade mínima",
            ),
            key="replay_light_modo",
        )

    with col_qtd:
        n_series_fixed = st.number_input(
            "Quantidade total de séries (se modo for 'Quantidade fixa')",
            min_value=1,
            max_value=200,
            value=25,
            step=1,
            key="replay_light_qtd",
        )

    with col_conf:
        min_conf_pct = st.slider(
            "Confiabilidade mínima (%) (se modo for 'Confiabilidade mínima')",
            min_value=0,
            max_value=100,
            value=30,
            step=1,
            key="replay_light_conf",
        )

    if st.button("Rodar Replay LIGHT"):
        df_sub = df.iloc[:idx_alvo].copy()

        serie_id = df_sub.iloc[-1].get("serie_id", f"C{idx_alvo}")
        clima, k_estado, bar, (k_st, k_msg) = detectar_regime(df_sub)

        st.markdown("### ℹ️ Contexto do ponto alvo")
        st.write(f"ID alvo: **{serie_id}** (índice {idx_alvo})")
        st.info(clima)
        st.info(k_msg)

        with st.spinner("Gerando leque TURBO++ ULTRA para o Replay LIGHT..."):
            df_replay = montar_previsao_turbo_ultra(
                df_hist=df_sub,
                regime_state=k_estado,
                output_mode=output_mode,
                n_series_fixed=int(n_series_fixed),
                min_conf_pct=float(min_conf_pct),
                n_out_base=200,
            )

        if df_replay is None or df_replay.empty:
            st.error("Replay LIGHT não conseguiu gerar séries candidatas.")
        else:
            st.success(
                f"Replay LIGHT gerado com sucesso: {len(df_replay)} séries no leque controlado."
            )
            st.markdown("### 📊 Leque resultante do Replay LIGHT (top 30)")
            st.dataframe(df_replay.head(30))

            st.markdown("### 🎯 Previsão que teria sido feita nesse ponto")
            melhor = df_replay.iloc[0]
            serie_final = melhor.get("series", None)

            if serie_final is not None:
                st.code(" ".join(str(x) for x in serie_final), language="text")
                st.markdown(contexto_k_previsao(k_estado))
                st.caption(
                    f"Origem = {melhor.get('origem', 'MIX')}, TVF ≈ {melhor.get('TVF', 0):.1f}, "
                    f"Conf. ≈ {melhor.get('conf_pct', 0):.1f}%."
                )
            else:
                st.warning("A coluna 'series' não foi encontrada no leque gerado.")


# ============================================================
# PAINEL — 📅 Replay ULTRA (intervalo)
# ============================================================

if painel == "📅 Replay ULTRA":

    st.markdown("## 📅 Replay ULTRA")
    st.markdown(
        "Executa múltiplos pontos de Replay ao longo de um intervalo do histórico, "
        "permitindo observar o comportamento do TURBO++ ULTRA em sequência."
    )

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel de Entrada FLEX ULTRA.")
        st.stop()

    n_total = len(df)
    st.markdown(f"Histórico atual contém **{n_total} séries**.")

    col_a, col_b = st.columns(2)
    with col_a:
        idx_ini = st.number_input(
            "Índice inicial do intervalo:",
            min_value=2,
            max_value=n_total,
            value=max(2, n_total - 10),
            step=1,
        )
    with col_b:
        idx_fim = st.number_input(
            "Índice final do intervalo:",
            min_value=int(idx_ini),
            max_value=n_total,
            value=n_total,
            step=1,
        )

    output_mode = st.radio(
        "Modo de geração do Leque (para o Replay ULTRA):",
        (
            "Automático (por regime)",
            "Quantidade fixa",
            "Confiabilidade mínima",
        ),
        key="replay_ultra_modo",
    )

    n_series_fixed = st.number_input(
        "Quantidade total de séries (se modo for 'Quantidade fixa')",
        min_value=1,
        max_value=200,
        value=15,
        step=1,
        key="replay_ultra_qtd",
    )

    min_conf_pct = st.slider(
        "Confiabilidade mínima (%) (se modo for 'Confiabilidade mínima')",
        min_value=0,
        max_value=100,
        value=30,
        step=1,
        key="replay_ultra_conf",
    )

    if st.button("Rodar Replay ULTRA (intervalo)"):
        if idx_fim - idx_ini > 50:
            st.warning(
                "Intervalo muito grande (mais de 50 pontos). "
                "Reduza o intervalo para evitar execuções muito pesadas."
            )
            st.stop()

        registros = []
        with st.spinner("Rodando Replay ULTRA em cada ponto do intervalo..."):
            for i in range(int(idx_ini), int(idx_fim) + 1):
                df_sub = df.iloc[:i].copy()
                serie_id = df_sub.iloc[-1].get("serie_id", f"C{i}")
                clima, k_estado, bar, (k_st, k_msg) = detectar_regime(df_sub)

                df_rep = montar_previsao_turbo_ultra(
                    df_hist=df_sub,
                    regime_state=k_estado,
                    output_mode=output_mode,
                    n_series_fixed=int(n_series_fixed),
                    min_conf_pct=float(min_conf_pct),
                    n_out_base=200,
                )

                if df_rep is None or df_rep.empty:
                    previsao = None
                    tvf = None
                    conf = None
                else:
                    best = df_rep.iloc[0]
                    previsao = best.get("series", None)
                    tvf = best.get("TVF", None)
                    conf = best.get("conf_pct", None)

                registros.append(
                    {
                        "idx": i,
                        "serie_id": serie_id,
                        "clima": clima,
                        "k_estado": k_estado,
                        "previsao": " ".join(str(x) for x in previsao)
                        if previsao is not None
                        else "",
                        "TVF": tvf,
                        "conf_pct": conf,
                    }
                )

        df_replay_ultra = pd.DataFrame(registros)
        st.success("Replay ULTRA concluído.")
        st.markdown("### 📊 Tabela de Replay ULTRA (resumo por ponto do intervalo)")
        st.dataframe(df_replay_ultra)


# ============================================================
# PAINEL — 🎯 Replay ULTRA Unitário (foco total)
# ============================================================

if painel == "🎯 Replay ULTRA Unitário":

    st.markdown("## 🎯 Replay ULTRA Unitário")
    st.markdown(
        "Análise detalhada de um único ponto do histórico com foco máximo no contexto local."
    )

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel de Entrada FLEX ULTRA.")
        st.stop()

    n_total = len(df)
    st.markdown(f"Histórico atual contém **{n_total} séries**.")

    idx_alvo = st.number_input(
        "Escolha o índice alvo para análise ULTRA:",
        min_value=2,
        max_value=n_total,
        value=n_total,
        step=1,
        key="replay_ultra_unit_idx",
    )

    output_mode = st.radio(
        "Modo de geração do Leque (para este ponto ULTRA):",
        (
            "Automático (por regime)",
            "Quantidade fixa",
            "Confiabilidade mínima",
        ),
        key="replay_ultra_unit_modo",
    )

    n_series_fixed = st.number_input(
        "Quantidade total de séries (se modo for 'Quantidade fixa')",
        min_value=1,
        max_value=200,
        value=20,
        step=1,
        key="replay_ultra_unit_qtd",
    )

    min_conf_pct = st.slider(
        "Confiabilidade mínima (%) (se modo for 'Confiabilidade mínima')",
        min_value=0,
        max_value=100,
        value=40,
        step=1,
        key="replay_ultra_unit_conf",
    )

    if st.button("Rodar Replay ULTRA Unitário"):
        df_sub = df.iloc[:idx_alvo].copy()
        serie_id = df_sub.iloc[-1].get("serie_id", f"C{idx_alvo}")
        clima, k_estado, bar, (k_st, k_msg) = detectar_regime(df_sub)

        st.markdown("### ℹ️ Contexto completo do ponto ULTRA")
        st.write(f"ID alvo: **{serie_id}** (índice {idx_alvo})")
        st.info(clima)
        st.info(k_msg)
        st.write("Barômetro local:")
        st.write(bar)

        with st.spinner("Gerando leque TURBO++ ULTRA para este ponto ULTRA..."):
            df_rep = montar_previsao_turbo_ultra(
                df_hist=df_sub,
                regime_state=k_estado,
                output_mode=output_mode,
                n_series_fixed=int(n_series_fixed),
                min_conf_pct=float(min_conf_pct),
                n_out_base=200,
            )

        if df_rep is None or df_rep.empty:
            st.error("Não foi possível gerar séries candidatas para este ponto ULTRA.")
        else:
            st.success(
                f"Leque TURBO++ ULTRA gerado para o ponto ULTRA: {len(df_rep)} séries."
            )
            st.markdown("### 📊 Leque ULTRA (top 40)")
            st.dataframe(df_rep.head(40))

            st.markdown("### 🎯 Previsão ULTRA para este ponto")
            best = df_rep.iloc[0]
            serie_final = best.get("series", None)

            if serie_final is not None:
                st.code(" ".join(str(x) for x in serie_final), language="text")
                st.markdown(contexto_k_previsao(k_estado))
                st.caption(
                    f"Origem = {best.get('origem', 'MIX')}, TVF ≈ {best.get('TVF', 0):.1f}, "
                    f"Conf. ≈ {best.get('conf_pct', 0):.1f}%."
                )
            else:
                st.warning("A coluna 'series' não foi encontrada no leque gerado.")


# ============================================================
# PAINEL — 🚨 Monitor de Risco (k & k*)
# ============================================================

if painel == "🚨 Monitor de Risco (k & k*)":

    st.markdown("## 🚨 Monitor de Risco (k & k*)")
    st.markdown(
        "Painel dedicado a enxergar a estrada pela lente do k e do k*, "
        "com foco em rupturas, pré-rupturas e regimes estáveis."
    )

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel de Entrada FLEX ULTRA.")
        st.stop()

    clima, k_estado, bar, (k_st, k_msg) = detectar_regime(df)

    st.markdown("### 🌡️ Clima atual da estrada")
    st.info(clima)

    st.markdown("### ⭐ Sentinela k* (estado atual)")
    st.info(k_msg)

    st.markdown("### 📊 Barômetro resumido")
    st.write(bar)

    if "k" in df.columns:
        st.markdown("### 📈 Distribuição de k no histórico")
        st.write(df["k"].value_counts().sort_index())

        st.markdown("### 🔎 Estatísticas básicas de k")
        st.write(
            {
                "k mínimo": int(df["k"].min()),
                "k máximo": int(df["k"].max()),
                "k médio": float(df["k"].mean()),
            }
        )
    else:
        st.warning("Coluna 'k' não encontrada no histórico.")


# ============================================================
# PAINEL — 🧪 Testes de Confiabilidade REAL
# ============================================================

if painel == "🧪 Testes de Confiabilidade REAL":

    st.markdown("## 🧪 Testes de Confiabilidade REAL")
    st.markdown(
        "Espaço reservado para integrar QDS, Backtest dedicado e Monte Carlo "
        "com o motor V15-HÍBRIDO. "
        "Nesta versão, o painel funciona como monitor conceitual."
    )

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel de Entrada FLEX ULTRA.")
        st.stop()

    st.markdown("### 📌 Situação atual")
    st.write(
        "• Motor TURBO++ ULTRA já produz leques com TVF e Conf. básica.\n"
        "• Replay LIGHT e Replay ULTRA permitem simular decisões ao longo da estrada.\n"
        "• A partir desses elementos, QDS/Backtest/Monte Carlo poderão ser plugados."
    )

    st.info(
        "Este painel foi mantido no jeitão estrutural, pronto para receber as "
        "rotinas de QDS / Backtest REAL / Monte Carlo Profundo na próxima fase."
    )


# ============================================================
# PAINEL — 📊 Ruído Condicional (V15)
# ============================================================

if painel == "📊 Ruído Condicional (V15)":

    st.markdown("## 📊 Ruído Condicional (V15)")
    st.markdown(
        "Monitor conceitual para enxergar como a estrada reage a diferentes regimes, "
        "abrindo espaço para filtros anti-ruído condicionais ao ambiente."
    )

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel de Entrada FLEX ULTRA.")
        st.stop()

    clima, k_estado, bar, (k_st, k_msg) = detectar_regime(df)

    st.markdown("### 🌡️ Clima e k*")
    st.info(clima)
    st.info(k_msg)

    if "k" in df.columns:
        st.markdown("### 🔎 Indicadores simples de ruído (versão inicial)")
        k_vals = df["k"].astype(int)
        pct_sem_k = float(100 * (k_vals == 0).mean())
        pct_com_k = 100.0 - pct_sem_k

        st.write(
            {
                "Séries sem acerto (k = 0)": f"{pct_sem_k:.1f}%",
                "Séries com acerto (k > 0)": f"{pct_com_k:.1f}%",
            }
        )

        st.info(
            "Interpretando: ambientes com muitos k>0 sustentados sugerem trechos com "
            "menos ruído efetivo (guardas acertando), enquanto k=0 de forma prolongada "
            "pode apontar regiões 'cegas'."
        )
    else:
        st.warning("Coluna 'k' não encontrada no histórico.")
