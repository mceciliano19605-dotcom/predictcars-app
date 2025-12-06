import textwrap
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# CONFIGURAÇÃO GERAL DO APP
# ============================================================

st.set_page_config(
    page_title="Predict Cars V15.2-HÍBRIDO — QDS REAL",
    layout="wide",
)

# ============================================================
# SESSION STATE
# ============================================================

def init_session_state() -> None:
    """Inicializa chaves principais na sessão, se ainda não existirem."""
    defaults = {
        "df": None,              # histórico original
        "df_limpo": None,        # histórico pós-tratamento de ruído Tipo A (V15.1)
        "n_passageiros": None,
        "fonte_historico": None,
        "historico_texto_bruto": "",
        "historico_csv_nome": None,
        "ruido_stats": None,     # métricas antes/depois do tratamento de ruído
        "qds_stats": None,       # métricas de QDS REAL (V15.2)
        "qds_config": None,      # parâmetros usados pelo QDS
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_df_base() -> Optional[pd.DataFrame]:
    """Retorna o DataFrame a ser usado pelo motor:

    Ordem de prioridade:
      1) df_limpo (pós-tratamento de ruído Tipo A)
      2) df original
    """
    df_limpo = st.session_state.get("df_limpo", None)
    if df_limpo is not None and not df_limpo.empty:
        return df_limpo
    return st.session_state.get("df", None)


# ============================================================
# UTILITÁRIOS DE ENTRADA FLEX ULTRA
# ============================================================

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
            continue

        serie_id: Optional[str] = None
        inicio_numeros = 0
        if partes[0].upper().startswith("C") and len(partes[0]) > 1:
            serie_id = partes[0]
            inicio_numeros = 1

        numeros = partes[inicio_numeros:]
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

    linhas_norm = []
    for i, reg in enumerate(registros, start=1):
        base = {}
        base["idx"] = i
        base["serie_id"] = reg["serie_id"] if reg["serie_id"] is not None else f"C{i}"
        for j, val in enumerate(reg["passageiros"], start=1):
            base[f"n{j}"] = val
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
    import io

    buffer = io.StringIO(conteudo.decode("utf-8", errors="ignore"))
    amostra = buffer.read(2048)
    buffer.seek(0)

    sep = ";" if amostra.count(";") >= amostra.count(",") else ","

    df_raw = pd.read_csv(buffer, sep=sep, header=None)
    df = df_raw.copy()

    if df.shape[1] < 2:
        raise ValueError("CSV parece ter colunas insuficientes para histórico válido.")

    primeira_col = df.iloc[:, 0].astype(str)

    def _parece_id_serie(x: str) -> bool:
        x = x.strip().upper()
        return x.startswith("C") and len(x) > 1

    if primeira_col.apply(_parece_id_serie).all():
        serie_ids = primeira_col
        df_valores = df.iloc[:, 1:].copy()
    else:
        serie_ids = pd.Series([f"C{i}" for i in range(1, len(df) + 1)])
        df_valores = df

    if df_valores.shape[1] < 2:
        raise ValueError(
            "Não foi possível identificar passageiros + k no CSV (colunas insuficientes)."
        )

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
    k_zeros = int((df["k"] == 0).sum()) if "k" in df.columns else 0
    k_pos = int((df["k"] > 0).sum()) if "k" in df.columns else 0
    return (
        f"Séries: {n_series} | Passageiros por série (máx): {n_pass} | "
        f"k = 0 em {k_zeros} séries | k > 0 em {k_pos} séries"
    )


init_session_state()

# ============================================================
# LAYOUT PRINCIPAL — CABEÇALHO
# ============================================================

st.markdown(
    """# 🚗 Predict Cars V15.2-HÍBRIDO — QDS REAL
Núcleo V14-FLEX ULTRA + Modo TURBO++ ULTRA Anti-Ruído + Ruído Tipo A/B + QDS REAL + Replay LIGHT/ULTRA + k & k*.
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
            "📥 Histórico — Entrada FLEX ULTRA (V15.2-HÍBRIDO)",
            "🔍 Pipeline V14-FLEX ULTRA (V15.2)",
            "📈 QDS REAL — Qualidade Dinâmica da Estrada (V15.2)",
            "🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15.2)",
            "💡 Replay LIGHT",
            "📅 Replay ULTRA",
            "🎯 Replay ULTRA Unitário",
            "🚨 Monitor de Risco (k & k*)",
            "📊 Ruído Condicional (V15.2)",
            "🧹 Tratamento de Ruído Tipo A+B (V15.2)",
            "🧪 Testes de Confiabilidade REAL",
        ),
    )

    st.markdown("---")
    df_base = get_df_base()
    if df_base is not None:
        st.markdown("### 📊 Resumo rápido do histórico (base atual):")
        st.info(resumo_rapido_historico(df_base))
        if st.session_state.get("df_limpo", None) is not None:
            st.caption("✔ Histórico pós-tratamento de ruído (Tipo A) em uso.")
        if st.session_state.get("qds_stats", None) is not None:
            st.caption("✔ QDS REAL já calculado para este histórico.")


# ============================================================
# PAINEL 1 — HISTÓRICO (ENTRADA FLEX ULTRA) V15.2
# ============================================================

if painel == "📥 Histórico — Entrada FLEX ULTRA (V15.2-HÍBRIDO)":
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
            key="uploader_v152_csv",
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
            if st.button(
                "Processar texto",
                type="primary",
                key="btn_processar_texto_v152",
            ):
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
            st.session_state["historico_csv_nome"] = getattr(arquivo_csv, "name", None)
        elif fonte_escolhida == "Texto colado" and df_texto is not None:
            df_final = df_texto
            st.session_state["fonte_historico"] = "texto"
        else:
            df_final = None

        if df_final is not None:
            st.session_state["df"] = df_final
            st.session_state["df_limpo"] = None  # reset do tratamento de ruído
            st.session_state["ruido_stats"] = None
            st.session_state["qds_stats"] = None
            st.session_state["qds_config"] = None

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
> para executar o **Pipeline V14-FLEX ULTRA (V15.2)**, **QDS REAL**,  
> **TURBO++ ULTRA Anti-Ruído**, **Replay LIGHT/ULTRA**,  
> **Monitor de Risco**, **Ruído Condicional** e **Tratamento de Ruído Tipo A+B**.
"""
)
# ============================================================
# PARTE 2/4 — PIPELINE, CLIMA, k*, S1–S5, LEQUES BASE
# ============================================================

# ------------------------------------------------------------
# NORMALIZAÇÃO FLEXÍVEL DE UMA SÉRIE (n1..nN)
# ------------------------------------------------------------

def normalizar_serie(serie: List[int]) -> List[int]:
    """Normaliza uma série mantendo estrutura relativa (conversão para int)."""
    try:
        return [int(x) for x in serie]
    except Exception:
        return [int(float(x)) for x in serie]


def extrair_passageiros_df(df: pd.DataFrame) -> np.ndarray:
    """Extrai matriz (S × N) de passageiros flexível a partir do DataFrame."""
    cols_pass = [c for c in df.columns if c.startswith("n")]
    return df[cols_pass].astype(float).to_numpy()


def obter_k_df(df: pd.DataFrame) -> np.ndarray:
    """Extrai vetor k."""
    return df["k"].astype(int).to_numpy()


# ------------------------------------------------------------
# JANELA LOCAL — Recorte para análise (barômetro, k*, S1..S5, QDS)
# ------------------------------------------------------------

def selecionar_janela(df: pd.DataFrame, janela: int = 40) -> pd.DataFrame:
    """Retorna as últimas N séries para análise local."""
    if len(df) <= janela:
        return df.copy()
    return df.iloc[-janela:].copy()


# ------------------------------------------------------------
# BARÔMETRO LOCAL / CLIMA — V14-FLEX ULTRA (base para V15.2)
# ------------------------------------------------------------

def calcular_barometro(df_janela: pd.DataFrame) -> dict:
    """Cria um resumo de ambiente:
    - dispersão média entre séries consecutivas
    - distribuição de k
    """
    cols_pass = [c for c in df_janela.columns if c.startswith("n")]

    matriz = df_janela[cols_pass].astype(float).to_numpy()
    if matriz.shape[0] <= 1:
        media_dif = 0.0
    else:
        diffs = np.abs(np.diff(matriz, axis=0)).mean(axis=1)
        media_dif = float(np.mean(diffs))

    if "k" in df_janela.columns:
        k_vals = df_janela["k"].astype(int).to_numpy()
        pct_k_pos = float(100 * np.mean(k_vals > 0))
    else:
        pct_k_pos = 0.0

    return {
        "media_diferenca": media_dif,
        "pct_k_positivo": pct_k_pos,
    }


# ------------------------------------------------------------
# k* LOCAL — SENTINELA (V15.2, baseado no barômetro)
# ------------------------------------------------------------

def avaliar_k_estrela(barometro: dict) -> Tuple[str, str]:
    """Define regime local do ambiente baseado no barômetro.

    Usa:
      - média das diferenças entre séries consecutivas
      - percentual de k > 0 na janela
    """
    media_dif = barometro["media_diferenca"]
    pct_k_pos = barometro["pct_k_positivo"]

    # Sensibilidade V15.2 (ligeiramente mais rígida que V15.1)
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
    - k* (estado + mensagem)
    """
    janela = selecionar_janela(df, janela=40)
    bar = calcular_barometro(janela)

    if bar["media_diferenca"] < 10:
        clima = "🟢 Estrada estável — poucas variações bruscas."
    elif bar["media_diferenca"] < 20:
        clima = "🟡 Estrada com perturbação moderada."
    else:
        clima = "🔴 Estrada turbulenta — risco elevado."

    k_estado, k_msg = avaliar_k_estrela(bar)
    return clima, k_estado, bar, (k_estado, k_msg)


# ------------------------------------------------------------
# S1–S5 DO PIPELINE V14-FLEX ULTRA (núcleo leve, mesmo jeitão V15.1)
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
    # Mantém o jeitão: nesta versão, aplicamos apenas identidade (núcleo leve),
    # mas o formato estrutural é preservado para expansões futuras.
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


def executar_s1_a_s5(df: pd.DataFrame) -> pd.DataFrame:
    """Executa S1–S5 encadeados, preservando o jeitão V14/V15."""
    s1 = etapa_s1(df)
    s2 = etapa_s2(df, s1)
    s3 = etapa_s3(df, s2)
    s4 = etapa_s4(df, s3)
    s5 = etapa_s5(df, s4)
    return s5


# ------------------------------------------------------------
# GERADOR DE SÉRIES BASE (LEQUE ORIGINAL)
# ------------------------------------------------------------

def gerar_series_base(
    df: pd.DataFrame,
    regime_state: str,
    n_out: int = 200,
) -> List[List[int]]:
    """Gera o leque ORIGINAL baseado nas faixas S1–S5.

    Mantém o mesmo jeitão do V15.1: usa as faixas (faixa_min/faixa_max)
    para amostrar valores por passageiro.
    """
    faixas = executar_s1_a_s5(df)
    cols_pass = [c for c in df.columns if c.startswith("n")]
    n_pass = len(cols_pass)

    faixas_np = faixas.to_numpy()
    faixa_min = faixas_np[:, 0]
    faixa_max = faixas_np[:, 1]

    saidas: List[List[int]] = []
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
# LEQUE CORRIGIDO (S6/S7 estrutural simples)
# ------------------------------------------------------------

def gerar_leque_corrigido(
    df: pd.DataFrame,
    regime_state: str,
    n_out: int = 200,
) -> List[List[int]]:
    """Gera o leque CORRIGIDO usando média + desvio global simples.

    Mantém o mesmo jeitão do V15.1:
      - usa média global e desvio global dos passageiros
      - gera séries em torno desses valores
    """
    cols_pass = [c for c in df.columns if c.startswith("n")]
    n_pass = len(cols_pass)

    saidas: List[List[int]] = []
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


def unir_leques(leque1: List[List[int]], leque2: List[List[int]]) -> List[List[int]]:
    """Une leques ORIGINAL e CORRIGIDO em um único MIX."""
    return leque1 + leque2


# ------------------------------------------------------------
# TABELA FLAT — n1..nN + coluna 'series'
# ------------------------------------------------------------

def build_flat_series_table(leque: List[List[int]]) -> pd.DataFrame:
    """Constrói tabela flat com:
      - id
      - series (lista original)
      - n1..nN (colunas individuais)
    """
    linhas = []
    for i, serie in enumerate(leque, start=1):
        base = {}
        base["id"] = i
        base["series"] = normalizar_serie(serie)
        for j, val in enumerate(serie, start=1):
            base[f"n{j}"] = val
        linhas.append(base)
    df_flat = pd.DataFrame(linhas).set_index("id")
    return df_flat


# ============================================================
# NÚCLEO QDS REAL — FUNÇÕES BÁSICAS (V15.2)
# ============================================================

def calcular_qds_estrada(
    df: pd.DataFrame,
    window_tam: int = 40,
) -> Tuple[pd.DataFrame, dict]:
    """Calcula QDS REAL (Qualidade Dinâmica da Série) ao longo da estrada.

    Para cada ponto i:
      - considera uma janela [i - window_tam + 1, i]
      - calcula:
          * dispersão média local (media_diferenca_local)
          * pct_k_positivo_local
          * k_atual (da série i)
      - combina em um score QDS (0 a 100)

    Retorna:
      - df_qds com colunas:
          idx_base, serie_id, qds_score,
          media_diferenca_local, pct_k_positivo_local, k_atual, nivel_qds
      - stats agregadas
    """
    if df is None or df.empty:
        raise ValueError("Histórico vazio para cálculo de QDS.")

    df = df.copy()
    n = len(df)
    if n < 2:
        raise ValueError("Histórico muito pequeno para cálculo de QDS.")

    if "k" not in df.columns:
        df["k"] = 0

    idx_list = []
    serie_ids = []
    disp_list = []
    pct_k_list = []
    k_atual_list = []

    for pos in range(n):
        i = pos + 1  # 1-based
        ini = max(0, pos - window_tam + 1)
        janela = df.iloc[ini : pos + 1].copy()
        bar = calcular_barometro(janela)
        media_dif_loc = bar["media_diferenca"]
        pct_k_pos_loc = bar["pct_k_positivo"]
        k_atual = int(df["k"].iloc[pos])

        idx_list.append(i)
        serie_ids.append(df["serie_id"].iloc[pos] if "serie_id" in df.columns else f"C{i}")
        disp_list.append(media_dif_loc)
        pct_k_list.append(pct_k_pos_loc)
        k_atual_list.append(k_atual)

    disp_arr = np.array(disp_list)
    kpos_arr = np.array(pct_k_list)
    k_atual_arr = np.array(k_atual_list)

    # Normalização dos componentes
    eps = 1e-6

    disp_min = float(disp_arr.min())
    disp_max = float(disp_arr.max())
    if disp_max - disp_min < eps:
        disp_score = np.ones_like(disp_arr)
    else:
        # Menor dispersão => melhor (score mais alto)
        disp_score = 1.0 - (disp_arr - disp_min) / (disp_max - disp_min + eps)

    kpos_min = float(kpos_arr.min())
    kpos_max = float(kpos_arr.max())
    if kpos_max - kpos_min < eps:
        kpos_score = np.ones_like(kpos_arr)
    else:
        # Maior pct_k_pos => melhor (score mais alto)
        kpos_score = (kpos_arr - kpos_min) / (kpos_max - kpos_min + eps)

    # k_atual: penaliza levemente k=0
    k_atual_factor = np.where(k_atual_arr > 0, 1.0, 0.7)

    # Combinação ponderada (pode ser ajustada futuramente)
    raw_score = (
        0.5 * disp_score +
        0.4 * kpos_score +
        0.1 * k_atual_factor
    )

    max_raw = float(raw_score.max())
    if max_raw <= 0:
        qds_score = np.zeros_like(raw_score)
    else:
        qds_score = 100.0 * raw_score / max_raw

    # Classificação em níveis
    niveis = []
    for s in qds_score:
        if s >= 80:
            niveis.append("PREMIUM")
        elif s >= 60:
            niveis.append("BOM")
        elif s >= 40:
            niveis.append("REGULAR")
        else:
            niveis.append("RUIM")

    df_qds = pd.DataFrame(
        {
            "idx_base": idx_list,
            "serie_id": serie_ids,
            "qds_score": qds_score,
            "media_diferenca_local": disp_arr,
            "pct_k_positivo_local": kpos_arr,
            "k_atual": k_atual_arr,
            "nivel_qds": niveis,
        }
    ).set_index("idx_base")

    stats = {
        "window_tam": int(window_tam),
        "qds_media": float(df_qds["qds_score"].mean()),
        "qds_min": float(df_qds["qds_score"].min()),
        "qds_max": float(df_qds["qds_score"].max()),
        "pct_premium": float((df_qds["nivel_qds"] == "PREMIUM").mean() * 100.0),
        "pct_bom_ou_melhor": float(
            (df_qds["nivel_qds"].isin(["PREMIUM", "BOM"])).mean() * 100.0
        ),
    }

    return df_qds, stats


# ============================================================
# PAINEL 2 — Pipeline V14-FLEX ULTRA (V15.2)
# ============================================================

if painel == "🔍 Pipeline V14-FLEX ULTRA (V15.2)":

    st.markdown("## 🔍 Pipeline V14-FLEX ULTRA (V15.2)")

    df = get_df_base()
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
# PARTE 3/4 — AVALIAÇÃO (TVF + RUÍDO TIPO B), QDS REAL, TURBO
# ============================================================

# ------------------------------------------------------------
# TRATAMENTO DE RUÍDO TIPO A — (já usado no V15.1, reaproveitado aqui)
# ------------------------------------------------------------

def calcular_metrica_ruido_global(df: pd.DataFrame) -> dict:
    """Mede ruído global aproximado: dispersão média entre séries consecutivas."""
    cols_pass = [c for c in df.columns if c.startswith("n")]
    if not cols_pass or len(df) <= 1:
        return {"media_diferenca": 0.0}

    matriz = df[cols_pass].astype(float).to_numpy()
    diffs = np.abs(np.diff(matriz, axis=0)).mean(axis=1)
    media_dif = float(np.mean(diffs))
    return {"media_diferenca": media_dif}


def aplicar_tratamento_ruido_tipo_a(
    df: pd.DataFrame,
    window: int = 7,
    limiar_sigma: float = 3.0,
) -> Tuple[pd.DataFrame, dict]:
    """Aplica suavização condicional (Tipo A) sobre n1..nN.

    Usa mediana + MAD (desvio absoluto mediano) em janelas deslizantes.
    Pontos cujo desvio é maior que 'limiar_sigma' * MAD são substituídos
    pela mediana local.

    Retorna:
      - df_limpo
      - stats com % de pontos ajustados e ruído antes/depois
    """
    df = df.copy()
    cols_pass = [c for c in df.columns if c.startswith("n")]
    if not cols_pass:
        return df, {"pct_ajustado": 0.0}

    total_pontos = len(df) * len(cols_pass)
    total_ajustes = 0

    for col in cols_pass:
        serie = df[col].astype(float)
        med = serie.rolling(window, center=True, min_periods=1).median()
        diff = (serie - med).abs()
        mad = diff.rolling(window, center=True, min_periods=1).median()

        eps = 1e-6
        z = diff / (mad + eps)
        mask = z > limiar_sigma

        total_ajustes += int(mask.sum())
        df[col] = serie.where(~mask, med)

    ruido_antes = calcular_metrica_ruido_global(df=st.session_state.get("df", df))
    ruido_depois = calcular_metrica_ruido_global(df=df)

    pct_ajustado = 0.0
    if total_pontos > 0:
        pct_ajustado = 100.0 * total_ajustes / total_pontos

    stats = {
        "pct_ajustado": pct_ajustado,
        "media_dif_antes": float(ruido_antes.get("media_diferenca", 0.0)),
        "media_dif_depois": float(ruido_depois.get("media_diferenca", 0.0)),
        "window": int(window),
        "limiar_sigma": float(limiar_sigma),
    }
    return df, stats


# ------------------------------------------------------------
# AVALIAÇÃO DAS SÉRIES — TVF + RUÍDO TIPO B
# ------------------------------------------------------------

def avaliar_series_candidatas(
    flat_df: pd.DataFrame, df_hist: pd.DataFrame
) -> pd.DataFrame:
    """Atribui confiança (TVF) às séries candidatas com ajuste de ruído Tipo B.

    Tipo B:
      - mede dispersão interna da série candidata (std)
      - penaliza séries muito "espalhadas" (ruído interno alto)
      - combina proximidade da última série histórica + fator anti-ruído
    """
    if flat_df is None or flat_df.empty:
        return flat_df

    flat_df = flat_df.copy()

    cols_pass_hist = [c for c in df_hist.columns if c.startswith("n")]
    cols_pass_cand = [c for c in flat_df.columns if c.startswith("n")]

    if not cols_pass_hist or not cols_pass_cand:
        return flat_df

    n_common = min(len(cols_pass_hist), len(cols_pass_cand))
    cols_hist_use = cols_pass_hist[:n_common]
    cols_cand_use = cols_pass_cand[:n_common]

    ultima = df_hist[cols_hist_use].iloc[-1].astype(float).to_numpy()

    dist_list = []
    std_list = []

    for _, row in flat_df[cols_cand_use].iterrows():
        v = row.astype(float).to_numpy()
        d = float(np.linalg.norm(v - ultima))
        dist_list.append(d)
        std_list.append(float(np.std(v)))

    dists = np.array(dist_list)
    stds = np.array(std_list)

    # Proximidade (distância reversa)
    if np.all(dists == 0):
        score_prox = np.ones_like(dists)
    else:
        score_prox = 1.0 / (1.0 + dists)

    # Fator anti-ruído (std): menor std → maior fator
    if np.all(stds == stds[0]):
        ruido_fator = np.ones_like(stds)
    else:
        std_min = float(stds.min())
        std_max = float(stds.max())
        denom = max(std_max - std_min, 1e-6)
        ruido_fator = 1.0 - (stds - std_min) / denom  # entre 0 e 1

    # Score final combinando proximidade + anti-ruído
    score_raw = score_prox * ruido_fator

    max_score = float(score_raw.max()) if len(score_raw) else 1.0
    if max_score <= 0:
        conf_pct = np.zeros_like(score_raw)
    else:
        conf_pct = 100.0 * score_raw / max_score

    flat_df["score_prox"] = score_prox
    flat_df["ruido_fator"] = ruido_fator
    flat_df["conf_pct"] = conf_pct
    flat_df["TVF"] = conf_pct

    return flat_df.sort_values(by="TVF", ascending=False)


# ------------------------------------------------------------
# LIMITADOR POR MODO DE SAÍDA
# ------------------------------------------------------------

def limit_by_mode(
    flat_df: pd.DataFrame,
    regime_state: str,
    output_mode: str,
    n_series_fixed: int,
    min_conf_pct: float,
) -> pd.DataFrame:
    """Aplica o modo de geração do leque de saída."""
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
        if regime_state == "estavel":
            n = 10
        elif regime_state == "atencao":
            n = 20
        else:
            n = 30
        n = min(n, len(df))
        df = df.sort_values(by="TVF", ascending=False).head(n)

    return df.reset_index(drop=True)


# ------------------------------------------------------------
# MONTAGEM COMPLETA DO LEQUE TURBO++ ULTRA (V15.2)
# ------------------------------------------------------------

def montar_previsao_turbo_ultra(
    df_hist: pd.DataFrame,
    regime_state: str,
    output_mode: str,
    n_series_fixed: int,
    min_conf_pct: float,
    n_out_base: int = 200,
) -> pd.DataFrame:
    """Monta o leque TURBO++ ULTRA com ruído Tipo B integrado.

    Etapas:
      - gera leque ORIGINAL (S1–S5)
      - gera leque CORRIGIDO (S6/S7 estrutural simples)
      - une em MIX
      - avalia TVF + ruído Tipo B
      - aplica limitador por modo de saída
    """
    leque_original = gerar_series_base(df_hist, regime_state, n_out=n_out_base)
    flat_original = build_flat_series_table(leque_original)
    flat_original["origem"] = "ORIGINAL"

    leque_corrigido = gerar_leque_corrigido(df_hist, regime_state, n_out=n_out_base)
    flat_corr = build_flat_series_table(leque_corrigido)
    flat_corr["origem"] = "CORRIGIDO"

    flat_mix = pd.concat([flat_original, flat_corr], ignore_index=True)

    flat_mix = avaliar_series_candidatas(flat_mix, df_hist)

    df_controlado = limit_by_mode(
        flat_mix, regime_state, output_mode, n_series_fixed, min_conf_pct
    )

    return df_controlado


# ------------------------------------------------------------
# CONTEXTO k* PARA IMPRESSÃO NA PREVISÃO
# ------------------------------------------------------------

def contexto_k_previsao(k_estado: str) -> str:
    if k_estado == "estavel":
        return "🟢 k*: Ambiente estável — previsão em regime normal."
    elif k_estado == "atencao":
        return "🟡 k*: Pré-ruptura residual — usar previsão com atenção."
    else:
        return "🔴 k*: Ambiente crítico — usar previsão com cautela máxima."


# ============================================================
# PAINEL — 📈 QDS REAL — Qualidade Dinâmica da Estrada (V15.2)
# ============================================================

if painel == "📈 QDS REAL — Qualidade Dinâmica da Estrada (V15.2)":

    st.markdown("## 📈 QDS REAL — Qualidade Dinâmica da Estrada (V15.2)")
    st.markdown(
        "Mede a **Qualidade Dinâmica da Série** ao longo da estrada, combinando:\n\n"
        "- dispersão local (diferença média entre séries)\n"
        "- percentual de k>0 na janela\n"
        "- k atual da série\n\n"
        "Produz um score QDS (0–100) e classifica trechos como **PREMIUM / BOM / REGULAR / RUIM**."
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

    if st.button("Calcular QDS REAL da estrada", type="primary", key="btn_qds_real_v152"):
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
        ult = df_qds.tail(200).copy()
        st.dataframe(ult)

        with st.expander("Visualização simplificada do QDS (tabela completa)", expanded=False):
            st.dataframe(df_qds)

        st.info(
            "Trechos **PREMIUM** indicam janelas onde o TURBO++ ULTRA tende a operar\n"
            "com maior consistência. Trechos **RUIM** indicam ambientes de baixa qualidade\n"
            "da estrada, mesmo após tratamento de ruído."
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
                "gerar o mapa de qualidade dinâmica."
            )


# ============================================================
# PAINEL — 🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15.2)
# ============================================================

if painel == "🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15.2)":

    st.markdown("## 🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15.2)")
    st.markdown(
        "Núcleo V14-FLEX ULTRA + Leque ORIGINAL/CORRIGIDO/MISTO + TVF + k* adaptativo + Ruído Tipo B + QDS REAL (contexto)."
    )

    df = get_df_base()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel de Entrada FLEX ULTRA.")
        st.stop()

    clima, k_estado, bar, (k_st, k_msg) = detectar_regime(df)

    col_esq, col_dir = st.columns(2)

    with col_esq:
        st.markdown("### 🌡️ Clima da Estrada (base atual)")
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
            key="turbo_modo_v152",
        )

    with col_qtd:
        n_series_fixed = st.number_input(
            "Quantidade total de séries (se modo for 'Quantidade fixa')",
            min_value=1,
            max_value=200,
            value=25,
            step=1,
            key="turbo_qtd_v152",
        )

    with col_conf:
        min_conf_pct = st.slider(
            "Confiabilidade mínima (%) (se modo for 'Confiabilidade mínima')",
            min_value=0,
            max_value=100,
            value=30,
            step=1,
            key="turbo_conf_v152",
        )

    st.markdown("---")

    if st.button("Gerar Leque TURBO++ ULTRA", type="primary", key="btn_turbo_v152"):
        with st.spinner("Gerando leque TURBO++ ULTRA, avaliando TVF e ruído Tipo B..."):
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

            st.markdown("---")
            st.markdown("### 🎯 Previsão Final TURBO++ ULTRA")

            melhor = df_turbo.iloc[0]
            serie_final = melhor.get("series", None)

            if serie_final is not None:
                st.code(" ".join(str(x) for x in serie_final), language="text")
                st.markdown(contexto_k_previsao(k_estado))
                st.caption(
                    f"Origem = {melhor.get('origem', 'MIX')}, "
                    f"TVF ≈ {melhor.get('TVF', 0):.1f}, "
                    f"Conf. ≈ {melhor.get('conf_pct', 0):.1f}%, "
                    f"Ruído fator ≈ {melhor.get('ruido_fator', 0):.2f}."
                )
            else:
                st.warning("A coluna 'series' não foi encontrada no leque gerado.")
# ============================================================
# PARTE 4/4 — REPLAYS, RISCO, RUÍDO, TRATAMENTO A+B, CONFIABILIDADE
# ============================================================

# ============================================================
# PAINEL — 💡 Replay LIGHT
# ============================================================

if painel == "💡 Replay LIGHT":

    st.markdown("## 💡 Replay LIGHT (com ruído Tipo B e QDS no contexto)")
    st.markdown(
        "Simula o que o TURBO++ ULTRA teria feito em um ponto específico do histórico "
        "(já podendo usar df_limpo + usando o mesmo motor com TVF + ruído Tipo B)."
    )

    df_original = st.session_state.get("df", None)
    df_base = get_df_base()

    if df_original is None or df_original.empty:
        st.warning("Carregue o histórico primeiro no painel de Entrada FLEX ULTRA.")
        st.stop()

    n_total = len(df_original)
    st.markdown(f"Histórico atual contém **{n_total} séries**.")

    idx_alvo = st.number_input(
        "Escolha o índice alvo (1 = primeira série carregada):",
        min_value=1,
        max_value=n_total,
        value=n_total,
        step=1,
        key="replay_light_idx_v152",
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
            key="replay_light_modo_v152",
        )

    with col_qtd:
        n_series_fixed = st.number_input(
            "Quantidade total de séries (se modo for 'Quantidade fixa')",
            min_value=1,
            max_value=200,
            value=25,
            step=1,
            key="replay_light_qtd_v152",
        )

    with col_conf:
        min_conf_pct = st.slider(
            "Confiabilidade mínima (%) (se modo for 'Confiabilidade mínima')",
            min_value=0,
            max_value=100,
            value=30,
            step=1,
            key="replay_light_conf_v152",
        )

    if st.button("Rodar Replay LIGHT", key="btn_replay_light_v152"):
        df_sub_base = df_base.iloc[:idx_alvo].copy()
        serie_id = df_sub_base.iloc[-1].get("serie_id", f"C{idx_alvo}")
        clima, k_estado, bar, (k_st, k_msg) = detectar_regime(df_sub_base)

        st.markdown("### ℹ️ Contexto do ponto alvo (base atual)")
        st.write(f"ID alvo: **{serie_id}** (índice {idx_alvo})")
        st.info(clima)
        st.info(k_msg)

        with st.spinner("Gerando leque TURBO++ ULTRA para o Replay LIGHT..."):
            df_replay = montar_previsao_turbo_ultra(
                df_hist=df_sub_base,
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
                    f"Origem = {melhor.get('origem', 'MIX')}, "
                    f"TVF ≈ {melhor.get('TVF', 0):.1f}, "
                    f"Conf. ≈ {melhor.get('conf_pct', 0):.1f}%, "
                    f"Ruído fator ≈ {melhor.get('ruido_fator', 0):.2f}."
                )
            else:
                st.warning("A coluna 'series' não foi encontrada no leque gerado.")


# ============================================================
# PAINEL — 📅 Replay ULTRA (intervalo)
# ============================================================

if painel == "📅 Replay ULTRA":

    st.markdown("## 📅 Replay ULTRA (intervalo, com ruído Tipo B e QDS no contexto)")

    df = get_df_base()
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
            key="replay_ultra_ini_v152",
        )
    with col_b:
        idx_fim = st.number_input(
            "Índice final do intervalo:",
            min_value=int(idx_ini),
            max_value=n_total,
            value=n_total,
            step=1,
            key="replay_ultra_fim_v152",
        )

    output_mode = st.radio(
        "Modo de geração do Leque (para o Replay ULTRA):",
        (
            "Automático (por regime)",
            "Quantidade fixa",
            "Confiabilidade mínima",
        ),
        key="replay_ultra_modo_v152",
    )

    n_series_fixed = st.number_input(
        "Quantidade total de séries (se modo for 'Quantidade fixa')",
        min_value=1,
        max_value=200,
        value=15,
        step=1,
        key="replay_ultra_qtd_v152",
    )

    min_conf_pct = st.slider(
        "Confiabilidade mínima (%) (se modo for 'Confiabilidade mínima')",
        min_value=0,
        max_value=100,
        value=30,
        step=1,
        key="replay_ultra_conf_v152",
    )

    if st.button("Rodar Replay ULTRA (intervalo)", key="btn_replay_ultra_v152"):
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
                    previsao = ""
                    tvf = None
                    conf = None
                    ruido_fator = None
                else:
                    best = df_rep.iloc[0]
                    serie_vals = best.get("series", None)
                    previsao = " ".join(str(x) for x in serie_vals) if serie_vals else ""
                    tvf = best.get("TVF", None)
                    conf = best.get("conf_pct", None)
                    ruido_fator = best.get("ruido_fator", None)

                registros.append(
                    {
                        "idx": i,
                        "serie_id": serie_id,
                        "clima": clima,
                        "k_estado": k_estado,
                        "previsao": previsao,
                        "TVF": tvf,
                        "conf_pct": conf,
                        "ruido_fator": ruido_fator,
                    }
                )

        df_replay_ultra = pd.DataFrame(registros)
        st.success("Replay ULTRA concluído.")
        st.markdown("### 📊 Tabela de Replay ULTRA (resumo por ponto do intervalo)")
        st.dataframe(df_replay_ultra)


# ============================================================
# PAINEL — 🎯 Replay ULTRA Unitário
# ============================================================

if painel == "🎯 Replay ULTRA Unitário":

    st.markdown("## 🎯 Replay ULTRA Unitário (foco total + ruído Tipo B + QDS no contexto)")

    df = get_df_base()
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
        key="replay_ultra_unit_idx_v152",
    )

    output_mode = st.radio(
        "Modo de geração do Leque (para este ponto ULTRA):",
        (
            "Automático (por regime)",
            "Quantidade fixa",
            "Confiabilidade mínima",
        ),
        key="replay_ultra_unit_modo_v152",
    )

    n_series_fixed = st.number_input(
        "Quantidade total de séries (se modo for 'Quantidade fixa')",
        min_value=1,
        max_value=200,
        value=20,
        step=1,
        key="replay_ultra_unit_qtd_v152",
    )

    min_conf_pct = st.slider(
        "Confiabilidade mínima (%) (se modo for 'Confiabilidade mínima')",
        min_value=0,
        max_value=100,
        value=40,
        step=1,
        key="replay_ultra_unit_conf_v152",
    )

    if st.button("Rodar Replay ULTRA Unitário", key="btn_replay_ultra_unit_v152"):
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
                    f"Origem = {best.get('origem', 'MIX')}, "
                    f"TVF ≈ {best.get('TVF', 0):.1f}, "
                    f"Conf. ≈ {best.get('conf_pct', 0):.1f}%, "
                    f"Ruído fator ≈ {best.get('ruido_fator', 0):.2f}."
                )
            else:
                st.warning("A coluna 'series' não foi encontrada no leque gerado.")


# ============================================================
# PAINEL — 🚨 Monitor de Risco (k & k*)
# ============================================================

if painel == "🚨 Monitor de Risco (k & k*)":

    st.markdown("## 🚨 Monitor de Risco (k & k*)")

    df_original = st.session_state.get("df", None)
    df_base = get_df_base()

    if df_original is None or df_original.empty:
        st.warning("Carregue o histórico primeiro no painel de Entrada FLEX ULTRA.")
        st.stop()

    clima, k_estado, bar, (k_st, k_msg) = detectar_regime(df_base)

    st.markdown("### 🌡️ Clima atual da estrada (base atual)")
    st.info(clima)

    st.markdown("### ⭐ Sentinela k* (estado atual)")
    st.info(k_msg)

    st.markdown("### 📊 Barômetro resumido")
    st.write(bar)

    if "k" in df_original.columns:
        st.markdown("### 📈 Distribuição de k no histórico original")
        st.write(df_original["k"].value_counts().sort_index())

        st.markdown("### 🔎 Estatísticas básicas de k (histórico original)")
        st.write(
            {
                "k mínimo": int(df_original["k"].min()),
                "k máximo": int(df_original["k"].max()),
                "k médio": float(df_original["k"].mean()),
            }
        )
    else:
        st.warning("Coluna 'k' não encontrada no histórico original.")

    stats_qds = st.session_state.get("qds_stats", None)
    if stats_qds is not None:
        st.markdown("### 📈 Resumo de QDS REAL (último cálculo)")
        st.write(
            {
                "QDS médio": stats_qds["qds_media"],
                "QDS mínimo": stats_qds["qds_min"],
                "QDS máximo": stats_qds["qds_max"],
                "% de trechos PREMIUM": f"{stats_qds['pct_premium']:.2f}%",
                "% de trechos BOM ou melhor": f"{stats_qds['pct_bom_ou_melhor']:.2f}%",
            }
        )
        st.info(
            "QDS REAL complementa o k/k*, mostrando **onde** a estrada está mais saudável "
            "para o TURBO++ ULTRA operar."
        )


# ============================================================
# PAINEL — 📊 Ruído Condicional (V15.2)
# ============================================================

if painel == "📊 Ruído Condicional (V15.2)":

    st.markdown("## 📊 Ruído Condicional (V15.2)")
    st.markdown(
        "Monitor para enxergar como a estrada reage a diferentes regimes, "
        "abrindo espaço para filtros anti-ruído condicionais ao ambiente.\n\n"
        "Agora integrado ao contexto de QDS REAL."
    )

    df_original = st.session_state.get("df", None)
    df_base = get_df_base()

    if df_original is None or df_original.empty:
        st.warning("Carregue o histórico primeiro no painel de Entrada FLEX ULTRA.")
        st.stop()

    clima, k_estado, bar, (k_st, k_msg) = detectar_regime(df_base)

    st.markdown("### 🌡️ Clima e k* (base atual)")
    st.info(clima)
    st.info(k_msg)

    if "k" in df_original.columns:
        st.markdown("### 🔎 Indicadores simples de ruído (versão inicial)")
        k_vals = df_original["k"].astype(int)
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
            "Trechos com QDS alto e k* estável tendem a ser regiões com **ruído efetivo "
            "mais controlado**, ideais para estratégias mais agressivas (como 6 acertos)."
        )


# ============================================================
# PAINEL — 🧹 Tratamento de Ruído Tipo A+B (V15.2)
# ============================================================

if painel == "🧹 Tratamento de Ruído Tipo A+B (V15.2)":

    st.markdown("## 🧹 Tratamento de Ruído Tipo A+B (V15.2)")
    st.markdown(
        "Tipo A: limpeza/suavização do histórico (df_limpo).\n\n"
        "Tipo B: penalização de séries ruidosas no TURBO++ (já integrada ao TVF).\n\n"
        "V15.2: este painel também alimenta o contexto do QDS REAL."
    )

    df_original = st.session_state.get("df", None)
    if df_original is None or df_original.empty:
        st.warning("Carregue o histórico primeiro no painel de Entrada FLEX ULTRA.")
        st.stop()

    st.markdown("### 🔎 Situação atual do ruído (histórico original)")
    ruido_orig = calcular_metrica_ruido_global(df_original)
    st.write(
        {
            "Dispersão média entre séries (original)": float(
                ruido_orig.get("media_diferenca", 0.0)
            )
        }
    )

    st.markdown("---")
    st.markdown("### ⚙️ Parâmetros do Tratamento de Ruído Tipo A")

    col_w, col_sig = st.columns(2)
    with col_w:
        window = st.slider(
            "Janela (tamanho da vizinhança)",
            min_value=3,
            max_value=31,
            value=7,
            step=2,
        )
    with col_sig:
        limiar_sigma = st.slider(
            "Limiar de ruído (multiplicador de MAD)",
            min_value=2.0,
            max_value=6.0,
            value=3.0,
            step=0.5,
        )

    if st.button("Aplicar Tratamento de Ruído Tipo A", type="primary", key="btn_ruido_tipo_a_v152"):
        with st.spinner("Aplicando suavização condicional (Tipo A) ao histórico..."):
            df_limpo, stats = aplicar_tratamento_ruido_tipo_a(
                df_original,
                window=int(window),
                limiar_sigma=float(limiar_sigma),
            )

        st.session_state["df_limpo"] = df_limpo
        st.session_state["ruido_stats"] = stats

        st.success("Tratamento de Ruído Tipo A aplicado com sucesso.")

        st.markdown("### 📊 Métricas antes/depois")
        st.write(
            {
                "Window": stats["window"],
                "Limiar sigma": stats["limiar_sigma"],
                "Dispersão média (antes)": stats["media_dif_antes"],
                "Dispersão média (depois)": stats["media_dif_depois"],
                "% de pontos ajustados (n1..nN)": f"{stats['pct_ajustado']:.3f}%",
            }
        )

        with st.expander("Prévia do histórico pós-tratamento (df_limpo)", expanded=False):
            st.dataframe(df_limpo.head(30))

        st.info(
            "A partir de agora, todos os painéis que usam o histórico base "
            "(Pipeline, QDS, TURBO, Replay, Ruído Condicional, etc.) passarão a "
            "usar **df_limpo** como estrada principal."
        )

    if st.session_state.get("df_limpo", None) is not None:
        st.markdown("---")
        st.markdown("### ✅ Tratamento ativo")
        st.success("Um histórico pós-ruído (df_limpo) está ativo e sendo usado pelo motor.")


# ============================================================
# PAINEL — 🧪 Testes de Confiabilidade REAL
# ============================================================

if painel == "🧪 Testes de Confiabilidade REAL":

    st.markdown("## 🧪 Testes de Confiabilidade REAL")
    st.markdown(
        "Espaço reservado para integrar QDS aprofundado, Backtest dedicado e Monte Carlo "
        "com o motor V15.2-HÍBRIDO. Nesta versão, o painel funciona como monitor conceitual, "
        "mas já lê o contexto de QDS e de ruído."
    )

    df_base = get_df_base()
    if df_base is None or df_base.empty:
        st.warning("Carregue o histórico primeiro no painel de Entrada FLEX ULTRA.")
        st.stop()

    st.markdown("### 📌 Situação atual")
    st.write(
        "• Motor TURBO++ ULTRA já produz leques com TVF + ajuste de ruído (Tipo B).\n"
        "• Tratamento de Ruído Tipo A pode reduzir turbulência do histórico (df_limpo).\n"
        "• QDS REAL já mede a qualidade dinâmica da estrada (PREMIUM / BOM / REGULAR / RUIM).\n"
        "• Replay LIGHT e Replay ULTRA permitem simular decisões ao longo da estrada.\n"
        "• A partir desses elementos, Backtest REAL / Monte Carlo Profundo poderão ser plugados."
    )

    stats_ruido = st.session_state.get("ruido_stats", None)
    if stats_ruido is not None:
        st.markdown("### 🔎 Efeito atual do Tratamento de Ruído Tipo A")
        st.write(
            {
                "Dispersão média (antes)": stats_ruido["media_dif_antes"],
                "Dispersão média (depois)": stats_ruido["media_dif_depois"],
                "% de pontos ajustados (n1..nN)": f"{stats_ruido['pct_ajustado']:.3f}%",
            }
        )

    stats_qds = st.session_state.get("qds_stats", None)
    if stats_qds is not None:
        st.markdown("### 📈 Resumo do QDS REAL (para apoiar futuros backtests)")
        st.write(
            {
                "Tamanho da janela": stats_qds["window_tam"],
                "QDS médio": stats_qds["qds_media"],
                "QDS mínimo": stats_qds["qds_min"],
                "QDS máximo": stats_qds["qds_max"],
                "% de trechos PREMIUM": f"{stats_qds['pct_premium']:.2f}%",
                "% de trechos BOM ou melhor": f"{stats_qds['pct_bom_ou_melhor']:.2f}%",
            }
        )

    st.info(
        "Este painel foi mantido no jeitão estrutural, pronto para receber as "
        "rotinas de Backtest REAL por trecho de QDS, Monte Carlo segmentado por regime, "
        "e avaliação de expectativa de acertos por faixa de qualidade da estrada "
        "nas próximas versões (V15.3, V15.4...)."
    )

