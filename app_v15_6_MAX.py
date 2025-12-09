# ============================================================
# Predict Cars V15.6 MAX
# Versão MAX: núcleo + coberturas + interseção estatística
# Pipeline V14-FLEX ULTRA + Replay LIGHT/ULTRA + TURBO++ ULTRA Anti-Ruído
# + Monitor de Risco (k & k*) + Painel de Ruído Condicional
# + Testes de Confiabilidade REAL + Modo 6 Acertos V15.6 MAX
# + Relatório Final V15.6 MAX
# ============================================================

import math
import itertools
import textwrap
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# Configuração da página (obrigatório V15.6 MAX)
# ------------------------------------------------------------
st.set_page_config(
    page_title="Predict Cars V15.6 MAX",
    page_icon="🚗",
    layout="wide",
)
# ============================================================
# Navegação oficial — Predict Cars V15.6 MAX
# ============================================================
def construir_navegacao_v156():
    st.markdown("### 🧭 Navegação — Predict Cars V15.6 MAX")

    opcoes = [
        "📥 Histórico — Entrada FLEX ULTRA",
        "🔍 Pipeline V14-FLEX ULTRA (V15.6 MAX)",
        "💡 Replay LIGHT",
        "📅 Replay ULTRA",
        "🎯 Replay ULTRA Unitário",
        "🚨 Monitor de Risco (k & k*)",
        "🧪 Testes de Confiabilidade REAL",
        "📊 Ruído Condicional (V15.6)",
        "🚀 Modo TURBO++ ULTRA Anti-Ruído",
        "🎯 Modo 6 Acertos — Execução (V15.6 MAX)",
        "📄 Relatório Final V15.6 MAX"
    ]

    painel = st.selectbox(
        "Selecione um painel:",
        opcoes,
        index=0
    )

    return painel

# ------------------------------------------------------------
# Estilos globais (mantendo jeitão denso das versões anteriores)
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .small-text {
        font-size: 0.85rem;
    }
    .very-small-text {
        font-size: 0.75rem;
    }
    .center {
        text-align: center;
    }
    .justified {
        text-align: justify;
    }
    .metric-green {
        color: #1b5e20;
        font-weight: 600;
    }
    .metric-red {
        color: #b71c1c;
        font-weight: 600;
    }
    .metric-yellow {
        color: #f57f17;
        font-weight: 600;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #eceff1;
        margin-right: 4px;
    }
    .badge-green {
        background-color: #c8e6c9;
        color: #1b5e20;
    }
    .badge-red {
        background-color: #ffcdd2;
        color: #b71c1c;
    }
    .badge-yellow {
        background-color: #fff9c4;
        color: #f57f17;
    }
    .section-title {
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
painel = construir_navegacao_v156()

# ============================================================
# CONSTANTES E CONFIGURAÇÕES GERAIS
# ============================================================

# Limite de linhas gigantes para evitar "zumbi" em trechos muito pesados
MAX_LINHAS_EXIBICAO = 500

# Coluna padrão do "k" (guardas que acertaram exatamente o carro)
K_COLNAME = "k"

# Nome padrão da coluna de índice estilo "Cxxxx"
ID_COLNAME = "ID"

# Chave de sessão principal para o histórico
SESSAO_HISTORICO = "df_historico_v156"

# Chave para metadados do histórico (n passageiros, range, etc.)
SESSAO_META = "meta_historico_v156"

# Chave para armazenar resultados de pipeline (para Replay/TURBO/M6/Relatório)
SESSAO_PIPELINE = "pipeline_v156"


# ============================================================
# FUNÇÕES AUXILIARES DE FORMATAÇÃO E UTILIDADE
# ============================================================

def wrap_text(text: str, width: int = 80) -> str:
    """Quebra textos longos para ficar mais legível em tooltips / markdown."""
    if not text:
        return ""
    return "\n".join(textwrap.wrap(text, width))


def gerar_id_series(idx: int) -> str:
    """
    Gera ID no padrão Cxxxx (ex.: C0001, C2945, etc.)
    idx é zero-based; output é 1-based na etiqueta.
    """
    return f"C{idx + 1}"


def detectar_separador(linha: str) -> str:
    """
    Detecta separador mais provável em uma linha de histórico.
    Aceita ; , espaço ou tab. Se houver ambiguidade, prioriza ';', depois ','.
    """
    if ";" in linha and "," not in linha:
        return ";"
    if "," in linha and ";" not in linha:
        return ","
    # Se tiver os dois, assumimos que ';' é o separador (padrão dos CSV antigos)
    if ";" in linha and "," in linha:
        return ";"
    # Se não encontrar, tentamos espaço
    if " " in linha:
        return " "
    # Fallback para ponto e vírgula
    return ";"


def limpar_linha(linha: str) -> str:
    """Remove espaços desnecessários e caracteres estranhos no começo/fim."""
    if linha is None:
        return ""
    return linha.strip().replace("\t", " ").replace("\r", "").replace("\n", "")


def converter_para_int(valor: Any) -> Optional[int]:
    """Converte valor para int, retornando None se não conseguir."""
    if valor is None:
        return None
    if isinstance(valor, (int, np.integer)):
        return int(valor)
    try:
        s = str(valor).strip()
        if s == "":
            return None
        return int(float(s))
    except Exception:
        return None


def extrair_passageiros_e_k(linha: str, n_passageiros: Optional[int] = None) -> Tuple[List[Optional[int]], Optional[int]]:
    """
    Converte uma linha de texto em lista de passageiros + k.
    Se n_passageiros for None, deduzimos que o último campo é o 'k'.
    Caso contrário, forçamos a quantidade de passageiros fornecida.
    """
    linha = limpar_linha(linha)
    if not linha:
        return [], None

    sep = detectar_separador(linha)
    partes = [p.strip() for p in linha.split(sep) if p.strip() != ""]

    if not partes:
        return [], None

    valores = [converter_para_int(p) for p in partes]

    if n_passageiros is None:
        # Deduzimos que o último campo é k
        if len(valores) == 0:
            return [], None
        if len(valores) == 1:
            # Linha só com k (sem passageiros) => histórico inválido
            return [], valores[0]
        passageiros = valores[:-1]
        k = valores[-1]
        return passageiros, k
    else:
        # Forçamos n_passageiros. Se tiver sobras, último é k.
        if len(valores) < n_passageiros + 1:
            # Linha incompleta
            return valores, None
        passageiros = valores[:n_passageiros]
        k = valores[n_passageiros]
        return passageiros, k


def montar_dataframe_historico(
    linhas: List[str],
    n_passageiros: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Monta o DataFrame do histórico a partir de linhas de texto.
    Mantém o jeitão FLEX: n_passageiros variável, com coluna k no final.
    """
    registros_passageiros: List[List[Optional[int]]] = []
    registros_k: List[Optional[int]] = []

    # Se n_passageiros não for fornecido, deduzimos a partir da primeira linha válida
    if n_passageiros is None:
        for linha in linhas:
            psg, k_val = extrair_passageiros_e_k(linha, n_passageiros=None)
            if psg:
                n_passageiros = len(psg)
                break

    if not n_passageiros:
        raise ValueError("Não foi possível determinar o número de passageiros por série.")

    for linha in linhas:
        psg, k_val = extrair_passageiros_e_k(linha, n_passageiros=n_passageiros)
        if not psg:
            continue
        # Se por algum motivo a linha veio com mais campos, ajustamos o tamanho
        if len(psg) < n_passageiros:
            psg = psg + [None] * (n_passageiros - len(psg))
        elif len(psg) > n_passageiros:
            psg = psg[:n_passageiros]

        registros_passageiros.append(psg)
        registros_k.append(k_val)

    if not registros_passageiros:
        raise ValueError("Nenhuma série válida foi encontrada no histórico.")

    # Cria colunas dos passageiros P0, P1, ..., P(n-1)
    col_passageiros = [f"P{i}" for i in range(n_passageiros)]
    df = pd.DataFrame(registros_passageiros, columns=col_passageiros)

    # Coluna k
    df[K_COLNAME] = registros_k

    # Coluna ID Cxxxx (1-based)
    df[ID_COLNAME] = [gerar_id_series(i) for i in range(len(df))]

    # Reorganiza colunas com ID primeiro, depois passageiros, depois k
    df = df[[ID_COLNAME] + col_passageiros + [K_COLNAME]]

    # Metadados do histórico
    meta: Dict[str, Any] = {
        "n_series": len(df),
        "n_passageiros": n_passageiros,
        "col_passageiros": col_passageiros,
        "tem_k": any(pd.notna(df[K_COLNAME])),
    }

    # Intervalo aproximado de valores (para barômetros e painéis de ruído)
    valores_psg = df[col_passageiros].values.flatten()
    valores_psg = valores_psg[~pd.isna(valores_psg)]
    if len(valores_psg) > 0:
        meta["min_val"] = int(np.nanmin(valores_psg))
        meta["max_val"] = int(np.nanmax(valores_psg))
    else:
        meta["min_val"] = None
        meta["max_val"] = None

    return df, meta


# ============================================================
# GERENCIAMENTO DE ESTADO DE SESSÃO (VERSÃO MAX)
# ============================================================

def init_session_state_v156() -> None:
    """
    Garante que todas as chaves importantes da V15.6 MAX estejam inicializadas.
    Mantém o mesmo jeitão de controle de sessão das versões anteriores,
    mas com nomes específicos da V15.6 MAX para evitar conflito.
    """
    if SESSAO_HISTORICO not in st.session_state:
        st.session_state[SESSAO_HISTORICO] = None

    if SESSAO_META not in st.session_state:
        st.session_state[SESSAO_META] = None

    if SESSAO_PIPELINE not in st.session_state:
        st.session_state[SESSAO_PIPELINE] = {
            "pipeline_pronto": False,
            "replay_light": None,
            "replay_ultra": None,
            "replay_unitario": None,
            "turbo_ultra": None,
            "risco": None,
            "ruido": None,
            "confiabilidade": None,
            "modo_6": None,
            "relatorio_final": None,
        }

    # Parametrizações auxiliares que ajudam a não travar a app
    if "v156_limite_janela" not in st.session_state:
        st.session_state["v156_limite_janela"] = 300  # tamanho máximo de janela para certos cálculos

    if "v156_debug" not in st.session_state:
        st.session_state["v156_debug"] = False


def set_historico_v156(df: pd.DataFrame, meta: Dict[str, Any]) -> None:
    """Atualiza o histórico e metadados na sessão da V15.6."""
    st.session_state[SESSAO_HISTORICO] = df
    st.session_state[SESSAO_META] = meta
    # Sempre que trocamos o histórico, invalidamos o pipeline completo
    st.session_state[SESSAO_PIPELINE] = {
        "pipeline_pronto": False,
        "replay_light": None,
        "replay_ultra": None,
        "replay_unitario": None,
        "turbo_ultra": None,
        "risco": None,
        "ruido": None,
        "confiabilidade": None,
        "modo_6": None,
        "relatorio_final": None,
    }


def get_historico_v156() -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Retorna (df, meta) do histórico da V15.6 MAX."""
    df = st.session_state.get(SESSAO_HISTORICO, None)
    meta = st.session_state.get(SESSAO_META, None)
    return df, meta


def marcar_pipeline_pronto_v156(
    replay_light: Any = None,
    replay_ultra: Any = None,
    replay_unitario: Any = None,
    turbo_ultra: Any = None,
    risco: Any = None,
    ruido: Any = None,
    confiabilidade: Any = None,
) -> None:
    """
    Marca que o pipeline principal da V15.6 MAX foi montado.
    Os detalhes de cada módulo serão preenchidos pelos painéis específicos.
    """
    pipeline = st.session_state.get(SESSAO_PIPELINE, {})
    pipeline["pipeline_pronto"] = True
    if replay_light is not None:
        pipeline["replay_light"] = replay_light
    if replay_ultra is not None:
        pipeline["replay_ultra"] = replay_ultra
    if replay_unitario is not None:
        pipeline["replay_unitario"] = replay_unitario
    if turbo_ultra is not None:
        pipeline["turbo_ultra"] = turbo_ultra
    if risco is not None:
        pipeline["risco"] = risco
    if ruido is not None:
        pipeline["ruido"] = ruido
    if confiabilidade is not None:
        pipeline["confiabilidade"] = confiabilidade
    st.session_state[SESSAO_PIPELINE] = pipeline


def atualizar_resultado_modo6_v156(resultado: Any) -> None:
    """Atualiza apenas o resultado do Modo 6 Acertos no pipeline."""
    pipeline = st.session_state.get(SESSAO_PIPELINE, {})
    pipeline["modo_6"] = resultado
    st.session_state[SESSAO_PIPELINE] = pipeline


def atualizar_relatorio_final_v156(relatorio: Any) -> None:
    """Atualiza o Relatório Final V15.6 MAX no pipeline."""
    pipeline = st.session_state.get(SESSAO_PIPELINE, {})
    pipeline["relatorio_final"] = relatorio
    st.session_state[SESSAO_PIPELINE] = pipeline


# ============================================================
# CARREGAMENTO DO HISTÓRICO — ENTRADA FLEX ULTRA (V15.6 MAX)
# ============================================================

def painel_historico_entrada_v156() -> None:
    """
    Painel de entrada do histórico (FLEX ULTRA).
    Mantém dupla forma de entrada: upload de arquivo e área de texto,
    com detecção automática de n_passageiros e k.
    """
    st.markdown("## 📥 Histórico — Entrada FLEX ULTRA (V15.6 MAX)")

    st.markdown(
        """
        <div class="small-text justified">
        Este painel permite carregar todo o histórico da estrada (séries de carros)
        tanto por <b>upload de arquivo</b> quanto por <b>copiar e colar</b> os dados.
        A V15.6 MAX mantém o modo FLEX: número variável de passageiros por carro,
        com a coluna de guardas (<code>k</code>) ao final de cada linha.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_upload, col_texto = st.columns(2)

    historico_atual, meta_atual = get_historico_v156()

    with col_upload:
        st.markdown("#### Upload de arquivo")
        arquivo = st.file_uploader(
            "Selecione um arquivo de histórico (.txt, .csv)",
            type=["txt", "csv"],
            key="v156_upload_hist",
        )

        n_passageiros_upload = st.number_input(
            "Número de passageiros por série (opcional — deixe 0 para deduzir automaticamente)",
            min_value=0,
            max_value=60,
            value=0,
            step=1,
            help="Se 0, o sistema tenta deduzir o número de passageiros a partir da primeira linha válida.",
            key="v156_n_passageiros_upload",
        )

        if arquivo is not None:
            try:
                conteudo_bruto = arquivo.read().decode("utf-8", errors="ignore")
                linhas = [l for l in conteudo_bruto.splitlines() if limpar_linha(l) != ""]
                n_psg = n_passageiros_upload if n_passageiros_upload > 0 else None
                df, meta = montar_dataframe_historico(linhas, n_passageiros=n_psg)
                set_historico_v156(df, meta)
                st.success(
                    f"Histórico carregado com sucesso via upload: {meta['n_series']} séries, "
                    f"{meta['n_passageiros']} passageiros por série."
                )
            except Exception as e:
                st.error(f"Erro ao carregar arquivo: {e}")

    with col_texto:
        st.markdown("#### Copiar e colar histórico")
        texto = st.text_area(
            "Cole aqui o histórico completo (uma série por linha, com k ao final):",
            height=260,
            key="v156_textarea",
        )

        n_passageiros_texto = st.number_input(
            "Número de passageiros por série (opcional — deixe 0 para deduzir automaticamente)",
            min_value=0,
            max_value=60,
            value=0,
            step=1,
            key="v156_n_passageiros_texto",
        )

        if st.button("Carregar histórico colado", key="v156_btn_carregar_texto"):
            try:
                linhas = [l for l in texto.splitlines() if limpar_linha(l) != ""]
                if not linhas:
                    st.warning("Nenhuma linha válida encontrada no texto colado.")
                else:
                    n_psg = n_passageiros_texto if n_passageiros_texto > 0 else None
                    df, meta = montar_dataframe_historico(linhas, n_passageiros=n_psg)
                    set_historico_v156(df, meta)
                    st.success(
                        f"Histórico carregado com sucesso via texto: {meta['n_series']} séries, "
                        f"{meta['n_passageiros']} passageiros por série."
                    )
            except Exception as e:
                st.error(f"Erro ao processar texto colado: {e}")

    st.markdown("---")

    # Pré-visualização do histórico atual
    historico_atual, meta_atual = get_historico_v156()
    if historico_atual is None or meta_atual is None:
        st.info(
            "Nenhum histórico carregado ainda. Use o upload de arquivo ou o copiar/colar "
            "para iniciar o pipeline V14-FLEX ULTRA da V15.6 MAX."
        )
        return

    st.markdown("### 🔎 Resumo do histórico atual (V15.6 MAX)")

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric(
            label="Quantidade de séries",
            value=meta_atual.get("n_series", "-"),
        )
    with col_info2:
        st.metric(
            label="Passageiros por série (FLEX)",
            value=meta_atual.get("n_passageiros", "-"),
        )
    with col_info3:
        intervalo = f"{meta_atual.get('min_val', '?')} a {meta_atual.get('max_val', '?')}"
        st.metric(
            label="Intervalo aproximado de valores",
            value=intervalo,
        )

    st.markdown(
        """
        <div class="very-small-text">
        Para evitar travamentos em históricos gigantes, a pré-visualização abaixo
        limita a quantidade de linhas mostradas. O pipeline e os módulos MAX
        (Replay, Monitor de Risco, Ruído, Confiabilidade, Modo 6, Relatório Final)
        trabalham internamente com todas as séries carregadas.
        </div>
        """,
        unsafe_allow_html=True,
    )

    n_exibir = min(len(historico_atual), MAX_LINHAS_EXIBICAO)
    st.dataframe(
        historico_atual.head(n_exibir),
        use_container_width=True,
        hide_index=True,
    )

    if len(historico_atual) > MAX_LINHAS_EXIBICAO:
        st.caption(
            f"Exibindo somente as primeiras {MAX_LINHAS_EXIBICAO} séries de "
            f"{len(historico_atual)} no total (para proteger contra modo zumbi)."
        )

# ============================================================
# (PLACEHOLDER) FUNÇÕES DO PIPELINE V14-FLEX ULTRA (V15.6 MAX)
# ============================================================
# Estes placeholders mantêm a estrutura do app funcionando
# enquanto as Partes 2/6, 3/6 e 4/6 não são coladas.
# Nada aqui executa lógica real — são versões mínimas
# apenas para permitir navegação e evitar erros.

def calcular_k_sentinela_v156(df: pd.DataFrame, janela: int = 200) -> pd.DataFrame:
    """Placeholder do cálculo de k* — será implementado na Parte 2/6."""
    df = df.copy()
    df["k_sentinela"] = np.nan
    return df


def pipeline_v14_flex_ultra_v156(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    idx_alvo: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Placeholder do Pipeline V14-FLEX ULTRA para V15.6 MAX.
    Será substituído pelas Partes 2/6 e 3/6.
    """
    return {
        "ok": True,
        "idx_alvo": idx_alvo,
        "df_com_k_sentinela": calcular_k_sentinela_v156(df),
        "mensagem": "Pipeline V14-FLEX ULTRA (V15.6 MAX) — aguardando implementação completa.",
    }


# ------------------------------------------------------------
# k* Sentinela de Risco (V15.6 MAX)
# ------------------------------------------------------------

def _calcular_k_sentinela_para_indice(
    serie_k: pd.Series,
    idx: int,
    janela: int,
) -> float:
    """
    Cálculo local de k* para um índice específico.

    - Considera uma janela [idx_início, idx] com tamanho máximo = janela.
    - k* é a porcentagem (0–100) de séries nessa janela com k > 0.
    - Se não houver dados suficientes, retorna NaN.
    """
    if idx < 0 or idx >= len(serie_k):
        return float("nan")

    inicio = max(0, idx - janela + 1)
    sub = serie_k.iloc[inicio : idx + 1]

    # Remove NaNs para não contaminar a estatística
    sub_valida = sub.dropna()
    if len(sub_valida) == 0:
        return float("nan")

    total = len(sub_valida)
    positivos = (sub_valida > 0).sum()

    return float(positivos * 100.0 / total)


def calcular_k_sentinela_v156(
    df: pd.DataFrame,
    janela: int = 200,
) -> pd.DataFrame:
    """
    Cálculo completo de k* (sentinela de risco) para V15.6 MAX.

    - Usa o campo 'k' (K_COLNAME) do histórico.
    - Para cada linha i, calcula k*(i) considerando a janela mais recente.
    - Resultado: coluna 'k_sentinela' com valores em 0–100 (percentual de
      presença de k>0 no trecho recente da estrada).
    """
    df = df.copy()
    if K_COLNAME not in df.columns:
        df["k_sentinela"] = np.nan
        return df

    serie_k = df[K_COLNAME]

    valores_k_star = []
    n = len(df)
    janela_efetiva = max(1, min(janela, n))

    for i in range(n):
        k_star_i = _calcular_k_sentinela_para_indice(serie_k, i, janela_efetiva)
        valores_k_star.append(k_star_i)

    df["k_sentinela"] = valores_k_star
    return df


def classificar_regime_k_sentinela_v156(k_star: Optional[float]) -> str:
    """
    Texto do barômetro de regime da estrada com base em k*.

    Faixas (0–100):
    - < 20  → Estrada estável
    - 20–40 → Aquecimento leve (pré-ruído)
    - 40–70 → Pré-ruptura / ruído moderado
    - ≥ 70  → Ruptura / turbulência pesada
    """
    if k_star is None or (isinstance(k_star, float) and (math.isnan(k_star))):
        return "⚪ Regime indeterminado — k* insuficiente para avaliar."

    try:
        v = float(k_star)
    except Exception:
        return "⚪ Regime indeterminado — k* inválido."

    if v < 20:
        return "🟢 Estrada estável — poucos guardas acertando exatamente o carro."
    if v < 40:
        return "🟡 Aquecimento leve — aumento discreto de guardas acertando."
    if v < 70:
        return "🟠 Pré-ruptura / ruído moderado — muitos guardas em alerta."
    return "🔴 Ruptura / turbulência pesada — guardas concentrados nesse trecho."


# ------------------------------------------------------------
# Extração de janela ativa e estatísticas básicas
# ------------------------------------------------------------

def extrair_janela_ativa_v156(
    df: pd.DataFrame,
    idx_alvo: int,
    tamanho_janela: int,
) -> Tuple[pd.DataFrame, int, int]:
    """
    Extrai a janela ativa [inicio, idx_alvo] respeitando o limite de tamanho_janela.

    Retorna:
    - df_janela (subconjunto do df)
    - idx_inicio (índice inteiro no df original)
    - idx_fim (idx_alvo normalizado)
    """
    n = len(df)
    if n == 0:
        raise ValueError("Histórico vazio ao tentar extrair janela ativa.")

    if idx_alvo < 0:
        idx_alvo = 0
    if idx_alvo >= n:
        idx_alvo = n - 1

    tamanho_janela = max(1, min(tamanho_janela, n))
    idx_inicio = max(0, idx_alvo - tamanho_janela + 1)

    df_janela = df.iloc[idx_inicio : idx_alvo + 1].copy()
    return df_janela, idx_inicio, idx_alvo


def calcular_estatisticas_basicas_janela_v156(
    df_janela: pd.DataFrame,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Estatísticas básicas da janela ativa:
    - Número de séries na janela
    - Dispersão dos passageiros (mín, máx, amplitude)
    - k médio / k máximo na janela
    - k* médio na janela (se disponível)
    """
    col_passageiros = meta.get("col_passageiros", [])
    if not col_passageiros:
        raise ValueError("Metadados do histórico sem colunas de passageiros.")

    estat = {}

    estat["n_series_janela"] = len(df_janela)

    # Dispersão dos passageiros
    valores_psg = df_janela[col_passageiros].values.flatten()
    valores_psg = valores_psg[~pd.isna(valores_psg)]
    if len(valores_psg) > 0:
        vmin = float(np.nanmin(valores_psg))
        vmax = float(np.nanmax(valores_psg))
        estat["min_val_janela"] = vmin
        estat["max_val_janela"] = vmax
        estat["amplitude_janela"] = vmax - vmin
    else:
        estat["min_val_janela"] = None
        estat["max_val_janela"] = None
        estat["amplitude_janela"] = None

    # k em janela
    if K_COLNAME in df_janela.columns:
        k_vals = df_janela[K_COLNAME].dropna()
        if len(k_vals) > 0:
            estat["k_medio_janela"] = float(k_vals.mean())
            estat["k_max_janela"] = float(k_vals.max())
        else:
            estat["k_medio_janela"] = None
            estat["k_max_janela"] = None
    else:
        estat["k_medio_janela"] = None
        estat["k_max_janela"] = None

    # k* em janela
    if "k_sentinela" in df_janela.columns:
        k_star_vals = df_janela["k_sentinela"].dropna()
        if len(k_star_vals) > 0:
            estat["k_star_medio_janela"] = float(k_star_vals.mean())
            estat["k_star_max_janela"] = float(k_star_vals.max())
        else:
            estat["k_star_medio_janela"] = None
            estat["k_star_max_janela"] = None
    else:
        estat["k_star_medio_janela"] = None
        estat["k_star_max_janela"] = None

    return estat


# ------------------------------------------------------------
# Núcleo + Coberturas + Interseção Estatística (base)
# ------------------------------------------------------------

def calcular_frequencias_passageiros_janela_v156(
    df_janela: pd.DataFrame,
    meta: Dict[str, Any],
) -> pd.DataFrame:
    """
    Calcula as frequências de cada número de passageiro na janela ativa.

    Retorna um DataFrame com:
    - valor
    - freq_abs
    - freq_rel (0–1)
    """
    col_passageiros = meta.get("col_passageiros", [])
    if not col_passageiros:
        raise ValueError("Metadados do histórico sem colunas de passageiros.")

    valores = df_janela[col_passageiros].values.flatten()
    valores = valores[~pd.isna(valores)]
    if len(valores) == 0:
        return pd.DataFrame(columns=["valor", "freq_abs", "freq_rel"])

    valores = valores.astype(int)
    unicos, contagens = np.unique(valores, return_counts=True)
    total = float(len(valores))

    df_freq = pd.DataFrame(
        {
            "valor": unicos.astype(int),
            "freq_abs": contagens.astype(int),
            "freq_rel": contagens.astype(float) / total,
        }
    ).sort_values(["freq_abs", "valor"], ascending=[False, True])

    return df_freq


def construir_nucleo_e_coberturas_v156(
    df_freq: pd.DataFrame,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Constrói o Núcleo e as Coberturas com base na distribuição de frequências:

    Estratégia:
    - NÚCLEO:
        Valores no quartil superior (Q3) de frequência absoluta, com pelo menos 2 ocorrências.
    - COBERTURA SUAVE:
        Valores entre mediana (Q2) e Q3 (excluindo os que já estão no núcleo).
    - COBERTURA AGRESSIVA:
        Demais valores com pelo menos 1 ocorrência.

    Retorna:
    - dict com:
        - nucleo: List[int]
        - cobertura_suave: List[int]
        - cobertura_agressiva: List[int]
        - df_freq_rotulado: DataFrame com coluna 'camada'
    """
    if df_freq.empty:
        return {
            "nucleo": [],
            "cobertura_suave": [],
            "cobertura_agressiva": [],
            "df_freq_rotulado": df_freq.copy(),
        }

    freq_abs = df_freq["freq_abs"].values.astype(float)

    if len(freq_abs) == 1:
        # Caso extremo: só um valor na janela
        valor_unico = int(df_freq.iloc[0]["valor"])
        df_freq_rot = df_freq.copy()
        df_freq_rot["camada"] = "núcleo"
        return {
            "nucleo": [valor_unico],
            "cobertura_suave": [],
            "cobertura_agressiva": [],
            "df_freq_rotulado": df_freq_rot,
        }

    q2 = np.percentile(freq_abs, 50)
    q3 = np.percentile(freq_abs, 75)

    nucleo = []
    cobertura_suave = []
    cobertura_agressiva = []
    camadas = []

    for _, row in df_freq.iterrows():
        v = int(row["valor"])
        f = float(row["freq_abs"])

        if f >= q3 and f >= 2:
            nucleo.append(v)
            camadas.append("núcleo")
        elif f >= q2:
            cobertura_suave.append(v)
            camadas.append("cobertura_suave")
        else:
            cobertura_agressiva.append(v)
            camadas.append("cobertura_agressiva")

    df_freq_rotulado = df_freq.copy()
    df_freq_rotulado["camada"] = camadas

    return {
        "nucleo": sorted(list(set(nucleo))),
        "cobertura_suave": sorted(list(set(cobertura_suave))),
        "cobertura_agressiva": sorted(list(set(cobertura_agressiva))),
        "df_freq_rotulado": df_freq_rotulado.sort_values(
            ["camada", "freq_abs", "valor"],
            ascending=[True, False, True],
        ),
    }


def construir_intersecao_estatistica_v156(camadas: Dict[str, List[int]]) -> Dict[str, Any]:
    """
    Estrutura base para interseção estatística de camadas de previsão.

    'camadas' é um dicionário:
        {
            "nucleo": [...],
            "cobertura_suave": [...],
            "cobertura_agressiva": [...],
            "s6": [...],          # (quando disponível nas próximas partes)
            "monte_carlo": [...], # (quando disponível)
            "micro_leque": [...], # (quando disponível)
            ...
        }

    A ideia é:
    - Calcular interseções entre as principais camadas (núcleo, s6, MC, etc.)
    - Usar isso mais à frente no Modo TURBO++ e Modo 6 Acertos.

    Nesta Parte 2/6, implementamos uma interseção genérica entre qualquer
    subconjunto de camadas já fornecidas.
    """
    # Conjunto total de valores envolvidos:
    todos_valores = set()
    for lista in camadas.values():
        todos_valores.update(lista)

    # Frequência de presença em camadas:
    contagem_por_valor = {v: 0 for v in todos_valores}
    for lista in camadas.values():
        for v in lista:
            contagem_por_valor[v] += 1

    # Ordenar por "força" de interseção (quantas camadas contêm o valor)
    intersecao_ordenada = sorted(
        contagem_por_valor.items(), key=lambda x: (-x[1], x[0])
    )

    # Valores que aparecem em pelo menos 2 camadas diferentes (interseção real)
    intersecao_forte = [v for v, c in intersecao_ordenada if c >= 2]

    return {
        "contagem_por_valor": contagem_por_valor,
        "intersecao_ordenada": intersecao_ordenada,
        "intersecao_forte": intersecao_forte,
    }


# ------------------------------------------------------------
# Pipeline V14-FLEX ULTRA (V15.6 MAX) — Núcleo
# ------------------------------------------------------------

def pipeline_v14_flex_ultra_v156(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    idx_alvo: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Pipeline principal V14-FLEX ULTRA adaptado para V15.6 MAX.

    Responsabilidades desta função:
    - Garantir que k* (sentinela) está calculado em todo o histórico.
    - Escolher e normalizar o índice alvo (série que queremos projetar).
    - Extrair a janela ativa em torno da série alvo.
    - Calcular estatísticas básicas da janela.
    - Construir o Núcleo + Coberturas + estrutura de interseção estatística base.
    - Registrar o fato de que o "pipeline base" está pronto na sessão.

    As camadas S6, Monte Carlo, Micro-Leque e os módulos de Replay/TURBO/M6
    vão se plugar depois em cima dessa estrutura.
    """
    if df is None or meta is None:
        raise ValueError("Histórico ou metadados ausentes ao chamar o pipeline V15.6 MAX.")

    n_series = meta.get("n_series", len(df))
    if n_series == 0:
        raise ValueError("Histórico vazio ao chamar o pipeline V15.6 MAX.")

    # 1) Normalizar índice alvo
    if idx_alvo is None:
        idx_alvo = n_series - 1  # Última série por padrão
    if idx_alvo < 0:
        idx_alvo = 0
    if idx_alvo >= n_series:
        idx_alvo = n_series - 1

    id_alvo = df.iloc[idx_alvo][ID_COLNAME]

    # 2) k* sentinela em todo o histórico
    limite_janela_global = int(st.session_state.get("v156_limite_janela", 300))
    df_kstar = calcular_k_sentinela_v156(df, janela=limite_janela_global)

    # 3) Extrair janela ativa
    df_janela, idx_inicio_janela, idx_fim_janela = extrair_janela_ativa_v156(
        df_kstar,
        idx_alvo=idx_alvo,
        tamanho_janela=limite_janela_global,
    )

    # 4) Estatísticas básicas da janela
    estat_janela = calcular_estatisticas_basicas_janela_v156(df_janela, meta)

    # 5) Frequências e Núcleo + Coberturas
    df_freq = calcular_frequencias_passageiros_janela_v156(df_janela, meta)
    camadas_nucleo = construir_nucleo_e_coberturas_v156(df_freq, meta)

    # 6) Regime de k* na série alvo
    k_star_alvo = df_kstar.iloc[idx_alvo]["k_sentinela"]
    regime_k_star = classificar_regime_k_sentinela_v156(k_star_alvo)

    # 7) Interseção estatística base (por enquanto só Núcleo/Coberturas)
    camadas_para_intersecao = {
        "nucleo": camadas_nucleo["nucleo"],
        "cobertura_suave": camadas_nucleo["cobertura_suave"],
        "cobertura_agressiva": camadas_nucleo["cobertura_agressiva"],
        # Nas próximas partes adicionaremos:
        # "s6": [...],
        # "monte_carlo": [...],
        # "micro_leque": [...],
    }
    intersecao_base = construir_intersecao_estatistica_v156(camadas_para_intersecao)

    resultado = {
        "ok": True,
        "mensagem": "Pipeline V14-FLEX ULTRA (V15.6 MAX) executado com sucesso.",
        "meta": meta,
        "idx_alvo": idx_alvo,
        "id_alvo": id_alvo,
        "janela_inicio": idx_inicio_janela,
        "janela_fim": idx_fim_janela,
        "tamanho_janela": idx_fim_janela - idx_inicio_janela + 1,
        "df_com_k_sentinela": df_kstar,
        "df_janela": df_janela,
        "estat_janela": estat_janela,
        "k_star_alvo": k_star_alvo,
        "regime_k_star": regime_k_star,
        "df_freq": df_freq,
        "nucleo": camadas_nucleo["nucleo"],
        "cobertura_suave": camadas_nucleo["cobertura_suave"],
        "cobertura_agressiva": camadas_nucleo["cobertura_agressiva"],
        "df_freq_rotulado": camadas_nucleo["df_freq_rotulado"],
        "intersecao_base": intersecao_base,
        # Espaço reservado para acoplarmos:
        "camadas_extra": {},  # S6, Monte Carlo, Micro-Leque, etc. nas próximas partes
    }

    # 8) Marcar na sessão que o pipeline "base" está pronto
    marcar_pipeline_pronto_v156()

    return resultado
# ============================================================
# PARTE 3/6 — REPLAY LIGHT / REPLAY ULTRA / REPLAY ULTRA UNITÁRIO
# ============================================================

# Nesta parte implementamos:
# - Seleção de série-alvo (janela completa)
# - Mecanismo de similaridade das séries recentes
# - Replay LIGHT: janela fixa + estatística direta
# - Replay ULTRA: busca adaptativa por vizinhos
# - Replay ULTRA Unitário: análise profunda da série específica
#
# Todos esses replays usam INTERNAMENTE o pipeline V14-FLEX ULTRA
# estruturado na Parte 2/6, com base em:
#   • k*
#   • Regime da estrada
#   • Núcleo + Coberturas
#   • Interseção estatística
#
# O Modo TURBO++, Modo 6 Acertos e Relatório Final usam estes
# três replays como módulos upstream.


# ============================================================
# FUNÇÕES AUXILIARES DE SIMILARIDADE ENTRE SÉRIES
# ============================================================

def distancia_entre_series_v156(
    a: List[int],
    b: List[int],
    penalidade_abs: float = 1.0,
) -> float:
    """
    Distância absoluta entre duas séries (lista de passageiros).
    A penalidade é somada a |a[i] - b[i]| ao longo dos passageiros.
    """
    if len(a) != len(b):
        return float("inf")

    total = 0.0
    for x, y in zip(a, b):
        if x is None or y is None:
            total += penalidade_abs * 2.0
        else:
            total += abs(int(x) - int(y)) * penalidade_abs

    return float(total)


def encontrar_vizinhos_semelhantes_v156(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    serie_ref: List[int],
    limite_busca: int = 200,
    k_minimo: int = 0,
) -> List[Tuple[int, float]]:
    """
    Busca vizinhos mais semelhantes à série de referência.

    Retorna lista de tuplas:
        (índice_df, distância)
    ordenadas pela distância crescente.
    """
    col_passageiros = meta["col_passageiros"]
    n = len(df)

    inicio_busca = max(0, n - limite_busca)
    candidatos = []

    for idx in range(inicio_busca, n - 1):  # exclui a última série
        linha = df.iloc[idx]
        if k_minimo > 0:
            if pd.isna(linha[K_COLNAME]) or linha[K_COLNAME] < k_minimo:
                continue

        serie_atual = [linha[col] for col in col_passageiros]
        dist = distancia_entre_series_v156(serie_ref, serie_atual)
        candidatos.append((idx, dist))

    candidatos_ordenados = sorted(candidatos, key=lambda x: x[1])
    return candidatos_ordenados


# ============================================================
# REPLAY LIGHT (V15.6 MAX)
# ============================================================

def replay_light_v156(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    idx_alvo: int,
) -> Dict[str, Any]:
    """
    REPLAY LIGHT — avaliação leve e rápida de similaridade e projeção.

    Estratégia:
    - Usa a janela ativa calculada no Pipeline.
    - Identifica as séries mais parecidas com a série alvo.
    - Produz um vetor-padrão de "continuidade" a partir de 1 a 4 vizinhos.
    - Usa núcleo + coberturas como filtro adaptativo.
    """

    col_passageiros = meta["col_passageiros"]
    df_pipeline = st.session_state[SESSAO_PIPELINE]

    df_kstar = df_pipeline.get("turbo_ultra_base_df", None)
    if df_kstar is None:
        df_kstar = df  # fallback

    # Série alvo
    serie_alvo = [df.iloc[idx_alvo][c] for c in col_passageiros]

    # Busca vizinhos (LIGHT)
    vizinhos = encontrar_vizinhos_semelhantes_v156(
        df,
        meta,
        serie_ref=serie_alvo,
        limite_busca=200,
        k_minimo=0,
    )

    # Selecionar até 4 vizinhos
    vizinhos_top = vizinhos[:4] if len(vizinhos) >= 4 else vizinhos
    previsoes = []

    for idx_viz, dist in vizinhos_top:
        if idx_viz + 1 < len(df):
            prox = df.iloc[idx_viz + 1]
            previsao_viz = [prox[c] for c in col_passageiros]
            previsoes.append(previsao_viz)

    # Consolidação LIGHT
    if not previsoes:
        return {
            "ok": False,
            "motivo": "Sem vizinhos disponíveis para Replay LIGHT.",
            "vizinhos": vizinhos_top,
            "previsao": [],
        }

    matriz = np.array(previsoes)
    previsao_final = np.round(np.mean(matriz, axis=0)).astype(int).tolist()

    return {
        "ok": True,
        "vizinhos": vizinhos_top,
        "previsoes_vizinhos": previsoes,
        "previsao_final": previsao_final,
    }


# ============================================================
# REPLAY ULTRA (V15.6 MAX)
# ============================================================

def replay_ultra_v156(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    idx_alvo: int,
) -> Dict[str, Any]:
    """
    REPLAY ULTRA — versão densa e adaptativa do Replay.

    Estratégia:
    - Similar ao Replay LIGHT, mas:
        ✓ exige vizinhos com k >= 1 (ruído menor)
        ✓ busca os 10-15 vizinhos mais semelhantes
        ✓ pondera as previsões pela semelhança
        ✓ usa núcleo + coberturas da janela ativa para refinar
        ✓ gera estatísticas estruturais para TURBO++ e Modo 6
    """

    col_passageiros = meta["col_passageiros"]

    # Série alvo
    serie_alvo = [df.iloc[idx_alvo][c] for c in col_passageiros]

    # Busca vizinhos ULTRA (apenas séries com k>=1)
    vizinhos = encontrar_vizinhos_semelhantes_v156(
        df,
        meta,
        serie_ref=serie_alvo,
        limite_busca=400,
        k_minimo=1,
    )

    top_n = 15
    vizinhos_top = vizinhos[:top_n] if len(vizinhos) >= top_n else vizinhos

    previsoes = []
    pesos = []

    for idx_viz, dist in vizinhos_top:
        if idx_viz + 1 >= len(df):
            continue

        prox = df.iloc[idx_viz + 1]
        previsao_viz = [prox[c] for c in col_passageiros]

        peso = 1.0 / (1.0 + dist)
        previsoes.append(previsao_viz)
        pesos.append(peso)

    if not previsoes:
        return {
            "ok": False,
            "motivo": "Sem vizinhos disponíveis para Replay ULTRA.",
            "vizinhos": vizinhos_top,
            "previsao": [],
        }

    matriz = np.array(previsoes)
    pesos_np = np.array(pesos).reshape(-1, 1)

    previsao_final = np.round(
        np.sum(matriz * pesos_np, axis=0) / np.sum(pesos_np)
    ).astype(int).tolist()

    return {
        "ok": True,
        "vizinhos": vizinhos_top,
        "previsoes_vizinhos": previsoes,
        "pesos": pesos,
        "previsao_final": previsao_final,
    }


# ============================================================
# REPLAY ULTRA UNITÁRIO (V15.6 MAX)
# ============================================================

def replay_ultra_unitario_v156(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    idx_alvo: int,
) -> Dict[str, Any]:
    """
    REPLAY ULTRA UNITÁRIO — versão profunda do Replay para 1 única série.

    Estratégia:
    - Considera apenas a série alvo em detalhe
    - Busca vizinhos extremamente parecidos (distância < threshold adaptativa)
    - Traz os 3 padrões principais
    - Gera previsões de "peaking" muito fortes para Modo TURBO e Modo 6
    """

    col_passageiros = meta["col_passageiros"]
    serie_ref = [df.iloc[idx_alvo][c] for c in col_passageiros]

    limite_busca = 300
    vizinhos = encontrar_vizinhos_semelhantes_v156(
        df,
        meta,
        serie_ref=serie_ref,
        limite_busca=limite_busca,
        k_minimo=0,
    )

    # Seleção adaptativa — vizinhos com distância significativamente pequena
    if not vizinhos:
        return {
            "ok": False,
            "motivo": "Sem vizinhos para Replay Unitário.",
            "vizinhos": [],
            "previsoes": [],
        }

    # threshold adaptativo
    d0 = vizinhos[0][1]
    limite_dist = d0 + 3.0

    vizinhos_filtrados = [(i, d) for (i, d) in vizinhos if d <= limite_dist]
    vizinhos_top = vizinhos_filtrados[:5]

    previsoes = []
    for idx_viz, _ in vizinhos_top:
        if idx_viz + 1 < len(df):
            prox = df.iloc[idx_viz + 1]
            previsoes.append([prox[c] for c in col_passageiros])

    if not previsoes:
        return {
            "ok": False,
            "motivo": "Nenhuma previsão possível no Replay Unitário.",
        }

    matriz = np.array(previsoes)
    previsao_final = np.round(np.mean(matriz, axis=0)).astype(int).tolist()

    return {
        "ok": True,
        "vizinhos": vizinhos_top,
        "previsoes": previsoes,
        "previsao_final": previsao_final,
    }


# ============================================================
# CONTROLADOR CENTRAL DOS REPLAYS
# ============================================================

def executar_replays_v156(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    idx_alvo: int,
) -> Dict[str, Any]:
    """
    Executa os três replays e retorna um pacote completo contendo:
    - Replay LIGHT
    - Replay ULTRA
    - Replay ULTRA Unitário
    """
    replay_l = replay_light_v156(df, meta, idx_alvo)
    replay_u = replay_ultra_v156(df, meta, idx_alvo)
    replay_uu = replay_ultra_unitario_v156(df, meta, idx_alvo)

    return {
        "light": replay_l,
        "ultra": replay_u,
        "unitario": replay_uu,
    }
# ============================================================
# PARTE 4/6 — TURBO++ ULTRA ANTI-RUÍDO + RUÍDO CONDICIONAL
# ============================================================

# Nesta parte conectamos:
# - Pipeline V14-FLEX ULTRA (resultado base da Parte 2/6)
# - Replays (Parte 3/6)
# - Núcleo + Coberturas + Interseção Estatística
#
# Para construir:
#   • 🚨 Monitor de Risco (k & k*)
#   • 🚀 Modo TURBO++ ULTRA Anti-Ruído (V15.6 MAX)
#   • 📊 Painel de Ruído Condicional (NR%, Tipo A/B)
#
# Também criamos os PAINÉIS de interface:
#   - 🔍 Pipeline V14-FLEX ULTRA (V15.6 MAX)
#   - 💡 Replay LIGHT (V15.6 MAX)
#   - 📅 Replay ULTRA (V15.6 MAX)
#   - 🎯 Replay ULTRA Unitário (V15.6 MAX)
#   - 🚨 Monitor de Risco (k & k*) (V15.6 MAX)
#   - 🚀 Modo TURBO++ ULTRA Anti-Ruído (V15.6 MAX)
#   - 📊 Ruído Condicional (V15.6 MAX)
#
# Modo 6 Acertos e Relatório Final virão nas Partes 5/6 e 6/6,
# consumindo tudo o que está sendo montado aqui.


# ============================================================
# RUÍDO CONDICIONAL (V15.6 MAX)
# ============================================================

def calcular_ruido_condicional_v156(
    meta: Dict[str, Any],
    estat_janela: Dict[str, Any],
    df_freq: pd.DataFrame,
    resultado_replays: Dict[str, Any],
    k_star_alvo: Optional[float],
) -> Dict[str, Any]:
    """
    Calcula um mapa de ruído condicional para a V15.6 MAX.

    Ideia geral:
    - Usa a entropia da distribuição de frequências da janela (quanto mais espalhado, mais ruído).
    - Compara a "densidade" do núcleo vs coberturas.
    - Usa os replays (LIGHT / ULTRA / Unitário) para ver se os padrões convergem ou divergem.
    - Integra k* do alvo como modulador de risco/ruído.

    Saída principal:
    - NR% (0–100) → intensidade de ruído condicional.
    - classificação_tipo (A / B / Misto)
    - textos explicativos prontos para o painel.
    """
    # 1) Entropia da distribuição de frequências
    if df_freq is None or df_freq.empty:
        entropia = None
        nr_percent = None
    else:
        probs = df_freq["freq_rel"].values
        # Evitar log(0)
        probs = probs[probs > 0]
        if len(probs) == 0:
            entropia = None
            nr_percent = None
        else:
            entropia = float(-np.sum(probs * np.log2(probs)))
            # Normalização grosseira pela quantidade de valores distintos
            max_entropia = float(np.log2(len(probs)))
            if max_entropia > 0:
                nr_percent = float(entropia / max_entropia * 100.0)
            else:
                nr_percent = None

    # 2) Divergência entre replays
    def _distancia_lista(a: List[int], b: List[int]) -> float:
        if not a or not b or len(a) != len(b):
            return float("nan")
        return float(np.mean(np.abs(np.array(a, dtype=float) - np.array(b, dtype=float))))

    dist_light_ultra = None
    dist_light_unit = None
    dist_ultra_unit = None

    prev_l = resultado_replays.get("light", {}).get("previsao_final", [])
    prev_u = resultado_replays.get("ultra", {}).get("previsao_final", [])
    prev_uu = resultado_replays.get("unitario", {}).get("previsao_final", [])

    if prev_l and prev_u and len(prev_l) == len(prev_u):
        dist_light_ultra = _distancia_lista(prev_l, prev_u)
    if prev_l and prev_uu and len(prev_l) == len(prev_uu):
        dist_light_unit = _distancia_lista(prev_l, prev_uu)
    if prev_u and prev_uu and len(prev_u) == len(prev_uu):
        dist_ultra_unit = _distancia_lista(prev_u, prev_uu)

    # Média das divergências conhecidas
    dists_validas = [d for d in [dist_light_ultra, dist_light_unit, dist_ultra_unit] if d is not None and not math.isnan(d)]
    if dists_validas:
        divergencia_media = float(np.mean(dists_validas))
    else:
        divergencia_media = None

    # 3) Ajuste condicional por k*
    if k_star_alvo is None or (isinstance(k_star_alvo, float) and math.isnan(k_star_alvo)):
        k_star_sinal = 0.0
    else:
        k_star_sinal = float(k_star_alvo)

    # Heurística de classificação de ruído:
    # NR% básico pela entropia
    if nr_percent is None:
        nr_basico = 0.0
    else:
        nr_basico = float(nr_percent)

    # Divergência normalizada (0-100) em função da amplitude típica (assumimos 60 como escala)
    if divergencia_media is None:
        nr_div = 0.0
    else:
        nr_div = float(min(100.0, divergencia_media * 2.0))

    # Combinação: entropia (peso 0.6), divergência (peso 0.4), modulada por k*
    nr_combinado = 0.6 * nr_basico + 0.4 * nr_div
    nr_final = nr_combinado * (0.5 + 0.5 * (k_star_sinal / 100.0))
    nr_final = float(max(0.0, min(100.0, nr_final)))

    # Tipo A/B (metáfora de ruído estrutural vs ruído de divergência)
    # Tipo A: entropia alta, divergência baixa → ruído "espalhado"
    # Tipo B: entropia moderada, divergência alta → ruído "direcional"
    if nr_basico >= 60 and nr_div < 40:
        tipo = "A"
        descricao_tipo = "Ruído Tipo A — dispersão estrutural dos números na estrada."
    elif nr_div >= 60:
        tipo = "B"
        descricao_tipo = "Ruído Tipo B — divergência forte entre os módulos de previsão."
    else:
        tipo = "Misto"
        descricao_tipo = "Ruído misto — combinação de dispersão e divergência entre módulos."

    texto_resumo = (
        f"NR% estimado ≈ {nr_final:.1f}%. {descricao_tipo} "
        f"(entropia={nr_basico:.1f}%, divergência={nr_div:.1f}%, k*={k_star_sinal:.1f}%)."
    )

    return {
        "ok": True,
        "entropia": entropia,
        "nr_percent": nr_final,
        "nr_basico_percent": nr_basico,
        "nr_div_percent": nr_div,
        "tipo": tipo,
        "descricao_tipo": descricao_tipo,
        "divergencia_media": divergencia_media,
        "texto_resumo": texto_resumo,
    }


# ============================================================
# MODO TURBO++ ULTRA ANTI-RUÍDO (V15.6 MAX)
# ============================================================

def montar_previsao_turbo_ultra_v156(
    meta: Dict[str, Any],
    resultado_pipeline: Dict[str, Any],
    resultado_replays: Dict[str, Any],
    info_ruido: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Constrói a previsão final TURBO++ ULTRA Anti-Ruído para a V15.6 MAX.

    Fontes que entram:
    - Núcleo (frequência alta na janela)
    - Cobertura suave + agressiva
    - Interseção estatística base (núcleo + coberturas)
    - Replay LIGHT
    - Replay ULTRA
    - Replay ULTRA Unitário
    - NR% (ruído condicional) e k* (regime da estrada)

    Estratégia (resumo):
    - Quando NR% é baixo e k* estável → peso maior para Replay ULTRA + Núcleo.
    - Quando NR% é alto e k* em ruptura → peso maior para Núcleo + Unitário, com penais.
    - Quando cenário intermediário → combina LIGHT + ULTRA + Núcleo de forma balanceada.
    """

    col_passageiros = meta["col_passageiros"]
    n_pass = len(col_passageiros)

    k_star_alvo = resultado_pipeline.get("k_star_alvo", None)
    if k_star_alvo is None or (isinstance(k_star_alvo, float) and math.isnan(k_star_alvo)):
        k_star_val = 0.0
    else:
        k_star_val = float(k_star_alvo)

    nr_percent = info_ruido.get("nr_percent", 0.0)
    try:
        nr_val = float(nr_percent)
    except Exception:
        nr_val = 0.0

    # Bases de previsão
    prev_light = resultado_replays.get("light", {}).get("previsao_final", [])
    prev_ultra = resultado_replays.get("ultra", {}).get("previsao_final", [])
    prev_unit = resultado_replays.get("unitario", {}).get("previsao_final", [])

    # Ajuste de comprimentos
    def _ajustar_tamanho(lista: List[int]) -> List[Optional[int]]:
        if not lista:
            return [None] * n_pass
        if len(lista) < n_pass:
            return lista + [None] * (n_pass - len(lista))
        if len(lista) > n_pass:
            return lista[:n_pass]
        return lista

    prev_light = _ajustar_tamanho(prev_light)
    prev_ultra = _ajustar_tamanho(prev_ultra)
    prev_unit = _ajustar_tamanho(prev_unit)

    # Núcleo + coberturas
    nucleo = resultado_pipeline.get("nucleo", [])
    cobertura_suave = resultado_pipeline.get("cobertura_suave", [])
    cobertura_agressiva = resultado_pipeline.get("cobertura_agressiva", [])
    intersecao_forte = resultado_pipeline.get("intersecao_base", {}).get("intersecao_forte", [])

    # Pesos adaptativos conforme regime de ruído e k*
    # Começamos com uma base neutra
    peso_light = 1.0
    peso_ultra = 1.0
    peso_unit = 1.0
    peso_nucleo = 1.0

    # Cenário de estrada estável (k* baixo) e ruído baixo (NR% baixo)
    if k_star_val < 30 and nr_val < 40:
        peso_ultra *= 1.6
        peso_light *= 1.0
        peso_unit *= 1.2
        peso_nucleo *= 1.4

    # Cenário de pré-ruptura / moderado
    elif k_star_val < 70 and nr_val < 70:
        peso_ultra *= 1.3
        peso_light *= 1.1
        peso_unit *= 1.3
        peso_nucleo *= 1.2

    # Cenário de ruptura (k* alto) e/ou ruído alto
    else:
        peso_ultra *= 0.8
        peso_light *= 0.9
        peso_unit *= 1.6
        peso_nucleo *= 1.8

    # Construir matriz de previsões
    base_listas = []
    base_pesos = []

    if any(v is not None for v in prev_light):
        base_listas.append(prev_light)
        base_pesos.append(peso_light)
    if any(v is not None for v in prev_ultra):
        base_listas.append(prev_ultra)
        base_pesos.append(peso_ultra)
    if any(v is not None for v in prev_unit):
        base_listas.append(prev_unit)
        base_pesos.append(peso_unit)

    if not base_listas:
        # fallback: usar núcleo como "projeção bruta"
        proj_nucleo = sorted(list(set(nucleo)))[:n_pass]
        return {
            "ok": False,
            "motivo": "Sem previsões de Replay disponíveis, usando Núcleo como fallback.",
            "previsao_final": proj_nucleo,
            "nucleo": nucleo,
            "cobertura_suave": cobertura_suave,
            "cobertura_agressiva": cobertura_agressiva,
            "intersecao_forte": intersecao_forte,
            "k_star_alvo": k_star_val,
            "nr_percent": nr_val,
        }

    matriz = np.array(base_listas, dtype=float)
    pesos_np = np.array(base_pesos, dtype=float).reshape(-1, 1)

    # Média ponderada posição-a-posição
    previsao_bruta = np.sum(matriz * pesos_np, axis=0) / np.sum(pesos_np)
    previsao_bruta = np.round(previsao_bruta).astype(int).tolist()

    # Filtro anti-ruído baseado em núcleo + interseção
    # Se um valor projetado não está em nenhuma camada forte, tentamos puxá-lo
    # para o valor mais próximo que esteja no núcleo ou na intersecção forte.
    candidatos_fortes = set(nucleo) | set(intersecao_forte)
    if not candidatos_fortes:
        candidatos_fortes = set(nucleo) | set(cobertura_suave) | set(cobertura_agressiva)

    def _corrigir_com_candidatos(valor: int) -> int:
        if valor in candidatos_fortes:
            return valor
        if not candidatos_fortes:
            return valor
        # Escolhe o candidato mais próximo
        melhor = None
        melhor_dist = None
        for c in candidatos_fortes:
            d = abs(int(c) - int(valor))
            if (melhor is None) or (d < melhor_dist):
                melhor = c
                melhor_dist = d
        return int(melhor)

    previsao_filtrada = [ _corrigir_com_candidatos(v) for v in previsao_bruta ]

    return {
        "ok": True,
        "previsao_bruta": previsao_bruta,
        "previsao_filtrada": previsao_filtrada,
        "nucleo": nucleo,
        "cobertura_suave": cobertura_suave,
        "cobertura_agressiva": cobertura_agressiva,
        "intersecao_forte": intersecao_forte,
        "k_star_alvo": k_star_val,
        "nr_percent": nr_val,
        "peso_light": peso_light,
        "peso_ultra": peso_ultra,
        "peso_unit": peso_unit,
        "peso_nucleo": peso_nucleo,
    }


# ============================================================
# PAINEL — PIPELINE V14-FLEX ULTRA (V15.6 MAX)
# ============================================================

def painel_pipeline_v14_flex_ultra_v156() -> None:
    """
    Painel que roda o pipeline V14-FLEX ULTRA completo para a V15.6 MAX,
    mostrando:
    - Série alvo
    - Janela ativa
    - Barômetro k*
    - Núcleo + Coberturas
    - Interseção estatística base
    """
    st.markdown("## 🔍 Pipeline V14-FLEX ULTRA (V15.6 MAX)")

    df, meta = get_historico_v156()
    if df is None or meta is None:
        st.warning("Carregue primeiro o histórico no painel de entrada FLEX ULTRA.")
        return

    n_series = meta["n_series"]
    col1, col2 = st.columns(2)
    with col1:
        idx_alvo_ui = st.number_input(
            "Índice alvo (1 = primeira série carregada)",
            min_value=1,
            max_value=n_series,
            value=n_series,
            step=1,
            key="v156_idx_alvo_pipeline_input",
        )
    with col2:
        st.markdown(
            """
            <div class="small-text">
            A série alvo é aquela para a qual o sistema V15.6 MAX
            irá preparar toda a estrutura de Núcleo + Coberturas + Replays,
            alimentando TURBO++, Ruído, Confiabilidade e Modo 6 Acertos.
            </div>
            """,
            unsafe_allow_html=True,
        )

    idx_alvo = int(idx_alvo_ui) - 1

    if st.button("Rodar Pipeline V14-FLEX ULTRA (V15.6 MAX)", key="v156_btn_pipeline"):
        try:
            resultado = pipeline_v14_flex_ultra_v156(df, meta, idx_alvo)

            # Atualiza sessão com base do pipeline
            pipeline_state = st.session_state.get(SESSAO_PIPELINE, {})
            pipeline_state["pipeline_base"] = resultado
            pipeline_state["turbo_ultra_base_df"] = resultado["df_com_k_sentinela"]
            pipeline_state["idx_alvo"] = idx_alvo
            pipeline_state["id_alvo"] = resultado["id_alvo"]
            st.session_state[SESSAO_PIPELINE] = pipeline_state

            st.success(
                f"Pipeline executado com sucesso para a série alvo {resultado['id_alvo']} "
                f"(janela {resultado['janela_inicio']+1} → {resultado['janela_fim']+1})."
            )

            # Exibir resumo
            estat = resultado["estat_janela"]
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Séries na janela", estat.get("n_series_janela", "-"))
                st.metric("Amplitude na janela", estat.get("amplitude_janela", "-"))
            with col_b:
                st.metric("k médio (janela)", estat.get("k_medio_janela", "-"))
                st.metric("k* médio (janela)", f"{estat.get('k_star_medio_janela', float('nan')):.1f}" if estat.get("k_star_medio_janela") is not None else "-")
            with col_c:
                st.metric("k máx (janela)", estat.get("k_max_janela", "-"))
                st.metric("k* alvo", f"{resultado.get('k_star_alvo', float('nan')):.1f}" if resultado.get("k_star_alvo") is not None else "-")

            st.markdown("#### Barômetro k* (regime da estrada)")
            st.info(resultado["regime_k_star"])

            st.markdown("#### Núcleo + Coberturas (frequências na janela)")
            st.dataframe(
                resultado["df_freq_rotulado"],
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                f"Núcleo: {resultado['nucleo']} | "
                f"Cobertura suave: {resultado['cobertura_suave']} | "
                f"Cobertura agressiva: {resultado['cobertura_agressiva']}"
            )

        except Exception as e:
            st.error(f"Erro ao executar pipeline: {e}")


# ============================================================
# PAINÉIS — REPLAY LIGHT / ULTRA / ULTRA UNITÁRIO
# ============================================================

def _obter_pipeline_base_v156() -> Optional[Dict[str, Any]]:
    pipeline_state = st.session_state.get(SESSAO_PIPELINE, {})
    return pipeline_state.get("pipeline_base", None)


def painel_replay_light_v156() -> None:
    st.markdown("## 💡 Replay LIGHT (V15.6 MAX)")

    df, meta = get_historico_v156()
    if df is None or meta is None:
        st.warning("Carregue o histórico e rode primeiro o Pipeline V14-FLEX ULTRA.")
        return

    base = _obter_pipeline_base_v156()
    if base is None:
        st.warning("Pipeline base ainda não foi executado. Vá ao painel 'Pipeline V14-FLEX ULTRA'.")
        return

    idx_alvo = base["idx_alvo"]
    id_alvo = base["id_alvo"]

    st.markdown(f"Série alvo: **{id_alvo}** (índice interno {idx_alvo})")

    resultado_light = replay_light_v156(df, meta, idx_alvo)

    pipeline_state = st.session_state.get(SESSAO_PIPELINE, {})
    pipeline_state["replay_light"] = resultado_light
    st.session_state[SESSAO_PIPELINE] = pipeline_state

    if not resultado_light.get("ok", False):
        st.error(resultado_light.get("motivo", "Replay LIGHT não conseguiu gerar previsão."))
        return

    st.markdown("#### Previsão FINAL (Replay LIGHT)")
    st.code(str(resultado_light["previsao_final"]))

    st.markdown("#### Vizinhos utilizados")
    st.write(resultado_light["vizinhos"])


def painel_replay_ultra_v156() -> None:
    st.markdown("## 📅 Replay ULTRA (V15.6 MAX)")

    df, meta = get_historico_v156()
    if df is None or meta is None:
        st.warning("Carregue o histórico e rode primeiro o Pipeline V14-FLEX ULTRA.")
        return

    base = _obter_pipeline_base_v156()
    if base is None:
        st.warning("Pipeline base ainda não foi executado. Vá ao painel 'Pipeline V14-FLEX ULTRA'.")
        return

    idx_alvo = base["idx_alvo"]
    id_alvo = base["id_alvo"]

    st.markdown(f"Série alvo: **{id_alvo}** (índice interno {idx_alvo})")

    resultado_ultra = replay_ultra_v156(df, meta, idx_alvo)

    pipeline_state = st.session_state.get(SESSAO_PIPELINE, {})
    pipeline_state["replay_ultra"] = resultado_ultra
    st.session_state[SESSAO_PIPELINE] = pipeline_state

    if not resultado_ultra.get("ok", False):
        st.error(resultado_ultra.get("motivo", "Replay ULTRA não conseguiu gerar previsão."))
        return

    st.markdown("#### Previsão FINAL (Replay ULTRA)")
    st.code(str(resultado_ultra["previsao_final"]))

    st.markdown("#### Vizinhos utilizados")
    st.write(resultado_ultra["vizinhos"])


def painel_replay_ultra_unitario_v156() -> None:
    st.markdown("## 🎯 Replay ULTRA Unitário (V15.6 MAX)")

    df, meta = get_historico_v156()
    if df is None or meta is None:
        st.warning("Carregue o histórico e rode primeiro o Pipeline V14-FLEX ULTRA.")
        return

    base = _obter_pipeline_base_v156()
    if base is None:
        st.warning("Pipeline base ainda não foi executado. Vá ao painel 'Pipeline V14-FLEX ULTRA'.")
        return

    idx_alvo = base["idx_alvo"]
    id_alvo = base["id_alvo"]

    st.markdown(f"Série alvo: **{id_alvo}** (índice interno {idx_alvo})")

    resultado_unit = replay_ultra_unitario_v156(df, meta, idx_alvo)

    pipeline_state = st.session_state.get(SESSAO_PIPELINE, {})
    pipeline_state["replay_unitario"] = resultado_unit
    st.session_state[SESSAO_PIPELINE] = pipeline_state

    if not resultado_unit.get("ok", False):
        st.error(resultado_unit.get("motivo", "Replay Unitário não conseguiu gerar previsão."))
        return

    st.markdown("#### Previsão FINAL (Replay ULTRA Unitário)")
    st.code(str(resultado_unit["previsao_final"]))

    st.markdown("#### Vizinhos utilizados")
    st.write(resultado_unit["vizinhos"])


# ============================================================
# PAINEL — MONITOR DE RISCO (k & k*) (V15.6 MAX)
# ============================================================

def painel_monitor_risco_v156() -> None:
    st.markdown("## 🚨 Monitor de Risco (k & k*) (V15.6 MAX)")

    df, meta = get_historico_v156()
    if df is None or meta is None:
        st.warning("Carregue o histórico e rode primeiro o Pipeline V14-FLEX ULTRA.")
        return

    base = _obter_pipeline_base_v156()
    if base is None:
        st.warning("Pipeline base ainda não foi executado. Vá ao painel 'Pipeline V14-FLEX ULTRA'.")
        return

    df_kstar = base["df_com_k_sentinela"]
    idx_alvo = base["idx_alvo"]
    id_alvo = base["id_alvo"]

    st.markdown(f"Série alvo: **{id_alvo}**")

    # Janela de visualização
    largura = st.slider(
        "Largura de visualização em torno da série alvo",
        min_value=20,
        max_value=200,
        value=60,
        step=10,
        key="v156_largura_monitor",
    )

    inicio = max(0, idx_alvo - largura // 2)
    fim = min(len(df_kstar) - 1, idx_alvo + largura // 2)

    df_view = df_kstar.iloc[inicio:fim+1][[ID_COLNAME, K_COLNAME, "k_sentinela"]].copy()

    st.markdown("#### Faixa local de k e k*")
    st.dataframe(df_view, use_container_width=True, hide_index=True)

    st.markdown("#### Barômetro k* (na série alvo)")
    st.info(base["regime_k_star"])


# ============================================================
# PAINEL — MODO TURBO++ ULTRA ANTI-RUÍDO (V15.6 MAX)
# ============================================================

def painel_modo_turbo_ultra_v156() -> None:
    st.markdown("## 🚀 Modo TURBO++ ULTRA Anti-Ruído (V15.6 MAX)")

    df, meta = get_historico_v156()
    if df is None or meta is None:
        st.warning("Carregue o histórico e rode primeiro o Pipeline V14-FLEX ULTRA.")
        return

    base = _obter_pipeline_base_v156()
    if base is None:
        st.warning("Pipeline base ainda não foi executado. Vá ao painel 'Pipeline V14-FLEX ULTRA'.")
        return

    # Garante que os replays foram executados
    pipeline_state = st.session_state.get(SESSAO_PIPELINE, {})
    replay_l = pipeline_state.get("replay_light", None)
    replay_u = pipeline_state.get("replay_ultra", None)
    replay_uu = pipeline_state.get("replay_unitario", None)

    if replay_l is None or replay_u is None or replay_uu is None:
        st.warning(
            "Execute antes os painéis de Replay (LIGHT, ULTRA e ULTRA Unitário) "
            "para alimentar o Modo TURBO++ ULTRA Anti-Ruído."
        )
        return

    resultado_replays = {
        "light": replay_l,
        "ultra": replay_u,
        "unitario": replay_uu,
    }

    # Calcula ruído condicional para o alvo
    ruido_info = calcular_ruido_condicional_v156(
        meta=meta,
        estat_janela=base["estat_janela"],
        df_freq=base["df_freq"],
        resultado_replays=resultado_replays,
        k_star_alvo=base["k_star_alvo"],
    )

    # Salva no estado
    pipeline_state["ruido"] = ruido_info
    st.session_state[SESSAO_PIPELINE] = pipeline_state

    # Monta TURBO++
    resultado_turbo = montar_previsao_turbo_ultra_v156(
        meta=meta,
        resultado_pipeline=base,
        resultado_replays=resultado_replays,
        info_ruido=ruido_info,
    )

    pipeline_state["turbo_ultra"] = resultado_turbo
    st.session_state[SESSAO_PIPELINE] = pipeline_state

    if not resultado_turbo.get("ok", False):
        st.warning("TURBO++ ULTRA entrou em modo fallback.")
    else:
        st.success("TURBO++ ULTRA Anti-Ruído rodado com sucesso.")

    st.markdown("#### 🔚 Previsão Final TURBO++ ULTRA (V15.6 MAX)")
    previsao = resultado_turbo["previsao_filtrada"] if resultado_turbo.get("ok", False) else resultado_turbo["previsao_final"]
    st.code(str(previsao))

    st.markdown("#### Detalhes da combinação de módulos")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Peso LIGHT", f"{resultado_turbo['peso_light']:.2f}")
        st.metric("Peso ULTRA", f"{resultado_turbo['peso_ultra']:.2f}")
    with col2:
        st.metric("Peso Unitário", f"{resultado_turbo['peso_unit']:.2f}")
        st.metric("Peso Núcleo", f"{resultado_turbo['peso_nucleo']:.2f}")

    st.markdown("#### Núcleo + Camadas fortes")
    st.caption(
        f"Núcleo: {resultado_turbo['nucleo']} | "
        f"Interseção forte: {resultado_turbo['intersecao_forte']}"
    )

    st.markdown("#### Ruído Condicional (resumo)")
    st.info(ruido_info["texto_resumo"])


# ============================================================
# PAINEL — RUÍDO CONDICIONAL (V15.6 MAX)
# ============================================================

def painel_ruido_condicional_v156() -> None:
    st.markdown("## 📊 Ruído Condicional (V15.6 MAX)")

    df, meta = get_historico_v156()
    if df is None or meta is None:
        st.warning("Carregue o histórico e rode primeiro o Pipeline V14-FLEX ULTRA.")
        return

    base = _obter_pipeline_base_v156()
    if base is None:
        st.warning("Pipeline base ainda não foi executado. Vá ao painel 'Pipeline V14-FLEX ULTRA'.")
        return

    pipeline_state = st.session_state.get(SESSAO_PIPELINE, {})
    ruido_info = pipeline_state.get("ruido", None)

    if ruido_info is None:
        st.warning("Ruído Condicional ainda não foi calculado. Rode o painel do Modo TURBO++ ULTRA.")
        return

    st.markdown("#### NR% — Intensidade de ruído condicional")
    st.metric("NR%", f"{ruido_info['nr_percent']:.1f}%")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Entropia (%)", f"{ruido_info['nr_basico_percent']:.1f}%")
    with col2:
        st.metric("Divergência (%)", f"{ruido_info['nr_div_percent']:.1f}%")
    with col3:
        st.metric("Tipo de Ruído", ruido_info["tipo"])

    st.markdown("#### Interpretação")
    st.info(ruido_info["texto_resumo"])



# ============================================================
# PARTE 5/6 — TESTES DE CONFIABILIDADE REAL + MODO 6 ACERTOS
# ============================================================

# Nesta parte ativamos:
#   🧪 Testes de Confiabilidade REAL (QDS / Backtest / Monte Carlo Profundo)
#   🎯 Modo 6 Acertos — Execução (V15.6 MAX)
#
# Tudo construído em cima:
#   - Pipeline V14-FLEX ULTRA (k* / Núcleo / Coberturas / Interseção)
#   - Replays LIGHT / ULTRA / ULTRA Unitário
#   - TURBO++ ULTRA Anti-Ruído
#   - Ruído Condicional (NR%)
#
# O Relatório Final V15.6 MAX será concluído na PARTE 6/6.


# ============================================================
# TESTES DE CONFIABILIDADE REAL (QDS / BACKTEST / MONTE CARLO)
# ============================================================

def _avaliar_previsao_contra_historico_v156(
    previsao: List[int],
    df_janela: pd.DataFrame,
    meta: Dict[str, Any],
    n_ultimos: int = 50,
) -> Dict[str, Any]:
    """
    Faz uma avaliação de "backtest estrutural" da previsão contra o histórico local da janela:

    - Para cada série da cauda da janela (até n_ultimos):
        • Conta quantos passageiros previstos estão presentes no carro.
        • Gera distribuição de acertos (0,1,2,...,n_passageiros).

    Isso não é um backtest de previsão temporal (pois ainda não temos o futuro),
    mas mede o quão "compatível" o padrão previsto é com a dinâmica recente da estrada.
    """
    col_passageiros = meta["col_passageiros"]
    n_pass = len(col_passageiros)

    if not previsao or len(previsao) != n_pass:
        return {
            "ok": False,
            "motivo": "Previsão incompatível com número de passageiros.",
        }

    n_series = len(df_janela)
    n_ultimos = max(1, min(n_ultimos, n_series))
    inicio = max(0, n_series - n_ultimos)

    previsao_set = set(int(x) for x in previsao)
    contagem_acertos = []

    for idx in range(inicio, n_series):
        linha = df_janela.iloc[idx]
        passageiros = [linha[c] for c in col_passageiros if not pd.isna(linha[c])]
        passageiros_set = set(int(x) for x in passageiros)
        acertos = len(previsao_set & passageiros_set)
        contagem_acertos.append(acertos)

    if not contagem_acertos:
        return {
            "ok": False,
            "motivo": "Janela insuficiente para avaliação.",
        }

    contagem_acertos = np.array(contagem_acertos)
    media = float(np.mean(contagem_acertos))
    mediana = float(np.median(contagem_acertos))
    maximo = int(np.max(contagem_acertos))

    # QDS estrutural: percentual de vezes com ≥1 acerto e com ≥2 acertos
    prop_ge1 = float(np.mean(contagem_acertos >= 1))
    prop_ge2 = float(np.mean(contagem_acertos >= 2))

    qds_ge1 = prop_ge1 * 100.0
    qds_ge2 = prop_ge2 * 100.0

    return {
        "ok": True,
        "media_acertos": media,
        "mediana_acertos": mediana,
        "max_acertos": maximo,
        "qds_ge1": qds_ge1,
        "qds_ge2": qds_ge2,
        "distribuicao": contagem_acertos,
        "n_amostras": len(contagem_acertos),
    }


def _monte_carlo_profundo_v156(
    meta: Dict[str, Any],
    df_freq: pd.DataFrame,
    previsao_referencia: List[int],
    n_sim: int = 2000,
) -> Dict[str, Any]:
    """
    Monte Carlo Profundo (V15.6 MAX):

    - Usa a distribuição de frequências da janela (df_freq) para sortear n_sim séries sintéticas.
    - Para cada série simulada, conta quantos passageiros coincidem com a previsão de referência.
    - Mede a "raridade" dos padrões de acerto em relação ao puro acaso condicionado à estrada atual.

    Saídas:
    - prob_ge1 / prob_ge2: probabilidade de obter ≥1 / ≥2 acertos por puro acaso condicionado.
    - taxa_raridade: 1 - prob_ge2 (quanto menor a probabilidade de ≥2 acertos, mais raro).
    """
    col_valor = "valor"
    col_freq_rel = "freq_rel"

    if df_freq is None or df_freq.empty or not previsao_referencia:
        return {
            "ok": False,
            "motivo": "Distribuição de frequências insuficiente para Monte Carlo.",
        }

    valores = df_freq[col_valor].values.astype(int)
    probs = df_freq[col_freq_rel].values.astype(float)

    if len(valores) == 0 or len(probs) == 0:
        return {
            "ok": False,
            "motivo": "Distribuição de frequências vazia.",
        }

    n_pass = len(previsao_referencia)
    previsao_set = set(int(x) for x in previsao_referencia)

    # Normalização de probs
    probs = probs / probs.sum()

    rng = np.random.default_rng()
    acertos_sim = []

    for _ in range(n_sim):
        sorteio = rng.choice(valores, size=n_pass, replace=False if len(valores) >= n_pass else True, p=probs)
        sorteio_set = set(int(x) for x in sorteio)
        acertos = len(previsao_set & sorteio_set)
        acertos_sim.append(acertos)

    acertos_sim = np.array(acertos_sim)
    prob_ge1 = float(np.mean(acertos_sim >= 1))
    prob_ge2 = float(np.mean(acertos_sim >= 2))

    taxa_raridade = 1.0 - prob_ge2  # quanto mais perto de 1, mais raro obter ≥2 acertos ao acaso

    return {
        "ok": True,
        "prob_ge1": prob_ge1,
        "prob_ge2": prob_ge2,
        "taxa_raridade": taxa_raridade,
        "distribuicao": acertos_sim,
        "n_sim": n_sim,
    }


def executar_testes_confiabilidade_v156(
    meta: Dict[str, Any],
    resultado_pipeline: Dict[str, Any],
    resultado_replays: Dict[str, Any],
    resultado_turbo: Dict[str, Any],
    info_ruido: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Amarra QDS / Backtest estrutural / Monte Carlo Profundo para a V15.6 MAX,
    focando na previsão TURBO++ ULTRA filtrada.
    """
    df_janela = resultado_pipeline["df_janela"]
    df_freq = resultado_pipeline["df_freq"]

    previsao_turbo = None
    if resultado_turbo.get("ok", False):
        previsao_turbo = resultado_turbo.get("previsao_filtrada", None)
    else:
        previsao_turbo = resultado_turbo.get("previsao_final", None)

    if not previsao_turbo:
        return {
            "ok": False,
            "motivo": "Sem previsão TURBO++ disponível para testes de confiabilidade.",
        }

    # Backtest estrutural (QDS-like)
    backtest_info = _avaliar_previsao_contra_historico_v156(
        previsao_turbo,
        df_janela,
        meta,
        n_ultimos=50,
    )

    # Monte Carlo profundo sobre a distribuição atual
    mc_info = _monte_carlo_profundo_v156(
        meta,
        df_freq,
        previsao_turbo,
        n_sim=2000,
    )

    if not backtest_info.get("ok", False) or not mc_info.get("ok", False):
        return {
            "ok": False,
            "motivo": "Falha parcial em backtest ou Monte Carlo.",
            "backtest": backtest_info,
            "monte_carlo": mc_info,
        }

    qds_ge1 = backtest_info["qds_ge1"]  # % de acertos ≥1
    qds_ge2 = backtest_info["qds_ge2"]  # % de acertos ≥2
    raridade = mc_info["taxa_raridade"] * 100.0  # 0–100

    nr_percent = info_ruido.get("nr_percent", 0.0)
    try:
        nr_val = float(nr_percent)
    except Exception:
        nr_val = 0.0

    # Confiabilidade composta (0–100):
    #   - Quanto maior QDS_ge2, melhor
    #   - Quanto maior raridade (difícil acertar ao acaso), melhor
    #   - Quanto menor NR% (ruído), melhor
    confiab_composta = (
        0.4 * qds_ge2 +
        0.3 * raridade +
        0.3 * (100.0 - nr_val)
    )
    confiab_composta = float(max(0.0, min(100.0, confiab_composta)))

    return {
        "ok": True,
        "previsao_turbo": previsao_turbo,
        "backtest": backtest_info,
        "monte_carlo": mc_info,
        "nr_percent": nr_val,
        "confiab_composta": confiab_composta,
        "qds_ge1": qds_ge1,
        "qds_ge2": qds_ge2,
    }


# ============================================================
# PAINEL — TESTES DE CONFIABILIDADE REAL (V15.6 MAX)
# ============================================================

def painel_testes_confiabilidade_real_v156() -> None:
    st.markdown("## 🧪 Testes de Confiabilidade REAL (V15.6 MAX)")

    df, meta = get_historico_v156()
    if df is None or meta is None:
        st.warning("Carregue o histórico e rode o Pipeline V14-FLEX ULTRA primeiro.")
        return

    pipeline_state = st.session_state.get(SESSAO_PIPELINE, {})
    base = pipeline_state.get("pipeline_base", None)
    replay_l = pipeline_state.get("replay_light", None)
    replay_u = pipeline_state.get("replay_ultra", None)
    replay_uu = pipeline_state.get("replay_unitario", None)
    turbo = pipeline_state.get("turbo_ultra", None)
    ruido_info = pipeline_state.get("ruido", None)

    if base is None or replay_l is None or replay_u is None or replay_uu is None or turbo is None or ruido_info is None:
        st.warning(
            "Para rodar os Testes de Confiabilidade REAL, é necessário:\n"
            "- Rodar o Pipeline V14-FLEX ULTRA\n"
            "- Rodar os Replays (LIGHT, ULTRA e Unitário)\n"
            "- Rodar o Modo TURBO++ ULTRA Anti-Ruído (que calcula também o Ruído Condicional)."
        )
        return

    resultado_replays = {
        "light": replay_l,
        "ultra": replay_u,
        "unitario": replay_uu,
    }

    resultado_conf = executar_testes_confiabilidade_v156(
        meta=meta,
        resultado_pipeline=base,
        resultado_replays=resultado_replays,
        resultado_turbo=turbo,
        info_ruido=ruido_info,
    )

    pipeline_state["confiabilidade"] = resultado_conf
    st.session_state[SESSAO_PIPELINE] = pipeline_state

    if not resultado_conf.get("ok", False):
        st.error(resultado_conf.get("motivo", "Falha nos Testes de Confiabilidade."))
        return

    st.success("Testes de Confiabilidade REAL executados com sucesso.")

    st.markdown("#### Confiabilidade Composta (0–100)")
    st.metric("Confiabilidade (V15.6 MAX)", f"{resultado_conf['confiab_composta']:.1f}%")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("QDS ≥1 acerto", f"{resultado_conf['qds_ge1']:.1f}%")
    with col2:
        st.metric("QDS ≥2 acertos", f"{resultado_conf['qds_ge2']:.1f}%")
    with col3:
        st.metric("NR% (ruído condicional)", f"{resultado_conf['nr_percent']:.1f}%")

    st.markdown("#### Monte Carlo Profundo — raridade dos acertos")
    mc = resultado_conf["monte_carlo"]
    st.metric("Probabilidade (≥1 acerto) ao acaso", f"{mc['prob_ge1']*100.0:.1f}%")
    st.metric("Probabilidade (≥2 acertos) ao acaso", f"{mc['prob_ge2']*100.0:.1f}%")
    st.metric("Taxa de raridade (≥2 acertos)", f"{mc['taxa_raridade']*100.0:.1f}%")


# ============================================================
# MODO 6 ACERTOS — EXECUÇÃO (V15.6 MAX)
# ============================================================

def _gerar_combinacoes_modo6_v156(
    candidatos: List[int],
    turbo_prev: List[int],
    nucleo: List[int],
    intersecao_forte: List[int],
    limite_combos: int = 64,
) -> List[Dict[str, Any]]:
    """
    Gera combinações de 6 números a partir de uma lista de candidatos,
    e atribui um score com base em:
        - Interseção com previsão TURBO++
        - Presença no Núcleo
        - Presença na intersecção forte
    """
    candidatos_unicos = sorted(list(set(candidatos)))
    n = len(candidatos_unicos)
    if n < 6:
        return []

    # Limitar número de candidatos para não explodir combinações
    if n > 10:
        candidatos_unicos = candidatos_unicos[:10]  # C(10,6)=210, ainda aceitável
        n = len(candidatos_unicos)

    turbo_set = set(int(x) for x in turbo_prev)
    nucleo_set = set(int(x) for x in nucleo)
    intersec_set = set(int(x) for x in intersecao_forte)

    combos = []
    for comb in itertools.combinations(candidatos_unicos, 6):
        comb_set = set(comb)
        score = 0.0

        inter_turbo = len(comb_set & turbo_set)
        inter_nucleo = len(comb_set & nucleo_set)
        inter_intersec = len(comb_set & intersec_set)

        score += 4.0 * inter_turbo
        score += 2.0 * inter_nucleo
        score += 3.0 * inter_intersec

        combos.append(
            {
                "combo": tuple(sorted(comb)),
                "score": score,
                "hits_turbo": inter_turbo,
                "hits_nucleo": inter_nucleo,
                "hits_intersec": inter_intersec,
            }
        )

    combos_ordenados = sorted(combos, key=lambda x: (-x["score"], x["combo"]))
    return combos_ordenados[:limite_combos]


def executar_modo_6_acertos_v156(
    meta: Dict[str, Any],
    resultado_pipeline: Dict[str, Any],
    resultado_turbo: Dict[str, Any],
    resultado_conf: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Monta as recomendações semi-automáticas do Modo 6 Acertos:

    - Usa previsão TURBO++ filtrada como eixo principal.
    - Usa Núcleo e Interseção Forte para formar um conjunto candidato.
    - Gera combinações de 6 números com score.
    - Leva em conta confiabilidade e ruído (já embutidos na escolha de candidatos).
    """
    nucleo = resultado_pipeline.get("nucleo", [])
    intersecao_forte = resultado_pipeline.get("intersecao_base", {}).get("intersecao_forte", [])
    cobertura_suave = resultado_pipeline.get("cobertura_suave", [])

    if resultado_turbo.get("ok", False):
        turbo_prev = resultado_turbo.get("previsao_filtrada", [])
    else:
        turbo_prev = resultado_turbo.get("previsao_final", [])

    if not turbo_prev:
        return {
            "ok": False,
            "motivo": "Sem previsão TURBO++ disponível para Modo 6 Acertos.",
        }

    # Conjunto de candidatos:
    candidatos = []
    candidatos.extend(turbo_prev)
    candidatos.extend(nucleo)
    candidatos.extend(intersecao_forte)
    candidatos.extend(cobertura_suave)

    combos = _gerar_combinacoes_modo6_v156(
        candidatos,
        turbo_prev,
        nucleo,
        intersecao_forte,
        limite_combos=32,
    )

    if not combos:
        return {
            "ok": False,
            "motivo": "Não foi possível gerar combinações de 6 acertos com os candidatos atuais.",
        }

    # Metainformações úteis para o Relatório Final
    confiab = resultado_conf.get("confiab_composta", 0.0)
    qds_ge2 = resultado_conf.get("qds_ge2", 0.0)

    return {
        "ok": True,
        "turbo_prev": turbo_prev,
        "nucleo": nucleo,
        "intersecao_forte": intersecao_forte,
        "cobertura_suave": cobertura_suave,
        "confiabilidade": confiab,
        "qds_ge2": qds_ge2,
        "combos": combos,
    }


# ============================================================
# PAINEL — MODO 6 ACERTOS — EXECUÇÃO (V15.6 MAX)
# ============================================================

def painel_modo_6_acertos_execucao_v156() -> None:
    st.markdown("## 🎯 Modo 6 Acertos — Execução (V15.6 MAX)")

    df, meta = get_historico_v156()
    if df is None or meta is None:
        st.warning("Carregue o histórico e rode o Pipeline V14-FLEX ULTRA primeiro.")
        return

    pipeline_state = st.session_state.get(SESSAO_PIPELINE, {})
    base = pipeline_state.get("pipeline_base", None)
    turbo = pipeline_state.get("turbo_ultra", None)
    ruido_info = pipeline_state.get("ruido", None)
    conf = pipeline_state.get("confiabilidade", None)

    if base is None or turbo is None or ruido_info is None or conf is None:
        st.warning(
            "Para ativar o Modo 6 Acertos, é necessário:\n"
            "- Pipeline V14-FLEX ULTRA rodado\n"
            "- Replays (LIGHT / ULTRA / Unitário) rodados\n"
            "- TURBO++ ULTRA Anti-Ruído rodado\n"
            "- Testes de Confiabilidade REAL rodados\n"
        )
        return

    resultado_modo6 = executar_modo_6_acertos_v156(
        meta=meta,
        resultado_pipeline=base,
        resultado_turbo=turbo,
        resultado_conf=conf,
    )

    pipeline_state["modo_6"] = resultado_modo6
    st.session_state[SESSAO_PIPELINE] = pipeline_state

    if not resultado_modo6.get("ok", False):
        st.error(resultado_modo6.get("motivo", "Falha ao montar Modo 6 Acertos."))
        return

    st.success("Modo 6 Acertos V15.6 MAX montado (modo semi-automático).")

    st.markdown("#### Contexto de Risco & Confiabilidade")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Confiabilidade (V15.6 MAX)", f"{conf['confiab_composta']:.1f}%")
    with col2:
        st.metric("QDS ≥2 acertos", f"{conf['qds_ge2']:.1f}%")
    with col3:
        st.metric("NR% (ruído)", f"{conf['nr_percent']:.1f}%")

    st.markdown("#### Previsão TURBO++ ULTRA usada como eixo")
    st.code(str(resultado_modo6["turbo_prev"]))

    st.markdown("#### Núcleo & Interseção em jogo")
    st.caption(
        f"Núcleo: {resultado_modo6['nucleo']} | "
        f"Interseção forte: {resultado_modo6['intersecao_forte']} | "
        f"Cobertura suave: {resultado_modo6['cobertura_suave']}"
    )

    st.markdown("#### Recomendações de Combinações — Modo 6 (semi-automático)")
    st.markdown(
        """
        <div class="small-text">
        As combinações abaixo são sugestões do sistema V15.6 MAX, já ponderadas
        por Núcleo, Interseção e aderência à previsão TURBO++ ULTRA.
        Você escolhe manualmente quais deseja utilizar.
        </div>
        """,
        unsafe_allow_html=True,
    )

    combos = resultado_modo6["combos"]
    textos_combos = []
    for c in combos:
        combo_txt = (
            f"{c['combo']}  |  score={c['score']:.1f}  |  "
            f"hits TURBO={c['hits_turbo']}  |  Núcleo={c['hits_nucleo']}  |  Intersec={c['hits_intersec']}"
        )
        textos_combos.append(combo_txt)

    selecao = st.multiselect(
        "Selecione as combinações que você considera mais interessantes (modo semi-automático):",
        options=textos_combos,
        default=textos_combos[: min(6, len(textos_combos))],
        key="v156_modo6_selecao",
    )

    st.markdown("#### Combinações selecionadas (para uso manual)")
    if selecao:
        for linha in selecao:
            st.write("- ", linha)
    else:
        st.write("Nenhuma combinação selecionada ainda.")

    # fim do painel
    return
# ============================================================
# PARTE 6/6 — RELATÓRIO FINAL V15.6 MAX
# ============================================================

# Nesta parte concluímos:
#   📜 Painel completo do Relatório Final V15.6 MAX
#
# O Relatório Final integra:
#   - Histórico / série alvo
#   - Pipeline V14-FLEX ULTRA (Núcleo + Coberturas + k*)
#   - Replays LIGHT / ULTRA / ULTRA Unitário
#   - TURBO++ ULTRA Anti-Ruído
#   - Ruído Condicional (NR%)
#   - Testes de Confiabilidade REAL (QDS / Backtest / Monte Carlo)
#   - Modo 6 Acertos — recomendações semi-automáticas
#
# Nada é simplificado: o Relatório é denso e explicativo, mantendo o jeitão
# de “laudo técnico da estrada” da V15.6 MAX.


# ============================================================
# GERADOR DO RELATÓRIO FINAL V15.6 MAX
# ============================================================

def gerar_relatorio_final_v156() -> Dict[str, Any]:
    """
    Consolida todas as informações do pipeline V15.6 MAX em um relatório final
    textual estruturado.

    Retorna um dicionário com:
    - ok
    - texto_markdown
    - blocos (subseções separadas, se necessário)
    """
    df, meta = get_historico_v156()
    if df is None or meta is None:
        return {
            "ok": False,
            "motivo": "Histórico não carregado.",
        }

    pipeline_state = st.session_state.get(SESSAO_PIPELINE, {})
    base = pipeline_state.get("pipeline_base", None)
    replay_l = pipeline_state.get("replay_light", None)
    replay_u = pipeline_state.get("replay_ultra", None)
    replay_uu = pipeline_state.get("replay_unitario", None)
    turbo = pipeline_state.get("turbo_ultra", None)
    ruido_info = pipeline_state.get("ruido", None)
    conf = pipeline_state.get("confiabilidade", None)
    modo6 = pipeline_state.get("modo_6", None)

    if base is None:
        return {
            "ok": False,
            "motivo": "Pipeline V14-FLEX ULTRA ainda não foi executado.",
        }
    if replay_l is None or replay_u is None or replay_uu is None:
        return {
            "ok": False,
            "motivo": "Replays (LIGHT / ULTRA / Unitário) ainda não foram executados.",
        }
    if turbo is None:
        return {
            "ok": False,
            "motivo": "Modo TURBO++ ULTRA Anti-Ruído ainda não foi executado.",
        }
    if ruido_info is None:
        return {
            "ok": False,
            "motivo": "Ruído Condicional ainda não foi calculado (rode TURBO++ ULTRA).",
        }
    if conf is None:
        return {
            "ok": False,
            "motivo": "Testes de Confiabilidade REAL ainda não foram executados.",
        }
    if modo6 is None:
        return {
            "ok": False,
            "motivo": "Modo 6 Acertos ainda não foi montado.",
        }

    # --------------------------------------------------------
    # Bloco 1 — Contexto da estrada e série alvo
    # --------------------------------------------------------
    id_alvo = base["id_alvo"]
    idx_alvo = base["idx_alvo"]
    janela_ini = base["janela_inicio"] + 1
    janela_fim = base["janela_fim"] + 1
    tamanho_janela = base["tamanho_janela"]

    k_star_alvo = base.get("k_star_alvo", float("nan"))
    regime_k_star = base.get("regime_k_star", "")
    estat_jan = base.get("estat_janela", {})

    bloco1 = []
    bloco1.append(f"### 1. Contexto da Estrada e Série Alvo\n")
    bloco1.append(
        f"- **Série alvo:** `{id_alvo}` (índice interno {idx_alvo})\n"
        f"- **Janela ativa analisada:** séries `{janela_ini}` até `{janela_fim}` "
        f"({tamanho_janela} séries recentes)\n"
    )
    bloco1.append(
        "- **Dispersão local dos passageiros:**\n"
        f"  - Mínimo na janela: `{estat_jan.get('min_val_janela', '-')}`\n"
        f"  - Máximo na janela: `{estat_jan.get('max_val_janela', '-')}`\n"
        f"  - Amplitude: `{estat_jan.get('amplitude_janela', '-')}`\n"
    )
    bloco1.append(
        "- **Sensores de guardas (k) na janela:**\n"
        f"  - k médio: `{estat_jan.get('k_medio_janela', '-')}`\n"
        f"  - k máximo: `{estat_jan.get('k_max_janela', '-')}`\n"
    )
    bloco1.append(
        "- **Sentinela k★ (k*):**\n"
        f"  - k* médio na janela: `{estat_jan.get('k_star_medio_janela', '-')}`\n"
        f"  - k* máximo na janela: `{estat_jan.get('k_star_max_janela', '-')}`\n"
        f"  - k* na série alvo: `{k_star_alvo:.1f}`\n"
        f"  - Barômetro de regime: {regime_k_star}\n"
    )

    # --------------------------------------------------------
    # Bloco 2 — Núcleo, Coberturas e Interseção Estatística
    # --------------------------------------------------------
    nucleo = base.get("nucleo", [])
    cobertura_suave = base.get("cobertura_suave", [])
    cobertura_agressiva = base.get("cobertura_agressiva", [])
    intersec_forte = base.get("intersecao_base", {}).get("intersecao_forte", [])

    bloco2 = []
    bloco2.append("### 2. Núcleo, Coberturas e Interseção Estatística\n")
    bloco2.append(
        f"- **Núcleo (valores mais resilientes na janela):** `{nucleo}`\n"
        f"- **Cobertura suave (faixa intermediária):** `{cobertura_suave}`\n"
        f"- **Cobertura agressiva (cauda de apoio):** `{cobertura_agressiva}`\n"
        f"- **Interseção forte entre camadas:** `{intersec_forte}`\n"
    )
    bloco2.append(
        "O Núcleo e suas coberturas representam o “coração” estatístico da estrada "
        "neste trecho, servindo como base de sustentação para as previsões dos módulos "
        "de Replay, TURBO++ e Modo 6 Acertos.\n"
    )

    # --------------------------------------------------------
    # Bloco 3 — Replays (LIGHT / ULTRA / Unitário)
    # --------------------------------------------------------
    prev_l = replay_l.get("previsao_final", [])
    prev_u = replay_u.get("previsao_final", [])
    prev_uu = replay_uu.get("previsao_final", [])

    bloco3 = []
    bloco3.append("### 3. Módulos de Replay (LIGHT / ULTRA / Unitário)\n")
    bloco3.append(
        f"- **Replay LIGHT — previsão média suave:** `{prev_l}`\n"
        f"- **Replay ULTRA — previsão ponderada por semelhança (k≥1):** `{prev_u}`\n"
        f"- **Replay ULTRA Unitário — previsão focada em vizinhos mais próximos:** `{prev_uu}`\n"
    )
    bloco3.append(
        "Os Replays analisam a repetição de padrões de carros similares no trecho recente "
        "da estrada, oferecendo três visões complementares: leve, densa e profundamente "
        "focada em um único padrão (Unitário).\n"
    )

    # --------------------------------------------------------
    # Bloco 4 — TURBO++ ULTRA Anti-Ruído & Ruído Condicional
    # --------------------------------------------------------
    if turbo.get("ok", False):
        prev_turbo = turbo.get("previsao_filtrada", turbo.get("previsao_bruta", []))
        turbo_modo_fallback = False
    else:
        prev_turbo = turbo.get("previsao_final", [])
        turbo_modo_fallback = True

    nr_percent = ruido_info.get("nr_percent", 0.0)
    tipo_ruido = ruido_info.get("tipo", "?")
    texto_ruido = ruido_info.get("texto_resumo", "")

    bloco4 = []
    bloco4.append("### 4. Modo TURBO++ ULTRA Anti-Ruído e Ruído Condicional\n")
    bloco4.append(
        f"- **Previsão consolidada TURBO++ ULTRA:** `{prev_turbo}`\n"
        f"- **Modo TURBO++:** {'fallback (Núcleo)📉' if turbo_modo_fallback else 'modo completo (Replays + Núcleo)🚀'}\n"
    )
    bloco4.append(
        f"- **NR% (intensidade de ruído condicional):** `{nr_percent:.1f}%` "
        f"(tipo de ruído: `{tipo_ruido}`)\n"
    )
    bloco4.append(f"- **Interpretação do ruído:** {texto_ruido}\n")

    bloco4.append(
        "O TURBO++ ULTRA combina as previsões dos Replays com o Núcleo da estrada, "
        "ajustando os pesos de cada módulo conforme o regime de k* e o nível de ruído (NR%).\n"
    )

    # --------------------------------------------------------
    # Bloco 5 — Testes de Confiabilidade REAL
    # --------------------------------------------------------
    confiab = conf.get("confiab_composta", 0.0)
    qds_ge1 = conf.get("qds_ge1", 0.0)
    qds_ge2 = conf.get("qds_ge2", 0.0)
    nr_conf = conf.get("nr_percent", nr_percent)

    mc = conf.get("monte_carlo", {})
    prob_ge1 = mc.get("prob_ge1", 0.0) * 100.0
    prob_ge2 = mc.get("prob_ge2", 0.0) * 100.0
    raridade = mc.get("taxa_raridade", 0.0) * 100.0

    bloco5 = []
    bloco5.append("### 5. Testes de Confiabilidade REAL\n")
    bloco5.append(
        f"- **Confiabilidade composta (V15.6 MAX):** `{confiab:.1f}%`\n"
        f"- **QDS ≥1 acerto (compatibilidade estrutural):** `{qds_ge1:.1f}%`\n"
        f"- **QDS ≥2 acertos (compatibilidade forte):** `{qds_ge2:.1f}%`\n"
    )
    bloco5.append(
        "- **Monte Carlo Profundo (condicionado à estrada atual):**\n"
        f"  - Prob. ≥1 acerto ao acaso: `{prob_ge1:.1f}%`\n"
        f"  - Prob. ≥2 acertos ao acaso: `{prob_ge2:.1f}%`\n"
        f"  - Taxa de raridade (≥2 acertos): `{raridade:.1f}%`\n"
    )
    bloco5.append(
        f"- **NR% usado na composição de confiabilidade:** `{nr_conf:.1f}%`\n"
    )
    bloco5.append(
        "Esses testes avaliam o quanto a previsão TURBO++ ULTRA se alinha com a "
        "dinâmica recente da estrada e quão improvável seria atingir os mesmos "
        "padrões de acerto apenas por acaso, considerando a distribuição real "
        "de passageiros.\n"
    )

    # --------------------------------------------------------
    # Bloco 6 — Modo 6 Acertos (recomendações)
    # --------------------------------------------------------
    combos_modo6 = modo6.get("combos", [])
    turbo_prev_m6 = modo6.get("turbo_prev", [])

    bloco6 = []
    bloco6.append("### 6. Modo 6 Acertos — Recomendações Semi-Automáticas\n")
    bloco6.append(
        f"- **Previsão TURBO++ utilizada como eixo:** `{turbo_prev_m6}`\n"
        f"- **Núcleo em jogo:** `{modo6.get('nucleo', [])}`\n"
        f"- **Interseção forte usada:** `{modo6.get('intersecao_forte', [])}`\n"
    )
    bloco6.append(
        "O sistema gera combinações de 6 números a partir de TURBO++, Núcleo, "
        "Interseção forte e Cobertura suave, atribuindo um score para cada combinação "
        "com base na aderência a esses blocos estruturais.\n\n"
        "Abaixo, as principais recomendações (ordenadas por score):\n"
    )

    linhas_combos = []
    for c in combos_modo6:
        combo = c["combo"]
        score = c["score"]
        hits_turbo = c["hits_turbo"]
        hits_nucleo = c["hits_nucleo"]
        hits_intersec = c["hits_intersec"]
        linhas_combos.append(
            f"- `{combo}`  | score={score:.1f}  | hits TURBO={hits_turbo}  | "
            f"Núcleo={hits_nucleo}  | Intersec={hits_intersec}"
        )

    if not linhas_combos:
        linhas_combos.append("- (não foi possível montar combinações robustas com os candidatos atuais)")

    bloco6.extend(linhas_combos)
    bloco6.append("\n> **Modo 6 Acertos V15.6 MAX** é semi-automático: o sistema aponta as "
                  "combinações com melhor sustentação estatística, mas a decisão final "
                  "sobre quais listas utilizar continua sendo manual.")

    # --------------------------------------------------------
    # Consolidação dos blocos
    # --------------------------------------------------------
    texto_final = "\n".join(bloco1 + ["\n"] + bloco2 + ["\n"] + bloco3 + ["\n"] +
                            bloco4 + ["\n"] + bloco5 + ["\n"] + bloco6)

    relatorio = {
        "ok": True,
        "texto_markdown": texto_final,
        "blocos": {
            "contexto": bloco1,
            "nucleo": bloco2,
            "replays": bloco3,
            "turbo_ruido": bloco4,
            "confiabilidade": bloco5,
            "modo6": bloco6,
        },
    }

    # Atualiza sessão
    atualizar_relatorio_final_v156(relatorio)

    return relatorio


# ============================================================
# PAINEL — RELATÓRIO FINAL V15.6 MAX
# ============================================================

def painel_relatorio_final_v156() -> None:
    st.markdown("## 📜 Relatório Final V15.6 MAX")

    st.markdown(
        """
        <div class="small-text justified">
        Este painel consolida todos os módulos da V15.6 MAX em um único laudo:
        Pipeline V14-FLEX ULTRA, Replays, Monitor de Risco (k & k*),
        TURBO++ ULTRA Anti-Ruído, Ruído Condicional, Testes de Confiabilidade REAL
        e Modo 6 Acertos semi-automático.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Gerar / Atualizar Relatório Final V15.6 MAX", key="v156_btn_relatorio_final"):
        rel = gerar_relatorio_final_v156()
    else:
        # Tenta usar o relatório já existente na sessão
        rel = st.session_state.get(SESSAO_PIPELINE, {}).get("relatorio_final", None)
        if rel is None or not rel.get("ok", False):
            st.info("Clique no botão acima para gerar o Relatório Final V15.6 MAX.")
            return

    if not rel.get("ok", False):
        st.error(rel.get("motivo", "Não foi possível gerar o Relatório Final."))
        return

    st.markdown("---")
    st.markdown("### 📌 Laudo consolidado da estrada (V15.6 MAX)")

    st.markdown(rel["texto_markdown"])


# ============================================================
# MAIN FINAL — VERSÃO COMPLETA V15.6 MAX (TODOS OS PAINÉIS)
# ============================================================

def main_v156():
    """
    Função principal da V15.6 MAX — controla toda a navegação.
    """
    init_session_state_v156()

    # 🔥 Captura o painel escolhido na navegação
    painel = construir_navegacao_v156()

    # 🔥 Roteamento correto para todos os painéis
    if painel == "📥 Histórico — Entrada FLEX ULTRA (V15.6 MAX)":
        painel_historico_entrada_v156()

    elif painel == "🔍 Pipeline V14-FLEX ULTRA (V15.6 MAX)":
        painel_pipeline_v14_flex_ultra_v156()

    elif painel == "💡 Replay LIGHT (V15.6 MAX)":
        painel_replay_light_v156()

    elif painel == "📅 Replay ULTRA (V15.6 MAX)":
        painel_replay_ultra_v156()

    elif painel == "🎯 Replay ULTRA Unitário (V15.6 MAX)":
        painel_replay_ultra_unitario_v156()

    elif painel == "🚨 Monitor de Risco (k & k*) (V15.6 MAX)":
        painel_monitor_risco_v156()

    elif painel == "📊 Ruído Condicional (V15.6 MAX)":
        painel_ruido_condicional_v156()

    elif painel == "🚀 Modo TURBO++ ULTRA Anti-Ruído (V15.6 MAX)":
        painel_modo_turbo_ultra_v156()

    elif painel == "🧪 Testes de Confiabilidade REAL (V15.6 MAX)":
        painel_testes_confiabilidade_real_v156()

    elif painel == "🎯 Modo 6 Acertos — Execução (V15.6 MAX)":
        painel_modo_6_acertos_execucao_v156()

    elif painel == "📜 Relatório Final V15.6 MAX":
        painel_relatorio_final_v156()

# ============================================================
# PONTO DE ENTRADA DA APLICAÇÃO
# ============================================================
if __name__ == "__main__":
    main_v156()

