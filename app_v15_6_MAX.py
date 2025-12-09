# ============================================================
# Predict Cars V15.6 MAX
# Versão MAX: núcleo + coberturas + interseção estatística
# Pipeline V14-FLEX ULTRA + Replay LIGHT/ULTRA + TURBO++ ULTRA Anti-Ruído
# + Monitor de Risco (k & k*) + Painel de Ruído Condicional (NR%)
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
# Configuração da página (V15.6 MAX)
# ------------------------------------------------------------
st.set_page_config(
    page_title="Predict Cars V15.6 MAX",
    page_icon="🚗",
    layout="wide",
)

# ------------------------------------------------------------
# Estilos globais (mantendo jeitão técnico e denso)
# ------------------------------------------------------------
CSS_GLOBAL = """
<style>
/* Layout geral */
.main > div {
    padding-top: 0.5rem;
    padding-bottom: 2rem;
}

/* Títulos principais */
h1, h2, h3 {
    font-family: "Segoe UI", Roboto, sans-serif;
}

/* Caixas de diagnóstico / alertas */
.blocao {
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    border: 1px solid rgba(255,255,255,0.08);
    background: linear-gradient(
        135deg,
        rgba(0,0,0,0.65),
        rgba(40,40,40,0.95)
    );
    color: #f2f2f2;
    font-size: 0.9rem;
}

/* Chips/Badges */
.badge-ok {
    display: inline-block;
    padding: 0.15rem 0.45rem;
    margin-right: 0.25rem;
    border-radius: 999px;
    background: #14532d;
    color: #bbf7d0;
    font-size: 0.75rem;
}
.badge-warn {
    display: inline-block;
    padding: 0.15rem 0.45rem;
    margin-right: 0.25rem;
    border-radius: 999px;
    background: #92400e;
    color: #fed7aa;
    font-size: 0.75rem;
}
.badge-risk {
    display: inline-block;
    padding: 0.15rem 0.45rem;
    margin-right: 0.25rem;
    border-radius: 999px;
    background: #7f1d1d;
    color: #fecaca;
    font-size: 0.75rem;
}

/* Tabelas compactas */
table {
    font-size: 0.85rem;
}
</style>
"""

st.markdown(CSS_GLOBAL, unsafe_allow_html=True)


# ============================================================
# Sessão / Estado Global V15.6 MAX
# ============================================================

def init_session_state_v156() -> None:
    """
    Inicializa todas as chaves de sessão necessárias para o V15.6 MAX.
    Mantém jeitão de app denso, com estado explícito e controlado.
    """
    ss = st.session_state

    # Histórico FLEX ULTRA
    ss.setdefault("df_historico", None)          # DataFrame principal (carros/pass.
    ss.setdefault("df_historico_bruto", None)    # Texto/DF bruto antes de limpeza
    ss.setdefault("n_passageiros", None)         # Número de passageiros detectado
    ss.setdefault("k_col_name", None)           # Nome da coluna k (se existir)
    ss.setdefault("indice_ultima_serie", None)   # Índice da última série (ex: 5788)

    # Sentinelas k / k*
    ss.setdefault("serie_k", None)               # Série com valores k (histórico)
    ss.setdefault("serie_k_star", None)          # Série com valores k* estimados
    ss.setdefault("diagnosticos_k_kstar", {})    # Diagnósticos locais por janela

    # Ruído condicional / NR%
    ss.setdefault("nr_mapa_global", None)        # Estrutura de ruído global
    ss.setdefault("nr_mapa_condicional", None)   # Ruído condicional por regime
    ss.setdefault("nr_resumo_textual", "")       # Texto explicativo global

    # Pipeline / Replay
    ss.setdefault("pipeline_cache", {})          # Resultados intermediários do pipeline
    ss.setdefault("replay_light_result", None)
    ss.setdefault("replay_ultra_result", None)
    ss.setdefault("replay_unitario_result", None)

    # TURBO++ ULTRA Anti-Ruído
    ss.setdefault("turbo_ultra_result", None)
    ss.setdefault("turbo_ultra_pesos", {
        "peso_s6": 0.5,
        "peso_mc": 0.35,
        "peso_micro": 0.15,
    })
    ss.setdefault("turbo_ultra_logs", [])

    # Testes de Confiabilidade REAL
    ss.setdefault("confiabilidade_resultados", None)
    ss.setdefault("confiabilidade_n_prev", 50)   # qtde padrão de previsões em testes
    ss.setdefault("confiabilidade_alvo", 0.65)   # alvo de acurácia, ex: 65%

    # Modo 6 Acertos V15.6 MAX
    ss.setdefault("modo6_inputs", {})
    ss.setdefault("modo6_resultados", None)
    ss.setdefault("modo6_risco_ok", False)       # liberado ou não para usar 6 acertos

    # Relatório Final V15.6 MAX
    ss.setdefault("relatorio_final_texto", "")
    ss.setdefault("relatorio_final_estrutura", {})

    # Diagnósticos de impacto & plano de calibração
    ss.setdefault("diag_impacto", {})
    ss.setdefault("plano_calibracao", {})

    # Anti-zumbi / proteções
    ss.setdefault("limite_max_janela", 600)      # limite padrão de janelas longas
    ss.setdefault("limite_max_prev", 300)        # limite padrão de previsões intensivas
    ss.setdefault("flag_modo_seguro", True)      # ativa modo seguro para DF grandes


init_session_state_v156()


# ============================================================
# Utilitários gerais
# ============================================================

def formatar_lista_passageiros(lista: List[int]) -> str:
    """Formata lista de passageiros no jeitão compacto do app."""
    return " ".join(str(int(x)) for x in lista)


def texto_em_blocos(texto: str, largura: int = 80) -> str:
    """Quebra texto longo em blocos, mantendo leitura agradável."""
    return "\n".join(textwrap.wrap(texto, width=largura))


def exibir_bloco_mensagem(titulo: str, corpo: str, emoji: str = "ℹ️") -> None:
    """Exibe um bloco padronizado de mensagem/diagnóstico."""
    st.markdown(
        f"""
        <div class="blocao">
            <strong>{emoji} {titulo}</strong><br/>
            {corpo}
        </div>
        """,
        unsafe_allow_html=True,
    )


def limitar_operacao(len_df: int, limite_max: int, contexto: str) -> bool:
    """
    Proteção anti-zumbi.
    Retorna True se for SEGURO rodar a operação, False se for melhor abortar.
    """
    if len_df <= 0:
        return False
    if len_df <= limite_max:
        return True

    exibir_bloco_mensagem(
        "Proteção Anti-Zumbi ativada",
        texto_em_blocos(
            f"A operação '{contexto}' foi bloqueada automaticamente "
            f"porque o histórico possui {len_df} séries, acima do limite "
            f"de segurança configurado ({limite_max}). "
            "Ajuste a janela ou o limite em 'Configurações avançadas' para rodar mesmo assim."
        ),
        emoji="🧱",
    )
    return False


# ============================================================
# Parsing FLEX ULTRA — Histórico (arquivo + copiar/colar)
# ============================================================

def detectar_separador_linha(bruto: str) -> str:
    """
    Detecta o separador mais provável nas linhas do histórico.
    Considera ; , e espaço. Mantém jeitão genérico/robusto.
    """
    amostra = "\n".join(bruto.strip().splitlines()[:10])
    candidatos = [";", ",", " "]
    contagens = {sep: amostra.count(sep) for sep in candidatos}
    if all(v == 0 for v in contagens.values()):
        return " "
    return max(contagens, key=contagens.get)


def carregar_historico_de_texto(
    texto: str,
    tem_coluna_k: bool = True,
) -> pd.DataFrame:
    """
    Converte texto bruto em DataFrame FLEX ULTRA.
    Cada linha = um carro; último campo opcional = k.
    """
    if not texto or not texto.strip():
        raise ValueError("Texto do histórico está vazio.")

    sep = detectar_separador_linha(texto)
    linhas = [l.strip() for l in texto.strip().splitlines() if l.strip()]
    registros: List[List[int]] = []

    for linha in linhas:
        partes = [p for p in linha.split(sep) if p != ""]
        try:
            nums = [int(p) for p in partes]
        except ValueError:
            # Tenta de novo removendo caracteres estranhos
            limpos = []
            for p in partes:
                p2 = "".join(ch for ch in p if ch.isdigit())
                if p2 == "":
                    raise
                limpos.append(int(p2))
            nums = limpos

        registros.append(nums)

    if not registros:
        raise ValueError("Nenhum registro numérico foi identificado no histórico.")

    # Descobrir nº de passageiros e se há k
    tamanhos = sorted(set(len(r) for r in registros))
    if len(tamanhos) > 2:
        raise ValueError(
            f"Foram encontrados registros com tamanhos diferentes: {tamanhos}. "
            "O histórico FLEX ULTRA deve ter todos os carros com mesmo tamanho, "
            "ou no máximo diferença de 1 por causa da coluna k."
        )

    n_cols = max(tamanhos)
    df = pd.DataFrame(registros)

    # Heurística: se tem_coluna_k=True e o menor tamanho == n_cols - 1, assumimos k na última coluna
    if tem_coluna_k and len(tamanhos) == 2 and min(tamanhos) == n_cols - 1:
        # Preenche NaN com 0 na última coluna
        df = df.reindex(columns=range(n_cols))
        df = df.fillna(method="ffill", axis=1)  # fallback simples, não crítico
        col_k = n_cols - 1
    elif tem_coluna_k and len(tamanhos) == 1:
        # Mesmo tamanho para todas — assumimos última col como k
        col_k = n_cols - 1
    else:
        col_k = None

    # Renomear colunas
    col_names = []
    for i in range(n_cols):
        if col_k is not None and i == col_k:
            col_names.append("k")
        else:
            col_names.append(f"P{i+1}")
    df.columns = col_names

    return df


def carregar_historico_de_arquivo(
    arquivo,
    tem_coluna_k: bool = True,
) -> pd.DataFrame:
    """
    Lê arquivo CSV/TXT para DataFrame FLEX ULTRA.
    Usa heurísticas similares às de texto.
    """
    if arquivo is None:
        raise ValueError("Nenhum arquivo enviado.")

    conteudo = arquivo.read()
    if isinstance(conteudo, bytes):
        conteudo = conteudo.decode("utf-8", errors="ignore")

    return carregar_historico_de_texto(conteudo, tem_coluna_k=tem_coluna_k)


def analisar_historico_flex_ultra(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extrai informações básicas do histórico FLEX ULTRA:
    - nº de séries
    - nº de passageiros
    - presença de k
    - índice da última série
    """
    if df is None or df.empty:
        raise ValueError("Histórico vazio para análise.")

    colunas = list(df.columns)
    tem_k = "k" in colunas
    n_series = len(df)
    n_passageiros = len(colunas) - (1 if tem_k else 0)
    idx_ultima = n_series - 1  # índice 0-based, mas é comum falar C1..C5788

    return {
        "tem_k": tem_k,
        "n_series": n_series,
        "n_passageiros": n_passageiros,
        "indice_ultima_serie": idx_ultima,
    }


def registrar_historico_na_sessao(df: pd.DataFrame) -> None:
    """
    Salva histórico e metadados na sessão V15.6 MAX.
    """
    info = analisar_historico_flex_ultra(df)

    st.session_state["df_historico"] = df
    st.session_state["n_passageiros"] = info["n_passageiros"]
    st.session_state["indice_ultima_serie"] = info["indice_ultima_serie"]
    st.session_state["k_col_name"] = "k" if info["tem_k"] else None

    exibir_bloco_mensagem(
        "Histórico FLEX ULTRA carregado",
        texto_em_blocos(
            f"Nº de séries (carros): {info['n_series']}\n"
            f"Nº de passageiros por carro: {info['n_passageiros']}\n"
            f"Coluna k detectada: {'Sim' if info['tem_k'] else 'Não'}\n"
            f"Última série: C{info['indice_ultima_serie'] + 1}"
        ),
        emoji="📥",
    )


def painel_entrada_historico_flex_ultra() -> None:
    """
    Painel oficial de entrada do histórico (FLEX ULTRA):
    - Opção de arquivo
    - Opção de copiar/colar
    - Diagnóstico inicial
    """
    st.markdown("## 📥 Histórico — Entrada FLEX ULTRA (V15.6 MAX)")

    tab_arquivo, tab_texto = st.tabs(["Upload de arquivo", "Copiar/colar texto"])

    with tab_arquivo:
        st.markdown(
            "Envie um arquivo `.csv` ou `.txt` contendo as séries (carros), "
            "onde cada linha representa um carro e os campos são separados "
            "por `;`, `,` ou espaço. A coluna `k` pode estar presente como "
            "último campo."
        )
        arquivo = st.file_uploader(
            "Selecione o arquivo de histórico",
            type=["csv", "txt"],
            key="upload_historico_v156",
        )
        tem_k_arquivo = st.checkbox(
            "Histórico possui coluna k na última posição?",
            value=True,
            key="ck_tem_k_arquivo_v156",
        )

        if st.button("Carregar histórico do arquivo", type="primary"):
            try:
                df = carregar_historico_de_arquivo(
                    arquivo,
                    tem_coluna_k=tem_k_arquivo,
                )
                registrar_historico_na_sessao(df)
            except Exception as e:
                st.error(f"Erro ao carregar histórico do arquivo: {e}")

    with tab_texto:
        st.markdown(
            "Cole abaixo o histórico no formato FLEX ULTRA. "
            "Cada linha = um carro. Último campo opcional = coluna k."
        )
        texto = st.text_area(
            "Cole aqui o histórico",
            height=250,
            key="txt_historico_v156",
        )
        tem_k_texto = st.checkbox(
            "Texto possui coluna k na última posição?",
            value=True,
            key="ck_tem_k_texto_v156",
        )

        if st.button("Carregar histórico do texto", type="primary"):
            try:
                df = carregar_historico_de_texto(
                    texto,
                    tem_coluna_k=tem_k_texto,
                )
                registrar_historico_na_sessao(df)
            except Exception as e:
                st.error(f"Erro ao carregar histórico do texto: {e}")

    # Resumo rápido se já houver histórico carregado
    df_hist = st.session_state.get("df_historico")
    if df_hist is not None:
        info = analisar_historico_flex_ultra(df_hist)
        st.markdown("### Visão rápida do histórico carregado")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(df_hist.tail(10), use_container_width=True)
        with col2:
            st.metric("Nº de séries", info["n_series"])
            st.metric("Passageiros por carro", info["n_passageiros"])
            st.metric(
                "Possui coluna k?",
                "Sim" if info["tem_k"] else "Não",
            )


# ============================================================
# Navegação lateral V15.6 MAX
# ============================================================

def construir_navegacao_v156() -> str:
    """
    Constrói a navegação lateral oficial do V15.6 MAX.
    Retorna o nome do painel selecionado.
    """
    st.sidebar.markdown("## 🚗 Predict Cars V15.6 MAX")

    st.sidebar.markdown(
        """
        <small>
        Fluxo sugerido:<br/>
        1️⃣ Histórico FLEX ULTRA<br/>
        2️⃣ Sentinelas k & k*<br/>
        3️⃣ Pipeline V14-FLEX ULTRA<br/>
        4️⃣ Replays (LIGHT / ULTRA / Unitário)<br/>
        5️⃣ TURBO++ ULTRA Anti-Ruído<br/>
        6️⃣ Ruído Condicional & Confiabilidade<br/>
        7️⃣ Modo 6 Acertos V15.6 MAX<br/>
        8️⃣ Relatório Final MAX
        </small>
        """,
        unsafe_allow_html=True,
    )

    painel = st.sidebar.radio(
        "Navegação",
        [
            "📥 Histórico — Entrada FLEX ULTRA (V15.6 MAX)",
            "🔍 Pipeline V14-FLEX ULTRA (V15.6 MAX)",
            "💡 Replay LIGHT (V15.6 MAX)",
            "📅 Replay ULTRA (V15.6 MAX)",
            "🎯 Replay ULTRA Unitário (V15.6 MAX)",
            "🚨 Monitor de Risco (k & k*) (V15.6 MAX)",
            "🧪 Testes de Confiabilidade REAL (V15.6 MAX)",
            "📊 Ruído Condicional (NR%) (V15.6 MAX)",
            "🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15.6 MAX)",
            "🎯 Modo 6 Acertos — Execução (V15.6 MAX)",
            "📑 Relatório Final V15.6 MAX",
        ],
        index=0,
        key="nav_v156",
    )

    # Configurações avançadas (anti-zumbi, etc.)
    with st.sidebar.expander("⚙️ Configurações avançadas / Anti-Zumbi"):
        limite_max_janela = st.number_input(
            "Limite máximo de séries para operações intensivas",
            min_value=100,
            max_value=10000,
            value=st.session_state.get("limite_max_janela", 600),
            step=50,
        )
        st.session_state["limite_max_janela"] = int(limite_max_janela)

        limite_max_prev = st.number_input(
            "Limite máximo de previsões em blocos de teste",
            min_value=50,
            max_value=2000,
            value=st.session_state.get("limite_max_prev", 300),
            step=50,
        )
        st.session_state["limite_max_prev"] = int(limite_max_prev)

        flag_modo_seguro = st.checkbox(
            "Ativar modo seguro para históricos muito grandes",
            value=st.session_state.get("flag_modo_seguro", True),
        )
        st.session_state["flag_modo_seguro"] = bool(flag_modo_seguro)

        st.caption(
            "Esses limites ajudam a evitar travamentos tipo 'zumbi' em janelas muito grandes."
        )

    return painel

# ------------------------------------------------------------
# FIM DA PARTE 1/6
# ------------------------------------------------------------
# ============================================================
# PARTE 2/6 — Núcleo estrutural V14-FLEX ULTRA + Diagnósticos
# ============================================================

from dataclasses import dataclass
import random  # será usado nas partes profundas (Monte Carlo etc.)

# ------------------------------------------------------------
# Dataclasses — estruturas de apoio (mantendo jeitão V14-FLEX)
# ------------------------------------------------------------

@dataclass
class RegimeInfo:
    janela_usada: int
    k_medio: float
    k_max: float
    estado: str
    descricao: str


@dataclass
class KStarInfo:
    janela_usada: int
    k_media_janela: float
    k_max_janela: float
    k_star_pct: float
    estado: str
    descricao: str


@dataclass
class BacktestResult:
    tabela: pd.DataFrame
    descricao: str


@dataclass
class QDSInfo:
    valor: float
    descricao: str


@dataclass
class DiagnosticoImpacto:
    idx_alvo: int
    regime_estado: str
    k_star_pct: float
    risco_modo6: str
    comentario: str


@dataclass
class PlanoCalibracao:
    foco: str
    passos: List[str]


# ============================================================
# Funções utilitárias básicas (compatíveis com V14-FLEX ULTRA)
# ============================================================

def obter_n_passageiros(df: pd.DataFrame) -> int:
    """
    Detecta automaticamente o número de passageiros (colunas P1..Pn ou p1..pn)
    a partir do dataframe carregado.
    """
    cols = [c for c in df.columns if c.lower().startswith("p")]
    cols_ordenadas = sorted(
        cols,
        key=lambda x: int("".join(filter(str.isdigit, x)) or 0)
    )
    return len(cols_ordenadas)


def extrair_passageiros(df: pd.DataFrame) -> List[str]:
    """
    Retorna a lista de colunas de passageiros (P1..Pn / p1..pn), ordenadas.
    """
    cols = [c for c in df.columns if c.lower().startswith("p")]
    return sorted(
        cols,
        key=lambda x: int("".join(filter(str.isdigit, x)) or 0)
    )


def obter_intervalo_indices(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Usa (ou cria) a coluna 'idx' para determinar o intervalo de índices.
    Compatível com histórico carregado pela PARTE 1/6 (que não tinha idx).
    """
    if "idx" not in df.columns:
        # Cria idx 1..n in-place (afeta o DF da sessão, o que é desejado)
        df["idx"] = np.arange(1, len(df) + 1, dtype=int)
    return int(df["idx"].min()), int(df["idx"].max())


def normalizar_serie(serie: Any) -> List[int]:
    """
    Normaliza uma série de passageiros (lista/array/Series/string) para lista de int.
    Mantém o mesmo jeitão V14-FLEX ULTRA.
    """
    if isinstance(serie, list):
        return [int(x) for x in serie]
    if isinstance(serie, np.ndarray):
        return [int(x) for x in serie.tolist()]
    if isinstance(serie, pd.Series):
        return [int(x) for x in serie.tolist()]
    if isinstance(serie, str):
        partes = [p for p in serie.replace(",", " ").split() if p.strip()]
        return [int(p) for p in partes]

    try:
        return [int(x) for x in list(serie)]
    except Exception:
        return []


def calcular_acerto_total(serie_real: List[int], serie_prevista: List[int]) -> int:
    """
    Calcula acerto total (n acertos exatos = n, comparando posição a posição).
    """
    if len(serie_real) != len(serie_prevista):
        return 0
    return sum(1 for a, b in zip(serie_real, serie_prevista) if int(a) == int(b))


# ============================================================
# REGIME ULTRA — BARÔMETRO DA ESTRADA (V15.6 MAX)
# ============================================================

def calcular_regime_ultra(df: pd.DataFrame, janela: int = 40) -> RegimeInfo:
    """
    Calcula o regime (Barômetro ULTRA) usando uma janela recente de k.

    - k_medio
    - k_max
    - estado (estavel / transicao / ruptura)
    """
    if "k" not in df.columns:
        raise KeyError("Coluna 'k' não encontrada no histórico.")

    janela = min(max(janela, 1), len(df))
    trecho = df.tail(janela)
    k_vals = trecho["k"].astype(float).values

    k_medio = float(np.mean(k_vals)) if len(k_vals) > 0 else 0.0
    k_max = float(np.max(k_vals)) if len(k_vals) > 0 else 0.0

    # Heurística de regime (mesma lógica base V14-FLEX ULTRA)
    if k_medio < 0.05 and k_max <= 1:
        estado = "ruptura"
        desc = (
            "🚨 Estrada em ruptura — muitos carros com k=0, "
            "baixíssima previsibilidade estrutural."
        )
    elif k_medio < 0.25:
        estado = "estavel"
        desc = (
            "🟢 Estrada estável — poucos guardas acertando exatamente o carro."
        )
    else:
        estado = "transicao"
        desc = (
            "🟡 Estrada em transição — guardas começando a acertar em alguns pontos."
        )

    return RegimeInfo(
        janela_usada=janela,
        k_medio=k_medio,
        k_max=k_max,
        estado=estado,
        descricao=desc,
    )


# ============================================================
# k* ULTRA REAL — SENTINELA DOS GUARDAS (V15.6 MAX)
# ============================================================

def calcular_k_star_ultra(df: pd.DataFrame, janela: int = 40) -> KStarInfo:
    """
    Calcula o k* ULTRA (sentinela preditivo) usando distribuição recente de k.

    Interpretação (mesma filosofia V14-FLEX):
    - k* baixo: ambiente estável-fraco (sem padrão forte, sem caos)
    - k* médio: pré-transição / sensibilidade moderada
    - k* alto: turbulência / ruptura de regime / pré-choque
    """
    if "k" not in df.columns:
        raise KeyError("Coluna 'k' não encontrada no histórico.")

    janela = min(max(janela, 1), len(df))
    trecho = df.tail(janela)
    k_vals = trecho["k"].astype(float).values

    if len(k_vals) == 0:
        return KStarInfo(
            janela_usada=0,
            k_media_janela=0.0,
            k_max_janela=0.0,
            k_star_pct=0.0,
            estado="neutro",
            descricao=(
                "⚪ k*: neutro — histórico insuficiente para avaliar sensibilidade."
            ),
        )

    k_media = float(np.mean(k_vals))
    k_max = float(np.max(k_vals))

    # k* simples: proporção de séries com k > 0 na janela (em %)
    proporcao_k_pos = float(np.mean(k_vals > 0))
    k_star_pct = round(proporcao_k_pos * 100.0, 1)

    if k_star_pct < 15:
        estado = "estavel"
        desc = "🟢 k*: Ambiente estável — poucos guardas acertando exatamente."
    elif k_star_pct < 40:
        estado = "atencao"
        desc = "🟡 k*: Pré-transição — sensibilidade crescente dos guardas."
    else:
        estado = "critico"
        desc = "🔴 k*: Ambiente crítico — guardas muito sensíveis, alta chance de rupturas."

    return KStarInfo(
        janela_usada=janela,
        k_media_janela=k_media,
        k_max_janela=k_max,
        k_star_pct=k_star_pct,
        estado=estado,
        descricao=desc,
    )


# ============================================================
# IDX ULTRA — CONTEXTO ESTRUTURAL GLOBAL (V15.6 MAX)
# ============================================================

def construir_contexto_idx_ultra(df: pd.DataFrame, idx_alvo: int) -> Dict[str, Any]:
    """
    Constrói um contexto IDX ULTRA simples:

    - janela usada
    - média de k
    - k max
    - índice global (proxy)

    Mantém apenas como insumo de contexto (não determinístico).
    Compatível com DF sem 'idx' (cria se necessário).
    """
    min_idx, max_idx = obter_intervalo_indices(df)

    idx_alvo = int(idx_alvo)
    if idx_alvo < min_idx:
        idx_alvo = min_idx
    if idx_alvo > max_idx:
        idx_alvo = max_idx

    janela = 40
    corte = df[df["idx"] <= idx_alvo].tail(janela)
    janela_real = len(corte)

    if janela_real == 0:
        return {
            "janela_usada": 0,
            "media_k": 0.0,
            "max_k": 0.0,
            "idx_global": 0.0,
        }

    k_vals = corte["k"].astype(float).values
    media_k = float(np.mean(k_vals))
    max_k = float(np.max(k_vals))

    # IDX ULTRA: proxy simples — quanto menor k, maior dispersão / índice.
    idx_global = (1.0 - (media_k / (1.0 + max_k))) * 50.0

    return {
        "janela_usada": janela_real,
        "media_k": media_k,
        "max_k": max_k,
        "idx_global": idx_global,
    }


# ============================================================
# Série k / k* seriada — apoio para ruído & monitor de risco
# ============================================================

def atualizar_series_k_e_kstar(df: pd.DataFrame, janela: int = 40) -> None:
    """
    Constrói e registra em sessão:

    - série_k: valores de k ao longo da estrada
    - série_k_star: k* em janela deslizante (proporção de k>0 em %)

    Usado por:
    - Painel de ruído condicional (NR%)
    - Monitor de risco (k & k*)
    - Modo 6 acertos V15.6 MAX
    """
    if df is None or df.empty or "k" not in df.columns:
        st.session_state["serie_k"] = None
        st.session_state["serie_k_star"] = None
        return

    s_k = df["k"].astype(float).reset_index(drop=True)
    w = min(max(janela, 1), len(s_k))

    # Rolling em "guardas que acertaram algo" (k>0)
    s_pos = (s_k > 0).astype(float)
    s_k_star = s_pos.rolling(window=w, min_periods=1).mean() * 100.0

    st.session_state["serie_k"] = s_k
    st.session_state["serie_k_star"] = s_k_star


# ============================================================
# Diagnóstico de Impacto & Mini Plano de Calibração
# ============================================================

def gerar_diagnostico_impacto_v156(
    idx_alvo: int,
    regime_info: RegimeInfo,
    k_info: KStarInfo,
    n_series: int,
) -> DiagnosticoImpacto:
    """
    Gera um diagnóstico qualitativo de impacto, considerando:

    - regime estrutural (Barômetro)
    - sensibilidade k* (sentinela)
    - posição na estrada (idx alvo vs total)

    Serve como texto-guia para decisões de:
    - uso de TURBO++ ULTRA
    - agressividade de leque
    - elegibilidade para Modo 6 Acertos
    """
    pos_pct = 100.0 * idx_alvo / max(n_series, 1)

    if regime_info.estado == "ruptura" or k_info.estado == "critico":
        risco_modo6 = "❌ Não recomendado agora"
        comentario = (
            "Estamos em ambiente de ruptura/turbulência. "
            "Priorize leituras estruturais, use leques mais amplos, "
            "reforce coberturas e adie execuções agressivas de 6 acertos "
            "até o regime voltar para estável ou transição suave."
        )
    elif regime_info.estado == "transicao" or k_info.estado == "atencao":
        risco_modo6 = "⚠️ Alta cautela"
        comentario = (
            "Estrada em transição / pré-transição. "
            "É possível usar Modo 6 Acertos de forma cirúrgica, "
            "desde que QDS, ruído condicional (NR%) e Monte Carlo "
            "confirmem consistência do núcleo."
        )
    else:
        risco_modo6 = "✅ Ambiente favorável (sob confirmação)"
        comentario = (
            "Estrada estruturalmente estável e k* baixo. "
            "Cenário favorável para usar TURBO++ ULTRA e considerar 6 acertos, "
            "desde que os testes de confiabilidade REAL e o painel de ruído "
            "mostrem coerência com esse diagnóstico."
        )

    # Pequeno ajuste de mensagem se estamos muito no início ou muito no fim
    if pos_pct < 10:
        comentario += (
            " Estamos ainda no início da estrada desta base — "
            "os diagnósticos tendem a ser mais voláteis."
        )
    elif pos_pct > 90:
        comentario += (
            " Estamos muito próximos da cauda da estrada — "
            "ideal para previsões finais e cenários focados na próxima série."
        )

    diag = DiagnosticoImpacto(
        idx_alvo=int(idx_alvo),
        regime_estado=regime_info.estado,
        k_star_pct=k_info.k_star_pct,
        risco_modo6=risco_modo6,
        comentario=comentario,
    )

    st.session_state["diag_impacto"] = diag
    return diag


def gerar_plano_calibracao_v156(
    regime_info: RegimeInfo,
    k_info: KStarInfo,
) -> PlanoCalibracao:
    """
    Gera um mini plano de calibração em 3–5 passos, em função do regime/k*.

    Ideia: em vez de só dizer "ok/não ok", deixar claro:
    - onde calibrar (S6, MC, Micro)
    - quanto apertar/afrouxar
    """
    passos: List[str] = []

    if regime_info.estado == "ruptura" or k_info.estado == "critico":
        foco = "Sobrevivência estrutural (ruptura / ambiente crítico)"
        passos = [
            "Reduzir o peso do núcleo determinístico (S6 Profundo) e "
            "aumentar peso de Monte Carlo Profundo ULTRA.",
            "Alongar leques de cobertura, aceitando maior dispersão controlada.",
            "Priorizar janelas mais curtas para leitura de regime (estrada recente).",
            "Suspender temporariamente execuções agressivas de Modo 6 Acertos.",
        ]
    elif regime_info.estado == "transicao" or k_info.estado == "atencao":
        foco = "Ajuste fino em transição (pré-ruptura / pré-estabilidade)"
        passos = [
            "Equilibrar pesos entre S6 Profundo e Monte Carlo, "
            "mantendo Micro-Leque como ajuste fino.",
            "Monitorar k* em janelas móveis — se subir demais, "
            "migrar para postura mais defensiva.",
            "Usar Replay LIGHT/ULTRA para validar se o núcleo se mantém resiliente.",
            "Liberar Modo 6 Acertos apenas quando QDS REAL estiver em patamar sólido.",
        ]
    else:
        foco = "Exploração eficiente em estrada estável"
        passos = [
            "Dar peso maior ao núcleo determinístico (S6 Profundo) "
            "e micro-variações (Micro-Leque).",
            "Usar Monte Carlo como apoio de robustez, não como motor principal.",
            "Rodar testes de confiabilidade REAL com mais previsões (n maior) "
            "para consolidar o nível de confiança.",
            "Mapear janelas de ruído condicional (NR%) para localizar trechos premium.",
        ]

    plano = PlanoCalibracao(foco=foco, passos=passos)
    st.session_state["plano_calibracao"] = plano
    return plano


# ============================================================
# Painel — 🔍 Pipeline V14-FLEX ULTRA (V15.6 MAX)
# (Visão estrutural + impacto + plano de calibração)
# ============================================================

def painel_pipeline_v14_flex_ultra_v156() -> None:
    """
    Painel estrutural do V14-FLEX ULTRA dentro do V15.6 MAX.

    Mostra:
    - Barômetro ULTRA (regime)
    - k* ULTRA (sentinela)
    - IDX ULTRA (contexto)
    - Diagnóstico de impacto
    - Mini plano de calibração
    - Proteção Anti-Zumbi integrada
    """
    st.markdown("## 🔍 Pipeline V14-FLEX ULTRA (V15.6 MAX)")

    df = st.session_state.get("df_historico")
    if df is None or df.empty:
        st.warning(
            "Carregue primeiro o histórico em "
            "`📥 Histórico — Entrada FLEX ULTRA (V15.6 MAX)`."
        )
        return

    # Proteção Anti-Zumbi
    limite_max = st.session_state.get("limite_max_janela", 600)
    if st.session_state.get("flag_modo_seguro", True):
        if not limitar_operacao(len(df), limite_max, "Pipeline V14-FLEX ULTRA"):
            return

    # Garante coluna idx compatível com as funções ULTRA
    min_idx, max_idx = obter_intervalo_indices(df)

    col_cfg1, col_cfg2 = st.columns([2, 1])
    with col_cfg1:
        idx_alvo = st.number_input(
            "Selecione o índice alvo (1 = primeira série carregada):",
            min_value=int(min_idx),
            max_value=int(max_idx),
            value=int(max_idx),
            step=1,
            help=(
                "O pipeline V14-FLEX ULTRA considera a estrada de C1 até C(idx_alvo). "
                "É a foto estrutural da estrada até esse ponto."
            ),
            key="idx_alvo_pipeline_v156",
        )
    with col_cfg2:
        janela_regime = st.slider(
            "Janela para Barômetro / k* (séries recentes):",
            min_value=20,
            max_value=min(200, int(len(df))),
            value=40,
            step=5,
            key="janela_regime_kstar_v156",
        )

    # Estrada até o alvo
    df_ate_alvo = df[df["idx"] <= int(idx_alvo)].copy()
    if df_ate_alvo.empty:
        df_ate_alvo = df.copy()

    # Cálculos estruturais
    regime_info = calcular_regime_ultra(df_ate_alvo, janela=int(janela_regime))
    k_info = calcular_k_star_ultra(df_ate_alvo, janela=int(janela_regime))
    contexto_idx = construir_contexto_idx_ultra(df_ate_alvo, idx_alvo=int(idx_alvo))

    # Atualiza séries k / k* para outros painéis
    atualizar_series_k_e_kstar(df_ate_alvo, janela=int(janela_regime))

    # Série alvo (estrutura)
    cols_pass = extrair_passageiros(df)
    df_alvo = df_ate_alvo[df_ate_alvo["idx"] == int(idx_alvo)]
    if not df_alvo.empty:
        serie_alvo = [int(df_alvo.iloc[0][c]) for c in cols_pass]
    else:
        serie_alvo = []

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📊 Barômetro ULTRA REAL")
        st.markdown(regime_info.descricao)
        st.json(
            {
                "janela": regime_info.janela_usada,
                "k_medio": round(regime_info.k_medio, 4),
                "k_max": int(regime_info.k_max),
                "estado": regime_info.estado,
            }
        )

    with col2:
        st.subheader("🛰️ k* ULTRA REAL — Sentinela")
        st.markdown(k_info.descricao)
        st.json(
            {
                "janela": k_info.janela_usada,
                "k_media_janela": round(k_info.k_media_janela, 4),
                "k_max_janela": int(k_info.k_max_janela),
                "k_star_pct": k_info.k_star_pct,
                "estado": k_info.estado,
            }
        )

    with col3:
        st.subheader("🧭 IDX ULTRA — Contexto")
        st.json(
            {
                "janela_usada": contexto_idx["janela_usada"],
                "media_k": round(contexto_idx["media_k"], 4),
                "max_k": int(contexto_idx["max_k"]),
                "indice_global": round(contexto_idx["idx_global"], 2),
            }
        )

    st.markdown("### 🚗 Série alvo (estrutura)")

    if serie_alvo:
        st.code(formatar_lista_passageiros(serie_alvo), language="text")
    else:
        st.info("Nenhuma série alvo encontrada para o índice selecionado.")

    # ---------------------------
    # Diagnóstico & plano (opt)
    # ---------------------------

    st.markdown("### 🔎 Diagnóstico de impacto & mini plano de calibração")

    col_diag, col_plano = st.columns(2)
    with col_diag:
        flag_diag = st.checkbox(
            "✔️ Gerar diagnóstico de impacto para este ponto da estrada",
            value=True,
            key="ck_diag_impacto_v156",
        )
    with col_plano:
        flag_plano = st.checkbox(
            "📋 Gerar mini plano de calibração",
            value=True,
            key="ck_plano_calibracao_v156",
        )

    n_series_total = len(df)

    if flag_diag:
        diag = gerar_diagnostico_impacto_v156(
            idx_alvo=int(idx_alvo),
            regime_info=regime_info,
            k_info=k_info,
            n_series=n_series_total,
        )
        exibir_bloco_mensagem(
            "Diagnóstico de impacto",
            texto_em_blocos(
                f"Risco para Modo 6 Acertos: {diag.risco_modo6}\n\n"
                f"Comentário: {diag.comentario}"
            ),
            emoji="🧠",
        )

    if flag_plano:
        plano = gerar_plano_calibracao_v156(regime_info, k_info)
        texto_passos = "\n".join(f"- {p}" for p in plano.passos)
        exibir_bloco_mensagem(
            f"Mini plano de calibração — {plano.foco}",
            texto_em_blocos(texto_passos),
            emoji="🛠️",
        )

    st.caption(
        "Este painel é a base estrutural do V15.6 MAX. "
        "Os demais módulos (Replays, TURBO++ ULTRA, Ruído, Confiabilidade e Modo 6 Acertos) "
        "consumirão esses diagnósticos e contextos."
    )

# ------------------------------------------------------------
# FIM DA PARTE 2/6
# ------------------------------------------------------------
# ============================================================
# PARTE 3/6 — Replay LIGHT / ULTRA / Unitário (V15.6 MAX)
# ============================================================

# ------------------------------------------------------------
# Funções de apoio para Replays (compactação, janelas, etc.)
# ------------------------------------------------------------

def obter_janela_para_replay(df: pd.DataFrame, idx_alvo: int, largura: int) -> pd.DataFrame:
    """
    Retorna uma janela DF[(idx >= idx_alvo-largura) & (idx <= idx_alvo-1)].
    Mantém 100% compatibilidade com FLEX ULTRA.
    """
    if "idx" not in df.columns:
        df["idx"] = np.arange(1, len(df) + 1, dtype=int)

    inicio = max(int(idx_alvo) - largura, 1)
    fim = int(idx_alvo) - 1
    janela = df[(df["idx"] >= inicio) & (df["idx"] <= fim)].copy()

    return janela


def extrair_series_para_replay(df: pd.DataFrame, idx_alvo: int, n_pass: int) -> Tuple[List[int], List[int]]:
    """
    Retorna (série_anterior, série_alvo) em formato LISTA com n_pass elementos.
    """
    cols_pass = sorted(
        [c for c in df.columns if c.lower().startswith("p")],
        key=lambda x: int("".join(filter(str.isdigit, x)) or 0),
    )

    df_prev = df[df["idx"] == int(idx_alvo) - 1]
    df_alvo = df[df["idx"] == int(idx_alvo)]

    if df_prev.empty or df_alvo.empty:
        return [], []

    serie_prev = [int(df_prev.iloc[0][c]) for c in cols_pass]
    serie_alvo = [int(df_alvo.iloc[0][c]) for c in cols_pass]

    return serie_prev, serie_alvo


def similaridade_basica(a: List[int], b: List[int]) -> float:
    """
    Similaridade simples: proporção de passageiros iguais.
    """
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(sum(1 for x, y in zip(a, b) if x == y)) / len(a)


def ponderar_listas(l1: List[int], l2: List[int], peso1: float = 0.5, peso2: float = 0.5) -> List[int]:
    """
    Combinação simples: média ponderada (com arredondamento).
    """
    if len(l1) != len(l2):
        return l1
    out = []
    for a, b in zip(l1, l2):
        val = int(round(a * peso1 + b * peso2))
        out.append(val)
    return out


# ------------------------------------------------------------
# Replay LIGHT (V15.6 MAX)
# ------------------------------------------------------------

def replay_light_v156(df: pd.DataFrame, idx_alvo: int, largura: int = 30) -> Dict[str, Any]:
    """
    Replay LIGHT:
    - Baseado em janelas curtas (robustez local)
    - Similaridade simples com séries próximas
    - Resultados diretos e ágeis
    """
    if "idx" not in df.columns:
        df["idx"] = np.arange(1, len(df) + 1)

    if idx_alvo <= 1:
        return {
            "serie_prevista": [],
            "serie_real": [],
            "descricao": "Índice alvo inválido (não existe série anterior).",
        }

    n_pass = obter_n_passageiros(df)

    janela_df = obter_janela_para_replay(df, idx_alvo, largura)
    serie_prev, serie_real = extrair_series_para_replay(df, idx_alvo, n_pass)

    if len(serie_prev) == 0 or len(serie_real) == 0:
        return {
            "serie_prevista": [],
            "serie_real": [],
            "descricao": "Não foi possível determinar séries anterior/atual."
        }

    melhores = []
    for _, row in janela_df.iterrows():
        linha = [int(row[c]) for c in sorted(janela_df.columns, key=lambda x: int("".join(filter(str.isdigit, x)) or 0)) if c.lower().startswith("p")]
        sim = similaridade_basica(serie_prev, linha)
        melhores.append((sim, linha))

    if not melhores:
        prev = serie_prev.copy()
    else:
        melhores.sort(reverse=True, key=lambda x: x[0])
        # média das 3 melhores
        top3 = melhores[:3]
        if len(top3) == 1:
            prev = top3[0][1]
        else:
            m1 = np.mean([p for _, p in top3], axis=0)
            prev = [int(round(x)) for x in m1]

    return {
        "serie_prevista": prev,
        "serie_real": serie_real,
        "descricao": f"Replay LIGHT com janela {largura}",
    }


# ------------------------------------------------------------
# Replay ULTRA (V15.6 MAX)
# ------------------------------------------------------------

def replay_ultra_v156(df: pd.DataFrame, idx_alvo: int, largura: int = 60) -> Dict[str, Any]:
    """
    Replay ULTRA:
    - Baseado em janelas maiores
    - Similaridade avançada (S6-like leve)
    - Combinação multi-pontos
    """
    if "idx" not in df.columns:
        df["idx"] = np.arange(1, len(df) + 1)

    if idx_alvo <= 1:
        return {
            "serie_prevista": [],
            "serie_real": [],
            "descricao": "Índice alvo inválido."
        }

    n_pass = obter_n_passageiros(df)
    janela_df = obter_janela_para_replay(df, idx_alvo, largura)
    serie_prev, serie_real = extrair_series_para_replay(df, idx_alvo, n_pass)

    if len(serie_prev) == 0:
        return {
            "serie_prevista": [],
            "serie_real": [],
            "descricao": "Não foi possível extrair série anterior real."
        }

    # Similaridade S6-like (simples)
    def sim_s6(a, b):
        if len(a) != len(b):
            return 0.0
        acertos = sum(1 for x, y in zip(a, b) if x == y)
        return acertos / len(a)

    melhores = []
    for _, row in janela_df.iterrows():
        linha = [int(row[c]) for c in sorted(janela_df.columns, key=lambda x: int("".join(filter(str.isdigit, x)) or 0)) if c.lower().startswith("p")]
        sim = sim_s6(serie_prev, linha)
        melhores.append((sim, linha))

    if not melhores:
        prev = serie_prev.copy()
    else:
        melhores.sort(reverse=True, key=lambda x: x[0])
        top5 = melhores[:5]

        # Combinação ponderada (mais forte que o LIGHT)
        pesos = np.linspace(1.0, 2.0, len(top5))
        pesos = pesos / pesos.sum()
        m = np.zeros(n_pass)
        for p, (sim, linha) in zip(pesos, top5):
            m += p * np.array(linha)
        prev = [int(round(x)) for x in m]

    return {
        "serie_prevista": prev,
        "serie_real": serie_real,
        "descricao": f"Replay ULTRA com janela {largura}",
    }


# ------------------------------------------------------------
# Replay ULTRA Unitário (V15.6 MAX)
# ------------------------------------------------------------

def replay_unitario_v156(df: pd.DataFrame, idx_alvo: int) -> Dict[str, Any]:
    """
    Replay ULTRA unitário:
    - Aplica o ULTRA somente no índice alvo exato.
    - Não usa médias entre janelas.
    """
    if "idx" not in df.columns:
        df["idx"] = np.arange(1, len(df) + 1)

    if idx_alvo <= 1:
        return {
            "serie_prevista": [],
            "serie_real": [],
            "descricao": "Índice alvo inválido."
        }

    n_pass = obter_n_passageiros(df)
    serie_prev, serie_real = extrair_series_para_replay(df, idx_alvo, n_pass)

    if len(serie_prev) == 0:
        return {
            "serie_prevista": [],
            "serie_real": [],
            "descricao": "Não foi possível extrair série anterior."
        }

    # Estratégia unitária:
    # deslocamento leve + ruído estrutural mínimo
    deslocado = serie_prev[1:] + serie_prev[:1]
    prev = []
    for a, b in zip(serie_prev, deslocado):
        prev.append(int(round((a * 0.6) + (b * 0.4))))

    return {
        "serie_prevista": prev,
        "serie_real": serie_real,
        "descricao": "Replay ULTRA Unitário",
    }


# ============================================================
# Painel — 💡 Replay LIGHT (V15.6 MAX)
# ============================================================

def painel_replay_light_v156() -> None:
    st.markdown("## 💡 Replay LIGHT (V15.6 MAX)")

    df = st.session_state.get("df_historico")
    if df is None or df.empty:
        st.warning("Carregue primeiro o histórico.")
        return

    min_idx, max_idx = obter_intervalo_indices(df)

    idx_alvo = st.number_input(
        "Índice alvo:",
        min_value=min_idx + 1,
        max_value=max_idx,
        value=max_idx,
        step=1,
        key="idx_replay_light_v156",
    )

    largura = st.slider(
        "Largura da janela (Replay LIGHT):",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
        key="largura_replay_light_v156",
    )

    if st.button("Rodar Replay LIGHT", type="primary"):
        limite_max = st.session_state.get("limite_max_janela", 600)
        if st.session_state.get("flag_modo_seguro", True):
            if not limitar_operacao(len(df), limite_max, "Replay LIGHT"):
                return

        r = replay_light_v156(df, idx_alvo=idx_alvo, largura=largura)
        st.session_state["replay_light_result"] = r

    r = st.session_state.get("replay_light_result")
    if r:
        st.markdown("### Resultado Replay LIGHT")
        col1, col2 = st.columns(2)
        with col1:
            st.code(formatar_lista_passageiros(r["serie_prevista"]), language="text")
        with col2:
            st.code(formatar_lista_passageiros(r["serie_real"]), language="text")
        st.caption(r["descricao"])


# ============================================================
# Painel — 📅 Replay ULTRA (V15.6 MAX)
# ============================================================

def painel_replay_ultra_v156() -> None:
    st.markdown("## 📅 Replay ULTRA (V15.6 MAX)")

    df = st.session_state.get("df_historico")
    if df is None or df.empty:
        st.warning("Carregue primeiro o histórico.")
        return

    min_idx, max_idx = obter_intervalo_indices(df)

    idx_alvo = st.number_input(
        "Índice alvo (ULTRA):",
        min_value=min_idx + 1,
        max_value=max_idx,
        value=max_idx,
        step=1,
        key="idx_replay_ultra_v156",
    )

    largura = st.slider(
        "Largura da janela (Replay ULTRA):",
        min_value=30,
        max_value=200,
        value=60,
        step=5,
        key="largura_replay_ultra_v156",
    )

    if st.button("Rodar Replay ULTRA", type="primary"):
        limite_max = st.session_state.get("limite_max_janela", 600)
        if st.session_state.get("flag_modo_seguro", True):
            if not limitar_operacao(len(df), limite_max, "Replay ULTRA"):
                return

        r = replay_ultra_v156(df, idx_alvo=idx_alvo, largura=largura)
        st.session_state["replay_ultra_result"] = r

    r = st.session_state.get("replay_ultra_result")
    if r:
        st.markdown("### Resultado Replay ULTRA")
        col1, col2 = st.columns(2)
        with col1:
            st.code(formatar_lista_passageiros(r["serie_prevista"]), language="text")
        with col2:
            st.code(formatar_lista_passageiros(r["serie_real"]), language="text")
        st.caption(r["descricao"])


# ============================================================
# Painel — 🎯 Replay ULTRA Unitário (V15.6 MAX)
# ============================================================

def painel_replay_unitario_v156() -> None:
    st.markdown("## 🎯 Replay ULTRA Unitário (V15.6 MAX)")

    df = st.session_state.get("df_historico")
    if df is None or df.empty:
        st.warning("Carregue o histórico antes de rodar o replay.")
        return

    min_idx, max_idx = obter_intervalo_indices(df)

    idx_alvo = st.number_input(
        "Índice alvo (unitário):",
        min_value=min_idx + 1,
        max_value=max_idx,
        value=max_idx,
        step=1,
        key="idx_replay_unitario_v156",
    )

    if st.button("Rodar Replay ULTRA Unitário", type="primary"):
        limite_max = st.session_state.get("limite_max_janela", 600)
        if st.session_state.get("flag_modo_seguro", True):
            if not limitar_operacao(len(df), limite_max, "Replay ULTRA Unitário"):
                return

        r = replay_unitario_v156(df, idx_alvo=idx_alvo)
        st.session_state["replay_unitario_result"] = r

    r = st.session_state.get("replay_unitario_result")
    if r:
        st.markdown("### Resultado Replay ULTRA Unitário")
        col1, col2 = st.columns(2)
        with col1:
            st.code(formatar_lista_passageiros(r["serie_prevista"]), language="text")
        with col2:
            st.code(formatar_lista_passageiros(r["serie_real"]), language="text")
        st.caption(r["descricao"])


# ------------------------------------------------------------
# FIM DA PARTE 3/6
# ------------------------------------------------------------
# ============================================================
# PARTE 4/6 — TURBO++ ULTRA ANTI-RUÍDO (V15.6 MAX)
# Núcleo + Micro-Leque + Monte Carlo + Pesos dinâmicos por k*
# ============================================================

# ------------------------------------------------------------
# S6 PROFUNDO — NÚCLEO (V15.6 MAX)
# ------------------------------------------------------------

def s6_profundo_v156(df: pd.DataFrame, idx_alvo: int) -> List[int]:
    """
    Núcleo determinístico (S6 Profundo):
    - Compara série anterior com padrões de alta similaridade
    - Usa as 6 melhores séries da estrada (por S6-like)
    - Média ponderada forte
    """
    if idx_alvo <= 1:
        return []

    n_pass = obter_n_passageiros(df)
    cols_pass = extrair_passageiros(df)

    serie_prev = df[df["idx"] == idx_alvo - 1]
    if serie_prev.empty:
        return []

    serie_prev = [int(serie_prev.iloc[0][c]) for c in cols_pass]

    # Similaridade S6-like (acertos exatos)
    def sim_s6(a, b):
        if len(a) != len(b):
            return 0.0
        return sum(1 for x, y in zip(a, b) if x == y)

    melhores = []
    for _, row in df.iterrows():
        linha = [int(row[c]) for c in cols_pass]
        sim = sim_s6(serie_prev, linha)
        melhores.append((sim, linha))

    melhores.sort(reverse=True, key=lambda x: x[0])
    top6 = melhores[:6]

    if len(top6) == 0:
        return serie_prev.copy()

    # Pesos fortes: 1.0, 1.2, 1.3, 1.4, 1.6, 1.8
    pesos = np.array([1.0, 1.2, 1.3, 1.4, 1.6, 1.8][:len(top6)], dtype=float)
    pesos = pesos / pesos.sum()

    acumulado = np.zeros(n_pass, dtype=float)
    for peso, (sim, linha) in zip(pesos, top6):
        acumulado += peso * np.array(linha)

    return [int(round(x)) for x in acumulado]


# ------------------------------------------------------------
# MICRO-LEQUE PROFUNDO (V15.6 MAX)
# ------------------------------------------------------------

def micro_leque_profundo_v156(serie_base: List[int], intensidade: float = 0.25) -> List[List[int]]:
    """
    Gera variações suaves em torno da série base.
    - intensidade controla deslocamento
    """
    leques = []
    for desloc in [-2, -1, 1, 2]:
        var = [max(0, int(round(x + desloc * intensidade))) for x in serie_base]
        leques.append(var)
    return leques


# ------------------------------------------------------------
# MONTE CARLO PROFUNDO ULTRA (V15.6 MAX)
# ------------------------------------------------------------

def monte_carlo_profundo_v156(
    serie_base: List[int],
    n_amostras: int = 250,
    ruido_base: float = 0.65,
) -> List[List[int]]:
    """
    Monte Carlo Profundo:
    - gera amostras ruidosas em torno da base
    - intensidade guiada por ruido_base
    """
    out = []
    for _ in range(n_amostras):
        am = []
        for x in serie_base:
            pert = np.random.normal(loc=x, scale=ruido_base)
            am.append(max(0, int(round(pert))))
        out.append(am)
    return out


# ------------------------------------------------------------
# DIVERGÊNCIA ENTRE S6 E MONTE CARLO
# ------------------------------------------------------------

def divergencia_s6_mc(serie_s6: List[int], mc_list: List[List[int]]) -> float:
    """
    Mede o quanto o Monte Carlo se afasta do núcleo S6.
    Retorna um valor entre 0 e ~1.
    """
    if not serie_s6 or not mc_list:
        return 0.0

    n = len(serie_s6)
    divergencias = []
    for amostra in mc_list:
        dif = sum(abs(a - b) for a, b in zip(serie_s6, amostra)) / (n * 10)
        divergencias.append(dif)

    return float(np.mean(divergencias))


# ------------------------------------------------------------
# PESOS DINÂMICOS CONTROLADOS POR k*
# ------------------------------------------------------------

def ajustar_pesos_por_kstar(kstar_info: KStarInfo) -> Dict[str, float]:
    """
    Ajusta pesos de S6 / MC / Micro-Leque conforme k*:
    - k* baixo → S6 domina
    - k* médio → equilíbrio
    - k* alto → MC domina (ambiente turbulento)
    """
    pct = kstar_info.k_star_pct

    if pct < 15:  # muito estável
        peso_s6 = 0.60
        peso_mc = 0.25
        peso_micro = 0.15
    elif pct < 35:  # transição suave
        peso_s6 = 0.45
        peso_mc = 0.35
        peso_micro = 0.20
    else:  # turbulência / ruptura
        peso_s6 = 0.30
        peso_mc = 0.55
        peso_micro = 0.15

    return {
        "peso_s6": peso_s6,
        "peso_mc": peso_mc,
        "peso_micro": peso_micro,
    }


# ------------------------------------------------------------
# FUNÇÃO PRINCIPAL — TURBO++ ULTRA (V15.6 MAX)
# ------------------------------------------------------------

def turbo_ultra_v156(
    df: pd.DataFrame,
    idx_alvo: int,
    peso_s6: float = 0.5,
    peso_mc: float = 0.35,
    peso_micro: float = 0.15,
    suavizacao_idx: float = 0.25,
    profundidade_micro: int = 15,
    fator_antirruido: float = 0.40,
    elasticidade_nucleo: float = 0.20,
    intensidade_turbulencia: float = 0.30,
) -> Dict[str, Any]:
 
    """
    Combina:
    - S6 PROFUNDO
    - Micro-Leque Profundo
    - Monte Carlo Profundo
    - Pesos guiados por k*
    - Interseção estatística (passo final)
    - Divergência S6 vs MC
    """

    # Validações básicas
    if "idx" not in df.columns:
        df["idx"] = np.arange(1, len(df) + 1)

    if idx_alvo <= 1:
        return {"erro": "Índice alvo inválido."}

    # ----------------------------------------
    # 1. Série base S6
    # ----------------------------------------
    serie_s6 = s6_profundo_v156(df, idx_alvo)
    if not serie_s6:
        return {"erro": "Não foi possível gerar S6 profundo."}

    # ----------------------------------------
    # 2. Micro-Leque
    # ----------------------------------------
    leques = micro_leque_profundo_v156(serie_s6)
    leques_flat = leques.copy()

    # ----------------------------------------
    # 3. Monte Carlo Profundo
    # ----------------------------------------
    mc_list = monte_carlo_profundo_v156(serie_s6, n_amostras=200)

    # Divergência estrutural
    div_s6_mc = divergencia_s6_mc(serie_s6, mc_list)

    # ----------------------------------------
    # 4. Pesos (guiados pelo k*)
    # ----------------------------------------
    kstar_series = st.session_state.get("serie_k_star")
    if kstar_series is not None and len(kstar_series) >= idx_alvo:
        k_star_local = float(kstar_series.iloc[idx_alvo - 1])
        if k_star_local < 15:
            k_info_local = KStarInfo(40, 0, 0, k_star_local, "estavel", "")
        elif k_star_local < 35:
            k_info_local = KStarInfo(40, 0, 0, k_star_local, "atencao", "")
        else:
            k_info_local = KStarInfo(40, 0, 0, k_star_local, "critico", "")
    else:
        # fallback total
        k_info_local = KStarInfo(40, 0, 0, 20, "atencao", "")

    pesos = ajustar_pesos_por_kstar(k_info_local)

    # ----------------------------------------
    # 5. Interseção estatística (combinação final)
    # ----------------------------------------
    n_pass = len(serie_s6)
    combinacoes = []

    # combinamos núcleo + primeiras amostras de MC + leques
    for p in [serie_s6] + mc_list[:20] + leques_flat:
        combinacoes.append(p)

    # Média ponderada geral (mantendo pesos originais)
    arr = np.zeros(n_pass)
    for c in combinacoes:
        tmp = np.array(c)
        arr += (
            pesos["peso_s6"] * tmp
            + pesos["peso_micro"] * tmp
            + pesos["peso_mc"] * tmp
        )

    arr = arr / len(combinacoes)
    final = [int(round(x)) for x in arr]

    # ------------------------------------------------------------
    # 🔹 REGISTRO EM SESSÃO — COMPATIBILIDADE V14/V15
    # ------------------------------------------------------------
    resultado_sessao = {
        "serie_s6": serie_s6,
        "micro_leque": leques_flat,
        "mc_amostras": mc_list[:20],
        "divergencia_s6_mc": div_s6_mc,
        "pesos": pesos,
        "final": final,
        "descricao": "Previsão TURBO++ ULTRA V15.6 MAX",
    }

    st.session_state["turbo_ultra_result"] = resultado_sessao

    st.session_state["turbo_ultra_logs"] = [
        f"S6: {serie_s6}",
        f"Pesos: {pesos}",
        f"Divergência S6-MC: {div_s6_mc}",
        f"Final: {final}",
    ]

    # ------------------------------------------------------------
    # 🔹 SEÇÃO FINAL — COMPILAÇÃO DE RESULTADOS (V15.6 MAX)
    # ------------------------------------------------------------
    loc = locals()  # garante que não haja NameError se variável não existir

    resultados = {
        # =====================================================================
        # 🎯 1) Núcleo S6 Profundo
        # =====================================================================
        "s6_nucleo": {
            "numeros": loc.get("s6_previsao", serie_s6),
            "dispersao": loc.get("s6_dispersao"),
            "entropia": loc.get("s6_entropia"),
            "faixa": loc.get("s6_faixa"),
            "probabilidades": loc.get("s6_probabilidades"),
        },

        # =====================================================================
        # 🌪️ 2) Micro-Leque Profundo
        # =====================================================================
        "micro_leque": {
            "numeros": loc.get(
                "micro_previsao",
                leques_flat[0] if leques_flat else serie_s6
            ),
            "dispersao": loc.get("micro_dispersao"),
            "entropia": loc.get("micro_entropia"),
            "faixa": loc.get("micro_faixa"),
            "profundidade": profundidade_micro,
            "probabilidades": loc.get("micro_probabilidades"),
        },

        # =====================================================================
        # 🎲 3) Monte Carlo Profundo (MC ULTRA)
        # =====================================================================
        "monte_carlo": {
            "numeros": loc.get(
                "mc_previsao",
                mc_list[0] if mc_list else final
            ),
            "convergencia": loc.get("mc_convergencia"),
            "entropia": loc.get("mc_entropia"),
            "faixa": loc.get("mc_faixa"),
            "iteracoes": loc.get("mc_iteracoes"),
            "probabilidades": loc.get("mc_probabilidades"),
        },

        # =====================================================================
        # 🧭 4) k* — Ambiente Dinâmico (Sentinela)
        # =====================================================================
        "k_star": {
            "valor": loc.get("k_star_valor"),
            "regime": loc.get("k_star_regime"),
            "forca": loc.get("k_star_forca"),
            "ajuste_pesos": {
                "peso_s6_final": loc.get("peso_s6_final", pesos.get("peso_s6")),
                "peso_mc_final": loc.get("peso_mc_final", pesos.get("peso_mc")),
                "peso_micro_final": loc.get("peso_micro_final", pesos.get("peso_micro")),
            },
        },

        # =====================================================================
        # 🔧 5) Índices de Estabilidade (IDX ULTRA)
        # =====================================================================
        "idx_ultra": {
            "t_norm": loc.get("idx_tnorm"),
            "estabilidade": loc.get("idx_estabilidade"),
            "classe": loc.get("idx_classe"),
            "descricao": loc.get("idx_descricao"),
        },

        # =====================================================================
        # ❗ 6) Divergência S6 vs MC (V15.6 MAX)
        # =====================================================================
        "divergencia": {
            "valor_absoluto": loc.get("divergencia_valor", div_s6_mc),
            "porcentagem": loc.get("divergencia_pct"),
            "classe": loc.get("divergencia_classe"),
            "descricao": loc.get("divergencia_texto"),
        },

        # =====================================================================
        # 🧩 7) Interseção Estatística (Núcleo + Coberturas)
        # =====================================================================
        "intersecao": {
            "numeros": loc.get("intersecao_numeros", final),
            "forca_intersecao": loc.get("intersecao_forca"),
            "classe": loc.get("intersecao_classe"),
            "descricao": loc.get("intersecao_texto"),
        },

        # =====================================================================
        # 🚀 8) Previsão Final TURBO++ ULTRA (V15.6 MAX)
        # =====================================================================
        "previsao_final": {
            "numeros": loc.get("previsao_final", final),
            "faixa_combinada": loc.get("faixa_combinada"),
            "entropia_combinada": loc.get("entropia_combinada"),
            "dispersao_combinada": loc.get("dispersao_combinada"),
            "classificacao": loc.get("classificacao_final"),
            "justificativas": {
                "uso_s6": loc.get("justificativa_s6"),
                "uso_mc": loc.get("justificativa_mc"),
                "uso_micro": loc.get("justificativa_micro"),
                "uso_intersecao": loc.get("justificativa_intersecao"),
            },
        },

        # =====================================================================
        # 🛡️ 9) Anti-Ruído — Mapa Condicional (V15.6 MAX)
        # =====================================================================
        "anti_ruido": {
            "fator": fator_antirruido,
            "elasticidade_nucleo": elasticidade_nucleo,
            "intensidade_turbulencia": intensidade_turbulencia,
            "nr_pct": loc.get("nr_pct"),
            "classe_ruido": loc.get("classe_ruido"),
            "descricao_ruido": loc.get("descricao_ruido"),
        },

        # =====================================================================
        # 📊 10) Painel Analítico Completo (DEBUG PROFUNDO)
        # =====================================================================
        "painel_debug": {
            "idx_alvo": idx_alvo,
            "s6_raw": loc.get("s6_raw", serie_s6),
            "mc_raw": loc.get("mc_raw", mc_list[:20]),
            "micro_raw": loc.get("micro_raw", leques_flat),
            "k_series": loc.get("k_series_local"),
            "k_star_series": loc.get("k_star_series"),
            "faixa_s6": loc.get("s6_faixa"),
            "faixa_mc": loc.get("mc_faixa"),
            "faixa_micro": loc.get("micro_faixa"),
            "detalhes_intersecao": loc.get("detalhes_intersecao"),
        },
    }

    # ------------------------------------------------------------
    # 🔹 COMPATIBILIDADE V14/V15 — CAMPOS SIMPLES PARA O PAINEL
    # ------------------------------------------------------------
    resultados["final"] = final
    resultados["serie_s6"] = serie_s6
    resultados["mc_amostras"] = mc_list[:20]
    resultados["divergencia_s6_mc"] = div_s6_mc
    resultados["pesos"] = pesos
    resultados["descricao"] = "Previsão TURBO++ ULTRA V15.6 MAX"

    return resultados




# ============================================================
# PAINEL — 🚀 TURBO++ ULTRA ANTI-RUÍDO (V15.6 MAX)
# ============================================================

def painel_turbo_ultra_v156() -> None:
    st.markdown("## 🚀 Modo TURBO++ ULTRA Anti-Ruído (V15.6 MAX)")

    df = st.session_state.get("df_historico")
    if df is None or df.empty:
        st.warning("Carregue primeiro o histórico.")
        return

    # Intervalo de índices possível
    min_idx, max_idx = obter_intervalo_indices(df)

    # Índice alvo
    idx_alvo = st.number_input(
        "Índice alvo (previsão TURBO++ ULTRA):",
        min_value=min_idx + 1,
        max_value=max_idx,
        value=max_idx,
        step=1,
        key="idx_turbo_ultra_v156",
    )

    # ------------------------------------------------------------
    # 🔧 Ajustes Manuais Básicos (igual ao que você já tinha)
    # ------------------------------------------------------------
    with st.expander("⚙️ Ajustes Manuais — Pesos do Motor"):
        peso_s6 = st.slider(
            "Peso S6",
            min_value=0.05,
            max_value=0.90,
            value=st.session_state.get("turbo_ultra_pesos", {}).get("peso_s6", 0.50),
            step=0.05,
        )
        peso_mc = st.slider(
            "Peso Monte Carlo",
            min_value=0.05,
            max_value=0.90,
            value=st.session_state.get("turbo_ultra_pesos", {}).get("peso_mc", 0.35),
            step=0.05,
        )
        peso_micro = st.slider(
            "Peso Micro-Leque",
            min_value=0.05,
            max_value=0.90,
            value=st.session_state.get("turbo_ultra_pesos", {}).get("peso_micro", 0.15),
            step=0.05,
        )

        st.session_state["turbo_ultra_pesos"] = {
            "peso_s6": float(peso_s6),
            "peso_mc": float(peso_mc),
            "peso_micro": float(peso_micro),
        }
    # ------------------------------------------------------------
    # 🔬 Ajustes Avançados do Motor (NOVO — V15.6 MAX completo)
    # ------------------------------------------------------------
    with st.expander("🔬 Ajustes Avançados do Motor — TURBO++ ULTRA"):

        suavizacao_idx = st.slider(
            "Suavização IDX (0 = bruto, 1 = ultra suave)",
            min_value=0.00,
            max_value=1.00,
            value=st.session_state.get("turbo_ultra_avancado", {}).get("suavizacao_idx", 0.25),
            step=0.05,
        )

        profundidade_micro = st.slider(
            "Profundidade Micro-Leque",
            min_value=3,
            max_value=40,
            value=st.session_state.get("turbo_ultra_avancado", {}).get("profundidade_micro", 15),
            step=1,
        )

        fator_antirruido = st.slider(
            "Fator Anti-Ruído (impacta S6/MC/Micro)",
            min_value=0.00,
            max_value=1.00,
            value=st.session_state.get("turbo_ultra_avancado", {}).get("fator_antirruido", 0.40),
            step=0.05,
        )

        elasticidade_nucleo = st.slider(
            "Elasticidade do Núcleo (0 = rígido, 1 = elástico)",
            min_value=0.00,
            max_value=1.00,
            value=st.session_state.get("turbo_ultra_avancado", {}).get("elasticidade_nucleo", 0.20),
            step=0.05,
        )

        intensidade_turbulencia = st.slider(
            "Intensidade Anti-Turbulência (reduz variações espúrias)",
            min_value=0.00,
            max_value=1.00,
            value=st.session_state.get("turbo_ultra_avancado", {}).get("intensidade_turbulencia", 0.30),
            step=0.05,
        )

        # Salvando tudo
        st.session_state["turbo_ultra_avancado"] = {
            "suavizacao_idx": float(suavizacao_idx),
            "profundidade_micro": int(profundidade_micro),
            "fator_antirruido": float(fator_antirruido),
            "elasticidade_nucleo": float(elasticidade_nucleo),
            "intensidade_turbulencia": float(intensidade_turbulencia),
        }

    # ------------------------------------------------------------
    # ▶️ Execução do TURBO++ ULTRA (com proteção Anti-Zumbi)
    # ------------------------------------------------------------
    if st.button("Rodar TURBO++ ULTRA", type="primary"):
        limite_max = st.session_state.get("limite_max_janela", 600)

        # Proteção contra travamento
        if st.session_state.get("flag_modo_seguro", True):
            if not limitar_operacao(len(df), limite_max, "TURBO++ ULTRA"):
                return

        # Recupera pesos básicos
        pesos = st.session_state.get("turbo_ultra_pesos", {})
        peso_s6 = float(pesos.get("peso_s6", 0.5))
        peso_mc = float(pesos.get("peso_mc", 0.35))
        peso_micro = float(pesos.get("peso_micro", 0.15))

        # Recupera ajustes avançados
        av = st.session_state.get("turbo_ultra_avancado", {})
        suavizacao_idx = float(av.get("suavizacao_idx", 0.25))
        profundidade_micro = int(av.get("profundidade_micro", 15))
        fator_antirruido = float(av.get("fator_antirruido", 0.40))
        elasticidade_nucleo = float(av.get("elasticidade_nucleo", 0.20))
        intensidade_turbulencia = float(av.get("intensidade_turbulencia", 0.30))

        # --------------------------------------------------------
        # 🔥 Chamada do motor TURBO++ ULTRA completo
        # --------------------------------------------------------
        r = turbo_ultra_v156(
            df,
            idx_alvo=idx_alvo,
            peso_s6=float(peso_s6),
            peso_mc=float(peso_mc),
            peso_micro=float(peso_micro),
            suavizacao_idx=float(suavizacao_idx),
            profundidade_micro=int(profundidade_micro),
            fator_antirruido=float(fator_antirruido),
            elasticidade_nucleo=float(elasticidade_nucleo),
            intensidade_turbulencia=float(intensidade_turbulencia),
        )

        st.session_state["turbo_ultra_result"] = r
    # ------------------------------------------------------------
    # 📊 Exibição dos resultados do TURBO++ ULTRA
    # ------------------------------------------------------------
    r = st.session_state.get("turbo_ultra_result")
    if r:
        st.markdown("### 🔥 PREVISÃO FINAL TURBO++ ULTRA")
        st.code(formatar_lista_passageiros(r["final"]), language="text")

        col1, col2, col3 = st.columns(3)

        # Núcleo S6
        with col1:
            st.subheader("Núcleo (S6)")
            st.code(formatar_lista_passageiros(r["serie_s6"]), language="text")

        # Micro-Leque
        with col2:
            st.subheader("Micro-Leque")
            for ml in r["micro_leque"]:
                st.text(formatar_lista_passageiros(ml))

        # Monte Carlo
        with col3:
            st.subheader("Monte Carlo (amostras)")
            for mc in r["mc_amostras"]:
                st.text(formatar_lista_passageiros(mc))

        # Divergência
        st.markdown("### 📉 Divergência S6 vs Monte Carlo")
        st.metric("Divergência média", round(r["divergencia_s6_mc"], 4))

        # Pesos usados (básicos + avançados)
        st.markdown("### ⚖️ Pesos e Parâmetros Utilizados")
        st.json({
            "peso_s6": r["pesos"].get("peso_s6"),
            "peso_mc": r["pesos"].get("peso_mc"),
            "peso_micro": r["pesos"].get("peso_micro"),
            "suavizacao_idx": r.get("suavizacao_idx"),
            "profundidade_micro": r.get("profundidade_micro"),
            "fator_antirruido": r.get("fator_antirruido"),
            "elasticidade_nucleo": r.get("elasticidade_nucleo"),
            "intensidade_turbulencia": r.get("intensidade_turbulencia"),
        })

        # Mensagem legendada do motor
        st.caption(r.get("descricao", ""))


# ============================================================
# PARTE 5/6 — Monitor de Risco, Ruído Condicional, Confiabilidade REAL
# ============================================================

# ------------------------------------------------------------
# MONITOR DE RISCO (série k / k*) — FUNÇÕES
# ------------------------------------------------------------

def montar_tabela_monitor_risco() -> Optional[pd.DataFrame]:
    """
    Monta uma tabela com:
    - idx
    - k
    - k* (rolling)
    - flags simplificadas de regime
    """
    df = st.session_state.get("df_historico")
    s_k = st.session_state.get("serie_k")
    s_kstar = st.session_state.get("serie_k_star")

    if df is None or df.empty or s_k is None or s_kstar is None:
        return None

    if "idx" not in df.columns:
        df["idx"] = np.arange(1, len(df) + 1, dtype=int)

    base = pd.DataFrame({
        "idx": df["idx"].values,
        "k": s_k.values,
        "k_star_pct": s_kstar.values,
    })

    def classificar_kstar(v: float) -> str:
        if v < 15:
            return "estavel"
        elif v < 35:
            return "atencao"
        else:
            return "critico"

    base["estado_kstar"] = base["k_star_pct"].apply(classificar_kstar)
    return base


# ------------------------------------------------------------
# RUÍDO CONDICIONAL (NR%) — FUNÇÕES
# ------------------------------------------------------------

def calcular_mapa_ruido_global(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Mapa estrutural de ruído (NR%) simplificado:
    - dispersão média por passageiro
    - variação relativa
    """
    cols_pass = extrair_passageiros(df)
    if not cols_pass:
        return {}

    sub = df[cols_pass].astype(float)
    medias = sub.mean()
    desvios = sub.std().replace(0, 1.0)

    nr_pct = (desvios / (medias.abs() + 1.0)) * 100.0

    mapa = {
        "medias": medias.to_dict(),
        "desvios": desvios.to_dict(),
        "nr_pct": nr_pct.to_dict(),
    }

    # Resumo textual
    nr_values = list(nr_pct.values)
    nr_med = float(np.mean(nr_values))
    nr_max = float(np.max(nr_values))

    if nr_med < 25:
        resumo = (
            "🟢 Ruído estrutural global baixo — dispersão bem comportada "
            "em relação às magnitudes médias."
        )
    elif nr_med < 45:
        resumo = (
            "🟡 Ruído estrutural moderado — alguns passageiros com dispersão "
            "acima do ideal, mas ainda gerenciável."
        )
    else:
        resumo = (
            "🔴 Ruído estrutural elevado — passageiros com alta dispersão relativa, "
            "exigem coberturas mais robustas e cautela em decisões agressivas."
        )

    st.session_state["nr_mapa_global"] = mapa
    st.session_state["nr_resumo_textual"] = resumo
    return mapa


def calcular_mapa_ruido_condicional(
    df: pd.DataFrame,
    tabela_risco: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Calcula ruído condicional por regime k* (estavel/atencao/critico).
    """
    cols_pass = extrair_passageiros(df)
    if not cols_pass:
        return {}

    df_local = df.copy()
    if "idx" not in df_local.columns:
        df_local["idx"] = np.arange(1, len(df_local) + 1, dtype=int)

    merged = df_local.merge(tabela_risco[["idx", "estado_kstar"]], on="idx", how="left")
    resultados = {}

    for estado in ["estavel", "atencao", "critico"]:
        sub = merged[merged["estado_kstar"] == estado]
        if sub.empty:
            continue
        sub_pass = sub[cols_pass].astype(float)
        medias = sub_pass.mean()
        desvios = sub_pass.std().replace(0, 1.0)
        nr_pct = (desvios / (medias.abs() + 1.0)) * 100.0

        resultados[estado] = {
            "medias": medias.to_dict(),
            "desvios": desvios.to_dict(),
            "nr_pct": nr_pct.to_dict(),
            "n_series": int(len(sub)),
        }

    st.session_state["nr_mapa_condicional"] = resultados
    return resultados


# ------------------------------------------------------------
# TESTES DE CONFIABILIDADE REAL — FUNÇÕES
# ------------------------------------------------------------

def rodar_backtest_turbo_ultra_v156(
    df: pd.DataFrame,
    n_prev: int = 50,
) -> BacktestResult:
    """
    Backtest simplificado usando TURBO++ ULTRA como motor:
    - escolhe os últimos n_prev índices possíveis
    - compara previsão vs real
    - calcula % acerto total, acerto parcial (>=1), etc.
    """
    if "idx" not in df.columns:
        df["idx"] = np.arange(1, len(df) + 1, dtype=int)

    min_idx, max_idx = obter_intervalo_indices(df)

    # Precisamos de idx >= 2 para prever
    candidatos = [i for i in range(min_idx + 1, max_idx + 1)]
    if not candidatos:
        return BacktestResult(
            tabela=pd.DataFrame(),
            descricao="Sem índices suficientes para backtest.",
        )

    candidatos = candidatos[-n_prev:]
    resultados = []

    for idx in candidatos:
        res = turbo_ultra_v156(df, idx_alvo=idx)
        if "erro" in res:
            continue

        cols_pass = extrair_passageiros(df)
        real_row = df[df["idx"] == idx]
        if real_row.empty:
            continue
        real = [int(real_row.iloc[0][c]) for c in cols_pass]

        prev = res["final"]
        ac_total = calcular_acerto_total(real, prev)
        ac_parcial = 1 if ac_total > 0 else 0

        resultados.append(
            {
                "idx": idx,
                "acertos_totais": ac_total,
                "teve_acerto": ac_parcial,
            }
        )

    if not resultados:
        return BacktestResult(
            tabela=pd.DataFrame(),
            descricao="Backtest não conseguiu produzir resultados.",
        )

    df_res = pd.DataFrame(resultados)
    return BacktestResult(
        tabela=df_res,
        descricao="Backtest REAL utilizando motor TURBO++ ULTRA (V15.6 MAX)",
    )


def sintetizar_confiabilidade(df_res: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula indicadores de confiabilidade a partir do DF de backtest:
    - probabilidade de pelo menos 1 acerto
    - média de acertos
    """
    if df_res is None or df_res.empty:
        return {
            "p_ao_menos_um": 0.0,
            "media_acertos": 0.0,
            "n_prev": 0,
        }

    n_prev = len(df_res)
    p_ao_menos_um = float(df_res["teve_acerto"].mean())
    media_acertos = float(df_res["acertos_totais"].mean())

    return {
        "p_ao_menos_um": p_ao_menos_um,
        "media_acertos": media_acertos,
        "n_prev": n_prev,
    }


def gerar_texto_confiabilidade(stats: Dict[str, Any], alvo: float) -> str:
    """
    Gera texto interpretando os números da confiabilidade vs alvo.
    """
    p = stats["p_ao_menos_um"]
    n_prev = stats["n_prev"]

    if n_prev == 0:
        return (
            "Não foi possível calcular confiabilidade REAL, pois não houve "
            "previsões válidas no backtest."
        )

    pct = round(p * 100.0, 1)
    alvo_pct = round(alvo * 100.0, 1)

    if p >= alvo:
        return (
            f"🟢 Confiabilidade em regime saudável: {pct}% de casos com pelo menos 1 acerto, "
            f"acima ou igual ao alvo de {alvo_pct}%, considerando {n_prev} previsões testadas."
        )
    elif p >= alvo * 0.8:
        return (
            f"🟡 Confiabilidade intermediária: {pct}% de casos com pelo menos 1 acerto, "
            f"próximo do alvo de {alvo_pct}%. Em trechos estáveis, o Modo 6 Acertos pode "
            "ser considerado, mas ainda com cautela e preferindo ambientes premium "
            "(baixa NR% e k* em patamar estável)."
        )
    else:
        return (
            f"🔴 Confiabilidade abaixo do desejado: {pct}% de casos com pelo menos 1 acerto, "
            f"abaixo do alvo de {alvo_pct}%. Ideal reforçar calibração, revisar pesos do "
            "TURBO++ ULTRA e investigar o mapa de ruído condicional antes de qualquer "
            "decisão agressiva."
        )


# ============================================================
# PAINEL — 🚨 Monitor de Risco (k & k*) (V15.6 MAX)
# ============================================================

def painel_monitor_risco_v156() -> None:
    st.markdown("## 🚨 Monitor de Risco (k & k*) — V15.6 MAX")

    df = st.session_state.get("df_historico")
    if df is None or df.empty:
        st.warning("Carregue primeiro o histórico.")
        return

    tabela = montar_tabela_monitor_risco()
    if tabela is None or tabela.empty:
        st.info(
            "Séries k/k* ainda não foram calculadas. "
            "Rode o Pipeline V14-FLEX ULTRA para atualizar os sentinelas."
        )
        return

    st.markdown("### Série k / k* ao longo da estrada")
    st.dataframe(tabela.tail(40), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        k_med = float(tabela["k"].mean())
        k_max = float(tabela["k"].max())
        st.metric("k médio (estrada completa)", round(k_med, 4))
        st.metric("k máximo (estrada completa)", int(k_max))

    with col2:
        kstar_med = float(tabela["k_star_pct"].mean())
        kstar_max = float(tabela["k_star_pct"].max())
        st.metric("k* médio (%)", round(kstar_med, 1))
        st.metric("k* máximo (%)", round(kstar_max, 1))

    dist_estados = tabela["estado_kstar"].value_counts(normalize=True) * 100.0
    st.markdown("### Distribuição de regimes k*")
    st.json({k: round(v, 1) for k, v in dist_estados.to_dict().items()})

    st.caption(
        "Este painel resume a sensibilidade dos guardas (k*) e ajuda a entender "
        "em quais trechos o sistema está em regime estável, em atenção ou crítico."
    )


# ============================================================
# PAINEL — 📊 Ruído Condicional (NR%) (V15.6 MAX)
# ============================================================

def painel_ruido_condicional_v156() -> None:
    st.markdown("## 📊 Ruído Condicional (NR%) — V15.6 MAX")

    df = st.session_state.get("df_historico")
    if df is None or df.empty:
        st.warning("Carregue primeiro o histórico.")
        return

    tabela_risco = montar_tabela_monitor_risco()
    if tabela_risco is None or tabela_risco.empty:
        st.info(
            "Para calcular ruído condicional, rode antes o Pipeline "
            "V14-FLEX ULTRA para atualizar k/k*."
        )
        return

    # Global
    st.markdown("### 🌐 Mapa de Ruído Estrutural Global")
    mapa_global = calcular_mapa_ruido_global(df)
    st.json(mapa_global.get("nr_pct", {}))
    st.markdown(st.session_state.get("nr_resumo_textual", ""))

    # Condicional
    st.markdown("### 🧬 Mapa de Ruído Condicional por regime k*")
    mapa_cond = calcular_mapa_ruido_condicional(df, tabela_risco)

    for estado, info in mapa_cond.items():
        st.markdown(f"#### Regime: **{estado}** (n={info['n_series']})")
        st.json(info.get("nr_pct", {}))

    st.caption(
        "O ruído condicional ajuda a localizar trechos premium da estrada, "
        "onde a dispersão relativa é menor, favorecendo Modo TURBO++ ULTRA e 6 Acertos."
    )


# ============================================================
# PAINEL — 🧪 Testes de Confiabilidade REAL (V15.6 MAX)
# ============================================================

def painel_testes_confiabilidade_v156() -> None:
    st.markdown("## 🧪 Testes de Confiabilidade REAL — V15.6 MAX")

    df = st.session_state.get("df_historico")
    if df is None or df.empty:
        st.warning("Carregue primeiro o histórico.")
        return

    limite_prev = st.session_state.get("limite_max_prev", 300)

    n_prev = st.number_input(
        "Quantidade de previsões para backtest:",
        min_value=10,
        max_value=min(limite_prev, 1000),
        value=st.session_state.get("confiabilidade_n_prev", 50),
        step=10,
        key="n_prev_confiab_v156",
    )
    st.session_state["confiabilidade_n_prev"] = int(n_prev)

    alvo = st.slider(
        "Alvo de confiabilidade (prob. de pelo menos 1 acerto):",
        min_value=0.40,
        max_value=0.90,
        value=float(st.session_state.get("confiabilidade_alvo", 0.65)),
        step=0.05,
        key="alvo_confiab_v156",
    )
    st.session_state["confiabilidade_alvo"] = float(alvo)

    if st.button("Rodar Testes de Confiabilidade REAL", type="primary"):
        if st.session_state.get("flag_modo_seguro", True):
            if not limitar_operacao(len(df), st.session_state.get("limite_max_janela", 600), "Confiabilidade REAL"):
                return

        back = rodar_backtest_turbo_ultra_v156(df, n_prev=int(n_prev))
        st.session_state["confiabilidade_resultados"] = back

    back = st.session_state.get("confiabilidade_resultados")
    if back is None or back.tabela.empty:
        st.info("Nenhum resultado de backtest disponível ainda. Rode o teste acima.")
        return

    st.markdown("### Tabela de Backtest (TURBO++ ULTRA)")
    st.dataframe(back.tabela, use_container_width=True)

    stats = sintetizar_confiabilidade(back.tabela)
    texto = gerar_texto_confiabilidade(stats, alvo=float(alvo))

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "P(≥1 acerto)",
            f"{round(stats['p_ao_menos_um'] * 100.0, 1)}%",
        )
        st.metric(
            "Média de acertos totais",
            round(stats["media_acertos"], 3),
        )
    with col2:
        st.metric("Nº de previsões testadas", int(stats["n_prev"]))

    exibir_bloco_mensagem(
        "Leitura da confiabilidade REAL",
        texto_em_blocos(texto),
        emoji="📈",
    )

    st.caption(
        "Esses testes de confiabilidade REAL ajudam a decidir se o ambiente está "
        "pronto para usar Modo 6 Acertos, e também orientar o plano de calibração."
    )

# ------------------------------------------------------------
# FIM DA PARTE 5/6
# ------------------------------------------------------------
# ============================================================
# PARTE 6/6 — Modo 6 Acertos + Relatório Final + main()
# ============================================================

# ------------------------------------------------------------
# PAINEL — 🎯 Modo 6 Acertos — Execução (V15.6 MAX)
# ------------------------------------------------------------

def avaliar_prontidao_modo6_v156() -> Tuple[bool, str]:
    """
    Faz uma leitura integrada de:
    - Diagnóstico de impacto (regime + k*)
    - Confiabilidade REAL
    - Ruído Condicional
    - k* local

    Retorna (ok, texto_explicativo).
    """
    diag: DiagnosticoImpacto = st.session_state.get("diag_impacto")
    conf: BacktestResult = st.session_state.get("confiabilidade_resultados")
    mapa_cond = st.session_state.get("nr_mapa_condicional")
    serie_kstar = st.session_state.get("serie_k_star")

    # Faltando peças essenciais → não libera
    if diag is None or conf is None or conf.tabela is None or conf.tabela.empty:
        return False, (
            "Ainda não há informações suficientes (diagnóstico estrutural + "
            "confiabilidade REAL) para avaliar a prontidão do Modo 6 Acertos. "
            "Rode o Pipeline V14-FLEX ULTRA e os Testes de Confiabilidade REAL."
        )

    stats = sintetizar_confiabilidade(conf.tabela)
    alvo = float(st.session_state.get("confiabilidade_alvo", 0.65))
    p = stats["p_ao_menos_um"]
    pct = round(p * 100.0, 1)
    alvo_pct = round(alvo * 100.0, 1)

    # Ruído condicional médio em regime estável (se existir)
    nr_estavel = None
    if mapa_cond and "estavel" in mapa_cond:
        nr_vals = list(mapa_cond["estavel"]["nr_pct"].values())
        if nr_vals:
            nr_estavel = float(np.mean(nr_vals))

    # k* local
    kstar_local = None
    if serie_kstar is not None and len(serie_kstar) > 0:
        kstar_local = float(serie_kstar.iloc[-1])

    # Regras simples / transparentes
    motivos = []

    if diag.regime_estado == "ruptura":
        motivos.append("Estrada em ruptura no diagnóstico de impacto.")
    if diag.risco_modo6.startswith("❌"):
        motivos.append("Diagnóstico estrutural marcou Modo 6 Acertos como não recomendado.")
    if p < alvo * 0.8:
        motivos.append(
            f"Confiabilidade REAL baixa ({pct}% &lt; 80% do alvo {alvo_pct}%)."
        )
    if nr_estavel is not None and nr_estavel > 45:
        motivos.append(
            f"Ruído condicional elevado em regime estável (NR% ≈ {round(nr_estavel,1)})."
        )
    if kstar_local is not None and kstar_local >= 40:
        motivos.append(
            f"k* local muito alto ({round(kstar_local,1)}%), indicando turbulência."
        )

    if motivos:
        texto = (
            "🔴 **Prontidão atual: NÃO RECOMENDADO utilizar Modo 6 Acertos em modo agressivo.**\n\n"
            "Motivos principais identificados:\n"
            + "\n".join(f"- {m}" for m in motivos)
            + "\n\nSugestão: fortalecer calibração, revisar pesos do TURBO++ ULTRA, "
              "buscar trechos premium com NR% mais baixo e rodar novos testes "
              "de confiabilidade antes de insistir no Modo 6."
        )
        st.session_state["modo6_risco_ok"] = False
        return False, texto

    # Caso passe pelos filtros
    texto = (
        "🟢 **Prontidão atual: AMBIENTE ELEGÍVEL para Modo 6 Acertos (sob decisão manual).**\n\n"
        f"- P(≥1 acerto) ≈ {pct}% (alvo {alvo_pct}%).\n"
        "- Diagnóstico estrutural não detectou ruptura nem risco crítico imediato.\n"
        "- Ruído condicional em patamar compatível com exploração controlada.\n\n"
        "Recomendação: utilizar Modo 6 Acertos de forma cirúrgica, sempre "
        "associando o núcleo TURBO++ ULTRA aos mapas de ruído e ao plano de calibração."
    )
    st.session_state["modo6_risco_ok"] = True
    return True, texto


def painel_modo_6_acertos_v156() -> None:
    st.markdown("## 🎯 Modo 6 Acertos — Execução (V15.6 MAX)")

    df = st.session_state.get("df_historico")
    if df is None or df.empty:
        st.warning(
            "Carregue o histórico e rode o Pipeline V14-FLEX ULTRA + "
            "Testes de Confiabilidade REAL antes de usar este painel."
        )
        return

    # Avaliação de prontidão
    ok, texto_prontidao = avaliar_prontidao_modo6_v156()
    exibir_bloco_mensagem(
        "Prontidão para Modo 6 Acertos",
        texto_em_blocos(texto_prontidao),
        emoji="🧪",
    )

    # Série alvo (a próxima depois da última conhecida)
    if "idx" not in df.columns:
        df["idx"] = np.arange(1, len(df) + 1, dtype=int)

    min_idx, max_idx = obter_intervalo_indices(df)
    idx_alvo = max_idx + 1  # próxima série a ser prevista

    st.markdown("### 🎯 Série alvo conceitual")
    st.write(
        f"A próxima série a ser prevista, conceitualmente, seria a **C{idx_alvo}**, "
        "tomando a estrada de C1 até C"
        f"{max_idx} como base."
    )

    # Núcleo vindo do TURBO++ ULTRA
    turbo = st.session_state.get("turbo_ultra_result")
    if turbo is None or not turbo.get("final"):
        st.info(
            "Ainda não há uma previsão TURBO++ ULTRA registrada. "
            "Vá até o painel `🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15.6 MAX)`, "
            "rode a previsão para o índice alvo desejado e depois volte aqui."
        )
        return

    serie_nucleo = turbo["final"]
    st.markdown("### 🔧 Núcleo proposto pelo TURBO++ ULTRA")
    st.code(formatar_lista_passageiros(serie_nucleo), language="text")

    # Geração de coberturas a partir do núcleo (usando Micro-Leque profundo)
    st.markdown("### 🛰️ Coberturas derivadas do núcleo (Micro-Leque Profundo)")
    leques = micro_leque_profundo_v156(serie_nucleo, intensidade=0.25)
    for i, lq in enumerate(leques, start=1):
        st.text(f"Cobertura {i}: {formatar_lista_passageiros(lq)}")

    st.markdown("### ⚖️ Decisão manual do operador")
    st.write(
        "A ideia do Modo 6 Acertos V15.6 MAX é ser **semi-automático**: "
        "o sistema apresenta núcleo + coberturas + diagnóstico de prontidão, "
        "mas **a decisão final** (quais listas usar, se vale a pena atuar, etc.) "
        "é sempre do operador."
    )

    escolha = st.multiselect(
        "Selecione quais listas você considera elegíveis para operação (núcleo/coberturas):",
        options=["Núcleo"] + [f"Cobertura {i}" for i in range(1, len(leques) + 1)],
        default=["Núcleo"],
        key="modo6_escolhas_v156",
    )

    resumo_escolha = {
        "nucleo": serie_nucleo if "Núcleo" in escolha else None,
        "coberturas": [
            leques[i]
            for i in range(len(leques))
            if f"Cobertura {i+1}" in escolha
        ],
        "prontidao_ok": ok,
    }
    st.session_state["modo6_resultados"] = resumo_escolha

    exibir_bloco_mensagem(
        "Resumo da seleção Modo 6 Acertos",
        texto_em_blocos(
            "Núcleo selecionado: "
            f"{'Sim' if resumo_escolha['nucleo'] is not None else 'Não'}\n"
            f"Qtd. de coberturas selecionadas: {len(resumo_escolha['coberturas'])}\n"
            f"Prontidão indicada pelo sistema: {'OK' if ok else 'Não recomendado'}."
        ),
        emoji="🎯",
    )

    st.caption(
        "Importante: o Modo 6 Acertos V15.6 MAX não executa nenhuma ação automática "
        "no mundo real. Ele apenas organiza a informação e sugere ambientes mais "
        "ou menos favoráveis, deixando a decisão final sempre com o operador."
    )


# ------------------------------------------------------------
# PAINEL — 📑 Relatório Final V15.6 MAX
# ------------------------------------------------------------

def montar_relatorio_final_v156() -> str:
    """
    Consolida em texto:
    - Histórico (n_series, n_pass, k)
    - Diagnóstico de impacto
    - k / k* global
    - Ruído global / condicional
    - Confiabilidade REAL
    - Núcleo TURBO++ + Modo 6 Acertos (se houver)
    """
    df = st.session_state.get("df_historico")
    diag: DiagnosticoImpacto = st.session_state.get("diag_impacto")
    plano: PlanoCalibracao = st.session_state.get("plano_calibracao")
    tabela_risco = montar_tabela_monitor_risco()
    mapa_global = st.session_state.get("nr_mapa_global")
    mapa_cond = st.session_state.get("nr_mapa_condicional")
    conf: BacktestResult = st.session_state.get("confiabilidade_resultados")
    turbo = st.session_state.get("turbo_ultra_result")
    modo6 = st.session_state.get("modo6_resultados")

    linhas = []

    # Histórico
    if df is not None and not df.empty:
        info_hist = analisar_historico_flex_ultra(df)
        linhas.append("=== HISTÓRICO FLEX ULTRA ===")
        linhas.append(f"- Nº de séries (carros): {info_hist['n_series']}")
        linhas.append(f"- Nº de passageiros por carro: {info_hist['n_passageiros']}")
        linhas.append(f"- Coluna k detectada: {'Sim' if info_hist['tem_k'] else 'Não'}")
        linhas.append(f"- Última série da estrada: C{info_hist['indice_ultima_serie'] + 1}")
        linhas.append("")
    else:
        linhas.append("Histórico não disponível no momento do relatório.")
        linhas.append("")

    # Diagnóstico de impacto
    linhas.append("=== DIAGNÓSTICO DE IMPACTO / REGIME ===")
    if diag is not None:
        linhas.append(f"- Índice alvo analisado: C{diag.idx_alvo}")
        linhas.append(f"- Regime estrutural: {diag.regime_estado}")
        linhas.append(f"- k* local aproximado: {round(diag.k_star_pct, 1)}%")
        linhas.append(f"- Leitura de risco para Modo 6 Acertos: {diag.risco_modo6}")
        linhas.append("")
        linhas.append("Comentário do diagnóstico:")
        linhas.append(texto_em_blocos(diag.comentario, largura=90))
        linhas.append("")
    else:
        linhas.append("Diagnóstico de impacto ainda não foi gerado.")
        linhas.append("")

    # k / k* global
    linhas.append("=== SENTINELAS k / k* (GLOBAL) ===")
    if tabela_risco is not None and not tabela_risco.empty:
        k_med = float(tabela_risco["k"].mean())
        k_max = float(tabela_risco["k"].max())
        kstar_med = float(tabela_risco["k_star_pct"].mean())
        kstar_max = float(tabela_risco["k_star_pct"].max())
        linhas.append(f"- k médio: {round(k_med, 4)} / k máximo: {int(k_max)}")
        linhas.append(
            f"- k* médio: {round(kstar_med, 1)}% / k* máximo: {round(kstar_max, 1)}%"
        )
        dist_est = tabela_risco["estado_kstar"].value_counts(normalize=True) * 100.0
        linhas.append("- Distribuição de regimes k* (em %):")
        for estado, pct in dist_est.to_dict().items():
            linhas.append(f"  - {estado}: {round(pct, 1)}%")
        linhas.append("")
    else:
        linhas.append("Séries k/k* ainda não avaliadas.")
        linhas.append("")

    # Ruído
    linhas.append("=== RUÍDO ESTRUTURAL / CONDICIONAL (NR%) ===")
    if mapa_global is not None and mapa_global.get("nr_pct"):
        nr_vals = list(mapa_global["nr_pct"].values())
        nr_med = float(np.mean(nr_vals))
        nr_max = float(np.max(nr_vals))
        linhas.append(
            f"- NR% médio global: {round(nr_med, 1)}% / NR% máximo global: {round(nr_max, 1)}%"
        )
        linhas.append("")
    else:
        linhas.append("Mapa de ruído global ainda não foi calculado.")
        linhas.append("")

    if mapa_cond:
        linhas.append("Resumo condicional (NR% por regime k*):")
        for estado, info in mapa_cond.items():
            vals = list(info["nr_pct"].values())
            if vals:
                linhas.append(
                    f"- {estado} (n={info['n_series']}): NR% médio ≈ {round(np.mean(vals),1)}%"
                )
        linhas.append("")
    else:
        linhas.append("Mapa de ruído condicional ainda não foi calculado.")
        linhas.append("")

    # Plano de calibração
    linhas.append("=== MINI PLANO DE CALIBRAÇÃO ===")
    if plano is not None:
        linhas.append(f"Foco: {plano.foco}")
        for passo in plano.passos:
            linhas.append(f"- {passo}")
        linhas.append("")
    else:
        linhas.append("Plano de calibração ainda não foi gerado.")
        linhas.append("")

    # Confiabilidade REAL
    linhas.append("=== TESTES DE CONFIABILIDADE REAL ===")
    if conf is not None and conf.tabela is not None and not conf.tabela.empty:
        stats = sintetizar_confiabilidade(conf.tabela)
        alvo = float(st.session_state.get("confiabilidade_alvo", 0.65))
        linhas.append(f"- Nº de previsões testadas: {stats['n_prev']}")
        linhas.append(
            f"- P(≥1 acerto): {round(stats['p_ao_menos_um'] * 100.0,1)}% "
            f"(alvo: {round(alvo*100.0,1)}%)"
        )
        linhas.append(
            f"- Média de acertos totais por previsão: {round(stats['media_acertos'],3)}"
        )
        linhas.append("")
        linhas.append("Leitura qualitativa da confiabilidade:")
        linhas.append(texto_em_blocos(gerar_texto_confiabilidade(stats, alvo), largura=90))
        linhas.append("")
    else:
        linhas.append("Ainda não foram rodados Testes de Confiabilidade REAL.")
        linhas.append("")

    # Núcleo / TURBO++ ULTRA
    linhas.append("=== NÚCLEO TURBO++ ULTRA (V15.6 MAX) ===")
    if turbo is not None and turbo.get("final"):
        linhas.append("Previsão final TURBO++ ULTRA (núcleo proposto):")
        linhas.append(formatar_lista_passageiros(turbo["final"]))
        linhas.append("")
        linhas.append("Núcleo (S6 Profundo):")
        linhas.append(formatar_lista_passageiros(turbo["serie_s6"]))
        linhas.append("")
        linhas.append("Pesos usados (S6 / Monte Carlo / Micro-Leque):")
        linhas.append(str(turbo["pesos"]))
        linhas.append("")
    else:
        linhas.append("Nenhuma previsão TURBO++ ULTRA registrada neste relatório.")
        linhas.append("")

    # Modo 6 Acertos
    linhas.append("=== MODO 6 ACERTOS — RESUMO ===")
    if modo6 is not None:
        linhas.append(
            f"- Núcleo selecionado pelo operador: "
            f"{'Sim' if modo6.get('nucleo') is not None else 'Não'}"
        )
        linhas.append(
            f"- Qtd. de coberturas selecionadas: {len(modo6.get('coberturas', []))}"
        )
        linhas.append(
            f"- Prontidão indicada pelo sistema: "
            f"{'OK' if modo6.get('prontidao_ok') else 'Não recomendado'}"
        )
        linhas.append("")
    else:
        linhas.append("Modo 6 Acertos ainda não foi utilizado nesta sessão.")
        linhas.append("")

    texto_final = "\n".join(linhas)
    st.session_state["relatorio_final_texto"] = texto_final
    st.session_state["relatorio_final_estrutura"] = {
        "linhas": linhas,
    }
    return texto_final


def painel_relatorio_final_v156() -> None:
    st.markdown("## 📑 Relatório Final V15.6 MAX")

    texto = montar_relatorio_final_v156()
    st.text_area(
        "Relatório consolidado (copie e cole para análise externa):",
        value=texto,
        height=500,
        key="txt_relatorio_final_v156",
    )

    st.caption(
        "Este relatório consolida a visão do V15.6 MAX: histórico, regime, ruído, "
        "confiabilidade, núcleo TURBO++ ULTRA e status do Modo 6 Acertos."
    )


# ------------------------------------------------------------
# FUNÇÃO PRINCIPAL — ROUTING GERAL DO APP
# ------------------------------------------------------------

def main():
    st.title("🚗 Predict Cars V15.6 MAX")

    painel = construir_navegacao_v156()

    if painel == "📥 Histórico — Entrada FLEX ULTRA (V15.6 MAX)":
        painel_entrada_historico_flex_ultra()

    elif painel == "🔍 Pipeline V14-FLEX ULTRA (V15.6 MAX)":
        painel_pipeline_v14_flex_ultra_v156()

    elif painel == "💡 Replay LIGHT (V15.6 MAX)":
        painel_replay_light_v156()

    elif painel == "📅 Replay ULTRA (V15.6 MAX)":
        painel_replay_ultra_v156()

    elif painel == "🎯 Replay ULTRA Unitário (V15.6 MAX)":
        painel_replay_unitario_v156()

    elif painel == "🚨 Monitor de Risco (k & k*) (V15.6 MAX)":
        painel_monitor_risco_v156()

    elif painel == "🧪 Testes de Confiabilidade REAL (V15.6 MAX)":
        painel_testes_confiabilidade_v156()

    elif painel == "📊 Ruído Condicional (NR%) (V15.6 MAX)":
        painel_ruido_condicional_v156()

    elif painel == "🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15.6 MAX)":
        painel_turbo_ultra_v156()

    elif painel == "🎯 Modo 6 Acertos — Execução (V15.6 MAX)":
        painel_modo_6_acertos_v156()

    elif painel == "📑 Relatório Final V15.6 MAX":
        painel_relatorio_final_v156()


if __name__ == "__main__":
    main()

# ------------------------------------------------------------
# FIM DA PARTE 6/6 — FIM DO APP V15.6 MAX
# ------------------------------------------------------------
