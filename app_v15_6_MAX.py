# ============================================================
# Predict Cars V15.7 MAX
# Versão MAX: núcleo + coberturas + interseção estatística
# Pipeline V14-FLEX ULTRA + Replay LIGHT/ULTRA + TURBO++ ULTRA Anti-Ruído
# + Monitor de Risco (k & k*) + Painel de Ruído Condicional (NR%)
# + Testes de Confiabilidade REAL + Modo 6 Acertos (núcleo V15.6 MAX)
# + Relatório Final V15.7 MAX (novo módulo)
# ============================================================

import math
import itertools
import textwrap
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Configuração da página (V15.7 MAX)
# ------------------------------------------------------------
st.set_page_config(
    page_title="Predict Cars V15.7 MAX",
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
# Sessão / Estado Global V15.7 MAX
# ============================================================

def init_session_state_v156() -> None:
    """
    Inicializa todas as chaves de sessão necessárias para o núcleo V15.6/V15.7 MAX.
    Mantém jeitão de app denso, com estado explícito e controlado.
    """
    ss = st.session_state

    # Histórico FLEX ULTRA
    ss.setdefault("df_historico", None)          # DataFrame principal (carros/pass.
    ss.setdefault("df_historico_bruto", None)    # Texto/DF bruto antes de limpeza
    ss.setdefault("n_passageiros", None)         # Número de passageiros detectado
    ss.setdefault("k_col_name", None)            # Nome da coluna k (se existir)
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

    # Modo 6 Acertos (núcleo V15.6 MAX)
    ss.setdefault("modo6_inputs", {})
    ss.setdefault("modo6_resultados", None)
    ss.setdefault("modo6_risco_ok", False)       # liberado ou não para usar 6 acertos

    # Relatório Final (texto e estrutura internos)
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
    Salva histórico e metadados na sessão V15.7 MAX.
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
    st.markdown("## 📥 Histórico — Entrada FLEX ULTRA (V15.7 MAX)")

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
            key="upload_historico_v157",
        )
        tem_k_arquivo = st.checkbox(
            "Histórico possui coluna k na última posição?",
            value=True,
            key="ck_tem_k_arquivo_v157",
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
            key="txt_historico_v157",
        )
        tem_k_texto = st.checkbox(
            "Texto possui coluna k na última posição?",
            value=True,
            key="ck_tem_k_texto_v157",
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
# Navegação lateral V15.7 MAX
# ============================================================

def construir_navegacao_v156() -> str:
    """
    Constrói a navegação lateral oficial do V15.7 MAX.
    Retorna o nome do painel selecionado.
    """
    st.sidebar.markdown("## 🚗 Predict Cars V15.7 MAX")

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
        7️⃣ Modo 6 Acertos (núcleo V15.6)<br/>
        8️⃣ Relatório Final V15.7 MAX
        </small>
        """,
        unsafe_allow_html=True,
    )

    painel = st.sidebar.radio(
        "Navegação",
        [
            "📥 Histórico — Entrada FLEX ULTRA (V15.7 MAX)",
            "🔍 Pipeline V14-FLEX ULTRA (V15.7 MAX)",
            "💡 Replay LIGHT (V15.7 MAX)",
            "📅 Replay ULTRA (V15.7 MAX)",
            "🎯 Replay ULTRA Unitário (V15.7 MAX)",
            "🚨 Monitor de Risco (k & k*) (V15.7 MAX)",
            "🧪 Testes de Confiabilidade REAL (V15.7 MAX)",
            "📊 Ruído Condicional (NR%) (V15.7 MAX)",
            "🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15.7 MAX)",
            "🎯 Modo 6 Acertos — Execução (V15.7 MAX)",
            "📑 Relatório Final V15.7 MAX",
        ],
        index=0,
        key="nav_v157",
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
# PARTE 2/6 — Pipeline V14-FLEX ULTRA + Sentinelas k/k*
# + Replay LIGHT / Replay ULTRA / Replay Unitário (V15.7 MAX)
# ============================================================

# ------------------------------------------------------------
# Funções utilitárias do Pipeline V14-FLEX ULTRA
# ------------------------------------------------------------

def obter_intervalo_indices(df: pd.DataFrame) -> Tuple[int, int]:
    """Retorna (min_idx, max_idx) assumindo que df['idx'] existe."""
    if "idx" not in df.columns:
        df = df.copy()
        df["idx"] = np.arange(1, len(df) + 1)
    return int(df["idx"].min()), int(df["idx"].max())


def extrair_passageiros(df: pd.DataFrame) -> List[str]:
    """Retorna lista de colunas de passageiros (P1..Pn)."""
    return [c for c in df.columns if c.lower().startswith("p")]


def obter_n_passageiros(df: pd.DataFrame) -> int:
    """Número de passageiros detectado no DF."""
    return len(extrair_passageiros(df))


# ------------------------------------------------------------
# Cálculo dos Sentinelas k e k* (V14-FLEX ULTRA)
# ------------------------------------------------------------

def calcular_k_series(df: pd.DataFrame) -> pd.Series:
    """Extrai a coluna k, se existir; caso contrário, gera zeros."""
    if "k" in df.columns:
        return df["k"].astype(int).copy()
    return pd.Series([0] * len(df))


def calcular_k_star_series(
    df: pd.DataFrame,
    janela: int = 40,
) -> pd.Series:
    """
    Calcula k* (rolling) como percentual de guardas que acertam exatamente
    o carro anterior, considerando uma janela de 40 valores.
    """
    if "idx" not in df.columns:
        df = df.copy()
        df["idx"] = np.arange(1, len(df) + 1)

    cols_pass = extrair_passageiros(df)
    valores = []

    for i in range(len(df)):
        if i == 0:
            valores.append(0.0)
            continue

        atual = df.iloc[i][cols_pass].tolist()
        ant = df.iloc[i - 1][cols_pass].tolist()
        acertos = sum(1 for a, b in zip(atual, ant) if a == b)
        pct = (acertos / len(cols_pass)) * 100.0
        valores.append(pct)

    s = pd.Series(valores)
    s_rolled = s.rolling(window=janela, min_periods=1).mean()
    return s_rolled.clip(lower=0.0, upper=100.0)


# ------------------------------------------------------------
# Painel — Pipeline V14-FLEX ULTRA (V15.7 MAX)
# ------------------------------------------------------------

def painel_pipeline_v14_flex_ultra_v156() -> None:
    st.markdown("## 🔍 Pipeline V14-FLEX ULTRA (V15.7 MAX)")

    df = st.session_state.get("df_historico")
    if df is None or df.empty:
        st.warning("Carregue primeiro o histórico FLEX ULTRA.")
        return

    if st.button("Rodar Pipeline V14-FLEX ULTRA", type="primary"):
        if st.session_state.get("flag_modo_seguro", True):
            if not limitar_operacao(
                len(df),
                st.session_state.get("limite_max_janela", 600),
                "Pipeline V14-FLEX ULTRA",
            ):
                return

        try:
            df_local = df.copy()
            if "idx" not in df_local.columns:
                df_local["idx"] = np.arange(1, len(df_local) + 1)

            # k
            s_k = calcular_k_series(df_local)

            # k*
            s_kstar = calcular_k_star_series(df_local, janela=40)

            # Registrar no estado
            st.session_state["serie_k"] = s_k
            st.session_state["serie_k_star"] = s_kstar

            st.success("Pipeline V14-FLEX ULTRA executado com sucesso.")

        except Exception as e:
            st.error(f"Erro ao rodar pipeline: {e}")

    # Exibir últimas linhas
    s_k = st.session_state.get("serie_k")
    s_kstar = st.session_state.get("serie_k_star")
    if s_k is not None and s_kstar is not None:
        st.markdown("### Últimas séries k / k*")
        df_view = pd.DataFrame({
            "idx": df["idx"] if "idx" in df.columns else np.arange(1, len(df) + 1),
            "k": s_k,
            "k_star_pct": s_kstar,
        })
        st.dataframe(df_view.tail(30), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("k* médio (%)", round(float(s_kstar.mean()), 2))
        with col2:
            st.metric("k* máximo (%)", round(float(s_kstar.max()), 2))


# ============================================================
# REPLAY LIGHT (V15.7 MAX)
# ============================================================

def calcular_acerto_total(real: List[int], prev: List[int]) -> int:
    """Conta quantos passageiros foram acertados exatamente."""
    return sum(1 for a, b in zip(real, prev) if a == b)


def replay_light_executar(df: pd.DataFrame, idx: int) -> Dict[str, Any]:
    """
    Replay LIGHT — versão enxuta, mas mantendo jeitão original.
    - Compara série idx-1 com idx.
    """
    if idx <= 1:
        return {"erro": "Índice inválido para replay LIGHT."}

    cols = extrair_passageiros(df)
    linha_ant = df[df["idx"] == idx - 1]
    linha_atual = df[df["idx"] == idx]

    if linha_ant.empty or linha_atual.empty:
        return {"erro": "Não foi possível localizar séries."}

    ant = [int(linha_ant.iloc[0][c]) for c in cols]
    atual = [int(linha_atual.iloc[0][c]) for c in cols]

    ac = calcular_acerto_total(atual, ant)

    return {
        "idx": idx,
        "anterior": ant,
        "real": atual,
        "acertos": ac,
        "descricao": "Replay LIGHT (V15.7 MAX)",
    }


def painel_replay_light_v156() -> None:
    st.markdown("## 💡 Replay LIGHT (V15.7 MAX)")

    df = st.session_state.get("df_historico")
    if df is None:
        st.warning("Carregue o histórico antes.")
        return

    min_idx, max_idx = obter_intervalo_indices(df)

    idx = st.number_input(
        "Índice alvo do replay LIGHT:",
        min_value=min_idx + 1,
        max_value=max_idx,
        value=max_idx,
        step=1,
        key="idx_replay_light_v157",
    )

    if st.button("Rodar Replay LIGHT", type="primary"):
        if st.session_state.get("flag_modo_seguro", True):
            if not limitar_operacao(len(df), 2000, "Replay LIGHT"):
                return

        r = replay_light_executar(df, idx)
        st.session_state["replay_light_result"] = r

    r = st.session_state.get("replay_light_result")
    if r:
        st.markdown("### Resultado do Replay LIGHT")
        st.json(r)


# ============================================================
# REPLAY ULTRA (V15.7 MAX)
# ============================================================

def replay_ultra_executar(df: pd.DataFrame, idx_alvo: int) -> Dict[str, Any]:
    """
    Replay ULTRA — versão detalhada.
    Compara real vs previsão S6, MC e final.
    """
    res_turbo = turbo_ultra_v156(df, idx_alvo)

    cols = extrair_passageiros(df)
    linha_real = df[df["idx"] == idx_alvo]

    if linha_real.empty:
        return {"erro": "Real não encontrado no replay ULTRA."}

    real = [int(linha_real.iloc[0][c]) for c in cols]

    ac_total = calcular_acerto_total(real, res_turbo["final"])

    return {
        "idx": idx_alvo,
        "real": real,
        "turbo": res_turbo,
        "acertos_totais": ac_total,
        "descricao": "Replay ULTRA (V15.7 MAX)",
    }


def painel_replay_ultra_v156() -> None:
    st.markdown("## 📅 Replay ULTRA (V15.7 MAX)")

    df = st.session_state.get("df_historico")
    if df is None:
        st.warning("Carregue o histórico antes.")
        return

    min_idx, max_idx = obter_intervalo_indices(df)

    idx = st.number_input(
        "Índice alvo para Replay ULTRA:",
        min_value=min_idx + 1,
        max_value=max_idx,
        value=max_idx,
        step=1,
        key="idx_replay_ultra_v157",
    )

    if st.button("Rodar Replay ULTRA", type="primary"):
        if st.session_state.get("flag_modo_seguro", True):
            if not limitar_operacao(len(df), 1500, "Replay ULTRA"):
                return

        r = replay_ultra_executar(df, idx)
        st.session_state["replay_ultra_result"] = r

    r = st.session_state.get("replay_ultra_result")
    if r and "erro" not in r:
        st.markdown("### Resultado Replay ULTRA")
        st.json(r)


# ============================================================
# REPLAY ULTRA UNITÁRIO (V15.7 MAX)
# ============================================================

def replay_unitario_executar(df: pd.DataFrame, idx_alvo: int) -> Dict[str, Any]:
    """Replay ULTRA Unitário — comparação final vs real."""
    if idx_alvo <= 1:
        return {"erro": "Índice inválido para replay unitário."}

    res_turbo = turbo_ultra_v156(df, idx_alvo)

    cols = extrair_passageiros(df)
    linha_real = df[df["idx"] == idx_alvo]

    if linha_real.empty:
        return {"erro": "Real não encontrado."}

    real = [int(linha_real.iloc[0][c]) for c in cols]

    ac = calcular_acerto_total(real, res_turbo["final"])

    return {
        "idx": idx_alvo,
        "real": real,
        "prev_final": res_turbo["final"],
        "acertos_totais": ac,
        "descricao": "Replay ULTRA Unitário (V15.7 MAX)",
    }


def painel_replay_unitario_v156() -> None:
    st.markdown("## 🎯 Replay ULTRA Unitário (V15.7 MAX)")

    df = st.session_state.get("df_historico")
    if df is None:
        st.warning("Carregue o histórico antes.")
        return

    min_idx, max_idx = obter_intervalo_indices(df)

    idx = st.number_input(
        "Índice alvo para Replay Unitário:",
        min_value=min_idx + 1,
        max_value=max_idx,
        value=max_idx,
        step=1,
        key="idx_replay_unitario_v157",
    )

    if st.button("Rodar Replay ULTRA Unitário", type="primary"):
        if st.session_state.get("flag_modo_seguro", True):
            if not limitar_operacao(len(df), 1500, "Replay ULTRA Unitário"):
                return

        r = replay_unitario_executar(df, idx)
        st.session_state["replay_unitario_result"] = r

    r = st.session_state.get("replay_unitario_result")
    if r and "erro" not in r:
        st.markdown("### Resultado Replay ULTRA Unitário")
        st.json(r)

# ------------------------------------------------------------
# FIM DA PARTE 2/6
# ------------------------------------------------------------
# ============================================================
# PARTE 3/6 — TURBO++ ULTRA ANTI-RUÍDO (V15.7 MAX)
# Núcleo + Micro-Leque + Monte Carlo + Pesos dinâmicos por k*
# ============================================================

# ------------------------------------------------------------
# S6 PROFUNDO — NÚCLEO (V15.7 MAX)
# ------------------------------------------------------------

def s6_profundo_v157(df: pd.DataFrame, idx_alvo: int) -> List[int]:
    """Núcleo S6 profundo — mesmo jeitão do V15.6, atualizado."""
    if idx_alvo <= 1:
        return []

    n_pass = obter_n_passageiros(df)
    cols_pass = extrair_passageiros(df)

    serie_prev_df = df[df["idx"] == idx_alvo - 1]
    if serie_prev_df.empty:
        return []

    serie_prev = [int(serie_prev_df.iloc[0][c]) for c in cols_pass]

    # similaridade S6-like
    def sim_s6(a, b):
        return sum(1 for x, y in zip(a, b) if x == y)

    melhores = []
    for _, row in df.iterrows():
        linha = [int(row[c]) for c in cols_pass]
        sim = sim_s6(serie_prev, linha)
        melhores.append((sim, linha))

    melhores.sort(reverse=True, key=lambda x: x[0])
    top6 = melhores[:6]

    if not top6:
        return serie_prev.copy()

    pesos = np.array([1.0, 1.2, 1.3, 1.4, 1.6, 1.8][:len(top6)])
    pesos = pesos / pesos.sum()

    acum = np.zeros(n_pass)
    for p, (_, linha) in zip(pesos, top6):
        acum += p * np.array(linha)

    return [int(round(x)) for x in acum]


# ------------------------------------------------------------
# MICRO-LEQUE PROFUNDO (V15.7 MAX)
# ------------------------------------------------------------

def micro_leque_profundo_v157(serie_base: List[int], intensidade: float = 0.25) -> List[List[int]]:
    """Gera micro-leques robustos em torno de uma série base."""
    leques = []
    for desloc in [-2, -1, 1, 2]:
        var = [max(0, int(round(x + desloc * intensidade))) for x in serie_base]
        leques.append(var)
    return leques


# ------------------------------------------------------------
# MONTE CARLO PROFUNDO ULTRA (V15.7 MAX)
# ------------------------------------------------------------

def monte_carlo_profundo_v157(
    serie_base: List[int],
    n_amostras: int = 250,
    ruido_base: float = 0.65
) -> List[List[int]]:
    """Monte Carlo profundo — idêntico ao V15.6, versão MAX."""
    out = []
    for _ in range(n_amostras):
        am = []
        for x in serie_base:
            pert = np.random.normal(loc=x, scale=ruido_base)
            am.append(max(0, int(round(pert))))
        out.append(am)
    return out


# ------------------------------------------------------------
# DIVERGÊNCIA S6 vs MONTE CARLO
# ------------------------------------------------------------

def divergencia_s6_mc_v157(serie_s6: List[int], mc_list: List[List[int]]) -> float:
    """Divergência média entre núcleo e amostras MC."""
    if not serie_s6 or not mc_list:
        return 0.0
    n = len(serie_s6)
    diffs = []
    for am in mc_list:
        dif = sum(abs(a - b) for a, b in zip(serie_s6, am)) / (n * 10)
        diffs.append(dif)
    return float(np.mean(diffs))


# ------------------------------------------------------------
# PESOS POR k*
# ------------------------------------------------------------

def ajustar_pesos_por_kstar_v157(kstar_pct: float) -> Dict[str, float]:
    """Pesos automáticos guiados por k*, mantendo 100% o padrão anterior."""
    if kstar_pct < 15:
        return {"peso_s6": 0.60, "peso_mc": 0.25, "peso_micro": 0.15}
    elif kstar_pct < 35:
        return {"peso_s6": 0.45, "peso_mc": 0.35, "peso_micro": 0.20}
    else:
        return {"peso_s6": 0.30, "peso_mc": 0.55, "peso_micro": 0.15}


# ------------------------------------------------------------
# MOTOR — TURBO++ ULTRA V15.7 MAX
# ------------------------------------------------------------

def turbo_ultra_v157(
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
    TURBO++ ULTRA — V15.7 MAX
    Mesma estrutura do V15.6, mas registrada para o novo Relatório Final.
    """

    # -------------------------------
    # 1) Série base S6
    # -------------------------------
    if "idx" not in df.columns:
        df = df.copy()
        df["idx"] = np.arange(1, len(df) + 1)

    if idx_alvo <= 1:
        return {"erro": "Índice alvo inválido."}

    serie_s6 = s6_profundo_v157(df, idx_alvo)
    if not serie_s6:
        return {"erro": "Falha ao gerar núcleo S6."}

    # -------------------------------
    # 2) Micro-Leque
    # -------------------------------
    leques = micro_leque_profundo_v157(serie_s6)
    leques_flat = leques.copy()

    # -------------------------------
    # 3) Monte Carlo profundo
    # -------------------------------
    mc_list = monte_carlo_profundo_v157(serie_s6, n_amostras=200)
    div_s6_mc = divergencia_s6_mc_v157(serie_s6, mc_list)

    # -------------------------------
    # 4) Pesos por k*
    # -------------------------------
    s_kstar = st.session_state.get("serie_k_star")
    if s_kstar is not None and len(s_kstar) >= idx_alvo:
        kstar_pct = float(s_kstar.iloc[idx_alvo - 1])
    else:
        kstar_pct = 20.0

    pesos_auto = ajustar_pesos_por_kstar_v157(kstar_pct)

    # -------------------------------
    # 5) Interseção estatística
    # -------------------------------
    n_pass = len(serie_s6)
    combinacoes = [serie_s6] + mc_list[:20] + leques_flat

    arr = np.zeros(n_pass)
    for c in combinacoes:
        tmp = np.array(c)
        arr += (
            pesos_auto["peso_s6"] * tmp +
            pesos_auto["peso_micro"] * tmp +
            pesos_auto["peso_mc"] * tmp
        )

    arr = arr / len(combinacoes)
    final = [int(round(x)) for x in arr]

    # -------------------------------
    # 6) Registrar sessão (logs V15.7)
    # -------------------------------
    st.session_state["turbo_ultra_result"] = {
        "final": final,
        "serie_s6": serie_s6,
        "micro_leque": leques_flat,
        "mc_amostras": mc_list[:20],
        "divergencia_s6_mc": div_s6_mc,
        "pesos": {
            "peso_s6": peso_s6,
            "peso_mc": peso_mc,
            "peso_micro": peso_micro,
            "suavizacao_idx": suavizacao_idx,
            "profundidade_micro": profundidade_micro,
            "fator_antirruido": fator_antirruido,
            "elasticidade_nucleo": elasticidade_nucleo,
            "intensidade_turbulencia": intensidade_turbulencia,
        },
        "pesos_auto": pesos_auto,
        "k_star_local": kstar_pct,
        "descricao": "Previsão TURBO++ ULTRA — V15.7 MAX",
    }

    return st.session_state["turbo_ultra_result"]
# ============================================================
# PARTE 4/6 — Painel TURBO++ ULTRA Anti-Ruído (V15.7 MAX)
# ============================================================

def painel_turbo_ultra_v157() -> None:
    st.markdown("## 🚀 Modo TURBO++ ULTRA Anti-Ruído (V15.7 MAX)")

    df = st.session_state.get("df_historico")
    if df is None or df.empty:
        st.warning("Carregue primeiro o histórico FLEX ULTRA.")
        return

    # Garante coluna idx
    if "idx" not in df.columns:
        df = df.copy()
        df["idx"] = np.arange(1, len(df) + 1)
        st.session_state["df_historico"] = df

    min_idx, max_idx = obter_intervalo_indices(df)

    # Índice alvo
    idx_alvo = st.number_input(
        "Índice alvo (previsão TURBO++ ULTRA):",
        min_value=min_idx + 1,
        max_value=max_idx,
        value=max_idx,
        step=1,
        key="idx_turbo_ultra_v157",
    )

    # ------------------------------------------------------------
    # 🔧 Ajustes Manuais Básicos — Pesos do motor
    # ------------------------------------------------------------
    with st.expander("⚙️ Ajustes Manuais — Pesos do Motor (S6 / MC / Micro-Leque)"):
        pesos_atual = st.session_state.get("turbo_ultra_pesos", {
            "peso_s6": 0.50,
            "peso_mc": 0.35,
            "peso_micro": 0.15,
        })

        peso_s6 = st.slider(
            "Peso S6",
            min_value=0.05,
            max_value=0.90,
            value=float(pesos_atual.get("peso_s6", 0.50)),
            step=0.05,
        )
        peso_mc = st.slider(
            "Peso Monte Carlo",
            min_value=0.05,
            max_value=0.90,
            value=float(pesos_atual.get("peso_mc", 0.35)),
            step=0.05,
        )
        peso_micro = st.slider(
            "Peso Micro-Leque",
            min_value=0.05,
            max_value=0.90,
            value=float(pesos_atual.get("peso_micro", 0.15)),
            step=0.05,
        )

        st.session_state["turbo_ultra_pesos"] = {
            "peso_s6": float(peso_s6),
            "peso_mc": float(peso_mc),
            "peso_micro": float(peso_micro),
        }

    # ------------------------------------------------------------
    # 🔬 Ajustes Avançados do Motor — Anti-Ruído / Turbulência
    # ------------------------------------------------------------
    with st.expander("🔬 Ajustes Avançados do Motor — Anti-Ruído / Turbulência"):
        avancado_atual = st.session_state.get("turbo_ultra_avancado", {
            "suavizacao_idx": 0.25,
            "profundidade_micro": 15,
            "fator_antirruido": 0.40,
            "elasticidade_nucleo": 0.20,
            "intensidade_turbulencia": 0.30,
        })

        suavizacao_idx = st.slider(
            "Suavização IDX (0 = bruto, 1 = ultra suave)",
            min_value=0.00,
            max_value=1.00,
            value=float(avancado_atual.get("suavizacao_idx", 0.25)),
            step=0.05,
        )
        profundidade_micro = st.slider(
            "Profundidade Micro-Leque",
            min_value=3,
            max_value=40,
            value=int(avancado_atual.get("profundidade_micro", 15)),
            step=1,
        )
        fator_antirruido = st.slider(
            "Fator Anti-Ruído (impacta S6/MC/Micro)",
            min_value=0.00,
            max_value=1.00,
            value=float(avancado_atual.get("fator_antirruido", 0.40)),
            step=0.05,
        )
        elasticidade_nucleo = st.slider(
            "Elasticidade do Núcleo (0 = rígido, 1 = elástico)",
            min_value=0.00,
            max_value=1.00,
            value=float(avancado_atual.get("elasticidade_nucleo", 0.20)),
            step=0.05,
        )
        intensidade_turbulencia = st.slider(
            "Intensidade Anti-Turbulência (reduz variações espúrias)",
            min_value=0.00,
            max_value=1.00,
            value=float(avancado_atual.get("intensidade_turbulencia", 0.30)),
            step=0.05,
        )

        st.session_state["turbo_ultra_avancado"] = {
            "suavizacao_idx": float(suavizacao_idx),
            "profundidade_micro": int(profundidade_micro),
            "fator_antirruido": float(fator_antirruido),
            "elasticidade_nucleo": float(elasticidade_nucleo),
            "intensidade_turbulencia": float(intensidade_turbulencia),
        }

    # ------------------------------------------------------------
    # ▶️ Execução do TURBO++ ULTRA (Anti-Zumbi)
    # ------------------------------------------------------------
    if st.button("Rodar TURBO++ ULTRA (V15.7 MAX)", type="primary"):
        limite_max = st.session_state.get("limite_max_janela", 600)
        if st.session_state.get("flag_modo_seguro", True):
            if not limitar_operacao(len(df), limite_max, "TURBO++ ULTRA"):
                return

        # Pesos manuais
        pesos = st.session_state.get("turbo_ultra_pesos", {})
        peso_s6 = float(pesos.get("peso_s6", 0.5))
        peso_mc = float(pesos.get("peso_mc", 0.35))
        peso_micro = float(pesos.get("peso_micro", 0.15))

        # Parâmetros avançados
        av = st.session_state.get("turbo_ultra_avancado", {})
        suavizacao_idx = float(av.get("suavizacao_idx", 0.25))
        profundidade_micro = int(av.get("profundidade_micro", 15))
        fator_antirruido = float(av.get("fator_antirruido", 0.40))
        elasticidade_nucleo = float(av.get("elasticidade_nucleo", 0.20))
        intensidade_turbulencia = float(av.get("intensidade_turbulencia", 0.30))

        # Chamada do motor V15.7 MAX
        r = turbo_ultra_v157(
            df,
            idx_alvo=int(idx_alvo),
            peso_s6=peso_s6,
            peso_mc=peso_mc,
            peso_micro=peso_micro,
            suavizacao_idx=suavizacao_idx,
            profundidade_micro=profundidade_micro,
            fator_antirruido=fator_antirruido,
            elasticidade_nucleo=elasticidade_nucleo,
            intensidade_turbulencia=intensidade_turbulencia,
        )
        st.session_state["turbo_ultra_result"] = r

    # ------------------------------------------------------------
    # 📊 Exibição dos resultados TURBO++ ULTRA
    # ------------------------------------------------------------
    r = st.session_state.get("turbo_ultra_result")
    if not r or "final" not in r:
        return

    st.markdown("### 🔥 PREVISÃO FINAL TURBO++ ULTRA (V15.7 MAX)")
    st.code(formatar_lista_passageiros(r["final"]), language="text")

    col1, col2, col3 = st.columns(3)

    # Núcleo S6
    with col1:
        st.subheader("Núcleo (S6 Profundo)")
        serie_s6 = r.get("serie_s6", [])
        if serie_s6:
            st.code(formatar_lista_passageiros(serie_s6), language="text")
        else:
            st.text("Núcleo S6 não disponível.")

    # Micro-Leque
    with col2:
        st.subheader("Micro-Leque Profundo")
        micro_leque = r.get("micro_leque", [])
        if micro_leque:
            for ml in micro_leque:
                st.text(formatar_lista_passageiros(ml))
        else:
            st.text("Micro-Leque não disponível.")

    # Monte Carlo
    with col3:
        st.subheader("Monte Carlo (amostras)")
        mc_ams = r.get("mc_amostras", [])
        if mc_ams:
            for mc in mc_ams:
                st.text(formatar_lista_passageiros(mc))
        else:
            st.text("Amostras de Monte Carlo não disponíveis.")

    # Divergência
    st.markdown("### 📉 Divergência S6 vs Monte Carlo")
    st.metric("Divergência média", round(float(r.get("divergencia_s6_mc", 0.0)), 4))

    # Pesos efetivos usados
    st.markdown("### ⚖️ Pesos e Parâmetros Utilizados (Motor V15.7 MAX)")
    pesos_dict = r.get("pesos", {})
    pesos_auto = r.get("pesos_auto", {})
    k_star_local = r.get("k_star_local", None)

    col_a, col_b = st.columns(2)
    with col_a:
        st.json(
            {
                "peso_s6_manual": pesos_dict.get("peso_s6"),
                "peso_mc_manual": pesos_dict.get("peso_mc"),
                "peso_micro_manual": pesos_dict.get("peso_micro"),
                "suavizacao_idx": pesos_dict.get("suavizacao_idx"),
                "profundidade_micro": pesos_dict.get("profundidade_micro"),
                "fator_antirruido": pesos_dict.get("fator_antirruido"),
                "elasticidade_nucleo": pesos_dict.get("elasticidade_nucleo"),
                "intensidade_turbulencia": pesos_dict.get("intensidade_turbulencia"),
            }
        )
    with col_b:
        st.json(
            {
                "peso_s6_auto_k*": pesos_auto.get("peso_s6"),
                "peso_mc_auto_k*": pesos_auto.get("peso_mc"),
                "peso_micro_auto_k*": pesos_auto.get("peso_micro"),
                "k_star_local(%)": k_star_local,
            }
        )

    st.caption(r.get("descricao", "Previsão TURBO++ ULTRA — V15.7 MAX"))

# ------------------------------------------------------------
# ALIASES DE COMPATIBILIDADE (NOMES V15.6)
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
    Alias para compatibilidade com versões anteriores (V15.6).
    Internamente usa o motor V15.7 MAX.
    """
    return turbo_ultra_v157(
        df,
        idx_alvo=idx_alvo,
        peso_s6=peso_s6,
        peso_mc=peso_mc,
        peso_micro=peso_micro,
        suavizacao_idx=suavizacao_idx,
        profundidade_micro=profundidade_micro,
        fator_antirruido=fator_antirruido,
        elasticidade_nucleo=elasticidade_nucleo,
        intensidade_turbulencia=intensidade_turbulencia,
    )


def painel_turbo_ultra_v156() -> None:
    """
    Alias para compatibilidade com versões anteriores (V15.6).
    Redireciona para o painel V15.7 MAX.
    """
    painel_turbo_ultra_v157()

# ------------------------------------------------------------
# FIM DA PARTE 4/6
# ------------------------------------------------------------
# ============================================================
# PARTE 5/6 — Monitor de Risco, Ruído Condicional, Confiabilidade REAL
# ============================================================

# ------------------------------------------------------------
# MONITOR DE RISCO (série k / k*) — FUNÇÕES (V15.7 MAX)
# ------------------------------------------------------------

def montar_tabela_monitor_risco_v157() -> Optional[pd.DataFrame]:
    """
    Tabela oficial do Monitor de Risco V15.7 MAX:
    - idx
    - k
    - k* (rolling)
    - estado_kstar (estavel / atencao / critico)
    
    Agora esta tabela é padronizada para alimentar:
    • Ruído Condicional Premium
    • Confiabilidade REAL
    • Relatório Final V15.7 MAX
    """
    df = st.session_state.get("df_historico")
    s_k = st.session_state.get("serie_k")
    s_kstar = st.session_state.get("serie_k_star")

    if df is None or df.empty or s_k is None or s_kstar is None:
        return None

    if "idx" not in df.columns:
        df = df.copy()
        df["idx"] = np.arange(1, len(df) + 1, dtype=int)
        st.session_state["df_historico"] = df

    tabela = pd.DataFrame({
        "idx": df["idx"].values,
        "k": s_k.values,
        "k_star_pct": s_kstar.values,
    })

    def classificar(v):
        if v < 15:
            return "estavel"
        elif v < 35:
            return "atencao"
        return "critico"

    tabela["estado_kstar"] = tabela["k_star_pct"].apply(classificar)
    st.session_state["monitor_risco_tabela"] = tabela
    return tabela


# ------------------------------------------------------------
# RUÍDO CONDICIONAL (NR%) — VERSÃO PREMIUM V15.7 MAX
# ------------------------------------------------------------

def calcular_mapa_ruido_global_v157(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Ruído Estrutural Global — versão premium:
    - usa mediana + MAD para robustez
    - evita distorção por outliers
    """
    cols_pass = extrair_passageiros(df)
    if not cols_pass:
        return {}

    sub = df[cols_pass].astype(float)

    medias = sub.mean()
    desvios = sub.std().replace(0, 1.0)
    mad = sub.mad().replace(0, 1.0)

    nr_pct = (mad / (medias.abs() + 1.0)) * 100.0

    mapa = {
        "medias": medias.to_dict(),
        "std": desvios.to_dict(),
        "mad": mad.to_dict(),
        "nr_pct": nr_pct.to_dict(),
    }

    nr_med = float(nr_pct.mean())
    nr_max = float(nr_pct.max())

    if nr_med < 25:
        resumo = "🟢 Ruído global baixo — regime premium."
    elif nr_med < 45:
        resumo = "🟡 Ruído global moderado — regime aceitável."
    else:
        resumo = "🔴 Ruído global alto — cautela máxima."

    st.session_state["nr_global_v157"] = mapa
    st.session_state["nr_global_resumo_v157"] = resumo
    return mapa


def calcular_mapa_ruido_condicional_v157(
    df: pd.DataFrame,
    tabela_risco: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Ruído Condicional Premium — V15.7:
    • Calculado separando regimes k*: estavel, atencao, critico
    • Usa MAD (robustez)
    • Feito para alimentar Relatório Final e Modo 6
    """
    cols_pass = extrair_passageiros(df)
    if not cols_pass:
        return {}

    base = df.copy()
    if "idx" not in base.columns:
        base["idx"] = np.arange(1, len(base) + 1)

    merged = base.merge(
        tabela_risco[["idx", "estado_kstar"]],
        on="idx",
        how="left"
    )

    resultados = {}

    for estado in ["estavel", "atencao", "critico"]:
        sub = merged[merged["estado_kstar"] == estado]
        if sub.empty:
            continue

        sub_pass = sub[cols_pass].astype(float)
        medias = sub_pass.mean()
        mad = sub_pass.mad().replace(0, 1.0)
        nr_pct = (mad / (medias.abs() + 1.0)) * 100.0

        resultados[estado] = {
            "medias": medias.to_dict(),
            "mad": mad.to_dict(),
            "nr_pct": nr_pct.to_dict(),
            "n_series": len(sub),
        }

    st.session_state["nr_condicional_v157"] = resultados
    return resultados


# ------------------------------------------------------------
# TESTES DE CONFIABILIDADE REAL — VERSÃO 15.7 MAX
# ------------------------------------------------------------

def rodar_backtest_turbo_ultra_v157(
    df: pd.DataFrame,
    n_prev: int = 50,
) -> BacktestResult:
    """
    Backtest REAL V15.7:
    - usa motor V15.7 MAX (novo)
    - integra ruído / divergência / k* diretamente
    - retorna tabela + estatísticas
    """
    if "idx" not in df.columns:
        df = df.copy()
        df["idx"] = np.arange(1, len(df) + 1)
        st.session_state["df_historico"] = df

    min_idx, max_idx = obter_intervalo_indices(df)
    candidatos = [i for i in range(min_idx + 1, max_idx + 1)]

    if not candidatos:
        return BacktestResult(pd.DataFrame(), "Sem índices suficientes para backtest.")

    candidatos = candidatos[-n_prev:]
    resultados = []

    cols_pass = extrair_passageiros(df)

    for idx in candidatos:
        res = turbo_ultra_v157(df, idx_alvo=idx)
        if not isinstance(res, dict) or "final" not in res:
            continue

        real_row = df[df["idx"] == idx]
        if real_row.empty:
            continue

        real = [int(real_row.iloc[0][c]) for c in cols_pass]
        prev = res["final"]

        ac_total = calcular_acerto_total(real, prev)
        ac_parcial = 1 if ac_total > 0 else 0

        resultados.append({
            "idx": idx,
            "acertos_totais": ac_total,
            "teve_acerto": ac_parcial,
            "div_s6_mc": res.get("divergencia_s6_mc", None),
            "k_star_local": res.get("k_star_local", None),
        })

    if not resultados:
        return BacktestResult(pd.DataFrame(), "Backtest sem resultados válidos.")

    df_res = pd.DataFrame(resultados)
    return BacktestResult(
        tabela=df_res,
        descricao="Backtest REAL utilizando o motor TURBO++ ULTRA V15.7 MAX",
    )


def sintetizar_confiabilidade_v157(df_res: pd.DataFrame) -> Dict[str, Any]:
    """
    Estatísticas oficiais V15.7 MAX:
    - probabilidade de ≥1 acerto
    - média de acertos
    - média divergência
    - k* médio
    """
    if df_res is None or df_res.empty:
        return {
            "p_ao_menos_um": 0.0,
            "media_acertos": 0.0,
            "media_div": None,
            "media_kstar": None,
            "n_prev": 0,
        }

    return {
        "p_ao_menos_um": float(df_res["teve_acerto"].mean()),
        "media_acertos": float(df_res["acertos_totais"].mean()),
        "media_div": float(df_res["div_s6_mc"].mean())
            if "div_s6_mc" in df_res else None,
        "media_kstar": float(df_res["k_star_local"].mean())
            if "k_star_local" in df_res else None,
        "n_prev": len(df_res),
    }


def gerar_texto_confiabilidade_v157(stats: Dict[str, Any], alvo: float) -> str:
    """
    Texto interpretativo — versão aprimorada:
    incorpora divergência e k* médio.
    """
    p = stats["p_ao_menos_um"]
    pct = round(p * 100.0, 1)
    alvo_pct = round(alvo * 100.0, 1)
    n_prev = stats["n_prev"]

    if n_prev == 0:
        return "Não foi possível calcular confiabilidade REAL (sem dados)."

    if p >= alvo:
        msg = (
            f"🟢 Confiabilidade forte: {pct}% ≥ alvo {alvo_pct}%. "
            "Regime propício para Modo TURBO++ e até 6 Acertos, "
            "desde que ruído e divergência estejam baixos."
        )
    elif p >= alvo * 0.8:
        msg = (
            f"🟡 Confiabilidade mediana: {pct}% próximo do alvo. "
            "Bom para exploração controlada. Evitar agressividade."
        )
    else:
        msg = (
            f"🔴 Confiabilidade fraca: {pct}% abaixo do desejado. "
            "Ajustar pesos / revisar ruído / reforçar calibração."
        )

    if stats.get("media_div") is not None:
        msg += f"\n• Divergência média S6–MC: {round(stats['media_div'],4)}"

    if stats.get("media_kstar") is not None:
        msg += f"\n• k* médio: {round(stats['media_kstar'],1)}%"

    return msg


# ------------------------------------------------------------
# PAINEL — 🚨 Monitor de Risco (V15.7 MAX)
# ------------------------------------------------------------

def painel_monitor_risco_v157() -> None:
    st.markdown("## 🚨 Monitor de Risco (k & k*) — V15.7 MAX")

    df = st.session_state.get("df_historico")
    if df is None or df.empty:
        st.warning("Carregue primeiro o histórico.")
        return

    tabela = montar_tabela_monitor_risco_v157()
    if tabela is None or tabela.empty:
        st.info("Rode o Pipeline FLEX ULTRA para atualizar k/k*.")
        return

    st.markdown("### Série k / k* — últimos 40 registros")
    st.dataframe(tabela.tail(40), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("k médio", round(float(tabela["k"].mean()), 4))
        st.metric("k máximo", int(tabela["k"].max()))
    with col2:
        st.metric("k* médio (%)", round(float(tabela["k_star_pct"].mean()), 1))
        st.metric("k* máximo (%)", round(float(tabela["k_star_pct"].max()), 1))

    dist = tabela["estado_kstar"].value_counts(normalize=True) * 100.0
    st.json({k: round(v, 1) for k, v in dist.to_dict().items()})

    st.caption("Monitor de sensibilidade dos guardas ao longo da estrada.")


# ------------------------------------------------------------
# PAINEL — 📊 Ruído Condicional Premium (V15.7 MAX)
# ------------------------------------------------------------

def painel_ruido_condicional_v157() -> None:
    st.markdown("## 📊 Ruído Condicional — V15.7 MAX")

    df = st.session_state.get("df_historico")
    if df is None or df.empty:
        st.warning("Carregue o histórico.")
        return

    tab_risco = montar_tabela_monitor_risco_v157()
    if tab_risco is None:
        st.info("Rode o Pipeline FLEX ULTRA para atualizar os sentinelas k/k*.")
        return

    st.markdown("### 🌐 Ruído Global (MAD / mediana)")
    mapa_global = calcular_mapa_ruido_global_v157(df)
    st.json(mapa_global.get("nr_pct", {}))
    st.markdown(st.session_state.get("nr_global_resumo_v157", ""))

    st.markdown("### 🧬 Ruído Condicional por regime k*")
    mapa_cond = calcular_mapa_ruido_condicional_v157(df, tab_risco)

    for estado, info in mapa_cond.items():
        st.markdown(f"#### Regime: **{estado}** (n={info['n_series']})")
        st.json(info.get("nr_pct", {}))

    st.caption("Ruído condicional premium identifica trechos premium e críticos.")


# ------------------------------------------------------------
# PAINEL — 🧪 Testes de Confiabilidade REAL — V15.7 MAX
# ------------------------------------------------------------

def painel_testes_confiabilidade_v157() -> None:
    st.markdown("## 🧪 Testes de Confiabilidade REAL — V15.7 MAX")

    df = st.session_state.get("df_historico")
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro.")
        return

    limite_prev = st.session_state.get("limite_max_prev", 300)

    n_prev = st.number_input(
        "Quantidade de previsões para backtest:",
        min_value=10,
        max_value=min(limite_prev, 1000),
        value=st.session_state.get("confiabilidade_n_prev", 50),
        step=10,
        key="n_prev_confiab_v157",
    )
    st.session_state["confiabilidade_n_prev"] = int(n_prev)

    alvo = st.slider(
        "Alvo de confiabilidade (prob. de ≥1 acerto):",
        min_value=0.40,
        max_value=0.90,
        value=float(st.session_state.get("confiabilidade_alvo", 0.65)),
        step=0.05,
        key="alvo_confiab_v157",
    )
    st.session_state["confiabilidade_alvo"] = float(alvo)

    if st.button("Rodar Confiabilidade REAL (V15.7)", type="primary"):
        if st.session_state.get("flag_modo_seguro", True):
            if not limitar_operacao(
                len(df),
                st.session_state.get("limite_max_janela", 600),
                "Confiabilidade REAL"
            ):
                return

        back = rodar_backtest_turbo_ultra_v157(df, n_prev=int(n_prev))
        st.session_state["confiabilidade_resultados_v157"] = back

    back = st.session_state.get("confiabilidade_resultados_v157")
    if back is None or back.tabela.empty:
        st.info("Nenhum resultado disponível ainda.")
        return

    st.markdown("### Tabela de Backtest (V15.7 MAX)")
    st.dataframe(back.tabela, use_container_width=True)

    stats = sintetizar_confiabilidade_v157(back.tabela)
    texto = gerar_texto_confiabilidade_v157(stats, alvo=float(alvo))

    col1, col2 = st.columns(2)
    with col1:
        st.metric("P(≥1 acerto)", f"{round(stats['p_ao_menos_um'] * 100.0, 1)}%")
        st.metric("Média de acertos totais", round(stats["media_acertos"], 3))
    with col2:
        if stats.get("media_div") is not None:
            st.metric("Divergência média S6–MC", round(stats["media_div"], 4))
        if stats.get("media_kstar") is not None:
            st.metric("k* médio (%)", round(stats["media_kstar"], 1))

        st.metric("Nº de previsões testadas", int(stats["n_prev"]))

    exibir_bloco_mensagem(
        "Leitura da Confiabilidade REAL — V15.7 MAX",
        texto_em_blocos(texto),
        emoji="📈",
    )

    st.caption(
        "Confiabilidade REAL V15.7 MAX inclui divergência, k*, NR% e motor reforçado."
    )

# ------------------------------------------------------------
# FIM DA PARTE 5/6
# ------------------------------------------------------------
# ============================================================
# PARTE 6/6 — Modo 6 Acertos + Relatório Final + main()
# ============================================================

# ------------------------------------------------------------
# AVALIAÇÃO DE PRONTIDÃO — MODO 6 ACERTOS (V15.7 MAX)
# ------------------------------------------------------------

def avaliar_prontidao_modo6_v157() -> Tuple[bool, str]:
    """
    Leitura integrada de prontidão para Modo 6 Acertos (V15.7 MAX):
    - Diagnóstico de impacto (regime + k*)
    - Confiabilidade REAL (V15.7)
    - Ruído Condicional Premium (NR%)
    - k* local na cauda da estrada
    """
    diag: DiagnosticoImpacto = st.session_state.get("diag_impacto")

    # Confiabilidade — prefere V15.7, cai para chave antiga se existir
    conf: BacktestResult = (
        st.session_state.get("confiabilidade_resultados_v157")
        or st.session_state.get("confiabilidade_resultados")
    )

    mapa_cond = (
        st.session_state.get("nr_condicional_v157")
        or st.session_state.get("nr_mapa_condicional")
    )
    serie_kstar = st.session_state.get("serie_k_star")

    if diag is None or conf is None or conf.tabela is None or conf.tabela.empty:
        return False, (
            "Ainda não há informações suficientes (diagnóstico estrutural + "
            "confiabilidade REAL) para avaliar a prontidão do Modo 6 Acertos.\n\n"
            "Checklist mínimo:\n"
            "1) Rode o Pipeline V14-FLEX ULTRA / IDX ULTRA.\n"
            "2) Rode o Monitor de Risco (k & k*).\n"
            "3) Rode os Testes de Confiabilidade REAL (V15.7 MAX)."
        )

    # Estatísticas de confiabilidade V15.7
    stats = sintetizar_confiabilidade_v157(conf.tabela)
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

    # k* local (cauda da estrada)
    kstar_local = None
    if serie_kstar is not None and len(serie_kstar) > 0:
        kstar_local = float(serie_kstar.iloc[-1])

    motivos = []

    # Regras duras
    if diag.regime_estado == "ruptura":
        motivos.append("Estrada em ruptura no diagnóstico de impacto.")
    if str(diag.risco_modo6).startswith("❌"):
        motivos.append("Diagnóstico estrutural marcou Modo 6 Acertos como NÃO recomendado.")
    if p < alvo * 0.8:
        motivos.append(
            f"Confiabilidade REAL baixa ({pct}% < 80% do alvo {alvo_pct}%)."
        )
    if nr_estavel is not None and nr_estavel > 45:
        motivos.append(
            f"Ruído condicional elevado em regime estável (NR% ≈ {round(nr_estavel, 1)})."
        )
    if kstar_local is not None and kstar_local >= 40:
        motivos.append(
            f"k* local muito alto ({round(kstar_local, 1)}%), indicando turbulência na cauda."
        )

    if motivos:
        texto = (
            "🔴 **Prontidão atual: NÃO RECOMENDADO utilizar Modo 6 Acertos em modo agressivo.**\n\n"
            "Motivos principais:\n"
            + "\n".join(f"- {m}" for m in motivos)
            + "\n\nChecklist sugerido antes de insistir no Modo 6:\n"
              "• Reforçar calibração do TURBO++ ULTRA (pesos / anti-ruído).\n"
              "• Buscar trechos premium (NR% menor, k* mais baixo).\n"
              "• Rodar novo ciclo de Confiabilidade REAL com parâmetros ajustados.\n"
        )
        st.session_state["modo6_risco_ok"] = False
        return False, texto

    texto = (
        "🟢 **Prontidão atual: AMBIENTE ELEGÍVEL para Modo 6 Acertos (decisão manual).**\n\n"
        f"- P(≥1 acerto) ≈ {pct}% (alvo {alvo_pct}%).\n"
        f"- Regime estrutural atual: {diag.regime_estado}.\n"
        f"- k* local: {round(kstar_local,1)}% (se disponível).\n"
        f"- NR% em regime estável: {round(nr_estavel,1)}% (se disponível).\n\n"
        "Modo recomendado:\n"
        "• Automático: usar núcleo TURBO++ + 1–2 coberturas em ambientes premium.\n"
        "• Avançado: ajustar pesos/NR% manualmente, olhando divergência S6–MC.\n"
        "• Super: combina leitura estrutural + Monte Carlo + mapa condicional para decisões cirúrgicas.\n"
    )
    st.session_state["modo6_risco_ok"] = True
    return True, texto


# ------------------------------------------------------------
# PAINEL — 🎯 Modo 6 Acertos — Execução (V15.7 MAX)
# ------------------------------------------------------------

def painel_modo_6_acertos_v157() -> None:
    st.markdown("## 🎯 Modo 6 Acertos — Execução (V15.7 MAX)")

    df = st.session_state.get("df_historico")
    if df is None or df.empty:
        st.warning(
            "Carregue o histórico e rode:\n"
            "1) Pipeline V14-FLEX ULTRA\n"
            "2) Monitor de Risco (k/k*)\n"
            "3) Confiabilidade REAL (V15.7 MAX)\n"
            "antes de usar este painel."
        )
        return

    # Prontidão integrada
    ok, texto_prontidao = avaliar_prontidao_modo6_v157()
    exibir_bloco_mensagem(
        "Prontidão para Modo 6 Acertos — Checklist Integrado",
        texto_em_blocos(texto_prontidao),
        emoji="🧪",
    )

    # Índice alvo conceitual: próxima série após a cauda
    if "idx" not in df.columns:
        df = df.copy()
        df["idx"] = np.arange(1, len(df) + 1, dtype=int)
        st.session_state["df_historico"] = df

    min_idx, max_idx = obter_intervalo_indices(df)
    idx_alvo = max_idx + 1

    st.markdown("### 🎯 Série alvo conceitual (próxima da estrada)")
    st.write(
        f"A próxima série conceitual é **C{idx_alvo}**, "
        f"tomando a estrada de C1 até C{max_idx} como base estrutural."
    )

    # Núcleo vindo do TURBO++ ULTRA V15.7
    turbo = st.session_state.get("turbo_ultra_result")
    if turbo is None or not turbo.get("final"):
        st.info(
            "Ainda não há uma previsão TURBO++ ULTRA registrada.\n\n"
            "Vá ao painel `🚀 Modo TURBO++ ULTRA Anti-Ruído (V15.7 MAX)`, "
            "rode o motor para o índice alvo desejado e depois volte aqui."
        )
        return

    serie_nucleo = turbo["final"]
    st.markdown("### 🔧 Núcleo proposto pelo TURBO++ ULTRA (V15.7 MAX)")
    st.code(formatar_lista_passageiros(serie_nucleo), language="text")

    # Coberturas a partir do núcleo (Micro-Leque Profundo)
    st.markdown("### 🛰️ Coberturas derivadas do núcleo (Micro-Leque Profundo)")
    leques = micro_leque_profundo_v156(serie_nucleo, intensidade=0.25)
    for i, lq in enumerate(leques, start=1):
        st.text(f"Cobertura {i}: {formatar_lista_passageiros(lq)}")

    st.markdown("### ⚖️ Decisão manual do operador (Modo Semi-Automático)")
    st.write(
        "O Modo 6 Acertos V15.7 MAX é **semi-automático**:\n"
        "• o sistema entrega núcleo + coberturas + prontidão;\n"
        "• a escolha final (quais listas usar, se vale a pena atuar, etc.) "
        "é sempre do operador."
    )

    escolha = st.multiselect(
        "Selecione quais listas considera elegíveis (núcleo/coberturas):",
        options=["Núcleo"] + [f"Cobertura {i}" for i in range(1, len(leques) + 1)],
        default=["Núcleo"],
        key="modo6_escolhas_v157",
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
    # Chave nova + chave antiga para compatibilidade
    st.session_state["modo6_resultados_v157"] = resumo_escolha
    st.session_state["modo6_resultados"] = resumo_escolha

    exibir_bloco_mensagem(
        "Resumo da seleção — Modo 6 Acertos V15.7 MAX",
        texto_em_blocos(
            "Núcleo selecionado: "
            f"{'Sim' if resumo_escolha['nucleo'] is not None else 'Não'}\n"
            f"Qtd. de coberturas selecionadas: {len(resumo_escolha['coberturas'])}\n"
            f"Prontidão indicada pelo sistema: {'OK' if ok else 'Não recomendado'}."
        ),
        emoji="🎯",
    )

    st.caption(
        "Importante: o Modo 6 Acertos V15.7 MAX não executa nenhuma ação no mundo real. "
        "A função é consolidar informação para decisões humanas, com checklist de risco completo."
    )


# ------------------------------------------------------------
# RELATÓRIO FINAL V15.7 MAX — CONSOLIDADO OFICIAL
# ------------------------------------------------------------

def montar_relatorio_final_v157() -> str:
    """
    Relatório Final V15.7 MAX:
    - Histórico / regime / k & k*
    - Ruído global + condicional premium
    - Confiabilidade REAL (V15.7)
    - Núcleo TURBO++ ULTRA V15.7
    - Modo 6 Acertos (seleção + prontidão)
    """
    df = st.session_state.get("df_historico")
    diag: DiagnosticoImpacto = st.session_state.get("diag_impacto")
    plano: PlanoCalibracao = st.session_state.get("plano_calibracao")

    tabela_risco = (
        st.session_state.get("monitor_risco_tabela")
        or montar_tabela_monitor_risco_v157()
    )

    mapa_global = (
        st.session_state.get("nr_global_v157")
        or st.session_state.get("nr_mapa_global")
    )
    mapa_cond = (
        st.session_state.get("nr_condicional_v157")
        or st.session_state.get("nr_mapa_condicional")
    )

    conf: BacktestResult = (
        st.session_state.get("confiabilidade_resultados_v157")
        or st.session_state.get("confiabilidade_resultados")
    )

    turbo = st.session_state.get("turbo_ultra_result")
    modo6 = (
        st.session_state.get("modo6_resultados_v157")
        or st.session_state.get("modo6_resultados")
    )

    linhas: List[str] = []

    # Cabeçalho
    linhas.append("============================================================")
    linhas.append(" RELATÓRIO FINAL — PREDICT CARS V15.7 MAX")
    linhas.append(" Núcleo + Coberturas + Interseção Estatística + Modo 6")
    linhas.append("============================================================")
    linhas.append("")

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
        linhas.append(f"- Índice alvo analisado (pipeline): C{diag.idx_alvo}")
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
    linhas.append("=== RUÍDO ESTRUTURAL / CONDICIONAL (NR% PREMIUM) ===")
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
                    f"- {estado} (n={info['n_series']}): "
                    f"NR% médio ≈ {round(np.mean(vals), 1)}%"
                )
        linhas.append("")
    else:
        linhas.append("Mapa de ruído condicional ainda não foi calculado.")
        linhas.append("")

    # Plano de calibração
    linhas.append("=== MINI PLANO DE CALIBRAÇÃO (ESTRUTURAL) ===")
    if plano is not None:
        linhas.append(f"Foco: {plano.foco}")
        for passo in plano.passos:
            linhas.append(f"- {passo}")
        linhas.append("")
    else:
        linhas.append("Plano de calibração ainda não foi gerado.")
        linhas.append("")

    # Confiabilidade REAL
    linhas.append("=== TESTES DE CONFIABILIDADE REAL (V15.7 MAX) ===")
    if conf is not None and conf.tabela is not None and not conf.tabela.empty:
        stats = sintetizar_confiabilidade_v157(conf.tabela)
        alvo = float(st.session_state.get("confiabilidade_alvo", 0.65))
        linhas.append(f"- Nº de previsões testadas: {stats['n_prev']}")
        linhas.append(
            f"- P(≥1 acerto): {round(stats['p_ao_menos_um'] * 100.0,1)}% "
            f"(alvo: {round(alvo*100.0,1)}%)"
        )
        linhas.append(
            f"- Média de acertos totais por previsão: {round(stats['media_acertos'],3)}"
        )
        if stats.get("media_div") is not None:
            linhas.append(
                f"- Divergência média S6–MC: {round(stats['media_div'],4)}"
            )
        if stats.get("media_kstar") is not None:
            linhas.append(
                f"- k* médio nas previsões: {round(stats['media_kstar'],1)}%"
            )
        linhas.append("")
        linhas.append("Leitura qualitativa da confiabilidade:")
        linhas.append(
            texto_em_blocos(gerar_texto_confiabilidade_v157(stats, alvo), largura=90)
        )
        linhas.append("")
    else:
        linhas.append("Ainda não foram rodados Testes de Confiabilidade REAL.")
        linhas.append("")

    # Núcleo TURBO++ ULTRA
    linhas.append("=== NÚCLEO TURBO++ ULTRA (V15.7 MAX) ===")
    if turbo is not None and turbo.get("final"):
        linhas.append("Previsão final TURBO++ ULTRA (núcleo proposto):")
        linhas.append(formatar_lista_passageiros(turbo["final"]))
        linhas.append("")
        if turbo.get("serie_s6"):
            linhas.append("Núcleo (S6 Profundo):")
            linhas.append(formatar_lista_passageiros(turbo["serie_s6"]))
            linhas.append("")
        if turbo.get("pesos"):
            linhas.append("Pesos usados (S6 / Monte Carlo / Micro-Leque + avançados):")
            linhas.append(str(turbo["pesos"]))
            if turbo.get("pesos_auto"):
                linhas.append("Pesos auto-ajustados por k* (guia interno):")
                linhas.append(str(turbo["pesos_auto"]))
        linhas.append("")
    else:
        linhas.append("Nenhuma previsão TURBO++ ULTRA registrada neste relatório.")
        linhas.append("")

    # Modo 6 Acertos
    linhas.append("=== MODO 6 ACERTOS — RESUMO OPERACIONAL ===")
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
        "versao": "V15.7 MAX",
    }
    return texto_final


def painel_relatorio_final_v157() -> None:
    st.markdown("## 📑 Relatório Final V15.7 MAX")

    texto = montar_relatorio_final_v157()
    st.text_area(
        "Relatório consolidado (copie e cole para análise externa):",
        value=texto,
        height=500,
        key="txt_relatorio_final_v157",
    )

    st.caption(
        "O Relatório Final V15.7 MAX integra histórico, regime, k/k*, ruído premium, "
        "confiabilidade REAL, núcleo TURBO++ ULTRA e Modo 6 Acertos em um único texto."
    )


# ------------------------------------------------------------
# FUNÇÃO PRINCIPAL — ROUTING GERAL DO APP V15.7 MAX
# ------------------------------------------------------------

def main():
    st.title("🚗 Predict Cars V15.7 MAX")

    # Usa a navegação oficial V15.7 (definida na PARTE 1/6)
    painel = construir_navegacao_v157()

    if painel == "📥 Histórico — Entrada FLEX ULTRA (V15.7 MAX)":
        painel_entrada_historico_flex_ultra_v157()

    elif painel == "🔍 Pipeline V14-FLEX ULTRA (V15.7 MAX)":
        painel_pipeline_v14_flex_ultra_v157()

    elif painel == "💡 Replay LIGHT (V15.7 MAX)":
        painel_replay_light_v157()

    elif painel == "📅 Replay ULTRA (V15.7 MAX)":
        painel_replay_ultra_v157()

    elif painel == "🎯 Replay ULTRA Unitário (V15.7 MAX)":
        painel_replay_unitario_v157()

    elif painel == "🚨 Monitor de Risco (k & k*) (V15.7 MAX)":
        painel_monitor_risco_v157()

    elif painel == "🧪 Testes de Confiabilidade REAL (V15.7 MAX)":
        painel_testes_confiabilidade_v157()

    elif painel == "📊 Ruído Condicional (NR%) (V15.7 MAX)":
        painel_ruido_condicional_v157()

    elif painel == "🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15.7 MAX)":
        painel_turbo_ultra_v157()

    elif painel == "🎯 Modo 6 Acertos — Execução (V15.7 MAX)":
        painel_modo_6_acertos_v157()

    elif painel == "📑 Relatório Final V15.7 MAX":
        painel_relatorio_final_v157()


# ------------------------------------------------------------
# ALIASES DE COMPATIBILIDADE (NOMES V15.6)
# ------------------------------------------------------------

# Monitor / Ruído / Confiabilidade
def montar_tabela_monitor_risco() -> Optional[pd.DataFrame]:
    return montar_tabela_monitor_risco_v157()


def calcular_mapa_ruido_global(df: pd.DataFrame) -> Dict[str, Any]:
    return calcular_mapa_ruido_global_v157(df)


def calcular_mapa_ruido_condicional(
    df: pd.DataFrame,
    tabela_risco: pd.DataFrame,
) -> Dict[str, Any]:
    return calcular_mapa_ruido_condicional_v157(df, tabela_risco)


def rodar_backtest_turbo_ultra_v156(
    df: pd.DataFrame,
    n_prev: int = 50,
) -> BacktestResult:
    return rodar_backtest_turbo_ultra_v157(df, n_prev=n_prev)


def sintetizar_confiabilidade(df_res: pd.DataFrame) -> Dict[str, Any]:
    return sintetizar_confiabilidade_v157(df_res)


def gerar_texto_confiabilidade(stats: Dict[str, Any], alvo: float) -> str:
    return gerar_texto_confiabilidade_v157(stats, alvo)


def painel_monitor_risco_v156() -> None:
    painel_monitor_risco_v157()


def painel_ruido_condicional_v156() -> None:
    painel_ruido_condicional_v157()


def painel_testes_confiabilidade_v156() -> None:
    painel_testes_confiabilidade_v157()


# Modo 6 / Relatório
def avaliar_prontidao_modo6_v156() -> Tuple[bool, str]:
    return avaliar_prontidao_modo6_v157()


def painel_modo_6_acertos_v156() -> None:
    painel_modo_6_acertos_v157()


def montar_relatorio_final_v156() -> str:
    return montar_relatorio_final_v157()


def painel_relatorio_final_v156() -> None:
    painel_relatorio_final_v157()


# TURBO++ aliases já foram definidos na PARTE 4/6:
# - turbo_ultra_v156 -> turbo_ultra_v157
# - painel_turbo_ultra_v156 -> painel_turbo_ultra_v157

# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":
    main()

# ------------------------------------------------------------
# FIM DA PARTE 6/6 — FIM DO APP V15.7 MAX
# ------------------------------------------------------------
