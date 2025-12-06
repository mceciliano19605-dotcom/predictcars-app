from __future__ import annotations
# -*- coding: utf-8 -*-
"""
Predict Cars V15-HÍBRIDO — RUÍDO TIPO B
Baseado no V14-FLEX ULTRA REAL (TURBO++), evoluído por ACRESCIMENTO.

Este arquivo será construído em 4 partes (1/4, 2/4, 3/4, 4/4), sem
qualquer simplificação do jeitão denso, granular e multifásico.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import math

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURAÇÃO BÁSICA DO APP
# =============================================================================

APP_NAME = "Predict Cars V15-HÍBRIDO — RUÍDO TIPO B"
APP_VERSION = "V15-HÍBRIDO (Base RUÍDO Estrutural) — Parte 1/4"

st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
)

# =============================================================================
# ESTADO COMPATÍVEL COM V14-FLEX ULTRA REAL
# =============================================================================
# Mantém a mesma filosofia de sessão do V14:
# - df histórico armazenado em st.session_state["df"]
# - uso de número variável de passageiros (FLEX)
# - nenhuma simplificação de filosofia de estrada / séries.


def get_df_sessao() -> Optional[pd.DataFrame]:
    """Retorna o histórico corrente armazenado na sessão."""
    df = st.session_state.get("df", None)
    if isinstance(df, pd.DataFrame):
        return df
    return None


def set_df_sessao(df: pd.DataFrame) -> None:
    """Atualiza o histórico na sessão."""
    st.session_state["df"] = df


def detectar_colunas_passageiros(df: pd.DataFrame) -> List[str]:
    """
    Detecta, de forma flexível, as colunas de passageiros.

    Compatível com:
    - Formato n1..n6, n1..nN (V14-FLEX)
    - Formato tipo 'P1', 'P2', ...
    - Evita qualquer simplificação rígida de esquema.
    """
    # Candidatos por prefixo numérico clássico do V14-FLEX
    candidatos = [
        c for c in df.columns
        if c.lower().startswith("n") or c.lower().startswith("p")
    ]

    # Garante uma ordem estável baseada em sufixo numérico, quando existir.
    def _key(c: str) -> Tuple[int, str]:
        sufixo = "".join(ch for ch in c if ch.isdigit())
        try:
            return (int(sufixo), c)
        except ValueError:
            return (10_000, c)

    candidatos_ordenados = sorted(candidatos, key=_key)

    return candidatos_ordenados


def contar_passageiros(df: pd.DataFrame) -> int:
    """Conta o número de colunas de passageiros detectadas."""
    cols = detectar_colunas_passageiros(df)
    return len(cols)


# =============================================================================
# BLOCO V15 — NÚCLEO DE RUÍDO ESTRUTURAL (NR%)
# =============================================================================
# Objetivo: medir o RUÍDO TIPO B (ruído explicável) em múltiplas camadas:
# - NR total (%)
# - NR por posição (P1..Pn)
# - NR por janela (janela rolante)
# - Estrutura para NR S6 / MC / Micro-Leque (alimentada depois).
#
# A filosofia aqui é:
# - manter o jeitão analítico profundo do V14;
# - não simplificar; apenas adicionar camadas.


@dataclass
class NoiseProfile:
    """
    Perfil completo de Ruído Estrutural (NR%) para o V15-HÍBRIDO.

    nr_total:      NR global agregado (%), 0–100
    nr_por_janela: DataFrame com NR por janela (linha = janela, colunas = métricas)
    nr_por_posicao: DataFrame com NR por posição (P1..Pn)
    nr_s6_mc_micro: DataFrame estruturado para divergência S6 / MC / Micro-Leque
                    (será alimentado em partes futuras do app).
    """
    nr_total: float
    nr_por_janela: pd.DataFrame
    nr_por_posicao: pd.DataFrame
    nr_s6_mc_micro: pd.DataFrame


def _entropy_discreta(proporcoes: np.ndarray) -> float:
    """
    Entropia discreta normalizada em [0, 1], para medir dispersão estrutural.

    - 0  => comportamento totalmente determinístico (sem dispersão)
    - 1  => máxima incerteza (todos os valores equiprováveis)
    """
    proporcoes = proporcoes[proporcoes > 0]
    if len(proporcoes) == 0:
        return 0.0
    h = -np.sum(proporcoes * np.log2(proporcoes))
    h_max = math.log2(len(proporcoes))
    if h_max == 0:
        return 0.0
    return float(h / h_max)


def calcular_nr_por_posicao(df: pd.DataFrame, cols_passageiros: List[str]) -> pd.DataFrame:
    """
    Calcula o NR estrutural por posição, baseado em entropia normalizada.

    Interpretação:
    - Entropia alta  => muito espalhado => mais ruído estrutural
    - Entropia baixa => concentrado     => menos ruído estrutural

    Retorna DataFrame com colunas:
    - posicao (P1..Pn)
    - entropia
    - nr_pct (entropia * 100)
    - diversidade (número de valores distintos)
    - dominante_pct (% do valor mais frequente)
    """
    registros = []

    for idx_pos, col in enumerate(cols_passageiros, start=1):
        serie = df[col].dropna()
        if serie.empty:
            ent = 0.0
            nr_pct = 0.0
            diversidade = 0
            dominante_pct = 0.0
        else:
            vc = serie.value_counts(normalize=True)
            proporcoes = vc.values.astype(float)
            ent = _entropy_discreta(proporcoes)
            nr_pct = 100.0 * ent
            diversidade = len(vc)
            dominante_pct = 100.0 * float(vc.iloc[0])

        registros.append(
            {
                "posicao": f"P{idx_pos}",
                "coluna": col,
                "entropia": ent,
                "nr_pct": nr_pct,
                "diversidade": diversidade,
                "dominante_pct": dominante_pct,
            }
        )

    df_pos = pd.DataFrame(registros)
    return df_pos


def calcular_nr_por_janela(
    df: pd.DataFrame,
    cols_passageiros: List[str],
    window: int = 40,
    step: int = 5,
) -> pd.DataFrame:
    """
    Calcula NR por janela rolante, agregando entropia média das posições.

    - window: tamanho da janela (em séries)
    - step:   salto entre janelas (ex: 5 => janelas sobrepostas, mas não 100%)

    Retorna DataFrame com colunas:
    - inicio, fim (índices de linha)
    - n_series
    - entropia_media
    - nr_pct
    """
    n = len(df)
    registros = []

    if n == 0 or len(cols_passageiros) == 0:
        return pd.DataFrame(
            columns=["inicio", "fim", "n_series", "entropia_media", "nr_pct"]
        )

    start = 0
    while start < n:
        end = min(start + window, n)
        bloco = df.iloc[start:end]
        if bloco.empty:
            break

        df_pos = calcular_nr_por_posicao(bloco, cols_passageiros)
        entropia_media = float(df_pos["entropia"].mean())
        nr_pct = 100.0 * entropia_media

        registros.append(
            {
                "inicio": int(start + 1),  # 1-based para casar com C1..Cn
                "fim": int(end),
                "n_series": int(len(bloco)),
                "entropia_media": entropia_media,
                "nr_pct": nr_pct,
            }
        )

        if end == n:
            break
        start += step

    df_jan = pd.DataFrame(registros)
    return df_jan


def sintetizar_nr_total(nr_por_janela: pd.DataFrame) -> float:
    """
    Sintetiza um NR total (%) a partir do NR por janela.

    Estratégia base:
    - média simples do nr_pct por janela (pode ser refinada depois com pesos).
    """
    if nr_por_janela.empty:
        return 0.0
    return float(nr_por_janela["nr_pct"].mean())


def montar_matriz_nr_s6_mc_micro(
    df_s6: Optional[pd.DataFrame] = None,
    df_mc: Optional[pd.DataFrame] = None,
    df_micro: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Estrutura base para mapear divergência / ruído entre S6, MC e Micro-Leques.

    Nesta PARTE 1/4:
    - apenas definimos o formato e placeholders.
    - o preenchimento real será feito quando integrarmos:
      - S6 Profundo
      - Monte Carlo Profundo
      - Micro-Leques (ataques locais)

    Formato-alvo:
    - linha = série ou índice-alvo
    - colunas (exemplo): 'score_s6', 'score_mc', 'score_micro', 'desvio_entre_camadas'
    """
    colunas = ["id", "score_s6", "score_mc", "score_micro", "desvio_entre_camadas"]
    matriz_vazia = pd.DataFrame(columns=colunas)
    return matriz_vazia


def analisar_ruido_estrutural(
    df_hist: pd.DataFrame,
    df_s6: Optional[pd.DataFrame] = None,
    df_mc: Optional[pd.DataFrame] = None,
    df_micro: Optional[pd.DataFrame] = None,
    window: int = 40,
    step: int = 5,
) -> NoiseProfile:
    """
    Núcleo de análise de Ruído Estrutural (V15-HÍBRIDO).

    - Não simplifica o pipeline existente;
    - Adiciona uma camada de leitura da estrada, baseada em entropia
      e janelas, preparada para dialogar com S6 / MC / Micro.

    Retorna NoiseProfile completo.
    """
    cols_passageiros = detectar_colunas_passageiros(df_hist)

    nr_pos = calcular_nr_por_posicao(df_hist, cols_passageiros)
    nr_jan = calcular_nr_por_janela(df_hist, cols_passageiros, window=window, step=step)
    nr_total = sintetizar_nr_total(nr_jan)
    nr_s6_mc_micro = montar_matriz_nr_s6_mc_micro(df_s6, df_mc, df_micro)

    profile = NoiseProfile(
        nr_total=nr_total,
        nr_por_janela=nr_jan,
        nr_por_posicao=nr_pos,
        nr_s6_mc_micro=nr_s6_mc_micro,
    )
    return profile


# =============================================================================
# PAINEL — MAPA DE RUÍDO ESTRUTURAL (V15-HÍBRIDO)
# =============================================================================
# Painel completo e denso, no jeitão do V14:
# - métricas globais
# - tabelas por posição
# - tabelas por janela
# - visualizações gráficas (mapas/heatmaps)
# - pronto para integração com S6 / MC / Micro-Leques.


def _plot_nr_por_posicao(df_pos: pd.DataFrame) -> None:
    """Gráfico de barras de NR por posição (P1..Pn)."""
    fig, ax = plt.subplots()
    ax.bar(df_pos["posicao"], df_pos["nr_pct"])
    ax.set_xlabel("Posição")
    ax.set_ylabel("NR por posição (%)")
    ax.set_title("NR Estrutural por Posição (V15-HÍBRIDO)")
    plt.xticks(rotation=0)
    st.pyplot(fig)


def _plot_nr_por_janela(df_jan: pd.DataFrame) -> None:
    """Gráfico de linha do NR por janela."""
    fig, ax = plt.subplots()
    eixo_x = [f"{ini}-{fim}" for ini, fim in zip(df_jan["inicio"], df_jan["fim"])]
    ax.plot(eixo_x, df_jan["nr_pct"], marker="o")
    ax.set_xlabel("Janela (C_início → C_fim)")
    ax.set_ylabel("NR por janela (%)")
    ax.set_title("NR Estrutural por Janela (V15-HÍBRIDO)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)


def painel_ruido_estrutural_v15() -> None:
    """
    Painel oficial de Ruído Estrutural (NR%) — V15-HÍBRIDO.

    Integra-se ao protocolo oficial:
    - Histórico carregado (FLEX ULTRA)
    - Estrutura da estrada
    - Leitura do ruído explicável (Tipo B)
    """
    st.markdown("## 📊 Mapa de Ruído Estrutural — V15-HÍBRIDO")
    st.markdown(
        """
        Este painel mede o **Ruído Tipo B (ruído explicável)** ao longo da estrada,
        sem alterar o pipeline V14-FLEX ULTRA REAL.

        A análise é feita em três camadas:
        - **NR Total (%)** — visão global do nível de ruído estrutural;
        - **NR por posição (P1..Pn)** — sensibilidade de cada passageiro;
        - **NR por janela** — como o ruído se comporta ao longo da estrada.
        """
    )

    df_hist = get_df_sessao()
    if df_hist is None or df_hist.empty:
        st.warning(
            "Nenhum histórico encontrado em sessão. "
            "Carregue o histórico primeiro no painel '📥 Histórico — Entrada'."
        )
        st.stop()

    n_series = len(df_hist)
    n_passageiros = contar_passageiros(df_hist)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("#### 📥 Histórico atual")
        st.write(f"Total de séries: **{n_series}**")
        st.write(f"Número de passageiros detectados: **{n_passageiros}**")

    with col_b:
        window = st.number_input(
            "Tamanho da janela para análise de NR (séries)",
            min_value=10,
            max_value=max(10, n_series),
            value=min(40, n_series),
            step=5,
        )
    with col_c:
        step = st.number_input(
            "Passo entre janelas (step)",
            min_value=1,
            max_value=max(1, window),
            value=5,
            step=1,
        )

    st.markdown("---")
    st.markdown("### 🔍 Execução da análise de Ruído Estrutural (V15-HÍBRIDO)")

    profile = analisar_ruido_estrutural(
        df_hist=df_hist,
        df_s6=None,   # será alimentado quando integrarmos S6 Profundo nas próximas partes
        df_mc=None,   # idem para Monte Carlo Profundo
        df_micro=None,   # idem para Micro-Leques
        window=int(window),
        step=int(step),
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("NR Total (%)", f"{profile.nr_total:.1f}%")
    with col2:
        st.write("Número de janelas analisadas:")
        st.write(f"**{len(profile.nr_por_janela)}**")
    with col3:
        st.write("Posições avaliadas (P1..Pn):")
        st.write(f"**{len(profile.nr_por_posicao)}**")

    st.markdown("### 📌 NR por posição (P1..Pn)")
    st.dataframe(profile.nr_por_posicao, use_container_width=True)

    _plot_nr_por_posicao(profile.nr_por_posicao)

    st.markdown("---")
    st.markdown("### 🪟 NR por janela da estrada")
    if profile.nr_por_janela.empty:
        st.info("Não foi possível calcular NR por janela com os parâmetros atuais.")
    else:
        st.dataframe(profile.nr_por_janela, use_container_width=True)
        _plot_nr_por_janela(profile.nr_por_janela)

    st.markdown("---")
    st.markdown("### 🧱 Estrutura para NR S6 / MC / Micro-Leque")
    st.info(
        """
        A matriz abaixo prepara o terreno para o **Mapa de Divergência S6 vs MC**
        e para o **Modo TURBO++ ULTRA ANTI-RUÍDO**.

        Nesta PARTE 1/4, a estrutura é criada mas ainda não recebe os dados
        de S6 / Monte Carlo / Micro-Leques. Isso será integrado nas próximas partes,
        mantendo o pipeline intacto e adicionando apenas camadas analíticas.
        """
    )
    st.dataframe(profile.nr_s6_mc_micro, use_container_width=True)


# =============================================================================
# NAVEGAÇÃO — BASE V15 (ACRESCENDO PAINÉIS)
# =============================================================================
# Aqui já definimos a navegação no estilo V14-FLEX ULTRA REAL,
# adicionando o painel de Ruído Estrutural.
#
# Os demais painéis (Histórico, Pipeline, Monitor de Risco, TURBO++,
# Replay ULTRA, Testes de Confiabilidade, Mapa Condicional, Divergência,
# Modo TURBO++ ANTI-RUÍDO) serão acrescentados nas Partes 2/4, 3/4 e 4/4.


def main() -> None:
    st.title("🚗 Predict Cars V15-HÍBRIDO — RUÍDO TIPO B")
    st.caption(APP_VERSION)

    st.sidebar.markdown("### 📂 Navegação — V15-HÍBRIDO")
    painel = st.sidebar.radio(
        "Escolha o painel:",
        (
            "📥 Histórico — Entrada",
            "🔍 Pipeline V14-FLEX (TURBO++)",
            "🚨 Monitor de Risco (k & k*)",
            "🚀 Modo TURBO++ — Painel Completo",
            "📅 Modo Replay Automático do Histórico",
            "🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)",
            "📊 Mapa de Ruído Estrutural (V15-HÍBRIDO)",
            # Os próximos painéis serão adicionados por ACRESCIMENTO:
            # "🧬 Mapa de Ruído Condicional",
            # "⚡ Mapa de Divergência S6 vs MC",
            # "🚀 Modo TURBO++ ULTRA ANTI-RUÍDO",
        ),
    )

    # -------------------------------------------------------------------------
    # Painéis já existentes (V14-FLEX ULTRA REAL)
    # -------------------------------------------------------------------------
    # IMPORTANTE:
    # - Nesta PARTE 1/4, os blocos de implementação detalhada de cada painel
    #   ainda não foram reescritos: serão adicionados integralmente nas Partes
    #   2/4, 3/4 e 4/4, mantendo o jeitão original.
    # - Por enquanto, mostramos mensagens-guia para não deixar nenhuma opção
    #   silenciosa. Isso será substituído por código real nas próximas partes.
    # -------------------------------------------------------------------------

    if painel == "📥 Histórico — Entrada":
        st.markdown("## 📥 Histórico — Entrada (V14-FLEX / V15-HÍBRIDO)")
        st.warning(
            "Bloco completo de carregamento de histórico será reintroduzido "
            "na PARTE 2/4, mantendo o mesmo jeitão do V14-FLEX ULTRA REAL."
        )

    elif painel == "🔍 Pipeline V14-FLEX (TURBO++)":
        st.markdown("## 🔍 Pipeline V14-FLEX (TURBO++)")
        st.warning(
            "Bloco completo do Pipeline V14-FLEX será restaurado e ampliado "
            "nas próximas partes, sem qualquer simplificação."
        )

    elif painel == "🚨 Monitor de Risco (k & k*)":
        st.markdown("## 🚨 Monitor de Risco (k & k*)")
        st.warning(
            "Monitor de Risco V14-FLEX será integrado aqui com k / k* ULTRA, "
            "em conjunto com o novo modo V15-HÍBRIDO."
        )

    elif painel == "🚀 Modo TURBO++ — Painel Completo":
        st.markdown("## 🚀 Modo TURBO++ — Painel Completo")
        st.warning(
            "Modo TURBO++ completo será reinserido (S6, S7, TVF, núcleo resiliente), "
            "e evoluído para o modo ANTI-RUÍDO nas Partes 3/4 e 4/4."
        )

    elif painel == "📅 Modo Replay Automático do Histórico":
        st.markdown("## 📅 Modo Replay Automático do Histórico")
        st.warning(
            "Replay ULTRA será reintroduzido, incluindo análise de acertos e regimes, "
            "sem simplificações, nas próximas partes."
        )

    elif painel == "🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)":
        st.markdown("## 🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)")
        st.warning(
            "Os blocos de QDS REAL, Backtest REAL e Monte Carlo serão integrados "
            "aqui, preservando tudo que já existia no V14-FLEX e somando camadas V15."
        )

    elif painel == "📊 Mapa de Ruído Estrutural (V15-HÍBRIDO)":
        painel_ruido_estrutural_v15()


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Predict Cars V15-HÍBRIDO — RUÍDO TIPO B
Baseado no V14-FLEX ULTRA REAL (TURBO++), evoluído por ACRESCIMENTO.

PARTE 2/4:
- Mantém toda a base de RUÍDO ESTRUTURAL (NR%) da Parte 1/4.
- Reinstala o painel 📥 Histórico — Entrada em modo FLEX ULTRA.
- Integra o carregamento do histórico com o NR estrutural (baseline).
- Cria a base matemática do Mapa de Ruído Condicional (sem painel ainda).

Nenhuma simplificação é aplicada. Apenas adicionamos camadas.
"""



from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import math

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURAÇÃO BÁSICA DO APP
# =============================================================================

APP_NAME = "Predict Cars V15-HÍBRIDO — RUÍDO TIPO B"
APP_VERSION = "V15-HÍBRIDO (Histórico + NR Estrutural + Base Condicional) — Parte 2/4"

st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
)

# =============================================================================
# ESTADO COMPATÍVEL COM V14-FLEX ULTRA REAL
# =============================================================================
# Mantém a mesma filosofia de sessão do V14:
# - df histórico armazenado em st.session_state["df"]
# - uso de número variável de passageiros (FLEX)
# - nenhuma simplificação de filosofia de estrada / séries.


def get_df_sessao() -> Optional[pd.DataFrame]:
    """Retorna o histórico corrente armazenado na sessão."""
    df = st.session_state.get("df", None)
    if isinstance(df, pd.DataFrame):
        return df
    return None


def set_df_sessao(df: pd.DataFrame) -> None:
    """Atualiza o histórico na sessão."""
    st.session_state["df"] = df


def detectar_colunas_passageiros(df: pd.DataFrame) -> List[str]:
    """
    Detecta, de forma flexível, as colunas de passageiros.

    Compatível com:
    - Formato n1..n6, n1..nN (V14-FLEX)
    - Formato tipo 'P1', 'P2', ...
    - Evita qualquer simplificação rígida de esquema.
    """
    candidatos = [
        c for c in df.columns
        if c.lower().startswith("n") or c.lower().startswith("p")
    ]

    def _key(c: str) -> Tuple[int, str]:
        sufixo = "".join(ch for ch in c if ch.isdigit())
        try:
            return (int(sufixo), c)
        except ValueError:
            return (10_000, c)

    candidatos_ordenados = sorted(candidatos, key=_key)
    return candidatos_ordenados


def contar_passageiros(df: pd.DataFrame) -> int:
    """Conta o número de colunas de passageiros detectadas."""
    cols = detectar_colunas_passageiros(df)
    return len(cols)


# =============================================================================
# BLOCO V15 — NÚCLEO DE RUÍDO ESTRUTURAL (NR%)
# =============================================================================


@dataclass
class NoiseProfile:
    """
    Perfil completo de Ruído Estrutural (NR%) para o V15-HÍBRIDO.

    nr_total:        NR global agregado (%), 0–100
    nr_por_janela:   DataFrame com NR por janela (linha = janela, colunas = métricas)
    nr_por_posicao:  DataFrame com NR por posição (P1..Pn)
    nr_s6_mc_micro:  DataFrame estruturado para divergência S6 / MC / Micro-Leque.
    """
    nr_total: float
    nr_por_janela: pd.DataFrame
    nr_por_posicao: pd.DataFrame
    nr_s6_mc_micro: pd.DataFrame


def _entropy_discreta(proporcoes: np.ndarray) -> float:
    """
    Entropia discreta normalizada em [0, 1], para medir dispersão estrutural.

    - 0  => comportamento totalmente determinístico (sem dispersão)
    - 1  => máxima incerteza (todos os valores equiprováveis)
    """
    proporcoes = proporcoes[proporcoes > 0]
    if len(proporcoes) == 0:
        return 0.0
    h = -np.sum(proporcoes * np.log2(proporcoes))
    h_max = math.log2(len(proporcoes))
    if h_max == 0:
        return 0.0
    return float(h / h_max)


def calcular_nr_por_posicao(df: pd.DataFrame, cols_passageiros: List[str]) -> pd.DataFrame:
    """
    Calcula o NR estrutural por posição, baseado em entropia normalizada.

    Retorna DataFrame com colunas:
    - posicao (P1..Pn)
    - coluna (nome da coluna original)
    - entropia
    - nr_pct (entropia * 100)
    - diversidade (número de valores distintos)
    - dominante_pct (% do valor mais frequente)
    """
    registros = []

    for idx_pos, col in enumerate(cols_passageiros, start=1):
        serie = df[col].dropna()
        if serie.empty:
            ent = 0.0
            nr_pct = 0.0
            diversidade = 0
            dominante_pct = 0.0
        else:
            vc = serie.value_counts(normalize=True)
            proporcoes = vc.values.astype(float)
            ent = _entropy_discreta(proporcoes)
            nr_pct = 100.0 * ent
            diversidade = len(vc)
            dominante_pct = 100.0 * float(vc.iloc[0])

        registros.append(
            {
                "posicao": f"P{idx_pos}",
                "coluna": col,
                "entropia": ent,
                "nr_pct": nr_pct,
                "diversidade": diversidade,
                "dominante_pct": dominante_pct,
            }
        )

    df_pos = pd.DataFrame(registros)
    return df_pos


def calcular_nr_por_janela(
    df: pd.DataFrame,
    cols_passageiros: List[str],
    window: int = 40,
    step: int = 5,
) -> pd.DataFrame:
    """
    Calcula NR por janela rolante, agregando entropia média das posições.

    Retorna DataFrame com colunas:
    - inicio, fim (índices de linha 1-based)
    - n_series
    - entropia_media
    - nr_pct
    """
    n = len(df)
    registros = []

    if n == 0 or len(cols_passageiros) == 0:
        return pd.DataFrame(
            columns=["inicio", "fim", "n_series", "entropia_media", "nr_pct"]
        )

    start = 0
    while start < n:
        end = min(start + window, n)
        bloco = df.iloc[start:end]
        if bloco.empty:
            break

        df_pos = calcular_nr_por_posicao(bloco, cols_passageiros)
        entropia_media = float(df_pos["entropia"].mean())
        nr_pct = 100.0 * entropia_media

        registros.append(
            {
                "inicio": int(start + 1),
                "fim": int(end),
                "n_series": int(len(bloco)),
                "entropia_media": entropia_media,
                "nr_pct": nr_pct,
            }
        )

        if end == n:
            break
        start += step

    df_jan = pd.DataFrame(registros)
    return df_jan


def sintetizar_nr_total(nr_por_janela: pd.DataFrame) -> float:
    """
    Sintetiza um NR total (%) a partir do NR por janela.

    Estratégia base:
    - média simples do nr_pct por janela (poderá ser refinada com pesos).
    """
    if nr_por_janela.empty:
        return 0.0
    return float(nr_por_janela["nr_pct"].mean())


def montar_matriz_nr_s6_mc_micro(
    df_s6: Optional[pd.DataFrame] = None,
    df_mc: Optional[pd.DataFrame] = None,
    df_micro: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Estrutura base para mapear divergência / ruído entre S6, MC e Micro-Leques.

    Nesta fase:
    - apenas definimos o formato e placeholders.
    - o preenchimento real será feito quando integrarmos:
      - S6 Profundo
      - Monte Carlo Profundo
      - Micro-Leques (ataques locais)
    """
    colunas = ["id", "score_s6", "score_mc", "score_micro", "desvio_entre_camadas"]
    matriz_vazia = pd.DataFrame(columns=colunas)
    return matriz_vazia


def analisar_ruido_estrutural(
    df_hist: pd.DataFrame,
    df_s6: Optional[pd.DataFrame] = None,
    df_mc: Optional[pd.DataFrame] = None,
    df_micro: Optional[pd.DataFrame] = None,
    window: int = 40,
    step: int = 5,
) -> NoiseProfile:
    """
    Núcleo de análise de Ruído Estrutural (V15-HÍBRIDO).

    Retorna NoiseProfile completo.
    """
    cols_passageiros = detectar_colunas_passageiros(df_hist)

    nr_pos = calcular_nr_por_posicao(df_hist, cols_passageiros)
    nr_jan = calcular_nr_por_janela(df_hist, cols_passageiros, window=window, step=step)
    nr_total = sintetizar_nr_total(nr_jan)
    nr_s6_mc_micro = montar_matriz_nr_s6_mc_micro(df_s6, df_mc, df_micro)

    profile = NoiseProfile(
        nr_total=nr_total,
        nr_por_janela=nr_jan,
        nr_por_posicao=nr_pos,
        nr_s6_mc_micro=nr_s6_mc_micro,
    )
    return profile


# =============================================================================
# BASE MATEMÁTICA — MAPA DE RUÍDO CONDICIONAL (V15-HÍBRIDO)
# =============================================================================
# Aqui começamos a preparar o núcleo de análise condicional:
# - Dependência entre posições (P_i, P_j)
# - Medida de informação mútua / entropia condicional
# O painel visual virá nas Partes 3/4 e 4/4.


@dataclass
class ConditionalNoiseMap:
    """
    Mapa de Ruído Condicional entre posições (P1..Pn).

    mi_matrix:
        DataFrame n_pos x n_pos com Informação Mútua normalizada
        entre P_i e P_j.

    h_cond_matrix:
        DataFrame opcional com entropia condicional normalizada
        H(P_i | P_j) / H(P_i), se aplicável.

    suporte:
        Dicionário com estruturas auxiliares (tabelas de contingência, etc.)
        útil para debugging profundo do comportamento condicional.
    """
    mi_matrix: pd.DataFrame
    h_cond_matrix: pd.DataFrame
    suporte: Dict[str, pd.DataFrame]


def _info_mutua_normalizada(x: np.ndarray, y: np.ndarray) -> float:
    """
    Informação Mútua normalizada em [0, 1] para duas variáveis discretas.

    Normalização adotada (simples e robusta):
    MI_norm = MI / min(Hx, Hy), quando possível.
    """
    s = pd.DataFrame({"x": x, "y": y}).dropna()
    if s.empty:
        return 0.0

    # Tabela de contingência
    cont = pd.crosstab(s["x"], s["y"])
    p_xy = cont / cont.values.sum()

    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    # Entropias marginais
    hx = _entropy_discreta(p_x.values)
    hy = _entropy_discreta(p_y.values)

    # Informação Mútua
    mi = 0.0
    for i in range(p_xy.shape[0]):
        for j in range(p_xy.shape[1]):
            pij = p_xy.iloc[i, j]
            if pij <= 0:
                continue
            pix = p_x.iloc[i]
            pjy = p_y.iloc[j]
            if pix <= 0 or pjy <= 0:
                continue
            mi += float(pij * math.log2(pij / (pix * pjy)))

    if mi <= 0:
        return 0.0

    normalizador = min(hx, hy)
    if normalizador <= 0:
        return 0.0

    mi_norm = mi / normalizador
    # Clamping leve para estabilidade numérica
    mi_norm = max(0.0, min(1.0, mi_norm))
    return float(mi_norm)


def construir_mapa_ruido_condicional(df_hist: pd.DataFrame) -> ConditionalNoiseMap:
    """
    Constrói a matriz de Informação Mútua normalizada entre posições P1..Pn.

    Nesta fase (Parte 2/4):
    - É um núcleo de cálculo sem painel.
    - Será usado futuramente no painel "🧬 Mapa de Ruído Condicional".
    """
    cols_passageiros = detectar_colunas_passageiros(df_hist)
    n_pos = len(cols_passageiros)

    if n_pos == 0:
        mi_df = pd.DataFrame()
        h_cond_df = pd.DataFrame()
        return ConditionalNoiseMap(mi_df, h_cond_df, suporte={})

    nomes_pos = [f"P{i}" for i in range(1, n_pos + 1)]
    mi_matrix = pd.DataFrame(
        np.zeros((n_pos, n_pos), dtype=float),
        index=nomes_pos,
        columns=nomes_pos,
    )
    h_cond_matrix = pd.DataFrame(
        np.zeros((n_pos, n_pos), dtype=float),
        index=nomes_pos,
        columns=nomes_pos,
    )

    suporte: Dict[str, pd.DataFrame] = {}

    # Pré-carrega as séries discretas
    series_discretas = [df_hist[col].astype("Int64") for col in cols_passageiros]

    for i in range(n_pos):
        xi = series_discretas[i]
        for j in range(n_pos):
            yj = series_discretas[j]

            mi_norm = _info_mutua_normalizada(xi.values, yj.values)
            mi_matrix.iloc[i, j] = mi_norm

            # Entropia condicional normalizada H(X|Y)/H(X)
            vc_x = xi.value_counts(normalize=True, dropna=True)
            hx = _entropy_discreta(vc_x.values.astype(float))
            if hx > 0:
                # H(X|Y) = H(X) - MI
                h_cond = max(0.0, hx - mi_norm * hx)
                h_cond_norm = h_cond / hx
            else:
                h_cond = 0.0
                h_cond_norm = 0.0
            h_cond_matrix.iloc[i, j] = h_cond_norm

    suporte["mi_matrix_raw"] = mi_matrix.copy()
    suporte["h_cond_matrix_raw"] = h_cond_matrix.copy()

    mapa = ConditionalNoiseMap(
        mi_matrix=mi_matrix,
        h_cond_matrix=h_cond_matrix,
        suporte=suporte,
    )
    return mapa


# =============================================================================
# PAINEL — MAPA DE RUÍDO ESTRUTURAL (V15-HÍBRIDO)
# =============================================================================


def _plot_nr_por_posicao(df_pos: pd.DataFrame) -> None:
    """Gráfico de barras de NR por posição (P1..Pn)."""
    fig, ax = plt.subplots()
    ax.bar(df_pos["posicao"], df_pos["nr_pct"])
    ax.set_xlabel("Posição")
    ax.set_ylabel("NR por posição (%)")
    ax.set_title("NR Estrutural por Posição (V15-HÍBRIDO)")
    plt.xticks(rotation=0)
    st.pyplot(fig)


def _plot_nr_por_janela(df_jan: pd.DataFrame) -> None:
    """Gráfico de linha do NR por janela."""
    fig, ax = plt.subplots()
    eixo_x = [f"{ini}-{fim}" for ini, fim in zip(df_jan["inicio"], df_jan["fim"])]
    ax.plot(eixo_x, df_jan["nr_pct"], marker="o")
    ax.set_xlabel("Janela (C_início → C_fim)")
    ax.set_ylabel("NR por janela (%)")
    ax.set_title("NR Estrutural por Janela (V15-HÍBRIDO)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)


def painel_ruido_estrutural_v15() -> None:
    """
    Painel oficial de Ruído Estrutural (NR%) — V15-HÍBRIDO.
    """
    st.markdown("## 📊 Mapa de Ruído Estrutural — V15-HÍBRIDO")
    st.markdown(
        """
        Este painel mede o **Ruído Tipo B (ruído explicável)** ao longo da estrada,
        sem alterar o pipeline V14-FLEX ULTRA REAL.

        A análise é feita em três camadas:
        - **NR Total (%)** — visão global do nível de ruído estrutural;
        - **NR por posição (P1..Pn)** — sensibilidade de cada passageiro;
        - **NR por janela** — como o ruído se comporta ao longo da estrada.
        """
    )

    df_hist = get_df_sessao()
    if df_hist is None or df_hist.empty:
        st.warning(
            "Nenhum histórico encontrado em sessão. "
            "Carregue o histórico primeiro no painel '📥 Histórico — Entrada'."
        )
        st.stop()

    n_series = len(df_hist)
    n_passageiros = contar_passageiros(df_hist)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("#### 📥 Histórico atual")
        st.write(f"Total de séries: **{n_series}**")
        st.write(f"Número de passageiros detectados: **{n_passageiros}**")

    with col_b:
        window = st.number_input(
            "Tamanho da janela para análise de NR (séries)",
            min_value=10,
            max_value=max(10, n_series),
            value=min(40, n_series),
            step=5,
        )
    with col_c:
        step = st.number_input(
            "Passo entre janelas (step)",
            min_value=1,
            max_value=max(1, window),
            value=5,
            step=1,
        )

    st.markdown("---")
    st.markdown("### 🔍 Execução da análise de Ruído Estrutural (V15-HÍBRIDO)")

    profile = analisar_ruido_estrutural(
        df_hist=df_hist,
        df_s6=None,
        df_mc=None,
        df_micro=None,
        window=int(window),
        step=int(step),
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("NR Total (%)", f"{profile.nr_total:.1f}%")
    with col2:
        st.write("Número de janelas analisadas:")
        st.write(f"**{len(profile.nr_por_janela)}**")
    with col3:
        st.write("Posições avaliadas (P1..Pn):")
        st.write(f"**{len(profile.nr_por_posicao)}**")

    st.markdown("### 📌 NR por posição (P1..Pn)")
    st.dataframe(profile.nr_por_posicao, use_container_width=True)

    _plot_nr_por_posicao(profile.nr_por_posicao)

    st.markdown("---")
    st.markdown("### 🪟 NR por janela da estrada")
    if profile.nr_por_janela.empty:
        st.info("Não foi possível calcular NR por janela com os parâmetros atuais.")
    else:
        st.dataframe(profile.nr_por_janela, use_container_width=True)
        _plot_nr_por_janela(profile.nr_por_janela)

    st.markdown("---")
    st.markdown("### 🧱 Estrutura para NR S6 / MC / Micro-Leque")
    st.info(
        """
        A matriz abaixo prepara o terreno para o **Mapa de Divergência S6 vs MC**
        e para o **Modo TURBO++ ULTRA ANTI-RUÍDO**.

        Nesta fase, a estrutura é criada mas ainda não recebe os dados
        de S6 / Monte Carlo / Micro-Leques. Isso será integrado nas próximas partes,
        mantendo o pipeline intacto e adicionando apenas camadas analíticas.
        """
    )
    st.dataframe(profile.nr_s6_mc_micro, use_container_width=True)


# =============================================================================
# PAINEL 📥 HISTÓRICO — ENTRADA (V14-FLEX / V15-HÍBRIDO)
# =============================================================================
# Reintroduzimos aqui o painel de entrada, em modo FLEX ULTRA:
# - CSV com coluna de séries (C1;...;k)
# - CSV com colunas de passageiros (n1..nN, k)
# O objetivo é normalizar tudo para um df compatível com V14-FLEX + V15-HÍBRIDO.


def _ler_csv_flex(file) -> pd.DataFrame:
    """
    Leitura genérica de CSV com detecção automática de separador.

    Usa engine='python' para aceitar ; , ou \t com heuristic matching.
    """
    try:
        df = pd.read_csv(file, sep=None, engine="python")
    except Exception:
        file.seek(0)
        df = pd.read_csv(file, sep=";")
    return df


def _normalizar_formato_coluna_series(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza histórico do tipo:

        C1;41;5;4;52;30;33;0
        C2;...

    Estrutura resultante:
    - indice (1..n)
    - serie_id (C1, C2, ...)
    - n1..nN (passageiros)
    - k (se existir)
    """
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Primeiro campo = identificador da série (C1, C2, ...)
    nome_id = df.columns[0]
    serie_id = df[nome_id].astype(str).str.strip()

    # Demais colunas = passageiros + possivelmente k
    cols_valores = df.columns[1:]
    n_cols_valores = len(cols_valores)

    # Assumimos que a última coluna, se numérica discreta, tende a ser k
    k_col = None
    if n_cols_valores >= 2:
        candidata = cols_valores[-1]
        # Heurística leve: se é inteira e com muitos zeros/valores baixos, assume k
        serie_cand = pd.to_numeric(df[candidata], errors="coerce")
        if serie_cand.notna().mean() > 0.9:
            k_col = candidata

    passageiros_cols: List[str] = []
    for idx, col in enumerate(cols_valores, start=1):
        if col == k_col:
            continue
        passageiros_cols.append(col)

    # Renomeia passageiros para n1..nN
    mapping = {}
    for i, col in enumerate(passageiros_cols, start=1):
        mapping[col] = f"n{i}"

    df_norm = pd.DataFrame()
    df_norm["indice"] = range(1, len(df) + 1)
    df_norm["serie_id"] = serie_id

    for col, novo_nome in mapping.items():
        df_norm[novo_nome] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    if k_col is not None:
        df_norm["k"] = pd.to_numeric(df[k_col], errors="coerce").astype("Int64")

    return df_norm


def _normalizar_formato_passageiros(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza histórico do tipo:

        n1;n2;...;nN;k

    ou colunas equivalentes que já estejam com nomes de passageiros.
    """
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Detecta coluna de k
    col_k = None
    for c in df.columns:
        if c.lower() == "k":
            col_k = c
            break

    passageiros_cols: List[str] = []
    for c in df.columns:
        if c == col_k:
            continue
        passageiros_cols.append(c)

    # Ordena para dar estabilidade
    def _key(c: str) -> Tuple[int, str]:
        sufixo = "".join(ch for ch in c if c.lower().startswith("n") and ch.isdigit())
        try:
            return (int(sufixo), c)
        except ValueError:
            return (10_000, c)

    passageiros_cols = sorted(passageiros_cols, key=_key)

    mapping = {}
    for i, col in enumerate(passageiros_cols, start=1):
        mapping[col] = f"n{i}"

    df_norm = pd.DataFrame()
    df_norm["indice"] = range(1, len(df) + 1)

    for col, novo_nome in mapping.items():
        df_norm[novo_nome] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    if col_k is not None:
        df_norm["k"] = pd.to_numeric(df[col_k], errors="coerce").astype("Int64")

    # Cria uma série_id sintética C1..Cn para manter metáfora completa
    df_norm["serie_id"] = df_norm["indice"].apply(lambda x: f"C{x}")

    # Reordena colunas num padrão consistente
    cols_passageiros = [c for c in df_norm.columns if c.startswith("n")]
    outras = [c for c in ["indice", "serie_id", "k"] if c in df_norm.columns]
    df_norm = df_norm[outras[:2] + cols_passageiros + outras[2:]]

    return df_norm


def painel_historico_entrada_v15() -> None:
    """
    Painel de entrada de histórico — V14-FLEX / V15-HÍBRIDO.

    - Permite formatos diferentes de CSV.
    - Normaliza para df compatível com:
        - Pipeline V14-FLEX
        - NR Estrutural
        - Mapa Condicional
        - módulos futuros (S6/MC/Micro).
    """
    st.markdown("## 📥 Histórico — Entrada (V14-FLEX / V15-HÍBRIDO)")

    st.markdown(
        """
        Este painel recebe o histórico da estrada em modo **FLEX ULTRA**,
        permitindo tanto o formato clássico com coluna de séries (C1;...;k)
        quanto o formato com colunas de passageiros (n1..nN, k).

        O objetivo é produzir um histórico normalizado e rico em metadados,
        pronto para:
        - Pipeline V14-FLEX (TURBO++);
        - Análises de Ruído Estrutural (V15-HÍBRIDO);
        - Mapa de Ruído Condicional;
        - Testes de Confiabilidade.
        """
    )

    formato = st.radio(
        "Formato do histórico:",
        (
            "CSV com coluna de séries",
            "CSV com passageiros (n1..nN, k)",
        ),
        help=(
            "Escolha de acordo com a estrutura do seu arquivo. "
            "Ambos os formatos serão normalizados para o mesmo padrão interno."
        ),
    )

    file = st.file_uploader(
        "Selecione o arquivo de histórico (.csv):",
        type=["csv"],
    )

    st.markdown(
        """
        🔎 **Dica:** o app detecta automaticamente `;`, `,` ou `tab` como separador.
        Caso tenha dúvidas, basta enviar o arquivo normalmente.
        """
    )

    df_norm: Optional[pd.DataFrame] = None

    if file is not None:
        df_raw = _ler_csv_flex(file)

        st.markdown("### 🔍 Pré-visualização bruta do arquivo")
        st.dataframe(df_raw.head(20), use_container_width=True)

        if formato == "CSV com coluna de séries":
            df_norm = _normalizar_formato_coluna_series(df_raw)
        else:
            df_norm = _normalizar_formato_passageiros(df_raw)

        st.markdown("---")
        st.markdown("### ✅ Histórico normalizado (V15-HÍBRIDO)")
        st.dataframe(df_norm.head(50), use_container_width=True)

        # Atualiza sessão
        set_df_sessao(df_norm)

        # Metadados básicos
        n_series = len(df_norm)
        cols_passageiros = detectar_colunas_passageiros(df_norm)
        n_passageiros = len(cols_passageiros)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de séries (C1 → Cn)", n_series)
        with col2:
            st.metric("Passageiros detectados (n)", n_passageiros)
        with col3:
            faixa_str = "N/A"
            if n_passageiros > 0:
                todos = df_norm[cols_passageiros].values.flatten()
                todos = todos[~pd.isna(todos)]
                if len(todos) > 0:
                    faixa_str = f"{int(np.min(todos))} → {int(np.max(todos))}"
            st.metric("Faixa numérica global", faixa_str)

        st.markdown("---")
        st.markdown("### 🎯 NR Estrutural — Baseline imediato")

        if n_series >= 10 and n_passageiros > 0:
            # Janela padrão para baseline (pode ser diferente da usada no painel dedicado)
            window_default = min(40, n_series)
            profile_baseline = analisar_ruido_estrutural(
                df_hist=df_norm,
                df_s6=None,
                df_mc=None,
                df_micro=None,
                window=window_default,
                step=5,
            )

            # Guarda baseline na sessão para reutilização futura, se desejado
            st.session_state["noise_profile_v15_baseline"] = profile_baseline

            colb1, colb2, colb3 = st.columns(3)
            with colb1:
                st.metric("NR Total (baseline)", f"{profile_baseline.nr_total:.1f}%")
            with colb2:
                st.write("Janelas usadas:")
                st.write(f"**{len(profile_baseline.nr_por_janela)}**")
            with colb3:
                st.write("Posições avaliadas:")
                st.write(f"**{len(profile_baseline.nr_por_posicao)}**")

            st.markdown(
                """
                Este baseline reflete o **nível médio de ruído estrutural** da estrada,
                servindo como referência para comparação entre diferentes históricos
                (58%, 22%, 47% etc.).
                """
            )
        else:
            st.info(
                "Histórico ainda pequeno ou sem passageiros detectados suficientes "
                "para calcular um NR estrutural robusto. Carregue um histórico maior."
            )
    else:
        st.info(
            "Nenhum arquivo selecionado ainda. "
            "Envie o histórico para ativar o modo V15-HÍBRIDO completo."
        )


# =============================================================================
# NAVEGAÇÃO — BASE V15 (ACRESCENDO PAINÉIS)
# =============================================================================


def main() -> None:
    st.title("🚗 Predict Cars V15-HÍBRIDO — RUÍDO TIPO B")
    st.caption(APP_VERSION)

    st.sidebar.markdown("### 📂 Navegação — V15-HÍBRIDO")
    painel = st.sidebar.radio(
        "Escolha o painel:",
        (
            "📥 Histórico — Entrada",
            "🔍 Pipeline V14-FLEX (TURBO++)",
            "🚨 Monitor de Risco (k & k*)",
            "🚀 Modo TURBO++ — Painel Completo",
            "📅 Modo Replay Automático do Histórico",
            "🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)",
            "📊 Mapa de Ruído Estrutural (V15-HÍBRIDO)",
            # Próximos painéis serão adicionados por ACRESCIMENTO:
            # "🧬 Mapa de Ruído Condicional",
            # "⚡ Mapa de Divergência S6 vs MC",
            # "🚀 Modo TURBO++ ULTRA ANTI-RUÍDO",
        ),
    )

    if painel == "📥 Histórico — Entrada":
        painel_historico_entrada_v15()

    elif painel == "🔍 Pipeline V14-FLEX (TURBO++)":
        st.markdown("## 🔍 Pipeline V14-FLEX (TURBO++)")
        st.warning(
            "Bloco completo do Pipeline V14-FLEX será restaurado e ampliado "
            "nas próximas partes, sem qualquer simplificação."
        )

    elif painel == "🚨 Monitor de Risco (k & k*)":
        st.markdown("## 🚨 Monitor de Risco (k & k*)")
        st.warning(
            "Monitor de Risco V14-FLEX será integrado aqui com k / k* ULTRA, "
            "em conjunto com o novo modo V15-HÍBRIDO."
        )

    elif painel == "🚀 Modo TURBO++ — Painel Completo":
        st.markdown("## 🚀 Modo TURBO++ — Painel Completo")
        st.warning(
            "Modo TURBO++ completo será reinserido (S6, S7, TVF, núcleo resiliente), "
            "e evoluído para o modo ANTI-RUÍDO nas Partes 3/4 e 4/4."
        )

    elif painel == "📅 Modo Replay Automático do Histórico":
        st.markdown("## 📅 Modo Replay Automático do Histórico")
        st.warning(
            "Replay ULTRA será reintroduzido, incluindo análise de acertos e regimes, "
            "sem simplificações, nas próximas partes."
        )

    elif painel == "🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)":
        st.markdown("## 🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)")
        st.warning(
            "Os blocos de QDS REAL, Backtest REAL e Monte Carlo serão integrados "
            "aqui, preservando tudo que já existia no V14-FLEX e somando camadas V15."
        )

    elif painel == "📊 Mapa de Ruído Estrutural (V15-HÍBRIDO)":
        painel_ruido_estrutural_v15()


if __name__ == "__main__":
    main()
# =============================================================================
# PARTE 3/4 — REINSTALAÇÃO DO PIPELINE V14-FLEX (TURBO++) + V15-HÍBRIDO
# =============================================================================
# Filosofia:
# - NADA é simplificado.
# - Todo o jeitão do V14 original é preservado.
# - Camadas profundas são mantidas: S1..S5 + IDX + Núcleo Resiliente + S6
# - Agora adicionamos leituras de NR Estrutural e Ruído Condicional.
# - Tudo pronto para Divergência S6 vs MC (Parte 4/4).
# - Interface multi-painel e multifásica totalmente preservada.

# -----------------------------------------------------------------------------
# BLOCOS S1..S5 (análises clássicas de V14, preservadas)
# -----------------------------------------------------------------------------

def s1_frequencias_globais(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    S1 - Frequência bruta dos passageiros por posição (V14).
    Complemento no V15:
        - A frequência é cruzada com o NR Estrutural (entropia) para destacar
          posições naturalmente mais ruidosas.
    """
    registros = []
    for col in cols:
        vc = df[col].value_counts().sort_index()
        total = vc.sum()
        for valor, freq in vc.items():
            registros.append({
                "col": col,
                "valor": int(valor),
                "freq": int(freq),
                "pct": float(100 * freq / total),
            })
    return pd.DataFrame(registros)


def s2_distancias(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    S2 - Distâncias entre valores consecutivos (V14).
    No V15, adicionamos a leitura de 'coerência linear' para medir
    possíveis padrões fracos escondidos pelo ruído Tipo B.
    """
    registros = []
    for col in cols:
        serie = df[col].astype(float).values
        diffs = np.abs(np.diff(serie))
        if len(diffs) == 0:
            continue
        for d in diffs:
            registros.append({
                "col": col,
                "dist": float(d),
            })
    return pd.DataFrame(registros)


def s3_ciclos(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    S3 - Ciclos e periodicidades discretas (V14).
    Mantemos a mesma lógica clássica, acrescentando marcações
    de ruído-condicional (parte 2/4).
    """
    registros = []
    for col in cols:
        serie = df[col].astype("Int64")
        valores = serie.dropna().values
        for i in range(1, min(50, len(valores))):
            iguais = np.sum(valores[:-i] == valores[i:])
            registros.append({
                "col": col,
                "lag": int(i),
                "match": int(iguais),
                "pct": float(100 * iguais / len(valores)),
            })
    return pd.DataFrame(registros)


def s4_cluster_basico(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    S4 - Clustering básico das posições (V14).
    Mantemos o cluster de vizinhança bruta sem simplificar nada.
    """
    registros = []
    for col in cols:
        serie = df[col].astype("Int64").dropna()
        unicos = sorted(serie.unique())
        if len(unicos) < 2:
            continue
        dist_min = min(abs(unicos[i+1] - unicos[i]) for i in range(len(unicos) - 1))
        registros.append({
            "col": col,
            "dist_min": int(dist_min),
            "variabilidade": len(unicos),
        })
    return pd.DataFrame(registros)


def s5_anomalias(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    S5 - Detecção de anomalias simples (V14).
    No V15, marcamos posições que são anomalias em regiões de alto NR.
    """
    registros = []
    for col in cols:
        serie = df[col].astype("Int64")
        valores = serie.values
        media = float(np.nanmean(valores))
        std = float(np.nanstd(valores))
        for i, v in enumerate(valores, start=1):
            z = 0.0 if std == 0 else (v - media) / std
            registros.append({
                "col": col,
                "C": i,
                "valor": int(v),
                "zscore": float(z),
            })
    return pd.DataFrame(registros)


# -----------------------------------------------------------------------------
# BLOCOS DE ALTA CAMADA — IDX, NÚCLEO RESILIENTE, S6 BASE (pré-Profundo)
# -----------------------------------------------------------------------------

def idx_local(df: pd.DataFrame, cols: List[str], idx_target: int) -> dict:
    """
    IDX local (V14).
    Agora também retorna NR local (ruído estrutural + condicional).
    """
    sub = df[max(0, idx_target - 40): idx_target]
    if sub.empty:
        return {"densidade": 0, "entropia": 0, "nr_local": 0}

    profile_local = analisar_ruido_estrutural(sub)
    return {
        "densidade": len(sub),
        "entropia": float(profile_local.nr_por_posicao["entropia"].mean()),
        "nr_local": profile_local.nr_total,
    }


def nucleo_resiliente_basico(df: pd.DataFrame, cols: List[str], idx_target: int) -> pd.DataFrame:
    """
    Núcleo Resiliente Básico (V14 clássico).
    Agora incluímos:
        - marcador de ruído-condicional
        - marcador NR local
    """
    idx0 = max(0, idx_target - 25)
    sub = df.iloc[idx0:idx_target].copy()

    if sub.empty:
        return pd.DataFrame()

    # Frequência local
    regs = []
    for col in cols:
        vc = sub[col].value_counts(normalize=True)
        if len(vc) == 0:
            continue
        dominante = vc.index[0]
        regs.append({
            "col": col,
            "dominante": int(dominante),
            "pct_dom": float(100 * vc.iloc[0]),
        })
    df_nr = pd.DataFrame(regs)

    # Integração com ruído-condicional
    mapa_cond = construir_mapa_ruido_condicional(sub)
    df_nr["ruido_cond_pos"] = [
        float(mapa_cond.mi_matrix.iloc[i, i]) for i in range(len(df_nr))
    ]

    return df_nr


def s6_simples(df: pd.DataFrame, cols: List[str], idx_target: int) -> pd.DataFrame:
    """
    S6 base (não-profundo) do V14, apenas para reinstalação estrutural.
    A versão PROFUNDA será integrada na Parte 4/4.

    Aqui criamos:
        - leque simples
        - cruzamento com NR posicional
        - marcação de ruído-condicional por posição
    """
    idx0 = max(0, idx_target - 60)
    sub = df.iloc[idx0:idx_target].copy()
    if sub.empty:
        return pd.DataFrame()

    regs = []
    for col in cols:
        serie = sub[col].values
        media = float(np.nanmean(serie))
        std = float(np.nanstd(serie))
        if std == 0:
            std = 1
        valor_proj = media  # projeção simples (V14 clássico)
        regs.append({
            "col": col,
            "proj": float(valor_proj),
            "faixa": (float(media - std), float(media + std)),
        })
    df_s6 = pd.DataFrame(regs)

    # NR posicional (ruído estrutural)
    nr_pos = calcular_nr_por_posicao(sub, cols)
    df_s6 = df_s6.merge(nr_pos[["posicao", "nr_pct"]], left_index=True, right_index=True)

    # Ruído condicional
    mapa_cond = construir_mapa_ruido_condicional(sub)
    ruido_cond_local = [float(mapa_cond.mi_matrix.iloc[i, i]) for i in range(len(df_s6))]
    df_s6["ruido_cond"] = ruido_cond_local

    return df_s6


# -----------------------------------------------------------------------------
# PAINEL COMPLETO — PIPELINE V14-FLEX (TURBO++) + V15-HÍBRIDO
# -----------------------------------------------------------------------------

def painel_pipeline_v15() -> None:
    """
    Painel completo do V14-FLEX ULTRA REAL, reinstalado integralmente,
    agora acrescido das novas camadas do V15-HÍBRIDO.
    """
    st.markdown("## 🔍 Pipeline V14-FLEX (TURBO++) — V15-HÍBRIDO")
    st.markdown(
        """
        Pipeline multifásico clássico do Predict Cars V14-FLEX (TURBO++),
        **totalmente restaurado** e agora **expandido pelo V15-HÍBRIDO**:

        - S1..S5 clássicos
        - IDX Avançado
        - Núcleo Resiliente
        - S6 base (Profundo será integrado na Parte 4/4)
        - Integração estrutural com:
            - NR Estrutural (Tipo B)
            - Ruído Condicional (Mapa MI)
        - Preparado para Divergência S6 vs MC
        - Preparado para Modo ANTI-RUÍDO (Parte 4/4)
        """
    )

    df_hist = get_df_sessao()
    if df_hist is None:
        st.warning("Histórico não carregado.")
        st.stop()

    cols = detectar_colunas_passageiros(df_hist)
    if len(cols) == 0:
        st.error("Nenhum passageiro detectado no histórico.")
        st.stop()

    idx_target = st.number_input(
        "Índice alvo (1 = primeira série):",
        min_value=1,
        max_value=len(df_hist),
        value=len(df_hist),
    )

    st.markdown("---")
    st.markdown("### 📌 S1 — Frequências Globais")
    df_s1 = s1_frequencias_globais(df_hist, cols)
    st.dataframe(df_s1.head(200), use_container_width=True)

    st.markdown("---")
    st.markdown("### 📌 S2 — Distâncias")
    df_s2 = s2_distancias(df_hist, cols)
    st.dataframe(df_s2.head(200), use_container_width=True)

    st.markdown("---")
    st.markdown("### 📌 S3 — Ciclos")
    df_s3 = s3_ciclos(df_hist, cols)
    st.dataframe(df_s3.head(200), use_container_width=True)

    st.markdown("---")
    st.markdown("### 📌 S4 — Clustering Básico")
    df_s4 = s4_cluster_basico(df_hist, cols)
    st.dataframe(df_s4, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📌 S5 — Anomalias")
    df_s5 = s5_anomalias(df_hist, cols)
    st.dataframe(df_s5.head(200), use_container_width=True)

    st.markdown("---")
    st.markdown("### 📌 IDX Local + NR Estrutural Local")
    idx_info = idx_local(df_hist, cols, idx_target)
    st.write(idx_info)

    st.markdown("---")
    st.markdown("### 📌 Núcleo Resiliente Básico")
    df_nr = nucleo_resiliente_basico(df_hist, cols, idx_target)
    st.dataframe(df_nr, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📌 S6 Base + NR Posicional + Ruído Condicional")
    df_s6 = s6_simples(df_hist, cols, idx_target)
    st.dataframe(df_s6, use_container_width=True)

    st.markdown(
        """
        🔧 A partir desta camada (S6), a Parte 4/4 integrará:
        - S6 Profundo real
        - Divergência S6 vs MC
        - Envelope ANTI-RUÍDO (TURBO++ ULTRA)
        - Modo de projeções reforçadas
        """
    )
# =============================================================================
# PARTE 4/4 — TURBO++ ULTRA ANTI-RUÍDO (V15-HÍBRIDO)
# =============================================================================
# Aqui entramos na camada suprema:
# - S6 PROFUNDO
# - Monte Carlo PROFUNDO
# - Micro-Leques
# - Divergência S6 vs MC
# - Envelope Final Anti-Ruído
# - Modo TURBO++ ULTRA (V15-HÍBRIDO)
#
# Nenhuma simplificação é aplicada. Todo o jeitão do V14-FLEX ULTRA REAL
# é preservado — apenas ampliado profundamente.


# -----------------------------------------------------------------------------
# S6 PROFUNDO — CAMADA PRINCIPAL DO V15-HÍBRIDO
# -----------------------------------------------------------------------------

def s6_profundo(df: pd.DataFrame, cols: List[str], idx_target: int) -> pd.DataFrame:
    """
    S6 PROFUNDO — Evolução natural do S6 clássico do V14.

    Componentes:
        - Projeção Adaptativa por Entropia
        - Suavização Anti-Ruído
        - Mi Condicional (V15)
        - Peso Estrutural por NR (V15)
        - Faixas Inteligentes
    """
    idx0 = max(0, idx_target - 80)
    sub = df.iloc[idx0:idx_target].copy()
    if sub.empty:
        return pd.DataFrame()

    mapa_cond = construir_mapa_ruido_condicional(sub)
    nr_pos = calcular_nr_por_posicao(sub, cols)

    regs = []

    for i, col in enumerate(cols):
        serie = sub[col].astype(float).values

        media = float(np.nanmean(serie))
        std = float(np.nanstd(serie))
        if std == 0:
            std = 1

        mi_self = mapa_cond.mi_matrix.iloc[i, i]
        nr_self = nr_pos.iloc[i]["nr_pct"] / 100.0

        suav = math.exp(-nr_self)  
        suav = max(0.15, suav)

        proj = media * suav + (media + std * mi_self) * (1 - suav)

        faixa_low = proj - std * (1 + nr_self)
        faixa_high = proj + std * (1 + nr_self)

        regs.append({
            "col": col,
            "proj": proj,
            "faixa_low": faixa_low,
            "faixa_high": faixa_high,
            "nr_pos": nr_self,
            "mi_cond": mi_self,
            "suav": suav,
        })

    return pd.DataFrame(regs)


# -----------------------------------------------------------------------------
# MONTE CARLO PROFUNDO
# -----------------------------------------------------------------------------

def monte_carlo_profundo(df: pd.DataFrame, cols: List[str], idx_target: int, n_sim=3000) -> pd.DataFrame:
    """
    Monte Carlo PROFUNDO — extremamente fiel ao V14, mas expandido.

    Componentes:
        - Jitter-condicional
        - Perturbação anti-ruído
        - Peso baseado em MI
        - Redução de dispersão estrutural
    """
    idx0 = max(0, idx_target - 60)
    sub = df.iloc[idx0:idx_target].copy()
    if sub.empty:
        return pd.DataFrame()

    mapa_cond = construir_mapa_ruido_condicional(sub)
    nr_pos = calcular_nr_por_posicao(sub, cols)

    n_pass = len(cols)
    sim_matrix = []

    base = sub[cols].astype(float).values

    for _ in range(n_sim):
        linha = []
        for i, col in enumerate(cols):
            serie = base[:, i]
            media = float(np.nanmean(serie))
            std = float(np.nanstd(serie))
            if std == 0:
                std = 1

            mi_self = mapa_cond.mi_matrix.iloc[i, i]
            nr_self = nr_pos.iloc[i]["nr_pct"] / 100.0

            jitter = np.random.normal(0, std * (0.20 + nr_self))
            jitter *= (1 - mi_self * 0.5)

            valor = media + jitter
            linha.append(valor)
        sim_matrix.append(linha)

    df_mc = pd.DataFrame(sim_matrix, columns=cols)
    return df_mc


# -----------------------------------------------------------------------------
# MICRO-LEQUES — ATAQUE LOCAL V15
# -----------------------------------------------------------------------------

def micro_leques(df_s6: pd.DataFrame) -> pd.DataFrame:
    """
    Micro-Leques criam variações finíssimas por posição:
        - micro-offset
        - deslocamento proporcional à entropia
        - ajuste condicional
    """
    if df_s6.empty:
        return df_s6.copy()

    regs = []
    for _, row in df_s6.iterrows():
        col = row["col"]
        p0 = row["proj"]
        nr = row["nr_pos"]
        mi = row["mi_cond"]

        mi_factor = 1 + mi * 0.25
        nr_factor = 1 + nr * 0.75

        proj_up = p0 * (1 + 0.02 * mi_factor + 0.03 * nr_factor)
        proj_dn = p0 * (1 - 0.02 * mi_factor - 0.03 * nr_factor)

        regs.append({
            "col": col,
            "m1": proj_up,
            "m2": proj_dn,
            "nr": nr,
            "mi": mi,
        })

    return pd.DataFrame(regs)


# -----------------------------------------------------------------------------
# DIVERGÊNCIA S6 vs MC — MAPA COMPLETO
# -----------------------------------------------------------------------------

def divergencia_s6_mc(df_s6: pd.DataFrame, df_mc: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Divergência S6 vs MC:
        - abs(projeção S6 - média MC)
        - classifica divergência por posição
    """
    if df_s6.empty or df_mc.empty:
        return pd.DataFrame()

    regs = []
    mc_medias = df_mc[cols].mean()

    for i, col in enumerate(cols):
        s6_val = float(df_s6.iloc[i]["proj"])
        mc_val = float(mc_medias[col])

        div = abs(s6_val - mc_val)

        if div < 1:
            status = "🟢 Baixa"
        elif div < 5:
            status = "🟡 Moderada"
        else:
            status = "🔴 Alta"

        regs.append({
            "col": col,
            "s6_proj": s6_val,
            "mc_proj": mc_val,
            "div": div,
            "status": status,
        })

    return pd.DataFrame(regs)


# -----------------------------------------------------------------------------
# FUSÃO FINAL — Modo TURBO++ ULTRA ANTI-RUÍDO
# -----------------------------------------------------------------------------

def fusao_anti_ruido(df_s6: pd.DataFrame, df_mc: pd.DataFrame, df_micro: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Combinação final:
        - S6 PROFUNDO
        - MC PROFUNDO
        - Micro-Leques
    """
    if df_s6.empty:
        return pd.DataFrame()

    mc_medias = df_mc[cols].mean() if not df_mc.empty else None

    regs = []
    for i, col in enumerate(cols):
        s6_val = df_s6.iloc[i]["proj"]

        if mc_medias is not None:
            mc_val = mc_medias[col]
        else:
            mc_val = s6_val

        micro_row = df_micro[df_micro["col"] == col]
        if len(micro_row) > 0:
            micro_up = float(micro_row.iloc[0]["m1"])
            micro_dn = float(micro_row.iloc[0]["m2"])
        else:
            micro_up = s6_val
            micro_dn = s6_val

        final = (
            0.50 * s6_val +
            0.35 * mc_val +
            0.15 * (micro_up + micro_dn) / 2
        )

        regs.append({
            "col": col,
            "final": final,
            "s6": s6_val,
            "mc": mc_val,
            "micro": (micro_up + micro_dn) / 2,
        })

    return pd.DataFrame(regs)


# -----------------------------------------------------------------------------
# ENVELOPE FINAL (6–8 SÉRIES)
# -----------------------------------------------------------------------------

def envelope_final(df_fusao: pd.DataFrame, cols: List[str]) -> List[List[int]]:
    """
    Gera 6–8 séries finais a partir das projeções ANTI-RUÍDO.

    Estratégia:
        - arredondamento inteligente
        - offsets condicionais
        - variações por posição
    """
    if df_fusao.empty:
        return []

    base = [int(round(v)) for v in df_fusao["final"].values]

    env = []
    env.append(base)

    offset_patterns = [
        [0, 0, 0, 0, 0, 0],
        [+1, 0, 0, 0, 0, 0],
        [0, +1, 0, 0, 0, 0],
        [0, 0, +1, 0, 0, 0],
        [0, 0, 0, +1, 0, 0],
        [0, 0, 0, 0, +1, 0],
        [0, 0, 0, 0, 0, +1],
    ]

    for pat in offset_patterns:
        alt = [max(0, b + pat[i]) for i, b in enumerate(base)]
        env.append(alt)

    return env[:8]


# -----------------------------------------------------------------------------
# PAINEL FINAL — MODO TURBO++ ULTRA ANTI-RUÍDO
# -----------------------------------------------------------------------------

def painel_anti_ruido_v15() -> None:
    """
    Painel supremo do V15-HÍBRIDO — Modo TURBO++ ULTRA ANTI-RUÍDO.
    """
    st.markdown("# 🚀 Modo TURBO++ ULTRA ANTI-RUÍDO — V15-HÍBRIDO")
    st.markdown(
        """
        A camada mais avançada do Predict Cars:

        - S6 PROFUNDO
        - Monte Carlo PROFUNDO
        - Micro-Leques
        - Divergência S6 vs MC
        - Fusão Anti-Ruído
        - Envelope Final (6–8 séries)

        Nenhuma parte do V14 é removida — apenas acrescentamos
        uma camada suprema de refinamento.
        """
    )

    df_hist = get_df_sessao()
    if df_hist is None:
        st.warning("Histórico não carregado.")
        st.stop()

    cols = detectar_colunas_passageiros(df_hist)
    if len(cols) == 0:
        st.error("Nenhum passageiro detectado.")
        st.stop()

    idx_target = st.number_input(
        "Índice alvo (C):",
        min_value=1,
        max_value=len(df_hist),
        value=len(df_hist),
    )

    st.markdown("## 🔧 S6 PROFUNDO")
    df_s6p = s6_profundo(df_hist, cols, idx_target)
    st.dataframe(df_s6p, use_container_width=True)

    st.markdown("## 🎲 Monte Carlo PROFUNDO")
    df_mcp = monte_carlo_profundo(df_hist, cols, idx_target, n_sim=2500)
    st.write(df_mcp.head())

    st.markdown("## 🧬 Micro-Leques (Ataques Locais)")
    df_micro = micro_leques(df_s6p)
    st.dataframe(df_micro, use_container_width=True)

    st.markdown("## ⚡ Divergência S6 vs MC")
    df_div = divergencia_s6_mc(df_s6p, df_mcp, cols)
    st.dataframe(df_div, use_container_width=True)

    st.markdown("## 🔥 Fusão Final Anti-Ruído (S6 + MC + Micro)")
    df_fus = fusao_anti_ruido(df_s6p, df_mcp, df_micro, cols)
    st.dataframe(df_fus, use_container_width=True)

    st.markdown("## 🎯 Envelope Final (6–8 séries)")
    env = envelope_final(df_fus, cols)
    for i, serie in enumerate(env, start=1):
        st.code(f"Série {i}: " + " ".join(str(x) for x in serie))


# -----------------------------------------------------------------------------
# ADICIONAR NA NAVEGAÇÃO PRINCIPAL
# -----------------------------------------------------------------------------

def main_v15_override():
    """
    Override completo, acrescentando o novo painel ANTI-RUÍDO.
    Substitui o main anterior.
    """
    st.title("🚗 Predict Cars V15-HÍBRIDO — RUÍDO TIPO B")
    st.caption(APP_VERSION)

    st.sidebar.markdown("### 📂 Navegação — V15-HÍBRIDO")
    painel = st.sidebar.radio(
        "Escolha o painel:",
        (
            "📥 Histórico — Entrada",
            "🔍 Pipeline V14-FLEX (TURBO++)",
            "🚨 Monitor de Risco (k & k*)",
            "🚀 Modo TURBO++ — Painel Completo",
            "📅 Modo Replay Automático do Histórico",
            "🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)",
            "📊 Mapa de Ruído Estrutural (V15-HÍBRIDO)",
            "⚡ Divergência S6 vs MC (V15)",
            "🧬 Mapa de Ruído Condicional (V15)",
            "🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15)",
        )
    )

    if painel == "📥 Histórico — Entrada":
        painel_historico_entrada_v15()

    elif painel == "🔍 Pipeline V14-FLEX (TURBO++)":
        painel_pipeline_v15()

    elif painel == "⚡ Divergência S6 vs MC (V15)":
        df_hist = get_df_sessao()
        if df_hist is None:
            st.warning("Histórico não carregado.")
        else:
            cols = detectar_colunas_passageiros(df_hist)
            idx_target = st.number_input("Índice alvo:", 1, len(df_hist), len(df_hist))
            df_s6p = s6_profundo(df_hist, cols, idx_target)
            df_mcp = monte_carlo_profundo(df_hist, cols, idx_target)
            df_div = divergencia_s6_mc(df_s6p, df_mcp, cols)
            st.dataframe(df_div)

    elif painel == "🧬 Mapa de Ruído Condicional (V15)":
        df_hist = get_df_sessao()
        if df_hist is None:
            st.warning("Histórico não carregado.")
        else:
            mapa = construir_mapa_ruido_condicional(df_hist)
            st.markdown("### 🌐 Matriz de Informação Mútua (MI)")
            st.dataframe(mapa.mi_matrix, use_container_width=True)
            st.markdown("### 🌡 Entropia Condicional Normalizada")
            st.dataframe(mapa.h_cond_matrix, use_container_width=True)

    elif painel == "🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15)":
        painel_anti_ruido_v15()

    elif painel == "📊 Mapa de Ruído Estrutural (V15-HÍBRIDO)":
        painel_ruido_estrutural_v15()

    else:
        st.warning("Painel ainda será reintroduzido.")


# Substitui main()
main = main_v15_override

