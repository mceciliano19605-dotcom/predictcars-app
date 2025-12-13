# ============================================================
# PredictCars V16 Premium — Camada sobre V15.7 MAX
# Arquivo: app_v16_premium.py
# ------------------------------------------------------------
# Este módulo NÃO substitui o app_v15_7_MAX.py.
# Ele assume que o V15.7 MAX já está rodando como motor base
# e consome:
#   - historico_df
#   - sentinela_kstar
#   - nr_percent
#   - div_s6_mc
#   - ultima_previsao (C1)
#   - modo6_listas (listas do Modo 6 clássico, se existirem)
#   - diagnostico_risco
# via st.session_state.
# ------------------------------------------------------------
# Integração sugerida no app principal:
#
#   from app_v16_premium import v16_obter_paineis, v16_renderizar_painel
#
#   opcoes_base = [ ... painéis V15.7 ... ]
#   opcoes_v16 = v16_obter_paineis()
#   opcoes = opcoes_base + opcoes_v16
#
#   painel = st.sidebar.selectbox("Selecione um painel:", opcoes)
#
#   if painel in opcoes_base:
#       ... lógica atual ...
#   else:
#       v16_renderizar_painel(painel)
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Dict, Any, Optional

# ============================================================
# Bloco utilitário V16 — funções auxiliares internas
# ============================================================

def v16_obter_paineis() -> List[str]:
    """
    Retorna a lista de painéis V16 para compor o menu único do app.
    """
    return [
        "🔮 V16 — Microjanelas & Diagnóstico",
        "🟦 V16 — Janelas Favoráveis",
        "⚙️ V16 — Engine Adaptativa",
        "🎛️ V16 — Painel Central de Parâmetros",
        "🎯 V16 — Execução Adaptativa",
        "🎯 V16 — Modo 6 Premium",
        "📘 V16 — Relatório Final Premium",
    ]


def _v16_exibir_bloco(titulo: str, corpo: str, tipo: str = "info") -> None:
    """
    Wrapper leve para manter o jeitão de mensagens do PredictCars
    sem depender das funções internas do app principal.
    """
    if tipo == "info":
        st.info(f"**{titulo}**\n\n{corpo}")
    elif tipo == "warning":
        st.warning(f"**{titulo}**\n\n{corpo}")
    elif tipo == "error":
        st.error(f"**{titulo}**\n\n{corpo}")
    elif tipo == "success":
        st.success(f"**{titulo}**\n\n{corpo}")
    else:
        st.markdown(f"**{titulo}**\n\n{corpo}")


def _v16_obter_historico() -> Optional[pd.DataFrame]:
    """
    Usa o mesmo DataFrame de histórico do app V15.7 MAX.
    """
    df = st.session_state.get("historico_df")
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df
    return None


def _v16_obter_risco_basico() -> Dict[str, Any]:
    """
    Usa o diagnóstico de risco original se existir,
    completando com k*, NR% e divergência reais.
    """
    risco = dict(st.session_state.get("diagnostico_risco") or {})
    k_star = st.session_state.get("sentinela_kstar")
    nr_percent = st.session_state.get("nr_percent")
    divergencia = st.session_state.get("div_s6_mc")

    if "k_star" not in risco:
        risco["k_star"] = k_star if isinstance(k_star, (int, float)) else 0.25
    if "nr_percent" not in risco:
        risco["nr_percent"] = nr_percent if isinstance(nr_percent, (int, float)) else 35.0
    if "divergencia" not in risco:
        risco["divergencia"] = divergencia if isinstance(divergencia, (int, float)) else 4.0

    if "indice_risco" not in risco:
        # reaplica a mesma lógica do Monitor de Risco como fallback
        k_norm = min(1.0, risco["k_star"] / 0.50)
        nr_norm = min(1.0, risco["nr_percent"] / 70.0)
        div_norm = min(1.0, risco["divergencia"] / 8.0)
        risco["indice_risco"] = float(0.40 * k_norm + 0.35 * nr_norm + 0.25 * div_norm)

    if "classe_risco" not in risco:
        ir = risco["indice_risco"]
        # Aqui é apenas uma leitura qualitativa, o regime V16 será definido depois (ouro/normal/ruim)
        if ir < 0.30:
            risco["classe_risco"] = "🟢 Janela Favorável"
        elif ir < 0.65:
            risco["classe_risco"] = "🟡 Ambiente Neutro"
        else:
       	    risco["classe_risco"] = "🔴 Ambiente Difícil"

    return risco


def _v16_microjanelas(df: pd.DataFrame, tamanhos: List[int] = [30, 60, 100, 150, 200]) -> Dict[str, Any]:
    """
    Núcleo de microjanelas V16.
    Não recalcula nada do motor, apenas observa a microestrutura do histórico.
    """
    n = len(df)
    resultados: Dict[str, Any] = {}

    col_pass = [c for c in df.columns if str(c).startswith("p")]
    if not col_pass or n <= 5:
        return {
            "microjanelas": {},
            "melhor_janela": None,
            "score_melhor": 0.0,
            "janela_ouro": False,
        }

    matriz = df[col_pass].astype(float).values

    for t in tamanhos:
        if n < t:
            continue
        bloco = matriz[-t:]
        # Medidas simples de ordem local
        medias = np.mean(bloco, axis=1)
        desvios = np.std(bloco, axis=1)

        energia = float(np.mean(medias))
        vol = float(np.mean(desvios))

        # Score de ordem: quanto menor a volatilidade, maior o score (limitado em [0,1])
        score_ordem = max(0.0, min(1.0, 1.0 - vol * 3.0))

        resultados[str(t)] = {
            "tamanho": t,
            "energia": energia,
            "volatilidade": vol,
            "score_ordem": score_ordem,
        }

    if not resultados:
        return {
            "microjanelas": {},
            "melhor_janela": None,
            "score_melhor": 0.0,
            "janela_ouro": False,
        }

    # Escolhe a janela com maior score de ordem
    chave_melhor = max(resultados.keys(), key=lambda k: resultados[k]["score_ordem"])
    score_melhor = resultados[chave_melhor]["score_ordem"]

    # Critério V16: janela de ouro quando o score local é muito alto
    janela_ouro = bool(score_melhor >= 0.80)

    return {
        "microjanelas": resultados,
        "melhor_janela": int(chave_melhor),
        "score_melhor": float(score_melhor),
        "janela_ouro": janela_ouro,
    }


def _v16_obter_parametros_padrao() -> Dict[str, Any]:
    """
    Parametrização central V16 — padrão inicial.
    Fica em st.session_state["v16_parametros"] e pode ser ajustada pelo Painel Central.
    """
    params = st.session_state.get("v16_parametros")
    if isinstance(params, dict):
        return params

    params = {
        "nucleo": {
            "forca_base": 0.60,
            "reforco_em_ouro": 0.15,
            "retracao_em_ruim": 0.25,
            "suavizacao": 0.20,
            "limiar_divergencia": 4.0,
        },
        "coberturas": {
            "amplitude_fina": 4,
            "amplitude_larga": 10,
            "limite_total": 200,
            "sensibilidade_movimento": 0.5,
        },
        "volumes": {
            "teto_ouro": 60,
            "teto_normal": 20,
            "teto_ruim": 8,
            "sensibilidade_confiabilidade": 1.0,
            "min_ouro": 15,
            "min_normal": 6,
            "min_ruim": 3,
            "min_respiratorio": 3,
        },
        "modo6": {
            "limite_kstar": 0.35,
            "limite_nr_percent": 60.0,
            "limite_divergencia": 7.0,
            "sensibilidade_ouro": 0.10,
        },
        "confiabilidade": {
            "minima_global": 0.15,
            "reforco_janela_ouro": 0.10,
            "penalidade_ruim": 0.20,
        },
    }

    st.session_state["v16_parametros"] = params
    return params


def _v16_calcular_regime_e_confiabilidade(
    risco: Dict[str, Any],
    microdiag: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Engine conceitual do regime V16:
    - regime ∈ {ouro, normal, ruim}
    - confiabilidade em [0,1]
    Usa:
      - indice_risco (k*, NR%, divergência)
      - score da melhor microjanela
      - flag de janela de ouro
    """
    indice_risco = float(risco.get("indice_risco", 0.5))
    k_star = float(risco.get("k_star", 0.25))
    nr_percent = float(risco.get("nr_percent", 35.0))
    divergencia = float(risco.get("divergencia", 4.0))

    score_micro = float(microdiag.get("score_melhor", 0.0))
    janela_ouro = bool(microdiag.get("janela_ouro", False))

    # Regime base pela combinação de risco e microjanela
    if janela_ouro and indice_risco <= 0.40 and score_micro >= 0.80:
        regime = "ouro"
    elif indice_risco <= 0.70:
        regime = "normal"
    else:
        regime = "ruim"

    conf_cfg = params["confiabilidade"]
    base_conf = max(0.0, min(1.0, 1.0 - indice_risco))

    # reforço em janela de ouro
    if janela_ouro:
        base_conf += conf_cfg.get("reforco_janela_ouro", 0.10)

    # penalização em ambiente ruim
    if regime == "ruim":
        base_conf -= conf_cfg.get("penalidade_ruim", 0.20)

    # ajustes suaves por k* e NR%
    base_conf -= 0.10 * max(0.0, (k_star - 0.30))
    base_conf -= 0.10 * max(0.0, (nr_percent - 40.0) / 60.0)

    confiabilidade = max(conf_cfg.get("minima_global", 0.15), min(1.0, base_conf))

    return {
        "regime": regime,
        "confiabilidade": confiabilidade,
        "indice_risco": indice_risco,
        "k_star": k_star,
        "nr_percent": nr_percent,
        "divergencia": divergencia,
        "score_micro": score_micro,
        "janela_ouro": janela_ouro,
    }


def _v16_calcular_dinamica_alvo(
    risco: Dict[str, Any],
    microdiag: Dict[str, Any],
) -> str:
    """
    Determina a dinâmica do alvo:
    - 'estavel'
    - 'em_movimento'
    usando NR%, divergência e volatilidade das microjanelas.
    """
    nr_percent = float(risco.get("nr_percent", 35.0))
    divergencia = float(risco.get("divergencia", 4.0))

    microjanelas = microdiag.get("microjanelas", {}) or {}
    vols = [v.get("volatilidade", 0.0) for v in microjanelas.values()]
    vol_media = float(np.mean(vols)) if vols else 0.0

    # critério qualitativo: muita turbulência => alvo em movimento
    if nr_percent >= 50.0 or divergencia >= 6.0 or vol_media >= 0.25:
        return "em_movimento"
    return "estavel"


def _v16_montar_engine_info(
    df: pd.DataFrame,
    risco: Dict[str, Any],
    microdiag: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Engine V16 Adaptativa:
    - regime
    - confiabilidade
    - dinâmica do alvo
    - tetos e volumes
    - amplitudes de cobertura
    - pesos relativos do núcleo/coberturas
    """
    base = _v16_calcular_regime_e_confiabilidade(risco, microdiag, params)
    regime = base["regime"]
    confiabilidade = base["confiabilidade"]
    dinamica = _v16_calcular_dinamica_alvo(risco, microdiag)

    vol_cfg = params["volumes"]
    cob_cfg = params["coberturas"]
    nuc_cfg = params["nucleo"]

    if regime == "ouro":
        teto = vol_cfg["teto_ouro"]
        minimo = vol_cfg["min_ouro"]
    elif regime == "normal":
        teto = vol_cfg["teto_normal"]
        minimo = vol_cfg["min_normal"]
    else:
        teto = vol_cfg["teto_ruim"]
        minimo = vol_cfg["min_ruim"]

    # volume bruto pelo Planejamento
    volume_bruto = teto * confiabilidade * vol_cfg.get("sensibilidade_confiabilidade", 1.0)
    volume_sugerido = int(round(volume_bruto))

    # mínimo respiratório garantido (nunca 0, mesmo em regime ruim)
    min_respiratorio = vol_cfg.get("min_respiratorio", 3)
    volume_final = max(minimo, volume_sugerido, min_respiratorio)

    # Anti-zumbi interno V16 — limite duro para volume final
    limite_total = int(cob_cfg.get("limite_total", 200))
    volume_final = min(volume_final, limite_total)

    # Coberturas proporcionalmente ajustadas pela dinâmica do alvo
    amplitude_fina = int(cob_cfg["amplitude_fina"])
    amplitude_larga = int(cob_cfg["amplitude_larga"])

    if dinamica == "em_movimento":
        amplitude_larga = int(amplitude_larga * (1.0 + cob_cfg.get("sensibilidade_movimento", 0.5)))

    # Pesos núcleo x coberturas (qualitativos, não reexecutam motores)
    forca_base = nuc_cfg["forca_base"]
    if regime == "ouro":
        peso_nucleo = min(1.0, forca_base + nuc_cfg.get("reforco_em_ouro", 0.15))
    elif regime == "normal":
        peso_nucleo = forca_base
    else:
        peso_nucleo = max(0.10, forca_base - nuc_cfg.get("retracao_em_ruim", 0.25))

    peso_coberturas = max(0.0, 1.0 - peso_nucleo)

    engine_info = {
        **base,
        "dinamica_alvo": dinamica,
        "teto_regime": teto,
        "volume_sugerido": volume_sugerido,
        "volume_final": volume_final,
        "amplitude_fina": amplitude_fina,
        "amplitude_larga": amplitude_larga,
        "peso_nucleo": peso_nucleo,
        "peso_coberturas": peso_coberturas,
        "limite_total_listas": limite_total,
    }

    st.session_state["v16_engine_info"] = engine_info
    return engine_info


def _v16_executar_selecao_listas_v16(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Execução Adaptativa V16:
    - NÃO cria motores novos
    - Usa o Modo 6 clássico e o Núcleo (ultima_previsao)
    para selecionar um subconjunto adaptativo de listas.
    """
    engine = st.session_state.get("v16_engine_info")
    microdiag = st.session_state.get("v16_microdiag")

    if not engine or not microdiag:
        return {}

    listas_m6 = st.session_state.get("modo6_listas") or []
    ultima_prev = st.session_state.get("ultima_previsao")

    if not listas_m6 or ultima_prev is None:
        return {}

    volume_final = int(engine.get("volume_final", 20))
    limite_total = int(engine.get("limite_total_listas", 200))
    volume_final = max(1, min(volume_final, limite_total))

    base = list(map(int, ultima_prev))

    def similaridade(a: List[int], b: List[int]) -> int:
        return len(set(a) & set(b))

    # Ordena as listas do Modo 6 clássico por similaridade ao núcleo
    ordenadas = sorted(
        [list(map(int, lst)) for lst in listas_m6],
        key=lambda x: similaridade(base, x),
        reverse=True,
    )

    listas_escolhidas = ordenadas[:volume_final]

    # C1 = núcleo (sempre presente)
    c1 = base
    # C2/C3 são subconjuntos da seleção, não reinventados
    meio = max(1, len(listas_escolhidas) // 3)
    c2 = listas_escolhidas[:meio]
    c3 = listas_escolhidas[meio:]

    resultado = {
        "C1": c1,
        "C2": c2,
        "C3": c3,
        "todas_listas": listas_escolhidas,
    }

    st.session_state["v16_execucao"] = resultado
    return resultado


def _v16_classificar_modo6(engine: Dict[str, Any], params: Dict[str, Any]) -> str:
    """
    Classificação adaptativa do Modo 6 Premium:
    - 'recomendado'
    - 'moderado'
    - 'proibido'
    baseada em:
      - regime
      - confiabilidade
      - k*
      - NR%
      - divergência
    """
    regime = engine.get("regime", "normal")
    conf = float(engine.get("confiabilidade", 0.5))
    k_star = float(engine.get("k_star", 0.25))
    nr_percent = float(engine.get("nr_percent", 35.0))
    divergencia = float(engine.get("divergencia", 4.0))

    cfg = params["modo6"]

    if k_star > cfg["limite_kstar"] or nr_percent > cfg["limite_nr_percent"] or divergencia > cfg["limite_divergencia"]:
        return "proibido"

    if regime == "ouro" and conf >= 0.75:
        return "recomendado"

    if regime == "normal" and conf >= 0.50:
        return "moderado"

    if regime == "ruim" and conf >= 0.60:
        return "moderado"

    return "proibido"


# ============================================================
# Bloco público — roteador de painéis V16
# ============================================================

def v16_renderizar_painel(nome_painel: str) -> None:
    """
    Roteia o painel selecionado para o módulo correspondente V16.
    """
    if nome_painel == "🔮 V16 — Microjanelas & Diagnóstico":
        _painel_v16_microjanelas()
    elif nome_painel == "🟦 V16 — Janelas Favoráveis":
        _painel_v16_janelas_favoraveis()
    elif nome_painel == "⚙️ V16 — Engine Adaptativa":
        _painel_v16_engine()
    elif nome_painel == "🎛️ V16 — Painel Central de Parâmetros":
        _painel_v16_painel_central()
    elif nome_painel == "🎯 V16 — Execução Adaptativa":
        _painel_v16_execucao()
    elif nome_painel == "🎯 V16 — Modo 6 Premium":
        _painel_v16_modo6_premium()
    elif nome_painel == "📘 V16 — Relatório Final Premium":
        _painel_v16_relatorio_final()
    else:
        _v16_exibir_bloco(
            "Painel V16 desconhecido",
            f"O painel '{nome_painel}' não foi reconhecido pelo módulo V16.",
            tipo="warning",
        )


# ============================================================
# BLOCO 1/7 — Núcleo de Microjanelas & Diagnóstico
# ============================================================

def _painel_v16_microjanelas() -> None:
    st.markdown("## 🔮 V16 — Microjanelas & Diagnóstico Profundo")

    df = _v16_obter_historico()
    if df is None:
        _v16_exibir_bloco(
            "Histórico não encontrado",
            "Carregue o histórico e rode pelo menos o Pipeline V14-FLEX ULTRA no app principal antes de usar o V16.",
            tipo="warning",
        )
        return

    risco = _v16_obter_risco_basico()
    microdiag = _v16_microjanelas(df)

    st.session_state["v16_microdiag"] = microdiag

    # Exibição resumida
    st.markdown("### 📊 Microjanelas avaliadas")
    if not microdiag["microjanelas"]:
        _v16_exibir_bloco(
            "Microjanelas insuficientes",
            "O histórico atual é muito curto para avaliação detalhada de microjanelas.",
            tipo="warning",
        )
        return

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Melhor janela (séries)", microdiag["melhor_janela"])
        st.metric("Score da melhor janela", f"{microdiag['score_melhor']:.4f}")
    with col2:
        st.metric("Janela de ouro ativa?", "Sim" if microdiag["janela_ouro"] else "Não")
        st.metric("k* (sentinela)", f"{risco['k_star']:.4f}")

    st.markdown("#### Detalhes por microjanela")
    linhas = []
    for chave, info in microdiag["microjanelas"].items():
        linhas.append(
            {
                "tamanho": info["tamanho"],
                "energia": round(info["energia"], 4),
                "volatilidade": round(info["volatilidade"], 4),
                "score_ordem": round(info["score_ordem"], 4),
            }
        )
    st.dataframe(pd.DataFrame(linhas), use_container_width=True)

    _v16_exibir_bloco(
        "Diagnóstico qualitativo",
        (
            "Este painel identifica **microtrechos de maior ordem local**. "
            "Scores elevados indicam **microjanelas mais organizadas**, com maior potencial "
            "para se tornarem janelas favoráveis ou até **janelas de ouro**."
        ),
        tipo="info",
    )


# ============================================================
# BLOCO 2/7 — Painel de Janelas Favoráveis
# ============================================================

def _painel_v16_janelas_favoraveis() -> None:
    st.markdown("## 🟦 V16 — Painel de Janelas Favoráveis")

    microdiag = st.session_state.get("v16_microdiag")
    if not microdiag:
        _v16_exibir_bloco(
            "Microjanelas não calculadas",
            "Acesse primeiro o painel **🔮 V16 — Microjanelas & Diagnóstico**.",
            tipo="warning",
        )
        return

    risco = _v16_obter_risco_basico()

    score = microdiag.get("score_melhor", 0.0)
    janela_ouro = bool(microdiag.get("janela_ouro", False))
    melhor_janela = microdiag.get("melhor_janela")

    if janela_ouro:
        status_janela = "🟢 Janela de Ouro detectada"
        comentario = (
            "A microjanela atual apresenta **alta ordem local** e **baixa volatilidade**, "
            "sugerindo um trecho especialmente favorável para decisões mais intensas, "
            "respeitando sempre o controle de risco."
        )
    elif score >= 0.60:
        status_janela = "🟡 Janela Boa / Favorável"
        comentario = (
            "A melhor microjanela indica um ambiente **razoavelmente organizado**, "
            "com potencial de exploração moderada. O sistema pode operar com intensidade "
            "intermediária, reforçando o núcleo e mantendo coberturas proporcionais."
        )
    else:
        status_janela = "🔴 Janela Normal / Difícil"
        comentario = (
            "Não há evidências fortes de ordem local robusta. O ambiente se comporta mais "
            "como uma janela normal ou ruim. O foco deve ser em **previsões respiratórias** "
            "e proteção estatística, sem agressividade excessiva."
        )

    corpo = (
        f"- Melhor microjanela avaliada: **{melhor_janela} séries**\n"
        f"- Score de ordem da melhor janela: **{score:.4f}**\n"
        f"- Estado da janela: **{status_janela}**\n"
        f"- k* atual: **{risco['k_star']:.4f}** · NR%: **{risco['nr_percent']:.2f}%** · "
        f"Divergência: **{risco['divergencia']:.4f}**"
    )

    _v16_exibir_bloco("Resumo das Janelas Favoráveis", corpo, tipo="info")

    _v16_exibir_bloco(
        "Interpretação V16",
        comentario,
        tipo="info",
    )


# ============================================================
# BLOCO 3/7 — Engine V16 Adaptativa
# ============================================================

def _painel_v16_engine() -> None:
    st.markdown("## ⚙️ V16 — Engine Adaptativa")

    df = _v16_obter_historico()
    if df is None:
        _v16_exibir_bloco(
            "Histórico não encontrado",
            "Carregue o histórico no app principal antes de usar a Engine V16.",
            tipo="warning",
        )
        return

    microdiag = st.session_state.get("v16_microdiag")
    if not microdiag:
        _v16_exibir_bloco(
            "Microjanelas não calculadas",
            "Acesse primeiro o painel **🔮 V16 — Microjanelas & Diagnóstico**.",
            tipo="warning",
        )
        return

    risco = _v16_obter_risco_basico()
    params = _v16_obter_parametros_padrao()

    engine_info = _v16_montar_engine_info(df, risco, microdiag, params)

    regime = engine_info["regime"]
    conf = engine_info["confiabilidade"]

    if regime == "ouro":
        regime_txt = "🟢 Regime OURO"
    elif regime == "normal":
        regime_txt = "🟠 Regime NORMAL"
    else:
        regime_txt = "🔴 Regime RUIM"

    corpo = (
        f"- Regime V16: **{regime_txt}**\n"
        f"- Confiabilidade V16: **{conf:.4f}**\n"
        f"- Dinâmica do alvo: **{engine_info['dinamica_alvo']}**\n"
        f"- Volume sugerido: **{engine_info['volume_sugerido']}**\n"
        f"- Volume final (aplicado): **{engine_info['volume_final']}**\n"
        f"- Teto do regime: **{engine_info['teto_regime']}**\n"
        f"- Amplitude fina: **{engine_info['amplitude_fina']}** · "
        f"Amplitude larga: **{engine_info['amplitude_larga']}**\n"
        f"- Peso do núcleo: **{engine_info['peso_nucleo']:.3f}** · "
        f"Peso das coberturas: **{engine_info['peso_coberturas']:.3f}**"
    )

    _v16_exibir_bloco(
        "Resumo da Engine V16 Adaptativa",
        corpo,
        tipo="info",
    )

    _v16_exibir_bloco(
        "Observação",
        (
            "A Engine V16 **não recria motores**. Ela apenas traduz o ambiente atual "
            "em parâmetros de regime, confiabilidade, volume e coberturas, que serão "
            "usados pela Execução Adaptativa e pelo Modo 6 Premium."
        ),
        tipo="info",
    )


# ============================================================
# BLOCO 4/7 — Painel Central de Parâmetros Premium
# ============================================================

def _painel_v16_painel_central() -> None:
    st.markdown("## 🎛️ V16 — Painel Central de Parâmetros Premium")

    params = _v16_obter_parametros_padrao()

    with st.expander("🔹 BLOCO 1 — Núcleo", expanded=True):
        nuc = params["nucleo"]
        nuc["forca_base"] = st.slider(
            "Força base do núcleo",
            0.10,
            0.90,
            float(nuc["forca_base"]),
            0.05,
        )
        nuc["reforco_em_ouro"] = st.slider(
            "Reforço em regime OURO",
            0.00,
            0.40,
            float(nuc["reforco_em_ouro"]),
            0.05,
        )
        nuc["retracao_em_ruim"] = st.slider(
            "Retração em regime RUIM",
            0.00,
            0.40,
            float(nuc["retracao_em_ruim"]),
            0.05,
        )

    with st.expander("🔹 BLOCO 2 — Coberturas", expanded=False):
        cob = params["coberturas"]
        cob["amplitude_fina"] = st.slider(
            "Amplitude fina (alvo estável)",
            1,
            15,
            int(cob["amplitude_fina"]),
        )
        cob["amplitude_larga"] = st.slider(
            "Amplitude larga (alvo em movimento)",
            3,
            30,
            int(cob["amplitude_larga"]),
        )
        cob["limite_total"] = st.slider(
            "Limite total de listas (anti-zumbi V16)",
            50,
            400,
            int(cob["limite_total"]),
            10,
        )
        cob["sensibilidade_movimento"] = st.slider(
            "Sensibilidade ao movimento do alvo",
            0.10,
            1.50,
            float(cob["sensibilidade_movimento"]),
            0.05,
        )

    with st.expander("🔹 BLOCO 3 — Volumes", expanded=False):
        vol = params["volumes"]
        vol["teto_ouro"] = st.slider(
            "Teto de volume em regime OURO",
            20,
            120,
            int(vol["teto_ouro"]),
            5,
        )
        vol["teto_normal"] = st.slider(
            "Teto de volume em regime NORMAL",
            10,
            40,
            int(vol["teto_normal"]),
            2,
        )
        vol["teto_ruim"] = st.slider(
            "Teto de volume em regime RUIM",
            3,
            20,
            int(vol["teto_ruim"]),
            1,
        )
        vol["min_ouro"] = st.slider(
            "Mínimo em regime OURO",
            5,
            30,
            int(vol["min_ouro"]),
        )
        vol["min_normal"] = st.slider(
            "Mínimo em regime NORMAL",
            3,
            15,
            int(vol["min_normal"]),
        )
        vol["min_ruim"] = st.slider(
            "Mínimo em regime RUIM",
            1,
            10,
            int(vol["min_ruim"]),
        )
        vol["min_respiratorio"] = st.slider(
            "Volume mínimo respiratório (independente do regime)",
            1,
            10,
            int(vol["min_respiratorio"]),
        )

    with st.expander("🔹 BLOCO 4 — Regras do Modo 6 Premium", expanded=False):
        m6 = params["modo6"]
        m6["limite_kstar"] = st.slider(
            "Limite de k* para permitir Modo 6 Premium",
            0.10,
            0.80,
            float(m6["limite_kstar"]),
            0.05,
        )
        m6["limite_nr_percent"] = st.slider(
            "Limite de NR% para permitir Modo 6 Premium",
            20.0,
            90.0,
            float(m6["limite_nr_percent"]),
            1.0,
        )
        m6["limite_divergencia"] = st.slider(
            "Limite de Divergência S6 vs MC",
            2.0,
            10.0,
            float(m6["limite_divergencia"]),
            0.5,
        )

    with st.expander("🔹 BLOCO 5 — Parâmetros de Confiabilidade", expanded=False):
        conf = params["confiabilidade"]
        conf["minima_global"] = st.slider(
            "Confiabilidade mínima global",
            0.05,
            0.50,
            float(conf["minima_global"]),
            0.01,
        )
        conf["reforco_janela_ouro"] = st.slider(
            "Reforço em janela de ouro",
            0.00,
            0.40,
            float(conf["reforco_janela_ouro"]),
            0.02,
        )
        conf["penalidade_ruim"] = st.slider(
            "Penalidade em regime RUIM",
            0.00,
            0.50,
            float(conf["penalidade_ruim"]),
            0.02,
        )

    st.session_state["v16_parametros"] = params

    _v16_exibir_bloco(
        "Parâmetros V16 atualizados",
        "Os ajustes acima serão usados pela Engine Adaptativa, Execução V16 e Modo 6 Premium.",
        tipo="success",
    )


# ============================================================
# BLOCO 5/7 — Execução V16 Adaptativa
# ============================================================

def _painel_v16_execucao() -> None:
    st.markdown("## 🎯 V16 — Execução Adaptativa")

    df = _v16_obter_historico()
    if df is None:
        _v16_exibir_bloco(
            "Histórico não encontrado",
            "Carregue o histórico e execute os módulos V15.7 antes de usar a Execução V16.",
            tipo="warning",
        )
        return

    engine = st.session_state.get("v16_engine_info")
    if not engine:
        _v16_exibir_bloco(
            "Engine V16 não preparada",
            "Rode primeiro o painel **⚙️ V16 — Engine Adaptativa**.",
            tipo="warning",
        )
        return

    resultado = _v16_executar_selecao_listas_v16(df)
    if not resultado:
        _v16_exibir_bloco(
            "Listas indisponíveis",
            "O V16 precisa do **Modo 6 clássico** e da **última previsão** (TURBO++ ULTRA) "
            "para montar a Execução Adaptativa.",
            tipo="warning",
        )
        return

    c1 = resultado["C1"]
    c2 = resultado["C2"]
    c3 = resultado["C3"]

    st.markdown("### 🔮 Núcleo (C1) — Herdado do TURBO++ ULTRA")
    st.success(", ".join(str(x) for x in c1))

    st.markdown("### 🛡️ Coberturas (C2/C3) — Seleção Adaptativa V16")
    st.markdown(f"- Quantidade total de listas aplicadas: **{len(resultado['todas_listas'])}**")

    st.markdown("#### C2 — Coberturas mais aderentes ao núcleo")
    for i, lst in enumerate(c2, 1):
        st.markdown(f"**C2-{i:02d})** " + ", ".join(str(x) for x in lst))

    st.markdown("#### C3 — Coberturas complementares / respiração estatística")
    for i, lst in enumerate(c3, 1):
        st.markdown(f"**C3-{i:02d})** " + ", ".join(str(x) for x in lst))

    _v16_exibir_bloco(
        "Observação",
        (
            "A Execução V16 **não cria listas do zero**. Ela organiza e filtra o universo "
            "gerado pelo Modo 6 clássico, respeitando o volume adaptativo, o regime, a "
            "confiabilidade e a dinâmica do alvo."
        ),
        tipo="info",
    )


# ============================================================
# BLOCO 6/7 — Modo 6 Premium V16
# ============================================================

def _painel_v16_modo6_premium() -> None:
    st.markdown("## 🎯 V16 — Modo 6 Premium")

    engine = st.session_state.get("v16_engine_info")
    microdiag = st.session_state.get("v16_microdiag")
    exec_v16 = st.session_state.get("v16_execucao")
    params = _v16_obter_parametros_padrao()

    if not engine or not microdiag or not exec_v16:
        _v16_exibir_bloco(
            "Ambiente V16 incompleto",
            "Certifique-se de ter executado: **Microjanelas**, **Engine V16** e **Execução V16 Adaptativa** antes deste painel.",
            tipo="warning",
        )
        return

    status = _v16_classificar_modo6(engine, params)

    if status == "recomendado":
        cor = "success"
        rotulo = "🟢 Modo 6 Premium — RECOMENDADO"
        texto = (
            "O ambiente atual é **favorável** para uso do Modo 6 Premium, com janela "
            "favorável ou de ouro, confiabilidade elevada e métricas de risco sob controle. "
            "O sistema pode operar com intensidade coerente ao volume V16."
        )
    elif status == "moderado":
        cor = "warning"
        rotulo = "🟡 Modo 6 Premium — MODERADO"
        texto = (
            "O ambiente exige **cautela**. O Modo 6 Premium pode ser utilizado, mas recomenda-se "
            "apenas uso parcial das listas, foco em coberturas mais aderentes ao núcleo e "
            "monitoramento contínuo de NR% e divergência."
        )
    else:
        cor = "error"
        rotulo = "🔴 Modo 6 Premium — PROIBIDO"
        texto = (
            "As condições atuais **não são adequadas** para uso do Modo 6 Premium: "
            "uma ou mais métricas (k*, NR% ou Divergência) ultrapassam os limites definidos. "
            "Recomenda-se reduzir agressividade, operar apenas com previsões respiratórias "
            "ou aguardar melhoria do ambiente."
        )

    _v16_exibir_bloco(rotulo, texto, tipo=cor)

    corpo_det = (
        f"- Regime V16: **{engine['regime']}**\n"
        f"- Confiabilidade V16: **{engine['confiabilidade']:.4f}**\n"
        f"- k*: **{engine['k_star']:.4f}** · NR%: **{engine['nr_percent']:.2f}%** · "
        f"Divergência: **{engine['divergencia']:.4f}**\n"
        f"- Janela de ouro ativa? **{'Sim' if engine['janela_ouro'] else 'Não'}**\n"
        f"- Volume final V16: **{engine['volume_final']}** listas"
    )

    _v16_exibir_bloco(
        "Detalhamento das condições do Modo 6 Premium",
        corpo_det,
        tipo="info",
    )


# ============================================================
# BLOCO 7/7 — Relatório Final Premium V16
# ============================================================

def _painel_v16_relatorio_final() -> None:
    st.markdown("## 📘 V16 — Relatório Final Premium")

    engine = st.session_state.get("v16_engine_info")
    microdiag = st.session_state.get("v16_microdiag")
    risco = st.session_state.get("diagnostico_risco") or _v16_obter_risco_basico()
    exec_v16 = st.session_state.get("v16_execucao")
    listas_m6 = st.session_state.get("modo6_listas")
    ultima_prev = st.session_state.get("ultima_previsao")

    if not engine or not microdiag:
        _v16_exibir_bloco(
            "Ambiente V16 ainda não preparado",
            (
                "Para gerar o Relatório Premium, execute ao menos:\n"
                "- 🔮 V16 — Microjanelas & Diagnóstico\n"
                "- ⚙️ V16 — Engine Adaptativa"
            ),
            tipo="warning",
        )
        return

    # -----------------------------
    # 4.1 — ESTADO DO AMBIENTE
    # -----------------------------
    st.markdown("### 📘 4.1 — Estado do Ambiente (macro + micro)")

    if engine["regime"] == "ouro":
        regime_txt = "🟢 OURO"
    elif engine["regime"] == "normal":
        regime_txt = "🟠 NORMAL"
    else:
        regime_txt = "🔴 RUIM"

    corpo_amb = (
        f"- Regime global V16: **{regime_txt}**\n"
        f"- Confiabilidade V16: **{engine['confiabilidade']:.4f}**\n"
        f"- k*: **{engine['k_star']:.4f}** · NR%: **{engine['nr_percent']:.2f}%** · "
        f"Divergência: **{engine['divergencia']:.4f}**\n"
        f"- Score da melhor microjanela: **{microdiag.get('score_melhor', 0.0):.4f}**\n"
        f"- Melhor microjanela (séries): **{microdiag.get('melhor_janela', 'N/A')}**\n"
        f"- Janela de ouro ativa? **{'Sim' if microdiag.get('janela_ouro') else 'Não'}**\n"
        f"- Dinâmica do alvo: **{engine['dinamica_alvo']}**"
    )

    _v16_exibir_bloco(
        "Resumo do Ambiente V16",
        corpo_amb,
        tipo="info",
    )

    # -----------------------------
    # 4.2 — CONFIGURAÇÕES E PARÂMETROS
    # -----------------------------
    st.markdown("### 📗 4.2 — Configurações e Parâmetros V16")

    params = _v16_obter_parametros_padrao()
    vol = params["volumes"]
    cob = params["coberturas"]

    corpo_conf = (
        f"- Teto OURO / NORMAL / RUIM: **{vol['teto_ouro']} / {vol['teto_normal']} / {vol['teto_ruim']}**\n"
        f"- Volume sugerido V16: **{engine['volume_sugerido']}**\n"
        f"- Volume final aplicado: **{engine['volume_final']}**\n"
        f"- Amplitude fina: **{engine['amplitude_fina']}**\n"
        f"- Amplitude larga (ajustada): **{engine['amplitude_larga']}**\n"
        f"- Limite total de listas (anti-zumbi V16): **{engine['limite_total_listas']}**"
    )

    _v16_exibir_bloco(
        "Configurações aplicadas pela Engine V16",
        corpo_conf,
        tipo="info",
    )

    # -----------------------------
    # 4.3 — ESTRUTURA DAS PREVISÕES
    # -----------------------------
    st.markdown("### 📙 4.3 — Estrutura das Previsões (C1 / C2 / C3)")

    if not exec_v16 or ultima_prev is None:
        _v16_exibir_bloco(
            "Execução V16 ausente",
            (
                "Não há Execução Adaptativa registrada. "
                "Rode o painel **🎯 V16 — Execução Adaptativa** para povoar C1/C2/C3."
            ),
            tipo="warning",
        )
    else:
        c1 = exec_v16.get("C1", [])
        c2 = exec_v16.get("C2", [])
        c3 = exec_v16.get("C3", [])
        todas = exec_v16.get("todas_listas", [])

        corpo_prev = (
            f"- Núcleo (C1): **{', '.join(str(x) for x in c1)}**\n"
            f"- Quantidade total de listas V16: **{len(todas)}**\n"
            f"- Coberturas C2: **{len(c2)}** listas\n"
            f"- Coberturas C3 (respiração estatística): **{len(c3)}** listas\n"
        )

        _v16_exibir_bloco(
            "Resumo da Estrutura de Previsões V16",
            corpo_prev,
            tipo="info",
        )

        st.markdown("#### C1 — Núcleo (herdado do TURBO++ ULTRA)")
        st.success(", ".join(str(x) for x in c1))

        if c2:
            st.markdown("#### C2 — Coberturas principais")
            for i, lst in enumerate(c2, 1):
                st.markdown(f"**C2-{i:02d})** " + ", ".join(str(x) for x in lst))

        if c3:
            st.markdown("#### C3 — Coberturas complementares / respiração")
            for i, lst in enumerate(c3, 1):
                st.markdown(f"**C3-{i:02d})** " + ", ".join(str(x) for x in lst))

    # -----------------------------
    # 4.4 — JUSTIFICATIVA PREMIUM
    # -----------------------------
    st.markdown("### 📕 4.4 — Justificativa Premium")

    regime = engine["regime"]
    conf = engine["confiabilidade"]
    janela_ouro = microdiag.get("janela_ouro", False)

    if regime == "ouro":
        comentario_regime = (
            "O sistema identificou um **regime OURO**, com microjanelas muito organizadas e "
            "risco relativamente controlado. Isso justifica um volume mais alto, próximo ao teto "
            "do regime, respeitando a confiabilidade."
        )
    elif regime == "normal":
        comentario_regime = (
            "O regime é **NORMAL**, com equilíbrio entre ordem e ruído. O volume foi ajustado "
            "para um nível intermediário, evitando excessos, mas mantendo intensidade suficiente "
            "para capturar oportunidades locais."
        )
    else:
        comentario_regime = (
            "O regime é **RUIM**, com ruído elevado e/ou divergência relevante. O volume foi "
            "reduzido, porém mantendo **previsões respiratórias**, de forma a não perder o pulso "
            "da estrada enquanto protege contra desperdícios."
        )

    if janela_ouro:
        comentario_ouro = (
            "Uma **janela de ouro** esteve ativa, reforçando a confiabilidade e permitindo "
            "um uso mais forte do núcleo em algumas listas."
        )
    else:
        comentario_ouro = (
            "Nenhuma janela de ouro forte foi detectada neste trecho. O sistema operou com "
            "base em janelas boas ou normais, sem alavancagem extrema."
        )

    comentario_conf = (
        f"A confiabilidade final V16 foi de **{conf:.3f}**, resultado da combinação entre "
        "índice de risco, k*, NR%, divergência e qualidade das microjanelas."
    )

    texto_final = (
        comentario_regime + "\n\n" +
        comentario_ouro + "\n\n" +
        comentario_conf + "\n\n" +
        "O Modo 6 Premium, quando liberado, atua como camada adicional de decisão, usando o "
        "mesmo núcleo do TURBO++ ULTRA e o universo de listas gerado pelo Modo 6 clássico, "
        "apenas reorganizados de forma adaptativa pelo V16."
    )

    _v16_exibir_bloco(
        "Justificativa Premium V16",
        texto_final,
        tipo="info",
    )

    _v16_exibir_bloco(
        "Conclusão",
        (
            "Este relatório **não recalcula motores** nem altera o fluxo V15.7 MAX. "
            "Ele apenas consolida, explica e justifica as decisões estatísticas da camada "
            "V16 Premium, mantendo total aderência ao Planejamento e ao Protocolo."
        ),
        tipo="success",
    )
# ============================================================
# BLOCO 8/7 — DECISÃO FINAL V16 — PACOTE DE PREVISÃO (AGORA)
# ============================================================

# Gerar as listas de previsões adaptativas baseadas no núcleo V16 e em coberturas.
# Usaremos o volume sugerido, a confiabilidade e a dinâmica do alvo para o cálculo.

def v16_gerar_previsoes_atuais(
    c1: List[int], c2: List[int], c3: List[int], regime: str, confiabilidade: float, 
    volume_sugerido: int, dinamica_alvo: Dict[str, Any], limite_max_listas: int
) -> List[List[int]]:
    """
    Função responsável por gerar as previsões com base no núcleo V16 (C1, C2, C3) e nas coberturas.
    """
    # Camada 1: Previsões principais (C1)
    previsoes_principais = [c1]  # Núcleo base

    # Camada 2: Coberturas principais (C2)
    previsoes_c2 = c2[:limite_max_listas]

    # Camada 3: Coberturas complementares (C3)
    previsoes_c3 = c3[:limite_max_listas]

    # Junta as previsões de todas as camadas
    previsoes_finais = previsoes_principais + previsoes_c2 + previsoes_c3

    return previsoes_finais

# Exemplo de chamada da função para gerar as previsões
previsoes = v16_gerar_previsoes_atuais(
    c1=[10, 16, 23, 33, 41, 45],
    c2=[11, 15, 22, 30, 35, 43],
    c3=[12, 16, 21, 28, 34, 46],
    regime="normal",
    confiabilidade=0.65,
    volume_sugerido=12,
    dinamica_alvo={"movimento": "lento"},
    limite_max_listas=12
)

# Exibindo as previsões geradas
_v16_exibir_bloco(
    "Pacote Final de Previsões (V16)",
    f"Previsões geradas: {previsoes}",
    tipo="success",
)

# ============================================================
# FIM DO BLOCO 8/7
# ============================================================
# ============================================================
# BLOCO 9/7 — APRENDIZADO PÓS-RODADA V16
# ============================================================

def v16_aprendizado_pos_rodada(
    previsoes: List[List[int]],
    resultado_real: List[int],
    regime: str,
    confiabilidade: float,
    dinamica_alvo: str,
) -> Dict[str, Any]:
    """
    Analisa o desempenho das previsões após a rodada real
    e gera ajustes automáticos para a próxima execução.
    """

    resultado_set = set(resultado_real)

    analise = {
        "acertos_max": 0,
        "melhor_lista": None,
        "media_acertos": 0,
        "ajuste_volume": 0,
        "ajuste_amplitude": 0,
        "observacao": "",
    }

    total_acertos = []

    for lista in previsoes:
        acertos = len(set(lista) & resultado_set)
        total_acertos.append(acertos)

        if acertos > analise["acertos_max"]:
            analise["acertos_max"] = acertos
            analise["melhor_lista"] = lista

    analise["media_acertos"] = round(sum(total_acertos) / max(len(total_acertos), 1), 2)

    # -----------------------------
    # LÓGICA DE APRENDIZADO
    # -----------------------------
    if analise["acertos_max"] >= 4:
        analise["ajuste_volume"] = -2
        analise["ajuste_amplitude"] = -1
        analise["observacao"] = "Alvo parcialmente capturado. Reduzindo dispersão."
    elif analise["acertos_max"] == 3:
        analise["ajuste_volume"] = 0
        analise["ajuste_amplitude"] = 0
        analise["observacao"] = "Zona morna. Manter estratégia."
    else:
        analise["ajuste_volume"] = +2
        analise["ajuste_amplitude"] = +2
        analise["observacao"] = "Alvo em movimento rápido. Aumentando respiração."

    # Ajustes adicionais por dinâmica do alvo
    if dinamica_alvo == "em_movimento":
        analise["ajuste_volume"] += 1

    return analise


# -----------------------------
# PAINEL — APRENDIZADO PÓS-RODADA
# -----------------------------
def painel_v16_aprendizado_pos_rodada():
    st.markdown("## 🧠 V16 — Aprendizado Pós-Rodada")

    if "v16_previsoes_finais" not in st.session_state:
        st.warning("Nenhuma previsão V16 encontrada.")
        st.stop()

    resultado_txt = st.text_input(
        "Informe o resultado real da rodada (ex: 11,16,22,30,35,43):"
    )

    if not resultado_txt:
        st.info("Digite o resultado real para ativar o aprendizado.")
        st.stop()

    try:
        resultado_real = [int(x.strip()) for x in resultado_txt.split(",")]
    except Exception:
        st.error("Formato inválido.")
        st.stop()

    info = st.session_state.get("v16_engine_info", {})

    aprendizado = v16_aprendizado_pos_rodada(
        previsoes=st.session_state["v16_previsoes_finais"],
        resultado_real=resultado_real,
        regime=info.get("regime", "normal"),
        confiabilidade=info.get("confiabilidade", 0.5),
        dinamica_alvo=info.get("dinamica_alvo", "estavel"),
    )

    st.session_state["v16_aprendizado"] = aprendizado

    _v16_exibir_bloco(
        "Resultado do Aprendizado V16",
        f"""
        🔢 Acertos máximos: {aprendizado['acertos_max']}
        📊 Média de acertos: {aprendizado['media_acertos']}
        🎯 Melhor lista: {aprendizado['melhor_lista']}
        🔄 Ajuste de volume sugerido: {aprendizado['ajuste_volume']}
        📐 Ajuste de amplitude sugerido: {aprendizado['ajuste_amplitude']}
        🧠 Observação: {aprendizado['observacao']}
        """,
        tipo="success",
    )


# ============================================================
# FIM DO BLOCO 9/7
# ============================================================
