# ============================================================
# PARTE 1/8 — INÍCIO
# ============================================================
"""PredictCars V15.7 MAX — V16 Premium
Âncora Estável (base: app_v15_7_MAX_com_orbita.py)
V8 — Ajuste Fino · Etapa 2 (Borda Qualificada, pré-Camada 4)
Arquivo único, íntegro e operacional.
"""


import streamlit as st

import math

def _pc_fmt_num(x, decimals: int = 4, nd: str = "N/D") -> str:
    """Formata número para UX (evita mostrar nan/inf cru)."""
    try:
        if x is None:
            return nd
        if isinstance(x, (int, float)):
            xf = float(x)
            if math.isnan(xf) or math.isinf(xf):
                return nd
            return f"{xf:.{decimals}f}"
        # tenta converter
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return nd
        return f"{xf:.{decimals}f}"
    except Exception:
        return nd

st.set_page_config(
    page_title="Predict Cars V15.7 MAX — V16 Premium",
    page_icon="🚗",
    layout="wide",
)



# ============================================================
# PredictCars V15.7 MAX — Âncora Estável
# (sem governança / sem fases extras / sem 'próximo passo')
# ============================================================

st.sidebar.warning("Rodando arquivo âncora: app_v15_7_MAX_com_orbita_V8_ETAPA2_BORDA_QUALIFICADA.py")
# ============================================================
# Predict Cars V15.7 MAX — V16 PREMIUM PROFUNDO
# Núcleo + Coberturas + Interseção Estatística
# Pipeline V14-FLEX ULTRA + Replay LIGHT/ULTRA + TURBO++ HÍBRIDO
# + TURBO++ ULTRA + Painel de Ruído Condicional
# + Painel de Divergência S6 vs MC + Monitor de Risco (k & k*)
# + Testes de Confiabilidade REAL + Modo 6 Acertos V15.7 MAX
# + Relatório Final COMPLETO V15.7 MAX
# Arquivo oficial: app_v15_7_MAX.py
# ============================================================
import math
import itertools
import textwrap
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# DEBUG TEMPORÁRIO — PROVA DE EXECUÇÃO DO ARQUIVO
st.sidebar.caption("🧪 DEBUG: arquivo carregado")
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ============================================================
# MÓDULO 1 — GOVERNANÇA & MIRROR (OBSERVACIONAL)
# Camada de VISIBILIDADE (read-only)
# - NÃO executa motores
# - NÃO altera comportamento
# - NÃO bloqueia operações
# - Falha silenciosa (nunca derruba o app)
# ============================================================

def _m1_collect_mirror_snapshot() -> Dict[str, Any]:
    """Coleta um snapshot read-only do estado atual (session_state + sinais básicos).
    Regra: nunca levanta exceção para o app; devolve N/D quando não existir.
    """

    ss = st.session_state

    # ------------------------------------------------------------------
    # UNIVERSO (auto-derivação) — especialmente para "Carregar Histórico (Colar)"
    # ------------------------------------------------------------------
    if (("universo_min" not in ss) or ("universo_max" not in ss)) and ss.get("historico_df") is not None:
        try:
            dfu = ss.get("historico_df")
            cols = [c for c in ["p1","p2","p3","p4","p5","p6"] if c in getattr(dfu, "columns", [])]
            if cols:
                ser = pd.to_numeric(dfu[cols].stack(), errors="coerce")
                vmin = ser.min()
                vmax = ser.max()
                if pd.notna(vmin) and pd.notna(vmax):
                    ss["universo_min"] = int(vmin)
                    ss["universo_max"] = int(vmax)
        except Exception:
            pass

    def g(key: str, default: Any = "N/D") -> Any:
        try:
            return ss.get(key, default)
        except Exception:
            return default

    # Base
    historico_df = g("historico_df", None)
    historico_ok = historico_df is not None

    # Sinais universais (quando existirem)
    n_alvo = g("n_alvo", "N/D")
    universo_min = g("universo_min", "N/D")
    universo_max = g("universo_max", "N/D")

    # Pipeline / Diagnóstico (nomes variam no arquivo; usar fallback N/D)
    pipeline_ok = bool(g("pipeline_flex_ultra_concluido", False) or g("pipeline_executado", False))
    regime = g("regime_identificado", g("regime", "N/D"))
    energia = g("energia_media", g("energia_media_estrada", "N/D"))
    volatilidade = g("volatilidade_media", "N/D")
    clusters = g("clusters_formados", "N/D")

    # Monitor de risco
    k_star = g("k_star", g("k*", "N/D"))
    nr_percent = g("nr_percent", g("nr%", "N/D"))
    divergencia = g("divergencia_s6_mc", g("divergencia", "N/D"))
    indice_risco = g("indice_risco", "N/D")
    classe_risco = g("classe_risco", "N/D")

    # Execução / TURBO / Modo 6
    turbo_tentado = bool(g("turbo_ultra_executado", False) or g("turbo_executado", False))
    turbo_bloqueado = bool(g("turbo_bloqueado", False))
    turbo_motivo = g("turbo_motivo_bloqueio", g("motivo_bloqueio", "N/D"))
    modo6_executado = bool(g("modo_6_ativo", False) or g("modo6_executado", False) or g("modo_6_executado", False))
    # Listas geradas (numérico quando possível)
    _lg = g("listas_geradas", None)
    if _lg is None:
        _lg = g("listas_finais", None)
    if _lg is None:
        _lg = g("pacote_atual", None)

    if isinstance(_lg, list):
        listas_geradas = len(_lg)
    elif isinstance(_lg, int):
        listas_geradas = _lg
    else:
        listas_geradas = "<não definido>"
    volumes_usados = g("volumes_usados", "N/D")
    estado_alvo = g("estado_alvo", "N/D")
    eco_status = g("eco_status", g("eco", "N/D"))
    dmo_status = g("estado_dmo", "N/D")

    # Rastro de navegação (quando existir)
    painel_atual = g("NAV_V157_CANONICA", "N/D")

    # Keys (para auditoria leve)
    try:
        keys = sorted([str(k) for k in ss.keys()])
    except Exception:
        keys = []

    return {
        "historico_ok": historico_ok,
        "historico_df": "definido" if historico_ok else "<não definido>",
        "n_alvo": n_alvo,
        "universo_min": universo_min,
        "universo_max": universo_max,
        "pipeline_ok": pipeline_ok,
        "regime": regime,
        "energia_media": energia,
        "volatilidade_media": volatilidade,
        "clusters": clusters,
        "k_star": k_star,
        "nr_percent": nr_percent,
        "divergencia_s6_mc": divergencia,
        "indice_risco": indice_risco,
        "classe_risco": classe_risco,
        "turbo_tentado": turbo_tentado,
        "turbo_bloqueado": turbo_bloqueado,
        "turbo_motivo": turbo_motivo,
        "modo6_executado": modo6_executado,
        "listas_geradas": "definidas" if (listas_geradas not in (None, "N/D", "<não definido>")) else "<não definido>",
        "volumes_usados": volumes_usados,
        "estado_alvo": estado_alvo,
        "eco_status": eco_status,
        "dmo_status": dmo_status,
        "painel_atual": painel_atual,
        "keys": keys,
    }


# ============================================================
# V16 — UNIVERSO (mín/max) — REGISTRO CANÔNICO
# - Objetivo: garantir que o snapshot mostre 1–50 / 1–60 etc
# - Regra: NÃO decide nada; só registra leitura.
# - Compatível com Upload e Colar.
# ============================================================

def v16_detectar_universo_do_historico(df, n_alvo=6):
    try:
        if df is None or len(df) == 0:
            return (None, None)
        cols = []
        for i in range(1, int(n_alvo) + 1):
            c = f"p{i}"
            if c in df.columns:
                cols.append(c)
        if not cols:
            cols = [c for c in df.columns if isinstance(c, str) and c.startswith("p") and c[1:].isdigit()]
            cols = sorted(cols, key=lambda x: int(x[1:]))[:int(n_alvo)]
        if not cols:
            return (None, None)

        vals = df[cols].values.ravel()
        clean = []
        for v in vals:
            if v is None:
                continue
            try:
                if isinstance(v, float) and (v != v):
                    continue
                clean.append(int(v))
            except Exception:
                continue

        if not clean:
            return (None, None)

        return (min(clean), max(clean))
    except Exception:
        return (None, None)


def v16_registrar_universo_session_state(df, n_alvo=6):
    try:
        umin, umax = v16_detectar_universo_do_historico(df, n_alvo=n_alvo)
        if umin is not None and umax is not None:
            st.session_state["universo_min"] = int(umin)
            st.session_state["universo_max"] = int(umax)
            st.session_state["universo_str"] = f"{int(umin)}–{int(umax)}"
        else:
            st.session_state.setdefault("universo_min", None)
            st.session_state.setdefault("universo_max", None)
            st.session_state.setdefault("universo_str", "N/D")
    except Exception:
        st.session_state.setdefault("universo_str", "N/D")

def _m1_classificar_estado(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Classifica estado S0–S6 (canônico) com base no snapshot.
    Regra: conservador; se faltar evidência, não avança estado.
    """

    S = "S0"
    nome = "Histórico inexistente"
    faltas: List[str] = []
    avisos: List[str] = []

    if snapshot.get("historico_ok"):
        S = "S1"
        nome = "Histórico carregado"
    else:
        faltas.append("Carregar Histórico")
        return {"estado": S, "nome": nome, "faltas": faltas, "avisos": avisos}

    if snapshot.get("pipeline_ok"):
        S = "S2"
        nome = "Pipeline consolidado"
    else:
        faltas.append("Rodar Pipeline V14-FLEX ULTRA")
        return {"estado": S, "nome": nome, "faltas": faltas, "avisos": avisos}

    # Diagnóstico completo (não exigir tudo; se houver k* e NR% já é bom indicativo)
    if snapshot.get("k_star") != "N/D" or snapshot.get("nr_percent") != "N/D":
        S = "S3"
        nome = "Diagnóstico disponível"
    else:
        avisos.append("Diagnóstico ainda parcial (k*/NR% N/D)")
        return {"estado": S, "nome": nome, "faltas": faltas, "avisos": avisos}

    if snapshot.get("turbo_tentado"):
        S = "S4"
        nome = "Sondagem de execução (TURBO)"
    else:
        avisos.append("TURBO não executado nesta sessão (permitido, mas reduz visibilidade do envelope)")

    if snapshot.get("modo6_executado"):
        S = "S5"
        nome = "Execução real (Modo 6)"
    else:
        faltas.append("Executar Modo 6")

    # Pós-execução: listas e/ou relatório já gerados na sessão (heurística conservadora)
    if snapshot.get("modo6_executado") and snapshot.get("listas_geradas") == "definidas":
        S = "S6"
        nome = "Pós-execução (Relatório / Governança)"

    return {"estado": S, "nome": nome, "faltas": faltas, "avisos": avisos}


def _m1_render_barra_estados(estado: str) -> None:
    ordem = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
    marcadores = []
    for s in ordem:
        if ordem.index(s) < ordem.index(estado):
            marcadores.append("✓")
        elif s == estado:
            marcadores.append("●")
        else:
            marcadores.append("○")
    st.write(" ".join([f"[{s}]" for s in ordem]))
    st.write(" ".join([f" {m} " for m in marcadores]))


def _m1_render_mirror_panel() -> None:
    """Painel Mirror canônico (observacional). Nunca derruba o app."""
    try:
        snapshot = _m1_collect_mirror_snapshot()
        meta = _m1_classificar_estado(snapshot)

        # MODULO 2 (infraestrutura): registrar memoria no gatilho canonico
        _m2_registrar_minimo_se_preciso(snapshot, meta)
        _m2_registrar_fechamento_se_preciso(snapshot, meta)

        st.markdown("## 🔍 Diagnóstico Espelho (Mirror)")
        st.caption("Painel somente leitura — estado real da execução · governança informativa · sem decisão")

        st.markdown("### 🧭 Estado Operacional Atual")
        st.markdown(f"**{meta['estado']} — {meta['nome']}**")
        _m1_render_barra_estados(meta["estado"])

        if meta.get("faltas"):
            st.info("Ainda não percorrido (na sessão): " + " · ".join(meta["faltas"]))
        if meta.get("avisos"):
            for a in meta["avisos"]:
                st.warning(a)

        st.markdown("---")
        st.markdown("### 📋 Snapshot (read-only)")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write({
                "historico_df": snapshot.get("historico_df"),
                "n_alvo": snapshot.get("n_alvo"),
                "universo": f"{snapshot.get('universo_min')}–{snapshot.get('universo_max')}",
                "pipeline_ok": snapshot.get("pipeline_ok"),
                "regime": snapshot.get("regime"),
            })
        with col2:
            st.write({
                "k_star": snapshot.get("k_star"),
                "nr_percent": snapshot.get("nr_percent"),
                "divergencia_s6_mc": snapshot.get("divergencia_s6_mc"),
                "indice_risco": snapshot.get("indice_risco"),
                "classe_risco": snapshot.get("classe_risco"),
            })
        with col3:
            st.write({
                "turbo_tentado": snapshot.get("turbo_tentado"),
                "turbo_bloqueado": snapshot.get("turbo_bloqueado"),
                "turbo_motivo": snapshot.get("turbo_motivo"),
                "modo6_executado": snapshot.get("modo6_executado"),
                "listas_geradas": snapshot.get("listas_geradas"),
            })

        with st.expander("🧪 Chaves do session_state (auditoria leve)"):
            st.write(snapshot.get("keys", []))


        with st.expander('🧠 M2 - Memoria de Estados (auditoria controlada)'):
            st.write(_m2_resumo_auditavel())

    except Exception as _e:
        # Falha silenciosa: não derrubar o app.
        st.warning(f"⚠️ Mirror falhou (silencioso): {_e}")


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# V16 — GUARDA CANÔNICA (ANTI-NAMEERROR) — TOPO DO ARQUIVO
# (DESATIVADA — substituída pela CAMADA D real)
# Mantida apenas como registro histórico
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# def v16_registrar_estado_alvo():
#     return {
#         "tipo": "indefinido",
#         "velocidade": "indefinida",
#         "comentario": "Estado ainda não disponível (carregue histórico e rode Sentinelas/Pipeline).",
#     }

# def v16_registrar_expectativa():
#     return {
#         "previsibilidade": "indefinida",
#         "erro_esperado": "indefinido",
#         "chance_janela_ouro": "baixa",
#         "comentario": "Expectativa ainda não disponível (carregue histórico e rode Sentinelas/Pipeline).",
#     }

# def v16_registrar_volume_e_confiabilidade():
#     return {
#         "minimo": "-",
#         "recomendado": "-",
#         "maximo_tecnico": "-",
#         "confiabilidades_estimadas": {},
#         "comentario": "Volume ainda não disponível (carregue histórico e rode Sentinelas/Pipeline).",
#     }

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# FIM — V16 — GUARDA CANÔNICA (ANTI-NAMEERROR) — DESATIVADA
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



# ============================================================
# FUNÇÃO — CARREGAMENTO UNIVERSAL DE HISTÓRICO (FLEX ULTRA)
# REGRA FIXA:
# - Último valor da linha = k
# - Quantidade de passageiros é LIVRE
# - Universo é derivado do histórico (SANIDADE)
# ============================================================

def carregar_historico_universal(linhas):
    """
    Formato esperado (exemplos válidos):
    C10;20;32;49;54;62;0
    C5790;4;5;6;23;35;43;0
    C15;01;02;03;04;05;06;07;08;09;10;1
    """

    registros = []
    universo_detectado = []

    for idx, linha in enumerate(linhas, start=1):
        linha = linha.strip()

        if not linha:
            continue

        partes = linha.split(";")

        if len(partes) < 3:
            raise ValueError(f"Linha {idx} inválida (campos insuficientes): {linha}")

        try:
            valores = partes[1:]          # ignora identificador
            k = int(valores[-1])          # último valor é k
            passageiros = [int(x) for x in valores[:-1]]
        except ValueError:
            raise ValueError(f"Linha {idx} contém valores não numéricos: {linha}")

        if not passageiros:
            raise ValueError(f"Linha {idx} sem passageiros válidos: {linha}")

        # coleta universo real
        universo_detectado.extend(passageiros)

        registro = {f"p{i+1}": p for i, p in enumerate(passageiros)}
        registro["k"] = k
        registro["serie"] = idx

        registros.append(registro)

    if not registros:
        raise ValueError("Histórico vazio ou inválido.")

    df = pd.DataFrame(registros)

    # ------------------------------------------------------------
    # SANIDADE DO UNIVERSO — CANÔNICA (MIN e MAX REAIS)
    # ------------------------------------------------------------
    try:
        universo_min = int(min(universo_detectado))
        universo_max = int(max(universo_detectado))
        st.session_state["universo_min"] = universo_min
        st.session_state["universo_max"] = universo_max
    except Exception:
        st.session_state["universo_min"] = None
        st.session_state["universo_max"] = None

    return df


# ============================================================
# V16 PREMIUM — IMPORTAÇÃO OFICIAL
# (Não altera nada do V15.7, apenas registra os painéis novos)
# ============================================================

from app_v16_premium import (
    v16_obter_paineis,
    v16_renderizar_painel,
)

# ============================================================
# Configuração da página (obrigatório V15.7 MAX)
# ============================================================

# ============================================================
# V16 — CAMADA ORBITA (E1) + GRADIENTE (G0–G3) + N_EXTRA
# (sem interceptação automática; sem travas; sem painel novo)
# ============================================================

def v16_orbita__interseccao(a, b):
    return len(set(a).intersection(set(b)))

def v16_orbita__pares_interseccao(listas):
    # retorna contagem de pares com intersecção >=2 e >=3
    if not listas or len(listas) < 2:
        return {"pares_total": 0, "pares_ge2": 0, "pares_ge3": 0}
    pares_total = 0
    ge2 = 0
    ge3 = 0
    for i in range(len(listas)):
        for j in range(i+1, len(listas)):
            pares_total += 1
            inter = v16_orbita__interseccao(listas[i], listas[j])
            if inter >= 2:
                ge2 += 1
            if inter >= 3:
                ge3 += 1
    return {"pares_total": pares_total, "pares_ge2": ge2, "pares_ge3": ge3}

def v16_calcular_orbita_pacote(listas_topN, universo_min, universo_max):
    """Calcula ORBITA_E0/E1 + métricas (f_max, range_8, pares>=2/3).
    Não altera listas; apenas descreve o pacote.
    """
    info = {
        "estado": "E0",
        "selo": "E0",
        "f_max": 0.0,
        "range_8": None,
        "range_lim": None,
        "pares_ge2": 0.0,
        "pares_ge3": 0.0,
        "ancoras": [],
        "top_passageiros": [],
        "listas_top": [],
    }
    try:
        if not listas_topN:
            return info

        info["listas_top"] = listas_topN

        # Frequências por passageiro
        from collections import Counter
        flat = [p for lst in listas_topN for p in lst]
        if not flat:
            return info
        c = Counter(flat)
        top_pass = [p for p, _ in c.most_common(12)]
        info["top_passageiros"] = top_pass

        # f_max normalizado por N (em quantas listas aparece o passageiro mais recorrente)
        # Atenção: usamos presença por lista (não contagem bruta).
        pres = Counter()
        for lst in listas_topN:
            for p in set(lst):
                pres[p] += 1
        if not pres:
            return info
        f_max = max(pres.values()) / float(len(listas_topN))
        info["f_max"] = float(round(f_max, 4))

        # âncoras: passageiros com presença >= 50% no pacote TopN
        ancoras = [p for p, v in pres.items() if (v / float(len(listas_topN))) >= 0.50]
        ancoras = sorted(ancoras)[:10]
        info["ancoras"] = ancoras

        # compressão de faixa (Top8 por frequência bruta)
        top8 = [p for p, _ in c.most_common(8)]
        if top8:
            r8 = max(top8) - min(top8)
            info["range_8"] = int(r8)
        else:
            info["range_8"] = None

        # limite de compressão depende do universo
        universo_size = int(universo_max) - int(universo_min) + 1
        lim = int(round(universo_size * 0.44))  # ~22 em 1–50, ~26 em 1–60
        info["range_lim"] = lim

        # coerência de interseção
        pares = v16_orbita__pares_interseccao(listas_topN)
        if pares["pares_total"] > 0:
            info["pares_ge2"] = float(round(pares["pares_ge2"] / pares["pares_total"], 4))
            info["pares_ge3"] = float(round(pares["pares_ge3"] / pares["pares_total"], 4))

        # decisão E1 (quase-órbita) — criteriosa mas sem "freio" no operador:
        # - f_max em zona de quase-âncora (0.35..0.70)
        # - range_8 comprimido (<= lim)
        # - pares>=2 moderado (>= 0.35)
        # - pares>=3 não explosivo (<= 0.35)
        passa_f = (info["f_max"] >= 0.35) and (info["f_max"] <= 0.70)
        passa_range = (info["range_8"] is not None) and (info["range_8"] <= lim)
        passa_ge2 = (info["pares_ge2"] >= 0.35)
        passa_ge3 = (info["pares_ge3"] <= 0.35)

        if passa_f and passa_range and passa_ge2 and passa_ge3:
            info["estado"] = "E1"
            info["selo"] = "E1"
        else:
            info["estado"] = "E0"
            info["selo"] = "E0"

        return info
    except Exception:
        # falha silenciosa: não derruba app
        return info

def v16_calcular_gradiente_E1(info_orbita):
    """Retorna G0..G3.
    G0: E0 (mar aberto)
    G1: E1 fraco
    G2: E1 consistente
    G3: E1 comprimido (quase E2)
    """
    try:
        if not info_orbita or info_orbita.get("estado") != "E1":
            return {"gradiente": "G0", "score": 0.0}

        f = float(info_orbita.get("f_max") or 0.0)
        ge2 = float(info_orbita.get("pares_ge2") or 0.0)
        ge3 = float(info_orbita.get("pares_ge3") or 0.0)
        r8 = info_orbita.get("range_8")
        lim = info_orbita.get("range_lim") or 1

        # componentes normalizados
        # f ideal ~0.55 (meio termo)
        f_score = 1.0 - min(1.0, abs(f - 0.55) / 0.20)  # tolerância 0.20
        # range: quanto menor que lim, melhor
        if r8 is None:
            r_score = 0.0
        else:
            r_score = 1.0 - min(1.0, max(0.0, (r8 / float(lim)) - 0.75) / 0.75)  # bom até 0.75*lim
        # ge2: quanto maior, melhor (até 0.85)
        ge2_score = min(1.0, ge2 / 0.85)
        # ge3: penaliza explosão de iguais
        ge3_pen = min(1.0, max(0.0, (ge3 - 0.15) / 0.35))

        score = (0.35*f_score + 0.35*r_score + 0.30*ge2_score) * (1.0 - 0.35*ge3_pen)
        score = float(round(max(0.0, min(1.0, score)), 4))

        if score >= 0.78:
            g = "G3"
        elif score >= 0.60:
            g = "G2"
        else:
            g = "G1"

        return {"gradiente": g, "score": score}
    except Exception:
        return {"gradiente": "G0", "score": 0.0}

def v16_calcular_N_extra(estado_orbita, gradiente, n_base, eco_forca=None, eco_acionabilidade=None):
    """Expansão condicional do pacote.
    - Não divide pacote (mantém N_BASE intacto)
    - Apenas adiciona N_EXTRA quando justificável
    - Sem travar operador (apenas informa + gera listas)
    """
    try:
        n_base = int(n_base or 0)
        if n_base <= 0:
            return 0

        # qualificador ECO (se disponível)
        eco_ok = True
        if eco_acionabilidade is not None:
            eco_ok = (str(eco_acionabilidade).lower() != "não_acionável") and (str(eco_acionabilidade).lower() != "nao_acionavel")
        # se não existe ECO, não bloqueia

        if not eco_ok and estado_orbita != "E2":
            return 0

        if estado_orbita == "E2":
            return int(max(2, min(8, round(0.5 * n_base))))
        if gradiente == "G3":
            return int(max(2, min(6, round(0.3 * n_base))))
        if gradiente == "G2":
            return int(max(1, min(4, round(0.2 * n_base))))
        return 0
    except Exception:
        return 0

def v16_gerar_listas_extra_por_orbita(info_orbita, universo_min, universo_max, n_carro, qtd, seed=0):
    """Gera listas extras (N_EXTRA) com viés de interseção/âncoras.
    Sem interceptação automática: é só expansão condicional do pacote.
    """
    try:
        import random
        rnd = random.Random(int(seed or 0) + 991)
        universo = list(range(int(universo_min), int(universo_max)+1))

        ancoras = list(info_orbita.get("ancoras") or [])
        top = list(info_orbita.get("top_passageiros") or [])
        base_pool = [p for p in (ancoras + top) if (p in universo)]
        base_pool = list(dict.fromkeys(base_pool))  # unique preserve order

        listas = []
        alvo_anchors = min(3, max(1, len(base_pool)//3)) if base_pool else 0

        for _ in range(int(qtd or 0)):
            lst = []
            # fixa 2–3 âncoras/top
            if base_pool:
                kfix = min(alvo_anchors + rnd.randint(0, 1), max(1, min(3, len(base_pool))))
                lst.extend(rnd.sample(base_pool, kfix))
            # completa aleatório do universo, evitando duplicatas
            while len(lst) < int(n_carro):
                p = rnd.choice(universo)
                if p not in lst:
                    lst.append(p)
            lst = sorted(lst)
            listas.append(lst)

        # remove duplicatas exatas
        uniq = []
        seen = set()
        for l in listas:
            t = tuple(l)
            if t not in seen:
                seen.add(t)
                uniq.append(l)
        return uniq
    except Exception:
        return []



# ============================================================
# ============================================================
# V16 — CAMADA REMOVIDA (âncora estável)
# Motivo: este arquivo âncora opera sem camadas experimentais.
# ============================================================

# ============================================================
# V16 — APS (Auditoria de Postura do Sistema) — Observacional
# ============================================================
def v16_calcular_aps_postura(nr_percent=None, orbita_selo=None, eco_acionabilidade=None, anti_exato_level=None):
    """APS = Auditoria de Postura do Sistema.
    - Observacional: NÃO muda listas, NÃO decide volume.
    - Classifica risco/postura e sugere forma compatível (denso/espalhado/duplo pacote) sem impor.
    """
    try:
        nr = float(nr_percent) if nr_percent is not None else None
    except Exception:
        nr = None

    selo = (orbita_selo or "E0").strip()
    eco = (eco_acionabilidade or "N/D").strip()

    # Anti-exato: "alto"/"médio"/"baixo" ou None
    ae = (str(anti_exato_level).strip().lower() if anti_exato_level is not None else "")

    # Regras deliberadamente conservadoras
    if (nr is not None) and (nr >= 75):
        return ("🔴", "Postura Crítica", "Ruído crítico (NR alto). Evitar ancoragem forte; preferir pacote espalhado e baixo volume. Observação: agir com cautela.")
    if (selo.startswith("E0")) and (nr is not None) and (nr >= 55):
        return ("🟡", "Postura Sensível", "E0 + ruído alto: ancoragem excessiva é perigosa. Preferir duplo pacote (base + anti-âncora) sem aumentar universo.")
    if (selo.startswith("E1") or selo.startswith("E2")) and (nr is not None) and (nr <= 55):
        return ("🟢", "Postura Operável", "Órbita emergente com ruído sob controle. Densidade moderada pode ser compatível; manter governança e observar persistência.")
    if eco.lower() in ("acionável", "acionavel") and (nr is not None) and (nr <= 60):
        return ("🟢", "Postura Operável", "ECO acionável com ruído aceitável. Operar com disciplina: microvariações/envelope e testes de consistência.")
    # fallback
    return ("⚪", "Postura Neutra", "Sem evidência suficiente para postura ativa. Manter pacote base e acompanhar série a série (detecção/sensibilidade/gradiente).")


# ============================================================
# V16 — SINCRONIZAÇÃO CANÔNICA (ALIASES) + ANTI-ÂNCORA (OBS)
# ============================================================

def v16_sync_aliases_canonicos(force: bool = False) -> dict:
    """Sincroniza variáveis canônicas usadas no Relatório Final / Registro.
    - Observacional: NÃO altera motores, NÃO altera listas, NÃO decide.
    - Objetivo: evitar N/D indevido quando dados existem sob chaves antigas/alternativas.
    """
    mudancas = {}

    # k* (sentinela)
    if force or (st.session_state.get("k_star") is None):
        if st.session_state.get("sentinela_kstar") is not None:
            st.session_state["k_star"] = st.session_state.get("sentinela_kstar")
            mudancas["k_star<=sentinela_kstar"] = True

    # Divergência S6 vs MC
    if force or (st.session_state.get("divergencia_s6_mc") is None):
        if st.session_state.get("div_s6_mc") is not None:
            st.session_state["divergencia_s6_mc"] = st.session_state.get("div_s6_mc")
            mudancas["divergencia_s6_mc<=div_s6_mc"] = True

    # ECO/Estado (painel V16 mastigado)
    diag = st.session_state.get("diagnostico_eco_estado_v16")
    if isinstance(diag, dict):
        eco_txt = f"{diag.get('eco_forca','')} · {diag.get('eco_persistencia','')} · {diag.get('eco_acionabilidade','')}"
        eco_txt = eco_txt.strip(" ·").strip()
        if eco_txt and (force or (not st.session_state.get("eco_status")) or (st.session_state.get("eco_status") in ("N/D", "DESCONHECIDO"))):
            st.session_state["eco_status"] = eco_txt
            mudancas["eco_status<=diagnostico_eco_estado_v16"] = True

        est_txt = str(diag.get("estado") or "").strip()
        if est_txt and (force or (not st.session_state.get("estado_atual")) or (st.session_state.get("estado_atual") in ("N/D", "DESCONHECIDO"))):
            st.session_state["estado_atual"] = est_txt
            mudancas["estado_atual<=diagnostico_eco_estado_v16"] = True

    return mudancas


def v16_analisar_duplo_pacote_base_anti_ancora(
    listas: list,
    base_n: int = 10,
    max_anti: int = 4,
    core_presenca_min: float = 0.60,
) -> dict:
    """Analisa ancoragem do pacote (OBSERVACIONAL).

    - Define um 'CORE' do pacote base (Top N) por presença em listas.
    - Mede overlap (0..len(core)) de cada lista com o CORE.
    - Sugere (sem impor) quais listas existentes podem servir como 'anti-âncora':
        listas com overlap baixo com o CORE (mas ainda dentro do mesmo universo).

    Retorna um dict com:
    - core: lista de passageiros do CORE
    - overlaps: lista de overlaps por lista (índice alinhado a 'listas')
    - base_idx: índices (1-based) do pacote base
    - anti_idx: índices (1-based) sugeridos como anti-âncora (existentes)
    """
    try:
        if not listas or not isinstance(listas, list):
            return {"core": [], "overlaps": [], "base_idx": [], "anti_idx": [], "nota": "sem_listas"}

        base_n = int(base_n or 0)
        base_n = max(1, min(base_n, len(listas)))
        base = listas[:base_n]

        # Presença por lista (não contagem bruta)
        from collections import Counter

        pres = Counter()
        for L in base:
            if not isinstance(L, (list, tuple)):
                continue
            for p in set(int(x) for x in L):
                pres[p] += 1

        if not pres:
            return {"core": [], "overlaps": [0 for _ in listas], "base_idx": list(range(1, base_n + 1)), "anti_idx": [], "nota": "sem_presenca"}

        # CORE: passageiros com presença >= core_presenca_min no pacote base.
        core = [p for p, v in pres.items() if (v / float(base_n)) >= float(core_presenca_min)]
        core = sorted(core)

        # fallback: se core ficou vazio, usa TOP3 por presença (mais conservador)
        if not core:
            core = [p for p, _ in pres.most_common(3)]
            core = sorted(core)

        core_set = set(core)

        overlaps = []
        for L in listas:
            if not isinstance(L, (list, tuple)):
                overlaps.append(0)
                continue
            overlaps.append(len(set(int(x) for x in L).intersection(core_set)))

        # Anti-âncora: fora do TopN (preferência), overlap baixo com CORE.
        # Threshold: <=1 quando core >=3, senão <=0.
        thr = 1 if len(core) >= 3 else 0

        candidatos = []
        for idx in range(base_n, len(listas)):
            ov = overlaps[idx]
            if ov <= thr:
                candidatos.append((ov, idx))

        # ordena por overlap menor e por índice estável
        candidatos = sorted(candidatos, key=lambda t: (t[0], t[1]))
        anti_idx = [i + 1 for _, i in candidatos[:max(0, int(max_anti))]]

        return {
            "core": core,
            "overlaps": overlaps,
            "base_idx": list(range(1, base_n + 1)),
            "anti_idx": anti_idx,
            "thr": thr,
            "base_n": base_n,
            "max_anti": int(max_anti),
            "core_presenca_min": float(core_presenca_min),
        }
    except Exception:
        return {"core": [], "overlaps": [], "base_idx": [], "anti_idx": [], "nota": "falha_silenciosa"}




# ============================================================
# V16 — DIAGNÓSTICO: RIGIDEZ DO "JEITÃO" (OBSERVACIONAL)
# - NÃO decide, NÃO altera listas, NÃO muda volume.
# - Objetivo: detectar quando o pacote está "rigidamente preso" a um jeitão
#   (ex.: concentração alta em faixa/âncoras) e sugerir apenas UMA "folga"
#   (folga qualitativa) como alerta diagnóstico — não como decisão.
# ============================================================

def v16_diagnostico_rigidez_jeitao(
    listas: list,
    universo_min: int = None,
    universo_max: int = None,
    base_n: int = 10,
    core_presenca_min: float = 0.60,
) -> dict:
    """
    Retorna:
      - rigido (bool)
      - score (0..1)
      - folga_qualitativa (str)   # apenas diagnóstico (sem prescrição numérica)
      - sinais (dict)           # métricas usadas
      - mensagem (str)
    """
    try:
        if not listas or not isinstance(listas, list):
            return {"rigido": False, "score": 0.0, "folga_qualitativa": "nenhuma", "sinais": {"motivo": "sem_listas"}, "mensagem": "Sem listas para diagnóstico."}

        base_n = int(base_n or 0)
        base_n = max(3, min(base_n, len(listas)))
        topN = listas[:base_n]

        # reaproveita lógica de CORE + overlaps
        anti = v16_analisar_duplo_pacote_base_anti_ancora(
            listas=listas,
            base_n=base_n,
            max_anti=4,
            core_presenca_min=float(core_presenca_min),
        )
        core = anti.get("core") or []
        overlaps = anti.get("overlaps") or []
        anti_idx = anti.get("anti_idx") or []

        core_sz = len(core)
        if core_sz <= 0 or not overlaps:
            return {"rigido": False, "score": 0.0, "folga_qualitativa": "nenhuma", "sinais": {"core_sz": core_sz, "motivo": "core_indisponivel"}, "mensagem": "CORE indisponível — diagnóstico de rigidez não aplicado."}

        # overlap médio e proporção de listas muito coladas no CORE
        ov_mean = float(sum([o for o in overlaps if isinstance(o, (int, float))]) / max(1, len(overlaps)))
        # "muito colado": overlap >= core_sz - 1 (quando core>=3), senão overlap==core_sz
        if core_sz >= 3:
            thr_colado = core_sz - 1
        else:
            thr_colado = core_sz
        colados = 0
        for o in overlaps:
            try:
                if int(o) >= int(thr_colado):
                    colados += 1
            except Exception:
                pass
        frac_colados = float(colados / max(1, len(overlaps)))

        # métricas de faixa / âncoras via órbita (se universo estiver disponível)
        umin = universo_min
        umax = universo_max
        if (umin is None or umax is None):
            try:
                umin = st.session_state.get("universo_min")
                umax = st.session_state.get("universo_max")
            except Exception:
                umin, umax = None, None

        orb = {}
        if (umin is not None) and (umax is not None):
            orb = v16_calcular_orbita_pacote(topN, int(umin), int(umax))
        f_max = float(orb.get("f_max") or 0.0)
        range_8 = orb.get("range_8")
        range_lim = orb.get("range_lim")

        # score de rigidez (conservador)
        score = 0.0

        # 1) colagem no CORE pesa muito
        score += 0.45 * min(1.0, frac_colados / 0.80)  # saturação em 80%

        # 2) overlap médio alto (normalizado por core_sz)
        score += 0.25 * min(1.0, (ov_mean / max(1.0, float(core_sz))) / 0.85)

        # 3) f_max alto = ancoragem forte (se disponível)
        if f_max > 0.0:
            score += 0.20 * min(1.0, max(0.0, (f_max - 0.45) / 0.35))  # acima de ~0.45 começa pesar

        # 4) faixa top8 comprimida (se disponível)
        if (range_8 is not None) and (range_lim is not None) and (range_lim > 0):
            # quanto menor a faixa vs limite, mais rígido
            comp = 1.0 - min(1.0, float(range_8) / float(range_lim))
            score += 0.10 * max(0.0, comp)

        score = float(round(max(0.0, min(1.0, score)), 4))

        # rigidez: score >= 0.62 (limiar deliberadamente conservador)
        rigido = score >= 0.62

        # folga qualitativa (diagnóstico, não decisão)
        folga_qual = "nenhuma"
        if rigido:
            if score >= 0.85:
                folga_qual = "moderada"
                msg = (
                    "Jeitão **muito rígido**: pode haver compressão excessiva. "
                    "Diagnóstico sugere **folga moderada** (alerta, não decisão)."
                )
            else:
                folga_qual = "mínima"
                msg = (
                    "Jeitão **rígido**: pode haver compressão excessiva. "
                    "Diagnóstico sugere **folga mínima** (alerta, não decisão)."
                )
        else:
            msg = (
                "Jeitão **não aparenta rigidez excessiva** (ou há folga/anti-âncora suficiente)."
            )
        sinais = {
            "core_sz": core_sz,
            "ov_mean": round(ov_mean, 4),
            "frac_colados": round(frac_colados, 4),
            "f_max": round(f_max, 4) if f_max is not None else None,
            "range_8": range_8,
            "range_lim": range_lim,
            "anti_idx_detectados": anti_idx,
        }

        return {"rigido": rigido, "score": score, "folga_qualitativa": folga_qual, "sinais": sinais, "mensagem": msg}

    except Exception:
        return {"rigido": False, "score": 0.0, "folga_qualitativa": "nenhuma", "sinais": {"motivo": "falha_silenciosa"}, "mensagem": "Falha silenciosa no diagnóstico de rigidez."}



# ============================================================
# V8 — AJUSTE FINO · ETAPA 2 — BORDA QUALIFICADA (PRÉ-CAMADA 4)
# - Observacional / governança legível
# - NÃO altera Modo 6 / TURBO / Bala Humano
# - NÃO decide ataque / volume
# - Classifica "borda interna" vs "borda externa" a partir do pacote (Top N)
# ============================================================

def v8_classificar_borda_qualificada(
    listas: list,
    base_n: int = 10,
    core_presenca_min: float = 0.60,
    quase_delta: float = 0.12,
    max_borda_interna: int = 6,
    universo_min: int = None,
    universo_max: int = None,
    rigidez_info: dict = None,
) -> dict:
    """V8 — Borda Qualificada (Etapa 2).
    Retorna um dict com:
      - core (lista)
      - quase_core (lista)
      - borda_interna (lista) + motivos
      - borda_externa (lista) + motivos
      - meta (base_n, thresholds, rigidez, etc.)
    """
    try:
        if not listas or not isinstance(listas, list):
            return {
                "core": [],
                "quase_core": [],
                "borda_interna": [],
                "borda_externa": [],
                "candidatos": [],
                "meta": {"status": "sem_listas"},
            }

        # base_n conservador
        base_n = int(base_n or 0)
        base_n = max(3, min(base_n, len(listas)))
        base = listas[:base_n]

        # universo (se não vier explícito, tenta session_state)
        umin = universo_min
        umax = universo_max
        if (umin is None or umax is None):
            try:
                umin = st.session_state.get("universo_min")
                umax = st.session_state.get("universo_max")
            except Exception:
                umin, umax = None, None

        # presença por lista (não contagem bruta)
        from collections import Counter
        pres = Counter()
        for lst in base:
            try:
                for p in set(lst):
                    pres[int(p)] += 1
            except Exception:
                continue

        if not pres:
            return {
                "core": [],
                "quase_core": [],
                "borda_interna": [],
                "borda_externa": [],
                "candidatos": [],
                "meta": {"status": "presenca_vazia", "base_n": base_n},
            }

        # thresholds
        cmin = float(core_presenca_min)
        delta = float(quase_delta)
        thr_quase_min = max(0.0, cmin - delta)

        # CORE e QUASE-CORE (por presença)
        core = []
        quase = []
        ratios = {}
        for p, v in pres.items():
            r = float(v) / float(base_n)
            ratios[int(p)] = float(round(r, 4))
            if r >= cmin:
                core.append(int(p))
            elif r >= thr_quase_min:
                quase.append(int(p))

        core = sorted(core)
        quase = sorted(quase)

        # rigidez (se não vier, tenta calcular)
        rig = rigidez_info or {}
        try:
            if not rig:
                rig = v16_diagnostico_rigidez_jeitao(
                    listas=listas,
                    universo_min=umin,
                    universo_max=umax,
                    base_n=base_n,
                    core_presenca_min=float(core_presenca_min),
                )
        except Exception:
            rig = rigidez_info or {}

        rigido = bool(rig.get("rigido", False))
        score_rig = rig.get("score", 0.0)
        folga_qual = rig.get("folga_qualitativa", "nenhuma")

        # geometria simples do CORE (para decidir borda interna vs externa)
        core_min = min(core) if core else None
        core_max = max(core) if core else None
        universo_size = None
        try:
            if (umin is not None) and (umax is not None):
                universo_size = int(umax) - int(umin) + 1
        except Exception:
            universo_size = None

        candidatos = []
        for p in quase:
            r = ratios.get(int(p), 0.0)
            motivos = []

            # 1) presença (sempre)
            motivos.append(f"quase-CORE por presença ({int(round(r*100))}%)")

            # 2) rigidez / compressão (condicional)
            if rigido:
                motivos.append(f"jeitão rígido (score {score_rig})")
                if str(folga_qual).strip().lower() in ("mínima", "minima", "moderada"):
                    motivos.append(f"folga {folga_qual} (alerta)")
            else:
                motivos.append("jeitão não rígido (diagnóstico)")

            # 3) distância do CORE (se disponível)
            dist = None
            if core_min is not None and core_max is not None:
                if int(p) < int(core_min):
                    dist = int(core_min) - int(p)
                elif int(p) > int(core_max):
                    dist = int(p) - int(core_max)
                else:
                    dist = 0
                motivos.append(f"distância do CORE: {dist}")

            # 4) sanidade: evitar "borda externa" que abre universo demais (heurística conservadora)
            externo = False
            if universo_size is not None and dist is not None:
                # se o candidato estiver muito fora do "miolo" do CORE em proporção ao universo, tende a dispersar
                if dist >= int(round(0.28 * float(universo_size))):
                    externo = True
                    motivos.append("muito distante do CORE (risco de dispersão)")

            # 5) qualificador final: "interna" exige presença alta dentro do quase-CORE + compatibilidade com rigidez
            # (não é regra de ataque; é governança legível)
            interna = False
            if not externo:
                if r >= (cmin - (delta * 0.50)):
                    # se jeitão rígido, prioriza borda interna; se não rígido, exige ainda mais presença
                    if rigido:
                        interna = True
                        motivos.append("classificada como BORDA INTERNA (rigidez + presença alta)")
                    else:
                        if r >= (cmin - (delta * 0.25)):
                            interna = True
                            motivos.append("classificada como BORDA INTERNA (presença muito alta)")
                        else:
                            motivos.append("presença boa, mas não suficiente para interna sem rigidez")
                else:
                    motivos.append("presença insuficiente para BORDA INTERNA (fica como externa/observacional)")

            classe = "borda_interna" if interna else "borda_externa"
            candidatos.append({
                "p": int(p),
                "ratio": r,
                "classe": classe,
                "motivos": motivos,
            })

        # ordena candidatos por ratio desc, depois por p
        candidatos = sorted(candidatos, key=lambda d: (-float(d.get("ratio") or 0.0), int(d.get("p") or 0)))

        borda_interna = [d["p"] for d in candidatos if d.get("classe") == "borda_interna"][:int(max_borda_interna)]
        borda_externa = [d["p"] for d in candidatos if d.get("classe") == "borda_externa"]

        # motivos por grupo (compacto)
        motivos_in = {d["p"]: d["motivos"] for d in candidatos if d.get("classe") == "borda_interna" and d.get("p") in borda_interna}
        motivos_ex = {d["p"]: d["motivos"] for d in candidatos if d.get("classe") == "borda_externa"}

        return {
            "core": core,
            "quase_core": quase,
            "borda_interna": borda_interna,
            "borda_externa": borda_externa,
            "motivos_interna": motivos_in,
            "motivos_externa": motivos_ex,
            "candidatos": candidatos,
            "meta": {
                "status": "ok",
                "base_n": base_n,
                "core_presenca_min": cmin,
                "quase_delta": delta,
                "thr_quase_min": round(thr_quase_min, 4),
                "rigido": rigido,
                "score_rigidez": score_rig,
                "folga_qualitativa": folga_qual,
                "universo_min": umin,
                "universo_max": umax,
            },
        }

    except Exception:
        return {
            "core": [],
            "quase_core": [],
            "borda_interna": [],
            "borda_externa": [],
            "candidatos": [],
            "meta": {"status": "falha_silenciosa"},
        }


# ============================================================
# Estilos globais — preservando jeitão V14-FLEX + V15.6 MAX
# ============================================================
st.markdown(
    """
    <style>
    .big-title { font-size: 32px; font-weight: bold; }
    .sub-title { font-size: 22px; font-weight: bold; margin-top: 25px; }
    .danger { color: red; font-weight: bold; }
    .success { color: green; font-weight: bold; }
    .warning { color: orange; font-weight: bold; }
    .gray-text { color: #888; }
    .info-box {
        background-color: #f5f5f5;
        padding: 10px 15px;
        border-left: 4px solid #4c8bf5;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# BLINDAGEM FINAL — SANIDADE DE UNIVERSO (V16)
# Aplica automaticamente o universo real do histórico
# em qualquer lista de previsão antes do uso operacional
# ------------------------------------------------------------
# ============================================================
# V16 — ÓRBITA: listas de interceptação automática (E2)
# (sem painel novo; muda listas quando justificado)
# ============================================================

def v16_gerar_listas_interceptacao_orbita(info_orbita: dict,
                                         universo_min: int,
                                         universo_max: int,
                                         n_carro: int,
                                         qtd: int = 4,
                                         seed: int = 0):
    """Gera listas densas adicionais quando ORBITA entra em E2.
    Objetivo: aumentar interseção e repetição controlada sem explodir universo.
    Retorna uma lista de listas (cada uma com n_carro passageiros).
    """
    import random

    try:
        qtd = int(qtd)
    except Exception:
        qtd = 4
    qtd = max(0, min(12, qtd))

    if qtd <= 0:
        return []

    rng = random.Random(int(seed) + 9173)

    # âncoras / candidatos principais (se não vierem, recalcula a partir das listas do pacote)
    anchors = list(info_orbita.get("ancoras") or info_orbita.get("anchors") or [])
    pool_top = list(info_orbita.get("top_passageiros") or info_orbita.get("top_passengers") or [])

    # fallback robusto: usa os passageiros mais frequentes no pacote
    if not anchors or not pool_top:
        listas = info_orbita.get("listas") or info_orbita.get("listas_top") or info_orbita.get("listas_topN") or info_orbita.get("listas_top_n") or []
        freq = {}
        for L in listas:
            for x in L:
                if isinstance(x, int):
                    freq[x] = freq.get(x, 0) + 1
        ordenados = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        pool_top = [k for k, v in ordenados[:max(12, n_carro * 3)]]
        anchors = [k for k, v in ordenados[:max(3, min(8, n_carro + 1))]]

    # garante domínio
    anchors = [x for x in anchors if isinstance(x, int) and universo_min <= x <= universo_max]
    pool_top = [x for x in pool_top if isinstance(x, int) and universo_min <= x <= universo_max]

    # fallback final: universo inteiro (último recurso)
    if not pool_top:
        pool_top = list(range(universo_min, universo_max + 1))

    # modelo: base fixa (2–4 âncoras) + completar com top, preservando diversidade mínima
    # densidade-alvo: mais forte em E2 (pelo menos 3 âncoras se possível)
    base_k = 3 if len(anchors) >= 3 else max(1, min(2, len(anchors)))
    if info_orbita.get("gradiente") in ("G2", "G3"):
        base_k = min(max(3, base_k), max(1, min(4, len(anchors))))

    geradas = []
    vistos = set()

    for i in range(qtd * 3):  # tenta mais para evitar duplicatas
        L = []

        # 1) âncoras (fixa)
        if anchors:
            picks = anchors[:]  # copia
            rng.shuffle(picks)
            L.extend(picks[:base_k])

        # 2) completar com top
        if len(L) < n_carro:
            picks = pool_top[:]
            rng.shuffle(picks)
            for x in picks:
                if x not in L:
                    L.append(x)
                if len(L) >= n_carro:
                    break

        # 3) completar (se ainda faltar) com universo
        if len(L) < n_carro:
            uni = list(range(universo_min, universo_max + 1))
            rng.shuffle(uni)
            for x in uni:
                if x not in L:
                    L.append(x)
                if len(L) >= n_carro:
                    break

        if len(L) != n_carro:
            continue

        L = sorted(L)
        key = tuple(L)
        if key in vistos:
            continue
        vistos.add(key)
        geradas.append(L)

        if len(geradas) >= qtd:
            break

    return geradas



def v16_blindar_ultima_previsao_universo():
    """
    Blindagem estrutural:
    - Garante que nenhuma lista contenha passageiros fora do universo real
    - Usa universo_min / universo_max detectados no carregamento
    - Atua como última barreira antes do uso operacional
    """

    if "ultima_previsao" not in st.session_state:
        return

    listas = st.session_state.get("ultima_previsao")
    if not listas or not isinstance(listas, list):
        return

    umin = st.session_state.get("universo_min")
    umax = st.session_state.get("universo_max")

    if umin is None or umax is None:
        return

    listas_sanas = []

    for lista in listas:
        if not isinstance(lista, (list, tuple)):
            continue

        lista_filtrada = [
            int(x) for x in lista
            if isinstance(x, (int, np.integer)) and umin <= int(x) <= umax
        ]

        if len(lista_filtrada) == len(lista):
            listas_sanas.append(lista)
        else:
            # se houve ajuste, preserva ordem e tamanho quando possível
            lista_corrigida = lista_filtrada[:len(lista)]
            if len(lista_corrigida) == len(lista):
                listas_sanas.append(lista_corrigida)

    if listas_sanas:
        st.session_state["ultima_previsao"] = listas_sanas

# ============================================================
# Sessão Streamlit — persistência para V15.7 MAX
# ============================================================

# Inicialização de estado
if "historico_df" not in st.session_state:
    st.session_state["historico_df"] = None

if "ultima_previsao" not in st.session_state:
    st.session_state["ultima_previsao"] = None

if "sentinela_kstar" not in st.session_state:
    st.session_state["sentinela_kstar"] = None

if "diagnostico_risco" not in st.session_state:
    st.session_state["diagnostico_risco"] = None

if "n_alvo" not in st.session_state:
    st.session_state["n_alvo"] = None




if "aps_postura_selo" not in st.session_state:
    st.session_state["aps_postura_selo"] = None

if "aps_postura_titulo" not in st.session_state:
    st.session_state["aps_postura_titulo"] = None

if "aps_postura_msg" not in st.session_state:
    st.session_state["aps_postura_msg"] = None



# ============================================================
# DETECÇÃO CANÔNICA DE n_alvo (PASSAGEIROS REAIS DA RODADA)
# REGRA FIXA:
# - Última coluna SEMPRE é k
# - Todas as colunas p* anteriores são passageiros
# - n_alvo é definido pela ÚLTIMA SÉRIE VÁLIDA
# ============================================================

def detectar_n_alvo(historico_df):
    if historico_df is None or historico_df.empty:
        return None

    col_pass = [c for c in historico_df.columns if c.startswith("p")]
    if not col_pass:
        return None

    ultima_linha = historico_df[col_pass].iloc[-1]
    return int(ultima_linha.dropna().shape[0])


# Atualização automática de n_alvo
if st.session_state.get("historico_df") is not None:
    st.session_state["n_alvo"] = detectar_n_alvo(
        st.session_state["historico_df"]
    )

# ============================================================
# V16 PREMIUM — INFRAESTRUTURA UNIVERSAL
# (REGRAS CANÔNICAS + ORÇAMENTO CONDICIONADO)
# ============================================================

# -----------------------------
# REGRA CANÔNICA: LISTAS >= n_real
# -----------------------------
def validar_lista_vs_n_real(lista, n_real):
    return isinstance(lista, (list, tuple)) and len(lista) >= int(n_real)

# -----------------------------
# ORÇAMENTOS CONDICIONADOS (TABELAS)
# -----------------------------
ORCAMENTOS_CONDICIONADOS = {
    5: {
        5: 3,
        6: 18,
        7: 63,
        8: 168,
        9: 378,
        10: 756,
    },
    6: {
        6: 6,
        7: 42,
        8: 168,
        9: 504,
        10: 1260,
        11: 2772,
    },
    15: {
        15: 3.5,
        16: 56,
        17: 476,
    },
}

# -----------------------------
# RESOLUÇÃO DE ORÇAMENTO
# -----------------------------
def resolver_orcamento(n_real, tamanho_lista, orcamento_manual=None):
    """
    Prioridade:
    1) Orçamento manual (se fornecido)
    2) Tabela condicionada por n_real
    3) None (não avalia custo)
    """
    if orcamento_manual is not None:
        try:
            return float(orcamento_manual)
        except Exception:
            return None

    tabela = ORCAMENTOS_CONDICIONADOS.get(int(n_real))
    if not tabela:
        return None

    return tabela.get(int(tamanho_lista))

# -----------------------------
# AVALIAÇÃO UNIVERSAL (OBSERVACIONAL)
# -----------------------------
def avaliar_listas_universal(listas, alvo_real, n_real, orcamento_manual=None):
    """
    Retorna métricas OBSERVACIONAIS:
    - acertos / n_real
    - custo (se disponível)
    """
    resultados = []
    alvo_set = set(int(v) for v in alvo_real if int(v) > 0)

    for idx, lst in enumerate(listas, start=1):
        if not validar_lista_vs_n_real(lst, n_real):
            continue

        lst_set = set(int(v) for v in lst if int(v) > 0)
        acertos = len(alvo_set.intersection(lst_set))
        custo = resolver_orcamento(n_real, len(lst), orcamento_manual)

        resultados.append({
            "lista_id": idx,
            "tamanho_lista": len(lst),
            "acertos": acertos,
            "n_real": int(n_real),
            "score": f"{acertos}/{int(n_real)}",
            "custo": custo,
        })

    return resultados





# ============================================================
# GUARDAS DE SEGURANÇA POR n_alvo
# (INFRAESTRUTURA — NÃO APLICADA A NENHUM PAINEL)
# ============================================================

def guarda_n_alvo(n_esperado, nome_modulo):
    n_alvo = st.session_state.get("n_alvo")

    if n_alvo is None:
        st.warning(
            f"⚠️ {nome_modulo}: n_alvo não detectado. "
            f"Carregue um histórico válido antes de executar este painel."
        )
        return False

    if n_alvo != n_esperado:
        st.warning(
            f"🚫 {nome_modulo} BLOQUEADO\n\n"
            f"n detectado = {n_alvo}\n"
            f"n esperado por este módulo = {n_esperado}\n\n"
            f"Este painel assume n fixo e foi bloqueado para evitar "
            f"cálculo incorreto ou truncamento silencioso."
        )
        return False

    return True



# ============================================================
# V16 PREMIUM — INSTRUMENTAÇÃO RETROSPECTIVA (ERRO POR REGIME)
# (PAINEL OBSERVACIONAL PERMANENTE — NÃO MUDA MOTOR)
# ============================================================

def _pc16_normalizar_series_6(historico_df: pd.DataFrame) -> np.ndarray:
    """
    Extrai exatamente as colunas p1..p6 do histórico V15.7 MAX.
    Retorna matriz shape (N, 6) com cada série ordenada.
    """
    if historico_df is None or historico_df.empty:
        return np.zeros((0, 6), dtype=float)

    colunas_esperadas = ["p1", "p2", "p3", "p4", "p5", "p6"]
    for c in colunas_esperadas:
        if c not in historico_df.columns:
            return np.zeros((0, 6), dtype=float)

    try:
        dfp = historico_df[colunas_esperadas].astype(float).dropna()
    except Exception:
        return np.zeros((0, 6), dtype=float)

    if len(dfp) < 10:
        return np.zeros((0, 6), dtype=float)

    arr = dfp.values
    arr.sort(axis=1)
    return arr



def _pc16_distancia_media(v: np.ndarray, centro: np.ndarray) -> float:
    """
    Distância média absoluta (L1 média) entre vetor de 6 e centro de 6.
    """
    return float(np.mean(np.abs(v - centro)))



def pc16_calcular_continuidade_por_janelas(
    historico_df: pd.DataFrame,
    janela: int = 60,
    step: int = 1,
    usar_quantis: bool = True
) -> Dict[str, Any]:
    """
    Analisa retrospectivamente o histórico em janelas móveis.
    Para cada janela [t-janela, t), calcula:
      - 'dx_janela': dispersão média das séries da janela em relação ao centróide da janela
      - 'erro_prox': erro da PRÓXIMA série (t) em relação ao centróide da janela (proxy de 'erro contido')
    Classifica regime por dx_janela (ECO / PRE / RUIM) e compara erro_prox por regime.

    Retorna dict com DataFrame e resumo.
    """
    X = _pc16_normalizar_series_6(historico_df)
    n = X.shape[0]
    if n < (janela + 5):
        return {
            "ok": False,
            "motivo": f"Histórico insuficiente para janela={janela}. Séries válidas: {n}.",
            "df": pd.DataFrame(),
            "resumo": {}
        }

    rows = []
    # percorre janelas, garantindo que exista a "próxima" série t
    for t in range(janela, n - 1, step):
        bloco = X[t - janela:t, :]
        centro = np.mean(bloco, axis=0)

        # dx_janela: média das distâncias das séries da janela ao centróide
        dists = [ _pc16_distancia_media(bloco[i], centro) for i in range(bloco.shape[0]) ]
        dx_janela = float(np.mean(dists))

        # erro_prox: distância da série seguinte (t) ao centróide da janela
        prox = X[t, :]
        erro_prox = _pc16_distancia_media(prox, centro)

        rows.append({
            "t": t,  # índice da série (0-based dentro do array)
            "dx_janela": dx_janela,
            "erro_prox": erro_prox
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return {
            "ok": False,
            "motivo": "Não foi possível gerar janelas (df vazio).",
            "df": pd.DataFrame(),
            "resumo": {}
        }

    # Classificação de regime (ECO/PRE/RUIM) baseada em dx_janela
    if usar_quantis:
        q1 = float(df["dx_janela"].quantile(0.33))
        q2 = float(df["dx_janela"].quantile(0.66))
    else:
        # fallback conservador: thresholds fixos (raramente usado)
        q1, q2 = 0.30, 0.45

    def _rotulo(dx: float) -> str:
        if dx <= q1:
            return "ECO"
        elif dx <= q2:
            return "PRE"
        return "RUIM"

    df["regime"] = df["dx_janela"].apply(_rotulo)

    # Métricas resumo
    resumo = {}
    for reg in ["ECO", "PRE", "RUIM"]:
        sub = df[df["regime"] == reg]
        if len(sub) == 0:
            resumo[reg] = {"n": 0}
            continue

        resumo[reg] = {
            "n": int(len(sub)),
            "dx_janela_medio": float(sub["dx_janela"].mean()),
            "erro_prox_medio": float(sub["erro_prox"].mean()),
            "erro_prox_mediana": float(sub["erro_prox"].median()),
        }

    # Métrica única que queremos: diferença ECO vs RUIM no erro_prox médio
    if resumo.get("ECO", {}).get("n", 0) > 0 and resumo.get("RUIM", {}).get("n", 0) > 0:
        diff = resumo["RUIM"]["erro_prox_medio"] - resumo["ECO"]["erro_prox_medio"]
    else:
        diff = None

    resumo_geral = {
        "janela": int(janela),
        "step": int(step),
        "q1_dx": q1,
        "q2_dx": q2,
        "diff_ruim_menos_eco_no_erro": diff,
        "n_total_janelas": int(len(df))
    }

    return {
        "ok": True,
        "motivo": "",
        "df": df,
        "resumo": resumo,
        "resumo_geral": resumo_geral
    }



# ============================================================
# Função utilitária — formatador geral
# ============================================================
def formatar_lista_passageiros(lista: List[int]) -> str:
    """Formata lista no padrão compacto V15.7 MAX"""
    return ", ".join(str(x) for x in lista)

# ============================================================
# Parsing FLEX ULTRA — versão robusta V15.7 MAX
# ============================================================
def analisar_historico_flex_ultra(conteudo: str) -> pd.DataFrame:
    """
    Parser oficial V15.7 MAX — leitura de histórico com:
    - prefixo C1, C2, C3 ...
    - 5 ou 6 passageiros
    - sensor k sempre na última coluna
    """
    linhas = conteudo.strip().split("\n")
    registros = []

    for linha in linhas:
        partes = linha.replace(" ", "").split(";")
        if len(partes) < 7:
            continue

        try:
            serie = partes[0]
            nums = list(map(int, partes[1:-1]))
            k_val = int(partes[-1])
            registros.append([serie] + nums + [k_val])
        except:
            continue

    colunas = ["serie", "p1", "p2", "p3", "p4", "p5", "p6", "k"]
    df = pd.DataFrame(registros, columns=colunas[: len(registros[0])])

    return df

# ============================================================
# Utilitários de texto e apresentação — V15.7 MAX
# ============================================================
def texto_em_blocos(texto: str, largura: int = 100) -> List[str]:
    if not texto:
        return []
    return textwrap.wrap(texto, width=largura)


def exibir_bloco_mensagem(
    titulo: str,
    corpo: str,
    tipo: str = "info",
) -> None:

    blocos = texto_em_blocos(corpo, largura=110)

    if tipo == "info":
        st.info(f"**{titulo}**\n\n" + "\n\n".join(blocos))
    elif tipo == "warning":
        st.warning(f"**{titulo}**\n\n" + "\n\n".join(blocos))
    elif tipo == "error":
        st.error(f"**{titulo}**\n\n" + "\n\n".join(blocos))
    elif tipo == "success":
        st.success(f"**{titulo}**\n\n" + "\n\n".join(blocos))
    else:
        st.markdown(
            f"""
            <div class="info-box">
                <div class="sub-title">{titulo}</div>
                <p>{"<br>".join(blocos)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# Configurações Anti-Zumbi — limites globais
# ============================================================
LIMITE_SERIES_REPLAY_ULTRA: int = 8000
LIMITE_SERIES_TURBO_ULTRA: int = 8000
LIMITE_PREVISOES_TURBO: int = 600
LIMITE_PREVISOES_MODO_6: int = 800


def limitar_operacao(
    qtd_series: int,
    limite_series: int,
    contexto: str = "",
    painel: str = "",
) -> bool:

    if qtd_series is None:
        return True

    if qtd_series <= limite_series:
        return True

    msg = (
        f"🔒 **Operação bloqueada pela Proteção Anti-Zumbi ({contexto}).**\n\n"
        f"- Séries detectadas: **{qtd_series}**\n"
        f"- Limite seguro: **{limite_series}**\n"
        f"Painel: **{painel}**\n\n"
        "👉 Evitamos travamento no Streamlit."
    )
    exibir_bloco_mensagem("Proteção Anti-Zumbi", msg, tipo="warning")
    return False


# ============================================================
# NÚCLEO V16 — Premium Profundo (Diagnóstico & Calibração)
# Compatível com V15.7 MAX, 100% opcional e retrocompatível
# ============================================================
from typing import Dict, Any, Optional, Tuple  # Reimportar não faz mal


def v16_identificar_df_base() -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    """
    Tenta descobrir qual DataFrame de histórico está ativo no app.
    Busca em chaves comuns do st.session_state para não quebrar nada.
    Se não encontrar nada, retorna (None, None).
    """
    candidatos = []
    for chave in ["historico_df", "df_historico", "df_base", "df", "df_hist"]:
        if chave in st.session_state:
            objeto = st.session_state[chave]
            if isinstance(objeto, pd.DataFrame) and not objeto.empty:
                candidatos.append((chave, objeto))

    if not candidatos:
        return None, None

    chave_escolhida, df_escolhido = candidatos[0]
    return chave_escolhida, df_escolhido


def v16_resumo_basico_historico(
    df: pd.DataFrame,
    limite_linhas: int = 3000,
) -> Dict[str, Any]:
    """
    Gera um resumo leve do histórico para diagnóstico:
    - Quantidade total de séries
    - Janela usada para diagnóstico (anti-zumbi)
    - Distribuição de k (se existir)
    - Presença de colunas relevantes (k*, NR%, QDS)
    Tudo protegido contra KeyError e DataFrames pequenos.
    """
    resumo: Dict[str, Any] = {}

    n_total = int(len(df))
    if n_total <= 0:
        resumo["n_total"] = 0
        resumo["n_usado"] = 0
        resumo["colunas"] = list(df.columns)
        resumo["dist_k"] = {}
        resumo["info_extra"] = {}
        return resumo

    limite_seguro = max(100, min(limite_linhas, n_total))
    df_uso = df.tail(limite_seguro).copy()

    resumo["n_total"] = n_total
    resumo["n_usado"] = int(len(df_uso))
    resumo["colunas"] = list(df_uso.columns)

    dist_k: Dict[Any, int] = {}
    if "k" in df_uso.columns:
        try:
            contagem_k = df_uso["k"].value_counts().sort_index()
            for k_val, qtd in contagem_k.items():
                dist_k[int(k_val)] = int(qtd)
        except Exception:
            dist_k = {}
    resumo["dist_k"] = dist_k

    info_extra: Dict[str, Any] = {}
    for col in df_uso.columns:
        col_lower = str(col).lower()
        if "k*" in col_lower or "k_est" in col_lower or "kstar" in col_lower:
            info_extra["tem_k_estrela"] = True
        if "nr" in col_lower and "%" in col_lower:
            info_extra["tem_nr_percent"] = True
        if "qds" in col_lower:
            info_extra["tem_qds"] = True
    resumo["info_extra"] = info_extra

    return resumo


def v16_mapear_confiabilidade_session_state() -> Dict[str, Any]:
    """
    Varre st.session_state e tenta localizar informações de confiabilidade,
    QDS, k*, NR%, etc., sem assumir nomes fixos.
    Não quebra o app se nada for encontrado.
    """
    mapeamento: Dict[str, Any] = {}

    try:
        for chave, valor in st.session_state.items():
            nome_lower = str(chave).lower()
            if any(token in nome_lower for token in ["confiab", "qds", "k_estrela", "k*", "nr%", "ruido"]):
                if isinstance(valor, (int, float, str)):
                    mapeamento[chave] = valor
                elif isinstance(valor, dict):
                    mapeamento[chave] = {"tipo": "dict", "tamanho": len(valor)}
                elif isinstance(valor, pd.DataFrame):
                    mapeamento[chave] = {
                        "tipo": "DataFrame",
                        "linhas": len(valor),
                        "colunas": list(valor.columns)[:10],
                    }
                else:
                    mapeamento[chave] = {"tipo": type(valor).__name__}
    except Exception:
        pass

    return mapeamento


# ============================================================
# Métricas básicas do histórico — V15.7 MAX
# ============================================================
def calcular_metricas_basicas_historico(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula métricas simples do histórico para uso em:
    - Painel de carregamento
    - Monitor de Risco — k & k*
    Tudo de forma leve, sem quebrar se faltarem colunas.
    """
    metricas: Dict[str, Any] = {}

    if df is None or df.empty:
        metricas["qtd_series"] = 0
        metricas["min_k"] = None
        metricas["max_k"] = None
        metricas["media_k"] = 0.0
        return metricas

    metricas["qtd_series"] = int(len(df))

    if "k" in df.columns:
        try:
            k_vals = df["k"].astype(float)
            metricas["min_k"] = float(k_vals.min())
            metricas["max_k"] = float(k_vals.max())
            metricas["media_k"] = float(k_vals.mean())
        except Exception:
            metricas["min_k"] = None
            metricas["max_k"] = None
            metricas["media_k"] = 0.0
    else:
        metricas["min_k"] = None
        metricas["max_k"] = None
        metricas["media_k"] = 0.0

    return metricas


def exibir_resumo_inicial_historico(metricas: Dict[str, Any]) -> None:
    """
    Exibe um resumo amigável logo após o carregamento do histórico.
    Usado no Painel 1 (Carregar Histórico) e como base para o Monitor de Risco.
    """
    qtd_series = metricas.get("qtd_series", 0)
    min_k = metricas.get("min_k")
    max_k = metricas.get("max_k")
    media_k = metricas.get("media_k", 0.0)

    corpo = (
        f"- Séries carregadas: **{qtd_series}**\n"
        f"- k mínimo: **{min_k}** · k máximo: **{max_k}** · k médio: **{media_k:.2f}**\n"
    )

    exibir_bloco_mensagem(
        "Resumo inicial do histórico (V15.7 MAX)",
        corpo,
        tipo="info",
    )

# ============================================================
# Cabeçalho visual principal
# ============================================================
st.markdown(
    '<div class="big-title">🚗 Predict Cars V15.7 MAX — V16 PREMIUM PROFUNDO</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p class="gray-text">
    Núcleo + Coberturas + Interseção Estatística · Pipeline V14-FLEX ULTRA ·
    Replay LIGHT/ULTRA · TURBO++ HÍBRIDO · TURBO++ ULTRA · Monitor de Risco (k & k*) ·
    Painel de Ruído Condicional · Divergência S6 vs MC · Testes de Confiabilidade REAL ·
    Modo 6 Acertos V15.7 MAX · Relatório Final Integrado.
    </p>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Construção da  — V15.7 MAX
# (LAUDO DE CÓDIGO — BLOCO 1-FIX | ORDEM OPERACIONAL FINAL)
# ============================================================

def construir_navegacao_v157() -> str:

    st.sidebar.markdown("## 🚦  PredictCars V15.7 MAX")
    st.sidebar.markdown("📌 Selecione o painel:")

    n_alvo = st.session_state.get("n_alvo")

    # =========================================================
    # ORDEM OPERACIONAL DEFINITIVA — USO DIÁRIO
    # =========================================================
    opcoes = [

        # -----------------------------------------------------
        # BLOCO 0 — ENTRADA
        # -----------------------------------------------------
        "📁 Carregar Histórico (Arquivo)",
        "📄 Carregar Histórico (Colar)",

        # -----------------------------------------------------
        # BLOCO 1 — ORIENTAÇÃO
        # -----------------------------------------------------
        "🧭 Modo Guiado Oficial — PredictCars",
        "🔍 Diagnóstico Espelho (Mirror)",

        # -----------------------------------------------------
        # BLOCO 2 — LEITURA DO AMBIENTE
        # -----------------------------------------------------
        "🛰️ Sentinelas — k* (Ambiente de Risco)",
        "🧭 Monitor de Risco — k & k*",
        "📊 Observação Histórica — Eventos k",
        "⏱️ Duração da Janela — Análise Histórica",

        # -----------------------------------------------------
        # BLOCO 3 — QUALIDADE DO SINAL
        # -----------------------------------------------------
        "📡 Painel de Ruído Condicional",
        "📉 Painel de Divergência S6 vs MC",

        # -----------------------------------------------------
        # BLOCO 4 — RITMO DO ALVO
        # -----------------------------------------------------
        "🔁 Replay LIGHT",
        "🔁 Replay ULTRA",
        "🧭 Replay Progressivo — Janela Móvel (Assistido)",
        "🧪 Replay Curto — Expectativa 1–3 Séries",

        # -----------------------------------------------------
        # BLOCO 5 — EIXO 1 | ESTRUTURA DAS LISTAS
        # -----------------------------------------------------
        "🧼 B1 — Higiene de Passageiros",
        "🧩 B2 — Coerência Interna das Listas",
        "🟢 B3 — Prontidão (Refinamento)",
        "🟣 B4 — Refinamento Leve de Passageiros",

        # -----------------------------------------------------
        # BLOCO 6 — DECISÃO ÚNICA
        # -----------------------------------------------------
        "🧭 Checklist Operacional — Decisão (AGORA)",

        # -----------------------------------------------------
        # BLOCO 7 — MOTOR
        # -----------------------------------------------------
        "🛣️ Pipeline V14-FLEX ULTRA",
        "⚙️ Modo TURBO++ HÍBRIDO",
        "⚙️ Modo TURBO++ ULTRA",

        # -----------------------------------------------------
        # BLOCO 7.5 — EIXO 2 | MOMENTO & ANTECIPAÇÃO
        # -----------------------------------------------------
        "📊 V16 Premium — Backtest Rápido do Pacote (N=60)",
        "🧭 V16 Premium — Rodadas Estratificadas (A/B)",

        "🧠 M5 — Pulo do Gato (Coleta Automática de Estados)",

        "📈 Expectativa Histórica — Contexto do Momento (V16)",

        # -----------------------------------------------------
        # BLOCO 8 — EXECUÇÃO
        # -----------------------------------------------------
        "🎯 Modo 6 Acertos — Execução",
        "🧪 Testes de Confiabilidade REAL",
        "📘 Relatório Final",

        # -----------------------------------------------------
        # BLOCO 9 — EXTENSÃO
        # -----------------------------------------------------
        "🔵 MODO ESPECIAL — Evento Condicionado",

        # -----------------------------------------------------
        # BLOCO 10 — CAMADA UNIVERSAL
        # -----------------------------------------------------
        "💰 MVP-U2 — Orçamento Universal",
        "🧩 MVP-U3 — Cobertura Universal",
        "📈 MVP-U4 — Eficiência Marginal por Custo",

        # -----------------------------------------------------
        # BLOCO 11 — DEPOIS | APRENDIZADO (EIXO 3)
        # -----------------------------------------------------
        "🧠 Memória Operacional",
        "🧠 Memória Operacional — Registro Semi-Automático",
        "🧠 Laudo Operacional V16",
        "🧠 Diagnóstico ECO & Estado (V16)",
        "🧭 RMO/DMO — Retrato do Momento (V16)",
        "🧾 APS — Auditoria de Postura (V16)",
        "📊 V16 Premium — Erro por Regime (Retrospectivo)",
        "📊 V16 Premium — EXATO por Regime (Proxy)",
        "📊 V16 Premium — PRÉ-ECO → ECO (Persistência & Continuidade)",
        "📊 V16 Premium — Passageiros Recorrentes em ECO (Interseção)",
        "📊 V16 Premium — PRÉ-ECO | Contribuição de Passageiros",
        "📊 V16 Premium — ANTI-EXATO | Passageiros Nocivos",
        "🎯 Compressão do Alvo — Observacional (V16)",
        "🔮 V16 Premium Profundo — Diagnóstico & Calibração",
    ]

    # ------------------------------------------------------------
    # INSERÇÃO CONDICIONAL — MODO N EXPERIMENTAL (n≠6)
    # ------------------------------------------------------------
    if (n_alvo is not None) and (int(n_alvo) != 6):
        opcoes.insert(
            opcoes.index("🎯 Modo 6 Acertos — Execução"),
            "🧪 Modo N Experimental (n≠6)"
        )
    # --- estabilidade do NAV (evita painel=None / TypeError)
    _nav_key = "NAV_V157_CANONICA"
    _prev = st.session_state.get(_nav_key, None)
    if (_prev is None) or (_prev not in opcoes):
        _prev = opcoes[0] if opcoes else "Carregar Histórico (Colar)"
        st.session_state[_nav_key] = _prev
    painel = st.sidebar.radio("", opcoes, index=opcoes.index(_prev) if _prev in opcoes else 0, key=_nav_key)
    return painel


# ============================================================
# FIM — Construção da  — V15.7 MAX
# ============================================================



# ============================================================
# Ativação da Navegação — V15.7 MAX
# ============================================================

painel = construir_navegacao_v157()
if painel is None:
    # fallback absoluto (protege contra TypeError em `in painel`)
    painel = "Carregar Histórico (Colar)"
st.sidebar.caption(f"Painel ativo: {painel}")

# ============================================================
# DEBUG — CARIMBO DE BUILD (QUAL ARQUIVO REALMENTE ESTÁ RODANDO)
# ============================================================
try:
    st.sidebar.markdown("---")
    st.sidebar.caption("✅ BUILD-ID: NAV_ANCORA_ESTAVEL_2026-01-18")
    # Observação: build-id atualizado quando há alteração canônica na navegação.
    st.sidebar.caption(f"📄 __file__: {__file__}")
    st.sidebar.caption(f"🔎 Primeiro item NAV: {construir_navegacao_v157.__name__}")
    st.sidebar.caption("🧭 TOP-5: (debug desativado — não chamar construir_navegacao_v157() aqui)")
except Exception as _e:
    st.sidebar.caption(f"⚠️ DEBUG build falhou: {_e}")



# ============================================================
# DEBUG MINIMAL — CONFIRMA PAINEL ATIVO
# (manter por enquanto para auditoria)
# ============================================================
st.sidebar.caption(f"Painel ativo: {painel}")



# ============================================================
# MODO ESPECIAL — EVENTO CONDICIONADO (C2955)
# AVALIAÇÃO MULTI-ORÇAMENTO | OBSERVACIONAL | 6 OU NADA
# ============================================================

def pc_especial_avaliar_pacote_contem_6(carro, alvo):
    """
    Retorna True se o carro contém TODOS os 6 números do alvo.
    Régua BINÁRIA: 6 ou nada.
    """
    try:
        return set(alvo).issubset(set(carro))
    except Exception:
        return False


def pc_especial_avaliar_historico_pacote(historico_df, pacote):
    """
    Percorre o histórico rodada a rodada e verifica se,
    em alguma rodada, algum carro do pacote contém os 6.
    Retorna contagem de sucessos.
    """
    if historico_df is None or historico_df.empty:
        return {
            "rodadas": 0,
            "sucessos": 0,
        }

    col_pass = [c for c in historico_df.columns if c.startswith("p")]
    rodadas = 0
    sucessos = 0

    for _, row in historico_df.iterrows():
        try:
            alvo = [int(row[c]) for c in col_pass[:6]]
        except Exception:
            continue

        rodadas += 1

        for carro in pacote:
            if pc_especial_avaliar_pacote_contem_6(carro, alvo):
                sucessos += 1
                break  # sucesso binário por rodada

    return {
        "rodadas": rodadas,
        "sucessos": sucessos,
    }


# ============================================================
# 🔵 MODO ESPECIAL — MVP2 (2–6 acertos + Estado do Alvo PROXY)
# OBSERVACIONAL | NÃO decide | NÃO gera pacotes | NÃO aprende
# ============================================================

def _pc_contar_hits_lista_vs_alvo(lista, alvo_set):
    """
    Retorna quantidade de acertos (interseção) entre uma lista (carro) e o alvo (set).
    """
    try:
        s = set(int(x) for x in lista)
    except Exception:
        return 0
    return len(s & alvo_set)


def _pc_melhor_hit_do_pacote(pacote_listas, alvo_set):
    """
    Dado um pacote (listas de previsão), retorna o MELHOR hit (0..6) encontrado contra o alvo.
    """
    if not pacote_listas:
        return 0

    best = 0
    for lst in pacote_listas:
        h = _pc_contar_hits_lista_vs_alvo(lst, alvo_set)
        if h > best:
            best = h
            if best >= 6:
                break
    return best


def _pc_extrair_carro_row(row):
    """
    Extrai os 6 passageiros da linha do histórico.
    Espera colunas p1..p6 (padrão do PredictCars).
    """
    try:
        return [int(row[f"p{i}"]) for i in range(1, 7)]
    except Exception:
        return None


def _pc_distancia_carros(carro_a, carro_b):
    """
    Distância simples entre dois carros (proxy):
    número de passageiros diferentes.
    """
    if carro_a is None or carro_b is None:
        return None
    try:
        return len(set(carro_a) ^ set(carro_b))
    except Exception:
        return None


def _pc_estado_alvo_proxy(dist):
    """
    Classificação simples do estado do alvo (proxy),
    baseada na distância entre carros consecutivos.
    """
    if dist is None:
        return "None"

    try:
        d = float(dist)
    except Exception:
        return "None"

    if d <= 1:
        return "parado"
    elif d <= 3:
        return "movimento_lento"
    else:
        return "movimento_brusco"


def pc_modo_especial_mvp2_avaliar_pacote(df_hist, pacote_listas):
    """
    MVP2:
    - Para cada série do histórico, computa:
        estado_alvo_proxy (parado/lento/brusco/None)
        melhor_hit (0..6) do pacote contra o alvo daquela série
    - Consolida em tabela: Estado x Hits(2..6) [contagem EXATA]
    Retorna (df_resumo, total_series_avaliadas).
    """
    if df_hist is None or df_hist.empty:
        return pd.DataFrame(), 0

    if not pacote_listas:
        return pd.DataFrame(), int(len(df_hist))

    cont = {
        "parado": {2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        "movimento_lento": {2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        "movimento_brusco": {2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        "None": {2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
    }

    rows = list(df_hist.iterrows())
    carro_prev = None

    for _, row in rows:
        carro_atual = _pc_extrair_carro_row(row)

        dist = (
            _pc_distancia_carros(carro_prev, carro_atual)
            if carro_prev is not None and carro_atual is not None
            else None
        )

        estado = _pc_estado_alvo_proxy(dist)
        estado_key = estado if estado in cont else "None"

        if carro_atual is None:
            carro_prev = carro_atual
            continue

        alvo_set = set(carro_atual)
        best_hit = _pc_melhor_hit_do_pacote(pacote_listas, alvo_set)

        if best_hit in [2, 3, 4, 5, 6]:
            cont[estado_key][best_hit] += 1

        carro_prev = carro_atual

    out = []
    for estado_key in ["parado", "movimento_lento", "movimento_brusco", "None"]:
        linha = {"Estado": estado_key}
        for h in [2, 3, 4, 5, 6]:
            linha[str(h)] = int(cont[estado_key][h])
        out.append(linha)

    df_out = pd.DataFrame(out)

    ordem = {"parado": 0, "movimento_lento": 1, "movimento_brusco": 2, "None": 3}
    df_out["__ord"] = df_out["Estado"].map(ordem).fillna(9).astype(int)
    df_out = df_out.sort_values("__ord").drop(columns=["__ord"])

    return df_out, int(len(df_hist))

# ============================================================
# 🔵 FIM — FUNÇÕES DO MODO ESPECIAL MVP2
# ============================================================


# ============================================================
# PAINEL — 🔵 MODO ESPECIAL (Evento Condicionado C2955)
# Avaliação MULTI-ORÇAMENTO | Observacional
# ============================================================

if painel == "🔵 MODO ESPECIAL — Evento Condicionado":

    st.markdown("## 🔵 MODO ESPECIAL — Evento Condicionado (C2955)")
    st.caption(
        "Avaliação OBSERVACIONAL de pacotes já gerados.\n\n"
        "✔ Régua extrema: **6 ou nada** (MVP1)\n"
        "✔ Avaliação realista: **2–6 por estado do alvo** (MVP2)\n"
        "✔ Sem aprendizado\n"
        "✔ Sem interferência no Modo Normal\n"
        "✔ Decisão HUMANA (Rogério + Auri)"
    )

    historico_df = st.session_state.get("historico_df")

    # ============================================================
    # 🔵 SELETOR DE FONTE DO PACOTE (TURBO × MODO 6)
    # OBSERVACIONAL | NÃO decide | NÃO aprende | NÃO interfere
    # ============================================================

    pacote_turbo_raw = st.session_state.get("ultima_previsao")

    pacote_m6_total = (
        st.session_state.get("modo6_listas_totais")
        or st.session_state.get("modo6_listas")
        or []
    )

    pacote_m6_top10 = st.session_state.get("modo6_listas_top10") or []

    fontes = []
    if pacote_turbo_raw:
        fontes.append("TURBO (núcleo)")
    if pacote_m6_total:
        fontes.append("MODO 6 (TOTAL)")
    if pacote_m6_top10:
        fontes.append("MODO 6 (TOP 10)")
    if pacote_turbo_raw and pacote_m6_total:
        fontes.append("MIX (TURBO + M6 TOTAL)")

    if not fontes:
        exibir_bloco_mensagem(
            "Pré-requisitos ausentes",
            "É necessário:\n"
            "- Histórico carregado\n"
            "- Pacotes gerados pelo TURBO ou Modo 6",
            tipo="warning",
        )
        st.stop()

    idx_default = fontes.index("MODO 6 (TOTAL)") if "MODO 6 (TOTAL)" in fontes else 0

    fonte_escolhida = st.selectbox(
        "Fonte do pacote para avaliação (observacional):",
        options=fontes,
        index=idx_default,
    )

    # -----------------------------
    # Construção do pacote ativo
    # -----------------------------
    if fonte_escolhida == "TURBO (núcleo)":
        pacotes_raw = pacote_turbo_raw
    elif fonte_escolhida == "MODO 6 (TOTAL)":
        pacotes_raw = pacote_m6_total
    elif fonte_escolhida == "MODO 6 (TOP 10)":
        pacotes_raw = pacote_m6_top10
    else:
        mix = []

        if isinstance(pacote_turbo_raw, list):
            if pacote_turbo_raw and isinstance(pacote_turbo_raw[0], int):
                mix.append(pacote_turbo_raw)
            else:
                mix.extend(pacote_turbo_raw)

        if isinstance(pacote_m6_total, list):
            mix.extend(pacote_m6_total)

        pacotes_raw = mix

    # ============================================================
    # ✅ NORMALIZAÇÃO FINAL — LISTA DE LISTAS
    # ============================================================
    if pacotes_raw is None:
        pacotes = []
    elif isinstance(pacotes_raw, list) and pacotes_raw and isinstance(pacotes_raw[0], int):
        pacotes = [pacotes_raw]
    elif isinstance(pacotes_raw, list):
        pacotes = pacotes_raw
    else:
        pacotes = []

    st.caption(
        f"Pacote ativo: **{fonte_escolhida}** | "
        f"Listas avaliadas: **{len(pacotes)}**"
    )

    if historico_df is None or historico_df.empty or not pacotes:
        exibir_bloco_mensagem(
            "Pré-requisitos ausentes",
            "Histórico vazio ou pacote inválido.",
            tipo="warning",
        )
        st.stop()

    # ============================================================
    # 🔵 MVP4 — ANÁLISE DE COMPOSIÇÕES DE COBERTURA (OBSERVACIONAL)
    # Núcleo / Fronteira automáticos — NÃO executa
    # ============================================================

    st.markdown("### 🔵 MVP4 — Análise de Composições de Cobertura")
    st.caption(
        "Painel analítico: sugere **composições candidatas** (6×6 até 1×9),\n"
        "com base em núcleo/fronteira extraídos automaticamente.\n"
        "❌ Não gera listas | ❌ Não decide | ❌ Não interfere"
    )

    from collections import Counter
    from math import comb

    todas = [n for lista in pacotes for n in lista]
    freq = Counter(todas)

    nucleo = sorted([n for n, c in freq.items() if c >= 3])
    fronteira = sorted([n for n, c in freq.items() if c == 2])
    ruido = sorted([n for n, c in freq.items() if c == 1])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🧱 Núcleo**")
        st.write(nucleo if nucleo else "—")
        if len(nucleo) < 4:
            st.warning("Núcleo fraco (<4).")
        if len(nucleo) > 5:
            st.warning("Núcleo grande (>5).")

    with col2:
        st.markdown("**🟡 Fronteira**")
        st.write(fronteira if fronteira else "—")
        if len(fronteira) > 6:
            st.warning("Fronteira extensa (ambiguidade elevada).")

    with col3:
        st.markdown("**🔴 Ruído**")
        st.write(ruido if ruido else "—")
        st.caption("Ruído excluído de carros >6.")

    st.markdown("#### 📦 Composições Candidatas (comparação teórica)")

    composicoes = [
        ("C1 — Foco puro", [(6, 6)]),
        ("C2 — Proteção leve", [(6, 4), (7, 1)]),
        ("C3 — Proteção + ambiguidade", [(6, 2), (7, 1), (8, 1)]),
        ("C4 — Envelope compacto", [(8, 1)]),
        ("C5 — Envelope amplo", [(9, 1)]),
    ]

    for nome, mix in composicoes:
        custo = 0
        combs = 0
        for m, q in mix:
            c = comb(m, 6)
            custo += c * 6 * q
            combs += c * q

        with st.expander(f"📘 {nome}"):
            st.write(f"Mix: {mix}")
            st.write(f"• Combinações de 6 cobertas: **{combs}**")
            st.write(f"• Custo teórico (régua): **{custo}**")

            if len(nucleo) < 4:
                st.warning("⚠️ Núcleo fraco — envelope pode diluir sinal.")
            if len(fronteira) > 6:
                st.warning("⚠️ Fronteira grande — risco de ilusão de cobertura.")

    # ============================================================
    # MVP2 — Avaliação 2–6 × Estado do Alvo (OBSERVACIONAL)
    # ============================================================

    st.markdown("### 📊 Resultado comparativo — MVP2 (2–6 × Estado do Alvo)")
    st.caption(
        "Leitura realista de aproximação.\n"
        "🟢 parado | 🟡 movimento lento | 🔴 movimento brusco\n"
        "O sistema **não decide**."
    )

    linhas = []

    orcamentos_disponiveis = [6, 42, 168, 504, 1260, 2772]

    orcamentos_sel = st.multiselect(
        "Selecione os orçamentos a avaliar (observacional):",
        options=orcamentos_disponiveis,
        default=[42],
    )

    if not orcamentos_sel:
        st.warning("Selecione ao menos um orçamento.")
        st.stop()

    for orc in orcamentos_sel:
        df_mvp2, total_series = pc_modo_especial_mvp2_avaliar_pacote(
            df_hist=historico_df,
            pacote_listas=pacotes,
        )

        if df_mvp2 is None or df_mvp2.empty:
            linhas.append({
                "Orçamento": orc,
                "Estado": "N/A",
                "Séries": int(total_series),
                "2": 0, "3": 0, "4": 0, "5": 0, "6": 0
            })
            continue

        for _, r in df_mvp2.iterrows():
            linhas.append({
                "Orçamento": int(orc),
                "Estado": str(r["Estado"]),
                "Séries": int(total_series),
                "2": int(r["2"]),
                "3": int(r["3"]),
                "4": int(r["4"]),
                "5": int(r["5"]),
                "6": int(r["6"]),
            })

    df_cmp = pd.DataFrame(linhas)
    st.dataframe(df_cmp, use_container_width=True, height=420)

    st.info(
        "📌 Interpretação HUMANA:\n"
        "- 🟢 Mais 4/5 em 'parado' → janela boa\n"
        "- 🟡 Predomínio de 3/4 → cautela\n"
        "- 🔴 Quase só 2/3 → reduzir agressividade\n"
        "- 6 é raro; 4/5 indicam proximidade real"
    )




# ============================================================
# CAMADA A — ESTADO DO ALVO (V16)
# Observador puro — NÃO decide, NÃO bloqueia, NÃO gera previsões
# ============================================================

def v16_diagnosticar_eco_estado():
    """
    Diagnóstico OBSERVACIONAL enriquecido (ECO A):
    - ECO: força + persistência + acionabilidade
    - ESTADO: parado / movimento_lento / movimento_brusco + confiabilidade
    NÃO altera motores | NÃO decide | NÃO bloqueia
    """

    historico_df = st.session_state.get("historico_df")

    # -----------------------------
    # Fallback seguro
    # -----------------------------
    if historico_df is None or historico_df.empty:
        diag = {
            "eco_forca": "indefinido",
            "eco_persistencia": "indefinida",
            "eco_acionabilidade": "não_acionável",
            "estado": "indefinido",
            "estado_confiavel": False,
            "contradicoes": ["histórico insuficiente"],
            "leitura_geral": "Histórico insuficiente para diagnóstico.",
        }
        st.session_state["diagnostico_eco_estado_v16"] = diag
        v16_sync_aliases_canonicos()
        return diag

    # =========================================================
    # ECO — sinais já existentes
    # =========================================================
    k_star = st.session_state.get("sentinela_kstar")
    nr_pct = st.session_state.get("nr_percent")
    divergencia = st.session_state.get("div_s6_mc")

    sinais = 0
    motivos = []

    if isinstance(k_star, (int, float)) and k_star < 0.15:
        sinais += 1
        motivos.append("k* favorável")

    if isinstance(nr_pct, (int, float)) and nr_pct < 30:
        sinais += 1
        motivos.append("ruído controlado")

    if isinstance(divergencia, (int, float)) and divergencia < 5:
        sinais += 1
        motivos.append("baixa divergência")

    if sinais >= 3:
        eco_forca = "forte"
    elif sinais == 2:
        eco_forca = "médio"
    else:
        eco_forca = "fraco"

    # Persistência curta
    hist_eco = st.session_state.get("historico_eco_v16", [])
    hist_eco.append(eco_forca)
    hist_eco = hist_eco[-5:]
    st.session_state["historico_eco_v16"] = hist_eco

    eco_persistencia = "persistente" if hist_eco.count(eco_forca) >= 3 else "instável"

    # =========================================================
    # CONTRADIÇÕES (leitura turva)
    # =========================================================
    contradicoes = []
    if eco_forca in ("fraco", "médio") and isinstance(divergencia, (int, float)) and divergencia > 20:
        contradicoes.append("divergência elevada")
    if eco_persistencia == "persistente" and isinstance(divergencia, (int, float)) and divergencia > 30:
        contradicoes.append("persistência enganosa")

    # =========================================================
    # ACIONABILIDADE (OBSERVACIONAL)
    # =========================================================
    if eco_forca == "forte" and eco_persistencia == "persistente" and not contradicoes:
        eco_acionabilidade = "favorável"
    elif eco_forca in ("médio", "forte") and not contradicoes:
        eco_acionabilidade = "cautela"
    else:
        eco_acionabilidade = "não_acionável"

    # =========================================================
    # ESTADO DO ALVO (proxy existente)
    # =========================================================
    estado_proxy = None
    try:
        col_pass = [c for c in historico_df.columns if c.startswith("p")]
        if len(col_pass) >= 6 and len(historico_df) >= 2:
            a = [int(historico_df.iloc[-1][c]) for c in col_pass[:6]]
            b = [int(historico_df.iloc[-2][c]) for c in col_pass[:6]]
            dist = len(set(a) ^ set(b))
            if dist <= 1:
                estado_proxy = "parado"
            elif dist <= 3:
                estado_proxy = "movimento_lento"
            else:
                estado_proxy = "movimento_brusco"
    except Exception:
        estado_proxy = None

    if estado_proxy is None:
        estado = "indefinido"
        estado_confiavel = False
    else:
        estado = estado_proxy
        hist_estado = st.session_state.get("historico_estado_v16", [])
        hist_estado.append(estado)
        hist_estado = hist_estado[-5:]
        st.session_state["historico_estado_v16"] = hist_estado
        estado_confiavel = hist_estado.count(estado) >= 3

    # =========================================================
    # LEITURA FINAL (MASTIGADA)
    # =========================================================
    leitura = (
        f"ECO {eco_forca}, {eco_persistencia}, {eco_acionabilidade}. "
        f"Estado {estado}. "
        f"{'Confiável' if estado_confiavel else 'Em transição'}."
    )
    if contradicoes:
        leitura += " Atenção: " + "; ".join(contradicoes) + "."

    diagnostico = {
        "eco_forca": eco_forca,
        "eco_persistencia": eco_persistencia,
        "eco_acionabilidade": eco_acionabilidade,
        "estado": estado,
        "estado_confiavel": estado_confiavel,
        "contradicoes": contradicoes,
        "leitura_geral": leitura,
        "motivos_eco": motivos,
    }

    st.session_state["diagnostico_eco_estado_v16"] = diagnostico
    return diagnostico

# ============================================================
# ATIVAÇÃO SILENCIOSA — DIAGNÓSTICO ECO & ESTADO (V16)
# ============================================================
if "historico_df" in st.session_state:
    try:
        v16_diagnosticar_eco_estado()
    except Exception:
        pass




# ============================================================
# CAMADA B — EXPECTATIVA DE CURTO PRAZO (V16)
# Laudo observacional: horizonte 1–3 séries (NÃO decide)
# ============================================================


def v16_calcular_expectativa_curto_prazo(
    df: Optional[pd.DataFrame],
    estado_alvo: Optional[Dict[str, Any]],
    k_star: Optional[float],
    nr_percent: Optional[float],
    divergencia: Optional[float],
) -> Dict[str, Any]:

    if df is None or df.empty:
        return {
            "horizonte": "1–3 séries",
            "previsibilidade": "indefinida",
            "erro_esperado": "indefinido",
            "chance_janela_ouro": "baixa",
            "comentario": "Histórico insuficiente para expectativa.",
        }

    k = float(k_star) if isinstance(k_star, (int, float)) else 0.25
    nr = float(nr_percent) if isinstance(nr_percent, (int, float)) else 35.0
    div = float(divergencia) if isinstance(divergencia, (int, float)) else 4.0

    tipo = (estado_alvo or {}).get("tipo", "movimento_lento")

    # Índice simples de previsibilidade
    risco_norm = min(1.0, (nr / 70.0) * 0.4 + (div / 10.0) * 0.3 + (k / 0.5) * 0.3)
    previsibilidade_score = max(0.0, 1.0 - risco_norm)

    if previsibilidade_score >= 0.65:
        previsibilidade = "alta"
        erro = "baixo"
    elif previsibilidade_score >= 0.40:
        previsibilidade = "média"
        erro = "médio"
    else:
        previsibilidade = "baixa"
        erro = "alto"

    # Chance de janela de ouro (qualitativa)
    if tipo == "parado" and previsibilidade_score >= 0.60:
        chance_ouro = "alta"
    elif tipo == "movimento_lento" and previsibilidade_score >= 0.45:
        chance_ouro = "média"
    else:
        chance_ouro = "baixa"

    comentario = (
        f"Alvo {tipo}. Previsibilidade {previsibilidade}. "
        f"Erro esperado {erro}. Chance de janela de ouro {chance_ouro}."
    )

    return {
        "horizonte": "1–3 séries",
        "previsibilidade": previsibilidade,
        "erro_esperado": erro,
        "chance_janela_ouro": chance_ouro,
        "score_previsibilidade": round(previsibilidade_score, 4),
        "comentario": comentario,
    }


def v16_registrar_expectativa():
    estado = st.session_state.get("estado_alvo_v16")
    expectativa = v16_calcular_expectativa_curto_prazo(
        st.session_state.get("historico_df"),
        estado,
        st.session_state.get("sentinela_kstar"),
        st.session_state.get("nr_percent"),
        st.session_state.get("div_s6_mc"),
    )
    st.session_state["expectativa_v16"] = expectativa
    return expectativa

# ============================================================
# CAMADA C — VOLUME & CONFIABILIDADE (V16)
# Sistema INFORMA; humano DECIDE
# ============================================================

def v16_estimativa_confiabilidade_por_volume(
    estado_alvo: Optional[Dict[str, Any]],
    expectativa: Optional[Dict[str, Any]],
    base_confiabilidade: Optional[float] = None,
) -> Dict[int, float]:
    """
    Retorna um mapa {volume: confiabilidade_estimada}.
    Não bloqueia execução; apenas informa trade-offs.
    """
    tipo = (estado_alvo or {}).get("tipo", "movimento_lento")
    score_prev = (expectativa or {}).get("score_previsibilidade", 0.4)

    # Base de confiabilidade (fallback seguro)
    base = float(base_confiabilidade) if isinstance(base_confiabilidade, (int, float)) else score_prev

    # Ajuste por tipo de alvo
    if tipo == "parado":
        fator = 1.15
    elif tipo == "movimento_lento":
        fator = 1.00
    else:
        fator = 0.80

    volumes = [3, 6, 12, 20, 30, 50, 80]
    estimativas: Dict[int, float] = {}

    for v in volumes:
        # Ganho marginal decrescente
        ganho = 1.0 - (1.0 / max(1.0, np.log(v + 1)))
        conf = base * fator * ganho
        estimativas[v] = round(max(0.05, min(0.95, conf)), 3)

    return estimativas


def v16_calcular_volume_operacional(
    estado_alvo: Optional[Dict[str, Any]],
    expectativa: Optional[Dict[str, Any]],
    confiabilidades: Dict[int, float],
) -> Dict[str, Any]:
    """
    Consolida recomendações de volume sem impor decisão.
    """
    tipo = (estado_alvo or {}).get("tipo", "movimento_lento")
    prev = (expectativa or {}).get("previsibilidade", "média")

    # Volume recomendado por heurística qualitativa
    if tipo == "parado" and prev == "alta":
        recomendado = 30
    elif tipo == "movimento_lento":
        recomendado = 20
    else:
        recomendado = 6

    # Limites técnicos (anti-zumbi conceitual, não bloqueante)
    minimo = 3
    maximo = max(confiabilidades.keys()) if confiabilidades else 30

    return {
        "minimo": minimo,
        "recomendado": recomendado,
        "maximo_tecnico": maximo,
        "confiabilidades_estimadas": confiabilidades,
        "comentario": (
            "O sistema informa volumes e confiabilidades. "
            "A decisão final de quantas previsões gerar é do usuário."
        ),
    }


def v16_registrar_volume_e_confiabilidade():
    estado = st.session_state.get("estado_alvo_v16")
    expectativa = st.session_state.get("expectativa_v16")

    confiabs = v16_estimativa_confiabilidade_por_volume(
        estado_alvo=estado,
        expectativa=expectativa,
        base_confiabilidade=(expectativa or {}).get("score_previsibilidade"),
    )

    volume_op = v16_calcular_volume_operacional(
        estado_alvo=estado,
        expectativa=expectativa,
        confiabilidades=confiabs,
    )

    st.session_state["volume_operacional_v16"] = volume_op
    return volume_op



# ============================================================
# PARTE 1/8 — FIM
# ============================================================


# ============================================================
# MODULO 2 - MEMORIA HISTORICA DE ESTADOS (OBSERVACIONAL)
# Objetivo:
# - Registrar estados canonicos (S0..S6) por rodada, sem listas e sem acerto
# - Alimentar o MODULO 3 (Expectativa) com base comparavel e sem vies
# Regras:
# - NUNCA decide
# - NUNCA altera motores
# - NUNCA bloqueia
# - Falha silenciosa (nao derruba o app)
# ============================================================

import json
from datetime import datetime


def _m2_init_memoria() -> None:
    """Inicializa a memoria em session_state (infraestrutura invisivel)."""
    try:
        ss = st.session_state
        if "m2_memoria_estados" not in ss or not isinstance(ss.get("m2_memoria_estados"), list):
            ss["m2_memoria_estados"] = []
        if "m2_memoria_selo_s3" not in ss:
            ss["m2_memoria_selo_s3"] = set()
        if "m2_memoria_selo_s6" not in ss:
            ss["m2_memoria_selo_s6"] = set()
    except Exception:
        # falha silenciosa
        pass


def _m2_guess_serie_id(snapshot: dict) -> str:
    """Tenta inferir o id da serie atual. Regra: nunca falhar."""
    try:
        ss = st.session_state
        # Preferencias: chaves explicitas (quando existirem)
        for k in ("serie_id", "serie_atual", "serie_corrente", "concurso_atual", "c_atual"):
            v = ss.get(k)
            if v not in (None, "", "N/D"):
                return str(v)
        # Fallback: tamanho do historico
        df = ss.get("historico_df")
        if df is not None:
            try:
                n = int(len(df))
                return f"C{n}"
            except Exception:
                pass
        # ultimo fallback
        return "N/D"
    except Exception:
        return "N/D"


def _m2_build_registro_minimo(snapshot: dict, meta: dict) -> dict:
    """Registro minimo (obrigatorio) - comparavel e sem vies."""
    ss = st.session_state
    serie_id = _m2_guess_serie_id(snapshot)
    # normalizar universo
    umin = snapshot.get("universo_min", ss.get("universo_min", "N/D"))
    umax = snapshot.get("universo_max", ss.get("universo_max", "N/D"))
    # normalizar diagnostico
    regime = snapshot.get("regime", ss.get("regime_identificado", "N/D"))
    classe = snapshot.get("classe_risco", ss.get("classe_risco", "N/D"))
    nrp = snapshot.get("nr_percent", ss.get("nr_percent", "N/D"))
    div = snapshot.get("divergencia_s6_mc", ss.get("divergencia_s6_mc", "N/D"))

    return {
        "ts": datetime.utcnow().isoformat() + "Z",
        "serie_id": serie_id,
        "estado": str(meta.get("estado", "N/D")),
        "n_alvo": snapshot.get("n_alvo", ss.get("n_alvo", "N/D")),
        "universo_min": umin,
        "universo_max": umax,
        "regime": regime,
        "classe_risco": classe,
        "nr_percent": nrp,
        "divergencia_s6_mc": div,
    }


def _m2_build_registro_estendido(snapshot: dict, meta: dict) -> dict:
    """Registro estendido (opcional) - fechamento de rodada, sem acerto."""
    ss = st.session_state
    base = _m2_build_registro_minimo(snapshot, meta)

    # postura humana (se existir) - NUNCA vira gatilho automatico
    postura = ss.get("postura_humana", ss.get("checklist_decisao", "N/D"))

    # exposicao e execucao
    base.update({
        "houve_execucao": bool(snapshot.get("modo6_executado")),
        "turbo_tentado": bool(snapshot.get("turbo_tentado")),
        "turbo_bloqueado": bool(snapshot.get("turbo_bloqueado")),
        "turbo_motivo": snapshot.get("turbo_motivo", "N/D"),
        "postura_humana": postura,
        "listas_geradas": snapshot.get("listas_geradas", "<nao definido>"),
    })

    return base


def _m2_persistir_linha_jsonl(registro: dict) -> None:
    """Persistencia best-effort em JSONL. Falha silenciosa (Streamlit Cloud pode restringir)."""
    try:
        # arquivo local no diretorio do app (best-effort)
        path = "memoria_estados_v16.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _m2_registrar_minimo_se_preciso(snapshot: dict, meta: dict) -> None:
    """Gatilho canonico: registrar no primeiro S3 auditavel (sem duplicar)."""
    try:
        _m2_init_memoria()
        estado = str(meta.get("estado", ""))
        if estado not in ("S3", "S4", "S5", "S6"):
            return

        serie_id = _m2_guess_serie_id(snapshot)
        chave = f"{serie_id}|S3"
        selos = st.session_state.get("m2_memoria_selo_s3")
        if isinstance(selos, set) and chave in selos:
            return

        reg = _m2_build_registro_minimo(snapshot, {"estado": "S3"})
        st.session_state["m2_memoria_estados"].append(reg)
        if isinstance(selos, set):
            selos.add(chave)
        st.session_state["m2_ts_ultimo_registro_s3"] = reg.get("ts")

        _m2_persistir_linha_jsonl(reg)
    except Exception:
        pass


def _m2_registrar_fechamento_se_preciso(snapshot: dict, meta: dict) -> None:
    """Gatilho complementar: registrar fechamento (S6 heuristico), sem duplicar."""
    try:
        _m2_init_memoria()
        estado = str(meta.get("estado", ""))
        if estado != "S6":
            return

        serie_id = _m2_guess_serie_id(snapshot)
        chave = f"{serie_id}|S6"
        selos = st.session_state.get("m2_memoria_selo_s6")
        if isinstance(selos, set) and chave in selos:
            return

        reg = _m2_build_registro_estendido(snapshot, {"estado": "S6"})
        st.session_state["m2_memoria_estados"].append(reg)
        if isinstance(selos, set):
            selos.add(chave)
        st.session_state["m2_ts_ultimo_registro_s6"] = reg.get("ts")

        _m2_persistir_linha_jsonl(reg)
    except Exception:
        pass


def _m2_resumo_auditavel() -> dict:
    """Resumo leve para aparecer no Mirror (auditoria controlada)."""
    try:
        _m2_init_memoria()
        mem = st.session_state.get("m2_memoria_estados", [])
        total = len(mem) if isinstance(mem, list) else 0

        # contagens simples por estado
        cont = {}
        if isinstance(mem, list):
            for r in mem:
                try:
                    e = str(r.get("estado", "N/D"))
                    cont[e] = cont.get(e, 0) + 1
                except Exception:
                    pass

        # diversidade de contexto (n e universo)
        ns = set()
        universos = set()
        if isinstance(mem, list):
            for r in mem:
                try:
                    ns.add(str(r.get("n_alvo", "N/D")))
                    universos.add(f"{r.get('universo_min','N/D')}–{r.get('universo_max','N/D')}")
                except Exception:
                    pass

        return {
            "memoria_total": total,
            "por_estado": cont,
            "n_distintos": sorted(list(ns))[:8],
            "universos_distintos": sorted(list(universos))[:8],
            "ts_ultimo_s3": st.session_state.get("m2_ts_ultimo_registro_s3", "N/D"),
            "ts_ultimo_s6": st.session_state.get("m2_ts_ultimo_registro_s6", "N/D"),
        }
    except Exception:
        return {"memoria_total": "N/D"}

# ============================================================
# PARTE 2/8 — INÍCIO
# ============================================================

# ============================================================
# >>> FUNÇÃO AUXILIAR — AJUSTE DE AMBIENTE PARA MODO 6
# (UNIVERSAL — respeita o fenômeno detectado)
# ============================================================

def ajustar_ambiente_modo6(
    *,
    df,
    k_star,
    nr_pct,
    divergencia_s6_mc,
    risco_composto,
    previsibilidade="baixa",
):
    """
    Ajusta volumes do Modo 6 sem bloquear execução.
    Sempre retorna configuração válida.

    BLOCO UNIVERSAL C:
    - Não assume n = 6
    - Lê PC_N_EFETIVO e PC_UNIVERSO_ATIVO se existirem
    - Não força alteração de comportamento
    """

    # --------------------------------------------------------
    # Leitura do fenômeno ativo (Blocos A + B + C)
    # --------------------------------------------------------
    pc_n_efetivo = st.session_state.get("PC_N_EFETIVO")
    pc_universo = st.session_state.get("PC_UNIVERSO_ATIVO")

    # --------------------------------------------------------
    # Valores base (comportamento LEGADO preservado)
    # --------------------------------------------------------
    volume_min = 3
    volume_recomendado = 6
    volume_max = 80

    # --------------------------------------------------------
    # Ajuste simples por previsibilidade (V16)
    # --------------------------------------------------------
    if previsibilidade == "alta":
        volume_min = 6
        volume_recomendado = 12
        volume_max = 40
    elif previsibilidade == "baixa":
        volume_min = 3
        volume_recomendado = 6
        volume_max = 20

    # --------------------------------------------------------
    # Ajuste UNIVERSAL SUAVE (não forçador)
    # --------------------------------------------------------
    aviso_universal = ""

    if pc_n_efetivo is not None:
        aviso_universal += f" | Fenômeno n={pc_n_efetivo}"

        # Regra conservadora:
        # quanto maior n, menor o volume máximo recomendado
        if pc_n_efetivo > 6:
            volume_max = min(volume_max, 20)
            volume_recomendado = min(volume_recomendado, 6)
            aviso_universal += " (redução preventiva)"

        elif pc_n_efetivo < 6:
            # Fenômenos menores toleram leve expansão
            volume_max = min(volume_max, 40)
            aviso_universal += " (fenômeno compacto)"

    if pc_universo is not None:
        u_min, u_max = pc_universo
        aviso_universal += f" | Univ:{u_min}-{u_max}"

    # --------------------------------------------------------
    # Retorno PADRÃO (compatível com todo o app)
    # --------------------------------------------------------
    return {
        "volume_min": volume_min,
        "volume_recomendado": volume_recomendado,
        "volume_max": volume_max,
        "confiabilidade_estimada": 0.05,
        "aviso_curto": (
            f"Modo 6 ativo | Volumes: "
            f"{volume_min}/{volume_recomendado}/{volume_max}"
            f"{aviso_universal}"
        ),
    }

# ============================================================
# <<< FIM — FUNÇÃO AUXILIAR — AJUSTE DE AMBIENTE PARA MODO 6
# ============================================================


# ============================================================
# GATILHO ECO — OBSERVADOR PASSIVO (V16 PREMIUM)
# NÃO decide | NÃO expande | NÃO altera volumes
# Apenas sinaliza prontidão para ECO
# (UNIVERSAL — consciente do fenômeno)
# ============================================================

def avaliar_gatilho_eco(
    k_star_atual: float,
    nr_pct: float,
    divergencia_s6_mc: float,
):
    """
    Avalia se o ambiente está tecnicamente pronto para ECO.
    BLOCO UNIVERSAL C:
    - Leitura do fenômeno ativo
    - Nenhuma decisão automática
    """

    pc_n_efetivo = st.session_state.get("PC_N_EFETIVO")
    pc_universo = st.session_state.get("PC_UNIVERSO_ATIVO")

    pronto_eco = False
    motivos = []

    # --------------------------------------------------------
    # Critérios técnicos (LEGADOS)
    # --------------------------------------------------------
    if k_star_atual < 0.15:
        motivos.append("k* favorável")

    if nr_pct < 0.30:
        motivos.append("ruído controlado")

    if divergencia_s6_mc < 5.0:
        motivos.append("baixa divergência S6 vs MC")

    if len(motivos) >= 2:
        pronto_eco = True

    # --------------------------------------------------------
    # Informação universal (observacional)
    # --------------------------------------------------------
    info_universal = ""

    if pc_n_efetivo is not None:
        info_universal += f" | Fenômeno n={pc_n_efetivo}"

    if pc_universo is not None:
        u_min, u_max = pc_universo
        info_universal += f" | Univ:{u_min}-{u_max}"

    return {
        "pronto_eco": pronto_eco,
        "motivos": motivos,
        "mensagem": (
            "ECO tecnicamente possível"
            if pronto_eco
            else "ECO ainda não recomendado"
        )
        + info_universal,
    }

# ============================================================
# <<< FIM — GATILHO ECO — OBSERVADOR PASSIVO (V16 PREMIUM)
# ============================================================






# ============================================================
# MÓDULO 3 — EXPECTATIVA HISTÓRICA (CONTEXTO DO MOMENTO) — V16
# Observacional | Retrospectivo | NÃO decide | NÃO gera listas
# Depende de S2 (Pipeline) + S3 (Diagnóstico de Risco mínimo: k*/NR%/classe)
# ============================================================

M3_PAINEL_EXPECTATIVA_NOME = "📈 Expectativa Histórica — Contexto do Momento (V16)"

# ============================================================
# MÓDULO 5 (V16) — “PULO DO GATO” OPERACIONAL
# Coleta automática de estados (para dar massa ao M2/M3)
# ============================================================

M5_PAINEL_PULO_GATO_NOME = "🧠 M5 — Pulo do Gato (Coleta Automática de Estados)"


def _m5_identidade_historico_para_coleta(df_full, n_alvo, universo_min, universo_max):
    """ID estável (best-effort) para limitar coleta por histórico sem depender de hash pesado."""
    try:
        tam = int(len(df_full)) if df_full is not None else -1
    except Exception:
        tam = -1
    return f"H|n={n_alvo}|U={universo_min}-{universo_max}|len={tam}"


def _m5_leitura_regime_light(df_cut, universo_min, universo_max):
    """Leitura LIGHT (rápida) para regime/volatilidade sem rodar pipeline completo.

    Objetivo: registrar um *sinal* coerente para M2 (não substituir o pipeline).
    """
    try:
        # janela curta para captar irregularidade recente
        w = min(120, max(30, int(len(df_cut) * 0.05)))
        dfw = df_cut.tail(w)
        # tenta extrair colunas numéricas de passageiros
        cols_num = [c for c in dfw.columns if str(c).strip().isdigit()]
        if not cols_num:
            # fallback: tenta padrão comum (p1..p6)
            cols_num = [c for c in dfw.columns if str(c).lower().startswith("p")]
        vals = []
        for _, row in dfw.iterrows():
            linha = []
            for c in cols_num:
                try:
                    v = int(row[c])
                    if universo_min <= v <= universo_max:
                        linha.append(v)
                except Exception:
                    pass
            vals.append(len(set(linha)))
        if not vals:
            return "N/D", None, None

        # “energia” simples = diversidade média normalizada
        u = max(1, (universo_max - universo_min + 1))
        energia = float(np.mean(vals)) / float(min(u, 60))
        # “volatilidade” simples = desvio da diversidade
        volatilidade = float(np.std(vals)) / float(max(1, np.mean(vals)))

        if volatilidade >= 0.28:
            regime = "🟥 Estrada Quente (Alta volatilidade)"
        elif volatilidade >= 0.18:
            regime = "🟨 Estrada Mista / Irregular"
        else:
            regime = "🟩 Estrada Neutra / Estável"

        return regime, energia, volatilidade
    except Exception:
        return "N/D", None, None


def m5_painel_pulo_do_gato_v16():
    """Painel canônico: coleta automática de estados para dar massa mínima ao M2/M3.

    - Não mexe no núcleo
    - Não decide
    - Não substitui pipeline/monitor
    """
    st.subheader("🧠 M5 — Pulo do Gato (Coleta Automática de Estados)")
    st.caption("Coleta assistida para preencher Memória de Estados (M2) e habilitar Expectativa Histórica (M3) sem exigir que você rode manualmente dezenas de vezes.")

    df_full = st.session_state.get("historico_df")
    if df_full is None or len(df_full) < 50:
        st.warning("Carregue um histórico válido antes de usar o M5.")
        return

    n_alvo = int(st.session_state.get("n_alvo", 6) or 6)
    universo_min = int(st.session_state.get("universo_min", 0) or 0)
    universo_max = int(st.session_state.get("universo_max", 50) or 50)

    # limites (anti-zumbi canônico do M5)
    max_sessao = int(st.session_state.get("m5_max_por_sessao", 25) or 25)
    max_hist = int(st.session_state.get("m5_max_por_historico", 50) or 50)

    st.markdown("""
**Como funciona (sem mágica):**

- Você escolhe quantas “fotos” o sistema deve coletar.
- Cada foto simula um ponto de corte do histórico (C... menor) e registra **um estado S3** com um *regime light*.
- Isso alimenta o M2; o M3 passa a ter base mínima para expectativa histórica.

**Importante:**

- O M5 **não** roda o Pipeline completo em cada corte (para não zumbizar).
- Ele registra um sinal *light* e honesto para memória.
""")

    restante_sessao = max(0, max_sessao - int(st.session_state.get("m5_contador_sessao", 0) or 0))
    hist_id = _m5_identidade_historico_para_coleta(df_full, n_alvo, universo_min, universo_max)
    por_hist = st.session_state.get("m5_contador_por_historico", {})
    restante_hist = max(0, max_hist - int(por_hist.get(hist_id, 0) or 0))

    st.info(f"Limites ativos: Máx. por sessão: {max_sessao} (restante: {restante_sessao}) · Máx. por histórico: {max_hist} (restante: {restante_hist})")

    if restante_sessao <= 0 or restante_hist <= 0:
        st.warning("Limite atingido. Para coletar mais: reinicie a sessão (para limite por sessão) ou troque o histórico (para limite por histórico).")
        return

    n_solicitado = st.slider("Quantas fotos coletar agora?", 1, min(25, restante_sessao, restante_hist), 12)
    passo = st.slider("Passo entre cortes (em séries)", 1, 10, 1)
    janela_recente = st.slider("Janela recente para coleta (últimas séries)", 60, 600, 180, step=30)

    btn = st.button("📸 Coletar agora (M5)")

    if not btn:
        return

    # prepara M2
    if "m2_memoria_estados" not in st.session_state:
        st.session_state["m2_memoria_estados"] = []

    # pontos de corte: fatiamos a janela recente para não varrer o histórico inteiro (anti-zumbi)
    total = len(df_full)
    base_ini = max(50, total - int(janela_recente))
    pontos = list(range(total, base_ini, -int(passo)))
    pontos = pontos[: int(n_solicitado)]

    adicionados = 0
    falhas = 0

    for cut_len in pontos:
        try:
            df_cut = df_full.iloc[:cut_len].copy()
            regime_light, energia_light, vol_light = _m5_leitura_regime_light(df_cut, universo_min, universo_max)

            registro = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "n_alvo": int(n_alvo),
                "universo": f"{universo_min}–{universo_max}",
                "tamanho": int(cut_len),
                "estado": "S3",
                "regime": regime_light,
                "energia_light": energia_light,
                "volatilidade_light": vol_light,
                "origem": "M5",
            }
            st.session_state["m2_memoria_estados"].append(registro)
            adicionados += 1
        except Exception:
            falhas += 1
            continue

    # atualiza contadores
    st.session_state["m5_contador_sessao"] = int(st.session_state.get("m5_contador_sessao", 0) or 0) + int(adicionados)
    por_hist = st.session_state.get("m5_contador_por_historico", {})
    por_hist[hist_id] = int(por_hist.get(hist_id, 0) or 0) + int(adicionados)
    st.session_state["m5_contador_por_historico"] = por_hist

    st.success(f"M5 concluído: {adicionados} fotos adicionadas à Memória de Estados (M2).")
    if falhas:
        st.caption(f"Falhas silenciosas (válidas): {falhas}")

    st.markdown("""
**Próximo passo canônico:**

1) Abra **📈 Expectativa Histórica — Contexto do Momento (V16)**
2) Veja se o N mínimo já foi atingido.
""")


# ============================================================
# M5 — ALIAS CANÔNICO (compatibilidade de chamada)
# ============================================================

def m5_painel_pulo_do_gato_operacional():
    """Painel canônico (nome histórico).

    Mantém compatibilidade com chamadas antigas/âncoras que apontam para
    `m5_painel_pulo_do_gato_operacional()`.
    """
    return m5_painel_pulo_do_gato_v16()


def _m3_has_s2_pipeline() -> bool:
    return bool(st.session_state.get("pipeline_flex_ultra_concluido") or st.session_state.get("pipeline_executado") or st.session_state.get("m1_ts_pipeline_ok"))


def _m3_has_s3_risco_minimo() -> bool:
    risco = st.session_state.get("diagnostico_risco") or {}

    k_star = risco.get("k_star", risco.get("kstar", None))
    nr = risco.get("nr_percent", None)
    classe = risco.get("classe_risco", None)

    # fallback: algumas chaves aparecem fora do pack
    if k_star is None:
        k_star = st.session_state.get("k_star")
    if nr is None:
        nr = st.session_state.get("nr_percent")
    if classe is None:
        classe = st.session_state.get("classe_risco")

    ok_k = isinstance(k_star, (int, float))
    ok_nr = isinstance(nr, (int, float))
    ok_classe = isinstance(classe, str) and classe.strip() not in ("", "N/D")

    return bool(ok_k and ok_nr and ok_classe)


def _m3_norm_int(v):
    try:
        return int(float(str(v).strip().replace(",", ".")))
    except Exception:
        return None


def _m3_dx_janela(df_window, cols_pass):
    vals = []
    for c in cols_pass:
        s = [_m3_norm_int(x) for x in df_window[c].values]
        s = [x for x in s if x is not None]
        if len(s) >= 2:
            try:
                vals.append(float(np.std(s, ddof=1)))
            except Exception:
                pass
    if not vals:
        return None
    try:
        return float(np.mean(vals))
    except Exception:
        return None


def _m3_classificar_regime_dx(dx, q1, q2):
    if dx is None:
        return "N/D"
    if dx <= q1:
        return "ECO"
    if dx <= q2:
        return "PRÉ-ECO"
    return "RUIM"


def m3_painel_expectativa_historica_contexto():
    st.markdown("## 📈 Expectativa Histórica — Contexto do Momento (V16)")
    st.caption(
        "Observacional e retrospectivo. Não gera listas. Não muda motores. "
        "Serve para responder: *em momentos parecidos no passado, o que costuma acontecer nas próximas 1–3 séries?*"
    )

    # --------------------------------------------------------
    # Governança (dependências)
    # --------------------------------------------------------
    # ============================================================
    # 📤 Auditoria externa (opcional) — NÃO é necessária no uso normal
    # ============================================================
    with st.expander("📤 Auditoria externa (opcional) — usar outro histórico (sem substituir a sessão)", expanded=False):
        st.caption("Uso normal: este painel usa o histórico já carregado na sessão.\n"
                   "Este uploader é apenas para auditoria/estudo com outro arquivo, sem afetar o histórico atual.")
        arquivo = st.file_uploader("Envie um histórico FLEX ULTRA (opcional)", type=["csv", "txt"], key="m3_upload_auditoria")
        if arquivo is not None:
            try:
                df_aud = carregar_historico_flex_ultra(arquivo)
                st.success("Histórico de auditoria carregado (não substitui a sessão).")
                metricas = calcular_metricas_basicas_historico(df_aud)
                exibir_resumo_inicial_historico(metricas)
                st.info("✅ Auditoria concluída. Para operar o fluxo normal, use o histórico carregado em 📁/📄 Carregar Histórico.")
            except Exception as e:
                st.error(f"Falha ao carregar histórico de auditoria: {e}")

    # --------------------------------------------------------
    # 1) Calcula dx nas janelas recentes para quantis
    # --------------------------------------------------------
    dx_list = []
    dx_por_t = {}

    # --------------------------------------------------------
    # Fonte CANÔNICA: histórico da sessão + colunas do Pipeline
    # (M3 é observacional: não gera listas, não decide)
    # --------------------------------------------------------
    if "pipeline_col_pass" not in st.session_state:
        st.warning("Execute primeiro o painel 🛣️ Pipeline V14-FLEX ULTRA (fonte canônica de passageiros).")
        return

    cols_pass = st.session_state["pipeline_col_pass"]
    nome_df, df_base = v16_identificar_df_base()
    if df_base is None:
        st.warning("Histórico não encontrado. Carregue o histórico e rode o Pipeline.")
        return

    # Janela canônica (alinhada aos painéis V16 Premium PRÉ-ECO/ECO)
    W = 60
    n = int(len(df_base))
    if n < (W + 5):
        st.warning("Histórico insuficiente para Expectativa Histórica (M3).")
        return

    # Intervalo automático (sem sliders): recorte recente suficiente para quantis e estabilidade
    t_final = n - 1
    max_janelas = min(600, max(180, n - (W + 2)))  # 180–600 janelas, conforme tamanho do histórico
    t_inicial = max(W, t_final - max_janelas)

    for t in range(t_inicial, t_final + 1):
        wdf = df_base.iloc[t - W : t]
        dx = _m3_dx_janela(wdf, cols_pass)
        if dx is not None:
            dx_list.append(dx)
            dx_por_t[t] = dx

    if len(dx_list) < 120:
        st.warning("⚠️ Poucas janelas válidas para estimar quantis com estabilidade. (Resultados ainda são informativos.)")

    try:
        q1 = float(np.quantile(dx_list, 0.33))
        q2 = float(np.quantile(dx_list, 0.66))
    except Exception:
        st.error("❌ Falha ao calcular quantis do dx (dados insuficientes ou inválidos).")
        return

    dx_atual = dx_por_t.get(t_final)
    regime_atual = _m3_classificar_regime_dx(dx_atual, q1, q2)

    # --------------------------------------------------------
    # 2) Regime por t (mesma regra do painel PRÉ-ECO→ECO)
    # --------------------------------------------------------
    regime_por_t = {}
    for t, dx in dx_por_t.items():
        regime_por_t[t] = _m3_classificar_regime_dx(dx, q1, q2)

    # --------------------------------------------------------
    # 3) Expectativa: quando estava no MESMO regime, o que ocorreu em 1–3 séries?
    # --------------------------------------------------------
    total = 0
    vira_eco_1 = 0
    vira_eco_2 = 0
    vira_eco_3 = 0

    permanece_mesmo_1 = 0
    permanece_mesmo_2 = 0
    permanece_mesmo_3 = 0

    for t in sorted(regime_por_t.keys()):
        if t + 1 > t_final:
            continue
        if regime_por_t[t] != regime_atual:
            continue

        total += 1

        r1 = regime_por_t.get(t + 1)
        r2 = regime_por_t.get(t + 2)
        r3 = regime_por_t.get(t + 3)

        if r1 == "ECO":
            vira_eco_1 += 1
        if (r1 == "ECO") or (r2 == "ECO"):
            vira_eco_2 += 1
        if (r1 == "ECO") or (r2 == "ECO") or (r3 == "ECO"):
            vira_eco_3 += 1

        if r1 == regime_atual:
            permanece_mesmo_1 += 1
        if (r1 == regime_atual) or (r2 == regime_atual):
            permanece_mesmo_2 += 1
        if (r1 == regime_atual) or (r2 == regime_atual) or (r3 == regime_atual):
            permanece_mesmo_3 += 1

    # --------------------------------------------------------
    # 4) Exibição
    # --------------------------------------------------------
    st.markdown("### 🧭 Momento atual (classificação por dx)")

    colA, colB, colC = st.columns(3)
    colA.metric("dx (janela)", f"{dx_atual:.6f}" if isinstance(dx_atual, (int, float)) else "N/D")
    colB.metric("Regime (dx)", regime_atual)
    colC.metric("Janelas analisadas", f"{len(dx_list)}")

    st.caption(
        "Regra: dx ≤ q1 → ECO | dx ≤ q2 → PRÉ-ECO | dx > q2 → RUIM. "
        "(Quantis calculados nas últimas janelas, com anti-zumbi interno.)"
    )

    st.markdown("### 📈 Expectativa histórica (condicional ao regime atual)")

    if total == 0:
        st.warning("⚠️ Não houve eventos suficientes no histórico para estimar expectativa condicional neste regime.")
        return

    taxa_eco_1 = vira_eco_1 / total
    taxa_eco_2 = vira_eco_2 / total
    taxa_eco_3 = vira_eco_3 / total

    taxa_perm_1 = permanece_mesmo_1 / total
    taxa_perm_2 = permanece_mesmo_2 / total
    taxa_perm_3 = permanece_mesmo_3 / total

    df_out = pd.DataFrame(
        [
            {
                "Regime atual (dx)": regime_atual,
                "Eventos similares": int(total),
                "Vira ECO em 1": round(taxa_eco_1, 4),
                "Vira ECO em 2": round(taxa_eco_2, 4),
                "Vira ECO em 3": round(taxa_eco_3, 4),
                "Permanece no mesmo (1)": round(taxa_perm_1, 4),
                "Permanece no mesmo (2)": round(taxa_perm_2, 4),
                "Permanece no mesmo (3)": round(taxa_perm_3, 4),
            }
        ]
    )

    st.dataframe(df_out, use_container_width=True, hide_index=True)



    # --- M3: exporta um resumo mínimo para uso em outros painéis (read-only)

    try:

        st.session_state["m3_ts"] = datetime.utcnow().isoformat() + "Z"

        st.session_state["m3_regime_dx"] = regime_atual

        st.session_state["m3_eventos_similares"] = int(total)

        st.session_state["m3_taxa_eco1"] = float(taxa_eco_1)

        st.session_state["m3_taxa_estado_bom"] = float(taxa_estado_bom)

        st.session_state["m3_taxa_transicao"] = float(taxa_transicao)

    except Exception:

        pass
    st.info("📌 Interpretação correta (sem viés):\n- Isso NÃO prevê o próximo alvo.\n- Isso mede *o que costuma acontecer* quando o ambiente cai no mesmo tipo de regime.\n- Serve para calibrar expectativa, postura e paciência — não para aumentar convicção por '3 acertos'.")

    st.markdown(
        "Envie um arquivo de histórico em formato **FLEX ULTRA**.\n\n"
        "📌 Regra universal: o **último valor da linha é sempre k**, "
        "independente da quantidade de passageiros."
    )

    arquivo = st.file_uploader(
        "Envie o arquivo de histórico",
        type=["txt", "csv"],
    )

    if arquivo is None:
        exibir_bloco_mensagem(
            "Aguardando arquivo de histórico",
            "Envie seu arquivo para iniciar o processamento do PredictCars.",
            tipo="info",
        )
        st.stop()

    try:
        conteudo = arquivo.getvalue().decode("utf-8")
        linhas = conteudo.strip().split("\n")

        if not limitar_operacao(
            len(linhas),
            limite_series=LIMITE_SERIES_REPLAY_ULTRA,
            contexto="Carregar Histórico (Arquivo)",
            painel="📁 Carregar Histórico (Arquivo)",
        ):
            st.stop()

        df = carregar_historico_universal(linhas)

    except Exception as erro:
        exibir_bloco_mensagem(
            "Erro ao processar histórico",
            f"Detalhes técnicos: {erro}",
            tipo="error",
        )
        st.stop()

    st.session_state["historico_df"] = df

    # 🔎 Universo (1–50 / 1–60) — registro canônico para snapshot/RF

    v16_registrar_universo_session_state(st.session_state["historico_df"], n_alvo=st.session_state.get("n_alvo", 6))

    metricas = calcular_metricas_basicas_historico(df)
    exibir_resumo_inicial_historico(metricas)

    # ============================================================
    # 🌐 BLOCO UNIVERSAL A — DETECTOR DO FENÔMENO
    # ============================================================

    st.markdown("### 🌐 Perfil do Fenômeno (detecção automática)")
    st.caption(
        "Detecção automática do formato real do fenômeno.\n"
        "✔ Última coluna = k\n"
        "✔ Quantidade de passageiros livre\n"
        "✔ Universo variável\n"
        "❌ Não há decisão automática"
    )

    import hashlib

    colunas = list(df.columns)
    col_id = colunas[0]
    col_k = colunas[-1]
    col_passageiros = colunas[1:-1]

    passageiros_por_linha = []
    todos_passageiros = []

    for _, row in df.iterrows():
        valores = [int(v) for v in row[col_passageiros] if pd.notna(v)]
        passageiros_por_linha.append(len(valores))
        todos_passageiros.extend(valores)

    n_set = sorted(set(passageiros_por_linha))
    mix_n_detectado = len(n_set) > 1
    n_passageiros = n_set[0] if not mix_n_detectado else None

    universo_min = int(min(todos_passageiros)) if todos_passageiros else None
    universo_max = int(max(todos_passageiros)) if todos_passageiros else None
    universo_set = sorted(set(todos_passageiros))

    hash_base = f"{n_set}-{universo_min}-{universo_max}"
    fenomeno_id = hashlib.md5(hash_base.encode()).hexdigest()[:8]

    st.session_state["pc_n_passageiros"] = n_passageiros
    st.session_state["pc_n_set_detectado"] = n_set
    st.session_state["pc_mix_n_detectado"] = mix_n_detectado
    st.session_state["pc_universo_min"] = universo_min
    st.session_state["pc_universo_max"] = universo_max
    st.session_state["pc_universo_set"] = universo_set
    st.session_state["pc_fenomeno_id"] = fenomeno_id

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📐 Estrutura**")
        st.write(f"Passageiros por série (n): **{n_set}**")
        if mix_n_detectado:
            st.warning("Mistura de n detectada no mesmo histórico.")
        st.write(f"Coluna ID: `{col_id}`")
        st.write(f"Coluna k: `{col_k}`")

    with col2:
        st.markdown("**🌍 Universo observado**")
        st.write(f"Mínimo: **{universo_min}**")
        st.write(f"Máximo: **{universo_max}**")
        st.write(f"Total distintos: **{len(universo_set)}**")

    st.markdown("**🆔 Fenômeno ID (auditoria)**")
    st.code(fenomeno_id)

    # ============================================================
    # 🌐 BLOCO UNIVERSAL B — PARAMETRIZAÇÃO DO FENÔMENO
    # ============================================================

    st.markdown("### 🌐 Parâmetros Ativos do Fenômeno")
    st.caption(
        "Parâmetros universais derivados do histórico.\n"
        "✔ Não executa\n"
        "✔ Não interfere\n"
        "✔ Não altera módulos existentes"
    )

    if not mix_n_detectado:
        pc_n_alvo = n_passageiros
        pc_n_status = "fixo"
    else:
        pc_n_alvo = None
        pc_n_status = "misto"

    st.session_state["pc_n_alvo"] = pc_n_alvo
    st.session_state["pc_range_min"] = universo_min
    st.session_state["pc_range_max"] = universo_max

    if pc_n_alvo:
        st.session_state["pc_regua_extrema"] = f"{pc_n_alvo} ou nada"
        st.session_state["pc_regua_mvp2"] = f"2–{pc_n_alvo}"
    else:
        st.session_state["pc_regua_extrema"] = "indefinida"
        st.session_state["pc_regua_mvp2"] = "indefinida"

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**🎯 n alvo**")
        st.write(f"Status: **{pc_n_status}**")
        st.write(f"n alvo: **{pc_n_alvo if pc_n_alvo else 'MISTO'}**")

    with col4:
        st.markdown("**📏 Universo ativo**")
        st.write(f"{universo_min} – {universo_max}")
        st.write("Origem: histórico observado")

    if mix_n_detectado:
        st.warning(
            "⚠️ Histórico contém mistura de quantidades de passageiros.\n\n"
            "Recomenda-se separar fenômenos antes de previsões."
        )

    if pc_n_alvo and pc_n_alvo != 6:
        st.info(
            f"ℹ️ Fenômeno com n = {pc_n_alvo} detectado.\n"
            "Módulos legados ainda podem assumir n=6.\n"
            "➡️ Próximo passo: BLOCO UNIVERSAL C."
        )

    st.success("Perfil e parâmetros do fenômeno definidos.")

    st.success("Histórico carregado com sucesso!")
    st.dataframe(df.head(20))


# ============================================================

# ============================================================
# Painel 1A — 📁 Carregar Histórico (Arquivo)
# ============================================================
if painel == "📁 Carregar Histórico (Arquivo)":

    st.markdown("## 📁 Carregar Histórico — Arquivo (V15.7 MAX)")
    st.caption(
        "Carregamento canônico via arquivo FLEX ULTRA (TXT/CSV).\n"
        "✔ Não muda motores\n"
        "✔ Não decide\n"
        "✔ Alimenta a sessão atual"
    )

    up = st.file_uploader(
        "Envie um histórico FLEX ULTRA",
        type=["txt", "csv"],
        accept_multiple_files=False,
    )

    if up is None:
        st.info("Envie um arquivo para iniciar o processamento do PredictCars.")
        st.stop()

    try:
        raw = up.getvalue()
        try:
            txt = raw.decode("utf-8")
        except Exception:
            txt = raw.decode("latin-1", errors="ignore")

        linhas = [l.strip() for l in txt.splitlines() if l.strip()]
        if not linhas:
            st.error("Histórico vazio")
            st.stop()

        df = carregar_historico_universal(linhas)
        st.session_state["historico_df"] = df
        # 🔎 Universo (1–50 / 1–60) — registro canônico para snapshot/RF
        v16_registrar_universo_session_state(st.session_state["historico_df"], n_alvo=st.session_state.get("n_alvo", 6))

        # Sincroniza chaves canônicas (evita N/D indevido no RF)
        try:
            v16_sync_aliases_canonicos()
        except Exception:
            pass

        umin = st.session_state.get("universo_min")
        umax = st.session_state.get("universo_max")
        if umin is not None and umax is not None:
            st.success(f"Histórico carregado com sucesso: {len(df)} séries | Universo detectado: {umin}–{umax}")
        else:
            umin = st.session_state.get("universo_min")
        umax = st.session_state.get("universo_max")
        if umin is not None and umax is not None:
            st.success(f"Histórico carregado com sucesso: {len(df)} séries | Universo detectado: {umin}–{umax}")
        else:
            st.success(f"Histórico carregado com sucesso: {len(df)} séries")

    except Exception as e:
        st.error(f"Erro ao processar histórico: {e}")

    st.stop()

# Painel 1B — 📄 Carregar Histórico (Colar)
# ============================================================
if "Carregar Histórico (Colar)" in str(painel):

    st.markdown("## 📄 Carregar Histórico — Copiar e Colar (V15.7 MAX)")

    texto = st.text_area(
        "Cole aqui o histórico completo",
        height=320,
        key="pc_colar_texto_simples",
    )

    clicked = st.button(
        "📥 Processar Histórico (Copiar e Colar)",
        key="pc_colar_btn_simples",
    )

    if clicked:

        st.write("PROCESSANDO HISTÓRICO...")

        if not texto.strip():
            st.error("Histórico vazio")
            st.stop()

        linhas = texto.strip().split("\n")

        df = carregar_historico_universal(linhas)

        st.session_state["historico_df"] = df

        # 🔎 Universo (1–50 / 1–60) — registro canônico para snapshot/RF

        v16_registrar_universo_session_state(st.session_state["historico_df"], n_alvo=st.session_state.get("n_alvo", 6))

        st.success(f"Histórico carregado com sucesso: {len(df)} séries")





# ============================================================
# BLOCO — OBSERVADOR HISTÓRICO DE EVENTOS k (V16)
# FASE 1 — OBSERVAÇÃO PURA | SEM IMPACTO OPERACIONAL
# ============================================================






# ============================================================
# PAINEL — 📊 V16 PREMIUM — ERRO POR REGIME (RETROSPECTIVO)
# (INSTRUMENTAÇÃO: mede continuidade do erro por janelas)
# ============================================================

# ============================================================
# PAINEL — 🧠 Diagnóstico ECO & Estado (V16)
# Observacional | NÃO decide | NÃO altera motores
# ============================================================

elif painel == "🧠 Diagnóstico ECO & Estado (V16)":

    st.markdown("## 🧠 Diagnóstico ECO & Estado — V16")
    st.caption("Leitura mastigada do ambiente e do alvo. Observacional.")

    # Sincroniza chaves canônicas (evita N/D indevido no RF)
    v16_sync_aliases_canonicos()


    diag = st.session_state.get("diagnostico_eco_estado_v16")

    if not diag:
        st.info("Diagnóstico ainda não disponível. Carregue um histórico.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌊 ECO")
        eco_forca = diag.get("eco_forca") or "fraco"
        st.write(f"Força: **{eco_forca}**")
        st.write(f"Persistência: **{diag.get('eco_persistencia')}**")
        st.write(f"Acionabilidade: **{diag.get('eco_acionabilidade')}**")

        motivos = diag.get("motivos_eco", [])
        if motivos:
            st.caption("Motivos: " + ", ".join(motivos))

        contradicoes = diag.get("contradicoes", [])
        if contradicoes:
            st.warning("⚠️ Contradições: " + "; ".join(contradicoes))

    with col2:
        st.markdown("### 🐟 Estado do Alvo")
        st.write(f"Estado: **{diag.get('estado')}**")
        st.write(
            "Confiabilidade: "
            f"**{'alta' if diag.get('estado_confiavel') else 'baixa / transição'}**"
        )

    st.markdown("### 🧠 Leitura Geral")
    
    st.success(diag.get("leitura_geral", "—"))


elif painel == "🧾 APS — Auditoria de Postura (V16)":

    st.markdown("## 🧾 APS — Auditoria de Postura (V16)")
    st.caption("Auditoria observacional do risco/postura do sistema. NÃO muda listas. NÃO decide volume. Serve para proteger contra postura errada (ex.: ancoragem excessiva em E0 + ruído alto).")

    # Coleta segura
    nr = st.session_state.get("nr_percent_v16") or st.session_state.get("nr_percent") or st.session_state.get("NR_PERCENT")
    orbita = st.session_state.get("orbita_selo_v16") or st.session_state.get("orbita_selo") or st.session_state.get("ORBITA_SELO") or "E0"
    diag = st.session_state.get("diagnostico_eco_estado_v16") or {}
    eco_acion = diag.get("eco_acionabilidade") or "N/D"

    anti_exato = st.session_state.get("anti_exato_level_v16") or st.session_state.get("anti_exato_level")  # opcional

    selo, titulo, msg = v16_calcular_aps_postura(nr_percent=nr, orbita_selo=orbita, eco_acionabilidade=eco_acion, anti_exato_level=anti_exato)

    # Registro canônico (observacional)

    try:

        st.session_state["aps_postura_selo"] = selo

        st.session_state["aps_postura_titulo"] = titulo

        st.session_state["aps_postura_msg"] = msg

    except Exception:

        pass

    st.markdown(f"### {selo} {titulo}")
    st.info(msg)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Entradas observadas**")
        st.write(f"- NR%: **{nr if nr is not None else 'N/D'}**")
        st.write(f"- Órbita (selo): **{orbita}**")
        st.write(f"- ECO (acionabilidade): **{eco_acion}**")
        if anti_exato is not None:
            st.write(f"- Anti-EXATO (nível): **{anti_exato}**")
    with col2:
        st.markdown("**Compatibilidades sugeridas (não obrigatórias)**")
        if selo == "🟡":
            st.write("- ✔ Duplo pacote: **base + anti-âncora**")
            st.write("- ✔ Envelope estreito / microvariações")
            st.write("- ⚠ Evitar ancoragem forte única")
        elif selo == "🔴":
            st.write("- ✔ Pacote mais espalhado e volume baixo")
            st.write("- ⚠ Evitar densidade e insistência")
        elif selo == "🟢":
            st.write("- ✔ Densidade moderada pode ser compatível")
            st.write("- ✔ Observar persistência por 1–3 séries")
        else:
            st.write("- ✔ Manter pacote base e acompanhar série a série")

    st.markdown("### 📌 Nota de governança")
    st.caption("Se a APS 'apontar o dedo', o sistema NÃO muda nada automaticamente nesta fase. A função aqui é blindar leitura e evitar postura errada; a execução segue com os pacotes já gerados.")


elif painel == "🧭 RMO/DMO — Retrato do Momento (V16)":
    st.markdown("## 🧭 RMO/DMO — Retrato do Momento (V16)")
    st.caption("Síntese integrada (RMO) + governança temporal (DMO) + voz operacional (VOS). Observacional. Não decide ação.")

    # Sincroniza chaves canônicas (ECO/Estado/k*/Divergência) antes do retrato
    v16_sync_aliases_canonicos()


    # -----------------------------
    # Coleta segura de sinais
    # -----------------------------
    risco_pack = st.session_state.get("diagnostico_risco") or {}
    diag = st.session_state.get("diagnostico_eco_estado_v16") or {}

    nr_ruido = st.session_state.get("nr_percent")  # Painel de Ruído Condicional (normalizado)
    nr_risco = risco_pack.get("nr_percent")        # Monitor de risco (NR% usado no índice)
    div = risco_pack.get("divergencia")
    classe_risco = risco_pack.get("classe_risco")
    indice_risco = risco_pack.get("indice_risco")

    orb = st.session_state.get("orbita_info") or {}
    orb_estado = orb.get("estado", "N/D")
    orb_selo = orb.get("selo", "N/D")
    grad = st.session_state.get("orbita_gradiente", "N/D")
    orb_score = st.session_state.get("orbita_score")

    eco_forca = diag.get("eco_forca", diag.get("forca", "N/D"))
    eco_persist = diag.get("eco_persistencia", diag.get("persistencia", "N/D"))
    eco_acion = diag.get("eco_acionabilidade", diag.get("acionabilidade", "N/D"))

    estado_alvo = diag.get("estado", "N/D")
    estado_conf = "alta" if diag.get("estado_confiavel") else "baixa / transição"

    b3_pronto = bool(st.session_state.get("b3_pronto_refinar", False))
    pipeline_ok = bool(st.session_state.get("pipeline_flex_ultra_concluido", False))
    turbo_ultra_rodou = bool(st.session_state.get("turbo_ultra_rodou", False))
    modo6_total = st.session_state.get("modo6_n_total")

    # -----------------------------
    # RMO — Retrato do Momento Operacional
    # -----------------------------
    st.markdown("### 🖼️ RMO — Retrato do Momento Operacional")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("NR% (Ruído)", f"{nr_ruido:.2f}%" if isinstance(nr_ruido, (int, float)) else "N/D")
        st.caption("Painel 📡 Ruído Condicional")
    with c2:
        st.metric("NR% (Risco)", f"{nr_risco:.2f}%" if isinstance(nr_risco, (int, float)) else "N/D")
        st.caption("Monitor k & k*")
    with c3:
        st.metric("Divergência S6×MC", f"{div:.4f}" if isinstance(div, (int, float)) else "N/D")
        st.caption("do Monitor (quando disponível)")
    with c4:
        st.metric("Índice de Risco", f"{indice_risco:.4f}" if isinstance(indice_risco, (int, float)) else "N/D")
        st.caption(classe_risco or "Classe N/D")

    st.markdown("#### 🌊 ECO & 🐟 Estado (leitura mastigada)")
    c5, c6, c7 = st.columns(3)
    with c5:
        st.write(f"**ECO**: {eco_forca} · {eco_persist} · {eco_acion}")
    with c6:
        st.write(f"**Estado do alvo**: {estado_alvo} (conf.: {estado_conf})")
    with c7:
        st.write(f"**Órbita**: {orb_estado} · {orb_selo} · grad {grad}")

    st.markdown("#### 🧱 Integridade operacional (sem julgamento)")
    c8, c9, c10 = st.columns(3)
    with c8:
        st.write(f"Pipeline FLEX ULTRA: **{'✅' if pipeline_ok else '—'}**")
    with c9:
        st.write(f"TURBO++ ULTRA rodou: **{'✅' if turbo_ultra_rodou else '—'}**")
    with c10:
        st.write(f"Modo 6 (N_total): **{modo6_total if modo6_total is not None else 'N/D'}**")

    st.markdown("#### 🧼 Perna B (prontidão)")
    st.write(f"B3 — Pronto para refinamento: **{'🟢 SIM' if b3_pronto else '🟡 AINDA NÃO'}**")

    # -----------------------------
    # DMO — Detector de Momento Operável (governança temporal)
    # -----------------------------
    st.markdown("### ⏳ DMO — Detector de Momento Operável (governança)")

    # histórico curto (memória leve, apenas dentro da sessão)
    if "dmo_hist_sinais" not in st.session_state:
        st.session_state["dmo_hist_sinais"] = []
    if "dmo_estado" not in st.session_state:
        st.session_state["dmo_estado"] = "🟥 SOBREVIVÊNCIA"

    sinais = []

    # Sinal A: Órbita sugere estrutura (E2 ou E1 forte via gradiente)
    if str(orb_estado).upper() in ["E2"]:
        sinais.append("Órbita E2 (interceptação plausível)")
    elif str(orb_estado).upper() in ["E1"] and str(grad).upper() in ["G2", "G3", "FORTE"]:
        sinais.append("Órbita E1 forte (gradiente alto)")

    # Sinal B: ECO persistente e (mesmo que fraco) não recuando
    if str(eco_persist).lower() in ["persistente", "sim", "alta", "ok"]:
        sinais.append("ECO com persistência")

    # Sinal C: Ruído não está piorando (tendência curta)
    # (usa NR do Painel de Ruído, quando disponível)
    hist = st.session_state["dmo_hist_sinais"]
    nr_ok = None
    try:
        if isinstance(nr_ruido, (int, float)):
            prev_nr = st.session_state.get("dmo_prev_nr_ruido")
            if isinstance(prev_nr, (int, float)):
                nr_ok = (nr_ruido <= prev_nr + 1e-9)
                if nr_ok:
                    sinais.append("NR não crescente (curto prazo)")
            st.session_state["dmo_prev_nr_ruido"] = float(nr_ruido)
    except Exception:
        pass

    # Sinal D: B3 pronto (refinamento viável)
    if b3_pronto:
        sinais.append("B3 pronto (refinamento viável)")

    # pontuação simples
    score = int(len(sinais))
    hist.append(score)
    hist[:] = hist[-5:]  # memória curta

    # regra de estados (consistente com o canônico)
    estado_atual = st.session_state.get("dmo_estado", "🟥 SOBREVIVÊNCIA")
    media2 = sum(hist[-2:]) / max(1, len(hist[-2:]))
    media3 = sum(hist[-3:]) / max(1, len(hist[-3:]))

    if len(hist) >= 3 and media3 >= 2.0:
        novo_estado = "🟩 OPERÁVEL"
    elif len(hist) >= 2 and media2 >= 1.0:
        novo_estado = "🟨 ATENÇÃO"
    else:
        novo_estado = "🟥 SOBREVIVÊNCIA"

    st.session_state["dmo_estado"] = novo_estado

    # exibição
    st.write(f"**Estado DMO:** {novo_estado}")
    st.caption("O DMO não decide ação. Ele governa o tempo (evita sair cedo demais).")

    if sinais:
        st.markdown("**Sinais ativos agora:**")
        for s in sinais:
            st.write(f"- {s}")
    else:
        st.markdown("**Sinais ativos agora:** nenhum (isso é um estado válido)")

    st.caption(f"Memória curta (scores últimas rodadas na sessão): {hist}")

    # -----------------------------
    # VOS — Voz Operacional do Sistema (1 frase, sem decisão)
    # -----------------------------
    st.markdown("### 🔊 VOS — Voz Operacional do Sistema (curta)")

    if novo_estado.startswith("🟥"):
        frase = "Ambiente não sustenta precisão. Permanecer ou trocar não altera o risco."
        st.warning(frase)
    elif novo_estado.startswith("🟨"):
        frase = "Estrutura começa a se repetir. Evite desmontar o que ainda está coerente."
        st.info(frase)
    else:
        frase = "Persistência custa menos que mudança. Reduza variação."
        st.success(frase)

    st.stop()

elif painel == "📊 V16 Premium — Erro por Regime (Retrospectivo)":

    st.subheader("📊 V16 Premium — Erro por Regime (Retrospectivo)")
    st.caption(
        "Instrumentação retrospectiva: janelas móveis → regime (ECO/PRE/RUIM) "
        "por dispersão da janela e erro da PRÓXIMA série como proxy de 'erro contido'. "
        "Não altera motor. Não escolhe passageiros."
    )

    # ============================================================
    # Localização ROBUSTA do histórico (padrão oficial V16)
    # ============================================================
    _, historico_df = v16_identificar_df_base()

    if historico_df is None or historico_df.empty:
        st.warning(
            "Histórico não encontrado no estado atual do app.\n\n"
            "👉 Recarregue o histórico e volte diretamente a este painel."
        )
        st.stop()

    if len(historico_df) < 100:
        st.warning(
            f"Histórico muito curto para análise retrospectiva.\n\n"
            f"Séries detectadas: {len(historico_df)}"
        )
        st.stop()

    # 🔒 Anti-zumbi automático (painel leve, invisível)
    janela = 60
    step = 1

    with st.spinner("Calculando análise retrospectiva por janelas (V16 Premium)..."):
        out = pc16_calcular_continuidade_por_janelas(
            historico_df=historico_df,
            janela=janela,
            step=step,
            usar_quantis=True
        )

    if not out.get("ok", False):
        st.error(f"Falha na análise: {out.get('motivo','Erro desconhecido')}")
        st.stop()

    resumo_geral = out.get("resumo_geral", {})
    resumo = out.get("resumo", {})
    df = out.get("df", pd.DataFrame())

    # ============================================================
    # RESULTADO OBJETIVO
    # ============================================================
    st.markdown("### ✅ Resultado objetivo — Continuidade do erro")

    diff = resumo_geral.get("diff_ruim_menos_eco_no_erro", None)
    if diff is None:
        st.info(
            "Ainda não há base suficiente para comparar ECO vs RUIM.\n\n"
            "Isso ocorre quando algum regime tem poucas janelas."
        )
    else:
        st.write(
            f"**Diferença RUIM − ECO no erro médio (erro_prox):** "
            f"`{diff:.6f}`\n\n"
            "➡️ Valores positivos indicam erro menor em ECO."
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de janelas", str(resumo_geral.get("n_total_janelas", "—")))
    col2.metric("Janela (W)", str(resumo_geral.get("janela", "—")))
    col3.metric("q1 dx (ECO ≤)", f"{resumo_geral.get('q1_dx', 0):.6f}")
    col4.metric("q2 dx (PRE ≤)", f"{resumo_geral.get('q2_dx', 0):.6f}")

    # ============================================================
    # TABELA POR REGIME
    # ============================================================
    st.markdown("### 🧭 Tabela por Regime (ECO / PRE / RUIM)")

    linhas = []
    for reg in ["ECO", "PRE", "RUIM"]:
        r = resumo.get(reg, {"n": 0})
        linhas.append({
            "Regime": reg,
            "n_janelas": r.get("n", 0),
            "dx_janela_medio": r.get("dx_janela_medio"),
            "erro_prox_medio": r.get("erro_prox_medio"),
            "erro_prox_mediana": r.get("erro_prox_mediana"),
        })

    df_reg = pd.DataFrame(linhas)
    st.dataframe(df_reg, use_container_width=True)

    # ============================================================
    # AUDITORIA LEVE
    # ============================================================
    st.markdown("### 🔎 Amostra das janelas (auditoria leve)")
    st.caption(
        "Exibe as primeiras linhas apenas para validação conceitual. "
        "`t` é um índice interno (0-based)."
    )
    st.dataframe(df.head(50), use_container_width=True)

    # ============================================================
    # LEITURA OPERACIONAL
    # ============================================================
    st.markdown("### 🧠 Leitura operacional (objetiva)")
    st.write(
        "- Se **ECO** apresentar **erro_prox_medio** consistentemente menor que **RUIM**, "
        "isso sustenta matematicamente que, em estados ECO, **o erro tende a permanecer contido**.\n"
        "- Este painel **não escolhe passageiros**.\n"
        "- Ele **autoriza** (ou não) a fase seguinte: **concentração para buscar 6**, "
        "sem alterar motor ou fluxo."
    )




# ============================================================
# PAINEL V16 — 🎯 Compressão do Alvo (OBSERVACIONAL)
# Leitura pura | NÃO prevê | NÃO decide | NÃO altera motores
# ============================================================

if painel == "🎯 Compressão do Alvo (Observacional)":

    st.markdown("## 🎯 Compressão do Alvo — Leitura Observacional (V16)")
    st.caption(
        "Este painel mede **se o erro provável está comprimindo**.\n\n"
        "⚠️ Não prevê números, não sugere volume, não altera o fluxo."
    )

    # -----------------------------
    # Coleta de sinais já existentes
    # -----------------------------
    nr = st.session_state.get("nr_percent")
    div = st.session_state.get("div_s6_mc")
    k_star = st.session_state.get("sentinela_kstar")
    risco = (st.session_state.get("diagnostico_risco") or {}).get("indice_risco")

    df = st.session_state.get("historico_df")

    if df is None or nr is None or div is None or k_star is None or risco is None:
        exibir_bloco_mensagem(
            "Pré-requisitos ausentes",
            "Execute os painéis de Sentinela, Ruído, Divergência e Monitor de Risco.",
            tipo="warning",
        )
        st.stop()

    # -----------------------------
    # 1) Estabilidade do ruído
    # -----------------------------
    nr_ok = nr < 45.0

    # -----------------------------
    # 2) Convergência dos motores
    # -----------------------------
    div_ok = div < 5.0

    # -----------------------------
    # 3) Regime não-hostil
    # -----------------------------
    risco_ok = risco < 0.55

    # -----------------------------
    # 4) k como marcador NORMAL (não extremo)
    # -----------------------------
    k_ok = 0.10 <= k_star <= 0.35

    # -----------------------------
    # 5) Repetição estrutural (passageiros)
    # -----------------------------
    col_pass = [c for c in df.columns if c.startswith("p")]
    ultimos = df[col_pass].iloc[-10:].values

    repeticoes = []
    for i in range(len(ultimos) - 1):
        repeticoes.append(len(set(ultimos[i]) & set(ultimos[i + 1])))

    repeticao_media = float(np.mean(repeticoes)) if repeticoes else 0.0
    repeticao_ok = repeticao_media >= 2.5

    # -----------------------------
    # Consolidação OBSERVACIONAL
    # -----------------------------
    sinais = {
        "NR% estável": nr_ok,
        "Convergência S6 × MC": div_ok,
        "Risco controlado": risco_ok,
        "k em faixa normal": k_ok,
        "Repetição estrutural": repeticao_ok,
    }

    positivos = sum(1 for v in sinais.values() if v)

    # -----------------------------
    # Exibição
    # -----------------------------
    st.markdown("### 📊 Sinais de Compressão do Erro")

    for nome, ok in sinais.items():
        st.markdown(
            f"- {'🟢' if ok else '🔴'} **{nome}**"
        )

    st.markdown("### 🧠 Leitura Consolidada")

    if positivos >= 4:
        leitura = (
            "🟢 **Alta compressão do erro provável**.\n\n"
            "O alvo está mais bem definido do que o normal.\n"
            "Se houver PRÉ-ECO / ECO, a convicção operacional aumenta."
        )
    elif positivos == 3:
        leitura = (
            "🟡 **Compressão parcial**.\n\n"
            "Há foco emergente, mas ainda com dispersão residual."
        )
    else:
        leitura = (
            "🔴 **Sem compressão clara**.\n\n"
            "Erro ainda espalhado. Operar com cautela."
        )

    exibir_bloco_mensagem(
        "Compressão do Alvo — Diagnóstico",
        leitura,
        tipo="info",
    )

    st.caption(
        f"Sinais positivos: {positivos}/5 | "
        "Este painel **não autoriza nem bloqueia** nenhuma ação."
    )

# ============================================================
# FIM — PAINEL V16 — COMPRESSÃO DO ALVO (OBSERVACIONAL)
# ============================================================


# ============================================================
# BLOCO — OBSERVADOR HISTÓRICO DE EVENTOS k (V16)
# REPLAY HISTÓRICO OBSERVACIONAL (MEMÓRIA REAL)
# NÃO decide | NÃO prevê | NÃO altera motores | NÃO altera volumes
# ============================================================


# ============================================================
# ALIAS CANÔNICO (V16) — M3
# Expectativa Histórica — Contexto do Momento
# ============================================================
def v16_painel_expectativa_historica_contexto():
    """Alias canônico para preservar âncoras/painéis que chamam a função V16.

    Regra: NÃO calcula listas, NÃO decide, NÃO altera fluxo.
    Encaminha para o painel observacional M3.
    """
    return m3_painel_expectativa_historica_contexto()


def v16_replay_historico_observacional(
    *,
    df,
    matriz_norm,
    janela_max=800,
):
    """
    Replay histórico OBSERVACIONAL.
    Executa leitura silenciosa série-a-série para preencher memória
    e eliminar campos None no Observador Histórico.

    - Usa somente dados já calculados
    - NÃO reexecuta motores pesados
    - NÃO interfere no fluxo operacional
    """

    if df is None or matriz_norm is None:
        return []

    n_total = len(df)
    inicio = max(0, n_total - int(janela_max))

    registros = []

    col_pass = [c for c in df.columns if c.startswith("p")]

    for idx in range(inicio, n_total):

        # --- NR% local (réplica leve) ---
        try:
            m = matriz_norm[: idx + 1]
            variancias = np.var(m, axis=1)
            ruido_A = float(np.mean(variancias))
            saltos = [
                np.linalg.norm(m[i] - m[i - 1])
                for i in range(1, len(m))
            ]
            ruido_B = float(np.mean(saltos)) if saltos else 0.0
            nr_pct = float(
                (0.55 * min(1.0, ruido_A / 0.08) +
                 0.45 * min(1.0, ruido_B / 1.20)) * 100.0
            )
        except Exception:
            nr_pct = None

        # --- Divergência local S6 vs MC (proxy leve) ---
        try:
            base = m[-1]
            candidatos = m[-10:] if len(m) >= 10 else m
            divergencia = float(
                np.linalg.norm(np.mean(candidatos, axis=0) - base)
            )
        except Exception:
            divergencia = None

        # --- Velocidade / estado do alvo (heurística coerente) ---
        try:
            vel = float(
                (nr_pct / 100.0 if nr_pct is not None else 0.5) +
                (divergencia / 15.0 if divergencia is not None else 0.5)
            ) / 2.0
        except Exception:
            vel = None

        if vel is None:
            estado = None
        elif vel < 0.30:
            estado = "parado"
        elif vel < 0.55:
            estado = "movimento_lento"
        elif vel < 0.80:
            estado = "movimento_rapido"
        else:
            estado = "disparado"

        # --- k histórico ---
        try:
            k_val = int(df.iloc[idx].get("k", 0))
        except Exception:
            k_val = 0

        registros.append({
            "serie_id": idx,
            "k_valor": k_val,
            "estado_alvo": estado,
            "nr_percent": nr_pct,
            "div_s6_mc": divergencia,
        })

    return registros


# ============================================================
# EXECUÇÃO AUTOMÁTICA — REPLAY OBSERVACIONAL (SE HISTÓRICO EXISTIR)
# ============================================================

if (
    "historico_df" in st.session_state
    and "pipeline_matriz_norm" in st.session_state
):
    registros_obs = v16_replay_historico_observacional(
        df=st.session_state.get("historico_df"),
        matriz_norm=st.session_state.get("pipeline_matriz_norm"),
        janela_max=800,  # DECISÃO DO COMANDO
    )

    st.session_state["observador_historico_v16"] = registros_obs

# ============================================================
# FIM — BLOCO OBSERVADOR HISTÓRICO (V16)
# ============================================================



# ============================================================
# BLOCO — OBSERVAÇÃO HISTÓRICA OFFLINE (V16)
# OPÇÃO B MÍNIMA | LEITURA PURA | NÃO DECIDE | NÃO OPERA
# ============================================================

def _pc_distancia_carros_offline(a, b):
    """
    Distância simples entre dois carros (listas de 6):
    quantos passageiros mudaram (0..6).
    Observacional, robusto e defensivo.
    """
    try:
        sa = set(int(x) for x in a)
        sb = set(int(x) for x in b)
        inter = len(sa & sb)
        return max(0, 6 - inter)
    except Exception:
        return None


def _pc_estado_alvo_proxy_offline(dist):
    """
    Mapeia distância (0..6) em estado do alvo (proxy observacional).
    NÃO é o estado V16 online. Uso EXCLUSIVO histórico.
    """
    if dist is None:
        return None
    if dist <= 1:
        return "parado"
    if dist <= 3:
        return "movimento_lento"
    if dist <= 5:
        return "movimento"
    return "movimento_brusco"


def _pc_extrair_carro_offline(row):
    """
    Extrai os 6 passageiros de uma linha do histórico.
    Compatível com p1..p6 ou colunas numéricas genéricas.
    """
    cols_p = ["p1", "p2", "p3", "p4", "p5", "p6"]
    if all(c in row.index for c in cols_p):
        return [row[c] for c in cols_p]

    candidatos = []
    for c in row.index:
        if str(c).lower() == "k":
            continue
        try:
            candidatos.append(int(row[c]))
        except Exception:
            continue

    return candidatos[:6] if len(candidatos) >= 6 else None


def construir_contexto_historico_offline_v16(df):
    """
    Constrói CONTEXTO HISTÓRICO OFFLINE mínimo:
    - estado_alvo_proxy_historico
    - delta_k_historico
    - eventos_k_historico (enriquecido)
    NÃO interfere em motores, painéis ou decisões.
    """

    if df is None or df.empty:
        return

    estado_proxy_hist = {}
    delta_k_hist = {}
    eventos_k = []

    carro_prev = None
    ultima_pos_k = None

    for pos, (idx, row) in enumerate(df.iterrows()):
        carro_atual = _pc_extrair_carro_offline(row)

        dist = (
            _pc_distancia_carros_offline(carro_prev, carro_atual)
            if carro_prev is not None and carro_atual is not None
            else None
        )

        estado_proxy = _pc_estado_alvo_proxy_offline(dist)
        estado_proxy_hist[idx] = estado_proxy

        # Evento k (observacional)
        try:
            k_val = int(row.get("k", 0))
        except Exception:
            k_val = 0

        if k_val > 0:
            delta = None if ultima_pos_k is None else int(pos - ultima_pos_k)
            delta_k_hist[idx] = delta

            eventos_k.append({
                "serie_id": idx,
                "pos": int(pos),
                "k_valor": int(k_val),
                "delta_series": delta,
                "estado_alvo_proxy": estado_proxy,
            })

            ultima_pos_k = pos

        carro_prev = carro_atual

    # Persistência PASSIVA (session_state)
    st.session_state["estado_alvo_proxy_historico"] = estado_proxy_hist
    st.session_state["delta_k_historico"] = delta_k_hist
    st.session_state["eventos_k_historico"] = eventos_k


# ============================================================
# EXECUÇÃO AUTOMÁTICA OFFLINE (SE HISTÓRICO EXISTIR)
# NÃO BLOQUEIA | NÃO DECIDE | NÃO OPERA
# ============================================================

if "historico_df" in st.session_state:
    try:
        construir_contexto_historico_offline_v16(
            st.session_state.get("historico_df")
        )
    except Exception:
        pass

# ============================================================
# FIM — OBSERVAÇÃO HISTÓRICA OFFLINE (V16) — OPÇÃO B MÍNIMA
# ============================================================

def extrair_eventos_k_historico(
    df,
    estados_alvo=None,
    k_star_series=None,
    nr_percent_series=None,
    divergencia_series=None,
    pre_eco_series=None,
    eco_series=None,
):
    """
    Extrai eventos k do histórico com contexto.
    NÃO decide, NÃO filtra operacionalmente, NÃO altera motores.
    Retorna lista de dicionários observacionais.
    """

    if df is None or df.empty:
        return []

    eventos = []
    ultima_serie_k = None

    for idx, row in df.iterrows():
        # Espera-se que o histórico tenha coluna 'k'
        k_valor = row.get("k", 0)

        if k_valor and k_valor > 0:
            # Delta desde último k
            if ultima_serie_k is None:
                delta = None
            else:
                delta = idx - ultima_serie_k

            evento = {
                "serie_id": idx,
                "k_valor": int(k_valor),
                "delta_series": delta,
                "estado_alvo": (
                    estados_alvo.get(idx)
                    if isinstance(estados_alvo, dict)
                    else None
                ),
                "k_star": (
                    k_star_series.get(idx)
                    if isinstance(k_star_series, dict)
                    else None
                ),
                "nr_percent": (
                    nr_percent_series.get(idx)
                    if isinstance(nr_percent_series, dict)
                    else None
                ),
                "div_s6_mc": (
                    divergencia_series.get(idx)
                    if isinstance(divergencia_series, dict)
                    else None
                ),
                "pre_eco": (
                    pre_eco_series.get(idx)
                    if isinstance(pre_eco_series, dict)
                    else False
                ),
                "eco": (
                    eco_series.get(idx)
                    if isinstance(eco_series, dict)
                    else False
                ),
            }

            eventos.append(evento)
            ultima_serie_k = idx

    return eventos


# ============================================================
# EXECUÇÃO AUTOMÁTICA (APENAS SE HISTÓRICO EXISTIR)
# ============================================================

if "historico_df" in st.session_state:
    df_hist = st.session_state.get("historico_df")

    eventos_k = extrair_eventos_k_historico(
        df=df_hist,
        estados_alvo=st.session_state.get("estado_alvo_historico"),
        k_star_series=st.session_state.get("kstar_historico"),
        nr_percent_series=st.session_state.get("nr_historico"),
        divergencia_series=st.session_state.get("div_s6_mc_historico"),
        pre_eco_series=st.session_state.get("pre_eco_historico"),
        eco_series=st.session_state.get("eco_historico"),
    )

    st.session_state["eventos_k_historico"] = eventos_k

# ============================================================
# BLOCO — FIM OBSERVADOR HISTÓRICO DE EVENTOS k
# ============================================================

# ============================================================
# Painel — 📊 Observador Histórico de Eventos k (V16)
# FASE 1 — OBSERVAÇÃO PURA | NÃO DECIDE | NÃO OPERA
# ============================================================

if painel == "📊 Observador k — Histórico":

    st.markdown("## 📊 Observador Histórico de Eventos k")
    st.caption(
        "Leitura puramente observacional. "
        "Este painel **não influencia** previsões, volumes ou decisões."
    )

    eventos = st.session_state.get("eventos_k_historico")

    if not eventos:
        exibir_bloco_mensagem(
            "Nenhum evento k disponível",
            "Carregue um histórico válido para observar eventos k.",
            tipo="info",
        )
        st.stop()

    df_k = pd.DataFrame(eventos)

    st.markdown("### 🔍 Tabela de Eventos k (Histórico)")
    st.dataframe(
        df_k,
        use_container_width=True,
        height=420,
    )

    # Métricas simples (somente leitura)
    st.markdown("### 📈 Métricas Observacionais Básicas")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total de eventos k",
            len(df_k),
        )

    with col2:
        delta_vals = df_k["delta_series"].dropna()
        st.metric(
            "Δ médio entre ks",
            round(delta_vals.mean(), 2) if not delta_vals.empty else "—",
        )

    with col3:
        st.metric(
            "Δ mínimo observado",
            int(delta_vals.min()) if not delta_vals.empty else "—",
        )

    st.info(
        "Interpretação é humana. "
        "Nenhum uso operacional é feito a partir destes dados."
    )

# ============================================================
# FIM — Painel Observador Histórico de Eventos k
# ============================================================

# ============================================================
# Painel — 🎯 Compressão do Alvo — Observacional (V16)
# LEITURA PURA | NÃO DECIDE | NÃO ALTERA MOTORES
# Objetivo: medir se o alvo está REALMENTE "na mira"
# ============================================================

if painel == "🎯 Compressão do Alvo — Observacional (V16)":

    st.markdown("## 🎯 Compressão do Alvo — Observacional (V16)")
    st.caption(
        "Painel **observacional puro**.\n\n"
        "Ele NÃO gera previsões, NÃO altera volumes e NÃO interfere no fluxo.\n"
        "Serve para responder: **o alvo está realmente comprimido / na mira?**"
    )

    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")

    if df is None or matriz_norm is None:
        exibir_bloco_mensagem(
            "Pipeline incompleto",
            "Execute **Carregar Histórico** e **Pipeline V14-FLEX ULTRA** antes.",
            tipo="warning",
        )
        st.stop()

    # ------------------------------------------------------------
    # Parâmetros fixos (observacionais)
    # ------------------------------------------------------------
    JANELA_ANALISE = 120   # últimas séries
    JANELA_LOCAL = 8       # microjanela para dispersão
    LIMIAR_COMPRESSAO = 0.65  # heurístico (não decisório)

    n = len(matriz_norm)
    if n < JANELA_ANALISE + JANELA_LOCAL:
        exibir_bloco_mensagem(
            "Histórico insuficiente",
            "São necessárias mais séries para analisar compressão do alvo.",
            tipo="warning",
        )
        st.stop()

    # ------------------------------------------------------------
    # Cálculo da compressão
    # ------------------------------------------------------------
    dispersoes = []
    centroides = []

    for i in range(n - JANELA_ANALISE, n):
        janela = matriz_norm[max(0, i - JANELA_LOCAL): i + 1]
        centro = np.mean(janela, axis=0)
        centroides.append(centro)

        dist = np.mean(
            [np.linalg.norm(linha - centro) for linha in janela]
        )
        dispersoes.append(dist)

    dispersao_media = float(np.mean(dispersoes))
    dispersao_std = float(np.std(dispersoes))

    # Compressão relativa (quanto menor a dispersão, maior a compressão)
    compressao_score = 1.0 - min(1.0, dispersao_media / (dispersao_media + dispersao_std + 1e-6))
    compressao_score = float(round(compressao_score, 4))

    # ------------------------------------------------------------
    # Interpretação QUALITATIVA (não decisória)
    # ------------------------------------------------------------
    if compressao_score >= 0.75:
        leitura = "🟢 Alvo fortemente comprimido"
        comentario = (
            "O histórico recente mostra **alta repetição estrutural**.\n"
            "O sistema está operando em zona de foco.\n"
            "Quando combinado com PRÉ-ECO / ECO, **permite acelerar**."
        )
    elif compressao_score >= LIMIAR_COMPRESSAO:
        leitura = "🟡 Compressão moderada"
        comentario = (
            "Existe coerência estrutural, mas ainda com respiração.\n"
            "Bom para operação equilibrada."
        )
    else:
        leitura = "🔴 Alvo disperso"
        comentario = (
            "Alta variabilidade estrutural.\n"
            "Mesmo que k apareça, **não indica alvo na mira**."
        )

    # ------------------------------------------------------------
    # Exibição
    # ------------------------------------------------------------
    st.markdown("### 📐 Métrica de Compressão do Alvo")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Score de Compressão", compressao_score)

    with col2:
        st.metric("Dispersão média", round(dispersao_media, 4))

    with col3:
        st.metric("Volatilidade da dispersão", round(dispersao_std, 4))

    exibir_bloco_mensagem(
        "Leitura Observacional",
        f"**{leitura}**\n\n{comentario}",
        tipo="info",
    )

    st.info("📌 Interpretação correta (sem viés):\n- Isso NÃO prevê o próximo alvo.\n- Isso mede *o que costuma acontecer* quando o ambiente cai no mesmo tipo de regime.\n- Serve para calibrar expectativa, postura e paciência — não para aumentar convicção por '3 acertos'.")

# ============================================================
# FIM — Painel 🎯 Compressão do Alvo — Observacional (V16)
# ============================================================



# ============================================================
# Observação Histórica — Eventos k (V16)
# Leitura passiva do histórico. Não interfere em decisões.
# + CRUZAMENTO k × ESTADO DO ALVO (PROXY)
# ============================================================

def _pc_distancia_carros(a, b):
    """
    Distância simples entre dois carros (listas de 6):
    quantos passageiros mudaram (0..6).
    """
    try:
        sa = set([int(x) for x in a])
        sb = set([int(x) for x in b])
        inter = len(sa & sb)
        return max(0, 6 - inter)
    except Exception:
        return None


def _pc_estado_alvo_proxy(dist):
    """
    Mapeia distância (0..6) em estado do alvo (proxy observacional).
    """
    if dist is None:
        return None
    if dist <= 1:
        return "parado"
    if dist <= 3:
        return "movimento_lento"
    if dist <= 5:
        return "movimento"
    return "movimento_brusco"


def _pc_extrair_carro_row(row):
    """
    Extrai os 6 passageiros da linha do df.
    Tentativa 1: colunas numéricas (6 colunas)
    Tentativa 2: colunas p1..p6 (se existir)
    """
    # Caso já tenha colunas p1..p6
    cols_p = ["p1", "p2", "p3", "p4", "p5", "p6"]
    if all(c in row.index for c in cols_p):
        return [row[c] for c in cols_p]

    # Caso seja DF com colunas misturadas: pega primeiros 6 inteiros que não sejam 'k'
    candidatos = []
    for c in row.index:
        if str(c).lower() == "k":
            continue
        try:
            v = int(row[c])
            candidatos.append(v)
        except Exception:
            continue

    if len(candidatos) >= 6:
        return candidatos[:6]

    return None


def extrair_eventos_k_historico_com_proxy(df):
    """
    Eventos k + delta + estado do alvo (proxy) calculado do próprio histórico.
    NÃO depende de estado_alvo_historico/kstar_historico/etc.
    """
    if df is None or df.empty:
        return [], {}

    eventos = []
    ultima_pos_k = None

    # Para estatística
    cont_estados = {"parado": 0, "movimento_lento": 0, "movimento": 0, "movimento_brusco": 0, "None": 0}

    # Vamos usar posição sequencial (0..n-1) para delta
    rows = list(df.iterrows())

    carro_prev = None

    for pos, (idx, row) in enumerate(rows):
        k_val = row.get("k", 0)
        carro_atual = _pc_extrair_carro_row(row)

        dist = _pc_distancia_carros(carro_prev, carro_atual) if (carro_prev is not None and carro_atual is not None) else None
        estado = _pc_estado_alvo_proxy(dist)

        # Contagem estados (para todas as séries, não só eventos k)
        if estado is None:
            cont_estados["None"] += 1
        else:
            cont_estados[estado] += 1

        # Evento k
        try:
            k_int = int(k_val) if k_val is not None else 0
        except Exception:
            k_int = 0

        if k_int > 0:
            delta = None if ultima_pos_k is None else int(pos - ultima_pos_k)

            eventos.append({
                "serie_id": idx,
                "pos": int(pos),
                "k_valor": int(k_int),
                "delta_series": delta,
                "distancia_prev": dist,
                "estado_alvo_proxy": estado,
            })

            ultima_pos_k = pos

        carro_prev = carro_atual

    return eventos, cont_estados


# ============================================================
# PAINEL (VISUALIZAÇÃO)
# ============================================================

if painel == "Observação Histórica — Eventos k":

    st.markdown("## Observação Histórica — Eventos k")
    st.caption("Leitura passiva do histórico. Não interfere em decisões.")

    df_hist = st.session_state.get("historico_df")

    if df_hist is None or df_hist.empty:
        exibir_bloco_mensagem(
            "Histórico ausente",
            "Carregue o histórico primeiro (Painel 1 / 1B).",
            tipo="warning",
        )
        st.stop()

    eventos_k, cont_estados = extrair_eventos_k_historico_com_proxy(df_hist)
    st.session_state["eventos_k_historico"] = eventos_k

    # ===========================
    # Resumo estatístico
    # ===========================
    total_eventos = len(eventos_k)

    deltas = [e["delta_series"] for e in eventos_k if isinstance(e.get("delta_series"), int)]
    delta_medio = round(sum(deltas) / max(1, len(deltas)), 2) if deltas else None
    max_k = max([e.get("k_valor", 0) for e in eventos_k], default=0)

    st.markdown("### Resumo Estatístico Simples")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total de eventos k", f"{total_eventos}")
    c2.metric("Δ médio entre ks", f"{delta_medio}" if delta_medio is not None else "—")
    c3.metric("Máx k observado", f"{max_k}")

    st.markdown("### Distribuição do Estado do Alvo (PROXY no histórico)")
    total_series = sum(cont_estados.values()) if isinstance(cont_estados, dict) else 0
    if total_series > 0:
        corpo = (
            f"- parado: **{cont_estados.get('parado', 0)}**\n"
            f"- movimento_lento: **{cont_estados.get('movimento_lento', 0)}**\n"
            f"- movimento: **{cont_estados.get('movimento', 0)}**\n"
            f"- movimento_brusco: **{cont_estados.get('movimento_brusco', 0)}**\n"
        )
        exibir_bloco_mensagem("Estado do alvo (proxy)", corpo, tipo="info")
    else:
        st.info("Não foi possível calcular distribuição de estado (proxy).")

    # ===========================
    # Tabela de eventos k
    # ===========================
    st.markdown("### 📋 Tabela de Eventos k (com estado proxy)")
    if total_eventos == 0:
        st.info("Nenhum evento k encontrado no histórico.")
        st.stop()

    mostrar = st.slider(
        "Quantos eventos k mostrar (mais recentes)?",
        min_value=20,
        max_value=min(300, total_eventos),
        value=min(80, total_eventos),
        step=10,
    )

    # Mostra os mais recentes
    df_evt = pd.DataFrame(eventos_k[-mostrar:])
    st.dataframe(df_evt, use_container_width=True)

    st.caption("Obs.: estado_alvo_proxy é calculado por mudança entre carros consecutivos (distância 0..6).")
    st.caption("k*/NR%/div/PRÉ-ECO/ECO ainda não estão historificados por série — isso é a próxima evolução (opcional).")

# ============================================================
# FIM — Observação Histórica — Eventos k (V16)
# ============================================================

        

# ============================================================
# Painel 2 — 🛰️ Sentinelas — k* (Ambiente de Risco)
# ============================================================

if painel == "🛰️ Sentinelas — k* (Ambiente de Risco)":

    st.markdown("## 🛰️ Sentinelas — k* (Ambiente de Risco) — V15.7 MAX")

    df = st.session_state.get("historico_df")

    if df is None:
        exibir_bloco_mensagem(
            "Histórico não carregado",
            "Vá primeiro ao painel **📁 Carregar Histórico**.",
            tipo="warning",
        )
    else:
        qtd_series = len(df)

        # Parâmetros do k*
        janela_curta = 12
        janela_media = 30
        janela_longa = 60

        # Anti-zumbi aplicado antes de cálculos longos
        if not limitar_operacao(
            qtd_series,
            limite_series=LIMITE_SERIES_REPLAY_ULTRA,
            contexto="Sentinela k*",
            painel="🛰️ Sentinelas — k*",
        ):
            st.stop()

        # -------------------------------------------
        # Cálculo do k* — versão V15.7 MAX / V16 Premium
        # -------------------------------------------
        try:
            k_vals = df["k"].astype(int).values

            def media_movel(vetor, janela):
                if len(vetor) < janela:
                    return np.mean(vetor)
                return np.mean(vetor[-janela:])

            k_curto = media_movel(k_vals, janela_curta)
            k_medio = media_movel(k_vals, janela_media)
            k_longo = media_movel(k_vals, janela_longa)

            # Fórmula nova do k* — ponderada
            k_star = (
                0.50 * k_curto
                + 0.35 * k_medio
                + 0.15 * k_longo
            )

        except Exception as erro:
            exibir_bloco_mensagem(
                "Erro no cálculo do k*",
                f"Ocorreu um erro interno: {erro}",
                tipo="error",
            )
            st.stop()

        # Guarda na sessão
        st.session_state["sentinela_kstar"] = k_star

        # Exibição amigável
        st.markdown(f"### 🌡️ k* calculado: **{k_star:.4f}**")

        # Diagnóstico de regime
        if k_star < 0.15:
            regime = "🟢 Ambiente Estável (Regime de Padrão)"
        elif k_star < 0.30:
            regime = "🟡 Pré-Ruptura (Atenção)"
        else:
            regime = "🔴 Ambiente de Ruptura (Alta Turbulência)"

        exibir_bloco_mensagem(
            "Diagnóstico do Ambiente",
            f"O regime identificado para o histórico atual é:\n\n{regime}",
            tipo="info",
        )

# ============================================================
# Painel X — 📊 Observação Histórica — Eventos k (V16)
# ============================================================

if painel == "📊 Observação Histórica — Eventos k":

    st.markdown("## 📊 Observação Histórica — Eventos k")
    st.caption("Leitura passiva do histórico. Não interfere em decisões.")

    eventos = st.session_state.get("eventos_k_historico", [])

    if not eventos:
        st.info("Nenhum evento k encontrado no histórico carregado.")
        st.stop()

    df_eventos = pd.DataFrame(eventos)

    st.markdown("### 📋 Tabela de Eventos k")
    st.dataframe(df_eventos, use_container_width=True)

    # Resumo rápido
    st.markdown("### 📈 Resumo Estatístico Simples")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total de eventos k", len(df_eventos))

    with col2:
        media_delta = (
            df_eventos["delta_series"].dropna().mean()
            if "delta_series" in df_eventos
            else None
        )
        st.metric(
            "Δ médio entre ks",
            f"{media_delta:.2f}" if media_delta else "—",
        )

    with col3:
        st.metric(
            "Máx k observado",
            df_eventos["k_valor"].max() if "k_valor" in df_eventos else "—",
        )

# ============================================================
# FIM — Painel X — Observação Histórica — Eventos k
# ============================================================


# ============================================================
# Painel 3 — 🛣️ Pipeline V14-FLEX ULTRA (Preparação)
# ============================================================
if painel == "🛣️ Pipeline V14-FLEX ULTRA":

    st.markdown("## 🛣️ Pipeline V14-FLEX ULTRA — V15.7 MAX")

    df = st.session_state.get("historico_df")

    if df is None:
        exibir_bloco_mensagem(
            "Histórico não carregado",
            "Vá ao painel **📁 Carregar Histórico** antes de continuar.",
            tipo="warning",
        )
        st.stop()

    qtd_series = len(df)

    # Proteção anti-zumbi do pipeline — mais duro que o k*
    if not limitar_operacao(
        qtd_series,
        limite_series=LIMITE_SERIES_REPLAY_ULTRA,
        contexto="Pipeline V14-FLEX ULTRA",
        painel="🛣️ Pipeline",
    ):
        st.stop()

    st.info("Iniciando processamento do Pipeline FLEX ULTRA...")

    col_pass = [c for c in df.columns if c.startswith("p")]
    matriz = df[col_pass].astype(float).values

    # ============================================================
    # Normalização
    # ============================================================
    try:
        minimo = matriz.min()
        maximo = matriz.max()
        amplitude = maximo - minimo if maximo != minimo else 1.0

        matriz_norm = (matriz - minimo) / amplitude

    except Exception as erro:
        exibir_bloco_mensagem(
            "Erro na normalização",
            f"Detalhes técnicos: {erro}",
            tipo="error",
        )
        st.stop()

    # ============================================================
    # Estatísticas da estrada (FLEX ULTRA)
    # ============================================================
    medias = np.mean(matriz_norm, axis=1)
    desvios = np.std(matriz_norm, axis=1)

    media_geral = float(np.mean(medias))
    desvio_geral = float(np.mean(desvios))

    # Classificação simples de regime da estrada
    if media_geral < 0.35:
        estrada = "🟦 Estrada Fria (Baixa energia)"
    elif media_geral < 0.65:
        estrada = "🟩 Estrada Neutra / Estável"
    else:
        estrada = "🟥 Estrada Quente (Alta volatilidade)"

    # ============================================================
    # Clusterização leve (DX — motor original FLEX ULTRA)
    # ============================================================
    try:
        from sklearn.cluster import KMeans

        n_clusters = 3
        modelo = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
        clusters = modelo.fit_predict(matriz_norm)

        centroides = modelo.cluster_centers_

    except Exception:
        clusters = np.zeros(len(matriz_norm))
        centroides = np.zeros((1, matriz_norm.shape[1]))

# ============================================================
# PARTE 2/8 — FIM
# ============================================================
# ============================================================
# PARTE 3/8 — INÍCIO
# ============================================================

    # ============================================================
    # Exibição final do pipeline
    # ============================================================
    st.markdown("### 📌 Diagnóstico do Pipeline FLEX ULTRA")

    corpo = (
        f"- Séries carregadas: **{qtd_series}**\n"
        f"- Passageiros por carro (n): **{len(col_pass)}**\n"
        f"- Energia média da estrada: **{_pc_fmt_num(media_geral, decimals=4)}**\n"
        f"- Volatilidade média: **{_pc_fmt_num(desvio_geral, decimals=4)}**\n"
        f"- Regime detectado: {estrada}\n"
        f"- Clusters formados: **{int(max(clusters)+1)}**"
    )

    exibir_bloco_mensagem(
        "Resumo do Pipeline FLEX ULTRA",
        corpo,
        tipo="info",
    )

    # ============================================================
    # Salvando na sessão para módulos seguintes (CANÔNICO)
    # ============================================================
    st.session_state["pipeline_col_pass"] = col_pass
    st.session_state["pipeline_clusters"] = clusters
    st.session_state["pipeline_centroides"] = centroides
    st.session_state["pipeline_matriz_norm"] = matriz_norm
    st.session_state["pipeline_estrada"] = estrada

    # Sinais observáveis (para governança / leitura — não altera motor)
    st.session_state["regime_identificado"] = estrada
    st.session_state["energia_media"] = float(media_geral)
    st.session_state["energia_media_estrada"] = float(media_geral)
    st.session_state["volatilidade_media"] = float(desvio_geral)
    st.session_state["clusters_formados"] = int(max(clusters) + 1) if len(np.atleast_1d(clusters)) else 0

    # ============================================================
    # M1 — SELO CANÔNICO (Governança/Mirror)
    # S2: Pipeline Consolidado (observável, mínimo, sem tocar no núcleo)
    # ============================================================
    # Regra: só carimba aqui, no caminho feliz (pós-cálculo, pré-success).
    # O Mirror lê este selo; nenhum motor depende dele.
    st.session_state["pipeline_flex_ultra_concluido"] = True
    st.session_state["pipeline_executado"] = True
    st.session_state["m1_selo_pipeline_ok"] = True
    try:
        from datetime import datetime
        st.session_state["m1_ts_pipeline_ok"] = datetime.now().isoformat(timespec="seconds")
    except Exception:
        # Falha silenciosa (observacional)
        pass

    st.success("Pipeline FLEX ULTRA concluído com sucesso!")

# ============================================================
# PARTE 3/8 — FIM
# ============================================================


# ============================================================
# Painel 4 — 🔁 Replay LIGHT
# ============================================================
if painel == "🔁 Replay LIGHT":

    st.markdown("## 🔁 Replay LIGHT — V15.7 MAX")

    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")

    if df is None or matriz_norm is None:
        exibir_bloco_mensagem(
            "Pipeline não encontrado",
            "Execute primeiro o painel **🛣️ Pipeline V14-FLEX ULTRA**.",
            tipo="warning",
        )
        st.stop()

    qtd_series = len(df)

    # Anti-zumbi para replays leves
    if not limitar_operacao(
        qtd_series,
        limite_series=LIMITE_SERIES_REPLAY_ULTRA,
        contexto="Replay LIGHT",
        painel="🔁 Replay LIGHT",
    ):
        st.stop()

    st.info("Executando Replay LIGHT...")

    try:
        # DX leve = simples proximidade média entre séries vizinhas
        proximidades = []
        for i in range(1, len(matriz_norm)):
            dist = np.linalg.norm(matriz_norm[i] - matriz_norm[i - 1])
            proximidades.append(dist)

        media_proximidade = float(np.mean(proximidades))
        desvio_proximidade = float(np.std(proximidades))

    except Exception as erro:
        exibir_bloco_mensagem(
            "Erro no Replay LIGHT",
            f"Detalhes técnicos: {erro}",
            tipo="error",
        )
        st.stop()

    corpo = (
        f"- Séries analisadas: **{qtd_series}**\n"
        f"- Proximidade média (DX Light): **{media_proximidade:.4f}**\n"
        f"- Desvio de proximidade: **{desvio_proximidade:.4f}**\n"
        "\nValores mais altos indicam maior irregularidade."
    )

    exibir_bloco_mensagem(
        "Resumo do Replay LIGHT",
        corpo,
        tipo="info",
    )

    st.success("Replay LIGHT concluído!")

# ============================================================
# Painel 5 — 🔁 Replay ULTRA
# ============================================================
if painel == "🔁 Replay ULTRA":

    st.markdown("## 🔁 Replay ULTRA — V15.7 MAX")

    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")

    if df is None or matriz_norm is None:
        exibir_bloco_mensagem(
            "Pipeline não encontrado",
            "Execute primeiro o painel **🛣️ Pipeline V14-FLEX ULTRA**.",
            tipo="warning",
        )
        st.stop()

    qtd_series = len(df)

    # Proteção anti-zumbi — Replay ULTRA é mais pesado
    if not limitar_operacao(
        qtd_series,
        limite_series=LIMITE_SERIES_REPLAY_ULTRA,
        contexto="Replay ULTRA completo",
        painel="🔁 Replay ULTRA",
    ):
        st.stop()

    st.info("Executando Replay ULTRA...")

    try:
        # DX Ultra = distância média entre cada série e o centróide global
        centr_global = np.mean(matriz_norm, axis=0)
        distancias = [
            np.linalg.norm(linha - centr_global) for linha in matriz_norm
        ]

        media_dx = float(np.mean(distancias))
        desvio_dx = float(np.std(distancias))

    except Exception as erro:
        exibir_bloco_mensagem(
            "Erro no Replay ULTRA",
            f"Detalhes técnicos: {erro}",
            tipo="error",
        )
        st.stop()

    corpo = (
        f"- Séries analisadas: **{qtd_series}**\n"
        f"- Distância média ao centróide (DX Ultra): **{media_dx:.4f}**\n"
        f"- Dispersão DX Ultra: **{desvio_dx:.4f}**\n"
        "\nValores maiores indicam estrada mais caótica."
    )

    exibir_bloco_mensagem(
        "Resumo do Replay ULTRA",
        corpo,
        tipo="info",
    )

    st.success("Replay ULTRA concluído!")


# ============================================================
# Painel X — 🧭 Replay Progressivo — Janela Móvel (Assistido)
# OBJETIVO:
# - Automatizar a "janela móvel" do histórico SEM você ter que ficar
#   editando arquivo/recortando histórico na mão.
# - NÃO roda motores sozinho. NÃO decide. NÃO altera Camada 4.
# - Ele apenas:
#     (1) Guarda o histórico completo uma única vez,
#     (2) Aplica um recorte (C1..Ck) como histórico ATIVO,
#     (3) Limpa chaves dependentes para forçar recalcular,
#     (4) Permite registrar o pacote gerado em cada janela,
#     (5) Avalia automaticamente o pacote registrado contra os 2 alvos seguintes.
#
# USO CANÔNICO (1 ciclo):
# 1) Carregue o histórico COMPLETO (ex.: até C5832).
# 2) Vá neste painel e aplique a janela (ex.: C5826).
# 3) Rode: Sentinelas -> Pipeline -> (TURBO opcional) -> Modo 6.
# 4) Volte aqui e "Registrar Pacote da Janela".
# 5) Veja avaliação automática em (C5827,C5828).
# 6) Repita com C5828, C5830, C5832...
# ============================================================
if painel == "🧭 Replay Progressivo — Janela Móvel (Assistido)":

    st.markdown("## 🧭 Replay Progressivo — Janela Móvel (Assistido)")
    st.caption(
        "Este painel **não gera listas sozinho** e **não muda Camada 4**. "
        "Ele só automatiza o recorte do histórico (janela móvel) e organiza o replay progressivo."
    )

    df_full = st.session_state.get("historico_df_full")
    df_atual = st.session_state.get("historico_df")

    if df_atual is None or df_atual.empty:
        st.warning("Histórico não encontrado. Use primeiro **📁 Carregar Histórico**.")
        st.stop()

    # 1) Guardar histórico completo uma única vez
    if df_full is None:
        st.session_state["historico_df_full"] = df_atual.copy()
        df_full = st.session_state.get("historico_df_full")
        st.info(
            f"📦 Histórico completo foi guardado em memória (histórico_full). "
            f"Séries disponíveis: **{len(df_full)}**"
        )

    # 2) Contagem e limites
    total_series = int(len(df_full))
    if total_series < 10:
        st.warning("Histórico muito curto para replay progressivo.")
        st.stop()

    # 3) Estado atual da janela
    janela_k = int(st.session_state.get("replay_janela_k_active", len(df_atual)))
    janela_k = max(1, min(janela_k, total_series))

    colA, colB, colC = st.columns([1, 1, 1])
    colA.metric("Séries no FULL", str(total_series))
    colB.metric("Séries no ATIVO", str(len(df_atual)))
    colC.metric("Janela ativa (k)", str(janela_k))

    st.markdown("---")

    # 4) Seleção de janela (k)
    st.markdown("### 1) Selecionar e aplicar janela móvel (C1..Ck)")
    st.caption(
        "Ao aplicar a janela, o painel **recorta o histórico ativo** e "
        "**limpa chaves dependentes** (pipeline/sentinelas/pacotes) para você recalcular com segurança."
    )

    # OBS: slider foi trocado por entrada numérica para evitar resets e facilitar uso.
    k_novo = st.number_input(
        "Informe k (última série INCLUÍDA no histórico ativo)",
        min_value=10,
        max_value=total_series,
        value=int(st.session_state.get("replay_janela_k_input", janela_k)),
        step=1,
        key="replay_janela_k_input",
        help="Dica: use valores como 5826, 5828, 5830... (para simular o replay progressivo).",
    )

    # 5) Função local: limpeza de chaves dependentes (conservadora)
    def _replay_limpar_chaves_dependentes():
        # Chaves típicas que dependem do histórico/pipeline/pacote
        chaves = [
            "pipeline_col_pass",
            "pipeline_clusters",
            "pipeline_centroides",
            "pipeline_matriz_norm",
            "pipeline_estrada",
            "pipeline_flex_ultra_concluido",
            "pipeline_executado",
            "m1_selo_pipeline_ok",
            "m1_ts_pipeline_ok",
            "sentinela_kstar",
            "k_star",
            "k*",
            "nr_percent",
            "nr_percent_v16",
            "div_s6_mc",
            "divergencia_s6_mc",
            "indice_risco",
            "classe_risco",
            "ultima_previsao",
            "pacote_listas_atual",
            "listas_geradas",
            "listas_finais",
            "modo6_executado",
            "modo_6_executado",
            "modo_6_ativo",
            "turbo_executado",
            "turbo_ultra_executado",
            "turbo_ultra_rodou",
            "turbo_bloqueado",
            "turbo_motivo_bloqueio",
            "motor_turbo_executado",
            "listas_intercept_orbita",
        ]
        for k in chaves:
            try:
                if k in st.session_state:
                    del st.session_state[k]
            except Exception:
                pass

    # 6) Aplicar janela
    if st.button("✅ Aplicar janela (recortar histórico ativo)", use_container_width=True):
        try:
            df_recorte = df_full.head(int(k_novo)).copy()
            st.session_state["historico_df"] = df_recorte
            st.session_state["replay_janela_k_active"] = int(k_novo)  # fixa janela ativa (não altera widget)
            _replay_limpar_chaves_dependentes()

            # Atualizar universo min/max canônico (derivado do recorte) — versão rápida (sem iterrows)
            try:
                col_pass = [c for c in df_recorte.columns if str(c).startswith("p")]
                if col_pass:
                    vals = df_recorte[col_pass].to_numpy().ravel()
                    # Filtra apenas inteiros positivos
                    vals = [int(v) for v in vals if str(v).strip() not in ("", "nan", "None")]
                    vals = [v for v in vals if v > 0]
                    if vals:
                        st.session_state["universo_min"] = int(min(vals))
                        st.session_state["universo_max"] = int(max(vals))
            except Exception:
                pass

            st.success(
                f"Janela aplicada: histórico ativo agora está em **C1..C{k_novo}** "
                f"(total: {len(df_recorte)} séries).\n\n"
                "Agora rode **Sentinelas → Pipeline → (TURBO opcional) → Modo 6**."
            )
        except Exception as e:
            st.error(f"Falha ao aplicar janela: {e}")
        st.stop()

    st.markdown("---")

    # 7) Registro do pacote da janela
    st.markdown("### 2) Registrar pacote gerado para esta janela")
    st.caption(
        "Depois de rodar o Modo 6 (e gerar o pacote), volte aqui e registre. "
        "O painel guarda um snapshot por janela (k)."
    )

    pacote_atual = st.session_state.get("pacote_listas_atual")
    if not pacote_atual:
        st.info(
            "Nenhum pacote atual encontrado. "
            "Rode o **🎯 Modo 6 Acertos — Execução** para gerar e congelar um pacote."
        )
    else:
        st.success(f"📦 Pacote atual detectado: **{len(pacote_atual)}** listas")

    # estrutura: {k: {"ts": "...", "qtd": int, "listas": [[...], ...]}}
    if "replay_progressivo_pacotes" not in st.session_state:
        st.session_state["replay_progressivo_pacotes"] = {}

    pacotes_reg = st.session_state.get("replay_progressivo_pacotes", {})
    st.caption(f"Pacotes registrados até agora: **{len(pacotes_reg)}**")

    colR1, colR2 = st.columns([1, 1])
    with colR1:
        if st.button("📌 Registrar pacote da janela atual", use_container_width=True, disabled=not bool(pacote_atual)):
            try:
                from datetime import datetime
                k_reg = int(st.session_state.get("replay_janela_k_active", k_novo))
                # --- V9 (BLOCO B) — snapshot estrutural do pacote (OBSERVACIONAL / ex-post) ---
                # Regra: separar o valor digitado (widget) do estado ativo e registrar um snapshot por janela.
                # Isso NÃO decide nada e NÃO altera listas — apenas guarda estrutura para avaliação posterior.
                try:
                    v8_snap = st.session_state.get("v8_borda_qualificada") or {}
                    # Se não houver snapshot V8 válido nesta rodada, recalcula de forma canônica a partir do pacote atual
                    if not isinstance(v8_snap, dict) or v8_snap.get("meta", {}).get("status") not in ("ok", "presenca_vazia"):
                        v8_snap = v8_classificar_borda_qualificada(
                            listas=[list(map(int, lst)) for lst in pacote_atual],
                            base_n=10,
                            core_presenca_min=0.60,
                            quase_delta=0.12,
                            max_borda_interna=6,
                            universo_min=st.session_state.get("universo_min"),
                            universo_max=st.session_state.get("universo_max"),
                            rigidez_info=st.session_state.get("v16_rigidez_info"),
                        )
                except Exception:
                    v8_snap = {"core": [], "quase_core": [], "borda_interna": [], "borda_externa": [], "meta": {"status": "snap_falhou"}}

                # Universo do pacote (união) — usado para classificar "miolo do pacote" vs "fora do pacote"
                try:
                    universo_pacote = sorted({int(x) for lst in pacote_atual for x in lst})
                except Exception:
                    universo_pacote = []

                pacotes_reg[k_reg] = {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "qtd": int(len(pacote_atual)),
                    "listas": [list(map(int, lst)) for lst in pacote_atual],
                    "snap_v9": {
                        "core": list(map(int, (v8_snap.get("core") or []))),
                        "quase_core": list(map(int, (v8_snap.get("quase_core") or []))),
                        "borda_interna": list(map(int, (v8_snap.get("borda_interna") or []))),
                        "borda_externa": list(map(int, (v8_snap.get("borda_externa") or []))),
                        "universo_pacote": list(map(int, universo_pacote)),
                        "meta": v8_snap.get("meta") or {},
                    },
                }
                st.session_state["replay_progressivo_pacotes"] = pacotes_reg
                st.success(f"Pacote registrado para janela C1..C{k_reg}.")
            except Exception as e:
                st.error(f"Falha ao registrar pacote: {e}")
    with colR2:
        if st.button("🧹 Limpar todos os pacotes registrados", use_container_width=True, disabled=(len(pacotes_reg) == 0)):
            st.session_state["replay_progressivo_pacotes"] = {}
            st.success("Pacotes registrados foram limpos.")

    st.markdown("---")

    # 8) Avaliação automática (contra os 2 alvos seguintes)
    st.markdown("### 3) Avaliar pacotes registrados (2 alvos seguintes)")
    st.caption(
        "Para cada janela k registrada, o painel testa o pacote contra os alvos **C(k+1)** e **C(k+2)** "
        "do histórico FULL (quando existirem)."
    )

    pacotes_reg = st.session_state.get("replay_progressivo_pacotes", {})
    if not pacotes_reg:
        st.info("Nenhum pacote registrado ainda.")
        st.stop()

    # colunas de passageiros
    col_pass_full = [c for c in df_full.columns if c.lower().startswith("p")]
    if not col_pass_full:
        st.error("Não consegui identificar colunas de passageiros no histórico FULL.")
        st.stop()

    def _alvo_da_linha(idx0: int):
        # idx0 é 0-based (linha do DF)
        row = df_full.iloc[idx0]
        alvo = []
        for c in col_pass_full:
            try:
                v = int(row[c])
                if v > 0:
                    alvo.append(v)
            except Exception:
                pass
        return set(alvo)

    
    # --- V9 (BLOCO B) — Memória de Borda (ex-post, observacional) ---
    # Para cada janela k registrada, além do "melhor acerto do pacote", também medimos:
    # onde caíram os acertos (CORE / quase-CORE / borda interna / borda externa / miolo do pacote / fora do pacote)
    def _v9_get_sets(info: dict):
        snap = info.get("snap_v9") or {}
        core = set(map(int, snap.get("core") or []))
        quase = set(map(int, snap.get("quase_core") or []))
        b_in = set(map(int, snap.get("borda_interna") or []))
        b_ex = set(map(int, snap.get("borda_externa") or []))
        uni = set(map(int, snap.get("universo_pacote") or []))
        return core, quase, b_in, b_ex, uni

    def _v9_contar_origens(alvo_set: set, core: set, quase: set, b_in: set, b_ex: set, uni: set):
        # Classificação disjunta (ordem canônica):
        # CORE > quase-CORE > borda interna > borda externa > miolo do pacote > fora do pacote
        out = {
            "core": 0,
            "quase": 0,
            "borda_in": 0,
            "borda_ex": 0,
            "miolo": 0,
            "fora": 0,
        }
        if not alvo_set:
            return out
        for n in alvo_set:
            if n in core:
                out["core"] += 1
            elif n in quase:
                out["quase"] += 1
            elif n in b_in:
                out["borda_in"] += 1
            elif n in b_ex:
                out["borda_ex"] += 1
            elif (uni and n in uni):
                out["miolo"] += 1
            else:
                out["fora"] += 1
        return out
    resultados = []
    for k_reg, info in sorted(pacotes_reg.items(), key=lambda x: int(x[0])):
        k_reg = int(k_reg)
        idx1 = k_reg  # C(k+1) em 0-based
        idx2 = k_reg + 1  # C(k+2)
        if idx1 >= total_series:
            continue

        alvo1 = _alvo_da_linha(idx1)
        alvo2 = _alvo_da_linha(idx2) if idx2 < total_series else None

        listas = info.get("listas", [])
        # melhor acerto em cada alvo
        best1 = 0
        best2 = 0
        if alvo1 and listas:
            for lst in listas:
                best1 = max(best1, len(set(lst) & alvo1))
        if alvo2 and listas:
            for lst in listas:
                best2 = max(best2, len(set(lst) & alvo2))

        # --- V9 (BLOCO B): onde caíram os acertos (ex-post) ---
        core, quase, b_in, b_ex, uni = _v9_get_sets(info)
        org1 = _v9_contar_origens(alvo1, core, quase, b_in, b_ex, uni) if alvo1 else None
        org2 = _v9_contar_origens(alvo2, core, quase, b_in, b_ex, uni) if alvo2 else None

        resultados.append({
            "janela_k": k_reg,
            "qtd_listas": int(info.get("qtd", 0)),
            "alvo_1": f"C{k_reg+1}",
            "best_acerto_alvo_1": int(best1),
            "core_hit_1": int(org1.get("core")) if org1 else None,
            "quase_hit_1": int(org1.get("quase")) if org1 else None,
            "borda_in_hit_1": int(org1.get("borda_in")) if org1 else None,
            "borda_ex_hit_1": int(org1.get("borda_ex")) if org1 else None,
            "miolo_hit_1": int(org1.get("miolo")) if org1 else None,
            "fora_hit_1": int(org1.get("fora")) if org1 else None,
            "alvo_2": f"C{k_reg+2}" if alvo2 is not None else "—",
            "best_acerto_alvo_2": int(best2) if alvo2 is not None else None,
            "core_hit_2": int(org2.get("core")) if org2 else None,
            "quase_hit_2": int(org2.get("quase")) if org2 else None,
            "borda_in_hit_2": int(org2.get("borda_in")) if org2 else None,
            "borda_ex_hit_2": int(org2.get("borda_ex")) if org2 else None,
            "miolo_hit_2": int(org2.get("miolo")) if org2 else None,
            "fora_hit_2": int(org2.get("fora")) if org2 else None,
            "ts_registro": str(info.get("ts", "")),
        })

    if not resultados:
        st.warning("Nenhum resultado para exibir (verifique se as janelas k registradas têm alvos seguintes no FULL).")
        st.stop()

    df_res = pd.DataFrame(resultados).sort_values(["janela_k"], ascending=True)
    st.dataframe(df_res, use_container_width=True, hide_index=True)

    # --- V9 (BLOCO B) — Resumo agregado (ex-post, observacional) ---
    try:
        cols1 = ["core_hit_1", "quase_hit_1", "borda_in_hit_1", "borda_ex_hit_1", "miolo_hit_1", "fora_hit_1"]
        cols2 = ["core_hit_2", "quase_hit_2", "borda_in_hit_2", "borda_ex_hit_2", "miolo_hit_2", "fora_hit_2"]

        tot = {"core": 0, "quase": 0, "borda_in": 0, "borda_ex": 0, "miolo": 0, "fora": 0}
        for c in cols1:
            if c in df_res.columns:
                tot[c.replace("_hit_1", "").replace("borda_in", "borda_in").replace("borda_ex", "borda_ex")] += int(df_res[c].fillna(0).sum())
        for c in cols2:
            if c in df_res.columns:
                tot[c.replace("_hit_2", "").replace("borda_in", "borda_in").replace("borda_ex", "borda_ex")] += int(df_res[c].fillna(0).sum())

        total_hits = sum(tot.values())
        if total_hits > 0:
            pct = {k: round(100.0 * v / total_hits, 1) for k, v in tot.items()}
        else:
            pct = {k: 0.0 for k in tot.keys()}

        st.markdown("### 🧠 V9 — Memória de Borda (Resumo Agregado · ex‑post)")
        st.caption("Agrega todos os alvos avaliados (C(k+1) e C(k+2) quando existirem). Não decide nada — só descreve onde os acertos nasceram.")
        st.write({
            "hits_total_agregado": int(total_hits),
            "CORE": f'{tot["core"]} ({pct["core"]}%)',
            "quase_CORE": f'{tot["quase"]} ({pct["quase"]}%)',
            "borda_interna": f'{tot["borda_in"]} ({pct["borda_in"]}%)',
            "borda_externa": f'{tot["borda_ex"]} ({pct["borda_ex"]}%)',
            "miolo_do_pacote": f'{tot["miolo"]} ({pct["miolo"]}%)',
            "fora_do_pacote": f'{tot["fora"]} ({pct["fora"]}%)',
        })
    except Exception:
        pass

    st.info(
        "Interpretação correta:\n"
        "- Este painel mede apenas o **melhor acerto** dentro do pacote para os 2 alvos seguintes.\n"
        "- Ele NÃO muda o motor, NÃO decide volume, e NÃO garante performance futura.\n"
        "- Serve para você comparar janelas e ver se o V8 está reduzindo perda por borda sem dispersar."
    )

    st.stop()


# ============================================================
# PARTE 3/8 — FIM
# ============================================================
# ============================================================
# PARTE 4/8 — INÍCIO
# ============================================================

# ============================================================
# Painel 6 — ⚙️ Modo TURBO++ HÍBRIDO
# ============================================================
if painel == "⚙️ Modo TURBO++ HÍBRIDO":

    st.markdown("## ⚙️ Modo TURBO++ HÍBRIDO — V15.7 MAX")

    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")

    if df is None or matriz_norm is None:
        exibir_bloco_mensagem(
            "Pipeline incompleto",
            "Execute o painel **🛣️ Pipeline V14-FLEX ULTRA**.",
            tipo="warning",
        )
        st.stop()

    qtd_series = len(df)

    # Anti-zumbi leve
    if not limitar_operacao(
        qtd_series,
        limite_series=LIMITE_PREVISOES_TURBO,
        contexto="TURBO++ HÍBRIDO",
        painel="⚙️ Modo TURBO++ HÍBRIDO",
    ):
        st.stop()

    st.info("Executando Modo TURBO++ HÍBRIDO...")

    # ============================================================
    # MOTOR HÍBRIDO — DX Light + S6 Light + Monte Carlo Light
    # ============================================================
    try:
        # DX Light — proximidade final
        vetor_final = matriz_norm[-1]
        distancias = [
            np.linalg.norm(vetor_final - linha) for linha in matriz_norm[:-1]
        ]

        # S6 Light — estatística simples dos passageiros
        col_pass = [c for c in df.columns if c.startswith("p")]
        ult = df[col_pass].iloc[-1].values

        s6_scores = []
        for idx in range(len(df) - 1):
            candidato = df[col_pass].iloc[idx].values
            intersec = len(set(candidato) & set(ult))
            s6_scores.append(intersec)

        # Monte Carlo Light — sorteio ponderado
        pesos_mc = np.array([1 / (1 + d) for d in distancias])
        soma_pesos = float(pesos_mc.sum()) if len(pesos_mc) > 0 else 0.0
        if soma_pesos <= 0.0 or np.isnan(soma_pesos):
            # fallback: distribuição uniforme (base insuficiente para ponderar)
            pesos_mc = np.ones(len(distancias), dtype=float)
            soma_pesos = float(pesos_mc.sum())
        pesos_mc = pesos_mc / soma_pesos

        escolha_idx = np.random.choice(len(pesos_mc), p=pesos_mc)
        previsao_mc = df[col_pass].iloc[escolha_idx].values.tolist()

        # Consolidação leve
        s6_melhor = df[col_pass].iloc[np.argmax(s6_scores)].values.tolist()
        dx_melhor = df[col_pass].iloc[np.argmin(distancias)].values.tolist()

        # Combinação híbrida
        previsao_final = list(
            np.round(
                0.4 * np.array(dx_melhor)
                + 0.3 * np.array(s6_melhor)
                + 0.3 * np.array(previsao_mc)
            )
        )
        previsao_final = [int(x) for x in previsao_final]

    except Exception as erro:
        exibir_bloco_mensagem(
            "Erro no TURBO++ HÍBRIDO",
            f"Detalhes: {erro}",
            tipo="error",
        )
        st.stop()

    # ============================================================
    # Exibição final
    # ============================================================
    st.markdown("### 🔮 Previsão HÍBRIDA (TURBO++)")
    st.success(f"**{formatar_lista_passageiros(previsao_final)}**")

    st.session_state["ultima_previsao"] = previsao_final

# ============================================================
# BLOCO 1/4 — ORQUESTRADOR DE TENTATIVA (V16) — INVISÍVEL
# Objetivo: traduzir diagnóstico (alvo/risco/confiabilidade) em
# "configuração de tentativa" para o Modo 6 (sem decidir listas).
# LISTAS SEMPRE EXISTEM: este orquestrador NUNCA retorna volume 0.
# ============================================================

from typing import Dict, Any, Optional


# ------------------------------------------------------------
# HELPERS (V16) — clamp + safe float
# ------------------------------------------------------------

def _clamp_v16(x: float, lo: float, hi: float) -> float:
    try:
        x = float(x)
    except Exception:
        x = lo
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _safe_float_v16(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


# ------------------------------------------------------------
# ORQUESTRADOR DE TENTATIVA (V16) — núcleo conceitual
# ------------------------------------------------------------

def orquestrar_tentativa_v16(
    *,
    series_count: int,
    alvo_tipo: Optional[str] = None,          # "parado" | "movimento_lento" | "movimento_rapido"
    alvo_velocidade: Optional[float] = None,  # ex: 0.9319 (se disponível)
    k_star: Optional[float] = None,           # ex: 0.2083
    nr_pct: Optional[float] = None,           # ex: 67.87  (0..100)
    divergencia_s6_mc: Optional[float] = None,# ex: 14.0480
    risco_composto: Optional[float] = None,   # ex: 0.7560  (0..1)
    confiabilidade_estimada: Optional[float] = None,  # 0..1 (se você já tiver)
    # --- Limites técnicos (anti-zumbi) ---
    limite_seguro_series_modo6: int = 800,    # padrão atual (já visto no app)
    # --- Volumes base (pode ser ajustado depois, mas COMEÇA CONSERVADOR) ---
    volume_min_base: int = 3,
    volume_rec_base: int = 6,
    volume_max_base: int = 80,
) -> Dict[str, Any]:
    """
    Retorna um dicionário com a "configuração de tentativa" (invisível),
    para o Modo 6 usar como guia de volume e forma (diversidade/variação).

    ✅ Regras implementadas aqui:
    - Objetivo único: tentar cravar 6 passageiros (não decide, só orienta).
    - Listas SEMPRE existem -> volume_min >= 1 (nunca 0).
    - Confiabilidade alta => explorar (mandar bala com critério).
    - Confiabilidade baixa => tentar com critério (degradado, mas não zero).
    - Anti-zumbi não censura: limita teto, mas não zera.
    """

    # -----------------------------
    # Sanitização básica
    # -----------------------------
    try:
        series_count = int(series_count)
    except Exception:
        series_count = 0

    k_star = _safe_float_v16(k_star, 0.0)
    nr_pct = _safe_float_v16(nr_pct, 0.0)
    divergencia_s6_mc = _safe_float_v16(divergencia_s6_mc, 0.0)
    risco_composto = _safe_float_v16(risco_composto, 0.0)

    # Normalizações defensivas
    nr_norm = _clamp_v16(nr_pct / 100.0, 0.0, 1.0)             # 0..1
    risco_norm = _clamp_v16(risco_composto, 0.0, 1.0)          # 0..1
    k_norm = _clamp_v16(k_star / 0.35, 0.0, 1.0)               # 0..1 (0.35 ~ teto típico de alerta)
    div_norm = _clamp_v16(divergencia_s6_mc / 15.0, 0.0, 1.0)  # 0..1 (15 ~ divergência crítica)

    # -----------------------------
    # Inferência do tipo de alvo (se não vier do Laudo)
    # -----------------------------
    alvo_tipo_norm = (alvo_tipo or "").strip().lower()

    if not alvo_tipo_norm:
        v = _safe_float_v16(alvo_velocidade, 0.0)
        # Heurística simples (pode refinar depois):
        # - <0.35: parado/lento
        # - 0.35..0.70: movimento_lento
        # - >0.70: movimento_rapido
        if v <= 0.35:
            alvo_tipo_norm = "parado"
        elif v <= 0.70:
            alvo_tipo_norm = "movimento_lento"
        else:
            alvo_tipo_norm = "movimento_rapido"

    if alvo_tipo_norm in ("lento", "movimento lento", "movimento-lento"):
        alvo_tipo_norm = "movimento_lento"
    if alvo_tipo_norm in ("rapido", "rápido", "movimento rapido", "movimento-rápido", "movimento_rapido"):
        alvo_tipo_norm = "movimento_rapido"
    if alvo_tipo_norm in ("parado", "estavel", "estável"):
        alvo_tipo_norm = "parado"

    if alvo_tipo_norm not in ("parado", "movimento_lento", "movimento_rapido"):
        alvo_tipo_norm = "movimento_rapido"  # default seguro: tratar como difícil

    # -----------------------------
    # Construção de uma "confiabilidade estimada" interna (se não vier)
    # -----------------------------
    # Ideia: confiabilidade cai com ruído, risco, k* alto e divergência alta.
    # (Não é promessa, é régua de orientação de intensidade.)
    if confiabilidade_estimada is None:
        penal = 0.40 * nr_norm + 0.25 * risco_norm + 0.20 * div_norm + 0.15 * k_norm
        conf = 1.0 - _clamp_v16(penal, 0.0, 1.0)
    else:
        conf = _clamp_v16(_safe_float_v16(confiabilidade_estimada, 0.0), 0.0, 1.0)

    # -----------------------------
    # Definição do "modo de tentativa" (conceito → controle interno)
    # -----------------------------
    # - exploração_intensa: alta confiança (mandar bala com critério)
    # - tentativa_controlada: meio termo
    # - tentativa_degradada: baixa confiança / alvo rápido / ambiente hostil
    if conf >= 0.55 and risco_norm <= 0.55 and nr_norm <= 0.55 and div_norm <= 0.60:
        modo = "exploracao_intensa"
    elif conf >= 0.30 and risco_norm <= 0.75 and nr_norm <= 0.75:
        modo = "tentativa_controlada"
    else:
        modo = "tentativa_degradada"

    # Alvo rápido puxa para degradado, a menos que seja realmente "bom"
    if alvo_tipo_norm == "movimento_rapido" and modo != "exploracao_intensa":
        modo = "tentativa_degradada"

    # -----------------------------
    # Volumes base (sempre > 0)
    # -----------------------------
    vol_min = max(1, int(volume_min_base))
    vol_rec = max(vol_min, int(volume_rec_base))
    vol_max = max(vol_rec, int(volume_max_base))

    # -----------------------------
    # Ajuste de intensidade por modo + confiabilidade
    # -----------------------------
    # Observação: "mandar bala" = aumentar volume e variação interna,
    # mas SEM explodir sem critério.
    if modo == "exploracao_intensa":
        # Escala com conf (0.55..1.0) -> multiplicador (1.1..1.9)
        mult = 1.1 + 0.8 * _clamp_v16((conf - 0.55) / 0.45, 0.0, 1.0)
        vol_rec = int(max(vol_rec, round(vol_rec * mult)))
        vol_max = int(max(vol_max, round(vol_max * mult)))

        diversidade = 0.55  # moderada (refino + variação)
        variacao_interna = 0.75
        aviso_curto = "🟢 Exploração intensa: mandar bala com critério (janela favorável)."

    elif modo == "tentativa_controlada":
        # Escala suave com conf (0.30..0.55) -> multiplicador (0.95..1.20)
        mult = 0.95 + 0.25 * _clamp_v16((conf - 0.30) / 0.25, 0.0, 1.0)
        vol_rec = int(max(vol_rec, round(vol_rec * mult)))
        vol_max = int(max(vol_max, round(vol_max * mult)))

        # diversidade depende do alvo
        if alvo_tipo_norm == "parado":
            diversidade = 0.35  # mais próximo (ajuste fino)
            variacao_interna = 0.60
        elif alvo_tipo_norm == "movimento_lento":
            diversidade = 0.50  # cercamento
            variacao_interna = 0.55
        else:
            diversidade = 0.65  # já puxa para hipóteses
            variacao_interna = 0.45

        aviso_curto = "🟡 Tentativa controlada: cercar com critério (sem exagero)."

    else:
        # Degradado: volume controlado, diversidade alta (hipóteses)
        # Garante mínimo, limita teto e aumenta diversidade.
        # Se conf for muito baixa, não adianta inflar volume: mantém enxuto.
        if conf <= 0.10:
            vol_rec = max(vol_min, min(vol_rec, 6))
            vol_max = max(vol_rec, min(vol_max, 12))
        elif conf <= 0.20:
            vol_rec = max(vol_min, min(vol_rec, 8))
            vol_max = max(vol_rec, min(vol_max, 18))
        else:
            vol_rec = max(vol_min, min(vol_rec, 10))
            vol_max = max(vol_rec, min(vol_max, 24))

        diversidade = 0.85  # alto (ali, lá, acolá)
        variacao_interna = 0.35
        aviso_curto = "🔴 Tentativa degradada: hipóteses espalhadas (chance baixa, mas listas existem)."

    # -----------------------------
    # Anti-zumbi como LIMITADOR (não censura)
    # -----------------------------
    # Se o histórico excede o limite seguro do modo 6:
    # - não bloqueia
    # - apenas derruba o teto e puxa recomendado para um patamar seguro
    # Mantém volume_min > 0 SEMPRE.
    if series_count > int(limite_seguro_series_modo6):
        # Fator de penalização pelo excesso de séries (piora custo)
        excesso = series_count - int(limite_seguro_series_modo6)
        fator = _clamp_v16(1.0 - (excesso / max(1.0, float(limite_seguro_series_modo6))) * 0.60, 0.25, 1.0)

        teto_seguro = int(max(vol_rec, round(vol_max * fator)))
        teto_seguro = int(_clamp_v16(teto_seguro, max(vol_rec, vol_min), vol_max))

        # puxa recomendado junto do teto seguro (mas nunca abaixo do mínimo)
        vol_max = max(vol_rec, teto_seguro)
        vol_rec = max(vol_min, min(vol_rec, vol_max))

        aviso_curto += " 🔒 Anti-Zumbi: volume limitado (sem bloquear geração)."

    # -----------------------------
    # Garantias finais (invioláveis)
    # -----------------------------
    vol_min = max(1, int(vol_min))
    vol_rec = max(vol_min, int(vol_rec))
    vol_max = max(vol_rec, int(vol_max))

    diversidade = _clamp_v16(diversidade, 0.10, 0.95)
    variacao_interna = _clamp_v16(variacao_interna, 0.10, 0.95)

    return {
        "modo_tentativa": modo,
        "alvo_tipo": alvo_tipo_norm,
        "confiabilidade_estimada": float(conf),
        "volume_min": int(vol_min),
        "volume_recomendado": int(vol_rec),
        "volume_max": int(vol_max),
        "diversidade": float(diversidade),
        "variacao_interna": float(variacao_interna),
        "aviso_curto": str(aviso_curto),
        "debug": {
            "nr_norm": float(nr_norm),
            "risco_norm": float(risco_norm),
            "k_norm": float(k_norm),
            "div_norm": float(div_norm),
            "series_count": int(series_count),
            "limite_seguro_series_modo6": int(limite_seguro_series_modo6),
        },
    }

# ============================================================
# BLOCO 2/4 — PONTE ORQUESTRADOR → TURBO++ ULTRA (V16)
# Objetivo: coletar diagnósticos existentes do app (Laudo/Risco)
# e preparar a configuração de tentativa para o Modo 6,
# SEM alterar UI e SEM decidir listas.
# ============================================================

def preparar_tentativa_turbo_ultra_v16(
    *,
    df,
    series_count: int,
    alvo_tipo: Optional[str] = None,
    alvo_velocidade: Optional[float] = None,
    k_star: Optional[float] = None,
    nr_pct: Optional[float] = None,
    divergencia_s6_mc: Optional[float] = None,
    risco_composto: Optional[float] = None,
    confiabilidade_estimada: Optional[float] = None,
    limite_seguro_series_modo6: int = 800,
) -> Dict[str, Any]:
    """
    Ponte invisível:
    - lê informações já calculadas no app
    - chama o Orquestrador de Tentativa (BLOCO 1)
    - devolve um dicionário pronto para o TURBO++ ULTRA usar

    NÃO gera listas
    NÃO executa motores
    NÃO decide nada
    """

    # Defesa básica
    try:
        series_count = int(series_count)
    except Exception:
        series_count = 0

    # Chamada central ao Orquestrador
    cfg = orquestrar_tentativa_v16(
        series_count=series_count,
        alvo_tipo=alvo_tipo,
        alvo_velocidade=alvo_velocidade,
        k_star=k_star,
        nr_pct=nr_pct,
        divergencia_s6_mc=divergencia_s6_mc,
        risco_composto=risco_composto,
        confiabilidade_estimada=confiabilidade_estimada,
        limite_seguro_series_modo6=limite_seguro_series_modo6,
    )

    # Normalização final (garantia extra)
    cfg["volume_min"] = max(1, int(cfg.get("volume_min", 1)))
    cfg["volume_recomendado"] = max(
        cfg["volume_min"],
        int(cfg.get("volume_recomendado", cfg["volume_min"]))
    )
    cfg["volume_max"] = max(
        cfg["volume_recomendado"],
        int(cfg.get("volume_max", cfg["volume_recomendado"]))
    )

    return cfg

# ============================================================
# >>> INÍCIO — BLOCO 3/4 — ORQUESTRADOR → TURBO++ ULTRA (V16)
# Camada invisível de conexão (não é painel, não gera listas)
# ============================================================

def _injetar_cfg_tentativa_turbo_ultra_v16(
    *,
    df,
    qtd_series: int,
    k_star,
    limite_series_padrao: int,
):
    """
    Injeta no session_state a configuração de tentativa calculada
    pelo Orquestrador (BLOCO 1 + BLOCO 2), sem bloquear execução.
    """

    # Coleta informações já existentes
    laudo_v16 = st.session_state.get("laudo_operacional_v16", {}) or {}

    alvo_tipo = laudo_v16.get("estado_alvo") or laudo_v16.get("alvo_tipo")
    alvo_velocidade = laudo_v16.get("velocidade_estimada")

    nr_pct = st.session_state.get("nr_pct")
    divergencia_s6_mc = st.session_state.get("divergencia_s6_mc")
    risco_composto = st.session_state.get("indice_risco")

    cfg = preparar_tentativa_turbo_ultra_v16(
        df=df,
        series_count=qtd_series,
        alvo_tipo=alvo_tipo,
        alvo_velocidade=alvo_velocidade,
        k_star=k_star,
        nr_pct=nr_pct,
        divergencia_s6_mc=divergencia_s6_mc,
        risco_composto=risco_composto,
        limite_seguro_series_modo6=limite_series_padrao,
    )

    # Guarda para uso posterior
    st.session_state["cfg_tentativa_turbo_ultra"] = cfg

    # Aviso curto (informativo, não bloqueante)
    aviso = cfg.get("aviso_curto")
    if aviso:
        st.caption(aviso)

    # Define limite efetivo (anti-zumbi vira limitador, não censura)
    limite_efetivo = min(
        limite_series_padrao,
        int(cfg.get("volume_max", limite_series_padrao))
    )

    return limite_efetivo


# ============================================================
# <<< FIM — BLOCO 3/4 — ORQUESTRADOR → TURBO++ ULTRA (V16)
# ============================================================

# ============================================================
# >>> PAINEL 7 — ⚙️ Modo TURBO++ ULTRA (MVP3 — VOLUME POR ORÇAMENTO)
# ============================================================

if painel == "⚙️ Modo TURBO++ ULTRA":

    st.markdown("## ⚙️ Modo TURBO++ ULTRA — MVP3")
    st.caption(
        "Exploração controlada.\n\n"
        "✔ Motor original preservado\n"
        "✔ Anti-zumbi respeitado\n"
        "✔ Volume liberado por orçamento\n"
        "✔ Falha silenciosa permitida\n"
        "✔ Sem decisão automática"
    )

    # ------------------------------------------------------------
    # BUSCA DE ESTADO (SOMENTE AQUI)
    # ------------------------------------------------------------
    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")

    _kstar_raw = st.session_state.get("sentinela_kstar")
    k_star = float(_kstar_raw) if isinstance(_kstar_raw, (int, float)) else 0.0

    if df is None or df.empty or matriz_norm is None:
        exibir_bloco_mensagem(
            "Pipeline incompleto",
            "Carregue o histórico e execute **🛣️ Pipeline V14-FLEX ULTRA**.",
            tipo="warning",
        )
        st.stop()

    qtd_series = len(df)

    # ------------------------------------------------------------
    # 🔒 MARCAÇÃO OFICIAL — TURBO EXECUTADO (ANTES DO ANTI-ZUMBI)
    # (Se o Anti-Zumbi bloquear com st.stop, o Modo 6 NÃO fica travado)
    # ------------------------------------------------------------
    st.session_state["turbo_ultra_executado"] = True
    st.session_state["turbo_executado"] = True
    st.session_state["turbo_ultra_rodou"] = True
    st.session_state["motor_turbo_executado"] = True

    # ------------------------------------------------------------
    # ANTI-ZUMBI — LIMITADOR OFICIAL
    # (COMPORTAMENTO OFICIAL PRESERVADO)
    # ------------------------------------------------------------
    LIMITE_SERIES_TURBO_ULTRA_EFETIVO = _injetar_cfg_tentativa_turbo_ultra_v16(
        df=df,
        qtd_series=qtd_series,
        k_star=k_star,
        limite_series_padrao=LIMITE_SERIES_TURBO_ULTRA,
    )

    limitar_operacao(
        qtd_series,
        limite_series=LIMITE_SERIES_TURBO_ULTRA_EFETIVO,
        contexto="TURBO++ ULTRA",
        painel="⚙️ Modo TURBO++ ULTRA",
    )
    # ⬆️ se bloquear, a própria função já dá st.stop()
    # ✅ e a marcação acima já foi gravada, então o Modo 6 destrava.

    # ------------------------------------------------------------
    # ORÇAMENTO
    # ------------------------------------------------------------
    orcamentos_disponiveis = [6, 42, 168, 504, 1260, 2772]

    orcamento = st.selectbox(
        "Selecione o orçamento para o TURBO++ ULTRA:",
        options=orcamentos_disponiveis,
        index=1,
    )

    mapa_execucoes = {
        6: 1,
        42: 1,
        168: 3,
        504: 6,
        1260: 10,
        2772: 20,
    }

    n_exec = mapa_execucoes.get(int(orcamento), 1)

    st.info(
        f"🔢 Orçamento selecionado: **{orcamento}**\n\n"
        f"▶️ Execuções do TURBO++ ULTRA: **{n_exec}**"
    )

    # ------------------------------------------------------------
    # JANELA LOCAL DE ATAQUE (GATILHO CANÔNICO)
    # ------------------------------------------------------------
    # Usa APENAS sinais já existentes em session_state.
    # Não decide postura. Só governa se o TURBO ofensivo pode tentar nesta rodada.
    m3_reg = st.session_state.get("m3_regime_dx") or st.session_state.get("m3_regime")
    classe_risco = st.session_state.get("classe_risco")
    if classe_risco is None:
        classe_risco = st.session_state.get("classe_risco_texto")
    nr_percent = st.session_state.get("nr_percent")
    div_s6_mc = st.session_state.get("divergencia_s6_mc")
    compressao_core = st.session_state.get("janela_compressao_core", False)

    # Critério mínimo (canônico): compressão + M3 em ECO/PRÉ-ECO + risco não hostil.
    m3_ok = str(m3_reg).upper() in ["ECO", "PRÉ-ECO", "PRE", "PRE-ECO", "PRE ECO", "PRÉ ECO"]
    risco_txt = str(classe_risco) if classe_risco is not None else ""
    risco_ok = ("Baixo" in risco_txt) or ("Moderado" in risco_txt) or ("🟡" in risco_txt) or ("🟢" in risco_txt)
    janela_ativa = bool(compressao_core and m3_ok and risco_ok)

    st.session_state["janela_local_ativa"] = janela_ativa
    st.session_state["janela_local_m3"] = m3_reg if m3_reg is not None else "N/D"
    st.session_state["janela_local_classe_risco"] = classe_risco if classe_risco is not None else "N/D"

    if not janela_ativa:
        st.info("🧨 Janela Local de Ataque: **NÃO ATIVA** — TURBO ofensivo não tentado nesta rodada (governança).")
        st.session_state["turbo_ultra_executado"] = False
        st.session_state["turbo_ultra_listas_leves"] = []
        st.session_state["turbo_ultra_listas"] = []
        # ainda marcamos como "tentado" no sentido de que o painel foi visitado e governou a tentativa
        st.session_state["turbo_ultra_tentado"] = True
        # encerra este painel aqui, sem gerar listas
        st.stop()

    # ------------------------------------------------------------
    # EXECUÇÃO SEGURA DO TURBO++ ULTRA
    # ------------------------------------------------------------
    st.info("Executando Modo TURBO++ ULTRA...")

    todas_listas = []

    for _ in range(n_exec):
        try:
            lista = turbo_ultra_v15_7(
                df=df,
                matriz_norm=matriz_norm,
                k_star=k_star,
            )
            if isinstance(lista, list) and len(lista) >= 6:
                todas_listas.append(lista)
        except Exception:
            pass

    # ------------------------------------------------------------
    # FECHAMENTO DE ESTADO DO PIPELINE
    # ------------------------------------------------------------
    st.session_state["pipeline_flex_ultra_concluido"] = True
    st.session_state["turbo_ultra_listas_leves"] = todas_listas.copy()
    st.session_state["ultima_previsao"] = todas_listas
    v16_blindar_ultima_previsao_universo()

    # Blindagem adicional — estado intermediário reutilizável
    st.session_state["turbo_ultra_listas_leves"] = st.session_state["ultima_previsao"]

    if not todas_listas:
        st.warning(
            "Nenhuma lista foi gerada nesta condição.\n\n"
            "Isso é um **resultado válido**.\n"
            "O motor foi executado (ou bloqueado) e falhou silenciosamente."
        )
        st.stop()

    st.success(
        f"✅ TURBO++ ULTRA executado com sucesso.\n\n"
        f"📦 Listas geradas: **{len(todas_listas)}**"
    )

    st.markdown("### 🔮 Listas geradas (amostra)")
    st.write(todas_listas[: min(5, len(todas_listas))])

# ============================================================
# <<< FIM — PAINEL 7 — ⚙️ Modo TURBO++ ULTRA (MVP3)
# ============================================================





# ============================================================
# MOTORES PROFUNDOS (PUROS)
# NÃO executam sozinhos
# NÃO acessam session_state
# NÃO exibem nada
# ============================================================

# --- S6 PROFUNDO ---
def s6_profundo_V157(df_local, col_pass, idx_alvo):
    ult_local = df_local[col_pass].iloc[idx_alvo].values
    scores_local = []
    for i_local in range(len(df_local) - 1):
        base_local = df_local[col_pass].iloc[i_local].values
        inter_local = len(set(base_local) & set(ult_local))
        scores_local.append(inter_local)
    melhores_idx_local = np.argsort(scores_local)[-25:]
    candidatos_local = df_local[col_pass].iloc[melhores_idx_local].values
    return candidatos_local


# --- MICRO-LEQUE PROFUNDO ---
def micro_leque_profundo(base, profundidade=20, universo_min=1, universo_max=60):
    leque = []
    umin = int(universo_min) if universo_min is not None else 1
    umax = int(universo_max) if universo_max is not None else 60

    for delta in range(-profundidade, profundidade + 1):
        novo = [max(umin, min(umax, int(x) + delta)) for x in base]
        leque.append(novo)

    return np.array(leque)


# --- MONTE CARLO PROFUNDO ---
def monte_carlo_profundo(base, n=800, universo_min=1, universo_max=60):
    sims = []
    umin = int(universo_min) if universo_min is not None else 1
    umax = int(universo_max) if universo_max is not None else 60

    base_arr = np.array([int(x) for x in base], dtype=int)

    for _ in range(n):
        ruido = np.random.randint(-5, 6, size=len(base_arr))
        candidato = base_arr + ruido
        candidato = np.clip(candidato, umin, umax)
        sims.append(candidato.tolist())

    return sims


# ============================================================
# Painel 8 — 📡 Painel de Ruído Condicional
# ============================================================


if painel == "📡 Painel de Ruído Condicional":

    st.markdown("## 📡 Painel de Ruído Condicional — V15.7 MAX")

    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")

    if df is None or matriz_norm is None:
        exibir_bloco_mensagem(
            "Pipeline não encontrado",
            "Execute primeiro **📁 Carregar Histórico** e **🛣️ Pipeline V14-FLEX ULTRA**.",
            tipo="warning",
        )
        st.stop()

    qtd_series = len(df)

    if not limitar_operacao(
        qtd_series,
        limite_series=LIMITE_SERIES_REPLAY_ULTRA,
        contexto="Ruído Condicional",
        painel="📡 Painel de Ruído Condicional",
    ):
        st.stop()

    st.info("Calculando indicadores de ruído condicional...")

# ============================================================
# PARTE 4/8 — FIM
# ============================================================
# ============================================================
# PARTE 5/8 — INÍCIO
# ============================================================

    try:
        if matriz_norm is None or len(matriz_norm) < 2:
            raise ValueError("Base insuficiente para medir ruído (matriz_norm < 2).")

        # Ruído Tipo A: dispersão intra-série (variação entre passageiros)
        variancias_intra = np.var(matriz_norm, axis=1)
        ruido_A_medio = float(np.mean(variancias_intra))

        # Ruído Tipo B: salto entre séries consecutivas (DX Light já usado)
        saltos = []
        for i in range(1, len(matriz_norm)):
            dist = np.linalg.norm(matriz_norm[i] - matriz_norm[i - 1])
            saltos.append(dist)
        ruido_B_medio = float(np.mean(saltos))

        # Normalização aproximada dos ruídos em [0,1]
        # (evitando divisão por zero)
        ruido_A_norm = min(1.0, ruido_A_medio / 0.08)   # escala empírica
        ruido_B_norm = min(1.0, ruido_B_medio / 1.20)   # escala empírica

        nr_percent = float((0.55 * ruido_A_norm + 0.45 * ruido_B_norm) * 100.0)

    except Exception as erro:
        exibir_bloco_mensagem(
            "Ruído indeterminado (base insuficiente / ruído técnico)",
            f"Métrica de ruído não pôde ser calculada com segurança.\n\nDetalhes técnicos: {erro}",
            tipo="warning",
        )
        st.session_state["nr_percent"] = None
        st.stop()

    # Classificação simples do NR%
    if nr_percent < 20:
        classe = "🟢 Baixo Ruído (Ambiente limpo)"
    elif nr_percent < 40:
        classe = "🟡 Ruído Moderado (Cuidado)"
    elif nr_percent < 60:
        classe = "🟠 Ruído Elevado (Atenção forte)"
    else:
        classe = "🔴 Ruído Crítico (Alta contaminação)"

    corpo = (
        f"- Séries analisadas: **{qtd_series}**\n"
        f"- Ruído Tipo A (intra-série, médio): **{ruido_A_medio:.4f}**\n"
        f"- Ruído Tipo B (entre séries, médio): **{ruido_B_medio:.4f}**\n"
        f"- NR% (Ruído Condicional Normalizado): **{nr_percent:.2f}%**\n"
        f"- Classe de ambiente: {classe}"
    )

    exibir_bloco_mensagem(
        "Resumo do Ruído Condicional",
        corpo,
        tipo="info",
    )

    st.session_state["nr_percent"] = nr_percent
    st.success("Cálculo de Ruído Condicional concluído!")


# ============================================================
# Painel 9 — 📉 Painel de Divergência S6 vs MC
# ============================================================
if painel == "📉 Painel de Divergência S6 vs MC":

    st.markdown("## 📉 Painel de Divergência S6 vs MC — V15.7 MAX")

    # Sincroniza alias de divergência (div_s6_mc -> divergencia_s6_mc)
    v16_sync_aliases_canonicos()


    divergencia = st.session_state.get("div_s6_mc", None)

    if divergencia is None:
        exibir_bloco_mensagem(
            "Divergência não calculada",
            "Execute o painel **⚙️ Modo TURBO++ ULTRA** para gerar a divergência S6 vs MC.",
            tipo="warning",
        )
        st.stop()

    # Classificação da divergência
    if divergencia < 2.0:
        classe = "🟢 Alta Convergência (S6 ≈ MC)"
        comentario = (
            "Os motores S6 Profundo e Monte Carlo Profundo estão altamente alinhados. "
            "O núcleo preditivo é mais confiável, favorecendo decisões mais agressivas."
        )
    elif divergencia < 5.0:
        classe = "🟡 Convergência Parcial"
        comentario = (
            "Há uma diferença moderada entre S6 e Monte Carlo. "
            "As decisões permanecem utilizáveis, mas requerem atenção adicional."
        )
    else:
        classe = "🔴 Alta Divergência (S6 distante de MC)"
        comentario = (
            "Os motores S6 e Monte Carlo estão em desacordo significativo. "
            "A recomendação é reduzir agressividade, aumentar coberturas ou aguardar estabilização."
        )

    corpo = (
        f"- Divergência S6 vs MC (norma): **{divergencia:.4f}**\n"
        f"- Classe de alinhamento: {classe}\n\n"
        f"{comentario}"
    )

    exibir_bloco_mensagem(
        "Resumo da Divergência S6 vs MC",
        corpo,
        tipo="info",
    )

    st.success("Análise de divergência concluída!")

# ============================================================
# PAINEL — 🧼 B1 | Higiene de Passageiros (V16)
# Observacional | NÃO decide | NÃO altera motores
# ============================================================

elif painel == "🧼 B1 — Higiene de Passageiros":

    st.markdown("## 🧼 B1 — Higiene de Passageiros (V16)")
    st.caption(
        "Leitura observacional para identificar passageiros resilientes e nocivos.\n"
        "Não remove números. Não decide listas. Preparação para Perna B."
    )

    df = st.session_state.get("historico_df")

    if df is None or df.empty:
        st.info("Histórico não carregado.")
        st.stop()

    # ------------------------------------------------------------
    # Detecta colunas de passageiros (n-base)
    # ------------------------------------------------------------
    col_pass = [c for c in df.columns if c.startswith("p")]

    if len(col_pass) < 6:
        st.warning("Não foi possível detectar passageiros suficientes.")
        st.stop()

    # ------------------------------------------------------------
    # Frequência simples dos passageiros
    # ------------------------------------------------------------
    freq = {}

    for _, row in df.iterrows():
        for c in col_pass:
            try:
                v = int(row[c])
                if v > 0:
                    freq[v] = freq.get(v, 0) + 1
            except Exception:
                pass

    if not freq:
        st.warning("Frequência de passageiros vazia.")
        st.stop()

    total_series = len(df)

    # ------------------------------------------------------------
    # Métricas observacionais
    # ------------------------------------------------------------
    dados = []

    for p, f in freq.items():
        taxa = f / total_series

        # heurísticas simples (OBSERVAÇÃO)
        resiliente = taxa >= 0.18
        nocivo = taxa <= 0.05

        dados.append({
            "Passageiro": p,
            "Ocorrências": f,
            "Taxa": round(taxa, 4),
            "Resiliente": "✅" if resiliente else "",
            "Nocivo": "⚠️" if nocivo else "",
        })

    df_pass = pd.DataFrame(dados).sort_values(
        by="Taxa", ascending=False
    )

    # ------------------------------------------------------------
    # Exibição
    # ------------------------------------------------------------
    st.markdown("### 📊 Leitura de Frequência dos Passageiros")
    st.dataframe(df_pass, use_container_width=True)

    # ------------------------------------------------------------
    # Síntese mastigada (NÃO decisória)
    # ------------------------------------------------------------
    resilientes = df_pass[df_pass["Resiliente"] == "✅"]["Passageiro"].tolist()
    nocivos = df_pass[df_pass["Nocivo"] == "⚠️"]["Passageiro"].tolist()

    leitura = (
        f"- Passageiros resilientes (recorrência alta): {resilientes[:12]}\n"
        f"- Passageiros potencialmente nocivos (recorrência muito baixa): {nocivos[:12]}\n\n"
        "⚠️ Nenhum passageiro foi removido.\n"
        "⚠️ Esta leitura serve apenas como preparação para refinamento futuro."
    )

    exibir_bloco_mensagem(
        "🧠 Leitura Observacional — Higiene de Passageiros",
        leitura,
        tipo="info",
    )

    # ------------------------------------------------------------
    # Registro silencioso para Perna B
    # ------------------------------------------------------------
    st.session_state["b1_resilientes"] = resilientes
    st.session_state["b1_nocivos"] = nocivos

    st.success("B1 concluído — leitura registrada com sucesso.")

# ============================================================
# <<< FIM — PAINEL 🧼 B1 | Higiene de Passageiros
# ============================================================

# ============================================================
# PAINEL — 🧩 B2 | Coerência Interna das Listas (V16)
# Observacional | NÃO decide | NÃO altera motores
# ============================================================

elif painel == "🧩 B2 — Coerência Interna das Listas":

    st.markdown("## 🧩 B2 — Coerência Interna das Listas (V16)")
    st.caption(
        "Leitura observacional de coesão e conflitos internos das listas.\n"
        "Não filtra, não prioriza, não decide."
    )

    # ------------------------------------------------------------
    # Fonte das listas (preferência: Modo 6)
    # ------------------------------------------------------------
    listas = (
        st.session_state.get("modo6_listas_totais")
        or st.session_state.get("modo6_listas")
        or []
    )

    if not listas:
        st.info("Nenhuma lista disponível para análise. Execute o Modo 6.")
        st.stop()

    # ------------------------------------------------------------
    # Universo e estatísticas globais
    # ------------------------------------------------------------
    todas = [x for lst in listas for x in lst if isinstance(x, int)]
    if not todas:
        st.warning("Listas inválidas para análise.")
        st.stop()

    freq_global = pd.Series(todas).value_counts(normalize=True)

    # ------------------------------------------------------------
    # Métricas por lista
    # ------------------------------------------------------------
    linhas = []

    for i, lst in enumerate(listas, start=1):
        lst = [int(x) for x in lst if isinstance(x, int)]
        if not lst:
            continue

        # Coesão: média da frequência global dos elementos
        coesao = float(freq_global.loc[lst].mean()) if set(lst).issubset(freq_global.index) else 0.0

        # Conflito simples: proporção de pares muito raros juntos
        pares = [(a, b) for a in lst for b in lst if a < b]
        raros = 0
        for a, b in pares:
            fa = freq_global.get(a, 0.0)
            fb = freq_global.get(b, 0.0)
            if fa < 0.05 and fb < 0.05:
                raros += 1

        conflito = raros / max(1, len(pares))

        linhas.append({
            "Lista": i,
            "Coesão (↑ melhor)": round(coesao, 4),
            "Conflito (↓ melhor)": round(conflito, 4),
        })

    df_b2 = pd.DataFrame(linhas)

    # ------------------------------------------------------------
    # Exibição
    # ------------------------------------------------------------
    st.markdown("### 📊 Coesão e Conflito por Lista")
    st.dataframe(df_b2, use_container_width=True)

    # ------------------------------------------------------------
    # Síntese mastigada (NÃO decisória)
    # ------------------------------------------------------------
    leitura = (
        "- **Coesão** alta indica elementos com histórico compatível.\n"
        "- **Conflito** alto indica combinações raras juntas.\n\n"
        "⚠️ Nenhuma lista foi removida ou priorizada.\n"
        "⚠️ Use esta leitura apenas para preparação."
    )

    exibir_bloco_mensagem(
        "🧠 Leitura Observacional — Coerência Interna",
        leitura,
        tipo="info",
    )

    # Registro silencioso
    st.session_state["b2_coerencia_df"] = df_b2

    st.success("B2 concluído — leitura registrada com sucesso.")

# ============================================================
# <<< FIM — PAINEL 🧩 B2 | Coerência Interna das Listas
# ============================================================

# ============================================================
# PAINEL — 🟢 B3 | Prontidão para Refinamento (V16)
# Observacional | NÃO decide | NÃO altera motores
# ============================================================

elif painel == "🟢 B3 — Prontidão (Refinamento)":

    st.markdown("## 🟢 B3 — Prontidão para Refinamento (V16)")
    st.caption(
        "Avalia se o contexto permite avançar da leitura (Perna B) "
        "para refinamento de passageiros. Não executa nada."
    )

    # ------------------------------------------------------------
    # Leituras já consolidadas
    # ------------------------------------------------------------
    diag = st.session_state.get("diagnostico_eco_estado_v16", {})
    df_b2 = st.session_state.get("b2_coerencia_df")

    if not diag or df_b2 is None or df_b2.empty:
        st.info(
            "Leituras insuficientes para avaliar prontidão.\n\n"
            "Execute B1, B2 e Diagnóstico ECO & Estado."
        )
        st.stop()

    eco = diag.get("eco")
    eco_persist = diag.get("eco_persistencia")
    acion = diag.get("eco_acionabilidade")
    estado = diag.get("estado")
    estado_ok = diag.get("estado_confiavel")

    # ------------------------------------------------------------
    # Heurísticas de prontidão (OBSERVAÇÃO)
    # ------------------------------------------------------------
    sinais_ok = []

    if eco in ("médio", "forte"):
        sinais_ok.append("ECO ≥ médio")

    if eco == "fraco" and eco_persist == "persistente" and estado in ("parado", "movimento_lento"):
        sinais_ok.append("ECO fraco porém estável com estado calmo")

    if estado_ok and estado in ("parado", "movimento_lento"):
        sinais_ok.append("Estado desacelerado e confiável")

    # Coesão média das listas
    coesao_media = float(df_b2["Coesão (↑ melhor)"].mean())

    if coesao_media >= 0.12:
        sinais_ok.append("Coesão média aceitável")

    # ------------------------------------------------------------
    # Veredito OBSERVACIONAL
    # ------------------------------------------------------------
    pronto = len(sinais_ok) >= 3

    if pronto:
        status = "🟢 PRONTO PARA REFINAMENTO"
        detalhe = (
            "O contexto permite iniciar refinamento controlado de passageiros.\n"
            "A Perna B pode evoluir para ações leves (sem afunilar)."
        )
        tipo = "success"
    else:
        status = "🟡 AINDA EM PREPARAÇÃO"
        detalhe = (
            "O contexto ainda pede dispersão.\n"
            "Continue observando e acumulando leitura."
        )
        tipo = "info"

    corpo = (
        f"**Status:** {status}\n\n"
        f"**Sinais atendidos:** {sinais_ok if sinais_ok else 'Nenhum'}\n\n"
        f"**Coesão média das listas:** {coesao_media:.4f}\n\n"
        f"⚠️ Este painel **não executa refinamento**.\n"
        f"⚠️ Serve apenas para indicar **prontidão**."
    )

    exibir_bloco_mensagem(
        "🧠 Veredito de Prontidão — Perna B",
        corpo,
        tipo=tipo,
    )

    # Registro silencioso
    st.session_state["b3_pronto_refinar"] = pronto

    st.success("B3 concluído — prontidão avaliada.")

# ============================================================
# <<< FIM — PAINEL 🟢 B3 | Prontidão para Refinamento
# ============================================================

# ============================================================
# PAINEL — 🟣 B4 | Refinamento Leve de Passageiros (V16)
# Ajuste leve | Reversível | NÃO decide | NÃO afunila
# ============================================================

elif painel == "🟣 B4 — Refinamento Leve de Passageiros":

    st.markdown("## 🟣 B4 — Refinamento Leve de Passageiros (V16)")
    st.caption(
        "Aplica ajustes leves e reversíveis nos passageiros das listas.\n"
        "Não reduz volume, não prioriza, não decide."
    )

    # ------------------------------------------------------------
    # Pré-condições
    # ------------------------------------------------------------
    pronto = st.session_state.get("b3_pronto_refinar", False)
    listas = (
        st.session_state.get("modo6_listas_totais")
        or st.session_state.get("modo6_listas")
        or []
    )

    resilientes = st.session_state.get("b1_resilientes", [])
    nocivos = st.session_state.get("b1_nocivos", [])

    if not listas:
        st.info("Nenhuma lista disponível. Execute o Modo 6.")
        st.stop()

    if not pronto:
        st.warning(
            "Contexto ainda não marcado como pronto para refinamento.\n"
            "Este painel é **apenas demonstrativo** neste estado."
        )

    # ------------------------------------------------------------
    # Universo de referência
    # ------------------------------------------------------------
    universo = sorted({int(x) for lst in listas for x in lst if isinstance(x, int)})
    if not universo:
        st.warning("Universo inválido para refinamento.")
        st.stop()

    rng = np.random.default_rng(42)

    # ------------------------------------------------------------
    # Refinamento leve (heurístico, reversível)
    # ------------------------------------------------------------
    listas_refinadas = []

    for lst in listas:
        nova = list(lst)

        # substitui no máx. 1 passageiro nocivo por um resiliente
        candidatos_nocivos = [x for x in nova if x in nocivos]
        candidatos_resilientes = [x for x in resilientes if x not in nova]

        if candidatos_nocivos and candidatos_resilientes:
            sai = rng.choice(candidatos_nocivos)
            entra = rng.choice(candidatos_resilientes)
            nova = [entra if x == sai else x for x in nova]

        listas_refinadas.append(sorted(set(nova)))

    # ------------------------------------------------------------
    # Exibição comparativa (leitura)
    # ------------------------------------------------------------
    st.markdown("### 🔍 Comparação — Antes × Depois (amostra)")
    limite = min(10, len(listas))

    for i in range(limite):
        col1, col2 = st.columns(2)
        with col1:
            st.code(f"Antes {i+1}: {sorted(listas[i])}", language="python")
        with col2:
            st.code(f"Depois {i+1}: {listas_refinadas[i]}", language="python")

    # ------------------------------------------------------------
    # Síntese observacional
    # ------------------------------------------------------------
    leitura = (
        "- Ajuste máximo: **1 passageiro por lista**\n"
        "- Volume total preservado\n"
        "- Refinamento **reversível**\n"
        "- Uso de passageiros resilientes\n\n"
        "⚠️ As listas refinadas **não substituem** as originais.\n"
        "⚠️ Cabe ao operador decidir se usa esta leitura."
    )

    exibir_bloco_mensagem(
        "🧠 Leitura Observacional — Refinamento Leve",
        leitura,
        tipo="info",
    )

    # Registro silencioso (não substitui listas)
    st.session_state["b4_listas_refinadas"] = listas_refinadas

    st.success("B4 concluído — refinamento leve avaliado.")

# ============================================================
# <<< FIM — PAINEL 🟣 B4 | Refinamento Leve de Passageiros
# ============================================================


# ============================================================
# Painel 10 — 🧭 Monitor de Risco — k & k*
# ============================================================
if painel == "🧭 Monitor de Risco — k & k*":

    st.markdown("## 🧭 Monitor de Risco — k & k* — V15.7 MAX")

    df = st.session_state.get("historico_df")
    k_star = st.session_state.get("sentinela_kstar")
    nr_percent = st.session_state.get("nr_percent")
    divergencia = st.session_state.get("div_s6_mc")

    if df is None:
        exibir_bloco_mensagem(
            "Histórico não carregado",
            "Vá ao painel **📁 Carregar Histórico** antes.",
            tipo="warning",
        )
        st.stop()

    metricas = calcular_metricas_basicas_historico(df)

    qtd_series = metricas.get("qtd_series", 0)
    min_k = metricas.get("min_k")
    max_k = metricas.get("max_k")
    media_k = metricas.get("media_k")

    # Garantias (se sentinelas/ruído/divergência não tiverem sido rodados)
    if k_star is None:
        k_star = 0.25  # valor neutro
    if nr_percent is None:
        nr_percent = 35.0  # ruído moderado default
    if divergencia is None:
        divergencia = 4.0  # divergência intermediária

    # Índice de risco composto (escala 0 a 1)
    # k* alto, NR% alto e divergência alta => risco maior
    kstar_norm = min(1.0, k_star / 0.50)
    nr_norm = min(1.0, nr_percent / 70.0)
    div_norm = min(1.0, divergencia / 8.0)

    indice_risco = float(0.40 * kstar_norm + 0.35 * nr_norm + 0.25 * div_norm)

    # Classificação de risco
    if indice_risco < 0.30:
        classe_risco = "🟢 Risco Baixo (Janela Favorável)"
        recomendacao = (
            "O ambiente está favorável para decisões mais agressivas, "
            "com menor necessidade de coberturas pesadas."
        )
    elif indice_risco < 0.55:
        classe_risco = "🟡 Risco Moderado"
        recomendacao = (
            "Ambiente misto. Recomenda-se equilíbrio entre núcleo e coberturas, "
            "com atenção à divergência e ao ruído."
        )
    elif indice_risco < 0.80:
        classe_risco = "🟠 Risco Elevado"
        recomendacao = (
            "Ambiente turbulento. Aumentar coberturas, reduzir exposição e "
            "observar de perto os painéis de Ruído e Divergência."
        )
    else:
        classe_risco = "🔴 Risco Crítico"
        recomendacao = (
            "Condição crítica. Sugere-se extrema cautela, priorizando preservação e "
            "eventualmente aguardando melhoria do regime antes de decisões mais fortes."
        )

    corpo = (
        f"- Séries no histórico: **{qtd_series}**\n"
        f"- k mínimo: **{min_k}** · k máximo: **{max_k}** · k médio: **{media_k:.2f}**\n"
        f"- k* (sentinela): **{k_star:.4f}**\n"
        f"- NR% (Ruído Condicional): **{nr_percent:.2f}%**\n"
        f"- Divergência S6 vs MC: **{divergencia:.4f}**\n"
        f"- Índice composto de risco: **{indice_risco:.4f}**\n"
        f"- Classe de risco: {classe_risco}\n\n"
        f"{recomendacao}"
    )

    exibir_bloco_mensagem(
        "Resumo do Monitor de Risco — k & k*",
        corpo,
        tipo="info",
    )

    st.session_state["diagnostico_risco"] = {
        "indice_risco": indice_risco,
        "classe_risco": classe_risco,
        "k_star": k_star,
        "nr_percent": nr_percent,
        "divergencia": divergencia,
    }


    # ============================================================
    # M1 — ESPELHO CANÔNICO DO RISCO (S3)
    # (apenas espelhamento: não recalcula, não decide, não altera motores)
    # ============================================================
    st.session_state["k_star"] = float(k_star) if isinstance(k_star, (int, float)) else k_star
    st.session_state["nr_percent"] = float(nr_percent) if isinstance(nr_percent, (int, float)) else nr_percent

    # Divergência: manter chave histórica (div_s6_mc) e chave canônica do Mirror
    st.session_state["div_s6_mc"] = float(divergencia) if isinstance(divergencia, (int, float)) else divergencia
    st.session_state["divergencia_s6_mc"] = float(divergencia) if isinstance(divergencia, (int, float)) else divergencia

    st.session_state["indice_risco"] = float(indice_risco) if isinstance(indice_risco, (int, float)) else indice_risco
    st.session_state["classe_risco"] = classe_risco

    # Selos M1 (S3)
    st.session_state["m1_selo_risco_ok"] = True
    st.session_state["m1_ts_risco_ok"] = __import__("time").time()

    st.success("Monitor de Risco atualizado com sucesso!")

# ============================================================
# PARTE 5/8 — FIM
# ============================================================
# ============================================================
# PARTE 6/8 — INÍCIO
# ============================================================

# ============================================================
# BLOCO V16 — PROTOCOLO PRÉ-ECO / ECO
# Observador tático — AJUSTA POSTURA PARA A PRÓXIMA SÉRIE
# NÃO prevê, NÃO altera motor, NÃO bloqueia
# ============================================================

def v16_avaliar_pre_eco_eco():
    """
    Usa SOMENTE o estado ATUAL (última série do histórico)
    para definir a postura de ataque da PRÓXIMA série.
    """

    k_star = st.session_state.get("sentinela_kstar")
    nr = st.session_state.get("nr_percent")
    div = st.session_state.get("div_s6_mc")
    risco = (st.session_state.get("diagnostico_risco") or {}).get("indice_risco")

    # Defaults defensivos
    k_star = float(k_star) if isinstance(k_star, (int, float)) else 0.30
    nr = float(nr) if isinstance(nr, (int, float)) else 50.0
    div = float(div) if isinstance(div, (int, float)) else 6.0
    risco = float(risco) if isinstance(risco, (int, float)) else 0.60

    sinais_ok = 0

    if k_star <= 0.30:
        sinais_ok += 1
    if nr <= 45.0:
        sinais_ok += 1
    if div <= 6.0:
        sinais_ok += 1
    if risco <= 0.55:
        sinais_ok += 1

    # Classificação
    if sinais_ok >= 3:
        status = "PRE_ECO_ATIVO"
        postura = "ATIVA"
        comentario = (
            "🟡 PRÉ-ECO detectado — ambiente NÃO piora.\n"
            "Postura ativa para a próxima série.\n"
            "Modo 6 ligado, volume moderado."
        )
    else:
        status = "SEM_ECO"
        postura = "DEFENSIVA"
        comentario = (
            "🔴 Nenhum pré-eco — ambiente instável.\n"
            "Operar apenas com coberturas."
        )

    resultado = {
        "status": status,
        "postura": postura,
        "sinais_ok": sinais_ok,
        "comentario": comentario,
    }

    st.session_state["v16_pre_eco"] = resultado
    return resultado

# ============================================================
# FUNÇÃO — SANIDADE FINAL DAS LISTAS (DISPONÍVEL AO MODO 6)
# Remove listas inválidas, duplicatas e permutações
# Válido para V15.7 MAX e V16 Premium
# ============================================================

def sanidade_final_listas(listas):
    """
    Sanidade final das listas de previsão.
    Regras:
    - Remove listas com números repetidos internamente
    - Remove permutações (ordem diferente, mesmos números)
    - Remove duplicatas exatas
    - Garante apenas listas válidas com 6 números distintos
    """
    if not listas:
        return []

    listas_saneadas = []
    vistos = set()

    for lista in listas:
        try:
            nums = [int(x) for x in lista]
        except Exception:
            continue

        # exatamente 6 números distintos
        if len(nums) != 6 or len(set(nums)) != 6:
            continue

        chave = tuple(sorted(nums))
        if chave in vistos:
            continue

        vistos.add(chave)
        listas_saneadas.append(nums)

    return listas_saneadas

# ============================================================
# FIM — FUNÇÃO SANIDADE FINAL DAS LISTAS
# ============================================================

# ============================================================
# B0 — SANIDADE DE UNIVERSO (V16)
# Observacional + corretivo leve
# Garante que nenhum passageiro fora do universo real apareça
# NÃO altera motores | NÃO decide | NÃO bloqueia
# ============================================================

def v16_sanidade_universo_listas(listas, historico_df):
    """
    Remove / ajusta números fora do universo real observado no histórico.
    Universo é inferido EXCLUSIVAMENTE do histórico carregado.
    """

    if historico_df is None or historico_df.empty:
        return listas  # sem histórico, não mexe

    col_pass = [c for c in historico_df.columns if c.startswith("p")]
    valores = []

    for _, row in historico_df.iterrows():
        for c in col_pass:
            try:
                v = int(row[c])
                if v > 0:
                    valores.append(v)
            except Exception:
                pass

    if not valores:
        return listas

    umin = min(valores)
    umax = max(valores)

    listas_sanas = []

    for lst in listas:
        nova = []
        for v in lst:
            try:
                iv = int(v)
                if iv < umin:
                    iv = umin
                elif iv > umax:
                    iv = umax
                nova.append(iv)
            except Exception:
                pass

        nova = list(dict.fromkeys(nova))  # remove duplicatas mantendo ordem
        listas_sanas.append(nova)

    return listas_sanas


# ============================================================
# >>> INÍCIO — BLOCO DO PAINEL 6 — MODO 6 ACERTOS (PRÉ-ECO)
# ============================================================

if painel == "🎯 Modo 6 Acertos — Execução":

    st.markdown("## 🎯 Modo 6 Acertos — Execução")

    df = st.session_state.get("historico_df")

    # ------------------------------------------------------------
    # k* (fallback seguro)
    # ------------------------------------------------------------
    _kstar_raw = st.session_state.get("sentinela_kstar")
    k_star = float(_kstar_raw) if isinstance(_kstar_raw, (int, float)) else 0.0

    nr_pct = st.session_state.get("nr_percent")
    divergencia_s6_mc = st.session_state.get("div_s6_mc")
    risco_composto = st.session_state.get("indice_risco")
    ultima_prev = st.session_state.get("ultima_previsao")

    # ------------------------------------------------------------
    # GUARDA — CRITÉRIO MÍNIMO (ORIGINAL PRESERVADO)
    # ------------------------------------------------------------
    pipeline_ok = st.session_state.get("pipeline_flex_ultra_concluido") is True

    turbo_executado_ok = any([
        st.session_state.get("turbo_ultra_executado"),
        st.session_state.get("turbo_executado"),
        st.session_state.get("turbo_ultra_rodou"),
        st.session_state.get("motor_turbo_executado"),
    ])

    if df is None or df.empty or not pipeline_ok:
        exibir_bloco_mensagem(
            "Pipeline incompleto",
            "É necessário:\n- Histórico carregado\n- Pipeline V14-FLEX ULTRA executado\n\nℹ️ O TURBO++ é opcional para o Modo 6. Você pode rodar o TURBO antes para tentar núcleo ofensivo, mas o pacote base do Modo 6 independe disso.",
            tipo="warning"
        )
        st.stop()

    # ------------------------------------------------------------
    # AJUSTE DE AMBIENTE (PRÉ-ECO) — ORIGINAL
    # ------------------------------------------------------------
    config = ajustar_ambiente_modo6(
        df=df,
        k_star=k_star,
        nr_pct=nr_pct,
        divergencia_s6_mc=divergencia_s6_mc,
        risco_composto=risco_composto,
        previsibilidade="alta",
    )

    st.caption(config["aviso_curto"] + " | PRÉ-ECO técnico ativo")

    volume = int(config["volume_recomendado"])
    volume = max(1, min(volume, int(config["volume_max"])))


    # ------------------------------------------------------------
    # DETECÇÃO DO FENÔMENO (n + UNIVERSO REAL)
    # ------------------------------------------------------------
    col_pass = [c for c in df.columns if c.startswith("p")]

    universo_tmp = []
    contagens = []

    for _, row in df.iterrows():
        vals = [int(row[c]) for c in col_pass if pd.notna(row[c])]
        if vals:
            contagens.append(len(vals))
            universo_tmp.extend(vals)

    if not contagens or not universo_tmp:
        st.warning("Não foi possível detectar universo válido no histórico.")
        st.stop()

    n_real = int(pd.Series(contagens).mode().iloc[0])
    st.session_state["n_alvo"] = n_real

    universo = sorted({v for v in universo_tmp if v > 0})
    umin, umax = min(universo), max(universo)   # EX: 1–50 (REAL)


    # ------------------------------------------------------------
    # REPRODUTIBILIDADE (ORIGINAL)
    # ------------------------------------------------------------
    seed = abs(hash(f"PC-M6-{len(df)}-{n_real}-{umin}-{umax}")) % (2**32)
    rng = np.random.default_rng(seed)


    # ------------------------------------------------------------
    # MAPA DE ÍNDICES (CORREÇÃO ESTRUTURAL)
    # ------------------------------------------------------------
    universo_idx = list(range(len(universo)))
    valor_por_idx = {i: universo[i] for i in universo_idx}
    idx_por_valor = {v: i for i, v in valor_por_idx.items()}


    def ajustar_para_n(lista):
        out_idx = []

        for v in lista:
            if v in idx_por_valor:
                idx = idx_por_valor[v]
                if idx not in out_idx:
                    out_idx.append(idx)

        while len(out_idx) < n_real:
            cand = rng.choice(universo_idx)
            if cand not in out_idx:
                out_idx.append(cand)

        return out_idx[:n_real]


    # ------------------------------------------------------------
    # BASE ULTRA (ORIGINAL, MAS EM ÍNDICES)
    # ------------------------------------------------------------
    if ultima_prev:
        base_vals = ultima_prev if isinstance(ultima_prev[0], int) else ultima_prev[0]
        base_idx = ajustar_para_n(base_vals)
    else:
        base_idx = rng.choice(universo_idx, size=n_real, replace=False).tolist()


    # ------------------------------------------------------------
    # GERAÇÃO PRÉ-ECO (SEM POSSIBILIDADE DE SAIR DO UNIVERSO)
    # ------------------------------------------------------------
    listas_brutas = []

    for _ in range(volume):
        ruido = rng.integers(-3, 4, size=n_real)  # deslocamento leve
        nova_idx = [
            max(0, min(len(universo_idx) - 1, idx + r))
            for idx, r in zip(base_idx, ruido)
        ]
        nova = [valor_por_idx[i] for i in nova_idx]
        listas_brutas.append(nova)


    # ------------------------------------------------------------
    # 🔒 FILTRO FINAL DE DOMÍNIO (ANTI-RESÍDUO)  ← CORREÇÃO
    # ------------------------------------------------------------
    listas_filtradas = []
    descartadas = 0

    for lista in listas_brutas:
        if all(umin <= int(v) <= umax for v in lista):
            listas_filtradas.append(lista)
        else:
            descartadas += 1

    if descartadas > 0:
        st.warning(
            f"⚠️ {descartadas} lista(s) descartada(s) por violar o domínio "
            f"dos passageiros ({umin}–{umax})."
        )

    listas_brutas = listas_filtradas


    # ------------------------------------------------------------
    # SANIDADE FINAL — SOMENTE ESTRUTURAL (ORIGINAL)
    # ------------------------------------------------------------
    listas_totais = sanidade_final_listas(listas_brutas)

    listas_top10 = listas_totais[:10]

    # ============================================================
    # Órbita (E1) + Gradiente + N_EXTRA
    # (sem interceptação automática; não divide pacote)
    # ============================================================
    try:
        info_orbita = v16_calcular_orbita_pacote(listas_top10, universo_min, universo_max)
        ginfo = v16_calcular_gradiente_E1(info_orbita)
        gradiente = ginfo.get("gradiente", "G0")
        orb_score = ginfo.get("score", 0.0)
    
        # ECO (se existir no estado)
        eco_forca = st.session_state.get("eco_forca", None)
        eco_acion = st.session_state.get("eco_acionabilidade", None)
    
        n_base = len(listas_totais)
    
        # memória para E2 (repetição consecutiva de quase-órbita)
        prev_estado = st.session_state.get("orbita_prev_estado", "E0")
        prev_ancoras = st.session_state.get("orbita_prev_ancoras", [])
        if info_orbita.get("estado") == "E1":
            overlap = 0
            if prev_ancoras and info_orbita.get("ancoras"):
                overlap = len(set(prev_ancoras).intersection(set(info_orbita.get("ancoras"))))
            if prev_estado == "E1" and overlap >= 2:
                info_orbita["estado"] = "E2"
                info_orbita["selo"] = "E2"
    
        st.session_state["orbita_prev_estado"] = info_orbita.get("estado")
        st.session_state["orbita_prev_ancoras"] = info_orbita.get("ancoras", [])
    
        n_extra = v16_calcular_N_extra(info_orbita.get("estado"), gradiente, n_base, eco_forca, eco_acion)
    
        # gera listas extras (se justificável) — não substitui as Top10, só expande
        listas_extra = []
        if n_extra and n_extra > 0:
            listas_extra = v16_gerar_listas_extra_por_orbita(
                info_orbita,
                universo_min=universo_min,
                universo_max=universo_max,
                n_carro=n_real,
                qtd=n_extra,
                seed=st.session_state.get("serie_base_idx", 0),
            )
    
        # listas de interceptação automática (somente em E2) — muda listas de verdade
        listas_intercept = []
        if info_orbita.get("estado") == "E2":
            base_i = 2
            if info_orbita.get("gradiente") in ("G2", "G3"):
                base_i = 3
            qtd_i = min(8, max(2, (n_extra or 0) + base_i))
            listas_intercept = v16_gerar_listas_interceptacao_orbita(
                info_orbita,
                universo_min=universo_min,
                universo_max=universo_max,
                n_carro=n_real,
                qtd=qtd_i,
                seed=st.session_state.get("serie_base_idx", 0),
            )

        if listas_intercept:
            st.session_state["listas_intercept_orbita"] = listas_intercept
            listas_totais = listas_totais + listas_intercept

        if listas_extra:
            listas_totais = listas_totais + listas_extra
            try:
                listas_totais = v16_priorizar_listas_por_contexto(
                    listas_totais,
                    estado_obj=st.session_state.get("estado_alvo_v16"),
                    k_star=st.session_state.get("k_star", None),
                )
            except Exception:
                pass
            listas_top10 = listas_totais[:10]
    
        # registro em sessão (para Relatório Final / Bala Humano)
        st.session_state["orbita_info"] = info_orbita
        st.session_state["orbita_gradiente"] = gradiente
        st.session_state["orbita_score"] = orb_score
        st.session_state["modo6_n_base"] = int(n_base)
        st.session_state["modo6_n_extra"] = int(n_extra)
        st.session_state["modo6_n_total"] = int(len(listas_totais))
    except Exception:
        st.session_state["orbita_info"] = {"estado": "E0", "selo": "E0"}
        st.session_state["orbita_gradiente"] = "G0"
        st.session_state["orbita_score"] = 0.0
        st.session_state["modo6_n_base"] = int(len(listas_totais))
        st.session_state["modo6_n_extra"] = 0
        st.session_state["modo6_n_total"] = int(len(listas_totais))

    st.session_state["modo6_listas_totais"] = listas_totais
    st.session_state["modo6_listas_top10"] = listas_top10
    st.session_state["modo6_listas"] = listas_totais

    # ------------------------------------------------------------
    # REGISTRO AUTOMÁTICO DO PACOTE ATUAL (Backtest Rápido N=60)
    # ------------------------------------------------------------
    # Regra: não decide ação e não muda geração.
    # Apenas "congela" qual pacote está ativo para o painel de Backtest.
    # Preferência: Top10 (priorizadas) quando existir; senão, usa o total.
    try:
        _pacote_bt = listas_top10 if (isinstance(listas_top10, list) and len(listas_top10) > 0) else listas_totais
        st.session_state["pacote_listas_atual"] = _pacote_bt
        st.session_state["pacote_listas_origem"] = "Modo 6 (Top10)" if _pacote_bt is listas_top10 else "Modo 6 (Total)"
    except Exception:
        # Falha silenciosa: não deve travar a execução do Modo 6.
        pass


    st.success(
        f"Modo 6 (PRÉ-ECO | n-base={n_real}) — "
        f"{len(listas_totais)} listas totais | "
        f"{len(listas_top10)} priorizadas (Top 10)."
    )
# ============================================================
# <<< FIM — BLOCO DO PAINEL 6 — MODO 6 ACERTOS (PRÉ-ECO)
# ============================================================


    # ✅ Snapshot canônico (para Relatório Final / Diagnóstico Espelho)
    try:
        st.session_state["modo6_executado"] = True
        st.session_state["listas_geradas"] = int(len(listas_top10) if isinstance(listas_top10, list) else len(listas_totais))
    except Exception:
        pass




# ============================================================
# 🧪 Modo N Experimental (n≠6)
# (LAUDO DE CÓDIGO — FASE 1 / BLOCO 2)
#
# OBJETIVO:
# - Roteamento mínimo + guardas explícitas
# - Avisos claros de EXPERIMENTAL
# - ZERO lógica de geração
#
# BLINDAGEM:
# - NÃO altera Modo 6
# - NÃO altera TURBO
# - NÃO altera ECO/PRÉ-ECO
# - NÃO escreve em session_state (somente leitura)
# ============================================================

elif painel == "🧪 Modo N Experimental (n≠6)":

    st.header("🧪 Modo N Experimental (n≠6)")
    st.warning(
        "EXPERIMENTAL — Este painel é isolado. "
        "Não substitui o Modo 6, não altera TURBO, "
        "não aprende e pode recusar geração."
    )

    # ------------------------------
    # Guardas canônicas (EVIDÊNCIA REAL)
    # ------------------------------
    historico_df = st.session_state.get("historico_df")
    n_alvo = st.session_state.get("n_alvo")
    k_calculado = st.session_state.get("k_calculado") or st.session_state.get("k_star")

    # Evidências indiretas do pipeline (como ele REALMENTE funciona)
    estrada_regime = st.session_state.get("estrada_regime")
    energia_media = st.session_state.get("energia_media")
    clusters_formados = st.session_state.get("clusters_formados")

    # Guarda 1 — histórico
    if historico_df is None or historico_df.empty:
        st.error("Pré-requisito ausente: histórico não carregado.")
        st.stop()

    # Guarda 2 — n_alvo válido e diferente de 6
    try:
        n_int = int(n_alvo)
    except Exception:
        st.error("Pré-requisito ausente: n_alvo inválido.")
        st.stop()

    if n_int == 6:
        st.info("Este painel é exclusivo para n≠6. Para n=6, utilize o Modo 6.")
        st.stop()

    # Guarda 3 — pipeline (por evidência observada)
    if estrada_regime is None and energia_media is None and clusters_formados is None:
        st.error("Pré-requisito ausente: Pipeline V14-FLEX ULTRA não concluído.")
        st.stop()

    # Guarda 4 — sentinelas
    if k_calculado is None:
        st.error("Pré-requisito ausente: Sentinelas (k/k*) não calculadas.")
        st.stop()

    # ------------------------------
    # Estado observado (laudo)
    # ------------------------------
    st.subheader("📋 Estado Observado (Laudo)")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("n do Fenômeno", n_int)
    with col2:
        st.metric("Séries", len(historico_df))
    with col3:
        st.metric("Pipeline", "CONCLUÍDO (por evidência)")
    with col4:
        st.metric(
            "Sentinela",
            f"{k_calculado:.4f}" if isinstance(k_calculado, (int, float)) else str(k_calculado),
        )

    st.markdown("---")

    st.info(
        "Este é o **BLOCO 2 (Laudo de Código)**.\n\n"
        "➡️ Nenhuma lista é gerada aqui.\n"
        "➡️ O próximo bloco ativará a lógica EXPERIMENTAL de geração, "
        "usando U2/U3/U4 como autorizadores."
    )



# ============================================================
# 🧪 Modo N Experimental (n≠6)
# BLOCO 3 — GERAÇÃO MÍNIMA EXPERIMENTAL (n=5)
# ============================================================

    st.markdown("### 🔬 Geração Experimental — n≠6")
    st.caption(
        "Modo EXPERIMENTAL. Geração mínima, consciente e auditável. "
        "Não substitui o Modo 6."
    )

    # ------------------------------------------------------------
    # Fonte canônica do pacote (somente leitura)
    # ------------------------------------------------------------
    listas_base = st.session_state.get("modo6_listas_totais", [])

    # ------------------------------------------------------------
    # Autorizadores (MVP-U2 / U3 / U4)
    # ------------------------------------------------------------
    autorizacao = {
        "orcamento_ok": False,
        "cobertura_ok": False,
        "eficiencia_ok": False,
    }

    # Autorização mínima por orçamento (U2)
    orcamento_manual = st.session_state.get("orcamento_manual_universal")
    if isinstance(orcamento_manual, (int, float)) and orcamento_manual > 0:
        autorizacao["orcamento_ok"] = True

    # Autorização mínima por cobertura (U3)
    if listas_base and len(listas_base) >= 1:
        autorizacao["cobertura_ok"] = True

    # Autorização mínima por eficiência (U4)
    # (critério mínimo: ao menos 1 lista viável)
    if autorizacao["orcamento_ok"] and autorizacao["cobertura_ok"]:
        autorizacao["eficiencia_ok"] = True

    # ------------------------------------------------------------
    # Decisão EXPERIMENTAL (sem fallback)
    # ------------------------------------------------------------
    if not all(autorizacao.values()):
        st.warning(
            "Geração NÃO autorizada pelos MVPs (U2/U3/U4).\n\n"
            "➡️ Resultado válido.\n"
            "➡️ Nenhuma lista foi gerada."
        )
    else:
        # --------------------------------------------------------
        # Geração mínima (1 a 3 listas) — n-base
        # --------------------------------------------------------
        max_listas = min(3, len(listas_base))
        listas_n = [sorted(lst)[:n_int] for lst in listas_base[:max_listas]]

        st.success(f"Geração EXPERIMENTAL autorizada — {len(listas_n)} lista(s).")

        for i, lst in enumerate(listas_n, start=1):
            st.code(f"Lista N{i}: {lst}", language="python")

        # --------------------------------------------------------
        # Mini-laudo automático
        # --------------------------------------------------------
        st.markdown("#### 📄 Mini-Laudo (Automático)")
        st.write(
            {
                "modo": "Modo N Experimental",
                "n": n_int,
                "listas_geradas": len(listas_n),
                "orcamento_manual": orcamento_manual,
                "regime": "OBSERVADO",
                "status": "GERADO" if listas_n else "RECUSADO",
            }
        )



# ============================================================
# 📊 V16 PREMIUM — MVP-U2 | ORÇAMENTO UNIVERSAL (OBSERVACIONAL)
# ============================================================
if painel == "📊 V16 Premium — Orçamento Universal":

    st.title("📊 MVP-U2 — Orçamento Universal (Observacional)")
    st.caption(
        "Observacional • Não gera listas • Não decide\n"
        "Avalia custo real dos pacotes já gerados (Modo 6 / Universal)."
    )

    listas = st.session_state.get("modo6_listas_totais", [])
    n_alvo = st.session_state.get("n_alvo")

    if not listas or n_alvo is None:
        st.warning(
            "Pacote indisponível.\n\n"
            "Execute primeiro:\n"
            "• Pipeline\n"
            "• Modo 6 (Painel 11)"
        )
        st.stop()

    st.markdown("---")

    # --------------------------------------------------------
    # TABELA DE CUSTO UNIVERSAL (CANÔNICA)
    # --------------------------------------------------------
    TABELA_CUSTO = {
        5:  {5: 3,   6: 18,   7: 63,   8: 168,   9: 378,   10: 756},
        6:  {6: 6,   7: 42,   8: 168,  9: 504,   10: 1260, 11: 2772},
        15: {15: 3.5, 16: 56, 17: 476},
    }

    st.markdown("### 📐 Tabela canônica de custo (fixa)")
    st.json(TABELA_CUSTO)

    st.markdown("---")

    # --------------------------------------------------------
    # Entrada de orçamento manual (opcional)
    # --------------------------------------------------------
    orcamento_manual = st.number_input(
        "Orçamento manual (opcional)",
        min_value=0.0,
        value=0.0,
        step=1.0,
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Avaliação das listas
    # --------------------------------------------------------
    linhas = []

    for i, lista in enumerate(listas, start=1):
        tamanho = len(lista)

        custo = None
        if n_alvo in TABELA_CUSTO and tamanho in TABELA_CUSTO[n_alvo]:
            custo = TABELA_CUSTO[n_alvo][tamanho]

        linhas.append({
            "lista_id": i,
            "n_lista": tamanho,
            "custo_estimado": custo,
            "cabe_no_orcamento_manual": (
                custo is not None and orcamento_manual > 0 and custo <= orcamento_manual
            ),
        })

    df_orc = pd.DataFrame(linhas)

    st.markdown("### 📊 Avaliação observacional do pacote")
    st.dataframe(df_orc, use_container_width=True, hide_index=True)

    st.markdown(
        """
🧠 **Leitura correta**
- Custo **None** = combinação não prevista na tabela
- Painel **não filtra**, **não decide**, **não prioriza**
- Serve apenas para **decisão HUMANA**
"""
    )

# ============================================================
# MVP-U3 — COBERTURA UNIVERSAL (OBSERVACIONAL)
# NÃO GERA LISTAS • NÃO DECIDE • NÃO ALTERA MOTOR
# ============================================================
if painel == "🧩 MVP-U3 — Cobertura Universal":

    st.markdown("## 🧩 MVP-U3 — Cobertura Universal (Observacional)")
    st.caption(
        "Avalia cobertura, redundância e custo teórico do pacote ATUAL.\n"
        "Funciona para qualquer n_alvo (5, 6, 15, etc.).\n"
        "❌ Não gera listas • ❌ Não decide • ✅ Apenas mede"
    )

    # ------------------------------------------------------------
    # Recuperação segura do histórico
    # ------------------------------------------------------------
    historico_df = st.session_state.get("historico_df")
    if historico_df is None or historico_df.empty:
        st.warning("Histórico não encontrado. Carregue o histórico antes.")
        st.stop()

    # ------------------------------------------------------------
    # Recuperação do pacote congelado
    # ------------------------------------------------------------
    pacote = (
        st.session_state.get("pacote_listas_atual")
        or st.session_state.get("modo6_listas_totais")
    )

    if not pacote:
        st.warning("Nenhum pacote de listas disponível para avaliação.")
        st.stop()

    # ------------------------------------------------------------
    # Detecção canônica de n_alvo
    # ------------------------------------------------------------
    n_alvo = st.session_state.get("n_alvo")
    if not n_alvo or n_alvo <= 0:
        st.warning("n_alvo não detectado. Execute o carregamento do histórico.")
        st.stop()

    # ------------------------------------------------------------
    # Universo real observado no histórico
    # ------------------------------------------------------------
    col_pass = [c for c in historico_df.columns if c.startswith("p")]
    universo = sorted(
        {
            int(v)
            for _, row in historico_df.iterrows()
            for v in row[col_pass]
            if pd.notna(v) and int(v) > 0
        }
    )

    if not universo:
        st.warning("Universo vazio. Histórico inválido.")
        st.stop()

    # ------------------------------------------------------------
    # Métricas de cobertura
    # ------------------------------------------------------------
    total_listas = len(pacote)

    tamanhos = [len(set(lst)) for lst in pacote]
    validas = [lst for lst in pacote if len(set(lst)) >= n_alvo]

    cobertura_unica = set()
    for lst in validas:
        cobertura_unica.update(lst)

    taxa_validas = len(validas) / total_listas if total_listas else 0.0
    cobertura_pct = (
        len(cobertura_unica) / len(universo) * 100 if universo else 0.0
    )

    # Redundância média
    freq = {}
    for lst in validas:
        for x in lst:
            freq[x] = freq.get(x, 0) + 1

    redundancia_media = (
        sum(freq.values()) / len(freq) if freq else 0.0
    )

    # ------------------------------------------------------------
    # Exibição — Métricas principais
    # ------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Listas totais", total_listas)
    col2.metric("Listas ≥ n_alvo", f"{len(validas)} ({taxa_validas*100:.1f}%)")
    col3.metric("Cobertura do universo", f"{cobertura_pct:.1f}%")
    col4.metric("Redundância média", f"{redundancia_media:.2f}")

    # ------------------------------------------------------------
    # Diagnóstico textual (OBSERVACIONAL)
    # ------------------------------------------------------------
    st.markdown("### 🧠 Leitura observacional")

    if taxa_validas < 0.6:
        st.warning(
            "Poucas listas atingem o tamanho mínimo do fenômeno.\n"
            "Cobertura estrutural fraca."
        )
    elif cobertura_pct < 40:
        st.warning(
            "Cobertura baixa do universo observado.\n"
            "Pacote concentrado demais."
        )
    else:
        st.success(
            "Cobertura estrutural aceitável para o fenômeno atual.\n"
            "Pacote coerente sob critério universal."
        )

    st.info(
        "📌 Este painel NÃO decide execução.\n"
        "Use apenas como régua de cobertura e redundância."
    )

# ============================================================
# <<< FIM — MVP-U3 — COBERTURA UNIVERSAL
# ============================================================

# ============================================================
# MVP-U4 — EFICIÊNCIA MARGINAL POR CUSTO (OBSERVACIONAL)
# NÃO GERA LISTAS • NÃO DECIDE • NÃO ALTERA MOTOR
# ============================================================
if painel == "📈 MVP-U4 — Eficiência Marginal por Custo":

    st.markdown("## 📈 MVP-U4 — Eficiência Marginal por Custo (Observacional)")
    st.caption(
        "Avalia quanto de cobertura adicional é obtida por unidade extra de orçamento.\n"
        "Depende de U2 (Orçamento) e U3 (Cobertura).\n"
        "❌ Não gera listas • ❌ Não decide • ✅ Apenas mede"
    )

    # ------------------------------------------------------------
    # Recuperação do histórico e n_alvo
    # ------------------------------------------------------------
    historico_df = st.session_state.get("historico_df")
    n_alvo = st.session_state.get("n_alvo")

    if historico_df is None or historico_df.empty or not n_alvo:
        st.warning("Histórico ou n_alvo indisponível. Carregue o histórico primeiro.")
        st.stop()

    # ------------------------------------------------------------
    # Recuperação do pacote congelado
    # ------------------------------------------------------------
    pacote = (
        st.session_state.get("pacote_listas_atual")
        or st.session_state.get("modo6_listas_totais")
    )

    if not pacote:
        st.warning("Nenhum pacote disponível para análise.")
        st.stop()

    # ------------------------------------------------------------
    # Universo real observado
    # ------------------------------------------------------------
    col_pass = [c for c in historico_df.columns if c.startswith("p")]
    universo = sorted(
        {
            int(v)
            for _, row in historico_df.iterrows()
            for v in row[col_pass]
            if pd.notna(v) and int(v) > 0
        }
    )

    if not universo:
        st.warning("Universo vazio. Histórico inválido.")
        st.stop()

    universo_size = len(universo)

    # ------------------------------------------------------------
    # Tabela canônica de custo (mesma do MVP-U2)
    # ------------------------------------------------------------
    TABELA_CUSTO = {
        5:  {5: 3,   6: 18,   7: 63,   8: 168,   9: 378,   10: 756},
        6:  {6: 6,   7: 42,   8: 168,  9: 504,   10: 1260, 11: 2772},
        15: {15: 3.5, 16: 56, 17: 476},
    }

    # ------------------------------------------------------------
    # Agrupamento por tamanho de lista (≥ n_alvo)
    # ------------------------------------------------------------
    grupos = {}
    for lst in pacote:
        if len(set(lst)) >= n_alvo:
            k = len(set(lst))
            grupos.setdefault(k, []).append(lst)

    if not grupos:
        st.warning("Nenhuma lista válida (≥ n_alvo) encontrada.")
        st.stop()

    # ------------------------------------------------------------
    # Cálculo de cobertura por grupo
    # ------------------------------------------------------------
    linhas = []

    for tamanho, listas in sorted(grupos.items()):
        cobertura = set()
        for lst in listas:
            cobertura.update(lst)

        cobertura_pct = len(cobertura) / universo_size * 100

        custo = None
        if n_alvo in TABELA_CUSTO and tamanho in TABELA_CUSTO[n_alvo]:
            custo = TABELA_CUSTO[n_alvo][tamanho]

        linhas.append({
            "n_lista": tamanho,
            "cobertura_pct": cobertura_pct,
            "custo": custo,
        })

    df = pd.DataFrame(linhas).sort_values("n_lista").reset_index(drop=True)

    if df.empty:
        st.warning("Não foi possível calcular métricas.")
        st.stop()

    # ------------------------------------------------------------
    # Base = menor tamanho válido
    # ------------------------------------------------------------
    base = df.iloc[0]
    base_cob = base["cobertura_pct"]
    base_custo = base["custo"]

    # ------------------------------------------------------------
    # Eficiência marginal
    # ------------------------------------------------------------
    em_linhas = []
    for _, row in df.iterrows():
        if row["custo"] is None or base_custo is None or row["custo"] == base_custo:
            em = None
            dc = None
            dd = None
        else:
            dc = row["custo"] - base_custo
            dd = row["cobertura_pct"] - base_cob
            em = dd / dc if dc > 0 else None

        em_linhas.append({
            "n_lista": row["n_lista"],
            "cobertura_pct": round(row["cobertura_pct"], 2),
            "custo": row["custo"],
            "Δcobertura": round(dd, 2) if dd is not None else None,
            "Δcusto": dc,
            "eficiencia_marginal": round(em, 4) if em is not None else None,
        })

    df_em = pd.DataFrame(em_linhas)

    # ------------------------------------------------------------
    # Exibição
    # ------------------------------------------------------------
    st.markdown("### 📊 Eficiência marginal por tamanho de lista")
    st.dataframe(df_em, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------
    # Leitura observacional
    # ------------------------------------------------------------
    st.markdown("### 🧠 Leitura observacional")

    valid_em = df_em.dropna(subset=["eficiencia_marginal"])
    if valid_em.empty:
        st.info("Eficiência marginal não disponível para comparação.")
    else:
        melhor = valid_em.sort_values("eficiencia_marginal", ascending=False).iloc[0]
        st.success(
            f"Maior eficiência marginal em n_lista = {int(melhor['n_lista'])} "
            f"(EM = {melhor['eficiencia_marginal']})."
        )

        baixos = valid_em[valid_em["eficiencia_marginal"] < 0.01]
        if not baixos.empty:
            st.warning(
                "Retorno decrescente detectado em alguns tamanhos:\n"
                + ", ".join(str(int(x)) for x in baixos["n_lista"].tolist())
            )

    st.info(
        "📌 Este painel é apenas observacional.\n"
        "Use para decidir até onde vale a pena aumentar o orçamento."
    )

# ============================================================
# <<< FIM — MVP-U4 — EFICIÊNCIA MARGINAL POR CUSTO
# ============================================================


# ============================================================
# Painel 12 — 🧪 Testes de Confiabilidade REAL
# ============================================================
if painel == "🧪 Testes de Confiabilidade REAL":

    st.markdown("## 🧪 Testes de Confiabilidade REAL — V15.7 MAX")

    df = st.session_state.get("historico_df")
    listas_m6 = st.session_state.get("modo6_listas")
    ultima_prev = st.session_state.get("ultima_previsao")

    if df is None or listas_m6 is None or ultima_prev is None:
        exibir_bloco_mensagem(
            "Pré-requisitos não atendidos",
            "Execute o pipeline até o Modo 6 Acertos.",
            tipo="warning",
        )
        st.stop()

    qtd_series = len(df)
    if qtd_series < 15:
        exibir_bloco_mensagem(
            "Histórico insuficiente",
            "São necessárias pelo menos 15 séries para validar a confiabilidade.",
            tipo="warning",
        )
        st.stop()

    st.info("Executando avaliação REAL de confiabilidade...")

    col_pass = [c for c in df.columns if c.startswith("p")]

    # Janela de teste recente
    janela = df[col_pass].iloc[-12:].values

    # ============================================================
    # Medição de acertos reais
    # ============================================================
    def acertos(lista, alvo):
        return len(set(lista) & set(alvo))

    acertos_nucleo = []
    acertos_coberturas = []

    for alvo in janela:
        # núcleo
        ac_nuc = acertos(ultima_prev, alvo)
        acertos_nucleo.append(ac_nuc)

        # coberturas
        max_cov = 0
        for lst in listas_m6:
            ac_lst = acertos(lst, alvo)
            if ac_lst > max_cov:
                max_cov = ac_lst
        acertos_coberturas.append(max_cov)

    # Médias
    media_nucleo = float(np.mean(acertos_nucleo))
    media_cob = float(np.mean(acertos_coberturas))

    # ============================================================
    # Diagnóstico
    # ============================================================
    corpo = (
        f"- Janela avaliada: **12 séries recentes**\n"
        f"- Média de acertos do Núcleo: **{media_nucleo:.2f}**\n"
        f"- Média de acertos das Coberturas: **{media_cob:.2f}**\n"
        "\n"
        "Coberturas devem superar o núcleo em ambientes turbulentos.\n"
        "Se o núcleo supera as coberturas, o ambiente está mais limpo."
    )

    exibir_bloco_mensagem(
        "Resumo da Confiabilidade REAL",
        corpo,
        tipo="info",
    )

    st.success("Teste de Confiabilidade REAL concluído com sucesso!")

# ============================================================
# BLOCO — SANIDADE FINAL DAS LISTAS DE PREVISÃO
# (Elimina permutações, duplicatas por conjunto
#  E listas com números repetidos internos)
# Válido para V15.7 MAX e V16 Premium
# ============================================================

def sanidade_final_listas(listas):
    """
    Sanidade final das listas de previsão.
    Regras:
    - Remove listas com números repetidos internamente
    - Remove permutações (ordem diferente, mesmos números)
    - Remove duplicatas exatas
    - Garante apenas listas válidas com 6 números distintos
    """

    listas_saneadas = []
    vistos = set()

    for lista in listas:
        try:
            nums = [int(x) for x in lista]
        except Exception:
            continue

        # 🔒 REGRA CRÍTICA — exatamente 6 números distintos
        if len(nums) != 6:
            continue

        if len(set(nums)) != 6:
            # Exemplo eliminado: [11, 12, 32, 32, 37, 42]
            continue

        # Normaliza ordem para detectar permutações
        chave = tuple(sorted(nums))

        if chave in vistos:
            continue

        vistos.add(chave)
        listas_saneadas.append(nums)

    return listas_saneadas


# ============================================================
# APLICAÇÃO AUTOMÁTICA DA SANIDADE (SE LISTAS EXISTIREM)
# ============================================================

# Sanear listas do Modo 6 (V15.7)
if "modo6_listas" in st.session_state:
    st.session_state["modo6_listas"] = sanidade_final_listas(
        st.session_state.get("modo6_listas", []),
    )

# Sanear Execução V16 (se existir)
if "v16_execucao" in st.session_state:
    exec_v16 = st.session_state.get("v16_execucao", {})

    for chave in ["C2", "C3", "todas_listas"]:
        if chave in exec_v16:
            exec_v16[chave] = sanidade_final_listas(
                exec_v16.get(chave, []),
            )

    st.session_state["v16_execucao"] = exec_v16

# ============================================================
# PARTE 6/8 — FIM
# ============================================================



# ============================================================
# PARTE 7/8 — INÍCIO
# ============================================================

# ============================================================
# Painel — 🧪 Replay Curto — Expectativa 1–3 Séries (V16)
# Diagnóstico apenas | NÃO gera previsões | NÃO altera fluxo
# ============================================================
if painel == "🧪 Replay Curto — Expectativa 1–3 Séries":

    st.markdown("## 🧪 Replay Curto — Expectativa 1–3 Séries (Diagnóstico)")
    st.caption(
        "Validação no passado da expectativa de curto prazo (1–3 séries). "
        "Este painel **não prevê números** e **não altera decisões**."
    )

    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")

    if df is None or matriz_norm is None:
        exibir_bloco_mensagem(
            "Pipeline incompleto",
            "Execute **Carregar Histórico** e **Pipeline V14-FLEX ULTRA**.",
            tipo="warning",
        )
        st.stop()

    # -------------------------------
    # Parâmetros FIXOS (sem bifurcação)
    # -------------------------------
    JANELA_REPLAY = 80       # pontos do passado
    HORIZONTE = 3            # 1–3 séries
    LIMIAR_NR = 0.02         # queda mínima de NR% para considerar melhora
    LIMIAR_DIV = 0.50        # queda mínima de divergência para considerar melhora

    n = len(df)
    if n < JANELA_REPLAY + HORIZONTE + 5:
        exibir_bloco_mensagem(
            "Histórico insuficiente",
            "É necessário mais histórico para o replay curto.",
            tipo="warning",
        )
        st.stop()

    # -------------------------------
    # Helpers locais (diagnóstico)
    # -------------------------------
    col_pass = [c for c in df.columns if c.startswith("p")]

    def calc_nr_local(matriz):
        # NR% aproximado (mesma lógica do painel, versão local)
        variancias = np.var(matriz, axis=1)
        ruido_A = float(np.mean(variancias))
        saltos = []
        for i in range(1, len(matriz)):
            saltos.append(np.linalg.norm(matriz[i] - matriz[i - 1]))
        ruido_B = float(np.mean(saltos)) if saltos else 0.0
        return (0.55 * min(1.0, ruido_A / 0.08) + 0.45 * min(1.0, ruido_B / 1.20))

    def calc_div_local(base, candidatos):
        return float(np.linalg.norm(np.mean(candidatos, axis=0) - base))

    def estado_sinal(nr_deriv, div_deriv, vel):
        # 🟢 melhora curta
        if nr_deriv < -LIMIAR_NR and div_deriv < -LIMIAR_DIV and vel < 0.75:
            return "🟢 Melhora curta"
        # 🔴 continuidade ruim
        if nr_deriv > 0 or div_deriv > 0 or vel >= 0.80:
            return "🔴 Continuidade ruim"
        # 🟡 transição
        return "🟡 Respiração / Transição"

    # -------------------------------
    # Replay
    # -------------------------------
    resultados = []
    base_ini = n - JANELA_REPLAY - HORIZONTE

    for i in range(base_ini, n - HORIZONTE):
        # Janela até o ponto i
        matriz_i = matriz_norm[: i + 1]
        nr_i = calc_nr_local(matriz_i)

        # Divergência local (proxy simples)
        base = matriz_i[-1]
        candidatos = matriz_i[-10:] if len(matriz_i) >= 10 else matriz_i
        div_i = calc_div_local(base, candidatos)

        # Velocidade (proxy simples)
        vel = float(np.mean(np.std(matriz_i[-5:], axis=1)))

        # Próximo trecho (1–3)
        matriz_f = matriz_norm[: i + 1 + HORIZONTE]
        nr_f = calc_nr_local(matriz_f)
        base_f = matriz_f[-1]
        candidatos_f = matriz_f[-10:] if len(matriz_f) >= 10 else matriz_f
        div_f = calc_div_local(base_f, candidatos_f)

        nr_deriv = nr_f - nr_i
        div_deriv = div_f - div_i

        estado = estado_sinal(nr_deriv, div_deriv, vel)

        melhora_real = (nr_deriv < -LIMIAR_NR) or (div_deriv < -LIMIAR_DIV)

        resultados.append({
            "estado": estado,
            "melhora_real": melhora_real
        })

    # -------------------------------
    # Consolidação
    # -------------------------------
    df_res = pd.DataFrame(resultados)
    resumo = (
        df_res.groupby("estado")["melhora_real"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={
            "count": "Ocorrências",
            "mean": "Taxa de Melhora"
        })
    )
    resumo["Taxa de Melhora"] = (resumo["Taxa de Melhora"] * 100).round(1)

    st.markdown("### 📊 Resultado do Replay Curto (passado)")
    st.dataframe(resumo, use_container_width=True)

    st.info(
        "Este painel valida **se o estado 🟢 precede melhora real** no curto prazo "
        "(1–3 séries) **mais vezes que o acaso**. "
        "Ele **não prevê o futuro**, apenas qualifica a expectativa."
    )

# ============================================================
# CAMADA B1 — PRIORIZAÇÃO OBSERVACIONAL DE LISTAS (V16)
# NÃO gera | NÃO decide | NÃO altera motores
# ============================================================

def v16_priorizar_listas_por_contexto(listas):
    """
    Ordena listas existentes usando ECO + Estado.
    Apenas PRIORIZA — não remove, não cria, não decide.
    """

    diag = st.session_state.get("diagnostico_eco_estado_v16", {})
    eco_acion = diag.get("eco_acionabilidade", "não_acionável")
    estado = diag.get("estado", "indefinido")

    if not listas or not isinstance(listas, list):
        return listas

    def score_lista(lst):
        score = 0

        # Preferência leve quando ambiente é favorável
        if eco_acion == "favorável":
            score += 2
        elif eco_acion == "cautela":
            score += 1

        # Penalização leve em movimento brusco
        if estado == "movimento_brusco":
            score -= 1

        # Listas mais compactas tendem a ser mais estáveis
        try:
            if len(set(lst)) <= len(lst):
                score += 1
        except Exception:
            pass

        return score

    try:
        listas_ordenadas = sorted(listas, key=score_lista, reverse=True)
        return listas_ordenadas
    except Exception:
        return listas

# ============================================================
# >>> PAINEL X — 🧠 Memória Operacional — Observacional
# ============================================================
if painel == "🧠 Memória Operacional — Observacional":
    st.markdown("## 🧠 Memória Operacional (Observacional)")
    st.caption("Este painel é um espelho: mostra registros já existentes. Não pede confirmação do operador para registros automáticos.")

    # Garantir estrutura mínima
    if "memoria_operacional" not in st.session_state or st.session_state["memoria_operacional"] is None:
        st.session_state["memoria_operacional"] = []

    registros = st.session_state["memoria_operacional"]

    if len(registros) == 0:
        st.info("Sem registros na Memória Operacional nesta sessão. (Isso não é erro.)")
        st.caption("📌 Observação: o M5 — Pulo do Gato registra automaticamente 'fotos' na Memória de Estados (M2). Para massa histórica, use 🧠 Memória de Estados (M2) e o 📈 M3.")
    else:
        st.success(f"{len(registros)} registro(s) nesta sessão.")
        # Exibição simples e segura (sem botões)
        for i, r in enumerate(registros[-50:], start=max(1, len(registros)-49)):
            st.markdown(f"**{i:02d})** `{r}`")
if painel == "🧠 Memória Operacional — Registro Semi-Automático":
    st.markdown("## 🧠 Memória Operacional — Registro Semi-Automático (Passivo)")
    st.caption("Mantido por compatibilidade de navegação. Operação passiva (sem botões). Use o painel 🧠 Memória Operacional para ver registros.")

    if "memoria_operacional" not in st.session_state or st.session_state["memoria_operacional"] is None:
        st.session_state["memoria_operacional"] = []

    if len(st.session_state["memoria_operacional"]) == 0:
        st.info("Sem registros nesta sessão.")
    else:
        st.success(f"Registros nesta sessão: {len(st.session_state['memoria_operacional'])}")
if painel == "📘 Relatório Final":

    st.markdown("## 📘 Relatório Final — V15.7 MAX — V16 Premium Profundo")

    # Sincroniza chaves canônicas (ECO/Estado/k*/Divergência) antes de consolidar
    v16_sync_aliases_canonicos()

    # ------------------------------------

    # ------------------------------------------------------------
    # 👁️ CAMADA 3 — Cegueiras ainda possíveis (hipóteses)
    # (Somente no RF: não cria sensores, não decide nada)
    # ------------------------------------------------------------
    try:
        m3_reg = st.session_state.get("m3_regime_dx") or st.session_state.get("m3_regime") or "N/D"
        nrp = st.session_state.get("nr_percent")
        divv = st.session_state.get("divergencia_s6_mc")
        cls_r = st.session_state.get("classe_risco") or "N/D"

        # tenta reaproveitar o diagnóstico da Camada 2 (se existir no escopo)
        rigido_flag = False
        try:
            rigido_flag = bool(locals().get("diag_j", {}).get("rigido"))
        except Exception:
            rigido_flag = False

        linhas = []
        linhas.append("🎛️ **Instrumento vs fenômeno:** a leitura pode estar limitada pela lente (ruído/divergência), não só pelo mundo.")
        linhas.append("🧱 **Compressão ≠ erro:** pacote estreito pode ser regime neutro/estreito real — não necessariamente rigidez ruim.")

        if str(m3_reg).upper() == "RUIM":
            linhas.append("🌫️ **RUIM com frestas:** RUIM pode ter micro‑aberturas locais (curtas) que não viram ECO/PRÉ‑ECO no agregado.")
        if rigido_flag:
            linhas.append("🧩 **Perda por borda:** jeitão pode estar correto, mas 1–2 passageiros de borda podem ficar fora quando o pacote fica colado.")
            linhas.append("⚠️ **Rigidez detectada:** hipótese ativa de perda por compressão excessiva (sinal p/ governança/cobertura).")

        # ausência de anti-âncora (se RF tiver essa info)
        try:
            if not st.session_state.get("anti_ancora_idx_detectados"):
                linhas.append("🧲 **Anti‑âncora ausente:** pode ser E0 real OU pouca amplitude do pacote (poucas listas / pouca variação).")
        except Exception:
            pass

        try:
            if isinstance(nrp, (int, float)) and nrp >= 50:
                linhas.append("🔴 **NR crítico:** ruído alto pode achatar leitura fina e mascarar sinal fraco; cuidado extra com 'miragem'.")
        except Exception:
            pass
        try:
            if isinstance(divv, (int, float)) and divv >= 3:
                linhas.append("🟡 **Divergência moderada/alta:** modelos discordando pode ocultar padrão local; trate como hipótese, não permissão de ataque.")
        except Exception:
            pass
        if "🟠" in str(cls_r) or "Elevado" in str(cls_r) or "🔴" in str(cls_r):
            linhas.append("🛑 **Risco elevado:** mesmo com estrada neutra, turbulência pode exigir postura de cobertura (não de invenção).")

        st.markdown("### 👁️ Camada 3 — Cegueiras ainda possíveis (hipóteses)")
        st.caption("Este bloco **NÃO cria sensores novos** e **NÃO decide nada**. Ele lista hipóteses de cegueira ainda possíveis (para não confundir fresta com miragem).")
        for ln in linhas:
            st.markdown(f"- {ln}")

        st.caption("Regra canônica: **mapa de hipóteses**, não motor. Mantém pressão evolutiva sem transformar leitura em fé.")
    except Exception:
        # falha silenciosa (não derruba o RF)
        pass
# ------------------------
    # 🧭 BLOCO -1 — SUMÁRIO EXECUTIVO (read-only)
    # ------------------------------------------------------------
    try:
        _snap = _m1_collect_mirror_snapshot() if '_m1_collect_mirror_snapshot' in globals() else {}
        _estado = _m1_classificar_estado(_snap) if '_m1_classificar_estado' in globals() else {'estado':'S0','avisos':[],'snapshot':_snap}
        st.markdown('### 🧭 Sumário Executivo (rodada atual)')
        # --- Regime por fonte (consolidação) ---
        st.markdown('### 🧷 Regime por fonte (consolidação)')
        reg_pipeline = st.session_state.get('pipeline_estrada', None)
        reg_global = st.session_state.get('regime', None)
        reg_m3 = st.session_state.get('m3_regime_dx', None)
        classe_risco = st.session_state.get('classe_risco', None)
        k_star = st.session_state.get('k_star', None)
        nr = st.session_state.get('nr_percent', None)
        div_s6_mc = st.session_state.get('divergencia_s6_mc', None)
        colA, colB, colC = st.columns(3)
        with colA:
            st.markdown('**🛣️ Pipeline (Estrada)**')
            st.write(reg_pipeline if reg_pipeline is not None else '—')
            st.caption('Regime global atual: {}'.format(reg_global) if reg_global is not None else 'Regime global atual: —')
        with colB:
            st.markdown('**🛰️ Sentinelas / Risco**')
            st.write('Classe: {}'.format(classe_risco) if classe_risco is not None else 'Classe: —')
            st.write('k*: {:.4f}'.format(k_star) if isinstance(k_star, (int, float)) else ('k*: {}'.format(k_star) if k_star is not None else 'k*: —'))
            st.write('NR%: {:.2f}%'.format(nr) if isinstance(nr, (int, float)) else ('NR%: {}'.format(nr) if nr is not None else 'NR%: —'))
            st.write('Div S6×MC: {:.4f}'.format(div_s6_mc) if isinstance(div_s6_mc, (int, float)) else ('Div S6×MC: {}'.format(div_s6_mc) if div_s6_mc is not None else 'Div S6×MC: —'))
        with colC:
            st.markdown('**📈 M3 / Expectativa (dx)**')
            st.write(reg_m3 if reg_m3 is not None else '—')
        st.caption('Pode haver divergência porque cada fonte mede uma coisa: Pipeline descreve a estrada, Sentinelas medem risco/turbulência, e M3 (dx) mede expectativa/analogia. Use cada leitura no seu uso canônico — sem misturar.')
        st.caption('Somente leitura. Não decide nada. Serve para você bater o olho e saber: **o que rodou**, **o que falta**, e **quais leituras estão disponíveis**.')
        if '_m1_render_barra_estados' in globals():
            _m1_render_barra_estados(_estado.get('estado','S0'))
        if _estado.get('avisos'):
            st.warning('Ainda não percorrido (na sessão): ' + ' · '.join(_estado.get('avisos', [])))
        # Snapshot resumido
        _s = _m1_collect_mirror_snapshot() if '_m1_collect_mirror_snapshot' in globals() else _estado.get('snapshot', {})
        _bl0 = {'historico_df': 'definido' if _s.get('historico_df') else '<não definido>', 'n_alvo': _s.get('n_alvo','N/D'), 'universo': _s.get('universo','N/D'), 'pipeline_ok': bool(_s.get('pipeline_ok')), 'regime': _s.get('regime','N/D')}
        _bl1 = {'k_star': _s.get('k_star','N/D'), 'nr_percent': _s.get('nr_percent','N/D'), 'divergencia_s6_mc': _s.get('divergencia_s6_mc','N/D'), 'indice_risco': _s.get('indice_risco','N/D'), 'classe_risco': _s.get('classe_risco','N/D')}
        _bl2 = {'turbo_tentado': bool(_s.get('turbo_tentado')), 'turbo_bloqueado': bool(_s.get('turbo_bloqueado')), 'turbo_motivo': _s.get('turbo_motivo','N/D'), 'modo6_executado': bool(_s.get('modo6_executado')), 'listas_geradas': _s.get('listas_geradas','<não definido>')}
        st.json(_bl0)
        st.json(_bl1)
        st.json(_bl2)
    except Exception:
        pass

    # ------------------------------------------------------------
    # 🎞️ BLOCO -0.5 — MEMÓRIA & EXPECTATIVA (read-only, se disponíveis)
    # ------------------------------------------------------------
    with st.expander('🎞️ Memória de Estados (M2) + Expectativa Histórica (M3) — resumo', expanded=False):
        try:
            m2 = st.session_state.get('m2_memoria_resumo_auditavel')
            if m2:
                st.markdown('#### 🎞️ M2 — Memória de Estados (resumo)')
                st.json(m2)
            else:
                st.info('M2 ainda sem massa mínima nesta sessão. (Isso não é erro.)')
            m3n = st.session_state.get('m3_eventos_similares')
            if m3n is not None:
                st.markdown('#### 📈 M3 — Expectativa Histórica (resumo)')
                st.json({'m3_regime_dx': st.session_state.get('m3_regime_dx','N/D'), 'm3_eventos_similares': m3n, 'taxa_eco1': st.session_state.get('m3_taxa_eco1','N/D'), 'taxa_estado_bom': st.session_state.get('m3_taxa_estado_bom','N/D'), 'taxa_transicao': st.session_state.get('m3_taxa_transicao','N/D'), 'ts': st.session_state.get('m3_ts','N/D')})
            else:
                st.info('Para preencher M3 no Relatório Final: rode o painel **📈 Expectativa Histórica — Contexto do Momento (V16)** nesta sessão.')
        except Exception:
            pass


    # ------------------------------------------------------------
    # 🧲 BLOCO 0 — SUGADOR DE ESTADO CONSOLIDADO
    # ------------------------------------------------------------
    historico_df = st.session_state.get("historico_df")
    n_alvo = st.session_state.get("n_alvo")

    pipeline_status = st.session_state.get("pipeline_flex_ultra_concluido")
    ultima_prev = st.session_state.get("ultima_previsao")

    listas_m6_totais = (
        st.session_state.get("modo6_listas_totais")
        or st.session_state.get("modo6_listas")
        or []
    )

    listas_ultra = st.session_state.get("turbo_ultra_listas_leves") or []

    # Validação mínima
    if not listas_m6_totais:
        exibir_bloco_mensagem(
            "Sem pacote do Modo 6",
            "Execute o painel **🎯 Modo 6 Acertos — Execução** antes.",
            tipo="warning",
        )
        st.stop()

    # ------------------------------------------------------------
    # Estado consolidado
    # ------------------------------------------------------------
    linhas = []

    if historico_df is not None:
        linhas.append(f"- Séries carregadas: **{len(historico_df)}**")

    if n_alvo is not None:
        linhas.append(f"- Passageiros por carro (n): **{n_alvo}**")

    if pipeline_status is True:
        linhas.append("- Pipeline FLEX ULTRA: ✅ **CONCLUÍDO**")

    exibir_bloco_mensagem(
        "🧲 Estado Consolidado da Rodada",
        "\n".join(linhas),
        tipo="info",
    )

    # ============================================================
    # 🧠 DIAGNÓSTICO CONSOLIDADO DA RODADA (NOVO — ADITIVO)
    # ============================================================
    eco_consolidado = st.session_state.get("eco_status", "DESCONHECIDO")
    estado_consolidado = st.session_state.get("estado_atual", "DESCONHECIDO")

    st.markdown("### 🧠 Diagnóstico Consolidado da Rodada")

    st.info(
        f"**ECO:** {eco_consolidado}\n\n"
        f"**Estado do alvo:** {estado_consolidado}"
    )

    st.caption(
        "Leitura consolidada do sistema nesta rodada.\n"
        "Não gera decisão automática."
    )

    # ------------------------------------------------------------
    # Núcleo TURBO (se existir)
    # ------------------------------------------------------------
    st.markdown("### 🔮 Previsão Principal (Núcleo — TURBO++ ULTRA)")

    if ultima_prev:
        st.success(formatar_lista_passageiros(ultima_prev))
    else:
        st.info(
            "Nenhuma previsão TURBO disponível nesta rodada "
            "(isso é válido em regime estável)."
        )

    # ------------------------------------------------------------
    # 🛡️ Pacote Prioritário — Top 10 (Modo 6)
    # ------------------------------------------------------------
    st.markdown("### 🛡️ Pacote Prioritário (Top 10) — Modo 6")

    top10 = listas_m6_totais[:10]
    for i, lst in enumerate(top10, 1):
        st.markdown(f"**{i:02d})** {formatar_lista_passageiros(lst)}")


    # ------------------------------------------------------------
    # 🧷 Anti-Âncora (OBSERVACIONAL) — rotulagem Base × Anti
    # ------------------------------------------------------------
    try:
        analise_anti = v16_analisar_duplo_pacote_base_anti_ancora(
            listas=listas_m6_totais,
            base_n=10,
            max_anti=4,
            core_presenca_min=0.60,
        )
        st.session_state["v16_anti_ancora"] = analise_anti

        st.markdown("### 🧷 Anti-Âncora — Observacional (Base × Anti)")
        core = analise_anti.get("core") or []
        # --- V16: registrar compressão/CORE para Janela Local (não decide) ---
        st.session_state["janela_core_top10"] = list(core) if core else []
        st.session_state["janela_compressao_core"] = True if core else False
        if core:
            st.write("**CORE do pacote base (presença alta no Top 10):** " + ", ".join(map(str, core)))
        else:
            st.write("CORE indisponível (sem base suficiente).")

        anti_idx = analise_anti.get("anti_idx") or []
        if anti_idx:
            st.success(
                "Sugestão (não obrigatória): **Duplo pacote** = Base (Top 10) + "
                + f"Anti-âncora (listas existentes): {', '.join('L'+str(i) for i in anti_idx)}"
            )
            for i in anti_idx:
                try:
                    lst = listas_m6_totais[int(i) - 1]
                    ov = (analise_anti.get("overlaps") or [None])[int(i) - 1]
                    st.write(f"**L{i:02d} (anti-âncora | overlap CORE={ov})** — {formatar_lista_passageiros(lst)}")
                except Exception:
                    pass
        else:
            st.info(
                "Nenhuma lista anti-âncora clara foi detectada entre as listas disponíveis. "
                "Isso é compatível com pacote muito comprimido (E0 + envelope estreito)."
            )
    except Exception:
        st.session_state["v16_anti_ancora"] = None
        # falha silenciosa (não derruba o RF)





    # ------------------------------------------------------------
    # 🧩 Diagnóstico — Rigidez do Jeitão (folga) [OBSERVACIONAL]
    # ------------------------------------------------------------
    st.markdown("### 🧩 Jeitão do Pacote — Rigidez × Folga (diagnóstico)")
    st.caption("Alerta diagnóstico: quando o pacote fica rígido demais, ele pode 'acertar o jeitão' mas perder passageiros por rigidez. Isso NÃO é decisão: é só sinal para governança/cobertura.")

    try:
        umin = st.session_state.get("universo_min")
        umax = st.session_state.get("universo_max")
        diag_j = v16_diagnostico_rigidez_jeitao(
            listas=listas_m6_totais,
            universo_min=umin,
            universo_max=umax,
            base_n=10,
            core_presenca_min=0.60,
        )

        if diag_j.get("rigido"):
            st.warning(f"⚠️ {diag_j.get('mensagem')}")
        else:
            st.info(f"✅ {diag_j.get('mensagem')}")

        sinais = diag_j.get("sinais") or {}
        if sinais:
            st.write({
                "score_rigidez": diag_j.get("score"),
                "folga_qualitativa(alerta)": diag_j.get("folga_qualitativa"),
                "core_sz": sinais.get("core_sz"),
                "frac_colados": sinais.get("frac_colados"),
                "ov_mean": sinais.get("ov_mean"),
                "f_max": sinais.get("f_max"),
                "range_8": sinais.get("range_8"),
                "range_lim": sinais.get("range_lim"),
                "anti_idx_detectados": sinais.get("anti_idx_detectados"),
            })
    except Exception:
        st.info("Diagnóstico de rigidez indisponível nesta rodada (falha silenciosa).")
# ------------------------------------------------------------

    # ------------------------------------------------------------
    # 📊 EIXO 1 — CONTRIBUIÇÃO DE PASSAGEIROS (OBSERVACIONAL)
    # ------------------------------------------------------------
    try:
        listas_pacote_eixo1 = listas_m6_totais[:]
    
        historico_label = (
            f"C1 → C{len(historico_df)}"
            if historico_df is not None
            else "Histórico indefinido"
        )
    
        eixo1_resultado = calcular_eixo1_contribuicao(
            listas_pacote=listas_pacote_eixo1,
            historico_label=historico_label,
            modo_geracao="Modo 6",
            n_base=n_alvo or 6,
            eco_status=st.session_state.get("eco_status", "DESCONHECIDO"),
            estado_status=st.session_state.get("estado_atual", "DESCONHECIDO"),
        )
    except Exception:
        eixo1_resultado = None
    
    if eixo1_resultado:
        st.markdown("### 📊 Eixo 1 — Contribuição de Passageiros (Observacional)")
    
        st.write(
            f"**Núcleo local detectado:** "
            f"{'SIM' if eixo1_resultado['nucleo']['detectado'] else 'NÃO'} "
            f"({eixo1_resultado['nucleo']['tipo']})"
        )
    
        st.write(
            "**Estruturais do pacote:** "
            + (
                ", ".join(map(str, eixo1_resultado["papeis"]["estruturais"]))
                if eixo1_resultado["papeis"]["estruturais"]
                else "—"
            )
        )
    
        st.write(
            "**Contribuintes:** "
            + (
                ", ".join(map(str, eixo1_resultado["papeis"]["contribuintes"]))
                if eixo1_resultado["papeis"]["contribuintes"]
                else "—"
            )
        )
    
        st.write(
            "**Leitura sintética:** "
            + " ".join(eixo1_resultado["leitura_sintetica"])
        )
    
        st.caption(eixo1_resultado["trava"])
    
    
    # ============================================================
    # 📌 REGISTRO CANÔNICO DO MOMENTO — DIAGNÓSTICO (COPIÁVEL)
    # ============================================================
    try:
            # ------------------------------------------------------------
            # 
            # (camada experimental removida na âncora estável)

        universo_min = st.session_state.get("universo_min", "N/D")
        universo_max = st.session_state.get("universo_max", "N/D")
        termometro_estagio = "N/D"
        termometro_score = "N/D"
        registro_txt = f"""
    SÉRIE_BASE: {serie_base}
    SÉRIES_ALVO: {series_alvo}
    
    ECO: {st.session_state.get("eco_status", "N/D")}
    ESTADO_ALVO: {st.session_state.get("estado_atual", "N/D")}
    REGIME: {st.session_state.get("pipeline_estrada", "N/D")}
    CLASSE_RISCO: {st.session_state.get("classe_risco", "N/D")}
    NR_PERCENT: {st.session_state.get("nr_percent", "N/D")}
    K_STAR: {st.session_state.get("k_star", "N/D")}
    DIVERGENCIA: {st.session_state.get("divergencia_s6_mc", "N/D")}
    UNIVERSO: {universo_min}-{universo_max}
    N_CARRO: {n_alvo if n_alvo is not None else "N/D"}
    EIXO1_NUCLEO_DETECTADO: {'SIM' if eixo1_resultado and eixo1_resultado['nucleo']['detectado'] else 'NÃO'}
    EIXO1_TIPO_NUCLEO: {eixo1_resultado['nucleo']['tipo'] if eixo1_resultado and eixo1_resultado['nucleo']['detectado'] else 'inexistente'}
    EIXO1_PUXADORES: {', '.join(map(str, (eixo1_resultado['papeis']['estruturais'] + eixo1_resultado['papeis']['contribuintes'])[:8])) if eixo1_resultado else '—'}
    EIXO1_CONVERGENCIA: {'alta' if eixo1_resultado and eixo1_resultado['nucleo']['detectado'] and len(eixo1_resultado['papeis']['estruturais'] + eixo1_resultado['papeis']['contribuintes']) >= 4 else 'média' if eixo1_resultado and eixo1_resultado['nucleo']['detectado'] and len(eixo1_resultado['papeis']['estruturais'] + eixo1_resultado['papeis']['contribuintes']) >= 2 else 'baixa'}
    EIXO1_LEITURA: {' '.join(eixo1_resultado['leitura_sintetica']) if eixo1_resultado else 'pacote disperso'}
    PACOTE_BASE: Top10
    PACOTE_ANTI_ANCORA: {", ".join("L"+str(i) for i in (st.session_state.get("v16_anti_ancora") or {}).get("anti_idx", [])) or "—"}
    """.strip()
    
        st.code(registro_txt, language="text")
    
    except Exception:
        pass
    
    

    # ============================================================
    # 🧨 JANELA LOCAL DE ATAQUE + 📦 PACOTES TÉCNICOS (RF CANÔNICO)
    # ============================================================
    try:
        reg_m3 = (
            st.session_state.get("m3_regime_dx")
            or st.session_state.get("m3_regime")
            or "N/D"
        )
        classe_risco_rf = (
            st.session_state.get("classe_risco")
            or (st.session_state.get("diagnostico_risco") or {}).get("classe_risco")
            or "N/D"
        )

        analise_anti_rf = st.session_state.get("v16_anti_ancora") or {}
        core_rf = analise_anti_rf.get("core") or []
        anti_idx_rf = analise_anti_rf.get("anti_idx") or []

        # Janela ATIVA = estrutura (CORE) + contexto minimamente favorável (ECO/PRÉ-ECO) + risco não vermelho
        _reg_norm = str(reg_m3).upper().replace("É", "E")
        janela_ativa_session = st.session_state.get("janela_local_ativa")
        if janela_ativa_session is None:
            janela_ativa = bool(core_rf) and (_reg_norm in ["ECO", "PRE-ECO", "PRÉ-ECO"]) and ("🔴" not in str(classe_risco_rf))
        else:
            janela_ativa = bool(janela_ativa_session)


        st.markdown("### 🧨 Estado da Janela Local de Ataque")
        st.write(f"**Status da Janela:** {'ATIVA' if janela_ativa else 'NÃO ATIVA'}")
        st.write("**Tipo:** Local · Recortada · Observacional")
        st.write("**Base da leitura:**")
        st.write(f"- Compressão (Modo 6 / CORE dominante): {'SIM' if bool(core_rf) else 'NÃO'}")
        st.write(f"- Contexto histórico (M3 / dx): {reg_m3}")
        st.write(f"- Classe de risco: {classe_risco_rf}")
        st.caption("A existência de janela não obriga ataque. Ela apenas qualifica a leitura do momento.")

        st.markdown("### 📦 Pacotes Técnicos (classificação informativa)")

        # 🛡️ PACOTE BASE — sempre existe (Modo 6)
        st.markdown("#### 🛡️ PACOTE BASE — CANÔNICO")
        st.caption("Origem: Modo 6. Função: continuidade estatística. Sempre existe.")

        # ⚖️ ALTERNATIVO — só se janela ATIVA e houver anti-âncora clara (listas já existentes)
        st.markdown("#### ⚖️ PACOTE ALTERNATIVO — BALIZADO")
        if janela_ativa and anti_idx_rf:
            st.caption("Condição: janela ATIVA. Origem: listas existentes com baixo overlap com CORE (anti-âncora).")
            for i in anti_idx_rf[:4]:
                if 0 <= int(i) < len(listas_m6_totais):
                    st.write(f"ALT{int(i)+1}: " + formatar_lista_passageiros(listas_m6_totais[int(i)]))
        else:
            st.info("Não aplicável nesta rodada (janela não ativa ou sem material anti-âncora claro).")

        # 🔥 OFENSIVO — TURBO (se houver material)
        
        # ============================================================
        # 🧩 V8 — BORDA QUALIFICADA (ETAPA 2) — PRÉ-CAMADA 4
        # - Governança legível: explica "por que entrou"
        # - NÃO altera listas reais; só classifica borda interna/externa
        # ============================================================
        try:
            st.markdown("#### 🧩 V8 — BORDA QUALIFICADA (pré‑Camada 4)")
            st.caption("Etapa 2 do Ajuste Fino: qualidade da borda · sem motor novo · sem mexer em Modo 6/TURBO/Bala")

            pacote_base_v8 = pacote_atual if isinstance(pacote_atual, list) else None
            if not pacote_base_v8:
                st.info("V8 Bordas: pacote atual indisponível (rode o 🎯 Modo 6 nesta sessão).")
            else:
                # parâmetros conservadores
                _base_n = int(min(10, max(3, len(pacote_base_v8))))
                _core_min = float((st.session_state.get("v8_core_presenca_min") or 0.60))
                _delta = float((st.session_state.get("v8_quase_delta") or 0.12))

                # rigidez do jeitão (já existe no V7)
                rig_info = v16_diagnostico_rigidez_jeitao(
                    listas=pacote_base_v8,
                    universo_min=st.session_state.get("universo_min"),
                    universo_max=st.session_state.get("universo_max"),
                    base_n=_base_n,
                    core_presenca_min=_core_min,
                )

                v8_borda = v8_classificar_borda_qualificada(
                    listas=pacote_base_v8,
                    base_n=_base_n,
                    core_presenca_min=_core_min,
                    quase_delta=_delta,
                    max_borda_interna=6,
                    universo_min=st.session_state.get("universo_min"),
                    universo_max=st.session_state.get("universo_max"),
                    rigidez_info=rig_info,
                )

                st.session_state["v8_borda_qualificada"] = v8_borda

                meta_v8 = v8_borda.get("meta") or {}
                st.write({
                    "base_n": meta_v8.get("base_n"),
                    "core_presenca_min": meta_v8.get("core_presenca_min"),
                    "quase_delta": meta_v8.get("quase_delta"),
                    "rigidez": f"{'SIM' if meta_v8.get('rigido') else 'NÃO'} (score {meta_v8.get('score_rigidez')})",
                    "folga_qualitativa": meta_v8.get("folga_qualitativa"),
                })

                core_v8 = v8_borda.get("core") or []
                quase_v8 = v8_borda.get("quase_core") or []
                bi = v8_borda.get("borda_interna") or []
                be = v8_borda.get("borda_externa") or []

                st.write(f"**CORE (por presença):** {core_v8 if core_v8 else '—'}")
                st.write(f"**QUASE‑CORE (candidatos):** {quase_v8 if quase_v8 else '—'}")

                st.markdown("**✅ BORDA INTERNA (entra sem dispersar — sugestão observacional):**")
                if bi:
                    for p in bi:
                        motivos = (v8_borda.get("motivos_interna") or {}).get(p) or []
                        st.write(f"- **{p}** · " + " · ".join(motivos[:4]))
                else:
                    st.write("- —")

                st.markdown("**⛔ BORDA EXTERNA (não entra — risco de dispersão / distância / presença insuficiente):**")
                if be:
                    # mostra só os primeiros para não poluir RF
                    for p in be[:10]:
                        motivos = (v8_borda.get("motivos_externa") or {}).get(p) or []
                        st.write(f"- {p} · " + " · ".join(motivos[:3]))
                    if len(be) > 10:
                        st.caption(f"… +{len(be)-10} candidatos externos (ocultos p/ legibilidade).")
                else:
                    st.write("- —")

        except Exception:
            # falha silenciosa: nunca derruba o RF
            pass


        st.markdown("#### 🔥 PACOTE OFENSIVO — CONDICIONAL")
        turbo_tentado_rf = bool(st.session_state.get("turbo_tentado", False))
        if janela_ativa and (listas_ultra or ultima_prev):
            st.caption("Condição: janela ATIVA. Motor ofensivo tentado. Uso pontual e consciente.")
            if listas_ultra:
                for j, L in enumerate(listas_ultra[:6], start=1):
                    st.write(f"OF{j}: " + formatar_lista_passageiros(L))
            elif ultima_prev:
                st.write(formatar_lista_passageiros(ultima_prev))
        else:
            st.info(
                "Motor ofensivo tentado, sem material válido produzido nesta condição."
                if turbo_tentado_rf else
                "Motor ofensivo não tentado nesta rodada."
            )
            st.caption("Falha silenciosa é um resultado válido e informativo quando não há janela ofensiva.")
    except Exception:
        pass

    # ============================================================
    # 📌 LISTAS DE PREVISÃO ASSOCIADAS AO MOMENTO (COPIÁVEL)
    # ============================================================
    try:
        st.markdown("### 📌 Listas de Previsão Associadas ao Momento")
    
        listas_para_registro = []
    
        if "pacote_operacional" in locals() and pacote_operacional:
            listas_para_registro = pacote_operacional[:]
        elif listas_m6_totais:
            listas_para_registro = listas_m6_totais[:]
    
        if listas_para_registro:
            linhas_listas = []
            for i, lst in enumerate(listas_para_registro[:20], start=1):
                linhas_listas.append(
                    f"L{i}: " + ", ".join(str(x) for x in lst)
                )
    
            st.code("\n".join(linhas_listas), language="text")
        else:
            st.info("Nenhuma lista disponível para registro neste momento.")
    
    except Exception:
        pass


    # ============================================================
    # 🧠 Painel — Aptidão do Evento (CANÔNICO | SOMENTE LEITURA)
    # Avaliação AUTOMÁTICA de aptidão para Memória Operacional
    # ============================================================
    try:
        st.markdown("## 🧠 Painel de Aptidão do Evento")
    
        # -------------------------------
        # Inicialização defensiva
        # -------------------------------
        status_aptidao = "NÃO APTO"
        motivo_principal = "Critérios mínimos não atendidos"
        compatibilidade = "indefinida"
        observacao = "Leitura automática do sistema"
        eixo1_resumo = "N/D"
    
        # -------------------------------
        # Fontes (já calculadas no app)
        # -------------------------------
        eixo1_ok = bool(
            eixo1_resultado
            and eixo1_resultado.get("nucleo", {}).get("detectado", False)
        )
    
        regime = st.session_state.get("pipeline_estrada", "N/D")
        nr_percent = st.session_state.get("nr_percent", None)
        divergencia = st.session_state.get("divergencia_s6_mc", None)
    
        # -------------------------------
        # Regras de APTIDÃO (sistema decide)
        # -------------------------------
        if eixo1_ok and regime in ["🟩 Estrada Neutra / Estável", "🟨 Estrada Moderada"]:
            status_aptidao = "APTO"
            motivo_principal = "Núcleo observável + regime compatível"
    
        elif eixo1_ok and regime not in ["🟥 Estrada Ruim / Instável"]:
            status_aptidao = "APTO"
            motivo_principal = "Núcleo fraco porém reutilizável"
    
        else:
            status_aptidao = "NÃO APTO"
            motivo_principal = "Ausência de núcleo ou regime incompatível"
    
        # -------------------------------
        # Compatibilidade de densidade
        # -------------------------------
        if eixo1_ok and regime.startswith("🟩"):
            compatibilidade = "microvariações / envelope estreito"
        elif eixo1_ok:
            compatibilidade = "repescagem controlada"
        else:
            compatibilidade = "nenhuma (densidade bloqueada)"
    
        # -------------------------------
        # Resumo do EIXO 1 (canônico)
        # -------------------------------
        if eixo1_resultado:
            eixo1_resumo = (
                f"Núcleo={ 'SIM' if eixo1_resultado['nucleo']['detectado'] else 'NÃO' } | "
                f"Tipo={ eixo1_resultado['nucleo']['tipo'] } | "
                f"Puxadores="
                + (
                    ", ".join(
                        map(
                            str,
                            (
                                eixo1_resultado["papeis"]["estruturais"]
                                + eixo1_resultado["papeis"]["contribuintes"]
                            )[:6],
                        )
                    )
                    if eixo1_resultado["papeis"]["estruturais"]
                    or eixo1_resultado["papeis"]["contribuintes"]
                    else "—"
                )
            )
    
        # -------------------------------
        # Exibição CANÔNICA (sem decisão)
        # -------------------------------
        st.markdown("### 📋 Resumo Canônico de Aptidão")
    
        aptidao_txt = f"""
    STATUS_APTIDAO: {status_aptidao}
    MOTIVO_PRINCIPAL: {motivo_principal}
    EIXO1_RESUMO: {eixo1_resumo}
    COMPATIBILIDADE_DENSIDADE: {compatibilidade}
    OBSERVACAO: {observacao}
    """.strip()
    
        st.code(aptidao_txt, language="text")
    
    except Exception as e:
        st.warning("Painel de Aptidão indisponível nesta rodada.")

    
    # ------------------------------------------------------------
    # 📦 Pacote Operacional TOTAL (Modo 6 + TURBO ULTRA)
    # ------------------------------------------------------------
    pacote_operacional = listas_m6_totais.copy()

    for lst in listas_ultra:
        if lst not in pacote_operacional:
            pacote_operacional.append(lst)

    try:
        pacote_operacional = v16_priorizar_listas_por_contexto(pacote_operacional)
    except Exception:
        pass

    total_listas = len(pacote_operacional)

    # ------------------------------------------------------------
    # 🧭 PAINEL CANÔNICO — BALA HUMANO DENSO (MODO ASSISTIDO)
    # (Somente leitura | sem execução | sem recomendação)
    # ------------------------------------------------------------
    try:
        st.markdown("## 🧭 Bala Humano Denso — Modo Assistido (Painel Canônico)")

        # Leituras já existentes no sistema (somente leitura)
        diag_risco = st.session_state.get("diagnostico_risco", {}) or {}
        estrada = st.session_state.get("pipeline_estrada", "N/D")

        classe_risco = diag_risco.get("classe_risco", "N/D")
        nr_percent = diag_risco.get("nr_percent", None)
        divergencia = diag_risco.get("divergencia", None)
        indice_risco = diag_risco.get("indice_risco", None)

        # ------------------------------------------------------------
        # BLOCO 1 — Condição do Momento (sem score mágico)
        # ------------------------------------------------------------
        st.markdown("### 1️⃣ Condição do Momento")

        st.write(f"- Estrada (Pipeline): **{estrada}**")
        st.write(f"- Classe de risco (Monitor): **{classe_risco}**")

        if nr_percent is not None:
            st.write(f"- NR% (Ruído Condicional): **{float(nr_percent):.2f}%**")
        else:
            st.write("- NR% (Ruído Condicional): **N/D**")

        if divergencia is not None:
            st.write(f"- Divergência S6 vs MC: **{float(divergencia):.4f}**")
        else:
            st.write("- Divergência S6 vs MC: **N/D**")

        if indice_risco is not None:
            st.write(f"- Índice composto de risco: **{float(indice_risco):.4f}**")
        else:
            st.write("- Índice composto de risco: **N/D**")

        # Nota canônica (a comparabilidade “momento passado vs atual” entra na Fase C)
        st.info(
            "Leitura informativa: este painel descreve o terreno atual com métricas já existentes. "
            "A comparabilidade com momentos passados e a seleção automática de densidade entram na fase seguinte."
        )

        # ------------------------------------------------------------
        # BLOCO 2 — Formas de Densidade Compatíveis (canônico)
        # ------------------------------------------------------------
        st.markdown("### 2️⃣ Formas de Densidade Compatíveis (canônico)")

        st.write("- ✔ **Microvariações controladas**")
        st.write("- ✔ **Envelope estreito**")
        st.write("- ⚠ **Repescagem controlada**")
        st.write("- ❌ **Expansão de universo** (incompatível com o espírito do Bala Humano)")

        st.caption(
            "Observação: aqui ainda não há escolha automática de formato. "
            "O sistema apenas delimita o que é compatível com densidade (aprofundar, não dispersar)."
        )

        # ------------------------------------------------------------
        # BLOCO 3 — Expectativa sob Densidade (canônico)
        # ------------------------------------------------------------
        st.markdown("### 3️⃣ Expectativa sob Densidade (informativo)")

        st.write("- Redistribuição típica para **4/6**")
        st.write("- Elevação marginal de **5/6**")
        st.write("- **6/6 não observado** como viável de forma consistente neste tipo de leitura")
        st.write("- Ganho associado a **volume controlado**, não a salto de acerto")

        st.caption("Regra: densidade altera **distribuição**, não compra **certeza**.")

        # ------------------------------------------------------------
        # BLOCO 4 — Cláusula de Responsabilidade (canônico)
        # ------------------------------------------------------------
        st.markdown("### 4️⃣ Decisão Humana — Fronteira de Responsabilidade")

        st.write("- O sistema **não recomenda ação**")
        st.write("- O sistema **não define volume**")
        st.write("- O sistema **não executa automaticamente**")
        st.write("- A decisão e a exposição são do **operador**")

        st.markdown("---")

    except Exception:
        # Falha silenciosa canônica: não derruba fluxo operacional
        pass
    
    # ------------------------------------------------------------
    # 🔥 MANDAR BALA — POSTURA OPERACIONAL
    # ------------------------------------------------------------
    st.markdown("### 🔥 Mandar Bala — Postura Operacional (Ação Consciente)")

    qtd_bala = st.slider(
        "Quantas listas você quer levar para a ação nesta rodada?",
        min_value=1,
        max_value=total_listas,
        value=min(10, total_listas),
        step=1,
        key="slider_mandar_bala_restaurado",
    )

    for i, lst in enumerate(pacote_operacional[:qtd_bala], 1):
        st.markdown(f"**🔥 {i:02d})** {formatar_lista_passageiros(lst)}")

    exibir_bloco_mensagem(
        "🧩 Fechamento Operacional",
        f"- Listas disponíveis: **{total_listas}**\n"
        f"- Listas levadas para ação: **{qtd_bala}**\n\n"
        "📌 O sistema **não decide**. O operador **assume a postura**.",
        tipo="success",
    )

    # ============================================================
    # 🧠 RF-GOV — GOVERNANÇA INFORMATIVA (AVISOS | SEM EFEITO)
    # ============================================================
    try:
        st.markdown("### 🧠 RF-GOV — Governança Informativa")

        fenomeno_id = st.session_state.get("fenomeno_id", "N/D")
        alvo_atual = st.session_state.get("n_alvo", "N/D")

        eco_status = st.session_state.get("eco_status", "N/D")
        estado_status = st.session_state.get("estado_atual", "N/D")

        mo = st.session_state.get("memoria_operacional", [])
        tentativas_mesmo_alvo = [r for r in mo if r.get("alvo") == alvo_atual]

        avisos = []

        if len(tentativas_mesmo_alvo) >= 2:
            avisos.append(
                "⚠️ Múltiplas tentativas recentes para o mesmo alvo registradas."
            )

        if eco_status in ("RUIM", "DESCONHECIDO"):
            avisos.append("ℹ️ ECO desfavorável ou indefinido.")

        if estado_status in ("RÁPIDO", "INSTÁVEL"):
            avisos.append("ℹ️ Estado do alvo indica instabilidade.")

        st.info(
            f"**Fenômeno ID:** {fenomeno_id}\n\n"
            f"**Alvo:** {alvo_atual}\n\n"
            f"**ECO:** {eco_status}\n"
            f"**Estado:** {estado_status}"
        )

        for a in avisos:
            st.warning(a)

        if not avisos:
            st.success("Nenhum alerta relevante de governança nesta rodada.")

    except Exception:
        st.caption("RF-GOV indisponível nesta execução.")

    st.success("Relatório Final gerado com sucesso!")

# ============================================================
# <<< FIM — PAINEL 13 — 📘 Relatório Final
# ============================================================












# ============================================================
# Painel — ⏱️ DURAÇÃO DA JANELA — ANÁLISE HISTÓRICA (V16)
# Diagnóstico PURO | Mede quantas séries janelas favoráveis duraram
# NÃO prevê | NÃO decide | NÃO altera motores
# ============================================================

# ============================================================
# Painel — 🔍 Cruzamento Histórico do k (Observacional)
# V16 | LEITURA PURA | NÃO DECIDE | NÃO ALTERA MOTORES
# ============================================================

if painel == "🔍 Cruzamento Histórico do k":

    st.markdown("## 🔍 Cruzamento Histórico do k")
    st.caption(
        "Leitura observacional do histórico. "
        "Este painel NÃO interfere em decisões, volumes ou modos."
    )

    eventos = st.session_state.get("eventos_k_historico", [])

    if not eventos:
        exibir_bloco_mensagem(
            "Nenhum evento k encontrado",
            "Carregue o histórico para analisar os eventos k.",
            tipo="warning",
        )
        st.stop()

    df_k = pd.DataFrame(eventos)

    # ============================================================
    # FILTROS SIMPLES (OBSERVACIONAIS)
    # ============================================================
    st.markdown("### 🎛️ Filtros Observacionais")

    col1, col2, col3 = st.columns(3)

    with col1:
        filtro_estado = st.multiselect(
            "Estado do alvo",
            options=sorted(df_k["estado_alvo"].dropna().unique().tolist()),
            default=None,
        )

    with col2:
        filtro_pre_eco = st.selectbox(
            "PRÉ-ECO",
            options=["Todos", "Sim", "Não"],
            index=0,
        )

    with col3:
        filtro_eco = st.selectbox(
            "ECO",
            options=["Todos", "Sim", "Não"],
            index=0,
        )

    df_f = df_k.copy()

    if filtro_estado:
        df_f = df_f[df_f["estado_alvo"].isin(filtro_estado)]

    if filtro_pre_eco != "Todos":
        df_f = df_f[df_f["pre_eco"] == (filtro_pre_eco == "Sim")]

    if filtro_eco != "Todos":
        df_f = df_f[df_f["eco"] == (filtro_eco == "Sim")]

    # ============================================================
    # MÉTRICAS RESUMIDAS
    # ============================================================
    st.markdown("### 📊 Resumo Estatístico")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Eventos k", len(df_f))

    with col2:
        st.metric(
            "Δ médio entre ks",
            round(df_f["delta_series"].dropna().mean(), 2)
            if "delta_series" in df_f else "—",
        )

    with col3:
        st.metric(
            "k médio",
            round(df_f["k_valor"].mean(), 2)
            if "k_valor" in df_f else "—",
        )

    with col4:
        st.metric(
            "Máx k observado",
            int(df_f["k_valor"].max())
            if "k_valor" in df_f else "—",
        )

    # ============================================================
    # TABELA FINAL (LEITURA CRUA)
    # ============================================================
    st.markdown("### 📋 Eventos k — Histórico")

    st.dataframe(
        df_f[
            [
                "serie_id",
                "k_valor",
                "delta_series",
                "estado_alvo",
                "k_star",
                "nr_percent",
                "div_s6_mc",
                "pre_eco",
                "eco",
            ]
        ].sort_values("serie_id"),
        use_container_width=True,
    )

# ============================================================
# FIM — Painel Cruzamento Histórico do k
# ============================================================


if painel == "⏱️ Duração da Janela — Análise Histórica":

    st.markdown("## ⏱️ Duração da Janela — Análise Histórica")

    st.info(
        "Este painel mede, **no passado**, quantas séries consecutivas "
        "as janelas favoráveis **REALMENTE duraram**, após serem confirmadas.\n\n"
        "📌 Definição usada:\n"
        "- Abertura: melhora conjunta (NR%, divergência, k*, desempenho real)\n"
        "- Fechamento: perda clara dessa coerência\n\n"
        "⚠️ Este painel NÃO prevê entrada de janela."
    )

    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")

    if df is None or matriz_norm is None:
        exibir_bloco_mensagem(
            "Pipeline incompleto",
            "Execute **Carregar Histórico** e **Pipeline V14-FLEX ULTRA**.",
            tipo="warning",
        )
        st.stop()

    # ------------------------------------------------------------
    # Parâmetros FIXOS (diagnóstico histórico)
    # ------------------------------------------------------------
    JANELA_ANALISE = 200
    LIMIAR_NR_QUEDA = 0.02
    LIMIAR_DIV_QUEDA = 0.50

    col_pass = [c for c in df.columns if c.startswith("p")]

    # Helpers locais (réplicas leves, sem tocar no motor)
    def _nr_local(m):
        variancias = np.var(m, axis=1)
        ruido_A = float(np.mean(variancias))
        saltos = [
            np.linalg.norm(m[i] - m[i - 1]) for i in range(1, len(m))
        ]
        ruido_B = float(np.mean(saltos)) if saltos else 0.0
        return 0.55 * min(1.0, ruido_A / 0.08) + 0.45 * min(1.0, ruido_B / 1.20)

    def _div_local(m):
        base = m[-1]
        candidatos = m[-10:] if len(m) >= 10 else m
        return float(np.linalg.norm(np.mean(candidatos, axis=0) - base))

    resultados = []
    n = len(matriz_norm)

    for i in range(max(30, n - JANELA_ANALISE), n - 3):
        m_i = matriz_norm[: i + 1]
        m_f = matriz_norm[: i + 4]

        nr_i = _nr_local(m_i)
        nr_f = _nr_local(m_f)
        div_i = _div_local(m_i)
        div_f = _div_local(m_f)

        abriu = (nr_f - nr_i) < -LIMIAR_NR_QUEDA and (div_f - div_i) < -LIMIAR_DIV_QUEDA

        if abriu:
            duracao = 1
            for j in range(i + 1, n - 1):
                m_j = matriz_norm[: j + 1]
                if _nr_local(m_j) <= nr_f and _div_local(m_j) <= div_f:
                    duracao += 1
                else:
                    break

            resultados.append(duracao)

    if not resultados:
        st.warning("Nenhuma janela favorável clara detectada no período analisado.")
        st.stop()

    df_res = pd.DataFrame({"Duração (séries)": resultados})

    st.markdown("### 📊 Distribuição Histórica da Duração das Janelas")
    st.dataframe(df_res.describe(), use_container_width=True)

    st.info(
        f"📌 Total de janelas detectadas: **{len(resultados)}**\n\n"
        "Este painel responde:\n"
        "👉 *Quando a janela abre, ela costuma durar quantas séries?*\n\n"
        "Use isso para **decidir até quando mandar bala**."
    )

# ============================================================
# V16 — CAMADA D
# Estado do Alvo · Expectativa · Volume × Confiabilidade
# (FIX: usa divergência correta do Monitor de Risco)
# ============================================================

def _v16_get_nr_div_risco():
    """
    Leitura segura e compatível com o app:
    - NR% vem do Ruído Condicional (nr_percent) OU do Monitor (diagnostico_risco.nr_percent)
    - Divergência vem do Monitor (diagnostico_risco.divergencia) OU fallback (div_s6_mc)
    - Risco vem do Monitor (diagnostico_risco.indice_risco)
    """
    risco_pack = st.session_state.get("diagnostico_risco") or {}

    nr = st.session_state.get("nr_percent")
    if nr is None:
        nr = risco_pack.get("nr_percent")

    # ⚠️ FIX PRINCIPAL: no seu app a divergência oficial está aqui:
    div = risco_pack.get("divergencia")
    if div is None:
        # fallback legado (se existir em alguma variação do app)
        div = st.session_state.get("div_s6_mc")

    indice_risco = risco_pack.get("indice_risco")

    return nr, div, indice_risco


def v16_registrar_estado_alvo():
    """
    Classifica o estado do alvo com base em:
    - NR%
    - Divergência S6 vs MC
    - Índice de risco (composto)
    """
    nr, div, risco = _v16_get_nr_div_risco()

    if nr is None or div is None or risco is None:
        estado = {
            "tipo": "indefinido",
            "velocidade": "indefinida",
            "comentario": "Histórico/monitor insuficiente para classificar o alvo (rode Monitor de Risco e Ruído Condicional).",
        }
        st.session_state["estado_alvo_v16"] = estado
        return estado

    # velocidade ∈ [~0, ~1+] (heurística)
    velocidade = round((float(nr) / 100.0 + float(div) / 15.0 + float(risco)) / 3.0, 3)

    if velocidade < 0.30:
        tipo = "alvo_parado"
        comentario = "🎯 Alvo praticamente parado — oportunidade rara. Volume alto recomendado."
    elif velocidade < 0.55:
        tipo = "movimento_lento"
        comentario = "🎯 Alvo em movimento lento — alternar rajadas e coberturas."
    elif velocidade < 0.80:
        tipo = "movimento_rapido"
        comentario = "⚠️ Alvo em movimento rápido — reduzir agressividade."
    else:
        tipo = "disparado"
        comentario = "🚨 Alvo disparado — ambiente hostil. Operar apenas de forma respiratória."

    estado = {
        "tipo": tipo,
        "velocidade": velocidade,
        "comentario": comentario,
    }

    st.session_state["estado_alvo_v16"] = estado
    return estado


def v16_registrar_expectativa():
    """
    Estima expectativa de curto prazo (1–3 séries)
    com base em microjanelas, ruído e divergência.
    """
    micro = st.session_state.get("v16_microdiag") or {}
    nr, div, _ = _v16_get_nr_div_risco()

    if not micro or nr is None or div is None:
        expectativa = {
            "previsibilidade": "indefinida",
            "erro_esperado": "indefinido",
            "chance_janela_ouro": "baixa",
            "comentario": "Expectativa indisponível (rode Microjanelas V16 e garanta NR/divergência).",
        }
        st.session_state["expectativa_v16"] = expectativa
        return expectativa

    score = float(micro.get("score_melhor", 0.0) or 0.0)
    janela_ouro = bool(micro.get("janela_ouro", False))

    if janela_ouro and score >= 0.80 and float(nr) < 40.0 and float(div) < 5.0:
        expectativa = {
            "previsibilidade": "alta",
            "erro_esperado": "baixo",
            "chance_janela_ouro": "alta",
            "comentario": "🟢 Forte expectativa positiva nas próximas 1–3 séries.",
        }
    elif score >= 0.50 and float(nr) < 60.0:
        expectativa = {
            "previsibilidade": "moderada",
            "erro_esperado": "moderado",
            "chance_janela_ouro": "média",
            "comentario": "🟡 Ambiente misto. Oportunidades pontuais podem surgir no curto prazo.",
        }
    else:
        expectativa = {
            "previsibilidade": "baixa",
            "erro_esperado": "alto",
            "chance_janela_ouro": "baixa",
            "comentario": "🔴 Baixa previsibilidade nas próximas 1–3 séries (ruído/divergência dominantes).",
        }

    st.session_state["expectativa_v16"] = expectativa
    return expectativa


def v16_registrar_volume_e_confiabilidade():
    """
    Relaciona quantidade de previsões com confiabilidade estimada.
    O sistema informa — a decisão é do operador.
    """
    risco_pack = st.session_state.get("diagnostico_risco") or {}
    indice = risco_pack.get("indice_risco")

    if indice is None:
        volume_op = {
            "minimo": 3,
            "recomendado": 6,
            "maximo_tecnico": 20,
            "confiabilidades_estimadas": {},
            "comentario": "Confiabilidade não calculada (rode o Monitor de Risco).",
        }
        st.session_state["volume_operacional_v16"] = volume_op
        return volume_op

    indice = float(indice)
    conf_base = max(0.05, 1.0 - indice)

    volumes = [3, 6, 10, 20, 40, 80]
    confs = {}
    for v in volumes:
        confs[v] = round(max(0.01, conf_base - v * 0.003), 3)

    recomendado = 20 if conf_base > 0.35 else 6

    volume_op = {
        "minimo": 3,
        "recomendado": int(recomendado),
        "maximo_tecnico": 80,
        "confiabilidades_estimadas": confs,
        "comentario": (
            "O sistema informa volumes e confiabilidades estimadas. "
            "A decisão final de quantas previsões gerar é do operador."
        ),
    }

    st.session_state["volume_operacional_v16"] = volume_op
    return volume_op





# ============================================================
# Painel X — 🧠 Laudo Operacional V16 (Estado, Expectativa, Volume)
# ============================================================

if painel == "🧠 Laudo Operacional V16":

    st.markdown("## 🧠 Laudo Operacional V16 — Leitura do Ambiente")

    # --------------------------------------------------------
    # Leitura segura (usa Camada D se existir, senão guarda)
    # --------------------------------------------------------
    try:
        estado = v16_registrar_estado_alvo()
    except Exception:
        estado = {
            "tipo": "indefinido",
            "velocidade": "indefinida",
            "comentario": "Estado ainda não disponível.",
        }

    try:
        expectativa = v16_registrar_expectativa()
    except Exception:
        expectativa = {
            "previsibilidade": "indefinida",
            "erro_esperado": "indefinido",
            "chance_janela_ouro": "baixa",
            "comentario": "Expectativa ainda não disponível.",
        }

    try:
        volume_op = v16_registrar_volume_e_confiabilidade()
    except Exception:
        volume_op = {
            "minimo": "-",
            "recomendado": "-",
            "maximo_tecnico": "-",
            "confiabilidades_estimadas": {},
            "comentario": "Volume ainda não disponível.",
        }

    # --------------------------------------------------------
    # 1) Estado do Alvo
    # --------------------------------------------------------
    st.markdown("### 🎯 Estado do Alvo")
    st.info(
        f"Tipo: **{estado.get('tipo')}**  \n"
        f"Velocidade estimada: **{estado.get('velocidade')}**  \n"
        f"Comentário: {estado.get('comentario')}"
    )

    # --------------------------------------------------------
    # 2) Expectativa de Curto Prazo
    # --------------------------------------------------------
    st.markdown("### 🔮 Expectativa (1–3 séries)")
    st.info(
        f"Previsibilidade: **{expectativa.get('previsibilidade')}**  \n"
        f"Erro esperado: **{expectativa.get('erro_esperado')}**  \n"
        f"Chance de janela de ouro: **{expectativa.get('chance_janela_ouro')}**  \n\n"
        f"{expectativa.get('comentario')}"
    )

    # --------------------------------------------------------
    # 3) Volume x Confiabilidade
    # --------------------------------------------------------
    st.markdown("### 📊 Volume × Confiabilidade (informativo)")

    confs = volume_op.get("confiabilidades_estimadas", {})
    if isinstance(confs, dict) and confs:
        df_conf = pd.DataFrame(
            [{"Previsões": k, "Confiabilidade estimada": v} for k, v in confs.items()]
        )
        st.dataframe(df_conf, use_container_width=True)

    st.warning(
        f"📌 Volume mínimo: **{volume_op.get('minimo')}**  \n"
        f"📌 Volume recomendado: **{volume_op.get('recomendado')}**  \n"
        f"📌 Volume máximo técnico: **{volume_op.get('maximo_tecnico')}**  \n\n"
        f"{volume_op.get('comentario')}"
    )

    st.success(
        "O PredictCars informa o ambiente e os trade-offs.\n"
        "A decisão final de quantas previsões gerar é do operador."
    )


    # --------------------------------------------------------
    # 4) Jeitão do Pacote — Rigidez (Camada 2 / observacional)
    # --------------------------------------------------------
    try:
        listas_m6_totais = (
            st.session_state.get("modo6_listas_totais")
            or st.session_state.get("modo6_listas")
            or []
        )
        umin = st.session_state.get("universo_min")
        umax = st.session_state.get("universo_max")

        if listas_m6_totais:
            st.markdown("### 🧩 Jeitão do Pacote — Rigidez (diagnóstico)")
            diag_j = v16_diagnostico_rigidez_jeitao(
                listas=listas_m6_totais,
                universo_min=umin,
                universo_max=umax,
                base_n=10,
                core_presenca_min=0.60,
            )

            st.info(
                "Alerta diagnóstico (Camada 2): quando o pacote fica rígido demais, ele pode 'acertar o jeitão' "
                "mas perder passageiros por compressão. Isso **não** decide nada — serve para governança/cobertura."
            )

            if diag_j.get("rigido"):
                st.warning(f"⚠️ {diag_j.get('mensagem')}")
            else:
                st.success(f"✅ {diag_j.get('mensagem')}")

            sinais = diag_j.get("sinais") or {}
            if sinais:
                st.write({
                    "score_rigidez": diag_j.get("score"),
                    "folga_qualitativa(alerta)": diag_j.get("folga_qualitativa"),
                    "core_sz": sinais.get("core_sz"),
                    "frac_colados": sinais.get("frac_colados"),
                    "ov_mean": sinais.get("ov_mean"),
                    "f_max": sinais.get("f_max"),
                    "range_8": sinais.get("range_8"),
                    "range_lim": sinais.get("range_lim"),
                    "anti_idx_detectados": sinais.get("anti_idx_detectados"),
                })
        else:
            # Sem pacote Modo 6 na sessão — nada a diagnosticar
            pass
    except Exception:
        # Falha silenciosa permitida (diagnóstico não pode quebrar laudo)
        pass



# ============================================================
# PARTE 7/8 — FIM
# ============================================================

# ============================================================
# PARTE 8/8 — INÍCIO
# ============================================================


# ============================================================
# 🔥 HOTFIX DEFINITIVO — EXATO PROXY (NORMALIZAÇÃO TOTAL)
# NÃO PROCURAR FUNÇÃO
# NÃO SUBSTITUIR CÓDIGO EXISTENTE
# ESTE BLOCO SOBRESCREVE O COMPORTAMENTO INTERNAMENTE
# ============================================================

def _v16_exato_proxy__normalizar_serie(valor):
    """
    Converte qualquer coisa em inteiro válido de passageiro.
    Aceita:
    - int
    - float
    - string ('12', '12.0', ' 12 ')
    Retorna None se inválido.
    """
    try:
        if valor is None:
            return None
        if isinstance(valor, str):
            valor = valor.strip().replace(",", ".")
        v = int(float(valor))
        return v
    except Exception:
        return None


def _v16_exato_proxy__topk_frequentes_FIX(window_df: pd.DataFrame, cols_pass: list, top_k: int) -> set:
    freq = {}
    for c in cols_pass:
        for v in window_df[c].values:
            vv = _v16_exato_proxy__normalizar_serie(v)
            if vv is not None:
                freq[vv] = freq.get(vv, 0) + 1
    if not freq:
        return set()
    return set(k for k, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:top_k])


def _v16_exato_proxy__serie_set_FIX(df_row: pd.Series, cols_pass: list) -> set:
    out = set()
    for c in cols_pass:
        vv = _v16_exato_proxy__normalizar_serie(df_row[c])
        if vv is not None:
            out.add(vv)
    return out


# 🔒 SOBRESCREVE FUNÇÕES USADAS PELO PAINEL (SEM VOCÊ CAÇAR NADA)
try:
    v16_exato_proxy__topk_frequentes = _v16_exato_proxy__topk_frequentes_FIX
    v16_exato_proxy__serie_set = _v16_exato_proxy__serie_set_FIX
except Exception:
    pass

# ============================================================
# 🔥 FIM HOTFIX DEFINITIVO — EXATO PROXY (NORMALIZAÇÃO TOTAL)
# ============================================================

# ============================================================
# 📊 BLOCO NOVO — V16 PREMIUM — EXATO POR REGIME (PROXY)
# VERSÃO FORÇADA — NÃO FICA EM BRANCO
# ============================================================

V16_PAINEL_EXATO_PROXY_NOME = "📊 V16 Premium — EXATO por Regime (Proxy)"


def v16_painel_exato_por_regime_proxy():
    st.markdown("## 📊 V16 Premium — EXATO por Regime (Proxy)")

    # --------------------------------------------------------
    # 0) Obter histórico BASE (FORÇADO)
    # --------------------------------------------------------
    try:
        nome_df, df_base = v16_identificar_df_base()
    except Exception:
        df_base = None

    if df_base is None or len(df_base) == 0:
        st.error("❌ Histórico não disponível. Painel abortado.")
        return

    st.success(f"✔ Histórico detectado: {len(df_base)} séries")

    # --------------------------------------------------------
    # 1) Extração FORÇADA dos passageiros
    # Regra: colunas 1..6
    # --------------------------------------------------------
    cols = list(df_base.columns)
    if len(cols) < 7:
        st.error("❌ Histórico não tem colunas suficientes.")
        return

    cols_pass = cols[1:7]
    st.info(f"Passageiros usados: {cols_pass}")

    # --------------------------------------------------------
    # 2) Normalização TOTAL
    # --------------------------------------------------------
    def norm(v):
        try:
            return int(float(str(v).strip()))
        except Exception:
            return None

    # --------------------------------------------------------
    # 3) Parâmetros FIXOS
    # --------------------------------------------------------
    W = 60
    TOP_K = 12

    if len(df_base) <= W:
        st.error("❌ Histórico insuficiente para janela W=60.")
        return

    # --------------------------------------------------------
    # 4) Loop FORÇADO (sem filtros que zeram tudo)
    # --------------------------------------------------------
    registros = []

    for t in range(W, len(df_base)):
        janela = df_base.iloc[t - W : t]
        prox = df_base.iloc[t]

        freq = {}
        for c in cols_pass:
            for v in janela[c].values:
                vv = norm(v)
                if vv is not None:
                    freq[vv] = freq.get(vv, 0) + 1

        if not freq:
            continue

        topk = set(k for k, _ in sorted(freq.items(), key=lambda x: -x[1])[:TOP_K])

        real = set()
        for c in cols_pass:
            vv = norm(prox[c])
            if vv is not None:
                real.add(vv)

        hits = len(topk & real)

        # regime SIMPLES (FORÇADO)
        if hits >= 3:
            regime = "ECO"
        elif hits >= 2:
            regime = "PRÉ-ECO"
        else:
            regime = "RUIM"

        registros.append(
            {"regime": regime, "hits": hits}
        )

    if not registros:
        st.error("❌ Nenhum registro gerado.")
        return

    df = pd.DataFrame(registros)

    # --------------------------------------------------------
    # 5) RESULTADOS GARANTIDOS
    # --------------------------------------------------------
    resumo = []
    for r in ["ECO", "PRÉ-ECO", "RUIM"]:
        sub = df[df["regime"] == r]
        resumo.append({
            "Regime": r,
            "Eventos": len(sub),
            "Hits ≥2 (%)": round((sub["hits"] >= 2).mean() * 100, 2) if len(sub) else 0.0,
            "Hits ≥3 (%)": round((sub["hits"] >= 3).mean() * 100, 2) if len(sub) else 0.0,
        })

    df_out = pd.DataFrame(resumo)


    # --------------------------------------------------------
    # 4) Diagnóstico — Rigidez do Jeitão (folga) [OBSERVACIONAL]
    # --------------------------------------------------------
    st.markdown("### 🧩 Jeitão do Pacote — Rigidez × Folga (diagnóstico)")
    st.caption("Isso NÃO decide nem altera listas. Serve só para alertar sobre possível rigidez excessiva do pacote e sugerir 'folga' qualitativa como hipótese.")

    try:
        listas_m6_totais = (
            st.session_state.get("modo6_listas_totais")
            or st.session_state.get("modo6_listas")
            or []
        )
        if listas_m6_totais:
            umin = st.session_state.get("universo_min")
            umax = st.session_state.get("universo_max")
            diag_j = v16_diagnostico_rigidez_jeitao(
                listas=listas_m6_totais,
                universo_min=umin,
                universo_max=umax,
                base_n=10,
                core_presenca_min=0.60,
            )

            if diag_j.get("rigido"):
                st.warning(f"⚠️ {diag_j.get('mensagem')}")
            else:
                st.info(f"✅ {diag_j.get('mensagem')}")

            with st.expander("🔎 Ver sinais (auditável)"):
                st.write(diag_j.get("sinais", {}))
                st.write(f"Score: {diag_j.get('score')} | Folga (qualitativa / alerta): {diag_j.get('folga_qualitativa')}")
        else:
            st.info("Sem listas do Modo 6 nesta sessão — diagnóstico de rigidez só aparece após executar o **🎯 Modo 6**.")
    except Exception:
        st.info("Diagnóstico de rigidez indisponível nesta sessão (falha silenciosa).")

    st.markdown("### 📊 Resultado (FORÇADO)")
    st.dataframe(df_out, use_container_width=True)

    st.success("✅ Painel executado com sucesso (versão forçada).")


def v16_registrar_painel_exato_proxy__no_router():
    if st.session_state.get("_v16_exato_proxy_router_ok", False):
        return

    g = globals()

    if "v16_obter_paineis" in g:
        orig = g["v16_obter_paineis"]

        def novo():
            try:
                lst = list(orig())
            except Exception:
                lst = []
            if V16_PAINEL_EXATO_PROXY_NOME not in lst:
                lst.append(V16_PAINEL_EXATO_PROXY_NOME)
            return lst

        g["v16_obter_paineis"] = novo

    if "v16_renderizar_painel" in g:
        orig_r = g["v16_renderizar_painel"]

        def render(p):
            if p == V16_PAINEL_EXATO_PROXY_NOME:
                return v16_painel_exato_por_regime_proxy()
            return orig_r(p)

        g["v16_renderizar_painel"] = render

    st.session_state["_v16_exato_proxy_router_ok"] = True


try:
    v16_registrar_painel_exato_proxy__no_router()
except Exception:
    pass

# ============================================================
# 📊 FIM DO BLOCO NOVO — V16 PREMIUM — EXATO POR REGIME (PROXY)
# ============================================================

# ============================================================
# 📊 BLOCO NOVO — V16 PREMIUM — PRÉ-ECO → ECO (PERSISTÊNCIA & CONTINUIDADE)
# (COLAR ENTRE: FIM DO EXATO PROXY  e  INÍCIO DO V16 PREMIUM PROFUNDO)
# ============================================================

V16_PAINEL_PRE_ECO_PERSIST_NOME = "📊 V16 Premium — PRÉ-ECO → ECO (Persistência & Continuidade)"


def v16_painel_pre_eco_persistencia_continuidade():
    st.markdown("## 📊 V16 Premium — PRÉ-ECO → ECO (Persistência & Continuidade)")
    st.markdown(
        """
Este painel é **100% observacional** e **retrospectivo**.

Ele responde:
- ✅ Qual % de **PRÉ-ECO** vira **ECO** em **1–3 séries**?
- ✅ Como separar **PRÉ-ECO fraco** vs **PRÉ-ECO forte**?
- ✅ Quais são os **últimos PRÉ-ECO fortes** (para prontidão humana)?

**Sem mudar motor. Sem decidir operação.**
        """
    )

    # --------------------------------------------------------
    # 0) Histórico base (obrigatório)
    # --------------------------------------------------------
    try:
        nome_df, df_base = v16_identificar_df_base()
    except Exception:
        nome_df, df_base = None, None

    if df_base is None or len(df_base) == 0:
        st.warning("⚠️ Histórico não disponível. Carregue o histórico e volte aqui.")
        return

    cols = list(df_base.columns)
    if len(cols) < 7:
        st.error("❌ Histórico não tem colunas suficientes (precisa: série + 6 passageiros).")
        return

    cols_pass = cols[1:7]

    st.success(f"✔ Histórico detectado: {len(df_base)} séries")
    st.info(f"Passageiros usados: {cols_pass}")

    # --------------------------------------------------------
    # 1) Normalização TOTAL (robusta)
    # --------------------------------------------------------
    def norm(v):
        try:
            return int(float(str(v).strip().replace(",", ".")))
        except Exception:
            return None

    # --------------------------------------------------------
    # 2) Parâmetros FIXOS (decisão sem bifurcação)
    # --------------------------------------------------------
    W = 60
    TOP_K = 12
    LOOKAHEAD_MAX = 3
    RUN_BACK = 5
    MAX_JANELAS = 4000  # anti-zumbi interno

    if len(df_base) <= W + LOOKAHEAD_MAX:
        st.error(f"❌ Histórico insuficiente para W={W} + lookahead.")
        return

    # Anti-zumbi: só últimas MAX_JANELAS
    t_final = len(df_base) - 1
    t_inicial = max(W, t_final - MAX_JANELAS)

    st.markdown("### ⚙️ Parâmetros (fixos)")
    st.code(
        f"W = {W}\nTOP_K = {TOP_K}\nLOOKAHEAD_MAX = {LOOKAHEAD_MAX}\nRUN_BACK = {RUN_BACK}\nMAX_JANELAS = {MAX_JANELAS}",
        language="python",
    )

    st.info(f"🧱 Anti-zumbi interno: analisando t={t_inicial} até t={t_final} (máx {MAX_JANELAS} janelas).")

    # --------------------------------------------------------
    # 3) Funções internas (dx, topk, real, hits)
    # --------------------------------------------------------
    def dx_janela(window_df):
        vals = []
        for c in cols_pass:
            s = [norm(x) for x in window_df[c].values]
            s = [x for x in s if x is not None]
            if len(s) >= 2:
                vals.append(float(np.std(s, ddof=1)))
        if not vals:
            return None
        return float(np.mean(vals))

    def topk_frequentes(window_df):
        freq = {}
        for c in cols_pass:
            for x in window_df[c].values:
                vv = norm(x)
                if vv is not None:
                    freq[vv] = freq.get(vv, 0) + 1
        if not freq:
            return set()
        return set(k for k, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:TOP_K])

    def serie_real_set(df_row):
        out = set()
        for c in cols_pass:
            vv = norm(df_row[c])
            if vv is not None:
                out.add(vv)
        return out

    # --------------------------------------------------------
    # 4) Primeiro passe: dx_list para quantis ECO/PRE/RUIM
    # --------------------------------------------------------
    dx_list = []
    dx_por_t = {}
    for t in range(t_inicial, t_final + 1):
        wdf = df_base.iloc[t - W : t]
        dx = dx_janela(wdf)
        if dx is not None:
            dx_list.append(dx)
            dx_por_t[t] = dx

    if len(dx_list) < 80:
        st.error(f"❌ Poucas janelas válidas para quantis. Válidas: {len(dx_list)}")
        return

    q1 = float(np.quantile(dx_list, 0.33))
    q2 = float(np.quantile(dx_list, 0.66))

    st.markdown("### 🧭 Regimes por quantis (dx_janela)")
    st.info(
        f"q1 (ECO ≤): **{q1:.6f}**  \n"
        f"q2 (PRÉ-ECO ≤): **{q2:.6f}**  \n\n"
        "Regra: dx ≤ q1 → ECO | dx ≤ q2 → PRÉ-ECO | dx > q2 → RUIM"
    )

    # --------------------------------------------------------
    # 5) Segundo passe: regime + hits por t
    # --------------------------------------------------------
    registros = []
    regime_por_t = {}
    hits_por_t = {}

    for t in range(t_inicial, t_final + 1):
        if t not in dx_por_t:
            continue

        dx = dx_por_t[t]
        if dx <= q1:
            regime = "ECO"
        elif dx <= q2:
            regime = "PRÉ-ECO"
        else:
            regime = "RUIM"

        wdf = df_base.iloc[t - W : t]
        top = topk_frequentes(wdf)
        real = serie_real_set(df_base.iloc[t])
        hits = int(len(top & real))

        regime_por_t[t] = regime
        hits_por_t[t] = hits

        registros.append({"t": int(t), "dx": float(dx), "regime": regime, "hits": hits})

    if not registros:
        st.error("❌ Não houve registros válidos.")
        return

    df = pd.DataFrame(registros)

    # --------------------------------------------------------
    # 6) Persistência PRÉ-ECO (run_len_pre)
    # --------------------------------------------------------
    run_len_pre = {}
    current = 0
    for t in sorted(regime_por_t.keys()):
        if regime_por_t[t] == "PRÉ-ECO":
            current += 1
        else:
            current = 0
        run_len_pre[t] = current

    # --------------------------------------------------------
    # 7) PRÉ-ECO → ECO em 1..3 séries (taxas)
    # --------------------------------------------------------
    total_pre = 0
    vira_eco_1 = 0
    vira_eco_2 = 0
    vira_eco_3 = 0

    eventos_pre = []

    for t in sorted(regime_por_t.keys()):
        if regime_por_t[t] != "PRÉ-ECO":
            continue

        total_pre += 1

        r1 = regime_por_t.get(t + 1)
        r2 = regime_por_t.get(t + 2)
        r3 = regime_por_t.get(t + 3)

        ok1 = (r1 == "ECO")
        ok2 = (r1 == "ECO") or (r2 == "ECO")
        ok3 = (r1 == "ECO") or (r2 == "ECO") or (r3 == "ECO")

        vira_eco_1 += 1 if ok1 else 0
        vira_eco_2 += 1 if ok2 else 0
        vira_eco_3 += 1 if ok3 else 0

        # dx trend e repetição de hits>=2 (últimos RUN_BACK)
        ts = [x for x in range(t - (RUN_BACK - 1), t + 1) if x in dx_por_t and x in hits_por_t and x in regime_por_t]
        dx_seq = [dx_por_t[x] for x in ts]
        hit_seq = [hits_por_t[x] for x in ts]
        hits_2plus = sum(1 for h in hit_seq if h >= 2)

        dx_trend = "estável"
        if len(dx_seq) >= 2:
            if dx_seq[-1] < dx_seq[0]:
                dx_trend = "caindo"
            elif dx_seq[-1] > dx_seq[0]:
                dx_trend = "subindo"

        # Score simples (informativo): persistência + hits repetidos + dx caindo
        score = 0
        score += min(run_len_pre.get(t, 0), 12)            # 0..12
        score += hits_2plus                               # 0..5
        score += 2 if dx_trend == "caindo" else 0
        score -= 2 if dx_trend == "subindo" else 0
        score += 1 if ok3 else 0

        eventos_pre.append(
            {
                "t": int(t),
                "run_len_pre": int(run_len_pre.get(t, 0)),
                "hits_t": int(hits_por_t.get(t, 0)),
                "hits_2plus_ult5": int(hits_2plus),
                "dx_trend_ult5": dx_trend,
                "vira_ECO_em_1": bool(ok1),
                "vira_ECO_em_2": bool(ok2),
                "vira_ECO_em_3": bool(ok3),
                "score_pre_forte": int(score),
            }
        )

    if total_pre == 0:
        st.error("❌ Não houve eventos PRÉ-ECO para avaliar.")
        return

    taxa1 = vira_eco_1 / total_pre
    taxa2 = vira_eco_2 / total_pre
    taxa3 = vira_eco_3 / total_pre

    st.markdown("### ✅ Taxas PRÉ-ECO → ECO (objetivas)")
    st.dataframe(
        pd.DataFrame(
            [{
                "Eventos PRÉ-ECO": int(total_pre),
                "Vira ECO em 1": round(taxa1, 4),
                "Vira ECO em 2": round(taxa2, 4),
                "Vira ECO em 3": round(taxa3, 4),
            }]
        ),
        use_container_width=True
    )

    # --------------------------------------------------------
    # 8) Top PRÉ-ECO fortes recentes (guia humano)
    # --------------------------------------------------------
    df_evt = pd.DataFrame(eventos_pre).sort_values(["t"], ascending=True)

    # Top 10 recentes com maior score
    df_top = (
        df_evt.sort_values(["score_pre_forte", "t"], ascending=[False, False])
        .head(10)
        .copy()
    )

    st.markdown("### 🟡 Top 10 PRÉ-ECO fortes (recentes / score)")
    st.dataframe(df_top, use_container_width=True)

    st.success(
        "✅ Painel PRÉ-ECO → ECO executado.\n"
        "Ele mede persistência/continuidade — a decisão de prontidão continua humana."
    )


def v16_registrar_painel_pre_eco_persist__no_router():
    """
    Integra este painel ao roteador V16 (idempotente).
    """
    if st.session_state.get("_v16_pre_eco_persist_router_ok", False):
        return

    g = globals()

    if "v16_obter_paineis" in g and callable(g["v16_obter_paineis"]):
        _orig_obter = g["v16_obter_paineis"]

        def _wrap_v16_obter_paineis__pre_eco():
            try:
                lst = list(_orig_obter())
            except Exception:
                lst = []
            if V16_PAINEL_PRE_ECO_PERSIST_NOME not in lst:
                lst.append(V16_PAINEL_PRE_ECO_PERSIST_NOME)
            return lst

        g["v16_obter_paineis"] = _wrap_v16_obter_paineis__pre_eco

    if "v16_renderizar_painel" in g and callable(g["v16_renderizar_painel"]):
        _orig_render = g["v16_renderizar_painel"]

        def _wrap_v16_renderizar_painel__pre_eco(painel_nome: str):
            if painel_nome == V16_PAINEL_PRE_ECO_PERSIST_NOME:
                return v16_painel_pre_eco_persistencia_continuidade()
            return _orig_render(painel_nome)

        g["v16_renderizar_painel"] = _wrap_v16_renderizar_painel__pre_eco

    st.session_state["_v16_pre_eco_persist_router_ok"] = True


# Registrar no router imediatamente (sem mexer em menu/motor)
try:
    v16_registrar_painel_pre_eco_persist__no_router()
except Exception:
    pass

# ============================================================
# 📊 FIM DO BLOCO NOVO — V16 PREMIUM — PRÉ-ECO → ECO (PERSISTÊNCIA & CONTINUIDADE)
# ============================================================

# ============================================================
# 📊 BLOCO NOVO — V16 PREMIUM — PASSAGEIROS RECORRENTES EM ECO (INTERSEÇÃO)
# (COLAR IMEDIATAMENTE ANTES DE: "INÍCIO DO PAINEL V16 PREMIUM PROFUNDO  (COLAR AQUI)")
# ============================================================

V16_PAINEL_ECO_RECORRENTES_NOME = "📊 V16 Premium — Passageiros Recorrentes em ECO (Interseção)"


def v16_painel_passageiros_recorrentes_eco_intersecao():
    st.markdown("## 📊 V16 Premium — Passageiros Recorrentes em ECO (Interseção)")
    st.markdown(
        """
Este painel é **100% observacional** e **retrospectivo**.

Ele responde:
- ✅ Em **trechos ECO**, quais passageiros aparecem de forma **recorrente** (persistência)?
- ✅ Em blocos ECO **consecutivos**, qual é a **interseção** real dos TOP-K por janela?
- ✅ Quais são os **passageiros ECO-resilientes** (candidatos estruturais para EXATO)?

**Sem mudar motor. Sem decidir operação.**
        """
    )

    # --------------------------------------------------------
    # 0) Histórico base (robusto, sem caça)
    # --------------------------------------------------------
    try:
        nome_df, df_base = v16_identificar_df_base()
    except Exception:
        nome_df, df_base = None, None

    if df_base is None or len(df_base) == 0:
        st.warning("⚠️ Histórico não disponível. Carregue o histórico e volte aqui.")
        return

    cols = list(df_base.columns)
    if len(cols) < 7:
        st.error("❌ Histórico insuficiente: precisa de (série + 6 passageiros).")
        return

    cols_pass = cols[1:7]
    st.success(f"✔ Histórico detectado: {len(df_base)} séries")
    st.info(f"Passageiros usados: {cols_pass}")

    # --------------------------------------------------------
    # 1) Normalização TOTAL (robusta)
    # --------------------------------------------------------
    def norm(v):
        try:
            return int(float(str(v).strip().replace(",", ".")))
        except Exception:
            return None

    # --------------------------------------------------------
    # 2) Parâmetros FIXOS (sem bifurcação)
    # --------------------------------------------------------
    W = 60
    TOP_K = 12
    RUN_MIN = 3            # só consideramos "bloco ECO" com pelo menos 3 janelas ECO consecutivas
    MAX_JANELAS = 4000     # anti-zumbi interno

    if len(df_base) <= W + 5:
        st.error(f"❌ Histórico insuficiente para W={W}.")
        return

    t_final = len(df_base) - 1
    t_inicial = max(W, t_final - MAX_JANELAS)

    st.markdown("### ⚙️ Parâmetros (fixos)")
    st.code(
        f"W = {W}\nTOP_K = {TOP_K}\nRUN_MIN = {RUN_MIN}\nMAX_JANELAS = {MAX_JANELAS}",
        language="python",
    )
    st.info(f"🧱 Anti-zumbi interno: analisando t={t_inicial} até t={t_final} (máx {MAX_JANELAS} janelas).")

    # --------------------------------------------------------
    # 3) Funções internas (dx, topk)
    # --------------------------------------------------------
    def dx_janela(window_df):
        vals = []
        for c in cols_pass:
            s = [norm(x) for x in window_df[c].values]
            s = [x for x in s if x is not None]
            if len(s) >= 2:
                vals.append(float(np.std(s, ddof=1)))
        if not vals:
            return None
        return float(np.mean(vals))

    def topk_frequentes(window_df):
        freq = {}
        for c in cols_pass:
            for x in window_df[c].values:
                vv = norm(x)
                if vv is not None:
                    freq[vv] = freq.get(vv, 0) + 1
        if not freq:
            return set()
        ordenado = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        return set(k for k, _ in ordenado[:TOP_K])

    # --------------------------------------------------------
    # 4) Primeiro passe: dx por t + quantis para ECO/PRE/RUIM
    # --------------------------------------------------------
    dx_list = []
    dx_por_t = {}
    for t in range(t_inicial, t_final + 1):
        wdf = df_base.iloc[t - W : t]
        dx = dx_janela(wdf)
        if dx is not None:
            dx_list.append(dx)
            dx_por_t[t] = dx

    if len(dx_list) < 80:
        st.error(f"❌ Poucas janelas válidas para quantis. Válidas: {len(dx_list)}")
        return

    q1 = float(np.quantile(dx_list, 0.33))
    q2 = float(np.quantile(dx_list, 0.66))

    st.markdown("### 🧭 Regimes por quantis (dx_janela)")
    st.info(
        f"q1 (ECO ≤): **{q1:.6f}**  \n"
        f"q2 (PRÉ-ECO ≤): **{q2:.6f}**  \n\n"
        "Regra: dx ≤ q1 → ECO | dx ≤ q2 → PRÉ-ECO | dx > q2 → RUIM"
    )

    # --------------------------------------------------------
    # 5) Segundo passe: regime por t + TOP-K por t (apenas ECO)
    # --------------------------------------------------------
    regime_por_t = {}
    top_por_t = {}

    for t in range(t_inicial, t_final + 1):
        dx = dx_por_t.get(t)
        if dx is None:
            continue

        if dx <= q1:
            regime = "ECO"
        elif dx <= q2:
            regime = "PRÉ-ECO"
        else:
            regime = "RUIM"

        regime_por_t[t] = regime

        if regime == "ECO":
            wdf = df_base.iloc[t - W : t]
            top_por_t[t] = topk_frequentes(wdf)

    if not top_por_t:
        st.warning("⚠️ Nenhuma janela ECO detectada neste recorte.")
        return

    # --------------------------------------------------------
    # 6) Detectar blocos ECO consecutivos (runs)
    # --------------------------------------------------------
    ts_eco = sorted(top_por_t.keys())

    runs = []
    start = ts_eco[0]
    prev = ts_eco[0]
    for t in ts_eco[1:]:
        if t == prev + 1:
            prev = t
        else:
            runs.append((start, prev))
            start = t
            prev = t
    runs.append((start, prev))

    # filtrar runs curtos
    runs = [r for r in runs if (r[1] - r[0] + 1) >= RUN_MIN]

    st.markdown("### 🟢 Blocos ECO consecutivos (detectados)")
    st.info(
        f"Total de runs ECO (≥ {RUN_MIN}): **{len(runs)}**  \n"
        f"Total de janelas ECO: **{len(ts_eco)}**"
    )

    if not runs:
        st.warning("⚠️ Existem janelas ECO, mas nenhuma sequência ECO longa o suficiente (RUN_MIN).")
        return

    # --------------------------------------------------------
    # 7) Para cada run ECO: interseções cumulativas e persistência
    # --------------------------------------------------------
    resumo_runs = []
    contagem_passageiros_eco = {}  # persistência global em ECO (conta presença em TOP-K por janela)
    total_janelas_eco = 0

    for (a, b) in runs:
        ts = list(range(a, b + 1))
        sets = [top_por_t[t] for t in ts if t in top_por_t]
        if len(sets) < RUN_MIN:
            continue

        # persistência global
        for s in sets:
            for p in s:
                contagem_passageiros_eco[p] = contagem_passageiros_eco.get(p, 0) + 1

        total_janelas_eco += len(sets)

        # interseções cumulativas (2..min(6, len))
        inter_2 = None
        inter_3 = None
        inter_4 = None
        inter_5 = None
        inter_6 = None

        def inter_size(n):
            if len(sets) < n:
                return None
            inter = sets[0].copy()
            for i in range(1, n):
                inter &= sets[i]
            return len(inter)

        inter_2 = inter_size(2)
        inter_3 = inter_size(3)
        inter_4 = inter_size(4)
        inter_5 = inter_size(5)
        inter_6 = inter_size(6)

        # score simples do run (informativo): inter_3 e inter_4 pesam mais
        score_run = 0
        if inter_2 is not None: score_run += inter_2
        if inter_3 is not None: score_run += 2 * inter_3
        if inter_4 is not None: score_run += 3 * inter_4

        resumo_runs.append(
            {
                "t_ini": int(a),
                "t_fim": int(b),
                "len_run": int(b - a + 1),
                "inter_2": inter_2 if inter_2 is not None else 0,
                "inter_3": inter_3 if inter_3 is not None else 0,
                "inter_4": inter_4 if inter_4 is not None else 0,
                "inter_5": inter_5 if inter_5 is not None else 0,
                "inter_6": inter_6 if inter_6 is not None else 0,
                "score_run": int(score_run),
            }
        )

    if not resumo_runs:
        st.warning("⚠️ Não consegui consolidar runs ECO (depois de filtros).")
        return

    df_runs = pd.DataFrame(resumo_runs).sort_values(["score_run", "len_run", "t_fim"], ascending=[False, False, False])

    st.markdown("### 📊 Runs ECO — Interseção TOP-K (cumulativa)")
    st.dataframe(df_runs, use_container_width=True)

    # --------------------------------------------------------
    # 8) Passageiros ECO-resilientes (persistência global em ECO)
    # --------------------------------------------------------
    st.markdown("### 🎯 Passageiros ECO-resilientes (persistência em TOP-K durante ECO)")

    if total_janelas_eco <= 0:
        st.warning("⚠️ Total de janelas ECO inválido.")
        return

    itens = []
    for p, cnt in contagem_passageiros_eco.items():
        itens.append(
            {
                "passageiro": int(p),
                "presencas_em_ECO": int(cnt),
                "taxa_presenca_ECO": round(float(cnt) / float(total_janelas_eco), 4),
            }
        )

    df_p = pd.DataFrame(itens).sort_values(["taxa_presenca_ECO", "presencas_em_ECO", "passageiro"], ascending=[False, False, True])

    st.info(f"Total de janelas ECO consideradas (em runs): **{total_janelas_eco}**")
    st.dataframe(df_p.head(25), use_container_width=True)

    # lista curta (top 12)
    top12 = df_p.head(12)["passageiro"].tolist()
    st.success("✅ Lista curta (TOP 12 ECO-resilientes) — informativa (não é previsão):")
    st.code(", ".join(str(x) for x in top12))

    st.success(
        "✅ Painel Passageiros Recorrentes em ECO executado.\n"
        "Ele mede persistência/interseção — a decisão de ataque e montagem para 6 continua humana."
    )


# ============================================================
# 📊 FIM DO BLOCO NOVO — V16 PREMIUM — PASSAGEIROS RECORRENTES EM ECO (INTERSEÇÃO)
# ============================================================


# ============================================================
# INÍCIO DO PAINEL V16 PREMIUM PROFUNDO  (COLAR AQUI)
# ============================================================

# ============================================================
# PAINEL — 🔮 V16 Premium Profundo — Diagnóstico & Calibração
# ============================================================
if painel == "🔮 V16 Premium Profundo — Diagnóstico & Calibração":
    st.markdown("## 🔮 V16 Premium Profundo — Diagnóstico & Calibração")
    st.markdown(
        """
        Este painel **não altera nada do fluxo V15.7 MAX**.

        Ele serve para:
        - 📊 **Inspecionar o histórico ativo** (tamanho, colunas, distribuição de k),
        - 🛡️ **Verificar rapidamente o regime de risco potencial** para o TURBO++ e Modo 6 Acertos,
        - 📐 **Organizar informações de confiabilidade/QDS/k*** já calculadas em outros painéis.

        Tudo com **anti-zumbi interno**, rodando apenas em uma janela segura do histórico.
        """
    )

    # --------------------------------------------------------
    # 1) Descobrir automaticamente qual DF de histórico usar
    # --------------------------------------------------------
    nome_df, df_base = v16_identificar_df_base()

    if df_base is None:
        st.warning(
            "⚠️ Não encontrei nenhum DataFrame de histórico ativo em `st.session_state`.\n\n"
            "Use primeiro um painel que carregue o histórico (por exemplo, **Carregar Histórico**), "
            "e depois volte aqui."
        )
        st.stop()

    st.info(
        f"📁 DataFrame detectado para diagnóstico: **{nome_df}**  \n"
        f"Séries totais disponíveis: **{len(df_base)}**"
    )

    # --------------------------------------------------------
    # 2) Controle Anti-Zumbi V16 (apenas para este painel)
    # --------------------------------------------------------
    n_total = int(len(df_base))
    limite_max_slider = int(min(6000, max(500, n_total)))

    st.markdown("### 🛡️ Anti-zumbi V16 — Janela de Diagnóstico")

    limite_linhas = st.slider(
        "Quantidade máxima de séries a considerar no diagnóstico (janela final do histórico):",
        min_value=200,
        max_value=limite_max_slider,
        value=min(2000, limite_max_slider),
        step=100,
    )

    # --------------------------------------------------------
    # 3) Resumo básico do histórico (janela segura)
    # --------------------------------------------------------
    resumo = v16_resumo_basico_historico(df_base, limite_linhas=limite_linhas)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Séries totais no histórico", resumo.get("n_total", 0))
    with col2:
        st.metric("Séries usadas no diagnóstico", resumo.get("n_usado", 0))
    with col3:
        st.metric("Qtd. de colunas detectadas", len(resumo.get("colunas", [])))

    st.markdown("### 🧬 Colunas detectadas na janela de diagnóstico")
    st.write(resumo.get("colunas", []))

    # Distribuição de k (se existir)
    dist_k = resumo.get("dist_k", {})
    if dist_k:
        st.markdown("### 🎯 Distribuição de k (janela final do histórico)")
        df_k = pd.DataFrame(
            {"k": list(dist_k.keys()), "qtd": list(dist_k.values())}
        ).sort_values("k")
        df_k["proporção (%)"] = (df_k["qtd"] / df_k["qtd"].sum() * 100).round(2)
        st.dataframe(df_k, use_container_width=True)
    else:
        st.info("ℹ️ Não foi possível calcular a distribuição de k.")

    # --------------------------------------------------------
    # 4) Mapa rápido de confiabilidade / QDS / k*
    # --------------------------------------------------------
    st.markdown("### 🧠 Mapa rápido de confiabilidade (session_state)")

    with st.expander("Ver variáveis relevantes detectadas"):
        mapeamento_conf = v16_mapear_confiabilidade_session_state()
        if not mapeamento_conf:
            st.write("Nenhuma variável relevante encontrada.")
        else:
            st.json(mapeamento_conf)

    # --------------------------------------------------------
    # 5) Interpretação qualitativa do regime
    # --------------------------------------------------------
    st.markdown("### 🩺 Interpretação qualitativa do regime")
    comentario_regime = []

    if dist_k:
        total_k = sum(dist_k.values())
        denom_k = max(1, int(total_k))
        proporcao_k_alto = round(
            sum(qtd for k_val, qtd in dist_k.items() if k_val >= 3) / denom_k * 100,
            2,
        )
        proporcao_k_baixo = round(
            sum(qtd for k_val, qtd in dist_k.items() if k_val <= 1) / denom_k * 100,
            2,
        )

        comentario_regime.append(f"- k ≥ 3: **{proporcao_k_alto}%**")
        comentario_regime.append(f"- k ≤ 1: **{proporcao_k_baixo}%**")

        if proporcao_k_alto >= 35:
            comentario_regime.append("- 🟢 Regime mais estável.")
        elif proporcao_k_baixo >= 50:
            comentario_regime.append("- 🔴 Regime turbulento.")
        else:
            comentario_regime.append("- 🟡 Regime intermediário.")
    else:
        comentario_regime.append("- ℹ️ Sem dados suficientes para avaliar o regime.")

    st.markdown("\n".join(comentario_regime))

    st.success("Painel V16 Premium Profundo executado com sucesso!")

# ======================================================================
# 📊 V16 PREMIUM — PRÉ-ECO | CONTRIBUIÇÃO DE PASSAGEIROS (OBSERVACIONAL)
# (CTRL+F ESTE BLOCO)
# ======================================================================

def _v16_laplace_rate(sucessos: int, total: int, alpha: int = 1) -> float:
    # Suavização Laplace: (a+α)/(A+2α)
    if total <= 0:
        return 0.0
    return float((sucessos + alpha) / (total + 2 * alpha))

def _v16_wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    # Wilson score interval para proporção
    if n <= 0:
        return (0.0, 1.0)
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2*n)) / denom
    margin = (z / denom) * math.sqrt((p*(1-p)/n) + (z**2)/(4*(n**2)))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (lo, hi)

def _v16_delta_ci_worstcase(p1_ci: Tuple[float, float], p0_ci: Tuple[float, float]) -> Tuple[float, float]:
    # IC conservador para Δ = P1 - P0 usando pior caso:
    # Δ_lo = P1_lo - P0_hi ; Δ_hi = P1_hi - P0_lo
    return (p1_ci[0] - p0_ci[1], p1_ci[1] - p0_ci[0])

def _v16_safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default

def _v16_build_pre_eco_mask(df_ctx: pd.DataFrame,
                           teto_nr: float,
                           teto_div: float,
                           kstar_delta_max: float = 0.0) -> pd.Series:
    """
    PRÉ-ECO = prontidão objetiva:
      - NR% não explode
      - Divergência não hostil
      - k* não piora (Δk* <= kstar_delta_max)
      - Laudo não hostil (se existir coluna)
    """
    # Colunas esperadas (se existirem): 'kstar', 'nr', 'div', 'laudo_hostil'
    nr = df_ctx["nr"] if "nr" in df_ctx.columns else pd.Series([np.nan]*len(df_ctx))
    div = df_ctx["div"] if "div" in df_ctx.columns else pd.Series([np.nan]*len(df_ctx))
    kstar = df_ctx["kstar"] if "kstar" in df_ctx.columns else pd.Series([np.nan]*len(df_ctx))

    # Δk*
    kstar_prev = kstar.shift(1)
    dk = (kstar - kstar_prev)

    ok_nr = nr.apply(lambda v: _v16_safe_float(v, 999.0) <= teto_nr)
    ok_div = div.apply(lambda v: _v16_safe_float(v, 999.0) <= teto_div)
    ok_k = dk.apply(lambda v: _v16_safe_float(v, 999.0) <= kstar_delta_max)

    if "laudo_hostil" in df_ctx.columns:
        # laudo_hostil True = hostil, então queremos False
        ok_laudo = (~df_ctx["laudo_hostil"].fillna(False)).astype(bool)
    else:
        ok_laudo = pd.Series([True]*len(df_ctx))

    preeco = (ok_nr & ok_div & ok_k & ok_laudo)
    return preeco

def _v16_hits_exatos(car_a: List[int], car_b: List[int]) -> int:
    # acertos exatos = interseção simples
    sa = set(car_a)
    sb = set(car_b)
    return len(sa.intersection(sb))

def _v16_extract_car_numbers(row: Any) -> List[int]:
    """
    Extrator robusto: tenta pegar lista/tupla/np.array; se for string, tenta parsear dígitos.
    Mantém só ints >=0.
    """
    if row is None:
        return []
    if isinstance(row, (list, tuple, np.ndarray)):
        out = []
        for v in row:
            try:
                out.append(int(v))
            except Exception:
                pass
        return out
    if isinstance(row, str):
        # Extrai números inteiros de uma string
        nums = []
        cur = ""
        for ch in row:
            if ch.isdigit():
                cur += ch
            else:
                if cur != "":
                    nums.append(int(cur))
                    cur = ""
        if cur != "":
            nums.append(int(cur))
        return nums
    # fallback
    try:
        return [int(row)]
    except Exception:
        return []

def _v16_compute_contrib_table(historico_carros: List[List[int]],
                               df_ctx: pd.DataFrame,
                               preeco_mask: pd.Series,
                               w: int = 60,
                               alpha: int = 1,
                               amin: int = 6,
                               bmin: int = 20) -> pd.DataFrame:
    """
    Contribuição de passageiros no PRÉ-ECO:
      Para cada t (dentro janela), observa passageiros do carro real em t,
      e mede hit2/hit3 do próximo alvo (t+1).
    """
    n = len(historico_carros)
    if n < (w + 2):
        return pd.DataFrame()

    # Índices válidos: precisamos de t e t+1 existirem
    t_ini = max(0, n - w - 2)
    t_fim = n - 2  # último t que ainda tem t+1

    # Subconjunto analisado
    idxs = list(range(t_ini, t_fim + 1))

    # PRÉ-ECO alinhado em t
    preeco_sub = preeco_mask.iloc[idxs].reset_index(drop=True) if len(preeco_mask) >= n else pd.Series([False]*len(idxs))

    # Monta targets hit2/hit3 do alvo (t+1) com referência no t?
    # Aqui seguimos a definição observacional: hits exatos entre carro(t) e carro(t+1).
    # (Não é acerto do sistema; é dinâmica do alvo entre séries consecutivas.)
    hit2 = []
    hit3 = []
    passageiros_t = []

    for t in idxs:
        car_t = historico_carros[t]
        car_next = historico_carros[t+1]
        h = _v16_hits_exatos(car_t, car_next)
        hit2.append(1 if h >= 2 else 0)
        hit3.append(1 if h >= 3 else 0)
        passageiros_t.append(set(car_t))

    # Filtra só PRÉ-ECO
    rows = []
    for i, t in enumerate(idxs):
        if bool(preeco_sub.iloc[i]):
            rows.append((i, passageiros_t[i], hit2[i], hit3[i]))

    if len(rows) < 5:
        return pd.DataFrame()

    # Universo de passageiros observados no PRÉ-ECO
    universo = set()
    for _, ps, _, _ in rows:
        universo |= set(ps)
    universo = sorted(list(universo))

    # Base rates (para suporte)
    base_hit2 = sum(r[2] for r in rows) / max(1, len(rows))
    base_hit3 = sum(r[3] for r in rows) / max(1, len(rows))

    # Para cada passageiro p: conta A/B/a/b para hit2 e hit3
    data = []
    for p in universo:
        A = 0
        B = 0

        a2 = 0
        b2 = 0
        a3 = 0
        b3 = 0

        for _, ps, y2, y3 in rows:
            if p in ps:
                A += 1
                a2 += y2
                a3 += y3
            else:
                B += 1
                b2 += y2
                b3 += y3

        # Gates
        if A < amin or B < bmin:
            cls = "INSUFICIENTE"
        else:
            cls = "PENDENTE"  # define abaixo

        # Taxas suavizadas
        p1_2 = _v16_laplace_rate(a2, A, alpha=alpha)
        p0_2 = _v16_laplace_rate(b2, B, alpha=alpha)
        p1_3 = _v16_laplace_rate(a3, A, alpha=alpha)
        p0_3 = _v16_laplace_rate(b3, B, alpha=alpha)

        # Lifts
        lift2 = (p1_2 / p0_2) if p0_2 > 0 else np.nan
        lift3 = (p1_3 / p0_3) if p0_3 > 0 else np.nan

        # IC Wilson para proporções (usando p sem Laplace para CI, mais “puro”)
        raw_p1_2 = (a2 / A) if A > 0 else 0.0
        raw_p0_2 = (b2 / B) if B > 0 else 0.0
        raw_p1_3 = (a3 / A) if A > 0 else 0.0
        raw_p0_3 = (b3 / B) if B > 0 else 0.0

        ci_p1_2 = _v16_wilson_ci(raw_p1_2, A)
        ci_p0_2 = _v16_wilson_ci(raw_p0_2, B)
        ci_p1_3 = _v16_wilson_ci(raw_p1_3, A)
        ci_p0_3 = _v16_wilson_ci(raw_p0_3, B)

        # Δ e IC conservador
        d2 = p1_2 - p0_2
        d3 = p1_3 - p0_3

        ci_d2 = _v16_delta_ci_worstcase(ci_p1_2, ci_p0_2)
        ci_d3 = _v16_delta_ci_worstcase(ci_p1_3, ci_p0_3)

        # Score (z aprox): z = Δ / SE(Δ) (SE aprox com raw, para não “embelezar”)
        se2 = math.sqrt((raw_p1_2*(1-raw_p1_2)/max(1, A)) + (raw_p0_2*(1-raw_p0_2)/max(1, B)))
        se3 = math.sqrt((raw_p1_3*(1-raw_p1_3)/max(1, A)) + (raw_p0_3*(1-raw_p0_3)/max(1, B)))

        z2 = ( (raw_p1_2 - raw_p0_2) / se2 ) if se2 > 0 else 0.0
        z3 = ( (raw_p1_3 - raw_p0_3) / se3 ) if se3 > 0 else 0.0

        score = (2.0 * z3) + (1.0 * z2)

        # Classificação (só se não for insuficiente)
        if cls != "INSUFICIENTE":
            # Regras conservadoras (fixas)
            leader = (ci_d3[0] > 0.0) and (not np.isnan(lift3)) and (lift3 >= 1.10) and (score >= 1.0)
            discard = (ci_d3[1] < 0.0) and (not np.isnan(lift3)) and (lift3 <= 0.90) and (score <= -1.0)

            if leader:
                cls = "LÍDER"
            elif discard:
                cls = "DESCARTÁVEL"
            else:
                cls = "NEUTRO"

        data.append({
            "passageiro": int(p),
            "A_presente": int(A),
            "a_hit2": int(a2),
            "a_hit3": int(a3),
            "B_ausente": int(B),
            "b_hit2": int(b2),
            "b_hit3": int(b3),
            "P1_hit2": float(p1_2),
            "P0_hit2": float(p0_2),
            "Δ_hit2": float(d2),
            "Lift_hit2": float(lift2) if not np.isnan(lift2) else np.nan,
            "ICΔ_hit2_lo": float(ci_d2[0]),
            "ICΔ_hit2_hi": float(ci_d2[1]),
            "P1_hit3": float(p1_3),
            "P0_hit3": float(p0_3),
            "Δ_hit3": float(d3),
            "Lift_hit3": float(lift3) if not np.isnan(lift3) else np.nan,
            "ICΔ_hit3_lo": float(ci_d3[0]),
            "ICΔ_hit3_hi": float(ci_d3[1]),
            "z_hit2": float(z2),
            "z_hit3": float(z3),
            "score": float(score),
            "classe": cls,
            "base_hit2_preEco": float(base_hit2),
            "base_hit3_preEco": float(base_hit3),
        })

    df = pd.DataFrame(data)
    if df.empty:
        return df

    # Ordenação: primeiro líderes por score, depois neutros, depois descartáveis, depois insuficientes
    ordem = {"LÍDER": 0, "NEUTRO": 1, "DESCARTÁVEL": 2, "INSUFICIENTE": 3}
    df["ordem_classe"] = df["classe"].map(ordem).fillna(9).astype(int)
    df = df.sort_values(by=["ordem_classe", "score"], ascending=[True, False]).drop(columns=["ordem_classe"])
    return df

def _v16_pairwise_coocorrencia(preeco_rows_passageiros: List[set], top_n: int = 25) -> pd.DataFrame:
    """
    Coocorrência (Jaccard) entre passageiros dentro do PRÉ-ECO.
    Retorna top pares com maior Jaccard (para alertar líder condicionado).
    """
    if len(preeco_rows_passageiros) < 8:
        return pd.DataFrame()

    # Universo
    uni = set()
    for s in preeco_rows_passageiros:
        uni |= set(s)
    uni = sorted(list(uni))

    # Contagens de presença
    pres = {p: 0 for p in uni}
    for s in preeco_rows_passageiros:
        for p in s:
            pres[p] += 1

    # Pairs
    pairs = []
    uni_len = len(uni)
    for i in range(uni_len):
        p = uni[i]
        for j in range(i+1, uni_len):
            q = uni[j]
            inter = 0
            union = 0
            for s in preeco_rows_passageiros:
                ip = (p in s)
                iq = (q in s)
                if ip or iq:
                    union += 1
                    if ip and iq:
                        inter += 1
            if union > 0:
                jac = inter / union
                if jac > 0:
                    pairs.append((p, q, inter, union, jac))

    if not pairs:
        return pd.DataFrame()

    dfp = pd.DataFrame(pairs, columns=["p", "q", "inter", "union", "jaccard"])
    dfp = dfp.sort_values(by="jaccard", ascending=False).head(top_n)
    return dfp

# ----------------------------------------------------------------------
# 📊 PAINEL — V16 PREMIUM — PRÉ-ECO | CONTRIBUIÇÃO DE PASSAGEIROS
# ----------------------------------------------------------------------
if "painel" in locals() and painel == "📊 V16 Premium — PRÉ-ECO | Contribuição de Passageiros":
    st.title("📊 V16 Premium — PRÉ-ECO | Contribuição de Passageiros")
    st.caption("Observacional, retrospectivo, objetivo e replicável. ❌ Sem motor. ❌ Sem listas. ✅ Só EXATO (Hit2/Hit3).")

    # -----------------------------
    # Parâmetros FIXOS (comando)
    # -----------------------------
    W_FIXO = 60
    ALPHA = 1
    AMIN = 6
    BMIN = 20

    with st.expander("🔒 Critério fixo (transparência total)", expanded=True):
        st.markdown(
            f"""
- **Janela W:** `{W_FIXO}` (fixo)
- **Suavização Laplace α:** `{ALPHA}` (fixo)
- **Amin / Bmin:** `{AMIN}` / `{BMIN}` (fixo)
- **Foco:** Hit3 (peso 2) + Hit2 (peso 1) → **score**
- **PRÉ-ECO:** filtro objetivo (NR, divergência, Δk*, laudo hostil se existir)
"""
        )

    # -----------------------------
    # Coleta do histórico (somente leitura)
    # -----------------------------
    # Tentamos chaves prováveis sem quebrar o app
    historico_carros = None

    # Opção 1: já existe lista pronta em session_state
    for k in ["historico_carros", "historico", "carros_historico", "dados_historico_carros"]:
        if k in st.session_state and st.session_state[k] is not None:
            historico_carros = st.session_state[k]
            break

    # Opção 2: tenta montar a partir de um DataFrame de histórico
    if historico_carros is None:
        for kdf in ["df_historico", "df", "dados", "historico_df"]:
            if kdf in st.session_state and isinstance(st.session_state[kdf], pd.DataFrame):
                dfh = st.session_state[kdf].copy()
                # Tenta inferir colunas com números
                cols_num = [c for c in dfh.columns if str(c).lower().strip() in ["n1","n2","n3","n4","n5","n6","a","b","c","d","e","f"]]
                if len(cols_num) >= 5:
                    historico_carros = []
                    for _, r in dfh.iterrows():
                        car = []
                        for c in cols_num[:6]:
                            try:
                                car.append(int(r[c]))
                            except Exception:
                                pass
                        historico_carros.append(car)
                break

    if not historico_carros or len(historico_carros) < (W_FIXO + 2):
        st.warning("Histórico insuficiente para o painel (precisa de W+2 séries). Carregue histórico completo e rode novamente.")
        st.stop()

    n_total = len(historico_carros)
    st.info(f"📁 Histórico detectado: **{n_total} séries**. Janela analisada: **últimas {W_FIXO} séries úteis (com alvo t+1)**.")

    # -----------------------------
    # Contexto de métricas (k*, NR, diverg, laudo)
    # -----------------------------
    # Este painel NÃO inventa métricas: ele lê o que existir.
    # Se não existir, ele opera com defaults conservadores → PRÉ-ECO vira “raríssimo” (ou vazio).
    df_ctx = pd.DataFrame({"idx": list(range(n_total))})

    # Tenta puxar séries de k*, NR, divergência, laudo hostil (se já existirem no seu app)
    # Chaves prováveis (mantendo robusto)
    series_map = [
        ("kstar", ["kstar_series", "serie_kstar", "kstar_hist", "kstar_por_serie"]),
        ("nr",    ["nr_series", "serie_nr", "nr_hist", "nr_por_serie"]),
        ("div",   ["div_series", "serie_div", "div_hist", "divergencia_series", "div_s6_mc_series"]),
        ("laudo_hostil", ["laudo_hostil_series", "serie_laudo_hostil"]),
    ]

    for col, keys in series_map:
        val = None
        for kk in keys:
            if kk in st.session_state and st.session_state[kk] is not None:
                val = st.session_state[kk]
                break
        if val is not None:
            try:
                s = pd.Series(list(val))
                if len(s) >= n_total:
                    s = s.iloc[:n_total]
                else:
                    # completa com NaN
                    s = s.reindex(range(n_total))
                df_ctx[col] = s
            except Exception:
                pass

    # Tetos PRÉ-ECO (fixos/visíveis — mas não “otimizáveis”)
    # Se você já tiver tetos globais no app, você pode substituir por leitura deles.
    teto_nr = 0.20
    teto_div = 0.35

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("🔎 Teto NR% (PRÉ-ECO)", f"{teto_nr:.2f}")
    with colB:
        st.metric("🔎 Teto Diverg (PRÉ-ECO)", f"{teto_div:.2f}")
    with colC:
        st.metric("🔎 Δk* máx (PRÉ-ECO)", "≤ 0.00")

    preeco_mask = _v16_build_pre_eco_mask(df_ctx=df_ctx, teto_nr=teto_nr, teto_div=teto_div, kstar_delta_max=0.0)

    # Aplica janela W (final do histórico)
    t_ini = max(0, n_total - W_FIXO - 2)
    t_fim = n_total - 2
    preeco_sub = preeco_mask.iloc[t_ini:t_fim+1].reset_index(drop=True)

    qtd_preeco = int(preeco_sub.sum())
    st.success(f"🟡 Rodadas PRÉ-ECO detectadas (na janela): **{qtd_preeco}** / {len(preeco_sub)}")

    if qtd_preeco < 5:
        st.warning("PRÉ-ECO muito raro nesta janela (ou métricas ausentes). O painel mantém honestidade: sem base, sem classificação forte.")
        # ainda assim tentamos rodar; provavelmente vai dar vazio/insuficiente.

    # -----------------------------
    # Calcula tabela de contribuição
    # -----------------------------
    df_contrib = _v16_compute_contrib_table(
        historico_carros=historico_carros,
        df_ctx=df_ctx,
        preeco_mask=preeco_mask,
        w=W_FIXO,
        alpha=ALPHA,
        amin=AMIN,
        bmin=BMIN
    )

    if df_contrib.empty:
        st.warning("Sem dados suficientes para medir contribuição (PRÉ-ECO insuficiente ou janela curta).")
        st.stop()

    # -----------------------------
    # Visões (Líder / Neutro / Descartável / Insuficiente)
    # -----------------------------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🏁 LÍDERES", int((df_contrib["classe"] == "LÍDER").sum()))
    with c2:
        st.metric("⚪ NEUTROS", int((df_contrib["classe"] == "NEUTRO").sum()))
    with c3:
        st.metric("❌ DESCARTÁVEIS", int((df_contrib["classe"] == "DESCARTÁVEL").sum()))
    with c4:
        st.metric("🟡 INSUF.", int((df_contrib["classe"] == "INSUFICIENTE").sum()))

    st.markdown("### 🧾 Tabela completa (ordenada por classe → score)")
    st.dataframe(
        df_contrib,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    st.markdown("### 🏁 Top LÍDERES (PRÉ-ECO)")
    st.dataframe(
        df_contrib[df_contrib["classe"] == "LÍDER"].head(25),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### ❌ Top DESCARTÁVEIS (PRÉ-ECO)")
    st.dataframe(
        df_contrib[df_contrib["classe"] == "DESCARTÁVEL"].head(25),
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------
    # Coocorrência (Líder condicionado)
    # -----------------------------
    st.markdown("---")
    st.markdown("### 🔗 Coocorrência (Jaccard) — alerta de “líder condicionado”")

    # Reconstroi sets PRÉ-ECO na janela
    idxs = list(range(t_ini, t_fim + 1))
    preeco_rows_sets = []
    for t in idxs:
        if bool(preeco_mask.iloc[t]):
            preeco_rows_sets.append(set(historico_carros[t]))

    df_pairs = _v16_pairwise_coocorrencia(preeco_rows_sets, top_n=30)
    if df_pairs.empty:
        st.info("Coocorrência insuficiente para análise robusta nesta janela (ou PRÉ-ECO raro).")
    else:
        st.dataframe(df_pairs, use_container_width=True, hide_index=True)
        st.caption("Quanto maior o Jaccard, mais “colados” os passageiros aparecem. Isso NÃO é corte — é alerta observacional.")

    st.markdown("---")
    st.caption("🔒 Este painel é 100% observacional: não gera listas, não decide, não altera motor. Ele mede contribuição condicional no PRÉ-ECO (Hit2/Hit3).")

# ============================================================
# 📊 V16 PREMIUM — ANTI-EXATO | PASSAGEIROS NOCIVOS CONSISTENTES
# ============================================================
if painel == "📊 V16 Premium — ANTI-EXATO | Passageiros Nocivos":

    st.title("📊 V16 Premium — ANTI-EXATO | Passageiros Nocivos Consistentes")
    st.caption(
        "Observacional • Retrospectivo • Objetivo\n"
        "Identifica passageiros que REDUZEM a chance de EXATO (≥2 / ≥3).\n"
        "❌ Não gera listas • ❌ Não decide • ✅ Apoia limpeza do Modo 6"
    )

    # --------------------------------------------------------
    # Parâmetros FIXOS (canônicos)
    # --------------------------------------------------------
    W = 60
    ALPHA = 1
    AMIN = 12
    BMIN = 40

    st.markdown(
        f"""
**Critério fixo**
- Janela: **{W}**
- Suavização Laplace: **α = {ALPHA}**
- Amostra mínima: **A ≥ {AMIN}**, **B ≥ {BMIN}**
- Evento-alvo: **Hit3 (principal)** + Hit2 (suporte)
"""
    )

    # --------------------------------------------------------
    # Fonte CANÔNICA de passageiros (Pipeline V14-FLEX ULTRA)
    # --------------------------------------------------------
    if "pipeline_col_pass" not in st.session_state:
        st.warning(
            "Fonte canônica de passageiros não encontrada.\n\n"
            "Execute primeiro o painel 🛣️ Pipeline V14-FLEX ULTRA."
        )
        st.stop()

    col_pass = st.session_state["pipeline_col_pass"]

    nome_df, df_base = v16_identificar_df_base()
    if df_base is None:
        st.warning("Histórico não encontrado. Carregue o histórico e rode o Pipeline.")
        st.stop()

    if len(col_pass) < 6:
        st.warning("Fonte de passageiros inválida (menos de 6 colunas).")
        st.stop()

    historico = df_base[col_pass].astype(int).values.tolist()
    n = len(historico)

    if n < (W + 2):
        st.warning("Histórico insuficiente para análise ANTI-EXATO.")
        st.stop()

    # --------------------------------------------------------
    # Construção das janelas móveis
    # --------------------------------------------------------
    def contar_hits(car_a, car_b):
        return len(set(car_a).intersection(set(car_b)))

    resultados = []

    for t in range(n - W - 1, n - 1):
        janela = historico[t - W + 1 : t + 1]
        alvo = historico[t + 1]

        for car in janela:
            hits = contar_hits(car, alvo)
            resultados.append({
                "passageiros": car,
                "hit2": 1 if hits >= 2 else 0,
                "hit3": 1 if hits >= 3 else 0,
            })

    df = pd.DataFrame(resultados)

    universo = sorted({p for car in df["passageiros"] for p in car})

    linhas = []

    for p in universo:
        presente = df["passageiros"].apply(lambda x: p in x)

        A = int(presente.sum())
        B = int((~presente).sum())

        if A < AMIN or B < BMIN:
            classe = "INSUFICIENTE"
        else:
            a3 = df.loc[presente, "hit3"].sum()
            b3 = df.loc[~presente, "hit3"].sum()

            p1 = (a3 + ALPHA) / (A + 2 * ALPHA)
            p0 = (b3 + ALPHA) / (B + 2 * ALPHA)

            delta = p1 - p0
            lift = p1 / p0 if p0 > 0 else 1.0

            if delta < 0 and lift <= 0.92:
                classe = "NOCIVO CONSISTENTE"
            else:
                classe = "NEUTRO"

        linhas.append({
            "passageiro": p,
            "A_presente": A,
            "B_ausente": B,
            "classe": classe,
        })

    df_out = pd.DataFrame(linhas).sort_values("classe")

    st.markdown("### 🧾 Classificação de Passageiros")
    st.dataframe(df_out, use_container_width=True, hide_index=True)

    st.markdown(
        """
🧠 **Como usar este painel**
- Passageiros **NOCIVOS CONSISTENTES** são candidatos a **EVITAR** no Modo 6
- Não é corte automático
- Serve para **limpar listas**, não para criar novas
"""
    )

# ============================================================
# PAINEL — 🌐 MODO UNIVERSAL — AVALIAÇÃO OBSERVACIONAL
# (N-AGNÓSTICO • SEM DECISÃO • SEM GERAÇÃO)
# ============================================================

if painel == "🌐 Modo Universal — Avaliação Observacional":

    st.title("🌐 Modo Universal — Avaliação Observacional")
    st.caption(
        "Observacional • N-agnóstico • Sem geração • Sem decisão\n"
        "Avalia listas existentes contra o alvo real (n_real)."
    )

    df = st.session_state.get("historico_df")
    n_real = st.session_state.get("n_alvo")
    listas = st.session_state.get("modo6_listas_totais") or []

    if df is None or n_real is None:
        st.warning(
            "Histórico ou n_real não disponível.\n\n"
            "Carregue o histórico antes de usar este painel."
        )
        st.stop()

    # -----------------------------
    # Alvo real (última série válida)
    # -----------------------------
    col_pass = [c for c in df.columns if c.startswith("p")]
    alvo_real = (
        df[col_pass]
        .iloc[-1]
        .dropna()
        .astype(int)
        .tolist()
    )

    if not listas:
        st.info("Nenhuma lista disponível para avaliação.")
        st.stop()

    # -----------------------------
    # Orçamento manual (opcional)
    # -----------------------------
    st.subheader("🔢 Orçamento (opcional)")
    orcamento_manual = st.text_input(
        "Informe um orçamento manual (opcional)",
        value="",
        help="Se preenchido, substitui a tabela condicionada."
    )
    if orcamento_manual == "":
        orcamento_manual = None

    # -----------------------------
    # Avaliação observacional
    # -----------------------------
    resultados = avaliar_listas_universal(
        listas=listas,
        alvo_real=alvo_real,
        n_real=n_real,
        orcamento_manual=orcamento_manual,
    )

    if not resultados:
        st.info("Nenhuma lista válida para avaliação (listas < n_real são ignoradas).")
        st.stop()

    df_out = pd.DataFrame(resultados)

    st.subheader("📊 Resultados (acertos / n_real)")
    st.dataframe(df_out, use_container_width=True, hide_index=True)

    st.caption(
        "Leitura sempre relativa ao n_real.\n"
        "Listas com tamanho menor que n_real são descartadas automaticamente."
    )



# ============================================================
# PAINEL V16 PREMIUM — BACKTEST RÁPIDO DO PACOTE (N = 60)
# ============================================================
if painel == "📊 V16 Premium — Backtest Rápido do Pacote (N=60)":

    st.subheader("📊 V16 Premium — Backtest Rápido do Pacote (N = 60)")
    st.caption(
        "Ensaio estatístico do pacote ATUAL de listas sobre os últimos 60 alvos. "
        "Não é previsão. Não decide volume. Mede apenas resistência sob pressão."
    )

    # ------------------------------------------------------------
    # Recuperação segura do histórico
    # ------------------------------------------------------------
    historico_df = st.session_state.get("historico_df")

    if historico_df is None or historico_df.empty:
        st.warning("Histórico não encontrado. Carregue o histórico antes.")
        st.stop()

    if historico_df.shape[0] < 60:
        st.warning("Histórico insuficiente para backtest (mínimo: 60 séries).")
        st.stop()

    # ------------------------------------------------------------
    # Recuperação do pacote congelado
    # ------------------------------------------------------------
    pacote = st.session_state.get("pacote_listas_atual")

    if not pacote:
        st.warning("Nenhum pacote de listas foi registrado ainda.")
        st.stop()

    # ------------------------------------------------------------
    # Identificação das colunas de passageiros
    # ------------------------------------------------------------
    colunas_passageiros = [c for c in historico_df.columns if c.lower().startswith("p")]

    if not colunas_passageiros:
        st.error("Não foi possível identificar colunas de passageiros no histórico.")
        st.stop()

    # ------------------------------------------------------------
    # Preparação do histórico (últimos 60 alvos)
    # ------------------------------------------------------------
    ultimos_60 = historico_df.tail(60)

    resultados = {
        ">=3": 0,
        ">=4": 0,
        ">=5": 0,
        ">=6": 0,
    }

    total_testes = 0

    # ------------------------------------------------------------
    # Execução do backtest
    # ------------------------------------------------------------
    for _, linha in ultimos_60.iterrows():

        # Alvo reconstruído a partir das colunas reais
        alvo = set(int(linha[c]) for c in colunas_passageiros if pd.notna(linha[c]))

        for lista in pacote:
            acertos = len(set(lista) & alvo)
            total_testes += 1

            if acertos >= 3:
                resultados[">=3"] += 1
            if acertos >= 4:
                resultados[">=4"] += 1
            if acertos >= 5:
                resultados[">=5"] += 1
            if acertos >= 6:
                resultados[">=6"] += 1

    # ------------------------------------------------------------
    # Cálculo das porcentagens
    # ------------------------------------------------------------
    perc = {
        k: (v / total_testes) * 100 if total_testes > 0 else 0.0
        for k, v in resultados.items()
    }

    # ------------------------------------------------------------
    # Exibição
    # ------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("≥ 3 acertos", f"{perc['>=3']:.2f}%")
    col2.metric("≥ 4 acertos", f"{perc['>=4']:.2f}%")
    col3.metric("≥ 5 acertos", f"{perc['>=5']:.2f}%")
    col4.metric("≥ 6 acertos", f"{resultados['>=6']} ocorrências")

    st.info(
        "📌 Interpretação correta:\n"
        "- Percentuais baixos indicam palco escorregadio\n"
        "- Percentuais estáveis indicam pacote resiliente\n"
        "- Isso NÃO prevê o próximo alvo\n"
        "- Serve apenas para calibrar postura e volume"
    )


# ============================================================
# PAINEL V16 PREMIUM — RODADAS ESTRATIFICADAS (A/B)
# (Preparação operacional: NÃO ativa motores; NÃO mistura pacotes)
#
# OBJETIVO:
# - Permitir que o operador registre DUAS execuções independentes
#   para o MESMO evento considerado "Bom + Oportunidade Rara".
# - Rodada A: modelo-base (n=6) — normalmente via Modo 6.
# - Rodada B: modelo alternativo (n≠6) — por colagem manual (por enquanto)
#
# REGRAS:
# - Nunca misturar listas/volumes/decisões.
# - Nunca somar resultados.
# - Registrar e analisar separadamente.
#
# IMPORTANTE:
# - Este painel é 100% opcional.
# - Se nada for marcado, o app se comporta como sempre.
# ============================================================
if painel == "🧭 V16 Premium — Rodadas Estratificadas (A/B)":

    st.subheader("🧭 V16 Premium — Rodadas Estratificadas (A/B)")
    st.caption(
        "Painel de preparação e registro. Não gera listas automaticamente. "
        "Não ativa camadas experimentais. Não muda Modo 6/TURBO. "
        "Serve apenas para organizar duas rodadas independentes no MESMO evento raro."
    )

    # ------------------------------
    # Leitura do momento (somente leitura)
    # ------------------------------
    dmo_estado = st.session_state.get("dmo_estado", "🟥 SOBREVIVÊNCIA")
    eco_status = st.session_state.get("eco_status", st.session_state.get("eco_acionabilidade", "N/D"))
    nr_ruido = st.session_state.get("nr_percent", st.session_state.get("nr_percent_v16"))

    colm1, colm2, colm3 = st.columns(3)
    colm1.metric("DMO", str(dmo_estado))
    colm2.metric("ECO", str(eco_status) if eco_status else "N/D")
    colm3.metric("NR%", f"{nr_ruido:.2f}%" if isinstance(nr_ruido, (int, float)) else "N/D")

    st.markdown("---")

    # ------------------------------
    # Travas (governança)
    # ------------------------------
    st.markdown("### 🔒 Travas (governança)")
    st.caption(
        "Este painel não decide por você. Ele só permite registro A/B quando você confirma conscientemente."
    )

    confirmar_momento_bom = st.checkbox(
        "Confirmo que este evento é: 🟢 Momento Bom + Oportunidade Rara (decisão do operador)",
        value=False,
        key="AB_CONFIRMAR_MOMENTO_BOM_RARO",
    )

    habilitar_rodada_b = st.checkbox(
        "(Opcional) Quero preparar Rodada B (Proteção de Modelo) — execução separada", 
        value=False,
        key="AB_HABILITAR_RODADA_B",
        disabled=(not confirmar_momento_bom),
    )

    # Guarda leve (não bloqueia o operador, só avisa)
    avisos = []
    if isinstance(nr_ruido, (int, float)) and nr_ruido >= 70:
        avisos.append("NR% alto (>=70): cuidado com leitura de momento.")
    if isinstance(dmo_estado, str) and dmo_estado.strip().startswith("🟥"):
        avisos.append("DMO em SOBREVIVÊNCIA: este cenário normalmente não é 'momento bom'.")
    if isinstance(eco_status, str) and eco_status.strip() in ("N/D", "DESCONHECIDO"):
        avisos.append("ECO indefinido: leitura parcial do momento.")

    if avisos:
        st.warning("⚠️ Avisos de governança:\n- " + "\n- ".join(avisos))

    st.markdown("---")

    # ------------------------------
    # Identificação do evento (rótulo humano)
    # ------------------------------
    st.markdown("### 🏷️ Identificação do evento")
    st.caption("Use um rótulo simples (ex.: C5823 / 'Evento Raro Jan-2026').")
    evento_id = st.text_input(
        "Evento (ID/rótulo):",
        value=st.session_state.get("AB_EVENTO_ID", ""),
        key="AB_EVENTO_ID_INPUT",
        disabled=(not confirmar_momento_bom),
    )

    if confirmar_momento_bom and evento_id:
        st.session_state["AB_EVENTO_ID"] = evento_id.strip()

    st.markdown("---")

    # ------------------------------
    # Rodada A — n=6 (captura do pacote atual)
    # ------------------------------
    st.markdown("## 🔵 Rodada A — Estratégia Principal (modelo-base)")
    st.caption(
        "Normalmente você gera o pacote no 🎯 Modo 6. Aqui você apenas registra uma fotografia desse pacote como Rodada A."
    )

    pacote_atual = st.session_state.get("pacote_listas_atual")
    pacote_origem = st.session_state.get("pacote_listas_origem", "N/D")

    if pacote_atual and isinstance(pacote_atual, list):
        st.success(f"Pacote atual detectado: {len(pacote_atual)} lista(s) — origem: {pacote_origem}")
        st.dataframe(pd.DataFrame({"Lista": [str(L) for L in pacote_atual]}), use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum pacote atual detectado. Gere listas no 🎯 Modo 6 para ter algo a registrar aqui.")

    registrar_a = st.button(
        "📦 Registrar Rodada A (capturar pacote atual)",
        disabled=(not (confirmar_momento_bom and evento_id and pacote_atual)),
        key="AB_REGISTRAR_A",
    )

    if registrar_a:
        st.session_state["AB_RODADA_A"] = {
            "evento": evento_id.strip(),
            "origem": pacote_origem,
            "listas": pacote_atual,
            "meta": {
                "dmo": dmo_estado,
                "eco": eco_status,
                "nr": nr_ruido,
            },
        }
        st.success("✅ Rodada A registrada como pacote independente.")

    st.markdown("---")

    # ------------------------------
    # Rodada B — n≠6 (proteção) — por colagem manual
    # ------------------------------
    st.markdown("## 🔴 Rodada B — Proteção de Modelo (execução paralela)")
    st.caption(
        "Esta rodada não complementa a A: ela coexiste como hipótese paralela. "
        "Por enquanto, o registro B é por colagem manual (listas 7/8/9/10)."
    )

    def _ab_parse_listas_texto(txt: str):
        """Parse simples: aceita linhas com números separados por vírgula, espaço ou ';'."""
        import re
        listas = []
        if not txt:
            return listas
        for linha in txt.splitlines():
            linha = linha.strip()
            if not linha:
                continue
            nums = re.findall(r"\d+", linha)
            if not nums:
                continue
            listas.append([int(x) for x in nums])
        return listas

    n_b = st.number_input(
        "n da Rodada B (tamanho esperado da lista):",
        min_value=7,
        max_value=12,
        value=int(st.session_state.get("AB_N_B", 7)),
        step=1,
        disabled=(not (confirmar_momento_bom and habilitar_rodada_b)),
        key="AB_N_B_INPUT",
    )
    if confirmar_momento_bom and habilitar_rodada_b:
        st.session_state["AB_N_B"] = int(n_b)

    texto_b = st.text_area(
        "Cole aqui as listas da Rodada B (uma por linha):",
        value=st.session_state.get("AB_LISTAS_B_TEXTO", ""),
        height=180,
        disabled=(not (confirmar_momento_bom and habilitar_rodada_b and evento_id)),
        key="AB_LISTAS_B_TEXTO",
    )

    listas_b = _ab_parse_listas_texto(texto_b)
    listas_b_validas = [L for L in listas_b if isinstance(L, list) and len(L) == int(n_b)]
    invalidas = len(listas_b) - len(listas_b_validas)

    if confirmar_momento_bom and habilitar_rodada_b:
        st.info(
            f"Leitura da Rodada B: {len(listas_b_validas)} lista(s) válidas (n={int(n_b)})"
            + (f" · {invalidas} inválida(s) (tamanho diferente)." if invalidas else "")
        )

    registrar_b = st.button(
        "📦 Registrar Rodada B (proteção) — pacote independente",
        disabled=(not (confirmar_momento_bom and habilitar_rodada_b and evento_id and len(listas_b_validas) > 0)),
        key="AB_REGISTRAR_B",
    )

    if registrar_b:
        st.session_state["AB_RODADA_B"] = {
            "evento": evento_id.strip(),
            "n": int(n_b),
            "origem": "Colagem manual (Rodada B)",
            "listas": listas_b_validas,
            "meta": {
                "dmo": dmo_estado,
                "eco": eco_status,
                "nr": nr_ruido,
            },
        }
        st.success("✅ Rodada B registrada como pacote independente.")

    st.markdown("---")

    # ------------------------------
    # Resumo final (A/B) — sem soma, sem mistura
    # ------------------------------
    st.markdown("### 📌 Resumo A/B (sem mistura)")
    a = st.session_state.get("AB_RODADA_A")
    b = st.session_state.get("AB_RODADA_B")

    colr1, colr2 = st.columns(2)
    with colr1:
        st.markdown("**Rodada A**")
        if a:
            st.success(f"Evento: {a.get('evento')} · listas: {len(a.get('listas') or [])} · origem: {a.get('origem')}")
        else:
            st.info("Ainda não registrada.")
    with colr2:
        st.markdown("**Rodada B**")
        if b:
            st.success(f"Evento: {b.get('evento')} · n={b.get('n')} · listas: {len(b.get('listas') or [])}")
        else:
            st.info("Ainda não registrada (opcional).")

    st.caption(
        "Regra canônica: A e B são pacotes distintos. Não somar volumes. "
        "Não interpretar como um único ataque. Replay/Backtest devem ser feitos separadamente."
    )


# ============================================================
# PAINEL — 🧠 M5 — PULO DO GATO (COLETA AUTOMÁTICA DE ESTADOS)
# ============================================================
if painel == M5_PAINEL_PULO_GATO_NOME:
    m5_painel_pulo_do_gato_operacional()


# ============================================================
# PAINEL — 📈 EXPECTATIVA HISTÓRICA — CONTEXTO DO MOMENTO (V16)
# ============================================================
if painel == M3_PAINEL_EXPECTATIVA_NOME:
    v16_painel_expectativa_historica_contexto()


# ============================================================
# PAINEL — 🧭 CHECKLIST OPERACIONAL — DECISÃO (AGORA)
# ============================================================
if painel == "🧭 Checklist Operacional — Decisão (AGORA)":

    st.markdown("## 🧭 Checklist Operacional — Decisão (AGORA)")
    st.caption(
        "Checklist obrigatório ANTES do Modo 6 / Mandar Bala.\n"
        "Não calcula, não cria listas, não decide automaticamente."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # 1) Estrada
    # --------------------------------------------------------
    st.markdown("### 1️⃣ Estrada permite ataque?")
    st.markdown(
        "- k* **não piorou**\n"
        "- NR% **não explodiu**\n"
        "- Divergência **não disparou**"
    )
    estrada_ok = st.radio(
        "Resultado da leitura da estrada:",
        ["SIM", "NÃO"],
        horizontal=True,
        key="CHECKLIST_ESTRADA_OK",
    )

    # --------------------------------------------------------
    # 2) Regime
    # --------------------------------------------------------
    st.markdown("### 2️⃣ Regime jogável?")
    regime = st.radio(
        "Regime identificado:",
        ["OURO", "PRATA", "RUIM"],
        horizontal=True,
        key="CHECKLIST_REGIME",
    )

    # --------------------------------------------------------
    # 3) Eixo
    # --------------------------------------------------------
    st.markdown("### 3️⃣ Existe eixo claro nas listas?")
    eixo = st.radio(
        "Eixo identificado:",
        ["SIM", "NÃO"],
        horizontal=True,
        key="CHECKLIST_EIXO",
    )

    # --------------------------------------------------------
    # 4) Nocivos
    # --------------------------------------------------------
    st.markdown("### 4️⃣ Nocivos concentrados nas mesmas listas?")
    nocivos = st.radio(
        "Nocivos:",
        ["SIM", "NÃO"],
        horizontal=True,
        key="CHECKLIST_NOCIVOS",
    )

    st.markdown("---")

    # --------------------------------------------------------
    # 5) Decisão humana
    # --------------------------------------------------------
    st.markdown("### 5️⃣ Decisão final (humana)")
    acao = st.radio(
        "Ação escolhida:",
        [
            "CONCENTRAR (6–8 listas)",
            "EQUILIBRAR (8–10 listas)",
            "EXPANDIR COM CRITÉRIO (10–12 listas)",
            "SEGURAR / NÃO ESCALAR",
        ],
        key="CHECKLIST_ACAO",
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Síntese
    # --------------------------------------------------------
    st.markdown("### 🧾 Síntese da decisão")
    st.write(
        {
            "Estrada OK": estrada_ok,
            "Regime": regime,
            "Eixo": eixo,
            "Nocivos concentrados": nocivos,
            "Ação escolhida": acao,
        }
    )

    st.success(
        "Checklist concluído. "
        "A decisão da rodada está FECHADA aqui. "
        "Prossiga para o Modo 6 e execução."
    )


# ============================================================
# PAINEL — 🧭 MODO GUIADO OFICIAL — PREDICTCARS
# ============================================================
if painel == "🧭 Modo Guiado Oficial — PredictCars":

    st.markdown("## 🧭 Modo Guiado Oficial — PredictCars")
    st.caption(
        "Guia operacional único · uso diário · contrato de uso do sistema.\n"
        "Não executa, não calcula, não decide — apenas orienta a sequência correta."
    )

    st.markdown("---")

    st.markdown("""
🧭 **MODO GUIADO OFICIAL — CONTRATO OPERACIONAL**

Este painel descreve **COMO o PredictCars deve ser usado**.
Ele existe para evitar decisões fora de ordem e misturas perigosas
entre leitura, decisão, execução e aprendizado.

━━━━━━━━━━━━━━━━━━━━
🔵 **AGORA — DECIDIR E JOGAR**
━━━━━━━━━━━━━━━━━━━━

**1️⃣ ENTRADA**
- 📁 Carregar Histórico (Arquivo ou Colar)

**2️⃣ EIXO 1 — ESTRUTURA DO AMBIENTE**
*(saúde da estrada · não números)*

Painéis:
- 🛰️ Sentinelas — k*
- 🧭 Monitor de Risco — k & k*
- 📡 Painel de Ruído Condicional
- 📉 Painel de Divergência S6 vs MC

Pergunta respondida:
- O ambiente permite ataque?

# ---

**3️⃣ EIXO 2 — MOMENTO & ANTECIPAÇÃO**
*(ritmo do alvo + evidência recente)*

Painéis:
- 🔁 Replay LIGHT
- 🔁 Replay ULTRA
- 🧪 Replay Curto — Expectativa 1–3 Séries
- 📊 V16 Premium — Backtest Rápido do Pacote (N=60)

Pergunta respondida:
- O momento favorece agir agora?

# ---

**4️⃣ DECISÃO ÚNICA (HUMANA)**
*(registrada · sem retorno)*

Painel:
- 🧭 Checklist Operacional — Decisão (AGORA)

Aqui você define:
- atacar ou não
- concentrar, equilibrar ou expandir
- volume de listas

📌 **Depois disso, não se volta atrás.**

# ---

**5️⃣ MOTOR**
- 🛣️ Pipeline V14-FLEX ULTRA
- ⚙️ Modo TURBO++ HÍBRIDO
- ⚙️ Modo TURBO++ ULTRA

# ---

**6️⃣ EXECUÇÃO**
- 🎯 Modo 6 Acertos — Execução
- 🧪 Testes de Confiabilidade REAL
- 📘 Relatório Final
- 🔥 Mandar Bala

━━━━━━━━━━━━━━━━━━━━
🟣 **EXTENSÃO CONDICIONAL — MODO ESPECIAL**
━━━━━━━━━━━━━━━━━━━━

Use **somente após** concluir o fluxo acima.

- 🔵 MODO ESPECIAL — Evento Condicionado
- Atua sobre pacotes já gerados
- Não cria listas novas
- Útil apenas para eventos únicos

━━━━━━━━━━━━━━━━━━━━
🟢 **DEPOIS — APRENDER**
━━━━━━━━━━━━━━━━━━━━

Painéis:
- 📊 V16 Premium — ANTI-EXATO | Passageiros Nocivos
- 📊 EXATO / ECO / Regime

📌 Aprendizado **somente para a próxima rodada**.

━━━━━━━━━━━━━━━━━━━━
🧱 **OS 3 EIXOS DO SISTEMA**
━━━━━━━━━━━━━━━━━━━━

- **Eixo 1** — Estrutura das Listas  
- **Eixo 2** — Momento & Antecipação  
- **Eixo 3** — Aprendizado  

━━━━━━━━━━━━━━━━━━━━
📜 **REGRA FINAL**
━━━━━━━━━━━━━━━━━━━━

A decisão acontece **ANTES**.  
O aprendizado acontece **DEPOIS**.  
**Nunca ao mesmo tempo.**
""")

    st.success(
        "Modo Guiado carregado com sucesso.\n"
        "Este painel é o contrato oficial de uso do PredictCars."
    )

# ============================================================
# <<< FIM — PAINEL 🧭 MODO GUIADO OFICIAL — PREDICTCARS
# ============================================================





# ============================================================
# ROTEADOR V16 PREMIUM — EXECUÇÃO DOS PAINÉIS (DEFINITIVO)
# ============================================================

# ------------------------------------------------------------
# ORIENTAÇÃO / USO
# ------------------------------------------------------------
if painel == "🧭 Modo Guiado Oficial — PredictCars":
    st.markdown("## 🧭 Modo Guiado Oficial — PredictCars")
    st.info(
        "Este painel apenas orienta o uso correto do sistema.\n"
        "Siga a sequência indicada no menu."
    )
    st.stop()

# ------------------------------------------------------------
# GOVERNANÇA / VISIBILIDADE (M1)
# ------------------------------------------------------------
if painel == "🔍 Diagnóstico Espelho (Mirror)":
    _m1_render_mirror_panel()
    st.stop()

# ------------------------------------------------------------
# DECISÃO OPERACIONAL (AGORA)
# ------------------------------------------------------------
if painel == "🧭 Checklist Operacional — Decisão (AGORA)":
    st.markdown("## 🧭 Checklist Operacional — Decisão (AGORA)")
    st.caption(
        "Checklist obrigatório ANTES do Modo 6 / Mandar Bala.\n"
        "Não calcula, não cria listas, não decide automaticamente."
    )

    st.markdown("---")

    st.markdown("### 1️⃣ Estrada permite ataque?")
    estrada_ok = st.radio(
        "Resultado da leitura da estrada:",
        ["SIM", "NÃO"],
        horizontal=True,
    )

    st.markdown("### 2️⃣ Regime jogável?")
    regime = st.radio(
        "Regime identificado:",
        ["OURO", "PRATA", "RUIM"],
        horizontal=True,
    )

    st.markdown("### 3️⃣ Existe eixo claro nas listas?")
    eixo = st.radio(
        "Eixo identificado:",
        ["SIM", "NÃO"],
        horizontal=True,
    )

    st.markdown("### 4️⃣ Nocivos concentrados nas mesmas listas?")
    nocivos = st.radio(
        "Nocivos:",
        ["SIM", "NÃO"],
        horizontal=True,
    )

    st.markdown("### 5️⃣ Decisão final (humana)")
    acao = st.radio(
        "Ação escolhida:",
        [
            "CONCENTRAR (6–8 listas)",
            "EQUILIBRAR (8–10 listas)",
            "EXPANDIR COM CRITÉRIO (10–12 listas)",
            "SEGURAR / NÃO ESCALAR",
        ],
    )

    st.markdown("---")
    st.markdown("### 🧾 Síntese da decisão")
    st.write(
        {
            "Estrada OK": estrada_ok,
            "Regime": regime,
            "Eixo": eixo,
            "Nocivos concentrados": nocivos,
            "Ação escolhida": acao,
        }
    )

    st.success(
        "Checklist concluído. "
        "A decisão da rodada está FECHADA aqui. "
        "Prossiga para o Modo 6 e execução."
    )
    st.stop()

# ------------------------------------------------------------
# EIXO 2 — MOMENTO & ANTECIPAÇÃO
# ------------------------------------------------------------
if painel == "📊 V16 Premium — Backtest Rápido do Pacote (N=60)":
    st.markdown("## 📊 Backtest Rápido do Pacote (N=60)")
    st.caption(
        "Avaliação observacional do pacote atual.\n"
        "Não decide. Apenas informa."
    )
    st.stop()

# ------------------------------------------------------------
# RITMO DO ALVO (SUPORTE)
# ------------------------------------------------------------
if painel == "🧪 Replay Curto — Expectativa 1–3 Séries":
    st.markdown("## 🧪 Replay Curto — Expectativa 1–3 Séries")
    st.stop()

if painel == "⏱️ Duração da Janela — Análise Histórica":
    st.markdown("## ⏱️ Duração da Janela — Análise Histórica")
    st.stop()

# ------------------------------------------------------------
# V16 PREMIUM — APRENDIZADO (DEPOIS)
# ------------------------------------------------------------

if painel == "🧠 Laudo Operacional V16":
    # ROTA DUPLICADA DESATIVADA (há um painel Laudo V16 completo em outro ponto do app)
    # Mantemos apenas UM ponto de execução do Laudo para evitar dupla execução/deriva.
    st.stop()


if painel == "📊 V16 Premium — Erro por Regime (Retrospectivo)":
    v16_painel_erro_por_regime_retrospectivo()
    st.stop()

if painel == "📊 V16 Premium — EXATO por Regime (Proxy)":
    v16_painel_exato_por_regime_proxy()
    st.stop()

if painel == "📊 V16 Premium — PRÉ-ECO → ECO (Persistência & Continuidade)":
    v16_painel_pre_eco_persistencia_continuidade()
    st.stop()

if painel == "📊 V16 Premium — Passageiros Recorrentes em ECO (Interseção)":
    v16_painel_passageiros_recorrentes_eco_intersecao()
    st.stop()

if painel == "🎯 Compressão do Alvo — Observacional (V16)":
    v16_painel_compressao_alvo()
    st.stop()

if painel == "🔮 V16 Premium Profundo — Diagnóstico & Calibração":
    v16_painel_premium_profundo()
    st.stop()

# ============================================================
# FIM DO ROTEADOR V16 PREMIUM — EXECUÇÃO DOS PAINÉIS
# ============================================================
