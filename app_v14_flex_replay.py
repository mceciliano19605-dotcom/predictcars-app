import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

# ============================================================
# CONFIGURAÇÃO BÁSICA DO APP
# ============================================================

st.set_page_config(
    page_title="Predict Cars V14-FLEX REPLAY",
    layout="wide",
)

st.markdown("""
# Predict Cars V14-FLEX REPLAY
Versão FLEX: número variável de passageiros + modo replay automático + validação empírica.
""")

# ============================================================
# FUNÇÕES AUXILIARES BÁSICAS
# ============================================================

NUM_MIN = 1
NUM_MAX = 60


def _coerce_int(x: Any) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return 0


def preparar_historico_V14(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Converte um DataFrame cru em formato padrão V14-FLEX.

    Saída: colunas ["id", "passageiros", "k", "n_passageiros"].
    - id: string (ex: "C2943")
    - passageiros: list[int]
    - k: int
    - n_passageiros: int
    """
    linhas = []
    for i, row in enumerate(df_raw.itertuples(index=False)):  # type: ignore
        valores = list(row)
        if not valores:
            continue

        primeiro = valores[0]
        # Detectar id: se não for puramente numérico, usar como id; senão gerar C{idx}
        s0 = str(primeiro).strip()
        if s0 and not s0.isdigit():
            id_serie = s0
            resto = valores[1:]
        else:
            id_serie = f"C{i+1}"
            resto = valores

        if len(resto) < 2:
            # precisa de pelo menos 1 passageiro + k
            continue

        k_val = _coerce_int(resto[-1])
        passageiros = [_coerce_int(x) for x in resto[:-1]]
        passageiros = [p for p in passageiros if p > 0]

        if not passageiros:
            continue

        linhas.append(
            {
                "id": id_serie,
                "passageiros": passageiros,
                "k": k_val,
                "n_passageiros": len(passageiros),
            }
        )

    df = pd.DataFrame(linhas)

    # Garantir ordenação crescente pelo índice numérico dentro do id (se possível)
    def extrair_idx(id_str: str) -> int:
        s = str(id_str)
        for ch in ["C", "c", "#", ":", ";", " "]:
            s = s.replace(ch, " ")
        tokens = [t for t in s.split() if t.isdigit()]
        return int(tokens[-1]) if tokens else 0

    if not df.empty:
        df["idx_numeric"] = df["id"].apply(extrair_idx)
        df = df.sort_values("idx_numeric").reset_index(drop=True)

    return df


def preparar_historico_de_texto(texto: str) -> pd.DataFrame:
    """Parser de histórico quando o usuário cola texto.

    Aceita linhas no formato:
    C1;41;5;4;52;30;33;0
    ou
    41;5;4;52;30;33;0
    ou com vírgulas.
    """
    linhas = []
    for i, raw_line in enumerate(texto.splitlines()):
        linha = raw_line.strip()
        if not linha:
            continue

        if ";" in linha:
            partes = [p.strip() for p in linha.split(";")]
        elif "," in linha:
            partes = [p.strip() for p in linha.split(",")]
        else:
            partes = [p.strip() for p in linha.split()]

        if not partes:
            continue

        s0 = partes[0]
        if s0 and not s0.isdigit():
            id_serie = s0
            resto = partes[1:]
        else:
            id_serie = f"C{i+1}"
            resto = partes

        if len(resto) < 2:
            continue

        k_val = _coerce_int(resto[-1])
        passageiros = [_coerce_int(x) for x in resto[:-1]]
        passageiros = [p for p in passageiros if p > 0]
        if not passageiros:
            continue

        linhas.append(
            {
                "id": id_serie,
                "passageiros": passageiros,
                "k": k_val,
                "n_passageiros": len(passageiros),
            }
        )

    df = pd.DataFrame(linhas)

    if not df.empty:
        df["idx_numeric"] = range(1, len(df) + 1)

    return df


# ============================================================
# MÓDULO DE RISCO (k e k*)
# ============================================================

def avaliar_risco_k(df: pd.DataFrame) -> Tuple[str, str]:
    """Avalia o ambiente de risco a partir da coluna k.

    Retorna (descricao_k, descricao_k_star).
    """
    if df.empty or "k" not in df.columns:
        return (
            "⚠️ k histórico da série alvo\nDados insuficientes para avaliar risco.",
            "⚡ k* (sentinela preditivo TURBO++)\nDados insuficientes para projetar risco.",
        )

    # k histórico da última série
    k_ultimo = int(df.iloc[-1]["k"])
    if k_ultimo <= 0:
        desc_k = "⚠️ k histórico da série alvo\n🟢 Ambiente estável — previsão em regime normal."
    elif k_ultimo == 1:
        desc_k = "⚠️ k histórico da série alvo\n🟡 Ambiente em atenção — sinais de pré-ruptura local."
    else:
        desc_k = "⚠️ k histórico da série alvo\n🔴 Ambiente crítico — trecho turbulento da estrada."

    # k* preditivo — baseado na frequência de k>0 nas últimas N séries
    n_janela = min(50, len(df))
    sub = df.tail(n_janela)
    if n_janela == 0:
        return desc_k, "⚡ k* (sentinela preditivo TURBO++)\nDados insuficientes para projetar risco."

    proporcao_eventos = float((sub["k"] > 0).mean())
    risco_pct = int(round(100 * proporcao_eventos))

    if risco_pct <= 15:
        desc_k_star = (
            f"⚡ k* (sentinela preditivo TURBO++)\n"
            f"🟢 k*: Ambiente tende a permanecer estável (risco ≈ {risco_pct}%)."
        )
    elif risco_pct <= 40:
        desc_k_star = (
            f"⚡ k* (sentinela preditivo TURBO++)\n"
            f"🟡 k*: Ambiente com ruído moderado (risco ≈ {risco_pct}%)."
        )
    else:
        desc_k_star = (
            f"⚡ k* (sentinela preditivo TURBO++)\n"
            f"🔴 k*: Ambiente com alta turbulência (risco ≈ {risco_pct}%)."
        )

    return desc_k, desc_k_star


# ============================================================
# MÓDULO V14-FLEX — LEQUES E SAÍDA FINAL (MODO E)
# ============================================================

def extrair_contexto(df: pd.DataFrame, idx_alvo: int, janela: int = 30) -> pd.DataFrame:
    """Extrai uma janela de contexto antes da série alvo.

    idx_alvo é 1-based (C1, C2...). A série alvo não entra no contexto.
    """
    if df.empty:
        return df

    idx_alvo = int(idx_alvo)
    # converter para zero-based
    pos = max(0, min(len(df) - 1, idx_alvo - 1))
    inicio = max(0, pos - janela)
    return df.iloc[inicio:pos].copy()


def gerar_leque_original(contexto: pd.DataFrame) -> List[int]:
    """Leque ORIGINAL (IPF bruto).

    Aqui usamos uma heurística inspirada no IPF:
    - conta frequências dos passageiros na janela de contexto
    - seleciona os mais frequentes
    - garante ordenação crescente
    """
    if contexto.empty:
        return []

    contagem: Dict[int, int] = {}
    for passageiros in contexto["passageiros"]:
        for p in passageiros:
            if NUM_MIN <= p <= NUM_MAX:
                contagem[p] = contagem.get(p, 0) + 1

    if not contagem:
        return []

    # ordena por frequência (desc) e por número (asc)
    ordenado = sorted(contagem.items(), key=lambda kv: (-kv[1], kv[0]))
    # pega até 25 números para o leque bruto
    numeros = [n for n, _ in ordenado[:25]]
    numeros = sorted(set(numeros))
    return numeros


def gerar_leque_corrigido(contexto: pd.DataFrame, leque_original: List[int]) -> List[int]:
    """Leque CORRIGIDO (IPO simplificado).

    Pequena correção estrutural:
    - reforça números que aparecem nas séries mais recentes
    - remove números muito raros
    """
    if contexto.empty:
        return leque_original

    if not leque_original:
        return leque_original

    # janela curta recente
    recente = contexto.tail(min(10, len(contexto)))
    contagem_recente: Dict[int, int] = {}
    for passageiros in recente["passageiros"]:
        for p in passageiros:
            if NUM_MIN <= p <= NUM_MAX:
                contagem_recente[p] = contagem_recente.get(p, 0) + 1

    # mantém apenas números do leque original que não sejam extremamente raros
    filtrado = []
    for n in leque_original:
        freq = contagem_recente.get(n, 0)
        if freq >= 1:
            filtrado.append(n)

    # se ficar vazio, volta para o original
    if not filtrado:
        filtrado = leque_original

    return sorted(set(filtrado))


def gerar_leque_misto(leque_original: List[int], leque_corrigido: List[int]) -> List[int]:
    """S6 Profundo — Leque MISTO (achado e ranqueado).

    Une ORIGINAL + CORRIGIDO de forma simples.
    """
    mix = sorted(set(leque_original) | set(leque_corrigido))
    return mix


def selecionar_serie_final_modo_E(leque_misto: List[int]) -> List[int]:
    """Seleciona a série final (6 passageiros) a partir do Leque MISTO.

    Modo E — MIX Inteligente (A + B + D), versão determinística:
    - A: respeita a estrutura central do leque (remove extremos quando houver muitos números)
    - B: espalha os números ao longo do leque (evita aglomerar em uma faixa só)
    - D: suaviza ruído ignorando duplicações / artefatos
    """
    if not leque_misto:
        return []

    numeros = sorted(set(leque_misto))
    n = len(numeros)

    if n <= 6:
        return numeros

    # A: recorte central (remove ~20% das pontas se houver muitos números)
    if n > 10:
        corte = max(1, int(0.2 * n))
        numeros_centro = numeros[corte:-corte]
        if len(numeros_centro) >= 6:
            numeros = numeros_centro
            n = len(numeros)

    # B + D: espalhar posições ao longo do leque de forma determinística
    indices_relativos = [0.12, 0.3, 0.48, 0.62, 0.78, 0.9]
    escolhidos: List[int] = []
    usados: set = set()

    for rel in indices_relativos:
        idx = int(round(rel * (n - 1)))
        idx = max(0, min(n - 1, idx))
        valor = numeros[idx]
        if valor not in usados:
            escolhidos.append(valor)
            usados.add(valor)

    # Se por algum motivo pegamos menos de 6 (colisões), completar sequencialmente
    if len(escolhidos) < 6:
        for v in numeros:
            if v not in usados:
                escolhidos.append(v)
                usados.add(v)
                if len(escolhidos) == 6:
                    break

    escolhidos = sorted(escolhidos)
    return escolhidos


def executar_pipeline_v14_flex(df: pd.DataFrame, idx_alvo: int) -> Dict[str, Any]:
    """Executa o pipeline V14-FLEX completo para uma série alvo.

    Retorna dict com:
    - id_alvo
    - passageiros_alvo
    - k_alvo
    - leque_original
    - leque_corrigido
    - leque_misto
    - leque_final (TURBO++ FLEX)
    - serie_final (lista com 6 passageiros)
    - desc_k
    - desc_k_star
    """
    if df.empty:
        return {}

    idx_alvo = int(idx_alvo)
    pos = max(0, min(len(df) - 1, idx_alvo - 1))
    alvo_row = df.iloc[pos]

    contexto = extrair_contexto(df, idx_alvo)
    leque_original = gerar_leque_original(contexto)
    leque_corrigido = gerar_leque_corrigido(contexto, leque_original)
    leque_misto = gerar_leque_misto(leque_original, leque_corrigido)

    # Núcleo TURBO++ FLEX: aqui usamos o próprio leque_misto como leque bruto final
    leque_final = leque_misto.copy()

    # Série final (modo E)
    serie_final = selecionar_serie_final_modo_E(leque_final)

    desc_k, desc_k_star = avaliar_risco_k(df.iloc[: pos + 1])

    return {
        "id_alvo": alvo_row["id"],
        "passageiros_alvo": alvo_row["passageiros"],
        "k_alvo": int(alvo_row["k"]),
        "leque_original": leque_original,
        "leque_corrigido": leque_corrigido,
        "leque_misto": leque_misto,
        "leque_final": leque_final,
        "serie_final": serie_final,
        "desc_k": desc_k,
        "desc_k_star": desc_k_star,
    }


# ============================================================
# PAINEL 1 — Histórico — Entrada
# ============================================================

def painel_historico_entrada() -> None:
    st.markdown("## 📥 Histórico — Entrada")

    df = st.session_state.get("df")
    if df is not None and not df.empty:
        st.success("Histórico já carregado na sessão.")
        st.dataframe(df[["id", "passageiros", "k", "n_passageiros"]])

    opc = st.radio(
        "Como deseja carregar o histórico?",
        ["Enviar arquivo CSV", "Copiar e colar o histórico"],
    )

    if opc == "Enviar arquivo CSV":
        file = st.file_uploader("Selecione o arquivo CSV:", type=["csv"])
        if file is not None:
            try:
                df_raw = pd.read_csv(file, header=None, sep=None, engine="python")
                df = preparar_historico_V14(df_raw)
                st.session_state["df"] = df
                st.success("Histórico carregado com sucesso!")
                st.dataframe(df[["id", "passageiros", "k", "n_passageiros"]])
            except Exception as e:
                st.error(f"Erro ao carregar CSV: {e}")

    else:  # Copiar e colar
        texto = st.text_area(
            "Cole o histórico aqui (uma série por linha):",
            height=200,
            placeholder="Exemplo:\nC1;41;5;4;52;30;33;0\nC2;9;39;37;49;43;41;1\n...",
        )
        if st.button("Processar histórico colado"):
            if not texto.strip():
                st.warning("Cole algum conteúdo antes de processar.")
            else:
                try:
                    df = preparar_historico_de_texto(texto)
                    if df.empty:
                        st.error("Não foi possível interpretar o histórico.")
                    else:
                        st.session_state["df"] = df
                        st.success("Histórico carregado com sucesso!")
                        st.dataframe(df[["id", "passageiros", "k", "n_passageiros"]])
                except Exception as e:
                    st.error(f"Erro ao processar texto: {e}")


# ============================================================
# PAINEL 2 — Pipeline V14-FLEX (TURBO++)
# ============================================================

def painel_pipeline_v14_flex() -> None:
    st.markdown("## 🔍 Pipeline V14-FLEX (TURBO++)")

    df = st.session_state.get("df")
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Seleção da série alvo")
        idx_min = 1
        idx_max = len(df)
        idx_alvo = st.number_input(
            "Índice alvo (1 = primeira série carregada):",
            min_value=idx_min,
            max_value=idx_max,
            value=idx_max,
            step=1,
        )

        alvo_row = df.iloc[int(idx_alvo) - 1]
        st.markdown(f"**ID alvo:** {alvo_row['id']}")
        st.markdown(f"**Passageiros alvo:** {alvo_row['passageiros']}")
        st.markdown(f"**k alvo:** {int(alvo_row['k'])}")

    with col2:
        st.write("### Configuração da janela")
        janela = st.slider(
            "Janela de contexto (nº de séries anteriores usadas no IPF/IPO):",
            min_value=10,
            max_value=100,
            value=min(30, len(df) - 1 if len(df) > 1 else 10),
            step=1,
        )

    if st.button("Executar Pipeline V14-FLEX TURBO++"):
        with st.spinner("Rodando pipeline V14-FLEX TURBO++..."):
            contexto = extrair_contexto(df, idx_alvo, janela=janela)
            resultado = executar_pipeline_v14_flex(df, idx_alvo)

        if not resultado:
            st.error("Não foi possível executar o pipeline.")
            return

        # Exibir leques
        st.markdown("### 🔧 Estrutura dos Leques")
        st.write("Leque ORIGINAL (IPF bruto)")
        st.code(" ".join(str(x) for x in resultado["leque_original"]), language="text")

        st.write("🔧 Leque CORRIGIDO (IPO simplificado)")
        st.code(" ".join(str(x) for x in resultado["leque_corrigido"]), language="text")

        st.write("🧬 S6 Profundo — Leque MISTO (achado e ranqueado)")
        st.code(" ".join(str(x) for x in resultado["leque_misto"]), language="text")

        st.write("🎯 Núcleo TURBO++ FLEX (previsão bruta do motor)")
        st.code(" ".join(str(x) for x in resultado["leque_final"]), language="text")

        # Risco k e k*
        st.markdown("---")
        st.markdown(resultado["desc_k"])
        st.markdown("")
        st.markdown(resultado["desc_k_star"])

        st.markdown("---")
        st.markdown("### 🎯 Previsão Final TURBO++ FLEX (Modo E)")
        serie_final = resultado["serie_final"]
        if serie_final:
            st.code(" ".join(str(x) for x in serie_final), language="text")
            st.success("Série final gerada com sucesso.")
        else:
            st.warning("Não foi possível gerar a série final a partir do leque.")

        # Guardar último resultado na sessão (para replay / testes)
        st.session_state["ultimo_pipeline"] = {
            "idx_alvo": int(idx_alvo),
            "resultado": resultado,
        }


# ============================================================
# PAINEL 3 — Monitor de Risco (k & k*)
# ============================================================

def painel_monitor_risco() -> None:
    st.markdown("## 🚨 Monitor de Risco (k & k*)")

    df = st.session_state.get("df")
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        return

    desc_k, desc_k_star = avaliar_risco_k(df)

    st.markdown("### Visão geral do risco")
    st.markdown(desc_k)
    st.markdown("")
    st.markdown(desc_k_star)

    st.markdown("---")
    st.markdown("### Distribuição de k no histórico")
    st.write(df[["id", "k"]].reset_index(drop=True))


# ============================================================
# PAINEL 4 — Modo TURBO++ — Painel Completo (resumo do último pipeline)
# ============================================================

def painel_modo_turbo_completo() -> None:
    st.markdown("## 🚀 Modo TURBO++ — Painel Completo")

    df = st.session_state.get("df")
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        return

    ultimo = st.session_state.get("ultimo_pipeline")
    if not ultimo:
        st.info("Execute primeiro o painel '🔍 Pipeline V14-FLEX (TURBO++)' para popular este modo.")
        return

    idx_alvo = ultimo["idx_alvo"]
    resultado = ultimo["resultado"]

    st.markdown(f"### Série alvo: **{resultado['id_alvo']}**")
    st.markdown(f"Passageiros alvo: **{resultado['passageiros_alvo']}**")
    st.markdown(f"k alvo: **{resultado['k_alvo']}**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Leque ORIGINAL (IPF)")
        st.code(" ".join(str(x) for x in resultado["leque_original"]), language="text")

        st.markdown("#### Leque CORRIGIDO (IPO)")
        st.code(" ".join(str(x) for x in resultado["leque_corrigido"]), language="text")

    with col2:
        st.markdown("#### Leque MISTO (S6 Profundo)")
        st.code(" ".join(str(x) for x in resultado["leque_misto"]), language="text")

        st.markdown("#### Núcleo TURBO++ FLEX (leque final)")
        st.code(" ".join(str(x) for x in resultado["leque_final"]), language="text")

    st.markdown("---")
    st.markdown("### 🎯 Previsão Final TURBO++ FLEX (Modo E)")
    serie_final = resultado["serie_final"]
    if serie_final:
        st.code(" ".join(str(x) for x in serie_final), language="text")
    else:
        st.warning("Série final não disponível.")

    st.markdown("---")
    st.markdown("### Contexto de risco")
    st.markdown(resultado["desc_k"])
    st.markdown("")
    st.markdown(resultado["desc_k_star"])


# ============================================================
# PAINEL 5 — Modo Replay Automático do Histórico
# ============================================================

def calcular_acertos(p_real: List[int], p_prev: List[int]) -> int:
    return len(set(p_real) & set(p_prev))


def painel_modo_replay() -> None:
    st.markdown("## 📅 Modo Replay Automático do Histórico")

    df = st.session_state.get("df")
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        return

    st.markdown(
        "Neste modo, o V14-FLEX REPLAY simula previsões ao longo do histórico e mede os acertos."
    )

    idx_min = 2
    idx_max = len(df) - 1 if len(df) > 2 else len(df)
    if idx_max <= idx_min:
        st.warning("Histórico muito curto para replay automático.")
        return

    col1, col2 = st.columns(2)

    with col1:
        inicio = st.number_input(
            "Índice inicial para replay (a previsão será para a próxima série):",
            min_value=idx_min,
            max_value=idx_max,
            value=idx_min,
            step=1,
        )

    with col2:
        fim = st.number_input(
            "Índice final para replay:",
            min_value=inicio,
            max_value=idx_max,
            value=idx_max,
            step=1,
        )

    if st.button("Executar Replay Automático"):
        resultados = []
        total = 0

        with st.spinner("Executando replay ao longo do histórico..."):
            for idx_alvo in range(int(inicio), int(fim) + 1):
                # previsão para a série idx_alvo usando contexto até idx_alvo-1
                resultado = executar_pipeline_v14_flex(df, idx_alvo - 1)
                if not resultado:
                    continue

                # série real alvo = idx_alvo
                real_row = df.iloc[idx_alvo - 1]
                passageiros_reais = list(real_row["passageiros"])
                serie_prev = resultado["serie_final"]
                acertos = calcular_acertos(passageiros_reais, serie_prev)

                resultados.append(
                    {
                        "idx_prev": idx_alvo - 1,
                        "idx_real": idx_alvo,
                        "id_prev": resultado["id_alvo"],
                        "id_real": real_row["id"],
                        "prev": serie_prev,
                        "real": passageiros_reais,
                        "acertos": acertos,
                    }
                )
                total += 1

        if not resultados:
            st.error("Replay não gerou resultados.")
            return

        df_res = pd.DataFrame(resultados)
        st.session_state["replay_resultados"] = df_res

        st.success(f"Replay concluído com {total} previsões.")

        st.markdown("### Amostra de resultados do Replay")
        st.dataframe(df_res.head(50))

        st.markdown("---")
        st.markdown("### Estatísticas rápidas")
        media_acertos = float(df_res["acertos"].mean())
        st.markdown(f"**Média de acertos por série:** {media_acertos:.2f}")

        for n in [2, 3, 4, 5, 6]:
            pct = 100 * float((df_res["acertos"] >= n).mean())
            st.markdown(f"Séries com **≥{n} acertos**: {pct:.1f}%")


# ============================================================
# PAINEL 6 — Testes de Confiabilidade (QDS / Backtest / Monte Carlo — empírico)
# ============================================================

def painel_testes_confiabilidade() -> None:
    st.markdown("## 🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)")

    df_replay = st.session_state.get("replay_resultados")
    if df_replay is None or df_replay.empty:
        st.info(
            "Execute primeiro o '📅 Modo Replay Automático do Histórico' para gerar dados de backtest."
        )
        return

    st.markdown("### Visão geral dos resultados do Replay")
    st.dataframe(df_replay)

    st.markdown("---")
    st.markdown("### Métricas de confiabilidade empírica")

    media_acertos = float(df_replay["acertos"].mean())
    st.markdown(f"**Média de acertos por série:** {media_acertos:.2f}")

    detalhes = []
    for n in [2, 3, 4, 5, 6]:
        pct = 100 * float((df_replay["acertos"] >= n).mean())
        detalhes.append((n, pct))

    for n, pct in detalhes:
        st.markdown(f"- Séries com **≥{n} acertos**: {pct:.1f}%")

    # Classificação simples da confiabilidade
    if media_acertos < 2.0:
        nivel = "Baixa"
        cor = "🔴"
    elif media_acertos < 3.5:
        nivel = "Intermediária"
        cor = "🟡"
    else:
        nivel = "Alta"
        cor = "🟢"

    st.markdown("---")
    st.markdown(f"### {cor} Nível de confiabilidade empírica: **{nivel}**")
    st.markdown(
        "Este painel usa apenas os resultados do Replay (backtest interno) como base para a confiabilidade.\n"
        "Os módulos QDS / Backtest avançado / Monte Carlo podem ser acoplados futuramente em cima destas métricas."
    )


# ============================================================
# ROTEADOR PRINCIPAL DE PAINÉIS
# ============================================================

painel = st.radio(
    "Escolha o painel:",
    [
        "📥 Histórico — Entrada",
        "🔍 Pipeline V14-FLEX (TURBO++)",
        "🚨 Monitor de Risco (k & k*)",
        "🚀 Modo TURBO++ — Painel Completo",
        "📅 Modo Replay Automático do Histórico",
        "🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)",
    ],
)

if painel == "📥 Histórico — Entrada":
    painel_historico_entrada()
elif painel == "🔍 Pipeline V14-FLEX (TURBO++)":
    painel_pipeline_v14_flex()
elif painel == "🚨 Monitor de Risco (k & k*)":
    painel_monitor_risco()
elif painel == "🚀 Modo TURBO++ — Painel Completo":
    painel_modo_turbo_completo()
elif painel == "📅 Modo Replay Automático do Histórico":
    painel_modo_replay()
elif painel == "🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)":
    painel_testes_confiabilidade()
