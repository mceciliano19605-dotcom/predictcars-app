import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------------------------------------
# Configuração geral do app
# -------------------------------------------------------------
st.set_page_config(page_title="Predict Cars V13.8", page_icon="🚗", layout="wide")


# -------------------------------------------------------------
# Funções auxiliares de histórico
# -------------------------------------------------------------
def set_historico(conteudo: str):
    if conteudo is not None and conteudo.strip():
        st.session_state["historico_bruto"] = conteudo


def get_historico():
    return st.session_state.get("historico_bruto", None)


# -------------------------------------------------------------
# Funções auxiliares de parsing e faixas
# -------------------------------------------------------------
def faixa_num(n: int) -> int:
    """Classifica número em faixas: 1-20, 21-40, 41-60, 61-80."""
    if 1 <= n <= 20:
        return 1
    elif 21 <= n <= 40:
        return 2
    elif 41 <= n <= 60:
        return 3
    elif 61 <= n <= 80:
        return 4
    return 0


def extrair_numeros(historico_bruto: str):
    """Extrai todos os passageiros (n1..n6) do histórico (protótipo)."""
    numeros = []
    if not historico_bruto:
        return numeros

    for line in historico_bruto.splitlines():
        line = line.strip()
        if not line:
            continue
        partes = [p.strip() for p in line.split(";") if p.strip()]
        if not partes:
            continue

        # Se começar com Cxxxx, ignora o primeiro campo
        if partes[0].upper().startswith("C"):
            partes = partes[1:]

        # Em geral, último é k (rótulo), então tentamos ignorar
        if len(partes) >= 2:
            possiveis_passageiros = partes[:-1]
        else:
            possiveis_passageiros = partes

        for p in possiveis_passageiros:
            try:
                n = int(p)
                numeros.append(n)
            except ValueError:
                pass

    return numeros


def parse_historico(historico_bruto: str):
    """
    Converte o histórico bruto em uma lista de dicionários:
    {
        'linha': índice,
        'id': 'Cxxxx' ou None,
        'passageiros': [...],
        'k': rótulo final,
        'texto': linha original
    }
    """
    registros = []
    if not historico_bruto:
        return registros

    for idx, line in enumerate(historico_bruto.splitlines()):
        original = line.rstrip("\n")
        line = line.strip()
        if not line:
            continue

        partes = [p.strip() for p in line.split(";") if p.strip()]
        if not partes:
            continue

        id_serie = None
        resto = partes

        if partes[0].upper().startswith("C"):
            id_serie = partes[0]
            resto = partes[1:]

        k = None
        if len(resto) >= 2:
            passageiros_str = resto[:-1]
            k = resto[-1]
        else:
            passageiros_str = resto

        passageiros = []
        for p in passageiros_str:
            try:
                n = int(p)
                passageiros.append(n)
            except ValueError:
                pass

        registros.append(
            {
                "linha": idx,
                "id": id_serie,
                "passageiros": passageiros,
                "k": k,
                "texto": original,
            }
        )
    return registros


def similaridade_faixas(passageiros_alvo, passageiros_cand):
    """Calcula similaridade de faixas entre alvo e candidato (0 a 1)."""
    if not passageiros_alvo or not passageiros_cand:
        return 0.0

    faixas_alvo = [faixa_num(n) for n in passageiros_alvo]
    faixas_cand = [faixa_num(n) for n in passageiros_cand]

    sim = 0
    for f in range(1, 5):
        sim += min(faixas_alvo.count(f), faixas_cand.count(f))

    return sim / max(len(passageiros_alvo), 1)


# -------------------------------------------------------------
# IDX Avançado
# -------------------------------------------------------------
def encontrar_similares_idx_avancado(
    registros, w_coinc=3.0, w_recencia=2.0, w_faixa=1.0
):
    """Retorna tabela IDX, série alvo e núcleo IDX ponderado."""
    if not registros or len(registros) < 2:
        return None, None, None

    alvo = registros[-1]
    alvo_set = set(alvo["passageiros"])
    if not alvo_set:
        return None, alvo, None

    max_linha = max(r["linha"] for r in registros) or 1

    candidatos = []
    for r in registros[:-1]:
        conj = set(r["passageiros"])
        inter = alvo_set.intersection(conj)
        coincidencias = len(inter)
        if coincidencias == 0:
            continue

        recencia_norm = r["linha"] / max_linha
        sim_fx = similaridade_faixas(alvo["passageiros"], r["passageiros"])

        score_total = (
            w_coinc * coincidencias
            + w_recencia * recencia_norm
            + w_faixa * sim_fx
        )

        candidatos.append(
            {
                "linha": r["linha"],
                "id": r["id"],
                "qtd_passageiros": len(r["passageiros"]),
                "coincidentes": coincidencias,
                "recencia_norm": recencia_norm,
                "sim_faixas": sim_fx,
                "score_total": score_total,
                "passageiros": r["passageiros"],
                "texto": r["texto"],
            }
        )

    if not candidatos:
        return None, alvo, None

    df = pd.DataFrame(candidatos)
    df = df.sort_values(
        by=["score_total", "coincidentes", "linha"],
        ascending=[False, False, False],
    )

    # ~20% dos melhores (min 5, max 25)
    num_cand = len(df)
    top_k = int(np.ceil(num_cand * 0.2))
    top_k = max(5, min(25, top_k))
    top_df = df.head(top_k)

    # Núcleo IDX
    pesos_por_numero = {}
    for _, row in top_df.iterrows():
        score = float(row["score_total"])
        for n in row["passageiros"]:
            pesos_por_numero[n] = pesos_por_numero.get(n, 0.0) + score

    if not pesos_por_numero:
        nucleo = None
    else:
        ordenados = sorted(
            pesos_por_numero.items(), key=lambda x: x[1], reverse=True
        )
        nucleo = [n for n, _ in ordenados[:6]]

    return top_df, alvo, nucleo


# -------------------------------------------------------------
# IPF Híbrido
# -------------------------------------------------------------
def calcular_ipf_hibrido(df_top, alvo):
    """Implementa IPF híbrido simples: ritmo, dispersão e pares fixos."""
    if df_top is None or df_top.empty:
        return None, None

    df = df_top.copy()

    # Ritmo = diferença de linhas entre séries vizinhas
    df["ritmo"] = df["linha"].diff().abs().fillna(0)

    # Dispersão = variação da quantidade de passageiros
    df["dispersao"] = df["qtd_passageiros"].rolling(2).std().fillna(0)

    # Pares fixos com o alvo
    alvo_set = set(alvo["passageiros"])
    pares = []
    for row in df["passageiros"]:
        pares.append(len(alvo_set.intersection(set(row))))
    df["pares_fixos"] = pares

    # Score IPF composto
    df["score_ipf"] = (
        df["coincidentes"] * 1.5
        + df["recencia_norm"] * 1.2
        + df["sim_faixas"] * 1.0
        + df["pares_fixos"] * 0.8
        - df["ritmo"] * 0.3
        - df["dispersao"] * 0.2
    )

    df_ipf = df.sort_values(by="score_ipf", ascending=False).head(12)

    # Núcleo IPF
    pesos_num = {}
    for _, r in df_ipf.iterrows():
        for n in r["passageiros"]:
            pesos_num[n] = pesos_num.get(n, 0.0) + float(r["score_ipf"])

    ordenados = sorted(pesos_num.items(), key=lambda x: x[1], reverse=True)
    nucleo_ipf = [n for n, _ in ordenados[:6]]

    return df_ipf, nucleo_ipf


# -------------------------------------------------------------
# IPO Profissional
# -------------------------------------------------------------
def calcular_ipo_profissional(df_top):
    """Implementa IPO profissional com suavização e microcorreção."""
    if df_top is None or df_top.empty:
        return None, None

    df = df_top.copy()

    # Suavização de ruído
    df = df[df["coincidentes"] >= 2]
    if df.empty:
        df = df_top.copy()

    # Correção microestrutural básica
    df["micro"] = df["sim_faixas"] * 0.5 + df["recencia_norm"] * 0.3

    df["score_ipo"] = (df["score_total"] * 0.6) + (df["micro"] * 0.4)

    df_ipo = df.sort_values(by="score_ipo", ascending=False).head(10)

    # Núcleo IPO
    pesos = {}
    for _, r in df_ipo.iterrows():
        for n in r["passageiros"]:
            pesos[n] = pesos.get(n, 0.0) + float(r["score_ipo"])

    ordenados = sorted(pesos.items(), key=lambda x: x[1], reverse=True)
    nucleo_ipo = [n for n, _ in ordenados[:6]]

    return df_ipo, nucleo_ipo


# -------------------------------------------------------------
# ANTI-SELFBIAS (ASB A + B)
# -------------------------------------------------------------
def aplicar_asb(nucleo_ipo, passageiros_alvo, modo):
    """
    Aplica Anti-SelfBias:
    - A (leve): troca 1 número em caso de autociclagem forte.
    - B (médio): troca 1–2 números em comum com o alvo.
    """
    if nucleo_ipo is None:
        return None

    alvo_set = set(passageiros_alvo)
    nuc = list(nucleo_ipo)

    comuns = len(alvo_set.intersection(nuc))

    # ASB A — Leve
    if modo == "A":
        if comuns == len(nuc):
            # troca o último número por um da faixa menos representada
            faixas = [faixa_num(n) for n in nuc]
            todas = {1, 2, 3, 4}
            faltantes = list(todas - set(faixas))
            if faltantes:
                f = min(faltantes)
                candidato = f * 20 - 5
            else:
                candidato = min(nuc) + 1
            nuc[-1] = candidato
        return sorted(nuc)

    # ASB B — Médio
    if modo == "B":
        if comuns >= len(nuc) - 1:
            # remove 1 número do alvo
            for n in list(nuc):
                if n in alvo_set:
                    nuc.remove(n)
                    break
            # adiciona um número estruturado próximo à média
            if nuc:
                media = int(np.mean(nuc))
            else:
                media = 40
            candidato = media + 1
            if candidato in nuc:
                candidato += 2
            if 1 <= candidato <= 80:
                nuc.append(candidato)

        nuc = nuc[:6]
        return sorted(nuc)

    return sorted(nuc)


# -------------------------------------------------------------
# Núcleo Resiliente (base IPO + ASB-B)
# -------------------------------------------------------------
def gerar_nucleo_resiliente(nucleo_ipo, nucleo_asb_b):
    """
    Núcleo Resiliente V13.8 — Combinação IPO + ASB-B
    Dá mais peso ao ASB-B (anti-selfbias médio),
    preservando coerência estrutural.
    """
    if not nucleo_ipo or not nucleo_asb_b:
        return None

    base = list(dict.fromkeys(nucleo_asb_b + nucleo_ipo))  # união preservando ordem

    pesos = {}
    for n in base:
        pesos[n] = 0.0
        if n in nucleo_asb_b:
            pesos[n] += 2.0
        if n in nucleo_ipo:
            pesos[n] += 1.0
        # pequeno ajuste por faixa (apenas para diversificar)
        faixa = faixa_num(n)
        pesos[n] += 0.1 * (5 - faixa)

    ordenados = sorted(pesos.items(), key=lambda x: x[1], reverse=True)
    resiliente = [n for n, _ in ordenados[:6]]
    resiliente.sort()
    return resiliente


# -------------------------------------------------------------
# ICA — Iterative Core Adjustment (ajuste fino do núcleo)
# -------------------------------------------------------------
def aplicar_ica_nucleo(
    nucleo_resiliente,
    nucleo_idx,
    nucleo_ipf,
    nucleo_ipo,
    passageiros_alvo,
):
    """
    Ajuste fino do Núcleo Resiliente usando IDX / IPF / IPO.

    Ideia:
    - Se há boa concordância entre Núcleo, IDX e IPO → mantém núcleo.
    - Se houver divergência forte → recalibra pesos combinando os três.
    - Penaliza levemente autociclagem com o alvo.
    """
    if not nucleo_resiliente or not nucleo_ipo or not nucleo_idx or not nucleo_ipf:
        return nucleo_resiliente

    set_res = set(nucleo_resiliente)
    set_idx = set(nucleo_idx)
    set_ipf = set(nucleo_ipf)
    set_ipo = set(nucleo_ipo)
    set_alvo = set(passageiros_alvo or [])

    overlap_idx_ipo = len(set_idx.intersection(set_ipo))
    overlap_res_ipo = len(set_res.intersection(set_ipo))
    overlap_res_idx = len(set_res.intersection(set_idx))

    # Se o cenário é bem alinhado (alta convergência), não mexe em nada.
    if overlap_idx_ipo >= 4 and overlap_res_ipo >= 4 and overlap_res_idx >= 4:
        return nucleo_resiliente

    # Caso contrário, recalcula pesos combinando as fontes
    union_core = set()
    for arr in (nucleo_resiliente, nucleo_idx, nucleo_ipf, nucleo_ipo):
        for n in arr:
            if 1 <= n <= 80:
                union_core.add(n)

    pesos = {}
    for n in union_core:
        pesos[n] = 0.0
        if n in set_res:
            pesos[n] += 2.0  # peso forte no núcleo atual
        if n in set_ipo:
            pesos[n] += 1.5
        if n in set_ipf:
            pesos[n] += 1.0
        if n in set_idx:
            pesos[n] += 1.0
        if n in set_alvo:
            pesos[n] -= 0.5  # leve penalização por autociclagem

    ordenados = sorted(pesos.items(), key=lambda x: x[1], reverse=True)
    ajustado = [n for n, _ in ordenados[:6]]
    ajustado.sort()

    # Se por acaso ficar idêntico ao núcleo original, apenas retorna
    return ajustado


# -------------------------------------------------------------
# Cobertura de Vento
# -------------------------------------------------------------
def gerar_cobertura_de_vento(nucleo_resiliente, passageiros_alvo):
    """
    Gera Cobertura de Vento simples em torno do Núcleo Resiliente
    (10–15 números, com adjacências e presença da série alvo).
    """
    if not nucleo_resiliente:
        return []

    candidatos = set(nucleo_resiliente)

    # Adjacências imediatas
    for n in nucleo_resiliente:
        for delta in (-1, 1):
            v = n + delta
            if 1 <= v <= 80:
                candidatos.add(v)

    # Inclui passageiros da série alvo
    for n in passageiros_alvo or []:
        if 1 <= n <= 80:
            candidatos.add(n)

    cobertura = sorted(candidatos)

    # Ajuste de tamanho (máx ~15)
    if len(cobertura) > 15:
        # corta bordas mais distantes do centro
        while len(cobertura) > 15:
            med = np.median(cobertura)
            dist_inicio = abs(cobertura[0] - med)
            dist_fim = abs(cobertura[-1] - med)
            if dist_inicio > dist_fim:
                cobertura.pop(0)
            else:
                cobertura.pop()

    return cobertura


# -------------------------------------------------------------
# Listas SA1 / MAX / Híbrida
# -------------------------------------------------------------
def gerar_listas_sa1_max_hibrida(cobertura, nucleo_resiliente):
    """
    Gera três listas:
    - SA1: parte estável da cobertura;
    - MAX: mais agressiva, priorizando núcleo + números altos da cobertura;
    - Híbrida: união organizada entre SA1 e Núcleo.
    """
    if not cobertura:
        return [], [], []

    sa1 = cobertura[: min(10, len(cobertura))]

    maiores = list(reversed(cobertura))
    max_lista = []
    for n in list(nucleo_resiliente) + maiores:
        if n not in max_lista:
            max_lista.append(n)
        if len(max_lista) >= 10:
            break

    h_set = set(sa1) | set(nucleo_resiliente)
    hibrida = sorted(h_set)

    return sa1, max_lista, hibrida


# -------------------------------------------------------------
# Espremer (versões -E)
# -------------------------------------------------------------
def espremer_listas(sa1, max_lista, hibrida, nucleo_resiliente):
    """
    Aplica compressão leve:
    - SA1-E: igual à SA1 (já equilibrada);
    - MAX-E: MAX sem o último elemento (redução leve);
    - Híbrida-E: aproximada ao Núcleo Resiliente.
    """
    sa1_e = list(sa1)

    max_e = list(max_lista[:-1]) if len(max_lista) > 0 else []
    if len(max_e) < 3:
        max_e = list(max_lista)

    hibrida_e = list(nucleo_resiliente) if nucleo_resiliente else list(hibrida)

    return sa1_e, max_e, hibrida_e


# -------------------------------------------------------------
# Modo 6 Acertos (S6)
# -------------------------------------------------------------
def gerar_s6(nucleo_resiliente, sa1_e, max_e, cobertura):
    """
    Monta S6 Alfa / Bravo / Charlie a partir de:
    - Núcleo Resiliente;
    - SA1-E / MAX-E;
    - Cobertura.
    """
    alfa = list(nucleo_resiliente) if nucleo_resiliente else []

    suporte = set(sa1_e) | set(max_e)
    cobertura_set = set(cobertura)

    bravo_candidatos = list((cobertura_set & suporte) - set(alfa))
    bravo = sorted(bravo_candidatos)[:4]

    usados = set(alfa) | set(bravo)
    charlie_cand = list(cobertura_set - usados)
    charlie = sorted(charlie_cand)[:3]

    return alfa, bravo, charlie


# -------------------------------------------------------------
# Ensamble Final
# -------------------------------------------------------------
def gerar_ensamble_final(nucleo_resiliente, sa1_e, max_e):
    """
    Gera lista única robusta (Ensamble) usando:
    Núcleo → SA1-E → MAX-E (sem duplicar).
    """
    ordem = list(nucleo_resiliente) + list(sa1_e) + list(max_e)
    vistos = []
    for n in ordem:
        if n not in vistos:
            vistos.append(n)

    if len(vistos) > 10:
        vistos = vistos[:10]

    return vistos


# -------------------------------------------------------------
# Faróis + Confiabilidade
# -------------------------------------------------------------
def avaliar_farol_e_confiabilidade(nucleo_resiliente, cobertura, ensamble):
    """
    Estima farol, barômetro e um percentual de confiabilidade simples
    (versão compacta para o app).
    """
    confianca = 70.0

    if cobertura and len(cobertura) <= 15:
        confianca += 5.0
    if ensamble and 8 <= len(ensamble) <= 10:
        confianca += 5.0
    if nucleo_resiliente and len(nucleo_resiliente) == 6:
        confianca += 5.0

    confianca = max(40.0, min(95.0, confianca))

    if confianca >= 80:
        farol = "🟢"
        barometro = "Resiliente"
    elif confianca >= 60:
        farol = "🟡"
        barometro = "Intermediário"
    elif confianca >= 50:
        farol = "🟠"
        barometro = "Turbulento"
    else:
        farol = "🔴"
        barometro = "Pré-ruptura"

    return farol, barometro, int(round(confianca))


# -------------------------------------------------------------
# Função de pipeline completo (IDX → IPF → IPO → ASB → Resiliente → ICA)
# -------------------------------------------------------------
def rodar_pipeline_completo(historico_bruto: str, modo_asb: str = "B"):
    """
    Executa todo o pipeline para uso nas páginas:
    - retorna dicionário com todas as estruturas principais.
    """
    registros = parse_historico(historico_bruto)
    if len(registros) < 2:
        return None

    df_idx, alvo, nuc_idx = encontrar_similares_idx_avancado(registros)
    if df_idx is None or df_idx.empty:
        return None

    df_ipf, nuc_ipf = calcular_ipf_hibrido(df_idx, alvo)
    df_ipo, nuc_ipo = calcular_ipo_profissional(df_idx)

    if nuc_ipo is None:
        nuc_asb_a = None
        nuc_asb_b = None
        nuc_res = None
        nuc_ica = None
    else:
        nuc_asb_a = aplicar_asb(
            nucleo_ipo=nuc_ipo, passageiros_alvo=alvo["passageiros"], modo="A"
        )
        nuc_asb_b = aplicar_asb(
            nucleo_ipo=nuc_ipo, passageiros_alvo=alvo["passageiros"], modo="B"
        )
        nuc_res = gerar_nucleo_resiliente(nuc_ipo, nuc_asb_b)
        nuc_ica = aplicar_ica_nucleo(
            nucleo_resiliente=nuc_res,
            nucleo_idx=nuc_idx,
            nucleo_ipf=nuc_ipf,
            nucleo_ipo=nuc_ipo,
            passageiros_alvo=alvo["passageiros"],
        )

    return {
        "alvo": alvo,
        "df_idx": df_idx,
        "nucleo_idx": nuc_idx,
        "df_ipf": df_ipf,
        "nucleo_ipf": nuc_ipf,
        "df_ipo": df_ipo,
        "nucleo_ipo": nuc_ipo,
        "nucleo_asb_a": nuc_asb_a,
        "nucleo_asb_b": nuc_asb_b,
        "nucleo_resiliente": nuc_res,
        "nucleo_ica": nuc_ica,
    }


# =============================================================
# SIDEBAR — Histórico + Navegação (menu em grupos)
# =============================================================
st.sidebar.title("🚗 Predict Cars V13.8")

st.sidebar.markdown("### 1. Histórico")

uploaded_file = st.sidebar.file_uploader(
    "Enviar arquivo de histórico (.txt ou .csv):",
    type=["txt", "csv"],
)
if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8", errors="ignore")
    set_historico(content)
    st.sidebar.success("Histórico carregado via arquivo.")

st.sidebar.markdown("Ou cole o histórico abaixo:")
historico_texto = st.sidebar.text_area(
    "Cole aqui as linhas do histórico (Cxxxx; n1; ...; k):",
    height=150,
    key="historico_text_area",
)

if st.sidebar.button("Carregar histórico do texto"):
    if historico_texto.strip():
        set_historico(historico_texto)
        st.sidebar.success("Histórico carregado a partir do texto.")
    else:
        st.sidebar.warning("Campo de texto vazio.")

st.sidebar.markdown("### 2. Navegação")

secao = st.sidebar.selectbox(
    "Escolha a seção:",
    ["📊 Painel", "📚 Documentação", "🧪 Pipeline Analítico", "🎯 Previsões", "🔧 Ajustes"],
)

subpagina = None
if secao == "📊 Painel":
    subpagina = st.sidebar.radio(
        "Visão geral:",
        ["Painel Principal"],
    )

elif secao == "📚 Documentação":
    subpagina = st.sidebar.radio(
        "Documentação:",
        ["Manual V13.8 (resumo)", "Bloco de Ativação V13.8"],
    )

elif secao == "🧪 Pipeline Analítico":
    subpagina = st.sidebar.radio(
        "Pipeline:",
        ["Modo Normal (protótipo)", "Modo IDX / IPF / IPO / ASB", "Núcleo Resiliente (visão rápida)"],
    )

elif secao == "🎯 Previsões":
    subpagina = st.sidebar.radio(
        "Previsões:",
        [
            "Previsões Finais (Núcleo Resiliente)",
            "Previsão Completa (V13.8)",
            "Modo 6 Acertos (S6) - detalhado",
            "Ensamble Final (detalhado)",
        ],
    )

elif secao == "🔧 Ajustes":
    subpagina = st.sidebar.radio(
        "Ajustes:",
        ["Ajuste Dinâmico (protótipo)"],
    )

historico_bruto = get_historico()

if historico_bruto:
    st.success("✅ Histórico carregado e disponível.")
else:
    st.info("ℹ️ Nenhum histórico carregado ainda.")


# =============================================================
# PÁGINAS PRINCIPAIS
# =============================================================

# -------------------------------------------------------------
# Painel Principal
# -------------------------------------------------------------
if subpagina == "Painel Principal":
    st.title("🚗 Predict Cars V13.8 — Painel Principal")
    st.markdown(
        "Use a barra lateral para carregar o histórico e navegar entre as seções.\n\n"
        "- **Pipeline Analítico** mostra IDX / IPF / IPO / ASB e Núcleo.\n"
        "- **Previsões** mostra Núcleo Resiliente (com ICA) e a Previsão Completa V13.8.\n"
        "- **Documentação** traz o manual resumido e o Bloco de Ativação."
    )

    if historico_bruto:
        with st.expander("Visualizar primeiras linhas do histórico"):
            st.text("\n".join(historico_bruto.splitlines()[:40]))


# -------------------------------------------------------------
# Documentação — Manual (resumo) / Bloco de Ativação
# -------------------------------------------------------------
elif subpagina == "Manual V13.8 (resumo)":
    st.title("📘 Manual Técnico — Resumo V13.8")
    st.markdown(
        "- Camadas principais: Modo Normal, IDX, IPF, IPO, Anti-SelfBias (ASB), Núcleo Resiliente, ICA, "
        "Cobertura, Listas SA1/MAX/Híbrida, Espremer, S6, Ensamble, Faróis e Confiabilidade.\n"
        "- O sistema busca trechos historicamente semelhantes à série atual e funde múltiplas evidências "
        "para formar um Núcleo Resiliente.\n"
        "- O ICA faz o ajuste fino estrutural do núcleo antes das camadas finais."
    )
    st.info(
        "Para detalhes completos, utilize o Manual Técnico Ultra-Híbrido V13.8 (Partes 1 a 5) "
        "no próprio ChatGPT."
    )

elif subpagina == "Bloco de Ativação V13.8":
    st.title("📦 Bloco de Ativação — Predict Cars V13.8")
    st.markdown(
        "Este é o bloco conceitual usado nos chats para ativar o modo V13.8.\n"
        "No app, ele é representado pelas funções de pipeline e pelas páginas de previsão."
    )
    st.code(
        "ATIVAR_PREDICT_CARS_V13.8\n"
        "Modo: Ultra-Híbrido Completo\n"
        "- Modo Normal\n"
        "- IDX + IPF + IPO\n"
        "- Anti-SelfBias (ASB)\n"
        "- Núcleo Resiliente\n"
        "- ICA (Iterative Core Adjustment)\n"
        "- Cobertura de Vento\n"
        "- SA1 / MAX / Híbrida\n"
        "- Espremer\n"
        "- S6 (6 acertos)\n"
        "- Ensamble Final\n"
        "- Faróis + Barômetro + Confiabilidade\n"
        "STATUS: OK — Pronto para uso",
        language="text",
    )


# -------------------------------------------------------------
# Pipeline — Modo Normal
# -------------------------------------------------------------
elif subpagina == "Modo Normal (protótipo)":
    st.title("⚙️ Modo Normal — Protótipo")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        st.markdown("Distribuição simples de frequência dos passageiros (protótipo).")
        nums = extrair_numeros(historico_bruto)
        if nums:
            st.bar_chart(pd.Series(nums).value_counts().sort_index())
        else:
            st.info("Não foi possível extrair números.")


# -------------------------------------------------------------
# Pipeline — IDX / IPF / IPO / ASB
# -------------------------------------------------------------
elif subpagina == "Modo IDX / IPF / IPO / ASB":
    st.title("🎯 IDX → IPF → IPO → ASB")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        resultado = rodar_pipeline_completo(historico_bruto, modo_asb="B")
        if resultado is None:
            st.warning("Histórico insuficiente para o pipeline.")
        else:
            alvo = resultado["alvo"]

            st.subheader("📌 Série atual (alvo)")
            st.write(f"Linha: {alvo['linha']}")
            st.write(f"ID: {alvo['id']}")
            st.write(f"Passageiros: {alvo['passageiros']}")
            st.code(alvo["texto"])

            # IDX
            st.markdown("---")
            st.subheader("🔍 IDX Avançado")
            st.dataframe(resultado["df_idx"], use_container_width=True)
            st.write(f"**Núcleo IDX (ponderado):** {resultado['nucleo_idx']}")

            # IPF
            st.markdown("---")
            st.subheader("🧩 IPF Híbrido")
            st.dataframe(resultado["df_ipf"], use_container_width=True)
            st.write(f"**Núcleo IPF (híbrido):** {resultado['nucleo_ipf']}")

            # IPO
            st.markdown("---")
            st.subheader("🚀 IPO Profissional")
            st.dataframe(resultado["df_ipo"], use_container_width=True)
            st.write(f"**Núcleo IPO (profissional):** {resultado['nucleo_ipo']}")

            # ASB A/B
            st.markdown("---")
            st.subheader("🧹 Anti-SelfBias (A/B)")

            modo_asb_label = st.selectbox(
                "Selecione o modo Anti-SelfBias para visualizar:",
                ["A (leve)", "B (médio)"],
                index=1,
            )
            if modo_asb_label.startswith("A"):
                nuc_asb = resultado["nucleo_asb_a"]
                modo_txt = "A (leve)"
            else:
                nuc_asb = resultado["nucleo_asb_b"]
                modo_txt = "B (médio)"

            st.write(f"**Núcleo IPO original:** {resultado['nucleo_ipo']}")
            st.write(f"**Núcleo IPO Anti-SelfBias {modo_txt}:** {nuc_asb}")

            st.success("Pipeline IDX → IPF → IPO → ASB executado com sucesso.")


# -------------------------------------------------------------
# Pipeline — Núcleo Resiliente (visão rápida)
# -------------------------------------------------------------
elif subpagina == "Núcleo Resiliente (visão rápida)":
    st.title("🔰 Núcleo Resiliente — Visão Rápida")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        resultado = rodar_pipeline_completo(historico_bruto, modo_asb="B")
        nuc_res = resultado["nucleo_resiliente"] if resultado else None
        nuc_ica = resultado["nucleo_ica"] if resultado else None

        if resultado is None or nuc_res is None:
            st.warning("Não foi possível gerar o Núcleo Resiliente. Verifique o histórico.")
        else:
            alvo = resultado["alvo"]

            st.subheader("📌 Série atual (alvo)")
            st.write(f"ID: {alvo['id']}")
            st.write(f"Passageiros: {alvo['passageiros']}")
            st.code(alvo["texto"])

            st.markdown("---")
            st.subheader("Núcleos intermediários")
            st.write(f"IDX: {resultado['nucleo_idx']}")
            st.write(f"IPF: {resultado['nucleo_ipf']}")
            st.write(f"IPO: {resultado['nucleo_ipo']}")

            st.markdown("---")
            st.subheader("Núcleo Anti-SelfBias (B)")
            st.write(f"ASB B: {resultado['nucleo_asb_b']}")

            st.markdown("---")
            st.subheader("🔰 Núcleo Resiliente V13.8 (pré-ICA)")
            st.write(nuc_res)

            if nuc_ica:
                st.subheader("🔧 Núcleo Resiliente (após ICA)")
                st.success(nuc_ica)
            else:
                st.info("ICA não alterou o núcleo neste cenário (mantido o pré-ICA).")


# -------------------------------------------------------------
# Previsões — Núcleo Resiliente
# -------------------------------------------------------------
elif subpagina == "Previsões Finais (Núcleo Resiliente)":
    st.title("📊 Previsões Finais — Núcleo Resiliente")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        resultado = rodar_pipeline_completo(historico_bruto, modo_asb="B")
        if resultado is None:
            st.warning("Histórico insuficiente para o pipeline.")
        else:
            alvo = resultado["alvo"]

            st.subheader("📌 Série atual (alvo)")
            st.write(f"ID: {alvo['id']}")
            st.write(f"Passageiros: {alvo['passageiros']}")
            st.code(alvo["texto"])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Núcleos intermediários")
                st.write(f"IDX: {resultado['nucleo_idx']}")
                st.write(f"IPF: {resultado['nucleo_ipf']}")
                st.write(f"IPO: {resultado['nucleo_ipo']}")

            with col2:
                st.markdown("### Núcleo Anti-SelfBias (B)")
                st.write(f"ASB B: {resultado['nucleo_asb_b']}")

            st.markdown("---")
            st.markdown("## 🔰 Núcleo Resiliente V13.8 (pré-ICA)")
            st.write(resultado["nucleo_resiliente"])

            if resultado["nucleo_ica"]:
                st.markdown("## 🔧 Núcleo Ajustado (ICA)")
                st.success(f"{resultado['nucleo_ica']}")
            else:
                st.info("ICA não alterou o núcleo neste cenário.")


# -------------------------------------------------------------
# Previsões — Previsão Completa V13.8
# -------------------------------------------------------------
elif subpagina == "Previsão Completa (V13.8)":
    st.title("📦 Previsão Completa — Predict Cars V13.8")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        resultado = rodar_pipeline_completo(historico_bruto, modo_asb="B")
        if (
            resultado is None
            or resultado["nucleo_resiliente"] is None
        ):
            st.warning("Não foi possível gerar o Núcleo Resiliente. Verifique o histórico.")
        else:
            alvo = resultado["alvo"]
            nuc_res_pre = resultado["nucleo_resiliente"]
            nuc_ica = resultado["nucleo_ica"]
            nuc_base = nuc_ica or nuc_res_pre

            st.subheader("📌 Série atual (alvo)")
            st.write(f"ID: {alvo['id']}")
            st.write(f"Passageiros: {alvo['passageiros']}")
            st.code(alvo["texto"])

            # Construção das camadas finais usando o núcleo ajustado (ICA quando existir)
            cobertura = gerar_cobertura_de_vento(nuc_base, alvo["passageiros"])
            sa1, max_lista, hibrida = gerar_listas_sa1_max_hibrida(cobertura, nuc_base)
            sa1_e, max_e, hibrida_e = espremer_listas(sa1, max_lista, hibrida, nuc_base)
            s6_alfa, s6_bravo, s6_charlie = gerar_s6(
                nuc_base, sa1_e, max_e, cobertura
            )
            ensamble = gerar_ensamble_final(nuc_base, sa1_e, max_e)
            farol, barometro, confiab = avaliar_farol_e_confiabilidade(
                nuc_base, cobertura, ensamble
            )

            # Exibição no espírito do Formato Oficial
            st.markdown("---")
            st.markdown("### 1) Núcleo Resiliente")
            st.write(f"(pré-ICA): {nuc_res_pre}")
            if nuc_ica:
                st.write(f"(após ICA): {nuc_base}")
            else:
                st.write("(ICA não alterou o núcleo)")

            st.markdown("### 2) Cobertura de Vento (10–15 números)")
            st.write(cobertura)

            st.markdown("### 3) Listas SA1 / MAX / Híbrida")
            st.write("**SA1:**")
            st.write(sa1)
            st.write("**MAX:**")
            st.write(max_lista)
            st.write("**Híbrida:**")
            st.write(hibrida)

            st.markdown("### 4) Versões Espremidas (SA1-E / MAX-E / Híbrida-E)")
            st.write("**SA1-E:**")
            st.write(sa1_e)
            st.write("**MAX-E:**")
            st.write(max_e)
            st.write("**Híbrida-E:**")
            st.write(hibrida_e)

            st.markdown("### 5) S6 — Alfa / Bravo / Charlie")
            st.write("**S6 Alfa:**")
            st.write(s6_alfa)
            st.write("**S6 Bravo:**")
            st.write(s6_bravo)
            st.write("**S6 Charlie:**")
            st.write(s6_charlie)

            st.markdown("### 6) Ensamble Final")
            st.write(ensamble)

            st.markdown("### 7) Faróis")
            st.write(farol)

            st.markdown("### 8) Barômetro")
            st.write(barometro)

            st.markdown("### 9) Confiabilidade (%)")
            st.write(f"{confiab}%")

            st.markdown("### 10) Observações Estruturais")
            st.markdown(
                "- Faixas dominantes e comportamento geral seguem o Núcleo Resiliente (após ICA) e a Cobertura.\n"
                "- O farol e o barômetro refletem a estabilidade atual do cenário.\n"
                "- Esta página implementa uma versão compacta do Formato Oficial do V13.8."
            )


# -------------------------------------------------------------
# Previsões — Modo 6 Acertos (S6) Detalhado
# -------------------------------------------------------------
elif subpagina == "Modo 6 Acertos (S6) - detalhado":
    st.title("🎯 Modo 6 Acertos (S6) — Detalhado")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        resultado = rodar_pipeline_completo(historico_bruto, modo_asb="B")
        if (
            resultado is None
            or resultado["nucleo_resiliente"] is None
        ):
            st.warning("Não foi possível gerar o Núcleo Resiliente. Verifique o histórico.")
        else:
            alvo = resultado["alvo"]
            nuc_res_pre = resultado["nucleo_resiliente"]
            nuc_ica = resultado["nucleo_ica"]
            nuc_base = nuc_ica or nuc_res_pre

            cobertura = gerar_cobertura_de_vento(nuc_base, alvo["passageiros"])
            sa1, max_lista, hibrida = gerar_listas_sa1_max_hibrida(cobertura, nuc_base)
            sa1_e, max_e, hibrida_e = espremer_listas(sa1, max_lista, hibrida, nuc_base)
            s6_alfa, s6_bravo, s6_charlie = gerar_s6(
                nuc_base, sa1_e, max_e, cobertura
            )

            st.subheader("📌 Série atual (alvo)")
            st.write(f"ID: {alvo['id']}")
            st.write(f"Passageiros: {alvo['passageiros']}")
            st.code(alvo["texto"])

            st.markdown("---")
            st.markdown("### Núcleo usado pelo S6")
            st.write(f"(pré-ICA): {nuc_res_pre}")
            if nuc_ica:
                st.write(f"(após ICA): {nuc_base}")
            else:
                st.write("(ICA não alterou o núcleo)")

            st.markdown("### S6 Alfa (núcleo máximo)")
            st.write(s6_alfa)

            st.markdown("### S6 Bravo (apoio forte)")
            st.write(s6_bravo)

            st.markdown("### S6 Charlie (apoio moderado)")
            st.write(s6_charlie)

            st.info(
                "O S6 concentra os passageiros com maior suporte estrutural (Alfa), "
                "seguidos dos apoios fortes (Bravo) e moderados (Charlie), conforme o V13.8."
            )


# -------------------------------------------------------------
# Previsões — Ensamble Final Detalhado
# -------------------------------------------------------------
elif subpagina == "Ensamble Final (detalhado)":
    st.title("🧠 Ensamble Final — Detalhado")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        resultado = rodar_pipeline_completo(historico_bruto, modo_asb="B")
        if (
            resultado is None
            or resultado["nucleo_resiliente"] is None
        ):
            st.warning("Não foi possível gerar o Núcleo Resiliente. Verifique o histórico.")
        else:
            alvo = resultado["alvo"]
            nuc_res_pre = resultado["nucleo_resiliente"]
            nuc_ica = resultado["nucleo_ica"]
            nuc_base = nuc_ica or nuc_res_pre

            cobertura = gerar_cobertura_de_vento(nuc_base, alvo["passageiros"])
            sa1, max_lista, hibrida = gerar_listas_sa1_max_hibrida(cobertura, nuc_base)
            sa1_e, max_e, hibrida_e = espremer_listas(sa1, max_lista, hibrida, nuc_base)
            s6_alfa, s6_bravo, s6_charlie = gerar_s6(
                nuc_base, sa1_e, max_e, cobertura
            )
            ensamble = gerar_ensamble_final(nuc_base, sa1_e, max_e)
            farol, barometro, confiab = avaliar_farol_e_confiabilidade(
                nuc_base, cobertura, ensamble
            )

            st.subheader("📌 Série atual (alvo)")
            st.write(f"ID: {alvo['id']}")
            st.write(f"Passageiros: {alvo['passageiros']}")
            st.code(alvo["texto"])

            st.markdown("---")
            st.markdown("### Componentes usados pelo Ensamble")
            st.write(f"Núcleo (pré-ICA): {nuc_res_pre}")
            if nuc_ica:
                st.write(f"Núcleo (após ICA): {nuc_base}")
            else:
                st.write("Núcleo após ICA: (sem alteração)")
            st.write(f"SA1-E: {sa1_e}")
            st.write(f"MAX-E: {max_e}")
            st.write(f"S6 Alfa: {s6_alfa}")
            st.write(f"S6 Bravo: {s6_bravo}")
            st.write(f"S6 Charlie: {s6_charlie}")

            st.markdown("---")
            st.markdown("### Ensamble Final (lista robusta)")
            st.write(ensamble)

            st.markdown("---")
            st.markdown("### Farol, Barômetro e Confiabilidade")
            st.write(f"Farol: {farol}")
            st.write(f"Barômetro: {barometro}")
            st.write(f"Confiabilidade: {confiab}%")

            st.info(
                "O Ensamble integra Núcleo (ajustado pelo ICA sempre que necessário), "
                "SA1-E, MAX-E e S6 em uma lista única, ponderando estabilidade, "
                "cobertura e convergência."
            )


# -------------------------------------------------------------
# Ajuste Dinâmico (protótipo)
# -------------------------------------------------------------
elif subpagina == "Ajuste Dinâmico (protótipo)":
    st.title("🔁 Ajuste Dinâmico — Protótipo")
    st.info(
        "Futuro módulo ICA/HLA ampliado para ajustes de pesos e microestruturas "
        "sobre o Núcleo Resiliente e as listas SA1/MAX/Híbrida."
    )
