import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------------------------------------
# Configuração geral do app
# -------------------------------------------------------------
st.set_page_config(
    page_title="Predict Cars V13.8",
    page_icon="🚗",
    layout="wide",
)

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
    """Extrai números das linhas do histórico (protótipo simples)."""
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
def encontrar_similares_idx_avancado(registros, w_coinc=3.0, w_recencia=2.0, w_faixa=1.0):
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
    df = df.sort_values(by=["score_total", "coincidentes", "linha"], ascending=[False, False, False])

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
        ordenados = sorted(pesos_por_numero.items(), key=lambda x: x[1], reverse=True)
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
    df_filtrado = df[df["coincidentes"] >= 2]
    if df_filtrado.empty:
        df_filtrado = df

    # Correção microestrutural básica
    df_filtrado = df_filtrado.copy()
    df_filtrado["micro"] = df_filtrado["sim_faixas"] * 0.5 + df_filtrado["recencia_norm"] * 0.3

    df_filtrado["score_ipo"] = (
        df_filtrado["score_total"] * 0.6
        + df_filtrado["micro"] * 0.4
    )

    df_ipo = df_filtrado.sort_values(by="score_ipo", ascending=False).head(10)

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
        return sorted(nuc[:6])

    return sorted(nuc)


# -------------------------------------------------------------
# Núcleo Resiliente (base IPO + ASB-B)
# -------------------------------------------------------------
def gerar_nucleo_resiliente(nucleo_ipo, nucleo_asb_b):
    """
    Núcleo Resiliente V13.8 — Combinação IPO + ASB-B.
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
# ICA — Ajuste de Núcleo (versão leve)
# -------------------------------------------------------------
def aplicar_ica(nucleo_resiliente, registros):
    """
    ICA (Iterative Core Adjustment) — versão leve.

    Nesta versão:
    - Em cenários resilientes, tende a manter o núcleo.
    - Pode ser expandido futuramente para ajustes mais fortes.
    """
    if not nucleo_resiliente:
        return None

    # Versão inicial: apenas retorna uma cópia (sem alteração estrutural).
    return list(nucleo_resiliente)


# -------------------------------------------------------------
# HLA — High-Level Adjustment (Ajuste de Alto Nível V13.8)
# -------------------------------------------------------------
def aplicar_hla(nucleo_ica, s6_alfa=None, cobertura=None, ensamble=None):
    """
    Ajuste de Alto Nível (HLA) sobre o núcleo após ICA.

    - Se S6 Alfa estiver totalmente alinhado com o núcleo: apenas homologa.
    - Se houver diferenças, usa S6 Alfa como miolo, Ensamble como estabilizador
      e Cobertura como complemento.
    """

    if not nucleo_ica:
        return None

    nucleo_ica = list(nucleo_ica)
    s6_alfa = list(s6_alfa) if s6_alfa else []
    cobertura = list(cobertura) if cobertura else []
    ensamble = list(ensamble) if ensamble else []

    # Caso ideal: núcleo alinhado = só homologa
    if s6_alfa and sorted(s6_alfa) == sorted(nucleo_ica):
        return sorted(nucleo_ica)

    # Começa com o núcleo ICA
    base = []
    for n in nucleo_ica:
        if n not in base:
            base.append(n)

    # Miolo forte por S6 Alfa
    if s6_alfa:
        intersec = [n for n in nucleo_ica if n in s6_alfa]
        if len(intersec) >= 4:
            base = intersec

    # Complementa com ensamble
    for n in ensamble:
        if n not in base:
            base.append(n)
        if len(base) >= 6:
            break

    # Se faltar, complementa com cobertura
    if len(base) < 6 and cobertura:
        def dist(x):
            return min(abs(x - k) for k in nucleo_ica)
        for n in sorted(cobertura, key=dist):
            if n not in base:
                base.append(n)
            if len(base) >= 6:
                break

    # Garantia final
    if len(base) < 6:
        for n in nucleo_ica:
            if n not in base:
                base.append(n)
            if len(base) >= 6:
                break

    return sorted(base[:6])


# -------------------------------------------------------------
# Cobertura de Vento + Listas + Espremer + S6 + Ensamble + Faróis
# -------------------------------------------------------------
def construir_cobertura(nucleo_res, df_ipo, alvo):
    """
    Gera Cobertura de Vento (10–15 números) em torno do núcleo,
    IPO e da série alvo.
    """
    if not nucleo_res:
        return []

    numeros = set(nucleo_res)

    # Adiciona vizinhos +/-1
    for n in nucleo_res:
        for delta in (-1, 1):
            v = n + delta
            if 1 <= v <= 80:
                numeros.add(v)

    # Adiciona alvo
    for n in alvo["passageiros"]:
        numeros.add(n)

    # Adiciona números mais pesados do IPO
    if df_ipo is not None and not df_ipo.empty:
        pesos = {}
        for _, r in df_ipo.iterrows():
            score = float(r["score_ipo"])
            for n in r["passageiros"]:
                pesos[n] = pesos.get(n, 0.0) + score
        mais_fortes = [n for n, _ in sorted(pesos.items(), key=lambda x: x[1], reverse=True)]
        for n in mais_fortes[:10]:
            numeros.add(n)

    cobertura = sorted(numeros)
    # Limita para 10–15
    if len(cobertura) > 15:
        cobertura = cobertura[:15]
    return cobertura


def construir_listas_principais(nucleo_res, cobertura):
    """
    Constrói SA1, MAX e Híbrida a partir de núcleo + cobertura.
    """
    if not nucleo_res:
        return [], [], []

    # SA1: primeiros 10 da cobertura (mais suaves)
    sa1 = list(cobertura[:10])

    # MAX: núcleo + complementos mais "fortes"
    max_list = list(dict.fromkeys(nucleo_res + cobertura[::-1]))
    max_list = max_list[:10]

    # Híbrida: união de SA1 + Núcleo
    hibrida = list(dict.fromkeys(sa1 + nucleo_res))
    return sa1, max_list, hibrida


def espremer_listas(sa1, max_list, hibrida):
    """
    Modo Espremer: remove redundâncias óbvias e ruído.
    Versão inicial: mantém a mesma estrutura, apenas garantindo corte consistente.
    """
    sa1_e = list(sa1)
    max_e = list(max_list)
    hibrida_e = list(hibrida)

    # Limites (10 para SA1 / MAX, 12 para Híbrida)
    sa1_e = sa1_e[:10]
    max_e = max_e[:10]
    hibrida_e = hibrida_e[:12]

    return sa1_e, max_e, hibrida_e


def construir_s6(nucleo_final, sa1_e, max_e, hibrida_e):
    """
    Constrói S6 Alfa / Bravo / Charlie.
    - Alfa: baseado no núcleo final.
    - Bravo: apoio forte das listas.
    - Charlie: apoio moderado.
    """
    if not nucleo_final:
        return [], [], []

    alfa = sorted(list(dict.fromkeys(nucleo_final)))[:6]

    apoio_forte = list(dict.fromkeys(sa1_e + max_e + hibrida_e))
    bravo = [n for n in apoio_forte if n not in alfa][:4]

    resto = [n for n in apoio_forte if n not in alfa and n not in bravo]
    charlie = resto[:3]

    return alfa, bravo, charlie


def construir_ensamble(nucleo_final, bravo, charlie):
    """
    Ensamble Final: Núcleo + parte de Bravo + parte de Charlie.
    """
    base = list(dict.fromkeys(nucleo_final + bravo + charlie))
    return base[:10]


def avaliar_farol_barometro_confiabilidade(nucleo_final, cobertura, bravo, charlie):
    """
    Heurística simples para farol, barômetro e confiabilidade.
    """
    if not nucleo_final:
        return "🟣", "Pós-ruptura", 20

    # Diversidade de faixas no núcleo
    faixas_nucleo = {faixa_num(n) for n in nucleo_final if faixa_num(n) > 0}
    n_faixas = len(faixas_nucleo)

    # Tamanho da cobertura
    tam_cob = len(cobertura)

    # Nível de apoio (Bravo / Charlie)
    apoio = len(bravo) + len(charlie)

    # Cenário básico
    if n_faixas in (2, 3) and 10 <= tam_cob <= 15 and apoio >= 3:
        farol = "🟢"
        barometro = "Resiliente"
        confiab = 85
    elif n_faixas >= 2 and 8 <= tam_cob <= 18:
        farol = "🟡"
        barometro = "Intermediário"
        confiab = 65
    else:
        farol = "🟠"
        barometro = "Turbulento"
        confiab = 45

    return farol, barometro, confiab


# -------------------------------------------------------------
# Função de pipeline completo
# -------------------------------------------------------------
def rodar_pipeline_completo(historico_bruto: str):
    """
    Executa todo o pipeline para uso nas páginas:
    - IDX, IPF, IPO, ASB, Núcleo Resiliente, ICA, HLA
    - Cobertura, SA1 / MAX / Híbrida, Espremer
    - S6, Ensamble, Faróis, Barômetro, Confiabilidade
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
    else:
        nuc_asb_a = aplicar_asb(nuc_ipo, alvo["passageiros"], "A")
        nuc_asb_b = aplicar_asb(nuc_ipo, alvo["passageiros"], "B")
        nuc_res = gerar_nucleo_resiliente(nuc_ipo, nuc_asb_b)

    # ICA
    nuc_ica = aplicar_ica(nuc_res, registros)

    # Cobertura + Listas etc.
    cobertura = construir_cobertura(nuc_ica, df_ipo, alvo) if nuc_ica else []
    sa1, max_list, hibrida = construir_listas_principais(nuc_ica, cobertura) if nuc_ica else ([], [], [])
    sa1_e, max_e, hibrida_e = espremer_listas(sa1, max_list, hibrida)

    # S6 (usando núcleo pós-ICA inicialmente)
    s6_alfa, s6_bravo, s6_charlie = construir_s6(nuc_ica, sa1_e, max_e, hibrida_e)

    # Ensamble inicial
    ensamble = construir_ensamble(nuc_ica, s6_bravo, s6_charlie)

    # HLA — Núcleo final
    nuc_hla = aplicar_hla(nucleo_ica=nuc_ica, s6_alfa=s6_alfa, cobertura=cobertura, ensamble=ensamble)

    # Recalcula S6 com núcleo HLA (mais coerente)
    s6_alfa_h, s6_bravo_h, s6_charlie_h = construir_s6(nuc_hla, sa1_e, max_e, hibrida_e)
    ensamble_h = construir_ensamble(nuc_hla, s6_bravo_h, s6_charlie_h)

    # Faróis / Barômetro / Confiabilidade
    farol, barometro, confiab = avaliar_farol_barometro_confiabilidade(
        nuc_hla, cobertura, s6_bravo_h, s6_charlie_h
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
        "cobertura": cobertura,
        "sa1": sa1,
        "max_list": max_list,
        "hibrida": hibrida,
        "sa1_e": sa1_e,
        "max_e": max_e,
        "hibrida_e": hibrida_e,
        "s6_alfa": s6_alfa_h,
        "s6_bravo": s6_bravo_h,
        "s6_charlie": s6_charlie_h,
        "ensamble": ensamble_h,
        "nucleo_hla": nuc_hla,
        "farol": farol,
        "barometro": barometro,
        "confiabilidade": confiab,
    }


# -------------------------------------------------------------
# SIDEBAR — Histórico + Navegação
# -------------------------------------------------------------
st.sidebar.title("🚗 Predict Cars V13.8")

st.sidebar.markdown("### 1. Histórico")

uploaded_file = st.sidebar.file_uploader(
    "Enviar arquivo de histórico (.txt ou .csv):",
    type=["txt", "csv"]
)
if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8", errors="ignore")
    set_historico(content)
    st.sidebar.success("Histórico carregado via arquivo.")

st.sidebar.markdown("Ou cole o histórico abaixo:")
historico_texto = st.sidebar.text_area(
    "Cole aqui as linhas do histórico (Cxxxx; n1; ...; k):",
    height=150,
    key="historico_text_area"
)

if st.sidebar.button("Carregar histórico do texto"):
    if historico_texto.strip():
        set_historico(historico_texto)
        st.sidebar.success("Histórico carregado a partir do texto.")
    else:
        st.sidebar.warning("Campo de texto vazio.")

st.sidebar.markdown("### 2. Navegação")

pagina = st.sidebar.radio(
    "Escolha a seção:",
    (
        "Painel Principal",
        "Manual V13.8 (resumo)",
        "Modo Normal (protótipo)",
        "Modo IDX / IPF / IPO / ASB",
        "Previsões Finais (Núcleo Resiliente)",
        "Previsão Completa (V13.8)",
        "Ajuste Dinâmico (protótipo)",
    )
)

historico_bruto = get_historico()

if historico_bruto:
    st.success("✅ Histórico carregado e disponível.")
else:
    st.info("ℹ️ Nenhum histórico carregado ainda.")


# -------------------------------------------------------------
# PÁGINAS
# -------------------------------------------------------------
if pagina == "Painel Principal":
    st.title("🚗 Predict Cars V13.8 — Painel Principal")
    st.markdown(
        "Use a barra lateral para carregar o histórico e navegar entre as seções.\n\n"
        "- **Modo IDX / IPF / IPO / ASB** mostra o pipeline analítico.\n"
        "- **Previsões Finais** mostra o Núcleo Resiliente pronto para uso.\n"
        "- **Previsão Completa (V13.8)** mostra todas as camadas (Cobertura, SA1/MAX, S6, Ensamble, Faróis etc.).\n"
        "- **Modo Normal** traz frequências simples (protótipo)."
    )

    if historico_bruto:
        with st.expander("Visualizar primeiras linhas do histórico"):
            st.text("\n".join(historico_bruto.splitlines()[:40]))


elif pagina == "Manual V13.8 (resumo)":
    st.title("📘 Manual Técnico — Resumo V13.8")
    st.markdown(
        "- Camadas principais: Modo Normal, IDX, IPF, IPO, Anti-SelfBias (ASB), Núcleo Resiliente.\n"
        "- Camadas avançadas: ICA, HLA, Cobertura de Vento, SA1/MAX/Híbrida, Modo Espremer, S6, Ensamble, Faróis, Confiabilidade.\n"
        "- O Núcleo Resiliente é a base para Núcleo + Cobertura + listas SA1/MAX + S6.\n"
        "- Este painel web segue o espírito do Manual Técnico Ultra-Híbrido — Predict Cars V13.8."
    )


elif pagina == "Modo Normal (protótipo)":
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


elif pagina == "Modo IDX / IPF / IPO / ASB":
    st.title("🎯 IDX → IPF → IPO → ASB")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        resultado = rodar_pipeline_completo(historico_bruto)
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


elif pagina == "Previsões Finais (Núcleo Resiliente)":
    st.title("📊 Previsões Finais — Núcleo Resiliente")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        resultado = rodar_pipeline_completo(historico_bruto)
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
                st.markdown("### Anti-SelfBias (B) + Núcleo Resiliente")
                st.write(f"ASB B: {resultado['nucleo_asb_b']}")
                st.write(f"Núcleo Resiliente (pré-ICA): {resultado['nucleo_resiliente']}")

            st.markdown("---")
            st.markdown("## 🔧 Núcleo Ajustado (ICA)")

            nuc_ica = resultado["nucleo_ica"]
            if nuc_ica:
                st.success(f"Núcleo após ICA: {nuc_ica}")
            else:
                st.info("ICA não pôde ser aplicado (núcleo ausente).")


elif pagina == "Previsão Completa (V13.8)":
    st.title("📜 Previsão Completa — Predict Cars V13.8")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        resultado = rodar_pipeline_completo(historico_bruto)
        if resultado is None:
            st.warning("Histórico insuficiente para o pipeline.")
        else:
            alvo = resultado["alvo"]

            st.subheader("📌 Série atual (alvo)")
            st.write(f"ID: {alvo['id']}")
            st.write(f"Passageiros: {alvo['passageiros']}")
            st.code(alvo["texto"])

            st.markdown("---")
            st.markdown("### 1) Núcleo Resiliente + ICA + HLA")

            st.write(f"(pré-ICA): {resultado['nucleo_resiliente']}")
            st.write(f"(após ICA): {resultado['nucleo_ica']}")
            st.success(f"(ICA + HLA — Núcleo Final): {resultado['nucleo_hla']}")

            st.markdown("---")
            st.markdown("### 2) Cobertura de Vento (10–15 números)")
            st.write(resultado["cobertura"])

            st.markdown("---")
            st.markdown("### 3) Listas SA1 / MAX / Híbrida")

            st.write("**SA1:**")
            st.write(resultado["sa1"])

            st.write("**MAX:**")
            st.write(resultado["max_list"])

            st.write("**Híbrida:**")
            st.write(resultado["hibrida"])

            st.markdown("---")
            st.markdown("### 4) Versões Espremidas (SA1-E / MAX-E / Híbrida-E)")

            st.write("**SA1-E:**")
            st.write(resultado["sa1_e"])

            st.write("**MAX-E:**")
            st.write(resultado["max_e"])

            st.write("**Híbrida-E:**")
            st.write(resultado["hibrida_e"])

            st.markdown("---")
            st.markdown("### 5) S6 — Alfa / Bravo / Charlie")

            st.write("**S6 Alfa:**")
            st.write(resultado["s6_alfa"])

            st.write("**S6 Bravo:**")
            st.write(resultado["s6_bravo"])

            st.write("**S6 Charlie:**")
            st.write(resultado["s6_charlie"])

            st.markdown("---")
            st.markdown("### 6) Ensamble Final")
            st.write(resultado["ensamble"])

            st.markdown("---")
            st.markdown("### 7) Faróis")
            st.write(resultado["farol"])

            st.markdown("---")
            st.markdown("### 8) Barômetro")
            st.write(resultado["barometro"])

            st.markdown("---")
            st.markdown("### 9) Confiabilidade (%)")
            st.write(f"{resultado['confiabilidade']}%")

            st.markdown("---")
            st.markdown("### 10) Observações Estruturais")
            st.write(
                "Faixas dominantes, motorista, dispersão e clima seguem o Núcleo Final (ICA + HLA) "
                "e a Cobertura de Vento. O farol e o barômetro refletem a estabilidade atual do cenário."
            )


elif pagina == "Ajuste Dinâmico (protótipo)":
    st.title("🔁 Ajuste Dinâmico — Protótipo")
    st.info(
        "Futuro módulo ICA/HLA avançado para ajustes sobre o Núcleo Resiliente, "
        "listas e comportamento dinâmico do cenário."
    )
