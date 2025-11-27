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
        ordenados = sorted(pesos_por_numero.items(), key=lambda x: x[1], reverse=True)
        nucleo = [n for n, _ in ordenados[:6]]

    return top_df, alvo, nucleo


# -------------------------------------------------------------
# IPF Híbrido (versão intermediária)
# -------------------------------------------------------------
def calcular_ipf_hibrido(df_top, alvo):
    """Implementa IPF híbrido simples: ritmo, dispersão e pares."""
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

    df["score_ipo"] = (
        df["score_total"] * 0.6
        + df["micro"] * 0.4
    )

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
            nuc.append(candidato)
        return sorted(nuc[:6])

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
# Cobertura de Vento (V13.8)
# -------------------------------------------------------------
def gerar_cobertura_vento(nucleo_resiliente, nucleo_idx, nucleo_ipo, nucleo_asb_b):
    """
    Gera a Cobertura de Vento do V13.8 (10 a 15 números),
    baseada na fusão estruturada das camadas:
    - Núcleo Resiliente
    - Núcleo IDX
    - Núcleo IPO
    - Núcleo ASB-B
    """
    if not nucleo_resiliente:
        return None

    cobertura = set(nucleo_resiliente)

    # 1) Recorrências estruturais (IDX + IPO + ASB-B)
    recorrencias = []
    for bloco in (nucleo_idx, nucleo_ipo, nucleo_asb_b):
        if bloco:
            recorrencias.extend(bloco)

    recorrencias = [n for n in recorrencias if isinstance(n, int) and 1 <= n <= 80]

    for n in recorrencias:
        cobertura.add(n)

    # 2) Reforço de vizinhança em torno do Núcleo Resiliente
    reforco = []
    for n in nucleo_resiliente:
        if n > 2:
            reforco.append(n - 1)
        if n < 79:
            reforco.append(n + 1)

    reforco = [n for n in reforco if 1 <= n <= 80]
    for n in reforco:
        cobertura.add(n)

    cobertura = sorted(list(cobertura))

    # 3) Alongamento mínimo (se menos de 10)
    if len(cobertura) < 10:
        extras = []
        for n in nucleo_resiliente:
            if n > 3:
                extras.append(n - 2)
            if n < 78:
                extras.append(n + 2)
        extras = [x for x in extras if 1 <= x <= 80]
        for x in extras:
            cobertura.append(x)
        cobertura = sorted(list(set(cobertura)))

    # 4) Compressão máxima (se mais de 15)
    if len(cobertura) > 15:
        cobertura = cobertura[:15]

    return cobertura


# -------------------------------------------------------------
# Listas SA1 / MAX / Híbrida
# -------------------------------------------------------------
def gerar_listas_sa1_max_hibrida(nucleo_resiliente, cobertura):
    if not nucleo_resiliente:
        return None, None, None

    nuc = sorted(set(nucleo_resiliente))
    cov = sorted(set(cobertura or []))

    # SA1: núcleo + adjacentes muito próximos
    candidatos_sa1 = set(nuc)
    for n in cov:
        if any(abs(n - c) <= 2 for c in nuc):
            candidatos_sa1.add(n)
    sa1 = sorted(list(candidatos_sa1))
    if len(sa1) > 10:
        sa1 = sa1[:10]

    # MAX: núcleo + extremos da cobertura
    candidatos_max = set(nuc)
    if cov:
        extremos = cov[:3] + cov[-3:]
        for n in extremos:
            candidatos_max.add(n)
    max_list = sorted(list(candidatos_max))
    if len(max_list) > 12:
        max_list = max_list[:12]

    # Híbrida: núcleo + complementos de SA1/MAX até ~11 números
    hibrida = list(sorted(set(nuc)))
    union_extra = [n for n in sorted(set(sa1 + max_list)) if n not in hibrida]
    for n in union_extra:
        if len(hibrida) >= 11:
            break
        hibrida.append(n)
    hibrida = sorted(hibrida)

    return sa1, max_list, hibrida


# -------------------------------------------------------------
# Modo Espremer (SA1-E / MAX-E / Híbrida-E)
# -------------------------------------------------------------
def gerar_espremer(nucleo_resiliente, sa1, max_list, hibrida):
    if not nucleo_resiliente:
        return None, None, None

    nuc = sorted(set(nucleo_resiliente))

    # SA1-E: SA1 filtrada pelos vizinhos do núcleo
    vizinhos = set(nuc)
    for n in nuc:
        vizinhos.add(n - 1)
        vizinhos.add(n + 1)
    sa1_e = sorted(set(x for x in (sa1 or []) if x in vizinhos))
    if len(sa1_e) < len(nuc):
        sa1_e = nuc.copy()

    # MAX-E: remove extremos pouco conectados ao núcleo
    max_e_raw = sorted(set(max_list or []))
    filtrados = []
    for x in max_e_raw:
        if any(abs(x - n) <= 5 for n in nuc):
            filtrados.append(x)
    max_e = sorted(set(filtrados))
    if len(max_e) > 9:
        max_e = max_e[:9]

    # Híbrida-E = próprio Núcleo Resiliente
    hibrida_e = nuc.copy()

    return sa1_e, max_e, hibrida_e


# -------------------------------------------------------------
# Modo 6 Acertos (S6) — Alfa / Bravo / Charlie
# -------------------------------------------------------------
def gerar_s6(nucleo_resiliente, sa1_e, max_e, hibrida_e, cobertura):
    if not nucleo_resiliente:
        return [], [], []

    nuc = sorted(set(nucleo_resiliente))
    s6_alfa = nuc.copy()

    base_bravo = sorted(
        set((sa1_e or []) + (max_e or []) + (hibrida_e or [])) - set(s6_alfa)
    )
    s6_bravo = base_bravo[:4]

    base_charlie = sorted(
        set(cobertura or []) - set(s6_alfa) - set(s6_bravo)
    )
    s6_charlie = base_charlie[:3]

    return s6_alfa, s6_bravo, s6_charlie


# -------------------------------------------------------------
# Ensamble Final
# -------------------------------------------------------------
def gerar_ensamble_final(nucleo_resiliente, sa1_e, max_e, hibrida_e,
                         s6_alfa, s6_bravo, s6_charlie):
    if not nucleo_resiliente:
        return None

    from collections import Counter

    c = Counter()

    fontes_peso = [
        (nucleo_resiliente, 3),
        (hibrida_e, 3),
        (sa1_e, 2),
        (max_e, 1),
        (s6_alfa, 3),
        (s6_bravo, 2),
        (s6_charlie, 1),
    ]

    for lista, w in fontes_peso:
        for x in lista or []:
            c[x] += w

    ordenados = sorted(c.items(), key=lambda t: (-t[1], t[0]))
    ens = [n for n, _ in ordenados[:10]]
    ens = sorted(set(ens))
    if len(ens) > 10:
        ens = ens[:10]

    return ens


# -------------------------------------------------------------
# Faróis, Barômetro, Confiabilidade
# -------------------------------------------------------------
def avaliar_farol_barometro_confiab(nucleo_resiliente, cobertura,
                                    s6_alfa, s6_bravo, s6_charlie,
                                    ensamble):
    if not nucleo_resiliente:
        return "🟣", "Ruptura", 20

    score = 0

    # Força da S6
    if len(s6_alfa) >= 4:
        score += 2
    if len(s6_alfa) == 6:
        score += 2
    if len(s6_bravo) <= 3:
        score += 1

    # Cobertura saudável
    cov_len = len(cobertura or [])
    if 8 <= cov_len <= 18:
        score += 2

    # Ensamble compacto
    if ensamble and 7 <= len(ensamble) <= 10:
        score += 1

    # Penalidades
    if len(s6_charlie or []) > 3 or cov_len > 25:
        score -= 1

    score = max(0, min(score, 8))
    confiab = 50 + score * 5  # 50% a 90%

    if score >= 7:
        farol_emoji = "🟢"
        barometro = "Resiliente"
    elif score >= 5:
        farol_emoji = "🟡"
        barometro = "Intermediário"
    elif score >= 3:
        farol_emoji = "🟠"
        barometro = "Turbulento"
    elif score >= 1:
        farol_emoji = "🔴"
        barometro = "Pré-ruptura"
    else:
        farol_emoji = "🟣"
        barometro = "Ruptura"

    return farol_emoji, barometro, confiab


# -------------------------------------------------------------
# Função de pipeline completo (IDX → IPF → IPO → ASB → Resiliente)
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
    else:
        nuc_asb_a = aplicar_asb(nucleo_ipo=nuc_ipo, passageiros_alvo=alvo["passageiros"], modo="A")
        nuc_asb_b = aplicar_asb(nucleo_ipo=nuc_ipo, passageiros_alvo=alvo["passageiros"], modo="B")
        nuc_res = gerar_nucleo_resiliente(nucleo_ipo=nuc_ipo, nucleo_asb_b=nuc_asb_b)

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
    }


# -------------------------------------------------------------
# Função para montar camadas avançadas (Cobertura, SA1/MAX, S6, Ensamble, etc.)
# -------------------------------------------------------------
def construir_camadas_avancadas(resultado_pipeline):
    if resultado_pipeline is None:
        return None

    nuc_res = resultado_pipeline["nucleo_resiliente"]
    nuc_idx = resultado_pipeline["nucleo_idx"]
    nuc_ipo = resultado_pipeline["nucleo_ipo"]
    nuc_asb_b = resultado_pipeline["nucleo_asb_b"]

    if not nuc_res:
        return {
            "cobertura": None,
            "sa1": None,
            "max": None,
            "hibrida": None,
            "sa1_e": None,
            "max_e": None,
            "hibrida_e": None,
            "s6_alfa": [],
            "s6_bravo": [],
            "s6_charlie": [],
            "ensamble": None,
            "farol_emoji": "🟣",
            "barometro": "Ruptura",
            "confiabilidade": 20,
        }

    cobertura = gerar_cobertura_vento(
        nucleo_resiliente=nuc_res,
        nucleo_idx=nuc_idx,
        nucleo_ipo=nuc_ipo,
        nucleo_asb_b=nuc_asb_b,
    )

    sa1, max_list, hibrida = gerar_listas_sa1_max_hibrida(
        nucleo_resiliente=nuc_res,
        cobertura=cobertura,
    )

    sa1_e, max_e, hibrida_e = gerar_espremer(
        nucleo_resiliente=nuc_res,
        sa1=sa1,
        max_list=max_list,
        hibrida=hibrida,
    )

    s6_alfa, s6_bravo, s6_charlie = gerar_s6(
        nucleo_resiliente=nuc_res,
        sa1_e=sa1_e,
        max_e=max_e,
        hibrida_e=hibrida_e,
        cobertura=cobertura,
    )

    ensamble = gerar_ensamble_final(
        nucleo_resiliente=nuc_res,
        sa1_e=sa1_e,
        max_e=max_e,
        hibrida_e=hibrida_e,
        s6_alfa=s6_alfa,
        s6_bravo=s6_bravo,
        s6_charlie=s6_charlie,
    )

    farol_emoji, barometro, confiab = avaliar_farol_barometro_confiab(
        nucleo_resiliente=nuc_res,
        cobertura=cobertura,
        s6_alfa=s6_alfa,
        s6_bravo=s6_bravo,
        s6_charlie=s6_charlie,
        ensamble=ensamble,
    )

    return {
        "cobertura": cobertura,
        "sa1": sa1,
        "max": max_list,
        "hibrida": hibrida,
        "sa1_e": sa1_e,
        "max_e": max_e,
        "hibrida_e": hibrida_e,
        "s6_alfa": s6_alfa,
        "s6_bravo": s6_bravo,
        "s6_charlie": s6_charlie,
        "ensamble": ensamble,
        "farol_emoji": farol_emoji,
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
        "Modo Normal",
        "Camada IDX / IPF / IPO / ASB",
        "Cobertura de Vento",
        "Listas SA1 / MAX / Híbrida",
        "Modo Espremer",
        "Modo 6 Acertos (S6)",
        "Ensamble Final",
        "Faróis e Confiabilidade",
        "Formato Oficial (V13.8)",
        "Previsões Finais (Núcleo Resiliente)",
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
        "- **Camada IDX / IPF / IPO / ASB** mostra o pipeline analítico.\n"
        "- **Previsões Finais (Núcleo Resiliente)** mostra o núcleo pronto para uso.\n"
        "- **Cobertura de Vento** e as camadas avançadas seguem o Manual V13.8."
    )

    if historico_bruto:
        with st.expander("Visualizar primeiras linhas do histórico"):
            st.text("\n".join(historico_bruto.splitlines()[:40]))


elif pagina == "Manual V13.8 (resumo)":
    st.title("📘 Manual Técnico — Resumo V13.8")
    st.markdown(
        "- Camadas principais: Modo Normal, IDX, IPF, IPO, Anti-SelfBias (ASB), Núcleo Resiliente.\n"
        "- Camadas avançadas: Cobertura de Vento, SA1/MAX/Híbrida, Espremer, S6, Ensamble, Faróis, Confiabilidade, Formato Oficial.\n"
        "- Este painel web segue o espírito do Manual Técnico Ultra-Híbrido — Predict Cars V13.8."
    )


elif pagina == "Modo Normal":
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


elif pagina == "Camada IDX / IPF / IPO / ASB":
    st.title("🎯 Camada IDX / IPF / IPO / ASB")

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


elif pagina == "Cobertura de Vento":
    st.title("🌬 Cobertura de Vento — V13.8")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        resultado = rodar_pipeline_completo(historico_bruto, modo_asb="B")
        if resultado is None:
            st.warning("Histórico insuficiente para o pipeline.")
        else:
            camadas = construir_camadas_avancadas(resultado)
            nuc_res = resultado["nucleo_resiliente"]

            if not nuc_res:
                st.warning("Núcleo Resiliente não disponível. Verifique IPO/ASB.")
            else:
                alvo = resultado["alvo"]
                st.subheader("📌 Série atual (alvo)")
                st.write(f"ID: {alvo['id']}")
                st.write(f"Passageiros: {alvo['passageiros']}")
                st.code(alvo["texto"])

                st.markdown("---")
                st.subheader("🔰 Núcleo Resiliente")
                st.write(nuc_res)

                st.markdown("---")
                st.subheader("🌬 Cobertura de Vento (10–15 números)")
                if camadas["cobertura"]:
                    st.success(camadas["cobertura"])
                else:
                    st.info("Não foi possível gerar a Cobertura de Vento.")

                st.caption("Geração conforme o Manual Técnico Ultra-Híbrido — Predict Cars V13.8.")


elif pagina == "Listas SA1 / MAX / Híbrida":
    st.title("📋 Listas SA1 / MAX / Híbrida")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        resultado = rodar_pipeline_completo(historico_bruto, modo_asb="B")
        if resultado is None:
            st.warning("Histórico insuficiente para o pipeline.")
        else:
            camadas = construir_camadas_avancadas(resultado)
            alvo = resultado["alvo"]

            st.subheader("📌 Série atual (alvo)")
            st.write(f"ID: {alvo['id']}")
            st.write(f"Passageiros: {alvo['passageiros']}")
            st.code(alvo["texto"])

            st.markdown("---")
            st.subheader("SA1 (estável)")
            st.write(camadas["sa1"])

            st.subheader("MAX (agressiva)")
            st.write(camadas["max"])

            st.subheader("Híbrida (compromisso)")
            st.write(camadas["hibrida"])


elif pagina == "Modo Espremer":
    st.title("🧱 Modo Espremer — SA1-E / MAX-E / Híbrida-E")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        resultado = rodar_pipeline_completo(historico_bruto, modo_asb="B")
        if resultado is None:
            st.warning("Histórico insuficiente para o pipeline.")
        else:
            camadas = construir_camadas_avancadas(resultado)

            st.subheader("SA1-E (estável espremida)")
            st.write(camadas["sa1_e"])

            st.subheader("MAX-E (agressiva espremida)")
            st.write(camadas["max_e"])

            st.subheader("Híbrida-E (núcleo convergente)")
            st.write(camadas["hibrida_e"])


elif pagina == "Modo 6 Acertos (S6)":
    st.title("🎯 Modo 6 Acertos (S6) — Alfa / Bravo / Charlie")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        resultado = rodar_pipeline_completo(historico_bruto, modo_asb="B")
        if resultado is None:
            st.warning("Histórico insuficiente para o pipeline.")
        else:
            camadas = construir_camadas_avancadas(resultado)

            st.subheader("S6 Alfa (núcleo máximo)")
            st.write(camadas["s6_alfa"])

            st.subheader("S6 Bravo (apoio forte)")
            st.write(camadas["s6_bravo"])

            st.subheader("S6 Charlie (apoio moderado)")
            st.write(camadas["s6_charlie"])


elif pagina == "Ensamble Final":
    st.title("🧠 Ensamble Final — Lista Compacta")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        resultado = rodar_pipeline_completo(historico_bruto, modo_asb="B")
        if resultado is None:
            st.warning("Histórico insuficiente para o pipeline.")
        else:
            camadas = construir_camadas_avancadas(resultado)

            st.subheader("Ensamble Final (lista robusta)")
            st.write(camadas["ensamble"])


elif pagina == "Faróis e Confiabilidade":
    st.title("🚦 Faróis e Confiabilidade")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        resultado = rodar_pipeline_completo(historico_bruto, modo_asb="B")
        if resultado is None:
            st.warning("Histórico insuficiente para o pipeline.")
        else:
            camadas = construir_camadas_avancadas(resultado)

            st.subheader("Farol do momento")
            st.write(camadas["farol_emoji"])

            st.subheader("Barômetro")
            st.write(camadas["barometro"])

            st.subheader("Confiabilidade estimada (%)")
            st.write(f"{camadas['confiabilidade']}%")


elif pagina == "Formato Oficial (V13.8)":
    st.title("📑 Formato Oficial — V13.8")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        resultado = rodar_pipeline_completo(historico_bruto, modo_asb="B")
        if resultado is None:
            st.warning("Histórico insuficiente para o pipeline.")
        else:
            camadas = construir_camadas_avancadas(resultado)
            alvo = resultado["alvo"]

            st.subheader("📌 Série atual (alvo)")
            st.write(f"ID: {alvo['id']}")
            st.write(f"Passageiros: {alvo['passageiros']}")
            st.code(alvo["texto"])

            nuc_res = resultado["nucleo_resiliente"]

            st.markdown("---")
            st.markdown("### 1) Núcleo Resiliente")
            st.write(nuc_res)

            st.markdown("### 2) Cobertura de Vento")
            st.write(camadas["cobertura"])

            st.markdown("### 3) Listas SA1 / MAX / Híbrida")
            st.write("SA1:", camadas["sa1"])
            st.write("MAX:", camadas["max"])
            st.write("Híbrida:", camadas["hibrida"])

            st.markdown("### 4) Versões Espremidas (SA1-E / MAX-E / Híbrida-E")
            st.write("SA1-E:", camadas["sa1_e"])
            st.write("MAX-E:", camadas["max_e"])
            st.write("Híbrida-E:", camadas["hibrida_e"])

            st.markdown("### 5) S6 — Alfa / Bravo / Charlie")
            st.write("S6 Alfa:", camadas["s6_alfa"])
            st.write("S6 Bravo:", camadas["s6_bravo"])
            st.write("S6 Charlie:", camadas["s6_charlie"])

            st.markdown("### 6) Ensamble Final")
            st.write(camadas["ensamble"])

            st.markdown("### 7) Faróis")
            st.write(camadas["farol_emoji"])

            st.markdown("### 8) Barômetro")
            st.write(camadas["barometro"])

            st.markdown("### 9) Confiabilidade (%)")
            st.write(f"{camadas['confiabilidade']}%")

            st.markdown("### 10) Observações Estruturais")
            st.write(
                "- Faixa dominante, motorista, dispersão e clima são inferidos conforme o V13.8.\n"
                "- Este bloco resume o estado estrutural da série no momento."
            )


elif pagina == "Previsões Finais (Núcleo Resiliente)":
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
            st.markdown("## 🔰 Núcleo Resiliente V13.8 (base ASB Médio)")

            nuc_res = resultado["nucleo_resiliente"]
            if nuc_res:
                st.success(f"Núcleo Resiliente: {nuc_res}")
                st.info(
                    "Este é o núcleo estrutural que servirá de base para Núcleo + Cobertura + "
                    "listas SA1/MAX e modos avançados (6 acertos, Espremer etc.)."
                )
            else:
                st.info("Não foi possível gerar o Núcleo Resiliente (verifique IPO e ASB).")


elif pagina == "Ajuste Dinâmico (protótipo)":
    st.title("🔁 Ajuste Dinâmico — Protótipo")
    st.info("Futuro módulo ICA/HLA para ajustes sobre o Núcleo Resiliente e listas.")
