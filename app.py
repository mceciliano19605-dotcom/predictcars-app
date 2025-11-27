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

    Implementa:
    - expansão por recorrência
    - reforço de vizinhança
    - compressão entre 10 e 15 números
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
        "- **Cobertura de Vento** e as demais camadas avançadas seguem o Manual V13.8."
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
            nuc_res = resultado["nucleo_resiliente"]
            if not nuc_res:
                st.warning("Núcleo Resiliente não disponível. Verifique IPO/ASB.")
            else:
                cobertura = gerar_cobertura_vento(
                    nucleo_resiliente=resultado["nucleo_resiliente"],
                    nucleo_idx=resultado["nucleo_idx"],
                    nucleo_ipo=resultado["nucleo_ipo"],
                    nucleo_asb_b=resultado["nucleo_asb_b"],
                )

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
                if cobertura:
                    st.success(cobertura)
                else:
                    st.info("Não foi possível gerar a Cobertura de Vento.")

                st.caption("Geração conforme o Manual Técnico Ultra-Híbrido — Predict Cars V13.8.")


elif pagina == "Listas SA1 / MAX / Híbrida":
    st.title("📋 Listas SA1 / MAX / Híbrida — Protótipo")
    st.info(
        "Esta seção será usada para gerar as listas SA1, MAX e Híbrida a partir do "
        "Núcleo Resiliente + Cobertura de Vento, conforme o V13.8. "
        "No momento, o módulo está em desenvolvimento."
    )


elif pagina == "Modo Espremer":
    st.title("🧱 Modo Espremer — Protótipo")
    st.info(
        "O Modo Espremer comprime SA1 / MAX / Híbrida em versões SA1-E / MAX-E / Híbrida-E, "
        "removendo ruído, anticiclagem excessiva e faixas incoerentes. "
        "Este módulo ainda será implementado."
    )


elif pagina == "Modo 6 Acertos (S6)":
    st.title("🎯 Modo 6 Acertos (S6) — Protótipo")
    st.info(
        "O Modo S6 realiza a convergência máxima (Alfa / Bravo / Charlie) a partir das listas "
        "espremidas e do Núcleo Resiliente. "
        "Este módulo será adicionado em uma próxima versão."
    )


elif pagina == "Ensamble Final":
    st.title("🧠 Ensamble Final — Protótipo")
    st.info(
        "O Ensamble Final integra Núcleo, Cobertura, SA1-E, MAX-E, Híbrida-E e S6 "
        "em uma lista compacta e altamente robusta. "
        "A lógica detalhada será implementada em breve."
    )


elif pagina == "Faróis e Confiabilidade":
    st.title("🚦 Faróis e Confiabilidade — Protótipo")
    st.info(
        "Esta seção exibirá os Faróis (🟢🟡🟠🔴🟣) e a Confiabilidade (%) do cenário, "
        "com base em dispersão, clima, convergência e comportamento das listas. "
        "Ainda será implementada conforme o Manual V13.8."
    )


elif pagina == "Formato Oficial (V13.8)":
    st.title("📑 Formato Oficial — V13.8")
    st.info(
        "Aqui será exibido o Formato Oficial completo (1 a 10 blocos): "
        "Núcleo, Cobertura, SA1/MAX/Híbrida, Espremidas, S6, Ensamble, Faróis, "
        "Barômetro, Confiabilidade e Observações Estruturais. "
        "O módulo será integrado após a implementação das camadas avançadas."
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
