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
# Funções auxiliares
# -------------------------------------------------------------
def set_historico(conteudo: str):
    if conteudo is not None and conteudo.strip():
        st.session_state["historico_bruto"] = conteudo

def get_historico():
    return st.session_state.get("historico_bruto", None)

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
# IDX Avançado + IPF + IPO
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


def calcular_ipf_hibrido(df_top, alvo):
    """Implementa IPF híbrido simples: ritmo, dispersão e pares."""
    if df_top is None or df_top.empty:
        return None, None

    # Ritmo = proximidade de posições
    df = df_top.copy()
    df["ritmo"] = df["linha"].diff().abs().fillna(0)

    # Dispersão = variação interna da quantidade de passageiros
    df["dispersao"] = df["qtd_passageiros"].rolling(2).std().fillna(0)

    # Pares fixos (contagem simples)
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


def calcular_ipo_profissional(df_top, alvo):
    """Implementa IPO profissional com suavização e microcorreção."""
    if df_top is None or df_top.empty:
        return None, None

    df = df_top.copy()

    # Suavização de ruído
    df = df[df["coincidentes"] >= 2]
    if df.empty:
        df = df_top.copy()

    # Correção microestrutural
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
            pesos[n] = pesos.get(n, 0) + float(r["score_ipo"])

    ordenados = sorted(pesos.items(), key=lambda x: x[1], reverse=True)
    nucleo_ipo = [n for n, _ in ordenados[:6]]

    return df_ipo, nucleo_ipo


# -------------------------------------------------------------
# ANTI-SELFBIAS (ASB A + B)
# -------------------------------------------------------------
def aplicar_asb(nucleo_ipo, passageiros_alvo, modo):
    """Aplica anticiclagem leve (A) ou média (B)."""

    if nucleo_ipo is None:
        return None

    alvo_set = set(passageiros_alvo)
    nuc = nucleo_ipo.copy()

    # Quantidade de números em comum
    comuns = len(alvo_set.intersection(nuc))

    # ASB A — Leve
    if modo == "A":
        if comuns == 6:
            # troca 1 número pela menor lacuna de faixa
            faixas = [faixa_num(n) for n in nuc]
            faltante_faixa = min(set([1,2,3,4]) - set(faixas))
            # escolhe um substituto simples
            candidato = faltante_faixa * 20 - 5
            nuc[-1] = candidato
        return nuc

    # ASB B — Médio
    if modo == "B":
        if comuns >= 5:
            # remove 1 ou 2 números iguais ao alvo
            for n in nuc:
                if n in alvo_set:
                    nuc.remove(n)
                    break
            # adiciona um número estruturado
            candidato = int(np.mean(nuc)) + 1
            if candidato in nuc:
                    candidato += 2
            nuc.append(candidato)
        return sorted(nuc[:6])

    return nuc


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
        "Modo IDX (avançado + IPF + IPO + ASB)",
        "Ajuste Dinâmico (protótipo)",
        "Previsões Finais (protótipo)",
    )
)

historico_bruto = get_historico()

if historico_bruto:
    st.success("✅ Histórico carregado e disponível.")
else:
    st.info("ℹ️ Nenhum histórico carregado ainda.")

# -------------------------------------------------------------
# CONTEÚDO DAS PÁGINAS
# -------------------------------------------------------------
if pagina == "Painel Principal":
    st.title("🚗 Predict Cars V13.8 — Painel Principal")
    st.markdown("Use a barra lateral para navegar e carregar o histórico.")

elif pagina == "Manual V13.8 (resumo)":
    st.title("📘 Manual Técnico — Resumo")
    st.markdown("Resumo das principais camadas (IDX, IPF, IPO, ASB, Núcleo Resiliente etc.).")

elif pagina == "Modo Normal (protótipo)":
    st.title("⚙️ Modo Normal — Protótipo")
    st.markdown("Frequência simples dos passageiros.")
    nums = extrair_numeros(historico_bruto)
    if nums:
        st.bar_chart(pd.Series(nums).value_counts().sort_index())

elif pagina == "Modo IDX (avançado + IPF + IPO + ASB)":
    st.title("🎯 IDX → IPF → IPO → ASB")

    registros = parse_historico(historico_bruto)

    if len(registros) < 2:
        st.warning("Histórico insuficiente.")
    else:
        # ===================================================
        # 1. IDX Avançado
        # ===================================================
        df_similares, alvo, nucleo_idx = encontrar_similares_idx_avancado(registros)

        st.subheader("📌 Série atual (alvo)")
        st.write(f"Linha: {alvo['linha']}")
        st.write(f"ID: {alvo['id']}")
        st.write(f"Passageiros: {alvo['passageiros']}")
        st.code(alvo["texto"])

        st.subheader("🔍 IDX Avançado")
        st.dataframe(df_similares, use_container_width=True)
        st.write(f"**Núcleo IDX (ponderado):** {nucleo_idx}")

        # ===================================================
        # 2. IPF Híbrido
        # ===================================================
        st.markdown("---")
        st.subheader("🧩 IPF Híbrido")
        df_ipf, nucleo_ipf = calcular_ipf_hibrido(df_similares, alvo)
        st.dataframe(df_ipf, use_container_width=True)
        st.write(f"**Núcleo IPF (híbrido):** {nucleo_ipf}")

        # ===================================================
        # 3. IPO Profissional
        # ===================================================
        st.markdown("---")
        st.subheader("🚀 IPO Profissional")
        df_ipo, nucleo_ipo = calcular_ipo_profissional(df_similares, alvo)
        st.dataframe(df_ipo, use_container_width=True)
        st.write(f"**Núcleo IPO (profissional):** {nucleo_ipo}")

        # ===================================================
        # 4. ANTI-SELFBIAS (A/B)
        # ===================================================
        st.markdown("---")
        st.subheader("🧹 Anti-SelfBias (A/B)")

        modo_asb = st.selectbox(
            "Selecione o modo Anti-SelfBias:",
            ["A (leve)", "B (médio)"],
            index=1,
        )

        modo = "A" if modo_asb.startswith("A") else "B"

        nucleo_final = aplicar_asb(nucleo_ipo, alvo["passageiros"], modo)

        st.write(f"**Núcleo IPO original:** {nucleo_ipo}")
        st.write(f"**Núcleo IPO Anti-SelfBias ({modo}):** {nucleo_final}")

        st.success("Pipeline IDX → IPF → IPO → ASB completo e funcional.")

elif pagina == "Ajuste Dinâmico (protótipo)":
    st.title("🔁 Ajuste Dinâmico — Protótipo")
    st.info("Futuro módulo ICA/HLA.")

elif pagina == "Previsões Finais (protótipo)":
    st.title("📊 Previsões Finais — Protótipo")
    st.info("Núcleo Resiliente e Listas SA1/MAX virão aqui após IPO + ASB.")
