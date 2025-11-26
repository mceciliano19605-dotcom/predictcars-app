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
        'linha': índice (0, 1, 2, ...),
        'id': 'Cxxxx' ou None,
        'passageiros': [n1, n2, ...],
        'k': rótulo final (ou None),
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

def encontrar_similares_idx_avancado(registros, w_coinc=3.0, w_recencia=2.0, w_faixa=1.0):
    """
    IDX avançado (versão intermediária):
    - coincidência de passageiros
    - recência (mais recente = mais peso)
    - similaridade de faixas
    - escolha adaptativa de quantidade de trechos
    - núcleo ponderado pelos scores
    """
    if not registros or len(registros) < 2:
        return None, None, None

    alvo = registros[-1]  # última série
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

        # Recência: linha mais próxima do alvo => valor maior
        recencia_norm = r["linha"] / max_linha

        # Similaridade de faixas
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

    # Escolha adaptativa da quantidade de trechos
    num_cand = len(df)
    top_k = int(np.ceil(num_cand * 0.2))  # ~20% dos melhores
    top_k = max(5, min(25, top_k))        # entre 5 e 25
    top_df = df.head(top_k)

    # Núcleo ponderado pelos scores
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
# SIDEBAR — Histórico + Navegação
# -------------------------------------------------------------
st.sidebar.title("🚗 Predict Cars V13.8")

st.sidebar.markdown("### 1. Histórico")

# Opção 1: Upload de arquivo
uploaded_file = st.sidebar.file_uploader(
    "Enviar arquivo de histórico (.txt ou .csv):",
    type=["txt", "csv"]
)
if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8", errors="ignore")
    set_historico(content)
    st.sidebar.success("Histórico carregado via arquivo.")

# Opção 2: Colar texto diretamente
st.sidebar.markdown("Ou cole o histórico abaixo:")
historico_texto = st.sidebar.text_area(
    "Cole aqui as linhas do histórico (Cxxxx; n1; n2; ...; k):",
    height=150,
    key="historico_text_area"
)

if st.sidebar.button("Carregar histórico do texto"):
    if historico_texto.strip():
        set_historico(historico_texto)
        st.sidebar.success("Histórico carregado a partir do texto.")
    else:
        st.sidebar.warning("Campo de texto vazio. Cole o histórico antes de carregar.")

# Navegação
st.sidebar.markdown("### 2. Navegação")
pagina = st.sidebar.radio(
    "Escolha a seção:",
    (
        "Painel Principal",
        "Manual V13.8 (resumo)",
        "Modo Normal (protótipo)",
        "Modo IDX (avançado)",
        "Modo IPO (otimizado)",     # NOVA PÁGINA
        "Ajuste Dinâmico (protótipo)",
        "Previsões Finais (protótipo)",
    )
)

historico_bruto = get_historico()

# Mensagem de status do histórico no topo da área principal
if historico_bruto:
    st.success("✅ Histórico carregado e disponível para as demais seções.")
else:
    st.info("ℹ️ Nenhum histórico carregado ainda. Use o arquivo ou o campo de texto na barra lateral.")

# -------------------------------------------------------------
# CONTEÚDO DAS PÁGINAS
# -------------------------------------------------------------
if pagina == "Painel Principal":
    st.title("🚗 Predict Cars V13.8 — Painel Principal")

    st.markdown(
        "Bem-vindo ao painel web do **Predict Cars V13.8**.\n\n"
        "Use a barra lateral para:\n"
        "- Carregar o histórico (arquivo ou texto);\n"
        "- Navegar entre Manual, Modo Normal, Modo IDX, Modo IPO, Ajuste Dinâmico e Previsões."
    )

    if historico_bruto:
        with st.expander("Pré-visualização das primeiras linhas do histórico carregado"):
            preview_lines = "\n".join(historico_bruto.splitlines()[:40])
            st.text(preview_lines)

elif pagina == "Manual V13.8 (resumo)":
    st.title("📘 Manual Técnico — Predict Cars V13.8 (Resumo)")

    st.markdown(
        "Esta página apresenta um **resumo navegável** do Manual Técnico Ultra-Híbrido "
        "**Predict Cars V13.8**, em formato compacto, para consulta rápida dentro do app."
    )

    with st.expander("1. Caracterização Geral", expanded=True):
        st.markdown(
            "- Sistema de análise histórica e previsão baseado em múltiplas camadas.\n"
            "- Integra estatística clássica, análise de regime (barômetro/turbulência), clustering comportamental "
            "(motoristas), backtesting, bootstrapping, simulação Monte Carlo e calibração via modelos tabulares.\n"
            "- Objetivo: produzir previsões estáveis, interpretáveis e consistentes para a próxima série."
        )

    with st.expander("2. Formato dos Dados (Histórico)"):
        st.markdown(
            "Cada linha do histórico segue, em geral, o padrão:\n\n"
            "`C1234; n1; n2; n3; n4; n5; k`\n\n"
            "- `C1234`: identificador da série (carro).\n"
            "- `n1..n5` (ou n1..n6): passageiros (números entre 1 e 80, sem repetição).\n"
            "- `k`: rótulo auxiliar (sensor/guarda)."
        )

    with st.expander("3. Camadas Principais do V13.8"):
        st.markdown(
            "Camadas conceituais do sistema:\n"
            "1. Pré-processamento: validação do histórico e consistência.\n"
            "2. Estatísticas básicas e frequências.\n"
            "3. Barômetro / Regime (Resiliente, Intermediário, Turbulento, Pré-Ruptura, Pós-Ruptura).\n"
            "4. Clustering / Motoristas (padrões de condução).\n"
            "5. Módulo IDX Puro Focado (IPF).\n"
            "6. Modo IDX Otimizado (IPO).\n"
            "7. Ajustes Dinâmicos (ICA, HLA, etc.).\n"
            "8. Construção de Núcleo + Cobertura de Vento.\n"
            "9. Geração de listas SA1 / MAX / híbridas.\n"
            "10. Confiabilidade, testes no passado e alertas (faróis)."
        )

    st.info(
        "Este é um resumo inicial. Quando quiser, podemos integrar aqui a versão completa do manual "
        "V13.8 com todos os capítulos."
    )

else:
    # As demais páginas exigem histórico
    if not historico_bruto:
        st.warning("Para usar esta seção, carregue primeiro o histórico pela barra lateral (arquivo ou texto).")
    else:
        if pagina == "Modo Normal (protótipo)":
            st.title("⚙️ Modo Normal — Pipeline Básico (Protótipo)")

            st.markdown(
                "Esta página representa o **Modo Normal** do V13.8.\n"
                "Por enquanto, utiliza uma análise simplificada de frequências apenas para teste da interface. "
                "Mais tarde, será substituída pela lógica completa do manual."
            )

            lines = [l.strip() for l in historico_bruto.splitlines() if l.strip()]
            st.subheader("📥 Resumo do Histórico Carregado")
            st.write(f"Total de linhas detectadas: **{len(lines)}**")

            with st.expander("Visualizar algumas linhas brutas"):
                st.text("\n".join(lines[:40]))

            numeros = extrair_numeros(historico_bruto)
            if numeros:
                serie = pd.Series(numeros)
                freq = serie.value_counts().sort_index()
                st.subheader("📊 Distribuição simples de frequência dos passageiros (protótipo)")
                st.bar_chart(freq)
            else:
                st.info("Não foi possível extrair números das linhas. Verifique o formato do histórico.")

            st.success(
                "Interface do Modo Normal pronta. A lógica interna do V13.8 poderá ser implantada aqui passo a passo."
            )

        elif pagina == "Modo IDX (avançado)":
            st.title("🎯 Modo IDX (Avançado) — IPF Intermediário")

            st.markdown(
                "Esta página implementa um **IDX avançado (versão intermediária)** do V13.8:\n"
                "- considera a **última série** do histórico como estado atual;\n"
                "- procura, no passado, as séries mais parecidas;\n"
                "- usa coincidência de passageiros, recência e similaridade de faixas;\n"
                "- escolhe automaticamente quantos trechos usar (entre 5 e 25);\n"
                "- monta um **núcleo IDX ponderado** pelos scores.\n\n"
                "Este é o primeiro passo em direção ao IPF/IPO completos."
            )

            registros = parse_historico(historico_bruto)

            if len(registros) < 2:
                st.warning("Histórico com poucas linhas para análise IDX. Adicione mais séries.")
            else:
                # Modo técnico (avançado) opcional
                with st.expander("🔧 Modo Técnico (avançado — opcional)", expanded=False):
                    st.markdown(
                        "Os pesos abaixo já estão configurados com valores recomendados.\n"
                        "Alterar é opcional e serve apenas para experimentos avançados."
                    )
                    w_coinc = st.slider("Peso de coincidência de passageiros", 0.0, 5.0, 3.0, 0.5)
                    w_rec = st.slider("Peso de recência", 0.0, 5.0, 2.0, 0.5)
                    w_fx = st.slider("Peso de similaridade de faixas", 0.0, 5.0, 1.0, 0.5)

                df_similares, alvo, nucleo = encontrar_similares_idx_avancado(
                    registros,
                    w_coinc=w_coinc,
                    w_recencia=w_rec,
                    w_faixa=w_fx,
                )

                st.subheader("📌 Série atual (alvo do IDX)")
                st.write(f"Linha: **{alvo['linha']}**")
                st.write(f"ID: **{alvo['id']}**")
                st.write(f"Passageiros: **{alvo['passageiros']}**")
                st.write(f"k: **{alvo['k']}**")
                st.code(alvo["texto"])

                if df_similares is None or df_similares.empty:
                    st.info("Nenhuma série semelhante encontrada. Verifique o histórico e os formatos.")
                else:
                    st.subheader("🔍 Séries mais semelhantes (IDX avançado)")
                    st.markdown(
                        "A tabela abaixo mostra as séries mais parecidas com a atual, já considerando:\n"
                        "- coincidência de passageiros;\n"
                        "- recência (mais recente = maior peso);\n"
                        "- similaridade de faixas numéricas;\n"
                        "- score total ponderado."
                    )
                    st.dataframe(
                        df_similares[[
                            "linha", "id", "coincidentes", "recencia_norm",
                            "sim_faixas", "score_total", "qtd_passageiros", "passageiros", "texto"
                        ]],
                        use_container_width=True,
                    )

                st.markdown("---")
                st.subheader("🧩 Núcleo IDX ponderado (versão intermediária)")
                if nucleo:
                    st.markdown(
                        "Passageiros com maior peso combinando trechos mais similares, recência e faixas:"
                    )
                    st.markdown(f"**Núcleo IDX (ponderado):** `{nucleo}`")
                else:
                    st.info(
                        "Ainda não foi possível montar um núcleo ponderado. "
                        "Verifique se o histórico possui formato e volume adequados."
                    )

                st.success(
                    "IDX avançado implementado. Nas próximas etapas, será possível incluir ritmo, motoristas, "
                    "barômetro e construção direta do Núcleo Resiliente e das listas SA1/MAX."
                )

                # ---------------------------------------------------------
                # 🔧 Seção interna IPO – Otimização do IDX (alfa)
                # ---------------------------------------------------------
                if df_similares is not None and not df_similares.empty:
                    st.markdown("---")
                    st.subheader("🔧 IPO – Otimização do IDX (versão alfa, interna)")

                    df_ipo = df_similares.copy()
                    max_coinc = df_ipo["coincidentes"].max() or 1
                    df_ipo["score_suavizado"] = df_ipo["score_total"] * (
                        0.8 + 0.2 * df_ipo["coincidentes"] / max_coinc
                    )

                    pesos_ipo = {}
                    for _, row in df_ipo.iterrows():
                        score_s = float(row["score_suavizado"])
                        for n in row["passageiros"]:
                            pesos_ipo[n] = pesos_ipo.get(n, 0.0) + score_s

                    nucleo_ipo = [
                        n for n, _ in sorted(pesos_ipo.items(), key=lambda x: x[1], reverse=True)[:6]
                    ]

                    st.markdown("**Núcleo IPO (suavizado, interno):**")
                    st.write(nucleo_ipo)

                    with st.expander("Ver tabela IPO interna"):
                        st.dataframe(df_ipo, use_container_width=True)

                    st.info(
                        "Esta é a versão interna (alfa) do IPO, derivada diretamente do IDX avançado. "
                        "Ela será refinada com dispersão, motorista secundário e clima nas próximas etapas."
                    )

        elif pagina == "Modo IPO (otimizado)":
            st.title("🎯 Modo IPO — IDX Otimizado (Protótipo)")

            st.markdown(
                "O IPO é a evolução do IDX avançado, aplicando suavização, correção de ruído e "
                "ajuste fino para gerar um Núcleo mais estável e coerente com o momento atual."
            )

            registros = parse_historico(historico_bruto)

            if len(registros) < 2:
                st.warning("Histórico insuficiente para gerar IPO.")
            else:
                df_similares, alvo, nucleo = encontrar_similares_idx_avancado(registros)

                if df_similares is None or df_similares.empty:
                    st.warning("Sem séries semelhantes para iniciar o IPO.")
                else:
                    st.subheader("📌 Série atual")
                    st.code(alvo["texto"])

                    df_ipo = df_similares.copy()
                    max_coinc = df_ipo["coincidentes"].max() or 1
                    df_ipo["score_suavizado"] = df_ipo["score_total"] * (
                        0.8 + 0.2 * df_ipo["coincidentes"] / max_coinc
                    )

                    pesos_ipo = {}
                    for _, row in df_ipo.iterrows():
                        score_s = float(row["score_suavizado"])
                        for n in row["passageiros"]:
                            pesos_ipo[n] = pesos_ipo.get(n, 0.0) + score_s

                    nucleo_ipo = [
                        n for n, _ in sorted(pesos_ipo.items(), key=lambda x: x[1], reverse=True)[:6]
                    ]

                    st.subheader("🧩 Núcleo IPO")
                    st.write(nucleo_ipo)

                    st.subheader("📊 Tabela IPO (versão alfa)")
                    st.dataframe(df_ipo, use_container_width=True)

                    st.info(
                        "Esta é a versão preliminar do IPO. A versão completa incluirá dispersão, "
                        "motorista secundário, faixas ajustadas e pesos inteligentes."
                    )

        elif pagina == "Ajuste Dinâmico (protótipo)":
            st.title("🔁 Ajuste Dinâmico — ICA / HLA (Protótipo)")

            st.markdown(
                "Esta página representa o módulo de **Ajuste Dinâmico** do V13.8.\n"
                "Ela será usada para recalibrar o sistema com base em desvios observados, sem alterar a essência do manual."
            )

            modo = st.selectbox(
                "Escolha o modo de ajuste (protótipo):",
                ["Ajuste Leve", "Ajuste Médio", "Ajuste Profundo"]
            )
            st.write(f"Modo selecionado: **{modo}**")

            st.info(
                "No futuro, esta página aplicará ajustes sobre o núcleo e as listas geradas, "
                "usando os critérios detalhados do Manual V13.8 (entropia, desvio, estabilidade do motorista etc.)."
            )

        elif pagina == "Previsões Finais (protótipo)":
            st.title("📊 Previsões Finais — Núcleo, Cobertura e Listas (Protótipo)")

            st.markdown(
                "Esta página consolida as **previsões finais**: núcleo, núcleo resiliente, coberturas, "
                "listas SA1/MAX e demais saídas previstas no V13.8.\n\n"
                "Por enquanto, serve apenas como estrutura visual para, depois, receber a lógica completa de previsão."
            )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Núcleo (protótipo)")
                st.write("[  ] [  ] [  ] [  ] [  ] [  ]")

            with col2:
                st.markdown("### Cobertura de Vento (protótipo)")
                st.write("[  ] [  ] [  ] [  ] [  ] [  ]")

            st.markdown("---")
            st.markdown("### Listas SA1 / MAX (protótipo)")
            st.write("Aqui exibiremos as listas estruturadas, com rótulos claros (SA1, MAX, híbridas, etc.).")

            st.info(
                "Quando a lógica de previsão estiver implementada, esta página será o painel principal de resultados "
                "para você copiar e trazer para o ChatGPT discutir."
            )
