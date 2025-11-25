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
        "Modo IDX (protótipo)",
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
        "- Navegar entre Manual, Modo Normal, Modo IDX, Ajuste Dinâmico e Previsões."
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

        elif pagina == "Modo IDX (protótipo)":
            st.title("🎯 Modo IDX — IPF / IPO (Protótipo)")

            st.markdown(
                "Esta página representa o **Modo IDX** do V13.8 (IPF e IPO).\n\n"
                "No futuro, aqui será implantada a lógica de similaridade estrutural: identificação do trecho alvo, "
                "busca de trechos historicamente semelhantes e construção do núcleo puro baseado em IDX."
            )

            lines = [l.strip() for l in historico_bruto.splitlines() if l.strip()]
            st.subheader("📥 Resumo do Histórico")
            st.write(f"Total de linhas disponíveis: **{len(lines)}**")

            st.info("Modo IDX pronto para receber a lógica detalhada do manual (IPF, IPO, seleção de trechos etc.).")

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
