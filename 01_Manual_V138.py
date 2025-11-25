
import streamlit as st

st.title("📘 Manual Técnico — Predict Cars V13.8")

st.markdown(
    "Esta página apresenta uma **versão resumida e navegável** do Manual Técnico Ultra-Híbrido "
    "**Predict Cars V13.8**. Ela serve como documentação dentro do próprio aplicativo web."
)

with st.expander("1. Caracterização Geral", expanded=True):
    st.markdown(
        "- Sistema de análise histórica e previsão baseado em múltiplas camadas.\n"
        "- Integra estatística clássica, análise de regime (barômetro), clustering comportamental (motoristas),\n"
        "  backtesting, bootstrapping, simulação Monte Carlo e calibração via modelos tabulares.\n"
        "- Objetivo: gerar previsões estáveis, interpretáveis e consistentes para a próxima série."
    )

with st.expander("2. Formato dos Dados (Histórico)"):
    st.markdown(
        "Cada linha do arquivo de entrada segue o padrão:\n\n"
        "`C1234; n1; n2; n3; n4; n5; k`\n\n"
        "- `C1234`: identificador da série (carro).\n"
        "- `n1..n5` (ou n1..n6): passageiros (números entre 1 e 80, sem repetição).\n"
        "- `k`: rótulo auxiliar (sensor/guarda)."
    )

with st.expander("3. Visão Geral das Camadas do V13.8"):
    st.markdown(
        "Camadas principais (visão conceitual):\n"
        "1. **Pré-processamento**: validação do histórico, consistência e limpeza.\n"
        "2. **Estatísticas Básicas e Frequências**.\n"
        "3. **Barômetro / Regime** (Resiliente, Intermediário, Turbulento, Pré-Ruptura, Pós-Ruptura).\n"
        "4. **Clustering / Motoristas** (padrões de condução da estrada).\n"
        "5. **Módulo IDX Puro Focado (IPF)**.\n"
        "6. **Modo IDX Otimizado (IPO)**.\n"
        "7. **Ajustes Dinâmicos (ICA, HLA, etc., conforme manual).\n"
        "8. **Construção do Núcleo e Cobertura de Vento**.\n"
        "9. **Geração de listas SA1 / MAX / híbridas**.\n"
        "10. **Confiabilidade, testes no passado e alertas (faróis)."
    )

st.info(
    "Esta é uma versão inicial e resumida do manual no app. "
    "À medida que evoluirmos, podemos inserir aqui todos os capítulos, com muito mais detalhes."
)
