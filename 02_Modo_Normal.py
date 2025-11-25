
import streamlit as st
import numpy as np
import pandas as pd

st.title("⚙️ Modo Normal — Pipeline Completo (Visão Inicial)")

st.markdown(
    "Esta página representa o **Modo Normal** do V13.8.\n"
    "Na versão completa, aqui será executado o pipeline completo: pré-processamento, "
    "estatísticas, barômetro, motoristas, núcleos, coberturas e listas finais."
)

if "historico_bruto" not in st.session_state:
    st.warning("Nenhum histórico foi carregado ainda. Volte à página principal e envie o arquivo.")
    st.stop()

raw = st.session_state["historico_bruto"]
lines = [l.strip() for l in raw.splitlines() if l.strip()]

st.subheader("📥 Resumo do Histórico Carregado")
st.write(f"Total de linhas detectadas: **{len(lines)}**")

with st.expander("Visualizar algumas linhas brutas"):
    st.text("\n".join(lines[:30]))

st.subheader("🧪 Simulação de Análise Básica (Protótipo)")
st.markdown(
    "Abaixo está apenas uma **simulação simplificada** para testar a interface. "
    "No futuro, esta lógica será substituída pelo pipeline real do Manual V13.8."
)

# Exemplo: contar frequência simples de números (protótipo)
numeros = []
for line in lines:
    partes = [p.strip() for p in line.split(";") if p.strip()]
    # Ignorar primeiro elemento se parecer com 'Cxxxx'
    if partes and partes[0].upper().startswith("C"):
        partes = partes[1:]
    # Ignorar último (k) se for numérico ou não
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

if numeros:
    serie = pd.Series(numeros)
    freq = serie.value_counts().sort_index()
    st.write("Distribuição simples de frequência dos passageiros (protótipo):")
    st.bar_chart(freq)
else:
    st.info("Não foi possível extrair números das linhas. Verifique o formato do arquivo.")

st.success(
    "Interface do Modo Normal pronta. A lógica interna do V13.8 poderá ser implantada aqui passo a passo."
)
