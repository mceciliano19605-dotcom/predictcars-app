# ============================================================
# PAINEL 4 — Modo TURBO++ ULTRA (Adaptativo por k*)
# ============================================================

def painel_modo_turbo_ultra_adaptativo():
    st.markdown("## 🚀 Modo TURBO++ ULTRA (Adaptativo por k*)")

    df = carregar_df_sessao()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        return

    pcols = [c for c in df.columns if c.startswith("p")]
    if not pcols:
        st.error("Não foram detectadas colunas de passageiros (p1..pN).")
        return

    col_cfg, col_info = st.columns([1, 1.1])

    with col_cfg:
        st.markdown("### ⚙️ Configurações do TURBO++ ULTRA")
        idx_alvo = st.number_input(
            "Índice alvo (1 = primeira série):",
            min_value=1,
            max_value=len(df),
            value=len(df),
            step=1,
        )
        idx_zero = idx_alvo - 1

        top_n = st.slider("Top-N final:", min_value=5, max_value=80, value=20, step=5)
        n_s6 = st.slider("Quantidade de séries S6 Profundo ULTRA:", 50, 400, 200, 50)
        n_mc = st.slider("Quantidade de séries Monte Carlo ULTRA:", 300, 1200, 800, 100)
        n_micro = st.slider("Micro-Leque (variações por série base):", 5, 40, 20, 5)

        rodar = st.button("Executar TURBO++ ULTRA ADAPTATIVO", type="primary")

    with col_info:
        st.markdown("### 🧱 Série alvo (carro atual)")
        st.write(f"ID: **{df.iloc[idx_zero]['id']}**")
        st.code(series_to_str(df.iloc[idx_zero][pcols].values.tolist()), language="text")

    if not rodar:
        st.info("Configure os parâmetros e clique em **Executar TURBO++ ULTRA ADAPTATIVO**.")
        return

    # Execução do motor
    with st.spinner("Rodando S6 Profundo ULTRA, Monte Carlo Profundo ULTRA e Micro-Leque ULTRA..."):
        res = executar_turbo_ultra_adaptativo_para_indice(
            df=df,
            idx_alvo=idx_zero,
            top_n=top_n,
            n_s6=n_s6,
            n_mc=n_mc,
            n_micro=n_micro,
        )

    df_s6 = res["df_s6"]
    df_mc = res["df_mc"]
    df_micro = res["df_micro"]
    df_fusao = res["df_fusao"]
    k_star = res["k_estrela"]
    regime = res["regime"]
    pesos = res["pesos"]
    qds_local = res["qds_local"]

    st.markdown("### 🌟 Contexto adaptativo")
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.metric("k* (sentinela)", f"{k_star:.1f} %")
    with col_c2:
        st.metric("QDS local (janela curta)", f"{qds_local:.3f}")
    with col_c3:
        s6_w, mc_w, micro_w = pesos
        st.write("**Pesos por regime:**")
        st.write(f"S6: **{s6_w:.2f}**  •  Monte Carlo: **{mc_w:.2f}**  •  Micro-Leque: **{micro_w:.2f}**")

    st.info(mensagem_contexto_kstar(k_star, regime))

    st.markdown("### 🧠 S6 Profundo ULTRA — núcleo determinístico")
    if df_s6 is not None and not df_s6.empty:
        st.dataframe(
            df_s6.head(min(30, len(df_s6)))[["series", "score_s6"]],
            use_container_width=True,
        )
    else:
        st.write("Nenhuma série gerada pelo S6 Profundo ULTRA (histórico muito curto).")

    st.markdown("### 🎲 Monte Carlo Profundo ULTRA — motor estocástico")
    if df_mc is not None and not df_mc.empty:
        st.dataframe(
            df_mc.head(min(30, len(df_mc)))[["series", "score_mc"]],
            use_container_width=True,
        )
    else:
        st.write("Nenhuma série gerada pelo Monte Carlo Profundo ULTRA (histórico muito curto).")

    st.markdown("### 🌪️ Micro-Leque ULTRA — variações finas")
    if df_micro is not None and not df_micro.empty:
        st.dataframe(
            df_micro.head(min(30, len(df_micro)))[["series", "score_micro"]],
            use_container_width=True,
        )
    else:
        st.write("Nenhuma série gerada pelo Micro-Leque ULTRA (falta de base ou histórico).")

    st.markdown("### 🔚 Fusão ULTRA ADAPTATIVA — Top-N final")
    if df_fusao is None or df_fusao.empty:
        st.error("Fusão não retornou nenhuma série. Verifique se há histórico suficiente.")
        return

    st.dataframe(
        df_fusao.head(top_n)[["rank_fusao", "series", "score_fusao", "rank_s6", "rank_mc", "rank_micro"]],
        use_container_width=True,
    )

    melhor = df_fusao.iloc[0]["series"]
    st.markdown("### 🎯 Previsão Final TURBO++ ULTRA (Adaptativo)")
    st.code(series_to_str(melhor), language="text")

    # Mensagem de regime
    st.success(descricao_regime(regime))


# ============================================================
# PAINEL 5 — Modo Replay Automático do Histórico
# ============================================================

def painel_replay_automatico():
    st.markdown("## 📅 Modo Replay Automático do Histórico")

    df = carregar_df_sessao()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        return

    pcols = [c for c in df.columns if c.startswith("p")]
    if not pcols:
        st.error("Não foram detectadas colunas de passageiros (p1..pN).")
        return

    st.markdown(
        "O Replay Automático executa o **TURBO++ ULTRA ADAPTATIVO** ao longo de um "
        "intervalo de índices e compara com o que realmente ocorreu, simulando um backtest."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        idx_inicio = st.number_input(
            "Índice inicial (1):",
            min_value=1,
            max_value=max(1, len(df) - 1),
            value=max(1, len(df) - 60),
            step=1,
        )
    with col_b:
        idx_fim = st.number_input(
            "Índice final (precisa ter próximo conhecido):",
            min_value=idx_inicio,
            max_value=len(df) - 1,
            value=len(df) - 1,
            step=1,
        )

    top_n = st.slider("Top-N usado para acerto no Replay ULTRA:", 5, 50, 20, 5)

    if st.button("Executar Replay ULTRA / Backtest REAL"):
        with st.spinner("Executando Replay ULTRA / Backtest REAL..."):
            res = executar_replay_ultra_backtest(
                df=df,
                idx_inicio=idx_inicio - 1,
                idx_fim=idx_fim - 1,
                top_n=top_n,
            )

        tabela = res["tabela"]
        hits = res["hits"]
        total = res["total"]
        taxa = res["taxa_acerto"]

        if tabela is None or tabela.empty:
            st.warning("Nenhum resultado produzido. Tente ajustar a janela ou verifique o histórico.")
            return

        st.markdown("### 📋 Resultado detalhado do Replay ULTRA")
        st.dataframe(tabela, use_container_width=True)

        st.markdown("### 📈 Síntese de desempenho")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tentativas (carros re-jogados)", total)
        with col2:
            st.metric("Acertos em Top-N", hits)
        with col3:
            st.metric("Taxa de acerto", f"{taxa*100:.2f} %")

        st.info(
            "Este Replay ULTRA funciona como um **Backtest REAL focal**, "
            "reproduzindo as decisões que o Modo TURBO++ ULTRA ADAPTATIVO tomaria em cada carro."
        )


# ============================================================
# PAINEL 6 — Testes de Confiabilidade (QDS / Backtest / Monte Carlo)
# ============================================================

def painel_testes_confiabilidade():
    st.markdown("## 🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)")

    df = carregar_df_sessao()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        return

    pcols = [c for c in df.columns if c.startswith("p")]
    if not pcols:
        st.error("Não foram detectadas colunas de passageiros (p1..pN).")
        return

    st.markdown(
        "Este painel consolida a visão de **QDS REAL**, "
        "**Backtest REAL** (via Replay ULTRA) e "
        "**Monte Carlo Profundo ULTRA** em janelas configuráveis."
    )

    janela_qds = st.slider("Janela para QDS REAL (nº de séries recentes):", 20, 200, 60, 10)
    top_n_qds = st.slider("Top-N para acerto no cálculo de QDS:", 5, 50, 20, 5)

    if st.button("Calcular QDS REAL (global da janela)"):
        with st.spinner("Calculando QDS REAL a partir de backtest interno..."):
            qds_val = calcular_qds_backtest_simples(
                df,
                passenger_cols=pcols,
                janela=min(janela_qds, len(df) - 2),
                top_n=top_n_qds,
            )
        st.metric("QDS REAL (janela global)", f"{qds_val:.3f}")
        if qds_val < 0.05:
            st.warning(
                "QDS muito baixo — regime de **ruptura prolongada**. "
                "A estrada não oferece padrão profundo confiável em janelas longas."
            )
        elif qds_val < 0.15:
            st.info(
                "QDS baixo, porém não nulo — regime de **transição / instabilidade**. "
                "Há bolsões de previsibilidade, mas o padrão global ainda é frágil."
            )
        else:
            st.success(
                "QDS moderado/alto — a estrada apresenta **padrão aproveitável** "
                "em janelas longas. S6 e micro-estruturas tendem a funcionar melhor."
            )

    st.markdown("### 🎲 Monte Carlo Profundo ULTRA — visão estatística global")
    if st.button("Gerar amostra Monte Carlo Profundo ULTRA para diagnóstico global"):
        with st.spinner("Gerando amostra global de Monte Carlo Profundo ULTRA..."):
            df_mc = gerar_monte_carlo_profundo_ultra(
                df,
                passenger_cols=pcols,
                n_series=1200,
                janela=200,
            )
        st.write("Prévia das séries mais frequentes (Monte Carlo ULTRA):")
        st.dataframe(df_mc.head(40), use_container_width=True)
        st.info(
            "As séries mais frequentes no Monte Carlo Profundo ULTRA indicam "
            "padrões estatísticos de curto prazo que o modelo está capturando "
            "no regime atual da estrada."
        )


# ============================================================
# MAIN
# ============================================================

def main():
    configurar_pagina()
    painel = main_sidebar()

    if painel == "📥 Histórico — Entrada":
        painel_historico_entrada()
    elif painel == "🔍 Pipeline V14-FLEX (TURBO++)":
        painel_pipeline_v14_flex()
    elif painel == "🚨 Monitor de Risco (k & k*)":
        painel_monitor_risco()
    elif painel == "🚀 Modo TURBO++ ULTRA (Adaptativo)":
        painel_modo_turbo_ultra_adaptativo()
    elif painel == "📅 Modo Replay Automático do Histórico":
        painel_replay_automatico()
    elif painel == "🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)":
        painel_testes_confiabilidade()
    else:
        st.write("Painel não reconhecido.")


if __name__ == "__main__":
    main()
# ============================================================
# IDX ULTRA (núcleo ponderado da estrada)
# ============================================================

def calcular_idx_ultra(df: pd.DataFrame) -> Dict[str, Any]:
    """
    IDX ULTRA: índice central da estrada, com médias ponderadas por k.
    Retorna:
      - idx_passageiros: média ponderada por posição (p1..pN)
      - idx_global: média de todos os passageiros ponderada
    """
    if df is None or df.empty:
        return {"idx_passageiros": {}, "idx_global": 0.0}

    passenger_cols = [c for c in df.columns if c.startswith("p")]
    if not passenger_cols:
        return {"idx_passageiros": {}, "idx_global": 0.0}

    k_vals = df["k"].astype(float).values
    w = k_vals + 1.0  # todo mundo conta, mas k>0 pesa mais

    idx_pass = {}
    all_vals = []

    for c in passenger_cols:
        vals = df[c].astype(float).values
        all_vals.extend(vals.tolist())
        if np.sum(w) == 0:
            m = float(np.mean(vals))
        else:
            m = float(np.average(vals, weights=w))
        idx_pass[c] = m

    if all_vals:
        if np.sum(w) == 0:
            idx_global = float(np.mean(all_vals))
        else:
            # aproximar usando média de idx_pass
            idx_global = float(np.mean(list(idx_pass.values())))
    else:
        idx_global = 0.0

    return {
        "idx_passageiros": idx_pass,
        "idx_global": idx_global,
    }


# ============================================================
# IPF / IPO REFINADOS (índices de padrão futuro / atual)
# ============================================================

def calcular_ipf_ipo(df: pd.DataFrame) -> Dict[str, Any]:
    """
    IPF (Índice de Padrão Futuro) e IPO (Índice Padrão Atual).
    Implementação simplificada porém real: correlações e tendências entre
    séries consecutivas, ponderadas por k.
    """
    if df is None or df.empty or len(df) < 2:
        return {"ipf": 0.0, "ipo": 0.0}

    passenger_cols = [c for c in df.columns if c.startswith("p")]
    if not passenger_cols:
        return {"ipf": 0.0, "ipo": 0.0}

    k_vals = df["k"].astype(float).values
    w = k_vals + 1.0

    # IPO: estabilidade local do padrão (variação média entre séries consecutivas)
    diffs = []
    for i in range(len(df) - 1):
        a = df.iloc[i][passenger_cols].values.astype(float)
        b = df.iloc[i + 1][passenger_cols].values.astype(float)
        diffs.append(np.linalg.norm(a - b, ord=1))
    if not diffs:
        ipo = 0.0
    else:
        diffs_arr = np.array(diffs)
        ipo = float(1.0 / (1.0 + np.mean(diffs_arr)))  # quanto menor a diferença, maior o IPO

    # IPF: "alinhamento" do futuro com o presente ponderado por k
    # Aqui tomamos a correlação média entre a série e a seguinte
    corrs = []
    for i in range(len(df) - 1):
        a = df.iloc[i][passenger_cols].values.astype(float)
        b = df.iloc[i + 1][passenger_cols].values.astype(float)
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        c = np.corrcoef(a, b)[0, 1]
        corrs.append(c)
    if not corrs:
        ipf = 0.0
    else:
        ipf = float(np.mean(corrs))

    return {"ipf": ipf, "ipo": ipo}


# ============================================================
# S6 PROFUNDO ULTRA — Núcleo determinístico
# ============================================================

def gerar_previsoes_s6_profundo_ultra(
    df: pd.DataFrame,
    idx_alvo: int,
    n_top: int = 200,
) -> pd.DataFrame:
    """
    S6 Profundo ULTRA:
      - baseia-se na similaridade entre a série alvo e séries anteriores
      - utiliza a série seguinte de cada vizinho como candidato futuro
      - agrega as distâncias para formar um score determinístico
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["series", "score_s6", "origem"])

    passenger_cols = [c for c in df.columns if c.startswith("p")]
    if not passenger_cols:
        return pd.DataFrame(columns=["series", "score_s6", "origem"])

    if idx_alvo <= 0 or idx_alvo >= len(df):
        idx_alvo = len(df) - 1

    target = df.iloc[idx_alvo][passenger_cols].values.astype(float)

    registros = []
    for i in range(0, idx_alvo):
        if i + 1 >= len(df):
            break
        atual = df.iloc[i][passenger_cols].values.astype(float)
        prox = df.iloc[i + 1][passenger_cols].values.astype(int).tolist()
        dist = float(np.linalg.norm(atual - target, ord=1))
        cid_atual = df.iloc[i]["id"]
        cid_prox = df.iloc[i + 1]["id"]
        registros.append((tuple(prox), dist, (cid_atual, cid_prox)))

    if not registros:
        return pd.DataFrame(columns=["series", "score_s6", "origem"])

    serie_map: Dict[Tuple[int, ...], List[float]] = {}
    origem_map: Dict[Tuple[int, ...], List[Tuple[str, str]]] = {}

    for serie, dist, origem in registros:
        serie_map.setdefault(serie, []).append(dist)
        origem_map.setdefault(serie, []).append(origem)

    rows = []
    for serie, ds in serie_map.items():
        score = float(np.mean(ds))
        origens = origem_map.get(serie, [])
        rows.append(
            {
                "series": list(serie),
                "score_s6": score,
                "origem": origens,
            }
        )

    df_s6 = pd.DataFrame(rows).sort_values("score_s6", ascending=True)
    df_s6 = df_s6.head(n_top).reset_index(drop=True)
    return df_s6


# ============================================================
# MICRO-LEQUE ULTRA — Variações finas em torno dos núcleos
# ============================================================

def gerar_micro_leque_ultra(
    base_df: pd.DataFrame,
    n_micro_por_serie: int = 15,
) -> pd.DataFrame:
    """
    Micro-Leque ULTRA:
      - recebe séries base (por ex. vindas do S6 ou da fusão parcial)
      - gera pequenas variações locais (±1, ±2) em alguns passageiros
      - mantém o tamanho e a faixa [MIN, MAX]
    """
    if base_df is None or base_df.empty or "series" not in base_df.columns:
        return pd.DataFrame(columns=["series", "score_micro"])

    geradas = []
    for _, row in base_df.iterrows():
        base_series = row["series"]
        base_series = normalizar_serie_lista(base_series)
        if not base_series:
            continue

        for _ in range(n_micro_por_serie):
            s = base_series.copy()
            # escolher 1 ou 2 posições para perturbar
            n_pert = random.choice([1, 2])
            for __ in range(n_pert):
                idx = random.randint(0, len(s) - 1)
                delta = random.choice([-2, -1, 1, 2])
                novo_val = s[idx] + delta
                if novo_val < MIN_PASSAGEIRO:
                    novo_val = MIN_PASSAGEIRO
                if novo_val > MAX_PASSAGEIRO:
                    novo_val = MAX_PASSAGEIRO
                s[idx] = novo_val
            s = normalizar_serie_lista(s)
            if len(s) != len(base_series):
                continue
            geradas.append(tuple(s))

    if not geradas:
        return pd.DataFrame(columns=["series", "score_micro"])

    cont = {}
    for s in geradas:
        cont[s] = cont.get(s, 0) + 1

    rows = []
    for s, freq in cont.items():
        rows.append({"series": list(s), "score_micro": 1.0 / (freq + 1e-9)})

    df_micro = pd.DataFrame(rows).sort_values("score_micro", ascending=True).reset_index(drop=True)
    return df_micro


# ============================================================
# MONTE CARLO PROFUNDO ULTRA — simulações em janela configurável
# ============================================================

def gerar_monte_carlo_profundo_ultra(
    df: pd.DataFrame,
    passenger_cols: List[str],
    n_series: int = 800,
    janela: int = 100,
) -> pd.DataFrame:
    """
    Monte Carlo Profundo ULTRA:
      - usa a janela recente para estimar distribuições empíricas
      - gera muitas séries aleatórias consistentes com a estrada local
      - foca previsibilidade curta (caos / ruptura)
    """
    if df is None or df.empty or not passenger_cols:
        return pd.DataFrame(columns=["series", "score_mc"])

    sub = df.tail(janela).copy()
    if sub.empty:
        sub = df.copy()

    distros = {}
    for c in passenger_cols:
        vals = sub[c].values
        vals = [v for v in vals if MIN_PASSAGEIRO <= v <= MAX_PASSAGEIRO]
        if not vals:
            vals = list(range(MIN_PASSAGEIRO, MAX_PASSAGEIRO + 1))
        distros[c] = vals

    geradas = []
    for _ in range(n_series):
        s = []
        for c in passenger_cols:
            vals = distros[c]
            v = random.choice(vals)
            s.append(v)
        s = normalizar_serie_lista(s)
        if len(s) != len(passenger_cols):
            # se perder tamanho, preenche com valores extras
            while len(s) < len(passenger_cols):
                v_extra = random.randint(MIN_PASSAGEIRO, MAX_PASSAGEIRO)
                if v_extra not in s:
                    s.append(v_extra)
            s = sorted(s)
        geradas.append(tuple(s))

    cont = {}
    for s in geradas:
        cont[s] = cont.get(s, 0) + 1

    rows = []
    for s, freq in cont.items():
        rows.append({"series": list(s), "score_mc": 1.0 / (freq + 1e-9)})

    df_mc = pd.DataFrame(rows).sort_values("score_mc", ascending=True).reset_index(drop=True)
    return df_mc


# ============================================================
# FUSÃO ULTRA ADAPTATIVA (S6 + MC + Micro) por regime
# ============================================================

def adicionar_rank(df: pd.DataFrame, col_score: str, col_rank: str) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(col_score, ascending=True).reset_index(drop=True)
    df[col_rank] = np.arange(1, len(df) + 1)
    return df


def fundir_candidatos_ultra_adaptativo(
    df_s6: pd.DataFrame,
    df_mc: pd.DataFrame,
    df_micro: pd.DataFrame,
    pesos: Tuple[float, float, float],
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Fusão ULTRA adaptativa:
      - combina S6, Monte Carlo e Micro-Leque via ranks e pesos
      - pesos dependem do regime (k* / QDS)
    """
    peso_s6, peso_mc, peso_micro = pesos

    if (df_s6 is None or df_s6.empty) and (df_mc is None or df_mc.empty) and (df_micro is None or df_micro.empty):
        return pd.DataFrame(columns=["series", "score_fusao", "rank_fusao"])

    if df_s6 is not None and not df_s6.empty:
        df_s6 = adicionar_rank(df_s6, "score_s6", "rank_s6")
    else:
        df_s6 = pd.DataFrame(columns=["series", "score_s6", "rank_s6"])

    if df_mc is not None and not df_mc.empty:
        df_mc = adicionar_rank(df_mc, "score_mc", "rank_mc")
    else:
        df_mc = pd.DataFrame(columns=["series", "score_mc", "rank_mc"])

    if df_micro is not None and not df_micro.empty:
        df_micro = adicionar_rank(df_micro, "score_micro", "rank_micro")
    else:
        df_micro = pd.DataFrame(columns=["series", "score_micro", "rank_micro"])

    # União de chaves
    all_keys = set()
    for s in df_s6["series"].tolist():
        all_keys.add(series_to_tuple(s))
    for s in df_mc["series"].tolist():
        all_keys.add(series_to_tuple(s))
    for s in df_micro["series"].tolist():
        all_keys.add(series_to_tuple(s))

    rows = []
    for key in all_keys:
        s_list = list(key)
        row = {"series": s_list}

        rank_s6 = 9999
        if not df_s6.empty:
            mask = df_s6["series"].apply(lambda x: series_to_tuple(x) == key)
            if mask.any():
                rank_s6 = int(df_s6.loc[mask, "rank_s6"].iloc[0])
        row["rank_s6"] = rank_s6

        rank_mc = 9999
        if not df_mc.empty:
            mask = df_mc["series"].apply(lambda x: series_to_tuple(x) == key)
            if mask.any():
                rank_mc = int(df_mc.loc[mask, "rank_mc"].iloc[0])
        row["rank_mc"] = rank_mc

        rank_micro = 9999
        if not df_micro.empty:
            mask = df_micro["series"].apply(lambda x: series_to_tuple(x) == key)
            if mask.any():
                rank_micro = int(df_micro.loc[mask, "rank_micro"].iloc[0])
        row["rank_micro"] = rank_micro

        score = (
            peso_s6 * rank_s6 +
            peso_mc * rank_mc +
            peso_micro * rank_micro
        )
        row["score_fusao"] = float(score)

        rows.append(row)

    df_mix = pd.DataFrame(rows).sort_values("score_fusao", ascending=True).reset_index(drop=True)
    df_mix["rank_fusao"] = np.arange(1, len(df_mix) + 1)
    return df_mix.head(max(top_n * 3, top_n))


# ============================================================
# MOTOR TURBO++ ULTRA ADAPTATIVO (para um índice alvo)
# ============================================================

def executar_turbo_ultra_adaptativo_para_indice(
    df: pd.DataFrame,
    idx_alvo: int,
    top_n: int = 20,
    n_s6: int = 200,
    n_mc: int = 800,
    n_micro: int = 20,
) -> Dict[str, Any]:
    """
    Executa todo o motor TURBO++ ULTRA ADAPTATIVO para um índice alvo:
      - S6 Profundo ULTRA
      - Monte Carlo Profundo ULTRA
      - Micro-Leque ULTRA
      - Fusão adaptativa por regime (k* + QDS)
    Retorna:
      {
        "df_s6": ...,
        "df_mc": ...,
        "df_micro": ...,
        "df_fusao": ...,
        "k_estrela": float,
        "regime": "padrao|transicao|ruptura",
        "pesos": (s6, mc, micro),
        "qds_local": float,
      }
    """
    res = {
        "df_s6": pd.DataFrame(),
        "df_mc": pd.DataFrame(),
        "df_micro": pd.DataFrame(),
        "df_fusao": pd.DataFrame(),
        "k_estrela": 0.0,
        "regime": "ruptura",
        "pesos": (0.1, 0.7, 0.2),
        "qds_local": 0.0,
    }

    if df is None or df.empty:
        return res

    passenger_cols = [c for c in df.columns if c.startswith("p")]
    if not passenger_cols:
        return res

    if idx_alvo < 0:
        idx_alvo = 0
    if idx_alvo >= len(df):
        idx_alvo = len(df) - 1

    # QDS local aproximado em janela curta ao redor do alvo
    # usamos apenas a parte até o índice alvo (para não olhar futuro)
    df_hist = df.iloc[: idx_alvo + 1].copy()
    qds_local = calcular_qds_backtest_simples(
        df_hist,
        passenger_cols=passenger_cols,
        janela=min(20, len(df_hist) - 2) if len(df_hist) > 2 else 0,
        top_n=min(top_n, 15),
    )

    # k* baseado também apenas no histórico disponível até o alvo
    k_star = calcular_k_estrela(df_hist, janela=80)

    regime = determinar_regime_por_kstar(k_star, qds_local=qds_local)
    pesos = obter_pesos_por_regime(regime)

    # S6 Profundo ULTRA
    df_s6 = gerar_previsoes_s6_profundo_ultra(
        df_hist,
        idx_alvo=len(df_hist) - 1,
        n_top=n_s6,
    )

    # Monte Carlo Profundo ULTRA
    df_mc = gerar_monte_carlo_profundo_ultra(
        df_hist,
        passenger_cols=passenger_cols,
        n_series=n_mc,
        janela=120,
    )

    # Micro-Leque ULTRA baseado na saída do S6 (se vazio, cair para MC)
    base_micro = df_s6 if not df_s6.empty else df_mc
    df_micro = gerar_micro_leque_ultra(base_micro, n_micro_por_serie=n_micro)

    # Fusão adaptativa
    df_fusao = fundir_candidatos_ultra_adaptativo(
        df_s6=df_s6,
        df_mc=df_mc,
        df_micro=df_micro,
        pesos=pesos,
        top_n=top_n,
    )

    res.update(
        {
            "df_s6": df_s6,
            "df_mc": df_mc,
            "df_micro": df_micro,
            "df_fusao": df_fusao,
            "k_estrela": k_star,
            "regime": regime,
            "pesos": pesos,
            "qds_local": qds_local,
        }
    )
    return res
# ============================================================
# REPLAY LIGHT — diagnóstico pontual por índice
# ============================================================

def executar_replay_light(
    df: pd.DataFrame,
    idx_alvo: int,
    top_n: int = 20,
) -> Dict[str, Any]:
    """
    Replay LIGHT:
      - Executa o TURBO++ ULTRA ADAPTATIVO em um único ponto do histórico
      - Compara a Previsão Final com a série real seguinte (se existir)
    """
    res_turbo = executar_turbo_ultra_adaptativo_para_indice(
        df=df,
        idx_alvo=idx_alvo,
        top_n=top_n,
        n_s6=200,
        n_mc=800,
        n_micro=20,
    )

    passenger_cols = [c for c in df.columns if c.startswith("p")]
    real_next_series = None
    hit = False

    if idx_alvo + 1 < len(df):
        real_next_series = df.iloc[idx_alvo + 1][passenger_cols].values.astype(int).tolist()
        real_tuple = series_to_tuple(real_next_series)
        if not res_turbo["df_fusao"].empty:
            top_series = [series_to_tuple(s) for s in res_turbo["df_fusao"]["series"].tolist()[:top_n]]
            hit = real_tuple in top_series

    return {
        "turbo": res_turbo,
        "real_next": real_next_series,
        "hit": hit,
    }


# ============================================================
# REPLAY ULTRA / BACKTEST REAL — janela de índices
# ============================================================

def executar_replay_ultra_backtest(
    df: pd.DataFrame,
    idx_inicio: int,
    idx_fim: int,
    top_n: int = 20,
) -> Dict[str, Any]:
    """
    Replay ULTRA / Backtest REAL:
      - para cada índice i em [idx_inicio, idx_fim], executa o TURBO++ ULTRA adaptativo
        usando apenas o histórico até i
      - compara com a série real seguinte, medindo taxa de acerto, regimes, etc.
    """
    passenger_cols = [c for c in df.columns if c.startswith("p")]
    if not passenger_cols:
        return {
            "tabela": pd.DataFrame(),
            "hits": 0,
            "total": 0,
            "taxa_acerto": 0.0,
        }

    idx_inicio = max(0, idx_inicio)
    idx_fim = min(len(df) - 2, idx_fim)  # precisa existir "próximo"
    if idx_fim <= idx_inicio:
        return {
            "tabela": pd.DataFrame(),
            "hits": 0,
            "total": 0,
            "taxa_acerto": 0.0,
        }

    registros = []
    hits = 0
    total = 0

    for i in range(idx_inicio, idx_fim + 1):
        df_hist = df.iloc[: i + 1].copy()
        if len(df_hist) < 6:
            continue

        turbo_res = executar_turbo_ultra_adaptativo_para_indice(
            df=df_hist,
            idx_alvo=len(df_hist) - 1,
            top_n=top_n,
            n_s6=150,
            n_mc=500,
            n_micro=15,
        )

        df_fusao = turbo_res["df_fusao"]
        if df_fusao is None or df_fusao.empty:
            continue

        top_series = [series_to_tuple(s) for s in df_fusao["series"].tolist()[:top_n]]

        real_series = df.iloc[i + 1][passenger_cols].values.astype(int).tolist()
        real_tuple = series_to_tuple(real_series)

        acerto = real_tuple in top_series
        total += 1
        if acerto:
            hits += 1

        melhor = df_fusao.iloc[0]["series"]
        registros.append(
            {
                "id_atual": df.iloc[i]["id"],
                "id_real_prox": df.iloc[i + 1]["id"],
                "serie_real": series_to_str(real_series),
                "melhor_prev": series_to_str(melhor),
                "acerto_topN": acerto,
                "k_estrela_local": turbo_res["k_estrela"],
                "regime_local": turbo_res["regime"],
                "qds_local": turbo_res["qds_local"],
            }
        )

    taxa = float(hits / total) if total > 0 else 0.0
    tabela = pd.DataFrame(registros) if registros else pd.DataFrame()
    return {
        "tabela": tabela,
        "hits": hits,
        "total": total,
        "taxa_acerto": taxa,
    }


# ============================================================
# SUPORTE À INTERFACE — mensagens de contexto por k* / regime
# ============================================================

def mensagem_contexto_kstar(k_star: float, regime: str) -> str:
    base = f"k* = {k_star:.1f}% — "
    if regime == "padrao":
        return base + "Ambiente estável forte, padrão profundo dominante. S6 lidera a fusão."
    elif regime == "transicao":
        return base + "Ambiente de transição / pré-ruptura. Mistura equilibrada de S6, Micro-Leque e Monte Carlo."
    else:
        return base + "Ruptura / macro-caos. Monte Carlo assume protagonismo para capturar previsibilidade curta."


def mensagem_barometro(bar: Dict[str, Any]) -> str:
    estado = bar.get("estado", "indefinido")
    k_medio = bar.get("k_medio", 0.0)
    freq_zero = bar.get("freq_k_zero", 0.0)
    if estado == "estavel":
        return f"🟢 Estrada estável — k médio ≈ {k_medio:.2f}, poucos carros com k=0 ({freq_zero*100:.1f}%)."
    elif estado == "transicao":
        return f"🟡 Estrada em transição — k médio ≈ {k_medio:.2f}, regime misto, atenção à mudança de padrão."
    elif estado == "ruptura":
        return f"🔴 Estrada em ruptura — k médio ≈ {k_medio:.2f}, muitos carros com k=0 ({freq_zero*100:.1f}%)."
    else:
        return "⚪ Barômetro indefinido — histórico insuficiente ou dados inconsistentes."


def mensagem_k_novo_significado() -> str:
    return (
        "📌 **Novo significado de k**:\n\n"
        "- k representa **o número de guardas** que acertaram **exatamente** o carro (todos os passageiros na ordem correta).\n"
        "- k=0: nenhum guarda cravou exatamente aquela série.\n"
        "- k>0: houve guardas que sabiam exatamente quais passageiros estariam naquele carro.\n"
        "- O painel de risco usa a distribuição de k para avaliar raridade, concentração e sensibilidade da estrada."
    )


# ============================================================
# INÍCIO DO APP STREAMLIT
# ============================================================

def configurar_pagina():
    st.set_page_config(
        page_title="Predict Cars V14-FLEX ULTRA REAL (TURBO++ ADAPTATIVO)",
        layout="wide",
    )


def carregar_df_sessao() -> Optional[pd.DataFrame]:
    return st.session_state.get("df", None)


def salvar_df_sessao(df: pd.DataFrame):
    st.session_state["df"] = df


def main_sidebar():
    st.sidebar.markdown("## 🚗 Predict Cars V14-FLEX ULTRA REAL (TURBO++)")
    st.sidebar.markdown("Versão FLEX + REPLAY + TURBO++ ULTRA ADAPTATIVO por k*")

    painel = st.sidebar.radio(
        "Escolha o painel:",
        [
            "📥 Histórico — Entrada",
            "🔍 Pipeline V14-FLEX (TURBO++)",
            "🚨 Monitor de Risco (k & k*)",
            "🚀 Modo TURBO++ ULTRA (Adaptativo)",
            "📅 Modo Replay Automático do Histórico",
            "🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)",
        ],
    )
    return painel


# ============================================================
# PAINEL 1 — Histórico — Entrada (FLEX)
# ============================================================

def painel_historico_entrada():
    st.markdown("## 📥 Histórico — Entrada (FLEX)")
    st.markdown(
        "Entrada FLEX com número variável de passageiros, "
        "detecção automática da coluna k e preparação completa para o pipeline ULTRA."
    )

    df = carregar_df_sessao()

    opc = st.radio(
        "Como deseja carregar o histórico?",
        ["Enviar arquivo CSV", "Copiar e colar o histórico"],
    )

    if opc == "Enviar arquivo CSV":
        file = st.file_uploader("Selecione o arquivo CSV:", type=["csv"])
        if file is not None:
            try:
                df_raw = pd.read_csv(file)
                df = preparar_historico_flex_from_csv(df_raw)
                salvar_df_sessao(df)
                st.success("Histórico carregado e preparado com sucesso!")
                st.write("Prévia do histórico:")
                st.dataframe(df.head(50))
            except Exception as e:
                st.error(f"Erro ao carregar CSV: {e}")

    else:
        texto = st.text_area(
            "Cole o histórico aqui (linhas no formato C1;41;5;4;52;30;33;0, por exemplo):",
            height=240,
        )
        if st.button("Processar histórico colado"):
            try:
                df = preparar_historico_flex_from_text(texto)
                if df is None or df.empty:
                    st.warning("Não foi possível interpretar o histórico. Verifique o formato.")
                else:
                    salvar_df_sessao(df)
                    st.success("Histórico colado e preparado com sucesso!")
                    st.write("Prévia do histórico:")
                    st.dataframe(df.head(50))
            except Exception as e:
                st.error(f"Erro ao processar o texto: {e}")

    if df is not None and not df.empty:
        st.markdown("### 📊 Resumo rápido do histórico")
        st.write(f"Total de séries: **{len(df)}**")
        pcols = [c for c in df.columns if c.startswith("p")]
        st.write(f"Número de passageiros por série (detectado): **{len(pcols)}**")
        st.write("Colunas:", ", ".join(df.columns))


# ============================================================
# PAINEL 2 — Pipeline V14-FLEX (TURBO++) — visão estrutural
# ============================================================

def painel_pipeline_v14_flex():
    st.markdown("## 🔍 Pipeline V14-FLEX (TURBO++) — Execução Estrutural")

    df = carregar_df_sessao()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        return

    pcols = [c for c in df.columns if c.startswith("p")]
    if not pcols:
        st.error("Não foram detectadas colunas de passageiros (p1..pN).")
        return

    idx_alvo = st.number_input(
        "Selecione o índice alvo (1 = primeira série carregada):",
        min_value=1,
        max_value=len(df),
        value=len(df),
        step=1,
    )
    idx_alvo_zero = idx_alvo - 1

    bar = calcular_barometro_ultra_real(df)
    k_star = calcular_k_estrela(df)
    idx_info = calcular_idx_ultra(df)
    ipf_ipo = calcular_ipf_ipo(df)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🌡️ Barômetro ULTRA REAL")
        st.write(mensagem_barometro(bar))
        st.markdown("### 🌟 k* ULTRA REAL")
        st.write(f"k* ≈ **{k_star:.1f}%**")
    with col2:
        st.markdown("### 🧭 IDX ULTRA")
        st.write(f"Índice global (média ponderada): **{idx_info['idx_global']:.2f}**")
        st.markdown("### 📐 IPF / IPO (refinados)")
        st.write(f"IPF ≈ **{ipf_ipo['ipf']:.3f}** — IPO ≈ **{ipf_ipo['ipo']:.3f}**")

    st.markdown("### 🧱 Série alvo (estrutura)")
    st.code(
        series_to_str(df.iloc[idx_alvo_zero][pcols].values.tolist()),
        language="text",
    )

    st.markdown(
        "Este painel mostra o **estado estrutural** da estrada (Barômetro, k*, IDX, IPF/IPO), "
        "que são insumos diretos para o **Modo TURBO++ ULTRA ADAPTATIVO**."
    )


# ============================================================
# PAINEL 3 — Monitor de Risco (k & k*)
# ============================================================

def painel_monitor_risco():
    st.markdown("## 🚨 Monitor de Risco (k & k*)")

    df = carregar_df_sessao()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        return

    bar = calcular_barometro_ultra_real(df)
    k_star = calcular_k_estrela(df)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🌡️ Barômetro ULTRA REAL")
        st.write(mensagem_barometro(bar))
        st.markdown("### 🌟 k* — Sentinela dos guardas")
        st.write(f"k* ≈ **{k_star:.1f}%**")
        regime = determinar_regime_por_kstar(k_star, None)
        st.write(descricao_regime(regime))
    with col2:
        st.markdown("### 📊 Distribuição de k (guardas que acertaram exatamente)")
        hist = df["k"].value_counts().sort_index()
        if not hist.empty:
            st.bar_chart(hist)
        else:
            st.write("Sem dados suficientes para plotar a distribuição de k.")

    st.markdown("### 📌 Interpretação do novo k")
    st.markdown(mensagem_k_novo_significado())
# ============================================================
# PAINEL 4 — Modo TURBO++ ULTRA (Adaptativo por k*)
# ============================================================

def painel_modo_turbo_ultra_adaptativo():
    st.markdown("## 🚀 Modo TURBO++ ULTRA (Adaptativo por k*)")

    df = carregar_df_sessao()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        return

    pcols = [c for c in df.columns if c.startswith("p")]
    if not pcols:
        st.error("Não foram detectadas colunas de passageiros (p1..pN).")
        return

    col_cfg, col_info = st.columns([1, 1.1])

    with col_cfg:
        st.markdown("### ⚙️ Configurações do TURBO++ ULTRA")
        idx_alvo = st.number_input(
            "Índice alvo (1 = primeira série):",
            min_value=1,
            max_value=len(df),
            value=len(df),
            step=1,
        )
        idx_zero = idx_alvo - 1

        top_n = st.slider("Top-N final:", min_value=5, max_value=80, value=20, step=5)
        n_s6 = st.slider("Quantidade de séries S6 Profundo ULTRA:", 50, 400, 200, 50)
        n_mc = st.slider("Quantidade de séries Monte Carlo ULTRA:", 300, 1200, 800, 100)
        n_micro = st.slider("Micro-Leque (variações por série base):", 5, 40, 20, 5)

        rodar = st.button("Executar TURBO++ ULTRA ADAPTATIVO", type="primary")

    with col_info:
        st.markdown("### 🧱 Série alvo (carro atual)")
        st.write(f"ID: **{df.iloc[idx_zero]['id']}**")
        st.code(series_to_str(df.iloc[idx_zero][pcols].values.tolist()), language="text")

    if not rodar:
        st.info("Configure os parâmetros e clique em **Executar TURBO++ ULTRA ADAPTATIVO**.")
        return

    # Execução do motor
    with st.spinner("Rodando S6 Profundo ULTRA, Monte Carlo Profundo ULTRA e Micro-Leque ULTRA..."):
        res = executar_turbo_ultra_adaptativo_para_indice(
            df=df,
            idx_alvo=idx_zero,
            top_n=top_n,
            n_s6=n_s6,
            n_mc=n_mc,
            n_micro=n_micro,
        )

    df_s6 = res["df_s6"]
    df_mc = res["df_mc"]
    df_micro = res["df_micro"]
    df_fusao = res["df_fusao"]
    k_star = res["k_estrela"]
    regime = res["regime"]
    pesos = res["pesos"]
    qds_local = res["qds_local"]

    st.markdown("### 🌟 Contexto adaptativo")
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.metric("k* (sentinela)", f"{k_star:.1f} %")
    with col_c2:
        st.metric("QDS local (janela curta)", f"{qds_local:.3f}")
    with col_c3:
        s6_w, mc_w, micro_w = pesos
        st.write("**Pesos por regime:**")
        st.write(f"S6: **{s6_w:.2f}**  •  Monte Carlo: **{mc_w:.2f}**  •  Micro-Leque: **{micro_w:.2f}**")

    st.info(mensagem_contexto_kstar(k_star, regime))

    st.markdown("### 🧠 S6 Profundo ULTRA — núcleo determinístico")
    if df_s6 is not None and not df_s6.empty:
        st.dataframe(
            df_s6.head(min(30, len(df_s6)))[["series", "score_s6"]],
            use_container_width=True,
        )
    else:
        st.write("Nenhuma série gerada pelo S6 Profundo ULTRA (histórico muito curto).")

    st.markdown("### 🎲 Monte Carlo Profundo ULTRA — motor estocástico")
    if df_mc is not None and not df_mc.empty:
        st.dataframe(
            df_mc.head(min(30, len(df_mc)))[["series", "score_mc"]],
            use_container_width=True,
        )
    else:
        st.write("Nenhuma série gerada pelo Monte Carlo Profundo ULTRA (histórico muito curto).")

    st.markdown("### 🌪️ Micro-Leque ULTRA — variações finas")
    if df_micro is not None and not df_micro.empty:
        st.dataframe(
            df_micro.head(min(30, len(df_micro)))[["series", "score_micro"]],
            use_container_width=True,
        )
    else:
        st.write("Nenhuma série gerada pelo Micro-Leque ULTRA (falta de base ou histórico).")

    st.markdown("### 🔚 Fusão ULTRA ADAPTATIVA — Top-N final")
    if df_fusao is None or df_fusao.empty:
        st.error("Fusão não retornou nenhuma série. Verifique se há histórico suficiente.")
        return

    st.dataframe(
        df_fusao.head(top_n)[["rank_fusao", "series", "score_fusao", "rank_s6", "rank_mc", "rank_micro"]],
        use_container_width=True,
    )

    melhor = df_fusao.iloc[0]["series"]
    st.markdown("### 🎯 Previsão Final TURBO++ ULTRA (Adaptativo)")
    st.code(series_to_str(melhor), language="text")

    # Mensagem de regime
    st.success(descricao_regime(regime))


# ============================================================
# PAINEL 5 — Modo Replay Automático do Histórico
# ============================================================

def painel_replay_automatico():
    st.markdown("## 📅 Modo Replay Automático do Histórico")

    df = carregar_df_sessao()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        return

    pcols = [c for c in df.columns if c.startswith("p")]
    if not pcols:
        st.error("Não foram detectadas colunas de passageiros (p1..pN).")
        return

    st.markdown(
        "O Replay Automático executa o **TURBO++ ULTRA ADAPTATIVO** ao longo de um "
        "intervalo de índices e compara com o que realmente ocorreu, simulando um backtest."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        idx_inicio = st.number_input(
            "Índice inicial (1):",
            min_value=1,
            max_value=max(1, len(df) - 1),
            value=max(1, len(df) - 60),
            step=1,
        )
    with col_b:
        idx_fim = st.number_input(
            "Índice final (precisa ter próximo conhecido):",
            min_value=idx_inicio,
            max_value=len(df) - 1,
            value=len(df) - 1,
            step=1,
        )

    top_n = st.slider("Top-N usado para acerto no Replay ULTRA:", 5, 50, 20, 5)

    if st.button("Executar Replay ULTRA / Backtest REAL"):
        with st.spinner("Executando Replay ULTRA / Backtest REAL..."):
            res = executar_replay_ultra_backtest(
                df=df,
                idx_inicio=idx_inicio - 1,
                idx_fim=idx_fim - 1,
                top_n=top_n,
            )

        tabela = res["tabela"]
        hits = res["hits"]
        total = res["total"]
        taxa = res["taxa_acerto"]

        if tabela is None or tabela.empty:
            st.warning("Nenhum resultado produzido. Tente ajustar a janela ou verifique o histórico.")
            return

        st.markdown("### 📋 Resultado detalhado do Replay ULTRA")
        st.dataframe(tabela, use_container_width=True)

        st.markdown("### 📈 Síntese de desempenho")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tentativas (carros re-jogados)", total)
        with col2:
            st.metric("Acertos em Top-N", hits)
        with col3:
            st.metric("Taxa de acerto", f"{taxa*100:.2f} %")

        st.info(
            "Este Replay ULTRA funciona como um **Backtest REAL focal**, "
            "reproduzindo as decisões que o Modo TURBO++ ULTRA ADAPTATIVO tomaria em cada carro."
        )


# ============================================================
# PAINEL 6 — Testes de Confiabilidade (QDS / Backtest / Monte Carlo)
# ============================================================

def painel_testes_confiabilidade():
    st.markdown("## 🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)")

    df = carregar_df_sessao()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada'.")
        return

    pcols = [c for c in df.columns if c.startswith("p")]
    if not pcols:
        st.error("Não foram detectadas colunas de passageiros (p1..pN).")
        return

    st.markdown(
        "Este painel consolida a visão de **QDS REAL**, "
        "**Backtest REAL** (via Replay ULTRA) e "
        "**Monte Carlo Profundo ULTRA** em janelas configuráveis."
    )

    janela_qds = st.slider("Janela para QDS REAL (nº de séries recentes):", 20, 200, 60, 10)
    top_n_qds = st.slider("Top-N para acerto no cálculo de QDS:", 5, 50, 20, 5)

    if st.button("Calcular QDS REAL (global da janela)"):
        with st.spinner("Calculando QDS REAL a partir de backtest interno..."):
            qds_val = calcular_qds_backtest_simples(
                df,
                passenger_cols=pcols,
                janela=min(janela_qds, len(df) - 2),
                top_n=top_n_qds,
            )
        st.metric("QDS REAL (janela global)", f"{qds_val:.3f}")
        if qds_val < 0.05:
            st.warning(
                "QDS muito baixo — regime de **ruptura prolongada**. "
                "A estrada não oferece padrão profundo confiável em janelas longas."
            )
        elif qds_val < 0.15:
            st.info(
                "QDS baixo, porém não nulo — regime de **transição / instabilidade**. "
                "Há bolsões de previsibilidade, mas o padrão global ainda é frágil."
            )
        else:
            st.success(
                "QDS moderado/alto — a estrada apresenta **padrão aproveitável** "
                "em janelas longas. S6 e micro-estruturas tendem a funcionar melhor."
            )

    st.markdown("### 🎲 Monte Carlo Profundo ULTRA — visão estatística global")
    if st.button("Gerar amostra Monte Carlo Profundo ULTRA para diagnóstico global"):
        with st.spinner("Gerando amostra global de Monte Carlo Profundo ULTRA..."):
            df_mc = gerar_monte_carlo_profundo_ultra(
                df,
                passenger_cols=pcols,
                n_series=1200,
                janela=200,
            )
        st.write("Prévia das séries mais frequentes (Monte Carlo ULTRA):")
        st.dataframe(df_mc.head(40), use_container_width=True)
        st.info(
            "As séries mais frequentes no Monte Carlo Profundo ULTRA indicam "
            "padrões estatísticos de curto prazo que o modelo está capturando "
            "no regime atual da estrada."
        )


# ============================================================
# MAIN
# ============================================================

def main():
    configurar_pagina()
    painel = main_sidebar()

    if painel == "📥 Histórico — Entrada":
        painel_historico_entrada()
    elif painel == "🔍 Pipeline V14-FLEX (TURBO++)":
        painel_pipeline_v14_flex()
    elif painel == "🚨 Monitor de Risco (k & k*)":
        painel_monitor_risco()
    elif painel == "🚀 Modo TURBO++ ULTRA (Adaptativo)":
        painel_modo_turbo_ultra_adaptativo()
    elif painel == "📅 Modo Replay Automático do Histórico":
        painel_replay_automatico()
    elif painel == "🧪 Testes de Confiabilidade (QDS / Backtest / Monte Carlo)":
        painel_testes_confiabilidade()
    else:
        st.write("Painel não reconhecido.")


if __name__ == "__main__":
    main()
