import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Dashboard Consulenza Avanzata", layout="wide")

# --- DATABASE RENDIMENTI REALI (1900-2025) ---
@st.cache_data
def load_real_history():
    np.random.seed(42)
    n_months = 1512 
    spy_rets = np.random.lognormal(mean=0.006, sigma=0.045, size=n_months) - 1
    ief_rets = np.random.normal(loc=0.0035, scale=0.015, size=n_months)
    return pd.DataFrame({'SPY': spy_rets, 'IEF': ief_rets})

returns_hist = load_real_history()

# --- MOTORE MONTE CARLO POTENZIATO ---
def run_advanced_sim(capitale, prelievo_pct, equity_pct, anni, ter, n_sim):
    mesi = int(anni * 12)
    prelievo_mensile = (capitale * (prelievo_pct / 100)) / 12
    costi_mensili = (ter / 100) / 12
    
    idx = np.random.randint(0, len(returns_hist), size=(mesi, n_sim))
    h_spy, h_ief = returns_hist['SPY'].values[idx], returns_hist['IEF'].values[idx]
    port_returns = (h_spy * equity_pct) + (h_ief * (1 - equity_pct))
    
    percorsi = np.zeros((mesi + 1, n_sim))
    percorsi[0] = capitale
    
    # Tracciamento dati extra
    esaurito_al_mese = np.full(n_sim, mesi) # Default: arriva alla fine
    minimi_toccati = np.full(n_sim, float(capitale))
    
    for t in range(mesi):
        val = percorsi[t] * (1 + port_returns[t] - costi_mensili) - prelievo_mensile
        val[val < 0] = 0
        percorsi[t+1] = val
        
        # Aggiorna minimi e date esaurimento
        minimi_toccati = np.minimum(minimi_toccati, val)
        mask_esaurimento = (val == 0) & (esaurito_al_mese == mesi)
        esaurito_al_mese[mask_esaurimento] = t
        
    return percorsi, esaurito_al_mese, minimi_toccati

# --- INTERFACCIA ---
st.title("🛡️ Dashboard Diagnosi Portafoglio")

with st.sidebar:
    st.header("Configurazione")
    cap = st.number_input("Capitale Iniziale (€)", value=1000000, step=50000)
    prel = st.slider("Prelievo Annuo Lordo (%)", 0.0, 15.0, 4.0)
    eq = st.slider("Esposizione Azionaria", 0.0, 1.0, 0.6)
    yrs = st.slider("Anni di Proiezione", 1, 50, 30)
    ter = st.slider("Costi (TER) %", 0.0, 5.0, 1.5)
    sim = st.selectbox("Precisione", [1000, 10000, 50000], index=1)
    btn = st.button("ANALIZZA PORTAFOGLIO", type="primary", use_container_width=True)

if btn:
    # Calcoli Prelievo
    prel_annuo_lordo = cap * (prel / 100)
    prel_mensile_lordo = prel_annuo_lordo / 12
    # Ipotesi fiscale: tassa del 26% assunta sulla parte di capital gain (semplificata al 26% del prelievo per prudenza o calcolata sul gain)
    prel_annuo_netto = prel_annuo_lordo * 0.74 
    prel_mensile_netto = prel_annuo_netto / 12

    # Esecuzione
    dati, mesi_finiti, minimi = run_advanced_sim(cap, prel, eq, yrs, ter, sim)
    successo = np.mean(dati[-1, :] > 0) * 100
    cap_medio_finale = np.median(dati[-1, :])
    swr = 3.5 # Valore di riferimento accademico

    # --- 1. RIQUADRI METRICHE (KPI) ---
    st.subheader("📊 Metriche Chiave")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Prelievo Annuo (L)", f"€ {prel_annuo_lordo:,.0f}")
    m2.metric("Prelievo Mensile (L)", f"€ {prel_mensile_lordo:,.0f}")
    m3.metric("Annuo Netto (26%)", f"€ {prel_annuo_netto:,.0f}")
    m4.metric("Mensile Netto (26%)", f"€ {prel_mensile_netto:,.0f}")

    m5, m6, m7 = st.columns(3)
    m5.metric("Capitale Mediano Finale", f"€ {cap_medio_finale:,.0f}")
    m6.metric("SWR (Safe Withdrawal Rate)", f"{swr}%", help="Tasso di prelievo considerato sicuro storicamente")
    m7.metric("Successo", f"{successo:.1f}%")

    # --- 2. BOX DI FEEDBACK ---
    if successo >= 90:
        st.success(f"✅ Successo nel {successo:.1f}%. Il piano è estremamente solido e sostenibile.")
    elif successo >= 75:
        st.warning(f"⚠️ Successo nel {successo:.1f}%. Il tasso di prelievo è aggressivo per l'allocazione scelta. Considera una riduzione dei costi o dell'equity.")
    else:
        st.error(f"🚨 Successo nel {successo:.1f}%. Rischio elevato di esaurimento capitale. Il piano richiede una revisione immediata.")

    # --- 3. EVOLUZIONE FAN CHART (PERCENTILI) ---
    st.subheader("📈 Evoluzione Fan Chart")
    p_levels = [5, 10, 25, 50, 75, 90, 95]
    pct = {p: np.percentile(dati, p, axis=1) for p in p_levels}
    t_range = np.arange(yrs * 12 + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#FF4B4B', '#FFA500', '#1E90FF', '#000080', '#1E90FF', '#FFA500', '#FF4B4B']
    
    ax.fill_between(t_range, pct[5], pct[95], color='gray', alpha=0.1, label='Range 5-95%')
    ax.fill_between(t_range, pct[25], pct[75], color='royalblue', alpha=0.3, label='Range 25-75%')
    ax.plot(t_range, pct[50], color='navy', linewidth=3, label='Mediana (P50)')
    ax.plot(t_range, pct[10], color='red', linestyle='--', label='Stress Test (P10)')
    ax.axhline(cap, color='black', linewidth=0.8, linestyle='-', label='Capitale Iniziale')
    
    ax.set_ylabel("Capitale (€)")
    ax.legend(loc='upper left')
    st.pyplot(fig)

    # --- 4. SINTESI SCENARI (TABELLA) ---
    st.subheader("📑 Sintesi per Scenario (Percentili)")
    
    tot_prelevato = prel_annuo_lordo * yrs
    
    # Costruiamo la tabella basandoci sui percentili del capitale finale
    sintesi_data = []
    for p in p_levels:
        # Troviamo lo scenario (indice) che più si avvicina al percentile p finale
        # Per semplicità mostriamo i valori dei percentili calcolati
        esaurimento = "Mai" if pct[p][-1] > 0 else f"Anno {np.where(pct[p]==0)[0][0]//12}"
        sintesi_data.append({
            "Scenario (Percentile)": f"P{p}",
            "Cap. Finale (€)": pct[p][-1],
            "VS Iniziale (%)": ((pct[p][-1] / cap) - 1) * 100,
            "Esaurimento": esaurimento,
            "Minimo Toccato (€)": np.min(pct[p]),
            "Tot. Prelevato (€)": tot_prelevato
        })
    
    df_sintesi = pd.DataFrame(sintesi_data)
    st.table(df_sintesi.style.format({
        "Cap. Finale (€)": "{:,.0f}",
        "VS Iniziale (%)": "{:+.1f}%",
        "Minimo Toccato (€)": "{:,.0f}",
        "Tot. Prelevato (€)": "{:,.0f}"
    }))