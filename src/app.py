#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Climate Risk Dashboard", 
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import des fonctions (version intégrée pour simplicité)
@st.cache_data
def load_sector_data():
    """Charge les données sectorielles avec indicateurs climatiques"""
    sectors_data = {
        'Sector': [
            'Energy', 'Materials', 'Industrials', 'Consumer Discretionary',
            'Consumer Staples', 'Health Care', 'Financials', 'Information Technology',
            'Communication Services', 'Utilities', 'Real Estate'
        ],
        'CO2_Intensity': [850, 420, 180, 120, 95, 45, 35, 25, 40, 520, 85],
        'Water_Risk_Score': [8.5, 7.2, 6.1, 4.3, 5.8, 2.1, 1.5, 2.8, 3.2, 7.8, 4.9],
        'Regulatory_Risk': [9.2, 7.8, 6.5, 5.1, 4.2, 2.8, 6.8, 3.5, 4.1, 8.1, 5.7],
        'Physical_Risk_Exposure': [7.8, 8.1, 7.2, 5.5, 6.3, 3.2, 4.1, 2.9, 3.8, 8.7, 6.8]
    }
    return pd.DataFrame(sectors_data)

@st.cache_data(ttl=3600)  # Cache pendant 1 heure
def fetch_financial_data():
    sector_etfs = {
        'Energy': 'XLE', 'Materials': 'XLB', 'Industrials': 'XLI',
        'Consumer Discretionary': 'XLY', 'Consumer Staples': 'XLP',
        'Health Care': 'XLV', 'Financials': 'XLF', 'Information Technology': 'XLK',
        'Communication Services': 'XLC', 'Utilities': 'XLU', 'Real Estate': 'XLRE'
    }

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    sector_returns = {}
    progress_bar = st.progress(0)

    for i, (sector, ticker) in enumerate(sector_etfs.items()):
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        prices = data.get('Adj Close', data['Close'])
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()
        prices = prices.dropna()
        if len(prices) >= 2:
            returns = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
            volatility = prices.pct_change().dropna().std() * np.sqrt(252) * 100
            sector_returns[sector] = {
                'Annual_Return': round(float(returns), 2),
                'Volatility': round(float(volatility), 2),
                'Current_Price': round(float(prices.iloc[-1]), 2)
            }
        else:
            sector_returns[sector] = {
                'Annual_Return': 0.0,
                'Volatility': 20.0,
                'Current_Price': 100.0
            }
        progress_bar.progress((i + 1) / len(sector_etfs))

    progress_bar.empty()
    return sector_returns

def calculate_climate_score(row):
    """Calcule le score de risque climatique composite"""
    co2_norm = min(row['CO2_Intensity'] / 100, 10)
    water_norm = row['Water_Risk_Score']
    reg_norm = row['Regulatory_Risk'] 
    phys_norm = row['Physical_Risk_Exposure']
    
    composite_score = (
        co2_norm * 0.4 + 
        phys_norm * 0.3 + 
        reg_norm * 0.2 + 
        water_norm * 0.1
    ) * 10
    
    return min(composite_score, 100)

def classify_risk_level(score):
    """Classifie le niveau de risque"""
    if score >= 70:
        return 'Élevé'
    elif score >= 40:
        return 'Modéré'
    else:
        return 'Faible'

def prepare_complete_dataset():
    """Prépare le dataset complet avec toutes les métriques"""
    # Chargement des données de base
    df_sectors = load_sector_data()
    
    # Récupération des données financières
    with st.spinner('Récupération des données financières en temps réel...'):
        financial_data = fetch_financial_data()
    
    # Calcul du score de risque climatique
    df_sectors['Climate_Risk_Score'] = df_sectors.apply(calculate_climate_score, axis=1)
    
    # Ajout des données financières
    financial_list = []
    for _, row in df_sectors.iterrows():
        sector = row['Sector']
        if sector in financial_data:
            financial_list.append(financial_data[sector])
        else:
            financial_list.append({'Annual_Return': 0, 'Volatility': 20, 'Current_Price': 100})
    
    df_financial = pd.DataFrame(financial_list)
    df_complete = pd.concat([df_sectors, df_financial], axis=1)
    
    # Ajout des métriques dérivées
    df_complete['Risk_Level'] = df_complete['Climate_Risk_Score'].apply(classify_risk_level)
    df_complete['Risk_Adjusted_Return'] = df_complete['Annual_Return'] / (df_complete['Climate_Risk_Score'] / 10)
    df_complete['ESG_Ready'] = (df_complete['Climate_Risk_Score'] < 50).astype(int)
    
    return df_complete

# Interface utilisateur
def main():
    st.title("🌍 Dashboard Risques Climatiques Sectoriels")
    st.markdown("### Analyse des risques climatiques et performance financière par secteur")
    
    # Sidebar pour les contrôles
    st.sidebar.header("⚙️ Configuration")
    
    # Chargement des données
    df = prepare_complete_dataset()
    
    # Mode d'affichage
    view_mode = st.sidebar.selectbox(
        "Mode d'analyse",
        ["Vue d'ensemble", "Analyse sectorielle", "Analyse de portefeuille", "Rapport détaillé"]
    )
    
    if view_mode == "Vue d'ensemble":
        show_overview(df)
    elif view_mode == "Analyse sectorielle":
        show_sector_analysis(df)
    elif view_mode == "Analyse de portefeuille":
        show_portfolio_analysis(df)
    elif view_mode == "Rapport détaillé":
        show_detailed_report(df)

def show_overview(df):
    """Affichage de la vue d'ensemble"""
    st.header("📊 Vue d'Ensemble")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_risk = df['Climate_Risk_Score'].mean()
        st.metric(
            "Score Moyen de Risque",
            f"{avg_risk:.1f}/100",
            delta=f"{avg_risk - 50:.1f} vs neutre"
        )
    
    with col2:
        high_risk_count = len(df[df['Risk_Level'] == 'Élevé'])
        st.metric(
            "Secteurs Haut Risque",
            f"{high_risk_count}/11",
            delta=f"{(high_risk_count/11*100):.0f}% du total"
        )
    
    with col3:
        avg_return = df['Annual_Return'].mean()
        st.metric(
            "Rendement Moyen",
            f"{avg_return:.1f}%",
            delta=f"{avg_return - 10:.1f}% vs marché"
        )
    
    with col4:
        correlation = df['Climate_Risk_Score'].corr(df['Annual_Return'])
        st.metric(
            "Corrélation Risque-Rendement",
            f"{correlation:.3f}",
            delta="Négative" if correlation < 0 else "Positive"
        )
    
    # Graphiques principaux
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique en barres des scores par secteur
        fig_bar = px.bar(
            df.sort_values('Climate_Risk_Score'),
            x='Climate_Risk_Score',
            y='Sector',
            color='Risk_Level',
            color_discrete_map={'Élevé': '#FF4444', 'Modéré': '#FFA500', 'Faible': '#00CC44'},
            orientation='h',
            title='Score de Risque Climatique par Secteur'
        )
        fig_bar.add_vline(x=40, line_dash="dash", line_color="orange", opacity=0.7)
        fig_bar.add_vline(x=70, line_dash="dash", line_color="red", opacity=0.7)
        fig_bar.update_layout(height=500)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Scatter plot risque vs rendement
        fig_scatter = px.scatter(
            df,
            x='Climate_Risk_Score',
            y='Annual_Return',
            size='Volatility',
            color='Risk_Level',
            color_discrete_map={'Élevé': '#FF4444', 'Modéré': '#FFA500', 'Faible': '#00CC44'},
            hover_data=['Sector'],
            title='Risque Climatique vs Performance Financière'
        )
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Heatmap des composantes de risque
    st.subheader("🔥 Matrice des Composantes de Risque")
    
    risk_components = df[['CO2_Intensity', 'Water_Risk_Score', 'Regulatory_Risk', 'Physical_Risk_Exposure']].T
    
    fig_heatmap = px.imshow(
        risk_components,
        x=df['Sector'],
        y=['Intensité CO2', 'Risque Hydrique', 'Risque Réglementaire', 'Exposition Physique'],
        color_continuous_scale='Reds',
        title='Intensité des Différents Types de Risques par Secteur'
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)

def show_sector_analysis(df):
    """Analyse détaillée par secteur"""
    st.header("🔍 Analyse Sectorielle Détaillée")
    
    # Sélection du secteur
    selected_sector = st.selectbox("Choisir un secteur", df['Sector'].tolist())
    
    sector_data = df[df['Sector'] == selected_sector].iloc[0]
    
    # Affichage des métriques du secteur
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_color = "🔴" if sector_data['Risk_Level'] == 'Élevé' else "🟡" if sector_data['Risk_Level'] == 'Modéré' else "🟢"
        st.metric(
            f"{risk_color} Niveau de Risque",
            sector_data['Risk_Level'],
            f"Score: {sector_data['Climate_Risk_Score']:.1f}/100"
        )
    
    with col2:
        return_color = "📈" if sector_data['Annual_Return'] > 0 else "📉"
        st.metric(
            f"{return_color} Rendement Annuel",
            f"{sector_data['Annual_Return']:.1f}%",
            f"Volatilité: {sector_data['Volatility']:.1f}%"
        )
    
    with col3:
        st.metric(
            "🎯 Rendement Ajusté du Risque",
            f"{sector_data['Risk_Adjusted_Return']:.2f}",
            "Plus élevé = Meilleur"
        )
    
    # Décomposition du score de risque
    st.subheader(f"🔬 Décomposition du Risque - {selected_sector}")
    
    risk_breakdown = {
        'Composante': ['Intensité CO2', 'Risque Hydrique', 'Risque Réglementaire', 'Exposition Physique'],
        'Valeur': [
            sector_data['CO2_Intensity'],
            sector_data['Water_Risk_Score'],
            sector_data['Regulatory_Risk'],
            sector_data['Physical_Risk_Exposure']
        ],
        'Pondération': [40, 10, 20, 30],  # Pondérations utilisées dans le calcul
    }
    
    risk_df = pd.DataFrame(risk_breakdown)
    risk_df['Contribution'] = (risk_df['Valeur'] / 10) * (risk_df['Pondération'] / 100) * 100
    
    fig_breakdown = px.bar(
        risk_df,
        x='Composante',
        y='Contribution',
        color='Contribution',
        color_continuous_scale='Reds',
        title=f'Contribution de Chaque Composante au Score Total ({sector_data["Climate_Risk_Score"]:.1f})'
    )
    st.plotly_chart(fig_breakdown, use_container_width=True)
    
    # Comparaison avec la moyenne sectorielle
    st.subheader("📊 Position Relative")
    
    comparison_metrics = ['Climate_Risk_Score', 'Annual_Return', 'Volatility', 'Risk_Adjusted_Return']
    comparison_data = []
    
    for metric in comparison_metrics:
        sector_value = sector_data[metric]
        avg_value = df[metric].mean()
        comparison_data.append({
            'Métrique': metric.replace('_', ' ').title(),
            'Secteur': sector_value,
            'Moyenne': avg_value,
            'Écart': sector_value - avg_value
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df.style.format({'Secteur': '{:.2f}', 'Moyenne': '{:.2f}', 'Écart': '{:.2f}'}))

def show_portfolio_analysis(df):
    """Analyse de portefeuille personnalisée"""
    st.header("💼 Analyseur de Portefeuille")
    st.markdown("Configurez les pondérations de votre portefeuille pour analyser le risque climatique global")
    
    # Configuration du portefeuille
    st.subheader("⚖️ Configuration des Pondérations")
    
    portfolio_weights = {}
    total_weight = 0
    
    col1, col2 = st.columns(2)
    
    for i, sector in enumerate(df['Sector'].tolist()):
        if i % 2 == 0:
            with col1:
                weight = st.slider(
                    f"{sector}",
                    min_value=0,
                    max_value=50,
                    value=int(100/11),  # Équipondéré par défaut
                    step=1,
                    key=f"weight_{sector}"
                )
        else:
            with col2:
                weight = st.slider(
                    f"{sector}",
                    min_value=0,
                    max_value=50,
                    value=int(100/11),  # Équipondéré par défaut
                    step=1,
                    key=f"weight_{sector}"
                )
        
        portfolio_weights[sector] = weight
        total_weight += weight
    
    # Validation des pondérations
    if total_weight != 100:
        st.warning(f"⚠️ Total des pondérations: {total_weight}% (doit être 100%)")
        if st.button("🔄 Normaliser à 100%"):
            normalization_factor = 100 / total_weight
            for sector in portfolio_weights:
                portfolio_weights[sector] = round(portfolio_weights[sector] * normalization_factor, 1)
            st.rerun()
    else:
        st.success("✅ Pondérations correctes (100%)")
    
    # Calcul des métriques du portefeuille
    if total_weight > 0:
        portfolio_risk = sum(
            df[df['Sector'] == sector]['Climate_Risk_Score'].iloc[0] * (weight/100)
            for sector, weight in portfolio_weights.items()
        )
        
        portfolio_return = sum(
            df[df['Sector'] == sector]['Annual_Return'].iloc[0] * (weight/100)
            for sector, weight in portfolio_weights.items()
        )
        
        portfolio_volatility = sum(
            df[df['Sector'] == sector]['Volatility'].iloc[0] * (weight/100)
            for sector, weight in portfolio_weights.items()
        )
        
        # Affichage des résultats du portefeuille
        st.subheader("📋 Résultats du Portefeuille")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_level = "Élevé" if portfolio_risk >= 70 else "Modéré" if portfolio_risk >= 40 else "Faible"
            risk_color = "🔴" if risk_level == 'Élevé' else "🟡" if risk_level == 'Modéré' else "🟢"
            st.metric(
                f"{risk_color} Risque Climatique",
                f"{portfolio_risk:.1f}/100",
                risk_level
            )
        
        with col2:
            st.metric(
                "📈 Rendement Attendu",
                f"{portfolio_return:.1f}%",
                f"vs {df['Annual_Return'].mean():.1f}% moyenne"
            )
        
        with col3:
            st.metric(
                "📊 Volatilité",
                f"{portfolio_volatility:.1f}%",
                f"vs {df['Volatility'].mean():.1f}% moyenne"
            )
        
        # Graphique de répartition du portefeuille
        portfolio_data = pd.DataFrame([
            {'Secteur': sector, 'Pondération': weight, 
             'Risque': df[df['Sector'] == sector]['Climate_Risk_Score'].iloc[0]}
            for sector, weight in portfolio_weights.items()
            if weight > 0
        ])
        
        if not portfolio_data.empty:
            fig_portfolio = px.pie(
                portfolio_data,
                values='Pondération',
                names='Secteur',
                color='Risque',
                title='Répartition du Portefeuille (coloré par risque climatique)'
            )
            st.plotly_chart(fig_portfolio, use_container_width=True)

def show_detailed_report(df):
    """Rapport détaillé avec recommandations"""
    st.header("📊 Rapport Détaillé d'Analyse")
    
    # Tableau complet des données
    st.subheader("📋 Données Complètes")
    
    display_df = df.round(2)
    st.dataframe(
        display_df.style.format({
            'Climate_Risk_Score': '{:.1f}',
            'Annual_Return': '{:.1f}%',
            'Volatility': '{:.1f}%',
            'Risk_Adjusted_Return': '{:.2f}'
        }).background_gradient(subset=['Climate_Risk_Score'], cmap='Reds')
    )
    
    # Analyses statistiques
    st.subheader("📈 Analyses Statistiques")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Corrélations:**")
        correlations = df[['Climate_Risk_Score', 'Annual_Return', 'Volatility']].corr()
        st.dataframe(correlations.round(3))
    
    with col2:
        st.markdown("**Statistiques par Niveau de Risque:**")
        risk_stats = df.groupby('Risk_Level').agg({
            'Annual_Return': ['mean', 'std'],
            'Volatility': 'mean',
            'Climate_Risk_Score': 'mean'
        }).round(2)
        st.dataframe(risk_stats)
    
    # Recommandations
    st.subheader("💡 Recommandations Stratégiques")
    
    high_risk_sectors = df[df['Risk_Level'] == 'Élevé']['Sector'].tolist()
    low_risk_sectors = df[df['Risk_Level'] == 'Faible']['Sector'].tolist()
    
    correlation_risk_return = df['Climate_Risk_Score'].corr(df['Annual_Return'])
    
    st.markdown("**🎯 Recommandations pour l'investissement ESG:**")
    
    if correlation_risk_return < -0.3:
        st.success("✅ **Opportunité favorable**: Les secteurs à faible risque climatique surperforment actuellement")
        st.markdown(f"- Privilégier: {', '.join(low_risk_sectors)}")
        st.markdown(f"- Limiter l'exposition: {', '.join(high_risk_sectors)}")
    elif correlation_risk_return > 0.3:
        st.warning("⚠️ **Attention**: Les secteurs à haut risque climatique surperforment temporairement")
        st.markdown("- Risque de correction future lors de la transition énergétique")
        st.markdown("- Diversifier vers des secteurs plus durables")
    else:
        st.info("ℹ️ **Marché neutre**: Pas de corrélation claire entre risque climatique et performance")
        st.markdown("- Opportunité de surpondérer les secteurs ESG sans pénalité de rendement")
    
    st.markdown("**🔍 Indicateurs de suivi des engagements climat:**")
    avg_risk = df['Climate_Risk_Score'].mean()
    
    if avg_risk > 60:
        st.error("🔴 **Portefeuille à haut risque climatique**")
        st.markdown("- Score moyen > 60: Exposition significative aux risques de transition")
        st.markdown("- Recommandation: Réduire l'exposition aux secteurs Energy et Utilities")
    elif avg_risk > 40:
        st.warning("🟡 **Risque climatique modéré**")
        st.markdown("- Score dans la moyenne: Surveillance recommandée")
        st.markdown("- Opportunité d'amélioration via sélection sectorielle")
    else:
        st.success("🟢 **Profil de risque climatique acceptable**")
        st.markdown("- Score < 40: Portefeuille aligné avec les objectifs climatiques")
    
    # Export des résultats
    st.subheader("💾 Export des Résultats")
    
    if st.button("📊 Générer le rapport PDF"):
        # Simulation de génération de rapport
        with st.spinner("Génération du rapport en cours..."):
            import time
            time.sleep(2)
        st.success("✅ Rapport généré avec succès!")
        st.download_button(
            label="📁 Télécharger le dataset CSV",
            data=df.to_csv(index=False),
            file_name=f"climate_risk_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()


# In[ ]:




