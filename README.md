# Analyse des Risques Climatiques

> Pipeline d'évaluation ESG avec **score composite climatique** et dashboard Streamlit interactif pour l'analyse de risques sectoriels

## Démo Live

**Dashboard Interactif :** [Climate Risk Dashboard](https://analyse-des-risques-climatiques.streamlit.app/)

## Résultats Clés

| Secteur | Score Risque | Rendement | Niveau |
|---------|-------------|-----------|---------|
| **Energy** | **84.3/100** | 1.8% | 🔴 **Élevé** |
| **Utilities** | **70.9/100** | 8.1% | 🔴 **Élevé** |
| **Materials** | 63.9/100 | -2.4% | 🟡 Modéré |
| **Health Care** | **19.1/100** | -9.9% | 🟢 **Faible** |
| **Information Technology** | **19.5/100** | 26.2% | 🟢 **Faible** |

**Corrélation Risque-Rendement :** -0.350 (corrélation négative modérée)

**Distribution des Risques :**
- 🟢 **Faible** : 6 secteurs (55%)
- 🟡 **Modéré** : 3 secteurs (27%)
- 🔴 **Élevé** : 2 secteurs (18%)

## Stack Technique & Compétences

**Data Engineering & Analysis :**
- `pandas` / `numpy` (Manipulation données financières)
- `yfinance` (Récupération données temps réel ETF sectoriels)
- `matplotlib` / `seaborn` / `plotly` (Visualisations avancées)
- Feature Engineering Multi-composantes (CO2, Eau, Réglementaire, Physique)

**Pipeline & Architecture :**
- Architecture Orientée Objet (ClimateRiskPipeline)
- Modularité & Réutilisabilité (fonctions séparées)
- Cache Streamlit (optimisation performance)
- Gestion d'erreurs robuste

**Déploiement & Production :**
- `streamlit` (Dashboard interactif multi-vues)
- Interface utilisateur responsive
- Déploiement Cloud ready

## Fonctionnalités Clés

- **Score Composite de Risque Climatique** : Algorithme pondéré (CO2: 40%, Physique: 30%, Réglementaire: 20%, Eau: 10%)
- **Données Financières Temps Réel** : Intégration yfinance pour 11 ETF sectoriels
- **Analyse de Corrélations** : Relation risque climatique vs performance financière
- **Classification Automatique** : Niveaux de risque (Faible/Modéré/Élevé)
- **Analyseur de Portefeuille** : Simulation pondérations personnalisées
- **Visualisations Interactives** : Heatmaps, scatter plots, graphiques sectoriels
- **Rapports ESG** : Recommandations stratégiques d'investissement

## Architecture du Projet

```
Risques_Climatiques/
├── .git/                # Version control
├── .gitignore          # Git ignore rules
├── data/
│   └── climate_risk_analysis.csv  # Dataset final exporté
├── notebooks/
│   ├── 01_cell.ipynb           # Notebook analyse exploratoire
│   ├── 02_functions.ipynb      # Fonctions modulaires
│   ├── 03_pipeline.ipynb       # Pipeline automatisé
│   └── app.ipynb              # Dashboard Streamlit
├── reports/
│   ├── 01_cell.pdf           # Export analyse exploratoire
│   ├── 02_functions.pdf      # Documentation fonctions
│   └── 03_pipeline-2.pdf     # Documentation pipeline
└── src/
    ├── app.py               # Application Streamlit principale
    └── functions.py         # Utilitaires (load, calculate, visualize)
```

## Utilisation

**Pipeline Automatisé Complet :**
```python
from pipeline import ClimateRiskPipeline

# Exécution complète
pipeline = ClimateRiskPipeline()
df_results = pipeline.run_complete_analysis(
    fetch_live_data=True,
    save_results=True
)

# Rapport de synthèse
pipeline.print_summary_report()

# Analyse de portefeuille
portfolio_risk = pipeline.calculate_portfolio_risk_score([10,10,10,10,10,10,10,10,10,10,10])
```

**Dashboard Streamlit :**
```bash
streamlit run app.py
```
