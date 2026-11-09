"""
Étape 1 : récupérer l'historique EUR/TND et explorer les données.

Questions auxquelles ce script répond :
- Comment l'euro a-t-il évolué face au dinar ?
- De combien le taux bouge-t-il en moyenne chaque jour (volatilité) ?
- Les gros chocs sont-ils plus fréquents que ne le prévoit la loi normale ?
- La série est-elle stationnaire (condition pour utiliser ARIMA) ?
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

DATA_FILE = Path("data/eur_tnd.csv")
START_DATE = "2015-01-01"
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_dates(col: pd.Series) -> pd.Series:
    """Lit les dates au format AAAA-MM-JJ, jour/mois (site français) ou mois/jour (site anglais)."""
    iso = pd.to_datetime(col, format="ISO8601", errors="coerce")
    if iso.notna().all():
        return iso
    best_dates, best_order = None, -1.0
    for dayfirst in (True, False):
        dates = pd.to_datetime(col, dayfirst=dayfirst, format="mixed", errors="coerce")
        steps = dates.diff().dropna().dt.days
        order = max((steps > 0).mean(), (steps < 0).mean())  # le fichier est trié : dates dans le bon ordre
        if order > best_order:
            best_dates, best_order = dates, order
    return best_dates


def load_rates() -> pd.Series:
    """Charge le taux EUR/TND : fichier local s'il existe, sinon Yahoo Finance."""
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
        date_col = [c for c in df.columns if c.lower() in ("date", "dates")][0]
        price_col = [c for c in df.columns if c.lower() in ("price", "close", "dernier", "cours", "rate")][0]
        df[date_col] = parse_dates(df[date_col])
        if not pd.api.types.is_numeric_dtype(df[price_col]):  # format "3,4123" -> 3.4123
            df[price_col] = pd.to_numeric(df[price_col].astype(str).str.replace(",", "."), errors="coerce")
        rates = df.set_index(date_col)[price_col].sort_index()
    else:
        import yfinance as yf
        data = yf.download("EURTND=X", start=START_DATE, progress=False, auto_adjust=False)
        if data.empty:
            raise SystemExit("Yahoo Finance n'a renvoyé aucune donnée : utilise un fichier CSV (voir README).")
        rates = data["Close"].squeeze()
        DATA_FILE.parent.mkdir(exist_ok=True)
        rates.rename("Close").to_csv(DATA_FILE, index_label="Date")
    rates = rates[~rates.index.duplicated()].dropna()
    rates = rates[rates > 0]
    rates.name = "EUR/TND"
    return rates


rates = load_rates()

# Rendements logarithmiques quotidiens : la variation en % d'un jour à l'autre
returns = np.log(rates).diff().dropna()

print(f"Période : {rates.index.min():%d/%m/%Y} -> {rates.index.max():%d/%m/%Y} ({len(rates)} jours)")
print(f"Premier taux : {rates.iloc[0]:.4f} | Dernier taux : {rates.iloc[-1]:.4f}")
print(f"Variation totale de l'euro : {rates.iloc[-1] / rates.iloc[0] - 1:+.1%}")
print()
print(f"Tendance moyenne par jour : {returns.mean():+.4%}")
print(f"Volatilité par jour : {returns.std():.3%}")
print(f"Volatilité annualisée : {returns.std() * np.sqrt(252):.2%}")
print(f"Pire jour : {returns.min():+.2%} | Meilleur jour : {returns.max():+.2%}")

# Queues épaisses : kurtosis > 0 = plus de gros chocs que la loi normale
print(f"Kurtosis (excès) : {returns.kurtosis():.2f}  (0 pour une loi normale)")

# Test ADF : p-value < 0,05 => série stationnaire
for name, serie in [("Taux (niveau)", rates), ("Rendements", returns)]:
    p_value = adfuller(serie)[1]
    verdict = "stationnaire" if p_value < 0.05 else "NON stationnaire"
    print(f"Test ADF - {name} : p-value = {p_value:.4f} -> {verdict}")

# Graphiques
fig, axes = plt.subplots(3, 1, figsize=(11, 10))
rates.plot(ax=axes[0], color="#1d4ed8", title="Taux EUR/TND (TND pour 1 €)")
returns.plot(ax=axes[1], color="#64748b", lw=0.6, title="Rendements quotidiens : périodes calmes et agitées")
(returns.rolling(60).std() * np.sqrt(252)).plot(ax=axes[2], color="#dc2626",
                                               title="Volatilité annualisée sur 60 jours glissants")
for ax in axes:
    ax.set_xlabel("")
plt.tight_layout()
plt.savefig(FIGURES / "01_exploration.png", dpi=150)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(returns, bins=80, density=True, color="#93c5fd", label="Rendements observés")
x = np.linspace(returns.min(), returns.max(), 300)
normal = np.exp(-0.5 * ((x - returns.mean()) / returns.std()) ** 2) / (returns.std() * np.sqrt(2 * np.pi))
ax.plot(x, normal, color="#dc2626", label="Loi normale")
ax.set_title("Distribution des rendements vs loi normale")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES / "01_distribution.png", dpi=150)
plt.show()
