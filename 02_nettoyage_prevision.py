"""
Étape 2 : nettoyer les données, puis comparer des modèles de prévision du taux EUR/TND.

Questions auxquelles ce script répond :
- Y a-t-il des valeurs aberrantes (allers-retours impossibles) dans les données ?
- Un modèle ARIMA prévoit-il mieux le taux de demain que la règle la plus simple :
  « demain = aujourd'hui » (marche aléatoire) ?
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

RAW_FILE = Path("data/eur_tnd.csv")
CLEAN_FILE = Path("data/eur_tnd_clean.csv")
FIGURES = Path("figures")
TEST_DAYS = 250  # environ une année de bourse pour tester les modèles

# ----------------------------------------------------------------------------
# 1. Chargement
# ----------------------------------------------------------------------------
df = pd.read_csv(RAW_FILE)
df["Date"] = pd.to_datetime(df["Date"])
rates = df.set_index("Date").iloc[:, 0].astype(float).sort_index()
rates = rates[~rates.index.duplicated()].dropna()
rates = rates[rates > 0]

# ----------------------------------------------------------------------------
# 2. Contrôle qualité : repérer les allers-retours aberrants
#    Un jour suspect = grosse hausse (ou baisse) immédiatement annulée le lendemain.
# ----------------------------------------------------------------------------
returns = np.log(rates).diff()
typical_move = returns.abs().rolling(60, min_periods=20).median().shift(1)
big_move = returns.abs() > 8 * typical_move
reversal = (np.sign(returns) != np.sign(returns.shift(-1))) & (
    returns.shift(-1).abs() > 0.5 * returns.abs()
)
suspects = rates.index[big_move & reversal]

print(f"Jours suspects détectés : {len(suspects)}")
for day in suspects:
    print(f"  {day:%d/%m/%Y} : {rates[day]:.4f} TND "
          f"(variation {returns[day]:+.2%}, puis {returns.shift(-1)[day]:+.2%} le lendemain)")

clean = rates.copy()
clean[suspects] = np.nan
clean = clean.interpolate(method="time")  # on remplace par la moyenne des jours voisins
clean.rename("Close").to_csv(CLEAN_FILE, index_label="Date")

clean_returns = np.log(clean).diff().dropna()
print(f"\nVolatilité annualisée avant nettoyage : {returns.std() * np.sqrt(252):.2%}")
print(f"Volatilité annualisée après nettoyage : {clean_returns.std() * np.sqrt(252):.2%}")
print(f"Kurtosis après nettoyage : {clean_returns.kurtosis():.2f}")

# ----------------------------------------------------------------------------
# 3. Séparation temporelle : on apprend sur le passé, on teste sur la dernière année
# ----------------------------------------------------------------------------
test_start = clean.index[-TEST_DAYS]
actual = clean[test_start:]
print(f"\nPériode de test : {test_start:%d/%m/%Y} -> {clean.index[-1]:%d/%m/%Y} ({TEST_DAYS} jours)")

predictions = pd.DataFrame(index=actual.index)

# Modèle 1 : marche aléatoire -> le taux de demain = le taux d'aujourd'hui
predictions["Marche aléatoire"] = clean.shift(1)[test_start:]

# Modèle 2 : ARIMA(1,1,1) sur le log du taux.
# Paramètres estimés sur la période d'entraînement uniquement, puis prévision
# jour par jour (à chaque date, le modèle ne connaît que les jours précédents).
log_rates = np.log(clean)
log_rates.index = pd.RangeIndex(len(log_rates))
train_end = len(clean) - TEST_DAYS
arima_fit = ARIMA(log_rates[:train_end], order=(1, 1, 1)).fit()
arima_all = arima_fit.apply(log_rates)
one_step = arima_all.get_prediction(start=train_end).predicted_mean
predictions["ARIMA(1,1,1)"] = np.exp(one_step.values)
print(f"\nARIMA(1,1,1) - coefficients : AR = {arima_fit.params['ar.L1']:.3f}, "
      f"MA = {arima_fit.params['ma.L1']:.3f}")

# ----------------------------------------------------------------------------
# 4. Comparaison
# ----------------------------------------------------------------------------
previous = clean.shift(1)[test_start:]
results = []
for model in predictions.columns:
    error = predictions[model] - actual
    row = {
        "Modèle": model,
        "RMSE (TND)": np.sqrt((error ** 2).mean()),
        "MAE (TND)": error.abs().mean(),
    }
    if model != "Marche aléatoire":
        right_direction = np.sign(predictions[model] - previous) == np.sign(actual - previous)
        row["Bon sens de variation"] = right_direction.mean()
    results.append(row)

results = pd.DataFrame(results).set_index("Modèle")
results["vs marche aléatoire"] = results["RMSE (TND)"] / results.loc["Marche aléatoire", "RMSE (TND)"] - 1
print("\n" + results.to_string(float_format=lambda x: f"{x:.5f}"))
print("\nLecture : un écart négatif dans « vs marche aléatoire » = le modèle fait mieux que « demain = aujourd'hui ».")

predictions.assign(Réel=actual).to_csv("data/predictions.csv", index_label="Date")

# Graphique : les 90 derniers jours
last = predictions.index[-90:]
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(actual[last], color="black", lw=2, label="Taux réel")
colors = {"Marche aléatoire": "#9ca3af", "ARIMA(1,1,1)": "#2563eb"}
for model, color in colors.items():
    ax.plot(predictions.loc[last, model], color=color, lw=1, label=model)
ax.set_title("Prévision du taux EUR/TND à 1 jour : 90 derniers jours de la période de test")
ax.set_ylabel("TND pour 1 €")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES / "02_prevision.png", dpi=150)
plt.show()
