"""Train the NALIKERA yield models (run once):  python train_model.py
Ridge (with last-year yield) + XGBoost + Random Forest (with year trend), district-level, 1981-2023.
Also runs a rolling-origin backtest to get honest forecast error bands (saved to models/meta.json)."""
import json, warnings, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent; OUT = ROOT / "models"; OUT.mkdir(exist_ok=True)
T = "Productivity_Nuts_per_Ha"
WEATHER = ["Temperature_C", "Humidity_Annual_Pct", "Monsoon_Rain", "HotSeason_Temp",
           "PreMonsoon_Rain", "PostMonsoon_Rain", "Monsoon_Humidity", "ENSO_ONI"]
df = pd.read_csv(ROOT / "data" / "multi_district_master.csv").sort_values(["District", "Year"]).reset_index(drop=True)
DISTRICTS = sorted(df.District.unique())
df["Year_c"] = df.Year - 2000
TREND = ["Year_c"] + [f"Tr_{d}" for d in DISTRICTS]
for d in DISTRICTS: df[f"Tr_{d}"] = (df.District == d) * df.Year_c
WL = ["Area_Ha"] + WEATHER + ["Yield_Lag1"]; WT = ["Area_Ha"] + WEATHER + TREND

def prep(cols, scale):
    return ColumnTransformer([("n", StandardScaler() if scale else "passthrough", cols),
                              ("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["District"])])
def ridge(): return Pipeline([("p", prep(WL, True)), ("m", RidgeCV(alphas=np.logspace(-2, 3, 30)))])
def xgb(): return Pipeline([("p", prep(WT, False)), ("m", XGBRegressor(n_estimators=400, learning_rate=.03, max_depth=3,
        subsample=.8, colsample_bytree=.8, reg_lambda=5, min_child_weight=3, random_state=42))])
def rf(): return Pipeline([("p", prep(WT, False)), ("m", RandomForestRegressor(n_estimators=500, max_depth=6,
        min_samples_leaf=3, max_features=.6, random_state=42, n_jobs=-1))])

dl = df.dropna(subset=WL + [T]); dt = df.dropna(subset=WT + [T])
models = {"ridge": ridge().fit(dl[WL + ["District"]], dl[T]),
          "xgb": xgb().fit(dt[WT + ["District"]], dt[T]), "rf": rf().fit(dt[WT + ["District"]], dt[T])}
for k, m in models.items(): joblib.dump(m, OUT / f"{k}.joblib")

# rolling-origin backtest: forecast 1..5 years ahead with average weather, using only data up to the origin
res = []
for O in range(2008, 2023):
    a = dl[dl.Year <= O]; b = dt[dt.Year <= O]
    mr, mx, mf = ridge().fit(a[WL + ["District"]], a[T]), xgb().fit(b[WT + ["District"]], b[T]), rf().fit(b[WT + ["District"]], b[T])
    for d in DISTRICTS:
        g = df[df.District == d]; c = g[g.Year <= O].tail(10)[["Area_Ha"] + WEATHER].mean(); cur = g[g.Year == O][T].iloc[0]
        for h in range(1, 6):
            Y = O + h
            if Y > 2023 or g[g.Year == Y].empty: continue
            r = c.copy(); r["Yield_Lag1"] = cur; r["District"] = d
            cur = float(mr.predict(pd.DataFrame([r])[WL + ["District"]])[0])
            rt = c.copy(); rt["Year_c"] = Y - 2000
            for k in DISTRICTS: rt[f"Tr_{k}"] = float(d == k) * (Y - 2000)
            rt["District"] = d; X = pd.DataFrame([rt])[WT + ["District"]]
            p = (cur + float(mx.predict(X)[0]) + float(mf.predict(X)[0])) / 3
            res.append((d, h, p - g[g.Year == Y][T].iloc[0]))
R = pd.DataFrame(res, columns=["d", "h", "e"]); rm = lambda s: float(np.sqrt((s ** 2).mean()))
rmse_all = rm(R.e)
meta = {"districts": DISTRICTS, "WL": WL, "WT": WT, "weather": WEATHER,
        "rmse": {d: round(rm(g.e)) for d, g in R.groupby("d")},
        "fh": {int(h): round(rm(g.e) / rmse_all, 3) for h, g in R.groupby("h")},
        "clim": {d: df[df.District == d].tail(10)[["Area_Ha"] + WEATHER].mean().to_dict() for d in DISTRICTS},
        "sd": {d: df[df.District == d][WEATHER].std().to_dict() for d in DISTRICTS},
        "rng": {d: {c: [float(df[df.District == d][c].min()), float(df[df.District == d][c].max())] for c in WEATHER} for d in DISTRICTS},
        "rain_ref": {d: [float((g.Precipitation_mm_day * 365).mean()), float((g.Precipitation_mm_day * 365).std())]
                     for d, g in df.groupby("District")},
        "last_year": int(df.Year.max()),
        "last_yield": {d: float(df[(df.District == d) & (df.Year == df.Year.max())][T].iloc[0]) for d in DISTRICTS},
        "area_ha": {d: float(df[(df.District == d) & (df.Year == df.Year.max())].Area_Ha.iloc[0]) for d in DISTRICTS},
        "hist": {d: df[df.District == d][["Year", T]].astype(int).values.tolist() for d in DISTRICTS},
        "avg": {d: float(df[df.District == d][T].mean()) for d in DISTRICTS}}
json.dump(meta, open(OUT / "meta.json", "w"))
print("Saved models to", OUT, "| backtest RMSE by district:", meta["rmse"])
