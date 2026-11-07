"""NALIKERA engine: (1) yield prediction from the trained district-level models,
(2) irrigation advisor, (3) recommendation rules, (4) indicative farm health score.
Only (1) is machine-learned. (2)-(4) are rule-based, from KAU / CPCRI / Coconut Development Board / TNAU guidance."""
import json, joblib, numpy as np, pandas as pd
from pathlib import Path
M = Path(__file__).resolve().parent.parent / "models"
if not (M / "meta.json").exists():
    raise RuntimeError("Models not found. Run:  python train_model.py   (inside the backend folder)")
meta = json.load(open(M / "meta.json")); mod = {k: joblib.load(M / f"{k}.joblib") for k in ("ridge", "xgb", "rf")}
WL, WT, DIST = meta["WL"], meta["WT"], meta["districts"]
RAIN = ["Monsoon_Rain", "PreMonsoon_Rain", "PostMonsoon_Rain"]; HUM = ["Humidity_Annual_Pct", "Monsoon_Humidity"]
PALMS_PER_HA = 175          # assumption (7.5 m spacing); shown to the user
FEEL = {"very_hot": 1.5, "mod_hot": .5, "pleasant": 0, "cold": 0}
RAINZ = {"very_low": -2, "low": -1, "moderate": 0, "heavy": 1, "very_heavy": 2}
PREV = {"lt5": 3, "5_20": 12, "20_40": 30, "gt40": 50}      # nuts per tree per year, category midpoints

def _wx(d, hot_z=0., rain_z=0., hum_z=0., hot_c=None, hum_pct=None, rain_mm=None, warn=None):
    c, s, r = meta["clim"][d], meta["sd"][d], meta["rng"][d]; w = dict(c)
    if rain_mm is not None:
        mu, sd = meta["rain_ref"][d]; rain_z = hum_z = (rain_mm - mu) / sd
    if hum_pct is not None: hum_z = (hum_pct - c["Humidity_Annual_Pct"]) / s["Humidity_Annual_Pct"]
    if hot_c is not None: hot_z = (hot_c - c["HotSeason_Temp"]) / s["HotSeason_Temp"]
    for f, z in [(RAIN, rain_z), (HUM, hum_z), (["HotSeason_Temp"], hot_z)]:
        for k in f:
            v = c[k] + z * s[k]; lo, hi = r[k]; w[k] = min(max(v, lo), hi)
            if warn is not None and abs(v - w[k]) > 1e-9 and "outside" not in " ".join(warn):
                warn.append("Some inputs are outside the range seen in 1981-2023 data, so they were capped. Treat the result with extra caution.")
    return w

def _ridge(d, w, lag):
    r = dict(w); r["Yield_Lag1"] = lag; r["District"] = d
    return float(mod["ridge"].predict(pd.DataFrame([r])[WL + ["District"]])[0])

def _tree(d, w, Y):
    r = dict(w); r["Year_c"] = Y - 2000
    for k in DIST: r[f"Tr_{k}"] = float(k == d) * (Y - 2000)
    r["District"] = d; X = pd.DataFrame([r])[WT + ["District"]]
    return (float(mod["xgb"].predict(X)[0]) + float(mod["rf"].predict(X)[0])) / 2

def _forecast(d, w, Y):
    lag = meta["last_yield"][d]
    for _ in range(Y - meta["last_year"]): lag = _ridge(d, w, lag)
    return (lag + 2 * _tree(d, w, Y)) / 3        # = mean of Ridge, XGBoost, Random Forest

def predict(q):
    d, Y = q["district"], int(q.get("year") or 2027); warn = []
    if d not in DIST: raise ValueError("Unknown district")
    if not meta["last_year"] < Y <= meta["last_year"] + 7: raise ValueError(f"Year must be {meta['last_year']+1}-{meta['last_year']+7}")
    simple = q.get("mode") == "simple"
    rz = RAINZ.get(q.get("rain_level"), 0.); args = dict(hot_z=FEEL.get(q.get("weather_feel"), 0.), rain_z=rz, hum_z=rz, warn=warn)
    if not simple:
        args = dict(hot_c=q.get("max_temp_c"), hum_pct=q.get("humidity_pct"), rain_mm=q.get("annual_rain_mm"), warn=warn)
    w = _wx(d, **args); p = _forecast(d, w, Y); h = Y - meta["last_year"]
    hw = 1.28 * meta["rmse"][d] * meta["fh"][str(min(h, 5))] * (1.3 if simple else 1.0)
    # factor effects (Ridge, one step): input weather vs the district's average weather
    base = _ridge(d, meta["clim"][d], meta["last_yield"][d]); fac = []
    for lab, grp in [("Rainfall", RAIN), ("Humidity", HUM), ("Heat in hot season", ["HotSeason_Temp"])]:
        a = dict(meta["clim"][d]); a.update({k: w[k] for k in grp})
        fac.append({"label": lab, "effect": round(_ridge(d, a, meta["last_yield"][d]) - base)})
    out = {"mode": "trained_model", "district": d, "year": Y,
           "model": "Ensemble of Ridge, XGBoost and Random Forest trained on district yields 1981-2023 (district-level, not farm-level)",
           "productivity": {"expected": round(p), "low": round(p - hw), "high": round(p + hw), "per_tree": round(p / PALMS_PER_HA, 1), "district_avg": round(meta["avg"][d])},
           "assumed_palms_per_ha": PALMS_PER_HA, "factors": fac, "warnings": warn,
           "uncertainty_note": "Range is an 80% interval from backtesting" + (", widened 30% because weather was described in simple terms." if simple else ".") + " Actual production also depends on pests, disease, nutrition and management, which this model does not see."}
    area, trees = q.get("area_ha"), q.get("trees")
    if area: out["production"] = {"basis": f"{area} ha x district productivity", "expected": round(p * area), "low": round((p - hw) * area), "high": round((p + hw) * area)}
    elif trees: f = trees / PALMS_PER_HA; out["production"] = {"basis": f"{trees} trees at {PALMS_PER_HA} palms/ha", "expected": round(p * f), "low": round((p - hw) * f), "high": round((p + hw) * f)}
    prev = q.get("prev_per_tree") or PREV.get(q.get("prev_prod"))
    if q.get("prev_total") and (trees or area): prev = q["prev_total"] / (trees or area * PALMS_PER_HA)
    if prev:
        out["previous"] = {"per_tree": round(prev, 1)}; out["low_yield_alert"] = (p / PALMS_PER_HA) < 0.85 * prev
        if q.get("prev_prod") and not q.get("prev_total"): out["previous"]["note"] = "Based on the range you selected (midpoint)."
    ctx = _ctx(q, w, d); ctx["pred_ratio"] = p / meta["avg"][d]
    out["irrigation"] = irrigation(ctx); out["recommendations"] = recs(ctx); out["health"] = health(ctx)
    return out

def _ctx(q, w, d):
    s = q.get("mode") == "simple"; z = lambda k: (w[k] - meta["clim"][d][k]) / meta["sd"][d][k]
    return {"soil": q.get("soil_state") or "unsure", "rain_z": z("Monsoon_Rain") if not s else RAINZ.get(q.get("rain_level"), 0),
            "hot_z": z("HotSeason_Temp"), "irr": q.get("irrigation") or "unknown", "age": q.get("tree_age") or "unsure",
            "tree": q.get("tree_state") or "unknown", "soil_type": q.get("soil_type"), "method": q.get("irrigation_method"),
            "ph": q.get("soil_ph"), "low": False, "pred_ratio": 1}

SOIL_BASIN = {"sandy": "600 L every 3-4 days", "sandy_loam": "900 L every 5 days", "loam": "1300 L every 7-8 days", "silty_clay": "1600 L every 9 days"}
def irrigation(c):
    soil, wet = c["soil"], c["rain_z"] >= 1 or c["soil"] in ("wet", "waterlogged"); dry = c["soil"] == "dry"; hot = c["hot_z"] >= .5
    adult = c["age"] not in ("lt5",); L = []
    if c["soil"] == "waterlogged" or (c["rain_z"] >= 2 and c["soil"] in ("wet", "unsure")):
        st, head = "blue", "Do NOT irrigate. Drain the garden."
        L = ["Open drains and remove standing water; waterlogged roots cannot breathe.", "Hold fertiliser until the soil drains.", "Watch the crown for bud rot signs after prolonged rain."]
    elif dry and (c["rain_z"] <= -1 or hot or c["irr"] == "none"):
        st, head = "red", "Water stress may be occurring. Irrigate."
    elif soil in ("dry", "slight", "unsure") and (c["rain_z"] <= -1 or hot) or dry:
        st, head = "yellow", "Monitor moisture. Irrigate if the soil is dry about 10 cm down."
    else: st, head = "green", "Soil moisture appears sufficient."
    if st in ("red", "yellow"):
        if not adult: L.append("Young palms (under 5 years) need smaller, more frequent watering. Follow Krishi Bhavan / KAU advice for the exact amount.")
        else:
            L.append("Adult palm by drip: 40-50 L per palm per day in the dry season (KAU).")
            L.append("Basin irrigation (1.8 m radius basin): " + (SOIL_BASIN.get(c["soil_type"]) + f" for {c['soil_type'].replace('_',' ')} soil (KAU)." if c["soil_type"] in SOIL_BASIN else "600 L every 3-4 days (sandy) to 1600 L every 9 days (silty clay) (KAU)."))
            L.append("Water-saving option: CPCRI trials at Kasaragod found 27-32 L per palm per day by drip gave similar yield.")
        L.append("Mulch the basin (about 100 husks, 15 dried leaves or 10 cm coir pith) to hold moisture.")
    if st == "green": L = ["No extra irrigation needed now. Check again in dry spells.", "Keep drains open before heavy rain."]
    return {"status": st, "headline": head, "advice": L, "assumptions": "Rule-based from KAU / CPCRI / TNAU guidance for adult bearing palms. Not measured soil data; check actual moisture in the field."}

R = lambda cat, ml, t, p, y, do, wh, b: dict(category=cat, ml_title=ml, title=t, problem=p, why=y, do=do, when=wh, benefit=b)
def recs(c):
    o = []
    if c["rain_z"] >= 1 or c["soil"] in ("wet", "waterlogged"):
        o.append(R("Water", "നീർവാർച്ച പരിശോധിക്കുക", "Check drainage", "Heavy rain or wet soil in the garden.", "Standing water damages roots and raises bud rot risk.", "Open drains, remove standing water, skip fertiliser until the soil drains.", "Before and during the monsoon.", "Healthier roots and lower disease risk (no yield guarantee)."))
    if c["hot_z"] >= .5 or c["rain_z"] <= -1 or c["soil"] == "dry":
        o.append(R("Water", "പുതയിടുക, ഈർപ്പം സംരക്ഷിക്കുക", "Mulch and conserve soil moisture", "Hot or dry conditions dry out the basin.", "Palms lose yield potential when water is short in the dry months.", "Mulch the basin (about 100 husks, 15 dried leaves or 10 cm coir pith). Bury husks in trenches between palms.", "Before summer (Jan-Feb).", "Soil stays moist longer, so less irrigation is needed."))
    if c["irr"] in ("none", "summer"):
        o.append(R("Water", "വേനലിൽ നനയ്ക്കാൻ ആസൂത്രണം ചെയ്യുക", "Plan irrigation for the dry months", "Little or no irrigation outside rains.", "Dry-season water supports flowering and nut set.", "Irrigate adult palms through Dec-May (40-50 L/palm/day by drip, or basin amounts by soil type).", "December to May.", "More stable production in dry years."))
    o.append(R("Soil", "മണ്ണ് പരിശോധിപ്പിക്കുക", "Test your soil", "Nutrient needs differ from garden to garden.", "Guessing doses wastes money or leaves nutrients short.", "Get a soil test (Krishi Bhavan / KAU lab). If acidic, apply about 1 kg dolomite or lime per palm 15 days before fertiliser.", "Once a year, before the main fertiliser dose.", "Fertiliser matched to what the soil lacks."))
    o.append(R("Soil", "വളം രണ്ടു തവണയായി നൽകുക", "Follow a split fertiliser schedule", "Nutrients applied irregularly.", "Palms need steady nitrogen, phosphorus and potash.", "Good-management dose is 500 g N, 320 g P2O5, 1200 g K2O per palm per year (about 1.1 kg urea, 1.1 kg rock phosphate, 2 kg potash), in 2 doses.", "1/3 in April-June, 2/3 in September-October.", "Consistent nutrition supports nut development."))
    o.append(R("Soil", "ജൈവവളം ചേർക്കുക", "Add organic manure", "Organic matter keeps soil healthy.", "It holds water and feeds soil life.", "Apply 20-50 kg farmyard manure or compost per palm, or grow green manure.", "At monsoon onset (June-July).", "Better soil structure and moisture holding."))
    if c["tree"] in ("yellow", "weak", "drying"):
        o.append(R("Tree", "ഓലകൾ മഞ്ഞളിക്കുന്നത് ശ്രദ്ധിക്കുക", "Check yellowing or drying leaves", "Leaves are yellowing, drying or palms look weak.", "Causes include nutrient deficiency, water stress or disease such as root wilt.", "Do a soil test, consider 0.5 kg magnesium sulphate per palm, and show the palm to Krishi Bhavan if symptoms are spreading.", "Now, and before the next fertiliser dose.", "Correct the cause early; this site cannot diagnose it."))
    if c["tree"] == "fewer" or c.get("low"):
        o.append(R("Productivity", "വിളവ് കുറഞ്ഞ തെങ്ങുകൾ പരിശോധിക്കുക", "Review low-yielding trees", "Fewer nuts than expected.", "A few poor palms can pull down the whole garden.", "Tag palms and count nuts per palm for a year; look for water, nutrient, pest or age problems.", "Every harvest.", "You learn which palms need care or replacement."))
    if c["age"] == "gt40":
        o.append(R("Productivity", "പഴയ തെങ്ങുകളുടെ പുനർനടീൽ പരിഗണിക്കുക", "Plan phased replanting", "Palms over 40 years old.", "Yield generally declines in very old palms.", "Plant new seedlings gradually between old palms; ask Krishi Bhavan or the Coconut Development Board about varieties.", "Start before the monsoon.", "A steady income as old palms retire."))
    o.append(R("Climate", "മഴയ്ക്ക് മുൻപ് മണ്ട വൃത്തിയാക്കുക", "Crown care before the monsoon", "Rain and humidity favour crown diseases such as bud rot.", "Bud rot can kill a palm if missed.", "Clean the crown and spray 1% Bordeaux mixture; repeat after about 45 days (CPCRI).", "Late May, before the monsoon.", "Lower chance of crown rot."))
    return o

def health(c):
    cl = lambda v: int(max(10, min(95, v)))
    water = {"dry": 35, "slight": 75, "wet": 80, "waterlogged": 25}.get(c["soil"], 60) - (15 if c["irr"] == "none" and c["rain_z"] <= -1 else 0)
    ph = c["ph"]; soil = 60 if ph is None else (85 if 5.5 <= ph <= 7.5 else 55)
    tree = {"healthy": 90, "yellow": 50, "drying": 40, "fewer": 55, "weak": 35}.get(c["tree"], 60) - (10 if c["age"] == "gt40" else 0)
    clim = 80 - 7.5 * abs(c["rain_z"]) - 10 * max(c["hot_z"], 0)
    prod = 50 + (c["pred_ratio"] - 1) * 100
    s = {"Water management": cl(water), "Soil health": cl(soil), "Tree health": cl(tree), "Climate suitability": cl(clim), "Productivity": cl(prod)}
    return {"scores": s, "overall": round(sum(s.values()) / 5), "how": "Simple 0-100 scores from your answers; unknown answers count as 60. Indicative only, not a validated scientific index."}

def history(d): return {"district": d, "history": meta["hist"][d]}
