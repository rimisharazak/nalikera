# NALIKERA – Smart Coconut Farming and Yield Prediction (Kerala)

Districts: Thrissur, Ernakulam, Palakkad, Malappuram.

## Structure
```
nalikera/
├── requirements.txt
├── README.md
├── frontend/
│   ├── index.html          # single-page app (landing, mode choice, simple + advanced input, dashboard)
│   └── hero.jpg            # OPTIONAL: add your own licensed Kerala plantation photo (gradient is used if missing)
└── backend/
    ├── train_model.py      # trains the models + backtest error bands -> backend/models/
    ├── data/multi_district_master.csv
    └── app/
        ├── main.py         # FastAPI routes (also serves the frontend)
        └── engine.py       # prediction + irrigation advisor + recommendations + health score
```

## Architecture
Browser (frontend/index.html) -> FastAPI (`/api/predict`, `/api/history/{district}`, `/api/health`) -> engine.py
- **Yield prediction (machine learning):** ensemble of Ridge, XGBoost and Random Forest, trained on district yields 1981-2023. It predicts *district-level productivity (nuts/ha)*. Farm totals are productivity x your area (or trees / 175 palms per ha).
- **Irrigation advisor, recommendations, health score (rule-based):** from KAU / CPCRI / Coconut Development Board / TNAU guidance for adult palms. They are NOT learned from the dataset (it has no irrigation or fertiliser data).
- Simple-mode answers become approximate weather shifts (e.g. "very hot" = +1.5 SD hot-season temperature), so the range is widened 30%.
- Profile and history are saved in the browser only (localStorage).

## Run locally (Windows PowerShell)
```
cd nalikera
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cd backend
python train_model.py
uvicorn app.main:app --reload
```
Open http://localhost:8000  (API docs: http://localhost:8000/docs)

## API
`POST /api/predict` JSON (only `district` required)
- common: `mode` ("simple"|"advanced"), `district`, `year` (2024-2030), `trees`, `area_ha`, `prev_total`
- simple: `weather_feel` (very_hot|mod_hot|pleasant|cold), `rain_level` (very_low|low|moderate|heavy|very_heavy), `soil_state` (dry|slight|wet|waterlogged|unsure), `irrigation` (regular|occasional|summer|none), `tree_state` (healthy|yellow|drying|fewer|weak), `tree_age` (lt5|5_10|10_20|20_40|gt40|unsure), `prev_prod` (lt5|5_20|20_40|gt40)
- advanced: `max_temp_c`, `humidity_pct`, `annual_rain_mm`, `soil_type` (sandy|sandy_loam|loam|silty_clay), `soil_ph`, `irrigation_method`
Returns: productivity (expected, 80% range, per tree), production, previous-year comparison, factor effects, irrigation status (green/yellow/red/blue), recommendations, health score, warnings.

## Honest limits
- District-level model: it cannot see your soil, age, irrigation, pests or disease. Those only affect advice, not the number.
- Forecast skill beyond 1-2 years is about as good as assuming a stable recent level; use the range.
- Farm health score is indicative, not validated.
- Not included yet: live weather (IMD / NASA POWER), voice input, full Malayalam text for every recommendation, accounts/login.
