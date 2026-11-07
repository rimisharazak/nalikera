"""NALIKERA API.  Run from the backend folder:  uvicorn app.main:app --reload   ->  http://localhost:8000"""
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from . import engine

app = FastAPI(title="NALIKERA API", version="1.0")
FRONT = Path(__file__).resolve().parents[2] / "frontend"

class PredictIn(BaseModel):
    mode: str = Field("simple", description="'simple' or 'advanced'")
    district: str; year: Optional[int] = 2027
    # simple-mode answers
    weather_feel: Optional[str] = None; rain_level: Optional[str] = None; soil_state: Optional[str] = None
    irrigation: Optional[str] = None; tree_state: Optional[str] = None; tree_age: Optional[str] = None; prev_prod: Optional[str] = None
    # advanced-mode measurements (all optional)
    max_temp_c: Optional[float] = Field(None, ge=15, le=50); humidity_pct: Optional[float] = Field(None, ge=20, le=100)
    annual_rain_mm: Optional[float] = Field(None, ge=0, le=6000); soil_type: Optional[str] = None
    soil_ph: Optional[float] = Field(None, ge=3, le=10); irrigation_method: Optional[str] = None
    # farm
    trees: Optional[int] = Field(None, ge=1, le=100000); area_ha: Optional[float] = Field(None, gt=0, le=10000)
    prev_total: Optional[float] = Field(None, ge=0); prev_per_tree: Optional[float] = Field(None, ge=0)

@app.get("/api/health")
def health(): return {"status": "ok", "model_loaded": True, "districts": engine.DIST, "forecast_years": [engine.meta["last_year"] + 1, engine.meta["last_year"] + 7]}

@app.post("/api/predict")
def predict(q: PredictIn):
    try: return engine.predict(q.model_dump())
    except ValueError as e: raise HTTPException(422, str(e))

@app.get("/api/history/{district}")
def history(district: str):
    if district not in engine.DIST: raise HTTPException(404, "Unknown district")
    return engine.history(district)

app.mount("/static", StaticFiles(directory=FRONT), name="static")
@app.get("/")
def index(): return FileResponse(FRONT / "index.html")
