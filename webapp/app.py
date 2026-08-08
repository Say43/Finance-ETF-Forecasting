"""Local web frontend for Kronos ETF forecasting.

Enter an ETF ticker, get a comprehensive forecast: a bias-corrected,
calibration-shrunk 20-trading-day path with 80%/90% split-conformal
prediction bands, a regime read, and an honest model-quality note.

Run:
    python webapp/app.py
then open http://127.0.0.1:5000 in a browser.

The heavy Kronos predictor is loaded once on the first request and reused;
GPU inference is serialised behind a lock so concurrent browser tabs cannot
collide on CUDA.
"""

from __future__ import annotations

import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kronos_etf.asset_config import get_asset_config  # noqa: E402
from kronos_etf.calibration import apply_calibration, load_calibration  # noqa: E402
from kronos_etf.kronos_utils import (  # noqa: E402
    DEVICE_INFO,
    choose_best_prediction,
    fetch_etf_klines,
    load_kronos_predictor,
    make_future_timestamps,
)

app = Flask(__name__)

# --- One-time, lazily-initialised model state (guarded by a lock) ---
_MODEL_LOCK = threading.Lock()
_STATE: dict[str, object] = {"predictor": None, "calibration": None}
CALIBRATION_PATH = PROJECT_ROOT / "finetune" / "checkpoints_etf" / "calibration_etf.json"

# Per-symbol v2 backtest credibility (20 walk-forward windows, calibrated).
# cal_mae = calibrated MAE improvement vs naive last-value baseline (%),
# cov80/cov90 = empirical coverage of the conformal bands (%).
# Symbols in the fine-tuning universe are "trained"; VTI/XLF/VNQ are honest
# out-of-sample holdouts. Everything else falls back to the trained average.
BACKTEST = {
    "SPY": {"cal_mae": 1.96, "cov80": 79.2, "cov90": 85.8, "group": "trained"},
    "QQQ": {"cal_mae": -0.83, "cov80": 73.8, "cov90": 83.5, "group": "trained"},
    "IWM": {"cal_mae": 3.27, "cov80": 72.7, "cov90": 84.5, "group": "trained"},
    "DIA": {"cal_mae": -1.17, "cov80": 78.0, "cov90": 86.2, "group": "trained"},
    "EFA": {"cal_mae": 2.10, "cov80": 75.2, "cov90": 84.8, "group": "trained"},
    "GLD": {"cal_mae": 9.02, "cov80": 84.0, "cov90": 94.2, "group": "trained"},
    "TLT": {"cal_mae": -7.62, "cov80": 86.2, "cov90": 94.5, "group": "trained"},
    "VTI": {"cal_mae": -3.39, "cov80": 77.2, "cov90": 84.8, "group": "holdout"},
    "XLF": {"cal_mae": -8.04, "cov80": 82.8, "cov90": 92.2, "group": "holdout"},
    "VNQ": {"cal_mae": -2.99, "cov80": 80.5, "cov90": 88.2, "group": "holdout"},
}
TRAINED_AVG = {"cal_mae": 0.96, "cov80": 78.4, "cov90": 87.6, "group": "untested"}


def get_predictor():
    """Load the fine-tuned ETF predictor + calibration once, then reuse."""
    if _STATE["predictor"] is None:
        with _MODEL_LOCK:
            if _STATE["predictor"] is None:
                config = get_asset_config("SPY", "etf")
                _STATE["predictor"] = load_kronos_predictor(
                    use_finetuned=True, checkpoint_path=config.checkpoint_path
                )
                _STATE["calibration"] = load_calibration(CALIBRATION_PATH)
    return _STATE["predictor"], _STATE["calibration"]


def classify_regime(close: pd.Series) -> tuple[str, float, float, float]:
    """Cheap current-regime read from recent simple returns."""
    close = close.astype(float).reset_index(drop=True)
    ret_20 = float(close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) > 21 else 0.0
    ret_60 = float(close.iloc[-1] / close.iloc[-61] - 1.0) if len(close) > 61 else 0.0
    daily = close.pct_change().dropna().tail(60)
    ann_vol = float(daily.std() * np.sqrt(252)) if not daily.empty else 0.0
    if ret_60 < -0.10:
        regime = "CRASH"
    elif ret_60 < -0.02:
        regime = "BEAR"
    elif ret_60 > 0.02:
        regime = "BULL"
    else:
        regime = "SIDEWAYS"
    return regime, ret_20 * 100, ret_60 * 100, ann_vol * 100


def run_forecast(symbol: str) -> dict:
    symbol = symbol.strip().upper()
    if not symbol or not all(c.isalnum() or c in ".-" for c in symbol):
        raise ValueError("Invalid ticker symbol.")

    config = get_asset_config(symbol, "etf")
    predictor, calibration = get_predictor()

    df = fetch_etf_klines(symbol, period="2y", interval=config.interval)
    # yfinance occasionally appends a partial/holiday bar with NaN OHLC; drop
    # any incomplete rows so the model never sees a NaN window.
    df = df.dropna(subset=list(config.feature_columns)).reset_index(drop=True)
    lookback, pred_len = config.lookback, config.pred_len
    if len(df) < lookback:
        raise ValueError(
            f"Only {len(df)} daily bars available for {symbol}; need {lookback}. "
            "Is this a valid, liquid ETF/stock ticker with 2y history?"
        )

    feature_columns = list(config.feature_columns)
    x_df = df.iloc[-lookback:][feature_columns].copy()
    x_ts = df.iloc[-lookback:]["timestamp"].reset_index(drop=True)
    y_ts = make_future_timestamps(x_ts.iloc[-1], pred_len, step=config.step, calendar=config.calendar)

    with _MODEL_LOCK:
        pred_df, *_ = choose_best_prediction(predictor, x_df, x_ts, y_ts, pred_len, log=None)

    last_close = float(x_df["close"].iloc[-1])
    if calibration is None:
        raise RuntimeError("Calibration file missing; run calibrate.py first.")
    band = apply_calibration(
        pred_df["close"].astype(float), last_close, x_df["close"].astype(float), calibration
    )

    close = band["close"].astype(float).reset_index(drop=True)
    lo80 = band["lo80"].astype(float).reset_index(drop=True)
    hi80 = band["hi80"].astype(float).reset_index(drop=True)
    lo90 = band["lo90"].astype(float).reset_index(drop=True)
    hi90 = band["hi90"].astype(float).reset_index(drop=True)

    end_price = float(close.iloc[-1])
    exp_return = (end_price / last_close - 1.0) * 100
    if exp_return > 0.3:
        direction = "AUFWÄRTS"
    elif exp_return < -0.3:
        direction = "ABWÄRTS"
    else:
        direction = "SEITWÄRTS"

    regime, ret20, ret60, ann_vol = classify_regime(df["close"])
    cred = BACKTEST.get(symbol, TRAINED_AVG)

    y_dates = [pd.Timestamp(t).strftime("%Y-%m-%d") for t in y_ts]
    horizon = []
    for i in (4, 9, 14, 19):  # ~1,2,3,4 weeks
        if i < pred_len:
            horizon.append(
                {
                    "day": i + 1,
                    "date": y_dates[i],
                    "close": round(float(close.iloc[i]), 2),
                    "ret": round((float(close.iloc[i]) / last_close - 1) * 100, 2),
                    "lo80": round(float(lo80.iloc[i]), 2),
                    "hi80": round(float(hi80.iloc[i]), 2),
                }
            )

    hist_tail = df.iloc[-90:]
    history = [
        {"d": pd.Timestamp(t).strftime("%Y-%m-%d"), "c": round(float(c), 2)}
        for t, c in zip(hist_tail["timestamp"], hist_tail["close"])
    ]
    path = [
        {
            "d": y_dates[i],
            "c": round(float(close.iloc[i]), 2),
            "lo80": round(float(lo80.iloc[i]), 2),
            "hi80": round(float(hi80.iloc[i]), 2),
            "lo90": round(float(lo90.iloc[i]), 2),
            "hi90": round(float(hi90.iloc[i]), 2),
        }
        for i in range(pred_len)
    ]

    return {
        "symbol": symbol,
        "as_of": pd.Timestamp(x_ts.iloc[-1]).strftime("%Y-%m-%d"),
        "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "device": DEVICE_INFO,
        "last_close": round(last_close, 2),
        "horizon_days": pred_len,
        "history": history,
        "path": path,
        "forecast": {
            "end_price": round(end_price, 2),
            "exp_return_pct": round(exp_return, 2),
            "direction": direction,
            "range80": [round(float(lo80.iloc[-1]), 2), round(float(hi80.iloc[-1]), 2)],
            "range90": [round(float(lo90.iloc[-1]), 2), round(float(hi90.iloc[-1]), 2)],
            "band_width80_pct": round(float((hi80 - lo80).mean() / last_close * 100), 2),
        },
        "regime": {"label": regime, "ret_20d": round(ret20, 2), "ret_60d": round(ret60, 2), "ann_vol_pct": round(ann_vol, 1)},
        "credibility": cred,
        "horizon_table": horizon,
    }


@app.route("/api/forecast")
def api_forecast():
    symbol = request.args.get("symbol", "").strip()
    if not symbol:
        return jsonify({"error": "Bitte ein Ticker-Symbol angeben."}), 400
    try:
        return jsonify(run_forecast(symbol))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001 - surface any model/data failure to the UI
        return jsonify({"error": f"Prognose fehlgeschlagen: {exc}"}), 500


@app.route("/")
def index():
    return INDEX_HTML


INDEX_HTML = r"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kronos — ETF Forecast</title>
<style>
  :root{
    --bg:#ffffff; --ink:#0a0a0a; --muted:#9a9a9a; --faint:#c9c9c9;
    --line:#ececec; --hair:#f4f4f4; --chip:#fafafa;
    --band90:#eeeeef; --band80:#dedee1;
  }
  *{box-sizing:border-box;}
  html,body{margin:0;background:var(--bg);color:var(--ink);
    font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
    -webkit-font-smoothing:antialiased;}
  .wrap{max-width:900px;margin:0 auto;padding:56px 24px 100px;}
  .brand{font-size:13px;font-weight:600;letter-spacing:.34em;text-transform:uppercase;}
  .tag{color:var(--muted);font-size:12.5px;margin-top:6px;letter-spacing:.01em;}
  form{display:flex;gap:10px;margin:34px 0 14px;}
  input{flex:1;background:var(--bg);border:1px solid var(--line);border-radius:12px;
    padding:15px 16px;font-size:16px;color:var(--ink);letter-spacing:.06em;text-transform:uppercase;
    transition:border-color .15s;}
  input:focus{outline:none;border-color:var(--ink);}
  input::placeholder{color:var(--faint);letter-spacing:.02em;text-transform:none;}
  button{background:var(--ink);color:#fff;border:0;border-radius:12px;padding:0 26px;
    font-size:14px;font-weight:600;letter-spacing:.02em;cursor:pointer;transition:opacity .15s;}
  button:disabled{opacity:.45;cursor:default;}
  .chips{display:flex;gap:8px;flex-wrap:wrap;}
  .chip{background:var(--chip);border:1px solid var(--line);color:var(--muted);border-radius:999px;
    padding:6px 13px;font-size:12.5px;letter-spacing:.04em;cursor:pointer;transition:.15s;}
  .chip:hover{color:var(--ink);border-color:var(--ink);}
  .err{margin-top:26px;border:1px solid var(--line);border-left:3px solid var(--ink);
    border-radius:10px;padding:14px 16px;font-size:14px;color:#555;}
  .hidden{display:none;}

  #out{margin-top:44px;}
  .hero{display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:20px;
    padding-bottom:26px;border-bottom:1px solid var(--line);}
  .hero .sym{font-size:15px;font-weight:600;letter-spacing:.22em;}
  .hero .horizon{color:var(--muted);font-size:12px;letter-spacing:.05em;margin-top:5px;text-transform:uppercase;}
  .hero .ret{font-size:52px;font-weight:300;line-height:1;letter-spacing:-.02em;}
  .hero .glyph{font-size:26px;vertical-align:middle;margin-right:8px;}
  .hero .side{text-align:right;}
  .hero .side .k{color:var(--muted);font-size:11px;letter-spacing:.09em;text-transform:uppercase;}
  .hero .side .v{font-size:19px;font-weight:400;margin-top:3px;}
  .hero .side .v small{color:var(--muted);font-weight:400;}

  .chart-card{margin:30px 0;position:relative;}
  #chart{position:relative;width:100%;}
  .cap{color:var(--muted);font-size:11px;letter-spacing:.05em;text-transform:uppercase;margin:0 0 4px;}
  .legend{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:11.5px;margin-top:10px;}
  .legend span{display:inline-flex;align-items:center;gap:7px;}
  .sw{width:16px;height:3px;border-radius:2px;display:inline-block;}
  .sw.line{background:var(--ink);height:2px;}
  .sw.b80{background:var(--band80);height:9px;}
  .sw.b90{background:var(--band90);height:9px;}
  .tip{position:absolute;pointer-events:none;background:#fff;border:1px solid var(--line);
    border-radius:9px;padding:8px 10px;font-size:12px;box-shadow:0 6px 22px rgba(0,0,0,.09);
    transform:translate(-50%,-115%);white-space:nowrap;opacity:0;transition:opacity .1s;z-index:5;}
  .tip b{font-weight:600;} .tip .d{color:var(--muted);font-size:11px;margin-bottom:2px;}
  .tip .r{color:var(--muted);font-size:11px;margin-top:2px;}

  .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);
    border:1px solid var(--line);border-radius:14px;overflow:hidden;margin:30px 0;}
  .cell{background:var(--bg);padding:18px 18px;}
  .cell .k{color:var(--muted);font-size:11px;letter-spacing:.07em;text-transform:uppercase;}
  .cell .v{font-size:21px;font-weight:400;margin-top:6px;letter-spacing:-.01em;}
  .cell .v small{color:var(--muted);font-size:13px;}

  h2{font-size:11px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);
    margin:44px 0 16px;}
  table{width:100%;border-collapse:collapse;font-size:14px;}
  th,td{text-align:right;padding:12px 4px;border-bottom:1px solid var(--hair);font-variant-numeric:tabular-nums;}
  th:first-child,td:first-child{text-align:left;color:var(--ink);}
  th{color:var(--muted);font-weight:500;font-size:11px;letter-spacing:.06em;text-transform:uppercase;}
  tr:last-child td{border-bottom:0;}
  .muted{color:var(--muted);}
  .note{color:var(--muted);font-size:12.5px;line-height:1.6;margin-top:14px;}
  .badge{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:3px 10px;
    font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-left:10px;
    vertical-align:middle;}
  .foot{margin-top:40px;padding-top:20px;border-top:1px solid var(--line);
    color:var(--faint);font-size:11px;letter-spacing:.03em;}
  .spin{display:inline-block;width:13px;height:13px;border:2px solid #fff;border-top-color:transparent;
    border-radius:50%;animation:s .7s linear infinite;vertical-align:-1px;margin-right:8px;}
  @keyframes s{to{transform:rotate(360deg);}}
  @media(max-width:640px){
    .grid{grid-template-columns:repeat(2,1fr);}
    .hero .ret{font-size:42px;} .wrap{padding:40px 18px 80px;}
  }
</style>
</head>
<body>
<div class="wrap">
  <div class="brand">Kronos</div>
  <div class="tag">ETF-Prognose · 20 Handelstage · kalibrierte Konformalbänder · keine Anlageberatung</div>

  <form id="f">
    <input id="sym" placeholder="ETF-Ticker eingeben – z. B. SPY" autocomplete="off" autofocus>
    <button id="go" type="submit">Analysieren</button>
  </form>
  <div class="chips" id="chips"></div>

  <div id="err" class="err hidden"></div>
  <div id="out" class="hidden"></div>
</div>
<script>
const CHIPS=["SPY","QQQ","IWM","DIA","EFA","GLD","TLT","VTI","XLF","VNQ"];
const chipbox=document.getElementById("chips");
CHIPS.forEach(s=>{const c=document.createElement("span");c.className="chip";c.textContent=s;
  c.onclick=()=>{document.getElementById("sym").value=s;document.getElementById("f").requestSubmit();};
  chipbox.appendChild(c);});

const f=document.getElementById("f"),out=document.getElementById("out"),err=document.getElementById("err"),go=document.getElementById("go");
const nf=new Intl.NumberFormat("de-DE",{minimumFractionDigits:2,maximumFractionDigits:2});
const money=v=>nf.format(v);
const sign=v=>(v>0?"+":"")+v.toFixed(2)+"%";
const glyph=v=>v>0.3?"▲":(v<-0.3?"▼":"→");
let LAST=null;

f.onsubmit=async e=>{
  e.preventDefault();
  const sym=document.getElementById("sym").value.trim().toUpperCase();
  if(!sym)return;
  err.classList.add("hidden");out.classList.add("hidden");
  go.disabled=true;go.innerHTML='<span class="spin"></span>Rechne';
  try{
    const r=await fetch("/api/forecast?symbol="+encodeURIComponent(sym));
    const d=await r.json();
    if(!r.ok)throw new Error(d.error||"Unbekannter Fehler");
    LAST=d;render(d);
  }catch(ex){err.textContent=ex.message;err.classList.remove("hidden");}
  finally{go.disabled=false;go.textContent="Analysieren";}
};

window.addEventListener("resize",()=>{if(LAST)drawChart(LAST);});

function render(d){
  const fc=d.forecast,rg=d.regime,cr=d.credibility;
  const grp=cr.group==="trained"?"trainiert":cr.group==="holdout"?"Out-of-Sample":"nicht getestet";
  const rows=d.horizon_table.map(h=>{
    const wk=Math.round((h.day)/5);
    return `<tr><td>Woche ${wk} · ${h.date}</td><td>${money(h.close)}</td>
      <td>${glyph(h.ret)} ${sign(h.ret)}</td><td class="muted">${money(h.lo80)} – ${money(h.hi80)}</td></tr>`;
  }).join("");
  out.innerHTML=`
    <div class="hero">
      <div>
        <div class="sym">${d.symbol}<span class="badge">${grp}</span></div>
        <div class="horizon">Erwartung · ${d.horizon_days} Handelstage</div>
        <div class="ret" style="margin-top:18px"><span class="glyph">${glyph(fc.exp_return_pct)}</span>${sign(fc.exp_return_pct)}</div>
      </div>
      <div class="side">
        <div class="k">Aktuell</div><div class="v">${money(d.last_close)}</div>
        <div class="k" style="margin-top:14px">Erwartet</div><div class="v">${money(fc.end_price)} <small>Ø</small></div>
      </div>
    </div>

    <div class="chart-card">
      <div class="cap">Prognosepfad · Historie 90 Tage + 20 Tage Forecast</div>
      <div id="chart"></div>
      <div class="legend">
        <span><i class="sw line"></i>Prognose</span>
        <span><i class="sw b80"></i>80% Band</span>
        <span><i class="sw b90"></i>90% Band</span>
        <span class="muted">gestrichelt = aktueller Kurs</span>
      </div>
    </div>

    <div class="grid">
      <div class="cell"><div class="k">80% Bereich</div><div class="v">${money(fc.range80[0])}<small> – </small>${money(fc.range80[1])}</div></div>
      <div class="cell"><div class="k">90% Bereich</div><div class="v">${money(fc.range90[0])}<small> – </small>${money(fc.range90[1])}</div></div>
      <div class="cell"><div class="k">Regime</div><div class="v">${rg.label}</div></div>
      <div class="cell"><div class="k">Ann. Volatilität</div><div class="v">${rg.ann_vol_pct}<small>%</small></div></div>
    </div>

    <h2>Wochen-Erwartung &amp; Unsicherheit</h2>
    <table>
      <thead><tr><th>Horizont</th><th>Erwartet</th><th>Rendite</th><th>80%-Bereich</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="note">Am Ende des Horizonts liegt der Kurs mit 80% Wahrscheinlichkeit zwischen
      ${money(fc.range80[0])} und ${money(fc.range80[1])}, mit 90% zwischen ${money(fc.range90[0])} und ${money(fc.range90[1])}.
      Mittlere Bandbreite (80%): ${fc.band_width80_pct}% des Kurses. Regime-Rendite 20T/60T: ${sign(rg.ret_20d)} / ${sign(rg.ret_60d)}.</div>

    <h2>Modell-Güte · historischer Backtest (20 Fenster)</h2>
    <div class="grid">
      <div class="cell"><div class="k">Kalibr. MAE vs. Naiv</div><div class="v">${sign(cr.cal_mae)}</div></div>
      <div class="cell"><div class="k">Coverage 80%</div><div class="v">${cr.cov80}<small>% / Soll 80</small></div></div>
      <div class="cell"><div class="k">Coverage 90%</div><div class="v">${cr.cov90}<small>% / Soll 90</small></div></div>
      <div class="cell"><div class="k">Datenklasse</div><div class="v" style="font-size:15px">${grp}</div></div>
    </div>
    <div class="note">Positiver „MAE vs. Naiv" = die kalibrierte Punktprognose war im Rücktest genauer als eine simple
      „Kurs bleibt gleich"-Annahme. Coverage nahe am Sollwert = zuverlässige Bänder.
      ${cr.group==="untested"?"Dieses Symbol wurde nicht einzeln getestet — gezeigt ist der Durchschnitt der 7 trainierten ETFs als grobe Richtgröße.":""}</div>

    <div class="note" style="margin-top:26px;color:#6b6b6b">
      <b style="color:var(--ink)">Ehrliche Einordnung.</b> Die Richtungstrefferquote liegt nahe einem Münzwurf — dieses Modell ist
      <b style="color:var(--ink)">kein Kauf-/Verkaufssignal</b>. Sein Wert liegt in den gut kalibrierten Unsicherheitsbändern und der
      Szenario-/Regime-Einschätzung. Als Bandbreiten-Werkzeug nutzen, nicht als Alpha-Quelle.</div>

    <div class="foot">${d.symbol} · Datenstand ${d.as_of} · erstellt ${d.generated} · ${d.device} · keine Anlageberatung</div>`;
  out.classList.remove("hidden");
  drawChart(d);
}

// ---- Interactive monochrome SVG chart (smooth curve + confidence fan) ----
function smooth(pts){
  if(pts.length<2)return pts.length?`M${pts[0][0]},${pts[0][1]}`:"";
  let s=`M${pts[0][0]},${pts[0][1]}`;
  for(let i=0;i<pts.length-1;i++){
    const p0=pts[i-1]||pts[i],p1=pts[i],p2=pts[i+1],p3=pts[i+2]||pts[i+1];
    const c1x=p1[0]+(p2[0]-p0[0])/6,c1y=p1[1]+(p2[1]-p0[1])/6;
    const c2x=p2[0]-(p3[0]-p1[0])/6,c2y=p2[1]-(p3[1]-p1[1])/6;
    s+=`C${c1x},${c1y} ${c2x},${c2y} ${p2[0]},${p2[1]}`;
  }
  return s;
}
function niceTicks(min,max,n){
  const span=max-min||1,raw=span/n,mag=Math.pow(10,Math.floor(Math.log10(raw)));
  const step=[1,2,2.5,5,10].map(m=>m*mag).find(s=>s>=raw)||10*mag;
  const start=Math.ceil(min/step)*step,out=[];
  for(let v=start;v<=max+1e-9;v+=step)out.push(v);
  return out;
}
function drawChart(d){
  const host=document.getElementById("chart");if(!host)return;
  const W=Math.max(320,host.clientWidth),H=380;
  const pad={t:16,r:54,b:30,l:12};
  const hist=d.history,path=d.path,last=d.last_close;
  const h=hist.length,p=path.length,n=h+p;
  let ymin=Infinity,ymax=-Infinity;
  hist.forEach(o=>{ymin=Math.min(ymin,o.c);ymax=Math.max(ymax,o.c);});
  path.forEach(o=>{ymin=Math.min(ymin,o.lo90);ymax=Math.max(ymax,o.hi90);});
  const py=(ymax-ymin)*0.08||1;ymin-=py;ymax+=py;
  const iw=W-pad.l-pad.r,ih=H-pad.t-pad.b;
  const X=i=>pad.l+(i/(n-1))*iw;
  const Y=v=>pad.t+(1-(v-ymin)/(ymax-ymin))*ih;

  const histPts=hist.map((o,i)=>[X(i),Y(o.c)]);
  const bx=X(h-1);
  const medPts=[[bx,Y(last)]].concat(path.map((o,j)=>[X(h+j),Y(o.c)]));
  const top90=[[bx,Y(last)]].concat(path.map((o,j)=>[X(h+j),Y(o.hi90)]));
  const bot90=[[bx,Y(last)]].concat(path.map((o,j)=>[X(h+j),Y(o.lo90)]));
  const top80=[[bx,Y(last)]].concat(path.map((o,j)=>[X(h+j),Y(o.hi80)]));
  const bot80=[[bx,Y(last)]].concat(path.map((o,j)=>[X(h+j),Y(o.lo80)]));
  const fan=(t,b)=>smooth(t)+smooth(b.slice().reverse()).replace("M","L")+"Z";

  const yt=niceTicks(ymin+py*0.5,ymax-py*0.5,4);
  const grid=yt.map(v=>`<line x1="${pad.l}" y1="${Y(v)}" x2="${W-pad.r}" y2="${Y(v)}" stroke="var(--hair)"/>
     <text x="${W-pad.r+8}" y="${Y(v)+3}" fill="var(--muted)" font-size="10">${money(v)}</text>`).join("");
  const xlab=(i,t,anch)=>`<text x="${X(i)}" y="${H-10}" fill="var(--muted)" font-size="10" text-anchor="${anch}">${t}</text>`;

  const allPts=hist.map((o,i)=>({x:X(i),y:Y(o.c),d:o.d,c:o.c,fc:false}))
    .concat(path.map((o,j)=>({x:X(h+j),y:Y(o.c),d:o.d,c:o.c,lo80:o.lo80,hi80:o.hi80,fc:true})));

  host.innerHTML=`
  <svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="display:block">
    ${grid}
    <path d="${fan(top90,bot90)}" fill="var(--band90)"/>
    <path d="${fan(top80,bot80)}" fill="var(--band80)"/>
    <line x1="${bx}" y1="${pad.t}" x2="${bx}" y2="${H-pad.b}" stroke="var(--faint)" stroke-dasharray="2 3"/>
    <line x1="${pad.l}" y1="${Y(last)}" x2="${W-pad.r}" y2="${Y(last)}" stroke="var(--faint)" stroke-dasharray="4 4"/>
    <path d="${smooth(histPts)}" fill="none" stroke="var(--ink)" stroke-width="1.6"/>
    <path d="${smooth(medPts)}" fill="none" stroke="var(--ink)" stroke-width="1.9"/>
    <circle cx="${bx}" cy="${Y(last)}" r="3" fill="var(--ink)"/>
    ${xlab(0,hist[0].d.slice(2),"start")}${xlab(h-1,"heute","middle")}${xlab(n-1,path[p-1].d.slice(2),"end")}
    <line id="cx" x1="0" y1="${pad.t}" x2="0" y2="${H-pad.b}" stroke="var(--ink)" stroke-width="1" opacity="0"/>
    <circle id="cd" r="3.5" fill="#fff" stroke="var(--ink)" stroke-width="1.5" opacity="0"/>
    <rect id="ov" x="${pad.l}" y="${pad.t}" width="${iw}" height="${ih}" fill="transparent" style="cursor:crosshair"/>
  </svg>
  <div class="tip" id="tip"></div>`;

  const svg=host.querySelector("svg"),ov=host.querySelector("#ov"),
    cx=host.querySelector("#cx"),cd=host.querySelector("#cd"),tip=host.querySelector("#tip");
  ov.addEventListener("mousemove",ev=>{
    const rect=svg.getBoundingClientRect(),mx=ev.clientX-rect.left;
    let best=allPts[0],bd=1e9;
    for(const q of allPts){const dd=Math.abs(q.x-mx);if(dd<bd){bd=dd;best=q;}}
    cx.setAttribute("x1",best.x);cx.setAttribute("x2",best.x);cx.setAttribute("opacity","0.5");
    cd.setAttribute("cx",best.x);cd.setAttribute("cy",best.y);cd.setAttribute("opacity","1");
    tip.style.left=best.x+"px";tip.style.top=best.y+"px";tip.style.opacity="1";
    tip.innerHTML=`<div class="d">${best.d}${best.fc?" · Prognose":""}</div><b>${money(best.c)}</b>`
      +(best.fc?`<div class="r">80%: ${money(best.lo80)} – ${money(best.hi80)}</div>`:"");
  });
  ov.addEventListener("mouseleave",()=>{cx.setAttribute("opacity","0");cd.setAttribute("opacity","0");tip.style.opacity="0";});
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("Kronos ETF Forecast UI  ->  http://127.0.0.1:5000", flush=True)
    # threaded=False: serialise requests so GPU inference never runs concurrently.
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False)
