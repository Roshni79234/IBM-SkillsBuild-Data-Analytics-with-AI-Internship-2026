# Airline Operations & Performance Analytics Dashboard

**Author:** Roshni  
**Tech Stack:** Python · Pandas · NumPy · Plotly · Streamlit

---

## Project Overview

A professional Business / Data Analytics dashboard that transforms raw flight-operations data into actionable executive insights. The project demonstrates a complete end-to-end analytics workflow:

```
Raw Data → Cleaning → Validation → KPI Analysis → Trends → Drivers → Risk → Opportunity → Action → Dashboard → Executive Insights
```

---

## Business Problem

Management needs to understand:

- How flight operations are performing across airlines, airports, and routes
- Whether delays and cancellations are increasing or decreasing over time
- Which airlines, airports, and routes show different operational performance levels
- Which delay causes contribute the most delay minutes
- Which time periods carry higher operational risk
- Where improvement opportunities may exist and what actions could be investigated

---

## Dataset

**Flight Delay and Cancellation Data 2019–2023**  
Source: [Kaggle — Patrick Zel](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-data-2019-2023-v2)

Sample used: `flights_sample_100k.csv` (100,000 rows · 32 columns)  
Date range: 2019-01-01 to 2023-08-31  
Airlines: 18 | Origin airports: 372 | Destination airports: 376

---

## Key Objectives

1. Inspect and document actual data quality
2. Clean data with fully transparent, documented decisions
3. Validate the cleaned dataset with 10 explicit checks
4. Calculate airline-specific operational KPIs
5. Analyse trends, drivers, risks, and opportunities
6. Deliver a professional interactive Streamlit dashboard
7. Provide verified, rule-based executive insights

---

## Technologies Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, cleaning, aggregation |
| `numpy` | Numeric operations |
| `plotly` | Interactive charts and heatmaps |
| `streamlit` | Dashboard UI and caching |

No ML libraries, no Flask, no JavaScript, no HTML/CSS.

---

## Project Structure

```
Roshni_AirlineAnalytics/
│
├── app.py                                   ← Single Streamlit application file
│
├── data/
│   └── flights_sample_100k.csv             ← Dataset (place here before running)
│
├── requirements.txt                         ← Python dependencies
│
├── README.md                                ← This file
│
└── Roshni_AirlineAnalytics_ProjectReport.docx  ← Professional Word report
```

---

## Data Cleaning

The cleaning pipeline (`clean_data()`) is fully documented. Key steps:

| Step | Action | Rationale |
|------|--------|-----------|
| Duplicates | Removed 0 exact duplicates | None found |
| `AIRLINE_DOT` | Dropped | Redundant concatenation of AIRLINE + CODE |
| `FL_DATE` | Parsed to datetime | Required for all time analysis |
| Date features | Derived YEAR, MONTH, QUARTER, DOW, YEAR_MONTH | Business groupings |
| `DEP_HOUR` | Derived from `CRS_DEP_TIME ÷ 100` | Scheduled hour (always present) |
| `CANCELLED` / `DIVERTED` | Converted float → int | Cleaner aggregation |
| `CANCELLATION_CODE` NaN | Retained as-is | Correct — only set for cancelled flights |
| `DELAY_DUE_*` NaN | Retained; filled with 0 only at aggregation | NaN = not delayed, not unknown |
| `IS_DELAYED` | Derived flag | `ARR_DELAY >= 15` AND NOT cancelled |
| `ROUTE` | Derived | `ORIGIN + '-' + DEST` |

---

## Data Validation

10 post-cleaning validation checks are run automatically and displayed in the Data Quality Report page:

- CANCELLED / DIVERTED values are only 0 or 1
- DISTANCE ≥ 0 for all rows
- DEP_HOUR in range 0–23
- FL_DATE has no nulls after parsing
- YEAR within 2019–2023
- Every cancelled flight has a CANCELLATION_CODE
- IS_DELAYED is never True for cancelled flights
- ARR_DELAY is numeric
- Delay causes absent for non-delayed flights

**All 10 checks pass on the sample dataset.**

---

## Feature Engineering

| Feature | Source | Purpose |
|---------|--------|---------|
| `YEAR / MONTH / QUARTER` | FL_DATE | Time grouping |
| `MONTH_NAME / DOW_NAME` | FL_DATE | Readable labels |
| `YEAR_MONTH` | FL_DATE | Monthly trend axis |
| `DEP_HOUR` | `CRS_DEP_TIME ÷ 100` | Time-of-day analysis |
| `IS_CANCELLED` | CANCELLED == 1 | Boolean flag |
| `IS_DIVERTED` | DIVERTED == 1 | Boolean flag |
| `IS_DELAYED` | ARR_DELAY ≥ 15 AND not cancelled | FAA-standard delay flag |
| `ROUTE` | ORIGIN + '-' + DEST | Route analysis |

---

## KPI Definitions

| KPI | Definition | Denominator |
|-----|-----------|-------------|
| On-Time Rate % | Flights with ARR_DELAY < 15 min | Operated flights |
| Cancellation Rate % | Cancelled flights | Total scheduled flights |
| Delay Rate % | Flights with ARR_DELAY ≥ 15 min | Operated flights |
| Diversion Rate % | Diverted flights | Total scheduled flights |
| Avg Arrival Delay | Mean ARR_DELAY | All operated flights |
| Avg Delay (delayed only) | Mean ARR_DELAY where delayed | Delayed flights only |
| Avg Departure Delay | Mean DEP_DELAY | All operated flights |

**Delay threshold: ARR_DELAY ≥ 15 minutes (FAA standard)**

---

## Business Analysis Framework

```
LEVEL 1 — KPIs       What is happening?
LEVEL 2 — TRENDS     How is it changing over time?
LEVEL 3 — DRIVERS    Where / why is performance concentrated?
LEVEL 4 — RISK       What historical patterns warrant attention?
LEVEL 5 — ACTION     What should management investigate?
```

---

## Dashboard Pages

### Page 1 — Executive Overview
- Top KPI cards (On-Time %, Cancellation %, Avg Delay)
- Monthly flight volume trend
- Monthly on-time rate trend
- Monthly cancellation rate trend
- Monthly average arrival delay trend
- Full KPI reference table

### Page 2 — Operations & Drivers
- Airline on-time rate comparison
- Airline cancellation rate comparison
- Airport delay performance (configurable top-N)
- Route performance by average delay (configurable top-N)
- Delay cause breakdown (pie + bar)
- Cancellation cause breakdown
- Day-of-week delay analysis
- Departure hour delay analysis
- Departure Hour × Day of Week heatmap
- Monthly seasonality analysis

### Page 3 — Risk, Opportunity & Action
- Historical risk patterns (above-average delay airlines, high-cancel airports, top-delay routes)
- Delay cause concentration warning
- Improvement opportunities with benchmark airline
- FACT → INSIGHT → OPPORTUNITY → ACTION framework (3 action items)
- Executive Analyst summary (5 Findings, 3 Risks, 3 Opportunities, 5 Actions)

### Page 4 — Data Quality Report
- Before / after cleaning summary table
- Missing value explanations
- 10 validation check results
- Year distribution chart
- Engineered features reference table

---

## AI Executive Analyst

The Executive Analyst section uses a **rule-based approach** — no external AI API is required. All statements are derived exclusively from the verified metrics calculated by the dashboard. No numbers are invented, no predictions are made.

The output is clearly labelled: *"Rule-Based Insights (derived exclusively from verified dashboard metrics — no AI API)"*

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open automatically in your default browser at `http://localhost:8501`.

---

## Expected Output

A fully interactive Streamlit dashboard with:
- Sidebar filters (Year, Airline, Origin Airport, Destination Airport)
- All KPIs and charts update dynamically when filters change
- Professional Plotly interactive charts throughout
- Graceful error messages if the CSV is missing or filters produce empty data

---

## Key Insights (from actual dataset)

> All numbers below are from the actual processed dataset.

- **Overall on-time rate: 81.52%** (79,381 / 97,373 operated flights)
- **Cancellation rate: 2.63%** (2,627 / 100,000 flights)
- **Average delay (delayed flights only): 67.1 minutes**
- **Largest delay cause: Carrier** (37.1% of total delay minutes — 448,232 min)
- **Late Aircraft** is the second-largest cause (37.0% — 446,947 min)
- **Best on-time airline in dataset: Endeavor Air (9E) — 87.3%**
- **Highest delay rate: JetBlue Airways (B6) — 28.7%**
- **June has the highest monthly delay rate (23.9%)**; September the lowest (13.2%)
- **Friday** is associated with the highest day-of-week delay rate (20.2%)

---

## Limitations

- Results are from a 100,000-row sample — full dataset may show different patterns
- 2020 volume is significantly lower due to COVID-19 (15,823 flights vs 25,083 in 2019)
- 2023 is a partial year (through August only)
- Delay causes are only available for flights where ARR_DELAY ≥ 15 min
- No revenue, cost, or passenger data is available — analysis is operational only
- Causal claims cannot be made from observational data alone

---

## Future Scope

- Connect to full multi-year dataset for more robust trend analysis
- Add aircraft type and tail-number analysis for maintenance-related delays
- Incorporate weather severity data to contextualise weather cancellations
- Add cost-per-delay-minute estimates if financial data becomes available
- Optional: OpenAI / Anthropic integration for natural-language insight generation
