import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Course Recommender API", version="1.0")

# ---------- Settings (read from the environment) ----------
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
API_KEY = os.getenv("API_KEY")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def check_key(x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------- Load the saved files once, at startup ----------
df = joblib.load("courses_df.joblib").reset_index(drop=True)
sim = np.asarray(joblib.load("sim.joblib"))
assert sim.shape[0] == len(df), "Course table and similarity table do not match"
R_MIN, R_MAX = df["course_rating"].min(), df["course_rating"].max()


# ---------- Ranking logic ----------
def rank(scores, exclude_idx, n, difficulty, rating_weight, pool=50):
    order = [int(i) for i in np.argsort(scores)[::-1] if i != exclude_idx][:pool]
    result = df.iloc[order].copy()
    result["similarity"] = scores[order]
    result = result[result["similarity"] > 0]

    if difficulty:
        result = result[result["course_difficulty"].str.lower() == difficulty.lower()]
    if result.empty:
        return []

    rating_norm = (result["course_rating"] - R_MIN) / (R_MAX - R_MIN)
    result["score"] = (1 - rating_weight) * result["similarity"] + rating_weight * rating_norm
    result = result.sort_values("score", ascending=False).head(n)
    result["course_id"] = result.index

    cols = ["course_id", "course_title", "course_organization",
            "course_rating", "course_difficulty", "similarity", "score"]
    return result[cols].round(3).to_dict(orient="records")


# ---------- Endpoints ----------
@app.get("/")
def home():
    return {"message": "Course Recommender API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "courses": len(df)}


@app.get("/courses", dependencies=[Depends(check_key)])
def list_courses(search: Optional[str] = None, limit: int = Query(20, ge=1, le=100)):
    data = df
    if search:
        data = df[df["course_title"].str.contains(search, case=False, regex=False)]
    out = data.head(limit)[["course_title", "course_organization", "course_difficulty"]].copy()
    out["course_id"] = out.index
    return out.to_dict(orient="records")


@app.get("/recommend/course/{course_id}", dependencies=[Depends(check_key)])
def recommend_by_course(
    course_id: int,
    n: int = Query(5, ge=1, le=20),
    difficulty: Optional[str] = None,
    rating_weight: float = Query(0.2, ge=0.0, le=0.5),
):
    if course_id < 0 or course_id >= len(df):
        raise HTTPException(status_code=404, detail="Course ID not found")
    scores = sim[course_id]
    return {
        "course": df.loc[course_id, "course_title"],
        "recommendations": rank(scores, course_id, n, difficulty, rating_weight),
    }