# -*- coding: utf-8 -*-
"""
03_트랙B_모델링.py
시트 11(통신카드CB결합정보) 기반 Track B 모델링

[데이터 구조 — Notion 'Track B 최종 데이터 구조' 문서 기준]
- track_b_features_train.csv : 신용이력 보유군 285,890명. X(46개, 원-핫 후 67개) + TARGET 포함.
  TARGET은 PYE_D1011000C(1년 내 연체건수)를 이진화한 것 (1건 이상=1, 0건=0). 양성 4.228%.
- track_b_features_apply.csv : 씬파일러 44,110명. X는 동일하게 포함되지만 TARGET 없음.
  씬파일러는 카드/대출 자체가 없어 "연체"라는 사건이 성립하지 않으므로 신뢰할 수 있는
  정답이 없다고 판단해 TARGET을 아예 만들지 않았음 (track_b_01 타겟변수 검증 노트북에서
  6가지 방식으로 확인).
- 씬파일러 정의에 쓰인 5개 필드(PYE_C1M210000, PYE_C18233003/004/005, PYE_L10210000)는
  순환논리를 피하기 위해 X에서 제외되어 있고, 애초에 두 CSV 어디에도 컬럼으로 들어있지 않음.
  즉 "어느 파일에 속해 있는가" 자체가 이미 씬파일러 여부를 나타냄.

[모델링 전략]
- 정답이 있는 train.csv로만 학습·검증한다 (Logistic Regression 베이스라인 + XGBoost 메인).
- 평가: AUC, PR-AUC (TARGET 불균형: 양성 4.23%)
- 학습된 모델을 apply.csv(씬파일러)에 적용해 위험 점수를 산출하되, 이는 "채점"이 아니라
  "예측 후 참고자료화"임 — 씬파일러 점수가 실제로 맞았는지는 검증하지 않는다(할 수 없다).
  대신 val셋(신용이력 보유군)의 점수 분포와 비교해, 씬파일러가 실제로 위험한 분포를 보이는지
  아니면 정보 부족으로 인한 애매한 분포를 보이는지를 참고자료로 남긴다.
- SHAP: XGBoost 기준 변수 중요도 해석
"""

import sys
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 한글 폰트 설정 (Windows: 맑은 고딕, 문서 툴체인과 동일한 폰트 사용 / 없으면 기본 폰트로 폴백)
try:
    plt.rcParams["font.family"] = "Malgun Gothic" if sys.platform.startswith("win") else "AppleGothic"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False

# Windows(cp949) 콘솔에서 한글 깨짐 방지 (기존 스크립트와 동일한 패턴)
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# ------------------------------------------------------------------
# 0. 경로 설정 (환경에 맞게 수정)
# ------------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\tehun\Desktop\multicamp\프로젝트\creditscore\cardCB")
DATA_DIR = BASE_DIR  # train/apply csv가 스크립트와 같은 폴더에 있다고 가정, 다르면 경로 수정
OUT_DIR = BASE_DIR / "outputs_trackB"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "track_b_features_train.csv"
APPLY_PATH = DATA_DIR / "track_b_features_apply.csv"

RANDOM_STATE = 42
ID_COL = "CUST_ID"
TARGET_COL = "TARGET"

# 명목형(순서 없는) 범주 변수 -> 원-핫 인코딩 대상
# JB_TP: 직업유형 코드 6종, HOME_ADM: 거주지 행정구역 코드 17종
NOMINAL_COLS = ["JB_TP", "HOME_ADM"]


def log(msg: str):
    print(f"[TrackB] {msg}")


# ------------------------------------------------------------------
# 1. 데이터 로드
# ------------------------------------------------------------------
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    apply_df = pd.read_csv(APPLY_PATH)

    log(f"train shape: {train.shape}, apply shape: {apply_df.shape}")

    assert TARGET_COL in train.columns, "train 파일에 TARGET 컬럼이 없습니다."
    assert TARGET_COL not in apply_df.columns, "apply 파일에는 TARGET이 없어야 합니다."

    missing_in_apply = set(train.columns) - set(apply_df.columns) - {TARGET_COL}
    missing_in_train = set(apply_df.columns) - set(train.columns)
    assert not missing_in_apply, f"apply에 없는 train 컬럼: {missing_in_apply}"
    assert not missing_in_train, f"train에 없는 apply 컬럼: {missing_in_train}"

    target_rate = train[TARGET_COL].mean()
    log(f"TARGET 양성 비율: {target_rate:.4%} (양성 {train[TARGET_COL].sum()}건 / 전체 {len(train)}건)")

    return train, apply_df


# ------------------------------------------------------------------
# 2. 전처리: 원-핫 인코딩 + train/apply 컬럼 정합성 맞추기
# ------------------------------------------------------------------
def preprocess(train: pd.DataFrame, apply_df: pd.DataFrame):
    train = train.copy()
    apply_df = apply_df.copy()

    y = train[TARGET_COL].astype(int)
    train_ids = train[ID_COL]
    apply_ids = apply_df[ID_COL]

    X_train_raw = train.drop(columns=[ID_COL, TARGET_COL])
    X_apply_raw = apply_df.drop(columns=[ID_COL])

    # 명목형 변수는 문자열로 변환 후 get_dummies (숫자 코드가 크기 순서를 갖지 않으므로)
    for col in NOMINAL_COLS:
        X_train_raw[col] = X_train_raw[col].astype(str)
        X_apply_raw[col] = X_apply_raw[col].astype(str)

    X_train_enc = pd.get_dummies(X_train_raw, columns=NOMINAL_COLS, prefix=NOMINAL_COLS)
    X_apply_enc = pd.get_dummies(X_apply_raw, columns=NOMINAL_COLS, prefix=NOMINAL_COLS)

    # apply셋에 없는 범주 컬럼은 0으로 채우고, train 기준 컬럼 순서로 정렬
    X_apply_enc = X_apply_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    log(f"인코딩 후 피처 수: {X_train_enc.shape[1]}개")

    return X_train_enc, y, train_ids, X_apply_enc, apply_ids


# ------------------------------------------------------------------
# 3. 모델 학습 + 평가
# ------------------------------------------------------------------
def train_logistic(X_tr, y_tr, X_val, y_val):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",  # TARGET 양성 4.23%로 불균형 -> 가중치 보정
        random_state=RANDOM_STATE,
    )
    model.fit(X_tr_s, y_tr)

    val_proba = model.predict_proba(X_val_s)[:, 1]
    auc = roc_auc_score(y_val, val_proba)
    pr_auc = average_precision_score(y_val, val_proba)
    log(f"[Logistic Regression] Val AUC={auc:.4f}, PR-AUC={pr_auc:.4f}")

    return model, scaler, auc, pr_auc


def train_xgboost(X_tr, y_tr, X_val, y_val):
    pos = y_tr.sum()
    neg = len(y_tr) - pos
    scale_pos_weight = neg / pos  # 불균형 보정

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)
    pr_auc = average_precision_score(y_val, val_proba)
    log(f"[XGBoost] Val AUC={auc:.4f}, PR-AUC={pr_auc:.4f} (best_iteration={model.best_iteration})")

    return model, auc, pr_auc


# ------------------------------------------------------------------
# 4. SHAP 분석 (XGBoost 기준)
# ------------------------------------------------------------------
def _fix_xgboost_shap_compat(model):
    """
    xgboost(신버전, 예: 2.x)로 학습한 모델을 구버전 shap(<0.44)의 TreeExplainer에 넣으면
    base_score가 '[5E-1]' 같은 형식으로 저장되어 float() 변환에 실패하는 호환성 버그가 있다.
    (ValueError: could not convert string to float: '[5E-1]')
    booster 설정에서 base_score를 순수 숫자 문자열로 정규화해서 우회한다.
    이미 정상 포맷이면 아무 영향 없음 (실패해도 조용히 넘어감).
    """
    try:
        booster = model.get_booster()
        config = json.loads(booster.save_config())
        base_score = config["learner"]["learner_model_param"]["base_score"]
        fixed = str(float(str(base_score).strip("[]")))
        if fixed != base_score:
            config["learner"]["learner_model_param"]["base_score"] = fixed
            booster.load_config(json.dumps(config))
            log(f"[호환성 우회] xgboost base_score 포맷 수정: {base_score!r} -> {fixed!r}")
    except Exception as e:
        log(f"[호환성 우회] base_score 정규화 스킵 (문제 없을 수 있음): {e}")
    return model


def run_shap(model, X_val, out_dir: Path, sample_n: int = 3000):
    log("SHAP 분석 시작...")
    model = _fix_xgboost_shap_compat(model)
    sample = X_val.sample(min(sample_n, len(X_val)), random_state=RANDOM_STATE)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
    except ValueError as e:
        log(f"[오류] shap.TreeExplainer 생성 실패: {e}")
        log("[안내] 'pip install --upgrade shap' 로 shap을 최신 버전(0.44 이상)으로 올리면 "
            "근본적으로 해결됩니다. 우회 로직으로도 안 되면 이 방법을 시도하세요.")
        raise

    # 요약 플롯 (변수 중요도)
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_bar_trackB.png", dpi=150)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_importance_trackB.png", dpi=150)
    plt.close()

    # 변수별 평균 |SHAP| 값 표
    mean_abs_shap = pd.DataFrame({
        "feature": sample.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    mean_abs_shap.to_csv(out_dir / "shap_feature_importance_trackB.csv", index=False, encoding="utf-8-sig")

    log("SHAP 결과 저장 완료 (shap_summary_bar_trackB.png, shap_importance_trackB.png, shap_feature_importance_trackB.csv)")
    return mean_abs_shap


# ------------------------------------------------------------------
# 5. 씬파일러(apply셋) 위험점수 산출 — "채점"이 아니라 "참고자료화"
#    씬파일러는 신뢰 가능한 정답(TARGET)이 없으므로 이 점수의 정오는 검증하지 않는다.
#    대신 val셋(정답이 있는 신용이력 보유군) 점수 분포를 기준으로 삼아,
#    같은 모델·같은 기준선 위에서 두 집단의 위험도가 어떻게 다른지를 비교 자료로 남긴다.
# ------------------------------------------------------------------
def score_thin_filers(model, apply_ids, X_apply, out_dir: Path):
    proba = model.predict_proba(X_apply)[:, 1]
    result = pd.DataFrame({
        ID_COL: apply_ids,
        "risk_score_trackB": proba,
    })
    result.to_csv(out_dir / "trackB_thinfiler_risk_scores.csv", index=False, encoding="utf-8-sig")
    log(f"씬파일러 위험점수 산출 완료: {len(result)}건 -> trackB_thinfiler_risk_scores.csv "
        f"(주의: 정답 없음. 채점이 아닌 참고자료)")
    log(f"씬파일러 점수 분포: mean={proba.mean():.4f}, median={np.median(proba):.4f}, "
        f"min={proba.min():.4f}, max={proba.max():.4f}")
    return proba


def compare_score_distributions(val_proba, val_y, thin_proba, out_dir: Path):
    """
    신용이력 보유군(val, 정답 있음) vs 씬파일러(apply, 정답 없음) 점수 분포 비교.
    - val셋 점수로 위험 10분위(decile) 경계를 만들고, 씬파일러 점수를 같은 경계로 등급화.
    - 씬파일러가 특정 등급(고위험/저위험)에 쏠리는지, 넓게 흩어지는지를 확인하는 참고자료.
    """
    log("val(신용이력 보유군) vs 씬파일러 점수 분포 비교 시작...")

    # val셋 기준 10분위 경계 (등급 1=저위험 ~ 10=고위험)
    deciles = np.quantile(val_proba, np.linspace(0, 1, 11))
    deciles[0], deciles[-1] = -np.inf, np.inf  # 경계 밖 값 포함

    val_grade = pd.cut(val_proba, bins=deciles, labels=range(1, 11), include_lowest=True)
    thin_grade = pd.cut(thin_proba, bins=deciles, labels=range(1, 11), include_lowest=True)

    # 등급별 분포 비교 표: val은 실제 TARGET 양성률도 같이 표기(씬파일러는 정답 없어 비교 불가)
    val_df = pd.DataFrame({"grade": val_grade, "target": val_y.values})
    val_summary = val_df.groupby("grade", observed=True).agg(
        val_count=("target", "size"),
        val_actual_bad_rate=("target", "mean"),
    )
    thin_summary = pd.Series(thin_grade).value_counts().sort_index().rename("thin_count").to_frame()
    thin_summary["thin_ratio"] = thin_summary["thin_count"] / thin_summary["thin_count"].sum()
    val_summary["val_ratio"] = val_summary["val_count"] / val_summary["val_count"].sum()

    comparison = val_summary.join(thin_summary, how="outer").sort_index()
    comparison.to_csv(out_dir / "trackB_val_vs_thinfiler_grade_comparison.csv", encoding="utf-8-sig")

    # 히스토그램 오버레이 (분포 형태 비교)
    plt.figure(figsize=(8, 5))
    plt.hist(val_proba, bins=30, alpha=0.5, density=True, label=f"신용이력 보유군 val (n={len(val_proba)})")
    plt.hist(thin_proba, bins=30, alpha=0.5, density=True, label=f"씬파일러 apply (n={len(thin_proba)})")
    plt.xlabel("예측 위험 확률")
    plt.ylabel("밀도")
    plt.title("Track B: val vs 씬파일러 위험점수 분포 비교")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "trackB_val_vs_thinfiler_distribution.png", dpi=150)
    plt.close()

    log("분포 비교 저장 완료 (trackB_val_vs_thinfiler_grade_comparison.csv, "
        "trackB_val_vs_thinfiler_distribution.png)")
    return comparison


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    train, apply_df = load_data()
    X, y, train_ids, X_apply, apply_ids = preprocess(train, apply_df)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    log(f"train/val split: train={X_tr.shape[0]}건, val={X_val.shape[0]}건")

    # 1) Logistic Regression 베이스라인
    logit_model, scaler, logit_auc, logit_prauc = train_logistic(X_tr, y_tr, X_val, y_val)

    # 2) XGBoost 메인 모델
    xgb_model, xgb_auc, xgb_prauc = train_xgboost(X_tr, y_tr, X_val, y_val)

    # 3) SHAP 해석 (XGBoost 기준)
    shap_importance = run_shap(xgb_model, X_val, OUT_DIR)

    # 4) 씬파일러(apply셋) 위험점수 산출 — XGBoost 최종 모델 사용, 정답 없으므로 채점 아님
    val_proba = xgb_model.predict_proba(X_val)[:, 1]
    thin_proba = score_thin_filers(xgb_model, apply_ids, X_apply, OUT_DIR)

    # 5) val(신용이력 보유군) vs 씬파일러 점수 분포 비교 — 참고자료
    compare_score_distributions(val_proba, y_val, thin_proba, OUT_DIR)

    # 6) 성능 요약 리포트 저장
    report = {
        "logistic_regression": {"val_auc": logit_auc, "val_pr_auc": logit_prauc},
        "xgboost": {"val_auc": xgb_auc, "val_pr_auc": xgb_prauc},
        "target_positive_rate": float(y.mean()),
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_val)),
        "n_thin_filers_scored": int(len(X_apply)),
        "n_features": int(X.shape[1]),
        "top10_shap_features": shap_importance.head(10)["feature"].tolist(),
        "note": "씬파일러(apply) 점수는 정답이 없어 검증 불가. 참고자료로만 사용할 것.",
    }
    with open(OUT_DIR / "trackB_model_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log("=== Track B 모델링 완료 ===")
    log(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()