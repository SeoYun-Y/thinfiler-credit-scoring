# -*- coding: utf-8 -*-
"""
01. 씬파일러 사전검증 스크립트
--------------------------------
새 씬파일러 정의: 아래 5개 필드가 전부 0인 경우
  - C1M210000  : 신용카드건수(미해지)
  - C18233003  : 3개월전 신용카드기관수(미해지)
  - C18233004  : 6개월전 신용카드기관수(미해지)
  - C18233005  : 1년전 신용카드기관수(미해지)
  - L10220000  : 대출 관련 보유 건수

모델링(02번 스크립트) 실행 전에 반드시 아래 3가지를 확인한다.
  1) 씬파일러군에서 PERF1~4 라벨이 실제로 채워져 있는가
  2) 이 정의가 기존 SCORE=0 정의와 얼마나 겹치는가 (H1 검증)
  3) 5개 정의 필드가 씬파일러군 내부에서 zero-variance인지 (모델 피처 오염 방지)
"""

import pandas as pd
import numpy as np
import glob
import sys
import io

# Windows 콘솔 기본 인코딩(cp949)에서 ⚠, ✅ 등 이모지 출력 시 발생하는
# UnicodeEncodeError 방지 — stdout을 UTF-8로 재설정
if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 120)

# ------------------------------------------------------------------
# 0. 데이터 로드 — 9번 시트가 4개 파일(월별 스냅샷)로 나뉘어 있으므로 concat
# ------------------------------------------------------------------
import os

# 스크립트 위치: .../프로젝트/creditscore/personalCB/01_씬파일러_사전검증.py
# 데이터 위치:   .../프로젝트/data/9.개인_CB정보/*.csv
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "9.개인_CB정보")

def find_sheet9_files(base_dir: str) -> list:
    patterns = ["*_개인CB.csv", "*_개인CB.xlsx", "*_개인CB.xls", "*.csv"]
    found = []
    for pat in patterns:
        found.extend(glob.glob(os.path.join(base_dir, pat)))
    found = sorted(set(found))
    if not found:
        print(f"⚠ '{base_dir}' 폴더에서 파일을 찾지 못했습니다.")
        if os.path.isdir(base_dir):
            print("현재 폴더 내 파일 목록:")
            for f in os.listdir(base_dir):
                print(f"  - {f}")
        else:
            print("해당 경로가 존재하지 않습니다. 폴더 구조를 다시 확인하세요.")
    return found

SHEET9_PATHS = find_sheet9_files(BASE_DIR)  # 201912_개인CB, 202012_개인CB, 202112_개인CB, 202212_개인CB

THIN_FIELDS = ["C1M210000", "C18233003", "C18233004", "C18233005", "L10220000"]
ID_COL = "ID"
PERF_COLS = ["PERF1", "PERF2", "PERF3", "PERF4"]
ALT_FEATURES = ["AP0910001", "AP0910002", "AL012G005", "AL012G011", "AL012G019"]

# 검증에 필요한 컬럼만 선택적으로 읽기 (132개 전체 컬럼 X)
# → 1,250만 행 x 132컬럼을 통째로 메모리에 올리면서 MemoryError가 났던 문제 해결
NEEDED_COLS = list(dict.fromkeys(
    [ID_COL, "SCORE", "SCORE_6M"] + THIN_FIELDS + PERF_COLS + ALT_FEATURES
))


def load_and_concat(paths: list, needed_cols: list = None) -> pd.DataFrame:
    if not paths:
        raise FileNotFoundError(
            "로드할 파일이 없습니다. 위에 출력된 폴더 내 파일 목록을 보고 "
            "find_sheet9_files()의 패턴이나 BASE_DIR을 실제 파일명에 맞게 "
            "수정하세요."
        )
    dfs = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()

        # 파일의 실제 컬럼명 먼저 확인 → needed_cols 중 존재하는 것만 usecols로 지정
        if ext in (".xlsx", ".xls"):
            header_cols = pd.read_excel(p, nrows=0).columns.tolist()
        else:
            try:
                header_cols = pd.read_csv(p, nrows=0, encoding="utf-8").columns.tolist()
            except UnicodeDecodeError:
                header_cols = pd.read_csv(p, nrows=0, encoding="cp949").columns.tolist()

        if needed_cols is not None:
            usecols = [c for c in needed_cols if c in header_cols]
            missing = [c for c in needed_cols if c not in header_cols]
            if missing:
                print(f"  ⚠ {os.path.basename(p)}: 다음 컬럼 없음 → {missing}")
        else:
            usecols = None

        if ext in (".xlsx", ".xls"):
            d = pd.read_excel(p, usecols=usecols)
        else:
            try:
                d = pd.read_csv(p, encoding="utf-8", usecols=usecols)
            except UnicodeDecodeError:
                d = pd.read_csv(p, encoding="cp949", usecols=usecols)

        # int64 → int32 다운캐스트로 메모리 추가 절감
        for col in d.select_dtypes(include=["int64"]).columns:
            d[col] = pd.to_numeric(d[col], downcast="integer")
        for col in d.select_dtypes(include=["float64"]).columns:
            d[col] = pd.to_numeric(d[col], downcast="float")

        d["__source_file"] = os.path.basename(p)
        dfs.append(d)
        print(f"  로드됨: {os.path.basename(p)} ({len(d):,}행, {len(d.columns)}개 컬럼)")

    df = pd.concat(dfs, ignore_index=True)
    print(f"파일 {len(paths)}개 로드 완료. 합계 행수: {len(df):,}")
    return df


# ------------------------------------------------------------------
# 1. 씬파일러 플래그 생성 + dedup 처리
# ------------------------------------------------------------------
def flag_thin_filers(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in THIN_FIELDS if c not in df.columns]
    if missing:
        raise ValueError(f"씬파일러 정의 필드 누락: {missing}")

    df["is_thin"] = (df[THIN_FIELDS] == 0).all(axis=1)

    n_total_rows = len(df)
    n_thin_rows = df["is_thin"].sum()
    print(f"\n[행 단위] 전체 {n_total_rows:,}건 중 씬파일러 {n_thin_rows:,}건 "
          f"({n_thin_rows / n_total_rows * 100:.2f}%)")

    # 패널 데이터 가능성 → ID 기준 dedup 필요
    if ID_COL in df.columns:
        n_unique_id = df[ID_COL].nunique()
        thin_ids = df.loc[df["is_thin"], ID_COL].nunique()
        print(f"[ID 기준] 전체 고유 ID {n_unique_id:,}명 중 "
              f"씬파일러로 한 번이라도 분류된 ID {thin_ids:,}명 "
              f"({thin_ids / n_unique_id * 100:.2f}%)")

        # 매월 일관되게 씬파일러인 ID만 카운트 (더 엄격한 기준, 참고용)
        consistent = df.groupby(ID_COL)["is_thin"].mean()
        always_thin = (consistent == 1.0).sum()
        print(f"[참고] 모든 관측월에서 일관되게 씬파일러인 ID: {always_thin:,}명")
    else:
        print(f"⚠ ID 컬럼('{ID_COL}')이 없어 dedup 불가 — 실제 컬럼명 확인 필요")

    return df


# ------------------------------------------------------------------
# 2. 확인사항 1 — PERF1~4 라벨 존재 여부
# ------------------------------------------------------------------
def check_perf_availability(df: pd.DataFrame):
    print("\n=== [확인사항 1] 씬파일러군 PERF1~4 라벨 존재 여부 ===")
    thin = df[df["is_thin"]]
    for col in PERF_COLS:
        if col not in df.columns:
            print(f"- {col}: 컬럼 없음")
            continue
        missing_rate = thin[col].isna().mean() * 100
        dist = thin[col].value_counts(dropna=False, normalize=True) * 100
        print(f"\n- {col} (씬파일러군, n={len(thin):,})")
        print(f"    결측률: {missing_rate:.2f}%")
        print(f"    분포(%):\n{dist.round(3)}")
    print("\n※ 결측률이 낮고 분포에 1(연체)이 일정 비율 존재해야 "
          "지도학습 목표변수로 사용 가능합니다.")


# ------------------------------------------------------------------
# 3. 확인사항 2 — 기존 SCORE=0 정의와의 중첩 (H1 검증)
# ------------------------------------------------------------------
def check_overlap_with_score0(df: pd.DataFrame):
    print("\n=== [확인사항 2] 신규 정의 vs SCORE=0 정의 중첩 (H1) ===")
    if "SCORE" not in df.columns:
        print("⚠ SCORE 컬럼 없음 — 확인 불가")
        return

    new_def = df["is_thin"]
    old_def = df["SCORE"] == 0

    both = (new_def & old_def).sum()
    only_new = (new_def & ~old_def).sum()
    only_old = (~new_def & old_def).sum()

    print(f"신규 정의만 해당:      {only_new:,}건")
    print(f"SCORE=0 정의만 해당:   {only_old:,}건")
    print(f"두 정의 모두 해당:     {both:,}건")

    if new_def.sum() > 0:
        recall = both / old_def.sum() if old_def.sum() > 0 else np.nan
        precision = both / new_def.sum()
        print(f"\nSCORE=0 집단 중 신규정의에도 포함되는 비율(재현율): {recall*100:.2f}%")
        print(f"신규정의 집단 중 SCORE=0에도 해당하는 비율(정밀도):   {precision*100:.2f}%")
        print("\n※ 재현율이 높다면(SCORE=0 대부분이 신규정의에도 포함) "
              "H1이 지지됩니다. 정밀도가 낮다면(신규정의 대부분은 SCORE=0이 "
              "아님) '거래이력 0'과 '평가불가(SCORE=0)'가 서로 다른 현상임을 "
              "시사하므로, 보고서에서 두 정의의 성격 차이를 명확히 설명해야 합니다.")


# ------------------------------------------------------------------
# 4. 확인사항 3 — zero-variance 체크 (모델 피처 오염 방지)
# ------------------------------------------------------------------
def check_zero_variance(df: pd.DataFrame):
    print("\n=== [확인사항 3] 씬파일러군 내부 zero-variance 체크 ===")
    thin = df[df["is_thin"]]
    for col in THIN_FIELDS:
        n_unique = thin[col].nunique()
        print(f"- {col}: 씬파일러군 내 고유값 개수 = {n_unique} "
              f"{'✅ 정의대로 상수(제외 대상)' if n_unique <= 1 else '⚠ 예상과 다름, 재확인 필요'}")
    print("\n※ 이 5개 필드는 씬파일러군 내부 모델(기준/대안모델)의 "
          "feature 목록에서 반드시 제외해야 합니다 (정보량 0).")


# ------------------------------------------------------------------
# 5. 실행부
# ------------------------------------------------------------------
if __name__ == "__main__":
    df = load_and_concat(SHEET9_PATHS, needed_cols=NEEDED_COLS)
    df = flag_thin_filers(df)

    check_perf_availability(df)
    check_overlap_with_score0(df)
    check_zero_variance(df)

    # 다음 단계(02_모델링)에서 재사용할 수 있도록 저장 (personalCB 폴더 안에 저장)
    out_path = os.path.join(SCRIPT_DIR, "sheet9_with_thin_flag.parquet")
    df.to_parquet(out_path, index=False)
    print(f"\n✅ 사전검증 완료. 'is_thin' 플래그가 추가된 데이터를 "
          f"{out_path} 로 저장했습니다.")