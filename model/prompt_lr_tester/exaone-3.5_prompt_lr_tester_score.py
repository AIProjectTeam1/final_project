import pandas as pd
from sentence_transformers import SentenceTransformer, util
import math
import os

# 모델 로드
model = SentenceTransformer("jhgan/ko-sbert-nli")

# 키워드 그룹
AGENCY_GROUP = [["저는", "제가", "본인", "나"], ["스스로", "직접", "주도적으로"]]
FIT_GROUP = [["개발", "프로그래밍", "코딩"], ["기술", "문제 해결", "기획"]]
SPE_GROUP = [["노력", "배운", "피드백"], ["계획", "개선", "향후", "어떻게", "무엇을"]]

def count_matches(text, group):
    if not isinstance(text, str):
        text = str(text)
    return sum(1 for subgroup in group if any(kw in text for kw in subgroup))

def level(count):
    return "높음" if count >= 2 else "중간" if count == 1 else "낮음"

def is_valid_text(text):
    if text is None:
        return False
    if isinstance(text, float):
        return not math.isnan(text)
    return str(text).strip() != ""

def evaluate_qual_meaning(text):
    if not is_valid_text(text):
        return {"주체성": "낮음", "직무적합성": "낮음", "맞춤조언": "낮음"}
    return {
        "주체성": level(count_matches(text, AGENCY_GROUP)),
        "직무적합성": level(count_matches(text, FIT_GROUP)),
        "맞춤조언": level(count_matches(text, SPE_GROUP))
    }

def evaluate_sim(ref, pred):
    if not is_valid_text(ref) or not is_valid_text(pred):
        return None
    emb1 = model.encode(str(ref).strip(), convert_to_tensor=True)
    emb2 = model.encode(str(pred).strip(), convert_to_tensor=True)
    return round(util.pytorch_cos_sim(emb1, emb2).item(), 4)

def evaluate_file(file_path, ref_column="eval_resume", pred_column="generated_text"):
    df = pd.read_csv(file_path)
    if ref_column not in df.columns or pred_column not in df.columns:
        print(f"필요한 컬럼이 존재하지 않습니다: {ref_column}, {pred_column}")
        return pd.DataFrame()
    
    results = []
    for _, row in df.iterrows():
        ref, pred = row[ref_column], row[pred_column]
        sim = evaluate_sim(ref, pred)
        if sim is None:
            continue
        qual = evaluate_qual_meaning(pred)
        results.append({"reference": ref, "prediction": pred, "similarity_score": sim, **qual})
    return pd.DataFrame(results)

# --- 실행 ---
if __name__ == "__main__":
    base_name = "inference_results"

    resume_path = f"{base_name}_resume.csv"
    selfintro_path = f"{base_name}_selfintro.csv"

    # 존재 여부 확인
    if not os.path.exists(resume_path) or not os.path.exists(selfintro_path):
        print("입력한 이름을 기반으로 한 CSV 파일이 존재하지 않습니다.")
        print(f"확인된 경로: {resume_path}, {selfintro_path}")
        exit(1)

    # 평가 실행
    df_resume = evaluate_file(resume_path, ref_column="eval_resume")
    df_selfintro = evaluate_file(selfintro_path, ref_column="eval_selfintro")

    def summarize(df, label):
        if df.empty:
            print(f"\n❌ {label} 평가 데이터 없음")
            return
        print(f"\n✅ [{label}] 평가 요약")
        print(f"- 평균 유사도: {df['similarity_score'].mean():.4f}")
        for col in ["주체성", "직무적합성", "맞춤조언"]:
            dist = df[col].value_counts(normalize=True) * 100
            print(f"  · {col}: 높음({dist.get('높음', 0):.2f}%) / 중간({dist.get('중간', 0):.2f}%) / 낮음({dist.get('낮음', 0):.2f}%)")

    summarize(df_resume, "이력서")
    summarize(df_selfintro, "자기소개서")
