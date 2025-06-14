import pandas as pd
from sentence_transformers import SentenceTransformer, util
import sys
import math

# 정량 평가 모델 (KoSimCSE)
model = SentenceTransformer("jhgan/ko-sbert-nli")

AGENCY_GROUP = [["저는", "제가", "본인", "나"], ["스스로", "직접", "주도적으로"]]
FIT_GROUP = [["개발", "프로그래밍", "코딩"], ["기술", "문제 해결", "기획"]]
SPE_GROUP = [["노력", "배운", "피드백"], ["계획", "개선", "향후", "어떻게", "무엇을"]]

def count_matches(text, group):
    return sum(any(kw in str(text) for kw in subgroup) for subgroup in group)

def evaluate_qual_meaning(text):
    def level(count):
        return "높음" if count >= 2 else "중간" if count == 1 else "낮음"
    return {
        "주체성": level(count_matches(text, AGENCY_GROUP)),
        "직무적합성": level(count_matches(text, FIT_GROUP)),
        "맞춤조언": level(count_matches(text, SPE_GROUP))
    }

def is_valid_text(text):
    if text is None:
        return False
    if isinstance(text, float):
        return not math.isnan(text)
    if isinstance(text, str):
        return text.strip() != ""
    return False

def evaluate_sim(ref, pred):
    if not is_valid_text(ref) or not is_valid_text(pred):
        return None
    ref = str(ref).strip()
    pred = str(pred).strip()
    emb1 = model.encode(ref, convert_to_tensor=True)
    emb2 = model.encode(pred, convert_to_tensor=True)
    return round(util.pytorch_cos_sim(emb1, emb2).item(), 4)

def evaluate_file(file_full_path):
    df = pd.read_csv(file_full_path)
    results = []
    for _, row in df.iterrows():
        ref, pred = row.iloc[0], row.iloc[1]
        sim_score = evaluate_sim(ref, pred)
        if sim_score is None:
            continue  # 건너뜀
        qual = evaluate_qual_meaning(pred)
        results.append({
            "reference": ref,
            "prediction": pred,
            "similarity_score": sim_score,
            **qual
        })
    return pd.DataFrame(results)

# 평가 실행
file_path = "../model/eval/"
file_ext = ".csv"
file_list = [
    "exaone-3.5_baseline_eval_resume_250531",
    "guideline_no_pseudo-label_resume",
    "exaone-3.5_baseline_eval_selfintro_250531",
    "guideline_no_pseudo-label_selfintro"
]

mode = int(input("select_mode (0 -> resume, 1 -> self_intro) : "))
if mode == 0:
    file_a = file_list[0]
    file_b = file_list[1]
elif mode == 1:
    file_a = file_list[2]
    file_b = file_list[3]
else:
    print("잘못된 mode값")
    sys.exit(1)

df_a = evaluate_file(file_path + file_a + file_ext)
df_b = evaluate_file(file_path + file_b + file_ext)

mean_a = df_a["similarity_score"].mean()
mean_b = df_b["similarity_score"].mean()

print(f"\n[Baseline] {file_a}")
print(f"  - 평균 유사도: {round(mean_a, 4)}")

print(f"\n[Fine-Tune] {file_b}")
print(f"  - 평균 유사도: {round(mean_b, 4)}")
