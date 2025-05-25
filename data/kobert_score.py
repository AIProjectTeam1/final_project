'''
필수 : pip3 install pandas bert-score transformers sentencepiece
오류 상황에 맞춰 각자 apt install 을 통한 패키지 다운로드 필요
'''

from transformers import AutoTokenizer, AutoModel
from bert_score import BERTScorer
import pandas as pd

# if print_mode True , print REF, CAND Data and ROW BERTScore
print_mode_check = input("Print mode 결정 0: False, 1: True -> ").strip()
print_mode = print_mode_check == "1"

model_choice = input("Model 결정 0: gemma, 1: exaone -> ").strip()

if model_choice == "0":
    file_list = ["../model/eval/gemma-ko-2b_simple_test_250525.csv"]

elif model_choice == "1":
    file_list = ["../model/eval/exaone-3.5_simple_eval_250525.csv"]

else:
    print("Invalid model choice. Exiting.")
    exit()

model_name = "skt/kobert-base-v1"

scorer = BERTScorer(
    model_type = model_name,
    lang="ko",
    num_layers = 12,
    rescale_with_baseline=False
)

# references 정답 | candidates : 생성

# references = [
#     "이 이력서는 관련 경험이 풍부하고 직무와 잘 맞습니다.",
#     "이력서에는 직무 관련 기술이 명확하게 나타나 있습니다.",
#     "경험은 부족하나 지원자의 가능성은 높게 평가됩니다.",
# ]

# candidates = [
#     "이 이력서는 관련 활동이 매우 많고 직무와 어울립니다.",
#     "이 후보자는 요구 조건에 맞는 스킬을 모두 보유하고 있습니다.",
#     "경험은 적지만 성장 가능성이 높습니다.",
# ]

total_refs = []
total_cands = []
total_f1_files = []

for file_name in file_list:
    df = pd.read_csv(file_name, header=None, encoding="utf-8")
    references = df[0].astype(str).tolist() # col 0 : references (정답)
    candidates = df[1].astype(str).tolist() # col 1 : candidates (생성)
    _, _, F1 =  scorer.score(candidates, references)


    total_refs.append(references)
    total_cands.append(candidates)
    total_f1_files.append(F1)

row_avg_list = []
num_rows = len(total_f1_files[0])

for i in range(num_rows):
    row_refs = [ref_list[i] for ref_list in total_refs]
    row_cands = [cand_list[i] for cand_list in total_cands]
    row_f1s = [f1_list[i].item() for f1_list in total_f1_files]

    avg = sum(row_f1s) / len(row_f1s)
    row_avg_list.append(avg)

    if print_mode:
        print(f"\nRow {i+1}")
        for j, fname in enumerate(file_list):
            print(f"[{fname}]")
            print(f"REF : {row_refs[j]}")
            print(f"CAND : {row_cands[j]}")
            print(f"F1 : {row_f1s[j]:.4f}")
        print(f">> Row F1 avg : {avg:.4f}")
        
# 모든 BERTScore의 평균 출력
overall_avg = sum(row_avg_list) / len(row_avg_list)
print(f"Model Mean BERTScore : {overall_avg:.4f}")

