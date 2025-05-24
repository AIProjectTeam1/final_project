'''
필수 : pip3 install pandas bert-score transformers sentencepiece
오류 상황에 맞춰 각자 apt install 을 통한 패키지 다운로드 필요
'''

from transformers import AutoTokenizer, AutoModel
from bert_score import BERTScorer
import pandas as pd

model_name = "skt/kobert-base-v1"

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

df = pd.read_csv("template.csv", header=None, encoding="euc-kr")
references = df[0].astype(str).tolist() # col 0 : references (정답)
candidates = df[1].astype(str).tolist() # col 1 : candidates (생성)

scorer = BERTScorer(
    model_type = model_name,
    lang="ko",
    num_layers = 12,
    rescale_with_baseline=False
)

P, R, F1 = scorer.score(candidates, references)

# 각 references, candidates 의 BERTScore 출력
# for i in range(len(candidates)):
#     print(f"\n[{i+1}]")
#     print(f"References: {references[i]}")
#     print(f"Candidates : {candidates[i]}")
#     print(f"BERTScore : {F1[i].item():.4f}")

# 모든 BERTScore의 평균 출력
print(f"Model Mean BERTScore : {F1.mean().item():.4f}")

