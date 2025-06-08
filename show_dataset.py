from datasets import load_from_disk

# 데이터셋 로드
dataset = load_from_disk('small_resume_dataset_final')

# 첫 번째 샘플 가져오기
sample = dataset['train'][0]

# 샘플의 모든 필드 출력
print("\n=== 데이터셋 샘플의 모든 필드 ===")
for key, value in sample.items():
    print(f"\n{key}:")
    print(value) 