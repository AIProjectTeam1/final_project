from datasets import DatasetDict, load_dataset

# 예를 들어 원본 dataset 로드된 상태라고 가정
dataset = load_dataset("Youseff1987/resume-matching-dataset-v2")

# 소량 샘플 개수 설정
train_sample_size = 200
val_sample_size = 50
test_sample_size = 50

# 데이터 섞고 추출 (필요시)
small_train = dataset["train"].shuffle(seed=42).select(range(train_sample_size))
small_val = dataset["validation"].shuffle(seed=42).select(range(val_sample_size))
small_test = dataset["test"].shuffle(seed=42).select(range(test_sample_size))

# DatasetDict 생성
small_dataset = DatasetDict({
    "train": small_train,
    "validation": small_val,
    "test": small_test,
})

# 로컬에 저장할 경로 지정
save_path = "./small_resume_dataset"

# 저장 (disk에 저장)
small_dataset.save_to_disk(save_path)

print(f"Saved small dataset to {save_path}")
