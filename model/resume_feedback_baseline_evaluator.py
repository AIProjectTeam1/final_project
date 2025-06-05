import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm

# === 설정 ===
PROMPT_PATH = "prompt/system_prompt_base.txt"
SHOT_DB_SIZE = 1000
TEST_SIZE = 500
MAX_NEW_TOKENS = 512
MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
BATCH_SIZE = 8

# === 프롬프트 로딩 ===
def load_system_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

system_prompt = load_system_prompt(PROMPT_PATH)

# === 데이터 로딩 ===
dataset = load_from_disk("small_resume_dataset_final")
shot_db = dataset["train"].select(range(SHOT_DB_SIZE))
test_set = dataset["test"].select(range(TEST_SIZE))

# === 모델 로딩 (양자화)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.padding_side = "left"  # ✅ 중요: decoder-only 모델에서는 left padding 필요

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

# === 평가 XML 파싱
def parse_evaluation(xml_string):
    root = ET.fromstring(xml_string)
    return {child.tag: child.text for child in root}

# === 임베딩 (샷 유사도 기반 선택용)
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
shot_db_embeddings = embed_model.encode(shot_db["selfintro"], convert_to_tensor=True)

# === 샷 선택 함수
def retrieve_shots(query_text, top_k=0):
    if top_k == 0:
        return []
    query_embedding = embed_model.encode(query_text, convert_to_tensor=True)
    scores = torch.nn.functional.cosine_similarity(query_embedding, shot_db_embeddings)
    top_indices = torch.topk(scores, k=top_k).indices.cpu().tolist()
    return [shot_db[i] for i in top_indices]

# === 프롬프트 구성 함수
def build_prompt(example, shot_examples):
    messages = [{"role": "system", "content": system_prompt}]
    if shot_examples:
        messages.append({
            "role": "user",
            "content": (
                "다음은 자기소개서에 대한 피드백 예시입니다. "
                "형식을 참고하여 이후 항목도 동일한 방식으로 작성해 주세요."
            )
        })
    for shot in shot_examples:
        eval_dict = parse_evaluation(shot["evaluation"])
        messages.append({"role": "user", "content": f"[자기소개서]\n{shot['selfintro']}"} )
        messages.append({"role": "assistant", "content": eval_dict.get('eval_selfintro', '')})
    messages.append({"role": "user", "content": f"[자기소개서]\n{example['selfintro']}"} )
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# === 배치 추론 함수 (left padding, truncation 제거)
def generate_feedback_batch(prompt_texts):
    # 프롬프트 길이 정확히 측정 (special token 제외)
    raw_inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, add_special_tokens=False)
    prompt_lens = [input_id.shape[0] for input_id in raw_inputs["input_ids"]]

    # 실제 입력
    inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

    # 프롬프트 이후만 정확히 슬라이스
    return [
        tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip()
        for output, prompt_len in zip(outputs, prompt_lens)
    ]

# === 전체 실행 루프
for shot_k in range(4):  # 0~3-shot
    results = []
    print(f"[INFO] {shot_k}-shot 추론 시작...")

    for batch_start in tqdm(range(0, len(test_set), BATCH_SIZE), desc=f"{shot_k}-shot 진행 중"):
        batch_examples = test_set.select(range(batch_start, min(batch_start + BATCH_SIZE, len(test_set))))

        batch_prompts = []
        batch_ids = []
        for i, ex in enumerate(batch_examples):
            shots = retrieve_shots(ex["selfintro"], top_k=shot_k)
            prompt = build_prompt(ex, shots)
            batch_prompts.append(prompt)
            batch_ids.append(ex.get("id", batch_start + i))

        batch_responses = generate_feedback_batch(batch_prompts)

        for i, (ex, response, shots) in enumerate(zip(batch_examples, batch_responses, [retrieve_shots(ex["selfintro"], top_k=shot_k) for ex in batch_examples])):
            rag_texts = "\n\n".join([shot["selfintro"] for shot in shots])
            results.append({
                "eval_selfintro": parse_evaluation(ex["evaluation"]).get("eval_selfintro", ""),
                "generated_text": response,
                "selfintro": ex["selfintro"],
                "retrieved_shots": rag_texts
            })

    output_file = f"results_{shot_k}shot.csv"
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"[INFO] 저장 완료: {output_file}")
