import os
from openai import AsyncOpenAI
from datasets import load_from_disk, Dataset, DatasetDict
from collections import Counter
import re
import json
from tqdm.asyncio import tqdm_asyncio
import asyncio

def postprocess_output(raw_output):
    """
    LLM의 출력을 후처리하는 함수
    """
    # 앞뒤 공백 제거
    output = raw_output.strip()
    
    # 마크다운 코드 블록 제거
    if output.startswith("```json"):
        output = output[7:]  # ```json 제거
    if output.startswith("```"):
        output = output[3:]  # ``` 제거
    if output.endswith("```"):
        output = output[:-3]  # ``` 제거
    
    # 앞뒤 공백 다시 제거
    output = output.strip()
    
    # JSON 형식이 아닌 경우 빈 문자열 반환
    if not output:
        print(f"처리 실패: {raw_output}")
        return ""
        
    return output

# OpenAI API 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY
)
model_name = "gpt-3.5-turbo"

async def ask_llm(question, answer, context, prompt_template, history=None):
    prompt = prompt_template.format(context=context, question=question, answer=answer)
    messages = history[:] if history else []
    messages.append({"role": "user", "content": prompt})

    response = await openai_client.chat.completions.create(
        model=model_name,
        messages=messages,
    )
    raw_content = response.choices[0].message.content
    raw_output = raw_content.strip() if raw_content else ""
    return postprocess_output(raw_output)

async def analyze_selfintro(selfintro):
    # 자소서 분석을 위한 프롬프트
    prompt_template = """
    다음은 지원자의 자기소개서입니다. 이 자기소개서에서 지원자의 특징과 강점을 잘 나타내는 핵심 키워드나 문장 10개를 추출해주세요.
    반드시 아래와 같은 JSON 형식으로만 응답해주세요. 마크다운 코드 블록이나 다른 설명은 포함하지 마세요.
    순수하게 JSON 객체만 반환해주세요.

    자기소개서:
    {context}

    {{
        "keywords": [
            "키워드/문장1",
            "키워드/문장2",
            "키워드/문장3",
            "키워드/문장4",
            "키워드/문장5",
            "키워드/문장6",
            "키워드/문장7",
            "키워드/문장8",
            "키워드/문장9",
            "키워드/문장10"
        ]
    }}
    """
    
    while True:  # 성공할 때까지 계속 시도
        # LLM에 분석 요청
        result = await ask_llm(
            question="자기소개서에서 지원자의 특징과 강점을 잘 나타내는 핵심 키워드나 문장을 추출해주세요. 순수 JSON 형식으로만 응답해주세요.",
            answer="",
            context=selfintro,
            prompt_template=prompt_template
        )
        
        try:
            # JSON 파싱
            analysis = json.loads(result)
            if analysis and "keywords" in analysis and len(analysis["keywords"]) > 0:
                return analysis
            print("키워드가 없는 결과, 재시도...")
        except json.JSONDecodeError:
            print(f"JSON 파싱 실패, 재시도...")
        
        # 잠시 대기 (API 제한 고려)
        await asyncio.sleep(1)

async def process_split(split_data, split_name):
    """각 데이터셋 분할(train/validation/test)을 처리하는 함수"""
    print(f"\n{split_name} 데이터 처리 중...")
    print(f"데이터 크기: {len(split_data)}")
    
    # 분석 결과를 저장할 리스트
    analysis_results = []
    
    # 30개씩 나눠서 처리
    batch_size = 30
    total_batches = (len(split_data) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(split_data))
        
        print(f"\n배치 {batch_idx + 1}/{total_batches} 처리 중...")
        
        # 현재 배치의 자소서들
        current_batch = split_data[start_idx:end_idx]
        current_batch_selfintros = current_batch["selfintro"]
        
        # 이미 분석된 항목 건너뛰기
        tasks = []
        for i, selfintro in enumerate(current_batch_selfintros):
            if "keywords_analysis" in split_data.features and split_data["keywords_analysis"][start_idx + i]:
                # 이미 분석된 항목은 기존 결과 사용
                analysis_results.append(split_data["keywords_analysis"][start_idx + i])
                print(f"자소서 {start_idx + i + 1}는 이미 분석되어 있습니다.")
            else:
                # 분석되지 않은 항목만 분석
                tasks.append(analyze_selfintro(selfintro))
        
        if tasks:  # 분석할 항목이 있는 경우에만 실행
            results = await tqdm_asyncio.gather(*tasks, desc=f"배치 {batch_idx + 1} 분석 중")
            
            # 결과 처리 및 출력 (첫 번째 결과만 출력)
            for i, analysis in enumerate(results):
                if analysis:
                    analysis_results.append(analysis)
                    if i == 0:  # 각 배치의 첫 번째 결과만 출력
                        print(f"\n자소서 {start_idx + 1} 분석 결과:")
                        print(json.dumps(analysis, ensure_ascii=False, indent=4))
                        print("-" * 50)
                else:
                    # 분석 실패 시 빈 결과 추가
                    analysis_results.append({"keywords": []})
        
        # 잠시 대기 (API 제한 고려)
        await asyncio.sleep(1)
    
    # 분석 결과 업데이트
    if "keywords_analysis" in split_data.features:
        # 기존 컬럼이 있으면 업데이트
        new_split_data = split_data.remove_columns(["keywords_analysis"])
        new_split_data = new_split_data.add_column("keywords_analysis", analysis_results)
    else:
        # 기존 컬럼이 없으면 새로 추가
        new_split_data = split_data.add_column("keywords_analysis", analysis_results)
    
    return new_split_data

async def process_dataset():
    # 데이터셋 로드
    dataset = load_from_disk("small_resume_dataset_with_keywords")
    
    # 데이터셋 구조 확인
    print("\n데이터셋 구조:")
    print(f"학습 데이터 크기: {len(dataset['train'])}")
    print(f"검증 데이터 크기: {len(dataset['validation'])}")
    print(f"테스트 데이터 크기: {len(dataset['test'])}")
    
    # validation과 test 데이터만 처리
    new_validation_data = await process_split(dataset["validation"], "검증")
    new_test_data = await process_split(dataset["test"], "테스트")
    
    # 새로운 데이터셋 생성 (train은 그대로 유지)
    new_dataset = DatasetDict({
        "train": dataset["train"],
        "validation": new_validation_data,
        "test": new_test_data
    })
    
    # 데이터셋 저장
    new_dataset.save_to_disk("small_resume_dataset_with_keywords_re")
    print("\n분석이 완료되었습니다.")
    print("결과가 small_resume_dataset_with_keywords_re 디렉토리에 저장되었습니다.")
    
    return new_dataset

if __name__ == "__main__":
    asyncio.run(process_dataset())