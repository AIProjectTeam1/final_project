import os
from openai import AsyncOpenAI
from datasets import load_from_disk, Dataset, DatasetDict
import json
from tqdm.asyncio import tqdm_asyncio
import asyncio
import shutil

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

async def summarize_all(jobpost, resume):
    # 모든 정보를 한 번에 요약하는 프롬프트
    prompt_template = """
    다음은 채용 공고와 이력서입니다. 이 정보들을 종합적으로 분석하여 요약해주세요.
    반드시 아래와 같은 JSON 형식으로만 응답해주세요. 마크다운 코드 블록이나 다른 설명은 포함하지 마세요.
    순수하게 JSON 객체만 반환해주세요.

    채용 공고:
    {context}

    {{
        "summary": {{
            "job_analysis": {{
                "position": "직무명",
                "required_skills": ["필요 스킬1", "필요 스킬2", "필요 스킬3, ..."],
                "preferred_skills": ["우대 스킬1", "우대 스킬2, ..."],
                "responsibilities": ["주요 업무1", "주요 업무2, ..."],
                "requirements": ["자격 요건1", "자격 요건2, ..."]
            }},
            "resume_analysis": {{
                "career_summary": "경력 요약",
                "key_skills": ["주요 스킬1", "주요 스킬2", "주요 스킬3, ..."],
                "education": ["학력1", "학력2, ..."],
                "certifications": ["자격증1", "자격증2, ..."],
                "projects": ["주요 프로젝트1", "주요 프로젝트2, ..."]
            }}
        }}
    }}
    """
    
    # LLM에 분석 요청
    result = await ask_llm(
        question="모든 정보를 종합적으로 분석하여 요약해주세요. 순수 JSON 형식으로만 응답해주세요.",
        answer="",
        context=f"채용 공고:\n{jobpost}\n\n이력서:\n{resume}",
        prompt_template=prompt_template
    )
    
    try:
        # JSON 파싱
        summary = json.loads(result)
        if summary and "summary" in summary:
            return summary
        return None
    except json.JSONDecodeError:
        return None

def create_empty_analysis():
    """빈 분석 결과를 생성하는 함수"""
    return {
        "position": "",
        "required_skills": [],
        "preferred_skills": [],
        "responsibilities": [],
        "requirements": []
    }

def create_empty_resume_analysis():
    """빈 이력서 분석 결과를 생성하는 함수"""
    return {
        "career_summary": "",
        "key_skills": [],
        "education": [],
        "certifications": [],
        "projects": []
    }

async def process_split(split_data, split_name, summaries_split, summaries_dataset):
    """각 데이터셋 분할(train/validation/test)을 처리하는 함수"""
    print(f"\n{split_name} 데이터 처리 중...")
    print(f"데이터 크기: {len(split_data)}")
    
    # 분석 결과를 저장할 리스트
    job_analysis_results = []
    resume_analysis_results = []
    
    # 30개씩 나눠서 처리
    batch_size = 30
    total_batches = (len(split_data) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(split_data))
        
        print(f"\n배치 {batch_idx + 1}/{total_batches} 처리 중...")
        
        # 현재 배치의 채용공고와 이력서들
        current_batch = split_data[start_idx:end_idx]
        current_batch_jobposts = current_batch["jobpost"]
        current_batch_resumes = current_batch["resume"]
        
        # 이미 분석된 항목 건너뛰기
        tasks = []
        new_analyses = False  # 새로운 분석이 있는지 추적
        
        for i, (jobpost, resume) in enumerate(zip(current_batch_jobposts, current_batch_resumes)):
            # 중간 결과에서 이미 분석된 항목 확인
            if (len(summaries_split) > start_idx + i and
                "job_analysis" in summaries_split.features and 
                "resume_analysis" in summaries_split.features and
                summaries_split["job_analysis"][start_idx + i] and
                summaries_split["resume_analysis"][start_idx + i]):
                # 중간 결과에서 가져오기
                job_analysis_results.append(summaries_split["job_analysis"][start_idx + i])
                resume_analysis_results.append(summaries_split["resume_analysis"][start_idx + i])
                print(f"채용공고/이력서 {start_idx + i + 1}는 중간 결과에서 가져왔습니다.")
            else:
                # 분석되지 않은 항목만 분석
                tasks.append(summarize_all(jobpost, resume))
                new_analyses = True
        
        if tasks:  # 분석할 항목이 있는 경우에만 실행
            results = await tqdm_asyncio.gather(*tasks, desc=f"배치 {batch_idx + 1} 분석 중")
            
            # 결과 처리 및 출력 (첫 번째 결과만 출력)
            for i, summary in enumerate(results):
                if summary and "summary" in summary:
                    job_analysis_results.append(summary["summary"]["job_analysis"])
                    resume_analysis_results.append(summary["summary"]["resume_analysis"])
                    if i == 0:  # 각 배치의 첫 번째 결과만 출력
                        print(f"\n채용공고/이력서 {start_idx + 1} 분석 결과:")
                        print(json.dumps(summary, ensure_ascii=False, indent=4))
                        print("-" * 50)
                else:
                    # 분석 실패 시 빈 결과 추가
                    job_analysis_results.append(create_empty_analysis())
                    resume_analysis_results.append(create_empty_resume_analysis())
        
        # 새로운 분석이 있었고, 2배치마다 또는 마지막 배치일 때 중간 결과 저장
        if new_analyses and ((batch_idx + 1) % 2 == 0 or batch_idx == total_batches - 1):
            print(f"\n{batch_idx + 1}번째 배치 완료. 중간 결과 저장 중...")
            
            # 현재까지의 결과로 데이터셋 업데이트
            temp_split_data = split_data.select(range(len(job_analysis_results)))
            
            # 기존 결과와 새 결과 병합
            if len(summaries_split) > 0:
                # 기존 결과의 job_analysis와 resume_analysis 가져오기
                existing_job_analysis = summaries_split["job_analysis"]
                existing_resume_analysis = summaries_split["resume_analysis"]
                
                # 새 결과 추가
                for i in range(len(existing_job_analysis), len(job_analysis_results)):
                    existing_job_analysis.append(job_analysis_results[i])
                    existing_resume_analysis.append(resume_analysis_results[i])
                
                # 병합된 결과로 데이터셋 업데이트
                temp_split_data = temp_split_data.add_column("job_analysis", existing_job_analysis)
                temp_split_data = temp_split_data.add_column("resume_analysis", existing_resume_analysis)
            else:
                # 첫 저장인 경우
                temp_split_data = temp_split_data.add_column("job_analysis", job_analysis_results)
                temp_split_data = temp_split_data.add_column("resume_analysis", resume_analysis_results)
            
            # 중간 결과 업데이트
            summaries_split = temp_split_data
            summaries_dataset[split_name] = summaries_split
            
            # 임시 디렉토리에 저장
            temp_dir = "final_temp"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            summaries_dataset.save_to_disk(temp_dir)
            
            # 기존 디렉토리가 있으면 삭제
            if os.path.exists("final"):
                shutil.rmtree("final")
            
            # 임시 디렉토리를 최종 위치로 이동
            
            print("중간 결과가 저장되었습니다.")
        
        # 잠시 대기 (API 제한 고려)
        await asyncio.sleep(1)
    
    # 분석 결과 업데이트
    if "job_analysis" in split_data.features:
        # 기존 컬럼이 있으면 업데이트
        new_split_data = split_data.remove_columns(["job_analysis", "resume_analysis"])
        new_split_data = new_split_data.add_column("job_analysis", job_analysis_results)
        new_split_data = new_split_data.add_column("resume_analysis", resume_analysis_results)
    else:
        # 기존 컬럼이 없으면 새로 추가
        new_split_data = split_data.add_column("job_analysis", job_analysis_results)
        new_split_data = new_split_data.add_column("resume_analysis", resume_analysis_results)
    
    return new_split_data

async def process_batch(batch_data):
    """배치 단위로 데이터를 처리하는 함수"""
    tasks = []
    for jobpost, resume in zip(batch_data["jobpost"], batch_data["resume"]):
        tasks.append(summarize_all(jobpost, resume))
    
    results = await tqdm_asyncio.gather(*tasks, desc="배치 분석 중")
    
    # 결과 처리
    new_analyses = []
    for summary in results:
        if summary and "summary" in summary:
            new_analyses.append((
                summary["summary"]["job_analysis"],
                summary["summary"]["resume_analysis"]
            ))
        else:
            # 분석 실패 시 빈 결과 추가
            new_analyses.append((
                create_empty_analysis(),
                create_empty_resume_analysis()
            ))
    
    return new_analyses

async def process_dataset():
    try:
        # 기존 데이터셋 로드
        final_dataset = load_from_disk("small_resume_dataset_with_summaries_final")
        original_dataset = load_from_disk("small_resume_dataset_with_keywords_re")
        
        # 마지막 두 배치 처리 (216번째: 30개, 217번째: 20개)
        batch_size = 30
        start_idx = 6450  # 216번째 배치 시작
        end_idx = 6500    # 217번째 배치 끝
        
        print(f"\n처리할 데이터 범위: {start_idx} ~ {end_idx-1}")
        print(f"final 데이터셋 크기: {len(final_dataset['train'])}")
        print(f"원본 데이터셋 크기: {len(original_dataset['train'])}")
        
        # 원본 데이터셋에서 처리되지 않은 데이터 추출
        target_data = original_dataset["train"].select(range(start_idx, end_idx))
        print(f"추출된 데이터 크기: {len(target_data)}")
        
        # 임시 디렉토리 설정
        temp_dir = "final_temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        # 새로운 데이터셋 생성 (모든 필드 포함)
        new_dataset = DatasetDict({
            "train": Dataset.from_dict({
                **{field: target_data[field] for field in target_data.features},
                "job_analysis": [create_empty_analysis() for _ in range(len(target_data))],
                "resume_analysis": [create_empty_resume_analysis() for _ in range(len(target_data))]
            })
        })
        
        # 배치 단위로 처리
        total_batches = (len(target_data) + batch_size - 1) // batch_size
        print(f"\n총 {total_batches}개의 배치를 처리합니다.")
        
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, len(target_data))
            batch_data = new_dataset["train"].select(range(start, end))
            
            print(f"\n배치 {batch_idx + 1}/{total_batches} 처리 중...")
            print(f"현재 처리 중인 데이터 인덱스: {start_idx + start} ~ {start_idx + end - 1}")
            
            # 배치 데이터 처리
            new_analyses = await process_batch(batch_data)
            
            if new_analyses:
                # 새로운 분석 결과로 데이터셋 업데이트
                for i, (job_analysis, resume_analysis) in enumerate(new_analyses):
                    idx = start + i
                    new_dataset["train"][idx]["job_analysis"] = job_analysis
                    new_dataset["train"][idx]["resume_analysis"] = resume_analysis
                
                # 중간 결과 저장
                new_dataset.save_to_disk(temp_dir)
                print(f"배치 {batch_idx + 1} 처리 완료 및 저장됨")
            
            # 2개의 배치마다 또는 마지막 배치에서 중간 결과 저장
            if new_analyses and (batch_idx % 2 == 1 or batch_idx == total_batches - 1):
                if os.path.exists("final"):
                    shutil.rmtree("final")
                shutil.move(temp_dir, "final")
                print(f"중간 결과가 'final' 디렉토리에 저장되었습니다.")
        
        # 최종 데이터셋 생성
        # 1. 기존 데이터셋의 모든 필드 가져오기
        train_data = {field: final_dataset["train"][field] for field in final_dataset["train"].features}
        
        # 2. 새로운 데이터 추가
        for field in train_data:
            if field in ["job_analysis", "resume_analysis"]:
                # 기존 데이터 유지하고 새로운 데이터 추가
                train_data[field] = list(train_data[field]) + list(new_dataset["train"][field])
            else:
                # 원본 데이터셋에서 가져온 데이터 추가
                train_data[field] = list(train_data[field]) + list(original_dataset["train"][field][start_idx:end_idx])
        
        final_dataset = DatasetDict({
            "train": Dataset.from_dict(train_data),
            "validation": final_dataset["validation"],
            "test": final_dataset["test"]
        })
        
        # 최종 결과 저장
        final_dataset.save_to_disk("small_resume_dataset_final")
        print("\n최종 데이터셋이 'small_resume_dataset_final' 디렉토리에 저장되었습니다.")
        
    except Exception as e:
        print(f"에러 발생: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(process_dataset()) 