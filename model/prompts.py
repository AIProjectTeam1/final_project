def dict_to_text(d, indent=0):
    lines = []
    indent_str = "  " * indent
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{indent_str}{k}:")
                lines.extend(dict_to_text(v, indent + 1))
            else:
                lines.append(f"{indent_str}{k}: {v}")
    elif isinstance(d, list):
        for item in d:
            if isinstance(item, (dict, list)):
                lines.append(f"{indent_str}-")
                lines.extend(dict_to_text(item, indent + 1))
            else:
                lines.append(f"{indent_str}- {item}")
    else:
        lines.append(f"{indent_str}{str(d)}")
    return lines

def format_dict_to_string(d):
    return "\n".join(dict_to_text(d))

def prompt_v1_selfintro_only(example, eval_dict):
    return [
        {"role": "system", "content": "You are a helpful assistant for resume evaluation."},
        {"role": "user", "content": f"""[Job-Post]\n{example['job_analysis']}\n
[Resume]\n{example['resume_analysis']}\n
[Keywords]\n{example['keywords_analysis']}\n
[Self-Introduction]\n{example['selfintro']}
위 정보를 바탕으로 자기소개서를 평가해 주세요. 피드백은 한 문단, 3~4문장 내외로 작성해 주세요."""},
        {"role": "assistant", "content": eval_dict.get("eval_selfintro", "")}
    ]

def prompt_v2_all_eval(example, eval_dict):
    return [
        {"role": "system", "content": "You are an assistant that evaluates resume documents."},
        {"role": "user", "content": f"""아래 정보를 바탕으로 각 평가 항목에 대해 작성하세요:\n
[Job-Post]\n{example['job_analysis']}\n
[Resume]\n{example['resume_analysis']}\n
[Keywords]\n{example['keywords_analysis']}\n
[Self-Introduction]\n{example['selfintro']}"""},
        {"role": "assistant", "content": f"""[eval_resume]: {eval_dict.get("eval_resume", "")}
[eval_selfintro]: {eval_dict.get("eval_selfintro", "")}
[summary]: {eval_dict.get("summary", "")}"""}
    ]

def prompt_v2_all_eval_formatted(example, eval_dict):
    job_text = format_dict_to_string(example['job_analysis'])
    resume_text = format_dict_to_string(example['resume_analysis'])
    keywords_text = format_dict_to_string(example['keywords_analysis'])

    return [
        {"role": "system", "content": "You are an assistant that evaluates resume documents."},
        {"role": "user", "content": f"""아래 정보를 바탕으로 각 평가 항목에 대해 작성하세요:\n
[Job-Post]\n{job_text}\n
[Resume]\n{resume_text}\n
[Keywords]\n{keywords_text}\n
[Self-Introduction]\n{example['selfintro']}"""},
        {"role": "assistant", "content": f"""[eval_resume]: {eval_dict.get("eval_resume", "")}
[eval_selfintro]: {eval_dict.get("eval_selfintro", "")}
[summary]: {eval_dict.get("summary", "")}"""}
    ]

def prompt_v2_minimal(example, eval_dict):
    return [
        {"role": "system", "content": "Evaluate resumes."},
        {"role": "user", "content": f"""Job: {example['job_analysis']}\nResume: {example['resume_analysis']}\nKeywords: {example['keywords_analysis']}\nSelf-Introduction: {example['selfintro']}"""},
        {"role": "assistant", "content": f"{eval_dict.get('eval_resume', '')}\n{eval_dict.get('eval_selfintro', '')}\n{eval_dict.get('summary', '')}"}
    ]

def prompt_v2_explicit_guidelines_ko_improved(example, eval_dict):
    job_text = format_dict_to_string(example['job_analysis'])
    resume_text = format_dict_to_string(example['resume_analysis'])
    keywords_text = format_dict_to_string(example['keywords_analysis'])

    return [
        {"role": "system", "content": "당신은 숙련된 인사(HR) 어시스턴트입니다. 지원자의 이력서와 자기소개서를 직무 요구사항에 맞추어 꼼꼼히 평가해 주세요."},
        {"role": "user", "content": f"""아래 문서를 바탕으로 평가 항목별로 구체적인 피드백을 작성해 주세요.\n
[직무 공고]\n{job_text}\n
[이력서]\n{resume_text}\n
[키워드]\n{keywords_text}\n
[자기소개서]\n{example['selfintro']}\n
**평가 지침**
- [eval_resume]: 이력서에서 언급된 기술 및 경험이 직무 요건과 얼마나 부합하는지 평가해 주세요. 기술별 언급이 포함되면 좋습니다. 구체적 근거와 함께 3~5문장으로 작성해 주세요.
- [eval_selfintro]: 자기소개서의 구성, 표현력, 지원 동기, 입사 후 포부, 어투 등을 평가해 주세요. 장점과 개선점을 함께 언급하며 3~5문장으로 작성해 주세요.
- [summary]: 이력서와 자기소개서를 종합하여 전체적인 인상 및 채용 적합성에 대한 의견을 2~4문장으로 요약해 주세요.
각 항목은 반드시 라벨([eval_resume], [eval_selfintro], [summary])을 포함한 형태로 작성해 주세요.
""" },
        {"role": "assistant", "content": f"""[summary]: {eval_dict.get("summary", "")}
[eval_resume]: {eval_dict.get("eval_resume", "")}
[eval_selfintro]: {eval_dict.get("eval_selfintro", "")}"""}
    ]





prompt_templates = {
    "v1_selfintro_only": prompt_v1_selfintro_only,
    "v2_all_eval": prompt_v2_all_eval,
    "v2_all_eval_formatted": prompt_v2_all_eval_formatted,
    "v2_minimal": prompt_v2_minimal,
    "v2_explicit_guidelines_ko_improved": prompt_v2_explicit_guidelines_ko_improved
}


