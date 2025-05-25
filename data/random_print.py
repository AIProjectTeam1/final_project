import pandas as pd
import random

model_choice = input("Model 결정 0: gemma, 1: exaone -> ").strip()

if model_choice == "0":
    file_list = ["../model/eval/gemma-ko-2b_simple_test_250525.csv"]
elif model_choice == "1":
    file_list = ["../model/eval/exaone-3.5_simple_eval_250525.csv"]
else:
    print("Invalid model choice. Exiting.")
    exit()

df = pd.read_csv(file_list[0], encoding="utf-8", header=None)
df = df.iloc[1:] 

sample_rows = df.sample(n=5)

print("\n랜덤 샘플 5개")
for idx, row in sample_rows.iterrows():
    print(f"\n[Row {idx}]")
    print(f"원본(정답) : {row[0]}")
    print(f"\n생성 : {row[1]}")