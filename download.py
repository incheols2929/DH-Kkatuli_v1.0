from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 모델명 설정
model_name = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"

# 토크나이저 다운로드
print("토크나이저 다운로드 중...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 모델 다운로드 (자동으로 캐시에 저장됨)
print("모델 다운로드 중... (약 20GB, 시간이 걸릴 수 있습니다)")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # 메모리 절약
    device_map="auto",
    trust_remote_code=True
)

print("다운로드 완료!")
print(f"모델 저장 위치: ~/.cache/huggingface/transformers/")
