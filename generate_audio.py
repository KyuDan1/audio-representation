
# export PYTHONPATH=third_party/Matcha-TTS 해야함
import os
import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# 1. 모델 로드
model_path = '/mnt/ddn/kyudan/pretrained_models/CosyVoice2-0.5B'
cosyvoice = CosyVoice2(model_path)

# 프롬프트 음성 로드 (16kHz)
repo_root = os.path.dirname(os.path.abspath(__file__))
prompt_wav = os.path.join(repo_root, 'asset', 'zero_shot_prompt.wav')
prompt_speech_16k = load_wav(prompt_wav, 16000)

# 2. 설정: 10개의 서로 다른 영어 문장 리스트
sentences = [
    "I can't believe you actually made it on time for the meeting.",
    "The view from the top of the mountain is absolutely breathtaking.",
    "Please put the documents on the table before you leave the room.",
    "I have been waiting for this specific moment for a very long time.",
    "What exactly are you trying to say to me right now?",
    "This is undoubtedly the best gift I have ever received in my life.",
    "Get out of my sight immediately and do not come back.",
    "I forgot to bring my umbrella and now I'm completely soaked.",
    "Let's go out and celebrate our victory with a delicious dinner tonight.",
    "I simply do not understand why this error keeps happening over and over."
]

output_dir = "emotion_pairs_output"
os.makedirs(output_dir, exist_ok=True)

# 3. 단일 생성 함수 (반복문 제거, 단일 파일 생성에 집중)
def generate_single_speech(text, emotion_instruction, filename):
    print(f"Generating: {filename}...")
    
    # CosyVoice2 Inference
    output = cosyvoice.inference_instruct2(
        tts_text=text,
        instruct_text=emotion_instruction,
        prompt_speech_16k=prompt_speech_16k,
    )

    # 결과 저장 (generator에서 첫 번째 결과만 가져옴)
    for res in output:
        audio_tensor = res['tts_speech']
        sample_rate = cosyvoice.sample_rate
        
        save_path = os.path.join(output_dir, filename)
        torchaudio.save(save_path, audio_tensor, sample_rate)
        print(f"Saved: {save_path}")
        break # 한 번만 저장하고 루프 탈출

# 4. 실행: 문장 리스트를 순회하며 Happy/Angry 쌍 생성
print(f"총 {len(sentences)}개의 문장에 대해 생성을 시작합니다.\n")

# 감정 프롬프트 정의
instruct_happy = "Speak with a very happy, excited, and cheerful tone."
instruct_angry = "Speak with a very huge angry tone."

for idx, text in enumerate(sentences, start=1):
    # 파일명에 인덱스 포함 (예: 01_happy.wav)
    prefix = f"{idx:02d}" 
    
    print(f"--- Processing Sentence {prefix} ---")
    print(f"Text: {text}")

    # Happy 버전 생성
    generate_single_speech(
        text=text,
        emotion_instruction=instruct_happy,
        filename=f"{prefix}_happy.wav"
    )

    # Angry 버전 생성
    generate_single_speech(
        text=text,
        emotion_instruction=instruct_angry,
        filename=f"{prefix}_angry.wav"
    )

print("\n모든 생성 작업이 완료되었습니다.")
