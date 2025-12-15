import streamlit as st
import whisper
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, pipeline
from datetime import datetime
import sqlite3
import re
import time

DB_PATH = "medical_chat_jinseong_final.db"

# ------------------------------- DB 초기화 -------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS consultations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            timestamp TEXT,
            stt_text TEXT,
            summary TEXT,
            feedback TEXT
        )
    """)
    conn.commit()
    conn.close()

def get_patient_history(patient_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, summary, feedback FROM consultations WHERE patient_id=? ORDER BY timestamp DESC", (patient_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows

# ------------------------------- Whisper 로드 -------------------------------
@st.cache_resource
def load_whisper():
    return whisper.load_model("large")

# ------------------------------- KoBART 요약 로드 -------------------------------
@st.cache_resource
def load_kobart():
    tokenizer = PreTrainedTokenizerFast.from_pretrained("EbanLee/kobart-summary-v3")
    model = BartForConditionalGeneration.from_pretrained("EbanLee/kobart-summary-v3")
    return tokenizer, model

def summarize_text_kobart_core(tokenizer, model, input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding="max_length",
                       truncation=True, max_length=1024)
    summary_ids = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        max_length=250,
        min_length=10,
        num_beams=5,
        repetition_penalty=1.4,
        no_repeat_ngram_size=2
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # 의료 키워드 기반 핵심 문장 강조
    medical_keywords = list(set([
    # 증상
    "기침", "가래", "두통", "복통", "구토", "설사", "피로", "어지럼", "통증", "열", "오한",
    "근육통", "체중감소", "식욕저하", "불면", "소화불량", "갈증", "피로감", "기력저하",
    "피부 발진", "가려움", "부종", "출혈", "체온 상승", "오한 지속", "피로 누적",
    "무기력", "전신 권태", "몸살", "몸이 떨림", "식은땀", "밤에 심해짐", "최근 심해짐",
    "점점 심해짐", "갑자기 발생", "지속적 통증", "간헐적 통증", "심해졌다", "완화되지 않는다",
    "증상 악화", "호전 없음", "호전과 악화 반복", "증상 지속", "최근 악화됨",

    # 호흡기 증상
    "기침", "가래", "호흡곤란", "숨참", "숨이 차다", "쌕쌕거림", "가슴 답답함",
    "흉통", "호흡 시 통증", "기침 악화", "밤에 기침", "가래 끈적임", "혈담", "가래 색 변화",
    "콧물", "코막힘", "재채기", "인후통", "목 통증", "목 따가움", "목소리 변함", "쉰 목소리",
    "삼키기 어렵다", "말하기 어렵다", "호흡곤란 심해짐", "천명음", "가슴 통증", "숨이 가빠짐",

    # 소화기 증상
    "복통", "명치 통증", "속쓰림", "속 울렁거림", "복부팽만", "트림", "복부 불쾌감",
    "구역", "구토", "구토 반복", "설사", "변비", "혈변", "흑변", "변 색 변화",
    "식후 통증", "식사 후 불편", "소화 지연", "속 더부룩함", "위통", "명치 쓰림",
    "복부 통증 간헐적", "설사 지속", "식욕 감소", "복부 팽창감", "가스 참", "복부 압통",
    "토혈", "구취", "역류", "속 울렁거림 심함",

    # 순환기 / 심혈관 증상
    "가슴 두근거림", "심계항진", "흉통", "가슴 통증", "호흡곤란 동반", "맥박 불규칙",
    "혈압 상승", "혈압 저하", "부정맥", "가슴 압박감", "가슴 답답함", "운동 후 흉통",
    "숨참과 흉통 동반", "식은땀 동반", "현기증", "실신", "어지럼증 심함", "심박수 증가",
    "저혈압", "빈맥", "서맥", "청색증", "산소포화도 감소", "손발 저림 증가",

    # 신경 / 정신 증상
    "두통", "어지럼증", "현기증", "시야 흐림", "청력 저하", "시야흐림", "손발 저림",
    "감각 저하", "얼굴 저림", "사지 마비", "경련", "발작", "실신", "불안감", "우울감",
    "집중력 저하", "기억력 저하", "수면 장애", "불면증", "불안 증폭", "짜증 증가",
    "두근거림 심해짐", "현기증 악화", "눈앞이 깜깜함", "귀 울림", "어깨 뻐근함",
    "편측 마비", "언어 장애", "말이 어눌함", "삼키기 어려움", "의식 저하", "기면",

    # 비뇨기 / 생식기 증상
    "야간뇨", "빈뇨", "배뇨통", "혈뇨", "소변량 감소", "소변량 증가", "잔뇨감",
    "급뇨", "요실금", "야간발열", "배뇨 불편감", "소변 색 변화", "요로 통증",
    "하복부 통증", "요통", "허리 통증", "골반 통증", "사타구니 통증",
    "월경통", "생리 불순", "질 분비물 증가", "질 가려움", "질 건조감",
    "배란통", "하복부 팽만", "배뇨 시 작열감",

    #  근골격계 증상
    "관절통", "근육통", "허리 통증", "목 통증", "어깨 통증", "무릎 통증", "팔 저림",
    "다리 저림", "손 저림", "근력 저하", "관절 부종", "뻣뻣함", "움직일 때 통증",
    "운동 후 통증", "운동 후 피로", "통증 악화", "통증 완화 없음",
    "근육 경련", "쥐 남", "근육 뭉침", "근육 약화", "관절 뻐근함",

    # 감각기관 증상
    "시야 흐림", "청력 감소", "청력저하", "귀 통증", "이명", "귀 먹먹함",
    "눈 피로", "눈 시림", "눈 통증", "눈 충혈", "눈물 과다", "시야 흐려짐",
    "시력 저하", "복시", "눈 가려움", "눈 건조", "입마름", "구강건조",
    "코막힘", "콧물", "코 가려움", "코피", "인후통", "목 이물감", "삼킴 곤란",

    # 피부 증상
    "피부 발진", "가려움", "붉은 반점", "부종", "두드러기", "피부 건조", "피부 벗겨짐",
    "피부 통증", "피부 따가움", "물집", "피부 색 변화", "멍", "출혈반", "발적",
    "피부 열감", "피부 궤양", "피부 트러블", "피부염", "각질", "피부 갈라짐",

    # 대사 / 내분비 증상
    "갈증 증가", "다뇨", "체중 감소", "체중 증가", "식욕 증가", "식욕 저하",
    "피로 누적", "무기력", "손떨림", "땀 과다", "열감", "추위 민감",
    "부종", "탈모", "피부 건조", "손발 차가움", "불면", "기분 변화",
    "월경 불순", "목 부종", "갑상선 비대", "체중 변동 심함",

    # 기타 표현
    "증상 악화", "증상 지속", "호전 없음", "간헐적 발작", "급격한 통증", "완화되지 않음",
    "점점 심해짐", "밤에 심해짐", "식후 악화", "운동 후 악화", "체온 상승",
    "식사 후 통증", "운동 후 호흡곤란", "야간 통증", "기상 시 두통",
    "수면 중 발작", "아침에 심함", "저녁에 심함", "갑작스런 어지럼", "피로 누적"

    # 질환
    "감기","독감","폐렴","천식","당뇨","고혈압","심근경색","협심증","뇌졸중","위염","장염","간염","신부전","치매","우울증",
    "불안장애","조현병","피부염","아토피","비염","골다공증","결핵","통풍","갑상선","간경화","중이염","이석증","빈혈",
    "알레르기","편도염","방광염","기관지염","위궤양","신장결석","심장부전","관절염","류마티스","편두통","수면무호흡",
    "심부전","심장질환","간질환","호흡기질환","소화기질환","신경계질환","정신질환","피부질환","요로감염","골절","탈골",
    "근육염","간손상","신손상","심장판막질환","심근질환","치주염","백내장","녹내장","중이염","편도비대","갑상선기능저하",
    "갑상선기능항진","골절 후 통증","심혈관질환","폐질환","뇌질환","신경통","류마티스관절염","척추질환","척추협착증","허리디스크",
    "요통","좌골신경통","대장염","위궤양","간경변","만성신부전","급성신부전","심방세동","심실빈맥","심정지","간암","위암","대장암",
    "폐암","유방암","전립선암","췌장암","신장암","갑상선암","백혈병","림프종","악성종양","양성종양","만성피로증후군","자가면역질환",

    # 신체 부위
    "머리","목","어깨","가슴","심장","폐","위","간","장","신장","팔","다리","허리","무릎","손목","발목","눈","귀","코",
    "입","치아","피부","손","발","발바닥","손가락","발가락","복부","옆구리","엉덩이","등","머리카락","손톱","발톱",
    "발등","발뒤꿈치","손등","손바닥","눈꺼풀","눈동자","안구","입술","혀","인후","후두","기관지","폐포","심장판막",
    "대퇴부","하퇴부","경부","흉부","복부","요추","천추","골반","사지","사지말단","관절","척추","척수","뇌","뇌간","소뇌",

    # 검사
    "검사","혈액검사","소변검사","CT","MRI","X선","초음파","심전도","내시경","혈압","혈당","체온","맥박","산소포화도",
    "진단","결과","수치","이상","정상","재검","추적관찰","영상검사","심장검사","호흡검사","간기능검사","신장기능검사",
    "호르몬검사","알레르기검사","암표지자검사","초음파검사","심장초음파","심장CT","뇌CT","뇌MRI","혈액응고검사","소변검사결과",

    # 치료
    "수술","치료","재활","물리치료","약물치료","주사","복용","처방","투약","입원","퇴원","통원","예방접종","백신",
    "항생제","진통제","해열제","호르몬제","응급처치","소독","봉합","마취","부작용","치료중","약 복용 중","증상 완화",
    "회복 중","치료 계획","추가 치료 필요","약물 부작용","재발 위험","통증 조절","항암치료","면역치료","물리재활",
    "통증 관리","영양치료","심리치료","호흡치료","재발 방지","수술 후 관리","투약 계획","재활치료","추적관찰 필요","임상시험",

    # 의료행위
    "진단서","소견서","처방전","진료기록","차트","진료비","보험","상담","진찰","수납","예약","대기","접수","검진",
    "상담 완료","결과 통보","수술 전","수술 후","추가검사 권고","재검 권고","치료 계획 수립","진료 연장","진료기록 입력","상담 기록",
    "검사 예약","결과 확인","추가 검사 요청","보험 청구","보험 승인","진료비 계산","입원 기록","퇴원 기록","통원 기록","진료 기록 열람",

    # 인물/역할
    "의사","간호사","약사","환자","보호자","선생님","검사결과","증상호전","증상악화","경과","추가검사","담당의","전문의",
    "간호조무사","주치의","상담사","치료팀","응급실 의사","병동 간호사","약사 상담","재활치료사","영상의학과 의사","마취통증의학과 의사",

    # 생활습관
    "운동", "식습관", "식이", "금연", "절주", "스트레스", "수면", "영양", "체중조절", "건강검진",
    "생활습관", "수분섭취", "카페인", "밤샘", "야식", "과식", "불규칙 수면", "운동 부족",
    "체중 증가", "체중 감소", "흡연", "음주", "카페인 과다", "식사 조절", "물 섭취 부족",
    "수면 패턴", "스트레스 증가", "운동 권장", "영양 보충", "체중 관리", "건강 생활",
    "규칙적 생활", "카페인 제한", "음주 제한", "금연 권장",
    "운동량 부족", "신체 활동 감소", "규칙적인 운동", "유산소 운동", "근력 운동",
    "스트레스 관리", "수면 위생", "불면", "수면 장애", "만성 피로",
    "영양 불균형", "단백질 섭취", "지방 섭취", "탄수화물 조절", "염분 섭취 제한",
    "칼로리 과다", "비만", "복부 비만", "체질량지수", "BMI 증가", "BMI 감소",
    "식이요법", "저염식", "저지방식", "고단백식", "균형 잡힌 식단",
    "폭식", "편식", "불규칙한 식사", "식사 거름", "간식 섭취",
    "수분 부족", "탈수", "카페인 섭취량", "에너지 음료 섭취",
    "음주 습관", "과음", "폭음", "알코올 의존", "알코올 섭취량",
    "흡연 습관", "전자담배", "니코틴 의존", "금연 시도", "금연 실패",
    "생활습관 개선", "건강한 식습관", "운동 습관 형성", "스트레스 완화", "명상", "이완 요법",
    "수면 개선", "수면 습관 교정", "취침 시간", "기상 시간", "수면의 질",
    "정신적 피로", "우울감", "불안", "스트레스성 폭식", "스트레스성 두통",
    "체중 감량", "체중 유지", "체중 증가 경향", "대사 증후군",
    "고혈압 예방", "당뇨 관리", "혈당 조절", "콜레스테롤 관리",
    "운동 처방", "생활습관 교정", "건강 상담", "건강 교육", "자기관리",
    "정기 검진", "건강검진 권장", "생활습관 평가", "건강위험요인",
    "생활습관 병력", "운동 부족 관련 피로", "식습관 개선 필요",
    "영양 상담", "운동 상담", "스트레스 상담", "수면 상담",
    "습관성 카페인 섭취", "습관성 음주", "습관성 흡연",
    "생활습관 관련 질환", "비만 관련 질환", "대사성 질환",
    "건강행동", "생활습관 질 개선", "행동 변화 유도",
    "규칙적인 식사", "규칙적인 수면", "하루 8시간 수면",
    "자기 전 스마트폰 사용", "수면 위생 교육", "신체 활동량 증가",
    "걷기 운동", "스트레칭", "운동 프로그램 참여", "운동 일지 기록",
    "체중 측정", "식사 일지", "건강 기록", "생활습관 점검",
    "스트레스 해소", "휴식 부족", "과로", "야간 근무", "수면 박탈",
    "음식 섭취 조절", "폭식 방지", "수분 보충", "카페인 섭취 제한",
    "건강관리 지도", "생활습관 코칭", "환자 교육", "자기 관리 능력 향상",
    "건강한 생활 유지", "생활습관 개선 목표", "생활습관 교정 계획"
    # 진료과 키워드
    "내과","외과","정형외과","신경과","정신건강의학과","피부과","산부인과","소아과","안과","이비인후과","비뇨기과","응급의학과",
    "치과","한방과","재활의학과","영상의학과","마취통증의학과","신장내과","심장내과","호흡기내과","소화기내과","내분비내과",
    "류마티스내과","혈액내과","신경외과","심장외과","간담췌외과","정형외과 재활","척추외과","소아청소년과","산부인과 내분비",
    "피부미용과","안과 시력검사","이비인후과 청력검사","비뇨기과 요로검사","응급실","중환자실","외래진료","입원실","수술실",
    
    # 의료기기/측정
    "혈압기","청진기","주사기","의료기록지","체온계","혈당계","수액","산소호흡기","환자모니터","심전도기","인공호흡기","검사장비",
    "산소포화도 측정기","혈압 측정기","혈당 측정기","체중계","심장 모니터","혈액검사 장비","초음파기","CT 스캐너","MRI 장비"
    ]))
    sentences = re.split(r'(?<=[.!?])\s+', summary)
    key_sentences = [s for s in sentences if any(k in s for k in medical_keywords)]
    if not key_sentences:
        key_sentences = sentences[:2]
    final_summary = ' '.join(dict.fromkeys(key_sentences))
    return final_summary.strip()

# ------------------------------- KoAlpaca 피드백 로드 -------------------------------
@st.cache_resource
def load_koalpaca():
    MODEL_NAME = 'beomi/KoAlpaca-Polyglot-5.8B'
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        do_sample=True,
        top_k=50,
        temperature=0.7,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id
    )
    return pipe

def generate_feedback(pipe, stt_text, summary_text):
    prompt = f"""
당신은 전문 의료 상담사입니다. 아래는 환자의 상담 내용과 요약입니다.
환자의 증상 중심 정보만 반영하고, 측정방법/행동 등은 무시합니다.
이번 상담에서 나타난 주요 증상을 중심으로 안전한 주의사항과 향후 관리 및 치료 권장사항만 작성하세요.

📜 환자 상담 원문:
{stt_text}

🩺 요약문:
{summary_text}
"""
    feedback = pipe(prompt)[0]['generated_text']
    return feedback


# ------------------------------- Streamlit UI -------------------------------
def main():
    st.set_page_config(page_title="의료 음성 요약 시스템", page_icon="🏥", layout="wide")
    st.title("🏥 의료 음성 요약 및 피드백 시스템")
    st.caption("음성 → STT → 요약 → 행동/의료 피드백")

    init_db()

    patient_id = st.text_input("🧍 환자 ID 입력", "patient001")
    audio_file = st.file_uploader("🎙 음성 파일 업로드 (wav/mp3/m4a)", type=["wav", "mp3", "m4a"])

    tokenizer_kobart, model_kobart = load_kobart()
    pipe_koalpaca = load_koalpaca()

    if patient_id:
        history = get_patient_history(patient_id)
        if history:
            st.markdown("### 📜 과거 상담 기록")
            for ts, summary, feedback in history:
                with st.expander(f"🕒 {ts}"):
                    st.markdown(f"**요약:** {summary}")
                    st.markdown(f"**피드백:** {feedback}")
        else:
            st.info("과거 기록이 없습니다.")

    if st.button("🚀 STT 변환 및 요약/피드백 생성"):
        if audio_file:
            progress = st.progress(0)
            status_text = st.empty()

            # 1️⃣ STT
            status_text.text("🎧 음성 인식 중...")
            model_whisper = load_whisper()
            temp_path = "temp_audio.wav"
            with open(temp_path, "wb") as f:
                f.write(audio_file.read())
            result = model_whisper.transcribe(temp_path)
            stt_text = result["text"]
            progress.progress(33)
            time.sleep(0.5)

            # 2️⃣ 요약
            status_text.text("🧠 요약문 생성 중...")
            summary_text = summarize_text_kobart_core(tokenizer_kobart, model_kobart, stt_text)
            progress.progress(66)
            time.sleep(0.5)

            # 3️⃣ 피드백
            status_text.text("💬 피드백 생성 중...")
            feedback_text = generate_feedback(pipe_koalpaca, stt_text, summary_text)
            progress.progress(100)
            status_text.text("✅ 완료!")

            # DB 저장
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO consultations (patient_id, timestamp, stt_text, summary, feedback) VALUES (?,?,?,?,?)",
                (patient_id, datetime.now().isoformat(), stt_text, summary_text, feedback_text)
            )
            conn.commit()
            conn.close()

            # 출력
            st.markdown("### 🎧 STT 결과")
            st.write(stt_text)
            st.markdown("### 🩺 요약문")
            st.write(summary_text)
            st.markdown("### 💡 피드백")
            st.write(feedback_text)

        else:
            st.warning("먼저 음성 파일을 업로드하세요.")

if __name__ == "__main__":
    main()
