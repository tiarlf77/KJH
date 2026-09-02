import json
import os
import re
import calendar
from datetime import date
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TypedDict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import unquote
from langgraph.graph import END, START, StateGraph


BASE_DIR = Path(__file__).resolve().parent
RULES_DIR = BASE_DIR / "규정"
MODEL = "gpt-5.6-luna"
SOURCE_FILES = {
    "경조금 지급기준.txt",
    "동호회 관리 규정.txt",
    "숙소지원금 운영 기준.txt",
    "여비관리기준.txt",
}
QUERY_SYNONYMS = {
    "동생": {"형제", "자매", "형제자매"},
    "형": {"형제", "형제자매"},
    "누나": {"형제", "자매", "형제자매"},
    "언니": {"형제", "자매", "형제자매"},
    "오빠": {"형제", "형제자매"},
    "장인어른": {"배우자", "부모"},
    "장모님": {"배우자", "부모"},
    "시어머니": {"배우자", "부모"},
    "시아버지": {"배우자", "부모"},
    "처제": {"배우자", "형제", "자매"},
    "처형": {"배우자", "형제", "자매"},
    "처남": {"배우자", "형제", "자매"},
    "시누이": {"배우자", "형제", "자매"},
    "시동생": {"배우자", "형제", "자매"},
    "아주버님": {"배우자", "형제", "자매"},
    "도련님": {"배우자", "형제", "자매"},
    "와이프": {"배우자"},
    "아내": {"배우자"},
    "남편": {"배우자"},
    "부인": {"배우자"},
}
OWN_SIBLINGS = ("형", "누나", "언니", "오빠", "남동생", "여동생", "동생", "형제", "자매")
SPOUSE_SIBLINGS = ("처제", "처형", "처남", "시누이", "시동생", "아주버님", "도련님")
SPOUSE_CUES = ("배우자", "와이프", "아내", "남편", "부인", "wife", "husband")


def load_env():
    """.env의 값을 읽어 환경변수에 없는 값만 설정합니다."""
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"\''))


def tokens(text):
    """한글·영문·숫자 단위로 검색 토큰을 만듭니다."""
    return set(re.findall(r"[가-힣A-Za-z0-9]+", text.lower()))


def retrieve(question, limit=12):
    """네 개 원문 규정에서 질문과 관련된 문단을 찾아 근거로 반환합니다."""
    query_tokens = tokens(question)
    if any(word in question for word in ("숙소", "숙소지원금", "기존 숙소", "전 근무지", "반납", "정리", "유지")):
        # 근무지 이동 관련 질문은 5.5 지원특례의 핵심 표현을 함께 검색합니다.
        query_tokens.update({"전근무지", "숙소정리", "3개월", "최장", "6개월", "처분", "발령"})
    for word, synonyms in QUERY_SYNONYMS.items():
        if word in question:
            query_tokens.update(synonyms)
    results = []
    for path in sorted(RULES_DIR.glob("*.txt")):
        # 공식 기준 4개 파일만 상담 근거로 사용하고 샘플 문서는 제외합니다.
        if path.name not in SOURCE_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        for paragraph in paragraphs:
            paragraph_tokens = tokens(paragraph)
            score = len(query_tokens & paragraph_tokens)
            if score:
                results.append({"file": path.name, "score": score, "text": paragraph[:3000]})
    results.sort(key=lambda item: item["score"], reverse=True)
    if not results:
        return []
    # 숙소지원금 질문에는 출장·여비 규정이 섞이지 않도록 전용 기준만 사용합니다.
    if any(word in question for word in ("숙소", "숙소지원금", "주거", "월세", "전세")):
        results = [item for item in results if item["file"] == "숙소지원금 운영 기준.txt"]
        if not results:
            return []
    # 최고 점수를 받은 규정 파일만 선택해 다른 제도 설명이 섞이지 않게 합니다.
    top_score = results[0]["score"]
    top_files = {item["file"] for item in results if item["score"] == top_score}
    results = [item for item in results if item["file"] in top_files]
    selected = []
    # 여러 규정이 함께 적용될 수 있으므로 규정별 상위 근거를 먼저 확보합니다.
    for path in sorted({item["file"] for item in results}):
        selected.extend([item for item in results if item["file"] == path][:3])
    selected.sort(key=lambda item: item["score"], reverse=True)
    return selected[:limit]


def add_months(value, months):
    """월말 날짜가 존재하지 않는 경우 해당 월의 마지막 날로 보정합니다."""
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def extract_birth_date(text):
    """숫자형·하이픈형·한글형 생년월일을 날짜로 변환합니다."""
    patterns = (
        r"(?<!\d)(19\d{2})(\d{2})(\d{2})(?!\d)",
        r"(?<!\d)(19\d{2})[-./](\d{1,2})[-./](\d{1,2})(?!\d)",
        r"(?<!\d)(19\d{2})년\s*(\d{1,2})월\s*(\d{1,2})일",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            except ValueError:
                return None
    return None


def date_deadline_context(question):
    """질문에 사유 발생일이 있으면 경조금 3개월 마감일을 계산합니다."""
    match = re.search(r"(20\d{2})[.\-/년](\d{1,2})[.\-/월](\d{1,2})일?", question)
    if not match or not any(word in question for word in ("경조", "회갑", "결혼", "출산", "사망", "신청")):
        return ""
    try:
        occurred = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        deadline = add_months(occurred, 3)
    except ValueError:
        return ""
    return (
        f"\n[계산된 신청기한]\n사유 발생일 {occurred.isoformat()} 기준 3개월 이내 신청 마감일은 "
        f"{deadline.isoformat()}이다. 사유 발생일의 일자가 다음 달에 없으면 월말로 보정한다. "
        "마감일 당일까지 신청 가능하다고 안내한다."
    )


def birthday_context(question):
    """회갑 질문의 생년월일을 계산해 모델이 연령을 추측하지 않게 합니다."""
    if "회갑" not in question:
        return ""
    birth = extract_birth_date(question)
    if not birth:
        return ""
    try:
        today = date.today()
        sixtieth = date(birth.year + 60, birth.month, birth.day)
    except ValueError:
        return ""
    deadline = add_months(sixtieth, 3)
    if sixtieth > today:
        status = f"회갑 사유 발생일은 {sixtieth.isoformat()}이며 현재 기준일 이후이다. 아직 신청할 수 없다."
    elif today <= deadline:
        status = (
            f"회갑 사유 발생일은 {sixtieth.isoformat()}이고 신청 마감일은 {deadline.isoformat()}이다. "
            "현재 기준일은 사유 발생일로부터 3개월 이내이므로 경조금 신청이 가능하다."
        )
    else:
        status = (
            f"회갑 사유 발생일은 {sixtieth.isoformat()}이고 신청 마감일은 {deadline.isoformat()}이다. "
            "현재 기준일은 신청 마감일 이후이므로 청구권이 소멸되어 신청할 수 없다."
        )
    return (
        f"\n[회갑 생년월일 계산 결과]\n생년월일: {birth.isoformat()} / 기준일: {today.isoformat()} / {status} "
        "이 계산 결과를 답변에 반영하고, 대상 관계와 회갑 연령을 구분해 설명한다."
    )


def conversation_context(question, history):
    """후속 답변의 관계와 생년월일을 직전 대화에서 보완합니다."""
    prior_text = " ".join(item.get("content", "") for item in (history or [])[-6:])
    relation = ""
    if any(word in question for word in ("우리 엄마", "우리 어머니", "우리 아버지", "우리 부모님", "우리 부모")):
        relation = "우리 엄마·아버지·부모님은 별도 배우자 표현이 없으므로 본인 부모로 해석한다."
    elif any(word in question for word in ("배우자 어머니", "배우자 엄마", "배우자 아버지", "장모님", "장인어른", "시어머니", "시아버지")):
        relation = "질문 대상은 배우자의 부모로 해석한다."
    birth = extract_birth_date(question + " " + prior_text)
    birthday = ""
    if birth and ("회갑" in question or "회갑" in prior_text):
        birthday = birthday_context(f"회갑 {birth.strftime('%Y%m%d')}")
    if not relation and not birthday:
        return ""
    return f"\n[대화 맥락 보완]\n{relation}\n{birthday}".strip()


def identify_parent_relation(text):
    """회갑 검토에 필요한 부모 관계를 정해진 표현으로 분류합니다."""
    if any(word in text for word in ("배우자 어머니", "배우자 엄마", "배우자 아버지", "장모님", "장인어른", "시어머니", "시아버지")):
        return "배우자 부모"
    if any(word in text for word in ("우리 엄마", "우리 어머니", "우리 아버지", "우리 부모님", "엄마", "어머니", "아빠", "아버지", "부모님")):
        return "본인 부모"
    return ""


def build_hoegap_answer(question, history):
    """관계와 생년월일이 확인된 회갑 문의에는 일관된 검토 양식을 반환합니다."""
    prior_text = " ".join(item.get("content", "") for item in (history or [])[-6:])
    # 새 질문에 값이 있으면 반드시 이전 대화보다 우선합니다.
    current_relation = identify_parent_relation(question)
    current_birth = extract_birth_date(question)
    relation = current_relation or identify_parent_relation(prior_text)
    birth = current_birth or extract_birth_date(prior_text)
    # 이전 회갑 대화는 생년월일·부모 관계처럼 명확한 후속 입력일 때만 이어받습니다.
    is_hoegap = (
        "회갑" in question
        or ("경조금" in question and current_relation and current_birth)
        or ("회갑" in prior_text and (current_relation or current_birth))
    )
    if not relation or not birth or not is_hoegap:
        return ""
    today = date.today()
    sixtieth = date(birth.year + 60, birth.month, birth.day)
    deadline = add_months(sixtieth, 3)
    if today < sixtieth:
        verdict = "회갑 사유 발생일 전"
        support_line = ""
        summary = f"회갑 사유 발생일인 {sixtieth.isoformat()}부터 신청 여부를 확인할 수 있습니다."
    elif today <= deadline:
        verdict = "신청 가능"
        support_line = "- 지원금: 200,000원\n"
        summary = "현재 사유 발생일로부터 3개월 이내이므로 경조금 신청이 가능합니다."
    else:
        verdict = "신청 불가"
        support_line = ""
        summary = "신청 마감일이 지나 청구권이 소멸되어 경조금 신청이 불가능합니다."
    return (
        "확인 결과\n"
        f"- 관계: {relation}\n"
        f"- 생년월일: {birth.isoformat()}\n"
        f"- 회갑 사유 발생일: {sixtieth.isoformat()}\n"
        f"- 신청 마감일: {deadline.isoformat()}\n"
        f"- 현재 기준일: {today.isoformat()}\n"
        f"- 판정: {verdict}\n"
        f"{support_line}\n"
        f"{summary}\n최종 승인·지급은 담당 부서의 서류 검토를 거쳐 결정됩니다."
    )


def build_sibling_marriage_answer(question):
    """형제자매 결혼 문의는 규정 기준으로 일관되게 안내합니다."""
    sibling_words = OWN_SIBLINGS + SPOUSE_SIBLINGS
    if "결혼" not in question or not any(word in question for word in sibling_words):
        return ""
    is_spouse_side = (
        any(word in question for word in SPOUSE_SIBLINGS)
        or (any(word in question.lower() for word in SPOUSE_CUES) and any(word in question for word in OWN_SIBLINGS))
    )
    if is_spouse_side:
        relation = "배우자 형제·자매"
        documents = "본인 가족관계증명서, 배우자 부모 기준 가족관계증명서, 청첩장"
    else:
        relation = "본인 형제·자매"
        documents = "부모 기준 가족관계증명서, 청첩장"
    return (
        f"네. {relation} 결혼은 경조금 지급 대상입니다.\n\n"
        "확인 결과\n"
        f"- 관계: {relation}\n"
        "- 지원금: 200,000원\n"
        f"- 필요 서류: {documents}\n"
        "- 신청기한: 경조사 사유 발생일로부터 3개월 이내\n\n"
        "최종 승인·지급은 담당 부서의 서류 검토를 거쳐 결정됩니다."
    )


def build_seungjungsang_answer(question):
    """승중상 문의는 규정에 명시된 금액과 서류를 고정해 안내합니다."""
    if "승중상" not in question:
        return ""
    return (
        "경조금 지급 대상입니다.\n\n"
        "확인 결과\n"
        "- 관계: 승중상\n"
        "- 판정: 지원 대상\n"
        "- 지원금: 500,000원\n"
        "- 필요 서류: 기본증명서(상세, 사망일 표기 확인), 가족관계증명서, 부고장\n\n"
        "○ 회사 경조 담당 업체(경조물품,화환 등)\n"
        "- 현진시닝 : 1600-0113(24시간)\n\n"
        "최종 승인·지급은 담당 부서의 서류 검토를 거쳐 결정됩니다."
    )


def build_death_answer(question):
    """사망 경조금 문의를 관계별로 판정해 불필요한 반복 없이 안내합니다."""
    if not any(word in question for word in ("돌아가", "사망", "별세", "상") ):
        return ""
    asks_items = any(word in question for word in ("물품", "화환", "장례용품", "조화"))
    relation_amount = (
        (("시아버지", "시어머니", "장인어른", "장모님", "배우자 부모"), "배우자 부모", "1,000,000원"),
        (("남편의 아버지", "남편 아버지", "아내의 아버지", "아내 아버지", "와이프 아버지", "배우자의 아버지", "배우자 아버지"), "배우자 부모", "1,000,000원"),
        (("남편의 어머니", "남편 어머니", "아내의 어머니", "아내 어머니", "와이프 어머니", "배우자의 어머니", "배우자 어머니"), "배우자 부모", "1,000,000원"),
        (("외할아버지", "외할아버님", "외할머니", "외할매", "외조부모"), "본인 외조부모", "300,000원"),
        (("아버지", "어머니", "엄마", "아빠", "본인 부모"), "본인 부모", "1,000,000원"),
        (("자녀", "아들", "딸"), "자녀", "1,000,000원"),
        (("조부모", "할아버지", "할머니"), "본인 및 배우자 조부모", "300,000원"),
        (("형제", "자매", "오빠", "언니", "누나", "형", "동생", "처남", "처제", "처형", "시누이", "시동생"), "본인 및 배우자 형제·자매", "300,000원"),
        (("배우자", "아내", "남편", "와이프"), "배우자", "2,000,000원"),
        (("본인",), "본인", "5,000,000원"),
    )
    for words, relation, amount in relation_amount:
        if any(word in question for word in words):
            opening = (
                f"네. {relation} 사망 시 경조 지원 물품을 신청할 수 있습니다."
                if asks_items else "경조금 지급 대상입니다."
            )
            item_line = ""
            # 사망 경조금 답변에는 질문 표현과 관계없이 규정의 물품 지급 여부를 표시합니다.
            wreath = "X" if relation == "본인 및 배우자 형제·자매" else "O"
            supplies = "O" if relation in ("본인", "배우자", "본인 부모", "배우자 부모", "자녀") else "X"
            item_line = f"- 화환: {wreath}\n- 장례용품: {supplies}\n"
            documents = "기본증명서(상세, 사망일 표기 확인), 가족관계증명서, 부고장"
            if relation == "본인 외조부모":
                documents = "기본증명서(상세, 사망일 표기 확인), 어머니 기준 가족관계증명서, 부고장"
            return (
                f"{opening}\n\n"
                "확인 결과\n"
                f"- 관계: {relation}\n"
                "- 판정: 지원 대상\n"
                f"- 지원금: {amount}\n"
                f"{item_line}"
                f"- 필요 서류: {documents}\n\n"
                "○ 회사 경조 담당 업체(경조물품,화환 등)\n"
                "- 현진시닝 : 1600-0113(24시간)\n\n"
                "최종 승인·지급은 담당 부서의 서류 검토를 거쳐 결정됩니다."
            )
    return ""


def starts_new_policy_topic(question):
    """이전 대화와 분리해야 하는 새 복리후생 질문인지 판단합니다."""
    topic_words = ("결혼", "사망", "출산", "동호회", "숙소", "출장", "여비", "부임", "건강검진")
    return any(word in question for word in topic_words) and "회갑" not in question


def build_clarification_answer(question):
    """제도 유형을 알 수 없는 질문에 전체 상담 범위와 재질문 형식을 안내합니다."""
    topics = ("출장", "파견", "부임", "경조", "결혼", "회갑", "출산", "사망", "돌아가", "별세", "승중상", "숙소", "동호회")
    if any(word in question for word in topics):
        return ""
    return (
        "질문의 대상이나 제도 유형을 정확히 확인하기 어렵습니다.\n\n"
        "현재 상담 가능한 복리후생 항목은 다음과 같습니다.\n"
        "- 국내·외 출장 및 파견·부임\n"
        "- 경조금: 결혼, 회갑, 출산장려금, 사망\n"
        "- 숙소지원금\n"
        "- 동호회 지원\n\n"
        "정확한 안내를 위해 제도 유형, 대상 또는 상황, 확인하고 싶은 내용을 포함해 다시 질문해 주세요.\n"
        "예: 지원금, 지원 물품, 자격, 신청기한, 필요 서류, 신청 방법\n\n"
        "예시: ‘서울에서 포항으로 출장 갈 때 교통비가 지원되나요?’ 또는 ‘타지역으로 부임하면 숙소지원금을 받을 수 있나요?’"
    )


def build_housing_move_answer(question):
    """기존 숙소지원 중 타 지역 부임 문의를 지원특례와 신규 부임 기준으로 안내합니다."""
    if "숙소" not in question or not any(word in question for word in ("발령", "부임", "타지", "타 지역")):
        return ""
    destination = "서울" if "서울" in question else "서울 외"
    amount = "월 60만 원" if destination == "서울" else "월 40만 원"
    return (
        "질문하신 상황은 기존 숙소지원금 수급 중 근무지가 변경되는 경우입니다.\n\n"
        "확인 결과\n"
        "- 기존 숙소: 정리 기간 3개월간 숙소지원금 한도 내 실비 지원\n"
        "- 연장 기준: 부득이한 경우 처분 노력 입증자료 제출 시 1개월 단위 최장 6개월\n"
        "- 중복 여부: 전 근무지 숙소 정리 비용과 현 근무지 숙소지원금은 중복 가능\n"
        f"- 신규 부임지: {destination}\n"
        f"- 신규 부임 숙소지원금: {amount}, 발령일로부터 3년간\n"
        "- 산정 기준: 월세는 월 차임만 지원, 전세는 전세금 1,000만 원당 월 10만 원\n"
        "- 필요 서류: 기존 숙소 정리 비용 증빙, 처분 노력 입증자료(연장 시), 신규 숙소 임대차계약서\n\n"
        "최종 지원 여부와 서류 인정 범위는 담당 부서의 규정 검토를 거쳐 결정됩니다."
    )


def call_openai(question, evidence, history=None):
    """검색 근거를 포함해 Responses API를 호출합니다."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(".env에 OPENAI_API_KEY가 입력되지 않았습니다.")
    evidence_text = "\n\n".join(
        f"[근거 {i}] {item['file']}\n{item['text']}" for i, item in enumerate(evidence, 1)
    ) or "관련 규정 근거를 찾지 못했습니다."
    instructions = (
        "너는 사내 복리후생 규정 상담 Agent다. 네 개의 제공된 원문 규정을 기준 데이터로 사용하며, 반드시 제공된 근거만 사용해 한국어로 답한다. "
        "근거에 없는 금액·조건·사실은 추측하지 않는다. 질문 의도를 먼저 파악하고, "
        "가능 여부를 단정하기 어려우면 필요한 추가 정보를 질문한다. 답변은 자연스러운 대화체로 작성하되 부연 설명은 최소화한다. "
        "답변 마지막에는 '확인한 규정'과 파일명을 간단히 표시한다. 최종 승인·지급은 담당 부서 검토임을 안내한다. "
        "규정에 명시된 예외만 적용하고, 규정에 없는 예외나 담당자 재량은 사용자에게 확인 질문으로 남긴다. "
        f"오늘 기준일은 {date.today().isoformat()}이다. 경조금은 회갑 대상 여부와 신청기한을 별도로 판단한다. 경조금 기준의 청구권은 사유 발생일로부터 3개월 이내이며, "
        "계산된 신청 마감일이 제공되면 그 날짜를 사용한다. 규정의 회갑 대상은 본인 및 배우자 부모이고 지급액은 20만원이다."
        " 부모·배우자 부모의 회갑 질문에서 생년월일이 없으면 연령을 추측하거나 지원을 확정하지 않는다. "
        "답변은 다음 흐름으로 작성한다: '해당 부모가 본인 부모인지 배우자 부모인지 관계 요건은 확인되지만, 현재 정보만으로 실제 회갑 대상인지 판단할 수 없다'고 먼저 설명한다. "
        "그 다음 현재 기준일을 알려주고 생년월일을 YYYYMMDD 형식으로 요청한다. 지급액 20만원과 신청기한은 회갑 대상 판정 이후에 안내한다. "
        "단, 사용자가 '우리 엄마', '우리 아버지', '우리 부모님'이라고 표현하면 별도 배우자 표현이 없는 한 본인 부모로 이해하고 관계를 다시 묻지 않는다. "
        "직전 대화에서 생년월일과 관계가 이미 확인되었으면 같은 질문을 반복하지 말고 회갑일 계산 결과를 안내한다. "
        "사용자가 결혼·사망·출산·동호회·숙소·출장 등 다른 제도를 새로 질문하면 이전 회갑 대화와 분리해 현재 질문의 의도와 규정만 사용한다. "
        "현재 질문과 무관한 제도는 비교·설명·언급하지 않는다. 예를 들어 형제자매 결혼 질문에는 회갑, 숙소, 동호회, 여비를 언급하지 않는다. "
        "결혼·회갑·출산장려금·사망 등 경조금 답변은 반드시 공통 양식을 따른다. 첫 줄에는 해당 여부를 한 문장으로만 답한다. "
        "이후 빈 줄 뒤에 '확인 결과'를 쓰고, '- 관계:', '- 생년월일:'(확인된 경우만), '- 사유 발생일:'(확인된 경우만), "
        "'- 신청 마감일:'(계산 가능한 경우만), '- 현재 기준일:', '- 판정:', '- 지원금:', '- 필요 서류:' 순서로 필요한 항목만 한 번씩 작성한다. "
        "그 뒤 빈 줄 뒤에 신청 가능 여부 또는 추가로 필요한 정보만 한두 문장으로 쓰고, 마지막 문장은 '최종 승인·지급은 담당 부서의 서류 검토를 거쳐 결정됩니다.'로 끝낸다. "
        "같은 내용을 반복하거나 현재 질문과 무관한 제도를 언급하지 않는다. "
        "사망 문의에서는 첫 문장을 '경조금 지급 대상입니다.'처럼 간단히 작성하고 '해당됩니다'로 시작하지 않는다. "
        "사망일을 아직 받지 못한 경우 사망일로부터 3개월 이내 신청해야 한다는 문장과 사망일 요청 문장을 쓰지 않는다. "
        "사망 문의에서는 화환·조화 지원 여부를 별도 부연 문장으로 설명하지 않는다. "
        "사망 문의의 필요 서류 안내 뒤에는 반드시 다음 담당 업체 정보를 표시한다: '○ 회사 경조 담당 업체(경조물품,화환 등) - 현진시닝 : 1600-0113(24시간)'. "
        "숙소지원금 질문에서 근무지역 이동, 기존 숙소 유지·반납·정리가 언급되면 숙소지원금 운영 기준 5.5 지원특례를 반드시 확인한다. "
        "해당 규정은 전 근무지 숙소 정리 비용을 3개월간 숙소지원금 한도 내 실비로 지원하고, 부득이한 경우 처분 노력 입증자료 제출을 전제로 1개월 단위 최장 6개월까지 연장할 수 있다고 정한다. "
        "회갑일이 지났더라도 사유 발생일로부터 3개월 이내이면 신청 가능하다. "
        "계산 결과가 신청 마감일 이후이면 '확정하기 어렵다', '추가 확인 필요' 같은 유보 표현을 쓰지 않는다. "
        "이 경우에는 회갑일·신청 마감일·청구권 소멸을 2~3문장으로 간단히 안내한다."
    )
    conversation = []
    for item in (history or [])[-8:]:
        if item.get("role") in ("user", "assistant") and item.get("content"):
            conversation.append({"role": item["role"], "content": item["content"]})
    prior_text = " ".join(item.get("content", "") for item in (history or [])[-4:])
    birthday_question = question if "회갑" in question else (f"회갑 {question}" if "회갑" in prior_text else question)
    conversation.append({"role": "user", "content": f"{question}{date_deadline_context(question)}{birthday_context(birthday_question)}{conversation_context(question, history)}\n\n검색된 규정 근거:\n{evidence_text}"})
    payload = {
        "model": MODEL,
        # 규정 검색 결과를 요약하는 상담은 낮은 지연을 우선합니다.
        "reasoning": {"effort": "none"},
        "instructions": instructions,
        "input": conversation,
    }
    request = Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=45) as response:
        data = json.loads(response.read().decode("utf-8"))
    if data.get("output_text"):
        return data["output_text"]
    parts = []
    for output in data.get("output", []):
        for content in output.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "\n".join(parts).strip() or "응답 내용을 확인하지 못했습니다."


class ConsultationState(TypedDict, total=False):
    """LangGraph가 상담 단계 사이에서 전달하는 최소 상태입니다."""

    question: str
    history: list[dict]
    intent: str
    evidence: list[dict]
    answer: str


def classify_question_node(state: ConsultationState):
    """질문을 분류해 불명확 안내와 규정 검색의 흐름을 나눕니다."""
    question = state["question"]
    if build_clarification_answer(question):
        return {"intent": "clarification"}
    if "숙소" in question or "숙소지원금" in question:
        return {"intent": "housing"}
    if any(word in question for word in ("결혼", "회갑", "출산", "사망", "돌아가", "별세", "승중상")):
        return {"intent": "ceremony"}
    return {"intent": "policy"}


def choose_after_classification(state: ConsultationState):
    """불명확 질문은 검색 없이 안내하고, 나머지는 해당 규정을 검색합니다."""
    return "clarify" if state["intent"] == "clarification" else "retrieve"


def retrieve_policy_node(state: ConsultationState):
    """현재 제도 질문에 맞는 규정 근거를 검색합니다."""
    return {"evidence": retrieve(state["question"])}


def apply_policy_rules_node(state: ConsultationState):
    """날짜·관계·금액처럼 규정으로 결정 가능한 항목을 우선 처리합니다."""
    question = state["question"]
    history = [] if starts_new_policy_topic(question) else state.get("history", [])
    marriage_answer = build_sibling_marriage_answer(question)
    seungjungsang_answer = build_seungjungsang_answer(question)
    hoegap_answer = build_hoegap_answer(question, history)
    death_answer = build_death_answer(question)
    housing_move_answer = build_housing_move_answer(question)
    answer = marriage_answer or seungjungsang_answer or hoegap_answer or death_answer or housing_move_answer
    if not answer:
        return {}
    if marriage_answer or seungjungsang_answer or hoegap_answer or death_answer:
        evidence = [{"file": "경조금 지급기준.txt", "score": 1, "text": "경조금 지급기준"}]
    else:
        evidence = [{"file": "숙소지원금 운영 기준.txt", "score": 1, "text": "숙소지원금 운영 기준"}]
    return {"answer": answer, "evidence": evidence}


def choose_after_rules(state: ConsultationState):
    """결정 규칙으로 처리하지 못한 질문만 LLM 답변 단계로 보냅니다."""
    return "end" if state.get("answer") else "generate"


def generate_answer_node(state: ConsultationState):
    """규칙으로 확정할 수 없는 일반 문의를 근거 기반 LLM으로 답변합니다."""
    question = state["question"]
    history = [] if starts_new_policy_topic(question) else state.get("history", [])
    return {"answer": call_openai(question, state.get("evidence", []), history)}


def clarification_answer_node(state: ConsultationState):
    """제도 유형이 모호한 질문에는 전체 복리후생 범위를 안내합니다."""
    return {"answer": build_clarification_answer(state["question"]), "evidence": []}


def build_consultation_graph():
    """상담 요청을 분류·검색·규칙판정·생성으로 연결한 LangGraph를 만듭니다."""
    graph = StateGraph(ConsultationState)
    graph.add_node("classify", classify_question_node)
    graph.add_node("retrieve", retrieve_policy_node)
    graph.add_node("rules", apply_policy_rules_node)
    graph.add_node("generate", generate_answer_node)
    graph.add_node("clarify", clarification_answer_node)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", choose_after_classification, {"clarify": "clarify", "retrieve": "retrieve"})
    graph.add_edge("retrieve", "rules")
    graph.add_conditional_edges("rules", choose_after_rules, {"end": END, "generate": "generate"})
    graph.add_edge("generate", END)
    graph.add_edge("clarify", END)
    return graph.compile()


CONSULTATION_GRAPH = build_consultation_graph()


class Handler(SimpleHTTPRequestHandler):
    """정적 화면과 상담 요청을 함께 제공하는 간단한 HTTP 핸들러입니다."""

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            question = str(body.get("question", "")).strip()
            if not question:
                raise ValueError("질문을 입력해 주세요.")
            result = CONSULTATION_GRAPH.invoke({"question": question, "history": body.get("history", [])})
            self.respond(200, {"answer": result["answer"], "evidence": result.get("evidence", [])})
        except (ValueError, RuntimeError, HTTPError, URLError) as error:
            self.respond(400, {"error": str(error)})
        except Exception:
            self.respond(500, {"error": "상담 처리 중 오류가 발생했습니다."})

    def do_GET(self):
        """규정 원문 링크와 기존 정적 파일을 제공합니다."""
        if self.path.startswith("/rules/"):
            requested = unquote(self.path[len("/rules/"):]).split("?", 1)[0]
            candidate = (RULES_DIR / requested).resolve()
            if candidate.parent != RULES_DIR.resolve() or not candidate.is_file():
                self.send_error(404)
                return
            content = candidate.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Content-Disposition", "inline")
            self.end_headers()
            self.wfile.write(content)
            return
        super().do_GET()

    def respond(self, status, payload):
        """JSON 응답을 반환합니다."""
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


if __name__ == "__main__":
    load_env()
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("http://127.0.0.1:8000 에서 실행 중입니다.")
    server.serve_forever()
