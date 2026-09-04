import json
import os
import re
import calendar
from datetime import datetime
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
    "고모": {"부모", "형제", "자매"},
    "이모": {"외조부모", "부모", "형제", "자매"},
    "외삼촌": {"외조부모", "부모", "형제", "자매"},
    "큰아버지": {"조부모", "부모", "형제"},
    "작은아버지": {"조부모", "부모", "형제"},
    "백숙부": {"조부모", "부모", "형제"},
    "백숙부모": {"조부모", "부모", "형제"},
    "매형": {"형제", "자매"},
    "매제": {"형제", "자매"},
    "제부": {"형제", "자매"},
    "형부": {"형제", "자매"},
    "올케": {"형제", "자매"},
    "와이프": {"배우자"},
    "아내": {"배우자"},
    "남편": {"배우자"},
    "부인": {"배우자"},
}
OWN_SIBLINGS = ("형", "누나", "언니", "오빠", "남동생", "여동생", "동생", "형제", "자매")
SPOUSE_SIBLINGS = ("처제", "처형", "처남", "시누이", "시동생", "아주버님", "도련님")
SPOUSE_CUES = ("배우자", "와이프", "아내", "남편", "부인", "wife", "husband")
SPECIAL_LEAVE_RELATIONS = ("백숙부모", "백숙부", "매형", "매제", "제부", "형부", "올케")
AMBIGUOUS_DEATH_RELATIONS = ("고모", "이모", "외삼촌")
FAMILY_OWNER_WORDS = (
    "가족", "본인", "배우자", "와이프", "아내", "남편", "부모", "엄마", "아빠", "아버지", "어머니",
    "자녀", "아들", "딸", "형", "누나", "언니", "오빠", "동생", "형제", "자매",
)
LEASE_WORDS = ("전세", "월세", "임대차", "세들어", "세 들어", "계약")
PROPERTY_WORDS = ("건물", "명의", "소유", "집", "주택")
COHABITATION_WORDS = ("동거", "같이 살", "함께 살", "와이프와", "아내와", "남편과", "배우자와", "가족과", "자녀와", "아이와", "친구와")


def find_workplace(question):
    """질문에 드러난 사업장을 답변에만 사용합니다."""
    destination_match = re.search(r"(?:에서|→|->)\s*(포항|광양|세종|서울)\s*(?:로|으로)", question)
    if destination_match:
        return destination_match.group(1)
    for workplace in ("포항", "광양", "세종", "서울"):
        if workplace in question:
            return workplace
    return ""


def is_relocation_question(question):
    """발령에 따른 부임비·이전비·숙소지원금 복합 문의를 식별합니다."""
    relocation_words = ("발령", "부임", "부임비", "이전비", "이사")
    return any(word in question for word in relocation_words)


def extract_relocation_facts(question):
    """부임 질문에서 이미 알려진 사실을 뽑아 같은 내용을 재질문하지 않습니다."""
    compact_question = question.replace(" ", "")
    not_moving = bool(re.search(r"(?:이사|이전)(?:는|를|가)?(?:하지)?않", compact_question)) or bool(
        re.search(r"(?:이사|이전)(?:는|를|가)?안", compact_question)
    )
    lives_alone = any(word in question for word in ("혼자", "단신", "나만", "본인만"))
    family_elsewhere = any(word in question for word in ("가족은", "배우자는", "아이들은", "자녀는"))
    return {
        "destination": find_workplace(question),
        "not_moving": not_moving,
        "lives_alone": lives_alone,
        "family_elsewhere": family_elsewhere,
    }


def find_family_owner_relation(question):
    """가족 명의 임대차에서 확인된 소유자 관계를 짧게 표시합니다."""
    labels = (
        ("배우자", ("배우자", "와이프", "아내", "남편")),
        ("부모", ("부모", "엄마", "아빠", "아버지", "어머니")),
        ("자녀", ("자녀", "아들", "딸")),
        ("형제·자매", ("형", "누나", "언니", "오빠", "동생", "형제", "자매")),
        ("가족", ("가족",)),
    )
    for label, words in labels:
        if any(word in question for word in words):
            return label
    return "가족"


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
    """승중상은 인정 조건이 확인된 경우에만 지급 기준을 안내합니다."""
    if "승중상" not in question:
        return ""
    confirmed = all(word in question for word in ("아버지", "장손", "상주")) and any(
        word in question for word in ("돌아가", "사망", "별세")
    )
    if not confirmed:
        return (
            "승중상은 조부모상에서 부친이 이미 사망해 장손자가 상주를 맡는 경우를 말합니다.\n\n"
            "확인 결과\n"
            "- 관계: 승중상 인정 조건 확인 필요\n"
            "- 확인 사항: 부친 사망 여부, 장손자 여부, 상주 여부\n"
            "- 필요 서류: 부친 사망 증빙, 본인 가족관계증명서, 상주 확인 자료, 부고장\n\n"
            "위 조건이 확인되면 경조금 500,000원과 화환·장례용품 지원 여부를 안내할 수 있습니다."
        )
    return (
        "경조금 지급 대상입니다.\n\n"
        "확인 결과\n"
        "- 관계: 승중상\n"
        "- 판정: 지원 대상\n"
        "- 지원금: 500,000원\n"
        "- 화환: O\n"
        "- 장례용품: O\n"
        "- 필요 서류: 기본증명서(상세, 사망일 표기 확인), 본인 가족관계증명서, 부친 사망 증빙, 장손자·상주 확인 자료, 부고장\n"
        "- 신청기한: 사유 발생일 당일부터 3개월 이내\n\n"
        "○ 회사 경조 담당 업체(경조물품,화환 등)\n"
        "- 현진시닝 : 1600-0113(24시간)\n\n"
        "최종 승인·지급은 담당 부서의 서류 검토를 거쳐 결정됩니다."
    )


def build_death_answer(question):
    """사망 경조금 문의를 관계별로 판정해 불필요한 반복 없이 안내합니다."""
    if not any(word in question for word in ("돌아가", "사망", "별세", "상") ):
        return ""
    special_relation = next((word for word in SPECIAL_LEAVE_RELATIONS if word in question), "")
    if special_relation:
        documents = "기본증명서(상세, 사망일 표기 확인), 가족관계증명서, 형제 가족관계증명서, 부고장"
        if special_relation in ("백숙부", "백숙부모"):
            documents = "기본증명서(상세, 사망일 표기 확인), 아버지 기준 가족관계증명서, 부고장"
        return (
            "경조금 지원 대상은 아니며, 경조휴가 2일이 지급됩니다.\n\n"
            "확인 결과\n"
            f"- 관계: {special_relation}상\n"
            "- 경조금: 없음\n"
            "- 경조휴가: 2일\n"
            "- 화환: X\n"
            "- 장례용품: X\n"
            f"- 필요 서류: {documents}\n\n"
            "휴가 신청 절차와 서류 인정 범위는 노사발전그룹 검토를 거쳐 결정됩니다."
        )
    if any(word in question for word in AMBIGUOUS_DEATH_RELATIONS):
        return (
            "질문의 관계는 경조금 지급대상 표에 명시되어 있지 않아 지원금액을 단정할 수 없습니다.\n\n"
            "확인 결과\n"
            "- 관계: 가족관계 및 적용 기준 확인 필요\n"
            "- 판정: 노사발전그룹 검토 필요\n"
            "- 제출 가능 서류: 기본증명서(상세, 사망일 표기 확인), 가족관계증명서, 관계를 확인할 수 있는 추가 가족관계증명서, 부고장\n\n"
            "경조금 지급기준에는 해당 관계의 사망 관련 제출서류 기준만 확인됩니다."
        )
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
            documents = "기본증명서(상세, 사망일 표기 확인), 본인 가족관계증명서, 부고장"
            if relation == "본인 및 배우자 조부모":
                documents = "기본증명서(상세, 사망일 표기 확인), 아버지 기준 가족관계증명서, 부고장"
            elif relation == "본인 외조부모":
                documents = "기본증명서(상세, 사망일 표기 확인), 어머니 기준 가족관계증명서, 부고장"
            elif relation == "배우자 부모":
                documents = "기본증명서(상세, 사망일 표기 확인), 배우자 기준 가족관계증명서, 부고장"
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
    topic_words = ("결혼", "사망", "출산", "동호회", "숙소", "출장", "여비", "부임", "발령", "이사", "부임비", "이전비", "건강검진")
    return any(word in question for word in topic_words) and "회갑" not in question


def build_clarification_answer(question):
    """제도 유형을 알 수 없는 질문에 전체 상담 범위와 재질문 형식을 안내합니다."""
    topics = (
        "출장", "파견", "부임", "경조", "결혼", "회갑", "출산", "사망", "돌아가", "별세", "승중상",
        "숙소", "동호회", "발령", "이사", "부임비", "이전비", "전세", "월세", "임대차", "건물", "명의", "동거",
        "백숙부", "매형", "매제", "제부", "형부", "올케",
    )
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
    has_existing_housing = any(
        word in question for word in ("기존 숙소", "기존숙소", "숙소지원금 받고", "숙소지원금을 받고", "숙소지원금 받다가", "수급", "정리", "반납", "유지")
    ) or bool(re.search(r"기존.*숙소|숙소지원금.{0,30}(?:받|수급)", question))
    if not has_existing_housing or not any(word in question for word in ("발령", "부임", "타지", "타 지역")):
        return ""
    duration_match = re.search(r"숙소지원금[^\n]{0,50}?(\d+)\s*년\s*(?:(\d+)\s*개월)?\s*(?:받|수급)", question)
    prior_duration = ""
    if duration_match:
        years = int(duration_match.group(1))
        months = int(duration_match.group(2) or 0)
        prior_duration = f"{years}년" + (f" {months}개월" if months else "")
    destination_workplace = find_workplace(question)
    destination = "서울" if destination_workplace == "서울" else "서울 외"
    amount = "월 60만 원" if destination == "서울" else "월 40만 원"
    asks_cleanup_cost = any(word in question for word in ("내 돈", "본인 부담", "비용", "정리 못", "정리 못해", "청소", "위약금", "중개"))
    period_line = f"- 신규 부임 기본 기준: {amount}, 발령일로부터 3년간\n"
    follow_up = ""
    if prior_duration:
        # 기존 수급 이력이 있으면 신규 부임의 기본 기간을 그대로 확정하지 않습니다.
        period_line = (
            f"- 기존 수급 이력: {prior_duration}\n"
            f"- 신규 부임 기본 기준: {amount}, 발령일로부터 3년간\n"
            "- 지급기간 판정: 기존 근무지 수급이 신규 채용 기준인지 부임 기준인지와 기존 적용기간을 함께 확인해야 함\n"
        )
        follow_up = (
            "기존 근무지에서 받은 숙소지원금이 신규 채용 기준인지 부임 기준인지 알려주시면, "
            "수급 이력을 반영해 새 근무지 부임 후 실제 지원기간을 안내하겠습니다."
        )
    cleanup_cost_note = ""
    if asks_cleanup_cost:
        cleanup_cost_note = (
            "전 근무지 숙소 정리 기간에 발생하는 비용은 3개월간 한도 내 실비 지원 대상입니다. "
            "다만 계약기간과 관계없는 청소비 등 기타 비용은 지원 대상에서 제외됩니다. "
            "중개수수료·위약금·이사비·관리비의 인정 여부는 현재 규정에 명시되어 있지 않아 증빙과 함께 노사발전그룹 검토가 필요합니다.\n"
        )
    return (
        "질문하신 상황은 기존 숙소지원금 수급 중 근무지가 변경되는 경우입니다.\n\n"
        "확인 결과\n"
        "- 전 근무지 숙소: 정리 기간 비용을 숙소지원금 한도 내 실비로 3개월 지원\n"
        "- 연장 기간: 처분 노력 입증자료 제출 시 1개월씩 최대 3회 연장(3개월+1개월+1개월+1개월, 최장 6개월)\n"
        "- 중복 여부: 전 근무지 정리 기간 비용과 새 근무지 숙소지원금은 중복 가능\n"
        f"- 신규 부임지: {destination_workplace or destination}\n"
        f"{period_line}"
        "- 산정 기준: 월세는 월 차임만 지원, 전세는 전세금 1,000만 원당 월 10만 원\n"
        "- 통근버스: 포항·세종 사업장 통근버스 운행 시 숙소지원금 지급 중단\n"
        "- 필요 서류: 전 근무지 정리 기간 비용 증빙, 처분 노력 입증자료(연장 시: 부동산 또는 매물 웹사이트 게시 자료 등), 신규 숙소 임대차계약서\n\n"
        f"{cleanup_cost_note}"
        f"{follow_up}\n"
        "최종 지원 여부와 서류 인정 범위는 담당 부서의 규정 검토를 거쳐 결정됩니다."
    )


def build_housing_exclusion_answer(question):
    """가족 명의 임대차와 실제 동거는 숙소지원금 지급 제외로 우선 판정합니다."""
    has_family_lease = (
        any(word in question for word in FAMILY_OWNER_WORDS)
        and any(word in question for word in LEASE_WORDS)
        and any(word in question for word in PROPERTY_WORDS)
    )
    has_cohabitation = any(word in question for word in COHABITATION_WORDS)
    if not has_family_lease and not has_cohabitation:
        return ""
    reasons = []
    if has_family_lease:
        relation = find_family_owner_relation(question)
        reasons.append(f"- 임대차: {relation} 명의 건물에 전세·월세 계약")
    if has_cohabitation:
        reasons.append("- 실제 거주 형태: 단신부임 기준에 맞지 않는 동거")
    reason_text = "\n".join(reasons)
    return (
        "숙소지원금 지원 대상이 아닙니다.\n\n"
        "확인 결과\n"
        f"{reason_text}\n"
        "- 판정: 지원 불가\n"
        "- 숙소지원금: 없음\n\n"
        "가족 명의 건물에 전세·월세로 거주하거나 단신부임 신청 후 실제 동거하는 경우는 숙소지원금 지급 제외 기준입니다. "
        "신청 내용이나 실제 거주 형태가 사실과 다르면 윤리위반으로 감사 대상이 될 수 있습니다."
    )


def build_housing_lease_answer(question):
    """일반 임대인 전세·월세 계약은 숙소지원금 요건만 간결하게 확인합니다."""
    if not any(word in question for word in ("전세", "월세", "임대차")):
        return ""
    return (
        "일반 임대인과 체결하는 전세·월세 계약은 숙소지원금 요건을 충족하면 검토할 수 있습니다.\n\n"
        "확인 결과\n"
        "- 전세 지원 기준: 전세금 10,000,000원당 월 100,000원\n"
        "- 월세 지원 기준: 월 차임만 지원하며 관리비·공과금 등은 제외\n"
        "- 확인 조건: 새 근무지, 실제 단신 거주 여부, 본인·배우자 주택 보유 여부, 임대인과의 가족관계, 임대차계약서\n\n"
        "계약 형태와 임대인 관계, 새 근무지를 알려주시면 지원 가능 여부를 확인해 드리겠습니다."
    )


def build_housing_contract_change_answer(question):
    """기존 수급자의 월세·전세 전환은 변경 계약 기준과 증빙을 안내합니다."""
    if not ("월세" in question and "전세" in question):
        return ""
    if not any(word in question for word in ("바꾸", "변경", "전환", "바뀌", "받고", "수급")):
        return ""
    monthly_first = question.find("월세") < question.find("전세")
    if monthly_first:
        change = "월세 → 전세"
        standard = "전세금 10,000,000원당 월 100,000원"
        documents = "변경된 임대차계약서, 계약조건 확인 자료"
        opening = "월세와 전세 간 계약 형태가 변경되면, 변경된 계약 기준으로 숙소지원금을 산정하기 위해 관련 증빙서류를 새로 제출해야 합니다."
    else:
        change = "전세 → 월세"
        standard = "월 차임만 지원하며 관리비·공과금은 제외"
        documents = "변경된 임대차계약서, 월세 이체내역, 계약조건 확인 자료"
        opening = "전세와 월세 간 계약 형태가 변경되면, 변경된 계약 기준으로 숙소지원금을 산정하기 위해 관련 증빙서류를 새로 제출해야 합니다."
    return (
        f"{opening}\n\n"
        "확인 결과\n"
        f"- 변경 내용: {change}\n"
        "- 필요 조치: 변경된 임대차계약서와 계약조건 확인 자료 제출\n"
        f"- 변경 후 지원 기준: {standard}\n"
        "- 지원 산정: 변경된 계약 형태 기준으로 재산정\n"
        f"- 필요 서류: {documents}\n\n"
        "세부 제출서류와 적용 시점은 숙소지원금 담당자에게 문의해 주세요."
    )


def build_relocation_answer(question):
    """부임비·이전비와 숙소지원금을 질문 의도에 맞춰 함께 안내합니다."""
    if not is_relocation_question(question):
        return ""
    facts = extract_relocation_facts(question)
    destination = facts["destination"]
    if not destination:
        return (
            "발령에 따른 부임비·이전비와 숙소지원금은 각각 기준이 다릅니다.\n\n"
            "확인 결과\n"
            "- 부임비·이전비: 실제 이사 여부에 따라 판단\n"
            "- 숙소지원금: 새 근무지, 실제 단신 거주, 주택 보유 여부에 따라 판단\n\n"
            "발령받은 근무지역과 이사 여부를 알려주시면 적용되는 지원만 안내해 드리겠습니다."
        )
    is_seoul = destination == "서울"
    housing_amount = "월 600,000원" if is_seoul else "월 400,000원"
    housing_line = f"- 숙소지원금 기준: {destination} 신규 부임 시 {housing_amount}, 발령일로부터 최대 3년\n"
    if facts["not_moving"]:
        opening = "이사를 하지 않으면 부임비와 이전비는 지급되지 않습니다."
        moving_line = "- 부임비·이전비: 이사하지 않으면 지급 없음\n"
    else:
        opening = "부임비와 이전비는 실제 이사 여부와 이사·중개 비용 증빙을 기준으로 판단합니다."
        moving_line = "- 부임비·이전비: 실제 이사 여부와 이사·중개 비용 증빙을 기준으로 판단\n"
    residence_line = "- 거주 계획: 본인이 새 근무지에서 혼자 거주 예정\n" if facts["lives_alone"] else ""
    if facts["family_elsewhere"]:
        residence_line += "- 가족 거주지: 기존 지역에 거주 예정\n"
    next_question = "새 근무지 숙소의 임대차계약 여부와 본인·배우자 주택 보유 여부를 알려주시면 숙소지원금 가능 여부를 확인해 드리겠습니다."
    return (
        f"{opening} 다만 {destination}에 별도 숙소를 구해 실제로 혼자 거주한다면 숙소지원금 대상 여부를 검토할 수 있습니다.\n\n"
        "확인 결과\n"
        f"- 신규 부임지: {destination}\n"
        f"{moving_line}"
        f"{residence_line}"
        f"{housing_line}"
        "- 숙소지원금 확인 조건: 새 근무지 주택 보유 여부, 실제 단신 거주 여부, 타지역 생활근거지, 임대차계약서\n"
        "- 신청기한: 발령일이 속한 달의 다음 달부터 6개월 이내\n\n"
        f"{next_question}"
    )


def build_domestic_trip_answer(question):
    """국내 출장의 핵심 지급 기준은 모델 해석 없이 고정 안내합니다."""
    if not any(word in question for word in ("출장", "국내여비", "교통비", "숙박비", "식비", "현지교통비")):
        return ""
    if any(word in question for word in ("해외", "파견", "부임")):
        return ""
    return (
        "국내 출장 여비는 사후 정산으로 지급합니다.\n\n"
        "확인 결과\n"
        "- 교통비: 철도·선박·항공·자동차 실비\n"
        "- 자가용: 유류비·통행료·감가상각비(50원/km) 실비, 통행료 영수증 필요\n"
        "- 숙박비: 1박 100,000원 한도 내 실비\n"
        "- 소액경비: 식비 1일 30,000원, 현지교통비 1일 20,000원\n"
        "- 식비 차감: 외부 또는 내부에서 제공받은 식사는 1회당 10,000원 차감\n"
        "- 정산: 귀임 후 30일 이내 증빙 제출\n\n"
        "출장 목적·기간·교통수단·숙박 여부를 알려주시면 적용 가능한 항목만 정리해 드리겠습니다."
    )


def build_parking_answer(question):
    """국내출장 주차비와 교육연수·해외출장 주차비를 서로 다른 기준으로 안내합니다."""
    has_parking = any(word in question for word in ("주차", "주차비", "주차장"))
    is_training_or_overseas = any(word in question for word in ("해외출장", "해외 출장", "교육연수", "교육 연수"))
    is_domestic_parking = any(word in question for word in ("국내", "공항"))
    is_domestic_trip = any(word in question for word in ("국내출장", "국내 출장"))
    if not has_parking:
        return ""
    if is_training_or_overseas and is_domestic_parking:
        trip_type = "교육연수" if any(word in question for word in ("교육연수", "교육 연수")) else "해외출장"
        return (
            f"{trip_type}을 위한 국내 주차비는 지원되지 않습니다.\n\n"
            "확인 결과\n"
            f"- 출장 유형: {trip_type}\n"
            "- 비용 항목: 국내 주차비\n"
            "- 판정: 지급 불가\n"
            "- 근거: 교육연수 및 해외출장 시 국내 주차비는 지급하지 않음"
        )
    if is_domestic_trip:
        return (
            "일반 국내출장 중 유료 주차비는 별도 실비 정산 항목이 아니라 소액경비의 현지교통비 범위에서 처리합니다.\n\n"
            "확인 결과\n"
            "- 출장 유형: 국내출장\n"
            "- 비용 항목: 유료 주차비\n"
            "- 처리 기준: 소액경비 중 현지교통비로 충당\n"
            "- 현지교통비 기준: 1일 20,000원\n\n"
            "다만 교육연수 및 해외출장 시 발생한 국내 주차비는 지급되지 않습니다."
        )
    return ""


def build_club_answer(question):
    """동호회 개설·가입·정기지원 문의에 공통 기준을 적용합니다."""
    if "동호회" not in question:
        return ""
    return (
        "동호회 지원은 노사발전그룹의 등록·활동 실적 확인 후 지급됩니다.\n\n"
        "확인 결과\n"
        "- 결성·활동 최소 인원: 5명\n"
        "- 가입: 소속 사업장과 실근무지를 합해 1인 최대 2개, 동일 분야 이중 가입 불가\n"
        "- 정기지원: 분기별 1회 이상 활동 시 1인 30,000원, 최대 300,000원\n"
        "- 특별지원: 연간 2회, 1인 10,000원, 최대 300,000원\n"
        "- 지급 시기: 분기 활동 후 실적 등록 및 검토 완료 후 지급\n"
        "- 필요 서류: 활동실적 보고서, 영수증, 활동사진, 전체 회원·참석자 명단\n\n"
        "신규 동호회는 등록 3개월 이후부터 지원하며, 향우회·동문회·부서 친목회는 지원 대상이 아닙니다."
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
        " 경조금 청구기한은 사유 발생일 당일부터 3개월 이내이며, 주말·공휴일도 별도 연장하지 않는다."
        " 백숙부모상·매형·매제상·형부·제부상·올케상은 경조금·화환·장례용품 지원은 없고 경조휴가 2일만 안내한다."
        " 승중상은 부친 사망으로 장손자가 조부모상의 상주가 된 경우에만 인정한다."
        " 근무지역 이동 시 '숙소 정리비'라는 별도 지원금 표현을 쓰지 말고, '전 근무지 숙소 정리 기간에 발생한 비용'으로 안내한다. "
        "해당 비용은 기본 3개월, 처분 노력 입증자료 제출 시 1개월씩 최대 3회 연장하여 최장 6개월(3개월+1개월+1개월+1개월)까지 지원할 수 있다. 청소비는 제외한다."
        " 국내 출장은 교통비 실비, 숙박비 1박 10만원 한도, 식비 1일 3만원, 현지교통비 1일 2만원을 기준으로 한다. 제공된 식사는 1회당 식비 1만원을 차감한다."
        " 동호회 정기지원은 분기 1회 이상 활동 시 인당 3만원·최대 30만원이며 영수증·활동사진·참석자 명단이 필요하다."
        " 사용자가 이미 말한 사실(관계, 날짜, 근무지, 이사 여부, 동거 여부, 질문 항목)은 다시 묻지 않는다."
        " 발령·부임·이사·부임비·이전비·가족 잔류·혼자 거주 표현이 있으면 부임 및 숙소지원금 복합 문의로 파악한다."
        " 이 경우 실제 이사하지 않으면 부임비 지급이 없다는 점을 먼저 답하고, 숙소지원금은 새 근무지·실제 단신 거주·주택 보유 여부·임대차계약서를 기준으로 필요한 만큼만 안내한다."
        " 가족 명의(본인·배우자·부모·자녀·형제자매) 건물에 전세 또는 월세로 거주하는 경우와 단신부임으로 신청한 뒤 실제 동거인이 있는 경우는 숙소지원금 지원 불가로 안내한다."
        " 이 경우 전세금·월세 지원금 계산을 하지 말고, '신청 내용이나 실제 거주 형태가 사실과 다르면 윤리위반으로 감사 대상이 될 수 있습니다.'라고 짧게 경고한다."
        " 기존 숙소지원금 수급 중 월세와 전세의 계약 형태가 바뀌면 변경된 임대차계약서와 계약조건 확인 자료를 새로 제출하도록 안내한다."
        " 월세에서 전세로 바뀌면 전세금 1,000만원당 월 10만원, 전세에서 월세로 바뀌면 월 차임만 지원하고 관리비·공과금은 제외한다고 표시한다."
        " 계약 전환의 세부 제출서류와 적용 시점은 숙소지원금 담당자에게 문의하도록 답변을 끝낸다."
        " 해외출장 또는 교육연수의 국내공항·국내 주차비는 지급하지 않는다고 명확히 안내한다. 회사 차량·개인 차량 여부에 따른 예외를 만들지 않는다."
        " 일반 국내출장의 유료 주차비는 별도 실비 정산 항목이 아니라 소액경비 중 현지교통비 1일 2만원 범위에서 처리한다고 안내한다."
        " 질문의 제도를 식별할 수 있으면 전체 복리후생 목록을 보여주지 않는다. 전체 목록은 제도와 상황을 전혀 알 수 없는 질문에서만 사용한다."
        " 모든 제도 답변은 질문에 대한 결론 한 문장, 확인 결과, 반드시 필요한 추가 확인 한두 항목 순서로 간결하게 작성한다."
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
    if is_relocation_question(question):
        return {"intent": "relocation"}
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
    housing_exclusion_answer = build_housing_exclusion_answer(question)
    housing_lease_answer = build_housing_lease_answer(question)
    housing_contract_change_answer = build_housing_contract_change_answer(question)
    housing_move_answer = build_housing_move_answer(question)
    relocation_answer = build_relocation_answer(question)
    trip_answer = build_domestic_trip_answer(question)
    parking_answer = build_parking_answer(question)
    club_answer = build_club_answer(question)
    answer = marriage_answer or seungjungsang_answer or hoegap_answer or death_answer or housing_exclusion_answer or housing_contract_change_answer or housing_move_answer or housing_lease_answer or relocation_answer or parking_answer or trip_answer or club_answer
    if not answer:
        return {}
    if marriage_answer or seungjungsang_answer or hoegap_answer or death_answer:
        evidence = [{"file": "경조금 지급기준.txt", "score": 1, "text": "경조금 지급기준"}]
    elif housing_exclusion_answer or housing_contract_change_answer or housing_lease_answer or housing_move_answer:
        evidence = [{"file": "숙소지원금 운영 기준.txt", "score": 1, "text": "숙소지원금 운영 기준"}]
    elif relocation_answer:
        evidence = [
            {"file": "숙소지원금 운영 기준.txt", "score": 1, "text": "숙소지원금 운영 기준"},
            {"file": "여비관리기준.txt", "score": 1, "text": "여비관리기준"},
        ]
    elif parking_answer or trip_answer:
        evidence = [{"file": "여비관리기준.txt", "score": 1, "text": "여비관리기준"}]
    else:
        evidence = [{"file": "동호회 관리 규정.txt", "score": 1, "text": "동호회 관리 규정"}]
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
INQUIRY_HISTORY = []


def build_inquiry_draft(question, answer, evidence):
    """상담 결과를 담당자 문의 메일 초안으로 변환합니다."""
    evidence_lines = "\n".join(
        f"- {item.get('file', '관련 규정')}"
        for item in evidence
    ) or "- 확인된 규정 근거 없음"
    body = (
        "안녕하세요.\n"
        "복리후생 지원 가능 여부를 확인 부탁드립니다.\n\n"
        "[문의 내용]\n"
        f"{question}\n\n"
        "[AI 사전 검토 내용]\n"
        f"{answer}\n\n"
        "[관련 규정]\n"
        f"{evidence_lines}\n\n"
        "최종 지원 가능 여부와 필요한 추가 서류를 확인해 주시면 감사하겠습니다.\n\n"
        "감사합니다.\n복리후생 상담 시스템 드림"
    )
    return {
        "recipient": "복리후생 담당자",
        "recipient_email": "welfare-demo@example.com",
        "subject": "복리후생 지원 가능 여부 확인 요청",
        "body": body,
        "evidence": evidence,
    }


class Handler(SimpleHTTPRequestHandler):
    """정적 화면과 상담 요청을 함께 제공하는 간단한 HTTP 핸들러입니다."""

    def do_POST(self):
        if self.path not in ("/api/chat", "/api/inquiries/draft", "/api/inquiries/send", "/api/inquiries/save", "/api/inquiries/delete"):
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/api/inquiries/draft":
                question = str(body.get("question", "")).strip()
                answer = str(body.get("answer", "")).strip()
                if not question or not answer:
                    raise ValueError("문의 초안을 만들 상담 내용이 없습니다.")
                self.respond(200, build_inquiry_draft(question, answer, body.get("evidence", [])))
                return
            if self.path == "/api/inquiries/send":
                subject = str(body.get("subject", "")).strip()
                inquiry_body = str(body.get("body", "")).strip()
                if not subject or not inquiry_body:
                    raise ValueError("메일 제목과 본문을 입력해 주세요.")
                inquiry = {
                    "id": len(INQUIRY_HISTORY) + 1,
                    "recipient": str(body.get("recipient", "복리후생 담당자")),
                    "recipient_email": str(body.get("recipient_email", "welfare-demo@example.com")),
                    "subject": subject,
                    "body": inquiry_body,
                    "sent_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "status": "메일 요청 완료",
                }
                INQUIRY_HISTORY.append(inquiry)
                self.respond(200, inquiry)
                return
            if self.path == "/api/inquiries/save":
                inquiry = {
                    "id": len(INQUIRY_HISTORY) + 1,
                    "recipient": str(body.get("recipient", "복리후생 담당자")),
                    "recipient_email": str(body.get("recipient_email", "welfare-demo@example.com")),
                    "subject": str(body.get("subject", "")).strip(),
                    "body": str(body.get("body", "")).strip(),
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "status": "작성 중",
                }
                if not inquiry["subject"] or not inquiry["body"]:
                    raise ValueError("메일 제목과 본문을 입력해 주세요.")
                INQUIRY_HISTORY.append(inquiry)
                self.respond(200, inquiry)
                return
            if self.path == "/api/inquiries/delete":
                inquiry_id = int(body.get("id", 0))
                INQUIRY_HISTORY[:] = [item for item in INQUIRY_HISTORY if item["id"] != inquiry_id]
                self.respond(200, {"deleted": inquiry_id})
                return
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
        if self.path == "/api/inquiries":
            self.respond(200, {"items": INQUIRY_HISTORY})
            return
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
