import hashlib
import json
import math
import os
import re
import calendar
from collections import Counter
from datetime import datetime
from datetime import date
from functools import lru_cache
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
    "경조금 지급기준.md",
    "동호회 관리 규정.md",
    "숙소지원금 운영 기준.md",
    "여비관리 FAQ.md",
    "여비관리기준.md",
}
# 규정의 '회갑'과 사용자가 자주 쓰는 '환갑'을 같은 의미로 처리합니다.
HOEGAP_TERMS = ("회갑", "환갑")
# 조사나 어미가 붙어도 규정의 핵심어로 검색해야 하는 표현입니다.
CANONICAL_QUERY_TERMS = ("결혼", "해외출장")
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


# 조사가 붙으면 "회사"와 "회사가"가 다른 토큰이 되어 흔한 단어가 희귀어로 잘못 계산됩니다.
PARTICLES = (
    "에서는", "에게는", "으로는", "이라도", "에서", "에게", "으로", "까지", "부터", "보다",
    "처럼", "마다", "조차", "밖에", "이나", "라도", "은", "는", "이", "가", "을", "를",
    "의", "에", "도", "만", "과", "와", "로", "나",
)


def strip_particle(word):
    """세 글자 이상 한글 어절에서 흔한 조사를 떼어 표제어에 가깝게 만듭니다."""
    if not re.fullmatch(r"[가-힣]+", word):
        return word
    for particle in PARTICLES:
        if len(word) - len(particle) >= 2 and word.endswith(particle):
            return word[: -len(particle)]
    return word


def tokens(text):
    """한글·영문·숫자 어절에서 조사를 떼어 검색 토큰을 만듭니다."""
    return {strip_particle(word) for word in re.findall(r"[가-힣A-Za-z0-9]+", text.lower())}


def bigrams(text):
    """한글 어절을 음절 2-gram으로 쪼갭니다."""
    # 조사·어미가 붙으면 어절이 달라지므로("해체하려면" vs "해체") 2-gram으로 겹치게 합니다.
    result = set()
    for word in re.findall(r"[가-힣]{2,}", text.lower()):
        result.update(word[index:index + 2] for index in range(len(word) - 1))
    return result


def split_policy_chunks(path) -> list[dict]:
    """리프 제목의 경로와 본문을 분리하고, 부모 서두는 첫 자식에 합칩니다."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() != ".md":
        return [{"path": "", "text": p.strip()} for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    headings = []
    current_lines = []
    for line in text.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2)
            # 더 깊은 제목이면 부모 서두를 유지하고, 리프가 끝날 때만 청크를 확정합니다.
            if headings and level <= headings[-1][0]:
                body = "\n".join(current_lines).strip()
                if body:
                    chunks.append({"path": " > ".join(title for _, title in headings), "text": body})
                current_lines = []
            # 제목 단계가 건너뛰어져도 실제 상위 제목만 경로에 남깁니다.
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, title))
            continue
        current_lines.append(line)
    body = "\n".join(current_lines).strip()
    if body:
        chunks.append({"path": " > ".join(title for _, title in headings), "text": body})

    return chunks


BM25_K1 = 1.5
BM25_B = 0.75
# 2-gram 가중치. 평가셋에서 1.0이 상위 3건 정확도가 가장 높아 어절과 같은 무게로 둡니다.
BIGRAM_WEIGHT = 1.0
# 최고 점수 파일 대비 이 비율 이상인 규정만 함께 근거로 사용합니다.
TOP_FILE_SCORE_RATIO = 0.6


@lru_cache(maxsize=1)
def load_policy_index():
    """승인된 규정의 청크와 idf를 한 번만 계산해 재사용합니다."""
    # ponytail: 규정 파일이 런타임에 바뀌지 않는다고 보고 캐시합니다. 문서를 고치면 재시작이 필요합니다.
    chunks = []
    for path in sorted((*RULES_DIR.glob("*.txt"), *RULES_DIR.glob("*.md"))):
        # 상담 근거로 승인된 파일만 사용하고 샘플 문서는 제외합니다.
        if path.name not in SOURCE_FILES:
            continue
        for chunk in split_policy_chunks(path):
            searchable = f"{chunk['path']}\n{chunk['text']}"
            chunks.append({
                "file": path.name,
                "stem": path.stem,
                "path": chunk["path"],
                "text": chunk["text"],
                "searchable": searchable,
                "tokens": tokens(searchable),
                "bigrams": bigrams(searchable),
            })
    total = len(chunks)
    frequency = Counter(token for chunk in chunks for token in chunk["tokens"] | chunk["bigrams"])
    # 흔한 단어(출장·지원)는 낮게, 희귀어(택시·해체)는 높게 가중합니다.
    idf = {
        token: math.log(1 + (total - count + 0.5) / (count + 0.5))
        for token, count in frequency.items()
    }
    average_length = sum(len(chunk["tokens"]) for chunk in chunks) / total if total else 1.0
    # 계층 경로를 본문과 함께 임베딩합니다. "5.2 지원 대상"처럼 본문에 제도명이 없는 청크는
    # 제목에만 의미가 있어, 본문만 임베딩하면 벡터 점수가 구조적으로 낮게 나옵니다.
    embeddings = embed_texts([chunk["searchable"] for chunk in chunks])
    for chunk in chunks:
        chunk["vector"] = embeddings.get(hashlib.sha256(chunk["searchable"].encode("utf-8")).hexdigest())
    return chunks, idf, average_length


def score_chunk(query_tokens, query_bigrams, chunk, idf, average_length):
    """희귀어 가중과 길이 정규화를 적용한 BM25 점수를 계산합니다."""
    matched = query_tokens & chunk["tokens"]
    # 2-gram은 조사·어미를 넘기 위한 보조 신호라 어절 일치보다 낮게 봅니다.
    matched_bigrams = (query_bigrams & chunk["bigrams"]) - matched
    if not matched and not matched_bigrams:
        return 0.0
    weight = sum(idf.get(token, 0.0) for token in matched)
    weight += BIGRAM_WEIGHT * sum(idf.get(token, 0.0) for token in matched_bigrams)
    length_ratio = len(chunk["tokens"]) / average_length if average_length else 1.0
    normalizer = BM25_K1 * (1 - BM25_B + BM25_B * length_ratio) + 1
    return weight * (BM25_K1 + 1) / normalizer


EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_CACHE_PATH = BASE_DIR / ".embedding_cache.json"
# 조사·어미 변형은 2-gram이, 동의어·패러프레이즈는 임베딩이 잡습니다.
# 44문항 스윕에서 0.5→0.8이 hit@1 45.5→63.6%, MRR 0.639→0.744로 개선됐습니다.
# 0.8을 넘기면 키워드가 지키던 문항("경조휴가 중 결근", "전세 계약 이사비")이 1위→9위 밖으로
# 밀리고, 평가셋이 패러프레이즈 위주라 벡터에 유리하게 편향돼 있어 여기서 멈춥니다.
VECTOR_WEIGHT = 0.8


def request_embeddings(texts):
    """임베딩 API를 호출합니다. 키가 없거나 실패하면 빈 리스트를 반환합니다."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return []
    vectors = []
    for start in range(0, len(texts), 64):
        batch = texts[start:start + 64]
        request = Request(
            "https://api.openai.com/v1/embeddings",
            data=json.dumps({"model": EMBEDDING_MODEL, "input": batch}).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=45) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError):
            # 임베딩이 실패하면 키워드 검색만으로 동작하도록 비워서 돌려줍니다.
            return []
        vectors.extend(item["embedding"] for item in payload["data"])
    return vectors


def embed_texts(texts):
    """텍스트별 임베딩을 파일 캐시와 함께 반환합니다."""
    cache = {}
    if EMBEDDING_CACHE_PATH.exists():
        cache = json.loads(EMBEDDING_CACHE_PATH.read_text(encoding="utf-8"))
    keys = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
    missing = [text for text, key in zip(texts, keys) if key not in cache]
    if missing:
        fresh = request_embeddings(missing)
        if not fresh:
            return {}
        for text, vector in zip(missing, fresh):
            cache[hashlib.sha256(text.encode("utf-8")).hexdigest()] = vector
        EMBEDDING_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    return {key: cache[key] for key in keys if key in cache}


def cosine(left, right):
    """두 벡터의 코사인 유사도를 계산합니다."""
    dot = sum(a * b for a, b in zip(left, right))
    size = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / size if size else 0.0


# 코퍼스가 작아 후보를 넓게 주고, 어떤 조항이 답인지는 판정 단계에서 가립니다.
def retrieve(question, limit=20):
    """Markdown 제목 청크와 기존 텍스트 규정에서 관련 근거를 찾아 반환합니다."""
    query_tokens = tokens(question)
    compact_question = question.replace(" ", "")
    for term in CANONICAL_QUERY_TERMS:
        if term in compact_question:
            query_tokens.add(term)
    if any(word in question for word in ("숙소", "숙소지원금", "기존 숙소", "전 근무지", "반납", "정리", "유지")):
        # 근무지 이동 관련 질문은 5.5 지원특례의 핵심 표현을 함께 검색합니다.
        query_tokens.update({"전근무지", "숙소정리", "3개월", "최장", "6개월", "처분", "발령"})
    for word, synonyms in QUERY_SYNONYMS.items():
        if word in question:
            query_tokens.update(synonyms)
    query_bigrams = bigrams(question)
    chunks, idf, average_length = load_policy_index()
    question_vectors = embed_texts([question])
    query_vector = next(iter(question_vectors.values()), None)
    scored = []
    for chunk in chunks:
        keyword_score = score_chunk(query_tokens, query_bigrams, chunk, idf, average_length)
        vector_score = 0.0
        if query_vector and chunk.get("vector"):
            vector_score = cosine(query_vector, chunk["vector"])
        if keyword_score or vector_score:
            scored.append((chunk, keyword_score, vector_score))
    if not scored:
        return []
    top_keyword = max(item[1] for item in scored) or 1.0
    top_vector = max(item[2] for item in scored) or 1.0
    results = []
    for chunk, keyword_score, vector_score in scored:
        # 임베딩을 못 쓰면 키워드 점수만으로 순위를 정합니다.
        if query_vector:
            fused = (1 - VECTOR_WEIGHT) * (keyword_score / top_keyword) + VECTOR_WEIGHT * (vector_score / top_vector)
        else:
            fused = keyword_score / top_keyword
        results.append({
            "file": chunk["file"],
            "score": round(fused, 3),
            "keyword_score": round(keyword_score, 3),
            "vector_score": round(vector_score, 3),
            "path": chunk["path"],
            "text": chunk["text"][:3000],
        })
    results.sort(key=lambda item: item["score"], reverse=True)
    if not results:
        return []
    # 최고 점수에 근접한 규정 파일만 선택해 다른 제도 설명이 섞이지 않게 합니다.
    threshold = results[0]["score"] * TOP_FILE_SCORE_RATIO
    top_files = {item["file"] for item in results if item["score"] >= threshold}
    results = [item for item in results if item["file"] in top_files]
    selected = []
    taken = set()
    # 여러 규정이 함께 적용될 수 있으므로 규정별 상위 근거를 먼저 확보합니다.
    for path in sorted({item["file"] for item in results}):
        for item in [item for item in results if item["file"] == path][:3]:
            selected.append(item)
            taken.add((item["file"], item["path"]))
    # 관련 규정이 하나뿐이면 파일별 상한 탓에 근거가 3건으로 잘리므로 남은 자리를 채웁니다.
    for item in results:
        if len(selected) >= limit:
            break
        if (item["file"], item["path"]) not in taken:
            selected.append(item)
            taken.add((item["file"], item["path"]))
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
    if not any(term in question for term in HOEGAP_TERMS):
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
    if birth and (any(term in question for term in HOEGAP_TERMS) or any(term in prior_text for term in HOEGAP_TERMS)):
        birthday = birthday_context(f"회갑 {birth.strftime('%Y%m%d')}")
    if not relation and not birthday:
        return ""
    return f"\n[대화 맥락 보완]\n{relation}\n{birthday}".strip()


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


def build_clarification_answer(question):
    """제도 유형을 알 수 없는 질문에 전체 상담 범위와 재질문 형식을 안내합니다."""
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


def build_unknown_policy_answer(question):
    """제도 식별이 불가능한 질의에 공통 확인 안내를 제공합니다."""
    return (
        "문의하신 내용과 직접 연결되는 복리후생 규정을 확인하지 못했습니다.\n\n"
        "어떤 지원 제도에 대한 문의인지, 확인하고 싶은 비용이나 신청 항목을 조금 더 구체적으로 알려주세요.\n"
        "현재 정보만으로는 지원 가능 여부를 판단하기 어렵습니다."
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


def build_overseas_personal_return_answer(question):
    """해외출장 후 개인 일정에 따른 귀국 항공편은 지급 불가로 안내합니다."""
    compact_question = question.replace(" ", "")
    is_overseas_trip = "해외출장" in compact_question
    has_personal_leave = any(word in question for word in ("개인휴가", "개인 휴가", "개인 일정", "연차", "휴가"))
    is_delayed_return = any(word in question for word in ("복귀", "귀국", "입국", "돌아오", "돌아가", "들어오", "와도"))
    asks_airfare = any(word in question for word in ("항공", "비행기", "항공권", "항공편", "교통비", "티켓"))
    if not (is_overseas_trip and has_personal_leave and is_delayed_return and asks_airfare):
        return ""
    return (
        "개인 연차나 휴가를 사용하여 해외출장 일정 종료 후 체류한 뒤 귀국하는 항공편은 "
        "회사에서 지원하지 않습니다.\n\n"
        "확인 결과\n"
        "- 복귀 원칙: 회사가 승인한 해외출장 일정 내 귀국\n"
        "- 출장 종료 후 체류: 개인 연차·휴가 또는 개인 여행\n"
        "- 항공편: 개인 일정으로 변경된 귀국 항공편은 지원하지 않음\n"
        "- 개인 부담: 귀국 항공료, 운임 차액 및 변경 수수료\n"
        "- 근태: 개인 연차·휴가와 추가 체류 기간은 별도 근태 승인 필요\n"
        "- 판정: 지급 불가\n\n"
        "승인된 해외출장 일정 내 귀국 항공편을 이용해 주세요. 업무상 사유로 귀국 일정 변경이 "
        "필요한 경우에만 변경 전에 출장 승인권자와 노무관리 주관부서의 승인을 받아야 합니다."
    )


def is_club_application_question(question):
    """동호회 신규 신청 양식과 담당자 안내를 묻는 질문을 식별합니다."""
    application_words = (
        "신청", "신청서", "신규", "결성", "개설", "창설", "만들", "등록", "양식", "회장", "총무", "계좌번호"
    )
    return "동호회" in question and any(word in question for word in application_words)




def call_openai(question, evidence, history=None):
    """검색 근거를 포함해 Responses API를 호출합니다."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(".env에 OPENAI_API_KEY가 입력되지 않았습니다.")
    evidence_text = "\n\n".join(
        f"[근거 {i}] {item['file']} ({item.get('path', '')})\n{item['text']}"
        for i, item in enumerate(evidence, 1)
    ) or "관련 규정 근거를 찾지 못했습니다."
    instructions = (
        "너는 사내 복리후생 규정 상담 Agent다. 네 개의 제공된 원문 규정을 기준 데이터로 사용하며, 반드시 제공된 근거만 사용해 한국어로 답한다. "
        "불확실한 질의는 세 유형으로 구분한다. 제도 자체를 식별할 수 없으면 어떤 제도·비용인지 구체화를 요청한다. 제도는 식별되지만 정보가 부족하면 부족한 정보만 요청한다. 관련 규정이 없거나 충돌하면 지급 여부를 확정하지 말고 주관 부서 문의를 안내한다. "
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
        # 국내출장 금액 기준은 규정에 있으므로 여기에 옮겨 적지 않는다. 옮겨 적었던 문장은
        # 5.14.2의 거리 조건(왕복 120km/50km/미만)을 떼고 단정했고, "식비 1일 3만원"은
        # 5.14.3 자가차량 동승자 기준을 일반 기준으로 잘못 옮긴 것이었다.
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
    birthday_question = question if any(term in question for term in HOEGAP_TERMS) else (f"회갑 {question}" if any(term in prior_text for term in HOEGAP_TERMS) else question)
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


POLICY_INTENTS = ("ceremony", "housing", "relocation", "trip", "club", "other")


GROUNDEDNESS_INSTRUCTIONS = (
    "너는 사내 복리후생 규정 상담의 근거 판정기다. 사용자 질문과 검색된 규정 근거를 읽고, "
    "그 근거만으로 질문에 답할 수 있는지 판정한다. 답을 작성하지 말고 판정만 한다.\n"
    "판정은 셋 중 하나다.\n"
    "- answerable: 근거에 질문의 답이 실제로 들어 있다.\n"
    "- clarify: 제도는 맞게 찾았으나 관계·금액·기간처럼 답을 정하는 정보가 질문에 빠져 있다.\n"
    "clarify일 때 missing에는 질문에 아직 없는 정보만 넣는다. 질문이 이미 밝힌 사실은 "
    "다시 묻지 않는다. 예를 들어 질문에 '개인 사정으로'라고 적혀 있으면 "
    "업무상인지 개인 사정인지는 묻지 않는다. 되물을 것이 남지 않으면 clarify가 아니다.\n"
    "missing에는 사용자가 스스로 답할 수 있는 사실만 넣는다. 날짜·금액·관계·실제 일정처럼 "
    "사용자가 아는 것이다. '회사가 인정하는지', '승인이 나는지', '규정상 가능한지'처럼 "
    "담당 부서의 판단이 필요한 항목은 missing이 아니다. 그것은 사용자가 물은 질문 자체이므로 "
    "되물으면 대화가 제자리를 돈다. 그런 항목만 남는다면 clarify가 아니라 escalate다.\n"
    "이전 대화가 함께 주어지면 그 안에서 이미 되물은 항목과 사용자가 답한 사실을 확인한다. "
    "표현을 바꿔서 다시 묻지 않는다. 같은 것을 두 번 물어야 할 상황이면 이미 답을 받은 것이므로 "
    "clarify가 아니라 answerable 또는 escalate로 판정한다.\n"
    "- escalate: 근거가 질문의 주제를 다루지 않는다. 어휘가 겹쳐도 다른 항목을 다루면 escalate다.\n"
    "예를 들어 근거가 '주차비는 기타 경비로 지급'인데 질문이 '주차 위반 과태료'라면, "
    "주차라는 단어가 겹쳐도 과태료를 다루지 않으므로 escalate다.\n"
    "판정과 함께 질문의 제도 영역과 대상 관계도 뽑는다.\n"
    "- intent: ceremony(경조금) | housing(숙소지원금) | relocation(부임비·이전비) | "
    "trip(출장·여비) | club(동호회) | other 중 하나. 질문이 실제로 묻는 제도를 고른다. "
    "단어가 겹쳐도 묻는 제도가 아니면 고르지 않는다. '이사회 참석 출장비'는 relocation이 아니라 trip이다.\n"
    "other는 회사의 복리후생·비용 지원과 무관한 질문에만 쓴다. 점심 메뉴 추천이 그런 예다. "
    "회사가 비용을 부담하는지 묻는 질문은 규정에 해당 항목이 없더라도 가장 가까운 영역을 고른다. "
    "'주차 위반 과태료를 회사가 내주는지'는 other가 아니라 trip이다.\n"
    "- relation: 경조사·경조금 질문에서 사유가 발생한 대상과 사용자의 관계. "
    "본인, 본인 부모, 배우자 부모, 본인 형제·자매, 배우자 형제·자매, 자녀, 조부모처럼 적는다. "
    "본인이 당사자면 '본인'이다. 해당 없으면 null.\n"
    "- finding: clarify일 때, 근거만으로 이미 확정할 수 있는 사실을 한두 문장으로 적는다. "
    "기한이 지났다거나 원칙은 무엇이고 어떤 예외가 남았는지처럼 사용자가 바로 알아야 할 내용이다. "
    "확정할 수 있는 것이 없으면 빈 문자열.\n"
    "JSON만 출력한다. 형식: "
    '{"verdict": "answerable|clarify|escalate", "reason": "한 문장", "missing": ["질문에 빠진 정보"], '
    '"finding": "이미 확정되는 사실", '
    '"intent": "ceremony|housing|relocation|trip|club|other", "relation": "관계 또는 null"}'
)


def judge_groundedness(question, evidence, history=None):
    """검색 근거만으로 답할 수 있는지 LLM에 판정을 맡깁니다."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or not evidence:
        return {"verdict": "escalate", "reason": "검색된 근거가 없습니다.", "missing": []}
    evidence_text = "\n\n".join(
        f"[근거 {index}] {item['file']} ({item.get('path', '')})\n{item['text'][:1200]}"
        for index, item in enumerate(evidence, 1)
    )
    # 이전 대화를 함께 넘깁니다. 판정이 자기가 무엇을 되물었는지 모르면 표현만 바꿔 같은 것을
    # 다시 묻고, 사용자가 답해도 대화가 끝나지 않습니다.
    prior = "\n".join(
        f"{'사용자' if item.get('role') == 'user' else '상담'}: {item.get('content', '')[:600]}"
        for item in (history or [])[-4:]
        if item.get("role") in ("user", "assistant") and item.get("content")
    )
    prior_text = f"이전 대화:\n{prior}\n\n" if prior else ""
    payload = {
        "model": MODEL,
        "reasoning": {"effort": "none"},
        # 신청기한처럼 경과일로 갈리는 판정에는 기준일이 있어야 합니다.
        "instructions": f"{GROUNDEDNESS_INSTRUCTIONS}\n오늘 기준일은 {date.today().isoformat()}이다.",
        "input": [{"role": "user", "content": f"{prior_text}질문: {question}\n\n검색된 규정 근거:\n{evidence_text}"}],
    }
    request = Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError):
        # 판정에 실패하면 단정하지 않고 담당 부서 확인으로 보냅니다.
        return {"verdict": "escalate", "reason": "근거 판정을 수행하지 못했습니다.", "missing": []}
    text = data.get("output_text") or ""
    if not text:
        for output in data.get("output", []):
            for content in output.get("content", []):
                if content.get("type") == "output_text":
                    text += content.get("text", "")
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {"verdict": "escalate", "reason": "판정 응답을 해석하지 못했습니다.", "missing": []}
    try:
        result = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"verdict": "escalate", "reason": "판정 응답을 해석하지 못했습니다.", "missing": []}
    if result.get("verdict") not in ("answerable", "clarify", "escalate"):
        result["verdict"] = "escalate"
    if result.get("intent") not in POLICY_INTENTS:
        # 의도를 못 뽑으면 규칙을 걸러내지 않고 기존 키워드 판별에 맡깁니다.
        result["intent"] = None
    result.setdefault("reason", "")
    result.setdefault("missing", [])
    result.setdefault("relation", None)
    result.setdefault("finding", "")
    return result


RESOLVE_INSTRUCTIONS = (
    "너는 사내 복리후생 상담의 질문 정리기다. 이전 대화와 사용자의 새 입력을 읽고, "
    "새 입력을 그것만 읽어도 이해되는 하나의 질문으로 다시 쓴다. 답변하지 말고 질문만 만든다.\n"
    "- 이전 대화에 나온 날짜·장소·금액·관계·이동 경로 등 판단에 필요한 사실을 새 질문에 옮겨 담는다.\n"
    "- 새 입력이 직전 답변의 확인 요청에 대한 대답이면, 무엇을 확인해 준 것인지 드러나게 쓴다.\n"
    "- 새 입력이 이전 대화와 무관한 새 질문이면 그대로 둔다. 이전 주제를 끌어오지 않는다.\n"
    "- 사실을 지어내지 않는다. 이전 대화에 없는 내용은 넣지 않는다.\n"
    "질문 문장 하나만 출력한다. 설명이나 따옴표를 붙이지 않는다."
)


def resolve_question(question, history):
    """후속 입력을 이전 대화의 사실을 담은 자립형 질문으로 다시 씁니다.

    낱말 목록으로 새 주제인지 판별하던 방식은 양방향으로 틀렸습니다. "해외출장입니다"는
    '출장'이 걸려 이력이 끊겼고, "부모님 상당했는데"는 아무것도 안 걸려 지난 주제가 남았습니다.
    후속인지 아닌지를 코드가 정하지 않고, 다시 쓴 질문 하나를 검색·판정·생성이 함께 씁니다.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not history or not api_key:
        return question
    conversation = [
        {"role": item["role"], "content": item["content"]}
        for item in history[-6:]
        if item.get("role") in ("user", "assistant") and item.get("content")
    ]
    if not conversation:
        return question
    conversation.append({"role": "user", "content": f"새 입력: {question}"})
    payload = {
        "model": MODEL,
        "reasoning": {"effort": "none"},
        "instructions": RESOLVE_INSTRUCTIONS,
        "input": conversation,
    }
    request = Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError):
        # 재작성에 실패하면 원문 질문으로 진행합니다. 맥락은 잃어도 답변 경로는 살립니다.
        return question
    text = (data.get("output_text") or "").strip()
    if not text:
        for output in data.get("output", []):
            for content in output.get("content", []):
                if content.get("type") == "output_text":
                    text += content.get("text", "")
    text = text.strip().strip('"')
    return text or question


class ConsultationState(TypedDict, total=False):
    """LangGraph가 상담 단계 사이에서 전달하는 최소 상태입니다."""

    question: str
    resolved: str
    history: list[dict]
    intent: str
    evidence: list[dict]
    analysis: dict
    answer: str
    ui_actions: list[str]


def resolve_question_node(state: ConsultationState):
    """이후 모든 단계가 함께 쓸 자립형 질문을 먼저 만듭니다."""
    return {"resolved": resolve_question(state["question"], state.get("history", []))}


def retrieve_policy_node(state: ConsultationState):
    """현재 제도 질문에 맞는 규정 근거를 검색합니다."""
    return {"evidence": retrieve(state.get("resolved") or state["question"])}


def analyze_question_node(state: ConsultationState):
    """검색 근거로 답할 수 있는지와 함께 제도 영역·대상 관계를 한 번에 뽑습니다."""
    question = state.get("resolved") or state["question"]
    return {"analysis": judge_groundedness(question, state.get("evidence", []), state.get("history", []))}


def rule_applies(analysis, *domains):
    """의도를 못 뽑았으면 기존 판별을 쓰고, 뽑았으면 해당 제도일 때만 규칙을 켭니다."""
    intent = (analysis or {}).get("intent")
    return intent is None or intent in domains


def apply_policy_rules_node(state: ConsultationState):
    """날짜·관계·금액처럼 규정으로 결정 가능한 항목을 우선 처리합니다."""
    question = state["question"]
    analysis = state.get("analysis") or {}
    relation = analysis.get("relation") or ""
    ceremony = rule_applies(analysis, "ceremony")
    # 관계를 못 뽑았으면 형제·자매로 단정하지 않습니다. "유형"의 '형'처럼 부분문자열만 걸린
    # 질문에 20만원 지급을 확정해 버리기 때문에, 판정이 관계를 명시했을 때만 규칙을 켭니다.
    sibling_case = ceremony and ("형제" in relation or "자매" in relation)
    marriage_answer = build_sibling_marriage_answer(question) if sibling_case else ""
    seungjungsang_answer = build_seungjungsang_answer(question) if ceremony else ""
    death_answer = build_death_answer(question) if ceremony else ""
    housing = rule_applies(analysis, "housing")
    housing_exclusion_answer = build_housing_exclusion_answer(question) if housing else ""
    housing_contract_change_answer = build_housing_contract_change_answer(question) if housing else ""
    relocation_answer = build_relocation_answer(question) if rule_applies(analysis, "relocation") else ""
    overseas_personal_return_answer = build_overseas_personal_return_answer(question) if rule_applies(analysis, "trip") else ""
    answer = marriage_answer or seungjungsang_answer or death_answer or housing_exclusion_answer or housing_contract_change_answer or relocation_answer or overseas_personal_return_answer
    if not answer:
        return {}
    if marriage_answer or seungjungsang_answer or death_answer:
        evidence = [{"file": "경조금 지급기준.md", "score": 1, "text": "경조금 지급기준"}]
    elif housing_exclusion_answer or housing_contract_change_answer:
        evidence = [{"file": "숙소지원금 운영 기준.md", "score": 1, "text": "숙소지원금 운영 기준"}]
    elif relocation_answer:
        evidence = [
            {"file": "숙소지원금 운영 기준.md", "score": 1, "text": "숙소지원금 운영 기준"},
            {"file": "여비관리기준.md", "score": 1, "text": "여비관리기준"},
        ]
    else:
        evidence = [{"file": "여비관리 FAQ.md", "score": 1, "text": "해외출장 종료 후 개인 일정 체류"}]
    return {"answer": answer, "evidence": evidence}


def choose_after_rules(state: ConsultationState):
    """결정 규칙으로 처리하지 못한 질문만 LLM 답변 단계로 보냅니다."""
    return "end" if state.get("answer") else "generate"


def build_clarify_answer(missing, finding=""):
    """확정된 사실을 먼저 알리고, 판정이 짚은 부족한 정보만 되묻습니다.

    선택지를 검색 상위 조항 제목으로 만들던 블록은 지웠습니다. 판정과 무관한 순위라
    교통비 문의에 "5.4 출장복명"이 올라오고, FAQ 파일의 리프인 "질문"·"답변"도 섞였습니다.
    무엇이 부족한지는 판정이 이미 missing으로 뽑아 두므로 그것만 되묻습니다.
    """
    if finding:
        lines = [f"{finding}\n"]
    else:
        lines = ["문의하신 제도는 확인했지만, 답변을 확정하려면 정보가 조금 더 필요합니다.\n"]
    if missing:
        lines.append("확인이 필요한 내용")
        lines.extend(f"- {item}" for item in missing[:4])
        lines.append("")
        closing = "위 내용을 알려주시면 해당 기준으로 안내해 드리겠습니다. "
    else:
        closing = "어떤 부분을 확인하고 싶은지 알려주시면 해당 기준으로 안내해 드리겠습니다. "
    lines.append(
        closing
        + "바로 담당자 확인이 필요하시면 아래 ‘담당자에게 문의하기’ 버튼으로 문의 메일 초안을 만들 수 있습니다."
    )
    return "\n".join(lines)


def build_escalation_answer(reason, evidence):
    """규정이 다루지 않는 문의는 단정하지 않고 담당 부서로 넘깁니다."""
    checked = sorted({item.get("file", "") for item in evidence if item.get("file")})
    lines = ["문의하신 내용은 현재 제공된 규정만으로 지원 여부를 확정할 수 없습니다.\n"]
    lines.append("확인 결과")
    if reason:
        lines.append(f"- 판정 사유: {reason}")
    if checked:
        lines.append(f"- 확인한 규정: {', '.join(Path(name).stem for name in checked)}")
    lines.append("- 판정: 주관 부서 확인 필요\n")
    lines.append(
        "아래 ‘담당자에게 문의하기’ 버튼을 누르면 지금까지의 문의 내용이 담긴 메일 초안을 만들 수 있습니다. "
        "최종 지원 여부는 노무관리 주관부서의 규정 검토를 거쳐 결정됩니다."
    )
    return "\n".join(lines)


def generate_answer_node(state: ConsultationState):
    """근거로 답할 수 있는지 먼저 판정하고, 답변·되묻기·이관으로 나눕니다."""
    # 재작성된 질문이 이전 대화의 사실을 이미 담고 있으므로 이력을 낱말로 끊지 않습니다.
    question = state.get("resolved") or state["question"]
    history = state.get("history", [])
    evidence = state.get("evidence", [])
    if not evidence:
        answer = build_unknown_policy_answer(question)
    else:
        judgement = state.get("analysis") or judge_groundedness(question, evidence, history)
        verdict = judgement["verdict"]
        if verdict == "clarify":
            answer = build_clarify_answer(judgement.get("missing", []), judgement.get("finding", ""))
        elif verdict == "escalate":
            # 복리후생과 무관한 질문은 담당 부서로 넘기지 않고 상담 범위를 안내합니다.
            if judgement.get("intent") == "other":
                answer = build_clarification_answer(question)
            else:
                answer = build_escalation_answer(judgement.get("reason", ""), evidence)
        else:
            answer = call_openai(question, evidence, history)
    result = {"answer": answer}
    # 동호회 신규 신청 문의에는 답변 경로와 무관하게 메일 초안 버튼을 띄웁니다.
    if is_club_application_question(question):
        result["ui_actions"] = ["club_application_draft"]
    return result




def build_consultation_graph():
    """상담 요청을 분류·검색·규칙판정·생성으로 연결한 LangGraph를 만듭니다."""
    graph = StateGraph(ConsultationState)
    graph.add_node("resolve", resolve_question_node)
    graph.add_node("retrieve", retrieve_policy_node)
    graph.add_node("analyze", analyze_question_node)
    graph.add_node("rules", apply_policy_rules_node)
    graph.add_node("generate", generate_answer_node)
    # 키워드로 미리 거르지 않고 모든 질문을 검색·판정에 태웁니다.
    graph.add_edge(START, "resolve")
    graph.add_edge("resolve", "retrieve")
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze", "rules")
    graph.add_conditional_edges("rules", choose_after_rules, {"end": END, "generate": "generate"})
    graph.add_edge("generate", END)
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


def build_club_application_draft():
    """동호회 신규 신청에 필요한 정보를 채울 수 있는 담당자 메일 초안을 만듭니다."""
    body = (
        "안녕하세요. 노사발전그룹 동호회 담당자님,\n\n"
        "아래와 같이 동호회 등록을 신청합니다.\n\n"
        "- 동호회 회장: [입력]\n"
        "- 동호회 총무: [입력]\n"
        "- 동호회 명: [입력]\n"
        "- 종목(스포츠, 문화 등): [입력]\n"
        "- 동호회 회칙 작성 및 게시판 등록: [완료 / 미완료]\n"
        "- 총무 계좌번호 동호회 시스템 입력: [완료 / 미완료]\n"
        "- 비고: [입력]\n\n"
        "신청 내용을 확인해 주시고, 등록 절차 및 추가 제출자료가 있으면 안내 부탁드립니다.\n\n"
        "감사합니다.\n[신청자명] 드림"
    )
    return {
        "recipient": "노사발전그룹 동호회 담당자",
        "recipient_email": "",
        "subject": "[동호회 신청] [동호회명] 등록 신청",
        "body": body,
        "evidence": [{"file": "동호회 관리 규정.md", "score": 1, "text": "동호회 신청 양식 및 담당자 안내"}],
    }


class Handler(SimpleHTTPRequestHandler):
    """정적 화면과 상담 요청을 함께 제공하는 간단한 HTTP 핸들러입니다."""

    def do_POST(self):
        if self.path not in ("/api/chat", "/api/inquiries/draft", "/api/club-applications/draft", "/api/inquiries/send", "/api/inquiries/save", "/api/inquiries/delete"):
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/api/club-applications/draft":
                self.respond(200, build_club_application_draft())
                return
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
            self.respond(200, {
                "answer": result["answer"],
                "evidence": result.get("evidence", []),
                "ui_actions": result.get("ui_actions", []),
            })
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
            content_type = "text/markdown" if candidate.suffix.lower() == ".md" else "text/plain"
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
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
