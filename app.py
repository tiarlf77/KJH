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
    # 사규에는 없지만 사내에서 운영해 온 기준. 프롬프트에 박혀 있던 것을 옮겼습니다.
    "사내 추가 기준.md",
}
# 규정의 '회갑'과 사용자가 자주 쓰는 '환갑'을 같은 의미로 처리합니다.
HOEGAP_TERMS = ("회갑", "환갑")
FINAL_REVIEW_NOTICE = "안내 내용은 사전 참고용이며, 최종 지급·승인 여부는 담당 부서의 확인 후 결정됩니다."
DISPATCH_DAILY_ALLOWANCE = 33_000
LODGING_PER_NIGHT = 50_000
FULL_RATE_DAYS = 14



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
            # 제목을 만나면 지금까지 모은 본문을 지금 경로로 확정합니다. 더 깊은 제목이
            # 올 때 부모 서두를 들고 내려가면 그 본문이 맨 아래 리프의 것이 됩니다.
            # 별첨 1의 국내여비기준표가 실제로 '별첨 1 > 주석 > 유류비 산식' 청크에
            # 흡수돼 있었고, 경로를 본문과 함께 임베딩하므로 벡터까지 유류비 쪽으로
            # 끌려가 숙박비 질문에서 20위 밖으로 밀렸습니다.
            body = "\n".join(current_lines).strip()
            if body and headings:
                chunks.append({"path": " > ".join(name for _, name in headings), "text": body})
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
# 2-gram 가중치. 검색어 주입 블록을 지운 뒤 다시 재니 0.5가 최선입니다(그 전에는 1.0).
# 주입이 살아 있을 때는 넣은 낱말이 어절로 안 맞아 2-gram이 보정을 하고 있었습니다.
# 주입을 지운 지금은 2-gram을 어절과 같은 무게로 두면 조사·어미 조각이 잡음이 됩니다.
BIGRAM_WEIGHT = 0.5
# 최고 점수 파일 대비 이 비율 이상인 규정만 함께 근거로 사용합니다.
# 0.0~1.0을 훑으면 0.7이 가장 좋습니다(44문항 MRR 0.786, 두 세트 모두 미검색 0건).
# 높일수록 좋아지는 이유는 이 컷이 limit=20 자리 배분보다 먼저 일어나기 때문입니다.
# 파일을 많이 남기면 스무 자리가 여러 규정에 흩어져 정답 파일의 조항이 밀려납니다.
# 0.85를 넘기면 반대로 나빠지고, 1.0에서는 사실상 한 파일만 남아 미검색이 5건이 됩니다.
TOP_FILE_SCORE_RATIO = 0.7


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
# 검색어 주입 3블록(QUERY_SYNONYMS·CANONICAL·숙소)을 지운 뒤 BIGRAM_WEIGHT와 함께
# 격자로 다시 훑어 0.85/0.5를 골랐습니다. 0.8/1.0 대비 두 세트 모두 올라갑니다.
#     0.80/1.0  44문항 hit@1 70.5% MRR 0.786 | 실사용 hit@1 55.6% MRR 0.727
#     0.85/0.5  44문항 hit@1 70.5% MRR 0.789 | 실사용 hit@1 61.1% MRR 0.764
# 0.9 이상은 두 세트 모두 다시 나빠집니다. "경조휴가 중 결근" 한 문항이 7위에서 9위로
# 밀리지만 상위 20 안이고, 실사용 세트(상담 로그에서 옮긴 문항)의 이득이 훨씬 큽니다.
VECTOR_WEIGHT = 0.85


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
# 이 개수 이하의 청크에만 나오는 낱말을 종목어로 봅니다. 낚시·헬스·등산·볼링은 각 1개입니다.
# 0·1·2·3 어느 값이든 62문항 지표는 같습니다. 이 값은 정확도를 올리는 노브가 아니라
# 근거 자리 잠식을 막는 상한입니다.
#
# 이 규칙의 값어치는 62문항으로 감시되지 않습니다. 평가셋에 종목 질문이 하나도 없어
# 껐다 켜도 지표가 움직이지 않습니다. 실제 근거는 test_rules E그룹뿐이고, 그중에서도
# 끄면 깨지는 것은 "낚시 동호회 개설 가능한가요?" 한 문항입니다("헬스"는 규칙 없이도
# 통과). 평가셋에 종목 질문을 넣어 제대로 재기 전까지는 이 규칙을 지표로 판단할 수 없습니다.
RARE_TERM_MAX_CHUNKS = 2


# 낱말로 근거 파일을 미리 좁히던 `focused_evidence_files`는 지웠습니다. 측정하면 전 지표가
# 나빠집니다(44문항 MRR 0.760 -> 0.685, 미검색 2 -> 6건). "출장 가면 숙소는 회사가
# 잡아주나요?"는 '숙소'가 걸려 숙소지원금 기준에 갇히지만 답은 여비관리기준에 있습니다.
# 제도 섞임은 낱말이 아니라 점수로 가리는 TOP_FILE_SCORE_RATIO가 이미 맡고 있습니다.
# 후보를 지우는 키워드 규칙은 해롭고, 더하는 확장(rare_query_terms)은 값을 합니다.


def retrieve(question, limit=20):
    """Markdown 제목 청크와 기존 텍스트 규정에서 관련 근거를 찾아 반환합니다."""
    query_tokens = tokens(question)
    # 아래에서 넓히기 전의 원문 토큰. 희소 종목어 판정에 씁니다.
    question_tokens = set(query_tokens)
    # 검색어 주입 두 블록을 지웠습니다. 62문항으로 껐다 켜며 재면 이렇습니다.
    #   결혼·해외출장 주입: 기여 없음(지표 동일)
    #   숙소 주입:         해로움. 끄면 44문항 MRR 0.762 -> 0.784, 실사용 0.676 -> 0.727
    # 숙소 블록은 5.5 지원특례 하나를 끌어올리려고 "3개월"·"발령" 같은 흔한 낱말을 넣었는데,
    # 그 낱말을 담은 무관한 조항까지 전부 점수를 얻어 주변 순위를 흐렸습니다.
    # 호칭 동의어 딕셔너리(QUERY_SYNONYMS)는 지웠습니다. 62문항에서 켠 것과 끈 것의 지표가
    # 소수점까지 같았습니다. VECTOR_WEIGHT를 0.8로 올린 뒤로는 임베딩이 "장인어른 ~ 배우자
    # 부모"를 직접 알아 딕셔너리가 메우던 자리가 없습니다. 손으로 호칭을 채워 넣는 유지보수만
    # 남아 있었습니다(외할매를 추가하던 커밋 0323b1c). VECTOR_WEIGHT를 다시 낮춘다면
    # 이 확장이 필요해질 수 있습니다.
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
    # 질문에만 등장하는 희소 종목어가 조항 본문에 있으면, 의미 유사도 순위와 관계없이
    # 해당 조항을 근거에 남깁니다. 예: "낚시 동호회", "헬스 동아리".
    # 두 가지로 좁힙니다. 첫째, 넓히기 전 원문 토큰만 봅니다. 주입한 검색어("3개월"·"발령")
    # 까지 희소어로 잡으면 그 낱말을 담은 청크가 근거를 채웁니다. 둘째, 코퍼스에서 몇 개
    # 청크에만 나오는 낱말로 제한합니다. idf만 보면 "해서"(8청크)·"신청"(9청크) 같은 구어가
    # 걸려 20자리를 상한 없이 잠식하고, 정작 답이 있는 조항이 밀려납니다. 실제로 숙소지원금
    # 문항 2건이 그렇게 미검색으로 떨어졌습니다. 이 규칙이 노리는 종목어(낚시·헬스·등산·볼링)는
    # 모두 1개 청크에만 나오므로, 청크 수로 막으면 잠식 폭이 구조적으로 묶입니다.
    rare_query_terms = {
        term for term in question_tokens
        if len(term) >= 2
        and sum(1 for chunk in chunks if term in chunk["text"]) <= RARE_TERM_MAX_CHUNKS
    }
    for item in results:
        if not any(term in item["text"] for term in rare_query_terms):
            continue
        selected.append(item)
        taken.add((item["file"], item["path"]))
    # 여러 규정이 함께 적용될 수 있으므로 규정별 상위 근거를 먼저 확보합니다.
    for path in sorted({item["file"] for item in results}):
        for item in [item for item in results if item["file"] == path][:3]:
            if (item["file"], item["path"]) in taken:
                continue
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
    return attach_referenced_chunks(selected[:limit], chunks)


# "별첨 1에 의한다", "제5.11에서 규정한" 처럼 규정이 스스로 가리키는 참조 표현입니다.
REFERENCE_PATTERN = re.compile(r"별첨\s*(\d+)|제\s*(\d+\.\d+)")
# 참조를 따라갈 상위 근거 수. 아래쪽 근거까지 따라가면 질문과 먼 별첨이 붙습니다.
REFERENCE_SOURCE_LIMIT = 5
# 한 청크가 이보다 많은 번호를 담고 있으면 가리키는 것이 아니라 나열하는 것으로 봅니다.
REFERENCE_MARKER_LIMIT = 2
# 참조 하나당, 그리고 한 질문당 붙일 수 있는 근거 수.
REFERENCE_CHUNK_LIMIT = 2
REFERENCE_TOTAL_LIMIT = 4


# 사규 원본이 아닌 보조 문서입니다. 감추지 않고 구분해서 보여줍니다.
def attach_referenced_chunks(selected, chunks):
    """근거가 가리키는 별첨·조항을 함께 붙입니다.

    5.10.2는 본문이 "소액경비 및 숙박료 지급기준은 별첨 1에 의한다"뿐이라 금액이 없습니다.
    검색이 이 조항을 1위로 올려도 판정은 근거에 답이 없다고 보고 이관으로 떨어뜨립니다.
    규정이 명시한 링크만 따라가므로 순위 계산에는 손대지 않고 근거에만 더합니다.
    참조가 다시 참조를 물고 오지 않도록 1홉만 따라갑니다.
    """
    if not selected:
        return selected
    taken = {(item["file"], item["path"]) for item in selected}
    targets = []
    for item in selected[:REFERENCE_SOURCE_LIMIT]:
        markers = {
            f"별첨 {attachment}." if attachment else f"{clause} "
            for attachment, clause in REFERENCE_PATTERN.findall(item["text"])
        }
        # 별첨 목록표('6. 기록 및 첨부')처럼 번호를 죽 늘어놓기만 하는 청크는 가리키는
        # 것이 아니라 나열하는 것입니다. 따라가면 별첨 전체가 근거로 딸려옵니다.
        if len(markers) > REFERENCE_MARKER_LIMIT:
            continue
        # 같은 규정 안의 참조만 따라갑니다. 파일이 다르면 번호가 겹쳐도 다른 문서입니다.
        targets.extend((item["file"], marker) for marker in markers)
    added = []
    for file_name, marker in targets:
        hits = [
            chunk for chunk in chunks
            if chunk["file"] == file_name
            and marker in chunk["path"]
            and (chunk["file"], chunk["path"]) not in taken
        ]
        # 별첨 아래 하위 절이 많으면 표가 있는 상위부터 가져옵니다.
        hits.sort(key=lambda chunk: len(chunk["path"]))
        for chunk in hits[:REFERENCE_CHUNK_LIMIT]:
            taken.add((chunk["file"], chunk["path"]))
            # 참조로 딸려온 근거는 검색 점수가 아니라 출처가 근거이므로 0점으로 둡니다.
            added.append({
                "file": chunk["file"],
                "score": 0.0,
                "keyword_score": 0.0,
                "vector_score": 0.0,
                "path": chunk["path"],
                "text": chunk["text"][:3000],
                "referenced": True,
            })
    return selected + added[:REFERENCE_TOTAL_LIMIT]


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


def append_final_review_notice(answer):
    """모든 상담 답변에 최종 담당 부서 확인 안내를 한 번만 덧붙입니다."""
    answer = answer.strip()
    if FINAL_REVIEW_NOTICE in answer:
        return answer
    return f"{answer}\n\n{FINAL_REVIEW_NOTICE}"


def extract_dispatch_days(question):
    """자연어 질문에서 파견기간을 일수로 읽습니다. 한 달은 계산 기준상 30일로 봅니다."""
    day_match = re.search(r"(?<!\d)(\d{1,4})\s*일(?:간)?", question)
    if day_match:
        return int(day_match.group(1))
    month_match = re.search(r"(?<!\d)(\d{1,2})\s*(?:개월|달)", question)
    if month_match:
        return int(month_match.group(1)) * 30
    return None


def extract_lodging_nights(question):
    """숙박비 계산에 필요한 실제 숙박일수를 읽습니다."""
    match = re.search(r"(?<!\d)(\d{1,4})\s*박", question)
    return int(match.group(1)) if match else None


def is_dispatch_calculation_question(question):
    """파견 기간별 금액을 묻는 경우에만 결정론적 계산 경로를 사용합니다."""
    calculation_words = ("계산", "얼마", "금액", "지급", "파견경비", "숙박비", "숙박료", "파견비")
    return "파견" in question and (
        extract_dispatch_days(question) is not None or any(word in question for word in calculation_words)
    )


def company_lodging_status(question):
    """서울 파견에서 회사 숙소 제공 여부를 부정 표현부터 판별합니다."""
    not_provided_terms = (
        "숙소 미제공", "숙소를 제공하지", "숙소 제공하지", "숙소 제공 안",
        "회사 숙소 없음", "회사 숙소 없",
    )
    if any(term in question for term in not_provided_terms):
        return False

    provided_terms = (
        "회사 숙소 제공", "회사에서 숙소 제공", "회사가 숙소 제공",
        "숙소를 제공", "숙소 제공받", "숙소 제공",
    )
    if any(term in question for term in provided_terms):
        return True
    return None


def build_dispatch_calculation_answer(question):
    """확정된 장기 파견 지급률로 파견경비와 숙박비를 계산합니다."""
    if not is_dispatch_calculation_question(question):
        return None

    days = extract_dispatch_days(question)
    if days is None:
        return "파견경비와 숙박비를 계산하려면 파견기간을 ‘90일’ 또는 ‘3개월’처럼 알려주세요."
    if days < 1:
        return "파견기간은 1일 이상으로 알려주세요."

    is_seoul = "서울" in question
    company_lodging = company_lodging_status(question) if is_seoul else False
    if is_seoul and company_lodging is None:
        return "서울 파견은 회사가 숙소를 제공하는지에 따라 계산 방식이 달라집니다. 회사가 숙소를 제공하는지 알려주세요."

    full_rate_days = min(days, FULL_RATE_DAYS)
    reduced_rate_days = max(days - FULL_RATE_DAYS, 0)
    full_dispatch = DISPATCH_DAILY_ALLOWANCE * full_rate_days

    if is_seoul and company_lodging:
        reduced_dispatch = DISPATCH_DAILY_ALLOWANCE * 80 // 100 * reduced_rate_days
        dispatch_total = full_dispatch + reduced_dispatch
        lines = [
            f"서울 파견(회사 숙소 제공) {days}일 기준 예상 금액입니다.",
            "",
            "계산 결과",
            f"- 1~{full_rate_days}일 파견경비: {DISPATCH_DAILY_ALLOWANCE:,}원 × {full_rate_days}일 = {full_dispatch:,}원",
        ]
        if reduced_rate_days:
            lines.append(
                f"- {FULL_RATE_DAYS + 1}~{days}일 파견경비: {DISPATCH_DAILY_ALLOWANCE:,}원 × 80% × {reduced_rate_days}일 = {reduced_dispatch:,}원"
            )
        lines.extend((
            "- 숙박비: 회사 숙소 제공으로 0원",
            f"- 총 예상 지급액: {dispatch_total:,}원",
            "",
            "교통비와 파견지에서 발생한 별도 출장비는 포함하지 않은 금액입니다.",
        ))
        return "\n".join(lines)

    nights = extract_lodging_nights(question)
    if nights is None:
        return f"파견기간은 {days}일로 확인됩니다. 숙박비까지 계산하려면 실제 숙박일수를 ‘{days}박’처럼 알려주세요."
    if nights < 1:
        return "실제 숙박일수는 1박 이상으로 알려주세요."

    full_rate_nights = min(nights, FULL_RATE_DAYS)
    reduced_rate_nights = max(nights - FULL_RATE_DAYS, 0)
    reduced_dispatch = DISPATCH_DAILY_ALLOWANCE * 60 // 100 * reduced_rate_days
    full_lodging = LODGING_PER_NIGHT * full_rate_nights
    reduced_lodging = LODGING_PER_NIGHT * 60 // 100 * reduced_rate_nights
    dispatch_total = full_dispatch + reduced_dispatch
    lodging_total = full_lodging + reduced_lodging
    location_label = "서울 파견(회사 숙소 미제공)" if is_seoul else "일반 지역 파견"
    lines = [
        f"{location_label} {days}일·숙박 {nights}박 기준 예상 금액입니다.",
        "",
        "계산 결과",
        f"- 1~{full_rate_days}일 파견경비: {DISPATCH_DAILY_ALLOWANCE:,}원 × {full_rate_days}일 = {full_dispatch:,}원",
        f"- 1~{full_rate_nights}박 숙박비: {LODGING_PER_NIGHT:,}원 × {full_rate_nights}박 = {full_lodging:,}원",
    ]
    if reduced_rate_days:
        lines.append(
            f"- {FULL_RATE_DAYS + 1}~{days}일 파견경비: {DISPATCH_DAILY_ALLOWANCE:,}원 × 60% × {reduced_rate_days}일 = {reduced_dispatch:,}원"
        )
    if reduced_rate_nights:
        lines.append(
            f"- {FULL_RATE_DAYS + 1}~{nights}박 숙박비: {LODGING_PER_NIGHT:,}원 × 60% × {reduced_rate_nights}박 = {reduced_lodging:,}원"
        )
    lines.extend((
        f"- 파견경비 합계: {dispatch_total:,}원",
        f"- 숙박비 합계: {lodging_total:,}원",
        f"- 총 예상 지급액: {dispatch_total + lodging_total:,}원",
        "",
        "교통비와 파견지에서 발생한 별도 출장비는 포함하지 않은 금액입니다.",
    ))
    return "\n".join(lines)



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
        "답변 마지막에는 '확인한 규정'과 파일명을 간단히 표시한다. 최종 문장은 '안내 내용은 사전 참고용이며, 최종 지급·승인 여부는 담당 부서의 확인 후 결정됩니다.'로 작성한다. "
        "규정에 명시된 예외만 적용하고, 규정에 없는 예외나 담당자 재량은 사용자에게 확인 질문으로 남긴다. "
        f"오늘 기준일은 {date.today().isoformat()}이다. 경조금은 회갑 대상 여부와 신청기한을 별도로 판단한다. "
        # 청구권 3개월·회갑 대상·지급액 20만원은 경조금 지급기준.md에 이미 있고 검색이 항상
        # 찾아온다(eval_retrieval.py 미검색 0건). 근거만 주고 되물었더니 그대로 맞혀 지웠다.
        "계산된 신청 마감일이 제공되면 그 날짜를 사용한다."
        " 부모·배우자 부모의 회갑 질문에서 생년월일이 없으면 연령을 추측하거나 지원을 확정하지 않는다. "
        "답변은 다음 흐름으로 작성한다: '해당 부모가 본인 부모인지 배우자 부모인지 관계 요건은 확인되지만, 현재 정보만으로 실제 회갑 대상인지 판단할 수 없다'고 먼저 설명한다. "
        "그 다음 현재 기준일을 알려주고 생년월일을 YYYYMMDD 형식으로 요청한다. 지급액과 신청기한은 회갑 대상 판정 이후에 안내한다. "
        "단, 사용자가 '우리 엄마', '우리 아버지', '우리 부모님'이라고 표현하면 별도 배우자 표현이 없는 한 본인 부모로 이해하고 관계를 다시 묻지 않는다. "
        "직전 대화에서 생년월일과 관계가 이미 확인되었으면 같은 질문을 반복하지 말고 회갑일 계산 결과를 안내한다. "
        "사용자가 결혼·사망·출산·동호회·숙소·출장 등 다른 제도를 새로 질문하면 이전 회갑 대화와 분리해 현재 질문의 의도와 규정만 사용한다. "
        "현재 질문과 무관한 제도는 비교·설명·언급하지 않는다. 예를 들어 형제자매 결혼 질문에는 회갑, 숙소, 동호회, 여비를 언급하지 않는다. "
        "결혼·회갑·출산장려금·사망 등 경조금 답변은 반드시 공통 양식을 따른다. 첫 줄에는 해당 여부를 한 문장으로만 답한다. "
        "이후 빈 줄 뒤에 '확인 결과'를 쓰고, '- 관계:', '- 생년월일:'(확인된 경우만), '- 사유 발생일:'(확인된 경우만), "
        "'- 신청 마감일:'(계산 가능한 경우만), '- 현재 기준일:', '- 판정:', '- 지원금:', '- 필요 서류:' 순서로 필요한 항목만 한 번씩 작성한다. "
        "그 뒤 빈 줄 뒤에 신청 가능 여부 또는 추가로 필요한 정보만 한두 문장으로 쓰고, 마지막 문장은 '안내 내용은 사전 참고용이며, 최종 지급·승인 여부는 담당 부서의 확인 후 결정됩니다.'로 끝낸다. "
        "같은 내용을 반복하거나 현재 질문과 무관한 제도를 언급하지 않는다. "
        "사망 문의에서는 첫 문장을 '경조금 지급 대상입니다.'처럼 간단히 작성하고 '해당됩니다'로 시작하지 않는다. "
        "사망일을 아직 받지 못한 경우 사망일로부터 3개월 이내 신청해야 한다는 문장과 사망일 요청 문장을 쓰지 않는다. "
        "사망 문의에서는 화환·조화 지원 여부를 별도 부연 문장으로 설명하지 않는다. "
        # 담당 업체 연락처와 아래 세 건은 '사내 추가 기준.md'로 옮겼습니다. 규정에 없는 값을
        # 지시문에 적어 두면 근거 없이 나오고, 근거로 표시되지도 않아 환각과 구분되지 않습니다.

        # 숙소지원금 운영 기준 5.5 지원특례(3개월 실비, 최장 6개월 연장, 청소비 제외)도
        # 위와 같은 이유로 지웠다. 근거는 항상 검색되고(eval_retrieval.py), 근거만 줬을 때도
        # 3개월/6개월/청소비 제외까지 정확히 재현했다.
        "회갑일이 지났더라도 사유 발생일로부터 3개월 이내이면 신청 가능하다. "
        "계산 결과가 신청 마감일 이후이면 '확정하기 어렵다', '추가 확인 필요' 같은 유보 표현을 쓰지 않는다. "
        "이 경우에는 회갑일·신청 마감일·청구권 소멸을 2~3문장으로 간단히 안내한다."
        " 근무지역 이동 시 '숙소 정리비'라는 별도 지원금 표현을 쓰지 말고, '전 근무지 숙소 정리 기간에 발생한 비용'으로 안내한다."
        # 국내출장 금액 기준은 규정에 있으므로 여기에 옮겨 적지 않는다. 옮겨 적었던 문장은
        # 5.14.2의 거리 조건(왕복 120km/50km/미만)을 떼고 단정했고, "식비 1일 3만원"은
        # 5.14.3 자가차량 동승자 기준을 일반 기준으로 잘못 옮긴 것이었다.
        # 동호회 정기지원 분기 인당 3만원·최대 30만원도 같은 이유로 지웠다. 동호회 관리
        # 규정.md에 이미 있고, 근거만 줬을 때도 정확히 재현했다.
        " 사용자가 이미 말한 사실(관계, 날짜, 근무지, 이사 여부, 동거 여부, 질문 항목)은 다시 묻지 않는다."
        " 발령·부임·이사·부임비·이전비·가족 잔류·혼자 거주 표현이 있으면 부임 및 숙소지원금 복합 문의로 파악한다."
        " 이 경우 실제 이사하지 않으면 부임비 지급이 없다는 점을 먼저 답하고, 숙소지원금은 새 근무지·실제 단신 거주·주택 보유 여부·임대차계약서를 기준으로 필요한 만큼만 안내한다."
        " 가족 명의(본인·배우자·부모·자녀·형제자매) 건물에 전세 또는 월세로 거주하는 경우와 단신부임으로 신청한 뒤 실제 동거인이 있는 사실이 확인된 경우에만 숙소지원금 지원 불가로 안내한다."
        " 주민등록상 주소 등록과 실제 동거를 동일한 사실로 추정하지 않는다. 주소만 전입했고 실제 동거하지 않는다고 사용자가 밝히면 자동 제외하지 않으며, 실제 동거 여부가 불명확하면 그 사실만 확인한다."
        " 허위 신청이나 신청 내용과 실제 거주 형태의 불일치가 확인된 경우에만 전세금·월세 지원금 계산을 하지 말고, '신청 내용이나 실제 거주 형태가 사실과 다르면 윤리위반으로 감사 대상이 될 수 있습니다.'라고 짧게 경고한다. 주소만 전입했거나 실제 동거 여부가 불명확한 단계에서는 이 경고를 조건부 설명으로도 언급하지 않는다."
        " 기존 숙소지원금 수급 중 월세와 전세의 계약 형태가 바뀌면 변경된 임대차계약서와 계약조건 확인 자료를 새로 제출하도록 안내한다."
        # 전세 전환 시 지원액(전세금 1,000만원당 월 10만원)은 숙소지원금 운영 기준.md 5.1에
        # 이미 있고, 근거만 줬을 때도 동일하게 답해 지웠다.
        " 계약 전환의 세부 제출서류와 적용 시점은 숙소지원금 담당자에게 문의하도록 답변을 끝낸다."
        " 해외출장 또는 교육연수의 국내공항·국내 주차비는 지급하지 않는다고 명확히 안내한다. 회사 차량·개인 차량 여부에 따른 예외를 만들지 않는다."
        # 일반 국내출장 주차비를 "현지교통비 2만원 범위에서 처리"로 단정하던 문장은 지웠고
        # '사내 추가 기준.md'로도 옮기지 않았다. 여비관리기준 165행이 "회사 차량 이용 시 발생한
        # 주유비, 주차비 등은 기타 경비 항목으로 지급할 수 있다"고 정해 정면으로 충돌한다.
        # 규정과 어긋나는 주장을 승인된 근거 파일에 올리면 환각을 근거로 승격시키는 셈이다.

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

# 판정기가 근거 번호를 넓게 고르더라도 현재 제도와 직접 관련된 문서만 사용자에게 표시합니다.
# 부임은 여비와 숙소 기준이 함께 적용될 수 있어 두 문서군을 모두 허용합니다.
INTENT_EVIDENCE_FILES = {
    "ceremony": {"경조금 지급기준.md", "사내 추가 기준.md"},
    "housing": {"숙소지원금 운영 기준.md", "사내 추가 기준.md"},
    "relocation": {"여비관리기준.md", "숙소지원금 운영 기준.md", "사내 추가 기준.md"},
    "trip": {"여비관리기준.md", "여비관리 FAQ.md", "사내 추가 기준.md"},
    "club": {"동호회 관리 규정.md"},
}
# 선택된 문서의 직접 어휘 점수가 최고 문서의 80%보다 낮으면 같은 제도 문서라도 표시하지 않습니다.
USED_EVIDENCE_KEYWORD_RATIO = 0.8
# 질문에 업무 항목이 명시된 경우에는 점수보다 해당 표현이 실제 근거에 있는지를 우선합니다.
EVIDENCE_QUERY_ANCHORS = {
    "주차": ("주차",),
    "숙소지원금": ("숙소지원금", "숙소 지원"),
    "부임비": ("부임비",),
    "이전비": ("이전비", "이전료"),
    "경조금": ("경조금",),
    "조의금": ("조의금", "사망"),
    "동호회": ("동호회",),
    "동아리": ("동호회", "동아리"),
}


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
    "missing의 각 항목은 {\"label\": 되물을 내용, \"options\": 흔한 답 후보} 형태다. "
    "options는 사용자가 클릭 한 번으로 답할 수 있도록 짧은 후보를 2~4개 넣는다. "
    "예를 들어 관계를 물으면 ['본인 부모', '배우자 부모'], 출장 형태를 물으면 "
    "['당일출장', '1박 이상 출장']처럼 넣는다. 생년월일·금액·날짜처럼 사람마다 값이 달라 "
    "후보를 미리 정할 수 없으면 options는 빈 배열로 둔다. 근거에 없는 후보를 지어내지 않는다.\n"
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
    "- evidence_ids: 판정·finding에 직접 사용한 근거 번호만 정수 배열로 적는다. 질문과 같은 제도라는 "
    "이유만으로 넣지 말고, 실제 판단 문장을 뒷받침하는 최소한의 근거만 고른다. 직접 근거가 없으면 빈 배열이다.\n"
    "JSON만 출력한다. 형식: "
    '{"verdict": "answerable|clarify|escalate", "reason": "한 문장", '
    '"missing": [{"label": "질문에 빠진 정보", "options": ["짧은 답 후보"]}], '
    '"finding": "이미 확정되는 사실", '
    '"intent": "ceremony|housing|relocation|trip|club|other", "relation": "관계 또는 null", '
    '"evidence_ids": [1, 2]}'
)


def judge_groundedness(question, evidence, history=None):
    """검색 근거만으로 답할 수 있는지 LLM에 판정을 맡깁니다."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or not evidence:
        return {"verdict": "escalate", "reason": "검색된 근거가 없습니다.", "missing": [], "evidence_ids": []}
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
        return {"verdict": "escalate", "reason": "근거 판정을 수행하지 못했습니다.", "missing": [], "evidence_ids": []}
    text = data.get("output_text") or ""
    if not text:
        for output in data.get("output", []):
            for content in output.get("content", []):
                if content.get("type") == "output_text":
                    text += content.get("text", "")
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {"verdict": "escalate", "reason": "판정 응답을 해석하지 못했습니다.", "missing": [], "evidence_ids": []}
    try:
        result = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"verdict": "escalate", "reason": "판정 응답을 해석하지 못했습니다.", "missing": [], "evidence_ids": []}
    if result.get("verdict") not in ("answerable", "clarify", "escalate"):
        result["verdict"] = "escalate"
    if result.get("intent") not in POLICY_INTENTS:
        # 의도를 못 뽑으면 규칙을 걸러내지 않고 기존 키워드 판별에 맡깁니다.
        result["intent"] = None
    result.setdefault("reason", "")
    result.setdefault("relation", None)
    result.setdefault("finding", "")
    # 판정이 지시를 안 따르고 missing을 예전처럼 문자열로 줄 수도 있어 형태를 맞춘다.
    normalized = []
    for item in result.get("missing", []):
        if isinstance(item, dict):
            label = str(item.get("label", "")).strip()
            options = [str(option).strip() for option in item.get("options", []) if str(option).strip()]
        else:
            label = str(item).strip()
            options = []
        if label:
            normalized.append({"label": label, "options": options[:4]})
    result["missing"] = normalized
    result.setdefault("evidence_ids", [])
    return result


def select_used_evidence(question, candidate_evidence, judgement):
    """검색 후보 중 판정에 직접 사용한 근거만 사용자 표시용으로 남깁니다.

    판정기의 번호는 신뢰하지 않고 현재 후보 범위의 정수만 허용합니다. 번호가 없거나 모두
    잘못됐으면 검색 1위와 같은 파일의 근거만 남겨, 후보 전체가 화면에 노출되지 않게 합니다.
    """
    intent = judgement.get("intent")
    allowed_files = INTENT_EVIDENCE_FILES.get(intent)
    eligible = [
        item for item in candidate_evidence
        if not allowed_files or item.get("file") in allowed_files
    ]
    if not eligible:
        eligible = list(candidate_evidence)

    # 같은 제도 안에서도 질문의 직접 어휘와 거리가 먼 문서는 표시 근거에서 제외합니다.
    # 의미 검색 후보는 판정용으로 유지하되, 사용자에게 보이는 출처에는 더 엄격한 기준을 씁니다.
    best_keyword_by_file = {}
    for item in eligible:
        file_name = item.get("file")
        best_keyword_by_file[file_name] = max(
            best_keyword_by_file.get(file_name, 0.0),
            float(item.get("keyword_score", 0.0)),
        )
    top_keyword = max(best_keyword_by_file.values(), default=0.0)
    anchor_terms = {
        evidence_term
        for query_term, evidence_terms in EVIDENCE_QUERY_ANCHORS.items()
        if query_term in question
        for evidence_term in evidence_terms
    }
    anchored_files = {
        item.get("file")
        for item in eligible
        if anchor_terms and any(term in f"{item.get('path', '')} {item.get('text', '')}" for term in anchor_terms)
    }
    strong_files = anchored_files or {
        file_name for file_name, score in best_keyword_by_file.items()
        if not top_keyword or score >= top_keyword * USED_EVIDENCE_KEYWORD_RATIO
    }

    selected = []
    seen = set()
    raw_ids = judgement.get("evidence_ids", [])
    if not isinstance(raw_ids, list):
        raw_ids = []
    for raw_id in raw_ids:
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            continue
        index = raw_id - 1
        if not 0 <= index < len(candidate_evidence) or index in seen:
            continue
        item = candidate_evidence[index]
        if item not in eligible or item.get("file") not in strong_files:
            continue
        selected.append(item)
        seen.add(index)
    if selected:
        # 화면에는 파일명이 표시되므로 선택 단위도 파일로 맞춥니다. 같은 규정 안에서 판정기가
        # 직접 고르지 않은 표·제출서류 청크도 생성에는 필요할 수 있어 선택된 파일의 후보를
        # 함께 전달합니다. 다른 파일은 계속 제외됩니다.
        selected_files = {item.get("file") for item in selected}
        return [item for item in eligible if item.get("file") in selected_files]
    if not eligible:
        return []
    top_file = max(best_keyword_by_file, key=best_keyword_by_file.get) if top_keyword else eligible[0].get("file")
    return [item for item in eligible if item.get("file") == top_file]


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
    candidate_evidence: list[dict]
    used_evidence: list[dict]
    analysis: dict
    answer: str
    ui_actions: list[str]
    missing: list[dict]


def resolve_question_node(state: ConsultationState):
    """이후 모든 단계가 함께 쓸 자립형 질문을 먼저 만듭니다."""
    return {"resolved": resolve_question(state["question"], state.get("history", []))}


def retrieve_policy_node(state: ConsultationState):
    """현재 제도 질문에 맞는 규정 근거를 검색합니다."""
    return {"candidate_evidence": retrieve(state.get("resolved") or state["question"])}


def analyze_question_node(state: ConsultationState):
    """검색 근거로 답할 수 있는지와 함께 제도 영역·대상 관계를 한 번에 뽑습니다."""
    question = state.get("resolved") or state["question"]
    return {
        "analysis": judge_groundedness(
            question,
            state.get("candidate_evidence", []),
            state.get("history", []),
        )
    }



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
        lines.extend(f"- {item['label']}" for item in missing[:4])
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
    candidate_evidence = state.get("candidate_evidence", [])
    missing = []
    calculation_answer = build_dispatch_calculation_answer(question)
    if calculation_answer:
        answer = calculation_answer
        used_evidence = [
            item for item in candidate_evidence
            if item.get("file") in {"여비관리기준.md", "사내 추가 기준.md"}
        ]
    elif not candidate_evidence:
        answer = build_unknown_policy_answer(question)
        used_evidence = []
    else:
        judgement = state.get("analysis") or judge_groundedness(question, candidate_evidence, history)
        used_evidence = select_used_evidence(question, candidate_evidence, judgement)
        verdict = judgement["verdict"]
        if verdict == "clarify":
            missing = judgement.get("missing", [])
            answer = build_clarify_answer(missing, judgement.get("finding", ""))
        elif verdict == "escalate":
            # 복리후생과 무관한 질문은 담당 부서로 넘기지 않고 상담 범위를 안내합니다.
            if judgement.get("intent") == "other":
                answer = build_clarification_answer(question)
            else:
                answer = build_escalation_answer(judgement.get("reason", ""), used_evidence)
        else:
            answer = call_openai(question, used_evidence, history)
    answer = append_final_review_notice(answer)
    result = {"answer": answer, "used_evidence": used_evidence}
    # 동호회 신규 신청 문의에는 답변 경로와 무관하게 메일 초안 버튼을 띄웁니다.
    if is_club_application_question(question):
        result["ui_actions"] = ["club_application_draft"]
    # clarify 판정일 때만 채운다. 화면이 이 목록을 클릭형 칩으로 보여줘 사용자가
    # 어떤 항목에 답하는지 "항목: 값" 형태로 명확히 남기게 한다.
    if missing:
        result["missing"] = missing[:4]
    return result




def build_consultation_graph():
    """상담 요청을 재작성·검색·판정·생성으로 연결한 LangGraph를 만듭니다."""
    graph = StateGraph(ConsultationState)
    graph.add_node("resolve", resolve_question_node)
    graph.add_node("retrieve", retrieve_policy_node)
    graph.add_node("analyze", analyze_question_node)
    graph.add_node("generate", generate_answer_node)
    # 규정 본문을 파이썬 문자열로 들고 있던 rules 노드는 지웠습니다. 답이 규정과 어긋나면
    # 고정 문구를 고치는 대신 검색·판정을 고칩니다. 모든 질문이 같은 경로를 지납니다.
    graph.add_edge(START, "resolve")
    graph.add_edge("resolve", "retrieve")
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


CONSULTATION_GRAPH = build_consultation_graph()
INQUIRY_HISTORY = []


def build_inquiry_draft(question, answer, evidence):
    """상담 결과를 담당자 문의 메일 초안으로 변환합니다."""
    # 검색 근거는 같은 규정의 여러 조항 청크를 포함할 수 있으므로, 메일에는 규정 파일별로 한 번만 적습니다.
    evidence_files = []
    seen_files = set()
    for item in evidence:
        file_name = str(item.get("file", "")).strip()
        if not file_name or file_name in seen_files:
            continue
        seen_files.add(file_name)
        evidence_files.append(Path(file_name).stem)
    evidence_lines = "\n".join(f"- {file_name}" for file_name in evidence_files) or "- 확인된 규정 근거 없음"
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
                "missing": result.get("missing", []),
                "evidence": result.get("used_evidence", []),
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
