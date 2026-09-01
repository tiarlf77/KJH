import json
import os
import re
import calendar
from datetime import date
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import unquote


BASE_DIR = Path(__file__).resolve().parent
RULES_DIR = BASE_DIR / "규정"
MODEL = "gpt-5.6-terra"
SOURCE_FILES = {
    "경조금 지급기준.txt",
    "동호회 관리 규정.txt",
    "숙소지원금 운영 기준.txt",
    "여비관리기준.txt",
}


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
    match = re.search(r"(?<!\d)(19\d{2})(\d{2})(\d{2})(?!\d)", question)
    if not match:
        return ""
    try:
        birth = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        today = date.today()
        sixtieth = date(birth.year + 60, birth.month, birth.day)
    except ValueError:
        return ""
    if sixtieth > today:
        status = f"회갑일은 {sixtieth.isoformat()}이며 현재 기준일 이후이다. 아직 회갑일이 지나지 않았다."
    elif sixtieth == today:
        status = "오늘이 회갑일이다."
    else:
        status = f"회갑일은 {sixtieth.isoformat()}이며 현재 기준일 이전이다. 2026년 회갑 대상이 아니다."
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
    dates = re.findall(r"(?<!\d)(19\d{2}\d{4})(?!\d)", question + " " + prior_text)
    birthday = ""
    if dates and ("회갑" in question or "회갑" in prior_text):
        birthday = birthday_context("회갑 " + dates[-1])
    if not relation and not birthday:
        return ""
    return f"\n[대화 맥락 보완]\n{relation}\n{birthday}".strip()


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
        "가능 여부를 단정하기 어려우면 필요한 추가 정보를 질문한다. 답변은 자연스러운 대화체로 작성한다. "
        "답변 마지막에는 '확인한 규정'과 파일명을 간단히 표시한다. 최종 승인·지급은 담당 부서 검토임을 안내한다. "
        "규정에 명시된 예외만 적용하고, 규정에 없는 예외나 담당자 재량은 사용자에게 확인 질문으로 남긴다. "
        f"오늘 기준일은 {date.today().isoformat()}이다. 경조금은 회갑 대상 여부와 신청기한을 별도로 판단한다. 경조금 기준의 청구권은 사유 발생일로부터 3개월 이내이며, "
        "계산된 신청 마감일이 제공되면 그 날짜를 사용한다. 규정의 회갑 대상은 본인 및 배우자 부모이고 지급액은 20만원이다."
        " 부모·배우자 부모의 회갑 질문에서 생년월일이 없으면 연령을 추측하거나 지원을 확정하지 않는다. "
        "답변은 다음 흐름으로 작성한다: '해당 부모가 본인 부모인지 배우자 부모인지 관계 요건은 확인되지만, 현재 정보만으로 실제 회갑 대상인지 판단할 수 없다'고 먼저 설명한다. "
        "그 다음 현재 기준일을 알려주고 생년월일을 YYYYMMDD 형식으로 요청한다. 지급액 20만원과 신청기한은 회갑 대상 판정 이후에 안내한다. "
        "단, 사용자가 '우리 엄마', '우리 아버지', '우리 부모님'이라고 표현하면 별도 배우자 표현이 없는 한 본인 부모로 이해하고 관계를 다시 묻지 않는다. "
        "직전 대화에서 생년월일과 관계가 이미 확인되었으면 같은 질문을 반복하지 말고 회갑일 계산 결과를 안내한다."
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
            evidence = retrieve(question)
            answer = call_openai(question, evidence, body.get("history", []))
            self.respond(200, {"answer": answer, "evidence": evidence})
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
