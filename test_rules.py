"""복리후생 상담 규칙의 사용자 관점 동작을 고정하는 회귀 테스트입니다."""

from app import (
    CONSULTATION_GRAPH,
    attach_referenced_chunks,
    load_env,
    resolve_question,
    retrieve_policy_node,
)

# 키를 읽지 않으면 LLM 경로 사례가 전부 담당 부서 이관으로 떨어져 검증이 되지 않습니다.
load_env()


# (질문, 답변에 있어야 할 문자열, 답변에 없어야 할 문자열[, 이전 대화 이력])
# 이력은 이전 주제가 새 질문을 덮어쓰는지 보는 사례에만 붙입니다.
PROTECTED_CASES = [
    # 경조금 지급기준 5.2·5.3: 본인 형제·자매 결혼은 20만원과 부모 기준 증빙을 안내한다.
    (
        "제 동생이 결혼하는데 경조금과 필요한 서류 알려줘",
        "지원금: 200,000원",
        "배우자 부모 기준 가족관계증명서",
    ),
    # 경조금 지급기준 5.2·5.3: 배우자 형제·자매 결혼에는 배우자 부모 기준 증빙이 필요하다.
    (
        "배우자의 여동생이 결혼하는데 필요한 서류 알려줘",
        "배우자 부모 기준 가족관계증명서",
        "관계: 본인 형제·자매",
    ),
    # 경조금 지급기준 5.2: 승중상 인정 조건이 없으면 금액을 확정하지 않고 확인사항을 안내한다.
    # 되묻기 항목은 판정이 그때그때 문장으로 만든다("장손자 여부" / "본인이 장손자인지
    # 여부"). 회차마다 흔들리지 않는 '장손자'로만 조건을 물었는지 확인하고, 금액을
    # 확정하지 않는지는 금지 문자열이 맡는다.
    (
        "승중상 경조금 문의",
        "장손자",
        "- 지원금: 500,000원",
    ),
    # 경조금 지급기준 5.2: 본인 부모 사망 경조금은 100만원이다.
    (
        "우리 아버지가 돌아가셨는데 경조금 얼마야?",
        "- 지원금: 1,000,000원",
        "- 지원금: 300,000원",
    ),
    # 경조금 지급기준 5.2: 본인 및 배우자 조부모 사망 경조금은 30만원이다.
    (
        "본인 조부모님이 돌아가셨는데 경조금 얼마야?",
        "- 지원금: 300,000원",
        "- 지원금: 1,000,000원",
    ),
    # 경조금 지급기준 5.2: 부친 사망·장손자·상주 조건이 확인된 승중상은 50만원이다.
    (
        "아버지가 이미 사망했고 제가 장손으로 상주를 맡아 조부모상을 치르는 승중상입니다. 경조금은?",
        "- 지원금: 500,000원",
        "승중상 인정 조건 확인 필요",
    ),
    # 경조금 지급기준 5.2·5.3: 지급표에 없는 고모 관계는 금액을 단정하지 않는다.
    # '노사발전그룹'은 삭제된 build_death_answer의 문구였다. 규정이 요구하는 것은
    # 금액 단정 금지(금지 문자열)이므로, 필수 쪽은 이관 판정이면 표현을 가리지 않는다.
    (
        "고모가 돌아가셨는데 경조금 받을 수 있어?",
        ("노사발전그룹", "주관 부서 확인 필요"),
        "- 지원금:",
    ),
    # 경조금 지급기준 5.2·5.6: 부모 환갑은 생년월일 확인 후 3개월 기한을 판단한다.
    (
        "부모님 환갑 경조금 얼마야?",
        "부모님의 생년월일",
        "- 판정: 신청 가능",
    ),
    # 숙소지원금 운영 기준 5.5: 기존 숙소 정리는 기본 3개월, 최장 6개월까지 지원한다.
    # 고정 문구 규칙을 지운 뒤로는 "최장"과 "최대"가 번갈아 나오므로 기간만 확인한다.
    (
        "숙소지원금 2년 받다가 포항으로 발령났는데 기존 숙소 정리 비용은?",
        "6개월",
        "전 근무지 정리 비용은 지원 대상이 아닙니다",
    ),
    # 숙소지원금 운영 기준 5.1-6: 재부임은 1년이 원칙이며 기존 수급 합이 3년 이하일 때만 3년까지다.
    # 삭제한 build_housing_move_answer가 재부임을 모르고 신규 부임 기준을 그대로 단정하던 사례다.
    (
        "예전에 포항에서 2년 숙소지원금 받았는데 이번에 다시 포항으로 재부임했어요. 얼마나 더 받을 수 있나요?",
        "1년",
        "발령일로부터 3년간",
    ),
    # 숙소지원금 운영 기준 5.1-4: 매월 15일 이후 신청자는 다음 달부터 지급한다.
    # 생성 답변은 "다음 달"과 "10월"을 번갈아 쓴다. 낱말이 아니라 어느 달부터인지가 판정이다.
    (
        "9월 20일에 숙소지원금 신청하면 9월분부터 나오나요?",
        ("10월", "다음 달"),
        "9월분부터 지급",
    ),
    # 숙소지원금 운영 기준 5.5: 월세에서 전세 전환 시 전세금 1천만원당 월 10만원 기준을 쓴다.
    # 규정 51행의 표기는 "전세금 1,000만원당 10만원/월"이다. 옛 기대값
    # '전세금 10,000,000원당 월 100,000원'은 삭제된 고정 문구 쪽 표기였다.
    (
        "숙소지원금 받는데 월세에서 전세로 바꾸면 어떻게 돼?",
        ("전세금 1,000만원당", "전세금 10,000,000원당"),
        "관리비·공과금은 제외",
    ),
    # 여비관리기준 5.11과 숙소 기준: 실제 이사가 없으면 부임비·이전비를 지급하지 않는다.
    # 규정 97행이 "주거 이전을 하는 부임자에게" 지급한다고 정한다. 옛 기대값
    # '이사하지 않으면 지급 없음'은 삭제된 build_relocation_answer의 문구였다.
    (
        "광양에서 포항으로 발령났는데 이사는 하지 않고 혼자 살 예정이야. 부임비 나와?",
        ("지급되지 않습니다", "지급 없음", "지급 불가"),
        "부임비·이전비: 지급 대상",
    ),
    # 여비관리 FAQ: 해외출장 후 개인 연차로 늦춘 귀국 항공편은 회사가 지원하지 않는다.
    (
        "해외출장 후 개인 연차를 쓰고 늦게 귀국하면 변경한 항공편도 지원돼?",
        "회사에서 지원하지 않습니다",
        "판정: 지원 대상",
    ),
    # 주말 복귀 출장: 승인 일정인지 개인 사유인지에 따라 갈리므로 지급을 단정하지 않는다.
    # A-14와 같이 답변 형태(이관·되묻기·조건부 안내)는 고정하지 않고 오지급 단정만 막는다.
    (
        "국내출장을 마치고 개인 사정으로 일요일에 복귀하면 교통비가 지급돼?",
        "",
        "지급 대상입니다",
    ),
    # 개인 사유 주말 선이동: 개인 일정으로 추가된 교통비에 지원 근거를 만들지 않는다.
    # 고정 문구 규칙을 지운 뒤로는 담당 부서 이관과 되묻기 사이에서 답변 형태가 갈린다.
    # 둘 다 규정에 맞는 처리이므로 형태를 고정하지 않고, 잘못 지급을 단정하지 않는지만 확인한다.
    (
        "개인 사유로 출장 전에 주말에 미리 이동하면 추가 교통비가 지급돼?",
        "",
        "지급 대상입니다",
    ),
    # 동호회 관리 규정 5.1: 개설 문의에는 최소 등록 인원 5명과 사전 협의 절차를 안내한다.
    # 고정 문구를 반환하던 규칙을 지웠으므로, 규칙의 표현이 아니라 규정 원문의 내용을 확인한다.
    (
        "동호회 개설을 하고 싶은데 방법은?",
        "5명",
        "동호회 지원은 노사발전그룹의 등록·활동 실적 확인 후 지급됩니다",
    ),
    # 숙소지원금 운영 기준 5.4: 반경 25km 이내 거주는 원칙적 대상이 아니며 예외 검토 대상이다.
    # "숙소지원금 받을 수 있나요"를 기존 수급으로 오인해 전 근무지 정리 규칙이 켜지던 사례다.
    (
        "본가가 경주 안강인데 포항양극재로 발령났어요. 직선거리 22km인데 숙소지원금 받을 수 있나요?",
        "25km",
        "기존 숙소지원금 수급 중",
    ),
    # 경조금 지급기준 5.2: 앞선 환갑 대화가 남아 있어도 새 부모상 문의는 사망 기준으로 답한다.
    # 삭제한 build_hoegap_answer가 "부모님"이라는 낱말만 보고 이전 회갑 주제를 이어받아,
    # 100만원 대상자에게 환갑 20만원 안내를 반환하며 그래프를 끝내던 사례다.
    (
        "부모님 상당했는데 지원 가능한가요",
        "1,000,000원",
        "환갑",
        [
            {"role": "user", "content": "부모님 환갑 경조금 얼마야?"},
            {"role": "assistant", "content": "환갑(회갑) 경조금 기준은 만 60세가 되는 경우이며, 지원금은 200,000원입니다."},
        ],
    ),
]


# 아직 미해결 — 리팩터링 후 통과해야 함
KNOWN_FAILURE_CASES = [
    # 사망 오탐: '이상·증상'의 '상'과 질문의 '본인'이 본인 사망으로 잘못 결합된다.
    (
        "출장 갔다가 이상 증상이 생기면 본인이 부담하나요?",
        "",
        "5,000,000원",
    ),
    # 결혼 오탐: '유형' 안의 '형'이 본인 형제·자매 관계로 잘못 인식된다.
    # 유형을 나열하며 형제·자매를 언급하는 것은 정상이므로, 본인 사안을 형제·자매로
    # 단정하는 판정 줄만 금지한다.
    (
        "결혼 준비 중인데 유형별로 어떤 지원이 있는지 알려줘",
        "",
        "관계: 본인 형제·자매",
    ),
    # 부임 오탐: '이사회' 안의 '이사'가 실제 이사를 동반한 부임 질문으로 잘못 인식된다.
    (
        "이사회 참석하러 서울 가는데 출장비 되나요?",
        "",
        "부임비",
    ),
    # 숙소 오탐: 자녀 동거 표현만으로 숙소지원금 부정 수급 상황을 단정한다.
    (
        "자녀와 함께 살 예정인데 부임비 나오나요?",
        "",
        "윤리위반",
    ),
]


# 후속 질문이 이전 대화의 사실을 이어받는지는 최종 답변 문구로 재면 회차마다 흔들린다.
# 재작성된 질문 자체를 확인한다. 낱말 목록으로 주제 전환을 판별하던 때는 "해외출장입니다"의
# '출장'이 걸려 이력이 통째로 끊겼고, 앞선 일정·경로가 답변에 하나도 반영되지 않았다.
TRIP_HISTORY = [
    {
        "role": "user",
        "content": (
            "제 본가가 수도권인 관계로, 9/11(금)에 미리 수도권으로 이동하여 주말을 보낸 뒤 "
            "9/14(월)에 공항으로 이동하고자 합니다. 상세 일정: 9/11 포항 자택-포항역 택시, "
            "9/11 포항역-광명역 KTX, 9/14 서울-인천공항 공항버스. "
            "개인 신용카드로 결제한 후 출장비로 정산하면 되는지 확인 부탁드립니다."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "9/11 개인 사유로 미리 이동해 주말을 보낸 비용은 출장비로 정산하기 어렵고, "
            "9/14 공항 이동 비용도 해외출장 관련 국내 이동인지 확인이 필요합니다."
        ),
    },
]

# (질문, 이력, 재작성 결과에 있어야 할 문자열들, 없어야 할 문자열들)
CONTINUITY_CASES = [
    ("해외출장입니다", TRIP_HISTORY, ["인천공항", "9/14"], []),
    # 반대 방향도 막는다. 무관한 새 질문에 지난 주제를 끌어오면 안 된다.
    ("동호회 개설하려면?", TRIP_HISTORY, ["동호회"], ["인천공항", "포항역"]),
]


# 검색과 규칙 답변 단계를 지난 뒤에도 현재 제도와 무관한 규정 링크가 남지 않아야 합니다.
EVIDENCE_CASES = [
    (
        "해외출장 전일 이동으로 보면 됩니다",
        (
            "9월 11일부터 9월 14일까지 수도권에 체류한 뒤 9월 14일 인천공항으로 이동하는 "
            "일정을 해외출장 전일 이동으로 볼 때, 숙소지원금 운영 기준과 해외출장 비용을 어떻게 적용하나요?"
        ),
        "trip",
        {"여비관리기준.md"},
        {"여비관리 FAQ.md", "숙소지원금 운영 기준.md", "경조금 지급기준.md"},
    ),
    (
        "해외출장 종료 후 개인 휴가로 이틀 더 머물렀는데 귀국 항공권 변경 비용도 회사가 부담하나요?",
        "해외출장 종료 후 개인 휴가로 이틀 더 머문 뒤 귀국 항공권 변경 비용의 회사 부담 여부",
        "trip",
        {"여비관리기준.md"},
        {"여비관리 FAQ.md", "숙소지원금 운영 기준.md", "경조금 지급기준.md"},
    ),
    (
        "조의금 신청 시 가족관계증명서를 즉시 제출하기 어려워도, 사실관계를 증명할 수 있는 다른 서류가 있으면 대체 인정될 수 있나요?",
        "조의금 신청 시 가족관계증명서를 대체할 수 있는 증빙서류",
        "ceremony",
        {"경조금 지급기준.md"},
        {"여비관리 FAQ.md", "숙소지원금 운영 기준.md", "동호회 관리 규정.md"},
    ),
]


CLUB_SCOPE_CASES = [
    ("낚시 동호회 개설 가능한가요?", "낚시"),
    ("헬스 동아리 만들어도 되나요?", "헬스"),
]


def check_continuity(question, history, required_texts, forbidden_texts):
    """후속 입력이 자립형 질문으로 다시 쓰이는지 확인합니다."""
    resolved = resolve_question(question, history)
    assert resolved, "재작성 결과가 비어 있습니다."
    for text in required_texts:
        assert text in resolved, f"이전 대화의 사실이 빠졌습니다: {text!r} / 결과: {resolved!r}"
    for text in forbidden_texts:
        assert text not in resolved, f"무관한 주제가 딸려왔습니다: {text!r} / 결과: {resolved!r}"


def check_evidence(question, resolved, intent, expected_files, forbidden_files):
    """검색 뒤에 현재 제도와 무관한 규정 링크가 남지 않는지 확인합니다.

    원래는 `apply_policy_rules_node`를 함께 태워 규칙이 덮어쓴 근거까지 봤습니다. 그 노드는
    규정 본문을 파이썬 문자열로 들고 있어 삭제했고, 규칙이 근거를 덮어쓰는 경로 자체가
    없어졌습니다. 검증하려던 것(무관한 파일이 근거에 남지 않는다)은 검색 결과로 그대로 봅니다.
    intent는 규칙을 켜는 데만 쓰였으므로 더 이상 필요하지 않습니다.
    """
    evidence = retrieve_policy_node({"question": question, "resolved": resolved})["evidence"]
    files = {item["file"] for item in evidence}
    assert files == expected_files, f"근거 파일이 다릅니다: {sorted(files)!r}"
    assert not files & forbidden_files, f"무관한 근거 파일이 포함됐습니다: {sorted(files & forbidden_files)!r}"


def check_club_scope(question, required_term):
    """동호회·동아리 질문이 분야별 운영 기준이 담긴 근거를 찾는지 확인합니다."""
    result = retrieve_policy_node({"question": question, "resolved": question})
    evidence = result["evidence"]
    files = {item["file"] for item in evidence}
    assert files == {"동호회 관리 규정.md"}, f"근거 파일이 다릅니다: {sorted(files)!r}"
    assert any(required_term in item["text"] for item in evidence), (
        f"분야별 운영 기준에 {required_term!r}가 포함되지 않았습니다."
    )


def check_case(question, required_text, forbidden_text, history=()):
    """전체 상담 그래프에서 필수·금지 문구를 확인합니다."""
    result = CONSULTATION_GRAPH.invoke({"question": question, "history": list(history)})
    answer = result.get("answer", "")
    assert answer, "답변이 비어 있습니다."
    if required_text:
        # 같은 판정을 다른 낱말로 쓰는 경우가 있어(다음 달/10월, 최장/최대) 대안을 튜플로 받습니다.
        options = required_text if isinstance(required_text, tuple) else (required_text,)
        assert any(option in answer for option in options), f"필수 문자열이 없습니다: {options!r}"
    if forbidden_text:
        assert forbidden_text not in answer, f"금지 문자열이 포함됐습니다: {forbidden_text!r}"


def run_cases(group, cases, checker=check_case):
    """한 그룹의 모든 사례를 실행하고 개별 결과와 합계를 반환합니다."""
    passed = failed = skipped = 0
    for index, case in enumerate(cases, 1):
        question = case[0]
        case_id = f"{group}-{index:02d}"
        try:
            checker(*case)
        except RuntimeError as error:
            if "OPENAI_API_KEY" in str(error):
                skipped += 1
                print(f"건너뜀 [{case_id}] {question}")
                print("  이유: API 키가 없어 LLM 답변이 필요한 사례를 실행할 수 없습니다.")
                continue
            failed += 1
            print(f"실패 [{case_id}] {question}")
            print(f"  이유: {error}")
        except AssertionError as error:
            failed += 1
            print(f"실패 [{case_id}] {question}")
            print(f"  이유: {error}")
        except Exception as error:
            failed += 1
            print(f"실패 [{case_id}] {question}")
            print(f"  이유: {type(error).__name__}: {error}")
        else:
            passed += 1
            print(f"통과 [{case_id}] {question}")
    return passed, failed, skipped


def check_reference_expansion():
    """참조 확장은 순수 함수라 API 없이 확인합니다. LLM 회차 흔들림과 무관하게 깨집니다."""
    chunks = [
        {"file": "여비관리기준.md", "path": "여비관리기준 > 7. 관련문서 > 별첨 1. 국내여비기준표", "text": "숙박비 100,000원"},
        {"file": "여비관리기준.md", "path": "여비관리기준 > 7. 관련문서 > 별첨 2. 국내이전료 정액표", "text": "이전료"},
        {"file": "경조금 지급기준.md", "path": "경조금 지급기준 > 별첨 1. 다른 문서의 같은 번호", "text": "끌려오면 안 됨"},
    ]
    pointing = [{"file": "여비관리기준.md", "path": "… > 5.10.2 소액경비 및 숙박비",
                 "text": "소액경비 및 숙박료 지급기준은 별첨 1에 의한다.", "score": 0.9}]
    added = attach_referenced_chunks(list(pointing), chunks)[1:]
    assert [item["path"] for item in added] == [chunks[0]["path"]], f"별첨 1만 붙어야 합니다: {added}"

    # 별첨 목록표처럼 번호를 늘어놓기만 하는 청크는 따라가지 않습니다.
    listing = [{"file": "여비관리기준.md", "path": "… > 6. 기록 및 첨부",
                "text": "별첨 1, 별첨 2, 별첨 3 보존연한", "score": 0.9}]
    assert attach_referenced_chunks(list(listing), chunks) == listing, "나열 청크는 따라가면 안 됩니다."

    # 이미 근거에 있는 청크는 중복해서 붙이지 않습니다.
    already = list(pointing) + [dict(chunks[0], score=0.5)]
    assert attach_referenced_chunks(already, chunks) == already, "중복 첨부가 발생했습니다."
    print("통과 [D-01] 참조 확장(별첨 지목·나열 제외·중복 방지)")


def main():
    """보호 동작과 알려진 오답을 실행하고 전체 결과를 요약합니다."""
    check_reference_expansion()
    protected = run_cases("A", PROTECTED_CASES)
    unresolved = run_cases("B", KNOWN_FAILURE_CASES)
    continuity = run_cases("C", CONTINUITY_CASES, check_continuity)
    evidence = run_cases("D", EVIDENCE_CASES, check_evidence)
    club_scope = run_cases("E", CLUB_SCOPE_CASES, check_club_scope)
    total = tuple(sum(values) for values in zip(protected, unresolved, continuity, evidence, club_scope))

    print()
    print(f"(A) 지켜야 할 동작: 통과 {protected[0]} / 실패 {protected[1]} / 건너뜀 {protected[2]}")
    print(f"(B) 아직 미해결: 통과 {unresolved[0]} / 실패 {unresolved[1]} / 건너뜀 {unresolved[2]}")
    print(f"(C) 대화 연속성: 통과 {continuity[0]} / 실패 {continuity[1]} / 건너뜀 {continuity[2]}")
    print(f"(D) 근거 링크 적합성: 통과 {evidence[0]} / 실패 {evidence[1]} / 건너뜀 {evidence[2]}")
    print(f"(E) 동호회 분야 기준 검색: 통과 {club_scope[0]} / 실패 {club_scope[1]} / 건너뜀 {club_scope[2]}")
    print(f"전체: 통과 {total[0]} / 실패 {total[1]} / 건너뜀 {total[2]}")
    raise SystemExit(1 if total[1] else 0)


if __name__ == "__main__":
    main()
