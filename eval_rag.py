"""Evaluate RAG answers on 10 fixed questions with known correct facts."""

from rag import answer_question

TEST_CASES = [
    ("What is the overall deductible for one person?", "500"),
    ("What is the deductible for a family?", "1,000"),
    ("What is the out-of-pocket limit for network providers?", "2,500"),
    ("How much do I pay for emergency room care?", "20%"),
    ("What is the copay for a specialist visit?", "$50"),
    ("Is there a separate deductible for prescription drugs?", "300"),
    ("Does the plan cover routine dental care for adults?", "not"),
    ("Does the plan cover pet insurance?", "I don't know"),
    ("What is the capital of France?", "I don't know"),
    ("Who won the hockey game last night?", "I don't know"),
]


def is_correct(answer, expected):
    """Return True if the answer contains the expected fact (and a citation)."""
    answer = answer.replace("\u2019", "'").lower()
    if expected == "I don't know":
        return "i don't know" in answer
    return expected.lower() in answer and "[page" in answer


if __name__ == "__main__":
    passed = 0

    for number, (question, expected) in enumerate(TEST_CASES, start=1):
        result = answer_question(question)
        correct = is_correct(result["answer"], expected)
        if correct:
            passed += 1

        status = "PASS" if correct else "FAIL"
        print(f"\n{number}. [{status}] {question}")
        print(f"   Expected: {expected}")
        print(f"   Answer:   {result['answer'][:150]}")

    print("\n" + "=" * 70)
    print(f"SCORE: {passed} out of {len(TEST_CASES)} correct")