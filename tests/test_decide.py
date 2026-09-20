def test_decide_returns_all_native_question_types(client):
    response = client.post(
        "/decide",
        json={
            "state": {
                "subject": "Duplicate subscription charge",
                "message": "I was billed twice and want the extra charge refunded today.",
            },
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which team should handle this request?",
                    "criteria": {
                        "billing": "Payments, invoices, charges, or refunds",
                        "technical": "Software or device problems",
                        "other": "Anything else",
                    },
                },
                "urgency": {
                    "type": "score",
                    "instructions": "How urgently should this be handled?",
                    "criteria": ["routine", "soon", "critical"],
                },
                "refund_requested": {
                    "type": "noul",
                    "instructions": "Is the customer explicitly requesting a refund?",
                },
            },
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert set(result["decisions"]) == {
        "department",
        "urgency",
        "refund_requested",
    }
    assert result["decisions"]["department"] in {"billing", "technical", "other"}
    assert result["decisions"]["urgency"] in {"routine", "soon", "critical"}
    assert isinstance(result["decisions"]["refund_requested"], bool)
