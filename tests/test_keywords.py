def test_keywords_returns_ranked_phrases(client):
    response = client.post(
        "/keywords",
        json={
            "text": (
                "KeyBERT extracts keywords and keyphrases by comparing document "
                "embeddings with candidate phrase embeddings."
            )
        },
    )

    assert response.status_code == 200
    keywords = response.json()["keywords"]
    assert 1 <= len(keywords) <= 10
    assert all(isinstance(keyword, str) for keyword in keywords)
