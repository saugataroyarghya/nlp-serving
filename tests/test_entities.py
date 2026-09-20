def test_extract_entities(client):
    r = client.post(
        "/extract_entities",
        json={
            "text": "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday.",
            "labels": ["company", "person", "product", "location"],
        },
    )
    assert r.status_code == 200
    entities = r.json()["entities"]
    assert entities["company"] == ["Apple"]
    assert entities["person"] == ["Tim Cook"]


def test_classify_text(client):
    r = client.post(
        "/classify_text",
        json={
            "text": "This laptop has amazing performance but terrible battery life!",
            "labels": {"sentiment": ["positive", "negative", "neutral"]},
        },
    )
    assert r.status_code == 200
    assert r.json()["sentiment"] == "negative"


def test_extract_structured(client):
    r = client.post(
        "/extract_structured",
        json={
            "text": "iPhone 15 Pro Max with 256GB storage, priced at $1199.",
            "extraction_schema": {
                "product": [
                    "name::str::Full product name",
                    "storage::str::Storage capacity",
                    "price::str::Product price",
                ]
            },
        },
    )
    assert r.status_code == 200
    product = r.json()["product"][0]
    assert product["name"] == "iPhone 15 Pro Max"


def test_extract_relations(client):
    r = client.post(
        "/extract_relations",
        json={
            "text": "John works for Apple Inc. and lives in San Francisco.",
            "relations": ["works_for", "lives_in"],
        },
    )
    assert r.status_code == 200
    relations = r.json()["relation_extraction"]
    assert relations["works_for"] == [["John", "Apple Inc."]]


def test_swagger_groups_endpoints_by_model(client):
    specification = client.get("/docs.json").json()

    assert [tag["name"] for tag in specification["tags"][:3]] == [
        "GLiNER",
        "Laya",
        "KeyBERT",
    ]
    assert specification["paths"]["/extract_entities"]["post"]["tags"] == [
        "GLiNER"
    ]
    assert specification["paths"]["/decide"]["post"]["tags"] == ["Laya"]
    assert specification["paths"]["/keywords"]["post"]["tags"] == ["KeyBERT"]
