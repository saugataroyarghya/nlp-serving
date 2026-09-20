# NLP model playground

An experimental BentoML playground for learning and comparing focused NLP
models without using a general-purpose generative LLM for every task.

GLiNER2.5, Laya, and KeyBERT share one BentoML service because their dependency
versions are compatible. Models with conflicting requirements remain isolated.

## Setup

The main environment contains GLiNER2.5, GLiNER-RelEx, GLiNER Decoder, Laya,
and KeyBERT:

```bash
uv sync
```

Model weights are downloaded from Hugging Face on first use and cached outside
the repository. GLiDRE has incompatible GLiNER and Transformers requirements,
so its separate environment is described in the GLiDRE section below.

## Services at a glance

| Port | Service | Purpose | Environment |
| ---: | --- | --- | --- |
| 3000 | GLiNER2.5 + Laya + KeyBERT | Extraction, classification, decisions, keywords | `.venv` |
| 3001 | GLiNER-RelEx | Joint entity and relation extraction | `.venv` |
| 3002 | GLiNER Decoder | Entity discovery without caller-supplied labels | `.venv` |
| 3003 | GLiDRE | Document-level relation extraction over supplied mentions | `.venv-glidre` |

The port numbers are local defaults, not installation locations. A different
port can be supplied with `-p` as long as another process is not using it.

## GLiNER2.5 service with Laya and KeyBERT

```bash
uv run bentoml serve services.entities:GliNER2Service --reload -p 3000
```

Swagger: <http://localhost:3000/docs>

This service exposes GLiNER's extraction endpoints together with Laya's
`/decide` and KeyBERT's `/keywords` endpoint.

GLiNER endpoints:

- `/extract_entities`: zero-shot named-entity extraction
- `/classify_text`: zero-shot text classification
- `/extract_structured`: schema-guided, extractive record construction
- `/extract_relations`: GLiNER2.5's native relation extraction

Laya and KeyBERT are in this same service because they install alongside
GLiNER2.5 without a dependency-version conflict.

## GLiNER-RelEx

Joint entity and relation extraction with entity, adjacency, and relation
thresholds:

```bash
uv run bentoml serve services.gliner_relex:GLiNERRelExService -p 3001
```

Swagger: <http://localhost:3001/docs>

Example `/extract` input:

```json
{
  "text": "John works for Apple Inc. and lives in San Francisco. Alex works for Google Inc and lives in London.",
  "entity_labels": ["person", "organization", "location"],
  "relations": ["works for", "lives in"],
  "entity_threshold": 0.4,
  "adjacency_threshold": 0.6,
  "relation_threshold": 0.75,
  "flat_ner": false,
  "multi_label": false
}
```

`entity_labels` also accepts descriptions:

```json
{
  "person": "A human individual",
  "organization": "A company, institution, or agency",
  "location": "A geographic place"
}
```

## GLiNER Decoder

Open-ontology NER that discovers spans and generates entity type names without
requiring the caller to provide a schema:

```bash
uv run bentoml serve services.gliner_decoder:GLiNERDecoderService -p 3002
```

Swagger: <http://localhost:3002/docs>

Example `/discover_entities` input:

```json
{
  "text": "Satya Nadella visited Dhaka to announce a new Microsoft cloud service for hospitals.",
  "threshold": 0.3,
  "num_generated_labels": 1,
  "flat_ner": true
}
```

The useful generated type is returned in `generated_labels`; the literal
`label` field remains the model's open-ontology sentinel.

## GLiDRE

GLiDRE pins older versions of GLiNER and Transformers, so it uses a dedicated
virtual environment:

```bash
uv venv .venv-glidre --python 3.13
uv pip install --python .venv-glidre/bin/python -r requirements-glidre.txt
.venv-glidre/bin/bentoml serve services.glidre:GLiDREService -p 3003
```

Swagger: <http://localhost:3003/docs>

GLiDRE does not discover entities. Supply coreference-grouped mentions with
zero-based, end-exclusive character offsets. The service validates every span
against the text before inference.

Example `/extract_relations` input:

```json
{
  "text": "The Loud Tour was the fourth overall and third world concert tour by Barbadian recording artist Rihanna.",
  "relations": [
    "COUNTRY_OF_CITIZENSHIP",
    "PUBLICATION_DATE",
    "PART_OF"
  ],
  "mentions": [
    {
      "id": 0,
      "type": "LOC",
      "mentions": [
        {"value": "Barbadian", "start": 69, "end": 78}
      ]
    },
    {
      "id": 1,
      "type": "PER",
      "mentions": [
        {"value": "Rihanna", "start": 96, "end": 103}
      ]
    }
  ],
  "threshold": 0.3,
  "multi_label": false
}
```

The GLiDRE checkpoint performs better when relation labels are uppercase with
underscores, as shown above.

## Laya

Laya is a non-generative decision model. It answers several typed questions in
one forward pass. The endpoint returns only the final decision for each
question; raw probabilities and internal action scores stay hidden:

I was testing Laya to understand how Jev works—especially the pattern of giving
a model dynamic state, typed questions, and a closed set of possible decisions
instead of asking a generative LLM to produce and format an answer.

Run the combined service shown above and open <http://localhost:3000/docs>.

Example `/decide` input:

```json
{
  "state": {
    "subject": "Duplicate subscription charge",
    "message": "I was billed twice and want the extra charge refunded today."
  },
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this request?",
      "criteria": {
        "billing": "Payments, invoices, charges, or refunds",
        "technical": "Software or device problems",
        "other": "Anything else"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgently should this be handled?",
      "criteria": ["routine", "soon", "critical"]
    },
    "refund_requested": {
      "type": "noul",
      "instructions": "Is the customer explicitly requesting a refund?"
    }
  }
}
```

`noul` is Laya's yes/no question type and is returned as `true` or `false`.
`choice` selects one supplied option, while `score` selects from an ordered
list. The state may be a string, an arbitrary JSON object, or a list of
conversation turns. This service currently uses the English checkpoint.

## KeyBERT

KeyBERT ranks candidate words and phrases by embedding similarity with the
document:

Run the combined service shown above and open <http://localhost:3000/docs>.

Example `/keywords` input:

```json
{
  "text": "Machine learning systems can extract useful keywords from product reviews and support tickets."
}
```

The endpoint applies its extraction settings internally and returns only the
keyword strings, without exposing similarity scores.

## Practical input limits

These are the limits configured in the installed checkpoints. GLiNER-family
limits are word-level tokens; Laya and KeyBERT use tokenizer subwords.

| Model | Configured input limit | Automatic chunking in this API |
| --- | ---: | --- |
| GLiNER2.5 | 4,096 | No |
| GLiNER-RelEx | 2,048 | No |
| GLiNER Decoder | 512 | No |
| GLiDRE | 512 | No |
| Laya English | 512 per question | No |
| KeyBERT MiniLM | 256 | No |

For Laya, the question instructions and options can consume up to 192 of the
512 tokens, leaving the remainder for the serialized state. Overlong states are
truncated at the end.

## Tests

Run the combined GLiNER2.5, Laya, KeyBERT, and Swagger tests with:

```bash
uv run pytest -q tests/test_entities.py tests/test_decide.py tests/test_keywords.py
```

The tests use one shared BentoML application so the three compatible models are
loaded only once for the test session.

## License and attribution

This project is licensed under the [Apache License 2.0](LICENSE).

The models and libraries used here remain under their respective upstream
licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for model links,
license attribution, and the GLiDRE source-package licensing caveat. Model
weights are downloaded at runtime and are not included in this repository.
