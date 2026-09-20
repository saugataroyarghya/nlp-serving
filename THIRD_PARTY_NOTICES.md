# Third-party software and model attribution

This project integrates third-party Python packages and downloads model
artifacts at runtime. Those components remain under their respective licenses;
the project's Apache-2.0 license does not replace or relicense them.

Model weights are not stored in this repository.

## Models

| Component | Upstream | Declared license | Use in this project |
| --- | --- | --- | --- |
| GLiNER2.5 Base | [fastino/gliner2.5-base-v1](https://huggingface.co/fastino/gliner2.5-base-v1) | Apache-2.0 | Schema-driven extraction and classification |
| GLiNER-RelEx Large | [knowledgator/gliner-relex-large-v1.0](https://huggingface.co/knowledgator/gliner-relex-large-v1.0) | Apache-2.0 | Joint entity and relation extraction |
| GLiNER Decoder Large | [knowledgator/gliner-decoder-large-v1.0](https://huggingface.co/knowledgator/gliner-decoder-large-v1.0) | Apache-2.0 | Open-ontology entity discovery |
| GLiDRE Large | [cea-list-ia/glidre_large](https://huggingface.co/cea-list-ia/glidre_large) | Apache-2.0 | Document-level relation extraction |
| Laya | [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) | Apache-2.0 | Typed non-generative decisions |
| all-MiniLM-L6-v2 | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Apache-2.0 | KeyBERT document and phrase embeddings |
| DistilBERT SST-2 | [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) | Apache-2.0 | Sentiment demonstration |

Review each upstream model card before production or commercial use. Model
licenses, training-data terms, limitations, and revisions can change
independently of this repository.

## Principal software dependencies

| Component | Upstream | Declared license |
| --- | --- | --- |
| BentoML | [bentoml/BentoML](https://github.com/bentoml/BentoML) | Apache-2.0 |
| GLiNER2 | [fastino-ai/GLiNER2](https://github.com/fastino-ai/GLiNER2) | Apache-2.0 |
| GLiNER | [urchade/GLiNER](https://github.com/urchade/GLiNER) | Apache-2.0 |
| Laya | [convaiinnovations/laya](https://pypi.org/project/laya/) | Apache-2.0 |
| KeyBERT | [MaartenGr/KeyBERT](https://github.com/MaartenGr/KeyBERT) | MIT |
| Sentence Transformers | [huggingface/sentence-transformers](https://github.com/huggingface/sentence-transformers) | Apache-2.0 |
| Transformers | [huggingface/transformers](https://github.com/huggingface/transformers) | Apache-2.0 |
| PyTorch | [pytorch/pytorch](https://github.com/pytorch/pytorch) | BSD-3-Clause |

The complete transitive dependency set and resolved versions are recorded in
`uv.lock`. Consult installed package metadata and upstream distributions for
their complete license texts and notices.

## GLiDRE source-package caveat

`requirements-glidre.txt` installs GLiDRE source from a pinned commit in
[cea-list-lasti/glidre](https://github.com/cea-list-lasti/glidre). The GLiDRE
model card declares Apache-2.0 for the checkpoint, but the pinned source
repository does not currently provide a clearly declared software license.
This project does not vendor or redistribute that source code. Confirm the
source-code terms with its maintainers before redistributing GLiDRE itself or
using it in a production distribution.
