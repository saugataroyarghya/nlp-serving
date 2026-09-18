"""Entry service — composes services/*.py and exposes the final routes:

/entities, /classify, /entailment, /decide, /keywords, /image/classify, /generate

TODO: as each services/<name>.py service is built, wire it in here with
bentoml.depends(...) and add a matching @bentoml.api method that calls it.
"""
