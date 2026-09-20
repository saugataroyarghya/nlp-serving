"""OpenAPI presentation helpers for BentoML's generated Swagger document."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

ASGIScope = dict[str, Any]
ASGIMessage = dict[str, Any]
ASGIReceive = Callable[[], Awaitable[ASGIMessage]]
ASGISend = Callable[[ASGIMessage], Awaitable[None]]
ASGIApp = Callable[[ASGIScope, ASGIReceive, ASGISend], Awaitable[None]]


class SwaggerTagMiddleware:
    """Group selected BentoML routes under custom Swagger tags.

    BentoML currently assigns all ``@bentoml.api`` methods to one generic tag.
    This middleware changes only the generated ``/docs.json`` response.
    """

    def __init__(
        self,
        app: ASGIApp,
        route_tags: dict[str, str],
        tag_descriptions: dict[str, str],
    ) -> None:
        self.app = app
        self.route_tags = route_tags
        self.tag_descriptions = tag_descriptions

    async def __call__(
        self,
        scope: ASGIScope,
        receive: ASGIReceive,
        send: ASGISend,
    ) -> None:
        if scope.get("type") != "http" or not scope.get("path", "").endswith(
            "/docs.json"
        ):
            await self.app(scope, receive, send)
            return

        messages: list[ASGIMessage] = []

        async def capture(message: ASGIMessage) -> None:
            messages.append(message)

        await self.app(scope, receive, capture)
        start = next(
            (message for message in messages if message["type"] == "http.response.start"),
            None,
        )
        if start is None or start.get("status") != 200:
            for message in messages:
                await send(message)
            return

        raw_body = b"".join(
            message.get("body", b"")
            for message in messages
            if message["type"] == "http.response.body"
        )
        try:
            specification = json.loads(raw_body)
        except (TypeError, ValueError):
            for message in messages:
                await send(message)
            return

        for route, tag in self.route_tags.items():
            for operation in specification.get("paths", {}).get(route, {}).values():
                if isinstance(operation, dict) and "responses" in operation:
                    operation["tags"] = [tag]

        infrastructure_tags = [
            tag
            for tag in specification.get("tags", [])
            if tag.get("name") == "Infrastructure"
        ]
        specification["tags"] = [
            {"name": name, "description": description}
            for name, description in self.tag_descriptions.items()
        ] + infrastructure_tags

        body = json.dumps(specification, separators=(",", ":")).encode()
        headers = [
            (name, value)
            for name, value in start.get("headers", [])
            if name.lower() != b"content-length"
        ]
        headers.append((b"content-length", str(len(body)).encode()))
        await send({**start, "headers": headers})
        await send({"type": "http.response.body", "body": body})
