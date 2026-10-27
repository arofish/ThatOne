
import unittest

from unittest.mock import patch

from fastapi.testclient import TestClient

from photoindex.api import create_app

from photoindex.search_service import (
    PreviewChangedError,
    SearchResponse,
    SearchResult,
    SearchService,
    SearchValidationError,
    ThumbnailNotFoundError,
    ThumbnailReadError,
)


TOKEN = "T" * 43

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
}


class FakeService:
    run_id = "chinese-clip-base-cpu-v2"
    asset_ids = tuple(range(1, 51))

    def __init__(self):
        self.searched = []
        self.preview = b"\x89PNG\r\n\x1a\nTEST"
        self.fail_search = False
        self.fail_preview = None

    def search(self, query):
        if not query.strip() or len(query) > 1000:
            raise SearchValidationError(
                "private query"
            )

        if self.fail_search:
            raise RuntimeError(
                "PRIVATE_TRACEBACK_OR_TOKEN"
            )

        self.searched.append(query)

        return SearchResponse(
            query,
            self.run_id,
            [
                SearchResult(
                    19,
                    0.777,
                    "a.jpg",
                    "/original/a.jpg",
                    "available",
                    "/assets/19/thumbnail",
                ),
                SearchResult(
                    22,
                    0.443,
                    "missing.jpg",
                    "/original/missing.jpg",
                    "missing",
                    "/assets/22/thumbnail",
                ),
            ],
        )

    def thumbnail_bytes(self, asset_id):
        if self.fail_preview == "changed":
            raise PreviewChangedError(
                "changed"
            )

        if self.fail_preview == "read":
            raise ThumbnailReadError(
                "cannot read"
            )

        if self.fail_preview == "internal":
            raise RuntimeError(
                "PRIVATE_TRACEBACK_OR_TOKEN"
            )

        if asset_id not in (19, 22):
            raise ThumbnailNotFoundError(
                "missing"
            )

        return self.preview


class ApiTests(unittest.TestCase):

    def setUp(self):
        self.fake = FakeService()

        self.app = create_app(
            service=self.fake,
            token=TOKEN,
        )

        self.client_context = TestClient(
            self.app,
            raise_server_exceptions=False,
        )

        self.client = self.client_context.__enter__()

        self.addCleanup(
            self.client_context.__exit__,
            None,
            None,
            None,
        )

    def test_health_and_auth(self):
        endpoints = [
            ("get", "/health", {}),
            (
                "post",
                "/search",
                {"json": {"query": "hello"}},
            ),
            (
                "get",
                "/assets/19/thumbnail",
                {},
            ),
        ]

        for method, path, kwargs in endpoints:
            response = getattr(
                self.client,
                method,
            )(path, **kwargs)

            self.assertEqual(
                response.status_code,
                401,
            )
            self.assertEqual(
                response.json()["error"]["code"],
                "UNAUTHORIZED",
            )
            self.assertEqual(
                response.headers["www-authenticate"],
                "Bearer",
            )

        wrong = self.client.get(
            "/health",
            headers={
                "Authorization": "Bearer WRONG"
            },
        )
        self.assertEqual(wrong.status_code, 401)

        good = self.client.get(
            "/health",
            headers=HEADERS,
        )

        self.assertEqual(good.status_code, 200)
        self.assertEqual(
            good.json(),
            {
                "status": "ready",
                "run_id": self.fake.run_id,
                "asset_count": 50,
            },
        )

    def test_search_response(self):
        response = self.client.post(
            "/search",
            headers=HEADERS,
            json={"query": " 中文 "},
        )

        self.assertEqual(response.status_code, 200)

        data = response.json()

        self.assertEqual(
            data["query"],
            " 中文 ",
        )

        self.assertEqual(
            [
                r["asset_id"]
                for r in data["results"]
            ],
            [19, 22],
        )

        self.assertEqual(
            data["results"][1]["original_status"],
            "missing",
        )

        self.assertEqual(
            data["results"][0]["thumbnail_url"],
            "/assets/19/thumbnail",
        )

    def test_422(self):
        invalid_requests = [
            {},
            {"query": 1},
            {"query": None},
            {"query": "x" * 1001},
            {"query": "hello", "other": 42},
        ]

        for payload in invalid_requests:
            with self.subTest(
                payload=str(payload)[:30]
            ):
                response = self.client.post(
                    "/search",
                    headers=HEADERS,
                    json=payload,
                )

                self.assertEqual(
                    response.status_code,
                    422,
                )
                self.assertEqual(
                    response.json()["error"]["code"],
                    "INVALID_REQUEST",
                )

        blank = self.client.post(
            "/search",
            headers=HEADERS,
            json={"query": " \t"},
        )

        self.assertEqual(blank.status_code, 422)
        self.assertEqual(
            blank.json()["error"]["code"],
            "INVALID_QUERY",
        )

        self.assertNotIn(" \t", blank.text)

    def test_thumbnail_errors(self):
        valid = self.client.get(
            "/assets/19/thumbnail",
            headers=HEADERS,
        )

        self.assertEqual(valid.status_code, 200)
        self.assertEqual(
            valid.content,
            self.fake.preview,
        )
        self.assertEqual(
            valid.headers["content-type"],
            "image/png",
        )
        self.assertEqual(
            valid.headers["cache-control"],
            "no-store",
        )

        missing = self.client.get(
            "/assets/99/thumbnail",
            headers=HEADERS,
        )

        self.assertEqual(missing.status_code, 404)
        self.assertEqual(
            missing.json()["error"]["code"],
            "THUMBNAIL_NOT_FOUND",
        )

        self.fake.fail_preview = "changed"

        changed = self.client.get(
            "/assets/19/thumbnail",
            headers=HEADERS,
        )

        self.assertEqual(changed.status_code, 409)
        self.assertEqual(
            changed.json()["error"]["code"],
            "PREVIEW_CHANGED",
        )

        self.fake.fail_preview = "read"

        unreadable = self.client.get(
            "/assets/19/thumbnail",
            headers=HEADERS,
        )

        self.assertEqual(unreadable.status_code, 500)
        self.assertEqual(
            unreadable.json()["error"]["code"],
            "THUMBNAIL_READ_FAILED",
        )

    def test_unexpected_errors_hidden(self):
        self.fake.fail_search = True

        response = self.client.post(
            "/search",
            headers=HEADERS,
            json={"query": "hello"},
        )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json()["error"]["code"],
            "INTERNAL_ERROR",
        )

        self.assertNotIn(
            "PRIVATE_TRACEBACK_OR_TOKEN",
            response.text,
        )

    def test_docs_disabled(self):
        for path in (
            "/docs",
            "/redoc",
            "/openapi.json",
        ):
            response = self.client.get(
                path,
                headers=HEADERS,
            )
            self.assertEqual(
                response.status_code,
                404,
            )

    def test_missing_token_prevents_creation(self):
        with patch.dict(
            "os.environ",
            {},
            clear=True,
        ):
            with self.assertRaises(RuntimeError):
                create_app()

    def test_startup_failure_does_not_enter_ready(self):
        with patch.object(
            SearchService,
            "from_project",
            side_effect=RuntimeError("broken"),
        ) as loader:
            with self.assertRaisesRegex(
                RuntimeError,
                "broken",
            ):
                with TestClient(
                    create_app(token=TOKEN)
                ):
                    pass

        loader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
