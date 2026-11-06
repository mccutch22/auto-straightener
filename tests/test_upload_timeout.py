import asyncio
import unittest
import types
import sys
from unittest.mock import patch, AsyncMock
from fastapi import HTTPException
from app.photodash import handle_binary, busy


class UploadTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_stalled_upload_releases_worker_and_next_request_succeeds(self):
        class Stalled:
            async def stream(self):
                yield b'first chunk'
                await asyncio.Event().wait()

        class Complete:
            async def stream(self):
                yield b'image'

        with patch.dict(sys.modules, {'app.main': types.SimpleNamespace(require_auth=lambda value: None)}), patch('app.photodash.UPLOAD_TIMEOUT_SECONDS', .02):
            with self.assertRaises(HTTPException) as raised:
                await handle_binary(Stalled(), 'Bearer secret')
            self.assertEqual(raised.exception.status_code, 408)
            self.assertFalse(busy.locked())
            with patch('app.photodash.run_in_threadpool', new=AsyncMock(return_value={'outcome': 'unchanged'})):
                self.assertEqual(await handle_binary(Complete(), 'Bearer secret'), {'outcome': 'unchanged'})
            self.assertFalse(busy.locked())

    async def test_cancelled_upload_releases_worker(self):
        class Cancelled:
            async def stream(self):
                raise asyncio.CancelledError()
                yield b''
        with patch.dict(sys.modules, {'app.main': types.SimpleNamespace(require_auth=lambda value: None)}):
            with self.assertRaises(asyncio.CancelledError):
                await handle_binary(Cancelled(), 'Bearer secret')
        self.assertFalse(busy.locked())