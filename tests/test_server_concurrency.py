import asyncio
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import caption_server


class SerializedInferenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        caption_server._caption_semaphore = asyncio.Semaphore(1)
        caption_server._active_inference_requests = 0
        caption_server._waiting_inference_requests = 0

    async def test_blocking_inference_does_not_block_event_loop(self):
        started = threading.Event()
        release = threading.Event()

        def blocking_inference():
            started.set()
            release.wait(timeout=2)
            return "caption"

        task = asyncio.create_task(
            caption_server.run_serialized_inference(blocking_inference)
        )
        self.assertTrue(await asyncio.to_thread(started.wait, 1))

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False)
        )
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            health = await asyncio.wait_for(
                caption_server.health_check(),
                timeout=0.2,
            )

        self.assertEqual(health["status"], "healthy")
        self.assertTrue(health["inference_busy"])
        self.assertEqual(caption_server._active_inference_requests, 1)
        self.assertEqual(caption_server._waiting_inference_requests, 0)

        release.set()
        self.assertEqual(await task, "caption")
        self.assertEqual(caption_server._active_inference_requests, 0)

    async def test_requests_remain_serialized(self):
        first_started = threading.Event()
        first_release = threading.Event()
        second_started = threading.Event()

        def first_inference():
            first_started.set()
            first_release.wait(timeout=2)
            return "first"

        def second_inference():
            second_started.set()
            return "second"

        first = asyncio.create_task(
            caption_server.run_serialized_inference(first_inference)
        )
        self.assertTrue(await asyncio.to_thread(first_started.wait, 1))
        second = asyncio.create_task(
            caption_server.run_serialized_inference(second_inference)
        )
        await asyncio.sleep(0.02)

        self.assertFalse(second_started.is_set())
        self.assertEqual(caption_server._active_inference_requests, 1)
        self.assertEqual(caption_server._waiting_inference_requests, 1)

        first_release.set()
        self.assertEqual(await first, "first")
        self.assertEqual(await second, "second")
        self.assertEqual(caption_server._active_inference_requests, 0)
        self.assertEqual(caption_server._waiting_inference_requests, 0)


if __name__ == "__main__":
    unittest.main()
