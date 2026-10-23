import subprocess
import unittest
from unittest.mock import patch
from fastapi import HTTPException
from app.photodash import process_bounded

class CorrectionTimeoutTests(unittest.TestCase):
    def test_timeout_is_recoverable_and_uses_killable_child(self):
        with patch("app.photodash.subprocess.run", side_effect=subprocess.TimeoutExpired("test", 85)) as run:
            with self.assertRaises(HTTPException) as failure:
                process_bounded(b"test")
            self.assertEqual(failure.exception.status_code, 503)
            self.assertEqual(run.call_args.kwargs["timeout"], 85)

    def test_bad_input_is_not_repeated_as_service_failure(self):
        with patch("app.photodash.subprocess.run", side_effect=subprocess.CalledProcessError(22, "test")):
            with self.assertRaises(HTTPException) as failure:
                process_bounded(b"test")
            self.assertEqual(failure.exception.status_code, 422)
