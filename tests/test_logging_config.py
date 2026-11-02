import json
import logging
import os
import unittest
from io import StringIO
from unittest.mock import patch

from dec_engine.logging_config import configure_logging, get_logger


class LoggingConfigTests(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("dec_engine")
        self.handlers = list(self.logger.handlers)
        self.level = self.logger.level
        self.propagate = self.logger.propagate

    def tearDown(self):
        for handler in tuple(self.logger.handlers):
            self.logger.removeHandler(handler)
        for handler in self.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(self.level)
        self.logger.propagate = self.propagate

    def test_json_logs_include_level_message_and_operation_context(self):
        stream = StringIO()
        configure_logging("DEBUG", stream=stream)
        get_logger("tests", file="level.DE").bind(
            operation="project_load", object_id="node-1"
        ).info("Project loaded")

        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["level"], "INFO")
        self.assertEqual(payload["message"], "Project loaded")
        self.assertEqual(
            payload["context"],
            {
                "file": "level.DE",
                "operation": "project_load",
                "object_id": "node-1",
            },
        )
        self.assertTrue(payload["timestamp"].endswith("+00:00"))

    def test_environment_configures_log_level(self):
        stream = StringIO()
        with patch.dict(os.environ, {"DEC_LOG_LEVEL": "WARNING"}):
            configure_logging(stream=stream)
        logger = get_logger("tests")
        logger.info("hidden")
        logger.warning("visible")

        self.assertEqual(
            [json.loads(line)["message"] for line in stream.getvalue().splitlines()],
            ["visible"],
        )

    def test_invalid_log_level_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid logging level"):
            configure_logging("VERBOSE", stream=StringIO())

    def test_context_rejects_non_scalar_values(self):
        with self.assertRaisesRegex(TypeError, "scalar values"):
            get_logger("tests", details={"secret": "value"})


if __name__ == "__main__":
    unittest.main()
