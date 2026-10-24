import copy
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from src.requirement_parser.parser import (
    RequirementValidationError,
    load_requirement,
    main,
    parse_requirement_data,
)


VALID_REQUEST = {
    "schema_version": "1.0",
    "request_id": "edge_cpu_demo",
    "target": {
        "task": "object_detection",
        "device": "cpu",
        "dataset": "KITTI",
    },
    "constraints": {
        "minimum_map50_95": 0.25,
        "maximum_median_latency_ms": 35.0,
        "maximum_model_size_mb": 6.0,
    },
    "preferences": {
        "optimization_goal": "balanced",
    },
}


class TestRequirementParser(unittest.TestCase):
    def test_valid_request(self) -> None:
        requirement = parse_requirement_data(
            copy.deepcopy(VALID_REQUEST)
        )

        self.assertEqual(
            requirement.request_id,
            "edge_cpu_demo",
        )
        self.assertEqual(
            requirement.target.device,
            "cpu",
        )
        self.assertEqual(
            requirement.constraints.minimum_map50_95,
            0.25,
        )

    def test_rejects_map_above_one(self) -> None:
        data = copy.deepcopy(VALID_REQUEST)
        data["constraints"]["minimum_map50_95"] = 1.5

        with self.assertRaisesRegex(
            RequirementValidationError,
            "between 0 and 1",
        ):
            parse_requirement_data(data)

    def test_rejects_negative_latency(self) -> None:
        data = copy.deepcopy(VALID_REQUEST)
        data["constraints"][
            "maximum_median_latency_ms"
        ] = -10

        with self.assertRaisesRegex(
            RequirementValidationError,
            "must be greater than 0",
        ):
            parse_requirement_data(data)

    def test_rejects_zero_model_size(self) -> None:
        data = copy.deepcopy(VALID_REQUEST)
        data["constraints"]["maximum_model_size_mb"] = 0

        with self.assertRaisesRegex(
            RequirementValidationError,
            "must be greater than 0",
        ):
            parse_requirement_data(data)

    def test_rejects_non_finite_numbers_for_all_constraints(self) -> None:
        fields = (
            "minimum_map50_95",
            "maximum_median_latency_ms",
            "maximum_model_size_mb",
        )
        for field in fields:
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(field=field, value=value):
                    data = copy.deepcopy(VALID_REQUEST)
                    data["constraints"][field] = value
                    with self.assertRaisesRegex(
                        RequirementValidationError,
                        "must be a finite number",
                    ):
                        parse_requirement_data(data)

    def test_rejects_boolean_and_overflowing_constraint_numbers(self) -> None:
        for field in (
            "minimum_map50_95",
            "maximum_median_latency_ms",
            "maximum_model_size_mb",
        ):
            for value in (True, 10**10000):
                with self.subTest(field=field, value=type(value).__name__):
                    data = copy.deepcopy(VALID_REQUEST)
                    data["constraints"][field] = value
                    with self.assertRaisesRegex(
                        RequirementValidationError,
                        "must be a finite number",
                    ):
                        parse_requirement_data(data)

    def test_loads_yaml_and_json_but_rejects_non_finite_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "request.yaml"
            for value in (".nan", ".inf", "-.inf"):
                with self.subTest(format="yaml", value=value):
                    path.write_text(
                        "schema_version: '1.0'\n"
                        "request_id: edge_cpu_demo\n"
                        "target: {task: object_detection, device: cpu, dataset: KITTI}\n"
                        "constraints: {minimum_map50_95: 0.25, maximum_median_latency_ms: "
                        f"{value}, maximum_model_size_mb: 6.0}}\n"
                        "preferences: {optimization_goal: balanced}\n",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        RequirementValidationError,
                        "maximum_median_latency_ms.*finite",
                    ):
                        load_requirement(path)

            path = Path(temporary_directory) / "request.json"
            json_text = json.dumps(VALID_REQUEST).replace(
                '"maximum_median_latency_ms": 35.0',
                '"maximum_median_latency_ms": 1e309',
            )
            path.write_text(json_text, encoding="utf-8")
            with self.assertRaisesRegex(
                RequirementValidationError,
                "maximum_median_latency_ms.*finite",
            ):
                load_requirement(path)

    def test_cli_rejects_non_finite_input_without_success_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "request.yaml"
            path.write_text(
                "schema_version: '1.0'\n"
                "request_id: edge_cpu_demo\n"
                "target: {task: object_detection, device: cpu, dataset: KITTI}\n"
                "constraints: {minimum_map50_95: 0.25, maximum_median_latency_ms: 35.0, "
                "maximum_model_size_mb: .inf}\n"
                "preferences: {optimization_goal: balanced}\n",
                encoding="utf-8",
            )
            stdout = StringIO()
            stderr = StringIO()
            with patch("sys.argv", ["requirement-parser", str(path)]):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as error:
                        main()

            self.assertEqual(error.exception.code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("maximum_model_size_mb", stderr.getvalue())

    def test_rejects_missing_request_id(self) -> None:
        data = copy.deepcopy(VALID_REQUEST)
        del data["request_id"]

        with self.assertRaisesRegex(
            RequirementValidationError,
            "Missing field",
        ):
            parse_requirement_data(data)

    def test_rejects_unknown_field(self) -> None:
        data = copy.deepcopy(VALID_REQUEST)
        data["unexpected_field"] = "invalid"

        with self.assertRaisesRegex(
            RequirementValidationError,
            "Unknown field",
        ):
            parse_requirement_data(data)

    def test_rejects_unsupported_device(self) -> None:
        data = copy.deepcopy(VALID_REQUEST)
        data["target"]["device"] = "cuda"

        with self.assertRaisesRegex(
            RequirementValidationError,
            "Unsupported target device",
        ):
            parse_requirement_data(data)

    def test_rejects_invalid_optimization_goal(self) -> None:
        data = copy.deepcopy(VALID_REQUEST)
        data["preferences"]["optimization_goal"] = "fastest"

        with self.assertRaisesRegex(
            RequirementValidationError,
            "Unsupported optimization goal",
        ):
            parse_requirement_data(data)


if __name__ == "__main__":
    unittest.main()