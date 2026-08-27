import copy
import unittest

from src.requirement_parser.parser import (
    RequirementValidationError,
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