"""Tests for the shared neutral tool runtime."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.graph.state import CandidateProgram, MaskCatalog, PlannerExecutionStep
from app.services.tool_runtime import execute_chain, execute_tool_lab_chain


class ToolRuntimeTest(unittest.TestCase):
    """Verify ToolLab and the agent share the same executor layer."""

    def test_agent_and_tool_lab_share_single_tool_executor(self) -> None:
        calls: list[tuple[str | None, str | None]] = []

        def fake_single_tool_call(
            *,
            current_image,
            operation,
            execution_trace,
            segmentation_trace,
            fallback_trace,
            candidate_outputs,
            mask_catalog,
            writer,
            round_id=None,
            candidate_id=None,
            **_kwargs,
        ):
            calls.append((round_id, candidate_id))
            output = f"{current_image}.{operation['op']}.png"
            execution_trace.append(
                {
                    "index": len(execution_trace),
                    "round_id": round_id,
                    "focus": "finish",
                    "candidate_id": candidate_id,
                    "op": operation["op"],
                    "region": operation.get("region", "whole_image"),
                    "ok": True,
                    "fallback_used": False,
                    "output_image": output,
                    "applied_params": operation.get("params", {}),
                    "mask_path": operation.get("params", {}).get("mask_path"),
                }
            )
            candidate_outputs.append(output)
            return output, {"op": operation["op"], "ok": True}, mask_catalog

        program = CandidateProgram(
            id="candidate_1",
            label="候选",
            focus="finish",
            source="direct",
            summary="测试",
            steps=[PlannerExecutionStep(op="adjust_exposure", params={"strength": 0.2})],
        )

        with patch("app.services.tool_runtime.chain_executor.execute_single_tool_call", side_effect=fake_single_tool_call):
            agent_result = execute_chain(
                input_image_path="/tmp/input.png",
                program=program,
                mask_catalog=MaskCatalog(),
                round_id="round_1",
                candidate_id="candidate_1",
                focus="finish",
            )
            tool_lab_output, tool_lab_steps = execute_tool_lab_chain(
                input_image_path="/tmp/input.png",
                steps=[{"op": "adjust_exposure", "region": "whole_image", "params": {"strength": 0.2}}],
            )

        self.assertEqual(calls[0], ("round_1", "candidate_1"))
        self.assertEqual(calls[1], ("tool_lab", "tool_lab"))
        self.assertEqual(agent_result.execution_trace[0]["candidate_id"], "candidate_1")
        self.assertEqual(tool_lab_steps[0].tool_name, "adjust_exposure")
        self.assertEqual(tool_lab_output, "/tmp/input.png.adjust_exposure.png")


if __name__ == "__main__":
    unittest.main()
