"""Tests for the shared neutral tool runtime."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from app.graph.state import CandidateProgram, MaskCatalog, PlannerExecutionStep
from app.services.tool_runtime import execute_chain, execute_tool_lab_chain
from app.services.tool_runtime.mask_runtime import normalized_mask_signature, record_mask_catalog_item
from app.tools.common import ToolExecutionResult


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
            source="model",
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

    def test_mask_catalog_reuses_generated_mask_inside_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.png"
            mask_path = Path(temp_dir) / "face_mask.png"
            preview_path = Path(temp_dir) / "face_preview.png"
            Image.new("RGB", (24, 24), (128, 128, 128)).save(input_path)
            Image.new("L", (24, 24), 0).save(mask_path)
            mask_image = Image.open(mask_path).convert("L")
            for x in range(6, 18):
                for y in range(6, 18):
                    mask_image.putpixel((x, y), 255)
            mask_image.save(mask_path)
            Image.new("RGBA", (24, 24), (255, 0, 0, 120)).save(preview_path)

            outputs: list[str] = []

            def fake_generate_mask(*_args, **_kwargs):
                return SimpleNamespace(
                    binary_mask_path=str(mask_path),
                    segmentation_rgba_path=str(preview_path),
                    provider="fal_sam3",
                    requested_provider="fal_sam3",
                    target_label="face",
                    prompt="face",
                    negative_prompt=None,
                    semantic_type=False,
                    fallback_used=False,
                    request_id="mask_1",
                    api_chain=["fal_sam3"],
                    attempt_index=0,
                    attempt_strategy="primary",
                    requested_prompt="face",
                    effective_prompt="face",
                    revert_mask=False,
                    attempts=[],
                )

            def fake_invoke(*, tool_name, tool_args, writer):
                output_path = Path(temp_dir) / f"out_{len(outputs)}.png"
                Image.new("RGB", (24, 24), (140, 140, 140)).save(output_path)
                outputs.append(str(output_path))
                return ToolExecutionResult(
                    ok=True,
                    tool=tool_name,
                    output_image=str(output_path),
                    applied_params={"params": tool_args},
                )

            program = CandidateProgram(
                id="candidate_mask_reuse",
                label="遮罩复用",
                focus="subject_cleanup",
                source="model",
                summary="测试同链路遮罩复用",
                steps=[
                    PlannerExecutionStep(op="adjust_exposure", region="face", params={"strength": 0.2, "mask_prompt": "face"}),
                    PlannerExecutionStep(op="adjust_exposure", region="face", params={"strength": 0.1, "mask_prompt": "face"}),
                ],
            )

            with (
                patch("app.services.tool_runtime.single_tool_executor.generate_mask", side_effect=fake_generate_mask) as generate_mock,
                patch("app.services.tool_runtime.single_tool_executor.invoke_tool_node", side_effect=fake_invoke),
            ):
                result = execute_chain(input_image_path=str(input_path), program=program, mask_catalog=MaskCatalog())

        self.assertEqual(generate_mock.call_count, 1)
        self.assertEqual(len(result.execution_trace), 2)
        self.assertEqual(len(result.segmentation_trace), 1)
        self.assertEqual(result.execution_trace[0]["mask_path"], str(mask_path))
        self.assertEqual(result.execution_trace[1]["mask_path"], str(mask_path))

    def test_cached_mask_is_resized_for_current_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.png"
            cached_mask_path = Path(temp_dir) / "preview_face_mask.png"
            Image.new("RGB", (32, 24), (128, 128, 128)).save(input_path)
            Image.new("L", (8, 6), 255).save(cached_mask_path)
            signature, payload = normalized_mask_signature({"provider": "auto", "prompt": "face"}, region="face") or ("", {})
            catalog = record_mask_catalog_item(
                MaskCatalog(),
                signature=signature,
                payload=payload,
                focus="subject_cleanup",
                op_name="adjust_exposure",
                region_label="face",
                mask_path=str(cached_mask_path),
                preview_path=None,
            )
            used_masks: list[str] = []

            def fake_invoke(*, tool_name, tool_args, writer):
                used_masks.append(str(tool_args["mask_path"]))
                output_path = Path(temp_dir) / "out.png"
                Image.new("RGB", (32, 24), (140, 140, 140)).save(output_path)
                return ToolExecutionResult(
                    ok=True,
                    tool=tool_name,
                    output_image=str(output_path),
                    applied_params={"params": tool_args},
                )

            program = CandidateProgram(
                id="candidate_cached_mask",
                label="缓存遮罩",
                focus="subject_cleanup",
                source="model",
                summary="测试缓存遮罩缩放",
                steps=[PlannerExecutionStep(op="adjust_exposure", region="face", params={"strength": 0.2, "mask_prompt": "face"})],
            )

            with (
                patch("app.services.tool_runtime.single_tool_executor.generate_mask", side_effect=AssertionError("mask should be cached")),
                patch("app.services.tool_runtime.single_tool_executor.invoke_tool_node", side_effect=fake_invoke),
            ):
                execute_chain(input_image_path=str(input_path), program=program, mask_catalog=catalog)

            with Image.open(used_masks[0]) as resized_mask:
                resized_size = resized_mask.size

        self.assertEqual(resized_size, (32, 24))


if __name__ == "__main__":
    unittest.main()
