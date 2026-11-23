import assert from "node:assert/strict";
import { test } from "node:test";
import { run } from "./helpers.js";

test("fewer microbatches than stages emits a shortage warning and bubbles", () => {
  const report = run({ numMicrobatches: 2 });
  assert.equal(report.feasible, true);
  assert.equal(
    report.warnings.some((warning) => warning.code === "microbatch-shortage"),
    true,
  );
  assert.ok(
    report.idleIntervals.some(
      (idle) => idle.cause === "pipeline-bubble-microbatch-shortage",
    ),
  );
});

test("a single microbatch leaves the deepest stage idle until the first forward arrives", () => {
  const report = run({ numMicrobatches: 1 });
  assert.equal(report.feasible, true);
  const firstIdle = (stageIndex) =>
    report.idleIntervals.find((idle) => idle.stageIndex === stageIndex && idle.startMs === 0);
  assert.equal(firstIdle(0), undefined);
  assert.ok(firstIdle(2).durationMs > firstIdle(1).durationMs);
  assert.equal(firstIdle(2).cause, "pipeline-bubble-microbatch-shortage");
});

test("enough microbatches remove the shortage warning", () => {
  const report = run({ numMicrobatches: 6 });
  assert.equal(
    report.warnings.some((warning) => warning.code === "microbatch-shortage"),
    false,
  );
});

test("shortage never overrides a hard infeasibility reason", () => {
  const report = run({
    numMicrobatches: 1,
    stages: [
      { memory: { capacityMb: 1 } },
      {},
      {},
    ],
  });
  assert.equal(report.feasible, false);
  assert.ok(report.reasons.some((reason) => reason.code === "memory"));
});
