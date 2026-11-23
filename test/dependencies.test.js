import assert from "node:assert/strict";
import { test } from "node:test";
import { run } from "./helpers.js";

test("valid extra dependency edge is honored in the timeline", () => {
  const report = run({
    dependencies: {
      extraDependencies: [{ from: "s0-f0-ar", to: "s0-f1" }],
    },
  });
  assert.equal(report.feasible, true);
  const ar = report.timeline.find((event) => event.id === "s0-f0-ar");
  const forward = report.timeline.find((event) => event.id === "s0-f1");
  assert.ok(ar.finishMs <= forward.startMs + 1e-6);
});

test("unknown task id in an extra edge is an invalid-dependency infeasibility", () => {
  const report = run({
    dependencies: {
      extraDependencies: [{ from: "ghost", to: "s0-f0" }],
    },
  });
  assert.equal(report.feasible, false);
  assert.ok(report.reasons.some((reason) => reason.code === "invalid-dependency"));
});

test("cyclic extra edge is reported as dependency-cycle without crashing", () => {
  const report = run({
    dependencies: {
      extraDependencies: [{ from: "s0-opt", to: "s0-f0" }],
    },
  });
  assert.equal(report.feasible, false);
  assert.deepEqual(
    report.reasons.map((reason) => reason.code),
    ["dependency-cycle"],
  );
});

test("infeasible reports never carry timeline fields", () => {
  const report = run({ numMicrobatches: 1, stages: [{ memory: { capacityMb: 1 } }, {}, {}] });
  assert.equal(report.feasible, false);
  assert.equal(report.timeline, undefined);
  assert.ok(report.reasons.length > 0);
});
