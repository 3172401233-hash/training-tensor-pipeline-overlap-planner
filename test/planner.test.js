import assert from "node:assert/strict";
import { test } from "node:test";
import { allreduceMs } from "../src/model.js";
import { gpuOrder, warmupDepth, KIND } from "../src/dag.js";
import {
  run,
  baseInput,
  findEvent,
  overlaps,
  assertNoResourceOverlap,
  eventsOnResource,
} from "./helpers.js";

test("baseline run is feasible and reports every required section", () => {
  const report = run();
  assert.equal(report.feasible, true);
  assert.ok(report.makespanMs > 0);
  assert.ok(Array.isArray(report.timeline) && report.timeline.length > 0);
  assert.ok(Array.isArray(report.idleIntervals));
  assert.ok(Array.isArray(report.bottleneckLinks));
  assert.ok(Array.isArray(report.stageMemory));
  assert.ok(Array.isArray(report.gpuUtilization));
  assert.ok(report.assumptions);
  for (const row of report.stageMemory) assert.equal(row.withinCapacity, true);
});

test("AllReduce runs on the rail and overlaps the next GPU compute", () => {
  const report = run();
  const ar = findEvent(report, (event) => event.kind === KIND.FORWARD_AR && event.stageIndex === 0);
  const nextForward = findEvent(report, (event) => event.id === "s0-f1");
  assert.equal(ar.resourceType, "rail");
  assert.ok(overlaps(ar, nextForward), "rail AllReduce should overlap GPU compute of another mb");

  const expected = allreduceMs(8, 400, 2, "ring");
  assert.ok(Math.abs(ar.durationMs - expected) < 1e-6);
});

test("resources never host overlapping events and deps are respected", () => {
  const report = run();
  const resources = new Set(report.timeline.map((event) => event.resource));
  for (const resource of resources) assertNoResourceOverlap(report, resource);

  const byId = new Map(report.timeline.map((event) => [event.id, event]));
  for (const event of report.timeline) {
    for (const depId of event.deps ?? []) {
      const dep = byId.get(depId);
      if (dep) assert.ok(dep.finishMs <= event.startMs + 1e-6, `${depId} must finish before ${event.id}`);
    }
  }
});

test("parameter update only starts after every gradient AllReduce of the stage", () => {
  const report = run();
  const optimizer = findEvent(report, (event) => event.kind === KIND.OPTIMIZER && event.stageIndex === 1);
  const gradientAr = report.timeline.filter(
    (event) => event.kind === KIND.BACKWARD_AR && event.stageIndex === 1,
  );
  assert.equal(gradientAr.length, 4);
  for (const ar of gradientAr) assert.ok(ar.finishMs <= optimizer.startMs + 1e-6);
});

test("1F1B order keeps warm-up depth bounded per stage", () => {
  assert.deepEqual(
    gpuOrder(3, 0, 4).map((op) => op.kind),
    [
      KIND.FORWARD,
      KIND.FORWARD,
      KIND.FORWARD,
      KIND.BACKWARD,
      KIND.FORWARD,
      KIND.BACKWARD,
      KIND.BACKWARD,
      KIND.BACKWARD,
    ],
  );
  assert.equal(warmupDepth(3, 0, 4), 3);
  assert.equal(warmupDepth(3, 2, 4), 1);
  assert.equal(warmupDepth(3, 0, 2), 2);
});

test("all events finish by the makespan and timeline is chronologically sorted", () => {
  const report = run();
  for (const event of report.timeline) assert.ok(event.finishMs <= report.makespanMs + 1e-6);
  for (let index = 1; index < report.timeline.length; index += 1) {
    assert.ok(report.timeline[index].startMs >= report.timeline[index - 1].startMs - 1e-9);
  }
});

test("invalid input is rejected with structured errors", () => {
  const report = run({ tpSize: 0 });
  assert.equal(report.feasible, false);
  assert.ok(report.reasons.some((reason) => reason.code === "invalid-input"));
});

test("tpSize 1 emits no collective events and still schedules", () => {
  const report = run({ tpSize: 1 });
  assert.equal(report.feasible, true);
  assert.equal(
    report.timeline.some((event) => event.resourceType === "rail"),
    false,
  );
  assert.equal(eventsOnResource(report, "gpu:0").length, 9);
});
