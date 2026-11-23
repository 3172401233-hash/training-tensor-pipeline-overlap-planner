import assert from "node:assert/strict";
import { test } from "node:test";
import { run, baseInput, totalIdleMs } from "./helpers.js";

test("slow communication dominates: makespan grows and link becomes the bottleneck", () => {
  const fast = run();
  const slow = run({
    topology: { railBandwidthGbPerSec: 400, crossRailBandwidthGbPerSec: 2, crossRailHopMs: 0.5 },
    stages: [
      { rail: 0 },
      { rail: 1 },
      { rail: 1 },
    ],
  });

  assert.equal(slow.feasible, true);
  assert.ok(slow.makespanMs > fast.makespanMs * 3);

  const bottleneck = slow.bottleneckLinks.find((link) => link.bottleneck);
  assert.equal(bottleneck.type, "cross-rail");
  assert.ok(bottleneck.busyMs > 0);
  assert.ok(
    slow.idleIntervals.some((idle) => idle.cause === "awaiting-cross-rail-transfer"),
    "GPU idle time should be attributed to the slow cross-rail link",
  );
});

test("slow AllReduce alone serializes collectives on the rail while GPUs stay busy", () => {
  const report = run({
    tpSize: 8,
    topology: { railBandwidthGbPerSec: 5 },
    stages: [
      { memory: { capacityMb: 4000 } },
      { memory: { capacityMb: 4000 } },
      { memory: { capacityMb: 4000 } },
    ],
  });
  assert.equal(report.feasible, true);

  const railEvents = report.timeline.filter((event) => event.resource === "rail:0");
  assert.ok(railEvents.length >= 2);
  for (let index = 1; index < railEvents.length; index += 1) {
    assert.ok(railEvents[index].startMs >= railEvents[index - 1].finishMs - 1e-6);
  }

  const gpuBusy = report.gpuUtilization.reduce((sum, row) => sum + row.busyMs, 0);
  const railBusy = railEvents.reduce((sum, event) => sum + event.durationMs, 0);
  assert.ok(railBusy > gpuBusy / report.input.ppSize, "collectives should occupy substantial rail time");
});

test("naive AllReduce is slower than ring under equal assumptions", () => {
  const ring = run({ allreduceAlgorithm: "ring" });
  const naive = run({ allreduceAlgorithm: "naive" });
  assert.ok(naive.makespanMs >= ring.makespanMs);
});

test("cross-rail explicit link is preferred over the implicit fallback", () => {
  const withLink = run({
    stages: [{ rail: 0 }, { rail: 1 }, { rail: 1 }],
    topology: {
      links: [
        { from: 0, to: 1, type: "cross-rail", bandwidthGbPerSec: 80, hopMs: 0.1 },
      ],
    },
  });
  const implicit = run({
    stages: [{ rail: 0 }, { rail: 1 }, { rail: 1 }],
    topology: { crossRailBandwidthGbPerSec: 10, crossRailHopMs: 1 },
  });
  assert.ok(withLink.makespanMs < implicit.makespanMs);
  const link = withLink.bottleneckLinks.find((item) => item.resource === "link:0->1");
  assert.equal(link.implicit, false);
  assert.equal(link.bandwidthGbPerSec, 80);
});

test("baseInput helper builds independent clones", () => {
  const a = baseInput();
  const b = baseInput();
  assert.notEqual(a.stages, b.stages);
  assert.equal(totalIdleMs(run(), 0) >= 0, true);
});
