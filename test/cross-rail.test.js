import assert from "node:assert/strict";
import { test } from "node:test";
import { run, findEvent } from "./helpers.js";

test("adjacent stages on different rails synthesize a cross-rail link", () => {
  const report = run({
    stages: [{ rail: 0 }, { rail: 1 }, { rail: 1 }],
  });
  assert.equal(report.feasible, true);
  const cross = report.bottleneckLinks.filter((link) => link.type === "cross-rail");
  assert.equal(cross.length, 2);
  for (const link of cross) assert.equal(link.implicit, true);

  const xfer = findEvent(
    report,
    (event) => event.resource === "link:0->1" && event.kind === "forward-transfer",
  );
  assert.equal(xfer.link.type, "cross-rail");
  assert.equal(xfer.link.bandwidthGbPerSec, 25);
});

test("missing link between rails is a hard topology infeasibility", () => {
  const report = run({
    stages: [{ rail: 0 }, { rail: 1 }, { rail: 1 }],
    topology: {
      links: [
        { from: 1, to: 2, type: "rail", bandwidthGbPerSec: 400, hopMs: 0 },
      ],
    },
  });
  assert.equal(report.feasible, false);
  const reason = report.reasons.find((item) => item.code === "topology");
  assert.ok(reason);
  assert.match(reason.message, /different rails/);
});

test("explicit cross-rail link removes the topology infeasibility", () => {
  const report = run({
    stages: [{ rail: 0 }, { rail: 1 }, { rail: 2 }],
    topology: {
      links: [
        { from: 0, to: 1, type: "cross-rail", bandwidthGbPerSec: 40, hopMs: 0.3 },
        { from: 1, to: 2, type: "cross-rail", bandwidthGbPerSec: 40, hopMs: 0.3 },
      ],
    },
  });
  assert.equal(report.feasible, true);
  assert.equal(
    report.bottleneckLinks.filter((link) => link.type === "cross-rail").length,
    4,
  );
});

test("cross-rail hop latency is included in every transfer duration", () => {
  const report = run({
    stages: [{ rail: 0 }, { rail: 1 }, { rail: 1 }],
    topology: { crossRailBandwidthGbPerSec: 10, crossRailHopMs: 2 },
  });
  const xfer = findEvent(report, (event) => event.resource === "link:0->1");
  const expected = (xfer.sizeMb / 10) * 1000 + 2;
  assert.ok(Math.abs(xfer.durationMs - expected) < 1e-6);
});

test("rail-only topology never schedules a cross-rail transfer", () => {
  const report = run();
  assert.equal(
    report.timeline.some((event) => event.link?.type === "cross-rail"),
    false,
  );
});
