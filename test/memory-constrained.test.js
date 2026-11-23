import assert from "node:assert/strict";
import { test } from "node:test";
import { run } from "./helpers.js";

test("stage that cannot hold even one microbatch is statically infeasible", () => {
  const report = run({
    stages: [
      { memory: { capacityMb: 1, paramMb: 100, gradMb: 100 } },
      {},
      {},
    ],
  });
  assert.equal(report.feasible, false);
  const reason = report.reasons.find((item) => item.code === "memory");
  assert.ok(reason);
  assert.equal(reason.stage, "s0");
  assert.ok(reason.message.includes("one resident microbatch"));
});

test("1F1B warm-up residency beyond capacity is infeasible with details", () => {
  const report = run({
    tpSize: 1,
    stages: [{ memory: { capacityMb: 230, paramMb: 50, gradMb: 50 } }, {}, {}],
  });
  assert.equal(report.feasible, false);
  const reason = report.reasons.find((item) => item.code === "memory");
  assert.equal(reason.warmupResidentMicrobatches, 3);
  assert.ok(reason.requiredMb > 230);
});

test("a feasible tight-memory run reports peak, headroom and stays within capacity", () => {
  const report = run({
    tpSize: 1,
    stages: [
      { memory: { capacityMb: 448, paramMb: 50, gradMb: 50 } },
      { memory: { capacityMb: 448, paramMb: 50, gradMb: 50 } },
      { memory: { capacityMb: 448, paramMb: 50, gradMb: 50 } },
    ],
  });
  assert.equal(report.feasible, true);
  for (const row of report.stageMemory) {
    assert.ok(row.peakMb <= row.capacityMb);
    assert.ok(row.headroomMb >= 0);
    assert.equal(row.withinCapacity, true);
  }
});

test("larger TP shards reduce per-GPU activation and persistent footprint", () => {
  const tp1 = run({ tpSize: 1 });
  const tp4 = run({
    tpSize: 4,
    stages: [
      { memory: { capacityMb: 4000 } },
      { memory: { capacityMb: 4000 } },
      { memory: { capacityMb: 4000 } },
    ],
  });
  assert.ok(tp4.stageMemory[0].peakMb < tp1.stageMemory[0].peakMb);
});

test("capacity headroom shrinks when more microbatches are kept resident", () => {
  const four = run();
  const eight = run({ numMicrobatches: 8 });
  assert.ok(
    eight.stageMemory[0].warmupResidentMicrobatches >=
      four.stageMemory[0].warmupResidentMicrobatches,
  );
});
