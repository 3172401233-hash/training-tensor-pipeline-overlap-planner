import assert from "node:assert/strict";
import { plan } from "../src/planner.js";

export function baseInput(overrides = {}) {
  const input = {
    tpSize: 2,
    ppSize: 3,
    numMicrobatches: 4,
    allreduceAlgorithm: "ring",
    stages: [
      {
        id: "s0",
        rail: 0,
        forwardMs: 10,
        backwardMs: 20,
        optimizerMs: 5,
        fwdActivationMb: 8,
        bwdActivationMb: 8,
        memory: { capacityMb: 2000, paramMb: 100, gradMb: 100 },
      },
      {
        id: "s1",
        rail: 0,
        forwardMs: 10,
        backwardMs: 20,
        optimizerMs: 5,
        fwdActivationMb: 8,
        bwdActivationMb: 8,
        memory: { capacityMb: 2000, paramMb: 100, gradMb: 100 },
      },
      {
        id: "s2",
        rail: 0,
        forwardMs: 10,
        backwardMs: 20,
        optimizerMs: 5,
        fwdActivationMb: 8,
        bwdActivationMb: 8,
        memory: { capacityMb: 2000, paramMb: 100, gradMb: 100 },
      },
    ],
    topology: {
      railBandwidthGbPerSec: 400,
      crossRailBandwidthGbPerSec: 25,
      crossRailHopMs: 0.5,
      links: [],
    },
  };
  return deepMerge(structuredClone(input), overrides);
}

function deepMerge(target, patch) {
  if (patch && typeof patch === "object") {
    if (Array.isArray(patch)) {
      const mergeObjects = patch.every((item) => item && typeof item === "object");
      if (!mergeObjects) return structuredClone(patch);
      return patch.map((item, index) => deepMerge(target[index] ?? {}, item));
    }
    for (const key of Object.keys(patch)) {
      if (patch[key] && typeof patch[key] === "object" && !Array.isArray(patch[key])) {
        target[key] = deepMerge(target[key] ?? {}, patch[key]);
      } else if (Array.isArray(patch[key])) {
        target[key] = deepMerge(target[key] ?? [], patch[key]);
      } else {
        target[key] = structuredClone(patch[key]);
      }
    }
  }
  return target;
}

export function run(overrides = {}) {
  return plan(baseInput(overrides));
}

export function findEvent(report, predicate) {
  const event = report.timeline.find(predicate);
  assert.ok(event, "expected a timeline event matching the predicate");
  return event;
}

export function overlaps(a, b) {
  return a.startMs < b.finishMs && b.startMs < a.finishMs;
}

export function eventsOnResource(report, resource) {
  return report.timeline
    .filter((event) => event.resource === resource)
    .sort((a, b) => a.startMs - b.startMs);
}

export function assertNoResourceOverlap(report, resource) {
  const events = eventsOnResource(report, resource);
  for (let index = 1; index < events.length; index += 1) {
    assert.ok(
      events[index].startMs >= events[index - 1].finishMs - 1e-6,
      `resource ${resource} has overlapping events ${events[index - 1].id} and ${events[index].id}`,
    );
  }
}

export function totalIdleMs(report, stageIndex) {
  return report.idleIntervals
    .filter((idle) => idle.stageIndex === stageIndex)
    .reduce((sum, idle) => sum + idle.durationMs, 0);
}
