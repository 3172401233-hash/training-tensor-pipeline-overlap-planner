/**
 * Top-level planning entry point: validate input, run static feasibility
 * checks, simulate the schedule, and assemble a self-describing report.
 */

import { normalizeInput, collectAssumptions } from "./schema.js";
import { staticFeasibility } from "./model.js";
import { schedule } from "./scheduler.js";

export function plan(rawInput) {
  const normalized = normalizeInput(rawInput);
  if (!normalized.ok) {
    return {
      feasible: false,
      reasons: normalized.errors.map((message) => ({ code: "invalid-input", message })),
    };
  }
  const input = normalized.input;
  const staticCheck = staticFeasibility(input);
  const warningCodes = new Set(["microbatch-shortage", "implicit-cross-rail"]);
  const hardReasons = staticCheck.reasons.filter((reason) => !warningCodes.has(reason.code));
  const warnings = staticCheck.reasons.filter((reason) => warningCodes.has(reason.code));

  const baseReport = {
    feasible: hardReasons.length === 0,
    assumptions: collectAssumptions(input),
    input: sanitizeInput(input),
    warnings,
  };

  if (hardReasons.length) {
    return { ...baseReport, feasible: false, reasons: hardReasons };
  }

  const result = schedule(input);
  if (!result.feasible) {
    return { ...baseReport, feasible: false, reasons: result.reasons };
  }

  return {
    ...baseReport,
    feasible: true,
    makespanMs: result.makespanMs,
    timeline: result.timeline,
    idleIntervals: result.idleIntervals,
    bottleneckLinks: result.bottlenecks,
    stageMemory: result.stageMemory,
    gpuUtilization: result.gpuUtilization,
  };
}

function sanitizeInput(input) {
  return {
    tpSize: input.tpSize,
    ppSize: input.ppSize,
    numMicrobatches: input.numMicrobatches,
    allreduceAlgorithm: input.allreduceAlgorithm,
    schedule: input.schedule,
    stages: input.stages.map((stage) => ({
      id: stage.id,
      rail: stage.rail,
      forwardMs: stage.forwardMs,
      backwardMs: stage.backwardMs,
      optimizerMs: stage.optimizerMs,
      fwdActivationMb: stage.fwdActivationMb,
      bwdActivationMb: stage.bwdActivationMb,
      memory: { ...stage.memory },
    })),
    topology: {
      railBandwidthGbPerSec: input.topology.railBandwidthGbPerSec,
      crossRailBandwidthGbPerSec: input.topology.crossRailBandwidthGbPerSec,
      crossRailHopMs: input.topology.crossRailHopMs,
      links: input.topology.links,
    },
    extraDependencies: input.extraDependencies,
  };
}
