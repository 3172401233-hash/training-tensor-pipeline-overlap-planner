/**
 * Input validation and normalization.
 *
 * Every timing / bandwidth / memory number used by the planner is an explicit
 * input assumption. The planner contains no hardware-specific constants: when a
 * value is omitted a documented default is applied and the assumption is echoed
 * back in `result.assumptions`.
 */

const POSITIVE_INTEGER_FIELDS = ["tpSize", "ppSize", "numMicrobatches"];

export const DEFAULTS = Object.freeze({
  forwardMs: 10,
  backwardMs: 20,
  optimizerMs: 5,
  fwdActivationMb: 16,
  bwdActivationMb: 16,
  paramMb: 64,
  gradMb: 64,
  optimizerStateMbPerParam: 2,
  weightsMbPerParam: 1,
  railBandwidthGbPerSec: 400,
  crossRailBandwidthGbPerSec: 50,
  crossRailHopMs: 0.5,
  allreduceAlgorithm: "ring",
  schedule: "1f1b",
});

function fail(errors, message) {
  errors.push(message);
}

function assertPositiveNumber(errors, value, path) {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    fail(errors, `${path} must be a finite positive number (got ${JSON.stringify(value)})`);
    return false;
  }
  return true;
}

function assertNonNegativeNumber(errors, value, path) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    fail(errors, `${path} must be a finite non-negative number (got ${JSON.stringify(value)})`);
    return false;
  }
  return true;
}

function normalizeStage(rawStage, stageIndex, global, errors) {
  const path = `stages[${stageIndex}]`;
  const stage = rawStage && typeof rawStage === "object" ? rawStage : {};

  const pick = (key, fallback) => {
    const value = stage[key] ?? global[key] ?? fallback;
    if (!assertPositiveNumber(errors, value, `${path}.${key}`)) return fallback;
    return value;
  };

  const pickMemory = () => {
    const memory = stage.memory ?? global.memory ?? {};
    if (typeof memory !== "object" || memory === null) {
      fail(errors, `${path}.memory must be an object`);
      return null;
    }
    const capacityMb = memory.capacityMb;
    if (!assertPositiveNumber(errors, capacityMb, `${path}.memory.capacityMb`)) return null;
    return {
      capacityMb,
      paramMb: memory.paramMb ?? global.paramMb ?? DEFAULTS.paramMb,
      gradMb: memory.gradMb ?? global.gradMb ?? DEFAULTS.gradMb,
      optimizerStateMbPerParam:
        memory.optimizerStateMbPerParam ??
        global.optimizerStateMbPerParam ??
        DEFAULTS.optimizerStateMbPerParam,
      weightsMbPerParam:
        memory.weightsMbPerParam ?? global.weightsMbPerParam ?? DEFAULTS.weightsMbPerParam,
    };
  };

  const memory = pickMemory();
  return {
    id: stage.id ?? `stage-${stageIndex}`,
    rail: Number.isInteger(stage.rail) ? stage.rail : stageIndex,
    forwardMs: pick("forwardMs", DEFAULTS.forwardMs),
    backwardMs: pick("backwardMs", DEFAULTS.backwardMs),
    optimizerMs: pick("optimizerMs", DEFAULTS.optimizerMs),
    fwdActivationMb: pick("fwdActivationMb", DEFAULTS.fwdActivationMb),
    bwdActivationMb: pick("bwdActivationMb", DEFAULTS.bwdActivationMb),
    memory,
  };
}

function normalizeTopology(raw, tpSize, ppSize, errors) {
  const topology = raw && typeof raw === "object" ? raw : {};
  const links = Array.isArray(topology.links) ? topology.links : [];
  const normalizedLinks = links.map((link, index) => {
    const path = `topology.links[${index}]`;
    if (!link || typeof link !== "object") {
      fail(errors, `${path} must be an object`);
      return null;
    }
    const { from, to, bandwidthGbPerSec, hopMs = 0, type = "rail" } = link;
    if (!Number.isInteger(from) || from < 0 || from >= ppSize) {
      fail(errors, `${path}.from must be a stage index in [0, ${ppSize - 1}]`);
    }
    if (!Number.isInteger(to) || to < 0 || to >= ppSize) {
      fail(errors, `${path}.to must be a stage index in [0, ${ppSize - 1}]`);
    }
    assertPositiveNumber(errors, bandwidthGbPerSec, `${path}.bandwidthGbPerSec`);
    assertNonNegativeNumber(errors, hopMs, `${path}.hopMs`);
    if (type !== "rail" && type !== "cross-rail") {
      fail(errors, `${path}.type must be "rail" or "cross-rail"`);
    }
    return { from, to, bandwidthGbPerSec, hopMs, type };
  });

  const railBandwidthGbPerSec =
    topology.railBandwidthGbPerSec ?? DEFAULTS.railBandwidthGbPerSec;
  const crossRailBandwidthGbPerSec =
    topology.crossRailBandwidthGbPerSec ?? DEFAULTS.crossRailBandwidthGbPerSec;
  const crossRailHopMs = topology.crossRailHopMs ?? DEFAULTS.crossRailHopMs;
  assertPositiveNumber(errors, railBandwidthGbPerSec, "topology.railBandwidthGbPerSec");
  assertPositiveNumber(errors, crossRailBandwidthGbPerSec, "topology.crossRailBandwidthGbPerSec");
  assertNonNegativeNumber(errors, crossRailHopMs, "topology.crossRailHopMs");

  return {
    tpSize,
    railBandwidthGbPerSec,
    crossRailBandwidthGbPerSec,
    crossRailHopMs,
    links: normalizedLinks.filter(Boolean),
  };
}

function normalizeDependencies(raw, numStages, errors) {
  const extra = raw?.extraDependencies;
  if (extra === undefined) return [];
  if (!Array.isArray(extra)) {
    fail(errors, "dependencies.extraDependencies must be an array");
    return [];
  }
  return extra.map((edge, index) => {
    const path = `dependencies.extraDependencies[${index}]`;
    if (!edge || typeof edge !== "object") {
      fail(errors, `${path} must be an object`);
      return null;
    }
    const { from, to } = edge;
    if (typeof from !== "string" || typeof to !== "string") {
      fail(errors, `${path}.from and ${path}.to must be task-id strings`);
    }
    return { from, to };
  }).filter(Boolean);
}

/**
 * Validate and normalize a planner input document.
 * @returns {{ ok: true, input: object } | { ok: false, errors: string[] }}
 */
export function normalizeInput(raw) {
  const errors = [];
  if (!raw || typeof raw !== "object") {
    return { ok: false, errors: ["input must be an object"] };
  }

  for (const field of POSITIVE_INTEGER_FIELDS) {
    const value = raw[field];
    if (!Number.isInteger(value) || value <= 0) {
      fail(errors, `${field} must be a positive integer (got ${JSON.stringify(value)})`);
    }
  }
  if (errors.length) return { ok: false, errors };

  const { tpSize, ppSize, numMicrobatches } = raw;
  if (!Array.isArray(raw.stages) || raw.stages.length !== ppSize) {
    fail(errors, `stages must be an array of length ppSize (${ppSize})`);
  }

  const schedule = raw.schedule ?? DEFAULTS.schedule;
  if (schedule !== "1f1b") fail(errors, `unsupported schedule "${schedule}" (only "1f1b" is supported)`);
  const allreduceAlgorithm = raw.allreduceAlgorithm ?? DEFAULTS.allreduceAlgorithm;
  if (allreduceAlgorithm !== "ring" && allreduceAlgorithm !== "naive") {
    fail(errors, `allreduceAlgorithm must be "ring" or "naive"`);
  }

  const globalMemory = raw.memory ?? {};
  const global = {
    forwardMs: raw.forwardMs,
    backwardMs: raw.backwardMs,
    fwdActivationMb: raw.fwdActivationMb,
    bwdActivationMb: raw.bwdActivationMb,
    paramMb: globalMemory.paramMb,
    gradMb: globalMemory.gradMb,
    optimizerStateMbPerParam: globalMemory.optimizerStateMbPerParam,
    weightsMbPerParam: globalMemory.weightsMbPerParam,
  };

  const stages = Array.isArray(raw.stages)
    ? raw.stages.map((stage, index) => normalizeStage(stage, index, global, errors))
    : [];

  const topology = normalizeTopology(raw.topology, tpSize, ppSize, errors);
  const extraDependencies = normalizeDependencies(raw.dependencies, ppSize, errors);

  if (errors.length) return { ok: false, errors };

  const input = {
    tpSize,
    ppSize,
    numMicrobatches,
    allreduceAlgorithm,
    schedule,
    stages,
    topology,
    extraDependencies,
  };
  return { ok: true, input };
}

/**
 * Assumptions that were applied while normalizing the input, suitable for
 * inclusion in the result so reports stay self-describing.
 */
export function collectAssumptions(input) {
  return {
    timeUnit: "ms",
    sizeUnit: "MB",
    bandwidthUnit: "GB/s",
    allreduceAlgorithm: input.allreduceAlgorithm,
    schedule: input.schedule,
    ringAllreduceFormula:
      "2 * (tpSize - 1) / tpSize * (tensorMb / bandwidthGBps) * 1000 ms per rail",
    pipelineTransferFormula: "tensorMb / bandwidthGBps * 1000 + hopMs",
    defaults: { ...DEFAULTS },
  };
}
