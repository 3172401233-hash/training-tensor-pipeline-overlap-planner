/**
 * Analytical performance model.
 *
 * All formulas consume user-supplied assumptions only (see src/schema.js):
 *   - ring AllReduce per rail: 2*(N-1)/N * S/B  (bandwidth-optimal term)
 *   - naive AllReduce:         2*(N-1)   * S/B  (no ring aggregation)
 *   - pipeline link transfer:  S/B + per-hop latency
 */

/** AllReduce duration in milliseconds for one tensor-parallel rail. */
export function allreduceMs(tensorMb, bandwidthGbPerSec, tpSize, algorithm) {
  const baseMs = (tensorMb / bandwidthGbPerSec) * 1000;
  const factor =
    algorithm === "naive" ? 2 * (tpSize - 1) : (2 * (tpSize - 1)) / tpSize;
  return factor * baseMs;
}

/** Point-to-point transfer duration in milliseconds over a pipeline link. */
export function transferMs(tensorMb, link) {
  return (tensorMb / link.bandwidthGbPerSec) * 1000 + link.hopMs;
}

/**
 * Persistent memory footprint of a stage (weights + gradients + optimizer
 * state), divided across the tpSize GPUs of the rail.
 */
export function persistentMbPerGpu(stage, tpSize) {
  const memory = stage.memory;
  const optimizerStateMb = memory.paramMb * memory.optimizerStateMbPerParam;
  return (memory.paramMb + memory.gradMb + optimizerStateMb) / tpSize;
}

/**
 * Peak resident activation memory for a single microbatch on a stage.
 * Forward activations stay resident until the matching backward completes.
 * At tpSize > 1 the shard is divided across the rail.
 */
export function activationMbPerGpu(stage, tpSize) {
  return (stage.fwdActivationMb + stage.bwdActivationMb) / tpSize;
}

/**
 * Hard static feasibility checks that do not require running the simulation.
 * @returns {{ feasible: boolean, reasons: object[] }}
 */
export function staticFeasibility(input) {
  const reasons = [];
  const { tpSize, ppSize, stages, topology, numMicrobatches } = input;

  stages.forEach((stage, stageIndex) => {
    const persistent = persistentMbPerGpu(stage, tpSize);
    const oneMicrobatch = activationMbPerGpu(stage, tpSize);
    if (persistent + oneMicrobatch > stage.memory.capacityMb) {
      reasons.push({
        code: "memory",
        stage: stage.id,
        stageIndex,
        capacityMb: stage.memory.capacityMb,
        persistentMb: round(persistent),
        minRequiredMb: round(persistent + oneMicrobatch),
        message:
          `stage ${stage.id}: even one resident microbatch needs ` +
          `${round(persistent + oneMicrobatch)} MB but capacity is ${stage.memory.capacityMb} MB`,
      });
    }
  });

  const railOfStage = (index) => stages[index].rail;
  const linkByPair = new Map();
  for (const link of topology.links) {
    linkByPair.set(`${link.from}->${link.to}`, link);
  }

  const explicitTopology = topology.links.length > 0;
  for (let stageIndex = 0; stageIndex < ppSize - 1; stageIndex += 1) {
    const explicit = linkByPair.get(`${stageIndex}->${stageIndex + 1}`);
    const sameRail = railOfStage(stageIndex) === railOfStage(stageIndex + 1);
    if (explicit || sameRail) continue;
    if (explicitTopology) {
      reasons.push({
        code: "topology",
        stage: stages[stageIndex].id,
        stageIndex,
        nextStage: stages[stageIndex + 1].id,
        message:
          `explicit topology is missing a link between ${stages[stageIndex].id} and ` +
          `${stages[stageIndex + 1].id} which sit on different rails`,
      });
    } else {
      reasons.push({
        code: "implicit-cross-rail",
        stage: stages[stageIndex].id,
        stageIndex,
        nextStage: stages[stageIndex + 1].id,
        message:
          `no link declared between ${stages[stageIndex].id} and ${stages[stageIndex + 1].id}; ` +
          `assuming an implicit cross-rail link at ${topology.crossRailBandwidthGbPerSec} GB/s`,
      });
    }
  }

  if (numMicrobatches < ppSize) {
    reasons.push({
      code: "microbatch-shortage",
      message:
        `only ${numMicrobatches} microbatches for ${ppSize} pipeline stages: ` +
        `the last stage cannot start until ${ppSize} forwards are queued, so an ` +
        `unavoidable pipeline bubble is reported as idle time`,
      numMicrobatches,
      ppSize,
    });
  }

  return {
    feasible:
      reasons.filter(
        (reason) =>
          reason.code !== "microbatch-shortage" && reason.code !== "implicit-cross-rail",
      ).length === 0,
    reasons,
  };
}

function round(value) {
  return Math.round(value * 1e6) / 1e6;
}

/**
 * Resolve the link carrying activations from one stage to the next.
 * Explicit topology links win; adjacent stages on one rail share the rail
 * fabric; otherwise an implicit (slow) cross-rail link is synthesized.
 */
export function resolveLink(input, fromIndex, toIndex) {
  const explicit = input.topology.links.find(
    (link) => link.from === fromIndex && link.to === toIndex,
  );
  if (explicit) return { ...explicit, implicit: false };

  const fromStage = input.stages[fromIndex];
  const toStage = input.stages[toIndex];
  if (fromStage.rail === toStage.rail) {
    return {
      from: fromIndex,
      to: toIndex,
      bandwidthGbPerSec: input.topology.railBandwidthGbPerSec,
      hopMs: 0,
      type: "rail",
      implicit: true,
    };
  }
  return {
    from: fromIndex,
    to: toIndex,
    bandwidthGbPerSec: input.topology.crossRailBandwidthGbPerSec,
    hopMs: input.topology.crossRailHopMs,
    type: "cross-rail",
    implicit: true,
  };
}
