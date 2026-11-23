/**
 * Construction of the per-iteration task DAG for a 1F1B pipeline.
 *
 * Task kinds:
 *   forward            GPU compute of one microbatch on one stage
 *   forward-allreduce  TP rail collective after the forward layer
 *   forward-transfer   activation send over a PP link to the next stage
 *   backward           GPU compute of one microbatch on one stage
 *   backward-allreduce TP gradient collective on the rail
 *   backward-transfer  gradient-activation send back over a PP link
 *   optimizer          parameter update (runs after every gradient AR)
 *
 * Parameter-update dependencies are structural and never reordered:
 * optimizer(s) starts only after all backward-allreduce tasks of stage s.
 */

import { allreduceMs, resolveLink, transferMs, activationMbPerGpu } from "./model.js";

export const KIND = Object.freeze({
  FORWARD: "forward",
  FORWARD_AR: "forward-allreduce",
  FORWARD_XFER: "forward-transfer",
  BACKWARD: "backward",
  BACKWARD_AR: "backward-allreduce",
  BACKWARD_XFER: "backward-transfer",
  OPTIMIZER: "optimizer",
});

/** Number of warm-up forwards at a stage under 1F1B (clamped to microbatch count). */
export function warmupDepth(ppSize, stageIndex, numMicrobatches) {
  return Math.min(numMicrobatches, ppSize - stageIndex);
}

/**
 * Fixed GPU op order for a stage: warm-up forwards, then alternating
 * backward/forward in steady state, then cool-down backwards.
 */
export function gpuOrder(ppSize, stageIndex, numMicrobatches) {
  const warmup = warmupDepth(ppSize, stageIndex, numMicrobatches);
  const order = [];
  for (let microbatch = 0; microbatch < warmup; microbatch += 1) {
    order.push({ kind: KIND.FORWARD, microbatch });
  }
  for (let microbatch = warmup; microbatch < numMicrobatches; microbatch += 1) {
    order.push({ kind: KIND.BACKWARD, microbatch: microbatch - warmup });
    order.push({ kind: KIND.FORWARD, microbatch });
  }
  for (let microbatch = numMicrobatches - warmup; microbatch < numMicrobatches; microbatch += 1) {
    order.push({ kind: KIND.BACKWARD, microbatch });
  }
  return order;
}

function gpuResource(stageIndex) {
  return `gpu:${stageIndex}`;
}

function railResource(rail) {
  return `rail:${rail}`;
}

function linkResource(fromIndex, toIndex) {
  return `link:${fromIndex}->${toIndex}`;
}

/**
 * Build the full task list plus resource metadata for an iteration.
 * @returns {{ tasks: object[], resources: object, links: object[] }}
 */
export function buildDag(input) {
  const { tpSize, ppSize, numMicrobatches, stages, topology, allreduceAlgorithm } = input;
  const tasks = [];
  const addTask = (task) => {
    tasks.push({ deps: [], ...task });
    return task.id;
  };

  const fwdArId = Array.from({ length: ppSize }, () => new Array(numMicrobatches));
  const bwdArId = Array.from({ length: ppSize }, () => new Array(numMicrobatches));
  const fwdXferId = Array.from({ length: ppSize }, () => new Array(numMicrobatches));
  const bwdXferId = Array.from({ length: ppSize }, () => new Array(numMicrobatches));
  const fwdId = Array.from({ length: ppSize }, () => new Array(numMicrobatches));
  const bwdId = Array.from({ length: ppSize }, () => new Array(numMicrobatches));

  const links = [];
  for (let stageIndex = 0; stageIndex < ppSize - 1; stageIndex += 1) {
    links.push(resolveLink(input, stageIndex, stageIndex + 1));
  }
  for (let stageIndex = 1; stageIndex < ppSize; stageIndex += 1) {
    links.push(resolveLink(input, stageIndex, stageIndex - 1));
  }

  for (let stageIndex = 0; stageIndex < ppSize; stageIndex += 1) {
    const stage = stages[stageIndex];
    const rail = stage.rail;
    const lastGpu = { id: null };
    const chainGpu = (id) => {
      if (lastGpu.id !== null) tasks.find((task) => task.id === id).deps.push(lastGpu.id);
      lastGpu.id = id;
    };

    for (const op of gpuOrder(ppSize, stageIndex, numMicrobatches)) {
      const microbatch = op.microbatch;
      if (op.kind === KIND.FORWARD) {
        const id = `s${stageIndex}-f${microbatch}`;
        const deps = [];
        if (stageIndex > 0) deps.push(fwdXferId[stageIndex - 1][microbatch]);
        fwdId[stageIndex][microbatch] = addTask({
          id,
          kind: KIND.FORWARD,
          stageIndex,
          microbatch,
          resource: gpuResource(stageIndex),
          resourceType: "gpu",
          durationMs: stage.forwardMs,
          sizeMb: stage.fwdActivationMb,
          memoryDeltaMb: activationMbPerGpu(stage, tpSize),
          deps: deps.filter(Boolean),
        });
        chainGpu(id);

        if (tpSize > 1) {
          const arId = `s${stageIndex}-f${microbatch}-ar`;
          fwdArId[stageIndex][microbatch] = addTask({
            id: arId,
            kind: KIND.FORWARD_AR,
            stageIndex,
            microbatch,
            resource: railResource(rail),
            resourceType: "rail",
            rail,
            durationMs: allreduceMs(
              stage.fwdActivationMb,
              topology.railBandwidthGbPerSec,
              tpSize,
              allreduceAlgorithm,
            ),
            sizeMb: stage.fwdActivationMb,
            memoryDeltaMb: 0,
            deps: [id],
          });
        }

        if (stageIndex < ppSize - 1) {
          const link = links.find((item) => item.from === stageIndex && item.to === stageIndex + 1);
          const xferId = `s${stageIndex}-f${microbatch}-xfer`;
          fwdXferId[stageIndex][microbatch] = addTask({
            id: xferId,
            kind: KIND.FORWARD_XFER,
            stageIndex,
            microbatch,
            resource: linkResource(stageIndex, stageIndex + 1),
            resourceType: "link",
            link,
            durationMs: transferMs(stage.fwdActivationMb, link),
            sizeMb: stage.fwdActivationMb,
            memoryDeltaMb: 0,
            deps: [fwdArId[stageIndex][microbatch] ?? id],
          });
        }
      } else {
        const id = `s${stageIndex}-b${microbatch}`;
        const deps = [];
        if (stageIndex < ppSize - 1) deps.push(bwdXferId[stageIndex + 1][microbatch]);
        bwdId[stageIndex][microbatch] = addTask({
          id,
          kind: KIND.BACKWARD,
          stageIndex,
          microbatch,
          resource: gpuResource(stageIndex),
          resourceType: "gpu",
          durationMs: stage.backwardMs,
          sizeMb: stage.bwdActivationMb,
          memoryDeltaMb: -activationMbPerGpu(stage, tpSize),
          deps: deps.filter(Boolean),
        });
        chainGpu(id);

        if (tpSize > 1) {
          const arId = `s${stageIndex}-b${microbatch}-ar`;
          bwdArId[stageIndex][microbatch] = addTask({
            id: arId,
            kind: KIND.BACKWARD_AR,
            stageIndex,
            microbatch,
            resource: railResource(rail),
            resourceType: "rail",
            rail,
            durationMs: allreduceMs(
              stage.memory.gradMb,
              topology.railBandwidthGbPerSec,
              tpSize,
              allreduceAlgorithm,
            ),
            sizeMb: stage.memory.gradMb,
            memoryDeltaMb: 0,
            deps: [id],
          });
        }

        if (stageIndex > 0) {
          const link = links.find((item) => item.from === stageIndex && item.to === stageIndex - 1);
          const xferId = `s${stageIndex}-b${microbatch}-xfer`;
          bwdXferId[stageIndex][microbatch] = addTask({
            id: xferId,
            kind: KIND.BACKWARD_XFER,
            stageIndex,
            microbatch,
            resource: linkResource(stageIndex, stageIndex - 1),
            resourceType: "link",
            link,
            durationMs: transferMs(stage.bwdActivationMb, link),
            sizeMb: stage.bwdActivationMb,
            memoryDeltaMb: 0,
            deps: [bwdArId[stageIndex][microbatch] ?? id],
          });
        }
      }
    }

    const gradientSyncDeps = [];
    for (let microbatch = 0; microbatch < numMicrobatches; microbatch += 1) {
      if (bwdArId[stageIndex][microbatch]) gradientSyncDeps.push(bwdArId[stageIndex][microbatch]);
      else gradientSyncDeps.push(bwdId[stageIndex][microbatch]);
    }
    addTask({
      id: `s${stageIndex}-opt`,
      kind: KIND.OPTIMIZER,
      stageIndex,
      microbatch: null,
      resource: gpuResource(stageIndex),
      resourceType: "gpu",
      durationMs: stages[stageIndex].optimizerMs,
      sizeMb: 0,
      memoryDeltaMb: 0,
      deps: gradientSyncDeps,
    });
    chainGpu(`s${stageIndex}-opt`);
  }

  return { tasks, links };
}

/**
 * Kahn cycle / dangling-edge check over the DAG, including user edges.
 * @returns {{ ok: boolean, reason?: object }}
 */
export function validateDag(tasks, extraDependencies) {
  const byId = new Map(tasks.map((task) => [task.id, task]));
  for (const edge of extraDependencies) {
    if (!byId.has(edge.from) || !byId.has(edge.to)) {
      return {
        ok: false,
        reason: {
          code: "invalid-dependency",
          message: `extra dependency ${edge.from} -> ${edge.to} references an unknown task id`,
          edge,
        },
      };
    }
  }

  const indegree = new Map(tasks.map((task) => [task.id, task.deps.length]));
  for (const edge of extraDependencies) indegree.set(edge.to, indegree.get(edge.to) + 1);
  const outgoing = new Map(tasks.map((task) => [task.id, []]));
  for (const task of tasks) {
    for (const dep of task.deps) outgoing.get(dep).push(task.id);
  }
  for (const edge of extraDependencies) outgoing.get(edge.from).push(edge.to);

  const queue = tasks.filter((task) => indegree.get(task.id) === 0).map((task) => task.id);
  let visited = 0;
  while (queue.length) {
    const id = queue.shift();
    visited += 1;
    for (const next of outgoing.get(id)) {
      indegree.set(next, indegree.get(next) - 1);
      if (indegree.get(next) === 0) queue.push(next);
    }
  }
  if (visited !== tasks.length) {
    return {
      ok: false,
      reason: {
        code: "dependency-cycle",
        message: "task dependencies contain a cycle; parameter-update ordering cannot be satisfied",
      },
    };
  }
  return { ok: true };
}
