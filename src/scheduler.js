/**
 * Event-driven list scheduler.
 *
 * GPU ops (forward/backward/optimizer) serialize per stage; rail AllReduces and
 * pipeline link transfers run off-GPU and are therefore free to overlap with
 * compute on other resources. Earliest-ready wins, with deterministic
 * tie-breaking, so the produced timeline is reproducible.
 */

import { KIND, buildDag, validateDag, warmupDepth } from "./dag.js";
import { activationMbPerGpu, persistentMbPerGpu } from "./model.js";

const EPS = 1e-9;

function round(value) {
  return Math.round(value * 1e6) / 1e6;
}

function warmupMemoryFeasible(input) {
  const reasons = [];
  for (let stageIndex = 0; stageIndex < input.ppSize; stageIndex += 1) {
    const stage = input.stages[stageIndex];
    const depth = warmupDepth(input.ppSize, stageIndex, input.numMicrobatches);
    const required =
      persistentMbPerGpu(stage, input.tpSize) + depth * activationMbPerGpu(stage, input.tpSize);
    if (required > stage.memory.capacityMb + EPS) {
      reasons.push({
        code: "memory",
        stage: stage.id,
        stageIndex,
        capacityMb: stage.memory.capacityMb,
        warmupResidentMicrobatches: depth,
        requiredMb: round(required),
        message:
          `stage ${stage.id}: 1F1B warm-up keeps ${depth} microbatches resident ` +
          `(${round(required)} MB) which exceeds capacity ${stage.memory.capacityMb} MB`,
      });
    }
  }
  return reasons;
}

/**
 * Run the scheduler.
 * @returns {{ feasible: true, timeline: object[], ... } | { feasible: false, reasons: object[] }}
 */
export function schedule(input) {
  const warmupReasons = warmupMemoryFeasible(input).filter(
    (reason) => reason.code !== "microbatch-shortage",
  );
  if (warmupReasons.length) return { feasible: false, reasons: warmupReasons };

  const { tasks, links } = buildDag(input);
  const dagCheck = validateDag(tasks, input.extraDependencies);
  if (dagCheck.ok) {
    for (const edge of input.extraDependencies) {
      tasks.find((task) => task.id === edge.to).deps.push(edge.from);
    }
  }
  if (!dagCheck.ok) return { feasible: false, reasons: [dagCheck.reason] };

  const remainingDeps = new Map(tasks.map((task) => [task.id, new Set(task.deps)]));
  const state = new Map(tasks.map((task) => [task.id, { status: "pending", finishAt: 0 }]));
  const resourceFreeAt = new Map();
  const scheduled = [];
  let now = 0;

  const pendingCount = () =>
    tasks.reduce((count, task) => count + (state.get(task.id).status !== "done" ? 1 : 0), 0);

  const markFinished = (upToTime) => {
    for (const task of tasks) {
      const taskState = state.get(task.id);
      if (taskState.status === "running" && taskState.finishAt <= upToTime + EPS) {
        taskState.status = "done";
        for (const [id, deps] of remainingDeps) deps.delete(task.id);
      }
    }
  };

  const readyTasks = () =>
    tasks.filter(
      (task) =>
        state.get(task.id).status === "pending" && remainingDeps.get(task.id).size === 0,
    );

  const residentAt = (stageIndex, atTime) => {
    let resident = persistentMbPerGpu(input.stages[stageIndex], input.tpSize);
    for (const event of scheduled) {
      if (event.stageIndex !== stageIndex) continue;
      if (event.kind === KIND.FORWARD || event.kind === KIND.BACKWARD) {
        if (event.finishMs <= atTime + EPS) {
          resident += event.memoryDeltaMb;
        } else if (event.startMs <= atTime + EPS) {
          resident += Math.abs(event.memoryDeltaMb);
        }
      }
    }
    return resident;
  };

  const earliestStart = (task) => {
    let start = now;
    for (const depId of task.deps) {
      const depState = state.get(depId);
      if (depState.status !== "done") return null;
      start = Math.max(start, depState.finishAt);
    }
    start = Math.max(start, resourceFreeAt.get(task.resource) ?? 0);
    return start;
  };

  let guard = 0;
  while (pendingCount() > 0) {
    markFinished(now);
    const candidates = readyTasks();
    const choices = [];

    for (const task of candidates) {
      const start = earliestStart(task);
      if (start === null || start > now + EPS) continue;
      if (task.kind === KIND.FORWARD) {
        const stage = input.stages[task.stageIndex];
        const projected = residentAt(task.stageIndex, now) + Math.abs(task.memoryDeltaMb);
        if (projected > stage.memory.capacityMb + EPS) continue;
      }
      choices.push({ task, start });
    }

    if (choices.length === 0) {
      if (pendingCount() === 0) break;
      const runningFinish = tasks
        .filter((task) => state.get(task.id).status === "running")
        .map((task) => state.get(task.id).finishAt);
      const dependencyReady = tasks
        .filter((task) => state.get(task.id).status === "pending")
        .map((task) =>
          Math.max(
            ...task.deps.map((depId) => {
              const depState = state.get(depId);
              return depState.status === "done" ? depState.finishAt : Infinity;
            }),
            resourceFreeAt.get(task.resource) ?? 0,
          ),
        )
        .filter((time) => Number.isFinite(time));
      const wakeAt = Math.min(...runningFinish, ...dependencyReady);
      if (!Number.isFinite(wakeAt) || guard > tasks.length * 4) {
        return {
          feasible: false,
          reasons: [
            {
              code: "memory",
              message:
                "scheduler deadlocked: no task can start without exceeding a stage memory capacity",
            },
          ],
        };
      }
      now = wakeAt;
      guard += 1;
      continue;
    }

    choices.sort(
      (a, b) =>
        a.start - b.start ||
        a.task.stageIndex - b.task.stageIndex ||
        (a.task.microbatch ?? -1) - (b.task.microbatch ?? -1) ||
        a.task.id.localeCompare(b.task.id),
    );
    const { task, start } = choices[0];
    const finish = start + task.durationMs;
    state.get(task.id).status = "running";
    state.get(task.id).finishAt = finish;
    resourceFreeAt.set(task.resource, finish);
    scheduled.push({
      id: task.id,
      kind: task.kind,
      stageIndex: task.stageIndex,
      rail: task.rail ?? input.stages[task.stageIndex].rail,
      microbatch: task.microbatch,
      resource: task.resource,
      resourceType: task.resourceType,
      startMs: round(start),
      finishMs: round(finish),
      durationMs: round(task.durationMs),
      sizeMb: task.sizeMb,
      deps: [...task.deps],
      link: task.link
        ? {
            from: task.link.from,
            to: task.link.to,
            type: task.link.type,
            bandwidthGbPerSec: task.link.bandwidthGbPerSec,
            implicit: task.link.implicit,
          }
        : undefined,
    });
    guard = 0;
  }

  const makespan = round(Math.max(...scheduled.map((event) => event.finishMs)));
  return {
    feasible: true,
    makespanMs: makespan,
    timeline: scheduled.sort((a, b) => a.startMs - b.startMs || a.id.localeCompare(b.id)),
    idleIntervals: collectIdleIntervals(input, scheduled, makespan),
    bottlenecks: collectBottlenecks(input, links, scheduled, makespan),
    stageMemory: collectMemory(input),
    gpuUtilization: collectGpuUtilization(input, scheduled, makespan),
  };
}

function collectIdleIntervals(input, scheduled, makespan) {
  const intervals = [];
  for (let stageIndex = 0; stageIndex < input.ppSize; stageIndex += 1) {
    const gpuEvents = scheduled
      .filter((event) => event.resourceType === "gpu" && event.stageIndex === stageIndex)
      .sort((a, b) => a.startMs - b.startMs);
    let cursor = 0;
    for (const event of gpuEvents) {
      if (event.startMs > cursor + EPS) {
        intervals.push({
          stage: input.stages[stageIndex].id,
          stageIndex,
          startMs: round(cursor),
          endMs: event.startMs,
          durationMs: round(event.startMs - cursor),
          cause: classifyIdle(input, scheduled, stageIndex, cursor, event.startMs),
        });
      }
      cursor = Math.max(cursor, event.finishMs);
    }
    if (cursor < makespan - EPS) {
      intervals.push({
        stage: input.stages[stageIndex].id,
        stageIndex,
        startMs: round(cursor),
        endMs: makespan,
        durationMs: round(makespan - cursor),
        cause: classifyIdle(input, scheduled, stageIndex, cursor, makespan),
      });
    }
  }
  return intervals.sort((a, b) => a.startMs - b.startMs || a.stageIndex - b.stageIndex);
}

function classifyIdle(input, scheduled, stageIndex, start, end) {
  if (input.numMicrobatches < input.ppSize) {
    const gpuEvents = scheduled
      .filter((event) => event.resourceType === "gpu" && event.stageIndex === stageIndex)
      .sort((a, b) => a.startMs - b.startMs);
    const workBefore = gpuEvents.some((event) => event.finishMs <= start + EPS);
    const workAfter = gpuEvents.some((event) => event.startMs >= end - EPS);
    if (!workBefore || !workAfter) return "pipeline-bubble-microbatch-shortage";
  }

  const activeTransfers = scheduled.filter(
    (event) =>
      event.resourceType === "link" &&
      event.finishMs > start + EPS &&
      event.startMs < end - EPS &&
      (event.link?.from === stageIndex || event.link?.to === stageIndex),
  );
  if (activeTransfers.some((event) => event.link?.type === "cross-rail")) {
    return "awaiting-cross-rail-transfer";
  }
  if (activeTransfers.length) return "awaiting-pipeline-transfer";

  const activeCollective = scheduled.find(
    (event) =>
      event.resourceType === "rail" &&
      event.rail === input.stages[stageIndex].rail &&
      event.finishMs > start + EPS &&
      event.startMs < end - EPS,
  );
  if (activeCollective) return "awaiting-allreduce";

  if (input.numMicrobatches < input.ppSize) return "pipeline-bubble-microbatch-shortage";
  return "pipeline-bubble";
}

function collectBottlenecks(input, links, scheduled, makespan) {
  const busy = new Map();
  for (const event of scheduled) {
    if (event.resourceType !== "link") continue;
    const key = event.resource;
    busy.set(key, (busy.get(key) ?? 0) + event.durationMs);
  }
  const rows = links.map((link) => {
    const key = `link:${link.from}->${link.to}`;
    const busyMs = round(busy.get(key) ?? 0);
    return {
      resource: key,
      from: link.from,
      to: link.to,
      type: link.type,
      implicit: link.implicit,
      bandwidthGbPerSec: link.bandwidthGbPerSec,
      busyMs,
      utilization: round(makespan > 0 ? busyMs / makespan : 0),
      transferredMb: round(
        scheduled
          .filter((event) => event.resource === key)
          .reduce((sum, event) => sum + event.sizeMb, 0),
      ),
    };
  });
  rows.sort((a, b) => b.utilization - a.utilization || a.resource.localeCompare(b.resource));
  if (rows.length) rows[0].bottleneck = true;
  return rows;
}

function collectMemory(input) {
  return input.stages.map((stage, stageIndex) => {
    const depth = warmupDepth(input.ppSize, stageIndex, input.numMicrobatches);
    const persistent = persistentMbPerGpu(stage, input.tpSize);
    const peak = persistent + depth * activationMbPerGpu(stage, input.tpSize);
    return {
      stage: stage.id,
      stageIndex,
      capacityMb: stage.memory.capacityMb,
      persistentMb: round(persistent),
      warmupResidentMicrobatches: depth,
      peakMb: round(peak),
      headroomMb: round(stage.memory.capacityMb - peak),
      withinCapacity: peak <= stage.memory.capacityMb + EPS,
    };
  });
}

function collectGpuUtilization(input, scheduled, makespan) {
  return input.stages.map((stage, stageIndex) => {
    const busyMs = scheduled
      .filter((event) => event.resourceType === "gpu" && event.stageIndex === stageIndex)
      .reduce((sum, event) => sum + event.durationMs, 0);
    return {
      stage: stage.id,
      stageIndex,
      busyMs: round(busyMs),
      idleMs: round(makespan - busyMs),
      utilization: round(makespan > 0 ? busyMs / makespan : 0),
    };
  });
}
