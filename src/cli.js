#!/usr/bin/env node
import { readFileSync } from "node:fs";
import { plan } from "./planner.js";

function printSummary(report) {
  if (!report.feasible) {
    console.log("INFEASIBLE");
    for (const reason of report.reasons) {
      console.log(`- [${reason.code}] ${reason.message}`);
    }
    if (report.warnings?.length) {
      for (const warning of report.warnings) console.log(`- [warning] ${warning.message}`);
    }
    process.exitCode = 2;
    return;
  }

  console.log(`FEASIBLE — makespan ${report.makespanMs} ms`);
  console.log("\nGPU utilization:");
  for (const row of report.gpuUtilization) {
    console.log(
      `  ${row.stage}: busy ${row.busyMs} ms, idle ${row.idleMs} ms, utilization ${row.utilization}`,
    );
  }
  console.log("\nMemory peaks:");
  for (const row of report.stageMemory) {
    console.log(
      `  ${row.stage}: peak ${row.peakMb} / ${row.capacityMb} MB (headroom ${row.headroomMb} MB)`,
    );
  }
  console.log("\nBottleneck links:");
  for (const link of report.bottleneckLinks) {
    const flag = link.bottleneck ? "  <== bottleneck" : "";
    console.log(
      `  ${link.resource} [${link.type}] ${link.bandwidthGbPerSec} GB/s, ` +
        `busy ${link.busyMs} ms, utilization ${link.utilization}${flag}`,
    );
  }
  if (report.idleIntervals.length) {
    console.log("\nIdle intervals:");
    for (const idle of report.idleIntervals) {
      console.log(
        `  ${idle.stage}: ${idle.startMs}-${idle.endMs} ms (${idle.durationMs} ms, ${idle.cause})`,
      );
    }
  }
  for (const warning of report.warnings) console.log(`\n[warning] ${warning.message}`);
}

const args = process.argv.slice(2);
const asJson = args.includes("--json");
const file = args.find((arg) => !arg.startsWith("--"));
if (!file) {
  console.error('usage: tp-pp-planner <input.json> (use "-" to read JSON from stdin)');
  process.exit(1);
}

const raw = file === "-" ? readFileSync(0, "utf8") : readFileSync(file, "utf8");
let parsed;
try {
  parsed = JSON.parse(raw);
} catch (error) {
  console.error(`invalid JSON: ${error.message}`);
  process.exit(1);
}

const report = plan(parsed);
if (asJson) {
  console.log(JSON.stringify(report, null, 2));
} else {
  printSummary(report);
}
