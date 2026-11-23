# training-tensor-pipeline-overlap-planner

确定性的训练调度规划器：给定每层计算时间、激活大小、张量并行（TP）/流水并行（PP）
分组、链路带宽、微批数量、显存容量与依赖关系，模拟 **1F1B** 流水调度，把 TP
AllReduce、PP 激活传输与 GPU 计算重叠，并输出时间线、空闲区间、瓶颈链路和不可行原因。

运行环境：Node.js `>= 24.16.0`，零运行时依赖，使用内置 `node:test`。

## 重要：性能数字全部是输入假设

本工具 **不内置任何硬件性能常数，也不做性能预测**。所有时间、带宽、显存数字都来自
输入 JSON，或输入缺省时采用的、在 `src/schema.js` 的 `DEFAULTS` 中显式列出的文档化默认值。
输出报告的 `assumptions` 字段会回显本次使用的全部假设与解析公式。

- 计算耗时（`forwardMs` / `backwardMs` / `optimizerMs`）是你提供的测量或估算值。
- 链路带宽（GB/s）、跳变延迟（`hopMs`）、激活/权重/梯度大小（MB）均为输入。
- AllReduce 与传输时长由输入按解析公式推导，公式同样回显在 `assumptions` 中。
- 更换硬件、内核或框架版本后，需要重新提供输入假设；旧报告不能代表新环境性能。

解析模型（仅做带宽项建模，不含协议开销建模）：

- Ring AllReduce（单条 rail）：`2*(tpSize-1)/tpSize * (tensorMb / 带宽GB/s) * 1000` ms
- Naive AllReduce：`2*(tpSize-1) * (tensorMb / 带宽GB/s) * 1000` ms
- PP 链路传输：`(tensorMb / 带宽GB/s) * 1000 + hopMs` ms

## 输入

```jsonc
{
  "tpSize": 4,                 // 每条 rail 上的 GPU 数（正整数）
  "ppSize": 4,                 // 流水阶段数（正整数）
  "numMicrobatches": 8,        // 微批数量
  "allreduceAlgorithm": "ring",// "ring"（默认）或 "naive"
  "schedule": "1f1b",          // 当前仅支持 1f1b
  "stages": [
    {
      "id": "stage-0",
      "rail": 0,               // 所在轨道，决定共享哪条 TP rail
      "forwardMs": 12,         // 单微批前向计算耗时（假设值）
      "backwardMs": 24,
      "optimizerMs": 6,
      "fwdActivationMb": 24,   // 前向激活张量大小（假设值）
      "bwdActivationMb": 24,
      "memory": {
        "capacityMb": 8192,    // 单 GPU 显存容量
        "paramMb": 512,        // 本阶段参数分片前大小
        "gradMb": 512,
        "optimizerStateMbPerParam": 2, // 优化器状态 = 该倍数 × paramMb
        "weightsMbPerParam": 1
      }
    }
  ],
  "topology": {
    "railBandwidthGbPerSec": 400,
    "crossRailBandwidthGbPerSec": 50, // 未显式给链路时的跨轨假设带宽
    "crossRailHopMs": 0.5,
    "links": [
      // 可选；显式声明相邻阶段间链路。缺省时同 rail 走 rail 带宽，
      // 跨 rail 按 crossRail 假设合成并给出 implicit-cross-rail 警告
      { "from": 1, "to": 2, "type": "cross-rail",
        "bandwidthGbPerSec": 50, "hopMs": 0.5 }
    ]
  },
  "dependencies": {
    // 可选的额外依赖边，只能增加约束，不能改变参数更新顺序
    "extraDependencies": [{ "from": "s0-f0", "to": "s1-opt" }]
  }
}
```

阶段级字段可省略，省略时回退到顶层同名字段，再回退到 `DEFAULTS`。

## 调度模型与不变量

- 每个阶段一条 GPU 资源，前向/反向/优化器在该资源上严格串行，顺序由 1F1B 固定。
- 每条 rail 一个 AllReduce 资源，每条 PP 链路一个传输资源；它们不占 GPU，可与计算重叠。
- **参数更新依赖不可改变**：优化器任务只在本阶段全部梯度 AllReduce 完成后启动，
  用户额外依赖只能增加边，成环或引用未知任务会被判为不可行。
- 显存按 GPU 建模：常驻 = 参数 + 梯度 + 优化器状态（除以 `tpSize`），
  前向激活在对应反向完成前常驻；调度器带显存门控，任何峰值都不得超过 `capacityMb`。
- 调度采用事件驱动贪心列表调度：最早可启动任务优先，平局按阶段/微批/任务 ID 确定，
  因此同一输入的时间线完全可复现。

## 输出

- `feasible` / `reasons`：硬不可行原因（`memory`、`topology`、`dependency-cycle`、
  `invalid-dependency`、`invalid-input`）。
- `warnings`：不阻断调度的提示（`microbatch-shortage`、`implicit-cross-rail`）。
- `timeline`：每个任务的 `kind`、资源、起止时间、张量大小、依赖与所用链路。
- `idleIntervals`：每段 GPU 空闲及归因（等待跨轨传输 / PP 传输 / AllReduce /
  微批不足导致的流水气泡）。
- `bottleneckLinks`：各链路忙时与利用率，利用率最高者标记 `bottleneck: true`。
- `stageMemory`：每阶段常驻、暖机驻留微批数、峰值、余量、是否在容量内。
- `gpuUtilization`：每阶段忙时、空闲时长与利用率。

## 使用

```bash
# 人类可读摘要
node src/cli.js examples/two-rail-1f1b.json

# 完整 JSON 报告
node src/cli.js examples/two-rail-1f1b.json --json

# 也可从 stdin 读取
cat examples/two-rail-1f1b.json | node src/cli.js - --json

# 运行测试
node --test
npm test
```

编程接口：

```js
import { plan } from "./src/planner.js";
const report = plan(inputJson);
```

## 测试覆盖

`node --test` 运行 32 个用例，覆盖题目要求的四类场景及核心不变量：

- `test/slow-communication.test.js`：通信慢于计算、AllReduce 占满 rail、跨轨链路成为瓶颈。
- `test/microbatch-shortage.test.js`：微批少于阶段数的警告、气泡归因与硬不可行优先级。
- `test/memory-constrained.test.js`：单批都放不下、暖机驻留超限、紧容量可行与 TP 分片收益。
- `test/cross-rail.test.js`：跨轨合成链路、显式链路缺失的拓扑不可行、跳变延迟计入。
- `test/dependencies.test.js`：额外依赖生效、未知边与成环的不可行判定。
- `test/planner.test.js`：重叠正确性、资源不重叠、依赖顺序、优化器屏障、输入校验。

## 目录

- `src/schema.js`：输入校验、归一化与默认假设。
- `src/model.js`：AllReduce/传输公式、显存模型与静态可行性检查。
- `src/dag.js`：1F1B 任务 DAG 构建与环检测。
- `src/scheduler.js`：事件驱动调度、显存门控、空闲/瓶颈/显存统计。
- `src/planner.js`：顶层入口，组装自描述报告。
- `src/cli.js`：命令行入口。
