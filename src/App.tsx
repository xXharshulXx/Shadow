import { useEffect, useMemo, useRef, useState } from "react";
import {
  ALGORITHM_LABELS,
  DEFAULT_END,
  DEFAULT_START,
  GRID_COLS,
  GRID_ROWS,
  type AlgorithmId,
  type Position,
  generateGuaranteedSolvableMaze,
  keyFromCoords,
  positionKey,
  positionsEqual,
  runMultiStagePathfinding,
} from "./lib/pathfinding";
import { cn } from "./utils/cn";

type CellKind = "empty" | "start" | "end" | "wall" | "visited" | "path" | "checkpoint";
type DragMode = "wall" | "erase" | "start" | "end" | null;

type Stats = {
  nodesVisited: number;
  pathLength: number;
  executionTime: number;
  status: "Ready" | "Running" | "Complete" | "No path";
};

type RunRecord = {
  id: number;
  algorithm: AlgorithmId;
  nodesVisited: number;
  pathLength: number;
  executionTime: number;
  found: boolean;
};

const SPEED_SETTINGS = [
  { label: "Slow", delay: 24 },
  { label: "Medium", delay: 9 },
  { label: "Fast", delay: 2 },
];

const algorithmGuide: Record<
  AlgorithmId,
  {
    summary: string;
    method: string;
    complexity: string;
    bestFor: string;
    guarantee: string;
  }
> = {
  bfs: {
    summary: "BFS expands outward in layers from the start node, checking all cells one move away before cells two moves away.",
    method: "Queue based: first node discovered is the first node explored.",
    complexity: "Time O(V + E), space O(V).",
    bestFor: "Unweighted grids where every move has the same cost.",
    guarantee: "Always finds the shortest path in this unweighted grid.",
  },
  dijkstra: {
    summary: "Dijkstra's algorithm repeatedly chooses the unsettled node with the smallest known distance from the start.",
    method: "Priority queue based: distance from the start controls exploration order.",
    complexity: "Time O((V + E) log V), space O(V).",
    bestFor: "Weighted graphs or routing problems where costs can differ.",
    guarantee: "Always finds the shortest path when all edge weights are non-negative.",
  },
  astar: {
    summary: "A* adds a Manhattan-distance estimate to guide the search toward the end node instead of expanding evenly.",
    method: "Priority queue based: cost so far plus estimated distance to the goal controls exploration order.",
    complexity: "Worst-case O((V + E) log V), space O(V).",
    bestFor: "Maps and grids where a good heuristic can point the search toward the target.",
    guarantee: "Finds the shortest path here because Manhattan distance is admissible for four-direction movement.",
  },
};

const initialStats = (): Stats => ({
  nodesVisited: 0,
  pathLength: 0,
  executionTime: 0,
  status: "Ready",
});

const sleep = (milliseconds: number) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="min-w-[7rem] rounded-2xl border border-slate-200/80 bg-white/70 px-3 py-2 shadow-sm shadow-slate-200/50 backdrop-blur dark:border-slate-700/70 dark:bg-slate-900/70 dark:shadow-none">
      <div className="text-[0.68rem] font-semibold uppercase tracking-[0.16em] text-slate-500 dark:text-slate-400">{label}</div>
      <div className="mt-1 text-lg font-semibold tabular-nums text-slate-950 dark:text-white">{value}</div>
    </div>
  );
}

function ControlButton({
  children,
  onClick,
  disabled,
  variant = "secondary",
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  variant?: "primary" | "secondary";
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "rounded-xl px-4 py-2.5 text-sm font-semibold transition duration-200 focus:outline-none focus:ring-2 focus:ring-sky-400 focus:ring-offset-2 focus:ring-offset-transparent disabled:cursor-not-allowed disabled:opacity-50",
        variant === "primary"
          ? "bg-sky-600 text-white shadow-lg shadow-sky-500/20 hover:bg-sky-500"
          : "border border-slate-200 bg-white text-slate-700 shadow-sm shadow-slate-200/60 hover:border-slate-300 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:shadow-none dark:hover:border-slate-600 dark:hover:bg-slate-800",
      )}
    >
      {children}
    </button>
  );
}

function LegendItem({ className, label }: { className: string; label: string }) {
  return (
    <div className="flex items-center gap-2 text-xs font-medium text-slate-600 dark:text-slate-300">
      <span className={cn("h-3.5 w-3.5 rounded-[0.25rem] border border-white/40 shadow-sm", className)} />
      {label}
    </div>
  );
}

export default function App() {
  const [algorithm, setAlgorithm] = useState<AlgorithmId>("astar");
  const [speedIndex, setSpeedIndex] = useState(1);
  const [interactionMode, setInteractionMode] = useState<"wall" | "checkpoint">("wall");
  const [checkpoints, setCheckpoints] = useState<Position[]>([]);
  const [walls, setWalls] = useState<Set<string>>(new Set());
  const [visitedCells, setVisitedCells] = useState<Set<string>>(new Set());
  const [pathCells, setPathCells] = useState<Set<string>>(new Set());
  const [start, setStart] = useState<Position>(DEFAULT_START);
  const [end, setEnd] = useState<Position>(DEFAULT_END);
  const [dragMode, setDragMode] = useState<DragMode>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [stats, setStats] = useState<Stats>(initialStats);
  const [history, setHistory] = useState<RunRecord[]>([]);
  const [currentStep, setCurrentStep] = useState(
    "Choose an algorithm, draw walls/checkpoints, and start the visualization.",
  );

  const runIdRef = useRef(0);
  const speedDelayRef = useRef(SPEED_SETTINGS[speedIndex].delay);
  const lastDraggedCellRef = useRef<string | null>(null);

  const selectedGuide = algorithmGuide[algorithm];

  const gridRows = useMemo(() => Array.from({ length: GRID_ROWS }, (_, row) => row), []);
  const gridCols = useMemo(() => Array.from({ length: GRID_COLS }, (_, col) => col), []);

  useEffect(() => {
    speedDelayRef.current = SPEED_SETTINGS[speedIndex].delay;
  }, [speedIndex]);

  useEffect(() => {
    const stopDragging = () => {
      setDragMode(null);
      lastDraggedCellRef.current = null;
    };

    window.addEventListener("pointerup", stopDragging);
    window.addEventListener("pointercancel", stopDragging);

    return () => {
      window.removeEventListener("pointerup", stopDragging);
      window.removeEventListener("pointercancel", stopDragging);
    };
  }, []);

  const cancelCurrentRun = () => {
    runIdRef.current += 1;
    setIsRunning(false);
  };

  const clearSearchState = () => {
    setVisitedCells(new Set());
    setPathCells(new Set());
    setStats(initialStats());
  };

  const prepareForGridEdit = () => {
    cancelCurrentRun();
    clearSearchState();
    setHistory([]);
  };

  const clearPath = () => {
    cancelCurrentRun();
    clearSearchState();
    setCurrentStep("Path data cleared. Walls, start, and end nodes were kept in place.");
  };

  const resetGrid = () => {
    cancelCurrentRun();
    setWalls(new Set());
    setCheckpoints([]);
    setStart(DEFAULT_START);
    setEnd(DEFAULT_END);
    clearSearchState();
    setHistory([]);
    setCurrentStep("Grid reset to an empty 20 by 40 environment.");
  };

  const generateMaze = () => {
    cancelCurrentRun();
    setCheckpoints([]);
    setWalls(generateGuaranteedSolvableMaze(GRID_ROWS, GRID_COLS, start, end));
    clearSearchState();
    setHistory([]);
    setCurrentStep("Guaranteed solvable maze generated. Start a search to evaluate the route.");
  };

  const applyWallMode = (position: Position, mode: "wall" | "erase") => {
    if (positionsEqual(position, start) || positionsEqual(position, end)) {
      return;
    }

    const key = positionKey(position);
    setWalls((previousWalls) => {
      const nextWalls = new Set(previousWalls);

      if (mode === "wall") {
        nextWalls.add(key);
      } else {
        nextWalls.delete(key);
      }

      return nextWalls;
    });
  };

  const moveSpecialNode = (position: Position, mode: "start" | "end") => {
    if (mode === "start" && positionsEqual(position, end)) {
      return;
    }

    if (mode === "end" && positionsEqual(position, start)) {
      return;
    }

    if (mode === "start") {
      setStart(position);
    } else {
      setEnd(position);
    }

    setWalls((previousWalls) => {
      const nextWalls = new Set(previousWalls);
      nextWalls.delete(positionKey(position));
      return nextWalls;
    });
  };

  const applyDragToCell = (row: number, col: number) => {
    if (!dragMode || isRunning) {
      return;
    }

    const key = keyFromCoords(row, col);

    if (lastDraggedCellRef.current === key) {
      return;
    }

    lastDraggedCellRef.current = key;
    const position = { row, col };

    if (dragMode === "wall" || dragMode === "erase") {
      applyWallMode(position, dragMode);
      return;
    }

    moveSpecialNode(position, dragMode);
  };

  const handlePointerDown = (row: number, col: number, event: React.PointerEvent<HTMLButtonElement>) => {
    if (isRunning) {
      return;
    }

    event.preventDefault();
    prepareForGridEdit();

    const position = { row, col };
    const key = positionKey(position);
    lastDraggedCellRef.current = key;

    if (positionsEqual(position, start)) {
      setDragMode("start");
      setCurrentStep("Dragging the start node. Drop it on any open cell.");
      return;
    }

    if (positionsEqual(position, end)) {
      setDragMode("end");
      setCurrentStep("Dragging the end node. Drop it on any open cell.");
      return;
    }

    if (interactionMode === "checkpoint") {
      if (walls.has(key)) {
        setCurrentStep("Cannot place a checkpoint on a wall.");
        return;
      }
      const isCp = checkpoints.some(cp => positionsEqual(position, cp));
      if (isCp) {
        setCheckpoints(prev => prev.filter(cp => !positionsEqual(position, cp)));
        setCurrentStep("Removed checkpoint.");
      } else {
        setCheckpoints(prev => [...prev, position]);
        setCurrentStep(`Added checkpoint ${checkpoints.length + 1}.`);
      }
      return;
    }

    const nextMode: "wall" | "erase" = walls.has(key) ? "erase" : "wall";
    setDragMode(nextMode);
    applyWallMode(position, nextMode);
    setCurrentStep(nextMode === "wall" ? "Drawing walls to block movement." : "Erasing walls to reopen cells.");
  };

  const handleGridPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!dragMode || isRunning) {
      return;
    }

    const element = document.elementFromPoint(event.clientX, event.clientY);
    const cell = element?.closest<HTMLButtonElement>("[data-grid-cell='true']");

    if (!cell) {
      return;
    }

    applyDragToCell(Number(cell.dataset.row), Number(cell.dataset.col));
  };

  const startVisualization = async () => {
    if (isRunning) {
      return;
    }

    runIdRef.current += 1;
    const runId = runIdRef.current;

    setIsRunning(true);
    setVisitedCells(new Set());
    setPathCells(new Set());
    setStats({ nodesVisited: 0, pathLength: 0, executionTime: 0, status: "Running" });
    setCurrentStep(`${ALGORITHM_LABELS[algorithm]} is preparing the frontier and checking the current grid.`);

    await sleep(120);

    if (runIdRef.current !== runId) {
      return;
    }

    const result = runMultiStagePathfinding(algorithm, {
      rows: GRID_ROWS,
      cols: GRID_COLS,
      start,
      end,
      checkpoints,
      walls,
    });

    const finalPathLength = result.found ? Math.max(0, result.path.length - 1) : 0;

    for (let index = 0; index < result.visitedOrder.length; index += 1) {
      if (runIdRef.current !== runId) {
        return;
      }

      const step = result.visitedOrder[index];
      const key = positionKey(step.position);

      setCurrentStep(step.description);
      setVisitedCells((previousVisited) => {
        const nextVisited = new Set(previousVisited);
        nextVisited.add(key);
        return nextVisited;
      });
      setStats((previousStats) => ({
        ...previousStats,
        nodesVisited: index + 1,
      }));

      await sleep(speedDelayRef.current);
    }

    if (runIdRef.current !== runId) {
      return;
    }

    if (result.found) {
      setCurrentStep(`Shortest path found. Reconstructing ${finalPathLength} moves from end back to start.`);

      for (let index = 0; index < result.path.length; index += 1) {
        if (runIdRef.current !== runId) {
          return;
        }

        const key = positionKey(result.path[index]);
        setPathCells((previousPath) => {
          const nextPath = new Set(previousPath);
          nextPath.add(key);
          return nextPath;
        });
        await sleep(Math.max(7, speedDelayRef.current * 1.7));
      }

      setCurrentStep(`${ALGORITHM_LABELS[algorithm]} completed. The yellow cells show the shortest path.`);
    } else {
      setCurrentStep(`${ALGORITHM_LABELS[algorithm]} explored all reachable cells but no path exists.`);
    }

    const completedStats: Stats = {
      nodesVisited: result.visitedOrder.length,
      pathLength: finalPathLength,
      executionTime: result.executionTime,
      status: result.found ? "Complete" : "No path",
    };

    setStats(completedStats);
    setHistory((previousHistory) => [
      {
        id: Date.now(),
        algorithm,
        nodesVisited: completedStats.nodesVisited,
        pathLength: completedStats.pathLength,
        executionTime: completedStats.executionTime,
        found: result.found,
      },
      ...previousHistory,
    ].slice(0, 5));
    setIsRunning(false);
  };

  const getCellKind = (row: number, col: number): CellKind => {
    const position = { row, col };
    const key = positionKey(position);

    if (positionsEqual(position, start)) {
      return "start";
    }

    if (positionsEqual(position, end)) {
      return "end";
    }

    if (checkpoints.some(cp => positionsEqual(position, cp))) {
      return "checkpoint";
    }

    if (walls.has(key)) {
      return "wall";
    }

    if (pathCells.has(key)) {
      return "path";
    }

    if (visitedCells.has(key)) {
      return "visited";
    }

    return "empty";
  };

  const getCellClassName = (kind: CellKind) => {
    if (kind === "start") {
      return "cell-start border-emerald-500 bg-emerald-600 text-white shadow-md shadow-emerald-600/30 font-bold";
    }

    if (kind === "end") {
      return "cell-end border-rose-500 bg-rose-600 text-white shadow-md shadow-rose-600/30 font-bold";
    }

    if (kind === "checkpoint") {
      return "cell-checkpoint border-violet-500 bg-violet-600 text-white shadow-md shadow-violet-600/30 font-bold";
    }

    if (kind === "wall") {
      return "cell-wall border-neutral-950 bg-neutral-950";
    }

    if (kind === "path") {
      return "cell-path border-amber-400 bg-amber-400 text-neutral-950 font-bold shadow-md shadow-amber-400/40";
    }

    if (kind === "visited") {
      return "cell-visited border-sky-600/40 bg-sky-500/30";
    }

    return "border-neutral-800 bg-neutral-900 hover:bg-neutral-800";
  };

  return (
    <div className="min-h-screen bg-[#0f0f11] text-slate-100 font-sans">
      <header className="mx-auto max-w-[1500px] px-4 pb-4 pt-5 sm:px-6">
        <div className="mb-5 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.24em] text-sky-400">Pathfinding & Optimization Visualizer</p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight text-white sm:text-4xl">
              SHADOW
            </h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-300 sm:text-base">
              xXharshulXx | Pareen Lodha
            </p>
          </div>
        </div>

        <section className="rounded-3xl border border-slate-800 bg-slate-900/50 p-4 shadow-xl backdrop-blur">
          <div className="grid gap-4 xl:grid-cols-[minmax(13rem,17rem)_1fr_auto] xl:items-end">
            <label className="block">
              <span className="text-xs font-bold uppercase tracking-[0.16em] text-slate-400">Algorithm</span>
              <select
                value={algorithm}
                disabled={isRunning}
                onChange={(event) => setAlgorithm(event.target.value as AlgorithmId)}
                className="mt-2 w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm font-semibold text-slate-100 shadow-sm outline-none transition focus:border-sky-500 focus:ring-2 focus:ring-sky-500/20 disabled:opacity-60"
              >
                <option value="dijkstra">Dijkstra's Algorithm</option>
                <option value="astar">A* Search</option>
                <option value="bfs">Breadth-First Search (BFS)</option>
              </select>
            </label>

            <div className="flex flex-wrap gap-2">
              <ControlButton variant="primary" onClick={startVisualization} disabled={isRunning}>
                {isRunning ? "Visualizing..." : "Start Visualization"}
              </ControlButton>
              <ControlButton onClick={resetGrid}>Reset Grid</ControlButton>
              <ControlButton onClick={clearPath}>Clear Path</ControlButton>
              <ControlButton onClick={generateMaze}>Generate Maze</ControlButton>
              
              <button
                type="button"
                disabled={isRunning}
                onClick={() => setInteractionMode(interactionMode === "wall" ? "checkpoint" : "wall")}
                className={cn(
                  "rounded-xl px-4 py-2.5 text-sm font-semibold transition duration-200 focus:outline-none focus:ring-2 focus:ring-sky-400 focus:ring-offset-2 focus:ring-offset-transparent disabled:opacity-50",
                  interactionMode === "checkpoint"
                    ? "bg-violet-600 text-white shadow-lg shadow-violet-500/30 hover:bg-violet-500"
                    : "border border-slate-700 bg-slate-950 text-slate-200 hover:bg-slate-800"
                )}
              >
                Mode: {interactionMode === "wall" ? "Draw Walls" : "Add Checkpoints"}
              </button>
            </div>

            <div className="min-w-[14rem]">
              <div className="flex items-center justify-between gap-4">
                <span className="text-xs font-bold uppercase tracking-[0.16em] text-slate-500 dark:text-slate-400">Speed</span>
                <span className="text-sm font-semibold text-slate-800 dark:text-slate-100">{SPEED_SETTINGS[speedIndex].label}</span>
              </div>
              <input
                type="range"
                min={0}
                max={2}
                step={1}
                value={speedIndex}
                onChange={(event) => setSpeedIndex(Number(event.target.value))}
                className="mt-3 w-full accent-sky-600"
                aria-label="Animation speed"
              />
              <div className="mt-1 flex justify-between text-[0.68rem] font-semibold uppercase tracking-[0.12em] text-slate-400">
                <span>Slow</span>
                <span>Medium</span>
                <span>Fast</span>
              </div>
            </div>
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Metric label="Nodes Visited" value={stats.nodesVisited} />
            <Metric label="Path Length" value={stats.pathLength} />
            <Metric label="Execution" value={`${stats.executionTime.toFixed(2)} ms`} />
            <Metric label="Status" value={stats.status} />
          </div>
        </section>
      </header>

      <main className="mx-auto grid max-w-[1500px] grid-cols-1 gap-5 px-4 pb-8 sm:px-6 xl:grid-cols-[minmax(0,1fr)_360px]">
        <section className="rounded-3xl border border-slate-800 bg-slate-900/40 p-4 shadow-xl">
          <div className="flex flex-col gap-3 pb-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <h2 className="text-xl font-bold tracking-tight text-white">Grid Environment</h2>
              <p className="mt-1 text-sm text-slate-300">
                Click/drag to draw walls. In Checkpoints mode, click to add numbered stops.
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <LegendItem className="bg-emerald-600" label="Start" />
              <LegendItem className="bg-rose-600" label="End" />
              <LegendItem className="bg-violet-600" label="Checkpoint" />
              <LegendItem className="bg-neutral-950" label="Wall" />
              <LegendItem className="bg-sky-500/30" label="Visited" />
              <LegendItem className="bg-amber-400" label="Path" />
            </div>
          </div>

          <div className="mb-4 rounded-2xl border border-sky-500/20 bg-sky-500/10 px-4 py-3 text-sm font-medium text-sky-100">
            Current step: {currentStep}
          </div>

          <div
            className="path-grid"
            style={{ gridTemplateColumns: `repeat(${GRID_COLS}, minmax(0, 1fr))` }}
            onPointerMove={handleGridPointerMove}
            onPointerLeave={() => {
              setDragMode(null);
              lastDraggedCellRef.current = null;
            }}
          >
            {gridRows.map((row) =>
              gridCols.map((col) => {
                const kind = getCellKind(row, col);
                const label = `${kind} cell at row ${row}, column ${col}`;
                const cpIndex = checkpoints.findIndex(cp => positionsEqual({row, col}, cp));

                return (
                  <button
                    key={keyFromCoords(row, col)}
                    type="button"
                    data-grid-cell="true"
                    data-row={row}
                    data-col={col}
                    aria-label={label}
                    tabIndex={-1}
                    onPointerDown={(event) => handlePointerDown(row, col, event)}
                    onPointerEnter={() => applyDragToCell(row, col)}
                    className={cn(
                      "grid-cell border text-[0.55rem] font-black uppercase transition duration-150",
                      isRunning ? "cursor-progress" : kind === "start" || kind === "end" ? "cursor-grab" : "cursor-crosshair",
                      getCellClassName(kind),
                    )}
                  >
                    {kind === "start" ? "S" : kind === "end" ? "E" : kind === "checkpoint" ? (cpIndex + 1).toString() : ""}
                  </button>
                );
              }),
            )}
          </div>
        </section>

        <aside className="space-y-5">
          <section className="rounded-3xl border border-slate-800 bg-slate-900/40 p-5 shadow-xl">
            <p className="text-xs font-bold uppercase tracking-[0.2em] text-sky-400">Methodology</p>
            <h2 className="mt-2 text-2xl font-bold tracking-tight text-white">{ALGORITHM_LABELS[algorithm]}</h2>
            <p className="mt-3 text-sm leading-6 text-slate-300">{selectedGuide.summary}</p>

            <div className="mt-5 space-y-3 text-sm">
              <div className="rounded-2xl bg-slate-950/70 p-3">
                <div className="font-semibold text-white">How it works</div>
                <p className="mt-1 leading-6 text-slate-300">{selectedGuide.method}</p>
              </div>
              <div className="rounded-2xl bg-slate-950/70 p-3">
                <div className="font-semibold text-white">Complexity</div>
                <p className="mt-1 leading-6 text-slate-300">{selectedGuide.complexity}</p>
              </div>
              <div className="rounded-2xl bg-slate-950/70 p-3">
                <div className="font-semibold text-white">Correctness</div>
                <p className="mt-1 leading-6 text-slate-300">{selectedGuide.guarantee}</p>
              </div>
              <div className="rounded-2xl bg-slate-950/70 p-3">
                <div className="font-semibold text-white">Used when</div>
                <p className="mt-1 leading-6 text-slate-300">{selectedGuide.bestFor}</p>
              </div>
            </div>
          </section>

          <section className="rounded-3xl border border-slate-800 bg-slate-900/40 p-5 shadow-xl">
            <p className="text-xs font-bold uppercase tracking-[0.2em] text-sky-400">Key differences</p>
            <div className="mt-4 space-y-3 text-sm text-slate-300">
              <p><span className="font-semibold text-white">BFS:</span> simplest shortest-path search for equal-cost moves.</p>
              <p><span className="font-semibold text-white">Dijkstra:</span> shortest-path search that also works when edge costs vary.</p>
              <p><span className="font-semibold text-white">A*:</span> usually visits fewer nodes by using Manhattan distance as a goal-directed heuristic.</p>
            </div>
          </section>

          <section className="rounded-3xl border border-slate-800 bg-slate-900/40 p-5 shadow-xl">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-xs font-bold uppercase tracking-[0.2em] text-sky-400">Results</p>
                <h2 className="mt-2 text-xl font-bold tracking-tight text-white">Run comparison</h2>
              </div>
              <span className="rounded-full bg-slate-800 px-3 py-1 text-xs font-semibold text-slate-300">
                Last {history.length}
              </span>
            </div>

            {history.length === 0 ? (
              <p className="mt-4 text-sm leading-6 text-slate-300">
                Run one or more algorithms on the same grid to compare visited nodes, path length, and computation time.
              </p>
            ) : (
              <div className="mt-4 overflow-hidden rounded-2xl border border-slate-800">
                <table className="w-full text-left text-sm">
                  <thead className="bg-slate-950 text-xs uppercase tracking-[0.12em] text-slate-400">
                    <tr>
                      <th className="px-3 py-2">Algorithm</th>
                      <th className="px-3 py-2 text-right">Visited</th>
                      <th className="px-3 py-2 text-right">Path</th>
                      <th className="px-3 py-2 text-right">ms</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800">
                    {history.map((record) => (
                      <tr key={record.id} className="text-slate-300">
                        <td className="px-3 py-2 font-semibold text-white">
                          {ALGORITHM_LABELS[record.algorithm].replace("Breadth-First Search (BFS)", "BFS")}
                          {!record.found ? <span className="ml-1 text-rose-500">No path</span> : null}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">{record.nodesVisited}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{record.pathLength}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{record.executionTime.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <p className="mt-3 text-xs leading-5 text-slate-400">
              Execution time measures algorithm computation only. Animation speed is controlled separately so comparisons stay fair.
            </p>
          </section>
        </aside>
      </main>
    </div>
  );
}
