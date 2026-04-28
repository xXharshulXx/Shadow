export type AlgorithmId = "bfs" | "dijkstra" | "astar";

export type Position = {
  row: number;
  col: number;
};

export type SearchStep = {
  position: Position;
  description: string;
};

export type SearchResult = {
  algorithm: AlgorithmId;
  visitedOrder: SearchStep[];
  path: Position[];
  found: boolean;
  executionTime: number;
};

export type SearchContext = {
  rows: number;
  cols: number;
  start: Position;
  end: Position;
  walls: Set<string>;
};

export const GRID_ROWS = 20;
export const GRID_COLS = 40;

export const DEFAULT_START: Position = { row: 10, col: 6 };
export const DEFAULT_END: Position = { row: 10, col: 33 };

export const ALGORITHM_LABELS: Record<AlgorithmId, string> = {
  bfs: "Breadth-First Search (BFS)",
  dijkstra: "Dijkstra's Algorithm",
  astar: "A* Search",
};

export function positionKey(position: Position) {
  return `${position.row}-${position.col}`;
}

export function keyFromCoords(row: number, col: number) {
  return `${row}-${col}`;
}

export function keyToPosition(key: string): Position {
  const [row, col] = key.split("-").map(Number);
  return { row, col };
}

export function positionsEqual(a: Position, b: Position) {
  return a.row === b.row && a.col === b.col;
}

const DIRECTIONS: Position[] = [
  { row: -1, col: 0 },
  { row: 0, col: 1 },
  { row: 1, col: 0 },
  { row: 0, col: -1 },
];

function isInsideGrid(position: Position, rows: number, cols: number) {
  return position.row >= 0 && position.row < rows && position.col >= 0 && position.col < cols;
}

function getWalkableNeighbors(position: Position, context: SearchContext) {
  return DIRECTIONS.map((direction) => ({
    row: position.row + direction.row,
    col: position.col + direction.col,
  })).filter((neighbor) => {
    return isInsideGrid(neighbor, context.rows, context.cols) && !context.walls.has(positionKey(neighbor));
  });
}

function manhattanDistance(a: Position, b: Position) {
  return Math.abs(a.row - b.row) + Math.abs(a.col - b.col);
}

function describePosition(position: Position) {
  return `(${position.row}, ${position.col})`;
}

function reconstructPath(previous: Map<string, string>, start: Position, end: Position) {
  const startKey = positionKey(start);
  const endKey = positionKey(end);

  if (startKey === endKey) {
    return [start];
  }

  if (!previous.has(endKey)) {
    return [];
  }

  const path: Position[] = [];
  let currentKey = endKey;

  while (currentKey !== startKey) {
    path.unshift(keyToPosition(currentKey));
    const nextKey = previous.get(currentKey);

    if (!nextKey) {
      return [];
    }

    currentKey = nextKey;
  }

  path.unshift(start);
  return path;
}

type PriorityQueueItem<T> = {
  item: T;
  priority: number;
  order: number;
};

class PriorityQueue<T> {
  private heap: PriorityQueueItem<T>[] = [];

  private order = 0;

  enqueue(item: T, priority: number) {
    this.heap.push({ item, priority, order: this.order++ });
    this.bubbleUp(this.heap.length - 1);
  }

  dequeue() {
    if (this.heap.length === 0) {
      return undefined;
    }

    const first = this.heap[0];
    const last = this.heap.pop();

    if (last && this.heap.length > 0) {
      this.heap[0] = last;
      this.sinkDown(0);
    }

    return first.item;
  }

  get isEmpty() {
    return this.heap.length === 0;
  }

  private isHigherPriority(a: PriorityQueueItem<T>, b: PriorityQueueItem<T>) {
    return a.priority < b.priority || (a.priority === b.priority && a.order < b.order);
  }

  private bubbleUp(index: number) {
    let currentIndex = index;

    while (currentIndex > 0) {
      const parentIndex = Math.floor((currentIndex - 1) / 2);

      if (!this.isHigherPriority(this.heap[currentIndex], this.heap[parentIndex])) {
        break;
      }

      [this.heap[currentIndex], this.heap[parentIndex]] = [this.heap[parentIndex], this.heap[currentIndex]];
      currentIndex = parentIndex;
    }
  }

  private sinkDown(index: number) {
    let currentIndex = index;

    while (true) {
      const leftChildIndex = currentIndex * 2 + 1;
      const rightChildIndex = currentIndex * 2 + 2;
      let smallestIndex = currentIndex;

      if (
        leftChildIndex < this.heap.length &&
        this.isHigherPriority(this.heap[leftChildIndex], this.heap[smallestIndex])
      ) {
        smallestIndex = leftChildIndex;
      }

      if (
        rightChildIndex < this.heap.length &&
        this.isHigherPriority(this.heap[rightChildIndex], this.heap[smallestIndex])
      ) {
        smallestIndex = rightChildIndex;
      }

      if (smallestIndex === currentIndex) {
        break;
      }

      [this.heap[currentIndex], this.heap[smallestIndex]] = [this.heap[smallestIndex], this.heap[currentIndex]];
      currentIndex = smallestIndex;
    }
  }
}

function runBfs(context: SearchContext): Omit<SearchResult, "algorithm" | "executionTime"> {
  const queue: Position[] = [context.start];
  const discovered = new Set<string>([positionKey(context.start)]);
  const previous = new Map<string, string>();
  const visitedOrder: SearchStep[] = [];
  let found = false;

  while (queue.length > 0) {
    const current = queue.shift()!;
    const currentKey = positionKey(current);

    visitedOrder.push({
      position: current,
      description: `BFS is exploring node ${describePosition(current)} and adding undiscovered neighbors to the queue.`,
    });

    if (positionsEqual(current, context.end)) {
      found = true;
      break;
    }

    for (const neighbor of getWalkableNeighbors(current, context)) {
      const neighborKey = positionKey(neighbor);

      if (discovered.has(neighborKey)) {
        continue;
      }

      discovered.add(neighborKey);
      previous.set(neighborKey, currentKey);
      queue.push(neighbor);
    }
  }

  return {
    visitedOrder,
    path: found ? reconstructPath(previous, context.start, context.end) : [],
    found,
  };
}

function runDijkstra(context: SearchContext): Omit<SearchResult, "algorithm" | "executionTime"> {
  const frontier = new PriorityQueue<Position>();
  const distances = new Map<string, number>([[positionKey(context.start), 0]]);
  const previous = new Map<string, string>();
  const settled = new Set<string>();
  const visitedOrder: SearchStep[] = [];
  let found = false;

  frontier.enqueue(context.start, 0);

  while (!frontier.isEmpty) {
    const current = frontier.dequeue()!;
    const currentKey = positionKey(current);

    if (settled.has(currentKey)) {
      continue;
    }

    settled.add(currentKey);
    visitedOrder.push({
      position: current,
      description: `Dijkstra selected ${describePosition(current)} because it currently has the lowest known distance.`,
    });

    if (positionsEqual(current, context.end)) {
      found = true;
      break;
    }

    const currentDistance = distances.get(currentKey) ?? Number.POSITIVE_INFINITY;

    for (const neighbor of getWalkableNeighbors(current, context)) {
      const neighborKey = positionKey(neighbor);

      if (settled.has(neighborKey)) {
        continue;
      }

      const candidateDistance = currentDistance + 1;

      if (candidateDistance < (distances.get(neighborKey) ?? Number.POSITIVE_INFINITY)) {
        distances.set(neighborKey, candidateDistance);
        previous.set(neighborKey, currentKey);
        frontier.enqueue(neighbor, candidateDistance);
      }
    }
  }

  return {
    visitedOrder,
    path: found ? reconstructPath(previous, context.start, context.end) : [],
    found,
  };
}

function runAStar(context: SearchContext): Omit<SearchResult, "algorithm" | "executionTime"> {
  const frontier = new PriorityQueue<Position>();
  const gScores = new Map<string, number>([[positionKey(context.start), 0]]);
  const previous = new Map<string, string>();
  const closed = new Set<string>();
  const visitedOrder: SearchStep[] = [];
  let found = false;

  frontier.enqueue(context.start, manhattanDistance(context.start, context.end));

  while (!frontier.isEmpty) {
    const current = frontier.dequeue()!;
    const currentKey = positionKey(current);

    if (closed.has(currentKey)) {
      continue;
    }

    closed.add(currentKey);
    visitedOrder.push({
      position: current,
      description: `A* is exploring ${describePosition(current)} by combining cost so far with Manhattan distance to the goal.`,
    });

    if (positionsEqual(current, context.end)) {
      found = true;
      break;
    }

    const currentScore = gScores.get(currentKey) ?? Number.POSITIVE_INFINITY;

    for (const neighbor of getWalkableNeighbors(current, context)) {
      const neighborKey = positionKey(neighbor);

      if (closed.has(neighborKey)) {
        continue;
      }

      const candidateScore = currentScore + 1;

      if (candidateScore < (gScores.get(neighborKey) ?? Number.POSITIVE_INFINITY)) {
        const heuristic = manhattanDistance(neighbor, context.end);

        gScores.set(neighborKey, candidateScore);
        previous.set(neighborKey, currentKey);
        frontier.enqueue(neighbor, candidateScore + heuristic + heuristic * 0.001);
      }
    }
  }

  return {
    visitedOrder,
    path: found ? reconstructPath(previous, context.start, context.end) : [],
    found,
  };
}

export function runPathfindingAlgorithm(algorithm: AlgorithmId, context: SearchContext): SearchResult {
  const startedAt = performance.now();
  const result = algorithm === "bfs" ? runBfs(context) : algorithm === "dijkstra" ? runDijkstra(context) : runAStar(context);
  const executionTime = performance.now() - startedAt;

  return {
    algorithm,
    executionTime,
    ...result,
  };
}



export function generateGuaranteedSolvableMaze(rows: number, cols: number, start: Position, end: Position): Set<string> {
  const walls = new Set<string>();

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      walls.add(keyFromCoords(r, c));
    }
  }

  const visited = new Set<string>();
  const frontier: Position[] = [];

  const startOdd = {
    row: Math.min(rows - 1, Math.max(1, Math.floor(start.row / 2) * 2 + 1)),
    col: Math.min(cols - 1, Math.max(1, Math.floor(start.col / 2) * 2 + 1)),
  };

  const addFrontier = (pos: Position) => {
    const neighbors = [
      { row: pos.row - 2, col: pos.col },
      { row: pos.row + 2, col: pos.col },
      { row: pos.row, col: pos.col - 2 },
      { row: pos.row, col: pos.col + 2 },
    ];
    for (const n of neighbors) {
      if (n.row > 0 && n.row < rows && n.col > 0 && n.col < cols) {
        const key = positionKey(n);
        if (!visited.has(key) && !frontier.some(f => f.row === n.row && f.col === n.col)) {
          frontier.push(n);
        }
      }
    }
  };

  visited.add(positionKey(startOdd));
  walls.delete(positionKey(startOdd));
  addFrontier(startOdd);

  while (frontier.length > 0) {
    const randomIndex = Math.floor(Math.random() * frontier.length);
    const current = frontier[randomIndex];
    frontier.splice(randomIndex, 1);

    const currentKey = positionKey(current);
    if (visited.has(currentKey)) continue;

    visited.add(currentKey);
    walls.delete(currentKey);

    const neighbors = [
      { row: current.row - 2, col: current.col },
      { row: current.row + 2, col: current.col },
      { row: current.row, col: current.col - 2 },
      { row: current.row, col: current.col + 2 },
    ];

    const visitedNeighbors = neighbors.filter(n =>
      n.row > 0 && n.row < rows && n.col > 0 && n.col < cols && visited.has(positionKey(n))
    );

    if (visitedNeighbors.length > 0) {
      const chosen = visitedNeighbors[Math.floor(Math.random() * visitedNeighbors.length)];
      const wallRow = (current.row + chosen.row) / 2;
      const wallCol = (current.col + chosen.col) / 2;
      walls.delete(keyFromCoords(wallRow, wallCol));
    }

    addFrontier(current);
  }

  const wallArray = Array.from(walls);
  for (let i = 0; i < wallArray.length * 0.15; i++) {
    const randomWall = wallArray[Math.floor(Math.random() * wallArray.length)];
    const [rStr, cStr] = randomWall.split("-");
    const r = Number(rStr);
    const c = Number(cStr);
    if (r > 0 && r < rows - 1 && c > 0 && c < cols - 1) {
      walls.delete(randomWall);
    }
  }

  const endOdd = {
    row: Math.min(rows - 1, Math.max(1, Math.floor(end.row / 2) * 2 + 1)),
    col: Math.min(cols - 1, Math.max(1, Math.floor(end.col / 2) * 2 + 1)),
  };

  let currR = start.row;
  let currC = start.col;
  walls.delete(keyFromCoords(currR, currC));
  while (currR !== startOdd.row) {
    currR += currR < startOdd.row ? 1 : -1;
    walls.delete(keyFromCoords(currR, currC));
  }
  while (currC !== startOdd.col) {
    currC += currC < startOdd.col ? 1 : -1;
    walls.delete(keyFromCoords(currR, currC));
  }

  currR = end.row;
  currC = end.col;
  walls.delete(keyFromCoords(currR, currC));
  while (currR !== endOdd.row) {
    currR += currR < endOdd.row ? 1 : -1;
    walls.delete(keyFromCoords(currR, currC));
  }
  while (currC !== endOdd.col) {
    currC += currC < endOdd.col ? 1 : -1;
    walls.delete(keyFromCoords(currR, currC));
  }

  return walls;
}

export type MultiStageSearchResult = {
  algorithm: AlgorithmId;
  visitedOrder: SearchStep[];
  path: Position[];
  found: boolean;
  executionTime: number;
};

export function runMultiStagePathfinding(
  algorithm: AlgorithmId,
  context: {
    rows: number;
    cols: number;
    start: Position;
    end: Position;
    checkpoints: Position[];
    walls: Set<string>;
  }
): MultiStageSearchResult {
  const startedAt = performance.now();
  const stages: { start: Position; end: Position }[] = [];

  let currentStart = context.start;
  for (const cp of context.checkpoints) {
    stages.push({ start: currentStart, end: cp });
    currentStart = cp;
  }
  stages.push({ start: currentStart, end: context.end });

  const visitedOrder: SearchStep[] = [];
  const fullPath: Position[] = [];
  let found = true;

  for (let i = 0; i < stages.length; i++) {
    const stage = stages[i];
    const searchContext: SearchContext = {
      rows: context.rows,
      cols: context.cols,
      start: stage.start,
      end: stage.end,
      walls: context.walls,
    };

    const stageResult =
      algorithm === "bfs"
        ? runBfs(searchContext)
        : algorithm === "dijkstra"
        ? runDijkstra(searchContext)
        : runAStar(searchContext);

    const mappedVisited = stageResult.visitedOrder.map((step) => ({
      ...step,
      description: `[Stage ${i + 1}/${stages.length}] ${step.description}`,
    }));
    visitedOrder.push(...mappedVisited);

    if (!stageResult.found) {
      found = false;
      break;
    }

    if (i === 0) {
      fullPath.push(...stageResult.path);
    } else {
      fullPath.push(...stageResult.path.slice(1));
    }
  }

  const executionTime = performance.now() - startedAt;

  return {
    algorithm,
    visitedOrder,
    path: found ? fullPath : [],
    found,
    executionTime,
  };
}