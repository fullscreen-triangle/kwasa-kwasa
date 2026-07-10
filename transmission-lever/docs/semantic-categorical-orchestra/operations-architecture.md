# Operations Architecture

How **kwasa-kwasa** (turbulance + the orchestrator), the **scheduler**, and **Buhera OS** fit together as one running system.

This document is the contract you implement against. Each component has one responsibility. The interfaces between them are typed, narrow, and stable. If you find yourself adding a third concern to a component, the cut is wrong — push it to the next layer.

---

## 0. The picture in one paragraph

A user writes a turbulance script. The script is the only input. The kwasa-kwasa orchestrator parses it into a typed task graph, statically computes each task's categorical complexity and sufficiency threshold, and hands the graph to the scheduler. The scheduler maintains a priority queue of live tasks and on each tick dispatches one or more *acts* — units of work — to the modules registered with Buhera OS. Buhera OS holds the modules, executes each act through the module's DSL interpreter, writes the result to the audit log, and updates the kernel's substrate. The orchestrator watches the audit log, evaluates necessity/holonomy/sufficiency on each tick's update, and decides whether to continue, decline, or terminate the task. When the task graph reaches quiescence (or sufficiency for all live tasks), the orchestrator surfaces the result to the user; the surface clears.

Three components, three responsibilities, one running system.

---

## 1. Responsibilities

| Component | Owns | Does Not Own |
|---|---|---|
| **kwasa-kwasa** | Turbulance grammar, parsing, task-graph construction, necessity gate (contribution, holonomy, sufficiency, order parameter), surface rendering | Module internals, hardware allocation, audit log storage |
| **Scheduler** | Priority queue, dispatch loop, per-tick budget, act-budget allocation, decline detection | What a "good" priority means semantically, what an act does, module state |
| **Buhera OS** | Substrate (CMM/PSS/DIC/PVE/TEM), module registry, audit log, address space, embedder, vaHera as a built-in module | Turbulance, intent, sufficiency, scheduling decisions |

The cut is sharp. The orchestrator decides *what* should be done; the scheduler decides *in what order and how much*; the OS *does it and records what was done*.

---

## 2. Data types crossing the interfaces

Define these once; they appear in all three components.

### 2.1 The fundamental types

```
TaskId          : 128-bit identifier (UUIDv7 or trajectory-state hash)
ModuleId        : string (e.g. "vahera", "code-test", "geolocate")
ActId           : 128-bit identifier, monotone in audit log
TrajectoryState : the composition-inflation clock state at commit
```

### 2.2 Categorical complexity (from `cor:log_termination`)

```rust
struct CategoricalComplexity {
    k: u128,          // distinguishable trajectories required
    d: u32,           // operation types (default 4)
}

impl CategoricalComplexity {
    fn n_term(&self) -> u32 {
        // ceil(log_{d+1}(K/d)) + 1
        1 + ((self.k as f64 / self.d as f64).log(self.d as f64 + 1.0)).ceil() as u32
    }
}
```

### 2.3 The dispatch unit

```rust
struct Dispatch {
    task_id: TaskId,
    module_id: ModuleId,
    instruction: ModuleInstruction,    // module-specific bytes
    act_budget: u32,                   // thoroughness budget
}

struct ActResult {
    task_id: TaskId,
    act_id: ActId,
    trajectory_state: TrajectoryState, // from the clock
    output_delta: OutputDelta,         // changes to the task's output cell
    residue: f64,                      // cost deposited
    completed: bool,                   // module says "I'm done"
}
```

### 2.4 The task envelope

```rust
struct LiveTask {
    id: TaskId,
    module_id: ModuleId,
    complexity: CategoricalComplexity,
    sufficiency_threshold: SufficiencyThreshold,
    committed_acts: u32,                // n_live
    current_output: OutputCell,         // accumulates from output_deltas
    state: TaskState,                   // Pending | Running | Sufficient | Terminated | Declined
    parent: Option<TaskId>,             // for sub-tasks
    dependencies: Vec<TaskId>,          // arcs into this task in the task graph
}
```

### 2.5 The surface

```rust
struct Surface {
    intent: String,                     // the original turbulance script
    result: Option<RenderedOutput>,     // None if Declined
    completion_reason: CompletionReason, // Terminated | Sufficient | Declined { reason }
    audit_summary: AuditSummary,        // committed acts, total residue, wall-clock time
}
```

These types are the entire vocabulary. Every interface in the system takes or returns one of them.

---

## 3. kwasa-kwasa (the orchestrator)

Three sub-components inside the orchestrator: **parser**, **gate**, **task graph manager**.

### 3.1 Input

A turbulance script as a UTF-8 string. Nothing else. The user types it; the terminal hands it over verbatim.

### 3.2 What it does, step by step

```
ORCHESTRATE(script: String, modules: Modules) -> Result<Surface, Decline> {
    let ast = parse(script)?;                          // 3.3
    let intent = ast.extract_intent();                  // top-level goal cell
    let mut task_graph = ast.build_task_graph(intent);  // 3.4
    
    // Static analysis: complexity and sufficiency per task
    for task in task_graph.tasks_mut() {
        task.complexity = analyse_complexity(task);     // K, n_term
        task.sufficiency_threshold = analyse_sufficiency(task, intent);
    }
    
    // Route each task to a module
    for task in task_graph.tasks_mut() {
        task.module_id = route(task, modules)?;
    }
    
    let mut scheduler = Scheduler::new(task_graph);
    
    loop {
        let tick_results = scheduler.tick();             // 4
        for result in tick_results {
            audit_log.append(result);
            task_graph.update(result);
            
            // The necessity gate runs every tick.
            if let Some(decision) = gate.evaluate(&task_graph, &audit_log) {
                match decision {
                    GateDecision::PruneTask(id)   => scheduler.drop(id),
                    GateDecision::DeclineEnsemble => return Decline(...),
                    GateDecision::ContinueAll     => {}
                }
            }
        }
        
        if task_graph.is_quiescent() {
            return Ok(render_surface(&task_graph, &audit_log));
        }
    }
}
```

### 3.3 The parser

- Owns the turbulance grammar (separate document; the BNF is the parser's spec).
- Produces an AST of typed expressions: declarations, sub-task invocations, intent statements, sufficiency clauses.
- Performs static type checking; rejects malformed scripts before any dispatch.

The parser is **not** the orchestrator's `run()` entry point. The parser is one phase of `run()`. The orchestrator's entry point is the script string.

### 3.4 The task graph

A directed acyclic graph in the common case, may contain cycles when the script declares iterative refinement. Vertices are `LiveTask`s. Arcs are typed dependencies: `task B consumes task A's output`.

The graph is what the gate measures over (per `def:taskgraph` in the orchestra paper). Contribution, holonomy, and order parameter are all functions of the graph plus the audit log.

### 3.5 The gate

Runs every tick on `(task_graph, audit_log)`. Four checks, all functions of graph-side data:

```rust
fn evaluate(graph: &TaskGraph, log: &AuditLog) -> Option<GateDecision> {
    // 1. Contribution: any task with δS = 0?
    for task in graph.live_tasks() {
        if contribution(task, graph) == 0.0 {
            return Some(GateDecision::PruneTask(task.id));
        }
    }
    
    // 2. Holonomy: any cycle drifted from spec?
    for cycle in graph.cycles() {
        if holonomy(cycle, log) > FLOOR {
            return Some(GateDecision::DeclineEnsemble(
                Decline::CycleInconsistency(cycle.id)
            ));
        }
    }
    
    // 3. Sufficiency: any task already in its intent cell?
    for task in graph.live_tasks() {
        if semantic_distance(&task.current_output, &task.intent_cell()) <= FLOOR {
            graph.mark_sufficient(task.id);
        }
    }
    
    // 4. Order parameter: ensemble coherent enough to surface?
    let r = order_parameter(graph);
    graph.set_regime(r);
    
    None
}
```

The gate **never** re-checks correctness. Correctness is the module's contract (per `thm:gate-orthogonal`).

### 3.6 Output

A `Surface`. The orchestrator's entire job is to convert `script` → `Surface`.

---

## 4. The Scheduler

One responsibility: pick the next dispatch, allocate budget to it, hand it to the OS, receive the result.

### 4.1 Input

- A reference to the task graph (read-mostly; the scheduler updates only `committed_acts` and `current_output` on each task).
- A reference to the module registry (for dispatch).
- A reference to the audit log (write-only from the scheduler's perspective).

### 4.2 The priority function

```rust
fn priority(task: &LiveTask) -> f64 {
    if task.state != TaskState::Running {
        return 0.0;
    }
    
    // Remaining acts to whichever termination point comes first.
    let to_termination = task.complexity.n_term()
        .saturating_sub(task.committed_acts);
    let to_sufficiency = task.sufficiency_threshold
        .estimated_remaining(task.committed_acts)
        .unwrap_or(to_termination);
    let remaining = to_termination.min(to_sufficiency);
    
    if remaining == 0 {
        return f64::INFINITY;  // task at the threshold, dispatch immediately
    }
    
    1.0 / (remaining as f64)
}
```

The lower the remaining-acts count, the higher the priority. Sufficiency overrides termination whenever it's reached first.

### 4.3 The tick loop

```rust
fn tick(&mut self) -> Vec<ActResult> {
    let mut tick_budget = self.tick_budget;
    let mut results = Vec::new();
    
    while tick_budget > 0 {
        let task = match self.queue.peek_mut() {
            Some(t) => t,
            None => break,
        };
        
        // Share of tick budget proportional to priority.
        let share_fraction = task.priority() / self.total_priority();
        let act_budget = (tick_budget as f64 * share_fraction).ceil() as u32;
        let act_budget = act_budget.min(tick_budget);
        
        let dispatch = Dispatch {
            task_id: task.id,
            module_id: task.module_id,
            instruction: task.next_instruction(),
            act_budget,
        };
        
        let result = self.os.dispatch(dispatch);  // blocks for one act
        task.commit(&result);
        results.push(result);
        
        tick_budget -= act_budget;
        
        if task.is_done() {
            self.queue.drop(task.id);
        }
    }
    
    results
}
```

### 4.4 What the scheduler controls

Per the previous discussion:

1. **Which task** gets the next dispatch (priority queue selection).
2. **How much of this tick's budget** goes to that task (the `share` calculation).
3. **How thoroughly each act is executed** (the `act_budget` passed to the module).

The scheduler does **not** control: CPU affinity, RAM allocation, disk I/O priority. Those are the OS kernel's (Linux's, on the Chromebook). The scheduler controls *turn allocation* among Buhera tasks.

### 4.5 The decline conditions

The scheduler tells the orchestrator to decline a task when:

- `committed_acts > n_term(K)` and the task hasn't completed → the script's declared complexity was wrong.
- A module returns an error → modules don't decline; they signal a structural failure and the scheduler bubbles up.
- The task graph as a whole stops advancing (no task's `committed_acts` increased for some configurable number of ticks) → deadlock or non-termination.

The scheduler does **not** detect sufficiency. That's the gate's job (see 3.5). The scheduler reads `task.state == Sufficient` and drops the task; it doesn't compute the sufficiency check itself.

### 4.6 Output

A stream of `ActResult`s, one per dispatched act, appended to the audit log. Nothing else leaves the scheduler.

---

## 5. Buhera OS

The substrate. Holds modules, executes acts, maintains the audit log and address space.

### 5.1 Module registry

```rust
struct ModuleRegistry {
    modules: HashMap<ModuleId, Box<dyn Module>>,
}

trait Module {
    fn id(&self) -> ModuleId;
    fn dsl_grammar(&self) -> &'static Grammar;
    fn execute(&mut self, instruction: &ModuleInstruction, act_budget: u32)
        -> ActResult;
    fn output_cell(&self, instruction: &ModuleInstruction)
        -> OutputCell;   // for sufficiency-check
}
```

Each module:
- Owns its DSL grammar (the parser's reference for that module's instructions).
- Terminates its domain (per `def:module`): every well-formed instruction is correct.
- Receives an `act_budget` and uses it to decide internally how thoroughly to execute.
- Returns one `ActResult` per call.

vaHera is a registered module. The 11+ other DSLs are registered modules. The orchestrator routes a task to a module by `ModuleId`.

### 5.2 The audit log

Append-only, monotone in `act_id`. Each entry:

```rust
struct AuditEntry {
    act_id: ActId,                    // monotone
    trajectory_state: TrajectoryState, // composition-inflation clock state
    task_id: TaskId,
    module_id: ModuleId,
    instruction: ModuleInstruction,
    residue: f64,
    output_delta: OutputDelta,
    wall_clock: SystemTime,
}
```

The audit log is the kernel's record. It is the single source of truth for "what has the system done". The orchestrator reads it for gate evaluation; the scheduler appends to it on each dispatch; the kernel persists it.

### 5.3 The substrate

The existing kernel: CMM (storage), PSS (process scheduling — but at the OS layer, not the scheduler's layer), DIC (instruction cache), PVE (per-vertex evaluation), TEM (triple-equivalence monitor). These keep working as they do. The new layer (orchestrator + scheduler) sits on top.

### 5.4 The dispatch entry point

```rust
impl BuheraOS {
    fn dispatch(&mut self, dispatch: Dispatch) -> ActResult {
        let module = self.registry.get_mut(&dispatch.module_id)?;
        let mut result = module.execute(&dispatch.instruction, dispatch.act_budget);
        
        // Stamp the act_id and trajectory state from the kernel's clock.
        result.act_id = self.next_act_id();
        result.trajectory_state = self.clock.tick(result.op_type());
        
        // Append to audit log.
        self.audit_log.append(AuditEntry::from(&result, &dispatch));
        
        // Update substrate if the act produced storable content.
        if let Some(payload) = result.storable() {
            self.substrate.allocate(payload);
        }
        
        result
    }
}
```

This is the one entry point the scheduler calls. The OS does **not** know about turbulance, intent, or sufficiency. It only knows about dispatches, acts, and the audit log.

### 5.5 The composition-inflation clock

Lives inside Buhera OS. Ticks once per dispatched act. Produces a `TrajectoryState` that is monotone, structurally ordered, and exponentially distinguishable per `thm:composition_inflation`. The clock is the kernel's time; wall-clock is auxiliary.

The clock state is the act_id's high-resolution form. Two acts at the same wall-clock millisecond have distinct trajectory states.

### 5.6 What the OS does **not** do

- Parse turbulance.
- Compute task priority.
- Decide sufficiency.
- Render the surface.

The OS is the substrate. It receives dispatches and produces act results. It records them. It updates state. Nothing more.

---

## 6. The complete dispatch cycle

One full trip through the system for one act:

```
TICK BEGINS
   |
   v
[Scheduler] picks highest-priority task T from queue
   |
   v
[Scheduler] computes act_budget for this tick share
   |
   v
[Scheduler] constructs Dispatch { task_id, module_id, instruction, act_budget }
   |
   v
[Scheduler] calls OS.dispatch(dispatch)
   |
   v
[OS] looks up module by module_id
   |
   v
[Module] receives (instruction, act_budget), executes one act,
         returns ActResult { output_delta, residue, completed }
   |
   v
[OS] stamps act_id and trajectory_state, appends to audit_log,
     updates substrate, returns ActResult to scheduler
   |
   v
[Scheduler] updates T.committed_acts += 1, T.current_output += output_delta
   |
   v
[Scheduler] recomputes T.priority
   |
   v
[Orchestrator's gate] runs on (task_graph, audit_log):
   - contribution check on each task
   - holonomy check on each cycle
   - sufficiency check on each task (sets T.state = Sufficient if in cell)
   - order parameter for the ensemble
   |
   v
[Orchestrator] returns gate decisions to scheduler:
   - PruneTask(id): scheduler drops task
   - DeclineEnsemble: orchestrator returns Decline to user
   - ContinueAll: scheduler continues
   |
   v
[Scheduler] checks tick budget; if remaining, loop;
            if exhausted, tick ends
   |
   v
TICK ENDS
```

Each tick produces a batch of `ActResult`s. The orchestrator's gate runs once per tick on the batch. The scheduler runs its loop within the tick.

---

## 7. The complete program lifecycle

```
1. USER opens the terminal (blank surface).
2. USER types a turbulance script.
3. TERMINAL passes the script to kwasa-kwasa.run(script, modules).
4. ORCHESTRATOR parses, builds task graph, computes complexity and sufficiency
   thresholds for each task, routes tasks to modules.
5. ORCHESTRATOR creates a Scheduler with the task graph and starts it.
6. SCHEDULER ticks repeatedly:
   - dispatches acts to OS
   - OS executes acts, returns results
   - scheduler updates tasks, audit log grows
   - orchestrator's gate evaluates after each tick
7. The loop continues until one of:
   (a) Task graph reaches quiescence (all tasks Sufficient or Terminated).
   (b) Gate triggers DeclineEnsemble (holonomy violation).
   (c) Scheduler detects deadlock or n_term overrun (Decline up to orchestrator).
8. ORCHESTRATOR renders the Surface:
   - If quiescent: result = best output across tasks reaching intent_cell.
   - If declined: result = None, completion_reason = Declined { reason }.
9. TERMINAL displays the surface, then clears.
10. SYSTEM returns to blank surface, awaiting the next script.
```

The audit log persists between scripts (per the OS's persistence policy). The orchestrator and scheduler are spawned per-script and torn down after the surface clears. The OS and the modules are long-running.

---

## 8. Component boundaries: what each can and cannot see

This is the part that decides whether the architecture stays clean as you build.

### 8.1 What the orchestrator can see

- The user's script (verbatim).
- The task graph it built from the script.
- The audit log (read-only).
- The module registry (read-only, for routing).
- The output cells of each task (read-only, for sufficiency).

### 8.2 What the orchestrator cannot see

- Module internal state.
- The OS substrate (CMM cells directly, etc.).
- The scheduler's queue.
- Wall-clock time at sub-tick resolution (it sees trajectory states).

### 8.3 What the scheduler can see

- The task graph (read-write on `committed_acts`, `current_output`, `state`).
- The module registry (for dispatch routing).
- The current tick's budget.

### 8.4 What the scheduler cannot see

- The script.
- The intent or sufficiency semantics.
- The audit log contents (it appends to it; it does not query it).
- Module internals.

### 8.5 What the OS can see

- All registered modules.
- The audit log (read-write).
- The substrate (full access).
- The composition-inflation clock.

### 8.6 What the OS cannot see

- The orchestrator's task graph (it sees individual dispatches).
- The user's script.
- Sufficiency thresholds.
- Priority computations.

---

## 9. Implementation order

If you're building this from scratch, the dependency order is:

1. **Define the types in §2.** They're the vocabulary; everything else uses them.
2. **Build Buhera OS's dispatch entry point and audit log.** This is the substrate everything else sits on. The composition-inflation clock goes here.
3. **Register vaHera as the first module.** Use the existing vaHera implementation; wrap it in the `Module` trait.
4. **Build the scheduler.** Priority queue, tick loop, dispatch. No gate logic yet — just dispatch acts in priority order and report results.
5. **Build kwasa-kwasa.** Parser, task graph, gate. The gate is the last piece; before it, the orchestrator just routes tasks to the scheduler and lets them run.
6. **Wire the terminal.** The terminal becomes a thin client of the orchestrator.
7. **Register two more modules.** Pick the two you can wire fastest. Each module is a `Module` trait implementation.

Each step is testable in isolation. The dispatch entry point can be tested with a stub module. The scheduler can be tested with a stub OS. The orchestrator can be tested with a stub scheduler.

---

## 10. What this architecture buys you

Five things, each falling out of the cuts above without separate engineering:

1. **The OS doesn't need to change for new DSLs.** Register a new module; the system knows about it. No new orchestrator logic, no new scheduler logic.
2. **The scheduler doesn't need to know what tasks mean.** It only knows priorities and budgets. Same scheduler works for code testing, simulation, video editing.
3. **The orchestrator doesn't need to know how acts execute.** It only knows the audit log entries. Same gate works regardless of module internals.
4. **The audit log is the single source of truth.** Replaying the log reconstructs everything: the task graph, the priorities, the gate decisions, the surface. This is your debugging tool, your reproducibility guarantee, your provenance system.
5. **Sufficiency-driven termination is a runtime property, not a build-time decision.** The orchestrator's gate decides; the scheduler obeys. Adding new sufficiency heuristics later changes the gate, not the rest.

The architecture is small because each component is small. If any one of them grows, check whether you're putting work in the wrong place.

---

## 11. What is explicitly out of scope for this document

- The turbulance grammar (separate spec, written when you build the parser).
- The composition-inflation clock's internal algorithm (separate doc; the math is in `computational-systems-structure.tex`).
- How specific modules' DSLs work internally (each module's concern).
- The substrate's persistence model (Buhera OS internal; not orchestrator's concern).
- The webtool / blank surface UI (separate doc; this layer talks to the orchestrator and renders surfaces).

If you find yourself wanting to specify any of these here, push it to the right document. This file is the contract between the three top-level components and nothing else.
