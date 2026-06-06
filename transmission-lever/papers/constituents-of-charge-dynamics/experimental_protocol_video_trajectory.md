# Experimental Protocol: Video Sampling via Trajectory-History
## Validating Charge Dynamics Framework in Human Consciousness

---

## EXECUTIVE SUMMARY

This experiment operationalizes the theoretical framework of charge circulation and consciousness by having participants sample a video non-sequentially. By controlling their sampling trajectory (via scribbling speed), environment (sentiment modulation), and introducing topological deformations (video morphing), we test five core theorems:

1. **Incompleteness Principle**: Consciousness from radically incomplete information
2. **Sufficiency Convergence**: Multiple trajectories → identical narrative understanding
3. **Sentiment Modulation**: Emotional state changes sampling speed → different trajectories → different awareness
4. **Trajectory-History Identity**: Memory is path geometry, not content
5. **Topological Coherence**: Consciousness requires narrative trajectory coherence

---

## PHASE 1: BASELINE VIDEO SAMPLING (Incompleteness + Sufficiency)

### Setup
- **Participant**: Seat at screen with scribble controller (e.g., mouse wheel, stylus pressure, or gamepad thumb stick)
- **Stimulus**: 5-minute narrative video (e.g., short film with clear plot progression)
- **Task**: "Scribble forward/backward through the video. Scribble speed should match how much sense the video is making to you."
  - Smooth narrative (clear causality) → scribble forward faster
  - Confusing moment → scribble backward to re-examine
  - Satisfactory understanding → advance forward at normal pace
- **Duration**: 20 minutes of continuous sampling

### Measurements

**Trajectory Data** (sampled every 500ms):
- Time position in video (0.0 - 300.0 s)
- Scribble direction (forward/backward)
- Scribble speed (samples per second)
- Acceleration (d²position/dt²)

**Content Completeness**:
- Total frames viewed: N out of M total frames
- Completeness ratio: N/M (expected ≈ 0.1-0.3, matching perception completeness 1%)
- Unique timepoint count: how many non-repeated samples

**Consciousness Moments** (trajectory-geometry extrema):
- Peak acceleration points: moment of maximum "sense-making"
- Direction reversals: moment of "that doesn't make sense"
- Plateaus: moment of "I understand this segment"

### Predictions (Theory)

**Theorem 1 (Incompleteness)**:
- Participants sample 10-30% of video (like perception getting 1% of light)
- Yet all participants converge on same narrative summary
- **Metric**: Narrative summary similarity (cosine distance on semantic embedding) > 0.85 despite sampling overlap < 0.3

**Theorem 2 (Sufficiency)**:
- Different participants take completely different sampling paths
- All reach identical action-cell (narrative understanding)
- **Metric**: Path variance is high (Hausdorff distance between trajectories > 50 timepoints), but summary variance is low (semantic cosine distance < 0.15)

**Theorem 3 (Trajectory-History)**:
- Participants remember "how they got there" (which timepoints they visited)
- NOT "what they saw" (the actual video content)
- **Test**: Show them a sequence of timepoints from THEIR trajectory (without video playing). They recognize it as "oh, that's the scene I re-watched three times." Show them the SAME timepoints from a different participant's trajectory. They don't recognize it.

---

## PHASE 2: SENTIMENT MODULATION (Emotional Field)

### Setup (Repeated 3 times, different environments)

**Condition A: Relaxed**
- Comfortable chair, warm lighting
- Soft background music (60 bpm)
- Instruction: "Take your time, enjoy exploring"

**Condition B: Time-Pressured (Stressful)**
- Straight-backed chair, bright lighting
- Countdown timer visible: "You have 20 minutes"
- Instruction: "Work as efficiently as possible"

**Condition C: Cognitively Load (Stressful)**
- Normal chair, normal lighting
- Simultaneous secondary task (counting backward from 1000 by 7s)
- Instruction: "Sample the video while completing the counting task"

Same participant, same video, 1 week between sessions (counterbalanced order).

### Predictions (Theory)

**Theorem 4 (Sentiment Modulation)**:
- Scribble speed differs across conditions: Relaxed > Time-Pressured > Cognitively-Loaded
- **Expected mean speeds**: 
  - Relaxed: 8 samples/s (thorough exploration)
  - Time-Pressured: 15 samples/s (fast scanning)
  - Cognitive Load: 5 samples/s (conservative, attention divided)
- **Metric**: Mean scribble speed significantly different (ANOVA, F > 5, p < 0.05)

**Theorem 5 (Different Trajectories, Same Awareness)**:
- Despite different sampling speeds (emotion field frequencies), participants report identical narrative summaries
- Same cup → different thoughts → same awareness
- **Metric**: Narrative summaries identical across conditions (> 0.9 cosine similarity), despite trajectory variance > 20 samples

---

## PHASE 3: VIDEO MORPHING (Topological Coherence Test)

### Setup

- Same participant as Phases 1-2
- **Original Video A**: 5-minute narrative (participants already sampled)
- **Morphing Process**: Slowly replace scenes from Video A with scenes from **Video B** (different narrative)
  - Clips: 0% A / 100% B → gradually to 100% A / 0% B
  - Introduce 5 morph points: 0%, 25%, 50%, 75%, 100%
- **Same task**: Scribble through the morphing video naturally
- **Participant unaware**: They think it's the same video gradually "changing" due to technical issues or director's cut

### Morphing Strategy

**Scenario A**: Video A = "Person makes coffee" | Video B = "Person makes tea"
- Same actor, same kitchen, same timeline
- Only the object and final action differ
- Minimal topological disruption initially (causality still makes sense: "person goes to kitchen, prepares beverage")

**Scenario B**: Video A = "Murder mystery resolution" | Video B = "Comedy at a restaurant"
- Completely different narrative
- Maximum topological disruption
- Cannot maintain coherent trajectory explanation

### Measurements

**Morphing Trajectory**:
- Scribble position when participant first detects something "off"
- Scribble pattern change (increased reversals, slower speed)
- Explicit statement: "Something changed"

**Consciousness Collapse Point**:
- At what morphing percentage does narrative coherence fail?
- **Prediction**: Collapse occurs when trajectory becomes topologically incoherent
- **Metric**: Participant realizes deception at morphing ~50-70%, not at 1-10%

### Predictions (Theory)

**Theorem 6 (Topological Closure = Consciousness Necessity)**:

If **trajectory-history is memory** (not content), then:
- Participants maintain narrative understanding even as video changes
- BUT at the point where their *sampling trajectory becomes incoherent with any single narrative*, consciousness of deception emerges
- **Expected behavior**:
  - 0-30% morph: Participants notice no change (trajectory still makes sense with either video)
  - 30-70% morph: Participants report confusion/dissonance ("Something weird happened, but I can't place it")
  - 70%+ morph: Participants realize "This is a completely different video"

**Key Insight**: The collapse is NOT linear with morphing percentage. It depends on whether the participant's specific trajectory remains coherent.

- **Example**: If participant sampled timepoints {5s, 15s, 25s, 35s, ...}, these points might remain coherent even as video morphs 50%.
- But if they sampled {10s, 50s, 200s, ...} (covering different narrative arcs), 20% morph might already break coherence.

---

## PHASE 4: MEMORY VALIDATION (Trajectory vs. Content)

### Setup

After completing the morphing phase, show participants:
1. **Their own timepoint sequence** (without video): Just a list of timestamps {5s, 17s, 28s, 45s, ...}
2. **Random timepoints from Video A**
3. **Random timepoints from Video B**
4. **Another participant's trajectory** (their actual sampled timepoints)

Ask: "Which of these timelines feel like 'yours'?"

### Predictions

**If memory is trajectory-history (theory prediction)**:
- Participants identify their own timepoints with high confidence (>90%)
- They do NOT reliably identify Video A timepoints vs. Video B timepoints
- They are uncertain about another participant's trajectory (it's a different path)

**If memory is content (alternative hypothesis)**:
- Participants identify Video A timepoints if they saw Video A
- They confuse their own path with Video A's timeline
- Memory is "what I saw," not "how I got there"

---

## PHASE 5: MULTI-PARTICIPANT COMPARISON (Representation Invariance)

### Setup

- Recruit 20 participants (10 in relaxed condition, 10 in stressful condition)
- All sample the **same video A** 
- All provide narrative summaries

### Measurements

**Trajectory Diversity**:
- Calculate pairwise Hausdorff distances between all 20 trajectories
- Expected: Very high diversity (mean distance > 50 timepoints)

**Summary Similarity**:
- Calculate cosine similarity between all 20 narrative summaries
- Expected: Very high similarity (mean cosine > 0.85)

**Prediction (Representation Invariance Theorem)**:
- Despite radically different sampling paths (different "encodings" of the video)
- All participants achieve identical narrative understanding (same "action-cell")
- **Metric**: Trajectory variance / Summary variance > 10:1 ratio

This proves that consciousness is **substrate-independent**: the specific sampling path doesn't matter; only convergence to sufficient understanding matters.

---

## DATA ANALYSIS PROTOCOL

### Primary Metrics

**Incompleteness Ratio**:
```
I = (Unique timepoints viewed) / (Total frames in video)
Expected: 0.10 < I < 0.35
Theorem validation: I << 1 (radical incompleteness) yet narrative convergence
```

**Sufficiency Convergence Distance**:
```
D_narrative = cosine_distance(summary_participant_A, summary_participant_B)
Expected: D < 0.15 (narratives nearly identical)
Despite: D_trajectory = Hausdorff_distance(trajectory_A, trajectory_B) > 50 samples
```

**Sentiment Modulation Effect**:
```
Speed_relaxed vs. Speed_stressed: t-test, expected t > 3, p < 0.01
Direction: Stressed participants scribble faster (higher frequency emotion field)
```

**Morphing Collapse Point**:
```
For each participant i, define:
  M_i = morphing percentage at which participant i reports "something changed"
  
Distribution of M_1, M_2, ..., M_n expected: mode ≈ 50-70%, not uniform
Validates: Collapse depends on trajectory coherence, not content similarity
```

**Trajectory-History Memory Validation**:
```
Recognition accuracy:
  Own trajectory: ~90% (high)
  Random Video A timepoints: ~50% (chance)
  Own Video A timepoints: ~55% (slightly above chance, not high)
  Another participant's trajectory: ~45% (chance)
  
This pattern proves memory is trajectory-geometry, not content.
```

---

## EXPECTED OUTCOMES

### If Theory is Correct (Charge Dynamics / Consciousness Framework)

1. **Participants sample 10-30% of video** (incompleteness validated)
2. **All converge on identical narrative** despite diverse sampling paths (sufficiency validated)
3. **Emotional state changes sampling speed** but not final understanding (sentiment modulation validated)
4. **Memory is recognized trajectory, not content** (trajectory-history identity validated)
5. **Consciousness of deception emerges at topological incoherence point**, not at content divergence point (topological closure necessity validated)
6. **Different participants with completely different paths understand the same narrative** (representation invariance validated)

### If Theory is Incorrect

- Participants would converge on diverse narratives (sufficiency false)
- Memory would be strongly correlated with content sampled (trajectory-history false)
- Consciousness of morphing would emerge linearly with morphing percentage (topological closure irrelevant)
- All six theorems would fail independently

---

## SAMPLE SIZE & STATISTICAL POWER

- **Primary analysis**: n = 20 participants (10 relaxed, 10 stressed)
- **Effect sizes expected**:
  - Sentiment effect on scribble speed: Cohen's d ≈ 1.5 (large)
  - Narrative similarity despite different paths: r ≈ 0.9 (very large)
  - Morphing collapse point variance: σ ≈ 10% (tight distribution despite diverse participants)
- **Statistical power**: > 0.95 for detecting predicted effects at α = 0.05

---

## NEUROSCIENCE EXTENSION (Optional)

If fMRI funding available:

- **Scan while participants sample video** (Phase 1)
- **Measure neural activity during topological-closure-violation** (Phase 3)
- **Prediction**: 
  - Posterior parietal cortex (memory integration) activates when trajectory is coherent
  - Prefrontal cortex (error detection) activates when trajectory becomes topologically incoherent
  - No activation difference based on content change alone (unless content breaks trajectory)

---

## REFERENCES TO THEORETICAL FRAMEWORK

- **Theorem 1**: Incompleteness Principle (Section 6, Incompleteness-Principle)
- **Theorem 2**: Sufficiency via Unconstrained Subtasks (Section 2, Sufficiency-Principle)
- **Theorem 3**: Trajectory-History as Reference, Not Storage (Section 7, Trajectory-History)
- **Theorem 4**: Sentiment as Charge Field (Section 5, Sentiment-Modulation)
- **Theorem 5**: Three-Curve Intersection Model (Section 1, Three-Curve-Intersection)
- **Theorem 6**: Topological Closure Necessity (Section 3, Closure-Requirement)

Each phase tests multiple theorems simultaneously, providing comprehensive validation of the unified framework.

---

## PUBLICATION PLAN

**Manuscript 1**: "Consciousness via Incomplete Sampling: Video Trajectory as Model of Awareness"
- Phases 1-2 data
- Validates Incompleteness, Sufficiency, Sentiment Modulation
- Target: *Consciousness and Cognition* or *Frontiers in Psychology*

**Manuscript 2**: "Topological Coherence and Consciousness: The Video-Morphing Experiment"
- Phase 3 data
- Validates Topological Closure Necessity
- Target: *Nature Human Behaviour* or *PLoS Biology*

**Manuscript 3**: "Memory as Path, Not Content: Experimental Evidence for Trajectory-History Identity"
- Phases 4-5 data
- Validates Trajectory-History and Representation Invariance
- Target: *eLife* or *Science Advances*

---

## TIMELINE

- **Weeks 1-2**: Pilot with 3 participants, refine protocol
- **Weeks 3-8**: Phase 1-2 (baseline + sentiment modulation), n=20
- **Weeks 9-12**: Phase 3 (video morphing), same n=20
- **Weeks 13-14**: Phase 4-5 (memory validation + comparison), same n=20
- **Weeks 15-20**: Data analysis, manuscript preparation
- **Weeks 21-24**: Replication study (if time/funding allow)

Total: 6 months for first complete iteration.

---

## BUDGET

- Participants (20 × 3 sessions × $20): $1,200
- Video production (2 videos, 5 mins each): $5,000
- Eye-tracking equipment (optional but valuable): $8,000
- Software (Psychopy, Matlab, Python): $0 (open source)
- Data analysis (statistical consulting): $2,000
- **Total**: ~$16,000 (excluding salaries)

---

## ETHICAL CONSIDERATIONS

- **Deception in Phase 3**: Participants are not told that video will morph
  - **Mitigation**: Full debriefing after completion. Deception is necessary to measure genuine consciousness of incoherence.
  - **IRB approval**: Required. Justification: Understanding consciousness of deception requires brief, harmless deception with full debriefing.
  
- **No harm**: Video-scribbling is benign task. No physical, psychological, or cognitive harm.
  
- **Informed consent**: Participants informed they're in consciousness study, but not told about morphing specifically.

---

## NOVEL CONTRIBUTIONS

This experiment simultaneously validates 6 major theorems from a unified framework, something no prior consciousness experiment has done. It operationalizes abstract concepts (trajectory-history, sufficiency, topological closure) into measurable behavioral phenomena. It provides empirical evidence that consciousness depends on topological coherence rather than content completeness—a paradigm shift in consciousness studies.
