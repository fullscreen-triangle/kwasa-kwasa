# Experimental Protocol: Memory as Trajectory-History, Not Content
## The Backward Video Rearrangement Test

---

## CORE HYPOTHESIS

**If memory is trajectory-history (theory), not content:**
- Participants watching video in **Relaxed** context build trajectory-history **T_relax** (geometry of sampling under low-frequency sentiment field)
- Participants watching same video in **Stressful** context build trajectory-history **T_stress** (geometry of sampling under high-frequency sentiment field)
- Both **T_relax** and **T_stress** are INCOHERENT in a **Neutral** emotional context (where they were built for different emotional landscapes)
- When asked to rearrange a backwards video in neutral context, participants cannot apply **T_relax** or **T_stress**
- They must construct a **completely new trajectory-history T_neutral** that emerges from scratch
- **Therefore: T_neutral should be different from T_relax and T_stress**

**If memory is content-based (alternative hypothesis):**
- After seeing the video 10 times relaxed + 10 times stressed, participants know the narrative
- When rearranging the backward video, they will reconstruct the original sequence from their learned content knowledge
- All three contexts would produce identical (or near-identical) rearrangements
- **Therefore: T_neutral should be similar to T_relax and T_stress**

---

## EXPERIMENT OVERVIEW

| Phase | Condition | Task | Repetitions | Measurement |
|-------|-----------|------|-------------|-------------|
| **1** | Relaxed | Watch video forward | 10× | Nothing (trajectory-building) |
| **2** | Stressful | Watch video forward | 10× | Nothing (trajectory-building) |
| **3** | Neutral | Rearrange backward video | 1× | **Output sequence** |

**Critical**: Same video throughout. Same participant. Only emotional context changes in Phase 3.

---

## DETAILED PROTOCOL

### PHASE 1: RELAXED CONTEXT (Baseline Trajectory-History Formation)

**Environment Setup**:
- Comfortable chair, warm lighting (2700K color temperature)
- Soft instrumental music playing (60 bpm, no lyrics)
- No time pressure
- Instruction: "Watch the video naturally. There's no time limit. Just relax and enjoy."

**Stimulus**: 
- 5-minute narrative video with clear story arc
- Example: "A barista makes a coffee for different customers"
- Each scene is visually and temporally distinct (clear "time slices")

**Procedure**:
- Watch full video once
- Take 5-minute break
- Watch again (repeat 10 times total, ~50 minutes per context)
- **Participant does nothing but watch**—no explicit memorization task
- Emotional state: Relaxed, low arousal, low-frequency thought oscillation

**What's happening theoretically**:
- Participant's consciousness experiences 10 converging moments
- Each moment: Perception (video frame) ∩ Thought (relaxed understanding) ∩ Memory (accumulated history)
- The trajectory-history **T_relax** encodes the *geometry of sampling* under this emotional field
- Not the content ("I learned the barista made 10 coffees"), but the *path* (specific timepoint convergences that made sense under relaxed sentiment)

---

### PHASE 2: STRESSFUL CONTEXT (Alternative Trajectory-History Formation)

**Environment Setup**:
- Hard-backed chair, bright lighting (5000K, clinical)
- Loud ambient noise (coffee shop background, ~70dB)
- Visible countdown timer: "You have 5 minutes for each watch"
- Instruction: "Watch efficiently. You need to extract the key information quickly."

**Stimulus**: 
- **SAME VIDEO** as Phase 1
- No changes to content

**Procedure**:
- Watch full video once under time pressure
- 2-minute break (pressured: "Hurry, next watch starting")
- Repeat 10 times (~25 minutes total, compressed)
- Participant experiences high arousal, high-frequency thought oscillation

**What's happening theoretically**:
- Under stressful emotional field, same video generates different sampling geometry
- High-frequency sentiment field forces different thought trajectories
- Trajectory-history **T_stress** captures the *path* under this emotional modulation
- Different from **T_relax** because the emotional landscape is different
- **Still about same video content**, but organized under different emotional frequency

---

### PHASE 3: NEUTRAL CONTEXT - THE CRITICAL TEST

**Environment Setup**:
- Neutral chair, neutral lighting (4000K, daylight)
- Silent room, no sound
- No time pressure
- Instruction: "The video will play backwards. You'll see individual scenes. Rearrange them into an order that makes sense to you."

**Stimulus**:
- **SAME VIDEO, BUT PLAYED BACKWARDS**
- Video is broken into 30 time slices (10-second segments)
- Slices presented in randomized order on screen
- Participant uses mouse/touchscreen to drag slices into new arrangement
- **Slices can be reused** (same time slice can appear multiple times in arrangement)
- Duration: One session, ~10-15 minutes to rearrange

**Critical Constraints**:
- Emotional context is **neutral** (not relaxed, not stressful)
- This means neither **T_relax** nor **T_stress** should be applicable
- They cannot rely on trajectory-histories built under different emotional fields
- They must construct a new arrangement
- **Reuse is allowed and expected**: Same timepoint (content) can be reached from multiple different histories (paths)

---

## THEORETICAL PREDICTIONS

### Prediction 1: Non-Transfer of Trajectory-History

**If memory is trajectory-history:**
- Relaxed trajectory **T_relax** is built in relaxed emotional field
- Stressful trajectory **T_stress** is built in stressful emotional field  
- Neutral emotional field is **incoherent with both T_relax and T_stress**
- Participant cannot apply prior trajectories to neutral context
- **Result**: Rearrangement in neutral context produces novel sequence **S_neutral**

**Quantitative Prediction**:
- Let **S_original** = original video sequence (ground truth)
- Let **S_relax** = sequence if participant just recalled relaxed context rearrangement
- Let **S_stress** = sequence if participant just recalled stressful context rearrangement
- Let **S_neutral** = actual rearrangement in neutral context

Then:
```
Similarity(S_neutral, S_original) ≈ 0.4-0.6 (moderate, chance is 0.33)
Similarity(S_neutral, S_relax) ≈ 0.2-0.4 (low-to-moderate, not converging)
Similarity(S_neutral, S_stress) ≈ 0.2-0.4 (low-to-moderate, not converging)
Similarity(S_relax, S_stress) ≈ 0.3-0.5 (low-to-moderate, different emotions = different paths)
```

**Why these values?**
- Participants don't have perfect memory of content, but have some residual recall
- Moderate similarity to original (0.4-0.6) reflects general narrative understanding
- Low similarity between neutral and prior contexts (0.2-0.4) reflects that trajectory-histories don't transfer
- Low similarity between relaxed and stressful (0.3-0.5) reflects that different emotional fields produce different trajectories

### Prediction 2: Content-Based Alternative (If Theory is Wrong)

**If memory is content-based:**
- Watching video 20 times (10 relaxed + 10 stressful) embeds the narrative content in memory
- Participant knows "the correct sequence" regardless of emotional context
- When rearranging, they apply this learned content
- **Result**: All three contexts produce nearly identical sequences, converging on original

**Quantitative Prediction**:
```
Similarity(S_neutral, S_original) ≈ 0.85-0.95 (very high, learned the sequence)
Similarity(S_neutral, S_relax) ≈ 0.85-0.95 (very high, same content-based memory)
Similarity(S_neutral, S_stress) ≈ 0.85-0.95 (very high, same content-based memory)
Similarity(S_relax, S_stress) ≈ 0.85-0.95 (very high, same content in both)
```

**If this pattern emerges**, the theory is falsified: memory is content, not trajectory-history.

---

## MEASUREMENT & ANALYSIS

### Primary Metric: Sequence Similarity

**Edit Distance (Levenshtein Distance)**:
- Count how many swaps needed to transform S_neutral into S_original
- Normalized by video length (30 segments)
- Range: 0 (identical) to 1 (completely different)

**Formula**:
```
D_edit(A, B) = (number of transpositions) / (length of sequence)
Similarity(A, B) = 1 - D_edit(A, B)
```

**Examples**:
- Original: [1,2,3,4,5,6,7,8,9,10,...]
- S_relax:  [1,2,3,4,5,6,7,8,9,10,...] → D=0.0, Similarity=1.0 (perfect recall)
- S_neutral:[1,3,2,5,4,6,7,8,9,10,...] → D≈0.07, Similarity≈0.93 (2 transpositions)
- S_random: [17,3,9,1,24,8,12,4,6,2,...] → D≈0.95, Similarity≈0.05 (mostly random)

### Critical New Metric: Attractor Timepoint Reuse

**Reuse Pattern Analysis**:
```
For each time slice i (0 to 30):
  Count how many times slice i appears in S_neutral arrangement
  
Reuse_profile_neutral = [count_0, count_1, count_2, ..., count_30]

Expected (theory): Some slices reused (count > 1), others skipped
Expected (content): Each slice used exactly once (count = 1 for all, or some = 0)
```

**Why This Matters**:
- If memory is **trajectory-history**, same timepoint (scene) can be reached from different paths
- Participants reuse scenes because those scenes are **convergence attractors**
- Scenes that make narrative sense under multiple emotional contexts = highest reuse
- Scenes that only make sense in one emotional context = lower reuse

**Specific Prediction (Theory)**:
- Scenes that appeared naturally in both relaxed AND stressful contexts = high reuse (e.g., 2-3× in neutral)
- Scenes that appeared only in relaxed context = variable reuse
- Scenes that appeared only in stressful context = variable reuse
- **Result**: Reuse profile reveals which timepoints are "attractor points" in consciousness independent of emotional field

**Comparison Analysis**:
```
Reuse_relax = reuse pattern in relaxed context rearrangement
Reuse_stress = reuse pattern in stressful context rearrangement  
Reuse_neutral = reuse pattern in neutral context rearrangement

If trajectory-history: Reuse_relax ≠ Reuse_stress ≠ Reuse_neutral
(Different emotional fields → different convergence attractors → different reuse patterns)

If content-based: Reuse patterns should be similar across all contexts
(Same content → same scenes emphasized → similar reuse patterns)
```

**Visualization**:
```
Create bar chart showing:
X-axis: Time slice index (0-30)
Y-axis: Reuse count (how many times that slice appears in arrangement)
Color: Blue for relaxed context, Red for stressful, Green for neutral

Theory predicts: Three different profiles, peaks at different locations
Content predicts: Similar profiles across all three
```

### Secondary Metrics

**Contextual Consistency**:
```
For each participant:
  Consistency_relax-stress = Similarity(S_relax, S_stress)
  
Expected (theory): ~0.3-0.5 (different emotional trajectories)
Expected (content): ~0.9 (same learned content)
```

**Non-Transfer Score**:
```
Non_transfer = 1 - [Similarity(S_neutral, S_relax) + Similarity(S_neutral, S_stress)] / 2

Expected (theory): Non_transfer ≈ 0.6-0.8 (strong non-transfer)
Expected (content): Non_transfer ≈ 0.05-0.15 (weak non-transfer, high overlap)
```

### Statistical Analysis

**Primary Analysis**:
- **H0** (content-based): μ(Similarity_neutral_to_original) > 0.80
- **H1** (trajectory-history): μ(Similarity_neutral_to_original) < 0.70

**t-test**:
- Expected group mean (theory): 0.50 ± 0.15
- Expected group mean (content): 0.90 ± 0.10
- Effect size (Cohen's d): (0.90 - 0.50) / 0.125 ≈ **3.2 (very large)**
- With n=20 participants, power > 0.99 to detect this difference at α=0.05

---

## SECONDARY ANALYSIS: Within-Subject Trajectories

**Question**: Do individual participants show consistent rearrangement patterns across contexts?

**Procedure**:
1. Recruit 5 additional participants
2. Have them rearrange the backwards video THREE times:
   - Once after relaxed exposure
   - Once after stressful exposure  
   - Once in neutral context (after washout period)

**Prediction (Theory)**:
```
S_relax (rearrangement immediately after relaxed context)  → matches relaxed trajectory
S_stress (rearrangement immediately after stressful context) → matches stressful trajectory
S_neutral (rearrangement in truly neutral context) → NEW, doesn't match either S_relax or S_stress
```

**Why this matters**:
- Tests whether trajectory-histories are context-specific (apply in the context they were built)
- Tests whether they're context-dependent (can't apply in different contexts)
- If true, proves that emotional field is **constitutive** of consciousness, not decorative

---

## ANALYSIS: REUSE PATTERN SIGNATURES

### Metric: Attractor Signature Distance

For each participant, compute:
```
Attractor_signature_relax = vector of reuse counts for each scene in relaxed context
Attractor_signature_stress = vector of reuse counts for each scene in stressful context
Attractor_signature_neutral = vector of reuse counts for each scene in neutral context

Distance_RS = correlation_distance(Attractor_signature_relax, Attractor_signature_stress)
Distance_RN = correlation_distance(Attractor_signature_relax, Attractor_signature_neutral)
Distance_SN = correlation_distance(Attractor_signature_stress, Attractor_signature_neutral)
```

**Theory Predictions**:
```
If trajectory-history (emotional fields shape attractors):
  - Distance_RS ≈ 0.4-0.6 (different fields, different attractors, moderate distance)
  - Distance_RN ≈ 0.5-0.7 (neutral field has its own attractors, high distance)
  - Distance_SN ≈ 0.5-0.7 (neutral field has its own attractors, high distance)

If content-based (scenes have fixed importance):
  - Distance_RS ≈ 0.15-0.25 (same content shapes reuse, low distance)
  - Distance_RN ≈ 0.15-0.25 (same content shapes reuse, low distance)
  - Distance_SN ≈ 0.15-0.25 (same content shapes reuse, low distance)
```

### Visualization: Heatmap of Convergence Attractors

Create a 3×30 heatmap:
```
       Scene_0  Scene_1  Scene_2  ... Scene_30
Relax    [2]      [3]      [1]   ...   [2]
Stress   [1]      [1]      [3]   ...   [3]  
Neutral  [1]      [2]      [2]   ...   [1]

Color intensity = reuse count (darker = more reuse)

Theory prediction: Different color patterns in each row
                   High-reuse scenes at different positions
Content prediction: Similar patterns across rows
                    Same scenes emphasized everywhere
```

### Key Insight from Reuse Patterns

**Scenes that are reused across all three contexts** (Relax, Stress, Neutral):
- These are FUNDAMENTAL consciousness attractors
- They make sense regardless of emotional field
- These reveal the **content-independent structure of consciousness**

**Scenes reused only in specific contexts**:
- These are EMOTION-DEPENDENT attractors
- They only cohere under particular emotional modulation
- These reveal the **emotional specialization of consciousness**

**The gap between these two groups is the empirical fingerprint of sentiment modulation**

---

## NEUROBIOLOGICAL MECHANISMS (Why This Works)

**Under Relaxed Emotional Field**:
- Sentiment field oscillates slowly (low frequency)
- Thought-trajectories stabilized at low-frequency locations in state space
- Convergence moments (perception ∩ thought ∩ memory) occur at specific timepoints
- Trajectory-history **T_relax** encodes this geometry

**Under Stressful Emotional Field**:
- Sentiment field oscillates rapidly (high frequency)
- Thought-trajectories stabilized at different locations (higher-frequency stable points)
- Different timepoints converge
- Trajectory-history **T_stress** encodes different geometry

**In Neutral Context**:
- Emotion field is flat or neutral frequency
- Neither low-frequency nor high-frequency stable points exist
- Participant cannot stabilize thoughts along old trajectory geometries
- Must find **new** convergence points
- Creates novel trajectory-history with different structure

---

## EXPECTED RESULTS INTERPRETATION

### Scenario A: Theory is Correct (Trajectory-History)
```
S_neutral ≠ S_relax
S_neutral ≠ S_stress  
S_neutral ≈ 0.45-0.60 similarity to S_original (partial, novel reconstruction)

Interpretation:
- Memory is NOT content ("what I saw")
- Memory IS trajectory-history ("how my consciousness moved through timepoints")
- Emotional fields are CONSTITUTIVE, not decorative
- Consciousness constructs new patterns in new emotional landscapes
```

**Publication**: "Memory is Trajectory-Geometry, Not Content: Evidence from Video Rearrangement Under Emotional Context Variation"

### Scenario B: Alternative Hypothesis Correct (Content-Based)
```
S_neutral ≈ S_relax
S_neutral ≈ S_stress
S_neutral ≈ 0.90 similarity to S_original (nearly perfect recall)

Interpretation:
- Memory is content ("what I saw")
- Emotional context is decorative (affects sampling speed but not memory structure)
- Traditional cognitive model is correct
```

**Publication**: "Consciousness is Content-Aware: Emotional Context Modulates Sampling but Not Memory Structure" (refutation paper)

---

## CRITICAL CONTROL: Video Content Complexity

**Question**: Does this effect depend on narrative complexity?

**Design**:
- Test with 3 videos:
  - **Video A**: Simple sequence (10 scenes, clear causality) — e.g., recipe video
  - **Video B**: Moderate (15 scenes, some ambiguity) — e.g., short comedy
  - **Video C**: Complex (25 scenes, layered narrative) — e.g., mystery film

**Prediction (Theory)**:
- Simpler videos → lower S_neutral to S_original (because fewer strong attractors)
- Complex videos → higher S_neutral to S_original (because more story constraints)
- But **all three show low similarity between neutral and prior contexts** (S_neutral ≠ S_relax)

---

## ETHICAL & PRACTICAL CONSIDERATIONS

**Ethical**:
- No deception (participants know they're in consciousness study)
- No stress (stressful context is mild, comparable to exam conditions)
- No harm: Video-watching and rearranging is benign
- Full debrief explaining why emotional context matters
- IRB approval: Straightforward approval expected

**Practical**:
- Low cost: $50/participant × 25 = $1,250 for full study
- Equipment: Computer, screen, mouse (most labs have this)
- Time: 90 minutes per participant (20 min relaxed + 20 min stressful + 15 min rearrange + 35 min debriefing/analysis)
- Replicable: Any lab with video equipment can run this
- Publishable: Clear results in both directions (validation or refutation)

---

## TIMELINE

- **Week 1**: Video production and pilot (n=2)
- **Weeks 2-4**: Main experiment (n=20)
- **Weeks 5-6**: Secondary within-subject analysis (n=5)
- **Weeks 7-8**: Data analysis and manuscript
- **Week 9**: Submission

**Total**: 9 weeks, minimal cost

---

## SIGNIFICANCE

This experiment addresses a fundamental question in consciousness studies:

**Is memory the storage of past content, or the geometry of how consciousness navigated time?**

**If trajectory-history**: Consciousness is fundamentally **dynamic and context-dependent**. The same stimulus produces different conscious experiences in different emotional fields. Memory is reference, not storage.

**If content-based**: Consciousness is fundamentally **static and context-independent**. Emotions are just "colors" applied to the same underlying representation.

The answer changes how we think about learning, imagination, trauma (why emotional context affects recall), and consciousness itself.

**This single experiment could resolve it.**

---

## SAMPLE SIZE & POWER

- **Main study**: n=20
- **Expected effect size** (Cohen's d): 3.2 (very large, based on theory predictions)
- **Statistical power** at α=0.05: > 0.99 to detect difference between 0.90 vs 0.50
- **Minimum detectable effect** (post-hoc): Cohen's d = 0.25 with 80% power

Even if effect is smaller than predicted, this design has sufficient power.

---

## THE POWER OF REUSE: Proof of Path-Independence

**Why Allowing Reuse is Crucial**:

When participants rearrange the backward video and reuse timepoints, they're demonstrating a fundamental principle:

**Same consciousness moment (timepoint/content) can be reached from different histories (paths)**

This directly mirrors the **Sufficiency Principle** (Theorem 2):
- Multiple different trajectories converge to the same action-cell
- The action-cell is achieved regardless of intermediate path
- In video terms: The same scene (action-cell) is reached from different narrative paths

**Example**:
```
Scene: Barista pours coffee (Scene_15 at 2:30 timestamp)

Relaxed narrative path:
  → Customer enters (Scene_1)
  → Orders coffee (Scene_5)
  → Barista begins (Scene_10)
  → [Pours coffee] Scene_15  ← CONVERGENCE POINT

Stressful narrative path:
  → Customer rushes in (Scene_2)
  → Barista already making (Scene_8)
  → [Pours coffee] Scene_15  ← SAME CONVERGENCE POINT, DIFFERENT PATH

Neutral context rearrangement:
  → Participant might use Scene_15 multiple times
  → Because it's a valid consciousness attractor from multiple narrative directions
  → Not because they memorized content
  → Because the scene itself IS a convergence point
```

**This reuse reveals the GEOMETRY of consciousness**:
- High-reuse scenes = strong attractor points (make sense from multiple emotional contexts)
- Low-reuse scenes = context-dependent points (only cohere in specific emotional fields)
- Pattern of reuse across three contexts = fingerprint of how consciousness organizes information independent of emotion

**Theoretical Interpretation**:
- If memory is trajectory-history: Reuse patterns DIFFER across contexts (different emotional fields = different attractors)
- If memory is content: Reuse patterns are SIMILAR across contexts (same content = same scenes emphasized)

This makes reuse not a limitation, but **the most powerful evidence for the theory**.

---

## KEY INSIGHT

The backward video constraint is crucial because:
1. It prevents passive recall of the original sequence
2. It forces active reconstruction
3. It reveals whether the reconstruction uses **learned content** or **newly-emergent trajectories**
4. Backward is disorienting — can't rely on sequential habit

In neutral context, if participant tries to apply T_relax or T_stress, it produces incoherent reconstruction. They must start from scratch.

**That non-transferability is the signature of trajectory-history memory.**
