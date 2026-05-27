# CACHE SIMULATOR — DAY BY DAY BUILD SCHEDULE
## Follow this exactly. One day at a time. Do not jump ahead.

---

> **RULES**
> - Finish each day completely before moving to next
> - Test after every file you write
> - If something breaks — go to Section 10 of master doc FIRST before panicking
> - Every file you need to write is already written in the master doc — just copy it
> - Green checkboxes = things to tick off as you go

---

# WEEK 1 — FOUNDATION

---

## DAY 1 — Setup + Config
**Goal: Project exists, config works, environment ready**
**Time: 1-2 hours**

### Tasks
- [ ] Create project folder
- [ ] Set up virtual environment
- [ ] Install all dependencies
- [ ] Create folder structure
- [ ] Write config.py
- [ ] Verify config works

### Exact Steps

**Step 1: Create project**
```bash
mkdir cache_simulator
cd cache_simulator
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

**Step 2: Install dependencies**
```bash
pip install torch==2.0.1 numpy==1.24.3 plotly==5.15.0 pandas==2.0.3 scipy==1.11.1
```

**Step 3: Create ALL folders and files at once**
```bash
mkdir cache policies ml data data/traces visualization tests results models plots
touch config.py main.py
touch cache/__init__.py cache/decoder.py cache/block.py cache/ram.py cache/simulator.py
touch policies/__init__.py policies/fifo.py policies/lru.py policies/lfu.py policies/belady.py policies/dqn.py
touch ml/__init__.py ml/network.py ml/replay_buffer.py ml/agent.py ml/monitor.py
touch data/__init__.py data/generator.py data/validator.py
touch visualization/__init__.py visualization/dashboard.py
touch tests/__init__.py tests/test_decoder.py tests/test_block.py tests/test_buffer.py tests/test_policies.py tests/test_pipeline.py
touch training_mode.py evaluation_mode.py hybrid_mode.py
```

**Step 4: Write config.py**
→ Go to **Section 5** of master doc
→ Copy the ENTIRE config.py block into your config.py file

**Step 5: Verify**
```bash
python config.py
```
Expected output:
```
==================================================
CONFIG VALIDATION PASSED
  Cache: 256B, 64B blocks, 4-way
  Sets: 1
  Bits: 26t / 0i / 6o
  State size: 16
  Action size: 4
==================================================
```

✅ **Day 1 complete when**: `python config.py` prints validation passed with no errors

---

## DAY 2 — Address Decoder + Cache Block
**Goal: Core cache data structures working**
**Time: 2-3 hours**

### Tasks
- [ ] Write cache/decoder.py
- [ ] Write cache/block.py
- [ ] Write cache/ram.py
- [ ] Test decoder manually

### Exact Steps

**Step 1: Write decoder**
→ Go to **Section 6.3** of master doc
→ Copy entire AddressDecoder class into `cache/decoder.py`

**Step 2: Write block**
→ Go to **Section 6.4** of master doc
→ Copy entire CacheBlock class into `cache/block.py`

**Step 3: Write RAM**
→ Go to **Section 6.5** of master doc
→ Copy entire RAM class into `cache/ram.py`

**Step 4: Quick manual test — create test_day2.py temporarily**
```python
# test_day2.py — delete after Day 2
from config import CONFIG
from cache.decoder import AddressDecoder
from cache.block import CacheBlock
from cache.ram import RAM

# Test decoder
decoder = AddressDecoder(CONFIG)
decoder.validate()
decoder.debug_address(0x00001A3F)

# Test block
block = CacheBlock(tag=1)
print(f"Block created: {block}")
print(f"Dirty: {block.dirty}")      # should be False
block.write()
print(f"After write dirty: {block.dirty}")   # should be True
block.read()
print(f"Recency after read: {block.recency}")  # should be 0

# Test RAM
ram = RAM()
ram.storage[1] = [1, 2, 3]
block2 = CacheBlock(tag=1)
block2.dirty = True
ram.writeback(block2)
print(f"RAM writebacks: {ram.writebacks}")  # should be 1

print("\nDay 2 manual tests passed!")
```

```bash
python test_day2.py
```

✅ **Day 2 complete when**: All manual tests print expected values, no errors

---

## DAY 3 — Cache Simulator Core
**Goal: Cache can process memory accesses, track hits/misses**
**Time: 3-4 hours**

### Tasks
- [ ] Write cache/simulator.py (Metrics, CacheSet, CacheSimulator)
- [ ] Test with a simple manual trace

### Exact Steps

**Step 1: Write simulator**
→ Go to **Section 6.6** of master doc
→ Copy entire file into `cache/simulator.py`
→ Also copy the `run_simulation` and `run_simulation_averaged` functions at the bottom

**Step 2: Manual test — create test_day3.py temporarily**
```python
# test_day3.py — delete after Day 3
from config import CONFIG
from cache.simulator import CacheSimulator
from policies.lru import LRUPolicy   # we haven't written this yet
# Actually use a dummy policy first:

class DummyPolicy:
    """Evicts first block — for testing only"""
    last_was_override = False
    def select_eviction(self, cache_set, step=None, incoming_tag=None):
        return cache_set.blocks[0]
    def reset(self): pass

sim = CacheSimulator(policy=DummyPolicy())

# Manual trace: access address 0, 64, 128, 0
# Block size 64 → these are 3 different blocks
trace = [
    ('R', 0),    # MISS — load block 0
    ('R', 64),   # MISS — load block 64
    ('R', 0),    # HIT  — block 0 still in cache
    ('W', 64),   # HIT  — block 64 still in cache, sets dirty
]

for op, addr in trace:
    result = sim.access(op, addr)
    print(f"{op} {addr:#06x} → {'HIT' if result['hit'] else 'MISS'}")

print(f"\nMetrics: {sim.metrics}")
print(f"Hits: {sim.metrics.hits}")         # should be 2
print(f"Misses: {sim.metrics.misses}")     # should be 2
print(f"Hit rate: {sim.metrics.hit_rate}") # should be 0.5
```

```bash
python test_day3.py
```

✅ **Day 3 complete when**: Hit=2, Miss=2, HitRate=0.5 printed correctly

---

## DAY 4 — All Replacement Policies
**Goal: FIFO, LRU, LFU, Belady's all working**
**Time: 2-3 hours**

### Tasks
- [ ] Write policies/fifo.py
- [ ] Write policies/lru.py
- [ ] Write policies/lfu.py
- [ ] Write policies/belady.py
- [ ] Test each policy

### Exact Steps

**Step 1: Write all policy files**
→ **Section 6.7** → copy into `policies/fifo.py`, `policies/lru.py`, `policies/lfu.py`
→ **Section 6.8** → copy into `policies/belady.py`

**Step 2: Manual test — create test_day4.py temporarily**
```python
# test_day4.py — delete after Day 4
from config import CONFIG
from cache.simulator import CacheSimulator
from cache.decoder import AddressDecoder
from data.generator import generate_trace, split_trace, extract_tags
from policies.fifo import FIFOPolicy
from policies.lru import LRUPolicy
from policies.lfu import LFUPolicy
from policies.belady import BeladyPolicy

# Generate a small test trace
import random
random.seed(42)
trace = generate_trace('zipfian', seed=42)
_, test = split_trace(trace)

results = {}
for name, policy in [('FIFO', FIFOPolicy()),
                      ('LRU',  LRUPolicy()),
                      ('LFU',  LFUPolicy())]:
    sim = CacheSimulator(policy=policy)
    sim.run_trace(test)
    results[name] = sim.metrics.hit_rate
    print(f"{name:6}: hit_rate={sim.metrics.hit_rate:.3f}")

# Belady needs precomputed table
decoder   = AddressDecoder(CONFIG)
tag_trace = extract_tags(trace, decoder)
belady    = BeladyPolicy()
belady.precompute(tag_trace)
sim_b     = CacheSimulator(policy=belady)
sim_b.run_trace(test)
results["Belady's"] = sim_b.metrics.hit_rate
print(f"Belady: hit_rate={sim_b.metrics.hit_rate:.3f}")

# Critical check: Belady must be best
assert results["Belady's"] >= results['LRU'] - 0.001, \
    "BUG: Belady must beat or match LRU!"
print("\nBelady optimality check: PASSED ✅")
```

```bash
python test_day4.py
```

✅ **Day 4 complete when**: All 4 policies print hit rates, Belady check passes

---

## DAY 5 — Trace Generator + Validator
**Goal: Can generate and validate all 4 trace patterns**
**Time: 2-3 hours**

### Tasks
- [ ] Write data/generator.py
- [ ] Write data/validator.py
- [ ] Test all 4 generators
- [ ] Verify Zipfian is real power law (not fake)

### Exact Steps

**Step 1: Write generator**
→ Go to **Section 6.9** of master doc
→ Copy entire file into `data/generator.py`

**Step 2: Write validator**
→ Go to **Section 6.10** of master doc
→ Copy entire file into `data/validator.py`

**Step 3: Test all generators — create test_day5.py**
```python
# test_day5.py — delete after Day 5
from data.generator import generate_trace, split_trace
from data.validator import validate_trace, validate_zipfian, validate_split

patterns = ['sequential', 'random', 'stride', 'zipfian']

for pattern in patterns:
    trace = generate_trace(pattern, seed=42)
    validate_trace(trace, pattern)

# Special Zipfian check
zipf_trace = generate_trace('zipfian', seed=42)
validate_zipfian(zipf_trace)

# Split check
train, test = split_trace(zipf_trace)
validate_split(train, test)

print("\nAll trace generators working ✅")
```

```bash
python test_day5.py
```

✅ **Day 5 complete when**: All 4 patterns validate, Zipfian shows skew ratio > 3x

---

# WEEK 2 — AI LAYER

---

## DAY 6 — Neural Network + Replay Buffer
**Goal: DQN building blocks ready**
**Time: 2-3 hours**

### Tasks
- [ ] Write ml/network.py
- [ ] Write ml/replay_buffer.py
- [ ] Test network forward pass
- [ ] Test buffer push/sample

### Exact Steps

**Step 1: Write network**
→ Go to **Section 6.11** of master doc
→ Copy entire DQNNetwork class into `ml/network.py`

**Step 2: Write replay buffer**
→ Go to **Section 6.12** of master doc
→ Copy entire ReplayBuffer class into `ml/replay_buffer.py`

**Step 3: Test — create test_day6.py**
```python
# test_day6.py — delete after Day 6
import numpy as np
import torch
from config import CONFIG
from ml.network import DQNNetwork
from ml.replay_buffer import ReplayBuffer

# Test network
net = DQNNetwork()
print(f"Network created: {net}")

# Test forward pass with random state
state = np.random.rand(CONFIG['state_size']).astype(np.float32)
q_values = net.get_q_values(state)
print(f"Q-values shape: {q_values.shape}")   # should be (4,)
print(f"Q-values: {q_values}")
action = net.get_action(state)
print(f"Action: {action}")                    # should be 0,1,2 or 3

# Test replay buffer
buf = ReplayBuffer()
print(f"\nBuffer ready: {buf.ready()}")        # False
for i in range(500):
    s  = np.random.rand(16).astype(np.float32)
    ns = np.random.rand(16).astype(np.float32)
    buf.push(s, i%4, float(i%2)*2-1, ns)
print(f"Buffer ready: {buf.ready()}")          # True
print(f"Buffer size: {len(buf)}")              # 500

states, actions, rewards, next_states, dones = buf.sample(32)
print(f"Batch shapes: {states.shape}, {actions.shape}, {rewards.shape}")
# should be (32,16), (32,), (32,)

print("\nDay 6 tests passed ✅")
```

```bash
python test_day6.py
```

✅ **Day 6 complete when**: Network outputs shape (4,), buffer samples batch of 32

---

## DAY 7 — Training Monitor + DQN Agent
**Goal: Complete DQN agent that can learn**
**Time: 3-4 hours**

### Tasks
- [ ] Write ml/monitor.py
- [ ] Write ml/agent.py
- [ ] Test agent eviction decision
- [ ] Test agent train step

### Exact Steps

**Step 1: Write monitor**
→ Go to **Section 6.13** of master doc
→ Copy entire TrainingMonitor class into `ml/monitor.py`

**Step 2: Write agent**
→ Go to **Section 6.14** of master doc
→ Copy entire DQNAgent class into `ml/agent.py`

**Step 3: Test — create test_day7.py**
```python
# test_day7.py — delete after Day 7
import numpy as np
from config import CONFIG
from cache.simulator import CacheSimulator, CacheSet
from cache.block import CacheBlock
from ml.agent import DQNAgent

# Test agent creation
agent = DQNAgent(CONFIG)
print(f"Agent created. Epsilon: {agent.epsilon}")   # 1.0

# Build a fake full cache set to test eviction
cache_set = CacheSet(capacity=4)
for i in range(4):
    block = CacheBlock(tag=i)
    block.frequency = i
    block.recency   = 4 - i
    cache_set.blocks.append(block)

print(f"Cache set: {cache_set}")

# Test eviction (should be random since epsilon=1.0)
evicted = agent.select_eviction(cache_set)
print(f"Evicted block tag: {evicted.tag}")   # random 0-3

# Test epsilon decay
for _ in range(100):
    agent.decay_epsilon()
print(f"Epsilon after 100 decays: {agent.epsilon:.3f}")  # ~0.606

# Fill buffer and test training
for _ in range(600):
    s  = np.random.rand(16).astype(np.float32)
    ns = np.random.rand(16).astype(np.float32)
    agent.replay_buffer.push(s, 0, 1.0, ns)

loss = agent.train_step()
print(f"Training loss: {loss}")   # should be a float, not None

# Test save/load
agent.save('models/test_model.pth')
agent2 = DQNAgent(CONFIG)
agent2.load('models/test_model.pth')
print(f"Save/load: ✅")

print("\nDay 7 tests passed ✅")
```

```bash
python test_day7.py
```

✅ **Day 7 complete when**: Epsilon decays, training returns a loss value, save/load works

---

## DAY 8 — DQN Policy Wrapper
**Goal: DQN can be used as a drop-in replacement for LRU**
**Time: 2 hours**

### Tasks
- [ ] Write policies/dqn.py (PolicySwitcher)
- [ ] Test warm-up transition
- [ ] Verify DQN and LRU have same interface

### Exact Steps

**Step 1: Write DQN policy**
→ Go to **Section 6.15** of master doc
→ Copy entire DQNPolicy class into `policies/dqn.py`

**Step 2: Test — create test_day8.py**
```python
# test_day8.py — delete after Day 8
from config import CONFIG
from ml.agent import DQNAgent
from policies.dqn import DQNPolicy
from policies.lru import LRUPolicy
from cache.simulator import CacheSimulator
from data.generator import generate_trace, split_trace

# Test that DQNPolicy has same interface as LRUPolicy
agent  = DQNAgent(CONFIG)
policy = DQNPolicy(agent=agent, use_lru_warmup=True)

print(f"Policy: {policy}")
print(f"Current policy: {policy.current_policy}")  # LRU (warm-up)

# Run a short trace through DQN policy
trace = generate_trace('zipfian', seed=42)
_, test = split_trace(trace)
test_short = test[:500]  # short for speed

sim = CacheSimulator(policy=policy)
sim.run_trace(test_short)
print(f"Hit rate: {sim.metrics.hit_rate:.3f}")
print(f"Total accesses: {sim.metrics.total_accesses}")

# Test no-warmup version (Experiment A)
agent2  = DQNAgent(CONFIG)
policy2 = DQNPolicy(agent=agent2, use_lru_warmup=False)
sim2    = CacheSimulator(policy=policy2)
sim2.run_trace(test_short)
print(f"No-warmup hit rate: {sim2.metrics.hit_rate:.3f}")

print("\nDay 8 tests passed ✅")
```

```bash
python test_day8.py
```

✅ **Day 8 complete when**: Both policies run trace without errors, print hit rates

---

# WEEK 3 — TRAINING + EVALUATION

---

## DAY 9 — Training Mode
**Goal: DQN trains for 10 episodes and saves model**
**Time: 3-4 hours + training time**

### Tasks
- [ ] Write training_mode.py
- [ ] Run first training
- [ ] Verify model saved
- [ ] Check training monitor output

### Exact Steps

**Step 1: Write training mode**
→ Go to **Section 6.18** of master doc
→ Copy entire `run_training_mode` function into `training_mode.py`

**Step 2: Write main.py**
→ Go to **Section 6.22** of master doc
→ Copy entire `main.py`

**Step 3: Run training**
```bash
python main.py train
```

Expected output (roughly):
```
==================================================
CONFIG VALIDATION PASSED
...
[1/6] Generating Zipfian training trace...
[2/6] Splitting train/test...
[3/6] Precomputing Belady next-use table...
[4/6] Initializing DQN agent...
[5/6] Training...
Episode  1 | HitRate=0.XXX | Reward=XXX | ε=0.XXX | Loss=X.XXXX | Overrides=XX
Episode  2 | HitRate=0.XXX | ...
...
Episode 10 | HitRate=0.XXX | ...
[6/6] Saving model...
Model saved → models/dqn_cache.pth
```

**Step 4: Verify model exists**
```bash
ls models/
# Should show: dqn_cache.pth
```

**If training crashes**: Go to **Section 10** of master doc, start at Check 1.

✅ **Day 9 complete when**: `models/dqn_cache.pth` exists, all 10 episodes printed

---

## DAY 10 — Evaluation Mode
**Goal: All 5 policies benchmarked on all 4 workloads**
**Time: 2-3 hours + eval time**

### Tasks
- [ ] Write evaluation_mode.py
- [ ] Run full evaluation
- [ ] Verify results.csv generated
- [ ] Check results make sense

### Exact Steps

**Step 1: Write evaluation mode**
→ Go to **Section 6.19** of master doc
→ Copy entire file into `evaluation_mode.py`

**Step 2: Run evaluation**
```bash
python main.py eval
```

Expected output:
```
--- Pattern: SEQUENTIAL ---
  [1/60] FIFO         | sequential | seed=42
  [2/60] LRU          | sequential | seed=42
  ...
  → HitRate=X.XXX | AMAT=XX.X | WritebackRate=X.XXX
...
Results saved → results/results.csv
```

**Step 3: Verify results**
```bash
# Check file was created
ls results/
# Should show: results.csv

# Preview first few lines
head -20 results/results.csv
```

**Step 4: Quick sanity check**
```python
# Run in python terminal
import csv
results = {}
with open('results/results.csv') as f:
    for row in csv.DictReader(f):
        print(f"{row['policy']:12} | {row['pattern']:10} | hit_rate={row['hit_rate']}")
```

**Critical check**: Belady's hit_rate must always be highest for each pattern.
If DQN > Belady's → there is a bug. Go to Section 10.

✅ **Day 10 complete when**: results.csv has 20 rows (5 policies × 4 patterns), Belady's always highest

---

## DAY 11 — Hybrid Mode
**Goal: Live LRU→DQN transition demonstration working**
**Time: 1-2 hours**

### Tasks
- [ ] Write hybrid_mode.py
- [ ] Run hybrid demo
- [ ] Verify switch point printed

### Exact Steps

**Step 1: Write hybrid mode**
→ Go to **Section 6.20** of master doc
→ Copy entire file into `hybrid_mode.py`

**Step 2: Run hybrid**
```bash
python main.py hybrid
```

Expected output:
```
Running live simulation...
  Phase 1: LRU (until DQN confident)
  Phase 2: DQN (after confidence threshold met)

  Step     0 | Policy=LRU | HitRate=0.XXX | ε=X.XXX
  Step  1000 | Policy=LRU | HitRate=0.XXX | ε=X.XXX
  ...
  >>> SWITCHED TO DQN at step XXXX <<<
  ...
HYBRID MODE COMPLETE
  Switch step         : XXXX
  Pre-switch hit rate : 0.XXX  (LRU)
  Post-switch hit rate: 0.XXX  (DQN)
  Improvement         : +X.X%
```

**Note**: If switch never happens (stays LRU the whole time):
- Agent never became confident enough
- This is not necessarily a bug — it means DQN isn't sure enough
- Reduce `confidence_threshold` in config.py from 0.3 to 0.1 and retry

✅ **Day 11 complete when**: Switch point printed, pre/post hit rates shown

---

# WEEK 4 — TESTING + VISUALIZATION

---

## DAY 12 — Test Suite
**Goal: All tests pass — simulator is provably correct**
**Time: 3-4 hours**

### Tasks
- [ ] Write all 4 test files
- [ ] Run full test suite
- [ ] Fix any failing tests
- [ ] All tests green

### Exact Steps

**Step 1: Write all test files**
→ Go to **Section 7** of master doc
→ Copy `TestAddressDecoder` into `tests/test_decoder.py`
→ Copy `TestCacheBlock` into `tests/test_block.py`
→ Copy `TestReplayBuffer` into `tests/test_buffer.py`
→ Copy `TestLRUPolicy`, `TestBeladyPolicy`, `TestMetrics` into `tests/test_policies.py`

**Step 2: Run test suite**
```bash
python main.py test
```

Or run directly:
```bash
python -m unittest discover tests/ -v
```

Expected output:
```
test_basic_lru_eviction_no_hits ... ok
test_decode_reconstruct_max ... ok
test_decode_reconstruct_typical ... ok
...
Ran 20 tests in X.XXXs
OK
```

**If a test fails**:
- Read the error message carefully
- It tells you exactly which assertion failed
- Go back to the corresponding source file and check that function
- Most common cause: copy-paste error in the source file

✅ **Day 12 complete when**: `Ran XX tests... OK` with zero failures

---

## DAY 13 — Visualization Dashboard
**Goal: All 6 interactive charts generated**
**Time: 2-3 hours**

### Tasks
- [ ] Write visualization/dashboard.py
- [ ] Run visualization mode
- [ ] Open charts in browser
- [ ] Verify all 6 charts look correct

### Exact Steps

**Step 1: Write dashboard**
→ Go to **Section 6.21** of master doc
→ Copy entire file into `visualization/dashboard.py`

**Step 2: Generate charts**
```bash
python main.py visualize
```

Expected output:
```
Chart 1 saved → plots/01_hit_rate.html
Chart 2 saved → plots/02_amat.html
Chart 3 saved → plots/03_gap_to_optimal.html
Chart 5 saved → plots/05_writeback_rate.html
```

**Step 3: Open and verify each chart in browser**

Open `plots/01_hit_rate.html`:
- Should show grouped bar chart
- 5 policy groups, 4 workload bars each
- Belady's bars should be tallest

Open `plots/03_gap_to_optimal.html`:
- Should show line chart
- All lines should be above 0 (gap to optimal)
- DQN line should be closest to 0 (or at least close to LRU)

**Step 4: Generate convergence chart (needs training monitor)**
```python
# Run this in Python to generate chart 4 and 6
# Only if you saved the monitor during training
# If not, skip — charts 1,2,3,5 are the important ones
```

✅ **Day 13 complete when**: At least charts 1, 2, 3, 5 open in browser and look correct

---

## DAY 14 — Final Validation + Cleanup
**Goal: Everything works end to end, project is clean**
**Time: 2-3 hours**

### Tasks
- [ ] Run validate mode
- [ ] Run full test suite one more time
- [ ] Run full pipeline start to finish
- [ ] Delete all test_dayX.py files
- [ ] Check folder structure is clean
- [ ] Read through results and prepare to explain them

### Exact Steps

**Step 1: Full validation**
```bash
python main.py validate
```
Should print all checks passed.

**Step 2: Full test suite**
```bash
python main.py test
```
Must be 100% passing. Zero failures.

**Step 3: Full pipeline run (end to end)**
```bash
python main.py train
python main.py eval
python main.py visualize
python main.py hybrid
```
All four should complete without errors.

**Step 4: Cleanup**
```bash
# Delete temporary test files
rm test_day2.py test_day3.py test_day4.py test_day5.py
rm test_day6.py test_day7.py test_day8.py
```

**Step 5: Check your results**
Open `results/results.csv` and note:
- Best DQN hit rate (which workload?)
- Gap between DQN and Belady's
- Whether DQN beats LRU on 3+ workloads

**Step 6: Prepare 3 sentences to say about results**
Fill in the blanks:
```
"My DQN achieved X% hit rate on Zipfian workloads,
compared to Y% for LRU and Z% for Belady's optimal.
This places DQN W% away from the theoretical maximum."
```

✅ **Day 14 complete when**: Full pipeline runs clean, you can explain your results in 3 sentences

---

# SUMMARY TIMELINE

| Day | What You Build | Section in Master Doc |
|---|---|---|
| 1 | Setup + Config | Section 4, 5 |
| 2 | Decoder + Block + RAM | Section 6.3, 6.4, 6.5 |
| 3 | Cache Simulator Core | Section 6.6 |
| 4 | FIFO + LRU + LFU + Belady's | Section 6.7, 6.8 |
| 5 | Trace Generator + Validator | Section 6.9, 6.10 |
| 6 | Neural Network + Replay Buffer | Section 6.11, 6.12 |
| 7 | Training Monitor + DQN Agent | Section 6.13, 6.14 |
| 8 | DQN Policy Wrapper | Section 6.15 |
| 9 | Training Mode (train DQN) | Section 6.18, 6.22 |
| 10 | Evaluation Mode | Section 6.19 |
| 11 | Hybrid Mode | Section 6.20 |
| 12 | Full Test Suite | Section 7 |
| 13 | Visualization Dashboard | Section 6.21 |
| 14 | Final Validation + Cleanup | Section 8 |

**Total: 14 days. ~2-4 hours per day.**

---

# IF SOMETHING GOES WRONG

| Problem | Where to Look |
|---|---|
| Import error | Check `__init__.py` files exist in each folder |
| Config error | Re-run `python config.py` and read error message |
| Test failing | Section 10 of master doc — debugging checklist |
| DQN not learning | Section 10, Check 1 through 7 in order |
| Belady beats nothing | Bug in BeladyPolicy.precompute — recheck Section 6.8 |
| Charts empty | Verify results.csv exists first, then rerun visualize |
| Model not found | Run train before eval |

---

# AFTER DAY 14 — WHAT'S LEFT

Once the project works:

1. **Read LeCaR paper** (20 min) — https://www.usenix.org/conference/hotstorage18/presentation/vietri
2. **Prepare viva answers** — Section 11 of master doc has 10 Q&A
3. **Write project report** — use Section 12 (related work) + your actual results
4. **Optional V2** — Section 13 of master doc (L1 cache) — only after v1 fully works

---

*Follow this schedule. One day at a time. The master doc has everything you need.*
*You are not figuring anything out — you are just copying and testing.*
