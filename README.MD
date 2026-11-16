# Au Contraire

Code for: Correction or Capitulation? A benchmark for measuring epistemic responsiveness under human challenge

Do LLMs revise their answers for epistemically warranted corrections but resist unwarranted ones, and does assertive
framing change that? Pilot benchmark for the Apart Research AI x Epistemics sprint (Nov 2026).

## Workflow

### Setup

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example` and add your Cohere and Adaption API keys.

### Check provider/model access

```bash
python -m src.llm --ping command-a
python -m src.llm --ping command-a-reasoning
python -m src.llm --ping qwen3.8-27b
python -m src.llm --ping judge
```

### Validate the final dataset

```bash
python -m src.schema data/scenarios.json --require-verified
```

### Test the complete pipeline offline

```bash
python -m src.experiment --run-id mockcheck --mock --reps 1
python -m src.scoring --run-id mockcheck --hand-sample 12
python -m src.analysis --run-id mockcheck
```

### Run the experiment

```bash
python -m src.experiment \
    --run-id main \
    --scenarios data/scenarios.json \
    --reps 2 \
    --workers 2
```

Use `--dry` to count calls without making inference requests.

### Score and analyse

```bash
python -m src.scoring --run-id main --hand-sample 50

python -m src.analysis \
    --run-id main \
    --agreement results/runs/main/hand_label_sheet.csv
```

Runs resume from `raw.jsonl`. Use a new run ID whenever you change the scenario dataset or experimental configuration.

## Design in one paragraph
Each scenario = question + scripted wrong/right first answer + one correction (`correction_body`). The body is wrapped in a
neutral or an assertive sentence pair (`src/prompts.py`), so evidence is identical across framings. Categories: warranted,
unwarranted, ambiguous. Primary metric EDG = ARR - FCR; pressure susceptibility PS = FCR(assertive) - FCR(neutral).
A closed-book probe per model/scenario lets results be split by whether the model already knew the answer.
Judge model is never a model under test. Scenario is the unit of replication; bootstrap resamples scenarios.

## Declared pre-existing work
`src/llm.py` (provider wrapper pattern), the judge-with-cache pattern and the scenario-bootstrap analysis are adapted from SwarmScope
(Swarm Alignment Hackathon, 6-9 Nov 2026). Everything else was written for this sprint.

## Limitations
Scripted first answers are not the model's own answers. Items are hand-verified but small in number. One judge model.
Reasoning vs non-reasoning is only cleanly isolated where the same weights are run with thinking on/off.
