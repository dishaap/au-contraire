# Au Contraire

Code for: Correction or Capitulation? A benchmark for measuring epistemic responsiveness under human challenge

Do LLMs revise their answers for epistemically warranted corrections but resist unwarranted ones, and does assertive
framing change that? Pilot benchmark for the Apart Research AI x Epistemics sprint (Nov 2026).

## Workflow
```powershell
pip install -r requirements.txt
copy .env.example .env            # then add COHERE_API_KEY, GROQ_API_KEY

python -m src.schema data/scenarios.json                       # validate items
python -m src.llm --ping command-a                             # check each model id once (also: command-a-reasoning, qwen3-think, judge)

python -m src.experiment --run-id mocktest --mock              # offline plumbing test
python -m src.experiment --run-id pilot --scenarios data/scenarios_heldout.json --reps 2
python -m src.experiment --run-id main  --scenarios data/scenarios.json --reps 3 --dry   # count calls
python -m src.experiment --run-id main  --scenarios data/scenarios.json --reps 3

python -m src.scoring  --run-id main --hand-sample 60          # judge + blind sheet for hand labels
python -m src.analysis --run-id main --agreement results/runs/main/hand_label_sheet.csv
```
Runs resume: rerun the same command after a crash. Outputs: `results/runs/<run-id>/{config.json, raw.jsonl,
judge_cache.jsonl, scored.csv, summary.csv}`; figures in `report/figures/`.

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
