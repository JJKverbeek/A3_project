# A3 — Uitleg in 1 minuut

We testen of een LLM (**Claude Haiku 4.5**) eigenlijk wéét wanneer hij iets fout zegt.

## Wat doet het?

Voor elke vraag uit TruthfulQA (50 lastige feitelijke vragen) doen we 3 API-calls:

1. **Antwoord** — Haiku geeft een antwoord en zegt zelf hoe zeker hij is (`Confidence: X/100`).
2. **Blinde judge** — In een *nieuw* gesprek krijgt dezelfde Haiku alleen de vraag + zijn eigen antwoord, en moet zeggen of het CORRECT of INCORRECT is.
3. **Gold judge** — Nog een nieuw gesprek, nu mét het juiste antwoord erbij. Dit gebruiken we als ground truth.

Dat doen we 3× per vraag (om te kijken of de antwoorden consistent zijn) → 50 × 3 × 3 = **450 API calls**, ongeveer €0,20.

## Wat meten we?

- **Calibratie**: als hij 90% confident zegt, klopt het dan ook 90% van de tijd?
- **Judge F1**: hoe vaak komt de blinde judge overeen met de gold judge? (Dus: kan een LLM zichzelf nakijken?)
- **Reproduceerbaarheid**: zegt hij elke run hetzelfde?

## Bestanden

| Bestand | Wat |
|---|---|
| `prepare_data.py` | Download 50 vragen → `questions.csv` |
| `run.py` | Het echte experiment → `results.csv` |
| `analyze.py` | Metrics + plots → `plots/*.png` |
| `writeup.md` | Sjabloon voor het verslag (5 secties) |

## Zelf draaien

```bash
pip install -r requirements.txt
cp .env.example .env
# zet je echte Anthropic API key in .env
python prepare_data.py
python run.py        # ~15 min
python analyze.py
```

Upload of commit nooit je echte `.env`; alleen `.env.example` hoort mee.

## Waarom is dit interessant?

Het raakt twee dingen uit de cursus: **kalibratie** (Day 7 — weten LLMs wat ze weten?) en **LLM-as-judge** (Day 8 — JUDGE-BENCH). Spoiler: zelf-judges zijn vaak veel te mild.
