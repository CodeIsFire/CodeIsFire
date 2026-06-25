# Aaditya Bhardwaj

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=500&size=20&duration=3000&pause=1200&color=2E9EF7&center=false&vCenter=true&width=600&lines=Signal+Processing+%E2%86%92+Machine+Learning;CS+%40+SRM+University;Open+to+AI%2FML+roles" alt="Typing SVG" />

I spent the last few months pulling 5G signals apart in GNU Radio to see how they actually sync up — now I'm pointing that same "find the pattern, then prove it" instinct at machine learning. CS student at SRM University, building toward AI/ML roles.

<br>

### What I'm doing right now

```text
🔭  Building ML projects — moving from signal data to model-driven systems
🌱  Deepening my NLP and data analysis skills
🎯  Looking for AI/ML internships and collaborations
💬  Happy to talk Python, ML, algorithms, or how 5G synchronization actually works
```

<br>

Here's roughly the shape of how I think about a problem — turn a raw, noisy signal into something a model can actually learn from:

```python
import numpy as np

def extract_features(iq_samples: np.ndarray, window: int = 256) -> np.ndarray:
    """Turn raw I/Q samples into features a classifier can use."""
    power = np.abs(iq_samples) ** 2
    windows = power[: len(power) // window * window].reshape(-1, window)

    return np.column_stack([
        windows.mean(axis=1),   # average energy
        windows.std(axis=1),    # variability — flags noisy segments
        windows.max(axis=1),    # peak detection
    ])

# This is the same instinct behind the GNU Radio work below:
# noisy input -> structured signal -> something you can measure and trust.
```

<br>

## How I got here

**Open Source Contributor — GNU Radio** · Jun 2025 – Oct 2025

My first real research problem: figure out whether a radio can find and lock onto a 5G cell fast and reliably, under bad conditions.

- Studied GNU Radio signal pipelines and tested synchronization approaches against 3GPP standards
- Built and evaluated detection algorithms (PSS/SSS), then measured how accuracy held up as signal quality degraded
- Automated the analysis in Python — what used to take manual review now runs as a repeatable pipeline, **cutting analysis time by ~70%**

The part that stuck with me wasn't the radio engineering — it was the workflow: noisy real-world data, a hypothesis, and a way to test it rigorously. That's the same loop I want to run on ML problems now.

<br>

## What I work with

| | |
|---|---|
| **Languages** | Python · C/C++ · Java · TypeScript/JavaScript · SQL |
| **ML & Data** | NumPy · Pandas · SciPy · TensorFlow · EDA · Feature Engineering |
| **Backend & Web** | FastAPI · Node.js · React.js · PostgreSQL · Firebase |
| **Tools** | Git · Linux · GNU Radio · Jupyter |

<br>

## Education

**B.S. + M.S., Computer Science** — SRM University · 2024 – 2028

<br>

<br>

## Contribution Activity

<p align="center">
  <img src="https://ghchart.rshah.org/2e9ef7/CodeIsFire" alt="GitHub Contribution Chart" />
</p>

<p align="center">
  <img src="https://github-readme-activity-graph.vercel.app/graph?username=CodeIsFire&bg_color=ffffff&color=2e9ef7&line=2e9ef7&point=181717&area=true&hide_border=true" alt="Contribution Graph" />
</p>

<br>

## Elsewhere

[LinkedIn](https://www.linkedin.com/in/aadityabhardwaj9/) · [Email](mailto:abrohanaditya@gmail.com)

<sub>If something here overlaps with what you're working on, my inbox is open.</sub>
