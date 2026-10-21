# Sharing the experiment

The public artifact is the standalone report at
`docs/results/cx-sentiment-routing/index.html`. The file embeds its charts and
styles, so it can be hosted without the research dataset or provider responses.

## Recommended link

After the repository history has been scrubbed of the excluded corpus and the
repository is public, enable GitHub Pages with `main` and `/docs` as the publishing
source. The report URL will be:

```text
https://dosorio79.github.io/jev_test/results/cx-sentiment-routing/
```

Share that report URL as the primary link. Use the repository URL as a secondary
link for code and methodology. A normal GitHub file/blob link is less suitable
because it shows repository navigation or source instead of serving the report
as a webpage.

For LinkedIn, attach
`docs/assets/cx-sentiment-routing-linkedin-v2.png`, summarize the 0.973 macro F1,
74% median-latency reduction, 22% estimated cost reduction, and 7% final-test
fallback rate, then link to the hosted report. State that the study used synthetic
data and is an independent personal project without company affiliation or
financial interest.

Do not publish the ignored `data/`, `.cache/`, or `experiments/runs/` directories.
