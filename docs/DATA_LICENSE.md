# Dataset provenance and redistribution status

The [MIT license](../LICENSE) covers project code and project-authored
documentation. It does **not** grant rights to the source corpus, bundled data
archives, extracted samples, or third-party material. Redistribution permission
for the dataset has **not been established**. Public release or redistribution
of these files needs a verified license or explicit permission from the rights
holder; their presence in this checkout is not evidence of permission.

## Evidence available in this checkout

Reviewed on 2026-10-18:

| Material | Recorded provenance |
| --- | --- |
| `data/raw/financial_sentiment_hf_v2.zip` | Its embedded `README.md` calls it “Financial customer feedback synthetic dataset — revised v2”. It describes six languages, 108 seed sentences, templated variations, and globally exact-deduplicated train/validation/test splits. It names no author, dataset URL, or license. |
| `data/raw/cx_sentiment_benchmark_starter.zip` | Its embedded README describes the locked test sampling and starter scripts. Its manifest identifies the source archive and checksum, but supplies no dataset license or original author. |
| `data/cx_sentiment_test_600.csv` and `.jsonl` | Locked 600-record test sample derived from the archive's test split; provenance is recorded in [`sample_manifest.json`](../data/sample_manifest.json). |
| Generated `data/validation_600.csv` | A deterministic sample of the archive's validation split. The generated manifest records selected source indices and checksum. These files are ignored by Git. |

The recorded source archive SHA256 is:

```text
d7521f24fdeb01b865e5885a2815e6ee6920fc13efd508e6ea472bd8b8e831a3
```

No authoritative upstream dataset page or rights holder is identified by these
materials. The substring `hf` in a filename does not establish a Hugging Face
origin or any particular license. We cannot verify an upstream license against
an unidentified source and do not invent one. Synthetic generation also does
not establish a right to license or redistribute seed sentences.

## Release requirement

Obtain the original dataset URL and author/rights-holder attribution, preserve
the applicable license text/version, and verify that it permits redistribution
of both the source archives and derived samples. Record any attribution,
noncommercial, or share-alike conditions before distribution. If permission
cannot be established, a separate release decision must replace or exclude the
data; this notice alone does not resolve redistribution rights. No existing
locked test data was changed during the documentation update.

Report artifacts can also contain source text. A project's code license does
not resolve rights to text copied into predictions, disagreements, caches, or
HTML reports. Apply the source's verified terms before publishing such artifacts.

## Recommended sharing path

Do **not** upload the bundled archives, CSV/JSONL samples, predictions containing
the comments, or reports that reproduce comments until the source and license
are verified. A private repository does not resolve this rights question.

For the current project, share these safer artifacts instead:

- the code and experiment configuration under the repository's MIT license;
- the aggregate final report and infographic, which contain metrics but no
  source comments;
- the dataset schema, sampling method, record counts, and SHA256 fingerprints;
  and
- a note explaining that the corpus is available only after provenance and
  redistribution rights are established.

If the rights holder and license are later confirmed, publish the data as a
versioned dataset rather than as an undocumented repository attachment:

1. Add a dataset card with creator/source attribution, license, collection or
   generation method, six languages, label mapping, splits, intended uses,
   limitations, template leakage, and privacy review.
2. Include CSV or Parquet data, a machine-readable schema, checksums, and the
   sampling manifest. Exclude credentials, caches, raw provider responses, and
   model predictions unless they have their own documented release rationale.
3. Use a Hugging Face dataset repository for discovery and interactive access,
   and archive the same immutable release on Zenodo for a citable DOI. Link the
   exact Zenodo version from the dataset card and repository.
4. Keep future real customer feedback private unless it has been lawfully
   collected, de-identified, privacy-reviewed, and explicitly cleared for the
   chosen release terms.

Hugging Face documents dataset cards and their license/language metadata in its
[Dataset Cards guide](https://huggingface.co/docs/hub/datasets-cards). Zenodo
requires a license for a deposit, assigns a DOI when it is published, and links
new file versions while preserving version-specific identifiers; see its
[license](https://help.zenodo.org/docs/deposit/describe-records/licenses/) and
[versioning](https://help.zenodo.org/docs/deposit/manage-versions/) guidance.

For the synthetic-data and template-leakage limitations, see
[METHODS.md](METHODS.md). Provenance checksums establish identity, not licensing.
