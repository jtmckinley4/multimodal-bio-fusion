# Data and biological targets

The prediction tasks in this project connect biological sequences with outcomes such as RNA stability and protein production. Understanding a result starts with knowing what one dataset row represents, what was measured, and how that measurement became the model's prediction target.

Each section follows that path from the source experiment to the stored target, then explains how to interpret predictions and evaluation metrics. Several datasets name their target column `Value`, but its contents differ: a stability score, logged expression and a protein-production class describe different outcomes on different scales. GTEx and the full-transcript benchmarks use other target-column names and transformations.

| Dataset or reader question | Where to read |
| --- | --- |
| How do the targets relate to gene expression? | [Production, processing and loss](#connecting-the-targets-to-gene-expression) |
| mRNA stability: what does the stability score measure? | [Standardized decay profiles](#mrna-stability-standardized-decay-profiles) |
| mRFP: how did fluorescence become expression? | [GFPuv normalization and natural log](#mrfp-fluorescence-to-the-value-column) |
| CoV-Vac: what was degraded and under which conditions? | [Chemical-degradation profiles](#cov-vac-chemical-degradation-of-designed-rna) |
| Fungal expression and E. coli: what does expression mean here? | [RNA expression and protein-production classes](#fungal-and-e-coli-benchmark-targets) |
| GTEx: what are the 30 tissue labels? | [Tissue-averaged transcript abundance](#gtex-tissue-expression-labels) |
| Full transcripts: are half-life and TE direct physical quantities? | [Consensus stability, adjusted occupancy and ribosome load](#full-transcript-targets-half-life-and-translation) |

Each explanation distinguishes steps described by the source study, transformations verified against source rows, and remaining gaps. The [October 28 target trace](../run-records.md#2026-10-28-dataset-target-provenance) records the checked releases and numerical comparisons. The notebooks keep the shorter explanation needed beside their loading and results cells.

Our analyses also use [calculated targets and controls](sequences-to-predictions.md#calculated-targets-and-controls). Length, GC, GC3 and Nussinov pairing fraction come from sequence strings; the synonymous-recoding check tests whether the embedding pipeline behaves as expected. [From sequences to predictions](sequences-to-predictions.md) explains those calculations and how the supplied sequences and targets enter prediction and evaluation.

## Connecting the targets to gene expression

Transcription produces an RNA copy of a DNA region. For a protein-coding gene, a mature messenger RNA can then be translated repeatedly to produce protein chains. In eukaryotes, RNA processing includes steps such as capping, splicing and formation of the transcript's 3′ end before the mature RNA is exported from the nucleus. RNA and protein molecules are also removed over time. Cells can regulate several of these steps, so the path from a gene to its protein product includes production, processing and loss. [Essential Cell Biology, 6th ed., ch. 7, “From DNA to RNA” and “From RNA to Protein”](../sources.md#essential-cell-biology-6th-edition); [Molecular Biology of the Cell, 7th ed., ch. 6, pp. 337–339 and 348–349; ch. 7, p. 401](../sources.md#molecular-biology-of-the-cell-7th-edition).

An amount measured at one time is different from a production or degradation rate. RNA abundance reflects both how much RNA is made and how much is lost. Two samples with similar RNA abundance could therefore have different rates of RNA production and degradation. Protein abundance adds further dependencies on translation and protein turnover. The cell type and experimental conditions matter because the same sequence can encounter different regulatory molecules and activities. [Molecular Biology of the Cell, 7th ed., ch. 7, pp. 397–401 and 459–460](../sources.md#molecular-biology-of-the-cell-7th-edition).

The datasets below observe different parts of this process. The fungal and GTEx targets summarize RNA abundance; the stability assays follow RNA persistence; the translation-efficiency and ribosome-load targets concern ribosome occupancy; and the E. coli and mRFP targets derive from protein-gel scores and reporter fluorescence. CoV-Vac instead measures chemical degradation under a specified laboratory treatment. Their source-specific meanings and transformations remain in the sections below: the textbook background explains how these outcomes relate, while each source experiment establishes what its target actually measures.

## mRNA stability: standardized decay profiles

Stability concerns how long an mRNA persists before degradation. In [mRNA_Stability.csv](../../datasets/mRNA_Stability.csv), each row pairs a protein-coding nucleotide sequence in `Sequence` with a stability score in `Value`. A sequence can occur in more than one row, so each row represents a sequence-and-score observation, not necessarily a distinct gene.

The source measurements are grouped into **experimental profiles**, defined by species, cell type and measurement technique. The collection combines nine profiles spanning zebrafish and Xenopus embryos, human cell lines and mouse embryonic stem cells. Scores are standardized separately within these profiles, making a higher `Value` indicate greater stability relative to other RNAs in the same profile. The measurement and standardization steps below explain how that relative score is obtained. [Medina-Munoz et al., estimation of mRNA stability and data allocation](https://doi.org/10.1186/s13059-020-02251-5), [iCodon, training data](https://doi.org/10.1038/s41598-022-15526-7)

Biologically, degradation involves enzymes breaking down RNA. In the eukaryotic examples described in the textbook, shortening the poly-A tail and removing the 5′ cap can expose an mRNA to further degradation. Proteins that recognize particular RNA regions can alter these processes. This provides a reason to investigate sequence information, but the observed lifetime depends on the surrounding cellular machinery as well. These examples do not establish a single decay mechanism for every organism or profile in the benchmark. [Molecular Biology of the Cell, 7th ed., ch. 7, “Changes in mRNA Stability Can Control Gene Expression,” pp. 459–460](../sources.md#molecular-biology-of-the-cell-7th-edition).

### From RNA measurements to a decay slope

The source experiments estimate persistence by following RNA abundance over time. They use two approaches:

- **Transcription blocking** reduces the production of new RNA so that existing RNA can be followed as it degrades.
- **Metabolic labeling** marks newly made RNA so that labeled molecules can be distinguished and followed over time. SLAM-seq uses chemical conversion of a nucleotide label to make those molecules identifiable in sequencing.

Both approaches can estimate RNA persistence, but their cell contexts and measurement assumptions differ. The source studies describe how these measurements were collected and combined. [Medina-Munoz et al., estimation of mRNA stability](https://doi.org/10.1186/s13059-020-02251-5), [iCodon, training data](https://doi.org/10.1038/s41598-022-15526-7)

To connect the measurements to a stability score, first consider a simple exponential-decay model:

$$
N(t)=N_0 e^{\beta t}.
$$

**In this notation:**

- $t$ is elapsed time, and $N(t)$ is RNA abundance at that time.
- $N_0$ is the starting abundance, at $t=0$.
- $e$ is the base of the natural exponential function.
- $\beta$ controls the rate of change and is measured per unit of time. For decaying RNA, it is negative.

**Interpretation.** With a negative $\beta$, the amount of RNA decreases as time passes. A value closer to zero describes slower loss. The model describes the decline in abundance; it is not yet the standardized score stored in `Value`.

Taking the natural logarithm makes this relationship a straight line:

$$
\ln N(t)=\ln N_0+\beta t.
$$

Here $\ln$ denotes the natural logarithm. On this log-abundance scale, $\ln N_0$ is the starting value and $\beta$ is the slope: the change in log abundance per unit of time.

Half-life expresses the same model in terms of the time required for abundance to fall to half its starting value. Its relationship to the slope is:

$$
\beta=-\frac{\ln 2}{h}.
$$

**In this notation:**

- $h$ is the half-life, in the same time units as $t$.
- $\beta$ is the signed slope defined above. The usual positive decay constant is $k=-\beta$.

**Interpretation.** A longer half-life gives a slope closer to zero. This is why a larger, less negative slope indicates slower degradation.

**Illustrative example.** Half-lives of two and four hours correspond to slopes of about $-0.347$ and $-0.173$ per hour. These values illustrate the sign and scale; they are not rows from our file.

**Connection to the source code.** The [source gathering code](https://github.com/santiago1234/MZT-rna-stability/blob/8b886d9b05cd1f304439e4735268bedbaf2d007a/data/19-04-29-AllDecayProfiles/gatherdata.R) converts supplied half-lives with `log(1/2) / Half_life`. It also reverses the sign of the HeLa and RPE source rates so that the profiles use a consistent direction before standardization. These are source-preparation steps, not conversions repeated by our notebook.

### Standardization and the released Value

The source preprocessing removes outliers within each profile using a boxplot rule based on the interquartile range (IQR). It then subtracts that profile's mean slope from each retained slope and divides by the profile's sample standard deviation. This expresses each observation relative to the other retained observations in its own profile. [Source preprocessing, outlier removal and scaling](https://github.com/santiago1234/MZT-rna-stability/blob/8b886d9b05cd1f304439e4735268bedbaf2d007a/results/19-04-30-PredictiveModelDecayAllSpecies/19-04-30-EDA/EDAanalysos.Rmd)

$$
\mathrm{Value}_i=\frac{\beta_i-\mu_e}{s_e}.
$$

**In this notation:**

- $i$ identifies an observation, and $e$ identifies its source experimental profile. This subscript $e$ labels a profile; it is separate from the exponential base used above.
- $\beta_i$ is that observation's slope, with its sign adjusted where needed to follow the common direction described above.
- $\mu_e$ and $s_e$ are the mean and sample standard deviation of the slopes retained in profile $e$ after source filtering.
- $\mathrm{Value}_i$ is the resulting standardized stability score stored for that observation.

**Interpretation.** A score of $+1$ means one profile standard deviation above that profile's mean slope. The score has no decay-rate units, even though the source table still names the standardized field `decay_rate`. Standardization puts the profiles on comparable numerical scales; it does not remove their differences in cells, organisms or assays.

**Traced example.** The first local row matches source gene `ENSG00000148660` in the human HeLa endogenous profile. Its source standardized score is `0.44422743057356273`, compared with local `Value = 0.4442274305735627`. It is about 0.444 standard deviations above that profile's mean stability slope. It is not a 0.444-hour half-life, and exponentiating it does not recover hours.

**Verification against the released data.** All 65,356 local sequence/value observations match the combined iCodon [training](https://github.com/santiago1234/iCodon/blob/f3afe2bcf36a6bc0298ef7ec0dc42b68835bcb62/data/training.rda) and [testing](https://github.com/santiago1234/iCodon/blob/f3afe2bcf36a6bc0298ef7ec0dc42b68835bcb62/data/testing.rda) tables after treating T and U as equivalent sequence letters for matching. The comparison preserves how many times each observation occurs, including duplicates. The maximum numerical difference is below $4.45\times10^{-16}$.

The local file also matches the pinned CodonBERT release after line-ending normalization. These checks establish the relationship between source scores and local values; the exact later export and split-generation script was not recovered. The [October 28 target trace](../run-records.md#2026-10-28-dataset-target-provenance) records this verification.

### What the stability results mean

Our loaders retain the supplied `Value` scores without applying another logarithm or target standardization. Stage 1's ridge probes predict these standardized stability scores and evaluate the predictions with $R^2$. Their feature standardization acts on the embeddings used as predictor inputs; it does not standardize the stability targets again.

Stage 4's fusion models fit squared error on the same supplied target and report Spearman correlation, which measures how well predictions order the scores. These results concern prediction of relative stability. Neither metric establishes accuracy in hours or a percentage change in biological persistence. `Value` itself is neither a half-life in hours nor a percentage of RNA remaining, and it differs from the target used in the later [full-transcript half-life benchmark](#full-transcript-targets-half-life-and-translation).

The released four-column file omits the source profile identifiers. The same coding sequence can therefore occur with different scores from different cells or assays. Those differences cannot all be treated as noise from repeating the same measurement: the experimental context may have changed.

The [Stage 1 audit](../../Code/stage1/Stage1_stability.ipynb#Dataset-audit-for-published-comparisons) reports this variation and an approximate limit on what sequence alone can predict in these pooled observations. That estimate is not a universal limit on predicting RNA stability. Grouping identical sequences in Stage 1's evaluated folds keeps repeated strings together, but does not recover their missing experimental context or establish independence among related sequences.

## mRFP: fluorescence to the Value column

The mRFP target is the **natural logarithm of GFPuv-normalized mean red fluorescence**. It describes a reporter's production under the source experiment's conditions; it is not a protein molecule count, an RNA-folding measurement or a direct translation-rate measurement. Stage 1 uses these synonymous sequences to check representation behavior. Stage 4 fits predictors to the supplied target in the [mRFP control notebook](../../Code/stage4/tier1_mrfp_control.ipynb).

### From engineered bacteria to fluorescence measurements

A fluorescent reporter connects several steps: transcription produces its RNA, translation produces its protein chain, and folding and maturation produce a fluorescent form that the instrument can excite and detect. The textbook's GFP example illustrates the distinction between making a protein chain and forming its light-emitting group, or **chromophore**, after translation. This helps explain why a reporter signal can reflect multiple steps in gene expression; it does not identify which step accounts for a particular mRFP variant's value. The source study below establishes the actual reporter constructs and assay. [Molecular Biology of the Cell, 7th ed., ch. 9, “Individual Proteins Can Be Fluorescently Tagged in Living Cells and Organisms,” pp. 573–574, Fig. 9–16](../sources.md#molecular-biology-of-the-cell-7th-edition).

Nieuwkoop et al. expressed synonymous mRFP coding variants in *E. coli* DH10B, with a separately expressed GFPuv reporter as an internal reference. Flow cytometry measured red and green fluorescence in cell populations from individual clones. The instrument supplied excitation light and detected emitted light; these measurements are fluorescence signals, not counts of spontaneously emitted bioluminescent photons. The paper describes gating out GFPuv outliers and checking sequence quality, the expected protein sequence, cell populations and agreement with plate-reader measurements. [Original study, Materials and Methods](https://academic.oup.com/nar/article/51/5/2363/7016452)

The published `Supplementary Data.xlsx` separates 2,400 observations in **S1 Raw Data** from 1,459 retained observations in **S2 Validated Data**. Each retained row identifies a clone/well and its sequence. `mRFP Mean` and `GFPuv Mean` summarize cell fluorescence for that row; the workbook does not establish that these are averages of independent biological replicates. The four repeated sequence strings remain separate observations with their own measurements. [Supplementary workbook](https://pmc.ncbi.nlm.nih.gov/articles/instance/10018363/bin/gkad035_supplemental_files.zip)

### GFPuv normalization and the natural logarithm

The validated sheet rescales each red mean to the same green-reference level. Its literal Excel formulas are `F2 = AVERAGE(D:D)/D2` and `H2 = F2*C2`, repeated down the sheet. Column C is `mRFP Mean`, D is `GFPuv Mean`, F is `GFP normalization factor`, and H is `mRFP Mean Corrected`. The CodonBERT preparation notebook then reads this sheet and applies `np.log` to column H. Together, these operations give:

$$
\bar G = \frac{1}{n}\sum_{j=1}^{n}G_j,
\qquad
F_i = R_i\frac{\bar G}{G_i},
\qquad
y_i = \ln(F_i).
$$

**In this notation:**

- $i$ identifies one validated clone/well observation; $j$ ranges over the validated observations.
- $n=1{,}459$ is the number of rows in the source sheet, before benchmark splitting or our 1,000-row sampling.
- $R_i$ and $G_i$ are that row's mean red and green flow-cytometry signals.
- $\bar G$ is the mean green signal across the validated sheet, approximately $32{,}502.8060$ in the supplied instrument scale.
- $F_i$ is corrected red fluorescence, and $y_i$ is the released `Value` target.

**Why use GFPuv as the reference?** The mRFP coding sequence varies between constructs, while the separately expressed GFPuv construct stays the same on the same plasmid. Its green fluorescence can be measured separately from mRFP's red signal. The study used weak GFPuv expression to limit interference with mRFP production while retaining a detectable internal reference. [Original study, Plasmid preparation and library generation](https://academic.oup.com/nar/article/51/5/2363/7016452)

A brighter red signal could reflect the mRFP variant or a broader difference in the expression system. For example, more plasmid copies could increase both reporters. The correction can reduce such variation **if it affects red and green proportionally**. For a shared multiplicative factor $k$, the ratio cancels that factor: $(kR_i)/(kG_i)=R_i/G_i$. Multiplication by the fixed dataset mean $\bar G$ then expresses red fluorescence at a common green-reference level. A green mean above that reference reduces the corrected red signal; one below it increases the corrected signal.

**Illustrative normalization example.** Suppose a hypothetical dataset has a mean green signal of $\bar G=100$. The three observations below illustrate that reference; they are not the entire hypothetical dataset or measured rows from our CSV.

| Observation | Red mean $R$ | Green mean $G$ | Corrected red $F=R\times100/G$ |
| --- | ---: | ---: | ---: |
| A | 100 | 50 | 200 |
| B | 200 | 100 | 200 |
| C | 200 | 50 | 400 |

B has twice A's red signal and twice its green signal, so their corrected red values agree. C has twice A's red signal without a corresponding green increase, so its corrected red remains twice as large. The adjustment does not identify or remove every biological or measurement difference. It relies on an appropriate reference: effects specific to one reporter do not cancel, and noise in the green measurement enters the ratio. If an mRFP variant itself changes GFPuv production through competition for cellular resources, normalization could also distort a meaningful biological difference. The weak-reference design aims to limit interference; it does not prove that the reference is unaffected in every sample.

**Why take the logarithm?** Once fluorescence has been corrected, the natural log changes the scale on which differences are represented. For two positive corrected signals, $F_1$ and $F_2$:

$$
\ln(F_2)-\ln(F_1)=\ln\left(\frac{F_2}{F_1}\right).
$$

An equal fold change therefore becomes an equal additive difference. Increasing corrected fluorescence from 100 to 200, or from 10,000 to 20,000, adds the same $\ln(2)\approx0.693$ to `Value`. In squared-error fitting, predicting twice the observed corrected fluorescence consequently produces the same log-scale error at either brightness. On the raw scale, those errors would be 100 and 10,000. Logging reduces the dominance of large absolute signals and makes errors depend on proportional discrepancies.

This mathematical explanation does not establish why the original benchmark authors chose the transformation. CodonBERT applies `np.log`, not `log2`, `log10` or `log1p`, and adds no pseudocount in this step; its inspected code does not state the rationale. Natural log and log base 2 express the same fold relationships in different units: a doubling is $\ln(2)$ in natural-log units and 1 in log-base-2 units. The base has no special biological meaning here. [CodonBERT preparation notebook at commit `451a1b1`](https://github.com/Sanofi-Public/CodonBERT/blob/451a1b167c06028dfbf2ff7aa2cfdea46fbcc4f4/benchmarks/notebooks/0_download_data.ipynb)

Logging can make multiplicative variation easier to model because products become sums. It can reduce right skew or unequal error variance, but those improvements are not guaranteed or established by the transformation alone. It does not remove biological confounding or turn fluorescence into an absolute protein quantity.

**Traced example.** The first data row of our CSV matches validated-sheet row 137, sample `THNI0088 crRFP-M`, well `03E01`. Its red mean is 19,179 and green mean is 24,002. The factor is approximately $32{,}502.8060/24{,}002=1.3541707371$, giving corrected fluorescence $25{,}971.6405666$. Its natural log is $10.1647604743$, matching the stored `Value` of `10.164760474265888` within floating-point precision. Exponentiating `Value` recovers this corrected fluorescence scale, not a protein count.

**The separate plate-reader check.** S1's plate-reader values are already corrected for optical density at 600 nm, a measure used to account for culture density. The workbook compares plate red fluorescence with flow-cytometry red fluorescence and uses a 25% deviation rule for quality assurance. Those OD-corrected plate readings are not the values passed into the S2 normalization formula or CodonBERT's logarithm. This is a separate measurement check, not an extra division by OD600 in `Value`. [Original study, Measurements and sequencing and Data validation](https://academic.oup.com/nar/article/51/5/2363/7016452)

### What our notebook predicts and scores

The [local CSV](../../datasets/mRFP_Expression.csv) has the same parsed rows, values, ordering and split labels as the [pinned CodonBERT release](https://github.com/Sanofi-Public/CodonBERT/blob/451a1b167c06028dfbf2ff7aa2cfdea46fbcc4f4/benchmarks/CodonBERT/data/fine-tune/mRFP_Expression.csv); only line endings differ. Matching source sequences after T/U normalization reproduces all 1,459 target values as $\ln(F_i)$, preserving repeated observations. The [October 28 verification record](../run-records.md#2026-10-28-mrfp-target-provenance) identifies the checked bytes and numerical tolerances. The preparation notebook writes an intermediate `dataset_mRFP.csv`; the exact later script that assigned the released row order and `Split` labels was not located. Numeric and sequence agreement establishes the target mapping, without inventing that missing procedural history.

The project loader reads `Value` without another log or target normalization. Stage 4 casts those labels to floating point, samples the established 1,000 rows and evaluates them on grouped folds. `StandardScaler` acts on predictor input features, not on these labels. The notebook does not exponentiate predictions before scoring. Its held-out $R^2$ and incremental $\Delta R^2$ therefore concern prediction error on the **log corrected-fluorescence scale**; they do not measure a percentage increase in protein production. The residual predictor also learns errors on this same log scale. See [linear probing](sequences-to-predictions.md#linear-probing) for the fitting and evaluation procedure.

Fluorescence remains a proxy for protein abundance: the source study's ten-variant proteomics check supports that relationship, but does not make each dataset value an absolute abundance measurement. The measurements also combine effects along the path from coding sequence to reporter production; predictive success alone does not isolate transcription, RNA stability, translation or protein maturation. [Original study, LC–MS/MS comparison](https://academic.oup.com/nar/article/51/5/2363/7016452)

## CoV-Vac: chemical degradation of designed RNA

The CoV-Vac target describes **chemical cleavage of short designed RNAs under an accelerated laboratory condition**. Researchers synthesized RNA from DNA templates, exposed it to degradation conditions, then reverse-transcribed and sequenced the resulting fragments to locate cleavage along each RNA. This In-line-seq measurement is distinct from cellular mRNA half-life or vaccine efficacy. [Wayment-Steele et al., Results and Experimental data generation](https://doi.org/10.1038/s42256-022-00571-8)

The selected measurement, `deg_Mg_50C`, comes from **10 mM magnesium chloride, PBS at pH 7.2, 50 °C, for one day**. The released profiles already incorporate background subtraction, signal correction and profile normalization; they are not raw sequencing counts or percentages of RNA lost. [Original measurement record, RYOS1_MG50_0000](https://rmdb.stanford.edu/detail/RYOS1_MG50_0000/)

CodonBERT describes averaging this measurement across nucleotides to obtain one target per sequence. It reports selecting this condition because it correlated most strongly with the other degradation conditions. [CodonBERT dataset description](https://github.com/Sanofi-Public/CodonBERT/blob/451a1b167c06028dfbf2ff7aa2cfdea46fbcc4f4/README.md)

Comparing the original measurement file with our [released CSV](../../datasets/CoV_Vaccine_Degradation.csv) establishes the remaining transformation:

$$
\begin{aligned}
a_i &= \frac{1}{68}\sum_{j=1}^{68}d_{ij},
&\mu &= \frac{1}{2400}\sum_{i=1}^{2400}a_i, \\
\sigma &= \sqrt{\frac{1}{2400}\sum_{i=1}^{2400}(a_i-\mu)^2},
&\mathrm{Value}_i &= \frac{a_i-\mu}{\sigma}.
\end{aligned}
$$

**In this notation:**

- $d_{ij}$ is the normalized degradation measurement at measured position $j$ of RNA $i$.
- $a_i$ averages that RNA's 68 measured positions into one number.
- $\mu$ and $\sigma$ are the mean and population standard deviation of those averages across all 2,400 released rows, before their training/validation/test split is used.

**Why average and standardize?** Averaging turns a cleavage profile into a sequence-level prediction task, but discards where cleavage occurs. Standardization expresses the averages in dataset standard-deviation units: `Value = 1` means one standard deviation above the dataset mean, indicating greater degradation under this condition. It preserves ordering and does not account for assay confounding. There is no logarithm in this mapping. The exact generation script and historical reason for standardizing were not located; the formula is numerically verified, rather than attributed to an inspected script.

**Traced example.** The first local row matches source construct `9868607`. Its 68 measurements sum to 36.6207, giving $a_i=0.5385397059$. With $\mu=0.4070300196$ and $\sigma=0.1441448346$, `(0.5385397059 - 0.4070300196) / 0.1441448346` gives `Value = 0.9123440785`. The formula reproduces all 2,400 targets within $5\times10^{-10}$, consistent with released decimal rounding.

**Input and target cover different windows.** Every local input is the original 107-nucleotide sequence cropped to Python slice `[5:86]`, retaining 81 nucleotides. The target averages measurements from the original first 68 positions; it is not an average of 81 measurements. This crop removes constant flanks but does not establish a protein-coding sequence or biological reading frame.

Our loader uses `Value` directly. Fusion training minimizes mean squared error on these standardized labels, while reported **Spearman correlation measures their predicted ordering**. Higher correlation means better ranking of this assay's degradation measurements; it does not establish accurate hours of persistence, protein yield or vaccine performance. Standardization itself leaves Spearman ranks unchanged. See the [shared loader](../../Code/shared_code/fusion/blf.py) and [training and scoring functions](../../Code/shared_code/fusion/train.py).

## Fungal and E. coli benchmark targets

Training fits the supplied `Value` column. In both files, a row is a sequence-label observation with a supplied split assignment; it is not automatically a distinct gene, organism or independent experimental replicate. Their four fields, `Sequence`, `Value`, `Dataset` and `Split`, omit accession, organism and assay identifiers. Source matching now resolves the target transformations below, while some earlier assay metadata remain unknown.

[BioLangFusion, Appendix A.2 and Table 2](../../Papers/core/BioLangFusion.pdf), obtains these benchmarks through CodonBERT. The [CodonBERT release](../sources.md#codonbert-repository) identifies Wint et al. for fungal expression and Ding et al.'s MPEPE study for E. coli. The numerical matches below connect those references to the local observations.

### Fungal expression

This target concerns RNA expression associated with a coding sequence. The cited study obtained expression evidence through RNA sequencing: RNA was collected, converted into sequencing libraries, and sequenced to estimate which transcripts were more abundant. Most transcriptomes in the wider study pooled material across conditions. RNA abundance reflects both RNA production and removal; it is not a direct count of proteins produced or a measurement of translation efficiency. [Wint et al., RNA sequencing](https://pmc.ncbi.nlm.nih.gov/articles/PMC8826637/)

The downstream numerical step is now traceable. All 7,089 local observations match a subset of the author's [example expression table](https://github.com/rhondene/Codon2Vec/blob/1d3e5df41425753502076927b3bf2f0aa9e927a5/test_input/test_exprs.csv) and [coding-sequence file](https://github.com/rhondene/Codon2Vec/blob/1d3e5df41425753502076927b3bf2f0aa9e927a5/test_input/test_CDS.fasta), with:

$$
\mathrm{Value}_i = \ln(\mathrm{Exprs}_i).
$$

Here, $\mathrm{Exprs}_i$ is the positive expression quantity supplied for source sequence $i$; $\mathrm{Value}_i$ is the label used by our models. Matching preserves repeated observations and allows for rounding in the release. For example, source sequence ID `1929` has `Exprs = 0.006815951`; its natural logarithm is $-4.988489678448099$, exactly the first local row's `Value`. A negative log value means the source quantity lies between zero and one; it does not mean negative RNA abundance.

**The earlier normalization remains unresolved.** The example table calls its quantity `Exprs` without defining the original RNA-read normalization, units or reference denominator. Its organism and assay metadata are also insufficient to reconstruct the full source chain. Do not call these values TPM or FPKM, or describe this example-derived benchmark as a verified collection spanning all 459 species in the wider paper. The author's separate high/low Codon2Vec classification procedure does not define this continuous regression task.

Our loader applies no further target transformation. Fusion models minimize squared error on the supplied log labels, making equal fold discrepancies comparable across expression levels. Reported Spearman correlation instead evaluates ordering: applying a strictly increasing log preserves target ranks, although fitting on logs changes the loss. A strong score therefore supports ranking these released RNA-expression quantities under the evaluated split, without establishing calibrated abundance, protein yield or causality.

### Protein expression in E. coli

This target measures coarse recombinant protein production in an *E. coli* host. Boël et al. expressed genes from 171 organisms using a T7/pET system in *E. coli* BL21(DE3), with the auxiliary pMGK plasmid. Researchers inspected whole-cell proteins separated on Coomassie-stained SDS-PAGE gels after overnight induction at 18 °C. The original score runs from 0, where the target protein was undetectable against the cellular-protein background, to 5, where it was the most abundant cellular protein. Two transformants were scored; when replicate scores differed, the maximum was retained because the authors judged most experimental errors to lower the apparent expression. This is not an average protein concentration or a measurement restricted to the soluble fraction. [Boël et al., Scoring of protein expression in the large-scale dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC5054687/)

The local class dictionary is verified by matching all 6,348 sequences against the [MPEPE sequence-and-score file](https://github.com/BRITian/MPEPE/blob/fab98420306f1f1918b1b375a44b7086849ec45f/Dataset/exp_nature_gene.faa):

| Original gel score | Local `Value` | Meaning |
|---|---:|---|
| 0, 1 or 2 | 0 | Low expression |
| 3 or 4 | 1 | Medium expression |
| 5 | 2 | High expression |

For example, construct `AaR17` has original score 0 and local `Value = 0`; `AaR5` has original score 5 and local `Value = 2`. This step groups and relabels ordinal scores. It does not take a logarithm or normalize a concentration.

Ding et al. numbers the original groups 1–6 and uses the low/high extremes for its binary training task, reserving the middle groups for validation. CodonBERT's three-class task is a different evaluation design. [MPEPE dataset documentation](https://github.com/BRITian/MPEPE/blob/fab98420306f1f1918b1b375a44b7086849ec45f/Dataset/Readme)

Our classification models use categorical cross-entropy, and accuracy reports the fraction assigned to the correct group. `Value = 2` does not mean twice the production of `Value = 1`; accuracy also counts a low-to-medium error and a low-to-high error equally. Better accuracy supports distinguishing these production categories under this host and assay context, without establishing absolute yield, soluble yield, native abundance or a causal benefit from editing a sequence.

### Splits, preprocessing and the claim being tested

Both local CSVs match the pinned CodonBERT release in every field and row, apart from file line endings. The released fungal splits contain 5,089 training, 1,000 validation and 1,000 test observations; E. coli contains 4,348, 1,000 and 1,000. These actual assignments differ from the paper's general 70%/15%/15% description. Their generation script was not recovered, and the stored split labels do not establish species-disjoint or gene-family-disjoint evaluation. The [published specification and release loader](../sources.md#codonbert-repository) are separate evidence: the inspected loader consumes the stored assignments.

Our [`blf_benchmark_rows`](../../Code/shared_code/datasets.py) keeps those assignments, removes missing sequences and drops sequences longer than 999 nucleotides. This is a project filtering choice. BioLangFusion describes truncation; the [saved preprocessing audit](../../Code/stage3/01_blf_datasets.ipynb#Recovering-BioLangFusion's-preprocessing) finds matching retained totals at several nearby cutoffs and a fungal raw-count discrepancy. Neither matching counts nor the helper's existing “published preprocessing” wording proves identical author preprocessing or row identity.

The [saved split audit](../../Code/stage3/01_blf_datasets.ipynb#Published-splits,-and-how-much-the-test-split-reuses) finds no exact normalized sequence crossing splits in fungal expression or E. coli. Fungal expression nevertheless contains six repeated-sequence groups with differing labels within splits. Relatedness, organism membership and assay dependence have not been established for the local rows. Report the exact-overlap finding rather than calling the splits “clean” or biologically independent.

For the protein stream, [`grid_protein`](../../Code/shared_code/fusion/token_embeddings.py) translates complete triplets from the first supplied base and represents stops with `X`. Preserving one position per triplet does not verify biological reading frame, the original construct's boundaries, or the authors' protein-input policy. The [sequence checks](../../Code/stage3/01_blf_datasets.ipynb#Sequence-checks) explain that implementation choice without changing the supplied benchmark.

The resulting comparisons test prediction on the retained benchmark rows under the declared input and evaluation rules. Keep missing source identities and biological qualifications beside the claims. This source review changes the interpretation boundary, not the labels, splits, saved metrics or completed fits.

## GTEx tissue expression labels

GTEx has 30 tissue-named prediction columns rather than one `Value`. Each describes the abundance of a particular transcript isoform in a tissue, derived from bulk RNA sequencing. RNA abundance reflects both production and removal; it is not a direct measurement of transcription rate, RNA half-life, or protein production.

IsoFormer's release starts from GTEx v8 RSEM transcript abundances in transcripts per million (TPM), averages measurements within each tissue, and describes the resulting labels as `log(x + 1)`. TPM expresses relative transcript abundance after accounting for transcript length and sample sequencing composition. Tissue averaging supplies a common target for a reference sequence and suppresses differences between donors. Each bulk-tissue measurement also combines the cell populations present. [IsoFormer dataset description](https://huggingface.co/datasets/InstaDeepAI/multi_omics_transcript_expression/blob/330ab1795abca683e692c7ee138a2e89c0c882f1/README.md)

**Why the log matters.** Adding one permits zero abundance to become `log(1) = 0`; logging compresses large values. For an illustrative tissue-average TPM of 0, 9, and 99, the stored-scale calculation is respectively `log(1)`, `log(10)`, and `log(100)`. All three transformed values are equally spaced on any fixed logarithm base, despite the raw TPM gaps of 9 and 90. Near zero, the added one matters: log differences concern ratios of `TPM + 1`, not exact ratios of TPM. This explains the transformation's mathematics; the inspected release description does not identify the log base.

**What our files preserve.** The pilot builder selects rows and copies the 30 released tissue columns unchanged. The upstream loader also returns those values unchanged, alongside separate normalization statistics. Although the [IsoFormer paper's training appendix](https://papers.nips.cc/paper_files/paper/2024/file/8f6b3692297e49e5d5c91ba00281379c-Paper-Conference.pdf) describes per-tissue normalization, our pilot builder and label loaders do not apply that additional step. The original raw-measurement-to-release transformation has not been numerically reconstructed.

Stage 1 evaluates prediction of these stored labels with $R^2$ averaged across tissues. Its scalar `mean_expression` averages already transformed tissue values, not raw TPM. The Stage 4 IsoFormer-style comparison averages Spearman correlations calculated across transcripts separately within each tissue. Neither score means a percentage increase in expression; they describe prediction quality on the chosen targets.

## Full-transcript targets: half-life and translation

The mRNABench tables use `target` or explicitly named target columns. They are separate sources from the coding-sequence `mRNA_Stability.csv`; sharing the word “mRNA” does not make their numerical scales interchangeable.

### Consensus half-life scores

The human and mouse half-life sources combine experiments that follow RNA abundance over time after transcription is blocked or newly made RNA is labeled. A simple decay model is

$$
N(t)=N_0e^{-kt}, \qquad t_{1/2}=\frac{\ln 2}{k}.
$$

Here $N(t)$ is remaining RNA, $k$ its decay constant, and $t_{1/2}$ the half-life. An illustrative decrease from 100 to 50 units in four hours indicates a four-hour half-life under this model. **Our stored target is not that four-hour measurement.**

Saluki combines studies into a consensus score: source measurements undergo log transformation and source-specific handling, duplicate gene IDs are averaged, samples are standardized, missing entries are imputed, and distributions are quantile-normalized. PC1, the first principal-component score, summarizes the dominant shared variation across 39 human or 27 mouse datasets. Standardization prevents numerical scale alone from dominating the combination; quantile normalization aligns distributions, while imputation estimates missing measurements. These operations improve comparability while changing the target from a time measurement into a relative summary. [Saluki Methods](https://doi.org/10.1186/s13059-022-02811-x)

The [pinned preprocessing code](https://github.com/vagarwal87/saluki_paper/blob/3aa4e56a19bbbf87ac9f5f0a251098cf749ad6bc/Fig1_S1_S2/master_comparison_human.R) generally uses `log10(half-life + 0.1)` for hour-valued sources and `log10(half-life + 1)` for minute-valued sources, with additional handling for decay-rate sources. These are upstream operations, not transformations repeated by our notebook.

mRNABench copies labels from an [Orthrus intermediate dataset](https://zenodo.org/records/14708163). All 12,939 local human labels exactly match that intermediate's retained `y` values. They occupy an approximately standardized scale: for example, M6PR has `target = 0.689453125`. The exact earlier centering/scaling cohort was not recovered. Neither this value nor its exponential is a half-life in hours; a negative score does not mean negative RNA survival.

### Translation efficiency as an abundance-adjusted occupancy score

Ribosome profiling measures protected RNA fragments associated with ribosomes, while matched RNA-seq measures transcript abundance. More ribosome-associated fragments can result simply from having more RNA. Accounting for RNA abundance helps distinguish that situation from greater ribosome occupancy per RNA.

For intuition, doubling both the RNA signal and the ribosome-associated signal is not by itself evidence of increased occupancy per RNA. **The actual RiboNN target uses a regression adjustment, not this simple ratio:** after quality filtering, count processing, and centered-log-ratio normalization, the study defines TE as a compositional-regression residual. A residual measures how much ribosome occupancy lies above or below its fitted relationship with RNA abundance. Results are averaged within cell types; the supplied `mean_te` summarizes cell types. It is not a direct measurement of protein synthesis speed. [RiboNN Methods and interpretation](https://doi.org/10.1038/s41587-025-02712-x)

The [mRNABench constructor](https://github.com/morrislab/mRNABench/blob/778d691ab91c51c03addeb9219229f2ec81d195c/mrna_bench/datasets/translation_efficiency_human.py) copies `mean_te` into `target` as float32, selecting the transcript with highest `mean_te` per gene. All 10,281 local human targets exactly match the study supplement after that cast. **Traced examples:** TSPAN6 has a stored target of approximately `0.974014`, while SCYL3 has approximately `-0.328011`. A negative mean residual places occupancy below the fitted abundance relationship on average across the included cell types; it does not mean negative translation. Our notebook applies no further log. Raw-read processing and zero-count handling were not independently reconstructed.

### Other screened targets and interpretation

Sugimoto's HP5 assay separates polysome fractions and sequences transcript 5′ ends. External RNA standards normalize fraction abundances; mean ribosome load is a weighted mean:

$$
\mathrm{MRL}=\frac{\sum_{i=1}^{8}i\,c_i}{\sum_{i=1}^{8}c_i},
$$

where $c_i$ is normalized abundance in the fraction assigned $i$ ribosomes. This estimates ribosome loading, not RNA persistence or completed proteins per second. The exact original-assay-to-Orthrus row mapping remains unverified. [HP5 Methods](https://doi.org/10.1038/s41594-022-00819-2)

The synthetic PERSIST-seq table separately provides in-cell half-life and ribosome load. It is screened but excluded from the retained comparisons by the notebook's row-count criterion. Its precise conversion into the released parquet remains unresolved. [PERSIST-seq](https://doi.org/10.1038/s41467-022-28776-w)

The primary encoder comparisons use the human consensus-half-life and TE targets directly. Ridge probes scale predictor features, not these labels. Their headline metric is mean test Pearson correlation: a gain of 0.05 means improved correlation with the supplied score, not 5% longer half-life or greater protein output.
