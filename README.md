# UNGA Climate-Related Discourse Analysis

This repository contains analysis scripts, statistical outputs, and supplementary material used in a diachronic analysis of climate-related discourse in United Nations General Assembly (UNGA) speeches.

## Repository contents

- `Code for diachronic analysis(2).py`  
  Python script for the diachronic BERTopic analysis of climate-related UNGA sentences and VADER sentiment analysis across diachronic zones.

- `continental distribution and Chi analysis(2).py`  
  Python script for visualising the continental distribution of climate-related sentences and conducting the Pearson chi-square analysis, including Cramér's V, expected frequencies, residual analyses, and cell contributions.

- `UNGA_spacy_sentence_counts_by_continent(1).xlsx`  
  Spreadsheet containing sentence-count information used for the continental distribution analysis.

- `Chi_Square_Continent_by_Diachronic_Zone(1).xlsx`  
  Spreadsheet containing output from the continent-by-diachronic-zone chi-square analysis.

- `Supplementary Table.docx`  
  Supplementary table reporting residual analysis for continental climate attention.

## Analyses

The repository contains code for the following components of the study:

1. BERTopic modelling of climate-related UNGA sentences across diachronic zones.
2. Hierarchical topic analysis and topic visualisation.
3. VADER sentiment analysis across diachronic periods.
4. Continental distribution visualisation of climate-related sentences.
5. Pearson chi-square analysis of the association between continent and diachronic zone.
6. Cramér's V effect-size estimation.
7. Standardised and adjusted standardised residual analyses.

## Software requirements

The analyses were conducted in Python. Required Python packages are listed in `requirements.txt`.

Install them with:

```bash
pip install -r requirements.txt
```

## Important note on file paths

The Python scripts are provided exactly as used for the analysis and have not been modified for this repository. They therefore contain local file paths from the original analysis environment. To rerun the analyses on another computer, users will need to adjust those local file paths to match the location of the corresponding data and output folders on their own system.

## Reproducibility note

The files in this repository are preserved in their original uploaded form. No analysis code, spreadsheet, or supplementary document has been altered, merged, or renamed as part of repository preparation.

## Citation

If you use these materials, please cite the associated publication or manuscript. Citation details can be added here once the final bibliographic information is available.
