# Model Comparison Report: 8B Original vs AWQ Quantized (All Languages)

## Model Information

| Property | Original (FP16) |
|----------|-----------------|
| **Model Name** | TomoroAI/tomoro-colqwen3-embed-8b |
| **Parameters** | 8.0B |
| **Memory Usage** | 16.7 GB |
| **Release Date** | 2025-11-26 |

## Quantization Configuration

All quantized models use **AutoRound with AutoAWQ** backend, calibrated with **NeelNanda/pile-10k** dataset.

| Property | seqlen=256 | seqlen=512 | seqlen=1024 |
|----------|------------|------------|-------------|
| **bits** | 4 | 4 | 4 |
| **group_size** | 128 | 128 | 128 |
| **sym** | True | True | True |
| **iters** | 1000 | 1000 | 1000 |
| **nsamples** | 300 | 300 | 500 |
| **batch_size** | 300 | 150 | 50 |
| **quant_method** | awq | awq | awq |
| **provider** | auto-round | auto-round | auto-round |

**Quantized Memory Usage:** ~7.9 GB

## NDCG@5 Performance Comparison

| Benchmark | Original | seqlen=256 | Δ% | seqlen=512 | Δ% | seqlen=1024 | Δ% |
|-----------|----------|------------|-----|------------|-----|------------|-----|
| Vidore2BioMedicalLecturesRetrieval [English] | 0.67838 | 0.68345 | +0.75% | 0.66814 | -1.51% | 0.67505 | -0.49% |
| Vidore2BioMedicalLecturesRetrieval [French] | 0.64321 | 0.65490 | +1.82% | 0.62812 | -2.35% | 0.64791 | +0.73% |
| Vidore2BioMedicalLecturesRetrieval [German] | 0.64826 | 0.65775 | +1.46% | 0.65091 | +0.41% | 0.65432 | +0.93% |
| Vidore2BioMedicalLecturesRetrieval [Spanish] | 0.64878 | 0.64974 | +0.15% | 0.64890 | +0.02% | 0.64811 | -0.10% |
| Vidore2ESGReportsHLRetrieval [English] | 0.75981 | 0.73684 | -3.02% | 0.75315 | -0.88% | 0.75673 | -0.41% |
| Vidore2ESGReportsRetrieval [English] | 0.65488 | 0.62584 | -4.43% | 0.63820 | -2.55% | 0.62255 | -4.94% |
| Vidore2ESGReportsRetrieval [French] | 0.59901 | 0.59570 | -0.55% | 0.61089 | +1.98% | 0.61192 | +2.16% |
| Vidore2ESGReportsRetrieval [German] | 0.58307 | 0.59137 | +1.42% | 0.60317 | +3.45% | 0.58684 | +0.65% |
| Vidore2ESGReportsRetrieval [Spanish] | 0.59135 | 0.60232 | +1.86% | 0.64206 | +8.58% | 0.62583 | +5.83% |
| Vidore2EconomicsReportsRetrieval [English] | 0.61587 | 0.62106 | +0.84% | 0.59014 | -4.18% | 0.62067 | +0.78% |
| Vidore2EconomicsReportsRetrieval [French] | 0.57608 | 0.55985 | -2.82% | 0.52960 | -8.07% | 0.57190 | -0.73% |
| Vidore2EconomicsReportsRetrieval [German] | 0.57701 | 0.57367 | -0.58% | 0.55906 | -3.11% | 0.58266 | +0.98% |
| Vidore2EconomicsReportsRetrieval [Spanish] | 0.60940 | 0.59782 | -1.90% | 0.55938 | -8.21% | 0.59634 | -2.14% |
| Vidore3ComputerScienceRetrieval [English] | 0.74431 | 0.75495 | +1.43% | 0.74155 | -0.37% | 0.75338 | +1.22% |
| Vidore3ComputerScienceRetrieval [French] | 0.71210 | 0.71159 | -0.07% | 0.71239 | +0.04% | 0.71694 | +0.68% |
| Vidore3ComputerScienceRetrieval [German] | 0.73190 | 0.72988 | -0.28% | 0.72050 | -1.56% | 0.73353 | +0.22% |
| Vidore3ComputerScienceRetrieval [Italian] | 0.70937 | 0.70742 | -0.27% | 0.72526 | +2.24% | 0.70730 | -0.29% |
| Vidore3ComputerScienceRetrieval [Portuguese] | 0.72496 | 0.72213 | -0.39% | 0.73159 | +0.91% | 0.72342 | -0.21% |
| Vidore3ComputerScienceRetrieval [Spanish] | 0.71888 | 0.71846 | -0.06% | 0.72124 | +0.33% | 0.72312 | +0.59% |
| Vidore3EnergyRetrieval [English] | 0.64907 | 0.64319 | -0.91% | 0.61018 | -5.99% | 0.64242 | -1.02% |
| Vidore3EnergyRetrieval [French] | 0.66292 | 0.66767 | +0.72% | 0.65802 | -0.74% | 0.66630 | +0.51% |
| Vidore3EnergyRetrieval [German] | 0.65172 | 0.65246 | +0.11% | 0.62375 | -4.29% | 0.65419 | +0.38% |
| Vidore3EnergyRetrieval [Italian] | 0.66164 | 0.66548 | +0.58% | 0.64221 | -2.94% | 0.65722 | -0.67% |
| Vidore3EnergyRetrieval [Portuguese] | 0.66518 | 0.65568 | -1.43% | 0.63437 | -4.63% | 0.65168 | -2.03% |
| Vidore3EnergyRetrieval [Spanish] | 0.66824 | 0.66152 | -1.01% | 0.64061 | -4.13% | 0.66989 | +0.25% |
| Vidore3FinanceEnRetrieval [English] | 0.68226 | 0.67461 | -1.12% | 0.67471 | -1.11% | 0.68427 | +0.29% |
| Vidore3FinanceEnRetrieval [French] | 0.61265 | 0.61589 | +0.53% | 0.58862 | -3.92% | 0.61400 | +0.22% |
| Vidore3FinanceEnRetrieval [German] | 0.60931 | 0.59313 | -2.66% | 0.59157 | -2.91% | 0.60733 | -0.32% |
| Vidore3FinanceEnRetrieval [Italian] | 0.62274 | 0.61746 | -0.85% | 0.61055 | -1.96% | 0.62008 | -0.43% |
| Vidore3FinanceEnRetrieval [Portuguese] | 0.61481 | 0.61814 | +0.54% | 0.60038 | -2.35% | 0.61914 | +0.70% |
| Vidore3FinanceEnRetrieval [Spanish] | 0.62669 | 0.62247 | -0.67% | 0.61558 | -1.77% | 0.62645 | -0.04% |
| Vidore3FinanceFrRetrieval [English] | 0.45463 | 0.45244 | -0.48% | 0.42045 | -7.52% | 0.44853 | -1.34% |
| Vidore3FinanceFrRetrieval [French] | 0.46140 | 0.47382 | +2.69% | 0.43176 | -6.42% | 0.46642 | +1.09% |
| Vidore3FinanceFrRetrieval [German] | 0.44498 | 0.46446 | +4.38% | 0.42005 | -5.60% | 0.45348 | +1.91% |
| Vidore3FinanceFrRetrieval [Italian] | 0.45387 | 0.45102 | -0.63% | 0.41925 | -7.63% | 0.45322 | -0.14% |
| Vidore3FinanceFrRetrieval [Portuguese] | 0.45347 | 0.45406 | +0.13% | 0.43487 | -4.10% | 0.45242 | -0.23% |
| Vidore3FinanceFrRetrieval [Spanish] | 0.47142 | 0.46027 | -2.37% | 0.43897 | -6.88% | 0.45875 | -2.69% |
| Vidore3HrRetrieval [English] | 0.64208 | 0.64230 | +0.03% | 0.60840 | -5.25% | 0.64230 | +0.03% |
| Vidore3HrRetrieval [French] | 0.60851 | 0.60398 | -0.74% | 0.55382 | -8.99% | 0.60336 | -0.85% |
| Vidore3HrRetrieval [German] | 0.60223 | 0.60588 | +0.61% | 0.57421 | -4.65% | 0.60522 | +0.50% |
| Vidore3HrRetrieval [Italian] | 0.61012 | 0.60338 | -1.10% | 0.56235 | -7.83% | 0.60503 | -0.83% |
| Vidore3HrRetrieval [Portuguese] | 0.61859 | 0.61976 | +0.19% | 0.57461 | -7.11% | 0.62209 | +0.57% |
| Vidore3HrRetrieval [Spanish] | 0.60888 | 0.60719 | -0.28% | 0.57509 | -5.55% | 0.59915 | -1.60% |
| Vidore3IndustrialRetrieval [English] | 0.57657 | 0.56474 | -2.05% | 0.57577 | -0.14% | 0.56862 | -1.38% |
| Vidore3IndustrialRetrieval [French] | 0.51532 | 0.51070 | -0.90% | 0.50532 | -1.94% | 0.51392 | -0.27% |
| Vidore3IndustrialRetrieval [German] | 0.50657 | 0.50224 | -0.85% | 0.51859 | +2.37% | 0.50605 | -0.10% |
| Vidore3IndustrialRetrieval [Italian] | 0.51296 | 0.50851 | -0.87% | 0.51327 | +0.06% | 0.50761 | -1.04% |
| Vidore3IndustrialRetrieval [Portuguese] | 0.52053 | 0.50656 | -2.68% | 0.51813 | -0.46% | 0.51967 | -0.17% |
| Vidore3IndustrialRetrieval [Spanish] | 0.52668 | 0.52081 | -1.11% | 0.52458 | -0.40% | 0.52544 | -0.24% |
| Vidore3PharmaceuticalsRetrieval [English] | 0.66648 | 0.66843 | +0.29% | 0.66572 | -0.11% | 0.66646 | -0.00% |
| Vidore3PharmaceuticalsRetrieval [French] | 0.64024 | 0.63780 | -0.38% | 0.62789 | -1.93% | 0.63545 | -0.75% |
| Vidore3PharmaceuticalsRetrieval [German] | 0.63307 | 0.63552 | +0.39% | 0.63286 | -0.03% | 0.63256 | -0.08% |
| Vidore3PharmaceuticalsRetrieval [Italian] | 0.64081 | 0.63341 | -1.15% | 0.63486 | -0.93% | 0.64078 | -0.00% |
| Vidore3PharmaceuticalsRetrieval [Portuguese] | 0.63926 | 0.64115 | +0.30% | 0.63734 | -0.30% | 0.64075 | +0.23% |
| Vidore3PharmaceuticalsRetrieval [Spanish] | 0.64837 | 0.64183 | -1.01% | 0.64129 | -1.09% | 0.64879 | +0.06% |
| Vidore3PhysicsRetrieval [English] | 0.47473 | 0.47110 | -0.76% | 0.46322 | -2.42% | 0.47101 | -0.78% |
| Vidore3PhysicsRetrieval [French] | 0.47655 | 0.46559 | -2.30% | 0.46877 | -1.63% | 0.47450 | -0.43% |
| Vidore3PhysicsRetrieval [German] | 0.44946 | 0.44726 | -0.49% | 0.45686 | +1.65% | 0.45579 | +1.41% |
| Vidore3PhysicsRetrieval [Italian] | 0.47512 | 0.46358 | -2.43% | 0.46902 | -1.28% | 0.46659 | -1.80% |
| Vidore3PhysicsRetrieval [Portuguese] | 0.47751 | 0.46068 | -3.52% | 0.46578 | -2.46% | 0.46152 | -3.35% |
| Vidore3PhysicsRetrieval [Spanish] | 0.47441 | 0.45781 | -3.50% | 0.47455 | +0.03% | 0.47053 | -0.82% |
| VidoreArxivQARetrieval [English] | 0.91151 | 0.90789 | -0.40% | 0.90922 | -0.25% | 0.90320 | -0.91% |
| VidoreDocVQARetrieval [English] | 0.66369 | 0.66496 | +0.19% | 0.65785 | -0.88% | 0.66049 | -0.48% |
| VidoreInfoVQARetrieval [English] | 0.94478 | 0.94577 | +0.10% | 0.94317 | -0.17% | 0.94699 | +0.23% |
| VidoreShiftProjectRetrieval [English] | 0.87889 | 0.92214 | +4.92% | 0.87202 | -0.78% | 0.90702 | +3.20% |
| VidoreSyntheticDocQAAIRetrieval [English] | 0.99262 | 0.99631 | +0.37% | 0.99262 | 0.00% | 0.99631 | +0.37% |
| VidoreSyntheticDocQAEnergyRetrieval [English] | 0.96710 | 0.96823 | +0.12% | 0.96524 | -0.19% | 0.97193 | +0.50% |
| VidoreSyntheticDocQAGovernmentReportsRetrieval [English] | 0.97579 | 0.96954 | -0.64% | 0.97172 | -0.42% | 0.97623 | +0.05% |
| VidoreSyntheticDocQAHealthcareIndustryRetrieval [English] | 0.99062 | 0.98693 | -0.37% | 0.99262 | +0.20% | 0.98693 | -0.37% |
| VidoreTabfquadRetrieval [English] | 0.94231 | 0.94361 | +0.14% | 0.94179 | -0.06% | 0.93863 | -0.39% |
| VidoreTatdqaRetrieval [English] | 0.80918 | 0.80714 | -0.25% | 0.79627 | -1.60% | 0.80553 | -0.45% |
|-----------|----------|------------|-----|------------|-----|------------|-----|
| **Average** | **0.64247** | **0.64044** | **-0.32%** | **0.63063** | **-1.84%** | **0.64198** | **-0.08%** |

## Summary

- **Benchmark files (Original):** 22
- **Total entries evaluated:** 71

### Performance by Calibration Sequence Length

| Metric | seqlen=256 | seqlen=512 | seqlen=1024 |
|--------|------------|------------|-------------|
| **Improved** | 28 | 14 | 30 |
| **Degraded** | 43 | 56 | 41 |
| **Unchanged** | 0 | 1 | 0 |

### Overall Scores

| Model | Average NDCG@5 | Change from Original |
|-------|----------------|----------------------|
| Original (FP16) | 0.64247 | - |
| AWQ (seqlen=256) | 0.64044 | -0.32% |
| AWQ (seqlen=512) | 0.63063 | -1.84% |
| AWQ (seqlen=1024) | 0.64198 | -0.08% |

## Performance Graphs

![Performance Comparison](performance_comparison_8B_all_languages.png)

![Performance Difference](performance_diff_8B_all_languages.png)
