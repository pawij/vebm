# Variational Event-Based Model (vEBM)
The vEBM is a probabilistic disease progression model that leverages optimal transport to scale to large feature sets, enabling rapid, low-compute inference of fine-grained multi-modal trajectories. It can also use any combination of multi-modal features, not just neuroimaging, e.g., clinical test scores, biofluid, genomic.

If you use the vEBM please cite this paper:

Wijeratne & Alexander (2024). "Unscrambling disease progression at scale: fast inference of event permutations with optimal transport". Advances in Neural Information Processing Systems. <https://doi.org/10.48550/arXiv.2410.14388>

Here is a demonstration of the vEBM using structural magnetic resonance imaging (MRI) data from the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset. It shows pixel-level disease progression events in the brain, providing new fine-grained insights into changes at the tissue-level caused by Alzheimer's disease. Training this model took only 5 minutes on a single laptop CPU.

![](https://github.com/pawij/vebm/blob/main/adni_vebm.gif)
