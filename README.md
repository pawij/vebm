# Variational Event-Based Model (vEBM)

# NOTE: this code will be deprecated soon. The new version, Latent Event Mapping (LEMING), will be found on the lab website here: https://github.com/lililab-sussex/leming/ 

The vEBM is a probabilistic disease progression model that leverages 
optimal transport to scale to large feature sets, enabling rapid, 
low-compute inference of fine-grained multi-modal trajectories. It can 
also use any combination of multi-modal features, not just neuroimaging, 
e.g., clinical test scores, biofluids, genomics.

If you use the vEBM, please cite this paper:

Wijeratne, PA & Alexander, DC (2024). "Unscrambling disease progression 
at scale: fast inference of event permutations with optimal transport". 
Advances in Neural Information Processing Systems. 
https://doi.org/10.48550/arXiv.2410.14388

## Installation

Install directly from GitHub using pip:

```bash
pip install git+https://github.com/pawij/birkhoff.git
```

### Dependencies

| Package      | Version     |
|--------------|-------------|
| numpy        | >=1.26, <2  |
| scipy        | >=1.11      |
| scikit-learn | >=1.3       |
| torch        | >=2.2       |
| matplotlib   | >=3.7       |

Python 3.10 or higher is required.

## Example

Here we apply the vEBM to structural MRI data from the Alzheimer's 
Disease Neuroimaging Initiative (ADNI) dataset. It shows pixel-level 
disease progression events in the brain, providing new fine-grained 
insights into changes at the tissue-level caused by Alzheimer's disease.

Training this model took only 5 minutes on a single laptop CPU.

![ADNI vEBM](adni_vebm.gif)

## Contributors
- Peter Wijeratne (p.wijeratne@pm.me)
- Misha Jairamani

## License: MIT
