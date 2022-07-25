# Overview

The code in this repository implements a four-step method for the reconstruction of the firing behaviour of an entire pool of motoneurons and its validation.
The code is associated with the following publication:

```
@article{caillet2022estimation,
  title={Estimation of the firing behaviour of a complete motoneuron pool by combining EMG signal decomposition and realistic motoneuron modelling},
  author={Caillet, Arnault and Phillips, Andrew TM and Farina, Dario and Modenese, Luca},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```

# Instructions

1. Set up an environment
2. To run the four-step method, run '1_MAIN_MN_model.py'
3. To run the iterative validation of the identified spike trains, run '2_MN_Model_validation.py'
4. To plot and display the results obtained with the four experimental datasets DTA_35, HTA_35, HTA_50, HGM_30, already stored in the Results folder, 
run '3_plot_stored_MN_model_results.py'
