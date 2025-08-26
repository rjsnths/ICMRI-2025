# ICMRI-2025

### Robust Self-Supervised Water-Fat Separation with Noise-Augmented and Self-Guided Learning

-------------------


#### Objectives

We propose an enhanced self-supervised learning–based method for water–fat separation. Although T<sub>2</sub><sup>\*</sup>-IDEAL with a multi-peak fat signal model is considered the gold standard, it is highly sensitive to initialization.
To address this, Jafari et al. introduced an unsupervised learning approach using a T<sub>2</sub><sup>*</sup>-IDEAL forward model. This included a scan-specific no-training DNN (NTD) employing deep image prior. However, NTD often suffers from instability during the training. \
To overcome this limitation, we incorporate a noise-augmented self-guided strategy, adding Gaussian noise during training, and apply total-variation (TV) regularization on the field map.  
<br>

![Fig 1](https://github.com/user-attachments/assets/9e8135fd-2e8a-476f-89a7-894d4a431a55)
*Fig 1. Proposed self-supervised framework for water–fat separation. Multi-echo GRE data (magnitude and phase) are processed by two U-Nets to estimate water/fat, R<sub>2</sub><sup>\*</sup>, and B0 field maps. Gaussian noise injection and total-variation (TV) regularization stabilize the training.</small>*

<br><br>
#### Methods

We evaluated the proposed method using retrospective six-echo multi-coil liver data (first TE = 2.948ms, echo spacing = 0.608ms) from the ISMRM 2012 workshop dataset.\
The network architecture (Fig. 1) consists of two U-Nets (depth 5, 2×2 kernels). One U-Net processes the magnitudes of multi-echo GRE data to generate water/fat magnitudes and R<sub>2</sub><sup>\*</sup> (=1/T<sub>2</sub><sup>*</sup>), while the other processes the phases to generate the water/fat phase and the B0 field map.\
At each training step, 5% Gaussian noise was added to the GRE images to stabilize learning and improve separation performance. A six-peak fat signal model was employed for more accurate decomposition. Additionally, TV regularization was applied to the B0 field map to enforce smoothness.

<br>

![Fig 2](https://github.com/user-attachments/assets/022df451-9615-43c2-a5ec-3dfe42d2169f)
*Fig 2. Comparison of water, fat, PDFF, field, and R<sub>2</sub><sup>\*</sup> maps obtained using 6-point T<sub>2</sub><sup>\*</sup>-IDEAL, NTD, and the proposed method. NTD shows unstable separation with flipped PDFF values (yellow boxes), whereas the proposed method provides stable PDFF and improved field maps while preserving structural details.*

<br>

![Fig 3](https://github.com/user-attachments/assets/05bbf0e6-aab3-4516-9849-cb42d27025ce)
*Fig 3. Bland–Altman analysis of PDFF compared with 6-point T<sub>2</sub><sup>\*</sup>-IDEAL. Twenty-five 5×5 ROIs (red boxes) were selected in the liver. The proposed method shows reduced bias and narrower limits of agreement compared to NTD.*

<br><br>
#### Results
Fig.2 shows the reconstructed water, fat, PDFF, B0 field map, and R<sub>2</sub><sup>\*</sup> maps. Comparisons were made among T<sub>2</sub><sup>\*</sup>-IDEAL with graph-cut, NTD, and the proposed method. In NTD, water–fat separation occasionally failed, producing flipped PDFF values (yellow box). Conversely, the proposed method produced stable PDFF maps while preserving structural details.\
Fig.3 presents Bland–Altman plots comparing NTD and the proposed method against six-point T<sub>2</sub><sup>*</sup>-IDEAL PDFF. The proposed method demonstrated lower bias and reduced variability compared to NTD, indicating improved agreement with the conventional IDEAL.

<br><br>
#### Conclusions

The proposed approach achieves more stable water–fat separation by incorporating noise augmentation and TV-regularization into a self-guided deep learning framework.
