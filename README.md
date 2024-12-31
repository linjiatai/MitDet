# Rethinking Mitosis Detection: Towards Diverse Data and Feature Representation for Better Domain Generalization
![outline](MitDet.png)

## Introduction
The iplementation of:

**[Rethinking Mitosis Detection: Towards Diverse Data and Feature Representation for Better Domain Generalization](https://ieeexplore.ieee.org/document/9740140)**

You can also download the repository from https://github.com/linjiatai/MitDet.git.

## Abtract
Mitosis detection is one of the fundamental tasks in computational pathology, which is extremely challenging due to the heterogeneity of mitotic cell. Most of the current studies solve the heterogeneity in the technical aspect by increasing the model complexity. However, lacking consideration of the biological knowledge and the complex model design may lead to the overfitting problem while limited the generalizability of the detection model. In this paper, we systematically study the morphological appearances in different mitotic phases as well as the ambiguous non-mitotic cells and identify that balancing the data and feature diversity can achieve better generalizability. Based on this observation, we propose a novel generalizable framework (MitDet) for mitosis detection. The data diversity is considered by the proposed diversity-guided sample balancing (DGSB). And the feature diversity is preserved by inter- and intra- class feature diversity-preserved module (InCDP). Stain enhancement (SE) module is introduced to enhance the domain-relevant diversity of both data and features simultaneously. Extensive experiments have demonstrated that our proposed model outperforms all the state-of-the-art (SOTA) approaches in several popular mitosis detection datasets in both internal and unseen test sets using point annotations only. Comprehensive ablation studies have also proven the effectiveness of the rethinking of data and feature diversity balancing. By analyzing the results quantitatively and qualitatively, we believe that our proposed model not only achieves SOTA performance but also might inspire the future studies in new perspectives.
