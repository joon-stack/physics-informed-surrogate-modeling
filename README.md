# physics-informed-surrogate-modeling
Source code for [Data-Efficient Surrogate Modeling Using Meta-Learning and Physics-Informed Deep Learning Approaches] (Under Review).

This paper proposes physics-informed meta-learning-based surrogate modeling (PI-MLSM), a novel approach that combines meta-learning and physics-informed deep learning to train surrogate models with limited labeled data. PI-MLSM consists of two stages: meta-learning and physics-informed task adaptation. The proposed approach is demonstrated to outperform other methods in four numerical examples while reducing errors in prediction and reliability analysis, exhibiting robustness, and requiring less labeled data during optimization. Moreover, compared to other approaches, the proposed approach exhibits better performance in solving out-of-distribution tasks. Although this paper acknowledges certain limitations and challenges, such as the subjective nature of physical information, it highlights the key contributions of PI-MLSM, including its effectiveness in solving a wide range of tasks and its ability in handling situations wherein physical laws are not explicitly known. Overall, PI-MLSM demonstrates potential as a powerful and versatile approach for surrogate modeling.

The following figure is the overall mechanism of the proposed approach: The approach consists of two stages: meta-learning and physics-informed task adaptation. Meta-knowledge is extracted
from meta-learning using the source task family at the meta-learning stage, which is used for the model initialization at the physics-informed adaptation stage. In this study, a “task” is defined to be a surrogate learning a single system.


<img width="544" alt="image" src="https://github.com/joon-stack/physics-informed-surrogate-modeling/assets/82109076/542b71f3-11db-4ba0-912e-cf953a267b6a">

This is the overall mechanism of physics-informed deep learning (PIDL): The neural network model (MLP in this paper) produces the approximate solution, and the data-driven loss is computed using the labeled data. Next, by computing the derivatives of the approximate solutions, the physics-informed loss is computed using the governing equation. The total loss is computed as the weighted sum of the data-driven loss and the physics-informed loss, and the weight is set as 1.0 in this study. Finally, the total loss is used to update the model.


<img width="568" alt="image" src="https://github.com/joon-stack/physics-informed-surrogate-modeling/assets/82109076/f49f5a7f-8f25-4a95-a027-ee5c9e503b8d">



[Data-Efficient Surrogate Modeling Using Meta-Learning and Physics-Informed Deep Learning Approaches]: http://dx.doi.org/10.2139/ssrn.4470991
 
