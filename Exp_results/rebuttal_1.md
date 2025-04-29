# Rebuttal


**Reviewer GLth:**


## Rebuttal to Reviewer GLth  

We greatly appreciate Reviewer GLth for providing a detailed summary of our strengths. And greatly appreciate Reviewer GLth for proposing these insightful questions and providing these constructive suggestions. 
We have revised our manuscript according to your suggestions. We also provided supplementary materials, which separate the newly added context for your convenience. Below, we provide our responses to the comments, denoted by [W] for weaknesses and [Q] for questions.


**Response to W1, W2 and Q1: Alternative distance metrics.** 

We greatly appreciate the Reviewer's comment. As you suggested, we conducted the experiments using other metrics, such as cosine similarity (the Normalized Temperature-Scaled Cross Entropy (NT-Xent) used in SimCLR). 
The results are presented as follows, where the distance using cosine similarity achieves similar model utility preservation as the L2-norm.
However, the cosine similarity achieves the lower unlearning effective for MIA (infer the unlearned samples are not in the unlearned model training set) than L2-norm.
We think when using similarity as the triplet distance, the similarity has a higher preference to optimize the (anchor, positive) pair, making unlearning more difficult than L2-norm.

The **Tab.R1** of evaluating the distance of L2-norm and similarity on MNIST: USS = 100

| On MNIST       | MIA         |  RA  | TA  |  RT (second) |   
| --------         | --------    | -------- | -------- |  -------- |    
| L2_norm           | 82.99%      | 99.41%   |  99.10%  |  0.161   |    
| Cosine Similarity (NT-Xent) | 62.99%  | 99.38%   | 99.04%   |  0.221  |     


 
 


**Response to W3: Strong assumptions on centroid stability** 

We greatly thank the Reviewer's insightful comments. We agree that the class centroid remains unchanged after unlearning is too strong. 
Therefore, we revised the constraints to preserve the local manifold representation of the top-k similar samples nearest to the unlearning samples. Inspired by existing unlearning methods, which set the constraints as the minimization term, minimizing the Kullback-Leibler Divergence (KLD) between the unlearned representation $z_{r,u}$ and original $z_r$ of the remaining samples, we also set our constraint term as a minimization term to minimize the distance between top-k samples to their center. It performs like an adversary term and will not be as strict as the previous constraint. We choose top-k nearest samples rather than all samples in the same class because we observe that in approximate unlearning, the most influenced samples are those samples similar to the erased sample. We present the revision as follows.

```  
\textbf{Constraints to Preserve the Learned Manifold Representation Knowledge.} Blindly pushing unlearned samples away risks drastically altering the representation of all samples, leading to a collapse in overall model performance. We need to preserve the manifold representation knowledge $\texttt{z}_r$ for data in $S_r$. Existing methods minimize Kullback-Leibler Divergence (KLD) between $\texttt{z}_{r,u}$ and $\texttt{z}_r$ to ensure the new unlearned representation is not too different from the original learned one on the remaining dataset to preserve the knowledge \cite{fu2022knowledge}. It is straightforward and effective but needs to iterate all the remaining samples, which is computationally expensive. We observe that in approximate unlearning, the most influenced samples are those samples similar to the erased sample. Therefore, we chose the top-k samples that were similar to the erased sample and set a knowledge-preserving goal for them to constrain the local manifold representation knowledge. 
\begin{equation} \label{constraint}
	\small
	\min_{\theta_u} \| f_{\theta_u}(x_i) - \frac{1}{ \|S_k\|} \sum_{i \in S_k} \texttt{z}_i \|^2, 
\end{equation}
where the $S_k$ is the set of top-k similar samples nearest the erased sample $x_u \in S_u$.

```  

**Response to W4: Limited Experimental Validation** 
We sincerely thank the Reviewer's comments and suggestions about conducting experiments on larger datasets and larger unlearning sample size.
We demonstrate new experimental results as follows. 
For approximate unlearning methods, adding more unlearning samples definitely increases the unlearning effectiveness (higher MIA) but decreases the model utility (RA and TA) at the same time.
We should notice that existing approximate unlearning methods keep a very good model utility (similar to retraining) highly relying on fine-tuning or retraining the unlearned model on the remaining dataset.
The experimental results only show the experiments directly based on the representation unlearning, while not adding additional fine-tuning to utilize the remaining dataset, but we can add it if necessary.

For larger dataset, we are still conducting experiments on imagenet, but it needs more time to get the results.
We also want to explain that CelebA (178x218) is already large as imagenet (224x224), which can be treat as a proof of our method for complex task. 
We are conducting more experiments on imagenet at least miniImagenet. 


The **Tab.R2** of evaluating TCU on larger unlearning sample size (USS):

| On MNIST       | 200|  400  | 600  |  800 | 1000 |  
| --------       | --------    | -------- | -------- |  -------- |   -------- | 
| MIA            | 65.98%      | 68.11%   | 73.18%  |  77.25%   |  82.89%    |  
| RA             | 99.50%      | 99.20%   | 98.70%   |  98.03%   |  96.71%    |  
| TA             | 99.03%      | 98.85%   | 97.78%  |  97.25%   |  96.32%    |  
| RT             | 0.331       | 0.593    | 0.963    |  1.293    |  1.429     |  
 



| On CIFAR10       | 200|  400  | 600  |  800 | 1000 |  
| --------         | --------    | -------- | -------- |  -------- |   -------- | 
| MIA              | 72.48%      | 75.19%   |  78.83%  |  82.52%   |  88.19%    |  
| RA               | 99.01%      | 98.93%   | 98.70%   |  98.53%   |  98.41%    |  
| TA               | 82.13%      | 82.05%   |  81.89%  |  81.59%   |  80.69%    |  
| RT               | 0.431       | 0.798    | 1.323    |  1.704    |  2.033     |  



The **Tab.R3** of evaluating TCU on larger and complex dataset:
| On miniImageNet   | 200|  400  | 600  |  800 | 1000 |  
| --------          | --------    | -------- | -------- |  -------- |   -------- | 
| MIA               | 65.98%      | -   |  -  |  -   |  -    |  
| RA                | 99.50%      | 99.20%   | 98.70%   |  98.03%   |  96.71%    |  
| TA                | 43.03%      | 98.85%   |  97.78%  |  97.25%   |  96.32%    |  
| RT                | 0.331       | 0.593    | 0.963    |  1.293    |  1.429     | 



**Response to W5: Lack of Discussion on Related Work.** 
We add a discussion about the mentioned references in related work.
We present the revision as follows.

```  
There are also some studies that implement unlearning by focusing on representation and model weights. In [1], they targeted the class unlearning problem based on sparse representation learned by Discrete Key-Value Bottleneck. Guo et al. proposed unlearning methods to forget the attribute instead of samples based on feature representations [2]. The study of [3] directed the attention of machine unlearning toward specific model weights rather than the entire model, trying to improve effectiveness and efficiency. Different from them, our unlearning method is based on a representation with maximum manifold representation capacity.
```  


**Response to Q2:** The centroid correlation is how correlated the centroids of different classes are, which can be measured using pairwise inner products between normalized class centroids.


**Response to Q3:** "Model-centric" means considering the unlearning from the model's perspective.


**Response to Q4:** In the new definition, we push the unlearned samples from their original representation and only choose top-k nearest remaining samples to preserve the remaining knowledge.
We test the performance of the generative models as follows.


**Response to Q5:** Algorithm 1 is an example. During experiments, we set the size of mini-batch = 16 on MNIST and CIFAR10.



## Rebuttal to Reviewer PjaX  





**Response to W1: Without a truly contribution.** 
We thank the Reviewer's comments. Our key contribution is to investigate machine unlearning solely through the model learned manifold representation perspective, and propose the corresponding solution. To enhance the technique contribution, we carefully revisit the existing representation learning and triplet learning methods. 
Also inspired by reviewer's comments, we find the fixed margin (\alpha) could be optimized for our TCU method. Hence, we revised our paper and proposed an adaptive margin based on self-mode connectivity. As the character's limitation, we provide a short introduction to the new adaptive unlearning margin in the second response, as it is related to the second weakness of the comments.





  
**Response to W2: Lack of theoretical differentiation between TCU and gradient based method, and lack of a quantifiable standard for unlearning success.** 
We appreciate the Reviewer's comments. 
We supplemented a sketch of why TCU is superior to gradient ascent-based methods. 

'''
Gradient ascent is used to maximize the error of the unlearning sample and its' original class, which makes the model classify it into the wrong category. At the representation level, the unlearned $z_u$ is pushed away from the previous $z_i$. The learned manifold representation has a larger space than the task-related output. Designing unlearning based on representation can provide a more-grained adjustment and avoid the limitation of direct misclassification, hence effectively preventing the model utility from decreasing.
'''


Moreover, we provide a short introduction of the adaptive unlearning margin method guided by local mode connectivity here.

'''
Ideally, if we have a retrained model, we can calculate the optimal margin \alpha for the triplet loss. However, we don't have the retrained model. Could we have a fast replacement? We propose to use the local mode connectivity [R1, R2] to reconstruct a local manifold representation for the top-k nearest samples (to the unlearned sample) in the remaining dataset.  From [R1, R2], we know mode connectivity can fast ensemble a new model based on several benign samples that can redeem negative influence from old models, even removing the backdooring or poison. We use the mode connectivity to fast reconstruct the local representation for top-k nearest samples without the unlearned sample. Specifically, an initial model is prepared, and we connect it with the unlearned and trained models. We optimize a path to best link the three models on the top-k remaining samples. The connecting model can be seen guided by the trained and unlearned model based only on the top-k remaining samples. The new local representation of the connecting model can be used as a guide to select a suitable margin value for successful unlearning.
'''

We revised our paper and the implementation. We also conducted new experiments with the adaptive margin value, which shows better unlearning effectiveness and less model utility degradation. 

[R1]. Garipov, Timur, et al. "Loss surfaces, mode connectivity, and fast ensembling of dnns." Advances in neural information processing systems 31 (2018).

[R2]. Zhao, Pu, et al. "Bridging mode connectivity in loss landscapes and adversarial robustness." ICLR (2020).

**Response to W3: a larger number of samples.** 
We sincerely thank the Reviewer's comments and suggestions about conducting experiments on a larger unlearning sample size.
We demonstrate new experimental results as follows. 
For approximate unlearning methods, adding more unlearning samples definitely increases the unlearning effectiveness (higher MIA) but decreases the model utility (RA and TA) at the same time.
We should notice that existing approximate unlearning methods keep a very good model utility (similar to retraining) highly relying on fine-tuning or retraining the unlearned model on the remaining dataset.
The experimental results only show the experiments directly based on the representation unlearning, while not adding additional fine-tuning to utilize the remaining dataset, but we can add it if necessary.

 


The **Tab.R2** of evaluating TCU on a larger unlearning sample size (USS):

| On MNIST       | 200|  400  | 600  |  800 | 1000 |  
| --------       | --------    | -------- | -------- |  -------- |   -------- | 
| MIA            | 65.98%      | 68.11%   | 73.18%  |  77.25%   |  82.89%    |  
| RA             | 99.50%      | 99.20%   | 98.70%   |  98.03%   |  96.71%    |  
| TA             | 99.03%      | 98.85%   | 97.78%  |  97.25%   |  96.32%    |  
| RT             | 0.331       | 0.593    | 0.963    |  1.293    |  1.429     |  
 



| On CIFAR10       | 200|  400  | 600  |  800 | 1000 |  
| --------         | --------    | -------- | -------- |  -------- |   -------- | 
| MIA              | 72.48%      | 75.19%   |  78.83%  |  82.52%   |  88.19%    |  
| RA               | 99.01%      | 98.93%   | 98.70%   |  98.53%   |  98.41%    |  
| TA               | 82.13%      | 82.05%   |  81.89%  |  81.59%   |  80.69%    |  
| RT               | 0.431       | 0.798    | 1.323    |  1.704    |  2.033     |  




## Rebuttal to Reviewer zJG2


 
 
**Response to Comment1: Difference between gradient ascent and TCU.**

Gradient ascent based on the penalty of the task and the representation distance penalty is different.
We supplement a new discussion of why unlearning-based manifold representation is better than gradient reverse based on task as follows. We will also enhance it and provide more formal proof.

'''
Gradient ascent is used to maximize the error of the unlearning sample and its' original class, which makes the model classify it into the wrong category. At the representation level, the unlearned $z_u$ is pushed away from the previous $z_i$. The learned manifold representation has a larger space than the task-related output. Designing unlearning based on representation can provide a more-grained adjustment and avoid the limitation of direct misclassification, hence effectively preventing the model utility from decreasing.
'''

**Response to C2:**

For unlearning for privacy from the get-go, we should explain that the experiment of unlearning the trojan backdoors could be an example in our experiments, as our method can effectively remove the influence of backdoored samples.


We conducted experiments on self-supervised models such as VAE as follows. 
The results show the effective unlearning for generative models.
 
The **Tab.R3** of evaluating TCU on the generative model (VAE): where the R-MSE and T-MSE are the reconstructed mean square error (MSE) on the remaining (R-) and test (T-) datasets. The original MIA for the trained model is 51.99%, R-MSE is 0.0358, and T-MSE is 0.0361

| On MNIST       | USS = 200|  400  | 600  |  800 | 1000 |  
| --------       | --------    | -------- | -------- |  -------- |   -------- | 
| MIA           | 54.50%      | 62.18%   | 69.88%  |  73.85%   |  82.10%    |  
| R-MSE         | 0.0371      | 0.0387   | 0.0403   |  0.0420   |  0.0450    |  
| T-MSE         | 0.0373      | 0.0389  | 0.0404 |  0.0423   |  0.0453    |  
| RT             | 0.440       | 0.593    | 0.963    |  1.293    |  1.924     |  

 


**Response to C3: A simple application of the MMCR seems more effective**
We observed this phenomenon, the intra-class compact representation manifold, i.e., MMCR can increase the learning and unlearning performance, which is also the inspiration of our method that plans to conduct unlearning within representation. However, MMCR is not an unlearning method, it cannot make the model learn or unlearn a sample, it just can enhance the learning or unlearning methods' performance.

Actually, our method without the intra-class compact limitation can also perform well as shown in the experiments, but MMCR enhances the unlearning effectiveness.


**Response to R1:**

We cited and added a discussion about these references as follows.

```  
There are also some studies that implement unlearning by focusing on representation, model weights, and even model editing in large language models [4]. In [1], they targeted the class unlearning problem based on sparse representation learned by Discrete Key-Value Bottleneck. Guo et al. proposed unlearning methods to forget the attribute instead of samples based on feature representations [2]. The study of [3] directed the attention of machine unlearning toward specific model weights rather than the entire model, trying to improve effectiveness and efficiency. Different from them, our unlearning method is based on a representation with maximum manifold representation capacity.
``` 

[3]. Fan et al., SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classificationand Generation, ICLR’24

[4]. Liu et al, Rethinking machine unlearning for large language models, NMI

**Response to W1 and W2: Questions about Eq.3 and 4.**
In Eq. 3, we use $c_{y,i}$ to denote the centroid of the original $$z_{i,o}$$.
However, after reading reviews, we find that pushing away from the original center may not be that suitable.
We change $c_{y,i}$ as $z_{u,o}$, hoping to push away from the erased sample's original representation, which is more reasonable. 
Moreover, we also change the constraint correspondingly. Since we find that the model-influenced samples in the remaining dataset are those samples near the unlearned sample, we modify our constraint the minimizing the distance between the top-k nearest samples and their original center to maintain the original learned local knowledge. 
Therefore, we also revised the Eq. 4 correspondingly. This revision of the definition and implementation of our code further improved the unlearning effectiveness and model utility preservation.
For the character's limitation, we omit the detailed revision here. 


 

**Response to W3: Is the Trojan samples are unlearned samples $S_u$**
Yes, the Trojan samples are the $S_u$, which need to be unlearned, eliminating backdoors.
This could also be an experiment for previous comments of unlearning privacy from the get-go, as our method can effectively remove the influence of backdoored samples.




**Response to Q1: Larger Unlearning sample size.**
Thank you for your questions, we additionally conduct experiments on a new sample size scale, ranging from 200 to 1000.
1000 sample is around 2% of the original training dataset. For approximate unlearning methods, adding more unlearning samples definitely increases the unlearning effectiveness (higher MIA) but decreases the model utility (RA and TA) at the same time. 
Although other existing methods are effective in unlearning 20% or 50% of the training dataset, they heavily rely on fine-tuning on the remaining dataset.
The experimental results only show the experiments directly based on the representation unlearning, while not adding additional fine-tuning to utilize the remaining dataset, but we can add it if necessary.


 
 


The **Tab.R2** of evaluating TCU on a larger unlearning sample size (USS):

| On MNIST       | USS=200|  400  | 600  |  800 | 1000 |  
| --------       | --------    | -------- | -------- |  -------- |   -------- | 
| MIA            | 65.98%      | 68.11%   | 73.18%  |  77.25%   |  82.89%    |  
| RA             | 99.50%      | 99.20%   | 98.70%   |  98.03%   |  96.71%    |  
| TA             | 99.03%      | 98.85%   | 97.78%  |  97.25%   |  96.32%    |  
| RT             | 0.331       | 0.593    | 0.963    |  1.293    |  1.429     |  
 



| On CIFAR10       | USS=200|  400  | 600  |  800 | 1000 |  
| --------         | --------    | -------- | -------- |  -------- |   -------- | 
| MIA              | 72.48%      | 75.19%   |  78.83%  |  82.52%   |  88.19%    |  
| RA               | 99.01%      | 98.93%   | 98.70%   |  98.53%   |  98.41%    |  
| TA               | 82.13%      | 82.05%   |  81.89%  |  81.59%   |  80.69%    |  
| RT               | 0.431       | 0.798    | 1.323    |  1.704    |  2.033     |  




**Response to Q2: Triplet Selection**

Thank you for your insightful questions, we agree that the mean of a class might be less critical, as unlearning a few samples may not heavily influence the whole class.
After discussion with experts in this area, we now realize that the most influenced samples are the samples most similar to the unlearned sample, i.e., the samples with the nearest representation to the erased samples' representation.
Therefore, we revised the constraints using the top-k nearest similar samples' center as the anchor. 
Moreover, we also designed an adaptive margin method guided by mode connectivity, which rapidly reconstructs the local manifold representation of top-k samples of a new model without the unlearned samples. Hence, we can use the new reconstructed representation to guide the margin value chosen for better unlearning effectiveness.



**Response to Q3:**

We thank the Reviewers' comments. And we agree that unlearning multimodal models is also important. However, we have not conducted experiments on VLMs. 
We believe that our solution at least provides some insights for unlearning the representation of VLMs. And it will be our future work to propose a good unlearning solution for VLMs.

**Response to Q4:**

We thank the Reviewer's insightful question. The margin value alpha is the distance that we at least push the unlearned sample from the original manifold representation.
A larger alpha means better unlearning effectiveness while degrading the model utility at the same time. 


We consider a fixed margin may not be that effective for all scenarios. Therefore, we designed an adaptive margin method guided by mode connectivity, which rapidly reconstructs the local manifold representation of top-k samples of a new model without the unlearned samples. Hence, we can use the new reconstructed representation to guide the margin value chosen for better unlearning effectiveness.




**Response to Q5:**

We thank the Reviewer's insightful question. The margin value alpha is the distance that we at least push the unlearned sample from the original manifold representation.
A larger alpha means better unlearning effectiveness while degrading the model utility at the same time. 


We consider a fixed margin may not be that effective for all scenarios. Therefore, we designed an adaptive margin method guided by mode connectivity, which rapidly reconstructs the local manifold representation of top-k samples of a new model without the unlearned samples. Hence, we can use the new reconstructed representation to guide the margin value chosen for better unlearning effectiveness.







## Rebuttal to Reviewer YutA


**Response to W1: the TCU method relies on the premise that the model has learned a good intra-class compact representation manifold.**

We observed this phenomenon, the intra-class compact representation manifold, i.e., MMCR can increase the learning and unlearning performance, which is also the inspiration of our method that plans to conduct unlearning within representation. 

Actually, our method can also perform well without the intra-class compact requirement as shown in the experiments, but MMCR does enhance the unlearning effectiveness.

To make the paper clearer, we revised the Preliminary section to emphasize that the observation of intra-class compact is the intuition of our method. In the experimental demonstration, we added the original model performance to show the effectiveness of our method without MMCRs. For the characters' limitation, we will not show the detailed revision here.


**Response to W2: Experiments on larger datasets or more complex tasks**
We sincerely thank the Reviewer's comments and suggestions about conducting experiments on a larger unlearning sample size.
We demonstrate new experimental results as follows. 
For approximate unlearning methods, adding more unlearning samples definitely increases the unlearning effectiveness (higher MIA) but decreases the model utility (RA and TA) at the same time.
We should notice that existing approximate unlearning methods keep a very good model utility (similar to retraining) highly relying on fine-tuning or retraining the unlearned model on the remaining dataset.
The experimental results only show the experiments directly based on the representation unlearning, while not adding additional fine-tuning to utilize the remaining dataset, but we can add it if necessary.

 


The **Tab.R2** of evaluating TCU on a larger unlearning sample size (USS):

| On MNIST       | 200|  400  | 600  |  800 | 1000 |  
| --------       | --------    | -------- | -------- |  -------- |   -------- | 
| MIA            | 65.98%      | 68.11%   | 73.18%  |  77.25%   |  82.89%    |  
| RA             | 99.50%      | 99.20%   | 98.70%   |  98.03%   |  96.71%    |  
| TA             | 99.03%      | 98.85%   | 97.78%  |  97.25%   |  96.32%    |  
| RT             | 0.331       | 0.593    | 0.963    |  1.293    |  1.429     |  
 



| On CIFAR10       | 200|  400  | 600  |  800 | 1000 |  
| --------         | --------    | -------- | -------- |  -------- |   -------- | 
| MIA              | 72.48%      | 75.19%   |  78.83%  |  82.52%   |  88.19%    |  
| RA               | 99.01%      | 98.93%   | 98.70%   |  98.53%   |  98.41%    |  
| TA               | 82.13%      | 82.05%   |  81.89%  |  81.59%   |  80.69%    |  
| RT               | 0.431       | 0.798    | 1.323    |  1.704    |  2.033     |  


For more complex tasks, we test our methods for generative task on VAE.
The results are shown as follows, which show the effective unlearning for generative models.
 
The **Tab.R3** of evaluating TCU on the generative model (VAE): where the R-MSE and T-MSE are the reconstructed mean square error (MSE) on the remaining (R-) and test (T-) datasets. The original MIA for the trained model is 51.99%, R-MSE is 0.0358, and T-MSE is 0.0361

| On MNIST       | USS = 200|  400  | 600  |  800 | 1000 |  
| --------       | --------    | -------- | -------- |  -------- |   -------- | 
| MIA           | 54.50%      | 62.18%   | 69.88%  |  73.85%   |  82.10%    |  
| R-MSE         | 0.0371      | 0.0387   | 0.0403   |  0.0420   |  0.0450    |  
| T-MSE         | 0.0373      | 0.0389  | 0.0404 |  0.0423   |  0.0453    |  
| RT             | 0.440       | 0.593    | 0.963    |  1.293    |  1.924     |  

 

**Response to W3: the theoretical derivation of the paper is not rigorous enough.** 
We thank the Reviewer's comments. We also realized that keeping the unlearned manifold representation compacity distributed within $\delta_p$ may not be strictly true in practice.

After discussion with experts in this area, we now realize that the most influenced samples are the samples most similar to the unlearned sample, i.e., the samples with the nearest representation to the erased samples' representation.
Therefore, we revised our learned representation knowledge constraints using the top-k nearest similar samples' center as the anchor. And the unlearning lower bound is derivated based on the top-k nearest remaining samples, which would be more practical. 
Moreover, we consider a fixed margin may not be that effective for all scenarios. Therefore, we designed an adaptive margin method guided by mode connectivity, which rapidly reconstructs the local manifold representation of top-k samples of a new model without the unlearned samples. Hence, we can use the new reconstructed representation to guide the margin value chosen for better unlearning effectiveness.






**Response to W4:**
Thank you for your comments. We revised and checked the paper to fix these typos. 











