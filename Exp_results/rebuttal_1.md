# Rebuttal


**Reviewer GLth:**


## Rebuttal to Reviewer GLth  

We greatly appreciate Reviewer GLth for providing a detailed summary of our strengths. And greatly appreciate Reviewer GLth for proposing these insightful questions and providing these constructive suggestions. 
We have revised our manuscript according to your suggestions. We also provided supplementary materials, which separate the newly added context for your convenience. Below, we provide our responses to the comments, denoted by [W] for weaknesses and [Q] for questions.


**Response to W1 and Q1:** We greatly appreciate the Reviewer's comment. As you suggested, we conducted the experiments using other metrics, such as cosine similarity. The results are presented as follows.

The **Tab.R1** of evaluating mixing unlearned data in the clean dataset on CIFAR10: USR=1%, ASR=1%, and CSR=1%

| On CIFAR10       | MIA|  RA  | TA  |  RT | 8% |  
| --------         | --------    | -------- | -------- |  -------- |   -------- | 
| L2      | 73.89%      | 73.85%   |  73.78%  |  73.25%   |  73.03%    |  
| Cosine Similarity    | 9.40%       | 13.60%   | 33.40%   |  35.40%   |  43.26%    |  


**Response to W2:** We sincerely thank the Reviewer's comment. Besides the cosine similarity, we also test the Normalized Temperature-Scaled Cross Entropy (NT-Xent) used in SimCLR and Mutual Information Estimation (MINE [R1]).


The **Tab.R2** of evaluating mixing unlearned data in the clean dataset on CIFAR10: USR=1%, ASR=1%, and CSR=1%

| On CIFAR10       | MIA|  RA  | TA  |  RT | 8% |  
| --------         | --------    | -------- | -------- |  -------- |   -------- | 
| L2               | 73.89%      | 73.85%   |  73.78%  |  73.25%   |  73.03%    |  
| Cosine Similarity    | 9.40%       | 13.60%   | 33.40%   |  35.40%   |  43.26%    |  
| NT-Xent          | 73.89%      | 73.85%   |  73.78%  |  73.25%   |  73.03%    |  
| Mutual information   | 9.40%       | 13.60%   | 33.40%   |  35.40%   |  43.26%    |  


**Response to W3:** We greatly thank the Reviewer's insightful comments. We agree that the class centroid remains unchanged within $\delta$ distance after unlearning is too strong. 
Therefore, we revised the constraints inspired by existing unlearning methods, which set the constraints as the minimization term, minimizing the Kullback-Leibler Divergence (KLD) between the unlearned representation $z_{r,u}$ and original $z_r$ of the remaining samples. Iterating all the remaining samples is computationally expensive. We observe that in approximate unlearning, the most influenced samples are those samples similar to the erased sample. Therefore, we revised the constraints to preserve the local manifold representation of the top-k similar samples. We present the revision as follows.

```  
\textbf{Constraints to Preserve the Learned Manifold Representation Knowledge.} Blindly pushing unlearned samples away risks drastically altering the representation of all samples, leading to a collapse in overall model performance. We need to preserve the manifold representation knowledge $\texttt{z}_r$ for data in $S_r$. Existing methods minimize Kullback-Leibler Divergence (KLD) between $\texttt{z}_{r,u}$ and $\texttt{z}_r$ to ensure the new unlearned representation is not too different from the original learned one on the remaining dataset to preserve the knowledge \cite{fu2022knowledge}. It is straightforward and effective but needs to iterate all the remaining samples, which is computationally expensive. We observe that in approximate unlearning, the most influenced samples are those samples similar to the erased sample. Therefore, we chose the top-k samples that were similar to the erased sample and set a knowledge-preserving goal for them to constrain the local manifold representation knowledge. 
\begin{equation} \label{constraint}
	\small
	\min_{\theta_u} \| f_{\theta_u}(x_i) - \frac{1}{ \|S_k\|} \sum_{i \in S_k} \texttt{z}_i \|^2, 
\end{equation}
where the $S_k$ is the set of top-k similar samples nearest the erased sample $x_u \in S_u$.

```  

**Response to W4:** We sincerely thank the Reviewer's comments and suggestions about conducting experiments on larger datasets and larger unlearning samples size.
We demonstrate new experimental results as follows.


The **Tab.R3** of evaluating mixing unlearned data in the clean dataset on CIFAR10: USR=1%, ASR=1%, and CSR=1%

| On MNIST       | 200|  400  | 600  |  800 | 1000 |  
| --------         | --------    | -------- | -------- |  -------- |   -------- | 
| Retraining        | 73.89%      | 73.85%   |  73.78%  |  73.25%   |  73.03%    |  
| GA                | 9.40%       | 13.60%   | 33.40%   |  35.40%   |  43.26%    |  
| VBU              | 73.89%      | 73.85%   |  73.78%  |  73.25%   |  73.03%    |  
| RFU             | 9.40%       | 13.60%   | 33.40%   |  35.40%   |  43.26%    |  
| TCU            | 9.40%       | 13.60%   | 33.40%   |  35.40%   |  43.26%    |  



| On CIFAR10       | 200|  400  | 600  |  800 | 1000 |  
| --------         | --------    | -------- | -------- |  -------- |   -------- | 
| Retraining        | 73.89%      | 73.85%   |  73.78%  |  73.25%   |  73.03%    |  
| GA                | 9.40%       | 13.60%   | 33.40%   |  35.40%   |  43.26%    |  
| VBU              | 73.89%      | 73.85%   |  73.78%  |  73.25%   |  73.03%    |  
| RFU             | 9.40%       | 13.60%   | 33.40%   |  35.40%   |  43.26%    |  
| TCU            | 9.40%       | 13.60%   | 33.40%   |  35.40%   |  43.26%    |  


| On ImageNet       | 200|  400  | 600  |  800 | 1000 |  
| --------         | --------    | -------- | -------- |  -------- |   -------- | 
| Retraining        | 73.89%      | 73.85%   |  73.78%  |  73.25%   |  73.03%    |  
| GA                | 9.40%       | 13.60%   | 33.40%   |  35.40%   |  43.26%    |  
| VBU              | 73.89%      | 73.85%   |  73.78%  |  73.25%   |  73.03%    |  
| RFU             | 9.40%       | 13.60%   | 33.40%   |  35.40%   |  43.26%    |  
| TCU            | 9.40%       | 13.60%   | 33.40%   |  35.40%   |  43.26%    |  



**Response to W5:** We add a discussion about the mentioned references in related work.
We present the revision as follows.

There are also some studies that implement unlearning by focusing on representation and model weights. In \cite{shahlow}, they targeted the class unlearning problem based on sparse representation learned by Discrete Key-Value Bottleneck \cite{trauble2023discrete}. \citeauthor{guo2022efficient} proposed unlearning methods to forget the attribute instead of samples based on feature representations \cite{guo2022efficient}. The study of \citep{fan2024salun} directed the attention of machine unlearning toward specific model weights rather than the entire model, trying to improve effectiveness and efficiency. Different from them, our unlearning method is based on a representation with maximum manifold representation capacity.



**Response to Q2:** The centroid correlation is how correlated the centroids of different class are, which can be measured using pairwise inner products between normalized class centroids.


**Response to Q3:** "Model-centric" means considering the unlearning from model' persipective.


**Response to Q4:** In the new definition, we push the unlearned samples from it's original representation and only choose top-k nearest remaining samples to preserving the remaining knowledge.
We test the performance on the generative models as follows.


**Response to Q5:** Algorithm 1 is an example. During experiments, we set the size of mini-batch = 16 on MNIST and CIFAR10.




## Rebuttal to Reviewer PjaX  





**Response to W1:** We appreciate the Reviewer's comments, our key contribution is investigate machine unlearning solely through the model learned manifold representation perspective, and proposed the corresponding solution. The representation method, MMCRs, maximum manifold capacity representations, is an observation that we find the representation with good manifold capacity can enhance unlearning effect and efficiency.




  
**Response to W2:** We appreciate the Reviewer's comments. 
We supplemented why TCU is superior to gradient ascent based methods. 
Moreover, in a sphere manifold representation, we prove that unlearning a orthogonal representation is the longest distance for unlearning, which can also be explain from the mutual information's perspective.


**Response to W3:** 


The **Tab.R3** of evaluating mixing unlearned data in the clean dataset on CIFAR10: USR=1%, ASR=1%, and CSR=1%

| On MNIST       | 200|  400  | 600  |  800 | 1000 |  
| --------         | --------    | -------- | -------- |  -------- |   -------- | 
| Retraining        | 73.89%      | 73.85%   |  73.78%  |  73.25%   |  73.03%    |  
| GA                | 9.40%       | 13.60%   | 33.40%   |  35.40%   |  43.26%    |  
| VBU              | 73.89%      | 73.85%   |  73.78%  |  73.25%   |  73.03%    |  
| RFU             | 9.40%       | 13.60%   | 33.40%   |  35.40%   |  43.26%    |  
| TCU            | 9.40%       | 13.60%   | 33.40%   |  35.40%   |  43.26%    |  



| On CIFAR10       | 200|  400  | 600  |  800 | 1000 |  
| --------         | --------    | -------- | -------- |  -------- |   -------- | 
| Retraining        | 73.89%      | 73.85%   |  73.78%  |  73.25%   |  73.03%    |  
| GA                | 9.40%       | 13.60%   | 33.40%   |  35.40%   |  43.26%    |  
| VBU              | 73.89%      | 73.85%   |  73.78%  |  73.25%   |  73.03%    |  
| RFU             | 9.40%       | 13.60%   | 33.40%   |  35.40%   |  43.26%    |  
| TCU            | 9.40%       | 13.60%   | 33.40%   |  35.40%   |  43.26%    |  





## Rebuttal to Reviewer zJG2


 
 
**Response to C1:**

Gradient ascent based on the penalty of task and the representation distance is different.
We provide a proof of why unlearning based manifold representation is better than gradient reverse based on task from a mutual information perspective.

**Response to C2:**

Experiments on sample wise

| On MNIST       | 200|  400  | 600  |  800 | 1000 |  
| --------         | --------    | -------- | -------- |  -------- |   -------- | 
| Retraining        | 73.89%      | 73.85%   |  73.78%  |  73.25%   |  73.03%    |  
| GA                | 9.40%       | 13.60%   | 33.40%   |  35.40%   |  43.26%    |  
| VBU              | 73.89%      | 73.85%   |  73.78%  |  73.25%   |  73.03%    |  
| RFU             | 9.40%       | 13.60%   | 33.40%   |  35.40%   |  43.26%    |  
| TCU            | 9.40%       | 13.60%   | 33.40%   |  35.40%   |  43.26%    |  




Experiments on generative model

| On MNIST       | 200|  400  | 600  |  800 | 1000 |  
| --------         | --------    | -------- | -------- |  -------- |   -------- | 
| Retraining        | 73.89%      | 73.85%   |  73.78%  |  73.25%   |  73.03%    |  
| GA                | 9.40%       | 13.60%   | 33.40%   |  35.40%   |  43.26%    |  
| VBU              | 73.89%      | 73.85%   |  73.78%  |  73.25%   |  73.03%    |  
| RFU             | 9.40%       | 13.60%   | 33.40%   |  35.40%   |  43.26%    |  
| TCU            | 9.40%       | 13.60%   | 33.40%   |  35.40%   |  43.26%    |  



**Response to C3:**


We observed this phenomenon, the intra-class compact representation manifold can increase the unlearning effectiveness. Actually, our method without the intra-class compact limitation can also performs well.


**Response to R1:**

We added a discussion about these references


**Response to W1:**
In Eq. 3, we use $c_{y,i}$ to denote the centroid of the original $$z_{i,o}$$.
However, after reading reviews, we find the pushing away from original center maybe not that suitable.
We change $c_{y,i}$ as $z_{u,o}$, hoping to pushing away from the erased sample's original representation, which is more reasonable.
 

**Response to W3:**
Yes, the Trojan samples are the $S_u$, which need to be unlearned, eliminating backdoors.
 


**Response to Q1:**
Thank you for your questions, we additionally conduct experiments on a new samples size scala, ranging from 200 to 1000.
1000 sample is around 2% of the original training dataset. Although other methods effective on unlearning 20% of training dataset, them heavily rely on fine tuning on the remaining dataset.



**Response to Q2:**
Thank you for your insightful questions, we agree that the mean of a class might be less critical, as unlearning few samples maybe not heavily influence the whole class.
After discussion with experts in this area, we now realize that the most influenced samples are the samples most similar to the erased sample, i.e., the samples with the nearest representation to the erased samples' representation.
Therefore, we revised the constraints using the top-k nearest similar samples' center as the anchor.



**Response to Q3:**
We thank the Reviewers' comments. And we agree that unlearning multimodal models is also important. However, we have not conducted experiments on VLMs. 
We believe that our solution at least provide the some insights for unlearning of the representation on VLMs. And it will be our future work for propose a good unlearning solution for VLMs.


**Response to Q4:**
We conducted the experiments on generative model such as VAE as follows.





**Response to Q5:**
We thank the Reviewer's insightful question. The margin value alpha is the distance that we at least push the unlearned sample from the original manifold representation.
Larger alpha means better unlearning effectiveness, while degrading the model utility at the same time.






## Rebuttal to Reviewer YutA


**Response to W1:**
We observed this phenomenon, the intra-class compact representation manifold can increase the unlearning effectiveness. Actually, our method without the intra-class compact limitation can also performs well.

The contributions of this paper include: 1) the new viewpoint of unlearning; 2) the new solution from this new viewpoint and the theoretical advantage of unlearning based on representation than traditional gradient inverse.
3) the extensive experiments on validate the effectiveness of our solution.

We provide the new theoretical analysis of the advantage of unlearning based on representation rather than task-related gradient inverse.


**Response to W2:**
We thank the reviewer's comments. We added new experiments on larger dataset and with larger unlearning samples size.
The results are presented as follows.



**Response to W3:** 
We thank the Reviewer's comments. We provide a theoretical derivation from the sphere representation and mutual information perspective, which indicates the longest unlearning distance.


**Response to W4:**
Thank you for your comments. We revised and checked the paper to fix these typos. 











