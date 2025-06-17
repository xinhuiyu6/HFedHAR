# HFedHAR

This is code for our paper: A heterogeneous federated learning framework for human activity recognition

## 📝 Abstract
To address the dual heterogeneity (i.e., model and data heterogeneity) in FedHAR, we propose HFedHAR. Under model heterogeneity, clients use different model structures, making gradient-based knowledge sharing ineffective in existing FedHAR approaches. To address this, we first design a new `bridge' (i.e., synthetic data generated from clients with high computational power), to link clients. Each client represents their knowledge as prediction logits on the bridge, enabling structure-agnostic knowledge sharing. To further enhance the bridge's effectiveness, we introduce an information entropy loss. To tackle data heterogeneity, we employ a similarity-based knowledge distillation strategy based on a relation graph constructed among clients, enabling each client to effectively absorb knowledge from others.


## 📊 Datasets
We use four HAR datasets:
- PAMAP2
- UCI-HAR
- WISDM
- USC-HAD
  
Preprocessed data are available for download here: [https://drive.google.com/drive/folders/1Jc9ePhLU8rLcqHEP6eeD9tcSIHTr-8ng?usp=drive_link]
