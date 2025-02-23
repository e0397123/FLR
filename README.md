# FLR
Repository for AAAI2025 Paper - Aligning Language Models Using Follow-up Likelihood as Reward Signal

## RM Evaluation / Output Demo Data
Please find the data at [here](https://drive.google.com/drive/folders/15qrklDPHXcNEowHi-RWR7z9WlEtdPMax?usp=sharing); 

(1) For pairwise preference benchmarks, A file name containing "accept" indicates the response is preferred while a file name containing "reject" means the response is not preferred. An "accept" file has a one-to-one data entry mapping to its corresponding "reject" file.
(2) The last word in a file name, such as "relevance", or "engagingness", etc, indicates the probabilities for follow-up utterances in those sub-categories of helpfulness. 
(3) For FLASK, MT-Bench-Single-Score, Feedback_Bench, and HelpSteer, the ground-truth overall scores are included in each data entry, and we can conduct correlation analysis for these four benchmarks.


