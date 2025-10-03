# Embedding-Aware Noise Modeling of Quantum Annealing

**Seon-Geun Jeong<sup>1</sup>, Mai Dinh Cong<sup>1</sup>, Dae-Il Noh<sup>2</sup>, Quoc-Viet Pham<sup>3</sup> and Won-Joo Hwang<sup>1,4,*</sup>**

<sup>1</sup> Department of Information Convergence Engineering, Pusan National University, Busan, Republic of Korea  
<sup>2</sup> Quantum AI Lab, Korea Quantum Computing Co., Ltd, Busan, Republic of Korea  
<sup>3</sup> School of Computer Science and Statistics, Trinity College Dublin, Dublin, Ireland  
<sup>4</sup> School of Computer Science and Engineering, Pusan National University, Busan, Republic of Korea  

**Keywords:** Quantum Annealing, embedding, chain, chain strength, integrated control error model  

---

## 📝 Abstract
> Quantum annealing provides a practical realization of adiabatic quantum computation and has emerged as a promising approach for solving large-scale combinatorial optimization problems. However, current devices remain constrained by sparse hardware connectivity, which requires embedding logical variables into chains of physical qubits. This embedding overhead limits scalability and reduces reliability as longer chains are more prone to noise-induced errors. In this work, building on the known structural result that the average chain length in clique embeddings grows linearly with the problem size, we develop a mathematical framework that connects embedding-induced overhead with hardware noise in D-Wave’s Zephyr topology. Our analysis derives closed-form expressions for chain-break probability and chain-break fraction under a Gaussian control error model, establishing how noise scales with embedding size and how chain strength should be adjusted with chain length to maintain reliability. Experimental results from the Zephyr topology-based quantum processing unit confirm the accuracy of these predictions, demonstrating both the validity of the theoretical noise model and the practical relevance of the derived scaling rule. Beyond validating a theoretical model against hardware data, our findings establish a general embedding-aware noise framework that explains the trade-off between chain stability and logical coupler fidelity. Our framework advances the understanding of noise amplification in current devices and provides quantitative guidance for embedding-aware parameter tuning strategies.
