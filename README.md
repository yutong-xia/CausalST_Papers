

# 📚 Causality meets ST Data - Awesome Papers
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

This is a curated collection of papers on the intersection of causality (including causal inference and causal discovery), spatio-temporal data (including spatio-temporal graph/series data, grid data, and trajectory data), and machine learning. For clarity, irregular grid data is categorized as spatio-temporal graph data here. In addition, some papers on multivariate time series, which share a similar data structure with spatio-temporal series, are also included.

Since the collection is curated by an individual, some important papers might have been unintentionally missed. **Your contributions are greatly appreciated!** Feel free to suggest additional papers through ``Issues`` or ``Pull Requests``.

# Table of Contents
- [🔍 Survey & Tutorial](#-survey--tutorial)
	- [Survey](#survey)
	- [Tutorial](#tutorial)
- [📄 Papers](#-papers)
	- [Causal Inference](#causal-inference)
		- [Spatio-Temporal Graphs](#spatio-temporal-graphsmultivariate-time-series)
		- [Spatio-Temporal Grid](#spatio-temporal-grid)
		- [Trajectory](#trajectory)
	- [Causal Discovery](#causal-discovery)
		- [Spatio-Temporal Graphs](#spatio-temporal-graphsmultivariate-time-series-1)
		- [Spatio-Temporal Grid](#spatio-temporal-grid-1)
	- [LLMs and ST Causality](#large-language-models-and-st-causality)
   
# Survey & Tutorial

## Survey
- **[2024]** Applying Causal Machine Learning to Spatiotemporal Data Analysis: An Investigation of Opportunities and Challenges [[pdf]](https://ieeexplore.ieee.org/abstract/document/11119537) (Causal machine learning, Spatio-temporal data)
 - **[2024]** Causality for Earth Science -- A Review on Time-series and Spatiotemporal Causality Methods [[pdf]](https://arxiv.org/abs/2404.05746v1) (Causal inference, Causal discovery, Spatio-temporal data)
 - **[2023]** When Graph Neural Network Meets Causality: Opportunities, Methodologies and An Outlook [[pdf]](https://arxiv.org/pdf/2312.12477) (Causality, Graph Neural Network)
 - **[2023]** Data-Driven Causal Effect Estimation Based on Graphical Causal Modelling: A Survey [[pdf]](https://arxiv.org/abs/2208.09590)  (Causal inference)
 - **[2023]** Causal Discovery from Temporal Data: An Overview and New Perspectives [[pdf]](https://arxiv.org/abs/2303.10112) (Causal discovery, Time series)
 - **[2023]** A Survey on Causal Discovery Methods for I.I.D. and Time Series Data [[pdf]](https://arxiv.org/abs/2303.15027) (Causal discovery, Time series)
 - **[2022]** Causal Machine Learning: A Survey and Open Problems [[pdf]](https://arxiv.org/abs/2206.15475) (CausalML - causal supervised learning, causal generative modeling, causal explanations, causal fairness, causal reinforcement learning)
 - **[2022]** Survey and Evaluation of Causal Discovery Methods for Time Series [[pdf]](https://www.jair.org/index.php/jair/article/view/13428) (Causal Discovery, Time Series)
 - **[2021]** D'ya like DAGs? A Survey on Structure Learning and Causal Discovery [[pdf]](https://arxiv.org/abs/2103.02582) (Causal discovery)

## Tutorial
- **[SIGSPATIAL'24]** Tutorial on Causal Inference with Spatiotemporal Data [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3681778.3698786) [[code]](https://github.com/SaharaAli16/spatiotemporal-causality/tree/main/stcausal2024)
- **[KDD'23]** Causal Discovery from Temporal Data [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3580305.3599552) [[website]](https://chaunceykung.github.io/temporal-causal-discovery-tutorial/)
- **[KDD'21]** Causal Inference from Network Data [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3447548.3470795)
- **[KDD'21]** Causal Inference and Machine Learning in Practice with EconML and CausalML: Industrial Use Cases at Microsoft, TripAdvisor, Uber [[pdf]](https://dl.acm.org/doi/10.1145/3447548.3470792) 



# Papers

##  Causal Inference

###  Spatio-Temporal Graphs/Multivariate Time Series
- **[KDD'25]** Seeing the Unseen: Learning Basis Confounder Representations for Robust Traffic Prediction [[pdf]](https://dl.acm.org/doi/10.1145/3690624.3709201) [[code]](https://github.com/bigscity/STEVE_CODE) (Traffic prediction, Confounder representation)
- **[NeurIPS'25]** Causal Spatio-Temporal Prediction: An Effective and Efficient Multi-Modal Approach [[pdf]](https://arxiv.org/pdf/2505.17637) (Multi-Model, GCN, Mamba)
- **[CIKM'25]** Solar Forecasting with Causality: A Graph-Transformer Approach to Spatiotemporal Dependencies [[pdf]](https://doi.org/10.1145/3746252.3760905) (Solar forecasting, Graph Transformer)
- **[Neural Networks'25]** Enhancing Multivariate Spatio-Temporal Forecasting via Complete Dynamic Causal Modeling [[pdf]](https://doi-org.libproxy1.nus.edu.sg/10.1016/j.neunet.2025.107826) (Dynamic causal modeling, Latent confounders, Variational inference)
- **[arXiv'25]** Spatiotemporal Causal Decoupling Model for Air Quality Forecasting [[pdf]](https://arxiv.org/pdf/2505.20119) [[code]](https://github.com/PoorOtterBob/AirCade) (Air quality forecasting, Causal decoupling, Causal intervention)
- **[ICCBR'24 Workshop]** Spatio-Temporal Graph Neural Network with Hidden Confounders for Causal Forecast [[pdf]](https://ceur-ws.org/Vol-3708/paper_13.pdf) [[code]](https://github.com/xinxinluo123/CISTGNN) (Forecasting, Hidden confounders, Graph neural networks, Spatio-temporal dependencies)
- **[ICLR'24]** Causality-Inspired Spatial-Temporal Explanations for Dynamic Graph Neural Networks [[pdf]](https://openreview.net/pdf?id=AJBkfwXh3u)  [[code]](https://github.com/kesenzhao/DyGNNExplainer) (Node and graph classification, Back-door adjustment)
- **[AAAI'24]** Instrumental Variable Estimation for Causal Inference in Longitudinal Data with Time-Dependent Latent Confounders [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/29029) (Instrumental variable, Causal effect estimation)
- **[CIKM'24]** Causality-Aware Spatiotemporal Graph Neural Networks for Spatiotemporal Time Series Imputation [[pdf]](https://dl.acm.org/doi/10.1145/3627673.3679642) (Imputation, Front-door adjustment, Causal attention)
- **[WSDM ’24]** CityCAN: Causal Attention Network for Citywide Spatio-Temporal Forecasting [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3616855.3635764) (Forecasting, Causal attention)
- **[BigData'24]** Spatiotemporal Learning With Decoupled Causal Attention for Multivariate Time Series][[pdf]](https://ieeexplore.ieee.org/abstract/document/10753616/?casa_token=AFgShF4Y9zMAAAAA:GDdSc8jAHhjTQTAhVbNA5WCMVSQ_Eq5tXmp_k2-EzbWx5MvoDT46g-Smz2woJ_DpRKGZII2SnrW_) (Forecasting, Causal attention)
- **[ICCBR'24]** Spatio-Temporal Graph Neural Network with Hidden Confounders for Causal Forecast [[pdf]](https://ceur-ws.org/Vol-3708/paper_13.pdf) [[code]](https://github.com/xinxinluo123/CISTGNN) (Forecasting, Back-door criterion)
- **[arXiv'24]** Causally-Aware Spatio-Temporal Multi-Graph Convolution Network for Accurate and Reliable Traffic Prediction [[pdf]](https://arxiv.org/abs/2408.13293) (Forecasting, Uncertainty quantification, Information fusion)
- **[NeurIPS'23]** Deciphering Spatio-Temporal Graph Forecasting: A Causal Lens and Treatment [[pdf]](https://proceedings.neurips.cc/paper_files/paper/2023/file/74fa3651b41560e1c7555e0958c70333-Paper-Conference.pdf)  [[code]](https://github.com/yutong-xia/CaST) (Forecasting, Back-door adjustment, Front-door adjustment)
- **[AAAI'23]** Spatio-temporal neural structural causal models for bike flow prediction [[pdf]](https://arxiv.org/abs/2301.07843) [[code]](https://github.com/EternityZY/STNSCM) (Forecasting, Front-door criterion)
- **[AAAI'23]** Causal conditional hidden Markov model for multimodal traffic prediction [[pdf]](https://dl.acm.org/doi/abs/10.1609/aaai.v37i4.25619)  [[code]](https://github.com/EternityZY/CCHMM) (Conditional Markov Process, Multimodal data)
- **[KDD'23]** Generative Causal Interpretation Model for Spatio-Temporal Representation Learning [[pdf]](https://doi.org/10.1145/3580305.3599363) [[code]](https://github.com/EternityZY/GCIM) (VAE, Forecasting, ICA Theory)
- **[ICMLA'23]** Quantifying Causes of Arctic Amplification via Deep Learning based Time-series Causal Inference [[pdf]](https://ieeexplore.ieee.org/abstract/document/10460053?casa_token=jYfB7zIzGm4AAAAA:XL9juR0uSrH1bs1UxGih-nn_MshDHMffsKYv5byHZKGadD2IE4mnqngEYFQHy32wPv6UII9EPS4_)  [[code]](https://github.com/iharp-institute/causality-for-arctic-amplification/tree/main/tcinet-icmla2023) (Counterfactual prediction, Earth science)
- **[AAAG'23]** Spatiotemporal Heterogeneities in the Causal Effects of Mobility Intervention Policies during the COVID-19 Outbreak: A Spatially Interrupted Time-Series (SITS) Analysis [[pdf]](https://www.tandfonline.com/doi/full/10.1080/24694452.2022.2161986?scroll=top&needAccess=true) (Spatio-temporal heterogeneity, Mobile phone data, Mobility control policy)
- **[Nature Reviews Earth & Environment’23]** Causal inference for time series [[pdf]](https://www.nature.com/articles/s43017-023-00431-y?fromPaywallRec=false) [[code]](https://github.com/jakobrunge/tigramite/tree/master/tutorials/case_studies) (Earth science, Causal effect estimation, Causal discovery)
  

###  Spatio-Temporal Grid

- **[NeurIPS'24]** Causal Deciphering and Inpainting in Spatio-Temporal Dynamics via Diffusion Model [[pdf]](https://arxiv.org/pdf/2409.19608) (Diffusion model, Backdoor adjustment, Frontdoor adjustment)

- **[ICLR'24]** NuwaDynamics: Discovering and Updating in Causal Spatio-Temporal Modeling [[pdf]](https://openreview.net/pdf?id=sLdVl0q68X)  [[code]](https://github.com/easylearningscores/NuwaDynamics) (Ocean system, Back-door adjustion)

- **[ECML'24]** Estimating Direct and Indirect Causal Effects of Spatiotemporal Interventions in Presence of Spatial Interference [[pdf]](https://arxiv.org/pdf/2405.08174)  [[code]](https://github.com/iharp-institute/causality-for-arctic-amplification/tree/main/stcinet) (Treatment effects estimation, Spillover effects)
- **[ICML'22]** CITRIS: Causal Identifiability from Temporal Intervened Sequences [[pdf]](https://proceedings.mlr.press/v162/lippe22a/lippe22a.pdf) [[code]](https://github.com/phlippe/CITRIS) (Causal representations learning, Multidimensional causal factors, Image data)


###  Trajectory

- **[IJCAI'24]** Towards Robust Trajectory Representations: Isolating Environmental Confounders with Causal Learning [[pdf]](https://arxiv.org/abs/2404.14073) (Representation learning, Intervention learning, Back-door adjustment)

- **[Inf.Fusion'24]** Reliable trajectory prediction in scene fusion based on spatio-temporal Structure Causal Model [[pdf]](https://dl.acm.org/doi/10.1016/j.inffus.2024.102309) (Front-door criterion)

  

##  Causal Discovery

###  Spatio-Temporal Graphs/Multivariate Time Series

- **[AAAI'25]** SpaceTime: Causal Discovery from Non-Stationary Time Series [[pdf]](https://arxiv.org/pdf/2501.10235) [[code]]([https://github.com/jarrycyx/UNN](https://github.com/srhmm/spacetime)) (Minimum description length, Temporal causal graph, Kernelized discrepancy testing)
-   **[AAAI'24]** CUTS+: High-Dimensional Causal Discovery from Irregular Time-Series [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/29034) [[code]](https://github.com/jarrycyx/UNN) (Irregular sampling, Granger causality)
-   **[ICLR'24]** CausalTime: Realistically Generated Time-series for Benchmarking of Causal Discovery [[pdf]](https://openreview.net/pdf?id=iad1yyyGme) [[code]](https://github.com/jarrycyx/UNN) [[websit]](https://www.causaltime.cc/) (Benchmarks, Time series)
-  **[ICML'24]** Causal Discovery via Conditional Independence Testing with Proxy Variables [[pdf]](https://arxiv.org/pdf/2305.05281) [[code]](https://github.com/lmz123321/proxy_causal_discovery) (Time series, Proxy causal learning, Conditional independence testing)
- **[TNNLS ’24]** Dynamic Causal Explanation Based Diffusion-Variational Graph Neural Network for Spatio-temporal Forecasting [[pdf]](https://ieeexplore.ieee.org/abstract/document/10589693) [[code]](https://github.com/gorgen2020/DVGNN) (Forecasting, Diffusion process)
-   **[ICLR'23]**  CUTS: Neural Causal Discovery from Irregular Time-Series Data [[pdf]](https://openreview.net/forum?id=UG8bQcD3Emv) [[code]](https://github.com/jarrycyx/UNN) (EM-Style, Imputation, Irregular temporal data)
-   **[NeurIPS'23]** Causal Discovery from Subsampled Time Series with Proxy Variables [[pdf]](https://arxiv.org/pdf/2305.05276) [[code]](https://github.com/lmz123321/proxy_causal_discovery) (Time series, Subsampling)
- **[AAAI'23]** Causal Recurrent Variational Autoencoder for Medical Time Series Generation [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/26031) [[code]](https://github.com/hongmingli1995/CR-VAE) (CR-VAE, Generative model)
- **[CIKM ’23]** STREAMS: Towards Spatio-Temporal Causal Discovery with Reinforcement Learning for Streamflow Rate Prediction [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3583780.3614719?casa_token=x2ZGXSpdv2oAAAAA:0TgQHB4OROEtXfzClY89QvfJ0aFXwviASBeNCjSTEnEaQlYhaaCZ0hEYQLuGBzte3-CkaB7pbWYB)  [[code]](https://github.com/paras2612/STREAMS/tree/main) (Forecasting, Reinforcement Learning)
- **[CIKM ’23]** Causal Discovery in Temporal Domain from Interventional Data [[pdf]](https://dl.acm.org/doi/10.1145/3583780.3615177)[[code]](https://github.com/lpwpower/TECDI) (Temporal reasoning, Multivariate time series)
-   **[CIKM'22]** Nonlinear Causal Discovery in Time Series [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3511808.3557660) (Functional Causal Model, Time series, Non-stationary data)
- **[CLeaR'22]** Amortized Causal Discovery: Learning to Infer Causal Graphs from Time-Series Data [[pdf]](https://proceedings.mlr.press/v177/lowe22a)[[code]](https://github.com/loeweX/AmortizedCausalDiscovery) (Granger Causality, Noisy Observations)
-   **[ICLR'21]** Interpretable Models for Granger Causality Using Self-explaining Neural Networks [[pdf]](https://arxiv.org/abs/2101.07600) [[code]](https://github.com/i6092467/GVAR) (Granger causality, Interpretable models)
 - **[ICML'21]** Necessary and sufficient conditions for causal feature selection in time series with latent common causes [[pdf]](https://proceedings.mlr.press/v139/mastakouri21a.html) (Causal feature selection)
- **[PAMI'21]** Neural Granger Causality [[pdf]](https://ieeexplore.ieee.org/abstract/document/9376668) [[code]](https://github.com/iancovert/Neural-GC) (Structured sparsity, Interpretability, Granger causality)
- **[NeurIPS'20]** High-recall causal discovery for autocorrelated time series with latent confounders [[pdf]](https://proceedings.neurips.cc/paper/2020/hash/94e70705efae423efda1088614128d0b-Abstract.html) [[code]](https://github.com/jakobrunge/tigramite) (Tigramite - Benchmark and python package, High-dimensional time series) 
-  **[ICML'20]** CAUSE: Learning Granger Causality from Event Sequences using Attribution Methods [[pdf]](https://proceedings.mlr.press/v119/zhang20v.html) [[code]](https://github.com/razhangwei/CAUSE) (Granger causality)
-   **[AISTATS'20]** DYNOTEARS: Structure Learning from Time-Series Data [[pdf]](http://proceedings.mlr.press/v108/pamfil20a/pamfil20a.pdf) (Dynamic Bayesian networks, Structure learning, Time series, Acyclicity constraint)
-   **[UAI'20]** Discovering Contemporaneous and Lagged Causal Relations in Autocorrelated Nonlinear Time Series Datasets [[pdf]](https://proceedings.mlr.press/v124/runge20a.html) [[code]](https://github.com/jakobrunge/tigramite) (Time series, Autocorrelation)
- **[Chaos'18]** Causal network reconstruction from time series: From theoretical assumptions to practical estimation [[pdf]](https://pubs.aip.org/aip/cha/article/28/7/075310/386353/Causal-network-reconstruction-from-time-series) [[code]](https://github.com/jakobrunge/tigramite) (Granger causality, Causal Markov condition)
- **[BigData'17]** pg-Causality: Identifying Spatiotemporal Causal Pathways for Air Pollutants with Urban Big Data [[pdf]](https://ieeexplore.ieee.org/abstract/document/7970191?casa_token=7lkgIeczDeAAAAAA:3r1LnamcYI2x2_n7OBZgcB62wz5Qnvkkp1JL71hHVj9otjPFIjFP2c34XD28BJpi6qa0_QPk1J8O) (Bayesian learning, Causal pathways, Air Quality)
    

###  Spatio-Temporal Grid
- **[ICML'25]** Discovering Latent Structural Causal Models from Spatio-Temporal Data [[pdf]](https://arxiv.org/abs/2411.05331) [[code]](https://github.com/Rose-STL-Lab/SPACY/) (Climate data, SPACY)
- **[JGR: MLC'25]** Space-Time Causal Discovery in Earth System Science: A Local Stencil Learning Approach [[pdf]](https://doi.org/10.1029/2024JH000546) (CaStLe, Causal dynamical structure learning)
- **[KDD'23]** Generative Causal Interpretation Model for Spatio-Temporal Representation Learning [[pdf]](https://doi.org/10.1145/3580305.3599363) [[code]]([https://anonymous.4open.science/r/spacy-572B/readme.md](https://github.com/EternityZY/GCIM)) (Casual Interpretation)
- **[BigData'22]** A spatio-temporal causal discovery framework for hydrological systems [[pdf]](https://ieeexplore.ieee.org/abstract/document/10020845?casa_token=Z55rZssq0DMAAAAA:LFwBbozUEBeV22D4nZMB-PTg5bLhUX5KWkT8xDpymn9Ha0jDMgaBvU1IUOdOjPEgIp4_5eBM0euk) [[code]](https://github.com/paras2612/STCD) (Hydrological systems)
- **[Environmental Data Science'22]** A spatiotemporal stochastic climate model for benchmarking causal discovery methods for teleconnections [[pdf]](https://www.cambridge.org/core/journals/environmental-data-science/article/spatiotemporal-stochastic-climate-model-for-benchmarking-causal-discovery-methods-for-teleconnections/0E066B8813BA2281D2B95279EF3272B4)  [[code]](https://github.com/xtibau/mapped_pcmci)  (Climate data, Teleconnections)
- **[Nature Communications'15]** Identifying causal gateways and mediators in complex spatio-temporal systems [[pdf]](https://www.nature.com/articles/ncomms9502) (Atmospheric dynamics, Complex systems, Causal effect estimation)

## Large Language Models and ST Causality

- **[IEEE Transactions on Cybernetics'25]** Causal Intervention Is What Large Language Models Need for Spatio-Temporal Forecasting. [[pdf]](https://ieeexplore.ieee.org/abstract/document/11017752)  [[code]](https://github.com/lishijie15/STCInterLLM) (Causal Intervention Encoder, Chain-of-Action Prompting for LLMs, Spatio-Temporal Adaptive Graphs)
- **[arXiv'25]** Augur: Modeling Covariate Causal Associations in Time Series via Large Language Models. [[pdf]](https://arxiv.org/abs/2510.07858) (Time Series, Causal Discovery, LLM-guided causal structure induction, Forecasting)
- **[arXiv'25]** Reimagining urban science: Scaling causal inference with large language models [[pdf]](https://arxiv.org/abs/2510.07858) (Urban Data, AI Scientiest, LLM-based Agent)
