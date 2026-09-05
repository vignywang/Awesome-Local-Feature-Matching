<h1 align="center">Awesome Local Feature Matching</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2401.17592"><img src="https://img.shields.io/badge/arXiv-2401.17592-b31b1b.svg?style=flat-square" alt="arXiv"></a>
  <a href="https://www.sciencedirect.com/science/article/pii/S1566253524001222"><img src="https://img.shields.io/badge/Paper-Information%20Fusion%202024-1672B8.svg?style=flat-square" alt="Paper"></a>
  <a href="https://github.com/vignywang/Awesome-Local-Feature-Matching/stargazers"><img src="https://img.shields.io/github/stars/vignywang/Awesome-Local-Feature-Matching?style=flat-square&amp;logo=github&amp;label=Stars" alt="GitHub Stars"></a>
  <a href="https://opensource.org/license/mit"><img src="https://img.shields.io/badge/License-MIT-22C55E.svg?style=flat-square" alt="MIT License"></a>
</p>

<p align="center">
  <strong><a href="https://scholar.google.com/citations?user=htmrWvUAAAAJ&amp;hl=zh-CN&amp;oi=ao">Shibiao Xu<sup>1</sup></a></strong> &nbsp;·&nbsp;
  <strong><a href="https://scholar.google.com/citations?user=azDgUMUAAAAJ&amp;hl=zh-CN&amp;oi=ao">Shunpeng Chen<sup>1</sup></a></strong> &nbsp;·&nbsp;
  <strong><a href="https://scholar.google.com/citations?user=_IUq7ooAAAAJ&amp;hl=zh-CN&amp;oi=ao">Rongtao Xu<sup>2</sup></a></strong> &nbsp;·&nbsp;
  <strong><a href="https://scholar.google.com/citations?user=DnJKQI8AAAAJ&amp;hl=zh-CN&amp;oi=ao">Changwei Wang<sup>2</sup></a></strong> &nbsp;·&nbsp;
  <strong><a href="https://openreview.net/profile?id=~Peng_Lu8">Peng Lu<sup>1</sup></a></strong> &nbsp;·&nbsp;
  <strong><a href="https://openreview.net/profile?id=~Li_Guo5">Li Guo<sup>1</sup></a></strong>
</p>

<p align="center">
  <sup>1</sup> School of Artificial Intelligence, Beijing University of Posts and Telecommunications, China<br>
  <sup>2</sup> The State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation, Chinese Academy of Sciences, China
</p>

<p align="center"><strong>✨ Information Fusion 2024 ✨</strong></p>

This is the official companion repository for the Information Fusion 2024 survey [Local feature matching using deep learning: A survey](https://www.sciencedirect.com/science/article/pii/S1566253524001222). It organizes deep learning-based local feature matching methods using the survey's detector-based and detector-free taxonomy. The README also tracks representative work published or accepted during 2024–2026, listed separately from the original survey coverage.

> Found an issue, missing paper, incorrect metadata, or broken link? Please open an [Issue](https://github.com/vignywang/Awesome-Local-Feature-Matching/issues) and mention [@chenshunpeng](https://github.com/chenshunpeng) so we can follow up.

## News

- **2026/09/05:** Added representative local feature matching papers from 2024–2026.
- **2024/03/03:** Our survey was accepted by Information Fusion.

## Table of Contents

- [News](#news)
- [Recent Advances (2024–Present)](#recent-advances-2024present)
  - [2026](#2026)
  - [2025](#2025)
  - [2024](#2024)
- [Survey Taxonomy](#survey-taxonomy)
  - [Detector-based Models](#detector-based-models)
    - [Detect-then-Describe](#detect-then-describe)
    - [Joint Detection and Description](#joint-detection-and-description)
    - [Describe-then-Detect](#describe-then-detect)
    - [Graph Based](#graph-based)
  - [Detector-free Models](#detector-free-models)
    - [CNN Based](#cnn-based)
    - [Transformer Based](#transformer-based)
    - [Patch Based](#patch-based)
- [Benchmarks and Evaluation](#benchmarks-and-evaluation)
- [Applications](#applications)
  - [Structure from Motion](#structure-from-motion)
  - [Remote Sensing Image Registration](#remote-sensing-image-registration)
  - [Medical Image Registration](#medical-image-registration)
- [Related Surveys](#related-surveys)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [License](#license)

<a id="recent-advances-2024present"></a>
## 🔥 Recent Advances (2024–Present)

A selective update beyond the survey, grouped by publication year.

### 2026

- **[SLiM](https://openaccess.thecvf.com/content/CVPR2026/papers/Choo_Scalable_Feature_Matching_via_State_Space_Modeling_and_Sparse_Correlation_CVPR_2026_paper.pdf)** ![CVPR 2026](https://img.shields.io/badge/CVPR-2026-2563EB?style=flat-square) — Scalable Feature Matching via State Space Modeling and Sparse Correlation [[paper]](https://openaccess.thecvf.com/content/CVPR2026/papers/Choo_Scalable_Feature_Matching_via_State_Space_Modeling_and_Sparse_Correlation_CVPR_2026_paper.pdf) [[code]](https://github.com/Band-127/SLiM)
- **[LoMa](https://arxiv.org/abs/2604.04931)** ![ECCV 2026](https://img.shields.io/badge/ECCV-2026-16A34A?style=flat-square) — Local Feature Matching Revisited [[paper]](https://arxiv.org/abs/2604.04931) [[venue record]](https://eccv.ecva.net/Conferences/2026/AcceptedPapers) [[code]](https://github.com/davnords/LoMa)
- **[RoMa v2](https://arxiv.org/abs/2511.15706)** ![ECCV 2026](https://img.shields.io/badge/ECCV-2026-16A34A?style=flat-square) — Harder Better Faster Denser Feature Matching [[paper]](https://arxiv.org/abs/2511.15706) [[venue record]](https://eccv.ecva.net/Conferences/2026/AcceptedPapers) [[code]](https://github.com/Parskatt/RoMaV2)
- **[TextFM](https://openaccess.thecvf.com/content/CVPR2026/html/Zheng_TextFM_Robust_Semi-dense_Feature_Matching_with_Language_Guidance_CVPR_2026_paper.html)** ![CVPR 2026](https://img.shields.io/badge/CVPR-2026-2563EB?style=flat-square) — Robust Semi-dense Feature Matching with Language Guidance [[paper]](https://openaccess.thecvf.com/content/CVPR2026/html/Zheng_TextFM_Robust_Semi-dense_Feature_Matching_with_Language_Guidance_CVPR_2026_paper.html)

### 2025

- **[EDM](https://openaccess.thecvf.com/content/ICCV2025/html/Li_EDM_Efficient_Deep_Feature_Matching_ICCV_2025_paper.html)** ![ICCV 2025](https://img.shields.io/badge/ICCV-2025-7C3AED?style=flat-square) — Efficient Deep Feature Matching [[paper]](https://openaccess.thecvf.com/content/ICCV2025/html/Li_EDM_Efficient_Deep_Feature_Matching_ICCV_2025_paper.html) [[arXiv]](https://arxiv.org/abs/2503.05122) [[code]](https://github.com/chicleee/EDM)
- **[JamMa](https://openaccess.thecvf.com/content/CVPR2025/html/Lu_JamMa_Ultra-lightweight_Local_Feature_Matching_with_Joint_Mamba_CVPR_2025_paper.html)** ![CVPR 2025](https://img.shields.io/badge/CVPR-2025-2563EB?style=flat-square) — Ultra-lightweight Local Feature Matching with Joint Mamba [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Lu_JamMa_Ultra-lightweight_Local_Feature_Matching_with_Joint_Mamba_CVPR_2025_paper.html) [[arXiv]](https://arxiv.org/abs/2503.03437) [[code]](https://github.com/leoluxxx/JamMa)
- **[L2M](https://openaccess.thecvf.com/content/ICCV2025/html/Liang_Learning_Dense_Feature_Matching_via_Lifting_Single_2D_Image_to_ICCV_2025_paper.html)** ![ICCV 2025](https://img.shields.io/badge/ICCV-2025-7C3AED?style=flat-square) — Learning Dense Feature Matching via Lifting Single 2D Image to 3D Space [[paper]](https://openaccess.thecvf.com/content/ICCV2025/html/Liang_Learning_Dense_Feature_Matching_via_Lifting_Single_2D_Image_to_ICCV_2025_paper.html) [[arXiv]](https://arxiv.org/abs/2507.00392) [[code]](https://github.com/Sharpiless/L2M)
- **[MATCHA](https://openaccess.thecvf.com/content/CVPR2025/html/Xue_MATCHA_Towards_Matching_Anything_CVPR_2025_paper.html)** ![CVPR 2025](https://img.shields.io/badge/CVPR-2025-2563EB?style=flat-square) — Towards Matching Anything [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Xue_MATCHA_Towards_Matching_Anything_CVPR_2025_paper.html) [[arXiv]](https://arxiv.org/abs/2501.14945) [[code]](https://github.com/feixue94/matcha)
- **[CasP](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_CasP_Improving_Semi-Dense_Feature_Matching_Pipeline_Leveraging_Cascaded_Correspondence_Priors_ICCV_2025_paper.html)** ![ICCV 2025](https://img.shields.io/badge/ICCV-2025-7C3AED?style=flat-square) — Improving Semi-Dense Feature Matching Pipeline Leveraging Cascaded Correspondence Priors for Guidance [[paper]](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_CasP_Improving_Semi-Dense_Feature_Matching_Pipeline_Leveraging_Cascaded_Correspondence_Priors_ICCV_2025_paper.html) [[arXiv]](https://arxiv.org/abs/2507.17312) [[code]](https://github.com/pq-chen/CasP)
- **[MINIMA](https://openaccess.thecvf.com/content/CVPR2025/html/Ren_MINIMA_Modality_Invariant_Image_Matching_CVPR_2025_paper.html)** ![CVPR 2025](https://img.shields.io/badge/CVPR-2025-2563EB?style=flat-square) — Modality Invariant Image Matching [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Ren_MINIMA_Modality_Invariant_Image_Matching_CVPR_2025_paper.html) [[arXiv]](https://arxiv.org/abs/2412.19412) [[code]](https://github.com/LSXI7/MINIMA)

### 2024

- **[Efficient LoFTR](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Efficient_LoFTR_Semi-Dense_Local_Feature_Matching_with_Sparse-Like_Speed_CVPR_2024_paper.html)** ![CVPR 2024](https://img.shields.io/badge/CVPR-2024-2563EB?style=flat-square) — Semi-Dense Local Feature Matching with Sparse-Like Speed [[paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Efficient_LoFTR_Semi-Dense_Local_Feature_Matching_with_Sparse-Like_Speed_CVPR_2024_paper.html) [[arXiv]](https://arxiv.org/abs/2403.04765) [[code]](https://github.com/zju3dv/EfficientLoFTR)
- **[XFeat](https://openaccess.thecvf.com/content/CVPR2024/html/Potje_XFeat_Accelerated_Features_for_Lightweight_Image_Matching_CVPR_2024_paper.html)** ![CVPR 2024](https://img.shields.io/badge/CVPR-2024-2563EB?style=flat-square) — Accelerated Features for Lightweight Image Matching [[paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Potje_XFeat_Accelerated_Features_for_Lightweight_Image_Matching_CVPR_2024_paper.html) [[arXiv]](https://arxiv.org/abs/2404.19174) [[code]](https://github.com/verlab/accelerated_features)
- **[MASt3R](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09080.pdf)** ![ECCV 2024](https://img.shields.io/badge/ECCV-2024-16A34A?style=flat-square) — Grounding Image Matching in 3D with MASt3R [[paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09080.pdf) [[arXiv]](https://arxiv.org/abs/2406.09756) [[code]](https://github.com/naver/mast3r)
- **[GIM](https://openreview.net/forum?id=NYN1b8GRGS)** ![ICLR 2024](https://img.shields.io/badge/ICLR-2024-EA580C?style=flat-square) — Learning Generalizable Image Matcher from Internet Videos [[paper]](https://openreview.net/forum?id=NYN1b8GRGS) [[arXiv]](https://arxiv.org/abs/2402.11095) [[code]](https://github.com/xuelunshen/gim)
- **[OmniGlue](https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_OmniGlue_Generalizable_Feature_Matching_with_Foundation_Model_Guidance_CVPR_2024_paper.html)** ![CVPR 2024](https://img.shields.io/badge/CVPR-2024-2563EB?style=flat-square) — Generalizable Feature Matching with Foundation Model Guidance [[paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_OmniGlue_Generalizable_Feature_Matching_with_Foundation_Model_Guidance_CVPR_2024_paper.html) [[arXiv]](https://arxiv.org/abs/2405.12979) [[code]](https://github.com/google-research/omniglue)
- **[DeDoDe](https://doi.org/10.1109/3DV62453.2024.00035)** ![3DV 2024](https://img.shields.io/badge/3DV-2024-0F766E?style=flat-square) — Detect, Don't Describe—Describe, Don't Detect for Local Feature Matching [[paper]](https://doi.org/10.1109/3DV62453.2024.00035) [[arXiv]](https://arxiv.org/abs/2308.08479) [[code]](https://github.com/Parskatt/DeDoDe)
- **[RCM](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05862.pdf)** ![ECCV 2024](https://img.shields.io/badge/ECCV-2024-16A34A?style=flat-square) — Raising the Ceiling: Conflict-Free Local Feature Matching with Dynamic View Switching [[paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05862.pdf) [[arXiv]](https://arxiv.org/abs/2407.07789) [[code]](https://github.com/leoluxxx/RCM)
- **[EcoMatcher](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08613.pdf)** ![ECCV 2024](https://img.shields.io/badge/ECCV-2024-16A34A?style=flat-square) — Efficient Clustering Oriented Matcher for Detector-free Image Matching [[paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08613.pdf)
- **[iMatching](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05586.pdf)** ![ECCV 2024](https://img.shields.io/badge/ECCV-2024-16A34A?style=flat-square) — Imperative Correspondence Learning [[paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05586.pdf) [[arXiv]](https://arxiv.org/abs/2312.02141) [[code]](https://github.com/sair-lab/iMatching)

<a id="survey-taxonomy"></a>
## 📚 Survey Taxonomy

The survey organizes deep local feature matching into two model families and seven subcategories. Entries are ordered from newer to older work, with venue metadata updated to the formal publication when available.

### Detector-based Models

<p align="center">
  <img src="figs/Detector-based.jpg" width="800" alt="Comparison of detector-based local feature pipelines"/>
</p>
<p align="center">
  Fig. 1. Detector-based pipelines: (a) Detect-then-Describe, (b) Joint Detection and Description, and (c) Describe-then-Detect.
</p>

#### Detect-then-Describe

- **[AWDesc](https://doi.org/10.1109/TPAMI.2023.3266728)** ![TPAMI 2023](https://img.shields.io/badge/TPAMI-2023-00629B?style=flat-square) — Attention Weighted Local Descriptors [[paper]](https://doi.org/10.1109/TPAMI.2023.3266728)
- **[S-TREK](https://openaccess.thecvf.com/content/ICCV2023/html/Santellani_S-TREK_Sequential_Translation_and_Rotation_Equivariant_Keypoints_for_Local_Feature_ICCV_2023_paper.html)** ![ICCV 2023](https://img.shields.io/badge/ICCV-2023-7C3AED?style=flat-square) — Sequential Translation and Rotation Equivariant Keypoints for Local Feature Extraction [[paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Santellani_S-TREK_Sequential_Translation_and_Rotation_Equivariant_Keypoints_for_Local_Feature_ICCV_2023_paper.html)
- **[ZippyPoint](https://openaccess.thecvf.com/content/CVPR2023W/IMW/papers/Kanakis_ZippyPoint_Fast_Interest_Point_Detection_Description_and_Matching_Through_Mixed_CVPRW_2023_paper.pdf)** ![CVPRW 2023](https://img.shields.io/badge/CVPRW-2023-60A5FA?style=flat-square) — Fast Interest Point Detection, Description, and Matching Through Mixed Precision Discretization [[paper]](https://openaccess.thecvf.com/content/CVPR2023W/IMW/papers/Kanakis_ZippyPoint_Fast_Interest_Point_Detection_Description_and_Matching_Through_Mixed_CVPRW_2023_paper.pdf) [[code]](https://github.com/menelaoskanakis/ZippyPoint)
- **[CNDesc](https://doi.org/10.1109/TMM.2022.3169331)** ![TMM 2023](https://img.shields.io/badge/TMM-2023-00629B?style=flat-square) — Cross Normalization for Local Descriptors Learning [[paper]](https://doi.org/10.1109/TMM.2022.3169331)
- **[ALIKE](https://doi.org/10.1109/TMM.2022.3155927)** ![TMM 2023](https://img.shields.io/badge/TMM-2023-00629B?style=flat-square) — Accurate and Lightweight Keypoint Detection and Descriptor Extraction [[paper]](https://doi.org/10.1109/TMM.2022.3155927) [[arXiv]](https://arxiv.org/abs/2112.02906) [[code]](https://github.com/Shiaoming/ALIKE)
- **[MTLDesc](https://ojs.aaai.org/index.php/AAAI/article/view/20138)** ![AAAI 2022](https://img.shields.io/badge/AAAI-2022-0369A1?style=flat-square) — Looking Wider to Describe Better [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/20138)
- **[KP2D](https://openreview.net/forum?id=XhrpFrneoIe)** ![ICLR 2020](https://img.shields.io/badge/ICLR-2020-EA580C?style=flat-square) — Neural Outlier Rejection for Self-Supervised Keypoint Learning [[paper]](https://openreview.net/forum?id=XhrpFrneoIe) [[arXiv]](https://arxiv.org/abs/1912.10615) [[code]](https://github.com/TRI-ML/KP2D)
- **[HyNet](https://proceedings.neurips.cc/paper/2020/file/52d2752b150f9c35ccb6869cbf074e48-Paper.pdf)** ![NeurIPS 2020](https://img.shields.io/badge/NeurIPS-2020-DC2626?style=flat-square) — Learning Local Descriptor with Hybrid Similarity Measure and Triplet Loss [[paper]](https://proceedings.neurips.cc/paper/2020/file/52d2752b150f9c35ccb6869cbf074e48-Paper.pdf) [[code]](https://github.com/yuruntian/HyNet)
- **[Key.Net](https://openaccess.thecvf.com/content_ICCV_2019/html/Barroso-Laguna_Key.Net_Keypoint_Detection_by_Handcrafted_and_Learned_CNN_Filters_ICCV_2019_paper.html)** ![ICCV 2019](https://img.shields.io/badge/ICCV-2019-7C3AED?style=flat-square) — Keypoint Detection by Handcrafted and Learned CNN Filters [[paper]](https://openaccess.thecvf.com/content_ICCV_2019/html/Barroso-Laguna_Key.Net_Keypoint_Detection_by_Handcrafted_and_Learned_CNN_Filters_ICCV_2019_paper.html) [[code]](https://github.com/axelBarroso/Key.Net)
- **[Log-Polar Descriptors](https://openaccess.thecvf.com/content_ICCV_2019/html/Ebel_Beyond_Cartesian_Representations_for_Local_Descriptors_ICCV_2019_paper.html)** ![ICCV 2019](https://img.shields.io/badge/ICCV-2019-7C3AED?style=flat-square) — Beyond Cartesian Representations for Local Descriptors [[paper]](https://openaccess.thecvf.com/content_ICCV_2019/html/Ebel_Beyond_Cartesian_Representations_for_Local_Descriptors_ICCV_2019_paper.html) [[code]](https://github.com/cvlab-epfl/log-polar-descriptors)
- **[SOSNet](https://openaccess.thecvf.com/content_CVPR_2019/html/Tian_SOSNet_Second_Order_Similarity_Regularization_for_Local_Descriptor_Learning_CVPR_2019_paper.html)** ![CVPR 2019](https://img.shields.io/badge/CVPR-2019-2563EB?style=flat-square) — Second Order Similarity Regularization for Local Descriptor Learning [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/html/Tian_SOSNet_Second_Order_Similarity_Regularization_for_Local_Descriptor_Learning_CVPR_2019_paper.html) [[code]](https://github.com/yuruntian/SOSNet)
- **[ContextDesc](https://openaccess.thecvf.com/content_CVPR_2019/html/Luo_ContextDesc_Local_Descriptor_Augmentation_With_Cross-Modality_Context_CVPR_2019_paper.html)** ![CVPR 2019](https://img.shields.io/badge/CVPR-2019-2563EB?style=flat-square) — Local Descriptor Augmentation with Cross-Modality Context [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/html/Luo_ContextDesc_Local_Descriptor_Augmentation_With_Cross-Modality_Context_CVPR_2019_paper.html) [[code]](https://github.com/lzx551402/contextdesc)
- **[GeoDesc](https://openaccess.thecvf.com/content_ECCV_2018/html/Zixin_Luo_Learning_Local_Descriptors_ECCV_2018_paper.html)** ![ECCV 2018](https://img.shields.io/badge/ECCV-2018-16A34A?style=flat-square) — Learning Local Descriptors by Integrating Geometry Constraints [[paper]](https://openaccess.thecvf.com/content_ECCV_2018/html/Zixin_Luo_Learning_Local_Descriptors_ECCV_2018_paper.html) [[code]](https://github.com/lzx551402/geodesc)
- **[HardNet](https://proceedings.neurips.cc/paper_files/paper/2017/file/831caa1b600f852b7844499430ecac17-Paper.pdf)** ![NeurIPS 2017](https://img.shields.io/badge/NeurIPS-2017-DC2626?style=flat-square) — Working Hard to Know Your Neighbor's Margins: Local Descriptor Learning Loss [[paper]](https://proceedings.neurips.cc/paper_files/paper/2017/file/831caa1b600f852b7844499430ecac17-Paper.pdf) [[code]](https://github.com/DagnyT/hardnet)
- **[L2-Net](https://openaccess.thecvf.com/content_cvpr_2017/html/Tian_L2-Net_Deep_Learning_CVPR_2017_paper.html)** ![CVPR 2017](https://img.shields.io/badge/CVPR-2017-2563EB?style=flat-square) — Deep Learning of Discriminative Patch Descriptor in Euclidean Space [[paper]](https://openaccess.thecvf.com/content_cvpr_2017/html/Tian_L2-Net_Deep_Learning_CVPR_2017_paper.html) [[code]](https://github.com/yuruntian/L2-Net)
- **[OriNet](https://openaccess.thecvf.com/content_cvpr_2016/html/Yi_Learning_to_Assign_CVPR_2016_paper.html)** ![CVPR 2016](https://img.shields.io/badge/CVPR-2016-2563EB?style=flat-square) — Learning to Assign Orientations to Feature Points [[paper]](https://openaccess.thecvf.com/content_cvpr_2016/html/Yi_Learning_to_Assign_CVPR_2016_paper.html)

#### Joint Detection and Description

- **[FeatureBooster](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_FeatureBooster_Boosting_Feature_Descriptors_With_a_Lightweight_Neural_Network_CVPR_2023_paper.html)** ![CVPR 2023](https://img.shields.io/badge/CVPR-2023-2563EB?style=flat-square) — Boosting Feature Descriptors with a Lightweight Neural Network [[paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_FeatureBooster_Boosting_Feature_Descriptors_With_a_Lightweight_Neural_Network_CVPR_2023_paper.html) [[code]](https://github.com/SJTU-ViSYS/FeatureBooster)
- **[SFD2](https://openaccess.thecvf.com/content/CVPR2023/html/Xue_SFD2_Semantic-Guided_Feature_Detection_and_Description_CVPR_2023_paper.html)** ![CVPR 2023](https://img.shields.io/badge/CVPR-2023-2563EB?style=flat-square) — Semantic-Guided Feature Detection and Description [[paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Xue_SFD2_Semantic-Guided_Feature_Detection_and_Description_CVPR_2023_paper.html) [[code]](https://github.com/feixue94/sfd2)
- **[RELF](https://openaccess.thecvf.com/content/CVPR2023/html/Lee_Learning_Rotation-Equivariant_Features_for_Visual_Correspondence_CVPR_2023_paper.html)** ![CVPR 2023](https://img.shields.io/badge/CVPR-2023-2563EB?style=flat-square) — Learning Rotation-Equivariant Features for Visual Correspondence [[paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Lee_Learning_Rotation-Equivariant_Features_for_Visual_Correspondence_CVPR_2023_paper.html)
- **[SeLF](https://ieeexplore.ieee.org/document/9829199)** ![TIP 2022](https://img.shields.io/badge/TIP-2022-00629B?style=flat-square) — Learning Semantic-Aware Local Features for Long Term Visual Localization [[paper]](https://ieeexplore.ieee.org/document/9829199)
- **[LLF](https://openaccess.thecvf.com/content/WACV2021/html/Suwanwimolkul_Learning_of_Low-Level_Feature_Keypoints_for_Accurate_and_Robust_Detection_WACV_2021_paper.html)** ![WACV 2021](https://img.shields.io/badge/WACV-2021-0F766E?style=flat-square) — Learning of Low-Level Feature Keypoints for Accurate and Robust Detection [[paper]](https://openaccess.thecvf.com/content/WACV2021/html/Suwanwimolkul_Learning_of_Low-Level_Feature_Keypoints_for_Accurate_and_Robust_Detection_WACV_2021_paper.html)
- **[RoRD](https://doi.org/10.1109/IROS51168.2021.9636619)** ![IROS 2021](https://img.shields.io/badge/IROS-2021-0F766E?style=flat-square) — Rotation-Robust Descriptors and Orthographic Views for Local Feature Matching [[paper]](https://doi.org/10.1109/IROS51168.2021.9636619) [[arXiv]](https://arxiv.org/abs/2103.08573) [[code]](https://github.com/UditSinghParihar/RoRD)
- **[ASLFeat](https://openaccess.thecvf.com/content_CVPR_2020/html/Luo_ASLFeat_Learning_Local_Features_of_Accurate_Shape_and_Localization_CVPR_2020_paper.html)** ![CVPR 2020](https://img.shields.io/badge/CVPR-2020-2563EB?style=flat-square) — Learning Local Features of Accurate Shape and Localization [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/html/Luo_ASLFeat_Learning_Local_Features_of_Accurate_Shape_and_Localization_CVPR_2020_paper.html) [[code]](https://github.com/lzx551402/ASLFeat)
- **[MLIFeat](https://openaccess.thecvf.com/content/ACCV2020/html/Zhang_MLIFeat_Multi-level_information_fusion_based_deep_local_features_ACCV_2020_paper.html)** ![ACCV 2020](https://img.shields.io/badge/ACCV-2020-0F766E?style=flat-square) — Multi-level Information Fusion Based Deep Local Features [[paper]](https://openaccess.thecvf.com/content/ACCV2020/html/Zhang_MLIFeat_Multi-level_information_fusion_based_deep_local_features_ACCV_2020_paper.html) [[code]](https://github.com/yyangzh/MLIFeat)
- **[HDD-Net](https://openaccess.thecvf.com/content/ACCV2020/papers/Barroso-Laguna_HDD-Net_Hybrid_Detector_Descriptor_with_Mutual_Interactive_Learning_ACCV_2020_paper.pdf)** ![ACCV 2020](https://img.shields.io/badge/ACCV-2020-0F766E?style=flat-square) — Hybrid Detector Descriptor with Mutual Interactive Learning [[paper]](https://openaccess.thecvf.com/content/ACCV2020/papers/Barroso-Laguna_HDD-Net_Hybrid_Detector_Descriptor_with_Mutual_Interactive_Learning_ACCV_2020_paper.pdf)
- **[Reinforced Feature Points](https://openaccess.thecvf.com/content_CVPR_2020/html/Bhowmik_Reinforced_Feature_Points_Optimizing_Feature_Detection_and_Description_for_a_CVPR_2020_paper.html)** ![CVPR 2020](https://img.shields.io/badge/CVPR-2020-2563EB?style=flat-square) — Optimizing Feature Detection and Description for a High-Level Task [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/html/Bhowmik_Reinforced_Feature_Points_Optimizing_Feature_Detection_and_Description_for_a_CVPR_2020_paper.html)
- **[DISK](https://proceedings.neurips.cc/paper/2020/hash/a42a596fc71e17828440030074d15e74-Abstract.html)** ![NeurIPS 2020](https://img.shields.io/badge/NeurIPS-2020-DC2626?style=flat-square) — Learning Local Features with Policy Gradient [[paper]](https://proceedings.neurips.cc/paper/2020/hash/a42a596fc71e17828440030074d15e74-Abstract.html) [[arXiv]](https://arxiv.org/abs/2006.13566) [[code]](https://github.com/cvlab-epfl/disk)
- **[RF-Net](https://openaccess.thecvf.com/content_CVPR_2019/html/Shen_RF-Net_An_End-To-End_Image_Matching_Network_Based_on_Receptive_Field_CVPR_2019_paper.html)** ![CVPR 2019](https://img.shields.io/badge/CVPR-2019-2563EB?style=flat-square) — An End-to-End Image Matching Network Based on Receptive Field [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/html/Shen_RF-Net_An_End-To-End_Image_Matching_Network_Based_on_Receptive_Field_CVPR_2019_paper.html)
- **[D2-Net](https://openaccess.thecvf.com/content_CVPR_2019/html/Dusmanu_D2-Net_A_Trainable_CNN_for_Joint_Description_and_Detection_of_CVPR_2019_paper.html)** ![CVPR 2019](https://img.shields.io/badge/CVPR-2019-2563EB?style=flat-square) — A Trainable CNN for Joint Description and Detection of Local Features [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/html/Dusmanu_D2-Net_A_Trainable_CNN_for_Joint_Description_and_Detection_of_CVPR_2019_paper.html) [[code]](https://github.com/mihaidusmanu/d2-net)
- **[R2D2](https://proceedings.neurips.cc/paper/2019/hash/3198dfd0aef271d22f7bcddd6f12f5cb-Abstract.html)** ![NeurIPS 2019](https://img.shields.io/badge/NeurIPS-2019-DC2626?style=flat-square) — Reliable and Repeatable Detector and Descriptor [[paper]](https://proceedings.neurips.cc/paper/2019/hash/3198dfd0aef271d22f7bcddd6f12f5cb-Abstract.html) [[arXiv]](https://arxiv.org/abs/1906.06195) [[code]](https://github.com/naver/r2d2)
- **[LF-Net](https://proceedings.neurips.cc/paper/2018/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf)** ![NeurIPS 2018](https://img.shields.io/badge/NeurIPS-2018-DC2626?style=flat-square) — Learning Local Features from Images [[paper]](https://proceedings.neurips.cc/paper/2018/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf) [[code]](https://github.com/vcg-uvic/lf-net-release)
- **[SuperPoint](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w9/html/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.html)** ![CVPRW 2018](https://img.shields.io/badge/CVPRW-2018-60A5FA?style=flat-square) — Self-Supervised Interest Point Detection and Description [[paper]](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w9/html/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.html)

#### Describe-then-Detect

- **[ReDFeat](https://doi.org/10.1109/TIP.2022.3231135)** ![TIP 2023](https://img.shields.io/badge/TIP-2023-00629B?style=flat-square) — Recoupling Detection and Description for Multimodal Feature Learning [[paper]](https://doi.org/10.1109/TIP.2022.3231135) [[arXiv]](https://arxiv.org/abs/2205.07439) [[code]](https://github.com/ACuOoOoO/ReDFeat)
- **[PoSFeat](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Decoupling_Makes_Weakly_Supervised_Local_Feature_Better_CVPR_2022_paper.html)** ![CVPR 2022](https://img.shields.io/badge/CVPR-2022-2563EB?style=flat-square) — Decoupling Makes Weakly Supervised Local Feature Better [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Decoupling_Makes_Weakly_Supervised_Local_Feature_Better_CVPR_2022_paper.html)
- **[SCFeat](https://arxiv.org/abs/2212.07047)** ![arXiv 2022](https://img.shields.io/badge/arXiv-2022-B31B1B?style=flat-square) — Shared Coupling-Bridge for Weakly Supervised Local Feature Learning [[paper]](https://arxiv.org/abs/2212.07047) [[code]](https://github.com/sunjiayuanro/SCFeat)
- **[D2D](https://openaccess.thecvf.com/content/ACCV2020/papers/Tian_D2D_Keypoint_Extraction_with_Describe_to_Detect_Approach_ACCV_2020_paper.pdf)** ![ACCV 2020](https://img.shields.io/badge/ACCV-2020-0F766E?style=flat-square) — Keypoint Extraction with Describe to Detect Approach [[paper]](https://openaccess.thecvf.com/content/ACCV2020/papers/Tian_D2D_Keypoint_Extraction_with_Describe_to_Detect_Approach_ACCV_2020_paper.pdf)

#### Graph Based

<p align="center">
  <img src="figs/GNN.jpg" width="800" alt="General graph-neural-network matching architecture"/>
</p>
<p align="center">
  Fig. 2. A general GNN matching pipeline: encode keypoint geometry and appearance, alternate self- and cross-attention, and estimate a partial assignment.
</p>

- **[MaKeGNN](https://doi.org/10.1109/TIP.2024.3512352)** ![TIP 2025](https://img.shields.io/badge/TIP-2025-00629B?style=flat-square) — Learning Feature Matching via Matchable Keypoint-Assisted Graph Neural Network [[paper]](https://doi.org/10.1109/TIP.2024.3512352) [[arXiv]](https://arxiv.org/abs/2307.01447)
- **[ResMatch](https://ojs.aaai.org/index.php/AAAI/article/view/27915)** ![AAAI 2024](https://img.shields.io/badge/AAAI-2024-0369A1?style=flat-square) — Residual Attention Learning for Feature Matching [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/27915) [[arXiv]](https://arxiv.org/abs/2307.05180) [[code]](https://github.com/ACuOoOoO/ResMatch)
- **[GlueStick](https://openaccess.thecvf.com/content/ICCV2023/html/Pautrat_GlueStick_Robust_Image_Matching_by_Sticking_Points_and_Lines_Together_ICCV_2023_paper.html)** ![ICCV 2023](https://img.shields.io/badge/ICCV-2023-7C3AED?style=flat-square) — Robust Image Matching by Sticking Points and Lines Together [[paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Pautrat_GlueStick_Robust_Image_Matching_by_Sticking_Points_and_Lines_Together_ICCV_2023_paper.html) [[code]](https://github.com/cvg/GlueStick)
- **[LightGlue](https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html)** ![ICCV 2023](https://img.shields.io/badge/ICCV-2023-7C3AED?style=flat-square) — Local Feature Matching at Light Speed [[paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html) [[arXiv]](https://arxiv.org/abs/2306.13643) [[code]](https://github.com/cvg/LightGlue)
- **[ParaFormer](https://ojs.aaai.org/index.php/AAAI/article/view/25275)** ![AAAI 2023](https://img.shields.io/badge/AAAI-2023-0369A1?style=flat-square) — Parallel Attention Transformer for Efficient Feature Matching [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/25275) [[arXiv]](https://arxiv.org/abs/2303.00941)
- **[HTMatch](https://www.sciencedirect.com/science/article/pii/S016516842200398X)** ![Signal Processing 2023](https://img.shields.io/badge/Signal%20Processing-2023-E9711C?style=flat-square) — An Efficient Hybrid Transformer Based Graph Neural Network for Local Feature Matching [[paper]](https://www.sciencedirect.com/science/article/pii/S016516842200398X)
- **[ClusterGNN](https://openaccess.thecvf.com/content/CVPR2022/html/Shi_ClusterGNN_Cluster-Based_Coarse-To-Fine_Graph_Neural_Network_for_Efficient_Feature_Matching_CVPR_2022_paper.html)** ![CVPR 2022](https://img.shields.io/badge/CVPR-2022-2563EB?style=flat-square) — Cluster-Based Coarse-To-Fine Graph Neural Network for Efficient Feature Matching [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Shi_ClusterGNN_Cluster-Based_Coarse-To-Fine_Graph_Neural_Network_for_Efficient_Feature_Matching_CVPR_2022_paper.html)
- **[DenseGAP](https://doi.org/10.1109/ICPR56361.2022.9956472)** ![ICPR 2022](https://img.shields.io/badge/ICPR-2022-0F766E?style=flat-square) — Graph-Structured Dense Correspondence Learning with Anchor Points [[paper]](https://doi.org/10.1109/ICPR56361.2022.9956472) [[arXiv]](https://arxiv.org/abs/2112.06910)
- **[SGMNet](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Learning_To_Match_Features_With_Seeded_Graph_Matching_Network_ICCV_2021_paper.html)** ![ICCV 2021](https://img.shields.io/badge/ICCV-2021-7C3AED?style=flat-square) — Learning to Match Features with Seeded Graph Matching Network [[paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Learning_To_Match_Features_With_Seeded_Graph_Matching_Network_ICCV_2021_paper.html) [[code]](https://github.com/vdvchen/SGMNet)
- **[SuperGlue](https://openaccess.thecvf.com/content_CVPR_2020/html/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.html)** ![CVPR 2020](https://img.shields.io/badge/CVPR-2020-2563EB?style=flat-square) — Learning Feature Matching with Graph Neural Networks [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/html/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.html) [[code]](https://github.com/magicleap/SuperGluePretrainedNetwork)

### Detector-free Models

Detector-free methods establish correspondences directly from dense or regularly sampled image features, rather than relying on an independently selected set of repeatable keypoints.

#### CNN Based

- **[PDC-Net+](https://doi.org/10.1109/TPAMI.2023.3249225)** ![TPAMI 2023](https://img.shields.io/badge/TPAMI-2023-00629B?style=flat-square) — Enhanced Probabilistic Dense Correspondence Network [[paper]](https://doi.org/10.1109/TPAMI.2023.3249225) [[arXiv]](https://arxiv.org/abs/2109.13912) [[code]](https://github.com/PruneTruong/DenseMatching)
- **[PUMP](https://openaccess.thecvf.com/content/CVPR2022/html/Revaud_PUMP_Pyramidal_and_Uniqueness_Matching_Priors_for_Unsupervised_Learning_of_CVPR_2022_paper.html)** ![CVPR 2022](https://img.shields.io/badge/CVPR-2022-2563EB?style=flat-square) — Pyramidal and Uniqueness Matching Priors for Unsupervised Learning of Local Descriptors [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Revaud_PUMP_Pyramidal_and_Uniqueness_Matching_Priors_for_Unsupervised_Learning_of_CVPR_2022_paper.html) [[code]](https://github.com/naver/pump)
- **[PDC-Net](https://openaccess.thecvf.com/content/CVPR2021/html/Truong_Learning_Accurate_Dense_Correspondences_and_When_To_Trust_Them_CVPR_2021_paper.html)** ![CVPR 2021](https://img.shields.io/badge/CVPR-2021-2563EB?style=flat-square) — Learning Accurate Dense Correspondences and When to Trust Them [[paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Truong_Learning_Accurate_Dense_Correspondences_and_When_To_Trust_Them_CVPR_2021_paper.html) [[code]](https://github.com/PruneTruong/DenseMatching)
- **[DFM](https://openaccess.thecvf.com/content/CVPR2021W/IMW/html/Efe_DFM_A_Performance_Baseline_for_Deep_Feature_Matching_CVPRW_2021_paper.html)** ![CVPRW 2021](https://img.shields.io/badge/CVPRW-2021-60A5FA?style=flat-square) — A Performance Baseline for Deep Feature Matching [[paper]](https://openaccess.thecvf.com/content/CVPR2021W/IMW/html/Efe_DFM_A_Performance_Baseline_for_Deep_Feature_Matching_CVPRW_2021_paper.html) [[code]](https://github.com/ufukefe/DFM)
- **[Sparse-NCNet](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540579.pdf)** ![ECCV 2020](https://img.shields.io/badge/ECCV-2020-16A34A?style=flat-square) — Efficient Neighbourhood Consensus Networks via Submanifold Sparse Convolutions [[paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540579.pdf) [[code]](https://github.com/ignacio-rocco/sparse-ncnet)
- **[DualRC-Net](https://proceedings.neurips.cc/paper/2020/hash/c91591a8d461c2869b9f535ded3e213e-Abstract.html)** ![NeurIPS 2020](https://img.shields.io/badge/NeurIPS-2020-DC2626?style=flat-square) — Dual-Resolution Correspondence Networks [[paper]](https://proceedings.neurips.cc/paper/2020/hash/c91591a8d461c2869b9f535ded3e213e-Abstract.html) [[code]](https://github.com/XiSHEN0220/DRC-Net)
- **[GLU-Net](https://openaccess.thecvf.com/content_CVPR_2020/html/Truong_GLU-Net_Global-Local_Universal_Network_for_Dense_Flow_and_Correspondences_CVPR_2020_paper.html)** ![CVPR 2020](https://img.shields.io/badge/CVPR-2020-2563EB?style=flat-square) — Global-Local Universal Network for Dense Flow and Correspondences [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/html/Truong_GLU-Net_Global-Local_Universal_Network_for_Dense_Flow_and_Correspondences_CVPR_2020_paper.html) [[code]](https://github.com/PruneTruong/GLU-Net)
- **[GOCor](https://proceedings.neurips.cc/paper/2020/hash/a4a8a31750a23de2da88ef6a491dfd5c-Abstract.html)** ![NeurIPS 2020](https://img.shields.io/badge/NeurIPS-2020-DC2626?style=flat-square) — Bringing Globally Optimized Correspondence Volumes into Your Neural Network [[paper]](https://proceedings.neurips.cc/paper/2020/hash/a4a8a31750a23de2da88ef6a491dfd5c-Abstract.html) [[code]](https://github.com/PruneTruong/GOCor)
- **[NCNet](https://proceedings.neurips.cc/paper/2018/hash/8f7d807e1f53eff5f9efbe5cb81090fb-Abstract.html)** ![NeurIPS 2018](https://img.shields.io/badge/NeurIPS-2018-DC2626?style=flat-square) — Neighbourhood Consensus Networks [[paper]](https://proceedings.neurips.cc/paper/2018/hash/8f7d807e1f53eff5f9efbe5cb81090fb-Abstract.html) [[code]](https://github.com/ignacio-rocco/ncnet)

#### Transformer Based

- **[RoMa](https://openaccess.thecvf.com/content/CVPR2024/html/Edstedt_RoMa_Robust_Dense_Feature_Matching_CVPR_2024_paper.html)** ![CVPR 2024](https://img.shields.io/badge/CVPR-2024-2563EB?style=flat-square) — Robust Dense Feature Matching [[paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Edstedt_RoMa_Robust_Dense_Feature_Matching_CVPR_2024_paper.html) [[arXiv]](https://arxiv.org/abs/2305.15404) [[code]](https://github.com/Parskatt/RoMa)
- **[DeepMatcher](https://www.sciencedirect.com/science/article/pii/S0957417423018638)** ![Expert Systems with Applications 2024](https://img.shields.io/badge/Expert%20Systems%20with%20Applications-2024-E9711C?style=flat-square) — A Deep Transformer-Based Network for Robust and Accurate Local Feature Matching [[paper]](https://www.sciencedirect.com/science/article/pii/S0957417423018638)
- **[OAMatcher](https://www.sciencedirect.com/science/article/pii/S0031320323007914)** ![Pattern Recognition 2024](https://img.shields.io/badge/Pattern%20Recognition-2024-E9711C?style=flat-square) — An Overlapping Areas-Based Network with Label Credibility for Robust and Accurate Feature Matching [[paper]](https://www.sciencedirect.com/science/article/pii/S0031320323007914) [[code]](https://github.com/DK-HU/OAMatcher)
- **[ASTR](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_Adaptive_Spot-Guided_Transformer_for_Consistent_Local_Feature_Matching_CVPR_2023_paper.html)** ![CVPR 2023](https://img.shields.io/badge/CVPR-2023-2563EB?style=flat-square) — Adaptive Spot-Guided Transformer for Consistent Local Feature Matching [[paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_Adaptive_Spot-Guided_Transformer_for_Consistent_Local_Feature_Matching_CVPR_2023_paper.html)
- **[CasMTR](https://openaccess.thecvf.com/content/ICCV2023/html/Cao_Improving_Transformer-based_Image_Matching_by_Cascaded_Capturing_Spatially_Informative_Keypoints_ICCV_2023_paper.html)** ![ICCV 2023](https://img.shields.io/badge/ICCV-2023-7C3AED?style=flat-square) — Improving Transformer-based Image Matching by Cascaded Capturing Spatially Informative Keypoints [[paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Cao_Improving_Transformer-based_Image_Matching_by_Cascaded_Capturing_Spatially_Informative_Keypoints_ICCV_2023_paper.html) [[code]](https://github.com/ewrfcas/CasMTR)
- **[PMatch](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_PMatch_Paired_Masked_Image_Modeling_for_Dense_Geometric_Matching_CVPR_2023_paper.html)** ![CVPR 2023](https://img.shields.io/badge/CVPR-2023-2563EB?style=flat-square) — Paired Masked Image Modeling for Dense Geometric Matching [[paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_PMatch_Paired_Masked_Image_Modeling_for_Dense_Geometric_Matching_CVPR_2023_paper.html)
- **[SEM](https://openaccess.thecvf.com/content/CVPR2023W/IMW/html/Chang_Structured_Epipolar_Matcher_for_Local_Feature_Matching_CVPRW_2023_paper.html)** ![CVPRW 2023](https://img.shields.io/badge/CVPRW-2023-60A5FA?style=flat-square) — Structured Epipolar Matcher for Local Feature Matching [[paper]](https://openaccess.thecvf.com/content/CVPR2023W/IMW/html/Chang_Structured_Epipolar_Matcher_for_Local_Feature_Matching_CVPRW_2023_paper.html)
- **[DKM](https://openaccess.thecvf.com/content/CVPR2023/html/Edstedt_DKM_Dense_Kernelized_Feature_Matching_for_Geometry_Estimation_CVPR_2023_paper.html)** ![CVPR 2023](https://img.shields.io/badge/CVPR-2023-2563EB?style=flat-square) — Dense Kernelized Feature Matching for Geometry Estimation [[paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Edstedt_DKM_Dense_Kernelized_Feature_Matching_for_Geometry_Estimation_CVPR_2023_paper.html) [[code]](https://github.com/Parskatt/DKM)
- **[TopicFM](https://ojs.aaai.org/index.php/AAAI/article/view/25341)** ![AAAI 2023](https://img.shields.io/badge/AAAI-2023-0369A1?style=flat-square) — Robust and Interpretable Topic-Assisted Feature Matching [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/25341) [[code]](https://github.com/TruongKhang/TopicFM)
- **[ASpanFormer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1701_ECCV_2022_paper.php)** ![ECCV 2022](https://img.shields.io/badge/ECCV-2022-16A34A?style=flat-square) — Detector-Free Image Matching with Adaptive Span Transformer [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1701_ECCV_2022_paper.php) [[code]](https://github.com/apple/ml-aspanformer)
- **[ECO-TR](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6054_ECCV_2022_paper.php)** ![ECCV 2022](https://img.shields.io/badge/ECCV-2022-16A34A?style=flat-square) — Efficient Correspondences Finding Via Coarse-to-Fine Refinement [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6054_ECCV_2022_paper.php) [[code]](https://github.com/dltan7/ECO-TR)
- **[SE2-LoFTR](https://openaccess.thecvf.com/content/CVPR2022W/IMW/html/Bokman_A_Case_for_Using_Rotation_Invariant_Features_in_State_of_CVPRW_2022_paper.html)** ![CVPRW 2022](https://img.shields.io/badge/CVPRW-2022-60A5FA?style=flat-square) — A Case for Using Rotation Invariant Features in State of the Art Feature Matchers [[paper]](https://openaccess.thecvf.com/content/CVPR2022W/IMW/html/Bokman_A_Case_for_Using_Rotation_Invariant_Features_in_State_of_CVPRW_2022_paper.html) [[code]](https://github.com/Parskatt/SE2-LoFTR)
- **[QuadTree](https://openreview.net/forum?id=fR-EnKWL_Zb)** ![ICLR 2022](https://img.shields.io/badge/ICLR-2022-EA580C?style=flat-square) — QuadTree Attention for Vision Transformers [[paper]](https://openreview.net/forum?id=fR-EnKWL_Zb) [[code]](https://github.com/Tangshitao/QuadtreeAttention)
- **[MatchFormer](https://openaccess.thecvf.com/content/ACCV2022/html/Wang_MatchFormer_Interleaving_Attention_in_Transformers_for_Feature_Matching_ACCV_2022_paper.html)** ![ACCV 2022](https://img.shields.io/badge/ACCV-2022-0F766E?style=flat-square) — Interleaving Attention in Transformers for Feature Matching [[paper]](https://openaccess.thecvf.com/content/ACCV2022/html/Wang_MatchFormer_Interleaving_Attention_in_Transformers_for_Feature_Matching_ACCV_2022_paper.html) [[code]](https://github.com/TencentYoutuResearch/ImageMatching-MatchFormer)
- **[COTR](https://openaccess.thecvf.com/content/ICCV2021/html/Jiang_COTR_Correspondence_Transformer_for_Matching_Across_Images_ICCV_2021_paper.html)** ![ICCV 2021](https://img.shields.io/badge/ICCV-2021-7C3AED?style=flat-square) — Correspondence Transformer for Matching Across Images [[paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Jiang_COTR_Correspondence_Transformer_for_Matching_Across_Images_ICCV_2021_paper.html) [[code]](https://github.com/ubc-vision/COTR)
- **[LoFTR](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_LoFTR_Detector-Free_Local_Feature_Matching_With_Transformers_CVPR_2021_paper.html)** ![CVPR 2021](https://img.shields.io/badge/CVPR-2021-2563EB?style=flat-square) — Detector-Free Local Feature Matching with Transformers [[paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_LoFTR_Detector-Free_Local_Feature_Matching_With_Transformers_CVPR_2021_paper.html) [[code]](https://github.com/zju3dv/LoFTR)

#### Patch Based

- **[SGAM](https://www.sciencedirect.com/science/article/pii/S003132032600885X)** ![Pattern Recognition 2026](https://img.shields.io/badge/Pattern%20Recognition-2026-E9711C?style=flat-square) — Searching from Area to Point: A Semantic Guided Framework with Geometric Consistency for Accurate Feature Matching [[paper]](https://www.sciencedirect.com/science/article/pii/S003132032600885X)
- **[AdaMatcher](https://openaccess.thecvf.com/content/CVPR2023/html/Huang_Adaptive_Assignment_for_Geometry_Aware_Local_Feature_Matching_CVPR_2023_paper.html)** ![CVPR 2023](https://img.shields.io/badge/CVPR-2023-2563EB?style=flat-square) — Adaptive Assignment for Geometry Aware Local Feature Matching [[paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Huang_Adaptive_Assignment_for_Geometry_Aware_Local_Feature_Matching_CVPR_2023_paper.html) [[code]](https://github.com/TencentYoutuResearch/AdaMatcher)
- **[PATS](https://openaccess.thecvf.com/content/CVPR2023/html/Ni_PATS_Patch_Area_Transportation_With_Subdivision_for_Local_Feature_Matching_CVPR_2023_paper.html)** ![CVPR 2023](https://img.shields.io/badge/CVPR-2023-2563EB?style=flat-square) — Patch Area Transportation with Subdivision for Local Feature Matching [[paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Ni_PATS_Patch_Area_Transportation_With_Subdivision_for_Local_Feature_Matching_CVPR_2023_paper.html) [[code]](https://github.com/zju3dv/PATS)
- **[Patch2Pix](https://openaccess.thecvf.com/content/CVPR2021/html/Zhou_Patch2Pix_Epipolar-Guided_Pixel-Level_Correspondences_CVPR_2021_paper.html)** ![CVPR 2021](https://img.shields.io/badge/CVPR-2021-2563EB?style=flat-square) — Epipolar-Guided Pixel-Level Correspondences [[paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Zhou_Patch2Pix_Epipolar-Guided_Pixel-Level_Correspondences_CVPR_2021_paper.html) [[code]](https://github.com/GrumpyZhou/patch2pix)

<a id="benchmarks-and-evaluation"></a>
## 📊 Benchmarks and Evaluation

The survey compares methods across patch matching, two-view geometry, visual localization, and reconstruction. Common resources and metrics include:

| Evaluation target | Common datasets | Typical metrics |
| --- | --- | --- |
| Patch matching and homography | [HPatches](https://github.com/hpatches/hpatches-dataset) | Mean Matching Accuracy (MMA), matching precision/recall |
| Outdoor and indoor relative pose | [MegaDepth](https://www.cs.cornell.edu/projects/megadepth/), [ScanNet](http://www.scan-net.org/), [YFCC100M](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/) | Pose-error AUC at fixed angular thresholds; inlier precision and match count |
| Long-term visual localization | [Aachen Day-Night](https://www.visuallocalization.net/datasets/) | Percentage localized within position/orientation thresholds |
| Structure from Motion | [ETH3D](https://www.eth3d.net/), Image Matching Challenge datasets | Camera-pose accuracy, reconstruction accuracy, and completeness |

Evaluation protocols are not interchangeable: image resizing, keypoint budgets, confidence thresholds, geometric verification, and pose solvers should be reported when comparing methods.

<a id="applications"></a>
## 🧩 Applications

### Structure from Motion

Local feature matching supplies pairwise correspondences and multi-view tracks for camera-pose estimation and 3D reconstruction. The survey contrasts conventional detector-based pipelines with featuremetric refinement and emerging detector-free SfM, where multi-view consistency must be recovered from dense pairwise matches.

- **[Detector-Free SfM](https://openaccess.thecvf.com/content/CVPR2024/html/He_Detector-Free_Structure_from_Motion_CVPR_2024_paper.html)** ![CVPR 2024](https://img.shields.io/badge/CVPR-2024-2563EB?style=flat-square) — Detector-Free Structure from Motion [[paper]](https://openaccess.thecvf.com/content/CVPR2024/html/He_Detector-Free_Structure_from_Motion_CVPR_2024_paper.html) [[code]](https://github.com/zju3dv/DetectorFreeSfM)
- **[PixSfM](https://openaccess.thecvf.com/content/ICCV2021/html/Lindenberger_Pixel-Perfect_Structure-From-Motion_With_Featuremetric_Refinement_ICCV_2021_paper.html)** ![ICCV 2021](https://img.shields.io/badge/ICCV-2021-7C3AED?style=flat-square) — Pixel-Perfect Structure-from-Motion with Featuremetric Refinement [[paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Lindenberger_Pixel-Perfect_Structure-From-Motion_With_Featuremetric_Refinement_ICCV_2021_paper.html) [[code]](https://github.com/cvg/pixel-perfect-sfm)
- **[COLMAP](https://openaccess.thecvf.com/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html)** ![CVPR 2016](https://img.shields.io/badge/CVPR-2016-2563EB?style=flat-square) — Structure-from-Motion Revisited [[paper]](https://openaccess.thecvf.com/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html) [[code]](https://github.com/colmap/colmap)

### Remote Sensing Image Registration

Multimodal remote-sensing registration must handle geometric distortion together with nonlinear radiometric differences between sensors. The survey highlights multiscale unsupervised registration, SAR–optical structural features, image-based matching, and learned correspondence filtering.

- **[MUNet](https://doi.org/10.1109/TGRS.2022.3167644)** ![TGRS 2022](https://img.shields.io/badge/TGRS-2022-00629B?style=flat-square) — A Multiscale Framework with Unsupervised Learning for Remote Sensing Image Registration [[paper]](https://doi.org/10.1109/TGRS.2022.3167644)
- **[HCA-Net](https://doi.org/10.1109/TGRS.2022.3165222)** ![TGRS 2022](https://img.shields.io/badge/TGRS-2022-00629B?style=flat-square) — A Hierarchical Consensus Attention Network for Feature Matching of Remote Sensing Images [[paper]](https://doi.org/10.1109/TGRS.2022.3165222)
- **[MAP-Net](https://doi.org/10.1109/TGRS.2021.3066432)** ![TGRS 2022](https://img.shields.io/badge/TGRS-2022-00629B?style=flat-square) — SAR and Optical Image Matching via Image-Based Convolutional Network with Attention Mechanism and Spatial Pyramid Aggregated Pooling [[paper]](https://doi.org/10.1109/TGRS.2021.3066432)
- **[MCGF](https://doi.org/10.1109/LGRS.2021.3105567)** ![GRSL 2022](https://img.shields.io/badge/GRSL-2022-00629B?style=flat-square) — Robust Matching for SAR and Optical Images Using Multiscale Convolutional Gradient Features [[paper]](https://doi.org/10.1109/LGRS.2021.3105567)

### Medical Image Registration

The survey focuses on motion estimation and 2D–3D registration. Representative systems estimate dense cardiac or tongue motion, learn deformable 4D-CT alignment, or bridge simulated and real X-ray/CT data without requiring paired annotations.

- **[HPRN](https://www.sciencedirect.com/science/article/pii/S1566253523003779)** ![Information Fusion 2024](https://img.shields.io/badge/Information%20Fusion-2024-E9711C?style=flat-square) — Hybrid Unsupervised Paradigm Based Deformable Image Fusion for 4D CT Lung Image Modality [[paper]](https://www.sciencedirect.com/science/article/pii/S1566253523003779)
- **[DRIMET](https://openreview.net/forum?id=jkSC4UHHVzy)** ![MIDL 2023, Oral](https://img.shields.io/badge/MIDL-2023%20Oral-0F766E?style=flat-square) — Deep Registration-Based 3D Incompressible Motion Estimation in Tagged-MRI with Application to the Tongue [[paper]](https://openreview.net/forum?id=jkSC4UHHVzy)
- **[Self-Supervised 2D/3D Registration](https://openaccess.thecvf.com/content/WACV2023/html/Jaganathan_Self-Supervised_2D3D_Registration_for_X-Ray_to_CT_Image_Fusion_WACV_2023_paper.html)** ![WACV 2023](https://img.shields.io/badge/WACV-2023-0F766E?style=flat-square) — X-Ray to CT Image Fusion [[paper]](https://openaccess.thecvf.com/content/WACV2023/html/Jaganathan_Self-Supervised_2D3D_Registration_for_X-Ray_to_CT_Image_Fusion_WACV_2023_paper.html)
- **[DeepTag](https://openaccess.thecvf.com/content/CVPR2021/html/Ye_DeepTag_An_Unsupervised_Deep_Learning_Method_for_Motion_Tracking_on_CVPR_2021_paper.html)** ![CVPR 2021](https://img.shields.io/badge/CVPR-2021-2563EB?style=flat-square) — An Unsupervised Deep Learning Method for Motion Tracking on Cardiac Tagging Magnetic Resonance Images [[paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Ye_DeepTag_An_Unsupervised_Deep_Learning_Method_for_Motion_Tracking_on_CVPR_2021_paper.html)

<a id="related-surveys"></a>
## Related Surveys

- [A Survey on Deep Learning in Medical Image Registration: New Technologies, Uncertainty, Evaluation Metrics, and Beyond](https://doi.org/10.1016/j.media.2024.103385) ![Medical Image Analysis 2024](https://img.shields.io/badge/Medical%20Image%20Analysis-2024-E9711C?style=flat-square) [[paper]](https://doi.org/10.1016/j.media.2024.103385)
- [Advances and Challenges in Multimodal Remote Sensing Image Registration](https://doi.org/10.1109/JMASS.2023.3244848) ![IEEE Journal on Miniaturization for Air and Space Systems 2023](https://img.shields.io/badge/IEEE%20Journal%20on%20Miniaturization%20for%20Air%20and%20Space%20Systems-2023-00629B?style=flat-square) [[paper]](https://doi.org/10.1109/JMASS.2023.3244848)
- [Image Feature Information Extraction for Interest Point Detection: A Comprehensive Review](https://doi.org/10.1109/TPAMI.2022.3201185) ![TPAMI 2023](https://img.shields.io/badge/TPAMI-2023-00629B?style=flat-square) [[paper]](https://doi.org/10.1109/TPAMI.2022.3201185)
- [Challenges in Image Matching for Cultural Heritage: An Overview and Perspective](https://doi.org/10.1007/978-3-031-13321-3_19) ![ICIAP 2022](https://img.shields.io/badge/ICIAP-2022-0F766E?style=flat-square) [[paper]](https://doi.org/10.1007/978-3-031-13321-3_19)
- [A Review of Multimodal Image Matching: Methods and Applications](https://www.sciencedirect.com/science/article/pii/S156625352100035X) ![Information Fusion 2021](https://img.shields.io/badge/Information%20Fusion-2021-E9711C?style=flat-square) [[paper]](https://www.sciencedirect.com/science/article/pii/S156625352100035X)
- [Image Matching from Handcrafted to Deep Features: A Survey](https://doi.org/10.1007/s11263-020-01359-2) ![IJCV 2021](https://img.shields.io/badge/IJCV-2021-16A34A?style=flat-square) [[paper]](https://doi.org/10.1007/s11263-020-01359-2)
- [Recent Advances in Local Feature Detector and Descriptor: A Literature Survey](https://doi.org/10.1007/s13735-020-00200-3) ![IJMIR 2020](https://img.shields.io/badge/IJMIR-2020-16A34A?style=flat-square) [[paper]](https://doi.org/10.1007/s13735-020-00200-3)
- [Local Feature Descriptor for Image Matching: A Survey](https://doi.org/10.1109/ACCESS.2018.2888856) ![IEEE Access 2019](https://img.shields.io/badge/IEEE%20Access-2019-00629B?style=flat-square) [[paper]](https://doi.org/10.1109/ACCESS.2018.2888856)

<a id="citation"></a>
## 📝 Citation

If this survey or list is useful in your research, please cite:

```bibtex
@article{xu2024_LFM_survey,
  title={Local feature matching using deep learning: A survey},
  author={Xu, Shibiao and Chen, Shunpeng and Xu, Rongtao and Wang, Changwei and Lu, Peng and Guo, Li},
  journal={Information Fusion},
  volume={107},
  pages={102344},
  year={2024},
  publisher={Elsevier}
}
```

<a id="acknowledgement"></a>
## Acknowledgement

This work is supported by Beijing Natural Science Foundation No. JQ23014, in part by the National Natural Science Foundation of China (No. 62271074).

<a id="license"></a>
## 📄 License

This project is licensed under the [MIT License](https://opensource.org/license/mit).

