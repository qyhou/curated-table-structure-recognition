# curated-table-structure-recognition
This repository provides a curated list of resources in the research domain of Table Structure Recognition (TSR), updated in my free time.

Methods based on heuristic rules are not included.

## Keywords
"Table Structure Recognition" OR "Table Recognition"

## Table of Contents
- [Methodology](methodology.md)
  - Review
  - Top-Down
  - Bottom-Up
    - Cell based / Grid based
    - Word based / Text-Line based
  - Image-to-Sequence / End-to-End
- [Data](data.md)
  - Metric
  - Dataset / Benchmark
  - Data Representation
  - Data Synthesis / Data Generation / Data Augmentation

## Popular Benchmarks

### FinTabNet Version 1.0.0 (10635(~~10656~~) samples)
- Note: The actual number of downloaded val and test samples is contrary to the split described in the paper and metadata.
- Links:
  - [Global Table Extractor (GTE): A Framework for Joint Table Identification and Cell Structure Recognition Using Visual Context](https://arxiv.org/abs/2005.00589)
  - [https://developer.ibm.com/exchanges/data/all/fintabnet/](https://developer.ibm.com/exchanges/data/all/fintabnet/)
- Supervised training only using the training set of FinTabNet Version 1.0.0 (91596 samples)
- Evaluation for FinTabNet:
  - [x] High TEDS
  - [ ] High TEDS-S only: insufficient

| Paper                                                                                                                                                                                  | Group                          | Date    | Publication              | TEDS                               | TEDS-S                             | AP-50 |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------|:--------|:-------------------------|:-----------------------------------|:-----------------------------------|:------|
| `OmniParser V2` [OmniParser V2: Structured-Points-of-Thought for Unified Visual Text Parsing and Its Generality to Multimodal Large Language Models](https://arxiv.org/abs/2502.16161) | HUST, Alibaba                  | 2025-02 | arXiv                    | 90.50                              | 93.20                              | -     |
| `BGTR` [Enhancing Table Structure Recognition via Bounding Box Guidance](https://link.springer.com/chapter/10.1007/978-3-031-78498-9_15)                                               | SCUT                           | 2024-12 | ICPR-2024                | -                                  | 98.89                              | -     |
| `TableAttention` [Multi-Modal Attention Based on 2D Structured Sequence for Table Recognition](https://link.springer.com/chapter/10.1007/978-981-97-8511-7_27)                         | CAS, UCAS, Zhongke Fanyu       | 2024-11 | PRCV-2024                | 97.20, Simple-97.50, Complex-97.10 | 98.54, Simple-99.14, Complex-98.23 | 97.90 |
| [Enhancing Transformer-Based Table Structure Recognition for Long Tables](https://link.springer.com/chapter/10.1007/978-981-97-8511-7_16)                                              | PKU                            | 2024-11 | PRCV-2024                | 96.82                              | 99.04                              | -     |
| `SPRINT` [SPRINT: Script-agnostic Structure Recognition in Tables](https://link.springer.com/chapter/10.1007/978-3-031-70549-6_21)                                                     | IIT Bombay                     | 2024-09 | ICDAR-2024               | -                                  | 98.03, Simple-98.35, Complex-97.74 | -     |
| `TFLOP` [TFLOP: Table Structure Recognition Framework with Layout Pointer Mechanism](https://www.ijcai.org/proceedings/2024/105)                                                       | Upstage AI                     | 2024-08 | IJCAI-2024               | 99.45                              | 99.56                              | -     |
| `MuTabNet` [Multi-Cell Decoder and Mutual Learning for Table Structure and Character Recognition](https://arxiv.org/abs/2404.13268)                                                    | PFN                            | 2024-04 | ICDAR-2024               | 97.69                              | 98.87                              | -     |
| `OmniParser` [OmniParser: A Unified Framework for Text Spotting, Key Information Extraction and Table Recognition](https://arxiv.org/abs/2403.19128)                                   | Alibaba, HUST                  | 2024-03 | CVPR-2024                | 89.75                              | 91.55                              | -     |
| `UniTable` [UniTable: Towards a Unified Framework for Table Structure Recognition via Self-Supervised Pretraining](https://arxiv.org/abs/2403.04822)                                   | Georgia Tech, ADP              | 2024-03 | NeurIPS-Workshop-2024    | -                                  | 98.89                              | -     |
| `GridFormer` [GridFormer: Towards Accurate Table Structure Recognition via Grid Prediction](https://arxiv.org/abs/2309.14962)                                                          | Baidu, SCUT                    | 2023-09 | ACM-MM-2023              | -                                  | 98.63                              | -     |
| [An End-to-End Local Attention Based Model for Table Recognition](https://link.springer.com/chapter/10.1007/978-3-031-41679-8_2)                                                       | NII                            | 2023-08 | ICDAR-2023               | 95.74                              | 98.85                              | -     |
| `TSRFormer(DQ-DETR)` [Robust Table Structure Recognition with Dynamic Queries Enhanced Detection Transformer](https://arxiv.org/abs/2303.11615)                                        | USTC, Microsoft, SJTU, Alibaba | 2023-03 | Pattern-Recognition.2023 | -                                  | 98.40                              | -     |
| `MTL-TabNet` [An End-to-End Multi-Task Learning Model for Image-based Table Recognition](https://arxiv.org/abs/2303.08648)                                                             | NII                            | 2023-03 | VISIGRAPP-2023           | -                                  | 98.79, Simple-99.07, Complex-98.46 | -     |
| `VAST` [Improving Table Structure Recognition with Visual-Alignment Sequential Coordinate Modeling](https://arxiv.org/abs/2303.06949)                                                  | Huawei, PKU                    | 2023-03 | CVPR-2023                | 98.21                              | 98.63                              | 96.20 |
| `WSTabNet` [Rethinking Image-based Table Recognition Using Weakly Supervised Methods](https://arxiv.org/abs/2303.07641)                                                                | NII                            | 2023-02 | ICPRAM-2023              | 95.32, Simple-95.24, Complex-95.41 | 98.72, Simple-99.06, Complex-98.33 | -     |
| `TableFormer` [TableFormer: Table Structure Understanding with Transformers](https://arxiv.org/abs/2203.01017)                                                                         | IBM                            | 2022-03 | CVPR-2022                | -                                  | 96.80, Simple-97.50, Complex-96.00 | -     |

### PubTabNet Version 2: val / development (9115 samples)
- Links:
  - [Image-based table recognition: data, model, and evaluation](https://arxiv.org/abs/1911.10683)
  - [https://github.com/ibm-aur-nlp/PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet)
  - [https://developer.ibm.com/exchanges/data/all/pubtabnet/](https://developer.ibm.com/exchanges/data/all/pubtabnet/)
- Supervised training only using the training set of PubTabNet Version 2 (500777 samples)
- Evaluation for PubTabNet:
  - [x] High TEDS
  - [x] High TEDS-S + High AP: to focus on the structure
  - [ ] High TEDS-S only: insufficient

| Paper                                                                                                                                                                                  | Group                          | Date    | Publication              | TEDS                               | TEDS-S                             | AP-50 |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------|:--------|:-------------------------|:-----------------------------------|:-----------------------------------|:------|
| `OmniParser V2` [OmniParser V2: Structured-Points-of-Thought for Unified Visual Text Parsing and Its Generality to Multimodal Large Language Models](https://arxiv.org/abs/2502.16161) | HUST, Alibaba                  | 2025-02 | arXiv                    | 88.90                              | 90.50                              | -     |
| `BGTR` [Enhancing Table Structure Recognition via Bounding Box Guidance](https://link.springer.com/chapter/10.1007/978-3-031-78498-9_15)                                               | SCUT                           | 2024-12 | ICPR-2024                | 96.57                              | 97.63                              | -     |
| `TableAttention` [Multi-Modal Attention Based on 2D Structured Sequence for Table Recognition](https://link.springer.com/chapter/10.1007/978-981-97-8511-7_27)                         | CAS, UCAS, Zhongke Fanyu       | 2024-11 | PRCV-2024                | 95.38, Simple-96.50, Complex-94.20 | 96.96, Simple-98.63, Complex-95.14 | 96.80 |
| [Enhancing Transformer-Based Table Structure Recognition for Long Tables](https://link.springer.com/chapter/10.1007/978-981-97-8511-7_16)                                              | PKU                            | 2024-11 | PRCV-2024                | 96.77                              | 98.82                              | -     |
| `TFLOP` [TFLOP: Table Structure Recognition Framework with Layout Pointer Mechanism](https://www.ijcai.org/proceedings/2024/105)                                                       | Upstage AI                     | 2024-08 | IJCAI-2024               | 98.00                              | 98.30                              | -     |
| `SEMv3` [SEMv3: A Fast and Robust Approach to Table Separation Line Detection](https://arxiv.org/abs/2405.11862)                                                                       | USTC, iFLYTEK                  | 2024-05 | IJCAI-2024               | 97.30                              | 97.50                              | -     |
| `MuTabNet` [Multi-Cell Decoder and Mutual Learning for Table Structure and Character Recognition](https://arxiv.org/abs/2404.13268)                                                    | PFN                            | 2024-04 | ICDAR-2024               | 96.87, Simple-98.16, Complex-95.53 | -                                  | -     |
| `OmniParser` [OmniParser: A Unified Framework for Text Spotting, Key Information Extraction and Table Recognition](https://arxiv.org/abs/2403.19128)                                   | Alibaba, HUST                  | 2024-03 | CVPR-2024                | 88.83                              | 90.45                              | -     |
| `UniTable` [UniTable: Towards a Unified Framework for Table Structure Recognition via Self-Supervised Pretraining](https://arxiv.org/abs/2403.04822)                                   | Georgia Tech, ADP              | 2024-03 | NeurIPS-Workshop-2024    | 96.50                              | 97.89                              | 98.43 |
| `LinearProj` [Self-Supervised Pre-Training for Table Structure Recognition Transformer](https://arxiv.org/abs/2402.15578)                                                              | Georgia Tech, ADP              | 2024-02 | AAAI-Workshop-2024       | -                                  | 96.83, Simple-98.48, Complex-95.11 | -     |
| `CNN-transformer` [High-Performance Transformers for Table Structure Recognition Need Early Convolutions](https://arxiv.org/abs/2311.05565)                                            | Georgia Tech, ADP              | 2023-11 | NeurIPS-Workshop-2023    | -                                  | 96.53, Simple-98.33, Complex-94.66 | -     |
| `GridFormer` [GridFormer: Towards Accurate Table Structure Recognition via Grid Prediction](https://arxiv.org/abs/2309.14962)                                                          | Baidu, SCUT                    | 2023-09 | ACM-MM-2023              | 95.84                              | 97.00                              | -     |
| `DRCC` [Divide Rows and Conquer Cells: Towards Structure Recognition for Large Tables](https://www.ijcai.org/proceedings/2023/152)                                                     | CAS, UCAS, Hikvision, CUC      | 2023-08 | IJCAI-2023               | 97.80                              | 98.90                              | -     |
| [An End-to-End Local Attention Based Model for Table Recognition](https://link.springer.com/chapter/10.1007/978-3-031-41679-8_2)                                                       | NII                            | 2023-08 | ICDAR-2023               | 96.77, Simple-98.07, Complex-95.42 | -                                  | -     |
| `TSRFormer(DQ-DETR)` [Robust Table Structure Recognition with Dynamic Queries Enhanced Detection Transformer](https://arxiv.org/abs/2303.11615)                                        | USTC, Microsoft, SJTU, Alibaba | 2023-03 | Pattern-Recognition.2023 | -                                  | 97.50                              | -     |
| `MTL-TabNet` [An End-to-End Multi-Task Learning Model for Image-based Table Recognition](https://arxiv.org/abs/2303.08648)                                                             | NII                            | 2023-03 | VISIGRAPP-2023           | 96.67, Simple-97.92, Complex-95.36 | 97.88, Simple-99.05, Complex-96.66 | -     |
| `VAST` [Improving Table Structure Recognition with Visual-Alignment Sequential Coordinate Modeling](https://arxiv.org/abs/2303.06949)                                                  | Huawei, PKU                    | 2023-03 | CVPR-2023                | 96.31                              | 97.23                              | 94.80 |
| `SEMv2` [SEMv2: Table Separation Line Detection Based on Conditional Convolution](https://arxiv.org/abs/2303.04384)                                                                    | USTC, iFLYTEK                  | 2023-03 | Pattern-Recognition.2024 | -                                  | 97.50                              | -     |
| `WSTabNet` [Rethinking Image-based Table Recognition Using Weakly Supervised Methods](https://arxiv.org/abs/2303.07641)                                                                | NII                            | 2023-02 | ICPRAM-2023              | 96.48, Simple-97.89, Complex-95.02 | 97.74, Simple-99.06, Complex-96.37 | -     |
| `TSRNet` [Table Structure Recognition and Form Parsing by End-to-End Object Detection and Relation Parsing](https://www.sciencedirect.com/science/article/abs/pii/S0031320322004265)   | CAS, UCAS                      | 2022-12 | Pattern-Recognition.2022 | -                                  | 95.64                              | -     |
| `SLANet` [PP-StructureV2: A Stronger Document Analysis System](https://arxiv.org/abs/2210.05391)                                                                                       | Baidu                          | 2022-10 | arXiv                    | 95.89                              | 97.01                              | -     |
| `TRUST` [TRUST: An Accurate and End-to-End Table structure Recognizer Using Splitting-based Transformers](https://arxiv.org/abs/2208.14687)                                            | Baidu, DUT                     | 2022-08 | arXiv                    | 96.20                              | 97.10                              | -     |
| `TSRFormer` [TSRFormer: Table Structure Recognition with Transformers](https://arxiv.org/abs/2208.04921)                                                                               | Microsoft, UCAS, USTC, SJTU    | 2022-08 | ACM-MM-2022              | -                                  | 97.50                              | -     |
| `RobusTabNet` [Robust Table Detection and Structure Recognition from Heterogeneous Document Images](https://arxiv.org/abs/2203.09056)                                                  | USTC, Microsoft                | 2022-03 | Pattern-Recognition.2023 | -                                  | 97.00                              | -     |
| `TableFormer` [TableFormer: Table Structure Understanding with Transformers](https://arxiv.org/abs/2203.01017)                                                                         | IBM                            | 2022-03 | CVPR-2022                | 93.60, Simple-95.40, Complex-90.10 | 96.75, Simple-98.50, Complex-95.00 | 82.10 |
| `SEM` [Split, Embed and Merge: An accurate table structure recognizer](https://arxiv.org/abs/2107.05214)                                                                               | USTC, iFLYTEK                  | 2021-07 | Pattern-Recognition.2022 | 93.70, Simple-94.80, Complex-92.50 | -                                  | -     |
| `LGPMA` [LGPMA: Complicated Table Structure Recognition with Local and Global Pyramid Mask Alignment](https://arxiv.org/abs/2105.06224)                                                | Hikvision, ZJU                 | 2021-05 | ICDAR-2021               | 94.60                              | 96.70                              | -     |
| `TableMASTER` [PingAn-VCGroup's Solution for ICDAR 2021 Competition on Scientific Literature Parsing Task B: Table Recognition to HTML](https://arxiv.org/abs/2105.01848)              | Ping An                        | 2021-05 | arXiv                    | 96.18(~~96.84~~)                   | -                                  | -     |

### PubTabNet Version 2: ICDAR-2021-SLP final evaluation (9064 samples)
- Links:
  - [ICDAR 2021 Competition on Scientific Literature Parsing](https://arxiv.org/abs/2106.14616)
  - [https://github.com/IBM/ICDAR2021-SLP](https://github.com/IBM/ICDAR2021-SLP)
  - [https://github.com/ajjimeno/icdar-task-b](https://github.com/ajjimeno/icdar-task-b)
- It is allowed to use additional third-party data or pre-trained models for performance improvement.
- HTML tags that define the text style including bold, italic, strike through, superscript, and subscript should be included in the cell content.
- Due to a problem with the final evaluation data set, bold tags `<b>` where not considered in the evaluation.

| Paper                                                                                                                                                               | Group      | Date    | Publication               | TEDS                               | TEDS-S | AP-50 |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------|:--------|:--------------------------|:-----------------------------------|:-------|:------|
| `TFLOP` [TFLOP: Table Structure Recognition Framework with Layout Pointer Mechanism](https://www.ijcai.org/proceedings/2024/105)                                    | Upstage AI | 2024-08 | IJCAI-2024                | 96.66                              | 98.38  | -     |
| `MuTabNet` [Multi-Cell Decoder and Mutual Learning for Table Structure and Character Recognition](https://arxiv.org/abs/2404.13268)                                 | PFN        | 2024-04 | ICDAR-2024                | 96.53, Simple-98.01, Complex-94.98 | -      | -     |
| [An End-to-End Local Attention Based Model for Table Recognition](https://link.springer.com/chapter/10.1007/978-3-031-41679-8_2)                                    | NII        | 2023-07 | ICDAR-2023                | 96.21, Simple-97.77, Complex-94.58 | -      | -     |
| `MTL-TabNet` [An End-to-End Multi-Task Learning Model for Image-based Table Recognition](https://arxiv.org/abs/2303.08648)                                          | NII        | 2023-03 | VISIGRAPP-2023            | 96.17, Simple-97.60, Complex-94.68 | -      | -     |
| `WSTabNet` [Rethinking Image-based Table Recognition Using Weakly Supervised Methods](https://arxiv.org/abs/2303.07641)                                             | NII        | 2023-02 | ICPRAM-2023               | 95.97, Simple-97.51, Complex-94.37 | -      | -     |
| `CoT_SRN` [Contextual transformer sequence-based recognition network for medical examination reports](https://link.springer.com/article/10.1007/s10489-022-04420-4) | SDNU       | 2022-12 | Applied-Intelligence.2023 | 92.34                              | 95.71  | -     |

| Team          | Group        | TEDS                               |
|:--------------|:-------------|:-----------------------------------|
| Davar-Lab-OCR | Hikvision    | 96.36, Simple-97.88, Complex-94.78 |
| VCGroup       | Ping An      | 96.32, Simple-97.90, Complex-94.68 |
| XM            | USTC-NELSLIP | 96.27, Simple-97.60, Complex-94.89 |
| YG            |              | 96.11, Simple-97.38, Complex-94.79 |
| DBJ           |              | 95.66, Simple-97.39, Complex-93.87 |
| TAL           | TAL          | 95.65, Simple-97.30, Complex-93.93 |
| PaodingAI     | Paoding      | 95.61, Simple-97.35, Complex-93.79 |
| anyone        |              | 95.23, Simple-96.95, Complex-93.43 |
| LTIAYN        |              | 94.84, Simple-97.18, Complex-92.40 |

## FPS
| Paper                                                                                                                                                                                  | Group                            | Date    | Publication              | Image Size | GPU                    |   FPS |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------|:--------|:-------------------------|-----------:|:-----------------------|------:|
| `OmniParser V2` [OmniParser V2: Structured-Points-of-Thought for Unified Visual Text Parsing and Its Generality to Multimodal Large Language Models](https://arxiv.org/abs/2502.16161) | (HUST, Alibaba)                  | 2025-02 | arXiv                    |       1024 | -                      |  1.70 |
| `DTSM` [DTSM: Toward Dense Table Structure Recognition with Text Query Encoder and Adjacent Feature Aggregator](https://link.springer.com/chapter/10.1007/978-3-031-70533-5_25)        | (SCUT)                           | 2024-09 | ICDAR-2024               |        500 | NVIDIA Titan Xp        |  1.12 |
| `OmniParser` [OmniParser: A Unified Framework for Text Spotting, Key Information Extraction and Table Recognition](https://arxiv.org/abs/2403.19128)                                   | (Alibaba, HUST)                  | 2024-03 | CVPR-2024                |       1024 | -                      |  1.30 |
| `TSRFormer(DQ-DETR)` [Robust Table Structure Recognition with Dynamic Queries Enhanced Detection Transformer](https://arxiv.org/abs/2303.11615)                                        | (USTC, Microsoft, SJTU, Alibaba) | 2023-03 | Pattern-Recognition.2023 |       1024 | NVIDIA Tesla V100      |  4.17 |
| `VAST` [Improving Table Structure Recognition with Visual-Alignment Sequential Coordinate Modeling](https://arxiv.org/abs/2303.06949)                                                  | (Huawei, PKU)                    | 2023-03 | CVPR-2023                |        608 | NVIDIA Tesla V100      |  1.38 |
| `SEMv2` [SEMv2: Table Separation Line Detection Based on Conditional Convolution](https://arxiv.org/abs/2303.04384)                                                                    | (USTC, iFLYTEK)                  | 2023-03 | Pattern-Recognition.2024 |          - | NVIDIA Tesla V100 32GB |  7.30 |
| `TRUST` [TRUST: An Accurate and End-to-End Table structure Recognizer Using Splitting-based Transformers](https://arxiv.org/abs/2208.14687)                                            | (Baidu, DUT)                     | 2022-08 | arXiv                    |        640 | NVIDIA Tesla A100 64GB | 10.00 |
| `TSRFormer` [TSRFormer: Table Structure Recognition with Transformers](https://arxiv.org/abs/2208.04921)                                                                               | (Microsoft, UCAS, USTC, SJTU)    | 2022-08 | ACM-MM-2022              |       1024 | NVIDIA Tesla V100      |  5.17 |
| `RobusTabNet` [Robust Table Detection and Structure Recognition from Heterogeneous Document Images](https://arxiv.org/abs/2203.09056)                                                  | (USTC, Microsoft)                | 2022-03 | Pattern-Recognition.2023 |       1024 | NVIDIA Tesla V100      |  5.19 |
| `SEM` [Split, Embed and Merge: An accurate table structure recognizer](https://arxiv.org/abs/2107.05214)                                                                               | (USTC, iFLYTEK)                  | 2021-07 | Pattern-Recognition.2022 |          - | NVIDIA Tesla V100 32GB |  1.94 |
