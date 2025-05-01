# Fashion Recommender System

A fashion recommendation system that leverages deep learning and data mining techniques to provide personalized outfit recommendations based on item attributes
and visual similarity

## Team Members
- Danishbir Singh Bhatti
- Jessica Lieu
- Sergio Talavera

## Project Overview

This project aims to develop an advanced fashion recommender system that goes beyond basic rating predictions by incorporating visual elements and style attributes. Using the DeepFashion Database, we implement multiple recommendation algorithms to suggest fashion items and with similarities in attribute, category, and visual features.


### Key Features
- Outfit recommendations based on attributes and categorical similarities
- Visual similarity analysis between fashion items

## Dataset

We use the Category and Attribute Prediction Benchmark from the Large-scale Fashion (DeepFashion) Database. The dataset contains:
- 289,222 fashion images
- 50 clothing categories (jacket, shirt, pants, etc.)
- 1,000 clothing attributes (floral, long sleeves, denim, etc.)
- Annotations with bounding boxes and clothing types
- Multiple poses and perspectives for each clothing item

The dataset is approximately 50GB and can be accessed [here](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html).

## Methodology

### Algorithms
We implement and compare two complementary approaches:

1. **Visual Similarity (CNN + KNN)**
   - Uses Convolutional Neural Networks to extract visual features
   - Applies K-Nearest Neighbors for similarity-based recommendations
   - Enables attribute-based filtering of fashion items

3. **Outfit Completion (Association Rule Mining)**
   - Discovers correlations between fashion items frequently worn together
   - Suggests complementary items to complete an outfit
   - Identifies commonly co-occurring fashion items

## Repository Structure

```
fashion-recommender/
├── data/                        # Datasets and cleaned dataset location
│   ├── Anno_course/             # Contains bounding box, fashion landmarks, category, and attribute annotations
│   └── cleaned_data/            # Location for cleaned data after preprocessing
│   ├── Eval/                    # Evaluation partitions
│   └── img/                     # Clothing and Fashion images
├── notebooks/                   # Model notebooks
│   ├── knn_cnn_implementation/  # CNN + KNN implementation
│   ├── nn_matrix/               # Matrix Factorization + Neural Networks
│   └── association_rules/       # Association Rule Mining
│   └── hybrid_recommendations/  # Main Hybrid Recommender System
│   └── eda_preprocessing/       # EDA and Preprocessing
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Setup Instructions

The two main notebooks are eda_preprocessing and hybrid_recommendations. The other notebooks are included for exploration, but do not necessarily add to our final implementation. For example, the nn_matrix notebook was abandoned after we decided to pursue other algorithms. 

1. **Clone the repository**
   ```
   git clone https://github.com/danish7x7/cmpe-256-project
   cd cmpe-256-project
   ```

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Download the DeepFashion Dataset from the [official website](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html)
   - Extract the files to the `data/` directory

4. **Preprocess the data**
   ```
   python notebooks/eda_preprocessing.ipynb
   ```

5. **Run the algorithms**
   ```
   python notebooks/hybrid_recommendations.ipynb
   ```

## References

- Liu, Z., Luo, P., Qiu, S., Wang, X., & Tang, X. (2016). DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

## License

This project is for educational purposes as part of a course assignment.
