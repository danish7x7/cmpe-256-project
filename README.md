# Fashion Recommender System

A fashion recommendation system that leverages deep learning and data mining techniques to provide personalized outfit recommendations based on user preferences and style profiles.

## Team Members
- Danishbir Singh Bhatti
- Jessica Lieu
- Sergio Talavera

## Project Overview

This project aims to develop an advanced fashion recommender system that goes beyond basic rating predictions by incorporating visual elements and style attributes. Using the DeepFashion Database, we implement multiple recommendation algorithms to suggest fashion items and complete outfits that match users' personal styles.

### Key Features
- Outfit recommendations based on user preferences
- Personalized fashion item recommendations
- Visual and stylistic similarity analysis between fashion items

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
We implement and compare three complementary approaches:

1. **Visual Similarity (CNN + KNN)**
   - Uses Convolutional Neural Networks to extract visual features
   - Applies K-Nearest Neighbors for similarity-based recommendations
   - Enables attribute-based filtering of fashion items

2. **Personalized Recommendations (Matrix Factorization + Neural Networks)**
   - Combines traditional matrix factorization techniques with neural networks
   - Creates a collaborative filtering system for personalized recommendations
   - Learns user preferences from interaction data

3. **Outfit Completion (Association Rule Mining)**
   - Discovers correlations between fashion items frequently worn together
   - Suggests complementary items to complete an outfit
   - Identifies commonly co-occurring fashion items

### Evaluation Metrics
- Precision@k
- Recall@k
- NDCG (Normalized Discounted Cumulative Gain)
- Hyperparameter tuning via grid search and cross-validation

## Repository Structure

```
fashion-recommender/
├── data/                     # Data processing scripts and processed datasets
├── models/                   # Model implementations
│   ├── cnn_knn/              # CNN + KNN implementation
│   ├── matrix_factorization/ # Matrix Factorization + Neural Networks
│   └── association_rules/    # Association Rule Mining
├── evaluation/               # Evaluation scripts and metrics
├── notebooks/                # Jupyter notebooks for exploration and visualization
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Setup Instructions

1. **Clone the repository**
   ```
   git clone https://github.com/your-username/fashion-recommender.git
   cd fashion-recommender
   ```

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Download the DeepFashion Dataset from the [official website](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html)
   - Extract the files to the `data/raw/` directory

4. **Preprocess the data**
   ```
   python data/preprocess.py
   ```

5. **Run the models**
   ```
   python models/train.py
   ```

## Results

Detailed results and model comparisons will be added upon completion of the project. We'll present findings through tables, graphs, and verbal descriptions to determine the most effective approach for fashion recommendation.

## References

- Liu, Z., Luo, P., Qiu, S., Wang, X., & Tang, X. (2016). DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
- [Additional references will be added as the project progresses]

## License

This project is for educational purposes as part of a course assignment.
