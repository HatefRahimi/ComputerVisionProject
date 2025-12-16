# Computer Vision Project

Implementation of classical and modern computer vision algorithms developed during the Computer Vision course at FAU Erlangen-Nürnberg (Summer 2025).

## Exercises

### Exercise 1: RANSAC & Robust Estimation
- **RANSAC**: Plane detection in 3D point clouds
- **PRANSAC (Preemptive RANSAC)**: Early termination for efficiency
- **MLESAC**: Maximum likelihood estimation variant
- Application: Box detection from ToF depth data

### Exercise 2: Writer Identification with VLAD
- **VLAD Encoding**: Vector of Locally Aggregated Descriptors
- **Custom SIFT**: Hellinger-normalized descriptors with angle forcing
- **Generalized Max Pooling**: Ridge regression-based aggregation
- **Multi-VLAD + PCA**: Multiple codebooks with whitening
- **E-SVM**: Exemplar SVM for refinement
- Dataset: ICDAR2017 Historical Writer Identification

### Exercise 3: Selective Search
- **Region Proposals**: Hierarchical segmentation merging
- **Similarity Metrics**: Color, texture, size, fill
- **Object Detection**: Region-based detection pipeline
- Dataset: Archaeological artifacts (Art History, Christian & Classical Archaeology)

### Exercise 4: Demosaicing & HDR Imaging
- **Bayer Pattern Demosaicing**: RGB reconstruction from RAW sensor data
- **HDR Fusion**: Combining multiple exposures
- **Tone Mapping**: iCAM06 algorithm
- **White Balance**: Gray world implementation

### Exercise 5: Face Recognition System
- **Face Detection**: MTCNN implementation
- **Face Tracking**: Template matching for video
- **Face Recognition**: FaceNet embeddings with k-NN
- **Face Clustering**: k-means for person re-identification
- **Evaluation**: DIR curves for open-set identification

## Technologies

`Python` `NumPy` `OpenCV` `scikit-learn` `SciPy` `Matplotlib` `MTCNN` `FaceNet` `rawpy`

## Setup

```bash
# Clone repository
git clone https://github.com/yourusername/computer-vision-project.git
cd computer-vision-project

# Install dependencies
pip install -r requirements.txt

# Run exercises
cd Exercise2
python main_individual.py --powernorm --gmp --gamma 1
```

## Project Structure

```
computer-vision-project/
├── Exercise1_RANSAC/
├── Exercise2_VLAD/
│   ├── main_individual.py
│   └── custom_sift_extractor.py
├── Exercise3_SelectiveSearch/
├── Exercise4_HDR/
└── Exercise5_FaceRecognition/
```

## Key Results

- **Exercise 1**: Robust plane detection with <5mm error
- **Exercise 2**: 0.75 mAP on writer identification (grade: 98/100)
- **Exercise 3**: ~2000 region proposals with high recall
- **Exercise 4**: Natural HDR images with proper color reproduction
- **Exercise 5**: >90% accuracy on closed-set face identification

## License

MIT License - Academic coursework at FAU Erlangen-Nürnberg
