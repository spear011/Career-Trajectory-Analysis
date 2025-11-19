# EvolveGCN-H Training Pipeline for Career Trajectory Link Prediction

## Overview

This document explains the complete training pipeline for predicting future occupation transitions using temporal graph neural networks. The model learns from historical career transition patterns to predict which occupation transitions are likely to occur in the future.

---

## 1. Data Preparation Pipeline

### 1.1 Raw Data Sources

The pipeline starts with preprocessed career trajectory data containing:
- **User career histories**: Job records with occupation codes, start/end dates, locations
- **Occupation transitions**: Sequential changes between occupations for each individual
- **Temporal information**: Time-stamped career events spanning multiple years

### 1.2 Network Construction

**Step 1: Build Temporal Occupation Networks**

For each year in the study period (2017-2024):
- Extract all occupation transitions that occurred during that year
- Create a directed graph where:
  - **Nodes** = Unique occupation codes
  - **Edges** = Transitions from one occupation to another
  - **Edge weights** = Number of people who made that transition

**Step 2: Add Node Attributes**

Each occupation node is enriched with:
- **Employment count**: Number of workers in that occupation
- **Average wage**: Mean salary for workers in that occupation

**Step 3: Save as Temporal Graph Sequence**

The result is a sequence of graphs: G_2017, G_2018, ..., G_2024
- Each graph represents the occupation transition network for one year
- All graphs share the same set of nodes (all occupations across all years)
- Edge structures evolve over time as transition patterns change

---

## 2. Graph Representation for Neural Networks

### 2.1 Occupation Indexing

**Create Global Occupation Mapping**:
- Collect all unique occupations across all years
- Assign each occupation a unique integer index (0 to N-1)
- This ensures consistent node ordering across all temporal graphs

Example:
```
"Software Developer" → Index 0
"Data Scientist" → Index 1
"Project Manager" → Index 2
...
```

### 2.2 Adjacency Matrix Construction

**For each temporal snapshot**:

Convert the NetworkX graph to a sparse adjacency matrix:
- Matrix size: N × N (where N = total number of occupations)
- Entry (i,j) = weight of edge from occupation i to occupation j
- Zero if no transition exists

**Normalization**:
- Add self-loops (identity matrix)
- Compute degree matrix D
- Apply symmetric normalization: D^(-1/2) × A × D^(-1/2)
- This prevents numerical instability during training

**Storage Format**:
- Store as sparse tensor with (indices, values) for memory efficiency
- Format: {'idx': edge_indices, 'vals': edge_weights}

### 2.3 Node Feature Matrix

Create feature matrix F of size N × d (d = number of features):
- **Feature 0**: log(1 + employment_count) - log scale for numerical stability
- **Feature 1**: log(1 + average_wage) - log scale wage information
- Missing values are zero-filled

### 2.4 Node Masks

Create binary mask vector of size N × 1:
- All ones (every occupation is considered active in this implementation)
- Allows for handling of dynamic node sets in future extensions

---

## 3. Training Data Generation

### 3.1 Temporal Sliding Window Approach

**History Window**:
- For prediction at time t, use graphs from time 0 to t as historical context
- Example: To predict 2020 transitions, use graphs from 2017-2020

**Train/Validation/Test Split**:
- **Training set**: Early timesteps (e.g., timesteps with sufficient history)
- **Validation set**: Last few timesteps before test (e.g., 1-2 timesteps)
- **Test set**: Final timesteps (e.g., last 1-2 timesteps)

### 3.2 Link Prediction Task Setup

**Positive Examples** (Ground Truth):
- Edges that exist in the current year's graph
- These represent actual occupation transitions that occurred

**Negative Examples** (For Contrast):
- Randomly sample non-existent edges (no transition occurred)
- Ensure negative edges are:
  - Not in the positive set
  - Not self-loops
- Sample ratio: Typically 1:1 or 1:N negative to positive

**Example for one timestep**:
- Positive edges: 150 actual transitions
- Negative edges: 150 randomly sampled non-transitions
- Total training examples: 300 edge predictions

---

## 4. Model Architecture: EvolveGCN-H

### 4.1 Core Concept

The model learns how occupation networks evolve over time by:
1. Processing temporal sequence of graphs
2. Updating node representations based on historical patterns
3. Evolving the GCN parameters themselves using RNN mechanics

### 4.2 Architecture Components

**Layer 1: GRCU (Graph Recurrent Convolutional Unit)**

For each timestep t:
1. **Input**: 
   - Adjacency matrix A_t (normalized)
   - Node features H_t
   - Node masks M_t

2. **Weight Evolution** (RNN-style):
   - Use GRU mechanism to evolve GCN weight matrix W_t
   - W_t = GRU(W_{t-1}, H_t, M_t)
   - This allows the graph convolution to adapt to temporal patterns

3. **Graph Convolution**:
   - Apply evolved weights: H_{t+1} = σ(A_t × H_t × W_t)
   - σ = activation function (ReLU)
   - Result: Updated node embeddings

**Layer 2: Second GRCU**
- Stack another GRCU layer for deeper representations
- Input is output from first layer
- Same mechanism: evolve weights → apply convolution

**Output**:
- Final node embeddings for all occupations
- Each occupation has a learned vector representation

### 4.3 Key Innovation

Traditional GCNs: Fixed weights across all timesteps
EvolveGCN-H: Weights evolve using RNN, capturing temporal dynamics

---

## 5. Loss Computation

### 5.1 Edge Score Calculation

**For each edge (source_occupation, target_occupation)**:
1. Extract embeddings: e_source, e_target
2. Compute score: score = e_source · e_target (dot product)
3. Higher score = stronger predicted connection

### 5.2 Binary Cross-Entropy Loss

**Combine positive and negative examples**:
- Positive edges: Label = 1
- Negative edges: Label = 0

**Compute loss**:
- Apply sigmoid to convert scores to probabilities
- BCE Loss = -[y × log(σ(score)) + (1-y) × log(1-σ(score))]
- Average over all edges

**Intuition**:
- Model learns to assign high scores to real transitions
- Model learns to assign low scores to non-transitions

---

## 6. Training Loop

### 6.1 Single Epoch Process

**For each training timestep t**:

1. **Prepare data**:
   - Load historical graphs: [G_0, G_1, ..., G_t]
   - Load corresponding features and masks
   - Move all tensors to GPU if available

2. **Forward pass**:
   - Feed historical sequence through EvolveGCN-H
   - Get final node embeddings

3. **Generate predictions**:
   - Extract positive edges from current graph
   - Sample negative edges randomly
   - Compute scores for all edges

4. **Compute loss**:
   - Calculate BCE loss between predictions and labels

5. **Backward pass**:
   - Compute gradients
   - Update model parameters using optimizer (Adam/SGD)

6. **Accumulate metrics**:
   - Track average loss across all timesteps

### 6.2 Epoch Summary

- Average loss across all training timesteps
- Log every 10 timesteps for monitoring

---

## 7. Evaluation Metrics

### 7.1 Classification Metrics

**AUC (Area Under ROC Curve)**:
- Measures ability to distinguish positive from negative edges
- Range: 0.5 (random) to 1.0 (perfect)
- Interpretation: Probability that model ranks random positive edge higher than random negative edge

**AP (Average Precision)**:
- Precision-recall curve summary
- Emphasizes performance on positive class
- Better for imbalanced datasets

### 7.2 Ranking Metrics

**MAP (Mean Average Precision)**:
- For each query, compute precision at each relevant item
- Average these precisions
- Then average across all queries
- Measures quality of ranked recommendation list

**MRR (Mean Reciprocal Rank)**:
- Find position of first relevant item in ranked list
- MRR = 1 / position
- Measures how quickly model finds correct answer

**Example**:
Ranked predictions: [Negative, Positive, Negative, Positive, ...]
- Position of first positive: 2
- MRR = 1/2 = 0.5

---

## 8. Validation Strategy

### 8.1 Validation Set Evaluation

**After each training epoch**:

1. Switch model to evaluation mode (disable dropout, etc.)
2. Process validation timesteps (no gradient computation)
3. Generate predictions for validation edges
4. Compute all metrics: AUC, AP, MAP, MRR
5. Compare against best validation performance

### 8.2 Model Selection

**Early Stopping**:
- Track validation AUC across epochs
- If AUC doesn't improve for N epochs (patience), stop training
- Prevents overfitting to training data

**Best Model Saving**:
- Save model checkpoint when validation AUC improves
- Keep only the best performing model
- This model is used for final test evaluation

---

## 9. Test Evaluation

### 9.1 Final Assessment

**Load best model** (from validation):
- Restore model parameters from saved checkpoint
- Ensures we evaluate the best performing version

**Test set prediction**:
- Process test timesteps (held-out future periods)
- Generate predictions for test edges
- Compute all evaluation metrics

### 9.2 Interpretation

**High AUC/AP**: Model accurately distinguishes likely transitions from unlikely ones

**High MAP**: Model ranks true transitions high in recommendation list

**High MRR**: Model's top predictions are often correct

**Use case**: 
- Predict which career transitions are most likely for policy planning
- Identify emerging occupation pathways
- Forecast labor market dynamics

---

## 10. Complete Pipeline Flow

```
Raw Career Data
    ↓
[Build Temporal Networks]
    ↓
Year-by-year occupation transition graphs
    ↓
[Convert to Tensors]
    ↓
Sparse adjacency matrices + node features
    ↓
[Training Loop]
    ↓
For each epoch:
    For each training timestep:
        - Feed historical graphs to EvolveGCN-H
        - Predict edge scores
        - Compute BCE loss
        - Update model weights
    Validate on validation set
    Save if best performance
    ↓
[Early Stopping Triggered]
    ↓
Load best model
    ↓
[Test Evaluation]
    ↓
Report final metrics: AUC, AP, MAP, MRR
```

---

## 11. Key Design Choices

### 11.1 Why Temporal GNN?

- Career transitions are inherently temporal
- Network structure evolves over time (COVID impact, tech boom, etc.)
- EvolveGCN captures these dynamics better than static GNNs

### 11.2 Why Link Prediction?

- Directly models transition probability
- Interpretable as "which career moves are likely?"
- Supports recommendation and forecasting use cases

### 11.3 Why These Metrics?

- **AUC/AP**: Standard for binary classification
- **MAP/MRR**: Critical for ranking quality in recommendation systems
- Together they provide comprehensive evaluation

### 11.4 Why Negative Sampling?

- Full matrix prediction is computationally prohibitive (N² edges)
- Most transitions never occur (sparse network)
- Negative sampling balances computation and learning signal

---

## 12. Practical Considerations

### 12.1 Computational Requirements

- GPU recommended for sparse matrix operations
- Memory scales with: number of occupations × sequence length
- Batch processing across timesteps for efficiency

### 12.2 Hyperparameters

**Critical parameters**:
- Learning rate: Controls optimization speed
- Hidden dimensions: Capacity of node embeddings
- Number of GRCU layers: Model depth
- Negative sampling ratio: Positive/negative balance
- Early stopping patience: Overfitting control

### 12.3 Data Quality Impact

**Network quality depends on**:
- Sufficient transition observations per year
- Consistent occupation coding across time
- Representative sample of workforce
- Accurate temporal information

---

## 13. Output and Results

### 13.1 Training Outputs

**During training**:
- Epoch-by-epoch loss curves
- Validation metric progression
- Early stopping indicators

**Saved artifacts**:
- Best model checkpoint
- Training configuration
- Metric history

### 13.2 Final Results

**Quantitative**:
- Test set performance metrics
- Comparison against baselines
- Temporal performance analysis

**Qualitative**:
- Top predicted transitions for each occupation
- Emerging pathways identification
- Anomaly detection (unexpected transitions)

---

## Conclusion

This pipeline transforms raw career trajectory data into actionable predictions about future occupation transitions. By combining temporal graph neural networks with careful data preparation and comprehensive evaluation, the system learns meaningful patterns in labor market dynamics and provides reliable forecasts for workforce planning and policy analysis.