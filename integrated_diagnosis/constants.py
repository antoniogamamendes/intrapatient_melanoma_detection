import torch


"""
File to set some parameters
"""

# Parameters for general training
early_stopping_patience = 30
epochs = 45
learning_rate = 1e-5
learning_rate_combined = 1e-5
batch_size = 16
network = 'EfficientNetB2'
weights = torch.Tensor([0.02, 0.98])
temperature = 0.05
update_embeddings_frequency = 10


# Parameters for logistic regression (only) training
log_reg_epochs = 5000
log_reg_learning_rate = 1e-6
log_reg_batch_size = 32  # 128
log_reg_weights = torch.Tensor([0.2, 0.8])





