import pandas
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import time
import models
import argparse
import sklearn.metrics as metrics
import numpy as np
import dataset_log_reg
from dataset import train_dataloader, val_dataloader, train_dataset, val_dataset
from utils import EarlyStopping, LRScheduler
from tqdm import tqdm
from constants import *
from sklearn.metrics.pairwise import cosine_similarity

matplotlib.style.use('ggplot')

# empty cache
torch.cuda.empty_cache()

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--lr-scheduler', dest='lr_scheduler', action='store_true')
parser.add_argument('--early-stopping', dest='early_stopping', action='store_true')
args = vars(parser.parse_args())

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")


def get_features(name):
    def hook(model, input, output):
        features_dict[name] = output.detach()

    return hook


# instantiate the model
if network == 'EfficientNetB2':
    model = models.EfficientNetB2_and_LR(pretrained=True, requires_grad=True).to(device)

    # Copy weights to do a warm start
    # model.load_state_dict(torch.load(f"C:\\Users\\AntonioM\\PycharmProjects\\baseline_and_lr_version6\\outputs\\es_model.pth"))
    # model.combinedOutput.load_state_dict(torch.load(f"C:\\Users\\AntonioM\\Desktop\\Baseline_and_LR_Version6\\combined_warmstart\\es_model_combined.pth"))
    # model.efficientNetB2.load_state_dict(torch.load(f"C:\\Users\\AntonioM\\Desktop\\efficientNetB2\\es_model.pth"))
    # model.logisticRegression.load_state_dict(torch.load('C:\\Users\\AntonioM\\Desktop\\LogisticRegression_EfficientNetB2_without_n_lesions\\es_model.pth'))

    # Instantiate features dictionary
    features_dict = {}
    model.efficientNetB2.classifier[0].register_forward_hook(get_features('feats'))

# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

# optimizer
optimizer = optim.Adam([{"params": model.combinedOutput.parameters(), "lr": learning_rate_combined}], lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate_combined)


# loss function
criterion = nn.CrossEntropyLoss(weight=weights.to(device))

# strings to save the loss plot, accuracy plot, and model with different ...
# ... names according to the training type
# if not using `--lr-scheduler` or `--early-stopping`, then use simple names
loss_plot_name = 'loss'
acc_plot_name = 'accuracy'
model_name = 'resnet18'

# either initialize early stopping or learning rate scheduler
if args['lr_scheduler']:
    print('INFO: Initializing learning rate scheduler')
    lr_scheduler = LRScheduler(optimizer)
    # change the accuracy, loss plot names and model name
    loss_plot_name = 'lrs_loss'
    acc_plot_name = 'lrs_accuracy'
    model_name = 'lrs_model'
if args['early_stopping']:
    print('INFO: Initializing early stopping')
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    # change the accuracy, loss plot names and model name
    loss_plot_name = 'es_loss'
    acc_plot_name = 'es_accuracy'
    model_name = 'es_model'

isic_training_df = pandas.read_excel(f"C:\\Users\\AntonioM\\Desktop\\Training_Patients_ISIC_2020.xlsx", index_col=0)
isic_validation_df = pandas.read_excel(f"C:\\Users\\AntonioM\\Desktop\\Validation_Patients_ISIC_2020.xlsx", index_col=0)

patient_stats_training_df = pandas.read_excel(
    f"C:\\Users\\AntonioM\\Desktop\\Patients_stats\\Training_{network}_ISIC_2020.xlsx", index_col=0)
patient_stats_validation_df = pandas.read_excel(
    f"C:\\Users\\AntonioM\\Desktop\\Patients_stats\\Validation_{network}_ISIC_2020.xlsx", index_col=0)

embeddings_mean_value = np.mean(patient_stats_training_df[['max_l2', 'max_cos', 'min_l2', 'min_cos', 'avg_l2',
                                                           'avg_cos', 'median_l2', 'median_cos', 'std_l2', 'std_cos']],
                                axis=0).values


def get_patients_stats(partition, corresponding_image_names):
    global patient_stats_training_df
    global embeddings_mean_value
    global patient_stats_validation_df

    if partition == 'Training':
        df = isic_training_df
        patients_stats_df = patient_stats_training_df
    else:
        df = isic_validation_df
        patients_stats_df = patient_stats_validation_df

    patients = df[df['image_name'].isin(corresponding_image_names)]
    patients = np.array(patients['patient_id'])

    patients_stats_list = list()
    for patient in patients:
        patient_stats = patients_stats_df[patients_stats_df['patient_id'] == patient]
        patient_stats = patient_stats.drop(columns=['patient_id', 'n_melanomas', 'n_lesions'], errors='ignore').copy()
        patients_stats_list.append(np.array(patient_stats))

    patients_stats_array = np.squeeze(np.array(patients_stats_list))
    # Normalize the embeddings because the loaded LogReg model was trained like this
    patients_stats_array = patients_stats_array - embeddings_mean_value
    patients_stats_tensor = torch.Tensor(patients_stats_array)
    patients_stats_tensor = patients_stats_tensor.to(device)

    return patients_stats_tensor, patients


def logit_normalization(logits, temperature):
    norms = torch.norm(logits, p=2, dim=-1, keepdim=True) + 1e-7  # adding constant for stability
    return torch.div(logits, norms) / temperature  # normalized logits


def custom_weighted_binary_cross_entropy_with_logit_normalization(logits, targets, weights=weights,
                                                                  temperature=temperature):
    logits = logit_normalization(logits, temperature)

    exp_logits = torch.exp(logits)
    exp_sum = torch.sum(exp_logits, dim=1)
    benign_terms = (1 - targets) * weights[0] * torch.log(exp_logits[:, 0] / exp_sum)
    melanoma_terms = targets * weights[1] * torch.log(exp_logits[:, 1] / exp_sum)

    return -0.5 * torch.sum(torch.cat((benign_terms, melanoma_terms)))


def distance_metrics(features):
    def euclidean_distance(points, item):
        return np.array(np.linalg.norm((item * np.ones((len(points), len(item)))) - points, axis=1))

    def cosine_distance(points):
        return 1 - cosine_similarity(points)

    l2_matrix = np.array([euclidean_distance(features, item) for item in features])
    l2_matrix = l2_matrix[np.triu_indices_from(l2_matrix, k=1)]  # upper triangular

    cosine_matrix = cosine_distance(features)
    cosine_matrix = cosine_matrix[np.triu_indices_from(cosine_matrix, k=1)]  # upper triangular

    return l2_matrix, cosine_matrix


def update_patient_embeddings(all_features, patients_id, partition):
    global patient_stats_training_df
    global embeddings_mean_value
    global patient_stats_validation_df

    columns_list = ['patient_id', 'n_melanomas', 'max_l2', 'max_cos', 'min_l2', 'min_cos', 'avg_l2',
                    'avg_cos', 'median_l2', 'median_cos', 'std_l2', 'std_cos']
    patients_stats_df = pandas.DataFrame(columns=columns_list)

    for patient in np.unique(patients_id):
        patient_indexes = np.where(patients_id == patient)[0]
        patient_features = all_features[patient_indexes, :]

        patient_l2_dists, patient_cosine_dists = distance_metrics((patient_features.cpu().numpy()))

        new_row = {'patient_id': patient,
                   'max_l2': np.amax(patient_l2_dists), 'max_cos': np.amax(patient_cosine_dists),
                   'min_l2': np.amin(patient_l2_dists), 'min_cos': np.amin(patient_cosine_dists),
                   'avg_l2': np.mean(patient_l2_dists), 'avg_cos': np.mean(patient_cosine_dists),
                   'median_l2': np.median(patient_l2_dists), 'median_cos': np.median(patient_cosine_dists),
                   'std_l2': np.std(patient_l2_dists), 'std_cos': np.std(patient_cosine_dists),
                   'n_melanomas': patient_stats_training_df[patient_stats_training_df['patient_id'] == patient][
                       'n_melanomas'].values[0] if partition == 'Training'
                   else patient_stats_validation_df[patient_stats_validation_df['patient_id'] == patient][
                       'n_melanomas'].values[0]}
        # append row to the dataframe
        patients_stats_df = patients_stats_df.append(new_row, ignore_index=True)

    if partition == 'Training':
        # print(embeddings_mean_value)
        patient_stats_training_df = patients_stats_df.copy()
        aux_patients_stats = patients_stats_df.copy()
        aux_patients_stats.drop(columns=['patient_id', 'n_melanomas'], inplace=True, errors='ignore')
        embeddings_mean_value = np.mean(aux_patients_stats, axis=0).to_numpy()
        # print(embeddings_mean_value)
    else:  # partition == 'Validation'
        # print(embeddings_mean_value)
        patient_stats_validation_df = patients_stats_df.copy()
        aux_patients_stats = patients_stats_df.copy()
        aux_patients_stats.drop(columns=['patient_id', 'n_melanomas'], inplace=True, errors='ignore')
        # print(embeddings_mean_value)


def fit(model, train_dataloader, train_dataset, optimizer, update_embeddings=False):
    print('Training')
    model.train()
    train_baseline_running_loss = 0.0
    train_combined_running_loss = 0.0
    train_baseline_running_correct = 0
    train_combined_running_correct = 0
    counter = 0
    total = 0
    all_train_features = torch.tensor([]).to(device)
    all_patients = np.array([])
    prog_bar = tqdm(enumerate(train_dataloader), total=int(len(train_dataset) / train_dataloader.batch_size))
    for i, data in prog_bar:
        counter += 1
        data, target, image_names = data[0].to(device), data[1].to(device), data[2]

        patient_stats_tensor, patients_id = get_patients_stats(partition='Training',
                                                               corresponding_image_names=image_names)

        total += target.size(0)
        optimizer.zero_grad()
        baseline_outputs, _, combined_outputs = model(data, patient_stats_tensor)
        all_train_features = torch.cat((all_train_features, features_dict['feats']))
        all_patients = np.concatenate((all_patients, patients_id), axis=None)

        # Baseline Loss and Accuracy
        # loss_baseline = criterion(baseline_outputs, target)
        loss_baseline = custom_weighted_binary_cross_entropy_with_logit_normalization(baseline_outputs, target)

        train_baseline_running_loss += loss_baseline.item()
        _, preds_baseline = torch.max(baseline_outputs.data, 1)
        train_baseline_running_correct += (preds_baseline == target).sum().item()

        # Combined (Baseline + Logistic Regression) Loss and Accuracy
        # loss_combined = criterion(combined_outputs, target)
        loss_combined = custom_weighted_binary_cross_entropy_with_logit_normalization(combined_outputs, target)

        train_combined_running_loss += loss_combined.item()
        _, preds_combined = torch.max(combined_outputs.data, 1)
        train_combined_running_correct += (preds_combined == target).sum().item()

        # Backpropagation
        loss_combined.backward()
        optimizer.step()

    if update_embeddings:
        update_patient_embeddings(all_train_features, all_patients, 'Training')

    train_baseline_loss = train_baseline_running_loss / counter
    train_baseline_accuracy = 100. * train_baseline_running_correct / total
    train_combined_loss = train_combined_running_loss / counter
    train_combined_accuracy = 100. * train_combined_running_correct / total

    return train_combined_loss, train_combined_accuracy, train_baseline_loss, train_baseline_accuracy, \
           torch.squeeze(all_train_features)


def validate(model, val_dataloader, val_dataset, update_embeddings=False):
    print('\nValidating')
    model.eval()
    val_combined_running_loss = 0.0
    val_baseline_running_loss = 0.0
    val_combined_running_correct = 0
    val_baseline_running_correct = 0
    counter = 0
    total = 0
    all_val_baseline_preds = torch.tensor([]).to(device)
    all_val_combined_preds = torch.tensor([]).to(device)
    all_val_targets = torch.tensor([]).to(device)
    all_val_features = torch.tensor([]).to(device)
    all_patients = np.array([])
    prog_bar = tqdm(enumerate(val_dataloader), total=int(len(val_dataset) / val_dataloader.batch_size))
    with torch.no_grad():
        for i, data in prog_bar:
            counter += 1
            data, target, image_names = data[0].to(device), data[1].to(device), data[2]

            patient_stats_tensor, patients_id = get_patients_stats(partition='Validation',
                                                                   corresponding_image_names=image_names)

            total += target.size(0)
            baseline_outputs, _, combined_outputs = model(data, patient_stats_tensor)
            all_val_features = torch.cat((all_val_features, features_dict['feats']))
            all_patients = np.concatenate((all_patients, patients_id), axis=None)

            # Baseline Loss and Accuracy
            # loss_baseline = criterion(baseline_outputs, target)
            loss_baseline = custom_weighted_binary_cross_entropy_with_logit_normalization(baseline_outputs, target)

            val_baseline_running_loss += loss_baseline.item()
            _, preds_baseline = torch.max(baseline_outputs.data, 1)
            val_baseline_running_correct += (preds_baseline == target).sum().item()

            # Combined (Baseline + Logistic Regression) Loss and Accuracy
            # loss_combined = criterion(combined_outputs, target)
            loss_combined = custom_weighted_binary_cross_entropy_with_logit_normalization(combined_outputs, target)

            val_combined_running_loss += loss_combined.item()
            _, preds_combined = torch.max(combined_outputs.data, 1)
            val_combined_running_correct += (preds_combined == target).sum().item()

            # Save Predictions and Target Labels
            all_val_baseline_preds = torch.cat((all_val_baseline_preds, preds_baseline))
            all_val_combined_preds = torch.cat((all_val_combined_preds, preds_combined))
            all_val_targets = torch.cat((all_val_targets, target))

        if update_embeddings:
            update_patient_embeddings(all_val_features, all_patients, 'Validation')

        # Metrics per image
        val_baseline_loss = val_baseline_running_loss / counter
        val_baseline_accuracy = 100. * val_baseline_running_correct / total
        confusion = metrics.confusion_matrix(all_val_targets.cpu(), all_val_baseline_preds.cpu())
        print("Confusion Matrix (baseline)\n", confusion)
        true_benign_rate_baseline = (confusion[0, 0] / confusion.sum(axis=1)[0]) * 100
        true_malign_rate_baseline = (confusion[1, 1] / confusion.sum(axis=1)[1]) * 100
        balanced_acc_baseline = metrics.balanced_accuracy_score(all_val_targets.cpu(),
                                                                all_val_baseline_preds.cpu()) * 100
        precision_baseline = metrics.precision_score(all_val_targets.cpu(), all_val_baseline_preds.cpu())
        f1_score_baseline = metrics.f1_score(all_val_targets.cpu(), all_val_baseline_preds.cpu())
        auc_baseline = metrics.roc_auc_score(all_val_targets.cpu(), all_val_baseline_preds.cpu())

        val_combined_loss = val_combined_running_loss / counter
        val_combined_accuracy = 100. * val_combined_running_correct / total
        confusion = metrics.confusion_matrix(all_val_targets.cpu(), all_val_combined_preds.cpu())
        print("Confusion Matrix (combined)\n", confusion)
        true_benign_rate_combined = (confusion[0, 0] / confusion.sum(axis=1)[0]) * 100
        true_malign_rate_combined = (confusion[1, 1] / confusion.sum(axis=1)[1]) * 100
        balanced_acc_combined = metrics.balanced_accuracy_score(all_val_targets.cpu(),
                                                                all_val_combined_preds.cpu()) * 100
        precision_combined = metrics.precision_score(all_val_targets.cpu(), all_val_combined_preds.cpu())
        f1_score_combined = metrics.f1_score(all_val_targets.cpu(), all_val_combined_preds.cpu())
        auc_combined = metrics.roc_auc_score(all_val_targets.cpu(), all_val_combined_preds.cpu())

        # Metrics per patient
        df = pandas.DataFrame({'patient_id': all_patients,
                               'val_combined_preds': (all_val_combined_preds.cpu()).numpy(),
                               'label': (all_val_targets.cpu()).numpy()})

        # Compute TBR, TMR, AUC per patient
        patients_confusion_combined = np.zeros((2, 2))
        for patient in np.unique(all_patients):
            individual_df = df[df['patient_id'] == patient].copy()
            individual_df.reset_index(drop=True)
            n_melanomas = individual_df['label'].sum()
            n_melanomas_pred_combined = individual_df['val_combined_preds'].sum()

            if n_melanomas == 0:
                if n_melanomas_pred_combined == 0:
                    patients_confusion_combined[0, 0] += 1
                else:
                    patients_confusion_combined[0, 1] += 1

            else:  # n_melanomas >= 1
                if n_melanomas_pred_combined == 0:
                    patients_confusion_combined[1, 0] += 1
                else:
                    patients_confusion_combined[1, 1] += 1

        tbr_per_patient = (patients_confusion_combined[0, 0] / patients_confusion_combined.sum(axis=1)[0])
        tmr_per_patient = (patients_confusion_combined[1, 1] / patients_confusion_combined.sum(axis=1)[1])
        auc_per_patient = 0.5 * (tbr_per_patient + tmr_per_patient)

        return val_combined_loss, val_combined_accuracy, true_benign_rate_combined, true_malign_rate_combined, \
               balanced_acc_combined, precision_combined, f1_score_combined, auc_combined, \
               val_baseline_loss, val_baseline_accuracy, true_benign_rate_baseline, true_malign_rate_baseline, \
               balanced_acc_baseline, precision_baseline, f1_score_baseline, auc_baseline, \
               torch.squeeze(all_val_features), tbr_per_patient, tmr_per_patient, auc_per_patient


def print_time_elapsed(start_time):
    end = time.time()
    hours, rem = divmod(end - start_time, 3600)
    minutes, seconds = divmod(rem, 60)

    if hours > 0:
        print(f"Training time: {hours}hrs {minutes}min {seconds:.0f}sec")
    else:
        print(f"Training time: {minutes}min {seconds:.0f}sec")


def train_logistic_regression_only(log_reg_model, log_reg_optimizer, log_reg_criterion):
    def get_dataset_and_dataloader():
        log_reg_train_dataset = dataset_log_reg.Dataset(patient_stats_training_df)
        log_reg_val_dataset = dataset_log_reg.Dataset(patient_stats_validation_df)
        log_reg_train_dataloader = dataset_log_reg.DataLoader(log_reg_train_dataset, batch_size=log_reg_batch_size,
                                                              shuffle=True)
        log_reg_val_dataloader = dataset_log_reg.DataLoader(log_reg_val_dataset, batch_size=log_reg_batch_size,
                                                            shuffle=False)

        return log_reg_train_dataset, log_reg_train_dataloader, log_reg_val_dataset, log_reg_val_dataloader

    def fit_logistic_regression_only(model, train_dataset, train_dataloader, optimizer, log_reg_criterion):
        # print('Training')

        # Unfreeze weights
        for param in model.parameters():
            param.requires_grad = True

        model.train()
        train_running_loss = 0.0
        train_running_correct = 0
        counter = 0
        total = 0
        prog_bar = tqdm(enumerate(train_dataloader), total=int(len(train_dataset) / train_dataloader.batch_size),
                        disable=True)
        for i, data in prog_bar:
            counter += 1
            data, target = data[0].to(device), data[1].to(device)
            total += target.size(0)
            optimizer.zero_grad()
            outputs = model(data)
            loss = log_reg_criterion(outputs, target)
            train_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            train_running_correct += (preds == target).sum().item()
            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / counter
        train_accuracy = 100. * train_running_correct / total
        return train_loss, train_accuracy

    def validate_logistic_regression_only(model, val_dataset, val_dataloader, log_reg_criterion):
        # print('\nValidating')

        # Freeze weights
        for param in model.parameters():
            param.requires_grad = False

        model.eval()
        val_running_loss = 0.0
        val_running_correct = 0
        counter = 0
        total = 0
        all_val_preds = torch.tensor([]).to(device)
        all_val_targets = torch.tensor([]).to(device)
        prog_bar = tqdm(enumerate(val_dataloader), total=int(len(val_dataset) / val_dataloader.batch_size),
                        disable=True)
        with torch.no_grad():
            for i, data in prog_bar:
                counter += 1
                data, target = data[0].to(device), data[1].to(device)
                total += target.size(0)
                outputs = model(data)
                loss = log_reg_criterion(outputs, target)

                val_running_loss += loss.item()
                _, preds = torch.max(outputs.data, 1)
                val_running_correct += (preds == target).sum().item()
                all_val_preds = torch.cat((all_val_preds, preds))
                all_val_targets = torch.cat((all_val_targets, target))

            val_loss = val_running_loss / counter
            val_accuracy = 100. * val_running_correct / total
            confusion = metrics.confusion_matrix(all_val_targets.cpu(), all_val_preds.cpu())
            # print("Confusion Matrix \n", confusion)
            true_benign_rate = (confusion[0, 0] / confusion.sum(axis=1)[0]) * 100
            true_malign_rate = (confusion[1, 1] / confusion.sum(axis=1)[1]) * 100
            balanced_acc = metrics.balanced_accuracy_score(all_val_targets.cpu(), all_val_preds.cpu()) * 100
            precision = metrics.precision_score(all_val_targets.cpu(), all_val_preds.cpu())
            f1_score = metrics.f1_score(all_val_targets.cpu(), all_val_preds.cpu())
            auc = metrics.roc_auc_score(all_val_targets.cpu(), all_val_preds.cpu())

            return val_loss, val_accuracy, true_benign_rate, true_malign_rate, balanced_acc, precision, f1_score, auc

    # datasets and dataloaders
    log_reg_train_dataset, log_reg_train_dataloader, log_reg_val_dataset, log_reg_val_dataloader = get_dataset_and_dataloader()

    # train and validate
    train_loss, train_accuracy = fit_logistic_regression_only(log_reg_model, log_reg_train_dataset,
                                                              log_reg_train_dataloader, log_reg_optimizer,
                                                              log_reg_criterion)
    val_loss, val_accuracy, true_benign_rate, true_malign_rate, balanced_acc, precision, f1_score, auc = validate_logistic_regression_only(
        log_reg_model, log_reg_val_dataset, log_reg_val_dataloader, log_reg_criterion)

    return true_benign_rate, true_malign_rate, auc, val_loss, log_reg_model.state_dict()


# lists to store per-epoch loss and accuracy values
train_loss, train_accuracy, train_baseline_loss, train_baseline_accuracy = [], [], [], []
val_loss, val_accuracy, val_true_benign_rate, val_true_malign_rate, val_balanced_acc, val_precision, val_f1_score, \
val_auc, val_baseline_loss, val_baseline_accuracy, val_true_benign_rate_baseline, val_true_malign_rate_baseline, \
val_balanced_acc_baseline, val_precision_baseline, val_f1_score_baseline, \
val_auc_baseline, val_tbr_per_patient, val_tmr_per_patient, val_auc_per_patient = [], [], [], [], [], [], [], [], [], \
                                                                                  [], [], [], [], [], [], [], [], [], []
start = time.time()
best_val_score = 0.0
best_val_score_baseline = 0.0
best_log_reg_auc = 0.71
for epoch in range(epochs):
    print(f"Epoch {epoch + 1} of {epochs}")

    # Define boolean to switch the embeddings update
    if ((epoch + 1) % update_embeddings_frequency) == 0:
        update_lower_path_or_embeddings = True
    else:
        update_lower_path_or_embeddings = False

    train_epoch_loss, train_epoch_accuracy, train_epoch_baseline_loss, train_epoch_baseline_accuracy, train_features = fit(
        model, train_dataloader, train_dataset, optimizer, update_lower_path_or_embeddings
    )
    val_epoch_loss, val_epoch_accuracy, val_epoch_true_benign_rate, val_epoch_true_malign_rate, val_epoch_balanced_acc, \
    val_epoch_precision, val_epoch_f1_score, val_epoch_auc, val_epoch_baseline_loss, val_epoch_baseline_accuracy, \
    val_epoch_true_benign_rate_baseline, val_epoch_true_malign_rate_baseline, val_epoch_balanced_acc_baseline, \
    val_epoch_precision_baseline, val_epoch_f1_score_baseline, val_epoch_auc_baseline, val_features, \
    val_epoch_tbr_per_patient, val_epoch_tmr_per_patient, val_epoch_auc_per_patient = validate(
        model, val_dataloader, val_dataset, update_lower_path_or_embeddings
    )

    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
    val_true_benign_rate.append(val_epoch_true_benign_rate)
    val_true_malign_rate.append(val_epoch_true_malign_rate)
    val_balanced_acc.append(val_epoch_balanced_acc)
    val_precision.append(val_epoch_precision)
    val_f1_score.append(val_epoch_f1_score)
    val_auc.append(val_epoch_auc)
    train_baseline_loss.append(train_epoch_baseline_loss)
    train_baseline_accuracy.append(train_epoch_baseline_accuracy)
    val_baseline_loss.append(val_epoch_baseline_loss)
    val_baseline_accuracy.append(val_epoch_baseline_accuracy)
    val_true_benign_rate_baseline.append(val_epoch_true_benign_rate_baseline)
    val_true_malign_rate_baseline.append(val_epoch_true_malign_rate_baseline)
    val_balanced_acc_baseline.append(val_epoch_balanced_acc_baseline)
    val_precision_baseline.append(val_epoch_precision_baseline)
    val_f1_score_baseline.append(val_epoch_f1_score_baseline)
    val_auc_baseline.append(val_epoch_auc_baseline)
    val_tbr_per_patient.append(val_epoch_tbr_per_patient)
    val_tmr_per_patient.append(val_epoch_tmr_per_patient)
    val_auc_per_patient.append(val_epoch_auc_per_patient)

    # Save the best models
    val_epoch_score = (val_epoch_auc + val_epoch_auc_per_patient) / 2

    print(f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}%")
    print(
        f"Train Loss (baseline): {train_epoch_baseline_loss:.4f}, Train Acc (baseline): {train_epoch_baseline_accuracy:.2f}%")
    print(
        f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.2f}%, Val Balanced Acc: {val_epoch_balanced_acc:.2f}%')
    print(
        f'Val TBR per patient: {val_epoch_tbr_per_patient:.2f}, Val TMR per patient: {val_epoch_tmr_per_patient:.2f}, Val AUC per patient: {val_epoch_auc_per_patient:.2f}, Epoch Score: {val_epoch_score:.2f}')
    print(
        f'Val Loss (baseline): {val_epoch_baseline_loss:.4f}, Val Acc (baseline): {val_epoch_baseline_accuracy:.2f}%, Val Balanced Acc (baseline): {val_epoch_balanced_acc_baseline:.2f}%')

    if best_val_score < val_epoch_score:
        best_val_score = val_epoch_score
        print(f"Saving best model until now with best_val_score = {best_val_score:.2f}")
        torch.save(model.state_dict(),
                   f"C:\\Users\\AntonioM\\PycharmProjects\\baseline_and_lr_version6\\outputs\\{model_name}.pth")
        torch.save(model.combinedOutput.state_dict(),
                   f"C:\\Users\\AntonioM\\PycharmProjects\\baseline_and_lr_version6\\outputs\\{model_name}_combined.pth")

    val_epoch_score_baseline = val_epoch_auc_baseline
    if best_val_score_baseline < val_epoch_score_baseline:
        best_val_score_baseline = val_epoch_score_baseline
        print(
            f"Saving best model until now with best_val_score_baseline (auc [baseline]) = {best_val_score_baseline:.2f}")
        if network == 'EfficientNetB2':
            torch.save(model.efficientNetB2.state_dict(),
                       f"C:\\Users\\AntonioM\\PycharmProjects\\baseline_and_lr_version6\\outputs\\{model_name}_baseline.pth")

    # Update scheduler and early stopping
    if args['lr_scheduler']:
        lr_scheduler(val_epoch_loss)
    if args['early_stopping']:
        early_stopping(val_epoch_loss)
        if early_stopping.early_stop:
            break

    if update_lower_path_or_embeddings:
        # Train LR only
        # instantiate log_reg (only) model
        log_reg_model = models.LogisticRegression(n_classes=2, n_features=10).to(device)
        log_reg_model.load_state_dict(model.logisticRegression.state_dict())  # import weights from global model

        # define optimizer, scheduler, early stopping, and criterion for the LR train
        log_reg_optimizer = optim.Adam(log_reg_model.parameters(), lr=log_reg_learning_rate)
        log_reg_criterion = nn.CrossEntropyLoss(weight=log_reg_weights.to(device))
        log_reg_scheduler = LRScheduler(log_reg_optimizer, patience=log_reg_epochs // 300)
        log_reg_early_stopping = EarlyStopping(patience=log_reg_epochs // 5, mute=True)

        for log_reg_epoch in range(log_reg_epochs):
            log_reg_tbr, log_reg_tmr, log_reg_auc, log_reg_val_loss, log_reg_model_state_dict = train_logistic_regression_only(
                log_reg_model, log_reg_optimizer, log_reg_criterion)
            if (log_reg_epoch + 1) % (log_reg_epochs // 10) == 0 or log_reg_epoch == 0:
                print(f"Epoch {log_reg_epoch + 1} of {log_reg_epochs}")
                print(
                    f'log_reg_tbr: {log_reg_tbr:.2f}%, log_reg_tmr: {log_reg_tmr:.2f}%, log_reg_auc: {log_reg_auc:.2f}')

            # Copy new weights to the general model if there is improvement in the auc
            if log_reg_auc > best_log_reg_auc:
                best_log_reg_auc = log_reg_auc
                print(f"Epoch {log_reg_epoch + 1}: Saving Best Log_Reg_Model with auc = {best_log_reg_auc: .2f}")
                model.logisticRegression.load_state_dict(log_reg_model_state_dict)
                torch.save(log_reg_model_state_dict,
                           f"C:\\Users\\AntonioM\\PycharmProjects\\baseline_and_lr_version6\\outputs\\{model_name}_log_reg.pth")

            # Update scheduler and early stopping
            log_reg_scheduler(log_reg_val_loss)
            log_reg_early_stopping(log_reg_val_loss)
            if log_reg_early_stopping.early_stop:
                break

    print("combined weights: ", model.combinedOutput.weight)

print_time_elapsed(start)

print('Saving loss and accuracy plots...')
# accuracy plot
plt.figure(figsize=(10, 7))
plt.plot(train_accuracy, color='orange', label='train accuracy')
plt.plot(train_baseline_accuracy, linestyle='--', color='orange', label='baseline train accuracy')
plt.plot(val_accuracy, color='black', label='validation accuracy')
plt.plot(val_baseline_accuracy, linestyle='--', color='black', label='baseline validation accuracy')
plt.plot(val_true_benign_rate, linestyle='--', color='green', label='validation TBR')
plt.plot(val_true_malign_rate, linestyle='--', color='red', label='validation TMR')
plt.plot(val_balanced_acc, linestyle=':', color='blue', label='validation balanced accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f"C:\\Users\\AntonioM\\PycharmProjects\\baseline_and_lr_version6\\outputs\\{acc_plot_name}.png")
plt.show()

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(train_baseline_loss, linestyle='--', color='orange', label='baseline train loss')
plt.plot(val_loss, color='black', label='validation loss')
plt.plot(val_baseline_loss, linestyle='--', color='black', label='baseline validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"C:\\Users\\AntonioM\\PycharmProjects\\baseline_and_lr_version6\\outputs\\{loss_plot_name}.png")
plt.show()

# Save metrics into a file
df = pandas.DataFrame({'train_acc': train_accuracy, 'train_acc_baseline': train_baseline_accuracy,
                       'train_loss': train_loss, 'train_loss_baseline': train_baseline_loss,
                       'val_acc': val_accuracy, 'val_acc_baseline': val_baseline_accuracy,
                       'val_loss': val_loss, 'val_loss_baseline': val_baseline_loss,
                       'val_TBR': val_true_benign_rate, 'val_TBR_baseline': val_true_benign_rate_baseline,
                       'val_TMR': val_true_malign_rate, 'val_TMR_baseline': val_true_malign_rate_baseline,
                       'val_f1_score': val_f1_score, 'val_f1_score_baseline': val_f1_score_baseline,
                       'val_precision': val_precision, 'val_precision_baseline': val_precision_baseline,
                       'val_auc': val_auc, 'val_auc_baseline': val_auc_baseline,
                       'val_tbr_per_patient': val_tbr_per_patient, 'val_tmr_per_patient': val_tmr_per_patient,
                       'val_auc_per_patient': val_auc_per_patient})
df.index = np.arange(1, len(df) + 1)  # so index starts at one
df.to_excel(f"C:\\Users\\AntonioM\\PycharmProjects\\baseline_and_lr_version6\\outputs\\metrics.xlsx")

# Save configs
configs = pandas.DataFrame({'batch_size': [batch_size], 'log_reg_batch_size': [log_reg_batch_size],
                            'early_stopping_patience': [early_stopping_patience],
                            'epochs': [epochs], 'log_reg_epochs': [log_reg_epochs],
                            'learning_rate': [learning_rate], 'log_reg_learning_rate': [log_reg_learning_rate],
                            'temperature': [temperature], 'network': [network],
                            'update_embeddings_frequency': [update_embeddings_frequency]})
configs.to_excel(f"C:\\Users\\AntonioM\\PycharmProjects\\baseline_and_lr_version6\\outputs\\configs.xlsx")

print('TRAINING COMPLETE')
