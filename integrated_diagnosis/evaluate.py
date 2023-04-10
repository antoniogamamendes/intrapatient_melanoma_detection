import pandas
import torch
import torch.nn as nn
import time
import models
import sklearn.metrics as metrics
import numpy as np
from dataset import val_dataloader
from dataset import val_dataset
from tqdm import tqdm
from constants import weights
from sklearn.metrics.pairwise import cosine_similarity


# Empty cache
torch.cuda.empty_cache()


def get_features(name):
    def hook(model, input, output):
        features_dict[name] = output.detach()

    return hook

# Define some parameters
network = 'EfficientNetB2'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

# instantiate the model
# CNN + LR
model = models.EfficientNetB2_and_LR(pretrained=False, requires_grad=False).to(device)
model.load_state_dict(torch.load(f"C:\\Users\\AntonioM\\Desktop\\Baseline_and_LR_Version6\\es_model.pth"))

# Instantiate features dictionary
features_dict = {}
model.efficientNetB2.classifier[0].register_forward_hook(get_features('feats'))

# CNN with baseline weights
cnn_alone = models.efficient_net_b2(pretrained=False, requires_grad=False).to(device)
cnn_alone.load_state_dict(torch.load(f"C:\\Users\\AntonioM\\Desktop\\efficientNetB2\\es_model.pth"))


# loss function
criterion = nn.CrossEntropyLoss(weight=weights.to(device))

isic_training_df = pandas.read_excel(f"C:\\Users\\AntonioM\\Desktop\\Training_Patients_ISIC_2020.xlsx", index_col=0)
#isic_validation_df = pandas.read_excel(f"C:\\Users\\AntonioM\\Desktop\\Validation_Patients_ISIC_2020.xlsx", index_col=0)
isic_validation_df = pandas.read_excel(f"C:\\Users\\AntonioM\\Desktop\\Test_Patients_ISIC_2020.xlsx", index_col=None)

patient_stats_training_df = pandas.read_excel(
    f"C:\\Users\\AntonioM\\Desktop\\Patients_stats\\Training_{network}_ISIC_2020.xlsx", index_col=0)
#patient_stats_validation_df = pandas.read_excel(
#    f"C:\\Users\\AntonioM\\Desktop\\Patients_stats\\Validation_{network}_ISIC_2020.xlsx", index_col=0)
patient_stats_validation_df = pandas.read_excel(
    f"C:\\Users\\AntonioM\\Desktop\\Patients_stats\\Test_{network}_ISIC_2020.xlsx", index_col=0)

#patient_stats_validation_df = pandas.read_excel(f"C:\\Users\\AntonioM\\Desktop\\Patients_stats\\Test_Autoencoder_Unet_Resnet18_ISIC_2020_for_rec_errors_and_EfficientNetB2_features.xlsx", index_col=None)
embeddings_mean_value = np.load(f"C:\\Users\\AntonioM\\PycharmProjects\\baseline_and_lr_version6\\input\\{network}_embeddings_mean.npy")


def get_patients_stats(partition, corresponding_image_names):
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
        #patient_stats = patient_stats.drop(columns=['patient_id', 'Unnamed: 0']).copy()  # for the test validation
        patients_stats_list.append(np.array(patient_stats))

    patients_stats_array = np.squeeze(np.array(patients_stats_list))
    # Normalize the embeddings because the loaded LogReg model was trained like this
    patients_stats_array = patients_stats_array - embeddings_mean_value
    patients_stats_tensor = torch.Tensor(patients_stats_array)
    patients_stats_tensor = patients_stats_tensor.to(device)

    return patients_stats_tensor, patients



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
                   'std_l2': np.std(patient_l2_dists), 'std_cos': np.std(patient_cosine_dists),}
                   #'n_melanomas': patient_stats_training_df[patient_stats_training_df['patient_id'] == patient][
                   #    'n_melanomas'].values[0] if partition == 'Training'
                   #else patient_stats_validation_df[patient_stats_validation_df['patient_id'] == patient][
                   #    'n_melanomas'].values[0]}
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



def validate(model, val_dataloader, val_dataset, update_embeddings=False):
    print('\nValidating CNN+LR')
    model.eval()
    all_val_baseline_preds = torch.tensor([]).to(device)
    all_val_baseline_probs = torch.tensor([]).to(device)
    all_val_combined_preds = torch.tensor([]).to(device)
    all_val_combined_probs = torch.tensor([]).to(device)
    all_val_lowerPath_preds = torch.tensor([]).to(device)
    all_val_lowerPath_probs = torch.tensor([]).to(device)
    all_val_targets = torch.tensor([]).to(device)
    all_val_features = torch.tensor([]).to(device)
    all_patients = np.array([])
    all_image_names = list()
    prog_bar = tqdm(enumerate(val_dataloader), total=int(len(val_dataset) / val_dataloader.batch_size))
    with torch.no_grad():
        for i, data in prog_bar:
            data, target, image_names = data[0].to(device), data[1].to(device), data[2]

            patient_stats_tensor, patients_id = get_patients_stats(partition='Validation', corresponding_image_names=image_names)

            baseline_outputs, logReg_outputs, combined_outputs = model(data, torch.squeeze(patient_stats_tensor))
            softmax = nn.Softmax(dim=1)

            all_val_features = torch.cat((all_val_features, features_dict['feats']))
            all_patients = np.concatenate((all_patients, patients_id), axis=None)

            # Upper Path
            probs_baseline = softmax(baseline_outputs.data)
            _, preds_baseline = torch.max(baseline_outputs.data, 1)

            # Lower Path
            probs_lowerPath = softmax(logReg_outputs.data)
            _, preds_lowerPath = torch.max(logReg_outputs.data, 1)

            # Combined (Baseline + Logistic Regression)
            probs_combined = softmax(combined_outputs.data)
            _, preds_combined = torch.max(combined_outputs.data, 1)


            # Save Predictions, Probabilities, Target Labels, and Images Names
            all_image_names.extend(image_names)
            all_val_baseline_preds = torch.cat((all_val_baseline_preds, preds_baseline))
            all_val_baseline_probs = torch.cat((all_val_baseline_probs, probs_baseline))
            all_val_combined_preds = torch.cat((all_val_combined_preds, preds_combined))
            all_val_combined_probs = torch.cat((all_val_combined_probs, probs_combined))
            all_val_lowerPath_preds = torch.cat((all_val_lowerPath_preds, preds_lowerPath))
            all_val_lowerPath_probs = torch.cat((all_val_lowerPath_probs, probs_lowerPath))
            all_val_targets = torch.cat((all_val_targets, target))

        if update_embeddings:
            update_patient_embeddings(all_val_features, all_patients, 'Validation')

        return all_image_names, all_val_baseline_preds, all_val_baseline_probs, \
               all_val_lowerPath_preds, all_val_lowerPath_probs, \
               all_val_combined_preds, all_val_combined_probs, all_val_targets


def validate_cnn_alone(model, val_dataloader, val_dataset):
    print('\nValidating CNN_alone')
    model.eval()
    all_val_preds = torch.tensor([]).to(device)
    all_val_probs = torch.tensor([]).to(device)
    prog_bar = tqdm(enumerate(val_dataloader), total=int(len(val_dataset) / val_dataloader.batch_size))
    with torch.no_grad():
        for i, data in prog_bar:
            data = data[0].to(device)
            outputs = model(data)
            softmax = nn.Softmax(dim=1)

            probs = softmax(outputs)
            _, preds = torch.max(outputs.data, 1)
            all_val_preds = torch.cat((all_val_preds, preds))
            all_val_probs = torch.cat((all_val_probs, probs))

        return all_val_preds, all_val_probs



def print_time_elapsed(start_time):
    end = time.time()
    hours, rem = divmod(end - start_time, 3600)
    minutes, seconds = divmod(rem, 60)

    if hours > 0:
        print(f"Training time: {hours}hrs {minutes}min {seconds:.0f}sec")
    else:
        print(f"Training time: {minutes}min {seconds:.0f}sec")


# A first pass just to update embeddings
_, _, _, _, _, _, _, _ = validate(model, val_dataloader, val_dataset, update_embeddings=True)

image_names, val_baseline_preds, val_baseline_probs, val_lowerPath_preds, val_lowerPath_probs, val_combined_preds, val_combined_probs, val_targets = validate(model, val_dataloader, val_dataset)
val_cnn_alone_preds, val_cnn_alone_probs = validate_cnn_alone(cnn_alone, val_dataloader, val_dataset)
patients = np.array(isic_validation_df['patient_id'])

df = pandas.DataFrame({'image_name': np.array(image_names),
                       'patient_id': patients,
                       'val_upperPath_preds': (val_baseline_preds.cpu()).numpy(),
                       'val_upperPath_probs_benign': (val_baseline_probs.cpu()).numpy()[:, 0],
                       'val_upperPath_probs_melanoma': (val_baseline_probs.cpu()).numpy()[:, 1],
                       'val_combined_preds': (val_combined_preds.cpu()).numpy(),
                       'val_combined_probs_benign': (val_combined_probs.cpu()).numpy()[:, 0],
                       'val_combined_probs_melanoma': (val_combined_probs.cpu()).numpy()[:, 1],
                       'val_cnn_preds': (val_cnn_alone_preds.cpu()).numpy(),
                       'val_cnn_probs_benign': (val_cnn_alone_probs.cpu()).numpy()[:, 0],
                       'val_cnn_probs_melanoma': (val_cnn_alone_probs.cpu()).numpy()[:, 1],
                       'label': (val_targets.cpu()).numpy()
                       })
df.index = np.arange(1, len(df) + 1)  # so index starts at one
#df.to_excel(f"C:\\Users\\AntonioM\\PycharmProjects\\baseline_and_lr_version6\\outputs\\evaluation_with_{network}.xlsx")
df[['image_name', 'val_combined_probs_melanoma']].to_csv(f"C:\\Users\\AntonioM\\Desktop\\INTEGRATED_NO_CAE_ISIC_SUBMISSION.csv", index=False, header=True)


def global_metrics_per_image(targets, preds):
    confusion = metrics.confusion_matrix(targets, preds)
    TBR = (confusion[0, 0] / confusion.sum(axis=1)[0])
    TMR = (confusion[1, 1] / confusion.sum(axis=1)[1])
    AUC = metrics.balanced_accuracy_score(targets, preds)

    return confusion, TBR, TMR, AUC


def global_metrics_per_patient(confusion):
    TBR = (confusion[0, 0] / confusion.sum(axis=1)[0])
    TMR = (confusion[1, 1] / confusion.sum(axis=1)[1])
    AUC = 0.5*(TBR + TMR)

    return TBR, TMR, AUC


# Compute TBR, TMR, AUC per image
upperPath_confusion, upperPath_TBR, upperPath_TMR, upperPath_AUC = global_metrics_per_image(val_targets.cpu(), val_baseline_preds.cpu())
combined_confusion, combined_TBR, combined_TMR, combined_AUC = global_metrics_per_image(val_targets.cpu(), val_combined_preds.cpu())
cnn_confusion, cnn_TBR, cnn_TMR, cnn_AUC = global_metrics_per_image(val_targets.cpu(), val_cnn_alone_preds.cpu())


# ----------- PATIENT EVALUATION --------------
"""
'val_lowerPath_preds': (val_lowerPath_preds.cpu()).numpy(),
'val_lowerPath_probs_benign': (val_lowerPath_probs.cpu()).numpy()[:, 0],
'val_lowerPath_probs_melanoma': (val_lowerPath_probs.cpu()).numpy()[:, 1],
"""

# Add info about lower path preds
df['val_lowerPath_preds'] = (val_lowerPath_preds.cpu()).numpy()

# Compute TBR, TMR, AUC per patient
patients_confusion_upperPath = np.zeros((2, 2))
patients_confusion_lowerPath = np.zeros((2, 2))
patients_confusion_combined = np.zeros((2, 2))
patients_confusion_cnn_alone = np.zeros((2, 2))
for patient in np.unique(patients):
    individual_df = df[df['patient_id'] == patient].copy()
    individual_df.reset_index(drop=True)
    n_melanomas = individual_df['label'].sum()
    n_melanomas_pred_upperPath = individual_df['val_upperPath_preds'].sum()
    n_melanomas_pred_lowerPath = individual_df['val_lowerPath_preds'].sum()
    n_melanomas_pred_combined = individual_df['val_combined_preds'].sum()
    n_melanomas_pred_cnn_alone = individual_df['val_cnn_preds'].sum()

    if n_melanomas == 0:
        if n_melanomas_pred_upperPath == 0:
            patients_confusion_upperPath[0, 0] += 1
        else:
            patients_confusion_upperPath[0, 1] += 1

        if n_melanomas_pred_combined == 0:
            patients_confusion_combined[0, 0] += 1
        else:
            patients_confusion_combined[0, 1] += 1

        if n_melanomas_pred_cnn_alone == 0:
            patients_confusion_cnn_alone[0, 0] += 1
        else:
            patients_confusion_cnn_alone[0, 1] += 1

        if n_melanomas_pred_lowerPath == 0:
            patients_confusion_lowerPath[0, 0] += 1
        else:
            patients_confusion_lowerPath[0, 1] += 1

    else:  # n_melanomas >= 1
        if n_melanomas_pred_upperPath == 0:
            patients_confusion_upperPath[1, 0] += 1
        else:
            patients_confusion_upperPath[1, 1] += 1

        if n_melanomas_pred_combined == 0:
            patients_confusion_combined[1, 0] += 1
        else:
            patients_confusion_combined[1, 1] += 1

        if n_melanomas_pred_cnn_alone == 0:
            patients_confusion_cnn_alone[1, 0] += 1
        else:
            patients_confusion_cnn_alone[1, 1] += 1

        if n_melanomas_pred_lowerPath == 0:
            patients_confusion_lowerPath[1, 0] += 1
        else:
            patients_confusion_lowerPath[1, 1] += 1


upperPath_TBR_patient, upperPath_TMR_patient, upperPath_AUC_patient = global_metrics_per_patient(patients_confusion_upperPath)
lowerPath_TBR_patient, lowerPath_TMR_patient, lowerPath_AUC_patient = global_metrics_per_patient(patients_confusion_lowerPath)
combined_TBR_patient, combined_TMR_patient, combined_AUC_patient = global_metrics_per_patient(patients_confusion_combined)
cnn_alone_TBR_patient, cnn_alone_TMR_patient, cnn_alone_AUC_patient = global_metrics_per_patient(patients_confusion_cnn_alone)

print('EVALUATION COMPLETE')
