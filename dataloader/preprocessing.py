import numpy as np
import scipy
import math
from pandas.api.types import is_numeric_dtype
from torchvision import transforms

def filterNumerical(df, keepCols = []):
    dropReasons = {}
    columnsToDrop = []
    
    for column_i in df.columns:
        if not is_numeric_dtype(df[column_i]) and column_i not in keepCols:
            columnsToDrop.append(column_i)
            dropReasons[column_i] = f"Dropped because type {df[column_i].dtype} is not numeric"

    df = df.drop(columns = columnsToDrop)
    return df, dropReasons

def filterFew(df, nan_thresh = 0.4, keepCols = []):
    dropReasons = {}
    columnsToDrop = []
    
    for column_i, count_nan in df.isna().sum().items():
        percent_nan = count_nan / df.shape[0]
        assert(0 <= percent_nan and percent_nan <= 1.0)
        
        if percent_nan >= nan_thresh and column_i not in keepCols:
            columnsToDrop.append(column_i)
            dropReasons[column_i] = f"Dropped because {percent_nan} nan is below threshold {nan_thresh}"

    df = df.drop(columns = columnsToDrop)
    return df, dropReasons

def filterCorrelation(df, result_i, keepCols, corr_thresh = 0.1):
    dropReasons = {}
    columnsToDrop = []

    pearson_mp = {}
    spearman_mp = {}
    
    for column_i in df.columns:
        if column_i in keepCols:
            continue
        mask = ~np.isnan(df[result_i]) & ~np.isnan(df[column_i])

        pearson_r = abs(scipy.stats.pearsonr(df[result_i][mask], df[column_i][mask]).statistic)
        spearman_r = abs(scipy.stats.spearmanr(df[result_i][mask], df[column_i][mask]).correlation)
        assert(-1 <= pearson_r and pearson_r <= 1 and -1 <= spearman_r and spearman_r <= 1 or math.isnan(pearson_r) and math.isnan(spearman_r))

        pearson_mp[column_i] = pearson_r
        spearman_mp[column_i] = spearman_r
        
        if pearson_r <= corr_thresh and spearman_r <= corr_thresh or math.isnan(pearson_r) and math.isnan(spearman_r): 
            
            columnsToDrop.append(column_i)
            dropReasons[column_i] = f"Pearson r {pearson_r} and spearman r {spearman_r} below threshold {corr_thresh}"

    df = df.drop(columns = columnsToDrop)
    return df, dropReasons


def removePostFeatures(df, keepCols):
    columnsToDrop = []
    dropReasons = {}

    for column_i in [
        'bHcg1', 
        'bHcg1Date', 
        'bHcg2', 
        'bHcg2Date', 
        'PregnancyUltrasoundForm_Date', 
        'embryo_N',
         'Pregnancy_EndPregnancyDate',
         'Pregnancy_PregnancyNo',
         'Pregnancy_CycleType',
         'Pregnancy_StartPregnancyDate',
         'Pregnancy_PregnancyInMedicalCenter',
         'Pregnancy_IsActive',
         'Pregnancy_LastUpdate',
         'PatientPregnancy_UpdatedUserId',
         'PatientPregnancyDetails_Child1_Sex',
         'PatientPregnancyDetails_Child2_Sex',
         'PatientPregnancyDetails_Child3_Sex',
         'PatientPregnancyDetails_Child1_Weight',
         'PatientPregnancyDetails_Child2_Weight',
         'PatientPregnancyDetails_Child3_Weight',
         'PatientPregnancyDetails_Child1_Stillbirth',
         'PatientPregnancyDetails_Child2_Stillbirth',
         'PatientPregnancyDetails_Child3_Stillbirth',
         'PatientPregnancyDetails_Child1_ApgarScore',
         'PatientPregnancyDetails_Child2_ApgarScore',
         'PatientPregnancyDetails_Child3_ApgarScore',
         'PatientPregnancyDetails_Child1_Comments',
         'PatientPregnancyDetails_Child2_Comments',
         'PatientPregnancyDetails_Child3_Comments',
        
        'children_N', 
        'bHcg_max', 
        'pulse_N', 
        'bag_N'
    ]:
        if column_i in df.columns and column_i not in keepCols:
            columnsToDrop.append(column_i)
            dropReasons[column_i] = "Feature observed after birth or ignored in IVF document"
    
    df = df.drop(columns = columnsToDrop)
    return df, dropReasons

# Define the custom transform function
def transform_frame(frame, angle, do_flip):
    # Apply rotation
    rotation_transform = transforms.RandomRotation([angle, angle])
    frame = rotation_transform(frame)

    # Apply horizontal flip
    if do_flip:
        flip_transform = transforms.RandomHorizontalFlip(p=1)
        frame = flip_transform(frame)

    return frame

def fix_sample_image_paths(image_file_path, fix_step):
    total_images = len(image_file_path)
    
    sampled_paths = [image_file_path[i] for i in range(0, total_images, fix_step)]

    return sampled_paths

def pad_np_array(np_array, max_frame):
    pad_shape = [max_frame] + list(np_array.shape[1:])
    padded = np.zeros(pad_shape).astype(np_array.dtype)
    padded[:np_array.shape[0]] = np_array
    return padded