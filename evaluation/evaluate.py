# -*- coding: utf-8 -*-
"""
PLOT - Evaluation (faithful to the paper's metric definitions)
==============================================================

Computes the two external metrics reported in the PLOT paper (Section IV-B)
from the pipeline's output files:

  * Topic Recall  (the primary semantic metric, paper Eq. 10):
        TopicRecall = TP / (TP + FN)
    A gold-standard event counts as detected (TP) when at least one of the
    system's representative keywords for that window overlaps with the
    gold-standard topic annotation. This is the binary keyword-overlap
    criterion defined in the paper - no tuning constants are applied.

  * Entropy  (class / cluster / total, structural diagnostic):
    The standard multi-label weighted entropy (log base 2), implemented in
    EvaluateFunctional.Entropy (paper Section IV-B-1).

The script runs in two steps:
  1. standardize_results(): join every post to the titles detected in its
     window (using the post text from AllData.npy) -> StandardizedResults/.
  2. evaluate_standardized_results(): compare against the gold standard and
     write Final_Evaluation_Report.xlsx.

INPUT  (relative to the PLOT folder, resolved automatically):
    ../SystemResults/Topic_Systemresult_*.xls   (event titles per window)
    ../SystemResults/ResultsToCompaire_*.xls     (event number per post)
    ../AllData.npy                               (post text)
    GoldenStandard/GoldenStandard_TopicID_and_TopicString.xlsx
OUTPUT:
    Final_Evaluation_Report.xlsx   (Topic/Keyword P/R/F1 + Class/Cluster/Total Entropy)
    StandardizedResults/standardized_*.xlsx

USAGE:
    python evaluate.py
"""

import os
import pandas as pd
import numpy as np
import glob
from openpyxl import load_workbook
import re
import EvaluateFunctional as Eval
from math import log

def normalize_text(text):
    """A simple text normalizer for Persian."""
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = text.replace('ي', 'ی')
    text = text.replace('ك', 'ک')
    return text

def append(a,b):
    c = np.empty(len(a)+1, dtype=object)
    for i in range(len(a)):
        c[i] = a[i]
    c[-1] = np.array(b)
    return c

def MergeTitleStrings_new(SR_Label,SR_Title):

    #Detect The Maximum LabelNumber (Labels are started from 0 and end by MaxLabelNum)
    MaxLabelNum = -1
    for L in SR_Label:
        maxlabelnum = np.max(L)
        if maxlabelnum>MaxLabelNum:
                MaxLabelNum = maxlabelnum

    #Detect All Title Strings for each Label
    TitleOfLabels = np.array([])
    for L in range(int(MaxLabelNum)+1):
        temp = set()
        for i,l in enumerate(SR_Label):
            for ii,ll in enumerate(l):
                if ll == L:
                    # In the new format, SR_Title[i] is an array of titles for the sequence.
                    # We assume all titles in that array are associated with the label.
                    for title_word in np.atleast_1d(SR_Title[i]):
                        temp.add(str(title_word))

        TitleOfLabels = append(TitleOfLabels,list(temp))

    # Build the merged output.
    SR_Title_Merged = np.array([])
    for _ in SR_Title:
        SR_Title_Merged = append(SR_Title_Merged,np.array([]))
    for i,L in enumerate(SR_Label):
        for labelnum_ in L:
            labelnum = int(labelnum_)
            if labelnum == -1:
                LabelMergedString = '-1'
            else:
                LabelMergedString = ','.join(TitleOfLabels[labelnum])

            SR_Title_Merged[i] = append(SR_Title_Merged[i],LabelMergedString)


    return SR_Title_Merged,TitleOfLabels


def TitleOfEachLabel(GS_Number,GS_String):
    #Detect The Maximum LabelNumber in GS(Labels are started from 0(0 means out of class) and end by MaxLabelNum)
    MaxLabelNum = -1
    for L in GS_Number:
        l = map(int, str(L).split(','))
        maxlabelnum = max(l)
        if maxlabelnum>MaxLabelNum:
                MaxLabelNum = maxlabelnum

    #Detect All Title Strings for each Label
    TitleOfLabels = np.array([])
    for L in range(MaxLabelNum+1):
        temp = set()
        for i,l in enumerate(GS_Number):
            ll =  map(int, str(l).split(','))
            for iii,lll in enumerate(ll):
                if lll == L:
                    CurrentSTR = (GS_String[i].split(','))[iii]
                    temp.add(CurrentSTR)
        if list(temp) == []:
            temp.add('WRONGDETECT')
        TitleOfLabels = append(TitleOfLabels,list(temp))

    return TitleOfLabels

def PrepareData_new(GoldenStandard,SystemResult):
    #SystemResult[0] : Sequence
    #SystemResult[1] : Label  Clustering
    #SystemResult[2] : Title  Clustering
    #SystemResult[3] : Label  Community
    #SystemResult[4] : Totle  Community

    GS = np.array([])
    GS = append(GS,GoldenStandard['Sequence']._values.tolist())
    GS = append(GS,GoldenStandard['Topics(Id)']._values.tolist())
    GS = append(GS,GoldenStandard['Topics(Str)']._values.tolist())
    GS = append(GS,np.array([]))
    #GS[0]: Samples
    #GS[1]: Classes_Number
    #GS[2]: Classes_String
    #GS[3]: TitleOfEachLabel_GS


    SR = np.array([])
    SR = append(SR,np.array([])) # create index 0 for Clusters_Com
    SR = append(SR,np.array([])) # create index 1 for Clusters_Clu
    SR = append(SR,np.array([])) # create index 2 for Clusters_Com_Str
    SR = append(SR,np.array([])) # create index 3 for Clusters_Clu_Str
    SR = append(SR,np.array([])) # create index 4 for TitleOfEachLabel_Com
    SR = append(SR,np.array([])) # create index 5 for TitleOfEachLabel_Clu
    #SR[0]: Clusters_Com
    #SR[1]: Clusters_Clu
    #SR[2]: Clusters_Com_Str
    #SR[3]: Clusters_Clu_Str
    #SR[4]: TitleOfEachLabel_Com
    #SR[5]: TitleOfEachLabel_Clu


    # Add To SR, Only the Sequences that exist in GS
    SeqOrderInSR = np.array([]) # sanity check that GS and SR share the same sequence order
    ExtraSEQinGS = np.array([])
    for seq in GS[0]:
        if seq in SystemResult[0]:
            idx = np.where(SystemResult[0]==seq)[0][0]

            SR[0] =  append(SR[0],  SystemResult[3][idx]) # Cluster_Com
            SR[2] =  append(SR[2],  SystemResult[4][idx]) # Clusters_Com_Str

            SR[1] =  append(SR[1],  SystemResult[1][idx]) # Clusters_Clu
            SR[3] =  append(SR[3],  SystemResult[2][idx]) # Clusters_Clu_Str


            SeqOrderInSR = np.append(SeqOrderInSR,SystemResult[0][idx])
        else:
            ExtraSEQinGS = append(ExtraSEQinGS,seq)

    #Delete Extra Sequence exist in GS that Not Exist In SR
    for seq in reversed(ExtraSEQinGS):
        idx = np.where(GS[0]==seq)[0][0]
        GS[2] = np.delete(GS[2],idx)
        GS[1] = np.delete(GS[1],idx)
        GS[0] = np.delete(GS[0],idx)

    # The sequences present in GS and SR must now match one-to-one.
    if len(GS[0]) != len(SR[2]):
        print('ERROR: GS and SR have a different number of sequences')
    for i in range(len(GS[0])):
        if GS[0][i] != SeqOrderInSR[i]:
            print('ERROR: GS and SR sequence order mismatch')

    # Label the empty entries in SR with -1 (unassigned).
    for i in range(len(SR[0])):
        for j in range(4): # the first 4 SR indices: mark label-less entries as -1
            if SR[j][i].size == 0:
                SR[j][i] = np.append(SR[j][i],-1)

    SR[2],SR[4] = MergeTitleStrings_new(SR[0],SR[2])
    SR[3],SR[5] = MergeTitleStrings_new(SR[1],SR[3])

    GS[3] = TitleOfEachLabel(GS[1],GS[2])

    # Remove out-of-class posts (gold label '0') from both GS and SR.
    L = len(GS[1])
    for i,label in enumerate(reversed(GS[1])):
        i_R = abs(i-L+1)
        if label == '0':
            GS[2] = np.delete(GS[2],i_R)
            GS[1] = np.delete(GS[1],i_R)
            GS[0] = np.delete(GS[0],i_R)

            SR[0] = np.delete(SR[0],i_R)
            SR[1] = np.delete(SR[1],i_R)
            SR[2] = np.delete(SR[2],i_R)
            SR[3] = np.delete(SR[3],i_R)

    return GS,SR


def standardize_results(raw_results_path, all_data_path, standardized_results_path):
    """
    Reads raw result files and standardizes them based on the correct logic:
    1. Create a map from Window Number -> Set of Titles.
    2. For each sequence, find its window's titles and search them in the sequence's full text.
    """
    if not os.path.exists(standardized_results_path):
        os.makedirs(standardized_results_path)

    # 1. Load AllData.npy to get sequence full texts
    loaded_data = np.load(all_data_path, allow_pickle=True)
    sequence_to_text = {}
    sequence_ids_windows = loaded_data[0]
    sequence_texts_windows = loaded_data[1]
    for i in range(len(sequence_ids_windows)):
        for j in range(len(sequence_ids_windows[i])):
            seq_id = sequence_ids_windows[i][j]
            sequence_to_text[seq_id] = " ".join(sequence_texts_windows[i][j])

    # --- Process each parameter set ---
    label_files = glob.glob(os.path.join(raw_results_path, "ResultsToCompaire_*.xls"))
    standardized_files = {}

    for label_file in label_files:
        param_string = os.path.basename(label_file).split("ResultsToCompaire_")[1].replace(".xls", "")
        topic_file_path = os.path.join(raw_results_path, f"Topic_Systemresult_{param_string}.xls")

        if not os.path.exists(topic_file_path):
            print(f"Warning: Topic file not found for {param_string}")
            continue

        print(f"Processing parameters: {param_string}")

        # 2. Build the Window -> Set of Titles map
        topic_xls = pd.ExcelFile(topic_file_path)
        window_to_titles_map = {}
        df_topics = topic_xls.parse(topic_xls.sheet_names[0], header=0)
        win_col = df_topics.columns[0]
        title_col = df_topics.columns[1]
        for window_num, group in df_topics.groupby(win_col):
            all_titles = set()
            for titles_str in group[title_col].dropna():
                all_titles.update(t.strip() for t in str(titles_str).split('|'))
            window_to_titles_map[window_num] = all_titles

        # 3. Read Label File and assign titles
        label_xls = pd.ExcelFile(label_file)
        all_sheets_data = []
        for sheet_name in label_xls.sheet_names:
            df = label_xls.parse(sheet_name)
            df['window'] = int(sheet_name.split('-')[-1])
            all_sheets_data.append(df)

        labels_df = pd.concat(all_sheets_data, ignore_index=True)
        event_col_name = [col for col in labels_df.columns if 'EventNumber' in col][0]
        labels_df.rename(columns={
            labels_df.columns[0]: 'sequence',
            event_col_name: 'EventNumber'
        }, inplace=True)

        # 3. Clean up and normalize the EventNumber column
        def clean_event_number(event):
            try:
                # Take the first number if it's a comma-separated string
                s_event = str(event).split(',')[0]
                return int(s_event)
            except (ValueError, TypeError):
                return -1 # Return a value that won't match

        labels_df['EventNumber'] = labels_df['EventNumber'].apply(clean_event_number)

        # 3. Assign titles using the per-window map
        def find_and_assign_titles(row):
            seq_id = row['sequence']
            window_num = row['window']

            seq_text = sequence_to_text.get(seq_id, "")
            candidate_titles = window_to_titles_map.get(window_num, set())

            found_titles = [title for title in candidate_titles if title in seq_text]
            return '|'.join(found_titles)

        labels_df['title'] = labels_df.apply(find_and_assign_titles, axis=1)

        # 4. Save standardized file
        output_path = os.path.join(standardized_results_path, f"standardized_{param_string}.xlsx")
        final_df = labels_df[['sequence', 'EventNumber', 'window', 'title']]
        final_df.to_excel(output_path, index=False)
        standardized_files[param_string] = output_path

    return standardized_files

def evaluate_standardized_results(standardized_files, golden_standard_path):
    """
    Evaluates the standardized results against the golden standard.

    Args:
        standardized_files (dict): Dict of param_string -> path to standardized file.
        golden_standard_path (str): Path to the golden standard Excel file.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation metrics for each parameter set.
    """
    golden_standard = pd.read_excel(golden_standard_path)

    all_evaluation_results = []

    for param_string, file_path in standardized_files.items():
        print(f"Evaluating parameters: {param_string}")

        system_results_df = pd.read_excel(file_path)

        # Prepare data in the format expected by the original evaluation functions
        # SystemResult[0] : Sequence
        # SystemResult[1] : Label  Clustering
        # SystemResult[2] : Title  Clustering
        # SystemResult[3] : Label  Community (not available, use clustering)
        # SystemResult[4] : Title  Community (not available, use clustering)

        # The original script expects a numpy array of objects
        system_result_for_eval = np.array([
            system_results_df['sequence'].values,
            system_results_df['EventNumber'].apply(lambda x: np.array([int(i) for i in str(x).split(',')])).values, # Clustering Label
            system_results_df['title'].apply(lambda x: np.array(str(x).split('|')) if pd.notna(x) and x else np.array([])).values, # Clustering Title
            system_results_df['EventNumber'].apply(lambda x: np.array([int(i) for i in str(x).split(',')])).values, # Community Label
            system_results_df['title'].apply(lambda x: np.array(str(x).split('|')) if pd.notna(x) and x else np.array([])).values  # Community Title
        ], dtype=object)

        # Call the new evaluation function
        eval_res, _ = EvalAndSaveRes_new(golden_standard.copy(), system_result_for_eval)

        # Parse the parameter stamp of the filename into a dictionary.
        params = {}
        for item in param_string.split('_'):
            if '-' in item:
                key, value = item.split('-')
                params[key] = value

        # Store results
        results = {
            **params,
            'Topic Precision': eval_res[12],
            'Topic Recall': eval_res[13],
            'Topic F1': eval_res[14],
            'Keyword Precision': eval_res[15],
            'Keyword Recall': eval_res[16],
            'Keyword F1': eval_res[17],
            'Class Entropy': eval_res[3],
            'Cluster Entropy': eval_res[2],
            'Total Entropy': eval_res[5]
        }
        all_evaluation_results.append(results)

    return pd.DataFrame(all_evaluation_results)


def Matched_To_GS(GS,event_str):
    words = event_str.split(' ')
    for gs_string in GS:
        for w in words:
            if w in gs_string:
                return True
    return False

def Matched_To_GS_WordCount(GS,event_str):
    WM=0 # Word Matched
    words = event_str.split(' ')
    WT=len(words) # Word Total

    gs_text = ' '.join([s[0] for s in GS])

    for w in words:
        if w in gs_text:
            WM+=1

    return WM,WT

def IsCorrectDetect(SR,topic_str):
    words = topic_str[0].split(' ')
    for event_i,event_list in enumerate(SR):
        for event_str in event_list:
            for w in words:
                if w in event_str:
                    return True
    return False

def IsCorrectDetect_WordCount(SR,topic_str):
    WM=0 # Word Matched
    words = topic_str[0].split(' ')
    WT=len(words) # Word Total

    sr_text = ' '.join([' '.join(event_list) for event_list in SR])

    for w in words:
        if w in sr_text:
            WM+=1

    return WM,WT


def TopicEvaluation_new(GS,SR):
    #1:Topic Evaluation
    SRM = 0
    for event_i,event_list in enumerate(SR):
        for event_str in event_list:
            if Matched_To_GS(GS,event_str):
                SRM+=1
                break
    SRC = len(SR)
    TopicPrecision = SRM/SRC if SRC > 0 else 0

    GSM = 0
    for topic_str in GS:
        if IsCorrectDetect(SR,topic_str):
            GSM+=1
    GSC = len(GS)
    TopicRecall = GSM/GSC if GSC > 0 else 0

    TopicF1 = 2*(TopicPrecision*TopicRecall)/(TopicPrecision+TopicRecall) if (TopicPrecision+TopicRecall) > 0 else 0

    #2: Keyword Evaluation
    SRMW = 0
    SRMWT = 0
    for event_i,event_list in enumerate(SR):
        event_str = ' '.join(event_list)
        WM,WT = Matched_To_GS_WordCount(GS,event_str)
        SRMW+=WM
        SRMWT+=WT

    KeywordPrecision = SRMW/SRMWT if SRMWT > 0 else 0

    GSMW = 0
    GSMWT = 0
    for topic_str in GS:
        WM,WT = IsCorrectDetect_WordCount(SR,topic_str)
        GSMW+=WM
        GSMWT+=WT

    KeywordRecall = GSMW/GSMWT if GSMWT > 0 else 0

    KeywordF1 = 2*(KeywordPrecision*KeywordRecall)/(KeywordPrecision+KeywordRecall) if (KeywordPrecision+KeywordRecall) > 0 else 0

    return TopicPrecision,TopicRecall,TopicF1, KeywordPrecision,KeywordRecall,KeywordF1

def EvalAndSaveRes_new(GoldenStandard,SystemResult):

    ##Samples, Classes_Number, Classes_String, Clusters_Com, Clusters_Clu, Clusters_Com_Str, Clusters_Clu_Str, TitleOfEachLabel_Com, TitleOfEachLabel_Clu, TitleOfEachLabel_GS= PrepareData(GS_Class_String, seq_label_community, seq_label_cluster, seq_title_community, seq_title_cluster)
    GS,SR = PrepareData_new(GoldenStandard,SystemResult)
    #GS[0]: Samples
    #GS[1]: Classes_Number
    #GS[2]: Classes_String
    #GS[3]: TitleOfEachLabel_GS

    #SR[0]: Clusters_Com
    #SR[1]: Clusters_Clu
    #SR[2]: Clusters_Com_Str
    #SR[3]: Clusters_Clu_Str
    #SR[4]: TitleOfEachLabel_Com
    #SR[5]: TitleOfEachLabel_Clu

    # In our new version, we override the title merging part
    SR[2],SR[4] = MergeTitleStrings_new(SR[0],SR[2])
    SR[3],SR[5] = MergeTitleStrings_new(SR[1],SR[3])

    ## Results of Entropy Measure:
    ClusterEntropy_Community = Eval.Entropy(GS[0].copy(), SR[0].copy(), GS[1].copy())
    ClassEntropy_Community = Eval.Entropy(GS[0].copy(), GS[1].copy(), SR[0].copy())
    ClusterEntropy_Clustering = Eval.Entropy(GS[0].copy(), SR[1].copy(), GS[1].copy())
    ClassEntropy_Clustering = Eval.Entropy(GS[0].copy(), GS[1].copy(), SR[1].copy())
    w1=1
    w2=1
    TotalEntropy_Community = ((w1*ClusterEntropy_Community)+(w2*ClassEntropy_Community))/(w1+w2)
    TotalEntropy_Clustering= ((w1*ClusterEntropy_Clustering)+(w2*ClassEntropy_Clustering))/(w1+w2)

    #Topic Evaluation:
    TopicPrecision_Com,TopicRecall_Com,TopicF1_Com, KeywordPrecision_Com,KeywordRecall_Com,KeywordF1_Com = TopicEvaluation_new(GS[3].copy(),SR[4].copy())
    TopicPrecision_Clu,TopicRecall_Clu,TopicF1_Clu, KeywordPrecision_Clu,KeywordRecall_Clu,KeywordF1_Clu  = TopicEvaluation_new(GS[3].copy(),SR[5].copy())

    #Append All Result Data to One Variable
    EvalRes = np.array([])
    EvalRes = np.append(EvalRes, ClusterEntropy_Community)
    EvalRes = np.append(EvalRes, ClassEntropy_Community)
    EvalRes = np.append(EvalRes, ClusterEntropy_Clustering)
    EvalRes = np.append(EvalRes, ClassEntropy_Clustering)
    EvalRes = np.append(EvalRes, TotalEntropy_Community)
    EvalRes = np.append(EvalRes, TotalEntropy_Clustering)
    EvalRes = np.append(EvalRes, TopicPrecision_Com)
    EvalRes = np.append(EvalRes, TopicRecall_Com)
    EvalRes = np.append(EvalRes, TopicF1_Com)
    EvalRes = np.append(EvalRes, KeywordPrecision_Com)
    EvalRes = np.append(EvalRes, KeywordRecall_Com)
    EvalRes = np.append(EvalRes, KeywordF1_Com)
    EvalRes = np.append(EvalRes, TopicPrecision_Clu)
    EvalRes = np.append(EvalRes, TopicRecall_Clu)
    EvalRes = np.append(EvalRes, TopicF1_Clu)
    EvalRes = np.append(EvalRes, KeywordPrecision_Clu)
    EvalRes = np.append(EvalRes, KeywordRecall_Clu)
    EvalRes = np.append(EvalRes, KeywordF1_Clu)


    return EvalRes, None # We don't need the data part


def main():
    """
    Main function to run the entire evaluation pipeline.
    """
    # Paths are resolved relative to this script so the PLOT folder is portable.
    # evaluate.py lives in   PLOT/evaluation/  ->  project_root is the PLOT folder.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    raw_results_path = os.path.join(project_root, "SystemResults")                 # pipeline output (.xls)
    all_data_path = os.path.join(project_root, "AllData.npy")                       # post text
    golden_standard_path = os.path.join(script_dir, "GoldenStandard", "GoldenStandard_TopicID_and_TopicString.xlsx")
    output_report_path = os.path.join(script_dir, "Final_Evaluation_Report.xlsx")
    standardized_results_path = os.path.join(script_dir, "StandardizedResults")

    print("Step 1: Standardizing raw results...")
    standardized_files = standardize_results(raw_results_path, all_data_path, standardized_results_path)
    print("Standardization complete.")

    print("Step 2: Evaluating standardized results...")
    evaluation_results = evaluate_standardized_results(standardized_files, golden_standard_path)
    print("Evaluation complete.")

    print(f"Step 3: Saving final report to {output_report_path}...")
    evaluation_results.to_excel(output_report_path, index=False)
    print("Report saved.")

if __name__ == "__main__":
    main()