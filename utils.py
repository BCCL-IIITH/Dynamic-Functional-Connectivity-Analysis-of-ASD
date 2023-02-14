import os
import sys
import numpy as np
import glob
import pandas as pd
import shutil
import gzip
import nibabel
import scipy.io
import h5py
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import math

from mne_connectivity.viz import plot_connectivity_circle
from mne.viz import circular_layout

def plot_connectogram(arr, nodes, n_lines, title, format = 'png'):
    
    nodes = list(str(node) for node in nodes)
    node_angles = circular_layout(nodes, nodes)
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black', subplot_kw=dict(projection='polar'))
    plot_connectivity_circle(arr, nodes, n_lines=n_lines,
                         node_angles=node_angles, node_colors=None,
                         title=title, ax=ax, vmin = -0.3, vmax = 0.8)
    fig.savefig(f'./Phenotypes/Stronger_CC_DM_New/{title}.{format}', facecolor='black', format = format)
    plt.close()

def plot_CC_DM_connections(state, centroid, title, format = 'png'):
    label_inds = np.array(range(1,54))
    centroid.index = label_inds
    centroid.columns = label_inds
    networks = ['CC-CC', 'CC-DM', 'DM-DM']
    state = state[state['Networks'].isin(networks)]
    n = len(state)
    print('Number of significant connection within and between CC, DM networks : ', n)
    centroid_mat = np.zeros((53,53))
    for i in range(n):
        idx1 = state.iloc[i]['First_ROI_Index'].item()
        idx2 = state.iloc[i]['Second_ROI_Index'].item()
        assert idx1>=26 and idx1<=49, "Error!"
        assert idx2>=26 and idx2<=49, "Error!"
        centroid_mat[idx1-1, idx2-1] = centroid.loc[idx1][idx2].item()
        centroid_mat[idx2-1, idx1-1] = centroid.loc[idx2][idx1].item()

    print('Number of non zero values in the above matrix :', (centroid_mat!=0).sum())
    centroid_mat_cc_dm = centroid_mat[25:49, 25:49]
    print('Shape of centroid of CC-DM networks only', centroid_mat_cc_dm.shape)
    print('Number of non zero values in the Centroid CC-DM matrix :', (centroid_mat_cc_dm!=0).sum())
    nodes = ['IC' + str(node) for node in range(26,50)]
    n_lines = (centroid_mat_cc_dm!=0).sum() // 2
    plot_connectogram(centroid_mat_cc_dm, nodes, n_lines, title, format = format)
    return

def get_key(path) :

    '''
        Extracts the SUB_ID from the path of the subject
    '''
    key = path.split('/')[-1]
    key = key.split('_func')[0]
    key = key.split('_00')[-1]
    key = int(key)
    return key

def plot_two_axes(y1, y2, x_title, y1_title, y2_title):
    plt.figure(figsize=(20, 5))
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(range(len(y1)), y1, color = 'red', marker = 'o')
    ax.set_ylabel(y1_title, color = 'red')
    ax2 = ax.twinx()
    ax2.plot(range(len(y2)), y2, color = 'blue', marker = 'o')
    ax2.set_xlabel(x_title)
    ax2.set_ylabel(y2_title, color = 'blue')
    plt.show()
    plt.close()

def print_details(arr) :

    '''
        Prints details like shape, min, median, max, mean and std of an array
    '''
    print('Shape : ', arr.shape)
    print(f'Min Value : {np.min(arr)}, Median Value : {np.median(arr)} and  Max Value : {np.max(arr)}')
    print(f'Mean Value : {np.mean(arr)} and SD Value : {np.std(arr)}')
    return

def tril_mat2vec(mat, vec, n):
    
    idx = 0
    for j in range(n):
        for i in range(j + 1, n):
            vec[idx] = mat[i,j]
            idx += 1
    return vec
    
def mat2vec(mat) :

    m = mat.shape[1]
    pairs = (m * (m-1)) // 2
    vec = np.ones(pairs)
    return tril_mat2vec(mat, vec, m)

def tril(temp, vec, n):
    
    idx = 0
    for j in range(n):
        for i in range(j + 1, n):
            temp[i,j] = vec[idx]
            idx += 1
    return temp

def vec2mat(vec):
    N = len(vec)
    n = 0.5 + math.sqrt(1+8*N)/2
    n = int(n)
    mat = np.ones((n,n))
    mat = tril(mat, vec, n)
    tempmat = np.flipud(np.rot90(mat))
    tempmat = tril(tempmat, vec, n)
    assert np.array_equal(tempmat, tempmat.T), "Error!"
    return tempmat

def plot_matrix(mat, threshold):
    mat_threshold = np.where(mat < threshold, mat, 1)
    plt.figure(figsize=(20,20))
    sns.heatmap(mat_threshold, annot=False, cmap = 'bwr')
    plt.show()
    plt.close()
    return mat_threshold

def get_pvalues(arr, groups, n_states):
    """
    Returns p-values comparing different groups across n_states 

    Parameters : 
    arr : <array>, shape = (n_subjects, n_states)
    groups : <dict>, format = {group_name : group_indices}
    n_states : <int>, 
        Number of states

    Returns : 
    pvalues : <dict>, format = {state : p-value}
    """
    pvalues = {}
    main_dict = {}
    group_names = list(groups.keys())
    print(group_names)
    n_groups = len(group_names)
    states = np.array(range(n_states))

    for state in states:
        main_dict[state] = {}
        for group in group_names:     
            main_dict[state][group] = arr[groups[group], state]
        
        if(n_groups == 2):
            group1 = np.array(list(main_dict[state][group_names[0]]))
            group2 = np.array(list(main_dict[state][group_names[1]]))
            _, pvalue = stats.ttest_ind(group1, group2)

        if(n_groups == 3):
            group1 = np.array(list(main_dict[state][group_names[0]]))
            group2 = np.array(list(main_dict[state][group_names[1]]))
            group3 = np.array(list(main_dict[state][group_names[2]]))
            _, pvalue = stats.f_oneway(group1, group2, group3)
        
        if(n_groups == 4):
            group1 = np.array(list(main_dict[state][group_names[0]]))
            group2 = np.array(list(main_dict[state][group_names[1]]))
            group3 = np.array(list(main_dict[state][group_names[2]]))
            group4 = np.array(list(main_dict[state][group_names[3]]))
            _, pvalue = stats.f_oneway(group1, group2, group3, group4)

        if(n_groups == 5):
            group1 = np.array(list(main_dict[state][group_names[0]]))
            group2 = np.array(list(main_dict[state][group_names[1]]))
            group3 = np.array(list(main_dict[state][group_names[2]]))
            group4 = np.array(list(main_dict[state][group_names[3]]))
            group5 = np.array(list(main_dict[state][group_names[4]]))
            _, pvalue = stats.f_oneway(group1, group2, group3, group4, group5)

        pvalues[state] = pvalue
    return pvalues

def get_mean_std(arr, groups, n_states):

    """
    Returns Mean and Standard Deviation different groups across n_states 

    Parameters : 
    arr : <array>, shape = (n_subjects, n_states)
    groups : <dict>, format = {group_name : group_indices}
    n_states : <int>, 
        Number of states

    Returns : 
    mean_groups : <dict>, format = {group_name : means of the group across n_states}
                key = <str> , value = <arr>, Shape : (n_states,)
    std_groups : <dict>, format = {group_name : Standard Deviation of the group across n_states}
                key = <str> , value = <arr>, Shape : (n_states,)
                
    """

    states = np.array(range(n_states))
    group_names = list(groups.keys())   
    print('Groups : ', group_names)
    mean_groups, std_groups = {}, {}

    for group in group_names: 
        means, stds = [], []
        for state in states :
            temp = arr[groups[group], state]
            # temp = temp[temp != 0]
            means.append(np.mean(temp))
            stds.append(np.std(temp))

        mean_groups[group] = means
        std_groups[group] = stds

    return mean_groups, std_groups
    
def plot_groups(mean_groups, std_groups, pvalues_dict, title = None):

    """
    Bar plot of Means with error bar representing STD for different groups across n_states 

    Parameters 
    -----------------------------
    mean_groups : <dict>, format = {group_name : means of the group across n_states}
                key = <str> , value = <arr>, Shape : (n_states,)

    std_groups : <dict>, format = {group_name : Standard Deviation of the group across n_states}
                key = <str> , value = <arr>, Shape : (n_states,)
    pvalues : <dict>, format = {state : p-value}

    title : <str>, 
            Title of the plot

    Returns : 
    -----------------------
    None
    """

    states = list(pvalues_dict.keys())
    n_states = len(states)
    group_names = list(mean_groups.keys()) 
    states = np.array(range(n_states))
    colors = ['red', 'blue', 'green', 'orange', 'yellow']
    # positions = [-0.5, -0.25, 0, 0.25, 0.5]
    positions = [-0.25, 0]
    fig, ax = plt.subplots(1, 1)
    for idx, group in enumerate(group_names):
        temp = [0] * len(std_groups[group])
        error = [temp, std_groups[group]]
        ax.bar(states + positions[idx], mean_groups[group],
            yerr = error,color=colors[idx], label=group, width = 0.25)
        

    plt.xticks([r - 0.15 for r in states], states + 1)
  
    ytick_labels = ax.get_yticks()
    print(ytick_labels)
    new_ytick_labels = []
    for i in range(len(ytick_labels)):
        t = ytick_labels[i]
        if t != 0.:
            t = np.round(t, 2) #+ '%'
        new_ytick_labels.append(t)

    positions = np.linspace(0.1,0.9,len(states))
    y_text_asd = np.array(mean_groups['ASD'], dtype = 'float32') + np.array(std_groups['ASD'], dtype = 'float32')
    y_text_td = np.array(mean_groups['TD'], dtype = 'float32') + np.array(std_groups['TD'], dtype = 'float32')
    y_text_location = np.max(np.vstack([y_text_asd, y_text_td]), axis = 0)
    if(np.max(y_text_location) > 1):    
        y_text_location = y_text_location / 100
    print(y_text_location)
    
    for idx, i in enumerate(states) :
        ax.text(positions[idx], y_text_location[idx] + 0.1,
                'p=' + str(np.round(pvalues_dict[i], decimals=3)),
                transform=ax.transAxes)
        if(pvalues_dict[i] < 0.05):
            ax.text(positions[idx], y_text_location[idx] + 0.05,'*',
                    transform=ax.transAxes)

    ax.set_yticklabels(new_ytick_labels)
    plt.title(f'{title} between groups across states')
    plt.ylabel('Mean Values')
    plt.legend(loc = 'upper left')
    plt.show()
    plt.close()


def hist_plot(arr, labels) :
    plt.figure(figsize=(10,6))
    n = arr.shape[0]
    k = math.ceil(math.sqrt(n))
    colors = ['red', 'blue', 'green', 'orange', 'yellow']
    for i in range(n):
        plt.subplot(k,k,i+1)
        sns.histplot(x = arr[i], color = colors[i], bins = 20, cbar='bright')
        plt.title(labels[i])
    plt.show()


# def kde_plot(arr, labels) :
#     colors = ['red', 'blue', 'green', 'orange', 'yellow']
#     for i in range(arr.shape[0]):
#         sns.kdeplot(x = arr[i], color = colors[i])
#     plt.legend(labels)
#     plt.show()

def get_roi_name(roi_index):
    """
    Returns ROI name from the template, given its index
    """
    nm_labels = pd.read_excel('./Phenotypes/NM_Template_Labels.xlsx', index_col = 'Indices')
    return nm_labels.loc[roi_index, 'ROI']

def get_roi_indices(conn_index):

    """
    Returns a Dictionary containing the ADOS_correlation value with all the significant connectivities and its correponding p-values
    
    Parameters 
    -----------------------------
        conn_index : Python index of significant distinct connecitivity in upper triangular part of correlation matrix 
        
    
    Returns : 
    -----------------------
        first_idx, second_idx : <tuple>, 
                    Matlab Index of region 1 and region 2
    
    """

    reference = np.array([i for i in range(1378)], dtype = 'int')
    reference_mat = np.array(vec2mat(reference), dtype='int')
    np.fill_diagonal(reference_mat, -1)

    twod_indices = np.array(np.where(reference_mat == conn_index), dtype = 'uint32') + 1  # To check with NM template, we need matlab indices
    first_idx = twod_indices[0,0]
    second_idx = twod_indices[0,1]
    return first_idx, second_idx

def get_ados_corr(all_states_ados, p_values, state, feature = 'ADOS_TOTAL') :

    """
    Returns a Dictionary containing the ADOS_correlation value with all the significant connectivities and its correponding p-values
    
    Parameters 
    -----------------------------
        all_states_ados : <dict<dict>>, format = {state : {key : value}}
                state = <int> , State number
                key = <str>, {'File_Index', 'File_Id', 'Correlation, 'ADOS_TOTAL'}
                
                File_Id = <arr>, Shape : (n_subjects per group,)
                File_Index = <arr>, Shape : (n_subjects per group,)
                Correlation value = <arr>, Shape : (n_subjects per group, 1378)
                    Median of windows per subject per group per state 
                ADOS_TOTAL = <arr>, Shape : (n_subjects per group,)

        p_values : <arr>, Shape = (number of distint connectivities, ) = (1378, )
                P-values of a given state generated by matlab 
        
        state : <int>, 
                State number (python index)
        
    
    Returns : 
    -----------------------
        state_ados_corr : <arr>, Shape = (number of distint connectivities, ) = (1378, )
                Correlation of significant connectivities with ADOS-TOTAL scores of ASD subjects per state

        p : <arr>, Shape = (number of distint connectivities, ) = (1378, )
                P-values (< 0.05) only significant ones
            
        indices : <arr>, 
                Indices of P-values (< 0.05) only significant ones
    
    """

    significant_inds = np.where(p_values < 0.05)[0]
    state_corr = all_states_ados[state]['Correlation']
    state_ados = all_states_ados[state][feature]
    print('State correlations matrix : ', state_corr.shape)
    print('State ados array shape : ', state_ados.shape)
    print('Number of connectivities turned to be significant : ', len(significant_inds))
    p = []
    state_ados_corr = []
    indices = []
    for index in significant_inds :
        corr = state_corr[:, index]
        ados_corr = np.corrcoef(corr, state_ados)
        state_ados_corr.append(ados_corr[0,1])
        p.append(p_values[index])
        indices.append(index)
        
    state_ados_corr = np.array(state_ados_corr, dtype = 'float32')
    p = np.array(p, dtype = 'float32')
    indices = np.array(indices, dtype = 'int32')
    print('ADOS correlation scores array shape : ', state_ados_corr.shape)
    print('Max correlation value : ', state_ados_corr.max())
    print('Max correlation p-value : ', p[state_ados_corr.argmax()])
    print('Min correlation value : ', state_ados_corr.min())
    print('Min correlation p-value : ', p[state_ados_corr.argmin()])
    return state_ados_corr, p, indices

def get_ados_dict(dfnc_corrs, df, input_paths, groups,  n_states, feature = 'ADOS_TOTAL'):

    """
    Returns a Dictionary containing the File_Id, distinct Correlation array (1378),  ADOS_TOTAL score per ASD subject per state
    
    Parameters 
    -----------------------------
    dfnc_corrs : <arr>, Shape : (n_subjects, 1378, n_states) 
            This array contains the medians of all windows per subject per state. 
            Here 1378 represents the number of distinct correlations between components. (53C2)
    
    df : <dataframe>, 
        DataFrame of final 383 subjects with ADOS Scores.

    input_paths : <arr>, 
        This array contains the list of all input files in alphabetical order (as processed by GIFT)
        
    groups : <dict>, format = {group_name : group_indices (Indices starts from zero)}
                Dictionary with group name and its indices

    n_states : <int>, 
                Number of states

    Returns : 
    -----------------------
    all_states_ados : <dict<dict>>, format = {state : {key : value}}
                state = <int> , State number
                key = <str>, {'File_Index', 'File_Id', 'Correlation, 'ADOS_TOTAL'}
                
                File_Id = <arr>, Shape : (n_subjects per group,)
                File_Index = <arr>, Shape : (n_subjects per group,)
                Correlation value = <arr>, Shape : (n_subjects per group, 1378)
                    Median of windows per subject per group per state 
                ADOS_TOTAL = <arr>, Shape : (n_subjects per group,)
    """

    states = np.array(range(n_states))
    all_states_ados = {}
    for state in states :  
        print('State : ', state + 1)
        indices = groups['ASD']
        all_states_ados[state] = {}
        files_idx, files_id, files_corr, files_ados = [], [], [], []
        for index in indices:
            if(~np.isnan(dfnc_corrs[index, :, state]).all()):   
                corr = dfnc_corrs[index, :, state]    # 1378 size array
                file_path = input_paths[index]
                file_id = get_key(file_path)
                ados = df.loc[file_id, feature].item()
                if(~np.isnan(ados) and ados != -9999.0):
                    files_idx.append(index)
                    files_id.append(file_id)
                    files_corr.append(corr)
                    files_ados.append(ados)

        files_idx = np.array(files_idx, dtype = 'int32')
        files_id = np.array(files_id, dtype = 'int32')
        files_corr = np.array(files_corr, dtype = 'float64')
        files_ados = np.array(files_ados, dtype = 'float64')

        all_states_ados[state]['File_Index'] = files_idx
        all_states_ados[state]['File_Id'] = files_id
        all_states_ados[state]['Correlation'] = files_corr
        all_states_ados[state][feature] = files_ados
        assert len(files_idx) == len(files_corr) == len(files_ados) == len(files_id), 'Error !'

        print(f'Group: ASD and DFNC shape : {files_corr.shape}')
        print(f'ADOS minimum: {files_ados.min()}, ADOS maximum: {files_ados.max()}')
        print('------------------------------------------------')

    return all_states_ados

def get_groups_dfnc(dfnc_corrs, groups, n_states, mode = 'all'):

    """
    Returns a Dictionary containing the centroids per subject per group per state
    
    Parameters 
    -----------------------------
    dfnc_corrs : <arr>, Shape : (n_subjects, 1378, n_states) 
            This array contains the medians of all windows per subject per state. 
            Here 1378 represents the number of distinct correlations between components. (53C2)
    
    groups : <dict>, format = {group_name : group_indices (Indices starts from zero)}
                Dictionary with group name and its indices

    n_states : <int>, 
                Number of states
    
    mode : <str> {'all', None}, 
                If mode == 'all', then centroids per group for all states will be computed. 
                Otherwise, only for nth state will be computed. 

    Returns : 
    -----------------------
    dfnc_dict : <dict<dict>>, format = {state : {group : array}}
                state = <int> , State number
                group = <str>, Group Name 
                array = <arr>, Shape : (n_subjects per group, 1378)
                    Median of windows per subject per group per state 
    """

    group_names = list(groups.keys())       # Group names 
    # n_groups = len(groups)
    dfnc_dict = {}

    if(mode == 'all'):
        states = np.array(range(n_states))
    else : 
        states = [n_states]

    for state in states : 
        dfnc_dict[state] = {}
        print('State : ', state + 1)

        for group in group_names : 
            temp = []
            indices = groups[group]
             # Checking whether the states occured in that subject or not
            for index in indices:
                if(~np.isnan(dfnc_corrs[index, :, state]).all()):   
                    temp.append(dfnc_corrs[index, :, state])    # 1378 size array
            temp = np.array(temp, dtype = 'float64')  # Shape : (n_subjects in group, 1378)
            dfnc_dict[state][group] = temp
            print(f'Group : {group} and DFNC shape : {temp.shape}')
        print('------------------------------------------------')

    return dfnc_dict

def two_sample_ttest(dfnc_dict, state):

    """
    Two sample ttest between the groups forall 1378 distinct connectivities
    
    Parameters 
    -----------------------------
    dfnc_dict : <dict<dict>>, format = {state : {group : array}}
                state = <int> , State number
                group = <str>, Group Name 
                array = <arr>, Shape : (n_subjects per group, 1378)
                    Centroids per subject per group per state 
    
    state : <int>, 
                State across which the two sample ttest should be done between the groups.  


    Returns : 
    -----------------------
    f_state : <arr>, Shape : (1378, )
            T-values of groups of the particular state for all 1378 distinct connectivities. 
    p_state : <arr>, Shape : (1378, )
            P-values of groups of the particular state for all 1378 distinct connectivities.
    """

    group_names = list(dfnc_dict[state].keys())
    t_state, p_state = [], []
    for i in range(1378):
        group1 = dfnc_dict[state][group_names[0]][:, i]
        group2 = dfnc_dict[state][group_names[1]][:, i]
        t, p = scipy.stats.ttest_ind(group1, group2)
        t_state.append(t)
        p_state.append(p)
        
    t_state = np.array(t_state, dtype = 'float64')
    p_state = np.array(p_state, dtype = 'float64')

    return t_state, p_state

def anova(dfnc_dict, state):

    """
    ANOVA between the groups for all 1378 distinct connectivities
    
    Parameters 
    -----------------------------
    dfnc_dict : <dict<dict>>, format = {state : {group : array}}
                state = <int> , State number
                group = <str>, Group Name 
                array = <arr>, Shape : (n_subjects per group, 1378)
                    Centroids per subject per group per state 
    
    state : <int>, 
                State across which the ANOVA should be done between the groups.  


    Returns : 
    -----------------------
    f_state : <arr>, Shape : (1378, )
            F-values of groups of the particular state for all 1378 distinct connectivities. 
    p_state : <arr>, Shape : (1378, )
            P-values of groups of the particular state for all 1378 distinct connectivities.
    """

    group_names = list(dfnc_dict[state].keys())
    f_state, p_state = [], []
    for i in range(1378):
        group1 = dfnc_dict[state][group_names[0]][:, i]
        group2 = dfnc_dict[state][group_names[1]][:, i]
        group3 = dfnc_dict[state][group_names[2]][:, i]
        group4 = dfnc_dict[state][group_names[3]][:, i]
        f, p = scipy.stats.f_oneway(group1, group2, group3, group4)
        f_state.append(f)
        p_state.append(p)
        
    f_state = np.array(f_state, dtype = 'float64')
    p_state = np.array(p_state, dtype = 'float64')

    return f_state, p_state


def gather_occurences(group, state, consider_none = False):
    
    """

    Given windows and states per subject, gather number of
    windows that belong to each state per ASD or TD group.

    Parameters : 
    ------------------------------
    group : State matrix of those subjects belonging to group
    state : Number of occurences of this particular state per subject will be computed.

    Returns : 
    ------------------------------
    number_of_occurences_per_subject : <arr>, Shape : (len(group), )
            Number of occurences of this particular state per subject
            
    """
    number_of_occurences_per_subject = []
    for i in range(group.shape[1]):
        each_asd_subject = group[:, i]
        # print('Number of windows in each subject : ', len(each_asd_subject))
        mask = each_asd_subject == (state +1)   
        # The states stored in matlab are [1,2,3,4] (index starts with 1)
        count = len(each_asd_subject[mask]) 
        if not consider_none :     
            if count > 0:
                number_of_occurences_per_subject.append(count)
        else : 
            number_of_occurences_per_subject.append(count)
    number_of_occurences_per_subject = np.array(number_of_occurences_per_subject, dtype = 'float32')
    return number_of_occurences_per_subject

def get_poc(file_path, groups, n_states, n_windows_per_sub, consider_none = False):

    """
    Returns a Dictionary containing the POC per subject.

    Parameters 
    -----------------------------
    file_path : <str>, 
            Mat file path to load states information. 
            Must contain 'clusterInfo' and 'states' fields.

    
    groups : <dict>, format = {group_name : group_indices (Indices starts from zero)}
                Path to save the plot

    n_states : <int>, 
                Number of states
    
    n_windows_per_sub : <int>, 
                Number of windows per subject

    Returns : 
    -----------------------
    poc_dict : <dict<dict>>, format = {state : {group : poc}}
                state = <int> , State number
                group = <str>, Group Name 
                poc = <arr>, Shape : (n_subjects per group)
                    POC of all subjects belonging to particular group and state 

    pvalues_dict : <dict>, format = {state : p-value when compared means of different groups across state}
                key = <str> , value = <arr>, Shape : (n_states,)
    """

    f = h5py.File(file_path, 'r')
    temp = f['clusterInfo']['states']
    temp = np.array(temp, dtype = 'uint32')
    temp = np.squeeze(temp, 1)
    print('States array shape : ', temp.shape)
    print('States values : ', np.unique(temp))   # These are states stored in mat files

    states = np.array(range(n_states))          # States which we are initializing[0,1,2,3]

    pvalues = {}
    main_dict = {}
    group_names = list(groups.keys())
    print(group_names)
    n_groups = len(group_names)

    for state in states:
        main_dict[state] = {}
        for group in group_names:     
            occurences_per_state = gather_occurences(temp[ : ,groups[group]], state, consider_none)
            main_dict[state][group] = occurences_per_state/n_windows_per_sub * 100
        
        if(n_groups == 2):
            group1 = np.array(list(main_dict[state][group_names[0]]))
            group2 = np.array(list(main_dict[state][group_names[1]]))
            _, pvalue = stats.ttest_ind(group1, group2)

        if(n_groups == 3):
            group1 = np.array(list(main_dict[state][group_names[0]]))
            group2 = np.array(list(main_dict[state][group_names[1]]))
            group3 = np.array(list(main_dict[state][group_names[2]]))
            _, pvalue = stats.f_oneway(group1, group2, group3)
        
        if(n_groups == 4):
            group1 = np.array(list(main_dict[state][group_names[0]]))
            group2 = np.array(list(main_dict[state][group_names[1]]))
            group3 = np.array(list(main_dict[state][group_names[2]]))
            group4 = np.array(list(main_dict[state][group_names[3]]))
            _, pvalue = stats.f_oneway(group1, group2, group3, group4)

        if(n_groups == 5):
            group1 = np.array(list(main_dict[state][group_names[0]]))
            group2 = np.array(list(main_dict[state][group_names[1]]))
            group3 = np.array(list(main_dict[state][group_names[2]]))
            group4 = np.array(list(main_dict[state][group_names[3]]))
            group5 = np.array(list(main_dict[state][group_names[4]]))
            _, pvalue = stats.f_oneway(group1, group2, group3, group4, group5)

        pvalues[state] = pvalue
    return main_dict, pvalues

def plot_poc(poc_dict, pvalues_dict, save_path = None):

    """
    Bar plot of Means with error bar representing STD of Percentage of occurrences (POC)
    for different groups across n_states 

    Parameters 
    -----------------------------
    poc_dict : <dict<dict>>, format = {state : {group : poc}}
                state = <int> , State number
                group = <str>, Group Name 
                poc = <arr>, Shape : (n_subjects per group)
                    POC of all subjects belonging to particular group and state 

    pvalues_dict : <dict>, format = {state : p-value when compared means of different groups across state}
                key = <str> , value = <arr>, Shape : (n_states,)
    save_path : <str>, 
                Path to save the plot

    Returns : 
    -----------------------
    None
    """

    states = np.array(list(pvalues_dict.keys()), dtype = 'float32')
    barWidth = 0.25
    fig, ax = plt.subplots(1, 1)
    group_names = list(poc_dict[states[0]].keys())
    n_groups = len(group_names)
    print('Number of groups : ', n_groups)
    print('Group names :', group_names)

    mean_groups, std_groups = [], []
    for group in group_names: 
        means, stds = [], []
        for state in states :
            means.append(np.mean(poc_dict[state][group]))
            stds.append(np.std(poc_dict[state][group]) + 0.0000001)
        mean_groups.append(means)
        std_groups.append(stds)

    mean_groups = np.array(mean_groups, dtype = 'float64')
    std_groups = np.array(std_groups, dtype = 'float64')
    print('Shape of Mean value of occurences array shape : ', mean_groups.shape)
    print('Shape of STD value of occurences array shape : ', std_groups.shape)

    colors = ['red', 'blue', 'green', 'orange', 'yellow']
    positions = [-0.5, -0.25, 0, 0.25, 0.5]
    for group in range(n_groups):
        ax.bar(states - positions[group], mean_groups[group],
            yerr = std_groups[group], color=colors[group], label=group_names[group], width = 0.25)

    plt.xticks([r - 0.15 for r in states], states + 1)
  
    ytick_labels = ax.get_yticks()
    new_ytick_labels = []
    for i in np.arange(len(ytick_labels)):
        t = ytick_labels[i]
        if t != 0.:
            t = str(t) + '%'
        new_ytick_labels.append(t)

    positions = np.linspace(0.1,0.9,len(states))
    for idx, i in enumerate(states) :
        ax.text(positions[idx], 0.95,
                'p=' + str(np.round(pvalues_dict[i], decimals=3)),
                transform=ax.transAxes)

    ax.set_yticklabels(new_ytick_labels)
    plt.title('Percentage of occurence of windows in different states')
    plt.ylabel('Percentage of occurences')
    plt.legend()
    if(save_path != None):
        plt.savefig(save_path)
    plt.show()
    print('Successfully plotted !!!!')