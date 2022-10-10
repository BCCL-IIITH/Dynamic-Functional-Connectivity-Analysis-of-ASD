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

def print_details(arr) :

    '''
        Prints details like shape, min, median, max, mean and std of an array
    '''
    print('Shape : ', arr.shape)
    print(f'Min Value : {np.min(arr)}, Median Value : {np.median(arr)} and  Max Value : {np.max(arr)}')
    print(f'Mean Value : {np.mean(arr)} and SD Value : {np.std(arr)}')
    return

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
            means.append(np.mean(arr[groups[group], state]))
            stds.append(np.std(arr[groups[group], state]) + 0.0000001)

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
    positions = [-0.5, -0.25, 0, 0.25, 0.5]
    fig, ax = plt.subplots(1, 1)
    for idx, group in enumerate(group_names):
        ax.bar(states - positions[idx], mean_groups[group],
            yerr=std_groups[group], color=colors[idx], label=group, width = 0.25)

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
    plt.title(f'{title} Mean and Variance differences between groups across states')
    plt.ylabel('Mean Values')
    plt.legend()
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
    n_groups = len(groups)
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
    f_state, p_state = [], []
    for i in range(1378):
        group1 = dfnc_dict[state][group_names[0]][:, i]
        group2 = dfnc_dict[state][group_names[1]][:, i]
        f, p = scipy.stats.ttest_ind(group1, group2)
        f_state.append(f)
        p_state.append(p)
        
    f_state = np.array(f_state, dtype = 'float64')
    p_state = np.array(p_state, dtype = 'float64')

    return f_state, p_state

def anova(dfnc_dict, state):

    """
    ANOVA between the groups forall 1378 distinct connectivities
    
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


def gather_occurences(group, state):
    
    """Given windows and states per subject, gather number of
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
        mask = each_asd_subject == (state +1)   
        # The states stored in matlab are [1,2,3,4] (index starts with 1)
        count = len(each_asd_subject[mask])      
        if count > 0:
            number_of_occurences_per_subject.append(count)
    number_of_occurences_per_subject = np.array(number_of_occurences_per_subject, dtype = 'float32')
    return number_of_occurences_per_subject

def get_poc(file_path, groups, n_states, n_windows_per_sub):

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
            occurences_per_state = gather_occurences(temp[ : ,groups[group]], state)
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