%% Modality Type                                                                                                                         
modalityType = 'fMRI';                                                                                                                   
                                                                                                                                         
%% Output Directory                                                                                                                      
outputDir = '/scratch/Krishna/Data_CCS/All_age_groups_ccs/Mask_ccs_all';                                                                 
                                                                                                                                         
%% All the output files will be preprended with the specified prefix.                                                                    
prefix = 'All_ccs_';                                                                                                                     
                                                                                                                                         
%% Group PCA performance settings. Best setting for each option will be selected based on variable MAX_AVAILABLE_RAM in icatb_defaults.m.
perfType = 1;                                                                                                                            
                                                                                                                                         
%% Input data file names written to a text file.                                                                                         
txtFileName = '/scratch/Krishna/Data_CCS/All_age_groups_ccs/Mask_ccs_all/All_ccs__input_files.txt';                                      
                                                                                                                                         
%% Data selection option. If option 4 is specified, file names must be entered in input_data_file_patternss                              
dataSelectionMethod = 4;                                                                                                                 
                                                                                                                                         
%% Input data file pattern for data-sets must be in a cell array. The no. of rows of cell array correspond to no. of data-sets.          
input_data_file_patterns = textread(txtFileName, '%s', 'delimiter', '\n');                                                               
                                                                                                                                         
%% Design matrix/matrices.                                                                                                               
input_design_matrices = {};                                                                                                              
                                                                                                                                         
%% Number of dummy scans.                                                                                                                
dummy_scans = 0;                                                                                                                         
                                                                                                                                         
%% Full file path of the mask file.                                                                                                      
maskFile = '/scratch/Krishna/Data_CCS/All_age_groups_ccs/Mask_ccs_all/All_ccs_Mask.nii';                                                 
                                                                                                                                         
%% Back-reconstruction type.                                                                                                             
backReconType = 'gica';                                                                                                                  
                                                                                                                                         
%% Data pre-processing option. By default, Remove Mean Per Timepoint is used                                                             
preproc_type = 1;                                                                                                                        
                                                                                                                                         
%% Number of data reduction steps used.                                                                                                  
numReductionSteps = 2;                                                                                                                   
                                                                                                                                         
%% Scale components. By default, components are converted to z-scores.                                                                   
scaleType = 2;                                                                                                                           
                                                                                                                                         
%% ICA algorithm to use. Infomax is the default choice.                                                                                  
algoType = 'infomax';                                                                                                                    
                                                                                                                                         
%% Number of principal components used in the first PCA step.                                                                            
numOfPC1 = 30;                                                                                                                           
                                                                                                                                         
%% Number of principal components used in the second PCA step. Also the number of independent components extracted from the data.        
numOfPC2 = 20;                                                                                                                           
                                                                                                                                         
