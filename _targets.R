library(targets)
source("R/functions.R")
tar_option_set(packages=c('readxl', 'dplyr', 'tidyverse', 'glue', 'mgcv'))

tgt_feature <- c('COPD')
tp_features_4 <- c('est_vmiles', 'est_ptrp', 'ht_ami', 'emp_gravity')
re_feature <- c('State')

list(
  tar_target(file, "merged_excel.xlsx", format="file"),
  tar_target(data, get_data(file)),
  
  tar_target(model_return_4, modG(data, tgt_feature, tp_features_4, re_feature)),
  tar_target(tp_features_3, reduce_features(tp_features_4)),

  tar_target(model_return_3, modG(data, tgt_feature, tp_features_3, re_feature)),
  tar_target(tp_features_2, reduce_features(tp_features_3)),

  tar_target(model_return_2, modG(data, tgt_feature, tp_features_2, re_feature)),
  tar_target(tp_features_1, reduce_features(tp_features_2)),

  tar_target(model_return_1, modG(data, tgt_feature, tp_features_1, re_feature)),
  tar_target(tp_features_0, reduce_features(tp_features_1))
)