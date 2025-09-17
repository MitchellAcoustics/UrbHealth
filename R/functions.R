
get_data <- function(file) {
  dvs <- c('CHD', 'Depression', 'COPD', 'DIABETES', 'HIGHCHOL', 'BPHIGH', 'Obesity')

  data <- read_excel(file) |> # non-cleaned data
    rename(PAProp = "Park Area - Proportion", StopsSqMile = "Stops per Sq Mile") |>
    select(!starts_with("Unnamed")) |> # remove extra columns
    filter(State != "FL") |> # remove Florida data
    filter(emp_gravity < 300000) # remove emp_gravity single outlier

  data$est_prtp <- na_if(data$est_ptrp, 0) # replace 0 with NA
  data$ht_ami <- na_if(data$ht_ami, 0)
  data$TractID <- factor(data$TractID) # convert to factor variable
  data$State <- factor(data$State)

  data[dvs] <- data[dvs] / 100 # convert to percentage 0-1

  data <- drop_na(data) # drop missing data
}

modG <- function(data, tgt_feature, tp_features, re_feature) {
  tp_string <- paste("s(", tp_features, collapse =") + ", sep="")
  
  formula <- as.formula(
    glue(
      "{tgt_feature} ~ {tp_string}) + s({re_feature}, bs='re')"
    )
  )

  modG <- bam(formula, data=data, method="fREML", family="quasibinomial")
  summary(modG)

  return(modG)
}

reduce_features <- function(features) {
  head(features, -1)
}