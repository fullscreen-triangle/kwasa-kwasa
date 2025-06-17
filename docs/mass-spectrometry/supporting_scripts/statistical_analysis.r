#!/usr/bin/env Rscript
# Statistical Analysis Module for Diabetes Biomarker Discovery
# Called by Kwasa-Kwasa framework for advanced statistical validation

library(MASS)
library(randomForest)
library(pls)
library(MetaboAnalystR)
library(jsonlite)
library(caret)

# Main function for biomarker validation
validate_biomarkers <- function(lavoisier_results_file, clinical_data_file) {
  # Load data from Lavoisier analysis
  lavoisier_results <- fromJSON(lavoisier_results_file)
  clinical_data <- fromJSON(clinical_data_file)
  
  cat("Starting statistical validation of biomarker candidates...\n")
  
  # Extract intensity matrix
  intensity_matrix <- create_intensity_matrix(lavoisier_results)
  
  # Create clinical metadata
  clinical_matrix <- create_clinical_matrix(clinical_data)
  
  # Perform advanced statistical tests
  statistical_results <- perform_advanced_statistics(intensity_matrix, clinical_matrix)
  
  # Machine learning validation
  ml_results <- perform_machine_learning_validation(intensity_matrix, clinical_matrix)
  
  # Power analysis
  power_analysis <- perform_power_analysis(intensity_matrix, clinical_matrix)
  
  # Cross-validation
  cv_results <- perform_cross_validation(intensity_matrix, clinical_matrix)
  
  # Generate final validation report
  validation_report <- list(
    statistical_validation = statistical_results,
    machine_learning = ml_results,
    power_analysis = power_analysis,
    cross_validation = cv_results,
    summary = generate_validation_summary(statistical_results, ml_results, cv_results)
  )
  
  return(validation_report)
}

create_intensity_matrix <- function(lavoisier_results) {
  # Extract biomarker candidates from Lavoisier results
  biomarkers <- lavoisier_results$biomarker_candidates
  samples <- lavoisier_results$sample_results
  
  # Create matrix: rows = samples, columns = biomarkers
  sample_ids <- sapply(samples, function(s) s$sample_id)
  biomarker_ids <- sapply(biomarkers, function(b) b$compound_id)
  
  intensity_matrix <- matrix(0, nrow=length(samples), ncol=length(biomarkers))
  rownames(intensity_matrix) <- sample_ids
  colnames(intensity_matrix) <- biomarker_ids
  
  # Fill matrix with intensity values
  for (i in seq_along(samples)) {
    sample <- samples[[i]]
    for (identification in sample$identifications) {
      compound_id <- identification$compound_id
      if (compound_id %in% biomarker_ids) {
        col_idx <- which(biomarker_ids == compound_id)
        intensity_matrix[i, col_idx] <- identification$score
      }
    }
  }
  
  return(intensity_matrix)
}

create_clinical_matrix <- function(clinical_data) {
  # Extract clinical variables
  clinical_df <- data.frame(
    sample_id = sapply(clinical_data, function(x) x$sample_id),
    group = factor(sapply(clinical_data, function(x) x$group)),
    age = as.numeric(sapply(clinical_data, function(x) ifelse(is.null(x$age), NA, x$age))),
    gender = factor(sapply(clinical_data, function(x) ifelse(is.null(x$gender), NA, x$gender))),
    bmi = as.numeric(sapply(clinical_data, function(x) ifelse(is.null(x$bmi), NA, x$bmi))),
    hba1c = as.numeric(sapply(clinical_data, function(x) ifelse(is.null(x$hba1c), NA, x$hba1c))),
    stringsAsFactors = FALSE
  )
  
  return(clinical_df)
}

perform_advanced_statistics <- function(intensity_matrix, clinical_matrix) {
  cat("Performing advanced statistical analysis...\n")
  
  results <- list()
  
  # Multivariate analysis
  results$manova <- perform_manova(intensity_matrix, clinical_matrix)
  
  # Partial Least Squares Discriminant Analysis
  results$plsda <- perform_plsda(intensity_matrix, clinical_matrix)
  
  # Univariate analysis with multiple testing correction
  results$univariate <- perform_univariate_analysis(intensity_matrix, clinical_matrix)
  
  # Effect size calculations
  results$effect_sizes <- calculate_effect_sizes(intensity_matrix, clinical_matrix)
  
  # Biomarker panel optimization
  results$panel_optimization <- optimize_biomarker_panel(intensity_matrix, clinical_matrix)
  
  return(results)
}

perform_manova <- function(intensity_matrix, clinical_matrix) {
  # Multivariate analysis of variance
  group <- clinical_matrix$group
  
  # Remove samples with missing group information
  valid_samples <- !is.na(group)
  intensity_clean <- intensity_matrix[valid_samples, ]
  group_clean <- group[valid_samples]
  
  # Perform MANOVA
  manova_result <- manova(intensity_clean ~ group_clean)
  manova_summary <- summary(manova_result)
  
  return(list(
    result = manova_result,
    summary = manova_summary,
    p_value = manova_summary$stats[1, "Pr(>F)"],
    interpretation = ifelse(manova_summary$stats[1, "Pr(>F)"] < 0.05, 
                           "Significant multivariate difference between groups", 
                           "No significant multivariate difference")
  ))
}

perform_plsda <- function(intensity_matrix, clinical_matrix) {
  group <- clinical_matrix$group
  
  # Remove samples with missing group information
  valid_samples <- !is.na(group)
  X <- intensity_matrix[valid_samples, ]
  Y <- group[valid_samples]
  
  # Perform PLS-DA
  plsda_model <- plsr(Y ~ X, ncomp = min(10, ncol(X)-1), validation = "CV")
  
  # Extract results
  scores <- plsda_model$scores
  loadings <- plsda_model$loadings
  explained_variance <- explvar(plsda_model)
  
  # Calculate classification accuracy
  predicted_classes <- predict(plsda_model, X, ncomp = 2, type = "response")
  accuracy <- mean(predicted_classes == Y)
  
  return(list(
    model = plsda_model,
    scores = scores,
    loadings = loadings,
    explained_variance = explained_variance,
    accuracy = accuracy,
    num_components = 2
  ))
}

perform_univariate_analysis <- function(intensity_matrix, clinical_matrix) {
  group <- clinical_matrix$group
  
  # Remove samples with missing group information
  valid_samples <- !is.na(group)
  intensity_clean <- intensity_matrix[valid_samples, ]
  group_clean <- group[valid_samples]
  
  # Perform t-tests for each biomarker
  p_values <- c()
  fold_changes <- c()
  
  for (i in 1:ncol(intensity_clean)) {
    biomarker_values <- intensity_clean[, i]
    
    # Split by group
    diabetes_values <- biomarker_values[group_clean == "diabetes"]
    control_values <- biomarker_values[group_clean == "control"]
    
    # T-test
    if (length(diabetes_values) > 1 && length(control_values) > 1) {
      t_test <- t.test(diabetes_values, control_values)
      p_values[i] <- t_test$p.value
      
      # Fold change (log2)
      mean_diabetes <- mean(diabetes_values, na.rm = TRUE)
      mean_control <- mean(control_values, na.rm = TRUE)
      fold_changes[i] <- log2((mean_diabetes + 1) / (mean_control + 1))
    } else {
      p_values[i] <- NA
      fold_changes[i] <- NA
    }
  }
  
  # Multiple testing correction
  p_values_adjusted <- p.adjust(p_values, method = "fdr")
  
  # Create results data frame
  univariate_results <- data.frame(
    biomarker = colnames(intensity_clean),
    p_value = p_values,
    p_value_adjusted = p_values_adjusted,
    fold_change = fold_changes,
    significant = p_values_adjusted < 0.05,
    stringsAsFactors = FALSE
  )
  
  return(univariate_results)
}

calculate_effect_sizes <- function(intensity_matrix, clinical_matrix) {
  group <- clinical_matrix$group
  
  # Remove samples with missing group information
  valid_samples <- !is.na(group)
  intensity_clean <- intensity_matrix[valid_samples, ]
  group_clean <- group[valid_samples]
  
  effect_sizes <- c()
  
  for (i in 1:ncol(intensity_clean)) {
    biomarker_values <- intensity_clean[, i]
    
    # Split by group
    diabetes_values <- biomarker_values[group_clean == "diabetes"]
    control_values <- biomarker_values[group_clean == "control"]
    
    # Cohen's d
    if (length(diabetes_values) > 1 && length(control_values) > 1) {
      mean_diff <- mean(diabetes_values, na.rm = TRUE) - mean(control_values, na.rm = TRUE)
      pooled_sd <- sqrt(((length(diabetes_values) - 1) * var(diabetes_values, na.rm = TRUE) + 
                        (length(control_values) - 1) * var(control_values, na.rm = TRUE)) / 
                       (length(diabetes_values) + length(control_values) - 2))
      effect_sizes[i] <- mean_diff / pooled_sd
    } else {
      effect_sizes[i] <- NA
    }
  }
  
  return(data.frame(
    biomarker = colnames(intensity_clean),
    cohens_d = effect_sizes,
    effect_size_interpretation = sapply(abs(effect_sizes), function(d) {
      if (is.na(d)) return("Unknown")
      if (d < 0.2) return("Negligible")
      if (d < 0.5) return("Small")
      if (d < 0.8) return("Medium")
      return("Large")
    })
  ))
}

optimize_biomarker_panel <- function(intensity_matrix, clinical_matrix) {
  group <- clinical_matrix$group
  
  # Remove samples with missing group information
  valid_samples <- !is.na(group)
  X <- intensity_matrix[valid_samples, ]
  Y <- as.factor(group[valid_samples])
  
  # Forward selection to find optimal biomarker combination
  selected_biomarkers <- c()
  best_accuracy <- 0
  
  for (n_biomarkers in 1:min(10, ncol(X))) {
    cat(sprintf("Testing panels with %d biomarkers...\n", n_biomarkers))
    
    # Try adding each remaining biomarker
    remaining_biomarkers <- setdiff(1:ncol(X), selected_biomarkers)
    candidate_accuracies <- c()
    
    for (candidate in remaining_biomarkers) {
      test_panel <- c(selected_biomarkers, candidate)
      X_subset <- X[, test_panel, drop = FALSE]
      
      # Cross-validation
      cv_folds <- createFolds(Y, k = 5)
      cv_accuracies <- c()
      
      for (fold in cv_folds) {
        train_X <- X_subset[-fold, , drop = FALSE]
        train_Y <- Y[-fold]
        test_X <- X_subset[fold, , drop = FALSE]
        test_Y <- Y[fold]
        
        # Simple logistic regression
        if (ncol(train_X) == 1) {
          model <- glm(train_Y ~ train_X[,1], family = binomial)
          predictions <- predict(model, data.frame(V1 = test_X[,1]), type = "response")
        } else {
          model <- glm(train_Y ~ ., data = data.frame(train_X), family = binomial)
          predictions <- predict(model, data.frame(test_X), type = "response")
        }
        
        predicted_classes <- ifelse(predictions > 0.5, levels(Y)[2], levels(Y)[1])
        cv_accuracies <- c(cv_accuracies, mean(predicted_classes == test_Y))
      }
      
      candidate_accuracies <- c(candidate_accuracies, mean(cv_accuracies))
    }
    
    # Select best candidate
    best_candidate_idx <- which.max(candidate_accuracies)
    best_candidate <- remaining_biomarkers[best_candidate_idx]
    best_candidate_accuracy <- candidate_accuracies[best_candidate_idx]
    
    # Add to panel if improvement
    if (best_candidate_accuracy > best_accuracy) {
      selected_biomarkers <- c(selected_biomarkers, best_candidate)
      best_accuracy <- best_candidate_accuracy
      cat(sprintf("Added biomarker %s (accuracy: %.3f)\n", 
                  colnames(X)[best_candidate], best_accuracy))
    } else {
      break
    }
  }
  
  return(list(
    selected_biomarkers = colnames(X)[selected_biomarkers],
    biomarker_indices = selected_biomarkers,
    panel_accuracy = best_accuracy,
    panel_size = length(selected_biomarkers)
  ))
}

perform_machine_learning_validation <- function(intensity_matrix, clinical_matrix) {
  cat("Performing machine learning validation...\n")
  
  group <- clinical_matrix$group
  
  # Remove samples with missing group information
  valid_samples <- !is.na(group)
  X <- intensity_matrix[valid_samples, ]
  Y <- as.factor(group[valid_samples])
  
  # Random Forest
  rf_model <- randomForest(X, Y, ntree = 500, importance = TRUE)
  rf_accuracy <- mean(predict(rf_model, X) == Y)
  
  # Feature importance
  importance_scores <- importance(rf_model)[, "MeanDecreaseGini"]
  
  return(list(
    random_forest = list(
      model = rf_model,
      accuracy = rf_accuracy,
      importance_scores = importance_scores,
      top_features = names(sort(importance_scores, decreasing = TRUE))[1:10]
    )
  ))
}

perform_power_analysis <- function(intensity_matrix, clinical_matrix) {
  # Estimate statistical power for current sample size
  group <- clinical_matrix$group
  
  # Remove samples with missing group information
  valid_samples <- !is.na(group)
  group_clean <- group[valid_samples]
  
  n_diabetes <- sum(group_clean == "diabetes")
  n_control <- sum(group_clean == "control")
  
  # Estimate effect size from data
  if (n_diabetes > 1 && n_control > 1) {
    # Use first biomarker as example
    example_values <- intensity_matrix[valid_samples, 1]
    diabetes_values <- example_values[group_clean == "diabetes"]
    control_values <- example_values[group_clean == "control"]
    
    effect_size <- (mean(diabetes_values) - mean(control_values)) / 
                   sqrt(((n_diabetes - 1) * var(diabetes_values) + 
                        (n_control - 1) * var(control_values)) / 
                       (n_diabetes + n_control - 2))
    
    # Simple power calculation (approximation)
    power_estimate <- pnorm(abs(effect_size) * sqrt(n_diabetes * n_control / (n_diabetes + n_control)) - 
                           qnorm(0.975))
  } else {
    effect_size <- NA
    power_estimate <- NA
  }
  
  return(list(
    sample_sizes = list(diabetes = n_diabetes, control = n_control),
    estimated_effect_size = effect_size,
    estimated_power = power_estimate,
    power_interpretation = ifelse(is.na(power_estimate), "Cannot calculate", 
                                 ifelse(power_estimate > 0.8, "Adequate", "Insufficient"))
  ))
}

perform_cross_validation <- function(intensity_matrix, clinical_matrix) {
  cat("Performing cross-validation...\n")
  
  group <- clinical_matrix$group
  
  # Remove samples with missing group information
  valid_samples <- !is.na(group)
  X <- intensity_matrix[valid_samples, ]
  Y <- as.factor(group[valid_samples])
  
  # 5-fold cross-validation
  cv_folds <- createFolds(Y, k = 5)
  cv_results <- list()
  
  for (i in seq_along(cv_folds)) {
    fold <- cv_folds[[i]]
    train_X <- X[-fold, ]
    train_Y <- Y[-fold]
    test_X <- X[fold, ]
    test_Y <- Y[fold]
    
    # Train random forest
    rf_model <- randomForest(train_X, train_Y, ntree = 100)
    predictions <- predict(rf_model, test_X)
    
    # Calculate metrics
    accuracy <- mean(predictions == test_Y)
    
    cv_results[[i]] <- list(
      fold = i,
      accuracy = accuracy,
      confusion_matrix = table(predictions, test_Y)
    )
  }
  
  # Overall CV performance
  cv_accuracies <- sapply(cv_results, function(x) x$accuracy)
  
  return(list(
    individual_folds = cv_results,
    mean_accuracy = mean(cv_accuracies),
    std_accuracy = sd(cv_accuracies),
    confidence_interval = mean(cv_accuracies) + c(-1, 1) * 1.96 * sd(cv_accuracies) / sqrt(length(cv_accuracies))
  ))
}

generate_validation_summary <- function(statistical_results, ml_results, cv_results) {
  summary <- list(
    overall_assessment = "Analysis complete",
    statistical_significance = any(statistical_results$univariate$significant, na.rm = TRUE),
    num_significant_biomarkers = sum(statistical_results$univariate$significant, na.rm = TRUE),
    plsda_accuracy = statistical_results$plsda$accuracy,
    random_forest_accuracy = ml_results$random_forest$accuracy,
    cross_validation_accuracy = cv_results$mean_accuracy,
    panel_optimization = statistical_results$panel_optimization,
    recommendations = c()
  )
  
  # Generate recommendations
  if (summary$cross_validation_accuracy > 0.8) {
    summary$recommendations <- c(summary$recommendations, "Strong predictive performance - consider clinical validation")
  } else if (summary$cross_validation_accuracy > 0.7) {
    summary$recommendations <- c(summary$recommendations, "Moderate predictive performance - consider panel optimization")
  } else {
    summary$recommendations <- c(summary$recommendations, "Weak predictive performance - consider additional biomarkers")
  }
  
  if (summary$num_significant_biomarkers < 3) {
    summary$recommendations <- c(summary$recommendations, "Few significant biomarkers found - consider larger sample size")
  }
  
  return(summary)
}

# Main execution when called from command line
if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) < 2) {
    cat("Usage: Rscript statistical_analysis.r <lavoisier_results.json> <clinical_data.json>\n")
    quit(status = 1)
  }
  
  lavoisier_results_file <- args[1]
  clinical_data_file <- args[2]
  
  # Perform validation
  validation_results <- validate_biomarkers(lavoisier_results_file, clinical_data_file)
  
  # Output results as JSON
  cat(toJSON(validation_results, auto_unbox = TRUE, pretty = TRUE))
} 