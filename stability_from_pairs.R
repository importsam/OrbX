#!/usr/bin/env Rscript
#
# Stability Analysis for HDBSCAN Clustering
# Uses the bootcluster package to calculate agreement between reference and bootstrap clusterings
#

library(bootcluster)

#' Calculate mean stability from bootstrap pair files
#' 
#' @param dir_pattern Regular expression pattern to match CSV files
#' @param data_dir Base directory containing the CSV files
#' @return Mean stability score across all bootstrap samples
#' 
stability_from_dir <- function(dir_pattern, data_dir = "data/analysis/stability_pairs") {
  
  # Check if directory exists
  if (!dir.exists(data_dir)) {
    stop("Directory does not exist: ", data_dir,
         "\nPlease create it and add your stability pair CSV files.")
  }
  
  # Find matching files
  files <- list.files(path = data_dir,
                      pattern = dir_pattern,
                      full.names = TRUE)
  
  if (length(files) == 0) {
    stop("No files matching pattern '", dir_pattern, "' in ", data_dir)
  }
  
  cat("Found", length(files), "files matching pattern:", dir_pattern, "\n")
  
  # Calculate agreement for each file
  scores <- numeric(length(files))
  
  for (i in seq_along(files)) {
    # Read the CSV file
    df <- read.csv(files[i])
    
    # Validate required columns
    if (!all(c("ref", "boot") %in% names(df))) {
      warning("File ", files[i], " missing 'ref' or 'boot' columns. Skipping.")
      scores[i] <- NA
      next
    }
    
    # Calculate agreement between reference and bootstrap clustering
    # agreement() returns a vector of agreement values (one per observation)
    agr <- agreement(df$ref, df$boot)
    
    # Take the mean to get overall agreement for this bootstrap sample
    scores[i] <- mean(agr, na.rm = TRUE)
    
    if (i %% 10 == 0) {
      cat("Processed", i, "of", length(files), "files\n")
    }
  }
  
  # Return mean across all bootstrap samples
  mean_stability <- mean(scores, na.rm = TRUE)
  
  cat("Mean stability:", round(mean_stability, 4), "\n")
  cat("SD of stability:", round(sd(scores, na.rm = TRUE), 4), "\n")
  cat("Min stability:", round(min(scores, na.rm = TRUE), 4), "\n")
  cat("Max stability:", round(max(scores, na.rm = TRUE), 4), "\n")
  
  return(list(
    mean = mean_stability,
    sd = sd(scores, na.rm = TRUE),
    individual_scores = scores,
    n_files = length(files)
  ))
}

# Main execution
tryCatch({
  
  # Calculate stability for HDBSCAN
  cat("\n=== Calculating HDBSCAN Stability ===\n")
  hdbscan_results <- stability_from_dir("^hdbscan_boot_.*\\.csv$")
  
  # Create results dataframe
  results <- data.frame(
    method = "HDBSCAN",
    stability = hdbscan_results$mean,
    stability_sd = hdbscan_results$sd,
    n_bootstrap = hdbscan_results$n_files,
    stringsAsFactors = FALSE
  )
  
  # Print results
  cat("\n=== Final Results ===\n")
  print(results)
  
  # Optionally save results
  output_file <- "stability_results.csv"
  write.csv(results, output_file, row.names = FALSE)
  cat("\nResults saved to:", output_file, "\n")
  
}, error = function(e) {
  cat("\nERROR:", conditionMessage(e), "\n")
  cat("\nTroubleshooting tips:\n")
  cat("1. Ensure the 'data/analysis/stability_pairs' directory exists\n")
  cat("2. Ensure CSV files with pattern 'hdbscan_boot_*.csv' are present\n")
  cat("3. Ensure CSV files have 'ref' and 'boot' columns with cluster assignments\n")
  cat("4. Check that the bootcluster package is properly installed\n")
  quit(status = 1)
})