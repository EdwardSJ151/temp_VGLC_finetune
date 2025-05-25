from sacrebleu.metrics import CHRF
import textdistance
import numpy as np
from tqdm import tqdm
import concurrent.futures


def print_level_from_columns(columns, line_size=50):
    """
    Helper function for SampledLevelEvaluator.column_analysis
    """
    height = len(columns[0]) if columns else 0
    
    level_grid = []
    for _ in range(height):
        level_grid.append([''] * len(columns))
    
    for col_idx, column in enumerate(columns):
        for row_idx, char in enumerate(column):
            level_grid[row_idx][col_idx] = char
    
    level_lines = [''.join(row) for row in level_grid]
    level_string = '\n'.join(level_lines)
    
    print(level_string)


class SampledLevelEvaluator:

    def column_analysis(self, level_str: str, reference_str: str, analyse_consecutive: bool = False, line_size: int = 50, progression_type: str = "horizontal") -> dict:
        """
        Function to analyse how many columns in the level are the same as the reference, as well as a consecutive analysis of columns.

        Args:
            level_str (str): The flattened string representation of the sampled level
            reference_str (str): The flattened string for the reference level
            analyse_consecutive (bool): Whether to analyze consecutive matching columns
            line_size (int): Size of each line in the level
            progression_type (str): Either "horizontal" or "vertical" to determine how to process columns

        Returns:
            dict: Contains metrics about column matching and the matching sequences themselves
        """
        # Split strings into lines based on line_size
        level_lines = [level_str[i:i+line_size] for i in range(0, len(level_str), line_size)]
        reference_lines = [reference_str[i:i+line_size] for i in range(0, len(reference_str), line_size)]

        # Process based on progression type
        if progression_type == "vertical":
            level_columns = level_lines
            reference_columns = reference_lines
        else:  # horizontal
            # Get number of lines (Y chars)
            num_lines = len(level_lines)
            # Create columns by taking nth character from each line
            level_columns = []
            reference_columns = []
            for i in range(line_size):
                level_column = ''.join(line[i] for line in level_lines[:num_lines])
                reference_column = ''.join(line[i] for line in reference_lines[:num_lines])
                level_columns.append(level_column)
                reference_columns.append(reference_column)

        # Count matching columns
        total_columns = len(level_columns)
        matching_columns = []
        matching_sequences = []  # Store the actual matching sequences

        # Find matching columns
        for i, level_col in enumerate(level_columns):
            if level_col in reference_columns:
                matching_columns.append(i)
                matching_sequences.append(level_col)  # Store the matching sequence

        # Format sequences for better visibility
        def format_sequence(sequence):
            # Split the sequence into lines of line_size
            lines = [sequence[i:i+line_size] for i in range(0, len(sequence), line_size)]
            # Join with newlines
            return '\n'.join(lines)

        # Format all matching sequences
        formatted_matching_sequences = [format_sequence(seq) for seq in matching_sequences]

        # Calculate basic metrics
        results = {
            "total_columns": total_columns,
            "matching_columns": len(matching_columns),
            "matching_ratio": len(matching_columns) / total_columns if total_columns > 0 else 0,
            "matching_sequences": formatted_matching_sequences  # Add formatted matching sequences
        }

        # Analyze consecutive matches if requested
        if analyse_consecutive:
            max_consecutive = 0
            current_consecutive = 0
            longest_consecutive_sequence = []  # Store the longest consecutive sequence
            current_sequence = []  # Store current sequence being checked

            # Check each possible starting position
            for start_idx in range(len(level_columns)):
                current_consecutive = 0
                ref_idx = 0
                current_sequence = []

                # Check consecutive matches from this starting position
                for level_idx in range(start_idx, len(level_columns)):
                    if ref_idx >= len(reference_columns):
                        break

                    if level_columns[level_idx] == reference_columns[ref_idx]:
                        current_consecutive += 1
                        current_sequence.append(level_columns[level_idx])
                        ref_idx += 1
                    else:
                        # If we found a longer sequence, update the longest
                        if current_consecutive > max_consecutive:
                            max_consecutive = current_consecutive
                            longest_consecutive_sequence = current_sequence.copy()
                        current_consecutive = 0
                        current_sequence = []
                        ref_idx = 0

                # Check if the last sequence was the longest
                if current_consecutive > max_consecutive:
                    max_consecutive = current_consecutive
                    longest_consecutive_sequence = current_sequence.copy()

            # Format the longest consecutive sequence
            formatted_longest_sequence = [format_sequence(seq) for seq in longest_consecutive_sequence]

            results["max_consecutive_matches"] = max_consecutive
            results["consecutive_ratio"] = max_consecutive / total_columns if total_columns > 0 else 0
            results["longest_consecutive_sequence"] = formatted_longest_sequence  # Add formatted longest consecutive sequence

        return results

    @staticmethod
    def calculate_generation_diff(original_count, level_to_compare, separator=None):
        current_len = len(level_to_compare)

        if separator:
            current_len = len(level_to_compare.replace(separator, ""))

        difference = abs(original_count - current_len)

        if original_count == 0:
            return float('inf') if difference != 0 else 0.0
        return (difference / original_count) * 100

    def evaluate_level_str(self, level_str: str, reference_str: str, metrics: list = None) -> dict:
        """
        Args:
            level_str (str): The flattened string representation of the sampled level.
            reference_str (str): The flattened string for the reference level.
            metrics (list): List of metric keys to compute. Options include:
                "chrF_score", "levenshtein_distance", "hamming_distance",
                "lcs_substring_similarity", "lcs_subsequence_similarity".
                If None, all are computed.

        Returns:
            dict: A dictionary containing scores for the selected metrics.
        """
        if metrics is None:
            metrics = [
                "chrF_score", "levenshtein_distance", "hamming_distance",
                "lcs_substring_similarity", "lcs_subsequence_similarity"
            ]

        results = {}

        if "chrF_score" in metrics:
            chrf = CHRF(word_order=0, char_order=70)
            chrf_score_obj = chrf.corpus_score([level_str], [[reference_str]])
            results["chrF_score"] = chrf_score_obj.score

        if "levenshtein_distance" in metrics:
            results["levenshtein_distance"] = textdistance.levenshtein.distance(level_str, reference_str)
            results["levenshtein_normalized"] = textdistance.levenshtein.normalized_similarity(level_str, reference_str)

        if "hamming_distance" in metrics:
            if len(level_str) == len(reference_str):
                results["hamming_distance"] = textdistance.hamming.distance(level_str, reference_str)
                results["hamming_normalized"] = textdistance.hamming.normalized_similarity(level_str, reference_str)
            else:
                results["hamming_distance"] = None
                results["hamming_normalized"] = None

        if "lcs_substring_similarity" in metrics:
            lcs_str = textdistance.LCSStr()
            results["lcs_substring_similarity"] = lcs_str.similarity(level_str, reference_str)
            results["lcs_substring_normalized"] = lcs_str.normalized_similarity(level_str, reference_str)

        if "lcs_subsequence_similarity" in metrics:
            lcs_seq = textdistance.LCSSeq()
            results["lcs_subsequence_similarity"] = lcs_seq.similarity(level_str, reference_str)
            results["lcs_subsequence_normalized"] = lcs_seq.normalized_similarity(level_str, reference_str)

        return results


    def evaluate_sample_on_dataset(self, sampled_levels: list, reference_str: str, metrics: list = None, max_workers: int = 6) -> dict:
        """
        Evaluate levels on a complete dataset.

        Returns the best level per metric (max for high, min for low).
        """
        if metrics is None:
            metrics = [
                "chrF_score", "levenshtein_distance", "hamming_distance",
                "lcs_substring_similarity", "lcs_subsequence_similarity"
            ]

        use_parallel = ("lcs_substring_similarity" in metrics or "lcs_subsequence_similarity" in metrics)
        higher_better = {"chrF_score", "lcs_substring_similarity", "lcs_subsequence_similarity", "hamming_normalized"}
        lower_better = {"levenshtein_distance", "hamming_distance"}

        best_metrics = {}
        for metric in metrics:
            if metric in higher_better:
                best_metrics[metric] = {"score": -float("inf"), "level": None}
            elif metric in lower_better:
                best_metrics[metric] = {"score": float("inf"), "level": None}
            else:
                raise ValueError(f"Issue with metrics, no specification on higher or lower better")

        all_sample_metrics = []
        if use_parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_sample = {
                    executor.submit(self.evaluate_level_str, sample, reference_str, metrics): sample
                    for sample in sampled_levels
                }

                for future in tqdm(concurrent.futures.as_completed(future_to_sample),
                                   total=len(sampled_levels),
                                   desc="Calculating metrics (parallel)"):
                    sample = future_to_sample[future]
                    try:
                        calculated_metrics = future.result()
                        all_sample_metrics.append({"level": sample, "metrics": calculated_metrics})
                    except Exception as exc:
                        print(f'Parallel sample generation for level "{sample[:20]}..." threw an error: {exc}')
        else:
            for sample in tqdm(sampled_levels, desc="Calculating metrics (sequential)"):
                calculated_metrics = self.evaluate_level_str(sample, reference_str, metrics=metrics)
                all_sample_metrics.append({"level": sample, "metrics": calculated_metrics})


        for result in tqdm(all_sample_metrics, desc="Finding best metrics"):
            level_str = result["level"]
            sample_metrics = result["metrics"]
            for metric in metrics:
                if metric not in sample_metrics:
                    raise ValueError(f"Metric {metric} not found in sample metrics")
                value = sample_metrics.get(metric)

                if metric in higher_better:
                    if value is not None and value > best_metrics[metric]["score"]:
                        best_metrics[metric]["score"] = value
                        best_metrics[metric]["level"] = level_str
                elif metric in lower_better:
                    if value is not None and value < best_metrics[metric]["score"]:
                        best_metrics[metric]["score"] = value
                        best_metrics[metric]["level"] = level_str

        return best_metrics