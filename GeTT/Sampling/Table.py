import numpy as np
import pandas as pd
from Sampling.Features import derive_meta_features
from Sampling.TFIDF_sampling import roulette_row_selection
from Sampling.columnSampling import value_frequency, random_keys_by_frequency
from Sampling.TFIDF_sampling import compute_avg_tfidf
from Sampling.Token import tokenize
from Sampling.dataTypes import determine_data_type


def get_df_sample(df, len_context, full=False, other_col=False, max_len=8000):
    column_samples = {}
    ignore_list = ["None", 'none', 'NaN', 'nan', 'N/A', 'na', '']
    for col in df.columns:
        sample_list = list(
            set(p[:max_len // (len_context * 3)] for p in pd.unique(df.astype(str)[col]) if p not in ignore_list))
        if full:
            meta_features = derive_meta_features(df[col])
            meta_features['rolling-mean-window-4'] = meta_features['rolling-mean-window-4'][:5]
        # Sampling from other columns
        if other_col:
            sample_list_fill_size = len_context - len(sample_list)
            nc = len(df.columns)
            per_column_context = max(1, sample_list_fill_size // nc)
            for idx, oc in enumerate(df.columns):
                items = df[oc].astype(str).iloc[0:per_column_context].tolist()
                sample_list = sample_list + ["OC: " + str(item) for item in items]
        if not sample_list:
            sample_list = ["None"]
        if len(sample_list) < len_context:
            sample_list = sample_list * len_context
        if len(sample_list) > len_context:
            sample_list = sample_list[:len_context]
        assert len(sample_list) == len_context, "An index in val_indices is length " + str(len(sample_list))
        if full:
            if meta_features['std'] == "N/A":
                sample_list = sample_list + ["" for k, v in meta_features.items()]
            else:
                sample_list = sample_list + [str(k) + ": " + str(v) for k, v in meta_features.items()]
        column_samples[col] = sample_list

    return pd.DataFrame.from_dict(column_samples)


def randomlySelect(df_column, sample_size=5):
    """
    randomly select [sample_size] unique tokens from the column
    """
    tokenized_values = df_column.apply(tokenize).explode()
    unique_tokens = tokenized_values.unique()
    return np.random.choice(unique_tokens, size=sample_size, replace=False)


def weighted_random_selection(column: pd.Series, num_samples):
    avg_tfidf_scores = compute_avg_tfidf(column)
    entries = list(avg_tfidf_scores.keys())
    weights = list(avg_tfidf_scores.values())

    try:
        selected_entries = np.random.choice(entries, size=num_samples, replace=False,
                                            p=np.array(weights) / np.sum(weights))
        return selected_entries.tolist()
    except ValueError as e:
        print("Error in selection process:", e)
        return []


def sample_column(col: pd.Series, sample=5, head=False):
    col_dict = value_frequency(col)
    if sample >= len(col):
        return col_dict, list(col)
    elif len(col_dict) <= sample:
        return col_dict, col_dict.keys()
    else:
        if head is True:
            return col_dict, list(col.head(5))
        else:
            sample_dict = random_keys_by_frequency(col, num_samples=sample)
            return col_dict, list(sample_dict.keys())


def column_context(column: pd.Series, fraction=5):
    if fraction >= len(column):
        return column
    else:
        col_dict, sample_cells = sample_column(column, sample=fraction, head=True)
    Meta_info = ""
    col_name = column.name
    sample_cells_text  = f"{col_name}: {', '.join(map(str, sample_cells))}"
    if pd.api.types.is_numeric_dtype(column) is True:
        Meta_info = derive_meta_features(column)
    if Meta_info != "":
        context_column = f"{col_name}: {sample_cells_text}, Meta_info: {Meta_info}"
    else:
        context_column = f"{col_name}: {sample_cells_text}"
    return context_column


def colMetadata(col: pd.Series, sample=5):
    # Step 1: Check the Column Header or Name
    header = col.name
    # Step 2: Examine the Data Type (Text, Number, Boolean, Date, DateTime, Time)
    dataType = determine_data_type(col)
    # Step 3: Evaluate Value Uniqueness and Cardinality and
    total_length = len(col)
    # Step 4: Assess Length and Structure of Values
    features = derive_meta_features(col)
    characteristics = "cell length" if dataType != "Number" else "cell"
    if dataType == "Number":
        col_dict, sample_cells = sample_column(col, sample=sample, head=True)
    else:
        col_dict, sample_cells = sample_column(col, sample=sample)
    cardinality = len(col_dict)
    sample_cell_distribution = ""
    if dataType == "Text":
        for i, fre in col_dict.items():
            if i in sample_cells:
                sample_cell_distribution += f"{i}: {fre}, "
        #sample_cell_distribution = f"7. sample cells frequency distribution: {sample_cell_distribution}\n"

    # Step 5: Analyze Frequency Distribution and Repetition (col_dict). here we only show the sampled cells
    context = (f"1. Column Header: {header}\n" 
               f"2. sampled cells: {sample_cells}\n"
               f"3. dataType: {dataType}\n"
               f"4. # unique cell values: {cardinality}\n"
               f"5.# cells: {total_length}\n"
               f"6. {characteristics} features:{features}\n"

               )
    return context


def example_data(df, method='simple', sample_size=5, summ_stats=False, other_col=False, MAX_LEN=8000):
    if len(df) <= sample_size:
        return df
    if method == 'ArcheType':
        sample_df = get_df_sample(df, len_context=sample_size, full=summ_stats, other_col=other_col, max_len=MAX_LEN)
    elif method == 'simple':
        sample_df = df.sample(n=sample_size)
    elif method == 'head':
        sample_df = df.head(sample_size)
    else:
        return df
    return sample_df
