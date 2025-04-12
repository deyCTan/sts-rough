# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
sts_cmb = dataiku.Dataset("sts_cmb")
sts_cmb_df = sts_cmb.get_dataframe(infer_with_pandas=False)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# List of columns to check
columns_to_check = ["observation_category_text", "observation", "solution", "solution_category", "problem_cause_text"]

# Check and print data types
for column in columns_to_check:
    dtype = sts_cmb_df[column].dtype if column in sts_cmb_df.columns else "Column not found"
    print(f"Data type of '{column}': {dtype}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# List of columns to check
columns_to_check = ["observation_category_text", "observation", "solution", "solution_category", "problem_cause_text"]

# Check for mixed data types
for column in columns_to_check:
    if column in sts_cmb_df.columns:
        unique_types = sts_cmb_df[column].apply(type).nunique()
        print(f"Column '{column}' has mixed data types: {unique_types > 1}")
    else:
        print(f"Column '{column}' does not exist in the DataFrame.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Columns to convert to string
columns_to_convert = ["observation_category_text", "solution_category", "problem_cause_text"]

# Convert columns to string and handle NaN
for column in columns_to_convert:
    if column in sts_cmb_df.columns:
        sts_cmb_df[column] = sts_cmb_df[column].apply(lambda x: str(x).strip() if not pd.isna(x) else "")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Create `observation_final` column
sts_cmb_df['observation_final'] = sts_cmb_df.apply(
    lambda row: row['observation'] if pd.isna(row['observation_category_text']) or row['observation_category_text'] == ''
    else f"{row['observation_category_text']}-{row['observation']}",
    axis=1
)

# Create `solution_final` column
sts_cmb_df['solution_final'] = sts_cmb_df.apply(
    lambda row: row['solution'] if pd.isna(row['solution_category']) or row['solution_category'] == ''
    else f"{row['solution_category']}-{row['solution']}",
    axis=1
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# List of columns to check for NoneType
columns_to_check = ["solution_final", "observation_final", "problem_cause_text"]

# Check for NoneType in each column
for column in columns_to_check:
    if column in sts_cmb_df.columns:
        has_none = sts_cmb_df[column].apply(lambda x: isinstance(x, type(None))).any()
        print(f"Column '{column}' contains NoneType: {has_none}")
    else:
        print(f"Column '{column}' does not exist in the DataFrame.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Import necessary libraries
import dataiku
import pandas as pd
import numpy as np
import datetime
import concurrent.futures
from tqdm.notebook import tqdm
import logging
import re

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Utility functions
def is_removable(val):
    """Check if a value should be considered removable."""
    return val is None or pd.isna(val) or str(val).strip().lower() in ["nan", "none", ""]

def is_numeric_or_alphanumeric(value):
    """Check if the value is numeric or alphanumeric."""
    return bool(re.fullmatch(r'[A-Za-z0-9]+', str(value).strip()))

def clean_columns(df, columns):
    """Clean specified columns in the DataFrame by replacing invalid values with an empty string."""
    for col in columns:
        df[col] = df[col].apply(lambda x: "" if is_removable(x) else str(x).strip())
    return df

def translate_record(record_data, llm, language_map):
    """Translate a single record using the provided LLM."""
    index, row_dict, language = record_data
    translations = {}
    lang_name = language_map.get(language, "Unknown")

    columns_to_translate = ["observation_final", "solution_final", "problem_cause_text"]

    for column in columns_to_translate:
        value = row_dict.get(column, "")
        original_text = str(value).strip() if value else ""

        # Skip translation for numeric or alphanumeric values
        if is_numeric_or_alphanumeric(original_text):
            translations[f"{column}_translated"] = original_text
            continue

        if not original_text:
            # If original text is empty, ensure translation column is also empty
            translations[f"{column}_translated"] = ""
            continue

        if language == "en":
            # If the language is English, no translation is needed
            translations[f"{column}_translated"] = original_text
            continue

        # Translation prompt
        message_text = f"""
        You are a professional language translator specializing in technical content. Your task is to translate the following text from {lang_name} to English with precision and clarity, adhering strictly to the following rules:

        1. Return **only** the translated text—no comments, explanations, or annotations.
        2. Ensure a **highly accurate** translation; do not introduce any fabricated or altered information.
        3. Clean the text by removing Unicode artifacts and special characters, but **do not add or alter punctuation**.
        4. If the input is empty or null, return an **empty string**.
        5. Preserve all numeric and alphanumeric strings (e.g., codes or identifiers) in the text phrase **exactly as they appear**.
        6. Maintain the **original technical meaning and context** without embellishment.

        Translate this input: '{original_text}'
        """

        for attempt in range(MAX_RETRIES):
            try:
                completion = llm.new_completion()
                completion.with_message(message_text)
                resp = completion.execute()

                if resp.success:
                    translations[f"{column}_translated"] = resp.text.strip()
                    break
                else:
                    logging.error(f"Translation failed for index {index}, column {column}, language {lang_name}. Response: {resp.text}")
                    if attempt == MAX_RETRIES - 1:
                        translations[f"{column}_translated"] = original_text
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    logging.error(f"Final error during translation for index {index}, column {column}: {e}")
                    translations[f"{column}_translated"] = original_text
                else:
                    logging.warning(f"Retrying translation for index {index}, column {column}, attempt {attempt + 1}: {e}")

        # Fallback to the original text if translation is unsuccessful
        if original_text and not translations.get(f"{column}_translated"):
            translations[f"{column}_translated"] = original_text

    return index, translations

def process_in_batches(df, llm, language_map, batch_size=100):
    """Process records in batches with parallelization."""
    results = {}
    new_records = df[df["status"] == "New"]

    if new_records.empty:
        logging.info("No new records to process.")
        return {}

    total_records = len(new_records)
    all_records = [(idx, row, row["language"]) for idx, row in new_records.iterrows()]

    for i in range(0, total_records, batch_size):
        batch = all_records[i: i + batch_size]
        logging.info(f"Processing batch {i // batch_size + 1}/{(total_records + batch_size - 1) // batch_size}")
        batch_start = datetime.datetime.now()

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_record = {executor.submit(translate_record, record, llm, language_map): record for record in batch}

            for future in tqdm(concurrent.futures.as_completed(future_to_record), total=len(batch), desc=f"Batch {i // batch_size + 1}"):
                try:
                    idx, translations = future.result()
                    results[idx] = translations
                except Exception as e:
                    record = future_to_record[future]
                    logging.error(f"Batch translation failed for record {record[0]}: {e}")

        logging.info(f"Batch {i // batch_size + 1} processed in {datetime.datetime.now() - batch_start}")

    return results

# --------------------------------------------------------------------------------
# Constants for chunking and processing
MAX_WORKERS = 4
MAX_RETRIES = 1

# Initialize LLM and language map
LLM_ID = "openai:Lite_llm_STS_Dev_GPT_4O:gpt-35-turbo-16k"
client = dataiku.api_client()
project = client.get_default_project()
llm = project.get_llm(LLM_ID)

language_map = {
    "en": "English",
    "fr": "French",
    "it": "Italian",
    "kk": "Kazakh",
    "ru": "Russian",
    "es": "Spanish",
    "sv": "Swedish",
}

# -------------------------------------------------------------------------------
# Clean translation columns
translation_columns = ["observation_final", "solution_final", "problem_cause_text"]
sts_cmb_df = clean_columns(sts_cmb_df, translation_columns)

# Add default values
sts_cmb_df["status"] = "New"

# Process translations in batches
start_time = datetime.datetime.now()
translation_results = process_in_batches(sts_cmb_df, llm, language_map)

# Update DataFrame with translations
for idx, translations in translation_results.items():
    for col, value in translations.items():
        sts_cmb_df.at[idx, col] = value

# Mark processed records
sts_cmb_df.loc[sts_cmb_df["status"] == "New", "status"] = "Processed"

# Log total processing time
end_time = datetime.datetime.now()
logging.info(f"Total processing time: {end_time - start_time}")

# Write translated records to output dataset
translated_df = sts_cmb_df[sts_cmb_df["status"] == "Processed"]
sts_cmb_trns = dataiku.Dataset("sts_cmb_trns")
sts_cmb_trns.write_with_schema(translated_df)
