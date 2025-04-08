# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
import pandas as pd
import datetime
import concurrent.futures
from tqdm.notebook import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Read recipe inputs
sts_cmb = dataiku.Dataset("sts_combined")
sts_cmb_df = sts_cmb.get_dataframe(infer_with_pandas=False)
sts_cmb_df['status'] = "New"

# Define the LLM ID
LLM_ID = "openai:Lite_llm_STS_Dev_GPT_4O:gpt-35-turbo-16k"

# Create a handle for the LLM
client = dataiku.api_client()
project = client.get_default_project()
llm = project.get_llm(LLM_ID)

# Dictionary of language codes to full names
language_map = {
    'en': 'English',
    'fr': 'French',
    'it': 'Italian',
    'kk': 'Kazakh',
    'ru': 'Russian',
    'es': 'Spanish',
    'sv': 'Swedish'
}

# Constants for chunking and processing
BATCH_SIZE = 100  # Process records in batches of 100
MAX_WORKERS = 4   # Number of parallel workers
MAX_RETRIES = 2   # Maximum number of retries for failed translations

# Language-specific prompts tailored for technical text
def get_translation_prompt(language, observation, solution):
    lang_name = language_map.get(language, "Unknown")
    if language == 'en':
        # Rephrase or clean English text
        return f"""
        You are an expert editor for technical troubleshooting documentation. Your task is to enhance the readability of the following English text without changing the meaning or modifying any technical terms.
        Observation: {observation}
        Solution: {solution}
        Provide the improved texts as follows:
        Observation: <improved_observation>
        Solution: <improved_solution>
        """
    elif language == 'fr':
        return f"""
        You are an expert translator for French technical troubleshooting text. Translate the following text into precise English while preserving technical terms and context:
        Observation: {observation}
        Solution: {solution}
        Do not simplify or modify technical terms.
        Provide the translations as follows:
        Observation: <translated_observation>
        Solution: <translated_solution>
        """
    elif language == 'it':
        return f"""
        You are an expert translator for Italian technical troubleshooting text. Translate the following text into precise English while preserving technical terms and context:
        Observation: {observation}
        Solution: {solution}
        Do not simplify or modify technical terms.
        Provide the translations as follows:
        Observation: <translated_observation>
        Solution: <translated_solution>
        """
    elif language == 'kk':
        return f"""
        You are an expert translator for Kazakh technical troubleshooting text. Translate the following text into precise English while preserving technical terms and context:
        Observation: {observation}
        Solution: {solution}
        Do not simplify or modify technical terms.
        Provide the translations as follows:
        Observation: <translated_observation>
        Solution: <translated_solution>
        """
    elif language == 'ru':
        return f"""
        You are an expert translator for Russian technical troubleshooting text. Translate the following text into precise English while preserving technical terms and context:
        Observation: {observation}
        Solution: {solution}
        Do not simplify or modify technical terms.
        Provide the translations as follows:
        Observation: <translated_observation>
        Solution: <translated_solution>
        """
    elif language == 'es':
        return f"""
        You are an expert translator for Spanish technical troubleshooting text. Translate the following text into precise English while preserving technical terms and context:
        Observation: {observation}
        Solution: {solution}
        Do not simplify or modify technical terms.
        Provide the translations as follows:
        Observation: <translated_observation>
        Solution: <translated_solution>
        """
    elif language == 'sv':
        return f"""
        You are an expert translator for Swedish technical troubleshooting text. Translate the following text into precise English while preserving technical terms and context:
        Observation: {observation}
        Solution: {solution}
        Do not simplify or modify technical terms.
        Provide the translations as follows:
        Observation: <translated_observation>
        Solution: <translated_solution>
        """
    else:
        # Generic fallback prompt
        return f"""
        Translate the following technical troubleshooting text into precise English:
        Observation: {observation}
        Solution: {solution}
        Preserve technical terms and context without simplification.
        Provide the translations as follows:
        Observation: <translated_observation>
        Solution: <translated_solution>
        """

# Function to process a single record
def translate_record(record_data):
    index, row, language = record_data
    lang_name = language_map.get(language, "Unknown")
    observation = row["observation"] if not pd.isna(row["observation"]) else ""
    solution = row["solution"] if not pd.isna(row["solution"]) else ""

    # Skip translation for English
    if language == 'en':
        return index, {
            "observation_translated": observation,
            "solution_translated": solution
        }

    # Prepare prompt
    message_text = get_translation_prompt(language, observation, solution)

    # Retry mechanism
    for attempt in range(MAX_RETRIES + 1):
        try:
            completion = llm.new_completion()
            completion.with_message(message_text)
            resp = completion.execute()

            if resp.success:
                # Extract translations from response
                translated_text = resp.text
                obs_translated, sol_translated = parse_translated_text(translated_text)
                return index, {
                    "observation_translated": obs_translated,
                    "solution_translated": sol_translated
                }
            else:
                logging.error(f"Translation failed for index {index}, language {lang_name}. Attempt {attempt+1}/{MAX_RETRIES}. Response: {resp.text}")

        except Exception as e:
            logging.error(f"Error during translation for index {index}, language {lang_name}, attempt {attempt+1}/{MAX_RETRIES}: {e}")

    # If all retries fail, keep the original values
    return index, {
        "observation_translated": observation,
        "solution_translated": solution
    }

# Function to parse translation response (assumes specific formatting)
def parse_translated_text(translated_text):
    try:
        obs_start = translated_text.find("Observation: ") + len("Observation: ")
        sol_start = translated_text.find("Solution: ")
        obs_translated = translated_text[obs_start:sol_start].strip()
        sol_translated = translated_text[sol_start + len("Solution: "):].strip()
        return obs_translated, sol_translated
    except Exception as e:
        logging.error(f"Error parsing translated text: {e}")
        return "", ""

# Function to process records in batches with parallelization
def process_in_batches(df, batch_size=BATCH_SIZE):
    results = {}
    new_records = df[df["status"] == "New"]
    total_records = len(new_records)

    # Create a list of all record data for processing
    all_records = [(idx, row, row['language']) for idx, row in new_records.iterrows()]

    # Process in batches
    for i in range(0, total_records, batch_size):
        batch = all_records[i:i+batch_size]
        logging.info(f"Processing batch {i//batch_size + 1}/{(total_records + batch_size - 1)//batch_size}")

        # Process batch with parallel workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_record = {executor.submit(translate_record, record): record for record in batch}

            # Show progress bar for current batch
            for future in tqdm(concurrent.futures.as_completed(future_to_record),
                              total=len(batch),
                              desc=f"Batch {i//batch_size + 1}"):
                idx, translations = future.result()
                results[idx] = translations

    return results

# Main execution
start = datetime.datetime.now()

# Process the data in batches with parallelization
translation_results = process_in_batches(sts_cmb_df)

# Update the main dataframe with translation results
for idx, translations in translation_results.items():
    for col, value in translations.items():
        sts_cmb_df.at[idx, col] = value

# Update status for processed records
sts_cmb_df.loc[sts_cmb_df["status"] == "New", "status"] = "Processed"

sts_cmb_trns_df = sts_cmb_df

for column in sts_cmb_trns_df.columns:
    if sts_cmb_trns_df[column].dtype == object:
        sts_cmb_trns_df[column] = sts_cmb_trns_df[column].fillna('')
    else:
        sts_cmb_trns_df[column] = sts_cmb_trns_df[column].fillna(0)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
sts_cmb_trns_out = dataiku.Dataset("sts_combined_translated")
sts_cmb_trns_out.write_with_schema(sts_cmb_trns_df)
