# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
import pandas as pd
import regex as re
import unicodedata
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# List of dataset names
dataset_names = [
    "LMRC", "sts_chile_ns16", "sts_dubai", "sts_222_emr", "sts_india",
    "sts_italy", "sts_itac_nantes", "sts_kz8a", "sts_kz4at", "sts_rem",
    "sts_panama", "sts_net2", "sts_spain", "sts_reg2n", "sts_tib",
    "sts_xtrapolis_chile", "sts_vline_rrsmc", "sts_u400_Lyon", "sts_u400"
]

# Load all datasets
logging.info("Loading Datasets... ")
dataframes = {name: dataiku.Dataset(name).get_dataframe() for name in dataset_names}
logging.info("Datasets Loaded! ✅")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Filter incomplete and invalid records
logging.info("Filtering incomplete and invalid records...")
invalid_patterns = [r"^$", r"^\s+$", r"####", r"#NAME?", r"-", r"^\d+$", r"^\s*$", r"^[!@#$%^&*(),.?\":{}|<>]+$", r"N/A - N/A"]

def filter_invalid_records(df, invalid_patterns):

    combined_pattern = "|".join(invalid_patterns)

    if "observation" in df.columns and "solution" in df.columns:
        df = df[
            ~df["observation"].astype(str).str.match(combined_pattern, na=False) &
            ~df["solution"].astype(str).str.match(combined_pattern, na=False)
        ]
    df = df.dropna(subset=["observation", "solution"], how="any")
    return df

dataframes = {name: filter_invalid_records(df, invalid_patterns) for name, df in dataframes.items()}
logging.info("Data cleaning complete! ✅")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Standardize metadata
logging.info("Standardizing metadata...")

def standardize_metadata(df, name):
    mode_values = {}
    for col in ["project", "database", "language"]:
        if col in df.columns and not df[col].dropna().empty:
            mode_values[col] = df[col].value_counts().idxmax()
        else:
            mode_values[col] = name
        df[col] = df[col].fillna(mode_values[col]) if col in df.columns else mode_values[col]
    return df

dataframes = {name: standardize_metadata(df, name) for name, df in dataframes.items()}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Standardize inconsistent values
logging.info("Standardizing inconsistent values...")

def standardize_values(df, name):
    if name == "LMRC" and "language" in df.columns:
        df["language"] = df["language"].replace({"ENGLISH": "English"})
    if name == "sts_222_emr" and "project" in df.columns:
        df["project"] = "222 - EMR"
    if name == "sts_xtrapolis_chile" and "project" in df.columns:
        df["project"] = df["project"].replace({"MERVAL": "Merval", "merval": "Merval"})
    if name == "sts_u400":
        if "language" in df.columns:
            df["language"] = df["language"].replace({"ENGLISH": "English", "SPANISH": "Spanish"})
    if "database" in df.columns:
        df = df[df["database"] != "STS_U400_6.0"]
    return df

dataframes = {name: standardize_values(df, name) for name, df in dataframes.items()}
logging.info("Standardization complete! ✅")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def clean_text(text):
    if pd.isna(text):
        return ""

    # Unicode normalization
    text = unicodedata.normalize('NFKC', str(text)).strip()

    # Fix spacing around punctuation (language-agnostic approach)
    text = re.sub(r'(?<=\p{L})\.(?=\p{L})', '. ', text)
    text = re.sub(r'(?<=\d)\s*\.\s*(?=\d)', '.', text)
    text = re.sub(r'\s+', ' ', text)

    # Keep all alphabetic characters from any language, plus common punctuation
    text = re.sub(r'[^\p{L}\p{N}\s\p{P}\p{S}]', '', text)

    return text.strip()

logging.info("Applying preprocessing...")
for name, df in dataframes.items():
    logging.info(f"Preprocessing {name}...")
    required_columns = ["observation", "problemcause", "solution", "problemcode"]

    # Check if all required columns are present
    if all(column in df.columns for column in required_columns):
        df[required_columns] = df[required_columns].map(clean_text)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Merge all dataframes into a single dataset
logging.info("Merging all dataframes into a single dataset...")
combined_df = pd.concat(dataframes.values(), ignore_index=True)
logging.info(f"Merging complete! ✅ Final dataset has {combined_df.shape[0]} rows and {combined_df.shape[1]} columns.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Standardize language values
logging.info("Standardizing language values...")
language_mapping = {
    "ENGLISH": "English",
    "RUS": "Russian",
    "kazakh": "Kazakh",
    "SWEDISH": "Swedish"
}
if "language" in combined_df.columns:
    combined_df["language"] = combined_df["language"].replace(language_mapping)

valid_languages = {
    'english': 'en',
    'french': 'fr',
    'italian': 'it',
    'kazakh': 'kk',
    'russian': 'ru',
    'spanish': 'es',
    'swedish': 'sv'
}

def standardize_language(lang):
    if pd.isna(lang):
        return 'unknown'
    lang = str(lang).strip().lower()
    if lang not in valid_languages:
        logging.warning(f"⚠️ Unknown language detected: {lang}")
    return valid_languages.get(lang, 'unknown')

if "language" in combined_df.columns:
    combined_df["language"] = combined_df["language"].apply(standardize_language)

logging.info("\n🔍 Unique values in 'language' after standardization:")
logging.info(combined_df["language"].unique())
logging.info(f"Total unique values in 'language': {combined_df['language'].nunique()}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Standardize database values
logging.info("Standardizing 'database' values...")
database_mapping = {"Rex": "REX"}
if "database" in combined_df.columns:
    combined_df["database"] = combined_df["database"].replace(database_mapping)

# Ensure metadata columns exist
logging.info("Ensuring metadata columns exist...")
METADATA_COLUMNS = [
    'project', 'fleet', 'subsystem', 'database', 'observationcategory',
    'problemcode', 'problemcause', 'solutioncategory', 'language',
    'failureclass', 'date'
]

for col in METADATA_COLUMNS:
    if col not in combined_df.columns:
        combined_df[col] = 'Unknown'

# Remove rows where 'observation' is an empty string
combined_df = combined_df[combined_df['observation'] != '']

# Remove rows where 'solution' is an empty string
combined_df = combined_df[combined_df['solution'] != '']

# Drop duplicate rows based on 'observation', 'solution', 'problemcause', and 'problemcode'
combined_df = combined_df.drop_duplicates(subset=['project', 'database', 'language', 'observation', 'solution', 'problemcause', 'problemcode', 'observationcategory', 'solutioncategory'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Rename columns
column_mapping = {
    "observationcategory": "observation_category",
    "problemcode": "problem_code",
    "problemcause": "problem_cause",
    "problemremedy": "problem_remedy",
    "functionallocation": "functional_location",
    "notificationsonumber": "notifications_number",
    "solutioncategory": "solution_category",
    "pbscode": "pbs_code",
    "symptomcode": "symptom_code",
    "rootcause": "root_cause",
    "documentlink": "document_link",
    "minresourcesneed": "min_resources_need",
    "maxresourceneed": "max_resource_need",
    "themostfrequentvalueforresource": "the_most_frequent_value_for_resource",
    "mintimeperoneperson": "min_time_per_one_person",
    "maxtimeperoneperson": "max_time_per_one_person",
    "averagetime": "average_time",
    "frequencyobs": "frequency_obs",
    "minresourcesneedsol": "min_resources_need_sol",
    "maxresourceneedsol": "max_resource_need_sol",
    "themostfrequentvalueforresourcesol": "the_most_frequent_value_for_resource_sol",
    "mintimeperonepersonsol": "min_time_per_one_person_sol",
    "maxtimeperonepersonsol": "max_time_per_one_person_sol",
    "averagetimesol": "average_time_sol",
    "failureclass": "failure_class"
}

# Rename the columns in the combined_df DataFrame
combined_df.rename(columns=column_mapping, inplace=True)

for column in combined_df.columns:
    if combined_df[column].dtype == object:
        combined_df[column] = combined_df[column].fillna('')
    else:
        combined_df[column] = combined_df[column].fillna(0)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Identify and convert problematic columns to strings
mixed_type_cols = [col for col in combined_df.columns if combined_df[col].map(type).nunique() > 1]
for col in mixed_type_cols:
    combined_df[col] = combined_df[col].astype(str)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Save the enhanced knowledge base
logging.info("Saving Processed knowledge base...")
output_file = 'sts_cmb'
output_dataset = dataiku.Dataset(output_file)
output_dataset.write_with_schema(combined_df)
logging.info(f"Processed knowledge base saved to '{output_file}'")
