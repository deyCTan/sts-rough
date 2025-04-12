# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
import logging
import unicodedata
import regex as re
import numpy as np
import pandas as pd

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of dataset names
dataset_names = [
    "LMRC", "sts_chile_ns16", "sts_dubai", "sts_222_emr", "sts_india",
    "sts_italy", "sts_itac_nantes", "sts_kz8a", "sts_kz4at", "sts_rem",
    "sts_panama", "sts_net2", "sts_spain", "sts_reg2n", "sts_tib",
    "sts_xtrapolis_chile", "sts_vline_rrsmc", "sts_u400_Lyon", "sts_u400"
]

# Load datasets into a dictionary of DataFrames
dataframes = {name: dataiku.Dataset(name).get_dataframe() for name in dataset_names}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Define patterns to identify absurd or meaningless text
INVALID_PATTERNS = [
    r"^\s*$",              # Empty strings
    r"^\d+$",              # Pure numbers
    r"^[!@#$%^&*(),.?\":{}|<>]+$",  # Only special characters
    r"^(.)\1{3,}$",        # Repeated characters (e.g., "aaaaaa")
    r"^(nan|na)$",         # Literal "nan" or "na" (case insensitive)
    r"^n/a\s*-\s*n/a$",    # Placeholder: N/A – N/A or n/a – n/a
    r"^\.\s*-\s*\.$",      # Placeholder: . - .
    r"####",               # Placeholder: ####
    r"#NAME\?",            # Placeholder: #NAME?
    r"^-+$"                # Placeholder: ---- or ---
]

# Replace invalid patterns with an empty string
def clean_column(dataframe, column_name):
    """
    Cleans a specific column of a DataFrame by replacing invalid patterns with an empty string.
    Ensures all values are strings before applying the regex cleaning.
    """
    if column_name in dataframe.columns:
        dataframe[column_name] = dataframe[column_name].astype(str).str.strip()
        for pattern in INVALID_PATTERNS:
            dataframe[column_name] = dataframe[column_name].replace(to_replace=pattern, value="", regex=True)
    return dataframe

# Apply cleaning to specified columns
def clean_specified_columns(df, columns_to_clean):
    for column in columns_to_clean:
        df = clean_column(df, column)
    return df

# Drop rows where 'observation' or 'solution' is empty
def drop_empty_observation_or_solution(df):
    if 'observation' in df.columns and 'solution' in df.columns:
        df = df[(df['observation'].str.strip() != "") & (df['solution'].str.strip() != "")]
    return df

# Columns to clean
columns_to_clean = ["observation", "solution", "observationcategory", "solutioncategory", "problemcause"]

# Process all datasets
for name, df in dataframes.items():
    df = clean_specified_columns(df, columns_to_clean)
    df = drop_empty_observation_or_solution(df)
    dataframes[name] = df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Standardize metadata
logger.info("Standardizing metadata...")

def standardize_metadata(df, name):
    mode_values = {}
    for col in ["project", "database", "language"]:
        mode_values[col] = df[col].value_counts().idxmax() if col in df.columns and not df[col].dropna().empty else name
        df[col] = df[col].fillna(mode_values[col]) if col in df.columns else mode_values[col]
    return df

dataframes = {name: standardize_metadata(df, name) for name, df in dataframes.items()}
logger.info("Metadata standardization complete! ✅")

# Standardize inconsistent values
logger.info("Standardizing inconsistent values...")

def standardize_values(df, name):
    if name == "LMRC" and "language" in df.columns:
        df["language"] = df["language"].replace({"ENGLISH": "English"})
    if name == "sts_222_emr" and "project" in df.columns:
        df["project"] = "222 - EMR"
    if name == "sts_xtrapolis_chile" and "project" in df.columns:
        df["project"] = df["project"].replace({"MERVAL": "Merval", "merval": "Merval"})
    if name == "sts_u400" and "language" in df.columns:
        df["language"] = df["language"].replace({"ENGLISH": "English", "SPANISH": "Spanish"})
    if "database" in df.columns:
        df = df[df["database"] != "STS_U400_6.0"]
    return df

dataframes = {name: standardize_values(df, name) for name, df in dataframes.items()}
logger.info("Standardization of values complete! ✅")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Merge all dataframes
logger.info("Merging all dataframes into a single dataset...")
combined_df = pd.concat(dataframes.values(), ignore_index=True)
logger.info(f"Merging complete! ✅ Final dataset has {combined_df.shape[0]} rows and {combined_df.shape[1]} columns.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Standardize language values
logger.info("Standardizing language values...")
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
        logger.warning(f"⚠️ Unknown language detected: {lang}")
    return valid_languages.get(lang, 'unknown')

if "language" in combined_df.columns:
    combined_df["language"] = combined_df["language"].apply(standardize_language)

# Log unique values and their count after standardization
unique_languages = combined_df["language"].unique()
logger.info(f"🔍 Unique values in 'language' after standardization: {unique_languages}")
logger.info(f"Total unique values in 'language': {len(unique_languages)}")

# Language-specific cleaning rules
LANGUAGE_CLEANERS = {
    'fr': lambda text: re.sub(r'[’]', "'", text).replace('é', 'e').replace('è', 'e').replace('ê', 'e').replace('ë', 'e')
                  .replace('à', 'a').replace('â', 'a').replace('ô', 'o'),
    'es': lambda text: re.sub(r'[áéíóúñ]', lambda m: m.group(0).replace('á', 'a').replace('é', 'e')
                                                    .replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
                                                    .replace('ñ', 'n'), text),
    'ru': lambda text: re.sub(r'[^\p{Cyrillic}\p{Latin}\p{P}\p{N}\s@#$%&/\\=_\-+~°±№«»“”\'"…<>†™®©€₸₽]', '', text),
    'kk': lambda text: re.sub(r'[^\p{Cyrillic}\p{Latin}\p{P}\p{N}\s@#$%&/\\=_\-+~°±№«»“”\'"…<>†™®©€₸₽]', '', text),
    'it': lambda text: re.sub(r'[àèéìòù]', lambda m: m.group(0).replace('à', 'a').replace('è', 'e').replace('é', 'e')
                                                    .replace('ì', 'i').replace('ò', 'o').replace('ù', 'u'), text),
    'sv': lambda text: re.sub(r'[åäö]', lambda m: m.group(0).lower().replace('å', 'a').replace('ä', 'a')
                                                               .replace('ö', 'o'), text),
}

# Main text cleaning function
def clean_text(text, lang='generic'):
    try:
        if pd.isna(text): return ""
        text = unicodedata.normalize('NFC', str(text)).strip()
        if lang in LANGUAGE_CLEANERS: text = LANGUAGE_CLEANERS[lang](text)
        text = re.sub(r'\s+', ' ', re.sub(r'[^\p{L}\p{N}\s\p{P}]', '', text).strip())
        return text
    except Exception as e:
        logger.error(f"Error cleaning text for language '{lang}': {e}. Input text: {text}")
        return text

# Apply language-specific cleaning to relevant columns
if "language" in combined_df.columns:
    combined_df["language"] = combined_df["language"].fillna('unknown')
    columns_to_clean = ["observationcategory", "observation", "problemcause", "solutioncategory", "solution"]

    for column in columns_to_clean:
        if column in combined_df.columns:
            combined_df[column] = combined_df.apply(
                lambda row: clean_text(row[column], row["language"]), axis=1
            )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Standardize database values
logger.info("Standardizing 'database' values...")
database_mapping = {"Rex": "REX"}
if "database" in combined_df.columns:
    combined_df["database"] = combined_df["database"].replace(database_mapping)

# Ensure metadata columns exist
logger.info("Ensuring metadata columns exist...")
METADATA_COLUMNS = [
    'project', 'fleet', 'subsystem', 'database', 'observationcategory',
    'problemcode', 'problemcause', 'solutioncategory', 'language',
    'failureclass', 'date'
]

for col in METADATA_COLUMNS:
    if col not in combined_df.columns:
        combined_df[col] = ""

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
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
    'frequencysol': 'frequency_sol',
    "failureclass": "failure_class"
}

# Rename the columns in the combined_df DataFrame
combined_df.rename(columns=column_mapping, inplace=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Add new columns with default empty string values
new_columns = ['category_id', 'obs_id', 'sol_category_id']
for col in new_columns:
    combined_df[col] = ""

# Detect columns with mixed data types and convert them to strings
mixed_type_columns = [
    col for col in combined_df.columns if combined_df[col].map(type).nunique() > 1
]
for col in mixed_type_columns:
    combined_df[col] = combined_df[col].astype(str)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def replace_nan_with_empty_string(df):
    return df.replace(["NaN", np.nan, pd.NA], "", regex=False)

combined_df = replace_nan_with_empty_string(combined_df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
try:
    # Define project configurations with exact case-sensitive names
    project_configs = {
        "LMRC": {
            "textual_columns": ["problem_cause"],
            "coded_columns": ["observation_category"]
        },
        "222 - EMR": {
            "textual_columns": ["observation_category"],
            "coded_columns": []
        },
        "NS16": {
            "textual_columns": [],
            "coded_columns": []
        },
        "Dubai": {
            "textual_columns": ["observation_category", "problem_cause"],
            "coded_columns": []
        },
        "IND_E_Loco": {
            "textual_columns": ["problem_cause"],
            "coded_columns": []
        },
        "iTAC-Nantes": {
            "textual_columns": ["problem_cause"],
            "coded_columns": []
        },
        "Italy": {
            "textual_columns": ["observation_category"],
            "coded_columns": []
        },
        "KZ4AT": {
            "textual_columns": [],
            "coded_columns": ["observation_category"]
        },
        "NET2": {
            "textual_columns": ["observation_category", "problem_cause"],
            "coded_columns": []
        },
        "KZ8A": {
            "textual_columns": [],
            "coded_columns": ["problem_cause"]
        },
        "Panama": {
            "textual_columns": [],
            "coded_columns": []
        },
        "REG2N": {
            "textual_columns": ["observation_category", "problem_cause"],
            "coded_columns": []
        },
        "REM": {
            "textual_columns": ["observation_category"],
            "coded_columns": ["problem_cause"]
        },
        "U400 - Lyon": {
            "textual_columns": ["observation_category"],
            "coded_columns": []
        },
        "VLINE RRSMC": {
            "textual_columns": ["observation_category", "problem_cause"],
            "coded_columns": []
        },
        "TIB": {
            "textual_columns": ["observation_category"],
            "coded_columns": []
        },
        "Spain": {
            "textual_columns": ["observation_category"],
            "coded_columns": []
        },
        "Merval": {
            "textual_columns": ["problem_cause"],
            "coded_columns": []
        },
        "U400": {
            "textual_columns": ["observation_category"],
            "coded_columns": []
        }
    }

    logger.info(f"Created project configuration mapping for {len(project_configs)} projects")

    # Apply transformations to the actual dataset
    logger.info("Starting to process the dataset rows")
    result_frames = []

    # Process each project's data separately
    for project_name, project_df in combined_df.groupby("project"):
        if project_name in project_configs:
            logger.info(f"Processing {len(project_df)} rows for project {project_name}")

            # Get project configuration
            config = project_configs[project_name]

            # Initialize new columns with empty strings
            project_df["observation_category_text"] = ""
            project_df["observation_category_code"] = ""
            project_df["problem_cause_text"] = ""
            project_df["problem_cause_code"] = ""

            # Handle `observation_category`
            if "observation_category" in config["textual_columns"]:
                project_df["observation_category_text"] = project_df["observation_category"]
            if "observation_category" in config["coded_columns"]:
                project_df["observation_category_code"] = project_df["observation_category"]

            # Handle `problem_cause`
            if "problem_cause" in config["textual_columns"]:
                project_df["problem_cause_text"] = project_df["problem_cause"]
            if "problem_cause" in config["coded_columns"]:
                project_df["problem_cause_code"] = project_df["problem_cause"]

            result_frames.append(project_df)
        else:
            logger.warning(f"No configuration found for project {project_name}")
            project_df["observation_category_text"] = ""
            project_df["observation_category_code"] = ""
            project_df["problem_cause_text"] = ""
            project_df["problem_cause_code"] = ""
            result_frames.append(project_df)

    # Combine all processed data
    sts_cmb_final_df = pd.concat(result_frames, ignore_index=True)
    logger.info(f"Created final DataFrame with {len(sts_cmb_final_df)} rows")
    logger.info("Data processing pipeline completed successfully")

except Exception as e:
    logger.error(f"Error in data processing pipeline: {str(e)}", exc_info=True)
    raise

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Save the enhanced knowledge base
logging.info("Saving Processed knowledge base...")
output_file = 'sts_cmb'
output_dataset = dataiku.Dataset(output_file)
output_dataset.write_with_schema(sts_cmb_final_df)
logging.info(f"Processed knowledge base saved to '{output_file}'")
