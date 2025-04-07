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
sts_cmb = dataiku.Dataset("sts_cmb")
sts_cmb_df = sts_cmb.get_dataframe(infer_with_pandas=False)
sts_cmb_df['status'] = "New"

# Sample function - comment out for full processing
# def sample_projects(df, project_column='project', n_samples=10):
#     return df.groupby(project_column).apply(lambda x: x.sample(n=n_samples)).reset_index(drop=True)
# sts_cmb_df = sample_projects(sts_cmb_df, project_column='project', n_samples=20)

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

# Function to translate a single record
def translate_record(record_data):
    index, row, language = record_data
    translations = {}
    lang_name = language_map.get(language, "Unknown")

    for column in ["observation", "problem_cause", "problem_code", "solution"]:
        # Skip empty fields for problem code and cause
        if column in ["problem_code", "problem_cause"] and pd.isna(row[column]):
            translations[f"{column}_translated"] = ''
            continue

        # Skip translation for English
        if language == 'en':
            translations[f"{column}_translated"] = row[column]
            continue

        message_text = f"""
        You are an expert language translator. Your task is to precisely translate the given text from {lang_name} to English. Please adhere to the following guidelines:
        1. Deliver only the translation without any additional commentary or explanations. Do not preamble.
        2. Ensure the translation is accurate and avoid generating any false or fabricated information. Clean the data by removing any Unicode and special characters.
        3. If the input text is empty, respond with an empty string.
        4. Do not add any punctuation.
        5. Ensure that the translated text maintains the meaning and context of the original text.

        Translate the following text: '{row[column]}'
        """

        try:
            completion = llm.new_completion()
            completion.with_message(message_text)
            resp = completion.execute()

            if resp.success:
                translations[f"{column}_translated"] = resp.text
            else:
                logging.error(f"Translation failed for index {index}, column {column}, language {lang_name}. Response: {resp.text}")
                translations[f"{column}_translated"] = row[column]  # Keep original on failure
        except Exception as e:
            logging.error(f"Error during translation for index {index}, column {column}: {e}")
            translations[f"{column}_translated"] = row[column]  # Keep original on failure

    return index, translations

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

end = datetime.datetime.now()
logging.info(f"Total processing time: {end - start}")

# Write recipe outputs
sts_cmb_trns = dataiku.Dataset("sts_cmb_trns")
sts_cmb_trns.write_with_schema(sts_cmb_df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
import pandas as pd
import datetime
import concurrent.futures
from tqdm.notebook import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# Function to translate a single record
def translate_record(record_data):
    index, row, language = record_data
    translations = {}
    lang_name = language_map.get(language, "Unknown")

    for column in ["observation", "problem_cause", "problem_code", "solution"]:
        # Skip empty fields for problem code and cause
        if column in ["problem_code", "problem_cause"] and pd.isna(row[column]):
            translations[f"{column}_translated"] = ''
            continue

        # Skip translation for English
        if language == 'en':
            translations[f"{column}_translated"] = row[column]
            continue

        message_text = f"""
        You are an expert language translator. Your task is to precisely translate the given text from {lang_name} to English. Please adhere to the following guidelines:
        1. Deliver only the translation without any additional commentary or explanations. Do not preamble.
        2. Ensure the translation is accurate and avoid generating any false or fabricated information. Clean the data by removing any Unicode and special characters.
        3. If the input text is empty, respond with an empty string.
        4. Do not add any punctuation.
        5. Ensure that the translated text maintains the meaning and context of the original text.

        Translate the following text: '{row[column]}'
        """

        try:
            completion = llm.new_completion()
            completion.with_message(message_text)
            resp = completion.execute()

            if resp.success:
                translations[f"{column}_translated"] = resp.text
            else:
                logging.error(f"Translation failed for index {index}, column {column}, language {lang_name}. Response: {resp.text}")
                translations[f"{column}_translated"] = row[column]  # Keep original on failure
        except Exception as e:
            logging.error(f"Error during translation for index {index}, column {column}: {e}")
            translations[f"{column}_translated"] = row[column]  # Keep original on failure

    return index, translations

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

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
sts_cmb_trns = dataiku.Dataset("sts_cmb_trns")
sts_cmb_trns_df = sts_cmb_trns.get_dataframe(infer_with_pandas=False)

for column in sts_cmb_trns_df.columns:
    if sts_cmb_trns_df[column].dtype == object:
        sts_cmb_trns_df[column] = sts_cmb_trns_df[column].fillna('')
    else:
        sts_cmb_trns_df[column] = sts_cmb_trns_df[column].fillna(0)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Define the columns to check for translation failures
columns_to_check = ["observation", "solution"]

def identify_failed_rows(df):
    # Filter for non-English rows
    non_english_rows = df[df['language'] != 'en']

    # Initialize a list to store indices of failed rows
    failed_indices = []

    # Check for matching original and translated values
    for idx, row in non_english_rows.iterrows():
        failed = False
        for col in columns_to_check:
            original_value = row[col]
            translated_value = row[f"{col}_translated"]

            # Skip checks if the original value is NaN
            if pd.isna(original_value):
                continue

            # Check if the translated value matches the original value
            if str(original_value).strip() == str(translated_value).strip():
                failed = True
                break  # No need to check further columns for this row

        if failed:
            failed_indices.append(idx)

    return failed_indices

# Identify failed rows
failed_indices = identify_failed_rows(sts_cmb_trns_df)

# Separate failed rows from the main dataset
failed_rows_df = sts_cmb_trns_df.loc[failed_indices].copy()
successful_rows_df = sts_cmb_trns_df.loc[~sts_cmb_trns_df.index.isin(failed_indices)].copy()

# Print summary
print(f"Total failed rows identified: {len(failed_rows_df)}")
print(f"Total successful rows: {len(successful_rows_df)}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
failed_rows_df['status'] = "New"

retry_results = process_in_batches(failed_rows_df, batch_size=100)

# Update the main DataFrame with retry results
for idx, translations in retry_results.items():
    for col, value in translations.items():
        sts_cmb_trns_df.at[idx, col] = value

# Mark retried rows as "Processed"
sts_cmb_trns_df.loc[failed_rows_df.index, 'status'] = "Processed"

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Identify remaining failed rows after retry
remaining_failed_indices = identify_failed_rows(sts_cmb_trns_df)

# Separate remaining failed rows
remaining_failed_rows_df = sts_cmb_trns_df.loc[remaining_failed_indices].copy()

# Log summary
print(f"Remaining failed rows after retry: {len(remaining_failed_rows_df)}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
sts_cmb_trns_out = dataiku.Dataset("sts_cmb_trns_final")
sts_cmb_trns_out.write_with_schema(sts_cmb_trns_df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
sts_cmb_trns_final = dataiku.Dataset("sts_cmb_trns_final")
translated_df = sts_cmb_trns_final.get_dataframe(infer_with_pandas=False)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def replace_empty_translations(df, original_col, translated_col):
    df[translated_col] = df.apply(
        lambda row: row[original_col] if pd.isna(row[translated_col]) or str(row[translated_col]).strip() == '' else row[translated_col],
        axis=1
    )

replace_empty_translations(translated_df, 'problem_cause', 'problem_cause_translated')
replace_empty_translations(translated_df, 'problem_code', 'problem_code_translated')
replace_empty_translations(translated_df, 'solution', 'solution_translated')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Original and translated columns
original_columns = ["observation"]
translated_columns = [f"{col}_translated" for col in original_columns]

# Identify incomplete translations
def is_blank(value):
    return pd.isna(value) or str(value).strip() == ''

conditions = [
    (translated_df[orig_col].notna() & translated_df[orig_col].apply(lambda x: not is_blank(x)) & translated_df[trans_col].apply(is_blank))
    for orig_col, trans_col in zip(original_columns, translated_columns)
]

combined_condition = conditions[0]
for condition in conditions[1:]:
    combined_condition |= condition

# Create two DataFrames
df1 = translated_df[~combined_condition]  # Complete translations
df2 = translated_df[combined_condition]   # Incomplete translations
df2.shape

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
translations = {
    "Esecuzione Test RTB come da lettera TRNIT-DT##": "Execution of RTB Test as per letter TRNIT-DT##",
    "порвано суфле 2тэд": "Torn soufflé 2TED",
    "ов не работает маш. цыганин": "OV does not work machine. gypsy",
    "выдавливание смазки буксы 3 кп справа": "Squeezing grease from the axle box 3 KP on the right",
    "сменит тормозной рукав каб1,бьет автомат эпт.": "Change the brake hose CAB1, hits the EPT automatic",
    "порван суфле 4тэд": "Torn soufflé 4TED",
    "выдавливания смазки 3 кп справа": "Squeezing grease from the axle box 3 KP on the right",
    "Veuillez procéder au graissage du coupleur automatique": "Please proceed with greasing the automatic coupler",
    "Boitier d'admission d'air a été remplacé suite a un écrou autobloquant qui était bloqué de l'interieur en devissant la vis sans es= t endommagé on a cannibalisé un boitier d'admission air du DMC2 au train T001 pour le poser au T021 du DMC1": "The air intake box was replaced following a self-locking nut that was stuck inside while unscrewing the screw. We cannibalized an air intake box from DMC2 on train T001 to fit it on T021 of DMC1",
    "49 Remettre en conformité le montant a l'intérieur du gangway 49 Put on conformity the vertical panel inside the gangway see picture 29 04 2024 06 49 05 Patricia Ashley (~468186) Brackets seat out to far d’ajuster in as far as possible un able to mate anymore d’ajustement": "49 Put the vertical panel inside the gangway back in conformity, see picture 29 04 2024 06 49 05 Patricia Ashley (~468186) Brackets seat out too far, adjust in as far as possible, unable to mate anymore",
    "Procédez a la lubrification des coupleurs": "Proceed with the lubrication of the couplers",
    "Software update": "Software update",
    "fff": "fff",
    "Train US18 mort au remisage": "Train US18 dead in storage",
    "los ast 12a/b estan mojados": "The AST 12A/B are wet",
    "cop enchufe butaca 14a r3 esta\x81 suelto": "COP seat plug 14A R3 is loose",
    "cop r1 asiento 4a enchufe suelto": "COP R1 seat 4A plug is loose",
    "enchufe 3a suelto": "Plug 3A loose",
    "enchufe butaca ast 1a suelto": "AST seat plug 1A loose",
    "enchufe plaza 4a suelto": "Plaza plug 4A loose",
    "r1 enchufe butaca 8a suelto": "R1 seat plug 8A loose",
    "persiana 8d 9d inutil": "Blinds 8D 9D useless",
    "puesta a 0 bombas wc": "WC pumps reset",
    "bomba bano izdo para cambiar y fuga en": "Left bathroom pump to change and leak in",
    "cop wcs r1 y r3 no funcionull": "COP WCs R1 and R3 not working",
    "cop wc del c3 averiado": "COP WC of C3 broken",
    "cop wcs r1 y r2 fuera de servicio": "COP WCs R1 and R2 out of service",
    "cop wcs r3 y r8 sin agua": "COP WCs R3 and R8 without water",
    "wcs r7 condenados por no tragar el": "WCs R7 condemned for not flushing",
    "wcs r1 y r2 llenos": "WCs R1 and R2 full",
    "cop c 2 ast3a(detra\x81s) enchufe suelto": "COP C 2 AST3A (behind) plug loose",
    "fuga en lavamanos del coche 7": "Leak in the washbasin of car 7",
    "cop wcs r1 y r8 llenos wc r2 atascado": "COP WCs R1 and R8 full, WC R2 clogged",
    "cop wcs r1 r3 r5 r6 r8 atascados": "COP WCs R1 R3 R5 R6 R8 clogged",
    "cop dos wc inutiles c 5 lado c d y": "COP two useless WCs C 5 side C D and",
    "cop wcs r7 y r8 sin agua": "COP WCs R7 and R8 without water",
    "wcs de todo el tren apestan": "WCs of the whole train stink",
    "enchufe suelto wc c5": "WC C5 plug loose",
    "wcs r2 y r3 llenos": "WCs R2 and R3 full",
    "wcs r1 r2 r3 y r7 llenos": "WCs R1 R2 R3 and R7 full",
    "utr >o03": "UTR >O03",
    "wcs r6 no sale agua": "WCs R6 no water",
    "no actua secamanos wc minusvalidos": "Disabled WC hand dryer not working",
    "golpe tranvia con coche": "Tram collision with car",
    "shunt": "Shunt",
    "MA URG CP HS": "MA URG CP HS",
    "MANQUE OEILLET DE LEVAGE MASU": "MISSING LIFTING EYELET MASU",
    "MCM 5 isolé au démarrage de la rame": "MCM 5 isolated at the start of the train",
    "Reprendre porte 7 vis 1 et 3 sur bras d'entrainements supérieurs.": "Resume door 7 screws 1 and 3 on upper drive arms",
    "A reprendre porte 6 vis 1 sur bras d'entrainements supérieurs": "Resume door 6 screws 1 on upper drive arms",
    "6 lisseuses HS": "6 smoothing machines HS",
    "MCM1 isolé au démarrage de la rame": "MCM1 isolated at the start of the train",
    "BRUIT ECHAPPEMENT D AIR EN ROULANT PAR INTERMITTENCE TOUTES LES 30SEC ENVIRON": "AIR EXHAUST NOISE WHILE DRIVING INTERMITTENTLY EVERY 30SEC APPROXIMATELY",
    "Caméra vidéo surveillance porte 15 HS": "Video surveillance camera door 15 HS",
    "Cannibalisation rack BCU V11": "Cannibalization rack BCU V11",
    "RAck Video surveillance V11 HS": "Rack Video surveillance V11 HS",
    "wc hs fuite dans cuvette": "WC HS leak in bowl",
    "suite avm butée fenetre nc": "Following AVM window stop NC",
    "Caméra 4 vidéosurveilance en V15 HS": "Camera 4 video surveillance in V15 HS",
    "Porte intersalle V15 Ext2 claque en fin de course": "Inter-room door V15 Ext2 slams at the end of the stroke",
    "Porte de salle V11 Courroie détendue": "Room door V11 Belt relaxed",
    "Fuite d’eau dans le WC PMR": "Water leak in the PMR WC",
    "Porte intersalle V13 Ext 2 claque lors de l'ouverture (butée)": "Inter-room door V13 Ext 2 slams during opening (stop)",
    "porte intersalle V19 ext 2 courroie déssérée": "Inter-room door V19 ext 2 belt loosened",
    "044R_V11_Attelage_SAV BT:Fuite après coupe": "044R_V11_Attelage_SAV BT: Leak after cut",
    "V17 Ext 1 Porte intersalle Courroie fortement détendue": "V17 Ext 1 Inter-room door Belt strongly relaxed",
    "Fuite intérieure WC V15 Esclave": "Interior leak WC V15 Slave",
    "Reprise fixation porte et poignée toilette maitre/esclave": "Resume door and handle fixing master/slave toilet",
    "V17 LSS Maitre poignée désemparée": "V17 LSS Master handle dismayed",
    "réglage porte WC LSS V17 esclave": "WC door adjustment LSS V17 slave",
    "Pas de condamnation de la porte WC LSS Esclave V17": "No condemnation of the WC door LSS Slave V17",
    "Purge Bogie grippée": "Seized bogie purge",
    "DMB SYNAS EFTER RÅDJURSPÅKÖRNING": "DMB inspection after deer collision",
    "DMB RÅDJUR PÅKÖRT. SANERINGSBEHOV.": "DMB deer hit. Cleaning needed.",
    "DMB SYNING PÅKÖRT DJUR SOM TOG PÅ VÄNSTER SIDA": "DMB inspection hit animal that took on the left side",
    "DMA LÖS SLANG I BOGGI 2": "DMA loose hose in bogie 2",
    "AXEL 1 OCH 2 DMB,BÖRJAN TILL MATERIALSLÄPP KLASS 1,HÅLLES UNDER-HÅLLES UNDER UPPSIKT! SVARV DMA DMB 8 ST AXLAR HGL 9 /11": "AXLE 1 AND 2 DMB, BEGINNING TO MATERIAL RELEASE CLASS 1, MONITORED! TURNING DMA DMB 8 AXLES HGL 9 /11",
    "TUGGUMMI-LITTERA DMA - SÄTE: 52 - DMA PLATS 52 NÄTFICKAN HAR KLADD AV TUGGUMMI PÅ SIG": "CHEWING GUM-LITTERA DMA - SEAT: 52 - DMA PLACE 52 NET POCKET HAS CHEWING GUM ON IT",
    "säten nr 37-43 avstängda. Blodfläckar och kiss.-Hysterisk resenär levde rövare": "seats no. 37-43 closed. Blood stains and pee.-Hysterical passenger caused trouble",
    "DMB SVANHALSMIKROFON GER INGET LJUD.": "DMB SWAN NECK MICROPHONE GIVES NO SOUND.",
    "DMA: SKADAD BUSSNING HUNDBEN BG1": "DMA: DAMAGED BUSHING DOG LEG BG1",
    "Uppstäld brandgavel dörr-Ställde upp brandgavel dörr eftersom den inte öppnade. Verkar som att sensorerna slutat att fungera. SENSORSLUT I LAGER": "Propped fire wall door-Proped fire wall door because it did not open. Seems like the sensors stopped working. SENSORS OUT OF STOCK",
    "HJÄRTSTARTARE SLUT PÅ BATTERI, SLUTAT LARMA/PIPA": "DEFIBRILLATOR OUT OF BATTERY, STOPPED ALARMING/BEEPING",
    "Älgkollision, blodstänk, skakig i DmB, lös vajer-Kraftig träff. Fordonet täckt i blod vänster sida DMB. Lös vajer funnen axel 3. Upplevs lite skakigt i vagnen DMB i hastigheter över 90 km/h.": "Moose collision, blood splatter, shaky in DmB, loose wire-Strong hit. Vehicle covered in blood on the left side DMB. Loose wire found axle 3. Feels a bit shaky in the DMB carriage at speeds over 90 km/h.",
    "klottrat i Ljusdal-Fordonet blev klottrat i Ljusdal natten mellan 13-14/7. Hela T0 och halva DMB.": "Graffitied in Ljusdal-The vehicle was graffitied in Ljusdal the night between 13-14/7. The whole T0 and half DMB.",
    "DMB SMUTSIGA/SLITNA STOLSTYGER PLATS 24,29 & 43": "DMB DIRTY/WORN SEAT FABRICS PLACE 24,29 & 43",
    "toa avstängd-Med säkringen i k14. Drog luft": "Toilet closed-With the fuse in k14. Drew air",
    "DMA PLÅTAR UNDERREDE KONTROLL, BOTTENPLÅT LAGAD MED SIKA.": "DMA PLATES UNDERCARRIAGE CONTROL, BOTTOM PLATE REPAIRED WITH SIKA.",
    "5 DMB HALKSKYDDSTEJP SAKNAS PÅ FLERA STÄLLEN-DMB TAK / KORRUGERAD PLÅT/ KÅPOR ANMÄRKNING RAPPORTERAD IGEN PGA EJ GODKÄNT ÅTGÄRDANDE. ANMÄRKNINGENS LÖPNUMMER I MSKS:22085982": "5 DMB ANTI-SLIP TAPE MISSING IN SEVERAL PLACES-DMB ROOF / CORRUGATED SHEET METAL / COVERS REMARK REPORTED AGAIN DUE TO NOT APPROVED MEASURES. REMARK NUMBER IN MSKS:22085982",
    "SKADADE KLACKAR FÖR UPPHÄNGNING AV EFTERSITSBORD. 2PLATSER": "DAMAGED HOOKS FOR HANGING AFTER-SEAT TABLES. 2 PLACES",
    "WO:2186908": "WO:2186908",
    "HALKSKYDDSTEJP SAKNAS VID STRÖMAVTAGARE (PÅBÖRJAT)-PÅBÖRJAD WO: 2178427": "ANTI-SLIP TAPE MISSING AT PANTOGRAPH (STARTED)-STARTED WO: 2178427",
    "SAKNAS HALKSKYDDSTEJP VID STRÖMAVTAGARE (PÅBÖRJAT)": "MISSING ANTI-SLIP TAPE AT PANTOGRAPH (STARTED)",
    "DMB - KLOTTER CA 35 KVM HÖGER SIDA": "DMB - GRAFFITI APPROX. 35 SQM RIGHT SIDE",
    "DMB:BOGGI1 LÄNKARMAR DEFEKTA BUSSNINGAR": "DMB:BOGIE 1 LINK ARMS DEFECTIVE BUSHINGS",
    "DMB BOGGI 1 SLAG I HJUL-FÖR TUNNA FÖR SVARV BYTE PLANERAS IN": "DMB BOGIE 1 HIT IN WHEEL-TOO THIN FOR TURNING, REPLACEMENT PLANNED",
    "VAGNSKORG. PÅKÖRD MÄNNISKA. OKLART VILKEN ÄNDE SOM VAR LEDANDE VI-HÄNDELSEDATUM: 2023/03/29 15:48TÅGNUMMER: 8533SAMMANFATTNING: PÅKÖRD MÄNNISKA. OKLART VILKEN ÄNDE SOM VAR LEDANDE VID TILLFÄLLET. SANERAD, KOPPELKÅPA TRASIG.": "CAR BODY. HIT PERSON. UNCLEAR WHICH END WAS LEADING WE-EVENT DATE: 2023/03/29 15:48TRAIN NUMBER: 8533SUMMARY: HIT PERSON. UNCLEAR WHICH END WAS LEADING AT THE TIME. SANITIZED, COUPLER HOOD DAMAGED.",
    "DMB LUFTLÄCKA I K4 SKÅPET, KRAN TILL ELHUVUD AVSTÄNGD": "DMB AIR LEAK IN K4 CABINET, CRANE TO ELECTRICAL HEAD TURNED OFF",
    "DMB. KOPPELKÅPA SKADAD EFTER VILTPÅKÖRNING.": "DMB. Coupler hood damaged after wildlife collision.",
    "HANDIKAPPLIFT UR FUNKTION-LARMAR HELATIDEN, LÄRORESA FUNKADE INTE": "Handicap lift out of order - alarms constantly, training trip did not work",
    "SKUMSLÄCKARE TÖMD I VAGNEN": "Foam extinguisher emptied in the carriage",
    "Älgkrock-Damask skadad vä sida i färdriktning. Älgkrock.": "Moose collision - Gaiter damaged on the left side in the direction of travel. Moose collision.",
    "DMB K4 KRAN ELHUVUD IN AVSTÄNGD, LÄCKER LUFT": "DMB K4 crane electric head in off, leaking air",
    "DMB KLOTTER HÖGER SIDA CA 15KVM": "DMB graffiti right side approx. 15 sqm",
    "123 DMA INVÄNDIGT TOALETTVÄGG, TEJPRESTER. LU": "123 DMA interior toilet wall, tape residue. LU",
    "109 DMA INVÄNDIGT TEJPRESTER I ÖVERKANT PÅ FÖNSTERRAM VID PLATS-67 LU": "109 DMA interior tape residue at the top of the window frame at seat 67. LU",
    "DMB K8 KLOTTERTEJP SAKNAS": "DMB K8 graffiti tape missing",
    "159 DMB INVÄNDIGT TVK, ROSTFRITT FÄLT OVAN FÖNSTER, TEJPRESTER.-LU": "159 DMB interior TVK, stainless steel field above the window, tape residue.-LU",
    "142 DMB FLERTALET SMUTSIGA/SLITNA STOLSTYGER-DMB INVÄNDIGT/KUPE/VESTIBUL KONTROLL AV STOLAR FUNKTION/HELHET/INFÄSTNING, ANSVAR LU": "142 DMB multiple dirty/worn seat fabrics - DMB interior/coupe/vestibule seat function/whole/attachment check, responsibility LU",
    "25 DMB UNDERREDE AXEL 2 STATUS, HÖ, JORDKABEL, KARDELER AV. LU": "25 DMB undercarriage axle 2 status, right, ground cable, strands off. LU",
    "TOAAV, HANDIKAPPTOALETT SPOLADE EJ, EFTER RESET DRAR DEN LUFT HE-LA TIDEN.": "TOAAV, handicap toilet did not flush, after reset it draws air all the time.",
    "DMB BOGGI A SLAG I HJUL": "DMB bogie A hit in the wheel",
    "Señalamiento: Vand atropello persona. Causa: vandalismo. Solución: Revisión de tren bajo bastidor con presencia de restos humanos": "Signal: Vandal hit person. Cause: vandalism. Solution: Train inspection under frame with presence of human remains",
    "Train 1117 jump station": "Train 1117 skipped station"
}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df2['observation_translated'] = df2['observation'].map(translations).fillna(df2['observation_translated'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
translated_df = pd.concat([df1, df2], ignore_index=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Columns to check
original_columns = ["observation"]
translated_columns = [f"{col}_translated" for col in original_columns]

# Function to check if a column is empty or contains only whitespace
def is_blank(value):
    return pd.isna(value) or str(value).strip() == ''

conditions = [
    (translated_df[orig_col].notna() & translated_df[orig_col].apply(lambda x: not is_blank(x)) & translated_df[trans_col].apply(is_blank))
    for orig_col, trans_col in zip(original_columns, translated_columns)
]

combined_condition = conditions[0]
for condition in conditions[1:]:
    combined_condition |= condition

incomplete_translations_df = translated_df[combined_condition]
incomplete_translations_df.shape

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
replace_empty_translations(translated_df, 'observation', 'observation_translated')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Columns to check
original_columns = ["observation"]
translated_columns = [f"{col}_translated" for col in original_columns]

# Function to check if a column is empty or contains only whitespace
def is_blank(value):
    return pd.isna(value) or str(value).strip() == ''

conditions = [
    (translated_df[orig_col].notna() & translated_df[orig_col].apply(lambda x: not is_blank(x)) & translated_df[trans_col].apply(is_blank))
    for orig_col, trans_col in zip(original_columns, translated_columns)
]

combined_condition = conditions[0]
for condition in conditions[1:]:
    combined_condition |= condition

incomplete_translations_df = translated_df[combined_condition]
incomplete_translations_df.shape

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
sts_cmb_trns_final_filled = dataiku.Dataset("sts_cmb_trns_final_filled")
sts_cmb_trns_final_filled.write_with_schema(translated_df)
