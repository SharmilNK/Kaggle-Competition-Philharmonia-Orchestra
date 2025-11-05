import warnings
from typing import Optional, Tuple, List
warnings.filterwarnings("ignore")
import re
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42
N_FOLDS = 5
TOP_K_FEATURES = 10

def load_csv():
    '''
    First we read all the csv files into dataframes
    Files are usually loaded with default UTF-8 encoding, this is to load numbers,characters, symbols etc without error.
    But, received an earlier error, so probablity that files could contain some inconsistent characters. Hence, we use encoding='latin1'.
    Input: CSV files- account.csv, subscriptions.csv, tickets_all.csv, concerts.csv, zipcodes.csv, train.csv, test.csv
    Output: Corresponding dataframes - df_acc, df_subs, df_tck, df_con, df_zip, df_train, df_test
    '''
    df_acc = pd.read_csv("account.csv", encoding='latin1')  
    df_subs = pd.read_csv("subscriptions.csv")
    df_tck = pd.read_csv("tickets_all.csv")
    df_con = pd.read_csv("concerts.csv")
    df_zip = pd.read_csv("zipcodes.csv")
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    return df_acc, df_subs, df_tck, df_con, df_zip, df_train, df_test

def clean_column_names(df):
    '''
    Column Name Cleaup:
    -remove leading/trailing spaces
    -set to lowercase 
    -replace dots (.) with underscores (_)
    String Columns Cleanup:
    -convert text columns to strings and trim leading/trailing spaces
    Input: df
    Output: df with cleaned column names and string columns.
    '''
    df = df.copy()
    #Column name cleanup
    df.columns = df.columns.str.strip().str.lower().str.replace('.', '_')

    # for each column with dataype =string (object) trim whitespace  
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].astype(str).str.strip()
    return df

def safe_to_numeric(df, cols):
    '''
    For each df, convert cols columns to numeric.
    There could be numeric columns that have invalid or non-numeric values, these should be converetd to NaN instead of throwing an error during type conversion,hence we use errors='coerce'.
    Input : df with cleaned column names and string columns, cols = numeric columns in df
    Output: df with the specified 'col' columns converted to numeric values (and invalid entries replaced by NaN).
    '''
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def impute_zip_logic(df_acc, df_zip):
    """
    Impute and populate shipping zip codes using account and zipcode reference df. 
    Else leave shipping_zip as NaN (no blind fill to '0').

    1. Standardize  column names 
    2. df_zip : This will be used as referrence to fill the shipping_zip_code 
       - convert city to lower case
       - use zipcodetype = Standard
       If multiple zips exist for a city, the function picks the most common one (mode)
    3. For each account_id in df_acc:
       a. If account-level shipping_zip_code is present, keep it (after normalizing to a 5-digit string)
       b. Else if shipping_city exists, try to find a matching zip from the df_zip city and populate it
       c. Else if account-level billing_zip_code exists, copy billing_zip_code to shipping_zip_code
       d. Else if the account has any tickets in df_tck, use location to find a matching zip from the df_zip city and populate it
       e. Otherwise leave shipping_zip as NaN (no blind fill to '0')
    
    Input : df_acc, df_zip
    Output: df_acc_new['shipping_zip_imputed'] 
    """

    df_acc = df_acc.copy()
    df_zip = df_zip.copy()

    # identify zip, city, account columns in df_acc, df_tck
    acc_ship_zip_col = 'shipping_zip_code'
    acc_ship_city_col = 'shipping_city'
    acc_bill_zip_col = 'billing_zip_code'
    acc_bill_city_col = 'billing_city'
    acc_account_id_col = 'account_id'
    
    # normalize zip value to 5-digit string or NaN 
    def normalize_zip(z):
        if pd.isna(z):
            return np.nan
        z = str(z).strip()
        if z == '' or z.lower() in {'nan', 'none', 'na'}:
            return np.nan
        # remove common separators
        z = re.sub(r'[^0-9]', '', z)
        if z == '':
            return np.nan
        # 5-digit prefer; if longer, take first 5 digits (common when 9-digit ZIP present)
        if len(z) >= 5:
            return z[:5].zfill(5)
        # if shorter than 5, left-pad with zeros 
        return z.zfill(5)

    # identify columns in df_zip: zip, city, zip code type
    # (use column names directly, not series)
    zip_col = 'zipcode'
    city_col = 'city'
    zip_type_col = 'zipcodetype'

    # Create a list that has maps each lowercase city name to its valid ZIP codes (and their types).
    city_to_zips = defaultdict(list)
    if zip_col in df_zip.columns and city_col in df_zip.columns:
        # standardize
        for _, r in df_zip[[zip_col, city_col, zip_type_col]].dropna(subset=[city_col]).iterrows():
            # If the ZIP value in the current row (r[zip_col]) is not missing (pd.notna), 
            # clean and standardize it with normalize_zip() else set to NaN.
            z = normalize_zip(r[zip_col]) if pd.notna(r[zip_col]) else np.nan
            if pd.isna(z):  # if z is missing (NaN) — that row is skipped and not added to the city–ZIP mapping.
                continue
            city = str(r[city_col]).strip().lower()
            ztype = str(r[zip_type_col]).strip().lower() if zip_type_col in r and pd.notna(r.get(zip_type_col)) else ''
            city_to_zips[city].append((z, ztype))

    # Populate shipping zipcode
    def pick_zip_for_city(city):
        if not city:
            return np.nan
        city_l = str(city).strip().lower()
        candidates = city_to_zips.get(city_l, [])
        if not candidates:
            return np.nan
        # populate zipcode type = standard, else use mode for that city.
        standard_zips = [z for z, zt in candidates if 'standard' in zt or 's' == zt]
        pool = standard_zips if len(standard_zips) > 0 else [z for z, _ in candidates]
        if not pool:
            return np.nan
        most_common = Counter(pool).most_common(1)
        return most_common[0][0] if most_common else np.nan

    # df_acc : create new column shipping_zip_imputed
    df_acc['shipping_zip_imputed'] = df_acc[acc_ship_zip_col] if acc_ship_zip_col in df_acc.columns else np.nan
    df_acc['shipping_zip_imputed'] = df_acc['shipping_zip_imputed'].apply(normalize_zip)

    # Attempt fills in order, if shipping_zip is still empty:
    #  1) check shipping_city in df_acc and check city_to_zips
    #  2) populate billing_zip_code from df_acc
    #  3) else check billing_city with check city_to_zips

    # 1) use shipping_city if shipping_zip missing
    if acc_ship_city_col in df_acc.columns:
        mask = df_acc['shipping_zip_imputed'].isna()
        # if any rows meet the condition (missing ZIPs) only then run this code
        if mask.any():
            df_acc.loc[mask, 'shipping_zip_imputed'] = df_acc.loc[mask, acc_ship_city_col].apply(pick_zip_for_city).values

    # 2) use billing_zip if still missing
    mask = df_acc['shipping_zip_imputed'].isna()
    if mask.any() and acc_bill_zip_col in df_acc.columns:
        df_acc.loc[mask, 'shipping_zip_imputed'] = df_acc.loc[mask, acc_bill_zip_col].apply(normalize_zip).values

    # 3) if shippping_zip still missing use billing_city with city_to_zips
    if 'shipping_zip_imputed' in df_acc.columns and acc_bill_city_col in df_acc.columns:
        mask = df_acc['shipping_zip_imputed'].isna()
        if mask.any():
            df_acc.loc[mask, 'shipping_zip_imputed'] = df_acc.loc[mask, acc_bill_city_col].apply(pick_zip_for_city).values

    # Similarly, Populate missing shipping_city values in df_acc
    # Order of preference:
    # 1) If shipping_city exists, keep it
    # 2) If billing_city exists, copy it
    # 3) If shipping_zip exists, use df_zip lookup to get city
    # 4) If billing_zip exists, use df_zip lookup to get city

    if 'shipping_city' not in df_acc.columns:
        df_acc['shipping_city'] = np.nan

    # 1) keep existing shipping_city (no action needed)

    # 2) fill from billing_city if available
    if acc_bill_city_col in df_acc.columns:
        mask = df_acc['shipping_city'].isna()
        if mask.any():
            df_acc.loc[mask, 'shipping_city'] = df_acc.loc[mask, acc_bill_city_col]

    # 3) fill from shipping_zip using df_zip reference
    if 'shipping_zip_imputed' in df_acc.columns and 'zipcode' in df_zip.columns and 'city' in df_zip.columns:
        zip_to_city = df_zip.set_index('zipcode')['city'].to_dict()
        mask = df_acc['shipping_city'].isna()
        if mask.any():
            df_acc.loc[mask, 'shipping_city'] = df_acc.loc[mask, 'shipping_zip_imputed'].map(zip_to_city)

    # 4) fill from billing_zip if still missing
    if acc_bill_zip_col in df_acc.columns and 'zipcode' in df_zip.columns and 'city' in df_zip.columns:
        mask = df_acc['shipping_city'].isna()
        if mask.any():
            df_acc.loc[mask, 'shipping_city'] = (
                df_acc.loc[mask, acc_bill_zip_col].map(zip_to_city)
            )
   
    # final normalize pass (ensure consistent 5-digit strings or NaN)
    df_acc['shipping_zip_imputed'] = df_acc['shipping_zip_imputed'].apply(normalize_zip)
    return df_acc

def merge_zip_demographics(df_acc, df_zip):
    """
    Merge zipcode demographics to accounts using 'shipping_zip_imputed' (5-digit strings).
    Input: df_acc, df_zip
    Output: df_acc (mergd with df_zip demographics)
    """
    df_zip = df_zip.copy()
    
    df_zip['zip_key'] = df_zip['zipcode'].astype(str).str.zfill(5).str[:5]
    df_acc['zip_key'] = df_acc['shipping_zip_imputed'].astype(str).str.zfill(5).str[:5]

    # select numeric columns from df_zip to merge (avoid duplicating strings)
    num_cols = df_zip.select_dtypes(include=[np.number]).columns.tolist()
    # rename them to be clear
    zip_subset = df_zip[['zip_key'] + list(num_cols)].drop_duplicates(subset=['zip_key'])
    df_acc = df_acc.merge(zip_subset, on='zip_key', how='left')
    #print(df_acc.head())
    return df_acc

def prepare_feature_tables(df_subs, df_tck, df_con, df_acc_demo, df_zip, all_accounts,df_train,df_test):
    """
    Create focused features tied to:
      - tickets (marketing_source,season,location)
      - subscriptions (section, package,counts, recency, avg price)
      - location (final_zip + zipcode demographics)
      - donors (amounts, flags)
      - concerts (concert_name,who,what,location,season)

    Input: df_acc, df_subs, df_tck, df_con, df_train
    Output: 
    - Identify correlation between features & create master_features per account_id (for train accounts).
    """
    
    df_acc = clean_column_names(df_acc_demo)
    df_subs = clean_column_names(df_subs)
    df_tck = clean_column_names(df_tck)
    df_train = clean_column_names(df_train)
    df_test = clean_column_names(df_test)
    df_zip = clean_column_names(df_zip)
    df_con = clean_column_names(df_con)



    # Convert numeric columns
    df_acc = safe_to_numeric(df_acc, ['amount_donated_2013','amount_donated_lifetime','no_donations_lifetime'])
    df_subs = safe_to_numeric(df_subs, ['price_level','no_seats','subscription_tier'])
    df_tck = safe_to_numeric(df_tck, ['price_level','no_seats','set','multiple_tickets'])

    # Parse seasons to numeric if present
    def parse_season(s):
        try:
            return int(str(s).split('-')[0])
        except:
            return np.nan
    for df in (df_subs, df_tck):
        if 'season' in df.columns:
            df['season_year'] = df['season'].apply(parse_season)

    # set df_tck : multiple_tickets to numeric = yes (1), no(0) and set df_subs : multiple_subs to numeric = yes (1), no(0)
    def map_yes_no_col(series):
    # keep original NA mask so we can re-apply real NaNs at the end
        na_mask = series.isna()
    # safe string-cleaning: replace NaN with empty string (so .str works), strip and lowercase
        s = series.fillna('').astype(str).str.strip().str.lower()
    # normalize common variants
        s = s.replace({'y': 'yes', 'n': 'no', 'true': 'yes', 'false': 'no', '1': 'yes', '0': 'no'})
    # map to numeric
        mapped = s.map({'yes': 1, 'no': 0})
    # restore NaNs for originally-missing values
        mapped[na_mask] = np.nan
        return mapped

    
    df_tck['multiple_tickets_num'] = map_yes_no_col(df_tck['multiple_tickets'])
    df_subs['multiple_subs_num'] = map_yes_no_col(df_subs['multiple_subs'])
    

    # helper function to get mode safely
    def mode_func(x):
        return x.mode().iloc[0] if not x.mode().empty else np.nan

    # Filetr subscriptions prior to 2014 
    sub_hist = df_subs[df_subs['season_year'] < 2014] if 'season_year' in df_subs.columns else df_subs.copy()

    ## SUBSCRIPTIONS 

    # For each account_id (df_subs) identify the aggregates based on season, price,seats. Also check mode of loaction, section and package for each user.
    if 'account_id' in sub_hist.columns:
        g = sub_hist.groupby('account_id')
        sub_feats = g.agg({
            'season_year': ['max','min','count','nunique',mode_func],
            'price_level': ['mean','max','min'],
            'no_seats': ['mean','sum'],
            'location': lambda s: s.nunique(),
            'section': [mode_func],
            'package': [mode_func],
            'multiple_subs_num': [mode_func]
        }).reset_index()
        # converting those multi-level column names into single strings
        sub_feats.columns = ['account_id'] + ['sub_' + '_'.join(filter(None, map(str, col))).strip() for col in sub_feats.columns[1:]]
    else:
        sub_feats = pd.DataFrame(columns=['account_id'])

    #print(sub_feats.columns)
    # Here are th column values: Index(['account_id', 'sub_season_year_max', 'sub_season_year_min','sub_season_year_count', 'sub_season_year_nunique', 'sub_season_year_mode_func', 'sub_price_level_mean','sub_price_level_max', 'sub_price_level_min', 'sub_no_seats_mean','sub_no_seats_sum', 'sub_location_<lambda>', 'sub_section_mode_func','sub_package_mode_func', 'sub_multiple_subs_num_mode_func'],dtype='object')

    # Create new features : sub_last_season= The year of the last subscription, sub_months_since_last = Number of months since last subscription
    if 'sub_season_year_max' in sub_feats.columns: 
        sub_feats['sub_last_season'] = sub_feats['sub_season_year_max']
        sub_feats['sub_months_since_last'] = (2014 - sub_feats['sub_last_season']) * 12
    else:
        sub_feats['sub_months_since_last'] = 999

    # Create new features from aggregats (sum,count): sub_total_seats= total number of seats purchased, sub_avg_price =average subscription price, sub_counts= number of subscription records 
    sub_feats['sub_total_seats'] = sub_feats.get('sub_no_seats_sum', 0)
    sub_feats['sub_avg_price'] = sub_feats.get('sub_price_level_mean', 0)
    sub_feats['sub_counts'] = sub_feats.get('sub_season_year_count', 0)

    # Create new feature (mode) most frequest location, section, package,season, multiple_subs
    sub_feats['sub_freq_location'] = sub_feats.get('sub_location_<lambda>', 0)
    sub_feats['sub_freq_section'] = sub_feats.get('sub_section_mode_func', 0)
    sub_feats['sub_freq_package'] = sub_feats.get('sub_package_mode_func', 0)
    sub_feats['sub_freq_season'] = sub_feats.get('sub_season_mode_func', 0)
    sub_feats['sub_freq_subscribers'] = sub_feats.get('sub_multiple_subs_mode_func', 0)

    ## ACCOUNTS - DONORS & LOCATION 

    # For ach account_id, identify if they are a donar, amount donated last year/lifetime and count/avg donations 
    # Also, check if they are major donors (within 75% quantile.)
    donor_feats = df_acc[['account_id']].copy()
    donor_feats['don_amount_2013'] = df_acc.get('amount_donated_2013', 0).fillna(0)
    donor_feats['don_amount_lifetime'] = df_acc.get('amount_donated_lifetime', 0).fillna(0)
    donor_feats['don_count_lifetime'] = df_acc.get('no_donations_lifetime', 0).fillna(0)
    donor_feats['is_donor'] = (donor_feats['don_amount_lifetime'] > 0).astype(int)
    donor_feats['avg_donation'] = (donor_feats['don_amount_lifetime'] / donor_feats['don_count_lifetime']).replace(np.inf,0).fillna(0)
    donor_feats['is_major_donor'] = (donor_feats['don_amount_lifetime'] > donor_feats['don_amount_lifetime'].quantile(0.75)).astype(int)

    #print(donor_feats.column)
    #Here are the columns: Index(['account_id', 'don_amount_2013', 'don_amount_lifetime','don_count_lifetime', 'is_donor', 'avg_donation', 'is_major_donor'],dtype='object')
    
    # Normalize and merge based on zip code for location demographics

    acc = df_acc.copy()
    tck = df_tck.copy()
    z = df_zip.copy()
    zip_col = 'zipcode'
    pop_col = 'estimatedpopulation'
    wages_col = 'totalwages'
    tax_col = 'taxreturnsfiled'
    lat_col = 'lat'
    long_col = 'long'

    # Normalize zip keys on all tables to 5-char strings for robust joins
    def normalize_zip_str(x):
        if pd.isna(x): 
            return np.nan
        s = str(x).strip()
        s = ''.join(ch for ch in s if ch.isdigit())
        if s == '':
            return np.nan
        if len(s) >= 5:
            return s[:5].zfill(5)
        return s.zfill(5)

    if zip_col is None:
        raise ValueError("df_zip must have a zipcode column ")

    z = z.copy()
    z['zip_key'] = z[zip_col].apply(normalize_zip_str)

    acc = acc.copy()
    acc['zip_key'] = acc['shipping_zip_imputed']

    #Create list with demographic columns
    zip_demo_cols = []
    if pop_col:
        zip_demo_cols.append('pop')
        z = z.rename(columns={pop_col: 'pop'})
    if wages_col:
        zip_demo_cols.append('wages')
        z = z.rename(columns={wages_col: 'wages'})
    if tax_col:
        zip_demo_cols.append('taxreturns')
        z = z.rename(columns={tax_col: 'taxreturns'})
    if lat_col:
        z = z.rename(columns={lat_col: 'lat'})
    if long_col:
        z = z.rename(columns={long_col: 'lon'})

     # keep only one row per zip_key (if duplicates, take median for numeric fields)
    agg_cols = {}
    for c in ['pop','wages','taxreturns','lat','lon']:
        if c in z.columns:
            agg_cols[c] = 'median'
    zip_demo = z[['zip_key'] + list(agg_cols.keys())].drop_duplicates(subset=['zip_key'])
    if agg_cols:
        zip_demo = z.groupby('zip_key').agg(agg_cols).reset_index()

    # Create features

    # For each zip code create features: zip_avg_wage_per_return=wages/tax returns, zip_taxreturns_per_capita=taxreturns/population,zip_wages_per_capita = wages/population
    def safe_div(a,b):
        return (a / b).replace([np.inf, -np.inf], np.nan)

    if 'wages' in zip_demo.columns and 'taxreturns' in zip_demo.columns:
        zip_demo['zip_avg_wage_per_return'] = safe_div(zip_demo['wages'], zip_demo['taxreturns'])
    if 'taxreturns' in zip_demo.columns and 'pop' in zip_demo.columns:
        zip_demo['zip_taxreturns_per_capita'] = safe_div(zip_demo['taxreturns'], zip_demo['pop'])
    if 'wages' in zip_demo.columns and 'pop' in zip_demo.columns:
        zip_demo['zip_wages_per_capita'] = safe_div(zip_demo['wages'], zip_demo['pop'])

    
    # merge zip demographic features with df_acc_demo
    acc_feats = acc[['account_id','zip_key']].drop_duplicates(subset=['account_id']).merge(zip_demo, on='zip_key', how='left')
    #print(acc_feats.head)


    # Now create tier features : Based on the population, wage, tax returns per zipcode create new tiers that can be used for comparision.
    ''' 
        1. zip_income_tier = Ranks zip_avg_wage_per_return as low, mid, high, very_high by dividing all zip codes into 4 equal groups based on quantile
        2. zip_pop_tier = Ranks pop as rural, suburban, urban, dense_urban by dividing all zip codes into 4 equal groups based on quantile
        Now, convert these categories into separate binary columns (e.g., inc_tier_low, inc_tier_mid, pop_tier_urban, etc.)
        Groupby: Removes any duplicate account_id rows by taking the first occurrence.
    '''
    if 'zip_avg_wage_per_return' in acc_feats.columns and acc_feats['zip_avg_wage_per_return'].nunique(dropna=True) > 1:
        acc_feats['zip_income_tier'] = pd.qcut(acc_feats['zip_avg_wage_per_return'].rank(method='first'), q=4, labels=['low','mid','high','very_high'])
    else:
        acc_feats['zip_income_tier'] = 'unknown'

    if 'pop' in acc_feats.columns and acc_feats['pop'].nunique(dropna=True) > 1:
        acc_feats['zip_pop_tier'] = pd.qcut(acc_feats['pop'].rank(method='first'), q=4, labels=['rural','suburban','urban','dense_urban'])
    else:
        acc_feats['zip_pop_tier'] = 'unknown'

    # one-hot encode the tiers 
    acc_dummies = pd.get_dummies(acc_feats[['account_id','zip_income_tier','zip_pop_tier']], columns=['zip_income_tier','zip_pop_tier'], prefix=['inc_tier','pop_tier'])
    # drop duplicate account_id column 
    acc_demo1 = acc_dummies.groupby('account_id').first().reset_index()


    # Now create interaction features : Based on a customer's donation amount and the average income level of their ZIP code
    '''
    don_col = acc['amount_donated_lifetime']
    acc_demo2 = acc[['account_id']].copy()
    #   - acc[don_col].fillna(0): replaces missing donation amounts with 0
    #   - acc_feats.get('zip_avg_wage_per_return', 0): pulls the corresponding ZIP's average wage; if that column doesn't exist, returns 0 instead
    #   - .fillna(0): replaces any missing wage values with 0
    
    # get zip avg wage per account from acc_feats (ensure alignment by account_id)
    if 'zip_avg_wage_per_return' in acc_feats.columns:
        wage_series = acc[['account_id']].merge(
            acc_feats[['account_id','zip_avg_wage_per_return']],
            on='account_id', how='left'
        )['zip_avg_wage_per_return'].fillna(0)
    else:
        wage_series = pd.Series(0, index=acc.index)

    # if donation column was found, compute interaction; otherwise set 0
    if don_col is not None and don_col in acc.columns:
        # fill NaN donations with 0, ensure same index alignment as wage_series
        donation_series = acc[don_col].fillna(0).reset_index(drop=True)
        wage_series = wage_series.reset_index(drop=True)
        acc_demo2['income_x_donation'] = donation_series * wage_series
    else:
        # If no donation column is found, create the feature as 0 for all accounts.
        acc_demo2['income_x_donation'] = 0
    '''
    # From the above features created, select all numeric features to keep
    keep_cols = ['account_id']
    for c in ['pop','wages','taxreturns','zip_avg_wage_per_return','zip_taxreturns_per_capita','zip_wages_per_capita','lat','lon']:
        if c in acc_feats.columns:
            keep_cols.append(c)

    out = acc_feats[keep_cols].copy()
    # merge acc_demo1 and acc_demo2 (cat and numeric features zip demographics)
    out = out.merge(acc_demo1, on='account_id', how='left')
    #out = out.merge(acc_demo2, on='account_id', how='left')

    # Fill remaining numeric NaNs with 0 (safe for ML features; or you can fill with medians instead)
    for c in out.select_dtypes(include=[np.number]).columns:
        out[c] = out[c].fillna(0)

    
    account_feats = out.copy()
    #print(account_feats.head())
        
    ## TICKETs

    tck_hist = df_tck[df_tck['season_year'] < 2014] if 'season_year' in df_tck.columns else df_tck.copy()
    if 'account_id' in tck_hist.columns:
        tg = tck_hist.groupby('account_id').agg({
            'season_year': ['count','nunique','max'],
            'price_level':  [mode_func],
            'no_seats': ['sum', mode_func],
            'set':  [mode_func],
            'multiple_tickets_num' :  [mode_func],
            'marketing_source' :  [mode_func]

        }).reset_index()
        tg.columns = ['account_id'] + ['tkt_' + '_'.join(filter(None,map(str,col))).strip() for col in tg.columns[1:]]
    else:
        tg = pd.DataFrame(columns=['account_id'])

    #print(tg.head())

    ## MASTER FEATURE MERGE 
    # Start from the train accounts list (labels)
    master = all_accounts[['account_id']].drop_duplicates().copy()

    for ft in [sub_feats, tg, donor_feats, account_feats]:
        if 'account_id' in ft.columns:
            master = master.merge(ft, on='account_id', how='left')
    
    #To the master, add mode(location) & who (musician)

    ## CONCERTS

    # helper: safe mode extraction
    def safe_mode(series):
        """Return the mode (most frequent) value of a Series or np.nan if none."""
        s = series.dropna()
        if s.empty:
            return np.nan
        m = s.mode()
        return m.iloc[0] if not m.empty else np.nan
    # Create 'musician' column in df_con by taking substring before the first comma in column 'who'
    # trim whitespace and normalize
    df_con = df_con.copy()
    df_con['who_str'] = df_con['who'].fillna('').astype(str)
    df_con['musician'] = df_con['who_str'].str.split(',', n=1).str[0].str.strip()
    # if musician ends up empty string, convert to NaN

    df_con.loc[df_con['musician'].str.len() == 0, 'musician'] = np.nan
    # Create 'concert' column in df_con 
    df_con['concert'] = df_con['concert_name'].fillna('').astype(str).str.strip()
    df_con.loc[df_con['concert'].str.len() == 0, 'concert'] = np.nan

    # Get unique values for concert
    unique_concerts = df_con['concert'].dropna().unique()
    
    # reduce to unique (season, location, musician) rows to avoid duplicate mappings
    con_map = df_con[['season', 'location', 'musician']].drop_duplicates()

    # Merge musician onto df_subs by season+location (left merge so subs keep their rows)
    df_subs = df_subs.copy()
    # ensure season/location columns are comparable: strip & lowercase for robust matching
    df_subs['season_key'] = df_subs['season'].astype(str).str.strip()
    df_subs['location_key'] = df_subs['location'].astype(str).str.strip().str.lower()
    con_map['season_key'] = con_map['season'].astype(str).str.strip()
    con_map['location_key'] = con_map['location'].astype(str).str.strip().str.lower()

    # on merge if multiple concerts at same season+location exist, musician may duplicate; we'll handle below
    df_subs = df_subs.merge(con_map[['season_key','location_key','musician']],
                            on=['season_key','location_key'],
                            how='left')
    musician_per_account = (
    df_subs.groupby('account_id')['musician']
    .agg(lambda s: safe_mode(s))
    .reset_index()
    .rename(columns={'musician': 'musician_mode'})
        )       

    # For each account_id, pick the most frequent musician across that account's subscriptions
    # this gives one musician feature per account; if none, it will be NaN
    df_subs_annot = df_subs.merge(musician_per_account, on='account_id', how='left')

    # for each account, try to find the location mode *where the musician equals the musician_mode*
    #    if no rows match (e.g., musician_mode is NaN or account never saw that musician), fall back to overall location mode for the account
    # first: musician-specific location mode
    mus_loc = (
        df_subs_annot[df_subs_annot['musician'].notna() & df_subs_annot['musician_mode'].notna() & 
                    (df_subs_annot['musician'] == df_subs_annot['musician_mode'])]
        .groupby('account_id')['location_key']
        .agg(lambda s: safe_mode(s))
        .reset_index()
        .rename(columns={'location_key': 'location_mode_for_musician'})
    )

    # second: overall location mode per account (fallback)
    overall_loc = (
        df_subs.groupby('account_id')['location_key']
        .agg(lambda s: safe_mode(s))
        .reset_index()
        .rename(columns={'location_key': 'location_mode_overall'})
    )

    # 4) combine: prefer musician-specific location, else overall
    location_per_account = musician_per_account.merge(mus_loc, on='account_id', how='left') \
                                            .merge(overall_loc, on='account_id', how='left')

    # create final location_mode column with fallback
    location_per_account['location_mode'] = location_per_account['location_mode_for_musician'].combine_first(
        location_per_account['location_mode_overall']
    )

    # reduce to desired columns
    location_per_account = location_per_account[['account_id', 'musician_mode', 'location_mode']]

    # 5) Merge musician_mode and location_mode into master_df (left join so all master rows kept)
    master_df = master.copy()
    master_df['account_id'] = master_df['account_id'].astype(str)
    location_per_account['account_id'] = location_per_account['account_id'].astype(str)

    master_df = master_df.merge(location_per_account, on='account_id', how='left')

    # 6) Label-encode both musician_mode and location_mode into integer features (0 = missing)


    def label_encode_col(df, col, prefix):
        if col not in df.columns:
            df[prefix + '_enc'] = 0
            return df
        s = df[col].fillna('').astype(str)
        non_missing = s.replace('', np.nan).dropna().unique().tolist()
        if len(non_missing) == 0:
            df[prefix + '_enc'] = 0
            return df
        le = LabelEncoder()
        le.fit([str(x) for x in sorted(non_missing)])
        # map known values to 1..N, unknown/missing -> 0
        mapping = {v: i+1 for i, v in enumerate(le.classes_)}
        df[prefix + '_enc'] = s.map(mapping).fillna(0).astype(int)
        return df

    master_df = label_encode_col(master_df, 'musician_mode', 'musician_mode')
    master_df = label_encode_col(master_df, 'location_mode', 'location_mode')

    # Add shipping location, Zip
    acc_small = acc[['account_id', 'shipping_zip_imputed', 'shipping_city']].copy()
    acc_dedup = acc_small.groupby('account_id').agg({
    'shipping_zip_imputed':  [mode_func],
    'shipping_city': [mode_func]
            }).reset_index()
    
    # Flatten the MultiIndex columns
    acc_dedup.columns = ['account_id', 'shipping_zip_imputed', 'shipping_city']
    master_df = master_df.merge(
    acc_dedup[['account_id','shipping_zip_imputed','shipping_city']],
    on='account_id',
    how='left')
    # Group rare values into 'other' to limit one-hot encoded columns
    zip_counts = master_df['shipping_zip_imputed'].value_counts()
    city_counts = master_df['shipping_city'].value_counts()

    top_zips = zip_counts[zip_counts >= 10].index
    top_cities = city_counts[city_counts >= 10].index

    master_df['shipping_zip_imputed'] = master_df['shipping_zip_imputed'].apply(
        lambda x: x if x in top_zips else 'other'
    )
    master_df['shipping_city'] = master_df['shipping_city'].apply(
        lambda x: x if x in top_cities else 'other'
    )

    # One-hot encode
    master_df = pd.get_dummies(master_df, columns=['shipping_zip_imputed', 'shipping_city'], prefix=['zip', 'city'])

    #master_df.to_csv('master_df.csv', index=False)
    return master_df

def select_top_features(master, df_train, top_k=TOP_K_FEATURES, ):
    """
    Select top_k features by absolute Pearson correlation with label.
    Returns a list of the top_k most correlated feature names.
    """
    # merge labels into the master feature table
    master_train = master[master['account_id'].isin(df_train['account_id'])]
    m = master_train.merge(df_train[['account_id', 'label']], on='account_id', how='left')
    
    

    # compute correlations only for numeric columns
    numeric_cols = m.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['label']]

    corrs = {}
    for c in numeric_cols:
        try:
            corr = m[c].corr(m['label'])
            corrs[c] = 0 if pd.isna(corr) else corr
        except Exception:
            corrs[c] = 0

    # print correlations sorted by absolute value
    print("\nFeature Correlations with Purchase:")
    sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
    for feature, r_value in sorted_corrs:
        print(f"{feature:40s} r = {r_value:7.4f}")

    # pick the top_k feature names
    chosen = [f for f, _ in sorted_corrs[:top_k]]
    # final safety: ensure chosen are in master columns
    chosen = [c for c in chosen if c in master.columns]

    return chosen


def build_train_matrix(master, df_train, chosen_features):
    '''
    Merge the master df with df_train and split into X and y
    Returns X,y, train_medians, train_modes
    '''
    #keep only chosen features + account_id + label in train dataset
    master_train = master[master['account_id'].isin(df_train['account_id'])].copy()
    train = master_train.merge(df_train[['account_id', 'label']], on='account_id', how='inner')
    

    #Ensure chosen features exist in master
    chosen = [c for c in chosen_features if c in train.columns]
    if len(chosen) == 0:
        raise ValueError("None of chosen_features were found in master columns.")

    # Create a train DataFrame that includes account_id, chosen features and label
    cols = ['account_id'] + chosen + ['label']
    train = train[cols].copy()

    # X keeps account_id and features
    X = train[['account_id'] + chosen].copy()
    y = train['label'].reset_index(drop=True)  

    # compute medians -on training data only
    train_idx = X['account_id'].isin(df_train['account_id'])
    train_medians = X.loc[train_idx, chosen].median(numeric_only=True)
    #    Using DataFrame.mode() which may return multiple rows; take first row if present.
    train_modes_df = X.loc[train_idx, chosen].mode(dropna=True)
    if not train_modes_df.empty:
        train_modes = train_modes_df.iloc[0]
    else:
        # if no mode could be computed (all NaN), create a Series of NaNs aligned to chosen
        
        train_modes = _pd.Series([_pd.NA] * len(chosen), index=chosen)

   # Impute chosen_features in X using training medians where numeric; for non-numeric fallback to mode
    #    We'll apply numeric median where available; for columns not in train_medians (non-numeric) use train_modes
    for col in chosen:
        if col in train_medians.index and not pd.isna(train_medians[col]):
            X[col] = X[col].fillna(train_medians[col])
        else:
            # fallback: if a mode exists for this column, use it; else leave NaN
            if col in train_modes.index and not pd.isna(train_modes[col]):
                X[col] = X[col].fillna(train_modes[col])

    # Prepare model matrix: drop account_id from features, but keep it as index for traceability
    X_model = X.set_index('account_id', drop=False)
    # ensure X_model and y have the same sorted index (account_id)
    X_model = X_model.sort_index()

    # confirm there are no remaining NaNs in these features
    #print(X[chosen_features].isna().sum())

    return X_model, y, train_medians, train_modes


def train_cv_ensemble(X, y, params=None, n_folds=5, random_state=42):
    """
    Train an ensemble of LightGBM + Ridge using Stratified K-Fold CV.
    For each fold:
      - fit a RobustScaler on the training split, transform train & val
      - train LightGBM on scaled features
      - train Ridge on the same scaled features
      - produce validation predictions from both, average them -> fold val pred
    Returns:
      lgb_models: list of LightGBM boosters (one per fold)
      ridge_models: list of Ridge models (one per fold)
      scalers: list of per-fold RobustScaler objects (fit on each fold's train)
      scaler_full: RobustScaler fitted on entire X (use for test-time scaling)
      fold_scores: list of fold ROC-AUCs (based on averaged preds)
      cv_score: overall OOF ROC-AUC
      oof_preds: full-length array of out-of-fold averaged predictions
    """
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'n_estimators': 10000,
            'learning_rate': 0.005,
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.2,
            'reg_lambda': 0.2,
            'random_state': random_state,
            'verbosity': -1,
            'is_unbalance': False
        }

    # Defensive copies
    X = X.copy()
    y = y.copy().reset_index(drop=True)

    # drop account_id if present (not for indexing)
    if 'account_id' in X.columns:
        X = X.drop(columns=['account_id'])

    # keep numeric columns only
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in X to train on. Provide numeric features only.")
    X_num = X[numeric_cols].apply(pd.to_numeric, errors='coerce').astype(float)

    # final global scaler fit on full training X (useful for test-time scaling)
    scaler_full = RobustScaler()
    scaler_full.fit(X_num)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    lgb_models = []
    ridge_models = []
    scalers = []           # per-fold scalers (fit on each fold's train)
    oof_preds = np.zeros(X_num.shape[0])
    fold_scores = []
        
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_num, y), 1):
        # split
        X_tr = X_num.iloc[tr_idx].copy()
        X_val = X_num.iloc[val_idx].copy()
        y_tr = y.iloc[tr_idx].copy()
        y_val = y.iloc[val_idx].copy()

        # fold-wise scaler (fit on X_tr only)
        scaler = RobustScaler()
        X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
        scalers.append(scaler)

        # ----- LightGBM -----
        lgb_train = lgb.Dataset(X_tr_scaled, label=y_tr)
        lgb_val = lgb.Dataset(X_val_scaled, label=y_val, reference=lgb_train)

        model_lgb = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train', 'valid'],
            num_boost_round=10000,
            callbacks=[lgb.early_stopping(stopping_rounds=100)],
            
        )
        lgb_models.append(model_lgb)

        # ----- Ridge -----
        # Ridge requires 1d y and numeric X; use default alpha=1.0 (tune separately if desired)
        model_ridge = Ridge(alpha=1.0, random_state=random_state)
        model_ridge.fit(X_tr_scaled, y_tr)
        ridge_models.append(model_ridge)

        # ----- predictions -----
        
        pred_lgb = model_lgb.predict(X_val_scaled, num_iteration=model_lgb.best_iteration)
        pred_ridge = model_ridge.predict(X_val_scaled)

        # Combine both (equal weight)
        pred_avg = (pred_lgb + pred_ridge) / 2.0

        # Optional: show first few predictions to compare LightGBM vs Ridge
        print(f"\nFold {fold} prediction samples:")
        display_df = pd.DataFrame({
            'LGB_pred': pred_lgb[:10],   # show first 10 predictions for clarity
            'Ridge_pred': pred_ridge[:10],
            'Avg_pred': pred_avg[:10]
        })
        print(display_df)
    
        # Store average predictions for OOF evaluation
        oof_preds[val_idx] = pred_avg


        score = roc_auc_score(y_val, pred_avg)
        fold_scores.append(score)

        print(f"Fold {fold} ROC-AUC (ensemble): {score:.4f} (lgb_best_iter={model_lgb.best_iteration})")

    cv_score = roc_auc_score(y, oof_preds)
    print(f"\nCV OOF ROC-AUC (ensemble): {cv_score:.4f}")

    return lgb_models, ridge_models, scalers, scaler_full, fold_scores, cv_score, oof_preds


def prepare_test_matrix(master_all_accounts, chosen_features, train_medians, train_modes):
    """
    Build test matrix aligned to chosen_features using train_medians/train_modes for imputation.
    Returns X_test indexed by account_id and containing the same feature columns (and _missing flags)
    as the training X_model returned by build_train_matrix
    
    """
    X_test = master_all_accounts.set_index('account_id')[chosen_features].reindex(master_all_accounts['account_id']).reset_index(drop=False)
    # Ensure columns exist
    for c in chosen_features:
        if c not in X_test.columns:
            X_test[c] = 0
    
        
    # Impute missing using train_medians & modes passed in
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    # Impute numeric columns with train_medians when available; otherwise use train_modes; else leave 0
    for col in chosen_features:
        if col in train_medians.index and not pd.isna(train_medians[col]):
            X_test[col] = X_test[col].fillna(train_medians[col])
        elif col in train_modes.index and not pd.isna(train_modes[col]):
            X_test[col] = X_test[col].fillna(train_modes[col])
        else:
            X_test[col] = X_test[col].fillna(0)   # last-resort fallback

    # set account_id as index for traceability 
    X_test = X_test.set_index('account_id', drop=True)

    return X_test

'''
Why we take the average prediction (mean)
When you train several cross-validated models (models list), each one has:the same architecture and hyperparameters,but slightly different training data (each fold excludes a different validation subset).Every model produces slightly different probability estimates — some higher, some lower — due to random variation and fold-specific patterns.By averaging their predicted probabilities, Reduce variance / noise→ each model’s random error tends to cancel out, giving a more stable and generalizable ensemble prediction.
cross-validation models are meant to approximate the same underlying predictor; averaging them mirrors “bagging” (Bootstrap Aggregating) — a standard technique for variance reduction.
'''

def ensemble_predict_mean(lgb_models, ridge_models, X_test, scaler_full=None):
    """
    Given lists of trained lgb_models and ridge_models (same length), produce
    average predictions across models and model types.

    Args:
      - lgb_models: list of trained lightgbm boosters
      - ridge_models: list of trained Ridge models (same number of folds)
      - X_test: DataFrame of test features (may include account_id column)
      - scaler_full: RobustScaler fitted on full training X (recommended). If provided,
                     it will be applied to X_test before Ridge prediction and to LGB if models were trained on scaled data.

    Returns:
      - avg_preds: numpy array of averaged predictions (shape = n_rows)
    """
    X = X_test.copy()
    if 'account_id' in X.columns:
        X = X.drop(columns=['account_id'])
    # keep numeric only, coerce to float
    X_num = X.select_dtypes(include=[np.number]).apply(pd.to_numeric, errors='coerce').astype(float)

    if scaler_full is not None:
        X_scaled = pd.DataFrame(scaler_full.transform(X_num), columns=X_num.columns, index=X_num.index)
    else:
        # If no scaler provided assume X_test is already scaled similarly to training folds
        X_scaled = X_num.copy()

    # We will predict with each fold's pair of models and average across them:
    preds_stack = []
    n_folds = max(len(lgb_models), len(ridge_models))
    for i in range(n_folds):
        # pick model i if exists else use last
        m_lgb = lgb_models[i] if i < len(lgb_models) else lgb_models[-1]
        m_ridge = ridge_models[i] if i < len(ridge_models) else ridge_models[-1]

        # LGB predict expects numpy array or DataFrame; use values
        pred_lgb = m_lgb.predict(X_scaled, num_iteration=m_lgb.best_iteration)
        pred_ridge = m_ridge.predict(X_scaled)

        pred_avg = (0.5*pred_lgb + 0.5 *pred_ridge) 
        preds_stack.append(pred_avg)

    preds_stack = np.vstack(preds_stack)  # shape: (n_folds, n_samples)
    avg_preds = preds_stack.mean(axis=0)  # average across folds

    return avg_preds

def main ():
    '''
    Run Pipeline
    '''
    df_acc, df_subs, df_tck, df_con, df_zip, df_train, df_test = load_csv()

    df_acc = clean_column_names(df_acc)
    df_subs = clean_column_names(df_subs)
    df_tck = clean_column_names(df_tck)
    df_zip = clean_column_names(df_zip)
    df_train = clean_column_names(df_train)
    df_test = clean_column_names(df_test)
    df_con = clean_column_names(df_con)



    df_acc_impute = impute_zip_logic(df_acc, df_zip)
    df_acc_demo = merge_zip_demographics(df_acc_impute, df_zip)


    # We also need a master for all accounts 
    id_sources = []
    if 'account_id' in df_acc.columns:
            id_sources.append(df_acc['account_id'])
    if 'account_id' in df_subs.columns:
            id_sources.append(df_subs['account_id'])
    if 'account_id' in df_tck.columns:
            id_sources.append(df_tck['account_id'])
    if id_sources:
            all_acc_ids = pd.Series(pd.concat(id_sources).unique(), name='account_id')
    else:
            all_acc_ids = pd.Series([], name='account_id')

    all_accounts = pd.DataFrame({'account_id': all_acc_ids})
    # Build same features for all_accounts 
    master = prepare_feature_tables(df_subs, df_tck, df_con, df_acc_demo, df_zip, all_accounts,df_train,df_test)

    # parameter = master
    chosen = select_top_features(master, df_train, top_k=TOP_K_FEATURES)

    X_model, y,train_medians, train_modes = build_train_matrix(master, df_train, chosen)

    lgb_models, ridge_models, scalers, scaler_full, fold_scores, cv_score, oof_preds = train_cv_ensemble(X_model, y, params=None, n_folds=5, random_state=42)
    
    
    #cdShould you add more folds? No. The current 5 folds are Highly consistent .More folds won't significantly improve performance, just add training time.

    # Prepare test features: master_all holds features for all accounts; we need test.account ids
    df_test['account_id'] = df_test['id'].astype(str).str.strip()

    # get test master features for those accounts in df_test
    # Ensure master_all index is account_id
    master_all = master.drop_duplicates(subset=['account_id'])
    test_master = master_all[master_all['account_id'].isin(df_test['account_id'].astype(str))].copy()
    missing_test_accounts = set(df_test['account_id'].astype(str)) - set(test_master['account_id'].astype(str))
    if missing_test_accounts:
            # create blank rows for missing accounts to keep consistent shapes
            add_rows = pd.DataFrame({'account_id': list(missing_test_accounts)})
            test_master = pd.concat([test_master, add_rows], ignore_index=True, sort=False).fillna(0)

    X_test = prepare_test_matrix(test_master, chosen, train_medians, train_modes)
    
    avg_pred =ensemble_predict_mean(lgb_models, ridge_models, X_test, scaler_full=scaler_full)
    # Prepare submission: match df_test order
    submission = pd.DataFrame({
            'id': df_test['account_id'].astype(str),
            'predicted': 0.0
        })
        # map predictions from test_master order (test_master.account_id -> preds)
    pred_map = dict(zip(test_master['account_id'].astype(str), avg_pred))
    submission['predicted'] = submission['id'].map(pred_map).fillna(0.0)

    submission.to_csv('submission.csv', index=False)
    print("Saved submission.csv. CV ROC-AUC:", cv_score)
    print("Fold scores:", fold_scores)
    print("Top features used:", chosen)

if __name__ == "__main__":
        main()