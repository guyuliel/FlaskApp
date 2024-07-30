import pandas as pd
import os
import pickle
os.chdir('C:\\Users\\guyul\\ProjectP3')
os.getcwd()


df = pd.read_csv('C:\\Users\\guyul\\ProjectP3\\dataset.csv')


# Remove commas from 'capacity_Engine' and 'Km' if they are strings
if df['capacity_Engine'].dtype == 'object':
    df['capacity_Engine'] = df['capacity_Engine'].str.replace(',', '')

if df['Km'].dtype == 'object':
    df['Km'] = df['Km'].str.replace(',', '')

# Convert to appropriate data types
df['Year'] = df['Year'].astype('Int64')
df['Hand'] = df['Hand'].astype('Int64')
df['Price'] = df['Price'].astype('float64')
df['Pic_num'] = df['Pic_num'].astype('Int64')
df['Supply_score'] = df['Supply_score'].astype('Int64')
df['capacity_Engine'] = pd.to_numeric(df['capacity_Engine'], errors='coerce').astype('Int64')
df['Km'] = pd.to_numeric(df['Km'], errors='coerce').astype('Int64')



# Convert columns to category
df['Gear'] = df['Gear'].astype('category')
df['Engine_type'] = df['Engine_type'].astype('category')
df['Prev_ownership'] = df['Prev_ownership'].astype('category')
df['Curr_ownership'] = df['Curr_ownership'].astype('category')
df['Color'] = df['Color'].astype('category')

# Convert date columns to datetime64
df['Cre_date'] = pd.to_datetime(df['Cre_date'], errors='coerce')
df['Repub_date'] = pd.to_datetime(df['Repub_date'], errors='coerce')

# Ensure object types for remaining columns
df['manufactor'] = df['manufactor'].astype('object')
df['model'] = df['model'].astype('object')
df['Area'] = df['Area'].astype('object')
df['City'] = df['City'].astype('object')
df['Description'] = df['Description'].astype('object')
df['Test'] = df['Test'].astype('object')

# Check for duplicates and remove them
df = df.drop_duplicates(keep='first')




# We grouped the data by 'Year' and aggregated the AVG 'Hand' for each group
year = df.groupby(['Year']).agg(hand_mean=('Hand', 'mean'),).reset_index()
year['hand_mean'] = year['hand_mean'].round().astype('Int64')

# Here we remove the extreme values in this column.
km_cleaned = df[df['Km'] <= 500000]
km_cleaned['Km'] = km_cleaned['Km'].apply(lambda x: x * 1000 if x < 1000 else x)

# We grouped the data by 'Year' and aggregated the AVG 'Km' for each group
km = km_cleaned.groupby(['Year']).agg(km_mean=('Km', 'mean'),).reset_index()
km['km_mean'] = km['km_mean'].round().astype('Int64')

# Here we remove the extreme values in this column.
engcap_cleaned = df[(df['capacity_Engine'] >= 800) & (df['capacity_Engine'] <= 6000)]

# Before starting, we will convert back the 2 categorical variables into objects in order to succeed in doing the group by.
# The problematic fields are Gear & Engine_type.
engcap_cleaned['Gear']= engcap_cleaned['Gear'].astype(str)
engcap_cleaned['Engine_type']= engcap_cleaned['Engine_type'].astype(str)

# Calculating the median for the clean data (without the extreme values) according to Group By 
engcap = engcap_cleaned.groupby(['manufactor', 'model', 'Year','Gear','Engine_type'])['capacity_Engine'].median().reset_index()
engcap = engcap.rename(columns={'capacity_Engine': 'median_capacity'})
engcap['median_capacity'] = engcap['median_capacity'].astype('Int64')


def prepare_data(df):
    # Add 'לא מוגדר' to the categories fields before filling missing values
    for col in ['Prev_ownership', 'Curr_ownership', 'Color']:
        if 'לא מוגדר' not in df[col].cat.categories:
            df[col] = df[col].cat.add_categories('לא מוגדר')

    # Handle missing values for categorical columns
    df['Prev_ownership'] = df['Prev_ownership'].fillna('לא מוגדר')
    df['Curr_ownership'] = df['Curr_ownership'].fillna('לא מוגדר')
    df['Color'] = df['Color'].fillna('לא מוגדר')
    df['Test'] = df['Test'].fillna('לא מוגדר')

    # Merge auxiliary dataframes for missing value imputation
    df = df.merge(year, on='Year', how='inner')
    df['Hand'] = df['Hand'].fillna(df['hand_mean'])
    df = df.drop(columns=['hand_mean'])

    df = df.merge(engcap, on=['manufactor', 'model', 'Year', 'Gear', 'Engine_type'], how='inner')
    df['capacity_Engine'] = df['capacity_Engine'].fillna(df['median_capacity'])
    df = df.drop(columns=['median_capacity'])

    df = df.merge(km, on='Year', how='inner')
    df['Km'] = df['Km'].fillna(df['km_mean'])
    df = df.drop(columns=['km_mean'])

    df['Pic_num'] = df['Pic_num'].fillna(0)

    # Drop unnecessary columns before one-hot encoding
    df = df.drop(['Description', 'Cre_date', 'Repub_date', 'Supply_score', 'Test'], axis=1)
    
    # One-hot encode categorical variables
    categorical_cols = ['manufactor','model','Area','City','Color','Gear','Engine_type', 'Prev_ownership', 'Curr_ownership']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Ensure all columns are of numeric type where applicable
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], downcast='integer', errors='coerce').astype('Int64')

    return df
    
df_prepared = prepare_data(df)

df_prepared.to_csv('training_data.csv',encoding='utf-8-sig',index=False)

# Create mappings for categorical columns
manufacturer_mapping = {value: idx for idx, value in enumerate(df['manufactor'].unique(), 1)}
model_mapping = {value: idx for idx, value in enumerate(df['model'].unique(), 1)}
gear_mapping = {value: idx for idx, value in enumerate(df['Gear'].unique(), 1)}
engine_type_mapping = {value: idx for idx, value in enumerate(df['Engine_type'].unique(), 1)}
prev_ownership_mapping = {value: idx for idx, value in enumerate(df['Prev_ownership'].unique(), 1)}
curr_ownership_mapping = {value: idx for idx, value in enumerate(df['Curr_ownership'].unique(), 1)}
area_mapping = {value: idx for idx, value in enumerate(df['Area'].unique(), 1)}
city_mapping = {value: idx for idx, value in enumerate(df['City'].unique(), 1)}
color_mapping = {value: idx for idx, value in enumerate(df['Color'].unique(), 1)}





# Apply mappings to the dataset
df['manufacturer'] = df['manufactor'].map(manufacturer_mapping)
df['model'] = df['model'].map(model_mapping)
df['gear'] = df['Gear'].map(gear_mapping)
df['engine_type'] = df['Engine_type'].map(engine_type_mapping)
df['prev_ownership'] = df['Prev_ownership'].map(prev_ownership_mapping)
df['curr_ownership'] = df['Curr_ownership'].map(curr_ownership_mapping)
df['area'] = df['Area'].map(area_mapping)
df['city'] = df['City'].map(city_mapping)
df['color'] = df['Color'].map(color_mapping)






# Save the mappings for later use
mappings = {
    'manufacturer': manufacturer_mapping,
    'model': model_mapping,
    'gear': gear_mapping,
    'engine_type': engine_type_mapping,
    'prev_ownership':prev_ownership_mapping,
    'curr_ownership':curr_ownership_mapping,
    'area': area_mapping,
    'city':city_mapping,
    'color':color_mapping
}

with open('C:\\Users\\guyul\\ProjectP3\\mappings.pkl', 'wb') as f:
    pickle.dump(mappings, f)
    