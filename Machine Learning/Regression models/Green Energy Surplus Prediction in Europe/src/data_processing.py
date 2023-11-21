import argparse
import pandas as pd
import os

def process_country_data(file_path):

    countries = ['SP', 'UK', 'DE', 'DK', 'SE', 'HU', 'IT', 'PO', 'NE']

    for country in countries:
                
        df = pd.read_csv(file_path, low_memory=False)

        # Dropping unnecessary columns
        columns_to_drop = ['EndTime', 'UnitName', 'AreaID']
        df = df.drop(columns=columns_to_drop)

        # Defining the output CSV file name and a common column
        load_csv_file = f'../data/load_{country}.csv'
        common_column = 'StartTime'

        # Condition to filter rows based on 'PsrType'
        condition = df['PsrType'].isin(['B01','B09', 'B10', 'B11', 'B12', 'B13', 'B15', 'B16', 'B18', 'B19'])
        
        filtered_df = df[condition].copy()
    
        # Calculating the total green energy  
        filtered_df['total'] = filtered_df.groupby('StartTime')['quantity'].transform('sum')

        # Removing duplicate rows after combining total green energy
        filtered_df = filtered_df.drop_duplicates(subset='StartTime') 

        # Merging load energy with total green energy for each country
        load_df = pd.read_csv(load_csv_file)
        load_df = load_df.drop(columns=columns_to_drop)
        df = pd.merge(load_df, filtered_df, on=common_column, how='left')
        
        output_path = '../data'
        df.to_csv(f'{output_path}/total_{country}.csv', index=False)

    return df

def normalize_by_hour(df):
    
    countries = ['SP', 'UK', 'DE', 'DK', 'SE', 'HU', 'IT', 'PO', 'NE']

    for country in countries:
        
        file_path = os.path.join('../data', f'total_{country}.csv')
        df = pd.read_csv(file_path, low_memory=False)

        # Removing unnecessary parts and converting to to datetime format
        df['StartTime'] = df['StartTime'].str[:-7].str.replace('T', ' ')
        df['StartTime'] =  pd.to_datetime(df['StartTime'])
    
        df.set_index('StartTime', inplace=True)

        # Resampling the DataFrame to hourly frequency
        df['total'] = df.resample('1H').sum()['total']
        df['load_norm'] = df.resample('1H').sum()['Load']
    
        df = df.reset_index()

        # Filtering rows where the minute part of 'StartTime' is 0
        df = df[df['StartTime'].dt.minute == 0]
    
        df.rename(columns={'total': f'green_energy_{country}', 
                         'load_norm': f'{country}_Load'}, inplace=True)    
    
         # Dropping unnecessary columns
        columns_to_drop = ['Load', 'PsrType', 'quantity' ]
        df = df.drop(columns=columns_to_drop)       

        df['StartTime'] = pd.to_datetime(df['StartTime'])
        output_path = '../data'
        df.to_csv(f'{output_path}/norm_{country}.csv', index=False)

    return df
    

def merge_files(df):

    countries = ['SP', 'UK', 'DE', 'DK', 'SE', 'HU', 'IT', 'PO', 'NE']

    for country in countries:
        
        # Initializing an empty DataFrame for the result
        result_df = pd.DataFrame()
    
        for country in countries:
            file_path = os.path.join('../data', f'norm_{country}.csv')
            df = pd.read_csv(file_path)
            
            # Assigning results to the first DataFrame
            if result_df.empty:
                result_df = df
            else:
                # Merging the current DataFrame with the result DataFrame on the 'StartTime' column
                result_df = pd.merge(result_df, df, on='StartTime', how = 'outer')
        
        result_df.set_index('StartTime', inplace=True, drop = True) 
                
        df = result_df.interpolate(method='linear', limit_direction='both', inplace=True)
    
        output_path = '../data'
        df.to_csv(f'{output_path}/merged_data.csv', index = True)

    return df

def normalize_surplus(df):
    
    countries = ['SP', 'UK', 'DE', 'DK', 'SE', 'HU', 'IT', 'PO', 'NE']

    for country in countries:
        
        file_path = os.path.join('../data', f'total_{country}.csv')
        df = pd.read_csv(file_path, low_memory=False)
    
        df['StartTime'] = df['StartTime'].str[:-7].str.replace('T', ' ')
        df['StartTime'] =  pd.to_datetime(df['StartTime'])
    
        df.set_index('StartTime', inplace=True)
    
        df['total'] = df.resample('1H').sum()['total']
        df['load_norm'] = df.resample('1H').sum()['Load']
    
        df = df.reset_index()
    
        df['StartTime'] = pd.to_datetime(df['StartTime'])
        df = df[df['StartTime'].dt.minute == 0]
    
        columns_to_drop = ['Load', 'PsrType', 'quantity' ]
        df = df.drop(columns=columns_to_drop)

        # Calculating 'surplus' and adding 'country' column
        df['surplus'] = df['total'] - df['load_norm']
        df['country'] = f'{country}'    
        output_path = '../data'
        df.to_csv(f'{output_path}/norm_surp_{country}.csv', index=False)

    return df

def surplus_calc(df):
    
    file_names = ['SP', 'UK', 'DE', 'DK', 'SE', 'HU', 'IT', 'PO', 'NE']
    
    dfs = {}
    
    for i, country_code in enumerate(file_names):
        file_path = f'../data/norm_surp_{country_code}.csv'
        df = pd.read_csv(file_path)
        dfs[f'df{i}'] = df
    
    # Concatenating the 'surplus' columns from all DataFrames
    df_new = pd.concat([df['surplus'] for df in dfs.values()], axis=1, join='outer') 
    
    df_new.columns = [f'surplus_{i}' if i < 9 else 'max_value' for i in range(9)]
    
    max_column_index = df_new.idxmax(axis=1)
    
    # Finding the maximum value in each row
    max_value = df_new.max(axis=1)
    
    # Creating a new column to store the maximum value
    df_new['max_value'] = max_value
    
    # Creating a new column to store the location of the maximum value
    df_new['max_column'] = max_column_index
    
    start_time_column = dfs['df0']['StartTime']
    df_new['StartTime'] = start_time_column
       
    df_new['max_column'] = df_new['max_column'].str[-1].astype(int)

    df = df_new
    
    df.to_csv('../data/surplus.csv', index=False)

    return df

def final_merge(df):

    # df = pd.read_csv('../data/surplus.csv')
    df2 = pd.read_csv('../data/merged_data.csv')
    
    df_fin = pd.merge(df2, df[['StartTime', 'max_column']], on='StartTime', how='inner')
    df_fin.rename(columns={'max_column': 'label'}, inplace=True)   
    
    
    df_fin.to_csv(output_file, index=False)

    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data processing script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file',
        type=str,
        default=['data/load_SP.csv', 'data/load_UK.csv', 'data/load_DE.csv', 'data/load_DK.csv',
                 'data/load_SE.csv', 'data/load_HU.csv', 'data/load_IT.csv', 'data/load_PO.csv',
                 'data/load_NE.csv', 'data/total_green_SP.csv', 'data/total_green_UK.csv','data/total_green_DE.csv',
                 'data/total_green_DK.csv', 'data/total_green_SE.csv', 'data/total_green_HU.csv',
                 'data/total_green_IT.csv', 'data/total_green_PO.csv', 'data/total_green_NE.csv']
                 
        help='Path to the raw data file to process'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='data/final_data.csv', 
        help='Path to save the processed data'
    )
    return parser.parse_args()

def main(input_file, output_file):
    df = process_country_data(input_file)
    normalized = normalize_by_hour(df)
    merged = merge_files(normalized)
    normalized_surplus = normalize_surplus(merged)
    final_merged = final_merge(normalized_surplus)
    surplus_calulated = surplus_calc(normalized_surplus)
    final_merge(surplus_calulated, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.output_file)