import pandas as pd
import os
from datetime import datetime

# Create a backup of the original file
def backup_file(file_path):
    backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.{backup_time}.bak"
    os.system(f"cp {file_path} {backup_path}")
    print(f"Backup created at: {backup_path}")

def update_register_numbers(file_path):
    # Create a backup
    backup_file(file_path)
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Check if 'Register Number' column exists
    column_name = None
    possible_column_names = ['Register Number', 'RegisterNumber', 'Reg No', 'RegNo']
    
    for col in possible_column_names:
        if col in df.columns:
            column_name = col
            break
    
    if not column_name:
        print(f"Error: Could not find register number column. Available columns are: {df.columns.tolist()}")
        return
    
    # Count original matching records
    original_count = sum(df[column_name].astype(str).str.startswith('714023202'))
    print(f"Found {original_count} register numbers starting with '714023202'")
    
    if original_count == 0:
        print("No register numbers need to be updated.")
        return
    
    # Create a function to update the register numbers
    def update_number(reg_num):
        reg_str = str(reg_num)
        if reg_str.startswith('714023202'):
            return '714023247' + reg_str[9:]
        return reg_num
    
    # Apply the transformation
    df[column_name] = df[column_name].apply(update_number)
    
    # Save the updated dataframe back to CSV
    df.to_csv(file_path, index=False)
    print(f"Successfully updated {original_count} register numbers in {file_path}")
    print("New format: 714023247xxx")

if __name__ == "__main__":
    csv_path = "NAME_LIST.csv"
    update_register_numbers(csv_path)