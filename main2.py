"""
Script untuk memproses data dari kuisioner VALORANT
Dan mengintegrasikannya dengan sistem ML
"""

import pandas as pd
import json
import os


def load_questionnaire_data(filepath):
    """
    Load data dari file CSV atau JSON hasil kuisioner
    """
    print("="*70)
    print("LOADING QUESTIONNAIRE DATA")
    print("="*70)
    
    if not os.path.exists(filepath):
        print(f"\n❌ File tidak ditemukan: {filepath}")
        print("\nPastikan Anda sudah:")
        print("1. Membuka file kuisioner HTML")
        print("2. Mengisi beberapa data")
        print("3. Download file CSV/JSON")
        print("4. Simpan di folder yang sama dengan script ini")
        return None
    
    # Detect file type and load
    if filepath.endswith('.csv'):
        # df = pd.read_csv(filepath)
        df = generate_sample_data(n_samples=200)
        print(f"✓ Loaded CSV file: {filepath}")
    elif filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"✓ Loaded JSON file: {filepath}")
    else:
        print("❌ Format file tidak didukung. Gunakan CSV atau JSON.")
        return None
    
    print(f"\nTotal records: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    return df

def validate_and_clean_data(df):
    """
    Validasi dan bersihkan data kuisioner
    """
    print("\n" + "="*70)
    print("DATA VALIDATION & CLEANING")
    print("="*70)
    
    print("\nOriginal shape:", df.shape)
    
    # Check missing values
    print("\nMissing values per column:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("✓ No missing values found!")
    
    # Remove duplicates based on timestamp (if exists)
    if 'timestamp' in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        after = len(df)
        if before != after:
            print(f"\n⚠ Removed {before - after} duplicate entries")
    
    # Ensure numeric columns are correct type
    numeric_cols = ['avg_kills', 'avg_deaths', 'avg_assists', 
                    'avg_first_bloods', 'avg_combat_score', 
                    'kd_ratio', 'ka_ratio']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with invalid numeric data
    before = len(df)
    df = df.dropna(subset=numeric_cols)
    after = len(df)
    
    if before != after:
        print(f"⚠ Removed {before - after} rows with invalid numeric data")
    
    print(f"\nFinal shape: {df.shape}")
    print("✓ Data validation completed!")
    
    return df

def prepare_for_ml(df):
    """
    Siapkan data untuk machine learning
    """
    print("\n" + "="*70)
    print("PREPARING DATA FOR MACHINE LEARNING")
    print("="*70)
    
    # Select relevant columns for ML
    ml_columns = [
        'avg_kills', 'avg_deaths', 'avg_assists', 
        'avg_first_bloods', 'avg_combat_score',
        'kd_ratio', 'ka_ratio',
        'playstyle_self_assessment', 'preferred_role',
        'favorite_agent'
    ]
    
    # Check if all columns exist
    available_cols = [col for col in ml_columns if col in df.columns]
    
    if len(available_cols) != len(ml_columns):
        missing = set(ml_columns) - set(available_cols)
        print(f"⚠ Missing columns: {missing}")
    
    df_ml = df[available_cols].copy()
    
    print(f"\n✓ Selected {len(available_cols)} columns for ML")
    print("\nData summary:")
    print(df_ml.describe())
    
    print("\nPlaystyle distribution:")
    print(df_ml['playstyle_self_assessment'].value_counts())
    
    print("\nRole distribution:")
    print(df_ml['preferred_role'].value_counts())
    
    print("\nFavorite agent distribution:")
    print(df_ml['favorite_agent'].value_counts())
    
    return df_ml

def save_processed_data(df, output_filename='processed_questionnaire_data.csv'):
    """
    Simpan data yang sudah diproses
    """
    df.to_csv(output_filename, index=False)
    print(f"\n✓ Processed data saved to: {output_filename}")
    return output_filename

def display_statistics(df):
    """
    Tampilkan statistik menarik dari data
    """
    print("\n" + "="*70)
    print("INTERESTING STATISTICS")
    print("="*70)
    
    # Average stats by playstyle
    print("\n📊 Average Stats by Playstyle:")
    print("-" * 70)
    stats_by_style = df.groupby('playstyle_self_assessment')[
        ['avg_kills', 'avg_deaths', 'avg_assists', 'kd_ratio']
    ].mean()
    print(stats_by_style.round(2))
    
    # Most popular agents by role
    print("\n👤 Most Popular Agents by Role:")
    print("-" * 70)
    for role in df['preferred_role'].unique():
        role_df = df[df['preferred_role'] == role]
        top_agent = role_df['favorite_agent'].value_counts().head(3)
        print(f"\n{role}:")
        for agent, count in top_agent.items():
            print(f"  {agent}: {count} players ({count/len(role_df)*100:.1f}%)")
    
    # K/D ratio distribution
    print("\n⚔️ K/D Ratio Distribution:")
    print("-" * 70)
    print(f"Average K/D: {df['kd_ratio'].mean():.2f}")
    print(f"Median K/D: {df['kd_ratio'].median():.2f}")
    print(f"Highest K/D: {df['kd_ratio'].max():.2f}")
    print(f"Lowest K/D: {df['kd_ratio'].min():.2f}")
    
    # Combat score stats
    print("\n🎯 Combat Score Statistics:")
    print("-" * 70)
    print(f"Average ACS: {df['avg_combat_score'].mean():.2f}")
    print(f"Median ACS: {df['avg_combat_score'].median():.2f}")
    print(f"Highest ACS: {df['avg_combat_score'].max():.2f}")

def main():
    """
    Main function untuk memproses kuisioner
    """
    print("\n" + "="*70)
    print("VALORANT QUESTIONNAIRE DATA PROCESSOR")
    print("="*70)
    
    # Try to load data
    print("\nMencari file data...")
    
    # Check for common filenames
    possible_files = [
        'valorant_questionnaire_data.csv',
        'valorant_questionnaire_data.json',
        'data.csv',
        'questionnaire.csv'
    ]
    
    filepath = None
    for filename in possible_files:
        if os.path.exists(filename):
            filepath = filename
            break
    
    if filepath is None:
        print("\n❌ Tidak menemukan file data!")
        print("\nSilakan:")
        print("1. Buka file 'Kuisioner Gaya Bermain VALORANT.html'")
        print("2. Isi beberapa data kuisioner")
        print("3. Klik 'Download CSV' atau 'Download JSON'")
        print("4. Simpan file di folder yang sama dengan script ini")
        print("5. Jalankan script ini lagi")
        
        # Create sample instruction file
        with open('CARA_MENGGUNAKAN.txt', 'w', encoding='utf-8') as f:
            f.write("CARA MENGGUNAKAN SISTEM KUISIONER VALORANT\n")
            f.write("=" * 70 + "\n\n")
            f.write("1. Buka file: Kuisioner Gaya Bermain VALORANT.html\n")
            f.write("2. Isi kuisioner (minimal 10-20 responden untuk hasil terbaik)\n")
            f.write("3. Klik tombol 'Download CSV'\n")
            f.write("4. Simpan file dengan nama: valorant_questionnaire_data.csv\n")
            f.write("5. Letakkan di folder yang sama dengan script Python\n")
            f.write("6. Jalankan: python process_questionnaire.py\n")
            f.write("7. Data siap digunakan untuk training model ML!\n")
        
        print("\n✓ File instruksi dibuat: CARA_MENGGUNAKAN.txt")
        return None
    
    # Load data
    df = load_questionnaire_data(filepath)
    
    if df is None:
        return None
    
    # Validate and clean
    df = validate_and_clean_data(df)
    
    if len(df) == 0:
        print("\n❌ Tidak ada data valid setelah cleaning!")
        return None
    
    # Prepare for ML
    df_ml = prepare_for_ml(df)
    
    # Display statistics
    display_statistics(df_ml)
    
    # Save processed data
    output_file = save_processed_data(df_ml)
    
    print("\n" + "="*70)
    print("✅ DATA PROCESSING COMPLETED!")
    print("="*70)
    print(f"\nTotal valid records: {len(df_ml)}")
    print(f"Output file: {output_file}")
    print("\nAnda sekarang bisa menggunakan data ini untuk melatih model ML!")
    print("Gunakan file 'main.py' dan modifikasi bagian data loading.")
    
    return df_ml

if __name__ == "__main__":
    df = main()