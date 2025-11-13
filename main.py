import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set style untuk visualisasi
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def generate_sample_data(n_samples=200):
    """
    Generate sample data yang mensimulasikan hasil kuisioner pemain
    """
    np.random.seed(42)
    
    data = []
    
    # Define agent categories and their characteristics
    agents = {
        'Jett': {'role': 'Duelist', 'style': 'Agresif'},
        'Raze': {'role': 'Duelist', 'style': 'Agresif'},
        'Reyna': {'role': 'Duelist', 'style': 'Agresif'},
        'Phoenix': {'role': 'Duelist', 'style': 'Agresif'},
        'Viper': {'role': 'Controller', 'style': 'Taktis'},
        'Brimstone': {'role': 'Controller', 'style': 'Taktis'},
        'Omen': {'role': 'Controller', 'style': 'Taktis'},
        'Astra': {'role': 'Controller', 'style': 'Taktis'},
        'Sova': {'role': 'Initiator', 'style': 'Supportif'},
        'Breach': {'role': 'Initiator', 'style': 'Supportif'},
        'Skye': {'role': 'Initiator', 'style': 'Supportif'},
        'Fade': {'role': 'Initiator', 'style': 'Supportif'},
        'Sage': {'role': 'Sentinel', 'style': 'Pasif'},
        'Cypher': {'role': 'Sentinel', 'style': 'Pasif'},
        'Killjoy': {'role': 'Sentinel', 'style': 'Pasif'},
        'Chamber': {'role': 'Sentinel', 'style': 'Taktis'}
    }
    
    for _ in range(n_samples):
        # Random agent selection
        agent = np.random.choice(list(agents.keys()))
        agent_info = agents[agent]
        
        # Generate stats based on agent style
        if agent_info['style'] == 'Agresif':
            kills = np.random.normal(18, 4)
            deaths = np.random.normal(14, 3)
            assists = np.random.normal(4, 2)
            first_bloods = np.random.normal(2, 1)
            combat_score = np.random.normal(250, 40)
            
        elif agent_info['style'] == 'Taktis':
            kills = np.random.normal(12, 3)
            deaths = np.random.normal(11, 2)
            assists = np.random.normal(8, 2)
            first_bloods = np.random.normal(1, 0.5)
            combat_score = np.random.normal(200, 30)
            
        elif agent_info['style'] == 'Supportif':
            kills = np.random.normal(10, 2)
            deaths = np.random.normal(10, 2)
            assists = np.random.normal(12, 3)
            first_bloods = np.random.normal(0.5, 0.3)
            combat_score = np.random.normal(180, 25)
            
        else:  # Pasif
            kills = np.random.normal(9, 2)
            deaths = np.random.normal(9, 2)
            assists = np.random.normal(7, 2)
            first_bloods = np.random.normal(0.3, 0.2)
            combat_score = np.random.normal(170, 20)
        
        # Calculate derived metrics
        kd_ratio = kills / max(deaths, 1)
        ka_ratio = (kills + assists) / max(deaths, 1)
        
        data.append({
            'avg_kills': max(0, kills),
            'avg_deaths': max(1, deaths),
            'avg_assists': max(0, assists),
            'avg_first_bloods': max(0, first_bloods),
            'avg_combat_score': max(0, combat_score),
            'kd_ratio': kd_ratio,
            'ka_ratio': ka_ratio,
            'preferred_role': agent_info['role'],
            'playstyle_self_assessment': agent_info['style'],
            'favorite_agent': agent
        })
    
    return pd.DataFrame(data)

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================

def preprocess_data(df):
    """
    Preprocessing data untuk modeling
    """
    print("="*70)
    print("DATA PREPROCESSING")
    print("="*70)
    
    # Display basic info
    print("\nDataset Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData Statistics:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Encode categorical variables
    le_role = LabelEncoder()
    le_style = LabelEncoder()
    
    df['role_encoded'] = le_role.fit_transform(df['preferred_role'])
    df['style_encoded'] = le_style.fit_transform(df['playstyle_self_assessment'])
    
    return df, le_role, le_style

# ============================================================================
# 3. UNSUPERVISED LEARNING - K-MEANS CLUSTERING
# ============================================================================

def perform_clustering(df, n_clusters=4):
    """
    K-Means Clustering untuk mengelompokkan gaya bermain
    """
    print("\n" + "="*70)
    print("K-MEANS CLUSTERING ANALYSIS")
    print("="*70)
    
    # Select features for clustering
    features_for_clustering = [
        'avg_kills', 'avg_deaths', 'avg_assists', 
        'avg_first_bloods', 'kd_ratio', 'ka_ratio', 
        'avg_combat_score'
    ]
    
    X_cluster = df[features_for_clustering].copy()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Elbow method to find optimal clusters
    inertias = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
    plt.title('Elbow Method For Optimal k', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Perform K-Means with optimal clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    print(f"\nClustering completed with {n_clusters} clusters")
    print("\nCluster Distribution:")
    print(df['cluster'].value_counts().sort_index())
    
    # Analyze cluster characteristics
    print("\nCluster Characteristics:")
    cluster_summary = df.groupby('cluster')[features_for_clustering].mean()
    print(cluster_summary)
    
    # Assign playstyle labels to clusters
    cluster_labels = assign_cluster_labels(cluster_summary)
    df['cluster_playstyle'] = df['cluster'].map(cluster_labels)
    
    print("\nCluster Playstyle Labels:")
    for cluster, label in cluster_labels.items():
        print(f"Cluster {cluster}: {label}")
    
    return df, kmeans, scaler, cluster_labels

def assign_cluster_labels(cluster_summary):
    """
    Assign meaningful labels to clusters based on characteristics
    """
    labels = {}
    
    for cluster in cluster_summary.index:
        kills = cluster_summary.loc[cluster, 'avg_kills']
        assists = cluster_summary.loc[cluster, 'avg_assists']
        kd = cluster_summary.loc[cluster, 'kd_ratio']
        
        if kills > cluster_summary['avg_kills'].median() and kd > 1.2:
            labels[cluster] = 'Agresif'
        elif assists > cluster_summary['avg_assists'].median():
            labels[cluster] = 'Supportif'
        elif kd < 1.0:
            labels[cluster] = 'Pasif'
        else:
            labels[cluster] = 'Taktis'
    
    return labels

# ============================================================================
# 4. SUPERVISED LEARNING - RANDOM FOREST CLASSIFIER
# ============================================================================

def train_classifier(df):
    """
    Train Random Forest Classifier untuk prediksi agent
    """
    print("\n" + "="*70)
    print("RANDOM FOREST CLASSIFIER TRAINING")
    print("="*70)
    
    # Select features for classification
    feature_columns = [
        'avg_kills', 'avg_deaths', 'avg_assists', 
        'avg_first_bloods', 'avg_combat_score',
        'kd_ratio', 'ka_ratio', 'cluster'
    ]
    
    X = df[feature_columns]
    y = df['favorite_agent']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train Random Forest
    rf_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("\nTraining Random Forest Classifier...")
    rf_classifier.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5)
    print(f"\nCross-Validation Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Predictions
    y_pred = rf_classifier.predict(X_test)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Feature Importance in Agent Prediction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return rf_classifier, X_test, y_test, y_pred

# ============================================================================
# 5. VISUALIZATION
# ============================================================================

def create_visualizations(df):
    """
    Create comprehensive visualizations
    """
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # 1. Cluster Visualization (2D PCA)
    from sklearn.decomposition import PCA
    
    features_for_viz = [
        'avg_kills', 'avg_deaths', 'avg_assists', 
        'kd_ratio', 'ka_ratio', 'avg_combat_score'
    ]
    
    X_viz = df[features_for_viz]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X_viz))
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=df['cluster'], cmap='viridis', 
                         s=100, alpha=0.6, edgecolors='black')
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.title('Player Playstyle Clustering (PCA Visualization)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cluster_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Playstyle Distribution
    plt.figure(figsize=(10, 6))
    df['cluster_playstyle'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.xlabel('Playstyle', fontsize=12)
    plt.ylabel('Number of Players', fontsize=12)
    plt.title('Distribution of Player Playstyles', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('playstyle_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Agent Distribution by Cluster
    plt.figure(figsize=(14, 8))
    pd.crosstab(df['cluster_playstyle'], df['favorite_agent']).plot(
        kind='bar', stacked=False, colormap='tab20'
    )
    plt.xlabel('Playstyle', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Agent Distribution by Playstyle Cluster', fontsize=14, fontweight='bold')
    plt.legend(title='Agent', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('agent_by_cluster.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. KDA Statistics by Playstyle
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['avg_kills', 'avg_deaths', 'avg_assists', 'kd_ratio']
    titles = ['Average Kills', 'Average Deaths', 'Average Assists', 'K/D Ratio']
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        df.boxplot(column=metric, by='cluster_playstyle', ax=ax)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Playstyle', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        plt.sca(ax)
        plt.xticks(rotation=45)
    
    plt.suptitle('Performance Metrics by Playstyle', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('metrics_by_playstyle.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nAll visualizations saved successfully!")

# ============================================================================
# 6. RECOMMENDATION SYSTEM
# ============================================================================

def recommend_agent(player_stats, kmeans_model, rf_classifier, scaler):
    """
    Recommend agent based on player statistics
    """
    print("\n" + "="*70)
    print("AGENT RECOMMENDATION SYSTEM")
    print("="*70)
    
    # Prepare input
    stats_df = pd.DataFrame([player_stats])
    
    # Calculate derived metrics
    stats_df['kd_ratio'] = stats_df['avg_kills'] / stats_df['avg_deaths']
    stats_df['ka_ratio'] = (stats_df['avg_kills'] + stats_df['avg_assists']) / stats_df['avg_deaths']
    
    # Predict cluster
    cluster_features = [
        'avg_kills', 'avg_deaths', 'avg_assists', 
        'avg_first_bloods', 'kd_ratio', 'ka_ratio', 
        'avg_combat_score'
    ]
    
    X_cluster = stats_df[cluster_features]
    X_scaled = scaler.transform(X_cluster)
    cluster = kmeans_model.predict(X_scaled)[0]
    
    stats_df['cluster'] = cluster
    
    # Predict agent with probabilities
    prediction_features = [
        'avg_kills', 'avg_deaths', 'avg_assists', 
        'avg_first_bloods', 'avg_combat_score',
        'kd_ratio', 'ka_ratio', 'cluster'
    ]
    
    X_pred = stats_df[prediction_features]
    
    # Get prediction probabilities
    probabilities = rf_classifier.predict_proba(X_pred)[0]
    agents = rf_classifier.classes_
    
    # Get top 5 recommendations
    top_indices = np.argsort(probabilities)[-5:][::-1]
    
    print("\nPlayer Statistics:")
    for key, value in player_stats.items():
        print(f"  {key}: {value:.2f}")
    
    print(f"\nDetected Playstyle Cluster: {cluster}")
    
    print("\nTop 5 Agent Recommendations:")
    print("-" * 50)
    for i, idx in enumerate(top_indices, 1):
        agent = agents[idx]
        prob = probabilities[idx]
        print(f"{i}. {agent:15s} - Match Score: {prob*100:.2f}%")
    
    return agents[top_indices[0]], probabilities[top_indices[0]]

# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    print("="*70)
    print("SISTEM REKOMENDASI AGENT VALORANT")
    print("Berdasarkan Gaya Bermain Pemain")
    print("="*70)
    
    # 1. Generate and load data
    print("\n[1/7] Generating sample data...")
    df = generate_sample_data(n_samples=200)
    
    # 2. Preprocess data
    print("\n[2/7] Preprocessing data...")
    df, le_role, le_style = preprocess_data(df)
    
    # 3. Perform clustering
    print("\n[3/7] Performing K-Means clustering...")
    df, kmeans_model, scaler, cluster_labels = perform_clustering(df, n_clusters=4)
    
    # 4. Train classifier
    print("\n[4/7] Training Random Forest classifier...")
    rf_classifier, X_test, y_test, y_pred = train_classifier(df)
    
    # 5. Create visualizations
    print("\n[5/7] Creating visualizations...")
    create_visualizations(df)
    
    # 6. Save models
    print("\n[6/7] Saving models...")
    import joblib
    joblib.dump(kmeans_model, 'kmeans_model.pkl')
    joblib.dump(rf_classifier, 'rf_classifier.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Models saved successfully!")
    
    # 7. Example recommendation
    print("\n[7/7] Testing recommendation system...")
    
    # Example: Aggressive player
    example_player = {
        'avg_kills': 20,
        'avg_deaths': 13,
        'avg_assists': 5,
        'avg_first_bloods': 2.5,
        'avg_combat_score': 270
    }
    
    recommended_agent, match_score = recommend_agent(
        example_player, kmeans_model, rf_classifier, scaler
    )
    
    print("\n" + "="*70)
    print("EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated Files:")
    print("  - elbow_method.png")
    print("  - cluster_visualization.png")
    print("  - playstyle_distribution.png")
    print("  - agent_by_cluster.png")
    print("  - metrics_by_playstyle.png")
    print("  - feature_importance.png")
    print("  - kmeans_model.pkl")
    print("  - rf_classifier.pkl")
    print("  - scaler.pkl")
    
    return df, kmeans_model, rf_classifier, scaler

if __name__ == "__main__":
    df, kmeans_model, rf_classifier, scaler = main()