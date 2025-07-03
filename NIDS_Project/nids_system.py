
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class NetworkIntrusionDetectionSystem:
    def __init__(self):
        """Initialize the NIDS system"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_trained = False
        
    def create_sample_data(self, num_samples=1000):
        """Create sample network traffic data for demonstration"""
        print("Creating sample network traffic data...")
        
        np.random.seed(42)  
        
        # Create realistic network traffic features
        data = []
        
        for i in range(num_samples):
            # Normal traffic for like 70% of the  data
            if i < num_samples * 0.7:
                record = {
                    'duration': np.random.normal(30, 15),  # Connection duration
                    'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], p=[0.6, 0.3, 0.1]),
                    'service': np.random.choice(['http', 'ftp', 'smtp', 'dns'], p=[0.4, 0.2, 0.2, 0.2]),
                    'bytes_sent': np.random.normal(1000, 500),
                    'bytes_received': np.random.normal(800, 400),
                    'packets_sent': np.random.normal(10, 5),
                    'packets_received': np.random.normal(8, 4),
                    'failed_logins': 0,
                    'root_access': 0,
                    'su_attempted': 0,
                    'attack_type': 'normal'
                }
            # Attack traffic would be like 30% of data
            else:
                attack_types = ['dos', 'probe', 'r2l', 'u2r']
                attack = np.random.choice(attack_types)
                # types of attacks
                if attack == 'dos':  # Denial of Service
                    record = {
                        'duration': np.random.normal(5, 2),
                        'protocol_type': np.random.choice(['tcp', 'udp'], p=[0.8, 0.2]),
                        'service': np.random.choice(['http', 'ftp'], p=[0.7, 0.3]),
                        'bytes_sent': np.random.normal(50, 20),
                        'bytes_received': np.random.normal(10, 5),
                        'packets_sent': np.random.normal(100, 50),
                        'packets_received': np.random.normal(5, 2),
                        'failed_logins': 0,
                        'root_access': 0,
                        'su_attempted': 0,
                        'attack_type': attack
                    }
                elif attack == 'probe':  # Surveillance/Probing
                    record = {
                        'duration': np.random.normal(1, 0.5),
                        'protocol_type': np.random.choice(['tcp', 'icmp'], p=[0.6, 0.4]),
                        'service': np.random.choice(['http', 'ftp', 'smtp'], p=[0.4, 0.3, 0.3]),
                        'bytes_sent': np.random.normal(100, 50),
                        'bytes_received': np.random.normal(50, 25),
                        'packets_sent': np.random.normal(20, 10),
                        'packets_received': np.random.normal(15, 8),
                        'failed_logins': 0,
                        'root_access': 0,
                        'su_attempted': 0,
                        'attack_type': attack
                    }
                elif attack == 'r2l':  # Remote to Local
                    record = {
                        'duration': np.random.normal(60, 30),
                        'protocol_type': 'tcp',
                        'service': np.random.choice(['ftp', 'smtp'], p=[0.6, 0.4]),
                        'bytes_sent': np.random.normal(200, 100),
                        'bytes_received': np.random.normal(150, 75),
                        'packets_sent': np.random.normal(15, 7),
                        'packets_received': np.random.normal(12, 6),
                        'failed_logins': np.random.randint(1, 5),
                        'root_access': 0,
                        'su_attempted': np.random.randint(0, 2),
                        'attack_type': attack
                    }
                else:  # u2r - User to Root
                    record = {
                        'duration': np.random.normal(120, 60),
                        'protocol_type': 'tcp',
                        'service': np.random.choice(['http', 'ftp'], p=[0.5, 0.5]),
                        'bytes_sent': np.random.normal(300, 150),
                        'bytes_received': np.random.normal(250, 125),
                        'packets_sent': np.random.normal(25, 12),
                        'packets_received': np.random.normal(20, 10),
                        'failed_logins': np.random.randint(0, 3),
                        'root_access': 1,
                        'su_attempted': np.random.randint(1, 3),
                        'attack_type': attack
                    }
            
            data.append(record)
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, df):
        """Preprocess the data for machine learning"""
        print("Preprocessing data...")
        
        # Handle categorical variables
        categorical_cols = ['protocol_type', 'service']
        for col in categorical_cols:
            df[col] = LabelEncoder().fit_transform(df[col])
        
        # Separate features and target
        X = df.drop('attack_type', axis=1)
        y = df['attack_type']
        
        # Convert target to binary (normal vs attack)
        y_binary = (y != 'normal').astype(int)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        return X, y_binary, y
    
    def train_model(self, X, y):
        """Train the machine learning model"""
        print("Training the machine learning model...")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        
        y_pred = self.model.predict(X_test_scaled)
        
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        return X_test_scaled, y_test, y_pred
    
    def predict_intrusion(self, network_data):
        """Predict if network traffic is an intrusion"""
        if not self.is_trained:
            print("Error: Model not trained yet!")
            return None
        
        # Preprocess the input 
        network_data_scaled = self.scaler.transform([network_data])
        
        # Make prediction
        prediction = self.model.predict(network_data_scaled)[0]
        probability = self.model.predict_proba(network_data_scaled)[0]
        
        return {
            'is_intrusion': bool(prediction),
            'confidence': max(probability),
            'prediction': 'INTRUSION DETECTED' if prediction else 'NORMAL TRAFFIC'
        }
    
    def analyze_feature_importance(self):
        """Analyze which features are most important for detection"""
        if not self.is_trained:
            print("Error: Model not trained yet!")
            return
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance (Top 5):")
        print(feature_importance.head())
        
        return feature_importance
    
    def display_results(self, y_test, y_pred):
        """Display detailed results"""
        print("\n" + "="*50)
        print("NETWORK INTRUSION DETECTION RESULTS")
        print("="*50)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"True Negatives (Normal correctly identified): {cm[0][0]}")
        print(f"False Positives (Normal wrongly flagged as intrusion): {cm[0][1]}")
        print(f"False Negatives (Intrusion missed): {cm[1][0]}")
        print(f"True Positives (Intrusion correctly detected): {cm[1][1]}")
        
        # Classification Report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Intrusion']))
    
    def test_real_time_detection(self):
        """Test the system with sample network traffic"""
        print("\n" + "="*50)
        print("REAL-TIME INTRUSION DETECTION TEST")
        print("="*50)
        
        # Test cases
        test_cases = [
            {
                'name': 'Normal HTTP Traffic',
                'data': [25, 0, 0, 1200, 900, 12, 10, 0, 0, 0]
            },
            {
                'name': 'Suspicious DOS Attack',
                'data': [2, 0, 0, 30, 5, 150, 3, 0, 0, 0]
            },
            {
                'name': 'Potential Probe Attack',
                'data': [0.5, 0, 0, 80, 40, 25, 18, 0, 0, 0]
            },
            {
                'name': 'Normal FTP Transfer',
                'data': [45, 0, 1, 2000, 1800, 20, 18, 0, 0, 0]
            }
        ]
        
        for test_case in test_cases:
            result = self.predict_intrusion(test_case['data'])
            print(f"\n{test_case['name']}:")
            print(f"  Result: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.4f}")

def main():
    """Main function to run the NIDS system"""
    print("="*60)
    print("NETWORK INTRUSION DETECTION SYSTEM WITH MACHINE LEARNING")
    print("="*60)
    
    # Initialize the system
    nids = NetworkIntrusionDetectionSystem()
    
    # Step 1: Create sample data
    print("\nStep 1: Creating sample network traffic data...")
    df = nids.create_sample_data(1000)
    print(f"Created {len(df)} network traffic records")
    
    # Step 2: Preprocess data
    print("\nStep 2: Preprocessing data...")
    X, y_binary, y_original = nids.preprocess_data(df)
    print(f"Features: {X.shape[1]}")
    print(f"Normal traffic: {sum(y_binary == 0)}")
    print(f"Attack traffic: {sum(y_binary == 1)}")
    
    # Step 3: Train model
    print("\nStep 3: Training the machine learning model...")
    X_test, y_test, y_pred = nids.train_model(X, y_binary)
    
    # Step 4: Display results
    nids.display_results(y_test, y_pred)
    
    # Step 5: Analyze feature importance
    print("\nStep 5: Analyzing feature importance...")
    nids.analyze_feature_importance()
    
    # Step 6: Test real-time detection
    nids.test_real_time_detection()
    
    print("\n" + "="*60)
    print("NIDS SYSTEM READY FOR DEPLOYMENT!")
    print("="*60)

if __name__ == "__main__":
    main()