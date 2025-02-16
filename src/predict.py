import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class BeamLoadPredictor:
    """
    A class to predict beam loads for various beam types using trained ML models.
    Supports Box Beam, Rectangular Beam, FRP Beam, and T-Beam configurations.
    """
    
    MODEL_DIR = "C:\\Users\\91620\\tendon_profiling_project\\models"
    TARGET_COLUMNS = ['Pcr (Cracking Load)', 'Pu (Ultimate Load)', 'Mu (Ultimate Moment)']
    
    # Updated to match exact column names from training data
    BEAM_FEATURES = {
        "Box_Beam": [
            "Width_b_mm", "Height_h_mm", "Span_L_mm", "Cover_c_mm", 
            "Web_hp_mm", "Cover_ap_mm", "As`/mm2", "As_mm2", "Ap_mm2", 
            "Dpre_mm", "fy_MPa", "fpu_MPa", "Es_MPa", "fc_MPa", 
            "Ec_MPa", "Prestress_Force_kN"
        ],
        "Rectangular_Beam": [
            "Width_b_mm", "Height_h_mm", "Span_L_mm", "Cover_c_mm", 
            "Web_hp_mm", "Cover_ap_mm", "As`/mm2", "As_mm2", "Ap_mm2", 
            "Dpre_mm", "fy_MPa", "fpu_MPa", "Es_MPa", "fc_MPa", 
            "Ec_MPa", "Prestress_Force_kN"
        ],
        "FRP_Beam": [
            "Width_b_mm", "Height_h_mm", "Span_L_mm", "Cover_c_mm", 
            "Web_hp_mm", "Cover_ap_mm", "As`/mm2", "As_mm2", "Ap_mm2", 
            "Dpre_mm", "fy_MPa", "fpu_MPa", "Es_MPa", "Ef_MPa", 
            "fc_MPa", "Prestress_Force_kN"
        ],
        "TBeam": [
            "Web_b_mm", "Height_h_mm", "Flange_bf_mm", "Flange_hf_mm",
            "LowFlange_bl_mm", "LowFlange_hl_mm", "Span_L_mm", "Cover_c_mm",
            "Web_hp_mm", "Cover_ap_mm", "As`/mm2", "As_mm2", "Ap_mm2",
            "Dpre_mm", "fy_MPa", "fpu_MPa", "Es_MPa", "fc_MPa",
            "Ec_MPa", "Prestress_Force_kN"
        ]
    }

    # Updated feature descriptions with proper units
    FEATURE_DESCRIPTIONS = {
        "Width_b_mm": "Beam width (mm)",
        "Height_h_mm": "Beam height (mm)",
        "Span_L_mm": "Span length (mm)",
        "Cover_c_mm": "Concrete cover (mm)",
        "Web_hp_mm": "Web thickness (mm)",
        "Cover_ap_mm": "Prestress cover (mm)",
        "As`/mm2": "Compression reinforcement area (mm²)",
        "As_mm2": "Tension reinforcement area (mm²)",
        "Ap_mm2": "Prestressing steel area (mm²)",
        "Dpre_mm": "Prestressing tendon diameter (mm)",
        "fy_MPa": "Reinforcement yield strength (MPa)",
        "fpu_MPa": "Prestressing steel ultimate strength (MPa)",
        "Es_MPa": "Steel elastic modulus (MPa)",
        "Ef_MPa": "FRP elastic modulus (MPa)",
        "fc_MPa": "Concrete compressive strength (MPa)",
        "Ec_MPa": "Concrete elastic modulus (MPa)",
        "Prestress_Force_kN": "Prestressing force (kN)",
        "Web_b_mm": "Web width (mm)",
        "Flange_bf_mm": "Flange width (mm)",
        "Flange_hf_mm": "Flange thickness (mm)",
        "LowFlange_bl_mm": "Lower flange width (mm)",
        "LowFlange_hl_mm": "Lower flange height (mm)"
    }

    def __init__(self):
        """Initialize the predictor with empty model and scaler."""
        self.model = None
        self.scaler = None
        self.current_beam_type = None
        self.feature_names = None

    def load_model(self, beam_type: str) -> None:
        """
        Load the appropriate model and scaler for the specified beam type.
        
        Args:
            beam_type: The type of beam to load the model for
            
        Raises:
            FileNotFoundError: If model or scaler files are not found
        """
        model_path = os.path.join(self.MODEL_DIR, f"model_{beam_type}.pkl")
        scaler_path = os.path.join(self.MODEL_DIR, f"scaler_{beam_type}.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Model or scaler not found for {beam_type}")
            
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.current_beam_type = beam_type
        self.feature_names = self.BEAM_FEATURES[beam_type]

    def predict(self, features: List[float]) -> Dict[str, float]:
        """
        Make predictions using the loaded model.
        
        Args:
            features: List of input features matching the beam type's requirements
            
        Returns:
            Dictionary containing predicted values for cracking load, ultimate load,
            and ultimate moment
            
        Raises:
            ValueError: If feature count doesn't match requirements
        """
        if len(features) != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {len(features)}")
            
        # Create a DataFrame with correct feature names
        input_df = pd.DataFrame([dict(zip(self.feature_names, features))])
        
        # Scale the features
        input_scaled = self.scaler.transform(input_df)
        predictions = self.model.predict(input_scaled)[0]
        
        return {
            'Cracking Load (Pcr)': predictions[0],
            'Ultimate Load (Pu)': predictions[1],
            'Ultimate Moment (Mu)': predictions[2]
        }

    @staticmethod
    def get_beam_types() -> List[str]:
        """Return list of available beam types."""
        return list(BeamLoadPredictor.BEAM_FEATURES.keys())

    def get_required_features(self) -> List[str]:
        """Return list of required features for current beam type."""
        return self.feature_names

    def get_feature_description(self, feature: str) -> str:
        """Get the description for a feature."""
        return self.FEATURE_DESCRIPTIONS.get(feature, feature)

def main():
    """Main function to run the beam load predictor interactively."""
    predictor = BeamLoadPredictor()
    
    # Display available beam types
    print("\n🏗️ Available Beam Types:")
    for i, beam in enumerate(predictor.get_beam_types(), 1):
        print(f"{i}. {beam.replace('_', ' ')}")
    
    # Get beam type selection
    while True:
        try:
            choice = int(input("\nEnter the number of the beam type (1-4): ")) - 1
            beam_type = predictor.get_beam_types()[choice]
            predictor.load_model(beam_type)
            break
        except (ValueError, IndexError):
            print("❌ Invalid selection. Please enter a number between 1 and 4.")
        except FileNotFoundError as e:
            print(f"❌ {str(e)}")
            return
    
    # Get condition and tension method
    condition = input("\nEnter Condition (B for Bounded, U for Unbounded): ").strip().upper()
    tension_method = input("Enter Tension Method (H for Pre-Tensioned, X for Post-Tensioned): ").strip().upper()
    
    if condition not in ['B', 'U'] or tension_method not in ['H', 'X']:
        print("❌ Invalid condition or tension method.")
        return
    
    # Collect feature inputs
    print(f"\n📝 Enter the following parameters for {beam_type.replace('_', ' ')}:")
    features = []
    for feature in predictor.get_required_features():
        description = predictor.get_feature_description(feature)
        while True:
            try:
                value = float(input(f"{description}: "))
                features.append(value)
                break
            except ValueError:
                print("❌ Please enter a valid number.")
    
    # Make prediction and display results
    try:
        results = predictor.predict(features)
        print("\n🎯 Predicted Values:")
        for key, value in results.items():
            print(f"🔹 {key}: {value:.2f} kN")
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()