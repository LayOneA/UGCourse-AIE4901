import numpy as np
from tensorflow import keras
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EPOCHS, BATCH_SIZE, PATIENCE, MIN_DELTA, MODEL_SAVE_DIR

# Model trainer module

class ModelTrainer:
    def __init__(self, model):
        self.model = model
        self.history = None
    
    def train(self, X_train, y_train, X_val, y_val):
        print("Starting model training...")

        callbacks = [
            # Early Stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=PATIENCE,
                min_delta=MIN_DELTA,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model Checkpoint
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(MODEL_SAVE_DIR, 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce Learning Rate
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )

        print("Training completed!")
        return self.history
    
    def evaluate(self, X_test, y_test):
        print("\nModel Evaluation...")

        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred = np.maximum(y_pred, 0)

        mse = np.mean((y_test - y_pred.flatten()) ** 2)
        rmse = np.sqrt(mse)
        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'loss': loss
        }
        print("Evaluation Results:")
        print(f"  MSE:  {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  Loss: {loss:.6f}")
        
        return results
    
    def predict(self, X):
        predictions = self.model.predict(X, verbose=0)
        predictions = np.maximum(predictions, 0)  # 确保预测值不小于0
        return predictions.flatten()
    
    def get_training_history(self):
        return self.history
