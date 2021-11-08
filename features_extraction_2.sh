echo "Ultimando il dataset..."
python3 pose_dataset_creation.py
python3 annotations_creations.py

echo "Il dataset è stato creato con successo."
echo "Per eseguire l'addestramento dei modelli di machine learning esegui lo script 'run_models.py'.
L'apprendimento durerà circa 1 giorno."