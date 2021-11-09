echo "Ultimando il dataset... (può richiedere qualche minuto)"
python3 pose_dataset_creation.py
python3 annotations_creations.py

echo "Il dataset è stato creato con successo. Lo puoi trovare nella directory 'my_dataset'."
echo "Per eseguire l'addestramento dei modelli di machine learning esegui lo script 'run_models.py'.
L'apprendimento durerà circa 1 giorno."
echo "Per creare un .csv con i risultati ottenuti dai modelli basta eseguire lo script 'results_dataframe_creator.py'
presente nella directory 'results'"