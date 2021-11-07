echo "Estraendo i landmark e le Action Units con OpenFace e MediaPipe... (può richiedere qualche ora)"
python3 data_extraction.py

echo "Ripulendo e interpolando i dati... (può richiedere qualche ora)"
python3 data_interpolation.py

echo "Creando le features di Delaunay..."
python3 delaunay.py

echo "##############################################################################################
Per poter eseguire 'my_demo_FSANET_sdd.py' leggi i requisiti scritti sulla pagina GitHub di FSA-Net
##############################################################################################"

cd ../FSA-Net/demo/
echo "Estraendo le features sull'orientazione del volto con FSA-Net... (può richiedere più ore)"

python3 my_demo_FSANET_sdd.py

cd ../../elderReact/
echo "Ultimando il dataset..."
python3 pose_dataset_creation.py
python3 annotations_creations.py

echo "Il dataset è stato creato con successo."
echo "Per eseguire l'addestramento dei modelli di machine learning esegui lo script 'run_models.py'.
L'apprendimento durerà circa 1 giorno."
