echo "Estraendo i landmark e le Actio Units con OpenFace... (può richiedere tanto tempo)"
python3 data_extraction.py

echo "Estraendo i landmark con MediaPipe... (può richiedere tanto tempo)"
python3 data_interpolation.py

echo "Creando le features di Delaunay..."
python3 delaunay.py

echo "##############################################################################################
Per poter eseguire 'my_demo_FSANET_sdd.py' leggi i requirements scritti sul github ufficiale di FSA-Net
##############################################################################################"

cd ../FSA-Net/demo/
echo "Estraendo le features sull'orientazione del volto con FSA-Net... (può richiedere tanto tempo)"

python3 my_demo_FSANET_sdd.py

cd ../../elderReact/
rm my_dataset/Features/train/interpolated_AU_/50_50_68.csv

echo "Ultimando il dataset..."
python3 pose_dataset_creation.py
python3 annotations_creations.py

echo "Il dataset è stato creato con successo."
