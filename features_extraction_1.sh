echo "Estraendo i landmark e le Action Units con OpenFace e MediaPipe... (può richiedere qualche ora)"
python3 data_extraction.py

echo "Ripulendo e interpolando i dati... (può richiedere qualche ora)"
python3 data_interpolation.py

echo "Creando le features di Delaunay..."
python3 delaunay.py

echo "
##############################################################################################################################
Per poter eseguire 'my_demo_FSANET_sdd.py' leggi i requisiti scritti sulla pagina GitHub di FSA-Net
##############################################################################################################################
"